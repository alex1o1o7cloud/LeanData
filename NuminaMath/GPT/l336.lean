import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.Algebra.Basic
import Mathlib.Algebra.Arithmetic.Geometric
import Mathlib.Algebra.Exponentiation
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Quadratic
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.Calculus.Opt
import Mathlib.Analysis.SpecialFunctions.Arctan
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Combinatorics.Simplex
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.ModEq
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Determinant
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Combinatorics.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.MeasureTheory.Measure.MeasureSpace
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic

namespace sum_primes_less_than_20_divisible_by_3_l336_336495

theorem sum_primes_less_than_20_divisible_by_3 : 
  (∑ p in {2, 3, 5, 7, 11, 13, 17, 19} | p < 20 ∧ Nat.Prime p ∧ p % 3 = 0, p) = 3 := 
by sorry

end sum_primes_less_than_20_divisible_by_3_l336_336495


namespace middle_digit_zero_l336_336921

theorem middle_digit_zero (a b c M : ℕ) (h1 : M = 36 * a + 6 * b + c) (h2 : M = 64 * a + 8 * b + c) (ha : 0 ≤ a ∧ a < 6) (hb : 0 ≤ b ∧ b < 6) (hc : 0 ≤ c ∧ c < 6) : 
  b = 0 := 
  by sorry

end middle_digit_zero_l336_336921


namespace natural_pair_condition_l336_336591

theorem natural_pair_condition (a b : ℕ) (h : b ^ a ∣ a ^ b - 1) : 
  (a = 3 ∧ b = 2) :=
begin
  sorry,
end

end natural_pair_condition_l336_336591


namespace log_13_equals_log_div_log_l336_336698

theorem log_13_equals_log_div_log (x : ℝ) (h : log 7 (x + 10) = 2) : log 13 x = log 39 / log 13 := by
  sorry

end log_13_equals_log_div_log_l336_336698


namespace circumradius_triangle_ABC_l336_336485

-- Definitions and assumptions
variables (A B C O1 O2 : Point)
variables (r1 r2 r3 : ℝ)
variables (d : ℝ := 13) -- Distance between the centers of the two spheres is 13
variables (sum_radii : ℝ := 7) -- Sum of the radii r1 + r2 is 7
variables (radii_third_sphere : ℝ := 5) -- Radius of the third sphere is 5

-- Conditions
variable (h1 : r1 + r2 = sum_radii)
variable (h2 : (O1 - O2).length = d)
variable (h3 : ∀ x, Sphere O1 r1 x ∧ Sphere O2 r2 x → Plane ABC x)
variable (h4 : Sphere B r3 C ∧ Sphere A r3 C)

-- Theorem to prove
theorem circumradius_triangle_ABC :
  circumradius_triangle A B C = Real.sqrt 30 :=
sorry

end circumradius_triangle_ABC_l336_336485


namespace square_side_length_l336_336324

-- Problem conditions as Lean definitions
def length_rect : ℕ := 400
def width_rect : ℕ := 300
def perimeter_rect := 2 * length_rect + 2 * width_rect
def perimeter_square := 2 * perimeter_rect
def length_square := perimeter_square / 4

-- Proof statement
theorem square_side_length : length_square = 700 := 
by 
  -- (Any necessary tactics to complete the proof would go here)
  sorry

end square_side_length_l336_336324


namespace emails_in_afternoon_l336_336002

theorem emails_in_afternoon (A : ℕ) 
  (morning_emails : A + 3 = 10) : A = 7 :=
by {
    sorry
}

end emails_in_afternoon_l336_336002


namespace triangle_return_to_original_position_l336_336760

structure Point :=
  (x : ℝ)
  (y : ℝ)

def rotate90 (p : Point) : Point := ⟨-p.y, p.x⟩
def rotate180 (p : Point) : Point := ⟨-p.x, -p.y⟩
def rotate270 (p : Point) : Point := ⟨p.y, -p.x⟩
def reflectX (p : Point) : Point := ⟨p.x, -p.y⟩
def reflectY (p : Point) : Point := ⟨-p.x, p.y⟩
def translate (p : Point) (dx dy : ℝ) : Point := ⟨p.x + dx, p.y + dy⟩

def identity (p : Point) : Point := p

def T := [Point.mk 0 0, Point.mk 6 0, Point.mk 0 4]

def valid_transformations :=
  [rotate90, rotate180, rotate270, reflectX, reflectY]

def apply_transformations (fs : List (Point → Point)) (t : List Point) : List Point :=
  t.map (λ p, fs.foldl (λ pt f, f pt) p)

def is_identity (t : List Point) : Prop :=
  t = T

noncomputable def count_valid_sequences : ℕ :=
  (List.replicateM 3 valid_transformations).count (λ fs, is_identity (apply_transformations fs T))

theorem triangle_return_to_original_position : count_valid_sequences = 9 := sorry

end triangle_return_to_original_position_l336_336760


namespace hyperbola_with_common_focus_l336_336305

noncomputable def hyperbola_equation (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ ∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1

noncomputable def asymptote_of_hyperbola (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (∀ x : ℝ, ∃ y : ℝ, y = (b / a) * x)

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 3 = 1

theorem hyperbola_with_common_focus (a b : ℝ) :
  (a > 0 ∧ b > 0) ∧ 
  (hyperbola_equation a b) ∧ 
  (asymptote_of_hyperbola a b) ∧ 
  (ellipse (±3 : ℝ) 0) → 
  (a = 2 ∧ b = real.sqrt 5) :=
by
  sorry

end hyperbola_with_common_focus_l336_336305


namespace sets_equality_solution_l336_336776

variable (x y : ℝ)

theorem sets_equality_solution :
  let A := {0, 1, x}
  let B := {x^2, y, -1}
  (A = B) → (y = 0) :=
by
  intros
  sorry

end sets_equality_solution_l336_336776


namespace stripe_division_l336_336611

def is_stripe (k : ℕ) : Prop := k > 0

theorem stripe_division (n : ℕ) :
  (∀ k, k ≤ 1995 → is_stripe k) →
  (∃ n, (1995 * n) % 2 = 0 ∧ (n ≤ 998 ∨ n ≥ 3989)) :=
by 
  intro k is_stripe
  existsi n
  sorry

end stripe_division_l336_336611


namespace solve_expression_l336_336608

theorem solve_expression (a b c : ℝ) (ha : a^3 - 2020*a^2 + 1010 = 0) (hb : b^3 - 2020*b^2 + 1010 = 0) (hc : c^3 - 2020*c^2 + 1010 = 0) (habc_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
    (1 / (a * b) + 1 / (b * c) + 1 / (a * c) = -2) := 
sorry

end solve_expression_l336_336608


namespace width_of_box_is_correct_l336_336453

noncomputable def length_of_box : ℝ := 62
noncomputable def height_lowered : ℝ := 0.5
noncomputable def volume_removed_in_gallons : ℝ := 5812.5
noncomputable def gallons_to_cubic_feet : ℝ := 1 / 7.48052

theorem width_of_box_is_correct :
  let volume_removed_in_cubic_feet := volume_removed_in_gallons * gallons_to_cubic_feet
  let area_of_base := length_of_box * W
  let needed_volume := area_of_base * height_lowered
  volume_removed_in_cubic_feet = needed_volume →
  W = 25.057 :=
by
  sorry

end width_of_box_is_correct_l336_336453


namespace kekai_remaining_money_l336_336378

-- Definitions based on given conditions
def shirts_sold := 5
def price_per_shirt := 1
def pants_sold := 5
def price_per_pant := 3
def half_fraction := 1 / 2 : ℝ

-- Proving that Kekai's remaining money is $10
theorem kekai_remaining_money : 
  let earnings_from_shirts := shirts_sold * price_per_shirt in
  let earnings_from_pants := pants_sold * price_per_pant in
  let total_earnings := earnings_from_shirts + earnings_from_pants in
  let money_given_to_parents := total_earnings * half_fraction in
  let remaining_money := total_earnings - money_given_to_parents in
  remaining_money = 10 :=
by
  sorry

end kekai_remaining_money_l336_336378


namespace scout_weekend_earnings_l336_336054

-- Define the constants and conditions
def base_pay : ℝ := 10.0
def tip_per_delivery : ℝ := 5.0
def saturday_hours : ℝ := 4.0
def sunday_hours : ℝ := 5.0
def saturday_deliveries : ℝ := 5.0
def sunday_deliveries : ℝ := 8.0

-- Calculate total hours worked
def total_hours : ℝ := saturday_hours + sunday_hours

-- Calculate base pay for the weekend
def total_base_pay : ℝ := total_hours * base_pay

-- Calculate total number of deliveries
def total_deliveries : ℝ := saturday_deliveries + sunday_deliveries

-- Calculate total earnings from tips
def total_tips : ℝ := total_deliveries * tip_per_delivery

-- Calculate total earnings
def total_earnings : ℝ := total_base_pay + total_tips

-- Theorem to prove the total earnings is $155.00
theorem scout_weekend_earnings : total_earnings = 155.0 := by
  sorry

end scout_weekend_earnings_l336_336054


namespace tetrahedron_projection_ratio_l336_336434

theorem tetrahedron_projection_ratio (tetrahedron : Tetrahedron) :
  ∃ (plane1 plane2 : Plane), area_projection_ratio tetrahedron plane1 plane2 ≥ real.sqrt 2 :=
begin
  sorry,
end

end tetrahedron_projection_ratio_l336_336434


namespace projection_matrix_l336_336012

-- Define the vector type
def vec3 := ℝ × ℝ × ℝ

-- Normal vector of plane Q
def n : vec3 := (2, -1, 2)

-- Definition of the projection matrix Q
def Q : vec3 → vec3 := λ w,
  let (x, y, z) := w in
  (
    (5 / 9 * x + 2 / 9 * y - 4 / 9 * z),
    (2 / 9 * x + 10 / 9 * y - 2 / 9 * z),
    (-4 / 9 * x + 2 / 9 * y + 5 / 9 * z)
  )

-- The theorem stating that Q projects w onto the plane Q
theorem projection_matrix (w : vec3) : 
  Q w = (
    let (x, y, z) := w in
    (
      (5 / 9 * x + 2 / 9 * y - 4 / 9 * z),
      (2 / 9 * x + 10 / 9 * y - 2 / 9 * z),
      (-4 / 9 * x + 2 / 9 * y + 5 / 9 * z)
    )) := 
  sorry

end projection_matrix_l336_336012


namespace stars_in_4x4_grid_l336_336263

theorem stars_in_4x4_grid 
  (grid : Fin 4 → Fin 4 → Prop)
  (stars_7 : ∃ cells : Fin 7, grid cells)
  (stars_6 : ∀ (cells : Fin 6), grid cells → ∃ rows cols, (rows : Finset (Fin 4), cols : Finset (Fin 4)) ∧ rows.card = 2 ∧ cols.card = 2 ∧ ∀ r ∈ rows ∀ c ∈ cols, ¬ grid (r, c)) :
  (∃ (cells : Fin 7), 
    (∃ rows columns, (rows : Finset (Fin 4), columns : Finset (Fin 4)) ∧ rows.card = 2 ∧ columns.card = 2 ∧ ∃ cell, cell ∉ rows × columns → grid cell)) ∧ 
  (∀ (cells : Fin 6), 
    (∃ rows columns, (rows : Finset (Fin 4), columns : Finset (Fin 4)) ∧ rows.card = 2 ∧ columns.card = 2 ∧ ∀ cell, cell ∉ rows × columns → ¬ grid cell)) :=
sorry 

end stars_in_4x4_grid_l336_336263


namespace Ceva_mass_point_l336_336429

variables (A B C O A1 B1 C1 : Type) [hasRatio : ∀ (a b c : Type), a → b → c → Prop]

-- Given conditions
axiom lines_intersect_at_O : (hasRatio A A1 O) ∧ (hasRatio C C1 O)
axiom AC1_C1B_ratio : ∀ {p : ℝ}, hasRatio A C1 B → p
axiom BA1_A1C_ratio : ∀ {q : ℝ}, hasRatio B A1 C → q

-- Prove that BB1 passes through O if and only if CB1/B1A = 1/(pq)
theorem Ceva_mass_point
  (p q : ℝ)
  (h1 : hasRatio A C1 B)
  (h2 : hasRatio B A1 C)
  (h3 : AC1_C1B_ratio h1 = p)
  (h4 : BA1_A1C_ratio h2 = q) :
  (hasRatio C B1 A → hasRatio B B1 O) ↔ (1 / p / q) := sorry

end Ceva_mass_point_l336_336429


namespace min_positive_period_sine_l336_336454

theorem min_positive_period_sine (A : ℝ) (ω : ℝ) (ϕ : ℝ) :
  (∀ x : ℝ, sin (ω * x + ϕ) = sin (ω * (x + (2 * π / ω)) + ϕ)) →
  ω = 2 →
  ∀ x : ℝ, sin (2 * x - π / 6) = sin (2 * (x + π) - π / 6) :=
by {
  intros Hperiod Hω x,
  rw Hω at *,
  exact Hperiod x,
}

end min_positive_period_sine_l336_336454


namespace sum_quotients_mod_2027_l336_336987

/-- Problem: Given the prime number 2027, we compute the sum of 
  i^2 divided by 9 + i^4 modulo 2027, 
  where the division is defined as multiplication by the multiplicative inverse in modulo 2027.

  To prove: the sum from 0 to 2026 of these values is equivalent to 1689 modulo 2027.
-/
theorem sum_quotients_mod_2027 :
  (∑ i in Finset.range 2027, (i^2 : ℤ) * ((9 + i^4 : ℤ)^(-1 : ℤ)) % 2027) % 2027 = 1689 :=
sorry

end sum_quotients_mod_2027_l336_336987


namespace find_ellipse_parameters_l336_336394

noncomputable def ellipse_centers_and_axes (F1 F2 : ℝ × ℝ) (d : ℝ) (tangent_slope : ℝ) :=
  let h := (F1.1 + F2.1) / 2
  let k := (F1.2 + F2.2) / 2
  let a := d / 2
  let c := (Real.sqrt ((F2.1 - F1.1)^2 + (F2.2 - F1.2)^2)) / 2
  let b := Real.sqrt (a^2 - c^2)
  (h, k, a, b)

theorem find_ellipse_parameters :
  let F1 := (-1, 1)
  let F2 := (5, 1)
  let d := 10
  let tangent_at_x_axis_slope := 1
  let (h, k, a, b) := ellipse_centers_and_axes F1 F2 d tangent_at_x_axis_slope
  h + k + a + b = 12 :=
by
  sorry

end find_ellipse_parameters_l336_336394


namespace lucas_can_afford_book_l336_336029

-- Definitions from the conditions
def book_cost : ℝ := 28.50
def two_ten_dollar_bills : ℝ := 2 * 10
def five_one_dollar_bills : ℝ := 5 * 1
def six_quarters : ℝ := 6 * 0.25
def nickel_value : ℝ := 0.05

-- Given the conditions, we need to prove that if Lucas has at least 40 nickels, he can afford the book.
theorem lucas_can_afford_book (m : ℝ) (h : m >= 40) : 
  (two_ten_dollar_bills + five_one_dollar_bills + six_quarters + m * nickel_value) >= book_cost :=
by {
  sorry
}

end lucas_can_afford_book_l336_336029


namespace find_range_of_a_l336_336816

def setA (x : ℝ) : Prop := 1 < x ∧ x < 2
def setB (x : ℝ) : Prop := 3 / 2 < x ∧ x < 4
def setUnion (x : ℝ) : Prop := 1 < x ∧ x < 4
def setP (a x : ℝ) : Prop := a < x ∧ x < a + 2

theorem find_range_of_a (a : ℝ) :
  (∀ x, setP a x → setUnion x) → 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end find_range_of_a_l336_336816


namespace rearrangements_vowels_end_l336_336688

theorem rearrangements_vowels_end : 
  let word := "REACTOR"
  let vowels := ['E', 'A', 'O']
  let consonants := ['R', 'C', 'T', 'R']
  ∃ n : ℕ, n = 72 ∧ 
  let consonant_permutations := (4! / (2!)) in
  let vowel_permutations := 3! in
  n = consonant_permutations * vowel_permutations :=
by
  sorry

end rearrangements_vowels_end_l336_336688


namespace angle_PA1A2_l336_336284

theorem angle_PA1A2 (a : ℝ) (h_a : a > 0) (P : ℝ × ℝ)
  (hP : P.1^2 - P.2^2 = a^2)
  (A1 A2 : ℝ × ℝ)
  (hA1 : A1 = (-a, 0))
  (hA2 : A2 = (a, 0))
  (alpha : ℝ)
  (h_alpha : ∠ A2 P A1 = 2 * ∠ P A1 A2) :
  ∠ P A1 A2 = π / 8 := sorry

end angle_PA1A2_l336_336284


namespace player_avg_increase_l336_336180

theorem player_avg_increase
  (matches_played : ℕ)
  (initial_avg : ℕ)
  (next_match_runs : ℕ)
  (total_runs : ℕ)
  (new_total_runs : ℕ)
  (new_avg : ℕ)
  (desired_avg_increase : ℕ) :
  matches_played = 10 ∧ initial_avg = 32 ∧ next_match_runs = 76 ∧ total_runs = 320 ∧ 
  new_total_runs = 396 ∧ new_avg = 32 + desired_avg_increase ∧ 
  11 * new_avg = new_total_runs → desired_avg_increase = 4 := 
by
  sorry

end player_avg_increase_l336_336180


namespace triangle_DEF_angles_l336_336167

-- Define the right triangle ABC with the right angle at C
variables (A B C D E F : Type*)
          [TriangleABC : RightTriangle A B C]
          (inscribedCircle : InscribedCircleInRightTriangle A B C D E F)

-- Define the angles formed by the angle bisectors AD, BE, and CF
def nature_of_angles_in_triangle_DEF : Prop :=
  (Triangle.DEF A B C D E F).angle_DFE.is_acute ∧
  (Triangle.DEF A B C D E F).angle_FED.is_acute ∧
  (Triangle.DEF A B C D E F).angle_EFD.is_right

-- State the theorem
theorem triangle_DEF_angles :
  nature_of_angles_in_triangle_DEF A B C D E F :=
sorry

end triangle_DEF_angles_l336_336167


namespace proof_bound_l336_336416

def P (x : ℝ) := (a : fin 22 → ℝ) → ∑ i in (finset.range 22), a i * x^i -- Defining the polynomial P

def valid_coeffs (a : fin 22 → ℝ) : Prop :=
  ∀ i, 0 ≤ i < 22 → 1011 ≤ a i ∧ a i ≤ 2021 -- Condition for coefficients

def has_integer_root (P : ℝ → ℝ) : Prop :=
  ∃ z : ℤ, P (z : ℝ) = 0 -- Condition for integer root

def bounded_difference (a : fin 22 → ℝ) (c : ℝ) : Prop :=
  ∀ k, 0 ≤ k ∧ k ≤ 19 → |a (k+2 : ℕ) - a k| ≤ c -- Condition for bounded differences

theorem proof_bound (a : fin 22 → ℝ) (c : ℝ) (P : ℝ → ℝ) :
  valid_coeffs a →
  has_integer_root P →
  bounded_difference a c →
  ∑ k in (finset.range 11), (a (2*k+1 : ℕ) - a (2*k : ℕ))^2 ≤ 44 * c^2 :=
by
  intros _ _ _
  sorry

end proof_bound_l336_336416


namespace pages_for_50_dollars_l336_336741

theorem pages_for_50_dollars :
  ∀ (cents_per_page : ℕ) (pages_per_cents : ℕ) (budget_cents : ℕ),
    cents_per_page = 8 →
    pages_per_cents = 6 →
    budget_cents = 5000 →
    (budget_cents * pages_per_cents) / cents_per_page = 3750 :=
  by
  intros cents_per_page pages_per_cents budget_cents h1 h2 h3
  rw [h1, h2, h3]
  sorry

end pages_for_50_dollars_l336_336741


namespace complex_conjugate_solution_l336_336403

noncomputable def complex_conjugate_problem (a b : ℝ) (i : ℂ) : Prop :=
  ∀ a b : ℝ, (conj (b - a * i) = 3 + 4 * i) → a - b = -7

-- stating the problem
theorem complex_conjugate_solution : complex_conjugate_problem :=
by
  intros a b h
  sorry

end complex_conjugate_solution_l336_336403


namespace distance_to_bookstore_l336_336350

variable (d s : ℝ)

def usual_time_in_hours := 1 / 2
def windy_time_in_hours := 3 / 10
def extra_speed := 12

theorem distance_to_bookstore :
  (d = s * usual_time_in_hours) ∧ (d = (s + extra_speed) * windy_time_in_hours) → d = 9 :=
by
  intro h
  have h1 : d = s * (1 / 2) := h.1
  have h2 : d = (s + 12) * (3 / 10) := h.2
  sorry

end distance_to_bookstore_l336_336350


namespace probability_of_pink_l336_336745

-- Defining probabilities
def P_BlueThenBlue : ℚ := 16 / 36

-- We will find the probability of drawing a pink gumball
def P_Pink : ℚ := 1 - (rat.sqrt $ 16 / 36)

theorem probability_of_pink :
  P_Pink = 1 / 3 :=
  sorry

end probability_of_pink_l336_336745


namespace segment_association_l336_336399

theorem segment_association (x y : ℝ) 
  (h1 : ∃ (D : ℝ), ∀ (P : ℝ), abs (P - D) ≤ 5) 
  (h2 : ∃ (D' : ℝ), ∀ (P' : ℝ), abs (P' - D') ≤ 9)
  (h3 : 3 * x - 2 * y = 6) : 
  x + y = 12 := 
by sorry

end segment_association_l336_336399


namespace fraction_product_correct_l336_336953

theorem fraction_product_correct : (3 / 5) * (4 / 7) * (5 / 9) = 4 / 21 :=
by
  sorry

end fraction_product_correct_l336_336953


namespace boy_usual_time_l336_336119

noncomputable def usual_rate (R : ℝ) := R
noncomputable def usual_time (T : ℝ) := T
noncomputable def faster_rate (R : ℝ) := (7 / 6) * R
noncomputable def faster_time (T : ℝ) := T - 5

theorem boy_usual_time
  (R : ℝ) (T : ℝ) 
  (h1 : usual_rate R * usual_time T = faster_rate R * faster_time T) :
  T = 35 :=
by 
  unfold usual_rate usual_time faster_rate faster_time at h1
  sorry

end boy_usual_time_l336_336119


namespace sum_divisors_420_l336_336133

theorem sum_divisors_420 : 
  ∃ d, d = 1344 ∧ 
  (∀ n, n ∣ 420 → n ∈ ℕ ∧ n > 0) ∧
  (420 = 2^2 * 3 * 5 * 7) →
  ∑ (n : ℕ) in (finset.filter (λ d, d ∣ 420) (finset.range (421))), n = 1344 :=
by
  sorry

end sum_divisors_420_l336_336133


namespace pirate_overtakes_at_8pm_l336_336540

noncomputable def pirate_overtake_trade : Prop :=
  let initial_distance := 15
  let pirate_speed_before_damage := 14
  let trade_speed := 10
  let time_before_damage := 3
  let pirate_distance_before_damage := pirate_speed_before_damage * time_before_damage
  let trade_distance_before_damage := trade_speed * time_before_damage
  let remaining_distance := initial_distance + trade_distance_before_damage - pirate_distance_before_damage
  let pirate_speed_after_damage := (18 / 17) * 10
  let relative_speed_after_damage := pirate_speed_after_damage - trade_speed
  let time_to_overtake_after_damage := remaining_distance / relative_speed_after_damage
  let total_time := time_before_damage + time_to_overtake_after_damage
  total_time = 8

theorem pirate_overtakes_at_8pm : pirate_overtake_trade :=
by
  sorry

end pirate_overtakes_at_8pm_l336_336540


namespace percentage_decrease_is_17_point_14_l336_336093

-- Define the conditions given in the problem
variable (S : ℝ) -- original salary
variable (D : ℝ) -- percentage decrease

-- Given conditions
def given_conditions : Prop :=
  1.40 * S - (D / 100) * 1.40 * S = 1.16 * S

-- The required proof problem, where we assert D = 17.14
theorem percentage_decrease_is_17_point_14 (S : ℝ) (h : given_conditions S D) : D = 17.14 := 
  sorry

end percentage_decrease_is_17_point_14_l336_336093


namespace fish_cost_l336_336702

theorem fish_cost (F P : ℝ) (h1 : 4 * F + 2 * P = 530) (h2 : 7 * F + 3 * P = 875) : F = 80 := 
by
  sorry

end fish_cost_l336_336702


namespace extreme_values_l336_336293

noncomputable def f (a b x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 3

theorem extreme_values (a b : ℝ) : 
  (f a b (-1) = 10) ∧ (f a b 2 = -17) →
  (6 * (-1)^2 + 2 * a * (-1) + b = 0) ∧ (6 * 2^2 + 2 * (a * 2) + b = 0) →
  a = -3 ∧ b = -12 :=
by 
  sorry

end extreme_values_l336_336293


namespace oranges_in_shop_l336_336778

-- Define the problem conditions
def ratio (M O A : ℕ) : Prop := (10 * O = 2 * M) ∧ (10 * A = 3 * M)

noncomputable def numMangoes : ℕ := 120
noncomputable def numApples : ℕ := 36

-- Statement of the problem
theorem oranges_in_shop (ratio_factor : ℕ) (h_ratio : ratio numMangoes (2 * ratio_factor) numApples) :
  (2 * ratio_factor) = 24 := by
  sorry

end oranges_in_shop_l336_336778


namespace paint_house_18_women_4_days_l336_336743

theorem paint_house_18_women_4_days :
  (∀ (m1 m2 : ℕ) (d1 d2 : ℕ), m1 * d1 = m2 * d2) →
  (12 * 6 = 72) →
  (72 = 18 * d) →
  d = 4.0 :=
by
  sorry

end paint_house_18_women_4_days_l336_336743


namespace smallest_product_of_two_distinct_primes_greater_than_50_l336_336225

theorem smallest_product_of_two_distinct_primes_greater_than_50 : 
  ∃ (p q : ℕ), p > 50 ∧ q > 50 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 3127 :=
by 
  sorry

end smallest_product_of_two_distinct_primes_greater_than_50_l336_336225


namespace max_M_l336_336610

def J_j (j : ℕ) (h : j > 0) : ℕ := 10 ^ (j + 2) + 32

def M (j : ℕ) (h : j > 0) : ℕ :=
let J := J_j j h in
J.factorization 2

theorem max_M : ∃ j : ℕ, j > 0 ∧ M j (by simp [Nat.one_pos]) = 6 :=
sorry

end max_M_l336_336610


namespace rectangle_length_l336_336229

theorem rectangle_length (area_abcd : ℝ) (area_abcd = 5760)
    (n_rectangles : ℕ) (n_rectangles = 8)
    (total_area : ℝ) (total_area = (n_rectangles : ℝ) * (x * x))
    (x : ℝ) :
  x = 27 :=
begin
  sorry
end

end rectangle_length_l336_336229


namespace find_real_num_l336_336339

noncomputable def com_num (a : ℝ) : ℂ := (a + 3 * Complex.I) / (1 + 2 * Complex.I)

theorem find_real_num (a : ℝ) : (∃ b : ℝ, com_num a = b * Complex.I) → a = -6 :=
by
  sorry

end find_real_num_l336_336339


namespace hiker_hears_blast_at_275_yards_l336_336536

def hiker_distance_when_heard_blast
  (t_flare : ℕ) -- Time set for the flare, in seconds
  (speed_hiker : ℕ) -- Speed of the hiker, in yards per second
  (speed_sound : ℕ) -- Speed of sound, in feet per second)
  : ℕ :=
  let p : ℕ → ℕ := fun t => 18 * t in -- Distance function for hiker in feet, converted from yards
  let q : ℕ → ℕ := fun t => 1200 * (t - t_flare) in -- Distance function for sound in feet
  let t_heard := (54000 + (18 * t_flare)) / (1200 + 18) in -- Time when distances are equal
  let distance_feet := p t_heard in -- Distance hiker has run in feet
  distance_feet / 3  -- Convert distance to yards

theorem hiker_hears_blast_at_275_yards :
  hiker_distance_when_heard_blast 45 6 1200 = 275 :=
  by
    unfold hiker_distance_when_heard_blast
    sorry

end hiker_hears_blast_at_275_yards_l336_336536


namespace increasing_function_l336_336195

-- Definitions of the functions
def f (x : ℝ) : ℝ := (1 / 2) * x
def g (x : ℝ) : ℝ := -3 * x
def h (x : ℝ) : ℝ := -x ^ 2
def k (x : ℝ) : ℝ := -1 / x

theorem increasing_function (x : ℝ) (y : ℝ) :
  (0 < x → 0 < (f x - f y)) ∧ ¬ (0 < x → 0 < (g x - g y)) ∧ 
  ¬ (0 < x → 0 < (h x - h y)) ∧ ¬ (0 < x → 0 < (k x - k y)) :=
by
  sorry

end increasing_function_l336_336195


namespace total_population_l336_336793

theorem total_population (n : ℕ) (avg_population : ℕ) (h1 : n = 20) (h2 : avg_population = 4750) :
  n * avg_population = 95000 := by
  subst_vars
  sorry

end total_population_l336_336793


namespace product_even_l336_336893

theorem product_even (x y : Fin 25 → ℤ) (perm : ∃ σ : Equiv.Perm (Fin 25), ∀ i, y i = x (σ i)) : Even (∏ i, (x i - y i)) :=
by
  sorry

end product_even_l336_336893


namespace bug_probability_at_vertex_A_after_8_meters_l336_336392

-- Definition of probability P(n)
def P : ℕ → ℚ
| 0     := 1
| 2     := 0
| (n+2) := (1 - P n) / 3

-- Given conditions in problem statement
def tetrahedron_edges : ℕ := 2
def vertices := {A, B, C, D}

-- Value of n
theorem bug_probability_at_vertex_A_after_8_meters :
  (P 8) = 7 / 27 ∧ (∃ n : ℕ, 6561 * P 8 = n ∧ n = 567) :=
by {
  -- Setting up proof steps
  have P2 : P 2 = 0 := by simp [P],
  have P4 : P 4 = 1 / 3 := by simp [P, P2]; norm_num,
  have P6 : P 6 = 2 / 9 := by simp [P, P4]; norm_num,
  have P8 : P 8 = 7 / 27 := by simp [P, P6]; norm_num,
  split,
  exact P8,
  use 567,
  norm_num,
  rw [P8],
  norm_num,
  sorry
}

end bug_probability_at_vertex_A_after_8_meters_l336_336392


namespace hyperbola_equation_l336_336307

theorem hyperbola_equation 
  (h1 : ∀ y, y^2 = 8 * (2 - x) ↔ y^2 = 8 * x) 
  (h2 : ∀ a b, ∃ f : ℝ × ℝ, f = (2:ℝ, 0:ℝ) ∧ (4:ℝ) * real.sqrt 5 / 5 = 2 * a / real.sqrt (a^2 + b^2))
  (h3 : ∃ c : ℝ, ∃ F1, F1 = (0, c) ∧ (real.sqrt 5)^2 + 4 = 9 ∧ 3 = real.dist F1 (2:ℝ, 0:ℝ) + 2) :
   ∃ a b, a = 2 ∧ b = 1 ∧ (∀ x y, y^2 / 4 - x = 1 ↔ y^2 / (a: ℝ)^2 - x^2 / (b: ℝ)^2 = 1) := 
sorry

end hyperbola_equation_l336_336307


namespace polygon_sides_count_l336_336879

-- Define the condition as the probability relation
def probability_condition (n : ℕ) : Prop :=
  (2 * n) / (n * (n - 3)) = 0.25

-- Define the main problem
theorem polygon_sides_count (n : ℕ) : probability_condition n → n = 11 := by
  sorry

end polygon_sides_count_l336_336879


namespace inequality_solution_set_l336_336467

theorem inequality_solution_set (x : ℝ) : (x^2 ≥ 4) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by sorry

end inequality_solution_set_l336_336467


namespace k_value_l336_336711

theorem k_value (k : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → |k * x - 4| ≤ 2) → k = 2 := 
by
  intros h
  sorry

end k_value_l336_336711


namespace surface_area_correctness_l336_336927

def surface_area_of_rotated_sector (R θ : ℝ) : ℝ :=
  2 * π * R^2 * sin (θ / 2) * (cos (θ / 2) + 2 * sin (θ / 2))

theorem surface_area_correctness (R θ : ℝ) :
  surface_area_of_rotated_sector R θ = 2 * π * R^2 * sin (θ / 2) * (cos (θ / 2) + 2 * sin (θ / 2)) :=
begin
  sorry
end

end surface_area_correctness_l336_336927


namespace train_speed_in_km_per_hr_l336_336939

variables (L : ℕ) (t : ℕ) (train_speed : ℕ)

-- Conditions
def length_of_train : ℕ := 1050
def length_of_platform : ℕ := 1050
def crossing_time : ℕ := 1

-- Given calculation of speed in meters per minute
def speed_in_m_per_min : ℕ := (length_of_train + length_of_platform) / crossing_time

-- Conversion units
def meters_to_kilometers (m : ℕ) : ℕ := m / 1000
def minutes_to_hours (min : ℕ) : ℕ := min / 60

-- Speed in km/hr
def speed_in_km_per_hr : ℕ := speed_in_m_per_min * (meters_to_kilometers 1000) * (minutes_to_hours 60)

theorem train_speed_in_km_per_hr : speed_in_km_per_hr = 35 :=
by {
  -- We will include the proof steps here, but for now, we just assert with sorry.
  sorry
}

end train_speed_in_km_per_hr_l336_336939


namespace find_x_l336_336701

theorem find_x (x : ℝ) (h : (3 * x - 7) / 4 = 14) : x = 21 :=
sorry

end find_x_l336_336701


namespace sum_inequality_l336_336769

theorem sum_inequality (n : ℕ) (a : ℕ → ℝ) (h_range : ∀ i, 1 ≤ i ∧ i ≤ n → a i ∈ Set.Icc (-1 : ℝ) 1)
  (h_not_neg_one : ∀ i, 1 ≤ i ∧ i ≤ n → a i * a ((i % n) + 1) ≠ -1)
  (h_cyclic: a (n+1) = a 1) :
  (∑ i in Finset.range n, 1 / (1 + a i * a (i + 1))) ≥ (∑ i in Finset.range n, 1 / (1 + (a i) ^ 2)) :=
sorry

end sum_inequality_l336_336769


namespace sum_of_inverses_inequality_l336_336646

theorem sum_of_inverses_inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum_eq : a + b + c = 3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a^2 + b^2 + c^2 :=
sorry

end sum_of_inverses_inequality_l336_336646


namespace trigonometric_identity_proof_l336_336209

theorem trigonometric_identity_proof :
  (real.sin (110 * real.pi / 180) * real.sin (20 * real.pi / 180)) / 
  (real.cos ((2 * 155) * real.pi / 180) - real.sin ((2 * 155) * real.pi / 180)) = 1 / 2 :=
by sorry

end trigonometric_identity_proof_l336_336209


namespace distinct_solutions_eq_l336_336151

theorem distinct_solutions_eq (θ : ℝ) : 
  (0 < θ ∧ θ < 2 * Real.pi) ∧ (1 - 3 * Real.sin θ + 5 * Real.cos (3 * θ) = 0) → False := by
-- conditions
let f := λ θ, 5 * Real.cos (3 * θ)
let g := λ θ, 3 * Real.sin θ - 1

-- analysis of \( f(\theta) \) and \( g(\theta) \)
-- f(θ) is periodic with period 2π/3
-- g(θ) is periodic with period 2π

sorry

end distinct_solutions_eq_l336_336151


namespace scout_weekend_earnings_l336_336055

theorem scout_weekend_earnings
  (base_pay_per_hour : ℕ)
  (tip_per_delivery : ℕ)
  (hours_worked_saturday : ℕ)
  (deliveries_saturday : ℕ)
  (hours_worked_sunday : ℕ)
  (deliveries_sunday : ℕ)
  (total_earnings : ℕ)
  (h_base_pay : base_pay_per_hour = 10)
  (h_tip : tip_per_delivery = 5)
  (h_hours_sat : hours_worked_saturday = 4)
  (h_deliveries_sat : deliveries_saturday = 5)
  (h_hours_sun : hours_worked_sunday = 5)
  (h_deliveries_sun : deliveries_sunday = 8) :
  total_earnings = 155 :=
by
  sorry

end scout_weekend_earnings_l336_336055


namespace integral_result_l336_336985

noncomputable def integral_ln_cubed_z_over_z :=
  ∫ (z : ℂ) in 1..Complex.I, (Complex.log z)^3 / z

theorem integral_result :
  integral_ln_cubed_z_over_z = (π^4 / 64 : ℂ) := by
  sorry

end integral_result_l336_336985


namespace solution_congruence_l336_336520

theorem solution_congruence (p : ℕ) (hp : Nat.Prime p) (a x : ℕ) (h : a * x ≡ 1 [MOD p]) : 
  (x = a → a ≡ 1 [MOD p] ∨ a ≡ -1 [MOD p]) :=
by sorry

end solution_congruence_l336_336520


namespace count_four_digit_numbers_l336_336320

theorem count_four_digit_numbers : 
  ∀ (n : ℕ), 1000 ≤ n → n ≤ 5000 → 4001.exists (λ n, 1000 ≤ n ∧ n ≤ 5000) :=
  by sorry

end count_four_digit_numbers_l336_336320


namespace probability_smallest_divides_product_l336_336104

open Finset
open Rat

def set := {1, 2, 3, 4, 5, 6}

def comb (s : Finset ℕ) : Finset (Finset ℕ) := s.powerset.filter (λ x, x.card = 3)

theorem probability_smallest_divides_product :
  let outcomes := comb set in
  let successful := outcomes.filter (λ x, let a := x.min' (by dec_trivial),
                                            b := (x.erase a).min' (by dec_trivial),
                                            c := (x.erase a).erase b in
                                        a ∣ b * c) in
  (successful.card : ℚ) / (outcomes.card : ℚ) = 1 / 2 :=
by
  sorry

end probability_smallest_divides_product_l336_336104


namespace probability_both_selected_l336_336491

theorem probability_both_selected (P_C : ℚ) (P_B : ℚ) (hC : P_C = 4/5) (hB : P_B = 3/5) : 
  ((4/5) * (3/5)) = (12/25) := by
  sorry

end probability_both_selected_l336_336491


namespace train_speed_kmph_l336_336938

theorem train_speed_kmph :
  (let train_length := 250
       bridge_length := 350
       crossing_time := 30
       distance := train_length + bridge_length
       speed_mps := distance / crossing_time
       speed_kmph := speed_mps * (3600 / 1000) in
   speed_kmph = 72) :=
by
  let train_length := 250
  let bridge_length := 350
  let crossing_time := 30
  let distance := train_length + bridge_length
  let speed_mps := distance / crossing_time
  let speed_kmph := speed_mps * (3600 / 1000)
  have h : speed_kmph = 72 := sorry
  exact h

end train_speed_kmph_l336_336938


namespace parallelogram_side_length_l336_336922

-- We need trigonometric functions and operations with real numbers.
open Real

theorem parallelogram_side_length (s : ℝ) 
  (h_side_lengths : s > 0 ∧ 3 * s > 0) 
  (h_angle : sin (30 / 180 * π) = 1 / 2) 
  (h_area : 3 * s * (s * sin (30 / 180 * π)) = 9 * sqrt 3) :
  s = 3 * sqrt 2 :=
by
  sorry

end parallelogram_side_length_l336_336922


namespace smallest_n_l336_336126

theorem smallest_n (n : ℕ) (h1 : ∃ k1, 5 * n = k1^2) (h2 : ∃ k2, 7 * n = k2^3) : n = 245 :=
sorry

end smallest_n_l336_336126


namespace largest_element_in_A_inter_B_l336_336311

def A : Set ℕ := { n | 1 ≤ n ∧ n ≤ 2023 }
def B : Set ℕ := { n | ∃ k : ℤ, n = 3 * k + 2 ∧ n > 0 }

theorem largest_element_in_A_inter_B : ∃ x ∈ (A ∩ B), ∀ y ∈ (A ∩ B), y ≤ x ∧ x = 2021 := by
  sorry

end largest_element_in_A_inter_B_l336_336311


namespace apples_collected_l336_336944

def isOdd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem apples_collected (a k n : ℕ):
  a = k * 100 / n → isOdd k → n ∈ {25, 300, 525, 1900, 9900} :=
sorry

end apples_collected_l336_336944


namespace circumcircle_radius_l336_336484

theorem circumcircle_radius (r1 r2 r3 : ℝ) (O1 O2 A B C : ℝ)
  (h1 : r1 + r2 = 7)
  (h2 : (O1 - O2)^2 = 169)
  (h3 : r3 = 5)
  (h4 : ∀ x : ℝ, sqrt x >= 0 ) -- Ensure non-negative results for sqrt
  : (c : ℝ) (h5 : a ^ 2 + b^2 = c^2) → c = sqrt 30 :=
by
  sorry

end circumcircle_radius_l336_336484


namespace car_speed_l336_336791

theorem car_speed {vp vc : ℚ} (h1 : vp = 7 / 2) (h2 : vc = 6 * vp) : 
  vc = 21 := 
by 
  sorry

end car_speed_l336_336791


namespace simplify_expression_l336_336810

theorem simplify_expression :
  sqrt (8 + 6 * sqrt 3) + sqrt (8 - 6 * sqrt 3) = 2 * sqrt 6 :=
by sorry

end simplify_expression_l336_336810


namespace least_trees_required_l336_336173

theorem least_trees_required : ∃ n, (∀ m : ℕ, (m = 4 ∨ m = 5 ∨ m = 6) → n % m = 0) ∧ n = 60 :=
by
  use 60
  split
  · intros m hm
    cases hm <;> simp [hm]
  · rfl

end least_trees_required_l336_336173


namespace matrix_property_l336_336391

-- Given conditions
variables (a b : ℝ)
variables (A : Matrix (Fin 2) (Fin 2) ℝ)

-- Assume b > a^2 and properties of matrix A
theorem matrix_property (h1 : b > a^2) (h2 : Matrix.trace A = 2 * a) (h3 : Matrix.det A = b) :
  Matrix.det (A * A - 2 * a • A + b • (1 : Matrix (Fin 2) (Fin 2) ℝ)) = 0 :=
sorry

end matrix_property_l336_336391


namespace num_terms_before_5_l336_336691

-- Define the arithmetic sequence as a function of n
def arith_seq (a d : ℤ) (n : ℕ) : ℤ := a + d * (n - 1)

-- Definitions matching the conditions
def first_term : ℤ := 105
def common_difference : ℤ := -5

-- The number of terms before the number 5 appears in the sequence
theorem num_terms_before_5 : (n : ℕ) (arith_seq first_term common_difference n = 5) -> n - 1 = 20 :=
  by
  sorry

end num_terms_before_5_l336_336691


namespace breadth_of_rectangular_plot_l336_336450

variable (b l : ℕ)

def length_eq_thrice_breadth (b : ℕ) : ℕ := 3 * b

def area_of_rectangle_eq_2700 (b l : ℕ) : Prop := l * b = 2700

theorem breadth_of_rectangular_plot (h1 : l = 3 * b) (h2 : l * b = 2700) : b = 30 :=
by
  sorry

end breadth_of_rectangular_plot_l336_336450


namespace mark_eats_fruit_l336_336786

-- Question: How many pieces of fruit did Mark eat in the first four days of the week?
theorem mark_eats_fruit (total_fruit : ℕ) (kept_fruit : ℕ) (friday_fruit : ℕ) :
  total_fruit = 10 → kept_fruit = 2 → friday_fruit = 3 → (total_fruit - kept_fruit - friday_fruit) = 5 :=
by
  intros h_total h_kept h_friday
  rw [h_total, h_kept, h_friday]
  simp
  exact rfl

end mark_eats_fruit_l336_336786


namespace cyclist_problem_l336_336108

theorem cyclist_problem (MP NP : ℝ) (h1 : NP = MP + 30) (h2 : ∀ t : ℝ, t*MP = 10*t) 
  (h3 : ∀ t : ℝ, t*NP = 10*t) 
  (h4 : ∀ t : ℝ, t*MP = 42 → t*(MP + 30) = t*42 - 1/3) : 
  MP = 180 := 
sorry

end cyclist_problem_l336_336108


namespace coloring_count_l336_336579

-- Define the set of colors
inductive Color
| green
| yellow
| purple

-- Define the vertices of the figure
inductive Vertex
| A | B | C | D | E | F | G | H | I

-- Define the edges between the vertices
def edge (v1 v2 : Vertex) : Prop :=
(v1 = Vertex.A ∧ v2 = Vertex.B) ∨ (v1 = Vertex.B ∧ v2 = Vertex.A) ∨
(v1 = Vertex.B ∧ v2 = Vertex.C) ∨ (v1 = Vertex.C ∧ v2 = Vertex.B) ∨
(v1 = Vertex.C ∧ v2 = Vertex.A) ∨ (v1 = Vertex.A ∧ v2 = Vertex.C) ∨
(v1 = Vertex.D ∧ v2 = Vertex.E) ∨ (v1 = Vertex.E ∧ v2 = Vertex.D) ∨
(v1 = Vertex.E ∧ v2 = Vertex.F) ∨ (v1 = Vertex.F ∧ v2 = Vertex.E) ∨
(v1 = Vertex.F ∧ v2 = Vertex.D) ∨ (v1 = Vertex.D ∧ v2 = Vertex.F) ∨
(v1 = Vertex.G ∧ v2 = Vertex.H) ∨ (v1 = Vertex.H ∧ v2 = Vertex.G) ∨
(v1 = Vertex.H ∧ v2 = Vertex.I) ∨ (v1 = Vertex.I ∧ v2 = Vertex.H) ∨
(v1 = Vertex.I ∧ v2 = Vertex.G) ∨ (v1 = Vertex.G ∧ v2 = Vertex.I) ∨
(v1 = Vertex.C ∧ v2 = Vertex.I) ∨ (v1 = Vertex.I ∧ v2 = Vertex.C) ∨
(v1 = Vertex.B ∧ v2 = Vertex.H) ∨ (v1 = Vertex.H ∧ v2 = Vertex.B)

-- Define a coloring function that assigns a color to each vertex
def Coloring := Vertex → Color

-- Define a property that validates a coloring
def valid_coloring (c : Coloring) : Prop :=
∀ v1 v2, edge v1 v2 → c v1 ≠ c v2

-- Main theorem statement
theorem coloring_count : 
  ∃ (count : ℕ), count = 96 ∧ 
  ∃ c : list Coloring, (∀ col ∈ c, valid_coloring col) ∧ c.length = count :=
sorry

end coloring_count_l336_336579


namespace categorize_numbers_l336_336590

-- Declare the given numbers
def given_numbers : set ℝ := {-2.5, 5.5, 0, 8, -2, Real.pi / 2, 0.7, -2/3, -1.121121112, 3/4, - (5 / 99)}

-- Define what it means to be a negative fraction
def is_negative_fraction (x : ℝ) := ∃ (p q : ℤ), q ≠ 0 ∧ x = -(p / q)

-- Define the sets we want to categorize into
def negative_fraction_set := {-2.5, -2 / 3, -(5 / 99)}
def integer_set := {0, 8, -2}
def rational_number_set := {-2.5, 5.5, 0, 8, -2, 0.7, -2/3, 3/4, - (5/99)}
def non_positive_integer_set := {0, -2}

-- Prove each set categorization
theorem categorize_numbers : 
  (∀ x ∈ negative_fraction_set, x ∈ given_numbers ∧ is_negative_fraction x) ∧ 
  (∀ x ∈ integer_set, x ∈ given_numbers ∧ x ∈ ℤ) ∧ 
  (∀ x ∈ rational_number_set, x ∈ given_numbers ∧ ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q) ∧ 
  (∀ x ∈ non_positive_integer_set, x ∈ given_numbers ∧ x ≤ 0 ∧ x ∈ ℤ) :=
by sorry

end categorize_numbers_l336_336590


namespace fill_time_correct_l336_336526

-- Define the conditions
def rightEyeTime := 2 * 24 -- hours
def leftEyeTime := 3 * 24 -- hours
def rightFootTime := 4 * 24 -- hours
def throatTime := 6       -- hours

def rightEyeRate := 1 / rightEyeTime
def leftEyeRate := 1 / leftEyeTime
def rightFootRate := 1 / rightFootTime
def throatRate := 1 / throatTime

-- Combined rate calculation
def combinedRate := rightEyeRate + leftEyeRate + rightFootRate + throatRate

-- Goal definition
def fillTime := 288 / 61 -- hours

-- Prove that the calculated time to fill the pool matches the given answer
theorem fill_time_correct : (1 / combinedRate) = fillTime :=
by {
  sorry
}

end fill_time_correct_l336_336526


namespace Sara_Jim_equal_savings_l336_336050

theorem Sara_Jim_equal_savings:
  ∃ (w : ℕ), (∃ (sara_saved jim_saved : ℕ),
  sara_saved = 4100 + 10 * w ∧
  jim_saved = 15 * w ∧
  sara_saved = jim_saved) → w = 820 :=
by
  sorry

end Sara_Jim_equal_savings_l336_336050


namespace kendra_shirts_for_two_weeks_l336_336384

def school_days := 5
def after_school_club_days := 3
def one_week_shirts := school_days + after_school_club_days + 1 (Saturday) + 2 (Sunday)
def two_weeks_shirts := one_week_shirts * 2

theorem kendra_shirts_for_two_weeks : two_weeks_shirts = 22 :=
by
  -- Prove the theorem
  sorry

end kendra_shirts_for_two_weeks_l336_336384


namespace b_seq_arithmetic_sum_of_c_seq_l336_336464

-- Definitions from the problem
def a_seq : ℕ → ℤ
| 1 := 1
| 2 := 2
| (n + 3) := 2 * a_seq (n + 2) - a_seq (n + 1) + 2

-- Problem (1): Proving {b_n} is an arithmetic sequence
def b_seq (n : ℕ) : ℤ := a_seq (n + 1) - a_seq n

theorem b_seq_arithmetic (n : ℕ) :
  b_seq (n + 1) = b_seq n + 2 :=
sorry

-- Problem (2): Sum of the first n terms of {c_n}
def c_seq (n : ℕ) : ℚ := 1 / (a_seq n + 5 * n)

def S (n : ℕ) : ℚ := ∑ i in finset.range n, c_seq (i + 1)

theorem sum_of_c_seq (n : ℕ) :
  S n = n / (2 * (n + 2)) :=
sorry

end b_seq_arithmetic_sum_of_c_seq_l336_336464


namespace function_zeros_range_l336_336670

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then (1 / 2)^x + 2 / x else x * Real.log x - a

theorem function_zeros_range (a : ℝ) :
  (∀ x : ℝ, f x a = 0 → x < 0) ∧ (∀ x : ℝ, f x a = 0 → x > 0 → (a > -1 / Real.exp 1 ∧ a < 0)) ↔
  (a > -1 / Real.exp 1 ∧ a < 0) :=
sorry

end function_zeros_range_l336_336670


namespace original_population_is_1500_l336_336170

variable (p : ℤ)

def population_after_increase (p : ℤ) : ℤ := p + 1500

def population_after_decrease (p : ℤ) : ℤ :=
  0.85 * population_after_increase p

def final_population (p : ℤ) : ℤ :=
  population_after_decrease p

theorem original_population_is_1500 :
  ∃ p : ℤ, final_population p = p + 50 -> p = 1500 := 
sorry

end original_population_is_1500_l336_336170


namespace select_two_doctors_l336_336099

theorem select_two_doctors (n k : ℕ) (h1 : n = 6) (h2 : k = 2) : 
  (nat.choose n k = 15) :=
by
  rw [h1, h2]
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)

end select_two_doctors_l336_336099


namespace kekai_garage_sale_l336_336381

theorem kekai_garage_sale :
  let shirts := 5
  let shirt_price := 1
  let pants := 5
  let pant_price := 3
  let total_money := (shirts * shirt_price) + (pants * pant_price)
  let money_kept := total_money / 2
  money_kept = 10 :=
by
  sorry

end kekai_garage_sale_l336_336381


namespace count_no_carry_pairs_l336_336622

-- Define the range of integers from 1500 to 2500
def range_integers : List ℕ := List.range' 1500 (2500 - 1500 + 1)

-- Define a function to check for no carry condition when adding two consecutive integers
def no_carry (n m : ℕ) : Prop :=
  let digits := List.zip (n.digits 10) (m.digits 10)
  ∀ (a b : ℕ) in digits, a + b < 10

-- Count pairs of consecutive integers that satisfy the no carry condition
def count_valid_pairs (lst : List ℕ) : ℕ :=
  (lst.zip (lst.tail)).count (λ (p : ℕ × ℕ), no_carry p.1 p.2)

-- The theorem to prove the total number of such valid pairs
theorem count_no_carry_pairs : count_valid_pairs range_integers = 1100 :=
by
  sorry

end count_no_carry_pairs_l336_336622


namespace hypotenuse_length_l336_336252

theorem hypotenuse_length (x : ℝ) (h₀ : 0 < x) (h₁ : x < π / 2) (sin_x cos_x : ℝ) 
  (h₂ : sin x = sin_x) (h₃ : cos x = cos_x) :
  ∃ l : ℝ, l = 3 * (sqrt (sin_x^2 + cos_x^2)) / √5 ∧ l = 3 * (sqrt (1 / 5)) ∧ l = 3 * (1 / √5)∧ l = (3 * (√5 / 5))  :=
by
  sorry

end hypotenuse_length_l336_336252


namespace total_amount_correct_l336_336075

-- Define necessary conditions
def cash_realized : ℝ := 106.25
def brokerage_rate : ℝ := 0.25 / 100

-- Define brokerage amount
def brokerage_amount : ℝ := (brokerage_rate) * cash_realized

-- Define rounded brokerage amount
def rounded_brokerage_amount : ℝ := Float.round (brokerage_amount * 100) / 100

-- Define the total amount including brokerage
def total_amount_including_brokerage : ℝ := cash_realized + rounded_brokerage_amount

-- State the proof problem
theorem total_amount_correct : total_amount_including_brokerage = 106.52 := by
  sorry

end total_amount_correct_l336_336075


namespace geometric_series_sum_l336_336207

theorem geometric_series_sum :
  let a := (1 / 4 : ℚ)
  let r := (1 / 4 : ℚ)
  let n := 5
  (a * (1 - r^n) / (1 - r)) = (1023 / 3072 : ℚ) :=
by
  let a := (1 / 4 : ℚ)
  let r := (1 / 4 : ℚ)
  let n := 5
  have h₁ : 1 - r^n = 1 - (1 / 4)^5 := by sorry
  have h₂ : (1 / 4)^5 = 1 / 1024 := by sorry
  have h₃ : 1 - 1 / 1024 = 1023 / 1024 := by sorry
  have h₄ : 1 - r = 3 / 4 := by sorry
  calc
    (a * (1 - r^n) / (1 - r)) = (1 / 4 * (1023 / 1024) / (3 / 4)) : by sorry
                           ... = 1023 / 3072 : by sorry

end geometric_series_sum_l336_336207


namespace perfect_number_unique_l336_336120

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p
def is_perfect (n : ℕ) : Prop := (∑ d in (finset.range (n + 1)).filter (λ d, d ∣ n), d) = 2 * n

theorem perfect_number_unique (n : ℕ) (h_perfect : is_perfect n) (h_prev_prime : is_prime (n - 1)) (h_next_prime : is_prime (n + 1)) : n = 6 :=
by {
  sorry
}

end perfect_number_unique_l336_336120


namespace simplest_square_root_l336_336887

theorem simplest_square_root (A B C D : ℝ) (hA : A = real.sqrt 2.5) (hB : B = real.sqrt 8) (hC : C = real.sqrt (1/3)) (hD : D = real.sqrt 5) :
  simplest_form D :=
by
  sorry

end simplest_square_root_l336_336887


namespace range_of_a_l336_336295

open Real

theorem range_of_a (a : ℝ) (m : ℝ) (h1: m > 1) 
    (h2: let f := (λ x : ℝ, exp x + a * x ^ 2) in
           let l := f m - exp m * m - a * m ^ 2 in
           l < 1) : 
    -1 ≤ a := sorry

end range_of_a_l336_336295


namespace square_side_length_l336_336326

theorem square_side_length (length_rect width_rect : ℕ) (h_length : length_rect = 400) (h_width : width_rect = 300)
  (h_perimeter : 4 * side_length = 2 * (2 * (length_rect + width_rect))) : side_length = 700 := by
  -- Proof goes here
  sorry

end square_side_length_l336_336326


namespace train_cross_time_l336_336740

def km_per_hr_to_m_per_s (v : ℝ) : ℝ := v * 1000 / 3600

def time_to_cross_train (length : ℝ) (speed : ℝ) : ℝ := length / speed

theorem train_cross_time
  (length_of_train : ℝ)
  (speed_of_train_km_hr : ℝ)
  (conversion_factor : ℝ = 1000 / 3600)
  (converted_speed : ℝ := km_per_hr_to_m_per_s speed_of_train_km_hr)
  (expected_time : ℝ := 1.60) :
  time_to_cross_train length_of_train converted_speed = expected_time := sorry

end train_cross_time_l336_336740


namespace area_ratio_l336_336556

variables (A B C P D E F O H : Type)

-- Assume triangle ABC
variables [is_triangle A B C]

-- D, E, F are feet of perpendiculars from P to BC, CA, AB respectively
variables [is_perpendicular_from P D B C]
variables [is_perpendicular_from P E C A]
variables [is_perpendicular_from P F A B]

-- O is the circumcenter of triangle ABC
variable [is_circumcenter O A B C]

-- H is the orthocenter of triangle ABC
variable [is_orthocenter H A B C]

-- Areas of triangles
variables (S_HDE S_HDF S_OAB S_OAC : ℝ)
variables [has_area S_HDE H D E]
variables [has_area S_HDF H D F]
variables [has_area S_OAB O A B]
variables [has_area S_OAC O A C]

-- The theorem to be proved
theorem area_ratio:
  S_HDE / S_HDF = S_OAB / S_OAC := sorry

end area_ratio_l336_336556


namespace min_sum_abc_l336_336845

theorem min_sum_abc (a b c : ℕ) (h1 : a * b * c = 3960) : a + b + c ≥ 150 :=
sorry

end min_sum_abc_l336_336845


namespace algebra_eq_iff_sum_eq_one_l336_336817

-- Definitions from conditions
def expr1 (a b c : ℝ) : ℝ := a + b * c
def expr2 (a b c : ℝ) : ℝ := (a + b) * (a + c)

-- Lean statement for the proof problem
theorem algebra_eq_iff_sum_eq_one (a b c : ℝ) : expr1 a b c = expr2 a b c ↔ a + b + c = 1 :=
by
  sorry

end algebra_eq_iff_sum_eq_one_l336_336817


namespace range_of_m_l336_336300

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + |x - 1| ≥ (m + 2) * x - 1) ↔ (-3 - 2 * Real.sqrt 2) ≤ m ∧ m ≤ 0 := 
sorry

end range_of_m_l336_336300


namespace circle_through_BC_l336_336166

theorem circle_through_BC
  (A B C P Q A1 : Type)
  (k : circle)
  (triangle_ABC : scalene_triangle A B C)
  (circle_k_passes_through_BC : k.passes_through B C)
  (k_meets_extensions : k.meets AB P ∧ k.meets AC Q)
  (A1_is_foot : foot A A1 BC)
  (A1P_eq_A1Q : A1P = A1Q)
  : ∠ PA1Q = 2 * ∠ BAC :=
sorry

end circle_through_BC_l336_336166


namespace qb_eq_qr_l336_336393

-- Given a triangle ABC with angles A, B, C
variables {A B C P Q D R : Point}
variables {angleB : ℝ} {angleC : ℝ}

-- Conditions given in the problem
axiom angle_B_gt_C : angleB > angleC
axiom angle_PBA : ∠PBA = angleC
axiom angle_QBA : ∠QBA = angleC
axiom A_between_P_and_C : between A P C
axiom D_on_BQ : D ∈ segment B Q
axiom PD_eq_PB : PD = PB
axiom R_on_AD_circumcircle : R ∈ (AD ∩ circumcircle ABC) ∧ R ≠ A

-- Prove that QR = QB
theorem qb_eq_qr : dist Q B = dist Q R :=
sorry

end qb_eq_qr_l336_336393


namespace integral_split_and_solve_l336_336560

theorem integral_split_and_solve :
  (∫ x in -2..2, x + real.sqrt (4 - x^2)) = 2 * real.pi :=
by
  have h1 : (∫ x in -2..2, x) = 0 := sorry
  have h2 : (∫ x in -2..2, real.sqrt (4 - x^2)) = 2 * real.pi := sorry
  calc
    (∫ x in -2..2, x + real.sqrt (4 - x^2))
        = (∫ x in -2..2, x) + (∫ x in -2..2, real.sqrt (4 - x^2)) : by sorry
    ... = 0 + 2 * real.pi : by rw [h1, h2]
    ... = 2 * real.pi : by sorry

end integral_split_and_solve_l336_336560


namespace sector_area_l336_336517

theorem sector_area (l d : ℕ) (hl : l = 20) (hd : d = 24) : 
  let r := d / 2 in
  (1 / 2 : ℚ) * l * r = 120 := by
  sorry

end sector_area_l336_336517


namespace correct_choice_is_D_l336_336197

def areStatementsCorrect : Prop :=
  let s1 := ∀ (v : Vector ℝ), v.length = 0 → v = 0
  let s2 := ∀ (v1 v2 : Vector ℝ), v1 = 0 ∧ v2 = 0 → v1.direction = v2.direction
  let s3 := ∀ (v : Vector ℝ), v.length = 1 → v.length = 1
  let s4 := ∀ (v1 v2 : Vector ℝ), v1.length = 1 ∧ v2.length = 1 → v1.direction = v2.direction
  let s5 := ∀ (v : Vector ℝ), collinear v 0
  s1 ∧ ¬s2 ∧ s3 ∧ ¬s4 ∧ s5

theorem correct_choice_is_D : areStatementsCorrect = true :=
by
  sorry

end correct_choice_is_D_l336_336197


namespace gross_profit_percentage_l336_336463

theorem gross_profit_percentage (sales_price gross_profit cost : ℝ) 
  (h1 : sales_price = 81) 
  (h2 : gross_profit = 51) 
  (h3 : cost = sales_price - gross_profit) : 
  (gross_profit / cost) * 100 = 170 := 
by
  simp [h1, h2, h3]
  sorry

end gross_profit_percentage_l336_336463


namespace jellybean_count_is_65_l336_336000

noncomputable def final_jellybean_count : ℕ :=
  let initial := 100
  let after_emma1 := initial / 2
  let after_emma2 := after_emma1 + 20
  let after_pat1 := after_emma2 - (after_emma2 / 4)
  let after_pat2 := after_pat1 + 15
  let after_lily := after_pat2 * 2
  let after_alex := after_lily - 12
  let after_george := after_alex + 7.5
  let after_noah := after_george * 1.5
  let after_michelle := after_noah - (after_noah * 2 / 3)
  after_michelle.toNat

theorem jellybean_count_is_65 :
  final_jellybean_count = 65 := sorry

end jellybean_count_is_65_l336_336000


namespace soil_bags_needed_l336_336190

def raised_bed_length : ℝ := 8
def raised_bed_width : ℝ := 4
def raised_bed_height : ℝ := 1
def soil_bag_volume : ℝ := 4
def num_raised_beds : ℕ := 2

theorem soil_bags_needed : (raised_bed_length * raised_bed_width * raised_bed_height * num_raised_beds) / soil_bag_volume = 16 := 
by
  sorry

end soil_bags_needed_l336_336190


namespace sum_of_c_with_4_solutions_l336_336832

def g (x : ℝ) : ℝ := (1 / 120) * (x - 4) * (x - 2) * (x + 2) * (x + 4) - 2

theorem sum_of_c_with_4_solutions :
  ∑ c in {c : ℤ | ∃ (x1 x2 x3 x4 : ℝ), g x1 = c ∧ g x2 = c ∧ g x3 = c ∧ g x4 = c ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4}.to_finset, c = -3 :=
by
  sorry

end sum_of_c_with_4_solutions_l336_336832


namespace incircle_tangent_circumcircle_l336_336833

-- Define the geometric structures and properties.
def triangle := Type
variable {A B C K D M N : triangle}
variable {Ω circ : set triangle}

-- Properties from the problem statement
variable (incircle : Ω.is_incircle A B C)
variable (tangent_at_K : Ω.is_tangent BC K)
variable (altitude : is_altitude A D)
variable (midpoint : is_midpoint AD M)
variable (common_intersection : Ω.common_intersection KM N)

-- Proving the tangent property
theorem incircle_tangent_circumcircle
  (incircle : Ω.is_incircle A B C)
  (tangent_at_K : Ω.is_tangent BC K)
  (altitude : is_altitude A D)
  (midpoint : is_midpoint AD M)
  (common_intersection : Ω.common_intersection KM N) :
  Ω.is_tangent (circumcircle B C N) N :=
sorry

end incircle_tangent_circumcircle_l336_336833


namespace a_initial_investment_l336_336183

noncomputable def initial_investment_A (y : ℝ) : ℝ :=
  let ratio := (y * 7 * 2) / (3 * 12) in
  ratio

theorem a_initial_investment (y : ℝ) (hy : y = 9000) : 
  initial_investment_A y = 3500 := by
  have h : initial_investment_A 9000 = 3500 := 
    calc initial_investment_A 9000
          = (9000 * 7 * 2) / (3 * 12) : by rfl
      ... = 42000 / 12 : by norm_num
      ... = 3500 : by norm_num
  exact h.subst hy

#eval a_initial_investment 9000 rfl

end a_initial_investment_l336_336183


namespace security_mistakes_and_measures_l336_336891

structure UserAction :=
  (received_email : String)
  (email_link : String)
  (entered_info : Bool)
  (bank_sms : String)
  (purchase_email : String)
  (temporary_block_lifted : Bool)

def suspiciousEmail (email : String) : Bool :=
  email.contains "aliexpress@best_prices.net"

def suspiciousLink (link : String) : Bool :=
  link.contains "aliexpres__best_prices.net"

def trustedUnverifiedEmail (email : String) (info_entered : Bool) : Prop :=
  suspiciousEmail email ∧ info_entered

def trustedUnverifiedLink (link : String) (info_entered : Bool) : Prop :=
  suspiciousLink link ∧ info_entered

def unusualOffer (email : String) (price_reduction : Int) : Prop :=
  email.contains "won a lottery" ∧ price_reduction > 35000

def didNotVerify (email : String) (link : String) (info_entered : Bool) : Prop :=
  info_entered ∧ suspiciousEmail email ∧ suspiciousLink link

def is_action_secure (user_action : UserAction) : Prop :=
  ¬trustedUnverifiedEmail user_action.received_email user_action.entered_info ∧
  ¬trustedUnverifiedLink user_action.email_link user_action.entered_info ∧
  ¬unusualOffer user_action.received_email 38990 ∧
  ¬didNotVerify user_action.received_email user_action.email_link user_action.entered_info

theorem security_mistakes_and_measures (user_action : UserAction) :
  is_action_secure user_action :=
sorry

end security_mistakes_and_measures_l336_336891


namespace liza_total_balance_l336_336796

-- Define the initial conditions and transactions in Lean
def initial_balance_usd := 800
def rent_payment_usd := 450
def euro_exchange_rate := 0.85
def paycheck_usd := 1500
def half_paycheck_usd := paycheck_usd / 2
def usd_to_gbp_exchange_rate := 0.72
def utilities_bill_usd := 217
def phone_bill_usd := 70
def bank_transfer_euro := 500
def euro_to_usd_exchange_rate := 1.21
def interest_rate := 0.015
def grocery_percent := 0.2

-- Final balances
def final_balance_usd := 1439.27
def final_balance_uk_gbp := 311.04
def gbp_to_usd_conversion_rate := 1 / usd_to_gbp_exchange_rate
def final_balance_uk_usd := final_balance_uk_gbp * gbp_to_usd_conversion_rate
def total_balance_usd := final_balance_usd + final_balance_uk_usd

theorem liza_total_balance : total_balance_usd = 1871.19 := 
by 
  have final_balance_usd_eq := initial_balance_usd - rent_payment_usd + half_paycheck_usd - utilities_bill_usd - phone_bill_usd + (bank_transfer_euro * euro_to_usd_exchange_rate) + (initial_balance_usd - rent_payment_usd + half_paycheck_usd - utilities_bill_usd - phone_bill_usd + (bank_transfer_euro * euro_to_usd_exchange_rate)) * interest_rate
  have final_balance_uk_eq := half_paycheck_usd * usd_to_gbp_exchange_rate * (1 - grocery_percent)
  have total_balance_eq := final_balance_usd_eq + (final_balance_uk_eq * gbp_to_usd_conversion_rate)
  exact sorry

end liza_total_balance_l336_336796


namespace intersection_point_ellipse_hyperbola_l336_336658

open Real

noncomputable def ellipse (x y m n : ℝ) :=
  x^2 / m + y^2 / n = 1

noncomputable def hyperbola (x y p q : ℝ) :=
  x^2 / p - y^2 / q = 1

theorem intersection_point_ellipse_hyperbola
  (m n p q x y : ℝ)
  (hm : 0 < m) (hn : 0 < n) (hp : 0 < p) (hq : 0 < q)
  (same_foci : ellipse x y m n ∧ hyperbola x y p q)
  :
  ∃ F1 F2 P,
    (|dist P F1| * |dist P F2| = m - p) := sorry

end intersection_point_ellipse_hyperbola_l336_336658


namespace sum_tangents_identity_l336_336043

theorem sum_tangents_identity (θ : ℝ) (n : ℕ) (h : 0 < n) :
  (∑ j in Finset.range n, Real.tan (θ + j * π / n)) = 
    if Odd n then n * Real.tan (n * θ) else -n * Real.cot (n * θ) :=
sorry

end sum_tangents_identity_l336_336043


namespace find_x_coordinate_l336_336266

open Real

noncomputable def point_on_parabola (x y : ℝ) : Prop :=
  y^2 = 6 * x ∧ x > 0 

noncomputable def is_twice_distance (x : ℝ) : Prop :=
  let focus_x : ℝ := 3 / 2
  let d1 := x + focus_x
  let d2 := x
  d1 = 2 * d2

theorem find_x_coordinate (x y : ℝ) :
  point_on_parabola x y →
  is_twice_distance x →
  x = 3 / 2 :=
by
  intros
  sorry

end find_x_coordinate_l336_336266


namespace total_distance_traveled_l336_336529

noncomputable def totalDistance
  (d1 d2 : ℝ) (s1 s2 : ℝ) (average_speed : ℝ) (total_time : ℝ) : ℝ := 
  average_speed * total_time

theorem total_distance_traveled :
  let d1 := 160
  let s1 := 64
  let d2 := 160
  let s2 := 80
  let average_speed := 71.11111111111111
  let total_time := d1 / s1 + d2 / s2
  totalDistance d1 d2 s1 s2 average_speed total_time = 320 :=
by
  -- This is the main statement theorem
  sorry

end total_distance_traveled_l336_336529


namespace tangents_from_point_to_circle_l336_336595

def point : Type := ℝ × ℝ

-- Define the point P(1, -2)
def P : point := (1, -2)

-- Define the equation of the circle
def circle (x y : ℝ) : Prop := x^2 + y^2 - 6 * x - 2 * y + 6 = 0

-- Define the first tangent line equation
def tangent1 (x y : ℝ) : Prop := 5 * x - 12 * y - 29 = 0

-- Define the second tangent line equation
def tangent2 (x y : ℝ) : Prop := x = 1

theorem tangents_from_point_to_circle :
  (∀ x y : ℝ, circle x y → (tangent1 x y ∨ tangent2 x y)) :=
sorry

end tangents_from_point_to_circle_l336_336595


namespace cos2Y_minus_cos4γ_equals_three_over_25_l336_336367

variable (Y γ : ℝ) 

theorem cos2Y_minus_cos4γ_equals_three_over_25 
  (h : 4 * (Real.tan Y)^2 + 4 * (Real.cot Y)^2 - 1 / (Real.sin γ)^2 - 1 / (Real.cos γ)^2 = 17) : 
  (Real.cos Y)^2 - (Real.cos γ)^4 = 3 / 25 := 
sorry

end cos2Y_minus_cos4γ_equals_three_over_25_l336_336367


namespace number_of_divisors_of_cube_l336_336341

theorem number_of_divisors_of_cube (x : ℕ) (hx : ∃ p : ℕ, Prime p ∧ x = p^2 ∧ ∀ d : ℕ, d ∣ x → d ∈ {1, p, p^2}) :
  ∃ n : ℕ, n = 7 ∧ (∀ d : ℕ, d ∣ x^3 → d ∈ {1, p, p^2, p^3, p^4, p^5, p^6}) := 
by
  sorry

end number_of_divisors_of_cube_l336_336341


namespace aqua_park_earnings_correct_l336_336943

def cost_admission := 12
def cost_tour := 6
def cost_meal := 10
def cost_souvenir := 8

def group10_count := 10
def group15_count := 15
def group8_count := 8

def group10_admission_cost := group10_count * cost_admission
def group10_tour_cost := group10_count * cost_tour
def group10_meal_cost := group10_count * cost_meal
def group10_souvenir_cost := group10_count * cost_souvenir
def group10_total := group10_admission_cost + group10_tour_cost + group10_meal_cost + group10_souvenir_cost

def group15_admission_cost := group15_count * cost_admission
def group15_meal_cost := group15_count * cost_meal
def group15_total := group15_admission_cost + group15_meal_cost

def group8_admission_cost := group8_count * cost_admission
def group8_tour_cost := group8_count * cost_tour
def group8_souvenir_cost := group8_count * cost_souvenir
def group8_total := group8_admission_cost + group8_tour_cost + group8_souvenir_cost

def total_earnings := group10_total + group15_total + group8_total

theorem aqua_park_earnings_correct : total_earnings = 898 :=
by
  -- decomposing the costs to ensure each part is accurate
  have group10_cost := group10_total
  have group15_cost := group15_total
  have group8_cost := group8_total
  have total := group10_cost + group15_cost + group8_cost
  show total = 898
  sorry

end aqua_park_earnings_correct_l336_336943


namespace base_b_sum_is_44_l336_336842

noncomputable def s_in_base_b (b : ℕ) : ℕ :=
  let x := b + 2
  let y := b + 5
  let z := b + 6
  x + y + z

theorem base_b_sum_is_44 :
  ∀ b : ℕ, (b - 3) * (b ^ 3 - 6 * b ^ 2 - 24 * b - 27) = 0 → s_in_base_b b = 44 :=
begin
  intros b h,
  -- proof will go here
  sorry
end

end base_b_sum_is_44_l336_336842


namespace candy_per_day_eq_eight_l336_336248

def candy_received_from_neighbors : ℝ := 11.0
def candy_received_from_sister : ℝ := 5.0
def days_candy_lasted : ℝ := 2.0

theorem candy_per_day_eq_eight :
  (candy_received_from_neighbors + candy_received_from_sister) / days_candy_lasted = 8.0 :=
by
  sorry

end candy_per_day_eq_eight_l336_336248


namespace directrix_of_parabola_l336_336240

-- Define the equation of the parabola and what we need to prove
def parabola_equation (x : ℝ) : ℝ := 2 * x^2 + 6

-- Theorem stating the directrix of the given parabola
theorem directrix_of_parabola :
  ∀ x : ℝ, y = parabola_equation x → y = 47 / 8 := 
by
  sorry

end directrix_of_parabola_l336_336240


namespace problem_a_add_b_eq_five_l336_336466

variable {a b : ℝ}

theorem problem_a_add_b_eq_five
  (h1 : ∀ x, -2 < x ∧ x < 3 → ax^2 + x + b > 0)
  (h2 : a < 0) :
  a + b = 5 :=
sorry

end problem_a_add_b_eq_five_l336_336466


namespace ratio_BK_PM_l336_336033

-- Given conditions
variable (A B C K M P : Point)
variable (hK_on_AB : K ∈ Seg A B)
variable (hP_on_AM : P ∈ Seg A M)
variable (hMedian_AM : isMedian A M B C)
variable (hCK_intersect : CK ∩ AM = {P})
variable (hAK_eq_AP : dist A K = dist A P)

-- Desired ratio
theorem ratio_BK_PM : ratio BK PM = 2 :=
  sorry

end ratio_BK_PM_l336_336033


namespace bags_needed_l336_336188

-- Define the dimensions of one raised bed
def length_of_bed := 8
def width_of_bed := 4
def height_of_bed := 1

-- Calculate the volume of one raised bed
def volume_of_one_bed := length_of_bed * width_of_bed * height_of_bed

-- Define the number of beds
def number_of_beds := 2

-- Calculate the total volume needed for both beds
def total_volume := number_of_beds * volume_of_one_bed

-- Define the volume of soil in one bag
def volume_per_bag := 4

-- Calculate the number of bags needed
def number_of_bags := total_volume / volume_per_bag

-- Prove that the number of bags needed is 16
theorem bags_needed : number_of_bags = 16 := by
  show number_of_bags = 16 from sorry

end bags_needed_l336_336188


namespace max_area_rectangle_l336_336547

-- Definition of the problem
def optimalRectangle (p : ℝ) : Prop :=
  let length := p / 2
  let width := p / 4
  length * width = (p^2) / 8

-- Statement of the theorem
theorem max_area_rectangle (p : ℝ) (h : 0 ≤ p) : 
  optimalRectangle p :=
  sorry

end max_area_rectangle_l336_336547


namespace car_distance_l336_336147

theorem car_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (distance : ℝ) 
  (h_speed : speed = 160) 
  (h_time : time = 5) 
  (h_dist_formula : distance = speed * time) : 
  distance = 800 :=
by sorry

end car_distance_l336_336147


namespace sum_roots_fractions_l336_336763

noncomputable def sum_fractions (roots : Fin 2020 → ℂ) : ℂ :=
  Finset.univ.sum (λ n, roots n / (1 - roots n))

theorem sum_roots_fractions :
  let a : Fin 2020 → ℂ := λ n, (polynomial.coeff (X^2020 + X^2019 + ... + X + 1365) n)
  sum_fractions a = 3100.3068702290076 :=
by
  sorry

end sum_roots_fractions_l336_336763


namespace simplify_sqrt_sum_l336_336809

theorem simplify_sqrt_sum : sqrt (8 + 6 * sqrt 3) + sqrt (8 - 6 * sqrt 3) = 2 * sqrt 6 := 
sorry

end simplify_sqrt_sum_l336_336809


namespace factor_x_squared_minus_169_l336_336588

theorem factor_x_squared_minus_169 (x : ℝ) : x^2 - 169 = (x - 13) * (x + 13) := 
by
  -- Recognize that 169 is a perfect square
  have h : 169 = 13^2 := by norm_num
  -- Use the difference of squares formula
  -- Sorry is used to skip the proof part
  sorry

end factor_x_squared_minus_169_l336_336588


namespace determine_remaining_areas_l336_336829

-- Definitions of conditions
def figure_composed_of_eight_squares : Prop := 
  true 

def X_is_midpoint_of_segment_KJ : Prop :=
  true

def Y_is_midpoint_of_segment_EX : Prop :=
  true

def segment_BZ_congruent_to_BC : Prop :=
  true

def area_of_black_part_is_seven_point_five : Prop :=
  7.5 = (5/2) * a^2

def a := sqrt(3) -- inferred from 7.5 = (5/2) * a^2

-- The areas of remaining parts
def white_part_area : Prop :=
  1.5 = (1/2) * a^2

def dark_gray_part_area : Prop :=
  6 = 2 * a^2

def light_gray_part_area : Prop :=
  5.25 = (7/4) * a^2

def shaded_part_area : Prop :=
  3.75 = (5/4) * a^2

-- Final theorem statement
theorem determine_remaining_areas (a : Real)
  (h1 : figure_composed_of_eight_squares)
  (h2 : X_is_midpoint_of_segment_KJ)
  (h3 : Y_is_midpoint_of_segment_EX)
  (h4 : segment_BZ_congruent_to_BC)
  (h5 : area_of_black_part_is_seven_point_five) :
  white_part_area ∧ dark_gray_part_area ∧ light_gray_part_area ∧ shaded_part_area := 
by {
  sorry
}

end determine_remaining_areas_l336_336829


namespace simplify_and_evaluate_expression_l336_336436

theorem simplify_and_evaluate_expression (m n : ℤ) (h_m : m = -1) (h_n : n = 2) :
  3 * m^2 * n - 2 * m * n^2 - 4 * m^2 * n + m * n^2 = 2 :=
by
  sorry

end simplify_and_evaluate_expression_l336_336436


namespace binom_12_6_eq_924_l336_336965

theorem binom_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binom_12_6_eq_924_l336_336965


namespace correct_num_arrangements_l336_336046

open Finset

variable (plants : Finset ℕ) (numLampsWhite : ℕ) (numLampsRed : ℕ) (numLampsBlue : ℕ)

def num_arrangements (plants : Finset ℕ) (numLampsWhite : ℕ) (numLampsRed : ℕ) (numLampsBlue : ℕ) : ℕ := by sorry

theorem correct_num_arrangements :
  num_arrangements (3 : Finset ℕ) 2 2 2 = 49 := by
sorry

end correct_num_arrangements_l336_336046


namespace smallest_prime_after_seven_nonprimes_l336_336501

open Nat

/-- Prove that the smallest prime number that occurs after a sequence of seven consecutive positive
integers, all of which are nonprime, is 97. -/
theorem smallest_prime_after_seven_nonprimes :
  ∃ p : ℕ, (p > 97) ∧ (Prime p) ∧
  (∀ n : ℕ, n ∈ range (p - 8, p) → ¬ Prime n) := sorry

end smallest_prime_after_seven_nonprimes_l336_336501


namespace profit_percentage_l336_336507

theorem profit_percentage (C S : ℝ) (hC : C = 60) (hS : S = 63) : 
  let P := S - C in 
  (P / C) * 100 = 5 :=
by
  -- Insert the proof here
  sorry

end profit_percentage_l336_336507


namespace relationship_S1_S2_S3_l336_336727

-- Definitions based on conditions
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def S1 (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a (i + 1)

def S2 (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a (n + i + 1)

def S3 (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a (2 * n + i + 1)

-- Statement to prove
theorem relationship_S1_S2_S3 {a : ℕ → ℝ} (n : ℕ) 
  (h : arithmetic_seq a) : 
  S1 a n + S3 a n = 2 * S2 a n :=
sorry

end relationship_S1_S2_S3_l336_336727


namespace binom_12_6_eq_924_l336_336964

theorem binom_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binom_12_6_eq_924_l336_336964


namespace range_of_a_l336_336314

theorem range_of_a (a : ℝ) : 
  (∀ x, (x > 2 ∨ x < -1) → ¬(x^2 + 4 * x + a < 0)) → a ≥ 3 :=
by
  sorry

end range_of_a_l336_336314


namespace discount_store_purchase_l336_336159

theorem discount_store_purchase (n x y : ℕ) (hn : 2 * n + (x + y) = 2 * n) 
(h1 : 8 * x + 9 * y = 172) (hx : 0 ≤ x) (hy : 0 ≤ y): 
x = 8 ∧ y = 12 :=
sorry

end discount_store_purchase_l336_336159


namespace intersection_A_B_l336_336010

-- Conditions
def A : Set (ℕ × ℕ) := { (1, 2), (2, 1) }
def B : Set (ℕ × ℕ) := { p | p.fst - p.snd = 1 }

-- Problem statement
theorem intersection_A_B : A ∩ B = { (2, 1) } :=
by
  sorry

end intersection_A_B_l336_336010


namespace shaded_area_of_four_intersecting_circles_l336_336355

theorem shaded_area_of_four_intersecting_circles : 
  (let r := 5 in let area_of_shaded_region := 50 * Real.pi - 100 in area_of_shaded_region) = 50 * Real.pi - 100 :=
by
  sorry

end shaded_area_of_four_intersecting_circles_l336_336355


namespace exists_integers_x_l336_336751

theorem exists_integers_x (a1 a2 a3 : ℤ) (h : 0 < a1 ∧ a1 < a2 ∧ a2 < a3) :
  ∃ (x1 x2 x3 : ℤ), (|x1| + |x2| + |x3| > 0) ∧ (a1 * x1 + a2 * x2 + a3 * x3 = 0) ∧ (max (max (|x1|) (|x2|)) (|x3|) < (2 / Real.sqrt 3 * Real.sqrt a3) + 1) := 
sorry

end exists_integers_x_l336_336751


namespace diameter_calculation_l336_336524

noncomputable def diameter_of_wheel (revolutions : ℝ) (distance : ℝ) : ℝ :=
  let circumference := distance / revolutions
  circumference / Real.pi

theorem diameter_calculation :
  diameter_of_wheel 424.6284501061571 1000 ≈ 0.7495172 :=
by
  sorry

end diameter_calculation_l336_336524


namespace fluctuation_property_sum_zero_l336_336335

theorem fluctuation_property_sum_zero (a : ℕ → ℤ) 
  (h1 : a 1 = 1)
  (h19 : a 19 = 1)
  (h_fluctuation : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n) :
  (∑ i in Finset.range 18, a (i + 1)) = 0 :=
by
  sorry

end fluctuation_property_sum_zero_l336_336335


namespace find_angle_x_l336_336716

theorem find_angle_x
  (A B C : Type)
  (angle_ABC : ℝ)
  (angle_BCA : ℝ)
  (external_angle : ℝ)
  (angle_ABC_eq : angle_ABC = 50)
  (angle_BCA_eq : angle_BCA = 40)
  (external_angle_eq : external_angle = 45) :
  let angle_BAC := 180 - (angle_ABC + angle_BCA) in
  ∃ x : ℝ, x = 180 - external_angle ∧ x = 135 :=
by
  sorry

end find_angle_x_l336_336716


namespace log_base_16_of_1024_eq_five_over_two_l336_336232

theorem log_base_16_of_1024_eq_five_over_two : log 16 1024 = 5 / 2 := by
  sorry

end log_base_16_of_1024_eq_five_over_two_l336_336232


namespace trapezoid_height_l336_336596

variables (a b h : ℝ)

def is_trapezoid (a b h : ℝ) (angle_diag : ℝ) (angle_ext : ℝ) : Prop :=
a < b ∧ angle_diag = 90 ∧ angle_ext = 45

theorem trapezoid_height
  (a b : ℝ) (ha : a < b)
  (angle_diag : ℝ) (h_angle_diag : angle_diag = 90)
  (angle_ext : ℝ) (h_angle_ext : angle_ext = 45)
  (h_def : is_trapezoid a b h angle_diag angle_ext) :
  h = a * b / (b - a) :=
sorry

end trapezoid_height_l336_336596


namespace max_area_triangle_S_l336_336578

theorem max_area_triangle_S (a b c : ℝ) (S : ℝ)
  (hS : S = sqrt (1 / 4 * (a^2 * c^2 - (1 / 2 * (a^2 + c^2 - b^2))^2)))
  (hb : b = 2)
  (hC : ∀ B : ℝ, tan (atan2 (sqrt 3 * sin B) (1 - sqrt 3 * cos B)) = sqrt 3 * sin B / (1 - sqrt 3 * cos B)) :
  ∃ a, S = sqrt 5 :=
by
  sorry

end max_area_triangle_S_l336_336578


namespace locomotive_distance_l336_336176

theorem locomotive_distance 
  (speed_train : ℝ) (speed_sound : ℝ) (time_diff : ℝ)
  (h_train : speed_train = 20) 
  (h_sound : speed_sound = 340) 
  (h_time : time_diff = 4) : 
  ∃ x : ℝ, x = 85 := 
by 
  sorry

end locomotive_distance_l336_336176


namespace pyramid_volume_l336_336074

theorem pyramid_volume (b : ℝ) (h₀ : b > 0) :
  let base_area := (b * b * (Real.sqrt 3)) / 4
  let height := b / 2
  let volume := (1 / 3) * base_area * height
  volume = (b^3 * (Real.sqrt 3)) / 24 :=
sorry

end pyramid_volume_l336_336074


namespace square_vectors_l336_336644

theorem square_vectors (AB CD AD : ℝ × ℝ)
  (side_length: ℝ)
  (M N : ℝ × ℝ)
  (x y: ℝ)
  (MN : ℝ × ℝ):
  side_length = 2 →
  M = ((AB.1 + CD.1) / 2, (AB.2 + CD.2) / 2) →
  N = ((CD.1 + AD.1) / 2, (CD.2 + AD.2) / 2) →
  MN = (x * AB.1 + y * AD.1, x * AB.2 + y * AD.2) →
  (x = -1/2) ∧ (y = 1/2) →
  (x * y = -1/4) ∧ ((N.1 - M.1) * AD.1 + (N.2 - M.2) * AD.2 - (N.1 - M.1) * AB.1 - (N.2 - M.2) * AB.2 = -1) :=
by
  intros side_length_cond M_cond N_cond MN_cond xy_cond
  sorry

end square_vectors_l336_336644


namespace cars_meet_time_l336_336146

-- Define the initial conditions as Lean definitions
def distance_car1 (t : ℝ) : ℝ := 15 * t
def distance_car2 (t : ℝ) : ℝ := 20 * t
def total_distance : ℝ := 105

-- Define the proposition we want to prove
theorem cars_meet_time : ∃ (t : ℝ), distance_car1 t + distance_car2 t = total_distance ∧ t = 3 :=
by
  sorry

end cars_meet_time_l336_336146


namespace minimum_value_of_function_l336_336286

theorem minimum_value_of_function
  (a b : ℝ)
  (h₁ : (a:ℝ) - b = 0)
  (h₂ : (2 * a:ℝ) - b = 4) :
  ∀ x : ℝ, x > 0 → x ≠ 1 → (f x = 4)
where 
  f (x : ℝ) : ℝ := a * Real.log x + b / x :=
by
  sorry

end minimum_value_of_function_l336_336286


namespace Tim_pencils_value_l336_336112

variable (Sarah_pencils : ℕ)
variable (Tyrah_pencils : ℕ)
variable (Tim_pencils : ℕ)

axiom Tyrah_condition : Tyrah_pencils = 6 * Sarah_pencils
axiom Tim_condition : Tim_pencils = 8 * Sarah_pencils
axiom Tyrah_pencils_value : Tyrah_pencils = 12

theorem Tim_pencils_value : Tim_pencils = 16 :=
by
  sorry

end Tim_pencils_value_l336_336112


namespace semicircle_perimeter_approx_l336_336514

def radius : ℝ := 7

def pi_approx : ℝ := 3.14159

def semicircle_perimeter (r : ℝ) (π : ℝ) : ℝ := 2 * r + π * r

theorem semicircle_perimeter_approx : semicircle_perimeter radius pi_approx ≈ 35.99 :=
by
  sorry

end semicircle_perimeter_approx_l336_336514


namespace percentage_increase_in_y_l336_336815

variable (x y k q : ℝ) (h1 : x * y = k) (h2 : x' = x * (1 - q / 100))

theorem percentage_increase_in_y (h1 : x * y = k) (h2 : x' = x * (1 - q / 100)) :
  (y * 100 / (100 - q) - y) / y * 100 = (100 * q) / (100 - q) :=
by
  sorry

end percentage_increase_in_y_l336_336815


namespace monotonic_increase_intervals_exists_pseudo_symmetry_point_l336_336301

noncomputable def f (a x : ℝ) : ℝ := x^2 - (a + 2) * x + a * Real.log x

theorem monotonic_increase_intervals (a : ℝ) (h : a > 2) :
  ∃ I1 I2 : Set ℝ, (I1 = Set.Ioo 0 1) ∧ (I2 = Set.Ioi (a/2)) ∧
  (∀ x ∈ I1, f a x > 0) ∧ (∀ x ∈ I2, f a x > 0) := sorry

structure pseudo_symmetry_point (a x0 : ℝ) :=
  (tangent_line : ℝ → ℝ)
  (property1 : ∀ x ∈ Set.Ioo 0 x0, f a x < tangent_line x)
  (property2 : ∀ x ∈ Set.Ioi x0, f a x > tangent_line x)

theorem exists_pseudo_symmetry_point (a : ℝ) (h : a = 4) :
  ∃ x0 : ℝ, x0 = Real.sqrt 2 ∧ pseudo_symmetry_point a x0 := sorry

end monotonic_increase_intervals_exists_pseudo_symmetry_point_l336_336301


namespace prob_green_is_correct_l336_336215

-- Define the probability of picking any container
def prob_pick_container : ℚ := 1 / 4

-- Define the probability of drawing a green ball from each container
def prob_green_A : ℚ := 6 / 10
def prob_green_B : ℚ := 3 / 10
def prob_green_C : ℚ := 3 / 10
def prob_green_D : ℚ := 5 / 10

-- Define the individual probabilities for a green ball, accounting for container selection
def prob_green_given_A : ℚ := prob_pick_container * prob_green_A
def prob_green_given_B : ℚ := prob_pick_container * prob_green_B
def prob_green_given_C : ℚ := prob_pick_container * prob_green_C
def prob_green_given_D : ℚ := prob_pick_container * prob_green_D

-- Calculate the total probability of selecting a green ball
def prob_green_total : ℚ := prob_green_given_A + prob_green_given_B + prob_green_given_C + prob_green_given_D

-- Theorem statement: The probability of selecting a green ball is 17/40
theorem prob_green_is_correct : prob_green_total = 17 / 40 :=
by
  -- Proof will be provided here.
  sorry

end prob_green_is_correct_l336_336215


namespace johns_children_probability_l336_336373

open Real

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the binomial probability formula
def binom_prob (k n : ℕ) (p : ℝ) : ℝ :=
  (binom n k) * p^k * (1 - p)^(n - k)

-- State the hypothesis and the result
theorem johns_children_probability :
  binom_prob 3 6 0.5 = 5 / 16 :=
sorry

end johns_children_probability_l336_336373


namespace professors_after_reduction_l336_336543

theorem professors_after_reduction (original_faculty : ℝ) (reduction_percentage : ℝ) : 
  original_faculty = 226.74 → reduction_percentage = 0.14 → 
  (original_faculty - (reduction_percentage * original_faculty)).toNat = 195 :=
by
  intros h1 h2
  sorry

end professors_after_reduction_l336_336543


namespace area_enclosed_by_trajectory_l336_336682

-- Define the fixed points A and B
def A : (ℝ × ℝ) := (-2, 0)
def B : (ℝ × ℝ) := (1, 0)

-- Define the trajectory condition for moving point P
def satisfies_condition (P : (ℝ × ℝ)) : Prop :=
  real.sqrt ((P.1 + 2)^2 + P.2^2) = 2 * real.sqrt ((P.1 - 1)^2 + P.2^2)

-- Define the statement to be proven
theorem area_enclosed_by_trajectory : 
  ∀ (P : (ℝ × ℝ)), satisfies_condition P → 
  ∃ (S : ℝ), S = 4 * real.pi :=
by
  sorry

end area_enclosed_by_trajectory_l336_336682


namespace equal_BX_BY_l336_336389

-- Definition of the problem conditions
variables (A B C A1 C1 X Y : Point)

-- Conditions
axiom altitude_AA1 : is_altitude_triangle A A1 B C
axiom altitude_CC1 : is_altitude_triangle C C1 A B
axiom incenter_AA1C : is_incenter_triangle A A1 C
axiom incenter_CC1A : is_incenter_triangle C C1 A
axiom intersect_line : let line := line_through_incenters A A1 C C1 in
                       intersects line AB X ∧ intersects line BC Y

-- The theorem to prove
theorem equal_BX_BY : distance B X = distance B Y :=
sorry

end equal_BX_BY_l336_336389


namespace at_least_one_not_less_than_two_l336_336438

theorem at_least_one_not_less_than_two
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  let a := x + 1 / y
  let b := y + 1 / z
  let c := z + 1 / x
  a >= 2 ∨ b >= 2 ∨ c >= 2 := 
sorry

end at_least_one_not_less_than_two_l336_336438


namespace first_two_bags_to_test_l336_336118

-- Definitions based on the conditions
def total_bags := 850
def bag_numbers : List Nat := List.range (total_bags + 1)
def start_position := (8, 7)

-- Random number table from row 7 to row 9
def random_number_table : List (List Nat) :=
  [[84, 42, 17, 53, 31, 57, 24, 55, 6, 88, 77, 4, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 6, 76],
   [63, 1, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 7, 44, 39, 52, 38, 79],
   [33, 21, 12, 34, 29, 78, 64, 56, 7, 82, 52, 42, 7, 44, 38, 15, 51, 0, 13, 42, 99, 66, 2, 79, 54]]

-- Proof statement
theorem first_two_bags_to_test (start : Nat × Nat := start_position) : 
  (random_number_table.nth! start.fst).nth! start.snd = 7 → 
  (random_number_table.nth! 7).nth! 6 = 57 → 
  (random_number_table.nth! 7).nth! 7 = 24 → 
  (random_number_table.nth! 7).nth! 8 = 55 → 
  (random_number_table.nth! 7).nth! 9 = 6 → 
  (random_number_table.nth! 7).nth! 10 = 88 →
  (random_number_table.nth! 7).nth! 11 = 77 →
  (random_number_table.nth! 7).nth! 12 = 4 →
  (random_number_table.nth! 7).nth! 13 = 74 →
  (random_number_table.nth! 7).nth! 14 = 47 →
  (random_number_table.nth! 7).nth! 15 = 67 →
  (random_number_table.nth! 7).nth! 16 = 21 →
  (random_number_table.nth! 7).nth! 17 = 76 →
  (random_number_table.nth! 7).nth! 18 = 33 →
  (random_number_table.nth! 7).nth! 19 = 50 →
  (random_number_table.nth! 7).nth! 20 = 25 →
  (random_number_table.nth! 7).nth! 21 = 83 →
  (random_number_table.nth! 7).nth! 22 = 92 →
  (random_number_table.nth! 7).nth! 23 = 12 →
  (random_number_table.nth! 7).nth! 24 = 6 →
  (random_number_table.nth! 7).nth! 25 = 76 →
  (random_number_table.nth! 8).nth! 0 = 63 →
  (random_number_table.nth! 8).nth! 1 = 1 →
  (random_number_table.nth! 8).nth! 2 = 63 →
  (random_number_table.nth! 8).nth! 3 = 78 →
  (random_number_table.nth! 8).nth! 4 = 59 →
  (random_number_table.nth! 8).nth! 5 = 16 →
  (random_number_table.nth! 8).nth! 6 = 95 →
  (random_number_table.nth! 8).nth! 7 = 55 →
  (random_number_table.nth! 8).nth! 8 = 67 →
  (random_number_table.nth! 8).nth! 9 = 19 →
  (random_number_table.nth! 8).nth! 10 = 98 →
  (random_number_table.nth! 8).nth! 11 = 10 →
  (random_number_table.nth! 8).nth! 12 = 50 →
  (random_number_table.nth! 8).nth! 13 = 71 →
  (random_number_table.nth! 8).nth! 14 = 75 →
  (random_number_table.nth! 8).nth! 15 = 12 →
  (random_number_table.nth! 8).nth! 16 = 86 →
  (random_number_table.nth! 8).nth! 17 = 73 →
  (random_number_table.nth! 8).nth! 18 = 58 →
  (random_number_table.nth! 8).nth! 19 = 7 →
  (random_number_table.nth! 8).nth! 20 = 44 →
  (random_number_table.nth! 8).nth! 21 = 39 →
  (random_number_table.nth! 8).nth! 22 = 52 →
  (random_number_table.nth! 8).nth! 23 = 38 →
  (random_number_table.nth! 8).nth! 24 = 79 → 
  (random_number_table.nth! 9).nth! 0 = 33 →
  (random_number_table.nth! 9).nth! 1 = 21 →
  (random_number_table.nth! 9).nth! 2 = 12 →
  (random_number_table.nth! 9).nth! 3 = 34 →
  (random_number_table.nth! 9).nth! 4 = 29 →
  (random_number_table.nth! 9).nth! 5 = 78 →
  (random_number_table.nth! 9).nth! 6 = 64 →
  (random_number_table.nth! 9).nth! 7 = 56 →
  (random_number_table.nth! 9).nth! 8 = 7 →
  (random_number_table.nth! 9).nth! 9 = 82 →
  (random_number_table.nth! 9).nth! 10 = 52 →
  (random_number_table.nth! 9).nth! 11 = 42 →
  (random_number_table.nth! 9).nth! 12 = 7 →
  (random_number_table.nth! 9).nth! 13 = 44 →
  (random_number_table.nth! 9).nth! 14 = 38 →
  (random_number_table.nth! 9).nth! 15 = 15 →
  (random_number_table.nth! 9).nth! 16 = 51 →
  (random_number_table.nth! 9).nth! 17 = 0 →
  (random_number_table.nth! 9).nth! 18 = 13 →
  (random_number_table.nth! 9).nth! 19 = 42 →
  (random_number_table.nth! 9).nth! 20 = 99 →
  (random_number_table.nth! 9).nth! 21 = 66 →
  (random_number_table.nth! 9).nth! 22 = 2 →
  (random_number_table.nth! 9).nth! 23 = 79 →
  (random_number_table.nth! 9).nth! 24 = 54 → 
  random_number_table.nth! 8 = some [785, 567] := sorry

end first_two_bags_to_test_l336_336118


namespace sequence_value_at_10_l336_336257

noncomputable def f (x : ℚ) : ℚ := (x^2) / (2*x + 1)
noncomputable def f_seq : ℕ → ℚ → ℚ 
| 0, x   := x
| (n+1), x := f (f_seq n x)

theorem sequence_value_at_10 : 
  f_seq 10 (1/2) = 1 / (3^1024 - 1) :=
sorry

end sequence_value_at_10_l336_336257


namespace min_value_M_l336_336309

noncomputable def a : ℕ → ℝ
| 0     := 2/3
| (n+1) := (∑ i in range (n + 1), a i^2)

theorem min_value_M :
  ∀ (n : ℕ), (∑ i in range (n + 1), 1/(a i + 1)) < 57/20 :=
sorry

end min_value_M_l336_336309


namespace angle_between_clock_hands_at_330_l336_336991

theorem angle_between_clock_hands_at_330 :
  let hour_hand_per_hour_deg := 30
      minute_hand_per_minute_deg := 6
      hours_position := 3
      minutes_position := 30
      hour_hand_deg := hours_position * hour_hand_per_hour_deg + (minutes_position * hour_hand_per_hour_deg / 60)
      minute_hand_deg := minutes_position * minute_hand_per_minute_deg 
    in abs (minute_hand_deg - hour_hand_deg) = 75 := 
by
  let hour_hand_per_hour_deg := 30
  let minute_hand_per_minute_deg := 6
  let hours_position := 3
  let minutes_position := 30
  let hour_hand_deg := hours_position * hour_hand_per_hour_deg + (minutes_position * hour_hand_per_hour_deg / 60)
  let minute_hand_deg := minutes_position * minute_hand_per_minute_deg 
  have h : abs (minute_hand_deg - hour_hand_deg) = 75
  exact h
  sorry

end angle_between_clock_hands_at_330_l336_336991


namespace no_conditions_satisfy_l336_336577

-- Define the conditions
def condition1 (a b c : ℤ) : Prop := a = 1 ∧ b = 1 ∧ c = 1
def condition2 (a b c : ℤ) : Prop := a = b - 1 ∧ b = c - 1
def condition3 (a b c : ℤ) : Prop := a = b ∧ b = c
def condition4 (a b c : ℤ) : Prop := a > c ∧ c = b - 1 

-- Define the equations
def equation1 (a b c : ℤ) : ℤ := a * (a - b)^3 + b * (b - c)^3 + c * (c - a)^3
def equation2 (a b c : ℤ) : Prop := a + b + c = 3

-- Proof statement for the original problem
theorem no_conditions_satisfy (a b c : ℤ) :
  ¬ (condition1 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) ∧
  ¬ (condition2 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) ∧
  ¬ (condition3 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) ∧
  ¬ (condition4 a b c ∧ equation1 a b c = 3 ∧ equation2 a b c) :=
sorry

end no_conditions_satisfy_l336_336577


namespace sum_complex_series_l336_336954

theorem sum_complex_series :
  let i := Complex.i in
  (∑ k in Finset.range 2010, (k + 1) * i ^ (k + 1)) = -1006 + 1005 * i := 
by 
  sorry

end sum_complex_series_l336_336954


namespace find_N_remainder_l336_336390

-- Define the set S and the number N in terms of binomial coefficients
def S := {n : ℕ | ∃ k : ℕ, n = k ∧ (decimal_repr_only_contains 3 7 n) ∧ (decimal_max_digits 1998 n) ∧ (decimal_digit_appears 999 n)}

def N := 2 * (Nat.choose 1999 999) + (Nat.choose 1998 999)

-- Theorem statement
theorem find_N_remainder : N % 1000 = 120 := by
  sorry

end find_N_remainder_l336_336390


namespace ratio_in_sequence_l336_336680

theorem ratio_in_sequence (a1 a2 b1 b2 b3 : ℝ)
  (h1 : ∃ d, a1 = 1 + d ∧ a2 = 1 + 2 * d ∧ 9 = 1 + 3 * d)
  (h2 : ∃ r, b1 = 1 * r ∧ b2 = 1 * r^2 ∧ b3 = 1 * r^3 ∧ 9 = 1 * r^4) :
  b2 / (a1 + a2) = 3 / 10 := by
  sorry

end ratio_in_sequence_l336_336680


namespace triangle_inequalities_l336_336738

theorem triangle_inequalities (A B C : ℝ) (h1: A + B + C = Real.pi) (h2: A > 0) (h3: B > 0) (h4: C > 0) :
  A⁻¹ + B⁻¹ + C⁻¹ ≥ 9 * Real.pi⁻¹ ∧ A⁻² + B⁻² + C⁻² ≥ 27 * Real.pi⁻² := by
  sorry

end triangle_inequalities_l336_336738


namespace calc_6_15_stars_l336_336612

def greatest_pos_even_leq (z: ℝ) : ℕ :=
  if h : ∃ k : ℕ, k % 2 = 0 ∧ (k : ℝ) ≤ z then
    max (classical.some h) 0
  else
    0

-- Example instance for z = 6.15
noncomputable def example_value : ℝ := greatest_pos_even_leq 6.15

theorem calc_6_15_stars :
  6.15 - (greatest_pos_even_leq 6.15) = 2.15 :=
by
  sorry

end calc_6_15_stars_l336_336612


namespace find_initial_days_provisions_last_l336_336915

def initial_days_provisions_last (initial_men reinforcements days_after_reinforcement : ℕ) (x : ℕ) : Prop :=
  initial_men * (x - 15) = (initial_men + reinforcements) * days_after_reinforcement

theorem find_initial_days_provisions_last
  (initial_men reinforcements days_after_reinforcement x : ℕ)
  (h1 : initial_men = 2000)
  (h2 : reinforcements = 1900)
  (h3 : days_after_reinforcement = 20)
  (h4 : initial_days_provisions_last initial_men reinforcements days_after_reinforcement x) :
  x = 54 :=
by
  sorry


end find_initial_days_provisions_last_l336_336915


namespace part1_part2_l336_336673

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := |x - m|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2 * f x m - f (x + m) m

theorem part1 (h : ∀ x, g x m ≥ -1) : m = 1 :=
  sorry

theorem part2 {a b m : ℝ} (ha : |a| < m) (hb : |b| < m) (a_ne_zero : a ≠ 0) (hm: m = 1) : 
  f (a * b) m > |a| * f (b / a) m :=
  sorry

end part1_part2_l336_336673


namespace simplify_expression_l336_336631

theorem simplify_expression (a : ℝ) (h : a < 1 / 4) : 4 * (4 * a - 1)^2 = (1 - 4 * a)^(2 : ℝ) :=
by sorry

end simplify_expression_l336_336631


namespace total_dots_not_visible_l336_336250

theorem total_dots_not_visible
    (num_dice : ℕ)
    (dots_per_die : ℕ)
    (visible_faces : ℕ → ℕ)
    (visible_faces_count : ℕ)
    (total_dots : ℕ)
    (dots_visible : ℕ) :
    num_dice = 4 →
    dots_per_die = 21 →
    visible_faces 0 = 1 →
    visible_faces 1 = 2 →
    visible_faces 2 = 2 →
    visible_faces 3 = 3 →
    visible_faces 4 = 4 →
    visible_faces 5 = 5 →
    visible_faces 6 = 6 →
    visible_faces 7 = 6 →
    visible_faces_count = 8 →
    total_dots = num_dice * dots_per_die →
    dots_visible = visible_faces 0 + visible_faces 1 + visible_faces 2 + visible_faces 3 + visible_faces 4 + visible_faces 5 + visible_faces 6 + visible_faces 7 →
    total_dots - dots_visible = 55 := by
  sorry

end total_dots_not_visible_l336_336250


namespace find_n_l336_336411

-- Define x and y
def x : ℕ := 3
def y : ℕ := 1

-- Define n based on the given expression.
def n : ℕ := x - y^(x - (y + 1))

-- State the theorem
theorem find_n : n = 2 := by
  sorry

end find_n_l336_336411


namespace find_b32_l336_336925

-- Define the polynomial function
def g (z : ℤ) (b : Fin 33 → ℤ) : ℤ :=
  ∏ k in Finset.range 33, (1 - z^k) ^ b ⟨k, nat.lt_succ_self 32⟩

-- Define the given simplification property modulo z^33
def g_mod (z : ℤ) (b : Fin 33 → ℤ) : Prop :=
  g z b = (1 - 2 * z) % (z ^ 33)

-- The theorem to prove
theorem find_b32 : 
  ∃ (b : Fin 33 → ℤ), g_mod z b ∧ b ⟨32, by decide⟩ = 2^27 - 2^11 := 
sorry

end find_b32_l336_336925


namespace exists_prime_and_cube_root_l336_336431

theorem exists_prime_and_cube_root (n : ℕ) (hn : 0 < n) :
  ∃ (p m : ℕ), p.Prime ∧ p % 6 = 5 ∧ ¬p ∣ n ∧ n ≡ m^3 [MOD p] :=
sorry

end exists_prime_and_cube_root_l336_336431


namespace smallest_prime_after_seven_consecutive_nonprimes_l336_336498

theorem smallest_prime_after_seven_consecutive_nonprimes
  (N : Type) [linear_order N] [has_zero N] [has_add N] [has_one N] [is_prime N]
  (s : set N) (pos : ∀ n ∈ s, n > 0) :
  ∃ p ∈ s, is_prime p ∧ 
    (∀ (n1 n2 : N), n1 ∈ s ∧ n2 ∈ s → n1 < n2 → (n2 - n1) > 8 ∧ n2 = 53 :=
begin
  sorry
end

end smallest_prime_after_seven_consecutive_nonprimes_l336_336498


namespace range_of_a_for_inequality_l336_336306

noncomputable def has_solution_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ (x^2 + a*x - 2 < 0)

theorem range_of_a_for_inequality : ∀ a : ℝ, has_solution_in_interval a ↔ a < 1 :=
by sorry

end range_of_a_for_inequality_l336_336306


namespace highlighted_cells_sum_l336_336794

theorem highlighted_cells_sum (numbers : set ℕ) (highlighted_cells : finset ℕ) : 
  (∀ n ∈ numbers, 5 ≤ n ∧ n ≤ 18 ∧ (numbers.card = 14)) →
  (highlighted_cells.card = 3) →
  set.disjoint numbers highlighted_cells →
  (∑ n in highlighted_cells, n = 50 ∨ ∑ n in highlighted_cells, n = 51) :=
by
  intros h_numbers h_highlighted_cells h_disjoint
  -- Sorry, we skip the proof part
  sorry

end highlighted_cells_sum_l336_336794


namespace length_of_first_train_l336_336866

theorem length_of_first_train
  (v1 : ℝ) (v2 : ℝ) (t : ℝ) (l2 : ℝ)
  (hv1 : v1 = 80) (hv2 : v2 = 65)
  (ht : t = 6.851865643851941) (hl2 : l2 = 165) :
  let relative_speed := (v1 + v2) * (1000 / 3600),
      total_distance := relative_speed * t,
      l1 := total_distance - l2 in
  l1 ≈ 111 :=
by
  sorry

end length_of_first_train_l336_336866


namespace calc_27_over_8_pow_2_over_3_l336_336205

theorem calc_27_over_8_pow_2_over_3 : (27 / 8 : ℚ)^(2 / 3 : ℚ) = 9 / 4 := 
  sorry

end calc_27_over_8_pow_2_over_3_l336_336205


namespace wire_length_l336_336820

theorem wire_length (d h1 h2 : ℝ) (h_d : d = 12) (h_h1 : h1 = 6) (h_h2 : h2 = 15) : 
  let height_diff := h2 - h1 in
  let wire_length := Math.sqrt (d^2 + height_diff^2) in
  wire_length = 15 := 
by 
  sorry

end wire_length_l336_336820


namespace maciek_total_cost_l336_336539

-- Define the cost of pretzels without discount
def pretzel_price : ℝ := 4.0

-- Define the discounted price of pretzels when buying 3 or more packs
def pretzel_discount_price : ℝ := 3.5

-- Define the cost of chips without discount
def chips_price : ℝ := 7.0

-- Define the discounted price of chips when buying 2 or more packs
def chips_discount_price : ℝ := 6.0

-- Define the number of pretzels Maciek buys
def pretzels_bought : ℕ := 3

-- Define the number of chips Maciek buys
def chips_bought : ℕ := 4

-- Calculate the total cost of pretzels
def pretzel_cost : ℝ :=
  if pretzels_bought >= 3 then pretzels_bought * pretzel_discount_price else pretzels_bought * pretzel_price

-- Calculate the total cost of chips
def chips_cost : ℝ :=
  if chips_bought >= 2 then chips_bought * chips_discount_price else chips_bought * chips_price

-- Calculate the total amount Maciek needs to pay
def total_cost : ℝ :=
  pretzel_cost + chips_cost

theorem maciek_total_cost :
  total_cost = 34.5 :=
by 
  sorry

end maciek_total_cost_l336_336539


namespace system_of_equations_solution_l336_336095

theorem system_of_equations_solution (x y z : ℝ) (h1 : x + y = 1) (h2 : x + z = 0) (h3 : y + z = -1) : 
    x = 1 ∧ y = 0 ∧ z = -1 := 
by 
  sorry

end system_of_equations_solution_l336_336095


namespace tangent_line_eq_smallest_positive_k_two_distinct_zeros_l336_336298

section math_problems

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Define the function f(x) = 2023x - e^x
def function_f (x : ℝ) : ℝ := 2023 * x - Real.exp x

-- Define the derivative f'(x) = 2023 - e^x
def function_f' (x : ℝ) : ℝ := 2023 - Real.exp x

noncomputable theory

-- Prove the equation of the tangent line to the curve y = f(x) at (0, f(0)) is y = 2022x - 1
theorem tangent_line_eq : 
  ∃ (m : ℝ), (m = 2022) ∧ (∀ x, (function_f' x) = m) ∧ (∀ x, function_f 0 = -1) := 
sorry

-- Prove the smallest positive integer k for which f(x) ≤ kx holds true is 2021
theorem smallest_positive_k : 
  ∃ (k : ℕ), (k = 2021) 
  ∧ ∀ x, function_f x ≤ (k : ℝ) * x := 
sorry

-- Prove f(x) has two distinct zeros m and n such that 2 < m + n < 16
theorem two_distinct_zeros : 
  ∃ m n : ℝ, (function_f m = 0) ∧ (function_f n = 0) ∧ (m ≠ n) ∧ (2 < m + n) ∧ (m + n < 16) := 
sorry

end math_problems

end tangent_line_eq_smallest_positive_k_two_distinct_zeros_l336_336298


namespace not_solution_B_l336_336942

theorem not_solution_B : ¬ (1 + 6 = 5) := by
  sorry

end not_solution_B_l336_336942


namespace total_volume_of_snowballs_l336_336749

-- Define the radii
def r1 : ℝ := 4
def r2 : ℝ := 6
def r3 : ℝ := 8

-- Define the volume formula for a sphere
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Calculate the volumes of individual spheres
def V1 : ℝ := volume_sphere r1
def V2 : ℝ := volume_sphere r2
def V3 : ℝ := volume_sphere r3

-- The total volume calculation proof statement
theorem total_volume_of_snowballs : 
  V1 + V2 + V3 = (3168 / 3) * Real.pi :=
by
  sorry

end total_volume_of_snowballs_l336_336749


namespace length_segment_AB_l336_336308

theorem length_segment_AB {x y : ℝ} (h_circle : x^2 + y^2 = 16) (h_line : √3 * x + y = 1) :
  ∃ A B, A ≠ B ∧ dist A B = 3 * √7 :=
sorry

end length_segment_AB_l336_336308


namespace squareable_natural_numbers_l336_336874

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def can_be_arranged (n : ℕ) (arrangement : list ℕ) : Prop :=
  (arrangement.length = n) ∧
  (∀ (k : ℕ), k < n → is_perfect_square ((arrangement.nth k).getD 0 + (k + 1)))

def is_squareable (n : ℕ) : Prop :=
  ∃ (arrangement : list ℕ), can_be_arranged n arrangement

theorem squareable_natural_numbers : (is_squareable 7 → False) ∧
                                     (is_squareable 9) ∧
                                     (is_squareable 11 → False) ∧
                                     (is_squareable 15) :=
by {
  sorry
}

end squareable_natural_numbers_l336_336874


namespace perimeter_inequality_l336_336404

noncomputable def k : ℝ := sorry
noncomputable def d : ℝ := sorry
noncomputable def λ : ℝ := sorry
noncomputable def k_λ : ℝ := sorry

axiom λ_bound : 1 / 2 ≤ λ ∧ λ ≤ 1

theorem perimeter_inequality (k d λ k_λ : ℝ) (hλ : 1 / 2 ≤ λ ∧ λ ≤ 1) :
  |2 * λ * d - (2 * λ - 1) * k| ≤ k_λ ∧ k_λ ≤ (2 * λ - 1) * k + 2 * (1 - λ) * d := sorry

end perimeter_inequality_l336_336404


namespace leak_empty_tank_l336_336916

theorem leak_empty_tank (
  (rate_inlet : ℝ) (filled_capacity : ℝ) (empty_time_with_inlet : ℝ)
  (filled_tank_time : ℝ)
  (net_empty_rate : ℝ)
  (full_empty_time_without_inlet : ℝ):

  rate_inlet = 3 ∧
  filled_capacity = 4320 ∧
  empty_time_with_inlet = 8 * 60 ∧
  net_empty_rate = filled_capacity / empty_time_with_inlet ∧

  full_empty_time_without_inlet = 
    filled_capacity / (net_empty_rate + rate_inlet) →

  full_empty_time_without_inlet / 60 = 6 :=
sorry

end leak_empty_tank_l336_336916


namespace compute_binom_12_6_eq_1848_l336_336980

def binomial (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

theorem compute_binom_12_6_eq_1848 : binomial 12 6 = 1848 :=
by
  sorry

end compute_binom_12_6_eq_1848_l336_336980


namespace circumradius_triangle_ABC_l336_336486

-- Definitions and assumptions
variables (A B C O1 O2 : Point)
variables (r1 r2 r3 : ℝ)
variables (d : ℝ := 13) -- Distance between the centers of the two spheres is 13
variables (sum_radii : ℝ := 7) -- Sum of the radii r1 + r2 is 7
variables (radii_third_sphere : ℝ := 5) -- Radius of the third sphere is 5

-- Conditions
variable (h1 : r1 + r2 = sum_radii)
variable (h2 : (O1 - O2).length = d)
variable (h3 : ∀ x, Sphere O1 r1 x ∧ Sphere O2 r2 x → Plane ABC x)
variable (h4 : Sphere B r3 C ∧ Sphere A r3 C)

-- Theorem to prove
theorem circumradius_triangle_ABC :
  circumradius_triangle A B C = Real.sqrt 30 :=
sorry

end circumradius_triangle_ABC_l336_336486


namespace find_w_l336_336025

-- Given conditions as Lean definitions
variables (j p t : ℝ) (w : ℝ)

-- j is 25% less than p
def condition1 : Prop := j = 0.75 * p

-- j is 20% less than t
def condition2 : Prop := j = 0.80 * t

-- t is w% less than p
def condition3 : Prop := t = p - (w / 100) * p

-- Prove that w = 6.25
theorem find_w : (condition1) → (condition2) → (condition3) → w = 6.25 := by
  intros h1 h2 h3
  sorry

end find_w_l336_336025


namespace angle_sum_ineq_l336_336766

def acute_triangle (A B C : Type) [PseudoMetricSpace A] (angle : A → A → A → ℝ) : Prop :=
  ∀ (α β γ : ℝ), angle A B C = α ∧ angle B C A = β ∧ angle C A B = γ → 
  0 < α ∧ α < 90 ∧ 0 < β ∧ β < 90 ∧ 0 < γ ∧ γ < 90

noncomputable def foot_of_altitude (A B C P : Type) [PseudoMetricSpace A] (angle : A → A → A → ℝ) : Prop :=
  ∃ (h : A → A → A → Prop), h A B P ∧ h A C P ∧ angle B A P = 90 ∧ angle C A P = 90

noncomputable def circumcenter (A B C O : Type) [PseudoMetricSpace A] (angle : A → A → A → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (X : A), X ≠ O → (dist X O = r ∧ angle A O B = 2 * angle A B C ∧ angle B O C = 2 * angle B C A)

noncomputable def angles_condition (A B C : Type) (angle_ACB angle_ABC : ℝ) : Prop :=
  angle_ACB ≥ angle_ABC + 30

theorem angle_sum_ineq (A B C P O : Type) [PseudoMetricSpace A] (angle : A → A → A → ℝ) :
  acute_triangle A B C angle → foot_of_altitude A B C P angle → circumcenter A B C O angle →
  angles_condition A B C (angle A C B) (angle A B C) →
  angle A B C + angle C O P < 90 :=
by sorry

end angle_sum_ineq_l336_336766


namespace school_C_paintings_l336_336051

theorem school_C_paintings
  (A B C : ℕ)
  (h1 : B + C = 41)
  (h2 : A + C = 38)
  (h3 : A + B = 43) : 
  C = 18 :=
by
  sorry

end school_C_paintings_l336_336051


namespace second_player_winning_strategy_l336_336482

theorem second_player_winning_strategy :
  ∃ (strategy : ℕ × ℕ → option (List (ℕ × ℕ))), 
  ∀ (moves : List (ℕ × ℕ × ℕ × ℕ)), 
    let A_moves := moves.filterMap (λ move, if move.1 = 1 then some (move.2, move.3) else none) in
    let B_moves := moves.filterMap (λ move, if move.1 = 2 then some (move.2, move.3) else none) in
    ∀ move ∈ B_moves,
    (∃ line, ∀ pos ∈ line, pos ∈ B_moves) :=
begin
  sorry
end

end second_player_winning_strategy_l336_336482


namespace scout_weekend_earnings_l336_336053

-- Define the constants and conditions
def base_pay : ℝ := 10.0
def tip_per_delivery : ℝ := 5.0
def saturday_hours : ℝ := 4.0
def sunday_hours : ℝ := 5.0
def saturday_deliveries : ℝ := 5.0
def sunday_deliveries : ℝ := 8.0

-- Calculate total hours worked
def total_hours : ℝ := saturday_hours + sunday_hours

-- Calculate base pay for the weekend
def total_base_pay : ℝ := total_hours * base_pay

-- Calculate total number of deliveries
def total_deliveries : ℝ := saturday_deliveries + sunday_deliveries

-- Calculate total earnings from tips
def total_tips : ℝ := total_deliveries * tip_per_delivery

-- Calculate total earnings
def total_earnings : ℝ := total_base_pay + total_tips

-- Theorem to prove the total earnings is $155.00
theorem scout_weekend_earnings : total_earnings = 155.0 := by
  sorry

end scout_weekend_earnings_l336_336053


namespace maximize_profit_l336_336546

noncomputable def profit_function (x : ℝ) : ℝ := -3 * x^2 + 252 * x - 4860

theorem maximize_profit :
  (∀ x : ℝ, 30 ≤ x ∧ x ≤ 54 → profit_function x ≤ 432) ∧ profit_function 42 = 432 := sorry

end maximize_profit_l336_336546


namespace Harriett_total_money_l336_336883

open Real

theorem Harriett_total_money :
    let quarters := 14 * 0.25
    let dimes := 7 * 0.10
    let nickels := 9 * 0.05
    let pennies := 13 * 0.01
    let half_dollars := 4 * 0.50
    quarters + dimes + nickels + pennies + half_dollars = 6.78 :=
by
    sorry

end Harriett_total_money_l336_336883


namespace sum_of_divisors_420_l336_336130

theorem sum_of_divisors_420 : ∑ i in (finset.range 421).filter (λ d, 420 % d = 0), i = 1344 :=
by
  sorry

end sum_of_divisors_420_l336_336130


namespace minimum_ab_condition_l336_336651

open Int

theorem minimum_ab_condition 
  (a b : ℕ) 
  (h_pos : 0 < a ∧ 0 < b)
  (h_div7_ab_sum : ab * (a + b) % 7 ≠ 0) 
  (h_div7_expansion : ((a + b) ^ 7 - a ^ 7 - b ^ 7) % 7 = 0) : 
  ab = 18 :=
sorry

end minimum_ab_condition_l336_336651


namespace smallest_nonprime_greater_than_with_no_prime_factors_less_than_15_l336_336018

-- Definitions for the given conditions.
def is_prime (p : ℕ) : Prop := (p > 1) ∧ ∀ d : ℕ, d ∣ p → (d = 1 ∨ d = p)

def has_no_prime_factors_less_than (n : ℕ) (m : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p ∣ n → p ≥ m

def is_nonprime (n : ℕ) : Prop := ¬ is_prime n

-- The main theorem based on the proof problem.
theorem smallest_nonprime_greater_than_with_no_prime_factors_less_than_15 
  (n : ℕ) (h1 : n > 1) (h2 : has_no_prime_factors_less_than n 15) (h3 : is_nonprime n) : 
  280 < n ∧ n ≤ 290 :=
by
  sorry

end smallest_nonprime_greater_than_with_no_prime_factors_less_than_15_l336_336018


namespace smallest_number_increased_by_3_divisible_l336_336881
open Nat

theorem smallest_number_increased_by_3_divisible (n : ℕ) :
  (∃ m, m = 3 ∧ ∃ k, k = lcm (lcm (lcm 9 35) 25) 21 ∧ n + m = k) → n = 1572 :=
by
  sorry

end smallest_number_increased_by_3_divisible_l336_336881


namespace riemann_sum_inequality_l336_336408

-- Definitions of the types
variable {α : Type*}

-- Let I be the least upper bound of the lower Riemann sums.
variable (I : α)

-- σ is an arbitrary Riemann sum.
variable (σ : α)

-- s and S are the lower and upper Riemann sums corresponding to the same partition as σ.
variable (s : α)
variable (S : α)

-- Assume the conditions for the Riemann sums.
variables [LinearOrder α] [HasAbs α]

-- State the theorem
theorem riemann_sum_inequality (h1 : ∀ a, s ≤ a → a ≤ S)
                               (h2 : ∀ b, s ≤ b ∧ b ≤ S → I ≤ b ∧ I ≤ S)
                               (h3 : s ≤ σ)
                               (h4 : σ ≤ S) :
  |σ - I| ≤ S - s :=
sorry

end riemann_sum_inequality_l336_336408


namespace circle_area_l336_336220

theorem circle_area (x y : ℝ) : 
  x^2 + y^2 - 18 * x + 8 * y = -72 → 
  ∃ r : ℝ, r = 5 ∧ π * r ^ 2 = 25 * π := 
by
  sorry

end circle_area_l336_336220


namespace parabola_equation_l336_336445

theorem parabola_equation 
  (p : ℝ × ℝ) (focus_y : ℝ) (symmetry_parallel_x : Prop) (vertex_on_y : Prop) 
  (h₁ : p = (2, -1))
  (h₂ : focus_y = -4)
  (h₃ : symmetry_parallel_x = true)
  (h₄ : vertex_on_y = true) 
  : ∃ a b c d e f : ℤ, a = 0 ∧ b = 0 ∧ c = 2 ∧ d = -9 ∧ e = 16 ∧ f = 32 ∧ 2 * (gcd.int.gcd (abs a) (gcd.int.gcd (abs b) (gcd.int.gcd (abs c) (gcd.int.gcd (abs d) (gcd.int.gcd (abs e) (abs f)))))) = 1 ∧ 
(ax + bxy + cy² + dx + ey + f = 0) :=
by {
  sorry
}

end parabola_equation_l336_336445


namespace cos_sum_diff_prod_l336_336234

variable {x y : ℝ}

theorem cos_sum_diff_prod :
  cos (x + y) - cos (x - y) = -2 * sin (x + y) * sin (x - y) :=
sorry

end cos_sum_diff_prod_l336_336234


namespace train_length_l336_336184

theorem train_length (speed_km_per_hr : ℝ) (time_seconds : ℝ)
  (speed_conversion : speed_km_per_hr / 3.6 = 25) (time_condition : time_seconds = 12) :
  (speed_conversion * time_condition = 300) :=
by
  /- The proof goes here. -/
  sorry

end train_length_l336_336184


namespace methanol_volume_l336_336933

theorem methanol_volume (mass : ℝ) (density : ℝ) (volume : ℝ) : mass = 30.0 → density = 0.7914 → volume = mass / density → volume = 37.9 :=
by
  intros h_mass h_density h_volume
  rw [h_mass, h_density] at h_volume
  exact h_volume

end methanol_volume_l336_336933


namespace gcd_f100_f101_l336_336764

-- Define the function f
def f (x : ℤ) : ℤ := x^2 - x + 2010

-- A statement asserting the greatest common divisor of f(100) and f(101) is 10
theorem gcd_f100_f101 : Int.gcd (f 100) (f 101) = 10 := by
  sorry

end gcd_f100_f101_l336_336764


namespace surface_area_of_glued_cubes_l336_336107

theorem surface_area_of_glued_cubes (edge_len_large_cube: ℝ) (h1: edge_len_large_cube = 3):
    let edge_len_small_cube := edge_len_large_cube / 3 in
    let SA_large_cube := 6 * (edge_len_large_cube ^ 2) in
    let SA_small_cube := 6 * (edge_len_small_cube ^ 2) in
    let exposed_SA_large_cube := SA_large_cube - (edge_len_large_cube ^ 2) in
    let contributing_SA_small_cube := 4 * (edge_len_small_cube ^ 2) in
    (exposed_SA_large_cube + contributing_SA_small_cube) = 49 :=
by
  sorry

end surface_area_of_glued_cubes_l336_336107


namespace maximize_profit_l336_336160

variable {k : ℝ} (hk : k > 0)
variable {x : ℝ} (hx : 0 < x ∧ x < 0.06)

def deposit_volume (x : ℝ) : ℝ := k * x
def interest_paid (x : ℝ) : ℝ := k * x ^ 2
def profit (x : ℝ) : ℝ := (0.06 * k^2 * x) - (k * x^2)

theorem maximize_profit : 0.03 = x :=
by
  sorry

end maximize_profit_l336_336160


namespace parallel_planes_suff_but_not_nec_l336_336013

variables (α β : Plane) (m n : Line)
variables (m_in_alpha : m ∈ α) (n_in_alpha : n ∈ α)
variables (distinct_planes : α ≠ β)

theorem parallel_planes_suff_but_not_nec : (α ∥ β) → ((m ∥ β) ∧ (n ∥ β)) ∧ (¬ ((m ∥ β) ∧ (n ∥ β) → α ∥ β)) := 
sorry

end parallel_planes_suff_but_not_nec_l336_336013


namespace min_length_AD_eq_sqrt3_div3_l336_336654

noncomputable def minimum_length_AD (a b c : ℝ) : ℝ :=
   min (λ AD, 4 * AD^2 ≥ 2 * (b^2 + c^2) - a^2)

theorem min_length_AD_eq_sqrt3_div3 
  (a b c AD : ℝ)
  (h1 : a = 2)
  (h2 : (b + 2) * (Real.sin (C / 2)) * (Real.sin ((A - B) / 2)) = c * (Real.cos (A / 2)) * (Real.cos ((B - C) / 2))) :
  minimum_length_AD a b c = Real.sqrt 3 / 3 :=
begin
  sorry
end

end min_length_AD_eq_sqrt3_div3_l336_336654


namespace smallest_n_l336_336125

theorem smallest_n (n : ℕ) (h1 : ∃ k1, 5 * n = k1^2) (h2 : ∃ k2, 7 * n = k2^3) : n = 245 :=
sorry

end smallest_n_l336_336125


namespace scout_weekend_earnings_l336_336052

-- Define the constants and conditions
def base_pay : ℝ := 10.0
def tip_per_delivery : ℝ := 5.0
def saturday_hours : ℝ := 4.0
def sunday_hours : ℝ := 5.0
def saturday_deliveries : ℝ := 5.0
def sunday_deliveries : ℝ := 8.0

-- Calculate total hours worked
def total_hours : ℝ := saturday_hours + sunday_hours

-- Calculate base pay for the weekend
def total_base_pay : ℝ := total_hours * base_pay

-- Calculate total number of deliveries
def total_deliveries : ℝ := saturday_deliveries + sunday_deliveries

-- Calculate total earnings from tips
def total_tips : ℝ := total_deliveries * tip_per_delivery

-- Calculate total earnings
def total_earnings : ℝ := total_base_pay + total_tips

-- Theorem to prove the total earnings is $155.00
theorem scout_weekend_earnings : total_earnings = 155.0 := by
  sorry

end scout_weekend_earnings_l336_336052


namespace sum_in_base_8_l336_336605

theorem sum_in_base_8 (a b : ℕ) (h_a : a = 3 * 8^2 + 2 * 8 + 7)
                                  (h_b : b = 7 * 8 + 3) :
  (a + b) = 4 * 8^2 + 2 * 8 + 2 :=
by
  sorry

end sum_in_base_8_l336_336605


namespace find_a_perpendicular_tangents_l336_336664

theorem find_a_perpendicular_tangents (a : ℝ) :
  (let C1 : ℝ → ℝ := λ x, a * x^3 - x^2 + 2 * x,
       C2 : ℝ → ℝ := λ x, Real.exp x,
       deriv_C1 := λ x, 3 * a * x^2 - 2 * x + 2,
       deriv_C2 := λ x, Real.exp x in
   deriv_C1 1 * deriv_C2 1 = -1)
  → a = -1 / (3 * Real.exp 1) := 
by 
  sorry

end find_a_perpendicular_tangents_l336_336664


namespace compute_binom_12_6_eq_1848_l336_336983

def binomial (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

theorem compute_binom_12_6_eq_1848 : binomial 12 6 = 1848 :=
by
  sorry

end compute_binom_12_6_eq_1848_l336_336983


namespace simplify_expression_l336_336400

variable {a b c : ℝ}
variable (h_a_nonzero : a ≠ 0) (h_b_nonzero : b ≠ 0) (h_c_nonzero : c ≠ 0)

def x : ℝ := b / c + c / b
def y : ℝ := (a^2) / (c^2) + (c^2) / (a^2)
def z : ℝ := a / b + b / a

theorem simplify_expression : 
  x^2 + y^2 + z^2 - x*y*z = 6 :=
sorry

end simplify_expression_l336_336400


namespace prove_inequalities_l336_336259

noncomputable def x := Real.log Real.pi
noncomputable def y := Real.logb 5 2
noncomputable def z := Real.exp (-1 / 2)

theorem prove_inequalities : y < z ∧ z < x := by
  unfold x y z
  sorry

end prove_inequalities_l336_336259


namespace problem1_problem2_l336_336521

open Real -- Open the Real namespace for trigonometric functions

-- Part 1: Prove cos(5π + α) * tan(α - 7π) = 4/5 given π < α < 2π and cos α = 3/5
theorem problem1 (α : ℝ) (hα1 : π < α) (hα2 : α < 2 * π) (hcos : cos α = 3 / 5) : 
  cos (5 * π + α) * tan (α - 7 * π) = 4 / 5 := sorry

-- Part 2: Prove sin(π/3 + α) = √3/3 given cos (π/6 - α) = √3/3
theorem problem2 (α : ℝ) (hcos : cos (π / 6 - α) = sqrt 3 / 3) : 
  sin (π / 3 + α) = sqrt 3 / 3 := sorry

end problem1_problem2_l336_336521


namespace six_player_round_robin_matches_l336_336208

theorem six_player_round_robin_matches : 
  ∀ (n : ℕ), n = 6 → ((n * (n - 1)) / 2) = 15 := by 
  intros n hn 
  rw hn 
  -- now we should have (6 * 5) / 2 = 15, but we will leave this to sorry
  sorry

end six_player_round_robin_matches_l336_336208


namespace proof_problem_l336_336834

variables {A B C a b c : ℝ}

noncomputable def triangle_condition_1 : Prop :=
  2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c

noncomputable def triangle_condition_2 : Prop :=
  c = Real.sqrt 7

noncomputable def area_triangle (a b C : ℝ) : ℝ :=
  0.5 * a * b * Real.sin C

noncomputable def area_condition : Prop :=
  area_triangle a b C = (3 * Real.sqrt 3) / 2

theorem proof_problem
  (h1 : triangle_condition_1)
  (h2 : triangle_condition_2)
  (h3 : area_condition) :
  C = Real.pi / 3 ∧ (a + b + c = 5 + Real.sqrt 7) :=
  sorry

end proof_problem_l336_336834


namespace box_office_revenue_growth_l336_336538

variable (x y : ℝ)

-- Assuming the given conditions
def revenue_2012 := 21
def annual_growth_rate := x
def revenue_2016 := y

theorem box_office_revenue_growth:
  revenue_2016 = revenue_2012 * (1 + annual_growth_rate) ^ 4 -> y = 21 * (1 + x) ^ 4 :=
by
  intro h
  rw [revenue_2012, revenue_2016] at h
  exact h

end box_office_revenue_growth_l336_336538


namespace minimum_sum_area_l336_336352

noncomputable def minimum_area_sum (y1 y2 : ℝ) : ℝ :=
  let OF := 1 / 2
  S_OFA := (1 / 2) * OF * abs y1
  S_OFB := (1 / 2) * OF * abs y2
  S_OFA + S_OFB

theorem minimum_sum_area : ∀ (y1 y2 : ℝ),
  (y1 * y2 = -2) → 
  (minimum_area_sum y1 y2 = sqrt 2 / 2) := 
by
  sorry

end minimum_sum_area_l336_336352


namespace triangle_angle_determinant_l336_336756

theorem triangle_angle_determinant (A B C : ℝ) 
  (h : A + B + C = π) : 
  det ![
    ![sin (2 * A), cot A, 1],
    ![sin (2 * B), cot B, 1],
    ![sin (2 * C), cot C, 1]
  ] = 0 :=
by sorry

end triangle_angle_determinant_l336_336756


namespace find_radius_inscribed_circle_l336_336185

noncomputable def radius_of_inscribed_circle (a b c : ℝ) (h : a^2 + b^2 = c^2) : ℝ :=
  let area := (1/2) * a * b
  let s := (a + b + c) / 2
  area / s

theorem find_radius_inscribed_circle : radius_of_inscribed_circle 8 15 17 (by 
  norm_num : 8^2 + 15^2 = 17^2) = 3 := by 
  sorry

end find_radius_inscribed_circle_l336_336185


namespace scout_weekend_earnings_l336_336060

theorem scout_weekend_earnings : 
  let base_pay_per_hour := 10.00
  let tip_per_customer := 5.00
  let saturday_hours := 4
  let saturday_customers := 5
  let sunday_hours := 5
  let sunday_customers := 8
  in
  (saturday_hours * base_pay_per_hour + saturday_customers * tip_per_customer) +
  (sunday_hours * base_pay_per_hour + sunday_customers * tip_per_customer) = 155.00 := sorry

end scout_weekend_earnings_l336_336060


namespace find_m_l336_336315

open Real

-- Definitions based on problem conditions
def vector_a (m : ℝ) : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)

-- The dot product
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Prove the final statement using given conditions
theorem find_m (m : ℝ) (h1 : dot_product (vector_a m) vector_b + dot_product vector_b vector_b = 0) :
  m = 8 :=
sorry

end find_m_l336_336315


namespace multiplication_vs_subtraction_difference_l336_336138

variable (x : ℕ)
variable (h : x = 10)

theorem multiplication_vs_subtraction_difference :
  3 * x - (26 - x) = 14 := by
  sorry

end multiplication_vs_subtraction_difference_l336_336138


namespace inequality_solution_l336_336437

theorem inequality_solution (x : ℝ) :
  (6 * x^2 + 12 * x - 35) / ((x - 2) * (3 * x + 6)) < 2 ↔ x ∈ set.Ioo (-2 : ℝ) (11 / 18 : ℝ) ∪ set.Ioi (2 : ℝ) :=
by
  sorry

end inequality_solution_l336_336437


namespace tangent_line_at_x0_l336_336828

def f (x : ℝ) : ℝ := Real.sin x + Real.exp x + 2
def f' (x : ℝ) : ℝ := Real.cos x + Real.exp x

theorem tangent_line_at_x0 :
    let x0 := (0 : ℝ)
    let y0 := f x0
    ∃ k b, k = f' x0 ∧ b = y0 ∧ ∀ x, (y0 + k * (x - x0) = 2 * x + 3) :=
begin
  sorry
end

end tangent_line_at_x0_l336_336828


namespace angle_BXD_is_150_l336_336731

open EuclideanGeometry

variables {A B C D E F X Y : Point}
variable (P : Line)
variable (l1 : Line)
variable (l2 : Line)

-- Definition for parallel lines
def parallel (l1 l2 : Line) : Prop := ∀ (P Q : Point), P ∈ l1 → Q ∈ l1 → P ≠ Q → P ∈ l2 → Q ∈ l2 → P = Q

-- Angles
variable {α β γ δ : Real}

-- Conditions
axiom parallel_AB_CD : parallel (line_through A B) (line_through C D)
axiom angle_AXE_eq_4CYX_minus_90 : ∠ A X E = 4 * ∠ C Y X - 90

-- Question
theorem angle_BXD_is_150 :
  ∠ B X D = 150 := 
sorry

end angle_BXD_is_150_l336_336731


namespace m_value_if_g_is_odd_set_of_all_upper_bounds_range_of_a_l336_336671

noncomputable def f (x a : ℝ) : ℝ := 1 + a * exp (-x) + exp (-2 * x)
noncomputable def g (x m : ℝ) : ℝ := Real.log ((x + 1) / (m * x - 1)) / Real.log (1 / 2)

theorem m_value_if_g_is_odd (m : ℝ) (h₁ : ∀ x : ℝ, g (-x) m = -g x m) : m = 1 := 
sorry

theorem set_of_all_upper_bounds (h₂ : ∀ x : ℝ, g (-x) 1 = -g x 1) : 
  { y : ℝ | ∀ x ∈ (Set.Icc (9 / 7 : ℝ) 3), g x 1 ≤ y } = [ 3, + ∞ ) :=
sorry

theorem range_of_a (a : ℝ) 
  (h₃ : ∀ x : ℝ, 0 ≤ x → f x a ≤ 3) : 
    -5 ≤ a ∧ a ≤ 1 :=
sorry

end m_value_if_g_is_odd_set_of_all_upper_bounds_range_of_a_l336_336671


namespace trapezoid_distances_l336_336838

-- Define the problem parameters
variables (AB CD AD BC : ℝ)
-- Assume given conditions
axiom h1 : AD > BC
noncomputable def k := AD / BC

-- Formalizing the proof problem in Lean 4
theorem trapezoid_distances (M : Type) (BM AM CM DM : ℝ) :
  BM = AB * BC / (AD - BC) →
  AM = AB * AD / (AD - BC) →
  CM = CD * BC / (AD - BC) →
  DM = CD * AD / (AD - BC) →
  true :=
sorry

end trapezoid_distances_l336_336838


namespace cos_sum_diff_prod_l336_336235

variable {x y : ℝ}

theorem cos_sum_diff_prod :
  cos (x + y) - cos (x - y) = -2 * sin (x + y) * sin (x - y) :=
sorry

end cos_sum_diff_prod_l336_336235


namespace lines_through_point_l336_336502

theorem lines_through_point (k : ℝ) : ∀ x y : ℝ, (y = k * (x - 1)) ↔ (x = 1 ∧ y = 0) ∨ (x ≠ 1 ∧ y / (x - 1) = k) :=
by
  sorry

end lines_through_point_l336_336502


namespace quadratic_two_distinct_roots_l336_336686

variables {a b c d e : ℝ}
hypothesis h1 : a > b
hypothesis h2 : b > c
hypothesis h3 : c > d
hypothesis h4 : e ≠ -1

theorem quadratic_two_distinct_roots (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : e ≠ -1) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (e+1) * x₁^2 - (a+c+b*e+d*e) * x₁ + (a*c + e*b*d) = 0 ∧ (e+1) * x₂^2 - (a+c+b*e+d*e) * x₂ + (a*c + e*b*d) = 0 := 
sorry

end quadratic_two_distinct_roots_l336_336686


namespace correct_option_d_l336_336506

theorem correct_option_d (a : ℝ) :
  (∀ x ∈ set.Icc 0 1, a ≥ real.exp x) ∧ (∃ x : ℝ, x^2 + 4*x + a ≤ 0) ↔ false :=
  (a ∈ set.Ioo (-∞) (real.exp 1) ∪ set.Ioi 4) :=
begin
  sorry
end

end correct_option_d_l336_336506


namespace faculty_married_percentage_l336_336557

theorem faculty_married_percentage (N : ℕ) (hN : N > 0) :
  let W := 0.6 * N in
  let M := 0.4 * N in
  let S := 0.75 * M in
  let M_married := 0.25 * M in
  (M_married / N) * 100 = 10 :=
by
  have h1 : M = 0.4 * N := sorry
  have h2 : S = 0.75 * M := sorry
  have h3 : M_married = 0.25 * M := sorry
  have h4 : (M_married / N) * 100 = 10 := sorry
  exact h4

end faculty_married_percentage_l336_336557


namespace sector_area_l336_336439

theorem sector_area (r : ℝ) (θ : ℝ) (h_r : r = 12) (h_θ : θ = 40) : (θ / 360) * π * r^2 = 16 * π :=
by
  rw [h_r, h_θ]
  sorry

end sector_area_l336_336439


namespace circle_area_l336_336070

theorem circle_area (x y : ℝ) :
  x^2 + y^2 - 14 * x + 10 * y + 65 = 0 → 
  ∃ r : ℝ, r = 3 ∧ ∃ A : ℝ, A = π * r^2 :=
begin
  intro h,
  -- Use the given condition to rewrite in standard form (this part is implicit in the setup)
  have h1 : (x - 7)^2 + (y + 5)^2 = 9, sorry,
  -- Set r to 3 because (x - 7)^2 + (y + 5)^2 = 3^2
  use 3,
  split,
  refl,
  -- Calculate the area
  use π * 3^2,
  refl,
end

end circle_area_l336_336070


namespace term_in_AP_is_zero_l336_336351

theorem term_in_AP_is_zero (a d : ℤ) 
  (h : (a + 4 * d) + (a + 20 * d) = (a + 7 * d) + (a + 14 * d) + (a + 12 * d)) :
  a + (-9) * d = 0 :=
by
  sorry

end term_in_AP_is_zero_l336_336351


namespace arithmetic_sequence_properties_l336_336270

theorem arithmetic_sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2)
  (h2 : d ≠ 0)
  (h3 : ∀ n, S n ≤ S 8) :
  d < 0 ∧ S 17 ≤ 0 := 
sorry

end arithmetic_sequence_properties_l336_336270


namespace scout_weekend_earnings_l336_336057

theorem scout_weekend_earnings
  (base_pay_per_hour : ℕ)
  (tip_per_delivery : ℕ)
  (hours_worked_saturday : ℕ)
  (deliveries_saturday : ℕ)
  (hours_worked_sunday : ℕ)
  (deliveries_sunday : ℕ)
  (total_earnings : ℕ)
  (h_base_pay : base_pay_per_hour = 10)
  (h_tip : tip_per_delivery = 5)
  (h_hours_sat : hours_worked_saturday = 4)
  (h_deliveries_sat : deliveries_saturday = 5)
  (h_hours_sun : hours_worked_sunday = 5)
  (h_deliveries_sun : deliveries_sunday = 8) :
  total_earnings = 155 :=
by
  sorry

end scout_weekend_earnings_l336_336057


namespace min_sum_of_factors_l336_336848

theorem min_sum_of_factors (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 3960) : 
  a + b + c = 72 :=
sorry

end min_sum_of_factors_l336_336848


namespace compute_binom_12_6_eq_1848_l336_336982

def binomial (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

theorem compute_binom_12_6_eq_1848 : binomial 12 6 = 1848 :=
by
  sorry

end compute_binom_12_6_eq_1848_l336_336982


namespace black_squares_count_l336_336049

def checkerboard_size : Nat := 32
def total_squares : Nat := checkerboard_size * checkerboard_size
def black_squares (n : Nat) : Nat := n / 2

theorem black_squares_count : black_squares total_squares = 512 := by
  let n := total_squares
  show black_squares n = 512
  sorry

end black_squares_count_l336_336049


namespace kendra_shirts_for_two_weeks_l336_336382

def school_days := 5
def after_school_club_days := 3
def one_week_shirts := school_days + after_school_club_days + 1 (Saturday) + 2 (Sunday)
def two_weeks_shirts := one_week_shirts * 2

theorem kendra_shirts_for_two_weeks : two_weeks_shirts = 22 :=
by
  -- Prove the theorem
  sorry

end kendra_shirts_for_two_weeks_l336_336382


namespace percent_not_crust_l336_336924

-- Definitions as conditions
def pie_total_weight : ℕ := 200
def crust_weight : ℕ := 50

-- The theorem to be proven
theorem percent_not_crust : (pie_total_weight - crust_weight) / pie_total_weight * 100 = 75 := 
by
  sorry

end percent_not_crust_l336_336924


namespace parabola_no_intersect_l336_336709

theorem parabola_no_intersect (m : ℝ) : 
  (¬ ∃ x : ℝ, -x^2 - 6*x + m = 0 ) ↔ m < -9 :=
by
  sorry

end parabola_no_intersect_l336_336709


namespace find_curve_F_l336_336287

variables {a b : ℝ}

/-- Matrix A -/
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![a, 2], ![7, 3]]

/-- Inverse of Matrix A -/
def A_inv : Matrix (Fin 2) (Fin 2) ℝ := ![![b, -2], ![-7, a]]

/-- Transformation of Curve F into y = 2x under transformation A -/
def transforms_curve (F : ℝ → ℝ → Prop) :=
  ∀ x y x' y', F x y → (A.mul_vec ![x, y] = ![x', y']) → (y' = 2 * x')

/-- Curve F -/
def curve_F (x y : ℝ) : Prop := y = -3 * x

theorem find_curve_F (hA_inv : A ⬝ A_inv = 1) (h_transform : transforms_curve curve_F) : 
  ∀ x y, curve_F x y → ∃ y', y' = 2 * x → y = -3 * x := 
sorry

end find_curve_F_l336_336287


namespace hexagon_center_distance_l336_336081

theorem hexagon_center_distance (side_length : ℝ) (h : side_length = 4) : 
  let a := side_length in 
  let h_eq_tri_height := (sqrt 3 / 2) * a 
  in h_eq_tri_height = 2 * sqrt 3 :=
by
  simp [h]
  apply sorry

end hexagon_center_distance_l336_336081


namespace value_of_g_at_x_minus_5_l336_336331

-- Definition of the function g
def g (x : ℝ) : ℝ := -3

-- The theorem we need to prove
theorem value_of_g_at_x_minus_5 (x : ℝ) : g (x - 5) = -3 := by
  sorry

end value_of_g_at_x_minus_5_l336_336331


namespace decagon_intersection_points_l336_336193

-- Define what a regular decagon is
def regular_decagon : Type := fin 10 → ℝ × ℝ

-- Function to calculate the total number of diagonals in a decagon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Function calculating the number of intersection points
def num_intersection_points (n : ℕ) : ℕ := nat.choose n 4

-- Theorem statement (what we need to prove)
theorem decagon_intersection_points (d : regular_decagon) :
  num_intersection_points 10 = 210 := sorry

end decagon_intersection_points_l336_336193


namespace cos_angle_between_planes_l336_336761

def normal_vector_plane1 : ℝ × ℝ × ℝ := (2, 1, -2)
def normal_vector_plane2 : ℝ × ℝ × ℝ := (6, 3, 2)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

def cos_theta (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
dot_product v1 v2 / (magnitude v1 * magnitude v2)

theorem cos_angle_between_planes : cos_theta normal_vector_plane1 normal_vector_plane2 = 11 / 21 :=
by sorry

end cos_angle_between_planes_l336_336761


namespace proof_target_l336_336843

noncomputable def parametric_eq_line (t : ℝ) := (1 + t / 2, (Real.sqrt 3 / 2) * t)

def polar_eq_curve (θ : ℝ) (ρ : ℝ) := (1 + Real.sin θ ^ 2) * ρ^2 = 2

def cartesian_eq_curve (x y : ℝ) := (x^2 / 2 + y^2 = 1)

noncomputable def intersection_params := { t : ℝ // (2 * (1 + t / 2)^2 + 3 * (Real.sqrt 3 / 2 * t)^2 = 2) }

def point_P := (1, 0)

def ap_squared (t : ℝ) := (1 + t / 2 - 1)^2 + (Real.sqrt 3 / 2 * t) ^ 2

def result (t1 t2 : ℝ) := (1 / ap_squared t1 + 1 / ap_squared t2)

theorem proof_target (t1 t2 : intersection_params) :
  (t1.val + t2.val = -4 / 7) ∧ (t1.val * t2.val = -4 / 7) →
  result t1.val t2.val = 9 / 2 :=
by
  sorry

end proof_target_l336_336843


namespace problem1_problem2_l336_336296

def f (x : ℝ) : ℝ := |x - 2| + |x + 1|

-- First proof problem
theorem problem1 (x : ℝ) : f(x) > 5 → x < -2 ∨ x > 3 := by
  sorry

-- Second proof problem
theorem problem2 (a : ℝ) (h : ∀ x : ℝ, f(x) ≥ (Real.log a / Real.log 2) ^ 2 - Real.log a / Real.log (Real.sqrt 2)) : 1/2 ≤ a ∧ a ≤ 8 := by
  sorry

end problem1_problem2_l336_336296


namespace max_alpha_for_2_alpha_divides_3n_plus_1_l336_336021

theorem max_alpha_for_2_alpha_divides_3n_plus_1 (n : ℕ) (hn : n > 0) : ∃ α : ℕ, (2 ^ α ∣ (3 ^ n + 1)) ∧ ¬ (2 ^ (α + 1) ∣ (3 ^ n + 1)) ∧ α = 1 :=
by
  sorry

end max_alpha_for_2_alpha_divides_3n_plus_1_l336_336021


namespace relationship_among_a_b_c_l336_336276

noncomputable def f : ℝ → ℝ := sorry -- Assume f is defined somewhere.

theorem relationship_among_a_b_c
  (h_odd : ∀ x : ℝ, f(-x) = -f(x))
  (h_decreasing : ∀ x y : ℝ, 0 ≤ x → x < y → f(y) < f(x)) :
  let a := -f(Real.log (1 / 2))
  let b := f(Real.log ((1 / Real.exp 1) - (1 / Real.exp 2)))
  let c := f(Real.exp 0.1)
  in c < a ∧ a < b :=
by
  sorry

end relationship_among_a_b_c_l336_336276


namespace percent_receiving_speeding_tickets_l336_336032

theorem percent_receiving_speeding_tickets
  (total_motorists : ℕ)
  (percent_exceeding_limit percent_exceeding_limit_without_ticket : ℚ)
  (h_exceeding_limit : percent_exceeding_limit = 0.5)
  (h_exceeding_limit_without_ticket : percent_exceeding_limit_without_ticket = 0.2) :
  let exceeding_limit := percent_exceeding_limit * total_motorists
  let without_tickets := percent_exceeding_limit_without_ticket * exceeding_limit
  let with_tickets := exceeding_limit - without_tickets
  (with_tickets / total_motorists) * 100 = 40 :=
by
  sorry

end percent_receiving_speeding_tickets_l336_336032


namespace min_value_expression_l336_336241

theorem min_value_expression (y : ℝ) (hy : 0 < y) : 
  ∃ (m : ℝ), (∀ (z : ℝ), (0 < z) → 9 * z^6 + 8 * z^(-3) ≥ m) ∧ m = 17 :=
by
  existsi 17
  split
  { intros z hz
    sorry }
  { refl }

end min_value_expression_l336_336241


namespace binomial_coeff_terms_l336_336639

theorem binomial_coeff_terms (m : ℝ) (n : ℕ) (h1 : m ≠ 0) (h2 : n ≥ 2) (h3 : ∃ k, n = 10) (h4 : ∃ c, c = 9) :
  (n = 10 ∧ m = 2) ∧ 
  (∃ a : ℕ → ℝ, (∀ k : ℕ, a k = (∑ i in Finset.range k, (-1) ^ i * (1 - 9 * 2) ^ 10)) 
  → (a 0 - a 1 + a 2 - a 3 + ... + (-1)^n * a n) % 6 = 1) :=
sorry

end binomial_coeff_terms_l336_336639


namespace carpet_length_l336_336530

-- Define the conditions as hypotheses
def width_of_carpet : ℝ := 4
def area_of_living_room : ℝ := 60

-- Formalize the corresponding proof problem
theorem carpet_length (h : 60 = width_of_carpet * length) : length = 15 :=
sorry

end carpet_length_l336_336530


namespace maximum_number_of_cars_l336_336858

def maxCars (wheels : ℕ) (x : ℕ) (y : ℕ) : ℕ :=
  if 7 * x + 4 * y = wheels then y else 0

theorem maximum_number_of_cars :
  ∃ (x y : ℕ), maxCars 115 x y = 27 := by
  exists 1 27
  simp [maxCars]
  sorry

end maximum_number_of_cars_l336_336858


namespace bill_total_amount_after_ninth_week_l336_336356

def bill_piggy_bank_end_amount (initial_amount: ℝ) (weekly_addition: ℝ) (weeks: ℕ) : ℝ :=
  initial_amount + weekly_addition * (weeks - 1)

theorem bill_total_amount_after_ninth_week (x: ℝ) (h1: x = 8) (h2: weekly_addition = 2) (h3: weeks = 9) (h4: bill_piggy_bank_end_amount x weekly_addition weeks = 3 * x):
  bill_piggy_bank_end_amount x 2 9 = 24 :=
by
  rw [bill_piggy_bank_end_amount, h1, h2, h3]
  simp
  sorry

end bill_total_amount_after_ninth_week_l336_336356


namespace find_f_1993_l336_336757

noncomputable def f (n : ℕ) : ℕ := sorry

theorem find_f_1993 :
  (∀ n : ℕ, f (f n) + f n = 2 * n + 3) →
  f 1993 = 1994 :=
begin
  intro h,
  sorry
end

end find_f_1993_l336_336757


namespace passing_time_for_platform_l336_336158

def train_length : ℕ := 1100
def time_to_cross_tree : ℕ := 110
def platform_length : ℕ := 700
def speed := train_length / time_to_cross_tree
def combined_length := train_length + platform_length

theorem passing_time_for_platform : 
  let speed := train_length / time_to_cross_tree
  let combined_length := train_length + platform_length
  combined_length / speed = 180 :=
by
  sorry

end passing_time_for_platform_l336_336158


namespace dodecagon_diagonals_l336_336562

theorem dodecagon_diagonals : 
  let n : ℕ := 12 in 
  n*(n-3)/2 = 54 :=
by
  sorry

end dodecagon_diagonals_l336_336562


namespace range_of_a_l336_336299

theorem range_of_a (a : ℝ) (h1 : a < 0) (h2 : ∃ x₀ ∈ set.Icc (-2 : ℝ) 2, (a * x₀ + 2) < 0) : 
  a < -1 :=
by
  sorry

end range_of_a_l336_336299


namespace digit_number_is_203_l336_336177

theorem digit_number_is_203 {A B C : ℕ} (h1 : A + B + C = 10) (h2 : B = A + C) (h3 : 100 * C + 10 * B + A = 100 * A + 10 * B + C + 99) :
  100 * A + 10 * B + C = 203 :=
by
  sorry

end digit_number_is_203_l336_336177


namespace sum_divisors_420_l336_336132

theorem sum_divisors_420 : 
  ∃ d, d = 1344 ∧ 
  (∀ n, n ∣ 420 → n ∈ ℕ ∧ n > 0) ∧
  (420 = 2^2 * 3 * 5 * 7) →
  ∑ (n : ℕ) in (finset.filter (λ d, d ∣ 420) (finset.range (421))), n = 1344 :=
by
  sorry

end sum_divisors_420_l336_336132


namespace part_I_part_II_l336_336297

def f (x a : ℝ) : ℝ := |x - a| - (1/2) * x

-- Part I
theorem part_I (x : ℝ) (hx : 2 < x ∧ x < 6) : f x 3 < 0 := sorry

-- Part II
theorem part_II (a : ℝ) (h : ∀ x : ℝ, f x a - f (x + a) a < a^2 + (a / 2)) : a > 1 := sorry

end part_I_part_II_l336_336297


namespace num_girls_respond_yes_l336_336946

-- Define the number of total girls, girls dressed in blue, and girls dressed in red.
def total_girls := 18
def blue_girls := 11
def red_girls := total_girls - blue_girls

-- Define the conditions
-- Condition: A girl tells the truth if both her neighbors are the same color
-- A function to determine if a girl says "yes" based on the color of her left neighbor
def responds_yes (left_neigh_col : Bool) : Bool :=
  left_neigh_col = true -- true if left neighbor is blue, false if left neighbor is red

-- Proof: Number of girls who respond "yes" is equal to the number of girls dressed in blue
theorem num_girls_respond_yes : 
  (∀ i, responds_yes (true) -> i < total_girls → 11) :=
sorry

end num_girls_respond_yes_l336_336946


namespace ratio_of_areas_l336_336364

-- We define the conditions

variables {α γ : ℝ} -- Angles A and C
variables {R : ℝ} -- Radius of the inscribed circle
variables {P Q M N : ℝ} -- We use the fact the ratios involve distances equivalent to angles dependent on α and γ

-- The goal is to prove the ratio of areas of triangles MNP and MQN

theorem ratio_of_areas (hα : 0 ≤ α ∧ α ≤ 2 * π) (hγ : 0 ≤ γ ∧ γ ≤ 2 * π) : 
  let PK := R * cos(γ / 2),
      QF := R * cos(α / 2) in
  ∀ (R : ℝ), R > 0 -> (PK / QF) = (cos (γ / 2)) / (cos (α / 2)) :=
by
  sorry

end ratio_of_areas_l336_336364


namespace area_of_triangle_formed_by_tangent_line_l336_336071

theorem area_of_triangle_formed_by_tangent_line : 
  let y := λ x : ℝ, x^3 in
  let tangent_line := 3 * (x - 1) + 1 in
  let x_intercept := (2 / 3 : ℝ) in
  let area := (1 / 2 : ℝ) * (2 - x_intercept) * 4 in
  area = (8 / 3 : ℝ) :=
by
  sorry

end area_of_triangle_formed_by_tangent_line_l336_336071


namespace colorings_equivalence_l336_336357

-- Define the problem setup
structure ProblemSetup where
  n : ℕ  -- Number of disks (8)
  blue : ℕ  -- Number of blue disks (3)
  red : ℕ  -- Number of red disks (3)
  green : ℕ  -- Number of green disks (2)
  rotations : ℕ  -- Number of rotations (4: 90°, 180°, 270°, 360°)
  reflections : ℕ  -- Number of reflections (8: 4 through vertices and 4 through midpoints)

def number_of_colorings (setup : ProblemSetup) : ℕ :=
  sorry -- This represents the complex implementation details

def correct_answer : ℕ := 43

theorem colorings_equivalence : ∀ (setup : ProblemSetup),
  setup.n = 8 → setup.blue = 3 → setup.red = 3 → setup.green = 2 → setup.rotations = 4 → setup.reflections = 8 →
  number_of_colorings setup = correct_answer :=
by
  intros setup h1 h2 h3 h4 h5 h6
  sorry

end colorings_equivalence_l336_336357


namespace num_non_empty_sets_l336_336457

theorem num_non_empty_sets : 
  let A := {1, 2, 3, 4, 5}
  in (finset.filter (λ S : finset ℕ, S ⊆ A ∧ ∀ a ∈ S, 6 - a ∈ S) (finset.powerset A)).card - 1 = 7 :=
by sorry

end num_non_empty_sets_l336_336457


namespace binomial_12_6_eq_1848_l336_336970

theorem binomial_12_6_eq_1848 : (Nat.choose 12 6) = 1848 :=
  sorry

end binomial_12_6_eq_1848_l336_336970


namespace books_per_bookshelf_l336_336948

theorem books_per_bookshelf (total_books : ℕ) (bookshelves : ℕ) (H_total_books : total_books = 42) (H_bookshelves : bookshelves = 21) :
  total_books / bookshelves = 2 := 
by
  rw [H_total_books, H_bookshelves]
  norm_num

end books_per_bookshelf_l336_336948


namespace sum_inequality_l336_336767

theorem sum_inequality (n : ℕ) (a : fin (n + 1) → ℝ)
  (h1 : ∀ i, a i ∈ set.Icc (-1 : ℝ) 1)
  (h2 : ∀ i, a i * a (i + 1) ≠ -1)
  (h3 : a ⟨n⟩ = a 0) :
  ∑ i in finset.range n, 1 / (1 + a i * a (i + 1)) ≥
  ∑ i in finset.range n, 1 / (1 + (a i) ^ 2) := 
sorry

end sum_inequality_l336_336767


namespace factor_difference_of_squares_l336_336585

theorem factor_difference_of_squares (x : ℝ) :
  x^2 - 169 = (x - 13) * (x + 13) := by
  have h : 169 = 13^2 := by norm_num
  rw h
  exact by ring

end factor_difference_of_squares_l336_336585


namespace isosceles_triangle_segment_equality_l336_336798

open EuclideanGeometry

variables {A B C E F K L D : Point}

-- Given conditions:

-- Isosceles triangle with base angles
variables (h_iso : distance A B = distance B C)
          (E F : Point)
          (hE : collinear A E C)
          (hF : collinear A F C)
          (D : Point)
          (h1 : ∃ α : Angle, angle E D K = angle A B C)
          (h2 : ∃ β : Angle, angle F D L = angle B C A)
          (hKD : collinear E K D)
          (hLD : collinear F L D)

-- Main goal: 
theorem isosceles_triangle_segment_equality :
  distance B K + distance K D = distance B L + distance L D :=
sorry

end isosceles_triangle_segment_equality_l336_336798


namespace inverse_exists_for_g_and_h_l336_336687

noncomputable def FunctionF := λ x : ℝ, x^2
noncomputable def FunctionG := Piecewise (λ x : ℝ, x) (λ x : ℝ, 2*x) (λ x, x ≤ 0)
noncomputable def FunctionH := λ x : ℝ, x^3

theorem inverse_exists_for_g_and_h :
  (∃ f_inv : ℝ → ℝ, ∀ x : ℝ, FunctionG (f_inv x) = x ∧ f_inv (FunctionG x) = x) ∧
  (∃ h_inv : ℝ → ℝ, ∀ x : ℝ, FunctionH (h_inv x) = x ∧ h_inv (FunctionH x) = x) ∧
  (¬ ∃ f_inv : ℝ → ℝ, ∀ x : ℝ, FunctionF (f_inv x) = x ∧ f_inv (FunctionF x) = x) :=
by
  sorry

end inverse_exists_for_g_and_h_l336_336687


namespace red_grapes_more_than_three_times_green_l336_336346

-- Definitions from conditions
variables (G R B : ℕ)
def condition1 := R = 3 * G + (R - 3 * G)
def condition2 := B = G - 5
def condition3 := R + G + B = 102
def condition4 := R = 67

-- The proof problem
theorem red_grapes_more_than_three_times_green : (R = 67) ∧ (R + G + (G - 5) = 102) ∧ (R = 3 * G + (R - 3 * G)) → R - 3 * G = 7 :=
by sorry

end red_grapes_more_than_three_times_green_l336_336346


namespace calculator_word_sum_correct_l336_336839

def alphabet_values : List Int := [2, -1, 3, 0, -3, -2, -1, 0]

def letter_value (ch : Char) : Int :=
  alphabet_values.get! ((ch.to_nat - 'a'.to_nat) % 8)

noncomputable def word_value_sum (word : String) : Int :=
  (List.map letter_value word.toList).sum

theorem calculator_word_sum_correct : word_value_sum "calculator" = 5 := by
  sorry

end calculator_word_sum_correct_l336_336839


namespace binom_expansion_correct_binom_expansion_mod_correct_l336_336040

-- Definitions and conditions
def binom (n k : ℕ) : ℕ := Nat.choose n k

def binomial_expansion (n : ℕ) (x : ℕ → ℕ) : ℕ → ℕ :=
  λ k => binom n k * x k

def P (n k m : ℕ) : ℕ := (binom n k) % m

def binomial_expansion_mod (n m : ℕ) (x : ℕ → ℕ) : ℕ → ℕ :=
  λ k => P n k m * x k

-- Statement of the proof problem
theorem binom_expansion_correct (n : ℕ) (x : ℕ → ℕ) :
  (1 + x) ^ n = ∑ k in (Finset.range (n + 1)), binom n k * x k := sorry

theorem binom_expansion_mod_correct (n m : ℕ) (x : ℕ → ℕ) :
  (1 + x) ^ n = ∑ k in (Finset.range (n + 1)), P n k m * x k := sorry

end binom_expansion_correct_binom_expansion_mod_correct_l336_336040


namespace complex_part1_complex_part2_l336_336262

noncomputable def complex_question (z : ℂ) : Prop :=
  |z| = |z + 1 + complex.I|

noncomputable def real_imag_equal (z : ℂ) : Prop :=
  z.re = z.im

theorem complex_part1 (z : ℂ) (h1 : complex_question z) (h2 : real_imag_equal z) : z = -1 - complex.I :=
sorry

theorem complex_part2 (z : ℂ) (h1 : complex_question z) : ∃ m, m = |z - 2 + complex.I| ∧ m = sqrt 2 :=
sorry

end complex_part1_complex_part2_l336_336262


namespace term_1500_l336_336219

-- Define the sequence using the properties described in the conditions
def sequence (n : ℕ) : ℕ :=
  let k := ((2 * n + 1 + (1 + 8 * n)^(1/2).toNat) : ℕ)
  in if h : n > 0 then
       let t := (2 * n) - k * (k - 1) / 2 in
       k * (t - 1) + (3 * (k - 1) + t)
     else
       1

theorem term_1500 :
  sequence 1500 = 4392 :=
by
  sorry

end term_1500_l336_336219


namespace cost_prices_l336_336172

theorem cost_prices (C_t C_c C_b : ℝ)
  (h1 : 2 * C_t = 1000)
  (h2 : 1.75 * C_c = 1750)
  (h3 : 0.75 * C_b = 1500) :
  C_t = 500 ∧ C_c = 1000 ∧ C_b = 2000 :=
by
  sorry

end cost_prices_l336_336172


namespace area_PQR_l336_336918

open Real

def point : Type := (ℝ × ℝ)

def line_eq (slope : ℝ) (P : point) : ℝ → ℝ := 
  λ x, slope * (x - P.1) + P.2

def x_intercept (slope : ℝ) (P : point) : ℝ :=
  -P.2 / slope + P.1

def triangle_area (P Q R : point) : ℝ :=
  (1 / 2) * |(Q.1 - R.1) * P.2|

def P : point := (2, 8)
def Q : point := (x_intercept 3 P, 0)
def R : point := (x_intercept (-1) P, 0)

theorem area_PQR : triangle_area P Q R = 128 / 3 := 
  by sorry

end area_PQR_l336_336918


namespace batsman_average_increase_l336_336161

theorem batsman_average_increase (A : ℝ) (X : ℝ) (runs_11th_inning : ℝ) (average_11th_inning : ℝ) 
  (h_runs_11th_inning : runs_11th_inning = 85) 
  (h_average_11th_inning : average_11th_inning = 35) 
  (h_eq : (10 * A + runs_11th_inning) / 11 = average_11th_inning) :
  X = 5 := 
by 
  sorry

end batsman_average_increase_l336_336161


namespace soup_weight_on_fourth_day_l336_336202

-- Definition of the weight function
def weight_on_day (initial_weight : ℝ) (n : ℕ) : ℝ :=
  initial_weight / (2 ^ n)

-- Theorem statement
theorem soup_weight_on_fourth_day (initial_weight : ℝ) (n : ℕ) (h_initial : initial_weight = 80) (h_n : n = 4) : 
  weight_on_day initial_weight n = 5 := 
by
  sorry

end soup_weight_on_fourth_day_l336_336202


namespace stock_price_decrease_l336_336931

theorem stock_price_decrease (x : ℝ) (hx : x > 0) :
  let new_price := 1.30 * x,
      p := 23.077 / 100
  in new_price * (1 - p) = x :=
by
  let new_price := 1.30 * x;
  let p := 23.077 / 100;
  sorry

end stock_price_decrease_l336_336931


namespace trigonometric_expression_evaluation_l336_336657

theorem trigonometric_expression_evaluation
  (α : ℝ)
  (h1 : Real.tan α = -3 / 4) :
  (3 * Real.sin (α / 2) ^ 2 + 
   2 * Real.sin (α / 2) * Real.cos (α / 2) + 
   Real.cos (α / 2) ^ 2 - 2) / 
  (Real.sin (π / 2 + α) * Real.tan (-3 * π + α) + 
   Real.cos (6 * π - α)) = -7 := 
by 
  sorry
  -- This will skip the proof and ensure the Lean code can be built successfully.

end trigonometric_expression_evaluation_l336_336657


namespace ticket_price_possible_values_count_l336_336936

theorem ticket_price_possible_values_count : 
  let x_vals := {x : ℕ | x ∣ 72 ∧ x ∣ 90 ∧ x ∣ 45} in
  x_vals.finite ∧ x_vals.card = 3 :=
by
  sorry

end ticket_price_possible_values_count_l336_336936


namespace remainder_division_l336_336136

theorem remainder_division (x y : ℕ) 
  (h1 : x / y = 96.12) 
  (h2 : y = 11.999999999999545) : 
  (x % y) = 0.12 := 
by
  sorry

end remainder_division_l336_336136


namespace roots_of_quadratic_l336_336152

theorem roots_of_quadratic (x a b c : ℝ) 
  (h1 : √(a-2) + |b+1| + (c+2)^2 = 0) :
  a = 2 ∧ b = -1 ∧ c = -2 → 
  (x = (1+ℝ.sqrt 17)/4 ∨ x = (1-ℝ.sqrt 17)/4) :=
sorry

end roots_of_quadratic_l336_336152


namespace determine_angle_G_l336_336347

theorem determine_angle_G 
  (C D E F G : ℝ)
  (hC : C = 120) 
  (h_linear_pair : C + D = 180)
  (hE : E = 50) 
  (hF : F = D) 
  (h_triangle_sum : E + F + G = 180) :
  G = 70 := 
sorry

end determine_angle_G_l336_336347


namespace first_player_wins_l336_336455

-- Define the initial setting of the game with the number 328
def initial_number := 328

-- Define the set of positive divisors of 328
def divisors_328 : Set ℕ := {1, 2, 4, 8, 41, 82, 164, 328}

-- Define the game conditions
structure game := 
  (number : ℕ)
  (divisors : Set ℕ)
  (no_divisors_of_previously_written : ∀ (a b : ℕ), a ∈ divisors → b ∈ divisors → a ∣ b → a = b)
  (player_writes_328_loses : ∀ player_sequence, (player_sequence.get_last = 328) → player_sequence.get_player.write_328_loses)

-- The theorem to be proved: The first player has a winning strategy
theorem first_player_wins (g : game) : 
  g.number = initial_number → 
  g.divisors = divisors_328 → 
  ∃ strategy : (move * player) . 
  ∀ game_seq, strategy(game_seq).player_1_wins :=
by {
  sorry
}

end first_player_wins_l336_336455


namespace train_waiting_probability_l336_336178

-- Conditions
def trains_per_hour : ℕ := 1
def total_minutes : ℕ := 60
def wait_time : ℕ := 10

-- Proposition
theorem train_waiting_probability : 
  (wait_time : ℝ) / (total_minutes / trains_per_hour) = 1 / 6 :=
by
  -- Here we assume the proof proceeds correctly
  sorry

end train_waiting_probability_l336_336178


namespace european_postcards_cost_l336_336156

def price_per_postcard (country : String) : ℝ :=
  if country = "Italy" ∨ country = "Germany" then 0.10
  else if country = "Canada" then 0.07
  else if country = "Mexico" then 0.08
  else 0.0

def num_postcards (decade : Nat) (country : String) : Nat :=
  if decade = 1950 then
    if country = "Italy" then 10
    else if country = "Germany" then 5
    else if country = "Canada" then 8
    else if country = "Mexico" then 12
    else 0
  else if decade = 1960 then
    if country = "Italy" then 16
    else if country = "Germany" then 12
    else if country = "Canada" then 10
    else if country = "Mexico" then 15
    else 0
  else if decade = 1970 then
    if country = "Italy" then 12
    else if country = "Germany" then 18
    else if country = "Canada" then 13
    else if country = "Mexico" then 9
    else 0
  else 0

def total_cost (country : String) : ℝ :=
  (price_per_postcard country) * (num_postcards 1950 country)
  + (price_per_postcard country) * (num_postcards 1960 country)
  + (price_per_postcard country) * (num_postcards 1970 country)

theorem european_postcards_cost : total_cost "Italy" + total_cost "Germany" = 7.30 := by
  sorry

end european_postcards_cost_l336_336156


namespace amy_remaining_money_l336_336497

-- Define initial amount and purchases
def initial_amount : ℝ := 15
def stuffed_toy_cost : ℝ := 2
def hot_dog_cost : ℝ := 3.5
def candy_apple_cost : ℝ := 1.5
def discount_rate : ℝ := 0.5

-- Define the discounted hot_dog_cost
def discounted_hot_dog_cost := hot_dog_cost * discount_rate

-- Define the total spent
def total_spent := stuffed_toy_cost + discounted_hot_dog_cost + candy_apple_cost

-- Define the remaining amount
def remaining_amount := initial_amount - total_spent

theorem amy_remaining_money : remaining_amount = 9.75 := by
  sorry

end amy_remaining_money_l336_336497


namespace binomial_expansion_term_coefficient_binomial_coefficient_condition_l336_336728

-- Assumption: T represents the terms in the binomial expansion of (√x - 2/x)^12.

theorem binomial_expansion_term_coefficient :
  ∀ (x : ℝ), (∃ T : ℝ, T = (coeff ( (sqrt x - (2/x) )^12 ) ).contains(x^3) 
  → T = 264) :=
by sorry

theorem binomial_coefficient_condition :
  ∃ k : ℕ, (binomial 12 (3*k-1) = binomial 12 (k+1)) → (k = 1 ∨ k = 3) :=
by sorry

end binomial_expansion_term_coefficient_binomial_coefficient_condition_l336_336728


namespace simplify_logical_expression_l336_336812

variables (A B C : Bool)

theorem simplify_logical_expression :
  (A && !B || B && !C || B && C || A && B) = (A || B) :=
by { sorry }

end simplify_logical_expression_l336_336812


namespace broken_marbles_total_l336_336629

theorem broken_marbles_total :
  let broken_set_1 := 0.10 * 50
  let broken_set_2 := 0.20 * 60
  let broken_set_3 := 0.30 * 70
  let broken_set_4 := 0.15 * 80
  let total_broken := broken_set_1 + broken_set_2 + broken_set_3 + broken_set_4
  total_broken = 50 :=
by
  sorry


end broken_marbles_total_l336_336629


namespace dave_lost_tickets_l336_336558

theorem dave_lost_tickets
  (initial_tickets : ℕ)
  (tickets_used : ℕ)
  (tickets_left : ℕ) :
  initial_tickets = 14 →
  tickets_used = 10 →
  tickets_left = 2 →
  initial_tickets - (tickets_used + tickets_left) = 2 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end dave_lost_tickets_l336_336558


namespace inequality_holds_l336_336648

variable (a b c : ℝ)
variable (h1 : a > b) (h2 : b > 1) (h3 : 0 < c) (h4 : c < 1)

theorem inequality_holds : ba^c < ab^c := sorry

end inequality_holds_l336_336648


namespace a_n_b_n_T_n_correct_l336_336854

noncomputable def a_n (n : ℕ) : ℕ := n

noncomputable def b_n (n : ℕ) : ℕ := 2^(n-1)

noncomputable def S (n : ℕ) : ℕ := (Finset.range n).sum a_n

def condition1 := S 2 * b_n 2 = 6
def condition2 := b_n 2 + S 3 = 8

noncomputable def T (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i, a_n (i+1) * b_n (i+1))

theorem a_n_b_n_T_n_correct :
  condition1 →
  condition2 →
  (∀ n, a_n n = n) ∧ 
  (∀ n, b_n n = 2^(n-1)) ∧ 
  (∀ n, T n = 1 + (n-1) * 2^n) :=
by
  sorry

end a_n_b_n_T_n_correct_l336_336854


namespace fraction_ordering_l336_336876

theorem fraction_ordering :
  (6:ℚ)/29 < (8:ℚ)/25 ∧ (8:ℚ)/25 < (10:ℚ)/31 :=
by
  sorry

end fraction_ordering_l336_336876


namespace factor_chain_length_and_count_l336_336899

open Nat

theorem factor_chain_length_and_count (k m n : ℕ) :
  let x := 5^k * 31^m * 1990^n in
  let Lx := 3 * n + k + m in
  let Rx := (3 * n + k + m)! / (n! * n! * n! * (n + k)! * m!) in
  (Lx = 3 * n + k + m) ∧ (Rx = (3 * n + k + m)! / (n! * n! * n! * (n + k)! * m!)) :=
by
  sorry

end factor_chain_length_and_count_l336_336899


namespace smaller_angle_in_parallelogram_l336_336344

theorem smaller_angle_in_parallelogram 
  (opposite_angles : ∀ A B C D : ℝ, A = C ∧ B = D)
  (adjacent_angles_supplementary : ∀ A B : ℝ, A + B = π)
  (angle_diff : ∀ A B : ℝ, B = A + π/9) :
  ∃ θ : ℝ, θ = 4 * π / 9 :=
by
  sorry

end smaller_angle_in_parallelogram_l336_336344


namespace linear_function_correct_max_profit_correct_min_selling_price_correct_l336_336912

-- Definition of the linear function
def linear_function (x : ℝ) : ℝ :=
  -2 * x + 360

-- Definition of monthly profit function
def profit_function (x : ℝ) : ℝ :=
  (-2 * x + 360) * (x - 30)

noncomputable def max_profit_statement : Prop :=
  ∃ x w, x = 105 ∧ w = 11250 ∧ profit_function x = w

noncomputable def min_selling_price (profit : ℝ) : Prop :=
  ∃ x, profit_function x ≥ profit ∧ x ≥ 80

-- The proof statements
theorem linear_function_correct : linear_function 30 = 300 ∧ linear_function 45 = 270 :=
  by
    sorry

theorem max_profit_correct : max_profit_statement :=
  by
    sorry

theorem min_selling_price_correct : min_selling_price 10000 :=
  by
    sorry

end linear_function_correct_max_profit_correct_min_selling_price_correct_l336_336912


namespace sin_4theta_l336_336330

theorem sin_4theta (θ : ℝ) (h : Complex.exp (Complex.I * θ) = (4 + Complex.I * Real.sqrt 7) / 5) :
  Real.sin (4 * θ) = (144 * Real.sqrt 7) / 625 := by
  sorry

end sin_4theta_l336_336330


namespace collinear_vectors_x_value_l336_336684

theorem collinear_vectors_x_value :
  let a := (3, 4 : ℝ × ℝ)
  let b := (2, 1 : ℝ × ℝ)
  let lhs := (3 + 2 * x, 4 + x)
  let rhs := (1, 3)
  (lhs.1 * rhs.2 = lhs.2 * rhs.1) → 
  x = -1 :=
by {
  intros a b lhs rhs parallel_lhs_rhs_eq,
  sorry
}

end collinear_vectors_x_value_l336_336684


namespace karl_savings_l336_336377

noncomputable def cost_per_notebook : ℝ := 3.75
noncomputable def notebooks_bought : ℕ := 8
noncomputable def discount_rate : ℝ := 0.25
noncomputable def original_total_cost : ℝ := notebooks_bought * cost_per_notebook
noncomputable def discount_per_notebook : ℝ := cost_per_notebook * discount_rate
noncomputable def discounted_price_per_notebook : ℝ := cost_per_notebook - discount_per_notebook
noncomputable def discounted_total_cost : ℝ := notebooks_bought * discounted_price_per_notebook
noncomputable def total_savings : ℝ := original_total_cost - discounted_total_cost

theorem karl_savings : total_savings = 7.50 := by 
  sorry

end karl_savings_l336_336377


namespace max_a4_l336_336269

variable (a1 d : ℝ)
variable S : ℕ → ℝ
variable a : ℕ → ℝ

axiom sum_arithmetic_sequence (n : ℕ) :
  S n = n * a1 + n * (n - 1) / 2 * d

axiom s4_ge_10 : S 4 ≥ 10
axiom s5_le_15 : S 5 ≤ 15

theorem max_a4 : a 4 ≤ 4 :=
by
  have h1 : 2 * a1 + 3 * d ≥ 5 := by
    sorry
  have h2 : a1 + 2 * d ≤ 3 := by
    sorry
  have h3 : a 4 = a1 + 3 * d := by
    sorry
  have h4 : a1 + 3 * d ≤ 4 := by
    sorry
  exact h4

end max_a4_l336_336269


namespace smallest_positive_integer_in_set_l336_336465

theorem smallest_positive_integer_in_set :
  let A := {x : ℝ | |x - 3| ≥ 2}
  ∃ n : ℕ, n > 0 ∧ n ∈ A ∧ (∀ m : ℕ, m > 0 ∧ m ∈ A → n ≤ m) := 
begin
  let A := {x : ℝ | |x - 3| ≥ 2},
  use 1,
  split,
  { exact nat.succ_pos' 0, },
  split,
  {
    change 1 ∈ A,
    unfold set.mem,
    dsimp,
    norm_num,
  },
  {
    intros m hm1 hm2,
    change m ∈ A at hm2,
    unfold set.mem at hm2,
    dsimp at hm2,
    cases hm1,
    cases hm2,
    {   exfalso,
        linarith, },
    {
      intros k hk,
      linarith,
    }
  }
end

end smallest_positive_integer_in_set_l336_336465


namespace problem_statement_l336_336292

def f (x : ℝ) : ℝ := 
  if x > 1 then Math.log x / Math.log 3
  else if -1 < x ∧ x ≤ 1 then x^2
  else 3^x

theorem problem_statement : 
  f (-f (Real.sqrt 3)) + f (f 0) + f (1 / f (-1)) = 5 / 4 :=
by 
  sorry

end problem_statement_l336_336292


namespace john_average_score_change_l336_336374

/-- Given John's scores on his biology exams, calculate the change in his average score after the fourth exam. -/
theorem john_average_score_change :
  let first_three_scores := [84, 88, 95]
  let fourth_score := 92
  let first_average := (84 + 88 + 95) / 3
  let new_average := (84 + 88 + 95 + 92) / 4
  new_average - first_average = 0.75 :=
by
  sorry

end john_average_score_change_l336_336374


namespace student_homework_sampling_is_systematic_l336_336840

theorem student_homework_sampling_is_systematic :
  (∀ studentID : ℕ, (studentID % 10 = 5) → true) →
  (∀ class : Type, (∃ students : set ℕ, true)) →
  ∀ method : String, method = "systematic" :=
by
  sorry

end student_homework_sampling_is_systematic_l336_336840


namespace coordinates_of_A_l336_336353

-- Definition of the distance function for any point (x, y)
def distance_to_x_axis (x y : ℝ) : ℝ := abs y
def distance_to_y_axis (x y : ℝ) : ℝ := abs x

-- Point A's coordinates
def point_is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- The main theorem to prove
theorem coordinates_of_A :
  ∃ (x y : ℝ), 
  point_is_in_fourth_quadrant x y ∧ 
  distance_to_x_axis x y = 3 ∧ 
  distance_to_y_axis x y = 6 ∧ 
  (x, y) = (6, -3) :=
by 
  sorry

end coordinates_of_A_l336_336353


namespace can_be_high_income_earner_income_range_sufficiency_middle_50_percent_range_sufficiency_median_income_estimation_l336_336199

def num_employees : ℕ := 50
def max_income : ℝ := 1000000
def average_income : ℝ := 35000
def min_income : ℝ := 5000
def middle_50_percent_income_range_lower : ℝ := 10000
def middle_50_percent_income_range_upper : ℝ := 30000

-- Question 1: Can you determine if you can become one of the high-income earners in this company?
theorem can_be_high_income_earner : 
  (min_income < 25000 ∧ max_income > 25000) → 
  ∃ e ∈ set.Icc(min_income,max_income), e = 25000 → false :=
by sorry

-- Question 2: Is the provided income range information sufficient to determine the income for new employees?
theorem income_range_sufficiency : 
  (min_income ≤ 5000 ∧ max_income ≥ 1000000) → false :=
by sorry

-- Question 3: Can the middle 50% income range help you make a decision about whether to accept a job offer?
theorem middle_50_percent_range_sufficiency : 
  (middle_50_percent_income_range_lower ≤ 25000 ∧ 
  middle_50_percent_income_range_upper ≥ 25000) → true :=
by sorry

-- Question 4: Can you estimate the median income? Why is the average higher than the estimated median?
theorem median_income_estimation : 
  average_income > middle_50_percent_income_range_lower ∧ 
  average_income < max_income → 
  (median_income : ℝ) = ∑ e ∈ finset.Icc(min_income,max_income), e / num_employees → false :=
by sorry

end can_be_high_income_earner_income_range_sufficiency_middle_50_percent_range_sufficiency_median_income_estimation_l336_336199


namespace triangle_area_expression_l336_336583

-- Define the input parameters for the triangle
variables (a : ℝ) (β γ : ℝ)

-- Define the area function in terms of a, β, γ
def triangle_area (a β γ : ℝ) : ℝ :=
  (a ^ 2 * Real.sin β * Real.sin γ) / (2 * Real.sin (β + γ))

-- State the theorem
theorem triangle_area_expression :
  ∃ S, S = (a ^ 2 * Real.sin β * Real.sin γ) / (2 * Real.sin (β + γ)) :=
begin
  use (a ^ 2 * Real.sin β * Real.sin γ) / (2 * Real.sin (β + γ)),
  sorry,
end

end triangle_area_expression_l336_336583


namespace parametric_line_through_F2_perpendicular_to_AF1_length_MN_l336_336665

-- Definitions
def conic_curve (θ : ℝ) : ℝ × ℝ :=
  (sqrt 2 * cos θ, sin θ)

def point_A : ℝ × ℝ :=
  (0, sqrt 3 / 3)

def focus_F1 : ℝ × ℝ :=
  (-1, 0)

def focus_F2 : ℝ × ℝ :=
  (1, 0)

-- Proof Problem Statements
theorem parametric_line_through_F2_perpendicular_to_AF1 :
  ∃ t : ℝ, (∃ (x y : ℝ), (x = 1 - (1 / 2) * t) ∧ (y = (sqrt 3 / 2) * t)) := by
  sorry

theorem length_MN :
  ∃ t1 t2 : ℝ, 
  (solving 7*(t1^2) - 4*t1 - 4 = 0 ∧ solving 7*(t2^2) - 4*t2 - 4 = 0) →
  (∃ MN : ℝ, MN = abs (t1 - t2) ∧ MN = 8 * sqrt 2 / 7) := by
  sorry

end parametric_line_through_F2_perpendicular_to_AF1_length_MN_l336_336665


namespace point_closest_to_origin_l336_336504

noncomputable def distance_2d (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)
noncomputable def distance_3d (x y z : ℝ) : ℝ := Real.sqrt (x^2 + y^2 + z^2)

theorem point_closest_to_origin :
  ∀ (p₁ p₂ p₃ p₄ p₅ : (ℝ × ℝ) ⊕ (ℝ × ℝ × ℝ)),
    p₁ = (sum.inl (2, 3)) →
    p₂ = (sum.inl (4, 0)) →
    p₃ = (sum.inl (1, -1)) →
    p₄ = (sum.inl (-3, 4)) →
    p₅ = (sum.inr (0, 0, 5)) →
    ∃ p, p = p₃ ∧
    (∀ q ∈ {p₁, p₂, p₃, p₄, p₅}, distance (sum.inl (fst q)) (sum.snd q) (sum.inr (0, 0, 0)) ≥ distance p (sum.inr (0, 0, 0))) 
  :=
by sorry

end point_closest_to_origin_l336_336504


namespace width_of_vessel_is_5_l336_336171

open Real

noncomputable def width_of_vessel : ℝ :=
  let edge := 5
  let rise := 2.5
  let base_length := 10
  let volume_cube := edge ^ 3
  let volume_displaced := volume_cube
  let width := volume_displaced / (base_length * rise)
  width

theorem width_of_vessel_is_5 :
  width_of_vessel = 5 := by
    sorry

end width_of_vessel_is_5_l336_336171


namespace crossed_out_number_is_29_l336_336797

theorem crossed_out_number_is_29 : 
  ∀ n : ℕ, (11 * n + 66 - (325 - (12 * n + 66 - 325))) = 29 :=
by sorry

end crossed_out_number_is_29_l336_336797


namespace cost_of_a_pen_l336_336609

theorem cost_of_a_pen:
  ∃ x y : ℕ, 5 * x + 4 * y = 345 ∧ 3 * x + 6 * y = 285 ∧ x = 52 :=
by
  sorry

end cost_of_a_pen_l336_336609


namespace general_term_formula_sum_of_first_n_terms_l336_336277

-- Problem 1: General Term Formula
theorem general_term_formula
  (p q : ℝ) (h_q_nonzero : q ≠ 0)
  (α β : ℝ)
  (h_roots : has_roots (x ^ 2 - p * x + q) α β)
  (a : ℕ → ℝ) (h_seq : sequence a p q) :
  ∀ n, a n = (α ^ (n + 1) - β ^ (n + 1)) / (α - β) := sorry

-- Problem 2: Sum of the First n Terms when p = 1 and q = 1/4
theorem sum_of_first_n_terms
  (p : ℝ := 1) (q : ℝ := 1/4)
  (α β : ℝ := 1/2)
  (a : ℕ → ℝ) (h_seq : sequence a p q)
  (S : ℕ → ℝ) (h_sum : ∀ n, S n = ∑ i in range (n + 1), a i) :
  ∀ n, S n = 3 - (n + 3) / 2 ^ n := sorry

end general_term_formula_sum_of_first_n_terms_l336_336277


namespace margaret_mean_score_l336_336479

theorem margaret_mean_score (scores : List ℕ) (c_scores : List ℕ) (m_scores : List ℕ) (h1 : scores = [85, 87, 92, 93, 94, 98])
  (h2 : c_scores.length = 3) (h3 : m_scores.length = 3) (h4 : c_scores.sum = 270) (h5 : scores.sum = c_scores.sum + m_scores.sum) : 
  m_scores.sum / m_scores.length = 93 := by
    have total_scores_len : scores.length = 6 := by simp [h1]
    have h6 : scores.sum = 549 := by simp [h1]
    have h7 : m_scores.sum = 549 - 270 := by rw [←h5, h4, h6, add_sub_cancel]
    simp [h7]
    norm_num
    sorry

end margaret_mean_score_l336_336479


namespace marbles_lost_l336_336004

theorem marbles_lost (n m l : ℕ) (h₁ : n = 9) (h₂ : m = 4) (h₃ : n - m = l) : l = 5 :=
by 
sor

end marbles_lost_l336_336004


namespace candy_bar_reduction_l336_336164

variable (W P x : ℝ)
noncomputable def percent_reduction := (x / W) * 100

theorem candy_bar_reduction (h_weight_reduced : W > 0) 
                            (h_price_same : P > 0) 
                            (h_price_increase : P / (W - x) = (5 / 3) * (P / W)) :
    percent_reduction W x = 40 := 
sorry

end candy_bar_reduction_l336_336164


namespace surface_area_of_sphere_l336_336179

-- Define the conditions from the problem.

variables (r R : ℝ) -- r is the radius of the cross-section, R is the radius of the sphere.
variables (π : ℝ := Real.pi) -- Define π using the real pi constant.
variables (h_dist : 1 = 1) -- Distance from the plane to the center is 1 unit.
variables (h_area_cross_section : π = π * r^2) -- Area of the cross-section is π.

-- State to prove the surface area of the sphere is 8π.
theorem surface_area_of_sphere :
    ∃ (R : ℝ), (R^2 = 2) → (4 * π * R^ 2 = 8 * π) := sorry

end surface_area_of_sphere_l336_336179


namespace rectangle_area_outside_circles_approx_l336_336048

theorem rectangle_area_outside_circles_approx :
  let EF := 4
  let FG := 6
  let rE := 2
  let rF := 3
  let rG := 4
  let area_rectangle := EF * FG
  let area_quarter_circles := (Math.pi * rE ^ 2) / 4 + (Math.pi * rF ^ 2) / 4 + (Math.pi * rG ^ 2) / 4
  let area_outside_circles := area_rectangle - area_quarter_circles
  area_outside_circles ≈ 1.2 := 
  sorry

end rectangle_area_outside_circles_approx_l336_336048


namespace problem_statement_l336_336067

noncomputable def verify_values (r : ℝ) (a w v : ℝ) (A B C D E : EuclideanGeometry.Point ℝ) : Prop :=
  ∃ (a b : ℕ),  ∃ (O : EuclideanGeometry.Point ℝ) (Ω : EuclideanGeometry.Circle ℝ),
  ∃ (AB : EuclideanGeometry.Line ℝ) (tangentC : EuclideanGeometry.Line ℝ),
  a + b = 37 ∧
  EuclideanGeometry.circle_center Ω = O ∧
  EuclideanGeometry.circle_radius Ω = r ∧
  EuclideanGeometry.on_circle A Ω ∧
  EuclideanGeometry.on_circle B Ω ∧
  EuclideanGeometry.on_circle C Ω ∧
  EuclideanGeometry.on_circle D Ω ∧
  EuclideanGeometry.distance D C = 8 ∧
  EuclideanGeometry.distance D B = 11 ∧
  EuclideanGeometry.intersect_line_with_tangent_at_circle AB tangentC C E ∧
  EuclideanGeometry.distance D E = a * Real.sqrt b
  
theorem problem_statement (r : ℝ) (a w v : ℝ) (A B C D E : EuclideanGeometry.Point ℝ) : verify_values r a w v A B C D E :=
  sorry

end problem_statement_l336_336067


namespace quadrilateral_inscribed_circle_l336_336645

theorem quadrilateral_inscribed_circle
  (AB BC CD DA BD : ℝ)
  (hAB : AB = 15)
  (hBC : BC = 36)
  (hCD : CD = 48)
  (hDA : DA = 27)
  (hBD : BD = 54) :
  (∃ d : ℝ, is_circumscribed_quadrilateral AB BC CD DA BD ∧ d = 54) :=
by
  sorry

end quadrilateral_inscribed_circle_l336_336645


namespace tan_75_l336_336984

namespace Tangent
open Real

-- Definitions of the angle tangents
def tan_60 := Real.sqrt 3
def tan_15 := 2 - Real.sqrt 3

-- The main statement to be proved
theorem tan_75 : Real.tan (75 * Real.pi / 180) = 2 + Real.sqrt 3 := by
  sorry
end Tangent

end tan_75_l336_336984


namespace max_value_of_M_l336_336708

theorem max_value_of_M (a b c : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c) :
  ∃ M : ℝ, (∀ a b c : ℝ, 0 ≤ a → 0 ≤ b → 0 ≤ c → a^3 + b^3 + c^3 - 3 * a * b * c ≥ M * (a - b) * (b - c) * (c - a))
        ∧ M = sqrt (9 + 6 * sqrt 3) := 
sorry

end max_value_of_M_l336_336708


namespace sum_gcd_lcm_eq_180195_l336_336882

def gcd_60_45045 := Nat.gcd 60 45045
def lcm_60_45045 := Nat.lcm 60 45045

theorem sum_gcd_lcm_eq_180195 : gcd_60_45045 + lcm_60_45045 = 180195 := by
  sorry

end sum_gcd_lcm_eq_180195_l336_336882


namespace find_n_l336_336597

theorem find_n (n : ℕ) (h : n * Nat.factorial n + Nat.factorial n = 5040) : n = 6 :=
sorry

end find_n_l336_336597


namespace negation_of_proposition_l336_336087

theorem negation_of_proposition (x : ℝ) : 
  ¬ (|x| < 2 → x < 2) ↔ (|x| ≥ 2 → x ≥ 2) :=
sorry

end negation_of_proposition_l336_336087


namespace find_f_7_l336_336649

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom function_period : ∀ x : ℝ, f (x + 2) = -f x
axiom function_value_range : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_7 : f 7 = -1 := by
  sorry

end find_f_7_l336_336649


namespace circumradius_medians_inequality_l336_336432

variables {α : Type*} [linear_ordered_field α] {a b c : α} (R_original R_medians : α) 

-- Definition of medians for sides a, b, c
def median (x y : α) : α := sqrt (2*x*x + 2*y*y - (1*y)^2) / 2

-- Definition of circumradius for a triangle given sides a, b, and c
def circumradius (a b c : α) (S : α) : α :=
  (a * b * c) / (4 * S)

-- Conditions: triangle is acute-angled
def is_acute_angled (a b c : α) : Prop :=
  (a^2 + b^2 > c^2) ∧ (a^2 + c^2 > b^2) ∧ (b^2 + c^2 > a^2)

-- Proof that radius of the circumcircle of triangle formed by medians is greater than 5/6 of the radius of the circumcircle of original triangle
theorem circumradius_medians_inequality (h_acute : is_acute_angled a b c)
  (R_original_eq : R_original = circumradius a b c (sqrt ((a+b+c) * (a+b-c) * (a+c-b) * (b+c-a)) / 4))
  (R_medians_eq : R_medians = circumradius (median a b) (median b c) (median c a) (sqrt ((median a b + median b c + median c a) * (median a b + median b c - median c a) * (median a b + median c a - median b c) * (median b c + median c a - median a b)) / 4)) :
  R_medians > 5/6 * R_original := 
sorry

end circumradius_medians_inequality_l336_336432


namespace problem_statement_l336_336772

theorem problem_statement (p q : ℕ) (hp1 : 1 < p) (hq1 : 1 < q) (h_coprime : Nat.coprime p (6 * q)) :
  (∑ k in Finset.range (q - 1), (⌊(p * k : ℝ)/q⌋)^2) ≡ 2 * p * (∑ k in Finset.range (q - 1), (k * ⌊(p * k : ℝ)/q⌋)) [MOD q - 1] :=
by
  sorry

end problem_statement_l336_336772


namespace tetrahedron_edge_length_l336_336496

theorem tetrahedron_edge_length (a : ℝ) (V : ℝ) 
  (h₀ : V = 0.11785113019775793) 
  (h₁ : V = (Real.sqrt 2 / 12) * a^3) : a = 1 := by
  sorry

end tetrahedron_edge_length_l336_336496


namespace bush_height_l336_336527

theorem bush_height (h : ℕ → ℕ) (h0 : h 5 = 81) (h1 : ∀ n, h (n + 1) = 3 * h n) :
  h 2 = 3 := 
sorry

end bush_height_l336_336527


namespace nuts_to_raisins_ratio_l336_336212

/-- 
Given that Chris mixed 3 pounds of raisins with 4 pounds of nuts 
and the total cost of the raisins was 0.15789473684210525 of the total cost of the mixture, 
prove that the ratio of the cost of a pound of nuts to the cost of a pound of raisins is 4:1. 
-/
theorem nuts_to_raisins_ratio (R N : ℝ)
    (h1 : 3 * R = 0.15789473684210525 * (3 * R + 4 * N)) :
    N / R = 4 :=
sorry  -- proof skipped

end nuts_to_raisins_ratio_l336_336212


namespace min_distance_l336_336285

def point_on_curve (θ : ℝ) : Prop :=
  ∃ ρ, ρ = 2 * Real.sin θ

def line_equation (θ : ℝ) : Prop :=
  ∃ ρ, ρ * Real.sin (θ + π / 3) = 4

theorem min_distance (θ : ℝ) (ρ : ℝ) (h_curve : point_on_curve θ) (h_line : line_equation θ) :
  true := -- Here we indicate the theorem is about proving the minimum distance
begin
  have h1 : ∀ (P : ℝ × ℝ),
  let r_circle := 1 in
  let center := (0, 1 : ℝ) in
  (√3 * P.1 + P.2 - 8)/√4 = 7/2 → -- Equation conversion
  sorry
end

end min_distance_l336_336285


namespace count_three_digit_numbers_satisfying_property_l336_336695

/-- Define digit for ease of use -/
def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

/-- Define the three-digit number satisfying the middle digit condition -/
def satisfies_condition (a b c : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧
  a ≠ 0 ∧
  b = (a + c) / 2 ∧
  (a + c) % 2 = 0

/-- Prove how many such numbers exist -/
theorem count_three_digit_numbers_satisfying_property :
  (finset.card {n : ℕ | ∃ a b c, satisfies_condition a b c ∧ n = 100 * a + 10 * b + c} = 45) :=
sorry

end count_three_digit_numbers_satisfying_property_l336_336695


namespace quadratic_nonzero_domain_l336_336706

theorem quadratic_nonzero_domain (m : ℝ) (h : 0 ≤ m ∧ m < 3/4) : ∀ x : ℝ, mx^2 + 4mx + 3 ≠ 0 := by
  sorry

end quadratic_nonzero_domain_l336_336706


namespace minimum_value_is_two_sqrt_two_l336_336601

noncomputable def minimum_value_expression (x : ℝ) : ℝ :=
  (Real.sqrt (x^2 + (2 - x)^2)) + (Real.sqrt ((2 - x)^2 + x^2))

theorem minimum_value_is_two_sqrt_two :
  ∃ x : ℝ, minimum_value_expression x = 2 * Real.sqrt 2 :=
by 
  sorry

end minimum_value_is_two_sqrt_two_l336_336601


namespace sum_inequality_l336_336770

theorem sum_inequality (n : ℕ) (a : ℕ → ℝ) (h_range : ∀ i, 1 ≤ i ∧ i ≤ n → a i ∈ Set.Icc (-1 : ℝ) 1)
  (h_not_neg_one : ∀ i, 1 ≤ i ∧ i ≤ n → a i * a ((i % n) + 1) ≠ -1)
  (h_cyclic: a (n+1) = a 1) :
  (∑ i in Finset.range n, 1 / (1 + a i * a (i + 1))) ≥ (∑ i in Finset.range n, 1 / (1 + (a i) ^ 2)) :=
sorry

end sum_inequality_l336_336770


namespace sum_of_squares_of_coeffs_l336_336134

def sum_of_squares_of_coefficients (p : Polynomial ℤ) (k : ℤ) : ℤ :=
  p.support.sum (λ n, (k * p.coeff n)^2)

-- Given polynomial and multiplier
def p := monomial 5 1 + monomial 3 2 + monomial 1 1 + monomial 0 3
def k := 6

theorem sum_of_squares_of_coeffs : sum_of_squares_of_coefficients p k = 540 :=
by simp [sum_of_squares_of_coefficients, p, k]; norm_num; sorry

end sum_of_squares_of_coeffs_l336_336134


namespace coefficient_x5_l336_336079

noncomputable def coefficient_of_x5_in_expansion : ℤ :=
  6

theorem coefficient_x5 (x : ℤ) :
  let f := (1 + x - x^2) in
  ∑ i in finset.range (7), (nat.choose 6 i) * ((1 + x)^(6 - i)) * ((-x^2)^i) = 6 :=
by
  sorry

end coefficient_x5_l336_336079


namespace result_when_j_divided_by_26_l336_336337

noncomputable def j := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 10 11) (Nat.lcm 12 13)) (Nat.lcm 14 15))

theorem result_when_j_divided_by_26 : j / 26 = 2310 := by 
  sorry

end result_when_j_divided_by_26_l336_336337


namespace rhombus_property_l336_336505

-- Definitions of properties according to the given problem
def diagonals_always_equal (R : Type) [Geometry R] (rhombus : R) : Prop :=
  ∀ (d1 d2 : R), (is_diagonal d1 rhombus ∧ is_diagonal d2 rhombus) → (length d1 = length d2)

def diagonals_perpendicular (R : Type) [Geometry R] (rhombus : R) : Prop :=
  ∀ (d1 d2 : R), (is_diagonal d1 rhombus ∧ is_diagonal d2 rhombus) → (is_perpendicular d1 d2)

def is_symmetrical (R : Type) [Geometry R] (rhombus : R) : Prop :=
  ∃ (axis : R), is_line_of_symmetry axis rhombus

def is_centrally_symmetrical (R : Type) [Geometry R] (rhombus : R) : Prop :=
  ∃ (point : R), is_central_symmetry point rhombus

-- The Lean statement for the problem description
theorem rhombus_property (R : Type) [Geometry R] (rhombus : R) :
  diagonals_perpendicular R rhombus →
  is_symmetrical R rhombus →
  is_centrally_symmetrical R rhombus →
  ¬ diagonals_always_equal R rhombus :=
sorry

end rhombus_property_l336_336505


namespace problem1_problem2_l336_336063

-- Problem 1

def a : ℚ := -1 / 2
def b : ℚ := -1

theorem problem1 :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) + a * b^2 = -3 / 4 :=
by
  sorry

-- Problem 2

def x : ℚ := 1 / 2
def y : ℚ := -2 / 3
axiom condition2 : abs (2 * x - 1) + (3 * y + 2)^2 = 0

theorem problem2 :
  5 * x^2 - (2 * x * y - 3 * (x * y / 3 + 2) + 5 * x^2) = 19 / 3 :=
by
  have h : abs (2 * x - 1) + (3 * y + 2)^2 = 0 := condition2
  sorry

end problem1_problem2_l336_336063


namespace circumcircle_radius_triangle_max_area_l336_336714

-- Definitions for the conditions
variables {A B C : Real} -- Angles in the triangle
variables {a b c : ℝ} -- Sides opposite to angles A, B, and C
variable (h_b : b = 2)

-- Problem 1: If angles A, B, C form an arithmetic progression, find the radius of the circumcircle
theorem circumcircle_radius (h_arith : B = π / 3) : 
  let R := circumcircle_radius ?m_1 ?m_2 ?m_3 := (2 * b) / sin B
  R = 2 * (sqrt 3) / 3 := sorry

-- Problem 2: If sides a, b, c form an arithmetic progression, find the maximum area of the triangle
theorem triangle_max_area (h_arith : 2 * b = a + c) :
  let area := 1 / 2 * a * c * sin B
  area ≤ sqrt 3 := sorry

end circumcircle_radius_triangle_max_area_l336_336714


namespace parallel_vectors_l336_336155

theorem parallel_vectors (x : ℝ) :
  let a := (-1, 2, -3) in
  let b := (2, x, 6) in
  (∃ k : ℝ, a = k • b) → x = -4 :=
by
  intros a b h
  cases a with a1 a23
  cases b with b1 b23
  cases a23 with a2 a3
  cases b23 with b2 b3
  sorry

end parallel_vectors_l336_336155


namespace gasoline_price_growth_l336_336998

variable (x : ℝ)

/-- The average monthly growth rate of the price of 92-octane gasoline. -/
theorem gasoline_price_growth (h₁ : 7.5 > 0) (h₂ : 1 + x > 0) (h₃ : 7.5 * (1 + x)^2 = 8.4) : 7.5 * (1 + x)^2 = 8.4 :=
by {
  exact h₃,
  sorry -- proof omitted
}

end gasoline_price_growth_l336_336998


namespace prove_value_of_m_l336_336662

variables {A B C D O : Type} -- Points
noncomputable def rhombus_side_length (ABCD: Type) (side_length: ℝ) : Prop := (side_length = 5)
noncomputable def diagonals_intersect (ABCD O: Type) : Prop := true -- Assuming diagonals intersect at point O
noncomputable def quadratic_roots {m : ℝ} (OA OB: ℝ) : Prop := ∀ x, x * x + (2 * m - 1) * x + (m * m + 3) = 0
noncomputable def value_of_m (m : ℝ) : Prop := m = -3

theorem prove_value_of_m
  (ABCD: Type)
  {side_length: ℝ} 
  (rhombus_side : rhombus_side_length ABCD side_length)
  (diagonals_inter : diagonals_intersect ABCD O)
  {m : ℝ} 
  {OA OB: ℝ}
  (roots: quadratic_roots OA OB) :
  value_of_m m :=
sorry

end prove_value_of_m_l336_336662


namespace minimum_value_of_f_l336_336765

def f (x : ℝ) : ℝ :=
if x ≤ 1 then 1 * x^2 + -2 * x + 1 else 2 * x + 0

theorem minimum_value_of_f : ∃ x : ℝ, f x = 0 :=
by {
  -- Proof skipped
  sorry
}

end minimum_value_of_f_l336_336765


namespace sum_sequence_l336_336470

noncomputable def sum_first_n_minus_1_terms (n : ℕ) : ℕ :=
  (2^n - n - 1)

theorem sum_sequence (n : ℕ) : 
  sum_first_n_minus_1_terms n = (2^n - n - 1) :=
by
  sorry 

end sum_sequence_l336_336470


namespace converse_of_rectangle_proposition_l336_336461

def eq_diagonals (P : Type) [parallelogram P] : Prop := sorry
def is_rectangle (P : Type) [parallelogram P] : Prop := sorry

theorem converse_of_rectangle_proposition (P : Type) [parallelogram P] :
  (∀ p : P, eq_diagonals p → is_rectangle p) ↔
  (∀ p : P, is_rectangle p → eq_diagonals p) :=
sorry

end converse_of_rectangle_proposition_l336_336461


namespace total_selling_price_is_correct_l336_336537

def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

def discount : ℝ := discount_rate * original_price
def sale_price : ℝ := original_price - discount
def tax : ℝ := tax_rate * sale_price
def total_selling_price : ℝ := sale_price + tax

theorem total_selling_price_is_correct : total_selling_price = 96.6 := by
  sorry

end total_selling_price_is_correct_l336_336537


namespace min_a_4_l336_336258

theorem min_a_4 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 9 * x + y = x * y) : 
  4 * x + y ≥ 25 :=
sorry

end min_a_4_l336_336258


namespace stack_of_paper_l336_336545

theorem stack_of_paper (sheet_thickness : ℝ) (num_sheets : ℕ) (stack_height : ℝ) 
                       (weight_per_sheet : ℝ) (new_stack_height : ℝ) :
  sheet_thickness = 4 / 800 ∧ stack_height = 4 ∧ weight_per_sheet = 4.5 ∧
  new_stack_height = 10 →
  let new_num_sheets := new_stack_height / sheet_thickness in
  new_num_sheets = 2000 ∧ new_num_sheets * weight_per_sheet = 9000 :=
by
  sorry

end stack_of_paper_l336_336545


namespace john_saving_percentage_l336_336003

theorem john_saving_percentage 
    (amount_saved : ℝ) 
    (amount_paid_after_first_discount : ℝ) 
    (additional_discount_rate : ℝ) 
    (original_price : ℝ) 
    (first_discount_percentage_saved : ℝ) 
    (total_savings : ℝ) 
    (total_discount_percentage_saved : ℝ) : 
    amount_saved = 120 ∧ 
    amount_paid_after_first_discount = 1080 ∧ 
    additional_discount_rate = 0.05 ∧ 
    original_price = amount_paid_after_first_discount + amount_saved ∧ 
    first_discount_percentage_saved = (amount_saved / original_price) * 100 ∧ 
    total_savings = amount_saved + (additional_discount_rate * amount_paid_after_first_discount) ∧ 
    total_discount_percentage_saved = (total_savings / original_price) * 100 
    → total_discount_percentage_saved = 14.5 :=
begin 
sorry 
end

end john_saving_percentage_l336_336003


namespace range_of_a_l336_336294

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a x : ℝ) : ℝ := f a x - g x
noncomputable def k (x : ℝ) : ℝ := (Real.log x + x) / x^2

theorem range_of_a (a : ℝ) (h_zero : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ h a x₁ = 0 ∧ h a x₂ = 0) :
  0 < a ∧ a < 1 :=
sorry

end range_of_a_l336_336294


namespace projections_are_equidistant_l336_336061

open Lean

variables {A B C O D : Type} [EuclideanSpace A] [EuclideanSpace B] [EuclideanSpace C] [EuclideanSpace O] [EuclideanSpace D]

-- Conditions defining the problem:
variables {tri : Triangle} {circ : Circle tri.circ} (COD : Diameter circ) (E F G: Point)
(h_projC : PerpendicularProjection circ.point C E AB)   
(h_projO : PerpendicularProjection circ.point O G AB) 
(h_projD : PerpendicularProjection circ.point D F AB)

theorem projections_are_equidistant (AE BF : Segment):
  AE = BF := 
begin
  sorry
end

end projections_are_equidistant_l336_336061


namespace calculate_train_speed_l336_336548

-- Define the given conditions
def length_train : ℝ := 165
def length_bridge : ℝ := 660
def time_cross : ℝ := 41.24670026397888

-- Define the correct answer in km/h
def expected_speed_kmh : ℝ := 72

-- The proof problem statement
theorem calculate_train_speed :
  let total_distance := length_train + length_bridge in
  let speed_mps := total_distance / time_cross in
  let speed_kmh := speed_mps * 3.6 in
  speed_kmh ≈ expected_speed_kmh :=
by
  sorry

end calculate_train_speed_l336_336548


namespace smallest_n_l336_336127

theorem smallest_n :
∃ (n : ℕ), (0 < n) ∧ (∃ k1 : ℕ, 5 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 7 * n = k2 ^ 3) ∧ n = 1225 :=
sorry

end smallest_n_l336_336127


namespace binomial_12_6_eq_924_l336_336977

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := sorry

end binomial_12_6_eq_924_l336_336977


namespace trig_identity_proof_l336_336255

theorem trig_identity_proof
  (α β : ℝ)
  (h1 : sin α + cos α = (3 * real.sqrt 5) / 5)
  (h2 : 0 < α ∧ α < π / 4)
  (h3 : sin (β - π / 4) = 3 / 5)
  (h4 : π / 4 < β ∧ β < π / 2)
  :
  sin (2 * α) = 4 / 5 ∧ tan (2 * α) = 4 / 3 ∧ cos (α + 2 * β) = - (11 * real.sqrt 5) / 25 :=
sorry

end trig_identity_proof_l336_336255


namespace complement_of_M_in_U_is_correct_l336_336523

def U : Set ℤ := {1, -2, 3, -4, 5, -6}
def M : Set ℤ := {1, -2, 3, -4}
def complement_M_in_U : Set ℤ := {5, -6}

theorem complement_of_M_in_U_is_correct : (U \ M) = complement_M_in_U := by
  sorry

end complement_of_M_in_U_is_correct_l336_336523


namespace tangent_circumcircles_l336_336726

noncomputable theory
open_locale classical

variables {α : Type*} [euclidean_geometry α]
variables (A B C F H K M Q : α)
variables (Γ : circle α)

-- Assume the conditions
variables (h_triangle : is_acute_triangle A B C)
          (h_ab_gt_ac : dist A B > dist A C)
          (h_circumcircle : circle_of_triangle A B C Γ)
          (h_orthocenter : orthocenter A B C H)
          (h_foot : foot_of_altitude A B C F)
          (h_midpoint : midpoint (B, C) M)
          (h_Q_on_circle : on_circle Γ Q)
          (h_K_on_circle : on_circle Γ K)
          (h_angles : angle H Q A = 90 ∧ angle H K Q = 90)
          (h_distinct_points : distinct_points_on_circle [A, B, C, K, Q] Γ)

-- Define the proof problem
theorem tangent_circumcircles :
  tangent (circumcircle_of_triangle K Q H) (circumcircle_of_triangle F K M) :=
sorry

end tangent_circumcircles_l336_336726


namespace rectangle_area_is_correct_l336_336168

noncomputable def inscribed_rectangle_area (r : ℝ) (l_to_w_ratio : ℝ) : ℝ :=
  let width := 2 * r
  let length := l_to_w_ratio * width
  length * width

theorem rectangle_area_is_correct :
  inscribed_rectangle_area 7 3 = 588 :=
  by
    -- The proof goes here
    sorry

end rectangle_area_is_correct_l336_336168


namespace bread_cost_l336_336142

theorem bread_cost
  (B : ℝ)
  (cost_peanut_butter : ℝ := 2)
  (initial_money : ℝ := 14)
  (money_leftover : ℝ := 5.25) :
  3 * B + cost_peanut_butter = (initial_money - money_leftover) → B = 2.25 :=
by
  sorry

end bread_cost_l336_336142


namespace triangle_area_expression_l336_336584

-- Define the input parameters for the triangle
variables (a : ℝ) (β γ : ℝ)

-- Define the area function in terms of a, β, γ
def triangle_area (a β γ : ℝ) : ℝ :=
  (a ^ 2 * Real.sin β * Real.sin γ) / (2 * Real.sin (β + γ))

-- State the theorem
theorem triangle_area_expression :
  ∃ S, S = (a ^ 2 * Real.sin β * Real.sin γ) / (2 * Real.sin (β + γ)) :=
begin
  use (a ^ 2 * Real.sin β * Real.sin γ) / (2 * Real.sin (β + γ)),
  sorry,
end

end triangle_area_expression_l336_336584


namespace num_play_both_l336_336898

-- Definitions based on the conditions
def total_members : ℕ := 30
def play_badminton : ℕ := 17
def play_tennis : ℕ := 19
def play_neither : ℕ := 2

-- The statement we want to prove
theorem num_play_both :
  play_badminton + play_tennis - 8 = total_members - play_neither := by
  -- Omitted proof
  sorry

end num_play_both_l336_336898


namespace mark_ate_in_first_four_days_l336_336781

-- Definitions based on conditions
def total_fruit : ℕ := 10
def fruit_kept : ℕ := 2
def fruit_brought_on_friday : ℕ := 3

-- Statement to be proved
theorem mark_ate_in_first_four_days : total_fruit - fruit_kept - fruit_brought_on_friday = 5 := 
by sorry

end mark_ate_in_first_four_days_l336_336781


namespace B_pow_five_l336_336398

def B : Matrix (Fin 2) (Fin 2) ℝ := 
  ![![2, 3], ![4, 6]]
  
theorem B_pow_five : 
  B^5 = (4096 : ℝ) • B + (0 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  sorry

end B_pow_five_l336_336398


namespace part1_part2_part3_l336_336635

open Real

-- Definitions and conditions from the problem
def f (a : ℝ) (x : ℝ) : ℝ := a * sin x

def g (x : ℝ) : ℝ := log x

def G (a : ℝ) (x : ℝ) : ℝ := f a (1 - x) + g x

def F (x : ℝ) (m b : ℝ) : ℝ := exp x - m * x^2 - 2 * (x + 1) + b

-- Statements
theorem part1 (a : ℝ) : (∀ x ∈ Ioo 0 1, (1 / x) - a * cos (1 - x) > 0) → a ≤ 1 :=
sorry

theorem part2 (n : ℕ) : (∀ k, 1 ≤ k → sin (1 - (k^2 + 2*k) / (1 + k)^2) < log ((k + 1)^2 / (k^2 + 2*k))) →
  ∑ k in finset.range n, sin (1 / (1 + k)^2) < log 2 :=
sorry

theorem part3 (m : ℝ) (h_m : m < 0) : (∀ x, F x m b > 0) → (∃ (b : ℤ), b ≥ 3) :=
sorry

end part1_part2_part3_l336_336635


namespace person_walks_distance_l336_336512

theorem person_walks_distance {D t : ℝ} (h1 : 5 * t = D) (h2 : 10 * t = D + 20) : D = 20 :=
by
  sorry

end person_walks_distance_l336_336512


namespace imag_part_of_z_is_neg_one_l336_336289

-- Define the given complex number z
noncomputable def z : ℂ := ((1 + Complex.i) ^ 3) / ((1 - Complex.i) ^ 2)

-- Define the imaginary part of z
def imag_part_of_z : ℂ := z.im

-- Stating the theorem to prove
theorem imag_part_of_z_is_neg_one : imag_part_of_z = -1 := by 
sorry

end imag_part_of_z_is_neg_one_l336_336289


namespace abs_neg_two_plus_exp_l336_336902

theorem abs_neg_two_plus_exp (a : ℝ) (b : ℝ) (c : ℝ) (h1 : a = -2) (h2 : b = 3) (h3 : c = 5) :
  |a| + (b - sqrt c)^0 = 3 :=
by
  sorry

end abs_neg_two_plus_exp_l336_336902


namespace pq_sum_eight_l336_336655

theorem pq_sum_eight
  (p q : ℤ)
  (hp1 : p > 1)
  (hq1 : q > 1)
  (hs1 : (2 * q - 1) % p = 0)
  (hs2 : (2 * p - 1) % q = 0) : p + q = 8 := 
sorry

end pq_sum_eight_l336_336655


namespace triangle_area_formula_l336_336582

variables {a b c S : ℝ} -- Define real numbers representing side lengths and area
variables {α β γ : ℝ}  -- Define real numbers representing angles

-- Given conditions
def sine_rule (a b c : ℝ) (α β γ : ℝ) : Prop := 
  (a / sin α = b / sin β) ∧ (a / sin α = c / sin γ)

def angle_sum (α β γ : ℝ) : Prop :=
  α = β + γ

-- The statement to be proved
theorem triangle_area_formula :
  sine_rule a b c α β γ → angle_sum α β γ → 
  S = a^2 * sin β * sin γ / (2 * sin (β + γ)) :=
by
  sorry

end triangle_area_formula_l336_336582


namespace intersection_on_y_axis_l336_336836

theorem intersection_on_y_axis (k : ℝ) (x y : ℝ) :
  (2 * x + 3 * y - k = 0) →
  (x - k * y + 12 = 0) →
  (x = 0) →
  k = 6 ∨ k = -6 :=
by
  sorry

end intersection_on_y_axis_l336_336836


namespace S_30_value_l336_336831

-- Define the sequence {a_n}
def a_n (n : ℕ) : ℝ := n ^ 2 * (Real.cos (n * Real.pi / 3) ^ 2 - Real.sin (n * Real.pi / 3) ^ 2)

-- Define the sum of the first n terms
def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a_n (i + 1)

-- Prove that S 30 = 470
theorem S_30_value : S 30 = 470 := by
  sorry

end S_30_value_l336_336831


namespace M_in_triangle_DEF_l336_336628

variable {A B C : Type} [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
variable (ΔABC : triangle A B C) {M : point}

-- Let D, E, F be the points where angle bisectors of triangle ABC intersect the sides BC, CA, and AB respectively.
def is_base_of_angle_bisector (D : point) : Prop :=
  -- Definition placeholder for D being the base of the angle bisector from A to BC
  sorry

def inside_triangle_DEF (M : point) (D E F : point) : Prop :=
  -- Definition placeholder for M being inside triangle DEF
  sorry

theorem M_in_triangle_DEF
  (hM_in_triangle_ABC : M ∈ ΔABC)
  (MK_perp_BC : is_perpendicular (line_segment M K) (line_segment B C))
  (ML_perp_CA : is_perpendicular (line_segment M L) (line_segment C A))
  (MN_perp_AB : is_perpendicular (line_segment M N) (line_segment A B))
  (triangle_ABC : triangle A B C)
  (D E F : point)
  (hD_base_bisector : is_base_of_angle_bisector D)
  (hE_base_bisector : is_base_of_angle_bisector E)
  (hF_base_bisector : is_base_of_angle_bisector F)
  (hDEF : triangle D E F) :
  inside_triangle_DEF M D E F :=
sorry

end M_in_triangle_DEF_l336_336628


namespace smallest_positive_e_l336_336088

theorem smallest_positive_e (e : ℕ) (h : e = |-((-3) * 6 * 10 * (-1/4))|) (he : e % 30 = 0) :
  e = 180 :=
by
  sorry

end smallest_positive_e_l336_336088


namespace scout_weekend_earnings_l336_336056

theorem scout_weekend_earnings
  (base_pay_per_hour : ℕ)
  (tip_per_delivery : ℕ)
  (hours_worked_saturday : ℕ)
  (deliveries_saturday : ℕ)
  (hours_worked_sunday : ℕ)
  (deliveries_sunday : ℕ)
  (total_earnings : ℕ)
  (h_base_pay : base_pay_per_hour = 10)
  (h_tip : tip_per_delivery = 5)
  (h_hours_sat : hours_worked_saturday = 4)
  (h_deliveries_sat : deliveries_saturday = 5)
  (h_hours_sun : hours_worked_sunday = 5)
  (h_deliveries_sun : deliveries_sunday = 8) :
  total_earnings = 155 :=
by
  sorry

end scout_weekend_earnings_l336_336056


namespace four_fours_l336_336487

theorem four_fours :
  ∃ e2 e3 e4 e5 e6 : ℕ,
  (e2 = ((4 / 4) + (4 / 4)) ∧
   e3 = (((4 + 4) / 4) + (4 / 4)) ∧
   e4 = (4 + (4 - 4)) ∧
   e5 = ((4 * 4 - 4) / 4) ∧
   e6 = (4 + (4 / 2))) ∧
   e2 = 2 ∧ e3 = 3 ∧ e4 = 4 ∧ e5 = 5 ∧ e6 = 6 :=
begin
  sorry
end

end four_fours_l336_336487


namespace problem_b_is_mapping_l336_336139

def is_function (M P : Set ℕ) (f : ℕ → ℕ) : Prop :=
  ∀ x y ∈ M, f x = f y → x = y

def image_within_set (M P : Set ℕ) (f : ℕ → ℕ) : Prop :=
  ∀ x ∈ M, f x ∈ P

def is_mapping (M P : Set ℕ) (f : ℕ → ℕ) : Prop :=
  is_function M P f ∧ image_within_set M P f

theorem problem_b_is_mapping : 
  is_mapping {0, 1} {-1, 0, 1} (λ x, Int.sqrt x) := 
by
  sorry

end problem_b_is_mapping_l336_336139


namespace num_subsets_conditioned_l336_336604

theorem num_subsets_conditioned :
  (∃ S ⊆ (finset.range 21).sto_set, S.card = 3 ∧
    ∀ a b c ∈ S, (a < b - 1 ∧ b - 1 < c - 3)) →
  (finset.card ((finset.powerset (finset.range 21)).filter (λ T, T.card = 3 ∧
    ∀ a b c ∈ T, a < b - 1 ∧ b - 1 < c - 3))) = 680 := sorry

end num_subsets_conditioned_l336_336604


namespace triangle_point_area_ratios_eq_zero_l336_336265

variables {α β γ : ℝ}
variables {O A B C : Point} -- Assuming Point is a predefined type in the context

-- Area ratios defined as assumptions
variables (hα : α = S_triangle O B C / S_triangle A B C)
variables (hβ : β = S_triangle O C A / S_triangle A B C)
variables (hγ : γ = S_triangle O A B / S_triangle A B C)

theorem triangle_point_area_ratios_eq_zero
  (hO_in_ABC : O ∈ triangle A B C)
  (h_sum : α + β + γ = 1)
  : α • (vector O A) + β • (vector O B) + γ • (vector O C) = 0 :=
sorry

end triangle_point_area_ratios_eq_zero_l336_336265


namespace solve_students_in_fifth_grade_class_l336_336418

noncomputable def number_of_students_in_each_fifth_grade_class 
    (third_grade_classes : ℕ) 
    (third_grade_students_per_class : ℕ)
    (fourth_grade_classes : ℕ) 
    (fourth_grade_students_per_class : ℕ) 
    (fifth_grade_classes : ℕ)
    (total_lunch_cost : ℝ)
    (hamburger_cost : ℝ)
    (carrot_cost : ℝ)
    (cookie_cost : ℝ) : ℝ :=
  
  let total_students_third := third_grade_classes * third_grade_students_per_class
  let total_students_fourth := fourth_grade_classes * fourth_grade_students_per_class
  let lunch_cost_per_student := hamburger_cost + carrot_cost + cookie_cost
  let total_students := total_students_third + total_students_fourth
  let total_cost_third_fourth := total_students * lunch_cost_per_student
  let total_cost_fifth := total_lunch_cost - total_cost_third_fourth
  let fifth_grade_students := total_cost_fifth / lunch_cost_per_student
  let students_per_fifth_class := fifth_grade_students / fifth_grade_classes
  students_per_fifth_class

theorem solve_students_in_fifth_grade_class : 
    number_of_students_in_each_fifth_grade_class 5 30 4 28 4 1036 2.10 0.50 0.20 = 27 := 
by 
  sorry

end solve_students_in_fifth_grade_class_l336_336418


namespace entropy_of_independent_variables_eq_sum_l336_336753

noncomputable def H (probs : List ℝ) : ℝ :=
  - (probs.map (λ p => p * log p / log 2)).sum

theorem entropy_of_independent_variables_eq_sum 
  {k l : ℕ} 
  (p : Fin k → ℝ) 
  (q : Fin l → ℝ) 
  (hp : ∀ i, 0 ≤ p i ∧ p i ≤ 1)
  (hq : ∀ j, 0 ≤ q j ∧ q j ≤ 1) 
  (hp_sum : (Finset.univ : Finset (Fin k)).sum p = 1)
  (hq_sum : (Finset.univ : Finset (Fin l)).sum q = 1) :
  H ((Finset.univ.product Finset.univ).map (λ ⟨i, j⟩ => p i * q j)) = 
  H ((Finset.univ.map p).val) + H ((Finset.univ.map q).val) :=
sorry

end entropy_of_independent_variables_eq_sum_l336_336753


namespace prove_expression_l336_336317

noncomputable def sine_cosine_relation (α m : ℝ) : Prop :=
  sin α - cos α = m

noncomputable def expression (α : ℝ) : ℝ :=
  (sin (4 * α) + sin (10 * α) - sin (6 * α)) / (cos (2 * α) + 1 - 2 * (sin (4 * α))^2)

theorem prove_expression (α m : ℝ) (h : sine_cosine_relation α m) : 
  expression α = 2 * (1 - m^2) :=
by
  sorry

end prove_expression_l336_336317


namespace hyperbola_focal_length_ellipse_foci_parabola_focus_directrix_probability_same_color_l336_336522

-- Problem 1
theorem hyperbola_focal_length (m : ℝ) : 
  (∀ x y : ℝ, x^2 / 9 - y^2 / m = 1 → (x^2 + y^2 = 25)) → m = 16 :=
by
  sorry

-- Problem 2
theorem ellipse_foci : 
  let a := 4 in 
  let b := 5 in 
  let c := √(b^2 - a^2) in 
  c = 3 ∧ 
  (∀ (x y : ℝ), x^2 / 16 + y^2 / 25 = 1 → (x, y) ∈ {(0, -c), (0, c)}) :=
by
  sorry

-- Problem 3
theorem parabola_focus_directrix :
  let c := 5 in 
  (∀ y x : ℝ, y^2 = 10 * x → 
    let p := c in 
    p = 5) :=
by
  sorry

-- Problem 4
theorem probability_same_color :
 (let choices := 3 in 
  let total_choices := choices * choices in 
  let favorable := choices in 
  let probability := favorable / total_choices in 
  probability = 1 / 3) :=
by
  sorry

end hyperbola_focal_length_ellipse_foci_parabola_focus_directrix_probability_same_color_l336_336522


namespace joe_max_money_l336_336441

noncomputable def max_guaranteed_money (initial_money : ℕ) (max_bet : ℕ) (num_bets : ℕ) : ℕ :=
  if initial_money = 100 ∧ max_bet = 17 ∧ num_bets = 5 then 98 else 0

theorem joe_max_money : max_guaranteed_money 100 17 5 = 98 := by
  sorry

end joe_max_money_l336_336441


namespace least_clock_equivalent_l336_336423

def clock_equivalent (a b : ℕ) : Prop :=
  ∃ k : ℕ, a + 12 * k = b

theorem least_clock_equivalent (h : ℕ) (hh : h > 3) (hq : clock_equivalent h (h * h)) :
  h = 4 :=
by
  sorry

end least_clock_equivalent_l336_336423


namespace exists_rightline_no_multiples_of_3_inf_disjoint_rightlines_no_multiples_of_3_l336_336096

-- Defining the condition of the numbered squares on a spiral chessboard
def numbered_squares (n : ℕ) : ℕ := n
-- A rightline is a sequence of numbers starting from a square and going to the right
def rightline (start : ℕ) : (ℕ → ℕ) := λ n, start + n

-- Problem part (a): Proving the existence of a rightline without multiples of 3
theorem exists_rightline_no_multiples_of_3 :
  ∃ start, ∀ n, ¬ (rightline (6 * n + 1) start % 3 = 0) :=
sorry

-- Problem part (b): Proving there are infinitely many pairwise disjoint rightlines
theorem inf_disjoint_rightlines_no_multiples_of_3 :
  ∃₀ (start : ℕ) in (range (6 * ℕ + 1)), ∀ n, ∃ k ≠ n, rightline (6 * k + 1) start % 3 ≠ 0 :=
sorry

end exists_rightline_no_multiples_of_3_inf_disjoint_rightlines_no_multiples_of_3_l336_336096


namespace frequency_of_group_samples_l336_336182

-- Conditions
def sample_capacity : ℕ := 32
def group_frequency : ℝ := 0.125

-- Theorem statement
theorem frequency_of_group_samples : group_frequency * sample_capacity = 4 :=
by sorry

end frequency_of_group_samples_l336_336182


namespace magnitude_of_complex_expression_l336_336638

theorem magnitude_of_complex_expression : 
  let z := (1 : ℂ) + (Complex.I) in
  complex.abs (z^2 + z) = Real.sqrt 10 := 
by 
  sorry

end magnitude_of_complex_expression_l336_336638


namespace scientific_notation_of_number_l336_336360

theorem scientific_notation_of_number :
  (0.0000000033 : ℝ) = 3.3 * 10^(-9) :=
sorry

end scientific_notation_of_number_l336_336360


namespace percentage_income_put_aside_l336_336106

theorem percentage_income_put_aside (
    cost1 cost2 cost3 : ℝ,
    hourly_wage : ℝ,
    hours_worked : ℝ,
    total_cost : ℝ := cost1 + cost2 + cost3,
    total_earned : ℝ := hourly_wage * hours_worked,
    amount_put_aside : ℝ := total_earned - total_cost,
    percentage_put_aside : ℝ := (amount_put_aside / total_earned) * 100
) : 
cost1 = 25.35 → cost2 = 70.69 → cost3 = 85.96 → 
hourly_wage = 6.50 → hours_worked = 31 → 
percentage_put_aside = 9.68 :=
by 
  intros 
  sorry

end percentage_income_put_aside_l336_336106


namespace find_missing_angle_l336_336542

theorem find_missing_angle 
  (A B C D E F : Point) 
  (α β γ δ ε ζ η θ ι κ λ μ : ℝ)
  (H1 : α = 15)
  (H2 : β = 20)
  (H3 : γ = 20)
  (H4 : δ = 50)
  (H5 : ε = 55)
  (H6 : ζ = 70)
  (H7 : η = 75)
  (H8 : θ = 75)
  (H9 : ι = 90)
  (H10 : κ = 90)
  (H11 : λ = 130)
  (H12 : A + B + C + D = 360)
  (H13 : α + β + γ + δ + ε + ζ + η + θ + ι + κ + λ + μ = 180 * 4)
  (H14 : segment_length BE > segment_length FC) :
  μ = 30 := 
sorry

end find_missing_angle_l336_336542


namespace smallest_a1_pos_l336_336762

theorem smallest_a1_pos (a : ℕ → ℝ) (h₁ : ∀ n > 1, a n = 9 * a (n - 1) - 2 * n) (h₂ : ∀ n, a n > 0) :
  a 1 ≥ 1 / 2 :=
begin
  sorry  -- Proof is omitted as per instructions.
end

end smallest_a1_pos_l336_336762


namespace solve_for_y_l336_336813


theorem solve_for_y : ∀ (y : ℝ), 3^(2*y + 3) = 27^(y - 1) → y = 6 :=
by
  -- sorry is used here to skip the proof
  sorry

end solve_for_y_l336_336813


namespace distance_to_plane_of_triangle_l336_336928

-- Define the sphere with center O and radius 8
variable {O : Type} [EuclideanGeometry O]
variable (radius : ℝ)
variable (sphere_center : O)

-- Define the triangle with sides 13, 13, 10 tangent to the sphere
variable {A B C : O}
variable (triangle_AB : ℝ)
variable (triangle_BC : ℝ)
variable (triangle_CA : ℝ)
variable (A_tangent_sphere : Tangent A sphere_center radius)
variable (B_tangent_sphere : Tangent B sphere_center radius)
variable (C_tangent_sphere : Tangent C sphere_center radius)

-- Assume the sides of the triangle
variable (side_AB : triangle_AB = 13)
variable (side_BC : triangle_BC = 10)
variable (side_CA : triangle_CA = 13)

-- Define the function to calculate the distance from sphere center to the plane of the triangle
def distance_from_sphere_center_to_plane (O : O) (A B C : O) : ℝ :=
  sorry  -- This will be the calculation part

-- Hypotheses
axiom radius_definition : radius = 8
axiom tangent_definitions : A_tangent_sphere ∧ B_tangent_sphere ∧ C_tangent_sphere

-- Prove that the distance from O to the triangle's plane is 2 * sqrt(119) / 3
theorem distance_to_plane_of_triangle : distance_from_sphere_center_to_plane sphere_center A B C = 2 * Real.sqrt 119 / 3 := 
sorry

end distance_to_plane_of_triangle_l336_336928


namespace triangle_split_points_l336_336362

noncomputable def smallest_n_for_split (AB BC CA : ℕ) : ℕ := 
  if AB = 13 ∧ BC = 14 ∧ CA = 15 then 27 else sorry

theorem triangle_split_points (AB BC CA : ℕ) (h : AB = 13 ∧ BC = 14 ∧ CA = 15) :
  smallest_n_for_split AB BC CA = 27 :=
by
  cases h with | intro h1 h23 => sorry

-- Assertions for the explicit values provided in the conditions
example : smallest_n_for_split 13 14 15 = 27 :=
  triangle_split_points 13 14 15 ⟨rfl, rfl, rfl⟩

end triangle_split_points_l336_336362


namespace combined_tax_rate_l336_336510

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) :
  let Mork_income := X in
  let Mindy_income := 4 * X in
  let Mork_tax := 0.4 * Mork_income in
  let Mindy_tax := 0.3 * Mindy_income in
  let combined_income := Mork_income + Mindy_income in
  let combined_tax := Mork_tax + Mindy_tax in
  let combined_tax_rate := (combined_tax / combined_income) * 100 in
  combined_tax_rate = 32 := 
sorry

end combined_tax_rate_l336_336510


namespace mark_ate_in_first_four_days_l336_336780

-- Definitions based on conditions
def total_fruit : ℕ := 10
def fruit_kept : ℕ := 2
def fruit_brought_on_friday : ℕ := 3

-- Statement to be proved
theorem mark_ate_in_first_four_days : total_fruit - fruit_kept - fruit_brought_on_friday = 5 := 
by sorry

end mark_ate_in_first_four_days_l336_336780


namespace problem1_problem2_l336_336211

theorem problem1 : sqrt 4 - 3 = -1 := 
by
  sorry

theorem problem2 : (1 / 2) * 2 + real.cbrt 8 - abs (1 - sqrt 9) = 1 := 
by
  sorry

end problem1_problem2_l336_336211


namespace kendra_shirts_needed_l336_336385

def shirts_needed_per_week (school_days after_school_club_days saturday_shirts sunday_church_shirt sunday_rest_of_day_shirt : ℕ) : ℕ :=
  school_days + after_school_club_days + saturday_shirts + sunday_church_shirt + sunday_rest_of_day_shirt

def shirts_needed (weeks shirts_per_week : ℕ) : ℕ :=
  weeks * shirts_per_week

theorem kendra_shirts_needed : shirts_needed 2 (
  shirts_needed_per_week 5 3 1 1 1
) = 22 :=
by
  simp [shirts_needed, shirts_needed_per_week]
  rfl

end kendra_shirts_needed_l336_336385


namespace difference_of_extremes_from_digits_l336_336490

theorem difference_of_extremes_from_digits :
  let digits := [9, 2, 1, 5]
  let largest := list.to_nat (list.reverse (list.sort digits))
  let smallest := list.to_nat (list.sort digits)
  in largest - smallest = 8262 := 
by 
  sorry

end difference_of_extremes_from_digits_l336_336490


namespace binomial_12_6_eq_924_l336_336959

theorem binomial_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binomial_12_6_eq_924_l336_336959


namespace Mrs_Smith_probability_l336_336031

noncomputable def probability_more_sons_or_daughters 
          (children : ℕ) 
          (prob_twins : ℚ) 
          (prob_gender : ℚ) 
          (P : ℚ)
          (n : ℕ → Prop) 
          (binom : ℕ → ℕ → ℕ) : Prop :=
  children = 8 ∧ 
  prob_twins = 1/5 ∧ 
  prob_gender = 1/2 ∧ 
  P = 1 - ∑ i in (finset.range (children + 1)), 
           if even i then (binom i (i/2)) * prob_twins^i * prob_gender^(children - i) else 0

theorem Mrs_Smith_probability 
  (children : ℕ) 
  (prob_twins : ℚ) 
  (prob_gender : ℚ) 
  (P : ℚ)
  (n : ℕ → Prop) 
  (binom : ℕ → ℕ → ℕ) 
  (h : probability_more_sons_or_daughters children prob_twins prob_gender P n binom):
  Mrs_Smith_probability :=
begin
  sorry
end

end Mrs_Smith_probability_l336_336031


namespace students_not_receiving_A_l336_336349

theorem students_not_receiving_A (total_students : ℕ) (students_A_physics : ℕ) (students_A_chemistry : ℕ) (students_A_both : ℕ) (h_total : total_students = 40) (h_A_physics : students_A_physics = 10) (h_A_chemistry : students_A_chemistry = 18) (h_A_both : students_A_both = 6) : (total_students - ((students_A_physics + students_A_chemistry) - students_A_both)) = 18 := 
by
  sorry

end students_not_receiving_A_l336_336349


namespace kendra_shirts_needed_l336_336387

def shirts_needed_per_week (school_days after_school_club_days saturday_shirts sunday_church_shirt sunday_rest_of_day_shirt : ℕ) : ℕ :=
  school_days + after_school_club_days + saturday_shirts + sunday_church_shirt + sunday_rest_of_day_shirt

def shirts_needed (weeks shirts_per_week : ℕ) : ℕ :=
  weeks * shirts_per_week

theorem kendra_shirts_needed : shirts_needed 2 (
  shirts_needed_per_week 5 3 1 1 1
) = 22 :=
by
  simp [shirts_needed, shirts_needed_per_week]
  rfl

end kendra_shirts_needed_l336_336387


namespace tan_α_value_l336_336279

noncomputable def α : ℝ := sorry  -- α is a real number
axiom α_interval : -π / 2 < α ∧ α < 0
axiom cos_α : Real.cos α = 3 / 5

theorem tan_α_value : Real.tan α = -4 / 3 :=
by
  sorry

end tan_α_value_l336_336279


namespace divisibility_by_n_l336_336448

variable (a b c : ℤ) (n : ℕ)

theorem divisibility_by_n
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 2 * n + 1) :
  ∃ k : ℤ, a^3 + b^2 - a^2 - b^3 = k * ↑n := 
sorry

end divisibility_by_n_l336_336448


namespace kekai_remaining_money_l336_336379

-- Definitions based on given conditions
def shirts_sold := 5
def price_per_shirt := 1
def pants_sold := 5
def price_per_pant := 3
def half_fraction := 1 / 2 : ℝ

-- Proving that Kekai's remaining money is $10
theorem kekai_remaining_money : 
  let earnings_from_shirts := shirts_sold * price_per_shirt in
  let earnings_from_pants := pants_sold * price_per_pant in
  let total_earnings := earnings_from_shirts + earnings_from_pants in
  let money_given_to_parents := total_earnings * half_fraction in
  let remaining_money := total_earnings - money_given_to_parents in
  remaining_money = 10 :=
by
  sorry

end kekai_remaining_money_l336_336379


namespace binomial_12_6_eq_1848_l336_336972

theorem binomial_12_6_eq_1848 : (Nat.choose 12 6) = 1848 :=
  sorry

end binomial_12_6_eq_1848_l336_336972


namespace darryl_made_85_dollars_l336_336217

-- Definitions for conditions
variables (started_cantaloupes started_honeydews cantaloupes_dropped honeydews_rotten cantaloupes_left honeydews_left : ℕ)
variables (price_cantaloupe price_honeydew : ℕ)

-- Actual conditions from the problem
def conditions := 
  started_cantaloupes = 30 ∧
  started_honeydews = 27 ∧
  cantaloupes_dropped = 2 ∧
  honeydews_rotten = 3 ∧
  cantaloupes_left = 8 ∧
  honeydews_left = 9 ∧
  price_cantaloupe = 2 ∧
  price_honeydew = 3

-- Calculation of sales and total money made
def money_made (started_cantaloupes : ℕ) (started_honeydews : ℕ)
               (cantaloupes_dropped : ℕ) (honeydews_rotten : ℕ)
               (cantaloupes_left : ℕ) (honeydews_left : ℕ)
               (price_cantaloupe : ℕ) (price_honeydew : ℕ) :=
  let cantaloupes_sold := started_cantaloupes - cantaloupes_dropped - cantaloupes_left in
  let honeydews_sold := started_honeydews - honeydews_rotten - honeydews_left in
  (cantaloupes_sold * price_cantaloupe) + (honeydews_sold * price_honeydew)

-- Statement of the theorem
theorem darryl_made_85_dollars (h : conditions) :
  money_made started_cantaloupes started_honeydews
             cantaloupes_dropped honeydews_rotten
             cantaloupes_left honeydews_left
             price_cantaloupe price_honeydew = 85 :=
by
  sorry

end darryl_made_85_dollars_l336_336217


namespace range_of_b_l336_336633

def f (x : ℝ) : ℝ :=
  if x < -1/2 then (2 * x + 1) / x^2 else Real.log (x + 1)

def g (x : ℝ) : ℝ :=
  x^2 - 4 * x - 4

theorem range_of_b (a b : ℝ)
  (h : f a + g b = 0) :
  -1 ≤ b ∧ b ≤ 5 :=
sorry

end range_of_b_l336_336633


namespace xy_sum_l336_336333

variable (x y : ℚ)

theorem xy_sum : (1/x + 1/y = 4) → (1/x - 1/y = -6) → x + y = -4/5 := by
  intros h1 h2
  sorry

end xy_sum_l336_336333


namespace lunch_total_price_l336_336844

theorem lunch_total_price:
  ∀ (P T : ℝ), 
    (P = 15 * 12) → 
    (T = 1.15 * P) → 
    T = 207 :=
by
  intros P T hP hT
  rw [hP] at hT
  linarith

end lunch_total_price_l336_336844


namespace lattice_points_count_l336_336175

def is_lattice_point (p : ℤ × ℤ) : Prop := true  -- Every integer coordinate point is a lattice point

def on_segment (p : ℤ × ℤ) (A B : ℤ × ℤ) : Prop :=
  ∃ (t : ℚ), (0 ≤ t ∧ t ≤ 1) ∧ p.1 = A.1 + t * (B.1 - A.1) ∧ p.2 = A.2 + t * (B.2 - A.2)

def lattice_points_on_segment (A B : ℤ × ℤ) : ℕ :=
  (finset.filter (λ p, is_lattice_point p ∧ on_segment p A B)
                 ((finset.Icc A.1 B.1).product (finset.Icc A.2 B.2))).card

/-- There are 4 lattice points on the line segment connecting (3, 17) and (48, 281). -/
theorem lattice_points_count : lattice_points_on_segment (3, 17) (48, 281) = 4 := sorry

end lattice_points_count_l336_336175


namespace difference_of_squares_65_55_l336_336567

theorem difference_of_squares_65_55 : (65^2 - 55^2 = 1200) :=
by
  let a := 65
  let b := 55
  have h1 : a^2 - b^2 = (a + b) * (a - b) := by exact Nat.sub_square (a) (b)
  have h2 : (a + b) = 120 := by rfl
  have h3 : (a - b) = 10 := by rfl
  rw [h1, h2, h3]
  norm_num
  sorry

end difference_of_squares_65_55_l336_336567


namespace cauchy_problem_solution_l336_336900

noncomputable def x_solution (t : ℝ) : ℝ := 4 * Real.exp t + 2 * Real.exp (-t)
noncomputable def y_solution (t : ℝ) : ℝ := -Real.exp t - Real.exp (-t)

theorem cauchy_problem_solution :
  (∀ t : ℝ, (differential.differential (λ t, x_solution t) = (λ t, 3 * x_solution t + 8 * y_solution t) t) ∧
            (differential.differential (λ t, y_solution t) = (λ t, -(x_solution t) - 3 * y_solution t) t)) ∧
  (x_solution 0 = 6) ∧
  (y_solution 0 = -2) :=
by
  sorry

end cauchy_problem_solution_l336_336900


namespace security_mistakes_and_measures_l336_336892

structure UserAction :=
  (received_email : String)
  (email_link : String)
  (entered_info : Bool)
  (bank_sms : String)
  (purchase_email : String)
  (temporary_block_lifted : Bool)

def suspiciousEmail (email : String) : Bool :=
  email.contains "aliexpress@best_prices.net"

def suspiciousLink (link : String) : Bool :=
  link.contains "aliexpres__best_prices.net"

def trustedUnverifiedEmail (email : String) (info_entered : Bool) : Prop :=
  suspiciousEmail email ∧ info_entered

def trustedUnverifiedLink (link : String) (info_entered : Bool) : Prop :=
  suspiciousLink link ∧ info_entered

def unusualOffer (email : String) (price_reduction : Int) : Prop :=
  email.contains "won a lottery" ∧ price_reduction > 35000

def didNotVerify (email : String) (link : String) (info_entered : Bool) : Prop :=
  info_entered ∧ suspiciousEmail email ∧ suspiciousLink link

def is_action_secure (user_action : UserAction) : Prop :=
  ¬trustedUnverifiedEmail user_action.received_email user_action.entered_info ∧
  ¬trustedUnverifiedLink user_action.email_link user_action.entered_info ∧
  ¬unusualOffer user_action.received_email 38990 ∧
  ¬didNotVerify user_action.received_email user_action.email_link user_action.entered_info

theorem security_mistakes_and_measures (user_action : UserAction) :
  is_action_secure user_action :=
sorry

end security_mistakes_and_measures_l336_336892


namespace at_least_one_less_than_equal_one_l336_336022

theorem at_least_one_less_than_equal_one
  (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 := 
by 
  sorry

end at_least_one_less_than_equal_one_l336_336022


namespace probability_at_least_one_multiple_of_4_is_correct_l336_336191

noncomputable def probability_at_least_one_multiple_of_4 : ℚ :=
  let total_numbers := 100
  let multiples_of_4 := 25
  let non_multiples_of_4 := total_numbers - multiples_of_4
  let p_non_multiple := (non_multiples_of_4 : ℚ) / total_numbers
  let p_both_non_multiples := p_non_multiple^2
  let p_at_least_one_multiple := 1 - p_both_non_multiples
  p_at_least_one_multiple

theorem probability_at_least_one_multiple_of_4_is_correct :
  probability_at_least_one_multiple_of_4 = 7 / 16 :=
by
  sorry

end probability_at_least_one_multiple_of_4_is_correct_l336_336191


namespace find_x_l336_336904

theorem find_x (x : ℝ) : 9 - (x / (1 / 3)) + 3 = 3 → x = 3 := by
  intro h
  sorry

end find_x_l336_336904


namespace find_urn_yellow_balls_l336_336555

theorem find_urn_yellow_balls :
  ∃ (M : ℝ), 
    (5 / 12) * (20 / (20 + M)) + (7 / 12) * (M / (20 + M)) = 0.62 ∧ 
    M = 111 := 
sorry

end find_urn_yellow_balls_l336_336555


namespace tan_alpha_plus_pi_over_4_l336_336256

theorem tan_alpha_plus_pi_over_4 
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  Real.tan (α + Real.pi / 4) = 3 / 22 :=
sorry

end tan_alpha_plus_pi_over_4_l336_336256


namespace kendra_shirts_for_two_weeks_l336_336383

def school_days := 5
def after_school_club_days := 3
def one_week_shirts := school_days + after_school_club_days + 1 (Saturday) + 2 (Sunday)
def two_weeks_shirts := one_week_shirts * 2

theorem kendra_shirts_for_two_weeks : two_weeks_shirts = 22 :=
by
  -- Prove the theorem
  sorry

end kendra_shirts_for_two_weeks_l336_336383


namespace integral_sin4_cos4_l336_336561

theorem integral_sin4_cos4 :
  ∫ x in 0..(2 * Real.pi), (sin x) ^ 4 * (cos x) ^ 4 = 3 * Real.pi / 64 := sorry

end integral_sin4_cos4_l336_336561


namespace distance_is_one_l336_336086

noncomputable def distance_between_bisectors_and_centroid : ℝ :=
  let AB := 9
  let AC := 12
  let BC := Real.sqrt (AB^2 + AC^2)
  let CD := BC / 2
  let CE := (2/3) * CD
  let r := (AB * AC) / (2 * (AB + AC + BC) / 2)
  let K := CE - r
  K

theorem distance_is_one : distance_between_bisectors_and_centroid = 1 :=
  sorry

end distance_is_one_l336_336086


namespace stratified_sampling_sichuan_university_l336_336549

theorem stratified_sampling_sichuan_university (total_sichuan : ℕ) (total_uestc : ℕ)
  (selected_students : ℕ) (sichuan_ratio : ℕ) (uestc_ratio : ℕ)  
  (h1 : total_sichuan = 25)
  (h2 : total_uestc = 15)
  (h3 : selected_students = 16)
  (h4 : sichuan_ratio = 5)
  (h5 : uestc_ratio = 3) :
  let total_students := total_sichuan + total_uestc in
  let total_ratio := sichuan_ratio + uestc_ratio in
  total_students = 40 ∧
  total_ratio = 8 →
  (selected_students * sichuan_ratio / total_ratio = 10) :=
by
  intros
  sorry

end stratified_sampling_sichuan_university_l336_336549


namespace quadratic_passes_through_neg3_n_l336_336447

-- Definition of the quadratic function with given conditions
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions provided in the problem
variables {a b c : ℝ}
axiom max_at_neg2 : ∀ x, quadratic a b c x ≤ 8
axiom value_at_neg2 : quadratic a b c (-2) = 8
axiom passes_through_1_4 : quadratic a b c 1 = 4

-- Statement to prove
theorem quadratic_passes_through_neg3_n : quadratic a b c (-3) = 68 / 9 :=
sorry

end quadratic_passes_through_neg3_n_l336_336447


namespace product_divisible_by_13_l336_336001

/-- Given that the sum of the 12th powers of a, b, c, d, e, and f is divisible by 13,
    prove that their product is divisible by 13^6. -/
theorem product_divisible_by_13^6
  (a b c d e f : ℤ)
  (h : (a ^ 12 + b ^ 12 + c ^ 12 + d ^ 12 + e ^ 12 + f ^ 12) % 13 = 0) :
  (a * b * c * d * e * f) % 13^6 = 0 :=
  sorry

end product_divisible_by_13_l336_336001


namespace part_a_no_solutions_part_a_infinite_solutions_l336_336509

theorem part_a_no_solutions (a : ℝ) (x y : ℝ) : 
    a = -1 → ¬(∃ x y : ℝ, a * x + y = a^2 ∧ x + a * y = 1) :=
sorry

theorem part_a_infinite_solutions (a : ℝ) (x y : ℝ) : 
    a = 1 → ∃ x : ℝ, ∃ y : ℝ, a * x + y = a^2 ∧ x + a * y = 1 :=
sorry

end part_a_no_solutions_part_a_infinite_solutions_l336_336509


namespace mark_ate_fruit_first_four_days_l336_336782

theorem mark_ate_fruit_first_four_days (total_fruit : ℕ) (kept_for_next_week : ℕ) (brought_to_school : ℕ) :
  total_fruit = 10 → kept_for_next_week = 2 → brought_to_school = 3 → 
  (total_fruit - kept_for_next_week - brought_to_school) = 5 :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
end

end mark_ate_fruit_first_four_days_l336_336782


namespace equivalent_prod_of_squares_l336_336206

theorem equivalent_prod_of_squares :
  (250 * 9.996 * 3.996 * 500) = (4998) ^ 2 := 
begin
  sorry
end

end equivalent_prod_of_squares_l336_336206


namespace correct_equation_l336_336995

variables (x : ℝ)
noncomputable def growth_rate_equation : Prop := 7.5 * (1 + x)^2 = 8.4

theorem correct_equation
  (price_june : 7.5)
  (price_august : 8.4)
  (average_growth : x) :
  growth_rate_equation x :=
by 
  sorry

end correct_equation_l336_336995


namespace simplify_cube_root_18_24_30_l336_336435

noncomputable def cube_root_simplification (a b c : ℕ) : ℕ :=
  let sum_cubes := a^3 + b^3 + c^3
  36

theorem simplify_cube_root_18_24_30 : 
  cube_root_simplification 18 24 30 = 36 :=
by {
  -- Proof steps would go here
  sorry
}

end simplify_cube_root_18_24_30_l336_336435


namespace car_distance_l336_336163

variable (T_initial : ℕ) (T_new : ℕ) (S : ℕ) (D : ℕ)

noncomputable def calculate_distance (T_initial T_new S : ℕ) : ℕ :=
  S * T_new

theorem car_distance :
  T_initial = 6 →
  T_new = (3 / 2) * T_initial →
  S = 16 →
  D = calculate_distance T_initial T_new S →
  D = 144 :=
by
  sorry

end car_distance_l336_336163


namespace red_ants_count_l336_336105

def total_ants : ℕ := 900
def black_ants : ℕ := 487
def red_ants (r : ℕ) : Prop := r + black_ants = total_ants

theorem red_ants_count : ∃ r : ℕ, red_ants r ∧ r = 413 := 
sorry

end red_ants_count_l336_336105


namespace range_of_a_l336_336647

variable (a : ℝ)
def is_second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0
def z : ℂ := 4 - 2 * Complex.I

theorem range_of_a (ha : is_second_quadrant ((z + a * Complex.I) ^ 2)) : a > 6 := by
  sorry

end range_of_a_l336_336647


namespace standard_equation_of_ellipse_line_pass_through_fixed_point_l336_336358

-- Step 1: Defining the problem conditions and statements

-- Defining ellipses and conditions
def ellipse_eq (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def point_on_ellipse_A (a b : ℝ) : Prop :=
  ellipse_eq 1 (3/2) a b

-- Given focal length condition
def focal_length_condition (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (a^2 - b^2 = 1)

-- Defining orthogonality condition
def dot_product_condition (OM ON : ℝ × ℝ) : Prop :=
  (OM.1 * ON.1) + (OM.2 * ON.2) = (9 / 4)

-- Line passing through a fixed point condition
def line_passing_through_fixed_point (k m : ℝ) : Prop :=
  (1, 3/2) = (k * y + m, y)

-- Proving standard equation of the ellipse
theorem standard_equation_of_ellipse :
  ∃ a b, focal_length_condition a b ∧ point_on_ellipse_A a b →
  ellipse_eq x y 2 (√3) :=
sorry

-- Proving line passes through a fixed point
theorem line_pass_through_fixed_point (k m : ℝ) :
  ∀ P Q : ℝ × ℝ, (point_on_ellipse P.1 P.2 2 (√3)) ∧ (point_on_ellipse Q.1 Q.2 2 (√3)) →
  (dot_product_condition (P.1, k * P.2 + m) (Q.1, k * Q.2 + m)) →
  (line_passing_through_fixed_point k m ∨ line_passing_through_fixed_point k m) :=
sorry

end standard_equation_of_ellipse_line_pass_through_fixed_point_l336_336358


namespace first_year_after_2009_with_property_l336_336855

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def no_permutation_smaller (n : ℕ) : Prop :=
  n.digits 10 ≠ [] ∧ ∀ l ∈ (List.permutations (n.digits 10)),
    l.head ≠ 0 → List.to_nat l ≥ n

theorem first_year_after_2009_with_property : ∃ (n : ℕ), 2009 < n ∧ is_four_digit n ∧ no_permutation_smaller n ∧
  (∀ m, 2009 < m ∧ m < n → ¬ no_permutation_smaller m) := 2022

end first_year_after_2009_with_property_l336_336855


namespace leastCookies_l336_336420

theorem leastCookies (b : ℕ) :
  (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) →
  b = 179 :=
by
  sorry

end leastCookies_l336_336420


namespace find_a_14_l336_336663

variable {α : Type} [LinearOrderedField α]

-- Define the arithmetic sequence sum formula
def arithmetic_seq_sum (a_1 d : α) (n : ℕ) : α :=
  n * a_1 + n * (n - 1) / 2 * d

-- Define the nth term of an arithmetic sequence
def arithmetic_seq_nth (a_1 d : α) (n : ℕ) : α :=
  a_1 + (n - 1 : ℕ) * d

theorem find_a_14
  (a_1 d : α)
  (h1 : arithmetic_seq_sum a_1 d 11 = 55)
  (h2 : arithmetic_seq_nth a_1 d 10 = 9) :
  arithmetic_seq_nth a_1 d 14 = 13 :=
by
  sorry

end find_a_14_l336_336663


namespace min_value_of_a_k_l336_336660

-- Define the conditions for our proof in Lean

-- a_n is a positive arithmetic sequence
def is_positive_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ d, ∀ m, a (m + 1) = a m + d

-- Given inequality condition for the sequence
def inequality_condition (a : ℕ → ℝ) (k : ℕ) : Prop :=
  k ≥ 2 ∧ (1 / a 1 + 4 / a (2 * k - 1) ≤ 1)

-- Prove the minimum value of a_k
theorem min_value_of_a_k (a : ℕ → ℝ) (k : ℕ) (h_arith : is_positive_arithmetic_seq a) (h_ineq : inequality_condition a k) :
  a k = 9 / 2 :=
sorry

end min_value_of_a_k_l336_336660


namespace overtaking_time_l336_336144

variable (a_speed b_speed k_speed : ℕ)
variable (b_delay : ℕ) 
variable (t : ℕ)
variable (t_k : ℕ)

theorem overtaking_time (h1 : a_speed = 30)
                        (h2 : b_speed = 40)
                        (h3 : k_speed = 60)
                        (h4 : b_delay = 5)
                        (h5 : 30 * t = 40 * (t - 5))
                        (h6 : 30 * t = 60 * t_k)
                         : k_speed / 3 = 10 :=
by sorry

end overtaking_time_l336_336144


namespace matching_probability_l336_336589

-- Definitions
variables (adults : ℕ) (left_shoes right_shoes : ℕ) (k : ℕ)

-- Conditions
def conditions : Prop :=
  adults = 15 ∧ left_shoes = 15 ∧ right_shoes = 15

-- Probability P that for every k < 7, no collection of k pairs contains shoes from exactly k adults
def probability_P (P : ℚ) : Prop :=
  (∀ k, k < 7 → P = 1 / 15)

-- Main statement
theorem matching_probability : 
  conditions ∧ (∃ P : ℚ, probability_P P) →
  ∃ m n : ℕ, nat.coprime m n ∧ m + n = 16 :=
begin
  sorry
end

end matching_probability_l336_336589


namespace problem_1_problem_2_l336_336672

-- Define the function f(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.exp x - a * Real.exp (2 * x)

-- Problem I: Prove that f(x) ≤ 0 when a ≥ 1 / Real.exp 1
theorem problem_1 (a : ℝ) (h : a ≥ 1 / Real.exp 1) (x : ℝ) : f x a ≤ 0 := 
sorry

-- Problem II: If the function f(x) has two extreme points, find the range of values for the real number a.
theorem problem_2 (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f' x₁ a = 0 ∧ f' x₂ a = 0) → 0 < a ∧ a < 1 / 2 := 
sorry

end problem_1_problem_2_l336_336672


namespace jacobs_hourly_wage_l336_336371

theorem jacobs_hourly_wage (jake_total_earnings : ℕ) (jake_days : ℕ) (hours_per_day : ℕ) (jake_thrice_jacob : ℕ) 
    (h_total_jake : jake_total_earnings = 720) 
    (h_jake_days : jake_days = 5) 
    (h_hours_per_day : hours_per_day = 8)
    (h_jake_thrice_jacob : jake_thrice_jacob = 3) 
    (jacob_hourly_wage : ℕ) :
  jacob_hourly_wage = 6 := 
by
  sorry

end jacobs_hourly_wage_l336_336371


namespace fewest_keystrokes_to_400_through_50_l336_336528

def next_steps (n : ℕ) : list ℕ :=
  [n + 1, 2 * n]

def min_keystrokes (start target : ℕ) (mandatory : ℕ) : ℕ :=
  sorry -- Function involving a search algorithm to determine the minimum number of keystrokes.

theorem fewest_keystrokes_to_400_through_50 : min_keystrokes 1 400 50 = 10 :=
  sorry

end fewest_keystrokes_to_400_through_50_l336_336528


namespace show_inequalities_l336_336551

noncomputable def polynomial_has_negative_real_roots (a b c d : ℝ) : Prop :=
  ∀ z : ℂ, ( (a * z^3 + b * z^2 + c * z + d = 0) → z.re < 0 )

theorem show_inequalities 
  (a b c d : ℝ) 
  (h: polynomial_has_negative_real_roots a b c d) :
  (a ≠ 0) → (b ≠ 0) → (c ≠ 0) → (d ≠ 0) → ab > 0 ∧ bc > ad ∧ ad > 0 :=
begin
  sorry
end

end show_inequalities_l336_336551


namespace cos_sum_diff_l336_336237

theorem cos_sum_diff (x y : ℝ) : 
  cos (x + y) - cos (x - y) = -2 * sin x * sin y := 
sorry

end cos_sum_diff_l336_336237


namespace option_B_correct_l336_336885

-- Define the commutativity of multiplication
def commutativity_of_mul (a b : Nat) : Prop :=
  a * b = b * a

-- State the problem, which is to prove that 2ab + 3ba = 5ab given commutativity
theorem option_B_correct (a b : Nat) : commutativity_of_mul a b → 2 * (a * b) + 3 * (b * a) = 5 * (a * b) :=
by
  intro h_comm
  rw [←h_comm]
  sorry

end option_B_correct_l336_336885


namespace eq_has_two_real_roots_iff_l336_336667

def f (x k : ℝ) : ℝ := x * real.exp (-2 * x) + k

noncomputable def f_prime (x : ℝ) : ℝ := (1 - 2 * x) * real.exp (-2 * x)

theorem eq_has_two_real_roots_iff (k : ℝ) :
  (∃! x1 x2 : ℝ, -2 < x1 ∧ x1 < 2 ∧ -2 < x2 ∧ x2 < 2 ∧ f x1 k = 0 ∧ f x2 k = 0) ↔ 
  -1 / (2 * real.exp 1) < k ∧ k < -2 / (real.exp 4):=
begin
  -- proof goes here
  sorry
end

end eq_has_two_real_roots_iff_l336_336667


namespace ella_running_time_l336_336230

theorem ella_running_time :
  (let lap_time_80 := 80 / 5 in
   let lap_time_180 := 180 / 4 in
   let total_active_time_lap := lap_time_80 + lap_time_180 in
   let break_time := 20 in
   let total_time_with_break := total_active_time_lap + break_time in
   2 * total_time_with_break + total_active_time_lap = 223) := by
  sorry

end ella_running_time_l336_336230


namespace convertibles_count_l336_336035

noncomputable def total_cars : ℕ := 125
noncomputable def percentage_regular_cars : ℝ := 0.64
noncomputable def percentage_trucks : ℝ := 0.08

noncomputable def number_regular_cars : ℕ := (percentage_regular_cars * total_cars).toInt
noncomputable def number_trucks : ℕ := (percentage_trucks * total_cars).toInt
noncomputable def number_convertibles : ℕ := total_cars - number_regular_cars - number_trucks

theorem convertibles_count : number_convertibles = 35 := by
  sorry

end convertibles_count_l336_336035


namespace max_elements_of_A_l336_336559

open Set

def M : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2016}

def valid_subset (A : Set ℕ) : Prop := 
  A ⊆ M ∧ ∀ x ∈ A, 15 * x ∉ A

theorem max_elements_of_A (A : Set ℕ) (hA : valid_subset A) : A.card ≤ 1890 := 
sorry

end max_elements_of_A_l336_336559


namespace sum_of_solutions_l336_336064

theorem sum_of_solutions : 
  ∀ (x y : ℝ), y = 5 ∧ x^2 + y^2 = 169 → (x = 12 ∨ x = -12) → (12 + -12 = 0) :=
by
  intros x y hxy hsol
  cases hxy with hy heq
  rw hy at heq
  rw ←eq_iff_iff
  rw sq

  sorry

end sum_of_solutions_l336_336064


namespace triangle_angle_A_eq_two_arcsin_two_thirds_l336_336342

/-- In triangle ABC, if b = c and the orthocenter H lies on the incircle,
    then the angle A is 2 * arcsin (2/3). -/
theorem triangle_angle_A_eq_two_arcsin_two_thirds
  {A B C H : Type} {b c : ℝ} [triangle A B C]
  (h1 : b = c) (h2 : orthocenter A B C = H) (h3 : lies_on_incircle H A B C) :
  ∠A = 2 * Real.arcsin (2 / 3) :=
sorry

end triangle_angle_A_eq_two_arcsin_two_thirds_l336_336342


namespace probability_no_adjacent_birch_l336_336535

theorem probability_no_adjacent_birch (m n : ℕ):
  let maple_trees := 5
  let oak_trees := 4
  let birch_trees := 6
  let total_trees := maple_trees + oak_trees + birch_trees
  (∀ (prob : ℚ), prob = (2 : ℚ) / 45) → (m + n = 47) := by
  sorry

end probability_no_adjacent_birch_l336_336535


namespace binomial_12_6_eq_924_l336_336974

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := sorry

end binomial_12_6_eq_924_l336_336974


namespace soup_weight_on_fourth_day_l336_336204

-- Define the initial condition and the halving process.
def initial_weight : ℝ := 80
def after_days (n : ℕ) : ℝ := initial_weight / (2^n)

-- Propose the theorem to validate.
theorem soup_weight_on_fourth_day : after_days 4 = 5 := by
  sorry

end soup_weight_on_fourth_day_l336_336204


namespace perimeter_of_8_sided_polygon_l336_336878

theorem perimeter_of_8_sided_polygon :
  ∀ (n : ℕ) (l : ℝ), n = 8 ∧ l = 3 → n * l = 24 :=
by
  intros n l h
  cases h with hn hl
  rw [hn, hl]
  exact rfl

end perimeter_of_8_sided_polygon_l336_336878


namespace Tim_has_16_pencils_l336_336111

variable (T_Sarah T_Tyrah T_Tim : Nat)

-- Conditions
def condition1 : Prop := T_Tyrah = 6 * T_Sarah
def condition2 : Prop := T_Tim = 8 * T_Sarah
def condition3 : Prop := T_Tyrah = 12

-- Theorem to prove
theorem Tim_has_16_pencils (h1 : condition1 T_Sarah T_Tyrah) (h2 : condition2 T_Sarah T_Tim) (h3 : condition3 T_Tyrah) : T_Tim = 16 :=
by
  sorry

end Tim_has_16_pencils_l336_336111


namespace quadratic_solution_l336_336094

theorem quadratic_solution (x : ℝ) : x^2 - 2 * x = 0 ↔ (x = 0 ∨ x = 2) := by
  sorry

end quadratic_solution_l336_336094


namespace find_speed_from_p_to_q_l336_336923

noncomputable def speed_from_p_to_q (v : ℝ) (d : ℝ) : Prop :=
  let return_speed := 1.5 * v
  let avg_speed := 75
  let total_distance := 2 * d
  let total_time := d / v + d / return_speed
  avg_speed = total_distance / total_time

theorem find_speed_from_p_to_q (v : ℝ) (d : ℝ) : speed_from_p_to_q v d → v = 62.5 :=
by
  intro h
  sorry

end find_speed_from_p_to_q_l336_336923


namespace chess_tournament_games_l336_336101

-- Define the problem
def total_chess_games (n_players games_per_player : ℕ) : ℕ :=
  (n_players * games_per_player) / 2

-- Conditions: 
-- 1. There are 6 chess amateurs.
-- 2. Each amateur plays exactly 4 games.

theorem chess_tournament_games :
  total_chess_games 6 4 = 10 :=
  sorry

end chess_tournament_games_l336_336101


namespace number_of_violas_l336_336165

theorem number_of_violas (V : ℕ) 
  (cellos : ℕ := 800) 
  (pairs : ℕ := 70) 
  (probability : ℝ := 0.00014583333333333335) 
  (h : probability = pairs / (cellos * V)) : V = 600 :=
by
  sorry

end number_of_violas_l336_336165


namespace correct_answer_l336_336227

def p : Prop := ∀ x : ℚ, x ∈ ℝ
def q : Prop := ∀ x : ℝ, (x^2 = -x^2)

theorem correct_answer (hp : p) (hq : ¬q) : ¬p ∨ ¬q :=
by
  sorry

end correct_answer_l336_336227


namespace chris_average_price_l336_336957

def total_cost (dvd_count dvd_price br_count br_price : ℕ) : ℕ :=
  dvd_count * dvd_price + br_count * br_price

def total_movies (dvd_count br_count : ℕ) : ℕ :=
  dvd_count + br_count

def average_price (total_cost total_movies : ℕ) : ℕ :=
  total_cost / total_movies

theorem chris_average_price :
  let dvd_count := 8
  let dvd_price := 12
  let br_count := 4
  let br_price := 18
  total_movies dvd_count br_count = 12 →
  total_cost dvd_count dvd_price br_count br_price = 168 →
  average_price (total_cost dvd_count dvd_price br_count br_price) (total_movies dvd_count br_count) = 14 :=
by 
  intros dvd_count dvd_price br_count br_price
  intros h1 h2,
  have h_total_cost : total_cost dvd_count dvd_price br_count br_price = 168 := h2,
  have h_total_movies : total_movies dvd_count br_count = 12 := h1,
  exact (show average_price 168 12 = 14 from rfl)

end chris_average_price_l336_336957


namespace sun_xing_zhe_problem_l336_336516

theorem sun_xing_zhe_problem (S X Z : ℕ) (h : S < 10 ∧ X < 10 ∧ Z < 10)
  (hprod : (100 * S + 10 * X + Z) * (100 * Z + 10 * X + S) = 78445) :
  (100 * S + 10 * X + Z) + (100 * Z + 10 * X + S) = 1372 := 
by
  sorry

end sun_xing_zhe_problem_l336_336516


namespace largest_possible_P10_l336_336007

noncomputable def P (x : ℤ) : ℤ := x^2 + 3*x + 3

theorem largest_possible_P10 : P 10 = 133 := by
  sorry

end largest_possible_P10_l336_336007


namespace sum_of_square_and_divisor_not_square_l336_336430

theorem sum_of_square_and_divisor_not_square {A B : ℕ} (hA : A ≠ 0) (hA_square : ∃ k : ℕ, A = k * k) (hB_divisor : B ∣ A) : ¬ (∃ m : ℕ, A + B = m * m) := by
  -- Proof is omitted
  sorry

end sum_of_square_and_divisor_not_square_l336_336430


namespace perimeter_of_triangle_LMN_l336_336365

variable (K L M N : Type)
variables [MetricSpace K]
variables [MetricSpace L]
variables [MetricSpace M]
variables [MetricSpace N]
variables (KL LN MN : ℝ)
variables (perimeter_LMN : ℝ)

-- Given conditions
axiom KL_eq_24 : KL = 24
axiom LN_eq_24 : LN = 24
axiom MN_eq_9  : MN = 9

-- Prove the perimeter is 57
theorem perimeter_of_triangle_LMN : perimeter_LMN = KL + LN + MN → perimeter_LMN = 57 :=
by sorry

end perimeter_of_triangle_LMN_l336_336365


namespace part1_mean_and_variance_part2_probability_l336_336717

noncomputable def mean_and_variance_for_20_students (μ_A σ2_A μ_B σ2_B : ℝ) (n_A n_B : ℝ) : ℝ × ℝ :=
let μ := (n_A * μ_A + n_B * μ_B) / (n_A + n_B),
    σ2 := (n_A * (σ2_A + (μ_A - μ)^2) + n_B * (σ2_B + (μ_B - μ)^2)) / (n_A + n_B) in
(μ, σ2)

theorem part1_mean_and_variance :
  mean_and_variance_for_20_students 1 1 1.5 0.25 12 8 = (1.2, 0.76) :=
by sorry

-- Probabilities of drawing questions in sequences
noncomputable def probability_A (p1 p2 p3 : ℝ) (pA_given_b1 pA_given_b2 pA_given_b3 : ℝ) : ℝ :=
p1 * pA_given_b1 + p2 * pA_given_b2 + p3 * pA_given_b3

noncomputable def conditional_probability (p : ℝ) (p_given : ℝ) : ℝ :=
(p_given * p) / p

theorem part2_probability :
  let p1 := 2 / 5,
      p2 := 8 / 15,
      p3 := 1 / 15,
      pA_given_b1 := 5 / 8,
      pA_given_b2 := 8 / 15,
      pA_given_b3 := 3 / 8 in
  conditional_probability (probability_A p1 p2 p3 pA_given_b1 pA_given_b2 pA_given_b3) pA_given_b1 = 6 / 13 :=
by sorry

end part1_mean_and_variance_part2_probability_l336_336717


namespace runner_catches_up_again_l336_336862

variable {V1 V2 S t : ℝ}
variable (k : ℝ)
variable (initial_distance : ℝ) (final_distance : ℝ)
variable h1 : V1 > V2
variable h2 : V1 = 3 * V2
variable h3 : initial_distance = V2 * t 
variable h4 : final_distance = 2 * V2 * t
variable h5 : initial_distance = (1/2) * S
variable h6 : final_distance = k * S
variable h7 : V1 * t = (k + 1) * S

theorem runner_catches_up_again (h1 : V1 > V2) (h2 : V1 = 3 * V2) (h3 : initial_distance = V2 * t)
                               (h4 : final_distance = 2 * V2 * t) (h5 : initial_distance = (1/2) * S)
                               (h6 : final_distance = k * S) (h7 : V1 * t = (k + 1) * S) :
  (∃ k : ℝ, 2(k + 1)= 3k) ∧ k + 0.5 = 2.5 :=
  by
  sorry

end runner_catches_up_again_l336_336862


namespace angle_D_not_80_l336_336713

theorem angle_D_not_80 (ABC DEF : Triangle)
  (h_congruent : congruent ABC DEF)
  (h_A : angle A = 50)
  (h_B : angle B = 70) : angle D ≠ 80 :=
sorry

end angle_D_not_80_l336_336713


namespace log_identity_l336_336039

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_identity
    (a b c : ℝ)
    (h1 : a ^ 2 + b ^ 2 = c ^ 2)
    (h2 : a > 0)
    (h3 : c > 0)
    (h4 : b > 0)
    (h5 : c > b) :
    log_base (c + b) a + log_base (c - b) a = 2 * log_base (c + b) a * log_base (c - b) a :=
sorry

end log_identity_l336_336039


namespace max_m_n_squared_l336_336223

theorem max_m_n_squared (m n : ℤ) 
  (hmn : 1 ≤ m ∧ m ≤ 1981 ∧ 1 ≤ n ∧ n ≤ 1981)
  (h_eq : (n^2 - m*n - m^2)^2 = 1) : 
  m^2 + n^2 ≤ 3524578 :=
sorry

end max_m_n_squared_l336_336223


namespace total_cost_stationery_l336_336932

def pencils_ordered := 20 * 100
def cost_pencil := 4
def pens_ordered := 2 * pencils_ordered + 300
def cost_pen := 5
def erasers_ordered := 150
def cost_eraser := 2
def discount_eraser := 0.05
def discount_pens_pencils := 0.10

def total_cost (pencils_ordered pencils_price pens_ordered pens_price erasers_ordered erasers_price discount_eraser discount_pens_pencils : ℕ) : ℕ :=
  let cost_pencils := pencils_ordered * pencils_price
  let cost_pens := pens_ordered * pens_price
  let cost_erasers := erasers_ordered * erasers_price
  let discounted_erasers := cost_erasers - (discount_eraser * cost_erasers).toNat
  let total_pens_pencils := cost_pencils + cost_pens
  let discounted_pens_pencils := total_pens_pencils - (discount_pens_pencils * total_pens_pencils).toNat
  discounted_pens_pencils + discounted_erasers

theorem total_cost_stationery : 
  total_cost pencils_ordered cost_pencil pens_ordered cost_pen erasers_ordered cost_eraser discount_eraser discount_pens_pencils = 26835 := 
by sorry

end total_cost_stationery_l336_336932


namespace collinear_S_T_K_l336_336407

theorem collinear_S_T_K
  (O : Type) [circle O]
  (A B C K H M S T : O)
  (l_a l_b : line O)
  (tangent : line O → O → Prop)
  (diameter : O × O → set O)
  (circumference : O → set O)
  (bc : O × O → set O)
  (angle_bisector : O × O × O → line O)
  (midpoint_arc : O × O × O → O)
  (intersects : line O → line O → O → Prop)
  (arc_midpoint : arc O → O)
  (is_diameter : diameter (A, B))
  (tangent_at_A : tangent l_a A)
  (tangent_at_B : tangent l_b B)
  (C_on_circumference : C ∈ circumference O)
  (BC_line : bc (B, C))
  (intersect_BC_la_at_K : intersects BC_line l_a K)
  (bisect_angle_CAK : angle_bisector (C, A, K))
  (bisect_intersect_CK_at_H : intersects (bisect_angle_CAK) (segment (C, K)) H)
  (M_midpoint_of_arc_CAB : M = midpoint_arc (arc A B C))
  (intersect_HM_circle_at_S : intersects (line (H, M)) (circumference O) S)
  (tangent_at_M : tangent l_b T) :
  collinear S T K :=
sorry

end collinear_S_T_K_l336_336407


namespace highest_price_per_shirt_l336_336941

theorem highest_price_per_shirt (x : ℝ) 
  (num_shirts : ℕ := 20)
  (total_money : ℝ := 180)
  (entrance_fee : ℝ := 5)
  (sales_tax : ℝ := 0.08)
  (whole_number: ∀ p : ℝ, ∃ n : ℕ, p = n) :
  (∀ (price_per_shirt : ℕ), price_per_shirt ≤ 8) :=
by
  sorry

end highest_price_per_shirt_l336_336941


namespace angle_between_a_b_l336_336312

/-- Definitions of vectors a and b -/
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, 0)

/-- The dot product of two vectors -/
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

/-- The magnitude of a vector -/
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

/-- The angle between two vectors -/
def angle_between (u v : ℝ × ℝ) : ℝ :=
  Real.arccos (dot_product u v / (magnitude u * magnitude v))

/-- Theorem stating the angle between vectors a and b is π/4 -/
theorem angle_between_a_b :
  angle_between a b = Real.pi / 4 :=
by
  sorry

end angle_between_a_b_l336_336312


namespace least_element_in_special_set_l336_336759

/-- 
  A set T ⊆ {1, 2, ..., 15} of cardinality 7 where no element is a 
  multiple of any other element has the least element equal to 4.
--/
theorem least_element_in_special_set :
  ∃ T : finset ℕ, T ⊆ finset.range 16 ∧ T.card = 7 ∧ (∀ a ∈ T, ∀ b ∈ T, a < b → ¬ (b % a = 0)) 
    ∧ (∀ x ∈ T, x ≥ 4) :=
begin
  sorry
end

end least_element_in_special_set_l336_336759


namespace greater_number_is_33_l336_336145

def hcf (a b : ℕ) : ℕ := 
  if b = 0 then a else hcf b (a % b)

theorem greater_number_is_33 (A B : ℕ) (hcf_A_B : hcf A B = 11) (prod_A_B : A * B = 363) : max A B = 33 :=
by
  sorry

end greater_number_is_33_l336_336145


namespace find_m_l336_336681

def A (m : ℝ) : Set ℝ := {x | x^2 - m * x + m^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
def C : Set ℝ := {2, -4}

theorem find_m (m : ℝ) : (A m ∩ B).Nonempty ∧ (A m ∩ C) = ∅ → m = -2 := by
  sorry

end find_m_l336_336681


namespace B_work_days_l336_336908

noncomputable theory

open Classical

-- Define the problem conditions
def A_work_rate : ℝ := 1 / 4
def B_work_rate (x : ℝ) : ℝ := 1 / x
def combined_work_rate : ℝ := 1 / 3

-- Show that given conditions imply B can do the work in 12 days
theorem B_work_days (x : ℝ) (h : A_work_rate + B_work_rate x = combined_work_rate) : x = 12 := by
  sorry

end B_work_days_l336_336908


namespace mean_of_five_numbers_l336_336852

theorem mean_of_five_numbers (sum_of_five : ℚ) (h : sum_of_five = 3/4) : sum_of_five / 5 = 3/20 := by
  rw [h]
  norm_num
  sorry

end mean_of_five_numbers_l336_336852


namespace set_of_possible_values_of_a_l336_336755

theorem set_of_possible_values_of_a :
  let A := {x : ℝ | x^2 + 2 * x - 3 = 0}
  let B (a : ℝ) := if a = 0 then ∅ else {x : ℝ | a * x = 3}
  ∀ a : ℝ, A ∩ B a = B a ↔ a ∈ {0, -1, 3} :=
by
  -- Definitions
  let A := {x : ℝ | x^2 + 2 * x - 3 = 0}
  let B := λ (a : ℝ), if a = 0 then ∅ else {x : ℝ | a * x = 3}
  -- Statement of theorem
  show ∀ a : ℝ, A ∩ B a = B a ↔ a ∈ {0, -1, 3}
  sorry

end set_of_possible_values_of_a_l336_336755


namespace smallest_positive_period_l336_336243

theorem smallest_positive_period :
  ∀ x : ℝ, ∃ T > 0, ∀ k : ℤ, 4 * sin(2 * (x + T) + π / 3) + 1 = 4 * sin(2 * x + π / 3) + 1 ↔ T = π :=
by
  sorry

end smallest_positive_period_l336_336243


namespace squareable_natural_numbers_l336_336873

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def can_be_arranged (n : ℕ) (arrangement : list ℕ) : Prop :=
  (arrangement.length = n) ∧
  (∀ (k : ℕ), k < n → is_perfect_square ((arrangement.nth k).getD 0 + (k + 1)))

def is_squareable (n : ℕ) : Prop :=
  ∃ (arrangement : list ℕ), can_be_arranged n arrangement

theorem squareable_natural_numbers : (is_squareable 7 → False) ∧
                                     (is_squareable 9) ∧
                                     (is_squareable 11 → False) ∧
                                     (is_squareable 15) :=
by {
  sorry
}

end squareable_natural_numbers_l336_336873


namespace length_real_axis_hyperbola_l336_336085

theorem length_real_axis_hyperbola :
  ∃ (C : ℝ → ℝ → Prop), 
    (∀ x y, C x y ↔ x^2 - y^2 = 4) ∧
    ∃ x1 y1 x2 y2, 
      C x1 y1 ∧ C x2 y2 ∧ 
      (x1 = -4 ∨ x1 = -4) ∧ (y2 = 2 * real.sqrt 3) ∧ 
      (abs (y1 - y2) = 4 * real.sqrt 3) ∧
      (2 * 2 = 4) :=
begin
  sorry
end

end length_real_axis_hyperbola_l336_336085


namespace domain_log_base_5_range_3_pow_neg_x_l336_336825

theorem domain_log_base_5 (x : ℝ) :
  (∃ y : ℝ, y = logBase 5 (1 - x)) ↔ x < 1 :=
by
  sorry

theorem range_3_pow_neg_x (y : ℝ) :
  (∃ x : ℝ, y = 3 ^ (-x)) ↔ y > 0 :=
by
  sorry

end domain_log_base_5_range_3_pow_neg_x_l336_336825


namespace curve_1_equiv_curve_2_equiv_eval_sum_recip_squares_func_f_geq_4_m_range_condition_l336_336515

section CoordinateSystem

variables {φ θ : ℝ} (ρ₁ ρ₂ : ℝ) {m x : ℝ}
def param_eqns_1 := (x = 2 * cos φ ∧ y = sin φ)
def param_eqns_2 := (x = 2 ∧ θ = π / 3)
def curve_eqn_1 := (x^2 / 4 + y^2 = 1)
def curve_eqn_2 := ((x - 2)^2 + y^2 = 4)
def polar_eqn_A := (ρ₁^2 = 4 / (4 * sin(θ)^2 + cos(θ)^2))
def polar_eqn_B := (ρ₂^2 = 4 / (sin(θ)^2 + 4 * cos(θ)^2))
def func_f := (f x = abs(x - 4 / m) + abs(x + m))

theorem curve_1_equiv : param_eqns_1 → curve_eqn_1 := sorry
theorem curve_2_equiv : param_eqns_2 → curve_eqn_2 := sorry
theorem eval_sum_recip_squares : polar_eqn_A → polar_eqn_B → (1 / ρ₁^2 + 1 / ρ₂^2 = 5 / 4) := sorry

end CoordinateSystem

section Inequalities

variables {m x : ℝ}
def func_f := (f x = abs(x - 4 / m) + abs(x + m))

theorem func_f_geq_4 : (m > 0) → ∀ x, f(x) ≥ 4 := sorry
theorem m_range_condition : (m > 0) → (f(2) > 5) → ((0 < m ∧ m < 1) ∨ (m > (1 + sqrt(17)) / 2)) := sorry

end Inequalities

end curve_1_equiv_curve_2_equiv_eval_sum_recip_squares_func_f_geq_4_m_range_condition_l336_336515


namespace initially_calculated_average_weight_l336_336440

theorem initially_calculated_average_weight 
  (A : ℚ)
  (h1 : ∀ sum_weight_corr : ℚ, sum_weight_corr = 20 * 58.65)
  (h2 : ∀ misread_weight_corr : ℚ, misread_weight_corr = 56)
  (h3 : ∀ correct_weight_corr : ℚ, correct_weight_corr = 61)
  (h4 : (20 * A + (correct_weight_corr - misread_weight_corr)) = 20 * 58.65) :
  A = 58.4 := 
sorry

end initially_calculated_average_weight_l336_336440


namespace log_domain_l336_336823

theorem log_domain (x : ℝ) : x + 2 > 0 ↔ x ∈ Set.Ioi (-2) :=
by
  sorry

end log_domain_l336_336823


namespace binom_12_6_eq_924_l336_336968

theorem binom_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binom_12_6_eq_924_l336_336968


namespace acuteAnglesSum_l336_336282

theorem acuteAnglesSum (A B C : ℝ) (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) 
  (hC : 0 < C ∧ C < π / 2) (h : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 1) :
  π / 2 ≤ A + B + C ∧ A + B + C ≤ π :=
by
  sorry

end acuteAnglesSum_l336_336282


namespace non_coplanar_points_count_l336_336935

theorem non_coplanar_points_count (tetra_midpoints : Finset (Fin 10)) : tetra_midpoints.card = 10 → (tetra_midpoints.powerset.filter (λ s, s.card = 4 ∧ ¬ ∃ (p : ℝ × ℝ × ℝ → Prop), ∀ x ∈ s, p x)).card = 141 :=
by
  sorry

end non_coplanar_points_count_l336_336935


namespace total_books_to_put_away_l336_336746

-- Definitions based on the conditions
def books_per_shelf := 4
def shelves_needed := 3

-- The proof problem translates to finding the total number of books
theorem total_books_to_put_away : shelves_needed * books_per_shelf = 12 := by
  sorry

end total_books_to_put_away_l336_336746


namespace select_4_officers_from_7_members_l336_336986

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Statement of the problem
theorem select_4_officers_from_7_members : binom 7 4 = 35 :=
by
  -- Proof not required, so we use sorry to skip it
  sorry

end select_4_officers_from_7_members_l336_336986


namespace scientific_notation_of_0_09_l336_336092

theorem scientific_notation_of_0_09 :
  (0.09 : ℝ) = (9 * 10 ^ (-2) : ℝ) :=
sorry

end scientific_notation_of_0_09_l336_336092


namespace new_weighted_avg_l336_336747

section
  -- Definitions for weights and original average
  variables {x : fin 15 → ℝ} {w : fin 15 → ℝ}
  def weights (i : fin 15) : ℝ := (i : ℝ) + 1
  def weighted_avg (x : fin 15 → ℝ) (w : fin 15 → ℝ) : ℝ :=
    ∑ i, w i * x i / ∑ i, w i

  -- Given conditions
  hypothesis (H_w_avg : weighted_avg x weights = 40)

  -- Prove that adding 11 to each number results in a new weighted average of 51
  theorem new_weighted_avg : weighted_avg (λ i, x i + 11) weights = 51 :=
  sorry
end

end new_weighted_avg_l336_336747


namespace no_carrying_pairs_count_l336_336617

theorem no_carrying_pairs_count : 
  let valid_digits_count (d : Nat) := if d < 9 then 1 else 0,
      valid_hundred_digits := ({5, 6, 7, 8}: Finset Nat),
      total_valid_combinations := (valid_hundred_digits.card * (0..8).sum valid_digits_count * (0..8).sum valid_digits_count) in
  total_valid_combinations = 324 :=
by
  sorry

end no_carrying_pairs_count_l336_336617


namespace sum_of_values_b_l336_336606

theorem sum_of_values_b (b : ℝ) : 
  (∃ b, (3 * x^2 - (b - 4) * x + 6 = 0 ∧ ((b - 4) ^ 2 - 4 * 3 * 6 = 0))) ->
  b = 4 + 6 * real.sqrt 2 ∨ b = 4 - 6 * real.sqrt 2 ->
  b.sum = 8 :=
by
  sorry

end sum_of_values_b_l336_336606


namespace other_asymptote_of_hyperbola_l336_336800

theorem other_asymptote_of_hyperbola (a b : ℝ) : 
  (∀ x : ℝ, y = 2 * x + 3) → 
  (∃ y : ℝ, x = -4) → 
  (∀ x : ℝ, y = - (1 / 2) * x - 7) := 
by {
  -- The proof will go here
  sorry
}

end other_asymptote_of_hyperbola_l336_336800


namespace rhombus_area_l336_336949

noncomputable def side_length (x y : ℝ) : ℝ :=
  real.sqrt (x^2 + y^2)

noncomputable def area_triangle (a R : ℝ) : ℝ :=
  a / (4 * R)

theorem rhombus_area 
  (x y : ℝ) 
  (r_EFG : ℝ := 15) 
  (r_EHG : ℝ := 30)
  (h1 : area_triangle (x * real.sqrt (x^2 + y^2)) r_EFG = area_triangle (y * real.sqrt (x^2 + y^2)) r_EHG)
  (h2 : y = 2 * x) : 
  (x * 2 * x / 2) = 562.5 := 
by
  sorry

end rhombus_area_l336_336949


namespace find_A_l336_336814

theorem find_A (A M C : Nat) (h1 : (10 * A^2 + 10 * M + C) * (A + M^2 + C^2) = 1050) (h2 : A < 10) (h3 : M < 10) (h4 : C < 10) : A = 2 := by
  sorry

end find_A_l336_336814


namespace acme_cheaper_min_shirts_l336_336940

theorem acme_cheaper_min_shirts :
  ∃ x : ℕ, 60 + 11 * x < 10 + 16 * x ∧ x = 11 :=
by {
  sorry
}

end acme_cheaper_min_shirts_l336_336940


namespace last_digit_of_expression_l336_336024

noncomputable def last_digit (n : ℕ) : ℤ :=
  (n % 10).toInt

theorem last_digit_of_expression (x : ℝ) (n : ℕ) (h1 : x ≠ 0) (h2 : x + x⁻¹ = 3) (h3 : n > 0) :
  last_digit (x^(2^n) + x^(-(2^n))) = 7 :=
by
  sorry

end last_digit_of_expression_l336_336024


namespace vehicle_value_last_year_l336_336473

variable (v_this_year v_last_year : ℝ)

theorem vehicle_value_last_year:
  v_this_year = 16000 ∧ v_this_year = 0.8 * v_last_year → v_last_year = 20000 :=
by
  -- Proof steps can be added here, but replaced with sorry as per instructions.
  sorry

end vehicle_value_last_year_l336_336473


namespace recurrence_relation_perfect_square_exist_sequences_l336_336679

-- Define the recursion for a_n
def a : ℕ → ℕ
| 0     := 1  -- start from a₀ when n = 0
| 1     := 9  -- for n = 1
| (n+2) := 10 * a (n+1) - a n

-- First problem: Prove the recurrence relation
theorem recurrence_relation (n : ℕ) : 
  a (n + 2) = 10 * a (n + 1) - a n := 
sorry

-- Second problem: Prove that (a_n * a_(n+1) - 1) / 2 is a perfect square
theorem perfect_square (n : ℕ) : 
  ∃ k : ℕ, (a n * a (n + 1) - 1) = 2 * k^2 :=
sorry

-- Third problem: Show existence of sequences x_n and y_n satisfying a_n
theorem exist_sequences (n : ℕ) : 
  ∃ (x y : ℕ), a n = (x^2 + 2) / (2 * (x + y)) :=
sorry

end recurrence_relation_perfect_square_exist_sequences_l336_336679


namespace arrangements_count_l336_336245

theorem arrangements_count (A B C D E : Type) (positions : Finset ℕ) (h₁ : positions = {1, 2, 3, 4}) :
  let people := {A, B, C, D, E}
  ∃ f : Fin 4 → people, (∀ i, f i ∉ (positions.filter (λ x, f x = A))) ∧ (positions.card = 4) ∧ (people.card = 5) → count f = 42 :=
by
  sorry

end arrangements_count_l336_336245


namespace circumcircle_radius_l336_336483

theorem circumcircle_radius (r1 r2 r3 : ℝ) (O1 O2 A B C : ℝ)
  (h1 : r1 + r2 = 7)
  (h2 : (O1 - O2)^2 = 169)
  (h3 : r3 = 5)
  (h4 : ∀ x : ℝ, sqrt x >= 0 ) -- Ensure non-negative results for sqrt
  : (c : ℝ) (h5 : a ^ 2 + b^2 = c^2) → c = sqrt 30 :=
by
  sorry

end circumcircle_radius_l336_336483


namespace polyhedron_vertices_faces_relation_l336_336345

-- Define the polyhedral scenario
variables (V E F : ℕ)
axiom euler_formula : V - E + F = 2
axiom degree_of_vertices (A : ℕ) (vertices : list ℕ) : 
  (A = 5) ∧ (∀ v ∈ vertices, v = 3)

-- Sum of degrees equals twice the number of edges
axiom sum_of_degrees : 5 + 3 * (V - 1) = 2 * E

theorem polyhedron_vertices_faces_relation : V = 2 * F - 6 :=
by {
  sorry
}

end polyhedron_vertices_faces_relation_l336_336345


namespace incorrect_statement_E_l336_336795

theorem incorrect_statement_E : 
  (∀ (b h : ℕ), let A := b * h in
  let A' := 2 * b * h in
  A' = 2 * A) ∧
  (∀ (a b h : ℕ), let A := (a + b) * h / 2 in
  let A' := (a + b) * (2 * h) / 2 in
  A' = 2 * A) ∧
  (∀ (a b : ℕ), let A := π * a * b in
  let A' := π * a * (2 * b) in
  A' = 2 * A) ∧
  (∀ (a b : ℕ), let Q := a / b in
  let Q' := (2 * a) / (2 * b) in
  Q' = Q) →
  (¬ ∃ x : ℕ, x + 5 ≤ x) :=
begin
  intros h,
  intro h1,
  cases h1 with x hx,
  linarith,
end

end incorrect_statement_E_l336_336795


namespace no_carrying_pairs_correct_l336_336616

noncomputable def count_no_carrying_pairs : ℕ :=
  let pairs := (1500 : ℕ, 2500 : ℕ)
  (1550 : ℕ)  -- correct answer

theorem no_carrying_pairs_correct :
  ∃ count : ℕ, count = count_no_carrying_pairs :=
sorry

end no_carrying_pairs_correct_l336_336616


namespace polygon_area_l336_336734

-- Define the problem conditions
def conditions (polygon : Type)
  (sides : polygon → ℕ) -- Number of sides
  (perimeter : polygon → ℕ) -- Perimeter
  (side_length : polygon → ℕ) -- Side length
  (short_side : polygon → ℕ)
  (long_side : polygon → ℕ): Type :=
  (sides polygon = 36) ∧
  (perimeter polygon = 72) ∧
  (side_length polygon = perimeter polygon / sides polygon) ∧
  (short_side polygon = side_length polygon / 2) ∧
  (long_side polygon = 2 * side_length polygon)

-- The proof problem statement
theorem polygon_area (P : Type) 
  [conditions P (λ _ => 36) (λ _ => 72) (λ _ => 2) (λ _ => 1) (λ _ => 4)] :
  (18 * (1 * 4) = 72) :=
by
  -- Providing a partial proof framework with sorry to indicate it is incomplete
  sorry

end polygon_area_l336_336734


namespace number_of_ways_to_form_4x4x4_cube_l336_336868

theorem number_of_ways_to_form_4x4x4_cube (white_cubes black_cube : ℕ) (edge_length : ℕ) :
  white_cubes = 63 ∧ black_cube = 1 ∧ edge_length = 1 →
  let total_cubes := 64 in
  let corners := 8 in
  let edges := 24 in
  let face_centers := 24 in
  let inside_cube := 8 in
  corners + edges + face_centers + inside_cube = total_cubes ∧ total_cubes / 4 + 10 / 2 / 5 = 15 :=
begin
  sorry
end

end number_of_ways_to_form_4x4x4_cube_l336_336868


namespace complex_division_l336_336901

-- Define the complex numbers used in the problem
def z1 : ℂ := 3 + I
def z2 : ℂ := 1 + I

-- The theorem to be proved
theorem complex_division : z1 / z2 = 2 - I :=
by
  sorry

end complex_division_l336_336901


namespace prove_f_sum_l336_336006

noncomputable def f : ℕ+ → ℝ := sorry

-- Conditions
axiom functional_eq (n : ℕ+) (h : n > 1) (p : ℕ) (hp : nat.prime p) (hp_dvd : p ∣ n) : 
  f n = f (n / p) - f p

axiom given_eq : f (2 ^ 2014) + f (3 ^ 2015) + f (5 ^ 2016) = 2013

-- Theorem to prove
theorem prove_f_sum : f (2014 ^ 2) + f (2015 ^ 3) + f (2016 ^ 5) = 49 / 3 :=
  sorry

end prove_f_sum_l336_336006


namespace effect_of_tax_and_consumption_change_on_revenue_percentage_change_in_revenue_effect_on_revenue_l336_336472

theorem effect_of_tax_and_consumption_change_on_revenue 
  (T C : ℝ) 
  (new_tax_rate : ℝ := 0.82 * T) 
  (new_consumption : ℝ := 1.15 * C) 
  : (new_tax_rate * new_consumption) / (T * C) = 0.943 :=
by 
  have h1 : new_tax_rate = 0.82 * T := rfl
  have h2 : new_consumption = 1.15 * C := rfl
  calc 
    (new_tax_rate * new_consumption) / (T * C)
    = (0.82 * T * 1.15 * C) / (T * C) : by rw [h1, h2]
    ... = 0.82 * 1.15 : by ring
    ... = 0.943 : by norm_num

theorem percentage_change_in_revenue 
  (T C : ℝ) 
  (new_tax_rate : ℝ := 0.82 * T) 
  (new_consumption : ℝ := 1.15 * C) 
  : (0.943 - 1) * 100 = -5.7 :=
by
  calc 
    (0.943 - 1) * 100 = -0.057 * 100 : by norm_num
    ... = -5.7 : by norm_num

-- Combine theorems to achieve the final statement
theorem effect_on_revenue 
  (T C : ℝ) 
  (new_tax_rate : ℝ := 0.82 * T) 
  (new_consumption : ℝ := 1.15 * C) 
  : (new_tax_rate * new_consumption) / (T * C) = 0.943 ∧ (0.943 - 1) * 100 = -5.7 := 
by
  apply and.intro
  { exact effect_of_tax_and_consumption_change_on_revenue T C new_tax_rate new_consumption }
  { exact percentage_change_in_revenue T C new_tax_rate new_consumption }

end effect_of_tax_and_consumption_change_on_revenue_percentage_change_in_revenue_effect_on_revenue_l336_336472


namespace a_6_value_l336_336469

noncomputable def a (n : ℕ) : ℕ := 
match n with 
| 0      => 0
| 1      => 1
| (n + 2) => 4 * a (n + 1)

def S (n : ℕ) : ℕ :=
(nat.sum_range n.succ a)

theorem a_6_value : a 6 = 3 * 4^4 := by
  sorry

end a_6_value_l336_336469


namespace parallel_vectors_with_same_direction_l336_336683

theorem parallel_vectors_with_same_direction (a b : ℝ^3) (ha : a ≠ 0) (hb : b ≠ 0) (h : ∥a + b∥ = ∥a∥ + ∥b∥) : 
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b := 
sorry

end parallel_vectors_with_same_direction_l336_336683


namespace binomial_12_6_eq_924_l336_336960

theorem binomial_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binomial_12_6_eq_924_l336_336960


namespace cecilia_prob_win_l336_336565

theorem cecilia_prob_win :
  let p_cecilia := 1 / 4
  let p_dexter := 1 / 3
  let p_eliza := 1 / 2
  let P := infinite_geometric_series p_cecilia (p_cecilia * (1 - p_dexter) * (1 - p_eliza))
  P = 1 / 2 :=
by
  sorry

end cecilia_prob_win_l336_336565


namespace smallest_even_sum_equals_200_l336_336851

theorem smallest_even_sum_equals_200 :
  ∃ (x : ℤ), (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 200) ∧ (x = 36) :=
by
  sorry

end smallest_even_sum_equals_200_l336_336851


namespace distance_between_points_l336_336777

-- Definitions of the points in the complex plane
def liam_point : ℂ := complex.mk 3 3
def ella_point : ℂ := complex.mk (-2) (real.sqrt 2)

-- The statement to prove: the distance between Liam's and Ella's points is sqrt(36 - 6 * sqrt(2))
theorem distance_between_points :
  complex.abs (liam_point - ella_point) = real.sqrt (36 - 6 * real.sqrt 2) :=
sorry

end distance_between_points_l336_336777


namespace more_numbers_without_1_in_1_to_1000_more_7_digit_numbers_with_1_l336_336894

-- Part (a): More numbers in [1, 1000] do not contain digit '1'
theorem more_numbers_without_1_in_1_to_1000 :
  let count_with_1 := (1000 - 512) in
  512 > count_with_1 :=
by sorry

-- Part (b): More 7-digit numbers contain digit '1'
theorem more_7_digit_numbers_with_1 :
  let total_7_digit_nums := 9 * 10^6 in
  let count_without_1 := 8 * 9^6 in
  (total_7_digit_nums - count_without_1) > count_without_1 := 
by sorry

end more_numbers_without_1_in_1_to_1000_more_7_digit_numbers_with_1_l336_336894


namespace find_some_number_l336_336332

theorem find_some_number (x : ℤ) (h : 45 - (28 - (x - (15 - 20))) = 59) : x = 37 :=
by
  sorry

end find_some_number_l336_336332


namespace isosceles_triangle_BKL_l336_336451

-- Define the necessary geometric entities and assumptions.
variables {ABC : Type} [triangle ABC]
variables (A B C : Point ABC)
variables (l : Line) (K L : Point ABC)
variables (O : Point ABC) (B_tangent : tangent l (circumcircle ABC) B)
variables (ortho : Point ABC) (K_projection : projection ortho l K)
variables (L_midpoint : midpoint L A C)
variables (acute : acute_triangle ABC)

-- Define the main theorem statement that needs to be proved.
theorem isosceles_triangle_BKL : isosceles_triangle B K L := 
sorry

end isosceles_triangle_BKL_l336_336451


namespace smallest_n_is_64_l336_336015

noncomputable def smallest_n (a : ℝ) : ℕ :=
  (Classical.find (λ n : ℕ, ∃ k : ℤ, (n + 1) * n = 4020 * k))

theorem smallest_n_is_64 : 
  let a := (Real.pi / 4020) in smallest_n a = 64 :=
by
  sorry

end smallest_n_is_64_l336_336015


namespace inner_cube_surface_area_is_32_l336_336544

-- Define the conditions
def volume_of_cube : ℝ := 64
def side_length_of_cube (v : ℝ) : ℝ := v^(1/3)
def diameter_of_sphere (s : ℝ) := s
def radius_of_sphere (d : ℝ) := d / 2
def diagonal_of_smaller_cube (l : ℝ) := l * Real.sqrt 3
def side_length_of_smaller_cube (d : ℝ) := d / Real.sqrt 3
def surface_area_of_cube (l : ℝ) := 6 * l^2

-- Given the volume, side length, and diameter relations, prove the surface area is 32
theorem inner_cube_surface_area_is_32 :
  surface_area_of_cube (side_length_of_smaller_cube (diameter_of_sphere (side_length_of_cube volume_of_cube))) = 32 :=
sorry

end inner_cube_surface_area_is_32_l336_336544


namespace problem_1_problem_2_l336_336302

def f (x : ℝ) : ℝ := abs (x - 2)
def g (x m : ℝ) : ℝ := -abs (x + 7) + 3 * m

theorem problem_1 (x : ℝ) : f x + x^2 - 4 > 0 ↔ (x > 2 ∨ x < -1) := sorry

theorem problem_2 {m : ℝ} (h : m > 3) : ∃ x : ℝ, f x < g x m := sorry

end problem_1_problem_2_l336_336302


namespace A_P_M_O_concyclic_l336_336253

open Real EuclideanGeometry

-- Definitions for points and conditions.
variable {A B C O D M E F P : Point}
variable {circumcircle : Circle}

-- Conditions from the problem.
axiom TriangleABC (A B C : Point) : Triangle A B C
axiom circumcenterO (O A B C : Point) : Circumcenter O A B C
axiom AngleABD_eq_AngleDCA (A B C D : Point) : ∠ A B D = ∠ D C A
axiom MidpointAD (A D M : Point) : Midpoint A D M
axiom BM_intersects_circle_at_E (B M E : Point) (circumcircle : Circle B M) : CircleIntersectsAt E circumcircle A
axiom CM_intersects_circle_at_F (C M F : Point) (circumcircle : Circle C M) : CircleIntersectsAt F circumcircle A
axiom P_on_EF (E F P : Point) : PointOnLineSegment E F P
axiom AP_tangent_to_circle (A P : Point) (circumcircle : Circle A P) : TangentToCircle A P circumcircle

-- Prove that A, P, M, O are concyclic
theorem A_P_M_O_concyclic {A B C O D M E F P : Point}
  (triangle_ABC : Triangle A B C)
  (circumcenter_O : Circumcenter O A B C)
  (angle_ABD_eq_angle_DCA : ∠ A B D = ∠ D C A)
  (midpoint_AD : Midpoint A D M)
  (BM_intersects_E : CircleIntersectsAt E (circumcircle ⟨B, M⟩) A)
  (CM_intersects_F : CircleIntersectsAt F (circumcircle ⟨C, M⟩) A)
  (P_on_EF : PointOnLineSegment E F P)
  (AP_tangent : TangentToCircle A P (circumcircle ⟨A, P⟩)) :
  Concyclic A P M O :=
by
  sorry

end A_P_M_O_concyclic_l336_336253


namespace problem_solution_l336_336875

theorem problem_solution : 15 * ((1 / 3 : ℚ) + (1 / 4) + (1 / 6))⁻¹ = 20 := by
  have h1 : (1 / 3 : ℚ) + (1 / 4) + (1 / 6) = 3 / 4 := sorry
  have h2 : ((3 / 4)⁻¹ : ℚ) = 4 / 3 := sorry
  calc
    15 * ((1 / 3 : ℚ) + (1 / 4) + (1 / 6))⁻¹
        = 15 * (3 / 4)⁻¹ : by rw [h1]
    ... = 15 * (4 / 3)   : by rw [h2]
    ... = 20             : by norm_num

end problem_solution_l336_336875


namespace spatial_relationship_l336_336274

-- Definitions for the conditions
variables (α : Type) [plane α] (a b : Type) [line a] [line b]
variables (a' b' : Type) [line a'] [line b']
variables (proj_a : orthogonal_projection a α) (proj_b : orthogonal_projection b α)

-- Given conditions
axiom contained_in_plane : a' ⊂ α ∧ b' ⊂ α
axiom parallel_proj : parallel a' b'
axiom proj_lines : (proj_a a' α = a') ∧ (proj_b b' α = b')

-- The theorem to be proven
theorem spatial_relationship : parallel a b ∨ skew a b :=
sorry

end spatial_relationship_l336_336274


namespace angles_of_triangle_DEF_l336_336363

-- Define the basic properties and conditions of the triangle ABC
-- Let ΔABC be a triangle with AB = AC and ∠ABC = 60°
def triangle_ABC (A B C D E F : Type) [Triangle A B C] (AB AC : Prop) (angle_ABC : ℝ) : Prop :=
  IsIsosceles A B C ∧ angle_ABC = 60

-- Define the properties of the segments and points D, E, and F
def segment_BD_bisects_angle_A (A B C D : Type) [Segment B D] (bisects : Prop) : Prop :=
  BisectsAngle B D A C

def DE_parallel_to_AB (A B C D E : Type) [Segment D E] [Segment A B] : Prop :=
  Parallel D E A B

def EF_parallel_to_BD (B D E F : Type) [Segment E F] [Segment B D] : Prop :=
  Parallel E F B D

-- Define triangle DEF and its properties
def triangle_DEF (D E F : Type) [Triangle D E F] (angle_DEF angle_EFD angle_DFE : ℝ) : Prop :=
  angle_DEF = 30 ∧ angle_EFD = 30 ∧ angle_DFE = 120

-- The main theorem to prove the angles of triangle DEF
theorem angles_of_triangle_DEF (A B C D E F : Type)
  [Triangle A B C] [Triangle D E F] [Segment B D] [Segment D E] [Segment E F] [Segment A B]
  (isosceles : IsIsosceles A B C) (angle_ABC : ∠A B C = 60)
  (bisects : BisectsAngle B D A C) (parallel_DE_AB : Parallel D E A B)
  (parallel_EF_BD : Parallel E F B D) :
  ∠D E F = 30 ∧ ∠E F D = 30 ∧ ∠D F E = 120 :=
sorry

end angles_of_triangle_DEF_l336_336363


namespace problem_1_problem_2_l336_336775

noncomputable def parabola (a : ℝ) : ℝ → ℝ → Prop := λ x y, y^2 = 4 * a * x
noncomputable def semicircle (a : ℝ) : ℝ → ℝ → Prop := λ x y, (x - (a + 4))^2 + y^2 = 16
noncomputable def focus (a : ℝ) : ℝ × ℝ := (a, 0)
noncomputable def pointB (a : ℝ) : ℝ × ℝ := (a + 4, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def intersects (p1 p2 : Point) (P : Point) : Prop :=
  ∃ M N : Point, parabola a M.1 M.2 ∧ semicircle a M.1 M.2 ∧
                 parabola a N.1 N.2 ∧ semicircle a N.1 N.2 ∧
                 M ≠ N ∧ midpoint P M N

theorem problem_1
  {a : ℝ}
  (h_pos : a > 0)
  (M N : Point)
  (intersect_MN : intersects a M N)
  (P : Point)
  (h_midpoint : midpoint P M N) :
  distance (focus a) M + distance (focus a) N = 8 := sorry

theorem problem_2
  {a : ℝ}
  (h_pos : a > 0)
  (M N P : Point)
  (intersect_MN : intersects a M N)
  (h_midpoint : midpoint P M N) :
  ¬ ∃ a : ℝ, distance (focus a) M + distance (midpoint P) + distance (focus a) N ∈ list.arithmetic_seq := sorry

end problem_1_problem_2_l336_336775


namespace construct_trapezoid_l336_336571

structure Trapezoid (A B C D : Type) :=
(parallel_sides : A → A → B → B → Prop)
(non_parallel_side : A → C → Prop)
(diagonal_ratio : C → D → Prop)

variable {A B C D : Type} [trapezoid : Trapezoid A B C D]

theorem construct_trapezoid (a b : A) (c : C) (ratio : D) 
  (H_parallel_sides : trapezoid.parallel_sides a a b b)
  (H_non_parallel_side : trapezoid.non_parallel_side a c)
  (H_diagonal_ratio : trapezoid.diagonal_ratio c ratio) :
  ∃ (trapezoid_constructed : Trapezoid A B C D), 
  trapezoid_constructed.parallel_sides a a b b ∧ 
  trapezoid_constructed.non_parallel_side a c ∧ 
  trapezoid_constructed.diagonal_ratio c ratio := 
sorry

end construct_trapezoid_l336_336571


namespace probability_triangle_has_side_of_decagon_is_7_over_12_l336_336417

-- Definitions based on the conditions
def vertices : ℕ := 10
def total_triangles := nat.choose vertices 3
def favorable_triangles :=
  let one_side := vertices * (vertices - 4)
  let two_sides := vertices
  one_side + two_sides
def probability := favorable_triangles.to_rat / total_triangles.to_rat

-- Proof problem statement
theorem probability_triangle_has_side_of_decagon_is_7_over_12 :
  probability = 7 / 12 :=
by
  sorry

end probability_triangle_has_side_of_decagon_is_7_over_12_l336_336417


namespace convex_polyhedron_submersion_l336_336554

theorem convex_polyhedron_submersion (V S : ℝ) (alpha : ℝ) (hV_pos : 0 < V) (hS_pos : 0 < S)
  (h90_percent_volume_submerged : 0.9 * V ≤ volume_of_submerged_part V)
  (hhalf_surface_area_above_water : S * (1 + cos alpha) > 2 * S * (0.9)^(2/3)) :
  ∃ alpha', cos alpha' > (2 * (0.9)^(2/3) - 1) :=
by
  sorry

end convex_polyhedron_submersion_l336_336554


namespace num_employees_l336_336072

-- Define the conditions given in the problem.
variables (n : ℕ) -- The number of employees initially.
parameters (avg_emp_salary : ℝ) (manager_salary : ℝ) (new_avg_salary : ℝ)

-- Given conditions.
def conditions : Prop :=
  avg_emp_salary = 1300 ∧ 
  manager_salary = 3400 ∧ 
  new_avg_salary = 1400

-- The proof statement that needs to be proved.
theorem num_employees (h : conditions) : n = 20 :=
by {
  cases h with h1 h2,
  cases h2 with h2 h3, sorry
}

end num_employees_l336_336072


namespace no_carrying_pairs_count_l336_336618

theorem no_carrying_pairs_count : 
  let valid_digits_count (d : Nat) := if d < 9 then 1 else 0,
      valid_hundred_digits := ({5, 6, 7, 8}: Finset Nat),
      total_valid_combinations := (valid_hundred_digits.card * (0..8).sum valid_digits_count * (0..8).sum valid_digits_count) in
  total_valid_combinations = 324 :=
by
  sorry

end no_carrying_pairs_count_l336_336618


namespace find_b_l336_336593

variable (x : ℝ)

noncomputable def d : ℝ := 3

theorem find_b (b c : ℝ) :
  (7 * x^2 - 5 * x + 11 / 4) * (d * x^2 + b * x + c) = 21 * x^4 - 26 * x^3 + 34 * x^2 - 55 / 4 * x + 33 / 4 →
  b = -11 / 7 :=
by
  sorry

end find_b_l336_336593


namespace no_ninety_nine_percent_confidence_self_phone_management_gender_probability_of_selection_is_four_fifths_l336_336934

theorem no_ninety_nine_percent_confidence_self_phone_management_gender :
  let total_students := 100
  let poor_self_phone_management_students := 20
  let good_self_phone_management_students := 80
  let male_students := 60
  let female_students := 40
  let male_good_self_phone_management := 52
  let female_good_self_phone_management := 28
  let male_poor_self_phone_management := poor_self_phone_management_students - 12
  let female_poor_self_phone_management := 12
  let contingency_table := [
    [male_good_self_phone_management, male_poor_self_phone_management],
    [female_good_self_phone_management, female_poor_self_phone_management]
  ]
  let k_squared := (total_students * ((male_good_self_phone_management * female_poor_self_phone_management - male_poor_self_phone_management * female_good_self_phone_management) ^ 2)) / (male_students * female_students * good_self_phone_management_students * poor_self_phone_management_students)
  in k_squared < 6.635 :=
by
  -- Add definitions and skip proof
  sorry

theorem probability_of_selection_is_four_fifths :
  let total_combinations := 10
  let favorable_combinations := 8
  in favorable_combinations / total_combinations = 4 / 5 :=
by
  -- Add definitions and skip proof
  sorry

end no_ninety_nine_percent_confidence_self_phone_management_gender_probability_of_selection_is_four_fifths_l336_336934


namespace fuel_tank_initial_capacity_l336_336375

variables (fuel_consumption : ℕ) (journey_distance remaining_fuel initial_fuel : ℕ)

-- Define conditions
def fuel_consumption_rate := 12      -- liters per 100 km
def journey := 275                  -- km
def remaining := 14                 -- liters
def fuel_converted := (fuel_consumption_rate * journey) / 100

-- Define the proposition to be proved
theorem fuel_tank_initial_capacity :
  initial_fuel = fuel_converted + remaining :=
sorry

end fuel_tank_initial_capacity_l336_336375


namespace min_marbles_l336_336869

theorem min_marbles (n a b : ℕ) :
  ∑ k in finset.range n, (a + k) * (b + k) = 
  n * ((n + 1) * (2 * n + 3 * (a + b) - 5) / 6 + (a - 1) * (b - 1)) :=
by sorry

end min_marbles_l336_336869


namespace area_difference_l336_336896

theorem area_difference (l1 w1 l2 w2 : ℝ) (h1 : l1 = 11) (h2 : w1 = 13) (h3 : l2 = 6.5) (h4 : w2 = 11) :
  2 * (l1 * w1) - 2 * (l2 * w2) = 143 :=
by
  -- Utilize the conditions
  rw [h1, h2, h3, h4]
  -- Simplify the expression
  norm_num
  sorry

end area_difference_l336_336896


namespace smallest_cut_length_l336_336186

theorem smallest_cut_length (x : ℕ) (h₁ : 9 ≥ x) (h₂ : 12 ≥ x) (h₃ : 15 ≥ x)
  (h₄ : x ≥ 6) (h₅ : x ≥ 12) (h₆ : x ≥ 18) : x = 6 :=
by
  sorry

end smallest_cut_length_l336_336186


namespace problem_l336_336675

noncomputable def f (x : ℝ) : ℝ := x^2 - 6 * x + 10

theorem problem (m : ℝ) (h1 : m > 1) (h2 : f m = 1) :
  m = 3 ∧ (∀ x ∈ (Set.Icc 3 5), f x ≤ 5) ∧ (∀ x ∈ (Set.Icc 3 5), f x ≥ 1) :=
by
  sorry

end problem_l336_336675


namespace sum_inequality_l336_336768

theorem sum_inequality (n : ℕ) (a : fin (n + 1) → ℝ)
  (h1 : ∀ i, a i ∈ set.Icc (-1 : ℝ) 1)
  (h2 : ∀ i, a i * a (i + 1) ≠ -1)
  (h3 : a ⟨n⟩ = a 0) :
  ∑ i in finset.range n, 1 / (1 + a i * a (i + 1)) ≥
  ∑ i in finset.range n, 1 / (1 + (a i) ^ 2) := 
sorry

end sum_inequality_l336_336768


namespace f_240_is_40_over_3_l336_336402

variable (f : ℝ → ℝ)
variable h : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y
variable h_f800 : f 800 = 4

theorem f_240_is_40_over_3 : f 240 = 40 / 3 :=
by
  sorry

end f_240_is_40_over_3_l336_336402


namespace mod_11_residue_l336_336952

theorem mod_11_residue : 
  ((312 - 3 * 52 + 9 * 165 + 6 * 22) % 11) = 2 :=
by
  sorry

end mod_11_residue_l336_336952


namespace min_paint_steps_l336_336718

-- Checkered square of size 2021x2021 where all cells initially white.
-- Ivan selects two cells and paints them black.
-- Cells with at least one black neighbor by side are painted black simultaneously each step.

-- Define a function to represent the steps required to paint the square black
noncomputable def min_steps_to_paint_black (n : ℕ) (a b : ℕ × ℕ) : ℕ :=
  sorry -- Placeholder for the actual function definition, as we're focusing on the statement.

-- Define the specific instance of the problem
def square_size := 2021
def initial_cells := ((505, 1010), (1515, 1010))

-- Theorem statement: Proving the minimal number of steps required is 1515
theorem min_paint_steps : min_steps_to_paint_black square_size initial_cells.1 initial_cells.2 = 1515 :=
sorry

end min_paint_steps_l336_336718


namespace find_f_π_4_l336_336674

def f (ω x : ℝ) : ℝ := Real.sin (ω * x)

def g (ω x : ℝ) : ℝ := Real.sin (ω * (x - Real.pi / 4))

theorem find_f_π_4 (ω : ℝ) (h₁ : 0 < ω)
    (h₂ : ∀ x₁ x₂ : ℝ, |f ω x₁ - g ω x₂| = 2 → |x₁ - x₂| = Real.pi / 4)
    : f 2 (Real.pi / 4) = 1 :=
by
  sorry

end find_f_π_4_l336_336674


namespace binomial_12_6_eq_924_l336_336976

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := sorry

end binomial_12_6_eq_924_l336_336976


namespace find_symmetric_complex_number_l336_336459

-- Definitions of the complex numbers and the problem statement
def z1 : ℂ := 3 + 2 * complex.I
def symmetric_with_respect_to_y_eq_x (z z_sym : ℂ) : Prop := 
  z_sym.re = z.im ∧ z_sym.im = z.re

theorem find_symmetric_complex_number :
  symmetric_with_respect_to_y_eq_x z1 z2 :=
by {
  let z2 := 2 + 3 * complex.I,
  exact ⟨rfl, rfl⟩
}

end find_symmetric_complex_number_l336_336459


namespace option_B_correct_l336_336886

-- Define the commutativity of multiplication
def commutativity_of_mul (a b : Nat) : Prop :=
  a * b = b * a

-- State the problem, which is to prove that 2ab + 3ba = 5ab given commutativity
theorem option_B_correct (a b : Nat) : commutativity_of_mul a b → 2 * (a * b) + 3 * (b * a) = 5 * (a * b) :=
by
  intro h_comm
  rw [←h_comm]
  sorry

end option_B_correct_l336_336886


namespace mod_sum_power_l336_336880

theorem mod_sum_power (n : ℤ) :
  (9^7 + 6^9 + 5^{11}) % 7 = 4 :=
by
  have h1 : 9 % 7 = 2 := by norm_num
  have h2 : 6 % 7 = -1 := by norm_num
  have h3 : 5 % 7 = -2 := by norm_num
  sorry

end mod_sum_power_l336_336880


namespace tax_is_one_l336_336573

-- Define costs
def cost_eggs : ℕ := 3
def cost_pancakes : ℕ := 2
def cost_cocoa : ℕ := 2

-- Initial order
def initial_eggs := 1
def initial_pancakes := 1
def initial_mugs_of_cocoa := 2

-- Additional order by Ben
def additional_pancakes := 1
def additional_mugs_of_cocoa := 1

-- Calculate costs
def initial_cost : ℕ := initial_eggs * cost_eggs + initial_pancakes * cost_pancakes + initial_mugs_of_cocoa * cost_cocoa
def additional_cost : ℕ := additional_pancakes * cost_pancakes + additional_mugs_of_cocoa * cost_cocoa
def total_cost_before_tax : ℕ := initial_cost + additional_cost

-- Payment and change
def total_paid : ℕ := 15
def change : ℕ := 1
def actual_payment : ℕ := total_paid - change

-- Calculate tax
def tax : ℕ := actual_payment - total_cost_before_tax

-- Prove that the tax is $1
theorem tax_is_one : tax = 1 :=
by
  sorry

end tax_is_one_l336_336573


namespace values_of_m_in_interval_l336_336626

noncomputable def inequality_holds_for_all_x (m : ℝ) : Prop :=
  ∀ x : ℝ, -6 < (2 * x^2 + m * x - 4) / (x^2 - x + 1) ∧ 
           (2 * x^2 + m * x - 4) / (x^2 - x + 1) < 4

theorem values_of_m_in_interval (-2 < m ∧ m < 4) : inequality_holds_for_all_x m :=
sorry

end values_of_m_in_interval_l336_336626


namespace math_problem_l336_336442

def sum_of_remainders : Prop :=
  ∑ k in (Finset.range 8), ((11111 * k + 43210) % 13) = 60

theorem math_problem : sum_of_remainders := sorry

end math_problem_l336_336442


namespace correct_investment_allocation_l336_336109

noncomputable def investment_division (x : ℤ) : Prop :=
  let s := 2000
  let w := 500
  let rogers_investment := 2500
  let total_initial_capital := (5 / 2 : ℚ) * x
  let new_total_capital := total_initial_capital + rogers_investment
  let equal_share := new_total_capital / 3
  s + w = rogers_investment ∧ 
  (3 / 2 : ℚ) * x + s = equal_share ∧ 
  x + w = equal_share

theorem correct_investment_allocation (x : ℤ) (hx : 3 * x % 2 = 0) :
  x > 0 ∧ investment_division x :=
by
  sorry

end correct_investment_allocation_l336_336109


namespace smallest_n_three_consecutive_same_factors_l336_336121

/-- 
Prove that the smallest n such that three consecutive natural numbers n, n+1, n+2
have the same number of factors is 33.
-/
theorem smallest_n_three_consecutive_same_factors : 
  ∃ n : ℕ, (∀ m : ℕ, m = n ∨ m = n + 1 ∨ m = n + 2 → 
  (nat.factor_count m = nat.factor_count n ∧ 
  nat.factor_count m = nat.factor_count (n + 1) ∧ 
  nat.factor_count m = nat.factor_count (n + 2))) ∧ n = 33 :=
sorry

end smallest_n_three_consecutive_same_factors_l336_336121


namespace find_dolls_l336_336143

namespace DollsProblem

variables (S D : ℕ) -- Define S and D as natural numbers

-- Conditions as per the problem
def cond1 : Prop := 4 * S + 3 = D
def cond2 : Prop := 5 * S = D + 6

-- Theorem stating the problem
theorem find_dolls (h1 : cond1 S D) (h2 : cond2 S D) : D = 39 :=
by
  sorry

end DollsProblem

end find_dolls_l336_336143


namespace smallest_class_size_meeting_requirement_l336_336914

theorem smallest_class_size_meeting_requirement :
  ∃ x : ℕ, 4 * x + (x + 2) ≥ 50 ∧ 5 * x + 2 = 52 :=
by
  -- Definitions derived from conditions
  let x := 10
  have h1 : 4 * x + (x + 2) = 52 := by sorry
  have h2 : 52 ≥ 50 := by sorry
  exists x
  -- Given conditions
  exact ⟨h2, h1⟩

end smallest_class_size_meeting_requirement_l336_336914


namespace quadratic_roots_real_or_imaginary_l336_336271

theorem quadratic_roots_real_or_imaginary (a b c d: ℝ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: d > 0) 
(h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
∃ (A B C: ℝ), (A = a ∨ A = b ∨ A = c ∨ A = d) ∧ (B = a ∨ B = b ∨ B = c ∨ B = d) ∧ (C = a ∨ C = b ∨ C = c ∨ C = d) ∧ 
(A ≠ B) ∧ (A ≠ C) ∧ (B ≠ C) ∧ 
((1 - 4*B*C ≥ 0 ∧ 1 - 4*C*A ≥ 0 ∧ 1 - 4*A*B ≥ 0) ∨ (1 - 4*B*C < 0 ∧ 1 - 4*C*A < 0 ∧ 1 - 4*A*B < 0)) :=
by
  sorry

end quadratic_roots_real_or_imaginary_l336_336271


namespace soil_bags_needed_l336_336189

def raised_bed_length : ℝ := 8
def raised_bed_width : ℝ := 4
def raised_bed_height : ℝ := 1
def soil_bag_volume : ℝ := 4
def num_raised_beds : ℕ := 2

theorem soil_bags_needed : (raised_bed_length * raised_bed_width * raised_bed_height * num_raised_beds) / soil_bag_volume = 16 := 
by
  sorry

end soil_bags_needed_l336_336189


namespace parabola_focus_distance_l336_336280

noncomputable def PF (x₁ : ℝ) : ℝ := x₁ + 1
noncomputable def QF (x₂ : ℝ) : ℝ := x₂ + 1

theorem parabola_focus_distance 
  (x₁ x₂ : ℝ) (h₁ : x₂ = 3 * x₁ + 2) : 
  QF x₂ / PF x₁ = 3 :=
by
  sorry

end parabola_focus_distance_l336_336280


namespace sin_minus_cos_eq_one_sol_l336_336592

theorem sin_minus_cos_eq_one_sol (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = 1) :
  x = Real.pi / 2 ∨ x = Real.pi :=
sorry

end sin_minus_cos_eq_one_sol_l336_336592


namespace matrix_power_proof_l336_336397

section
  variables (A : Matrix (Fin 2) (Fin 2) ℝ)
  variables (v : Vector ℝ (Fin 2))
  variables (w : Vector ℝ (Fin 2))

  -- Given condition: A * v = w
  def condition := A.mulVec v = w

  -- Eigenvector and eigenvalue relation found from condition
  def eigen_condition := v = vector.of_fn ![7, -3] ∧ w = -2 • v

  -- The statement to prove
  theorem matrix_power_proof (h : eigen_condition A v w) :
    A ^ 4 • v = vector.of_fn ![112, -48] :=
  sorry
end

end matrix_power_proof_l336_336397


namespace runner_catches_up_again_l336_336863

variable {V1 V2 S t : ℝ}
variable (k : ℝ)
variable (initial_distance : ℝ) (final_distance : ℝ)
variable h1 : V1 > V2
variable h2 : V1 = 3 * V2
variable h3 : initial_distance = V2 * t 
variable h4 : final_distance = 2 * V2 * t
variable h5 : initial_distance = (1/2) * S
variable h6 : final_distance = k * S
variable h7 : V1 * t = (k + 1) * S

theorem runner_catches_up_again (h1 : V1 > V2) (h2 : V1 = 3 * V2) (h3 : initial_distance = V2 * t)
                               (h4 : final_distance = 2 * V2 * t) (h5 : initial_distance = (1/2) * S)
                               (h6 : final_distance = k * S) (h7 : V1 * t = (k + 1) * S) :
  (∃ k : ℝ, 2(k + 1)= 3k) ∧ k + 0.5 = 2.5 :=
  by
  sorry

end runner_catches_up_again_l336_336863


namespace Petya_wins_with_perfect_play_l336_336427

-- Define the game board
structure Board :=
  (size : ℕ)
  -- A function representing if a cell is white or black
  (cell : ℕ × ℕ → Prop)

-- Initialize a 100 × 100 board with all white cells
def initial_board : Board :=
  { size := 100,
    cell := λ ⟨i, j⟩, i < 100 ∧ j < 100 }

-- Define the conditions of the game
def Petya_turn_condition (b : Board) : Prop :=
  ∃ diagonal : ℕ × ℕ → Prop,
    -- Petya can paint one or more consecutive white cells on a diagonal
    (∀ (i j : ℕ), diagonal ⟨i, j⟩ → b.cell ⟨i, j⟩) ∧
    (∀ (i j k l : ℕ), diagonal ⟨i, j⟩ → diagonal ⟨k, l⟩ →
                    (i = j ∧ k = l ∧ (i+1 = k ∨ i-1 = k) ∨ (i = j ∧ k-1 = l) ))  

def Vasya_turn_condition (b : Board) : Prop :=
  ∃ vertical : ℕ → ℕ → Prop,
    -- Vasya can paint one or more consecutive white cells in a vertical line
    (∀ (i j : ℕ), vertical i j → b.cell ⟨i, j⟩) ∧
    (∀ (i j k l : ℕ), vertical i j → vertical k l → (i = k ∧ (j+1 = l ∨ j-1 = l)))

-- Define the winning condition: the player who cannot make a move loses
def winning_condition (petya_wins : Prop) : Prop :=
  ∃ (perfect_play : ∀ (turn : ℕ) (b : Board), (turn % 2 = 0 → Petya_turn_condition b) ∧ (turn % 2 = 1 → Vasya_turn_condition b)), petya_wins

-- Prove that Petya wins with perfect play
theorem Petya_wins_with_perfect_play :
  ∀ (b : Board),
  b = initial_board →
  (∃ perfect_play : ∀ (turn : ℕ), (turn % 2 = 0 → Petya_turn_condition b) ∧ (turn % 2 = 1 → Vasya_turn_condition b), 
  (winning_condition true)) :=
by
  intros b hb,
  -- Starting with the main diagonal condition and follow up the steps
  sorry

end Petya_wins_with_perfect_play_l336_336427


namespace f_0_value_a_range_l336_336444

variable (f : ℝ → ℝ)

noncomputable def condition1 := ∀ x y : ℝ, f(x + y) - f(y) = (x + 2*y + 2)*x
noncomputable def condition2 := f 2 = 12
noncomputable def condition3 := ∀ x : ℝ, f x = x^2 + 2*x + 4

theorem f_0_value (hf1 : condition1 f) (hf2 : condition2 f) : f 0 = 4 :=
sorry

theorem a_range (hf3 : condition3 f) (hx0 : ∃ x_0 : ℝ, 1 < x_0 ∧ x_0 < 4)
  (hax : ∀ x_0 : ℝ, f x_0 - 8 = a * x_0) : -1 < a ∧ a < 5 :=
sorry

end f_0_value_a_range_l336_336444


namespace ratio_of_coats_to_tshirts_l336_336028

variable (L_Tshirts L_Jeans L_Coats C_Tshirts C_Jeans C_Coats : ℝ)
variable (x : ℝ)

-- Conditions
def condition1 : L_Tshirts = 40 := rfl
def condition2 : L_Jeans = 20 := rfl
def condition3 : L_Coats = 40 * x := rfl
def condition4 : C_Tshirts = L_Tshirts / 4 := rfl
def condition5 : C_Jeans = 3 * L_Jeans := rfl
def condition6 : C_Coats = L_Coats / 4 := rfl
def condition7 (total_spent : ℝ) : total_spent = L_Tshirts + L_Jeans + L_Coats + C_Tshirts + C_Jeans + C_Coats := rfl

-- The total spending is $230
def assumption : 230 = L_Tshirts + L_Jeans + L_Coats + C_Tshirts + C_Jeans + C_Coats := 230 = 40 + 20 + 40 * x + (40 / 4) + (3 * 20) + (40 * x / 4)

-- Goal: To find the ratio of the amount Lisa spent on coats to the amount she spent on t-shirts
theorem ratio_of_coats_to_tshirts (h : assumption) : L_Coats / L_Tshirts = 2 :=
by sorry

end ratio_of_coats_to_tshirts_l336_336028


namespace largest_sample_number_l336_336721

theorem largest_sample_number (n : ℕ) (start interval total : ℕ) (h1 : start = 7) (h2 : interval = 25) (h3 : total = 500) (h4 : n = total / interval) : 
(start + interval * (n - 1) = 482) :=
sorry

end largest_sample_number_l336_336721


namespace problem1_problem2_l336_336685

-- Define vectors ⃗m and ⃗n
def vector_m (x : ℝ) : ℝ × ℝ := (sqrt 3 * cos x, -1)
def vector_n (x : ℝ) : ℝ × ℝ := (sin x, cos x ^ 2)

-- Problem 1: For x = π / 3, prove ⃗m ⋅ ⃗n = 1/2
theorem problem1 (x : ℝ) (hx : x = π / 3) :
  (vector_m x).1 * (vector_n x).1 + (vector_m x).2 * (vector_n x).2 = 1 / 2 :=
sorry

-- Problem 2: For x ∈ [0, π / 4], prove the implication ⃗m ⋅ ⃗n = √3 / 3 - 1/2 → cos 2x = (3√2 - √3) / 6
theorem problem2 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 4)
  (h_dot : (vector_m x).1 * (vector_n x).1 + (vector_m x).2 * (vector_n x).2 = sqrt 3 / 3 - 1 / 2) :
  cos (2 * x) = (3 * sqrt 2 - sqrt 3) / 6 :=
sorry

end problem1_problem2_l336_336685


namespace island_of_misfortune_l336_336150

def statement (n : ℕ) (knight : ℕ → Prop) (liar : ℕ → Prop) : Prop :=
  ∀ k : ℕ, k < n → (
    if k = 0 then ∀ m : ℕ, (m % 2 = 1) ↔ liar m
    else if k = 1 then ∀ m : ℕ, (m % 3 = 1) ↔ liar m
    else ∀ m : ℕ, (m % (k + 1) = 1) ↔ liar m
  )

theorem island_of_misfortune :
  ∃ n : ℕ, n >= 2 ∧ statement n knight liar
:= sorry

end island_of_misfortune_l336_336150


namespace find_m_n_intervals_of_increase_l336_336316

-- Problem (1)
theorem find_m_n
  (m n : ℝ)
  (f : ℝ → ℝ := λ x, m * sin(2 * x) + n * cos(2 * x))
  (h1 : f (π / 12) = sqrt 3)
  (h2 : f (2 * π / 3) = -2) :
  m = sqrt 3 ∧ n = 1 :=
sorry

-- Problem (2)
theorem intervals_of_increase
  (g : ℝ → ℝ := λ x, 2 * cos(2 * x)) :
  ∀ k : ℤ, increasing_on g (Icc (-π / 2 + k * π) (k * π)) :=
sorry

end find_m_n_intervals_of_increase_l336_336316


namespace binomial_12_6_eq_1848_l336_336969

theorem binomial_12_6_eq_1848 : (Nat.choose 12 6) = 1848 :=
  sorry

end binomial_12_6_eq_1848_l336_336969


namespace dot_product_of_vectors_l336_336254

noncomputable def a (x : ℝ) : ℝ × ℝ := (x, 1)
def b : ℝ × ℝ := (2, Real.sqrt 2)
def c (u v : ℝ) : ℝ × ℝ := (u, v)
def norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem dot_product_of_vectors (x u v : ℝ) (h_c : norm (c u v) = 2) (h_parallel : ∃ k, b = k • a x) (h_angle : Real.arccos ((a x).fst * u + (a x).snd * v) = (Real.pi / 6)) : 
  (a x).fst * u + (a x).snd * v = 3 :=
sorry

end dot_product_of_vectors_l336_336254


namespace willie_cream_l336_336623

theorem willie_cream : ∀ (total_cream needed_cream: ℕ), total_cream = 300 → needed_cream = 149 → (total_cream - needed_cream) = 151 :=
by
  intros total_cream needed_cream h1 h2
  sorry

end willie_cream_l336_336623


namespace parallel_ac_bk_l336_336424

theorem parallel_ac_bk (A B C M K : Type) [EuclideanGeometry A B C M K] :
  (equilateral_triangle A B C) ∧ (M ∈ line A B) ∧ (equilateral_triangle M K C) → parallel (line A C) (line B K) :=
begin
  sorry
end

end parallel_ac_bk_l336_336424


namespace vertices_on_incircle_l336_336773

open_locale classical

noncomputable theory

variables {A B C H1 H2 H3 T1 T2 T3 : Type}

-- Given:
-- 1. Triangle ABC
def is_triangle (A B C : Type) : Prop := true

-- 2. H1, H2, H3 are feet of the altitudes from A, B, C respectively
def feet_of_altitudes (A B C H1 H2 H3 : Type) : Prop := true

-- 3. The incircle of ABC touches BC, CA, and AB at T1, T2, and T3 respectively
def incircle_touches (ABC T1 T2 T3 : Type) : Prop := true

-- 4. l1, l2, l3 are lines symmetric to H2H3, H3H1, H1H2 with respect to T2T3, T3T1, T1T2 respectively
def symmetric_lines (H2 H3 H1 T2 T3 T1 l1 l2 l3 : Type) : Prop := true

-- Question:
-- Show that l1, l2, and l3 determine a triangle whose vertices lie on the incircle of ABC
theorem vertices_on_incircle
  (A B C H1 H2 H3 T1 T2 T3 : Type)
  (h1 : is_triangle A B C)
  (h2 : feet_of_altitudes A B C H1 H2 H3)
  (h3 : incircle_touches (Λ : Type) T1 T2 T3)
  (h4 : symmetric_lines H2 H3 H1 T2 T3 T1 (λ1 : Type) (λ2 : Type) (λ3 : Type))
  : ∀ λ1 λ2 λ3 : Type, ∃ X1 X2 X3, X1=X2=X3 ∧ X1 ∈ incircle (Λ : Type) :=
sorry

end vertices_on_incircle_l336_336773


namespace find_perpendicular_line_l336_336594

theorem find_perpendicular_line (x y : ℝ) (h : x - 2 * y = 3) : ∃ b : ℝ, ∀ x y : ℝ, y = -2 * x + b :=
by {
  let P : ℝ × ℝ := (1, 2),
  let l := {p : ℝ × ℝ | p.1 - 2 * p.2 = 3},
  let perpendicular_slope : ℝ := -2,
  use ((P.2) - perpendicular_slope * (P.1)),
  intro x,
  intro y,
  sorry
}

end find_perpendicular_line_l336_336594


namespace kolya_wins_perfect_play_l336_336905

theorem kolya_wins_perfect_play :
  ∀ (m n : ℕ) (initial target : Σ (x : ℕ), Σ (y : ℕ), x ≤ m ∧ y ≤ n),
  initial.1 = 1 ∧ initial.2.1 = 1 ∧ 
  target.1 = 5 ∧ target.2.1 = 9 ∧
  (∀ a b, (a + b) % 2 = 0 → (1 ≤ a ∧ a ≤ m) ∧ (1 ≤ b ∧ b ≤ n)) ∧
  (∀ a b, (a + b) % 2 ≠ 0 → (1 ≤ a ∧ a ≤ m) ∧ (1 ≤ b ∧ b ≤ n)) →
  (m = 5 ∧ n = 9 → "Kolya wins with perfect play") :=
sorry

end kolya_wins_perfect_play_l336_336905


namespace parallelogram_product_l336_336723

theorem parallelogram_product (EF FG GH HE : ℝ) (x z : ℝ)
  (hEF : EF = 46) 
  (hFG : FG = 4 * (z^3) + 1) 
  (hGH : GH = 3 * x + 6) 
  (hHE : HE = 35) 
  (hEFGH_eq : EF = GH) 
  (hFGHE_eq : FG = HE) :
  (x * z) = (40 / 3) * Real.cbrt(8.5) :=
by
  sorry

end parallelogram_product_l336_336723


namespace solution_set_empty_range_a_l336_336710

theorem solution_set_empty_range_a (a : ℝ) :
  (∀ x : ℝ, ¬((a - 1) * x^2 + 2 * (a - 1) * x - 4 ≥ 0)) ↔ -3 < a ∧ a ≤ 1 :=
by
  sorry

end solution_set_empty_range_a_l336_336710


namespace min_sum_of_factors_l336_336847

theorem min_sum_of_factors (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 3960) : 
  a + b + c = 72 :=
sorry

end min_sum_of_factors_l336_336847


namespace count_no_carry_pairs_l336_336621

-- Define the range of integers from 1500 to 2500
def range_integers : List ℕ := List.range' 1500 (2500 - 1500 + 1)

-- Define a function to check for no carry condition when adding two consecutive integers
def no_carry (n m : ℕ) : Prop :=
  let digits := List.zip (n.digits 10) (m.digits 10)
  ∀ (a b : ℕ) in digits, a + b < 10

-- Count pairs of consecutive integers that satisfy the no carry condition
def count_valid_pairs (lst : List ℕ) : ℕ :=
  (lst.zip (lst.tail)).count (λ (p : ℕ × ℕ), no_carry p.1 p.2)

-- The theorem to prove the total number of such valid pairs
theorem count_no_carry_pairs : count_valid_pairs range_integers = 1100 :=
by
  sorry

end count_no_carry_pairs_l336_336621


namespace find_moles_of_NaOH_combined_l336_336603

/-- Define the chemical reaction constants -/
def reaction (NaOH H2SO4 NaHSO4 H2O : ℕ) : Prop := 
  NaOH + H2SO4 = NaHSO4 + H2O

-- Define the conditions given in the problem
variables (NaHSO4_produced NaOH_required H2SO4_used : ℕ)
hypothesis H_conditions : NaHSO4_produced = 3 ∧ H2SO4_used = 3
hypothesis H_balanced : reaction NaOH_required 3 NaHSO4_produced 0

-- Prove the number of moles of Sodium hydroxide combined is 3
theorem find_moles_of_NaOH_combined : NaOH_required = 3 :=
  sorry

end find_moles_of_NaOH_combined_l336_336603


namespace sin_value_l336_336283

theorem sin_value (α : ℝ) (h : cos (α + π / 6) = 1 / 3) : 
  sin (2 * α - π / 6) = 7 / 9 :=
sorry

end sin_value_l336_336283


namespace sphere_volume_proof_l336_336288

noncomputable def sphere_radius : ℝ := 24
noncomputable def cone_height : ℝ := 2 * sphere_radius
noncomputable def sphere_surface_area : ℝ := 4 * Real.pi * sphere_radius^2
noncomputable def cone_surface_area_eq_sphere_surface_area (r : ℝ) : Prop := Real.pi * r * (r + Real.sqrt (r^2 + cone_height^2)) = sphere_surface_area
noncomputable def cone_volume (r : ℝ) : ℝ := (1/3) * Real.pi * r^2 * cone_height

theorem sphere_volume_proof : ∃ (r : ℝ), cone_surface_area_eq_sphere_surface_area r ∧ cone_volume r = 12288 * Real.pi :=
by
  sorry

end sphere_volume_proof_l336_336288


namespace max_children_l336_336789

theorem max_children (x : ℕ) (h1 : x * (x - 2) + 2 * 5 = 58) : x = 8 :=
by
  sorry

end max_children_l336_336789


namespace proof_theorem_l336_336456

noncomputable def proof_problem (a b c : ℝ) := 
  (2 * b = a + c) ∧ 
  (2 / b = 1 / a + 1 / c ∨ 2 / a = 1 / b + 1 / c ∨ 2 / c = 1 / a + 1 / b) ∧ 
  (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0)

theorem proof_theorem (a b c : ℝ) (h : proof_problem a b c) :
  (a = b ∧ b = c) ∨ 
  (∃ (x : ℝ), x ≠ 0 ∧ a = -4 * x ∧ b = -x ∧ c = 2 * x) :=
by
  sorry

end proof_theorem_l336_336456


namespace problem_solution_l336_336552

-- Define monotonicity and inverse functions
def is_monotonic {α β : Type*} [LinearOrder α] [LinearOrder β] (f : α → β) : Prop :=
  ∀ ⦃x y⦄, x ≤ y → f x ≤ f y

def inverse_functions {α β : Type*} [LinearOrder α] [LinearOrder β] (f : α → β) (g : β → α) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

-- Define symmetry condition
def is_symmetric_origin {α : Type*} [LinearOrderedAddCommGroup α] (f : α → α) : Prop :=
  ∀ x, f x = -f (-x)

-- Define odd function and inverse
def is_odd {α : Type*} [LinearOrderedAddCommGroup α] (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

def has_inverse {α β : Type*} [LinearOrder α] [LinearOrder β] (f : α → β) (g : β → α) : Prop :=
  ∀ x, f (g x) = x ∧ g (f x) = x

theorem problem_solution (f g : ℝ → ℝ) :
  (inverse_functions f g ∧ is_monotonic f ∧ is_monotonic g) →
  (∀ x, f x = -f (-x)) → ¬(∀ h, is_odd h → ∃ h_inv, has_inverse h h_inv) :=
by
  intro h₁ h₂ h₃
  sorry

end problem_solution_l336_336552


namespace percentage_increase_l336_336066

variable (A B y : ℝ)

theorem percentage_increase (h1 : B > A) (h2 : A > 0) :
  B = A + y / 100 * A ↔ y = 100 * (B - A) / A :=
by
  sorry

end percentage_increase_l336_336066


namespace correct_conclusions_l336_336303

def f (x : ℝ) : ℝ := Real.log x

def sequence (a₁ : ℝ) (n : ℕ) : ℝ
| 0     := a₁
| (k+1) := Real.log (sequence a₁ k) - 1

def is_arithmetic_progression (a : ℕ → ℝ) (k : ℕ) : Prop :=
  ∀ i j, i < k → j < k → (a i - a j) = (i - j) * (a 1 - a 0)

theorem correct_conclusions (a₁ : ℝ) (h₁ : 0 < a₁) :
  (∀ n, n ≥ 2 → sequence a₁ n = Real.log (sequence a₁ (n-1)) - 1) ∧
  (∀ n, n ≥ 2 → sequence a₁ n ≤ sequence a₁ (n-1) - 2) ∧
  (∃ k, is_arithmetic_progression (sequence a₁) (k+1) → k = 2) ∧
  ¬ (0 < a₁ ∧ a₁ < Real.exp 1) :=
sorry

end correct_conclusions_l336_336303


namespace compute_binom_12_6_eq_1848_l336_336981

def binomial (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

theorem compute_binom_12_6_eq_1848 : binomial 12 6 = 1848 :=
by
  sorry

end compute_binom_12_6_eq_1848_l336_336981


namespace time_for_each_student_l336_336532

-- Define the conditions as variables
variables (num_students : ℕ) (period_length : ℕ) (num_periods : ℕ)
-- Assume the conditions from the problem
def conditions := num_students = 32 ∧ period_length = 40 ∧ num_periods = 4

-- Define the total time available
def total_time (num_periods period_length : ℕ) := num_periods * period_length

-- Define the time per student
def time_per_student (total_time num_students : ℕ) := total_time / num_students

-- State the theorem to be proven
theorem time_for_each_student : 
  conditions num_students period_length num_periods →
  time_per_student (total_time num_periods period_length) num_students = 5 := sorry

end time_for_each_student_l336_336532


namespace remainder_29_times_171997_pow_2000_mod_7_l336_336091

theorem remainder_29_times_171997_pow_2000_mod_7 :
  (29 * 171997^2000) % 7 = 4 :=
by
  sorry

end remainder_29_times_171997_pow_2000_mod_7_l336_336091


namespace max_value_of_m_l336_336222

theorem max_value_of_m : ∃ (m : ℤ), (∃ (C D : ℤ), (3 * D + C) = m ∧ (C * D) = -60) ∧ 
  (∀ (C' D' : ℤ), (C' * D') = -60 → (3 * D' + C') ≤ m) ∧ m = 57 :=
begin
  sorry

end max_value_of_m_l336_336222


namespace Wendy_not_recycled_bags_l336_336489

theorem Wendy_not_recycled_bags:
  ∀ (points_per_bag total_points total_bags bags_not_recycled : ℕ),
  points_per_bag = 5 →
  total_points = 45 →
  total_bags = 11 →
  total_bags - total_points / points_per_bag = bags_not_recycled →
  bags_not_recycled = 2 :=
by
  intros points_per_bag total_points total_bags bags_not_recycled
  assume h1 : points_per_bag = 5
  assume h2 : total_points = 45
  assume h3 : total_bags = 11
  assume h4 : total_bags - total_points / points_per_bag = bags_not_recycled
  sorry

end Wendy_not_recycled_bags_l336_336489


namespace solve_equation_l336_336328

theorem solve_equation (x : ℝ) (h : 3 * x + 2 = 11) : 5 * x + 3 = 18 :=
sorry

end solve_equation_l336_336328


namespace smallest_prime_after_seven_nonprimes_l336_336500

open Nat

/-- Prove that the smallest prime number that occurs after a sequence of seven consecutive positive
integers, all of which are nonprime, is 97. -/
theorem smallest_prime_after_seven_nonprimes :
  ∃ p : ℕ, (p > 97) ∧ (Prime p) ∧
  (∀ n : ℕ, n ∈ range (p - 8, p) → ¬ Prime n) := sorry

end smallest_prime_after_seven_nonprimes_l336_336500


namespace integral_arcsin_squared_plus_one_l336_336951

noncomputable def integrand (x : ℝ) : ℝ :=
  ((Real.arcsin x) ^ 2 + 1) / Real.sqrt (1 - x ^ 2)

theorem integral_arcsin_squared_plus_one :
  ∫ x in 0..Real.sin 1, integrand x = 4 / 3 :=
by
  sorry

end integral_arcsin_squared_plus_one_l336_336951


namespace complex_expression_real_l336_336278

open Complex

theorem complex_expression_real (z1 z2 : ℂ) : (z1 * conj z2 + conj z1 * z2).im = 0 :=
by
  sorry

end complex_expression_real_l336_336278


namespace percentage_of_women_in_study_group_l336_336911

theorem percentage_of_women_in_study_group
  (W : ℝ)
  (H1 : 0 ≤ W ∧ W ≤ 1)
  (H2 : 0.60 * W = 0.54) :
  W = 0.9 :=
sorry

end percentage_of_women_in_study_group_l336_336911


namespace eval_trig_log_exp_l336_336580

theorem eval_trig_log_exp :
  | (4^(2 - 8*(Real.sin 3 - 12)^2))^2 | - | Real.cos(|Real.sin (5 * Real.pi / 6) - Real.cos (11 * Real.pi / 3)| + Real.log 2)^3 | = 0.4551 :=
by
  sorry

end eval_trig_log_exp_l336_336580


namespace biased_die_probability_l336_336907

theorem biased_die_probability :
  let p := 1 / 3
  let P_e := 2 * p
  let P_o := p
  P_e ^ 3 + P_o ^ 3 + 3 * P_e ^ 2 * P_o = 7 / 9 :=
by
  simp [P_e, P_o]
  sorry

end biased_die_probability_l336_336907


namespace cars_with_neither_l336_336513

theorem cars_with_neither (total_cars : ℕ) (cars_with_airbags : ℕ) (cars_with_power_windows : ℕ) (cars_with_both : ℕ) :
  total_cars = 65 →
  cars_with_airbags = 45 →
  cars_with_power_windows = 30 →
  cars_with_both = 12 →
  (total_cars - (cars_with_airbags + cars_with_power_windows - cars_with_both)) = 2 :=
by
  intros h_total h_airbags h_windows h_both
  rw [h_total, h_airbags, h_windows, h_both]
  norm_num
  sorry

end cars_with_neither_l336_336513


namespace finite_hedgehogs_on_wonder_island_l336_336069

def Hedgehog :=
  { x : ℝ × ℝ // ∀ (y z : ℝ × ℝ), 
    (dist x y = 1 ∧ dist x z = 1 ∧ 
     angle x y z = 2 * π / 3 ∧ angle x z y = 2 * π / 3) }

theorem finite_hedgehogs_on_wonder_island
  (H : set Hedgehog)
  (flat : ∀ h ∈ H, h.1.2 = 0) -- Hedgehogs lying flat on the island
  (no_touch : ∀ (h1 h2 ∈ H), h1 ≠ h2 → ∀ (x ∉ {h1.1, h2.1} ), dist h1.1 h2.1 ≥ 1) 
  : H.finite :=
sorry

end finite_hedgehogs_on_wonder_island_l336_336069


namespace sum_of_gcd_and_lcm_l336_336494

-- Definitions of gcd and lcm for the conditions
def gcd_of_42_and_56 : ℕ := Nat.gcd 42 56
def lcm_of_24_and_18 : ℕ := Nat.lcm 24 18

-- Lean statement that the sum of the gcd and lcm is 86
theorem sum_of_gcd_and_lcm : gcd_of_42_and_56 + lcm_of_24_and_18 = 86 := by
  sorry

end sum_of_gcd_and_lcm_l336_336494


namespace dividend_is_correct_l336_336897

def quotient : ℕ := 36
def divisor : ℕ := 85
def remainder : ℕ := 26

theorem dividend_is_correct : divisor * quotient + remainder = 3086 := by
  sorry

end dividend_is_correct_l336_336897


namespace fraction_of_population_married_is_correct_l336_336799

-- Definitions for the given conditions
variables {M W : ℕ} 
def fraction_of_men_married : ℚ := 2 / 3
def fraction_of_women_married : ℚ := 3 / 5
def total_population : ℚ := M + (10 / 9) * M
def married_population : ℚ := (4 / 3) * M

-- The proof statement
theorem fraction_of_population_married_is_correct 
  (M : ℕ) (W : ℕ) 
  (h1 : W = (10 / 9 : ℚ) * M) 
  (h2 : married_population / total_population = 12 / 19) : 
  married_population / total_population = 12 / 19 :=
by sorry

end fraction_of_population_married_is_correct_l336_336799


namespace sum_first_10_terms_l336_336268

variable (a : Nat → ℝ)
noncomputable def common_difference := a 1 - a 0

def arithmetic_sequence := ∀ n : Nat, a (n + 1) - a n = common_difference a

def geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem sum_first_10_terms (h_seq : arithmetic_sequence a)
  (h1 : a 0 + a 2 = 4)
  (h_geo : geometric_sequence (a 1) (a 2) (a 4))
  (h_diff_non_zero : common_difference a ≠ 0) :
  (∑ i in Finset.range 10, a i) = 90 := by
  sorry

end sum_first_10_terms_l336_336268


namespace factor_x_squared_minus_169_l336_336587

theorem factor_x_squared_minus_169 (x : ℝ) : x^2 - 169 = (x - 13) * (x + 13) := 
by
  -- Recognize that 169 is a perfect square
  have h : 169 = 13^2 := by norm_num
  -- Use the difference of squares formula
  -- Sorry is used to skip the proof part
  sorry

end factor_x_squared_minus_169_l336_336587


namespace maximal_m_divides_f_l336_336068

def f (n : ℕ) : ℤ := (2 * n - 7) * 3 ^ n + 9

theorem maximal_m_divides_f : ∀ n : ℕ, 0 < n → ∃ m : ℕ, m = 6 ∧ m ∣ f n :=
by
  intro n hn
  use 6
  constructor
  · rfl
  · sorry

end maximal_m_divides_f_l336_336068


namespace square_area_l336_336837

variable (l r s : ℝ)
variable (breadth : ℝ := 10)
variable (area_rect : ℝ := 200)

-- Conditions:
-- 1. The length of the rectangle is two-fifths of the radius of the circle.
-- 2. The radius of the circle is equal to the side of a square.
-- 3. The area of the rectangle with a breadth of 10 units is 200 sq. units.

theorem square_area :
  (l = (2/5) * r) →
  (r = s) →
  (area_rect = l * breadth) →
  (s = 50) →
  s^2 = 2500 :=
by
  intros h1 h2 h3 h4
  rw [h4]
  simp [h4]
  sorry

end square_area_l336_336837


namespace norm_inequality_equality_condition_l336_336038

noncomputable def L2Space (Ω : Type*) (𝓕 : Type*) (ℙ : Type*) : Type :=
sorry -- definition of L2 space

noncomputable def Expectation {Ω : Type*} {𝓕 𝓨 : Type*} (ξ : L2Space Ω 𝓕 ℙ) (𝓨 : 𝓕 → 𝓕) : L2Space Ω 𝓨 ℙ :=
sorry -- definition of conditional expectation

theorem norm_inequality {Ω : Type*} {𝓕 : Type*} (ℙ : Type*) 
  (ξ : L2Space Ω 𝓕 ℙ) (𝓖 : 𝓕 → 𝓕) [is_σ_subalgebra 𝓖 𝓕] : 
  ∥ξ∥ ≥ ∥Expectation ξ 𝓖∥ :=
begin
  sorry
end

theorem equality_condition {Ω : Type*} {𝓕 : Type*} (ℙ : Type*)
  (ξ : L2Space Ω 𝓕 ℙ) (𝓖 : 𝓕 → 𝓕) [is_σ_subalgebra 𝓖 𝓕] :
  ∥ξ∥ = ∥Expectation ξ 𝓖∥ ↔ ∀ₚ ω, ξ ω = (Expectation ξ 𝓖) ω :=
begin
  sorry
end

end norm_inequality_equality_condition_l336_336038


namespace janice_overtime_pay_l336_336744

noncomputable def regular_rate : ℝ := 10
noncomputable def regular_hours : ℝ := 40
noncomputable def total_hours_worked : ℝ := 60
noncomputable def total_earnings : ℝ := 700

theorem janice_overtime_pay : 
    let overtime_hours := total_hours_worked - regular_hours,
        earnings_first_40_hours := regular_hours * regular_rate,
        overtime_earnings := total_earnings - earnings_first_40_hours,
        overtime_rate := overtime_earnings / overtime_hours
    in overtime_rate = 15 := 
by
  sorry

end janice_overtime_pay_l336_336744


namespace g_of_32_div_49_lt_0_l336_336410

def f (g : ℚ → ℚ) : Prop :=
  (∀ (a b : ℚ), a > 0 → b > 0 → g (a * b) = g a + g b) ∧
  (∀ (p : ℕ) (k : ℤ), prime p → g ((p : ℚ) ^ k) = k * p)

theorem g_of_32_div_49_lt_0 (g : ℚ → ℚ) (hg : f g) :
  g (32/49) < 0 :=
sorry

end g_of_32_div_49_lt_0_l336_336410


namespace compute_binom_12_6_eq_1848_l336_336979

def binomial (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

theorem compute_binom_12_6_eq_1848 : binomial 12 6 = 1848 :=
by
  sorry

end compute_binom_12_6_eq_1848_l336_336979


namespace boundary_value_problem_to_integral_equation_l336_336480

theorem boundary_value_problem_to_integral_equation 
  (λ x : ℝ) (y G : ℝ → ℝ → ℝ):
  (∀ x, y'' + λ * y = x ) ∧ (y 0 = 0) ∧ (y (π / 2) = 0) ∧ 
  (∀ x ξ, G(x, ξ) = 
    if 0 ≤ x ∧ x ≤ ξ then ((2 / π * ξ - 1) * x)
    else if ξ ≤ x ∧ x ≤ π / 2 then ((2 / π * x - 1) * ξ)) →
  (∀ x ∈ Icc (0:ℝ) (π / 2), 
    y x + λ * ∫ ξ in 0..π/2, G x ξ * y ξ = x^3 / 6 - (π^2 / 24) * x) :=
  sorry

end boundary_value_problem_to_integral_equation_l336_336480


namespace friend_made_mistakes_and_needs_security_measures_l336_336890

namespace OnlineSecurity

-- Conditions definitions
def email_address := "aliexpress@best_prices.net"
def website_url := "aliexpres__best_prices.net/shop"
def personal_info := {last_name : String, first_name : String, card_number : String, cvv_code : String, address : String}
def verification_block := true
def sms_password_entered := true
def purchase_confirmed := true

-- Mistakes made by the friend
def mistakes_made (trusted_email : Bool) (trusted_url : Bool) (entered_personal_info : Bool) 
                  (believed_offer : Bool) (verified_legitimacy : Bool) : Prop := 
trusted_email ∧ trusted_url ∧ entered_personal_info ∧ believed_offer ∧ ¬ verified_legitimacy

-- Additional security measures
def additional_security_measures (secure_network : Bool) (antivirus : Bool) 
                                 (updated_apps : Bool) (check_https : Bool) 
                                 (strong_password : Bool) (two_factor_auth : Bool) 
                                 (recognize_bank_protocols : Bool) : Prop :=
secure_network ∧ antivirus ∧ updated_apps ∧ check_https ∧ strong_password ∧ two_factor_auth ∧ recognize_bank_protocols

-- Statement for the mathematically equivalent proof problem in Lean
theorem friend_made_mistakes_and_needs_security_measures :
  let trusted_email := true,
      trusted_url := true,
      entered_personal_info := true,
      believed_offer := true,
      verified_legitimacy := false,
      secure_network := true,
      antivirus := true,
      updated_apps := true,
      check_https := true,
      strong_password := true,
      two_factor_auth := true,
      recognize_bank_protocols := true in
  mistakes_made trusted_email trusted_url entered_personal_info believed_offer verified_legitimacy ∧
  additional_security_measures secure_network antivirus updated_apps check_https strong_password two_factor_auth recognize_bank_protocols :=
sorry

end OnlineSecurity

end friend_made_mistakes_and_needs_security_measures_l336_336890


namespace parallelogram_perimeter_36_l336_336348

-- Definitions for the problem conditions
def isParallelogram (A B C D : Point) : Prop := 
  (A B = C D) ∧ (A D = B C) ∧ (A B ∥ C D) ∧ (A D ∥ B C)

def perpendicular (A B C : Point) : Prop :=
  ∠ (A, B) = 90 ∧ ∠ (A, C) = 90 -- Points A, B, C form perpendicular

def diagonalLength (A, B : Point) (length : ℝ): Prop := 
  (dist A B) = length

noncomputable def parallelogramPerimeter (A B C D : Point) : ℝ := 
  2 * (dist A B) + 2 * (dist A D)

-- Given a parallelogram with the conditions, prove the perimeter is 36
theorem parallelogram_perimeter_36
  (A B C D : Point)
  (h_parallelogram : isParallelogram A B C D)
  (h_perpendiculars : ∃ P Q : Point, perpendicular A P B ∧ perpendicular A Q D ∧ dist A P = 12 ∧ dist A Q = 12)
  (h_diagonal : diagonalLength A C 15) :
  parallelogramPerimeter A B C D = 36 :=
sorry

end parallelogram_perimeter_36_l336_336348


namespace number_of_main_characters_l336_336372

namespace TVShow

def total_payment_per_episode : ℕ := 285000
def payment_per_minor_character : ℕ := 15000
def number_of_minor_characters : ℕ := 4
def payment_per_main_character : ℕ := 3 * payment_per_minor_character

theorem number_of_main_characters :
  ∃ M : ℕ, (number_of_minor_characters * payment_per_minor_character
  + M * payment_per_main_character = total_payment_per_episode ∧ M = 5) :=
begin
  -- proof omitted
  sorry
end

end TVShow

end number_of_main_characters_l336_336372


namespace maximum_value_n_permutation_l336_336409

open Nat

theorem maximum_value_n_permutation : ∃ (a : Fin 17 → Fin 18), (PermOfSeq a (Finset.range 17) ∧ 
  ( ∏ i : Fin 17, (a i) - (a (i + 1) % 17) ) = (6 : ℕ) ^ 17 ) :=
sorry

end maximum_value_n_permutation_l336_336409


namespace eccentricity_of_ellipse_l336_336077

theorem eccentricity_of_ellipse:
  ∃ (a b c : ℝ), 
    (0 < b ∧ b < a) ∧ 
    (b^2 + c^2 = a^2) ∧ 
    (let x_mid := 1/2 in 
      let line_eq := 3*x_mid - 2 in 
      let ell_eq := (9*b^2 + a^2) * x_mid^2 - 12*b^2 * x_mid + 4*b^2 - a^2*b^2 in 
      ∃ (x1 x2 : ℝ), 
        x1 + x2 = 12*b^2/(9*b^2 + a^2) ∧ 
        12*b^2 = 9*b^2 + a^2) ∧ 
      (let e := c/a in 
        e = sqrt(2/3)) := by
sorry

end eccentricity_of_ellipse_l336_336077


namespace value_of_t_l336_336805

theorem value_of_t (m t : ℂ) (hm : m = 4) (ht : t = 4 + 100 * complex.I) (h : 4 * m - t = 8000) : 
    t = -7988 - 100 * complex.I := 
by 
  sorry

end value_of_t_l336_336805


namespace rate_percent_simple_interest_l336_336148

theorem rate_percent_simple_interest (P SI T : ℝ) (hP : P = 720) (hSI : SI = 180) (hT : T = 4) :
  (SI = P * (R / 100) * T) → R = 6.25 :=
by
  sorry

end rate_percent_simple_interest_l336_336148


namespace smallest_prime_after_seven_consecutive_nonprimes_l336_336499

theorem smallest_prime_after_seven_consecutive_nonprimes
  (N : Type) [linear_order N] [has_zero N] [has_add N] [has_one N] [is_prime N]
  (s : set N) (pos : ∀ n ∈ s, n > 0) :
  ∃ p ∈ s, is_prime p ∧ 
    (∀ (n1 n2 : N), n1 ∈ s ∧ n2 ∈ s → n1 < n2 → (n2 - n1) > 8 ∧ n2 = 53 :=
begin
  sorry
end

end smallest_prime_after_seven_consecutive_nonprimes_l336_336499


namespace second_group_people_count_l336_336475

theorem second_group_people_count (initial_people food_per_day days: ℕ) (remaining_food after_meeting_days: ℕ) : 
    initial_people = 9 →
    food_per_day = 1 →
    days = 5 →
    remaining_food = 36 →
    after_meeting_days = 3 →
    ∀ x : ℕ, 3 * (initial_people + x) = remaining_food → x = 3 :=
by
  intros h1 h2 h3 h4 h5 x h6
  have h7 : initial_people = 9 := h1
  have h8 : food_per_day = 1 := h2
  have h9 : days = 5 := h3
  have h10 : remaining_food = 36 := h4
  have h11 : after_meeting_days = 3 := h5
  simp at h6
  sorry

end second_group_people_count_l336_336475


namespace sharon_trip_distance_l336_336228

-- Definitions based on the problem's conditions
def usual_time : ℕ := 180 -- usual travel time in minutes
def reduced_speed : ℕ := 30 -- speed reduction in miles per hour
def total_time_with_rain : ℕ := 330 -- total time taken on the rainy day

/-- Given the conditions, the total distance from Sharon's house to her mother's house is 171 miles. -/
theorem sharon_trip_distance : x: ℕ :=
  ∃ (x : ℝ), 45 + (3 * x / 4) / ((x / 180) - 0.5) = 330 ∧ x = 171
:= sorry

end sharon_trip_distance_l336_336228


namespace bella_steps_l336_336947

/-- Bella begins to walk from her house toward her friend Ella's house. At the same time, Ella starts to skate toward Bella's house. They each maintain a constant speed, and Ella skates three times as fast as Bella walks. The distance between their houses is 10560 feet, and Bella covers 3 feet with each step. Prove that Bella will take 880 steps by the time she meets Ella. -/
theorem bella_steps 
  (d : ℝ)    -- distance between their houses in feet
  (s_bella : ℝ)    -- speed of Bella in feet per minute
  (s_ella : ℝ)    -- speed of Ella in feet per minute
  (steps_per_ft : ℝ)    -- feet per step of Bella
  (h1 : d = 10560)    -- distance between their houses is 10560 feet
  (h2 : s_ella = 3 * s_bella)    -- Ella skates three times as fast as Bella
  (h3 : steps_per_ft = 3)    -- Bella covers 3 feet with each step
  : (10560 / (4 * s_bella)) * s_bella / 3 = 880 :=
by
  -- proof here 
  sorry

end bella_steps_l336_336947


namespace cookies_yesterday_l336_336327

theorem cookies_yesterday (cookies_today : ℕ) (difference : ℕ)
  (h1 : cookies_today = 140)
  (h2 : difference = 30) :
  cookies_today - difference = 110 :=
by
  sorry

end cookies_yesterday_l336_336327


namespace geometry_statements_correct_l336_336140

/-- A type for points and a type for planes -/
structure Point : Type :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure Plane : Type :=
(a : ℝ)
(b : ℝ)
(c : ℝ)
(d : ℝ)

def three_points_determine_plane (p1 p2 p3 : Point) : Plane := sorry
def on_circle (center : Point) (radius : ℝ) (p : Point) : Prop := sorry
def two_parallel_lines_determine_plane (l1 l2 : Plane) : Plane := sorry

theorem geometry_statements_correct : 
  (∀ p1 p2 p3 : Point, three_points_determine_plane p1 p2 p3) ∧
  (∀ (center : Point) (radius : ℝ) (p1 p2 p3 : Point), on_circle center radius p1 
    ∧ on_circle center radius p2 ∧ on_circle center radius p3 → three_points_determine_plane p1 p2 p3) ∧
  ¬(∀ (center : Point) (radius : ℝ) (p1 p2 : Point), on_circle center radius p1 
    ∧ on_circle center radius p2 → three_points_determine_plane center p1 p2) ∧
  ¬(∀ (l1 l2 : Plane), l1 ≠ l2 → two_parallel_lines_determine_plane l1 l2) := 
sorry

end geometry_statements_correct_l336_336140


namespace jack_recycled_cans_l336_336369

theorem jack_recycled_cans :
  ∀ (bottle_count : ℕ) (total_money : ℚ) (bottle_rate : ℚ) (can_rate : ℚ),
  bottle_count = 80 →
  total_money = 15 →
  bottle_rate = 0.10 →
  can_rate = 0.05 →
  ∃ (can_count : ℕ), (can_count * can_rate) = (total_money - (bottle_count * bottle_rate)) ∧ can_count = 140 :=
by
  intros bottle_count total_money bottle_rate can_rate Hbottles Htotal Hbrate Hcrate
  use 140
  split
  sorry
  refl

end jack_recycled_cans_l336_336369


namespace computation_of_sqrt_expr_l336_336213

theorem computation_of_sqrt_expr : 
  (Real.sqrt ((52 : ℝ) * 51 * 50 * 49 + 1) = 2549) := 
by
  sorry

end computation_of_sqrt_expr_l336_336213


namespace scout_weekend_earnings_l336_336058

theorem scout_weekend_earnings : 
  let base_pay_per_hour := 10.00
  let tip_per_customer := 5.00
  let saturday_hours := 4
  let saturday_customers := 5
  let sunday_hours := 5
  let sunday_customers := 8
  in
  (saturday_hours * base_pay_per_hour + saturday_customers * tip_per_customer) +
  (sunday_hours * base_pay_per_hour + sunday_customers * tip_per_customer) = 155.00 := sorry

end scout_weekend_earnings_l336_336058


namespace equipment_value_after_n_years_l336_336906

variables (a : ℝ) (b : ℝ) (n : ℕ)

def depreciation_rate := b / 100
def value_after_n_years := a * (1 - depreciation_rate) ^ n

theorem equipment_value_after_n_years :
  value_after_n_years a b n = a * (1 - b / 100) ^ n := 
sorry

end equipment_value_after_n_years_l336_336906


namespace average_weight_of_whole_class_is_37_25_l336_336100

def numberOfStudentsA : ℕ := 36
def numberOfStudentsB : ℕ := 44

def averageWeightA : ℝ := 40
def averageWeightB : ℝ := 35

def totalWeightA : ℝ := numberOfStudentsA * averageWeightA
def totalWeightB : ℝ := numberOfStudentsB * averageWeightB

def totalWeight : ℝ := totalWeightA + totalWeightB
def totalStudents : ℕ := numberOfStudentsA + numberOfStudentsB

def averageWeightClass : ℝ := totalWeight / totalStudents

theorem average_weight_of_whole_class_is_37_25 : averageWeightClass = 37.25 := 
by
sory

end average_weight_of_whole_class_is_37_25_l336_336100


namespace real_roots_prime_equation_l336_336652

noncomputable def has_rational_roots (p q : ℕ) : Prop :=
  ∃ x : ℚ, x^2 + p^2 * x + q^3 = 0

theorem real_roots_prime_equation (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  has_rational_roots p q ↔ (p = 3 ∧ q = 2) :=
sorry

end real_roots_prime_equation_l336_336652


namespace find_BE_l336_336368

variables 
  (A B C D E : Type)
  [trapezoid ABCD]
  [isosceles_trapezoid ABCD]
  (BC AD AE DC DE BE : ℝ)
  (cos_αβ : ℝ)
  (α β : ℝ)

def ABCD_isosceles : Prop := isosceles_trapezoid ABCD ∧ (BC > AD) ∧ (BC ∥ AD)
def ECDA_isosceles : Prop := isosceles_trapezoid ECDA ∧ (AE > DC) ∧ (AE ∥ DC)
def DE_value : Prop := DE = 7
def cos_condition : Prop := cos (α + β) = 1 / 3

theorem find_BE :
  ABCD_isosceles A B C D →
  ECDA_isosceles E C D A →
  DE_value (DE) →
  cos_condition (cos_αβ) →
  BE = 14 / 3 :=
by
  sorry

end find_BE_l336_336368


namespace terms_before_5_l336_336694

def arithmetic_sequence (a₁ d n : ℤ) : ℤ :=
  a₁ + d * (n - 1)

theorem terms_before_5 : 
  ∀ (a₁ d : ℤ), a₁ = 105 → d = -5 → 
  ∃ n : ℤ, arithmetic_sequence a₁ d n = 5 ∧ n - 1 = 20 :=
by
  intros a₁ d h₁ h₂
  use 21
  split
  { rw [arithmetic_sequence, h₁, h₂]
    norm_num },
  { norm_num }

end terms_before_5_l336_336694


namespace number_with_two_consecutive_ones_l336_336319

def number_of_12_digit_numbers (d : ℕ) : ℕ := 3^12 - a d - b d

def a : ℕ → ℕ
| 1 := 3
| 2 := 9
| 3 := 27
| n := a (n-1) + a (n-2) + a (n-3)

def b : ℕ → ℕ
| 1 := 0
| 2 := 0
| 3 := 1
| n := -- Add appropriate recurrence for b_n here

noncomputable def c_12 : ℕ := number_of_12_digit_numbers 12

#eval c_12 -- This will evaluate the expression to get the answer for c_12

theorem number_with_two_consecutive_ones :
  ∃ n : ℕ, n = c_12 ∧ n = 3^12 - a 12 - b 12 :=
sorry

end number_with_two_consecutive_ones_l336_336319


namespace probability_even_sum_balls_l336_336477

theorem probability_even_sum_balls :
  let balls := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
  let pairs := (balls × balls).filter (λ p, p.1 ≠ p.2)
  let favorable_pairs := pairs.filter (λ p, (p.1 + p.2) % 2 = 0)
  (favorable_pairs.card / pairs.card : ℚ) = 6 / 13 := by
  sorry

end probability_even_sum_balls_l336_336477


namespace Julia_remaining_money_l336_336376

def initial_money : ℝ := 200
def tablet_cost (x : ℝ) : ℝ := 0.40 * x
def remaining_after_tablet (x : ℝ) : ℝ := x - tablet_cost x
def phone_game_cost (x : ℝ) : ℝ := (1/3) * remaining_after_tablet x
def remaining_after_phone_game (x : ℝ) : ℝ := remaining_after_tablet x - phone_game_cost x
def in_game_purchase_cost (x : ℝ) : ℝ := 0.15 * remaining_after_phone_game x
def remaining_after_in_game_purchase (x : ℝ) : ℝ := remaining_after_phone_game x - in_game_purchase_cost x
def tablet_case_cost (x : ℝ) : ℝ := 0.20 * remaining_after_in_game_purchase x
def remaining_after_tablet_case (x : ℝ) : ℝ := remaining_after_in_game_purchase x - tablet_case_cost x
def phone_cover_cost (x : ℝ) : ℝ := 0.05 * remaining_after_tablet_case x
def remaining_after_phone_cover (x : ℝ) : ℝ := remaining_after_tablet_case x - phone_cover_cost x
def power_bank_cost (x : ℝ) : ℝ := 0.12 * remaining_after_phone_cover x
def remaining_after_power_bank (x : ℝ) : ℝ := remaining_after_phone_cover x - power_bank_cost x

theorem Julia_remaining_money : remaining_after_power_bank initial_money = 45.4784 := by
  /- proof goes here -/
  sorry

end Julia_remaining_money_l336_336376


namespace time_to_put_50_toys_is_36_75_minutes_l336_336419

-- Define the conditions
def add_toys_per_cycle := 4
def remove_toys_per_cycle := 3
def net_toys_per_cycle := add_toys_per_cycle - remove_toys_per_cycle
def seconds_per_cycle := 45
def total_toys_needed := 50

-- We need to prove that the required time in minutes to put all 50 toys into the box is 36.75
theorem time_to_put_50_toys_is_36_75_minutes : 
  let cycles := (total_toys_needed - 2) / net_toys_per_cycle + 1 in
  let total_seconds := cycles * seconds_per_cycle in
  total_seconds / 60 = 36.75 :=
by
  -- Placeholder proof
  sorry

end time_to_put_50_toys_is_36_75_minutes_l336_336419


namespace eval_oplus_otimes_l336_336272

-- Define the operations ⊕ and ⊗
def my_oplus (a b : ℕ) := a + b + 1
def my_otimes (a b : ℕ) := a * b - 1

-- Statement of the proof problem
theorem eval_oplus_otimes : my_oplus (my_oplus 5 7) (my_otimes 2 4) = 21 :=
by
  sorry

end eval_oplus_otimes_l336_336272


namespace transformation_of_T_l336_336413

-- Define the transformation vector space and its properties
variables {V : Type*} [add_comm_group V] [module ℝ V]
variables {T : V → V}
variables (v w : V) (a b : ℝ)

-- Assumptions
def linearity (T : V → V) :=
  ∀ (a b : ℝ) (v w : V), T (a • v + b • w) = a • (T v) + b • (T w)

def preserves_cross_product {T : V → V} [add_comm_group V] [module ℝ V] :=
  ∀ (v w : V), T (v × w) = (T v) × (T w)

def T_v1 := T ⟨5, 10, 0⟩ = ⟨2, -2, 10⟩
def T_v2 := T ⟨10, 0, 5⟩ = ⟨8, 10, -2⟩

-- Prove the required transformation
theorem transformation_of_T : 
  T ⟨8, 10, 5⟩ = ⟨7.6, 0.16, 9.36⟩ :=
by {
  assumption : linearity T,
  assumption $ preserves_cross_product T,
  assumption T_v1,
  assumption T_v2,
  sorry
}

end transformation_of_T_l336_336413


namespace find_ordered_pair_l336_336242

theorem find_ordered_pair : ∃ x y : ℤ, 3 * x - 7 * y = 2 ∧ 4 * y - x = 6 ∧ x = 10 ∧ y = 4 :=
by
  use 10, 4
  split
  {
    rw [int.mul_sub, int.mul_eq_coe_nat],
    norm_num,
  }
  split
  {
    norm_num,
  }
  split
  {
    norm_num,
  }
  {
    norm_num,
  }

end find_ordered_pair_l336_336242


namespace terms_before_5_l336_336693

def arithmetic_sequence (a₁ d n : ℤ) : ℤ :=
  a₁ + d * (n - 1)

theorem terms_before_5 : 
  ∀ (a₁ d : ℤ), a₁ = 105 → d = -5 → 
  ∃ n : ℤ, arithmetic_sequence a₁ d n = 5 ∧ n - 1 = 20 :=
by
  intros a₁ d h₁ h₂
  use 21
  split
  { rw [arithmetic_sequence, h₁, h₂]
    norm_num },
  { norm_num }

end terms_before_5_l336_336693


namespace incorrect_statements_l336_336888

theorem incorrect_statements (X : ℝ → Prop) (Y : ℝ → Prop):
  (∀ (r : ℝ), abs r < 1 → X r = false) →
  (∀ (X : ℝ → measure_theory.Measure ℝ), X = measure_theory.Measure.gaussian 30 100 → 
         measure_theory.expectation X ≠ 10) →
  (∀ (X : ℝ → measure_theory.Measure ℝ), X = measure_theory.Measure.gaussian 4 1 → 
         measure_theory.Measure.prob (λ x, x ≥ 5) X = 0.1587 →  
         measure_theory.Measure.prob (λ x, 3 < x ∧ x < 5) X = 0.6826) →
  (∀ (x : ℕ → ℝ), variance x = 3 → (std_dev (λ i, 4 * x i - 1)) ≠ 12) →
  ¬ (X ∧ Y) :=
by 
  intros A B C D
  sorry

end incorrect_statements_l336_336888


namespace find_circumcircle_coeffs_l336_336730

theorem find_circumcircle_coeffs :
  ∃ α β : ℂ, 
    α = -1 + complex.I ∧ β = -3 ∧
    ∀ z : ℂ, 
      let x : ℝ := z.re,
          y : ℝ := z.im in
      (x^2 + y^2 + α * z + complex.conj α * complex.conj z + β = 0) ↔
      (x - 3)^2 + (y - 1)^2 = 5 :=
begin
  sorry
end

end find_circumcircle_coeffs_l336_336730


namespace third_consecutive_even_number_l336_336850

theorem third_consecutive_even_number (n : ℕ) (h : n % 2 = 0) (sum_eq : n + (n + 2) + (n + 4) = 246) : (n + 4) = 84 :=
by
  -- This statement sets up the conditions and the goal of the proof.
  sorry

end third_consecutive_even_number_l336_336850


namespace length_of_longest_diagonal_l336_336181

theorem length_of_longest_diagonal (area : ℝ) (ratio : ℝ) (d1 d2 : ℝ) 
  (h_area : area = 200) (h_ratio : d1 / d2 = 4 / 3) 
  (h_formula : d1 * d2 = 2 * area) : 
  ∃ d1, d1 = (40 * real.sqrt 3) / 3 := 
sorry

end length_of_longest_diagonal_l336_336181


namespace simplify_sqrt_sum_l336_336808

theorem simplify_sqrt_sum : sqrt (8 + 6 * sqrt 3) + sqrt (8 - 6 * sqrt 3) = 2 * sqrt 6 := 
sorry

end simplify_sqrt_sum_l336_336808


namespace polynomial_divisibility_l336_336625

theorem polynomial_divisibility (a b c : ℝ) :
  (∀ x : ℝ, (x ^ 4 + a * x ^ 2 + b * x + c) = (x - 1) ^ 3 * (x + 1) →
  a = 0 ∧ b = 2 ∧ c = -1) :=
by
  intros x h
  sorry

end polynomial_divisibility_l336_336625


namespace min_sum_abc_l336_336846

theorem min_sum_abc (a b c : ℕ) (h1 : a * b * c = 3960) : a + b + c ≥ 150 :=
sorry

end min_sum_abc_l336_336846


namespace solve_for_y_l336_336696

theorem solve_for_y (x y : ℝ) (h1 : 2 * x - y = 10) (h2 : x + 3 * y = 2) : y = -6 / 7 := 
by
  sorry

end solve_for_y_l336_336696


namespace question1_question2_l336_336027

-- Definitions:
def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a - 1) * x + (a^2 - 5) = 0}

-- Question 1 Statement:
theorem question1 (a : ℝ) (h : A ∩ B a = {2}) : a = -5 ∨ a = 1 := by
  sorry

-- Question 2 Statement:
theorem question2 (a : ℝ) (h : A ∪ B a = A) : a > 3 := by
  sorry

end question1_question2_l336_336027


namespace sum_a1_a3_a5_l336_336853

noncomputable def a : ℕ → ℕ
| 1     := 16
| (n+1) := 3 * a n / 2

theorem sum_a1_a3_a5 : a 1 + a 3 + a 5 = 133 := by
  sorry

end sum_a1_a3_a5_l336_336853


namespace circleC_polar_eq_l336_336724

-- Define the point P in polar coordinates
def pointP_polar : ℝ × ℝ := (real.sqrt 2, π / 4)

-- Define the intersection line equation in polar coordinates
def line_equation (ρ θ : ℝ) : Prop := ρ * real.sin (θ - π / 3) = -real.sqrt 3 / 2

-- Define the circle C equation in polar coordinates
def circleC_equation (ρ θ : ℝ) : Prop := ρ = 2 * real.cos θ

-- Hypotheize that the point lies on the circle and the defined conditions
theorem circleC_polar_eq :
  (∃ ρ θ, (ρ, θ) = pointP_polar ∧ circleC_equation ρ θ) ∧
  (∃ ρ θ, line_equation ρ θ ∧ θ = 0) →
  ∀ ρ θ, circleC_equation ρ θ :=
sorry

end circleC_polar_eq_l336_336724


namespace square_side_length_l336_336325

theorem square_side_length (length_rect width_rect : ℕ) (h_length : length_rect = 400) (h_width : width_rect = 300)
  (h_perimeter : 4 * side_length = 2 * (2 * (length_rect + width_rect))) : side_length = 700 := by
  -- Proof goes here
  sorry

end square_side_length_l336_336325


namespace no_carrying_pairs_count_l336_336619

theorem no_carrying_pairs_count : 
  let valid_digits_count (d : Nat) := if d < 9 then 1 else 0,
      valid_hundred_digits := ({5, 6, 7, 8}: Finset Nat),
      total_valid_combinations := (valid_hundred_digits.card * (0..8).sum valid_digits_count * (0..8).sum valid_digits_count) in
  total_valid_combinations = 324 :=
by
  sorry

end no_carrying_pairs_count_l336_336619


namespace worm_length_1_covered_by_semicircle_l336_336732

noncomputable def worm_covered_by_semicircle (γ : ℝ → ℝ×ℝ) (length_γ : ℝ) (diameter : ℝ) : Prop :=
  ∀ t₁ t₂ : ℝ, continuous γ → (|t₁ - t₂| = 1) → 
  (∃ c : ℝ×ℝ, ∀ t : ℝ, |γ t - c| ≤ diameter / 2)

theorem worm_length_1_covered_by_semicircle :
  ∀ γ : ℝ → ℝ×ℝ, (continuous γ) → (∃ s t : ℝ, γ s ≠ γ t) → 
  (∃ d : ℝ, worm_covered_by_semicircle γ 1 d) :=
by
  sorry

end worm_length_1_covered_by_semicircle_l336_336732


namespace unique_solution_l336_336023

theorem unique_solution (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : sqrt a + sqrt b + sqrt c = sqrt π / 2) :
  ∃! (x y z : ℝ), sqrt (y - a) + sqrt (z - a) = 1 ∧
                  sqrt (z - b) + sqrt (x - b) = 1 ∧
                  sqrt (x - c) + sqrt (y - c) = 1 :=
by sorry

end unique_solution_l336_336023


namespace smallest_difference_is_482_l336_336116

-- Define the sets and functions that describe the problem conditions.
def digits : List ℕ := [1, 3, 5, 7, 8]

def three_digit_numbers : List ℕ :=
  digits.permutations.filter (λ l, l.length = 3).map (λ l, l[0] * 100 + l[1] * 10 + l[2])

def four_digit_numbers : List ℕ :=
  digits.permutations.filter (λ l, l.length = 4).map (λ l, l[0] * 1000 + l[1] * 100 + l[2] * 10 + l[3])

noncomputable def smallest_difference : ℕ :=
  (four_digit_numbers.bind (λ b, three_digit_numbers.map (λ a, b - a))).min

theorem smallest_difference_is_482 : smallest_difference = 482 := by sorry

end smallest_difference_is_482_l336_336116


namespace quadrilateral_perimeter_l336_336541

theorem quadrilateral_perimeter :
  let triangle_sides := (10 : ℝ, 6 : ℝ, 8 : ℝ)
  let (lt, mt, st) := (max triangle_sides.1 triangle_sides.2, max (min triangle_sides.1 triangle_sides.2) triangle_sides.3, min (max triangle_sides.1 triangle_sides.2) triangle_sides.3)
  let rectangle_length := 2 * lt
  let rectangle_width := mt
  let shared_side := lt
  let perimeter_triangle := triangle_sides.1 + triangle_sides.2 + triangle_sides.3
  let perimeter_rectangle := 2 * (rectangle_length + rectangle_width)
  let perimeter_quadrilateral := perimeter_triangle + perimeter_rectangle - 2 * shared_side
  perimeter_quadrilateral = 60 :=
by
  sorry

end quadrilateral_perimeter_l336_336541


namespace tangent_line_at_2_eq_neg_x_plus_4_l336_336659

noncomputable def f (x : ℝ) : ℝ := 2 * f (4 - x) - 2 * x^2 + 5 * x

theorem tangent_line_at_2_eq_neg_x_plus_4 : 
  ∃ k b, (∀ x, f x = k * x + b) ∧ k = -1 ∧ b = 4 := 
sorry

end tangent_line_at_2_eq_neg_x_plus_4_l336_336659


namespace pair_points_in_circle_l336_336366

theorem pair_points_in_circle
    (P : set (ℝ × ℝ))
    (hP_card : P.card = 100)
    (hP_inside_circle : ∀ {p : ℝ × ℝ}, p ∈ P → p.1^2 + p.2^2 < r^2)
    (hP_no_three_collinear : ∀ {p₁ p₂ p₃ : ℝ × ℝ}, p₁ ∈ P → p₂ ∈ P → p₃ ∈ P → 
        (p₁.1 * p₂.2 + p₂.1 * p₃.2 + p₃.1 * p₁.2 ≠ p₁.2 * p₂.1 + p₂.2 * p₃.1 + p₃.2 * p₁.1)) :
  ∃ (pairing : ∃ (pairs : list (ℝ × ℝ) × list (ℝ × ℝ)),
    (pairs.1.length = 50 ∧ pairs.2.length = 50 ∧ ∀ i, i < 50 → (pairs.1.nth i ∈ P ∧ pairs.2.nth i ∈ P ∧
    (∀ j, j < 50 → ∃ k, k < 50 ∧ ∃ intersection, 
    intersection ∈ circle ∧ intersection = line_intersection (pairs.1.nth i) (pairs.2.nth i) 
    (pairs.1.nth j) (pairs.2.nth j))))) :=
sorry

end pair_points_in_circle_l336_336366


namespace squareable_numbers_l336_336872

def is_squareable (n : ℕ) : Prop :=
  ∃ (perm : ℕ → ℕ), (∀ i, 1 ≤ perm i ∧ perm i ≤ n) ∧ (∀ i, ∃ k, perm i + i = k * k)

theorem squareable_numbers : is_squareable 9 ∧ is_squareable 15 ∧ ¬ is_squareable 7 ∧ ¬ is_squareable 11 :=
by sorry

end squareable_numbers_l336_336872


namespace exists_c_bound_l336_336750

open Nat

-- Definitions:
variable (N : ℕ) (x : ℕ → ℝ)

-- Conditions:
axiom hN_pos : N > 0
axiom hx_nonneg : ∀ n, 0 ≤ x n
axiom hx_equation : ∀ n > N, (x n)^2 = ∑ i in range (n-1), sqrt((x i) * (x (n - i)))

-- Theorem statement:
theorem exists_c_bound : ∃ c > 0, ∀ n, x n ≤ n / 2 + c := by
  sorry

end exists_c_bound_l336_336750


namespace count_congruent_to_6_mod_13_under_2000_l336_336321

theorem count_congruent_to_6_mod_13_under_2000 : 
  { n : ℕ | n < 2000 ∧ n % 13 = 6 }.finite.to_finset.card = 154 := 
by
  sorry

end count_congruent_to_6_mod_13_under_2000_l336_336321


namespace range_of_a_l336_336712

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 3| + |x - 7| ≥ a^2 - 3*a) → a ∈ set.Icc (-2 : ℝ) 5 :=
by
  intro h
  sorry

end range_of_a_l336_336712


namespace probability_both_courses_reviewed_l336_336198

theorem probability_both_courses_reviewed (students_total students_german students_italian : ℕ) 
(h_total : students_total = 30) (h_german : students_german = 22) (h_italian : students_italian = 25) :
  let students_both := students_german + students_italian - students_total in
  let probability_both := 1 - (Nat.choose (students_german - students_both) 2 + Nat.choose (students_italian - students_both) 2) / Nat.choose students_total 2 in
  probability_both = (397 / 435 : ℚ) := 
by
  sorry

end probability_both_courses_reviewed_l336_336198


namespace number_of_points_l336_336700

theorem number_of_points (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m + n ≤ 8) :
  (∑ m in finset.range 8, ∑ n in finset.Ico 1 (8 - m)) = 28 :=
sorry

end number_of_points_l336_336700


namespace max_digit_sum_less_than_1728_l336_336122

noncomputable def greatest_digit_sum_base_eight (n : ℕ) : ℕ :=
  let digits := (nat.digits 8 n) in digits.sum

theorem max_digit_sum_less_than_1728 : 
  ∀ n, n < 1728 → greatest_digit_sum_base_eight n ≤ 23 ∧ (∃ m, m < 1728 ∧ greatest_digit_sum_base_eight m = 23) := 
by
  sorry

end max_digit_sum_less_than_1728_l336_336122


namespace find_locus_of_Q_l336_336668

open Real

def point := ℝ × ℝ

def locus_of_Q (Q : point) : Prop :=
  ∃ P: point, P = (0, 1) ∧
  let midpoint := (Q.1 / 2, (Q.2 + 1) / 2) in
  let c := (((2 : ℝ) * (Q.2 - 1)) - 1) * Q.1 * Q.1 + (2 : ℝ) * (Q.2 + 1) * (Q.2 - 1) * (Q.2 - 1) in
  c = 0 ∧ (-1 ≤ Q.2 ∧ Q.2 < 1 / 2)

theorem find_locus_of_Q (Q : point) : 
  (∃ P: point, P = (0, 1) ∧ 
  let midpoint := (Q.1 / 2, (Q.2 + 1) / 2) in
  let c := (((2 : ℝ) * (Q.2 - 1)) - 1) * Q.1 * Q.1 + (2 : ℝ) * (Q.2 + 1) * (Q.2 - 1) * (Q.2 - 1) in
  c = 0 ∧ (-1 ≤ Q.2 ∧ Q.2 < 1 / 2)) := sorry

end find_locus_of_Q_l336_336668


namespace first_fly_is_faster_and_returns_first_l336_336861

-- Define the conditions
variables {d v : ℝ} (hv_pos : 0 < v) (hd_pos : 0 < d)

def first_fly_time : ℝ := 2 * d / v

def second_fly_time : ℝ := (d / (2 * v)) + (2 * d / (v / 2))

def first_fly_avg_speed : ℝ := 2 * d / first_fly_time

def second_fly_avg_speed : ℝ := 2 * d / second_fly_time

-- State the theorem that needs to be proven
theorem first_fly_is_faster_and_returns_first
  (hv_pos : 0 < v) (hd_pos : 0 < d) :
  first_fly_time < second_fly_time ∧ first_fly_avg_speed > second_fly_avg_speed :=
by
  sorry

end first_fly_is_faster_and_returns_first_l336_336861


namespace largest_independent_subsets_l336_336643

theorem largest_independent_subsets {n : ℕ} : 
    (∀ (a b : Finset (Fin n)), a ⊆ b → a = b) → 
    set.card (Finset.powerset (Fin n) \ {s ∈ Finset (Fin n) | ∃ t, t ∈ Finset.powerset (Fin n) ∧ t ⊂ s}) = nat.choose n (n/2) :=
by
  sorry

end largest_independent_subsets_l336_336643


namespace tan_A_tan_C_eq_3_l336_336739

variable {A B C : ℝ}

theorem tan_A_tan_C_eq_3 (h : cos A ^ 2 + cos B ^ 2 + cos C ^ 2 = sin B ^ 2) : tan A * tan C = 3 := 
by
  sorry

end tan_A_tan_C_eq_3_l336_336739


namespace largest_prime_factor_of_57_is_largest_among_85_57_119_143_169_l336_336503

theorem largest_prime_factor_of_57_is_largest_among_85_57_119_143_169 :
  ∀ (p : ℕ → ℕ),
    p 85 = 17 →
    p 57 = 19 →
    p 119 = 17 →
    p 143 = 13 →
    p 169 = 13 →
    (∀ n∈{85, 57, 119, 143, 169}, p n ≤ 19) →
    p 57 = 19 := by
sorry

end largest_prime_factor_of_57_is_largest_among_85_57_119_143_169_l336_336503


namespace find_x2_minus_y2_l336_336412

theorem find_x2_minus_y2 : 
  let x := 10^5 - 10^(-5)
  let y := 10^5 + 10^(-5)
  in x^2 - y^2 = -4 :=
by 
  let x := 10^5 - 10^(-5)
  let y := 10^5 + 10^(-5)
  sorry

end find_x2_minus_y2_l336_336412


namespace wicks_count_l336_336553

theorem wicks_count (foot_to_inches : nat := 12)
    (spool_length : nat := 25)
    (len_wick_6 : nat := 6)
    (len_wick_9 : nat := 9)
    (len_wick_12 : nat := 12)
    (lcm_6_9_12 : nat := 36)
    (total_inches := spool_length * foot_to_inches) :
    total_inches / lcm_6_9_12 * 3 = 24 :=
by
  sorry

end wicks_count_l336_336553


namespace jacob_twice_as_old_l336_336370

theorem jacob_twice_as_old (x : ℕ) : 18 + x = 2 * (9 + x) → x = 0 := by
  intro h
  linarith

end jacob_twice_as_old_l336_336370


namespace simplify_expression_l336_336989

theorem simplify_expression (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -3) :
  (3 * x^2 + 2 * x) / ((x - 1) * (x + 3)) - (5 * x + 3) / ((x - 1) * (x + 3))
  = 3 * (x^2 - x - 1) / ((x - 1) * (x + 3)) :=
by
  sorry

end simplify_expression_l336_336989


namespace car_owners_without_motorcycles_l336_336722

theorem car_owners_without_motorcycles
  (total_adults : ℕ)
  (car_owners : ℕ)
  (motorcycle_owners : ℕ)
  (all_owners : total_adults = 400)
  (john_owns_cars : car_owners = 370)
  (john_owns_motorcycles : motorcycle_owners = 50)
  (all_adult_owners : total_adults = car_owners + motorcycle_owners - (car_owners - motorcycle_owners)) : 
  (car_owners - (car_owners + motorcycle_owners - total_adults) = 350) :=
by {
  sorry
}

end car_owners_without_motorcycles_l336_336722


namespace exists_digit_sum_divisible_by_11_l336_336044

noncomputable def sum_of_digits (n : ℕ) : ℕ := 
  if n < 10 then n else 
  let rec digits_sum (n : ℕ) : ℕ :=
    if n == 0 then 0 
    else n % 10 + digits_sum (n / 10)
  digits_sum n

theorem exists_digit_sum_divisible_by_11 (N : ℕ) : 
  ∃ k, k ∈ finset.range 39 ∧ sum_of_digits (N + k) % 11 = 0 := 
sorry

end exists_digit_sum_divisible_by_11_l336_336044


namespace alternating_sum_10000_l336_336135

def alternating_sum (n : ℕ) : ℤ := ∑ k in (Finset.range (n + 1)).filter (λ k, k > 0), (if k % 2 = 1 then 1 else -1) * k

theorem alternating_sum_10000 : alternating_sum 10000 = -5000 := by
  sorry

end alternating_sum_10000_l336_336135


namespace valid_k_range_l336_336304

noncomputable def fx (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + k * x + k + 3

theorem valid_k_range:
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → fx k x ≥ 0) ↔ (k ≥ -3 / 13) :=
by
  sorry

end valid_k_range_l336_336304


namespace shift_graph_of_sine_to_cosine_l336_336859

theorem shift_graph_of_sine_to_cosine :
  ∀ x : ℝ, 2 * sin (2 * x) = 2 * cos (2 * (x + π / 6) - π / 6) :=
by
  intros x
  rw [cos_sub, add_sub_cancel']
  rfl


end shift_graph_of_sine_to_cosine_l336_336859


namespace exercise_solution_l336_336396

noncomputable def n_s_product : ℝ :=
  let S := {x : ℝ // 0 < x};
  ∃ f : S → ℝ, 
    (∀ (x y : S), f x * f y = f (⟨x * y, mul_pos x.prop y.prop⟩) + 1001 * (x⁻¹ + y⁻¹ + 1000)) ∧
    let fx2 := f (⟨2, by norm_num⟩);
    let n := 1; -- Derive n from the mathematical logic
    let s := (1 / 2) + 1001; -- Derive s from the mathematical logic
    n * s = 2003 / 2

theorem exercise_solution : n_s_product = 2003 / 2 := 
by
  sorry

end exercise_solution_l336_336396


namespace second_tap_empties_in_11_hours_l336_336169

-- Definitions and given conditions
def first_tap_rate : ℝ := 1 / 4
def combined_rate : ℝ := 7 / 44

-- Define the property of the second tap's time to empty the cistern
def second_tap_time := ∀ (T : ℝ), 1 / T = first_tap_rate - combined_rate → T = 11

-- Statement to prove
theorem second_tap_empties_in_11_hours : second_tap_time :=
by {
  sorry
}

end second_tap_empties_in_11_hours_l336_336169


namespace range_of_x_l336_336676

def f (x a : ℝ) : ℝ := x^3 + 3 * a * x - 1

def g (x a : ℝ) : ℝ := 3 * x^2 - a * x + 3 * a - 5

def condition (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 1

theorem range_of_x (x a : ℝ) (h : condition a) : g x a < 0 → -2/3 < x ∧ x < 1 := 
sorry

end range_of_x_l336_336676


namespace farmer_rectangle_partition_l336_336534

theorem farmer_rectangle_partition (m : ℝ) :
  (3 * m + 8) * (m - 3) = 70 ↔ m = (1 + Real.sqrt 1129) / 6 := 
begin
  sorry,
end

end farmer_rectangle_partition_l336_336534


namespace smallest_n_l336_336493

theorem smallest_n (n : ℕ) (h : 23 * n ≡ 789 [MOD 11]) : n = 9 :=
sorry

end smallest_n_l336_336493


namespace valentina_burger_length_l336_336488

-- Definitions and conditions
def share : ℕ := 6
def total_length (share : ℕ) : ℕ := 2 * share

-- Proof statement
theorem valentina_burger_length : total_length share = 12 := by
  sorry

end valentina_burger_length_l336_336488


namespace tan_alpha_complex_expr_sin_expr_l336_336275

variable (α : Real)
variable (h1 : 0 < α ∧ α < π / 2)
variable (h2 : sin α = 4 / 5)

theorem tan_alpha : tan α = 4 / 3 :=
sorry

theorem complex_expr (h3 : tan α = 4 / 3) :
  (sin (α + π) - 2 * cos (π / 2 + α)) / (-sin (-α) + cos (π + α)) = 4 :=
sorry

theorem sin_expr (h3 : tan α = 4 / 3) :
  sin (2 * α + π / 4) = 17 * sqrt 2 / 50 :=
sorry

end tan_alpha_complex_expr_sin_expr_l336_336275


namespace integral_sin4_cos4_l336_336950

theorem integral_sin4_cos4 (a b : ℝ) :
  ∫ (x : ℝ) in 0..2 * Real.pi, (Real.sin (3 * x))^4 * (Real.cos (3 * x))^4 = 3 * Real.pi / 64 :=
by
  sorry

end integral_sin4_cos4_l336_336950


namespace calc_2002_sq_minus_2001_mul_2003_l336_336563

theorem calc_2002_sq_minus_2001_mul_2003 : 2002 ^ 2 - 2001 * 2003 = 1 := 
by
  sorry

end calc_2002_sq_minus_2001_mul_2003_l336_336563


namespace g_bounded_l336_336752

noncomputable def problem (f f' f'' : ℝ → ℝ) (g : ℝ → ℝ) :=
  (∀ x ≥ 0, f'' x = 1 / (x^2 + (f' x)^2 + 1)) ∧
  f 0 = 0 ∧ f' 0 = 0 ∧
  (∀ x ≥ 0, g x = f x / x) ∧ g 0 = 0

theorem g_bounded {f f' f'' g : ℝ → ℝ} (h : problem f f' f'' g) :
  ∀ x ≥ 0, g x ≤ Real.pi / 2 :=
sorry

end g_bounded_l336_336752


namespace find_circumcircle_coeffs_l336_336729

theorem find_circumcircle_coeffs :
  ∃ α β : ℂ, 
    α = -1 + complex.I ∧ β = -3 ∧
    ∀ z : ℂ, 
      let x : ℝ := z.re,
          y : ℝ := z.im in
      (x^2 + y^2 + α * z + complex.conj α * complex.conj z + β = 0) ↔
      (x - 3)^2 + (y - 1)^2 = 5 :=
begin
  sorry
end

end find_circumcircle_coeffs_l336_336729


namespace equal_areas_implies_equal_segments_l336_336343

theorem equal_areas_implies_equal_segments 
(P Q R S : Point) 
(h_triangle : Triangle P Q R) 
(h_S_on_QR : S ∈ (segment Q R)) 
(h_equal_areas : area (Triangle P Q S) = area (Triangle P R S)) 
: (distance Q S) = (distance S R) := 
begin
  sorry,  -- The proof goes here.
end

end equal_areas_implies_equal_segments_l336_336343


namespace binomial_12_6_eq_924_l336_336961

theorem binomial_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binomial_12_6_eq_924_l336_336961


namespace problem_p_equals_1_plus_q_plus_q_squared_l336_336154

open Nat

theorem problem_p_equals_1_plus_q_plus_q_squared (P Q : ℕ) (prime_P : Prime P) (prime_Q : Prime Q) 
  (div_P_Q3_minus_1 : P ∣ Q^3 - 1) (div_P_minus_1_Q : (P - 1) ∣ Q) : P = 1 + Q + Q^2 := by
  sorry

end problem_p_equals_1_plus_q_plus_q_squared_l336_336154


namespace range_of_a_l336_336630

-- Define conditions
def setA : Set ℝ := {x | x^2 - x ≤ 0}
def setB (a : ℝ) : Set ℝ := {x | 2^(1 - x) + a ≤ 0}

-- Problem statement in Lean 4
theorem range_of_a (a : ℝ) (h : setA ⊆ setB a) : a ≤ -2 :=
by
  sorry

end range_of_a_l336_336630


namespace binom_12_6_eq_924_l336_336967

theorem binom_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binom_12_6_eq_924_l336_336967


namespace bridge_length_increase_l336_336525

open Real

def elevation_change : ℝ := 800
def original_gradient : ℝ := 0.02
def new_gradient : ℝ := 0.015

theorem bridge_length_increase :
  let original_length := elevation_change / original_gradient
  let new_length := elevation_change / new_gradient
  new_length - original_length = 13333 := by
  sorry

end bridge_length_increase_l336_336525


namespace problem_statement_l336_336273

open EuclideanGeometry

-- Definitions and given conditions
variables {A B C D E F : Point}
variable (ABC : Triangle A B C)

def Equilateral (a b c : Point) : Prop :=
∀ x y z, Triangle a b c → Length a b = Length b c ∧ Length b c = Length c a

-- Conditions
noncomputable def TriangleEquilateralABC : Prop := 
  ∃ D E, Equilateral A B D ∧ Equilateral A C E ∧ Intersect DC BE F

-- Problem reformulation
theorem problem_statement (ABC : Triangle A B C) (TriangleEquilateralABC: Prop):
  DC = BE ∧ ∠AFB = 120° :=
by
  sorry

end problem_statement_l336_336273


namespace product_of_three_consecutive_integers_divisible_by_six_l336_336123

theorem product_of_three_consecutive_integers_divisible_by_six (n : ℕ) : 
  6 ∣ (n * (n + 1) * (n + 2)) :=
sorry

end product_of_three_consecutive_integers_divisible_by_six_l336_336123


namespace area_of_triangle_rational_l336_336771

theorem area_of_triangle_rational 
  (m n p q : ℚ) 
  (hmn : m > n) 
  (hpq : p > q) 
  (a : ℚ := m * n * (p^2 + q^2)) 
  (b : ℚ := p * q * (m^2 + n^2)) 
  (c : ℚ := (m * q + n * p) * (m * p - n * q)) : 
  ∃ (t : ℚ), t = sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)) :=
sorry

end area_of_triangle_rational_l336_336771


namespace sum_of_divisors_420_l336_336131

theorem sum_of_divisors_420 : ∑ i in (finset.range 421).filter (λ d, 420 % d = 0), i = 1344 :=
by
  sorry

end sum_of_divisors_420_l336_336131


namespace total_score_is_248_l336_336318

def geography_score : ℕ := 50
def math_score : ℕ := 70
def english_score : ℕ := 66

def history_score : ℕ := (geography_score + math_score + english_score) / 3

theorem total_score_is_248 : geography_score + math_score + english_score + history_score = 248 := by
  -- proofs go here
  sorry

end total_score_is_248_l336_336318


namespace rational_roots_count_l336_336926

theorem rational_roots_count (b_4 b_3 b_2 b_1 : ℤ) :
  polynomial.leading_coeff (16 * X^5 + C (b_4) * X^4 + C (b_3) * X^3 + C (b_2) * X^2 + C (b_1) * X + 24) = 16 →
  polynomial.nat_degree (16 * X^5 + C (b_4) * X^4 + C (b_3) * X^3 + C (b_2) * X^2 + C (b_1) * X + 24) = 5 →
  (∀ p q : ℤ, (p ∣ 24) ∧ (q ∣ 16) → p = 0 → 16 = 1) :=
sorry

end rational_roots_count_l336_336926


namespace domain_of_log_base_5_range_of_3_pow_neg_l336_336827

theorem domain_of_log_base_5 (x : ℝ) : (1 - x > 0) -> (x < 1) :=
sorry

theorem range_of_3_pow_neg (y : ℝ) : (∃ x : ℝ, y = 3 ^ (-x)) -> (y > 0) :=
sorry

end domain_of_log_base_5_range_of_3_pow_neg_l336_336827


namespace kekai_garage_sale_l336_336380

theorem kekai_garage_sale :
  let shirts := 5
  let shirt_price := 1
  let pants := 5
  let pant_price := 3
  let total_money := (shirts * shirt_price) + (pants * pant_price)
  let money_kept := total_money / 2
  money_kept = 10 :=
by
  sorry

end kekai_garage_sale_l336_336380


namespace range_cos_sq_sub_two_sin_l336_336462

open Real

theorem range_cos_sq_sub_two_sin : 
  set.range (λ x : ℝ, cos x ^ 2 - 2 * sin x) = set.Icc (-2) 2 := by 
  sorry

end range_cos_sq_sub_two_sin_l336_336462


namespace boys_more_than_girls_l336_336336

theorem boys_more_than_girls
  (x y a b : ℕ)
  (h1 : x > y)
  (h2 : x * a + y * b = x * b + y * a - 1) :
  x = y + 1 :=
sorry

end boys_more_than_girls_l336_336336


namespace triangle_area_formula_l336_336581

variables {a b c S : ℝ} -- Define real numbers representing side lengths and area
variables {α β γ : ℝ}  -- Define real numbers representing angles

-- Given conditions
def sine_rule (a b c : ℝ) (α β γ : ℝ) : Prop := 
  (a / sin α = b / sin β) ∧ (a / sin α = c / sin γ)

def angle_sum (α β γ : ℝ) : Prop :=
  α = β + γ

-- The statement to be proved
theorem triangle_area_formula :
  sine_rule a b c α β γ → angle_sum α β γ → 
  S = a^2 * sin β * sin γ / (2 * sin (β + γ)) :=
by
  sorry

end triangle_area_formula_l336_336581


namespace alice_notebook_savings_l336_336192

theorem alice_notebook_savings :
  let original_price_per_notebook := 3.00
  let discount_rate := 0.25
  let number_of_notebooks := 8
  let discount_per_notebook := original_price_per_notebook * discount_rate
  let discounted_price_per_notebook := original_price_per_notebook - discount_per_notebook
  let total_cost_without_discount := number_of_notebooks * original_price_per_notebook
  let total_cost_with_discount := number_of_notebooks * discounted_price_per_notebook
  let total_savings := total_cost_without_discount - total_cost_with_discount
  in total_savings = 6.00 := 
sorry

end alice_notebook_savings_l336_336192


namespace count_x_values_l336_336334

theorem count_x_values : 
  {x : ℕ | (1 ≤ x ∧ x < 100) ∧ ∃ n : ℕ, x = n^2 - 10}.finite.card = 7 :=
by sorry

end count_x_values_l336_336334


namespace pollen_allergy_expected_count_l336_336426

theorem pollen_allergy_expected_count : 
  ∀ (sample_size : ℕ) (pollen_allergy_ratio : ℚ), 
  pollen_allergy_ratio = 1/4 ∧ sample_size = 400 → sample_size * pollen_allergy_ratio = 100 :=
  by 
    intros
    sorry

end pollen_allergy_expected_count_l336_336426


namespace sequence_n_5_l336_336624

theorem sequence_n_5 (a : ℤ) (n : ℕ → ℤ) 
  (h1 : ∀ i > 1, n i = 2 * n (i - 1) + a)
  (h2 : n 2 = 5)
  (h3 : n 8 = 257) : n 5 = 33 :=
by
  sorry

end sequence_n_5_l336_336624


namespace max_h_without_reflection_max_h_with_reflection_l336_336988

theorem max_h_without_reflection (h : ℕ) :
  ∀ (n : ℕ) (A : fin (n+1) → fin 64 → ℕ), 
  (∀ i j : fin (n+1), i ≠ j → (∀ k : fin 16, A i k ≠ A j k)) → 
  h ≤ 15 := 
sorry

theorem max_h_with_reflection (h : ℕ) :
  ∀ (n : ℕ) (A : fin (n+1) → fin 64 → ℕ), 
  (∀ i j : fin (n+1), i ≠ j → (∀ k : fin 32, A i k ≠ A j k)) → 
  h ≤ 7 := 
sorry

end max_h_without_reflection_max_h_with_reflection_l336_336988


namespace nba_conferences_division_l336_336725

theorem nba_conferences_division (teams : ℕ) (games_per_team : ℕ) (E : ℕ) :
  teams = 30 ∧ games_per_team = 82 ∧
  (teams = E + (teams - E)) ∧
  (games_per_team / 2 * E) + (games_per_team / 2 * (teams - E))  ≠ teams * games_per_team / 2 :=
by
  sorry

end nba_conferences_division_l336_336725


namespace logarithmic_sum_of_tangent_sequence_l336_336231

theorem logarithmic_sum_of_tangent_sequence :
  ∑ k in finset.range 44, real.log10 (real.tan (2 * (k + 1) * real.pi / 180)) = 0 :=
begin
  sorry
end

end logarithmic_sum_of_tangent_sequence_l336_336231


namespace cut_square_form_three_squares_l336_336216

theorem cut_square_form_three_squares (a : ℝ) (h : a > 0) :
  ∃ (parts : list (ℝ × ℝ)), 
  (∀ (p : ℝ × ℝ), p ∈ parts -> p.1 >= 0 ∧ p.2 >= 0) ∧
  (parts.length = 4) ∧ 
  ((parts.map (λ (p : ℝ × ℝ), p.1 * p.2)).sum = a^2) ∧
  (∃ (squares : list (ℝ × ℝ)), 
  (∀ (s : ℝ × ℝ), s ∈ squares -> s.1 = s.2) ∧ 
  (squares.length = 3) ∧ 
  ((squares.map (λ (s : ℝ × ℝ), s.1 * s.2)).sum = a^2 / 2))
:= sorry

end cut_square_form_three_squares_l336_336216


namespace valid_ns_l336_336218

noncomputable def valid_n (n : ℕ) : Prop :=
∃ m : ℕ, m ≥ 10 ∧ (∀ (d : ℕ), d ∈ digits 10 m → d ≠ 0) ∧
(all_unique (digits 10 m)) ∧
(m % n = 0) ∧ 
(∀ (p : (List ℕ)), p ∈ permutations (digits 10 m) → (from_digits 10 p) % n = 0)

theorem valid_ns : {n : ℕ | valid_n n} = {1, 2, 3, 4, 9} := 
sorry

end valid_ns_l336_336218


namespace cheese_needed_for_sandwiches_l336_336047

theorem cheese_needed_for_sandwiches :
  (∀ (m_per_sandwich cheese_per_sandwich total_cheese_needed : ℝ),
    m_per_sandwich = 4 / 10 →
    cheese_per_sandwich = 1 / 2 * m_per_sandwich →
    total_cheese_needed = cheese_per_sandwich * 30 →
    total_cheese_needed = 6) :=
begin
  sorry
end

end cheese_needed_for_sandwiches_l336_336047


namespace mark_ate_in_first_four_days_l336_336779

-- Definitions based on conditions
def total_fruit : ℕ := 10
def fruit_kept : ℕ := 2
def fruit_brought_on_friday : ℕ := 3

-- Statement to be proved
theorem mark_ate_in_first_four_days : total_fruit - fruit_kept - fruit_brought_on_friday = 5 := 
by sorry

end mark_ate_in_first_four_days_l336_336779


namespace region_area_l336_336992

theorem region_area (x y : ℝ) : 
  (|2 * x - 16| + |3 * y + 9| ≤ 6) → ∃ A, A = 72 :=
sorry

end region_area_l336_336992


namespace triangle_area_l336_336239

def point : Type := (ℝ × ℝ)

def A : point := (2, -4)
def B : point := (0, 3)
def C : point := (5, -3)

def vector_sub (p1 p2 : point) : point :=
  (p1.1 - p2.1, p1.2 - p2.2)

def determinant (v w : point) : ℝ :=
  v.1 * w.2 - v.2 * w.1

def area_triangle (A B C : point) : ℝ :=
  0.5 * | determinant (vector_sub C A) (vector_sub C B) |

theorem triangle_area :
  area_triangle A B C = 23 / 2 := 
  sorry

end triangle_area_l336_336239


namespace collinear_probability_in_grid_l336_336568

-- Define the dimensions of the grid
def rows : ℕ := 4
def columns : ℕ := 5
def totalDots : ℕ := 20

-- Define the total number of ways to choose 4 dots from 20
def totalWaysToChoose4Dots : ℕ := Nat.choose totalDots 4

-- Define the number of sets of 4 collinear dots (horizontally, vertically, diagonally)
def collinearSets : ℕ := 17

-- Define the probability of four randomly chosen dots being collinear
def collinearProbability : ℚ := collinearSets / totalWaysToChoose4Dots

-- Main statement to be proved
theorem collinear_probability_in_grid :
  collinearProbability = 17 / 4845 := by
  sorry

end collinear_probability_in_grid_l336_336568


namespace students_exam_percentage_l336_336715

theorem students_exam_percentage 
  (total_students : ℕ) 
  (avg_assigned_day : ℚ) 
  (avg_makeup_day : ℚ)
  (overall_avg : ℚ) 
  (h_total : total_students = 100)
  (h_avg_assigned_day : avg_assigned_day = 0.60) 
  (h_avg_makeup_day : avg_makeup_day = 0.80) 
  (h_overall_avg : overall_avg = 0.66) : 
  ∃ x : ℚ, x = 70 / 100 :=
by
  sorry

end students_exam_percentage_l336_336715


namespace work_completion_time_l336_336903

noncomputable def work_done_by_woman_per_day : ℝ := 1 / 50
noncomputable def work_done_by_child_per_day : ℝ := 1 / 100
noncomputable def total_work_done_by_5_women_per_day : ℝ := 5 * work_done_by_woman_per_day
noncomputable def total_work_done_by_10_children_per_day : ℝ := 10 * work_done_by_child_per_day
noncomputable def combined_work_per_day : ℝ := total_work_done_by_5_women_per_day + total_work_done_by_10_children_per_day

theorem work_completion_time (h1 : 10 / 5 = 2) (h2 : 10 / 10 = 1) :
  1 / combined_work_per_day = 5 :=
by
  sorry

end work_completion_time_l336_336903


namespace min_value_proof_l336_336020

noncomputable def min_value (a b c d : ℝ) (h : (a + c) * (b + d) = a * c + b * d) : ℝ :=
  ∑ in! [a/b, b/c, c/d, d/a], id

theorem min_value_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : (a + c) * (b + d) = a * c + b * d) : min_value a b c d h = 8 :=
by sorry

end min_value_proof_l336_336020


namespace cubic_sum_identity_l336_336704

variables (x y z : ℝ)

theorem cubic_sum_identity (h1 : x + y + z = 10) (h2 : xy + xz + yz = 30) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 100 :=
sorry

end cubic_sum_identity_l336_336704


namespace inscribed_circle_center_intersection_rhombus_l336_336041

theorem inscribed_circle_center_intersection_rhombus
  (ABCD : Type) [quadrilateral ABCD]
  (O : Point)
  (h1 : is_incenter ABCD O)
  (h2 : is_intersection_of_diagonals ABCD O) :
  is_rhombus ABCD :=
by 
  sorry

end inscribed_circle_center_intersection_rhombus_l336_336041


namespace area_between_concentric_l336_336860

/- Define the key geometric points and lengths -/
def Point : Type := ℝ × ℝ

def concentricCircles {O : Point} (radius_outer radius_inner : ℝ) : Prop :=
  ∀ (x y : Point), (dist x O = radius_outer ∨ dist x O = radius_inner) →
                   (dist y O = radius_outer ∨ dist y O = radius_inner) →
                   (dist x O = radius_inner ∨ dist y O = radius_inner → dist x y ≠ 0)

/- Define the chord and its properties -/
def tangentChord (O G E F : Point) (r_inner : ℝ) (length_chord : ℝ) : Prop :=
  dist O G = r_inner ∧ dist E G = length_chord / 2 ∧ dist F G = length_chord / 2 ∧
  (∃ radius_inner : ℝ, 
    tangentLineAtCircle G (circle O radius_inner)) 

/- Main theorem that we need to prove -/
theorem area_between_concentric (O E F G : Point) (r_outer r_inner : ℝ) :
  concentricCircles O r_outer r_inner →
  tangentChord O G E F r_inner 20 →
  dist O E = 12 →
  dist E F = 20 →
  ∃ area_outer area_inner: ℝ, 
    area_outer = π * (r_outer ^ 2) ∧ area_inner = π * (r_inner ^ 2) ∧
    area_outer - area_inner = 100 * π :=
by
  sorry

end area_between_concentric_l336_336860


namespace find_n_prime_divisors_l336_336401

def d (N : ℕ) : ℕ := Nat.divisors N |>.length

theorem find_n_prime_divisors (N P : ℕ) (h : P = N / (d N)) (h_prime : Nat.Prime P) :
  N = 8 ∨ N = 9 ∨ N = 12 ∨ N = 18 ∨ N = 24 ∨ 
  (∃ p, Nat.Prime p ∧ p > 3 ∧ (N = 8 * p ∨ N = 12 * p ∨ N = 18 * p)) :=
sorry

end find_n_prime_divisors_l336_336401


namespace infinite_even_k_composite_l336_336804

theorem infinite_even_k_composite (k : ℕ) :
  (∃ t : ℕ, t > 0 ∧ k = 66 * t + 2) →
  (∀ p : ℕ, p.prime → ¬ (p^2 + k).prime) :=
by
  sorry

end infinite_even_k_composite_l336_336804


namespace max_min_value_of_integral_l336_336600

theorem max_min_value_of_integral :
  let f (a b : ℝ) := ∫ x in 0..π, (a * sin x + b * cos x)^3
  (a_max, b_max) = (1, 1)
  (a_min, b_min) = (-1, -1)
  (0 ≤ |a| ≤ 1) ∧ (0 ≤ |b| ≤ 1) →
  (f a_max b_max) = 10/3 ∧ (f a_min b_min) = -10/3 :=
by
  intro f a_max b_max a_min b_min h
  sorry

end max_min_value_of_integral_l336_336600


namespace no_carrying_pairs_correct_l336_336615

noncomputable def count_no_carrying_pairs : ℕ :=
  let pairs := (1500 : ℕ, 2500 : ℕ)
  (1550 : ℕ)  -- correct answer

theorem no_carrying_pairs_correct :
  ∃ count : ℕ, count = count_no_carrying_pairs :=
sorry

end no_carrying_pairs_correct_l336_336615


namespace hyperbola_equation_l336_336097

theorem hyperbola_equation (b c a : ℝ) (hb : b = 4) (hc : c = 6) (ha : a^2 = c^2 - b^2) : 
  ((y : ℝ)² / 20 - (x : ℝ)² / 16 = 1) := 
begin
  have : b ^ 2 = 16 := by rw [hb, sq],
  have : c ^ 2 = 36 := by rw [hc, sq],
  have : a ^ 2 = 20 := by rw [ha, this, this],
  sorry,
end

end hyperbola_equation_l336_336097


namespace weight_of_replaced_person_l336_336819

variable (average_weight_increase : ℝ)
variable (num_persons : ℝ)
variable (weight_new_person : ℝ)

theorem weight_of_replaced_person 
    (h1 : average_weight_increase = 2.5) 
    (h2 : num_persons = 10) 
    (h3 : weight_new_person = 90)
    : ∃ weight_replaced : ℝ, weight_replaced = 65 := 
by
  sorry

end weight_of_replaced_person_l336_336819


namespace pipe_A_fill_time_with_leak_l336_336801

-- Definitions from the conditions
def Pipe_A_rate : ℚ := 1 / 3
def Leak_rate : ℚ := 2 / 9
def Combined_rate : ℚ := Pipe_A_rate - Leak_rate

-- Theorem stating the time it takes for Pipe A to fill the tank with the leak present
theorem pipe_A_fill_time_with_leak : Combined_rate = 1 / 9 → 1 / Combined_rate = 9 :=
by
  intro h₁
  rw h₁
  norm_num
  done

end pipe_A_fill_time_with_leak_l336_336801


namespace common_ratio_of_geometric_sequence_l336_336468

theorem common_ratio_of_geometric_sequence (S : ℕ → ℝ) (a_1 a_2 : ℝ) (q : ℝ)
  (h1 : S 3 = a_1 * (1 + q + q^2))
  (h2 : 2 * S 3 = 2 * a_1 + a_2) : 
  q = -1/2 := 
sorry

end common_ratio_of_geometric_sequence_l336_336468


namespace length_EF_area_ABC_l336_336478

-- Definitions of given conditions
def r : ℝ := 3
def d : ℝ := 3

-- Prove length of segment EF and area of triangle ABC
theorem length_EF_area_ABC (r_pos : r > 2)
                           (A B C E F : ℝ) -- Define points 
                           (distance_AB : dist (A, 0) (B, 0) = d)
                           (distance_BC : dist (B, 0) (C, sqrt (9 - 1.5^2)) = d)
                           (distance_CA : dist (C, 0) (A, sqrt (9 - 1.5^2)) = d) :
  let EF := 3,
      area_ABC := sqrt (31.5) / 2
  in EF = 3 ∧ area_ABC = sqrt (31.5) / 2 := 
  sorry

end length_EF_area_ABC_l336_336478


namespace Tim_pencils_value_l336_336113

variable (Sarah_pencils : ℕ)
variable (Tyrah_pencils : ℕ)
variable (Tim_pencils : ℕ)

axiom Tyrah_condition : Tyrah_pencils = 6 * Sarah_pencils
axiom Tim_condition : Tim_pencils = 8 * Sarah_pencils
axiom Tyrah_pencils_value : Tyrah_pencils = 12

theorem Tim_pencils_value : Tim_pencils = 16 :=
by
  sorry

end Tim_pencils_value_l336_336113


namespace number_of_ways_to_adjust_items_l336_336742

theorem number_of_ways_to_adjust_items :
  let items_on_upper_shelf := 4
  let items_on_lower_shelf := 8
  let move_items := 2
  let total_ways := Nat.choose items_on_lower_shelf move_items
  total_ways = 840 :=
by
  sorry

end number_of_ways_to_adjust_items_l336_336742


namespace square_side_length_l336_336323

-- Problem conditions as Lean definitions
def length_rect : ℕ := 400
def width_rect : ℕ := 300
def perimeter_rect := 2 * length_rect + 2 * width_rect
def perimeter_square := 2 * perimeter_rect
def length_square := perimeter_square / 4

-- Proof statement
theorem square_side_length : length_square = 700 := 
by 
  -- (Any necessary tactics to complete the proof would go here)
  sorry

end square_side_length_l336_336323


namespace tan_eq_cos_has_three_solutions_l336_336690

def f (x : ℝ) : ℝ := tan (2 * x)
def g (x : ℝ) : ℝ := cos (2 * x)

theorem tan_eq_cos_has_three_solutions :
  (∃ (a b c : ℝ), a ∈ set.Icc 0 real.pi ∧ b ∈ set.Icc 0 real.pi ∧ c ∈ set.Icc 0 real.pi ∧
                  f a = g a ∧ f b = g b ∧ f c = g c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  ∀ x ∈ set.Icc 0 real.pi, f x = g x → (x = a ∨ x = b ∨ x = c) :=
by sorry

end tan_eq_cos_has_three_solutions_l336_336690


namespace maximum_value_fraction_l336_336019

theorem maximum_value_fraction (x : ℝ) (hx : 0 < x) :
  ∃ M, M = 0 ∧ ∀ y, (y = (x^2 + 1 - real.sqrt (x^4 + x^2 + 4))/ x) → y ≤ M :=
sorry

end maximum_value_fraction_l336_336019


namespace max_min_values_l336_336677

def func (x : ℝ) : ℝ := 1 - 2 * Real.cos (Real.pi / 2 * x)

theorem max_min_values :
  (∀ x : ℝ, func x ≤ 3) ∧ (∃ x : ℝ, func x = 3) ∧
  (∀ x : ℝ, -1 ≤ func x) ∧ (∃ x : ℝ, func x = -1) :=
by 
  -- Proof is skipped with sorry
  sorry

end max_min_values_l336_336677


namespace point_Q_in_second_quadrant_l336_336705

theorem point_Q_in_second_quadrant (a b : ℝ) (ha : a > 0) (hb : b < 0) : 
    (-a < 0 ∧ -b > 0) :=
begin
  split,
  { -- Proving -a < 0
    linarith, },
  { -- Proving -b > 0
    linarith, }
end

end point_Q_in_second_quadrant_l336_336705


namespace range_of_m_values_l336_336636

def prop_P (m : ℝ) : Prop := -3 ≤ m - 5 ∧ m - 5 ≤ 3

def prop_Q (m : ℝ) : Prop :=
  let f (x : ℝ) := 3 * x^2 + 2 * m * x + m + 4 / 3
  (let Δ := 4 * m^2 - 12 * (m + 4 / 3) in Δ > 0)

theorem range_of_m_values (m : ℝ) : prop_P m ∨ prop_Q m ↔ (m ≥ 2 ∨ m < -1) :=
by sorry

end range_of_m_values_l336_336636


namespace modulus_of_z_l336_336650

noncomputable def z (i : ℂ) : ℂ := 1 + i

theorem modulus_of_z (i : ℂ) (h : i^3 * z = 1 + i) : complex.abs z = √2 :=
by
  sorry

end modulus_of_z_l336_336650


namespace count_no_carry_pairs_l336_336620

-- Define the range of integers from 1500 to 2500
def range_integers : List ℕ := List.range' 1500 (2500 - 1500 + 1)

-- Define a function to check for no carry condition when adding two consecutive integers
def no_carry (n m : ℕ) : Prop :=
  let digits := List.zip (n.digits 10) (m.digits 10)
  ∀ (a b : ℕ) in digits, a + b < 10

-- Count pairs of consecutive integers that satisfy the no carry condition
def count_valid_pairs (lst : List ℕ) : ℕ :=
  (lst.zip (lst.tail)).count (λ (p : ℕ × ℕ), no_carry p.1 p.2)

-- The theorem to prove the total number of such valid pairs
theorem count_no_carry_pairs : count_valid_pairs range_integers = 1100 :=
by
  sorry

end count_no_carry_pairs_l336_336620


namespace min_p_value_l336_336669

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 then (3 + x) / (1 + x^2) else 3 / 5

def a_seq (n : ℕ) : ℝ := sorry -- Sequence a_n satisfying the given conditions

theorem min_p_value : 
  (∀ n, 1 ≤ n → n ≤ 2009 → 0 < a_seq n ∧ a_seq n ≤ 3) →
  (∀ n, 1 ≤ n → n ≤ 2009 → 0 < a_seq n ∧ a_seq n ≤ 3) →
  ∑ i in finset.range 2009, a_seq (i + 1) = 2009 / 3 →
  (∀ x : ℝ, p < x → ∑ i in finset.range 2009, f (a_seq (i + 1)) ≤ x - real.log (x - p)) →
  p ≥ 6026 :=
begin
  intros,
  sorry,
end

end min_p_value_l336_336669


namespace find_two_digit_number_l336_336857

theorem find_two_digit_number (n : ℕ) (h1 : n % 9 = 7) (h2 : n % 7 = 5) (h3 : n % 3 = 1) (h4 : 10 ≤ n) (h5 : n < 100) : n = 61 := 
by
  sorry

end find_two_digit_number_l336_336857


namespace middle_of_sorted_numbers_l336_336084

theorem middle_of_sorted_numbers :
  let nums := [1/4, 4/10, 41/100, 0.04, 0.404]
  let sorted_nums := List.sort (· ≤ ·) nums
  sorted_nums.get! 2 = 4/10 :=
by
  sorry

end middle_of_sorted_numbers_l336_336084


namespace equal_areas_of_triangles_l336_336354

-- Definitions and conditions for the geometric problem
variables {V : Type*} [inner_product_space ℝ V] -- Euclidean space.

-- Points in the space
variables {A B C T P M K : V}

-- Conditions: convex quadrilateral and given lengths
variables (H1 : function.injective (λ x : fin 4, [A, B, C, T].nth x))
variables (H2 : dist A B = dist B C)
variables (H3 : dist A T = dist T C)
variables (H4 : ∃ P : V, ∃ t : ℝ, t ∈ Icc 0 1 ∧ P = t • B + (1 - t) • T) -- P on diagonal BT
variables (H5 : collinear [P, M, C] ∧ collinear [P, M, T])
variables (H6 : collinear [P, K, C] ∧ collinear [P, K, B])

-- Question: Equate the areas of the two triangles
theorem equal_areas_of_triangles 
  (h₁ : H1)
  (h₂ : H2)
  (h₃ : H3)
  (h₄ : H4)
  (h₅ : H5)
  (h₆ : H6)
  : area (triangle.mk P T K) = area (triangle.mk P B M) :=
begin
  sorry
end

end equal_areas_of_triangles_l336_336354


namespace problem1_problem2_l336_336210

theorem problem1 :
  (2 / 3) * Real.sqrt 24 / (-Real.sqrt 3) * (1 / 3) * Real.sqrt 27 = - (4 / 3) * Real.sqrt 6 :=
sorry

theorem problem2 :
  Real.sqrt 3 * Real.sqrt 12 + (Real.sqrt 3 + 1)^2 = 10 + 2 * Real.sqrt 3 :=
sorry

end problem1_problem2_l336_336210


namespace find_sum_of_angles_l336_336656

theorem find_sum_of_angles (α β : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hβ1 : 0 < β ∧ β < π / 2)
  (hα2 : Real.sin α = (Real.sqrt 5) / 5) (hβ2 : Real.cos β = 3 * (Real.sqrt 10) / 10) :
  α + β = π / 4 :=
begin
  sorry
end

end find_sum_of_angles_l336_336656


namespace correctness_of_statements_l336_336830

theorem correctness_of_statements :
  (statement1 ∧ statement4 ∧ statement5) :=
by sorry

end correctness_of_statements_l336_336830


namespace monotone_sequence_l336_336008

theorem monotone_sequence (n : ℕ) (h : 0 < n) (S : set ℝ) (hS : S.card = (nat.choose (2 * n) n) + 1) :
  ∃ (a : ℕ → ℝ) (h_mono : ∀ {i j : ℕ}, i < j → a i ≤ a j ) (h_sub : ∀ i, a i ∈ S), 
    ∀ i ∈ finset.range(n + 1), |a (i+1) - a 0| ≥ 2 * |a i - a 0| :=
sorry

end monotone_sequence_l336_336008


namespace cubic_sum_divisible_by_9_l336_336115

theorem cubic_sum_divisible_by_9 (n : ℕ) (hn : n > 0) : 
  ∃ k, n^3 + (n+1)^3 + (n+2)^3 = 9*k := by
  sorry

end cubic_sum_divisible_by_9_l336_336115


namespace area_of_smaller_segment_is_correct_l336_336531

variables {R : ℝ} (hR_pos : R > 0)

def side_length_of_inscribed_square (R : ℝ) : ℝ :=
  R * Real.sqrt 2

def area_of_circle (R : ℝ) : ℝ :=
  Real.pi * R^2

def area_of_inscribed_square (R : ℝ) : ℝ :=
  2 * R^2

def area_of_smaller_segment (R : ℝ) : ℝ :=
  1/4 * (Real.pi * R^2 - 2 * R^2)

theorem area_of_smaller_segment_is_correct :
  ∀ R (hR_pos : R > 0),
  area_of_smaller_segment R = (R^2 * (Real.pi - 2) / 4) :=
by
  intros R hR_pos
  sorry

end area_of_smaller_segment_is_correct_l336_336531


namespace total_sugar_in_all_candy_l336_336030

-- definitions based on the conditions
def chocolateBars : ℕ := 14
def sugarPerChocolateBar : ℕ := 10
def lollipopSugar : ℕ := 37

-- proof statement
theorem total_sugar_in_all_candy :
  (chocolateBars * sugarPerChocolateBar + lollipopSugar) = 177 := 
by
  sorry

end total_sugar_in_all_candy_l336_336030


namespace distance_between_vertices_l336_336993

theorem distance_between_vertices :
  ∀ x y : ℝ, (sqrt (x^2 + y^2) + abs (y - 3) = 5) → (abs ((-1/16:ℝ) * (0:ℝ)^2 + 4 - (1/4:ℝ) * (0:ℝ)^2 + (-1)) = 5) :=
by 
  sorry

end distance_between_vertices_l336_336993


namespace point_P_illuminated_point_P_l336_336406

variables {A B : Type} 
variables (F1 F2 : ℝ) (m n a x y z b : ℝ)

-- Conditions
def luminous_ratio (F1 F2 m n : ℝ) := F1 / F2 = m / n
def distance_between_A_B (A B : ℝ) := A - B = a
def is_point_P_on_line (x a : ℝ) := a - x
def is_point_P'_outside_line (y z a b : ℝ) := y^2 = b^2 + x^2 ∧ z^2 = b^2 + (a - x)^2

theorem point_P_illuminated (F1 F2 m n a : ℝ) (h1 : luminous_ratio F1 F2 m n) :
  x = a * (m + real.sqrt (m * n)) / (m - n) ∨ x = a * (m - real.sqrt (m * n)) / (m - n) :=
by
  sorry

theorem point_P'_illuminated (F1 F2 m n a b : ℝ) (h1 : luminous_ratio F1 F2 m n) :
  x = (m * a + real.sqrt (m * a^2 * n) - (b^2 * (n - m)^2)) / (n - m) ∨ 
  x = (m * a - real.sqrt (m * a^2 * n) + (b^2 * (n - m)^2)) / (n - m) :=
by
  sorry

end point_P_illuminated_point_P_l336_336406


namespace twentieth_term_arith_seq_l336_336570

theorem twentieth_term_arith_seq (a1 d n : ℕ) (h₁ : a1 = 1) (h₂ : d = 5) (h₃ : n = 20) :
  a1 + (n - 1) * d = 96 := 
by
  rw [h₁, h₂, h₃]
  norm_num

end twentieth_term_arith_seq_l336_336570


namespace soup_weight_on_fourth_day_l336_336201

-- Definition of the weight function
def weight_on_day (initial_weight : ℝ) (n : ℕ) : ℝ :=
  initial_weight / (2 ^ n)

-- Theorem statement
theorem soup_weight_on_fourth_day (initial_weight : ℝ) (n : ℕ) (h_initial : initial_weight = 80) (h_n : n = 4) : 
  weight_on_day initial_weight n = 5 := 
by
  sorry

end soup_weight_on_fourth_day_l336_336201


namespace triangle_relation_l336_336149

variables {V : Type} [InnerProductSpace ℝ V]
variables (A B C : V)

theorem triangle_relation 
  (h : ⟪B - A, C - A⟫ + 2 * ⟪A - B, C - B⟫ = 3 * ⟪A - C, B - C⟫) :
  let a := dist B C
      b := dist C A
      c := dist A B
  in a^2 + 2 * b^2 = 3 * c^2 :=
sorry

end triangle_relation_l336_336149


namespace cousin_reading_time_l336_336792

theorem cousin_reading_time :
  ∀ (cousin_speed_factor : ℕ) (my_reading_time_hr : ℕ) (minute_per_hour : ℕ),
  cousin_speed_factor = 4 →
  my_reading_time_hr = 3 →
  minute_per_hour = 60 →
  let my_reading_time_min := my_reading_time_hr * minute_per_hour in
  let cousin_reading_time_min := my_reading_time_min / cousin_speed_factor in
  cousin_reading_time_min = 45 :=
by
  intro cousin_speed_factor my_reading_time_hr minute_per_hour h1 h2 h3
  let my_reading_time_min := my_reading_time_hr * minute_per_hour
  let cousin_reading_time_min := my_reading_time_min / cousin_speed_factor
  sorry

end cousin_reading_time_l336_336792


namespace zeros_and_extreme_points_l336_336856

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x ^ 2)

theorem zeros_and_extreme_points :
  (∃ x, f x = 0 ∧ x = 1) ∧ 
  (∃ x, f' x = 0 ∧ x = 2) := 
by
  sorry

end zeros_and_extreme_points_l336_336856


namespace arrangements_count_l336_336247

-- Defining the conditions
def person : Type := {A, B, C, D, E}
def position : Type := {posA, posB, posC, posD}
def arrangements (p : person → position) : Prop := (p A ≠ posA)

-- Statement that proves the total number of arrangements equals 42
theorem arrangements_count : ∃ (count : ℕ), count = 42 ∧ (∀ (p : person → position), arrangements p) -> sorry := sorry

end arrangements_count_l336_336247


namespace larger_number_l336_336471

theorem larger_number (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 4) : x = 17 :=
by
sorry

end larger_number_l336_336471


namespace simplify_expression_l336_336811

theorem simplify_expression :
  sqrt (8 + 6 * sqrt 3) + sqrt (8 - 6 * sqrt 3) = 2 * sqrt 6 :=
by sorry

end simplify_expression_l336_336811


namespace find_f1_l336_336774

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then 2 * x^2 - x else - sorry -- To represent arbitrary extension for positive x due to oddness

-- Definition of odd function
def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f(-x) = -f(x)

-- Conditions
axiom f_odd : odd_function f
axiom f_defn_neg : ∀ x ≤ 0, f(x) = 2 * x^2 - x

-- Proof statement
theorem find_f1 : f(1) = -3 :=
sorry

end find_f1_l336_336774


namespace max_sides_touched_l336_336913

noncomputable def hexagon := Type -- Placeholder for a regular hexagon type
noncomputable def circle := Type -- Placeholder for a circle type

def is_regular_hexagon (H : hexagon) : Prop := sorry
def is_contained_in (C : circle) (H : hexagon) : Prop := sorry
def touches_sides (C : circle) (H : hexagon) (n : ℕ) : Prop := sorry

theorem max_sides_touched (H : hexagon) (C : circle) 
    (H_hex : is_regular_hexagon H) 
    (H_cont : is_contained_in C H) 
    (H_not_all : ¬ touches_sides C H 6) : 
    ∃ n : ℕ, touches_sides C H n ∧ n ≤ 2 :=
sorry

end max_sides_touched_l336_336913


namespace Tim_has_16_pencils_l336_336110

variable (T_Sarah T_Tyrah T_Tim : Nat)

-- Conditions
def condition1 : Prop := T_Tyrah = 6 * T_Sarah
def condition2 : Prop := T_Tim = 8 * T_Sarah
def condition3 : Prop := T_Tyrah = 12

-- Theorem to prove
theorem Tim_has_16_pencils (h1 : condition1 T_Sarah T_Tyrah) (h2 : condition2 T_Sarah T_Tim) (h3 : condition3 T_Tyrah) : T_Tim = 16 :=
by
  sorry

end Tim_has_16_pencils_l336_336110


namespace kendra_shirts_needed_l336_336386

def shirts_needed_per_week (school_days after_school_club_days saturday_shirts sunday_church_shirt sunday_rest_of_day_shirt : ℕ) : ℕ :=
  school_days + after_school_club_days + saturday_shirts + sunday_church_shirt + sunday_rest_of_day_shirt

def shirts_needed (weeks shirts_per_week : ℕ) : ℕ :=
  weeks * shirts_per_week

theorem kendra_shirts_needed : shirts_needed 2 (
  shirts_needed_per_week 5 3 1 1 1
) = 22 :=
by
  simp [shirts_needed, shirts_needed_per_week]
  rfl

end kendra_shirts_needed_l336_336386


namespace parameterization_of_line_l336_336452

theorem parameterization_of_line :
  ∃ s l : ℝ, 
    (∀ t : ℝ, 
      let x := -9 + t * l,
          y := s + t * (-4)
      in y = (1 / 3) * x + 3) 
    ∧ s = 0 ∧ l = -12 := 
by 
  use 0, -12
  intros t
  dsimp
  split
  { exact sorry }
  { exact sorry }
sorry

end parameterization_of_line_l336_336452


namespace sum_first_n_terms_l336_336226

-- Definitions based on the problem conditions
def a₁ : ℕ := 2
def a₂ : ℕ := 8 - a₁
def a₃ : ℕ := 20 - (a₁ + a₂)

def sequence (n : ℕ) : ℕ :=
  if n = 1 then a₁
  else if n = 2 then a₂
  else if n = 3 then a₃
  else n^2 + n

/--
Given the conditions:
  - a₁ = 2
  - a₁ + a₂ = 8
  - a₁ + a₂ + a₃ = 20
Show that the sum of the first n terms of the sequence can be expressed as:
  ∑ k=1 to n a_k = (n(n + 1)(2n + 4)) / 3
-/
theorem sum_first_n_terms (n : ℕ) : 
  (∑ k in Finset.range n, sequence (k + 1)) = n * (n + 1) * (2 * n + 4) / 3 :=
sorry

end sum_first_n_terms_l336_336226


namespace arrangements_count_l336_336244

theorem arrangements_count (A B C D E : Type) (positions : Finset ℕ) (h₁ : positions = {1, 2, 3, 4}) :
  let people := {A, B, C, D, E}
  ∃ f : Fin 4 → people, (∀ i, f i ∉ (positions.filter (λ x, f x = A))) ∧ (positions.card = 4) ∧ (people.card = 5) → count f = 42 :=
by
  sorry

end arrangements_count_l336_336244


namespace hexagon_dot_product_result_l336_336642

variable {V : Type*} [InnerProductSpace ℝ V]
variable {A B C D E F : V}
variable {side_length : ℝ}

-- Definition of a regular hexagon with side length 1
def is_regular_hexagon (A B C D E F : V) (side_length : ℝ) : Prop :=
  dist A B = side_length ∧
  dist B C = side_length ∧
  dist C D = side_length ∧
  dist D E = side_length ∧
  dist E F = side_length ∧
  dist F A = side_length ∧
  angle A B C = π / 3 ∧
  angle B C D = π / 3 ∧
  angle C D E = π / 3 ∧
  angle D E F = π / 3 ∧
  angle E F A = π / 3 ∧
  angle F A B = π / 3

-- Given a regular hexagon with side length 1
axiom hexagon_is_regular : is_regular_hexagon A B C D E F 1

-- Statement of the proof problem
theorem hexagon_dot_product_result :
  ((B - A) + (C - D)) ⬝ ((D - A) + (E - B)) = -1 :=
by
  -- The proof steps would go here
  sorry

end hexagon_dot_product_result_l336_336642


namespace binomial_12_6_eq_924_l336_336963

theorem binomial_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binomial_12_6_eq_924_l336_336963


namespace acceptable_N_value_l336_336261

noncomputable def discriminant_game (N : ℕ) : Prop :=
  ∀ (a : ℕ -> ℕ) (n : ℕ), ∃ (β : ℕ), ∀ s₁ s₂ : ℕ → ℕ, 
  (∑ k in fin_range n + 1, a k * β ^ k) = (∑ k in fin_range n + 1, s₁ k * β ^ k) → a = s₁ 

theorem acceptable_N_value : discriminant_game N → N = 1 :=
by
  sorry

end acceptable_N_value_l336_336261


namespace nested_expression_rational_count_l336_336613

theorem nested_expression_rational_count : 
  let count := Nat.card {n : ℕ // 1 ≤ n ∧ n ≤ 2021 ∧ ∃ m : ℕ, m % 2 = 1 ∧ m * m = 1 + 4 * n}
  count = 44 := 
by sorry

end nested_expression_rational_count_l336_336613


namespace find_B_squared_l336_336238

def f (x : ℝ) : ℝ := sqrt 27 + 100 / (sqrt 27 + 100 / (sqrt 27 + 100 * x / (x + 1)))

theorem find_B_squared :
  let B := (abs ((- sqrt 27 + sqrt (27 + 40000)) / 200)
           + abs ((- sqrt 27 - sqrt (27 + 40000)) / 200)) in
  B^2 = 4.0027 :=
by
  sorry

end find_B_squared_l336_336238


namespace increasing_function_l336_336196

-- Definitions of the functions
def f (x : ℝ) : ℝ := (1 / 2) * x
def g (x : ℝ) : ℝ := -3 * x
def h (x : ℝ) : ℝ := -x ^ 2
def k (x : ℝ) : ℝ := -1 / x

theorem increasing_function (x : ℝ) (y : ℝ) :
  (0 < x → 0 < (f x - f y)) ∧ ¬ (0 < x → 0 < (g x - g y)) ∧ 
  ¬ (0 < x → 0 < (h x - h y)) ∧ ¬ (0 < x → 0 < (k x - k y)) :=
by
  sorry

end increasing_function_l336_336196


namespace cube_sum_181_5_l336_336082

theorem cube_sum_181_5
  (u v w : ℝ)
  (h : (u - real.cbrt 17) * (v - real.cbrt 67) * (w - real.cbrt 97) = 1/2)
  (huvw_distinct : u ≠ v ∧ u ≠ w ∧ v ≠ w):
  u^3 + v^3 + w^3 = 181.5 :=
sorry

end cube_sum_181_5_l336_336082


namespace det_A_power_five_l336_336329

variable (A : Matrix n n ℝ) -- Defining a matrix A of dimension n x n over the field of real numbers

-- Given condition
axiom h : det A = -3

-- Statement of the math proof problem
theorem det_A_power_five : det (A ^ 5) = -243 :=
by sorry

end det_A_power_five_l336_336329


namespace find_matrix_N_l336_336599

theorem find_matrix_N (N : Matrix (Fin 2) (Fin 2) ℝ)
    (h : ∀ (w : Fin 2 → ℝ), Matrix.mulVec N w = 7 • w) :
    N = Matrix.scalar 2 7 := by
    sorry

end find_matrix_N_l336_336599


namespace find_symmetric_point_l336_336264

-- Define the given conditions and line properties in Lean
def inclination_angle := 135
def passing_point := (1, 1)

-- Define the symmetric point function with the conditions and equations given
def symmetric_point (a b : ℝ) := 
  let l := λ x y: ℝ, x + y - 2 = 0 in 
  ∃ x y, 
    (x = 3 ∧ y = 4) ∧
    (
      (b - y) / (a - x) = -1 ∧ 
      (a + x) / 2 + (b + y) / 2 - 2 = 0
    )

theorem find_symmetric_point :
  (∃ a b, symmetric_point a b ∧ a = -2 ∧ b = -1) :=
sorry

end find_symmetric_point_l336_336264


namespace decimal_sum_sqrt_l336_336433

theorem decimal_sum_sqrt (a b : ℝ) (h₁ : a = Real.sqrt 5 - 2) (h₂ : b = Real.sqrt 13 - 3) : 
  a + b - Real.sqrt 5 = Real.sqrt 13 - 5 := by
  sorry

end decimal_sum_sqrt_l336_336433


namespace geom_sequence_min_b_n_l336_336359

section
variables {a : ℝ} (a_n : ℕ → ℝ) (b_n : ℕ → ℝ)

-- Given conditions
axiom (h1 : a_1 = 2)
axiom (h2 : ∀ n, a_n = a_1 * 2^(n-1))
axiom (h3 : a_1 + a_2 = (a_1 + a_1 * 2))
axiom (h4 : b_n n = (1 - 2 / (a_n n))^2 + a * (1 + 1 / (a_n n)))

-- Part I: Prove the general formula for the sequence {a_n}
theorem geom_sequence (n : ℕ) : a_n n = 2^n :=
by sorry

-- Part II: Prove the minimum term of the sequence {b_n} for a in [0,2]
theorem min_b_n : 
  if 0 ≤ a ∧ a < 1 then 
    ∃ n_0, ∀ n, b_n n_0 = (3 / 2) * a
  else if a = 1 then 
    ∃ n_0, ∀ n, b_n n_0 = 3 / 2
  else if 1 < a ∧ a ≤ 2 then 
    ∃ n_0, ∀ n, b_n n_0 = (5 / 4) * a + 1 / 4 
  else 
    False :=
by sorry
end

end geom_sequence_min_b_n_l336_336359


namespace circle_circumference_range_l336_336078

theorem circle_circumference_range (d : ℝ) (h : d = 1) : 
  let C := Real.pi * d in 3 < C ∧ C < 4 := 
by
  sorry

end circle_circumference_range_l336_336078


namespace mila_list_has_49_integers_l336_336790

-- Definitions based on the conditions
def smallest_multiple_of_24_perfect_square := 144
def smallest_multiple_of_24_perfect_fourth_power := 1296

def mila_list_count : ℕ :=
  let multiples_of_24 := {n | ∃ k : ℕ, n = 24 * k}
  let mila_list := {n ∈ multiples_of_24 | smallest_multiple_of_24_perfect_square ≤ n ∧ n ≤ smallest_multiple_of_24_perfect_fourth_power}
  mila_list.card

-- Proof statement
theorem mila_list_has_49_integers :
  mila_list_count = 49 :=
sorry

end mila_list_has_49_integers_l336_336790


namespace domino_swap_correct_multiplication_l336_336748

theorem domino_swap_correct_multiplication :
  ∃ (a b c d e f : ℕ), 
    a = 2 ∧ b = 3 ∧ c = 1 ∧ d = 3 ∧ e = 12 ∧ f = 3 ∧ 
    a * b = 6 ∧ c * d = 3 ∧ e * f = 36 ∧
    ∃ (x y : ℕ), x * y = 36 := sorry

end domino_swap_correct_multiplication_l336_336748


namespace runners_catch_up_again_l336_336864

noncomputable def catch_up_again (V1 V2 : ℝ) (initial_laps_second : ℝ) : ℝ :=
  let new_V2 := 2 * V2
  in if V1 > new_V2 then 
      (initial_laps_second + 2)
     else sorry

theorem runners_catch_up_again :
  ∀ V1 V2 : ℝ, 
    (V1 = 3 * V2) → 
    catch_up_again V1 V2 (1 / 2) = 2.5 :=
begin
  intros,
  rw [catch_up_again, 
      if_pos],
  { norm_num },
  { apply lt_of_sub_pos,
    norm_num,
    linarith }
end

end runners_catch_up_again_l336_336864


namespace polar_circle_equation_l336_336733

-- Define the center and radius
def c := (1 : ℝ, 0 : ℝ)
def r := 1

-- Define the Cartesian equation of the circle with given center and radius
def circle_cartesian (x y : ℝ) := (x - c.1)^2 + y^2 = r^2

-- Define the polar coordinates substitution
def x (ρ θ : ℝ) := ρ * Real.cos θ
def y (ρ θ : ℝ) := ρ * Real.sin θ

-- State the final problem to be proved
theorem polar_circle_equation (θ : ℝ) : ∃ ρ : ℝ, ρ = 2 * Real.cos θ := 
sorry

end polar_circle_equation_l336_336733


namespace brokerage_percentage_calculation_l336_336076

-- Defining the conditions as constants
constant total_amount_before_brokerage : ℝ := 106
constant cash_realized : ℝ := 106.25

-- Statement that defines the proof
theorem brokerage_percentage_calculation :
  ∃ (brokerage_percentage : ℝ), brokerage_percentage ≈ 0.2358 ∧ 
  brokerage_percentage = ( (total_amount_before_brokerage - cash_realized) / total_amount_before_brokerage ) * 100 := 
by
  sorry

end brokerage_percentage_calculation_l336_336076


namespace no_more_than_50_red_suits_l336_336034

theorem no_more_than_50_red_suits :
  ∃ (red blue : Fin 100 → Prop),
  (∀ i, red i ∨ blue i) ∧
  (∀ i, red i → blue (i + 9)) ∧
  (∀ i, red i → blue (i - 9)) ∧
  (∀ i, blue i → red (i + 9)) ∧
  (∀ i, blue i → red (i - 9)) ∧
  F | > ∃ i (proof_not_obtrusive: ((red i ∧ blue i) so)(independent_event: (red i suppersumes "blue i) ∧ (blue i justifies "red i",
  set_of (red) ≤ 50)
:= sorry

end no_more_than_50_red_suits_l336_336034


namespace concurrency_of_six_lines_l336_336267

/-- 
Given a triangle ABC and a point D.
Points E, F, G are on lines AD, BD, and CD respectively.
K is the intersection of AF and BE, 
L is the intersection of BG and CF, 
M is the intersection of CE and AG.
P, Q, R are intersections of lines DK and AB, DL and BC, DM and AC, respectively.
Prove that the six lines AL, EQ, BM, FR, CK, GP intersect at a single point.
--/
theorem concurrency_of_six_lines
  (A B C D E F G K L M P Q R : Type)
  (hE : on_line E (line A D))
  (hF : on_line F (line B D))
  (hG : on_line G (line C D))
  (hK : intersection K (line A F) (line B E))
  (hL : intersection L (line B G) (line C F))
  (hM : intersection M (line C E) (line A G))
  (hP : intersection P (line D K) (line A B))
  (hQ : intersection Q (line D L) (line B C))
  (hR : intersection R (line D M) (line A C))
  : concurrency (line A L) (line E Q) (line B M) (line F R) (line C K) (line G P) := 
sorry

end concurrency_of_six_lines_l336_336267


namespace greatest_integer_with_gcd_6_l336_336877

theorem greatest_integer_with_gcd_6 (x : ℕ) :
  x < 150 ∧ gcd x 12 = 6 → x = 138 :=
by
  sorry

end greatest_integer_with_gcd_6_l336_336877


namespace domain_of_log_base_5_range_of_3_pow_neg_l336_336826

theorem domain_of_log_base_5 (x : ℝ) : (1 - x > 0) -> (x < 1) :=
sorry

theorem range_of_3_pow_neg (y : ℝ) : (∃ x : ℝ, y = 3 ^ (-x)) -> (y > 0) :=
sorry

end domain_of_log_base_5_range_of_3_pow_neg_l336_336826


namespace find_cost_price_l336_336508

-- Define the conditions and the problem
variables (CP SP1 SP2 : ℝ)

-- Condition 1: The watch was sold at a loss of 9%
def loss_condition := SP1 = CP * 0.91

-- Condition 2: If it was sold for Rs. 220 more, there would have been a gain of 4%
def gain_condition := SP2 = CP * 1.04

-- Additional condition: SP2 is Rs. 220 more than SP1
def additional_condition := SP2 = SP1 + 220

-- The final statement we need to prove
theorem find_cost_price (h1 : loss_condition CP SP1) (h2 : gain_condition CP SP2) (h3 : additional_condition SP1 SP2) : 
  CP ≈ 1692.31 :=
by
  sorry

end find_cost_price_l336_336508


namespace hyperbola_condition_l336_336849

noncomputable def hyperbola_eccentricity_difference (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  let e_2pi_over_3 := Real.sqrt 3 + 1
  let e_pi_over_3 := (Real.sqrt 3) / 3 + 1
  e_2pi_over_3 - e_pi_over_3

theorem hyperbola_condition (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  hyperbola_eccentricity_difference a b h1 h2 = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end hyperbola_condition_l336_336849


namespace Kim_gum_needed_l336_336388

theorem Kim_gum_needed :
  let cousins := 4
  let gum_per_cousin := 5
  let gum_needed := cousins * gum_per_cousin
  gum_needed = 20 :=
by
  let cousins := 4
  let gum_per_cousin := 5
  let gum_needed := cousins * gum_per_cousin
  calc
    gum_needed = 4 * 5 := by rfl
             ... = 20   := by norm_num

end Kim_gum_needed_l336_336388


namespace circumscribed_sphere_surface_area_l336_336361

theorem circumscribed_sphere_surface_area (PA PB PC : ℝ) (h1 : PA = 1) (h2 : PB = 2) (h3 : PC = 3) 
  (h_perpendicular : ∀ {P A B C : Type}, perpendicular PA PB ∧ perpendicular PB PC ∧ perpendicular PC PA) :
  4 * Real.pi * (Real.sqrt (PA ^ 2 + PB ^ 2 + PC ^ 2) / 2) ^ 2 = 14 * Real.pi :=
by 
  sorry

end circumscribed_sphere_surface_area_l336_336361


namespace find_ellipse_equation_find_m_range_prove_slope_sum_constant_l336_336569

noncomputable def ellipse_equation : String := 
  "The equation of the ellipse is " ++ "x^2 / 4 + y^2 = 1"

def eccentricity : ℝ := sqrt 3 / 2

def segment_length : ℝ := 1

def intersect_length (a b : ℝ) : Prop := 
  (2 * b^2 / a = segment_length)

theorem find_ellipse_equation :
  ∃ (a b c : ℝ), 
    a > b ∧ b > 0 ∧ 
    (c / a = eccentricity) ∧
    (a^2 = b^2 + c^2) ∧ 
    intersect_length a b ∧ 
    (x^2 / a^2 + y^2 / b^2 = 1) := 
sorry

def angle_bisector_range (m : ℝ) : Prop := 
  (-3 / 2 < m ∧ m < 3 / 2)

theorem find_m_range (a b c : ℝ) :
  a > b ∧ b > 0 ∧
  (c / a = eccentricity) ∧
  (a = 2) ∧ (b = 1) ∧ (c = sqrt 3) →
  ∃ m : ℝ, angle_bisector_range m := 
sorry

def slope_sum_constant (k k1 k2 : ℝ) : Prop := 
  (1 / (k * k1) + 1 / (k * k2) = -8)

theorem prove_slope_sum_constant (P: ℝ×ℝ, a b c: ℝ, k k1 k2: ℝ):
  a > b ∧ b > 0 ∧ 
  (c / a = eccentricity) ∧ 
  (a = 2) ∧ (b = 1) ∧ (c = sqrt 3) →
  (∃ x y, (P = (x, y)) ∧ 
          (y = sqrt (1 - x^2 / 4)) ∧ 
          k ≠ 0 ∧ 
          (k1 = y / (x + sqrt 3)) ∧ 
          (k2 = y / (x - sqrt 3)) ∧
          slope_sum_constant k k1 k2) 
:= sorry

end find_ellipse_equation_find_m_range_prove_slope_sum_constant_l336_336569


namespace consecutive_non_divisible_by_3_l336_336920

def sum_of_digits_first_100 (n : ℕ) : ℕ :=
  let digits := (to_digits n).take 100
  List.sum digits

theorem consecutive_non_divisible_by_3 :
  ∃ (a b c : ℕ), 
  (a % 3 ≠ 0) ∧ (b % 3 ≠ 0) ∧ (c % 3 ≠ 0) ∧
  ∀ k : ℕ, (k > 103) → 
  ∃ n : ℕ, 
  (sum_of_digits_first_100 n) ≤ 900 ∧
  (b = a + sum_of_digits_first_100 a) ∧ 
  (c = b + sum_of_digits_first_100 b) :=
by
  sorry

end consecutive_non_divisible_by_3_l336_336920


namespace mark_eats_fruit_l336_336787

-- Question: How many pieces of fruit did Mark eat in the first four days of the week?
theorem mark_eats_fruit (total_fruit : ℕ) (kept_fruit : ℕ) (friday_fruit : ℕ) :
  total_fruit = 10 → kept_fruit = 2 → friday_fruit = 3 → (total_fruit - kept_fruit - friday_fruit) = 5 :=
by
  intros h_total h_kept h_friday
  rw [h_total, h_kept, h_friday]
  simp
  exact rfl

end mark_eats_fruit_l336_336787


namespace equivalent_proof_l336_336310

-- Define the sequence {a_n}
def a_n (n : ℕ) : ℚ := (2/3) * (-1/3)^(n-1)

-- Sum of the first n terms in the sequence {a_n}
def S_n (n : ℕ) : ℚ := (sum (finset.range n) (λ k, a_n k))

-- Define the sequence {b_n}
def b_n (n : ℕ) : ℚ := (1 / 2)

-- Definition for a_2
def a_2 := 3

-- Condition for sequence {a_n} 
def a_condition (n : ℕ) : Prop := ∀ n ≥ 2, a_n (n-1) + a_n (n+1) = 2 * a_n n

-- General term formula for {a_n}
def a_general (n : ℕ) : ℚ := n + 1

-- Define the sequence {c_n}
def c_n (n : ℕ) : ℚ := a_n n / b_n n

-- The proof statement
theorem equivalent_proof:
  ∀ n, 
    (b_n n = 1/2) ∧
    (a_condition n → a_n n = n + 1) ∧
    (c_n n =  (n + 1) / n) ∧
    (∃ k t, t ≠ n ∧ k ≠ n ∧ k ≠ t ∧ c_n n = c_n k * c_n t) := 
by
  sorry -- Proof omitted

end equivalent_proof_l336_336310


namespace arrangements_count_l336_336246

-- Defining the conditions
def person : Type := {A, B, C, D, E}
def position : Type := {posA, posB, posC, posD}
def arrangements (p : person → position) : Prop := (p A ≠ posA)

-- Statement that proves the total number of arrangements equals 42
theorem arrangements_count : ∃ (count : ℕ), count = 42 ∧ (∀ (p : person → position), arrangements p) -> sorry := sorry

end arrangements_count_l336_336246


namespace initial_milk_quantity_l336_336550

theorem initial_milk_quantity (A B C D : ℝ) (hA : A > 0)
  (hB : B = 0.55 * A)
  (hC : C = 1.125 * A)
  (hD : D = 0.8 * A)
  (hTransferBC : B + 150 = C - 150 + 100)
  (hTransferDC : C - 50 = D - 100)
  (hEqual : B + 150 = D - 100) : 
  A = 1000 :=
by sorry

end initial_milk_quantity_l336_336550


namespace abs_diff_eq_sqrt_l336_336637

theorem abs_diff_eq_sqrt (x1 x2 a b : ℝ) (h1 : x1 + x2 = a) (h2 : x1 * x2 = b) : 
  |x1 - x2| = Real.sqrt (a^2 - 4 * b) :=
by
  sorry

end abs_diff_eq_sqrt_l336_336637


namespace arithmetic_expression_result_l336_336117

theorem arithmetic_expression_result :
  ∃ (a b c d : ℕ), 
    a = 2 ∧ b = 4 ∧ c = 12 ∧ d = 40 ∧ 
    (d / b + c + a = 24) :=
begin
  use [2, 4, 12, 40],
  split, exact rfl,
  split, exact rfl,
  split, exact rfl,
  split, exact rfl,
  simp,
  linarith,
end

end arithmetic_expression_result_l336_336117


namespace pb_value_l336_336428

variable {α : Type*} [metric_space α] [ordered_ring α] [normed_linear_ordered_field α]

theorem pb_value (A B C D P : α) (PA PD PC sqrt_2 : α)
  (hPA : PA = 3)
  (hPD : PD = 4)
  (hPC : PC = 5)
  (PB : α) :
  PB = 3 * sqrt_2 :=
by
  sorry

end pb_value_l336_336428


namespace perimeter_difference_is_zero_l336_336990

-- Define the given conditions for the two figures.
def rect1_width : ℕ := 4
def rect1_height : ℕ := 3

def rect2a_width : ℕ := 3
def rect2a_height : ℕ := 2

def rect2b_width : ℕ := 2
def rect2b_height : ℕ := 3

-- Calculate the perimeters using the given dimensions.
def perimeter (width height : ℕ) : ℕ :=
  2 * (width + height)

def perimeter_rect1 : ℕ := perimeter rect1_width rect1_height

def perimeter_rect2 : ℕ := (perimeter rect2a_width rect2a_height) + (perimeter rect2b_width rect2b_height) - 2 * rect2a_width

-- The theorem to be proven: the perimeters' difference is 0.
theorem perimeter_difference_is_zero :
  |perimeter_rect1 - perimeter_rect2| = 0 :=
  by
    sorry

end perimeter_difference_is_zero_l336_336990


namespace candle_height_relation_l336_336162

variable (t : ℝ) -- burning time in hours
def initial_height := 20 -- initial height in cm
def burn_rate := 4 -- burn rate in cm per hour

theorem candle_height_relation :
  ∃ (h : ℝ), h = initial_height - burn_rate * t :=
by
  exists 20 - 4 * t
  sorry

end candle_height_relation_l336_336162


namespace prove_mutually_exclusive_l336_336627

def bag : List String := ["red", "red", "red", "black", "black"]

def at_least_one_black (drawn : List String) : Prop :=
  "black" ∈ drawn

def all_red (drawn : List String) : Prop :=
  ∀ b ∈ drawn, b = "red"

def events_mutually_exclusive : Prop :=
  ∀ drawn, at_least_one_black drawn → ¬all_red drawn

theorem prove_mutually_exclusive :
  events_mutually_exclusive
:= by
  sorry

end prove_mutually_exclusive_l336_336627


namespace factor_difference_of_squares_l336_336586

theorem factor_difference_of_squares (x : ℝ) :
  x^2 - 169 = (x - 13) * (x + 13) := by
  have h : 169 = 13^2 := by norm_num
  rw h
  exact by ring

end factor_difference_of_squares_l336_336586


namespace percentage_paid_to_X_l336_336481

theorem percentage_paid_to_X (X Y : ℝ) (h_sum : X + Y = 500) (h_Y : Y = 227.27) : 
  (X / Y) * 100 = 120 :=
by
  have h_X : X = 500 - 227.27 := 
  begin
    rw [h_Y] at h_sum,
    linarith,
  end,
  rw [h_X, h_Y],
  norm_num,
  sorry

end percentage_paid_to_X_l336_336481


namespace cities_distance_l336_336822

theorem cities_distance (map_distance_in_inches : ℝ)
  (scale_in_inches : ℝ) (scale_in_miles : ℝ)
  (mile_to_km : ℝ) : 
  map_distance_in_inches = 30 ∧ scale_in_inches = 0.5 ∧ scale_in_miles = 8 ∧ mile_to_km = 1.60934 →
  (let distance_in_miles := (map_distance_in_inches / scale_in_inches) * scale_in_miles in
  let distance_in_kilometers := distance_in_miles * mile_to_km in
  distance_in_miles = 480 ∧ distance_in_kilometers = 772.4832) :=
by 
  intros h 
  cases h with h_map_distance h_rest
  cases h_rest with h_scale_in_inches h_rest 
  cases h_rest with h_scale_in_miles h_mile_to_km
  let distance_in_miles := (map_distance_in_inches / scale_in_inches) * scale_in_miles
  let distance_in_kilometers := distance_in_miles * mile_to_km
  have h_distance_in_miles : distance_in_miles = 480
    := by sorry
  have h_distance_in_kilometers : distance_in_kilometers = 772.4832
    := by sorry
  exact ⟨h_distance_in_miles, h_distance_in_kilometers⟩

end cities_distance_l336_336822


namespace remainder_when_four_times_n_minus_nine_divided_by_7_l336_336640

theorem remainder_when_four_times_n_minus_nine_divided_by_7 (n : ℤ) (h : n % 7 = 3) : (4 * n - 9) % 7 = 3 := by
  sorry

end remainder_when_four_times_n_minus_nine_divided_by_7_l336_336640


namespace erased_digit_is_4_l336_336919

def sum_of_digits (n : ℕ) : ℕ := 
  sorry -- definition of sum of digits

def D (N : ℕ) : ℕ := N - sum_of_digits N

theorem erased_digit_is_4 (N : ℕ) (x : ℕ) 
  (hD : D N % 9 = 0) 
  (h_sum : sum_of_digits (D N) - x = 131) 
  : x = 4 :=
by
  sorry

end erased_digit_is_4_l336_336919


namespace last_digit_product_3_2001_7_2002_13_2003_l336_336449

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_product_3_2001_7_2002_13_2003 :
  last_digit (3^2001 * 7^2002 * 13^2003) = 9 :=
by
  sorry

end last_digit_product_3_2001_7_2002_13_2003_l336_336449


namespace find_k_for_coplanar_lines_l336_336425

def line1 (s k : ℝ) : ℝ × ℝ × ℝ :=
  (2 + s, 1 - 2 * k * s, 4 + k * s)

def line2 (t : ℝ) : ℝ × ℝ × ℝ :=
  (-t / 3, 2 + t, 5 - 2 * t)

def coplanar_lines (k : ℝ) : Prop :=
  ∃ s t : ℝ, line1 s k = line2 t

theorem find_k_for_coplanar_lines : coplanar_lines (-3) :=
by
  sorry

end find_k_for_coplanar_lines_l336_336425


namespace find_flights_of_stairs_l336_336421

def t_flight : ℕ := 11
def t_bomb : ℕ := 72
def t_spent : ℕ := 165
def t_diffuse : ℕ := 17

def total_time_running : ℕ := t_spent + (t_bomb - t_diffuse)
def flights_of_stairs (t_run: ℕ) (time_per_flight: ℕ) : ℕ := t_run / time_per_flight

theorem find_flights_of_stairs :
  flights_of_stairs total_time_running t_flight = 20 :=
by
  sorry

end find_flights_of_stairs_l336_336421


namespace soup_weight_on_fourth_day_l336_336203

-- Define the initial condition and the halving process.
def initial_weight : ℝ := 80
def after_days (n : ℕ) : ℝ := initial_weight / (2^n)

-- Propose the theorem to validate.
theorem soup_weight_on_fourth_day : after_days 4 = 5 := by
  sorry

end soup_weight_on_fourth_day_l336_336203


namespace purely_imaginary_value_of_a_l336_336340

theorem purely_imaginary_value_of_a (a : ℝ) : 
  (∃ z : ℂ, z = (a^2 + a - 2 : ℂ) + (a^2 - 1 : ℂ) * complex.I ∧ z.im = z) → 
  a = -2 :=
sorry

end purely_imaginary_value_of_a_l336_336340


namespace frank_peanuts_average_l336_336251

theorem frank_peanuts_average :
  let one_dollar := 7 * 1
  let five_dollar := 4 * 5
  let ten_dollar := 2 * 10
  let twenty_dollar := 1 * 20
  let total_money := one_dollar + five_dollar + ten_dollar + twenty_dollar
  let change := 4
  let money_spent := total_money - change
  let cost_per_pound := 3
  let total_pounds := money_spent / cost_per_pound
  let days := 7
  let average_per_day := total_pounds / days
  average_per_day = 3 :=
by
  sorry

end frank_peanuts_average_l336_336251


namespace find_x_squared_plus_y_squared_plus_z_squared_l336_336653

theorem find_x_squared_plus_y_squared_plus_z_squared
  (x y z : ℤ)
  (h1 : x + y + z = 3)
  (h2 : x^3 + y^3 + z^3 = 3) :
  x^2 + y^2 + z^2 = 57 :=
by
  sorry

end find_x_squared_plus_y_squared_plus_z_squared_l336_336653


namespace number_divisible_by_k_cube_l336_336338

theorem number_divisible_by_k_cube (k : ℕ) (h : k = 42) : ∃ n, (k^3) % n = 0 ∧ n = 74088 := by
  sorry

end number_divisible_by_k_cube_l336_336338


namespace students_taking_both_l336_336909

theorem students_taking_both (total_students music_students art_students neither_students : ℕ)
  (h1 : total_students = 500)
  (h2 : music_students = 30)
  (h3 : art_students = 20)
  (h4 : neither_students = 460) :
  ∃ both_students, (music_students + art_students - both_students = total_students - neither_students) ∧ both_students = 10 :=
by
  use 10
  split
  sorry -- The actual proof steps are skipped as instructed.

end students_taking_both_l336_336909


namespace largest_y_coordinate_of_degenerate_ellipse_l336_336083

theorem largest_y_coordinate_of_degenerate_ellipse :
  ∀ (x y : ℝ), (x^2 / 36 + (y + 5)^2 / 16 = 0) → y = -5 :=
by
  intros x y h
  sorry

end largest_y_coordinate_of_degenerate_ellipse_l336_336083


namespace triangle_arithmetic_angles_l336_336821

theorem triangle_arithmetic_angles (A B C : ℝ) (h1 : A + B + C = 180) :
  (∃ b : ℝ, b = 60 ∧ (A + C) = 2 * b) ↔ (A + C = 2 * B) :=
begin
  sorry
end

end triangle_arithmetic_angles_l336_336821


namespace relationship_between_y1_and_y2_l336_336036

-- Define the linear function
def linear_function (x : ℝ) : ℝ := -4 * x + 3

-- Define the points on the linear function
def P1 := (1 : ℝ, linear_function 1)
def P2 := (-3 : ℝ, linear_function (-3))

-- Define the y-coordinates
def y1 := P1.snd
def y2 := P2.snd

-- The theorem stating the relationship between y1 and y2
theorem relationship_between_y1_and_y2 : y1 < y2 :=
by
  -- Proof goes here
  sorry

end relationship_between_y1_and_y2_l336_336036


namespace sum_power_of_two_in_subset_l336_336011

theorem sum_power_of_two_in_subset (H : set ℕ) (hH1 : H ⊆ { x | x ≤ 1998}) (hH2 : H.card = 1000) :
  ∃ a b ∈ H, ∃ k : ℕ, a + b = 2^k :=
sorry

end sum_power_of_two_in_subset_l336_336011


namespace rational_part_irrational_l336_336414

theorem rational_part_irrational (p q : ℕ) (hp : Prime p) (hq : Prime q) (h : 1 ≤ q ∧ q ≤ p) :
  Irrational ((sqrt (p^2 + q) + p)^2) ∧ frac ((sqrt (p^2 + q) + p)^2) > 3 / 4 :=
by
  sorry

end rational_part_irrational_l336_336414


namespace knight_tour_impossible_l336_336955

/-- Define the colors of the squares on an 8x8 chessboard --/
inductive Color
| black
| white

/-- Function to determine the color of a given square --/
def square_color (file rank : Nat) : Color :=
  if (file + rank) % 2 == 0 then Color.black else Color.white

/-- Define the initial (a1) and final (h8) positions --/
def initial_position : (Nat × Nat) := (1, 1) -- a1
def final_position : (Nat × Nat) := (8, 8) -- h8

/-- Define the number of moves required for the knight to visit all squares exactly once --/
def required_moves : Nat := 63

/-- Define the knight's move that alternates the position across colors --/
lemma knight_alternates_colors (start_pos end_pos : Nat × Nat) (n_moves : Nat) :
  (square_color start_pos.1 start_pos.2) =
  (square_color end_pos.1 end_pos.2) → false :=
by
  sorry

/-- Main theorem stating the impossibility of the knight's tour from a1 to h8 --
 theorem knight_tour_impossible : ∀ (start_pos end_pos : Nat × Nat),
   start_pos = initial_position ∧ end_pos = final_position →
   required_moves % 2 = 1 →  
   (square_color start_pos.1 start_pos.2 = Color.black) →
   (square_color end_pos.1 end_pos.2 = Color.black) →
   ¬ ∃ (f : Nat → (Nat × Nat)),
     (f 0 = start_pos) ∧ (f required_moves = end_pos) ∧
     (∀ n < required_moves, (f (n + 1) = knight_move (f n))) :=
by
  sorry

end knight_tour_impossible_l336_336955


namespace sequence_properties_l336_336037

def a_n (n : ℕ) : ℕ :=
if n % 2 = 0 then 2^n + 1 else 2^n - 1

theorem sequence_properties :
  ∀ n : ℕ, 
    a_n n ∈ ℕ ∧ 
    (n % 2 = 0 → ∃ m : ℕ, a_n n = 5 * m^2) ∧ 
    (n % 2 = 1 → ∃ m : ℕ, a_n n = m^2) :=
by
  sorry

end sequence_properties_l336_336037


namespace ellipse_problem_l336_336666

theorem ellipse_problem (a b: ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : b^2 = a^2 - 3) 
  (h5 : a/2 = sqrt 3/2) (hpq : ∀ P Q : ℝ × ℝ, (P ≠ (0, b) ∧ P ≠ (0, -b)) ∧ (Q ≠ (0, b) ∧ Q ≠ (0, -b))
  ∧ (P, Q ∈ C)) :
  (∃ C : set (ℝ × ℝ), ∀ p : ℝ × ℝ, p ∈ C ↔ (1/a^2) * p.fst^2 + (1/b^2) * p.snd^2 = 1) 
  ∧ ∃ fixed_point : ℝ × ℝ, fixed_point = (0, 1/2) ∧ 
  ∀ P Q : ℝ × ℝ, line_through P Q ∩ C ⊆ line_through P fixed_point := sorry

def line_through (P Q : ℝ × ℝ) : set (ℝ × ℝ) := sorry

end ellipse_problem_l336_336666


namespace discount_percentage_l336_336080

theorem discount_percentage (cost_per_copy : ℝ) (copies : ℕ) (savings_per_person : ℝ) 
    (total_copies : ℕ) (total_savings : ℝ) : 
    cost_per_copy = 0.02 ∧ copies = 80 ∧ savings_per_person = 0.40 ∧ total_copies = 160 ∧ 
    total_savings = 0.80 → 
    let total_cost_without_discount := 3.20 in
    let total_cost_with_discount := 2.40 in
    let discount_amount := total_cost_without_discount - total_cost_with_discount in
    let discount_percentage := (discount_amount / total_cost_without_discount) * 100 in
    discount_percentage = 25 :=
by
  intros 
  simp only [*, mul_eq_mul_left_iff, sub_add_cancel]
  reals
  field_simp 
  norm_num
  sorry

end discount_percentage_l336_336080


namespace mark_eats_fruit_l336_336785

-- Question: How many pieces of fruit did Mark eat in the first four days of the week?
theorem mark_eats_fruit (total_fruit : ℕ) (kept_fruit : ℕ) (friday_fruit : ℕ) :
  total_fruit = 10 → kept_fruit = 2 → friday_fruit = 3 → (total_fruit - kept_fruit - friday_fruit) = 5 :=
by
  intros h_total h_kept h_friday
  rw [h_total, h_kept, h_friday]
  simp
  exact rfl

end mark_eats_fruit_l336_336785


namespace bags_needed_l336_336187

-- Define the dimensions of one raised bed
def length_of_bed := 8
def width_of_bed := 4
def height_of_bed := 1

-- Calculate the volume of one raised bed
def volume_of_one_bed := length_of_bed * width_of_bed * height_of_bed

-- Define the number of beds
def number_of_beds := 2

-- Calculate the total volume needed for both beds
def total_volume := number_of_beds * volume_of_one_bed

-- Define the volume of soil in one bag
def volume_per_bag := 4

-- Calculate the number of bags needed
def number_of_bags := total_volume / volume_per_bag

-- Prove that the number of bags needed is 16
theorem bags_needed : number_of_bags = 16 := by
  show number_of_bags = 16 from sorry

end bags_needed_l336_336187


namespace binomial_12_6_eq_924_l336_336962

theorem binomial_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binomial_12_6_eq_924_l336_336962


namespace concyclic_PFBM_ratio_EMEN_BDBP_l336_336735

variables (A B C I D E F P Q M N : Type) [triangle A B C]
          [incircle I (triangle A B C)] [touches I B C D] [touches I C A E] [touches I A B F]
          [point_on_arc P E F (opposite D)] [bp_intersects Q] 
          [ep_meet_bc_at M] [eq_meet_bc_at N]

-- Assuming the needed properties for isosceles triangle and touches
axiom equal_sides : AB = AC
axiom BP_intersects_Q : Q ∈ BP ∧ Q ∈ I
axiom EP_meets_BC_at_M : E ∈ EP ∧ M ∈ BC
axiom EQ_meets_BC_at_N : E ∈ EQ ∧ N ∈ BC

-- Prove the following statements:
theorem concyclic_PFBM : is_concyclic P F B M := sorry
theorem ratio_EMEN_BDBP : EM / EN = BD / BP := sorry

end concyclic_PFBM_ratio_EMEN_BDBP_l336_336735


namespace cos_theta_eq_four_fifths_l336_336415

open Real

theorem cos_theta_eq_four_fifths (a b : ℝ × ℝ) (theta : ℝ) 
  (h1 : a = (2,1)) 
  (h2 : a.1 + 2 * b.1 = 4 ∧ a.2 + 2 * b.2 = 5) :
  cos theta = (4 / 5) := 
sorry

end cos_theta_eq_four_fifths_l336_336415


namespace scout_weekend_earnings_l336_336059

theorem scout_weekend_earnings : 
  let base_pay_per_hour := 10.00
  let tip_per_customer := 5.00
  let saturday_hours := 4
  let saturday_customers := 5
  let sunday_hours := 5
  let sunday_customers := 8
  in
  (saturday_hours * base_pay_per_hour + saturday_customers * tip_per_customer) +
  (sunday_hours * base_pay_per_hour + sunday_customers * tip_per_customer) = 155.00 := sorry

end scout_weekend_earnings_l336_336059


namespace solve_system_l336_336576

theorem solve_system (x y : ℚ) 
  (h₁ : 7 * x - 14 * y = 3) 
  (h₂ : 3 * y - x = 5) : 
  x = 79 / 7 ∧ y = 38 / 7 := 
by 
  sorry

end solve_system_l336_336576


namespace math_problem_tan_l336_336518

-- Definitions based on problem conditions
def tan_sum_identity (a b : ℝ) : ℝ :=
  (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)

-- Problem statement as a Lean theorem
theorem math_problem_tan:
  (1 + Real.tan (23 * (Real.pi / 180))) * (1 + Real.tan (22 * (Real.pi / 180))) = 2 :=
  sorry

end math_problem_tan_l336_336518


namespace count_of_quadratic_equations_l336_336841

def is_quadratic_equation_in_one_variable (eq : ℚ[X]) : Prop :=
  ∃ (a b c : ℚ), eq = a * X^2 + b * X + c ∧ a ≠ 0

def eq1 : ℚ[X] := X^2 + 1
def eq2 : ℚ[X] := 2 * X^2 - 3 * X * X
def eq3 : ℚ[X] := X^2 - 1 / X  -- This is symbolic and not valid in polynomials
def eq4 (a : ℚ) : ℚ[X] := a * X^2 - X + 2

theorem count_of_quadratic_equations :
  (if is_quadratic_equation_in_one_variable eq1 then 1 else 0) +
  (if is_quadratic_equation_in_one_variable eq2 then 1 else 0) +
  (if is_quadratic_equation_in_one_variable eq3 then 1 else 0) +
  (if is_quadratic_equation_in_one_variable (eq4 1) then 1 else 0) = 1 :=
by {
  sorry
}

end count_of_quadratic_equations_l336_336841


namespace points_in_circle_of_radius_1_over_7_l336_336802

theorem points_in_circle_of_radius_1_over_7 :
  ∀ (points : Fin 51 → (ℝ × ℝ)), ∃ (c : ℝ × ℝ),
    let radius := 1/7
    in (∃ p1 p2 p3, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
      dist (fst points p1, snd points p1) c < radius ∧
      dist (fst points p2, snd points p2) c < radius ∧
      dist (fst points p3, snd points p3) c < radius) :=
by sorry

end points_in_circle_of_radius_1_over_7_l336_336802


namespace binomial_12_6_eq_1848_l336_336973

theorem binomial_12_6_eq_1848 : (Nat.choose 12 6) = 1848 :=
  sorry

end binomial_12_6_eq_1848_l336_336973


namespace problem_statement_l336_336290

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def range_of_n (n : ℝ) : Prop :=
  n ≥ (4 * Real.sqrt 3 / 3) ∧ n ≤ 4

theorem problem_statement :
  let a := 2
  let b := Real.sqrt 3
  ellipse_equation ∧ range_of_n n :=
by
  sorry

end problem_statement_l336_336290


namespace bound_of_absolute_deviation_of_proportion_of_germinated_seeds_l336_336157

theorem bound_of_absolute_deviation_of_proportion_of_germinated_seeds : 
  ∀ (n : ℕ) (p bound : ℝ) (P : ℝ), n = 600 → p = 0.9 → P = 0.995 → bound = 0.034 →
    probability (| (observed_frequency_germinated_seeds / n) - p | < bound) = P :=
by
  sorry

end bound_of_absolute_deviation_of_proportion_of_germinated_seeds_l336_336157


namespace pieces_missing_l336_336065

def total_pieces : ℕ := 32
def pieces_present : ℕ := 24

theorem pieces_missing : total_pieces - pieces_present = 8 := by
sorry

end pieces_missing_l336_336065


namespace find_f_f_10_l336_336026

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else real.log10 x

theorem find_f_f_10 : f (f 10) = 2 :=
  by
    sorry

end find_f_f_10_l336_336026


namespace arithmetic_sequence_general_term_sum_of_first_20_terms_l336_336661

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℤ := (-1)^n * (2 * n^2 - 1)
noncomputable def S_n (n : ℕ) : ℚ := n / (2 * n + 1)

theorem arithmetic_sequence_general_term (S_n_condition : ∀ n : ℕ, S_n n = n / (2 * n + 1)) :
  ∀ n : ℕ, a_n n = 2 * n - 1 :=
sorry

theorem sum_of_first_20_terms (S_n_condition : ∀ n : ℕ, S_n n = n / (2 * n + 1)) :
  ∑ i in finset.range 20, b_n (i+1) = 420 :=
sorry

end arithmetic_sequence_general_term_sum_of_first_20_terms_l336_336661


namespace sin_HAC_one_l336_336153

noncomputable def sin_angle_HAC : Prop :=
  let A := (0 : ℝ, 0 : ℝ, 0 : ℝ)
  let H := (0 : ℝ, 1 : ℝ, 1 : ℝ)
  let C := (1 : ℝ, 1 : ℝ, 0 : ℝ)
  let HA := (H.1 - A.1, H.2 - A.2, H.3 - A.3)
  let AC := (C.1 - A.1, C.2 - A.2, C.3 - A.3)
  let len_HA := real.sqrt (HA.1 ^ 2 + HA.2 ^ 2 + HA.3 ^ 2)
  let len_AC := real.sqrt (AC.1 ^ 2 + AC.2 ^ 2 + AC.3 ^ 2)
  let HA_dot_AC := HA.1 * AC.1 + HA.2 * AC.2 + HA.3 * AC.3
  let cos_theta := HA_dot_AC / (len_HA * len_AC)
  let sin_theta := real.sqrt (1 - cos_theta ^ 2)
  sin_theta = 1

theorem sin_HAC_one : sin_angle_HAC :=
by
  sorry

end sin_HAC_one_l336_336153


namespace binomial_12_6_eq_924_l336_336975

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := sorry

end binomial_12_6_eq_924_l336_336975


namespace cos_sum_diff_l336_336236

theorem cos_sum_diff (x y : ℝ) : 
  cos (x + y) - cos (x - y) = -2 * sin x * sin y := 
sorry

end cos_sum_diff_l336_336236


namespace domain_log_base_5_range_3_pow_neg_x_l336_336824

theorem domain_log_base_5 (x : ℝ) :
  (∃ y : ℝ, y = logBase 5 (1 - x)) ↔ x < 1 :=
by
  sorry

theorem range_3_pow_neg_x (y : ℝ) :
  (∃ x : ℝ, y = 3 ^ (-x)) ↔ y > 0 :=
by
  sorry

end domain_log_base_5_range_3_pow_neg_x_l336_336824


namespace sum_of_convex_polygons_eq_one_l336_336005

-- Define the basic setting of points S, convex polygons, and the functions a(P) and b(P)
open Set Finset

variable {α : Type*} [LinearOrderedField α]

/-- 
  S is a finite set of points in the plane with no three points collinear.
  For each convex polygon P with vertices in S, 
    a(P) is the number of vertices of P,
    b(P) is the number of points in S outside P.
    We consider line segments, points, and the empty set as convex polygons with 
    2, 1, and 0 vertices respectively.
-/
def is_convex_polygon (P : Finset (Fin 2 × Fin 2)) (S : Finset (Fin 2 × Fin 2)) : Prop :=
  P ⊆ S ∧ ∀ (a b c : Fin 2 × Fin 2), a ≠ b → a ≠ c → b ≠ c → a ∈ P → b ∈ P → c ∈ P → ¬ are_collinear a b c

noncomputable def a (P : Finset (Fin 2 × Fin 2)) : ℕ := P.card

noncomputable def b (P : Finset (Fin 2 × Fin 2)) (S : Finset (Fin 2 × Fin 2)) : ℕ :=
  (S \ P).card

-- The main theorem
theorem sum_of_convex_polygons_eq_one (S : Finset (Fin 2 × Fin 2)) (h : ∀ a b c ∈ S, ¬ are_collinear a b c)
  (x : α) :
  ∑ P in S.powerset.filter (λ P, is_convex_polygon P S), (x ^ a P * (1 - x) ^ b P S) = 1 :=
sorry

end sum_of_convex_polygons_eq_one_l336_336005


namespace even_function_solution_l336_336707

theorem even_function_solution :
  (∃ f : ℝ → ℝ, (∀ x : ℝ, f(-x) = f(x)) ∧ (∀ x : ℝ, x * f(x + 2) = (x + 2) * f(x) + 2) ∧ (∃ x : ℝ, f x ≠ 0)) →
  ∀ f : ℝ → ℝ, (∀ x : ℝ, f(-x) = f(x)) → (∀ x : ℝ, x * f(x + 2) = (x + 2) * f(x) + 2) → f(2023) = -1 :=
by
  intros h f hf1 hf2
  -- proof would go here
  sorry

end even_function_solution_l336_336707


namespace non_empty_subsets_count_l336_336689

def odd_set : Finset ℕ := {1, 3, 5, 7, 9}
def even_set : Finset ℕ := {2, 4, 6, 8}

noncomputable def num_non_empty_subsets_odd : ℕ := 2 ^ odd_set.card - 1
noncomputable def num_non_empty_subsets_even : ℕ := 2 ^ even_set.card - 1

theorem non_empty_subsets_count :
  num_non_empty_subsets_odd + num_non_empty_subsets_even = 46 :=
by sorry

end non_empty_subsets_count_l336_336689


namespace paper_plates_cost_l336_336098

theorem paper_plates_cost (P C x : ℝ) 
(h1 : 100 * P + 200 * C = 6.00) 
(h2 : x * P + 40 * C = 1.20) : 
x = 20 := 
sorry

end paper_plates_cost_l336_336098


namespace num_students_taking_music_l336_336910

-- Definitions based on given conditions
def total_students : ℕ := 500
def students_taking_art : ℕ := 20
def students_taking_both_music_and_art : ℕ := 10
def students_taking_neither_music_nor_art : ℕ := 450

-- Theorem statement to prove the number of students taking music
theorem num_students_taking_music :
  ∃ (M : ℕ), M = 40 ∧ 
  (total_students - students_taking_neither_music_nor_art = M + students_taking_art - students_taking_both_music_and_art) := 
by
  sorry

end num_students_taking_music_l336_336910


namespace remainder_of_poly_div_l336_336492

theorem remainder_of_poly_div (x : ℤ) : 
  (x + 1)^2009 % (x^2 + x + 1) = x + 1 :=
by
  sorry

end remainder_of_poly_div_l336_336492


namespace apollonian_circle_eq_sym_curve_vertical_sym_curve_slope_l336_336945

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the condition for the moving point M and the ratio of distances
def is_on_apollonian_circle (M : ℝ × ℝ) : Prop :=
  let |MA| := Real.sqrt ((M.1 + 1)^2 + M.2^2)
  let |MB| := Real.sqrt ((M.1 - 2)^2 + M.2^2)
  |MA| / |MB| = 2

-- Prove the equation of the curve C
theorem apollonian_circle_eq :
  ∀ M : ℝ × ℝ, is_on_apollonian_circle M ↔ (M.1 - 3)^2 + M.2^2 = 4 := sorry

-- Define the points for the tangent line condition
def P : ℝ × ℝ := (5, -4)
def C_center : ℝ × ℝ := (3, 0)

-- Define the reflection over a vertical line
def reflect_vertical (line_x : ℝ) (pt : ℝ × ℝ) : ℝ × ℝ := (2 * line_x - pt.1, pt.2)

-- Prove the symmetric curve C' for vertical tangent line
theorem sym_curve_vertical :
  ∀ M : ℝ × ℝ, reflect_vertical 5 (3, 0) = (7, 0) ↔ (M.1 - 7)^2 + M.2^2 = 4 := sorry

-- Define the reflection over a line with slope
def reflect_over_line (a b c : ℝ) (pt : ℝ × ℝ) : ℝ × ℝ :=
  let d := (a * pt.1 + b * pt.2 + c) / (a^2 + b^2)
  (pt.1 - 2 * a * d, pt.2 - 2 * b * d)

-- Prove the symmetric curve C' for line with slope -3/4
theorem sym_curve_slope :
  ∀ M : ℝ × ℝ, reflect_over_line 3 4 1 (3, 0) = (3/5, -16/5) ↔ 
  (M.1 - 3/5)^2 + (M.2 + 16/5)^2 = 4 := sorry

end apollonian_circle_eq_sym_curve_vertical_sym_curve_slope_l336_336945


namespace relationship_abc_l336_336632

theorem relationship_abc (a b c : ℝ) 
  (h₁ : a = Real.log 0.5 / Real.log 2) 
  (h₂ : b = Real.sqrt 2) 
  (h₃ : c = 0.5 ^ 2) : 
  a < c ∧ c < b := by
  sorry

end relationship_abc_l336_336632


namespace squareable_numbers_l336_336871

def is_squareable (n : ℕ) : Prop :=
  ∃ (perm : ℕ → ℕ), (∀ i, 1 ≤ perm i ∧ perm i ≤ n) ∧ (∀ i, ∃ k, perm i + i = k * k)

theorem squareable_numbers : is_squareable 9 ∧ is_squareable 15 ∧ ¬ is_squareable 7 ∧ ¬ is_squareable 11 :=
by sorry

end squareable_numbers_l336_336871


namespace smallest_n_properties_l336_336129

noncomputable def smallest_n : ℕ :=
  3584

theorem smallest_n_properties :
  (∃ n : ℕ, n = smallest_n ∧
    (∃ a b : ℕ, n = 2^a * 5^b) ∧
    n % 7 = 0 ∧
    (∃ d : ℕ, 9 ∈ (n.digits 10))
  ) :=
by
  use smallest_n
  split
  {
    refl
  }
  split
  {
    use [9, 0]
    norm_num
  }
  split
  {
    norm_num
  }
  {
    use 9
    norm_num
  }

end smallest_n_properties_l336_336129


namespace typing_cost_equation_l336_336090

def typing_cost (x : ℝ) : ℝ :=
  200 * x + 80 * 3 + 20 * 6

theorem typing_cost_equation (x : ℝ) (h : typing_cost x = 1360) : x = 5 :=
by
  sorry

end typing_cost_equation_l336_336090


namespace angle_half_in_first_quadrant_l336_336281

theorem angle_half_in_first_quadrant (α : ℝ) (hα : 90 < α ∧ α < 180) : 0 < α / 2 ∧ α / 2 < 90 := 
sorry

end angle_half_in_first_quadrant_l336_336281


namespace find_EC_length_l336_336519
open Real EuclideanGeometry

-- Definitions
structure Trapezoid (A B C D : Point) : Prop :=
  (parallel : Line.through A B ∥ Line.through C D)

def three_times_length (A B C D : Point) [Trapezoid A B C D] : Prop :=
  dist A B = 3 * dist C D

def intersection_point (A C B D E : Point) : Prop :=
  Collinear A C E ∧ Collinear B D E

-- Main statement
theorem find_EC_length
  {A B C D E : Point}
  (h_trap : Trapezoid A B C D)
  (h_three_times : three_times_length A B C D)
  (h_intersection : intersection_point A C B D E)
  (h_AC_length : dist A C = 15) :
  dist E C = 15 / 4 :=
sorry

end find_EC_length_l336_336519


namespace least_subtracted_number_l336_336598

theorem least_subtracted_number (n : ℕ) (k : ℕ) : 
  let p := 2 * 3 * 5 * 7 * 11 in 
  n = 899830 → 
  k = n % p → 
  k = 2000 :=
begin
  intros,
  sorry
end

end least_subtracted_number_l336_336598


namespace mark_ate_fruit_first_four_days_l336_336784

theorem mark_ate_fruit_first_four_days (total_fruit : ℕ) (kept_for_next_week : ℕ) (brought_to_school : ℕ) :
  total_fruit = 10 → kept_for_next_week = 2 → brought_to_school = 3 → 
  (total_fruit - kept_for_next_week - brought_to_school) = 5 :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
end

end mark_ate_fruit_first_four_days_l336_336784


namespace greatest_six_digit_multiple_of_6_l336_336114

theorem greatest_six_digit_multiple_of_6 : 
  ∃ n : ℕ, 
    (∀ m : ℕ, 
      (m = 123567 ∨ m = 123756 ∨ m = 126357 ∨ m = 126735 ∨ m = 132567 ∨ m = 132756 ∨ 
       m = 135267 ∨ m = 135726 ∨ m = 152367 ∨ m = 152736 ∨ m = 153267 ∨ m = 153726 ∨ 
       m = 153726 ∨ m = 156237 ∨ m = 156273 ∨ m = 162357 ∨ m = 162735 ∨ m = 172356 ∨ 
       m = 172536 ∨ m = 173256 ∨ m = 173625 ∨ m = 175236 ∨ m = 175362 ∨ m = 213567 ∨ 
       m = 213756 ∨ m = 215367 ∨ m = 215736 ∨ m = 216357 ∨ m = 216735 ∨ m = 231567 ∨ 
       m = 231756 ∨ m = 236175 ∨ m = 236715 ∨ m = 237615 ∨ m = 237651 ∨ m = 251367 ∨ 
       m = 251736 ∨ m = 253167 ∨ m = 253617 ∨ m = 253671 ∨ m = 256137 ∨ m = 256173 ∨ 
       m = 257136 ∨ m = 261357 ∨ m = 261735 ∨ m = 275136 ∨ m = 275361 ∨ m = 276153 ∨ 
       m = 276513 ∨ m = 276531 ∨ m = 312567 ∨ m = 312756 ∨ m = 315267 ∨ m = 315726 ∨ 
       m = 351267 ∨ m = 351726 ∨ m = 352167 ∨ m = 352617 ∨ m = 352671 ∨ m = 357216 ∨ 
       m = 361257 ∨ m = 361725 ∨ m = 375216 ∨ m = 376215 ∨ m = 376251 ∨ m = 376521 ∨ 
       m = 376512 ∨ m = 376521 ∨ m = 376512 ∨ m = 351726 ∨ m = 312756 ∨ m = 315267 ∨ 
       m = 315726 ∨ m = 316257 ∨ m = 361725 ∨ m = 362157 ∨ m = 365127 ∨ m = 367215 ∨ 
       m = 367512 ∨ m = 367521 ∨ m = 372615 ∨ m = 372651 ∨ m = 376215 ∨ m = 376251 ∨ 
       m = 376521 ∨ m = 376512 ∨ m = 375216 ∨ m = 413567 ∨ m = 413756 ∨ m = 415367 ∨ 
       m = 415736 ∨ m = 416537 ∨ m = 416735 ∨ m = 417356 ∨ m = 431567 ∨ m = 432567 ∨ 
       m = 470113 ∨ m = 524672 ∨ m = 546372 ∨ m = 563217 ∨ m = 564172 ∨ m = 571263 ∨ 
       m = 573162 ∨ m = 576231 ∨ m = 576423 ∨ m = 714321 ∨ m = 716432 ∨ m = 723615 ∨ 
       m = 753216 ∨ m = 756123 ∨ m := 756132) ∧ n = 753216) :=
sorry

end greatest_six_digit_multiple_of_6_l336_336114


namespace concurrency_of_CP_DQ_AB_l336_336313

open Real

variables {O1 O2 O3 : Type} [circle O1] [circle O2] [circle O3]
variables (A B C D P Q : point)

-- Function and definitions used in the conditions
variables [h_inter_O1_O2 : O1 ∩ O2 = {A, B}]
variables [h_line_A_parallel_O1O2 : through(A) ∥ (O1 ∩ O2)]
variables [h_line_A_intersects : through(A) ∩ O1 = {C}]
variables [h_line_A_intersects' : through(A) ∩ O2 = {D}]
variables [h_circle_O3_diameter_CD : diameter O3 = CD]
variables [h_circle_O3_intersect_O1 : O3 ∩ O1 = {P}]
variables [h_circle_O3_intersect_O2 : O3 ∩ O2 = {Q}]

-- Statement to prove
theorem concurrency_of_CP_DQ_AB :
  concurrent (through C P) (through D Q) (through A B) :=
sorry

end concurrency_of_CP_DQ_AB_l336_336313


namespace exists_one_less_than_fraction_l336_336009

theorem exists_one_less_than_fraction (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) :
  (ab(1 - c)^2) ≤ (1 / 16) ∨ (bc(1 - a)^2) ≤ (1 / 16) ∨ (ca(1 - b)^2) ≤ (1 / 16) :=
sorry

end exists_one_less_than_fraction_l336_336009


namespace average_weight_of_class_l336_336474

/-
  Define the number of students and their average weights in sections A, B, C, and D.
-/

def num_students_A : ℕ := 60
def avg_weight_A : ℚ := 60
def num_students_B : ℕ := 70
def avg_weight_B : ℚ := 80
def num_students_C : ℕ := 50
def avg_weight_C : ℚ := 55
def num_students_D : ℕ := 65
def avg_weight_D : ℚ := 75

/-
  Define the total number of students and the total weight of the class
-/

def total_students : ℕ := num_students_A + num_students_B + num_students_C + num_students_D
def total_weight : ℚ :=
  (num_students_A * avg_weight_A) +
  (num_students_B * avg_weight_B) +
  (num_students_C * avg_weight_C) +
  (num_students_D * avg_weight_D)

/-
  Prove that the average weight of the entire class is approximately 68.67 kg.
-/

theorem average_weight_of_class :
  total_weight / total_students ≈ 68.67 :=
by
  sorry

end average_weight_of_class_l336_336474


namespace blanket_warmth_l336_336884

theorem blanket_warmth (blankets_added : ℕ) (total_warmth : ℕ) (half_blankets : ℕ) (correct_warmth : ℕ) :
  (blankets_added = 14 / 2) →
  (total_warmth = 21) →
  (blankets_added = half_blankets) →
  (correct_warmth = total_warmth / half_blankets) →
  (correct_warmth = 3) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end blanket_warmth_l336_336884


namespace sum_of_roots_tan_quadratic_l336_336607

theorem sum_of_roots_tan_quadratic :
  (∑ x in filter (λ x, 0 ≤ x ∧ x ≤ 2 * real.pi)
            (λ x => x ∈ {x | tan x - 4 - sqrt 14 = 0 ∨ tan x - 4 + sqrt 14 = 0}), x) = 3 * real.pi :=
sorry

end sum_of_roots_tan_quadratic_l336_336607


namespace binom_12_6_eq_924_l336_336966

theorem binom_12_6_eq_924 : Nat.choose 12 6 = 924 := 
by
  sorry

end binom_12_6_eq_924_l336_336966


namespace radius_of_incircle_l336_336124

noncomputable def radius_incircle (a b c h : ℕ) : ℕ :=
let s := (a + b + c) / 2 in
let A := 1 / 2 * b * h in
A / s

theorem radius_of_incircle {a b c h r : ℕ} 
  (h_sides : a = 5 ∧ b = 12 ∧ c = 13) 
  (h_altitude : h = 10) 
  (h_result : r = 4) :
  radius_incircle a b c h = r :=
by
  sorry

end radius_of_incircle_l336_336124


namespace Amelia_sell_JetBars_l336_336194

theorem Amelia_sell_JetBars (M : ℕ) (h : 2 * M - 16 = 74) : M = 45 := by
  sorry

end Amelia_sell_JetBars_l336_336194


namespace value_of_expression_l336_336703

-- Defining the given conditions as Lean definitions
def x : ℚ := 2 / 3
def y : ℚ := 5 / 2

-- The theorem statement to prove that the given expression equals the correct answer
theorem value_of_expression : (1 / 3) * x^7 * y^6 = 125 / 261 :=
by
  sorry

end value_of_expression_l336_336703


namespace smallest_n_l336_336128

theorem smallest_n :
∃ (n : ℕ), (0 < n) ∧ (∃ k1 : ℕ, 5 * n = k1 ^ 2) ∧ (∃ k2 : ℕ, 7 * n = k2 ^ 3) ∧ n = 1225 :=
sorry

end smallest_n_l336_336128


namespace triangle_probability_l336_336422

noncomputable def stick_lengths : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

def valid_triangles (lst: List ℕ) : ℕ :=
  -- Define a function that counts valid triangle combinations
  -- where the sum of the two smaller sides is greater than the third side.
  (lst.combination 3).count (λ triplet,
    let sorted_triplet := tripletSorted triplet in
    sorted_triplet.head! + sorted_triplet.tail.head! > sorted_triplet.last!)

noncomputable def total_combinations (lst: List ℕ) : ℕ :=
  lst.combination 3 |>.length

theorem triangle_probability :
  let valid_count := valid_triangles stick_lengths in
  let total_count := total_combinations stick_lengths in
  (valid_count, total_count) = (25, 84) ∧ (valid_count : ℚ) / total_count = 25 / 84 :=
by
  sorry

end triangle_probability_l336_336422


namespace angle_measure_of_PRS_l336_336719

theorem angle_measure_of_PRS
  (triangle_PQR : Type)
  [NonEmpty triangle_PQR]
  (P Q R S : triangle_PQR)
  (angle_Q : ℝ)
  (angle_QRS : ℝ)
  (h1 : angle_Q = 75)
  (h2 : angle_QRS = 140)
  (is_triangle : ∀ P Q R, angle_P + angle_Q + ∠ R = 180)
  (is_straight_line : ∀ P S, ∠ P S = 180) :
  ∠ PRS = 70 := 
  sorry

end angle_measure_of_PRS_l336_336719


namespace det_B4_eq_16_l336_336697

theorem det_B4_eq_16 (B : Matrix ℝ ℕ ℕ) (h : det B = -2) : det (B^4) = 16 :=
sorry

end det_B4_eq_16_l336_336697


namespace gasoline_price_growth_l336_336997

variable (x : ℝ)

/-- The average monthly growth rate of the price of 92-octane gasoline. -/
theorem gasoline_price_growth (h₁ : 7.5 > 0) (h₂ : 1 + x > 0) (h₃ : 7.5 * (1 + x)^2 = 8.4) : 7.5 * (1 + x)^2 = 8.4 :=
by {
  exact h₃,
  sorry -- proof omitted
}

end gasoline_price_growth_l336_336997


namespace xiao_ming_climb_stairs_8_l336_336788

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | (n+2) => fibonacci n + fibonacci (n + 1)

theorem xiao_ming_climb_stairs_8 :
  fibonacci 8 = 34 :=
sorry

end xiao_ming_climb_stairs_8_l336_336788


namespace projection_of_a_on_b_l336_336260

variables {a b : V} [NormedSpace ℝ V]
variables (norm_a : ∥a∥ = 5) (norm_b : ∥b∥ = 3) (dot_ab : inner a b = -12)

theorem projection_of_a_on_b : (inner a b) / (∥b∥) = -4 :=
by 
  sorry

end projection_of_a_on_b_l336_336260


namespace binomial_12_6_eq_924_l336_336978

theorem binomial_12_6_eq_924 : nat.choose 12 6 = 924 := sorry

end binomial_12_6_eq_924_l336_336978


namespace eccentricity_of_ellipse_l336_336917

-- Define the problem parameters
variable (a b : ℝ) (c : ℝ)
variable (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
variable (h₄ : c = sqrt (a^2 - b^2))
variable (h_line : ∀ x y, x / c + y / b = 1 → 
                         (x = a ∨ x = -a) ∧ 
                         (y = b / sqrt (a^2 - b^2) ∨ y = -b / sqrt (a^2 - b^2)))
variable (h_distance : (1 / sqrt (1 / c^2 + 1 / b^2)) = b / 2)

-- Statement to prove the eccentricity
theorem eccentricity_of_ellipse : c / a = 1 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l336_336917


namespace solve_triangle_compute_triangle_x_l336_336574

noncomputable def triangle (a b : ℝ) : ℝ := if b = a then 2 else (a / b) * b

lemma triangle_property_1 {a b c : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  triangle a (triangle b c) = (a / b) * c :=
by
  unfold triangle
  split_ifs
  { sorry }
  { sorry }

lemma triangle_property_2 {a : ℝ} (ha : a ≠ 0) : triangle a a = 2 :=
by
  unfold triangle
  split_ifs
  { exact rfl }
  { exfalso, apply h, refl }

theorem solve_triangle (x : ℝ) (hx : x ≠ 0) : triangle 8 (triangle 4 x) = 48 → x = 1/6 :=
by
  intro h
  unfold triangle at *
  split_ifs at *
  { sorry }
  { sorry }
  { sorry }

theorem compute_triangle_x (x : ℝ) (hx : x ≠ 0) : x = 1/6 → triangle 2 x = 2 :=
by
  intro h
  unfold triangle
  split_ifs
  { sorry }
  { sorry }

end solve_triangle_compute_triangle_x_l336_336574


namespace sin_intersection_value_l336_336835

theorem sin_intersection_value :
  ∃ (x₀ : ℝ), 0 < x₀ ∧ x₀ < π / 2 ∧ 3 * cos x₀ = 8 * tan x₀ ∧ sin x₀ = 1 / 3 :=
by
  sorry

end sin_intersection_value_l336_336835


namespace rational_solutions_infinite_l336_336807

theorem rational_solutions_infinite (a : ℚ) : ∃ (S : Set (ℚ × ℚ)), S.Infinite ∧ ∀ p ∈ S, (y = p.2) = (x = p.1) ∧ y = real.sqrt (x^2 + a) :=
by sorry

end rational_solutions_infinite_l336_336807


namespace solution_l336_336141

noncomputable def proof_problem : Prop :=
  ∃ (α β : Circle) (A B : ℕ → Point),
    (β ⊂ α) ∧
    (∀ n, A n ∈ α) ∧
    (∀ n, B n ∈ α) ∧
    (∀ n, tangent (A n) (A (n + 1)) β) ∧
    (∀ n, tangent (B n) (B (n + 1)) β) ∧ 
    (∃ γ, ∀ n, tangent (A n) (B n) γ ∧ 
                  (γ.center ∈ line α.center β.center))

theorem solution : proof_problem :=
sorry

end solution_l336_336141


namespace total_acorns_l336_336806

theorem total_acorns (s_a : ℕ) (s_b : ℕ) (d : ℕ)
  (h1 : s_a = 7)
  (h2 : s_b = 5 * s_a)
  (h3 : s_b + 3 = d) :
  s_a + s_b + d = 80 :=
by
  sorry

end total_acorns_l336_336806


namespace markup_correct_l336_336089

theorem markup_correct (purchase_price : ℝ) (overhead_percentage : ℝ) (net_profit : ℝ) :
  purchase_price = 48 → overhead_percentage = 0.15 → net_profit = 12 →
  (purchase_price * (1 + overhead_percentage) + net_profit - purchase_price) = 19.2 :=
by
  intros
  sorry

end markup_correct_l336_336089


namespace permutations_red_l336_336322

theorem permutations_red (n : ℕ) (h : (3 * n)! / ((n!) * (n!) * (n!)) = 6) : n = 1 :=
by
  sorry

end permutations_red_l336_336322


namespace sum_integral_ineq_l336_336042

open Real

theorem sum_integral_ineq (n : ℕ) : 
  (∫ x in 0..(π / 2), ∑ k in Finset.range (n + 1), (sin (k.succ * x) / k.succ)^2) < (61 / 144) * π :=
by
  sorry

end sum_integral_ineq_l336_336042


namespace friend_made_mistakes_and_needs_security_measures_l336_336889

namespace OnlineSecurity

-- Conditions definitions
def email_address := "aliexpress@best_prices.net"
def website_url := "aliexpres__best_prices.net/shop"
def personal_info := {last_name : String, first_name : String, card_number : String, cvv_code : String, address : String}
def verification_block := true
def sms_password_entered := true
def purchase_confirmed := true

-- Mistakes made by the friend
def mistakes_made (trusted_email : Bool) (trusted_url : Bool) (entered_personal_info : Bool) 
                  (believed_offer : Bool) (verified_legitimacy : Bool) : Prop := 
trusted_email ∧ trusted_url ∧ entered_personal_info ∧ believed_offer ∧ ¬ verified_legitimacy

-- Additional security measures
def additional_security_measures (secure_network : Bool) (antivirus : Bool) 
                                 (updated_apps : Bool) (check_https : Bool) 
                                 (strong_password : Bool) (two_factor_auth : Bool) 
                                 (recognize_bank_protocols : Bool) : Prop :=
secure_network ∧ antivirus ∧ updated_apps ∧ check_https ∧ strong_password ∧ two_factor_auth ∧ recognize_bank_protocols

-- Statement for the mathematically equivalent proof problem in Lean
theorem friend_made_mistakes_and_needs_security_measures :
  let trusted_email := true,
      trusted_url := true,
      entered_personal_info := true,
      believed_offer := true,
      verified_legitimacy := false,
      secure_network := true,
      antivirus := true,
      updated_apps := true,
      check_https := true,
      strong_password := true,
      two_factor_auth := true,
      recognize_bank_protocols := true in
  mistakes_made trusted_email trusted_url entered_personal_info believed_offer verified_legitimacy ∧
  additional_security_measures secure_network antivirus updated_apps check_https strong_password two_factor_auth recognize_bank_protocols :=
sorry

end OnlineSecurity

end friend_made_mistakes_and_needs_security_measures_l336_336889


namespace smallest_n_correct_l336_336994

noncomputable def smallest_n : ℕ :=
  if h : ∃ n ≥ 3, ∀ (z : ℕ → ℂ), (∀ i, |z i| = 1) ∧ (z 0 = 1) ∧ (Finset.sum (Finset.range n) z = 0)
    → ∃ k : ℕ, ∀ i, z i = exp (2 * real.pi * complex.I * ↑k / ↑n * ↑i)
  then dite (h : (∃ n, n ≥ 3) ∧ (∀ i, |z i| = 1) ∧ (z 0 = 1) ∧ (Finset.sum (Finset.range (classical.some h)) z = 0) 
    → ∃ k : ℕ, ∀ i, z i = exp (2 * real.pi * complex.I * ↑k / (classical.some h) * ↑i))
    (λ n, classical.some h)
    (λ _, 3)
  else 3 

theorem smallest_n_correct : smallest_n = 3 := 
  by
  sorry

end smallest_n_correct_l336_336994


namespace quadratic_root_sq_condition_l336_336045

noncomputable def necessary_sufficient_condition_quadratic_root_sq (p q : ℝ) : Prop :=
  p^2 - 4*q ≥ 0 ∧ ∃ x1, x1 + x1^2 = -p ∧ x1^3 = q → p = - (real.cbrt q + real.cbrt (q^2))

theorem quadratic_root_sq_condition (p q : ℝ) : 
  necessary_sufficient_condition_quadratic_root_sq p q :=
sorry

end quadratic_root_sq_condition_l336_336045


namespace distance_is_correct_l336_336754

noncomputable def distance_between_points 
  (p q m a b c : ℝ) : ℝ :=
|q - p| * Real.sqrt(1 + (a * q^2 + b * q + c - m * p - k)^2 / (q - p)^2)

theorem distance_is_correct 
  (p q m a b c : ℝ) : 
  let x1 := p,
      y1 := m * p + k,
      x2 := q,
      y2 := a * q^2 + b * q + c 
  in distance_between_points p q m a b c = 
       |q - p| * Real.sqrt(1 + (a * q^2 + b * q + c - m * p - k)^2 / (q - p)^2) :=
sorry

end distance_is_correct_l336_336754


namespace solve_exp_eq_l336_336575

theorem solve_exp_eq (x : ℝ) : 3^(2*x) - 9 * 3^x + 6 = 0 ↔ x = Real.log 4 / Real.log 3 ∨ x = Real.log 1.5 / Real.log 3 := 
by
  sorry

end solve_exp_eq_l336_336575


namespace blocks_to_store_l336_336200

theorem blocks_to_store
  (T : ℕ) (S : ℕ)
  (hT : T = 25)
  (h_total_walk : S + 6 + 8 = T) :
  S = 11 :=
by
  sorry

end blocks_to_store_l336_336200


namespace number_of_functions_l336_336602

-- Define the quadratic function g(x) of the form ax^2 + bx + c
def g (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

-- Define the condition g(x) * g(-x) = g(x^3)
def satisfies_condition (a b c : ℝ) : Prop :=
  (g a b c = λ x, a * x^2 + b * x + c) → 
  (g a b c * g a b c (-x)) = (g a b c (x^3))

-- State that the number of such functions is 6
theorem number_of_functions : 
  (∀ a b c : ℝ, satisfies_condition a b c) → 
  {p : ℕ // ∃ a b c : ℝ, satisfies_condition a b c ∧ p = 6} :=
by
  sorry

end number_of_functions_l336_336602


namespace binomial_12_6_eq_1848_l336_336971

theorem binomial_12_6_eq_1848 : (Nat.choose 12 6) = 1848 :=
  sorry

end binomial_12_6_eq_1848_l336_336971


namespace chris_average_price_l336_336958

def total_cost (dvd_count dvd_price br_count br_price : ℕ) : ℕ :=
  dvd_count * dvd_price + br_count * br_price

def total_movies (dvd_count br_count : ℕ) : ℕ :=
  dvd_count + br_count

def average_price (total_cost total_movies : ℕ) : ℕ :=
  total_cost / total_movies

theorem chris_average_price :
  let dvd_count := 8
  let dvd_price := 12
  let br_count := 4
  let br_price := 18
  total_movies dvd_count br_count = 12 →
  total_cost dvd_count dvd_price br_count br_price = 168 →
  average_price (total_cost dvd_count dvd_price br_count br_price) (total_movies dvd_count br_count) = 14 :=
by 
  intros dvd_count dvd_price br_count br_price
  intros h1 h2,
  have h_total_cost : total_cost dvd_count dvd_price br_count br_price = 168 := h2,
  have h_total_movies : total_movies dvd_count br_count = 12 := h1,
  exact (show average_price 168 12 = 14 from rfl)

end chris_average_price_l336_336958


namespace sum_of_norms_squared_l336_336014

open Real

variables (a b : ℝ × ℝ)
def midpoint (p q : ℝ × ℝ) : ℝ × ℝ := ((p.1 + q.1) / 2, (p.2 + q.2) / 2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def norm_sq (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2

theorem sum_of_norms_squared :
  midpoint a b = (4, 5) ∧ dot_product a b = 8 → norm_sq a + norm_sq b = 148 :=
by
  intro h
  sorry

end sum_of_norms_squared_l336_336014


namespace definite_integral_value_l336_336291

noncomputable def piecewiseFunction (x : ℝ) : ℝ :=
  if x ≤ 2 then -x + 2 else real.sqrt (1 - (x - 3)^2)

theorem definite_integral_value :
  ∫ x in (1/2)..4, piecewiseFunction x = (9 + 4 * real.pi) / 8 := by
  sorry

end definite_integral_value_l336_336291


namespace sum_of_coordinates_l336_336803

theorem sum_of_coordinates (y : ℝ) : 
  let C := (3, y)
  let D := (y, 3)
  C.1 + C.2 + D.1 + D.2 = 2 * y + 6 :=
by
  -- Definitions
  let C := (3, y)
  let D := (y, 3)
  -- Sum of coordinates
  have h : C.1 + C.2 + D.1 + D.2 = 3 + y + y + 3 := rfl
  rw [add_assoc, add_comm y, add_assoc, add_assoc, add_comm y, add_assoc]
  sorry

end sum_of_coordinates_l336_336803


namespace visited_neither_l336_336720

theorem visited_neither (total : ℕ) (visited_Iceland : ℕ) (visited_Norway : ℕ) (visited_both : ℕ) (h_total : total = 50)
  (h_Iceland : visited_Iceland = 25) (h_Norway : visited_Norway = 23) (h_both : visited_both = 21) : 
  total - (visited_Iceland + visited_Norway - visited_both) = 23 :=
by
  rw [h_total, h_Iceland, h_Norway, h_both]
  sorry

end visited_neither_l336_336720


namespace sin_A_eq_length_BC_eq_area_triangle_eq_l336_336736

-- Definitions and Conditions
def angle_B := 45
def side_b := Real.sqrt 10
def cos_C := 2 * Real.sqrt 5 / 5

-- Proof Targets
theorem sin_A_eq : ∃ (A : ℝ), A = 3 * Real.sqrt 10 / 10 :=
by sorry

theorem length_BC_eq : ∃ (BC : ℝ), BC = 3 * Real.sqrt 2 :=
by sorry

theorem area_triangle_eq : ∃ (area : ℝ), area = 3 :=
by sorry

end sin_A_eq_length_BC_eq_area_triangle_eq_l336_336736


namespace solve_quadratic_l336_336443

theorem solve_quadratic (c d : ℝ) 
  (h1 : c * c - 6 * c + 14 = 31)
  (h2 : d * d - 6 * d + 14 = 31)
  (h3 : c ≥ d) : 
  c + 2 * d = 9 - real.sqrt 26 := 
sorry

end solve_quadratic_l336_336443


namespace carlson_handkerchief_usage_l336_336956

def problem_statement : Prop :=
  let handkerchief_area := 25 * 25 -- Area in cm²
  let total_fabric_area := 3 * 10000 -- Total fabric area in cm²
  let days := 8
  let total_handkerchiefs := total_fabric_area / handkerchief_area
  let handkerchiefs_per_day := total_handkerchiefs / days
  handkerchiefs_per_day = 6

theorem carlson_handkerchief_usage : problem_statement := by
  sorry

end carlson_handkerchief_usage_l336_336956


namespace num_terms_before_5_l336_336692

-- Define the arithmetic sequence as a function of n
def arith_seq (a d : ℤ) (n : ℕ) : ℤ := a + d * (n - 1)

-- Definitions matching the conditions
def first_term : ℤ := 105
def common_difference : ℤ := -5

-- The number of terms before the number 5 appears in the sequence
theorem num_terms_before_5 : (n : ℕ) (arith_seq first_term common_difference n = 5) -> n - 1 = 20 :=
  by
  sorry

end num_terms_before_5_l336_336692


namespace correct_equation_l336_336996

variables (x : ℝ)
noncomputable def growth_rate_equation : Prop := 7.5 * (1 + x)^2 = 8.4

theorem correct_equation
  (price_june : 7.5)
  (price_august : 8.4)
  (average_growth : x) :
  growth_rate_equation x :=
by 
  sorry

end correct_equation_l336_336996


namespace last_digit_11_power_11_last_digit_9_power_9_last_digit_9219_power_9219_last_digit_2014_power_2014_l336_336870

-- Definition of function to calculate the last digit of a number
def last_digit (n : ℕ) : ℕ :=
  n % 10

-- Proof statements
theorem last_digit_11_power_11 : last_digit (11 ^ 11) = 1 := sorry

theorem last_digit_9_power_9 : last_digit (9 ^ 9) = 9 := sorry

theorem last_digit_9219_power_9219 : last_digit (9219 ^ 9219) = 9 := sorry

theorem last_digit_2014_power_2014 : last_digit (2014 ^ 2014) = 6 := sorry

end last_digit_11_power_11_last_digit_9_power_9_last_digit_9219_power_9219_last_digit_2014_power_2014_l336_336870


namespace product_of_rational_solutions_eq_twelve_l336_336476

theorem product_of_rational_solutions_eq_twelve :
  ∃ c1 c2 : ℕ, (c1 > 0) ∧ (c2 > 0) ∧ 
               (∀ x : ℚ, ∃ (a b : ℤ), 5 * x^2 + 11 * x + c1 = 0 → 
                             ∃ (d : ℤ), b^2 - 4 * a * c1 = d^2) ∧
               (∀ x : ℚ, ∃ (a b : ℤ), 5 * x^2 + 11 * x + c2 = 0 → 
                             ∃ (d : ℤ), b^2 - 4 * a * c2 = d^2) ∧
               c1 * c2 = 12 := sorry

end product_of_rational_solutions_eq_twelve_l336_336476


namespace denom_excess_over_num_l336_336395

-- Define G as given in the condition
def G : ℚ := 871 / 999

-- Statement that the difference between the denominator and numerator is 128
theorem denom_excess_over_num : (G.denom - G.num) = 128 := 
by
  -- Using the definition and properties of rational numbers
  have h1 : G = 871 / 999 := rfl,
  -- Substitute G
  rw h1,
  -- Simplify
  have h2 : (871 / 999).num = 871 := by sorry,
  have h3 : (871 / 999).denom = 999 := by sorry,
  -- Final calculation
  calc
    999 - 871 = 128 : by sorry

end denom_excess_over_num_l336_336395


namespace runners_catch_up_again_l336_336865

noncomputable def catch_up_again (V1 V2 : ℝ) (initial_laps_second : ℝ) : ℝ :=
  let new_V2 := 2 * V2
  in if V1 > new_V2 then 
      (initial_laps_second + 2)
     else sorry

theorem runners_catch_up_again :
  ∀ V1 V2 : ℝ, 
    (V1 = 3 * V2) → 
    catch_up_again V1 V2 (1 / 2) = 2.5 :=
begin
  intros,
  rw [catch_up_again, 
      if_pos],
  { norm_num },
  { apply lt_of_sub_pos,
    norm_num,
    linarith }
end

end runners_catch_up_again_l336_336865


namespace coin_flip_possible_l336_336102

-- Condition definitions
inductive CoinState
| R  -- tails up
| O  -- heads up
deriving DecidableEq

def initial_state : List CoinState := [CoinState.R, CoinState.R, CoinState.O, CoinState.R, CoinState.R]

-- Define the flip operation
def flip (l : List CoinState) (i : Nat) : List CoinState :=
  if i + 2 < l.length then
    l.take i ++ l.drop i (l.get! i).reverse ++ l.drop (i + 3)
  else
    l

-- Target state
def target_state : List CoinState := [CoinState.O, CoinState.O, CoinState.O, CoinState.O, CoinState.O]

-- The problem statement: Is it possible to achieve the target_state from initial_state using a series of flips?
theorem coin_flip_possible : ∃ flips : List Nat, List.foldl flip initial_state flips = target_state :=
  sorry

end coin_flip_possible_l336_336102


namespace lambda_range_l336_336678

theorem lambda_range {λ : ℝ} :
  (∀ (m : ℝ) (n : ℝ), n > 0 → (m - n)^2 + (m - real.log n + λ)^2 ≥ 2) →
  (λ ≥ 2 ∨ λ ≤ -2) :=
by
  sorry

end lambda_range_l336_336678


namespace mark_ate_fruit_first_four_days_l336_336783

theorem mark_ate_fruit_first_four_days (total_fruit : ℕ) (kept_for_next_week : ℕ) (brought_to_school : ℕ) :
  total_fruit = 10 → kept_for_next_week = 2 → brought_to_school = 3 → 
  (total_fruit - kept_for_next_week - brought_to_school) = 5 :=
begin
  intros h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
end

end mark_ate_fruit_first_four_days_l336_336783


namespace prove_sum_l336_336511

theorem prove_sum (a b : ℝ) (h1 : a * (a - 4) = 12) (h2 : b * (b - 4) = 12) (h3 : a ≠ b) : a + b = 4 := by
  sorry

end prove_sum_l336_336511


namespace digits_count_l336_336073

theorem digits_count (n : ℕ) (avg_total_digits : n * 18) (avg_4_digits : 4 * 8) (avg_5_digits : 5 * 26) : n = 9 :=
by 
  have h1 : n * 18 = 32 + 130 := by 
    have : avg_total_digits = 18 * n := by sorry
    have : avg_4_digits = 4 * 8 := by sorry
    have : avg_5_digits = 5 * 26 := by sorry
    sorry -- complete the proof steps to establish the equality
  have h2 : n * 18 = 162 := by sorry
  have h3 : n = 9 := by linarith
  exact h3

end digits_count_l336_336073


namespace sin_A_value_side_values_l336_336737

namespace TriangleProof

variables {A B C : ℝ}
variables {a b c : ℝ} (h₁ : a = 2) (h₂ : cos B = 3 / 5) (h₃ : sin B = 4 / 5)

-- Problem 1
theorem sin_A_value (h₄ : b = 4) : sin A = 2 / 5 := by
  sorry

-- Problem 2
theorem side_values {S : ℝ} (h₄ : S = 4) : b = Real.sqrt 17 ∧ c = 5 := by
  sorry

end TriangleProof

end sin_A_value_side_values_l336_336737


namespace train_pass_jogger_in_40_seconds_l336_336174

noncomputable def time_to_pass_jogger (jogger_speed_kmh : ℝ) (train_speed_kmh : ℝ) (initial_distance_m : ℝ) (train_length_m : ℝ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh - jogger_speed_kmh
  let relative_speed_ms := relative_speed_kmh * (5 / 18)  -- Conversion from km/hr to m/s
  let total_distance_m := initial_distance_m + train_length_m
  total_distance_m / relative_speed_ms

theorem train_pass_jogger_in_40_seconds :
  time_to_pass_jogger 9 45 280 120 = 40 := by
  sorry

end train_pass_jogger_in_40_seconds_l336_336174


namespace equivalent_proof_problem_l336_336137

def original_function (x : ℝ) : ℝ := 2 * x^2 - x + 5

def shifted_right_function (x : ℝ) : ℝ := 2 * (x - 7)^2 - (x - 7) + 5

noncomputable def final_function (x : ℝ) : ℝ := shifted_right_function x + 3

theorem equivalent_proof_problem : 
  (∃ a b c : ℝ, final_function = λ x, a * x^2 + b * x + c ∧ a + b + c = 86) :=
sorry

end equivalent_proof_problem_l336_336137


namespace f_2014_odd_f_2014_not_even_l336_336214

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => 1 / x
| (n + 1), x => 1 / (x + f n x)

theorem f_2014_odd :
  ∀ x : ℝ, f 2014 x = - f 2014 (-x) :=
sorry

theorem f_2014_not_even :
  ∃ x : ℝ, f 2014 x ≠ f 2014 (-x) :=
sorry

end f_2014_odd_f_2014_not_even_l336_336214


namespace simplify_expression_l336_336062

theorem simplify_expression (w x : ℤ) :
  3 * w + 6 * w + 9 * w + 12 * w + 15 * w + 20 * x + 24 = 45 * w + 20 * x + 24 :=
by sorry

end simplify_expression_l336_336062


namespace fraction_remaining_distance_l336_336999

theorem fraction_remaining_distance
  (total_distance : ℕ)
  (first_stop_fraction : ℚ)
  (remaining_distance_after_second_stop : ℕ)
  (fraction_between_stops : ℚ) :
  total_distance = 280 →
  first_stop_fraction = 1/2 →
  remaining_distance_after_second_stop = 105 →
  (fraction_between_stops * (total_distance - (first_stop_fraction * total_distance)) + remaining_distance_after_second_stop = (total_distance - (first_stop_fraction * total_distance))) →
  fraction_between_stops = 1/4 :=
by
  sorry

end fraction_remaining_distance_l336_336999


namespace digit_families_count_l336_336867

theorem digit_families_count :
  {p : ℕ × ℕ // let (a, b) := p in a + b = 9 ∧ a ≤ 9 ∧ b ≤ 9}.to_finset.card = 8 :=
by
  sorry

end digit_families_count_l336_336867


namespace cos_value_l336_336699

theorem cos_value (α : ℝ) (h : sin (π / 6 - α) = 1/4) : cos (2 * α - π / 3) = 7/8 := 
by 
  sorry

end cos_value_l336_336699


namespace carol_packs_l336_336564

theorem carol_packs (n_invites n_per_pack : ℕ) (h1 : n_invites = 12) (h2 : n_per_pack = 4) : n_invites / n_per_pack = 3 :=
by
  sorry

end carol_packs_l336_336564


namespace solve_expression_l336_336233

variable (x : ℚ)

theorem solve_expression (h : x = 1 / 3) : 
  (let a := (x + 2) / (x - 2) in 
  a^2) = 961 / 1369 := 
by
  sorry

end solve_expression_l336_336233


namespace transformations_count_l336_336758

def transformations := {rotation := (λ (p : ℝ × ℝ), (-p.2, p.1)),
                        reflection_x := (λ (p : ℝ × ℝ), (p.1, -p.2)),
                        reflection_y := (λ (p : ℝ × ℝ), (-p.1, p.2)),
                        translation := (λ (p : ℝ × ℝ), (p.1 + 6, p.2 + 2))}

def apply_transformations (seq : list (ℝ × ℝ → ℝ × ℝ)) (p : ℝ × ℝ)  :=
seq.foldl (λ acc f, f acc) p 

noncomputable def count_valid_sequences : ℕ :=
(finset.univ : finset (vector (ℝ × ℝ → ℝ × ℝ) 4)).filter (λ seq,
  let transformed_rectangle := set.image (apply_transformations seq.val) (set.of (λ pt, pt ∈ {(0,0), (6,0), (6,2), (0,2)})) in
  transformed_rectangle = {(0,0), (6,0), (6,2), (0,2)}).card

theorem transformations_count : count_valid_sequences = 3 :=
sorry

end transformations_count_l336_336758


namespace min_omega_value_l336_336446

noncomputable theory
open Real

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem min_omega_value
  (ω : ℝ)
  (hω : ω > 0)
  (f : ℝ → ℝ := λ x, Real.cos (ω * x + π / 4))
  (g : ℝ → ℝ := λ x, Real.cos (ω * (x + π / 3) + π / 4))
  (h_odd_g : is_odd_function g) :
  ω = 3 / 4 :=
sorry

end min_omega_value_l336_336446


namespace min_socks_for_15_pairs_l336_336533

-- Define the conditions
def num_red_socks : ℕ := 120
def num_green_socks : ℕ := 100
def num_blue_socks : ℕ := 80
def num_black_socks : ℕ := 60

-- Define a function that checks pairs
def pairs_in_selection (num_socks : ℕ) : ℕ := 15

-- Prove the minimum number of socks needed to guarantee at least 15 pairs
theorem min_socks_for_15_pairs : ∀ (socks_required : ℕ), 
  (socks_required >= 33) → (pairs_in_selection socks_required >= 15) :=
begin
  intros socks_required h_ge_33,
  sorry
end

end min_socks_for_15_pairs_l336_336533


namespace tangent_inequality_l336_336016

variables {n : ℕ} {a_i : fin n → ℝ} {α_i : fin n → ℝ} {x x0 a α : ℝ}
variables (h1 : ∀ i, 0 < a_i i) (h2 : ∀ i, 0 < α_i i) (h3 : ∃ (x : ℝ) (hx : 0 < x), x = x0)
variables (h4 : ∀ i j, i ≠ j → α_i i ≠ α_i j) (h5 : ∀ x, x > 0 → ∑ i, a_i i * x ^ α_i i = (a * x ^ α))

noncomputable def f (x : ℝ) := ∑ i, a_i i * x ^ α_i i

theorem tangent_inequality (h : ∀ x > 0, f x ≥ a * x ^ α ∧ (f x = a * x ^ α ↔ x = x0)) : ∀ x > 0, f x ≥ a * x ^ α ∧ (f x = a * x ^ α ↔ x = x0) := 
sorry

end tangent_inequality_l336_336016


namespace no_carrying_pairs_correct_l336_336614

noncomputable def count_no_carrying_pairs : ℕ :=
  let pairs := (1500 : ℕ, 2500 : ℕ)
  (1550 : ℕ)  -- correct answer

theorem no_carrying_pairs_correct :
  ∃ count : ℕ, count = count_no_carrying_pairs :=
sorry

end no_carrying_pairs_correct_l336_336614


namespace sum_ineq_l336_336458

theorem sum_ineq (a : Fin 1980 → ℝ) 
  (h_bounds : ∀ i, 1 - 1/1980 ≤ a i ∧ a i ≤ 1 + 1/1980) :
  ((Finset.univ.sum a) * (Finset.univ.sum (λ i, 1 / a i))) ≤ (1980^4) / (1980^2 - 1) := 
by
  sorry

end sum_ineq_l336_336458


namespace boundary_length_correct_l336_336930

noncomputable def boundary_length : Real :=
  let side := Real.sqrt 64
  let segment := side / 4
  let straight_segments := segment * 4
  let circle_length := Real.pi * 2 * segment * 2  -- 2 full circles
  let total_length := straight_segments + circle_length
  total_length

theorem boundary_length_correct : (Real.floor (boundary_length * 10) / 10 = 33.1) :=
by
  sorry

end boundary_length_correct_l336_336930


namespace Dacid_weighted_average_l336_336572

noncomputable def DacidMarks := 86 * 3 + 85 * 4 + 92 * 4 + 87 * 3 + 95 * 3 + 89 * 2 + 75 * 1
noncomputable def TotalCreditHours := 3 + 4 + 4 + 3 + 3 + 2 + 1
noncomputable def WeightedAverageMarks := (DacidMarks : ℝ) / (TotalCreditHours : ℝ)

theorem Dacid_weighted_average :
  WeightedAverageMarks = 88.25 :=
sorry

end Dacid_weighted_average_l336_336572


namespace geometric_seq_nine_l336_336641

theorem geometric_seq_nine (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 - a 2 = 8)
  (h2 : a 3 - a 4 = 2) (h3 : (∏ i in Finset.range n, a i.succ) = 1) : 
  n = 9 := 
sorry

end geometric_seq_nine_l336_336641


namespace area_of_tangency_triangle_l336_336103

theorem area_of_tangency_triangle (r1 r2 r3 : ℝ) 
  (h1 : r1 = 2) (h2 : r2 = 3) (h3 : r3 = 4)
  (externally_tangent : ∀ (c1 c2 : ℝ), c1 ≠ c2 → c1 + c2 = 5 ∨ c1 + c2 = 7 ∨ c1 + c2 = 6) :
  area_tangency_triangle(r1, r2, r3) = 8 / 3 := 
begin
  sorry
end

end area_of_tangency_triangle_l336_336103


namespace smallest_c_for_polynomial_l336_336460

theorem smallest_c_for_polynomial :
  ∃ r1 r2 r3 : ℕ, (r1 * r2 * r3 = 2310) ∧ (r1 + r2 + r3 = 52) := sorry

end smallest_c_for_polynomial_l336_336460


namespace compute_value_of_expression_l336_336405

theorem compute_value_of_expression (p q : ℝ) (hpq : 3 * p^2 - 5 * p - 8 = 0) (hq : 3 * q^2 - 5 * q - 8 = 0) (hneq : p ≠ q) :
  3 * (p^2 - q^2) / (p - q) = 5 :=
by
  have hpq_sum : p + q = 5 / 3 := sorry
  exact sorry

end compute_value_of_expression_l336_336405


namespace inequality_solution_l336_336224

theorem inequality_solution (x : ℝ) : 3 * x^2 - x > 9 ↔ x < -3 ∨ x > 1 := by
  sorry

end inequality_solution_l336_336224


namespace largest_sum_fraction_l336_336566

theorem largest_sum_fraction :
  max (max (max (max ((1/3) + (1/2)) ((1/3) + (1/4))) ((1/3) + (1/5))) ((1/3) + (1/7))) ((1/3) + (1/9)) = 5/6 :=
by
  sorry

end largest_sum_fraction_l336_336566


namespace polynomial_factor_coefficients_l336_336221

theorem polynomial_factor_coefficients :
  ∃ p q : ℚ, 
    (∀ x : ℚ, (p * x^4 + q * x^3 + 20 * x^2 - 10 * x + 15) = 
              (x^2 * (p * x^2 + q * x + 20) - x * (3 * x + 10) + 15)) ∧ 
    (p = 0) ∧ (q = 25 / 3) :=
by {
  use [0, 25 / 3],
  split, 
  { sorry, }, 
  split, 
  { refl, },
  { refl, }
}

end polynomial_factor_coefficients_l336_336221


namespace train_speed_is_45_km_per_hr_l336_336937

/-- 
  Given the length of the train (135 m), the time to cross a bridge (30 s),
  and the length of the bridge (240 m), we want to prove that the speed of the 
  train is 45 km/hr.
--/

def length_of_train : ℕ := 135
def time_to_cross_bridge : ℕ := 30
def length_of_bridge : ℕ := 240
def speed_of_train_in_km_per_hr (L_t t L_b : ℕ) : ℕ := 
  ((L_t + L_b) * 36 / 10) / t

theorem train_speed_is_45_km_per_hr : 
  speed_of_train_in_km_per_hr length_of_train time_to_cross_bridge length_of_bridge = 45 :=
by 
  -- Assuming the calculations are correct, the expected speed is provided here directly
  sorry

end train_speed_is_45_km_per_hr_l336_336937


namespace solution_set_inequality_l336_336634

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then -Real.log x - x else -Real.log (-x) + x

theorem solution_set_inequality (m : ℝ) :
  (f (1 / m) < Real.log (1 / 2) - 2) ↔ (m ∈ Ioo (-1/2 : ℝ) 0 ∪ Ioo (0 : ℝ) 1/2) :=
begin
  sorry
end

end solution_set_inequality_l336_336634


namespace delta_four_equal_zero_l336_336249

-- Define the sequence u_n
def u (n : ℕ) : ℤ := n^3 + n

-- Define the ∆ operator
def delta1 (u : ℕ → ℤ) (n : ℕ) : ℤ := u (n + 1) - u n

def delta (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0   => u
  | k+1 => delta1 (delta k u)

-- The theorem statement
theorem delta_four_equal_zero (n : ℕ) : delta 4 u n = 0 :=
by sorry

end delta_four_equal_zero_l336_336249


namespace maximal_area_of_AMNQ_l336_336929

theorem maximal_area_of_AMNQ (s q : ℝ) (Hq1 : 0 ≤ q) (Hq2 : q ≤ s) :
  let Q := (s, q)
  ∃ M N : ℝ × ℝ, 
    (M.1 ∈ [0,s] ∧ M.2 = 0) ∧ 
    (N.1 = s ∧ N.2 ∈ [0,s]) ∧ 
    if q ≤ (2/3) * s 
    then 
      (M.1 * M.2 / 2 = (CQ/2)) 
    else 
      (N = (s, s)) :=
by sorry

end maximal_area_of_AMNQ_l336_336929


namespace perimeter_of_structure_l336_336818

noncomputable def structure_area : ℝ := 576
noncomputable def num_squares : ℕ := 9
noncomputable def square_area : ℝ := structure_area / num_squares
noncomputable def side_length : ℝ := Real.sqrt square_area
noncomputable def perimeter (side_length : ℝ) : ℝ := 8 * side_length

theorem perimeter_of_structure : perimeter side_length = 64 := by
  -- proof will follow here
  sorry

end perimeter_of_structure_l336_336818


namespace train_stop_duration_l336_336895

theorem train_stop_duration (speed_without_stoppages speed_with_stoppages : ℕ) (h1 : speed_without_stoppages = 45) (h2 : speed_with_stoppages = 42) :
  ∃ t : ℕ, t = 4 :=
by
  sorry

end train_stop_duration_l336_336895


namespace infinite_series_sum_l336_336017

theorem infinite_series_sum (c d : ℝ) (h1 : 0 < d) (h2 : 0 < c) (h3 : c > d) :
  (∑' n, 1 / (((2 * n + 1) * c - n * d) * ((2 * (n+1) - 1) * c - (n + 1 - 1) * d))) = 1 / ((c - d) * d) :=
sorry

end infinite_series_sum_l336_336017
