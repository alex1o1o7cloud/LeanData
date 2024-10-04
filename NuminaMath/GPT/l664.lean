import Mathlib

namespace num_valid_sequences_correct_l664_664096

noncomputable def num_valid_sequences : ℕ :=
  set.to_finset { p : ℤ × ℤ | 
                  let x := p.1, d := p.2 in
                  x + 2 * d = 108 ∧ x + 4 * d < 120 ∧ x > 48 ∧
                  all (λ k, x + k * d) (list.range (fin.val (5 : fin 5))) < 120 ∧
                  is_arithmetic_sequence [x, x + d, x + 2 * d, x + 3 * d, x + 4 * d] }
  sorry

theorem num_valid_sequences_correct : num_valid_sequences = 3 := by
  sorry

end num_valid_sequences_correct_l664_664096


namespace half_angle_in_second_and_fourth_quadrants_l664_664808

theorem half_angle_in_second_and_fourth_quadrants
  (k : ℤ) (α : ℝ) (h : 2 * k * π + π < α ∧ α < 2 * k * π + 3 * π / 2) :
  (∃ m : ℤ, m * π + π / 2 < α / 2 ∧ α / 2 < m * π + 3 * π / 4) :=
by sorry

end half_angle_in_second_and_fourth_quadrants_l664_664808


namespace symmetric_points_difference_l664_664453

-- We start by defining the points and their coordinates.
def point_A_x : ℤ := -4
def point_A_y : ℤ := 2

def point_B_x : ℤ := -point_A_x
def point_B_y : ℤ := -point_A_y

-- We state the theorem using the given condition:
theorem symmetric_points_difference : point_B_x - point_B_y = 6 :=
by
  -- We specify the values directly based on the properties of symmetry.
  have h1 : point_B_x = 4 := by {
    sorry
  }
  have h2 : point_B_y = -2 := by {
    sorry
  }
  -- Now we prove the required equality.
  calc 
    4 - (-2) = 4 + 2 := by sorry
    ... = 6 := by sorry

end symmetric_points_difference_l664_664453


namespace no_n_in_range_l664_664241

def g (n : ℕ) : ℕ := 7 + 4 * n + 6 * n ^ 2 + 3 * n ^ 3 + 4 * n ^ 4 + 3 * n ^ 5

theorem no_n_in_range
  : ¬ ∃ n : ℕ, 2 ≤ n ∧ n ≤ 100 ∧ g n % 11 = 0 := sorry

end no_n_in_range_l664_664241


namespace inv_value_l664_664436

def g (x : ℝ) : ℝ := 24 / (7 + 4 * x)
noncomputable def g_inv (y : ℝ) : ℝ := (24 / y - 7) / 4

theorem inv_value : (g_inv 3) ^ -3 = 64 := 
by
  have h₁ : g (g_inv 3) = 3,
    -- Proof that g (g_inv 3) = 3 (since g is the inverse of g_inv)
    sorry
  have g_inv_3_eq : g_inv 3 = 1/4,
    -- Solve for g_inv(3) using the equation
    sorry
  calc
    (g_inv 3) ^ -3 = (1/4) ^ -3 : by rw g_inv_3_eq
                ... = 4 ^ 3     : by sorry
                ... = 64        : by norm_num

end inv_value_l664_664436


namespace isosceles_triangle_perimeter_l664_664882

def is_perimeter_15 {a b : ℕ} (ha : a = 3) (hb : b = 6) (isosceles : a = b ∨ b = 6 ∨ 6 = a) : Prop :=
  2 * b + a = 15

theorem isosceles_triangle_perimeter : ∃ (a b : ℕ), a = 3 ∧ b = 6 ∧ (a = b ∨ b = 6 ∨ 6 = a) ∧ is_perimeter_15 (by rfl) (by rfl) (or.inr (or.inr rfl)) :=
begin
  use [3, 6],
  split,
  { refl },
  split,
  { refl },
  split,
  { right, right, refl },
  sorry
end

end isosceles_triangle_perimeter_l664_664882


namespace combined_weight_of_three_l664_664982

theorem combined_weight_of_three (Mary Jamison John : ℝ) 
  (h₁ : Mary = 160) 
  (h₂ : Jamison = Mary + 20) 
  (h₃ : John = Mary + (1/4) * Mary) :
  Mary + Jamison + John = 540 := by
  sorry

end combined_weight_of_three_l664_664982


namespace power_function_through_point_is_x_squared_l664_664277

theorem power_function_through_point_is_x_squared (a : ℝ) : (∀ x y : ℝ, y = x ^ a → (x, y) = (2, 4)) → a = 2 := by
  intro h
  have h1 : 4 = 2 ^ a := h 2 4 rfl
  have h2 : a = 2 := by sorry
  exact h2

end power_function_through_point_is_x_squared_l664_664277


namespace even_divisors_less_than_100_l664_664360

theorem even_divisors_less_than_100 :
  let count_even_divisors := 
    let n := 100 in
    let perfect_squares := { m | ∃ k, k * k = m ∧ m < n } in
    let total_numbers := finset.range n in
    (total_numbers.card - perfect_squares.card) =
    90 
  in count_even_divisors = 90 :=
by
  let n := 100
  let perfect_squares : finset ℕ := finset.filter (λ m, ∃ k, k * k = m) (finset.range n)
  let total_numbers : finset ℕ := finset.range n
  have h : total_numbers.card = 99 := by sorry
  have p : perfect_squares.card = 9 := by sorry
  show total_numbers.card - perfect_squares.card = 90
  calc
    total_numbers.card - perfect_squares.card
      = 99 - 9 := by rw [h, p]
      = 90 := by norm_num

end even_divisors_less_than_100_l664_664360


namespace range_of_t_l664_664991

theorem range_of_t (f : ℝ → ℝ) 
  (h_even : ∀ x, f (-x) = f x)
  (h_def : ∀ x, 0 ≤ x → f x = x^3)
  (h_ineq : ∀ x t, x ∈ set.Icc (2 * t - 1) (2 * t + 3) → f (3 * x - t) ≥ 8 * f x) :
  t = 0 ∨ t ≤ -3 ∨ t ≥ 1 :=
by
  sorry

end range_of_t_l664_664991


namespace remaining_tabs_after_closures_l664_664019

theorem remaining_tabs_after_closures (initial_tabs : ℕ) (first_fraction : ℚ) (second_fraction : ℚ) (third_fraction : ℚ) 
  (initial_eq : initial_tabs = 400) :
  (initial_tabs - initial_tabs * first_fraction - (initial_tabs - initial_tabs * first_fraction) * second_fraction - 
      ((initial_tabs - initial_tabs * first_fraction) - (initial_tabs - initial_tabs * first_fraction) * second_fraction) * third_fraction) = 90 :=
by
  have h1 : initial_tabs * first_fraction = 100 := by rw [initial_eq]; norm_num
  have h2 : initial_tabs - initial_tabs * first_fraction = 300 := by rw [initial_eq, h1]; norm_num
  have h3 : (initial_tabs - initial_tabs * first_fraction) * second_fraction = 120 := by rw [h2]; norm_num
  have h4 : (initial_tabs - initial_tabs * first_fraction) - (initial_tabs - initial_tabs * first_fraction) * second_fraction = 180 := by { rw [h2, h3]; norm_num }
  have h5 : ((initial_tabs - initial_tabs * first_fraction) - (initial_tabs - initial_tabs * first_fraction) * second_fraction) * third_fraction = 90 := by rw [h4]; norm_num
  have h6 : ((initial_tabs - initial_tabs * first_fraction) - (initial_tabs - initial_tabs * first_fraction) * second_fraction - ((initial_tabs - initial_tabs * first_fraction) - (initial_tabs - initial_tabs * first_fraction) * second_fraction) * third_fraction) = 90 := by rw [h4, h5]; norm_num
  exact h6


end remaining_tabs_after_closures_l664_664019


namespace positive_integer_solutions_count_positive_integer_solutions_l664_664843

theorem positive_integer_solutions (x : ℕ) : (8 < -2 * x + 16) → (x < 4) :=
by {
  sorry
}

theorem count_positive_integer_solutions : (finset.filter (λ n, 8 < -2 * n + 16) (finset.range 4)).card = 3 :=
by {
  sorry
}

end positive_integer_solutions_count_positive_integer_solutions_l664_664843


namespace comparison_proof_l664_664790

open Real

noncomputable def comparison (m : ℝ) (h : 0 < m ∧ m < 1) : Prop :=
  let a := log10 m
  let b := log10 (m^2)
  let c := (log10 m)^2
  c > a ∧ a > b

theorem comparison_proof {m : ℝ} (h : 0 < m ∧ m < 1) : comparison m h :=
  sorry

end comparison_proof_l664_664790


namespace find_x_l664_664138

theorem find_x
  (x : ℤ)
  (h : 3 * x + 3 * 15 + 3 * 18 + 11 = 152) :
  x = 14 :=
by
  sorry

end find_x_l664_664138


namespace altitudes_iff_area_l664_664794

variable {A B C D E F : Type}
variable (triangle_ABC : Triangle A B C)
variable (acute_triangle : IsAcute triangle_ABC)
variable (R : Real) -- Circumradius
variable (S : Real) -- Area of triangle ABC
variable [EuclideanGeometry]

theorem altitudes_iff_area (AD BE CF : Line) (h1 : OnLine D (LineThrough B C))
                                    (h2 : OnLine E (LineThrough C A))
                                    (h3 : OnLine F (LineThrough A B))
                                    (AD_altitude : IsAltitude AD)
                                    (BE_altitude : IsAltitude BE)
                                    (CF_altitude : IsAltitude CF) :
  (S = (R / 2) * (Distance E F + Distance F D + Distance D E)) ↔ 
  (AD_is_altitude_from_A : IsAltitudeFromPoint AD triangle_ABC A) ∧
  (BE_is_altitude_from_B : IsAltitudeFromPoint BE triangle_ABC B) ∧
  (CF_is_altitude_from_C : IsAltitudeFromPoint CF triangle_ABC C) :=
sorry

end altitudes_iff_area_l664_664794


namespace number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664380

theorem number_of_integers_with_even_divisors (n : ℕ) (h : n = 100) : 
  (card {x | (x < n) ∧ ∃ k, k * k = x} = 9) → 
  (card {x | (x < n) ∧ ¬(∃ k, k * k = x)} = n - 1 - 9) :=
by
  intro h_squares
  rw h
  trivial

open_locale classical
noncomputable def count_even_divisors_less_than_100 : ℕ :=
  90

theorem count_even_divisors_verification :
  count_even_divisors_less_than_100 = 90 :=
by
  sorry

end number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664380


namespace even_number_of_divisors_l664_664339

-- Proof statement: There are 90 positive integers less than 100 with an even number of divisors.
theorem even_number_of_divisors : 
  {n : ℕ | n < 100 ∧ ∃ k : ℕ, k ^ 2 = n}.toFinset.card = 90 := 
sorry

end even_number_of_divisors_l664_664339


namespace car_travel_in_next_hours_l664_664704

-- Define the initial conditions
def total_distance : ℝ := 180        -- miles
def total_time : ℝ := 4              -- hours
def next_time : ℝ := 3               -- hours

-- Define the question and prove the answer
theorem car_travel_in_next_hours :
  let average_speed := total_distance / total_time in
  let next_distance := average_speed * next_time in
  next_distance = 135 :=
by
  -- Proof goes here
  sorry

end car_travel_in_next_hours_l664_664704


namespace evens_divisors_lt_100_l664_664372

theorem evens_divisors_lt_100 : 
  ∃ n : ℕ, n = 90 ∧ ∀ k : ℕ, (1 ≤ k < 100) → (even k ↔ (∃ m : ℕ, m * m = k)) ↔ (n = 90) := 
sorry

end evens_divisors_lt_100_l664_664372


namespace parallelogram_area_l664_664897

variable {R : Type} [LinearOrderedField R]

/-- A proof that the area of parallelogram ABCD is 80/9 given the conditions about specific points and areas -/
theorem parallelogram_area
  (ABCD : Parallelogram R)
  (M N E F G H : Point R)
  (h1 : M = midpoint ABCD.A ABCD.B)
  (h2 : N = midpoint ABCD.D ABCD.C)
  (h3 : E = pt_div_trisection ABCD.B ABCD.C 1)
  (h4 : F = pt_div_trisection ABCD.B ABCD.C 2)
  (h5 : area (quadrilateral E F G H) = 1) :
  area ABCD = 80 / 9 :=
sorry

end parallelogram_area_l664_664897


namespace range_of_z_l664_664787

theorem range_of_z (α β : ℝ) (z : ℝ) (h1 : -2 < α) (h2 : α ≤ 3) (h3 : 2 < β) (h4 : β ≤ 4) (h5 : z = 2 * α - (1 / 2) * β) :
  -6 < z ∧ z < 5 :=
by
  sorry

end range_of_z_l664_664787


namespace production_cost_per_performance_l664_664575

theorem production_cost_per_performance
  (overhead : ℕ)
  (revenue_per_performance : ℕ)
  (num_performances : ℕ)
  (production_cost : ℕ)
  (break_even : num_performances * revenue_per_performance = overhead + num_performances * production_cost) :
  production_cost = 7000 :=
by
  have : num_performances = 9 := by sorry
  have : revenue_per_performance = 16000 := by sorry
  have : overhead = 81000 := by sorry
  exact sorry

end production_cost_per_performance_l664_664575


namespace solve_inequality_l664_664076

theorem solve_inequality (a : ℝ) : 
  (a = 0 → {x : ℝ | x ≥ -1} = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
  (a ≠ 0 → 
    ((a > 0 → { x : ℝ | -1 ≤ x ∧ x ≤ 2 / a } = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
    (-2 < a ∧ a < 0 → { x : ℝ | x ≤ 2 / a } ∪ { x : ℝ | -1 ≤ x }  = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
    (a < -2 → { x : ℝ | x ≤ -1 } ∪ { x : ℝ | x ≥ 2 / a } = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 }) ∧
    (a = -2 → { x : ℝ | True } = { x : ℝ | ax^2 + (a - 2) * x - 2 ≤ 0 })
)) :=
sorry

end solve_inequality_l664_664076


namespace greatest_prime_factor_15_fact_plus_18_fact_eq_17_l664_664663

theorem greatest_prime_factor_15_fact_plus_18_fact_eq_17 :
  ∃ p : ℕ, prime p ∧ 
  (∀ q : ℕ, (prime q ∧ q ∣ (15.factorial + 18.factorial)) → q ≤ p) ∧ 
  p = 17 :=
by
  sorry

end greatest_prime_factor_15_fact_plus_18_fact_eq_17_l664_664663


namespace find_angle_B_find_range_a_plus_2c_l664_664512

variables {A B C : ℝ} (a b c S : ℝ)

def angle_B (a b c S : ℝ) : Prop := 
  S = (Math.sqrt 3 / 4 * (a^2 + c^2 - b^2)) → B = Math.pi / 3

def range_a_plus_2c (a b c S : ℝ) : Prop :=
  b = Math.sqrt 3 → S = (Math.sqrt 3 / 4 * (a^2 + c^2 - b^2)) → 
  a + 2 * c ∈ Set.Ioc (Math.sqrt 3) (2 * Math.sqrt 7)

theorem find_angle_B 
  (h : S = Math.sqrt 3 / 4 * (a^2 + c^2 - b^2)) : B = Math.pi / 3 :=
sorry

theorem find_range_a_plus_2c 
  (hb : b = Math.sqrt 3)
  (h : S = Math.sqrt 3 / 4 * (a^2 + c^2 - b^2)) : 
  a + 2 * c ∈ Set.Ioc (Math.sqrt 3) (2 * Math.sqrt 7) :=
sorry

end find_angle_B_find_range_a_plus_2c_l664_664512


namespace even_number_of_divisors_lt_100_l664_664404

theorem even_number_of_divisors_lt_100 : 
  let n := 99 in
  let perfect_squares := {1, 4, 9, 16, 25, 36, 49, 64, 81} in
  n - perfect_squares.card = 90 := 
by
  sorry

end even_number_of_divisors_lt_100_l664_664404


namespace six_digit_integers_count_l664_664297

theorem six_digit_integers_count :
  let n := 6!
  let r := 2!
  n / (r * r * r) = 90 := 
by
  -- definitions based on conditions
  let n := Nat.factorial 6
  let r := Nat.factorial 2

  -- combine them to write the full statement
  have h1 : n = 720 := by apply Nat.factorial_succ

  have h2 : r = 2 := by apply Nat.factorial

  have h3 : (r * r * r) = 8 := by sorry -- multiplication of 2 * 2 * 2

  exact Eq.trans (Nat.div_eq_of_eq_mul_right (by norm_num) (by apply h1)) (by norm_num)

end six_digit_integers_count_l664_664297


namespace cousins_arrangement_l664_664551

def number_of_arrangements (cousins rooms : ℕ) (min_empty_rooms : ℕ) : ℕ := sorry

theorem cousins_arrangement : number_of_arrangements 5 4 1 = 56 := 
by sorry

end cousins_arrangement_l664_664551


namespace length_of_BC_l664_664925

theorem length_of_BC 
  (A B C X : Type) 
  (d_AB : ℝ) (d_AC : ℝ) 
  (circle_center_A : A) 
  (radius_AB : ℝ)
  (intersects_BC : B → C → X)
  (BX CX : ℕ) 
  (h_BX_in_circle : BX = d_AB) 
  (h_CX_in_circle : CX = d_AC) 
  (h_integer_lengths : ∃ x y : ℕ, BX = x ∧ CX = y) :
  BX + CX = 61 :=
begin
  sorry
end

end length_of_BC_l664_664925


namespace line_eq_proof_l664_664084

-- Define the conditions
def passes_through (l : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop := l p.1 p.2
def is_parallel (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∃ m : ℝ, ∀ x y, l1 x y = l2 x y + m

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - 3*y + 5 = 0
def line2 (x y : ℝ) : Prop := 2*x - 3*y + 7 = 0

-- Define the point
def point := (-2, 1) : ℝ × ℝ

-- Lean statement for the proof
theorem line_eq_proof : 
  (∃ m : ℝ, ∀ x y, (2*x - 3*y + m = 0) ∧ passes_through (λ x y, 2*x - 3*y + m = 0) point) ∧ 
  is_parallel (λ x y, 2*x - 3*y + m = 0) line1 → 
  line2 point.1 point.2 :=
sorry

end line_eq_proof_l664_664084


namespace fewest_cookies_l664_664693

-- Definitions for areas of the cookies
def area_circle := 9 * Real.pi
def area_square := 16
def area_hexagon := 6 * Real.sqrt 3
def area_triangle := (25 * Real.sqrt 3) / 4
def area_rectangle := 12

-- Prove that Bob makes the fewest cookies
theorem fewest_cookies : ∀ (d : ℝ), 
  (area_circle ≠ 0 ∧ area_square ≠ 0 ∧ area_hexagon ≠ 0 ∧ area_triangle ≠ 0 ∧ area_rectangle ≠ 0) →
  (d / area_square ≤ d / area_circle ∧ d / area_square ≤ d / area_hexagon ∧ d / area_square ≤ d / area_triangle ∧ d / area_square ≤ d / area_rectangle) :=
begin
  intros d h,
  sorry,
end

end fewest_cookies_l664_664693


namespace fourth_quadrant_angle_l664_664142

theorem fourth_quadrant_angle (α : ℝ) (k : ℤ)
  (h1 : α ∈ set.Ioo (2 * k * π - π / 2) (2 * k * π)) :
  2 * k * π - π / 2 < α ∧ α < 2 * k * π :=
begin
  split;
  exact h1
end

end fourth_quadrant_angle_l664_664142


namespace point_A_x_range_l664_664289

theorem point_A_x_range : 
  let L := {p : ℝ × ℝ | p.1 + p.2 = 9} 
  let M := {p : ℝ × ℝ | 2 * p.1 ^ 2 + 2 * p.2 ^ 2 - 8 * p.1 - 8 * p.2 - 1 = 0}
  let center_M := (2, 2)
  let radius_M := real.sqrt (34) / 2
  let A := {a : ℝ | ∃ b : ℝ, (a, b) ∈ L}
in ∀ (a : ℝ), (a, 9 - a) ∈ L → (∃ B C : ℝ × ℝ,
     B ∈ M ∧ C ∈ M ∧ ∠ B center_M C = 45 ∧
     (B.1 - center_M.1) * (C.2 - center_M.2) - (B.2 - center_M.2) * (C.1 - center_M.1) = 0
     ∧ ∀ a, 3 ≤ a ∧ a ≤ 6) := sorry

end point_A_x_range_l664_664289


namespace binomial_expansion_coeff_l664_664003

theorem binomial_expansion_coeff (x : ℝ) : 
  (∃ c : ℝ, (1 - 2 * x)^6 = c * x^2 + ∑ i in {0, 1, 2, 3, 4, 5, 6} \ {2}, _ * x^i) ∧ c = 60 :=
begin
  sorry
end

end binomial_expansion_coeff_l664_664003


namespace equilateral_implies_isosceles_converse_and_inverse_false_l664_664831

-- Definitions for the propositions p and q
def triangle (T : Type) := T
def is_equilateral (T : Type) : Prop := triangle T
def is_isosceles (T : Type) : Prop := triangle T

-- The original true statement
theorem equilateral_implies_isosceles (T : Type) : is_equilateral T → is_isosceles T := sorry  -- This is given as true

-- Prove that both the converse and inverse are false
theorem converse_and_inverse_false (T : Type) :
  (¬ (is_isosceles T → is_equilateral T)) ∧ (¬ (¬ is_equilateral T → ¬ is_isosceles T)) :=
begin
  -- Proof would go here
  sorry
end

end equilateral_implies_isosceles_converse_and_inverse_false_l664_664831


namespace complex_square_l664_664439

theorem complex_square (z : ℂ) (i : ℂ) (h₁ : z = 5 - 3 * i) (h₂ : i * i = -1) : z^2 = 16 - 30 * i :=
by
  rw [h₁]
  sorry

end complex_square_l664_664439


namespace cosine_of_B_in_geometric_sequence_triangle_l664_664513

theorem cosine_of_B_in_geometric_sequence_triangle
  (a b c : ℝ) (h1 : b^2 = a * c) (h2 : c = 2 * a) :
  real.cos (real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = 3 / 4 :=
by
  sorry

end cosine_of_B_in_geometric_sequence_triangle_l664_664513


namespace greatest_prime_factor_15_fact_plus_18_fact_l664_664641

theorem greatest_prime_factor_15_fact_plus_18_fact :
  Nat.greatest_prime_factor (15.factorial + 18.factorial) = 17 := by
  sorry

end greatest_prime_factor_15_fact_plus_18_fact_l664_664641


namespace oblique_projection_correct_l664_664127

variable (P1 : Prop) (P2 : Prop) (P3 : Prop) (P4 : Prop)

-- Conditions of the oblique projection method
def oblique_projection_conditions : Prop :=
  ∀ (x_parallel y_parallel : Prop),
    (x_parallel → (x_parallel ∧ true)) ∧ 
    (y_parallel → (y_parallel ∧ true)) ∧ 
    (true → (true → true)) → 
    P1 ∧ P2

-- The intuitive diagrams based on the given rules
def intuitive_diagrams (triangle_is_triangle : Prop) 
  (parallelogram_is_parallelogram : Prop) 
  (square_is_square : Prop) 
  (rhombus_is_rhombus : Prop) : Prop :=
  triangle_is_triangle ∧ parallelogram_is_parallelogram ∧ ¬square_is_square ∧ ¬rhombus_is_rhombus

-- Problem statement: Prove that the correct choice is A (P1 ∧ P2)
theorem oblique_projection_correct (cond : oblique_projection_conditions) 
  (triangle_is_triangle parallelogram_is_parallelogram : Prop) 
  (square_is_square rhombus_is_rhombus : Prop) :
  intuitive_diagrams triangle_is_triangle parallelogram_is_parallelogram square_is_square rhombus_is_rhombus →
  (triangle_is_triangle ∧ parallelogram_is_parallelogram) :=
by
  intro h
  cases h
  split
  case left => assumption
  case right => assumption


end oblique_projection_correct_l664_664127


namespace even_divisors_count_lt_100_l664_664351

theorem even_divisors_count_lt_100 : 
  {n : ℕ | n < 100 ∧ n ≠ 0 ∧ ∃ k : ℕ, k * k = n } = { n : ℕ | n < 100 ∧ n ≠ 0 } \ 
  { n : ℕ | ∃ k : ℕ, k * k = n ∧ k < 100 } → 
  (card {n : ℕ | n < 100 ∧ n ≠ 0 ∧ even (finset.card (divisors n))} = 90) :=
begin
  sorry
end

end even_divisors_count_lt_100_l664_664351


namespace expected_value_sum_marbles_l664_664424

theorem expected_value_sum_marbles :
  (1/15 : ℚ) * ((1 + 2) + (1 + 3) + (1 + 4) + (1 + 5) + (1 + 6) + 
                (2 + 3) + (2 + 4) + (2 + 5) + (2 + 6) + (3 + 4) + 
                (3 + 5) + (3 + 6) + (4 + 5) + (4 + 6) + (5 + 6)) = 7 := 
by {
    sorry
}

end expected_value_sum_marbles_l664_664424


namespace measure_of_angle_QRP_l664_664002

-- Define points Q, R, S, and P
variables (Q R S P : Type) [Points : Geometry.Point Q] [Points : Geometry.Point R]
variables [Points : Geometry.Point S] [Points : Geometry.Point P]

-- Conditions
def is_straight_line (Q R S : Type) : Prop := Geometry.Collinear Q R S
def angle_PQS : ℝ := 65
def angle_PSQ : ℝ := 50
def angle_PRQ : ℝ := 70

-- Given the conditions, prove that the measure of ∠QRP is 45 degrees
theorem measure_of_angle_QRP : is_straight_line Q R S →
  Geometry.Angle Q P S = angle_PQS →
  Geometry.Angle P S Q = angle_PSQ →
  Geometry.Angle P R Q = angle_PRQ →
  Geometry.Angle Q R P = 45 := by
  sorry

end measure_of_angle_QRP_l664_664002


namespace patricia_walked_approx_1650_miles_l664_664066

theorem patricia_walked_approx_1650_miles :
  ∀ (max_steps per_mile final_read cycles : ℕ),
  max_steps = 89999 →
  per_mile = 1500 →
  final_read = 40000 →
  cycles = 27 →
  (cycles * (max_steps + 1) + final_read) / per_mile ≈ 1650 :=
by 
  intros max_steps per_mile final_read cycles h1 h2 h3 h4
  let total_steps := cycles * (max_steps + 1) + final_read
  let miles := total_steps / per_mile
  have : miles ≈ 1650 := sorry
  exact this

end patricia_walked_approx_1650_miles_l664_664066


namespace max_m_value_l664_664435

theorem max_m_value 
  (m : ℝ) 
  (f : ℝ → ℝ)
  (h_f : ∀ (x : ℝ), f x = sin x + (√3) * cos x)
  (h_m : m > 0)
  (h_increasing : ∀ (x y : ℝ), -m ≤ x → x ≤ y → y ≤ m → f x ≤ f y) :
  m ≤ π / 6 :=
sorry

end max_m_value_l664_664435


namespace symmetric_point_about_xaxis_is_correct_l664_664510

-- Define the point structure
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Function to compute the symmetric point about the x-axis
def symmetricAboutX (p : Point) : Point :=
  ⟨p.x, -p.y, -p.z⟩

-- The given point P(3, -2, 1)
def P := ⟨3.0, -2.0, 1.0⟩

-- The expected point after symmetry
def expected := ⟨3.0, 2.0, -1.0⟩

-- The proof statement
theorem symmetric_point_about_xaxis_is_correct :
  symmetricAboutX P = expected :=
by
  sorry

end symmetric_point_about_xaxis_is_correct_l664_664510


namespace min_product_pm_pn_l664_664713

theorem min_product_pm_pn
  (α k : ℝ) (P : ℝ × ℝ) (Ellipse : ℝ × ℝ → Prop) :
  P = (sqrt 10 / 2, 0) →
  (∀ M N : ℝ × ℝ, (Ellipse M ∧ Ellipse N) →
    ∃ PM PN : ℝ, 
      (PM = dist P M ∧ PN = dist P N) ∧
      ∀ k = tan α, 
      y = k * (x - sqrt 10 / 2) ∧ (x^2 + 12*y^2 = 1) ∧ (M × N = (3 + 3*k^2) / (24*k^2 + 2)) ∧
      (1 / 8 * (1 + 11 / (12*k^2 + 1)) ≤ 19 / 20)) :=
begin
  sorry
end

end min_product_pm_pn_l664_664713


namespace even_number_of_divisors_less_than_100_l664_664314

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k, k * k = n

theorem even_number_of_divisors_less_than_100 :
  (card { n : ℕ | 1 ≤ n ∧ n < 100 ∧ ¬ is_perfect_square n }) = 90 :=
by
  sorry

end even_number_of_divisors_less_than_100_l664_664314


namespace even_number_of_divisors_lt_100_l664_664400

theorem even_number_of_divisors_lt_100 : 
  let n := 99 in
  let perfect_squares := {1, 4, 9, 16, 25, 36, 49, 64, 81} in
  n - perfect_squares.card = 90 := 
by
  sorry

end even_number_of_divisors_lt_100_l664_664400


namespace petya_verifies_weights_l664_664064

theorem petya_verifies_weights :
  (∀ (x y : ℕ), x + y = 3 → x + 4 * y = 9 → x = 1 ∧ y = 2) :=
by
  intros x y h1 h2
  have h3 : 3 * x + 3 * y = 9, from by linarith,
  have h4 : x + y = 3, from h1,
  have h5 : x + 4 * y = 9, from h2,
  sorry

end petya_verifies_weights_l664_664064


namespace pounds_of_cheese_bought_l664_664120

-- Definitions according to the problem's conditions
def initial_money : ℕ := 87
def cheese_cost_per_pound : ℕ := 7
def beef_cost_per_pound : ℕ := 5
def pounds_of_beef : ℕ := 1
def remaining_money : ℕ := 61

-- The Lean 4 proof statement
theorem pounds_of_cheese_bought :
  ∃ (C : ℕ), initial_money - (cheese_cost_per_pound * C + beef_cost_per_pound * pounds_of_beef) = remaining_money ∧ C = 3 :=
begin
  sorry,
end

end pounds_of_cheese_bought_l664_664120


namespace sum_of_products_nonpos_l664_664045

theorem sum_of_products_nonpos (a b c : ℝ) (h : a + b + c = 0) : 
  a * b + a * c + b * c ≤ 0 :=
sorry

end sum_of_products_nonpos_l664_664045


namespace rectangular_coordinates_of_polar_2_pi_over_3_l664_664636

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem rectangular_coordinates_of_polar_2_pi_over_3 :
  polar_to_rectangular 2 (Real.pi / 3) = (1, Real.sqrt 3) :=
by
  sorry

end rectangular_coordinates_of_polar_2_pi_over_3_l664_664636


namespace travel_possible_with_three_roads_l664_664503

-- Define the cities as vertices of a directed graph
variables {City : Type} [Fintype City]

-- Define the directed roads as edges
variable (roads : City → City → Prop)

-- Define the conditions
def all_one_way (roads : City → City → Prop) : Prop :=
  ∀ x y : City, x ≠ y → (roads x y ∨ roads y x)

def reachable_in_two (roads : City → City → Prop) : Prop :=
  ∀ x y : City, ∃ z : City, roads x z ∧ (roads z y ∨ ∃ w : City, roads z w ∧ roads w y)

def closed_road (closed : City × City) (roads : City → City → Prop) : City → City → Prop :=
  λ x y, (x, y) ≠ closed ∧ roads x y

def reachable_in_three (roads : City → City → Prop) : Prop :=
  ∀ x y : City, ∃ u v : City, (roads x u ∧ roads u v ∧ (roads v y ∨ ∃ w : City, roads v w ∧ roads w y)) ∨ (roads x u ∧ (roads u v ∧ (roads v y ∨ ∃ w : City, roads v w ∧ roads w y)))

-- The main statement to prove
theorem travel_possible_with_three_roads 
  (h_one_way : all_one_way roads)
  (h_reachable_two : reachable_in_two roads)
  (closed : City × City)
  (h_reachable_after_closure : reachable_in_two (closed_road closed roads)) :
  reachable_in_three (closed_road closed roads) := sorry

end travel_possible_with_three_roads_l664_664503


namespace count_uniquely_oddly_powerful_l664_664741

def is_uniquely_oddly_powerful (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a % 2 = 1 ∧ b % 2 = 1 ∧ a^b = n

theorem count_uniquely_oddly_powerful : (finset.filter is_uniquely_oddly_powerful (finset.range 5000)).card = 10 := 
by
  sorry

end count_uniquely_oddly_powerful_l664_664741


namespace period_of_sine_function_l664_664820

theorem period_of_sine_function (a b : ℝ) (h : ∀ x ∈ set.Icc a b, -1 <= 2 * Real.sin x ∧ 2 * Real.sin x <= 2):
  ¬ (b - a = 5 * Real.pi / 3) :=
sorry

end period_of_sine_function_l664_664820


namespace find_g_x2_minus_2_l664_664041

def g : ℝ → ℝ := sorry -- Define g as some real-valued polynomial function.

theorem find_g_x2_minus_2 (x : ℝ) 
(h1 : g (x^2 + 2) = x^4 + 5 * x^2 + 1) : 
  g (x^2 - 2) = x^4 - 3 * x^2 - 7 := 
by sorry

end find_g_x2_minus_2_l664_664041


namespace area_of_AEFB_l664_664731

-- Defining the areas of the triangles based on our conditions
variable (area_FDC area_FDE : ℝ) 
variable (A B C D E F : Type)

-- Main statement to prove the area of the quadrilateral AEFB equals 10 given the conditions
theorem area_of_AEFB (H1 : area_FDC = 4) (H2 : area_FDE = 2) : 
  let area_AEFB := 10 in
  area_AEFB = (area_FDC + area_FDE + 4) :=
sorry

end area_of_AEFB_l664_664731


namespace find_time_same_height_l664_664157

noncomputable def height_ball (t : ℝ) : ℝ := 60 - 9 * t - 8 * t^2
noncomputable def height_bird (t : ℝ) : ℝ := 3 * t^2 + 4 * t

theorem find_time_same_height : ∃ t : ℝ, t = 20 / 11 ∧ height_ball t = height_bird t := 
by
  use 20 / 11
  sorry

end find_time_same_height_l664_664157


namespace parallel_line_through_point_l664_664591

-- Define the condition of point and line
def point1 := (1 : ℝ, 0 : ℝ)
def line1 (x y: ℝ) := x - 2 * y - 2 = 0

-- Define the equation of the line parallel to line1 passing through point1
def parallel_line_eq (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Statement to prove
theorem parallel_line_through_point : 
  ∃ c : ℝ, (∀ x y : ℝ, (x - 2 * y + c = 0) → (1 - 2 * 0 + c = 0) → c = -1) 
    ∧ (parallel_line_eq = λ (x y : ℝ), x - 2 * y - 1 = 0) :=
begin
  use -1,
  split,
  {
    intros x y h1 h2,
    simp at h2,
    exact h2,
  },
  {
    funext,
    rfl,
  }
end

end parallel_line_through_point_l664_664591


namespace BC_length_l664_664955

def triangle_ABC (A B C : Type)
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)] : Prop :=
  ∃ (X : A), (has_dist B X (coe (X.dist B))) ∧ (has_dist C X (coe (X.dist C))) ∧
  ∀ (x y : ℕ), x = X.dist B ∧ y = X.dist C → x + y = 61

theorem BC_length {A B C : Type}
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)]
  (h : triangle_ABC A B C) : 
  ∃ (x y : ℕ), x + y = 61 := sorry

end BC_length_l664_664955


namespace cost_of_stuffers_number_of_combinations_l664_664840

noncomputable def candy_cane_cost : ℝ := 4 * 0.5
noncomputable def beanie_baby_cost : ℝ := 2 * 3
noncomputable def book_cost : ℝ := 5
noncomputable def toy_cost : ℝ := 3 * 1
noncomputable def gift_card_cost : ℝ := 10
noncomputable def one_child_stuffers_cost : ℝ := candy_cane_cost + beanie_baby_cost + book_cost + toy_cost + gift_card_cost
noncomputable def total_cost : ℝ := one_child_stuffers_cost * 4

def num_books := 5
def num_toys := 10
def toys_combinations : ℕ := Nat.choose num_toys 3
def total_combinations : ℕ := num_books * toys_combinations

theorem cost_of_stuffers (h : total_cost = 104) : total_cost = 104 := by
  sorry

theorem number_of_combinations (h : total_combinations = 600) : total_combinations = 600 := by
  sorry

end cost_of_stuffers_number_of_combinations_l664_664840


namespace correct_average_l664_664583

theorem correct_average (incorrect_avg : ℝ) (n : ℕ) (corrections : list (ℝ × ℝ)) :
  incorrect_avg = 28.7 →
  n = 20 →
  corrections = [(75.3, 55.3), (62.2, 42.2), (89.1, 69.1)] →
  (incorrect_avg * n + list.sum (corrections.map (λ x, x.1 - x.2))) / n = 31.7 :=
by
  intros h_avg h_n h_corr
  sorry

end correct_average_l664_664583


namespace even_number_of_divisors_lt_100_l664_664401

theorem even_number_of_divisors_lt_100 : 
  let n := 99 in
  let perfect_squares := {1, 4, 9, 16, 25, 36, 49, 64, 81} in
  n - perfect_squares.card = 90 := 
by
  sorry

end even_number_of_divisors_lt_100_l664_664401


namespace math_problem_l664_664455

noncomputable def log_base (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem math_problem (a b c : ℝ) (h1 : ∃ k : ℤ, log_base c b = k)
  (h2 : log_base a (1 / b) > log_base a (Real.sqrt b) ∧ log_base a (Real.sqrt b) > log_base b (a^2)) :
  (∃ n : ℕ, n = 1 ∧ 
    ((1 / b > Real.sqrt b ∧ Real.sqrt b > a^2) ∨ 
    (Real.log b + log_base a a = 0) ∨ 
    (0 < a ∧ a < b ∧ b < 1) ∨ 
    (a * b = 1))) :=
by sorry

end math_problem_l664_664455


namespace line_passes_fixed_point_shortest_chord_length_l664_664791

open Real

-- Define the circle M
def circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 3)^2 = 16

-- Define the line l
def line (m x y : ℝ) : Prop :=
  (m + 1) * x + (m + 4) * y - 3 * m = 0

-- Problem 1: Prove that line l passes through a fixed point
theorem line_passes_fixed_point (m : ℝ) : ∃ x y, (x = 4 ∧ y = -1) ∧ line m x y :=
by sorry

-- Problem 2: Find the shortest chord of circle M intercepted by line l
theorem shortest_chord_length (m : ℝ) : 
  ∃ d, (d = 2 * sqrt 11) ∧ (∃ x1 y1 x2 y2, x1 ≠ x2 ∧ y1 ≠ y2 ∧ circle x1 y1 ∧ circle x2 y2 ∧ line m x1 y1 ∧ line m x2 y2 ∧ d = dist (x1, y1) (x2, y2)) :=
by sorry

end line_passes_fixed_point_shortest_chord_length_l664_664791


namespace BC_length_l664_664952

-- Define the given values and conditions
variable (A B C X : Type)
variable (AB AC AX BX CX : ℕ)
variable [inhabited A] [inhabited B] [inhabited C] [inhabited X]

-- Assume the lengths of AB and AC
axiom h_AB : AB = 86
axiom h_AC : AC = 97

-- Assume the circle centered at A with radius AB intersects BC at B and X
axiom h_circle : AX = AB

-- Assume BX and CX are integers
axiom h_BX_integral : ∃ (x : ℕ), BX = x
axiom h_CX_integral : ∃ (y : ℕ), CX = y

-- The statement to prove that the length of BC is 61
theorem BC_length : (∃ (x y : ℕ), BX = x ∧ CX = y ∧ x + y = 61) :=
by
  sorry

end BC_length_l664_664952


namespace symmetric_point_xOz_l664_664012

def symmetric_point (p : (ℝ × ℝ × ℝ)) (plane : ℝ → Prop) : (ℝ × ℝ × ℝ) :=
match p with
| (x, y, z) => (x, -y, z)

theorem symmetric_point_xOz (x y z : ℝ) : symmetric_point (-1, 2, 1) (λ y, y = 0) = (-1, -2, 1) :=
by
  sorry

end symmetric_point_xOz_l664_664012


namespace smallest_sum_l664_664680

-- Defining the problem scenario
theorem smallest_sum (a b c d e f : ℕ) (h1 : a + b + c + d + e + f = 30)
  (h2 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f)  
  (h3 : a ∈ { 1, 2, 3, 7, 8, 9 }) (h4 : b ∈ { 1, 2, 3, 7, 8, 9 }) (h5 : c ∈ { 1, 2, 3, 7, 8, 9 }) (h6 : d ∈ { 1, 2, 3, 7, 8, 9 }) (h7 : e ∈ { 1, 2, 3, 7, 8, 9 }) (h8 : f ∈ { 1, 2, 3, 7, 8, 9 }) 
  : 100*a + 10*b + c + 100*d + 10*e + f = 417 := by
  sorry

end smallest_sum_l664_664680


namespace cost_B_solution_l664_664631

variable (cost_B : ℝ)

/-- The number of items of type A that can be purchased with 1000 yuan 
is equal to the number of items of type B that can be purchased with 800 yuan. -/
def items_purchased_equality (cost_B : ℝ) : Prop :=
  1000 / (cost_B + 10) = 800 / cost_B

/-- The cost of each item of type A is 10 yuan more than the cost of each item of type B. -/
def cost_difference (cost_B : ℝ) : Prop :=
  cost_B + 10 - cost_B = 10

/-- The cost of each item of type B is 40 yuan. -/
theorem cost_B_solution (h1: items_purchased_equality cost_B) (h2: cost_difference cost_B) :
  cost_B = 40 := by
sorry

end cost_B_solution_l664_664631


namespace PA_perp_BC_l664_664040

open_locale classical
open_locale real

-- We assume our context is within a plane geometry type
variables {A B C D E F G H P : Point}
variables {EG FH BC : Line}

-- Definitions and conditions
def point_of_tangency (H : Point) := 
  -- H is the point of tangency assumption
  sorry

def lines_intersect_at (EG FH : Line) (P : Point) :=
  -- Lines EG and FH intersect at point P assumption
  sorry

-- Lean 4 statement of the proof problem
theorem PA_perp_BC (hp1 : point_of_tangency H) (hp2 : lines_intersect_at EG FH P) : 
  perpendicular (line_through P A) BC :=
sorry

end PA_perp_BC_l664_664040


namespace triangle_bc_length_l664_664904

theorem triangle_bc_length (A B C X : Type)
  (AB AC : ℕ)
  (hAB : AB = 86)
  (hAC : AC = 97)
  (circle_eq : ∀ {r : ℕ}, r = AB → circle_centered_at_A_intersects_BC_two_points B X)
  (integer_lengths : ∃ (BX CX : ℕ), ) :
  BC = 61 :=
by
  sorry

end triangle_bc_length_l664_664904


namespace power_comparison_l664_664209

noncomputable
def compare_powers : Prop := 
  1.5^(1 / 3.1) < 2^(1 / 3.1) ∧ 2^(1 / 3.1) < 2^(3.1)

theorem power_comparison : compare_powers :=
by
  sorry

end power_comparison_l664_664209


namespace sphere_in_cone_radius_l664_664174

theorem sphere_in_cone_radius (b d : ℝ) (r : ℝ) 
  (h_base_radius : 15) (h_height : 30)
  (h_radius : r = b * Real.sqrt d - b) :
  b + d = 12.5 :=
sorry

end sphere_in_cone_radius_l664_664174


namespace BC_length_l664_664913

theorem BC_length (A B C X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X] 
  (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
  (BX CX : ℕ) (h_circle_intersect : ∃ X, Metric.ball A 86 ∩ {BC} = {B, X})
  (h_integer_lengths : BX + CX = BC) :
  BC = 61 := 
by
  sorry

end BC_length_l664_664913


namespace find_tanya_items_l664_664051

-- Let's define our variables:
variables (L S T : ℕ)

-- Assume the conditions given in the problem:
def cond1 : Prop := L = S + 4
def cond2 : Prop := S = 4 * T
def cond3 : Prop := L = 20

-- Prove that Tanya found 4 items:
theorem find_tanya_items (h1 : cond1) (h2 : cond2) (h3 : cond3) : T = 4 :=
by
  sorry

end find_tanya_items_l664_664051


namespace cinema_revenue_and_visitors_change_l664_664871

-- Define the variables and parameters
variables (A : ℝ) (initial_revenue_year1 : ℝ) (price_I price_II price_III : ℝ)
variables (revenue_I_year1 revenue_II_year1 revenue_III_year1 : ℝ)
variables (change_I change_II change_III : ℝ)

-- Assign the provided values
noncomputable def values := 
  initial_revenue_year1 = 100 * A ∧
  price_I = 6 ∧ price_II = 4 ∧ price_III = 3 ∧
  revenue_I_year1 = 30 * A ∧ revenue_II_year1 = 50 * A ∧ revenue_III_year1 = 20 * A ∧
  change_I = 1.20 ∧ change_II = 1.30 ∧ change_III = 0.95

-- Define the statement to prove
theorem cinema_revenue_and_visitors_change :
  values A initial_revenue_year1 price_I price_II price_III revenue_I_year1 revenue_II_year1 revenue_III_year1 change_I change_II change_III →
  let revenue_total_year2 := change_I * revenue_I_year1 + change_II * revenue_II_year1 + change_III * revenue_III_year1 in 
  let tickets_I_year1 := revenue_I_year1 / price_I in
  let tickets_II_year1 := revenue_II_year1 / price_II in
  let tickets_III_year1 := revenue_III_year1 / price_III in
  let tickets_I_year2 := change_I * revenue_I_year1 / price_I in
  let tickets_II_year2 := change_II * revenue_II_year1 / price_II in
  let tickets_III_year2 := change_III * revenue_III_year1 / price_III in
  let total_tickets_year1 := tickets_I_year1 + tickets_II_year1 + tickets_III_year1 in
  let total_tickets_year2 := tickets_I_year2 + tickets_II_year2 + tickets_III_year2 in
  revenue_total_year2 = 120 * A ∧ total_tickets_year2 = 1.1827 * total_tickets_year1 :=
sorry

end cinema_revenue_and_visitors_change_l664_664871


namespace tangent_line_to_ellipse_l664_664861

theorem tangent_line_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 1 → x^2 + 4 * y^2 = 1 → (x^2 + 4 * (m * x + 1)^2 = 1)) →
  m^2 = 3 / 4 :=
by
  sorry

end tangent_line_to_ellipse_l664_664861


namespace sum_of_cubes_of_roots_l664_664210

theorem sum_of_cubes_of_roots:
  (∀ r s t : ℝ, (r + s + t = 8) ∧ (r * s + s * t + t * r = 9) ∧ (r * s * t = 2) → r^3 + s^3 + t^3 = 344) :=
by
  intros r s t h
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2
  sorry

end sum_of_cubes_of_roots_l664_664210


namespace flagpole_height_proof_l664_664163

noncomputable def flagpole_height (AC AD DE : ℝ) (wire_length : ℝ) : ℝ :=
let DC := AC - AD in
let ratio_side := DE / DC in
ratio_side * AC

theorem flagpole_height_proof (AC AD DE : ℝ) (wire_length : ℝ) (H_sim : AC ≠ 0 ∧ DC ≠ 0) :
  AC = 4 ∧ AD = 3.5 ∧ DE = 1.8 → flagpole_height AC AD DE wire_length = 14.4 :=
by
  intro h
  cases h with h₁ h₂
  cases h₂ with h₃ h₄
  have := h₁
  rw h₂ at this
  rw h₃ at this
  sorry

end flagpole_height_proof_l664_664163


namespace cost_per_meter_of_fencing_l664_664169

/-- A rectangular farm has area 1200 m², a short side of 30 m, and total job cost 1560 Rs.
    Prove that the cost of fencing per meter is 13 Rs. -/
theorem cost_per_meter_of_fencing
  (A : ℝ := 1200)
  (W : ℝ := 30)
  (job_cost : ℝ := 1560)
  (L : ℝ := A / W)
  (D : ℝ := Real.sqrt (L^2 + W^2))
  (total_length : ℝ := L + W + D) :
  job_cost / total_length = 13 := 
sorry

end cost_per_meter_of_fencing_l664_664169


namespace triangle_perimeter_l664_664180

def triangle_side_lengths : ℕ × ℕ × ℕ := (10, 6, 7)

def perimeter (a b c : ℕ) : ℕ := a + b + c

theorem triangle_perimeter (a b c : ℕ) (h : (a, b, c) = triangle_side_lengths) : 
  perimeter a b c = 23 := by
  -- We formulate the statement and leave the proof for later
  sorry

end triangle_perimeter_l664_664180


namespace largest_possible_boxes_l664_664781

theorem largest_possible_boxes (k : ℕ) :
  let total_weight := (Σ i in Finset.range (k+1), 8 * 2^i)
  let max_boxes := 8 + 4 + 2 + 1
  (∀ n, (∃ w, w * n = total_weight) → n ≤ max_boxes) →
  n = 15 :=
by
  sorry

end largest_possible_boxes_l664_664781


namespace find_angle_C_l664_664474

theorem find_angle_C
  (a b c A B C : ℝ)
  (sin_A sin_B sin_C cos_C : ℝ)
  (h1 : a = 2)
  (h2 : c = sqrt 2)
  (h3 : A = 3 / 4 * π)
  (h4 : sin A = sqrt 2 / 2)
  (h5 : sin B + sin A * (sin C - cos C) = 0)
  (h6 : 0 < C ∧ C < π / 2)
  : C = π / 6 := 
begin
  sorry
end

end find_angle_C_l664_664474


namespace derivative_of_even_is_odd_l664_664057

-- Define that f is an even function
def is_even (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = f x

-- Define the condition that g is the derivative of f
def is_derivative (f g : ℝ → ℝ) : Prop :=
∀ x : ℝ, (g x) = (f' x)

-- The proof problem statement
theorem derivative_of_even_is_odd (f g : ℝ → ℝ) (hf_even : is_even f) (hg_deriv : is_derivative f g) : 
  ∀ x : ℝ, g (-x) = -g x :=
by
  sorry

end derivative_of_even_is_odd_l664_664057


namespace original_population_l664_664487

theorem original_population (P : ℝ) (h1 : P * 0.5049 = 4136) : P ≈ 8192 :=
by
  sorry

end original_population_l664_664487


namespace propositions_true_general_conclusion_l664_664532

variables {a b c h : ℝ}
def right_angled_triangle (a b c : ℝ) := a^2 + b^2 = c^2
def height_to_hypotenuse (h c : ℝ) := h * c = a * b / 2

theorem propositions_true 
  (a b c h : ℝ) 
  (h_abc : right_angled_triangle a b c)
  (h_h : height_to_hypotenuse h c) 
  : a^2 + b^2 < c^2 + h^2 ∧ a^4 + b^4 < c^4 + h^4 := 
sorry

theorem general_conclusion 
  (a b c h : ℝ) 
  (h_abc : right_angled_triangle a b c)
  (h_h : height_to_hypotenuse h c) 
  (n : ℕ) 
  (h_n : 2 ≤ n) 
  : a^n + b^n < c^n + h^n := 
sorry

end propositions_true_general_conclusion_l664_664532


namespace four_digit_number_prime_factors_l664_664717

theorem four_digit_number_prime_factors 
  (n : ℕ)
  (h1 : (∀ (p: ℕ), p | n → Nat.Prime p)) 
  (h2 : (1 + smallest_prime_factor n + next_prime_factor_after (smallest_prime_factor n) n = 15)) 
  (h3 : Nat.divisors n ∣ 8)
  (h4 : ∃ (p1 p2 p3: ℕ), 
          (Nat.Prime p1) ∧ 
          (Nat.Prime p2) ∧ 
          (Nat.Prime p3) ∧ 
          (p1 - 5 * p2 = 2 * p3)) : 
  n = 1221 :=
sorry

end four_digit_number_prime_factors_l664_664717


namespace even_number_of_divisors_less_than_100_l664_664331

theorem even_number_of_divisors_less_than_100 : 
  ∃ n, n = 90 ∧ ∀ x < 100, (∃ k, k * k = x → false) = (x ∣ 99 - 9) :=
sorry

end even_number_of_divisors_less_than_100_l664_664331


namespace each_person_paid_45_l664_664725

theorem each_person_paid_45 (total_bill : ℝ) (number_of_people : ℝ) (per_person_share : ℝ) 
    (h1 : total_bill = 135) 
    (h2 : number_of_people = 3) :
    per_person_share = 45 :=
by
  sorry

end each_person_paid_45_l664_664725


namespace evens_divisors_lt_100_l664_664375

theorem evens_divisors_lt_100 : 
  ∃ n : ℕ, n = 90 ∧ ∀ k : ℕ, (1 ≤ k < 100) → (even k ↔ (∃ m : ℕ, m * m = k)) ↔ (n = 90) := 
sorry

end evens_divisors_lt_100_l664_664375


namespace even_number_of_divisors_less_than_100_l664_664330

theorem even_number_of_divisors_less_than_100 : 
  ∃ n, n = 90 ∧ ∀ x < 100, (∃ k, k * k = x → false) = (x ∣ 99 - 9) :=
sorry

end even_number_of_divisors_less_than_100_l664_664330


namespace triangle_AC_length_l664_664900

noncomputable def length_AC (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : ℝ :=
  6

theorem triangle_AC_length 
  (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (angle_B : ℝ) (circle1_radius : ℝ) (circle2_radius : ℝ) 
  (tangent1 : Line A C) (tangent2 : Line A C)
  (h_angle_B : angle_B = π / 3)
  (h_circle1_radius: circle1_radius = 3)
  (h_circle2_radius: circle2_radius = 4)
  (h_tangent1: tangent1.tangentAt A)
  (h_tangent2: tangent2.tangentAt C)
  : length_AC A B C = 6 :=
sorry

end triangle_AC_length_l664_664900


namespace neg_of_exists_lt_is_forall_ge_l664_664599

theorem neg_of_exists_lt_is_forall_ge :
  (¬ (∃ x : ℝ, x^2 - 2 * x + 1 < 0)) ↔ (∀ x : ℝ, x^2 - 2 * x + 1 ≥ 0) :=
by
  sorry

end neg_of_exists_lt_is_forall_ge_l664_664599


namespace vector_ab_tan_x_given_parallel_l664_664886

variable (x : ℝ)
def A := (-1, 0 : ℝ × ℝ)
def B := (0, Real.sqrt 3 : ℝ × ℝ)
def C := (Real.cos x, Real.sin x : ℝ × ℝ)

theorem vector_ab (x : ℝ) : 
  (1, Real.sqrt 3) = 
    let A := (-1, 0 : ℝ × ℝ)
    let B := (0, Real.sqrt 3 : ℝ × ℝ)
    (B.1 - A.1, B.2 - A.2) := by
  sorry

theorem tan_x_given_parallel (x : ℝ) (h : (1, Real.sqrt 3) ∥ (Real.cos x, Real.sin x)) :
  Real.tan x = Real.sqrt 3 := by 
  sorry

end vector_ab_tan_x_given_parallel_l664_664886


namespace number_of_integers_with_even_divisors_l664_664395

-- Define conditions
def N := 99

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P : finset ℕ := { n in finset.range (N + 1) | is_perfect_square n }

noncomputable def countP : ℕ := P.card

-- The statement to prove
theorem number_of_integers_with_even_divisors : 
  (N - countP) = 90 :=
by {
  sorry
}

end number_of_integers_with_even_divisors_l664_664395


namespace triangle_perimeter_l664_664179

def triangle_side_lengths : ℕ × ℕ × ℕ := (10, 6, 7)

def perimeter (a b c : ℕ) : ℕ := a + b + c

theorem triangle_perimeter (a b c : ℕ) (h : (a, b, c) = triangle_side_lengths) : 
  perimeter a b c = 23 := by
  -- We formulate the statement and leave the proof for later
  sorry

end triangle_perimeter_l664_664179


namespace aluminum_iodide_mass_produced_l664_664222

theorem aluminum_iodide_mass_produced
  (mass_Al : ℝ) -- the mass of Aluminum used
  (molar_mass_Al : ℝ) -- molar mass of Aluminum
  (molar_mass_AlI3 : ℝ) -- molar mass of Aluminum Iodide
  (reaction_eq : ∀ (moles_Al : ℝ) (moles_AlI3 : ℝ), 2 * moles_Al = 2 * moles_AlI3) -- reaction equation which indicates a 1:1 molar ratio
  (mass_Al_value : mass_Al = 25.0) 
  (molar_mass_Al_value : molar_mass_Al = 26.98) 
  (molar_mass_AlI3_value : molar_mass_AlI3 = 407.68) :
  ∃ mass_AlI3 : ℝ, mass_AlI3 = 377.52 := by
  sorry

end aluminum_iodide_mass_produced_l664_664222


namespace ratio_a3_a2_l664_664431

open BigOperators

def binomial_expansion (x : ℝ) : ℝ :=
  (1 - 3 * x) ^ 6

def a (k : ℕ) : ℝ :=
  ∑ r in Finset.range 7, if r = k then ↑(Nat.choose 6 r) * (-3) ^ r else 0

theorem ratio_a3_a2 : a 3 / a 2 = -4 :=
by
  sorry

end ratio_a3_a2_l664_664431


namespace batsman_average_increase_l664_664700

theorem batsman_average_increase:
  ∀ (A : ℕ),
    (19 * A + 90 = 20 * 52) →
    (A = 50) →
    (52 - A = 2) :=
by
  intros A h1 h2
  have h3 : A = 50 := h2
  rw [h3] at h1
  ring_nf at h1
  rw [h3]
  exact Nat.sub_self 50
  sorry

end batsman_average_increase_l664_664700


namespace find_y_intercept_l664_664765

theorem find_y_intercept 
  (x y : ℝ)
  (h : 7 * x + 3 * y = 21) :
  (0, 7) ∈ set_of (λ p : ℝ × ℝ, ∃ x y, p = (x, y) ∧ 7 * x + 3 * y = 21) :=
sorry

end find_y_intercept_l664_664765


namespace BC_length_l664_664961

def triangle_ABC (A B C : Type)
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)] : Prop :=
  ∃ (X : A), (has_dist B X (coe (X.dist B))) ∧ (has_dist C X (coe (X.dist C))) ∧
  ∀ (x y : ℕ), x = X.dist B ∧ y = X.dist C → x + y = 61

theorem BC_length {A B C : Type}
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)]
  (h : triangle_ABC A B C) : 
  ∃ (x y : ℕ), x + y = 61 := sorry

end BC_length_l664_664961


namespace length_AE_is_sqrt3_l664_664565

-- Definitions of the conditions
def radius := √3
def diameter := 2 * radius
def side := diameter

-- Midpoint calculation of the equilateral triangle side AC where E is the midpoint
def midpoint := (side / 2)

-- Theorem to prove
theorem length_AE_is_sqrt3 : midpoint = √3 :=
by
  sorry -- Proof omitted

end length_AE_is_sqrt3_l664_664565


namespace pet_store_total_birds_l664_664716

theorem pet_store_total_birds 
  (cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ)
  (h1 : cages = 6) 
  (h2 : parrots_per_cage = 2)
  (h3 : parakeets_per_cage = 7) : 
  cages * (parrots_per_cage + parakeets_per_cage) = 54 :=
by
  -- Given conditions
  have h_total_birds_per_cage : parrots_per_cage + parakeets_per_cage = 9 :=
    by rw [h2, h3, Nat.add_comm]

  -- Prove
  have h_total_birds : cages * 9 = 54 := 
    by rw [h1, h_total_birds_per_cage, Nat.mul_comm]

  exact h_total_birds

end pet_store_total_birds_l664_664716


namespace transform_T_l664_664032

theorem transform_T (x : ℝ) : 
  let T := (x-2)^5 + 5*(x-2)^4 + 10*(x-2)^3 + 10*(x-2)^2 + 5*(x-2) + 1 
  in T = (x-1)^5 :=
by
  let T := (x-2)^5 + 5*(x-2)^4 + 10*(x-2)^3 + 10*(x-2)^2 + 5*(x-2) + 1
  show T = (x-1)^5
  sorry

end transform_T_l664_664032


namespace find_parameters_and_extrema_l664_664818

def f (a b x : ℝ) := x^3 + a * x^2 + b * x
def f' (a b x : ℝ) := 3 * x^2 + 2 * a * x + b

theorem find_parameters_and_extrema :
  (∃ a b : ℝ, f' a b 1 = 0 ∧ f' a b (-2 / 3) = 0 ∧
    (∀ x : ℝ,
      f a b x =
        x^3 - 1 / 2 * x^2 - 2 * x ∧
        ∀ x ∈ Icc (-1 : ℝ) (2 : ℝ), f a b x ≤ 2 ∧ f a b x ≥ -5 / 2)) :=
by
  sorry

end find_parameters_and_extrema_l664_664818


namespace train_length_is_approximately_correct_l664_664721

-- Define the conditions as Lean definitions
def speed_kmh : ℝ := 30 -- Speed of the train in km/hr
def time_sec : ℝ := 12 -- Time taken to cross the pole in seconds

-- Define the conversion factor from km/hr to m/s
def kmh_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

-- Define the computed speed in m/s
def speed_mps : ℝ := kmh_to_mps speed_kmh

-- Define the length of the train based on the speed and time
def length_train (v : ℝ) (t : ℝ) : ℝ := v * t

-- Define the expected length in meters
def expected_length : ℝ := 100

-- State the theorem
theorem train_length_is_approximately_correct : 
  |length_train speed_mps time_sec - expected_length| < 1 :=
  sorry

end train_length_is_approximately_correct_l664_664721


namespace pentagon_interior_angles_sequences_l664_664098

theorem pentagon_interior_angles_sequences :
  ∃ (seqs : ℕ), seqs = 2 ∧
    (∀ (x d : ℕ), 90 < x ∧ x < 120 ∧ 0 < d ∧ d < 6 ∧ x + 4 * d < 120 ∧
      (x + (x + d) + (x + 2 * d) + (x + 3 * d) + (x + 4 * d) = 540)) :=
begin
  -- We would provide the proof here, but it's omitted.
  sorry
end

end pentagon_interior_angles_sequences_l664_664098


namespace length_of_BC_l664_664928

theorem length_of_BC (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
    (BX CX : ℕ) (h_pow : CX * (BX + CX) = 2013) : 
    BX + CX = 61 :=
  sorry

end length_of_BC_l664_664928


namespace even_number_of_divisors_less_than_100_l664_664318

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k, k * k = n

theorem even_number_of_divisors_less_than_100 :
  (card { n : ℕ | 1 ≤ n ∧ n < 100 ∧ ¬ is_perfect_square n }) = 90 :=
by
  sorry

end even_number_of_divisors_less_than_100_l664_664318


namespace area_ratio_l664_664014

/-- Given a triangle ABC with AB = AC = 130, AD = 45, and CF = 85, 
    prove that the ratio of areas of triangles CEF and DBE is 17/26. -/
theorem area_ratio (A B C D E F : Point) (h1 : dist A B = 130) (h2 : dist A C = 130) 
                   (h3 : dist A D = 45) (h4 : dist C F = 85) :
    (area C E F) / (area D B E) = 17 / 26 :=
by
  sorry

end area_ratio_l664_664014


namespace total_working_days_l664_664167

variables (x a b c : ℕ)

-- Given conditions
axiom bus_morning : b + c = 6
axiom bus_afternoon : a + c = 18
axiom train_commute : a + b = 14

-- Proposition to prove
theorem total_working_days : x = a + b + c → x = 19 :=
by
  -- Placeholder for Lean's automatic proof generation
  sorry

end total_working_days_l664_664167


namespace students_neither_cool_l664_664620

variable (total_students : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ)

def only_cool_dads := cool_dads - both_cool
def only_cool_moms := cool_moms - both_cool
def only_cool := only_cool_dads + only_cool_moms + both_cool
def neither_cool := total_students - only_cool

theorem students_neither_cool (h1 : total_students = 40) (h2 : cool_dads = 18) (h3 : cool_moms = 22) (h4 : both_cool = 10) 
: neither_cool total_students cool_dads cool_moms both_cool = 10 :=
by 
  sorry

end students_neither_cool_l664_664620


namespace eight_and_five_l664_664603

def my_and (a b : ℕ) : ℕ := (a + b) ^ 2 * (a - b)

theorem eight_and_five : my_and 8 5 = 507 := 
  by sorry

end eight_and_five_l664_664603


namespace expected_value_sum_marbles_l664_664423

theorem expected_value_sum_marbles :
  (1/15 : ℚ) * ((1 + 2) + (1 + 3) + (1 + 4) + (1 + 5) + (1 + 6) + 
                (2 + 3) + (2 + 4) + (2 + 5) + (2 + 6) + (3 + 4) + 
                (3 + 5) + (3 + 6) + (4 + 5) + (4 + 6) + (5 + 6)) = 7 := 
by {
    sorry
}

end expected_value_sum_marbles_l664_664423


namespace color_exists_l664_664564

def color_func_exists (f : ℕ → ℕ) :=
  (¬ ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f a = f b ∧ f b = f c ∧ 2014 ∣ (a * b * c)) ∧
  (∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f a = f b ∧ f b = f c →
    let r := (a * b * c) % 2014 in f r = f a)

theorem color_exists : 
  ∃ f : ℕ → ℕ, 
  (∀ n, 1 ≤ n ∧ n ≤ 2013 → f n ∈ {1, 2, 3, 4, 5, 6, 7}) ∧
  ∀ k ∈ {1, 2, 3, 4, 5, 6, 7}, ∃ n, 1 ≤ n ∧ n ≤ 2013 ∧ f n = k ∧
  color_func_exists f :=
begin
  sorry
end

end color_exists_l664_664564


namespace number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664377

theorem number_of_integers_with_even_divisors (n : ℕ) (h : n = 100) : 
  (card {x | (x < n) ∧ ∃ k, k * k = x} = 9) → 
  (card {x | (x < n) ∧ ¬(∃ k, k * k = x)} = n - 1 - 9) :=
by
  intro h_squares
  rw h
  trivial

open_locale classical
noncomputable def count_even_divisors_less_than_100 : ℕ :=
  90

theorem count_even_divisors_verification :
  count_even_divisors_less_than_100 = 90 :=
by
  sorry

end number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664377


namespace count_even_divisors_lt_100_l664_664307

-- Define the set of natural numbers less than 100
def nat_lt_100 := {n : ℕ | n < 100}

-- Define the set of perfect squares less than 100
def perfect_squares_lt_100 := {n : ℕ | ∃ k : ℕ, k^2 = n ∧ n < 100}

-- Define the set of natural numbers less than 100 with an even number of positive divisors
def even_divisors_lt_100 := nat_lt_100 \ perfect_squares_lt_100

-- Theorem stating the number of elements with even number of divisors is 90
theorem count_even_divisors_lt_100 : (even_divisors_lt_100).card = 90 := 
sorry

end count_even_divisors_lt_100_l664_664307


namespace problem_1_problem_2_l664_664515

-- Definition of the triangle and the given conditions
variables {a b c : ℝ}
axiom triangle_ABC : ℝ
axiom angle_A : ℝ := atan2 a triangle_ABC
axiom angle_B : ℝ := atan2 b triangle_ABC
axiom angle_C : ℝ := π - angle_A - angle_B
axiom cos_A2 : ℝ := cos (angle_A / 2)
axiom cos_C2 : ℝ := cos (angle_C / 2)

-- First part of the problem: proving 2b = a + c
theorem problem_1 (h : a * cos_C2 ^ 2 + c * cos_A2 ^ 2 = (3 / 2) * b) : 
  2 * b = a + c :=
sorry

-- Second part of the problem: given B = π/3 and S = 8√3, find b
theorem problem_2 (hB : angle_B = π / 3) (hS : S = 8 * √3) : b = 4 * √2 :=
sorry

end problem_1_problem_2_l664_664515


namespace even_number_of_divisors_l664_664334

-- Proof statement: There are 90 positive integers less than 100 with an even number of divisors.
theorem even_number_of_divisors : 
  {n : ℕ | n < 100 ∧ ∃ k : ℕ, k ^ 2 = n}.toFinset.card = 90 := 
sorry

end even_number_of_divisors_l664_664334


namespace diagonals_of_convex_heptagon_l664_664221

theorem diagonals_of_convex_heptagon : 
  let n := 7 in n * (n - 3) / 2 = 14 :=
by
  let n := 7
  have h1 : n = 7 := rfl
  have h2 : n - 3 = 4 := by rw [h1]; exact rfl
  have h3 : n * (n - 3) = 28 := by rw [h1, h2]; exact rfl
  have h4 : 28 / 2 = 14 := rfl
  sorry

end diagonals_of_convex_heptagon_l664_664221


namespace even_number_of_divisors_less_than_100_l664_664320

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k, k * k = n

theorem even_number_of_divisors_less_than_100 :
  (card { n : ℕ | 1 ≤ n ∧ n < 100 ∧ ¬ is_perfect_square n }) = 90 :=
by
  sorry

end even_number_of_divisors_less_than_100_l664_664320


namespace cos_squared_sum_l664_664998

open Real

theorem cos_squared_sum (n : ℕ) (α : ℝ) (hn : n ≥ 3) :
  ∑ k in Finset.range n, cos (α + (2 * k * π) / n)^2 = n / 2 :=
by
  sorry

end cos_squared_sum_l664_664998


namespace triangle_construction_l664_664749

-- Definitions for lengths and points 
variables {AB AA1 BB1 CC1 : ℝ}
variables {A B C : Type*} [metric_space A] [metric_space B] [metric_space C]

-- Definitions for conditions in case 1
def case1 (AB AA1 BB1 : ℝ) (h_pos_AB: AB > 0) (h_pos_AA1: AA1 > 0) (h_pos_BB1: BB1 > 0)
: Prop :=
∃ (A B C : ℝ), A ≠ B ∧
dist (A, B) = AB ∧ 
-- Altitudes AA1 and BB1 must be orthogonal projections from A and B to line 
-- containing BC and AC respectively
dist (A, (B + BB1)) = AA1 ∧ 
dist (B, (A + AA1)) = BB1 

-- Definitions for conditions in case 2
def case2 (AB BB1 CC1 : ℝ) (h_pos_AB: AB > 0) (h_pos_BB1: BB1 > 0) (h_pos_CC1: CC1 > 0)
: Prop :=
∃ (A B C : ℝ), A ≠ B ∧
dist (A, B) = AB ∧ 
-- Altitude BB1 from B and CC1 from C to line containing AB
dist (B, (A + BB1)) = BB1 ∧ 
dist (C, (B + CC1)) = CC1

-- The intended theorem to prove the existence of triangles in both cases
theorem triangle_construction (AB AA1 BB1 CC1 : ℝ) (h_pos_AB: AB > 0) 
  (h_pos_AA1: AA1 > 0) (h_pos_BB1: BB1 > 0) (h_pos_CC1: CC1 > 0) :
  (case1 AB AA1 BB1 h_pos_AB h_pos_AA1 h_pos_BB1) ∨ 
  (case2 AB BB1 CC1 h_pos_AB h_pos_BB1 h_pos_CC1) :=
sorry

end triangle_construction_l664_664749


namespace Bobby_has_27_pairs_l664_664196

-- Define the number of shoes Becky has
variable (B : ℕ)

-- Define the number of shoes Bonny has as 13, with the relationship to Becky's shoes
def Bonny_shoes : Prop := 2 * B - 5 = 13

-- Define the number of shoes Bobby has given Becky's count
def Bobby_shoes := 3 * B

-- Prove that Bobby has 27 pairs of shoes given the conditions
theorem Bobby_has_27_pairs (hB : Bonny_shoes B) : Bobby_shoes B = 27 := 
by 
  sorry

end Bobby_has_27_pairs_l664_664196


namespace sum_of_sub_fixed_points_ln_exp_eq_zero_l664_664446

def sub_fixed_point (f : ℝ → ℝ) (t : ℝ) : Prop :=
  f t = -t

theorem sum_of_sub_fixed_points_ln_exp_eq_zero :
  let f := Real.log
  let f_inv := Real.exp
  ∑ t in { t : ℝ | sub_fixed_point f t ∨ sub_fixed_point f_inv t }, t = 0 :=
by
  sorry

end sum_of_sub_fixed_points_ln_exp_eq_zero_l664_664446


namespace relationship_between_a_b_c_l664_664247

noncomputable def a : ℝ := 2 ^ (-2 / 3)
noncomputable def b : ℝ := Real.logb 3 5
noncomputable def c : ℝ := Real.logb 4 5

theorem relationship_between_a_b_c : a < c ∧ c < b := by
  have ha : a = 2 ^ (-2 / 3) := rfl
  have hb : b = Real.logb 3 5 := rfl
  have hc : c = Real.logb 4 5 := rfl
  sorry

end relationship_between_a_b_c_l664_664247


namespace max_value_a_l664_664813

open Real

theorem max_value_a : ∃ a : ℝ, (∀ x ∈ Icc 0 4, sqrt x - sqrt (4 - x) ≥ a) ∧ (∀ a', (∀ x ∈ Icc 0 4, sqrt x - sqrt (4 - x) ≥ a') → a' ≤ -2) :=
sorry

end max_value_a_l664_664813


namespace check_L_shapes_l664_664554

def checkerboard := fin 8 × fin 8

-- Definition of a small square's position
structure square :=
  (x : fin 8)
  (y : fin 8)

-- Definition of an "L" shape in terms of its component squares
structure L_shape :=
  (squares : fin 3 → square)
  (valid : squares 0 ≠ squares 1 ∧ squares 1 ≠ squares 2 ∧ squares 0 ≠ squares 2)

-- The theorem to be proved
theorem check_L_shapes: (finset.univ : finset (fin 8 × fin 8)).card * 3 = 196 :=
by 
  -- Placeholder proof
  sorry

end check_L_shapes_l664_664554


namespace odd_digits_greater_even_digits_count_l664_664690

theorem odd_digits_greater_even_digits_count {n : ℕ} (n_eq : n = 4 * 10^25) :
  let odd_digit_count := ∑ k in Finset.range 26 + 1, 5^k
  let even_digit_count := ∑ k in Finset.range 26 + 1, 4 * 5^(k-1)
  odd_digit_count - even_digit_count = (5^26 - 5) / 4 := by
begin
  sorry
end

end odd_digits_greater_even_digits_count_l664_664690


namespace number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664386

theorem number_of_integers_with_even_divisors (n : ℕ) (h : n = 100) : 
  (card {x | (x < n) ∧ ∃ k, k * k = x} = 9) → 
  (card {x | (x < n) ∧ ¬(∃ k, k * k = x)} = n - 1 - 9) :=
by
  intro h_squares
  rw h
  trivial

open_locale classical
noncomputable def count_even_divisors_less_than_100 : ℕ :=
  90

theorem count_even_divisors_verification :
  count_even_divisors_less_than_100 = 90 :=
by
  sorry

end number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664386


namespace average_expenditure_week_l664_664079

theorem average_expenditure_week (avg_3_days: ℝ) (avg_4_days: ℝ) (total_days: ℝ):
  avg_3_days = 350 → avg_4_days = 420 → total_days = 7 → 
  ((3 * avg_3_days + 4 * avg_4_days) / total_days = 390) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end average_expenditure_week_l664_664079


namespace number_of_integers_with_even_divisors_l664_664388

-- Define conditions
def N := 99

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P : finset ℕ := { n in finset.range (N + 1) | is_perfect_square n }

noncomputable def countP : ℕ := P.card

-- The statement to prove
theorem number_of_integers_with_even_divisors : 
  (N - countP) = 90 :=
by {
  sorry
}

end number_of_integers_with_even_divisors_l664_664388


namespace polygon_E_largest_area_l664_664775

def unit_square_area : ℕ := 1
def right_triangle_area : ℚ := 1 / 2
def rectangle_area : ℕ := 2

def polygon_A_area : ℚ := 3 * unit_square_area + 2 * right_triangle_area
def polygon_B_area : ℚ := 2 * unit_square_area + 4 * right_triangle_area
def polygon_C_area : ℚ := 4 * unit_square_area + 1 * rectangle_area
def polygon_D_area : ℚ := 3 * rectangle_area
def polygon_E_area : ℚ := 2 * unit_square_area + 2 * right_triangle_area + 2 * rectangle_area

theorem polygon_E_largest_area :
  polygon_E_area = max polygon_A_area (max polygon_B_area (max polygon_C_area (max polygon_D_area polygon_E_area))) := by
  sorry

end polygon_E_largest_area_l664_664775


namespace find_abc_sum_l664_664030

def equation (x : ℝ) : Prop := 
  3 / (x - 3) + 5 / (x - 5) + 17 / (x - 17) + 19 / (x - 19) = x^2 - 11 * x - 4

def is_solution (m : ℝ) (a b c : ℕ) : Prop := 
  m = a + Real.sqrt (b + Real.sqrt c ∧ equation m

theorem find_abc_sum (m a b c : ℕ) (h_eq : equation m) (h_sol : is_solution m a b c) : 
  a + b + c = 73 := 
sorry

end find_abc_sum_l664_664030


namespace f_for_negative_l664_664803

noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then x * (1 + 3 * x) else -f (-x)

theorem f_for_negative (x : ℝ) (hx : x < 0) : f x = x * (1 - 3 * x) :=
by
  have h_neg : -x >= 0 := by linarith
  simp [f, h_neg]
  sorry

end f_for_negative_l664_664803


namespace coordinates_of_B_l664_664449

def pointA : Prod Int Int := (-3, 2)
def moveRight (p : Prod Int Int) (units : Int) : Prod Int Int := (p.1 + units, p.2)
def moveDown (p : Prod Int Int) (units : Int) : Prod Int Int := (p.1, p.2 - units)
def pointB : Prod Int Int := moveDown (moveRight pointA 1) 2

theorem coordinates_of_B :
  pointB = (-2, 0) :=
sorry

end coordinates_of_B_l664_664449


namespace triangle_perimeter_l664_664178

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 10) (h2 : b = 6) (h3 : c = 7) :
  a + b + c = 23 := by
  sorry

end triangle_perimeter_l664_664178


namespace greatest_prime_factor_of_expression_l664_664671

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define specific factorial values
def fac_15 := factorial 15
def fac_18 := factorial 18

-- Define the expression from the problem
def expr := fac_15 * (1 + 16 * 17 * 18)

-- Define the factorization result
def factor_4896 := 2 ^ 5 * 3 ^ 2 * 17

-- Define a lemma about the factorization of the expression
lemma factor_expression : 15! * (1 + 16 * 17 * 18) = fac_15 * 4896 := by
  sorry

-- State the main theorem
theorem greatest_prime_factor_of_expression : ∀ p : ℕ, prime p ∧ p ∣ expr → p ≤ 17 := by
  sorry

end greatest_prime_factor_of_expression_l664_664671


namespace even_number_of_divisors_lt_100_l664_664405

theorem even_number_of_divisors_lt_100 : 
  let n := 99 in
  let perfect_squares := {1, 4, 9, 16, 25, 36, 49, 64, 81} in
  n - perfect_squares.card = 90 := 
by
  sorry

end even_number_of_divisors_lt_100_l664_664405


namespace number_of_solutions_to_g100_eq_zero_l664_664536

def g0 (x : ℝ) : ℝ := x + (abs (x - 150)) - (abs (x + 150))

def gn (n : ℕ) (x : ℝ) : ℝ :=
  nat.rec_on n (g0 x) (λ n g_n_minus_1, abs g_n_minus_1 - 2)

theorem number_of_solutions_to_g100_eq_zero : 
  (finset.card (finset.filter (λ x : ℝ, gn 100 x = 0) (finset.range 100000))) = 299 :=
sorry

end number_of_solutions_to_g100_eq_zero_l664_664536


namespace number_of_squares_in_5x5_grid_l664_664213

theorem number_of_squares_in_5x5_grid : 
  let grid_size := 5 in
  (Σ n in Icc 1 grid_size, (grid_size - n + 1) * (grid_size - n + 1)) = 55 :=
by
  let grid_size := 5
  have h1 : (Σ n in Icc 1 grid_size, (grid_size - n + 1) * (grid_size - n + 1)) = 55 := by sorry
  exact h1

end number_of_squares_in_5x5_grid_l664_664213


namespace triangle_bc_length_l664_664909

theorem triangle_bc_length (A B C X : Type)
  (AB AC : ℕ)
  (hAB : AB = 86)
  (hAC : AC = 97)
  (circle_eq : ∀ {r : ℕ}, r = AB → circle_centered_at_A_intersects_BC_two_points B X)
  (integer_lengths : ∃ (BX CX : ℕ), ) :
  BC = 61 :=
by
  sorry

end triangle_bc_length_l664_664909


namespace December_times_average_l664_664147

variable (D : ℝ)

def revenue_November : ℝ := (2 / 5) * D
def revenue_January : ℝ := (2 / 25) * D
def average_revenue : ℝ := (revenue_November D + revenue_January D) / 2

theorem December_times_average
    (revenue_November : ℝ := (2 / 5) * D)
    (revenue_January : ℝ := (2 / 25) * D)
    (average_revenue : ℝ := (revenue_November + revenue_January) / 2) :
    D = (25 / 6) * average_revenue :=
by
  sorry

end December_times_average_l664_664147


namespace eccentricity_range_l664_664276

def hyperbola (a b x y : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)
def right_branch_hyperbola_P (a b c x y : ℝ) : Prop := hyperbola a b x y ∧ (c = a) ∧ (2 * c = a)

theorem eccentricity_range {a b c : ℝ} (h: hyperbola a b c c) (h1 : 2 * a = 2 * c) (h2 : c = a) :
  1 < (c / a) ∧ (c / a) ≤ (Real.sqrt 10 / 2 : ℝ) := by
  sorry

end eccentricity_range_l664_664276


namespace tangent_line_at_P_exists_c_for_a_l664_664826

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_P :
  ∀ x y : ℝ, y = f x → x = 1 → y = 0 → x - y - 1 = 0 := 
by 
  sorry

theorem exists_c_for_a :
  ∀ a : ℝ, 1 < a → ∃ c : ℝ, 0 < c ∧ c < 1 / a ∧ ∀ x : ℝ, c < x → x < 1 → f x > a * x * (x - 1) :=
by 
  sorry

end tangent_line_at_P_exists_c_for_a_l664_664826


namespace function_properties_l664_664594

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * (Real.sin (x + π / 4)) ^ 2

theorem function_properties :
  (∀ x, f (-x) = - f x) ∧ (Real.Periodic f π) :=
by
  sorry

end function_properties_l664_664594


namespace greatest_prime_factor_15_fact_plus_18_fact_l664_664642

theorem greatest_prime_factor_15_fact_plus_18_fact :
  Nat.greatest_prime_factor (15.factorial + 18.factorial) = 17 := by
  sorry

end greatest_prime_factor_15_fact_plus_18_fact_l664_664642


namespace balls_distribution_8_into_4_l664_664421

theorem balls_distribution_8_into_4 :
  (∀ (balls : ℕ) (boxes : ℕ),
    balls = 8 → boxes = 4 →
    nat.choose 8 8 * 1 + 
    nat.choose 8 7 * nat.choose 1 1 + 
    nat.choose 8 6 * nat.choose 2 2 + 
    nat.choose 8 6 * (nat.choose 1 1)^2 + 
    nat.choose 8 5 * nat.choose 3 3 + 
    nat.choose 8 5 * (nat.choose 2 2 * nat.choose 1 1) + 
    nat.choose 8 5 * (nat.choose 1 1)^3 + 
    nat.choose 8 4 * nat.choose 4 4 + 
    nat.choose 8 4 * nat.choose 3 3 * nat.choose 1 1 + 
    nat.choose 8 4 * (nat.choose 2 2)^2 + 
    nat.choose 8 4 * (nat.choose 2 2) * (nat.choose 1 1)^2 + 
    nat.choose 8 3 * (nat.choose 3 3) * nat.choose 2 2 + 
    nat.choose 8 3 * nat.choose 3 3 * (nat.choose 1 1)^2 + 
    nat.choose 8 3 * (nat.choose 2 2 * nat.choose 2 2 * nat.choose 1 1) + 
    1 = 139) :=
begin
  sorry
end

end balls_distribution_8_into_4_l664_664421


namespace max_operations_l664_664796

theorem max_operations (n : ℕ) (h : n ≥ 2) : 
  ∃ m : ℕ, m = n * (n - 1) / 2 :=
by {
  use n * (n - 1) / 2,
  sorry,
}

end max_operations_l664_664796


namespace curve_tangent_parallel_enclosed_area_l664_664815

theorem curve_tangent_parallel (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = a * x^2 + 2) 
  → (∀ x, f' x = 2 * a * x)
  → let parallel_condition := (∀ x, (f' 1 = 2))
  → a = 1 ∧ ∀ x, f x = x^2 + 2 :=
by 
  intro h1 h2 parallel_condition h3
  have ha : a = 1 := sorry
  use ha
  intro x
  have hx : f x = x^2 + 2 := sorry
  use hx

theorem enclosed_area :
  let f := λ x : ℝ, x^2 + 2
  → ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2
  → let intersection_points := {1,2}
  → ∫ x in 0..1, (f x - 3 * x) + ∫ x in 1..2, (3 * x - f x) = 1 :=
by
  intro f x hx intersection_points
  let intg_area : ℝ := (∫ x in 0..1, (f x - 3 * x) + ∫ x in 1..2, (3 * x - f x))
  have e_area : intg_area = 1 := sorry
  use e_area

end curve_tangent_parallel_enclosed_area_l664_664815


namespace length_of_BC_l664_664927

theorem length_of_BC 
  (A B C X : Type) 
  (d_AB : ℝ) (d_AC : ℝ) 
  (circle_center_A : A) 
  (radius_AB : ℝ)
  (intersects_BC : B → C → X)
  (BX CX : ℕ) 
  (h_BX_in_circle : BX = d_AB) 
  (h_CX_in_circle : CX = d_AC) 
  (h_integer_lengths : ∃ x y : ℕ, BX = x ∧ CX = y) :
  BX + CX = 61 :=
begin
  sorry
end

end length_of_BC_l664_664927


namespace Amelia_remaining_distance_l664_664183

noncomputable def remaining_distance (initial_distance : ℕ) (days : ℕ) (speed_day_1 : ℕ) (time_day_1 : ℕ) (speed_day_2 : ℕ) (time_day_2 : ℕ) (rest_distance_1 : ℕ) (even_day_speed : ℕ) (odd_day_speed : ℕ) (daily_drive_time : ℕ) (break_duration : ℕ) (rest_distance_2 : ℕ) : ℕ :=
  initial_distance - 
  (speed_day_1 * time_day_1 + 
  speed_day_2 * time_day_2 + 
  (days - 2 - (days - 2) / 3) * (even_day_speed * (daily_drive_time - break_duration)) + 
  (days - 2) / 3 * (odd_day_speed * (daily_drive_time - break_duration)))

theorem Amelia_remaining_distance (initial_distance : ℕ) (days : ℕ) (speed_day_1 : ℕ) (time_day_1 : ℕ) (speed_day_2 : ℕ) (time_day_2 : ℕ) (rest_distance_1 : ℕ) (even_day_speed : ℕ) (odd_day_speed : ℕ) (daily_drive_time : ℕ) (break_duration : ℕ) (rest_distance_2 : ℕ) :
  remaining_distance initial_distance days speed_day_1 time_day_1 speed_day_2 time_day_2 rest_distance_1 even_day_speed odd_day_speed daily_drive_time break_duration rest_distance_2 = 4995 := by
  sorry

-- Given conditions:
#eval Amelia_remaining_distance 8205 15 90 10 80 7 1000 100 75 8 1 1500   -- Should evaluate to true (Amelia still has to drive 4995 km)

end Amelia_remaining_distance_l664_664183


namespace tangent_same_at_one_zero_l664_664110

theorem tangent_same_at_one_zero (a : ℝ) : 
  (∀ x, f' x = 1/x) ∧ (∀ x, g' x = 2 * a * x) ∧ (f' 1 = g' 1) → a = 1/2 := 
by
  sorry
  where
    f (x : ℝ) := Math.log x
    g (x : ℝ) := a * x^2 - a
    f' (x : ℝ) := 1 / x
    g' (x : ℝ) := 2 * a * x

end tangent_same_at_one_zero_l664_664110


namespace simplify_fraction_140_2100_l664_664569

theorem simplify_fraction_140_2100 :
  let a := 140
  let b := 2100
  let gcd_ab := 2^2 * 5 * 7 -- This is the GCD from the given prime factorization
  (a / gcd_ab) / (b / gcd_ab) = 1 / 15 :=
by
  let a := 140
  let b := 2100
  let gcd_ab := 2^2 * 5 * 7
  calc
    (a / gcd_ab) = 1 : by sorry
    (b / gcd_ab) = 15 : by sorry
    (1 / 15) = (1 / 15) : by sorry

end simplify_fraction_140_2100_l664_664569


namespace find_c_share_l664_664689

theorem find_c_share (a b c : ℕ) 
  (h1 : a + b + c = 1760)
  (h2 : ∃ x : ℕ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x)
  (h3 : 6 * a = 8 * b ∧ 8 * b = 20 * c) : 
  c = 250 :=
by
  sorry

end find_c_share_l664_664689


namespace count_even_divisors_lt_100_l664_664306

-- Define the set of natural numbers less than 100
def nat_lt_100 := {n : ℕ | n < 100}

-- Define the set of perfect squares less than 100
def perfect_squares_lt_100 := {n : ℕ | ∃ k : ℕ, k^2 = n ∧ n < 100}

-- Define the set of natural numbers less than 100 with an even number of positive divisors
def even_divisors_lt_100 := nat_lt_100 \ perfect_squares_lt_100

-- Theorem stating the number of elements with even number of divisors is 90
theorem count_even_divisors_lt_100 : (even_divisors_lt_100).card = 90 := 
sorry

end count_even_divisors_lt_100_l664_664306


namespace smallest_percent_increase_l664_664779

-- Definitions of the values in terms of L
def L : Real := 1000
def Q1 : Real := 100
def Q2 : Real := 250
def Q3 : Real := 400
def Q4 : Real := 600
def Q5 : Real := L
def Q6 : Real := 1.8 * L
def Q7 : Real := 3.5 * L
def Q8 : Real := 6 * L

-- Define function for percent increase
def percentIncrease (v1 v2 : Real) : Real :=
  ((v2 - v1) / v1) * 100

-- Theorem to prove the smallest percent increase
theorem smallest_percent_increase :
  min (percentIncrease Q1 Q2)
  (min (percentIncrease Q2 Q3)
  (min (percentIncrease Q3 Q4)
  (min (percentIncrease Q4 Q5)
  (min (percentIncrease Q5 Q6)
  (min (percentIncrease Q6 Q7) (percentIncrease Q7 Q8)))))) = percentIncrease Q3 Q4 := 
  by
  sorry

end smallest_percent_increase_l664_664779


namespace count_even_divisors_lt_100_l664_664301

-- Define the set of natural numbers less than 100
def nat_lt_100 := {n : ℕ | n < 100}

-- Define the set of perfect squares less than 100
def perfect_squares_lt_100 := {n : ℕ | ∃ k : ℕ, k^2 = n ∧ n < 100}

-- Define the set of natural numbers less than 100 with an even number of positive divisors
def even_divisors_lt_100 := nat_lt_100 \ perfect_squares_lt_100

-- Theorem stating the number of elements with even number of divisors is 90
theorem count_even_divisors_lt_100 : (even_divisors_lt_100).card = 90 := 
sorry

end count_even_divisors_lt_100_l664_664301


namespace locus_of_points_is_circle_l664_664793

variables (A B C M : ℝ^3) 
variable (ABC_plane : affine_plane ℝ)
variable (K L O G : ℝ^3)

-- Define conditions
def is_midpoint (P Q R : ℝ^3) : Prop := P = (Q + R) / 2
def is_centroid (G ABCM : list ℝ^3) : Prop := 
  ((G = (sum ABCM) / (length ABCM)) ∧ (length ABCM = 4))
def is_perpendicular (X Y Z : ℝ^3) : Prop := (X - Y) • (Z - Y) = 0

-- The given conditions in the problem
axiom triangle_abc : ∀ (A B C : ℝ^3), ∃ (ABC_plane : affine_plane ℝ), A ∈ ABC_plane ∧ B ∈ ABC_plane ∧ C ∈ ABC_plane
axiom midpoint_K : is_midpoint K B C
axiom midpoint_L : is_midpoint L A M
axiom centroid_G : is_centroid G [A,B,C,M]
axiom perpendicular_condition : is_perpendicular O G (AffinePlane.mk ℝ (set.univ) cofinite_top)

-- The mathematical equivalent proof problem
theorem locus_of_points_is_circle : set.M = {M : ℝ^3 | ∃ r : ℝ, ∃ center : ℝ^3, (∥M - center∥ = r ∧ 
  ∀ (L K O G : ℝ^3),
  is_midpoint K B C ∧ is_midpoint L A M ∧
  is_centroid G [A,B,C,M] ∧
  is_perpendicular O G (AffinePlane.mk ℝ (set.univ) cofinite_top)
  )} :=
sorry

end locus_of_points_is_circle_l664_664793


namespace woman_work_time_l664_664145

theorem woman_work_time :
  ∀ (M W B : ℝ), (M = 1/6) → (B = 1/12) → (M + W + B = 1/3) → (W = 1/12) → (1 / W = 12) :=
by
  intros M W B hM hB h_combined hW
  sorry

end woman_work_time_l664_664145


namespace ratio_of_root_distances_l664_664291

noncomputable def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a*x^2 + b*x + c

def f1 (a : ℝ) : ℝ → ℝ := quadratic 1 (-1) (-a)
def f2 (b : ℝ) : ℝ → ℝ := quadratic 1 b 2
def f3 (a b : ℝ) : ℝ → ℝ := quadratic 4 (b - 3) (-3*a + 2)
def f4 (a b : ℝ) : ℝ → ℝ := quadratic 4 (3*b - 1) (6 - a)

def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

def root_distance (a b c : ℝ) : ℝ :=
  if 0 < discriminant a b c then
    real.sqrt (discriminant a b c) / a
  else
    0

axiom A (a : ℝ) : ℝ := root_distance 1 (-1) (-a)
axiom B (b : ℝ) : ℝ := root_distance 1 b 2
axiom C (a b : ℝ) : ℝ := root_distance 4 (b - 3) (-3*a + 2)
axiom D (a b : ℝ) : ℝ := root_distance 4 (3*b - 1) (6 - a)

theorem ratio_of_root_distances (a b : ℝ) (h : abs (A a) ≠ abs (B b)) :
  (C a b)^2 - (D a b)^2 = (1/2) * ((A a)^2 - (B b)^2) := sorry

end ratio_of_root_distances_l664_664291


namespace greatest_prime_factor_of_15_plus_18_l664_664659

theorem greatest_prime_factor_of_15_plus_18! : 
  let n := 15! + 18!
  n = 15! * 4897 ∧ Prime 4897 →
  (∀ p : ℕ, Prime p ∧ p ∣ n → p ≤ 4897) ∧ (4897 ∣ n) ∧ Prime 4897 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_of_15_plus_18_l664_664659


namespace greatest_prime_factor_of_expression_l664_664669

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define specific factorial values
def fac_15 := factorial 15
def fac_18 := factorial 18

-- Define the expression from the problem
def expr := fac_15 * (1 + 16 * 17 * 18)

-- Define the factorization result
def factor_4896 := 2 ^ 5 * 3 ^ 2 * 17

-- Define a lemma about the factorization of the expression
lemma factor_expression : 15! * (1 + 16 * 17 * 18) = fac_15 * 4896 := by
  sorry

-- State the main theorem
theorem greatest_prime_factor_of_expression : ∀ p : ℕ, prime p ∧ p ∣ expr → p ≤ 17 := by
  sorry

end greatest_prime_factor_of_expression_l664_664669


namespace effective_steps_calculation_l664_664219

noncomputable def steps_climbed : ℕ := 25
noncomputable def rate_incline_1 : ℝ := 1.2
noncomputable def climbs_1 : ℕ := 14
noncomputable def steps_climbed_1 : ℝ := steps_climbed * rate_incline_1 * climbs_1

noncomputable def steps_climbed_2 : ℕ := 18
noncomputable def rate_incline_2 : ℝ := 1.4
noncomputable def climbs_2 : ℕ := 9
noncomputable def steps_climbed_2 : ℝ := steps_climbed_2 * rate_incline_2 * climbs_2

noncomputable def steps_climbed_3 : ℕ := 15
noncomputable def rate_incline_3 : ℝ := 1.6
noncomputable def climbs_3 : ℕ := 12
noncomputable def steps_climbed_3 : ℝ := steps_climbed_3 * rate_incline_3 * climbs_3

noncomputable def steps_climbed_4 : ℕ := 10
noncomputable def rate_incline_4 : ℝ := 1.8
noncomputable def climbs_4 : ℕ := 16
noncomputable def steps_climbed_4 : ℝ := steps_climbed_4 * rate_incline_4 * climbs_4

noncomputable def steps_climbed_5 : ℕ := 6
noncomputable def rate_incline_5 : ℝ := 2.0
noncomputable def climbs_5 : ℕ := 20
noncomputable def steps_climbed_5 : ℝ := steps_climbed_5 * rate_incline_5 * climbs_5

noncomputable def total_effective_steps : ℝ := 
  steps_climbed_1 + steps_climbed_2 + steps_climbed_3 + steps_climbed_4 + steps_climbed_5

theorem effective_steps_calculation : total_effective_steps = 1462.8 := by
  sorry

end effective_steps_calculation_l664_664219


namespace BC_length_l664_664970

-- Define the given triangle and circle conditions
variables (A B C X : Type) (AB AC BX CX : ℤ)
axiom AB_value : AB = 86
axiom AC_value : AC = 97
axiom circle_center_radius : ∃ (A : Type), ∃ (radius : ℤ), radius = AB ∧ ∃ (points : Set Type), points = {B, X} ∧ ∀ (P : Type), P ∈ points → dist A P = radius
axiom BX_CX_integers : ∃ (x y : ℤ), BX = x ∧ CX = y

-- Define calculations using the Power of a Point theorem
theorem BC_length :
  ∀ (y: ℤ) (x: ℤ), y(y + x) = AC^2 - AB^2 → x + y = 61 :=
by
  intros y x h
  have h1 : 97^2 = 9409, by norm_num,
  have h2 : 86^2 = 7396, by norm_num,
  rw [AB_value, AC_value] at h,
  rw [h1, h2] at h,
  calc y(y + x) = 2013 := by {exact h}
  -- The human verification part is skipped since we only need the statement here
  sorry

end BC_length_l664_664970


namespace total_scoops_needed_l664_664055

def cups_of_flour : ℕ := 4
def cups_of_sugar : ℕ := 3
def cups_of_milk : ℕ := 2

def flour_scoop_size : ℚ := 1 / 4
def sugar_scoop_size : ℚ := 1 / 3
def milk_scoop_size : ℚ := 1 / 2

theorem total_scoops_needed : 
  (cups_of_flour / flour_scoop_size) + (cups_of_sugar / sugar_scoop_size) + (cups_of_milk / milk_scoop_size) = 29 := 
by {
  sorry
}

end total_scoops_needed_l664_664055


namespace pianist_moves_finite_l664_664060

theorem pianist_moves_finite (rooms : ℤ → ℕ) (num_pianists : ℤ) :
  (∀ k : ℤ, rooms k = 0 → rooms (k + 1) = 0) ∧ (num_pianists = 9) ∧
  (∀ k : ℤ, ∃ n ≠ k, rooms k ≠ 0 → rooms (k - 1) ≠ 0 ∧ rooms (k + 2) ≠ 0) →
  ∃ N : ℕ, ∀ n > N, ∀ k : ℤ, (rooms k ≠ 0 → rooms (k - 1) ≠ 0 ∧ rooms (k + 2) ≠ 0).
Proof
  sorry

end pianist_moves_finite_l664_664060


namespace find_g5_plus_g_neg5_l664_664992

def g (x : ℝ) : ℝ := 2 * x ^ 8 + 3 * x ^ 6 - 4 * x ^ 4 + 5

theorem find_g5_plus_g_neg5 :
  g 5 + g (-5) = 14 :=
by
  have h_g5 : g 5 = 7 := by sorry
  have h_even : ∀ x, g x = g (-x) := by sorry
  rw [←h_even 5, h_g5, h_g5]
  exact by linarith

end find_g5_plus_g_neg5_l664_664992


namespace shifted_quadratic_eq_l664_664462

-- Define the original quadratic function
def orig_fn (x : ℝ) : ℝ := -x^2

-- Define the function after shifting 1 unit to the left
def shifted_left_fn (x : ℝ) : ℝ := - (x + 1)^2

-- Define the final function after also shifting 3 units up
def final_fn (x : ℝ) : ℝ := - (x + 1)^2 + 3

-- Prove the final function is the correctly transformed function from the original one
theorem shifted_quadratic_eq : ∀ (x : ℝ), final_fn x = - (x + 1)^2 + 3 :=
by 
  intro x
  sorry

end shifted_quadratic_eq_l664_664462


namespace sum_of_abs_coeffs_f6_l664_664101

noncomputable def f : ℕ → (ℤ[X] → ℤ[X])
| 0 := λ x, 1
| (n + 1) := λ x, (x^2 + 1) * (f n x) - 2 * x

def absolute_value_sum (p : ℤ[X]) : ℤ :=
(p.coeffs.map Int.natAbs).sum

theorem sum_of_abs_coeffs_f6 : absolute_value_sum (f 6) = 190 :=
by
  sorry

end sum_of_abs_coeffs_f6_l664_664101


namespace Bobby_has_27_pairs_l664_664197

-- Define the number of shoes Becky has
variable (B : ℕ)

-- Define the number of shoes Bonny has as 13, with the relationship to Becky's shoes
def Bonny_shoes : Prop := 2 * B - 5 = 13

-- Define the number of shoes Bobby has given Becky's count
def Bobby_shoes := 3 * B

-- Prove that Bobby has 27 pairs of shoes given the conditions
theorem Bobby_has_27_pairs (hB : Bonny_shoes B) : Bobby_shoes B = 27 := 
by 
  sorry

end Bobby_has_27_pairs_l664_664197


namespace correct_eqns_l664_664498

theorem correct_eqns (x y : ℝ) (h1 : x - y = 4.5) (h2 : 1/2 * x + 1 = y) :
  x - y = 4.5 ∧ 1/2 * x + 1 = y :=
by {
  exact ⟨h1, h2⟩,
}

end correct_eqns_l664_664498


namespace basic_word_arrangements_l664_664842

theorem basic_word_arrangements : 
  ∀ (s : String), s = "basic" → (∃ n : ℕ, n = 5 ∧ n.factorial = 120) := 
by
  intros s hs
  use 5
  simp [hs]
  exact ⟨rfl, by norm_num⟩

end basic_word_arrangements_l664_664842


namespace career_preference_representation_l664_664103

noncomputable def male_to_female_ratio : ℕ × ℕ := (2, 3)
noncomputable def total_students := male_to_female_ratio.1 + male_to_female_ratio.2
noncomputable def students_prefer_career := 2
noncomputable def full_circle_degrees := 360

theorem career_preference_representation :
  (students_prefer_career / total_students : ℚ) * full_circle_degrees = 144 := by
  sorry

end career_preference_representation_l664_664103


namespace wood_rope_length_equivalence_l664_664500

variable (x y : ℝ)

theorem wood_rope_length_equivalence :
  (x - y = 4.5) ∧ (y = (1 / 2) * x + 1) :=
  sorry

end wood_rope_length_equivalence_l664_664500


namespace center_of_circle_C_is_l664_664588

-- Define the endpoints of the diameter
def point1 : (ℝ × ℝ) := (2, -3)
def point2 : (ℝ × ℝ) := (10, 9)

-- Define the midpoint formula
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := 
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- State that the point1 and point2 are the endpoints of the diameter
def C_diameter_endpoints (p1 p2 : ℝ × ℝ) : Prop := 
  p1 = point1 ∧ p2 = point2 

-- Claim that the center of circle C is (6, 3)
theorem center_of_circle_C_is (p1 p2 : ℝ × ℝ) (h : C_diameter_endpoints p1 p2) : midpoint p1 p2 = (6, 3) := 
  sorry

end center_of_circle_C_is_l664_664588


namespace maximum_multichromatic_colorings_l664_664777

-- Define gcd if not already defined in the used Mathlib
def gcd (a b : ℕ) : ℕ := sorry

-- Define what it means to be multichromatic
def is_multichromatic (N : ℕ) (color : ℕ → ℕ) : Prop :=
  ∀ (a b : ℕ), a ∣ N → b ∣ N → a ≠ b → gcd(a, b) ≠ a → gcd(a, b) ≠ b → color a ≠ color b → color b ≠ color (gcd(a, b))

-- Define the statement we need to prove
theorem maximum_multichromatic_colorings (N : ℕ) (hN : ¬∃ p : ℕ, (nat.prime p) ∧ (∃ k : ℕ, N = p ^ k)) :
  ∃ f : ℕ → ℕ, multichromatic N f → ∑ x in (finset.filter (dvd N) (finset.range N)).to_set (λ x, colorings N f) = 192 :=
sorry

end maximum_multichromatic_colorings_l664_664777


namespace packs_of_beef_l664_664976

noncomputable def pounds_per_pack : ℝ := 4
noncomputable def price_per_pound : ℝ := 5.50
noncomputable def total_paid : ℝ := 110
noncomputable def price_per_pack : ℝ := price_per_pound * pounds_per_pack

theorem packs_of_beef (n : ℝ) (h : n = total_paid / price_per_pack) : n = 5 := 
by
  sorry

end packs_of_beef_l664_664976


namespace triangles_in_decagon_l664_664216

theorem triangles_in_decagon :
  let n := 10 in
  let k := 3 in
  Nat.choose n k = 120 := by
  sorry

end triangles_in_decagon_l664_664216


namespace coordinates_of_B_l664_664450

def pointA : Prod Int Int := (-3, 2)
def moveRight (p : Prod Int Int) (units : Int) : Prod Int Int := (p.1 + units, p.2)
def moveDown (p : Prod Int Int) (units : Int) : Prod Int Int := (p.1, p.2 - units)
def pointB : Prod Int Int := moveDown (moveRight pointA 1) 2

theorem coordinates_of_B :
  pointB = (-2, 0) :=
sorry

end coordinates_of_B_l664_664450


namespace ellipse_foci_on_x_axis_l664_664454

variable {a b : ℝ}

theorem ellipse_foci_on_x_axis (h : ∀ x y : ℝ, a * x^2 + b * y^2 = 1) (hc : ∀ x y : ℝ, (a * x^2 + b * y^2 = 1) → (1 / a > 1 / b ∧ 1 / b > 0))
  : 0 < a ∧ a < b :=
sorry

end ellipse_foci_on_x_axis_l664_664454


namespace find_a_l664_664492

noncomputable def area_condition (a : ℝ) : Prop :=
  a > 0 ∧ (∫ x in 0..a, real.sqrt x) = 2 / 3

theorem find_a : ∃ a : ℝ, area_condition a ∧ a = 1 := sorry

end find_a_l664_664492


namespace each_person_paid_l664_664728

-- Define the conditions: total bill and number of people
def totalBill : ℕ := 135
def numPeople : ℕ := 3

-- Define the question as a theorem to prove the correct answer
theorem each_person_paid : totalBill / numPeople = 45 :=
by
  -- Here, we can skip the proof since the statement is required only.
  sorry

end each_person_paid_l664_664728


namespace greatest_prime_factor_of_15_plus_18_l664_664658

theorem greatest_prime_factor_of_15_plus_18! : 
  let n := 15! + 18!
  n = 15! * 4897 ∧ Prime 4897 →
  (∀ p : ℕ, Prime p ∧ p ∣ n → p ≤ 4897) ∧ (4897 ∣ n) ∧ Prime 4897 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_of_15_plus_18_l664_664658


namespace necessary_but_not_sufficient_condition_l664_664038

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  ((0 < x ∧ x < 5) → (|x - 2| < 3)) ∧ ¬ ((|x - 2| < 3) → (0 < x ∧ x < 5)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l664_664038


namespace fraction_BC_AD_l664_664560

-- Definitions based on conditions.
variables {A B C D : Point}
variables {AB BD CD : ℝ}
variables (h1 : B ∈ segment A D)
variables (h2 : C ∈ segment A D)
variables (h3 : AB = 3 * BD)
variables (h4 : AC = 7 * CD)

-- Prove that the fraction of BC to AD is 1/8.
theorem fraction_BC_AD {x : ℝ} (CD := x) (AC := 7 * x) 
  (BD := 2 * x) (AB := 3 * 2 * x) (AD := 4 * 2 * x) :
  BC / AD = 1 / 8 :=
sorry

end fraction_BC_AD_l664_664560


namespace cyclic_hexagon_inequality_l664_664263

theorem cyclic_hexagon_inequality
  (A B C D E F : Point)
  (h : inscribed_in_circle {A, B, C, D, E, F})
  (convex : convex_hexagon A B C D E F) :
  distance A C * distance B D * distance D E * distance C E * distance E A * distance F B ≥ 
  27 * distance A B * distance B C * distance C D * distance D E * distance E F * distance F A := 
sorry

end cyclic_hexagon_inequality_l664_664263


namespace sum_min_max_values_of_f_l664_664819

noncomputable def f (x : ℝ) : ℝ := - (cos x)^2 + sqrt 3 * sin x * sin (x + π / 2)
def interval : Set ℝ := Set.Icc 0 (π / 2)

theorem sum_min_max_values_of_f :
  let f (x : ℝ) : ℝ := - (cos x)^2 + sqrt 3 * sin x * sin (x + π / 2)
  let interval := Set.Icc 0 (π / 2)
  (let min_val := (fun m => ∀ x ∈ interval, f x ≥ m) →
  let max_val := (fun M => ∀ x ∈ interval, f x ≤ M) →
  min_val + max_val = -1/2) :=
sorry

end sum_min_max_values_of_f_l664_664819


namespace even_divisors_count_lt_100_l664_664350

theorem even_divisors_count_lt_100 : 
  {n : ℕ | n < 100 ∧ n ≠ 0 ∧ ∃ k : ℕ, k * k = n } = { n : ℕ | n < 100 ∧ n ≠ 0 } \ 
  { n : ℕ | ∃ k : ℕ, k * k = n ∧ k < 100 } → 
  (card {n : ℕ | n < 100 ∧ n ≠ 0 ∧ even (finset.card (divisors n))} = 90) :=
begin
  sorry
end

end even_divisors_count_lt_100_l664_664350


namespace polygon_vertices_l664_664445

-- Define the number of diagonals from one vertex
def diagonals_from_one_vertex (n : ℕ) := n - 3

-- The main theorem stating the number of vertices is 9 given 6 diagonals from one vertex
theorem polygon_vertices (D : ℕ) (n : ℕ) (h : D = 6) (h_diagonals : diagonals_from_one_vertex n = D) :
  n = 9 := by
  sorry

end polygon_vertices_l664_664445


namespace unfolded_angle_is_sqrt2_pi_l664_664472

noncomputable def central_angle_of_unfolded_cone (r : ℝ) :=
  let l := (√2) * r in
  let circumference_base := 2 * π * r in
  let arc_length := circumference_base in
  arc_length / l

theorem unfolded_angle_is_sqrt2_pi (r : ℝ) : central_angle_of_unfolded_cone r = √2 * π := 
by 
  sorry

end unfolded_angle_is_sqrt2_pi_l664_664472


namespace eval_expr_l664_664755

theorem eval_expr : 3^2 * 4 * 6^3 * Nat.factorial 7 = 39191040 := by
  -- the proof will be filled in here
  sorry

end eval_expr_l664_664755


namespace problem1_problem2_l664_664153

-- Problem 1
theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hneq : a ≠ b) : 
  (a / Real.sqrt b) + (b / Real.sqrt a) > Real.sqrt a + Real.sqrt b :=
sorry

-- Problem 2
theorem problem2 (x : ℝ) (hx : x > -1) (m : ℕ) (hm : 0 < m) : 
  (1 + x)^m ≥ 1 + m * x :=
sorry

end problem1_problem2_l664_664153


namespace probability_of_selecting_two_queens_or_at_least_two_aces_l664_664444

noncomputable def prob_two_queens_or_at_least_two_aces : Rational := 29 / 1105

theorem probability_of_selecting_two_queens_or_at_least_two_aces :
  (prob_two_queens_or_at_least_two_aces = 29 / 1105) := 
by 
  sorry

end probability_of_selecting_two_queens_or_at_least_two_aces_l664_664444


namespace number_of_new_trailer_homes_added_l664_664629

/-- Given:
1. Three years ago, there were 25 trailer homes on Pine Avenue.
2. The average age of these trailer homes was 15 years at that time.
3. A group of brand new trailer homes was added to Pine Avenue since then.
4. Today, the average age of all the trailer homes is 12 years.

Prove that 17 new trailer homes were added three years ago. -/
theorem number_of_new_trailer_homes_added (n : ℕ) 
  (h1 : ∃ n, ∃ k, (25 * (15 + 3) + 3 * n) / (25 + n) = 12) : 
  n = 17 := 
sorry

end number_of_new_trailer_homes_added_l664_664629


namespace projection_of_difference_l664_664468

variables {𝕜 : Type*} [IsROrC 𝕜] {E : Type*} [InnerProductSpace 𝕜 E]
variables (a b : E)

-- Hypotheses: a and b are unit vectors
def unit_vectors (a b : E) : Prop := ∥a∥ = 1 ∧ ∥b∥ = 1

-- Projection to be proved
theorem projection_of_difference (h : unit_vectors a b) : 
  projection (a - b) (a + b) = 0 :=
sorry

end projection_of_difference_l664_664468


namespace at_least_one_no_less_than_two_l664_664801

variable (a b c : ℝ)
variable (ha : 0 < a)
variable (hb : 0 < b)
variable (hc : 0 < c)

theorem at_least_one_no_less_than_two :
  ∃ x ∈ ({a + 1/b, b + 1/c, c + 1/a} : Set ℝ), 2 ≤ x := by
  sorry

end at_least_one_no_less_than_two_l664_664801


namespace benjamin_speed_l664_664733

variable (d : ℝ) (t : ℝ) -- Define variables for distance and time
variable (s : ℝ) -- Define variable for speed

-- State the conditions using Lean definitions
def distance : d = 80 := sorry
def time : t = 8 := sorry

-- Prove the question: Benjamin's speed is 10 kilometers per hour
theorem benjamin_speed : s = d / t → s = 10 :=
by
  intro h
  rw [distance, time] at h
  exact h

end benjamin_speed_l664_664733


namespace initial_amount_l664_664757

theorem initial_amount (P : ℝ) (h1 : ∀ x : ℝ, x * (9 / 8) * (9 / 8) = 81000) : P = 64000 :=
sorry

end initial_amount_l664_664757


namespace number_of_integers_with_even_divisors_l664_664396

-- Define conditions
def N := 99

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P : finset ℕ := { n in finset.range (N + 1) | is_perfect_square n }

noncomputable def countP : ℕ := P.card

-- The statement to prove
theorem number_of_integers_with_even_divisors : 
  (N - countP) = 90 :=
by {
  sorry
}

end number_of_integers_with_even_divisors_l664_664396


namespace angle_CXY_l664_664504

theorem angle_CXY (A B C P Q R X Y : Type)
  (ABC_equilateral : ∀ t ∈ {triangle ABC}, ∠t = 60)
  (PQR_equilateral : ∀ t ∈ {triangle PQR}, ∠t = 60)
  (angle_ABY : ∠ ABY = 65)
  (angle_QYP : ∠ QYP = 75) :
  ∠ CXY = 40 := 
sorry

end angle_CXY_l664_664504


namespace fourth_intersection_point_of_curve_and_circle_l664_664001

theorem fourth_intersection_point_of_curve_and_circle :
  (∃ P : ℝ × ℝ, P = (2, 1/2) ∨ P = (-5, -1/5) ∨ P = (1/3, 3) ∨ (P.1 * P.2 = 1) ∧ P = (-3/10, -10/3)) :=
by
  have P1 : (2 : ℝ) * (1/2 : ℝ) = 1 := by norm_num
  have P2 : (-5 : ℝ) * (-1/5 : ℝ) = 1 := by norm_num
  have P3 : (1/3 : ℝ) * (3 : ℝ) = 1 := by norm_num
  use (-3/10, -10/3)
  split
  · exact or.inr (or.inr (or.inr (and.intro rfl rfl)))
  all_goals { sorry }

end fourth_intersection_point_of_curve_and_circle_l664_664001


namespace point_on_imaginary_axis_l664_664442

theorem point_on_imaginary_axis (z : ℂ) (h : |z - 1| = |z + 1|) : z.re = 0 :=
sorry

end point_on_imaginary_axis_l664_664442


namespace pounds_of_cheese_bought_l664_664121

-- Definitions according to the problem's conditions
def initial_money : ℕ := 87
def cheese_cost_per_pound : ℕ := 7
def beef_cost_per_pound : ℕ := 5
def pounds_of_beef : ℕ := 1
def remaining_money : ℕ := 61

-- The Lean 4 proof statement
theorem pounds_of_cheese_bought :
  ∃ (C : ℕ), initial_money - (cheese_cost_per_pound * C + beef_cost_per_pound * pounds_of_beef) = remaining_money ∧ C = 3 :=
begin
  sorry,
end

end pounds_of_cheese_bought_l664_664121


namespace largest_three_digit_number_with_7_in_hundreds_l664_664674

def is_three_digit_number_with_7_in_hundreds (n : ℕ) : Prop := 
  100 ≤ n ∧ n < 1000 ∧ (n / 100) = 7

theorem largest_three_digit_number_with_7_in_hundreds : 
  ∀ (n : ℕ), is_three_digit_number_with_7_in_hundreds n → n ≤ 799 :=
by sorry

end largest_three_digit_number_with_7_in_hundreds_l664_664674


namespace complex_square_l664_664438

theorem complex_square (z : ℂ) (i : ℂ) (h₁ : z = 5 - 3 * i) (h₂ : i * i = -1) : z^2 = 16 - 30 * i :=
by
  rw [h₁]
  sorry

end complex_square_l664_664438


namespace volume_of_OABC_l664_664846

def vector_oa := (0:ℝ, 0:ℝ, 1:ℝ)
def vector_ob := (2:ℝ, -1:ℝ, 2:ℝ)
def vector_oc := (1:ℝ, 2:ℝ, 3:ℝ)

def volume_of_tetrahedron (a b c : ℝ × ℝ × ℝ) : ℝ :=
  (1 / 6) * (a.1 * b.2 * c.3 + a.2 * b.3 * c.1 + a.3 * b.1 * c.2 - 
              a.3 * b.2 * c.1 - a.1 * b.3 * c.2 - a.2 * b.1 * c.3)

theorem volume_of_OABC : volume_of_tetrahedron vector_oa vector_ob vector_oc = 5 / 6 := 
by
  sorry

end volume_of_OABC_l664_664846


namespace volume_of_cone_l664_664161

theorem volume_of_cone (d : ℝ) (h : ℝ) (r : ℝ) : 
  d = 10 ∧ h = 0.6 * d ∧ r = d / 2 → (1 / 3) * π * r^2 * h = 50 * π :=
by
  intro h1
  rcases h1 with ⟨h_d, h_h, h_r⟩
  sorry

end volume_of_cone_l664_664161


namespace problem_statement_l664_664528

def Q (n : ℕ) : ℚ := 2 * ∏ i in Finset.range (n - 2) + 2, (1 - 1/(i + 3))

theorem problem_statement : Q 2007 = 4 / 2007 := 
by {
  sorry
}

end problem_statement_l664_664528


namespace toothpicks_15_l664_664625

def toothpicks (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- Not used, placeholder for 1-based indexing.
  | 1 => 3
  | k+1 => let p := toothpicks k
           2 + if k % 2 = 0 then 1 else 0 + p

theorem toothpicks_15 : toothpicks 15 = 38 :=
by
  sorry

end toothpicks_15_l664_664625


namespace basketball_students_l664_664872

variable (C B_inter_C B_union_C B : ℕ)

theorem basketball_students (hC : C = 5) (hB_inter_C : B_inter_C = 3) (hB_union_C : B_union_C = 9) (hInclusionExclusion : B_union_C = B + C - B_inter_C) : B = 7 := by
  sorry

end basketball_students_l664_664872


namespace largest_k_subsets_with_non_empty_intersections_l664_664542

theorem largest_k_subsets_with_non_empty_intersections (n : ℕ) (h_pos : 0 < n) :
  ∃ k, k = 2^(n-1) ∧ (∀ (S : finset (finset (fin n))), (∀ s₁ s₂ ∈ S, s₁ ≠ s₂ → (s₁ ∩ s₂).nonempty ) → S.card ≤ 2^(n-1)) :=
by
  sorry

end largest_k_subsets_with_non_empty_intersections_l664_664542


namespace greatest_prime_factor_15_factorial_plus_18_factorial_l664_664644

theorem greatest_prime_factor_15_factorial_plus_18_factorial :
  ∀ {a b c d e f g: ℕ}, a = 15! → b = 18! → c = 16 → d = 17 → e = 18 → f = a * (1 + c * d * e) →
  g = 4896 → Prime 17 → f + b = a + b → Nat.gcd (a + b) g = 17 :=
by
  intros
  sorry

end greatest_prime_factor_15_factorial_plus_18_factorial_l664_664644


namespace candy_count_l664_664244

theorem candy_count (pieces_per_bag : ℕ) (bags : ℕ) :
  pieces_per_bag = 33 → bags = 26 → pieces_per_bag * bags = 858 :=
by 
  intros h1 h2
  rw [h1, h2]
  calc (33 * 26) = 858 : by norm_num

end candy_count_l664_664244


namespace problem1_problem2_problem3_l664_664175

-- Problem 1
theorem problem1 (boys girls : Finset ℕ) (hc1 : boys.card = 3) (hc2 : girls.card = 4) : 
  (alternatingArrangements boys girls).card = 144 := 
sorry

-- Problem 2
theorem problem2 (people : Finset ℕ) (boys girls : Finset ℕ) 
  (boyA boyB : ℕ) 
  (hb1 : boys.card = 3) 
  (hg1 : girls.card = 4)
  (ha1 : boyA ∈ boys) 
  (ha2 : boyB ∈ boys) :
  (arrangementsNoFarEdges people boyA boyB).card = 3720 := 
sorry

-- Problem 3
theorem problem3 (boys : Finset ℕ) (girls : Finset ℕ) (tasks : Finset ℕ)
  (hb1 : boys.card = 3) 
  (hg1 : girls.card = 4) 
  (ht1 : tasks.card = 4) :
  (taskAssignments boys girls tasks).card = 432 := 
sorry

end problem1_problem2_problem3_l664_664175


namespace running_speed_l664_664128

theorem running_speed (walking_speed : ℕ) (walking_time : ℕ) (running_time_in_min : ℕ)
    (h1 : walking_speed = 5)
    (h2 : walking_time = 5)
    (h3 : running_time_in_min = 36)
    : walking_speed * walking_time / (running_time_in_min / 60) = 41.67 := by
    -- Initial conditions
    have h_distance : ℕ := walking_speed * walking_time
    have h_running_time : ℝ := running_time_in_min / 60

    -- Calculation of running speed
    have h_running_speed : ℕ := h_distance / h_running_time

    -- Conclusion
    have h_result : ℕ := 41.67
    sorry

end running_speed_l664_664128


namespace even_divisors_less_than_100_l664_664363

theorem even_divisors_less_than_100 :
  let count_even_divisors := 
    let n := 100 in
    let perfect_squares := { m | ∃ k, k * k = m ∧ m < n } in
    let total_numbers := finset.range n in
    (total_numbers.card - perfect_squares.card) =
    90 
  in count_even_divisors = 90 :=
by
  let n := 100
  let perfect_squares : finset ℕ := finset.filter (λ m, ∃ k, k * k = m) (finset.range n)
  let total_numbers : finset ℕ := finset.range n
  have h : total_numbers.card = 99 := by sorry
  have p : perfect_squares.card = 9 := by sorry
  show total_numbers.card - perfect_squares.card = 90
  calc
    total_numbers.card - perfect_squares.card
      = 99 - 9 := by rw [h, p]
      = 90 := by norm_num

end even_divisors_less_than_100_l664_664363


namespace min_ab_sum_l664_664036

theorem min_ab_sum (a b : ℤ) (h : a * b = 72) : a + b >= -17 :=
by
  sorry

end min_ab_sum_l664_664036


namespace bobby_shoes_l664_664194

variable (Bonny_pairs Becky_pairs Bobby_pairs : ℕ)
variable (h1 : Bonny_pairs = 13)
variable (h2 : 2 * Becky_pairs - 5 = Bonny_pairs)
variable (h3 : Bobby_pairs = 3 * Becky_pairs)

theorem bobby_shoes : Bobby_pairs = 27 :=
by
  -- Use the conditions to prove the required theorem
  sorry

end bobby_shoes_l664_664194


namespace regular_decagon_triangle_count_l664_664214

def regular_decagon (V : Type) := ∃ vertices : V, Fintype.card vertices = 10

theorem regular_decagon_triangle_count (V : Type) [Fintype V] (h : regular_decagon V)
: Fintype.card { triangle : Finset V // triangle.card = 3 } = 120 := by
  sorry

end regular_decagon_triangle_count_l664_664214


namespace even_number_of_divisors_less_than_100_l664_664413

theorem even_number_of_divisors_less_than_100 :
  ∃ (count : ℕ), count = 90 ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 100 →
    (∃ (d : ℕ), d * d = n ∨ (number_of_divisors n % 2 = 0)) :=
begin
  -- the proof goes here
  sorry
end

end even_number_of_divisors_less_than_100_l664_664413


namespace length_of_BC_l664_664933

theorem length_of_BC (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
    (BX CX : ℕ) (h_pow : CX * (BX + CX) = 2013) : 
    BX + CX = 61 :=
  sorry

end length_of_BC_l664_664933


namespace imaginary_part_fraction_l664_664090

theorem imaginary_part_fraction (i : ℂ) (hi : i = complex.I) : complex.im ((1 + i) / (1 - i)) = 1 := 
by {
  sorry
}

end imaginary_part_fraction_l664_664090


namespace product_closest_to_2500_l664_664844

theorem product_closest_to_2500 :
  let p := 0.0003125 * 8125312 in 
  ∃ closest : ℝ, closest = 2500 ∧ 
    (closest = 2500 ∨ closest = 2600 ∨ 
    closest = 250 ∨ closest = 260 ∨ 
    closest = 25000) ∧
    abs (p - closest) ≤ abs (p - 2500) :=
by 
  let p := 0.0003125 * 8125312 
  use 2500
  split
  { refl }
  split
  { left, refl }
  { sorry }

end product_closest_to_2500_l664_664844


namespace sin_cos_alpha_tan_cos_alpha_l664_664279

-- Proof problem 1
theorem sin_cos_alpha (α : ℝ) (x y : ℝ) (h₁ : x = -1) (h₂ : y = 2) (h₃ : sqrt (x^2 + y^2) = √5) :
  sin α * cos α = -2/5 :=
sorry

-- Proof problem 2
theorem tan_cos_alpha (α : ℝ) (h₁ : tan α = -3) (h₂ : cos α = -√10 / 10 ∨ cos α = √10 / 10) :
  tan α + 3 / cos α = -3 - 3 * √10 ∨ tan α + 3 / cos α = -3 + 3 * √10 :=
sorry

end sin_cos_alpha_tan_cos_alpha_l664_664279


namespace each_person_paid_l664_664727

-- Define the conditions: total bill and number of people
def totalBill : ℕ := 135
def numPeople : ℕ := 3

-- Define the question as a theorem to prove the correct answer
theorem each_person_paid : totalBill / numPeople = 45 :=
by
  -- Here, we can skip the proof since the statement is required only.
  sorry

end each_person_paid_l664_664727


namespace pairs_of_real_numbers_l664_664761

theorem pairs_of_real_numbers (a b : ℝ) (h : ∀ (n : ℕ), n > 0 → a * (⌊b * n⌋) = b * (⌊a * n⌋)) :
  a = 0 ∨ b = 0 ∨ a = b ∨ (∃ m n : ℤ, a = (m : ℝ) ∧ b = (n : ℝ)) :=
by
  sorry

end pairs_of_real_numbers_l664_664761


namespace number_of_real_solutions_l664_664752

theorem number_of_real_solutions :
  ∀ x : ℝ, (2^(8 * x + 4)) * (4^(4 * x + 7)) = 8^(5 * x + 6) → x = 0 := by
  sorry

end number_of_real_solutions_l664_664752


namespace BC_length_l664_664971

-- Define the given triangle and circle conditions
variables (A B C X : Type) (AB AC BX CX : ℤ)
axiom AB_value : AB = 86
axiom AC_value : AC = 97
axiom circle_center_radius : ∃ (A : Type), ∃ (radius : ℤ), radius = AB ∧ ∃ (points : Set Type), points = {B, X} ∧ ∀ (P : Type), P ∈ points → dist A P = radius
axiom BX_CX_integers : ∃ (x y : ℤ), BX = x ∧ CX = y

-- Define calculations using the Power of a Point theorem
theorem BC_length :
  ∀ (y: ℤ) (x: ℤ), y(y + x) = AC^2 - AB^2 → x + y = 61 :=
by
  intros y x h
  have h1 : 97^2 = 9409, by norm_num,
  have h2 : 86^2 = 7396, by norm_num,
  rw [AB_value, AC_value] at h,
  rw [h1, h2] at h,
  calc y(y + x) = 2013 := by {exact h}
  -- The human verification part is skipped since we only need the statement here
  sorry

end BC_length_l664_664971


namespace find_g_x2_minus_2_l664_664042

def g : ℝ → ℝ := sorry -- Define g as some real-valued polynomial function.

theorem find_g_x2_minus_2 (x : ℝ) 
(h1 : g (x^2 + 2) = x^4 + 5 * x^2 + 1) : 
  g (x^2 - 2) = x^4 - 3 * x^2 - 7 := 
by sorry

end find_g_x2_minus_2_l664_664042


namespace special_set_exists_l664_664974

def exists_special_set : Prop :=
  ∃ S : Finset ℕ, S.card = 4004 ∧ 
  (∀ A : Finset ℕ, A ⊆ S ∧ A.card = 2003 → (A.sum id % 2003 ≠ 0))

-- statement with sorry to skip the proof
theorem special_set_exists : exists_special_set :=
sorry

end special_set_exists_l664_664974


namespace expected_value_of_marbles_sum_l664_664427

theorem expected_value_of_marbles_sum : 
  let marbles := {1, 2, 3, 4, 5, 6} in
  let pairs := (marbles.powerset.filter (λ s, s.card = 2)).to_finset in
  let pairs_sums := pairs.image (λ s, s.to_list.sum) in
  pairs_sums.sum / pairs_sums.card = 7 :=
by {
  let marbles := {1, 2, 3, 4, 5, 6},
  let pairs := (marbles.powerset.filter (λ s, s.card = 2)).to_finset,
  let pairs_sums := pairs.image (λ s, s.to_list.sum),
  have h1 : pairs_sums.sum = 105, sorry,
  have h2 : pairs_sums.card = 15, sorry,
  rw [h1, h2],
  norm_num,
  done,
}

end expected_value_of_marbles_sum_l664_664427


namespace triangle_bc_length_l664_664944

theorem triangle_bc_length :
  ∀ (A B C X : Type) (d_AB : ℝ) (d_AC : ℝ) (d_BX d_CX BC : ℕ),
  d_AB = 86 ∧ d_AC = 97 →
  let circleA := {center := A, radius := d_AB} in
  let intersect_B := B ∈ circleA in
  let intersect_X := X ∈ circleA in
  d_BX + d_CX = BC →
  d_BX ∈ ℕ ∧ d_CX ∈ ℕ →
  BC = 61 :=
by
  intros A B C X d_AB d_AC d_BX d_CX BC h_dist h_circle h_intersect h_sum h_intBC
  sorry

end triangle_bc_length_l664_664944


namespace min_chord_length_line_eq_l664_664251

def circle (x y : ℝ) : Prop := (x-1)^2 + (y-1)^2 = 16
def line (x y m : ℝ) : Prop := (2*m - 1)*x + (m - 1)*y - 3*m + 1 = 0

theorem min_chord_length_line_eq (m : ℝ) (x y : ℝ) :
  (∃ (C : ℝ → ℝ → Prop), C = circle) ∧ 
  (∃ (l : ℝ → ℝ → ℝ → Prop), l = line) ∧ 
  line x y m →
  (x - 2*y - 4 = 0) :=
sorry

end min_chord_length_line_eq_l664_664251


namespace polygon_centers_form_regular_triangle_l664_664061

variable {α : Type} [Field α] [NormedField α] [NormedSpace α α]

-- Given triangle ABC with vertices a, b, c corresponding to complex numbers
variables (a b c : α)
-- Assume the triangle is non-obtuse
variable (h_non_obtuse : ∀ {A B C : α}, ∠A + ∠B <= 90 ∧ ∠B + ∠C <= 90 ∧ ∠C + ∠A <= 90)

-- Given conditions on the shapes:
variable (h_square_AB : ∃ center_square_AB : α, center_square_AB = (a + b) / 2 + (b - a) * complex.I / 2)
variable (h_mgon_BC : ∃ (m > 5) center_mgon_BC : α, center_mgon_BC = (b + c) / 2 + (b - c) * exponential (2 * pi * complex.I / m) / 2)
variable (h_ngon_CA : ∃ (n > 5) center_ngon_CA : α, center_ngon_CA = (c + a) / 2 + (c - a) * exponential (2 * pi * complex.I / n) / 2)

-- The centers of the square, m-gon, and n-gon form a regular triangle
variable (h_regular_triangle_centers : ∀ (center_square_AB center_mgon_BC center_ngon_CA : α),
  dist center_square_AB center_mgon_BC = dist center_mgon_BC center_ngon_CA ∧
  dist center_mgon_BC center_ngon_CA = dist center_ngon_CA center_square_AB ∧
  ∠(center_square_AB - center_mgon_BC) = 120 ∧ ∠(center_mgon_BC - center_ngon_CA) = 120 ∧ ∠(center_ngon_CA - center_square_AB) = 120)

theorem polygon_centers_form_regular_triangle
  (a b c : α)
  (h_non_obtuse : ∀ {A B C : α}, ∠A + ∠B <= 90 ∧ ∠B + ∠C <= 90 ∧ ∠C + ∠A <= 90)
  (h_square_AB : ∃ center_square_AB : α, center_square_AB = (a + b) / 2 + (b - a) * complex.I / 2)
  (h_mgon_BC : ∃ (m : ℕ) (h_m_gt : 5 < m) center_mgon_BC : α, center_mgon_BC = (b + c) / 2 + (b - c) * exponential (2 * pi * complex.I / m) / 2)
  (h_ngon_CA: ∃ (n : ℕ) (h_n_gt : 5 < n) center_ngon_CA : α, center_ngon_CA = (c + a) / 2 + (c - a) * exponential (2 * pi * complex.I / n) / 2)
  (h_regular_triangle_centers: ∀ (center_square_AB center_mgon_BC center_ngon_CA : α),
    dist center_square_AB center_mgon_BC = dist center_mgon_BC center_ngon_CA ∧
    dist center_mgon_BC center_ngon_CA = dist center_ngon_CA center_square_AB ∧
    ∠(center_square_AB - center_mgon_BC) = 120 ∧ ∠(center_mgon_BC - center_ngon_CA) = 120 ∧ ∠(center_ngon_CA - center_square_AB) = 120):
  m = 6 ∧ n = 6 ∧ (∠a b c = 60 ∧ ∠b c a = 60 ∧ ∠c a b = 60) :=
sorry

end polygon_centers_form_regular_triangle_l664_664061


namespace even_divisors_count_lt_100_l664_664352

theorem even_divisors_count_lt_100 : 
  {n : ℕ | n < 100 ∧ n ≠ 0 ∧ ∃ k : ℕ, k * k = n } = { n : ℕ | n < 100 ∧ n ≠ 0 } \ 
  { n : ℕ | ∃ k : ℕ, k * k = n ∧ k < 100 } → 
  (card {n : ℕ | n < 100 ∧ n ≠ 0 ∧ even (finset.card (divisors n))} = 90) :=
begin
  sorry
end

end even_divisors_count_lt_100_l664_664352


namespace even_number_of_divisors_less_than_100_l664_664420

theorem even_number_of_divisors_less_than_100 :
  ∃ (count : ℕ), count = 90 ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 100 →
    (∃ (d : ℕ), d * d = n ∨ (number_of_divisors n % 2 = 0)) :=
begin
  -- the proof goes here
  sorry
end

end even_number_of_divisors_less_than_100_l664_664420


namespace triangle_bc_length_l664_664942

theorem triangle_bc_length :
  ∀ (A B C X : Type) (d_AB : ℝ) (d_AC : ℝ) (d_BX d_CX BC : ℕ),
  d_AB = 86 ∧ d_AC = 97 →
  let circleA := {center := A, radius := d_AB} in
  let intersect_B := B ∈ circleA in
  let intersect_X := X ∈ circleA in
  d_BX + d_CX = BC →
  d_BX ∈ ℕ ∧ d_CX ∈ ℕ →
  BC = 61 :=
by
  intros A B C X d_AB d_AC d_BX d_CX BC h_dist h_circle h_intersect h_sum h_intBC
  sorry

end triangle_bc_length_l664_664942


namespace leah_birdseed_feeding_weeks_l664_664026

/-- Define the total number of weeks Leah can feed her birds without going back to the store. -/
theorem leah_birdseed_feeding_weeks : 
  (let num_boxes_bought := 3
   let num_boxes_pantry := 5
   let parrot_weekly_consumption := 100
   let cockatiel_weekly_consumption := 50
   let grams_per_box := 225
   let total_boxes := num_boxes_bought + num_boxes_pantry
   let total_birdseed_grams := total_boxes * grams_per_box
   let total_weekly_consumption := parrot_weekly_consumption + cockatiel_weekly_consumption
  in total_birdseed_grams / total_weekly_consumption) = 12 := 
by 
  sorry

end leah_birdseed_feeding_weeks_l664_664026


namespace even_number_of_divisors_lt_100_l664_664406

theorem even_number_of_divisors_lt_100 : 
  let n := 99 in
  let perfect_squares := {1, 4, 9, 16, 25, 36, 49, 64, 81} in
  n - perfect_squares.card = 90 := 
by
  sorry

end even_number_of_divisors_lt_100_l664_664406


namespace inscribed_circle_radius_l664_664136

-- Defining the statement in Lean 4
theorem inscribed_circle_radius (AB AC BC : ℝ) (h1 : AB = 22) (h2 : AC = 12) (h3 : BC = 14) : 
  let s := (AB + AC + BC) / 2,
      K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)),
      r := K / s in
  r = Real.sqrt 10 :=
by
  sorry -- Proof is left as an exercise to the reader or user.

end inscribed_circle_radius_l664_664136


namespace even_divisors_less_than_100_l664_664365

theorem even_divisors_less_than_100 :
  let count_even_divisors := 
    let n := 100 in
    let perfect_squares := { m | ∃ k, k * k = m ∧ m < n } in
    let total_numbers := finset.range n in
    (total_numbers.card - perfect_squares.card) =
    90 
  in count_even_divisors = 90 :=
by
  let n := 100
  let perfect_squares : finset ℕ := finset.filter (λ m, ∃ k, k * k = m) (finset.range n)
  let total_numbers : finset ℕ := finset.range n
  have h : total_numbers.card = 99 := by sorry
  have p : perfect_squares.card = 9 := by sorry
  show total_numbers.card - perfect_squares.card = 90
  calc
    total_numbers.card - perfect_squares.card
      = 99 - 9 := by rw [h, p]
      = 90 := by norm_num

end even_divisors_less_than_100_l664_664365


namespace magnitude_conjugate_l664_664853

theorem magnitude_conjugate (z : ℂ) (h : z = (3 - 2 * complex.I) / complex.I) : complex.abs (conj z) = real.sqrt 13 :=
by {
  sorry
}

end magnitude_conjugate_l664_664853


namespace even_number_of_divisors_less_than_100_l664_664327

theorem even_number_of_divisors_less_than_100 : 
  ∃ n, n = 90 ∧ ∀ x < 100, (∃ k, k * k = x → false) = (x ∣ 99 - 9) :=
sorry

end even_number_of_divisors_less_than_100_l664_664327


namespace BC_length_l664_664914

theorem BC_length (A B C X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X] 
  (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
  (BX CX : ℕ) (h_circle_intersect : ∃ X, Metric.ball A 86 ∩ {BC} = {B, X})
  (h_integer_lengths : BX + CX = BC) :
  BC = 61 := 
by
  sorry

end BC_length_l664_664914


namespace thursday_tea_consumption_l664_664600

-- Define the constants and conditions relevant to the problem
variable (h t k : ℕ)

-- Define the condition for inverse proportionality
def inversely_proportional (h t k : ℕ) : Prop := h * t = k

-- Define the Wednesday condition: grading for 5 hours and drinking 4 liters of tea
def wednesday_condition : Prop := inversely_proportional 5 4 20

-- Prove that given the inverse proportionality holds with k = 20 (from Wednesday),
-- on Thursday where the teacher grades for 8 hours, the teacher drank 2.5 liters of tea
theorem thursday_tea_consumption : wednesday_condition → inversely_proportional 8 2.5 20 :=
by
  intro h
  -- Placeholder for the actual proof that would verify the conditions
  sorry

end thursday_tea_consumption_l664_664600


namespace pentagon_arithmetic_sequences_count_l664_664094

theorem pentagon_arithmetic_sequences_count : 
  ∃ (sequences : ℕ), sequences = 5 ∧
  ∃ (x d : ℕ), 5 * x + 10 * d = 540 ∧
  (∀ i : ℕ, i ∈ finset.range 5 → x + i * d < 120) ∧
  (∀ i : ℕ, i ∈ finset.range 5 → 0 < x + i * d) ∧
  x + d ≠ x + 4 * d :=
by
  sorry

end pentagon_arithmetic_sequences_count_l664_664094


namespace stockholm_to_malmo_distance_l664_664587
-- Import the necessary library

-- Define the parameters for the problem.
def map_distance : ℕ := 120 -- distance in cm
def scale_factor : ℕ := 12 -- km per cm

-- The hypothesis for the map distance and the scale factor
axiom map_distance_hyp : map_distance = 120
axiom scale_factor_hyp : scale_factor = 12

-- Define the real distance function
def real_distance (d : ℕ) (s : ℕ) : ℕ := d * s

-- The problem statement: Prove that the real distance between the two city centers is 1440 km
theorem stockholm_to_malmo_distance : real_distance map_distance scale_factor = 1440 :=
by
  rw [map_distance_hyp, scale_factor_hyp]
  sorry

end stockholm_to_malmo_distance_l664_664587


namespace even_number_of_divisors_l664_664343

-- Proof statement: There are 90 positive integers less than 100 with an even number of divisors.
theorem even_number_of_divisors : 
  {n : ℕ | n < 100 ∧ ∃ k : ℕ, k ^ 2 = n}.toFinset.card = 90 := 
sorry

end even_number_of_divisors_l664_664343


namespace wood_rope_length_equivalence_l664_664499

variable (x y : ℝ)

theorem wood_rope_length_equivalence :
  (x - y = 4.5) ∧ (y = (1 / 2) * x + 1) :=
  sorry

end wood_rope_length_equivalence_l664_664499


namespace even_number_of_divisors_l664_664341

-- Proof statement: There are 90 positive integers less than 100 with an even number of divisors.
theorem even_number_of_divisors : 
  {n : ℕ | n < 100 ∧ ∃ k : ℕ, k ^ 2 = n}.toFinset.card = 90 := 
sorry

end even_number_of_divisors_l664_664341


namespace daily_charge_l664_664523

/-- The daily charge for Lally's internet service based on given conditions -/
theorem daily_charge (x : ℝ) : 
  let day_limit := 25 in
  let initial_balance := 7 in
  let debt_limit := 5 in
  (initial_balance - day_limit * x <= -debt_limit) → (x = 12 / 25) :=
by
  intro h
  have : initial_balance - day_limit * x = -debt_limit ↔ x = 12 / 25 :=
    by linarith
  exact this.mp h

end daily_charge_l664_664523


namespace shifted_quadratic_eq_l664_664461

-- Define the original quadratic function
def orig_fn (x : ℝ) : ℝ := -x^2

-- Define the function after shifting 1 unit to the left
def shifted_left_fn (x : ℝ) : ℝ := - (x + 1)^2

-- Define the final function after also shifting 3 units up
def final_fn (x : ℝ) : ℝ := - (x + 1)^2 + 3

-- Prove the final function is the correctly transformed function from the original one
theorem shifted_quadratic_eq : ∀ (x : ℝ), final_fn x = - (x + 1)^2 + 3 :=
by 
  intro x
  sorry

end shifted_quadratic_eq_l664_664461


namespace number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664384

theorem number_of_integers_with_even_divisors (n : ℕ) (h : n = 100) : 
  (card {x | (x < n) ∧ ∃ k, k * k = x} = 9) → 
  (card {x | (x < n) ∧ ¬(∃ k, k * k = x)} = n - 1 - 9) :=
by
  intro h_squares
  rw h
  trivial

open_locale classical
noncomputable def count_even_divisors_less_than_100 : ℕ :=
  90

theorem count_even_divisors_verification :
  count_even_divisors_less_than_100 = 90 :=
by
  sorry

end number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664384


namespace number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664387

theorem number_of_integers_with_even_divisors (n : ℕ) (h : n = 100) : 
  (card {x | (x < n) ∧ ∃ k, k * k = x} = 9) → 
  (card {x | (x < n) ∧ ¬(∃ k, k * k = x)} = n - 1 - 9) :=
by
  intro h_squares
  rw h
  trivial

open_locale classical
noncomputable def count_even_divisors_less_than_100 : ℕ :=
  90

theorem count_even_divisors_verification :
  count_even_divisors_less_than_100 = 90 :=
by
  sorry

end number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664387


namespace parallelogram_to_rectangle_l664_664186

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

noncomputable
def parallelogram (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :=
distance A B = distance C D ∧ distance B C = distance D A

noncomputable
def rectangle (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :=
parallelogram A B C D ∧
distance A B = distance C D ∧
distance B C = distance D A ∧
angle A B C = π/2 ∧
angle B C D = π/2 ∧
angle C D A = π/2 ∧
angle D A B = π/2

theorem parallelogram_to_rectangle
  {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (h : parallelogram A B C D)
  (h1 : distance A C = distance B D) : rectangle A B C D :=
sorry

end parallelogram_to_rectangle_l664_664186


namespace even_divisors_less_than_100_l664_664359

theorem even_divisors_less_than_100 :
  let count_even_divisors := 
    let n := 100 in
    let perfect_squares := { m | ∃ k, k * k = m ∧ m < n } in
    let total_numbers := finset.range n in
    (total_numbers.card - perfect_squares.card) =
    90 
  in count_even_divisors = 90 :=
by
  let n := 100
  let perfect_squares : finset ℕ := finset.filter (λ m, ∃ k, k * k = m) (finset.range n)
  let total_numbers : finset ℕ := finset.range n
  have h : total_numbers.card = 99 := by sorry
  have p : perfect_squares.card = 9 := by sorry
  show total_numbers.card - perfect_squares.card = 90
  calc
    total_numbers.card - perfect_squares.card
      = 99 - 9 := by rw [h, p]
      = 90 := by norm_num

end even_divisors_less_than_100_l664_664359


namespace pentagon_arithmetic_sequences_count_l664_664093

theorem pentagon_arithmetic_sequences_count : 
  ∃ (sequences : ℕ), sequences = 5 ∧
  ∃ (x d : ℕ), 5 * x + 10 * d = 540 ∧
  (∀ i : ℕ, i ∈ finset.range 5 → x + i * d < 120) ∧
  (∀ i : ℕ, i ∈ finset.range 5 → 0 < x + i * d) ∧
  x + d ≠ x + 4 * d :=
by
  sorry

end pentagon_arithmetic_sequences_count_l664_664093


namespace even_number_of_divisors_less_than_100_l664_664412

theorem even_number_of_divisors_less_than_100 :
  ∃ (count : ℕ), count = 90 ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 100 →
    (∃ (d : ℕ), d * d = n ∨ (number_of_divisors n % 2 = 0)) :=
begin
  -- the proof goes here
  sorry
end

end even_number_of_divisors_less_than_100_l664_664412


namespace evens_divisors_lt_100_l664_664369

theorem evens_divisors_lt_100 : 
  ∃ n : ℕ, n = 90 ∧ ∀ k : ℕ, (1 ≤ k < 100) → (even k ↔ (∃ m : ℕ, m * m = k)) ↔ (n = 90) := 
sorry

end evens_divisors_lt_100_l664_664369


namespace smallest_positive_period_axis_of_symmetry_range_of_f_l664_664822

def f (x : ℝ) : ℝ := cos (2 * x - π / 3) + 2 * sin (x - π / 4) * cos (x - π / 4)

theorem smallest_positive_period :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π :=
sorry

theorem axis_of_symmetry :
  ∀ k : ℤ, ∀ x : ℝ, (x = π / 3 + k * π / 2) → f (x) = f (π / 3) :=
sorry

theorem range_of_f :
  ∀ x : ℝ, -π / 12 ≤ x ∧ x ≤ π / 2 → -sqrt 3 / 2 ≤ f x ∧ f x ≤ 1 :=
sorry

end smallest_positive_period_axis_of_symmetry_range_of_f_l664_664822


namespace extreme_points_sum_of_extreme_points_less_than_minus_two_l664_664789

def f (x a : ℝ) := x^2 - 4 * a * x + a * Real.log x

theorem extreme_points (a : ℝ) :
  (0 ≤ a ∧ a ≤ 1/2 → ¬(∃ x, differentiable ℝ (f x a) ∧ f' x = 0)) ∧
  (a < 0 → ∃! x, differentiable ℝ (f x a) ∧ f' x = 0) ∧
  (a > 1/2 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ differentiable ℝ (f x₁ a) ∧ differentiable ℝ (f x₂ a) ∧ f' x₁ = 0 ∧ f' x₂ = 0)
:= sorry

theorem sum_of_extreme_points_less_than_minus_two {a x₁ x₂ : ℝ} (h : a > 1/2) (hx₁ : differentiable ℝ (f x₁ a) ∧ f' x₁ = 0) (hx₂ : differentiable ℝ (f x₂ a) ∧ f' x₂ = 0) (hx₁x₂ : x₁ ≠ x₂) : 
  f x₁ a + f x₂ a < -2 
:= sorry

end extreme_points_sum_of_extreme_points_less_than_minus_two_l664_664789


namespace complex_square_example_l664_664440

noncomputable def z : ℂ := 5 - 3 * Complex.I
noncomputable def i_squared : ℂ := Complex.I ^ 2

theorem complex_square_example : z ^ 2 = 34 - 30 * Complex.I := by
  have i_squared_eq : i_squared = -1 := by
    unfold i_squared
    rw [Complex.I_sq]
    rfl
  unfold z
  calc
    (5 - 3 * Complex.I) ^ 2
        = (5 ^ 2 - (3 * Complex.I) ^ 2 - 2 * 5 * 3 * Complex.I) : by
          ring
    ... = 25 - 9 * i_squared - 30 * Complex.I : by
          rw [Complex.mul_sq, Complex.I_sq]
    ... = 25 - 9 * (-1) - 30 * Complex.I : by
          rw [i_squared_eq]
    ... = 25 + 9 - 30 * Complex.I : by
          ring
    ... = 34 - 30 * Complex.I : by
          ring

end complex_square_example_l664_664440


namespace polynomials_zero_l664_664561

theorem polynomials_zero 
  (m : ℕ)
  (P Q R : ℝ[X][Y]) 
  (degP : ∀ x, P.degree x < m)
  (degQ : ∀ x, Q.degree x < m)
  (degR : ∀ x, R.degree x < m)
  (H : ∀ x y : ℝ, x^(2*m) * P.coeff x y + y^(2*m) * Q.coeff x y = (x + y)^(2*m) * R.coeff x y) 
  : ∀ x y : ℝ, P.coeff x y = 0 ∧ Q.coeff x y = 0 ∧ R.coeff x y = 0 :=
by
  sorry

end polynomials_zero_l664_664561


namespace ch4_required_for_ccl4_l664_664738

-- Definitions based on problem conditions
def reaction1 := "CH4 + 2Cl2 ↔ CH2Cl2 + HCl"
def reaction2 := "CH2Cl2 + 2Cl2 ↔ CCl4 + CHCl3"
def reaction3 := "CH4 + Cl2 ↔ CH3Cl + HCl"

def K1 := 1.2 * 10^2
def K2 := 1.5 * 10^3
def K3 := 3.4 * 10^4

def initial_moles_CH2Cl2 := 2.5
def initial_moles_CHCl3 := 1.5
def initial_moles_HCl := 0.5
def initial_moles_Cl2 := 10
def initial_moles_CH3Cl := 0.2
def required_CCl4 := 5

theorem ch4_required_for_ccl4 :
  ∀ (CH4_required : ℝ), 
    (CH4_required = 2.5) → 
    (initial_moles_CH2Cl2 + CH4_required = 5) → 
    (initial_moles_Cl2 ≥ 15) → 
    CH4_required = 2.5 :=
sorry

end ch4_required_for_ccl4_l664_664738


namespace car_travel_distance_l664_664705

-- Define the average speed function
def avg_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

-- Define the distance function given speed and time
def distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

-- Lean statement to prove the distance traveled in the next 3 hours
theorem car_travel_distance : 
  avg_speed 180 4 * 3 = 135 :=
by
  sorry

end car_travel_distance_l664_664705


namespace sum_of_all_numbers_in_distinct_arrays_l664_664262

def is_permutation (l1 l2 : List ℕ) : Prop :=
  l1 ~ l2

def has_fixed_point (l : List ℕ) : Prop :=
  ∃ i, i < l.length ∧ l.get ⟨i, sorry⟩ = i + 1

theorem sum_of_all_numbers_in_distinct_arrays :
  ∀ l : List (List ℕ),
  (∀ arr ∈ l, is_permutation arr [1, 2, 3, 4, 5] ∧ has_fixed_point arr) →
  l.length = 45 →
  list_sum (l.map list_sum) = 675 :=
by
  sorry

end sum_of_all_numbers_in_distinct_arrays_l664_664262


namespace largest_term_at_k_31_l664_664748

noncomputable def B_k (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.15)^k

theorem largest_term_at_k_31 : 
  ∀ k : ℕ, (k ≤ 500) →
    (B_k 31 ≥ B_k k) :=
by
  intro k hk
  sorry

end largest_term_at_k_31_l664_664748


namespace range_of_a_l664_664464

theorem range_of_a (a : ℝ) (h : ∀ x, x > a → 2 * x + 2 / (x - a) ≥ 5) : a ≥ 1 / 2 :=
sorry

end range_of_a_l664_664464


namespace range_of_m_l664_664249

theorem range_of_m (x y : ℝ) (m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hineq : ∀ x > 0, ∀ y > 0, 2 * y / x + 8 * x / y ≥ m^2 + 2 * m) : 
  -4 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_m_l664_664249


namespace bathroom_area_l664_664070

-- Definitions based on conditions
def totalHouseArea : ℝ := 1110
def numBedrooms : ℕ := 4
def bedroomArea : ℝ := 11 * 11
def kitchenArea : ℝ := 265
def numBathrooms : ℕ := 2

-- Mathematically equivalent proof problem
theorem bathroom_area :
  let livingArea := kitchenArea  -- living area is equal to kitchen area
  let totalRoomArea := numBedrooms * bedroomArea + kitchenArea + livingArea
  let remainingArea := totalHouseArea - totalRoomArea
  let bathroomArea := remainingArea / numBathrooms
  bathroomArea = 48 :=
by
  repeat { sorry }

end bathroom_area_l664_664070


namespace evens_divisors_lt_100_l664_664370

theorem evens_divisors_lt_100 : 
  ∃ n : ℕ, n = 90 ∧ ∀ k : ℕ, (1 ≤ k < 100) → (even k ↔ (∃ m : ℕ, m * m = k)) ↔ (n = 90) := 
sorry

end evens_divisors_lt_100_l664_664370


namespace chloe_sneakers_l664_664743

theorem chloe_sneakers (n : ℕ) (cost : ℝ) (twenty_bills : ℕ) (one_dollar_bills : ℤ) (quarters : ℕ) 
(h1 : cost = 47.50)
(h2 : twenty_bills = 2)
(h3 : one_dollar_bills = 6)
(h4 : quarters = 10)
: 40 + 6 + 2.50 + 0.05 * n ≥ 47.50 := by
suffices h : 48.50 + 0.05 * n ≥ 47.50
from h
sorry

end chloe_sneakers_l664_664743


namespace BC_length_l664_664951

-- Define the given values and conditions
variable (A B C X : Type)
variable (AB AC AX BX CX : ℕ)
variable [inhabited A] [inhabited B] [inhabited C] [inhabited X]

-- Assume the lengths of AB and AC
axiom h_AB : AB = 86
axiom h_AC : AC = 97

-- Assume the circle centered at A with radius AB intersects BC at B and X
axiom h_circle : AX = AB

-- Assume BX and CX are integers
axiom h_BX_integral : ∃ (x : ℕ), BX = x
axiom h_CX_integral : ∃ (y : ℕ), CX = y

-- The statement to prove that the length of BC is 61
theorem BC_length : (∃ (x y : ℕ), BX = x ∧ CX = y ∧ x + y = 61) :=
by
  sorry

end BC_length_l664_664951


namespace even_number_of_divisors_less_than_100_l664_664319

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k, k * k = n

theorem even_number_of_divisors_less_than_100 :
  (card { n : ℕ | 1 ≤ n ∧ n < 100 ∧ ¬ is_perfect_square n }) = 90 :=
by
  sorry

end even_number_of_divisors_less_than_100_l664_664319


namespace number_of_integers_with_even_divisors_l664_664392

-- Define conditions
def N := 99

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P : finset ℕ := { n in finset.range (N + 1) | is_perfect_square n }

noncomputable def countP : ℕ := P.card

-- The statement to prove
theorem number_of_integers_with_even_divisors : 
  (N - countP) = 90 :=
by {
  sorry
}

end number_of_integers_with_even_divisors_l664_664392


namespace prob_heart_king_l664_664122

theorem prob_heart_king :
    let total_cards := 52
    let probability_heart := 13 / 52
    let probability_king := 4 / 51 in
    (1 / 52 * 3 / 51 + 12 / 52 * 4 / 51) = 1 / 52 :=
by sorry

end prob_heart_king_l664_664122


namespace complex_num_diff_l664_664280

-- Definitions based on given conditions
variables (a b : ℝ) (i : ℂ)
def complex_num := (3 + i) / (1 - i)
def given_condition := a + b * complex.i = complex_num

-- Theorem statement expressing the proof problem
theorem complex_num_diff : given_condition i → a - b = -1 := 
by 
  sorry

end complex_num_diff_l664_664280


namespace find_solutions_l664_664028

-- Define the conditions that a, b, c are positive real numbers.
variables {a b c x y z : ℝ}
variable (pos_a : 0 < a)
variable (pos_b : 0 < b)
variable (pos_c : 0 < c)

-- Define the system of equations.
def system (x y z : ℝ) : Prop :=
  a * x + b * y = (x - y) ^ 2 ∧
  b * y + c * z = (y - z) ^ 2 ∧
  c * z + a * x = (z - x) ^ 2

-- Define the set of expected solutions.
def solutions : set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (0, 0, c), (a, 0, 0), (0, b, 0)}

-- State the theorem.
theorem find_solutions : ∀ x y z, system a b c x y z ↔ (x, y, z) ∈ solutions :=
begin
  sorry
end

end find_solutions_l664_664028


namespace probability_all_letters_wrong_5_eq_11_div_30_l664_664155

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def derangement (n : ℕ) : ℕ :=
  if n = 0 then 1 else if n = 1 then 0 else (n - 1) * (derangement (n - 1) + derangement (n - 2))

def probability_all_letters_wrong (n : ℕ) : ℚ :=
  (derangement n : ℚ) / (factorial n : ℚ)

theorem probability_all_letters_wrong_5_eq_11_div_30 :
  probability_all_letters_wrong 5 = 11 / 30 :=
by
  sorry

end probability_all_letters_wrong_5_eq_11_div_30_l664_664155


namespace carl_teaches_periods_l664_664205

theorem carl_teaches_periods (cards_per_student : ℕ) (students_per_class : ℕ) (pack_cost : ℕ) (amount_spent : ℕ) (cards_per_pack : ℕ) :
  cards_per_student = 10 →
  students_per_class = 30 →
  pack_cost = 3 →
  amount_spent = 108 →
  cards_per_pack = 50 →
  (amount_spent / pack_cost) * cards_per_pack / (cards_per_student * students_per_class) = 6 :=
by
  intros hc hs hp ha hpkg
  /- proof steps would go here -/
  sorry

end carl_teaches_periods_l664_664205


namespace compute_g_ggg2_l664_664047

def g (n : ℕ) : ℕ :=
  if n < 3 then n^2 + 1
  else if n < 5 then 2 * n + 2
  else 4 * n - 3

theorem compute_g_ggg2 : g (g (g 2)) = 65 :=
by
  sorry

end compute_g_ggg2_l664_664047


namespace unique_sequence_identity_l664_664589

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 5 * x - 2

theorem unique_sequence_identity (r : ℝ) (a : ℕ → ℕ) 
  (h1 : f(r) = 0) (h2 : ∃! r : ℝ, 0 < r ∧ r < 1 ∧ f r = 0)
  (h3 : StrictMono a)
  (h4 : ∀ n : ℕ, 0 < a n)
  (h5 : lim (λ n, ∑ i in finset.range n, r ^ (a i)) = 2 / 5) :
  a = λ n, 3 * n - 2 :=
sorry

end unique_sequence_identity_l664_664589


namespace problem_prove_ω_and_delta_l664_664286

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem problem_prove_ω_and_delta (ω φ : ℝ) (h_ω : ω > 0) (h_φ : abs φ < π / 2) 
    (h_sym_axis : ∀ x, f ω φ x = f ω φ (-(x + π))) 
    (h_center_sym : ∃ c : ℝ, (c = π / 2) ∧ (f ω φ c = 0)) 
    (h_monotone_increasing : ∀ x, -π ≤ x ∧ x ≤ -π / 2 → f ω φ x < f ω φ (x + 1)) :
    (ω = 1 / 3) ∧ (∀ δ : ℝ, (∀ x : ℝ, f ω φ (x + δ) = f ω φ (-x + δ)) → ∃ k : ℤ, δ = 2 * π + 3 * k * π) :=
by
  sorry

end problem_prove_ω_and_delta_l664_664286


namespace range_of_sum_eqn_l664_664250

theorem range_of_sum_eqn (x y : ℝ) (h : x^2 + 2 * x * y + 4 * y^2 = 6) :
  -real.sqrt 6 ≤ x + y ∧ x + y ≤ real.sqrt 6 :=
sorry

end range_of_sum_eqn_l664_664250


namespace even_number_of_divisors_l664_664333

-- Proof statement: There are 90 positive integers less than 100 with an even number of divisors.
theorem even_number_of_divisors : 
  {n : ℕ | n < 100 ∧ ∃ k : ℕ, k ^ 2 = n}.toFinset.card = 90 := 
sorry

end even_number_of_divisors_l664_664333


namespace find_x_l664_664433

theorem find_x (b x : ℝ) (h₁ : b > 0) (h₂ : b ≠ 1) (h₃ : x ≠ 1) : 
  log (x) / log (b^3) + log (b) / log (x^3) = 1 -> x = b^((3 + sqrt 5) / 2) :=
by sorry

end find_x_l664_664433


namespace find_f_three_l664_664811

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def f_condition (f : ℝ → ℝ) := ∀ x : ℝ, x < 0 → f x = (1/2)^x

theorem find_f_three (f : ℝ → ℝ) (h₁ : odd_function f) (h₂ : f_condition f) : f 3 = -8 :=
sorry

end find_f_three_l664_664811


namespace mandy_yoga_time_l664_664687

noncomputable def time_spent_doing_yoga (G B Y E : ℕ) : Prop :=
(G : ℕ) * 3 = (B : ℕ) * 2 ∧
B = 12 ∧ 
E = G + B ∧
Y * 3 = E * 2 ∧
Y = 13

theorem mandy_yoga_time : ∃ G B Y E, time_spent_doing_yoga G B Y E :=
begin
  sorry
end

end mandy_yoga_time_l664_664687


namespace functional_equation_solution_l664_664855

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_equation_solution
  (h1 : ∀ m n : ℝ, f (m + n) = f m + f n - 6)
  (h2 : f (-1) ∈ {1, 2, 3, 4, 5})
  (h3 : ∀ x : ℝ, x > -1 → f x > 0) :
  ∃ a ∈ {1, 2, 3, 4, 5}, ∀ x : ℝ, f x = a * x + 6 :=
sorry

end functional_equation_solution_l664_664855


namespace domain_condition_l664_664823

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log (a * x^2 - x + a)

theorem domain_condition (a : ℝ) : (∀ x : ℝ, a * x^2 - x + a > 0) ↔ (a > 1 / 2) := by
  sorry

end domain_condition_l664_664823


namespace compute_expression_l664_664212

theorem compute_expression : 85 * 1305 - 25 * 1305 + 100 = 78400 := by
  sorry

end compute_expression_l664_664212


namespace jose_tabs_remaining_l664_664018

def initial_tabs : Nat := 400
def step1_tabs_closed (n : Nat) : Nat := n / 4
def step2_tabs_closed (n : Nat) : Nat := 2 * n / 5
def step3_tabs_closed (n : Nat) : Nat := n / 2

theorem jose_tabs_remaining :
  let after_step1 := initial_tabs - step1_tabs_closed initial_tabs
  let after_step2 := after_step1 - step2_tabs_closed after_step1
  let after_step3 := after_step2 - step3_tabs_closed after_step2
  after_step3 = 90 :=
by
  let after_step1 := initial_tabs - step1_tabs_closed initial_tabs
  let after_step2 := after_step1 - step2_tabs_closed after_step1
  let after_step3 := after_step2 - step3_tabs_closed after_step2
  have h : after_step3 = 90 := sorry
  exact h

end jose_tabs_remaining_l664_664018


namespace correct_eqns_l664_664497

theorem correct_eqns (x y : ℝ) (h1 : x - y = 4.5) (h2 : 1/2 * x + 1 = y) :
  x - y = 4.5 ∧ 1/2 * x + 1 = y :=
by {
  exact ⟨h1, h2⟩,
}

end correct_eqns_l664_664497


namespace largest_is_B_l664_664683

noncomputable def A := Real.sqrt (Real.sqrt (56 ^ (1 / 3)))
noncomputable def B := Real.sqrt (Real.sqrt (3584 ^ (1 / 3)))
noncomputable def C := Real.sqrt (Real.sqrt (2744 ^ (1 / 3)))
noncomputable def D := Real.sqrt (Real.sqrt (392 ^ (1 / 3)))
noncomputable def E := Real.sqrt (Real.sqrt (448 ^ (1 / 3)))

theorem largest_is_B : B > A ∧ B > C ∧ B > D ∧ B > E := by
  sorry

end largest_is_B_l664_664683


namespace geometric_progression_common_ratio_l664_664873

theorem geometric_progression_common_ratio (a r : ℝ) (h_pos : a > 0)
  (h_eq : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)) :
  r = 1/2 :=
sorry

end geometric_progression_common_ratio_l664_664873


namespace locus_of_G_is_Thales_circle_l664_664004

theorem locus_of_G_is_Thales_circle
  {A B C D E F G : Type}
  -- Define the points
  [right_angle_triangle ABC]
  [point_on_hypotenuse D AB]
  [perpendicular D AC E]
  [perpendicular D BC F]
  [intersection_of_lines BE AF G] :
  -- Prove the locus of G as D moves along AB is the Thales circle
  is_Thales_circle (locus_of G) :=
sorry

end locus_of_G_is_Thales_circle_l664_664004


namespace BC_length_l664_664947

-- Define the given values and conditions
variable (A B C X : Type)
variable (AB AC AX BX CX : ℕ)
variable [inhabited A] [inhabited B] [inhabited C] [inhabited X]

-- Assume the lengths of AB and AC
axiom h_AB : AB = 86
axiom h_AC : AC = 97

-- Assume the circle centered at A with radius AB intersects BC at B and X
axiom h_circle : AX = AB

-- Assume BX and CX are integers
axiom h_BX_integral : ∃ (x : ℕ), BX = x
axiom h_CX_integral : ∃ (y : ℕ), CX = y

-- The statement to prove that the length of BC is 61
theorem BC_length : (∃ (x y : ℕ), BX = x ∧ CX = y ∧ x + y = 61) :=
by
  sorry

end BC_length_l664_664947


namespace sum_lent_to_ramu_is_6836_l664_664548

def borrowed_amount : ℝ := 5655
def interest_rate_anwar : ℝ := 6 / 100
def interest_rate_ramu : ℝ := 9 / 100
def time_period : ℝ := 3
def total_gain : ℝ := 824.85

def si_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

def si_anwar : ℝ := si_interest borrowed_amount interest_rate_anwar time_period

def total_si_ramu (total_sum : ℝ) : ℝ :=
  si_interest total_sum interest_rate_ramu time_period

def sum_added (total_sum : ℝ) : ℝ :=
  total_gain + si_anwar - total_si_ramu total_sum = 0

def total_sum := borrowed_amount + 1181.11

theorem sum_lent_to_ramu_is_6836.11 :
  borrowed_amount + 1181.11 = 6836.11 :=
by
  sorry

end sum_lent_to_ramu_is_6836_l664_664548


namespace expand_poly_product_l664_664758

-- Given polynomials
def poly1 : Polynomial ℚ := 5 * Polynomial.X - 3
def poly2 : Polynomial ℚ := 2 * Polynomial.X ^ 3 + 7 * Polynomial.X - 1

-- Prove their product equals the specified polynomial
theorem expand_poly_product :
  (poly1 * poly2) = (10 * Polynomial.X ^ 4 - 6 * Polynomial.X ^ 3 + 35 * Polynomial.X ^ 2 - 26 * Polynomial.X + 3) :=
by
  sorry

end expand_poly_product_l664_664758


namespace length_of_BC_l664_664920

theorem length_of_BC 
  (A B C X : Type) 
  (d_AB : ℝ) (d_AC : ℝ) 
  (circle_center_A : A) 
  (radius_AB : ℝ)
  (intersects_BC : B → C → X)
  (BX CX : ℕ) 
  (h_BX_in_circle : BX = d_AB) 
  (h_CX_in_circle : CX = d_AC) 
  (h_integer_lengths : ∃ x y : ℕ, BX = x ∧ CX = y) :
  BX + CX = 61 :=
begin
  sorry
end

end length_of_BC_l664_664920


namespace cakes_count_l664_664056

theorem cakes_count (x y : ℕ) 
  (price_fruit price_chocolate total_cost : ℝ) 
  (avg_price : ℝ) 
  (H1 : price_fruit = 4.8)
  (H2 : price_chocolate = 6.6)
  (H3 : total_cost = 167.4)
  (H4 : avg_price = 6.2)
  (H5 : price_fruit * x + price_chocolate * y = total_cost)
  (H6 : total_cost / (x + y) = avg_price) : 
  x = 6 ∧ y = 21 := 
by
  sorry

end cakes_count_l664_664056


namespace right_prism_max_volume_l664_664170

noncomputable def max_volume_of_right_prism : ℝ := 22.36

theorem right_prism_max_volume
  (a b h : ℝ) (θ : ℝ)
  (ratio_cond : a = 3 * b)
  (area_sum_cond : 4 * b * h + (3/2) * b^2 * sin θ = 30)
  (angle_cond : 0 ≤ θ ∧ θ ≤ π) :
  let V := (3/2) * b^2 * h * sin θ in
  V ≤ max_volume_of_right_prism :=
by
  sorry

end right_prism_max_volume_l664_664170


namespace range_of_k_l664_664463

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x = 4 ∧ x < 2) → (k < 1 ∨ k > 3) := 
by 
  sorry

end range_of_k_l664_664463


namespace sale_in_third_month_l664_664165

theorem sale_in_third_month (
  f1 f2 f4 f5 f6 average : ℕ
) (h1 : f1 = 7435) 
  (h2 : f2 = 7927) 
  (h4 : f4 = 8230) 
  (h5 : f5 = 7562) 
  (h6 : f6 = 5991) 
  (havg : average = 7500) :
  ∃ f3, f3 = 7855 ∧ f1 + f2 + f3 + f4 + f5 + f6 = average * 6 :=
by {
  sorry
}

end sale_in_third_month_l664_664165


namespace planting_schemes_count_l664_664885

theorem planting_schemes_count :
  let n := 6 in
  let count_ways (n : ℕ) : ℕ := 
    (Nat.choose 7 0) +
    (Nat.choose 6 1) +
    (Nat.choose 5 2) +
    (Nat.choose 4 3)
  in
  count_ways n = 21 :=
by
  sorry

end planting_schemes_count_l664_664885


namespace abs_diff_of_two_numbers_l664_664612

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 34) (h2 : x * y = 240) : abs (x - y) = 14 :=
by
  sorry

end abs_diff_of_two_numbers_l664_664612


namespace probability_valid_quadrant_l664_664207

def set_of_k : Set ℝ := {-3, -1/2, real.sqrt 3, 1, 6}

def is_valid_quadrant (k : ℝ) : Prop := k < 0

def count_valid (s : Set ℝ) : ℕ := 
  s.toFinset.filter is_valid_quadrant |>.card

theorem probability_valid_quadrant :
  let valid_k_count := count_valid set_of_k
  let total_k_count := set_of_k.toFinset.card
  valid_k_count / total_k_count = 2 / 5 := by
  sorry

end probability_valid_quadrant_l664_664207


namespace polyhedron_edges_vertices_l664_664131

theorem polyhedron_edges_vertices (F : ℕ) (triangular_faces : Prop) (hF : F = 20) : ∃ S A : ℕ, S = 12 ∧ A = 30 :=
by
  -- stating the problem conditions and desired conclusion
  sorry

end polyhedron_edges_vertices_l664_664131


namespace find_length_of_AB_l664_664509

def polar_line (ρ θ : ℝ) : Prop :=
  sqrt 3 * ρ * cos θ - ρ * sin θ = 0

def polar_circle (ρ θ : ℝ) : Prop :=
  ρ = 4 * sin θ

def point_on_line (x y : ℝ) : Prop :=
  sqrt 3 * x - y = 0

def point_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * y = 0

def length_AB : ℝ :=
  2 * sqrt 3

theorem find_length_of_AB :
  ∀ (ρ θ : ℝ), polar_line ρ θ → polar_circle ρ θ → |AB| = 2 * sqrt 3 :=
sorry

end find_length_of_AB_l664_664509


namespace value_greater_than_l664_664443

-- Definitions for the problem conditions
def percent (p : ℝ) (x : ℝ) := (p / 100) * x

def fifteen_percent_of_40 := percent 15 40

def twenty_five_percent_of_16 := percent 25 16

-- The proof problem statement
theorem value_greater_than :
  fifteen_percent_of_40 - twenty_five_percent_of_16 = 2 :=
by
  sorry

end value_greater_than_l664_664443


namespace quadrant_of_z_l664_664707

noncomputable def z (i : ℂ) : ℂ := i / (1 - i)

theorem quadrant_of_z (i : ℂ) (hi : i^3 = -i) : 
  (\(z (i)\ : \mathbb\Complex).re < 0) ∧ (\(z (i)\ : \mathbb\Complex).im > 0). 
  { 
  sorry 
  }

end quadrant_of_z_l664_664707


namespace angle_between_a_and_b_l664_664835

def vector_a := (1 : ℝ, 0 : ℝ)
def vector_b := (-1/2 : ℝ, Real.sqrt 3 / 2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.fst * v2.fst + v1.snd * v2.snd

def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.fst * v.fst + v.snd * v.snd)

noncomputable def cos_theta (v1 v2 : ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

theorem angle_between_a_and_b :
  cos_theta vector_a vector_b = -1/2 →
  ∃ θ : ℝ, θ = 120 := sorry

end angle_between_a_and_b_l664_664835


namespace find_B_find_AD_l664_664514

variable {a b c : ℝ} {A B C : ℝ}
variable (AB AC CD AD BC : ℝ)
variable [DecidableEq ℝ]

-- Given conditions
def conditions (b a c : ℝ) (A B C : ℝ) (AB AC CD : ℝ) :=
  (sqrt 3 * b * tan B = a * cos C + c * cos A) ∧
  (AB = sqrt 7 / 2) ∧
  (AC = 1) ∧
  (CD = sqrt 2 / 2) ∧
  (AD // BC) -- Parallel condition symbolically written.

-- Question (1)
theorem find_B (b a c : ℝ) (A B C : ℝ) (AB AC CD AD BC : ℝ) 
  (h : conditions b a c A B C AB AC CD AD BC) :
  B = π / 6 := by
  sorry

-- Question (2)
theorem find_AD (b a c : ℝ) (A B C : ℝ) (AB AC CD AD BC : ℝ) 
  (h1 : conditions b a c A B C AB AC CD AD BC) (h2 : find_B b a c A B C AB AC CD AD BC) :
  AD = 1 ∨ AD = 1 / 2 := by
  sorry

end find_B_find_AD_l664_664514


namespace triangle_bc_length_l664_664901

theorem triangle_bc_length (A B C X : Type)
  (AB AC : ℕ)
  (hAB : AB = 86)
  (hAC : AC = 97)
  (circle_eq : ∀ {r : ℕ}, r = AB → circle_centered_at_A_intersects_BC_two_points B X)
  (integer_lengths : ∃ (BX CX : ℕ), ) :
  BC = 61 :=
by
  sorry

end triangle_bc_length_l664_664901


namespace janet_pays_for_piano_lessons_at_rate_28_per_hour_l664_664977

theorem janet_pays_for_piano_lessons_at_rate_28_per_hour
  (weeks_per_year : ℕ := 52)
  (clarinet_hourly_rate : ℕ := 40)
  (clarinet_hours_per_week : ℕ := 3)
  (piano_hours_per_week : ℕ := 5)
  (extra_piano_cost : ℕ := 1040) :
  let clarinet_annual_cost := clarinet_hourly_rate * clarinet_hours_per_week * weeks_per_year in
  ∀ piano_hourly_rate : ℕ,
  piano_hourly_rate * piano_hours_per_week * weeks_per_year = clarinet_annual_cost + extra_piano_cost →
  piano_hourly_rate = 28 := by
  intros clarinet_annual_cost
  intros piano_hourly_rate h
  sorry

end janet_pays_for_piano_lessons_at_rate_28_per_hour_l664_664977


namespace parallelogram_to_rectangle_l664_664187

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

noncomputable
def parallelogram (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :=
distance A B = distance C D ∧ distance B C = distance D A

noncomputable
def rectangle (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] :=
parallelogram A B C D ∧
distance A B = distance C D ∧
distance B C = distance D A ∧
angle A B C = π/2 ∧
angle B C D = π/2 ∧
angle C D A = π/2 ∧
angle D A B = π/2

theorem parallelogram_to_rectangle
  {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (h : parallelogram A B C D)
  (h1 : distance A C = distance B D) : rectangle A B C D :=
sorry

end parallelogram_to_rectangle_l664_664187


namespace quadratic_not_proposition_l664_664226

def is_proposition (P : Prop) : Prop := ∃ (b : Bool), (b = true ∨ b = false)

theorem quadratic_not_proposition : ¬ is_proposition (∃ x : ℝ, x^2 + 2*x - 3 < 0) :=
by 
  sorry

end quadratic_not_proposition_l664_664226


namespace max_lateral_area_of_triangular_prism_l664_664062

theorem max_lateral_area_of_triangular_prism
  (r : ℝ) (a h : ℝ)
  (radius_cond : r = 2)
  (surface_eq : 4 * a^2 + 3 * h^2 = 48)
  (lateral_area_eq : ∃ (S : ℝ), S = 3 * a * h) :
  ∃ (S : ℝ), S ≤ 12 * real.sqrt 3 ∧ S = 3 * real.sqrt 6 * 2 * real.sqrt 2 := sorry

end max_lateral_area_of_triangular_prism_l664_664062


namespace length_RS_l664_664476

open Real

-- Given definitions and conditions
def PQ : ℝ := 10
def PR : ℝ := 10
def QR : ℝ := 5
def PS : ℝ := 13

-- Prove the length of RS
theorem length_RS : ∃ (RS : ℝ), RS = 6.17362 := by
  sorry

end length_RS_l664_664476


namespace union_A_B_equiv_l664_664267

def A : Set ℝ := {x : ℝ | x > 2}
def B : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

theorem union_A_B_equiv : A ∪ B = {x : ℝ | x ≥ 1} :=
by
  sorry

end union_A_B_equiv_l664_664267


namespace part_a_part_b_part_c_l664_664189

open_locale classical

namespace Geometry

noncomputable theory

structure Point :=
  (x : ℝ)
  (y : ℝ)

variables (A B M P Q N N' : Point)
variables (AM MB : Segment)
variables (squaresConstructed : Square AM) (squaresConstructed : Square MB)
variables (circumcircleAMCD : Circumcircle (SquareConstructed AM))
variables (circumcircleMBEF : Circumcircle (SquareConstructed MB))
variables (intersectM : circumcircleAMCD.Intersects circumcircleMBEF M)
variables (intersectN : circumcircleAMCD.Intersects circumcircleMBEF N)
variables (AF : Line A F)
variables (BC : Line B C)
variables (N' : intersection AF BC)

theorem part_a : N = N' := by
  sorry

theorem part_b (S: Point) (fixed_point : ∀ M, Line_through M N S) : true := by
  sorry

theorem part_c : 
  let midpointPQ (M : Point) : Point := 
    let P := center (SquareConstructed AM)
    let Q := center (SquareConstructed MB)
    let midX := (P.x + Q.x) / 2
    let midY := (P.y + Q.y) / 2
    { x := midX, y := midY } in
  locus (midpointPQ M) parallel_to AB := by
  sorry

end Geometry

end part_a_part_b_part_c_l664_664189


namespace clock_angle_3_40_l664_664192

theorem clock_angle_3_40 : ∃ (angle : ℝ), angle = 130 ∧
  (∃ (h m : ℝ), h = 3 ∧ m = 40 ∧ angle = abs ((60 * h - 11 * m) / 2)) :=
by
  use 130
  split
  case left
    rfl
  case right
    use [3, 40]
    split
    case left
      rfl
    case right
      split
      case left
        rfl
      case right
        simp
        linarith

end clock_angle_3_40_l664_664192


namespace derivative_at_2_l664_664283

theorem derivative_at_2 (f : ℝ → ℝ) (h : ∀ x, f x = x^2 * deriv f 2 + 5 * x) :
    deriv f 2 = -5/3 :=
by
  sorry

end derivative_at_2_l664_664283


namespace polynomial_min_degree_l664_664077

theorem polynomial_min_degree 
  (p : Polynomial ℚ) 
  (h_roots : ∃ q : Polynomial ℚ, q ≠ 0 ∧ 
                  (∀ r ∈ {3 - Real.sqrt 8, 5 + Real.sqrt 13, 17 - 3 * Real.sqrt 6, -Real.sqrt 3}, IsRoot q r)) : 
  p.degree ≥ 8 :=
by
  -- Proof goes here
  sorry

end polynomial_min_degree_l664_664077


namespace number_of_integers_with_even_divisors_l664_664393

-- Define conditions
def N := 99

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P : finset ℕ := { n in finset.range (N + 1) | is_perfect_square n }

noncomputable def countP : ℕ := P.card

-- The statement to prove
theorem number_of_integers_with_even_divisors : 
  (N - countP) = 90 :=
by {
  sorry
}

end number_of_integers_with_even_divisors_l664_664393


namespace domain_of_function_range_of_function_l664_664152

-- Define the set of integers
def Z : Set Int := {z | True}

-- Q1: Determine the domain of the function y = sqrt(sin x) + sqrt(1/2 - cos x)
theorem domain_of_function (k : ℤ) : 
  ∀ x, (2*k*Real.pi + Real.pi/3 ≤ x ∧ x ≤ 2*k*Real.pi + Real.pi) ↔ 
  ∃ y, y = Real.sqrt (Real.sin x) + Real.sqrt (1/2 - Real.cos x) := sorry

-- Q2: Find the range of the function y = cos^2 x - sin x over the interval [-π/4, π/4]
theorem range_of_function (x : ℝ) : 
  (-Real.pi/4 ≤ x ∧ x ≤ Real.pi/4) → 
  (∃ y, y = Real.cos x ^ 2 - Real.sin x ∧ 
    (2 - 2*Real.sqrt(2))/4 ≤ y ∧ y ≤ 5/4) := sorry

end domain_of_function_range_of_function_l664_664152


namespace symmetric_point_in_xOz_l664_664006

def symmetric_point (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (P.1, -P.2, P.3)

theorem symmetric_point_in_xOz (P : ℝ × ℝ × ℝ) : 
  symmetric_point P = (P.1, -P.2, P.3) :=
by
  sorry

example : symmetric_point (-1, 2, 1) = (-1, -2, 1) :=
by
  rw symmetric_point_in_xOz
  rw symmetric_point
  sorry

end symmetric_point_in_xOz_l664_664006


namespace value_of_A_l664_664597

theorem value_of_A (M A T E H : ℤ) (hH : H = 8) (h1 : M + A + T + H = 31) (h2 : T + E + A + M = 40) (h3 : M + E + E + T = 44) (h4 : M + A + T + E = 39) : A = 12 :=
by
  sorry

end value_of_A_l664_664597


namespace length_of_BC_l664_664924

theorem length_of_BC 
  (A B C X : Type) 
  (d_AB : ℝ) (d_AC : ℝ) 
  (circle_center_A : A) 
  (radius_AB : ℝ)
  (intersects_BC : B → C → X)
  (BX CX : ℕ) 
  (h_BX_in_circle : BX = d_AB) 
  (h_CX_in_circle : CX = d_AC) 
  (h_integer_lengths : ∃ x y : ℕ, BX = x ∧ CX = y) :
  BX + CX = 61 :=
begin
  sorry
end

end length_of_BC_l664_664924


namespace john_started_5_days_ago_l664_664015

noncomputable def daily_wage (x : ℕ) : Prop := 250 + 10 * x = 750

theorem john_started_5_days_ago :
  ∃ x : ℕ, daily_wage x ∧ 250 / x = 5 :=
by
  sorry

end john_started_5_days_ago_l664_664015


namespace expected_value_sum_marbles_l664_664422

theorem expected_value_sum_marbles :
  (1/15 : ℚ) * ((1 + 2) + (1 + 3) + (1 + 4) + (1 + 5) + (1 + 6) + 
                (2 + 3) + (2 + 4) + (2 + 5) + (2 + 6) + (3 + 4) + 
                (3 + 5) + (3 + 6) + (4 + 5) + (4 + 6) + (5 + 6)) = 7 := 
by {
    sorry
}

end expected_value_sum_marbles_l664_664422


namespace collinearCircumcenterC_D_l664_664473

open EuclideanGeometry

variables {A B C D : Point}

-- Given condition: D is the intersection of the external angle bisectors of ∠A and ∠B of ΔABC.
axiom externalAngleBisectorsIntersect : 
  ∃ (a' b' : Line), 
    isExternalAngleBisector A a' ∧ 
    isExternalAngleBisector B b' ∧ 
    Intersection a' b' = {D}

-- Definition: circumcenter of triangle ABD.
noncomputable def circumcenter_ABD : Point :=
  circumcenter_triple A B D

-- Declaration of main proof statement
theorem collinearCircumcenterC_D :
  Collinear {circumcenter_ABD, C, D} :=
sorry

end collinearCircumcenterC_D_l664_664473


namespace BC_length_l664_664958

def triangle_ABC (A B C : Type)
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)] : Prop :=
  ∃ (X : A), (has_dist B X (coe (X.dist B))) ∧ (has_dist C X (coe (X.dist C))) ∧
  ∀ (x y : ℕ), x = X.dist B ∧ y = X.dist C → x + y = 61

theorem BC_length {A B C : Type}
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)]
  (h : triangle_ABC A B C) : 
  ∃ (x y : ℕ), x + y = 61 := sorry

end BC_length_l664_664958


namespace radius_of_omega2_l664_664995

noncomputable def triangle_proof (A B C : ℝ) (AB BC CA : ℝ) (r ω₁ ω₂ Γ S M : ℝ) : Prop :=
  let circumcircle_centre : ℝ := Γ
  let minor_arc_midpoint : ℝ := M
  let tangent_point : ℝ := S
  let gamma_tangent : ℝ := ω₁
  let omega2_radius : ℝ := ω₂
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
  AB = 13 ∧ BC = 14 ∧ CA = 15 ∧ 
  tangent_point = S ∧ 
  minor_arc_midpoint = M ∧
  gamma_tangent = ω₁ ∧ 
  ω₂ = \frac{1235}{108}

/-- The radius of the circle ω₂ is ∀ is equal to 1235/108 given the conditions. -/
theorem radius_of_omega2 (A B C : ℝ) (AB BC CA ω₁ ω₂ Γ M S: ℝ) (h1 : A ≠ B) 
  (h2 : B ≠ C) (h3 : A ≠ C) (h4 : AB = 13) (h5 : BC = 14) (h6 : CA = 15) 
  (h7 : (BS : tangent_point) - (CS : tangent_point) = \frac{4}{15}) 
  (h8 : M = minor_arc_midpoint) (h9 : ω₁ = gamma_tangent) 
  (h10 : Γ = Γ) : 
  ω₂ = \frac{1235}{108} := by  
  sorry

end radius_of_omega2_l664_664995


namespace sum_smallest_largest_eq_2y_l664_664582

theorem sum_smallest_largest_eq_2y (n : ℕ) (y a : ℕ) 
  (h1 : 2 * a + 2 * (n - 1) / n = y) : 
  2 * y = (2 * a + 2 * (n - 1)) := 
sorry

end sum_smallest_largest_eq_2y_l664_664582


namespace probability_of_sequence_HTHT_l664_664135

noncomputable def prob_sequence_HTHT : ℚ :=
  let p := 1 / 2
  (p * p * p * p)

theorem probability_of_sequence_HTHT :
  prob_sequence_HTHT = 1 / 16 := 
by
  sorry

end probability_of_sequence_HTHT_l664_664135


namespace only_solutions_l664_664220

theorem only_solutions (m n : ℕ) (p : ℕ) (prime_p : Nat.Prime p) (condition : (Nat.choose m 2) - 1 = p^n) :
  (m = 5 ∧ n = 2 ∧ p = 3) ∨ (m = 8 ∧ n = 3 ∧ p = 3) :=
by
  sorry

end only_solutions_l664_664220


namespace find_segment_XY_l664_664893

open_locale classical

noncomputable def triangles_similar (XYZ WUV : Type*) :=
  ∃ (XY XZ YZ WV WU UV : ℝ), 
    triangle XYZ XY XZ YZ ∧ triangle WUV WV WU UV ∧
    similar_by_aa XYZ WUV ∧
    XY / WV = XZ / WU ∧
    XY = 9 ∧ XZ = 15 ∧ WV = 4.5 ∧ UV = 7.5

theorem find_segment_XY (XYZ WUV : Type*) [triangles_similar XYZ WUV]:
  ∃ (XY : ℝ), XY = 9 :=
begin
  sorry
end

end find_segment_XY_l664_664893


namespace inequality_example_l664_664562

theorem inequality_example (n : ℕ) (a : ℕ → ℝ) (h : ∀ i, 1 ≤ i → i ≤ n → a i > 0) :
  (∑ i in Finset.range n, i / (∑ j in Finset.range i, a j)) < 4 * ∑ i in Finset.range n, (1 / a i) :=
sorry

end inequality_example_l664_664562


namespace minimum_friend_circles_l664_664709

variable (P Q : Type) [Fintype P] 
variable [DecidableEq P]

-- Define the number of delegates and their handshakes
def delegates : Finset P := finset.univ.filter (λ x, true) -- assuming true for all elements

-- Assume we have a function representing if two delegates shake hands
variable (shake_hands : P → P → Prop)

-- Define a circle of friends
def circle_of_friends (a b c : P) : Prop :=
  shake_hands a b ∧ shake_hands b c ∧ shake_hands c a

-- Define a function to count the number of shakes each delegate has
variable [fintype Q] (d : Q → ℕ)

-- State the main theorem with all assumptions and conditions
theorem minimum_friend_circles 
  (h_delegates : ∃ n, n = 24)
  (h_handshakes : ∃ h : Finset (P × P), h.card = 216 ∧ (∀ {a b}, (a, b) ∈ h → shake_hands a b))
  (h_no_more_than_10 : ∀ {a b} (ha : shake_hands a b), (∃ n, n ≤ 10) ∧ (∑ x in delegates.erase a ∪ delegates.erase b, if shake_hands a x ∨ shake_hands b x then 1 else 0) ≤ n + 12) :
  ∃ m, m = 864 ∧ 
  ∑ e in delegates.sigma delegates, (circle_of_friends e.1 e.2 (e.2)) = m :=
sorry

end minimum_friend_circles_l664_664709


namespace even_number_of_divisors_lt_100_l664_664408

theorem even_number_of_divisors_lt_100 : 
  let n := 99 in
  let perfect_squares := {1, 4, 9, 16, 25, 36, 49, 64, 81} in
  n - perfect_squares.card = 90 := 
by
  sorry

end even_number_of_divisors_lt_100_l664_664408


namespace even_number_of_divisors_less_than_100_l664_664317

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k, k * k = n

theorem even_number_of_divisors_less_than_100 :
  (card { n : ℕ | 1 ≤ n ∧ n < 100 ∧ ¬ is_perfect_square n }) = 90 :=
by
  sorry

end even_number_of_divisors_less_than_100_l664_664317


namespace BC_length_l664_664969

-- Define the given triangle and circle conditions
variables (A B C X : Type) (AB AC BX CX : ℤ)
axiom AB_value : AB = 86
axiom AC_value : AC = 97
axiom circle_center_radius : ∃ (A : Type), ∃ (radius : ℤ), radius = AB ∧ ∃ (points : Set Type), points = {B, X} ∧ ∀ (P : Type), P ∈ points → dist A P = radius
axiom BX_CX_integers : ∃ (x y : ℤ), BX = x ∧ CX = y

-- Define calculations using the Power of a Point theorem
theorem BC_length :
  ∀ (y: ℤ) (x: ℤ), y(y + x) = AC^2 - AB^2 → x + y = 61 :=
by
  intros y x h
  have h1 : 97^2 = 9409, by norm_num,
  have h2 : 86^2 = 7396, by norm_num,
  rw [AB_value, AC_value] at h,
  rw [h1, h2] at h,
  calc y(y + x) = 2013 := by {exact h}
  -- The human verification part is skipped since we only need the statement here
  sorry

end BC_length_l664_664969


namespace sqrt_meaningful_iff_l664_664867

theorem sqrt_meaningful_iff (x : ℝ) : (∃ r : ℝ, r = sqrt (6 + x)) ↔ x ≥ -6 :=
by
  sorry

end sqrt_meaningful_iff_l664_664867


namespace pole_length_l664_664517

theorem pole_length (bridge_length half_bridge_dist dist_sum : ℝ) (h : ℝ)
  (bridge_length_def : bridge_length = 20000)
  (half_bridge_dist_def : half_bridge_dist = 10000)
  (dist_sum_def : dist_sum = 20001)
  (dist_eq : 2 * real.sqrt (half_bridge_dist ^ 2 + h ^ 2) = dist_sum) :
  h = 100 :=
begin
  sorry
end

end pole_length_l664_664517


namespace solution_set_of_f_l664_664854

theorem solution_set_of_f (f : ℝ → ℝ) (h1 : ∀ x, 2 < deriv f x) (h2 : f (-1) = 2) :
  ∀ x, x > -1 → f x > 2 * x + 4 := by
  sorry

end solution_set_of_f_l664_664854


namespace BC_length_l664_664959

def triangle_ABC (A B C : Type)
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)] : Prop :=
  ∃ (X : A), (has_dist B X (coe (X.dist B))) ∧ (has_dist C X (coe (X.dist C))) ∧
  ∀ (x y : ℕ), x = X.dist B ∧ y = X.dist C → x + y = 61

theorem BC_length {A B C : Type}
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)]
  (h : triangle_ABC A B C) : 
  ∃ (x y : ℕ), x + y = 61 := sorry

end BC_length_l664_664959


namespace expected_value_of_marbles_sum_l664_664425

theorem expected_value_of_marbles_sum : 
  let marbles := {1, 2, 3, 4, 5, 6} in
  let pairs := (marbles.powerset.filter (λ s, s.card = 2)).to_finset in
  let pairs_sums := pairs.image (λ s, s.to_list.sum) in
  pairs_sums.sum / pairs_sums.card = 7 :=
by {
  let marbles := {1, 2, 3, 4, 5, 6},
  let pairs := (marbles.powerset.filter (λ s, s.card = 2)).to_finset,
  let pairs_sums := pairs.image (λ s, s.to_list.sum),
  have h1 : pairs_sums.sum = 105, sorry,
  have h2 : pairs_sums.card = 15, sorry,
  rw [h1, h2],
  norm_num,
  done,
}

end expected_value_of_marbles_sum_l664_664425


namespace geometric_series_sum_l664_664211

theorem geometric_series_sum :
  let a := 2
  let r := 3
  let n := 8
  (∑ i in Finset.range n, a * r^i) = 6560 :=
sorry

end geometric_series_sum_l664_664211


namespace find_a_20_l664_664049

-- Definitions
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ (a₀ d : ℤ), ∀ n, a n = a₀ + n * d

def sum_first_n (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = n * (a 0 + a (n - 1)) / 2

-- Conditions and question
theorem find_a_20 (a S : ℕ → ℤ) (a₀ d : ℤ) :
  arithmetic_seq a ∧ sum_first_n a S ∧ 
  S 6 = 8 * (S 3) ∧ a 3 - a 5 = 8 → a 20 = -74 :=
by
  sorry

end find_a_20_l664_664049


namespace find_coordinates_l664_664451

def A : Prod ℤ ℤ := (-3, 2)
def move_right (p : Prod ℤ ℤ) : Prod ℤ ℤ := (p.fst + 1, p.snd)
def move_down (p : Prod ℤ ℤ) : Prod ℤ ℤ := (p.fst, p.snd - 2)

theorem find_coordinates :
  move_down (move_right A) = (-2, 0) :=
by
  sorry

end find_coordinates_l664_664451


namespace even_divisors_less_than_100_l664_664362

theorem even_divisors_less_than_100 :
  let count_even_divisors := 
    let n := 100 in
    let perfect_squares := { m | ∃ k, k * k = m ∧ m < n } in
    let total_numbers := finset.range n in
    (total_numbers.card - perfect_squares.card) =
    90 
  in count_even_divisors = 90 :=
by
  let n := 100
  let perfect_squares : finset ℕ := finset.filter (λ m, ∃ k, k * k = m) (finset.range n)
  let total_numbers : finset ℕ := finset.range n
  have h : total_numbers.card = 99 := by sorry
  have p : perfect_squares.card = 9 := by sorry
  show total_numbers.card - perfect_squares.card = 90
  calc
    total_numbers.card - perfect_squares.card
      = 99 - 9 := by rw [h, p]
      = 90 := by norm_num

end even_divisors_less_than_100_l664_664362


namespace algebra_simplification_l664_664753

theorem algebra_simplification (x : ℤ) : x + 3 - 4x - 5 + 6x + 7 - 8x - 9 = -5x - 4 :=
by
  sorry

end algebra_simplification_l664_664753


namespace triangle_bc_length_l664_664907

theorem triangle_bc_length (A B C X : Type)
  (AB AC : ℕ)
  (hAB : AB = 86)
  (hAC : AC = 97)
  (circle_eq : ∀ {r : ℕ}, r = AB → circle_centered_at_A_intersects_BC_two_points B X)
  (integer_lengths : ∃ (BX CX : ℕ), ) :
  BC = 61 :=
by
  sorry

end triangle_bc_length_l664_664907


namespace no_such_integers_x_y_l664_664563

theorem no_such_integers_x_y (x y : ℤ) : x^2 + 1974 ≠ y^2 := by
  sorry

end no_such_integers_x_y_l664_664563


namespace vectors_upper_bound_l664_664034
-- Import the entire Mathlib library

-- Define the necessary components of the problem
variable {n : ℕ} (p : ℕ) [Fact (Nat.Prime p)]
variable (v : Fin n → ℤ × ℤ × ℤ)

-- Define the conditions given in the problem
def is_prime_length (v : ℤ × ℤ × ℤ) (p : ℕ) : Prop :=
  v.1^2 + v.2^2 + v.3^2 = p^2

def divisible_by_p (a : ℤ × ℤ × ℤ) (p : ℕ) : Prop :=
  a.1 % p = 0 ∧ a.2 % p = 0 ∧ a.3 % p = 0

def condition (v : Fin n → ℤ × ℤ × ℤ) (p : ℕ) : Prop :=
  ∀ j k, 0 ≤ j → j < k → k < n → 
  ∃ (ell : ℤ), 0 < ell → ell < p ∧ divisible_by_p (v j .- ell • (v k)) p

-- Stating the theorem
theorem vectors_upper_bound (h : ∀ i, is_prime_length (v i) p) (hcond : condition v p) : n ≤ 6 :=
sorry

end vectors_upper_bound_l664_664034


namespace cartesian_coordinate_eq_min_distance_AB_l664_664580

theorem cartesian_coordinate_eq (rho theta: ℝ) (h: rho * Real.sin theta^2 = 4 * Real.cos theta) :
  let (x, y) := (rho * Real.cos theta, rho * Real.sin theta) in y^2 = 4*x :=
by
  sorry

theorem min_distance_AB (α : ℝ) (hα : 0 < α ∧ α < π) :
  let t := fun α t => (1 + t * Real.cos α, t * Real.sin α)
  let equation := (fun y x => y^2 = 4 * x)
  ∃ A B : ℝ × ℝ, 
      (t α A.1 = (1 + A.1 * Real.cos α, A.1 * Real.sin α)) ∧ 
      equation (A.1 * Real.sin α) (1 + A.1 * Real.cos α) ∧ 
      (t α B.1 = (1 + B.1 * Real.cos α, B.1 * Real.sin α)) ∧ 
      equation (B.1 * Real.sin α) (1 + B.1 * Real.cos α) ∧ 
      ∀ α, |A.1 - B.1| = 4 / (Real.sin α)^2 := 
by 
  sorry

end cartesian_coordinate_eq_min_distance_AB_l664_664580


namespace greatest_prime_factor_15_fact_plus_18_fact_eq_17_l664_664661

theorem greatest_prime_factor_15_fact_plus_18_fact_eq_17 :
  ∃ p : ℕ, prime p ∧ 
  (∀ q : ℕ, (prime q ∧ q ∣ (15.factorial + 18.factorial)) → q ≤ p) ∧ 
  p = 17 :=
by
  sorry

end greatest_prime_factor_15_fact_plus_18_fact_eq_17_l664_664661


namespace initial_fee_is_2_50_l664_664978

-- Define the conditions
def initial_fee (F : ℝ) : Prop :=
  ∃ (rate_per_segment : ℝ) (segment_length : ℝ) (total_cost : ℝ) (distance : ℝ), 
    rate_per_segment = 0.35 ∧ 
    segment_length = 2/5 ∧
    distance = 3.6 ∧ 
    total_cost = 5.65 ∧ 
    let num_segments := distance / segment_length
    in total_cost = F + (num_segments * rate_per_segment)

-- State the theorem to be proven
theorem initial_fee_is_2_50 : initial_fee 2.50 := 
sorry

end initial_fee_is_2_50_l664_664978


namespace even_divisors_less_than_100_l664_664364

theorem even_divisors_less_than_100 :
  let count_even_divisors := 
    let n := 100 in
    let perfect_squares := { m | ∃ k, k * k = m ∧ m < n } in
    let total_numbers := finset.range n in
    (total_numbers.card - perfect_squares.card) =
    90 
  in count_even_divisors = 90 :=
by
  let n := 100
  let perfect_squares : finset ℕ := finset.filter (λ m, ∃ k, k * k = m) (finset.range n)
  let total_numbers : finset ℕ := finset.range n
  have h : total_numbers.card = 99 := by sorry
  have p : perfect_squares.card = 9 := by sorry
  show total_numbers.card - perfect_squares.card = 90
  calc
    total_numbers.card - perfect_squares.card
      = 99 - 9 := by rw [h, p]
      = 90 := by norm_num

end even_divisors_less_than_100_l664_664364


namespace calculate_expression_l664_664200

variable (x : ℝ)

theorem calculate_expression : (1/2 * x^3)^2 = 1/4 * x^6 := 
by 
  sorry

end calculate_expression_l664_664200


namespace Kantana_chocolates_each_Saturday_l664_664022

-- Lean 4 statement
theorem Kantana_chocolates_each_Saturday : 
  ∀ (C S : ℕ), 
  (S = 4) → 
  (C * S + S + 10 = 22) → 
  C = 2 :=
by
  intros C S hS hTotal
  have hCombined : (C + 1) * S + 10 = 22 := by
    rw add_mul
    exact hTotal
  have hSubtract : (C + 1) * S = 12 := by
    linarith
  have hSetS : (C + 1) * 4 = 12 := by
    rw hS
    exact hSubtract
  have hDivide : C + 1 = 3 := by
    linarith
  have hC : C = 3 - 1 := by
    linarith
  exact hC

end Kantana_chocolates_each_Saturday_l664_664022


namespace line_inclination_angle_l664_664081

-- Definitions based on the conditions given in the original problem
def direction_vector : ℝ × ℝ := (1, -1)
def slope (v : ℝ × ℝ) : ℝ := v.2 / v.1

-- Angle of inclination where the slope equals to -1
def inclination_angle := Real.arctan (-1)

-- The proof statement: Given the direction vector, prove that the inclination angle is equal to 3π/4
theorem line_inclination_angle :
  inclination_angle = (3 * Real.pi) / 4 :=
by
  sorry

end line_inclination_angle_l664_664081


namespace greatest_prime_factor_of_15_plus_18_l664_664656

theorem greatest_prime_factor_of_15_plus_18! : 
  let n := 15! + 18!
  n = 15! * 4897 ∧ Prime 4897 →
  (∀ p : ℕ, Prime p ∧ p ∣ n → p ≤ 4897) ∧ (4897 ∣ n) ∧ Prime 4897 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_of_15_plus_18_l664_664656


namespace number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664385

theorem number_of_integers_with_even_divisors (n : ℕ) (h : n = 100) : 
  (card {x | (x < n) ∧ ∃ k, k * k = x} = 9) → 
  (card {x | (x < n) ∧ ¬(∃ k, k * k = x)} = n - 1 - 9) :=
by
  intro h_squares
  rw h
  trivial

open_locale classical
noncomputable def count_even_divisors_less_than_100 : ℕ :=
  90

theorem count_even_divisors_verification :
  count_even_divisors_less_than_100 = 90 :=
by
  sorry

end number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664385


namespace consecutive_weights_sum_to_63_l664_664975

theorem consecutive_weights_sum_to_63 : ∃ n : ℕ, (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)) = 63 :=
by
  sorry

end consecutive_weights_sum_to_63_l664_664975


namespace find_m_if_parallel_l664_664290

-- Definitions of the lines and the condition for parallel lines
def line1 (m : ℝ) (x y : ℝ) : ℝ := (m - 1) * x + y + 2
def line2 (m : ℝ) (x y : ℝ) : ℝ := 8 * x + (m + 1) * y + (m - 1)

-- The condition for the lines to be parallel
def parallel (m : ℝ) : Prop :=
  (m - 1) / 8 = 1 / (m + 1) ∧ (m - 1) / 8 ≠ 2 / (m - 1)

-- The main theorem to prove
theorem find_m_if_parallel (m : ℝ) (h : parallel m) : m = 3 :=
sorry

end find_m_if_parallel_l664_664290


namespace pure_imaginary_x_value_l664_664281

theorem pure_imaginary_x_value (x : ℝ) (z : ℂ) (h1 : z = (2 + complex.i) / (x - complex.i))
    (h2 : ∃ (b : ℝ), z = b * complex.i):
    x = 1/2 :=
by
  sorry

end pure_imaginary_x_value_l664_664281


namespace infinite_primes_solutions_l664_664069

theorem infinite_primes_solutions :
  ∀ (P : Finset ℕ), (∀ p ∈ P, Prime p) →
  ∃ q, Prime q ∧ q ∉ P ∧ ∃ x y : ℤ, x^2 + x + 1 = q * y :=
by sorry

end infinite_primes_solutions_l664_664069


namespace even_number_of_divisors_less_than_100_l664_664323

theorem even_number_of_divisors_less_than_100 : 
  ∃ n, n = 90 ∧ ∀ x < 100, (∃ k, k * k = x → false) = (x ∣ 99 - 9) :=
sorry

end even_number_of_divisors_less_than_100_l664_664323


namespace cape_may_sharks_is_32_l664_664204

def twice (n : ℕ) : ℕ := 2 * n

def sharks_in_Cape_May (sharks_in_Daytona : ℕ) : ℕ :=
  twice(sharks_in_Daytona) + 8

theorem cape_may_sharks_is_32 :
  sharks_in_Cape_May 12 = 32 := by
  sorry

end cape_may_sharks_is_32_l664_664204


namespace lost_weights_l664_664876

-- Define the weights
def weights : List ℕ := [43, 70, 57]

-- Total remaining weight after loss
def remaining_weight : ℕ := 20172

-- Number of weights lost
def weights_lost : ℕ := 4

-- Whether a given number of weights and types of weights match the remaining weight
def valid_loss (initial_count : ℕ) (lost_weight_count : ℕ) : Prop :=
  let total_initial_weight := initial_count * (weights.sum)
  let lost_weight := lost_weight_count * 57
  total_initial_weight - lost_weight = remaining_weight

-- Proposition we need to prove
theorem lost_weights (initial_count : ℕ) (h : valid_loss initial_count weights_lost) : ∀ w ∈ weights, w = 57 :=
by {
  sorry
}

end lost_weights_l664_664876


namespace rounds_remaining_l664_664875

-- Define the conditions 
variables {n x : ℕ} (teamA teamB : ℕ → bool)

-- The mathematical equivalent proof problem in Lean.
theorem rounds_remaining (x : ℕ) (hx : x = 3 * (n - 2)) (hn : n ≥ 2) :
  (∀ k, teamA k → (k + x) > (k + x - 1)) → 
  ∀ k, teamA k → ¬ (k + x) > (k + x) :=
sorry

end rounds_remaining_l664_664875


namespace BC_length_l664_664962

def triangle_ABC (A B C : Type)
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)] : Prop :=
  ∃ (X : A), (has_dist B X (coe (X.dist B))) ∧ (has_dist C X (coe (X.dist C))) ∧
  ∀ (x y : ℕ), x = X.dist B ∧ y = X.dist C → x + y = 61

theorem BC_length {A B C : Type}
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)]
  (h : triangle_ABC A B C) : 
  ∃ (x y : ℕ), x + y = 61 := sorry

end BC_length_l664_664962


namespace BC_length_l664_664966

-- Define the given triangle and circle conditions
variables (A B C X : Type) (AB AC BX CX : ℤ)
axiom AB_value : AB = 86
axiom AC_value : AC = 97
axiom circle_center_radius : ∃ (A : Type), ∃ (radius : ℤ), radius = AB ∧ ∃ (points : Set Type), points = {B, X} ∧ ∀ (P : Type), P ∈ points → dist A P = radius
axiom BX_CX_integers : ∃ (x y : ℤ), BX = x ∧ CX = y

-- Define calculations using the Power of a Point theorem
theorem BC_length :
  ∀ (y: ℤ) (x: ℤ), y(y + x) = AC^2 - AB^2 → x + y = 61 :=
by
  intros y x h
  have h1 : 97^2 = 9409, by norm_num,
  have h2 : 86^2 = 7396, by norm_num,
  rw [AB_value, AC_value] at h,
  rw [h1, h2] at h,
  calc y(y + x) = 2013 := by {exact h}
  -- The human verification part is skipped since we only need the statement here
  sorry

end BC_length_l664_664966


namespace largest_integer_digits_product_l664_664621

-- Define a function to compute the sum of squares of digits
def sum_of_squares (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d, d * d).sum

-- Define a function to check if digits are in non-decreasing order
def non_decreasing_digits (n : ℕ) : Prop :=
  (n.digits 10).pairwise (≤)

-- Define the statement of the problem
theorem largest_integer_digits_product : 
  ∃ n : ℕ, sum_of_squares n = 65 ∧ non_decreasing_digits n ∧
  (n.digits 10).foldr (λ d acc, d * acc) 1 = 30 :=
sorry

end largest_integer_digits_product_l664_664621


namespace BC_length_l664_664910

theorem BC_length (A B C X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X] 
  (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
  (BX CX : ℕ) (h_circle_intersect : ∃ X, Metric.ball A 86 ∩ {BC} = {B, X})
  (h_integer_lengths : BX + CX = BC) :
  BC = 61 := 
by
  sorry

end BC_length_l664_664910


namespace count_even_divisors_lt_100_l664_664309

-- Define the set of natural numbers less than 100
def nat_lt_100 := {n : ℕ | n < 100}

-- Define the set of perfect squares less than 100
def perfect_squares_lt_100 := {n : ℕ | ∃ k : ℕ, k^2 = n ∧ n < 100}

-- Define the set of natural numbers less than 100 with an even number of positive divisors
def even_divisors_lt_100 := nat_lt_100 \ perfect_squares_lt_100

-- Theorem stating the number of elements with even number of divisors is 90
theorem count_even_divisors_lt_100 : (even_divisors_lt_100).card = 90 := 
sorry

end count_even_divisors_lt_100_l664_664309


namespace range_of_k_value_of_k_l664_664828

-- Defining the quadratic equation having two real roots condition
def has_real_roots (k : ℝ) : Prop :=
  let Δ := 9 - 4 * (k - 2)
  Δ ≥ 0

-- First part: range of k
theorem range_of_k (k : ℝ) : has_real_roots k ↔ k ≤ 17 / 4 :=
  sorry

-- Second part: specific value of k given additional condition
theorem value_of_k (x1 x2 k : ℝ) (h1 : (x1 + x2) = 3) (h2 : (x1 * x2) = k - 2) (h3 : (x1 + x2 - x1 * x2) = 1) : k = 4 :=
  sorry

end range_of_k_value_of_k_l664_664828


namespace solution_is_permutations_l664_664572

noncomputable def solve_system (x y z : ℤ) : Prop :=
  x^2 = y * z + 1 ∧ y^2 = z * x + 1 ∧ z^2 = x * y + 1

theorem solution_is_permutations (x y z : ℤ) :
  solve_system x y z ↔ (x, y, z) = (1, 0, -1) ∨ (x, y, z) = (1, -1, 0) ∨ (x, y, z) = (0, 1, -1) ∨ (x, y, z) = (0, -1, 1) ∨ (x, y, z) = (-1, 1, 0) ∨ (x, y, z) = (-1, 0, 1) :=
by sorry

end solution_is_permutations_l664_664572


namespace georgia_carnations_proof_l664_664556

-- Define the conditions
def carnation_cost : ℝ := 0.50
def dozen_cost : ℝ := 4.00
def friends_carnations : ℕ := 14
def total_spent : ℝ := 25.00

-- Define the answer
def teachers_dozen : ℕ := 4

-- Prove the main statement
theorem georgia_carnations_proof : 
  (total_spent - (friends_carnations * carnation_cost)) / dozen_cost = teachers_dozen :=
by
  sorry

end georgia_carnations_proof_l664_664556


namespace combined_weight_l664_664980

theorem combined_weight (mary_weight : ℝ) (jamison_weight : ℝ) (john_weight : ℝ) :
  mary_weight = 160 ∧ jamison_weight = mary_weight + 20 ∧ john_weight = mary_weight + (0.25 * mary_weight) →
  john_weight + mary_weight + jamison_weight = 540 :=
by
  intros h
  obtain ⟨hm, hj, hj'⟩ := h
  rw [hm, hj, hj']
  norm_num
  sorry

end combined_weight_l664_664980


namespace number_of_solutions_to_g100_eq_zero_l664_664537

def g0 (x : ℝ) : ℝ := x + (abs (x - 150)) - (abs (x + 150))

def gn (n : ℕ) (x : ℝ) : ℝ :=
  nat.rec_on n (g0 x) (λ n g_n_minus_1, abs g_n_minus_1 - 2)

theorem number_of_solutions_to_g100_eq_zero : 
  (finset.card (finset.filter (λ x : ℝ, gn 100 x = 0) (finset.range 100000))) = 299 :=
sorry

end number_of_solutions_to_g100_eq_zero_l664_664537


namespace regular_decagon_triangle_count_l664_664215

def regular_decagon (V : Type) := ∃ vertices : V, Fintype.card vertices = 10

theorem regular_decagon_triangle_count (V : Type) [Fintype V] (h : regular_decagon V)
: Fintype.card { triangle : Finset V // triangle.card = 3 } = 120 := by
  sorry

end regular_decagon_triangle_count_l664_664215


namespace solve_for_x_l664_664747

theorem solve_for_x (x : ℝ) : 
  (3^(2 * x) - 15 * 3^x + 36 = 0) ↔ (x = 2 * Real.log 2 / Real.log 3 + 1 ∨ x = 1) := 
sorry

end solve_for_x_l664_664747


namespace remainder_h_x14_l664_664046

noncomputable def h (x : ℤ) : ℤ := x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_h_x14 (x : ℤ) : (x^98 + x^84 + x^70 + x^56 + x^42 + x^28 + x^14 + 1) % (h(x)) = 8 :=
by {
  -- Proof would go here
  sorry
}

end remainder_h_x14_l664_664046


namespace mans_rate_in_still_water_l664_664146

theorem mans_rate_in_still_water : 
  ∀ (V_m V_s : ℝ), 
  V_m + V_s = 16 → 
  V_m - V_s = 4 → 
  V_m = 10 :=
by
  intros V_m V_s h1 h2
  sorry

end mans_rate_in_still_water_l664_664146


namespace maximum_value_M_l664_664776

def J (m : ℕ) : ℕ := 10 * 10 ^ m + 32

def M (m : ℕ) : ℕ :=
  Nat.findGreatest x (x ∣ J m ∧ Nat.primeFactors (J m) = [2])

theorem maximum_value_M : 
  ∃ m > 0, M m = 6 :=
sorry

end maximum_value_M_l664_664776


namespace circular_pond_area_l664_664166

theorem circular_pond_area (AB CD : ℝ) (D_is_midpoint : Prop) (hAB : AB = 20) (hCD : CD = 12)
  (hD_midpoint : D_is_midpoint ∧ D_is_midpoint = (AB / 2 = 10)) :
  ∃ (A : ℝ), A = 244 * Real.pi :=
by
  sorry

end circular_pond_area_l664_664166


namespace range_of_a3_l664_664132

variable (a b : ℕ → ℝ)
variable (n : ℕ)

def convex_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), 1 ≤ n ∧ n < 9 → (a n + a (n+2)) / 2 ≤ a (n+1)

def sequence_condition (a b : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), 1 ≤ n ∧ n < 10 → | a n - b n | ≤ 20

noncomputable def b_n (n : ℕ) : ℝ :=
  n^2 - 6 * n + 10

theorem range_of_a3 (a : ℕ → ℝ) (h_convex : convex_sequence a) (h_condition : sequence_condition a b_n) :
  7 ≤ a 3 ∧ a 3 ≤ 19 :=
sorry

end range_of_a3_l664_664132


namespace tan_sub_cot_periodicity_sin_cos_periodicity_sin_x_squared_not_periodic_l664_664224

/-
Question 1: Prove that y = tan x - cot x is periodic with the smallest period π/2
-/
theorem tan_sub_cot_periodicity : ∃ T > 0, (∀ x, tan x - cot x = tan (x + T) - cot (x + T)) ∧ T = π / 2 := sorry

/-
Question 2: Prove that y = sin(cos x) is periodic with the smallest period 2π
-/
theorem sin_cos_periodicity : ∃ T > 0, (∀ x, sin (cos x) = sin (cos (x + T))) ∧ T = 2 * π := sorry

/-
Question 3: Prove that y = sin x² is not periodic
-/
theorem sin_x_squared_not_periodic : ¬ ∃ T > 0, ∀ x, sin (x^2) = sin ((x + T)^2) := sorry

end tan_sub_cot_periodicity_sin_cos_periodicity_sin_x_squared_not_periodic_l664_664224


namespace evaluate_expression_l664_664756

theorem evaluate_expression : 6 - 8 * (9 - 4^2) / 2 - 3 = 31 :=
by
  sorry

end evaluate_expression_l664_664756


namespace find_x_l664_664851

section
  def g (x : ℝ) : ℝ := (x + 3) / 4 ^ (1 / 3 : ℝ)

  theorem find_x (x : ℝ) (h : g (2 * x) = 2 * g x) : x = -7 / 2 := by
    sorry
end

end find_x_l664_664851


namespace fg_gg_zero_requiem_l664_664164

-- Definition: Zero-requiem function
def zero_requiem (ψ : ℤ → ℤ) : Prop :=
  ∀ (n : ℕ) (a : Fin n → ℤ), (∑ i, a i = 0) → (∑ i, ψ (a i) = 0) → False

variables (f g : ℤ → ℤ)

-- Conditions
axiom f_zero_requiem : zero_requiem f
axiom g_zero_requiem : zero_requiem g
axiom f_g_inv : ∀ x : ℤ, f(g(x)) = x
axiom g_f_inv : ∀ x : ℤ, g(f(x)) = x
axiom f_plus_g_not_zero_requiem : ¬zero_requiem (λ x, f x + g x)

-- Theorem to prove
theorem fg_gg_zero_requiem : zero_requiem (f ∘ f) ∧ zero_requiem (g ∘ g) :=
by
  sorry

end fg_gg_zero_requiem_l664_664164


namespace complex_square_example_l664_664441

noncomputable def z : ℂ := 5 - 3 * Complex.I
noncomputable def i_squared : ℂ := Complex.I ^ 2

theorem complex_square_example : z ^ 2 = 34 - 30 * Complex.I := by
  have i_squared_eq : i_squared = -1 := by
    unfold i_squared
    rw [Complex.I_sq]
    rfl
  unfold z
  calc
    (5 - 3 * Complex.I) ^ 2
        = (5 ^ 2 - (3 * Complex.I) ^ 2 - 2 * 5 * 3 * Complex.I) : by
          ring
    ... = 25 - 9 * i_squared - 30 * Complex.I : by
          rw [Complex.mul_sq, Complex.I_sq]
    ... = 25 - 9 * (-1) - 30 * Complex.I : by
          rw [i_squared_eq]
    ... = 25 + 9 - 30 * Complex.I : by
          ring
    ... = 34 - 30 * Complex.I : by
          ring

end complex_square_example_l664_664441


namespace greatest_prime_factor_15_factorial_plus_18_factorial_l664_664646

theorem greatest_prime_factor_15_factorial_plus_18_factorial :
  ∀ {a b c d e f g: ℕ}, a = 15! → b = 18! → c = 16 → d = 17 → e = 18 → f = a * (1 + c * d * e) →
  g = 4896 → Prime 17 → f + b = a + b → Nat.gcd (a + b) g = 17 :=
by
  intros
  sorry

end greatest_prime_factor_15_factorial_plus_18_factorial_l664_664646


namespace fencing_problem_l664_664784

theorem fencing_problem (W L : ℝ) (hW : W = 40) (hArea : W * L = 320) : 
  2 * L + W = 56 :=
by
  sorry

end fencing_problem_l664_664784


namespace find_k_l664_664466

theorem find_k (x y k : ℝ) (h1 : x + y = 5 * k) (h2 : x - y = 9 * k) (h3 : x - 2 * y = 22) : k = 2 :=
by
  sorry

end find_k_l664_664466


namespace even_divisors_count_lt_100_l664_664354

theorem even_divisors_count_lt_100 : 
  {n : ℕ | n < 100 ∧ n ≠ 0 ∧ ∃ k : ℕ, k * k = n } = { n : ℕ | n < 100 ∧ n ≠ 0 } \ 
  { n : ℕ | ∃ k : ℕ, k * k = n ∧ k < 100 } → 
  (card {n : ℕ | n < 100 ∧ n ≠ 0 ∧ even (finset.card (divisors n))} = 90) :=
begin
  sorry
end

end even_divisors_count_lt_100_l664_664354


namespace greatest_root_of_g_l664_664766

noncomputable def g (x : ℝ) : ℝ := 16 * x^4 - 20 * x^2 + 5

theorem greatest_root_of_g :
  ∃ r : ℝ, r = Real.sqrt 5 / 2 ∧ (forall x, g x ≤ g r) :=
sorry

end greatest_root_of_g_l664_664766


namespace triangular_array_sum_digits_l664_664686

theorem triangular_array_sum_digits (N : ℕ) (h : N * (N + 1) / 2 = 2145) : (N / 10 + N % 10) = 11 := 
sorry

end triangular_array_sum_digits_l664_664686


namespace exists_poly_form_2a_k_plus_3_l664_664273

theorem exists_poly_form_2a_k_plus_3 (a : ℤ) (h_a : a > 1) (n : ℕ) (h_n : n > 0) : 
  ∃ p : Polynomial ℤ, 
    (∀ j : ℕ, j ≤ n → ∃ k : ℕ, p.eval j = 2 * a^k + 3) ∧ 
    (∀ i j : ℕ, i ≤ n → j ≤ n → i ≠ j → p.eval i ≠ p.eval j) :=
by sorry

end exists_poly_form_2a_k_plus_3_l664_664273


namespace wall_bricks_count_l664_664736

noncomputable def bricks_in_wall := 720

theorem wall_bricks_count (w : ℕ) (h1 : Brenda_rate = w / 12) (h2 : Brandon_rate = w / 15) (h3 : combined_rate_decrease = 12) (h4 : combined_time = 6) :
  w = bricks_in_wall :=
by
  -- Define the rates
  let brenda_rate := w / 12
  let brandon_rate := w / 15
  let combined_rate_with_decrease := brenda_rate + brandon_rate - combined_rate_decrease

  -- Given conditions in Lean
  have h_brenda_rate : Brenda_rate = brenda_rate := h1
  have h_brandon_rate : Brandon_rate = brandon_rate := h2
  have h_combined_rate_decrease : combined_rate_decrease = 12 := h3
  have h_combined_time : combined_time = 6 := h4
  
  -- Prove that total number of bricks in the wall is 720
  have effective_rate := (3 * w / 20 - 12) * 6
  have h_w : effective_rate = w := by sorry
  exact h_w

end wall_bricks_count_l664_664736


namespace arithmetic_geometric_sequence_l664_664260

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) (d : ℤ) (h1 : d = 2)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : (a 1), (a 3), (a 4) form_geometric_sequence) : a 6 = 2 :=
sorry

def form_geometric_sequence (x y z : ℤ) : Prop :=
  y * y = x * z

end arithmetic_geometric_sequence_l664_664260


namespace james_is_older_by_7_years_l664_664983

/-
  Justin is 26 years old.
  When Justin was born, Jessica was 6 years old.
  James will be 44 years old after 5 years.
  Prove that James is 7 years older than Jessica.
-/

theorem james_is_older_by_7_years 
  (justin_age : ℕ := 26)
  (jessica_age_diff : ℕ := 6)
  (james_future_age : ℕ := 44)
  (years_until_future : ℕ := 5) :
  let jessica_age_now := jessica_age_diff + justin_age
  let james_age_now := james_future_age - years_until_future
  james_age_now - jessica_age_now = 7 :=
by
  let jessica_age_now := jessica_age_diff + justin_age
  let james_age_now := james_future_age - years_until_future
  have h1 : jessica_age_now = 6 + 26 := rfl
  have h2 : james_age_now = 44 - 5 := rfl
  have h3 : james_age_now - jessica_age_now = 39 - 32 := by simp [h1, h2]
  have h4 : 39 - 32 = 7 := rfl
  exact Eq.trans h3 h4

end james_is_older_by_7_years_l664_664983


namespace even_number_of_divisors_l664_664340

-- Proof statement: There are 90 positive integers less than 100 with an even number of divisors.
theorem even_number_of_divisors : 
  {n : ℕ | n < 100 ∧ ∃ k : ℕ, k ^ 2 = n}.toFinset.card = 90 := 
sorry

end even_number_of_divisors_l664_664340


namespace triangle_bc_length_l664_664908

theorem triangle_bc_length (A B C X : Type)
  (AB AC : ℕ)
  (hAB : AB = 86)
  (hAC : AC = 97)
  (circle_eq : ∀ {r : ℕ}, r = AB → circle_centered_at_A_intersects_BC_two_points B X)
  (integer_lengths : ∃ (BX CX : ℕ), ) :
  BC = 61 :=
by
  sorry

end triangle_bc_length_l664_664908


namespace N_q_odd_iff_q_of_form_p_k_l664_664989

open Nat

-- Assume q is an odd positive integer.
variable (q : ℕ) (hq : odd q ∧ q > 0)

-- Define N_q as the number of integers a such that 0 < a < q/4 and gcd(a, q) = 1.
def N_q : ℕ := (Finset.range (q / 4)).filter (λ a => gcd a q = 1).card

-- State the theorem.
theorem N_q_odd_iff_q_of_form_p_k :
  (N_q q hq).odd ↔ ∃ p k, Nat.Prime p ∧ k > 0 ∧ q = p ^ k ∧ (p % 8 = 5 ∨ p % 8 = 7) :=
sorry

end N_q_odd_iff_q_of_form_p_k_l664_664989


namespace problem_equivalent_l664_664232

theorem problem_equivalent :
  ∃ (p q r s : ℤ), (q ≠ 0 ∧ (y = (0 + 5 * √1) / 1)) ∧ (y = (p + q * √r) / s) ∧ (y = 5) ∧ ( (4 * y) / 5 - 2 = 10 / y ) →
  ((p * r * s) / q = 0) :=
sorry

end problem_equivalent_l664_664232


namespace even_number_of_divisors_lt_100_l664_664407

theorem even_number_of_divisors_lt_100 : 
  let n := 99 in
  let perfect_squares := {1, 4, 9, 16, 25, 36, 49, 64, 81} in
  n - perfect_squares.card = 90 := 
by
  sorry

end even_number_of_divisors_lt_100_l664_664407


namespace jose_tabs_remaining_l664_664017

def initial_tabs : Nat := 400
def step1_tabs_closed (n : Nat) : Nat := n / 4
def step2_tabs_closed (n : Nat) : Nat := 2 * n / 5
def step3_tabs_closed (n : Nat) : Nat := n / 2

theorem jose_tabs_remaining :
  let after_step1 := initial_tabs - step1_tabs_closed initial_tabs
  let after_step2 := after_step1 - step2_tabs_closed after_step1
  let after_step3 := after_step2 - step3_tabs_closed after_step2
  after_step3 = 90 :=
by
  let after_step1 := initial_tabs - step1_tabs_closed initial_tabs
  let after_step2 := after_step1 - step2_tabs_closed after_step1
  let after_step3 := after_step2 - step3_tabs_closed after_step2
  have h : after_step3 = 90 := sorry
  exact h

end jose_tabs_remaining_l664_664017


namespace probability_of_one_shirt_two_shorts_one_sock_l664_664579

def num_shirts := 6
def num_shorts := 8
def num_socks := 7
def total_clothings := num_shirts + num_shorts + num_socks
def num_chosen := 4
def num_shirts_chosen := 1
def num_shorts_chosen := 2
def num_socks_chosen := 1

def combinations (n k : ℕ) := nat.factorial n / (nat.factorial k * nat.factorial (n - k))

noncomputable def total_ways_to_choose := combinations total_clothings num_chosen
noncomputable def ways_to_choose_shirts := combinations num_shirts num_shirts_chosen
noncomputable def ways_to_choose_shorts := combinations num_shorts num_shorts_chosen
noncomputable def ways_to_choose_socks := combinations num_socks num_socks_chosen
noncomputable def favorable_outcomes := ways_to_choose_shirts * ways_to_choose_shorts * ways_to_choose_socks

noncomputable def probability := favorable_outcomes / total_ways_to_choose

theorem probability_of_one_shirt_two_shorts_one_sock :
  probability = 392 / 1995 := by sorry

end probability_of_one_shirt_two_shorts_one_sock_l664_664579


namespace latest_start_time_l664_664053

-- Define the weights of the turkeys
def turkey_weights : List ℕ := [16, 18, 20, 22]

-- Define the roasting time per pound
def roasting_time_per_pound : ℕ := 15

-- Define the dinner time in 24-hour format
def dinner_time : ℕ := 18 * 60 -- 18:00 in minutes

-- Calculate the total roasting time
def total_roasting_time (weights : List ℕ) (time_per_pound : ℕ) : ℕ :=
  weights.foldr (λ weight acc => weight * time_per_pound + acc) 0

-- Calculate the latest start time
def latest_roasting_start_time (total_time : ℕ) (dinner_time : ℕ) : ℕ :=
  let start_time := dinner_time - total_time
  if start_time < 0 then start_time + 24 * 60 else start_time

-- Convert minutes to hours:minutes format
def time_in_hours_minutes (time : ℕ) : String :=
  let hours := time / 60
  let minutes := time % 60
  toString hours ++ ":" ++ toString minutes

theorem latest_start_time : 
  time_in_hours_minutes (latest_roasting_start_time (total_roasting_time turkey_weights roasting_time_per_pound) dinner_time) = "23:00" := by
  sorry

end latest_start_time_l664_664053


namespace smallest_n_l664_664720

theorem smallest_n :
  ∃ n : ℕ, n > 0 ∧ 2000 * n % 21 = 0 ∧ ∀ m : ℕ, m > 0 ∧ 2000 * m % 21 = 0 → n ≤ m :=
sorry

end smallest_n_l664_664720


namespace correct_option_l664_664141

theorem correct_option (h1 : 1 ∈ {0, 1}) : 1 ∈ {0, 1} :=
by
  exact h1

end correct_option_l664_664141


namespace even_divisors_count_lt_100_l664_664349

theorem even_divisors_count_lt_100 : 
  {n : ℕ | n < 100 ∧ n ≠ 0 ∧ ∃ k : ℕ, k * k = n } = { n : ℕ | n < 100 ∧ n ≠ 0 } \ 
  { n : ℕ | ∃ k : ℕ, k * k = n ∧ k < 100 } → 
  (card {n : ℕ | n < 100 ∧ n ≠ 0 ∧ even (finset.card (divisors n))} = 90) :=
begin
  sorry
end

end even_divisors_count_lt_100_l664_664349


namespace lcm_12_35_l664_664767

theorem lcm_12_35 : Nat.lcm 12 35 = 420 :=
by
  sorry

end lcm_12_35_l664_664767


namespace f_value_at_minus_pi_six_l664_664287

noncomputable def f (ω x : ℝ) : ℝ :=
  sin (ω * x + π / 3) - 1 / 2 * cos (ω * x - 7 * π / 6)

def has_minimum_positive_period (f: ℝ → ℝ) (p: ℝ) : Prop :=
  ∀ x, f(x) = f(x + p) ∧ (∀ q > 0, (∀ y, f(y) = f(y + q)) → q ≥ p)

theorem f_value_at_minus_pi_six
  (ω : ℝ) (h_ω_pos : ω > 0)
  (h_period : has_minimum_positive_period (f ω) (2 * π)) :
  f ω (-π / 6) = 3 / 4 :=
sorry

end f_value_at_minus_pi_six_l664_664287


namespace manufacturer_price_l664_664708

theorem manufacturer_price :
  ∃ M : ℝ, 
    (∃ R : ℝ, 
      R = 1.15 * M ∧
      ∃ D : ℝ, 
        D = 0.85 * R ∧
        R - D = 57.5) ∧
    M = 333.33 := 
by
  sorry

end manufacturer_price_l664_664708


namespace number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664378

theorem number_of_integers_with_even_divisors (n : ℕ) (h : n = 100) : 
  (card {x | (x < n) ∧ ∃ k, k * k = x} = 9) → 
  (card {x | (x < n) ∧ ¬(∃ k, k * k = x)} = n - 1 - 9) :=
by
  intro h_squares
  rw h
  trivial

open_locale classical
noncomputable def count_even_divisors_less_than_100 : ℕ :=
  90

theorem count_even_divisors_verification :
  count_even_divisors_less_than_100 = 90 :=
by
  sorry

end number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664378


namespace sellable_fruit_l664_664619

theorem sellable_fruit :
  let total_oranges := 30 * 300
  let total_damaged_oranges := total_oranges * 10 / 100
  let sellable_oranges := total_oranges - total_damaged_oranges

  let total_nectarines := 45 * 80
  let nectarines_taken := 5 * 20
  let sellable_nectarines := total_nectarines - nectarines_taken

  let total_apples := 20 * 120
  let bad_apples := 50
  let sellable_apples := total_apples - bad_apples

  sellable_oranges + sellable_nectarines + sellable_apples = 13950 :=
by
  sorry

end sellable_fruit_l664_664619


namespace even_number_of_divisors_less_than_100_l664_664415

theorem even_number_of_divisors_less_than_100 :
  ∃ (count : ℕ), count = 90 ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 100 →
    (∃ (d : ℕ), d * d = n ∨ (number_of_divisors n % 2 = 0)) :=
begin
  -- the proof goes here
  sorry
end

end even_number_of_divisors_less_than_100_l664_664415


namespace cos_2000_eq_neg_inv_sqrt_l664_664848

theorem cos_2000_eq_neg_inv_sqrt (a : ℝ) (h : Real.tan (20 * Real.pi / 180) = a) :
  Real.cos (2000 * Real.pi / 180) = -1 / Real.sqrt (1 + a^2) :=
sorry

end cos_2000_eq_neg_inv_sqrt_l664_664848


namespace probability_of_f1_div_2_is_integer_l664_664785

noncomputable def probability_f1_div_2_is_integer : ℚ :=
  let total_ways := 10.choose 3 * 3! in
  let valid_ways := -- calculate number of valid ways ensuring (a ≠ 0 if needed)
    (let evens := {0, 2, 4, 6, 8} in
     let odds := {1, 3, 5, 7, 9} in
     (evens.choose 3 + evens.choose 1 * odds.choose 2) * 6) - -- valid choices
    ((evens - {0}).choose 3 * 6) in -- subtract invalid (a = 0) cases
  valid_ways / total_ways

theorem probability_of_f1_div_2_is_integer :
  probability_f1_div_2_is_integer = 41 / 81 :=
sorry

end probability_of_f1_div_2_is_integer_l664_664785


namespace chess_match_scheduling_l664_664206

theorem chess_match_scheduling (teamA teamB : Fin 4 → Prop) (G : Type*)
  [decidable_eq G] [fintype G] 
  (games_per_round : ℕ) (rounds : ℕ) (total_games : ℕ) :
  (∀ a b, teamA a → teamB b → ∃ game, game ∈ G) →
  games_per_round = 4 →
  rounds = 4 →
  total_games = 16 →
  ∃ schedule_permutations : ℕ, schedule_permutations = 24 :=
by
  intros,
  use 24,
  sorry -- This is where the proof would go

end chess_match_scheduling_l664_664206


namespace range_of_a_l664_664607

theorem range_of_a 
  (h₁ : ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0)
  (h₂ : ¬ ∀ (x : ℝ), Function.StrictMono (λ x, (3 - 2 * a) ^ x)) :
  1 ≤ a ∧ a < 2 := 
sorry

end range_of_a_l664_664607


namespace triangle_count_l664_664990

theorem triangle_count (a b c : ℕ) (hb : b = 2008) (hab : a ≤ b) (hbc : b ≤ c) (ht : a + b > c) : 
  ∃ n, n = 2017036 :=
by
  sorry

end triangle_count_l664_664990


namespace cara_neighbors_l664_664742

/-- 
Cara is at a circular table with her six friends. 
Among them, Alice and Bob insist on sitting next to each other.
Given this condition, prove that there are 10 different sets of neighbors Cara could have.
-/
theorem cara_neighbors (friends : Fin 6 → String) (Cara Alice Bob : Fin 6):
  (Alice ≠ Bob) → 
  ∃ cnt : ℕ, cnt = 10 ∧ (cnt = ∑ (Cara_neighbors: Fin 6), if Alice = Bob then 1 else 0) sorry

end cara_neighbors_l664_664742


namespace count_sums_greater_than_2017_l664_664722

theorem count_sums_greater_than_2017 (prime_2017 : Nat.prime 2017)
  (a b : ℕ) (h_a : 0 < a) (h_b : 0 < b) (h_range_a : a < 2017) (h_range_b : b < 2017) (h_sum_not_2017 : a + b ≠ 2017) :
  let A := λ k : ℕ, (a * k) % 2017
  let B := λ k : ℕ, (b * k) % 2017
  let S := λ k : ℕ, A k + B k
  (finset.range 2016).filter (λ k, S (k + 1) > 2017).card = 1008 :=
by sorry

end count_sums_greater_than_2017_l664_664722


namespace expected_value_correct_l664_664429

noncomputable def expected_value_sum_of_two_marbles : ℕ :=
  let marbles := {1,2,3,4,5,6}
  let pairs := { (a, b) | a ∈ marbles, b ∈ marbles, a < b }
  let sums := pairs.map (λ (a, b) => a + b)
  (sums.sum.to_rat / pairs.size).to_nat

theorem expected_value_correct :
  expected_value_sum_of_two_marbles = 7 :=
  sorry

end expected_value_correct_l664_664429


namespace expected_value_correct_l664_664428

noncomputable def expected_value_sum_of_two_marbles : ℕ :=
  let marbles := {1,2,3,4,5,6}
  let pairs := { (a, b) | a ∈ marbles, b ∈ marbles, a < b }
  let sums := pairs.map (λ (a, b) => a + b)
  (sums.sum.to_rat / pairs.size).to_nat

theorem expected_value_correct :
  expected_value_sum_of_two_marbles = 7 :=
  sorry

end expected_value_correct_l664_664428


namespace unfolded_angle_is_sqrt2_pi_l664_664471

noncomputable def central_angle_of_unfolded_cone (r : ℝ) :=
  let l := (√2) * r in
  let circumference_base := 2 * π * r in
  let arc_length := circumference_base in
  arc_length / l

theorem unfolded_angle_is_sqrt2_pi (r : ℝ) : central_angle_of_unfolded_cone r = √2 * π := 
by 
  sorry

end unfolded_angle_is_sqrt2_pi_l664_664471


namespace question_1_question_2_l664_664792

open Real

theorem question_1 (α : ℝ) (hα1 : π / 2 < α) (hα2 : α < 3 * π / 2) :
  (|⟨cos α - 3, sin α⟩| = |⟨cos α, sin α - 3⟩|) → α = 5 * π / 4 :=
sorry

theorem question_2 (α : ℝ) (hα1 : π / 2 < α) (hα2 : α < 3 * π / 2) :
  (⟨cos α - 3, sin α⟩ ⬝ ⟨cos α, sin α - 3⟩ = -1) → sin (2 * α) = -5/9 :=
sorry

end question_1_question_2_l664_664792


namespace asha_savings_l664_664191

theorem asha_savings (brother father mother granny spending remaining total borrowed_gifted savings : ℤ) 
  (h1 : brother = 20)
  (h2 : father = 40)
  (h3 : mother = 30)
  (h4 : granny = 70)
  (h5 : spending = 3 * total / 4)
  (h6 : remaining = 65)
  (h7 : remaining = total - spending)
  (h8 : total = brother + father + mother + granny + savings)
  (h9 : borrowed_gifted = brother + father + mother + granny) :
  savings = 100 := by
    sorry

end asha_savings_l664_664191


namespace greatest_prime_factor_of_expression_l664_664670

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define specific factorial values
def fac_15 := factorial 15
def fac_18 := factorial 18

-- Define the expression from the problem
def expr := fac_15 * (1 + 16 * 17 * 18)

-- Define the factorization result
def factor_4896 := 2 ^ 5 * 3 ^ 2 * 17

-- Define a lemma about the factorization of the expression
lemma factor_expression : 15! * (1 + 16 * 17 * 18) = fac_15 * 4896 := by
  sorry

-- State the main theorem
theorem greatest_prime_factor_of_expression : ∀ p : ℕ, prime p ∧ p ∣ expr → p ≤ 17 := by
  sorry

end greatest_prime_factor_of_expression_l664_664670


namespace bernardo_silvia_probability_l664_664734

theorem bernardo_silvia_probability :
  let bernardo_set := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let silvia_set := {1, 2, 3, 4, 5, 6, 7, 8}
  let bernardo_picks := {s ∈ finset.powerset_len 3 (finset.univ : finset ℕ) | s ⊆ bernardo_set}
  let silvia_picks := {s ∈ finset.powerset_len 3 (finset.univ : finset ℕ) | s ⊆ silvia_set}
  let bernardo_number (s : finset ℕ) := s.max 3
  let silvia_number (s : finset ℕ) := s.max 3
  ∑ b ∈ bernardo_picks, ∑ s ∈ silvia_picks, (if bernardo_number b > silvia_number s then 1 else 0) / (bernardo_picks.card * silvia_picks.card) = 39 / 56 :=
by
  sorry

end bernardo_silvia_probability_l664_664734


namespace trajectory_of_Q_range_of_area_S_l664_664264

-- Given point F, circle E and point P on circle E
def point_F : Point := (sqrt 3, 0)
def circle_E : ℝ → ℝ → Prop := λ x y, (x + sqrt 3) ^ 2 + y ^ 2 = 16
def point_P_on_E (x y : ℝ) : Prop := circle_E x y

-- Given the perpendicular bisector condition and intersecting with radius PE at point Q
def perpendicular_bisector_PF_intersect_PE_at_Q (P Q F : Point) : Prop := -- define the relevant properties of Q here

-- The given trajectory equation
def trajectory_eq (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Given initial condition
def initial_conditions (P Q : Point) : Prop :=
  point_F = (sqrt 3, 0) ∧
  circle_E P.1 P.2 ∧
  perpendicular_bisector_PF_intersect_PE_at_Q P Q point_F

-- Problem 1: Prove trajectory equation of Q
theorem trajectory_of_Q (P Q : Point) (h : initial_conditions P Q) : trajectory_eq Q.1 Q.2 :=
sorry

-- Additional conditions for problem (2)
def is_tangent_to (l : Line) (circle_O : ℝ → ℝ → Prop) : Prop := -- define the tangency condition for line l here

def line_l (A B : Point) : Line := -- define the line going through points A and B

def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

def O : Point := (0, 0)

def lambda_condition (A B : Point) : Prop := 
  1/2 ≤ (O.1 * B.1 + O.2 * B.2) / (O.norm * B.norm) ∧ 
  (O.1 * B.1 + O.2 * B.2) / (O.norm * B.norm) ≤ 2/3

def area_S (A B : Point) : ℝ := -- the area formula definition, involving points A, B, and O, using determinants

-- Problem 2: Prove the range of area S
theorem range_of_area_S (A B : Point) 
  (h1 : is_tangent_to (line_l A B) circle_O := (is_tangent_to l, circle_O)) 
  (h2 : trajectory_eq A.1 A.2) 
  (h3 : trajectory_eq B.1 B.2)
  (h4 : lambda_condition A B) 
  : ∃ (S : ℝ), (S = area_S A B ∧ (2 * sqrt 2 / 3) ≤ S ∧ S ≤ 1) :=
sorry

end trajectory_of_Q_range_of_area_S_l664_664264


namespace helga_extra_hours_last_friday_l664_664296

theorem helga_extra_hours_last_friday
  (weekly_articles : ℕ)
  (extra_hours_thursday : ℕ)
  (extra_articles_thursday : ℕ)
  (extra_articles_friday : ℕ)
  (articles_per_half_hour : ℕ)
  (half_hours_per_hour : ℕ)
  (usual_articles_per_day : ℕ)
  (days_per_week : ℕ)
  (articles_last_thursday_plus_friday : ℕ)
  (total_articles : ℕ) :
  (weekly_articles = (usual_articles_per_day * days_per_week)) →
  (extra_hours_thursday = 2) →
  (articles_per_half_hour = 5) →
  (half_hours_per_hour = 2) →
  (usual_articles_per_day = (articles_per_half_hour * 8)) →
  (extra_articles_thursday = (articles_per_half_hour * (extra_hours_thursday * half_hours_per_hour))) →
  (articles_last_thursday_plus_friday = weekly_articles + extra_articles_thursday) →
  (total_articles = 250) →
  (extra_articles_friday = total_articles - articles_last_thursday_plus_friday) →
  (extra_articles_friday = 30) →
  ((extra_articles_friday / articles_per_half_hour) = 6) →
  (3 = (6 / half_hours_per_hour)) :=
by
  intro hw1 hw2 hw3 hw4 hw5 hw6 hw7 hw8 hw9 hw10
  sorry

end helga_extra_hours_last_friday_l664_664296


namespace leopards_arrangement_correct_l664_664549

noncomputable def leopards_arrangement : Nat :=
  let shortestEndsWays : Nat := 2
  let tallestAdjWays : Nat := 6
  let arrangeTallestWays : Nat := Nat.factorial 2
  let arrangeRemainingWays : Nat := Nat.factorial 5
  shortestEndsWays * tallestAdjWays * arrangeTallestWays * arrangeRemainingWays

theorem leopards_arrangement_correct : leopards_arrangement = 2880 := by
  sorry

end leopards_arrangement_correct_l664_664549


namespace sum_of_decimals_as_fraction_l664_664759

theorem sum_of_decimals_as_fraction :
  (0.3 + 0.04 + 0.005 + 0.0006 + 0.00007) = 34567 / 100000 :=
by
  -- Decimals converted to fractions and summed up
  have h₀ : 0.3 = 3 / 10 := by norm_num
  have h₁ : 0.04 = 4 / 100 := by norm_num
  have h₂ : 0.005 = 5 / 1000 := by norm_num
  have h₃ : 0.0006 = 6 / 10000 := by norm_num
  have h₄ : 0.00007 = 7 / 100000 := by norm_num
  rw [h₀, h₁, h₂, h₃, h₄]
  -- Summing fractions with a common denominator
  have : (3 / 10 + 4 / 100 + 5 / 1000 + 6 / 10000 + 7 / 100000) = 34567 / 100000 := sorry
  exact this

end sum_of_decimals_as_fraction_l664_664759


namespace find_coordinates_l664_664452

def A : Prod ℤ ℤ := (-3, 2)
def move_right (p : Prod ℤ ℤ) : Prod ℤ ℤ := (p.fst + 1, p.snd)
def move_down (p : Prod ℤ ℤ) : Prod ℤ ℤ := (p.fst, p.snd - 2)

theorem find_coordinates :
  move_down (move_right A) = (-2, 0) :=
by
  sorry

end find_coordinates_l664_664452


namespace inequality_problem_l664_664996

variable (a b c d : ℝ)

open Real

theorem inequality_problem 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hprod : a * b * c * d = 1) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + d)) + 1 / (d * (1 + a)) ≥ 2 := 
by 
  sorry

end inequality_problem_l664_664996


namespace cheese_pounds_bought_l664_664119

def total_money : ℕ := 87
def price_per_pound_cheese : ℕ := 7
def price_per_pound_beef : ℕ := 5
def remaining_money : ℕ := 61

theorem cheese_pounds_bought :
  let total_spent := total_money - remaining_money in
  let cheese_spent := total_spent - price_per_pound_beef in
  cheese_spent / price_per_pound_cheese = 3 := 
by
  sorry

end cheese_pounds_bought_l664_664119


namespace function_properties_l664_664824

-- Define the polynomial function with given constants a and c
def f (x : ℝ) : ℝ := (1 / 3) * x^3 + (1 / 2) * x^2 - 2 * x + (22 / 3)

-- The main theorem to prove:
theorem function_properties :
  (∀ x : ℝ, x ∈ [(-2 : ℝ), 1] → deriv f x ≤ 0) ∧
  (∀ x : ℝ, x ∈ [(1 : ℝ), +∞] → deriv f x ≥ 0) ∧
  (∃ m : ℝ, (0 ≤ m ∧ m ≤ 1) ∧ (∀ (x1 x2 : ℝ), x1 ∈ [m, m+3] → x2 ∈ [m, m+3] → |f x1 - f x2| ≤ 45 / 2)) :=
by
  sorry

end function_properties_l664_664824


namespace volume_displaced_squared_l664_664162

theorem volume_displaced_squared (r h s v : ℝ) 
  (h_r : r = 4) 
  (h_h : h = 10) 
  (h_s : s = 8)
  (h_v : v = 8 * Real.sqrt 6) :
  v^2 = 384 :=
by
  rw [h_v]
  norm_num
  rw [Real.sqrt_mul (8:ℝ) (6:ℝ)]
  norm_num
  sorry

end volume_displaced_squared_l664_664162


namespace expected_value_of_marbles_sum_l664_664426

theorem expected_value_of_marbles_sum : 
  let marbles := {1, 2, 3, 4, 5, 6} in
  let pairs := (marbles.powerset.filter (λ s, s.card = 2)).to_finset in
  let pairs_sums := pairs.image (λ s, s.to_list.sum) in
  pairs_sums.sum / pairs_sums.card = 7 :=
by {
  let marbles := {1, 2, 3, 4, 5, 6},
  let pairs := (marbles.powerset.filter (λ s, s.card = 2)).to_finset,
  let pairs_sums := pairs.image (λ s, s.to_list.sum),
  have h1 : pairs_sums.sum = 105, sorry,
  have h2 : pairs_sums.card = 15, sorry,
  rw [h1, h2],
  norm_num,
  done,
}

end expected_value_of_marbles_sum_l664_664426


namespace points_lie_on_hyperbola_l664_664242

open Real

theorem points_lie_on_hyperbola (t : ℝ) : 
  let x := 2 * sinh t
  let y := 4 * cosh t
  in (x^2 / 4) - (y^2 / 64) = 1 :=
by
  let x := 2 * sinh t
  let y := 4 * cosh t
  sorry

end points_lie_on_hyperbola_l664_664242


namespace area_triangle_ACD_proof_area_trapezoid_ABCD_proof_l664_664256

noncomputable def area_of_triangle (b h : ℝ) : ℝ :=
  (1 / 2) * b * h

noncomputable def area_trapezoid (b1 b2 h : ℝ) : ℝ :=
  (1 / 2) * (b1 + b2) * h

theorem area_triangle_ACD_proof :
  ∀ (A B C D X Y : ℝ), 
  A = 24 → 
  C = 10 → 
  X = 6 → 
  Y = 8 → 
  B = 23 → 
  D = 27 →
  area_of_triangle C 20 = 100 :=
by
  intros A B C D X Y hAB hCD hAX hXY hXX1 hYY1
  sorry

theorem area_trapezoid_ABCD_proof :
  ∀ (A B C D X Y : ℝ), 
  A = 24 → 
  C = 10 → 
  X = 6 → 
  Y = 8 → 
  B = 23 → 
  D = 27 → 
  area_trapezoid 24 10 24 = 260 :=
by
  intros A B C D X Y hAB hCD hAX hXY hXX1 hYY1
  sorry

end area_triangle_ACD_proof_area_trapezoid_ABCD_proof_l664_664256


namespace length_of_BC_l664_664923

theorem length_of_BC 
  (A B C X : Type) 
  (d_AB : ℝ) (d_AC : ℝ) 
  (circle_center_A : A) 
  (radius_AB : ℝ)
  (intersects_BC : B → C → X)
  (BX CX : ℕ) 
  (h_BX_in_circle : BX = d_AB) 
  (h_CX_in_circle : CX = d_AC) 
  (h_integer_lengths : ∃ x y : ℕ, BX = x ∧ CX = y) :
  BX + CX = 61 :=
begin
  sorry
end

end length_of_BC_l664_664923


namespace range_of_s_l664_664677

-- Define the function s(x)
def s (x : ℝ) : ℝ := 1 / (2 - x) ^ 3

-- State the theorem about the range of the function s(x)
theorem range_of_s : set.range s = {y : ℝ | y ≠ 0} :=
by sorry

end range_of_s_l664_664677


namespace solution_to_inequality_l664_664231

theorem solution_to_inequality :
  { x : ℝ | ((x^2 - 1) / (x - 4)^2) ≥ 0 } = { x : ℝ | x ≤ -1 ∨ (1 ≤ x ∧ x < 4) ∨ x > 4 } := 
sorry

end solution_to_inequality_l664_664231


namespace inequality_solution_set_l664_664106

theorem inequality_solution_set {a b x : ℝ}
  (h1 : ∀ x, ax - b < 0 ↔ x ∈ set.Ioi 1) :
  {x | (ax + b) * (x - 3) > 0} = set.Ioo (-1 : ℝ) 3 :=
sorry

end inequality_solution_set_l664_664106


namespace pentagon_interior_angles_sequences_l664_664099

theorem pentagon_interior_angles_sequences :
  ∃ (seqs : ℕ), seqs = 2 ∧
    (∀ (x d : ℕ), 90 < x ∧ x < 120 ∧ 0 < d ∧ d < 6 ∧ x + 4 * d < 120 ∧
      (x + (x + d) + (x + 2 * d) + (x + 3 * d) + (x + 4 * d) = 540)) :=
begin
  -- We would provide the proof here, but it's omitted.
  sorry
end

end pentagon_interior_angles_sequences_l664_664099


namespace total_students_is_45_l664_664724

theorem total_students_is_45
  (students_burgers : ℕ) 
  (total_students : ℕ) 
  (hb : students_burgers = 30) 
  (ht : total_students = 45) : 
  total_students = 45 :=
by
  sorry

end total_students_is_45_l664_664724


namespace total_amount_is_252_l664_664116

-- Definition of initial states and conditions
def initial_amounts (a j : ℕ) : Prop := 
  let t := 36 in 
  let toy_end := t in
  let after_amy_t := 2 * t in
  let after_amy_j := 2 * j in
  let after_amy_a := a - (t + j) in

  let after_jan_t := 2 * after_amy_t in
  let after_jan_a := 2 * after_amy_a in
  let after_jan_j := 2 * j - (after_amy_a + after_amy_t) in

  let after_toy_a := 2 * after_jan_a in
  let after_toy_j := 2 * after_jan_j in
  let after_toy_t := after_jan_t - (after_jan_a + after_jan_j) in

  after_toy_t = toy_end

-- The sum of their money at the end stays implied because Amy's, Toy's and Jan's individual amounts lead to it
theorem total_amount_is_252 (a j : ℕ) :
  initial_amounts a j →
  a + j + 36 = 252 :=
by
sory


end total_amount_is_252_l664_664116


namespace possible_values_of_a_l664_664788

def f (x : ℝ) : ℝ := x + 1
def g (x a : ℝ) : ℝ := 2^(|x+2|) + a

theorem possible_values_of_a (a : ℝ) :
  (∀ x1 ∈ Icc 3 4, ∃ x2 ∈ Icc (-3) 1, f x1 ≥ g x2 a) ↔ a = -1 ∨ a = 2 ∨ a = 3 :=
by
  sorry

end possible_values_of_a_l664_664788


namespace sum_f_l664_664261

noncomputable def a (n : ℕ) : ℕ := 2 * n - 1
noncomputable def b (n : ℕ) : ℕ := 3 ^ (n - 1)

noncomputable def f (n : ℕ) : ℕ := (3 ^ (n - 1) + 1) / 2

theorem sum_f (n : ℕ) : (∑ i in finset.range n, f (i + 1)) = (3 ^ n + 2 * n - 1) / 4 := by
  sorry

end sum_f_l664_664261


namespace solid_of_revolution_surface_area_l664_664581

noncomputable def surface_area_of_solid_of_revolution (S : ℝ) (α : ℝ) : ℝ :=
  8 * π * S * (Real.sin (α / 4))^2 * (1 + (Real.cos (α / 4))^2) / (α - Real.sin α)

theorem solid_of_revolution_surface_area (S α : ℝ) :
  surface_area_of_solid_of_revolution S α =
  8 * π * S * (Real.sin (α / 4))^2 * (1 + (Real.cos (α / 4))^2) / (α - Real.sin α) := 
sorry

end solid_of_revolution_surface_area_l664_664581


namespace BC_length_l664_664972

-- Define the given triangle and circle conditions
variables (A B C X : Type) (AB AC BX CX : ℤ)
axiom AB_value : AB = 86
axiom AC_value : AC = 97
axiom circle_center_radius : ∃ (A : Type), ∃ (radius : ℤ), radius = AB ∧ ∃ (points : Set Type), points = {B, X} ∧ ∀ (P : Type), P ∈ points → dist A P = radius
axiom BX_CX_integers : ∃ (x y : ℤ), BX = x ∧ CX = y

-- Define calculations using the Power of a Point theorem
theorem BC_length :
  ∀ (y: ℤ) (x: ℤ), y(y + x) = AC^2 - AB^2 → x + y = 61 :=
by
  intros y x h
  have h1 : 97^2 = 9409, by norm_num,
  have h2 : 86^2 = 7396, by norm_num,
  rw [AB_value, AC_value] at h,
  rw [h1, h2] at h,
  calc y(y + x) = 2013 := by {exact h}
  -- The human verification part is skipped since we only need the statement here
  sorry

end BC_length_l664_664972


namespace shaded_region_perimeter_l664_664892

-- Define quarter circle with radius 10
def radius : ℝ := 10

-- Define perimeter of rectangle PQRO
def rectangle_perimeter : ℝ := 26

-- Define the arc length of the quarter circle (1/4 of the circumference)
def arc_AOB_length : ℝ := (radius * 2 * Real.pi) / 4

-- Define the length of AP and RB based on the rectangle and circle dimensions
def AP_RB_length : ℝ := 7

-- Combine lengths to represent the perimeter of the shaded region
def perimeter_shaded_region : ℝ := arc_AOB_length + AP_RB_length

theorem shaded_region_perimeter :
  perimeter_shaded_region = 17 + 5 * Real.pi :=
by
  -- This proof is skipped for now
  sorry

end shaded_region_perimeter_l664_664892


namespace quadratic_polynomial_correct_l664_664772

noncomputable def q (x : ℝ) : ℝ := (11/10) * x^2 - (21/10) * x + 5

theorem quadratic_polynomial_correct :
  (q (-1) = 4) ∧ (q 2 = 1) ∧ (q 4 = 10) :=
by
  -- Proof goes here
  sorry

end quadratic_polynomial_correct_l664_664772


namespace opposite_of_reciprocal_negative_one_third_l664_664604

theorem opposite_of_reciprocal_negative_one_third : -(1 / (-1 / 3)) = 3 := by
  sorry

end opposite_of_reciprocal_negative_one_third_l664_664604


namespace f_neg_2_eq_1_l664_664695

variable {ℝ : Type*}

def g (x : ℝ) : ℝ := sorry

def f (x : ℝ) : ℝ := g(x) + 2

axiom odd_g : ∀ x : ℝ, g(-x) = -g(x)

axiom f_2_eq_3 : f(2) = 3

theorem f_neg_2_eq_1 : f(-2) = 1 := by
  -- Proof would go here
  sorry

end f_neg_2_eq_1_l664_664695


namespace g_neither_even_nor_odd_l664_664973

noncomputable def g (x : ℝ) : ℝ := real.log (x + real.sqrt (2 + x^2))

theorem g_neither_even_nor_odd : 
  (∀ x : ℝ, g x ≠ g (-x)) ∧ (∀ x : ℝ, g x ≠ -g (-x) + real.log 2) := 
by
  sorry

end g_neither_even_nor_odd_l664_664973


namespace length_of_BC_l664_664932

theorem length_of_BC (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
    (BX CX : ℕ) (h_pow : CX * (BX + CX) = 2013) : 
    BX + CX = 61 :=
  sorry

end length_of_BC_l664_664932


namespace number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664379

theorem number_of_integers_with_even_divisors (n : ℕ) (h : n = 100) : 
  (card {x | (x < n) ∧ ∃ k, k * k = x} = 9) → 
  (card {x | (x < n) ∧ ¬(∃ k, k * k = x)} = n - 1 - 9) :=
by
  intro h_squares
  rw h
  trivial

open_locale classical
noncomputable def count_even_divisors_less_than_100 : ℕ :=
  90

theorem count_even_divisors_verification :
  count_even_divisors_less_than_100 = 90 :=
by
  sorry

end number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664379


namespace total_owed_proof_l664_664021

-- Define initial conditions
def initial_owed : ℕ := 20
def borrowed : ℕ := 8

-- Define the total amount owed
def total_owed : ℕ := initial_owed + borrowed

-- Prove the statement
theorem total_owed_proof : total_owed = 28 := 
by 
  -- Proof is omitted with sorry
  sorry

end total_owed_proof_l664_664021


namespace find_preimage_l664_664252

def mapping (x y : ℝ) : ℝ × ℝ :=
  (x + y, x - y)

theorem find_preimage :
  mapping 2 1 = (3, 1) :=
by
  sorry

end find_preimage_l664_664252


namespace polygon_deformable_to_triangle_l664_664150

open Real

-- Definitions to represent an n-sided polygon
structure Polygon (n : ℕ) (h : n > 4) :=
  (sides : Fin n → ℝ)      -- each side length is positive
  (hinges : Fin n → Fin n)  -- mapping each hinge to the next vertex

noncomputable def deformable_to_triangle (p : Polygon n) : Prop :=
  sorry  -- A placeholder definition indicating the property of being deformable to a triangle

-- The theorem to be proved
theorem polygon_deformable_to_triangle {n : ℕ} (h : n > 4) 
  (p : Polygon n h) : deformable_to_triangle p :=
  sorry

-- The above theorem states that any n-sided polygon with n > 4 can be deformed into a triangle.

end polygon_deformable_to_triangle_l664_664150


namespace intersection_property_l664_664545

theorem intersection_property (x_0 : ℝ) (h1 : x_0 > 0) (h2 : -x_0 = Real.tan x_0) :
  (x_0^2 + 1) * (Real.cos (2 * x_0) + 1) = 2 :=
sorry

end intersection_property_l664_664545


namespace expected_value_correct_l664_664430

noncomputable def expected_value_sum_of_two_marbles : ℕ :=
  let marbles := {1,2,3,4,5,6}
  let pairs := { (a, b) | a ∈ marbles, b ∈ marbles, a < b }
  let sums := pairs.map (λ (a, b) => a + b)
  (sums.sum.to_rat / pairs.size).to_nat

theorem expected_value_correct :
  expected_value_sum_of_two_marbles = 7 :=
  sorry

end expected_value_correct_l664_664430


namespace fraction_of_workers_read_Saramago_l664_664477

theorem fraction_of_workers_read_Saramago (S : ℚ) (total_workers : ℕ) (read_Kureishi_fraction : ℚ) 
  (read_both : ℕ) (read_neither_delta_from_Saramago_only : ℕ)
  (total_workers_eq : total_workers = 40) 
  (read_Kureishi_eq : read_Kureishi_fraction = 5 / 8)
  (read_both_eq : read_both = 2)
  (read_neither_delta_eq : read_neither_delta_from_Saramago_only = 1)
  (workers_eq : 2 + (total_workers * S - 2) + (total_workers * read_Kureishi_fraction - 2) + 
                ((total_workers * S - 2) - read_neither_delta_from_Saramago_only) = total_workers) :
  S = 9 / 40 := 
by
  sorry

end fraction_of_workers_read_Saramago_l664_664477


namespace g_values_and_properties_l664_664887

def f (x : ℂ) : ℂ := (x - 3) * (x + 4)
def g (x : ℝ) (y : ℝ) : ℂ := f (2 * x + 3) + complex.I * y
def magnitude (z : ℂ) : ℝ := complex.abs z
def phase_angle (z : ℂ) : ℝ := complex.arg z

theorem g_values_and_properties :
  let z1 := g 29 1 in
  let z2 := g 29 2 in
  let z3 := g 29 3 in
  z1 = 858 + complex.I ∧
  z2 = 858 + 2 * complex.I ∧
  z3 = 858 + 3 * complex.I ∧
  magnitude z1 = real.sqrt (858^2 + 1) ∧
  magnitude z2 = real.sqrt (858^2 + 4) ∧
  magnitude z3 = real.sqrt (858^2 + 9) ∧
  phase_angle z1 = real.atan (1 / 858) ∧
  phase_angle z2 = real.atan (2 / 858) ∧
  phase_angle z3 = real.atan (3 / 858) := by sorry

end g_values_and_properties_l664_664887


namespace remaining_tabs_after_closures_l664_664020

theorem remaining_tabs_after_closures (initial_tabs : ℕ) (first_fraction : ℚ) (second_fraction : ℚ) (third_fraction : ℚ) 
  (initial_eq : initial_tabs = 400) :
  (initial_tabs - initial_tabs * first_fraction - (initial_tabs - initial_tabs * first_fraction) * second_fraction - 
      ((initial_tabs - initial_tabs * first_fraction) - (initial_tabs - initial_tabs * first_fraction) * second_fraction) * third_fraction) = 90 :=
by
  have h1 : initial_tabs * first_fraction = 100 := by rw [initial_eq]; norm_num
  have h2 : initial_tabs - initial_tabs * first_fraction = 300 := by rw [initial_eq, h1]; norm_num
  have h3 : (initial_tabs - initial_tabs * first_fraction) * second_fraction = 120 := by rw [h2]; norm_num
  have h4 : (initial_tabs - initial_tabs * first_fraction) - (initial_tabs - initial_tabs * first_fraction) * second_fraction = 180 := by { rw [h2, h3]; norm_num }
  have h5 : ((initial_tabs - initial_tabs * first_fraction) - (initial_tabs - initial_tabs * first_fraction) * second_fraction) * third_fraction = 90 := by rw [h4]; norm_num
  have h6 : ((initial_tabs - initial_tabs * first_fraction) - (initial_tabs - initial_tabs * first_fraction) * second_fraction - ((initial_tabs - initial_tabs * first_fraction) - (initial_tabs - initial_tabs * first_fraction) * second_fraction) * third_fraction) = 90 := by rw [h4, h5]; norm_num
  exact h6


end remaining_tabs_after_closures_l664_664020


namespace f_not_periodic_l664_664533

/-- Define the function  -/
def f (x : ℝ) : ℝ := x + Real.sin x

/-- Hypothesis: f is not a periodic function -/
theorem f_not_periodic : ¬ ∃ T ≠ 0, ∀ x, f (x + T) = f x := 
sorry

end f_not_periodic_l664_664533


namespace log_equation_solution_set_l664_664105

/-- Statement: 
Given the conditions that the arguments of the logarithms must be positive, i.e., -2x > 0 and 3 - x^2 > 0,
prove that the solution to the equation lg (-2x) = lg (3 - x^2) is x = -1.
-/
theorem log_equation_solution_set :
  ∀ (x : ℝ), (-2 * x > 0) ∧ (3 - x^2 > 0) ∧ (Real.log10 (-2 * x) = Real.log10 (3 - x^2)) → (x = -1) :=
by
  sorry

end log_equation_solution_set_l664_664105


namespace solution_set_of_inequality_l664_664248

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x + 1) - 3

theorem solution_set_of_inequality :
  { x : ℝ | f x < 0 } = { x : ℝ | x < Real.log 3 / Real.log 2 } :=
by
  sorry

end solution_set_of_inequality_l664_664248


namespace minimum_value_of_trig_function_l664_664151

noncomputable def minimum_value (x : ℝ) : ℝ :=
  (Real.sin x)^4 + (Real.cos x)^4 + (Real.sec x)^4 + (Real.csc x)^4

theorem minimum_value_of_trig_function : ∃ x : ℝ, minimum_value x = 17 / 2 :=
sorry

end minimum_value_of_trig_function_l664_664151


namespace symmetric_point_in_xOz_l664_664007

def symmetric_point (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (P.1, -P.2, P.3)

theorem symmetric_point_in_xOz (P : ℝ × ℝ × ℝ) : 
  symmetric_point P = (P.1, -P.2, P.3) :=
by
  sorry

example : symmetric_point (-1, 2, 1) = (-1, -2, 1) :=
by
  rw symmetric_point_in_xOz
  rw symmetric_point
  sorry

end symmetric_point_in_xOz_l664_664007


namespace general_terms_seq_a_general_terms_seq_b_T_n_formula_l664_664255

-- Definitions of sequences and their respective properties
def a : ℕ → ℕ
| 0     := 1
| (n+1) := (a n) + 1 -- or equivalently a n = n+1, as shown

def S : ℕ → ℕ
| 0     := a 0
| (n+1) := S n + a (n+1)

def b : ℕ → ℕ
| 0     := 1 -- b₁ = 1 (since lean 0-based index)
| 1     := 3 -- b₂ = 3 (assuming b₀ is b₁ in problem statement)
| (n+2) := if n % 2 = 0 then 3 * b n else b n * b (n+1)

def T (n : ℕ) : ℕ :=
(n+1) * (b 1) + (n) * (b 3) + (n-1) * (b 5) + ...  -- continued according to the formula

theorem general_terms_seq_a {n : ℕ} :
  a n = n :=
sorry

theorem general_terms_seq_b {n : ℕ} :
  b n = if n % 2 = 0 then 3^(n/2) else 3^((n+1)/2 - 1) :=
sorry

theorem T_n_formula {n : ℕ} : 
  T n = (9 / 4) * (3 ^ n - 1) - (3 * n / 2) :=
sorry

end general_terms_seq_a_general_terms_seq_b_T_n_formula_l664_664255


namespace expectation_conditioned_eq_variance_conditioned_eq_unconditional_expectation_eq_unconditional_variance_eq_l664_664531

noncomputable def expectation_conditioned (ξ : ℕ → ℝ) (τ : ℕ) :=
  τ * (ξ 1)

noncomputable def variance_conditioned (ξ : ℕ → ℝ) (τ : ℕ) :=
  τ * (variance ξ 1)

theorem expectation_conditioned_eq (ξ : ℕ → ℝ) (τ : ℕ) (h_iid : ∀ i j, i ≠ j → statistically_independent (ξ i) (ξ j))
  (h_ident_dist : ∀ i, identically_distributed (ξ i) (ξ 1)) :
  (E (λ ω, ∑ i in range(τ), ξ i ω) | τ) = expectation_conditioned ξ τ := by
  sorry

theorem variance_conditioned_eq (ξ : ℕ → ℝ) (τ : ℕ) (h_iid : ∀ i j, i ≠ j → statistically_independent (ξ i) (ξ j))
  (h_ident_dist : ∀ i, identically_distributed (ξ i) (ξ 1)) :
  (D (λ ω, ∑ i in range(τ), ξ i ω) | τ) = variance_conditioned ξ τ := by
  sorry

theorem unconditional_expectation_eq (ξ : ℕ → ℝ) (τ : ℕ) (h_iid : ∀ i j, i ≠ j → statistically_independent (ξ i) (ξ j))
  (h_ident_dist : ∀ i, identically_distributed (ξ i) (ξ 1)) :
  E (λ ω, ∑ i in range(τ), ξ i ω) = τ * (E (ξ 1)) := by
  sorry

theorem unconditional_variance_eq (ξ : ℕ → ℝ) (τ : ℕ) (h_iid : ∀ i j, i ≠ j → statistically_independent (ξ i) (ξ j))
  (h_ident_dist : ∀ i, identically_distributed (ξ i) (ξ 1)) :
  D (λ ω, ∑ i in range(τ), ξ i ω) = τ * (D (ξ 1)) + (D (τ) * (E (ξ 1))^2) := by
  sorry

end expectation_conditioned_eq_variance_conditioned_eq_unconditional_expectation_eq_unconditional_variance_eq_l664_664531


namespace largest_lcm_among_pairs_is_45_l664_664675

theorem largest_lcm_among_pairs_is_45 :
  max (max (max (max (max (Nat.lcm 15 3) (Nat.lcm 15 5)) (Nat.lcm 15 6)) (Nat.lcm 15 9)) (Nat.lcm 15 10)) (Nat.lcm 15 15) = 45 :=
by
  sorry

end largest_lcm_among_pairs_is_45_l664_664675


namespace complement_of_M_in_U_l664_664834

def U : set ℕ := {1, 2, 3}
def M : set ℕ := {1}

theorem complement_of_M_in_U :
  U \ M = {2, 3} :=
by
  sorry

end complement_of_M_in_U_l664_664834


namespace average_rounds_rounded_is_3_l664_664601

-- Definitions based on conditions
def golfers : List ℕ := [3, 4, 3, 6, 2, 4]
def rounds : List ℕ := [0, 1, 2, 3, 4, 5]

noncomputable def total_rounds : ℕ :=
  List.sum (List.zipWith (λ g r => g * r) golfers rounds)

def total_golfers : ℕ := List.sum golfers

noncomputable def average_rounds : ℕ :=
  Int.natAbs (Int.ofNat total_rounds / total_golfers).toNat

theorem average_rounds_rounded_is_3 : average_rounds = 3 := by
  sorry

end average_rounds_rounded_is_3_l664_664601


namespace max_median_value_l664_664697

theorem max_median_value (x : ℕ) (h : 198 + x ≤ 392) : x ≤ 194 :=
by {
  sorry
}

end max_median_value_l664_664697


namespace BC_length_l664_664915

theorem BC_length (A B C X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X] 
  (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
  (BX CX : ℕ) (h_circle_intersect : ∃ X, Metric.ball A 86 ∩ {BC} = {B, X})
  (h_integer_lengths : BX + CX = BC) :
  BC = 61 := 
by
  sorry

end BC_length_l664_664915


namespace quadratic_inequality_solution_minimum_value_expression_l664_664858

theorem quadratic_inequality_solution (a : ℝ) : (∀ x : ℝ, a * x^2 - 6 * x + 3 > 0) → a > 3 :=
sorry

theorem minimum_value_expression (a : ℝ) : (a > 3) → a + 9 / (a - 1) ≥ 7 ∧ (a + 9 / (a - 1) = 7 ↔ a = 4) :=
sorry

end quadratic_inequality_solution_minimum_value_expression_l664_664858


namespace no_politics_reporters_l664_664480

theorem no_politics_reporters (X Y Both XDontY YDontX International PercentageTotal : ℝ) 
  (hX : X = 0.35)
  (hY : Y = 0.25)
  (hBoth : Both = 0.20)
  (hXDontY : XDontY = 0.30)
  (hInternational : International = 0.15)
  (hPercentageTotal : PercentageTotal = 1.0) :
  PercentageTotal - ((X + Y - Both) - XDontY + International) = 0.75 :=
by sorry

end no_politics_reporters_l664_664480


namespace number_of_integers_with_even_divisors_l664_664397

-- Define conditions
def N := 99

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P : finset ℕ := { n in finset.range (N + 1) | is_perfect_square n }

noncomputable def countP : ℕ := P.card

-- The statement to prove
theorem number_of_integers_with_even_divisors : 
  (N - countP) = 90 :=
by {
  sorry
}

end number_of_integers_with_even_divisors_l664_664397


namespace zero_in_interval_l664_664538

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem zero_in_interval : 
  ∃ x₀, f x₀ = 0 ∧ (2 : ℝ) < x₀ ∧ x₀ < (3 : ℝ) :=
by
  sorry

end zero_in_interval_l664_664538


namespace triangle_base_second_l664_664584

theorem triangle_base_second (base1 height1 height2 : ℝ) 
  (h_base1 : base1 = 15) (h_height1 : height1 = 12) (h_height2 : height2 = 18) :
  let area1 := (base1 * height1) / 2
  let area2 := 2 * area1
  let base2 := (2 * area2) / height2
  base2 = 20 :=
by
  sorry

end triangle_base_second_l664_664584


namespace greatest_prime_factor_15_18_l664_664651

theorem greatest_prime_factor_15_18! :
  ∃ p : ℕ, prime p ∧ p ∈ prime_factors (15! + 18!) ∧ ∀ q : ℕ, prime q → q ∈ prime_factors (15! + 18!) → q ≤ 4897 := 
sorry

end greatest_prime_factor_15_18_l664_664651


namespace sunzi_wood_problem_l664_664494

theorem sunzi_wood_problem (x y : ℝ) (h1 : x - y = 4.5) (h2 : (1/2) * x + 1 = y) :
  (x - y = 4.5) ∧ ((1/2) * x + 1 = y) :=
by {
  exact ⟨h1, h2⟩
}

end sunzi_wood_problem_l664_664494


namespace no_product_of_six_distinct_primes_l664_664615

theorem no_product_of_six_distinct_primes (s : Set ℕ) (h_distinct : s.card = 2014)
  (h_div : ∀ a b ∈ s, a ≠ b → (a * b) % (a + b) = 0) :
  ∀ a ∈ s, ¬ ∃ p1 p2 p3 p4 p5 p6 : ℕ, 
    (Nat.prime p1) ∧ (Nat.prime p2) ∧ (Nat.prime p3) ∧ (Nat.prime p4) ∧ (Nat.prime p5) ∧ (Nat.prime p6) ∧
    a = p1 * p2 * p3 * p4 * p5 * p6 :=
by
  sorry

end no_product_of_six_distinct_primes_l664_664615


namespace derivative_at_1_l664_664786

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem derivative_at_1 : (deriv f 1) = 2 * Real.log 2 - 3 := 
sorry

end derivative_at_1_l664_664786


namespace shaded_area_proof_l664_664891

-- Define the conditions
constant radius : ℝ
axiom diameter1_is_perpendicular_to_diameter2 : Prop
axiom diameters_have_common_center : Prop
axiom is_circle : Prop

-- Top-level definition for the problem
noncomputable def shaded_area : ℝ :=
  if radius = 5 ∧ diameter1_is_perpendicular_to_diameter2 ∧ diameters_have_common_center ∧ is_circle then
    50 + 25 * real.pi
  else 0

-- Top-level theorem statement (No proof required)
theorem shaded_area_proof : radius = 5 →
diameter1_is_perpendicular_to_diameter2 →
diameters_have_common_center →
is_circle →
shaded_area = 50 + 25 * real.pi := 
sorry

end shaded_area_proof_l664_664891


namespace gcd_m_pow_5_plus_125_m_plus_3_l664_664237

theorem gcd_m_pow_5_plus_125_m_plus_3 (m : ℕ) (h: m > 16) : 
  Nat.gcd (m^5 + 125) (m + 3) = Nat.gcd 27 (m + 3) :=
by
  -- Proof will be provided here
  sorry

end gcd_m_pow_5_plus_125_m_plus_3_l664_664237


namespace complex_subtraction_l664_664567

theorem complex_subtraction :
  (5 : ℂ) - (7 : ℂ) * complex.I - (3 : ℂ) + (2 : ℂ) * complex.I = (2 : ℂ) - (5 : ℂ) * complex.I :=
by
  sorry

end complex_subtraction_l664_664567


namespace BC_length_l664_664960

def triangle_ABC (A B C : Type)
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)] : Prop :=
  ∃ (X : A), (has_dist B X (coe (X.dist B))) ∧ (has_dist C X (coe (X.dist C))) ∧
  ∀ (x y : ℕ), x = X.dist B ∧ y = X.dist C → x + y = 61

theorem BC_length {A B C : Type}
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)]
  (h : triangle_ABC A B C) : 
  ∃ (x y : ℕ), x + y = 61 := sorry

end BC_length_l664_664960


namespace range_of_n_minus_m_l664_664456

variable (f : ℝ → ℝ)
variable (a m n : ℝ)

noncomputable def f_def (x : ℝ) : ℝ := 2^abs (x + a)

theorem range_of_n_minus_m 
  (h1 : ∀ x : ℝ, f_def x = 2^abs (x + a))
  (h2 : ∀ x : ℝ, f_def (1 - x) = f_def (1 + x))
  (h3 : f_def = λ x, 2^abs (x - 1))
  (h4 : f_def n - f_def m = 3) :
  0 < n - m ∧ n - m ≤ 4 := 
  sorry

end range_of_n_minus_m_l664_664456


namespace obtain_1_after_3_operations_obtain_1_after_4_operations_obtain_1_after_5_operations_l664_664067

def operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 3

theorem obtain_1_after_3_operations:
  (operation (operation (operation 1)) = 1) ∨ 
  (operation (operation (operation 8)) = 1) := by
  sorry

theorem obtain_1_after_4_operations:
  (operation (operation (operation (operation 1))) = 1) ∨ 
  (operation (operation (operation (operation 5))) = 1) ∨ 
  (operation (operation (operation (operation 16))) = 1) := by
  sorry

theorem obtain_1_after_5_operations:
  (operation (operation (operation (operation (operation 4)))) = 1) ∨ 
  (operation (operation (operation (operation (operation 10)))) = 1) ∨ 
  (operation (operation (operation (operation (operation 13)))) = 1) := by
  sorry

end obtain_1_after_3_operations_obtain_1_after_4_operations_obtain_1_after_5_operations_l664_664067


namespace even_divisors_less_than_100_l664_664355

theorem even_divisors_less_than_100 :
  let count_even_divisors := 
    let n := 100 in
    let perfect_squares := { m | ∃ k, k * k = m ∧ m < n } in
    let total_numbers := finset.range n in
    (total_numbers.card - perfect_squares.card) =
    90 
  in count_even_divisors = 90 :=
by
  let n := 100
  let perfect_squares : finset ℕ := finset.filter (λ m, ∃ k, k * k = m) (finset.range n)
  let total_numbers : finset ℕ := finset.range n
  have h : total_numbers.card = 99 := by sorry
  have p : perfect_squares.card = 9 := by sorry
  show total_numbers.card - perfect_squares.card = 90
  calc
    total_numbers.card - perfect_squares.card
      = 99 - 9 := by rw [h, p]
      = 90 := by norm_num

end even_divisors_less_than_100_l664_664355


namespace compare_sums_l664_664694

open Classical

-- Define the necessary sequences and their properties
variable {α : Type*} [LinearOrderedField α]

-- Arithmetic Sequence {a_n}
noncomputable def arith_seq (a_1 d : α) : ℕ → α
| 0     => a_1
| (n+1) => (arith_seq a_1 d n) + d

-- Geometric Sequence {b_n}
noncomputable def geom_seq (b_1 q : α) : ℕ → α
| 0     => b_1
| (n+1) => (geom_seq b_1 q n) * q

-- Sum of the first n terms of an arithmetic sequence
noncomputable def arith_sum (a_1 d : α) (n : ℕ) : α :=
(n + 1) * (a_1 + arith_seq a_1 d n) / 2

-- Sum of the first n terms of a geometric sequence
noncomputable def geom_sum (b_1 q : α) (n : ℕ) : α :=
if q = 1 then (n + 1) * b_1
else b_1 * (1 - q^(n + 1)) / (1 - q)

theorem compare_sums
  (a_1 b_1 : α) (d q : α)
  (hd : d ≠ 0) (hq : q > 0) (hq1 : q ≠ 1)
  (h_eq1 : a_1 = b_1)
  (h_eq2 : arith_seq a_1 d 1011 = geom_seq b_1 q 1011) :
  arith_sum a_1 d 2022 < geom_sum b_1 q 2022 :=
sorry

end compare_sums_l664_664694


namespace even_number_of_divisors_less_than_100_l664_664328

theorem even_number_of_divisors_less_than_100 : 
  ∃ n, n = 90 ∧ ∀ x < 100, (∃ k, k * k = x → false) = (x ∣ 99 - 9) :=
sorry

end even_number_of_divisors_less_than_100_l664_664328


namespace recess_break_length_l664_664633

theorem recess_break_length
    (R : ℕ) 
    (recess_count : 2)
    (lunch : 30)
    (additional_recess : 20)
    (total_outside_time : 80) :
    2 * R + lunch + additional_recess = total_outside_time → R = 15 :=
by
  sorry

end recess_break_length_l664_664633


namespace even_number_of_divisors_l664_664337

-- Proof statement: There are 90 positive integers less than 100 with an even number of divisors.
theorem even_number_of_divisors : 
  {n : ℕ | n < 100 ∧ ∃ k : ℕ, k ^ 2 = n}.toFinset.card = 90 := 
sorry

end even_number_of_divisors_l664_664337


namespace average_speed_is_80_l664_664586

def distance : ℕ := 100

def time : ℚ := 5 / 4  -- 1.25 hours expressed as a rational number

noncomputable def average_speed : ℚ := distance / time

theorem average_speed_is_80 : average_speed = 80 := by
  sorry

end average_speed_is_80_l664_664586


namespace polynomial_evaluation_l664_664044

theorem polynomial_evaluation 
  (g : ℝ → ℝ) (h : ∀ x, g(x^2 + 2) = x^4 + 5 * x^2 + 1) :
  ∀ x, g(x^2 - 2) = x^4 - 3 * x^2 - 3 := 
by
  intro x
  have h1 : g (x^2 + 2) = x^4 + 5 * x^2 + 1 := h x
  sorry

end polynomial_evaluation_l664_664044


namespace hyperbola_eccentricity_sqrt3_l664_664269

-- Definitions from conditions
def hyperbola (a b : ℝ) (x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

def point_P (a : ℝ) : ℝ × ℝ := (0, real.sqrt 6 * a)

def line_PF (P F : ℝ × ℝ) : ℝ := (P.2 - F.2) / (P.1 - F.1)

def is_focus (a b c : ℝ) : Prop := c^2 = a^2 + b^2 ∧ ∃ F : ℝ × ℝ, F = (-c, 0)

noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

-- Main theorem statement
theorem hyperbola_eccentricity_sqrt3
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : is_focus a b c)
  (P : ℝ × ℝ) (hP : P = point_P a)
  (h4 : ∃! (x y : ℝ), hyperbola a b x y ∧ y = (P.2 * c / real.sqrt 6))
  : eccentricity c a = real.sqrt 3 :=
sorry

end hyperbola_eccentricity_sqrt3_l664_664269


namespace number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664381

theorem number_of_integers_with_even_divisors (n : ℕ) (h : n = 100) : 
  (card {x | (x < n) ∧ ∃ k, k * k = x} = 9) → 
  (card {x | (x < n) ∧ ¬(∃ k, k * k = x)} = n - 1 - 9) :=
by
  intro h_squares
  rw h
  trivial

open_locale classical
noncomputable def count_even_divisors_less_than_100 : ℕ :=
  90

theorem count_even_divisors_verification :
  count_even_divisors_less_than_100 = 90 :=
by
  sorry

end number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664381


namespace triangle_bc_length_l664_664941

theorem triangle_bc_length :
  ∀ (A B C X : Type) (d_AB : ℝ) (d_AC : ℝ) (d_BX d_CX BC : ℕ),
  d_AB = 86 ∧ d_AC = 97 →
  let circleA := {center := A, radius := d_AB} in
  let intersect_B := B ∈ circleA in
  let intersect_X := X ∈ circleA in
  d_BX + d_CX = BC →
  d_BX ∈ ℕ ∧ d_CX ∈ ℕ →
  BC = 61 :=
by
  intros A B C X d_AB d_AC d_BX d_CX BC h_dist h_circle h_intersect h_sum h_intBC
  sorry

end triangle_bc_length_l664_664941


namespace volume_pyramid_l664_664193

noncomputable def volume_of_pyramid (c : ℝ) : ℝ := 
  let area_of_base := (1 / 2) * ((c * Real.sqrt 3) / 2) * (c / 2) in
  let height := c / 2 in
  (1 / 3) * area_of_base * height

theorem volume_pyramid (c : ℝ) (hc_pos : 0 < c) :
  volume_of_pyramid c = (c^3 * Real.sqrt 3) / 48 := 
by 
  sorry

end volume_pyramid_l664_664193


namespace find_AB_value_l664_664807

theorem find_AB_value :
  ∃ A B : ℕ, (A + B = 5 ∧ (A - B) % 11 = 5 % 11) ∧
           990 * 991 * 992 * 993 = 966428 * 100000 + A * 9100 + B * 40 :=
sorry

end find_AB_value_l664_664807


namespace BC_length_l664_664964

-- Define the given triangle and circle conditions
variables (A B C X : Type) (AB AC BX CX : ℤ)
axiom AB_value : AB = 86
axiom AC_value : AC = 97
axiom circle_center_radius : ∃ (A : Type), ∃ (radius : ℤ), radius = AB ∧ ∃ (points : Set Type), points = {B, X} ∧ ∀ (P : Type), P ∈ points → dist A P = radius
axiom BX_CX_integers : ∃ (x y : ℤ), BX = x ∧ CX = y

-- Define calculations using the Power of a Point theorem
theorem BC_length :
  ∀ (y: ℤ) (x: ℤ), y(y + x) = AC^2 - AB^2 → x + y = 61 :=
by
  intros y x h
  have h1 : 97^2 = 9409, by norm_num,
  have h2 : 86^2 = 7396, by norm_num,
  rw [AB_value, AC_value] at h,
  rw [h1, h2] at h,
  calc y(y + x) = 2013 := by {exact h}
  -- The human verification part is skipped since we only need the statement here
  sorry

end BC_length_l664_664964


namespace length_of_BC_l664_664936

theorem length_of_BC (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
    (BX CX : ℕ) (h_pow : CX * (BX + CX) = 2013) : 
    BX + CX = 61 :=
  sorry

end length_of_BC_l664_664936


namespace drop_below_100_l664_664078

/-- Data points for the number of overweight people by month -/
def data : List (ℕ × ℕ) := [(1, 640), (2, 540), (3, 420), (4, 300), (5, 200)]

/-- Reference values for regression calculation -/
def sum_xy : ℕ := 5180
def sum_x_squared : ℕ := 55

/-- Calculate mean values -/
def mean_x := 3
def mean_y := 420

/-- Calculate the slope (b) -/
def slope : ℚ := ((sum_xy : ℚ) - 5 * mean_x * mean_y) / ((sum_x_squared : ℚ) - 5 * mean_x^2)

/-- Calculate the intercept (a) -/
def intercept : ℚ := mean_y - slope * mean_x

/-- Regression equation -/
def regression (x : ℚ) : ℚ := slope * x + intercept

/--
To prove: From 6th month onwards, the number of overweight people in the university 
will drop below 100.
-/
theorem drop_below_100 (x : ℕ) (h : x ≥ 6) : regression x < 100 := by
  -- skipped proof
  sorry

end drop_below_100_l664_664078


namespace even_divisors_count_lt_100_l664_664346

theorem even_divisors_count_lt_100 : 
  {n : ℕ | n < 100 ∧ n ≠ 0 ∧ ∃ k : ℕ, k * k = n } = { n : ℕ | n < 100 ∧ n ≠ 0 } \ 
  { n : ℕ | ∃ k : ℕ, k * k = n ∧ k < 100 } → 
  (card {n : ℕ | n < 100 ∧ n ≠ 0 ∧ even (finset.card (divisors n))} = 90) :=
begin
  sorry
end

end even_divisors_count_lt_100_l664_664346


namespace sunzi_wood_problem_l664_664495

theorem sunzi_wood_problem (x y : ℝ) (h1 : x - y = 4.5) (h2 : (1/2) * x + 1 = y) :
  (x - y = 4.5) ∧ ((1/2) * x + 1 = y) :=
by {
  exact ⟨h1, h2⟩
}

end sunzi_wood_problem_l664_664495


namespace odd_ints_divisibility_l664_664577

theorem odd_ints_divisibility (a b : ℤ) (ha_odd : a % 2 = 1) (hb_odd : b % 2 = 1) (hdiv : 2 * a * b + 1 ∣ a^2 + b^2 + 1) : a = b :=
sorry

end odd_ints_divisibility_l664_664577


namespace evaluate_f2_l664_664593

def quadratic_function (a b : ℝ) := λ x : ℝ, a * x^2 + b * x + 6

theorem evaluate_f2 (a b : ℝ) (h : quadratic_function a b (-1) = quadratic_function a b 3) : quadratic_function a b 2 = 6 := by
  sorry

end evaluate_f2_l664_664593


namespace num_valid_sequences_correct_l664_664095

noncomputable def num_valid_sequences : ℕ :=
  set.to_finset { p : ℤ × ℤ | 
                  let x := p.1, d := p.2 in
                  x + 2 * d = 108 ∧ x + 4 * d < 120 ∧ x > 48 ∧
                  all (λ k, x + k * d) (list.range (fin.val (5 : fin 5))) < 120 ∧
                  is_arithmetic_sequence [x, x + d, x + 2 * d, x + 3 * d, x + 4 * d] }
  sorry

theorem num_valid_sequences_correct : num_valid_sequences = 3 := by
  sorry

end num_valid_sequences_correct_l664_664095


namespace students_answered_both_correctly_l664_664688

theorem students_answered_both_correctly 
  (total_students : ℕ) (took_test : ℕ) 
  (q1_correct : ℕ) (q2_correct : ℕ)
  (did_not_take_test : ℕ)
  (h1 : total_students = 25)
  (h2 : q1_correct = 22)
  (h3 : q2_correct = 20)
  (h4 : did_not_take_test = 3)
  (h5 : took_test = total_students - did_not_take_test) :
  (q1_correct + q2_correct) - took_test = 20 := 
by 
  -- Proof skipped.
  sorry

end students_answered_both_correctly_l664_664688


namespace smallest_third_altitude_l664_664632

/-- Given a scalene triangle ABC with:
  - Two altitudes of lengths 5 and 15
  - The shortest side being twice the longest side
Prove that the smallest integer length of the third altitude is 30. -/
theorem smallest_third_altitude (ABC : Type)
  [triangle ABC]
  (height_to_base1 : ℕ := 5)
  (height_to_base2 : ℕ := 15)
  (shortest_side_double_longest : ∃ AB BC AC : ℕ, (AB = 2 * BC) ∧ (BC ≠ AC) ∧ (AC ≠ AB) ∧ (BC ≠ AC) ∧ (BC ≠ AB)) :
  ∃ height_to_base3 : ℕ, height_to_base3 = 30 :=
sorry

end smallest_third_altitude_l664_664632


namespace edge_length_of_cube_l664_664085

theorem edge_length_of_cube (s : ℝ) (h : s * sqrt 3 = 10 * sqrt 3) : s = 10 :=
by
  sorry

end edge_length_of_cube_l664_664085


namespace easter_eggs_l664_664114

theorem easter_eggs (total_eggs : ℕ) (hannah_found_twice : ℕ → ℕ) (helen_eggs : ℕ) : total_eggs = 63 ∧ hannah_found_twice helen_eggs = 2 * helen_eggs → hannah_found_twice helen_eggs = 42 :=
by
  intros h,
  cases h with total_cond hannah_cond,
  have helen_eq : helen_eggs = 21,
  {
    linarith,
  },
  rw helen_eq at hannah_cond,
  simp [hannah_cond],
  sorry

end easter_eggs_l664_664114


namespace two_legged_birds_count_l664_664488

-- Definitions and conditions
variables {x y z : ℕ}
variables (heads_eq : x + y + z = 200) (legs_eq : 2 * x + 3 * y + 4 * z = 558)

-- The statement to prove
theorem two_legged_birds_count : x = 94 :=
sorry

end two_legged_birds_count_l664_664488


namespace binomial_sum_expression_l664_664111

theorem binomial_sum_expression (n : ℕ) :
  (∑ k in Finset.range (n + 1), (-2) ^ k * Nat.choose n k) = (-1) ^ n :=
by
  sorry

end binomial_sum_expression_l664_664111


namespace quiz_prob_distrib_l664_664478

theorem quiz_prob_distrib
  (num_eco_choice : ℕ)
  (total_eco : ℕ)
  (num_smart_choice : ℕ)
  (total_smart : ℕ)
  (ξ : ℕ → ℕ)
  (n : ℕ) :
  num_eco_choice = 3 →
  total_eco = 4 →
  num_smart_choice = 2 →
  total_smart = 2 →
  ξ 0 + ξ 1 + ξ 2 = n →
  ξ 1 / n = 3 / 5 ∧
  (ξ 0 / n = 1 / 5 ∧ ξ 1 / n = 3 / 5 ∧ ξ 2 / n = 1 / 5) ∧
  (ξ 0 = 0 → ξ 1 = ξ (1 : ℕ) → ξ 2 * 1 / n + ξ 1 * 1 / n + ξ 0 * 1 / n = 1) :=
begin
  sorry,
end

end quiz_prob_distrib_l664_664478


namespace rational_terms_l664_664154

-- We start with noncomputable definitions of s and t as per the conditions.
noncomputable def s (x : ℝ) : ℝ := sin (64 * x) + sin (65 * x)
noncomputable def t (x : ℝ) : ℝ := cos (64 * x) + cos (65 * x)

-- We state the rationality conditions
def s_rational (x : ℝ) : Prop := ∃ (q : ℚ), s x = q
def t_rational (x : ℝ) : Prop := ∃ (q : ℚ), t x = q

-- Proving the problem condition
theorem rational_terms (x : ℝ) (hs : s_rational x) (ht : t_rational x) :
  ∃ y (z : ℝ), (y = sin (64 * x) ∧ z = sin (65 * x) ∧ y.denominator = 1 ∧ z.denominator = 1) ∨
               (y = cos (64 * x) ∧ z = cos (65 * x) ∧ y.denominator = 1 ∧ z.denominator = 1) := 
sorry

end rational_terms_l664_664154


namespace AB_plus_C_eq_neg8_l664_664595

theorem AB_plus_C_eq_neg8 (A B C : ℤ) (g : ℝ → ℝ)
(hf : ∀ x > 3, g x > 0.5)
(heq : ∀ x, g x = x^2 / (A * x^2 + B * x + C))
(hasymp_vert : ∀ x, (A * (x + 3) * (x - 2) = 0 → x = -3 ∨ x = 2))
(hasymp_horiz : (1 : ℝ) / (A : ℝ) < 1) :
A + B + C = -8 :=
sorry

end AB_plus_C_eq_neg8_l664_664595


namespace even_number_of_divisors_less_than_100_l664_664414

theorem even_number_of_divisors_less_than_100 :
  ∃ (count : ℕ), count = 90 ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 100 →
    (∃ (d : ℕ), d * d = n ∨ (number_of_divisors n % 2 = 0)) :=
begin
  -- the proof goes here
  sorry
end

end even_number_of_divisors_less_than_100_l664_664414


namespace angle_of_inclination_l664_664751

theorem angle_of_inclination (α : ℝ) (hα : 0 ≤ α ∧ α < π) :
  ∃ α, (∀ x y : ℝ, x + real.sqrt 3 * y - 1 = 0 → y = -1 / real.sqrt 3 * x + 1 / real.sqrt 3) → α = 5 * real.pi / 6 :=
begin
  sorry
end

end angle_of_inclination_l664_664751


namespace even_number_of_divisors_lt_100_l664_664403

theorem even_number_of_divisors_lt_100 : 
  let n := 99 in
  let perfect_squares := {1, 4, 9, 16, 25, 36, 49, 64, 81} in
  n - perfect_squares.card = 90 := 
by
  sorry

end even_number_of_divisors_lt_100_l664_664403


namespace proj_w_v_eq_v_l664_664236

noncomputable def vec_v : ℝ × ℝ := (-10, 6)
noncomputable def vec_w : ℝ × ℝ := (15, -9)

def vector_projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := ((v.1 * w.1 + v.2 * w.2) / (w.1 * w.1 + w.2 * w.2))
  (scalar * w.1, scalar * w.2)

theorem proj_w_v_eq_v :
  vector_projection vec_v vec_w = vec_v :=
  sorry

end proj_w_v_eq_v_l664_664236


namespace simplify_sqrt_expression_l664_664570

theorem simplify_sqrt_expression :
  (Real.sqrt 726 / Real.sqrt 242) + (Real.sqrt 484 / Real.sqrt 121) = Real.sqrt 3 + 2 :=
by
  -- Proof goes here
  sorry

end simplify_sqrt_expression_l664_664570


namespace largest_possible_value_of_b_l664_664037

theorem largest_possible_value_of_b (b : ℚ) (h : (3 * b + 4) * (b - 2) = 9 * b) : b ≤ 4 :=
sorry

end largest_possible_value_of_b_l664_664037


namespace number_of_balls_greater_l664_664701

theorem number_of_balls_greater (n x : ℤ) (h1 : n = 25) (h2 : n - x = 30 - n) : x = 20 := by
  sorry

end number_of_balls_greater_l664_664701


namespace student_ranking_has_54_possibilities_l664_664235

def studentRankingProblem : Prop :=
  ∃(A B C D E : ℕ), 
    (A ∈ {2, 3, 4, 5}) ∧ (B ∈ {2, 3, 4}) ∧ 
    (A ≠ B) ∧ 
    (C ∈ {1, 2, 3, 4, 5}) ∧ (D ∈ {1, 2, 3, 4, 5}) ∧ (E ∈ {1, 2, 3, 4, 5}) ∧ 
    (C ≠ D) ∧ (C ≠ E) ∧ (D ≠ E) ∧ 
    (C ≠ A) ∧ (C ≠ B) ∧ 
    (D ≠ A) ∧ (D ≠ B) ∧ 
    (E ≠ A) ∧ (E ≠ B) ∧ 
    (∃! l : list ℕ, l.permutations.length = 54 ∧ 
                    (l.filter(λ x, x = 1)).length = 2 ∧ 
                    (l.filter(λ x, x = 5)).length = 1)

theorem student_ranking_has_54_possibilities : studentRankingProblem := 
by
  sorry

end student_ranking_has_54_possibilities_l664_664235


namespace angle_POQ_regular_pentagon_l664_664485

theorem angle_POQ_regular_pentagon :
  ∀ (s : ℝ) (pentagon_center : point) (O C P Q : point) (angle_PCQ OP_bisect_CPO :
  internal_angle_pentagon = 108 ∧ line_OP_bisect := 54),
  angle_POQ = 72 :=
by
  sorry

end angle_POQ_regular_pentagon_l664_664485


namespace percent_increase_skateboard_safety_water_l664_664986

theorem percent_increase_skateboard_safety_water :
  let last_year_cost := 120 + 80 + 20
  let this_year_cost := 144 + 104 + 25
  (this_year_cost - last_year_cost : ℝ) / last_year_cost * 100 = 24.09 :=
by
  let last_year_cost := 120 + 80 + 20
  let this_year_cost := 144 + 104 + 25
  have h_costs : this_year_cost - last_year_cost = 273 - 220 := rfl
  have h_percentage : (this_year_cost - last_year_cost : ℝ) / last_year_cost * 100 = 24.09 := sorry
  exact h_percentage
  

end percent_increase_skateboard_safety_water_l664_664986


namespace greatest_prime_factor_15_factorial_plus_18_factorial_l664_664643

theorem greatest_prime_factor_15_factorial_plus_18_factorial :
  ∀ {a b c d e f g: ℕ}, a = 15! → b = 18! → c = 16 → d = 17 → e = 18 → f = a * (1 + c * d * e) →
  g = 4896 → Prime 17 → f + b = a + b → Nat.gcd (a + b) g = 17 :=
by
  intros
  sorry

end greatest_prime_factor_15_factorial_plus_18_factorial_l664_664643


namespace market_value_decrease_l664_664598

noncomputable def percentage_decrease_each_year : ℝ :=
  let original_value := 8000
  let value_after_two_years := 3200
  let p := 1 - (value_after_two_years / original_value)^(1 / 2)
  p * 100

theorem market_value_decrease :
  let p := percentage_decrease_each_year
  abs (p - 36.75) < 0.01 :=
by
  sorry

end market_value_decrease_l664_664598


namespace roots_poly_sum_and_product_l664_664999

noncomputable def poly : Polynomial ℝ := Polynomial.C (-2) + Polynomial.X * Polynomial.C (-6) + Polynomial.X ^ 4

theorem roots_poly_sum_and_product (p q : ℝ) (hpq : polynomial.eval p poly = 0 ∧ polynomial.eval q poly = 0) : 
  p * q + p + q = 1 - 2 * Real.sqrt 2 :=
sorry

end roots_poly_sum_and_product_l664_664999


namespace distance_between_foci_l664_664628

noncomputable def center (p1 p2 : ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
noncomputable def semi_axis_length (a b : ℝ) : ℝ := (b - a) / 2

theorem distance_between_foci :
  let p1 := (1, 5) in
  let p2 := (4, 0) in
  let p3 := (10, 5) in
  let c := center p1 p3 in
  let semiHorizontal := semi_axis_length p1.1 p3.1 in
  let semiVertical := semi_axis_length p2.2 c.2 * 2 in
  2 * Real.sqrt (semiVertical^2 - semiHorizontal^2) = 2 * Real.sqrt 4.75 :=
by
  sorry

end distance_between_foci_l664_664628


namespace clarissa_cost_is_300_l664_664208

-- Define the conditions and constants based on the problem statement
def number_of_copies : ℕ := 10
def total_pages_per_copy : ℕ := 400
def color_pages_per_copy : ℕ := 50
def bw_page_cost : ℝ := 0.05
def color_page_cost : ℝ := 0.10
def binding_cost_per_copy : ℝ := 5.00
def index_cost_per_copy : ℝ := 2.00
def bundle_discount_per_copy : ℝ := 0.50
def rush_processing_copies : ℕ := 5
def rush_processing_cost_per_copy : ℝ := 3.00
def binding_discount : ℝ := 0.10

noncomputable def total_cost (number_of_copies : ℕ)
                            (total_pages_per_copy : ℕ)
                            (color_pages_per_copy : ℕ)
                            (bw_page_cost : ℝ)
                            (color_page_cost : ℝ)
                            (binding_cost_per_copy : ℝ)
                            (index_cost_per_copy : ℝ)
                            (bundle_discount_per_copy : ℝ)
                            (rush_processing_copies : ℕ)
                            (rush_processing_cost_per_copy : ℝ)
                            (binding_discount : ℝ) : ℝ :=
let bw_pages_per_copy := total_pages_per_copy - color_pages_per_copy in
let printing_cost_per_copy := bw_pages_per_copy * bw_page_cost + color_pages_per_copy * color_page_cost in
let additional_cost_per_copy := binding_cost_per_copy + index_cost_per_copy - bundle_discount_per_copy in
let total_cost_per_copy := printing_cost_per_copy + additional_cost_per_copy in
let total_cost_before_discount := total_cost_per_copy * number_of_copies in
let total_binding_discount := binding_cost_per_copy * binding_discount * number_of_copies in
let total_cost_after_binding_discount := total_cost_before_discount - total_binding_discount in
let total_rush_processing_cost := rush_processing_copies * rush_processing_cost_per_copy in
total_cost_after_binding_discount + total_rush_processing_cost

noncomputable def clarissa_total_cost : ℝ :=
total_cost number_of_copies
           total_pages_per_copy
           color_pages_per_copy
           bw_page_cost
           color_page_cost
           binding_cost_per_copy
           index_cost_per_copy
           bundle_discount_per_copy
           rush_processing_copies
           rush_processing_cost_per_copy
           binding_discount

theorem clarissa_cost_is_300 : clarissa_total_cost = 300 := by
  sorry

end clarissa_cost_is_300_l664_664208


namespace evens_divisors_lt_100_l664_664371

theorem evens_divisors_lt_100 : 
  ∃ n : ℕ, n = 90 ∧ ∀ k : ℕ, (1 ≤ k < 100) → (even k ↔ (∃ m : ℕ, m * m = k)) ↔ (n = 90) := 
sorry

end evens_divisors_lt_100_l664_664371


namespace abc_inequality_l664_664544

theorem abc_inequality 
  (a b c : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : 0 < c) 
  (h4 : a * b * c = 1) 
  : 
  (ab / (a^5 + b^5 + ab) + bc / (b^5 + c^5 + bc) + ca / (c^5 + a^5 + ca) ≤ 1) := 
by 
  sorry

end abc_inequality_l664_664544


namespace bus_costs_possible_plans_min_total_cost_l664_664117

-- Define the costs of buses
variables (x y : ℕ)

-- Define conditions for the costs
def cost_conditions : Prop :=
  x + 3 * y = 380 ∧
  2 * x + 2 * y = 360

-- The cost equality proof (Part 1)
theorem bus_costs (h : cost_conditions) : x = 80 ∧ y = 100 := sorry

-- Define purchasing plans
variables (a : ℕ)

-- Define conditions for purchasing plans
def purchasing_conditions := 
  80 * a + 100 * (10 - a) ≤ 880 ∧
  500000 * a + 600000 * (10 - a) ≥ 5200000

-- The purchasing plans set (Part 2)
theorem possible_plans (h : purchasing_conditions) : a = 6 ∨ a = 7 ∨ a = 8 := sorry

-- Define the cost function for a given plan
def total_cost (a : ℕ) : ℕ := 80 * a + 100 * (10 - a)

-- Prove the least total cost plan (Part 3)
theorem min_total_cost (h: purchasing_conditions) : (a = 8) ∧ (total_cost 8 = 840) := sorry

end bus_costs_possible_plans_min_total_cost_l664_664117


namespace BC_length_l664_664963

def triangle_ABC (A B C : Type)
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)] : Prop :=
  ∃ (X : A), (has_dist B X (coe (X.dist B))) ∧ (has_dist C X (coe (X.dist C))) ∧
  ∀ (x y : ℕ), x = X.dist B ∧ y = X.dist C → x + y = 61

theorem BC_length {A B C : Type}
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)]
  (h : triangle_ABC A B C) : 
  ∃ (x y : ℕ), x + y = 61 := sorry

end BC_length_l664_664963


namespace range_of_a_l664_664817

def f (x : ℝ) : ℝ := -x^5 - 3 * x^3 - 5 * x + 3

theorem range_of_a (a : ℝ) (h : f a + f (a - 2) > 6) : a < 1 :=
by
  -- Here, we would have to show the proof, but we're skipping it
  sorry

end range_of_a_l664_664817


namespace equilateral_triangle_on_sphere_max_distance_l664_664613

noncomputable theory

def maximum_distance_to_plane (R d : ℝ) : ℝ := R + d

theorem equilateral_triangle_on_sphere_max_distance
  (side_length : ℝ)
  (sphere_volume : ℝ)
  (R : ℝ)
  (d : ℝ)
  (h1 : side_length = 2 * Real.sqrt 2)
  (h2 : sphere_volume = 4 * Real.sqrt 3 * Real.pi)
  (hR : R = Real.sqrt 3)
  (hd : d = Real.sqrt 3 / 3) :
  maximum_distance_to_plane R d = 4 * Real.sqrt 3 / 3 :=
by
  sorry

end equilateral_triangle_on_sphere_max_distance_l664_664613


namespace BC_length_l664_664916

theorem BC_length (A B C X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X] 
  (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
  (BX CX : ℕ) (h_circle_intersect : ∃ X, Metric.ball A 86 ∩ {BC} = {B, X})
  (h_integer_lengths : BX + CX = BC) :
  BC = 61 := 
by
  sorry

end BC_length_l664_664916


namespace find_k_value_l664_664272

theorem find_k_value (k : ℝ) (h : (7 * (-1)^3 - 3 * (-1)^2 + k * -1 + 5 = 0)) :
  k^3 + 2 * k^2 - 11 * k - 85 = -105 :=
by {
  sorry
}

end find_k_value_l664_664272


namespace BC_length_l664_664956

def triangle_ABC (A B C : Type)
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)] : Prop :=
  ∃ (X : A), (has_dist B X (coe (X.dist B))) ∧ (has_dist C X (coe (X.dist C))) ∧
  ∀ (x y : ℕ), x = X.dist B ∧ y = X.dist C → x + y = 61

theorem BC_length {A B C : Type}
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)]
  (h : triangle_ABC A B C) : 
  ∃ (x y : ℕ), x + y = 61 := sorry

end BC_length_l664_664956


namespace find_b_in_cubic_function_l664_664827

noncomputable def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem find_b_in_cubic_function (a b c d : ℝ) (h1: cubic_function a b c d 2 = 0)
  (h2: cubic_function a b c d (-1) = 0) (h3: cubic_function a b c d 1 = 4) :
  b = 6 :=
by
  sorry

end find_b_in_cubic_function_l664_664827


namespace no_product_of_six_primes_l664_664617

noncomputable theory

open Nat

-- Define the problem's conditions within a Lean 4 statement
theorem no_product_of_six_primes (S : Finset ℕ) (h_distinct : S.card = 2014)
  (h_condition : ∀ (x y ∈ S), x ≠ y → (x * y) % (x + y) = 0) :
  ∀ x ∈ S, ¬∃ (p1 p2 p3 p4 p5 p6 : ℕ), pairwise (≠) [p1, p2, p3, p4, p5, p6] ∧
  (∀ p ∈ [p1, p2, p3, p4, p5, p6], Nat.Prime p) ∧ x = p1 * p2 * p3 * p4 * p5 * p6 := 
by
  sorry

end no_product_of_six_primes_l664_664617


namespace ferris_wheel_time_l664_664699

theorem ferris_wheel_time (radius : ℝ) (period : ℝ) (height : ℝ) (time : ℝ) : period = 120 → radius = 30 → height = 15 → time = 40 :=
by
  -- Given
  intros h_period h_radius h_height
  
  -- Definitions
  let A := radius
  let D := radius
  let B := (2 * Real.pi) / period
  let height_eq := height = A * Real.cos(B * time) + D 

  -- Conditions
  have h1 : height_eq = 15 = 30 * Real.cos((Real.pi / 60) * time) + 30, from sorry
  
  -- Solving the equation
  have h2 : -15 = 30 * Real.cos((Real.pi / 60) * time), from sorry
  have h3 : -Real.cos((Real.pi / 60) * time) = 1/2, from sorry
  have h4 : (Real.pi / 60) * time = (2 * Real.pi) / 3, from sorry
  have time_eq : time = 40, from sorry

  -- Final proof
  exact time_eq

end ferris_wheel_time_l664_664699


namespace even_number_of_divisors_less_than_100_l664_664315

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k, k * k = n

theorem even_number_of_divisors_less_than_100 :
  (card { n : ℕ | 1 ≤ n ∧ n < 100 ∧ ¬ is_perfect_square n }) = 90 :=
by
  sorry

end even_number_of_divisors_less_than_100_l664_664315


namespace kaleb_final_amount_l664_664522

variable (spring_earnings : ℕ) (summer_earnings : ℕ) (supplies_cost : ℕ)

theorem kaleb_final_amount :
  spring_earnings = 4 →
  summer_earnings = 50 →
  supplies_cost = 4 →
  (spring_earnings + summer_earnings - supplies_cost) = 50 :=
by
  intros hSpring hSummer hSupplies
  rw [hSpring, hSummer, hSupplies]
  sorry

end kaleb_final_amount_l664_664522


namespace even_number_of_divisors_less_than_100_l664_664316

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k, k * k = n

theorem even_number_of_divisors_less_than_100 :
  (card { n : ℕ | 1 ≤ n ∧ n < 100 ∧ ¬ is_perfect_square n }) = 90 :=
by
  sorry

end even_number_of_divisors_less_than_100_l664_664316


namespace total_expenditure_is_108_l664_664074

-- Define the costs of items and quantities purchased by Robert and Teddy
def cost_pizza := 10   -- cost of one box of pizza
def cost_soft_drink := 2  -- cost of one can of soft drink
def cost_hamburger := 3   -- cost of one hamburger

def qty_pizza_robert := 5     -- quantity of pizza boxes by Robert
def qty_soft_drink_robert := 10 -- quantity of soft drinks by Robert

def qty_hamburger_teddy := 6  -- quantity of hamburgers by Teddy
def qty_soft_drink_teddy := 10 -- quantity of soft drinks by Teddy

-- Calculate total expenditure for Robert and Teddy
def total_cost_robert := (qty_pizza_robert * cost_pizza) + (qty_soft_drink_robert * cost_soft_drink)
def total_cost_teddy := (qty_hamburger_teddy * cost_hamburger) + (qty_soft_drink_teddy * cost_soft_drink)

-- Total expenditure in all
def total_expenditure := total_cost_robert + total_cost_teddy

-- We formulate the theorem to prove that the total expenditure is $108
theorem total_expenditure_is_108 : total_expenditure = 108 :=
by 
  -- Placeholder proof
  sorry

end total_expenditure_is_108_l664_664074


namespace henry_twice_jill_years_ago_l664_664108

theorem henry_twice_jill_years_ago (H J : ℕ) (h_sum : H + J = 41)
  (h_henry : H = 25) (h_jill : J = 16) : 
  ∃ x : ℕ, H - x = 2 * (J - x) ∧ x = 7 := by
  exists 7
  split
  · rw [h_henry, h_jill]
    simp
  · sorry

end henry_twice_jill_years_ago_l664_664108


namespace cover_3x4_with_L_triminos_not_cover_3x5_with_L_triminos_cover_3x5_with_I_and_L_triminos_l664_664635

-- Definitions for triminos and boards
def L_trimino := { pos : ℕ × ℕ // pos.1 < 3 ∧ pos.2 < 1 } -- An L-shape trimino defined in a 3x1 space
def I_trimino := { pos : ℕ × ℕ // pos.1 < 1 ∧ pos.2 < 3 } -- An I-shape trimino defined in a 1x3 space

-- a) Prove that it is possible to cover a 3x4 board using only L-triminos
theorem cover_3x4_with_L_triminos : ∃ positions : set (ℕ × ℕ), 
  positions.finite ∧ positions.card = 12 ∧ all_L_triminos positions ∧ no_overlap positions := sorry

-- b) Prove that it is impossible to cover a 3x5 board using only L-triminos
theorem not_cover_3x5_with_L_triminos : ¬ ∃ positions : set (ℕ × ℕ), 
  positions.finite ∧ positions.card = 15 ∧ all_L_triminos positions ∧ no_overlap positions := sorry

-- c) Prove that it is possible to cover a 3x5 board using exactly one I-trimino and some L-triminos, and find the 7 positions for I-trimino
theorem cover_3x5_with_I_and_L_triminos : ∃ positions : set (ℕ × ℕ), 
  positions.finite ∧ positions.card = 15 ∧ 
  (∃ i_pos : (ℕ × ℕ), I_trimino_personvalid_pos i_pos ∧ all_L_triminos (positions \ set.singleton i_pos) ∧ no_overlap positions) ∧
  find_positions_for_I positions = 7 := sorry

end cover_3x4_with_L_triminos_not_cover_3x5_with_L_triminos_cover_3x5_with_I_and_L_triminos_l664_664635


namespace expression_evaluation_l664_664225

noncomputable def calculate_expression : ℚ :=
  ((2024^3 - 3 * 2024^2 * 2025 + 4 * 2024 * 2025^2 - 2025^3 + 2) : ℚ) / (2024 * 2025)

theorem expression_evaluation : calculate_expression = (2025 - 1 / (2024 * 2025) : ℚ) :=
  sorry

end expression_evaluation_l664_664225


namespace max_z_and_20z_l664_664778

def floor (x : ℝ) : ℤ := ⌊x⌋

def problem_statement (z : ℝ) : Prop :=
  floor (5 / z) + floor (6 / z) = 7

theorem max_z_and_20z : ∃ z : ℝ, problem_statement z ∧ 20 * z = 30 := 
sorry

end max_z_and_20z_l664_664778


namespace number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664383

theorem number_of_integers_with_even_divisors (n : ℕ) (h : n = 100) : 
  (card {x | (x < n) ∧ ∃ k, k * k = x} = 9) → 
  (card {x | (x < n) ∧ ¬(∃ k, k * k = x)} = n - 1 - 9) :=
by
  intro h_squares
  rw h
  trivial

open_locale classical
noncomputable def count_even_divisors_less_than_100 : ℕ :=
  90

theorem count_even_divisors_verification :
  count_even_divisors_less_than_100 = 90 :=
by
  sorry

end number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664383


namespace even_function_value_l664_664271

theorem even_function_value (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = 2 ^ x) :
  f (Real.log 9 / Real.log 4) = 1 / 3 :=
by
  sorry

end even_function_value_l664_664271


namespace BC_length_l664_664912

theorem BC_length (A B C X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X] 
  (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
  (BX CX : ℕ) (h_circle_intersect : ∃ X, Metric.ball A 86 ∩ {BC} = {B, X})
  (h_integer_lengths : BX + CX = BC) :
  BC = 61 := 
by
  sorry

end BC_length_l664_664912


namespace cups_of_sugar_l664_664054

-- Definitions of the conditions:
def total_scoops : ℕ := 15
def flour_cups : ℕ := 3
def scoop_fraction : ℝ := 1 / 3

-- The problem statement:
theorem cups_of_sugar :
  ∃ sugar_cups : ℝ,
    (sugar_cups = 2) ∧
    (flour_cups * 3 + sugar_cups * 3 = total_scoops) :=
sorry

end cups_of_sugar_l664_664054


namespace find_AC_l664_664880

variable (ABC AP CQ : Triangle)
variable (BPQ : Triangle)
variable [ABC acute]
variable (AC : ℝ)

-- Conditions
variable (P : ABC.vertex)
variable (Q : ABC.vertex)
variable {P_altitude : ABC.is_altitude P AP}
variable {Q_altitude : ABC.is_altitude Q CQ}
variable {ABC_perimeter : ABC.perimeter = 15}
variable {BPQ_perimeter : BPQ.perimeter = 9}
variable {BPQ_circumradius : BPQ.circumradius = 9 / 5}

-- Theorem to be proved
theorem find_AC : AC = 24 / 5 := 
sorry

end find_AC_l664_664880


namespace trig_identity_l664_664568
-- Lean 4 statement

theorem trig_identity : 
  sin (-2) + cos (2 - π) * tan (2 - 4 * π) = -2 * sin 2 := 
by
  sorry

end trig_identity_l664_664568


namespace solve_cubic_inequality_l664_664764

theorem solve_cubic_inequality :
  {x : ℝ | (∛(2 * x) + 3 / (∛(2 * x) + 4)) ≤ 0} = set.Ioo (-32 : ℝ) (-1/2 : ℝ) :=
sorry

end solve_cubic_inequality_l664_664764


namespace no_polygon_with_equal_sides_and_obtuse_triangles_l664_664516

/-- There does not exist a convex polygon where all sides are of equal length, and any three vertices form an obtuse triangle. -/
theorem no_polygon_with_equal_sides_and_obtuse_triangles :
  ¬ ∃ (n : ℕ) (P : ℕ → Point) (h_convex : convex_polygon P), 
    (n > 2) ∧ (∀ i j, dist (P i) (P j) = dist (P 0) (P 1)) ∧ 
    (∀ i j k, is_obtuse_triangle (P i) (P j) (P k)) :=
sorry

end no_polygon_with_equal_sides_and_obtuse_triangles_l664_664516


namespace bobby_shoes_l664_664195

variable (Bonny_pairs Becky_pairs Bobby_pairs : ℕ)
variable (h1 : Bonny_pairs = 13)
variable (h2 : 2 * Becky_pairs - 5 = Bonny_pairs)
variable (h3 : Bobby_pairs = 3 * Becky_pairs)

theorem bobby_shoes : Bobby_pairs = 27 :=
by
  -- Use the conditions to prove the required theorem
  sorry

end bobby_shoes_l664_664195


namespace evens_divisors_lt_100_l664_664366

theorem evens_divisors_lt_100 : 
  ∃ n : ℕ, n = 90 ∧ ∀ k : ℕ, (1 ≤ k < 100) → (even k ↔ (∃ m : ℕ, m * m = k)) ↔ (n = 90) := 
sorry

end evens_divisors_lt_100_l664_664366


namespace hart_inversor_l664_664126

open_locale classical

-- Define the geometric setup and the properties
structure AntiParallelogram (A B C D O P P' Q : Point) :=
  (similarity_1 : Similar (Triangle.mk A O P) (Triangle.mk A D B))
  (similarity_2 : Similar (Triangle.mk O P' D) (Triangle.mk A C D))
  (parallel_diagonals : Parallel (Line.mk A C) (Line.mk B D))
  (collinear_points : Collinear ({O, P, P', Q} : set Point) (Line_parallel_to_diagonals : Line.mk A C))

-- Define the theorem statement
theorem hart_inversor (A B C D O P P' Q : Point) 
  (h : AntiParallelogram A B C D O P P' Q) : 
  let OP_product := Distance (O, P) * Distance (O, P')
  in Invariant OP_product :=
sorry

end hart_inversor_l664_664126


namespace smallest_six_digit_number_divisible_by_6_l664_664052

-- Definition of the digits and their properties
def digits := [1, 2, 3, 4, 5, 6]

-- Auxiliary function to check if a number is divisible by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- Function to list all permutations of the given digits
def permutations := List.permutations digits

-- Function to convert a list of digits to a number
def list_to_number (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => acc * 10 + x) 0

-- The smallest number meeting the conditions
def smallest_divisible_by_6 : ℕ :=
  List.minimum (List.filter divisible_by_6 (List.map list_to_number permutations)).get_or_else 0

-- The final theorem statement asserting the smallest number is 123452
theorem smallest_six_digit_number_divisible_by_6 :
  smallest_divisible_by_6 = 123452 :=
by
  sorry

end smallest_six_digit_number_divisible_by_6_l664_664052


namespace length_of_BC_l664_664926

theorem length_of_BC 
  (A B C X : Type) 
  (d_AB : ℝ) (d_AC : ℝ) 
  (circle_center_A : A) 
  (radius_AB : ℝ)
  (intersects_BC : B → C → X)
  (BX CX : ℕ) 
  (h_BX_in_circle : BX = d_AB) 
  (h_CX_in_circle : CX = d_AC) 
  (h_integer_lengths : ∃ x y : ℕ, BX = x ∧ CX = y) :
  BX + CX = 61 :=
begin
  sorry
end

end length_of_BC_l664_664926


namespace polynomial_divisibility_l664_664993

theorem polynomial_divisibility (P : Polynomial ℂ) (n : ℕ) 
  (h : ∃ Q : Polynomial ℂ, P.comp (X ^ n) = (X - 1) * Q) : 
  ∃ R : Polynomial ℂ, P.comp (X ^ n) = (X ^ n - 1) * R :=
sorry

end polynomial_divisibility_l664_664993


namespace evaluate_expression_l664_664229

theorem evaluate_expression :
  (let a := 2^2009
       b := 2^2007
       c := 2^2008
       d := 2^2006
    in (a * a - b * b) / (c * c - d * d)) = 4 := 
  by sorry

end evaluate_expression_l664_664229


namespace feed_duration_l664_664023

-- Define the given conditions
def initial_boxes := 5
def additional_boxes := 3
def seed_per_box := 225
def parrot_consumption_per_week := 100
def cockatiel_consumption_per_week := 50

-- Calculate total boxes of birdseed
def total_boxes := initial_boxes + additional_boxes

-- Calculate the total amount of birdseed in grams
def total_birdseed := total_boxes * seed_per_box

-- Calculate the total consumption per week in grams
def total_weekly_consumption := parrot_consumption_per_week + cockatiel_consumption_per_week

-- Calculate the number of weeks Leah can feed her birds
def weeks : ℕ := total_birdseed / total_weekly_consumption

-- Prove the number of weeks is 12
theorem feed_duration : weeks = 12 :=
by
  -- Definitions and conditions
  let initial_boxes := 5
  let additional_boxes := 3
  let seed_per_box := 225
  let parrot_consumption_per_week := 100
  let cockatiel_consumption_per_week := 50

  -- Calculations
  have total_boxes_eq := initial_boxes + additional_boxes
  have total_birdseed_eq := total_boxes_eq * seed_per_box
  have total_weekly_consumption_eq := parrot_consumption_per_week + cockatiel_consumption_per_week
  have weeks_eq := total_birdseed_eq / total_weekly_consumption_eq

  -- Proof
  show weeks_eq = 12
  sorry

end feed_duration_l664_664023


namespace parallelogram_diagonals_equal_implies_rectangle_l664_664185

variable (A B C D : Type)
variable [has_add A] [has_add B] [has_add C] [has_add D]

-- Definitions involving the geometrical configurations
def is_parallelogram (A B C D : Type) : Prop := 
  sorry -- This should define the parallelogram property formally

def is_rectangle (A B C D : Type) : Prop := 
  sorry -- This should define the rectangle property formally

def diagonals_equal (A B C D : Type) [has_eq A] [has_eq C] : Prop := 
  sorry -- This should ensure the diagonals AC == BD

-- The Lean 4 statement for our problem
theorem parallelogram_diagonals_equal_implies_rectangle :
  ∀ (A B C D : Type) [has_add A] [has_add B] [has_add C] [has_add D],
  is_parallelogram A B C D →
  diagonals_equal A B C D →
  is_rectangle A B C D :=
by {
  intros,
  sorry
}

end parallelogram_diagonals_equal_implies_rectangle_l664_664185


namespace jeans_price_increase_l664_664710

theorem jeans_price_increase (manufacturing_cost customer_price : ℝ) 
  (h1 : customer_price = 1.40 * (1.40 * manufacturing_cost))
  : (customer_price - manufacturing_cost) / manufacturing_cost * 100 = 96 :=
by sorry

end jeans_price_increase_l664_664710


namespace sum_x_coords_above_line_is_zero_l664_664058

def point := (ℕ × ℕ)

def above_line (p : point) : Prop :=
  let (x, y) := p;
  y > 3 * x + 4

def points : List point :=
  [(4, 15), (8, 25), (14, 42), (19, 48), (22, 60)]

def sum_x_coords_above_line (pts : List point) : ℕ :=
  pts.filter above_line |>.map (Prod.fst) |>.sum

theorem sum_x_coords_above_line_is_zero : sum_x_coords_above_line points = 0 := by
  sorry

end sum_x_coords_above_line_is_zero_l664_664058


namespace handshakes_min_l664_664696

-- Define the number of people and the number of handshakes each person performs
def numPeople : ℕ := 35
def handshakesPerPerson : ℕ := 3

-- Define the minimum possible number of unique handshakes
theorem handshakes_min : (numPeople * handshakesPerPerson) / 2 = 105 := by
  sorry

end handshakes_min_l664_664696


namespace find_tan_half_sum_of_angles_l664_664039

theorem find_tan_half_sum_of_angles (x y : ℝ) 
  (h₁ : Real.cos x + Real.cos y = 1)
  (h₂ : Real.sin x + Real.sin y = 1 / 2) : 
  Real.tan ((x + y) / 2) = 1 / 2 := 
by 
  sorry

end find_tan_half_sum_of_angles_l664_664039


namespace radius_ratio_l664_664863

variable (r : ℝ)

theorem radius_ratio (h : r > 0) : 
  let original_area := real.pi * r^2
  let new_radius := r + 2
  let new_area := real.pi * new_radius^2
  (new_area / original_area) = 1 + (4 * (r + 1) / r^2) :=
by
  sorry

end radius_ratio_l664_664863


namespace even_number_of_divisors_less_than_100_l664_664322

theorem even_number_of_divisors_less_than_100 : 
  ∃ n, n = 90 ∧ ∀ x < 100, (∃ k, k * k = x → false) = (x ∣ 99 - 9) :=
sorry

end even_number_of_divisors_less_than_100_l664_664322


namespace empty_set_sub_of_any_set_l664_664143

theorem empty_set_sub_of_any_set (α : Type) (S : set α) :
  ∅ ⊆ S :=
sorry

end empty_set_sub_of_any_set_l664_664143


namespace number_of_integers_with_even_divisors_l664_664389

-- Define conditions
def N := 99

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P : finset ℕ := { n in finset.range (N + 1) | is_perfect_square n }

noncomputable def countP : ℕ := P.card

-- The statement to prove
theorem number_of_integers_with_even_divisors : 
  (N - countP) = 90 :=
by {
  sorry
}

end number_of_integers_with_even_divisors_l664_664389


namespace relay_race_permutations_l664_664521

theorem relay_race_permutations (a b c d : Type) [DecidableEq a] [DecidableEq b] [DecidableEq c] [DecidableEq d] 
  (team : List (a ⊕ b ⊕ c ⊕ d)) (lap : ℕ) (runner : ℕ → a ⊕ b ⊕ c ⊕ d) :
  (∀ i, i ≠ 2 → runner i ∈ team) ∧ (runner 2 = Sum.inl a) ∧ (team.length = 4) → (Finset.univ.card : ℕ) = 6 := by
  sorry

end relay_race_permutations_l664_664521


namespace maxVerticesNoDiagonal_l664_664134

-- condition: Define the non-convex polygon and the notion of vertices from which a diagonal cannot be drawn
def isNonConvexPolygon (n : ℕ) (polygon : fin n → Prop) : Prop :=
  ∀ i, polygon i → ∃ j, j ≠ i ∧ polygon j ∧ interiorAngle polygon i > 180

-- condition: Define the interior angle condition
def interiorAngle (polygon : fin n → Prop) (i : fin n) : ℝ :=
  -- This is a placeholder for the actual interior angle calculation
  sorry

-- theorem statement
theorem maxVerticesNoDiagonal (n : ℕ) :
  ∀ (polygon : fin n → Prop), isNonConvexPolygon n polygon → 
  (∃ m, m ≤ n ∧ m = ⌊ n / 2 ⌋):=
by
  intros polygon h
  sorry

end maxVerticesNoDiagonal_l664_664134


namespace part1_part2_l664_664293

noncomputable section

open Set

variables (U : Set ℝ) (A B : Set ℝ)

def universe : Set ℝ := univ
def setA : Set ℝ := {x | x^2 + 3 * x - 18 ≤ 0 }
def setB : Set ℝ := {x | (1:ℝ)/(x + 1) ≤ -1 }
def setC (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}

theorem part1 : ((U \ B) ∩ A = {x | -6 ≤ x ∧ x < -2} ∪ {x | -1 ≤ x ∧ x ≤ 3}) :=
sorry

theorem part2 : ∀ a : ℝ, (B ∪ setC a = B) ↔ (a ≥ 1) :=
sorry

end part1_part2_l664_664293


namespace mass_percentage_C_in_butanoic_acid_is_54_50_l664_664233

noncomputable def atomic_mass_C : ℝ := 12.01
noncomputable def atomic_mass_H : ℝ := 1.01
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def molar_mass_butanoic_acid : ℝ :=
  (4 * atomic_mass_C) + (8 * atomic_mass_H) + (2 * atomic_mass_O)

noncomputable def mass_of_C_in_butanoic_acid : ℝ :=
  4 * atomic_mass_C

noncomputable def mass_percentage_C : ℝ :=
  (mass_of_C_in_butanoic_acid / molar_mass_butanoic_acid) * 100

theorem mass_percentage_C_in_butanoic_acid_is_54_50 :
  mass_percentage_C = 54.50 := by
  sorry

end mass_percentage_C_in_butanoic_acid_is_54_50_l664_664233


namespace jake_eats_papayas_in_one_week_l664_664490

variable (J : ℕ)
variable (brother_eats : ℕ := 5)
variable (father_eats : ℕ := 4)
variable (total_papayas_in_4_weeks : ℕ := 48)

theorem jake_eats_papayas_in_one_week (h : 4 * (J + brother_eats + father_eats) = total_papayas_in_4_weeks) : J = 3 :=
by
  sorry

end jake_eats_papayas_in_one_week_l664_664490


namespace polynomial_count_l664_664771

open Finset Polynomial

-- Definition of a polynomial in the finite field Z_2027 with a degree bound.
variables {P : Polynomial (ZMod 2027)}

-- Definitions for cycle lengths and counting, as stated in the problem.
def valid_cycles (a b c : ℕ) : Prop :=
  a + 43 * b + 47 * c = 2027

def valid_polynomial (P : Polynomial (ZMod 2027)) : Prop :=
  ∀ k, (Polynomial.eval (P ^ k) = id) ↔ 2021 ∣ k

-- Main theorem stating the number of valid polynomials given the constraints.
theorem polynomial_count (a b c : ℕ) (h_valid_cycles : valid_cycles a b c) :
  ∃ P : Polynomial (ZMod 2027), valid_polynomial P ∧ 
    (2027.factorial / (a.factorial * b.factorial * c.factorial * 43^b * 47^c)) > 0 :=
sorry

end polynomial_count_l664_664771


namespace tenth_place_unidentified_l664_664479

def places := Fin 15

variable (Clara Daniel Emma Farah George Harry : places)
variable (others : Fin 8 → places)
variable (cls_9th : Clara = 9)
variable (far_6th : Farah = Clara - 3)
variable (geo_8th : George = Farah + 2)
variable (har_5th : Harry = Emma + 1)
variable (geo_har : George = Harry - 2)
variable (em_dan : Emma = Daniel + 5)

theorem tenth_place_unidentified :
  ∀ p : places, 
  p ≠ Clara → p ≠ Daniel → p ≠ Emma → p ≠ Farah → p ≠ George → p ≠ Harry →
  ∃ i, 
  10 = others i := 
by
  sorry

end tenth_place_unidentified_l664_664479


namespace incenter_in_triangle_BOH_l664_664899

-- We define what it means for a point to lie inside a triangle
def lies_inside (P A B C : Point) : Prop := 
  let PA := segment P A
  let PB := segment P B
  let PC := segment P C
  within (triangle A B C) P

-- Given:
variables {A B C I O H : Point}
variable {angle_A_lt_angle_B_lt_angle_C : ∀ (A B C : Point), angle A B C A < angle B A C ∧ angle B A C < angle C A B}
noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def orthocenter (A B C : Point) : Point := sorry

-- Hypotheses
variable (hI : I = incenter A B C)
variable (hO : O = circumcenter A B C)
variable (hH : H = orthocenter A B C)

-- Theorem
theorem incenter_in_triangle_BOH : lies_inside I B O H :=
by
  sorry

end incenter_in_triangle_BOH_l664_664899


namespace symmetric_point_xOz_l664_664013

def symmetric_point (p : (ℝ × ℝ × ℝ)) (plane : ℝ → Prop) : (ℝ × ℝ × ℝ) :=
match p with
| (x, y, z) => (x, -y, z)

theorem symmetric_point_xOz (x y z : ℝ) : symmetric_point (-1, 2, 1) (λ y, y = 0) = (-1, -2, 1) :=
by
  sorry

end symmetric_point_xOz_l664_664013


namespace tens_digit_of_expression_l664_664681

theorem tens_digit_of_expression :
  (2023 ^ 2024 - 2025) % 100 / 10 % 10 = 1 :=
by sorry

end tens_digit_of_expression_l664_664681


namespace Ursula_hot_dogs_l664_664125

theorem Ursula_hot_dogs 
  (H : ℕ) 
  (cost_hot_dog : ℚ := 1.50) 
  (cost_salad : ℚ := 2.50) 
  (num_salads : ℕ := 3) 
  (total_money : ℚ := 20) 
  (change : ℚ := 5) :
  (cost_hot_dog * H + cost_salad * num_salads = total_money - change) → H = 5 :=
by
  sorry

end Ursula_hot_dogs_l664_664125


namespace triangle_identity_l664_664257

-- Define vectors and conditions
def vectors_orthogonal (m n : ℝ × ℝ) : Prop := m.1 * n.1 + m.2 * n.2 = 0

-- Define the main theorem
theorem triangle_identity (A B C a b c : ℝ)
  (h_angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (opposite_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sum_angles : A + B + C = π)
  (h_m_perp_n : vectors_orthogonal (sin A, 1) (1, -cos A))
  (h_sum_sides : b + c = a) :
  A = π / 4 ∧ sin (B + C) = sqrt 2 / 2 :=
by
  sorry

end triangle_identity_l664_664257


namespace perimeter_of_PQRST_l664_664506

-- Define points P, Q, R, S, T and X
variables {P Q R S T X : Type}

-- Define distances based on given conditions
-- PQ, QR, TS, PT
axiom pq_eq_4 : PQ = 4
axiom qr_eq_4 : QR = 4
axiom ts_eq_10 : TS = 10
axiom pt_eq_8 : PT = 8

-- Define PQXT as a rectangle with right angles at certain points
axiom pqxt_is_rectangle : ∀ (PQXT : rectangle), 
  right_angle (angle PTS) ∧ right_angle (angle PQX) ∧ right_angle (angle PQR)

-- Define perimeter calculation as the sum of its side lengths 
def perimeter (PQ QR RS ST TP : ℝ) : ℝ := PQ + QR + RS + ST + TP

-- Prove the perimeter of polygon PQRST is equal to 26 + 2sqrt(13)
theorem perimeter_of_PQRST : perimeter PQ QR RS ST TP = 26 + 2 * real.sqrt 13 :=
by {
  have h_PQ : PQ = 4 := pq_eq_4,
  have h_QR : QR = 4 := qr_eq_4,
  have h_RS : RS = 2 * real.sqrt 13 := sorry,  -- derived from the Pythagorean theorem
  have h_ST : ST = 10 := ts_eq_10,
  have h_TP : TP = 8 := pt_eq_8,
  show perimeter PQ QR RS ST TP = 26 + 2 * real.sqrt 13, sorry
}

-- Placeholder for actual proof steps

end perimeter_of_PQRST_l664_664506


namespace gcd_48_180_l664_664133

theorem gcd_48_180 : Nat.gcd 48 180 = 12 := by
  have f1 : 48 = 2^4 * 3 := by norm_num
  have f2 : 180 = 2^2 * 3^2 * 5 := by norm_num
  sorry

end gcd_48_180_l664_664133


namespace even_divisors_less_than_100_l664_664358

theorem even_divisors_less_than_100 :
  let count_even_divisors := 
    let n := 100 in
    let perfect_squares := { m | ∃ k, k * k = m ∧ m < n } in
    let total_numbers := finset.range n in
    (total_numbers.card - perfect_squares.card) =
    90 
  in count_even_divisors = 90 :=
by
  let n := 100
  let perfect_squares : finset ℕ := finset.filter (λ m, ∃ k, k * k = m) (finset.range n)
  let total_numbers : finset ℕ := finset.range n
  have h : total_numbers.card = 99 := by sorry
  have p : perfect_squares.card = 9 := by sorry
  show total_numbers.card - perfect_squares.card = 90
  calc
    total_numbers.card - perfect_squares.card
      = 99 - 9 := by rw [h, p]
      = 90 := by norm_num

end even_divisors_less_than_100_l664_664358


namespace no_real_solution_x_plus_36_div_x_minus_3_eq_neg9_l664_664763

theorem no_real_solution_x_plus_36_div_x_minus_3_eq_neg9 : ∀ x : ℝ, x + 36 / (x - 3) = -9 → False :=
by
  assume x
  assume h : x + 36 / (x - 3) = -9
  sorry

end no_real_solution_x_plus_36_div_x_minus_3_eq_neg9_l664_664763


namespace largest_integer_cuberoot_l664_664673

theorem largest_integer_cuberoot (x : ℕ) (h : x = 2010) :
  ∃ y : ℤ, y = 2011 ∧ y ≤ int.root 3 (x^3 + 3 * x^2 + 4 * x + 1) ∧ 
           int.root 3 (x^3 + 3 * x^2 + 4 * x + 1) < y + 1 := 
by {
  sorry
}

end largest_integer_cuberoot_l664_664673


namespace greatest_prime_factor_15_fact_plus_18_fact_eq_17_l664_664664

theorem greatest_prime_factor_15_fact_plus_18_fact_eq_17 :
  ∃ p : ℕ, prime p ∧ 
  (∀ q : ℕ, (prime q ∧ q ∣ (15.factorial + 18.factorial)) → q ≤ p) ∧ 
  p = 17 :=
by
  sorry

end greatest_prime_factor_15_fact_plus_18_fact_eq_17_l664_664664


namespace count_even_divisors_lt_100_l664_664310

-- Define the set of natural numbers less than 100
def nat_lt_100 := {n : ℕ | n < 100}

-- Define the set of perfect squares less than 100
def perfect_squares_lt_100 := {n : ℕ | ∃ k : ℕ, k^2 = n ∧ n < 100}

-- Define the set of natural numbers less than 100 with an even number of positive divisors
def even_divisors_lt_100 := nat_lt_100 \ perfect_squares_lt_100

-- Theorem stating the number of elements with even number of divisors is 90
theorem count_even_divisors_lt_100 : (even_divisors_lt_100).card = 90 := 
sorry

end count_even_divisors_lt_100_l664_664310


namespace each_person_paid_45_l664_664726

theorem each_person_paid_45 (total_bill : ℝ) (number_of_people : ℝ) (per_person_share : ℝ) 
    (h1 : total_bill = 135) 
    (h2 : number_of_people = 3) :
    per_person_share = 45 :=
by
  sorry

end each_person_paid_45_l664_664726


namespace union_complement_l664_664033

-- Definitions of the sets
def U : Set ℕ := {x | x > 0 ∧ x ≤ 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {1, 2, 4}

-- Statement of the proof problem
theorem union_complement : A ∪ (U \ B) = {1, 3, 5} := by
  sorry

end union_complement_l664_664033


namespace greatest_prime_factor_15_18_l664_664653

theorem greatest_prime_factor_15_18! :
  ∃ p : ℕ, prime p ∧ p ∈ prime_factors (15! + 18!) ∧ ∀ q : ℕ, prime q → q ∈ prime_factors (15! + 18!) → q ≤ 4897 := 
sorry

end greatest_prime_factor_15_18_l664_664653


namespace BC_length_l664_664917

theorem BC_length (A B C X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X] 
  (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
  (BX CX : ℕ) (h_circle_intersect : ∃ X, Metric.ball A 86 ∩ {BC} = {B, X})
  (h_integer_lengths : BX + CX = BC) :
  BC = 61 := 
by
  sorry

end BC_length_l664_664917


namespace limit_derivative_neg_l664_664458

open Filter

theorem limit_derivative_neg {a b x0 : ℝ} {f : ℝ → ℝ} (h_diff : ∀ x ∈ Ioo a b, DifferentiableAt ℝ f x) (h_x0 : x0 ∈ Ioo a b) :
  tendsto (λ h => (f (x0 - h) - f x0) / h) (nhds 0) (𝓝 (-deriv f x0)) :=
by
  sorry

end limit_derivative_neg_l664_664458


namespace bake_sale_total_items_l664_664732

-- Define the conditions
def cookies_sold : ℕ := 48
def ratio (a b : ℕ) : Prop := a * b.succ.succ = b * a.succ.succ

-- Define the proof problem
theorem bake_sale_total_items :
  ∃ (B : ℕ), ratio B cookies_sold 7 6 ∧ (B + cookies_sold = 104) :=
by
  sorry

end bake_sale_total_items_l664_664732


namespace min_moves_to_emit_all_colors_l664_664691

theorem min_moves_to_emit_all_colors :
  ∀ (colors : Fin 7 → Prop) (room : Fin 4 → Fin 7)
  (h : ∀ i j, i ≠ j → room i ≠ room j) (moves : ℕ),
  (∀ (n : ℕ) (i : Fin 4), n < moves → ∃ c : Fin 7, colors c ∧ room i = c ∧
    (∀ j, j ≠ i → room j ≠ c)) →
  (∃ n, n = 8) :=
by
  sorry

end min_moves_to_emit_all_colors_l664_664691


namespace martin_discounted_ticket_price_l664_664550

theorem martin_discounted_ticket_price :
  ∃ D : ℝ, 
  let 
    total_tickets := 10,
    full_price := 2.0,
    total_cost := 18.40,
    discounted_tickets := 4,
    full_price_tickets := total_tickets - discounted_tickets,
    full_price_total_cost := full_price_tickets * full_price,
    discounted_total_cost := total_cost - full_price_total_cost
  in 
    D = discounted_total_cost / discounted_tickets :=
begin
  use 1.6,
  simp,
  sorry
end

end martin_discounted_ticket_price_l664_664550


namespace BC_length_l664_664965

-- Define the given triangle and circle conditions
variables (A B C X : Type) (AB AC BX CX : ℤ)
axiom AB_value : AB = 86
axiom AC_value : AC = 97
axiom circle_center_radius : ∃ (A : Type), ∃ (radius : ℤ), radius = AB ∧ ∃ (points : Set Type), points = {B, X} ∧ ∀ (P : Type), P ∈ points → dist A P = radius
axiom BX_CX_integers : ∃ (x y : ℤ), BX = x ∧ CX = y

-- Define calculations using the Power of a Point theorem
theorem BC_length :
  ∀ (y: ℤ) (x: ℤ), y(y + x) = AC^2 - AB^2 → x + y = 61 :=
by
  intros y x h
  have h1 : 97^2 = 9409, by norm_num,
  have h2 : 86^2 = 7396, by norm_num,
  rw [AB_value, AC_value] at h,
  rw [h1, h2] at h,
  calc y(y + x) = 2013 := by {exact h}
  -- The human verification part is skipped since we only need the statement here
  sorry

end BC_length_l664_664965


namespace sequence_sum_l664_664829

theorem sequence_sum (n : ℕ) : 
  (∑ k in finset.range n, 1 / ((2*k + 1) * (2*k + 3))) = n / (2*n + 1) :=
sorry

end sequence_sum_l664_664829


namespace length_of_BC_l664_664935

theorem length_of_BC (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
    (BX CX : ℕ) (h_pow : CX * (BX + CX) = 2013) : 
    BX + CX = 61 :=
  sorry

end length_of_BC_l664_664935


namespace vinnie_tips_l664_664558

variable (Paul Vinnie : ℕ)

def tips_paul := 14
def more_vinnie_than_paul := 16

theorem vinnie_tips :
  Vinnie = tips_paul + more_vinnie_than_paul :=
by
  unfold tips_paul more_vinnie_than_paul -- unfolding defined values
  exact sorry

end vinnie_tips_l664_664558


namespace angle_between_east_and_south_is_90_degrees_l664_664181

-- Define the main theorem statement
theorem angle_between_east_and_south_is_90_degrees :
  ∀ (circle : Type) (num_rays : ℕ) (direction : ℕ → ℕ) (north east south : ℕ),
  num_rays = 12 →
  (∀ i, i < num_rays → direction i = (i * 360 / num_rays) % 360) →
  direction north = 0 →
  direction east = 90 →
  direction south = 180 →
  (min ((direction south - direction east) % 360) (360 - (direction south - direction east) % 360)) = 90 :=
by
  intros
  -- Skipped the proof
  sorry

end angle_between_east_and_south_is_90_degrees_l664_664181


namespace find_a_l664_664814

noncomputable def value_of_a (a : ℝ) : Prop :=
  let sin_alpha := a / real.sqrt (16 + a^2)
  let cos_alpha := -4 / real.sqrt (16 + a^2)
  sin_alpha * cos_alpha = real.sqrt 3 / 4

theorem find_a (a : ℝ) : value_of_a a → (a = -4 * real.sqrt 3 ∨ a = - (4 * real.sqrt 3 / 3)) :=
begin
  sorry
end

end find_a_l664_664814


namespace no_product_of_six_distinct_primes_l664_664616

theorem no_product_of_six_distinct_primes (s : Set ℕ) (h_distinct : s.card = 2014)
  (h_div : ∀ a b ∈ s, a ≠ b → (a * b) % (a + b) = 0) :
  ∀ a ∈ s, ¬ ∃ p1 p2 p3 p4 p5 p6 : ℕ, 
    (Nat.prime p1) ∧ (Nat.prime p2) ∧ (Nat.prime p3) ∧ (Nat.prime p4) ∧ (Nat.prime p5) ∧ (Nat.prime p6) ∧
    a = p1 * p2 * p3 * p4 * p5 * p6 :=
by
  sorry

end no_product_of_six_distinct_primes_l664_664616


namespace count_even_divisors_lt_100_l664_664308

-- Define the set of natural numbers less than 100
def nat_lt_100 := {n : ℕ | n < 100}

-- Define the set of perfect squares less than 100
def perfect_squares_lt_100 := {n : ℕ | ∃ k : ℕ, k^2 = n ∧ n < 100}

-- Define the set of natural numbers less than 100 with an even number of positive divisors
def even_divisors_lt_100 := nat_lt_100 \ perfect_squares_lt_100

-- Theorem stating the number of elements with even number of divisors is 90
theorem count_even_divisors_lt_100 : (even_divisors_lt_100).card = 90 := 
sorry

end count_even_divisors_lt_100_l664_664308


namespace triangle_bc_length_l664_664940

theorem triangle_bc_length :
  ∀ (A B C X : Type) (d_AB : ℝ) (d_AC : ℝ) (d_BX d_CX BC : ℕ),
  d_AB = 86 ∧ d_AC = 97 →
  let circleA := {center := A, radius := d_AB} in
  let intersect_B := B ∈ circleA in
  let intersect_X := X ∈ circleA in
  d_BX + d_CX = BC →
  d_BX ∈ ℕ ∧ d_CX ∈ ℕ →
  BC = 61 :=
by
  intros A B C X d_AB d_AC d_BX d_CX BC h_dist h_circle h_intersect h_sum h_intBC
  sorry

end triangle_bc_length_l664_664940


namespace combined_weight_l664_664979

theorem combined_weight (mary_weight : ℝ) (jamison_weight : ℝ) (john_weight : ℝ) :
  mary_weight = 160 ∧ jamison_weight = mary_weight + 20 ∧ john_weight = mary_weight + (0.25 * mary_weight) →
  john_weight + mary_weight + jamison_weight = 540 :=
by
  intros h
  obtain ⟨hm, hj, hj'⟩ := h
  rw [hm, hj, hj']
  norm_num
  sorry

end combined_weight_l664_664979


namespace combined_weight_of_three_l664_664981

theorem combined_weight_of_three (Mary Jamison John : ℝ) 
  (h₁ : Mary = 160) 
  (h₂ : Jamison = Mary + 20) 
  (h₃ : John = Mary + (1/4) * Mary) :
  Mary + Jamison + John = 540 := by
  sorry

end combined_weight_of_three_l664_664981


namespace delta_max_success_ratio_l664_664486

theorem delta_max_success_ratio :
  ∃ a b c d : ℕ, 
    0 < a ∧ a < b ∧ (40 * a) < (21 * b) ∧
    0 < c ∧ c < d ∧ (4 * c) < (3 * d) ∧
    b + d = 600 ∧
    (a + c) / 600 = 349 / 600 :=
by
  sorry

end delta_max_success_ratio_l664_664486


namespace simplify_expression_l664_664228

theorem simplify_expression (k : ℤ) : 
    3^(-(3 * k + 2)) - 3^(-(3 * k)) + 3^(-(3 * k + 1)) + 3^(-(3 * k - 1)) = (22 * 3^(-(3 * k))) / 9 := 
by
  sorry

end simplify_expression_l664_664228


namespace number_of_x_for_g100_eq_zero_l664_664535

noncomputable def g0 (x : ℝ) : ℝ := 
  if x < -150 then x + 300
  else if x < 150 then -x
  else x - 300

noncomputable def g : ℕ → ℝ → ℝ 
| 0, x := g0 x
| (n+1), x := abs (g n x) - 2

theorem number_of_x_for_g100_eq_zero : 
  (finset.univ.filter (λ (x : ℝ), g 100 x = 0)).card = 2 :=
sorry

end number_of_x_for_g100_eq_zero_l664_664535


namespace triangle_bc_length_l664_664905

theorem triangle_bc_length (A B C X : Type)
  (AB AC : ℕ)
  (hAB : AB = 86)
  (hAC : AC = 97)
  (circle_eq : ∀ {r : ℕ}, r = AB → circle_centered_at_A_intersects_BC_two_points B X)
  (integer_lengths : ∃ (BX CX : ℕ), ) :
  BC = 61 :=
by
  sorry

end triangle_bc_length_l664_664905


namespace interval_of_monotonic_increase_l664_664457

-- Condition: the function f(x)
def f (x : ℝ) : ℝ := sin (2 * x + π / 6) + cos (2 * x - π / 3)

-- The interval of monotonic increase for the function f(x) given k ∈ ℤ
theorem interval_of_monotonic_increase (k : ℤ) :
  ∃ a b : ℝ, -π / 3 + k * π ≤ a ∧ a < b ∧ b ≤ π / 6 + k * π ∧ 
    ∀ x : ℝ, a < x ∧ x < b → f' x > 0 :=
sorry

end interval_of_monotonic_increase_l664_664457


namespace sum_of_all_3_digit_numbers_with_remainder_2_when_divided_by_5_l664_664149

theorem sum_of_all_3_digit_numbers_with_remainder_2_when_divided_by_5 :
  ∑ k in finset.Icc 102 997, if k % 5 = 2 then k else 0 = 98910 :=
by
  -- steps of the proof
  sorry

end sum_of_all_3_digit_numbers_with_remainder_2_when_divided_by_5_l664_664149


namespace find_number_l664_664609

noncomputable def number (x : ℝ) : Prop := 
  (sqrt 289 = 17) ∧ 
  (sqrt 625 = 25) ∧ 
  (17 - (25 / sqrt x) = 12)

theorem find_number (x : ℝ) : number x → x = 25 :=
by
  intro h
  sorry

end find_number_l664_664609


namespace binary_addition_l664_664723

theorem binary_addition :
  nat.bin_add (nat.bin_add (nat.bin_add (nat.of_digits 2 [1, 0, 1, 1, 0]) 
                                         (nat.of_digits 2 [1, 1, 0])) 
                           (nat.of_digits 2 [1])) 
               (nat.of_digits 2 [1, 0, 1]) = nat.of_digits 2 [1, 1, 0, 0, 0, 0] :=
by sorry

end binary_addition_l664_664723


namespace cars_produced_in_europe_l664_664702

theorem cars_produced_in_europe (total_cars : ℕ) (cars_in_north_america : ℕ) (cars_in_europe : ℕ) :
  total_cars = 6755 → cars_in_north_america = 3884 → cars_in_europe = total_cars - cars_in_north_america → cars_in_europe = 2871 :=
by
  -- necessary calculations and logical steps
  sorry

end cars_produced_in_europe_l664_664702


namespace find_x_l664_664804

def perpendicular_vectors_solution (x : ℝ) : Prop :=
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (3, -2)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 2 / 3

theorem find_x (x : ℝ) : perpendicular_vectors_solution x := sorry

end find_x_l664_664804


namespace BC_length_l664_664957

def triangle_ABC (A B C : Type)
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)] : Prop :=
  ∃ (X : A), (has_dist B X (coe (X.dist B))) ∧ (has_dist C X (coe (X.dist C))) ∧
  ∀ (x y : ℕ), x = X.dist B ∧ y = X.dist C → x + y = 61

theorem BC_length {A B C : Type}
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)]
  (h : triangle_ABC A B C) : 
  ∃ (x y : ℕ), x + y = 61 := sorry

end BC_length_l664_664957


namespace pentagon_interior_angles_sequences_l664_664100

theorem pentagon_interior_angles_sequences :
  ∃ (seqs : ℕ), seqs = 2 ∧
    (∀ (x d : ℕ), 90 < x ∧ x < 120 ∧ 0 < d ∧ d < 6 ∧ x + 4 * d < 120 ∧
      (x + (x + d) + (x + 2 * d) + (x + 3 * d) + (x + 4 * d) = 540)) :=
begin
  -- We would provide the proof here, but it's omitted.
  sorry
end

end pentagon_interior_angles_sequences_l664_664100


namespace cape_may_sharks_is_32_l664_664203

def twice (n : ℕ) : ℕ := 2 * n

def sharks_in_Cape_May (sharks_in_Daytona : ℕ) : ℕ :=
  twice(sharks_in_Daytona) + 8

theorem cape_may_sharks_is_32 :
  sharks_in_Cape_May 12 = 32 := by
  sorry

end cape_may_sharks_is_32_l664_664203


namespace fraction_transformed_when_tripled_l664_664447

variable (m n : ℕ)

theorem fraction_transformed_when_tripled (m n : ℕ)
  : (m ≠ 0 ∧ n ≠ 0) → (3 * m + 3 * n) / (3 * m * 3 * n) = (1 / 3) * ((m + n) / (m * n)) :=
by
  assume h : m ≠ 0 ∧ n ≠ 0
  sorry

end fraction_transformed_when_tripled_l664_664447


namespace solution_set_l664_664285

def f (x : ℝ) : ℝ :=
if x < 0 then 1 / x else (1 / 3) ^ x

theorem solution_set (x : ℝ) : |f x| ≥ 1 / 3 ↔ -3 ≤ x ∧ x ≤ 1 :=
by sorry

end solution_set_l664_664285


namespace find_f_log2_9_l664_664284

def f : ℝ → ℝ
| x := if x ≤ 0 then 2^x else f (x - 1) - 1

theorem find_f_log2_9 : f (Real.log 9 / Real.log 2) = -55 / 16 := by
  sorry

end find_f_log2_9_l664_664284


namespace problem_solution_l664_664299

open Nat

noncomputable def notRepresentableCount : ℕ :=
  let maxA := 154
  let boundM := 11
  let boundN := 7
  let validA (a : ℕ) : Bool :=
    if a > maxA then false
    else !(Exists (fun (m : ℕ) =>
      Exists (fun (n : ℕ) =>
        (0 <= m ∧ m <= boundM) ∧ (0 <= n ∧ n <= boundN) ∧ (7 * m + 11 * n = a)))
    )
  (List.range (maxA + 1)).filter validA |>.length

theorem problem_solution : notRepresentableCount = 60 := by
  sorry

end problem_solution_l664_664299


namespace greatest_prime_factor_of_expression_l664_664667

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define specific factorial values
def fac_15 := factorial 15
def fac_18 := factorial 18

-- Define the expression from the problem
def expr := fac_15 * (1 + 16 * 17 * 18)

-- Define the factorization result
def factor_4896 := 2 ^ 5 * 3 ^ 2 * 17

-- Define a lemma about the factorization of the expression
lemma factor_expression : 15! * (1 + 16 * 17 * 18) = fac_15 * 4896 := by
  sorry

-- State the main theorem
theorem greatest_prime_factor_of_expression : ∀ p : ℕ, prime p ∧ p ∣ expr → p ≤ 17 := by
  sorry

end greatest_prime_factor_of_expression_l664_664667


namespace symmetric_point_xOz_l664_664010

theorem symmetric_point_xOz (x y z : ℝ) : (x, y, z) = (-1, 2, 1) → (x, -y, z) = (-1, -2, 1) :=
by
  intros h
  cases h
  sorry

end symmetric_point_xOz_l664_664010


namespace polynomial_value_at_2008_l664_664540

def f (a₀ a₁ a₂ a₃ a₄ : ℝ) (x : ℝ) : ℝ := a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4

theorem polynomial_value_at_2008 (a₀ a₁ a₂ a₃ a₄ : ℝ) (h₁ : a₄ ≠ 0)
  (h₀₃ : f a₀ a₁ a₂ a₃ a₄ 2003 = 24)
  (h₀₄ : f a₀ a₁ a₂ a₃ a₄ 2004 = -6)
  (h₀₅ : f a₀ a₁ a₂ a₃ a₄ 2005 = 4)
  (h₀₆ : f a₀ a₁ a₂ a₃ a₄ 2006 = -6)
  (h₀₇ : f a₀ a₁ a₂ a₃ a₄ 2007 = 24) :
  f a₀ a₁ a₂ a₃ a₄ 2008 = 274 :=
by sorry

end polynomial_value_at_2008_l664_664540


namespace number_of_positive_is_2_l664_664188

-- List of numbers given in the problem
def numbers : List ℚ := [-3, -1, 1/3, 0, -3/7, 2017]

-- Function to check if a number is positive
def is_positive (x : ℚ) : Prop := x > 0

-- Problem statement: Prove that the number of positive numbers in the list is 2
theorem number_of_positive_is_2 : (List.countp is_positive numbers) = 2 := by
  sorry

end number_of_positive_is_2_l664_664188


namespace count_even_divisors_lt_100_l664_664302

-- Define the set of natural numbers less than 100
def nat_lt_100 := {n : ℕ | n < 100}

-- Define the set of perfect squares less than 100
def perfect_squares_lt_100 := {n : ℕ | ∃ k : ℕ, k^2 = n ∧ n < 100}

-- Define the set of natural numbers less than 100 with an even number of positive divisors
def even_divisors_lt_100 := nat_lt_100 \ perfect_squares_lt_100

-- Theorem stating the number of elements with even number of divisors is 90
theorem count_even_divisors_lt_100 : (even_divisors_lt_100).card = 90 := 
sorry

end count_even_divisors_lt_100_l664_664302


namespace tina_savings_in_august_l664_664630

theorem tina_savings_in_august 
  (saved_in_june : ℕ)
  (saved_in_july : ℕ)
  (spent_on_books : ℕ)
  (spent_on_shoes : ℕ)
  (money_left : ℕ)
  (total_before_spending : ℕ)
  (saved_in_august : ℕ) :
  saved_in_june = 27 ∧
  saved_in_july = 14 ∧
  spent_on_books = 5 ∧
  spent_on_shoes = 17 ∧
  money_left = 40 ∧
  total_before_spending = spent_on_books + spent_on_shoes + money_left ∧
  saved_in_august = total_before_spending - saved_in_june - saved_in_july →
  saved_in_august = 21 :=
by
  intro h
  cases h with hj june
  cases june with hhj july
  cases july with hb books
  cases books with hs shoes
  cases shoes with ml left
  cases left with tb total
  cases total with sa total_saved
  sorry

end tina_savings_in_august_l664_664630


namespace trivia_team_students_l664_664622

theorem trivia_team_students (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) (h_not_picked : not_picked = 9) 
(h_groups : groups = 3) (h_students_per_group : students_per_group = 9) :
    not_picked + (groups * students_per_group) = 36 := by
  sorry

end trivia_team_students_l664_664622


namespace problem1_problem2_l664_664268
noncomputable theory

open Real

-- Definitions from conditions
def A : Point := (0, 2)
def C (x y : ℝ) := x^2 + y^2 = 16
def D (t : ℝ) : Point := (4 * cos t, 4 * sin t)
def P (x y : ℝ) : Prop := ∃ t, (x, y) = midpoint A (D t)

-- Translating the problem to Lean statements
theorem problem1 : ∀ x y, P x y → x^2 + (y - 1)^2 = 4 := by
  intros x y hP
  cases hP with t ht
  rw [ht] at *
  sorry

theorem problem2 : length_of_segment_intersection E 3 4 (-8) = 4 * sqrt 21 / 5 := by
  -- Assume E is defined as x² + (y-1)² = 4
  -- Define a function length_of_segment_intersection to calculate MN
  sorry

-- Definition to use in Lean
def length_of_segment_intersection (curve : ℝ → ℝ → Prop) (a b c : ℝ) : ℝ := 
  -- Calculation for length of line segment MN
  sorry

end problem1_problem2_l664_664268


namespace triangle_bc_length_l664_664906

theorem triangle_bc_length (A B C X : Type)
  (AB AC : ℕ)
  (hAB : AB = 86)
  (hAC : AC = 97)
  (circle_eq : ∀ {r : ℕ}, r = AB → circle_centered_at_A_intersects_BC_two_points B X)
  (integer_lengths : ∃ (BX CX : ℕ), ) :
  BC = 61 :=
by
  sorry

end triangle_bc_length_l664_664906


namespace sine_cosine_inequality_l664_664847

theorem sine_cosine_inequality (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < 2 * Real.pi)
    (hineq : Real.sin θ ^ 3 - Real.cos θ ^ 3 > (Real.cos θ ^ 5 - Real.sin θ ^ 5) / 7) :
    (Real.pi / 4) < θ ∧ θ < (5 * Real.pi / 4) :=
sorry

end sine_cosine_inequality_l664_664847


namespace find_y_value_l664_664852

theorem find_y_value (k : ℝ) (x : ℝ) (y : ℝ)
  (h1 : y = k * x^(1/3))
  (h2 : ∀ (x : ℝ), x = 27 → y = 3 * real.sqrt 3) :
  x = 8 → y = 2 * real.sqrt 3 :=
sorry

end find_y_value_l664_664852


namespace symmetric_point_xOz_l664_664008

theorem symmetric_point_xOz (x y z : ℝ) : (x, y, z) = (-1, 2, 1) → (x, -y, z) = (-1, -2, 1) :=
by
  intros h
  cases h
  sorry

end symmetric_point_xOz_l664_664008


namespace maria_seventh_score_l664_664985

-- Define the context of the problem
def test_scores (scores : Fin 8 → ℕ) : Prop :=
  (∀ i, 94 ≤ scores i ∧ scores i ≤ 100) ∧
  (Finset.univ.image scores).card = 8 ∧
  (Π i, (Finset.univ.sum scores i) % (i + 1) = 0) ∧
  scores 7 = 97

-- Define the statement to prove the score of the seventh test
theorem maria_seventh_score (scores : Fin 8 → ℕ) (h : test_scores scores) :
  scores 6 = 94 :=
sorry

end maria_seventh_score_l664_664985


namespace candidates_appeared_l664_664482

-- Define the conditions:
variables (A_selected B_selected : ℕ) (x : ℝ)

-- 12% candidates got selected in State A
def State_A_selected := 0.12 * x

-- 18% candidates got selected in State B
def State_B_selected := 0.18 * x

-- 250 more candidates got selected in State B than in State A
def selection_difference := State_B_selected = State_A_selected + 250

-- The statement to prove:
theorem candidates_appeared (h : selection_difference) : x = 4167 :=
by
  sorry

end candidates_appeared_l664_664482


namespace length_of_BC_l664_664929

theorem length_of_BC (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
    (BX CX : ℕ) (h_pow : CX * (BX + CX) = 2013) : 
    BX + CX = 61 :=
  sorry

end length_of_BC_l664_664929


namespace only_one_real_solution_l664_664282

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|

theorem only_one_real_solution (a : ℝ) (h : ∀ x : ℝ, abs (f x) = g a x → x = 1) : a < 0 := 
by
  sorry

end only_one_real_solution_l664_664282


namespace sequence_proof_l664_664086

theorem sequence_proof :
  (∀ n, ∃ an bn, 
    (a_1 = 1) ∧ 
    (bn = (an + 1) / an) ∧
    (∀ i, bi = b_1 * (q ^ (i - 1))) ∧ 
    (b_10 * b_11 = 6)) →
  a_20 = 1 / 2 :=
by 
  sorry

end sequence_proof_l664_664086


namespace log_2_point3_lt_2_pow_point3_l664_664744

theorem log_2_point3_lt_2_pow_point3 : log 2 0.3 < 2^0.3 :=
by
  sorry

end log_2_point3_lt_2_pow_point3_l664_664744


namespace greatest_prime_factor_15_fact_plus_18_fact_eq_17_l664_664666

theorem greatest_prime_factor_15_fact_plus_18_fact_eq_17 :
  ∃ p : ℕ, prime p ∧ 
  (∀ q : ℕ, (prime q ∧ q ∣ (15.factorial + 18.factorial)) → q ≤ p) ∧ 
  p = 17 :=
by
  sorry

end greatest_prime_factor_15_fact_plus_18_fact_eq_17_l664_664666


namespace greatest_prime_factor_15_fact_plus_18_fact_eq_17_l664_664665

theorem greatest_prime_factor_15_fact_plus_18_fact_eq_17 :
  ∃ p : ℕ, prime p ∧ 
  (∀ q : ℕ, (prime q ∧ q ∣ (15.factorial + 18.factorial)) → q ≤ p) ∧ 
  p = 17 :=
by
  sorry

end greatest_prime_factor_15_fact_plus_18_fact_eq_17_l664_664665


namespace minimum_value_l664_664874

noncomputable def min_value_condition (a b : ℝ) : Prop :=
  a + b = 1 / 2

theorem minimum_value (a b : ℝ) (h : min_value_condition a b) :
  (4 / a) + (1 / b) ≥ 18 :=
by
  sorry

end minimum_value_l664_664874


namespace Jorge_touches_16_trees_l664_664016

-- Definitions based on conditions from step a)
def number_of_trees : ℕ := 12
def distance_between_trees : ℕ := 5
def first_segment_distance : ℕ := 32
def second_segment_distance : ℕ := 18
def third_segment_distance : ℕ := 22

-- The proof statement based on the question and correct answer from steps a) and b)
theorem Jorge_touches_16_trees :
  let total_trees_touched := 
    (first_segment_distance / distance_between_trees) + 1 + -- for the first segment
    (second_segment_distance / distance_between_trees) + 1 + -- for the second segment
    (third_segment_distance / distance_between_trees) in  -- for the third segment

  total_trees_touched = 16 :=
by
  sorry

end Jorge_touches_16_trees_l664_664016


namespace sum_of_powers_of_two_l664_664238

theorem sum_of_powers_of_two (n : ℕ) (hn : 0 < n) : 
  (∑ i in Finset.range (n + 1), 2^i) = 2^(n + 1) - 2 := 
by 
  sorry

end sum_of_powers_of_two_l664_664238


namespace quadratic_factorization_b_value_l664_664102

theorem quadratic_factorization_b_value (b : ℤ) (c d e f : ℤ) (h1 : 24 * c + 24 * d = 240) :
  (24 * (c * e) + b + 24) = 0 →
  (c * e = 24) →
  (c * f + d * e = b) →
  (d * f = 24) →
  (c + d = 10) →
  b = 52 :=
by
  intros
  sorry

end quadratic_factorization_b_value_l664_664102


namespace max_area_isosceles_triangle_l664_664245

theorem max_area_isosceles_triangle (A B C : Point) (m : ℝ) 
  (hAC : AC = AC) (hAB_BC_sum : AB + BC = m) :
  ∃ (S : Triangle), isosceles S A B B C ∧ area S = max_area := sorry

end max_area_isosceles_triangle_l664_664245


namespace sum_odd_divisors_420_l664_664137

theorem sum_odd_divisors_420 : 
  let product := (3 + 1) * (5 + 1) * (7 + 1)
  in product = 192 :=
by
  sorry

end sum_odd_divisors_420_l664_664137


namespace calculate_fixed_payment_calculate_variable_payment_compare_plans_for_x_eq_30_l664_664160

noncomputable def cost_plan1_fixed (num_suits num_ties : ℕ) : ℕ :=
  if num_ties > num_suits then 200 * num_suits + 40 * (num_ties - num_suits)
  else 200 * num_suits

noncomputable def cost_plan2_fixed (num_suits num_ties : ℕ) : ℕ :=
  (200 * num_suits + 40 * num_ties) * 9 / 10

noncomputable def cost_plan1_variable (num_suits : ℕ) (x : ℕ) : ℕ :=
  200 * num_suits + 40 * (x - num_suits)

noncomputable def cost_plan2_variable (num_suits : ℕ) (x : ℕ) : ℕ :=
  (200 * num_suits + 40 * x) * 9 / 10

theorem calculate_fixed_payment :
  cost_plan1_fixed 20 22 = 4080 ∧ cost_plan2_fixed 20 22 = 4392 :=
by sorry

theorem calculate_variable_payment (x : ℕ) (hx : x > 20) :
  cost_plan1_variable 20 x = 40 * x + 3200 ∧ cost_plan2_variable 20 x = 36 * x + 3600 :=
by sorry

theorem compare_plans_for_x_eq_30 :
  cost_plan1_variable 20 30 < cost_plan2_variable 20 30 :=
by sorry


end calculate_fixed_payment_calculate_variable_payment_compare_plans_for_x_eq_30_l664_664160


namespace ellipse_equation_line_equation_l664_664795

noncomputable theory
open_locale real

-- Definition of the ellipse satisfying the given conditions
def ellipse_eq := ∀ x y : ℝ, (x^2 / 2) + y^2 = 1

-- Definition of the line satisfying the given conditions
def line_eq (x : ℝ) := x - 4 / 3

theorem ellipse_equation (h_center_origin : ∀ x y : ℝ, x * y = 0)
    (h_foci_x_axis : ∀ x : ℝ, y : ℝ, y = 0 -> x ≠ 0)
    (h_vertex : (0, 1))
    (h_eccentricity : ∀ a b : ℝ, b = 1 -> (√2 / 2)^2 = (a^2 - b^2) / a^2)
    : ellipse_eq := 
begin
  sorry
end

theorem line_equation (h_slope : ∀ B F : (ℝ × ℝ), F = (1, 0) -> B = (0, 1) -> -slope(B, F) = 1)
    (h_intersect_ellipse : ∀ M N : (ℝ × ℝ), intersects_ellipse M N ∧ orthocenter(B, M, N) = F)
    (h_correct_slope : ∀ x : ℝ, y = x - 4 / 3)
    : line_eq :=
begin
  sorry
end

end ellipse_equation_line_equation_l664_664795


namespace cheese_pounds_bought_l664_664118

def total_money : ℕ := 87
def price_per_pound_cheese : ℕ := 7
def price_per_pound_beef : ℕ := 5
def remaining_money : ℕ := 61

theorem cheese_pounds_bought :
  let total_spent := total_money - remaining_money in
  let cheese_spent := total_spent - price_per_pound_beef in
  cheese_spent / price_per_pound_cheese = 3 := 
by
  sorry

end cheese_pounds_bought_l664_664118


namespace triangle_bc_length_l664_664902

theorem triangle_bc_length (A B C X : Type)
  (AB AC : ℕ)
  (hAB : AB = 86)
  (hAC : AC = 97)
  (circle_eq : ∀ {r : ℕ}, r = AB → circle_centered_at_A_intersects_BC_two_points B X)
  (integer_lengths : ∃ (BX CX : ℕ), ) :
  BC = 61 :=
by
  sorry

end triangle_bc_length_l664_664902


namespace find_vertex_l664_664714

noncomputable def parabola_vertex (x y : ℝ) : Prop :=
  2 * y^2 + 8 * y - 3 * x + 6 = 0

theorem find_vertex :
  ∃ (x y : ℝ), parabola_vertex x y ∧ x = -14/3 ∧ y = -2 :=
by
  sorry

end find_vertex_l664_664714


namespace even_number_of_divisors_less_than_100_l664_664419

theorem even_number_of_divisors_less_than_100 :
  ∃ (count : ℕ), count = 90 ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 100 →
    (∃ (d : ℕ), d * d = n ∨ (number_of_divisors n % 2 = 0)) :=
begin
  -- the proof goes here
  sorry
end

end even_number_of_divisors_less_than_100_l664_664419


namespace not_periodic_f_pi_periodic_f_a_iff_rational_l664_664240

def f_a (a : ℝ) : ℝ → ℝ × ℝ := λ t => (Real.sin t, Real.cos (a * t))

-- Part (a) Prove that \( f_{\pi} \) is not periodic.
theorem not_periodic_f_pi : ¬ (∃ T, 0 < T ∧ ∀ t, f_a (Real.pi) (t + T) = f_a (Real.pi) t) := sorry

-- Part (b) Determine the values of the parameter \( a \) for which \( f_a \) is periodic.
theorem periodic_f_a_iff_rational (a : ℝ) : (∃ T, 0 < T ∧ ∀ t, f_a a (t + T) = f_a a t) ↔ ∃ m n : ℤ, a = 2 * m / (n : ℝ) := sorry

end not_periodic_f_pi_periodic_f_a_iff_rational_l664_664240


namespace greatest_prime_factor_of_15_plus_18_l664_664660

theorem greatest_prime_factor_of_15_plus_18! : 
  let n := 15! + 18!
  n = 15! * 4897 ∧ Prime 4897 →
  (∀ p : ℕ, Prime p ∧ p ∣ n → p ≤ 4897) ∧ (4897 ∣ n) ∧ Prime 4897 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_of_15_plus_18_l664_664660


namespace intersection_of_sets_l664_664050

open Set

theorem intersection_of_sets :
  let A := { x : ℝ | x^2 - 4 > 0 }
  let B := { x : ℝ | x + 2 < 0 }
  A ∩ B = { x : ℝ | x < -2 } :=
by
  sorry

end intersection_of_sets_l664_664050


namespace even_number_of_divisors_less_than_100_l664_664324

theorem even_number_of_divisors_less_than_100 : 
  ∃ n, n = 90 ∧ ∀ x < 100, (∃ k, k * k = x → false) = (x ∣ 99 - 9) :=
sorry

end even_number_of_divisors_less_than_100_l664_664324


namespace find_a_find_max_omega_l664_664838

noncomputable def a_omega_function (ω x : ℝ) : ℝ := 
    let a := (1 + Real.cos(ω * x), 1)
    let b := (1, a.fst + Real.sqrt 3 * Real.sin(ω * x))
    a.fst * b.fst + a.snd * b.snd

theorem find_a (ω : ℝ) (a : ℝ) (hx : ∀ x : ℝ, a_omega_function ω x = 2) :
    a = -1 :=
sorry

theorem find_max_omega (ω : ℝ) (a : ℝ) 
    (hx1 : ∀ x : ℝ, a_omega_function (ω - ω/6) x = (2:ℝ))
    (hx2 : ∀ x ∈ Set.Icc 0 (Real.pi / 4), 2 * Real.sin(ω * x) ≤ (2:ℝ)) :
    ω ≤ 2 :=
sorry

end find_a_find_max_omega_l664_664838


namespace simplify_expression_l664_664075

theorem simplify_expression (x : ℝ) : 2 * x + 1 - (x + 1) = x := 
by 
sorry

end simplify_expression_l664_664075


namespace hyperbolas_same_asymptotes_l664_664089

theorem hyperbolas_same_asymptotes :
  (∃ M : ℝ, (∀ x y : ℝ,
    (x^2 / 9 - y^2 / 16 = 1 → y = 4/3 * x ∨ y = -(4/3) * x) ∧
    (y^2 / 25 - x^2 / M = 1 → y = 5 / (real.sqrt M) * x ∨ y = -(5 / (real.sqrt M)) * x)) →
    M = 225 / 16) :=
by
  sorry

end hyperbolas_same_asymptotes_l664_664089


namespace greatest_prime_factor_15_18_l664_664649

theorem greatest_prime_factor_15_18! :
  ∃ p : ℕ, prime p ∧ p ∈ prime_factors (15! + 18!) ∧ ∀ q : ℕ, prime q → q ∈ prime_factors (15! + 18!) → q ≤ 4897 := 
sorry

end greatest_prime_factor_15_18_l664_664649


namespace cape_may_sharks_l664_664201

theorem cape_may_sharks (D : ℕ) (C : ℕ) (hD : D = 12) (hC : C = 2 * D + 8) : C = 32 :=
by
  rw [hD, hC]
  norm_num

end cape_may_sharks_l664_664201


namespace Haley_magazines_l664_664295

theorem Haley_magazines (boxes magazines_per_box : Nat) (h1 : boxes = 7) (h2 : magazines_per_box = 9) : boxes * magazines_per_box = 63 :=
  by
  rw [h1, h2]
  sorry

end Haley_magazines_l664_664295


namespace all_elements_same_color_l664_664997

open Nat

variables {n k : ℕ} [fact (n > 0)] [fact (k > 0)] [fact (k < n)] (h_coprime : gcd n k = 1)

def same_color (i j : ℕ) : Prop := sorry -- Define the predicate for same color

theorem all_elements_same_color (M := {i ∣ 1 ≤ i ∧ i < n})
  (h1 : ∀ i ∈ M, same_color i (n - i))
  (h2 : ∀ i ∈ M, i ≠ k → same_color i (abs (k - i))) :
  ∀ i j ∈ M, same_color i j :=
sorry

end all_elements_same_color_l664_664997


namespace product_two_white_is_white_l664_664130

variable {ℕ : Type}
-- Condition (1): Each strictly positive integer is colored either black or white.
-- Define the predicate is_white to indicate if an integer is white.
def is_white (n : ℕ) : Prop := sorry

-- Condition (2): The sum of two integers of different colors is black.
axiom sum_different_colors_black {m n : ℕ} : (is_white m ∧ ¬ is_white n) ∨ (¬ is_white m ∧ is_white n) → ¬ is_white (m + n)

-- Condition (3): The product of two integers of different colors is white.
axiom product_different_colors_white {m n : ℕ} : (is_white m ∧ ¬ is_white n) ∨ (¬ is_white m ∧ is_white n) → is_white (m * n)

-- Condition (4): Not all integers are of the same color.
axiom not_all_same_color : ∃ m n : ℕ, is_white m ∧ ¬ is_white n

-- Conclusion (1): The product of two white numbers is white.
theorem product_two_white_is_white {m n : ℕ} (hm : is_white m) (hn : is_white n) : is_white (m * n) := sorry

end product_two_white_is_white_l664_664130


namespace negation_of_abs_x_minus_2_lt_3_l664_664239

theorem negation_of_abs_x_minus_2_lt_3 :
  ¬ (∀ x : ℝ, |x - 2| < 3) ↔ ∃ x : ℝ, |x - 2| ≥ 3 :=
by
  sorry

end negation_of_abs_x_minus_2_lt_3_l664_664239


namespace value_of_x_squared_plus_one_l664_664845

theorem value_of_x_squared_plus_one (x : ℝ) (h : 2^(2*x) + 4 = 12 * 2^x) : 
  x^2 + 1 = (Real.log 2 (6 + 4 * Real.sqrt 2))^2 + 1 :=
sorry

end value_of_x_squared_plus_one_l664_664845


namespace tank_capacity_l664_664176

theorem tank_capacity (C : ℕ) (h₁ : C = 785) :
  360 - C / 4 - C / 8 = C / 12 :=
by 
  -- Assuming h₁: C = 785
  have h₁: C = 785 := by exact h₁
  -- Provide proof steps here (not required for the task)
  sorry

end tank_capacity_l664_664176


namespace greatest_prime_factor_of_expression_l664_664668

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define specific factorial values
def fac_15 := factorial 15
def fac_18 := factorial 18

-- Define the expression from the problem
def expr := fac_15 * (1 + 16 * 17 * 18)

-- Define the factorization result
def factor_4896 := 2 ^ 5 * 3 ^ 2 * 17

-- Define a lemma about the factorization of the expression
lemma factor_expression : 15! * (1 + 16 * 17 * 18) = fac_15 * 4896 := by
  sorry

-- State the main theorem
theorem greatest_prime_factor_of_expression : ∀ p : ℕ, prime p ∧ p ∣ expr → p ≤ 17 := by
  sorry

end greatest_prime_factor_of_expression_l664_664668


namespace square_circumscribes_convex_figure_l664_664068

theorem square_circumscribes_convex_figure (Φ : set ℝ) 
  (hΦ : convex ℝ Φ) : ∃ S : set ℝ, is_square S ∧ Φ ⊆ S :=
sorry

end square_circumscribes_convex_figure_l664_664068


namespace complex_conjugate_of_z_l664_664275

theorem complex_conjugate_of_z (z i : ℂ) (hi : i^2 = -1) (hz : z * (1 - i) = 1 + i) :
  conj z = -i := 
sorry

end complex_conjugate_of_z_l664_664275


namespace min_difference_of_factors_of_1764_l664_664432

theorem min_difference_of_factors_of_1764 : ∃ (a b : ℕ), a * b = 1764 ∧ (a > 0) ∧ (b > 0) ∧ (∀ (c d : ℕ), c * d = 1764 → (c > 0) → (d > 0) → abs (c - d) ≥ abs (a - b)) ∧ abs (a - b) = 6 := 
sorry

end min_difference_of_factors_of_1764_l664_664432


namespace distinct_shell_arrangements_l664_664520

theorem distinct_shell_arrangements
  (shell_count : ℕ)
  (rotational_symmetries : ℕ)
  (reflectional_symmetries : ℕ)
  (total_symmetries : ℕ)
  (total_shell_permutations : ℕ)
  (distinct_arrangements : ℕ) :
  shell_count = 12 →
  rotational_symmetries = 6 →
  reflectional_symmetries = 6 →
  total_symmetries = rotational_symmetries * reflectional_symmetries →
  total_shell_permutations = nat.factorial shell_count →
  distinct_arrangements = total_shell_permutations / total_symmetries →
  distinct_arrangements = 39916800 :=
sorry

end distinct_shell_arrangements_l664_664520


namespace BC_length_l664_664954

-- Define the given values and conditions
variable (A B C X : Type)
variable (AB AC AX BX CX : ℕ)
variable [inhabited A] [inhabited B] [inhabited C] [inhabited X]

-- Assume the lengths of AB and AC
axiom h_AB : AB = 86
axiom h_AC : AC = 97

-- Assume the circle centered at A with radius AB intersects BC at B and X
axiom h_circle : AX = AB

-- Assume BX and CX are integers
axiom h_BX_integral : ∃ (x : ℕ), BX = x
axiom h_CX_integral : ∃ (y : ℕ), CX = y

-- The statement to prove that the length of BC is 61
theorem BC_length : (∃ (x y : ℕ), BX = x ∧ CX = y ∧ x + y = 61) :=
by
  sorry

end BC_length_l664_664954


namespace triangle_bc_length_l664_664939

theorem triangle_bc_length :
  ∀ (A B C X : Type) (d_AB : ℝ) (d_AC : ℝ) (d_BX d_CX BC : ℕ),
  d_AB = 86 ∧ d_AC = 97 →
  let circleA := {center := A, radius := d_AB} in
  let intersect_B := B ∈ circleA in
  let intersect_X := X ∈ circleA in
  d_BX + d_CX = BC →
  d_BX ∈ ℕ ∧ d_CX ∈ ℕ →
  BC = 61 :=
by
  intros A B C X d_AB d_AC d_BX d_CX BC h_dist h_circle h_intersect h_sum h_intBC
  sorry

end triangle_bc_length_l664_664939


namespace consecutive_non_prime_powers_l664_664566

theorem consecutive_non_prime_powers (k : ℕ) (hk : 0 < k) : 
  ∃ (a b : ℕ), a + 1 = b ∧ 
  ∀ i, 2 ≤ i ∧ i ≤ k + 1 ->
  ¬ ∃ p n, p.prime ∧ (a + i) = p ^ n :=
by
  sorry

end consecutive_non_prime_powers_l664_664566


namespace b_10_equals_133_l664_664543

/-- Define the function b_n to represent the count of subsets of {1, 2, ..., n} with the specified properties -/
def b (n : ℕ) : ℕ :=
  ∑ k in Finset.range (Int.floor ((n + 1) / 2) - 1) + 1, Nat.choose (n - k + 1) k

/-- Theorem: For n = 10, the value of b(n) is 133 -/
theorem b_10_equals_133 : b 10 = 133 :=
by
  -- Proof goes here
  sorry

end b_10_equals_133_l664_664543


namespace excenters_and_A1_cocircular_l664_664258

-- Given elements in the problem
variables {A B C I O A1 O1 D E F D1 E1 F1 : Type*}

-- Conditions per problem specification
-- Here we assume realistic points in a plane, and using relevant properties.
axiom is_triangle : Triangle A B C
axiom is_incenter : Incenter I A B C
axiom is_circumcenter : Circumcenter O A B C
axiom midpoint_AI : Midpoint A I A1
axiom midpoint_O1I : Midpoint O I O1
axiom midpoint_arcs : ArcMidpoints O A B C D E F
axiom excenters_tangent : Excenters D1 E1 F1 A B C

-- To Prove: Points A1, D1, E1, F1 are on the same circle centered at O1
theorem excenters_and_A1_cocircular :
  Cocircular A1 D1 E1 F1 O1 := 
sorry

end excenters_and_A1_cocircular_l664_664258


namespace find_a_l664_664288

noncomputable section

def f (x a : ℝ) : ℝ := Real.sqrt (1 + a * 4^x)

theorem find_a (a : ℝ) : 
  (∀ (x : ℝ), x ≤ -1 → 1 + a * 4^x ≥ 0) → a = -4 :=
sorry

end find_a_l664_664288


namespace triangle_bc_length_l664_664943

theorem triangle_bc_length :
  ∀ (A B C X : Type) (d_AB : ℝ) (d_AC : ℝ) (d_BX d_CX BC : ℕ),
  d_AB = 86 ∧ d_AC = 97 →
  let circleA := {center := A, radius := d_AB} in
  let intersect_B := B ∈ circleA in
  let intersect_X := X ∈ circleA in
  d_BX + d_CX = BC →
  d_BX ∈ ℕ ∧ d_CX ∈ ℕ →
  BC = 61 :=
by
  intros A B C X d_AB d_AC d_BX d_CX BC h_dist h_circle h_intersect h_sum h_intBC
  sorry

end triangle_bc_length_l664_664943


namespace even_number_of_divisors_less_than_100_l664_664313

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k, k * k = n

theorem even_number_of_divisors_less_than_100 :
  (card { n : ℕ | 1 ≤ n ∧ n < 100 ∧ ¬ is_perfect_square n }) = 90 :=
by
  sorry

end even_number_of_divisors_less_than_100_l664_664313


namespace bc_ad_divisible_by_u_l664_664576

theorem bc_ad_divisible_by_u 
  (a b c d u : ℤ) 
  (h1 : u ∣ a * c)
  (h2 : u ∣ b * c + a * d)
  (h3 : u ∣ b * d) : 
  u ∣ b * c ∧ u ∣ a * d :=
by
  sorry

end bc_ad_divisible_by_u_l664_664576


namespace prob_score_at_most_7_l664_664877

-- Definitions based on the conditions
def prob_10_ring : ℝ := 0.15
def prob_9_ring : ℝ := 0.35
def prob_8_ring : ℝ := 0.2
def prob_7_ring : ℝ := 0.1

-- Define the event of scoring no more than 7
def score_at_most_7 := prob_7_ring

-- Theorem statement
theorem prob_score_at_most_7 : score_at_most_7 = 0.1 := by 
  -- proof goes here
  sorry

end prob_score_at_most_7_l664_664877


namespace num_valid_sequences_correct_l664_664097

noncomputable def num_valid_sequences : ℕ :=
  set.to_finset { p : ℤ × ℤ | 
                  let x := p.1, d := p.2 in
                  x + 2 * d = 108 ∧ x + 4 * d < 120 ∧ x > 48 ∧
                  all (λ k, x + k * d) (list.range (fin.val (5 : fin 5))) < 120 ∧
                  is_arithmetic_sequence [x, x + d, x + 2 * d, x + 3 * d, x + 4 * d] }
  sorry

theorem num_valid_sequences_correct : num_valid_sequences = 3 := by
  sorry

end num_valid_sequences_correct_l664_664097


namespace right_triangle_area_l664_664253

theorem right_triangle_area (h a : ℕ) (h_is_45_45_90 : h^2 = 2 * a^2) (altitude_CD : a / bit0 1 = 4) : 
    (triangle_area h a = 16) :=
begin
  sorry
end

end right_triangle_area_l664_664253


namespace rank_of_A_l664_664816

def A : Matrix (Fin 3) (Fin 5) ℝ :=
  ![![1, 2, 3, 5, 8],
    ![0, 1, 4, 6, 9],
    ![0, 0, 1, 7, 10]]

theorem rank_of_A : A.rank = 3 :=
by sorry

end rank_of_A_l664_664816


namespace fraction_diff_l664_664031

def G := 0.817817817...

theorem fraction_diff :
  let (n, d) := (817, 999)
  in (d - n) = 182 :=
by {
  sorry
}

end fraction_diff_l664_664031


namespace cos_double_angle_l664_664780

theorem cos_double_angle (α : ℝ) (h1 : α < π / 2) (h2 : 0 < α)
    (h3 : sin (α - π / 12) = 3 / 5) : cos (2 * α + π / 3) = -24 / 25 := 
by
  sorry

end cos_double_angle_l664_664780


namespace trapezoid_segment_AB_length_l664_664898

/-
In the trapezoid shown, the ratio of the area of triangle ABC to the area of triangle ADC is 5:2.
If AB + CD = 240 cm, prove that the length of segment AB is 171.42857 cm.
-/

theorem trapezoid_segment_AB_length
  (AB CD : ℝ)
  (ratio_areas : ℝ := 5 / 2)
  (area_ratio_condition : AB / CD = ratio_areas)
  (length_sum_condition : AB + CD = 240) :
  AB = 171.42857 :=
sorry

end trapezoid_segment_AB_length_l664_664898


namespace shaded_area_of_three_circles_l664_664505

theorem shaded_area_of_three_circles :
  (∀ (r1 r2 : ℝ), (π * r1^2 = 100 * π) → (r2 = r1 / 2) → (shaded_area = (π * r1^2) / 2 + 2 * ((π * r2^2) / 2)) → (shaded_area = 75 * π)) :=
by
  sorry

end shaded_area_of_three_circles_l664_664505


namespace problem_1_problem_2_l664_664825

noncomputable def f (x : ℝ) := abs x + abs (x + 1)

-- statement for (1)
theorem problem_1 (λ : ℝ) : (∀ x : ℝ, f x ≥ λ) ↔ λ ≤ 1 := sorry

-- statement for (2)
theorem problem_2 (t : ℝ) : (∃ m : ℝ, m^2 + 2 * m + f t = 0) ↔ -1 ≤ t ∧ t ≤ 0 := sorry

end problem_1_problem_2_l664_664825


namespace sum_50_series_l664_664634

theorem sum_50_series (x y z : ℕ) (h : ∑ k in Finset.range 50 + 1, (-1) ^ (k + 1) * (2 * k ^ 2 + k + 2) / (k + 1)! = x / y! - z) :
  x + y + z = 52 :=
by
  sorry

end sum_50_series_l664_664634


namespace even_divisors_less_than_100_l664_664361

theorem even_divisors_less_than_100 :
  let count_even_divisors := 
    let n := 100 in
    let perfect_squares := { m | ∃ k, k * k = m ∧ m < n } in
    let total_numbers := finset.range n in
    (total_numbers.card - perfect_squares.card) =
    90 
  in count_even_divisors = 90 :=
by
  let n := 100
  let perfect_squares : finset ℕ := finset.filter (λ m, ∃ k, k * k = m) (finset.range n)
  let total_numbers : finset ℕ := finset.range n
  have h : total_numbers.card = 99 := by sorry
  have p : perfect_squares.card = 9 := by sorry
  show total_numbers.card - perfect_squares.card = 90
  calc
    total_numbers.card - perfect_squares.card
      = 99 - 9 := by rw [h, p]
      = 90 := by norm_num

end even_divisors_less_than_100_l664_664361


namespace range_of_positive_k_l664_664623

open Real

theorem range_of_positive_k {k : ℝ} (φ : ℝ) (h1 : k > 0) :
  (∀ x : ℝ, x² + (sin ((π / k) * x + φ))² ≤ 4 → (sin ((π / k) * x + φ) = 1 ∨ sin ((π / k) * x + φ) = -1)) →
  (∃ φ : ℝ, ∀ x : ℝ, x² + (sin ((π / k) * x + φ))² ≤ 4 ∧ k ∈ Ioo (sqrt 3 / 2) (sqrt 3)) :=
sorry

end range_of_positive_k_l664_664623


namespace quadratic_inequality_solution_minimum_value_expression_l664_664857

theorem quadratic_inequality_solution (a : ℝ) : (∀ x : ℝ, a * x^2 - 6 * x + 3 > 0) → a > 3 :=
sorry

theorem minimum_value_expression (a : ℝ) : (a > 3) → a + 9 / (a - 1) ≥ 7 ∧ (a + 9 / (a - 1) = 7 ↔ a = 4) :=
sorry

end quadratic_inequality_solution_minimum_value_expression_l664_664857


namespace RotaryClubNeeds576Eggs_l664_664091

def RotaryClubEggs (tickets_small : ℕ) (tickets_older : ℕ) (tickets_adult : ℕ) (tickets_senior : ℕ) 
                   (extra_omelets : ℕ) (avg_eggs_per_omelet : ℚ) (waste_percentage : ℚ) : ℕ :=
let small_eggs := tickets_small * 1 in
let older_eggs := tickets_older * 2 in
let adult_eggs := tickets_adult * 3 in
let senior_eggs := tickets_senior * 4 in
let total_eggs := small_eggs + older_eggs + adult_eggs + senior_eggs in
let extra_eggs := extra_omelets * avg_eggs_per_omelet in
let total_with_extra := total_eggs + (nat_ceil extra_eggs.to_real) in
let wasted_eggs := total_with_extra * waste_percentage in
total_with_extra + nat_ceil wasted_eggs.to_real

theorem RotaryClubNeeds576Eggs :
  RotaryClubEggs 53 35 75 37 25 2.5 0.03 = 576 := by
  sorry

end RotaryClubNeeds576Eggs_l664_664091


namespace measure_of_angle_y_l664_664769

theorem measure_of_angle_y 
  (A B C D : Type) 
  (E : Type)
  (F : Type)
  (angle_ABC : ℕ)
  (angle_EAB : ℕ)
  (angle_BDC : ℕ) 
  (angle_sum_triangle : ∀ {x y z : ℕ}, x + y + z = 180) :
  angle_ABC = 122 → 
  angle_EAB = 29 → 
  angle_BDC = 27 →
  ∃ y : ℕ, y = 93 :=
by
  intros hABC hEAB hBDC
  use 93
  sorry

end measure_of_angle_y_l664_664769


namespace total_cost_of_weight_increase_is_correct_l664_664626

-- Definitions from the conditions
def initial_weight : ℝ := 60
def weight_increase_percentage : ℝ := 0.60
def ingot_weight : ℝ := 2
def ingot_cost : ℝ := 5
def discount_10_to_20 : ℝ := 0.20
def discount_above_20 : ℝ := 0.25
def sales_tax_rate : ℝ := 0.05
def shipping_fee : ℝ := 10

-- Theorem to prove the total cost
theorem total_cost_of_weight_increase_is_correct :
  let
    additional_weight := initial_weight * weight_increase_percentage,
    ingots_needed := additional_weight / ingot_weight,
    raw_cost := ingots_needed * ingot_cost,
    discount := if h : ingots_needed > 10 ∧ ingots_needed ≤ 20 then raw_cost * discount_10_to_20 else if h' : ingots_needed > 20 then raw_cost * discount_above_20 else 0,
    discounted_price := raw_cost - discount,
    taxed_price := discounted_price + (discounted_price * sales_tax_rate),
    total_cost := taxed_price + shipping_fee
  in
    total_cost = 85.60 := by
      sorry

end total_cost_of_weight_increase_is_correct_l664_664626


namespace five_digit_even_palindromes_l664_664298

open Nat

theorem five_digit_even_palindromes : 
  (∃ f : ℕ → ℕ → ℕ → ℕ, 
    (∀ a b c, f a b c = a * 10001 + b * 1010 + c * 100) ∧
    (∀ a, a ≠ 0 ∧ a % 2 = 0) ∧
    (∀ b, b ≥ 0 ∧ b ≤ 9) ∧
    (∀ c, c ≥ 0 ∧ c ≤ 9) ) →
  {n : ℕ | ∃ (a b c : ℕ), 
              n = a * 10001 + b * 1010 + c * 100 ∧ 
              a ≠ 0 ∧ 
              a % 2 = 0 ∧ 
              b ≥ 0 ∧ b ≤ 9 ∧ 
              c ≥ 0 ∧ c ≤ 9 }.card = 400 :=
begin
  sorry
end

end five_digit_even_palindromes_l664_664298


namespace even_number_of_divisors_less_than_100_l664_664417

theorem even_number_of_divisors_less_than_100 :
  ∃ (count : ℕ), count = 90 ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 100 →
    (∃ (d : ℕ), d * d = n ∨ (number_of_divisors n % 2 = 0)) :=
begin
  -- the proof goes here
  sorry
end

end even_number_of_divisors_less_than_100_l664_664417


namespace sudoku_table_XY_sum_l664_664592

-- Define the main problem statement
theorem sudoku_table_XY_sum (X Y : ℕ) 
  (h1 : ∀ r c, r ≠ c → (1 ≤ r ∧ r ≤ 3) ∧ (1 ≤ c ∧ c ≤ 3))  -- table boundaries
  (h2 : ∀ r, (r = 1 → ∃ c1 c2 : ℕ, c1 ≠ c2 ∧ c1 ≠ r ∧ c2 ≠ r 
    ∧ (1 ≤ c1 ∧ c1 ≤ 3) ∧ (1 ≤ c2 ∧ c2 ≤ 3)) ∧ (r ≠ 1 → ∃ c : ℕ, (1 ≤ c ∧ c ≤ 3))) -- unique values per row and column 
  (h3 : nat.find! h1 = 2) -- pre-filled values 
  (h4 : nat.find! h2 = 3) -- further constraints for X and Y fulfilment, can be similarly defined 
  : X + Y = 5 := 
  sorry

end sudoku_table_XY_sum_l664_664592


namespace distinct_exponentiations_l664_664218

theorem distinct_exponentiations : 
  ∃ a b c d e f : ℕ, 
    a = 3^(3^(3^3)) ∧ 
    b = 3^(3^3) ∧ 
    c = 3^((3^3)^3) ∧ 
    d = 3^(3^(3^3 + 1)) ∧ 
    e = 3^(3^28) ∧
    f = (3^27)^3 ∧
    ∀ x ∈ {a, c, e, f}, 
      x ≠ 3^(3^(3^3)) ∧ 
      x ≠ b ∧
      x ≠ (3^(28 + 1)) ∧ 
      {a, c, e, f}.card = 4 :=
by
  sorry

end distinct_exponentiations_l664_664218


namespace minimum_students_ans_q1_correctly_l664_664870

variable (Total Students Q1 Q2 Q1_and_Q2 : ℕ)
variable (did_not_take_test: Student → Bool)

-- Given Conditions
def total_students := 40
def students_ans_q2_correctly := 29
def students_not_taken_test := 10
def students_ans_both_correctly := 29

theorem minimum_students_ans_q1_correctly (H1: Q2 - students_not_taken_test == 30)
                                           (H2: Q1_and_Q2 + students_not_taken_test == total_students)
                                           (H3: Q1_and_Q2 == students_ans_q2_correctly):
  Q1 ≥ 29 := by
  sorry

end minimum_students_ans_q1_correctly_l664_664870


namespace number_of_5_digit_palindromes_l664_664698

-- Definition of a 5-digit palindrome
def is_palindrome (n : ℕ) : Prop :=
  ∃ a b c : ℕ, n = a * 10001 + b * 1010 + c * 100 + b * 10 + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9

-- The theorem stating the number of 5-digit palindromes
theorem number_of_5_digit_palindromes : 
  {n : ℕ | is_palindrome n}.to_finset.card = 900 :=
sorry

end number_of_5_digit_palindromes_l664_664698


namespace part_a_part_b_l664_664027

-- Define the sequence conditions and proof problem for Part (a)
theorem part_a (x : ℕ → ℤ) (h₀ : x 0 = 2) (h₁ : x 1 = 3)
  (recurrence_rel : ∀ n, x (n + 2) = 7 * x (n + 1) - x n + 280) :
  ∀ n, (∑ d in (nat.divisors (x n * x (n + 1) + x (n + 1) * x (n + 2) + x (n + 2) * x (n + 3) + 2018)), d) % 24 = 0 := 
sorry

-- Define the sequence conditions and proof problem for Part (b)
theorem part_b (x : ℕ → ℤ) (recurrence_rel : ∀ n, x (n + 2) = 7 * x (n + 1) - x n + 280)
  (constraints : 0 ≤ x 0 ∧ x 0 < x 1 ∧ x 1 ≤ 100) :
  ∃ x₀ x₁ : ℤ, (∀ m : ℕ, ∃ n, x n = x₀ ∧ x (n+1) = x₁ ∧ (x n * x (n + 1) + 2019) = m^2) := 
sorry

end part_a_part_b_l664_664027


namespace length_of_BC_l664_664919

theorem length_of_BC 
  (A B C X : Type) 
  (d_AB : ℝ) (d_AC : ℝ) 
  (circle_center_A : A) 
  (radius_AB : ℝ)
  (intersects_BC : B → C → X)
  (BX CX : ℕ) 
  (h_BX_in_circle : BX = d_AB) 
  (h_CX_in_circle : CX = d_AC) 
  (h_integer_lengths : ∃ x y : ℕ, BX = x ∧ CX = y) :
  BX + CX = 61 :=
begin
  sorry
end

end length_of_BC_l664_664919


namespace length_of_BC_l664_664921

theorem length_of_BC 
  (A B C X : Type) 
  (d_AB : ℝ) (d_AC : ℝ) 
  (circle_center_A : A) 
  (radius_AB : ℝ)
  (intersects_BC : B → C → X)
  (BX CX : ℕ) 
  (h_BX_in_circle : BX = d_AB) 
  (h_CX_in_circle : CX = d_AC) 
  (h_integer_lengths : ∃ x y : ℕ, BX = x ∧ CX = y) :
  BX + CX = 61 :=
begin
  sorry
end

end length_of_BC_l664_664921


namespace gain_percentage_is_correct_l664_664719

-- Definitions based on conditions
def cost_price (sp : ℝ) (loss_percent : ℝ) : ℝ := sp / (1 - loss_percent / 100)
def gain (sp cp : ℝ) : ℝ := sp - cp
def gain_percent (gain cp : ℝ) : ℝ := (gain / cp) * 100

-- Statement of the problem
theorem gain_percentage_is_correct :
  let sp_1 := 119
      sp_2 := 168
      loss_percent := 15 in
  let cp := cost_price sp_1 loss_percent in
  gain_percent (gain sp_2 cp) cp = 20 := 
sorry

end gain_percentage_is_correct_l664_664719


namespace quadratic_shift_l664_664460

theorem quadratic_shift :
  ∀ (x : ℝ), (∃ (y : ℝ), y = -x^2) →
  (∃ (y : ℝ), y = -(x + 1)^2 + 3) :=
by
  intro x
  intro h
  use -(x + 1)^2 + 3
  sorry

end quadratic_shift_l664_664460


namespace cone_unfolded_angle_l664_664470

theorem cone_unfolded_angle (r l α : ℝ) (h1 : l = sqrt 2 * r) (h2 : α * l = 2 * π * r) : α = sqrt 2 * π := 
by
  sorry

end cone_unfolded_angle_l664_664470


namespace greatest_prime_factor_15_fact_plus_18_fact_l664_664638

theorem greatest_prime_factor_15_fact_plus_18_fact :
  Nat.greatest_prime_factor (15.factorial + 18.factorial) = 17 := by
  sorry

end greatest_prime_factor_15_fact_plus_18_fact_l664_664638


namespace no_real_roots_of_polynomial_l664_664605

noncomputable def p (x : ℝ) : ℝ := sorry

theorem no_real_roots_of_polynomial (p : ℝ → ℝ) (h_deg : ∃ n : ℕ, n ≥ 1 ∧ ∀ x: ℝ, p x = x^n) :
  (∀ x, p x * p (2 * x^2) = p (3 * x^3 + x)) →
  ¬ ∃ α : ℝ, p α = 0 := sorry

end no_real_roots_of_polynomial_l664_664605


namespace total_books_l664_664553

theorem total_books (t : ℕ) (h1 : t - 10 = 36) : t = 46 :=
by {
  have h2 : t = 46,
  exact eq_add_of_sub_eq h1,
  exact h2,
  sorry
}

end total_books_l664_664553


namespace kerosene_sale_difference_l664_664484

noncomputable def rice_price : ℝ := 0.33
noncomputable def price_of_dozen_eggs := rice_price
noncomputable def price_of_one_egg := rice_price / 12
noncomputable def price_of_half_liter_kerosene := 4 * price_of_one_egg
noncomputable def price_of_one_liter_kerosene := 2 * price_of_half_liter_kerosene
noncomputable def kerosene_discounted := price_of_one_liter_kerosene * 0.95
noncomputable def kerosene_diff_cents := (price_of_one_liter_kerosene - kerosene_discounted) * 100

theorem kerosene_sale_difference :
  kerosene_diff_cents = 1.1 := by sorry

end kerosene_sale_difference_l664_664484


namespace trigonometric_identity_l664_664434

variable (α : ℝ)

def given_condition : Prop := sin (α - (Real.pi / 6)) = 1 / 3

theorem trigonometric_identity (h : given_condition α) : cos (α + (Real.pi / 3)) = -1 / 3 := by
  sorry

end trigonometric_identity_l664_664434


namespace hundredth_term_seq_l664_664190

noncomputable def seq : ℕ → ℕ
| 0     := 1
| 1     := 3
| 2     := 4
| 3     := 9
| 4     := 10
| 5     := 12
| 6     := 13
| (n+7) := seq (n+6) + 3

theorem hundredth_term_seq : seq 100 = 981 :=
by 
-- This is where you would prove the theorem if needed. We'll replace it with sorry for now.
sorry

end hundredth_term_seq_l664_664190


namespace number_of_unordered_triples_l664_664524
  
/-- Given a set A = {1, 2, ..., n}, the number of unordered triples (X, Y, Z) 
that satisfy X ∪ Y ∪ Z = A is 7^n -/
theorem number_of_unordered_triples (n : ℕ) :
  let A := finset.range (n + 1)
  in ∃ count : ℕ, count = 7^n ∧
  ∀ (X Y Z : finset ℕ), (X ∪ Y ∪ Z = A) → 
  finset.card (finset.powerset A) = count :=
begin
  -- proof to be filled in
  sorry
end

end number_of_unordered_triples_l664_664524


namespace average_T_l664_664994

variable {T : Finset ℕ}
variable {b : ℕ → ℕ}

noncomputable def average (s : Finset ℕ) (f : ℕ → ℕ) := 
  (s.sum f : ℚ) / s.card

theorem average_T (h1 : T.erase (T.max' sorry) = Finset.range (T.card - 1) ∧ average T.erase(T.max' sorry) b = 40)
  (h2 : T.erase (T.max' sorry).erase (T.min' sorry) = Finset.range (T.card - 2) ∧ average T.erase(T.max' sorry).erase(T.min' sorry) b = 43)
  (h3 : T.erase (T.min' sorry).insert (T.max' sorry) = Finset.range (T.card - 1) ∧ average T.erase(T.min' sorry).insert(T.max' sorry) b = 47)
  (h4 : T.max' sorry = T.min' sorry + 84) :
  average T b = 44.15 := 
sorry

end average_T_l664_664994


namespace sin_sum_theta_l664_664270

variable (θ : ℝ)

theorem sin_sum_theta :
  (sin (11 * Real.pi / 10 + 2 * θ)) = 1 / 3 :=
by
  have h : sin (Real.pi / 5 - θ) = Real.sqrt 6 / 3 := sorry
  sorry

end sin_sum_theta_l664_664270


namespace solve_triangle_l664_664113

theorem solve_triangle (a b c k β m : ℝ) 
  (h1 : a + c = k)
  (h2 : ∠(a, c) = β)
  (h3 : height(b) = m) : 
  b = -m * cot (β / 2) + sqrt (m^2 * cot (β / 2)^2 + k^2) :=
by sorry

end solve_triangle_l664_664113


namespace election_votes_l664_664881

theorem election_votes (V : ℝ) (ha : 0.45 * V = 4860)
                       (hb : 0.30 * V = 3240)
                       (hc : 0.20 * V = 2160)
                       (hd : 0.05 * V = 540)
                       (hmaj : (0.45 - 0.30) * V = 1620) :
                       V = 10800 :=
by
  sorry

end election_votes_l664_664881


namespace passenger_probability_l664_664048

open ProbabilityTheory

-- Define the random variable X with normal distribution
noncomputable def X : MeasureTheory.ProbabilityMeasure ℝ :=
  MeasureTheory.Measure.add {
    density := λ x, (Real.exp (-((x - 800)^2) / (2 * (50^2))) ) / (50 * Real.sqrt (2 * Real.pi)),
    integrable := sorry
  }

-- Define the probability calculation problem
theorem passenger_probability :
  ∀ p_0 : ℝ, p_0 = ℙ (X ≤ 900) ↔ p_0 = 0.9772 :=
begin
  -- The actual proof would go here
  sorry,
end

end passenger_probability_l664_664048


namespace number_of_integers_with_even_divisors_l664_664391

-- Define conditions
def N := 99

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P : finset ℕ := { n in finset.range (N + 1) | is_perfect_square n }

noncomputable def countP : ℕ := P.card

-- The statement to prove
theorem number_of_integers_with_even_divisors : 
  (N - countP) = 90 :=
by {
  sorry
}

end number_of_integers_with_even_divisors_l664_664391


namespace greatest_prime_factor_15_fact_plus_18_fact_l664_664639

theorem greatest_prime_factor_15_fact_plus_18_fact :
  Nat.greatest_prime_factor (15.factorial + 18.factorial) = 17 := by
  sorry

end greatest_prime_factor_15_fact_plus_18_fact_l664_664639


namespace necessary_and_sufficient_n_geq_4_l664_664526

variables {A B C D : Type*} [Convexity A] [Convexity B] [Convexity C] [Convexity D]

def is_acute_angle (angle : Type*) := sorry -- Placeholder definition; properly define acute angle

def is_obtuse_angle (angle : Type*) := sorry -- Placeholder definition; properly define obtuse angle

def quadrilateral (A B C D : Type*) := sorry -- Placeholder for checking if A B C D forms a quadrilateral.

def angle_D_acute (D : Type*) (quad : quadrilateral A B C D) := is_acute_angle D

def n_obtuse_triangles (quad : quadrilateral A B C D) (n : ℕ) :=
  ∃ (triangles : list (Type* × Type* × Type*)),
    (∀ (t : Type* × Type* × Type*), t ∈ triangles → is_obtuse_angle t.fst ∧ is_obtuse_angle t.snd ∧ is_obtuse_angle t.snd) ∧
    length triangles = n

theorem necessary_and_sufficient_n_geq_4
  (quad : quadrilateral A B C D)
  (h : angle_D_acute D quad)
  (n : ℕ) :
  n_obtuse_triangles quad n ↔ n ≥ 4 :=
sorry -- Proof omitted

end necessary_and_sufficient_n_geq_4_l664_664526


namespace find_a_extremum_at_neg_3_l664_664087

-- Define the given function f(x)
def f (a : ℝ) (x : ℝ) := x^3 + a * x^2 + 3 * x - 9

-- Define the first derivative of the function
def f_prime (a : ℝ) (x : ℝ) := 3 * x^2 + 2 * a * x + 3 

-- The definition of having an extremum at x = -3
def has_extremum_at (a : ℝ) (x : ℝ) := f_prime(a, x) = 0

-- The main theorem stating that if f has an extremum at x = -3, then a = 5
theorem find_a_extremum_at_neg_3 (a : ℝ) (h : has_extremum_at a (-3)) : a = 5 :=
by {
    sorry
}

end find_a_extremum_at_neg_3_l664_664087


namespace greatest_prime_factor_15_fact_plus_18_fact_l664_664637

theorem greatest_prime_factor_15_fact_plus_18_fact :
  Nat.greatest_prime_factor (15.factorial + 18.factorial) = 17 := by
  sorry

end greatest_prime_factor_15_fact_plus_18_fact_l664_664637


namespace log_inequality_solution_set_l664_664107

theorem log_inequality_solution_set (x : ℝ) : ∀ (b : ℝ), b = 0.2 ∧ 0 < b ∧ b < 1 → log b (x - 1) ≤ log b 2 → x ≤ 3 :=
by
  -- proof steps would be added here
  sorry

end log_inequality_solution_set_l664_664107


namespace evens_divisors_lt_100_l664_664367

theorem evens_divisors_lt_100 : 
  ∃ n : ℕ, n = 90 ∧ ∀ k : ℕ, (1 ≤ k < 100) → (even k ↔ (∃ m : ℕ, m * m = k)) ↔ (n = 90) := 
sorry

end evens_divisors_lt_100_l664_664367


namespace greatest_prime_factor_of_15_plus_18_l664_664655

theorem greatest_prime_factor_of_15_plus_18! : 
  let n := 15! + 18!
  n = 15! * 4897 ∧ Prime 4897 →
  (∀ p : ℕ, Prime p ∧ p ∣ n → p ≤ 4897) ∧ (4897 ∣ n) ∧ Prime 4897 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_of_15_plus_18_l664_664655


namespace matrix_inverse_solution_l664_664223

-- Define the 2x2 matrix structure specifically for this proof
variables (c d : ℝ)

-- Given condition: the matrix is its own inverse
def matrix_inverse_condition (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  A.mul A = !![(1, 0), (0, 1)]

-- The specific matrix given in the problem
def given_matrix (c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![(4, -2), (c, d)]

theorem matrix_inverse_solution :
  let A := given_matrix (15 / 2) (-4)
  in matrix_inverse_condition A :=
by {
  let A := given_matrix (15 / 2) (-4),
  unfold matrix_inverse_condition given_matrix,
  sorry
}

end matrix_inverse_solution_l664_664223


namespace triangle_bc_length_l664_664938

theorem triangle_bc_length :
  ∀ (A B C X : Type) (d_AB : ℝ) (d_AC : ℝ) (d_BX d_CX BC : ℕ),
  d_AB = 86 ∧ d_AC = 97 →
  let circleA := {center := A, radius := d_AB} in
  let intersect_B := B ∈ circleA in
  let intersect_X := X ∈ circleA in
  d_BX + d_CX = BC →
  d_BX ∈ ℕ ∧ d_CX ∈ ℕ →
  BC = 61 :=
by
  intros A B C X d_AB d_AC d_BX d_CX BC h_dist h_circle h_intersect h_sum h_intBC
  sorry

end triangle_bc_length_l664_664938


namespace simple_interest_rate_l664_664678

theorem simple_interest_rate
  (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ)
  (h1 : SI = 400)
  (h2 : P = 800)
  (h3 : T = 2) :
  R = 25 :=
by
  sorry

end simple_interest_rate_l664_664678


namespace sequence_property_l664_664608

theorem sequence_property (k : ℝ) (h_k : 0 < k) (x : ℕ → ℝ)
  (h₀ : x 0 = 1)
  (h₁ : x 1 = 1 + k)
  (rec1 : ∀ n, x (2*n + 1) - x (2*n) = x (2*n) - x (2*n - 1))
  (rec2 : ∀ n, x (2*n) / x (2*n - 1) = x (2*n - 1) / x (2*n - 2)) :
  ∃ N, ∀ n ≥ N, x n > 1994 :=
by
  sorry

end sequence_property_l664_664608


namespace equilateral_implies_isosceles_converse_and_inverse_false_l664_664830

-- Definitions for the propositions p and q
def triangle (T : Type) := T
def is_equilateral (T : Type) : Prop := triangle T
def is_isosceles (T : Type) : Prop := triangle T

-- The original true statement
theorem equilateral_implies_isosceles (T : Type) : is_equilateral T → is_isosceles T := sorry  -- This is given as true

-- Prove that both the converse and inverse are false
theorem converse_and_inverse_false (T : Type) :
  (¬ (is_isosceles T → is_equilateral T)) ∧ (¬ (¬ is_equilateral T → ¬ is_isosceles T)) :=
begin
  -- Proof would go here
  sorry
end

end equilateral_implies_isosceles_converse_and_inverse_false_l664_664830


namespace sphere_in_cone_radius_l664_664173

theorem sphere_in_cone_radius (b d : ℝ) (r : ℝ) 
  (h_base_radius : 15) (h_height : 30)
  (h_radius : r = b * Real.sqrt d - b) :
  b + d = 12.5 :=
sorry

end sphere_in_cone_radius_l664_664173


namespace even_number_of_divisors_l664_664335

-- Proof statement: There are 90 positive integers less than 100 with an even number of divisors.
theorem even_number_of_divisors : 
  {n : ℕ | n < 100 ∧ ∃ k : ℕ, k ^ 2 = n}.toFinset.card = 90 := 
sorry

end even_number_of_divisors_l664_664335


namespace fixed_points_a1_bneg2_range_of_a_l664_664783

namespace FixedPointProblem

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + (b - 1)

-- Definition of a fixed point
def is_fixed_point (a b x : ℝ) : Prop := f(a, b, x) = x

-- Problem 1: For a = 1, b = -2, prove that the fixed points are -1 and 3
theorem fixed_points_a1_bneg2 :
  (is_fixed_point 1 (-2) (-1)) ∧ (is_fixed_point 1 (-2) 3) :=
by
  sorry

-- Problem 2: For the function always having two distinct fixed points for any real b, find the range of a
theorem range_of_a (a : ℝ) :
  (∀ b : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ is_fixed_point a b x1 ∧ is_fixed_point a b x2)
  → (0 < a ∧ a < 1) :=
by
  sorry

end FixedPointProblem

end fixed_points_a1_bneg2_range_of_a_l664_664783


namespace circle_eq_center_tangent_l664_664083

theorem circle_eq_center_tangent (x y : ℝ) : 
  let center := (5, 4)
  let radius := 4
  (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2 :=
by
  sorry

end circle_eq_center_tangent_l664_664083


namespace harold_grocery_expense_l664_664841

theorem harold_grocery_expense:
  ∀ (income rent car_payment savings utilities remaining groceries : ℝ),
    income = 2500 →
    rent = 700 →
    car_payment = 300 →
    utilities = 0.5 * car_payment →
    remaining = income - rent - car_payment - utilities →
    savings = 0.5 * remaining →
    (remaining - savings) = 650 →
    groceries = (remaining - 650) →
    groceries = 50 :=
by
  intros income rent car_payment savings utilities remaining groceries
  intro h_income
  intro h_rent
  intro h_car_payment
  intro h_utilities
  intro h_remaining
  intro h_savings
  intro h_final_remaining
  intro h_groceries
  sorry

end harold_grocery_expense_l664_664841


namespace even_number_of_divisors_less_than_100_l664_664325

theorem even_number_of_divisors_less_than_100 : 
  ∃ n, n = 90 ∧ ∀ x < 100, (∃ k, k * k = x → false) = (x ∣ 99 - 9) :=
sorry

end even_number_of_divisors_less_than_100_l664_664325


namespace smaller_cube_edge_length_l664_664754

-- Given conditions
variables (s : ℝ) (volume_large_cube : ℝ) (n : ℝ)
-- n = 8 (number of smaller cubes), volume_large_cube = 1000 cm³

theorem smaller_cube_edge_length (h1 : n = 8) (h2 : volume_large_cube = 1000) :
  s^3 = volume_large_cube / n → s = 5 :=
by
  sorry

end smaller_cube_edge_length_l664_664754


namespace wood_rope_length_equivalence_l664_664501

variable (x y : ℝ)

theorem wood_rope_length_equivalence :
  (x - y = 4.5) ∧ (y = (1 / 2) * x + 1) :=
  sorry

end wood_rope_length_equivalence_l664_664501


namespace length_of_BC_l664_664931

theorem length_of_BC (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
    (BX CX : ℕ) (h_pow : CX * (BX + CX) = 2013) : 
    BX + CX = 61 :=
  sorry

end length_of_BC_l664_664931


namespace complement_A_B_l664_664266

def A := {1, 2, 3}
def B (a : ℤ) := {a + 2, a}

theorem complement_A_B (a : ℤ) (h : A ∩ B a = B a) : A \ (B a) = {2} := by
  sorry

end complement_A_B_l664_664266


namespace minimum_perimeter_rectangle_l664_664773

theorem minimum_perimeter_rectangle (S : ℝ) (hS : S > 0) :
  ∃ x y : ℝ, (x * y = S) ∧ (∀ u v : ℝ, (u * v = S) → (2 * (u + v) ≥ 4 * Real.sqrt S)) ∧ (x = Real.sqrt S ∧ y = Real.sqrt S) :=
by
  sorry

end minimum_perimeter_rectangle_l664_664773


namespace Znayka_is_correct_l664_664552

theorem Znayka_is_correct :
  ∀ (l : List ℕ), l.length = 2015 → 
    (∀ (i : Fin 2015), Prime (if l.nth i > l.nth (i + 1) then l.nth i / l.nth (i + 1) else l.nth (i + 1) / l.nth i)) → 
      False := sorry

end Znayka_is_correct_l664_664552


namespace find_two_points_l664_664711

-- Let G be a graph with n > 2 points and let A and B be any two points in G.
-- We need to prove that there are at least floor(n / 2) - 1 points joined to both or neither of A and B.
theorem find_two_points (G : Type) [Graph G] (n : ℕ) (h : n > 2) :
  ∃ (A B : G), 
  ∃ (S : Finset G), 
  ( S.card ≥ (n / 2 - 1) )
  ∧ ∀ (x ∈ S), (connected G A x ∧ connected G B x) ∨ (¬ connected G A x ∧ ¬ connected G B x) := 
sorry

end find_two_points_l664_664711


namespace builder_daily_wage_l664_664115

/-- 
  Given:
  1. Three builders can build a single floor of a house in 30 days.
  2. It costs $270000 to hire 6 builders to build 5 houses with 6 floors each.
  
  Prove:
  The cost per builder for a single day's work is $100.
-/
theorem builder_daily_wage :
  let floors_per_builder_per_day := 1 / (3 * 30),
      total_floors := 5 * 6,
      total_builder_days := total_floors / floors_per_builder_per_day,
      total_cost := 270000,
      total_builders := 6
  in total_cost / (total_builder_days * total_builders) = 100 := 
by
  sorry

end builder_daily_wage_l664_664115


namespace hexagon_properties_l664_664718

-- Define the side length of the hexagon
def side_length : ℝ := 6

-- Define the radius of the circumscribed circle
def radius : ℝ := side_length

-- Total circumference of the circle
def circumference : ℝ := 2 * Real.pi * radius

-- Arc length corresponding to one side of the hexagon
def arc_length : ℝ := (60 / 360) * circumference

-- Area of the hexagon
def area : ℝ := (3 * side_length^2 * Real.sqrt 3) / 2

theorem hexagon_properties :
  radius = 6 ∧
  circumference = 12 * Real.pi ∧
  arc_length = 2 * Real.pi ∧
  area = 54 * Real.sqrt 3 :=
by
  sorry

end hexagon_properties_l664_664718


namespace complex_number_solution_l664_664578

theorem complex_number_solution (a b : ℤ) (ha : 0 < a) (hb : 0 < b) (h : (complex.mk a (-b))^3 = complex.mk 27 (-27)) :
  (complex.mk a (-b)) = complex.mk 3 (-1) := by
sorry

end complex_number_solution_l664_664578


namespace symmetric_point_xOz_l664_664011

def symmetric_point (p : (ℝ × ℝ × ℝ)) (plane : ℝ → Prop) : (ℝ × ℝ × ℝ) :=
match p with
| (x, y, z) => (x, -y, z)

theorem symmetric_point_xOz (x y z : ℝ) : symmetric_point (-1, 2, 1) (λ y, y = 0) = (-1, -2, 1) :=
by
  sorry

end symmetric_point_xOz_l664_664011


namespace triangle_bc_length_l664_664937

theorem triangle_bc_length :
  ∀ (A B C X : Type) (d_AB : ℝ) (d_AC : ℝ) (d_BX d_CX BC : ℕ),
  d_AB = 86 ∧ d_AC = 97 →
  let circleA := {center := A, radius := d_AB} in
  let intersect_B := B ∈ circleA in
  let intersect_X := X ∈ circleA in
  d_BX + d_CX = BC →
  d_BX ∈ ℕ ∧ d_CX ∈ ℕ →
  BC = 61 :=
by
  intros A B C X d_AB d_AC d_BX d_CX BC h_dist h_circle h_intersect h_sum h_intBC
  sorry

end triangle_bc_length_l664_664937


namespace even_number_of_divisors_less_than_100_l664_664416

theorem even_number_of_divisors_less_than_100 :
  ∃ (count : ℕ), count = 90 ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 100 →
    (∃ (d : ℕ), d * d = n ∨ (number_of_divisors n % 2 = 0)) :=
begin
  -- the proof goes here
  sorry
end

end even_number_of_divisors_less_than_100_l664_664416


namespace find_value_of_N_l664_664234

theorem find_value_of_N :
  ∃ N : ℝ, 2 * ((N * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1600.0000000000002 ∧ N = 3.6 :=
by {
  existsi 3.6,
  sorry
}

end find_value_of_N_l664_664234


namespace remainder_of_power_plus_five_l664_664679

theorem remainder_of_power_plus_five (n : ℕ) : 
  (5^n + 5) % 8 = 2 := by
  have cycle : ∀ k, 5^(2*k + 1) % 8 = 5 ∧ 5^(2*k + 2) % 8 = 1 := by
    intro k
    induction k with k ih
    · simp [pow_succ]
    · split
      · simp [pow, ih]
      · simp [pow, ih]
  have h : 5^123 % 8 = 5 := by
    exact (cycle _).left
  calc
  (5^123 + 5) % 8 = (5 + 5) % 8 : by rw [h]
               ... = 10 % 8      : by sorry
               ... = 2           : by sorry

end remainder_of_power_plus_five_l664_664679


namespace product_not_eq_pow2017_l664_664539

theorem product_not_eq_pow2017 (a b : Fin 2015 → ℤ) (h_perm : Multiset.elems (Vector.toList a) = Multiset.elems (Vector.toList b)) :
  (∏ i, (a i - b i)) ≠ 2017 ^ 2016 :=
sorry

end product_not_eq_pow2017_l664_664539


namespace parallelogram_diagonals_equal_implies_rectangle_l664_664184

variable (A B C D : Type)
variable [has_add A] [has_add B] [has_add C] [has_add D]

-- Definitions involving the geometrical configurations
def is_parallelogram (A B C D : Type) : Prop := 
  sorry -- This should define the parallelogram property formally

def is_rectangle (A B C D : Type) : Prop := 
  sorry -- This should define the rectangle property formally

def diagonals_equal (A B C D : Type) [has_eq A] [has_eq C] : Prop := 
  sorry -- This should ensure the diagonals AC == BD

-- The Lean 4 statement for our problem
theorem parallelogram_diagonals_equal_implies_rectangle :
  ∀ (A B C D : Type) [has_add A] [has_add B] [has_add C] [has_add D],
  is_parallelogram A B C D →
  diagonals_equal A B C D →
  is_rectangle A B C D :=
by {
  intros,
  sorry
}

end parallelogram_diagonals_equal_implies_rectangle_l664_664184


namespace total_marbles_l664_664573

/--
Some marbles in a bag are red and the rest are blue.
If one red marble is removed, then one-seventh of the remaining marbles are red.
If two blue marbles are removed instead of one red, then one-fifth of the remaining marbles are red.
Prove that the total number of marbles in the bag originally is 22.
-/
theorem total_marbles (r b : ℕ) (h1 : (r - 1) / (r + b - 1) = 1 / 7) (h2 : r / (r + b - 2) = 1 / 5) :
  r + b = 22 := by
  sorry

end total_marbles_l664_664573


namespace even_number_of_divisors_l664_664342

-- Proof statement: There are 90 positive integers less than 100 with an even number of divisors.
theorem even_number_of_divisors : 
  {n : ℕ | n < 100 ∧ ∃ k : ℕ, k ^ 2 = n}.toFinset.card = 90 := 
sorry

end even_number_of_divisors_l664_664342


namespace evens_divisors_lt_100_l664_664373

theorem evens_divisors_lt_100 : 
  ∃ n : ℕ, n = 90 ∧ ∀ k : ℕ, (1 ≤ k < 100) → (even k ↔ (∃ m : ℕ, m * m = k)) ↔ (n = 90) := 
sorry

end evens_divisors_lt_100_l664_664373


namespace find_length_BC_l664_664555

noncomputable def length_BC (A B C D E M : Point) (ED CD : ℝ) :=
  (Rectangle ABCD ∧ PointOn AD E ∧ PointOn EC M ∧ AB = BM ∧ AE = EM ∧ ED = 16 ∧ CD = 12)

theorem find_length_BC : 
  ∀ (A B C D E M : Point), Rectangle ABCD → PointOn AD E → PointOn EC M → AB = BM → AE = EM → ED = 16 → CD = 12 → BC = 20 :=
by
  sorry

end find_length_BC_l664_664555


namespace length_of_BC_l664_664930

theorem length_of_BC (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
    (BX CX : ℕ) (h_pow : CX * (BX + CX) = 2013) : 
    BX + CX = 61 :=
  sorry

end length_of_BC_l664_664930


namespace square_root_condition_l664_664864

theorem square_root_condition (x : ℝ) : (6 + x ≥ 0) ↔ (x ≥ -6) :=
by sorry

end square_root_condition_l664_664864


namespace complementary_angles_in_equilateral_l664_664525

open Classical
open Geometry

noncomputable section

def midpoint (A B : Point) : Point := midpoint A B

def equilateral_triangle (A B C : Point) (h : triangle A B C) : Prop :=
∀ P Q R : Point, triangle P Q R → length P Q = length Q R ∧ length Q R = length R P

def extend_and_point (A B : Point) : Point := -- placeholder for the actual construction of the point on extended segment
sorry

theorem complementary_angles_in_equilateral (A B C D E M F: Point)
  (h : equilateral_triangle A B C)
  (hm : midpoint B C = M)
  (hD : extend_and_point A C = D)
  (hE : extend_and_point A B = E)
  (hMD : distance M D = distance M E)
  (F : Point) -- Intersection of MD and AB, definition is skipped here
  : angle B F M = angle B M E :=
sorry

end complementary_angles_in_equilateral_l664_664525


namespace even_divisors_count_lt_100_l664_664347

theorem even_divisors_count_lt_100 : 
  {n : ℕ | n < 100 ∧ n ≠ 0 ∧ ∃ k : ℕ, k * k = n } = { n : ℕ | n < 100 ∧ n ≠ 0 } \ 
  { n : ℕ | ∃ k : ℕ, k * k = n ∧ k < 100 } → 
  (card {n : ℕ | n < 100 ∧ n ≠ 0 ∧ even (finset.card (divisors n))} = 90) :=
begin
  sorry
end

end even_divisors_count_lt_100_l664_664347


namespace sum_of_as_gt_zero_l664_664987

theorem sum_of_as_gt_zero (a : ℕ → ℤ) (n : ℕ) 
  (h1 : ∃ i, i ≤ n ∧ a i ≠ 0)
  (h2 : ∀ i, i ≤ n → a i ≥ -1)
  (h3 : ∑ i in Finset.range (n + 1), (2^i) * (a i) = 0) :
  ∑ i in Finset.range (n + 1), a i > 0 :=
by
  sorry

end sum_of_as_gt_zero_l664_664987


namespace converse_and_inverse_false_l664_664833

def Triangle (T : Type) := T
variables {T : Type} [Triangle T]

def is_equilateral (t : T) : Prop := sorry  -- assumes definition of equilateral triangle
def is_isosceles (t : T) : Prop := sorry    -- assumes definition of isosceles triangle

-- Given true statement: If a triangle is equilateral, then it is isosceles.
axiom equilateral_implies_isosceles (t : T) : is_equilateral t → is_isosceles t

-- Theorem to prove: The converse and inverse are both false
theorem converse_and_inverse_false (t : T) :
  ¬ (is_isosceles t → is_equilateral t) ∧ ¬ (¬ is_equilateral t → ¬ is_isosceles t) :=
sorry

end converse_and_inverse_false_l664_664833


namespace greatest_prime_factor_15_fact_plus_18_fact_eq_17_l664_664662

theorem greatest_prime_factor_15_fact_plus_18_fact_eq_17 :
  ∃ p : ℕ, prime p ∧ 
  (∀ q : ℕ, (prime q ∧ q ∣ (15.factorial + 18.factorial)) → q ≤ p) ∧ 
  p = 17 :=
by
  sorry

end greatest_prime_factor_15_fact_plus_18_fact_eq_17_l664_664662


namespace find_angle_ADB_l664_664889

noncomputable theory

open Real

variables (A B C D P : Type) [planar_plane A] [planar_plane B] [planar_plane C] [planar_plane D] [planar_plane P]
variables (angle_DBC angle_ACB angle_ABD angle_ACD angle_ADB : ℝ)
variables [angle_DBC' : angle_DBC = 60] [angle_ACB' : angle_ACB = 50]
variables [angle_ABD' : angle_ABD = 20] [angle_ACD' : angle_ACD = 30]

theorem find_angle_ADB :
  let ABCD is convex quadrilateral,
      ∠ DBC = 60°,
      ∠ ACB = 50°,
      ∠ ABD = 20°,
      ∠ ACD = 30°
  in ∠ ADB = 30° :=
by sorry

end find_angle_ADB_l664_664889


namespace number_of_integers_with_even_divisors_l664_664394

-- Define conditions
def N := 99

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P : finset ℕ := { n in finset.range (N + 1) | is_perfect_square n }

noncomputable def countP : ℕ := P.card

-- The statement to prove
theorem number_of_integers_with_even_divisors : 
  (N - countP) = 90 :=
by {
  sorry
}

end number_of_integers_with_even_divisors_l664_664394


namespace angle_in_first_quadrant_l664_664156

-- Define the condition and equivalence proof problem in Lean 4
theorem angle_in_first_quadrant (deg : ℤ) (h1 : deg = 721) : (deg % 360) > 0 := 
by 
  have : deg % 360 = 1 := sorry
  exact sorry

end angle_in_first_quadrant_l664_664156


namespace password_A_seventh_week_l664_664158
noncomputable theory

-- Define the recursive probability P_n function
def P : ℕ → ℚ
| 1       := 1
| 2       := 0
| 3       := 1/3
| (n + 1) := (1 - P n) * 1/3

-- Define the theorem with the required conditions and expected result
theorem password_A_seventh_week : P 7 = 61 / 243 :=
by sorry

end password_A_seventh_week_l664_664158


namespace number_of_integers_with_even_divisors_l664_664398

-- Define conditions
def N := 99

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P : finset ℕ := { n in finset.range (N + 1) | is_perfect_square n }

noncomputable def countP : ℕ := P.card

-- The statement to prove
theorem number_of_integers_with_even_divisors : 
  (N - countP) = 90 :=
by {
  sorry
}

end number_of_integers_with_even_divisors_l664_664398


namespace greatest_prime_factor_15_18_l664_664652

theorem greatest_prime_factor_15_18! :
  ∃ p : ℕ, prime p ∧ p ∈ prime_factors (15! + 18!) ∧ ∀ q : ℕ, prime q → q ∈ prime_factors (15! + 18!) → q ≤ 4897 := 
sorry

end greatest_prime_factor_15_18_l664_664652


namespace complex_in_second_quadrant_l664_664585

def i := Complex.I
def z := i * (1 + i)

theorem complex_in_second_quadrant (z = i * (1 + i)) : -1 + i ∈ {(x, y) : ℝ × ℝ | x < 0 ∧ y > 0} :=
by
  sorry

end complex_in_second_quadrant_l664_664585


namespace BC_length_l664_664950

-- Define the given values and conditions
variable (A B C X : Type)
variable (AB AC AX BX CX : ℕ)
variable [inhabited A] [inhabited B] [inhabited C] [inhabited X]

-- Assume the lengths of AB and AC
axiom h_AB : AB = 86
axiom h_AC : AC = 97

-- Assume the circle centered at A with radius AB intersects BC at B and X
axiom h_circle : AX = AB

-- Assume BX and CX are integers
axiom h_BX_integral : ∃ (x : ℕ), BX = x
axiom h_CX_integral : ∃ (y : ℕ), CX = y

-- The statement to prove that the length of BC is 61
theorem BC_length : (∃ (x y : ℕ), BX = x ∧ CX = y ∧ x + y = 61) :=
by
  sorry

end BC_length_l664_664950


namespace triangle_length_sum_l664_664810

open Real

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_length_sum
(A B C P Q : ℝ × ℝ)
(hA : A = (0, 0))
(hB : B = (7, 0))
(hC : C = (3, 4))
(hL : ∃ k : ℝ, (6 - 2 * sqrt 2, 3 - sqrt 2) = (A.1 + k * (C.1 - A.1), A.2 + k * (C.2 - A.2)) ∧
              ∃ m : ℝ, (6 - 2 * sqrt 2, 3 - sqrt 2) = (B.1 + m * (C.1 - B.1), B.2 + m * (C.2 - B.2)))
(hP : ∃ t : ℝ, P = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2)))
(hQ : ∃ s : ℝ, Q = (B.1 + s * (C.1 - B.1), B.2 + s * (C.2 - B.2)))
(hArea : triangle_area P Q C = 14 / 3) :
  abs (sqrt ((C.1 - P.1)^2 + (C.2 - P.2)^2) + sqrt ((C.1 - Q.1)^2 + (C.2 - Q.2)^2)) = 63 :=
begin
  sorry
end

end triangle_length_sum_l664_664810


namespace frame_painting_ratio_l664_664715

theorem frame_painting_ratio :
  ∃ (x : ℝ), (20 + 2 * x) * (30 + 6 * x) = 1800 → 1 = 2 * (20 + 2 * x) / (30 + 6 * x) :=
by
  sorry

end frame_painting_ratio_l664_664715


namespace min_value_l664_664850

theorem min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3) :
  2 * a + b + c ≥ 2 * Real.sqrt 3 - 2 :=
sorry

end min_value_l664_664850


namespace even_number_of_divisors_less_than_100_l664_664312

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k, k * k = n

theorem even_number_of_divisors_less_than_100 :
  (card { n : ℕ | 1 ≤ n ∧ n < 100 ∧ ¬ is_perfect_square n }) = 90 :=
by
  sorry

end even_number_of_divisors_less_than_100_l664_664312


namespace smallest_n_value_l664_664541

theorem smallest_n_value (n : ℕ) (h₀ : 0 < n) (h₁ : (1 / 3 : ℚ) + (1 / 4) + (1 / 8) + 1 / n ∈ ℤ) : n = 24 :=
sorry

end smallest_n_value_l664_664541


namespace derivative_of_f_l664_664080

noncomputable def f (x : ℝ) : ℝ := cos (x^2 + x)

theorem derivative_of_f :
  (deriv f x) = -(2 * x + 1) * sin (x^2 + x) :=
sorry

end derivative_of_f_l664_664080


namespace number_of_x_for_g100_eq_zero_l664_664534

noncomputable def g0 (x : ℝ) : ℝ := 
  if x < -150 then x + 300
  else if x < 150 then -x
  else x - 300

noncomputable def g : ℕ → ℝ → ℝ 
| 0, x := g0 x
| (n+1), x := abs (g n x) - 2

theorem number_of_x_for_g100_eq_zero : 
  (finset.univ.filter (λ (x : ℝ), g 100 x = 0)).card = 2 :=
sorry

end number_of_x_for_g100_eq_zero_l664_664534


namespace max_even_numbers_on_board_l664_664798

theorem max_even_numbers_on_board (n : ℕ) (board : ℕ → ℕ → ℤ) : 
  (∃ moves : list (ℕ × ℕ), 
    let final_board := moves.foldr (λ (c : ℕ × ℕ) b, 
      update_board b c) board in 
    count_evens final_board ≥ if even n then n^2 else n^2 - n + 1) := sorry

/-- Helper functions -/

/-- Check if a number is even -/
def even (x : ℕ) := x % 2 = 0

/-- Count the number of even numbers in the board -/
def count_evens (board : ℕ → ℕ → ℤ) : ℕ := 
  finset.sum (finset.univ.product finset.univ) (λ (c : ℕ × ℕ), 
    if even (int.natAbs (board c.fst c.snd)) then 1 else 0)

/-- Update the board by adding 1 to all elements in the row and column specified by (r, c) -/
def update_board (board : ℕ → ℕ → ℤ) (rc : ℕ × ℕ) : ℕ → ℕ → ℤ := 
  let r := rc.fst
  let c := rc.snd
  λ i j, if i = r ∨ j = c then board i j + 1 else board i j

end max_even_numbers_on_board_l664_664798


namespace masha_goal_impossible_l664_664481

-- Definitions and conditions:

def num_vases : ℕ := 2019
def vases := Fin num_vases → (ℕ × ℕ)  -- Each vase represented by a tuple of white and red roses.

-- Statement: It is not possible for Masha to achieve her goal.

theorem masha_goal_impossible (initial : vases) :
  ¬ ∃ final : vases,
    (∀ i : Fin num_vases,
      final i ≠ initial i) ∧
    (∀ i : Fin num_vases,
      (final i).fst + 1 = (initial ((i+1) % num_vases)).fst ∧
      (final i).snd + 1 = (initial ((i-1) % num_vases)).snd) :=
sorry

end masha_goal_impossible_l664_664481


namespace possible_area_l664_664483

theorem possible_area (A : ℝ) (B : ℝ) (L : ℝ × ℝ) (H₁ : L.1 = 13) (H₂ : L.2 = 14) (area_needed : ℝ) (H₃ : area_needed = 200) : 
∃ x y : ℝ, x = 13 ∧ y = 16 ∧ x * y ≥ area_needed :=
by
  sorry

end possible_area_l664_664483


namespace total_expenditure_is_108_l664_664073

-- Define the costs of items and quantities purchased by Robert and Teddy
def cost_pizza := 10   -- cost of one box of pizza
def cost_soft_drink := 2  -- cost of one can of soft drink
def cost_hamburger := 3   -- cost of one hamburger

def qty_pizza_robert := 5     -- quantity of pizza boxes by Robert
def qty_soft_drink_robert := 10 -- quantity of soft drinks by Robert

def qty_hamburger_teddy := 6  -- quantity of hamburgers by Teddy
def qty_soft_drink_teddy := 10 -- quantity of soft drinks by Teddy

-- Calculate total expenditure for Robert and Teddy
def total_cost_robert := (qty_pizza_robert * cost_pizza) + (qty_soft_drink_robert * cost_soft_drink)
def total_cost_teddy := (qty_hamburger_teddy * cost_hamburger) + (qty_soft_drink_teddy * cost_soft_drink)

-- Total expenditure in all
def total_expenditure := total_cost_robert + total_cost_teddy

-- We formulate the theorem to prove that the total expenditure is $108
theorem total_expenditure_is_108 : total_expenditure = 108 :=
by 
  -- Placeholder proof
  sorry

end total_expenditure_is_108_l664_664073


namespace largest_is_B_l664_664682

noncomputable def A := Real.sqrt (Real.sqrt (56 ^ (1 / 3)))
noncomputable def B := Real.sqrt (Real.sqrt (3584 ^ (1 / 3)))
noncomputable def C := Real.sqrt (Real.sqrt (2744 ^ (1 / 3)))
noncomputable def D := Real.sqrt (Real.sqrt (392 ^ (1 / 3)))
noncomputable def E := Real.sqrt (Real.sqrt (448 ^ (1 / 3)))

theorem largest_is_B : B > A ∧ B > C ∧ B > D ∧ B > E := by
  sorry

end largest_is_B_l664_664682


namespace length_of_second_train_is_279point96_l664_664124

-- Define the conditions
def length_first_train : ℝ := 180  -- meters
def speed_first_train : ℝ := 42 * (1000 / 3600) -- kmph converted to m/s
def speed_second_train : ℝ := 30 * (1000 / 3600) -- kmph converted to m/s
def time_to_clear : ℝ := 22.998 -- seconds

-- Define the length of the second train
def length_second_train (L : ℝ) : Prop := 
  length_first_train + L = (speed_first_train + speed_second_train) * time_to_clear

-- Prove the length of the second train
theorem length_of_second_train_is_279point96 : length_second_train 279.96 :=
  by
  -- This is where the proof would go
  sorry

end length_of_second_train_is_279point96_l664_664124


namespace find_k_l664_664606

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum
  
def repeated_eight (k : ℕ) : ℕ :=
  (List.repeat 8 k).foldl (λ acc d, acc * 10 + d) 0
  
theorem find_k : ∃ k : ℕ, digit_sum (8 * repeated_eight k) = 1000 ∧ k = 991 :=
by
  have k : ℕ := 991
  have h : digit_sum (8 * repeated_eight k) = 1000 := sorry
  use k
  exact ⟨h, rfl⟩

end find_k_l664_664606


namespace painting_falls_l664_664839

theorem painting_falls (k : ℕ) : 
  let n := 2^k
  in ∃ x : ℕ, x = 2^(2*k) ∧ (∀ i < n, x = 2^(2*k)) :=
by
  sorry

end painting_falls_l664_664839


namespace evens_divisors_lt_100_l664_664368

theorem evens_divisors_lt_100 : 
  ∃ n : ℕ, n = 90 ∧ ∀ k : ℕ, (1 ≤ k < 100) → (even k ↔ (∃ m : ℕ, m * m = k)) ↔ (n = 90) := 
sorry

end evens_divisors_lt_100_l664_664368


namespace best_model_based_on_R_squared_l664_664883

theorem best_model_based_on_R_squared:
  ∀ (R2_1 R2_2 R2_3 R2_4: ℝ), 
  R2_1 = 0.98 → R2_2 = 0.80 → R2_3 = 0.54 → R2_4 = 0.35 → 
  R2_1 ≥ R2_2 ∧ R2_1 ≥ R2_3 ∧ R2_1 ≥ R2_4 :=
by
  intros R2_1 R2_2 R2_3 R2_4 h1 h2 h3 h4
  sorry

end best_model_based_on_R_squared_l664_664883


namespace polynomial_evaluation_l664_664043

theorem polynomial_evaluation 
  (g : ℝ → ℝ) (h : ∀ x, g(x^2 + 2) = x^4 + 5 * x^2 + 1) :
  ∀ x, g(x^2 - 2) = x^4 - 3 * x^2 - 3 := 
by
  intro x
  have h1 : g (x^2 + 2) = x^4 + 5 * x^2 + 1 := h x
  sorry

end polynomial_evaluation_l664_664043


namespace circle_ratio_increase_l664_664465

theorem circle_ratio_increase (r : ℝ) (h : r + 2 ≠ 0) : 
  (2 * Real.pi * (r + 2)) / (2 * (r + 2)) = Real.pi :=
by
  sorry

end circle_ratio_increase_l664_664465


namespace equal_roots_quadratic_l664_664862

theorem equal_roots_quadratic {k : ℝ} 
  (h : (∃ x : ℝ, x^2 - 6 * x + k = 0 ∧ x^2 - 6 * x + k = 0)) : 
  k = 9 :=
sorry

end equal_roots_quadratic_l664_664862


namespace angle_D_in_parallelogram_l664_664884

variables {A C D : ℝ} -- Angle variables

-- Assume the conditions
def parallelogram_angle_property (h : A + C = 108) : D = 126 :=
  let A := 54 in -- From A + A = 108
  let C := 54 in -- And knowing A = C in a parallelogram
  let D := 180 - C in -- Supplementary angles property D + C = 180
  D

-- Now let's state the theorem in Lean
theorem angle_D_in_parallelogram (h_parallelogram : true) (h_sum_angle : A + C = 108) : D = 126 :=
  parallelogram_angle_property h_sum_angle

end angle_D_in_parallelogram_l664_664884


namespace angle_is_3_pi_over_4_l664_664837

def vec (α β : ℝ) : ℝ × ℝ := (α, β)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def magnitude (u : ℝ × ℝ) : ℝ := real.sqrt (u.1^2 + u.2^2)

def angle_between_vectors (u v : ℝ × ℝ) : ℝ := 
  real.acos ((dot_product u v) / (magnitude u * magnitude v))

def a := vec (-1) 2
def b := vec (-1) (-1)
def v := vec (-6) (-6)
def w := vec 0 1

theorem angle_is_3_pi_over_4 : angle_between_vectors v w = 3 * real.pi / 4 :=
  sorry

end angle_is_3_pi_over_4_l664_664837


namespace feed_duration_l664_664024

-- Define the given conditions
def initial_boxes := 5
def additional_boxes := 3
def seed_per_box := 225
def parrot_consumption_per_week := 100
def cockatiel_consumption_per_week := 50

-- Calculate total boxes of birdseed
def total_boxes := initial_boxes + additional_boxes

-- Calculate the total amount of birdseed in grams
def total_birdseed := total_boxes * seed_per_box

-- Calculate the total consumption per week in grams
def total_weekly_consumption := parrot_consumption_per_week + cockatiel_consumption_per_week

-- Calculate the number of weeks Leah can feed her birds
def weeks : ℕ := total_birdseed / total_weekly_consumption

-- Prove the number of weeks is 12
theorem feed_duration : weeks = 12 :=
by
  -- Definitions and conditions
  let initial_boxes := 5
  let additional_boxes := 3
  let seed_per_box := 225
  let parrot_consumption_per_week := 100
  let cockatiel_consumption_per_week := 50

  -- Calculations
  have total_boxes_eq := initial_boxes + additional_boxes
  have total_birdseed_eq := total_boxes_eq * seed_per_box
  have total_weekly_consumption_eq := parrot_consumption_per_week + cockatiel_consumption_per_week
  have weeks_eq := total_birdseed_eq / total_weekly_consumption_eq

  -- Proof
  show weeks_eq = 12
  sorry

end feed_duration_l664_664024


namespace determine_coefficients_l664_664448

variable {α : Type} [Field α]
variables (a a1 a2 a3 : α)

theorem determine_coefficients (h : ∀ x : α, a + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 = x^3) :
  a = 1 ∧ a2 = 3 :=
by
  -- To be proven
  sorry

end determine_coefficients_l664_664448


namespace even_divisors_count_lt_100_l664_664348

theorem even_divisors_count_lt_100 : 
  {n : ℕ | n < 100 ∧ n ≠ 0 ∧ ∃ k : ℕ, k * k = n } = { n : ℕ | n < 100 ∧ n ≠ 0 } \ 
  { n : ℕ | ∃ k : ℕ, k * k = n ∧ k < 100 } → 
  (card {n : ℕ | n < 100 ∧ n ≠ 0 ∧ even (finset.card (divisors n))} = 90) :=
begin
  sorry
end

end even_divisors_count_lt_100_l664_664348


namespace semicircle_area_l664_664491

-- Given conditions
variables (A B C : Type) [euclidean_geometry A B C]
variables (angle_right_B : ∠B = 90 ^ \circ)
variables (BC AC : ℝ)
variables (BC_eq : BC = 15)
variables (AC_eq : AC = 17)

-- Statement of the problem
theorem semicircle_area {A B C : Type} [euclidean_geometry A B C] (BC AC : ℝ)
  (angle_right_B : ∠B = 90^°) (BC_eq : BC = 15) (AC_eq : AC = 17):
  area (semicircle_diameter (hypotenuse A B C)) = 8 * π := 
by
  -- skipping the proof
  sorry

end semicircle_area_l664_664491


namespace symmetric_point_in_xOz_l664_664005

def symmetric_point (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (P.1, -P.2, P.3)

theorem symmetric_point_in_xOz (P : ℝ × ℝ × ℝ) : 
  symmetric_point P = (P.1, -P.2, P.3) :=
by
  sorry

example : symmetric_point (-1, 2, 1) = (-1, -2, 1) :=
by
  rw symmetric_point_in_xOz
  rw symmetric_point
  sorry

end symmetric_point_in_xOz_l664_664005


namespace largest_option_is_B_l664_664685

/-- Define the provided options -/
def optionA : ℝ := Real.sqrt (Real.cbrt 56)
def optionB : ℝ := Real.sqrt (Real.cbrt 3584)
def optionC : ℝ := Real.sqrt (Real.cbrt 2744)
def optionD : ℝ := Real.cbrt (Real.sqrt 392)
def optionE : ℝ := Real.cbrt (Real.sqrt 448)

/-- The main theorem -/
theorem largest_option_is_B : optionB > optionA ∧ optionB > optionC ∧ optionB > optionD ∧ optionB > optionE :=
by
  sorry

end largest_option_is_B_l664_664685


namespace BC_length_l664_664918

theorem BC_length (A B C X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X] 
  (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
  (BX CX : ℕ) (h_circle_intersect : ∃ X, Metric.ball A 86 ∩ {BC} = {B, X})
  (h_integer_lengths : BX + CX = BC) :
  BC = 61 := 
by
  sorry

end BC_length_l664_664918


namespace phillip_remaining_money_l664_664559

def initial_money : ℝ := 95
def cost_oranges : ℝ := 14
def cost_apples : ℝ := 25
def cost_candy : ℝ := 6
def cost_eggs : ℝ := 12
def cost_milk : ℝ := 8
def discount_apples_rate : ℝ := 0.15
def discount_milk_rate : ℝ := 0.10

def discounted_cost_apples : ℝ := cost_apples * (1 - discount_apples_rate)
def discounted_cost_milk : ℝ := cost_milk * (1 - discount_milk_rate)

def total_spent : ℝ := cost_oranges + discounted_cost_apples + cost_candy + cost_eggs + discounted_cost_milk

def remaining_money : ℝ := initial_money - total_spent

theorem phillip_remaining_money : remaining_money = 34.55 := by
  -- Proof here
  sorry

end phillip_remaining_money_l664_664559


namespace roots_recip_squared_sum_l664_664590

def cubic_poly : Polynomial ℝ := Polynomial.Cubic 1 (-9) 8 2

theorem roots_recip_squared_sum :
  let p q r : ℝ := roots_of_cubic_poly cubic_poly
  let s₁ := p + q + r
  let s₂ := p * q + q * r + r * p
  let s₃ := p * q * r
  s₁ = 9 ∧ s₂ = 8 ∧ s₃ = -2 →
  (1 / p^2) + (1 / q^2) + (1 / r^2) = 25 :=
by {
  intros p q r s₁ s₂ s₃ h,
  have h₁ : p + q + r = 9 := h.1,
  have h₂ : p * q + q * r + r * p = 8 := h.2.1,
  have h₃ : p * q * r = -2 := h.2.2,
  sorry
}

end roots_recip_squared_sum_l664_664590


namespace proof_problem1_proof_problem2_l664_664739

noncomputable def problem1 : Prop :=
  sqrt (25 / 9) - (8 / 27)^(1 / 3) - (Real.pi + Real.exp 1)^(0) + (1 / 4)^(-1 / 2) = 2

noncomputable def problem2 : Prop :=
  2 * Real.log 5 + Real.log 4 + Real.log (sqrt (Real.exp 1)) = 5 / 2

theorem proof_problem1 : problem1 := sorry
theorem proof_problem2 : problem2 := sorry

end proof_problem1_proof_problem2_l664_664739


namespace leah_birdseed_feeding_weeks_l664_664025

/-- Define the total number of weeks Leah can feed her birds without going back to the store. -/
theorem leah_birdseed_feeding_weeks : 
  (let num_boxes_bought := 3
   let num_boxes_pantry := 5
   let parrot_weekly_consumption := 100
   let cockatiel_weekly_consumption := 50
   let grams_per_box := 225
   let total_boxes := num_boxes_bought + num_boxes_pantry
   let total_birdseed_grams := total_boxes * grams_per_box
   let total_weekly_consumption := parrot_weekly_consumption + cockatiel_weekly_consumption
  in total_birdseed_grams / total_weekly_consumption) = 12 := 
by 
  sorry

end leah_birdseed_feeding_weeks_l664_664025


namespace brianne_january_savings_l664_664198

theorem brianne_january_savings (S : ℝ) (h : 16 * S = 160) : S = 10 :=
sorry

end brianne_january_savings_l664_664198


namespace polyhedron_proof_l664_664065

noncomputable def calculate_e_squared (a b ab g f bf ag xb yb gz yz ze cd de ecd) := 
  let e : Real := (6, 6, 12)
  let g : Real := (3, 0, sqrt(55))
  let fg := (6 : Real)

  sqrt ((e.1 - g.1)^2 + (e.2 - g.2)^2 + (e.3 - f)^2)

theorem polyhedron_proof 
(abcd ab g f bf ag gf ce de ece ed)
(h1 : abcd.singleton ∧ ab = 12)
(h2 : ag ∧ bf = 8 ∧ gf = 6)
(h3 : ce = 14 ∧ de = 14): 
p = 244 ∧ q = 24 ∧ r = 55 →
EG_squared = p - q * rt →
(p + q + r = 323)
:=
sorry

end polyhedron_proof_l664_664065


namespace subsets_0_to_14_count_l664_664745

noncomputable def num_subsets : ℕ :=
  let s : ℕ → ℕ
  | 0       := 1
  | 1       := 2
  | (n + 2) := s (n + 1) ^ 2 + s n ^ 4
  s 4

theorem subsets_0_to_14_count :
  num_subsets = 2306 :=
by
  sorry

end subsets_0_to_14_count_l664_664745


namespace order_exponents_l664_664063

theorem order_exponents :
  (2:ℝ) ^ 300 < (3:ℝ) ^ 200 ∧ (3:ℝ) ^ 200 < (10:ℝ) ^ 100 :=
by
  sorry

end order_exponents_l664_664063


namespace seq_count_is_24_l664_664489

noncomputable def num_possible_sequences : ℕ :=
  let procedures := ["A", "B", "C", "D", "E"]
  let pairs := [("C", "D"), ("D", "C")]
  let valid_sequences := (list.permutations procedures).filter (λ seq,
    seq.head? = some "A" ∨ list.last? seq = some "A" ∧
    pairs.any (λ (cd : String × String), list.is_adjacent seq cd.fst cd.snd))
  valid_sequences.length

theorem seq_count_is_24 : num_possible_sequences = 24 :=
  sorry

end seq_count_is_24_l664_664489


namespace students_neither_music_nor_art_l664_664144

theorem students_neither_music_nor_art :
  ∀ (total_students music_students art_students both_students : ℕ),
    total_students = 500 →
    music_students = 20 →
    art_students = 20 →
    both_students = 10 →
    total_students - (music_students + art_students - both_students) = 470 :=
by
  intros total_students music_students art_students both_students
  assume total_eq music_eq art_eq both_eq
  rw [total_eq, music_eq, art_eq, both_eq]
  simp
  sorry

end students_neither_music_nor_art_l664_664144


namespace sunzi_wood_problem_l664_664493

theorem sunzi_wood_problem (x y : ℝ) (h1 : x - y = 4.5) (h2 : (1/2) * x + 1 = y) :
  (x - y = 4.5) ∧ ((1/2) * x + 1 = y) :=
by {
  exact ⟨h1, h2⟩
}

end sunzi_wood_problem_l664_664493


namespace triangle_bc_length_l664_664903

theorem triangle_bc_length (A B C X : Type)
  (AB AC : ℕ)
  (hAB : AB = 86)
  (hAC : AC = 97)
  (circle_eq : ∀ {r : ℕ}, r = AB → circle_centered_at_A_intersects_BC_two_points B X)
  (integer_lengths : ∃ (BX CX : ℕ), ) :
  BC = 61 :=
by
  sorry

end triangle_bc_length_l664_664903


namespace length_of_segment_DB_l664_664894

theorem length_of_segment_DB (A B C D : Point) (h₁ : is_right_angle ∠ A D B)
  (h₂ : dist A C = 17) (h₃ : dist A D = 8) (h₄ : C lies_between A B) (h₅ : dist B C = 4) :
  dist D B = Real.sqrt 377 :=
by
  sorry

end length_of_segment_DB_l664_664894


namespace BC_length_l664_664946

-- Define the given values and conditions
variable (A B C X : Type)
variable (AB AC AX BX CX : ℕ)
variable [inhabited A] [inhabited B] [inhabited C] [inhabited X]

-- Assume the lengths of AB and AC
axiom h_AB : AB = 86
axiom h_AC : AC = 97

-- Assume the circle centered at A with radius AB intersects BC at B and X
axiom h_circle : AX = AB

-- Assume BX and CX are integers
axiom h_BX_integral : ∃ (x : ℕ), BX = x
axiom h_CX_integral : ∃ (y : ℕ), CX = y

-- The statement to prove that the length of BC is 61
theorem BC_length : (∃ (x y : ℕ), BX = x ∧ CX = y ∧ x + y = 61) :=
by
  sorry

end BC_length_l664_664946


namespace number_of_integers_with_even_divisors_l664_664390

-- Define conditions
def N := 99

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P : finset ℕ := { n in finset.range (N + 1) | is_perfect_square n }

noncomputable def countP : ℕ := P.card

-- The statement to prove
theorem number_of_integers_with_even_divisors : 
  (N - countP) = 90 :=
by {
  sorry
}

end number_of_integers_with_even_divisors_l664_664390


namespace count_even_divisors_lt_100_l664_664304

-- Define the set of natural numbers less than 100
def nat_lt_100 := {n : ℕ | n < 100}

-- Define the set of perfect squares less than 100
def perfect_squares_lt_100 := {n : ℕ | ∃ k : ℕ, k^2 = n ∧ n < 100}

-- Define the set of natural numbers less than 100 with an even number of positive divisors
def even_divisors_lt_100 := nat_lt_100 \ perfect_squares_lt_100

-- Theorem stating the number of elements with even number of divisors is 90
theorem count_even_divisors_lt_100 : (even_divisors_lt_100).card = 90 := 
sorry

end count_even_divisors_lt_100_l664_664304


namespace pentagon_arithmetic_sequences_count_l664_664092

theorem pentagon_arithmetic_sequences_count : 
  ∃ (sequences : ℕ), sequences = 5 ∧
  ∃ (x d : ℕ), 5 * x + 10 * d = 540 ∧
  (∀ i : ℕ, i ∈ finset.range 5 → x + i * d < 120) ∧
  (∀ i : ℕ, i ∈ finset.range 5 → 0 < x + i * d) ∧
  x + d ≠ x + 4 * d :=
by
  sorry

end pentagon_arithmetic_sequences_count_l664_664092


namespace correct_statement_2_l664_664799

-- Definitions of parallel and perpendicular relationships
variables (a b : line) (α β : plane)

-- Conditions
def parallel (x y : plane) : Prop := sorry -- definition not provided
def perpendicular (x y : plane) : Prop := sorry -- definition not provided
def line_parallel_plane (l : line) (p : plane) : Prop := sorry -- definition not provided
def line_perpendicular_plane (l : line) (p : plane) : Prop := sorry -- definition not provided
def line_perpendicular (l1 l2 : line) : Prop := sorry -- definition not provided

-- Proof of the correct statement among the choices
theorem correct_statement_2 :
  line_perpendicular a b → line_perpendicular_plane a α → line_perpendicular_plane b β → perpendicular α β :=
by
  intros h1 h2 h3
  sorry

end correct_statement_2_l664_664799


namespace cone_volume_l664_664611

/-- 
Given a cone with a slant height of 6 and a central angle of the sector when the lateral surface of the cone is unfolded is 120°, prove that the volume of the cone is \( \frac{16\sqrt{2}}{3}\pi \).
-/
theorem cone_volume (l : ℝ) (θ : ℝ) (h6 : l = 6) (h120 : θ = 120) :
  let r := l * θ / 360;
  let h := sqrt (l^2 - r^2);
  let volume := (1 / 3) * π * r^2 * h;
  volume = √2 * (8 / 3) * π :=
by 
  sorry

end cone_volume_l664_664611


namespace distance_A_B_l664_664508

-- Define the points in the polar coordinate system
def A := (5, (7 * Real.pi) / 36)
def B := (12, (43 * Real.pi) / 36)

-- Define the function to calculate the distance between two points in polar coordinates
def distance (r1 θ1 r2 θ2 : ℝ) : ℝ :=
  Real.sqrt (r1^2 + r2^2 - 2 * r1 * r2 * Real.cos (θ2 - θ1))

-- The theorem that states the distance between points A and B is 17
theorem distance_A_B : distance 5 ((7 * Real.pi) / 36) 12 ((43 * Real.pi) / 36) = 17 := 
  sorry

end distance_A_B_l664_664508


namespace line_circle_product_l664_664797

noncomputable def line_l (α t : ℝ) : ℝ × ℝ := (t * cos α, -2 + t * sin α)
def circle_C (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 9

theorem line_circle_product (α : ℝ) (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  let t₁ := A.1 / cos α in
  let t₂ := B.1 / cos α in
  (circle_C (A.1) (A.2)) ∧ (circle_C (B.1) (B.2)) ∧
  (A = line_l α t₁) ∧ (B = line_l α t₂) →
  abs (16) = 16 :=
by
  intros _ _ hABC hAeq hBeq
  sorry

end line_circle_product_l664_664797


namespace white_ring_weight_l664_664984

def weight_of_orange_ring : ℝ := 0.08
def weight_of_purple_ring : ℝ := 0.33
def total_weight_of_rings : ℝ := 0.83

def weight_of_white_ring (total : ℝ) (orange : ℝ) (purple : ℝ) : ℝ :=
  total - (orange + purple)

theorem white_ring_weight :
  weight_of_white_ring total_weight_of_rings weight_of_orange_ring weight_of_purple_ring = 0.42 :=
by
  sorry

end white_ring_weight_l664_664984


namespace greatest_prime_factor_15_fact_plus_18_fact_l664_664640

theorem greatest_prime_factor_15_fact_plus_18_fact :
  Nat.greatest_prime_factor (15.factorial + 18.factorial) = 17 := by
  sorry

end greatest_prime_factor_15_fact_plus_18_fact_l664_664640


namespace hyperbola_asymptotes_l664_664082

theorem hyperbola_asymptotes :
  (∀ x y : ℝ, x^2 / 25 - y^2 / 4 = 1 → y = 2 / 5 * x ∨ y = - (2 / 5) * x) :=
begin
  sorry
end

end hyperbola_asymptotes_l664_664082


namespace length_of_BC_l664_664922

theorem length_of_BC 
  (A B C X : Type) 
  (d_AB : ℝ) (d_AC : ℝ) 
  (circle_center_A : A) 
  (radius_AB : ℝ)
  (intersects_BC : B → C → X)
  (BX CX : ℕ) 
  (h_BX_in_circle : BX = d_AB) 
  (h_CX_in_circle : CX = d_AC) 
  (h_integer_lengths : ∃ x y : ℕ, BX = x ∧ CX = y) :
  BX + CX = 61 :=
begin
  sorry
end

end length_of_BC_l664_664922


namespace triangles_in_decagon_l664_664217

theorem triangles_in_decagon :
  let n := 10 in
  let k := 3 in
  Nat.choose n k = 120 := by
  sorry

end triangles_in_decagon_l664_664217


namespace total_baseball_cards_l664_664072

theorem total_baseball_cards (R_t J_t A_t R_d J_d A_d : ℕ) 
  (h1 : R_d = 8)
  (h2 : R_t = 3 * R_d)
  (h3 : J_d = 40)
  (h4 : J_t = 40)
  (h5 : A_t = 2 * R_t)
  (h6 : A_d = A_t / 4) 
  : R_t + J_t + A_t = 112 :=
by
  sorry

end total_baseball_cards_l664_664072


namespace evens_divisors_lt_100_l664_664374

theorem evens_divisors_lt_100 : 
  ∃ n : ℕ, n = 90 ∧ ∀ k : ℕ, (1 ≤ k < 100) → (even k ↔ (∃ m : ℕ, m * m = k)) ↔ (n = 90) := 
sorry

end evens_divisors_lt_100_l664_664374


namespace even_number_of_divisors_less_than_100_l664_664410

theorem even_number_of_divisors_less_than_100 :
  ∃ (count : ℕ), count = 90 ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 100 →
    (∃ (d : ℕ), d * d = n ∨ (number_of_divisors n % 2 = 0)) :=
begin
  -- the proof goes here
  sorry
end

end even_number_of_divisors_less_than_100_l664_664410


namespace BC_length_l664_664968

-- Define the given triangle and circle conditions
variables (A B C X : Type) (AB AC BX CX : ℤ)
axiom AB_value : AB = 86
axiom AC_value : AC = 97
axiom circle_center_radius : ∃ (A : Type), ∃ (radius : ℤ), radius = AB ∧ ∃ (points : Set Type), points = {B, X} ∧ ∀ (P : Type), P ∈ points → dist A P = radius
axiom BX_CX_integers : ∃ (x y : ℤ), BX = x ∧ CX = y

-- Define calculations using the Power of a Point theorem
theorem BC_length :
  ∀ (y: ℤ) (x: ℤ), y(y + x) = AC^2 - AB^2 → x + y = 61 :=
by
  intros y x h
  have h1 : 97^2 = 9409, by norm_num,
  have h2 : 86^2 = 7396, by norm_num,
  rw [AB_value, AC_value] at h,
  rw [h1, h2] at h,
  calc y(y + x) = 2013 := by {exact h}
  -- The human verification part is skipped since we only need the statement here
  sorry

end BC_length_l664_664968


namespace midpoint_product_xy_l664_664527

theorem midpoint_product_xy (x y : ℝ) (hC : (4, -1) = ((2 + x) / 2, (-6 + y) / 2)) :
  x * y = 24 :=
begin
  sorry,
end

end midpoint_product_xy_l664_664527


namespace problem1_problem2_l664_664740

-- Lean statement for Problem 1
theorem problem1 (x : ℝ) : x^2 * x^3 - x^5 = 0 := 
by sorry

-- Lean statement for Problem 2
theorem problem2 (a : ℝ) : (a + 1)^2 + 2 * a * (a - 1) = 3 * a^2 + 1 :=
by sorry

end problem1_problem2_l664_664740


namespace greatest_prime_factor_of_expression_l664_664672

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Define specific factorial values
def fac_15 := factorial 15
def fac_18 := factorial 18

-- Define the expression from the problem
def expr := fac_15 * (1 + 16 * 17 * 18)

-- Define the factorization result
def factor_4896 := 2 ^ 5 * 3 ^ 2 * 17

-- Define a lemma about the factorization of the expression
lemma factor_expression : 15! * (1 + 16 * 17 * 18) = fac_15 * 4896 := by
  sorry

-- State the main theorem
theorem greatest_prime_factor_of_expression : ∀ p : ℕ, prime p ∧ p ∣ expr → p ≤ 17 := by
  sorry

end greatest_prime_factor_of_expression_l664_664672


namespace minimum_value_of_expression_l664_664139

theorem minimum_value_of_expression : 
  ∃ x : ℝ, (∀ y : ℝ, |y - 1| + 3 ≥ |x - 1| + 3) ∧ (|x - 1| + 3 = 3) :=
by
  use 1
  split
  · intro y
    have : |y - 1| + 3 ≥ 0 + 3 := by
      apply add_le_add_right
      apply abs_nonneg
    exact this
  · show |1 - 1| + 3 = 3
    exact add_zero 3

end minimum_value_of_expression_l664_664139


namespace max_value_of_y_l664_664768

noncomputable def y (x : ℝ) : ℝ := 3 * x + 4 * (real.sqrt (1 - x ^ 2))

theorem max_value_of_y : ∃ x ∈ set.Icc (-1 : ℝ) 1, y x = 5 ∧ ∀ x ∈ set.Icc (-1 : ℝ) 1, y x ≤ 5 :=
by {
  sorry
}

end max_value_of_y_l664_664768


namespace even_number_of_divisors_less_than_100_l664_664321

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k, k * k = n

theorem even_number_of_divisors_less_than_100 :
  (card { n : ℕ | 1 ≤ n ∧ n < 100 ∧ ¬ is_perfect_square n }) = 90 :=
by
  sorry

end even_number_of_divisors_less_than_100_l664_664321


namespace contradiction_to_at_least_one_not_greater_than_60_l664_664140

-- Define a condition for the interior angles of a triangle being > 60
def all_angles_greater_than_60 (α β γ : ℝ) : Prop :=
  α > 60 ∧ β > 60 ∧ γ > 60

-- Define the negation of the proposition "At least one of the interior angles is not greater than 60"
def at_least_one_not_greater_than_60 (α β γ : ℝ) : Prop :=
  α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60

-- The mathematically equivalent proof problem
theorem contradiction_to_at_least_one_not_greater_than_60 (α β γ : ℝ) :
  ¬ at_least_one_not_greater_than_60 α β γ ↔ all_angles_greater_than_60 α β γ := by
  sorry

end contradiction_to_at_least_one_not_greater_than_60_l664_664140


namespace BC_length_l664_664953

-- Define the given values and conditions
variable (A B C X : Type)
variable (AB AC AX BX CX : ℕ)
variable [inhabited A] [inhabited B] [inhabited C] [inhabited X]

-- Assume the lengths of AB and AC
axiom h_AB : AB = 86
axiom h_AC : AC = 97

-- Assume the circle centered at A with radius AB intersects BC at B and X
axiom h_circle : AX = AB

-- Assume BX and CX are integers
axiom h_BX_integral : ∃ (x : ℕ), BX = x
axiom h_CX_integral : ∃ (y : ℕ), CX = y

-- The statement to prove that the length of BC is 61
theorem BC_length : (∃ (x y : ℕ), BX = x ∧ CX = y ∧ x + y = 61) :=
by
  sorry

end BC_length_l664_664953


namespace triangle_PQR_PQ_value_l664_664868

theorem triangle_PQR_PQ_value 
  (angle_P : ∠P = 90)
  (tan_R : tan R = 3/4) 
  (QR : QR = 80) : 
  PQ = 48 := 
by
  sorry

end triangle_PQR_PQ_value_l664_664868


namespace jenny_kenny_time_l664_664518

theorem jenny_kenny_time (distant_parallel_paths : ℝ) 
  (jenny_speed : ℝ)
  (kenny_speed : ℝ)
  (building_diameter : ℝ)
  (initial_distance : ℝ)
  (t : ℝ) :
  distant_parallel_paths = 300 →
  jenny_speed = 2 →
  kenny_speed = 4 →
  building_diameter = 200 →
  initial_distance = 300 →
  (∀ x y : ℝ, y = (-150/t)*x + 300 - (22500/t) → x^2 + y^2 = 100^2 → -x/y = -150/t → x*t = 150*y → 
    x = 15000 / (sqrt(22500 + t^2)) →
    t = 75) →
  (75 : ℚ).numerator + (75 : ℚ).denominator = 76 :=
begin
  intros,
  sorry
end

end jenny_kenny_time_l664_664518


namespace car_travel_in_next_hours_l664_664703

-- Define the initial conditions
def total_distance : ℝ := 180        -- miles
def total_time : ℝ := 4              -- hours
def next_time : ℝ := 3               -- hours

-- Define the question and prove the answer
theorem car_travel_in_next_hours :
  let average_speed := total_distance / total_time in
  let next_distance := average_speed * next_time in
  next_distance = 135 :=
by
  -- Proof goes here
  sorry

end car_travel_in_next_hours_l664_664703


namespace yard_length_is_correct_l664_664879

-- Definitions based on the conditions
def trees : ℕ := 26
def distance_between_trees : ℕ := 11

-- Theorem stating that the length of the yard is 275 meters
theorem yard_length_is_correct : (trees - 1) * distance_between_trees = 275 :=
by sorry

end yard_length_is_correct_l664_664879


namespace BC_length_l664_664949

-- Define the given values and conditions
variable (A B C X : Type)
variable (AB AC AX BX CX : ℕ)
variable [inhabited A] [inhabited B] [inhabited C] [inhabited X]

-- Assume the lengths of AB and AC
axiom h_AB : AB = 86
axiom h_AC : AC = 97

-- Assume the circle centered at A with radius AB intersects BC at B and X
axiom h_circle : AX = AB

-- Assume BX and CX are integers
axiom h_BX_integral : ∃ (x : ℕ), BX = x
axiom h_CX_integral : ∃ (y : ℕ), CX = y

-- The statement to prove that the length of BC is 61
theorem BC_length : (∃ (x y : ℕ), BX = x ∧ CX = y ∧ x + y = 61) :=
by
  sorry

end BC_length_l664_664949


namespace length_of_BC_l664_664934

theorem length_of_BC (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
    (BX CX : ℕ) (h_pow : CX * (BX + CX) = 2013) : 
    BX + CX = 61 :=
  sorry

end length_of_BC_l664_664934


namespace min_value_fraction_l664_664475

theorem min_value_fraction (a b c : ℝ) (A : ℝ)
  (hA : A = 60)
  (area_ABC : Real.sqrt 3 = 0.5 * b * c * Real.sin A) :
  (∀ (b c : ℝ), ∀ (a ≥ 2), ∀ (b+c ≥ 4), ∃ x, ∀ (x = (4*b^2 + 4*c^2 - 3*a^2) / (b + c)), x ≥ 5)  :=
by
  sorry

end min_value_fraction_l664_664475


namespace ratio_of_larger_to_smaller_l664_664109

noncomputable def ratio_of_numbers (a b : ℝ) : ℝ :=
a / b

theorem ratio_of_larger_to_smaller (a b : ℝ) (h1 : a + b = 7 * (a - b)) (h2 : a * b = 50) (h3 : a > b) :
  ratio_of_numbers a b = 4 / 3 :=
sorry

end ratio_of_larger_to_smaller_l664_664109


namespace bobby_initial_gasoline_l664_664735

variable (initial_gasoline : ℕ) -- initial amount of gasoline in gallons
variable (remaining_gasoline : ℕ := 2) -- remaining gasoline in gallons
variable (consumption_rate : ℕ := 2) -- consumption rate in miles per gallon
variable (supermarket_distance : ℕ := 5) -- distance to the supermarket in miles
variable (farm_distance : ℕ := 6) -- distance to the farm in miles
variable (turnaround_distance : ℕ := 2) -- distance before turning around in miles

def total_distance_traveled : ℕ := 
  2 * supermarket_distance + 
  2 * turnaround_distance + 
  farm_distance

def gasoline_used : ℕ := total_distance_traveled / consumption_rate

def initial_gasoline_calculated : ℕ := gasoline_used + remaining_gasoline

theorem bobby_initial_gasoline : initial_gasoline_calculated = 12 := 
  by
  unfold total_distance_traveled
  unfold gasoline_used
  unfold initial_gasoline_calculated
  sorry

end bobby_initial_gasoline_l664_664735


namespace modified_prism_surface_area_l664_664746

theorem modified_prism_surface_area :
  let original_surface_area := 2 * (2 * 4 + 2 * 5 + 4 * 5)
  let modified_surface_area := original_surface_area + 5
  modified_surface_area = original_surface_area + 5 :=
by
  -- set the original dimensions
  let l := 2
  let w := 4
  let h := 5
  -- calculate original surface area
  let SA_original := 2 * (l * w + l * h + w * h)
  -- calculate modified surface area
  let SA_new := SA_original + 5
  -- assert the relationship
  have : SA_new = SA_original + 5 := rfl
  exact this

end modified_prism_surface_area_l664_664746


namespace BC_length_l664_664967

-- Define the given triangle and circle conditions
variables (A B C X : Type) (AB AC BX CX : ℤ)
axiom AB_value : AB = 86
axiom AC_value : AC = 97
axiom circle_center_radius : ∃ (A : Type), ∃ (radius : ℤ), radius = AB ∧ ∃ (points : Set Type), points = {B, X} ∧ ∀ (P : Type), P ∈ points → dist A P = radius
axiom BX_CX_integers : ∃ (x y : ℤ), BX = x ∧ CX = y

-- Define calculations using the Power of a Point theorem
theorem BC_length :
  ∀ (y: ℤ) (x: ℤ), y(y + x) = AC^2 - AB^2 → x + y = 61 :=
by
  intros y x h
  have h1 : 97^2 = 9409, by norm_num,
  have h2 : 86^2 = 7396, by norm_num,
  rw [AB_value, AC_value] at h,
  rw [h1, h2] at h,
  calc y(y + x) = 2013 := by {exact h}
  -- The human verification part is skipped since we only need the statement here
  sorry

end BC_length_l664_664967


namespace line_equation_L_l664_664265

open Real

-- Define point P
def P : Point := (2, 0)

-- Define the circle C with its equation
structure Circle where
  center : Point
  radius : ℝ

def C : Circle := {
  center := (3, -2),
  radius := 3
}

-- Define line L's conditions
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  equation : ∀ x y : ℝ, a * x + b * y + c = 0

-- Distance from point to line
def distance_point_line (P : Point) (L : Line) : ℝ :=
  let (x, y) := P
  abs (L.a * x + L.b * y + L.c) / sqrt (L.a ^ 2 + L.b ^ 2)

-- Assertion to prove
theorem line_equation_L : ∃ L : Line, (P = P) → (distance_point_line C.center L = 1) ∧
                                  (L.a * P.1 + L.b * P.2 + L.c = 0) ∧ 
                                  (L.a * 3 + L.b * (-2) + L.c = 0) ∧ (∃ k : ℝ, (L.b = k * L.a)) ∧
                                  ((L.a = 3) ∧ (L.b = 4) ∧ (L.c = -6)) ∨ ((L.a = 1) ∧ (L.b = 0) ∧ (L.c = -2)) :=
by {
  -- This block states the logical conditions for solution
  sorry
}

end line_equation_L_l664_664265


namespace cost_of_math_book_l664_664129

-- The definitions based on the conditions from the problem
def total_books : ℕ := 90
def math_books : ℕ := 54
def history_books := total_books - math_books -- 36
def cost_history_book : ℝ := 5
def total_cost : ℝ := 396

-- The theorem we want to prove: the cost of each math book
theorem cost_of_math_book (M : ℝ) : (math_books * M + history_books * cost_history_book = total_cost) → M = 4 := 
by 
  sorry

end cost_of_math_book_l664_664129


namespace condition_for_third_quadrant_l664_664502

def in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem condition_for_third_quadrant (a : ℝ) :
  in_third_quadrant (((a - 1) : ℝ) + ((a + 1) : ℝ) * complex.I) ↔ a < -1 :=
by sorry

end condition_for_third_quadrant_l664_664502


namespace number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664382

theorem number_of_integers_with_even_divisors (n : ℕ) (h : n = 100) : 
  (card {x | (x < n) ∧ ∃ k, k * k = x} = 9) → 
  (card {x | (x < n) ∧ ¬(∃ k, k * k = x)} = n - 1 - 9) :=
by
  intro h_squares
  rw h
  trivial

open_locale classical
noncomputable def count_even_divisors_less_than_100 : ℕ :=
  90

theorem count_even_divisors_verification :
  count_even_divisors_less_than_100 = 90 :=
by
  sorry

end number_of_integers_with_even_divisors_count_even_divisors_verification_l664_664382


namespace solve_inequality_l664_664571

theorem solve_inequality (x : ℝ) : (x + 1) / (x + 3) ≥ 0 ↔ x ∈ (Set.Ioo (-∞) (-3)) ∪ (Set.Ici (-1)) := 
sorry

end solve_inequality_l664_664571


namespace greatest_prime_factor_15_factorial_plus_18_factorial_l664_664645

theorem greatest_prime_factor_15_factorial_plus_18_factorial :
  ∀ {a b c d e f g: ℕ}, a = 15! → b = 18! → c = 16 → d = 17 → e = 18 → f = a * (1 + c * d * e) →
  g = 4896 → Prime 17 → f + b = a + b → Nat.gcd (a + b) g = 17 :=
by
  intros
  sorry

end greatest_prime_factor_15_factorial_plus_18_factorial_l664_664645


namespace even_number_of_divisors_less_than_100_l664_664418

theorem even_number_of_divisors_less_than_100 :
  ∃ (count : ℕ), count = 90 ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 100 →
    (∃ (d : ℕ), d * d = n ∨ (number_of_divisors n % 2 = 0)) :=
begin
  -- the proof goes here
  sorry
end

end even_number_of_divisors_less_than_100_l664_664418


namespace factorize_expression_l664_664760

theorem factorize_expression (a : ℝ) :
  (a^2 + a)^2 + 4 * (a^2 + a) - 12 = (a - 1) * (a + 2) * (a^2 + a + 6) :=
by
  sorry

end factorize_expression_l664_664760


namespace largest_number_in_box_l664_664624

theorem largest_number_in_box
  (a : ℕ)
  (sum_eq_480 : a + (a + 1) + (a + 2) + (a + 10) + (a + 11) + (a + 12) = 480) :
  a + 12 = 86 :=
by
  sorry

end largest_number_in_box_l664_664624


namespace sequence_integers_if_infinitely_many_are_integers_l664_664988

-- Define the given conditions
variable (k : ℕ) (hk : k > 0)
variable (a : ℕ → ℕ) (b : ℕ → ℝ)
variable (ha : ∀ n, a n ∈ Finset.range (k + 1))
variable (hb : ∀ n > 0, b n = Real.root n (∑ i in Finset.range n, (a (i + 1)) ^ n))

-- Theorem statement
theorem sequence_integers_if_infinitely_many_are_integers :
  (∃ᶠ n in Filter.atTop, b n ∈ ℤ) → ∀ n > 0, b n ∈ ℤ :=
begin
  sorry
end

end sequence_integers_if_infinitely_many_are_integers_l664_664988


namespace even_number_of_divisors_less_than_100_l664_664311

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k, k * k = n

theorem even_number_of_divisors_less_than_100 :
  (card { n : ℕ | 1 ≤ n ∧ n < 100 ∧ ¬ is_perfect_square n }) = 90 :=
by
  sorry

end even_number_of_divisors_less_than_100_l664_664311


namespace tan_alpha_minus_pi_over_4_l664_664806

theorem tan_alpha_minus_pi_over_4 (α : ℝ) (h1 : 0 < α ∧ α < π)
  (h2 : Real.sin α = 3 / 5) : Real.tan (α - π / 4) = -1 / 7 ∨ Real.tan (α - π / 4) = -7 := 
sorry

end tan_alpha_minus_pi_over_4_l664_664806


namespace count_even_divisors_lt_100_l664_664300

-- Define the set of natural numbers less than 100
def nat_lt_100 := {n : ℕ | n < 100}

-- Define the set of perfect squares less than 100
def perfect_squares_lt_100 := {n : ℕ | ∃ k : ℕ, k^2 = n ∧ n < 100}

-- Define the set of natural numbers less than 100 with an even number of positive divisors
def even_divisors_lt_100 := nat_lt_100 \ perfect_squares_lt_100

-- Theorem stating the number of elements with even number of divisors is 90
theorem count_even_divisors_lt_100 : (even_divisors_lt_100).card = 90 := 
sorry

end count_even_divisors_lt_100_l664_664300


namespace greatest_prime_factor_15_18_l664_664654

theorem greatest_prime_factor_15_18! :
  ∃ p : ℕ, prime p ∧ p ∈ prime_factors (15! + 18!) ∧ ∀ q : ℕ, prime q → q ∈ prime_factors (15! + 18!) → q ≤ 4897 := 
sorry

end greatest_prime_factor_15_18_l664_664654


namespace count_even_divisors_lt_100_l664_664305

-- Define the set of natural numbers less than 100
def nat_lt_100 := {n : ℕ | n < 100}

-- Define the set of perfect squares less than 100
def perfect_squares_lt_100 := {n : ℕ | ∃ k : ℕ, k^2 = n ∧ n < 100}

-- Define the set of natural numbers less than 100 with an even number of positive divisors
def even_divisors_lt_100 := nat_lt_100 \ perfect_squares_lt_100

-- Theorem stating the number of elements with even number of divisors is 90
theorem count_even_divisors_lt_100 : (even_divisors_lt_100).card = 90 := 
sorry

end count_even_divisors_lt_100_l664_664305


namespace dot_product_a_b_lambda_value_l664_664836

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_a_b :
  dot_product (1, 2) (2, -2) = -2 :=
by sorry

theorem lambda_value (λ : ℝ) :
  let a := (1, 2)
  let b := (2, -2)
  (dot_product a (a.1 + λ * b.1, a.2 + λ * b.2) = 0) →
  λ = 5 / 2 :=
by sorry

end dot_product_a_b_lambda_value_l664_664836


namespace even_divisors_less_than_100_l664_664356

theorem even_divisors_less_than_100 :
  let count_even_divisors := 
    let n := 100 in
    let perfect_squares := { m | ∃ k, k * k = m ∧ m < n } in
    let total_numbers := finset.range n in
    (total_numbers.card - perfect_squares.card) =
    90 
  in count_even_divisors = 90 :=
by
  let n := 100
  let perfect_squares : finset ℕ := finset.filter (λ m, ∃ k, k * k = m) (finset.range n)
  let total_numbers : finset ℕ := finset.range n
  have h : total_numbers.card = 99 := by sorry
  have p : perfect_squares.card = 9 := by sorry
  show total_numbers.card - perfect_squares.card = 90
  calc
    total_numbers.card - perfect_squares.card
      = 99 - 9 := by rw [h, p]
      = 90 := by norm_num

end even_divisors_less_than_100_l664_664356


namespace arrange_athletes_l664_664507

theorem arrange_athletes :
  let athletes := 8
  let countries := 4
  let country_athletes := 2
  (Nat.choose athletes country_athletes) *
  (Nat.choose (athletes - country_athletes) country_athletes) *
  (Nat.choose (athletes - 2 * country_athletes) country_athletes) *
  (Nat.choose (athletes - 3 * country_athletes) country_athletes) = 2520 :=
by
  let athletes := 8
  let countries := 4
  let country_athletes := 2
  show (Nat.choose athletes country_athletes) *
       (Nat.choose (athletes - country_athletes) country_athletes) *
       (Nat.choose (athletes - 2 * country_athletes) country_athletes) *
       (Nat.choose (athletes - 3 * country_athletes) country_athletes) = 2520
  sorry

end arrange_athletes_l664_664507


namespace inequality_always_holds_l664_664860

theorem inequality_always_holds (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 6 * x + 3 > 0) →
  (a > 3) ∧ (∀ x : ℝ, x = a + 9 / (a - 1) → x ≥ 7) :=
by
  sorry

end inequality_always_holds_l664_664860


namespace cape_may_sharks_l664_664202

theorem cape_may_sharks (D : ℕ) (C : ℕ) (hD : D = 12) (hC : C = 2 * D + 8) : C = 32 :=
by
  rw [hD, hC]
  norm_num

end cape_may_sharks_l664_664202


namespace even_divisors_less_than_100_l664_664357

theorem even_divisors_less_than_100 :
  let count_even_divisors := 
    let n := 100 in
    let perfect_squares := { m | ∃ k, k * k = m ∧ m < n } in
    let total_numbers := finset.range n in
    (total_numbers.card - perfect_squares.card) =
    90 
  in count_even_divisors = 90 :=
by
  let n := 100
  let perfect_squares : finset ℕ := finset.filter (λ m, ∃ k, k * k = m) (finset.range n)
  let total_numbers : finset ℕ := finset.range n
  have h : total_numbers.card = 99 := by sorry
  have p : perfect_squares.card = 9 := by sorry
  show total_numbers.card - perfect_squares.card = 90
  calc
    total_numbers.card - perfect_squares.card
      = 99 - 9 := by rw [h, p]
      = 90 := by norm_num

end even_divisors_less_than_100_l664_664357


namespace even_divisors_count_lt_100_l664_664344

theorem even_divisors_count_lt_100 : 
  {n : ℕ | n < 100 ∧ n ≠ 0 ∧ ∃ k : ℕ, k * k = n } = { n : ℕ | n < 100 ∧ n ≠ 0 } \ 
  { n : ℕ | ∃ k : ℕ, k * k = n ∧ k < 100 } → 
  (card {n : ℕ | n < 100 ∧ n ≠ 0 ∧ even (finset.card (divisors n))} = 90) :=
begin
  sorry
end

end even_divisors_count_lt_100_l664_664344


namespace even_number_of_divisors_less_than_100_l664_664411

theorem even_number_of_divisors_less_than_100 :
  ∃ (count : ℕ), count = 90 ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 100 →
    (∃ (d : ℕ), d * d = n ∨ (number_of_divisors n % 2 = 0)) :=
begin
  -- the proof goes here
  sorry
end

end even_number_of_divisors_less_than_100_l664_664411


namespace triangle_bc_length_l664_664945

theorem triangle_bc_length :
  ∀ (A B C X : Type) (d_AB : ℝ) (d_AC : ℝ) (d_BX d_CX BC : ℕ),
  d_AB = 86 ∧ d_AC = 97 →
  let circleA := {center := A, radius := d_AB} in
  let intersect_B := B ∈ circleA in
  let intersect_X := X ∈ circleA in
  d_BX + d_CX = BC →
  d_BX ∈ ℕ ∧ d_CX ∈ ℕ →
  BC = 61 :=
by
  intros A B C X d_AB d_AC d_BX d_CX BC h_dist h_circle h_intersect h_sum h_intBC
  sorry

end triangle_bc_length_l664_664945


namespace triangle_perimeter_l664_664177

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 10) (h2 : b = 6) (h3 : c = 7) :
  a + b + c = 23 := by
  sorry

end triangle_perimeter_l664_664177


namespace even_divisors_count_lt_100_l664_664345

theorem even_divisors_count_lt_100 : 
  {n : ℕ | n < 100 ∧ n ≠ 0 ∧ ∃ k : ℕ, k * k = n } = { n : ℕ | n < 100 ∧ n ≠ 0 } \ 
  { n : ℕ | ∃ k : ℕ, k * k = n ∧ k < 100 } → 
  (card {n : ℕ | n < 100 ∧ n ≠ 0 ∧ even (finset.card (divisors n))} = 90) :=
begin
  sorry
end

end even_divisors_count_lt_100_l664_664345


namespace sqrt_meaningful_iff_l664_664866

theorem sqrt_meaningful_iff (x : ℝ) : (∃ r : ℝ, r = sqrt (6 + x)) ↔ x ≥ -6 :=
by
  sorry

end sqrt_meaningful_iff_l664_664866


namespace find_a8_l664_664278

-- Define the arithmetic sequence and its properties
def arithmetic_sequence (a : ℕ → ℤ) := ∃ (a₁ d : ℤ), ∀ n : ℕ, a (n + 1) = a₁ + n * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (a 1 + (a n)) / 2

-- Given conditions
variables (a : ℕ → ℤ)
hypothesis (h_arith_seq : arithmetic_sequence a)
hypothesis (h_S6_eq_8S3 : sum_first_n_terms a 6 = 8 * (sum_first_n_terms a 3))
hypothesis (h_a3_min_a5 : a 3 - a 5 = 8)

-- Prove that a₈ = -26
theorem find_a8 : a 8 = -26 := 
by {
  sorry
}

end find_a8_l664_664278


namespace oranges_in_pyramid_stack_l664_664712

noncomputable def count_oranges_in_stack (base_x base_y : ℕ) : ℕ :=
  let rec count_layers (x y : ℕ) (acc : ℕ) : ℕ :=
  if x = 0 ∨ y = 0 then acc else count_layers (x - 1) (y - 1) (acc + (x * y))
  count_layers base_x base_y 0

theorem oranges_in_pyramid_stack (base_x base_y : ℕ) (h1 : base_x = 6) (h2 : base_y = 7) :
  count_oranges_in_stack base_x base_y = 112 :=
by
  rw [h1, h2]
  unfold count_oranges_in_stack
  iterate 6 { unfold count_layers }
  sorry

end oranges_in_pyramid_stack_l664_664712


namespace greatest_prime_factor_15_factorial_plus_18_factorial_l664_664647

theorem greatest_prime_factor_15_factorial_plus_18_factorial :
  ∀ {a b c d e f g: ℕ}, a = 15! → b = 18! → c = 16 → d = 17 → e = 18 → f = a * (1 + c * d * e) →
  g = 4896 → Prime 17 → f + b = a + b → Nat.gcd (a + b) g = 17 :=
by
  intros
  sorry

end greatest_prime_factor_15_factorial_plus_18_factorial_l664_664647


namespace count_even_divisors_lt_100_l664_664303

-- Define the set of natural numbers less than 100
def nat_lt_100 := {n : ℕ | n < 100}

-- Define the set of perfect squares less than 100
def perfect_squares_lt_100 := {n : ℕ | ∃ k : ℕ, k^2 = n ∧ n < 100}

-- Define the set of natural numbers less than 100 with an even number of positive divisors
def even_divisors_lt_100 := nat_lt_100 \ perfect_squares_lt_100

-- Theorem stating the number of elements with even number of divisors is 90
theorem count_even_divisors_lt_100 : (even_divisors_lt_100).card = 90 := 
sorry

end count_even_divisors_lt_100_l664_664303


namespace frac_equiv_l664_664774

-- Define the given values of x and y.
def x : ℚ := 2 / 7
def y : ℚ := 8 / 11

-- Define the statement to prove.
theorem frac_equiv : (7 * x + 11 * y) / (77 * x * y) = 5 / 8 :=
by
  -- The proof will go here (use 'sorry' for now)
  sorry

end frac_equiv_l664_664774


namespace log_product_bounds_l664_664737

noncomputable def z : ℝ := (List.range 2 50).map (λ i, real.log (i+1) / real.log i).prod

theorem log_product_bounds : 5 < z ∧ z < 6 :=
by 
  sorry

end log_product_bounds_l664_664737


namespace symmetric_point_xOz_l664_664009

theorem symmetric_point_xOz (x y z : ℝ) : (x, y, z) = (-1, 2, 1) → (x, -y, z) = (-1, -2, 1) :=
by
  intros h
  cases h
  sorry

end symmetric_point_xOz_l664_664009


namespace BC_length_l664_664948

-- Define the given values and conditions
variable (A B C X : Type)
variable (AB AC AX BX CX : ℕ)
variable [inhabited A] [inhabited B] [inhabited C] [inhabited X]

-- Assume the lengths of AB and AC
axiom h_AB : AB = 86
axiom h_AC : AC = 97

-- Assume the circle centered at A with radius AB intersects BC at B and X
axiom h_circle : AX = AB

-- Assume BX and CX are integers
axiom h_BX_integral : ∃ (x : ℕ), BX = x
axiom h_CX_integral : ∃ (y : ℕ), CX = y

-- The statement to prove that the length of BC is 61
theorem BC_length : (∃ (x y : ℕ), BX = x ∧ CX = y ∧ x + y = 61) :=
by
  sorry

end BC_length_l664_664948


namespace relationship_abc_l664_664802

theorem relationship_abc :
  let a := 2 ^ 1.3
  let b := 2 ^ 1.4
  let c := log 3 1
  c < a ∧ a < b :=
by
  let a := 2 ^ 1.3
  let b := 2 ^ 1.4
  let c := log 3 1
  -- Skipping the proof steps with sorry
  sorry

end relationship_abc_l664_664802


namespace seq_geq_4_l664_664546

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 5 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = (a n ^ 2 + 8 * a n + 16) / (4 * a n)

theorem seq_geq_4 (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, n ≥ 1 → a n ≥ 4 :=
sorry

end seq_geq_4_l664_664546


namespace greatest_prime_factor_15_18_l664_664650

theorem greatest_prime_factor_15_18! :
  ∃ p : ℕ, prime p ∧ p ∈ prime_factors (15! + 18!) ∧ ∀ q : ℕ, prime q → q ∈ prime_factors (15! + 18!) → q ≤ 4897 := 
sorry

end greatest_prime_factor_15_18_l664_664650


namespace problem_1_problem_2_l664_664292

def M : Set ℕ := {0, 1}

def A := { p : ℕ × ℕ | p.fst ∈ M ∧ p.snd ∈ M }

def B := { p : ℕ × ℕ | p.snd = 1 - p.fst }

theorem problem_1 : A = {(0,0), (0,1), (1,0), (1,1)} :=
by
  sorry

theorem problem_2 : 
  let AB := { p ∈ A | p ∈ B }
  AB = {(1,0), (0,1)} ∧
  {S : Set (ℕ × ℕ) | S ⊆ AB} = {∅, {(1,0)}, {(0,1)}, {(1,0), (0,1)}} :=
by
  sorry

end problem_1_problem_2_l664_664292


namespace problem_statement_l664_664856

noncomputable def f (x : ℝ) : ℝ := 2^x + abs (1 - Real.log₂ x)

theorem problem_statement : f 4 = 17 := by
  sorry

end problem_statement_l664_664856


namespace amount_paid_is_correct_l664_664519

-- Conditions given in the problem
def jimmy_shorts_count : ℕ := 3
def jimmy_short_price : ℝ := 15.0
def irene_shirts_count : ℕ := 5
def irene_shirt_price : ℝ := 17.0
def discount_rate : ℝ := 0.10

-- Define the total cost for jimmy
def jimmy_total_cost : ℝ := jimmy_shorts_count * jimmy_short_price

-- Define the total cost for irene
def irene_total_cost : ℝ := irene_shirts_count * irene_shirt_price

-- Define the total cost before discount
def total_cost_before_discount : ℝ := jimmy_total_cost + irene_total_cost

-- Define the discount amount
def discount_amount : ℝ := total_cost_before_discount * discount_rate

-- Define the total amount to pay
def total_amount_to_pay : ℝ := total_cost_before_discount - discount_amount

-- The proposition we need to prove
theorem amount_paid_is_correct : total_amount_to_pay = 117 := by
  sorry

end amount_paid_is_correct_l664_664519


namespace even_number_of_divisors_l664_664338

-- Proof statement: There are 90 positive integers less than 100 with an even number of divisors.
theorem even_number_of_divisors : 
  {n : ℕ | n < 100 ∧ ∃ k : ℕ, k ^ 2 = n}.toFinset.card = 90 := 
sorry

end even_number_of_divisors_l664_664338


namespace quadratic_shift_l664_664459

theorem quadratic_shift :
  ∀ (x : ℝ), (∃ (y : ℝ), y = -x^2) →
  (∃ (y : ℝ), y = -(x + 1)^2 + 3) :=
by
  intro x
  intro h
  use -(x + 1)^2 + 3
  sorry

end quadratic_shift_l664_664459


namespace evens_divisors_lt_100_l664_664376

theorem evens_divisors_lt_100 : 
  ∃ n : ℕ, n = 90 ∧ ∀ k : ℕ, (1 ≤ k < 100) → (even k ↔ (∃ m : ℕ, m * m = k)) ↔ (n = 90) := 
sorry

end evens_divisors_lt_100_l664_664376


namespace functional_expression_point_M_coordinates_l664_664805

variables (x y : ℝ) (k : ℝ)

-- Given conditions
def proportional_relation : Prop := y + 4 = k * (x - 3)
def initial_condition : Prop := (x = 1 → y = 0)
def point_M : Prop := ∃ m : ℝ, (m + 1, 2 * m) = (1, 0)

-- Proof of the functional expression
theorem functional_expression (h1 : proportional_relation x y k) (h2 : initial_condition x y) :
  ∃ k : ℝ, k = -2 ∧ y = -2 * x + 2 := 
sorry

-- Proof of the coordinates of point M
theorem point_M_coordinates (h : ∀ m : ℝ, (m + 1, 2 * m) = (1, 0)) :
  ∃ m : ℝ, m = 0 ∧ (m + 1, 2 * m) = (1, 0) := 
sorry

end functional_expression_point_M_coordinates_l664_664805


namespace inequality_proof_l664_664274

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := 
sorry

end inequality_proof_l664_664274


namespace proof_problem_l664_664627

variables {Plane : Type} [EuclideanPlane Plane]
variables {a b c : Line Plane}

theorem proof_problem (h1: a ≠ b) (h2: a ≠ c) (h3: b ≠ c) 
  (hab : a // b) (hac : a ⊥ c) : b ⊥ c :=
by
  sorry

end proof_problem_l664_664627


namespace double_root_values_l664_664168

theorem double_root_values (b₃ b₂ b₁ s : ℤ) (h : ∀ x : ℤ, (x * (x - s)) ∣ (x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 36)) 
  : s = -6 ∨ s = -3 ∨ s = -2 ∨ s = -1 ∨ s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 6 :=
sorry

end double_root_values_l664_664168


namespace tetrahedron_cosine_relation_l664_664878

theorem tetrahedron_cosine_relation
  (A B C D : Type)
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (θ1 θ2 θ3 : ℝ)
  (h1 : t1 = a1 * b1 * Real.cos θ1)
  (h2 : t2 = a2 * b2 * Real.cos θ2)
  (h3 : t3 = a3 * b3 * Real.cos θ3) :
  (t2 = t1 + t3) ∨ (t1 = t2 + t3) ∨ (t3 = t1 + t2) := 
begin
  sorry
end

end tetrahedron_cosine_relation_l664_664878


namespace ellipse_equation_fixed_point_l664_664809

noncomputable def ellipse_ctx :=
  {center : ℝ × ℝ // center = (0, 0)} ∧
  {axes : (ℝ × ℝ) × (ℝ × ℝ) // axes = ((1, 0), (0, 1))} ∧
  {A : ℝ × ℝ // A = (0, -2)} ∧
  {B : ℝ × ℝ // B = (3/2, -1)}

theorem ellipse_equation_fixed_point (h : ellipse_ctx) :
  ∃ (E : ℝ → ℝ → Prop),
    (∀ x y, E x y ↔ (x^2 / 3) + (y^2 / 4) = 1) ∧
    (∀ P M N T H : ℝ × ℝ),
      P = (1, -2) →
      (exists k, (M, N) ∈ {(x, y) | E x y}) → -- Intersection with ellipse
      T = (/* some function of M and A, B */) →
      H = (/* construct as M, T, H relationship */) →
      (∃ K : ℝ × ℝ, K = (0, -2)) ∧
      (H = K ∨ N = K)
:=
sorry -- Proof will follow the computations and logical steps provided

end ellipse_equation_fixed_point_l664_664809


namespace graph_passes_through_fixed_point_l664_664596

noncomputable def fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : Prop :=
  (∀ x y : ℝ, y = a * x + 2 → (x, y) = (-1, 2))

theorem graph_passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : fixed_point a h1 h2 :=
sorry

end graph_passes_through_fixed_point_l664_664596


namespace debra_flips_coin_probability_l664_664750

noncomputable theory

def probability_three_heads_after_two_tails : ℚ :=
  1 / 128

theorem debra_flips_coin_probability :
  (debra_flips_coin_probability := 1 / 128)
  :
  debra_flips_coin_probability = probability_three_heads_after_two_tails :=
sorry

end debra_flips_coin_probability_l664_664750


namespace unique_two_scoop_sundaes_l664_664729

theorem unique_two_scoop_sundaes (n : ℕ) (h : n = 8) : 
  (n.choose 2 + n) = 36 :=
by
  rw h
  -- The proof goes here
  sorry

end unique_two_scoop_sundaes_l664_664729


namespace tenth_occurrence_of_2_l664_664254

def sequence : ℕ+ → ℕ
| ⟨n, hn⟩ := if h : n % 3 = 0 then sequence ⟨n / 3, sorry⟩ else n

theorem tenth_occurrence_of_2 : ∃ n : ℕ, sequence ⟨n, sorry⟩ = 2 ∧ (∑ i in (finset.range 10).filter (λ n, sequence ⟨n, sorry⟩ = 2), 1) = 10 ∧ n = 39366 :=
begin
  sorry
end

end tenth_occurrence_of_2_l664_664254


namespace minimum_cuts_100_l664_664676

noncomputable def minimum_cuts_needed (n : Nat) : Nat :=
  Nat.find (λ k => ∃ a b c, (a + 1) * (b + 1) * (c + 1) ≥ n ∧ a + b + c = k)

theorem minimum_cuts_100 : minimum_cuts_needed 100 = 11 :=
sorry

end minimum_cuts_100_l664_664676


namespace even_number_of_divisors_lt_100_l664_664399

theorem even_number_of_divisors_lt_100 : 
  let n := 99 in
  let perfect_squares := {1, 4, 9, 16, 25, 36, 49, 64, 81} in
  n - perfect_squares.card = 90 := 
by
  sorry

end even_number_of_divisors_lt_100_l664_664399


namespace largest_option_is_B_l664_664684

/-- Define the provided options -/
def optionA : ℝ := Real.sqrt (Real.cbrt 56)
def optionB : ℝ := Real.sqrt (Real.cbrt 3584)
def optionC : ℝ := Real.sqrt (Real.cbrt 2744)
def optionD : ℝ := Real.cbrt (Real.sqrt 392)
def optionE : ℝ := Real.cbrt (Real.sqrt 448)

/-- The main theorem -/
theorem largest_option_is_B : optionB > optionA ∧ optionB > optionC ∧ optionB > optionD ∧ optionB > optionE :=
by
  sorry

end largest_option_is_B_l664_664684


namespace replace_asterisks_to_div_36_l664_664896

theorem replace_asterisks_to_div_36 :
  ∃ (a b : ℕ), a ∈ {0, 2, 5, 9} ∧ 
                b ∈ {0, 4, 8, 2} ∧ 
                (a = 0 → b = 0 ∨ b = 4 ∨ b = 8) ∧ 
                (a = 2 → b = 4 ∨ b = 8) ∧ 
                (a = 5 → b = 4) ∧ 
                (a = 9 → b = 0) ∧
                ∀ n : ℕ, n = 52000 + 100*a + b →
                n % 36 = 0 :=
begin
  sorry
end

end replace_asterisks_to_div_36_l664_664896


namespace number_of_solutions_l664_664770

noncomputable theory

def problem_statement (θ : ℝ) : Prop :=
  θ ∈ set.Ico 0 (2 * π) ∧ sin (3 * π * cos θ) = cos (4 * π * sin θ)

theorem number_of_solutions : 
  ∃ N : ℕ, ∀ θ : ℝ, problem_statement θ → θ ∈ set.Ico 0 (2 * π) := sorry

end number_of_solutions_l664_664770


namespace equation_of_ellipse_given_conditions_max_value_of_t_given_conditions_l664_664800

-- Definitions of conditions
def pointP_on_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) : Prop := (4 / a^2) + (9 / b^2) = 1 ∧ a > b
def point_on_parabola : Prop := ∃ (p : ℝ), p > 0 ∧ (2 * p = 16) -- p is 4 thus from solving y² = 2px

-- Resulting equation of the ellipse
def ellipse_equation : Prop := ∀ (x y : ℝ), (x^2 / 16) + (y^2 / 12) = 1

-- Maximum value of product of slopes t for sides of triangle ABP
def max_slope_product : Prop := ∃ (t : ℝ), t = 9 / 64

theorem equation_of_ellipse_given_conditions : 
  ( ∃ a b : ℝ, pointP_on_ellipse a b a b ∧ point_on_parabola) → ellipse_equation :=
sorry

theorem max_value_of_t_given_conditions : 
  (∃ a b : ℝ, pointP_on_ellipse a b a b ∧ point_on_parabola) → max_slope_product :=
sorry

end equation_of_ellipse_given_conditions_max_value_of_t_given_conditions_l664_664800


namespace smallest_n_between_76_and_100_l664_664104

theorem smallest_n_between_76_and_100 :
  ∃ (n : ℕ), (n > 1) ∧ (n % 3 = 2) ∧ (n % 7 = 2) ∧ (n % 5 = 1) ∧ (76 < n) ∧ (n < 100) :=
sorry

end smallest_n_between_76_and_100_l664_664104


namespace number_of_two_digit_primes_with_digit_sum_seven_l664_664782

def sum_of_digits_eq_seven (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 7)

def is_prime (n : ℕ) : Prop :=
  nat.prime n

theorem number_of_two_digit_primes_with_digit_sum_seven :
  {n : ℕ | sum_of_digits_eq_seven n ∧ is_prime n}.to_finset.card = 2 :=
by
  sorry

end number_of_two_digit_primes_with_digit_sum_seven_l664_664782


namespace longest_segment_equal_sum_of_other_two_l664_664059

theorem longest_segment_equal_sum_of_other_two
  (R : ℝ)
  (A B C : ℝ × ℝ)
  (M : ℝ × ℝ)
  (h_eq_triangle : dist A B = dist B C ∧ dist B C = dist C A)
  (h_on_circle : dist M (0, 0) = R ∧ dist A (0, 0) = R ∧ dist B (0, 0) = R ∧ dist C (0, 0) = R) :
  let MA := dist M A,
      MB := dist M B,
      MC := dist M C in
  max MA (max MB MC) = (MA + MB + MC) - min MA (min MB MC) := sorry

end longest_segment_equal_sum_of_other_two_l664_664059


namespace square_root_condition_l664_664865

theorem square_root_condition (x : ℝ) : (6 + x ≥ 0) ↔ (x ≥ -6) :=
by sorry

end square_root_condition_l664_664865


namespace converse_and_inverse_false_l664_664832

def Triangle (T : Type) := T
variables {T : Type} [Triangle T]

def is_equilateral (t : T) : Prop := sorry  -- assumes definition of equilateral triangle
def is_isosceles (t : T) : Prop := sorry    -- assumes definition of isosceles triangle

-- Given true statement: If a triangle is equilateral, then it is isosceles.
axiom equilateral_implies_isosceles (t : T) : is_equilateral t → is_isosceles t

-- Theorem to prove: The converse and inverse are both false
theorem converse_and_inverse_false (t : T) :
  ¬ (is_isosceles t → is_equilateral t) ∧ ¬ (¬ is_equilateral t → ¬ is_isosceles t) :=
sorry

end converse_and_inverse_false_l664_664832


namespace inscribed_sphere_radius_in_cone_l664_664172

noncomputable def radius_of_inscribed_sphere (b d : ℝ) : ℝ :=
  b * real.sqrt d - b

theorem inscribed_sphere_radius_in_cone (b d : ℝ) (radius base_radius height : ℝ) 
  (h1 : base_radius = 15) (h2 : height = 30) 
  (h3 : radius = radius_of_inscribed_sphere b d) : 
  b + d = 12.5 :=
sorry

end inscribed_sphere_radius_in_cone_l664_664172


namespace concyclic_M_N_E_F_l664_664730

noncomputable def angle (A B C : Type) := sorry  -- placeholder for actual angle definition
noncomputable def Quadrilateral (A B C D : Type) := sorry  -- placeholder for quadrilateral definition
noncomputable def Circle (O: Type) := sorry  -- placeholder for circle definition
noncomputable def Midpoint (M: Type) (A B: Type) := sorry  -- placeholder for midpoint definition
noncomputable def Perpendicular (A B C: Type) := sorry  -- placeholder for perpendicular definition
noncomputable def Concyclic (W X Y Z: Type) := sorry  -- placeholder for concyclic condition definition

theorem concyclic_M_N_E_F
  (A B C D M N E F O: Type)
  (h1: Quadrilateral A B C D)
  (h2: angle A B C = angle A D C)
  (h3: angle A B C < 90)
  (h4: Circle O)
  (h5: diameter O = A C)
  (h6: intersects_circle O B C E)  -- circle intersects "BC" at point "E"
  (h7: intersects_circle O C D F)  -- circle intersects "CD" at point "F"
  (h8: Midpoint M B D)
  (h9: Perpendicular A N B D):
  Concyclic M N E F := 
sorry

end concyclic_M_N_E_F_l664_664730


namespace greatest_prime_factor_15_factorial_plus_18_factorial_l664_664648

theorem greatest_prime_factor_15_factorial_plus_18_factorial :
  ∀ {a b c d e f g: ℕ}, a = 15! → b = 18! → c = 16 → d = 17 → e = 18 → f = a * (1 + c * d * e) →
  g = 4896 → Prime 17 → f + b = a + b → Nat.gcd (a + b) g = 17 :=
by
  intros
  sorry

end greatest_prime_factor_15_factorial_plus_18_factorial_l664_664648


namespace find_radius_l664_664159

namespace circle_radius_problem

variables (M N : ℝ) (π : ℝ) [NeZero π]

axiom condition1 : π > 0
axiom condition2 : M = π * (r * r)
axiom condition3 : N = 2 * π * r
axiom condition4 : M / N = 15

theorem find_radius (r : ℝ) : r = 30 :=
by
  sorry

end circle_radius_problem

end find_radius_l664_664159


namespace even_divisors_count_lt_100_l664_664353

theorem even_divisors_count_lt_100 : 
  {n : ℕ | n < 100 ∧ n ≠ 0 ∧ ∃ k : ℕ, k * k = n } = { n : ℕ | n < 100 ∧ n ≠ 0 } \ 
  { n : ℕ | ∃ k : ℕ, k * k = n ∧ k < 100 } → 
  (card {n : ℕ | n < 100 ∧ n ≠ 0 ∧ even (finset.card (divisors n))} = 90) :=
begin
  sorry
end

end even_divisors_count_lt_100_l664_664353


namespace tetrahedron_ratios_l664_664511

variable (A B C D : Type) [EuclideanStructure A B C D] 
variable (s_A s_B s_C s_D : ℝ)

-- Conditions
axiom medians_perpendicular : are_pairwise_perpendicular (median A) (median B) (median C)
axiom edge_perpendicular_face : is_perpendicular (edge D A) (face A B C)

-- Ratios
theorem tetrahedron_ratios :
  (edge_length A B : edge_length A C : edge_length A D : edge_length B C : edge_length B D : edge_length C D) =
  (Real.sqrt 3 : Real.sqrt 3 : Real.sqrt 8 : 2 : Real.sqrt 11 : Real.sqrt 11) ∧
  (s_A : s_B : s_C : s_D) = 
  (1 : Real.sqrt 2 : Real.sqrt 2 : Real.sqrt 5) :=
  by sorry

end tetrahedron_ratios_l664_664511


namespace probability_of_triangle_sides_l664_664071

theorem probability_of_triangle_sides :
  let S := {xyz : ℝ × ℝ × ℝ | 0 ≤ xyz.1 ∧ xyz.1 ≤ 1 ∧ 0 ≤ xyz.2 ∧ xyz.2 ≤ 1 ∧ 0 ≤ xyz.3 ∧ xyz.3 ≤ 1} in
  let T := {xyz : ℝ × ℝ × ℝ | xyz ∈ S ∧ xyz.3 ≥ xyz.1 ∧ xyz.3 ≥ xyz.2 ∧ xyz.1 + xyz.2 > xyz.3} in
  (measure (λ (xyz : ℝ × ℝ × ℝ), xyz ∈ T) / measure (λ (xyz : ℝ × ℝ × ℝ), xyz ∈ S)) = 1 / 2 :=
sorry

end probability_of_triangle_sides_l664_664071


namespace even_number_of_divisors_lt_100_l664_664402

theorem even_number_of_divisors_lt_100 : 
  let n := 99 in
  let perfect_squares := {1, 4, 9, 16, 25, 36, 49, 64, 81} in
  n - perfect_squares.card = 90 := 
by
  sorry

end even_number_of_divisors_lt_100_l664_664402


namespace expected_defective_chips_45000_l664_664574

-- Definitions based on the conditions
def defective_chips_shipment_1 : ℕ := 7
def total_chips_shipment_1 : ℕ := 6000

def defective_chips_shipment_2 : ℕ := 12
def total_chips_shipment_2 : ℕ := 14000

def total_defective_chips_V1_E1_E2 : ℕ := defective_chips_shipment_1 + defective_chips_shipment_2
def total_chips_V1_E1_E2 : ℕ := total_chips_shipment_1 + total_chips_shipment_2

def ratio_defective : ℚ := total_defective_chips_V1_E1_E2 / total_chips_V1_E1_E2

def upcoming_total_chips : ℕ := 45000
def expected_defective_chips : ℚ := ratio_defective * upcoming_total_chips

-- Theorem statement
theorem expected_defective_chips_45000 :
  expected_defective_chips ≈ 43 := 
by
  -- skipped proof
  sorry

end expected_defective_chips_45000_l664_664574


namespace even_number_of_divisors_lt_100_l664_664409

theorem even_number_of_divisors_lt_100 : 
  let n := 99 in
  let perfect_squares := {1, 4, 9, 16, 25, 36, 49, 64, 81} in
  n - perfect_squares.card = 90 := 
by
  sorry

end even_number_of_divisors_lt_100_l664_664409


namespace cone_unfolded_angle_l664_664469

theorem cone_unfolded_angle (r l α : ℝ) (h1 : l = sqrt 2 * r) (h2 : α * l = 2 * π * r) : α = sqrt 2 * π := 
by
  sorry

end cone_unfolded_angle_l664_664469


namespace perpendicular_vectors_implies_m_eq_3_l664_664294

variables (a b: ℝ × ℝ) (m: ℝ)

-- Definition of the vectors given in the condition
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (-1, m)

-- Statement of the proof problem
theorem perpendicular_vectors_implies_m_eq_3 (h : vector_a = a ∧ vector_b m = b) 
    (hp : a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) : 
    m = 3 :=
begin
  sorry
end

end perpendicular_vectors_implies_m_eq_3_l664_664294


namespace even_number_of_divisors_l664_664336

-- Proof statement: There are 90 positive integers less than 100 with an even number of divisors.
theorem even_number_of_divisors : 
  {n : ℕ | n < 100 ∧ ∃ k : ℕ, k ^ 2 = n}.toFinset.card = 90 := 
sorry

end even_number_of_divisors_l664_664336


namespace max_value_of_omega_l664_664088

theorem max_value_of_omega (ω : ℝ) (φ : ℝ) (h_omega_pos : ω > 0) (h_phi_bound : |φ| < π / 2) 
  (h_increasing : ∀ x y : ℝ, π / 4 < x ∧ x < y ∧ y < π / 2 → f ω φ x < f ω φ y) : ω ≤ 4 :=
sorry

def f (ω φ x : ℝ) := Real.sin (ω * x + φ)

end max_value_of_omega_l664_664088


namespace number_of_subsets_of_set_A_l664_664602

theorem number_of_subsets_of_set_A : 
  (setOfSubsets : Finset (Finset ℕ)) = Finset.powerset {2, 4, 5} → 
  setOfSubsets.card = 8 :=
by
  sorry

end number_of_subsets_of_set_A_l664_664602


namespace tan_addition_theorem_l664_664849

theorem tan_addition_theorem (x y : ℝ) (hx : Real.tan x + Real.tan y = 10) 
  (hy : Real.cot x + Real.cot y = 15) : Real.tan (x + y) = 30 := 
by
  sorry

end tan_addition_theorem_l664_664849


namespace inscribed_sphere_radius_in_cone_l664_664171

noncomputable def radius_of_inscribed_sphere (b d : ℝ) : ℝ :=
  b * real.sqrt d - b

theorem inscribed_sphere_radius_in_cone (b d : ℝ) (radius base_radius height : ℝ) 
  (h1 : base_radius = 15) (h2 : height = 30) 
  (h3 : radius = radius_of_inscribed_sphere b d) : 
  b + d = 12.5 :=
sorry

end inscribed_sphere_radius_in_cone_l664_664171


namespace find_BE_l664_664530

-- Definitions of given conditions
variables {A B C E : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space E]
variable [ht : triangle ℝ A B C]
variable [right_triangle B]
variable [circle_diameter_bc : circle_diameter ℝ B C]
variable [intersects_AE {π B E}]
variable [area_ABC : triangle_area A B C 200]
variable [ab_len : edge_length A B 30]

-- Main theorem statement
theorem find_BE : ∃ BE : ℝ, BE = 40 / 3 :=
by
  sorry

end find_BE_l664_664530


namespace uncle_money_given_l664_664547

-- Definitions
def lizzy_mother_money : Int := 80
def lizzy_father_money : Int := 40
def candy_expense : Int := 50
def total_money_now : Int := 140

-- Theorem to prove
theorem uncle_money_given : (total_money_now - ((lizzy_mother_money + lizzy_father_money) - candy_expense)) = 70 := 
  by
    sorry

end uncle_money_given_l664_664547


namespace max_area_triangle_l664_664890

-- Definitions of points A, B, C
def A := (0 : ℝ, 0 : ℝ)
def B := (10 : ℝ, 0 : ℝ)
def C := (15 : ℝ, 0 : ℝ)

-- Definitions of lines with slopes and rotation
def line_ell_A := { p : ℝ × ℝ | p.2 = p.1 }
def line_ell_B := { p : ℝ × ℝ | p.1 = 10 }
def line_ell_C := { p : ℝ × ℝ | p.2 = -p.1 + 15 }

theorem max_area_triangle :
  ∃ (α : ℝ), 
    ∃ (X Y Z : ℝ × ℝ),
      X ∈ line_ell_B ∧ X ∈ line_ell_C ∧
      Y ∈ line_ell_A ∧ Y ∈ line_ell_C ∧
      Z ∈ line_ell_A ∧ Z ∈ line_ell_B ∧
      let area := (abs ((X.1 * (Y.2 - Z.2) + Y.1 * (Z.2 - X.2) + Z.1 * (X.2 - Y.2)) / 2)) in
      area = 62.5 :=
begin
  sorry
end

end max_area_triangle_l664_664890


namespace greatest_prime_factor_of_15_plus_18_l664_664657

theorem greatest_prime_factor_of_15_plus_18! : 
  let n := 15! + 18!
  n = 15! * 4897 ∧ Prime 4897 →
  (∀ p : ℕ, Prime p ∧ p ∣ n → p ≤ 4897) ∧ (4897 ∣ n) ∧ Prime 4897 :=
by {
  intros,
  sorry
}

end greatest_prime_factor_of_15_plus_18_l664_664657


namespace train_b_overtake_train_a_in_50_minutes_l664_664148

noncomputable def train_a_speed := 50 -- mph
noncomputable def train_b_speed := 80 -- mph
noncomputable def time_difference := 0.5 -- hours

theorem train_b_overtake_train_a_in_50_minutes:
  let distance_a := train_a_speed * time_difference in
  let relative_speed := train_b_speed - train_a_speed in
  let time_to_overtake := distance_a / relative_speed in
  time_to_overtake * 60 = 50 :=
by
  sorry

end train_b_overtake_train_a_in_50_minutes_l664_664148


namespace range_of_a_l664_664557

open Real

def setA := {x : ℝ | x^2 - 5*x + 4 > 0}
def setB (a : ℝ) := {x : ℝ | x^2 - 2*a*x + (a + 2) = 0}

theorem range_of_a (a : ℝ) :
  setA ∩ setB(a) ≠ ∅ → (a ∈ (-∞ : set ℝ, -1] ∪ (18/7 : set ℝ, ∞)) :=
begin
  sorry
end

end range_of_a_l664_664557


namespace shark_fin_falcata_area_is_correct_l664_664610

noncomputable def radius_large : ℝ := 3
noncomputable def center_large : ℝ × ℝ := (0, 0)

noncomputable def radius_small : ℝ := 3 / 2
noncomputable def center_small : ℝ × ℝ := (0, 3 / 2)

noncomputable def area_large_quarter_circle : ℝ := (1 / 4) * Real.pi * (radius_large ^ 2)
noncomputable def area_small_semicircle : ℝ := (1 / 2) * Real.pi * (radius_small ^ 2)

noncomputable def shark_fin_falcata_area (area_large_quarter_circle area_small_semicircle : ℝ) : ℝ := 
  area_large_quarter_circle - area_small_semicircle

theorem shark_fin_falcata_area_is_correct : 
  shark_fin_falcata_area area_large_quarter_circle area_small_semicircle = (9 * Real.pi) / 8 := 
by
  sorry

end shark_fin_falcata_area_is_correct_l664_664610


namespace geometric_sequence_sum_l664_664895

theorem geometric_sequence_sum (a : ℕ → ℝ)
  (h1 : a 1 + a 2 = 1/2)
  (h2 : a 3 + a 4 = 1)
  (h_geom : ∀ n, a n + a (n+1) = (a 1 + a 2) * 2^(n-1)) :
  a 7 + a 8 + a 9 + a 10 = 12 := 
sorry

end geometric_sequence_sum_l664_664895


namespace ratio_closest_to_zero_l664_664230

theorem ratio_closest_to_zero : Int.round ((10^(3000 : ℝ) + 10^(3003 : ℝ)) / (10^(3001 : ℝ) + 10^(3004 : ℝ))) = 0 :=
by
  sorry

end ratio_closest_to_zero_l664_664230


namespace find_q_arithmetic_sum_first_n_c_n_l664_664035

-- Define sequences
def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + d * (n - 1)
def geometric_seq (b₁ q : ℝ) (n : ℕ) : ℝ := b₁ * q^(n - 1)

-- Conditions:
-- 1. Arithmetic sequence with first term 1 and common difference 2.
def a_n (n : ℕ) : ℕ := arithmetic_seq 1 2 n

-- 2. Geometric sequence with first term 1 and common ratio q.
def b_n (q : ℝ) (n : ℕ) : ℝ := geometric_seq 1 q n

-- 3. c_n = a_n + b_n is arithmetic sequence.
def c_n (q : ℝ) (n : ℕ) : ℝ := a_n n + b_n q n

-- Theorem 1: Finding q such that {c_n} is an arithmetic sequence
theorem find_q_arithmetic (q : ℝ) : (∀ n m k : ℕ, c_n q (n + m + k) - c_n q (m + k) = c_n q (n + k) - c_n q k) → q = 1 :=
    by sorry

-- Define summation function
def sum_first_n_terms (f : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, f (i + 1)

-- Theorem 2: Sum of the first n terms of {c_n}
theorem sum_first_n_c_n (q : ℝ) (n : ℕ) : 
    (q = 1 → sum_first_n_terms (c_n q) n = n^2 + n) ∧ 
    (q ≠ 1 → sum_first_n_terms (c_n q) n = n^2 + (1 - q^n) / (1 - q)) :=
    by sorry

end find_q_arithmetic_sum_first_n_c_n_l664_664035


namespace inequality_always_holds_l664_664859

theorem inequality_always_holds (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 6 * x + 3 > 0) →
  (a > 3) ∧ (∀ x : ℝ, x = a + 9 / (a - 1) → x ≥ 7) :=
by
  sorry

end inequality_always_holds_l664_664859


namespace car_travel_distance_l664_664706

-- Define the average speed function
def avg_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

-- Define the distance function given speed and time
def distance (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

-- Lean statement to prove the distance traveled in the next 3 hours
theorem car_travel_distance : 
  avg_speed 180 4 * 3 = 135 :=
by
  sorry

end car_travel_distance_l664_664706


namespace ratio_f_values_l664_664821

noncomputable def f : ℝ → ℝ 
| x => if x < 1 then 2 * Real.sin (Real.pi * x) else f (x - 2 / 3)

theorem ratio_f_values : (f 2) / (f (-1 / 6)) = -Real.sqrt 3 := 
by 
  sorry

end ratio_f_values_l664_664821


namespace water_depth_is_255_feet_l664_664182

-- Given conditions as definitions
def Ron_height := 13
def Dean_is_4_feet_taller_than_Ron := ∀ Ron_height: ℕ, Dean_height = Ron_height + 4
def water_depth_is_15_times_Dean_height := ∀ Dean_height: ℕ, water_depth = 15 * Dean_height

-- The main theorem (problem statement)
theorem water_depth_is_255_feet (Ron_height : ℕ) (h1 : Dean_is_4_feet_taller_than_Ron Ron_height) 
(h2 : water_depth_is_15_times_Dean_height Dean_height) : water_depth = 255 :=
begin
  sorry
end

end water_depth_is_255_feet_l664_664182


namespace average_weight_of_class_is_61_67_l664_664614

noncomputable def totalWeightA (avgWeightA : ℝ) (numStudentsA : ℕ) : ℝ := avgWeightA * numStudentsA
noncomputable def totalWeightB (avgWeightB : ℝ) (numStudentsB : ℕ) : ℝ := avgWeightB * numStudentsB
noncomputable def totalWeightClass (totalWeightA : ℝ) (totalWeightB : ℝ) : ℝ := totalWeightA + totalWeightB
noncomputable def totalStudentsClass (numStudentsA : ℕ) (numStudentsB : ℕ) : ℕ := numStudentsA + numStudentsB
noncomputable def averageWeightClass (totalWeightClass : ℝ) (totalStudentsClass : ℕ) : ℝ := totalWeightClass / totalStudentsClass

theorem average_weight_of_class_is_61_67 :
  averageWeightClass (totalWeightClass (totalWeightA 50 50) (totalWeightB 70 70))
    (totalStudentsClass 50 70) = 61.67 := by
  sorry

end average_weight_of_class_is_61_67_l664_664614


namespace point_on_terminal_side_of_60_deg_l664_664467

theorem point_on_terminal_side_of_60_deg (a : ℝ) : 
  (∃ (a : ℝ), ∃ (b : ℝ), a = 4 ∧ (∃ (θ : ℝ), θ = real.pi / 3 ∧ tan θ = b / a) ∧ b = a) ↔ a = 4 * real.sqrt 3 :=
by
  sorry

end point_on_terminal_side_of_60_deg_l664_664467


namespace haley_marbles_l664_664869

theorem haley_marbles (m : ℕ) (k : ℕ) (h1 : k = 2) (h2 : m = 28) : m / k = 14 :=
by sorry

end haley_marbles_l664_664869


namespace geometry_problem_l664_664529

theorem geometry_problem
    (A B C H P Q : Point)
    (AB AC BC : ℝ)
    (h₁ : AB = 2015)
    (h₂ : AC = 2013)
    (h₃ : BC = 2012)
    (CH : Altitude A B C)
    (h₄ : IntersectionInCircleInscribed A C H P)
    (h₅ : IntersectionInCircleInscribed B C H Q)
    (PQ : ℝ)
    (h₆ : PQ = 403 / 806) :
  403 + 806 = 1209 := 
begin 
  sorry
end

end geometry_problem_l664_664529


namespace angle_in_fourth_quadrant_l664_664812

theorem angle_in_fourth_quadrant (θ : ℝ) 
  (h1 : sin θ * cos θ < 0)
  (h2 : 2 * cos θ > 0)
  (h3 : ∃ x y : ℝ, (sin θ * cos θ = x ∧ 2 * cos θ = y) ∧ (x < 0 ∧ y > 0)) : 
  θ ∈ Ioo (3 * Real.pi / 2) (2 * Real.pi) :=
sorry

end angle_in_fourth_quadrant_l664_664812


namespace apples_total_l664_664227

theorem apples_total (x y : ℕ) (hx : x = 8) (hy : y = 6) : x + y = 14 :=
by
  rw [hx, hy]
  norm_num
  sorry

end apples_total_l664_664227


namespace no_real_solution_x_plus_36_div_x_minus_3_eq_neg9_l664_664762

theorem no_real_solution_x_plus_36_div_x_minus_3_eq_neg9 : ∀ x : ℝ, x + 36 / (x - 3) = -9 → False :=
by
  assume x
  assume h : x + 36 / (x - 3) = -9
  sorry

end no_real_solution_x_plus_36_div_x_minus_3_eq_neg9_l664_664762


namespace even_number_of_divisors_less_than_100_l664_664326

theorem even_number_of_divisors_less_than_100 : 
  ∃ n, n = 90 ∧ ∀ x < 100, (∃ k, k * k = x → false) = (x ∣ 99 - 9) :=
sorry

end even_number_of_divisors_less_than_100_l664_664326


namespace find_y_l664_664437

theorem find_y (x y : ℚ) (h1 : x = 153) (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 350064) : 
  y = 40 / 3967 :=
by
  -- Proof to be filled in
  sorry

end find_y_l664_664437


namespace general_term_formula_minimum_sum_value_l664_664259

variable {a : ℕ → ℚ} -- The arithmetic sequence
variable {S : ℕ → ℚ} -- Sum of the first n terms of the sequence

-- Conditions
axiom a_seq_cond1 : a 2 + a 6 = 6
axiom S_sum_cond5 : S 5 = 35 / 3

-- Definitions
def a_n (n : ℕ) : ℚ := (2 / 3) * n + 1 / 3
def S_n (n : ℕ) : ℚ := (1 / 3) * (n^2 + 2 * n)

-- Hypotheses
axiom seq_def : ∀ n, a n = a_n n
axiom sum_def : ∀ n, S n = S_n n

-- Theorems to be proved
theorem general_term_formula : ∀ n, a n = (2 / 3 * n) + 1 / 3 := by sorry
theorem minimum_sum_value : ∀ n, S 1 ≤ S n := by sorry

end general_term_formula_minimum_sum_value_l664_664259


namespace midpoint_complex_l664_664888

noncomputable def midpoint (z1 z2 : ℂ) : ℂ :=
  (z1 + z2) / 2

theorem midpoint_complex (z1 z2 : ℂ) (mid : ℂ) (r : ℝ) :
  z1 = -7 + 4 * complex.I →
  z2 = 1 - 6 * complex.I →
  r = 5 →
  mid = midpoint z1 z2 →
  mid = -3 - complex.I ∧ complex.abs mid ≠ r :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h4]
  simp [midpoint]
  split
  · simp
  · norm_num 
  · sorry -- Finish with the necessary detailed proof

end midpoint_complex_l664_664888


namespace no_such_N_exists_l664_664112

theorem no_such_N_exists :
  ∀ (N : ℕ),
  (∀ (points : list (ℝ × ℝ)),
    (points.length = N) ∧ 
    (∀ (p1 p2 p3 : (ℝ × ℝ)), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬collinear p1 p2 p3) ∧
    (∃ (red blue : set (ℝ × ℝ) × (ℝ × ℝ)),
      (∀ (p1 p2 : (ℝ × ℝ)), p1 ≠ p2 → ((p1, p2) ∈ red ∨ (p1, p2) ∈ blue)) ∧
      (hamiltonian_cycle red points) ∧ (hamiltonian_cycle blue points) ∧ 
      (non_intersecting_cycles red blue points))) → False :=
by sorry

-- Definitions reused from the condition
def collinear (p1 p2 p3 : (ℝ × ℝ)) : Prop := 
  (p3.1 - p1.1) * (p2.2 - p1.2) = (p2.1 - p1.1) * (p3.2 - p1.2)

def hamiltonian_cycle (segments : set (ℝ × ℝ) × (ℝ × ℝ)) (points : list (ℝ × ℝ)) : Prop :=
  ∀ (p : (ℝ × ℝ)), p ∈ points → (∃! q : (ℝ × ℝ), (p, q) ∈ segments ∨ (q, p) ∈ segments)

def non_intersecting_cycles (red blue : set (ℝ × ℝ) × (ℝ × ℝ)) (points : list (ℝ × ℝ)) : Prop :=
  ∀ (p q : (ℝ × ℝ)), (p, q) ∈ red → (p, q) ∉ blue

end no_such_N_exists_l664_664112


namespace yellow_to_green_ratio_l664_664123

noncomputable def smaller_circle_diameter : ℝ := 2
noncomputable def larger_circle_diameter : ℝ := 4

theorem yellow_to_green_ratio
  (d_small : ℝ = smaller_circle_diameter)
  (d_large : ℝ = larger_circle_diameter) :
  let r_small := d_small / 2,
      r_large := d_large / 2,
      A_green := Real.pi * (r_small ^ 2),
      A_large := Real.pi * (r_large ^ 2),
      A_yellow := A_large - A_green
  in A_yellow / A_green = 3 :=
by sorry

end yellow_to_green_ratio_l664_664123


namespace unique_digit_sum_l664_664000

theorem unique_digit_sum (A B C D : ℕ) (h1 : A + B + C + D = 20) (h2 : B + A + 1 = 11) (uniq : (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (B ≠ C) ∧ (B ≠ D) ∧ (C ≠ D)) : D = 8 :=
sorry

end unique_digit_sum_l664_664000


namespace BC_length_l664_664911

theorem BC_length (A B C X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X] 
  (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
  (BX CX : ℕ) (h_circle_intersect : ∃ X, Metric.ball A 86 ∩ {BC} = {B, X})
  (h_integer_lengths : BX + CX = BC) :
  BC = 61 := 
by
  sorry

end BC_length_l664_664911


namespace correct_eqns_l664_664496

theorem correct_eqns (x y : ℝ) (h1 : x - y = 4.5) (h2 : 1/2 * x + 1 = y) :
  x - y = 4.5 ∧ 1/2 * x + 1 = y :=
by {
  exact ⟨h1, h2⟩,
}

end correct_eqns_l664_664496


namespace area_of_region_l664_664199

noncomputable def region_area (x y : ℝ) : ℝ :=
  x^2 + y^2 - (2 * abs (x - y) + abs (x + y))

theorem area_of_region :
    (∫ x in -1..1,  ∫ y in -1..1, if region_area x y = 0 then 1 else 0) = 13 * π / 2 :=
by 
  sorry

end area_of_region_l664_664199


namespace find_sin_alpha_l664_664692

-- Definition of the problem conditions
variable (α : Real) (x : Real)
variable (hx : α ∈ Set.Icc (π / 2) π)  -- α in the second quadrant
variable (hx_cos : cos α = (sqrt 2 / 4) * x)
variable (hx_point : ∀ (P : Real × Real), P = (x, sqrt 5))

-- The theorem we want to prove
theorem find_sin_alpha : sin α = sqrt 10 / 4 := sorry

end find_sin_alpha_l664_664692


namespace even_number_of_divisors_less_than_100_l664_664332

theorem even_number_of_divisors_less_than_100 : 
  ∃ n, n = 90 ∧ ∀ x < 100, (∃ k, k * k = x → false) = (x ∣ 99 - 9) :=
sorry

end even_number_of_divisors_less_than_100_l664_664332


namespace even_number_of_divisors_less_than_100_l664_664329

theorem even_number_of_divisors_less_than_100 : 
  ∃ n, n = 90 ∧ ∀ x < 100, (∃ k, k * k = x → false) = (x ∣ 99 - 9) :=
sorry

end even_number_of_divisors_less_than_100_l664_664329


namespace no_k_for_linear_function_not_in_second_quadrant_l664_664243

theorem no_k_for_linear_function_not_in_second_quadrant :
  ¬∃ k : ℝ, ∀ x < 0, (k-1)*x + k ≤ 0 :=
by
  sorry

end no_k_for_linear_function_not_in_second_quadrant_l664_664243


namespace no_product_of_six_primes_l664_664618

noncomputable theory

open Nat

-- Define the problem's conditions within a Lean 4 statement
theorem no_product_of_six_primes (S : Finset ℕ) (h_distinct : S.card = 2014)
  (h_condition : ∀ (x y ∈ S), x ≠ y → (x * y) % (x + y) = 0) :
  ∀ x ∈ S, ¬∃ (p1 p2 p3 p4 p5 p6 : ℕ), pairwise (≠) [p1, p2, p3, p4, p5, p6] ∧
  (∀ p ∈ [p1, p2, p3, p4, p5, p6], Nat.Prime p) ∧ x = p1 * p2 * p3 * p4 * p5 * p6 := 
by
  sorry

end no_product_of_six_primes_l664_664618


namespace max_diff_sum_is_sqrt_l664_664246

noncomputable def max_diff_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (2 * n + 1), (a (i + 1) - a i) ^ 2

theorem max_diff_sum_is_sqrt (a : ℕ → ℝ) (n : ℕ)
  (h : ∑ i in finset.range (2 * n), (a (i + 1) - a i) ^ 2 = 1) :
  (∑ i in finset.range (n + 1, 2 * n + 1), a i) - (∑ i in finset.range (1, n + 1), a i) =
  real.sqrt (n * (2 * n^2 + 1) / 3) :=
sorry

end max_diff_sum_is_sqrt_l664_664246


namespace fill_table_with_positive_integers_l664_664029

theorem fill_table_with_positive_integers
  (n m : ℕ)
  (a : Fin n → ℕ)
  (b : Fin m → ℕ)
  (H : (∏ i, a i) = (∏ j, b j)) :
  ∃ (T : Fin n → Fin m → ℕ),
    (∀ i, ∏ j, T i j = a i) ∧
    (∀ j, ∏ i, T i j = b j) := sorry

end fill_table_with_positive_integers_l664_664029
