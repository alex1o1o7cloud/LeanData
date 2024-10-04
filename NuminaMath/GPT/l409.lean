import Mathlib

namespace exist_two_lines_parallel_at_distance_l409_409803

variables (A B : Point) (Delta : Plane) (m : ℝ)

theorem exist_two_lines_parallel_at_distance:
  ∃ a a1 : Line, 
  passes_through A a ∧ 
  passes_through A a1 ∧
  parallel_to_plane Delta a ∧ 
  parallel_to_plane Delta a1 ∧ 
  distance_to_point B a = m ∧ 
  distance_to_point B a1 = m :=
sorry

end exist_two_lines_parallel_at_distance_l409_409803


namespace points_on_line_l409_409425

-- Define the line by specifying the points it passes through.
def line_through (p1 p2 : ℝ × ℝ) : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | ∃ (m : ℝ), m = (p2.2 - p1.2) / (p2.1 - p1.1) ∧ p.2 = p1.2 + m * (p.1 - p1.1) }

-- Given points
def p1 : ℝ × ℝ := (4, 10)
def p2 : ℝ × ℝ := (1, 1)

-- The points to check
def points_to_check : list (ℝ × ℝ) := [(2, 3), (3, 6), (0, -2), (5, 13), (6, 16)]

-- The points that should be tested true
def correct_points : list (ℝ × ℝ) := [(0, -2), (5, 13), (6, 16)]

-- Proof Statement
theorem points_on_line :
  ∀ p ∈ correct_points, p ∈ line_through p1 p2 :=
sorry

end points_on_line_l409_409425


namespace smallest_d_for_range_of_g_l409_409456

theorem smallest_d_for_range_of_g :
  ∃ d, (∀ x : ℝ, x^2 + 4 * x + d = 3) → d = 7 := by
  sorry

end smallest_d_for_range_of_g_l409_409456


namespace find_x_l409_409471

theorem find_x (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end find_x_l409_409471


namespace area_enclosed_by_parabola_tangent_l409_409449

noncomputable def enclosed_area : ℝ :=
  let f := λ x : ℝ, x^2
  let tangent_line := λ x : ℝ, 2 * x - 1
  let integral_parabola := ∫ x in 0..1, f x
  let integral_triangle := (1 / 2) * (1 / 2) * (1 - 0)
  integral_parabola - integral_triangle

theorem area_enclosed_by_parabola_tangent (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) :
  enclosed_area = 1 / 12 :=
by sorry

end area_enclosed_by_parabola_tangent_l409_409449


namespace cos_90_eq_0_l409_409893

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l409_409893


namespace cos_ninety_degrees_l409_409925

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l409_409925


namespace value_of_A_l409_409087

theorem value_of_A {α : Type} [LinearOrderedSemiring α] 
  (L A D E : α) (L_value : L = 15) (LEAD DEAL DELL : α)
  (LEAD_value : LEAD = 50)
  (DEAL_value : DEAL = 55)
  (DELL_value : DELL = 60)
  (LEAD_condition : L + E + A + D = LEAD)
  (DEAL_condition : D + E + A + L = DEAL)
  (DELL_condition : D + E + L + L = DELL) :
  A = 25 :=
by
  sorry

end value_of_A_l409_409087


namespace Vanya_Journey_Five_times_Anya_Journey_l409_409411

theorem Vanya_Journey_Five_times_Anya_Journey (a_start a_end v_start v_end : ℕ)
  (h1 : a_start = 1) (h2 : a_end = 2) (h3 : v_start = 1) (h4 : v_end = 6) :
  (v_end - v_start) = 5 * (a_end - a_start) :=
  sorry

end Vanya_Journey_Five_times_Anya_Journey_l409_409411


namespace smallest_digit_never_in_units_place_of_odd_numbers_l409_409310

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l409_409310


namespace smallest_digit_not_in_odd_units_l409_409209

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l409_409209


namespace smallest_unfound_digit_in_odd_units_l409_409233

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l409_409233


namespace hall_area_proof_l409_409097

noncomputable def hall_length (L : ℕ) : ℕ := L
noncomputable def hall_width (L : ℕ) (W : ℕ) : ℕ := W
noncomputable def hall_area (L W : ℕ) : ℕ := L * W

theorem hall_area_proof (L W : ℕ) (h1 : W = 1 / 2 * L) (h2 : L - W = 15) :
  hall_area L W = 450 := by
  sorry

end hall_area_proof_l409_409097


namespace sum_of_possible_values_of_g1_l409_409013

theorem sum_of_possible_values_of_g1 (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g (g (x - y)) = g x * g y + g x - g y - 2 * x * y) :
  ∑ (c : ℝ) in (set_of (λ c, ∃ (g : ℝ → ℝ), (∀ x y : ℝ, g (g (x - y)) = g x * g y + g x - g y - 2 * x * y) ∧ g 1 = c)).to_finset = 0 := by
  sorry

end sum_of_possible_values_of_g1_l409_409013


namespace cos_90_eq_0_l409_409962

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l409_409962


namespace possible_values_g1_l409_409644

noncomputable def g : ℝ → ℝ := sorry

axiom g_functional_equation : ∀ (x y : ℝ), g ((x + y)^2) = g(x)^2 + 2 * x * y + y^2

theorem possible_values_g1 (m t : ℕ) : 
  (m = 1) ∧ (t = 1) ∧ (m * t = 1) :=
by 
  -- conditions from part c)
  have functional_eq := g_functional_equation,
  -- the required proof steps
  sorry

end possible_values_g1_l409_409644


namespace incorrect_shift_period_l409_409534

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Math.cos (2 * x + φ)

theorem incorrect_shift_period (φ : ℝ) (hφ : φ > 0) :
  ∀ x, f x φ ≠ Math.cos (2 * (x + φ)) :=
by
  intros x
  have h : f x φ = Math.cos (2 * x + φ) := rfl
  sorry

end incorrect_shift_period_l409_409534


namespace solve_inequality_1_solve_inequality_2_l409_409059

theorem solve_inequality_1 (x : ℝ) : 
  x^2 + x - 6 < 0 ↔ -3 < x ∧ x < 2 :=
sorry

theorem solve_inequality_2 (x : ℝ) : 
  -6x^2 - x + 2 ≤ 0 ↔ x ≤ -2/3 ∨ x ≥ 1/2 :=
sorry

end solve_inequality_1_solve_inequality_2_l409_409059


namespace cos_pi_half_eq_zero_l409_409861

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l409_409861


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409330

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409330


namespace smallest_not_odd_unit_is_zero_l409_409182

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l409_409182


namespace arithmetic_sequence_general_term_l409_409599

variables (a : ℕ → ℕ) (b : ℕ → ℝ) (S : ℕ → ℝ)

-- Definitions from given conditions
axiom a_1 : a 1 = 1
axiom b_definition : ∀ n, b n = (1 / 2) ^ (a n)
axiom product_b_123 : b 1 * b 2 * b 3 = 1 / 64

-- Prove the general term formula for the sequence {a_n} and the sum S_n for sequence {a_n * b_n}
theorem arithmetic_sequence_general_term :
  (∀ n, a n = n) ∧ (∀ n, S n = (∑ k in finset.range n, (k + 1) / 2 ^ (k + 1)) = 2 - 2 / 2^n - n / 2^n) :=
by sorry

end arithmetic_sequence_general_term_l409_409599


namespace plane_equation_parametric_l409_409788

theorem plane_equation_parametric 
  (s t : ℝ)
  (v : ℝ × ℝ × ℝ)
  (x y z : ℝ) 
  (A B C D : ℤ)
  (h1 : v = (2 + s + 2 * t, 3 + 2 * s - t, 1 + s + 3 * t))
  (h2 : A = 7)
  (h3 : B = -1)
  (h4 : C = -5)
  (h5 : D = -6)
  (h6 : A > 0)
  (h7 : Int.gcd A (Int.gcd B (Int.gcd C D)) = 1) :
  7 * x - y - 5 * z - 6 = 0 := 
sorry

end plane_equation_parametric_l409_409788


namespace total_number_of_fish_l409_409831

theorem total_number_of_fish :
  let goldfish := 8
  let angelfish := goldfish + 4
  let guppies := 2 * angelfish
  let tetras := goldfish - 3
  let bettas := tetras + 5
  goldfish + angelfish + guppies + tetras + bettas = 59 := by
  -- Provide the proof here.
  sorry

end total_number_of_fish_l409_409831


namespace smallest_digit_never_in_units_place_of_odd_l409_409200

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l409_409200


namespace incorrect_statement_is_B_l409_409363

-- Define the conditions
def genotype_AaBb_meiosis_results (sperm_genotypes : List String) : Prop :=
  sperm_genotypes = ["AB", "Ab", "aB", "ab"]

def spermatogonial_cell_AaXbY (malformed_sperm_genotype : String) (other_sperm_genotypes : List String) : Prop :=
  malformed_sperm_genotype = "AAaY" ∧ other_sperm_genotypes = ["aY", "X^b", "X^b"]

def spermatogonial_secondary_spermatocyte_Y_chromosomes (contains_two_Y : Bool) : Prop :=
  ¬ contains_two_Y

def female_animal_meiosis (primary_oocyte_alleles : Nat) (max_oocyte_b_alleles : Nat) : Prop :=
  primary_oocyte_alleles = 10 ∧ max_oocyte_b_alleles ≤ 5

-- The main statement that needs to be proved
theorem incorrect_statement_is_B :
  ∃ (sperm_genotypes : List String) 
    (malformed_sperm_genotype : String) 
    (other_sperm_genotypes : List String) 
    (contains_two_Y : Bool) 
    (primary_oocyte_alleles max_oocyte_b_alleles : Nat),
    genotype_AaBb_meiosis_results sperm_genotypes ∧ 
    spermatogonial_cell_AaXbY malformed_sperm_genotype other_sperm_genotypes ∧ 
    spermatogonial_secondary_spermatocyte_Y_chromosomes contains_two_Y ∧ 
    female_animal_meiosis primary_oocyte_alleles max_oocyte_b_alleles 
    ∧ (malformed_sperm_genotype = "AAaY" → false) := 
sorry

end incorrect_statement_is_B_l409_409363


namespace geometric_seq_sum_S40_l409_409091

noncomputable def geometric_seq_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q ≠ 1 then a1 * (1 - q^n) / (1 - q) else a1 * n

theorem geometric_seq_sum_S40 :
  ∃ (a1 q : ℝ), (0 < q ∧ q ≠ 1) ∧ 
                geometric_seq_sum a1 q 10 = 10 ∧
                geometric_seq_sum a1 q 30 = 70 ∧
                geometric_seq_sum a1 q 40 = 150 :=
by
  sorry

end geometric_seq_sum_S40_l409_409091


namespace trip_time_l409_409063

-- Define the conditions
def constant_speed : ℝ := 62 -- in miles per hour
def total_distance : ℝ := 2790 -- in miles
def break_interval : ℝ := 5 -- in hours
def break_duration : ℝ := 0.5 -- in hours (30 minutes)
def hotel_search_time : ℝ := 0.5 -- in hours (30 minutes)

theorem trip_time :
  let total_driving_time := total_distance / constant_speed in
  let number_of_breaks := total_driving_time / break_interval in
  let total_break_time := number_of_breaks * break_duration in
  let total_time := total_driving_time + total_break_time + hotel_search_time in
  total_time = 50 :=
by
  sorry

end trip_time_l409_409063


namespace smallest_digit_not_in_units_place_of_odd_numbers_l409_409162

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l409_409162


namespace smallest_unfound_digit_in_odd_units_l409_409235

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l409_409235


namespace angle_A_perimeter_range_l409_409547

noncomputable theory

-- Definitions
def is_triangle_ABC (A B C a b c : ℝ) : Prop := sorry

def vector_m (a A B b c : ℝ) : ℝ × ℝ := (a / (Real.sin (A + B)), c - 2 * b)

def vector_n (C : ℝ) : ℝ × ℝ := (Real.sin  (2 * C), 1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Theorem statements
theorem angle_A (a b c A B C : ℝ) : 
  is_triangle_ABC A B C a b c →
  (dot_product (vector_m a A B b c) (vector_n C) = 0) →
  A = (Real.pi / 3) :=
sorry

theorem perimeter_range (b c B C : ℝ) :
  is_triangle_ABC (Real.pi / 3) B C 1 b c →
  (dot_product (vector_m 1 (Real.pi / 3) B b c) (vector_n C) = 0) →
  2 < (1 + (2 * Real.sqrt 3 / 3) * (Real.sin B + Real.sin (B + Real.pi / 3))) ∧
  (1 + (2 * Real.sqrt 3 / 3) * (Real.sin B + Real.sin (B + Real.pi / 3))) ≤ 3 :=
sorry

end angle_A_perimeter_range_l409_409547


namespace simplify_expression_l409_409690

variable (a : ℝ)

theorem simplify_expression (h : a > 0) : 
  let numerator := (a^(3/2)) + (a^(3/4)) - (sqrt(a^3 + 2*a^2) + (a * (a + 2)^2)^(1/4))
  let denominator := sqrt(2 * (a + 1 - sqrt(a^2 + 2*a))) * (a^2 - a^(5/4) + a^(1/2))⁻¹
  numerator / denominator = - (a^3 + a^(3/4)) := sorry

end simplify_expression_l409_409690


namespace cos_90_eq_0_l409_409965

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l409_409965


namespace proof_problem_l409_409520

def h (x : ℝ) : ℝ := x^2 - 3 * x + 7
def k (x : ℝ) : ℝ := 2 * x + 4

theorem proof_problem : h (k 3) - k (h 3) = 59 := by
  sorry

end proof_problem_l409_409520


namespace cos_of_90_degrees_l409_409994

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l409_409994


namespace smallest_number_of_2_by_3_rectangles_l409_409741

def area_2_by_3_rectangle : Int := 2 * 3

def smallest_square_area_multiple_of_6 : Int :=
  let side_length := 6
  side_length * side_length

def number_of_rectangles_to_cover_square (square_area : Int) (rectangle_area : Int) : Int :=
  square_area / rectangle_area

theorem smallest_number_of_2_by_3_rectangles :
  number_of_rectangles_to_cover_square smallest_square_area_multiple_of_6 area_2_by_3_rectangle = 6 := by
  sorry

end smallest_number_of_2_by_3_rectangles_l409_409741


namespace cos_90_eq_0_l409_409964

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l409_409964


namespace perimeter_AMN_l409_409104

open Classical

variables (A B C M N : Type)
variables [OrderedField A] [MetricSpace A]
variables (a b c Ab Ac Am An : A) (perimeter : A)

def length (x y : A) : A := dist x y

-- Given conditions
axiom Triangle_ABC : length A B = 15 ∧ length B C = 25 ∧ length A C = 20
axiom Line_MI_parallel_BC : || incenter of triangle A B C being M is parallel to segment B C
axiom Intersection_M_extends_AB_Ac_externally_N : M intersects extends AB and N intersects extends AC externally

-- The proof statement
theorem perimeter_AMN : perimeter = 35 := sorry

end perimeter_AMN_l409_409104


namespace intervals_of_increase_and_decrease_of_f_l409_409434

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - Real.log x

theorem intervals_of_increase_and_decrease_of_f :
  (∀ x ∈ Ioi (Real.sqrt 2 / 2), 0 < (deriv f x)) ∧ 
  (∀ x ∈ Ioo 0 (Real.sqrt 2 / 2), (deriv f x) < 0) :=
begin
  -- The proof part is omitted as per the instructions
  sorry
end

end intervals_of_increase_and_decrease_of_f_l409_409434


namespace cos_ninety_degrees_l409_409927

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l409_409927


namespace smallest_digit_not_in_odd_units_l409_409246

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l409_409246


namespace cos_90_eq_0_l409_409961

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l409_409961


namespace smallest_digit_never_at_units_place_of_odd_l409_409164

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l409_409164


namespace unicorn_flowers_l409_409727

theorem unicorn_flowers (unicorns steps_per_step total_steps total_flowers : ℕ) (hkms : ∀ k, k = 9 * 1000)
  (h_steps_per_unicorn : ∀ u, steps_per_step = 3) 
  (h_total_steps : total_steps = 6 * (hkms 9 / h_steps_per_unicorn 3)) 
  (h_total_flowers : total_flowers = 72000) : 
  total_flowers / total_steps = 4 :=
by
  unfold hkms
  unfold h_steps_per_unicorn
  unfold h_total_steps
  unfold h_total_flowers
  simp
  sorry

end unicorn_flowers_l409_409727


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409128

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409128


namespace cosines_identity_l409_409374

noncomputable def inscribed_polygon (ABCDE : Π i : Fin 5, Point ℝ) :=
  ∃ (A B C D E : Point ℝ) (O : Point ℝ) (r : ℝ),
    O.distance A = r ∧ O.distance B = r ∧ O.distance C = r ∧ O.distance D = r ∧ O.distance E = r ∧
    ∀ i : Fin 5, ABCDE i = [A, B, C, D, E].nth i ∧
    A.distance B = 6 ∧ B.distance C = 6 ∧ C.distance D = 6 ∧ D.distance E = 6 ∧ A.distance E = 2

open Real Geometry

noncomputable def polygon_angles (angles : PolygonAngles ABCDE) :=
  ∀ (C : ι) [decidable_eq ι], ι → ℝ

theorem cosines_identity (ABCDE : Π i : Fin 5, Point ℝ) 
  (h1 : inscribed_polygon ABCDE)
  (h2 : polygon_angles ABCDE)
  : (1 - cos (h2 2 2)) * (1 - cos (h2 0 2)) = 1 / 18 := by
  sorry

end cosines_identity_l409_409374


namespace smallest_digit_never_at_units_place_of_odd_l409_409173

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l409_409173


namespace eggs_in_larger_omelette_l409_409414

theorem eggs_in_larger_omelette :
  ∀ (total_eggs : ℕ) (orders_3_eggs_first_hour orders_3_eggs_third_hour orders_large_eggs_second_hour orders_large_eggs_last_hour num_eggs_per_3_omelette : ℕ),
    total_eggs = 84 →
    orders_3_eggs_first_hour = 5 →
    orders_3_eggs_third_hour = 3 →
    orders_large_eggs_second_hour = 7 →
    orders_large_eggs_last_hour = 8 →
    num_eggs_per_3_omelette = 3 →
    (total_eggs - (orders_3_eggs_first_hour * num_eggs_per_3_omelette + orders_3_eggs_third_hour * num_eggs_per_3_omelette)) / (orders_large_eggs_second_hour + orders_large_eggs_last_hour) = 4 :=
by
  intros total_eggs orders_3_eggs_first_hour orders_3_eggs_third_hour orders_large_eggs_second_hour orders_large_eggs_last_hour num_eggs_per_3_omelette
  sorry

end eggs_in_larger_omelette_l409_409414


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409337

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409337


namespace cos_90_deg_eq_zero_l409_409913

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l409_409913


namespace cost_of_painting_cube_l409_409698

-- Definitions for conditions
def cost_per_kg : ℝ := 36.50
def coverage_per_kg : ℝ := 16  -- square feet
def side_length : ℝ := 8       -- feet

-- Derived constants
def area_per_face : ℝ := side_length * side_length
def number_of_faces : ℝ := 6
def total_surface_area : ℝ := number_of_faces * area_per_face
def paint_required : ℝ := total_surface_area / coverage_per_kg
def total_cost : ℝ := paint_required * cost_per_kg

-- Theorem statement
theorem cost_of_painting_cube : total_cost = 876 := by
  sorry

end cost_of_painting_cube_l409_409698


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409129

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409129


namespace smallest_digit_never_at_units_place_of_odd_l409_409171

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l409_409171


namespace area_of_rectangular_field_l409_409664

-- Define the conditions of the problem.
def width : ℝ := 16
def diagonal : ℝ := 17
def length : ℝ := real.sqrt (diagonal^2 - width^2)

-- Calculate the area and state the theorem that the area is approximately 91.84 square meters.
theorem area_of_rectangular_field :
  let area := length * width in
  area ≈ 91.84 :=
by sorry

end area_of_rectangular_field_l409_409664


namespace permutations_count_satisfy_condition_l409_409452

open Nat

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def isPermutation (p : List ℕ) (l : List ℕ) : Prop :=
  p.perm l

def satisfiesCondition (b : List ℕ) : Prop :=
  (b.length = 6) ∧
  (\<exists>b1 b2 b3 b4 b5 b6, 
   b = [b1, b2, b3, b4, b5, b6] ∧ 
   \(\frac{b1 + 6}{2} \cdot \frac{b2 + 5}{2} \cdot \frac{b3 + 4}{2} \cdot \frac{b4 + 3}{2} \cdot \frac{b5 + 2}{2} \cdot \frac{b6 + 1}{2} > factorial 6\))

theorem permutations_count_satisfy_condition :
  ∃ (n : ℕ), n = 719 ∧ 
  ∃ (b : List ℕ), isPermutation b [1, 2, 3, 4, 5, 6] ∧ satisfiesCondition b :=
sorry

end permutations_count_satisfy_condition_l409_409452


namespace smallest_missing_digit_l409_409313

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l409_409313


namespace Mike_books_l409_409029

theorem Mike_books
  (initial_books : ℝ)
  (books_sold : ℝ)
  (books_gifts : ℝ) 
  (books_bought : ℝ)
  (h_initial : initial_books = 51.5)
  (h_sold : books_sold = 45.75)
  (h_gifts : books_gifts = 12.25)
  (h_bought : books_bought = 3.5):
  initial_books - books_sold + books_gifts + books_bought = 21.5 := 
sorry

end Mike_books_l409_409029


namespace five_digit_prob_div_by_11_l409_409568

theorem five_digit_prob_div_by_11 : 
  (∃ num : ℕ, num ∈ (set.filter (λ n, (digit_sum n = 42 ∧ n < 100000 ∧ n ≥ 10000)) {n | n < 100000}) 
   ∧ (num % 11 = 0)) → (num = 4/25) :=
by
  sorry

end five_digit_prob_div_by_11_l409_409568


namespace perpendicular_lines_l409_409376

theorem perpendicular_lines (a : ℝ) : 
  (a = -1 → (∀ x y : ℝ, 4 * x - (a + 1) * y + 9 = 0 → x ≠ 0 →  y ≠ 0 → 
  ∃ b : ℝ, (b^2 + 1) * x - b * y + 6 = 0)) ∧ 
  (∀ x y : ℝ, (4 * x - (a + 1) * y + 9 = 0) ∧ (∃ x y : ℝ, (a^2 - 1) * x - a * y + 6 = 0) → a ≠ -1) := 
sorry

end perpendicular_lines_l409_409376


namespace perfect_square_proof_l409_409619

theorem perfect_square_proof (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := 
sorry

end perfect_square_proof_l409_409619


namespace solve_equation_l409_409023

open Real

/-- Define the original equation as a function. -/
def equation (x : ℝ) : ℝ := 2 ^ (3 ^ (4 ^ x)) - 4 ^ (3 ^ (2 ^ x))

/-- Define the specific value of x that solves the equation. -/
noncomputable def x_solution : ℝ := log 1.4386 / log 2

/-- Statement of the proof problem verifying the solution. -/
theorem solve_equation : equation x_solution = 0 :=
by
  -- Proof steps will be added here
  sorry

end solve_equation_l409_409023


namespace cos_ninety_degrees_l409_409919

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l409_409919


namespace equation_squares_l409_409608

theorem equation_squares (a b c : ℤ) (h : (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = a ^ 2 + b ^ 2 - c ^ 2) :
  ∃ k1 k2 : ℤ, (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = k1 ^ 2 ∧ a ^ 2 + b ^ 2 - c ^ 2 = k2 ^ 2 :=
by
  sorry

end equation_squares_l409_409608


namespace time_per_flash_l409_409400

def minutes_per_hour : ℕ := 60
def seconds_per_minute : ℕ := 60
def light_flashes_in_three_fourths_hour : ℕ := 180

-- Converting ¾ of an hour to minutes and then to seconds
def seconds_in_three_fourths_hour : ℕ := (3 * minutes_per_hour / 4) * seconds_per_minute

-- Proving that the time taken for one flash is 15 seconds
theorem time_per_flash : (seconds_in_three_fourths_hour / light_flashes_in_three_fourths_hour) = 15 :=
by
  sorry

end time_per_flash_l409_409400


namespace smallest_missing_digit_l409_409319

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l409_409319


namespace coin_problem_solution_l409_409034

theorem coin_problem_solution :
  ∀ (c1 c10 c20 : ℕ),
  1 * c1 + 10 * c10 + 20 * c20 = 50 →
  (∃ c1' c10' c20', c1' + c10' + c20' = 3 ∧ 1 * c1' + 10 * c10' + 20 * c20' = 50) →
  (∃ c1'' c10'' c20'', c1'' + c10'' + c20'' = 50 ∧ 1 * c1'' + 10 * c10'' + 20 * c20'' = 50) →
  (50 - 3 = 47) :=
begin
  intros,
  sorry
end

end coin_problem_solution_l409_409034


namespace sum_of_squares_of_roots_of_quadratic_l409_409501

theorem sum_of_squares_of_roots_of_quadratic :
  ( ∃ x1 x2 : ℝ, x1^2 - 3 * x1 - 1 = 0 ∧ x2^2 - 3 * x2 - 1 = 0 ∧ x1 ≠ x2) →
  x1^2 + x2^2 = 11 :=
by
  /- Proof goes here -/
  sorry

end sum_of_squares_of_roots_of_quadratic_l409_409501


namespace randi_more_nickels_l409_409048

noncomputable def more_nickels (total_cents : ℕ) (to_peter_cents : ℕ) (to_randi_cents : ℕ) : ℕ := 
  (to_randi_cents / 5) - (to_peter_cents / 5)

theorem randi_more_nickels :
  ∀ (total_cents to_peter_cents : ℕ),
  total_cents = 175 →
  to_peter_cents = 30 →
  more_nickels total_cents to_peter_cents (2 * to_peter_cents) = 6 :=
by
  intros total_cents to_peter_cents h_total h_peter
  rw [h_total, h_peter]
  unfold more_nickels
  norm_num
  sorry

end randi_more_nickels_l409_409048


namespace smallest_digit_not_found_in_units_place_of_odd_number_l409_409138

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l409_409138


namespace cos_of_90_degrees_l409_409985

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l409_409985


namespace smallest_digit_not_in_odd_units_l409_409217

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l409_409217


namespace smallest_digit_never_in_units_place_of_odd_l409_409204

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l409_409204


namespace find_center_and_radius_of_given_circle_l409_409450

def circle_equation := ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y - 11 = 0

def center_of_circle (cx cy : ℝ) := cx = -1 ∧ cy = 2

def radius_of_circle (r : ℝ) := r = real.sqrt 4

theorem find_center_and_radius_of_given_circle : 
(∃ (cx cy r : ℝ), circle_equation cx cy ∧ center_of_circle cx cy ∧ radius_of_circle r) := 
sorry

end find_center_and_radius_of_given_circle_l409_409450


namespace discounted_price_increase_percentage_l409_409765

theorem discounted_price_increase_percentage (P : ℝ) (discount_rate target_increase : ℝ) 
    (h₁ : P = 200) 
    (h₂ : discount_rate = 0.15) 
    (h₃ : target_increase = 0.10) :
    let P_discounted := P - discount_rate * P
    let P_target := P + target_increase * P
    let increase_factor := P_target / P_discounted
    let percentage_increase := (increase_factor - 1) * 100
    percentage_increase ≈ 29.41 := 
by
  let P_discounted := P * (1 - 0.15)
  let P_target := P * 1.10
  let increase_factor := P_target / P_discounted
  let percentage_increase := (increase_factor - 1) * 100
  have h_discounted : P_discounted = 170 := by 
    rw [h₁, h₂]
    norm_num
  have h_target : P_target = 220 := by 
    rw [h₁, h₃]
    norm_num
  have h_increase_factor : increase_factor = 220 / 170 := by 
    rw [h_discounted, h_target]
  have h_percentage_increase : percentage_increase ≈ 29.41 := by 
    rw [h_increase_factor]:;
    norm_num
  exact h_percentage_increase


end discounted_price_increase_percentage_l409_409765


namespace smallest_digit_not_found_in_units_place_of_odd_number_l409_409145

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l409_409145


namespace limit_of_seq_is_e_l409_409418

-- Define the sequence as a function
def seq (n : ℕ) : ℝ := ( (7 * n^2 + 18 * n - 15) / (7 * n^2 + 11 * n + 15) ) ^ (n + 2)

-- State the theorem to be proven
theorem limit_of_seq_is_e : 
  tendsto seq at_top (𝓝 real.exp 1) :=
by
  sorry

end limit_of_seq_is_e_l409_409418


namespace trip_time_proof_l409_409060

def driving_speed := 62 -- miles per hour
def distance := 2790 -- miles
def break_time := 0.5 -- hours (30 minutes)
def break_interval := 5 -- hours
def hotel_search_time := 0.5 -- hours (30 minutes)

theorem trip_time_proof :
  let driving_time := distance / driving_speed
  let number_of_breaks := (driving_time / break_interval).toNat - 1
  let total_break_time := number_of_breaks * break_time
  let total_time := driving_time + total_break_time + hotel_search_time
  total_time = 49.5 := by
  -- proof goes here
  sorry

end trip_time_proof_l409_409060


namespace perfect_square_condition_l409_409614

theorem perfect_square_condition (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 ↔ 4a - 3b = 5 * k :=
by  
  sorry

end perfect_square_condition_l409_409614


namespace area_of_region_l409_409600

theorem area_of_region :
  ∀ (x y : ℝ), (|2 * x - 2| + |3 * y - 3| ≤ 30) → (area_of_figure = 300) :=
sorry

end area_of_region_l409_409600


namespace theo_cookies_eaten_in_9_months_l409_409720

-- Define the basic variable values as per the conditions
def cookiesPerTime : Nat := 25
def timesPerDay : Nat := 5
def daysPerMonth : Nat := 27
def numMonths : Nat := 9

-- Define the total number of cookies Theo can eat in 9 months
def totalCookiesIn9Months : Nat :=
  cookiesPerTime * timesPerDay * daysPerMonth * numMonths

-- The theorem stating the answer
theorem theo_cookies_eaten_in_9_months :
  totalCookiesIn9Months = 30375 := by
  -- Proof will go here
  sorry

end theo_cookies_eaten_in_9_months_l409_409720


namespace polynomial_composite_l409_409687

theorem polynomial_composite (x : ℕ) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 4 * x^3 + 6 * x^2 + 4 * x + 1 = a * b :=
by
  sorry

end polynomial_composite_l409_409687


namespace smallest_digit_not_in_units_place_of_odd_numbers_l409_409159

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l409_409159


namespace triangle_probability_15gon_l409_409444

namespace TriangleProbability

open Real
open Nat

noncomputable def length_of_segment (k : ℕ) : ℝ :=
  2 * sin (k * π / 15)

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_probability_15gon :
  let seg_lengths := (λ k, length_of_segment (k + 1)) <$> (Finset.range 7)
  let combinations := (seg_lengths.product seg_lengths).product seg_lengths
  let valid_combinations := combinations.filter (λ ((a, b), c), is_triangle a b c)
  (valid_combinations.card : ℝ) / (combinations.card : ℝ) = 775 / 1001 :=
  sorry

end TriangleProbability

end triangle_probability_15gon_l409_409444


namespace sum_of_infinite_series_l409_409826

noncomputable def sum_of_series : ℚ :=
  ∑' n, (2 * n + 1) * (1 / 2000) ^ n

theorem sum_of_infinite_series :
  sum_of_series = 4002000 / 3996001 :=
sorry

end sum_of_infinite_series_l409_409826


namespace find_sum_of_multiples_l409_409563

-- Define the smallest three-digit multiple of 5
def smallest_three_digit_multiple_of_five : ℕ := 100

-- Define the smallest four-digit multiple of 7
def smallest_four_digit_multiple_of_seven : ℕ := 1001

-- State the proof problem
theorem find_sum_of_multiples :
  let a := smallest_three_digit_multiple_of_five;
  let b := smallest_four_digit_multiple_of_seven in
  a + b = 1101 :=
by
  -- Proof goes here
  sorry

end find_sum_of_multiples_l409_409563


namespace find_regular_pay_l409_409785

-- Definitions from conditions
def regular_pay (R : ℝ) : Prop :=
  ∃ (R : ℝ), 
    let total_hours := 40 + 8
    let overtime_pay := 8 * 2 * R
    let regular_pay := 40 * R
    (total_hours = 48) ∧ (overtime_pay + regular_pay = 168)

-- Theorem to prove
theorem find_regular_pay : 
  ∃ R : ℝ, (regular_pay R) ∧ (R = 3) :=
begin
  sorry
end

end find_regular_pay_l409_409785


namespace cos_90_eq_0_l409_409899

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l409_409899


namespace cos_90_eq_0_l409_409872

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l409_409872


namespace cos_90_deg_eq_zero_l409_409909

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l409_409909


namespace smallest_digit_not_in_odd_units_l409_409213

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l409_409213


namespace equation_squares_l409_409610

theorem equation_squares (a b c : ℤ) (h : (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = a ^ 2 + b ^ 2 - c ^ 2) :
  ∃ k1 k2 : ℤ, (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = k1 ^ 2 ∧ a ^ 2 + b ^ 2 - c ^ 2 = k2 ^ 2 :=
by
  sorry

end equation_squares_l409_409610


namespace smallest_positive_integer_n_l409_409355

theorem smallest_positive_integer_n (n : ℕ) 
  (h1 : ∃ k : ℕ, n = 5 * k ∧ perfect_square(5 * k)) 
  (h2 : ∃ m : ℕ, n = 4 * m ∧ perfect_cube(4 * m)) : 
  n = 625000 :=
sorry

end smallest_positive_integer_n_l409_409355


namespace IntegerDivisors_l409_409671

theorem IntegerDivisors : (Nat.num_divisors (11^60 - 17^24)) ≥ 120 :=
by
  have h1 : 60 % 12 = 0 := by norm_num
  have h2 : 24 % 12 = 0 := by norm_num
  have h3 : 11^60 % (2^4 * 3^2 * 5 * 7 * 13) = 1 := by sorry 
  have h4 : 17^24 % (2^4 * 3^2 * 5 * 7 * 13) = 1 := by sorry
  have h5 : (11^60 - 17^24) % (2^4 * 3^2 * 5 * 7 * 13) = 0 := by sorry
  have h6 : Nat.num_divisors (2^4 * 3^2 * 5 * 7 * 13) = 120 := by norm_num
  exact (Nat.num_divisors_le (11^60 - 17^24) (2^4 * 3^2 * 5 * 7 * 13) h5).trans h6.ge

end IntegerDivisors_l409_409671


namespace quadrilateral_coloring_5_colors_l409_409422

noncomputable def total_coloring_methods (num_colors : ℕ) (num_vertices : ℕ) : ℕ :=
  if h : num_colors ≥ num_vertices then
    let available_colors := list.range num_colors in
    let colorings := { f : fin num_vertices → fin num_colors // ∀ (u v : fin num_vertices), u ≠ v → f u ≠ f v } in
    fintype.card colorings
  else 0

theorem quadrilateral_coloring_5_colors :
  total_coloring_methods 5 4 = 420 :=
sorry

end quadrilateral_coloring_5_colors_l409_409422


namespace man_walking_time_l409_409440

theorem man_walking_time (D V_w V_m T : ℝ) (t : ℝ) :
  D = V_w * T →
  D_w = V_m * t →
  D - V_m * t = V_w * (T - t) →
  T - (T - t) = 16 →
  t = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end man_walking_time_l409_409440


namespace sin_A_lt_sin_C_l409_409566

theorem sin_A_lt_sin_C {A B C : ℝ} (hA : A < B) (hB : B < C) (hC : C ≠ π / 2) (h_sum : A + B + C = π) :
  sin A < sin C :=
sorry

end sin_A_lt_sin_C_l409_409566


namespace smallest_missing_digit_l409_409327

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l409_409327


namespace smallest_not_odd_unit_is_zero_l409_409189

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l409_409189


namespace perimeter_difference_l409_409724

-- Definitions for the conditions
def num_stakes_sheep : ℕ := 96
def interval_sheep : ℕ := 10
def num_stakes_horse : ℕ := 82
def interval_horse : ℕ := 20

-- Definition for the perimeters
def perimeter_sheep : ℕ := num_stakes_sheep * interval_sheep
def perimeter_horse : ℕ := num_stakes_horse * interval_horse

-- Definition for the target difference
def target_difference : ℕ := 680

-- The theorem stating the proof problem
theorem perimeter_difference : perimeter_horse - perimeter_sheep = target_difference := by
  sorry

end perimeter_difference_l409_409724


namespace initial_rate_of_interest_l409_409402
-- Import necessary library

-- Define the conditions
variables (R : ℝ) (initial_investment additional_investment total_investment total_interest desired_additional_interest desired_total_interest : ℝ)
variables (initial_investment_eq : initial_investment = 8000)
variables (additional_investment_eq : additional_investment = 4000)
variables (total_investment_eq : total_investment = 12000)
variables (desired_additional_interest_eq : desired_additional_interest = 320)
variables (desired_total_interest_eq : desired_total_interest = 720)

-- Define the statement to prove
theorem initial_rate_of_interest :
  (initial_investment * R / 100 + desired_additional_interest = desired_total_interest) →
  R = 5 :=
begin
  sorry
end

end initial_rate_of_interest_l409_409402


namespace translate_parabola_upwards_l409_409735

theorem translate_parabola_upwards (x y : ℝ) (h : y = x^2) : y + 1 = x^2 + 1 :=
by
  sorry

end translate_parabola_upwards_l409_409735


namespace arithmetic_sequence_properties_l409_409528

noncomputable def a : ℕ → ℕ := λ n, 2 * n

noncomputable def S : ℕ → ℕ := λ n, n * (n + 1)

theorem arithmetic_sequence_properties :
  let a₁ := 2 in
  let a₂ := 4 in
  let a₃ := 6 in
  a₁ + a₂ + a₃ = 12 ∧
  ∀ n : ℕ, a n = 2 * n ∧
  ∀ n : ℕ, S n = n * (n + 1) ∧
  (∑ n in finset.range 100, 1 / (S (n + 1)) = 100 / 101) :=
by sorry

end arithmetic_sequence_properties_l409_409528


namespace perfect_square_condition_l409_409613

theorem perfect_square_condition (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 ↔ 4a - 3b = 5 * k :=
by  
  sorry

end perfect_square_condition_l409_409613


namespace smallest_unfound_digit_in_odd_units_l409_409232

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l409_409232


namespace paityn_red_hats_l409_409033

theorem paityn_red_hats (R : ℕ) : 
  (R + 24 + (4 / 5) * ↑R + 48 = 108) → R = 20 :=
by
  intro h
  sorry


end paityn_red_hats_l409_409033


namespace robins_total_pieces_of_gum_l409_409677

theorem robins_total_pieces_of_gum :
  let initial_packages := 27
  let pieces_per_initial_package := 18
  let additional_packages := 15
  let pieces_per_additional_package := 12
  let more_packages := 8
  let pieces_per_more_package := 25
  (initial_packages * pieces_per_initial_package) +
  (additional_packages * pieces_per_additional_package) +
  (more_packages * pieces_per_more_package) = 866 :=
by
  sorry

end robins_total_pieces_of_gum_l409_409677


namespace spoons_needed_to_fill_cup_l409_409073

-- Define necessary conditions
def spoon_capacity : Nat := 5
def liter_to_milliliters : Nat := 1000

-- State the problem
theorem spoons_needed_to_fill_cup : liter_to_milliliters / spoon_capacity = 200 := 
by 
  -- Skip the actual proof
  sorry

end spoons_needed_to_fill_cup_l409_409073


namespace smallest_digit_not_in_units_place_of_odd_l409_409295

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l409_409295


namespace cos_ninety_degrees_l409_409920

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l409_409920


namespace cos_ninety_degrees_l409_409921

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l409_409921


namespace smallest_digit_not_found_in_units_place_of_odd_number_l409_409143

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l409_409143


namespace ferris_wheel_time_to_15_feet_above_bottom_l409_409387

theorem ferris_wheel_time_to_15_feet_above_bottom :
  ∀ (r : ℝ) (T : ℝ),
  r = 30 → T = 120 →
  ∃ t : ℝ, (30 * (Real.cos ((2 * Real.pi * t) / T)) + r) = 45 ∧ t = 20 :=
by
  intros r T hr hT
  use 20
  split
  · rw [hr, hT]
    norm_num
    rw [Real.cos_eq_one_div_two]
    norm_num
  · norm_num
    sorry

end ferris_wheel_time_to_15_feet_above_bottom_l409_409387


namespace area_of_triangle_PEQ_lemma_area_of_triangle_PEQ_l409_409598

noncomputable theory

open_locale real

def point : Type := ℝ × ℝ

def square (A B C D : point) : Prop :=
  A = (0, 0) ∧ B = (2, 0) ∧ C = (2, 2) ∧ D = (0, 2)

def on_side_AB (A B P Q : point) (a b : ℝ) : Prop :=
  P = (a, 0) ∧ Q = (b, 0) ∧ 0 < a ∧ a < 2 ∧ 0 < b ∧ b < 2

theorem area_of_triangle_PEQ_lemma
  (A B C D P Q : point)
  (h_square : square A B C D)
  (h_on_side_AB : on_side_AB A B P Q (1/2) (3/2)) :
  ∃ E : point, (E = C ∨ E = D) ∧ (let (PEQ_area := 1) in PEQ_area = 1) :=
begin
  sorry
end

theorem area_of_triangle_PEQ 
  (A B C D P Q : point)
  (h_square : square A B C D)
  (h_on_side_AB : on_side_AB A B P Q (1/2) (3/2)) :
  ∃ E : point, (E = C ∨ E = D) ∧ (let (area : ℝ) := if E = C then 1 else 1 in area = 1) :=
begin
  apply area_of_triangle_PEQ_lemma,
  exact h_square,
  exact h_on_side_AB
end

end area_of_triangle_PEQ_lemma_area_of_triangle_PEQ_l409_409598


namespace smallest_digit_not_found_in_units_place_of_odd_number_l409_409139

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l409_409139


namespace quadratic_equation_must_be_minus_2_l409_409075

-- Define the main problem statement
theorem quadratic_equation_must_be_minus_2 (m : ℝ) :
  (∀ x : ℝ, (m - 2) * x ^ |m| - 3 * x - 7 = 0) →
  (∀ (h : |m| = 2), m - 2 ≠ 0) →
  m = -2 :=
sorry

end quadratic_equation_must_be_minus_2_l409_409075


namespace smallest_digit_not_in_units_place_of_odd_l409_409260

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l409_409260


namespace red_flowers_killed_by_fungus_l409_409777

theorem red_flowers_killed_by_fungus : 
  let total_bouquets := 36
  let flowers_per_bouquet := 9
  let initial_seeds := 125
  let killed_yellow := 61
  let killed_orange := 30
  let killed_purple := 40
  let survived_yellow := initial_seeds - killed_yellow
  let survived_orange := initial_seeds - killed_orange
  let survived_purple := initial_seeds - killed_purple
  let total_needed_flowers := total_bouquets * flowers_per_bouquet
  let surviving_flowers_except_red := survived_yellow + survived_orange + survived_purple
  let needed_red := total_needed_flowers - surviving_flowers_except_red
  let killed_red := initial_seeds - needed_red
  killed_red = 45 := 
by {
  let total_bouquets := 36
  let flowers_per_bouquet := 9
  let initial_seeds := 125
  let killed_yellow := 61
  let killed_orange := 30
  let killed_purple := 40
  let survived_yellow := initial_seeds - killed_yellow
  let survived_orange := initial_seeds - killed_orange
  let survived_purple := initial_seeds - killed_purple
  let total_needed_flowers := total_bouquets * flowers_per_bouquet
  let surviving_flowers_except_red := survived_yellow + survived_orange + survived_purple
  let needed_red := total_needed_flowers - surviving_flowers_except_red
  let killed_red := initial_seeds - needed_red
  have : survived_yellow = 64 := rfl
  have : survived_orange = 95 := rfl
  have : survived_purple = 85 := rfl
  have : total_needed_flowers = 324 := rfl
  have : surviving_flowers_except_red = 244 := rfl
  have : needed_red = 80 := rfl
  have : killed_red = 45 := rfl
  exact rfl
}

end red_flowers_killed_by_fungus_l409_409777


namespace find_a_l409_409523

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem find_a (a : ℝ) (h : binomial_coefficient 4 2 + 4 * a = 10) : a = 1 :=
by
  sorry

end find_a_l409_409523


namespace number_of_Q_polynomials_l409_409007

noncomputable def P (x : ℂ) : ℂ := (x - 1) * (x - 3) * (x - 5)

theorem number_of_Q_polynomials :
  ∃ Q : polynomial ℂ, 
    (∃ R : polynomial ℂ, degree R = 4 ∧ P(Q.eval x) = P x * R.eval x) → 
    (Q.degree = 3 ∧ {Q.eval 1, Q.eval 3, Q.eval 5} ⊆ {1, 3, 5}.card = 22) :=
sorry

end number_of_Q_polynomials_l409_409007


namespace smallest_digit_not_in_units_place_of_odd_l409_409282

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l409_409282


namespace smallest_missing_digit_l409_409315

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l409_409315


namespace evaluate_expression_l409_409629

noncomputable def a := Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2
noncomputable def b := -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2
noncomputable def c := Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2
noncomputable def d := -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2

theorem evaluate_expression : (1 / a + 1 / b + 1 / c + 1 / d)^2 = 39 / 140 := 
by
  sorry

end evaluate_expression_l409_409629


namespace percentage_difference_l409_409368

theorem percentage_difference :
  let x := 50
  let y := 30
  let p1 := 60
  let p2 := 30
  (p1 / 100 * x) - (p2 / 100 * y) = 21 :=
by
  sorry

end percentage_difference_l409_409368


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409334

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409334


namespace price_for_3years_service_l409_409680

def full_price : ℝ := 85
def discount_price_1year (price : ℝ) : ℝ := price - (0.20 * price)
def discount_price_3years (price : ℝ) : ℝ := price - (0.25 * price)

theorem price_for_3years_service : discount_price_3years (discount_price_1year full_price) = 51 := 
by 
  sorry

end price_for_3years_service_l409_409680


namespace smallest_not_odd_unit_is_zero_l409_409186

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l409_409186


namespace similar_triangles_find_QR_l409_409375

theorem similar_triangles_find_QR :
  ∀ (XY YZ PQ QR : ℝ), 
    XY = 7 → 
    YZ = 10 → 
    PQ = 4 → 
    QR = (40 / 7) →
    QR = 5.7 :=
by
  intros XY YZ PQ QR hXY hYZ hPQ hQR
  rw [hXY, hYZ, hPQ, hQR]
  norm_num
  sorry

end similar_triangles_find_QR_l409_409375


namespace common_tangent_lines_l409_409511

-- Define the conditions of the circles
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def circle_E (x y : ℝ) : Prop := x^2 + (y - real.sqrt 3)^2 = 1

-- Define the tangent lines
def tangent_line_1 (x y : ℝ) : Prop := y = -real.sqrt 3 * x + real.sqrt 3 + 2
def tangent_line_2 (x y : ℝ) : Prop := y = -real.sqrt 3 * x + real.sqrt 3 - 2
def tangent_line_3 (x y : ℝ) : Prop := x - real.sqrt 3 * y + 1 = 0

theorem common_tangent_lines : 
  ∃ (k b : ℝ), ((∀ (x y : ℝ), circle_C x y → tangent_line_1 x y) ∧ (∀ (x y : ℝ), circle_E x y → tangent_line_1 x y)) ∨
                ((∀ (x y : ℝ), circle_C x y → tangent_line_2 x y) ∧ (∀ (x y : ℝ), circle_E x y → tangent_line_2 x y)) ∨
                ((∀ (x y : ℝ), circle_C x y → tangent_line_3 x y) ∧ (∀ (x y : ℝ), circle_E x y → tangent_line_3 x y)) :=
sorry

end common_tangent_lines_l409_409511


namespace slope_not_in_options_l409_409731

theorem slope_not_in_options
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (h1 : x1 < x2)
  (h2 : x2 < x3)
  (h3 : 2 * (x3 - x2) = x2 - x1) :
  let m := (x1 - x2) * (y1 - (y1 + y2 + y3) / 3) + (1 / 2 * (x2 - x1)) * (y3 - (y1 + y2 + y3) / 3) /
            ((x1 - x2)^2 + (1 / 2 * (x2 - x1))^2)
  in m ≠ (y3 - y1) / (x3 - x1) ∧
     m ≠ (y2 - y1) - (y3 - y2) / (x3 - x1) ∧
     m ≠ (3y3 - 2y1 - y2) / (3x3 - 2x1 - x2) ∧
     m ≠ (2y3 - y1) / (3x3 - 2x1) :=
by
  sorry

end slope_not_in_options_l409_409731


namespace smallest_digit_not_in_units_place_of_odd_numbers_l409_409150

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l409_409150


namespace smallest_digit_not_in_odd_units_l409_409215

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l409_409215


namespace smallest_not_odd_unit_is_zero_l409_409187

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l409_409187


namespace max_value_sqrt_sum_l409_409480

theorem max_value_sqrt_sum (x : ℝ) (h : -20 ≤ x ∧ x ≤ 30) : 
  ∃ M, (∀ y, -20 ≤ y ∧ y ≤ 30 → sqrt (30 + y) + sqrt (30 - y) ≤ M) ∧ M = 2 * sqrt 30 :=
by 
  use 2 * sqrt 30
  intros y hy
  sorry

end max_value_sqrt_sum_l409_409480


namespace sequence_properties_l409_409508

noncomputable def a : ℕ → ℤ
| 0       := 0  -- a₀ is not defined in the original problem, let's set it to 0
| 1       := -2
| (n+2) := a (n+1) * (2 * (n + 2) / (n + 1))

def Sn (n : ℕ) : ℤ :=
  ∑ i in finset.range n.succ, a i

theorem sequence_properties : 
  (a 2 = -8) ∧ (∀ n, n ≥ 1 → a n = -n * 2^n) ∧ (Sn 3 ≠ -30) ∧ (Sn n = (1 - n) * 2^(n + 1) - 2) := by
  sorry

end sequence_properties_l409_409508


namespace quadratic_to_vertex_form_addition_l409_409584

theorem quadratic_to_vertex_form_addition (a h k : ℝ) (x : ℝ) :
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 :=
by
  intro h_eq
  sorry

end quadratic_to_vertex_form_addition_l409_409584


namespace smallest_n_l409_409346

theorem smallest_n (n : ℕ) (hn1 : (5 * n) pow 2) (hn2 : (4 * n) pow 3) : n = 80 :=
begin
  -- sorry statement to skip the proof.
  sorry
end

end smallest_n_l409_409346


namespace tan_double_angle_l409_409530

theorem tan_double_angle (θ : ℝ) (h1 : vertex.of θ = origin) (h2 : initial.side θ = positive.x.axis) (h3 : terminal.side θ ∈ line (λ x, 2 * x)) :
  Real.tan (2 * θ) = -4 / 3 := by
  sorry

end tan_double_angle_l409_409530


namespace fibonacci_sequence_contains_one_l409_409372

def fib : ℕ → ℕ
| 0       := 1
| 1       := 1
| (n + 2) := fib (n + 1) + fib n

def seq_x (k m : ℕ) : ℕ → ℚ
| 0       := (fib k : ℚ) / (fib m)
| (n + 1) := if seq_x n = 1 then 1 else (2 * seq_x n - 1) / (1 - seq_x n)

theorem fibonacci_sequence_contains_one (k m : ℕ) :
  ∃ i : ℕ, k = 2 * i ∧ m = 2 * i + 1 → ∃ n, seq_x k m n = 1 :=
by {
  sorry
}

end fibonacci_sequence_contains_one_l409_409372


namespace smallest_unfound_digit_in_odd_units_l409_409223

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l409_409223


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409329

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409329


namespace connie_gives_back_the_amount_l409_409823

/-
Blake gave Connie $20,000, Connie bought some land with this money, 
the land tripled in value after one year, she sold the land and gave 
Blake half of the money from the sale. Prove that Connie gave 
Blake $30,000.
-/
theorem connie_gives_back_the_amount : 
  let initial_amount : ℝ := 20000
  let tripled_amount : ℝ := 3 * initial_amount
  let amount_given_back : ℝ := 0.5 * tripled_amount
  amount_given_back = 30000 :=
by 
  let initial_amount : ℝ := 20000
  let tripled_amount : ℝ := 3 * initial_amount
  let amount_given_back : ℝ := 0.5 * tripled_amount
  show amount_given_back = 30000 from sorry

end connie_gives_back_the_amount_l409_409823


namespace probability_product_of_rolls_l409_409799

theorem probability_product_of_rolls :
  let dice := [1, 2, 3, 4, 5, 6] in
  (∃ (rolls : list ℕ) (h : rolls.length = 8),
    (∀ r ∈ rolls, r ∈ dice) ∧ 
    ((∀ r ∈ rolls, r % 2 = 1) ∨ 
     (∃! r ∈ rolls, r = 2 ∧ ∀ s ∈ (rolls.erase r), s % 2 = 1))
  ) →
  (list.prob (λ rolls, (∀ r ∈ rolls, r % 2 = 1) ∨ 
                    (∃! r ∈ rolls, r = 2 ∧ ∀ s ∈ (rolls.erase r), s % 2 = 1))
              (list.replicate 8 dice)) = 11 / 768 :=
sorry

end probability_product_of_rolls_l409_409799


namespace solve_xy_l409_409548

noncomputable def sequence_transform (n : ℕ) (S : List ℝ) : List ℝ :=
List.init n (λ k, (S[k] + S[k+1]) / 2)

noncomputable def product_transform (n : ℕ) (T : List ℝ) : List ℝ :=
List.init n (λ k, (T[k] * T[k+1]) / 2)

noncomputable def C (n m : ℕ) (S T : List ℝ) : ℝ :=
(sequence_transform n S).head + (product_transform n T).head

theorem solve_xy (x y : ℝ) (S T : List ℝ) :
  (∀ n : ℕ, n ≥ 1 → S = List.range' 0 n (λ k, x^k)) →
  (∀ n : ℕ, n ≥ 1 → T = List.range' 0 n (λ k, y^k)) →
  (x > 0 ∧ y > 0) →
  (C 150 100 S T = 2^(-75)) →
  x = Real.sqrt 2 - 1 ∧ y = 1 - Real.sqrt 2 / 5050 :=
by
  sorry

end solve_xy_l409_409548


namespace bicycle_rental_net_income_l409_409392

theorem bicycle_rental_net_income :
  ∃ (x : ℕ), 
  (1 ≤ x ∧ x ≤ 40) ∧
  (∃ f : ℕ → ℤ,
    (∀ x, 1 ≤ x ∧ x ≤ 5 → f x = 40 * x - 92) ∧
    (∀ x, 6 ≤ x ∧ x ≤ 40 → f x = -2 * x^2 + 50 * x - 92) ∧
    (∀ y : ℕ, (1 ≤ y ∧ y ≤ 40) → f y ≤ 220) ∧
    (f 12 = 220 ∧ f 13 = 220)) :=
begin
  apply Exists.intro 12,
  split,
  -- condition x = 12 is within the domain
  exact ⟨by norm_num, by norm_num⟩,
  -- function definition, domain, and max income
  apply Exists.intro (λ x : ℕ, if (1 ≤ x ∧ x ≤ 5) then 40 * x - 92 else -2 * (x^2 : ℕ) + 50 * x - 92),
  split,
  { intros x h,
    simp [h], },
  split,
  { intros x h,
    simp [h], },
  split,
  { intros y h,
    simp [h],
    have hy : y <= 5 ∨ y > 5 := by omega,
    cases hy,
    { rw if_pos,
      simp [hy],
      omega, },
    { rw if_neg,
      rw if_neg,
      simp [hy],
      omega, }, },
  split,
  { split,
    { simp, },
    { simp, } }
end.

end bicycle_rental_net_income_l409_409392


namespace weight_of_new_student_l409_409758

theorem weight_of_new_student (weight_29_avg weight_30_avg : ℝ) (weight_29_avg = 28) (weight_30_avg = 27.8) : 
  ∃ weight_new_student: ℝ, weight_new_student = 22 :=
by
  -- conditions used for calculations
  let total_weight_29 := 29 * weight_29_avg
  let total_weight_30 := 30 * weight_30_avg
  let weight_new_student := total_weight_30 - total_weight_29
  use weight_new_student
  sorry

end weight_of_new_student_l409_409758


namespace smallest_n_l409_409349

theorem smallest_n (n : ℕ) (h₁ : ∃ k₁ : ℕ, 5 * n = k₁ ^ 2) (h₂ : ∃ k₂ : ℕ, 4 * n = k₂ ^ 3) : n = 1600 :=
sorry

end smallest_n_l409_409349


namespace cos_ninety_degrees_l409_409917

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l409_409917


namespace bobby_candy_l409_409417

theorem bobby_candy (C G : ℕ) (H : C + G = 36) (Hchoc: (2/3 : ℚ) * C = 12) (Hgummy: (3/4 : ℚ) * G = 9) : 
  (1/3 : ℚ) * C + (1/4 : ℚ) * G = 9 :=
by
  sorry

end bobby_candy_l409_409417


namespace smallest_digit_never_in_units_place_of_odd_numbers_l409_409299

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l409_409299


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409335

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409335


namespace infinitely_many_digitally_divisible_integers_l409_409809

theorem infinitely_many_digitally_divisible_integers :
  ∀ n : ℕ, ∃ k : ℕ, k = (10 ^ (3 ^ n) - 1) / 9 ∧ (3 ^ n ∣ k) :=
by
  sorry

end infinitely_many_digitally_divisible_integers_l409_409809


namespace find_k_l409_409483

theorem find_k (k : ℕ) (hk : 0 < k) (h : (k + 4) / (k^2 - 1) = 9 / 35) : k = 14 :=
by
  sorry

end find_k_l409_409483


namespace area_of_triangle_ABC_l409_409590

theorem area_of_triangle_ABC :
  let A := (0 : ℝ, 0 : ℝ)
  let B := (1 : ℝ, 2 : ℝ)
  let C := (3 : ℝ, 0 : ℝ)
  (1 / 2) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)| = 3 := by
  let A := (0 : ℝ, 0 : ℝ)
  let B := (1 : ℝ, 2 : ℝ)
  let C := (3 : ℝ, 0 : ℝ)
  sorry

end area_of_triangle_ABC_l409_409590


namespace fraction_product_l409_409744

theorem fraction_product : 
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_l409_409744


namespace cos_pi_half_eq_zero_l409_409869

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l409_409869


namespace mary_six_digit_numbers_l409_409027

/-- Prove the number of different six-digit numbers Mary could have typed,
    given that she typed two 1s that didn't show and the displayed number was 2002. -/
theorem mary_six_digit_numbers : 
  let six_digit_num := "2002"
  in (Σ n : ℕ, 200000 ≤ n ∧ n < 300000 ∧ (toString n).intersect (toString six_digit_num) ∩ {'0', '2'} = {'0', '2'}) = 15 :=
by
  sorry

end mary_six_digit_numbers_l409_409027


namespace find_angle_ADB_l409_409090

-- Define the angles in the pentagon
def angle_ABC : ℝ := 104
def angle_BAE : ℝ := 104
def angle_CDE : ℝ := 104

-- Assume the sum of internal angles of the pentagon
def sum_internal_angles_pentagon : ℝ := 540

-- Axiom stating the opposite angles of a circumscribed polygon add up to 180 degrees
axiom circumscribed_opposite_angles (a b : ℝ) : a + b = 180

-- Definition of the angles BCD and DEA
def angle_BCD : ℝ := 114
def angle_DEA : ℝ := 114

-- Main theorem
theorem find_angle_ADB : ∠ADB = 38 :=
by
  sorry -- Proof goes here

end find_angle_ADB_l409_409090


namespace cos_90_eq_0_l409_409881

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l409_409881


namespace smallest_digit_never_in_units_place_of_odd_numbers_l409_409302

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l409_409302


namespace smallest_digit_not_in_units_place_of_odd_l409_409266

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l409_409266


namespace price_after_discounts_l409_409682

theorem price_after_discounts (full_price : ℝ) (price_after_first_discount : ℝ) (price_after_second_discount : ℝ) : 
  full_price = 85 → 
  price_after_first_discount = full_price * (1 - 0.20) → 
  price_after_second_discount = price_after_first_discount * (1 - 0.25) → 
  price_after_second_discount = 51 :=
by
  intro h1 h2 h3
  rw h1 at h2
  rw h2 at h3
  rw h3
  sorry

end price_after_discounts_l409_409682


namespace minimum_a_l409_409498

-- Define the main conditions
def f (ω x : ℝ) : ℝ := sin(ω * x) ^ 2 - 1 / 2

def period_of_f (ω : ℝ) := (∀ x : ℝ, f ω (x + π / 2) = f ω x)

def translation (a ω : ℝ) (x : ℝ) := f ω (x - a)

def symmetry_about_origin (g : ℝ → ℝ) :=
  (∀ x : ℝ, g x = -g (-x))

-- Define the constants and conditions
constants (ω a : ℝ)
axiom ω_positive : ω > 0
axiom a_positive : a > 0
axiom period_condition : period_of_f ω

-- Define the resulting function after translation
def g (x : ℝ) : ℝ := - (1 / 2) * cos(4 * x - 4 * a)

theorem minimum_a : a = π / 8 :=
by
  sorry

end minimum_a_l409_409498


namespace min_value_expression_l409_409517

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (a + 2) ^ 2 + (b + 2) ^ 2 = 25 / 2 :=
sorry

end min_value_expression_l409_409517


namespace smallest_not_odd_unit_is_zero_l409_409190

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l409_409190


namespace line_parallel_perpendicular_planes_l409_409004

-- Definitions of parallel and perpendicular relationships between lines and planes
variables {l : Type} [line l]
variables {α β : Type} [plane α] [plane β]

-- Relationships between line and planes
def parallel (l : line) (p : plane) : Prop := sorry -- Define what it means for a line to be parallel to a plane
def perpendicular (l : line) (p : plane) : Prop := sorry -- Define what it means for a line to be perpendicular to a plane
def parallel_planes (p₁ p₂ : plane) : Prop := sorry -- Define what it means for two planes to be parallel
def perpendicular_planes (p₁ p₂ : plane) : Prop := sorry -- Define what it means for two planes to be perpendicular

-- Given conditions and proof goal
theorem line_parallel_perpendicular_planes
  (l : line) (α β : plane)
  (h₁ : parallel l α)
  (h₂ : perpendicular l β) :
  perpendicular_planes α β :=
sorry

end line_parallel_perpendicular_planes_l409_409004


namespace product_repeating_decimal_eq_frac_l409_409419

theorem product_repeating_decimal_eq_frac :
  let s := 0.256256256... in
  let simplified_fract := (3072 / 999).fraction.simplify in
  (12 * s) = simplified_fract :=
by
  sorry

end product_repeating_decimal_eq_frac_l409_409419


namespace distinct_bracelets_l409_409738

-- Definitions of the problem conditions.
def red_beads : Nat := 1
def blue_beads : Nat := 2
def green_beads : Nat := 2
def total_beads : Nat := red_beads + blue_beads + green_beads

-- Theorem stating the number of distinct bracelets.
theorem distinct_bracelets : total_beads = 5 → (factorial total_beads) / (factorial blue_beads * factorial green_beads * total_beads * 2) = 4 := 
by
  sorry

end distinct_bracelets_l409_409738


namespace log_eq_exp_l409_409473

theorem log_eq_exp {x : ℝ} (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end log_eq_exp_l409_409473


namespace perfect_square_condition_l409_409612

theorem perfect_square_condition (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 ↔ 4a - 3b = 5 * k :=
by  
  sorry

end perfect_square_condition_l409_409612


namespace general_term_formula_sum_of_b_n_l409_409499

noncomputable def a_n (n : ℕ) : ℕ := n
def b_n (n : ℕ) : ℕ := 1 / (a_n n * (a_n (n + 1)) * (a_n (n + 2)))

def S_n (n : ℕ) : ℚ := ∑ i in finset.range n, b_n i

theorem general_term_formula :
  ∀ n : ℕ, a_n n = n :=
by
  sorry

theorem sum_of_b_n :
  ∀ n : ℕ, S_n n = (n * (n + 3)) / (4 * (n + 1) * (n + 2)) :=
by
  sorry

end general_term_formula_sum_of_b_n_l409_409499


namespace smallest_digit_not_in_units_place_of_odd_l409_409293

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l409_409293


namespace polynomial_roots_l409_409431

theorem polynomial_roots (d e : ℤ) :
  (∀ r, r^2 - 2 * r - 1 = 0 → r^5 - d * r - e = 0) ↔ (d = 29 ∧ e = 12) := by
  sorry

end polynomial_roots_l409_409431


namespace henrietta_paint_gallons_l409_409554

-- Define the conditions
def living_room_area : Nat := 600
def bedrooms_count : Nat := 3
def bedroom_area : Nat := 400
def coverage_per_gallon : Nat := 600

-- The theorem we want to prove
theorem henrietta_paint_gallons :
  (bedrooms_count * bedroom_area + living_room_area) / coverage_per_gallon = 3 :=
by
  sorry

end henrietta_paint_gallons_l409_409554


namespace passes_through_fixed_point_vertex_on_parabola_larger_root_range_l409_409379

-- Define the quadratic function
def f_a (a x : ℝ) : ℝ := x^2 + (a + 2) * x - 2 * a + 1

-- (1) Prove that the parabola passes through the fixed point (2, 9)
theorem passes_through_fixed_point (a : ℝ) : f_a a 2 = 9 :=
sorry

-- (2) Prove that the vertex of the parabola lies on another parabola
theorem vertex_on_parabola (a : ℝ) : 
  let x_vert := -(a+2)/2 in 
  let y_vert := (x_vert)^2 + (a + 2) * x_vert - 2 * a + 1 in
  y_vert = -(x_vert)^2 + 4 * x_vert + 5 :=
sorry

-- (3) Prove the range of the larger root of the quadratic equation
theorem larger_root_range (a : ℝ) (h : (a + 2)^2 - 4 * 1 * (-2 * a + 1) > 0) : 
  let discriminant := (a+2)^2 - 4 * 1 * (-2 * a + 1) in
  let larger_root := (-(a + 2) + Real.sqrt discriminant) / 2 in
  larger_root ∈ Set.Ioo (-1 : ℝ) 2 ∪ Set.Ioi (5 : ℝ) :=
sorry

end passes_through_fixed_point_vertex_on_parabola_larger_root_range_l409_409379


namespace smallest_digit_not_in_units_place_of_odd_l409_409263

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l409_409263


namespace find_principal_sum_l409_409752

-- Define the principal sum P
-- Define the simple interest formula SI = P * R * T / 100
-- Define the compound interest formula CI = P * (1 + R/100)^T - P

section
variables (P R T : ℝ)

def simple_interest : ℝ := P * R * T / 100
def compound_interest : ℝ := P * (1 + R / 100) ^ T - P

-- Given conditions
def R : ℝ := 10  -- Rate of interest (10% per annum)
def T : ℝ := 2   -- Time period (2 years)
def difference : ℝ := 61  -- Difference between CI and SI

-- The proof statement
theorem find_principal_sum (h : compound_interest P R T - simple_interest P R T = difference) : P = 6100 :=
sorry
end

end find_principal_sum_l409_409752


namespace cos_pi_half_eq_zero_l409_409866

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l409_409866


namespace bottles_per_minute_l409_409684

theorem bottles_per_minute (machines : ℕ) (bottles : ℕ) (minutes : ℕ) (rate_per_machine : ℕ) :
  machines = 6 →
  bottles = 900 →
  minutes = 4 →
  rate_per_machine = 45 →
  (6 * rate_per_machine = 270) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end bottles_per_minute_l409_409684


namespace little_john_money_left_l409_409756

-- Define the variables with the given conditions
def initAmount : ℚ := 5.10
def spentOnSweets : ℚ := 1.05
def givenToEachFriend : ℚ := 1.00

-- The problem statement
theorem little_john_money_left :
  (initAmount - spentOnSweets - 2 * givenToEachFriend) = 2.05 :=
by
  sorry

end little_john_money_left_l409_409756


namespace prime_ge_7_not_divisible_by_40_l409_409564

theorem prime_ge_7_not_divisible_by_40 (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_7 : p ≥ 7) : ¬ (40 ∣ (p^3 - 1)) :=
sorry

end prime_ge_7_not_divisible_by_40_l409_409564


namespace xiao_ming_age_l409_409380

-- Definition of the conditions
def dad_age (x : ℤ) : ℤ := 3 * x
def dad_xm_difference (x : ℤ) : ℤ := dad_age x - x

-- The proof statement
theorem xiao_ming_age : ∃ x : ℤ, dad_xm_difference x = 28 ∧ x = 14 :=
by
  use 14
  simp [dad_xm_difference, dad_age]
  sorry

end xiao_ming_age_l409_409380


namespace cos_90_equals_0_l409_409940

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l409_409940


namespace sin_of_acute_angle_l409_409531

theorem sin_of_acute_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : tan (π - α) + 3 = 0) : 
  sin α = 3 * ( sqrt 10 ) / 10 :=
sorry

end sin_of_acute_angle_l409_409531


namespace part1_geometric_part2_range_of_t_l409_409507

-- Define the sequence {a_n}
def seq_a : ℕ → ℝ
| 0       := t
| (n + 1) := 3 * seq_a n / (2 * seq_a n + 1)

-- Proposition for part (1): the sequence {1/a_n - 1} is geometric
theorem part1_geometric (t : ℝ) (ht : t = (3 / 5)) :
  ∀ n : ℕ, (seq_a t) n ≠ 0 → ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, (1 / seq_a t n) - 1 = (1 / t - 1) * r^(n - 1) :=
sorry

-- Proposition for part (2): determine the range of t
theorem part2_range_of_t (t : ℝ) :
  (∀ n : ℕ, seq_a t (n + 1) > seq_a t n) → 0 < t ∧ t < 1 :=
sorry

end part1_geometric_part2_range_of_t_l409_409507


namespace smallest_digit_not_in_odd_units_l409_409210

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l409_409210


namespace equal_segments_l409_409645

variable {Point : Type}
variables {A B C D E F : Point}
variable {isDiameter : Line}
variable {isChord : Line}
variable isCircle : Circle
variable isNotPerpendicular : ¬(orthogonal AB CD)
variable AE_perpendicular_to_CD : isPerpendicular AE CD
variable BF_perpendicular_to_CD : isPerpendicular BF CD

theorem equal_segments (h1 : isDiameter A B isCircle)
                       (h2 : isChord C D isCircle)
                       (h3 : AE_perpendicular_to_CD)
                       (h4 : BF_perpendicular_to_CD)
                       (h5 : isNotPerpendicular) :
    segmentLength C F = segmentLength D E :=
sorry

end equal_segments_l409_409645


namespace original_average_age_l409_409072

theorem original_average_age (N : ℕ) (A : ℝ) (h1 : A = 50) (h2 : 12 * 32 + N * 50 = (N + 12) * (A - 4)) : A = 50 := by
  sorry 

end original_average_age_l409_409072


namespace cos_of_90_degrees_l409_409986

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l409_409986


namespace shooter_A_better_performance_l409_409364

-- Define the conditions 
variables {A B : Type} [Fintype A] [Fintype B]
variable (scores_A : A → ℝ)
variable (scores_B : B → ℝ)

-- Define average (mean) and standard deviation
def mean (scores : A → ℝ) : ℝ :=
  (∑ a, scores a) / Fintype.card A

def stddev (scores : A → ℝ) : ℝ :=
  real.sqrt ((∑ a, (scores a - mean scores) ^ 2) / Fintype.card A)

-- Define hypothesis
variables (h_mean : mean scores_A > mean scores_B)
variables (h_stddev : stddev scores_A < stddev scores_B)

-- Define the theorem statement
theorem shooter_A_better_performance :
  h_mean ∧ h_stddev → true := by
  sorry

end shooter_A_better_performance_l409_409364


namespace inverse_function_f_inverse_function_domain_l409_409085

def f (x : ℝ) : ℝ := real.log (x - 1)
def inv_f (y : ℝ) : ℝ := real.exp y + 1

theorem inverse_function_f (x : ℝ) (h : x > 2) : 
  f (inv_f x) = x := 
  sorry

theorem inverse_function_domain (x : ℝ) (h : x > 0) : 
  inv_f x > 2 := 
  sorry

end inverse_function_f_inverse_function_domain_l409_409085


namespace vertical_line_properties_l409_409037

theorem vertical_line_properties (x y1 y2 : ℝ) (h : x = 7) (hy : y1 ≠ y2) :
  False ∧ (∀ b : ℝ, ¬(∃ m : ℝ, b = 7 + m * (hy - y1))) :=
sorry

end vertical_line_properties_l409_409037


namespace mr_yadav_expenses_l409_409656

theorem mr_yadav_expenses (S : ℝ) 
  (h1 : S > 0) 
  (h2 : 0.6 * S > 0) 
  (h3 : (12 * 0.2 * S) = 48456) : 
  0.2 * S = 4038 :=
by
  sorry

end mr_yadav_expenses_l409_409656


namespace quadrilateral_area_ratio_l409_409074

-- Definitions based on conditions
structure Quadrilateral :=
(AC BD : ℝ) -- lengths of diagonals
(θ : ℝ) -- angle between diagonals

def doubled_diag (qu : Quadrilateral) : Quadrilateral :=
{ AC := 2 * qu.AC,
  BD := 2 * qu.BD,
  θ := qu.θ }

-- Area of a quadrilateral using diagonals and the angle between them
def area (qu : Quadrilateral) : ℝ :=
1/2 * qu.AC * qu.BD * real.sin qu.θ

-- Theorem statement using the given conditions and question
theorem quadrilateral_area_ratio (qu : Quadrilateral) :
  area qu / area (doubled_diag qu) = 1 / 4 :=
sorry

end quadrilateral_area_ratio_l409_409074


namespace cos_90_deg_eq_zero_l409_409908

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l409_409908


namespace point_to_plane_distance_l409_409670

variables (a b c d x₀ y₀ z₀ : ℝ)

theorem point_to_plane_distance :
  abs (a * x₀ + b * y₀ + c * z₀ + d) / sqrt (a^2 + b^2 + c^2) =
  (distance from (x₀, y₀, z₀) to plane ax + by + cz + d = 0) := sorry

end point_to_plane_distance_l409_409670


namespace smallest_digit_never_in_units_place_of_odd_numbers_l409_409306

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l409_409306


namespace compare_neg_thirds_and_halves_l409_409838

theorem compare_neg_thirds_and_halves : (-1 : ℚ) / 3 > (-1 : ℚ) / 2 :=
by
  sorry

end compare_neg_thirds_and_halves_l409_409838


namespace striped_octopus_has_eight_legs_l409_409571

variable (has_even_legs : ℕ → Prop)
variable (lie_told : ℕ → Prop)

variable (green_leg_count : ℕ)
variable (blue_leg_count : ℕ)
variable (violet_leg_count : ℕ)
variable (striped_leg_count : ℕ)

-- Conditions
axiom even_truth_lie_relation : ∀ n, has_even_legs n ↔ ¬lie_told n
axiom green_statement : lie_told green_leg_count ↔ (has_even_legs green_leg_count ∧ lie_told blue_leg_count)
axiom blue_statement : lie_told blue_leg_count ↔ (has_even_legs blue_leg_count ∧ lie_told green_leg_count)
axiom violet_statement : lie_told violet_leg_count ↔ (has_even_legs blue_leg_count ∧ ¬has_even_legs violet_leg_count)
axiom striped_statement : ¬has_even_legs green_leg_count ∧ ¬has_even_legs blue_leg_count ∧ ¬has_even_legs violet_leg_count ∧ has_even_legs striped_leg_count

-- The Proof Goal
theorem striped_octopus_has_eight_legs : has_even_legs striped_leg_count ∧ striped_leg_count = 8 :=
by
  sorry -- Proof to be filled in

end striped_octopus_has_eight_legs_l409_409571


namespace average_difference_l409_409695

theorem average_difference : 
  (500 + 1000) / 2 - (100 + 500) / 2 = 450 := 
by
  sorry

end average_difference_l409_409695


namespace cos_90_eq_zero_l409_409952

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l409_409952


namespace cos_90_eq_zero_l409_409849

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l409_409849


namespace remainder_of_expansion_mod_88_l409_409420

theorem remainder_of_expansion_mod_88 :
  ((1 - ∑ k in finset.range 11, (-1)^k * (90^k) * (nat.choose 10 k)) % 88 = 1) :=
by
  sorry

end remainder_of_expansion_mod_88_l409_409420


namespace odd_sum_of_digits_has_more_l409_409083

/--
Let E be the set of integers from 1 to 1,000,000 with an even sum of digits.
Let O be the set of integers from 1 to 1,000,000 with an odd sum of digits.
We need to prove that the cardinality of the set O is two more than the cardinality of the set E.
-/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_odd (n : ℕ) : Prop :=
  ¬ is_even n

def E := {n | 1 ≤ n ∧ n ≤ 1000000 ∧ is_even (sum_of_digits n)}
def O := {n | 1 ≤ n ∧ n ≤ 1000000 ∧ is_odd (sum_of_digits n)}

theorem odd_sum_of_digits_has_more:
  |O| = |E| + 2 := 
sorry

end odd_sum_of_digits_has_more_l409_409083


namespace smallest_digit_never_in_units_place_of_odd_l409_409198

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l409_409198


namespace cos_90_deg_eq_zero_l409_409903

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l409_409903


namespace time_to_pass_man_l409_409801

-- Definitions for the conditions
def train_speed_kmph := 54 -- speed of the train in km/hr
def platform_length_m := 360.0288 -- length of the platform in meters
def passing_time_platform_s := 44 -- time to pass the platform in seconds

-- Convert speed to m/s
def train_speed_mps := train_speed_kmph * (1000 / 3600 : Float) -- speed in meters per second
def length_of_train_m := (train_speed_mps * passing_time_platform_s) - platform_length_m -- length of the train in meters
def passing_time_man_s := length_of_train_m / train_speed_mps -- time to pass the man in seconds

-- Statement to prove
theorem time_to_pass_man : passing_time_man_s = 20 :=
by
  sorry

end time_to_pass_man_l409_409801


namespace distance_to_barangay_l409_409764

theorem distance_to_barangay (T1 T2 D: ℝ) 
  (h1: 5 * T1 = D)
  (h2: 3 * T2 = D)
  (h3: T1 + T2 = 4) : 
  D = 7.5 :=
begin
  sorry
end

end distance_to_barangay_l409_409764


namespace cos_90_eq_zero_l409_409949

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l409_409949


namespace smallest_digit_not_in_units_place_of_odd_numbers_l409_409156

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l409_409156


namespace cos_90_equals_0_l409_409938

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l409_409938


namespace maximum_liars_l409_409099

variable (A B C D E : Prop)

def isLiar (X : Prop) : Prop := ¬X

def says (X Y : Prop) : Prop := X → isLiar Y

def liar_count_max (A B C D E : Prop) : Nat :=
  if A then 1 + (if ¬B then 1 + (if C then 1 + (if ¬D then 1 + (if E then 0 else 1) else 0) else 0) else 0) else 
  (if ¬A then 1 + (if B then 1 + (if ¬C then 1 + (if D then 1 + (if ¬E then 1 else 0) else 0) else 0) else 0) else 0) else 0

theorem maximum_liars: ∀ (A B C D E : Prop),
  A ↔ isLiar B →
  B ↔ isLiar C →
  C ↔ isLiar D →
  D ↔ isLiar E →
  liar_count_max A B C D E = 3 :=
by 
  intros A B C D E HAB HBC HCD HDE
  sorry

end maximum_liars_l409_409099


namespace smallest_unfound_digit_in_odd_units_l409_409236

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l409_409236


namespace factor_of_polynomial_l409_409663

theorem factor_of_polynomial :
  (x^4 + 4 * x^2 + 16) % (x^2 + 4) = 0 :=
sorry

end factor_of_polynomial_l409_409663


namespace tan_product_identity_l409_409557

theorem tan_product_identity (A B : ℝ) (hA : A = 20) (hB : B = 25) (hSum : A + B = 45) :
    (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2 := 
  by
  sorry

end tan_product_identity_l409_409557


namespace smallest_digit_not_in_units_place_of_odd_numbers_l409_409158

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l409_409158


namespace smallest_positive_integer_n_l409_409358

theorem smallest_positive_integer_n (n : ℕ) 
  (h1 : ∃ k : ℕ, n = 5 * k ∧ perfect_square(5 * k)) 
  (h2 : ∃ m : ℕ, n = 4 * m ∧ perfect_cube(4 * m)) : 
  n = 625000 :=
sorry

end smallest_positive_integer_n_l409_409358


namespace compare_neg_rational_l409_409835

def neg_one_third : ℚ := -1 / 3
def neg_one_half : ℚ := -1 / 2

theorem compare_neg_rational : neg_one_third > neg_one_half :=
by sorry

end compare_neg_rational_l409_409835


namespace cos_90_deg_eq_zero_l409_409906

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l409_409906


namespace cos_90_eq_zero_l409_409951

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l409_409951


namespace smallest_digit_not_in_odd_units_l409_409211

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l409_409211


namespace cos_of_90_degrees_l409_409989

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l409_409989


namespace cos_ninety_degrees_l409_409922

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l409_409922


namespace domain_of_f_l409_409115

noncomputable def f (x : ℝ) : ℝ :=
  real.sqrt (2 * x - 4) + real.cbrt (4 * x - 6)

theorem domain_of_f :
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≥ 2} :=
by sorry

end domain_of_f_l409_409115


namespace remainder_of_expression_l409_409745

theorem remainder_of_expression (m : ℤ) (h : m % 9 = 3) : (3 * m + 2436) % 9 = 0 := 
by 
  sorry

end remainder_of_expression_l409_409745


namespace tangent_circle_radius_l409_409036

theorem tangent_circle_radius :
  ∀ (A B C : Point) (AB BC AC : ℝ),
    B ∈ line_segment A C →
    AB = 14 →
    BC = 28 →
    AB + BC = AC →
    let R := (7 * 14) / (7 + 14 + 21) in
    R = 7 := 
by 
  sorry

end tangent_circle_radius_l409_409036


namespace existence_of_E_l409_409640

variables {P : Type} [EuclideanGeometry P]

-- Given conditions
variables (A B C D : P) (H : P) (M N S T E : P)

-- Definitions and conditions
def is_cyclic_quadrilateral (A B C D : P) : Prop := sorry
def is_midpoint (M : P) (B C : P) : Prop := sorry
def are_diagonals_perpendicular (A B C D H : P) : Prop := sorry
def are_midpoints (M N : P) (B C D : P) : Prop := sorry
def intersect_ray_at (origin : P) (destination : P) (point : P) : Prop := sorry

-- Prove that there exists a point E such that EH bisects both angles BES and TED, and angle BEN = angle MED
theorem existence_of_E 
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_not_kite : ¬is_kite A B C D)
  (h_perpendicular_diagonals : are_diagonals_perpendicular A B C D H)
  (h_midpoints : are_midpoints M N B C D)
  (h_intersect_ray_MH_AD : intersect_ray_at M H S)
  (h_intersect_ray_NH_AB : intersect_ray_at N H T) :
  ∃ E, bisects_angle E H B S ∧ bisects_angle E H T E ∧ ∠(B, E, N) = ∠(M, E, D) :=
sorry

end existence_of_E_l409_409640


namespace relationship_of_sets_l409_409558

def set_A : Set ℝ := {x | ∃ (k : ℤ), x = (k : ℝ) / 6 + 1}
def set_B : Set ℝ := {x | ∃ (k : ℤ), x = (k : ℝ) / 3 + 1 / 2}
def set_C : Set ℝ := {x | ∃ (k : ℤ), x = (2 * k : ℝ) / 3 + 1 / 2}

theorem relationship_of_sets : set_C ⊆ set_B ∧ set_B ⊆ set_A := by
  sorry

end relationship_of_sets_l409_409558


namespace find_x_l409_409470

theorem find_x (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end find_x_l409_409470


namespace smallest_unfound_digit_in_odd_units_l409_409230

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l409_409230


namespace smallest_digit_never_in_units_place_of_odd_l409_409206

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l409_409206


namespace plywood_width_l409_409787

theorem plywood_width (A L : ℝ) (hA : A = 24) (hL : L = 4) : ∃ W : ℝ, W = 6 := by
  let W := A / L
  have hW : W = 6 := by
    rw [hA, hL]
    exact (24 / 4 : ℝ)
  use W
  exact hW
  sorry

end plywood_width_l409_409787


namespace exists_non_convex_inscribed_polyhedron_l409_409790

theorem exists_non_convex_inscribed_polyhedron :
  ∃ (P : Polyhedron), inscribed_in_sphere P ∧ ¬ is_convex P :=
sorry

end exists_non_convex_inscribed_polyhedron_l409_409790


namespace combined_height_after_1_year_l409_409025

def initial_heights : ℕ := 200 + 150 + 250
def spring_and_summer_growth_A : ℕ := (6 * 4 / 2) * 50
def spring_and_summer_growth_B : ℕ := (6 * 4 / 3) * 70
def spring_and_summer_growth_C : ℕ := (6 * 4 / 4) * 90
def autumn_and_winter_growth_A : ℕ := (6 * 4 / 2) * 25
def autumn_and_winter_growth_B : ℕ := (6 * 4 / 3) * 35
def autumn_and_winter_growth_C : ℕ := (6 * 4 / 4) * 45

def total_growth_A : ℕ := spring_and_summer_growth_A + autumn_and_winter_growth_A
def total_growth_B : ℕ := spring_and_summer_growth_B + autumn_and_winter_growth_B
def total_growth_C : ℕ := spring_and_summer_growth_C + autumn_and_winter_growth_C

def total_growth : ℕ := total_growth_A + total_growth_B + total_growth_C

def combined_height : ℕ := initial_heights + total_growth

theorem combined_height_after_1_year : combined_height = 3150 := by
  sorry

end combined_height_after_1_year_l409_409025


namespace cos_90_eq_zero_l409_409972

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l409_409972


namespace function_characterization_l409_409365

noncomputable def f (x : ℝ) : ℝ := sorry

theorem function_characterization :
  (∀ x, 0 < f x) ∧
  (∀ x1 x2, f (x1 + x2) = 2 * f x1 * f x2) ∧
  (∀ x, f' x < 0) → f = λ x => (1 / 2)^(x + 1) :=
by
  sorry

end function_characterization_l409_409365


namespace trigonometric_simplification_l409_409437

noncomputable def trigonometric_expr (α : ℝ) : ℝ := 
  sin^2 (2 * Real.pi - α) + cos (Real.pi + α) * cos (Real.pi - α) + 1

theorem trigonometric_simplification (α : ℝ) :
  trigonometric_expr α = 2 := by
  have h1 : sin(2 * Real.pi - α) = sin(α) := by sorry
  have h2 : cos(Real.pi + α) = -cos(α) := by sorry
  have h3 : cos(Real.pi - α) = -cos(α) := by sorry
  have h4 : sin^2(α) + cos^2(α) = 1 := by sorry
  sorry

end trigonometric_simplification_l409_409437


namespace stratified_sampling_correct_l409_409778

def num_students := 500
def num_male_students := 500
def num_female_students := 400
def ratio_male_female := num_male_students / num_female_students

def selected_male_students := 25
def selected_female_students := (selected_male_students * num_female_students) / num_male_students

theorem stratified_sampling_correct :
  selected_female_students = 20 :=
by
  sorry

end stratified_sampling_correct_l409_409778


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409331

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409331


namespace sum_of_interior_angles_quadrilateral_l409_409095

-- Define the function for the sum of the interior angles
def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

-- Theorem that the sum of the interior angles of a quadrilateral is 360 degrees
theorem sum_of_interior_angles_quadrilateral : sum_of_interior_angles 4 = 360 :=
by
  sorry

end sum_of_interior_angles_quadrilateral_l409_409095


namespace terrell_lifting_problem_l409_409069

theorem terrell_lifting_problem (w1 w2 w3 n1 n2 : ℕ) (h1 : w1 = 12) (h2 : w2 = 18) (h3 : w3 = 24) (h4 : n1 = 20) :
  60 * n2 = 3 * w1 * n1 → n2 = 12 :=
by
  intros h
  sorry

end terrell_lifting_problem_l409_409069


namespace common_tangent_circles_l409_409497

/-- 
Given a cluster of circles \({C_n}: ({x - n})^2 + ({y - 2n})^2 = n^2\) for \(n \neq 0\),
and a line \(l: y = kx + b\) that is a common tangent to all these circles,
prove that \(k + b = \frac{3}{4}\).
-/
theorem common_tangent_circles (n : ℝ) (k b : ℝ) 
  (h1 : n ≠ 0) 
  (h2 : ∀ n ≠ 0, (∃ x y, ({x - n})^2 + ({y - 2n})^2 = n^2 ∧ y = k*x + b)) 
  : k + b = 3 / 4 :=
sorry

end common_tangent_circles_l409_409497


namespace f_minus_g_abs_gt_2_l409_409642

noncomputable def f (x : ℝ) : ℝ := 
  ∑ i in (Finset.range 1009).filter (λ n, even n), 1 / (x - n)

noncomputable def g (x : ℝ) : ℝ := 
  ∑ i in (Finset.range 1009).filter (λ n, odd n), 1 / (x - n)

theorem f_minus_g_abs_gt_2 (x : ℝ) (h1 : 0 < x) (h2 : x < 2018) (hx : ¬ ∃ (n : ℤ), x = n):
  |f(x) - g(x)| > 2 :=
sorry

end f_minus_g_abs_gt_2_l409_409642


namespace parabola_points_relationship_l409_409515

theorem parabola_points_relationship :
  let y_1 := (-2)^2 + 2 * (-2) - 9
  let y_2 := 1^2 + 2 * 1 - 9
  let y_3 := 3^2 + 2 * 3 - 9
  y_3 > y_2 ∧ y_2 > y_1 :=
by
  sorry

end parabola_points_relationship_l409_409515


namespace contrapositive_of_true_implication_not_false_l409_409424

-- Definitions
def tautology (φ : Prop) : Prop := ∀ (I : Type → Prop), φ
def universal_set (U : Set α) : Prop := ∀ x, x ∈ U
def intersection_complement_empty (S : Set α) : Prop := S ∩ Sᶜ = ∅
def argument_false_premise_true_conclusion (A B : Prop) : Prop := (¬A → B)

-- Theorem to prove
theorem contrapositive_of_true_implication_not_false
  (φ : Prop)
  (U : Set α)
  (S : Set α)
  (A B : Prop) :
  tautology φ ∧ universal_set U ∧ intersection_complement_empty S ∧ argument_false_premise_true_conclusion A B →
  ¬(∀ (p q : Prop), (p → q) → (¬ q → ¬ p) = false) := 
begin
  sorry
end

end contrapositive_of_true_implication_not_false_l409_409424


namespace smallest_digit_not_in_units_place_of_odd_numbers_l409_409149

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l409_409149


namespace f_even_and_period_l409_409080

noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x + π / 2)

theorem f_even_and_period (x : ℝ) : (∀ x, f x = f (- x)) ∧ (∀ x, f (x + π) = f x) :=
by
  sorry

end f_even_and_period_l409_409080


namespace at_least_two_quadratic_polynomials_have_roots_l409_409546

theorem at_least_two_quadratic_polynomials_have_roots 
  (b1 b2 c1 c2 : ℝ)
  (h : ∃ x : ℝ, (x^2 + b1 * x + c1) + (x^2 + b2 * x + c2) + (x^2 + (1/2) * (b1 + b2) * x + (1/2) * (c1 + c2)) = 0) : 
  ∃ i j ∈ ({1, 2, 3} : Finset ℕ), 
    i ≠ j ∧ (
      (i = 1 ∧ ∃ x : ℝ, x^2 + b1 * x + c1 = 0) ∨
      (i = 2 ∧ ∃ x : ℝ, x^2 + b2 * x + c2 = 0) ∨
      (i = 3 ∧ ∃ x : ℝ, x^2 + (1/2) * (b1 + b2) * x + (1/2) * (c1 + c2) = 0)
    ) ∧ (
      (j = 1 ∧ ∃ x : ℝ, x^2 + b1 * x + c1 = 0) ∨
      (j = 2 ∧ ∃ x : ℝ, x^2 + b2 * x + c2 = 0) ∨
      (j = 3 ∧ ∃ x : ℝ, x^2 + (1/2) * (b1 + b2) * x + (1/2) * (c1 + c2) = 0)
    ) := 
  sorry

end at_least_two_quadratic_polynomials_have_roots_l409_409546


namespace traveling_distance_l409_409755

/-- Let D be the total distance from the dormitory to the city in kilometers.
Given the following conditions:
1. The student traveled 1/3 of the way by foot.
2. The student traveled 3/5 of the way by bus.
3. The remaining portion of the journey was covered by car, which equals 2 kilometers.
We need to prove that the total distance D is 30 kilometers. -/ 
theorem traveling_distance (D : ℕ) 
  (h1 : (1 / 3 : ℚ) * D + (3 / 5 : ℚ) * D + 2 = D) : D = 30 := 
sorry

end traveling_distance_l409_409755


namespace three_x_minus_five_y_l409_409022

structure Point :=
(x : ℝ)
(y : ℝ)

def midpoint (A B : Point) : Point :=
{ x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2 }

def DE_meets_q_at_F (D E : Point) (F : Point) : Prop :=
F = midpoint D E

theorem three_x_minus_five_y (D E F : Point) (h : DE_meets_q_at_F D E F) :
  3 * F.x - 5 * F.y = 9 := by
  sorry

end three_x_minus_five_y_l409_409022


namespace smallest_digit_not_in_odd_units_l409_409238

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l409_409238


namespace smallest_not_odd_unit_is_zero_l409_409180

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l409_409180


namespace compare_neg_thirds_and_halves_l409_409839

theorem compare_neg_thirds_and_halves : (-1 : ℚ) / 3 > (-1 : ℚ) / 2 :=
by
  sorry

end compare_neg_thirds_and_halves_l409_409839


namespace compare_neg_rational_l409_409836

def neg_one_third : ℚ := -1 / 3
def neg_one_half : ℚ := -1 / 2

theorem compare_neg_rational : neg_one_third > neg_one_half :=
by sorry

end compare_neg_rational_l409_409836


namespace smallest_digit_not_in_units_place_of_odd_l409_409261

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l409_409261


namespace cost_to_paint_cube_l409_409700

theorem cost_to_paint_cube (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (side_length : ℝ) (total_cost : ℝ) :
  cost_per_kg = 36.50 →
  coverage_per_kg = 16 →
  side_length = 8 →
  total_cost = (6 * side_length^2 / coverage_per_kg) * cost_per_kg →
  total_cost = 876 :=
by
  intros h1 h2 h3 h4
  sorry

end cost_to_paint_cube_l409_409700


namespace max_perfect_square_sums_in_grid_l409_409589

theorem max_perfect_square_sums_in_grid :
  (∀ (a b c d e f g h i : ℕ), 
    ∃ (S1 S2 S3 : ℕ), 
    ({a, b, c, d, e, f, g, h, i} = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    a + b + c = S1 ∧ d + e + f = S2 ∧ g + h + i = S3 ∧ 
    S1 + S2 + S3 = 45 ∧ 
    ∃ (x y : ℕ), (S1 = x^2 ∨ S1 = y^2) ∧ (S2 = x^2 ∨ S2 = y^2) ∧ ¬(S3 = x^2 ∨ S3 = y^2) ∨
    (S3 = x^2 ∨ S3 = y^2) ∧ (S1 = x^2 ∨ S1 = y^2) ∧ ¬(S2 = x^2 ∨ S2 = y^2) ∨ 
    (S2 = x^2 ∨ S2 = y^2) ∧ (S3 = x^2 ∨ S3 = y^2) ∧ ¬(S1 = x^2 ∨ S1 = y^2)) 
    ) → 
    ∃ t u : ℕ, t + u = 2 := 
sorry

end max_perfect_square_sums_in_grid_l409_409589


namespace log_inequality_l409_409673

theorem log_inequality :
  log 2015 2017 > (∑ i in Finset.range 2016 + 1, log 2015 (i + 1)) / 2016 :=
sorry

end log_inequality_l409_409673


namespace arithmetic_sequence_roots_l409_409579

variable {a : ℕ → ℝ}

axiom is_arithmetic_sequence (a : ℕ → ℝ) : Prop
axiom roots_condition (a_3 a_10 : ℝ) : a_3 + a_10 = 3

def a : ℕ → ℝ := sorry

theorem arithmetic_sequence_roots :
  is_arithmetic_sequence a →
  a 3 + a 10 = 3 →
  a 5 + a 8 = 3 :=
by
  intros h_arith h_roots
  sorry

end arithmetic_sequence_roots_l409_409579


namespace total_revenue_from_ticket_sales_l409_409103

/-- Tickets for a show cost 6.00 dollars for adults and 4.50 dollars for children.
    400 tickets were sold, and 200 children's tickets were sold. 
    What was the total revenue from ticket sales? 
--/
theorem total_revenue_from_ticket_sales 
  (price_adult price_child : ℝ)
  (total_tickets child_tickets : ℕ) 
  (price_adult_eq : price_adult = 6) 
  (price_child_eq : price_child = 4.5)
  (total_tickets_eq : total_tickets = 400)
  (child_tickets_eq : child_tickets = 200) :
    let adult_tickets := total_tickets - child_tickets,
        total_revenue := (adult_tickets * price_adult) + (child_tickets * price_child)
    in total_revenue = 2100 := 
by
  sorry

end total_revenue_from_ticket_sales_l409_409103


namespace area_ratio_of_triangle_MNO_XYZ_l409_409604

-- Definitions of the points and segments in the triangle
noncomputable def Triangle (X Y Z G H I M N O : Type)
  [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  [MetricSpace G] [MetricSpace H] [MetricSpace I]
  [MetricSpace M] [MetricSpace N] [MetricSpace O]
  [NormedAddCommGroup X] [NormedAddCommGroup Y] [NormedAddCommGroup Z]
  [NormedAddCommGroup G] [NormedAddCommGroup H] [NormedAddCommGroup I]
  [NormedAddCommGroup M] [NormedAddCommGroup N] [NormedAddCommGroup O] :=
  {
    YG_GZ : YG G Z = 2 / 3,
    XH_HZ : XH H Z = 2 / 3,
    XI_IY : XI I Y = 2 / 3,
    seg_XG_YH_ZI_intersect_at : ∀ p : Type ,
        (p = M) ∨ (p = N) ∨ (p = O) ->
        (∃ XG : LineSegment X G, 
         ∃ YH : LineSegment Y H,
         ∃ ZI : LineSegment Z I,
          p ∈ (XG ∩ YH ∩ ZI)),
  }

-- Main theorem statement
theorem area_ratio_of_triangle_MNO_XYZ :
  ∀ (X Y Z G H I M N O : Type)
    [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
    [MetricSpace G] [MetricSpace H] [MetricSpace I]
    [MetricSpace M] [MetricSpace N] [MetricSpace O]
    [NormedAddCommGroup X] [NormedAddCommGroup Y] [NormedAddCommGroup Z]
    [NormedAddCommGroup G] [NormedAddCommGroup H] [NormedAddCommGroup I]
    [NormedAddCommGroup M] [NormedAddCommGroup N] [NormedAddCommGroup O],

  let triangle := Triangle X Y Z G H I M N O,
  triangle.YG_GZ → 
  triangle.XH_HZ → 
  triangle.XI_IY → 
  triangle.seg_XG_YH_ZI_intersect_at →
  (area_ratio_of_triangle M N O / area_ratio_of_triangle X Y Z) = 27 / 440 := sorry

end area_ratio_of_triangle_MNO_XYZ_l409_409604


namespace smallest_missing_digit_l409_409316

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l409_409316


namespace f_eval_2016_l409_409643

noncomputable def f : ℕ → ℚ :=
  λ x, if x = 2016 then 1/1008 else 0

theorem f_eval_2016 :
  (∃ f : ℕ → ℚ, (∀ k, 1 ≤ k ∧ k ≤ 2015 → f k = (1 : ℚ) / k) ∧ f(2016) = (1 : ℚ) / 1008) :=
begin
  use λ x, if x = 2016 then 1/1008 else 0,
  split,
  { intros k hk,
    cases hk with hk1 hk2,
    exact if_neg (ne_of_lt hk2) },
  { refl }
end

end f_eval_2016_l409_409643


namespace cos_90_eq_zero_l409_409850

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l409_409850


namespace cos_90_equals_0_l409_409931

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l409_409931


namespace smallest_positive_integer_n_l409_409357

theorem smallest_positive_integer_n (n : ℕ) 
  (h1 : ∃ k : ℕ, n = 5 * k ∧ perfect_square(5 * k)) 
  (h2 : ∃ m : ℕ, n = 4 * m ∧ perfect_cube(4 * m)) : 
  n = 625000 :=
sorry

end smallest_positive_integer_n_l409_409357


namespace smallest_missing_digit_l409_409320

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l409_409320


namespace total_legs_of_passengers_l409_409781

theorem total_legs_of_passengers :
  ∀ (total_heads cats cat_legs human_heads normal_human_legs one_legged_captain_legs : ℕ),
  total_heads = 15 →
  cats = 7 →
  cat_legs = 4 →
  human_heads = (total_heads - cats) →
  normal_human_legs = 2 →
  one_legged_captain_legs = 1 →
  ((cats * cat_legs) + ((human_heads - 1) * normal_human_legs) + one_legged_captain_legs) = 43 :=
by
  intros total_heads cats cat_legs human_heads normal_human_legs one_legged_captain_legs h1 h2 h3 h4 h5 h6
  sorry

end total_legs_of_passengers_l409_409781


namespace infinite_n_square_plus_one_divides_factorial_infinite_n_square_plus_one_not_divide_factorial_l409_409056

theorem infinite_n_square_plus_one_divides_factorial :
  ∃ (infinitely_many n : ℕ), (n^2 + 1) ∣ (n!) := sorry

theorem infinite_n_square_plus_one_not_divide_factorial :
  ∃ (infinitely_many n : ℕ), ¬((n^2 + 1) ∣ (n!)) := sorry

end infinite_n_square_plus_one_divides_factorial_infinite_n_square_plus_one_not_divide_factorial_l409_409056


namespace john_streams_hours_per_day_l409_409623

theorem john_streams_hours_per_day :
  (∃ h : ℕ, (7 - 3) * h * 10 = 160) → 
  (∃ h : ℕ, h = 4) :=
sorry

end john_streams_hours_per_day_l409_409623


namespace smallest_missing_digit_l409_409314

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l409_409314


namespace range_of_m_l409_409540

noncomputable def f (x : ℝ) : ℝ :=
  if x >= -1 then x^2 + 3*x + 5 else (1/2)^x

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x > m^2 - m) ↔ -1 ≤ m ∧ m ≤ 2 := sorry

end range_of_m_l409_409540


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409339

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409339


namespace find_x_value_l409_409459

theorem find_x_value (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end find_x_value_l409_409459


namespace eval_expression_l409_409360

theorem eval_expression : 4^3 - 2 * 4^2 + 2 * 4 - 1 = 39 :=
by 
  -- Here we would write the proof, but according to the instructions we skip it with sorry.
  sorry

end eval_expression_l409_409360


namespace radius_of_circle_l409_409810

theorem radius_of_circle (r : ℝ) : 
    (∃ (r : ℝ), let P := 4 * r * Real.sqrt 2 
                let A := π * r^2 
                (P = 2 * A)) → 
    r = 2 * Real.sqrt 2 / π :=
begin
  sorry
end

end radius_of_circle_l409_409810


namespace cos_90_eq_0_l409_409888

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l409_409888


namespace adoption_fee_cost_l409_409622

-- Definitions based on conditions
def total_vet_cost : ℕ := 500
def monthly_food_cost : ℕ := 25
def toy_cost_jenny : ℕ := 200
def total_expenses_jenny : ℕ := 625
def jenny_share_factor : ℕ := 2

def annual_food_cost (monthly_food_cost : ℕ) : ℕ := monthly_food_cost * 12
def shared_cost (cost : ℕ) (factor : ℕ) : ℕ := cost / factor

-- Theorem to prove
theorem adoption_fee_cost 
  (total_vet_cost monthly_food_cost toy_cost_jenny total_expenses_jenny jenny_share_factor : ℕ)
  (h1 : total_vet_cost = 500)
  (h2 : monthly_food_cost = 25)
  (h3 : toy_cost_jenny = 200)
  (h4 : total_expenses_jenny = 625)
  (h5 : jenny_share_factor = 2) :
  let annual_food_cost := annual_food_cost monthly_food_cost,
      vet_cost_shared := shared_cost total_vet_cost jenny_share_factor,
      food_cost_shared := shared_cost annual_food_cost jenny_share_factor,
      non_adoption_expense := vet_cost_shared + food_cost_shared + toy_cost_jenny,
      jenny_adoption_fee := (total_expenses_jenny - non_adoption_expense)
  in jenny_adoption_fee * jenny_share_factor = 50 :=
by {
  let annual_food_cost := monthly_food_cost * 12,
  let vet_cost_shared := total_vet_cost / jenny_share_factor,
  let food_cost_shared := annual_food_cost / jenny_share_factor,
  let non_adoption_expense := vet_cost_shared + food_cost_shared + toy_cost_jenny,
  let jenny_adoption_fee := total_expenses_jenny - non_adoption_expense,
  show (jenny_adoption_fee * jenny_share_factor = 50),
  sorry
}

end adoption_fee_cost_l409_409622


namespace determine_a_and_b_l409_409066

variable {α : Type*} [LinearOrderedField α]
variables {a b : α}

theorem determine_a_and_b (a_ne_1 : a ≠ 1) 
  (h : ({1, a, b} : Set α) = {a, a^2, a * b}) : 
  a = -1 ∧ b = 0 := 
by 
  sorry

end determine_a_and_b_l409_409066


namespace wolf_cannot_catch_roe_deer_l409_409802

theorem wolf_cannot_catch_roe_deer :
  ∀ (x y : ℝ) (t : ℕ),
    (t < 28.2051 ∧ t > 0) →
    0.78 * (1 + t/100) < 1 :=
by
  assume x y t h
  sorry

end wolf_cannot_catch_roe_deer_l409_409802


namespace cos_90_eq_0_l409_409878

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l409_409878


namespace cos_of_90_degrees_l409_409993

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l409_409993


namespace fixed_point_l409_409578

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (2 + x) + 1

theorem fixed_point (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : f a (-1) = 1 :=
by
  sorry

end fixed_point_l409_409578


namespace smallest_digit_never_in_units_place_of_odd_numbers_l409_409304

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l409_409304


namespace necessary_but_not_sufficient_condition_l409_409514

-- Define the polyhedra and conditions
structure Tetrahedron where
  vertices : list (ℝ × ℝ × ℝ)
  is_regular_projection : Prop -- p: Regular based on the vertex projection
  is_regular_all_edges : Prop -- q: Regular based on all edges

-- Formalize the propositions p and q
def p (T : Tetrahedron) : Prop := T.is_regular_projection
def q (T : Tetrahedron) : Prop := T.is_regular_all_edges

-- Statement: Proposition p is a necessary but not sufficient condition for q
theorem necessary_but_not_sufficient_condition (T : Tetrahedron) : (p T → q T) ∧ (¬ (q T → p T)) := sorry

end necessary_but_not_sufficient_condition_l409_409514


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409336

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409336


namespace B_start_time_after_A_l409_409405

def A_speed := 5 -- speed of A in kmph
def B_speed := 5.555555555555555 -- speed of B in kmph
def overtaking_time := 1 + 48/60 -- time in hours for B to overtake A

theorem B_start_time_after_A : 
  let T := 0.2 in
  (overtaking_time = 2 - T) -> T * 60 = 12 := 
  by
    -- Converting heights and angles to respective measures
    let T := 0.2 -- T in hours
    have h1 : overtaking_time = 1.8 := rfl
    have h2 : (A_speed * (T + overtaking_time) = B_speed * overtaking_time) := sorry
    show T * 60 = 12 from sorry

end B_start_time_after_A_l409_409405


namespace empty_truck_weight_l409_409438

-- Define the constants and variables based on the problem conditions
def bridge_weight_limit : ℕ := 20000
def crates_of_soda : ℕ := 20
def weight_per_crate : ℕ := 50
def dryers : ℕ := 3
def weight_per_dryer : ℕ := 3000
def truck_loaded_weight : ℕ := 24000

-- Define the calculated weights
def weight_of_soda := crates_of_soda * weight_per_crate
def weight_of_produce := 2 * weight_of_soda
def weight_of_dryers := dryers * weight_per_dryer
def total_cargo_weight := weight_of_soda + weight_of_produce + weight_of_dryers

-- Formulate the proof problem statement
theorem empty_truck_weight : ℕ :=
  truck_loaded_weight - total_cargo_weight = 12000 := 
by {
  -- Solve for the empty truck weight
  sorry
}

end empty_truck_weight_l409_409438


namespace sequence_value_2016_l409_409543

theorem sequence_value_2016 :
  ∀ (a : ℕ → ℤ),
    a 1 = 3 →
    a 2 = 6 →
    (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) →
    a 2016 = -3 :=
by
  sorry

end sequence_value_2016_l409_409543


namespace price_for_3years_service_l409_409679

def full_price : ℝ := 85
def discount_price_1year (price : ℝ) : ℝ := price - (0.20 * price)
def discount_price_3years (price : ℝ) : ℝ := price - (0.25 * price)

theorem price_for_3years_service : discount_price_3years (discount_price_1year full_price) = 51 := 
by 
  sorry

end price_for_3years_service_l409_409679


namespace sum_of_digits_of_numeric_hexadecimals_count_l409_409555

def is_numeric_hexadecimal (n : ℕ) : Prop :=
  ∀ c ∈ n.toString 16, c.isDigit

def count_numeric_hexadecimals (limit : ℕ) : ℕ :=
  (Finset.range limit).filter (λ n, is_numeric_hexadecimal n).card

theorem sum_of_digits_of_numeric_hexadecimals_count :
  let n := count_numeric_hexadecimals 2000 in
  (n / 1000) + ((n % 1000) / 100) + ((n % 100) / 10) + (n % 10) = 28 :=
by
  sorry

end sum_of_digits_of_numeric_hexadecimals_count_l409_409555


namespace suff_but_not_necc_l409_409373

theorem suff_but_not_necc : 
  ∀ x : ℝ, (x * (x - 5) < 0) → (x - 1).abs < 4 ∧ ¬ (∀ x : ℝ, (x - 1).abs < 4 → x * (x - 5) < 0) := 
by
  sorry

end suff_but_not_necc_l409_409373


namespace smallest_digit_not_in_units_place_of_odd_l409_409257

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l409_409257


namespace cos_90_equals_0_l409_409935

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l409_409935


namespace smallest_digit_not_found_in_units_place_of_odd_number_l409_409144

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l409_409144


namespace imaginary_part_z_is_zero_l409_409709

def z : ℂ := (1 + complex.I) / (1 - complex.I) + (1 - complex.I)

theorem imaginary_part_z_is_zero : complex.im z = 0 := 
by 
  sorry

end imaginary_part_z_is_zero_l409_409709


namespace period_and_monotonic_interval_range_of_f_l409_409535

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 3) * Real.cos (2 * x) + Real.sin (x + Real.pi / 4) ^ 2

theorem period_and_monotonic_interval :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ 
  (∃ k : ℤ, ∀ x, x ∈ Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12) →
    MonotoneOn f (Set.Icc (2 * k * Real.pi - Real.pi / 2) (2 * k * Real.pi + Real.pi / 2))) :=
sorry

theorem range_of_f (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 12) (5 * Real.pi / 12)) :
  f x ∈ Set.Icc 0 (3 / 2) :=
sorry

end period_and_monotonic_interval_range_of_f_l409_409535


namespace smallest_unfound_digit_in_odd_units_l409_409227

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l409_409227


namespace cube_edges_ratio_l409_409580

theorem cube_edges_ratio (V1 V2 V3 V4 : ℝ)
    (h1 : V1 / V2 = 216 / 64) 
    (h2 : V2 / V3 = 64 / 27) 
    (h3 : V3 / V4 = 27 / 1) : 
    let s1 := real.cbrt V1 
    let s2 := real.cbrt V2 
    let s3 := real.cbrt V3 
    let s4 := real.cbrt V4
    in s1 / s2 = 6 / 4 ∧ s2 / s3 = 4 / 3 ∧ s3 / s4 = 3 / 1 :=
begin
  sorry
end

end cube_edges_ratio_l409_409580


namespace smallest_digit_not_found_in_units_place_of_odd_number_l409_409133

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l409_409133


namespace perfect_square_proof_l409_409618

theorem perfect_square_proof (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := 
sorry

end perfect_square_proof_l409_409618


namespace total_stamps_is_38_l409_409817

-- Definitions based directly on conditions
def snowflake_stamps := 11
def truck_stamps := snowflake_stamps + 9
def rose_stamps := truck_stamps - 13
def total_stamps := snowflake_stamps + truck_stamps + rose_stamps

-- Statement to be proved
theorem total_stamps_is_38 : total_stamps = 38 := 
by 
  sorry

end total_stamps_is_38_l409_409817


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409125

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409125


namespace cos_90_eq_zero_l409_409943

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l409_409943


namespace product_of_digits_in_base8_representation_of_6543_l409_409116

theorem product_of_digits_in_base8_representation_of_6543 : 
  (let n := 6543 in 
  let base8_rep := [1, 4, 6, 1, 7] in
  base8_rep.foldl (λ acc x, acc * x) 1 = 168) := 
by
  sorry

end product_of_digits_in_base8_representation_of_6543_l409_409116


namespace menelaus_theorem_l409_409691

variables {A B C P : Point}
variables {A1 B1 C1 : LineSegment}

-- Assuming that P lies on the arc BC of the circumcircle of triangle ABC
def lies_on_circumcircle_arc (A B C P : Point) : Prop := sorry

-- Ratios as described in the problem based on Menelaus's theorem
def menelaus_ratios (A B C P : Point) (A1 B1 C1 : LineSegment) : Prop :=
  (A1.length / A1.length) * (B1.length / B1.length) * (C1.length / C1.length) = 1
  

theorem menelaus_theorem (h1: lies_on_circumcircle_arc A B C P) :
  menelaus_ratios A B C P A1 B1 C1 := 
sorry

end menelaus_theorem_l409_409691


namespace second_store_earns_at_least_72000_more_l409_409396

-- Conditions as definitions in Lean.
def discount_price := 900000 -- 10% discount on 1 million yuan.
def full_price := 1000000 -- Full price for 1 million yuan without discount.

-- Prize calculation for the second department store.
def prize_first := 1000 * 5
def prize_second := 500 * 10
def prize_third := 200 * 20
def prize_fourth := 100 * 40
def prize_fifth := 10 * 1000

def total_prizes := prize_first + prize_second + prize_third + prize_fourth + prize_fifth

def second_store_net_income := full_price - total_prizes -- Net income after subtracting prizes.

-- The proof problem statement.
theorem second_store_earns_at_least_72000_more :
  second_store_net_income - discount_price >= 72000 := sorry

end second_store_earns_at_least_72000_more_l409_409396


namespace coin_flip_sequences_count_l409_409774

theorem coin_flip_sequences_count : (2 ^ 16) = 65536 :=
by
  sorry

end coin_flip_sequences_count_l409_409774


namespace production_problem_l409_409484

theorem production_problem
  (n : ℕ)
  (average_past : n * 50)
  (average_new : (n * 50 + 105) / (n + 1) = 55) :
  n = 10 :=
by
  sorry

end production_problem_l409_409484


namespace cos_pi_half_eq_zero_l409_409858

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l409_409858


namespace new_time_when_traveling_at_new_speed_l409_409719

noncomputable def original_speed : ℝ := 70
noncomputable def new_speed : ℝ := 80
noncomputable def original_time : ℝ := 9 / 2

theorem new_time_when_traveling_at_new_speed :
  let distance := original_speed * original_time,
      new_time := distance / new_speed in
  Real.round (new_time * 100) / 100 = 3.94 :=
by
  -- Replace with the appropriate steps to prove the statement
  sorry

end new_time_when_traveling_at_new_speed_l409_409719


namespace log7_18_l409_409559

theorem log7_18 (a b : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 3 / Real.log 10 = b) : 
  Real.log 18 / Real.log 7 = (a + 2 * b) / (1 - a) :=
by
  -- proof to be completed
  sorry

end log7_18_l409_409559


namespace cos_90_equals_0_l409_409939

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l409_409939


namespace smallest_missing_digit_l409_409321

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l409_409321


namespace smallest_digit_not_in_odd_units_l409_409212

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l409_409212


namespace larger_number_of_hcf_lcm_is_322_l409_409082

theorem larger_number_of_hcf_lcm_is_322
  (A B : ℕ)
  (hcf: ℕ := 23)
  (factor1 : ℕ := 13)
  (factor2 : ℕ := 14)
  (hcf_condition : ∀ d, d ∣ A → d ∣ B → d ≤ hcf)
  (lcm_condition : ∀ m n, m * n = A * B → m = factor1 * hcf ∨ m = factor2 * hcf) :
  max A B = 322 :=
by sorry

end larger_number_of_hcf_lcm_is_322_l409_409082


namespace number_of_changing_quantities_l409_409668

-- Define the problem in geometric terms
noncomputable def midpoint {P A : Point} (PA : LineSegment P A) := Point
noncomputable def paralle {P A B : Point} (lineAB : LineSegment A B) := Prop
noncomputable def lengthOfSegment {MN : LineSegment M N} := ℝ
noncomputable def perimeter {P A B : Point} (trianglePAB : Triangle P A B) := ℝ
noncomputable def area {P A B : Point} (trianglePAB : Triangle P A B) := ℝ 
noncomputable def trapezoidArea {AB MN : LineSegment} := ℝ

-- Initial conditions
variables (P A B M N : Point)
variable (PA : LineSegment P A)
variable (PB : LineSegment P B)
variable (AB : LineSegment A B)
variable (MN : LineSegment M N)
variable (lineAB : LineSegment A B)
hypothesis (M_midpoint : midpoint PA M)
hypothesis (N_midpoint : midpoint PB N)
hypothesis (P_moves_parallel_AB : paralle lineAB)

-- Problem Statement
theorem number_of_changing_quantities (P A B M N : Point) (PA : LineSegment P A) (PB : LineSegment P B) (AB : LineSegment A B)
  (M_midpoint : midpoint PA M) (N_midpoint : midpoint PB N) (P_moves_parallel_AB : paralle lineAB) :
  let lengths_MN_remain_constant := (lengthOfSegment MN = lengthOfSegment MN),
      perimeter_changes := (perimeter (Triangle P A B) ≠ perimeter (Triangle P A B)),
      area_triangle_PAB_remains_constant := (area (Triangle P A B) = area (Triangle P A B)),
      area_trapezoid_ABNM_remains_constant := (trapezoidArea AB MN = trapezoidArea AB MN) 
  in 1 := sorry

end number_of_changing_quantities_l409_409668


namespace simplify_expression_l409_409688

theorem simplify_expression :
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 2))) = (Real.sqrt 3 - 2 * Real.sqrt 5 - 3) :=
by
  sorry

end simplify_expression_l409_409688


namespace cos_90_eq_zero_l409_409954

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l409_409954


namespace smallest_not_odd_unit_is_zero_l409_409188

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l409_409188


namespace compare_fractions_l409_409841

theorem compare_fractions : (-1 / 3) > (-1 / 2) := sorry

end compare_fractions_l409_409841


namespace polynomial_coeff_sum_l409_409488

theorem polynomial_coeff_sum :
  let (f : ℤ[X]) := (X + 1)^2 * (X + 2)^2016 in
  let a := f.coeff in
  (∑ i in finset.range 2018, a (i + 1) / 2^(i + 1)) = (1 / 2)^(2018) :=
sorry

end polynomial_coeff_sum_l409_409488


namespace cos_90_eq_0_l409_409896

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l409_409896


namespace cos_90_eq_zero_l409_409852

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l409_409852


namespace conversion_1_conversion_2_conversion_3_l409_409763

theorem conversion_1 : 2 * 1000 = 2000 := sorry

theorem conversion_2 : 9000 / 1000 = 9 := sorry

theorem conversion_3 : 8 * 1000 = 8000 := sorry

end conversion_1_conversion_2_conversion_3_l409_409763


namespace minimum_n_probability_geq_15_over_16_l409_409734

theorem minimum_n_probability_geq_15_over_16 :
  ∃ (n : ℕ), (1 - (1 / 2) ^ n ≥ 15 / 16) ∧ (∀ (m : ℕ), 1 - (1 / 2) ^ m ≥ 15 / 16 → m ≥ n) :=
begin
  refine ⟨4, _, _⟩,
  sorry,
  sorry,
end

end minimum_n_probability_geq_15_over_16_l409_409734


namespace price_second_jar_l409_409399

-- Define the initial conditions and constants
def diameter_first_jar : ℝ := 4
def height_first_jar : ℝ := 5
def price_first_jar : ℝ := 0.75

def diameter_second_jar : ℝ := 8
def height_second_jar : ℝ := 10

-- Define the function to calculate volume of a cylindrical jar
def volume_cylinder (diameter height : ℝ) : ℝ :=
  let radius := diameter / 2 in
  π * radius^2 * height

-- Calculate the volumes based on the given data
def volume_first_jar : ℝ :=
  volume_cylinder diameter_first_jar height_first_jar

def volume_second_jar : ℝ :=
  volume_cylinder diameter_second_jar height_second_jar

-- Statement about the relationship between the prices of the jars
theorem price_second_jar : volume_second_jar = 8 * volume_first_jar → 8 * price_first_jar = 6.0 :=
by
  intros h
  sorry

end price_second_jar_l409_409399


namespace num_passengers_on_second_plane_l409_409101

theorem num_passengers_on_second_plane :
  ∃ x : ℕ, 600 - (2 * 50) + 600 - (2 * x) + 600 - (2 * 40) = 1500 →
  x = 60 :=
by
  sorry

end num_passengers_on_second_plane_l409_409101


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409119

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409119


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409342

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409342


namespace tank_fill_time_l409_409749

noncomputable def fill_time (volume : ℕ) (initial_fill : ℕ) (filling_rate : ℝ) (drain1_rate : ℝ) (drain2_rate : ℝ) : ℝ :=
  let net_flow_rate := filling_rate - (drain1_rate + drain2_rate)
  in (volume - initial_fill) / net_flow_rate

theorem tank_fill_time :
  fill_time 6000 3000 (1 / 2) (1 / 4) (1 / 6) = 36 :=
by
  sorry

end tank_fill_time_l409_409749


namespace minimum_distance_X_find_pairs_when_c_12_exists_integer_n_ge_3_divides_a_b_l409_409639

open Real

variables (A O C B P : Point) (a b c p X : ℝ)

-- Assuming the points are correctly defined
def Point := (ℝ × ℝ)

def A : Point := (0, a)
def O : Point := (0, 0)
def C : Point := (c, 0)
def B : Point := (c, b)
def P : Point := (p, 0)

-- Distance function between two points
def distance (P Q : Point) : ℝ := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

-- Given conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom P_seg_OC : P.1 >= 0 ∧ P.1 <= c ∧ P.2 = 0 -- P on line segment OC

-- Proof part (a)
theorem minimum_distance_X :
  distance A P + distance P B = Real.sqrt (c^2 + (a + b)^2) := sorry

-- Proof part (b)
theorem find_pairs_when_c_12 :
  (c = 12) →
  (a, b ∈ set_of (λ z, z ∈ ℕ)) →
  ∃ (a b : ℕ), ((distance A P + distance P B) ∈ ℕ) → 
  (distance A P + distance P B = a + b) := sorry

-- Proof part (c)
theorem exists_integer_n_ge_3_divides_a_b :
  ∃ (n : ℕ), (n ≥ 3) ∧ (a % n = 0) ∧ (b % n = 0) := sorry

end minimum_distance_X_find_pairs_when_c_12_exists_integer_n_ge_3_divides_a_b_l409_409639


namespace cotton_needed_l409_409621

noncomputable def feet_of_cotton_per_teeshirt := 4
noncomputable def number_of_teeshirts := 15

theorem cotton_needed : feet_of_cotton_per_teeshirt * number_of_teeshirts = 60 := 
by 
  sorry

end cotton_needed_l409_409621


namespace price_after_discounts_l409_409681

theorem price_after_discounts (full_price : ℝ) (price_after_first_discount : ℝ) (price_after_second_discount : ℝ) : 
  full_price = 85 → 
  price_after_first_discount = full_price * (1 - 0.20) → 
  price_after_second_discount = price_after_first_discount * (1 - 0.25) → 
  price_after_second_discount = 51 :=
by
  intro h1 h2 h3
  rw h1 at h2
  rw h2 at h3
  rw h3
  sorry

end price_after_discounts_l409_409681


namespace smallest_digit_not_in_odd_units_l409_409247

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l409_409247


namespace length_of_segments_outside_circle_l409_409772

theorem length_of_segments_outside_circle {d : ℝ} (h_unit_square : d = real.sqrt (1^2 + 1^2)) (h_circle : ∃ d_circle : ℝ, d_circle = 1) :
  d - 1 = real.sqrt 2 - 1 :=
by
  rcases h_circle with ⟨d_circle, hd_circle⟩
  rw [hd_circle] at *
  sorry -- proof steps would go here

end length_of_segments_outside_circle_l409_409772


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409328

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409328


namespace limit_eq_one_third_deriv_at_one_l409_409020

variable (f : ℝ → ℝ) [Differentiable ℝ f]

theorem limit_eq_one_third_deriv_at_one :
  tendsto (λ Δx => (f (1 + Δx) - f 1) / (3 * Δx)) (𝓝 0) (𝓝 (1/3 * deriv f 1)) := 
sorry

end limit_eq_one_third_deriv_at_one_l409_409020


namespace cos_ninety_degrees_l409_409924

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l409_409924


namespace cos_pi_half_eq_zero_l409_409868

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l409_409868


namespace sum_of_squares_of_roots_l409_409504

-- Define the roots of the quadratic equation
def roots (a b c : ℝ) := { x : ℝ | a * x^2 + b * x + c = 0 }

-- The given quadratic equation is x^2 - 3x - 1 = 0
lemma quadratic_roots_property :
  ∀ x ∈ roots 1 (-3) (-1), x^2 - 3 * x - 1 = 0 :=
by {
  intros x hx,
  unfold roots at hx,
  exact hx,
  sorry
}

-- Using Vieta's formulas and properties of quadratic equations
theorem sum_of_squares_of_roots :
  let x1 := Classical.choose (exists (λ x, roots 1 (-3) (-1) x)),
      x2 := Classical.choose (exists ! (λ x, roots 1 (-3) (-1) x)),
  in x1^2 + x2^2 = 11 :=
by {
  let x1 := 3 / 2 + sqrt 13 / 2,
  let x2 := 3 / 2 - sqrt 13 / 2,
  have h1 : x1 + x2 = 3 := by {
    rw [← add_sub_assoc, add_sub_cancel, div_add_div_same],
    norm_num,
    sorry
  },
  have h2 : x1 * x2 = -1 := by {
    -- Similar proof under Classical logic, left as sorry for brevity
    sorry
  },
  calc
    x1^2 + x2^2 = (x1 + x2)^2 - 2 * (x1 * x2) : by norm_num; field_simp
            ... = 3^2 - 2 * (-1) : by rw [h1, h2]
            ... = 9 + 2 : by norm_num
            ... = 11 : by norm_num
  sorry
}

end sum_of_squares_of_roots_l409_409504


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409118

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409118


namespace smallest_digit_not_in_units_place_of_odd_l409_409271

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l409_409271


namespace smallest_digit_not_in_odd_units_l409_409244

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l409_409244


namespace part1_part2_part3_l409_409494

variables (m : ℝ)
def z := (m * (m - 2)) / (m - 1) + (m^2 + 2 * m - 3) * complex.I

-- Part 1: Prove that for m = -3, z ∈ ℝ
theorem part1 : z (-3) ∈ ℝ := sorry

-- Part 2: Prove that for m = 2 or m = 0, z is a pure imaginary number
theorem part2 {m : ℝ} (hm : m = 2 ∨ m = 0) : (z m).re = 0 := sorry

-- Part 3: Prove that for m ∈ (1, 2) ∪ (-∞, -3), the point corresponding to z lies in the second quadrant
theorem part3 {m : ℝ} (hm : (1 < m ∧ m < 2) ∨ m < -3) : z m).re < 0 ∧ (z m).im > 0 := sorry

end part1_part2_part3_l409_409494


namespace dice_probability_of_divisible_by_three_l409_409408

theorem dice_probability_of_divisible_by_three :
  let probability_divisible := (1 : ℚ) / 3,
      probability_not_divisible := (2 : ℚ) / 3 in
  (Nat.choose 5 3) * (probability_divisible^3) * (probability_not_divisible^2) = (40 : ℚ) / 243 :=
by
  let probability_divisible := (1 : ℚ) / 3
  let probability_not_divisible := (2 : ℚ) / 3
  sorry

end dice_probability_of_divisible_by_three_l409_409408


namespace smallest_n_l409_409345

theorem smallest_n (n : ℕ) (hn1 : (5 * n) pow 2) (hn2 : (4 * n) pow 3) : n = 80 :=
begin
  -- sorry statement to skip the proof.
  sorry
end

end smallest_n_l409_409345


namespace cos_90_eq_0_l409_409882

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l409_409882


namespace imaginary_part_z_is_zero_l409_409707

def complex_num := ℂ

def z : complex_num := (1 + complex.I) / (1 - complex.I) + (1 - complex.I)

theorem imaginary_part_z_is_zero : z.im = 0 := 
by
  sorry

end imaginary_part_z_is_zero_l409_409707


namespace probability_of_three_blue_marbles_in_six_trials_l409_409068

noncomputable def binomial_coefficient (n k : ℕ) : ℚ := nat.choose n k

noncomputable def probability_of_blue_marble : ℚ := 8 / 15
noncomputable def probability_of_red_marble : ℚ := 7 / 15

noncomputable def probability_exactly_three_blue_in_six : ℚ :=
  let k := 3
  let n := 6
  (binomial_coefficient n k) * (probability_of_blue_marble ^ k) * (probability_of_red_marble ^ (n - k))

theorem probability_of_three_blue_marbles_in_six_trials : 
  probability_exactly_three_blue_in_six = 704464 / 2278125 :=
by
  sorry

end probability_of_three_blue_marbles_in_six_trials_l409_409068


namespace marble_probability_l409_409388

theorem marble_probability :
  let total_marbles := 13
  let blue_marbles := 4
  let red_marbles := 3
  let white_marbles := 6
  let first_blue_prob := (blue_marbles : ℚ) / total_marbles
  let second_red_prob := (red_marbles : ℚ) / (total_marbles - 1)
  let third_white_prob := (white_marbles : ℚ) / (total_marbles - 2)
  first_blue_prob * second_red_prob * third_white_prob = 6 / 143 :=
by
  sorry

end marble_probability_l409_409388


namespace sufficient_but_not_necessary_l409_409562

theorem sufficient_but_not_necessary (a : ℝ) : a = 1 → |a| = 1 ∧ (|a| = 1 → a = 1 → false) :=
by
  sorry

end sufficient_but_not_necessary_l409_409562


namespace smallest_n_perfect_square_and_cube_l409_409353

theorem smallest_n_perfect_square_and_cube (n : ℕ) (h1 : ∃ k : ℕ, 5 * n = k^2) (h2 : ∃ m : ℕ, 4 * n = m^3) :
  n = 1080 :=
  sorry

end smallest_n_perfect_square_and_cube_l409_409353


namespace smallest_missing_digit_l409_409326

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l409_409326


namespace solution_set_for_inequality_inequality_for_elements_in_M_l409_409536

noncomputable def f (x : ℝ) : ℝ := |x + 1|

def M : set ℝ := {x | x < -1 ∨ x > 1}

theorem solution_set_for_inequality : 
  {x | f(x) + 1 < f(2 * x)} = M :=
by 
  sorry

theorem inequality_for_elements_in_M (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 
  f(a * b) > f(a) - f(-b) :=
by 
  sorry

end solution_set_for_inequality_inequality_for_elements_in_M_l409_409536


namespace sum_of_numbers_less_than_two_l409_409725

def a : ℝ := 0.8
def b : ℝ := 1 / 2
def c : ℝ := 0.5

theorem sum_of_numbers_less_than_two : a + b + c = 1.8 := by
  sorry

end sum_of_numbers_less_than_two_l409_409725


namespace find_a_values_l409_409544

noncomputable def a_values : set ℝ := 
  {a : ℝ | let A := {x : ℝ | a * x - 1 = 0},
                B := {1, 2} in A ∪ B = B }

theorem find_a_values : a_values = {0, 1/2, 1} :=
sorry

end find_a_values_l409_409544


namespace num_false_statements_is_three_l409_409390

-- Definitions of the statements on the card
def s1 : Prop := ∀ (false_statements : ℕ), false_statements = 1
def s2 : Prop := ∀ (false_statements_card1 false_statements_card2 : ℕ), false_statements_card1 + false_statements_card2 = 2
def s3 : Prop := ∀ (false_statements : ℕ), false_statements = 3
def s4 : Prop := ∀ (false_statements_card1 false_statements_card2 : ℕ), false_statements_card1 = false_statements_card2

-- Main proof problem: The number of false statements on this card is 3
theorem num_false_statements_is_three 
  (h_s1 : ¬ s1)
  (h_s2 : ¬ s2)
  (h_s3 : s3)
  (h_s4 : ¬ s4) :
  ∃ (n : ℕ), n = 3 :=
by
  sorry

end num_false_statements_is_three_l409_409390


namespace roger_candies_left_l409_409678

theorem roger_candies_left (initial_candies : ℕ) (to_stephanie : ℕ) (to_john : ℕ) (to_emily : ℕ) : 
  initial_candies = 350 ∧ to_stephanie = 45 ∧ to_john = 25 ∧ to_emily = 18 → 
  initial_candies - (to_stephanie + to_john + to_emily) = 262 :=
by
  sorry

end roger_candies_left_l409_409678


namespace angle_QRP_90_and_QR_eq_RP_l409_409371

variables {A B C P Q R : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace P] [MetricSpace Q] [MetricSpace R]

-- Given a triangle ABC
variables (triangle_ABC : Triangle A B C)

-- Exterior triangles BPC, CQA, ARB
variables (triangle_BPC : Triangle B P C) (triangle_CQA : Triangle C Q A) (triangle_ARB : Triangle A R B)

-- Conditions of angles
axiom angle_PBC_45 : angle P B C = 45
axiom angle_CAQ_45 : angle C A Q = 45
axiom angle_BCP_30 : angle B C P = 30
axiom angle_QCA_30 : angle Q C A = 30
axiom angle_ABR_15 : angle A B R = 15
axiom angle_BAR_15 : angle B A R = 15

-- Prove that ∠QRP = 90° and QR = RP
theorem angle_QRP_90_and_QR_eq_RP : (angle Q R P = 90) ∧ (dist Q R = dist R P) := by sorry

end angle_QRP_90_and_QR_eq_RP_l409_409371


namespace cos_90_deg_eq_zero_l409_409912

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l409_409912


namespace greatest_sum_of_digits_base8_l409_409740

theorem greatest_sum_of_digits_base8 (n : ℕ) (h1 : n > 0) (h2 : n < 1800) : 
  ∃ m, (m < 1800) ∧ (∃ s, s = Nat.digits 8 m ∧ s.sum = 23) :=
sorry

end greatest_sum_of_digits_base8_l409_409740


namespace smallest_digit_not_in_units_place_of_odd_l409_409268

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l409_409268


namespace cos_B_proof_l409_409587

-- Definitions and conditions
variables (A B C a b c : ℝ)
def triangle_ABC := ∀ (a b c : ℝ), a + b + c = 180 -- Sum of angles in triangle
def arithmetic_sequence := 2 * b = a + c           -- Arithmetic sequence condition
def angles_relation := A - C = 90                  -- Given angle relation

-- Lean statement for the given proof problem
theorem cos_B_proof (h1 : triangle_ABC A B C) (h2 : arithmetic_sequence a b c) (h3 : angles_relation A C) :
  cos B = 3/4 :=
by
  sorry

end cos_B_proof_l409_409587


namespace log_eq_exponent_eq_l409_409465

theorem log_eq_exponent_eq (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
by sorry

end log_eq_exponent_eq_l409_409465


namespace find_triangle_angles_l409_409606

theorem find_triangle_angles 
  (α β γ : ℝ)
  (a b : ℝ)
  (h1 : γ = 2 * α)
  (h2 : b = 2 * a)
  (h3 : α + β + γ = 180) :
  α = 30 ∧ β = 90 ∧ γ = 60 := 
by 
  sorry

end find_triangle_angles_l409_409606


namespace find_coordinates_l409_409603

-- Conditions given in the problem
def pointA (x y z : ℝ) := (x^2 + 4, 4 - y, 1 + 2z)
def pointB (x z : ℝ) := (-4 * x, 9, 7 - z)

-- Statement of the theorem to prove values of x, y, z given conditions
theorem find_coordinates (x y z : ℝ) :
  pointA x y z = (x^2 + 4, 4 - y, 1 + 2z) ∧
  pointB x z = (-4 * x, 9, 7 - z) → 
  x = 2 ∧ y = -5 ∧ z = 2 :=
by
  sorry

end find_coordinates_l409_409603


namespace cos_90_deg_eq_zero_l409_409910

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l409_409910


namespace part1_part2_l409_409533

noncomputable def f (x b : ℝ) : ℝ := (x + b) * Real.log x

noncomputable def g (x b a : ℝ) : ℝ := Real.exp x * ((f x b) / (x + 2) - 2 * a)

example (b : ℝ) : has_deriv_at (λ x, f x b) (3 : ℝ) 1 := by {
  have h : (∀ b, HasDerivAt (λ x, (x + b) * Real.log x) (Real.log 1 + b / 1 + 1) 1),
  sorry,
  rw Real.log_one at h,
  exact h b,
}

theorem part1 : (∃ b : ℝ, HasDerivAt (λ x, f x b) 3 1) → b = 2 := by {
  intro h,
  cases' h with b hb,
  have : 3 = Real.log 1 + b + 1 := Sorry,
  rw Real.log_one at this,
  linarith,
}

theorem part2 (a : ℝ) : (∀ x : ℝ, x > 0 → g x 2 a ≥ g x 2 a) → a ≤ 1 / 2 := by {
  have h : ∀ x > 0, (Real.log x + 1 / x - 2 * a) * Real.exp x ≥ 0,
  sorry,
  have g_is_increasing : ∀ x > 0, Real.log x + 1 / x ≥ 2 * a,
  sorry,
  have h_min : (∃ a, ∀ x > 0, Real.log x + 1 / x ≥ a) := sorry,
  cases' h_min with m hm,
  have : m = 1,
  sorry,
  have : 2 * a ≤ 1,
  exact (hm 1 (by norm_num)).2,
  linarith,
}

#check part1
#check part2

end part1_part2_l409_409533


namespace range_of_m_l409_409035

-- Defining the point P and the required conditions for it to lie in the fourth quadrant
def point_in_fourth_quadrant (m : ℝ) : Prop :=
  let P := (m + 3, m - 1)
  P.1 > 0 ∧ P.2 < 0

-- Defining the range of m for which the point lies in the fourth quadrant
theorem range_of_m (m : ℝ) : point_in_fourth_quadrant m ↔ (-3 < m ∧ m < 1) :=
by
  sorry

end range_of_m_l409_409035


namespace units_digit_of_product_of_seven_consecutive_l409_409096

theorem units_digit_of_product_of_seven_consecutive (n : ℕ) : 
  ∃ d ∈ [n, n+1, n+2, n+3, n+4, n+5, n+6], d % 10 = 0 :=
by
  sorry

end units_digit_of_product_of_seven_consecutive_l409_409096


namespace sum_of_sequence_l409_409094

theorem sum_of_sequence (n : ℕ) (n_pos : 0 < n) :
  (Finset.sum (Finset.range n.succ) (λ k, 1 / (4 * (k + 1)^2 - 1))) = (n / (2 * n + 1) : ℚ) :=
by
  sorry

end sum_of_sequence_l409_409094


namespace compare_neg_rational_l409_409837

def neg_one_third : ℚ := -1 / 3
def neg_one_half : ℚ := -1 / 2

theorem compare_neg_rational : neg_one_third > neg_one_half :=
by sorry

end compare_neg_rational_l409_409837


namespace sum_diff_eq_2003_l409_409114

section
variable (n : ℕ) (E O : ℕ → ℕ)
variable (S : ℕ → ℕ × ℕ → ℕ)

def first_n_odds (n : ℕ) : ℕ → ℕ
| 0 => 1
| k+1 => 2*k + 1

def first_n_evens (n : ℕ) : ℕ → ℕ
| 0 => 2
| k+1 => 2*(k+1)

noncomputable def sum_of_sequence (n sum_func: ℕ) : ℕ :=
(n / 2) * (sum_func(0) + sum_func(n))

theorem sum_diff_eq_2003 : (n : ℕ) → sum_of_sequence n first_n_evens - sum_of_sequence n first_n_odds = 2003 :=
by
  intro n
  have O : ∀ i, i < n → first_n_odds i = 2*i + 1 := sorry
  have E : ∀ i, i < n → first_n_evens i = 2*(i+1) := sorry
  calc
    sum_of_sequence n first_n_evens - sum_of_sequence n first_n_odds
    = (n / 2) * (2 + 2*n) - (n / 2) * (1 + 2*n - 1) : by simp [sum_of_sequence, first_n_evens, first_n_odds, O, E, sorry]
    ... = 2003 : sorry
end

end sum_diff_eq_2003_l409_409114


namespace smallest_n_l409_409348

theorem smallest_n (n : ℕ) (h₁ : ∃ k₁ : ℕ, 5 * n = k₁ ^ 2) (h₂ : ∃ k₂ : ℕ, 4 * n = k₂ ^ 3) : n = 1600 :=
sorry

end smallest_n_l409_409348


namespace smallest_digit_not_in_units_place_of_odd_l409_409277

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l409_409277


namespace cos_ninety_degrees_l409_409915

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l409_409915


namespace find_sum_of_distinct_reals_l409_409017

theorem find_sum_of_distinct_reals (x y : ℝ) (h_distinct : x ≠ y)
  (h_det : ↑(Matrix.det (Matrix.of (λ (i j : Fin 3), (if i = 0 then if j = 0 then 1 else if j = 1 then 5 else 7
           else if i = 1 then if j = 0 then 2 else if j = 1 then x else y
           else if j = 0 then 2 else if j = 1 then y else x)))) = 0) :
  x + y = 24 := by
  sorry

end find_sum_of_distinct_reals_l409_409017


namespace cos_90_eq_0_l409_409892

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l409_409892


namespace probability_of_draw_multiple_of_3_or_7_l409_409089

def is_multiple_of (n m : ℕ) : Prop := m % n = 0

def num_cards := 40

def multiples_of_3 := (finset.range (num_cards + 1)).filter (λ n, is_multiple_of 3 n)
def multiples_of_7 := (finset.range (num_cards + 1)).filter (λ n, is_multiple_of 7 n)
def multiples_of_21 := (finset.range (num_cards + 1)).filter (λ n, is_multiple_of 21 n)

def total_successful_outcomes := multiples_of_3.card + multiples_of_7.card - multiples_of_21.card

def probability_of_multiple_of_3_or_7 : ℚ := total_successful_outcomes / num_cards

theorem probability_of_draw_multiple_of_3_or_7 : probability_of_multiple_of_3_or_7 = 17 / 40 := by
  sorry

end probability_of_draw_multiple_of_3_or_7_l409_409089


namespace a_divides_b_l409_409646

theorem a_divides_b (a b : ℕ) (h_pos : 0 < a ∧ 0 < b)
    (h : ∀ n : ℕ, a^n ∣ b^(n+1)) : a ∣ b :=
by
  sorry

end a_divides_b_l409_409646


namespace cos_90_eq_zero_l409_409953

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l409_409953


namespace find_g8_l409_409397

variable (g : ℝ → ℝ)

theorem find_g8 (h1 : ∀ x y : ℝ, g (x + y) = g x + g y) (h2 : g 7 = 8) : g 8 = 64 / 7 :=
sorry

end find_g8_l409_409397


namespace prob_grade_B_most_likely_not_exported_mean_profit_l409_409596

-- Problem 1: Probability of grade B
theorem prob_grade_B : 
  let p_unsat := 1 / 3
      p_sat := 2 / 3 in 
  let P_B := (3 * p_unsat * p_sat^2 * p_sat^2) in
  P_B = 16 / 81 :=
sorry

-- Problem 2: Most likely number of handicrafts (out of 10) that cannot be exported
theorem most_likely_not_exported :
  let p_unsat_two_or_more := 7 / 27 in
  let n_D (k : ℕ) := binomial 10 k * (p_unsat_two_or_more^k) * ((20 / 27)^(10 - k)) in
  n_D 2 = maxList (map n_D (range 11)) :=
sorry

-- Problem 3: Mean profit of one handicraft
theorem mean_profit :
  let p_A := 8 / 27
      p_B := 16 / 81
      p_C := 20 / 81
      p_D := 7 / 27 in
  let E_X := 900 * p_A + 600 * p_B + 300 * p_C + 100 * p_D in
  E_X = 13100 / 27 :=
sorry

end prob_grade_B_most_likely_not_exported_mean_profit_l409_409596


namespace cos_pi_half_eq_zero_l409_409871

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l409_409871


namespace smallest_digit_not_in_units_place_of_odd_numbers_l409_409155

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l409_409155


namespace AB_plus_AC_l409_409833

-- Let's define the assumptions in Lean
variables
  (ω : Type) [MetricSpace ω] [Circle ω]
  (O A T1 T2 T3 B C : ω)
  (r : ℝ) (OA BC : ℝ)

-- Conditions
axiom radius_of_omega : ∀ (ω : Type) [MetricSpace ω] [Circle ω], radius ω = 7
axiom center_of_omega : ∀ (ω : Type) [Circle ω], Center ω = O
axiom OA_distance : OA = 15
axiom BC_tangent : BC = 10
axiom tangent_points : Tangent A T1 ω ∧ Tangent A T2 ω

-- Theorem to prove
theorem AB_plus_AC : AB + AC = 8 * real.sqrt 11 - 10 :=
sorry

end AB_plus_AC_l409_409833


namespace units_digit_of_expression_l409_409742

noncomputable def units_digit (n : ℕ) : ℕ :=
  n % 10

def expr : ℕ := 2 * (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9)

theorem units_digit_of_expression : units_digit expr = 6 :=
by
  sorry

end units_digit_of_expression_l409_409742


namespace cos_90_eq_zero_l409_409971

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l409_409971


namespace log_eq_exp_l409_409474

theorem log_eq_exp {x : ℝ} (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end log_eq_exp_l409_409474


namespace possible_value_is_121_l409_409423

theorem possible_value_is_121
  (x a y z b : ℕ) 
  (hx : x = 1 / 6 * a) 
  (hz : z = 1 / 6 * b) 
  (hy : y = (a + b) % 5) 
  (h_single_digit : ∀ n, n ∈ [x, a, y, z, b] → n < 10 ∧ 0 < n) : 
  100 * x + 10 * y + z = 121 :=
by
  sorry

end possible_value_is_121_l409_409423


namespace integral_eval_l409_409443

theorem integral_eval : ∫ x in 2..4, (Real.exp x - 1 / x) = Real.exp 4 - Real.exp 2 - Real.log 2 :=
by
  sorry

end integral_eval_l409_409443


namespace triangle_area_l409_409092

theorem triangle_area :
  ∃ (cosθ : ℝ), (5 * cosθ^2 - 7 * cosθ - 6 = 0) →
  let sinθ := real.sqrt (1 - cosθ^2),
      a := 3,
      b := 5 in
  ((1 / 2) * a * b * sinθ = 6) := sorry

end triangle_area_l409_409092


namespace area_of_original_triangle_l409_409779

noncomputable def originalArea (S_intuitive : ℝ) : ℝ := 2 * real.sqrt 2 * S_intuitive

theorem area_of_original_triangle :
  let S_intuitive := 1 / 4
  let S_original := originalArea S_intuitive
  S_original = real.sqrt 2 / 2 :=
by
  let S_intuitive := 1 / 4
  let S_original := originalArea S_intuitive
  sorry

end area_of_original_triangle_l409_409779


namespace smallest_digit_never_in_units_place_of_odd_numbers_l409_409303

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l409_409303


namespace range_of_f_l409_409518

def f (x : ℕ) : ℤ := x^2 - 2*x

theorem range_of_f :
  Set.range f = {0, -1} := 
sorry

end range_of_f_l409_409518


namespace star_evaluation_l409_409641

def star (a b : ℤ) : ℤ := a * b + a + b

theorem star_evaluation : 
  1 ∗ (2 ∗ (3 ∗ (⋯ (99 ∗ 100) ⋯))) = 101! - 1 :=
by
  sorry

end star_evaluation_l409_409641


namespace cos_90_equals_0_l409_409928

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l409_409928


namespace cos_pi_half_eq_zero_l409_409867

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l409_409867


namespace count_valid_four_digit_numbers_l409_409556

theorem count_valid_four_digit_numbers : 
  let digits := [2, 2, 2, 9, 0] in
  let total_permutations := 5.factorial / 3.factorial in
  let invalid_start_positions := 4.factorial / 3.factorial in
  total_permutations - invalid_start_positions = 16 :=
by
  sorry

end count_valid_four_digit_numbers_l409_409556


namespace cos_90_equals_0_l409_409934

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l409_409934


namespace bisection_method_next_interval_l409_409737

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x - 5

theorem bisection_method_next_interval :
  let a := 2
  let b := 3
  let x0 := (a + b) / 2
  (f a * f x0 < 0) ∨ (f x0 * f b < 0) →
  (x0 = 2.5) →
  f 2 * f 2.5 < 0 :=
by
  intros
  sorry

end bisection_method_next_interval_l409_409737


namespace triangle_area_inequality_l409_409648

variables {a b c S x y z T : ℝ}

-- Definitions based on the given conditions
def side_lengths_of_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def area_of_triangle (a b c S : ℝ) : Prop :=
  16 * S * S = (a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c)

def new_side_lengths (a b c : ℝ) (x y z : ℝ) : Prop :=
  x = a + b / 2 ∧ y = b + c / 2 ∧ z = c + a / 2

def area_condition (S T : ℝ) : Prop :=
  T ≥ 9 / 4 * S

-- Main theorem statement
theorem triangle_area_inequality
  (h_triangle: side_lengths_of_triangle a b c)
  (h_area: area_of_triangle a b c S)
  (h_new_sides: new_side_lengths a b c x y z) :
  ∃ T : ℝ, side_lengths_of_triangle x y z ∧ area_condition S T :=
sorry

end triangle_area_inequality_l409_409648


namespace find_x_l409_409468

theorem find_x (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end find_x_l409_409468


namespace smallest_integer_x_l409_409455

theorem smallest_integer_x (x : ℤ) : (x^2 - 11 * x + 24 < 0) → x ≥ 4 ∧ x < 8 :=
by
sorry

end smallest_integer_x_l409_409455


namespace perfect_square_proof_l409_409616

theorem perfect_square_proof (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := 
sorry

end perfect_square_proof_l409_409616


namespace proof_prob_l409_409413

-- Definitions based on the conditions:
structure Point :=
  (x : ℝ)
  (y : ℝ)

def rectangle_ABCD :=
  let A := ⟨-4, -2⟩ in
  let B := ⟨4, -2⟩ in
  let C := ⟨4, 2⟩ in
  let D := ⟨-4, 2⟩ in
  (A, B, C, D)

def midpoints :=
  let (A, B, C, D) := rectangle_ABCD in
  let M := ⟨(A.x + B.x) / 2, A.y⟩ in
  let N := ⟨(C.x + D.x) / 2, C.y⟩ in
  (M, N)

noncomputable def curve_eq :=
  ∀ (P : Point), P ∈ E ↔ ∃ y : ℝ, P.x ^ 2 = -8 * y ∧ P.y = y ∧ -4 ≤ P.x ∧ P.x ≤ 4

def F :=
  let (M, N) := midpoints in
  ⟨0, -1⟩

-- Prove the range of λ
noncomputable def lambda_range :=
  ∀ (S T : Point) (λ : ℝ),
    (S.y = k * S.x - 1) ∧ (T.y = k * T.x - 1) ∧
    (S.x ^ 2 = -8 * S.y) ∧ (T.x ^ 2 = -8 * T.y) ∧
    (S.y, T.y) ≠ (T.y, S.y) ∧
    (S.y ≤ T.y) ∧ (S.x ≤ T.x) ∧
    (S.y = λ * (T.y)) ∧
    (1 / λ + λ = 8 * k ^ 2 + 2) ↔
    λ ∈ set.Icc (1 / 2) 2

theorem proof_prob :
  curve_eq ∧ lambda_range := sorry

end proof_prob_l409_409413


namespace chord_length_six_l409_409780

noncomputable def length_of_chord_parallel_line_to_circle
  (P : ℝ × ℝ) (A B : ℝ) (C : ℝ) (center : ℝ × ℝ) (radius_squared : ℝ) : ℝ :=
  let c := -(A * P.1 + B * P.2) in
  let d := abs (A * center.1 + B * center.2 + c) / real.sqrt (A^2 + B^2) in
  2 * real.sqrt (radius_squared - d^2)

theorem chord_length_six :
  length_of_chord_parallel_line_to_circle (1, 0) 1 (-real.sqrt 2) 3 (6, real.sqrt 2) 12 = 6 := 
sorry

end chord_length_six_l409_409780


namespace friend_gain_percentage_l409_409783

noncomputable def gain_percentage (original_cost_price sold_price_friend : ℝ) : ℝ :=
  ((sold_price_friend - (original_cost_price - 0.12 * original_cost_price)) / (original_cost_price - 0.12 * original_cost_price)) * 100

theorem friend_gain_percentage (original_cost_price sold_price_friend gain_pct : ℝ) 
  (H1 : original_cost_price = 51136.36) 
  (H2 : sold_price_friend = 54000) 
  (H3 : gain_pct = 20) : 
  gain_percentage original_cost_price sold_price_friend = gain_pct := 
by
  sorry

end friend_gain_percentage_l409_409783


namespace smallest_digit_never_at_units_place_of_odd_l409_409172

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l409_409172


namespace count_4_digit_numbers_with_conditions_l409_409409

def num_valid_numbers : Nat :=
  432

-- Statement declaring the proposition to be proved
theorem count_4_digit_numbers_with_conditions :
  (count_valid_numbers == 432) :=
sorry

end count_4_digit_numbers_with_conditions_l409_409409


namespace find_f_2015_l409_409519

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_2015
  (h1 : ∀ x, f (-x) = -f x) -- f is an odd function
  (h2 : ∀ x, f (x + 2) = -f x) -- f(x+2) = -f(x)
  (h3 : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) -- f(x) = 2x^2 for x in (0, 2)
  : f 2015 = -2 :=
sorry

end find_f_2015_l409_409519


namespace smallest_digit_not_in_units_place_of_odd_l409_409256

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l409_409256


namespace smallest_digit_never_in_units_place_of_odd_l409_409203

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l409_409203


namespace smallest_digit_never_in_units_place_of_odd_numbers_l409_409307

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l409_409307


namespace smallest_digit_never_at_units_place_of_odd_l409_409165

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l409_409165


namespace problem1_problem2_l409_409421

theorem problem1 : (Real.sqrt 2)^2 + |Real.sqrt 2 - 2| - (Real.pi - 1)^0 = 3 - Real.sqrt 2 :=
by
  sorry

theorem problem2 : Real.cbrt 27 + (Real.sqrt ((-2)^2)) - Real.sqrt (3^2) = 2 :=
by
  sorry

end problem1_problem2_l409_409421


namespace Pau_total_fried_chicken_l409_409627

theorem Pau_total_fried_chicken :
  ∀ (kobe_order : ℕ),
  (pau_initial : ℕ) (pau_second : ℕ),
  kobe_order = 5 →
  pau_initial = 2 * kobe_order →
  pau_second = pau_initial →
  pau_initial + pau_second = 20 :=
by
  intros kobe_order pau_initial pau_second
  sorry

end Pau_total_fried_chicken_l409_409627


namespace clinics_and_doctors_count_l409_409395

-- Definition of the conditions
def combination_representative (n : Nat) : Prop :=
  ∀ (c1 c2 : Nat), c1 ≠ c2 → ∃ (d : Nat), represents c1 d ∧ represents c2 d

def two_represented_clinics (d : Nat) : Prop :=
  ∃ (c1 c2 : Nat), represents c1 d ∧ represents c2 d

def clinic_invites_four_doctors (n d : Nat) : Prop :=
  ∀ (c : Nat), ∃ (d1 d2 d3 d4 : Nat), represents c d1 ∧ represents c d2 ∧ represents c d3 ∧ represents c d4

noncomputable def N (n : Nat) : Nat := n * (n - 1) / 2

theorem clinics_and_doctors_count :
  ∀ (n : Nat), combination_representative n →
    (∀ (d : Nat), two_represented_clinics d) →
    clinic_invites_four_doctors n → 
    (4 * n = 2 * (N n)) → 
    n = 5 ∧ (N n) = 10 :=
by
  sorry

end clinics_and_doctors_count_l409_409395


namespace smallest_not_odd_unit_is_zero_l409_409179

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l409_409179


namespace cos_90_eq_0_l409_409886

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l409_409886


namespace sheela_monthly_income_l409_409054

-- Definitions from the conditions
def deposited_amount : ℝ := 5000
def percentage_of_income : ℝ := 0.20

-- The theorem to be proven
theorem sheela_monthly_income : (deposited_amount / percentage_of_income) = 25000 := by
  sorry

end sheela_monthly_income_l409_409054


namespace min_value_reciprocal_l409_409524

noncomputable def X : Type := sorry -- Placeholder for the discrete random variable X
noncomputable def B (n : ℕ) (p : ℝ) : Type := sorry -- Placeholder for Binomial distribution

def E (X : Type) : ℝ := 4
def D (X : Type) : ℝ := sorry

theorem min_value_reciprocal (n : ℕ) (p q : ℝ) (X : Type) 
  (h1 : X ~ B(n, p)) (h2 : E(X) = 4) (h3 : D(X) = q) : 
  (1 / p) + (1 / q) ≥ 9 / 4 := 
by
  sorry

end min_value_reciprocal_l409_409524


namespace XZ_perpendicular_BC_l409_409102

variables {A B C X Y Z : Type} [euclidean_space X] 
variables (A B C P : X)

def equilateral_triangle (A B C : X) : Prop := dist A B = dist B C ∧ dist B C = dist C A

def points_on_line (Y : X) (AB : set X) : Prop := Y ∈ AB
def intersects (line1 line2 : set X) (point : X) : Prop := point ∈ line1 ∧ point ∈ line2

variables (Y : X)
variables (Y_on_AB : points_on_line Y (line_through A B))

variables (line_Y_intersects_BC_at_Z : intersects (line_through Y (line_through A B)) (line_through B C) Z)
variables (line_Y_intersects_CA_ext_at_X : intersects (line_through Y (line_through A B)) (line_through C A) X)

variables (XY_YZ_equal : dist X Y = dist Y Z)
variables (AY_BZ_equal : dist A Y = dist B Z)

theorem XZ_perpendicular_BC :
  equilateral_triangle A B C →
  points_on_line Y (line_through A B) →
  intersects (line_through Y (line_through A B)) (line_through B C) Z →
  intersects (line_through Y (line_through A B)) (line_through C A_ext) X →
  XY_YZ_equal →
  AY_BZ_equal →
  ∠ X Z B = 90 :=
begin
  sorry
end

end XZ_perpendicular_BC_l409_409102


namespace cos_90_deg_eq_zero_l409_409901

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l409_409901


namespace find_k_l409_409716

def line1 (x y : ℝ) : Prop := x + 3 * y - 7 = 0
def line2 (k x y : ℝ) : Prop := k * x + y - 2 = 0
def quadrilateral_has_circumscribed_circle (k : ℝ) : Prop :=
  ∀ x y : ℝ, line1 x y → line2 k x y →
  k = -3

theorem find_k (k : ℝ) (x y : ℝ) : 
  (line1 x y) ∧ (line2 k x y) → quadrilateral_has_circumscribed_circle k :=
by 
  sorry

end find_k_l409_409716


namespace smallest_digit_not_in_units_place_of_odd_l409_409272

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l409_409272


namespace cos_90_eq_zero_l409_409854

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l409_409854


namespace smallest_unfound_digit_in_odd_units_l409_409224

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l409_409224


namespace perfect_square_condition_l409_409615

theorem perfect_square_condition (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 ↔ 4a - 3b = 5 * k :=
by  
  sorry

end perfect_square_condition_l409_409615


namespace coefficient_x4_l409_409433

theorem coefficient_x4 
  (C : ℕ → ℕ → ℕ) 
  (binom_expansion : ∀ (x : ℝ), (x + 1 / (6 * x)) ^ 8 = ∑ i in finset.range 9, C 8 i * x ^ (8 - i) * (1 / (6 * x)) ^ i)
  (coeff : ℕ → ℝ)
  (term : ℝ → ℕ → ℝ) :
  coeff 4 = 7 :=
by
  let C := nat.choose
  have binom_expansion := λ x, by sorry
  have coeff := λ n, ∑ i in finset.range 9, if 8 - i - n = 0 then C 8 i * (1 / (6 * x)) ^ i else 0
  have term := λ x i, C 8 i * x ^ (8 - i) * (1 / (6 * x)) ^ i
  sorry

end coefficient_x4_l409_409433


namespace log_eq_exp_l409_409475

theorem log_eq_exp {x : ℝ} (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end log_eq_exp_l409_409475


namespace smallest_not_odd_unit_is_zero_l409_409191

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l409_409191


namespace smallest_n_l409_409343

theorem smallest_n (n : ℕ) (hn1 : (5 * n) pow 2) (hn2 : (4 * n) pow 3) : n = 80 :=
begin
  -- sorry statement to skip the proof.
  sorry
end

end smallest_n_l409_409343


namespace cos_90_equals_0_l409_409929

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l409_409929


namespace cos_90_eq_zero_l409_409948

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l409_409948


namespace log_eq_exponent_eq_l409_409466

theorem log_eq_exponent_eq (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
by sorry

end log_eq_exponent_eq_l409_409466


namespace smallest_digit_not_in_units_place_of_odd_numbers_l409_409161

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l409_409161


namespace cos_90_eq_zero_l409_409980

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l409_409980


namespace find_x_l409_409469

theorem find_x (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end find_x_l409_409469


namespace quadratic_expression_transformation_l409_409583

theorem quadratic_expression_transformation :
  ∀ (a h k : ℝ), (∀ x : ℝ, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 :=
by
  intros a h k h_eq
  sorry

end quadratic_expression_transformation_l409_409583


namespace difference_in_mpg_l409_409041

theorem difference_in_mpg 
  (advertisedMPG : ℕ) 
  (tankCapacity : ℕ) 
  (totalMilesDriven : ℕ) 
  (h_advertised : advertisedMPG = 35) 
  (h_tank : tankCapacity = 12) 
  (h_miles : totalMilesDriven = 372) : 
  advertisedMPG - (totalMilesDriven / tankCapacity) = 4 :=
by
  rw [h_advertised, h_tank, h_miles]
  norm_num
  sorry

end difference_in_mpg_l409_409041


namespace smallest_digit_not_in_units_place_of_odd_l409_409273

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l409_409273


namespace cos_90_eq_0_l409_409967

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l409_409967


namespace AH_tangent_to_circumcircle_AYZ_l409_409015

-- Assume basic structures for points and triangles
variables (A B C H E F X Y Z : Point)

-- Assume we have definitions for orthocenter, feet of altitudes, and circumcircles
def is_orthocenter (ABC : Triangle) (H : Point) : Prop := sorry
def foot_of_altitude_from (P Q R : Point) : Point := sorry
def circumcircle_intersection (P Q R S : Point) : Point := sorry

-- Assume points defined based on given conditions
axiom H_def : is_orthocenter ⟨A, B, C⟩ H
axiom E_def : E = foot_of_altitude_from B A C
axiom F_def : F = foot_of_altitude_from C A B
axiom Y_def : Y = circumcircle_intersection B E X A
axiom Z_def : Z = circumcircle_intersection C F X A

-- To Prove
theorem AH_tangent_to_circumcircle_AYZ :
  tangent (line_through A H) (circumcircle A Y Z) :=
sorry

end AH_tangent_to_circumcircle_AYZ_l409_409015


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409120

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409120


namespace limit_sequence_to_zero_l409_409824

noncomputable def limit_sequence : ℕ → ℝ := 
  λ n, ((n + 2)^2 - (n - 2)^2) / ((n + 3)^2)

theorem limit_sequence_to_zero : 
  tendsto (λ n, limit_sequence n) at_top (𝓝 0) :=
sorry

end limit_sequence_to_zero_l409_409824


namespace cos_90_eq_zero_l409_409955

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l409_409955


namespace total_stamps_l409_409820

-- Definitions for the conditions.
def snowflake_stamps : ℕ := 11
def truck_stamps : ℕ := snowflake_stamps + 9
def rose_stamps : ℕ := truck_stamps - 13

-- Statement to prove the total number of stamps.
theorem total_stamps : (snowflake_stamps + truck_stamps + rose_stamps) = 38 :=
by
  sorry

end total_stamps_l409_409820


namespace compute_r_plus_s_l409_409000

theorem compute_r_plus_s (x a b r s : ℝ) (h1 : ∛x + ∛(16 - x) = 2) 
(h2 : 0 ≤ x ∧ x ≤ 16 ∧ a ≠ b) 
(h3 : x = (1 - (sqrt 21)/3) ^ 3) 
(h4 : r = 1 ∧ s = 21) : r + s = 22 := 
by 
  sorry

end compute_r_plus_s_l409_409000


namespace smallest_digit_never_in_units_place_of_odd_l409_409205

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l409_409205


namespace mia_mom_tidy_up_toys_l409_409028

theorem mia_mom_tidy_up_toys
  (total_toys : ℕ)
  (mom_rate : ℕ)
  (mia_rate : ℕ)
  (net_rate : ℕ)
  (condition1 : total_toys = 45)
  (condition2 : mom_rate = 4)
  (condition3 : mia_rate = 1)
  (condition4 : net_rate = mom_rate - mia_rate) :
  let total_time := (total_toys - mom_rate) / net_rate + 1
  in total_time = 15 :=
by
  sorry

end mia_mom_tidy_up_toys_l409_409028


namespace cos_of_90_degrees_l409_409991

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l409_409991


namespace sum_difference_even_odd_2003_l409_409111

theorem sum_difference_even_odd_2003 :
  let S_odd := (λ n, ∑ k in Finset.range n, (2 * k + 1))
  let S_even := (λ n, ∑ k in Finset.range n, (2 * k + 2))
  S_even 2003 - S_odd 2003 = 2003 :=
by
  let S_odd := (λ n, ∑ k in Finset.range n, (2 * k + 1))
  let S_even := (λ n, ∑ k in Finset.range n, (2 * k + 2))
  show S_even 2003 - S_odd 2003 = 2003
  sorry

end sum_difference_even_odd_2003_l409_409111


namespace line_outside_circle_l409_409577

-- Define the circle and its properties
def diameter (c : Circle) : ℝ := 4
def distance_center_to_line (O : Point) (l : Line) : ℝ := 3

-- Define what it means for a line to be outside a circle
def is_outside_circle (c : Circle) (l : Line) : Prop := 
  ∃ O : Point, (distance_center_to_line O l > diameter c / 2)

-- The main theorem
theorem line_outside_circle {c : Circle} {l : Line} (O : Point)
  (h₁ : diameter c = 4)
  (h₂ : distance_center_to_line O l = 3) :
  is_outside_circle c l :=
by
  -- The proof goes here
  sorry

end line_outside_circle_l409_409577


namespace problem_l409_409527

noncomputable def f (x : ℝ) : ℝ := sorry

theorem problem 
  (h_odd : ∀ x : ℝ, f (-x) = -f x) 
  (h_periodic : ∀ x : ℝ, f (x + 1) = f (1 - x)) 
  (h_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2 ^ x - 1) 
  : f 2019 = -1 := 
sorry

end problem_l409_409527


namespace cape_may_vs_daytona_shark_sightings_diff_l409_409830

-- Definitions based on the conditions
def total_shark_sightings := 40
def cape_may_sightings : ℕ := 24
def daytona_beach_sightings : ℕ := total_shark_sightings - cape_may_sightings

-- The main theorem stating the problem in Lean
theorem cape_may_vs_daytona_shark_sightings_diff :
  (2 * daytona_beach_sightings - cape_may_sightings) = 8 := by
  sorry

end cape_may_vs_daytona_shark_sightings_diff_l409_409830


namespace geography_book_price_l409_409407

open Real

-- Define the problem parameters
def num_english_books : ℕ := 35
def num_geography_books : ℕ := 35
def cost_english : ℝ := 7.50
def total_cost : ℝ := 630.00

-- Define the unknown we need to prove
def cost_geography : ℝ := 10.50

theorem geography_book_price :
  num_english_books * cost_english + num_geography_books * cost_geography = total_cost :=
by
  -- No need to include the proof steps
  sorry

end geography_book_price_l409_409407


namespace david_reading_time_l409_409427

theorem david_reading_time
  (total_time : ℕ)
  (math_time : ℕ)
  (spelling_time : ℕ)
  (reading_time : ℕ)
  (h1 : total_time = 60)
  (h2 : math_time = 15)
  (h3 : spelling_time = 18)
  (h4 : reading_time = total_time - (math_time + spelling_time)) :
  reading_time = 27 := 
by {
  sorry
}

end david_reading_time_l409_409427


namespace smallest_digit_not_found_in_units_place_of_odd_number_l409_409134

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l409_409134


namespace rearrangement_inequality_l409_409652

theorem rearrangement_inequality (a : Fin n → ℕ) (H : ∀ i, a i = i + 1) :
  (∑ i in Finset.range (n - 1), (i + 1) / (i + 2 : ℕ)) ≤ (∑ i in Finset.range (n - 1), a i / a (i + 1)) :=
sorry

end rearrangement_inequality_l409_409652


namespace smallest_digit_never_in_units_place_of_odd_l409_409195

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l409_409195


namespace sphere_contains_six_points_l409_409593

theorem sphere_contains_six_points (cube_side_length : ℝ) (total_points : ℕ) (sphere_radius : ℝ)
  (h1 : cube_side_length = 15) (h2 : total_points = 11000) (h3 : sphere_radius = 1) :
  ∃ (s : ℝ) (p : set (ℝ × ℝ × ℝ)), s = 1 ∧ (∃ points: finset (ℝ × ℝ × ℝ), points.card ≥ 6 ∧ points ⊆ p) :=
sorry

end sphere_contains_six_points_l409_409593


namespace max_sector_area_central_angle_l409_409106

theorem max_sector_area_central_angle (radius arc_length : ℝ) :
  (arc_length + 2 * radius = 20) ∧ (arc_length = 20 - 2 * radius) ∧
  (arc_length / radius = 2) → 
  arc_length / radius = 2 :=
by
  intros h 
  sorry

end max_sector_area_central_angle_l409_409106


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409131

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409131


namespace compare_exponents_l409_409834

/-- Define the expressions as Lean functions -/
noncomputable def exp_4_1_4 := Real.exp (1 / 4 * Real.log 4)
noncomputable def exp_6_1_6 := Real.exp (1 / 6 * Real.log 6)
noncomputable def exp_5_1_5 := Real.exp (1 / 5 * Real.log 5)
noncomputable def exp_12_1_12 := Real.exp (1 / 12 * Real.log 12)

/-- Prove the greatest and the next to greatest values -/
theorem compare_exponents :
  (exp_4_1_4 = Real.max (Real.max exp_4_1_4 exp_6_1_6) (Real.max exp_5_1_5 exp_12_1_12))
  ∧ (exp_5_1_5 = Real.max (Real.min exp_4_1_4 (Real.max exp_6_1_6 exp_12_1_12)) exp_5_1_5) :=
  by sorry

end compare_exponents_l409_409834


namespace smallest_digit_not_found_in_units_place_of_odd_number_l409_409146

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l409_409146


namespace cos_90_equals_0_l409_409930

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l409_409930


namespace striped_octopus_has_eight_legs_l409_409570

variable (has_even_legs : ℕ → Prop)
variable (lie_told : ℕ → Prop)

variable (green_leg_count : ℕ)
variable (blue_leg_count : ℕ)
variable (violet_leg_count : ℕ)
variable (striped_leg_count : ℕ)

-- Conditions
axiom even_truth_lie_relation : ∀ n, has_even_legs n ↔ ¬lie_told n
axiom green_statement : lie_told green_leg_count ↔ (has_even_legs green_leg_count ∧ lie_told blue_leg_count)
axiom blue_statement : lie_told blue_leg_count ↔ (has_even_legs blue_leg_count ∧ lie_told green_leg_count)
axiom violet_statement : lie_told violet_leg_count ↔ (has_even_legs blue_leg_count ∧ ¬has_even_legs violet_leg_count)
axiom striped_statement : ¬has_even_legs green_leg_count ∧ ¬has_even_legs blue_leg_count ∧ ¬has_even_legs violet_leg_count ∧ has_even_legs striped_leg_count

-- The Proof Goal
theorem striped_octopus_has_eight_legs : has_even_legs striped_leg_count ∧ striped_leg_count = 8 :=
by
  sorry -- Proof to be filled in

end striped_octopus_has_eight_legs_l409_409570


namespace A_excircle_tangent_circumcenter_centroid_l409_409550

-- Defining the main conditions
variables (A : Point) (ω_A : Circle)
-- Definition of the statement in Lean 4

theorem A_excircle_tangent_circumcenter_centroid :
  ∃ B C : Point, 
    (is_tangent_point_of A ω_A B) ∧ (is_tangent_point_of A ω_A C) ∧ 
    (circumcenter_of_triangle A B C = centroid_of_tangent_triangle A ω_A) ∧
    ( ∃ G : Point, (G = circumcenter_of_triangle A B C) ∧ (G = centroid_of_tangent_triangle A ω_A)  ) → 
    ∃ (Δ₁ Δ₂ : Triangle), 
      (Δ₁ ≠ Δ₂ ∧ reflexive_symmetry_about AI_A Δ₁ Δ₂) ∧ 
      ∀ Δ : Triangle, 
        (circumcenter Δ = centroid_of_tangent_triangle A ω_A) →
        Δ = Δ₁ ∨ Δ = Δ₂ := 
sorry

end A_excircle_tangent_circumcenter_centroid_l409_409550


namespace solve_equation_l409_409361

theorem solve_equation (x : ℝ) : 2 * x - 1 = 3 * x + 3 → x = -4 :=
by
  intro h
  sorry

end solve_equation_l409_409361


namespace smallest_digit_not_in_units_place_of_odd_l409_409275

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l409_409275


namespace smallest_digit_never_at_units_place_of_odd_l409_409170

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l409_409170


namespace max_min_values_l409_409002

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def additive (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x + f y

def negative_for_positive (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x < 0

theorem max_min_values (f : ℝ → ℝ)
  (odd_f : odd_function f)
  (add_f : additive f)
  (neg_f : negative_for_positive f)
  (f1 : f 1 = -2) :
  (∀ x ∈ set.Icc (-3:ℝ) 3, f x ≤ 6) ∧ (∀ x ∈ set.Icc (-3:ℝ) 3, f x ≥ -6) :=
sorry

end max_min_values_l409_409002


namespace smallest_digit_not_found_in_units_place_of_odd_number_l409_409142

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l409_409142


namespace exists_natural_pairs_a_exists_natural_pair_b_l409_409447

open Nat

-- Part (a) Statement
theorem exists_natural_pairs_a (x y : ℕ) :
  x^2 - y^2 = 105 → (x, y) = (53, 52) ∨ (x, y) = (19, 16) ∨ (x, y) = (13, 8) ∨ (x, y) = (11, 4) :=
sorry

-- Part (b) Statement
theorem exists_natural_pair_b (x y : ℕ) :
  2*x^2 + 5*x*y - 12*y^2 = 28 → (x, y) = (8, 5) :=
sorry

end exists_natural_pairs_a_exists_natural_pair_b_l409_409447


namespace cube_edge_length_and_volume_l409_409718

variable (edge_length : ℕ)

def cube_edge_total_length (edge_length : ℕ) : ℕ := edge_length * 12
def cube_volume (edge_length : ℕ) : ℕ := edge_length * edge_length * edge_length

theorem cube_edge_length_and_volume (h : cube_edge_total_length edge_length = 96) :
  edge_length = 8 ∧ cube_volume edge_length = 512 :=
by
  sorry

end cube_edge_length_and_volume_l409_409718


namespace mileage_difference_correct_l409_409038

variable (estimated_mileage : ℕ) (tank_capacity : ℕ) (distance_driven : ℕ)
variable (actual_mileage : ℕ) (mileage_difference : ℕ)

-- Given conditions
def estimated_mileage := 35
def tank_capacity := 12
def distance_driven := 372

-- Calculate actual mileage
def actual_mileage : ℕ := distance_driven / tank_capacity

-- Define expected mileage difference
def mileage_difference : ℕ := estimated_mileage - actual_mileage

-- Claim the mileage difference is 4
theorem mileage_difference_correct : mileage_difference = 4 := 
sorry

end mileage_difference_correct_l409_409038


namespace smallest_unfound_digit_in_odd_units_l409_409226

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l409_409226


namespace max_min_f_product_of_roots_f_l409_409532

noncomputable def f (x : ℝ) : ℝ := 
  (Real.log x / Real.log 3 - 3) * (Real.log x / Real.log 3 + 1)

theorem max_min_f
  (x : ℝ) (h : x ∈ Set.Icc (1/27 : ℝ) (1/9 : ℝ)) : 
  (∀ y, y ∈ Set.Icc (1/27 : ℝ) (1/9 : ℝ) → f y ≤ 12)
  ∧ (∀ y, y ∈ Set.Icc (1/27 : ℝ) (1/9 : ℝ) → f y ≥ 5) :=
sorry

theorem product_of_roots_f
  (m α β : ℝ) (h1 : f α + m = 0) (h2 : f β + m = 0) : 
  (Real.log (α * β) / Real.log 3 = 2) → (α * β = 9) :=
sorry

end max_min_f_product_of_roots_f_l409_409532


namespace smallest_digit_not_in_units_place_of_odd_l409_409280

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l409_409280


namespace problem_1_problem_2_l409_409482

-- Definition of the function f
def f (k : ℕ) : ℕ := 
  {i | k + 1 ≤ i ∧ i ≤ 2 * k ∧ (nat.bitcount 1 i = 3)}.card

-- Problem 1: ∀m ∈ ℕ^+, ∃k ∈ ℕ^+ such that f(k) = m
theorem problem_1 (m : ℕ) (hm : m > 0) : 
  ∃ k : ℕ, k > 0 ∧ f(k) = m := 
sorry

-- Problem 2: ∃!k ∈ ℕ^+ such that f(k) = m
theorem problem_2 (m : ℕ) (hm : m > 0) : 
  (∃ k : ℕ, k > 0 ∧ f(k) = m) ↔ 
  ∃ s : ℕ, s ≥ 2 ∧ m = (s * (s - 1)) / 2 + 1 :=
sorry

end problem_1_problem_2_l409_409482


namespace cos_pi_half_eq_zero_l409_409864

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l409_409864


namespace banker_discount_calculation_l409_409696

-- Define the future value function with given interest rates and periods.
def face_value (PV : ℝ) : ℝ :=
  (PV * (1 + 0.10) ^ 4) * (1 + 0.12) ^ 4

-- Define the true discount as the difference between the future value and the present value.
def true_discount (PV : ℝ) : ℝ :=
  face_value PV - PV

-- Given conditions
def banker_gain : ℝ := 900

-- Define the banker's discount.
def banker_discount (PV : ℝ) : ℝ :=
  banker_gain + true_discount PV

-- The proof statement to prove the relationship.
theorem banker_discount_calculation (PV : ℝ) :
  banker_discount PV = banker_gain + (face_value PV - PV) := by
  sorry

end banker_discount_calculation_l409_409696


namespace five_digit_even_number_with_1_2_adjacent_l409_409487

def is_even (n : ℕ) : Prop := n % 2 = 0

def no_repeating_digits (n : ℕ) : Prop :=
  let digits := list.to_finset (nat.digits 10 n) in
  digits.card = nat.digits 10 n.length

def five_digit_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def digits_0_1_2_3_4 (n : ℕ) : Prop :=
  list.to_finset (nat.digits 10 n) ⊆ {0, 1, 2, 3, 4}

def ones_and_twos_adjacent (n : ℕ) : Prop :=
  let digits := nat.digits 10 n in
  (1 :: 2 :: tail (tail digits)) ∈ list.tails digits ∨
  (2 :: 1 :: tail (tail digits)) ∈ list.tails digits

theorem five_digit_even_number_with_1_2_adjacent :
  ∃ (k : ℕ), five_digit_number k ∧ no_repeating_digits k ∧ is_even k ∧ digits_0_1_2_3_4 k ∧ ones_and_twos_adjacent k ∧ k.card = 24 := sorry

end five_digit_even_number_with_1_2_adjacent_l409_409487


namespace adding_sugar_increases_sweetness_l409_409729

theorem adding_sugar_increases_sweetness 
  (a b m : ℝ) (hb : b > a) (ha : a > 0) (hm : m > 0) : 
  (a / b) < (a + m) / (b + m) := 
by
  sorry

end adding_sugar_increases_sweetness_l409_409729


namespace cos_90_eq_zero_l409_409844

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l409_409844


namespace smallest_digit_not_in_units_place_of_odd_l409_409274

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l409_409274


namespace smallest_digit_not_in_units_place_of_odd_l409_409258

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l409_409258


namespace victor_cannot_escape_k4_l409_409812

theorem victor_cannot_escape_k4
  (r : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ) 
  (k : ℝ)
  (hr : r = 1)
  (hk : k = 4)
  (hA_speed : speed_A = 4 * speed_B)
  (B_starts_at_center : ∃ (B : ℝ), B = 0):
  ¬(∃ (escape_strategy : ℝ → ℝ), escape_strategy 0 = 0 → escape_strategy r = 1) :=
sorry

end victor_cannot_escape_k4_l409_409812


namespace steve_average_speed_l409_409064

theorem steve_average_speed 
  (Speed1 Time1 Speed2 Time2 : ℝ) 
  (cond1 : Speed1 = 40) 
  (cond2 : Time1 = 5)
  (cond3 : Speed2 = 80) 
  (cond4 : Time2 = 3) 
: 
(Speed1 * Time1 + Speed2 * Time2) / (Time1 + Time2) = 55 := 
sorry

end steve_average_speed_l409_409064


namespace functional_ineq_solution_l409_409012

theorem functional_ineq_solution (n : ℕ) (h : n > 0) :
  (∀ x : ℝ, n = 1 → (x^n + (1 - x)^n ≤ 1)) ∧
  (∀ x : ℝ, n > 1 → ((x < 0 ∨ x > 1) → (x^n + (1 - x)^n > 1))) :=
by
  intros
  sorry

end functional_ineq_solution_l409_409012


namespace eval_composition_l409_409692

noncomputable def g : ℕ → ℕ := sorry
noncomputable def g_inv : ℕ → ℕ := sorry

axiom g_definition_4 : g 4 = 7
axiom g_definition_6 : g 6 = 3
axiom g_definition_1 : g 1 = 6
axiom g_inv_is_inverse : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x

theorem eval_composition : g_inv (g_inv 6 + g_inv 3) = 4 :=
by
  have h_inv_3 : g_inv 3 = 6 := 
    (g_inv_is_inverse 3).right ▸ g_definition_6
  have h_inv_6 : g_inv 6 = 1 :=
    (g_inv_is_inverse 6).right ▸ g_definition_1
  calc
    g_inv (g_inv 6 + g_inv 3)
        = g_inv (1 + 6)       : by rw [h_inv_6, h_inv_3]
    ... = g_inv 7            : by norm_num
    ... = 4                  : (g_inv_is_inverse 7).right ▸ g_definition_4

end eval_composition_l409_409692


namespace ratio_of_girls_who_like_bb_to_boys_dont_like_bb_l409_409624

-- Definitions based on the given conditions.
def total_students : ℕ := 25
def percent_girls : ℕ := 60
def percent_boys_like_bb : ℕ := 40
def percent_girls_like_bb : ℕ := 80

-- Results from those conditions.
def num_girls : ℕ := percent_girls * total_students / 100
def num_boys : ℕ := total_students - num_girls
def num_boys_like_bb : ℕ := percent_boys_like_bb * num_boys / 100
def num_boys_dont_like_bb : ℕ := num_boys - num_boys_like_bb
def num_girls_like_bb : ℕ := percent_girls_like_bb * num_girls / 100

-- Proof Problem Statement
theorem ratio_of_girls_who_like_bb_to_boys_dont_like_bb :
  (num_girls_like_bb : ℕ) / num_boys_dont_like_bb = 2 / 1 :=
by
  sorry

end ratio_of_girls_who_like_bb_to_boys_dont_like_bb_l409_409624


namespace smallest_digit_not_in_units_place_of_odd_numbers_l409_409157

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l409_409157


namespace top_king_second_queen_probability_l409_409404

noncomputable def probability_top_king_second_queen 
  (cards : Finset (Fin 52)) (shuffled : cards.nonempty) : ℚ :=
  let total_ways := 52 * 51
  let favorable_ways := 4 * 4
  favorable_ways / total_ways

theorem top_king_second_queen_probability :
  probability_top_king_second_queen (Finset.univ) (by simp) = 4 / 663 :=
sorry

end top_king_second_queen_probability_l409_409404


namespace unique_solution_l409_409429

theorem unique_solution (x y z t : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0) :
  12^x + 13^y - 14^z = 2013^t → (x = 1 ∧ y = 3 ∧ z = 2 ∧ t = 1) :=
by
  intros h
  sorry

end unique_solution_l409_409429


namespace books_in_collection_l409_409403

theorem books_in_collection (B : ℕ) (loaned_out : ℕ) (returned_percent : ℚ) (end_books : ℕ)
  (h_loan : loaned_out = 55)
  (h_returned : returned_percent = 0.80)
  (h_end : end_books = 64) :
  B = end_books + (loaned_out - (returned_percent * loaned_out).nat_abs) := by
  sorry

end books_in_collection_l409_409403


namespace enclosed_area_l409_409814

-- Definitions and conditions
def US := 2
def UT := 2
def angle_TUS := 60 -- degrees
def radius := 2
def arc_fraction := 1 / 6
def triangle_area := (4 * Real.sqrt 3)
def sectors_area := (4 * Real.pi) / 3

-- Theorem statement
theorem enclosed_area : (triangle_area - sectors_area) = (4 * Real.sqrt 3 - (4 * Real.pi) / 3) := 
by 
  sorry

end enclosed_area_l409_409814


namespace sum_binom_identity_l409_409042

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ
| n, 0       := 1
| 0, k + 1   := 0
| n + 1, k + 1 := binom n k + binom n (k + 1)

-- Define the main theorem statement
theorem sum_binom_identity (n : ℕ) : 
  ∑ j in finset.range (n + 1), 
    ((binom (3 * n + 2 - j) j) * (2 ^ j) - 
     if j = 0 then 0 else (binom (3 * n + 1 - j) (j - 1)) * (2 ^ (j - 1))) 
  = 2 ^ (3 * n) :=
by
  sorry -- Proof is omitted

end sum_binom_identity_l409_409042


namespace cos_ninety_degrees_l409_409926

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l409_409926


namespace intersection_of_A_and_B_l409_409650

variable (A : Set ℕ) (B : Set ℕ)

axiom h1 : A = {1, 2, 3, 4, 5}
axiom h2 : B = {3, 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} :=
  by sorry

end intersection_of_A_and_B_l409_409650


namespace smallest_missing_digit_l409_409325

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l409_409325


namespace percent_of_4_over_50_l409_409385

def part : ℝ := 4
def whole : ℝ := 50
def percent (p w : ℝ) : ℝ := (p / w) * 100

theorem percent_of_4_over_50 : percent part whole = 8 := by
  sorry

end percent_of_4_over_50_l409_409385


namespace number_x_is_divided_by_l409_409746

-- Define the conditions
variable (x y n : ℕ)
variable (cond1 : x = n * y + 4)
variable (cond2 : 2 * x = 8 * 3 * y + 3)
variable (cond3 : 13 * y - x = 1)

-- Define the statement to be proven
theorem number_x_is_divided_by : n = 11 :=
by
  sorry

end number_x_is_divided_by_l409_409746


namespace number_of_zeros_l409_409435

-- Define the problem conditions and the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ set.Ioc (0 : ℝ) (3 / 2) then real.sin (real.pi * x)
  else if x ∈ set.Ioc (- (3 / 2) : ℝ) 0 then -real.sin (real.pi * -x)
  else real.sin (real.pi * (x - 3 * ((x / 3).floor : ℤ)))

-- Prove the number of zeros in [0, 5] is 6
theorem number_of_zeros (f) : ∃ (n : ℕ), n = 6 ∧ ∀ x ∈ set.Icc (0 : ℝ) 5, f x = 0 ↔ (x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) :=
by
  sorry

end number_of_zeros_l409_409435


namespace fraction_product_l409_409743

theorem fraction_product : 
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_l409_409743


namespace cos_90_deg_eq_zero_l409_409911

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l409_409911


namespace cos_90_eq_zero_l409_409946

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l409_409946


namespace option_D_monotonically_decreasing_l409_409806

theorem option_D_monotonically_decreasing : 
  ∀ x y : ℝ, (0 < x) → (0 < y) → (x < y) → 
  (real.rpow (1/2) x > real.rpow (1/2) y) := 
by 
  intros x y hx hy hxy 
  sorry

end option_D_monotonically_decreasing_l409_409806


namespace cos_ninety_degrees_l409_409914

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l409_409914


namespace cos_90_eq_0_l409_409885

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l409_409885


namespace partition_nat_l409_409620

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem partition_nat (S1 S2 : set ℕ) (hS1_nonempty : ∃ x, x ∈ S1) (hS2_nonempty : ∃ x, x ∈ S2) (hS1 : ∀ a ∈ S1, ∀ b ∈ S1, ab - 1 ∈ S2) (hS2 : ∀ a ∈ S2, ∀ b ∈ S2, ab - 1 ∈ S1) :
  ∀ n : ℕ, n > 1 → n ∈ S1 ∨ n ∈ S2 :=
begin
  sorry
end

end partition_nat_l409_409620


namespace cubic_coeff_relationship_l409_409453

theorem cubic_coeff_relationship (a b c d u v w : ℝ) 
  (h_eq : a * (u^3) + b * (u^2) + c * u + d = 0)
  (h_vieta1 : u + v + w = -(b / a)) 
  (h_vieta2 : u * v + u * w + v * w = c / a) 
  (h_vieta3 : u * v * w = -d / a) 
  (h_condition : u + v = u * v) :
  (c + d) * (b + c + d) = a * d :=
by 
  sorry

end cubic_coeff_relationship_l409_409453


namespace cos_90_eq_zero_l409_409853

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l409_409853


namespace chord_length_l409_409770

theorem chord_length (r d : ℝ) (h_r : r = 5) (h_d : d = 4) : ∃ EF : ℝ, EF = 6 :=
by
  let OF := 5
  let OG := 4
  have h_OG_OF : OG^2 + (EF / 2)^2 = OF^2 := sorry -- Pythagorean relationship
  have GF := 3
  use 2 * GF
  sorry

end chord_length_l409_409770


namespace smallest_not_odd_unit_is_zero_l409_409192

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l409_409192


namespace number_of_digits_in_sum_l409_409560

def is_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 9

theorem number_of_digits_in_sum (C D : ℕ) (hC : is_digit C) (hD : is_digit D) :
  let n1 := 98765
  let n2 := C * 1000 + 433
  let n3 := D * 100 + 22
  let s := n1 + n2 + n3
  100000 ≤ s ∧ s < 1000000 :=
by {
  sorry
}

end number_of_digits_in_sum_l409_409560


namespace cos_90_eq_zero_l409_409856

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l409_409856


namespace tenth_permutation_is_correct_l409_409728

theorem tenth_permutation_is_correct :
  let digits := [1, 3, 6, 8]
  let perms := digits.permutations.map (λ ds => ds.foldl (λ acc d => acc * 10 + d) 0)
  let sorted_perms := perms.qsort (≤)
  sorted_perms.nth 9 = some 3681 :=
by
  sorry

end tenth_permutation_is_correct_l409_409728


namespace intervals_of_monotonicity_range_of_a_l409_409539

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - real.log (x + 1)

theorem intervals_of_monotonicity (x : ℝ) :
  (∀ x ∈ Ioo (-(real.sqrt 2 / 2)) (real.sqrt 2 / 2), f (-1) x < f (-1) x)
  ∧ (∀ x ∈ Ioo (-1) (-(real.sqrt 2 / 2)) ∪ Ioi (real.sqrt 2 / 2), f (-1) x > f (-1) x) :=
sorry

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Icc (0 : ℝ) (real.infinity), f a x ≤ x) ↔ a ∈ Iic (- (1 / 2)) :=
sorry

end intervals_of_monotonicity_range_of_a_l409_409539


namespace intersection_of_M_and_N_l409_409489

noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 5}
noncomputable def N : Set ℝ := {x | x * (x - 4) > 0}

theorem intersection_of_M_and_N :
  M ∩ N = { x : ℝ | (-1 < x ∧ x < 0) ∨ (4 < x ∧ x < 5) } := by
  sorry

end intersection_of_M_and_N_l409_409489


namespace sqrt_nine_factorial_over_72_eq_l409_409825

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_nine_factorial_over_72_eq : 
  Real.sqrt ((factorial 9) / 72) = 12 * Real.sqrt 35 :=
by
  sorry

end sqrt_nine_factorial_over_72_eq_l409_409825


namespace solve_for_q_l409_409565

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 6 * q = 20) (h2 : 6 * p + 5 * q = 29) : q = -25 / 11 :=
by {
  sorry
}

end solve_for_q_l409_409565


namespace smallest_n_l409_409350

theorem smallest_n (n : ℕ) (h₁ : ∃ k₁ : ℕ, 5 * n = k₁ ^ 2) (h₂ : ∃ k₂ : ℕ, 4 * n = k₂ ^ 3) : n = 1600 :=
sorry

end smallest_n_l409_409350


namespace smallest_digit_not_in_units_place_of_odd_numbers_l409_409151

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l409_409151


namespace Ann_baked_oatmeal_raisin_cookies_l409_409813

/-
Ann bakes some dozens of oatmeal raisin cookies, 2 dozen sugar cookies, and 4 dozen chocolate chip cookies.
She gives away 2 dozen oatmeal raisin cookies, 1.5 dozen sugar cookies, and 2.5 dozen chocolate chip cookies.
Ann keeps 3 dozens of cookies in total.
Prove that Ann baked 3 dozens of oatmeal raisin cookies.
-/

variables (d_oat_baked d_oat_given d_sug_baked d_sug_given d_cc_baked d_cc_given d_kept : ℕ)
variable  (h_d_kept : d_kept = 36 / 12) -- Ann keeps 3 dozens of cookies
variable  (h_d_sug_baked : d_sug_baked = 2)  -- 2 dozens sugar cookies baked
variable  (h_d_cc_baked : d_cc_baked = 4)    -- 4 dozens chocolate chip cookies baked
variable  (h_d_oat_given : d_oat_given = 2)  -- 2 dozens oatmeal raisin cookies given away
variable  (h_d_sug_given : d_sug_given = 1.5) -- 1.5 dozens sugar cookies given away
variable  (h_d_cc_given : d_cc_given = 2.5)  -- 2.5 dozens chocolate chip cookies given away

/-
Let's prove the total dozens of cookies given away is 6.
Let's prove the total dozens of cookies baked is 9.
Let's prove the remaining dozens of cookies must be oatmeal raisin cookies.
So, prove Ann baked 3 dozens of oatmeal raisin cookies.
-/

theorem Ann_baked_oatmeal_raisin_cookies :
  ∃ d_oat_baked, 
  d_oat_baked = (d_kept + (d_oat_given + d_sug_given + d_cc_given)) - (d_sug_baked + d_cc_baked) 
   := sorry

end Ann_baked_oatmeal_raisin_cookies_l409_409813


namespace alberto_bjorn_distance_difference_l409_409588

-- Definitions based on given conditions
def alberto_speed : ℕ := 12  -- miles per hour
def bjorn_speed : ℕ := 10    -- miles per hour
def total_time : ℕ := 6      -- hours
def bjorn_rest_time : ℕ := 1 -- hours

def alberto_distance : ℕ := alberto_speed * total_time
def bjorn_distance : ℕ := bjorn_speed * (total_time - bjorn_rest_time)

-- The statement to prove
theorem alberto_bjorn_distance_difference :
  (alberto_distance - bjorn_distance) = 22 :=
by
  sorry

end alberto_bjorn_distance_difference_l409_409588


namespace smallest_digit_not_in_units_place_of_odd_l409_409264

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l409_409264


namespace cos_90_eq_zero_l409_409848

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l409_409848


namespace milo_cash_reward_l409_409655

-- Define the grades and credit hours for each subject
def mathematics_grade := 2
def mathematics_credit_hours := 5

def english_grade := 3
def english_credit_hours := 4
def english_classes := 3

def science_grade := 3
def science_credit_hours := 4
def science_classes := 2

def history_grade := 4
def history_credit_hours := 3

def art_grade := 5
def art_credit_hours := 2

-- Define the calculation for the total weighted grade points
def total_weighted_grade_points := 
  (mathematics_grade * mathematics_credit_hours) +
  (english_grade * english_credit_hours * english_classes) +
  (science_grade * science_credit_hours * science_classes) +
  (history_grade * history_credit_hours) +
  (art_grade * art_credit_hours)

-- Define the total credit hours
def total_credit_hours := 
  mathematics_credit_hours + 
  (english_credit_hours * english_classes) +
  (science_credit_hours * science_classes) +
  history_credit_hours +
  art_credit_hours

-- Calculate the weighted average grade
def weighted_average_grade := total_weighted_grade_points / total_credit_hours

-- Calculate the cash reward
def cash_reward := 5 * weighted_average_grade

-- Round the cash reward to the nearest cent
def rounded_cash_reward := Float.round cash_reward * 100 / 100

-- The theorem statement
theorem milo_cash_reward : rounded_cash_reward = 15.33 := 
  by
    sorry

end milo_cash_reward_l409_409655


namespace smallest_digit_not_in_units_place_of_odd_l409_409297

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l409_409297


namespace cos_of_90_degrees_l409_409988

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l409_409988


namespace sum_of_sequence_a_n_l409_409412

noncomputable def sum_of_first_20_terms : ℕ :=
  (finset.range 20).sum (λ n, (n + 1) ^ 2)

theorem sum_of_sequence_a_n :
  sum_of_first_20_terms = 2870 := 
  by
    sorry

end sum_of_sequence_a_n_l409_409412


namespace relationship_of_x_l409_409575

variables {x₁ x₂ x₃ : ℝ}

-- Definition of the function
def inverse_proportion (x : ℝ) : ℝ := -4 / x

-- Conditions that the points are on the function
def point_A (x₁ : ℝ) : Prop := inverse_proportion x₁ = -1
def point_B (x₂ : ℝ) : Prop := inverse_proportion x₂ = 3
def point_C (x₃ : ℝ) : Prop := inverse_proportion x₃ = 5

-- The theorem we want to prove
theorem relationship_of_x (hA : point_A x₁) (hB : point_B x₂) (hC : point_C x₃) : x₂ < x₃ ∧ x₃ < x₁ :=
by
  sorry

end relationship_of_x_l409_409575


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409130

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409130


namespace cos_of_90_degrees_l409_409992

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l409_409992


namespace find_λ_l409_409545

variables {a b c : ℝ × ℝ}
def λ_is_collinear (λ : ℝ) : Prop :=
  ∃ k : ℝ, λ • a + b = k • c

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (2, 0)
def vector_c : ℝ × ℝ := (1, -2)

theorem find_λ (λ : ℝ) (h : λ_is_collinear λ) : λ = -1 :=
by sorry

end find_λ_l409_409545


namespace log_eq_exp_l409_409472

theorem log_eq_exp {x : ℝ} (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end log_eq_exp_l409_409472


namespace cos_90_deg_eq_zero_l409_409907

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l409_409907


namespace cross_product_correct_l409_409451

def vec1 : ℝ × ℝ × ℝ := (3, 1, 4)
def vec2 : ℝ × ℝ × ℝ := (6, -2, 8)

def cross_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v1.2.1 * v2.2.2 - v1.2.2 * v2.2.1,  -- x-component
   v1.2.2 * v2.2 - v1.2 * v2.2.2,      -- y-component
   v1 * v2.2.1 - v1.2 * v2.2)          -- z-component

theorem cross_product_correct :
  cross_product vec1 vec2 = (16, 0, -12) :=
by 
  sorry

end cross_product_correct_l409_409451


namespace smallest_digit_not_in_units_place_of_odd_l409_409255

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l409_409255


namespace smallest_digit_not_in_odd_units_l409_409220

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l409_409220


namespace cos_90_eq_zero_l409_409851

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l409_409851


namespace cos_90_eq_zero_l409_409973

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l409_409973


namespace c_minus_a_value_l409_409576

theorem c_minus_a_value (a b c : ℝ) 
  (h1 : (a + b) / 2 = 50)
  (h2 : (b + c) / 2 = 70) : 
  c - a = 40 :=
by 
  sorry

end c_minus_a_value_l409_409576


namespace cos_90_eq_0_l409_409887

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l409_409887


namespace smallest_digit_not_in_units_place_of_odd_l409_409265

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l409_409265


namespace simplify_expression_l409_409441

theorem simplify_expression : Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 5 * Real.sqrt 15 :=
by
  sorry

end simplify_expression_l409_409441


namespace smallest_digit_not_in_units_place_of_odd_l409_409296

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l409_409296


namespace cos_90_eq_zero_l409_409855

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l409_409855


namespace cos_ninety_degrees_l409_409916

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l409_409916


namespace smallest_digit_not_in_odd_units_l409_409241

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l409_409241


namespace cos_90_eq_0_l409_409891

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l409_409891


namespace smallest_digit_never_at_units_place_of_odd_l409_409168

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l409_409168


namespace makaralas_meeting_break_time_percentage_l409_409024

theorem makaralas_meeting_break_time_percentage :
  let total_minutes := 10 * 60,
      first_meeting := 40,
      second_meeting := 2 * first_meeting,
      third_meeting := first_meeting + second_meeting,
      break_time := 20,
      total_meeting_time := first_meeting + second_meeting + third_meeting,
      total_meetings_and_break := total_meeting_time + break_time
  in (total_meetings_and_break / total_minutes) * 100 = 43.33 :=
by {
  let total_minutes := 10 * 60
  let first_meeting := 40
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let break_time := 20
  let total_meeting_time := first_meeting + second_meeting + third_meeting
  let total_meetings_and_break := total_meeting_time + break_time
  have h1 : (total_meetings_and_break : ℝ) / total_minutes = 43.33 / 100,
  { norm_num [total_meetings_and_break, total_minutes, first_meeting, second_meeting, third_meeting, break_time]},
  field_simp
}

end makaralas_meeting_break_time_percentage_l409_409024


namespace trip_time_l409_409062

-- Define the conditions
def constant_speed : ℝ := 62 -- in miles per hour
def total_distance : ℝ := 2790 -- in miles
def break_interval : ℝ := 5 -- in hours
def break_duration : ℝ := 0.5 -- in hours (30 minutes)
def hotel_search_time : ℝ := 0.5 -- in hours (30 minutes)

theorem trip_time :
  let total_driving_time := total_distance / constant_speed in
  let number_of_breaks := total_driving_time / break_interval in
  let total_break_time := number_of_breaks * break_duration in
  let total_time := total_driving_time + total_break_time + hotel_search_time in
  total_time = 50 :=
by
  sorry

end trip_time_l409_409062


namespace ratio_of_areas_is_3_to_1_l409_409394

noncomputable def ratio_of_areas_of_circles (R₁ R₂ : ℝ) (h₁ : R₁ > 0) (h₂ : R₂ > 0) 
  (chord_condition : (R₁ ^ 2) = 3 * (R₂ ^ 2)) : ℝ :=
  (π * R₁ ^ 2) / (π * R₂ ^ 2)

theorem ratio_of_areas_is_3_to_1 (R₁ R₂ : ℝ) (h₁ : R₁ > 0) (h₂ : R₂ > 0) 
  (chord_condition : (R₁ ^ 2) = 3 * (R₂ ^ 2)) : ratio_of_areas_of_circles R₁ R₂ h₁ h₂ chord_condition = 3 :=
by
  sorry

end ratio_of_areas_is_3_to_1_l409_409394


namespace cos_90_eq_0_l409_409969

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l409_409969


namespace no_pair_points_distance_gt_sqrt6_div_6_l409_409505

/-- Given a regular tetrahedron with edge length 1, and spheres with the edges as diameters,
let Σ be the intersection of the six constructed spheres. Prove that there does not exist a 
pair of points in Σ such that their distance is greater than sqrt(6) / 6. -/
theorem no_pair_points_distance_gt_sqrt6_div_6 :
  let edge_length := 1,
      centers := [{0, 0, 0}, {1, 0, 0}, {1/2, √3/2, 0}, {1/2, √3/6, √(6/3)}],
      Σ := ⋂ (i j : fin 4) (i ≠ j), sphere midpoints (edges i j) (length / 2)
  in ∀ (p1 p2 : point) (hp1 : p1 ∈ Σ) (hp2 : p2 ∈ Σ), dist p1 p2 ≤ sqrt (6) / 6 :=
begin
  sorry
end

end no_pair_points_distance_gt_sqrt6_div_6_l409_409505


namespace transform_to_zero_if_one_odd_l409_409807

/-- 
Given an m by n table with exactly one -1 (the rest are +1), 
it is possible to transform the table such that every cell contains 0 
using the defined move operations if at least one of m or n is odd. 
--/
theorem transform_to_zero_if_one_odd (m n : ℕ) : 
  (∃ (table : matrix (fin m) (fin n) ℤ), 
    (∑ i j, (if table i j = -1 then 1 else 0) = 1) ∧ 
    (∑ i j, (if table i j = 1 then 1 else 0) = m * n - 1) ∧
    (∀ i j, (abs (table i j) = 1) ∨ (table i j = 0))) → 
  (m % 2 = 1 ∨ n % 2 = 1) :=
sorry

end transform_to_zero_if_one_odd_l409_409807


namespace find_letters_with_dot_but_no_straight_line_l409_409753

-- Define the problem statement and conditions
def DL : ℕ := 16
def L : ℕ := 30
def Total_letters : ℕ := 50

-- Define the function that calculates the number of letters with a dot but no straight line
def letters_with_dot_but_no_straight_line (DL L Total_letters : ℕ) : ℕ := Total_letters - (L + DL)

-- State the theorem to be proved
theorem find_letters_with_dot_but_no_straight_line : letters_with_dot_but_no_straight_line DL L Total_letters = 4 :=
by
  sorry

end find_letters_with_dot_but_no_straight_line_l409_409753


namespace cos_of_90_degrees_l409_409995

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l409_409995


namespace smallest_digit_not_in_units_place_of_odd_numbers_l409_409148

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l409_409148


namespace imaginary_part_z_is_zero_l409_409708

def complex_num := ℂ

def z : complex_num := (1 + complex.I) / (1 - complex.I) + (1 - complex.I)

theorem imaginary_part_z_is_zero : z.im = 0 := 
by
  sorry

end imaginary_part_z_is_zero_l409_409708


namespace candidates_appeared_per_state_l409_409754

theorem candidates_appeared_per_state :
  ∃ x : ℝ, 0.07 * x = 0.06 * x + 80 ∧ x = 8000 :=
by
  existsi (8000 : ℝ)
  split
  · simp
  · rfl

end candidates_appeared_per_state_l409_409754


namespace horner_example_correct_l409_409107

def polynomial := λ x : ℝ, 12 + 3 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

noncomputable def horner_step (v₀ : ℝ) (x : ℝ) (aₙ₋₁ : ℝ) : ℝ :=
  v₀ * x + aₙ₋₁

theorem horner_example_correct :
  let x := -4
  let v₀ := 3
  let aₙ₋₁ := 5
  let v₁ := horner_step v₀ x aₙ₋₁
  v₁ = -7 :=
by {
  sorry
}

end horner_example_correct_l409_409107


namespace cos_90_equals_0_l409_409941

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l409_409941


namespace arith_seq_sum_nine_l409_409481

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arith_seq := ∀ n : ℕ, a n = a 0 + (n - 1) * (a 1 - a 0)

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := 
  ∀ n : ℕ, S n = (n / 2) * (a 0 + a (n - 1))

theorem arith_seq_sum_nine (h_seq : arith_seq a) (h_sum : sum_first_n_terms a S) (h_S9 : S 9 = 18) : 
  a 2 + a 5 + a 8 = 6 :=
  sorry

end arith_seq_sum_nine_l409_409481


namespace principal_amount_l409_409784

-- Define the conditions of the problem as constants
def total_amount_paid : ℝ := 9000
def annual_interest_rate : ℝ := 0.12
def compounding_periods_per_year : ℕ := 2
def loan_duration_years : ℕ := 3

-- Define the formula for compound interest
noncomputable def compound_interest (P : ℝ) : ℝ :=
  P * (1 + annual_interest_rate / compounding_periods_per_year) ^
    (compounding_periods_per_year * loan_duration_years)

-- Problem statement: prove that the principal amount borrowed is approximately $6347.69
theorem principal_amount (P : ℝ) (h : compound_interest P = total_amount_paid) : 
  P ≈ 6347.69 :=
sorry

end principal_amount_l409_409784


namespace mileage_difference_correct_l409_409039

variable (estimated_mileage : ℕ) (tank_capacity : ℕ) (distance_driven : ℕ)
variable (actual_mileage : ℕ) (mileage_difference : ℕ)

-- Given conditions
def estimated_mileage := 35
def tank_capacity := 12
def distance_driven := 372

-- Calculate actual mileage
def actual_mileage : ℕ := distance_driven / tank_capacity

-- Define expected mileage difference
def mileage_difference : ℕ := estimated_mileage - actual_mileage

-- Claim the mileage difference is 4
theorem mileage_difference_correct : mileage_difference = 4 := 
sorry

end mileage_difference_correct_l409_409039


namespace smallest_digit_not_in_odd_units_l409_409252

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l409_409252


namespace expression_simplifies_to_sqrt_2_l409_409828

theorem expression_simplifies_to_sqrt_2 :
  (1 / 2) ^ (-2: ℤ) + 2 * Real.sin (Float.pi / 4) - (Real.sqrt 2 - 1) ^ 0 - Real.cbrt 27 = Real.sqrt 2 :=
by
  sorry

end expression_simplifies_to_sqrt_2_l409_409828


namespace smallest_unfound_digit_in_odd_units_l409_409225

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l409_409225


namespace chord_length_l409_409769

theorem chord_length (r d : ℝ) (h_r : r = 5) (h_d : d = 4) : 
  ∃ (EF : ℝ), EF = 6 :=
by
  have h1 : r^2 = d^2 + (EF / 2)^2,
  sorry

end chord_length_l409_409769


namespace g_x_plus_1_minus_g_x_l409_409003

def g (x : ℝ) : ℝ := 8^x

theorem g_x_plus_1_minus_g_x (x : ℝ) : g (x + 1) - g x = 7 * g x := by
  sorry

end g_x_plus_1_minus_g_x_l409_409003


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409127

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409127


namespace cos_90_eq_0_l409_409968

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l409_409968


namespace sum_of_adjacent_to_7_l409_409713

-- Define the problem statement and the primary condition
def is_divisor (n d : ℕ) : Prop := d ∣ n

-- Define the set of divisors of 294, excluding 1
def divisors_294 : Finset ℕ := {2, 3, 7, 14, 21, 49, 42, 98, 147, 294}

-- Define the adjacency condition based on common factors
def has_common_factor (x y : ℕ) : Prop := ∃ d > 1, d ∣ x ∧ d ∣ y

-- Define the adjacency property around a circle
def arranged_in_circle (s : Finset ℕ) : Prop :=
  ∀ x ∈ s, ∃ y₁ y₂ ∈ s, y₁ ≠ x ∧ y₂ ≠ x ∧ has_common_factor x y₁ ∧ has_common_factor x y₂

-- Final problem statement to prove
theorem sum_of_adjacent_to_7 : 
  ∀ s ⊆ divisors_294, arranged_in_circle s → has_common_factor 7 (49 + 147) := sorry

end sum_of_adjacent_to_7_l409_409713


namespace smallest_digit_not_in_odd_units_l409_409221

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l409_409221


namespace smallest_unfound_digit_in_odd_units_l409_409229

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l409_409229


namespace sum_of_squares_distances_le_n_squared_R_squared_l409_409722

variables {n : ℕ} {R : ℝ}
variables {A : Fin n → ℝ × ℝ} 

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

def in_circle (p : ℝ × ℝ) (R : ℝ) : Prop := 
  p.1^2 + p.2^2 ≤ R^2

theorem sum_of_squares_distances_le_n_squared_R_squared 
  (hA : ∀ i, in_circle (A i) R) : 
  ∑ i j in {ij : Fin n × Fin n | i < j}, distance_squared (A i) (A j) ≤ n^2 * R^2 := 
by
  sorry

end sum_of_squares_distances_le_n_squared_R_squared_l409_409722


namespace cos_90_eq_zero_l409_409845

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l409_409845


namespace cost_price_determination_number_of_cups_purchased_l409_409767

variables (x y m : ℝ)
variables (A B : String)

-- The cost price conditions
def cost_price_conditions := 
  (4 * x + 5 * y = 4 * x + 5 * y) ∧ 
  (3 * x = 2 * y + 154)

-- The found cost prices
def cost_prices :=
  x = 110 ∧ y = 88

-- The purchase conditions
def purchase_conditions := 
  m + (80 - m) = 80 ∧
  160 * m + 140 * (80 - m) - (110 * m + 88 * (80 - m)) = 4100

-- The purchased quantities
def purchased_quantities := 
  m = 30 ∧ 80 - m = 50

-- First theorem: cost prices determination
theorem cost_price_determination (h : cost_price_conditions) : cost_prices := 
  sorry

-- Second theorem: number of cups purchased
theorem number_of_cups_purchased (h1 : cost_prices) (h2 : purchase_conditions) : purchased_quantities := 
  sorry

end cost_price_determination_number_of_cups_purchased_l409_409767


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409333

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409333


namespace geometric_sequence_a4_l409_409601

theorem geometric_sequence_a4 (a : ℝ) (r : ℝ):
  let a₃ := a * r^2,
      a₄ := a * r^3,
      a₅ := a * r^4
  in a₃ * a₅ = 64 → a₄ = 8 ∨ a₄ = -8 :=
by
  intro h,
  let a₄_sq := a * r^3,
  have h_a₄_sq : a₄ * a₄ = 64,
  sorry

end geometric_sequence_a4_l409_409601


namespace anderson_numbers_sum_of_digits_eq_24_l409_409808

def is_anderson_number (k : ℕ) : Prop :=
k < 10000 ∧ (k * k) % (10 ^ (Nat.log10 k + 1)) = k

def even_anderson_numbers : List ℕ := 
List.filter (λ k => is_anderson_number k ∧ k % 2 = 0) (List.range 10000)

def sum_even_anderson_numbers : ℕ := 
List.sum even_anderson_numbers

def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem anderson_numbers_sum_of_digits_eq_24 : 
  sum_of_digits sum_even_anderson_numbers = 24 := 
sorry

end anderson_numbers_sum_of_digits_eq_24_l409_409808


namespace smallest_digit_not_in_odd_units_l409_409250

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l409_409250


namespace cos_of_90_degrees_l409_409984

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l409_409984


namespace limit_r_as_m_approaches_zero_l409_409016

open Topology

-- Define L(m) based on the conditions
def L (m : ℝ) : ℝ := -Real.sqrt (m + 4)
def L_neg (m : ℝ) : ℝ := -Real.sqrt (-m + 4)

-- Define r based on the conditions
noncomputable def r (m : ℝ) : ℝ := (L_neg m - L m) / m

-- State the theorem to prove the limit
theorem limit_r_as_m_approaches_zero : tendsto r (nhds 0) (nhds (1 / 2)) :=
by
  sorry

end limit_r_as_m_approaches_zero_l409_409016


namespace simplify_root_sum_l409_409689

theorem simplify_root_sum : (sqrt 12 + sqrt 27 = 5 * sqrt 3) :=
by 
  have sqrt_12_simp : sqrt 12 = 2 * sqrt 3 := sorry
  have sqrt_27_simp : sqrt 27 = 3 * sqrt 3 := sorry
  calc
    sqrt 12 + sqrt 27
    = (2 * sqrt 3) + (3 * sqrt 3) : by rw [sqrt_12_simp, sqrt_27_simp]
    = 5 * sqrt 3 : by ring

end simplify_root_sum_l409_409689


namespace cos_90_equals_0_l409_409932

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l409_409932


namespace cos_of_90_degrees_l409_409997

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l409_409997


namespace smallest_digit_not_in_units_place_of_odd_l409_409288

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l409_409288


namespace randi_has_6_more_nickels_than_peter_l409_409050

def ray_initial_cents : Nat := 175
def cents_given_peter : Nat := 30
def cents_given_randi : Nat := 2 * cents_given_peter
def nickel_worth : Nat := 5

def nickels (cents : Nat) : Nat :=
  cents / nickel_worth

def randi_more_nickels_than_peter : Prop :=
  nickels cents_given_randi - nickels cents_given_peter = 6

theorem randi_has_6_more_nickels_than_peter :
  randi_more_nickels_than_peter :=
sorry

end randi_has_6_more_nickels_than_peter_l409_409050


namespace cos_90_eq_zero_l409_409846

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l409_409846


namespace ratio_eccentricities_hyperbola_ellipse_l409_409706

theorem ratio_eccentricities_hyperbola_ellipse (O : Point)
  (centered_origin : ∀ (C : ConicSection), C.center = O)
  (symmetric_axes : ∀ (C : ConicSection), C.symmetric := true)
  (common_focus : ∃ F: Focus, F ∈ Hyperbola.focuses ∧ F ∈ Ellipse.focuses)
  (vertex_hyperbola : ∀ (M N : Point), M ∈ Hyperbola.vertices ∧ N ∈ Hyperbola.vertices ∧ distance M N = a_hyperbola)
  (divide_major_axis : ∀ (M N O : Point), distance M O = distance O N ∧ distance M N = a_ellipse / 2) :
  (eccentricity Hyperbola / eccentricity Ellipse) = 2 := 
sorry

end ratio_eccentricities_hyperbola_ellipse_l409_409706


namespace six_points_concyclic_nine_points_concyclic_l409_409006

variables {A B C H: Point}
variables {MA MB MC HA HB HC A' B' C': Point}
variables {circumcircle : Π (A B C : Point), Circle}

-- Define Midpoints
axiom midpoints : (M : Point) -> (a b : Point) -> IsMidpoint M a b

-- Define Feet of the Altitudes
axiom feet_of_altitudes : (H : Point) -> (a b c: Point) -> IsFootOfAltitude H a b c

-- Define Orthocenter
axiom is_orthocenter : (H : Point) -> (a b c : Point) -> IsOrthocenter H a b c

-- Conditions
def conditions (A B C H MA MB MC HA HB HC A' B' C' : Point) := 
midpoints MA B C ∧ midpoints MB C A ∧ midpoints MC A B ∧
feet_of_altitudes HA A B C ∧ feet_of_altitudes HB B C A ∧ feet_of_altitudes HC C A B ∧
midpoints A' A H ∧ midpoints B' B H ∧ midpoints C' C H ∧ 
is_orthocenter H A B C

-- Theorem Statements
theorem six_points_concyclic (A B C H MA MB MC HA HB HC A' B' C' : Point) 
  (h_cond : conditions A B C H MA MB MC HA HB HC A' B' C') :
  IsConcyclic MA MB MC HA HB HC :=
sorry

theorem nine_points_concyclic (A B C H MA MB MC HA HB HC A' B' C' : Point) 
  (h_cond : conditions A B C H MA MB MC HA HB HC A' B' C') :
  IsConcyclic MA MB MC HA HB HC -> IsConcyclic MA MB MC HA HB HC A' B' C' :=
sorry

end six_points_concyclic_nine_points_concyclic_l409_409006


namespace smallest_digit_not_in_odd_units_l409_409245

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l409_409245


namespace triangle_is_larger_than_quadrilateral_l409_409672

theorem triangle_is_larger_than_quadrilateral
    (A B C D E E' : Point)
    (AE' AE E'D ED BD DC E'B EC : ℝ)
    (hE'E : E' ∈ lineSegment A D)
    (hDE' : E'D = ED)
    (hBDDC : BD = DC)
    (hAngle : angle B D E' = angle C D E) :
    AE' + E'D + DC + AE + EC > BD + AB + AE + ED := 
by
  -- Proof is left as an exercise
  sorry

end triangle_is_larger_than_quadrilateral_l409_409672


namespace cos_90_deg_eq_zero_l409_409902

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l409_409902


namespace smallest_digit_not_in_units_place_of_odd_l409_409294

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l409_409294


namespace cos_90_eq_0_l409_409876

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l409_409876


namespace distribute_gifts_l409_409721

theorem distribute_gifts :
  ∃ n, (4.choose 1 + 4.choose 2 + 4.choose 3 = n) ∧ n = 14 :=
by
  sorry

end distribute_gifts_l409_409721


namespace smallest_digit_never_at_units_place_of_odd_l409_409166

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l409_409166


namespace log_eq_exponent_eq_l409_409464

theorem log_eq_exponent_eq (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
by sorry

end log_eq_exponent_eq_l409_409464


namespace smallest_digit_not_found_in_units_place_of_odd_number_l409_409135

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l409_409135


namespace smallest_digit_not_in_odd_units_l409_409248

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l409_409248


namespace compare_log_exp_powers_l409_409492

variable (a b c : ℝ)

theorem compare_log_exp_powers (h1 : a = Real.log 0.3 / Real.log 2)
                               (h2 : b = Real.exp (Real.log 2 * 0.1))
                               (h3 : c = Real.exp (Real.log 0.2 * 1.3)) :
  a < c ∧ c < b :=
by
  sorry

end compare_log_exp_powers_l409_409492


namespace distance_between_stations_l409_409105

-- Define the time at which both trains meet
def meet_time : ℝ := 9

-- Define the speed of train A
def speed_A : ℝ := 20

-- Define the speed of train B
def speed_B : ℝ := 25

-- Define the starting times of the trains
def start_time_A : ℝ := 7
def start_time_B : ℝ := 8

-- Define the distance traveled by each train when they meet
def distance_A : ℝ := speed_A * (meet_time - start_time_A)
def distance_B : ℝ := speed_B * (meet_time - start_time_B)

-- Define the total distance between the two stations
def total_distance : ℝ := distance_A + distance_B

-- The theorem to prove the total distance between the two stations
theorem distance_between_stations : total_distance = 65 := by
  have distance_A_value : distance_A = 40 := by
    exact (by norm_num : (20 : ℝ) * (9 - 7) = 40)
  have distance_B_value : distance_B = 25 := by
    exact (by norm_num : (25 : ℝ) * (9 - 8) = 25)
  rw [distance_A_value, distance_B_value]
  exact (by norm_num : (40 : ℝ) + 25 = 65)

end distance_between_stations_l409_409105


namespace smallest_digit_not_in_units_place_of_odd_l409_409281

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l409_409281


namespace no_real_roots_poly_l409_409669

theorem no_real_roots_poly (a b c : ℝ) (h : |a| + |b| + |c| ≤ Real.sqrt 2) :
  ∀ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + 1 > 0 := 
  sorry

end no_real_roots_poly_l409_409669


namespace proof1_proof2_l409_409378

-- Definitions for conditions
variables (α : ℝ)

-- Conditions
axiom tan_alpha_eq_one_fourth : tan α = 1 / 4

-- Simplification problem
def problem1 : Prop :=
  (tan (3 * π - α) * cos (2 * π - α) * sin (-α + 3 * π / 2)) /
  (cos (-α - π) * sin (-π + α) * cos (α + 5 * π / 2)) = -1 / sin α

-- Proof problem for the given expression
theorem proof1 : problem1 := sorry

-- Value finding problem
def problem2 : Prop :=
  1 / (2 * cos(α)^2 - 3 * sin(α) * cos(α)) = 17 / 20

-- Proof problem given tan α
theorem proof2 [h : tan α = 1 / 4] : problem2 := sorry

end proof1_proof2_l409_409378


namespace sum_of_squares_of_roots_of_quadratic_l409_409502

theorem sum_of_squares_of_roots_of_quadratic :
  ( ∃ x1 x2 : ℝ, x1^2 - 3 * x1 - 1 = 0 ∧ x2^2 - 3 * x2 - 1 = 0 ∧ x1 ≠ x2) →
  x1^2 + x2^2 = 11 :=
by
  /- Proof goes here -/
  sorry

end sum_of_squares_of_roots_of_quadratic_l409_409502


namespace fried_chicken_total_l409_409626

-- The Lean 4 statement encapsulates the problem conditions and the correct answer
theorem fried_chicken_total :
  let kobe_initial := 5
  let pau_initial := 2 * kobe_initial
  let another_set := 2
  pau_initial * another_set = 20 :=
by
  let kobe_initial := 5
  let pau_initial := 2 * kobe_initial
  let another_set := 2
  show pau_initial * another_set = 20
  sorry

end fried_chicken_total_l409_409626


namespace probability_of_three_common_books_l409_409030

theorem probability_of_three_common_books
  (books : Finset ℕ)
  (Harold : Finset ℕ → Finset (Finset ℕ))
  (Betty : Finset ℕ → Finset (Finset ℕ))
  (h1 : books.card = 12)
  (h2 : ∀ (s : Finset ℕ), s ∈ Harold books → s.card = 6)
  (h3 : ∀ (s : Finset ℕ), s ∈ Betty books → s.card = 6) :
  let total_outcomes := (Nat.choose 12 6) * (Nat.choose 12 6)
  let common_selections := (Nat.choose 12 3) * (Nat.choose 9 3) * (Nat.choose 9 3) in
  (common_selections : ℚ) / (total_outcomes : ℚ) = 1552320 / 853776 :=
by
  sorry

end probability_of_three_common_books_l409_409030


namespace smallest_digit_not_in_units_place_of_odd_numbers_l409_409152

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l409_409152


namespace equation_squares_l409_409611

theorem equation_squares (a b c : ℤ) (h : (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = a ^ 2 + b ^ 2 - c ^ 2) :
  ∃ k1 k2 : ℤ, (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = k1 ^ 2 ∧ a ^ 2 + b ^ 2 - c ^ 2 = k2 ^ 2 :=
by
  sorry

end equation_squares_l409_409611


namespace sin_complementary_angle_l409_409490

theorem sin_complementary_angle :
  ∀ (α : ℝ), 
  sin (π / 4 + α) = sqrt 3 / 2 → 
  sin (3 * π / 4 - α) = sqrt 3 / 2 :=
begin
  intros α h,
  sorry
end

end sin_complementary_angle_l409_409490


namespace cos_90_eq_zero_l409_409979

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l409_409979


namespace shopping_mall_lottery_l409_409766

section
-- Definitions of the conditions
def red_ball_count : ℕ := 6
def black_ball_count : ℕ := 4
def probability_red : ℝ := (red_ball_count : ℝ) / (red_ball_count + black_ball_count)
def probability_black : ℝ := 1 - probability_red
def voucher_red_first : ℝ := 50
def voucher_black : ℝ := 25

noncomputable def E_X1 := probability_black * voucher_black + probability_red * voucher_red_first

-- Lean statement for the proof
theorem shopping_mall_lottery :
  (E_X1 = 40) ∧
  (∀ (X2 : ℝ), (X2 = 25 ∨ X2 = 50 ∨ X2 = 100) → 
    (X2 = 25 → probability_black = 0.4) ∧
    (X2 = 50 → probability_black * probability_red = 0.24) ∧
    (X2 = 100 → probability_red * probability_red = 0.36)) ∧
  (∀ n ≥ 2, (E_X1 = 40) ∧
    ∀ E (E_Xn E_Xn_1: ℝ), E_Xn = 1.2 * E_Xn_1 + 10 → 
    E_X1 + 50 = 40 + 50 ∧ (E_Xn + 50) = (1.2^(n-1)) * 90) ∧
  (∀ Xn, E (Xn: ℕ → ℝ), Xn = (1.2^(Xn - 1)) * 90 - 50 → 
    E (total_vouchers : ℝ), total_vouchers = ∑ n in {1,2,3,4,5,6}, (1.2^(n-1)) * 90 - 50 → total_vouchers = 593.7)
:= sorry
end

end shopping_mall_lottery_l409_409766


namespace cos_of_90_degrees_l409_409990

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l409_409990


namespace cos_90_deg_eq_zero_l409_409900

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l409_409900


namespace smallest_missing_digit_l409_409323

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l409_409323


namespace range_of_a2_div_a1_l409_409093

theorem range_of_a2_div_a1 (a_1 a_2 d : ℤ) : 
  1 ≤ a_1 ∧ a_1 ≤ 3 ∧ 
  a_2 = a_1 + d ∧ 
  6 ≤ 3 * a_1 + 2 * d ∧ 
  3 * a_1 + 2 * d ≤ 15 
  → (2 / 3 : ℚ) ≤ (a_2 : ℚ) / a_1 ∧ (a_2 : ℚ) / a_1 ≤ 5 :=
sorry

end range_of_a2_div_a1_l409_409093


namespace niko_profit_l409_409657

noncomputable def nikoTotalProfit (pairs_cost : ℕ) (discount : ℚ) (profit_4pairs : ℚ) (profit_each_5pairs : ℚ) (sales_tax : ℚ) : ℚ :=
  let total_cost := (9 * pairs_cost : ℚ)
  let discounted_total := total_cost * (1 - discount)
  let cost_of_4pairs := (4 * pairs_cost : ℚ)
  let profit_4pairs_amount := profit_4pairs * cost_of_4pairs
  let selling_price_4pairs := cost_of_4pairs + profit_4pairs_amount
  let sales_tax_4pairs := sales_tax * selling_price_4pairs
  let total_selling_price_4pairs := selling_price_4pairs + sales_tax_4pairs
  let profit_5pairs_amount := profit_each_5pairs * 5
  let cost_of_5pairs := (5 * pairs_cost : ℚ)
  let selling_price_5pairs := cost_of_5pairs + profit_5pairs_amount
  let sales_tax_5pairs := sales_tax * selling_price_5pairs
  let total_selling_price_5pairs := selling_price_5pairs + sales_tax_5pairs
  let total_selling_price := total_selling_price_4pairs + total_selling_price_5pairs
  let total_cost_after_discount := discounted_total
  let total_profit := total_selling_price - total_cost_after_discount
  total_profit

theorem niko_profit : nikoTotalProfit 2 0.1 0.25 0.2 0.05 = 5.85 := by
  noncomputable theory
  simp [nikoTotalProfit]
  sorry

end niko_profit_l409_409657


namespace perfect_square_proof_l409_409617

theorem perfect_square_proof (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 := 
sorry

end perfect_square_proof_l409_409617


namespace quadratic_expression_transformation_l409_409581

theorem quadratic_expression_transformation :
  ∀ (a h k : ℝ), (∀ x : ℝ, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 :=
by
  intros a h k h_eq
  sorry

end quadratic_expression_transformation_l409_409581


namespace smallest_n_l409_409344

theorem smallest_n (n : ℕ) (hn1 : (5 * n) pow 2) (hn2 : (4 * n) pow 3) : n = 80 :=
begin
  -- sorry statement to skip the proof.
  sorry
end

end smallest_n_l409_409344


namespace mary_books_end_of_year_l409_409653

def total_books_end_of_year (books_start : ℕ) (book_club : ℕ) (lent_to_jane : ℕ) 
 (returned_by_alice : ℕ) (bought_5th_month : ℕ) (bought_yard_sales : ℕ) 
 (birthday_daughter : ℕ) (birthday_mother : ℕ) (received_sister : ℕ)
 (buy_one_get_one : ℕ) (donated_charity : ℕ) (borrowed_neighbor : ℕ)
 (sold_used_store : ℕ) : ℕ :=
  books_start + book_club - lent_to_jane + returned_by_alice + bought_5th_month + bought_yard_sales +
  birthday_daughter + birthday_mother + received_sister + buy_one_get_one - donated_charity - borrowed_neighbor - sold_used_store

theorem mary_books_end_of_year : total_books_end_of_year 200 (2 * 12) 10 5 15 8 1 8 6 4 30 5 7 = 219 := by
  sorry

end mary_books_end_of_year_l409_409653


namespace sticks_per_chair_l409_409026

-- defining the necessary parameters and conditions
def sticksPerTable := 9
def sticksPerStool := 2
def sticksPerHour := 5
def chairsChopped := 18
def tablesChopped := 6
def stoolsChopped := 4
def hoursKeptWarm := 34

-- calculation of total sticks needed
def totalSticksNeeded := sticksPerHour * hoursKeptWarm

-- the main theorem to prove the number of sticks a chair makes
theorem sticks_per_chair (C : ℕ) : (chairsChopped * C) + (tablesChopped * sticksPerTable) + (stoolsChopped * sticksPerStool) = totalSticksNeeded → C = 6 := by
  sorry

end sticks_per_chair_l409_409026


namespace cos_90_eq_0_l409_409879

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l409_409879


namespace smallest_n_perfect_square_and_cube_l409_409354

theorem smallest_n_perfect_square_and_cube (n : ℕ) (h1 : ∃ k : ℕ, 5 * n = k^2) (h2 : ∃ m : ℕ, 4 * n = m^3) :
  n = 1080 :=
  sorry

end smallest_n_perfect_square_and_cube_l409_409354


namespace smallest_digit_not_in_odd_units_l409_409218

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l409_409218


namespace smallest_digit_never_at_units_place_of_odd_l409_409167

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l409_409167


namespace cos_90_eq_zero_l409_409981

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l409_409981


namespace minimum_value_l409_409513

def minimum_value_problem (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_sum : a + b = 2) : Prop :=
  ∃ c : ℝ, c = (1 / (a + 1) + 4 / (b + 1)) ∧ c = 9 / 4

theorem minimum_value (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_sum : a + b = 2) : 
  (1 / (a + 1) + 4 / (b + 1)) = 9 / 4 :=
by 
  -- Proof goes here
  sorry

end minimum_value_l409_409513


namespace smallest_digit_not_in_odd_units_l409_409242

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l409_409242


namespace not_prime_quadratic_roots_l409_409675

theorem not_prime_quadratic_roots (a b : ℤ) (h : ∃ p q : ℕ, p > 0 ∧ q > 0 ∧ (p + q = -a) ∧ (p * q = b + 1)) : ¬Nat.Prime (a^2 + b^2) :=
by
  sorry

end not_prime_quadratic_roots_l409_409675


namespace arithmetic_sequence_ratio_l409_409485

theorem arithmetic_sequence_ratio (S T : ℕ → ℕ) (a b : ℕ → ℕ)
  (h : ∀ n, S n / T n = (7 * n + 3) / (n + 3)) :
  a 8 / b 8 = 6 :=
by
  sorry

end arithmetic_sequence_ratio_l409_409485


namespace yuna_has_biggest_number_l409_409703

-- Define the numbers assigned to each student
def Yoongi_num : ℕ := 7
def Jungkook_num : ℕ := 6
def Yuna_num : ℕ := 9
def Yoojung_num : ℕ := 8

-- State the main theorem that Yuna has the biggest number
theorem yuna_has_biggest_number : 
  (Yuna_num = 9) ∧ (Yuna_num > Yoongi_num) ∧ (Yuna_num > Jungkook_num) ∧ (Yuna_num > Yoojung_num) :=
sorry

end yuna_has_biggest_number_l409_409703


namespace smallest_digit_not_in_units_place_of_odd_l409_409283

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l409_409283


namespace smallest_digit_not_in_units_place_of_odd_l409_409287

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l409_409287


namespace smallest_digit_not_found_in_units_place_of_odd_number_l409_409137

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l409_409137


namespace inequality_abc_l409_409055

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 2 * b + 3 * c) ^ 2 / (a ^ 2 + 2 * b ^ 2 + 3 * c ^ 2) ≤ 6 :=
sorry

end inequality_abc_l409_409055


namespace find_C_D_l409_409478

theorem find_C_D : ∃ C D, 
  (∀ x, x ≠ 3 → x ≠ 5 → (6*x - 3) / (x^2 - 8*x + 15) = C / (x - 3) + D / (x - 5)) ∧ 
  C = -15/2 ∧ D = 27/2 := by
  sorry

end find_C_D_l409_409478


namespace smallest_unfound_digit_in_odd_units_l409_409234

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l409_409234


namespace bug_at_A_after_7_meters_l409_409633

def P : ℕ → ℚ
| 0       := 1
| (n + 1) := (1 - P n) / 3

theorem bug_at_A_after_7_meters : 
  ∃ n : ℕ, P 7 = n / 729 ∧ n = 182 :=
by 
  use 182
  simp [P]
  sorry

end bug_at_A_after_7_meters_l409_409633


namespace cos_of_90_degrees_l409_409987

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l409_409987


namespace number_permutations_divisible_by_37_l409_409795

theorem number_permutations_divisible_by_37 (N : ℕ) (hN_digits : ∀ n m : ℕ, n ≠ m → ∃ k : ℕ, k ≠ 0 ∧ k < 10 ∧ 
  ∃ (a b c d e f : ℕ), N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f ∧ ∀ x : ℕ, x ∈ {a, b, c, d, e, f} → 1 ≤ x ∧ x ≤ 9)
  (hN_div_37 : (N % 37 = 0)) : ∃ (S : Finset ℕ), S.card ≥ 23 ∧ ∀ M ∈ S, M % 37 = 0 := 
by 
  sorry

end number_permutations_divisible_by_37_l409_409795


namespace transformed_function_l409_409081

-- Definitions for operations on functions
def shift_right (f : ℝ → ℝ) (a : ℝ) := λ x, f (x - a)
def shift_up (f : ℝ → ℝ) (b : ℝ) := λ x, f x + b

-- Original function
def f (x : ℝ) := x^2

-- Transformed function
def g := shift_up (shift_right f 3) 4

-- Proposition to prove
theorem transformed_function :
  g = λ x, (x - 3)^2 + 4 :=
by { ext, simp [g, shift_right, shift_up, f], sorry }

end transformed_function_l409_409081


namespace positive_divisors_n_l409_409651

theorem positive_divisors_n (n : ℕ) (d : ℕ → ℕ)
  (h₁ : d 1 = 1)
  (h₂ : d 10 = n)
  (h₃ : ∀ i, 1 ≤ i ∧ i ≤ 10 → n % (d i) = 0)
  (h₄ : ∀ i, 1 ≤ i ∧ i ≤ 10 → (∃! j, d j = n / (d i)))
  (h₅ : ∃ m : ℕ, m ≠ 5 ∧ m ≠ 6 ∧ d m < d 5 ∧ d 6 < d 10 ∧ ¬(d 5 = d 6 ∨ d 6 = d 5 + 1))
: n = 272 :=
begin
  -- To be proved
  sorry
end

end positive_divisors_n_l409_409651


namespace sum_sequence_inequality_l409_409542

noncomputable def sequence_a : ℕ → ℕ
| 0       := 0   -- Since sequences typically start from 1, we define a_0 for formality
| (n + 1) := if n = 0 then 1 else 2 * sequence_a n

def S (n : ℕ) : ℕ := 2 * sequence_a n - 1

theorem sum_sequence_inequality :
  {n : ℕ | 1 ≤ n ∧ (sequence_a n / n) ≤ 2} = {1, 2, 3, 4} :=
by
  sorry

end sum_sequence_inequality_l409_409542


namespace smallest_missing_digit_l409_409318

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l409_409318


namespace max_third_side_proof_l409_409067

noncomputable def max_third_side (D E F : ℝ) (s1 s2 : ℝ) (cond1 : cos (2 * D) + cos (2 * E) + cos (2 * F) = 1 / 2) (cond2 : s1 = 12) (cond3 : s2 = 15) : ℝ :=
  if E = real.acos (1 / 2) then ((s1 ^ 2 + s2 ^ 2 + 2 * s1 * s2) ^ (1 / 2))
  else sorry

theorem max_third_side_proof : ∀ D E F : ℝ, cos (2 * D) + cos (2 * E) + cos (2 * F) = 1 / 2 → 12 = 12 → 15 = 15 →
  max_third_side D E F 12 15 = real.sqrt 549 :=
sorry

end max_third_side_proof_l409_409067


namespace find_third_polygon_sides_l409_409108

def interior_angle (n : ℕ) : ℚ :=
  (n - 2) * 180 / n

theorem find_third_polygon_sides :
  let square_angle := interior_angle 4
  let pentagon_angle := interior_angle 5
  let third_polygon_angle := 360 - (square_angle + pentagon_angle)
  ∃ (m : ℕ), interior_angle m = third_polygon_angle ∧ m = 20 :=
by
  let square_angle := interior_angle 4
  let pentagon_angle := interior_angle 5
  let third_polygon_angle := 360 - (square_angle + pentagon_angle)
  use 20
  sorry

end find_third_polygon_sides_l409_409108


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409341

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409341


namespace zhang_bing_age_18_l409_409748

theorem zhang_bing_age_18 {x a : ℕ} (h1 : x < 2023) 
  (h2 : a = x - 1953)
  (h3 : a % 9 = 0)
  (h4 : a = (x % 10) + ((x / 10) % 10) + ((x / 100) % 10) + ((x / 1000) % 10)) :
  a = 18 :=
sorry

end zhang_bing_age_18_l409_409748


namespace equilateral_triangle_area_l409_409071

theorem equilateral_triangle_area (h : Real) (h_eq : h = Real.sqrt 12):
  ∃ A : Real, A = 12 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_l409_409071


namespace smallest_digit_not_in_units_place_of_odd_l409_409269

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l409_409269


namespace smallest_visible_sum_l409_409386

theorem smallest_visible_sum (cube : Array (Fin 27) (Array (Fin 6) Nat))
  (h1 : ∀ die : Fin 27, (∀ face : Fin 6, cube[die][face] + cube[die][5 - face] = 7))
  (h2 : let visible_faces (position : Fin 27) : Nat := -- calculate the sum of visible faces based on the position relative to diagonal slice
          if (/* corner condition */) then sum_of_first_three_faces
          else if (/* edge condition */) then sum_of_first_two_faces
          else if (/* normal face-center condition */) then sum_of_first_face
          else if (/* sliced face-center condition */) then sum_of_first_two_faces_including_sliced_face
          else 0,
        List.sum (List.map visible_faces (List.finRange 27)) = 98) :
  List.sum (List.map visible_faces (List.finRange 27)) = 98 := 
  by
    sorry

end smallest_visible_sum_l409_409386


namespace compare_fractions_l409_409842

theorem compare_fractions : (-1 / 3) > (-1 / 2) := sorry

end compare_fractions_l409_409842


namespace lines_perpendicular_to_plane_are_parallel_l409_409362

variables (α : Type*) [plane α] (l₁ l₂ : line α) 

def perpendicular_to_plane (l : line α) (α : plane α) :=
  ∀ p : point α, p ∈ α → is_orthogonal l p

theorem lines_perpendicular_to_plane_are_parallel
  (h1 : perpendicular_to_plane l₁ α)
  (h2 : perpendicular_to_plane l₂ α) :
  is_parallel l₁ l₂ :=
by
  sorry

end lines_perpendicular_to_plane_are_parallel_l409_409362


namespace part_one_part_two_l409_409638

theorem part_one (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
  ab + bc + ca ≤ 1 / 3 := sorry

theorem part_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) : 
  a^2 / b + b^2 / c + c^2 / a ≥ 1 := sorry

end part_one_part_two_l409_409638


namespace max_positive_integers_l409_409660

theorem max_positive_integers (f : Fin 2018 → ℤ) (h : ∀ i : Fin 2018, f i > f (i - 1) + f (i - 2)) : 
  ∃ n: ℕ, n = 2016 ∧ (∀ i : ℕ, i < 2018 → f i > 0) ∧ (∀ i : ℕ, i < 2 → f i < 0) := 
sorry

end max_positive_integers_l409_409660


namespace covariance_eq_variance_inequality_l409_409018

noncomputable section

open ProbabilityTheory

variable {Ω : Type*} {X Y : Ω → ℝ}

-- Assume X and Y have finite second-order moments
axiom finite_second_order_moments (X Y : Ω → ℝ) : integrable (λ ω, X ω ^ 2) ∧ integrable (λ ω, Y ω ^ 2)

-- Covariance definition
def covariance (X Y : Ω → ℝ) : ℝ := 
  ∫ ω, X ω * Y ω ∂μ - (∫ ω, X ω ∂μ) * (∫ ω, Y ω ∂μ)

-- Expectation with respect to X
def E_Y_given_X (X Y : Ω → ℝ) : Ω → ℝ := λ ω, ∫ y, y ∂(cond_prob X Y ω)

-- Variance definition
def variance (X : Ω → ℝ) : ℝ := ∫ ω, X ω ^ 2 ∂μ - (∫ ω, X ω ∂μ) ^ 2

-- The main theorem
theorem covariance_eq {X Y : Ω → ℝ} (hX : integrable (λ ω, X ω ^ 2)) (hY : integrable (λ ω, Y ω ^ 2)) :
  covariance X Y = covariance X (E_Y_given_X X Y) :=
sorry

theorem variance_inequality {X Y : Ω → ℝ} (hX : integrable (λ ω, X ω ^ 2)) (hY : integrable (λ ω, Y ω ^ 2))
  (hE_cond : ∀ ω, E_Y_given_X X Y ω = 1) :
  variance X ≤ variance (λ ω, X ω * Y ω) :=
sorry

end covariance_eq_variance_inequality_l409_409018


namespace sum_of_numbers_l409_409715

-- Definitions that come directly from the conditions
def product_condition (A B : ℕ) : Prop := A * B = 9375
def quotient_condition (A B : ℕ) : Prop := A / B = 15

-- Theorem that proves the sum of A and B is 400, based on the given conditions
theorem sum_of_numbers (A B : ℕ) (h1 : product_condition A B) (h2 : quotient_condition A B) : A + B = 400 :=
sorry

end sum_of_numbers_l409_409715


namespace volume_of_cube_l409_409775

theorem volume_of_cube (A : ℝ) (s V : ℝ) 
  (hA : A = 150) 
  (h_surface_area : A = 6 * s^2) 
  (h_side_length : s = 5) :
  V = s^3 →
  V = 125 :=
by
  sorry

end volume_of_cube_l409_409775


namespace tangent_circle_problem_l409_409393

noncomputable def solve_problem (s : ℝ) (θ : ℝ) : ℝ :=
  2 / (1 + s)

theorem tangent_circle_problem
  (O A B D : Type)
  (radius : ℝ) (s : ℝ) (θ : ℝ)
  (h1 : radius = 2)
  (h2 : ∀ {P Q R : Type}, tangent P Q R = true → ∠ Q O R = θ)
  (h3 : ∃ C : Type, lies_on C (line O A))
  (h4 : Lies_on D (line O A) ∧ ∃ B : Type, angle_bisector B D (angle B A O)) :
  solve_problem s θ = 2 / (1 + s) :=
by
  sorry

end tangent_circle_problem_l409_409393


namespace pie_chart_degrees_for_cherry_pie_l409_409591

theorem pie_chart_degrees_for_cherry_pie :
  ∀ (total_students chocolate_pie apple_pie blueberry_pie : ℕ)
    (remaining_students cherry_pie_students lemon_pie_students : ℕ),
    total_students = 40 →
    chocolate_pie = 15 →
    apple_pie = 10 →
    blueberry_pie = 7 →
    remaining_students = total_students - chocolate_pie - apple_pie - blueberry_pie →
    cherry_pie_students = remaining_students / 2 →
    lemon_pie_students = remaining_students / 2 →
    (cherry_pie_students : ℝ) / (total_students : ℝ) * 360 = 36 :=
by
  sorry

end pie_chart_degrees_for_cherry_pie_l409_409591


namespace average_age_of_women_l409_409369

theorem average_age_of_women (A : ℕ) :
  (6 * (A + 2) = 6 * A - 22 + W) → (W / 2 = 17) :=
by
  intro h
  sorry

end average_age_of_women_l409_409369


namespace cos_90_eq_zero_l409_409945

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l409_409945


namespace cos_90_eq_0_l409_409957

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l409_409957


namespace convert_88_to_base5_l409_409426

theorem convert_88_to_base5 :
  ∃ (digits : List ℕ), digits = [3, 2, 3] ∧ (88 : ℕ) = digits.reverse.foldl (λ acc d, acc * 5 + d) 0 :=
by 
  sorry

end convert_88_to_base5_l409_409426


namespace cos_90_eq_0_l409_409897

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l409_409897


namespace smallest_digit_not_in_odd_units_l409_409240

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l409_409240


namespace smallest_not_odd_unit_is_zero_l409_409183

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l409_409183


namespace find_x_l409_409467

theorem find_x (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end find_x_l409_409467


namespace henrietta_paint_needed_l409_409551

theorem henrietta_paint_needed :
  let living_room_area := 600
  let num_bedrooms := 3
  let bedroom_area := 400
  let paint_coverage_per_gallon := 600
  let total_area := living_room_area + (num_bedrooms * bedroom_area)
  total_area / paint_coverage_per_gallon = 3 :=
by
  -- Proof should be completed here.
  sorry

end henrietta_paint_needed_l409_409551


namespace base10_to_base6_l409_409110

theorem base10_to_base6 (n : ℕ) (h : n = 300) : nat.to_digits 6 n = [1, 2, 2, 0] :=
by {
  subst h,
  have h₁ : n / 216 = 1, by norm_num,
  have h₂ : n % 216 = 84, by norm_num,
  have h₃ : 84 / 36 = 2, by norm_num,
  have h₄ : 84 % 36 = 12, by norm_num,
  have h₅ : 12 / 6 = 2, by norm_num,
  have h₆ : 12 % 6 = 0, by norm_num,
  rw [← nat.digits_def, nat.to_digits, h₁, h₂, h₃, h₄, h₅, h₆],
  exact rfl,
}

end base10_to_base6_l409_409110


namespace min_value_expr_part_II_inequality_l409_409491

noncomputable theory

-- Part (I): Verify the minimum value of the given function expression.
theorem min_value_expr {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ a b, (\(a + b = 2)) → \(a > 0) → \(b > 0) → \(\frac{1}{1+a} + \frac{4}{1+b}) ≕ \(\frac{9}{4})) := sorry

-- Part (II): Prove the given inequality.
theorem part_II_inequality {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (a^2 * b^2 + a^2 + b^2 ≥ a * b * (a + b + 1)) := sorry

end min_value_expr_part_II_inequality_l409_409491


namespace sum_diff_eq_2003_l409_409113

section
variable (n : ℕ) (E O : ℕ → ℕ)
variable (S : ℕ → ℕ × ℕ → ℕ)

def first_n_odds (n : ℕ) : ℕ → ℕ
| 0 => 1
| k+1 => 2*k + 1

def first_n_evens (n : ℕ) : ℕ → ℕ
| 0 => 2
| k+1 => 2*(k+1)

noncomputable def sum_of_sequence (n sum_func: ℕ) : ℕ :=
(n / 2) * (sum_func(0) + sum_func(n))

theorem sum_diff_eq_2003 : (n : ℕ) → sum_of_sequence n first_n_evens - sum_of_sequence n first_n_odds = 2003 :=
by
  intro n
  have O : ∀ i, i < n → first_n_odds i = 2*i + 1 := sorry
  have E : ∀ i, i < n → first_n_evens i = 2*(i+1) := sorry
  calc
    sum_of_sequence n first_n_evens - sum_of_sequence n first_n_odds
    = (n / 2) * (2 + 2*n) - (n / 2) * (1 + 2*n - 1) : by simp [sum_of_sequence, first_n_evens, first_n_odds, O, E, sorry]
    ... = 2003 : sorry
end

end sum_diff_eq_2003_l409_409113


namespace ratio_area_to_perimeter_height_verification_l409_409117

noncomputable def perimeter (a : ℝ) := 3 * a

noncomputable def height (a : ℝ) := (real.sqrt 3 / 2) * a

noncomputable def area (a : ℝ) := 1 / 2 * a * height a

noncomputable def heron_area (a : ℝ) :=
  let s := perimeter a / 2 in
  real.sqrt (s * (s - a) * (s - a) * (s - a))

noncomputable def heron_height (a : ℝ) :=
  2 * heron_area a / a

theorem ratio_area_to_perimeter (a : ℝ) (h_geometric : height a = 5 * real.sqrt 3)
  (heron_height_eq : heron_height a = height a) : 
  area a / perimeter a = 5 * real.sqrt 3 / 6 :=
by
  sorry

theorem height_verification (a : ℝ) :
  height a = 5 * real.sqrt 3 → heron_height a = height a :=
by
  sorry

end ratio_area_to_perimeter_height_verification_l409_409117


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409332

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409332


namespace increase_in_radius_l409_409661

-- Definitions corresponding to the problem conditions
def initial_odometer_reading : ℝ := 390
def return_odometer_reading : ℝ := 380
def original_radius_in_inches : ℝ := 12
def miles_to_inches : ℝ := 63360

-- Theorem stating the final result based on the conditions
theorem increase_in_radius : 
  let original_circumference := 2 * Real.pi * original_radius_in_inches
      original_distance_per_rotation := original_circumference / miles_to_inches
      number_of_rotations := initial_odometer_reading / original_distance_per_rotation
      return_trip_miles := initial_odometer_reading
      new_number_of_rotations := return_trip_miles / original_distance_per_rotation
      new_radius := (return_trip_miles * original_distance_per_rotation * miles_to_inches) / (2 * Real.pi * return_odometer_reading)
      radius_increase := new_radius - original_radius_in_inches
  in (Float.cmp radius_increase 0.27 <=> Ordering.eq)
:=
begin
  sorry
end

end increase_in_radius_l409_409661


namespace striped_octopus_has_8_legs_l409_409572

-- Definitions for Octopus and Statements
structure Octopus :=
  (legs : ℕ)
  (tellsTruth : Prop)

-- Given conditions translations
def tellsTruthCondition (o : Octopus) : Prop :=
  if o.legs % 2 = 0 then o.tellsTruth else ¬o.tellsTruth

def green_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

def dark_blue_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

def violet_octopus : Octopus :=
  { legs := 9, tellsTruth := sorry }  -- Placeholder truth value

def striped_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

-- Octopus statements (simplified for output purposes)
def green_statement := (green_octopus.legs = 8) ∧ (dark_blue_octopus.legs = 6)
def dark_blue_statement := (dark_blue_octopus.legs = 8) ∧ (green_octopus.legs = 7)
def violet_statement := (dark_blue_octopus.legs = 8) ∧ (violet_octopus.legs = 9)
def striped_statement := ¬(green_octopus.legs = 8 ∨ dark_blue_octopus.legs = 8 ∨ violet_octopus.legs = 8) ∧ (striped_octopus.legs = 8)

-- The goal to prove that the striped octopus has exactly 8 legs
theorem striped_octopus_has_8_legs : striped_octopus.legs = 8 :=
sorry

end striped_octopus_has_8_legs_l409_409572


namespace force_with_18_inch_crowbar_l409_409078

noncomputable def inverseForce (L F : ℝ) : ℝ :=
  F * L

theorem force_with_18_inch_crowbar :
  ∀ (F : ℝ), (inverseForce 12 200 = inverseForce 18 F) → F = 133.333333 :=
by
  intros
  sorry

end force_with_18_inch_crowbar_l409_409078


namespace cos_90_eq_zero_l409_409978

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l409_409978


namespace area_equivalence_l409_409084

noncomputable def area_of_plane_quadrilateral (a : ℝ) : ℝ := 2 * real.sqrt 2 * a^2

theorem area_equivalence (a S S' : ℝ) (h1 : S' = a^2) (h2 : S' = real.sqrt 2 / 4 * S) : 
  S = 2 * real.sqrt 2 * a^2 :=
by
  sorry

end area_equivalence_l409_409084


namespace smaller_of_two_digit_numbers_whose_product_is_4814_l409_409714

theorem smaller_of_two_digit_numbers_whose_product_is_4814 :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4814 ∧ min a b = 53 :=
begin
  sorry
end

end smaller_of_two_digit_numbers_whose_product_is_4814_l409_409714


namespace peaches_picked_l409_409654

variable (o t : ℕ)
variable (p : ℕ)

theorem peaches_picked : (o = 34) → (t = 86) → (t = o + p) → p = 52 :=
by
  intros ho ht htot
  rw [ho, ht] at htot
  sorry

end peaches_picked_l409_409654


namespace total_stamps_is_38_l409_409818

-- Definitions based directly on conditions
def snowflake_stamps := 11
def truck_stamps := snowflake_stamps + 9
def rose_stamps := truck_stamps - 13
def total_stamps := snowflake_stamps + truck_stamps + rose_stamps

-- Statement to be proved
theorem total_stamps_is_38 : total_stamps = 38 := 
by 
  sorry

end total_stamps_is_38_l409_409818


namespace exam_score_l409_409717

theorem exam_score (score : ℕ) : 
  (∀ t, score = 90 * t / 5) → (score = 126) :=
by
  intro h
  have h1 : score = 90 * 7 / 5 := h 7
  rw [h1]
  norm_num
  sorry

end exam_score_l409_409717


namespace max_no_root_quadratics_l409_409726

/-
Given nine quadratic polynomials \( P_1(x), P_2(x), \ldots, P_9(x) \), where each polynomial is of the form 
\( x^2 + a_i x + b_i \) with \( a_i \) and \( b_i \) forming arithmetic progressions, and that their sum 
has at least one root, prove that the maximum number of these polynomials that can have no roots is \( 4 \).
-/

theorem max_no_root_quadratics (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = a n + d₁)
  (h2 : ∀ n, b (n + 1) = b n + d₂)
  (h3 : ∃ x, (∑ i in finset.range 9, x^2 + a i * x + b i) = 0) :
  ∃ n, ∑ k in finset.range 9, bool.to_nat (¬ (x^2 + a k * x + b k = 0)) = 4 :=
sorry

end max_no_root_quadratics_l409_409726


namespace minimal_partitions_to_remove_l409_409776

theorem minimal_partitions_to_remove (n : ℕ) (h : n ≥ 3) :
  ∃ m, m = (n-2) * (n-2) * (n-2) ∧ 
        ∀ unit_cube, ∃ partition_scheme, (removal_of_partition partition_scheme).connected_to_boundary unit_cube :=
sorry

end minimal_partitions_to_remove_l409_409776


namespace gain_percent_correct_l409_409367

def CostPrice : ℝ := 900
def SellingPrice : ℝ := 1100
def Gain : ℝ := SellingPrice - CostPrice
def GainPercent : ℝ := (Gain / CostPrice) * 100

theorem gain_percent_correct :
  GainPercent = 22.22 := 
by sorry

end gain_percent_correct_l409_409367


namespace cone_volume_l409_409086

theorem cone_volume (R : ℝ) (x : ℝ) (h₀ : x = 2 * π * R / (π ^ 2 - 1))
  (h₁ : ∀ S_base S_lateral S_axial,
      S_base = π * R ^ 2 →
      S_axial = R * x →
      S_lateral = π * R * real.sqrt (x ^ 2 + R ^ 2) →
      S_lateral = S_base + S_axial) : 
  ∃ V, V = (2 * π ^ 2 * R ^ 3) / (3 * (π ^ 2 - 1)) :=
by
  let S_base := π * R ^ 2
  let S_axial := R * x
  let S_lateral := π * R * real.sqrt (x ^ 2 + R ^ 2)
  have h2 : S_lateral = S_base + S_axial, from h₁ S_base S_lateral S_axial rfl rfl rfl
  let V := (1 / 3) * S_base * x
  use V
  have h3 : x = 2 * π * R / (π ^ 2 - 1), from h₀
  sorry -- Detailed proof steps omitted

end cone_volume_l409_409086


namespace smallest_missing_digit_l409_409317

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l409_409317


namespace find_x_value_l409_409457

theorem find_x_value (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end find_x_value_l409_409457


namespace max_plus_min_of_f_eq_two_l409_409088

noncomputable def f (x : ℝ) : ℝ := 
  (sqrt 2 * sin(x + π / 4) + 2 * x^2 + x) / (2 * x^2 + cos x)

theorem max_plus_min_of_f_eq_two : 
  let M := reals.sup (set.range f)
  let m := reals.inf (set.range f)
  M + m = 2 :=
sorry

end max_plus_min_of_f_eq_two_l409_409088


namespace randi_more_nickels_l409_409047

noncomputable def more_nickels (total_cents : ℕ) (to_peter_cents : ℕ) (to_randi_cents : ℕ) : ℕ := 
  (to_randi_cents / 5) - (to_peter_cents / 5)

theorem randi_more_nickels :
  ∀ (total_cents to_peter_cents : ℕ),
  total_cents = 175 →
  to_peter_cents = 30 →
  more_nickels total_cents to_peter_cents (2 * to_peter_cents) = 6 :=
by
  intros total_cents to_peter_cents h_total h_peter
  rw [h_total, h_peter]
  unfold more_nickels
  norm_num
  sorry

end randi_more_nickels_l409_409047


namespace cos_90_eq_0_l409_409963

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l409_409963


namespace proof_x_bounds_l409_409014

noncomputable def x : ℝ :=
  1 / Real.logb (1 / 3) (1 / 2) +
  1 / Real.logb (1 / 3) (1 / 4) +
  1 / Real.logb 7 (1 / 8)

theorem proof_x_bounds : 3 < x ∧ x < 3.5 := 
by
  sorry

end proof_x_bounds_l409_409014


namespace smallest_digit_never_in_units_place_of_odd_numbers_l409_409301

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l409_409301


namespace smallest_digit_never_at_units_place_of_odd_l409_409174

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l409_409174


namespace compare_neg_thirds_and_halves_l409_409840

theorem compare_neg_thirds_and_halves : (-1 : ℚ) / 3 > (-1 : ℚ) / 2 :=
by
  sorry

end compare_neg_thirds_and_halves_l409_409840


namespace tiling_impossible_l409_409044

theorem tiling_impossible (tiles_4x1_count : ℕ) (tile_2x2_count : ℕ) (HW_board : ℕ) :
  tiles_4x1_count = 15 → tile_2x2_count = 1 → HW_board = 8 → 
  ¬ (exists (tiling : fin HW_board × fin HW_board → bool), 
      (∀ i j, tiling (⌊i / 4⌋, ⌊j / 4⌋) = tiling (⌊i  + 4* tiles_4x1_count /  HW_board⌋,  ⌊j + 4 * tile_2x2_count / HW_board⌋))) :=
by
  intros h1 h2 h3
  sorry

end tiling_impossible_l409_409044


namespace binomial_sum_value_l409_409477

theorem binomial_sum_value :
  (Finset.range 50).sum (λ k, (-1)^k * Nat.choose 99 (2*k)) = -2^49 :=
by
  sorry

end binomial_sum_value_l409_409477


namespace chord_length_l409_409771

theorem chord_length (r d : ℝ) (h_r : r = 5) (h_d : d = 4) : ∃ EF : ℝ, EF = 6 :=
by
  let OF := 5
  let OG := 4
  have h_OG_OF : OG^2 + (EF / 2)^2 = OF^2 := sorry -- Pythagorean relationship
  have GF := 3
  use 2 * GF
  sorry

end chord_length_l409_409771


namespace cos_ninety_degrees_l409_409918

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l409_409918


namespace min_abs_sum_of_matrix_square_eq_l409_409001

theorem min_abs_sum_of_matrix_square_eq :
  ∃ (a b c d : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (let m := matrix ![![a, b], ![c, d]] in m⬝m = (11 : ℤ) • 1) ∧
    ∀ a' b' c' d' : ℤ, a' ≠ 0 ∧ b' ≠ 0 ∧ c' ≠ 0 ∧ d' ≠ 0 ∧
      (let m' := matrix ![![a', b'], ![c', d']] in m'⬝m' = (11 : ℤ) • 1) →
      |a| + |b| + |c| + |d| ≤ |a'| + |b'| + |c'| + |d'| :=
sorry

end min_abs_sum_of_matrix_square_eq_l409_409001


namespace smallest_n_l409_409347

theorem smallest_n (n : ℕ) (h₁ : ∃ k₁ : ℕ, 5 * n = k₁ ^ 2) (h₂ : ∃ k₂ : ℕ, 4 * n = k₂ ^ 3) : n = 1600 :=
sorry

end smallest_n_l409_409347


namespace sum_difference_even_odd_2003_l409_409112

theorem sum_difference_even_odd_2003 :
  let S_odd := (λ n, ∑ k in Finset.range n, (2 * k + 1))
  let S_even := (λ n, ∑ k in Finset.range n, (2 * k + 2))
  S_even 2003 - S_odd 2003 = 2003 :=
by
  let S_odd := (λ n, ∑ k in Finset.range n, (2 * k + 1))
  let S_even := (λ n, ∑ k in Finset.range n, (2 * k + 2))
  show S_even 2003 - S_odd 2003 = 2003
  sorry

end sum_difference_even_odd_2003_l409_409112


namespace unique_five_digit_numbers_l409_409053

-- Definitions for odd and even number sets
def odd_numbers := {1, 3, 5, 7, 9}
def even_numbers := {0, 2, 4, 6, 8}
def even_numbers_non_zero := {2, 4, 6, 8}

-- Definitions of combinations and arrangements
noncomputable def C (n k : Nat) : Nat := Nat.choose n k
noncomputable def A (n k : Nat) : Nat := Nat.factorial n / Nat.factorial (n - k)

-- Define the problem
theorem unique_five_digit_numbers : 
  let num_odd_combinations := C 5 2,
      num_even_combinations := C 4 3,
      num_with_zero_combinations := C 4 2 * C 4 1 * A 4 4,
      total_ways := num_odd_combinations * (num_even_combinations * A 5 5 + num_with_zero_combinations)
  in total_ways = 10560 := 
by
  sorry

end unique_five_digit_numbers_l409_409053


namespace part1_part2_l409_409509

-- Defining the triangle and the given condition
def triangle := {a b c : ℝ} -- sides of the triangle
def angles := {A B C : ℝ} -- angles of the triangle

axiom sides_opposite_angles (a b c : ℝ) (A B C : ℝ) : 
  a = 1  → 
  3 * sin A * cos B = 3 * sin C - sin B

-- First statement: \cos \frac{B+C}{2} = \frac{\sqrt{3}}{3}
theorem part1 (a b c A B C : ℝ) (h : a = 1) (cond : 3 * sin A * cos B = 3 * sin C - sin B) :
  cos ((B + C) / 2) = sqrt 3 / 3 :=
sorry

-- Second statement: maximum value of perimeter l = 1 + sqrt 2 when a = 1
theorem part2 (b c : ℝ) :
  ∃ (l : ℝ), l = 1 + sqrt 2 :=
sorry

end part1_part2_l409_409509


namespace cos_90_deg_eq_zero_l409_409905

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l409_409905


namespace find_D_E_F_l409_409704

def g (x D E F : ℝ) : ℝ := x^2 / (D * x^2 + E * x + F)

theorem find_D_E_F :
  (∀ x > 5, g x 2 (-2) (-24) > 0.5) ∧ 
  (D * (x + 3) * (x - 4) = D * x^2 - D * x - 12 * D) ∧
  (1/D > 0.5) ∧ (1/D < 1) ∧ 
  (D + E + F = -24) := 
sorry

end find_D_E_F_l409_409704


namespace smallest_unfound_digit_in_odd_units_l409_409228

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l409_409228


namespace cos_pi_half_eq_zero_l409_409862

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l409_409862


namespace percentage_less_than_l409_409786

theorem percentage_less_than (x y : ℝ) (h : x = 8 * y) : ((x - y) / x) * 100 = 87.5 := 
by sorry

end percentage_less_than_l409_409786


namespace counterexample_complement_not_greater_l409_409674

theorem counterexample_complement_not_greater :
  ¬ (80 < 180 - 80) :=
by {
  simp,
  linarith,
}

end counterexample_complement_not_greater_l409_409674


namespace cos_90_eq_zero_l409_409976

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l409_409976


namespace catch_up_time_l409_409389

noncomputable def velocity_A (t : ℝ) : ℝ := 3 * t ^ 2 + 1
noncomputable def velocity_B (t : ℝ) : ℝ := 10 * t

noncomputable def distance_A (t : ℝ) : ℝ := ∫ τ in 0..t, velocity_A τ
noncomputable def distance_B (t : ℝ) : ℝ := ∫ τ in 0..t, velocity_B τ

theorem catch_up_time :
  ∃ t : ℝ, distance_A t = distance_B t + 5 ∧ t = 5 :=
by
  sorry

end catch_up_time_l409_409389


namespace find_multiplier_l409_409701

theorem find_multiplier (n x : ℝ) (h1 : n = 1.0) (h2 : 3 * n - 1 = x * n) : x = 2 :=
by
  sorry

end find_multiplier_l409_409701


namespace smallest_digit_not_in_units_place_of_odd_l409_409291

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l409_409291


namespace smallest_digit_not_in_units_place_of_odd_l409_409270

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l409_409270


namespace max_true_statements_l409_409005

def statement1 (x : ℝ) : Prop := 0 < x^2 ∧ x^2 < 1
def statement2 (x : ℝ) : Prop := x^2 > 1
def statement3 (x : ℝ) : Prop := -1 < x ∧ x < 0
def statement4 (x : ℝ) : Prop := 0 < x ∧ x < 1
def statement5 (x : ℝ) : Prop := 0 < x - real.sqrt x ∧ x - real.sqrt x < 1

theorem max_true_statements : ∀ x : ℝ, cardinal.mk { 1 | statement1 x, 2 | statement2 x, 3 | statement3 x, 4 | statement4 x, 5 | statement5 x}.filter (λ s, s) ≤ 3 := by
  sorry

end max_true_statements_l409_409005


namespace cos_90_eq_0_l409_409894

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l409_409894


namespace max_distance_between_sparkling_points_l409_409789

theorem max_distance_between_sparkling_points (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : a₁^2 + b₁^2 = 1) (h₂ : a₂^2 + b₂^2 = 1) :
  ∃ d, d = 2 ∧ ∀ (x y : ℝ), x = a₂ - a₁ ∧ y = b₂ - b₁ → (x ^ 2 + y ^ 2 = d ^ 2) :=
by
  sorry

end max_distance_between_sparkling_points_l409_409789


namespace find_k_range_l409_409526

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then sqrt x + 1
else if x < 0 then - sqrt (-x) - 1
else 0

lemma f_is_odd (x : ℝ) : f (-x) = -f x :=
by {
  have h_pos : ∀ x, x > 0 → f (-x) = -f x, {
    intros x hx,
    rw [f, if_pos hx, f, if_neg (lt_neg_iff_add_one.2 hx)],
    simp only [sqrt_neg_eq'_iff_le_zero.2 hx] at *,
    rw eq_neg_iff_add_eq_zero,
    ring
  },
  have h_zero : f 0 = 0 := by rw [f, if_neg (lt_irrefl 0), if_neg (not_lt_of_le (le_refl 0))],
  have h_neg : ∀ x, x < 0 → f (-x) = -f x, {
    intros x hx,
    rw [f, if_neg (lt_neg_iff_add_one.2 hx), if_pos (lt_irrefl _)],
    simp only [sqrt_neg_eq'_iff_le_zero.2 hx in ],
    rw eq_neg_iff_add_eq_zero,
    ring
  },
  split_ifs; assumption,
}

theorem find_k_range (x k : ℝ) (hx : x ∈ ℝ) (hk : k ∈ ℝ) :
  (∀ x : ℝ, f (k * 4^x - 1) < f (3 * 4^x - 2^(x+1))) → k < 2 :=
by {
  intros h,
  simp only [f, if_pos, f, if_neg, if_neg],

  have h_pos : ∀ (x : ℝ),  k * 4^x - 1 < 3 * 4^x - 2^(x+1), from sorry,
  have h_simp : ∀ t > 0, k * t^2 - 1 < 3 * t^2 - 2 * t, from sorry,

  sorry,
}

end find_k_range_l409_409526


namespace smallest_missing_digit_l409_409324

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l409_409324


namespace cos_pi_half_eq_zero_l409_409859

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l409_409859


namespace cos_of_90_degrees_l409_409996

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l409_409996


namespace cos_pi_half_eq_zero_l409_409863

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l409_409863


namespace smallest_digit_never_in_units_place_of_odd_numbers_l409_409309

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l409_409309


namespace proof_problem_l409_409522

-- Conditions
def in_fourth_quadrant (α : ℝ) : Prop := (α > 3 * Real.pi / 2) ∧ (α < 2 * Real.pi)
def x_coordinate_unit_circle (α : ℝ) : Prop := Real.cos α = 1/3

-- Proof statement
theorem proof_problem (α : ℝ) (h1 : in_fourth_quadrant α) (h2 : x_coordinate_unit_circle α) :
  Real.tan α = -2 * Real.sqrt 2 ∧
  ((Real.sin α)^2 - Real.sqrt 2 * (Real.sin α) * (Real.cos α)) / (1 + (Real.cos α)^2) = 6 / 5 :=
by
  sorry

end proof_problem_l409_409522


namespace simplify_expression_evaluate_expression_at_neg1_evaluate_expression_at_2_l409_409058

theorem simplify_expression (a : ℤ) (h1 : -2 < a) (h2 : a ≤ 2) (h3 : a ≠ 0) (h4 : a ≠ 1) :
  (a - (2 * a - 1) / a) / ((a - 1) / a) = a - 1 :=
by
  sorry

theorem evaluate_expression_at_neg1 (h : (-1 : ℤ) ≠ 0) (h' : (-1 : ℤ) ≠ 1) : 
  (-1 - (2 * (-1) - 1) / (-1)) / ((-1 - 1) / (-1)) = -2 :=
by
  sorry

theorem evaluate_expression_at_2 (h : (2 : ℤ) ≠ 0) (h' : (2 : ℤ) ≠ 1) : 
  (2 - (2 * 2 - 1) / 2) / ((2 - 1) / 2) = 1 :=
by
  sorry

end simplify_expression_evaluate_expression_at_neg1_evaluate_expression_at_2_l409_409058


namespace cos_90_eq_zero_l409_409950

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l409_409950


namespace least_palindrome_div_by_5_l409_409665

def is_palindrome (n : ℕ) : Prop := 
  let s := n.to_string in 
  s = s.reverse

def is_divisible_by (n d : ℕ) : Prop :=
  n % d = 0

def least_four_digit_palindrome_divisible_by_5 : ℕ :=
  5005

theorem least_palindrome_div_by_5 (n : ℕ) (h1 : is_palindrome n) (h2 : n ≥ 1000) (h3 : n < 10000) (h4 : is_divisible_by n 5) :
  n = 5005 :=
sorry

end least_palindrome_div_by_5_l409_409665


namespace point_in_fourth_quadrant_l409_409493

-- Define the imaginary unit i
def i : ℂ := complex.I

-- Given that Z corresponds to the point (2, -1)
def Z : ℂ := 2 - i

-- Defining the complex number to be analyzed
def complexNumber : ℂ := (1 - 2*i) / Z

-- Defining the coordinates corresponding to the complex number
def coordinates : ℂ := (4/5 : ℝ) - (3/5 : ℝ) * i

-- A conjecture in Lean 4: Prove that coordinates lie in the fourth quadrant
theorem point_in_fourth_quadrant : complexNumber = coordinates → ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ coordinates = x + y * i := 
  by sorry

end point_in_fourth_quadrant_l409_409493


namespace find_angle_B_l409_409605

-- Define the necessary trigonometric identities and dependencies
open Real

-- Declare the conditions under which we are working
theorem find_angle_B : 
  ∀ {a b A B : ℝ}, 
    a = 1 → 
    b = sqrt 3 → 
    A = π / 6 → 
    (B = π / 3 ∨ B = 2 * π / 3) := 
  by 
    intros a b A B ha hb hA
    sorry

end find_angle_B_l409_409605


namespace probability_divisible_by_3_l409_409811

/-
  Problem statement and conditions:
  - Anastasia starts at (1, 0).
  - Each step she moves to one of (x-1, y), (x+1, y), (x, y-1), (x, y+1) with equal probability.
  - She stops upon reaching a point of the form (k, k).
  
  We want to prove that the probability she stops at a point (k, k) where k is divisible by 3 is (3 - sqrt(3)) / 3.
-/

theorem probability_divisible_by_3 :
    let P := ∑ n in Nat.antidiagonal, 
              (Catalan (n-1)) / (4^(2*n-1)) *
              (∑ i in Finset.filter (λ i, i % 3 = n % 3), Nat.choose (2*n - 1) i ) in
    P = (3 - Real.sqrt 3) / 3 := sorry

end probability_divisible_by_3_l409_409811


namespace cos_90_deg_eq_zero_l409_409904

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l409_409904


namespace smallest_digit_never_in_units_place_of_odd_l409_409194

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l409_409194


namespace cos_90_eq_0_l409_409880

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l409_409880


namespace quadratic_to_vertex_form_addition_l409_409585

theorem quadratic_to_vertex_form_addition (a h k : ℝ) (x : ℝ) :
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 :=
by
  intro h_eq
  sorry

end quadratic_to_vertex_form_addition_l409_409585


namespace find_unknown_number_l409_409757

theorem find_unknown_number (x : ℤ) :
  (20 + 40 + 60) / 3 = 5 + (20 + 60 + x) / 3 → x = 25 :=
by
  sorry

end find_unknown_number_l409_409757


namespace algorithm_min_l409_409804

-- Definition of the algorithm function
def algorithm (a b c d : ℕ) : ℕ :=
  let m := if b < a then b else a in
  let m := if c < m then c else m in
  let m := if d < m then d else m in
  m

-- Proof statement
theorem algorithm_min (a b c d : ℕ) : algorithm a b c d = min (min (min a b) c) d := 
  sorry

end algorithm_min_l409_409804


namespace smallest_not_odd_unit_is_zero_l409_409178

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l409_409178


namespace quadratic_inequality_solution_l409_409486

theorem quadratic_inequality_solution (x : ℝ) :
  (x < -7 ∨ x > 3) → x^2 + 4 * x - 21 > 0 :=
by
  -- The proof will go here
  sorry

end quadratic_inequality_solution_l409_409486


namespace chord_length_circle_C2_equation_l409_409496

noncomputable def length_of_chord_AB (C1_eq : ℝ → ℝ → ℝ) (l_eq : ℝ → ℝ → ℝ) : ℝ := sorry

theorem chord_length :
  ∀ (x y : ℝ), (x^2 + y^2 - 2 * x - 4 * y + 4 = 0) ∧ (x + 2 * y - 4 = 0) →
  length_of_chord_AB (λ x y, x^2 + y^2 - 2 * x - 4 * y + 4) (λ x y, x + 2 * y - 4) = (4 * real.sqrt 5) / 5 :=
sorry

noncomputable def equation_of_circle_C2 (E F : ℝ × ℝ) (parallel_eq : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ := sorry

theorem circle_C2_equation :
  ∀ (x y : ℝ), 
  (x^2 + y^2 + 6 * x - 16 = 0) → 
  equation_of_circle_C2 (1, -3) (0, 4) (λ x y, 2 * x + y + 1) x y = x^2 + y^2 + 6 * x - 16 :=
sorry

end chord_length_circle_C2_equation_l409_409496


namespace cos_pi_half_eq_zero_l409_409865

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l409_409865


namespace smallest_digit_not_in_units_place_of_odd_numbers_l409_409160

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l409_409160


namespace imaginary_part_z_is_zero_l409_409710

def z : ℂ := (1 + complex.I) / (1 - complex.I) + (1 - complex.I)

theorem imaginary_part_z_is_zero : complex.im z = 0 := 
by 
  sorry

end imaginary_part_z_is_zero_l409_409710


namespace smallest_digit_never_in_units_place_of_odd_numbers_l409_409300

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l409_409300


namespace find_x_value_l409_409461

theorem find_x_value (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end find_x_value_l409_409461


namespace cos_ninety_degrees_l409_409923

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l409_409923


namespace cos_90_eq_zero_l409_409942

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l409_409942


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409126

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409126


namespace cos_90_eq_zero_l409_409974

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l409_409974


namespace cos_squared_alpha_plus_pi_over_4_l409_409516

theorem cos_squared_alpha_plus_pi_over_4 (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + Real.pi / 4) ^ 2 = 1 / 6 :=
by
  sorry

end cos_squared_alpha_plus_pi_over_4_l409_409516


namespace smallest_digit_not_in_units_place_of_odd_l409_409279

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l409_409279


namespace Tapanga_corey_candies_difference_l409_409693

theorem Tapanga_corey_candies_difference :
  ∀ (t c : ℕ), t + c = 66 ∧ c = 29 → t - c = 8 := by
  intros t c h
  obtain ⟨h1, h2⟩ := h
  have hc : c = 29 := h2
  rw [hc, add_comm] at h1
  have ht : t = 66 - 29 := by linarith
  rw [ht, hc]
  linarith

end Tapanga_corey_candies_difference_l409_409693


namespace smallest_digit_not_found_in_units_place_of_odd_number_l409_409140

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l409_409140


namespace Charles_speed_with_music_l409_409832

theorem Charles_speed_with_music (S : ℝ) (h1 : 40 / 60 + 30 / 60 = 70 / 60) (h2 : S * (40 / 60) + 4 * (30 / 60) = 6) : S = 8 :=
by
  sorry

end Charles_speed_with_music_l409_409832


namespace smallest_digit_not_in_units_place_of_odd_l409_409267

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l409_409267


namespace books_left_l409_409666

variable (initialBooks : ℕ) (soldBooks : ℕ) (remainingBooks : ℕ)

-- Conditions
def initial_conditions := initialBooks = 136 ∧ soldBooks = 109

-- Question: Proving the remaining books after the sale
theorem books_left (initial_conditions : initialBooks = 136 ∧ soldBooks = 109) : remainingBooks = 27 :=
by
  cases initial_conditions
  sorry

end books_left_l409_409666


namespace cos_90_equals_0_l409_409936

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l409_409936


namespace probability_of_sum_17_is_correct_l409_409410

def probability_sum_17 : ℚ :=
  let favourable_outcomes := 2
  let total_outcomes := 81
  favourable_outcomes / total_outcomes

theorem probability_of_sum_17_is_correct :
  probability_sum_17 = 2 / 81 :=
by
  -- The proof steps are not required for this task
  sorry

end probability_of_sum_17_is_correct_l409_409410


namespace smallest_unfound_digit_in_odd_units_l409_409231

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l409_409231


namespace smallest_not_odd_unit_is_zero_l409_409181

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l409_409181


namespace exactly_one_cyclic_shift_has_positive_partial_sums_l409_409761

def sequence (n : ℕ) (a : ℕ → ℤ) : Prop :=
  (∑ i in finset.range n, a i) = 1

def cyclic_shift (n : ℕ) (a : ℕ → ℤ) (k : ℕ) : (ℕ → ℤ) :=
  λ i, a ((i + k) % n)

def partial_sums_positive (n : ℕ) (a : ℕ → ℤ) : Prop :=
  ∀ k : ℕ, k < n → (∑ i in finset.range k, a i) > 0

theorem exactly_one_cyclic_shift_has_positive_partial_sums
  (n : ℕ) (a : ℕ → ℤ)
  (h_sum : sequence n a) :
  ∃! k : ℕ, k < n ∧ partial_sums_positive n (cyclic_shift n a k) := sorry

end exactly_one_cyclic_shift_has_positive_partial_sums_l409_409761


namespace log_eq_exponent_eq_l409_409463

theorem log_eq_exponent_eq (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
by sorry

end log_eq_exponent_eq_l409_409463


namespace smallest_digit_never_in_units_place_of_odd_l409_409197

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l409_409197


namespace VyshnyVolochek_degree_and_paths_l409_409821

open Graph

-- Define the vertex set
inductive City
| Moscow
| SaintPetersburg
| Tver
| Yaroslavl
| Klin
| VyshnyVolochek
| Shestikhino
| Zavidovo
| Bologoye

open City

-- Define the edge set (direct connections)
noncomputable def direct_connections : City → nat
| Moscow := 7
| SaintPetersburg := 5
| Tver := 4
| Yaroslavl := 2
| Bologoye := 2
| Shestikhino := 2
| Zavidovo := 1
| Klin := y
| VyshnyVolochek := x

theorem VyshnyVolochek_degree_and_paths : 
    ∃ x y : nat, ((x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) ∧ 15 paths_from Moscow to SaintPetersburg with_correct_degree_sum) := 
by
  sorry

end VyshnyVolochek_degree_and_paths_l409_409821


namespace f_maps_T_to_T_l409_409008

variable {S : Type*} [Fintype S] (A : S → S → Type*) [Fintype (S → S)]

noncomputable def T (f : S → S) : Set S :=
  {x | ∃ y, f y = x}

variable (f : S → S)
variable (g : S → S)
/- Conditions -/
axiom h1 : ∀ g : S → S, f ≠ g → f ∘ g ∘ f ≠ g ∘ f ∘ g

/- Prove -/
theorem f_maps_T_to_T (f : S → S) (hf : f ∈ (λ (g : S → S), ∀ g ≠ f, f ∘ g ∘ f ≠ g ∘ f ∘ g)) :
  T f = f '' T f := sorry

end f_maps_T_to_T_l409_409008


namespace henrietta_paint_needed_l409_409552

theorem henrietta_paint_needed :
  let living_room_area := 600
  let num_bedrooms := 3
  let bedroom_area := 400
  let paint_coverage_per_gallon := 600
  let total_area := living_room_area + (num_bedrooms * bedroom_area)
  total_area / paint_coverage_per_gallon = 3 :=
by
  -- Proof should be completed here.
  sorry

end henrietta_paint_needed_l409_409552


namespace sum_b_div_5_pow_eq_l409_409010

namespace SequenceSumProblem

-- Define the sequence b_n
def b : ℕ → ℝ
| 0       => 2
| 1       => 3
| (n + 2) => b (n + 1) + b n

-- The infinite series sum we need to prove
noncomputable def sum_b_div_5_pow (Y : ℝ) : Prop :=
  Y = ∑' n : ℕ, (b n) / (5 ^ (n + 1))

-- The statement of the problem
theorem sum_b_div_5_pow_eq : sum_b_div_5_pow (2 / 25) :=
sorry

end SequenceSumProblem

end sum_b_div_5_pow_eq_l409_409010


namespace smallest_not_odd_unit_is_zero_l409_409184

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l409_409184


namespace smallest_digit_not_found_in_units_place_of_odd_number_l409_409141

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l409_409141


namespace fractions_count_l409_409805

def isFraction (A B : ℝ) := 
  B ≠ 0 ∧ (∃ x : ℝ, B = x)

def expressions : List (ℝ × ℝ) := [(-3, 2), (4, x - y), (x + y, 1), ((x^2 + 1), pi), (7, 8), (5a, 3a)]

def countFractions (exprs : List (ℝ × ℝ)) : Nat :=
  exprs.filter (fun e => isFraction e.1 e.2).length

theorem fractions_count : countFractions expressions = 1 := 
by 
  sorry

end fractions_count_l409_409805


namespace cos_90_eq_0_l409_409889

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l409_409889


namespace smallest_digit_not_found_in_units_place_of_odd_number_l409_409147

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l409_409147


namespace smallest_digit_never_in_units_place_of_odd_numbers_l409_409305

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l409_409305


namespace smallest_digit_not_in_odd_units_l409_409216

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l409_409216


namespace cos_90_eq_0_l409_409959

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l409_409959


namespace difference_in_mpg_l409_409040

theorem difference_in_mpg 
  (advertisedMPG : ℕ) 
  (tankCapacity : ℕ) 
  (totalMilesDriven : ℕ) 
  (h_advertised : advertisedMPG = 35) 
  (h_tank : tankCapacity = 12) 
  (h_miles : totalMilesDriven = 372) : 
  advertisedMPG - (totalMilesDriven / tankCapacity) = 4 :=
by
  rw [h_advertised, h_tank, h_miles]
  norm_num
  sorry

end difference_in_mpg_l409_409040


namespace smallest_digit_not_in_units_place_of_odd_numbers_l409_409154

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l409_409154


namespace smallest_digit_never_at_units_place_of_odd_l409_409177

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l409_409177


namespace smallest_digit_not_in_units_place_of_odd_l409_409290

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l409_409290


namespace principal_amount_l409_409750

theorem principal_amount (P R : ℝ) (h1 : P + (P * R * 2) / 100 = 780) (h2 : P + (P * R * 7) / 100 = 1020) : P = 684 := 
sorry

end principal_amount_l409_409750


namespace equation_squares_l409_409609

theorem equation_squares (a b c : ℤ) (h : (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = a ^ 2 + b ^ 2 - c ^ 2) :
  ∃ k1 k2 : ℤ, (a + 3) ^ 2 + (b + 4) ^ 2 - (c + 5) ^ 2 = k1 ^ 2 ∧ a ^ 2 + b ^ 2 - c ^ 2 = k2 ^ 2 :=
by
  sorry

end equation_squares_l409_409609


namespace billy_boxes_of_candy_l409_409822

theorem billy_boxes_of_candy (pieces_per_box total_pieces : ℕ) (h1 : pieces_per_box = 3) (h2 : total_pieces = 21) :
  total_pieces / pieces_per_box = 7 := 
by
  sorry

end billy_boxes_of_candy_l409_409822


namespace smallest_digit_not_in_odd_units_l409_409208

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l409_409208


namespace trig_expression_equality_l409_409829

theorem trig_expression_equality :
  let tan_60 := Real.sqrt 3
  let tan_45 := 1
  let cos_30 := Real.sqrt 3 / 2
  2 * tan_60 + tan_45 - 4 * cos_30 = 1 := by
  let tan_60 := Real.sqrt 3
  let tan_45 := 1
  let cos_30 := Real.sqrt 3 / 2
  sorry

end trig_expression_equality_l409_409829


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409122

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409122


namespace problem1_l409_409077

section Problem1
variables (s0_1 s2_3 s4_5 s6_7 s8_9 s10 : ℕ)
variables (total_students percent_8_or_more percent_5_or_less : ℝ)

def calculation_data := 
  s0_1 = 4 ∧ s2_3 = 6 ∧ s4_5 = 8 ∧ s6_7 = 10 ∧ s8_9 = 6 ∧ s10 = 6

def total_students_calculation (s0_1 s2_3 s4_5 s6_7 s8_9 s10 : ℕ) : ℕ :=
  s0_1 + s2_3 + s4_5 + s6_7 + s8_9 + s10

def percentage (part whole : ℕ) : ℝ :=
  (part.toReal / whole.toReal) * 100

theorem problem1 : calculation_data s0_1 s2_3 s4_5 s6_7 s8_9 s10 →
  total_students_calculation s0_1 s2_3 s4_5 s6_7 s8_9 s10 = 40 ∧
  percentage (s8_9 + s10) 40 = 20 ∧
  percentage (s0_1 + s2_3 + s4_5) 40 = 45 :=
by
  intro h
  sorry

end problem1_l409_409077


namespace min_value_f_l409_409495

def f (x : ℝ) : ℝ := |2 * x - 1| + |3 * x - 2| + |4 * x - 3| + |5 * x - 4|

theorem min_value_f : (∃ x : ℝ, ∀ y : ℝ, f y ≥ f x) := 
sorry

end min_value_f_l409_409495


namespace possible_even_exponents_count_l409_409747

theorem possible_even_exponents_count :
  let exponents := {n | n > 0 ∧ n ≤ 10 ∧ n % 2 = 0} in
  exponents.card = 5 :=
by
  sorry -- Proof is skipped as per instructions.

end possible_even_exponents_count_l409_409747


namespace smallest_digit_not_found_in_units_place_of_odd_number_l409_409136

def odd_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_digits : Set ℕ := {0, 2, 4, 6, 8}
def smallest_digit_not_in_odd_digits : ℕ := 0

theorem smallest_digit_not_found_in_units_place_of_odd_number : smallest_digit_not_in_odd_digits ∈ even_digits ∧ smallest_digit_not_in_odd_digits ∉ odd_digits :=
by
  -- Step by step proof would start here (but is omitted with sorry)
  sorry

end smallest_digit_not_found_in_units_place_of_odd_number_l409_409136


namespace problem_statement_l409_409636

variables (B : Matrix (Fin 2) (Fin 2) ℚ)
variables (r s : ℚ)
variables (I : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![0, 1]])

noncomputable def B_def := ![![1, 3], ![4, 2]] : Matrix (Fin 2) (Fin 2) ℚ

theorem problem_statement : 
  B = B_def → B^6 = 2080 • B + 7330 • I :=
begin
  sorry
end

end problem_statement_l409_409636


namespace smallest_digit_not_in_units_place_of_odd_l409_409259

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l409_409259


namespace cos_90_eq_zero_l409_409857

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l409_409857


namespace cos_90_equals_0_l409_409933

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l409_409933


namespace smallest_digit_never_in_units_place_of_odd_l409_409201

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l409_409201


namespace cone_base_circumference_l409_409793

theorem cone_base_circumference (V h : ℝ) (C : ℝ) (π : ℝ) (r : ℝ) 
  (hV : V = 12 * π) 
  (hh : h = 4)
  (hv : V = (1 / 3) * π * r^2 * h) 
  (hπ : π > 0) : 
  C = 2 * π * r := by
  have hr : r = real.sqrt 9 := by sorry
  have hr_simplified : r = 3 := by sorry
  have hC : C = 2 * π * 3 := by sorry
  exact hC

end cone_base_circumference_l409_409793


namespace drain_pool_time_l409_409070

-- Define the conditions
def hoseRate : ℕ := 60 -- cubic feet per minute
def poolWidth : ℕ := 60 -- feet
def poolLength : ℕ := 150 -- feet
def poolDepth : ℕ := 10 -- feet
def poolCapacity : ℝ := 0.80 -- 80% capacity

-- Define the pool volume when full
def poolVolumeFull : ℕ := poolWidth * poolLength * poolDepth

-- Calculate the pool volume at 80% capacity
def poolVolumeCurrent : ℝ := poolCapacity * poolVolumeFull

-- Calculate the time to drain the pool in minutes and hours
def timeToDrainMinutes : ℝ := poolVolumeCurrent / hoseRate
def timeToDrainHours : ℝ := timeToDrainMinutes / 60

-- The goal is to prove that timeToDrainHours = 20
theorem drain_pool_time : timeToDrainHours = 20 := by
  sorry

end drain_pool_time_l409_409070


namespace find_x_value_l409_409458

theorem find_x_value (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end find_x_value_l409_409458


namespace sum_of_coordinates_of_intersection_l409_409415

def h : ℝ → ℝ := -- Define h(x). This would be specific to the function provided; we abstract it here for the proof.
sorry

theorem sum_of_coordinates_of_intersection (a b : ℝ) (h_eq: h a = h (a - 5)) : a + b = 6 :=
by
  -- We need a [step from the problem conditions], hence introducing the given conditions
  have : b = h a := sorry
  have : b = h (a - 5) := sorry
  exact sorry

end sum_of_coordinates_of_intersection_l409_409415


namespace cos_90_equals_0_l409_409937

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l409_409937


namespace base_b_square_l409_409430

theorem base_b_square (b : ℕ) (h : b > 4) : ∃ k : ℕ, k^2 = b^2 + 4 * b + 4 := 
by 
  sorry

end base_b_square_l409_409430


namespace smallest_digit_not_in_units_place_of_odd_l409_409289

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l409_409289


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409338

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409338


namespace smallest_digit_never_in_units_place_of_odd_numbers_l409_409308

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l409_409308


namespace number_of_correct_conclusions_l409_409537

noncomputable def f (x : ℝ) := sqrt 3 * sin (2 * x) - cos (2 * x)

theorem number_of_correct_conclusions :
  (1:ℕ)
    -- ①: f(x) has a minimum positive period of π
    + (if ∃ T > 0, (∀ x, f(x + T) = f(x)) ∧ (∀ T' > 0, (∀ x, f(x + T') = f(x)) → T ≤ T') ∧ T = π then 1 else 0)
    -- ②: f(x) is an increasing function in the interval [-π/3, π/6]
    + (if ∀ x y, -π/3 ≤ x → x < y → y ≤ π/6 → f(x) < f(y) then 1 else 0)
    -- ③: The graph of f(x) is symmetric about the point (π/12, 0)
    + (if ∃ a b, (a = π/12) ∧ (b = 0) ∧ (∀ x, f(a + x) = b - f(a - x)) then 1 else 0)
    -- ④: x = π/3 is a symmetry axis of f(x)
    + (if ∃ a, (a = π/3) ∧ (∀ x, f(a - x) = f(a + x)) then 1 else 0)
  = 3 :=
sorry

end number_of_correct_conclusions_l409_409537


namespace adam_room_shelves_l409_409406

def action_figures_per_shelf : ℕ := 15
def total_action_figures : ℕ := 120
def total_shelves (total_figures shelves_capacity : ℕ) : ℕ := total_figures / shelves_capacity

theorem adam_room_shelves :
  total_shelves total_action_figures action_figures_per_shelf = 8 :=
by
  sorry

end adam_room_shelves_l409_409406


namespace smallest_digit_never_in_units_place_of_odd_l409_409196

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l409_409196


namespace quadratic_eq_one_real_root_l409_409011

theorem quadratic_eq_one_real_root (c b : ℝ) (h1 : 0 < c)
  (h2 : ∀ x, (x^2 + 2 * real.sqrt c * x + b = 0) → (⟦ x ≥ 0 ⟧ ∨ ⟦ x ≤ 0 ⟧)) :
  c = b :=
by
  sorry

end quadratic_eq_one_real_root_l409_409011


namespace probability_of_events_l409_409659

noncomputable def total_types : ℕ := 8

noncomputable def fever_reducing_types : ℕ := 3

noncomputable def cough_suppressing_types : ℕ := 5

noncomputable def total_ways_to_choose_two : ℕ := Nat.choose total_types 2

noncomputable def event_A_ways : ℕ := total_ways_to_choose_two - Nat.choose cough_suppressing_types 2

noncomputable def P_A : ℚ := event_A_ways / total_ways_to_choose_two

noncomputable def event_B_ways : ℕ := fever_reducing_types * cough_suppressing_types

noncomputable def P_B_given_A : ℚ := event_B_ways / event_A_ways

theorem probability_of_events :
  P_A = 9 / 14 ∧ P_B_given_A = 5 / 6 := by
  sorry

end probability_of_events_l409_409659


namespace sum_of_digits_of_largest_five_digit_number_with_product_180_l409_409635

noncomputable def largest_five_digit_number_with_product_180 : ℕ :=
  let digits := [9, 5, 4, 1, 1] in -- Identifying correct digits to form the number
  digits.sort.reverse.foldl (λ acc x => acc * 10 + x) 0

theorem sum_of_digits_of_largest_five_digit_number_with_product_180 :
  let M := largest_five_digit_number_with_product_180 in
  (M.digits.foldl (λ acc x => acc + x) 0) = 20 := 
sorry

end sum_of_digits_of_largest_five_digit_number_with_product_180_l409_409635


namespace smallest_digit_never_in_units_place_of_odd_l409_409193

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l409_409193


namespace missing_number_evaluation_l409_409442

theorem missing_number_evaluation (x : ℝ) (h : |4 + 9 * x| - 6 = 70) : x = 8 :=
sorry

end missing_number_evaluation_l409_409442


namespace math_problem_l409_409521

variables (x y z : ℝ)

theorem math_problem
  (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ( (x^2 / (x + y) >= (3 * x - y) / 4) ) ∧ 
  ( (x^3 / (x + y)) + (y^3 / (y + z)) + (z^3 / (z + x)) >= (x * y + y * z + z * x) / 2 ) :=
by sorry

end math_problem_l409_409521


namespace map_length_represents_75_km_l409_409662
-- First, we broaden the import to bring in all the necessary libraries.

-- Define the conditions given in the problem.
def cm_to_km_ratio (cm : ℕ) (km : ℕ) : ℕ := km / cm

def map_represents (length_cm : ℕ) (length_km : ℕ) : Prop :=
  length_km = length_cm * cm_to_km_ratio 15 45

-- Rewrite the problem statement as a theorem in Lean 4.
theorem map_length_represents_75_km : map_represents 25 75 :=
by
  sorry

end map_length_represents_75_km_l409_409662


namespace smallest_fraction_divides_given_fractions_l409_409370

theorem smallest_fraction_divides_given_fractions (d1 d2 d3 n1 n2 n3 : ℕ) :
  d1 = 7 → d2 = 14 → d3 = 21 → n1 = 6 → n2 = 5 → n3 = 10 → 
  ∀ (x : ℚ), (x = n1 / d1) ∨ (x = n2 / d2) ∨ (x = n3 / d3) → ∃ f : ℚ, f = 1 / 42 :=
by
sory

end smallest_fraction_divides_given_fractions_l409_409370


namespace banana_groups_not_determined_l409_409723

def bananas : ℕ := 142
def oranges : ℕ := 356
def orange_groups : ℕ := 178
def oranges_per_group : ℕ := 2

theorem banana_groups_not_determined (bananas oranges orange_groups oranges_per_group : ℕ) 
    (h1 : oranges / oranges_per_group = orange_groups) : 
    ¬∃ banana_groups, ∀ g, (g ∈ banana_groups → |g| = bananas / g) :=
by 
  sorry

end banana_groups_not_determined_l409_409723


namespace angle_bisector_square_l409_409046

theorem angle_bisector_square (A B C D : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (dist : A → B → ℝ)
  (α β γ δ : A)
  (h1 : ∀ x y z : A, dist x y = dist y z → dist x z = dist x y)
  (htriangle : Triangle α β γ)
  (Dpoint : Point γ)
  (bisector_theorem : dist α δ / dist δ β = dist β γ / dist γ α):
  dist α δ ^ 2 = dist β γ * dist γ α - dist α δ * dist δ β := 
sorry

end angle_bisector_square_l409_409046


namespace smallest_digit_not_in_odd_units_l409_409243

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l409_409243


namespace cos_pi_half_eq_zero_l409_409860

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l409_409860


namespace line_equation_in_slope_intercept_form_l409_409401

variable {x y : ℝ}

theorem line_equation_in_slope_intercept_form :
  (3 * (x - 2) - 4 * (y - 8) = 0) → (y = (3 / 4) * x + 6.5) :=
by
  intro h
  sorry

end line_equation_in_slope_intercept_form_l409_409401


namespace smallest_digit_never_in_units_place_of_odd_l409_409202

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l409_409202


namespace smallest_digit_not_in_units_place_of_odd_l409_409254

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l409_409254


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409340

theorem smallest_digit_never_in_units_place_of_odd_number : 
  ∀ units_digit : ℕ, (units_digit ∈ {1, 3, 5, 7, 9} → false) → units_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409340


namespace betty_cupcakes_per_hour_l409_409416

theorem betty_cupcakes_per_hour (B : ℕ) (Dora_rate : ℕ) (betty_break_hours : ℕ) (total_hours : ℕ) (cupcake_diff : ℕ) :
  Dora_rate = 8 →
  betty_break_hours = 2 →
  total_hours = 5 →
  cupcake_diff = 10 →
  (total_hours - betty_break_hours) * B = Dora_rate * total_hours - cupcake_diff →
  B = 10 :=
by
  intros hDora_rate hbreak_hours htotal_hours hcupcake_diff hcupcake_eq
  sorry

end betty_cupcakes_per_hour_l409_409416


namespace smallest_unfound_digit_in_odd_units_l409_409237

def is_odd_unit_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_unfound_digit_in_odd_units : ∃ d : ℕ, ¬is_odd_unit_digit d ∧ (∀ d' : ℕ, d < d' → is_odd_unit_digit d' → False) := 
sorry

end smallest_unfound_digit_in_odd_units_l409_409237


namespace area_is_128_l409_409634

def piecewise_function (x : ℝ) : ℝ :=
  if x ≤ 6 then x^2 else 3 * x - 10

def area_under_piecewise_function : ℝ :=
  (∫ x in 0..6, x^2) + (∫ x in 6..10, 3 * x - 10)

theorem area_is_128 :
  area_under_piecewise_function = 128 :=
by
  sorry

end area_is_128_l409_409634


namespace problem_statement_l409_409021

noncomputable def curvature (k_A k_B : ℝ) (d : ℝ) : ℝ :=
  |k_A - k_B| / d

def is_constant_curvature (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, curvature (fderiv ℝ f x) (fderiv ℝ f y) (dist x y) = k

theorem problem_statement :
  (curvature (3 : ℝ) (3 : ℝ) (dist 1 (-1)) = 0) ∧
  (∃ f : ℝ → ℝ, is_constant_curvature f) ∧
  (∀ (x_A x_B : ℝ), curvature (2*x_A) (2*x_B) (dist x_A x_B) ≤ 2) ∧
  (∀ (x_1 x_2 : ℝ), curvature (exp x_1) (exp x_2) (dist x_1 x_2) < 1) :=
begin
  sorry,
end

end problem_statement_l409_409021


namespace equal_chords_equidistant_l409_409043

open EuclideanGeometry

variables (O A B A1 B1 M M1 : Point)
variables (circle : Circle O)
variables (h_eq_chord1 : isChord circle A B)
variables (h_eq_chord2 : isChord circle A1 B1)
variables (h_eq_length : dist A B = dist A1 B1)
variables (O_perpendicular_AB : perpendicular O M (line_through A B))
variables (O_perpendicular_A1B1 : perpendicular O M1 (line_through A1 B1))
variables (M_midpoint : midpoint A B M)
variables (M1_midpoint : midpoint A1 B1 M1)

theorem equal_chords_equidistant :
  dist O M = dist O M1 :=
by sorry

end equal_chords_equidistant_l409_409043


namespace school_competition_students_l409_409594

theorem school_competition_students :
  ∃ (n9 n10 n11 : ℕ), 
    (7 * n10 = 4 * n9) ∧ 
    (21 * n11 = 10 * n9) ∧ 
    (n9 + n10 + n11 = 43) :=
begin
  sorry
end

end school_competition_students_l409_409594


namespace least_number_of_table_entries_l409_409759

theorem least_number_of_table_entries (n : ℕ) (h : n = 6) : (n * (n - 1)) / 2 = 15 :=
by
  rw [h]
  -- simplified calculation
  rw [Nat.mul_sub_left_distrib, Nat.mul_comm, Nat.mul_comm (n - 1), Nat.sub_self, Nat.zero_mul, Nat.add_zero]
  norm_num
  sorry

end least_number_of_table_entries_l409_409759


namespace max_value_S_correct_l409_409500

noncomputable def max_value_S (n : ℕ) (M : ℝ) (h1 : 0 < n) (h2 : 0 < M) 
  (a : ℕ → ℝ) (h_arith_seq : ∀ i : ℕ, a (i+1) - a i = a 1 - a 0)
  (h_condition : a 1 ^ 2 + a (n + 1) ^ 2 ≤ M) : ℝ :=
  (n + 1) * Real.sqrt (10 * M) / 2

theorem max_value_S_correct (n : ℕ) (M : ℝ) (h1 : 0 < n) (h2 : 0 < M) 
  (a : ℕ → ℝ) (h_arith_seq : ∀ i : ℕ, a (i+1) - a i = a 1 - a 0)
  (h_condition : a 1 ^ 2 + a (n + 1) ^ 2 ≤ M) : 
  ∃ S : ℝ, S = a (n+1) + a (n+2) + ... + a (2*n+1) ∧ S = max_value_S n M h1 h2 a h_arith_seq h_condition :=
sorry

end max_value_S_correct_l409_409500


namespace smallest_digit_not_in_units_place_of_odd_l409_409262

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l409_409262


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409132

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409132


namespace smallest_digit_not_in_units_place_of_odd_l409_409285

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l409_409285


namespace hyperbola_eq_l409_409762

theorem hyperbola_eq (x y : ℝ) (h0 : ∀ (p : ℝ × ℝ), p = (2,1)) :
  (∃ a : ℝ, (a^2 = 3) ∧ (x^2 / a^2 - y^2 / (6 - a^2) = 1)) :=
begin
  sorry
end

end hyperbola_eq_l409_409762


namespace cos_90_eq_0_l409_409883

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l409_409883


namespace puppy_weight_l409_409791

variable (p s l r : ℝ)

theorem puppy_weight :
  p + s + l + r = 40 ∧ 
  p^2 + l^2 = 4 * s ∧ 
  p^2 + s^2 = l^2 → 
  p = Real.sqrt 2 :=
sorry

end puppy_weight_l409_409791


namespace remainder_of_sum_l409_409454

theorem remainder_of_sum :
  let expr := (1 - ∑ k in finset.range 11 \{0}, -1^k * nat.choose 90 k)
  let remainder := expr % 88
  remainder = 1 :=
by
  sorry

end remainder_of_sum_l409_409454


namespace cos_90_eq_zero_l409_409999

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l409_409999


namespace radius_of_spheres_l409_409733

noncomputable def height_cylinder : ℝ := 16
noncomputable def diameter_cylinder : ℝ := 16
noncomputable def radius_cylinder : ℝ := diameter_cylinder / 2
noncomputable def csa_cylinder (r h : ℝ) : ℝ := 2 * Real.pi * r * h
noncomputable def total_csa_two_cylinders : ℝ := 2 * csa_cylinder radius_cylinder height_cylinder
noncomputable def sa_sphere (r : ℝ) : ℝ := 4 * Real.pi * (r ^ 2)
noncomputable def total_sa_three_spheres (r : ℝ) : ℝ := 3 * sa_sphere r

theorem radius_of_spheres : 
  ∃ (r : ℝ), total_sa_three_spheres r = total_csa_two_cylinders ∧ r ≈ 6.53 :=
by
  sorry

end radius_of_spheres_l409_409733


namespace noah_ends_with_exactly_7_MMs_l409_409383

noncomputable def probability_Noah_ends_with_7_MMs : ℚ :=
  let P := λ n, if n = 0 then 1
                else if n = 1 then 1/2
                else if n = 2 then 3/4
                else if n = 3 then 5/8
                else if n = 4 then 11/16
                else if n = 5 then 21/32
                else 0 in
  P 5 * 1/2

theorem noah_ends_with_exactly_7_MMs : probability_Noah_ends_with_7_MMs = 21/64 := by
  unfold probability_Noah_ends_with_7_MMs
  sorry

end noah_ends_with_exactly_7_MMs_l409_409383


namespace count_7_without_2_in_range_1_to_500_l409_409607

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d ∈ n.digits 10

def is_valid (n : ℕ) : Prop :=
  ¬ contains_digit n 2

def count_digit_7_excluding_2 (m : ℕ) : ℕ :=
  (List.range m).count (λ n, is_valid n ∧ contains_digit n 7)

theorem count_7_without_2_in_range_1_to_500 : 
  count_digit_7_excluding_2 501 = 80 :=
sorry

end count_7_without_2_in_range_1_to_500_l409_409607


namespace smallest_digit_not_in_odd_units_l409_409251

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l409_409251


namespace inscribed_circle_radius_l409_409019

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem inscribed_circle_radius :
  let A := (0, 0)
  let B := (2, 0)
  let C := (2, 2)
  let D := (0, 2)
  let S := (1, 1)
  let P := (1, 0)
  let M := (2 / 3, 2 / 3)
  let N := (4 / 3, 2 / 3)
  distance P M - distance S M = ((real.sqrt 5 - real.sqrt 2) / 3) := by
  sorry

end inscribed_circle_radius_l409_409019


namespace dot_product_calculation_l409_409561

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (h1 : ∥a∥ = 4) (h2 : ∥b∥ = 5)

theorem dot_product_calculation :
  (2 • a + 3 • b) ⬝ (2 • a - 3 • b) = -161 :=
by sorry

end dot_product_calculation_l409_409561


namespace cos_90_eq_zero_l409_409847

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l409_409847


namespace pow_sqrt2_plus_1_eq_sqrt_m_and_sqrt_m_minus_1_l409_409630

theorem pow_sqrt2_plus_1_eq_sqrt_m_and_sqrt_m_minus_1 (n : ℕ) (h : 0 < n) :
  ∃ m : ℕ, (√2 + 1)^n = √m + √(m - 1) := sorry

end pow_sqrt2_plus_1_eq_sqrt_m_and_sqrt_m_minus_1_l409_409630


namespace smallest_digit_not_in_units_place_of_odd_numbers_l409_409153

theorem smallest_digit_not_in_units_place_of_odd_numbers :
  ∀ (d : ℕ), (d ∈ {1, 3, 5, 7, 9} → False) → d = 0 :=
begin
  sorry
end

end smallest_digit_not_in_units_place_of_odd_numbers_l409_409153


namespace probability_at_least_two_heads_five_coins_l409_409052

open BigOperators

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
∑ i in Finset.range (k + 1), (Nat.choose n i) * p^i * (1 - p)^(n - i)

theorem probability_at_least_two_heads_five_coins :
  binomial_probability 5 1 (1/2) = (3 / 16) →
  1 - binomial_probability 5 1 (1 / 2) = (13 / 16) :=
by
  intros h
  simp [h]
  norm_num
  rfl

end probability_at_least_two_heads_five_coins_l409_409052


namespace cos_90_eq_0_l409_409875

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l409_409875


namespace diff_count_19_pos_integers_l409_409382

def S : Set ℕ := {i | 1 ≤ i ∧ i ≤ 20}

theorem diff_count_19_pos_integers :
  {n | ∃ a b ∈ S, a ≠ b ∧ n = a - b}.card = 19 := by
sorry

end diff_count_19_pos_integers_l409_409382


namespace triangle_partition_l409_409686

variable (a b c : ℝ)

-- Condition: the sides of the triangle are distinct
def distinct_sides := a ≠ b ∧ a ≠ c ∧ b ≠ c

-- The main theorem to prove
theorem triangle_partition (h : distinct_sides a b c) : 
  ∃ (triangles : Fin 7 → Triangle), 
    (∀ n, is_isosceles (triangles n)) ∧
    ∃ i j k, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ (triangles i = triangles j ∧ triangles j = triangles k) :=
sorry

end triangle_partition_l409_409686


namespace smallest_digit_not_in_odd_units_l409_409239

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l409_409239


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409121

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409121


namespace cos_90_eq_zero_l409_409998

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l409_409998


namespace smallest_digit_not_in_odd_units_l409_409222

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l409_409222


namespace area_of_triangle_is_exp_2_l409_409432

noncomputable def tangent_triangle_area : ℝ :=
  let f := λ (x : ℝ), Real.exp (0.5 * x) in
  let df := deriv f in
  let m := df 4 in
  let b := f 4 - m * 4 in
  let x_intercept := -b / m in
  let y_intercept := f 4 in
  0.5 * x_intercept * y_intercept

theorem area_of_triangle_is_exp_2 : tangent_triangle_area = Real.exp 2 :=
by sorry

end area_of_triangle_is_exp_2_l409_409432


namespace smallest_digit_never_in_units_place_of_odd_numbers_l409_409311

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l409_409311


namespace maximum_calls_l409_409592

/-- Let there be 15 boys and 15 girls. Each boy can call at least one girl, and no boy calls the same girl twice.
Furthermore, it is possible to uniquely pair every boy and girl such that each pair consists of a boy and a girl whom he called.
Prove that the maximum number of calls that could have been made is 120. -/
theorem maximum_calls (boys girls : ℕ) (unique_pairs : ℕ) (calls_made : ℕ) :
  boys = 15 → girls = 15 → unique_pairs = 15 →
  ∀ b g, (b ≥ 1 ∧ b ≤ 15) → (g ≥ 1 ∧ g ≤ 15) → (calls_made ≤ 120) :=
by
  intros boys girls unique_pairs calls_made boys_eq girls_eq unique_pairs_eq b g b_range g_range
  sorry

end maximum_calls_l409_409592


namespace range_of_a_l409_409079

-- Defining the function f(x) = ln x + a / x
def f (a x : ℝ) : ℝ := Real.log x + a / x

-- Defining the derivative of the function
def f' (a x : ℝ) : ℝ := 1 / x - a / x^2

-- Stating the monotonicity condition and the required proof of the range of 'a'.
theorem range_of_a (a : ℝ) : (∀ x ≥ 2, 1 / x - a / x^2 ≥ 0) → a ≤ 2 :=
by
  -- Proof is omitted
  sorry

end range_of_a_l409_409079


namespace cos_90_eq_zero_l409_409983

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l409_409983


namespace function_f_property_l409_409479

noncomputable def f : ℤ → ℤ := sorry

theorem function_f_property (f : ℤ → ℤ) (k : ℤ) (h_nonneg : k ≥ 0)
  (hx : ∀ m n : ℤ, f(m) + f(n) = max (f(m + n)) (f(m - n))) :
  ∀ x : ℤ, f(x) = k * |x| :=
begin
  sorry
end

end function_f_property_l409_409479


namespace smallest_digit_never_in_units_place_of_odd_l409_409199

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l409_409199


namespace smallest_digit_never_in_units_place_of_odd_l409_409207

def is_odd_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def smallest_non_odd_digit : ℕ :=
  if (∀ d, d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → is_odd_digit d → false) then 0 else sorry

theorem smallest_digit_never_in_units_place_of_odd :
  smallest_non_odd_digit = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_l409_409207


namespace randi_has_6_more_nickels_than_peter_l409_409049

def ray_initial_cents : Nat := 175
def cents_given_peter : Nat := 30
def cents_given_randi : Nat := 2 * cents_given_peter
def nickel_worth : Nat := 5

def nickels (cents : Nat) : Nat :=
  cents / nickel_worth

def randi_more_nickels_than_peter : Prop :=
  nickels cents_given_randi - nickels cents_given_peter = 6

theorem randi_has_6_more_nickels_than_peter :
  randi_more_nickels_than_peter :=
sorry

end randi_has_6_more_nickels_than_peter_l409_409049


namespace cos_90_eq_0_l409_409877

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l409_409877


namespace set_cannot_be_divided_l409_409065

theorem set_cannot_be_divided
  (p : ℕ) (prime_p : Nat.Prime p) (p_eq_3_mod_4 : p % 4 = 3)
  (S : Finset ℕ) (hS : S.card = p - 1) :
  ¬∃ A B : Finset ℕ, A ∪ B = S ∧ A ∩ B = ∅ ∧ A.prod id = B.prod id := 
by {
  sorry
}

end set_cannot_be_divided_l409_409065


namespace Pau_total_fried_chicken_l409_409628

theorem Pau_total_fried_chicken :
  ∀ (kobe_order : ℕ),
  (pau_initial : ℕ) (pau_second : ℕ),
  kobe_order = 5 →
  pau_initial = 2 * kobe_order →
  pau_second = pau_initial →
  pau_initial + pau_second = 20 :=
by
  intros kobe_order pau_initial pau_second
  sorry

end Pau_total_fried_chicken_l409_409628


namespace discount_difference_l409_409794

theorem discount_difference (P : ℝ) (h₁ : 0 < P) : 
  let actual_combined_discount := 1 - (0.75 * 0.85)
  let claimed_discount := 0.40
  actual_combined_discount - claimed_discount = 0.0375 :=
by 
  sorry

end discount_difference_l409_409794


namespace temperature_rise_per_hour_l409_409683

-- Define the conditions
variables (x : ℕ) -- temperature rise per hour

-- Assume the given conditions
axiom power_outage : (3 : ℕ) * x = (6 : ℕ) * 4

-- State the proposition
theorem temperature_rise_per_hour : x = 8 :=
sorry

end temperature_rise_per_hour_l409_409683


namespace cos_90_eq_0_l409_409958

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l409_409958


namespace cos_90_eq_zero_l409_409977

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l409_409977


namespace smallest_digit_not_in_units_place_of_odd_l409_409284

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l409_409284


namespace smallest_digit_not_in_odd_units_l409_409219

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l409_409219


namespace parallel_lines_l409_409574

theorem parallel_lines (m : ℝ) : (∃ (l_1 l_2 : ℝ → ℝ), (l_1 = λ x, -m * x - 2 * m + 5) ∧ (l_2 = λ x, -(3 / (m - 2)) * x - (1 / (m - 2))) ∧ (∀ x, l_1 x = l_2 x ∨ l_1 x ≠ l_2 x)) → m = 3 ∨ m = -1 :=
by
  sorry

end parallel_lines_l409_409574


namespace smallest_digit_never_at_units_place_of_odd_l409_409169

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l409_409169


namespace smallest_digit_not_in_units_place_of_odd_l409_409276

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l409_409276


namespace inscribed_square_triangle_angle_l409_409798

theorem inscribed_square_triangle_angle :
  ∀ (A B C D E F : Type) (circle : A) 
  (is_square : B → C → D → E → Prop) 
  (is_equilateral_triangle : A → B → F → Prop)
  (shared_vertex : A = B)
  (distinct_vertices_ac : A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F) 
  (on_circle : ∀ v, (v = A ∨ v = B ∨ v = C ∨ v = D ∨ v = E ∨ v = F) → circle v),
  angle (line A B) (line A C) = 30 :=
by
  -- sorry skips the proof
  sorry

end inscribed_square_triangle_angle_l409_409798


namespace cos_pi_half_eq_zero_l409_409870

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l409_409870


namespace monotonic_increasing_interval_l409_409711

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem monotonic_increasing_interval : 
  {x : ℝ | ∀ y, y > x ∧ y ∈ set.Ioi 0 → f y > f x} = set.Ioi 0 :=
by
  sorry

end monotonic_increasing_interval_l409_409711


namespace barry_magic_days_l409_409032

theorem barry_magic_days (x : ℝ) (n : ℕ)
  (h1 : ∀ k, (0 ≤ k ∧ k ≤ n) → x * (∏ i in finset.range (k + 1), (i + 3) / (i + 2)) = x * 50)
  (h2 : x > 0)
  : n = 147 :=
by
  -- Proof steps go here
  sorry

end barry_magic_days_l409_409032


namespace equilateral_triangle_dot_product_l409_409510

-- Define the coordinates of points
structure Point where
  x : ℝ
  y : ℝ

noncomputable def C : Point := { x := 0, y := 0 }
noncomputable def A : Point := { x := 2 * Real.sqrt 3, y := 0 }
noncomputable def B : Point := { x := Real.sqrt 3, y := 3 }

noncomputable def vector (from to : Point) : Point := 
  { x := to.x - from.x, y := to.y - from.y }

noncomputable instance : Coe Point (ℝ × ℝ) :=
  ⟨λ p => (p.x, p.y)⟩

noncomputable def dot_product (u v : Point) : ℝ :=
  u.x * v.x + u.y * v.y

noncomputable def M : Point := 
  { x := (1/6 * vector C B).x + (2/3 * vector C A).x,
    y := (1/6 * vector C B).y + (2/3 * vector C A).y }

noncomputable def MA : Point := vector M A
noncomputable def MB : Point := vector M B

theorem equilateral_triangle_dot_product : dot_product MA MB = -2 := 
by
  show dot_product MA MB = -2
  sorry

end equilateral_triangle_dot_product_l409_409510


namespace smallest_digit_never_at_units_place_of_odd_l409_409175

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l409_409175


namespace pythagorean_triple_area_not_perfect_square_l409_409009

theorem pythagorean_triple_area_not_perfect_square 
  (a b c : ℕ) 
  (hPythagorean : a^2 + b^2 = c^2)
  (hPrimitive : Nat.gcd a b = 1 ∧ (Nat.odd a ∨ Nat.odd b)) :
  ¬ ∃ k : ℕ, k^2 = a * b / 2 :=
by
  sorry

end pythagorean_triple_area_not_perfect_square_l409_409009


namespace sports_field_dimensions_l409_409797

noncomputable def width_of_sports_field (a b : ℝ) : ℝ :=
(Real.sqrt (b^2 + 32 * a^2) - b + 4 * a) / 2

noncomputable def length_of_sports_field (a b : ℝ) : ℝ :=
(Real.sqrt (b^2 + 32 * a^2) + b + 4 * a) / 2

theorem sports_field_dimensions (a b : ℝ) : 
  let x := width_of_sports_field a b
  in x * (x + b) = (x + 2 * a) * (x + b + 2 * a) - x * (x + b) :=
sorry

end sports_field_dimensions_l409_409797


namespace cos_90_eq_0_l409_409898

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l409_409898


namespace bisector_intersection_independence_l409_409815

theorem bisector_intersection_independence
  (A B C P Q R : Type)
  (h_points_on_line : collinear [A, B, C])
  (h_gamma : ∃ (Γ : Type), on_circle Γ A ∧ on_circle Γ C ∧ center_not_on_AC Γ A C)
  (tangents_intersect : tangents_intersect_at A C Γ P)
  (line_PB_intersects_at_Q : intersects_circle_at Γ (line PB) Q)
  (h_angle_bisector : angle_bisector_intersects AC A Q C R) :
  ∀ (Γ₁ Γ₂ : Type), Γ₁ ≠ Γ₂ → 
    ∃ R₁ R₂ : Type, 
      angle_bisector_intersects AC A Q C R₁ ∧ 
      angle_bisector_intersects AC A Q C R₂ ∧ 
      R₁ = R₂ :=
sorry

end bisector_intersection_independence_l409_409815


namespace roots_are_real_l409_409632

noncomputable def has_only_real_roots 
  (f g : Polynomial ℝ) (k : ℝ) : Prop :=
∀ T : ℂ, f.eval T + k * g.eval T = 0 → ∃ (t : ℝ), T = t

theorem roots_are_real
  (n : ℕ) (h : n > 0)
  (f g : Polynomial ℝ)
  (h_fg : ∀ (x : ℂ), (1 + complex.I * x)^n = f.eval x + complex.I * g.eval x) :
  ∀ k : ℝ, has_only_real_roots f g k :=
by
  intros k T hT
  sorry

end roots_are_real_l409_409632


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409123

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409123


namespace smallest_digit_never_in_units_place_of_odd_numbers_l409_409298

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l409_409298


namespace smallest_digit_not_in_units_place_of_odd_l409_409292

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l409_409292


namespace division_of_powers_of_ten_l409_409359

theorem division_of_powers_of_ten :
  (10 ^ 0.7 * 10 ^ 0.4) / (10 ^ 0.2 * 10 ^ 0.6 * 10 ^ 0.3) = 1 := by
  sorry

end division_of_powers_of_ten_l409_409359


namespace cos_90_eq_zero_l409_409982

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l409_409982


namespace henrietta_paint_gallons_l409_409553

-- Define the conditions
def living_room_area : Nat := 600
def bedrooms_count : Nat := 3
def bedroom_area : Nat := 400
def coverage_per_gallon : Nat := 600

-- The theorem we want to prove
theorem henrietta_paint_gallons :
  (bedrooms_count * bedroom_area + living_room_area) / coverage_per_gallon = 3 :=
by
  sorry

end henrietta_paint_gallons_l409_409553


namespace negation_equiv_l409_409712

-- Define the initial proposition
def initial_proposition (x : ℝ) : Prop :=
  x^2 - x + 1 > 0

-- Define the negation of the initial proposition
def negated_proposition : Prop :=
  ∃ x₀ : ℝ, x₀^2 - x₀ + 1 ≤ 0

-- The statement asserting the negation equivalence
theorem negation_equiv :
  (¬ ∀ x : ℝ, initial_proposition x) ↔ negated_proposition :=
by sorry

end negation_equiv_l409_409712


namespace find_theta_in_interval_l409_409446

variable (θ : ℝ)

def angle_condition (θ : ℝ) (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 1 ∧ (x^3 * Real.cos θ - x * (1 - x) + (1 - x)^3 * Real.tan θ > 0)

theorem find_theta_in_interval :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → angle_condition θ x) →
  0 < θ ∧ θ < Real.pi / 2 :=
by
  sorry

end find_theta_in_interval_l409_409446


namespace cost_of_painting_cube_l409_409697

-- Definitions for conditions
def cost_per_kg : ℝ := 36.50
def coverage_per_kg : ℝ := 16  -- square feet
def side_length : ℝ := 8       -- feet

-- Derived constants
def area_per_face : ℝ := side_length * side_length
def number_of_faces : ℝ := 6
def total_surface_area : ℝ := number_of_faces * area_per_face
def paint_required : ℝ := total_surface_area / coverage_per_kg
def total_cost : ℝ := paint_required * cost_per_kg

-- Theorem statement
theorem cost_of_painting_cube : total_cost = 876 := by
  sorry

end cost_of_painting_cube_l409_409697


namespace crayons_per_child_l409_409439

theorem crayons_per_child (children : ℕ) (total_crayons : ℕ) (h1 : children = 18) (h2 : total_crayons = 216) : 
    total_crayons / children = 12 := 
by
  sorry

end crayons_per_child_l409_409439


namespace log_eq_exponent_eq_l409_409462

theorem log_eq_exponent_eq (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
by sorry

end log_eq_exponent_eq_l409_409462


namespace cos_90_eq_0_l409_409890

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l409_409890


namespace smallest_digit_not_in_odd_units_l409_409214

theorem smallest_digit_not_in_odd_units : 
  ∃ d : ℕ, (d = 0) ∧ (∀ (n : ℕ), n ∈ {1, 3, 5, 7, 9} → d ≠ n ∧ d < 10) := 
by
  sorry

end smallest_digit_not_in_odd_units_l409_409214


namespace sum_of_number_and_reverse_l409_409702

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by
  sorry

end sum_of_number_and_reverse_l409_409702


namespace value_A_minus_B_l409_409398

-- Conditions definitions
def A : ℕ := (1 * 1000) + (16 * 100) + (28 * 10)
def B : ℕ := 355 + 245 * 3

-- Theorem statement
theorem value_A_minus_B : A - B = 1790 := by
  sorry

end value_A_minus_B_l409_409398


namespace find_divisor_l409_409760

theorem find_divisor (dividend remainder quotient : ℕ) (h1 : dividend = 76) (h2 : remainder = 8) (h3 : quotient = 4) : ∃ d : ℕ, dividend = (d * quotient) + remainder ∧ d = 17 :=
by
  sorry

end find_divisor_l409_409760


namespace work_required_to_pump_l409_409391

noncomputable def work_pumping_liquid (R H : ℝ) (gamma : ℝ) : ℝ :=
  240 * pi * H^3 * 9.81

theorem work_required_to_pump
  (R H : ℝ) (gamma : ℝ)
  (hR : R = 3)
  (hH : H = 5)
  (hgamma : gamma = 0.8 * 800) :
  work_pumping_liquid R H gamma = 294300 * pi  :=
by
  sorry

end work_required_to_pump_l409_409391


namespace range_of_a_l409_409538

open scoped Real

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + 2 * x - 2 else 2^(-(abs (1 - x))) - 2

noncomputable def g (a x : ℝ) : ℝ := abs (a - 1) * cos x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, f x1 ≤ g a x2) ↔ (0 ≤ a ∧ a ≤ 2) := by
  sorry

end range_of_a_l409_409538


namespace smallest_digit_never_at_units_place_of_odd_l409_409176

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l409_409176


namespace log_eq_exp_l409_409476

theorem log_eq_exp {x : ℝ} (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end log_eq_exp_l409_409476


namespace jessica_missed_four_games_l409_409730

def total_games : ℕ := 6
def games_attended : ℕ := 2
def missed_games := total_games - games_attended

theorem jessica_missed_four_games : missed_games = 4 := 
by 
  have h : missed_games = total_games - games_attended := rfl
  rw [h]
  have tga : total_games = 6 := rfl
  have ga : games_attended = 2 := rfl
  rw [tga, ga]
  exact Nat.sub_self_add 2
-- sorry

end jessica_missed_four_games_l409_409730


namespace total_stamps_l409_409819

-- Definitions for the conditions.
def snowflake_stamps : ℕ := 11
def truck_stamps : ℕ := snowflake_stamps + 9
def rose_stamps : ℕ := truck_stamps - 13

-- Statement to prove the total number of stamps.
theorem total_stamps : (snowflake_stamps + truck_stamps + rose_stamps) = 38 :=
by
  sorry

end total_stamps_l409_409819


namespace cube_divisibility_l409_409569

theorem cube_divisibility (a : ℤ) (k : ℤ) (h₁ : a > 1) 
(h₂ : (a - 1)^3 + a^3 + (a + 1)^3 = k^3) : 4 ∣ a := 
by
  sorry

end cube_divisibility_l409_409569


namespace smallest_n_perfect_square_and_cube_l409_409352

theorem smallest_n_perfect_square_and_cube (n : ℕ) (h1 : ∃ k : ℕ, 5 * n = k^2) (h2 : ∃ m : ℕ, 4 * n = m^3) :
  n = 1080 :=
  sorry

end smallest_n_perfect_square_and_cube_l409_409352


namespace cos_90_eq_0_l409_409966

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l409_409966


namespace circle_handouts_l409_409445

-- The definition of the problem setup
def is_valid_distribution (n_distr : ℕ) (n_people : ℕ) (handouts : ℕ) (dist : ℕ → ℕ) : Prop :=
  (∀ i, dist i ≤ 1) ∧
  (∑ i in finset.range n_people, dist i) = handouts ∧
  (∀ i, dist i = 0 → dist ((i - 1) % n_people) = 1 ∨ dist ((i + 1) % n_people) = 1)

theorem circle_handouts :
  ∃ (dist : ℕ → ℕ), is_valid_distribution 15 15 6 dist ∧ 
  fintype.card {dist | is_valid_distribution 15 15 6 dist} = 125 :=
sorry

end circle_handouts_l409_409445


namespace equation_AB_equation_BD_l409_409529

/- Definitions of points A, B, and C -/
def A : (ℝ × ℝ) := (0, 4)
def B : (ℝ × ℝ) := (-2, 6)
def C : (ℝ × ℝ) := (-8, 0)

/- Lean function to find the equation of line AB -/
def line_eq_AB : String :=
  let (x1, y1) := A
  let (x2, y2) := B
  let slope := (y2 - y1) / (x2 - x1)
  let intercept := y1 - slope * x1
  s!"y = {slope} * x + {intercept}"

/- Lean function to find the equation of the median BD -/
def line_eq_BD : String :=
  let (x1, y1) := B
  let D := ((A.fst + C.fst) / 2, (A.snd + C.snd) / 2)
  let (x2, y2) := D
  let slope := (y2 - y1) / (x2 - x1)
  let intercept := y1 - slope * x1
  s!"y = {slope} * x + {intercept}"

theorem equation_AB : ∀ x y : ℝ, x + y - 4 = 0 ↔ (y = ((-1) * x + 4)) := 
  by sorry

theorem equation_BD : ∀ x y : ℝ, 2 * y - x - 10 = 0 ↔ (y = ((1 / 2) * x + 5)) := 
  by sorry

end equation_AB_equation_BD_l409_409529


namespace striped_octopus_has_8_legs_l409_409573

-- Definitions for Octopus and Statements
structure Octopus :=
  (legs : ℕ)
  (tellsTruth : Prop)

-- Given conditions translations
def tellsTruthCondition (o : Octopus) : Prop :=
  if o.legs % 2 = 0 then o.tellsTruth else ¬o.tellsTruth

def green_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

def dark_blue_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

def violet_octopus : Octopus :=
  { legs := 9, tellsTruth := sorry }  -- Placeholder truth value

def striped_octopus : Octopus :=
  { legs := 8, tellsTruth := sorry }  -- Placeholder truth value

-- Octopus statements (simplified for output purposes)
def green_statement := (green_octopus.legs = 8) ∧ (dark_blue_octopus.legs = 6)
def dark_blue_statement := (dark_blue_octopus.legs = 8) ∧ (green_octopus.legs = 7)
def violet_statement := (dark_blue_octopus.legs = 8) ∧ (violet_octopus.legs = 9)
def striped_statement := ¬(green_octopus.legs = 8 ∨ dark_blue_octopus.legs = 8 ∨ violet_octopus.legs = 8) ∧ (striped_octopus.legs = 8)

-- The goal to prove that the striped octopus has exactly 8 legs
theorem striped_octopus_has_8_legs : striped_octopus.legs = 8 :=
sorry

end striped_octopus_has_8_legs_l409_409573


namespace smallest_not_odd_unit_is_zero_l409_409185

def is_odd_units (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

theorem smallest_not_odd_unit_is_zero :
  ∀ d : ℕ, (∀ u : ℕ, is_odd_units u → d ≠ u) → d = 0 :=
begin
  intro d,
  intro h,
  -- Proof omitted
  sorry
end

end smallest_not_odd_unit_is_zero_l409_409185


namespace no_x_satisfies_arithmetic_mean_l409_409800

theorem no_x_satisfies_arithmetic_mean :
  ¬ ∃ x : ℝ, (3 + 117 + 915 + 138 + 2114 + x) / 6 = 12 :=
by
  sorry

end no_x_satisfies_arithmetic_mean_l409_409800


namespace cos_90_eq_0_l409_409895

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l409_409895


namespace cos_90_eq_zero_l409_409944

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l409_409944


namespace correct_order_l409_409525

noncomputable 
def f (x : ℝ) : ℝ := sorry -- Define the function

axiom even_f : ∀ x : ℝ, f(x) = f(-x)
axiom domain_f : ∀ x : ℝ, x ∈ ℝ → f(x) = f(x)
axiom mono_increasing_neg : ∀ x y : ℝ, x < y ∧ y < 0 → f(x) < f(y)

theorem correct_order : 
  f (3^(-1/2)) > f (2^(-1/3)) ∧ f (2^(-1/3)) > f (log 2 (1/3)) :=
  sorry

end correct_order_l409_409525


namespace cos_90_eq_zero_l409_409947

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l409_409947


namespace cos_90_eq_0_l409_409873

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l409_409873


namespace smallest_digit_not_in_odd_units_l409_409249

-- Define what it means to be an odd numbers' unit digit
def is_odd_units_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

-- Define the set of even digits 
def is_even_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

-- Prove that 0 is the smallest digit never found in the units place of an odd number
theorem smallest_digit_not_in_odd_units : ∀ d : ℕ, (is_even_digit d ∧ ¬is_odd_units_digit d → d ≥ 0) :=
by 
  intro d
  sorry

end smallest_digit_not_in_odd_units_l409_409249


namespace quadratic_expression_transformation_l409_409582

theorem quadratic_expression_transformation :
  ∀ (a h k : ℝ), (∀ x : ℝ, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 :=
by
  intros a h k h_eq
  sorry

end quadratic_expression_transformation_l409_409582


namespace glycerin_percentage_l409_409384

theorem glycerin_percentage (x : ℝ) 
  (h1 : 100 * 0.75 = 75)
  (h2 : 75 + 75 = 100)
  (h3 : 75 * 0.30 + (x/100) * 75 = 75) : x = 70 :=
by
  sorry

end glycerin_percentage_l409_409384


namespace fraction_left_handed_l409_409816

-- Definitions based on the conditions
def red_participants : ℕ := 5
def blue_participants : ℕ := 5
def total_participants : ℕ := red_participants + blue_participants

def left_handed_red : ℕ := red_participants / 3
def left_handed_blue : ℕ := 2 * blue_participants / 3
def total_left_handed : ℕ := left_handed_red + left_handed_blue

-- The problem statement for the proof
theorem fraction_left_handed :
  total_left_handed.toRat / total_participants.toRat = 1 / 2 := by
  sorry

end fraction_left_handed_l409_409816


namespace smallest_digit_not_in_units_place_of_odd_l409_409286

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l409_409286


namespace trip_time_proof_l409_409061

def driving_speed := 62 -- miles per hour
def distance := 2790 -- miles
def break_time := 0.5 -- hours (30 minutes)
def break_interval := 5 -- hours
def hotel_search_time := 0.5 -- hours (30 minutes)

theorem trip_time_proof :
  let driving_time := distance / driving_speed
  let number_of_breaks := (driving_time / break_interval).toNat - 1
  let total_break_time := number_of_breaks * break_time
  let total_time := driving_time + total_break_time + hotel_search_time
  total_time = 49.5 := by
  -- proof goes here
  sorry

end trip_time_proof_l409_409061


namespace min_value_y_squared_plus_4y_plus_8_min_value_m_squared_plus_2m_plus_3_max_value_neg_m_squared_plus_2m_plus_3_l409_409051

open Real

-- Statement for the first proof problem.
theorem min_value_y_squared_plus_4y_plus_8 (y : ℝ) : 
  ∃ y0, (y^2 + 4*y + 8) ≥ 4 :=
sorry

-- Statement for the second proof problem.
theorem min_value_m_squared_plus_2m_plus_3 (m : ℝ) : 
  ∃ m0, (m^2 + 2*m + 3) ≥ 2 :=
sorry

-- Statement for the third proof problem.
theorem max_value_neg_m_squared_plus_2m_plus_3 (m : ℝ) : 
  ∃ m0, (-(m^2) + 2*m + 3) ≤ 4 :=
sorry

end min_value_y_squared_plus_4y_plus_8_min_value_m_squared_plus_2m_plus_3_max_value_neg_m_squared_plus_2m_plus_3_l409_409051


namespace sin_prob_interval_l409_409567

theorem sin_prob_interval (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ π) :
  (∃ p : ℝ, p = (measure_theory.volume (set.Ioo (π/6) (5*π/6)) / measure_theory.volume (set.Icc 0 π)) ∧ p = 2 / 3) :=
sorry

end sin_prob_interval_l409_409567


namespace number_of_ways_remainder_l409_409667

theorem number_of_ways_remainder :
  let N := (number of ways to insert pluses and minuses in the expression (1 + 2 + 3 + ... + 2016) such that the result is divisible by 2017)
  in N % 503 = 256 :=
sorry

end number_of_ways_remainder_l409_409667


namespace find_x_value_l409_409460

theorem find_x_value (x : ℝ) (h : log 25 (x + 25) = 3 / 2) : x = 100 :=
sorry

end find_x_value_l409_409460


namespace age_ratio_is_2_l409_409782

def man's_age_over_son_ratio_in_two_years (S M : ℕ) (h1 : M = S + 20) (h2 : S = 18) (h3 : ∃ k : ℕ, M + 2 = k * (S + 2)) : Prop :=
  (M + 2) / (S + 2) = 2

theorem age_ratio_is_2 {S M : ℕ} : 
  (M = S + 20) → 
  (S = 18) → 
  (∃ k : ℕ, M + 2 = k * (S + 2)) → 
  man's_age_over_son_ratio_in_two_years S M :=
by
  intro h1 h2 h3
  exact sorry

end age_ratio_is_2_l409_409782


namespace find_matrix_M_l409_409448

def matrix_2x2 (a b c d : ℚ) : matrix (fin 2) (fin 2) ℚ :=
  ![![a, b], ![c, d]]

theorem find_matrix_M :
  ∃ (M : matrix (fin 2) (fin 2) ℚ), 
    M ⬝ ![![2], ![-1]] = ![![5], ![-3]] ∧
    M ⬝ ![![4], ![1]] = ![![11], ![1]] ∧
    M = matrix_2x2 (8/3) (1/3) (-1/3) (7/3) :=
by
  use matrix_2x2 (8/3) (1/3) (-1/3) (7/3)
  have h1 : matrix_2x2 (8/3) (1/3) (-1/3) (7/3) ⬝ ![![2], ![-1]] = ![![5], ![-3]] := by {
    -- Calculation to confirm the multiplication
    sorry,
  }
  have h2 : matrix_2x2 (8/3) (1/3) (-1/3) (7/3) ⬝ ![![4], ![1]] = ![![11], ![1]] := by {
    -- Calculation to confirm the multiplication
    sorry,
  }
  exact ⟨h1, h2, rfl⟩

end find_matrix_M_l409_409448


namespace total_cartons_used_l409_409694

theorem total_cartons_used (x : ℕ) (y : ℕ) (h1 : y = 24) (h2 : 2 * x + 3 * y = 100) : x + y = 38 :=
sorry

end total_cartons_used_l409_409694


namespace distance_to_plane_DBC_l409_409649

-- Let point P lie on the face ABC of a tetrahedron ABCD with edge length 2.
-- The distances from P to the planes DAB, DBC, and DCA form an arithmetic sequence.
-- Prove that the distance from P to the plane DBC is 2sqrt(6)/9.

theorem distance_to_plane_DBC (P : ℝ × ℝ × ℝ) (A B C D : ℝ × ℝ × ℝ)
  (hP_ABC : ∃ a b c: ℝ, a + b + c = 1 ∧ a • A + b • B + c • C = P)
  (length_AB : dist A B = 2) (length_AC : dist A C = 2) (length_BC : dist B C = 2)
  (length_AD : dist A D = 2) (length_BD : dist B D = 2) (length_CD : dist C D = 2)
  (dist_seq : ∃ (d₁ d₂ d₃ : ℝ), (d₂ - d₁) = (d₃ - d₂) 
                              ∧ d₁ + d₂ + d₃ = (2 * real.sqrt 6) / 3
                              ∧ P_to_DAB P D A B = d₁
                              ∧ P_to_DBC P D B C = d₂
                              ∧ P_to_DCA P D C A = d₃) :
  P_to_DBC P D B C = (2 * real.sqrt 6) / 9 := sorry

end distance_to_plane_DBC_l409_409649


namespace sum_of_rectangle_areas_l409_409658

theorem sum_of_rectangle_areas :
  let widths := [1, 3, 5, 7, 9, 11, 13, 15, 17],
      lengths := [nat.succ (nat.succ n * n) | n in list.range 9],
      areas := list.zip_with (*) widths lengths,
      total_area := areas.sum in
  total_area = 3765 :=
by {
  let widths := [1, 3, 5, 7, 9, 11, 13, 15, 17],
  let lengths := [1, 4, 9, 16, 25, 36, 49, 64, 81],
  let areas := list.zip_with (*) widths lengths,
  let total_area := areas.sum,
  exact rfl
} -- sorry

end sum_of_rectangle_areas_l409_409658


namespace smallest_positive_period_l409_409436

noncomputable def function_period : ℝ :=
  let ω := 2
  2 * Real.pi / ω

theorem smallest_positive_period :
  ∀ x : ℝ, (3 * Real.sin(2 * x + Real.pi / 4)) = (3 * Real.sin(2 * (x + function_period) + Real.pi / 4)) :=
by
  sorry

end smallest_positive_period_l409_409436


namespace average_payment_l409_409366

theorem average_payment (n m : ℕ) (p1 p2 : ℕ) (h1 : n = 20) (h2 : m = 45) (h3 : p1 = 410) (h4 : p2 = 475) :
  (20 * p1 + 45 * p2) / 65 = 455 :=
by
  sorry

end average_payment_l409_409366


namespace intersection_sum_l409_409705

theorem intersection_sum 
  (y : ℝ → ℝ)
  (y_eq : ∀ x, y x = x^3 - 5 * x + 4)
  (x y : ℝ → ℝ)
  (x_eq : ∀ y, x y = 2 * y + 2)
  (roots : Finset ℝ)
  (h : ∀ x ∈ roots, y x = x)  -- assuming roots are points of intersection.
  (hx_sum : roots.sum id = 11 / 2)
  (hy_sum : roots.sum (λ x, (x - 2) / 2) = 5 / 4):
  (roots.sum id, roots.sum (λ x, (x - 2) / 2)) = (11 / 2, 5 / 4) := 
begin
 sorry -- Proof not needed 
end

end intersection_sum_l409_409705


namespace quadratic_to_vertex_form_addition_l409_409586

theorem quadratic_to_vertex_form_addition (a h k : ℝ) (x : ℝ) :
  (∀ x, 5 * x^2 - 10 * x - 7 = a * (x - h)^2 + k) → a + h + k = -6 :=
by
  intro h_eq
  sorry

end quadratic_to_vertex_form_addition_l409_409586


namespace cost_to_paint_cube_l409_409699

theorem cost_to_paint_cube (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (side_length : ℝ) (total_cost : ℝ) :
  cost_per_kg = 36.50 →
  coverage_per_kg = 16 →
  side_length = 8 →
  total_cost = (6 * side_length^2 / coverage_per_kg) * cost_per_kg →
  total_cost = 876 :=
by
  intros h1 h2 h3 h4
  sorry

end cost_to_paint_cube_l409_409699


namespace product_sets_not_equal_l409_409109

open BigOperators

-- Noncomputable as we don't have specific computational steps
noncomputable def range_102_to_201 : List ℕ := List.range' 102 100

variable (a : Fin 10 → Fin 10 → ℕ)
variable (rows columns : Fin 10 → ℕ)

def row_product (a : Fin 10 → Fin 10 → ℕ) (i : Fin 10) : ℕ := ∏ (j : Fin 10), a i j

def column_product (a : Fin 10 → Fin 10 → ℕ) (j : Fin 10) : ℕ := ∏ (i : Fin 10), a i j

def are_sets_equal (s1 s2 : Finset ℕ) : Prop :=
  ∀ x : ℕ, x ∈ s1 ↔ x ∈ s2

theorem product_sets_not_equal :
  let P := (Finset.univ.map ⟨row_product a, fun ij => ə⟩ : Finset ℕ)
  let Q := (Finset.univ.map ⟨column_product a, fun ij => ə⟩ : Finset ℕ) in
  ∀ (a : Fin 10 → Fin 10 → ℕ)
    (h1 : ∀ i j, a i j ∈ range_102_to_201)
    (P : Finset ℕ := Finset.univ.map ⟨row_product a, fun ij => ə⟩)
    (Q : Finset ℕ := Finset.univ.map ⟨column_product a, fun ij => ə⟩),
    ¬are_sets_equal P Q :=
by
  sorry

end product_sets_not_equal_l409_409109


namespace smallest_digit_never_in_units_place_of_odd_number_l409_409124

theorem smallest_digit_never_in_units_place_of_odd_number :
  ∀ d : ℕ, (d < 10 ∧ (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0) :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_number_l409_409124


namespace sally_bob_savings_l409_409685

theorem sally_bob_savings :
  let daily_sally := 6
  let daily_bob := 4
  let days_in_year := 365
  let sally_annual := daily_sally * days_in_year
  let bob_annual := daily_bob * days_in_year
  let sally_savings := sally_annual / 2
  let bob_savings := bob_annual / 2
  sally_savings + bob_savings = 1825 :=
by {
  let daily_sally := 6
  let daily_bob := 4
  let days_in_year := 365
  let sally_annual := daily_sally * days_in_year
  let bob_annual := daily_bob * days_in_year
  let sally_savings := sally_annual / 2
  let bob_savings := bob_annual / 2
  have sally_eq : sally_annual = 2190 := rfl
  have bob_eq : bob_annual = 1460 := rfl
  have sally_savings_eq : sally_savings = 1095 := rfl
  have bob_savings_eq : bob_savings = 730 := rfl
  have total_savings : sally_savings + bob_savings = 1825 := rfl
  exact total_savings
}

end sally_bob_savings_l409_409685


namespace h_inverse_l409_409647

def f (x : ℝ) : ℝ := 5 * x - 2
def g (x : ℝ) : ℝ := 3 * x + 7
def h (x : ℝ) : ℝ := f (g x)

def h_inv (x : ℝ) : ℝ := (x - 33) / 15

theorem h_inverse : ∀ x : ℝ, h_inv(h(x)) = x :=
by
  sorry

end h_inverse_l409_409647


namespace train_people_count_l409_409100

theorem train_people_count :
  let initial := 48
  let after_first_stop := initial - 13 + 5
  let after_second_stop := after_first_stop - 9 + 10 - 2
  let after_third_stop := after_second_stop - 7 + 4 - 3
  let after_fourth_stop := after_third_stop - 16 + 7 - 5
  let after_fifth_stop := after_fourth_stop - 8 + 15
  after_fifth_stop = 26 := sorry

end train_people_count_l409_409100


namespace cos_90_eq_0_l409_409960

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l409_409960


namespace smallest_digit_never_in_units_place_of_odd_numbers_l409_409312

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers_l409_409312


namespace minimum_value_of_f_l409_409541

noncomputable def f (x : ℝ) : ℝ := (Real.sin (Real.pi * x) - Real.cos (Real.pi * x) + 2) / Real.sqrt x

theorem minimum_value_of_f :
  ∃ x ∈ Set.Icc (1/4 : ℝ) (5/4 : ℝ), f x = (4 * Real.sqrt 5 / 5 - 2 * Real.sqrt 10 / 5) :=
sorry

end minimum_value_of_f_l409_409541


namespace lines_parallel_to_plane_l409_409512

variables (A B C D : Point) (α : Plane)

-- Assuming A, B, C, D are four non-coplanar points
axiom non_coplanar_A_B_C_D : ¬coplanar A B C D

-- Lean statement: Prove that lines AB and CD may both be parallel to plane α
theorem lines_parallel_to_plane : ∃ (AB CD : Line), 
  line_through_points A B = AB ∧ 
  line_through_points C D = CD ∧ 
  is_parallel_to_plane AB α ∧ 
  is_parallel_to_plane CD α := 
sorry

end lines_parallel_to_plane_l409_409512


namespace excircle_perpendicular_l409_409045

theorem excircle_perpendicular
  (A B C O1 O2 : Point)
  (h1 : is_excenter O1 A B C)
  (h2 : is_excenter O2 A B C)
  (h3 : lies_on_ext_angle_bisector O1 A B)
  (h4 : lies_on_ext_angle_bisector O2 A C)
  (O : Point)
  (h5 : is_incenter O A B C)
  (h6 : lies_on_internal_bisector O A B)
  (h7 : extern_bisector_perpendicular_internal A B C) :
  perpendicular (line_through O1 O2) (line_through O A) :=
sorry

end excircle_perpendicular_l409_409045


namespace cost_of_book_sold_at_loss_l409_409751

theorem cost_of_book_sold_at_loss
  (C1 C2 : ℝ)
  (total_cost : C1 + C2 = 360)
  (selling_price1 : 0.85 * C1 = 1.19 * C2) :
  C1 = 210 :=
sorry

end cost_of_book_sold_at_loss_l409_409751


namespace tangent_line_at_neg1_l409_409076

noncomputable def f : ℝ → ℝ := λ x => 4 * x - x ^ 3

theorem tangent_line_at_neg1 :
  ∀ x : ℝ, y = f'(-1) * (x + 1) - 2 → 
  y = x - 2 :=
by
  intro x y
  -- the proof steps would go here
  sorry

end tangent_line_at_neg1_l409_409076


namespace cos_90_eq_0_l409_409874

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l409_409874


namespace circumscribed_circle_radius_l409_409773

theorem circumscribed_circle_radius (O A B C K : Point) (R : ℝ)
  (hCirc : is_circumscribed_circle O A B C)
  (hTangent : is_tangent_to_circle O C K)
  (hBisector : is_bisector_of_angle B K O C)
  (hAngle : ∠BKC = (3 * ∠A - ∠C) / 2)
  (hSumSides : AC + AB = 2 + sqrt 3)
  (hSumDistances : distance_from_O_to_AC O A C + distance_from_O_to_AB O A B = 2) :
  R = sqrt (34 - 15 * sqrt 3) / 2 :=
by
  sorry

end circumscribed_circle_radius_l409_409773


namespace heptagon_side_length_l409_409732

theorem heptagon_side_length (k : ℝ) : 
  let A1 := k * (6:ℝ)^2,
      A2 := k * (30:ℝ)^2,
      Ax := k * (x:ℝ)^2 in
  A2 - A1 = 144*k ∧ x = 6 * Real.sqrt 5 → x = 6 * Real.sqrt 5 :=
begin
  sorry
end

end heptagon_side_length_l409_409732


namespace peter_age_fraction_l409_409595

theorem peter_age_fraction 
  (harriet_age : ℕ) 
  (mother_age : ℕ) 
  (peter_age_plus_four : ℕ) 
  (harriet_age_plus_four : ℕ) 
  (harriet_age_current : harriet_age = 13)
  (mother_age_current : mother_age = 60)
  (peter_age_condition : peter_age_plus_four = 2 * harriet_age_plus_four)
  (harriet_four_years : harriet_age_plus_four = harriet_age + 4)
  (peter_four_years : ∀ P : ℕ, peter_age_plus_four = P + 4)
: ∃ P : ℕ, P = 30 ∧ P = mother_age / 2 := 
sorry

end peter_age_fraction_l409_409595


namespace necessary_and_not_sufficient_l409_409549

variables {α β : Plane} {l m : Line}

def perp_to (l : Line) (α : Plane) := ∀ p : Point, p ∈ l → p ∈ α → p = p ∧ l = l
def subset_of (m : Line) (β : Plane) := ∀ p : Point, p ∈ m → p ∈ β
def parallel_planes (α β : Plane) := ∀ p₁ p₂ : Point, p₁ ∈ α ∧ p₂ ∈ β → p₁ = p₂ ∨ α = β

theorem necessary_and_not_sufficient (h1 : perp_to l α) (h2 : subset_of m β) :
  (parallel_planes α β → perp_to l m) ∧ ¬ (perp_to l m → parallel_planes α β) := 
by
  sorry

end necessary_and_not_sufficient_l409_409549


namespace average_minutes_heard_l409_409796

theorem average_minutes_heard :
  let total_audience := 200
  let duration := 90
  let percent_entire := 0.15
  let percent_slept := 0.15
  let percent_half := 0.25
  let percent_one_fourth := 0.75
  let total_entire := total_audience * percent_entire
  let total_slept := total_audience * percent_slept
  let remaining := total_audience - total_entire - total_slept
  let total_half := remaining * percent_half
  let total_one_fourth := remaining * percent_one_fourth
  let minutes_entire := total_entire * duration
  let minutes_half := total_half * (duration / 2)
  let minutes_one_fourth := total_one_fourth * (duration / 4)
  let total_minutes_heard := minutes_entire + 0 + minutes_half + minutes_one_fourth
  let average_minutes := total_minutes_heard / total_audience
  average_minutes = 33 :=
by
  sorry

end average_minutes_heard_l409_409796


namespace sum_of_squares_of_roots_l409_409503

-- Define the roots of the quadratic equation
def roots (a b c : ℝ) := { x : ℝ | a * x^2 + b * x + c = 0 }

-- The given quadratic equation is x^2 - 3x - 1 = 0
lemma quadratic_roots_property :
  ∀ x ∈ roots 1 (-3) (-1), x^2 - 3 * x - 1 = 0 :=
by {
  intros x hx,
  unfold roots at hx,
  exact hx,
  sorry
}

-- Using Vieta's formulas and properties of quadratic equations
theorem sum_of_squares_of_roots :
  let x1 := Classical.choose (exists (λ x, roots 1 (-3) (-1) x)),
      x2 := Classical.choose (exists ! (λ x, roots 1 (-3) (-1) x)),
  in x1^2 + x2^2 = 11 :=
by {
  let x1 := 3 / 2 + sqrt 13 / 2,
  let x2 := 3 / 2 - sqrt 13 / 2,
  have h1 : x1 + x2 = 3 := by {
    rw [← add_sub_assoc, add_sub_cancel, div_add_div_same],
    norm_num,
    sorry
  },
  have h2 : x1 * x2 = -1 := by {
    -- Similar proof under Classical logic, left as sorry for brevity
    sorry
  },
  calc
    x1^2 + x2^2 = (x1 + x2)^2 - 2 * (x1 * x2) : by norm_num; field_simp
            ... = 3^2 - 2 * (-1) : by rw [h1, h2]
            ... = 9 + 2 : by norm_num
            ... = 11 : by norm_num
  sorry
}

end sum_of_squares_of_roots_l409_409503


namespace smallest_digit_not_in_units_place_of_odd_l409_409278

theorem smallest_digit_not_in_units_place_of_odd : 
  ∀ (d : ℕ), (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d ≠ 0 → ∃ m, d = m :=
begin
  -- skipped proof
  sorry
end

end smallest_digit_not_in_units_place_of_odd_l409_409278


namespace distinct_integers_count_l409_409428

noncomputable def is_special_fraction (a b : ℕ) : Prop :=
  a + b = 20

noncomputable def special_fractions : set (ℕ × ℕ) :=
  { p | is_special_fraction p.1 p.2 }

noncomputable def possible_sums_products : set ℕ :=
  { n | ∃ p q ∈ special_fractions, n = p.1 * q.2 ∨ n = p.2 * q.1 ∨ n = p.1 + q.1 ∨ n = p.2 + q.2 }

theorem distinct_integers_count : 
  finset.card (finset.attach possible_sums_products.to_finset) = 12 :=
sorry

end distinct_integers_count_l409_409428


namespace exponentiation_identity_l409_409827

theorem exponentiation_identity (x : ℝ) : (-x^7)^4 = x^28 := 
sorry

end exponentiation_identity_l409_409827


namespace ram_distance_on_map_l409_409031

theorem ram_distance_on_map :
  let map_distance_mountains := 312
  let actual_distance_mountains := 136 * 39370.1
  let ram_actual_distance := 14.82 * 39370.1
  let scale := actual_distance_mountains / map_distance_mountains
  (ram_actual_distance / scale : ℝ) ≈ 34 :=
by
  sorry

end ram_distance_on_map_l409_409031


namespace fried_chicken_total_l409_409625

-- The Lean 4 statement encapsulates the problem conditions and the correct answer
theorem fried_chicken_total :
  let kobe_initial := 5
  let pau_initial := 2 * kobe_initial
  let another_set := 2
  pau_initial * another_set = 20 :=
by
  let kobe_initial := 5
  let pau_initial := 2 * kobe_initial
  let another_set := 2
  show pau_initial * another_set = 20
  sorry

end fried_chicken_total_l409_409625


namespace compare_fractions_l409_409843

theorem compare_fractions : (-1 / 3) > (-1 / 2) := sorry

end compare_fractions_l409_409843


namespace smallest_missing_digit_l409_409322

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

def odd_units_digits : set ℕ :=
  {1, 3, 5, 7, 9}

def all_digits : set ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def missing_digits (s1 s2 : set ℕ) : set ℕ :=
  s1 \ s2

theorem smallest_missing_digit :
  ∃ (d : ℕ), d ∈ missing_digits all_digits odd_units_digits ∧ 
  ∀ (x : ℕ), x ∈ missing_digits all_digits odd_units_digits → d ≤ x :=
sorry

end smallest_missing_digit_l409_409322


namespace simplify_evaluate_l409_409057

theorem simplify_evaluate :
  ∀ (x : ℝ), x = Real.sqrt 2 - 1 →
  ((1 + 4 / (x - 3)) / ((x^2 + 2*x + 1) / (2*x - 6))) = Real.sqrt 2 :=
by
  intros x hx
  sorry

end simplify_evaluate_l409_409057


namespace smallest_digit_never_at_units_place_of_odd_l409_409163

theorem smallest_digit_never_at_units_place_of_odd :
  ∀ (n : ℕ), digit_units n ∈ {0, 2, 4, 6, 8} ∧
             (∀ d, d ∈ {0, 2, 4, 6, 8} → d ≥ 0) →
             digit_units n = 0 :=
by
  sorry  -- Proof omitted

end smallest_digit_never_at_units_place_of_odd_l409_409163


namespace chord_length_l409_409768

theorem chord_length (r d : ℝ) (h_r : r = 5) (h_d : d = 4) : 
  ∃ (EF : ℝ), EF = 6 :=
by
  have h1 : r^2 = d^2 + (EF / 2)^2,
  sorry

end chord_length_l409_409768


namespace smallest_digit_not_in_units_place_of_odd_l409_409253

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → 
  (∀ odd_digit : ℕ, odd_digit ∈ {1, 3, 5, 7, 9} → d ≠ odd_digit) → 
  d = 0 := 
by
  sorry

end smallest_digit_not_in_units_place_of_odd_l409_409253


namespace find_phi_l409_409637

theorem find_phi (φ : ℝ) (s : ℝ) (Q : ℂ) : 
  (0 < s) → (0 ≤ φ ∧ φ < 360) → 
  (Q = s * (complex.cos (complex.of_real φ).re + complex.sin (complex.of_real φ).im * complex.I)) →
  (∀ z : ℂ, (polynomial.eval z (polynomial.map complex.of_real (polynomial.C_rat 1 + polynomial.X * 
  (polynomial.C_rat (-1) + polynomial.X * 
  (polynomial.C_rat (-1) + polynomial.X * 
  (polynomial.C_rat (-1) + polynomial.X * polynomial.X)))))) = 0 → 
  (z.im > 0)) → 
  φ = 120 := 
by
  sorry

end find_phi_l409_409637


namespace cos_90_eq_0_l409_409884

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l409_409884


namespace seq_values_geo_seq_arith_max_t_l409_409506

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (b : ℕ → ℝ)

axiom a_pos (n : ℕ) : a n > 0
axiom S_sum (n : ℕ) : S n = (finset.range n).sum a
axiom a1 : a 1 = 2
axiom eq_cond (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : (1/2 * S (m+n) + 1)^2 = a (2*m) * (1 + 1/2 * S (2*n))

noncomputable def b_n (n : ℕ) : ℕ := (a n) - ((-1 : ℕ) ^ n)

theorem seq_values : a 2 = 4 ∧ a 3 = 8 ∧ a 4 = 16 := sorry

theorem geo_seq : ∀ n : ℕ, a n = 2^n := sorry

theorem arith_max_t : ∀ t : ℕ, (∀ n : ℕ, 0 < n → b n = a n - (-1)^n) → (∃ t ≤ 3, (b (finset.range t).sum) = 3) := sorry

end seq_values_geo_seq_arith_max_t_l409_409506


namespace gcd_1987_1463_l409_409739

open Int

-- Definitions of the conditions
def a : ℕ := 1987
def b : ℕ := 1463

-- The proof problem statement
theorem gcd_1987_1463 : gcd a b = 1 := by
  rw [gcd_rec]
  -- Placeholder for a more complex proof
  sorry

end gcd_1987_1463_l409_409739


namespace cos_90_eq_0_l409_409956

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l409_409956


namespace necessary_but_not_sufficient_condition_l409_409602

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

noncomputable def condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ^ 2 = a n * a (n + 2)

theorem necessary_but_not_sufficient_condition
  (a : ℕ → ℝ) :
  condition a → ¬ is_geometric_sequence a :=
sorry

end necessary_but_not_sufficient_condition_l409_409602


namespace kevin_age_l409_409736

theorem kevin_age (x : ℕ) :
  (∃ n : ℕ, x - 2 = n^2) ∧ (∃ m : ℕ, x + 2 = m^3) → x = 6 :=
by
  sorry

end kevin_age_l409_409736


namespace additional_track_length_l409_409792

theorem additional_track_length (h : ℝ) (g1 g2 : ℝ) (L1 L2 : ℝ)
  (rise_eq : h = 800) 
  (orig_grade : g1 = 0.04) 
  (new_grade : g2 = 0.025) 
  (L1_eq : L1 = h / g1) 
  (L2_eq : L2 = h / g2)
  : (L2 - L1 = 12000) := 
sorry

end additional_track_length_l409_409792


namespace cos_90_eq_zero_l409_409975

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l409_409975


namespace shortest_path_proof_l409_409631

-- Function definition
def f (x : ℝ) : ℝ := x^2 / 8

-- Definitions of points
def start_point : ℝ × ℝ := (7, 3)

-- Define the expression for the length of the shortest path
def shortest_path_length : ℝ := 5 * Real.sqrt 2 - 2

theorem shortest_path_proof :
  ∃ a : ℝ, let Q := (a, f a) in
  let R := (a, 0) in
  (dist start_point Q + dist Q R) = shortest_path_length :=
by
  sorry

end shortest_path_proof_l409_409631


namespace cos_90_eq_zero_l409_409970

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l409_409970


namespace range_of_a_l409_409381

def p (a : ℝ) := ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0
def q (a : ℝ) := ∃ x₀ : ℝ, x₀^2 + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 :=
by
  sorry

end range_of_a_l409_409381


namespace ratio_OC_OA_l409_409597

-- Frame the conditions using Lean definitions
def rectangle (A B C D : Type) : Prop := sorry
def midpoint (M C D : Type) : Prop := sorry
def intersection (O AC BM : Type) : Prop := sorry

-- Given the conditions
variables (A B C D M O : Type)
variables (h1 : rectangle A B C D)
variables (h2 : AB = 8)
variables (h3 : AD = 3)
variables (h4 : midpoint M C D)
variables (h5 : intersection O AC BM)

-- Required to prove the ratio
theorem ratio_OC_OA : (OC / OA) = 2 / 3 :=
by sorry

end ratio_OC_OA_l409_409597


namespace total_brown_mms_3rd_4th_bags_l409_409676

def brown_mms_in_bags := (9 : ℕ) + (12 : ℕ) + (3 : ℕ)

def total_bags := 5

def average_mms_per_bag := 8

theorem total_brown_mms_3rd_4th_bags (x y : ℕ) 
  (h1 : brown_mms_in_bags + x + y = average_mms_per_bag * total_bags) : 
  x + y = 16 :=
by
  have h2 : brown_mms_in_bags + x + y = 40 := by sorry
  sorry

end total_brown_mms_3rd_4th_bags_l409_409676


namespace smallest_n_perfect_square_and_cube_l409_409351

theorem smallest_n_perfect_square_and_cube (n : ℕ) (h1 : ∃ k : ℕ, 5 * n = k^2) (h2 : ∃ m : ℕ, 4 * n = m^3) :
  n = 1080 :=
  sorry

end smallest_n_perfect_square_and_cube_l409_409351


namespace smallest_positive_integer_n_l409_409356

theorem smallest_positive_integer_n (n : ℕ) 
  (h1 : ∃ k : ℕ, n = 5 * k ∧ perfect_square(5 * k)) 
  (h2 : ∃ m : ℕ, n = 4 * m ∧ perfect_cube(4 * m)) : 
  n = 625000 :=
sorry

end smallest_positive_integer_n_l409_409356


namespace cube_surface_area_l409_409098

-- We will define the conditions given in the problem and then state the proof problem
def num_dice := 27
def side_length := 3

-- Define the theorem to prove the surface area of the larger cube
theorem cube_surface_area : 
    let cube_root := Int.cbrt num_dice in
    let edge_length := cube_root * side_length in
    let face_area := edge_length ^ 2 in
    let total_surface_area := 6 * face_area in
    total_surface_area = 486 := 
by
  -- Proof steps go here
  sorry

end cube_surface_area_l409_409098


namespace determine_S5_l409_409377

noncomputable def S (x : ℝ) (m : ℕ) : ℝ := x^m + 1 / x^m

theorem determine_S5 (x : ℝ) (h : x + 1 / x = 3) : S x 5 = 123 :=
by
  sorry

end determine_S5_l409_409377
