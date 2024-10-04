import Mathlib

namespace width_of_field_l456_456600

noncomputable def field_width 
  (field_length : ℝ) 
  (rope_length : ℝ)
  (grazing_area : ℝ) : ℝ :=
if field_length > 2 * rope_length 
then rope_length
else grazing_area

theorem width_of_field 
  (field_length : ℝ := 45)
  (rope_length : ℝ := 22)
  (grazing_area : ℝ := 380.132711084365) : field_width field_length rope_length grazing_area = rope_length :=
by 
  sorry

end width_of_field_l456_456600


namespace f_odd_f_monotonic_increasing_intervals_f_no_max_value_f_extreme_points_l456_456384

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem f_odd (x : ℝ) : f (-x) = -f (x) :=
by sorry

theorem f_monotonic_increasing_intervals :
  ∀ x : ℝ, (x < -Real.sqrt 3 / 3 ∨ x > Real.sqrt 3 / 3) → f x' > f x :=
by sorry

theorem f_no_max_value :
  ∀ x : ℝ, ¬(∃ M, f x ≤ M) :=
by sorry

theorem f_extreme_points :
  f (-Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 ∧ f (Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 :=
by sorry

end f_odd_f_monotonic_increasing_intervals_f_no_max_value_f_extreme_points_l456_456384


namespace flat_distance_4_km_l456_456578

-- Define variables and constants
variables (x : ℝ) (total_time : ℝ) (distance_AB : ℝ)
variables (speed_uphill : ℝ) (speed_flat : ℝ) (speed_downhill : ℝ)

-- Assign values to constants
def total_time : ℝ := 221 / 60
def distance_AB : ℝ := 9
def speed_uphill : ℝ := 4
def speed_flat : ℝ := 5
def speed_downhill : ℝ := 6

-- Define the problem as a theorem
theorem flat_distance_4_km :
  (∃ x : ℝ, 
   (speed_uphill > 0 ∧ speed_flat > 0 ∧ speed_downhill > 0 ∧ distance_AB > 0 ∧ total_time > 0) ∧
   (2 * x) / speed_flat + (distance_AB - x) / speed_uphill + (distance_AB - x) / speed_downhill = total_time) → x = 4 := 
sorry

end flat_distance_4_km_l456_456578


namespace f_increasing_in_neg_infinity_to_neg_one_l456_456337

noncomputable def g (a x : ℝ) : ℝ := log a (abs (x + 1))

noncomputable def f (a x : ℝ) : ℝ := a^(abs (x + 1))

theorem f_increasing_in_neg_infinity_to_neg_one (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
    (h3 : ∀ x, -1 < x ∧ x < 0 → log a (x + 1) > 0) :
    ∀ x1 x2, x1 < x2 ∧ x1 < -1 ∧ x2 < -1 → f a x1 < f a x2 := 
by
  sorry

end f_increasing_in_neg_infinity_to_neg_one_l456_456337


namespace distance_after_four_steps_l456_456258

theorem distance_after_four_steps (total_distance : ℝ) (steps : ℕ) (steps_taken : ℕ) :
   total_distance = 25 → steps = 7 → steps_taken = 4 → (steps_taken * (total_distance / steps) = 100 / 7) :=
by
    intro h1 h2 h3
    rw [h1, h2, h3]
    simp
    sorry

end distance_after_four_steps_l456_456258


namespace roots_of_quadratic_l456_456185

theorem roots_of_quadratic (x : ℝ) : (5 * x^2 = 4 * x) → (x = 0 ∨ x = 4 / 5) :=
by
  sorry

end roots_of_quadratic_l456_456185


namespace octagon_area_ratio_l456_456272

theorem octagon_area_ratio (r : ℝ) : 
  let S := (1 : ℝ) in -- Scale factor for inscribed octagon side
  let C := (√2 : ℝ) in -- Scale factor for circumscribed octagon side
  let area_inscribed := 2 * (1 + √2) * r^2 in -- Inscribed octagon
  let area_circumscribed := 2 * (1 + √2) * (r * √2)^2 in -- Circumscribed octagon
  area_circumscribed / area_inscribed = 2 :=
by
  sorry

end octagon_area_ratio_l456_456272


namespace vector_problem_l456_456050

-- Definitions for vectors
def m (a b : ℝ) : ℝ × ℝ := (a, b^2 - b + 7/3)
def n (a b : ℝ) : ℝ × ℝ := (a + b + 2, 1)
def mu : ℝ × ℝ := (2, 1)

-- Main theorem statement
theorem vector_problem (a b : ℝ) :
  (∃ b, ∃ a, m a b = 2 * mu) →
  a >= 25 / 6 ∧
  (∀ a b, (let (m₁, m₂) := m a b in
   let (n₁, n₂) := n a b in
   m₁ * n₁ + m₂ * n₂ >= 0)) :=
by
  assume h
  sorry

end vector_problem_l456_456050


namespace building_distances_l456_456608

def angle_in_degrees := ℝ

-- Conditions
def A : angle_in_degrees := 45
def D : ℝ := 400
def B : angle_in_degrees := 15
def B' : angle_in_degrees := 75

noncomputable def cos_deg (x : angle_in_degrees) : ℝ := Real.cos (x * Real.pi / 180)
noncomputable def tan_deg (x : angle_in_degrees) : ℝ := Real.tan (x * Real.pi / 180)

-- Calculations
noncomputable def x_distance : ℝ := D * tan_deg B
noncomputable def y_distance : ℝ := D * tan_deg B'
noncomputable def buildings_distance : ℝ := Real.sqrt (x_distance^2 + y_distance^2 + 2 * x_distance * y_distance * cos_deg 60)

theorem building_distances:
  x_distance ≈ 84.53 ∧
  y_distance ≈ 315.47 ∧
  buildings_distance ≈ 326.6 :=
by
  sorry

end building_distances_l456_456608


namespace orthocenter_condition_l456_456445

theorem orthocenter_condition 
  {A B C D : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
  (DA DB DC AB BC CA : ℝ)
  (h : DA * DB * AB + DB * DC * BC + DC * DA * CA = AB * BC * CA) :
    is_orthocenter D A B C :=
sorry

end orthocenter_condition_l456_456445


namespace prime_factors_le_3_l456_456654

theorem prime_factors_le_3 :
  ∀ a b c : ℕ, 0 < a → a < b → b < c → b - a = c - b → 
  (∀ x ∈ {a, b, c}, ∀ p, prime p → p ∣ x → p = 2 ∨ p = 3) → 
  ∃ k m n : ℕ, (k = 2^m * 3^n) ∧ ((a = k ∧ b = 2*k ∧ c = 3*k) ∨
                                     (a = 2*k ∧ b = 3*k ∧ c = 4*k) ∨
                                     (a = 2*k ∧ b = 9*k ∧ c = 16*k)) := 
by
  sorry

end prime_factors_le_3_l456_456654


namespace sum_mult_base8_l456_456316

theorem sum_mult_base8 : 
  let sum_10 := (1 + 2 + 3 + ⋯ + 24)
  let product_10 := sum_10 * 3
  let result_8 := base10_to_base8 product_10 -- you will need to define base10_to_base8
  result_8 = 1604 := by
  sorry

-- Assumed definitions that must be provided to complete the theorem.
def base10_to_base8 (n : ℕ) : ℕ := 
  sorry

end sum_mult_base8_l456_456316


namespace m_value_l456_456297

open Polynomial

noncomputable def f (m : ℚ) : Polynomial ℚ := X^4 - 5*X^2 + 4*X - C m

theorem m_value (m : ℚ) : (2 * X + 1) ∣ f m ↔ m = -51/16 := by sorry

end m_value_l456_456297


namespace equal_volumes_l456_456523

variable (A B C A1 B1 C1 Q P : Type) 
variable (t T m : ℝ)
variable [plane_geometry A B C]
variable [plane_geometry A1 B1 C1]
variable [point_on_plane P (plane A1 B1 C1)]
variable [parallel_planes (plane B C P) (plane B1 C1)]
variable [parallel_planes (plane C A P) (plane C1 A1)]
variable [parallel_planes (plane A B P) (plane A1 B1)]
variable [common_point Q [plane (B1 C1 P), plane (C1 A1 P), plane (A1 B1 P)]]

theorem equal_volumes :
    volume_pyramid A B B1 A1 C = volume_tetrahedron A B C Q := sorry

end equal_volumes_l456_456523


namespace find_abbbb_l456_456048

theorem find_abbbb (a b : ℕ) (h₁ : a ≠ b) (h₂ : a = 9) (h₃ : b = 7) : 
  let n := 10000 * a + 1111 * b in
  n = 97777 ∧ (n ^ 2 - 1) < 10000000000 ∧ (n ^ 2 - 1) ≥ 1000000000 ∧ (∃ d : finset ℕ, (∀ x ∈ d, x < 10) ∧ d.card = 10) :=
by 
  let n := 10000 * a + 1111 * b;
  have h₄ : n = 97777 := sorry;
  have h₅ : (n ^ 2 - 1) < 10000000000 := sorry;
  have h₆ : (n ^ 2 - 1) ≥ 1000000000 := sorry;
  have h_d : ∃ d : finset ℕ, (∀ x ∈ d, x < 10) ∧ d.card = 10 := sorry;
  exact ⟨h₄, h₅, h₆, h_d⟩;

end find_abbbb_l456_456048


namespace sum_powers_up_to_7_l456_456865

-- Define the complex number (2 + i)
def c : ℂ := 2 + complex.I

-- Define the sum of powers of c from 0 to 7
def sum_powers (n : ℕ) : ℂ :=
  if n ≤ 7 then 
    ∑ k in finset.range (n + 1), c^k
  else 
    0

-- State the theorem that the sum from 0 to 7 is the expected result
theorem sum_powers_up_to_7 :
  sum_powers 7 = (2 + complex.I)^0 + (2 + complex.I)^1 + (2 + complex.I)^2 + (2 + complex.I)^3 + (2 + complex.I)^4 + (2 + complex.I)^5 + (2 + complex.I)^6 + (2 + complex.I)^7 :=
by sorry

end sum_powers_up_to_7_l456_456865


namespace radical_axis_of_intersecting_circles_l456_456149

-- Definition of the power of a point with respect to a circle
def power_of_point (A : Point) (O : Point) (R : ℝ) : ℝ :=
  (dist A O)^2 - R^2

-- Definition of the radical axis condition
def radical_axis_condition (P Q : Point) (O1 O2 : Point) (R1 R2 : ℝ) : Prop :=
  power_of_point P O1 R1 = 0 ∧ power_of_point P O2 R2 = 0 ∧ 
  power_of_point Q O1 R1 = 0 ∧ power_of_point Q O2 R2 = 0

-- The Lean statement proving the radical axis passes through points of intersection
theorem radical_axis_of_intersecting_circles 
  (O1 O2 P Q : Point) (R1 R2 : ℝ) 
  (h1 : dist P O1 = R1) (h2 : dist P O2 = R2)
  (h3 : dist Q O1 = R1) (h4 : dist Q O2 = R2) :
  radical_axis_condition P Q O1 O2 R1 R2 :=
by {
  sorry
}

end radical_axis_of_intersecting_circles_l456_456149


namespace quad_area_l456_456265

/-- A quadrilateral in the coordinate plane has vertices whose y-coordinates are 
0, 3, 6, and 9. The figure is a rectangle combined with a semicircle on one of the 
longer sides. Prove that the area of the figure is 24 + 2π square units. -/
theorem quad_area : 
  (∀ x₁ x₂ y₁ y₂ y₃ y₄ : ℝ, (y₁ = 0) → (y₂ = 3) → (y₃ = 6) → (y₄ = 9) → 
  ∃ x : ℝ, 
    let rectangle_area := 6 * x,
        semicircle_area := (π * x^2) / 8
    in rectangle_area + semicircle_area = 24 + 2 * π) := 
by { sorry }

end quad_area_l456_456265


namespace carter_stretching_legs_frequency_l456_456289

-- Given conditions
def tripDuration : ℤ := 14 * 60 -- in minutes
def foodStops : ℤ := 2
def gasStops : ℤ := 3
def pitStopDuration : ℤ := 20 -- in minutes
def totalTripDuration : ℤ := 18 * 60 -- in minutes

-- Prove that Carter stops to stretch his legs every 2 hours
theorem carter_stretching_legs_frequency :
  ∃ (stretchingStops : ℤ), (totalTripDuration - tripDuration = (foodStops + gasStops + stretchingStops) * pitStopDuration) ∧
    (stretchingStops * pitStopDuration = totalTripDuration - (tripDuration + (foodStops + gasStops) * pitStopDuration)) ∧
    (14 / stretchingStops = 2) :=
by sorry

end carter_stretching_legs_frequency_l456_456289


namespace sufficient_condition_parallel_line_plane_l456_456063

variables {Line Plane : Type}
variables (m n : Line) (alpha beta : Plane)

-- Define non-overlapping lines and planes
def non_overlapping_lines (m n : Line) : Prop := m ≠ n
def non_overlapping_planes (alpha beta : Plane) : Prop := alpha ≠ beta

-- Define parallelism and intersections in abstract way for lines and planes
def line_parallel_line (m n : Line) : Prop := ∀ p : Line, (m = p ∨ n = p) → m ≠ n
def line_parallel_plane (m : Line) (alpha : Plane) : Prop := ∀ q : Plane, q = alpha → ∀ l : Line, l ∈ q → m ≠ l
def plane_intersects_plane (alpha beta : Plane) (n : Line) : Prop := ∃ l : Line, l ∈ alpha ∧ l ∈ beta ∧ l = n
def line_not_in_plane (m : Line) (alpha : Plane) : Prop := ∀ p : Plane, p = alpha → m ∉ p

-- Define the proof statement
theorem sufficient_condition_parallel_line_plane 
  (h1 : plane_intersects_plane alpha beta n) 
  (h2 : line_not_in_plane m alpha) 
  (h3 : line_parallel_line m n) :
  line_parallel_plane m alpha :=
sorry

end sufficient_condition_parallel_line_plane_l456_456063


namespace imaginary_part_of_complex_expression_l456_456068

def imaginary_part (z : ℂ) : ℝ :=
  z.im

theorem imaginary_part_of_complex_expression :
  let z : ℂ := (2 - I) * I
  in imaginary_part z = 2 := by
  sorry

end imaginary_part_of_complex_expression_l456_456068


namespace fill_blank_with_any_l456_456524

-- This represents the conditions
def sentence_conditions (word : String) : Prop :=
  (word = "each" → False) ∧     -- "each" does not fit
  (word = "some" → False) ∧     -- "some" does not fit
  (word = "certain" → False) ∧  -- "certain" does not fit
  (word = "any" → True)         -- "any" is the correct fit

-- Define the theorem to prove
theorem fill_blank_with_any : ∃ word, sentence_conditions word ∧ word = "any" :=
by
  existsi "any"
  split
  exact ⟨λ h, by contradiction, λ h, by contradiction, λ h, by contradiction, by trivial⟩
  rfl

end fill_blank_with_any_l456_456524


namespace octagon_area_ratio_l456_456274

theorem octagon_area_ratio (r : ℝ) : 
  let S := (1 : ℝ) in -- Scale factor for inscribed octagon side
  let C := (√2 : ℝ) in -- Scale factor for circumscribed octagon side
  let area_inscribed := 2 * (1 + √2) * r^2 in -- Inscribed octagon
  let area_circumscribed := 2 * (1 + √2) * (r * √2)^2 in -- Circumscribed octagon
  area_circumscribed / area_inscribed = 2 :=
by
  sorry

end octagon_area_ratio_l456_456274


namespace remainder_T_mod_500_l456_456111

def R3 (n : ℕ) : ℕ := (3 ^ n) % 500

def unique_remainders_set : finset ℕ :=
  finset.range 500 |>.filter (λ n, ∃ k, (3 ^ k) % 500 = n)

def T : ℕ := unique_remainders_set.sum id

theorem remainder_T_mod_500 : 
  (T % 500) = ?answer :=
by
  sorry

end remainder_T_mod_500_l456_456111


namespace sum_of_divisors_not_divisible_by_2_of_540_l456_456923

open Finset

noncomputable def divisors_not_divisible_by_2 : Finset ℕ :=
  (range 4).product (range 2) |>.image (λ (p : ℕ × ℕ), 3^p.1 * 5^p.2)

noncomputable def sum_divisors_not_divisible_by_2 : ℕ :=
  (divisors_not_divisible_by_2).sum id

theorem sum_of_divisors_not_divisible_by_2_of_540 : sum_divisors_not_divisible_by_2 = 240 := by
  sorry

end sum_of_divisors_not_divisible_by_2_of_540_l456_456923


namespace no_function_satisfying_condition_l456_456855

theorem no_function_satisfying_condition :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f(f(n)) = n + 1987 :=
by
  sorry

end no_function_satisfying_condition_l456_456855


namespace sum_of_sequence_l456_456891

open Nat

def sequence (n : ℕ) : (ℕ → ℝ) → Prop :=
  λ a, ∀ (n > 0), a n + a (n + 1) = n

def defined_sum (a : ℕ → ℝ) (s : ℕ) := (s > 0) → a 1 = 1 ∧ ∑ i in range (s + 1), a i = 105.5

theorem sum_of_sequence :
  ∃ a : ℕ → ℝ, sequence a 21 ∧ defined_sum a 21 :=
by
  sorry

end sum_of_sequence_l456_456891


namespace average_pages_per_book_l456_456094

-- Conditions
def book_thickness_in_inches : ℕ := 12
def pages_per_inch : ℕ := 80
def number_of_books : ℕ := 6

-- Given these conditions, we need to prove the average number of pages per book is 160.
theorem average_pages_per_book (book_thickness_in_inches : ℕ) (pages_per_inch : ℕ) (number_of_books : ℕ)
  (h1 : book_thickness_in_inches = 12)
  (h2 : pages_per_inch = 80)
  (h3 : number_of_books = 6) :
  (book_thickness_in_inches * pages_per_inch) / number_of_books = 160 := by
  sorry

end average_pages_per_book_l456_456094


namespace vector_problem_l456_456677

def vec_a := (2, -1 : Int × Int)
def vec_b := (1, 3 : Int × Int)

theorem vector_problem :
  -2 * vec_a.1 + 3 * vec_b.1 = -1 ∧ -2 * vec_a.2 + 3 * vec_b.2 = 11 := by
  sorry

end vector_problem_l456_456677


namespace ratio_of_areas_is_one_l456_456844

variable {α : Type*}

-- Given conditions:
variables (A B C M D P Q R S : α)
variables (hABC : Triangle ABC)  -- Triangle ABC is acute-angled
variables (hM_on_AB : PointOnLineSegment M A B)  -- Point M on side AB
variables (hD_inside : InsideTriangle D ABC)  -- Point D inside triangle ABC
variables (ω_A ω_B : Circumcircle α)  -- Circles ω_A and ω_B
variables (hω_A : CircumcircleOfTriangle ω_A A M D)  -- Circle ω_A circumscribed around triangle AMD
variables (hω_B : CircumcircleOfTriangle ω_B B M D)  -- Circle ω_B circumscribed around triangle BMD
variables (hP_second_time : IntersectsSecondTime AC ω_A P)  -- AC intersects ω_A second time at P
variables (hQ_second_time : IntersectsSecondTime BC ω_B Q)  -- BC intersects ω_B second time at Q
variables (hR_PD_ω_B : IntersectsSecondTime (Ray PD) ω_B R)  -- PD intersects ω_B second time at R
variables (hS_QD_ω_A : IntersectsSecondTime (Ray QD) ω_A S)  -- QD intersects ω_A second time at S

-- Conclude:
theorem ratio_of_areas_is_one
  (h_parallel_BR_AC : Parallel BR AC)
  (h_parallel_AS_BC : Parallel AS BC)
  : AreaOfTriangle ACR = AreaOfTriangle BCS :=
sorry

end ratio_of_areas_is_one_l456_456844


namespace red_cars_in_lot_l456_456533

theorem red_cars_in_lot (B : ℕ) (hB : B = 90) (ratio_condition : 3 * B = 8 * R) : R = 33 :=
by
  -- Given
  have h1 : B = 90 := hB
  have h2 : 3 * B = 8 * R := ratio_condition

  -- To solve
  sorry

end red_cars_in_lot_l456_456533


namespace find_m_l456_456744

theorem find_m (m : ℕ) (h : m * (m - 1) * (m - 2) * (m - 3) * (m - 4) = 2 * m * (m - 1) * (m - 2)) : m = 5 :=
sorry

end find_m_l456_456744


namespace geometric_sequence_property_l456_456777

theorem geometric_sequence_property (n : ℕ) (h₀ : n < 17) (hn : n > 0)
  (b : ℕ → ℝ) (h₁ : b 9 = 1) :
  (∏ i in finset.range n, b (i + 1)) = (∏ i in finset.range (17 - n), b (i + 1)) :=
sorry

end geometric_sequence_property_l456_456777


namespace area_quadrilateral_l456_456426

theorem area_quadrilateral (
  A D E H B C G F W X Y Z : Point 
  (trisect_AD : trisect A D B C)
  (trisect_HE : trisect H E G F)
  (rectangle_ADEH : rectangle A D E H)
  (AH_eq : distance A H = 2)
  (AC_eq : distance A C = 2)
  (AF_eq : line A F)
  (BE_eq : line B E)
  (DG_eq : line D G)
  (CH_eq : line C H)
  (intersection_W : intersect (line B E) (line C H) = W)
  (intersection_X : intersect (line B E) (line D G) = X)
  (intersection_Y : intersect (line A F) (line D G) = Y)
  (intersection_Z : intersect (line A F) (line C H) = Z)
) :
  area_quadrilateral W X Y Z = 1 / 2 :=
by
  have intersections := sorry -- computed intersection points
  have shoelace_formula := sorry -- area computed using Shoelace Theorem
  exact shoelace_formula
  sorry

end area_quadrilateral_l456_456426


namespace greatest_int_less_than_M_div_100_l456_456359

theorem greatest_int_less_than_M_div_100 (M : ℕ) : 
  (fraction_sum M -> 
   let k := M / 100 in 
   k = 5242) :=
by
  intro h
  sorry

where fraction_sum (M : ℕ) : Prop :=
  M = (Nat.factorial 20) * (
    (1 / (fact 1 * fact 19)) +
    (1 / (fact 2 * fact 18)) +
    (1 / (fact 3 * fact 17)) +
    (1 / (fact 4 * fact 16)) +
    (1 / (fact 5 * fact 15)) +
    (1 / (fact 6 * fact 14)) +
    (1 / (fact 7 * fact 13)) +
    (1 / (fact 8 * fact 12)) +
    (1 / (fact 9 * fact 11)) +
    (1 / (fact 10 * fact 10)))

end greatest_int_less_than_M_div_100_l456_456359


namespace scientific_notation_of_star_diameter_l456_456257

theorem scientific_notation_of_star_diameter:
    (∃ (c : ℝ) (n : ℕ), 1 ≤ c ∧ c < 10 ∧ 16600000000 = c * 10^n) → 
    16600000000 = 1.66 * 10^10 :=
by
  sorry

end scientific_notation_of_star_diameter_l456_456257


namespace probability_correct_l456_456597

noncomputable def hexagon_side_length (s : ℝ) := s

def area_of_center_square (s : ℝ) := (s / Real.sqrt 3) ^ 2

def area_of_regular_hexagon (s : ℝ) := (3 * Real.sqrt 3 / 2) * s ^ 2

def probability_dart_in_center_square (s : ℝ) : ℝ := 
  (area_of_center_square s) / (area_of_regular_hexagon s)

theorem probability_correct (s : ℝ) (hs : s > 0) : 
  probability_dart_in_center_square s = 2 * Real.sqrt 3 / 27 :=
by
  sorry

end probability_correct_l456_456597


namespace intersection_P_Q_l456_456014

-- Definitions based on conditions
def P : Set ℝ := { y | ∃ x : ℝ, y = x + 1 }
def Q : Set ℝ := { y | ∃ x : ℝ, y = 1 - x }

-- Proof statement to show P ∩ Q = Set.univ
theorem intersection_P_Q : P ∩ Q = Set.univ := by
  sorry

end intersection_P_Q_l456_456014


namespace polynomial_form_of_divisibility_l456_456574

-- Definitions of the conditions in the problem
variables (p : Polynomial ℝ) (a b : ℕ)
variable [Fact (0 < a)] -- a is a positive integer
variable [Fact (0 < b)] -- b is a positive integer

noncomputable def r : Polynomial ℝ := Polynomial.derivative p

-- State the theorem
theorem polynomial_form_of_divisibility (hp : ∀ a b : ℕ, (r^a) ∣ (p^b)) :
  ∃ (A α : ℝ) (n : ℕ), p = (Polynomial.C A) * (Polynomial.X - Polynomial.C α) ^ n :=
sorry

end polynomial_form_of_divisibility_l456_456574


namespace myrtle_eggs_l456_456840

theorem myrtle_eggs :
  ∀ (daily_rate per_hen : ℕ) (num_hens : ℕ) (days_away : ℕ) (eggs_taken : ℕ) (eggs_dropped : ℕ),
    daily_rate = 3 →
    num_hens = 3 →
    days_away = 7 →
    eggs_taken = 12 →
    eggs_dropped = 5 →
    (num_hens * daily_rate * days_away - eggs_taken - eggs_dropped) = 46 :=
by
  intros daily_rate per_hen num_hens days_away eggs_taken eggs_dropped
  assume h_rate h_hens h_days h_taken h_dropped
  rw [h_rate, h_hens, h_days, h_taken, h_dropped]
  calc 3 * 3 * 7 - 12 - 5 = 63 - 12 - 5 : by norm_num
                     ... = 51 - 5     : by norm_num
                     ... = 46         : by norm_num
  done

end myrtle_eggs_l456_456840


namespace cot_sum_ineq_l456_456371

theorem cot_sum_ineq
  (α β γ : ℝ) 
  (h1 : ∀ x, x = α ∨ x = β ∨ x = γ → 0 < x ∧ x < π / 2)
  (h2 : cos(α)^2 + cos(β)^2 + cos(γ)^2 = 1) :
  (cot α) * (cot β) + (cot β) * (cot γ) + (cot γ) * (cot α) ≤ 3 / 2 := sorry

end cot_sum_ineq_l456_456371


namespace rectangle_construction_l456_456613

noncomputable
def triangle := Type

noncomputable
def line (t : triangle) := Type

noncomputable
def point (t : triangle) := Type

variables (t : triangle) (K L M : point t) (KL KM LM : line t)
variables (A : point t) (on_halfline_opposite_to_KL : A ∈ KL)

def possible_rectangle (A B C D : point t) : Prop :=
  B ∈ KM ∧ C ∈ KL ∧ D ∈ LM

theorem rectangle_construction (h_triangle : triangle)
  (h_pointA : A ∈ KL) (h_possible_rectangle : ∃ B C D, possible_rectangle A B C D) :
  ∃ B C D, possible_rectangle A B C D := 
sorry

end rectangle_construction_l456_456613


namespace sum_of_areas_is_1_over_150_l456_456811

-- Defining the conditions as a structure for clarity.
structure unit_square (A B C D Q1 : Point) (P : ℕ → Point) :=
  (ABCD_unit: dist A B = 1 ∧ dist B C = 1 ∧ dist C D = 1 ∧ dist D A = 1)
  (Q1_on_CD : on_segment Q1 C D)
  (Q1_ratio  : dist C Q1 / dist Q1 D = 3)
  (Pi_inter_BD : ∀ (i:ℕ), on_line (P i) (line_through A (Q i)) ∧ 
                       on_line (P i) (line_through B D))
  (Qi_next : ∀ (i:ℕ), exists foot : Point, orthogonal_projection (P i) C D (Q (i + 1)) = some foot)

-- The area of each triangle
def triangle_area (D Q P : Point) : ℝ :=
  abs (det (Q - D) (P - D)) / 2

-- The infinite series sum of the areas of the triangles
def sum_triangle_areas (Q P : ℕ → Point) : ℝ :=
  (1 : ℝ) / 150

-- Formalize the goal in Lean
theorem sum_of_areas_is_1_over_150 :
  ∀ (A B C D Q1 : Point) (P Q : ℕ → Point),
  unit_square A B C D Q1 P Q →
  ∑' i, triangle_area D (Q i) (P i) = 1 / 150 :=
by {
  intros A B C D Q1 P Q h,
  -- Proof omitted
  sorry
}

end sum_of_areas_is_1_over_150_l456_456811


namespace volume_pentahedron_FDEABC_l456_456973

-- Define the initial properties and conditions
def regular_tetrahedron_volume (S A B C : Point) (V : ℝ) : Prop :=
  ∃ B h, V = (1 / 3) * B * h ∧ regular_tetrahedron S A B C

def midpoint (P Q: Point) (M: Point) : Prop :=
  M = (P + Q) / 2

def partition_ratio (P Q: Point) (R: Point) (a b : ℝ) : Prop :=
  R = (a * Q + b * P) / (a + b)

-- Lean statement for the volume of the pentahedron
theorem volume_pentahedron_FDEABC (S A B C D E F : Point) (V : ℝ)
  (h1 : regular_tetrahedron_volume S A B C V)
  (h2 : midpoint S A D)
  (h3 : midpoint S B E)
  (h4 : partition_ratio S C F 1 3)
  : volume_pentahedron F D E A B C = (15 / 16) * V :=
sorry

end volume_pentahedron_FDEABC_l456_456973


namespace value_of_a_l456_456753

theorem value_of_a (a : ℝ) : (a^2 - 4) / (a - 2) = 0 → a ≠ 2 → a = -2 :=
by 
  intro h1 h2
  sorry

end value_of_a_l456_456753


namespace carly_burgers_l456_456995

theorem carly_burgers : 
  ∀ (B T G : ℕ), 
  (∀ x, B = 30 ∧ T = 72 ∧ G = 45 → 
   (G / x) * 8 = T → x = 5) := 
begin
  intros B T G h xt,
  have h1 := h 5,
  simp at *,
  sorry,
end

end carly_burgers_l456_456995


namespace eventuallyAllFaceUp_l456_456945

-- Definition of the card deck and operation
def Card := ℕ
def Deck := List Card

-- Predicate for if the deck is all face-up
def isAllFaceUp (d: Deck) : Prop :=
  ∀ c ∈ d, c = 1

-- Function to flip a given consecutive block of cards
def flipBlock (d: Deck) (i j: ℕ) : Deck :=
  if 0 ≤ i ∧ i < j ∧ j < d.length ∧ d.get i = 2 ∧ d.get j = 2 then
    d.take i ++ (d.drop i).take (j - i + 1).reverse ++ d.drop (j + 1)
  else
    d

-- The theorem to prove: Eventually all cards are face-up
theorem eventuallyAllFaceUp (d: Deck) :
  ∃ n, (flipBlock^n) d = repeat 1 d.length :=
  sorry

end eventuallyAllFaceUp_l456_456945


namespace integral_case_a_value_integral_case_b_undefined_integral_case_c_value_l456_456991

noncomputable def integral_case_a : ℝ :=
  ∮ (x : ℝ) in Real.circle 1, y / (x^2 + y^2) * dx - x / (x^2 + y^2) * dy

theorem integral_case_a_value :
  integral_case_a = -2 * Real.pi :=
sorry

noncomputable def integral_case_b : ℝ :=
  ∮ (x : ℝ) in {(x-1, y) | (x-1)^2 + y^2 = 1}, y / (x^2 + y^2) * dx - x / (x^2 + y^2) * dy

theorem integral_case_b_undefined :
  ¬(∃ val : ℝ, integral_case_b = val) :=
sorry

noncomputable def integral_case_c : ℝ :=
  ∮ (x : ℝ) in {(x-1, y-1) | (x-1)^2 + (y-1)^2 = 1}, y / (x^2 + y^2) * dx - x / (x^2 + y^2) * dy

theorem integral_case_c_value :
  integral_case_c = 0 :=
sorry

end integral_case_a_value_integral_case_b_undefined_integral_case_c_value_l456_456991


namespace barrels_in_one_ton_l456_456241

-- Definitions (conditions)
def barrel_weight : ℕ := 10 -- in kilograms
def ton_in_kilograms : ℕ := 1000

-- Theorem Statement
theorem barrels_in_one_ton : ton_in_kilograms / barrel_weight = 100 :=
by
  sorry

end barrels_in_one_ton_l456_456241


namespace increasing_g_on_neg_l456_456703

variable {R : Type*} [LinearOrderedField R]

-- Assumptions: 
-- 1. f is an increasing function on R
-- 2. (h_neg : ∀ x : R, f x < 0)

theorem increasing_g_on_neg (f : R → R) (h_inc : ∀ x y : R, x < y → f x < f y) (h_neg : ∀ x : R, f x < 0) :
  ∀ x y : R, x < y → x < 0 → y < 0 → (x^2 * f x < y^2 * f y) :=
by
  sorry

end increasing_g_on_neg_l456_456703


namespace florist_picked_roses_l456_456251

def initial_roses : ℕ := 11
def sold_roses : ℕ := 2
def final_roses : ℕ := 41
def remaining_roses := initial_roses - sold_roses
def picked_roses := final_roses - remaining_roses

theorem florist_picked_roses : picked_roses = 32 :=
by
  -- This is where the proof would go, but we are leaving it empty on purpose
  sorry

end florist_picked_roses_l456_456251


namespace geometric_series_sum_l456_456459

theorem geometric_series_sum 
  (a : ℝ) (r : ℝ) (s : ℝ)
  (h_a : a = 9)
  (h_r : r = -2/3)
  (h_abs_r : |r| < 1)
  (h_s : s = a / (1 - r)) : 
  s = 5.4 := by
  sorry

end geometric_series_sum_l456_456459


namespace minimum_triangle_formation_l456_456675

theorem minimum_triangle_formation (s : Finset ℕ) (h : s.card = 17) (hs : ∀ x ∈ s, x ≤ 2005) :
  ∃ (a b c ∈ s), a + b > c ∧ a + c > b ∧ b + c > a := by
  sorry

end minimum_triangle_formation_l456_456675


namespace sum_of_2016_integers_positive_l456_456870

theorem sum_of_2016_integers_positive 
  (a : Finₓ 2016 → ℤ) 
  (h : ∀ S : Finset (Finₓ 2016), S.card = 1008 → ∑ i in S, a i > 0) 
  : ∑ i, a i > 0 := sorry

end sum_of_2016_integers_positive_l456_456870


namespace elasticity_ratio_approximation_l456_456629

def qN := 1.01
def pN := 0.61

theorem elasticity_ratio_approximation : (qN / pN) ≈ 1.7 := by
  sorry

end elasticity_ratio_approximation_l456_456629


namespace geometric_sequence_sum_111_l456_456546

theorem geometric_sequence_sum_111 (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_geo : ∃ r : ℕ, b = a * r ∧ c = a * r ^ 2) (h_sum : a + b + c = 111) :
  {a, b, c} = {1, 10, 100} ∨ {a, b, c} = {27, 36, 48} :=
  sorry

end geometric_sequence_sum_111_l456_456546


namespace maximum_cars_quota_div_10_eq_200_l456_456136

-- Define the conditions
def car_length := 5  -- meters
def speed_distance_relation (v : ℕ) : ℕ := (v / 20) + if v % 20 = 0 then 0 else 1

-- Define the speed in meters per hour
def speed_in_meters_per_hour (v : ℕ) : ℕ := 20_000 * speed_distance_relation v

-- Define the length of each unit (car + space behind it)
def unit_length (m : ℕ) : ℕ := 10 * m + 5

-- Define the number of units passing per hour
def units_per_hour (v : ℕ) := speed_in_meters_per_hour v / unit_length (speed_distance_relation v)

-- Problem statement
theorem maximum_cars_quota_div_10_eq_200 : (units_per_hour 20_000 / 10) = 200 :=
by
  sorry

end maximum_cars_quota_div_10_eq_200_l456_456136


namespace sin_double_angle_l456_456406

variable (θ : ℝ)
hypothesis (h : tan θ + 1 / tan θ = Real.sqrt 5)

theorem sin_double_angle :
  Real.sin (2 * θ) = (2 * Real.sqrt 5) / 5 :=
by
  sorry

end sin_double_angle_l456_456406


namespace differentiable_difference_constant_l456_456457

variable {R : Type*} [AddCommGroup R] [Module ℝ R]

theorem differentiable_difference_constant (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) 
  (h : ∀ x, fderiv ℝ f x = fderiv ℝ g x) : 
  ∃ C : ℝ, ∀ x, f x - g x = C := 
sorry

end differentiable_difference_constant_l456_456457


namespace exists_subset_sum_divisible_by_2n_l456_456816

open BigOperators

theorem exists_subset_sum_divisible_by_2n (n : ℕ) (hn : n ≥ 4) (a : Fin n → ℤ)
  (h_distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j)
  (h_interval : ∀ i : Fin n, 0 < a i ∧ a i < 2 * n) :
  ∃ (s : Finset (Fin n)), (∑ i in s, a i) % (2 * n) = 0 :=
sorry

end exists_subset_sum_divisible_by_2n_l456_456816


namespace find_tangent_point_and_slope_l456_456305

theorem find_tangent_point_and_slope :
  ∃ m n : ℝ, (m = 1 ∧ n = Real.exp 1 ∧ 
    (∀ x y : ℝ, y - n = (Real.exp m) * (x - m) → x = 0 ∧ y = 0) ∧ 
    (Real.exp m = Real.exp 1)) :=
sorry

end find_tangent_point_and_slope_l456_456305


namespace number_2013th_is_2_l456_456325

def init_numbers : List ℕ := [1, 2]

def next_number (a b : ℕ) : ℕ := (a * b) % 10

-- We define a function that generates the sequence of numbers reported by the students.
def generate_sequence (n : ℕ) : List ℕ :=
  let rec aux (i : ℕ) (lst : List ℕ) : List ℕ :=
    if h : i < n then
      let len := List.length lst
      let a := lst.get! ((len - 1) % 5)
      let b := lst.get! ((len - 2) % 5)
      aux (i + 1) (lst ++ [next_number a b])
    else
      lst
  aux (2 : ℕ) init_numbers

-- Given that n = 2013, we need to prove that the 2013th number in the sequence is 2.
theorem number_2013th_is_2 : (generate_sequence 2013).get! (2013 - 1) = 2 :=
by
  sorry

end number_2013th_is_2_l456_456325


namespace circle_center_radius_l456_456375

theorem circle_center_radius (x y : ℝ) :
  (x - 1)^2 + (y - 3)^2 = 4 → (1, 3) = (1, 3) ∧ 2 = 2 :=
by
  intro h
  exact ⟨rfl, rfl⟩

end circle_center_radius_l456_456375


namespace calculate_g_inv_sum_l456_456467

def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3 * x - x^2

def g_inv (y : ℝ) : ℝ :=
  if y = -4 then 4
  else if y = 1 then 2
  else if y = 5 then -2
  else 0  -- Assume 0 for other undefined cases for the purpose of this exercise

theorem calculate_g_inv_sum : g_inv (-4) + g_inv 1 + g_inv 5 = 4 := by
  sorry

end calculate_g_inv_sum_l456_456467


namespace solution_set_f_less_x_plus_1_l456_456246

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_continuous : Continuous f
axiom f_at_1 : f 1 = 2
axiom f_derivative : ∀ x, deriv f x < 1

theorem solution_set_f_less_x_plus_1 : 
  ∀ x : ℝ, (f x < x + 1) ↔ (x > 1) :=
by
  sorry

end solution_set_f_less_x_plus_1_l456_456246


namespace range_of_a_l456_456736

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ ({ x | ∃ y : ℝ, y = sqrt (4 - x^2) }) ∩ { x | a < x ∧ x < a + 1 } → x ∈ { x | a < x ∧ x < a + 1 }) → (-2 ≤ a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l456_456736


namespace exists_function_f_l456_456854

noncomputable def f (x : ℝ) : ℝ := (1 / 45 + real.sqrt x) ^ 2

theorem exists_function_f :
  ∃ (f : ℝ → ℝ), (∀ x ≥ 0, (f^[45] x) = 1 + x + 2 * real.sqrt x) := by
  use λ x, (1 / 45 + real.sqrt x) ^ 2
  sorry

end exists_function_f_l456_456854


namespace construct_origin_from_A_and_B_l456_456538

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 2⟩
def B : Point := ⟨3, 1⟩
def isAboveAndToLeft (p₁ p₂ : Point) : Prop := p₁.x < p₂.x ∧ p₁.y > p₂.y
def isOriginConstructed (A B : Point) : Prop := ∃ O : Point, O = ⟨0, 0⟩

theorem construct_origin_from_A_and_B : 
  isAboveAndToLeft A B → isOriginConstructed A B :=
by
  sorry

end construct_origin_from_A_and_B_l456_456538


namespace distinct_non_zero_reals_square_rational_l456_456504

theorem distinct_non_zero_reals_square_rational
  {a : Fin 10 → ℝ}
  (distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (non_zero : ∀ i, a i ≠ 0)
  (rational_condition : ∀ i j, ∃ (q : ℚ), a i + a j = q ∨ a i * a j = q) :
  ∀ i, ∃ (q : ℚ), (a i)^2 = q :=
by
  sorry

end distinct_non_zero_reals_square_rational_l456_456504


namespace triangle_shape_l456_456087

noncomputable def triangle := {A B C : ℝ // 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π}

theorem triangle_shape (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h4 : A + B + C = π)
  (h : Real.sin C + Real.sin (B - A) = Real.sin (2 * A)) : 
  (A = B) ∨ (A = π / 2) ∨ (B = π / 2) ∨ (C = π / 2) :=
sorry

end triangle_shape_l456_456087


namespace multiply_powers_zero_exponent_distribute_term_divide_powers_l456_456286

-- 1. Prove a^{2} \cdot a^{3} = a^{5}
theorem multiply_powers (a : ℝ) : a^2 * a^3 = a^5 := 
sorry

-- 2. Prove (3.142 - π)^{0} = 1
theorem zero_exponent : (3.142 - Real.pi)^0 = 1 := 
sorry

-- 3. Prove 2a(a^{2} - 1) = 2a^{3} - 2a
theorem distribute_term (a : ℝ) : 2 * a * (a^2 - 1) = 2 * a^3 - 2 * a := 
sorry

-- 4. Prove (-m^{3})^{2} \div m^{4} = m^{2}
theorem divide_powers (m : ℝ) : ((-m^3)^2) / (m^4) = m^2 := 
sorry

end multiply_powers_zero_exponent_distribute_term_divide_powers_l456_456286


namespace right_angled_triangle_example_l456_456983

-- Let a, b, and c be the lengths of the sides.
def forms_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_angled_triangle_example : forms_right_angled_triangle 6 8 10 :=
by {
  unfold forms_right_angled_triangle,
  -- The final proof steps would go here, but we assume correctness
  sorry,
}

end right_angled_triangle_example_l456_456983


namespace shortest_path_from_A_to_D_not_inside_circle_l456_456783

noncomputable def shortest_path_length : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (18, 24)
  let O : ℝ × ℝ := (9, 12)
  let r : ℝ := 15
  15 * Real.pi

theorem shortest_path_from_A_to_D_not_inside_circle :
  let A := (0, 0)
  let D := (18, 24)
  let O := (9, 12)
  let r := 15
  shortest_path_length = 15 * Real.pi := 
by
  sorry

end shortest_path_from_A_to_D_not_inside_circle_l456_456783


namespace price_of_whole_pizza_l456_456164

theorem price_of_whole_pizza
    (price_per_slice : ℕ)
    (num_slices_sold : ℕ)
    (num_whole_pizzas_sold : ℕ)
    (total_revenue : ℕ) 
    (H : price_per_slice * num_slices_sold + num_whole_pizzas_sold * P = total_revenue) : 
    P = 15 :=
by
  let price_per_slice := 3
  let num_slices_sold := 24
  let num_whole_pizzas_sold := 3
  let total_revenue := 117
  sorry

end price_of_whole_pizza_l456_456164


namespace taxi_fare_calculation_l456_456537

def fare_per_km : ℝ := 1.8
def starting_fare : ℝ := 8
def starting_distance : ℝ := 2
def total_distance : ℝ := 12

theorem taxi_fare_calculation : 
  (if total_distance <= starting_distance then starting_fare
   else starting_fare + (total_distance - starting_distance) * fare_per_km) = 26 := by
  sorry

end taxi_fare_calculation_l456_456537


namespace length_of_segment_AB_l456_456695

-- Define the points A and B in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨1, 2, 3⟩
def B : Point3D := ⟨0, 4, 5⟩

-- Define the distance formula in 3D space
def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

-- State the proof problem
theorem length_of_segment_AB : distance A B = 3 := by
  sorry

end length_of_segment_AB_l456_456695


namespace modulus_of_z_l456_456824

noncomputable def complex_z : ℂ :=
  2 / (1 + complex.I) + (1 + complex.I) ^ 2

theorem modulus_of_z :
  complex.norm complex_z = real.sqrt 2 := 
by 
  sorry

end modulus_of_z_l456_456824


namespace intersecting_segments_l456_456135

theorem intersecting_segments 
  (n : ℕ) 
  (segments : Fin (n+1) → Set ℝ) 
  (common_point : ℝ) 
  (h : ∀ i, common_point ∈ segments i) 
  :
  ∃ (I J : Fin (n+1)), 
  I ≠ J ∧ 
  ∃ x y ∈ segments I ∩ segments J, 
  |y - x| ≥ (n-1 : ℝ) / n * (d : ℝ) :=
begin
  sorry
end

end intersecting_segments_l456_456135


namespace max_l_trominos_proof_l456_456647

-- Defining a 4x4 grid with squares colored as red, green, or blue.
inductive Color
| red
| green
| blue

structure Grid := 
(color : ℕ → ℕ → Color)
(valid_coord : ∀ x y, 1 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 4)

structure L_tromino :=
(coord1 coord2 coord3 : ℕ × ℕ)
(shared_corner : ∃ x y, Set.to_finset {coord1, coord2, coord3} ⊆ {(x, y), (x+1, y), (x, y+1), (x+1, y+1)}

-- Defining whether an L-tromino contains exactly one square of each color
def contains_one_of_each_color (g : Grid) (t : L_tromino) : Prop :=
let colors := {g.color (fst t.coord1) (snd t.coord1), g.color (fst t.coord2) (snd t.coord2), g.color (fst t.coord3) (snd t.coord3)} in
colors = {Color.red, Color.green, Color.blue}

-- Define the maximum number of L-trominos containing exactly one square of each color in a 4x4 grid
noncomputable def max_l_trominos_with_one_of_each_color (g : Grid) : ℕ :=
sorry -- Here would be the logic to count the maximum L-trominos according to the problem

theorem max_l_trominos_proof : ∀ g : Grid, max_l_trominos_with_one_of_each_color g = 18 :=
sorry

end max_l_trominos_proof_l456_456647


namespace sum_of_first_100_terms_l456_456007

noncomputable def S_n (n : ℕ) : ℚ := (n * (n + 1)) / 2

theorem sum_of_first_100_terms :
  (Finset.range 100).sum (λ n, 1 / S_n (n + 1)) = 200 / 101 :=
by sorry

end sum_of_first_100_terms_l456_456007


namespace problem_statement_l456_456823

noncomputable def S := { x : ℝ // x > 0 }

def f (s : S) : ℝ := 
    sorry -- The exact definition of f is derived in the solution steps.

theorem problem_statement
  (f : S → ℝ)
  (h : ∀ x y : S, f x * f y = f (⟨x * y, by nlinarith⟩) + 1004 * (1 / (x:ℝ) + 1 / (y:ℝ) + 1003))
  (n s : ℝ)
  (h1 : n = 1)
  (h2 : s = 3010 / 3) :
  n * s = 1003 + 1 / 3 :=
sorry

end problem_statement_l456_456823


namespace max_switches_not_exceed_comb_l456_456896

theorem max_switches_not_exceed_comb :
  ∀ (n : ℕ), ∃ (h : ℕ → ℕ),
  strict_mono h ∧
  (∀ (k : ℕ), (2 ≤ k ∧ k < n) → (h k = k + 1)) ∧
  (maximum_switches h n ≤ nat.choose n 3) :=
begin
  sorry
end

end max_switches_not_exceed_comb_l456_456896


namespace damage_ratio_l456_456124

variable (H g τ M : ℝ)
variable (n : ℕ)
variable (k : ℝ)
variable (H_pos : H > 0) (g_pos : g > 0) (n_pos : n > 0) (k_pos : k > 0)

def V_I := sqrt (2 * g * H)
def h := H / n
def V_1 := sqrt (2 * g * h)
def V_1' := (1 / k) * sqrt (2 * g * h)
def V_II := sqrt (2 * g * h / k^2 + 2 * g * (H - h))

def I_I := M * V_I * τ
def I_II := M * τ * ((V_1 - V_1') + V_II)

theorem damage_ratio : 
  I_II / I_I = (k - 1) / (sqrt n * k) + sqrt ((n - 1) * k^2 + 1) / (sqrt n * k^2) → I_II / I_I = 5 / 4 :=
by sorry

end damage_ratio_l456_456124


namespace area_of_paper_l456_456219

theorem area_of_paper (L W : ℕ) (h1 : L + 2 * W = 34) (h2 : 2 * L + W = 38) : L * W = 140 := by
  sorry

end area_of_paper_l456_456219


namespace minimum_value_l456_456393

theorem minimum_value (x y : ℝ) (l : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = l) (hl1 : l = 1) :
  (\dfrac{1}{x} + \dfrac{9}{y}) ≥ 16 :=
by sorry

end minimum_value_l456_456393


namespace no_four_digit_palindrome_perfect_squares_with_middle_11_l456_456605

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

noncomputable def has_middle_11 (n : ℕ) : Prop :=
  let s := n.to_string in
  s.length = 4 ∧ s[1] = '1' ∧ s[2] = '1'

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_four_digit_palindrome_perfect_squares_with_middle_11 :
  ∀ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ is_perfect_square n ∧ is_palindrome n ∧ has_middle_11 n) → false :=
by
  sorry

end no_four_digit_palindrome_perfect_squares_with_middle_11_l456_456605


namespace company_management_structure_l456_456595

theorem company_management_structure : 
  let num_employees := 13
  let choose (n k : ℕ) := nat.choose n k
  let ways_to_chose_CEO := num_employees
  let remaining_after_CEO := num_employees - 1
  let ways_to_choose_VPs := choose remaining_after_CEO 2
  let remaining_after_VPs := remaining_after_CEO - 2
  let ways_to_choose_managers_VP1 := choose remaining_after_VPs 3
  let remaining_after_VP1_mgrs := remaining_after_VPs - 3
  let ways_to_choose_managers_VP2 := choose remaining_after_VP1_mgrs 3
  let total_ways := ways_to_chose_CEO * ways_to_choose_VPs * ways_to_choose_managers_VP1 * ways_to_choose_managers_VP2
  total_ways = 349800 := by
    sorry

end company_management_structure_l456_456595


namespace line_tangent_to_circle_l456_456465

open EuclideanGeometry

theorem line_tangent_to_circle
  {A B C O E F R P N M : Point}
  (h1 : ¬Collinear A B C)
  (h2 : TangentCircleAt ω O A C E)
  (h3 : TangentCircleAt ω A B F)
  (h4 : SegmentInLine R E F)
  (h5 : LineThroughAndParallelTo O EF IntersectsLineAt AB P)
  (h6 : IntersectionOfLines PR AC N)
  (h7 : IntersectionOfLines AB AC RParallelToAC M)
  : TangentLineThrough M N ω :=
begin
  sorry
end

end line_tangent_to_circle_l456_456465


namespace existence_of_100_pairs_l456_456489

def digits_at_least_6 (n : ℕ) : Prop := 
  ∀ d ∈ n.digits 10, d ≥ 6

theorem existence_of_100_pairs :
  ∃ (pairs : list (ℕ × ℕ)), 
    pairs.length = 100 ∧ 
    ∀ (x y : ℕ), (x, y) ∈ pairs → 
    digits_at_least_6 x ∧ digits_at_least_6 y ∧ digits_at_least_6 (x * y) :=
sorry

end existence_of_100_pairs_l456_456489


namespace hypotenuse_length_l456_456522

theorem hypotenuse_length
  (x : ℝ) 
  (shorter_leg_pos : x > 0)
  (area_eq : (1 / 2) * x * (3 * x - 3) = 108)
  (longer_leg_eq : 3 * x - 3 > 0) :
  (real.sqrt (x^2 + (3 * x - 3)^2)) = real.sqrt 657 :=
by
  sorry

end hypotenuse_length_l456_456522


namespace line_in_one_plane_parallel_to_other_l456_456213

-- Definition: Two planes are parallel.
def planes_parallel (π1 π2 : Plane) : Prop :=
  ∀ x, x ∈ π1 ↔ x ∈ π2

-- Definition: A line is in a plane.
def line_in_plane (l : Line) (π : Plane) : Prop :=
  ∀ x, x ∈ l → x ∈ π

-- Definition: A line is parallel to a plane.
def line_parallel_plane (l : Line) (π : Plane) : Prop :=
  ¬ (∃ y, y ∈ l ∧ y ∈ π)

-- Theorem: In two parallel planes, a line in one plane must be parallel to the other plane.
theorem line_in_one_plane_parallel_to_other {π1 π2 : Plane} (l : Line) :
  planes_parallel π1 π2 → line_in_plane l π1 → line_parallel_plane l π2 :=
by
  sorry

end line_in_one_plane_parallel_to_other_l456_456213


namespace symmetric_point_y_axis_l456_456895

theorem symmetric_point_y_axis (B : ℝ × ℝ) (hB : B = (-3, 4)) : 
  ∃ A : ℝ × ℝ, A = (3, 4) ∧ A.2 = B.2 ∧ A.1 = -B.1 :=
by
  use (3, 4)
  sorry

end symmetric_point_y_axis_l456_456895


namespace area_of_triangle_ABC_is_correct_l456_456089

-- Definitions for the given problem
def base (A B : Point) : ℝ := AB
def height (C : Point) (AB : Line) : ℝ := CD
def median (A : Point) (E : Point) (BC : Line) : ℝ := AE
def area (ABC : Triangle) : ℝ := sorry

variable (A B C D E : Point)
variable (ABC ADC BDC : Triangle)

-- Problem conditions
axiom IsBase : D ∈ Line A B
axiom IsMedian : AE = 5
axiom IsHeight : CD = 6
axiom AreaRelation : (Area ADC) = 3 * (Area BDC)

-- Lean 4 statement for the equivalent proof problem
theorem area_of_triangle_ABC_is_correct :
  Area ABC = (96 / 7) :=
sorry

end area_of_triangle_ABC_is_correct_l456_456089


namespace variance_linear_combination_l456_456712

-- Define the property of variance
variable {X : Type} [has_scalar ℝ X] [add_group X] [topological_space X] [measurable_space X]

-- Given the variance condition
variable (D : X → ℝ)
axiom D_X : D X = 2

-- State the theorem to be proved
theorem variance_linear_combination :
  D (3 * X + 2) = 18 := sorry

end variance_linear_combination_l456_456712


namespace find_y_l456_456318

theorem find_y (y : ℝ) (hy : log y 81 = 4 / 2) : y = 9 := 
by {
  sorry
}

end find_y_l456_456318


namespace evaluate_expression_l456_456453

def spadesuit (a b : ℚ) := (3 * a + b) / (a + b)

theorem evaluate_expression : spadesuit (5 : ℚ) (spadesuit (3 : ℚ) (6 : ℚ)) 1 = (17 : ℚ) / 7 := by
  sorry

end evaluate_expression_l456_456453


namespace nth_equation_sequence_l456_456516

theorem nth_equation_sequence (n : ℕ) : 
  2^n * (∏ k in finset.range n + 1, (2 * k - 1)) = ∏ k in finset.Ico (n + 1) (2 * n + 1) id :=
sorry

end nth_equation_sequence_l456_456516


namespace number_of_keepers_l456_456570

theorem number_of_keepers (hens goats camels : ℕ) (keepers feet heads : ℕ)
  (h_hens : hens = 50)
  (h_goats : goats = 45)
  (h_camels : camels = 8)
  (h_equation : (2 * hens + 4 * goats + 4 * camels + 2 * keepers) = (hens + goats + camels + keepers + 224))
  : keepers = 15 :=
by
sorry

end number_of_keepers_l456_456570


namespace simplify_cos_subtraction_l456_456495

theorem simplify_cos_subtraction :
  let c := Real.cos (72 * Real.sin (180 / Real.pi))
  let d := Real.cos (144 * Real.sin (180 / Real.pi))
  in c - d = 1.117962 :=
by
  sorry

end simplify_cos_subtraction_l456_456495


namespace x_intercept_of_line_through_points_is_eight_ninths_l456_456276

def x_intercept (p1 p2 : ℝ × ℝ) : ℝ :=
  let ⟨x1, y1⟩ := p1
  let ⟨x2, y2⟩ := p2
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  -b / m

theorem x_intercept_of_line_through_points_is_eight_ninths :
  x_intercept (2, -2) (-3, 7) = 8 / 9 := by
  sorry

end x_intercept_of_line_through_points_is_eight_ninths_l456_456276


namespace range_of_a_l456_456070

theorem range_of_a (a : ℝ) : 
  (∃ (x : ℝ), ¬(a * x > a - 1)) ↔ 
  a ∈ set.Iio 0 ∪ set.Ioi 0 :=
by
  sorry

end range_of_a_l456_456070


namespace count_valid_pairs_l456_456205

-- Definitions of the conditions
def is_valid_pair (a b : ℕ) : Prop :=
  2 * a + 2 * b = 34 ∧ a ≥ b

-- Define the statement translating the proof problem
theorem count_valid_pairs : 
  ∃ (s : Finset (ℕ × ℕ)), s.filter (λ p, is_valid_pair p.1 p.2) .card = 8 :=
sorry

end count_valid_pairs_l456_456205


namespace trips_to_collect_all_trays_l456_456206

-- Definition of conditions
def trays_at_once : ℕ := 7
def trays_one_table : ℕ := 23
def trays_other_table : ℕ := 5

-- Theorem statement
theorem trips_to_collect_all_trays : 
  (trays_one_table / trays_at_once) + (if trays_one_table % trays_at_once = 0 then 0 else 1) + 
  (trays_other_table / trays_at_once) + (if trays_other_table % trays_at_once = 0 then 0 else 1) = 5 := 
by
  sorry

end trips_to_collect_all_trays_l456_456206


namespace probability_of_at_least_three_distinct_numbers_with_six_l456_456554

theorem probability_of_at_least_three_distinct_numbers_with_six :
  let total_outcomes := 6^4 in
  let favorable_outcomes := (Nat.choose 5 3 * 4!) + (Nat.choose 5 2 * Nat.choose 4 2 * 2!) in
  (favorable_outcomes / total_outcomes : ℚ) = 5 / 18 :=
by
  sorry

end probability_of_at_least_three_distinct_numbers_with_six_l456_456554


namespace windmill_visits_infinitely_l456_456451

noncomputable theory
open_locale classic

open set

def is_collinear {P Q R : point} : Prop :=
∃ (k : ℝ), (P - Q) = k • (R - Q)

def finite_set_no_collinear (s : set point) : Prop := 
finite s ∧ ∀ P Q R ∈ s, ¬ is_collinear P Q R

def windmill_condition (S : set point) (P : point) (ℓ : line) : Prop :=
∃ (P0 ∈ S) (Q ∈ S), P ∉ {Q} ∧
∀ (ℓ : line) (x : ℝ), ℓ.contains P0 ∧ ¬ ℓ.contains Q

theorem windmill_visits_infinitely 
  {S : set point}
  (h1 : finite S)
  (h2 : ∀ P Q R ∈ S, ¬ is_collinear P Q R) :
  ∃ (P : point) (ℓ : line), (P ∈ S) ∧ windmill_condition S P ℓ :=
sorry

end windmill_visits_infinitely_l456_456451


namespace counting_numbers_divide_90_l456_456399

theorem counting_numbers_divide_90 (S : Set ℕ) :
  S = {n : ℕ | n ∣ 90 ∧ n > 11} →
  S.card = 5 :=
by
  sorry

end counting_numbers_divide_90_l456_456399


namespace carl_owes_15300_l456_456633

def total_property_damage : ℝ := 40000
def total_medical_bills : ℝ := 70000
def insurance_coverage_property_damage : ℝ := 0.80
def insurance_coverage_medical_bills : ℝ := 0.75
def carl_responsibility : ℝ := 0.60

def carl_personally_owes : ℝ :=
  let insurance_paid_property_damage := insurance_coverage_property_damage * total_property_damage
  let insurance_paid_medical_bills := insurance_coverage_medical_bills * total_medical_bills
  let remaining_property_damage := total_property_damage - insurance_paid_property_damage
  let remaining_medical_bills := total_medical_bills - insurance_paid_medical_bills
  let carl_share_property_damage := carl_responsibility * remaining_property_damage
  let carl_share_medical_bills := carl_responsibility * remaining_medical_bills
  carl_share_property_damage + carl_share_medical_bills

theorem carl_owes_15300 :
  carl_personally_owes = 15300 := by
  sorry

end carl_owes_15300_l456_456633


namespace EG_over_GF_half_l456_456086

-- Definitions and conditions
variables {A B C E F M G : Type} [AddCommGroup A] [VectorSpace ℝ A]
variables (a b c e f m g : A)
variable (x : ℝ)

-- Conditions
noncomputable def midpoint (u v : A) : A := (u + v) / 2
noncomputable def segment_ratio (u v r : ℝ) (p q : A) : A := (r * p + (1 - r) * q)

-- Given problem conditions
axiom condition_1 : m = midpoint b c
axiom condition_2 : dist a b = 15
axiom condition_3 : dist a c = 20
axiom condition_4 : e ∈ [a, c]
axiom condition_5 : f ∈ [a, b]
axiom condition_6 : G ∈ [segment_ratio (x / 15) m a, segment_ratio ((3 * x) / 20) e f]
axiom condition_7 : dist a e = 3 * dist a f

-- Problem statement
theorem EG_over_GF_half : (dist e g) / (dist g f) = 1 / 2 :=
sorry

end EG_over_GF_half_l456_456086


namespace probability_point_in_sphere_eq_2pi_div_3_l456_456609

open Real Topology

noncomputable def volume_of_region := 4 * 2 * 2

noncomputable def volume_of_sphere_radius_2 : ℝ :=
  (4 / 3) * π * (2 ^ 3)

noncomputable def probability_in_sphere : ℝ :=
  volume_of_sphere_radius_2 / volume_of_region

theorem probability_point_in_sphere_eq_2pi_div_3 :
  probability_in_sphere = (2 * π) / 3 :=
by
  sorry

end probability_point_in_sphere_eq_2pi_div_3_l456_456609


namespace find_value_l456_456221

theorem find_value :
  3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800 :=
by
  sorry

end find_value_l456_456221


namespace planar_region_solution_l456_456039

section

variables {x y : ℝ}

/--
Given the planar region defined by the system of inequalities:
1. x ≥ 0
2. y ≥ 0
3. x + sqrt(3) * y - sqrt(3) ≤ 0
-/
def planar_region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + sqrt 3 * y - sqrt 3 ≤ 0

/--
Find the equation of the smallest circle (⊙)_C that covers the region
bounded by the inequalities:
1. x ≥ 0
2. y ≥ 0
3. x + sqrt(3) * y - sqrt(3) ≤ 0
-/
def circle_C_eq : Prop :=
  ∀ (x y : ℝ), planar_region x y → (x - sqrt 3 / 2)^2 + (y - 1 / 2)^2 ≤ 1

/--
Find the equation of the line l with slope 60 degrees that intersects this circle
(⊙)_C at points A and B such that AB = sqrt(3).
-/
def line_l_eq : Prop :=
  ∀ (x y : ℝ), planar_region x y →
    (∃ m : ℝ, (y = sqrt 3 * x + m ∨ y = sqrt 3 * x - 2) ∧
      (x - sqrt 3 / 2)^2 + (y - 1 / 2)^2 = 1 ∧
      ∃ A B : ℝ × ℝ, A ≠ B ∧
        (A, B are on line) ∧ dist A B = sqrt 3)

end

-- Final statement combining the two definitions:
theorem planar_region_solution :
  circle_C_eq ∧ line_l_eq :=
begin
  sorry,  -- Proof to be filled
end

end planar_region_solution_l456_456039


namespace divisors_of_3003_3003_l456_456054

theorem divisors_of_3003_3003 :
  ∃ (a b c d : ℕ), (a + 1) * (b + 1) * (c + 1) * (d + 1) = 3003 ∧ 
  (a, b, c, d).pairwise (≠)
:=
by
  let num_solutions := 24 -- This is the correct answer from identifying the distinct permutations
  sorry

end divisors_of_3003_3003_l456_456054


namespace area_of_annulus_l456_456082

theorem area_of_annulus (R r t : ℝ) (h : R > r) (h_tangent : R^2 = r^2 + t^2) : 
  π * (R^2 - r^2) = π * t^2 :=
by 
  sorry

end area_of_annulus_l456_456082


namespace julie_compound_interest_l456_456799

-- Definitions of conditions
def initial_savings : ℝ := 3600
def half_savings : ℝ := initial_savings / 2
def time_period : ℝ := 2
def simple_interest_earned : ℝ := 120
def interest_rate : ℝ := simple_interest_earned / (half_savings * time_period)
def compound_interest_formula (P R T : ℝ) : ℝ := P * ((1 + R)^T - 1)

-- Statement of the problem
theorem julie_compound_interest :
  compound_interest_formula half_savings interest_rate time_period = 121.84 :=
by
  -- Skipping the proof
  sorry

end julie_compound_interest_l456_456799


namespace swimming_club_cars_l456_456536

theorem swimming_club_cars (c : ℕ) :
  let vans := 3
  let people_per_car := 5
  let people_per_van := 3
  let max_people_per_car := 6
  let max_people_per_van := 8
  let extra_people := 17
  let total_people := 5 * c + (people_per_van * vans)
  let max_capacity := max_people_per_car * c + (max_people_per_van * vans)
  (total_people + extra_people = max_capacity) → c = 2 := by
  sorry

end swimming_club_cars_l456_456536


namespace base_of_500_in_decimal_is_7_l456_456331

theorem base_of_500_in_decimal_is_7 :
  ∃ b : ℕ, 5 ≤ b ∧ b ≤ 7 ∧
  ∀ n, (500 : ℕ).digits b = n.digits b ∧ 
  n.length = 4 ∧ (n.last % 2 = 1) :=
begin
  sorry
end

end base_of_500_in_decimal_is_7_l456_456331


namespace limit_tn_one_over_n_l456_456107

variable (G : Type) [Group G] [Fintype G]

def t_n (n : ℕ) : ℕ := (Fintype.card G) ^ (n + 1) / (Fintype.card (Subgroup.center G)) ^ n

theorem limit_tn_one_over_n (G : Type)
  [Group G] [Fintype G] :
  tendsto (λ n : ℕ, (t_n G n) ^ (1 / (n : ℝ))) at_top (nhds (↑(Fintype.card G) / ↑(Fintype.card (Subgroup.center G)))) :=
sorry

end limit_tn_one_over_n_l456_456107


namespace f_6_equals_2_l456_456380

noncomputable def f : ℝ → ℝ 
| x < 0          => x^3 - 1
| -1 ≤ x ∧ x ≤ 1 => if 0 ≤ x then x^3 - 1 else -(x^3 - 1)
| x > 0.5        => f (x - 0.5)

theorem f_6_equals_2 : f 6 = 2 := 
by
-- sorry is used to skip the proof
sorry

end f_6_equals_2_l456_456380


namespace plane_equation_l456_456126

variables {x y z : ℝ}

/-- Intersection of planes and calculation -/
theorem plane_equation
  (L_inter : ∀ x y z, (2 * x - y + 2 * z = 4) ∧ (3 * x + 4 * y - z = 6))
  (P_diff : ¬(∀ x y z, (2 * x - y + 2 * z = 4) ∧ (∀ x y z, 3 * x + 4 * y - z = 6) ) )
  (dist : ∀ x y z, (x = 4) ∧ (y = -2) ∧ (z = 2) → (| (2 * x + 3 * y) - 4 | / real.sqrt ( 6)) = (3 / real.sqrt(6))) :
  ∃ (A B C D : ℤ), (A * x + B * y + C * z + D = 0 ∧ 
  A = 1 ∧ B = 63 ∧ C = -35 ∧ D = -34) :=
by sorry

end plane_equation_l456_456126


namespace slower_whale_length_is_101_25_l456_456914

def length_of_slower_whale (v_i_f v_i_s a_f a_s t : ℝ) : ℝ :=
  let D_f := v_i_f * t + 0.5 * a_f * t^2
  let D_s := v_i_s * t + 0.5 * a_s * t^2
  D_f - D_s

theorem slower_whale_length_is_101_25
  (v_i_f v_i_s a_f a_s t L : ℝ)
  (h1 : v_i_f = 18)
  (h2 : v_i_s = 15)
  (h3 : a_f = 1)
  (h4 : a_s = 0.5)
  (h5 : t = 15)
  (h6 : length_of_slower_whale v_i_f v_i_s a_f a_s t = L) :
  L = 101.25 :=
by
  sorry

end slower_whale_length_is_101_25_l456_456914


namespace not_prime_in_any_numeral_system_l456_456469

theorem not_prime_in_any_numeral_system (n : ℕ) (hn : n > 2) (even_n : n % 2 = 0) (a : ℕ) (ha : a ≥ 2) :
  ¬ prime (∑ i in finset.range n, a ^ i) :=
by sorry

end not_prime_in_any_numeral_system_l456_456469


namespace diagonals_pass_through_orthocenter_l456_456830

theorem diagonals_pass_through_orthocenter
  (triangle : Type)
  [preorder triangle]
  {points : triangle → triangle → triangle → triangle}
  (M M1 M2 M3 : triangle)
  (hexagon : set triangle)
  (is_orthocenter : ∀ (A B C : triangle), orthocenter M A B C)
  (are_reflections : ∀ (side : triangle → triangle → triangle), 
    reflection_on_side M side M1 ∧ 
    reflection_on_side M side M2 ∧ 
    reflection_on_side M side M3)
  (hexagon_formed : ∀ (intersections : triangle → triangle → triangle), 
    convex_hexagon hexagon intersections (M1, M2, M3)) :
  ∀ (diagonals : set (triangle × triangle)), 
  (∀ (P Q : diagonals), P.1 ≠ P.2 → P.2 ≠ Q.1 → Q.2 ≠ P.1) →
  ∃ (intersection : triangle), 
  intersection = M :=
sorry

end diagonals_pass_through_orthocenter_l456_456830


namespace permutation_diff_invariant_l456_456804

-- Definitions
def P (n : ℕ) : ℕ := 
⟨n!⟩ -- Placeholder: Define P(n) properly according to the condition.

-- Main theorem
theorem permutation_diff_invariant (n : ℕ) (h : n ≥ 2) : 
  P (n + 5) - P (n + 4) - P (n + 3) + P n = 4 :=
sorry

end permutation_diff_invariant_l456_456804


namespace minimum_next_test_score_l456_456798

-- Given the current scores of Josanna
def current_scores : List ℕ := [90, 80, 70, 60, 85]

-- Define the current average
def current_average (scores : List ℕ) : ℚ :=
  (scores.foldr (·+·) 0) / (scores.length)

-- Define the target increase in average
def target_increase : ℚ := 3

-- Define the desired average
def desired_average : ℚ :=
  current_average current_scores + target_increase

-- Define the total number of tests after the next test
def total_tests : ℕ := current_scores.length + 1

-- Define the total score required to reach the desired average
def required_total_score : ℚ :=
  desired_average * total_tests

-- Sum of the current scores
def current_total_score : ℚ :=
  current_scores.foldr (·+·) 0

-- Prove the minimum score required on the next test
theorem minimum_next_test_score :
  ∃ (x : ℚ), x ≥ 95 ∧ (current_total_score + x) = required_total_score :=
by
  sorry

end minimum_next_test_score_l456_456798


namespace sin_pi_minus_a_l456_456334

theorem sin_pi_minus_a (a : ℝ) (h_cos_a : Real.cos a = Real.sqrt 5 / 3) (h_range_a : a ∈ Set.Ioo (-Real.pi / 2) 0) : 
  Real.sin (Real.pi - a) = -2 / 3 :=
by sorry

end sin_pi_minus_a_l456_456334


namespace rearrange_pyramid_possible_l456_456155

/-- Rearrange the cubes in a pyramid shape such that each cube touches only new cubes. -/
def rearrange_pyramid (pyramid : Pyramid) (new_pyramid : Pyramid) : Prop :=
  pyramid.shape = new_pyramid.shape ∧
  ∀ cube ∈ pyramid.cubes, ∀ neighbor ∈ pyramid.neighbors(cube), neighbor ∉ new_pyramid.neighbors(cube)

theorem rearrange_pyramid_possible (pyramid : Pyramid) :
  ∃ new_pyramid : Pyramid, rearrange_pyramid pyramid new_pyramid :=
sorry

end rearrange_pyramid_possible_l456_456155


namespace inequality_proof_l456_456707

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (h_sum : a + b + c = 1) :
    (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := 
by 
  sorry

end inequality_proof_l456_456707


namespace part1_part2_l456_456737
noncomputable def equation1 (x k : ℝ) := 3 * (2 * x - 1) = k + 2 * x
noncomputable def equation2 (x k : ℝ) := (x - k) / 2 = x + 2 * k

theorem part1 (x k : ℝ) (h1 : equation1 4 k) : equation2 x k ↔ x = -65 := sorry

theorem part2 (x k : ℝ) (h1 : equation1 x k) (h2 : equation2 x k) : k = -1 / 7 := sorry

end part1_part2_l456_456737


namespace hexagon_theorem_l456_456294

-- Define a structure for the hexagon with its sides
structure Hexagon :=
(side1 side2 side3 side4 side5 side6 : ℕ)

-- Define the conditions of the problem
def hexagon_conditions (h : Hexagon) : Prop :=
  h.side1 = 5 ∧ h.side2 = 6 ∧ h.side3 = 7 ∧
  (h.side1 + h.side2 + h.side3 + h.side4 + h.side5 + h.side6 = 38)

-- Define the proposition that we need to prove
def hexagon_proposition (h : Hexagon) : Prop :=
  (h.side3 = 7 ∨ h.side4 = 7 ∨ h.side5 = 7 ∨ h.side6 = 7) → 
  (h.side1 = 5 ∧ h.side2 = 6 ∧ h.side3 = 7 ∧ h.side4 = 7 ∧ h.side5 = 7 ∧ h.side6 = 7 → 3 = 3)

-- The proof statement combining conditions and the to-be-proven proposition
theorem hexagon_theorem (h : Hexagon) (hc : hexagon_conditions h) : hexagon_proposition h :=
by
  sorry -- No proof is required

end hexagon_theorem_l456_456294


namespace min_value_of_Box_l456_456401

theorem min_value_of_Box {a b : ℤ} (h_ab : a * b = 15) (h_distinct : a ≠ b ∧ b ≠ 34 ∧ a ≠ 34):
  let Box := a^2 + b^2 in
  34 ≤ Box :=
begin
  sorry
end

end min_value_of_Box_l456_456401


namespace projection_vector_example_l456_456324

noncomputable def vector_projection (u v : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
let dot_uv := u.1 * v.1 + u.2 * v.2 + u.3 * v.3 + u.4 * v.4 in
let dot_vv := v.1 * v.1 + v.2 * v.2 + v.3 * v.3 + v.4 * v.4 in
let k := dot_uv / dot_vv in
(k * v.1, k * v.2, k * v.3, k * v.4)

/-- 
Given vector u = (3, 2, -1, 4) and the line direction vector (1, -3, 6, 3),
prove that the projection of u onto the line is (3/55, -9/55, 18/55, 9/55).
-/
theorem projection_vector_example :
  vector_projection (3, 2, -1, 4) (1, -3, 6, 3) = (3/55, -9/55, 18/55, 9/55) :=
sorry

end projection_vector_example_l456_456324


namespace value_of_f_at_five_thirds_pi_l456_456641

def f : ℝ → ℝ := sorry  -- We'll define this as per the conditions later

-- Given conditions in Lean 4:
variable (f : ℝ → ℝ)

-- Condition 1: f is an odd function
axiom odd_function : ∀ x : ℝ, f (-x) = -f x

-- Condition 2: f is a periodic function with period π
axiom periodic_function : ∀ x : ℝ, f (x + π) = f x

-- Condition 3: On the interval [0, π/2], f(x) = sin x
axiom interval_condition : ∀ x : ℝ, 0 ≤ x → x ≤ π / 2 → f x = Real.sin x

-- Theorem to prove
theorem value_of_f_at_five_thirds_pi : f (5 / 3 * π) = -√3 / 2 := 
  sorry

end value_of_f_at_five_thirds_pi_l456_456641


namespace find_g_at_2_l456_456113

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ (x : ℝ), x ≠ 0 → x ≠ 1 / 2015 → g(x - 2) + g(x) + g(x + 2) = (g(x))^2 / (2015 * x - 1)

theorem find_g_at_2 : g(2) = 12060 :=
by {
  sorry
}

end find_g_at_2_l456_456113


namespace person_B_completion_time_l456_456900

variables {A B : ℝ} (H : A + B = 1/6 ∧ (A + 10 * B = 1/6))

theorem person_B_completion_time :
    (1 / (1 - 2 * (A + B)) / B = 15) :=
by
  sorry

end person_B_completion_time_l456_456900


namespace lunch_customers_is_127_l456_456951

-- Define the conditions based on the given problem
def breakfast_customers : ℕ := 73
def dinner_customers : ℕ := 87
def total_customers_on_saturday : ℕ := 574
def total_customers_on_friday : ℕ := total_customers_on_saturday / 2

-- Define the variable representing the lunch customers
variable (L : ℕ)

-- State the proposition we want to prove
theorem lunch_customers_is_127 :
  breakfast_customers + L + dinner_customers = total_customers_on_friday → L = 127 := by {
  sorry
}

end lunch_customers_is_127_l456_456951


namespace restaurant_sales_l456_456134

theorem restaurant_sales (monday tuesday wednesday thursday : ℕ) 
  (h1 : monday = 40) 
  (h2 : tuesday = monday + 40) 
  (h3 : wednesday = tuesday / 2) 
  (h4 : monday + tuesday + wednesday + thursday = 203) : 
  thursday = wednesday + 3 := 
by sorry

end restaurant_sales_l456_456134


namespace order_of_trig_values_l456_456119

theorem order_of_trig_values (x : ℝ) (hx : x ∈ Set.Ioo (-1/2 : ℝ) 0) : 
  let a1 := Real.cos (Real.sin (x * Real.pi))
  let a2 := Real.sin (Real.cos (x * Real.pi))
  let a3 := Real.cos ((x + 1) * Real.pi)
  in a3 < a2 ∧ a2 < a1 :=
sorry

end order_of_trig_values_l456_456119


namespace monotonicity_range_of_a_l456_456515

noncomputable def is_monotonically_increasing (a : ℝ) := 
  ∀ x : ℝ, x > -1 → 3 * a * x^2 + 2 * (a - 3) * x ≥ 0

theorem monotonicity_range_of_a : 
  (∀ x : ℝ, x > -1 → 3 * a * x^2 + 2 * (a - 3) * x ≥ 0) ↔ a ≥ -3 :=
begin
  sorry
end

end monotonicity_range_of_a_l456_456515


namespace contrapositive_l456_456509

variable (Line Circle : Type) (distance : Line → Circle → ℝ) (radius : Circle → ℝ)
variable (is_tangent : Line → Circle → Prop)

-- Original proposition in Lean notation:
def original_proposition (l : Line) (c : Circle) : Prop :=
  distance l c ≠ radius c → ¬ is_tangent l c

-- Contrapositive of the original proposition:
theorem contrapositive (l : Line) (c : Circle) : Prop :=
  is_tangent l c → distance l c = radius c

end contrapositive_l456_456509


namespace problem_part_I_problem_part_II_l456_456422

theorem problem_part_I (A B C : ℝ)
  (h1 : 0 < A) 
  (h2 : A < π / 2)
  (h3 : 1 + (Real.sqrt 3 / 3) * Real.sin (2 * A) = 2 * (Real.sin ((B + C) / 2))^2) : 
  A = π / 3 := 
sorry

theorem problem_part_II (A B C R S : ℝ)
  (h1 : A = π / 3)
  (h2 : R = 2 * Real.sqrt 3) 
  (h3 : S = (1 / 2) * (6 * (Real.sin A)) * (Real.sqrt 3 / 2)) :
  S = 9 * Real.sqrt 3 :=
sorry

end problem_part_I_problem_part_II_l456_456422


namespace jay_and_paul_distance_apart_l456_456980

-- Defining Jay's walking rate
def jay_walking_rate : ℝ := 1 / 20 -- miles per minute

-- Defining Paul's walking rate
def paul_walking_rate : ℝ := 3 / 40 -- miles per minute

-- Time walked in minutes
def time_walked : ℝ := 120

-- Distance Jay walks
def jay_distance_walked : ℝ := jay_walking_rate * time_walked

-- Distance Paul walks
def paul_distance_walked : ℝ := paul_walking_rate * time_walked

-- Total distance apart after 2 hours
def total_distance_apart : ℝ := jay_distance_walked + paul_distance_walked

theorem jay_and_paul_distance_apart : total_distance_apart = 15 :=
by 
  sorry  -- Proof is not required

end jay_and_paul_distance_apart_l456_456980


namespace coefficient_x_in_binomial_expansion_l456_456372

theorem coefficient_x_in_binomial_expansion :
  let n := 6 in
  let binom_exp := (Real.sqrt (x) + 3 / (Real.sqrt (x))) ^ n in
  ∑ k in Finset.range (n + 1), (binom_exp.coeff k) = 64 →
  (Real.sqrt (x) + 3 / (Real.sqrt (x))) ^ n.proj_terms x = 135 := sorry

end coefficient_x_in_binomial_expansion_l456_456372


namespace complex_exp_sequence_cos_sin_eq_l456_456478

def complex_exp_sequence (z : ℂ) (n : ℕ) : ℂ :=
  z^n

theorem complex_exp_sequence_cos_sin_eq (n : ℕ) (hn : 0 < n) :
  (complex_exp_sequence (⟨1 / 2, √(3) / 2⟩ : ℂ) n) = 
    complex.exp (complex.I * (n : ℝ) * (π / 3)) :=
by sorry

end complex_exp_sequence_cos_sin_eq_l456_456478


namespace gcd_2023_2048_l456_456921

theorem gcd_2023_2048 : Nat.gcd 2023 2048 = 1 := by
  sorry

end gcd_2023_2048_l456_456921


namespace original_price_of_movie_ticket_l456_456104

theorem original_price_of_movie_ticket
    (P : ℝ)
    (new_price : ℝ)
    (h1 : new_price = 80)
    (h2 : new_price = 0.80 * P) :
    P = 100 :=
by
  sorry

end original_price_of_movie_ticket_l456_456104


namespace exists_integer_with_digit_sum_and_divisibility_l456_456863

theorem exists_integer_with_digit_sum_and_divisibility :
  ∃ (N : ℤ), (1996 ∣ N) ∧ (∑ d in (N.digits 10), d) = 1996 :=
sorry

end exists_integer_with_digit_sum_and_divisibility_l456_456863


namespace length_of_bridge_correct_l456_456603

noncomputable def length_of_bridge (speed_kmh : ℝ) (time_min : ℝ) : ℝ :=
  let speed_mpm := (speed_kmh * 1000) / 60  -- Convert speed from km/hr to m/min
  speed_mpm * time_min  -- Length of the bridge in meters

theorem length_of_bridge_correct :
  length_of_bridge 10 10 = 1666.7 :=
by
  sorry

end length_of_bridge_correct_l456_456603


namespace find_α_l456_456500

-- Define the inverse proportionality condition and specific values provided
axiom αβ_inverse (α β k : ℝ) : α * β = k

-- Specific invariant: α = 4 when β = 9
axiom α4_β9 (k : ℝ) : αβ_inverse 4 9 k

-- Specific case: Find α when β = -72
theorem find_α (k : ℝ) (h1 : αβ_inverse 4 9 k) (h2 : αβ_inverse α (-72) k) : α = -1/2 := 
by
  sorry

end find_α_l456_456500


namespace sum_of_distinct_digits_remainder_l456_456449

theorem sum_of_distinct_digits_remainder :
  let S := ∑ n in (finset.filter (λ n : ℕ, n >= 500 ∧ n <= 999 ∧
    (n / 100 ≠ (n / 10) % 10) ∧ ((n / 100) ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10)),
     finset.range 1000), n in
  S % 1000 = 720 :=
by
  let S := ∑ n in (finset.filter (λ n : ℕ, n >= 500 ∧ n <= 999 ∧
      (n / 100 ≠ (n / 10) % 10) ∧ ((n / 100) ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10)),
      finset.range 1000), n in
  have h1 : S = 223720 := sorry, -- Sum calculation from the solution steps
  have h2 : 223720 % 1000 = 720 := sorry,
  exact h2

end sum_of_distinct_digits_remainder_l456_456449


namespace range_of_a_l456_456680

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2

def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x < 1 then (2 * a - 1) * x + 3 * a else a ^ x

theorem range_of_a (a : ℝ) (h : decreasing_function (f a)) : a ∈ Ioo (1 / 4) 1 :=
by
  sorry

end range_of_a_l456_456680


namespace opposite_of_negative_six_is_six_l456_456180

-- Define what it means for one number to be the opposite of another.
def is_opposite (a b : Int) : Prop :=
  a = -b

-- The statement to be proved: the opposite number of -6 is 6.
theorem opposite_of_negative_six_is_six : is_opposite (-6) 6 :=
  by sorry

end opposite_of_negative_six_is_six_l456_456180


namespace problem_l456_456060

def g(x : ℤ) : ℤ := 3 * x^2 + 3 * x - 2

theorem problem : g(g(3)) = 3568 := by
  sorry

end problem_l456_456060


namespace tangent_line_at_zero_f_ge_x2_plus_x_l456_456036

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x^2 - 1

theorem tangent_line_at_zero :
  let y := λ x, x in
  f' 0 = 1 ∧ f 0 = 0 ∧ ∀ x, f x = 0 → y = x :=
by sorry

theorem f_ge_x2_plus_x (x : ℝ) : f x ≥ x^2 + x :=
by sorry

end tangent_line_at_zero_f_ge_x2_plus_x_l456_456036


namespace cafeteria_apples_pies_l456_456171

theorem cafeteria_apples_pies (initial_apples handed_out_apples apples_per_pie remaining_apples pies : ℕ) 
    (h_initial: initial_apples = 62) 
    (h_handed_out: handed_out_apples = 8) 
    (h_apples_per_pie: apples_per_pie = 9)
    (h_remaining: remaining_apples = initial_apples - handed_out_apples) 
    (h_pies: pies = remaining_apples / apples_per_pie) : 
    pies = 6 := by
  sorry

end cafeteria_apples_pies_l456_456171


namespace eval_expression_1_eq_16_eval_expression_2_eq_0_l456_456990

noncomputable def eval_expression_1 : ℝ :=
  Real.sqrt (6.25) - (Real.pi - 1)^0 - ((27 / 8)^(1 / 3)) + ((1 / 64)^(-2 / 3))

theorem eval_expression_1_eq_16 : eval_expression_1 = 16 := by
  sorry

noncomputable def eval_expression_2 : ℝ :=
  Real.sin (11 * Real.pi / 6) + Real.cos (-20 * Real.pi / 3) + Real.tan (29 * Real.pi / 4)

theorem eval_expression_2_eq_0 : eval_expression_2 = 0 := by
  sorry

end eval_expression_1_eq_16_eval_expression_2_eq_0_l456_456990


namespace joggers_meeting_times_product_l456_456912

theorem joggers_meeting_times_product:
  (∀ t_1 : ℝ, 80 < t_1 ∧ t_1 < 100 → 
    ∃ t_2 : ℤ, 
      let t_min := Int.ceil (900 / 16) in
      let t_max := Int.floor (720 / 11) in
      (t_2 = t_min ∨ t_2 = t_max) ∧
      (36 * (1 / t_1 + 1 / (t_2 : ℝ)) = 1) →
      ((t_min * t_max) = 3705)) :=
begin
  sorry
end

end joggers_meeting_times_product_l456_456912


namespace problem_statement_l456_456930

theorem problem_statement (a b c d e : ℕ) (h : {a, b, c, d, e} = {2, 3, 5, 6, 7}) :
    ∃ x y z w : ℕ, x = 2 ∧ y = 3 ∧ z = 5 ∧ w = 67 ∧ (x * y * z * w) = 2010 :=
by
  use 2, 3, 5, 67
  simp
  sorry

end problem_statement_l456_456930


namespace central_angle_of_section_l456_456955

theorem central_angle_of_section (A : ℝ) (x: ℝ) (H : (1 / 8 : ℝ) = (x / 360)) : x = 45 :=
by
  sorry

end central_angle_of_section_l456_456955


namespace triangle_angle_correct_l456_456893

noncomputable def triangle_angles
  (a b c : ℝ)
  (ha : a = 2)
  (hb : b = Real.sqrt 6)
  (hc : c = 1 + Real.sqrt 3) : ℝ × ℝ × ℝ := 
  let α := Real.acos ((b^2 + c^2 - a^2) / (2 * b * c))
  let β := Real.acos ((a^2 + c^2 - b^2) / (2 * a * c))
  let γ := Real.acos ((a^2 + b^2 - c^2) / (2 * a * b))
  (α, β, γ)

theorem triangle_angle_correct
  (α β γ : ℝ)
  (hα : α = Real.acos (1/2)) 
  (hβ : β = Real.acos (Real.sqrt 2/2))
  (hγ : γ = Real.acos ((Real.sqrt 3 + 1 - 2) / (2 * Real.sqrt 6 * (1 + Real.sqrt 3))))
  : (
    α = Real.pi / 3 ∧ 
    β = Real.pi / 4 ∧ 
    γ = 5 * Real.pi / 12
  ) :=
  by
    sorry

end triangle_angle_correct_l456_456893


namespace fraction_of_color_in_selected_correct_l456_456256

variable (x y z : ℝ)

def main_committee_black_and_white : ℝ := 30 * x
def main_committee_color : ℝ := 6 * y
def total_films : ℝ := main_committee_black_and_white x y z + main_committee_color x y z

def selected_percentage : ℝ := z / 100
def selected_films : ℝ := selected_percentage x y z * total_films x y z

def selected_black_and_white_percentage : ℝ := (y / x) / 100
def selected_black_and_white : ℝ := selected_black_and_white_percentage x y z * main_committee_black_and_white x y z

def selected_color : ℝ := main_committee_color x y z

def fraction_of_color_in_selected : ℝ := selected_color x y z / (selected_black_and_white x y z + selected_color x y z)

theorem fraction_of_color_in_selected_correct :
  fraction_of_color_in_selected x y z = 20 / 21 :=
by
  sorry

end fraction_of_color_in_selected_correct_l456_456256


namespace negation_of_existential_statement_l456_456525

theorem negation_of_existential_statement :
  (¬ ∃ x : ℝ, x ≥ 1 ∨ x > 2) ↔ ∀ x : ℝ, x < 1 :=
by
  sorry

end negation_of_existential_statement_l456_456525


namespace boat_speed_still_water_l456_456571

theorem boat_speed_still_water (b s : ℝ) (h1 : b + s = 21) (h2 : b - s = 9) : b = 15 := 
by 
  -- Solve the system of equations
  sorry

end boat_speed_still_water_l456_456571


namespace number_of_equilateral_triangles_in_S_l456_456814

open Set

-- Define the set S
def S := {p : ℝ × ℝ × ℝ | (p.1 ∈ ({0, 1, 3} : Set ℝ)) ∧ 
                              (p.2 ∈ ({0, 1, 3} : Set ℝ)) ∧ 
                              (p.3 ∈ ({0, 1, 3} : Set ℝ))}

-- Define the property of being an equilateral triangle
def is_equilateral (a b c : ℝ × ℝ × ℝ) : Prop :=
  let d := λ p q : ℝ × ℝ × ℝ, (dist (p.1, p.2, p.3) (q.1, q.2, q.3)) in
  d a b = d b c ∧ d b c = d c a

-- Statement of the problem
theorem number_of_equilateral_triangles_in_S : 
  finsupp.card (finset.filter (λ (t : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ), 
    is_equilateral (t.1, t.2, t.3) (t.4, t.5, t.6) ∘ prod id) 
  (finset.univ.filter (λ t : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ),
    (t.1 ∈ S) ∧ (t.2 ∈ S))) = 72 := 
sorry

end number_of_equilateral_triangles_in_S_l456_456814


namespace perpendicular_bisectors_concur_l456_456424

variable {P : Type} [EuclideanGeometry P]

variables A B C D E F : P
variables convex_hexagon : ConvexHexagon A B C D E F
variables h1 : ∠ A ≅ ∠ B
variables h2 : ∠ C ≅ ∠ D
variables h3 : ∠ E ≅ ∠ F

theorem perpendicular_bisectors_concur :
  ∃ O : P, PerpendicularBisector A B O ∧ PerpendicularBisector C D O ∧ PerpendicularBisector E F O :=
by
  -- proof to be completed
  sorry

end perpendicular_bisectors_concur_l456_456424


namespace no_four_points_with_all_odd_distances_l456_456091

theorem no_four_points_with_all_odd_distances :
  ¬ ∃ (A B C D : ℝ × ℝ), 
      (∀ (P Q : ℝ × ℝ), (P = A ∨ P = B ∨ P = C ∨ P = D) ∧ 
                          (Q = A ∨ Q = B ∨ Q = C ∨ Q = D) → 
                          (dist P Q) ∈ ℤ ∧ 
                          (dist P Q) % 2 = 1) :=
sorry

end no_four_points_with_all_odd_distances_l456_456091


namespace isosceles_triangle_perimeter_l456_456009

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
  (a + b + b = 25) ∧ (a + a + b ≤ b → False) :=
by
  sorry

end isosceles_triangle_perimeter_l456_456009


namespace broken_line_bisector_l456_456764

theorem broken_line_bisector
  {A B M K H : Type*}
  [MetricSpace K]
  [InCircle A B M K]
  (AM MB : ℝ)
  (h_gt : AM > MB)
  [MidpointArc A B K]
  [Perpendicular K H AM] :

  Bisects H AMB :=
begin
  sorry
end

end broken_line_bisector_l456_456764


namespace lambda_value_condition_l456_456394

def vec (x y : ℝ) : (ℝ × ℝ) := (x, y)

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ u = (k * v.1, k * v.2)

theorem lambda_value_condition (λ : ℝ) :
  let a := vec 1 2
  let b := vec 2 3
  let c := vec (-4) (-7)
  collinear (λ * a.1 + b.1, λ * a.2 + b.2) c →
  λ = 2 :=
  by
    intros a b c h
    have := sorry -- Proof omitted

end lambda_value_condition_l456_456394


namespace quadratic_has_real_roots_l456_456751

theorem quadratic_has_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m-2) * x^2 - 2 * x + 1 = 0) ↔ m ≤ 3 :=
by sorry

end quadratic_has_real_roots_l456_456751


namespace travel_time_l456_456952

theorem travel_time (speed distance time : ℕ) (h_speed : speed = 60) (h_distance : distance = 180) : 
  time = distance / speed → time = 3 := by
  sorry

end travel_time_l456_456952


namespace find_ab_l456_456020

theorem find_ab (a b : ℝ) (h1 : a + b = 4) (h2 : a^3 + b^3 = 136) : a * b = -6 :=
by
  sorry

end find_ab_l456_456020


namespace part1_part2_part3_part4_l456_456000

section QuadraticFunction

variable {x : ℝ} {y : ℝ} 

-- 1. Prove that if a quadratic function y = x^2 + bx - 3 intersects the x-axis at (3, 0), 
-- then b = -2 and the other intersection point is (-1, 0).
theorem part1 (b : ℝ) : 
  ((3:ℝ) ^ 2 + b * (3:ℝ) - 3 = 0) → 
  b = -2 ∧ ∃ x : ℝ, (x = -1 ∧ x^2 + b * x - 3 = 0) := 
  sorry

-- 2. For the function y = x^2 + bx - 3 where b = -2, 
-- prove that when 0 < y < 5, x is in -2 < x < -1 or 3 < x < 4.
theorem part2 (b : ℝ) :
  b = -2 → 
  (0 < y ∧ y < 5 → ∃ x : ℝ, (x^2 + b * x - 3 = y) → (-2 < x ∧ x < -1) ∨ (3 < x ∧ x < 4)) :=
  sorry

-- 3. Prove that the value t such that y = x^2 + bx - 3 and y > t always holds for all x
-- is t < -((b ^ 2 + 12) / 4).
theorem part3 (b t : ℝ) :
  (∀ x : ℝ, (x ^ 2 + b * x - 3 > t)) → t < -(b ^ 2 + 12) / 4 :=
  sorry

-- 4. Given y = x^2 - 3x - 3 and 1 < x < 2, 
-- prove that m < y < n with n = -5, b = -3, and m ≤ -21 / 4.
theorem part4 (m n : ℝ) :
  (1 < x ∧ x < 2 → m < x^2 - 3 * x - 3 ∧ x^2 - 3 * x - 3 < n) →
  n = -5 ∧ -21 / 4 ≤ m :=
  sorry

end QuadraticFunction

end part1_part2_part3_part4_l456_456000


namespace candy_bar_cost_l456_456903

theorem candy_bar_cost (initial_amount change : ℕ) (h : initial_amount = 50) (hc : change = 5) : 
  initial_amount - change = 45 :=
by
  -- sorry is used to skip the proof
  sorry

end candy_bar_cost_l456_456903


namespace average_capacity_2050_l456_456763

-- Define the conditions
variables (X : ℕ) -- year when average capacity was 0.1 TB
def initial_capacity : ℝ := 0.1 -- initial capacity in TB
def doubling_period : ℕ := 5 -- years it takes to double capacity

-- Statement to be proved
theorem average_capacity_2050 : 
  let n := (2050 - X) / doubling_period in
  (initial_capacity * (2 ^ n) = 0.1 * (2 ^ ((2050 - X) / 5))) :=
by 
  sorry

end average_capacity_2050_l456_456763


namespace gcd_323_391_l456_456880

theorem gcd_323_391 : Nat.gcd 323 391 = 17 := 
by sorry

end gcd_323_391_l456_456880


namespace area_transformed_triangle_l456_456871

variable {F : Type} [Field F]
variable {x1 x2 x3 : F}
variable {g : F → F}

-- Conditions from the problem
def original_points_form_triangle (x1 x2 x3 : F) (g : F → F) : Prop :=
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3

axiom area_original_triangle : original_points_form_triangle x1 x2 x3 g → 
  (let area : F := 45 in true)

-- Statement to prove
theorem area_transformed_triangle :
  original_points_form_triangle x1 x2 x3 g →
  let area := 45 in 
  -- Then the area of the triangle formed by points on the graph of y = 3g(3x)
  let transformed_area : F := 45 in
  transformed_area = area := by 
  sorry

end area_transformed_triangle_l456_456871


namespace imaginary_part_of_complex_l456_456828

noncomputable def imaginary_part (z : Complex) : ℂ :=
  let conj_z := Complex.conj z
  ((4 : ℝ) / z - conj_z).im

theorem imaginary_part_of_complex (z : ℂ) (H1 : z = 1 + Complex.i) (H2 : z.conj = 1 - Complex.i) :
  imaginary_part z = -1 :=
by
  -- Given definitions and conditions
  sorry

end imaginary_part_of_complex_l456_456828


namespace linear_regression_equation_predict_savings_l456_456548

theorem linear_regression_equation :
  (∑ i in (Finset.range 10), (λ i, i)) = 80 →
  (∑ j in (Finset.range 10), (λ j, j)) = 20 →
  (∑ i in (Finset.range 10), (λ i, i * i)) = 184 →
  (∑ i in (Finset.range 10), (λ i, i ^ 2)) = 720 →
  ∃ (a b : ℝ), ∀ (x : ℝ), y = a + b * x :=
begin
  sorry
end

theorem predict_savings :
  let x := 8 in
  let y := -1.2 + 0.4 * x in
  y = 2 :=
begin
  sorry
end

end linear_regression_equation_predict_savings_l456_456548


namespace percent_gain_transaction_l456_456958

theorem percent_gain_transaction (x : ℝ) (h1 : x ≠ 0) :
  let total_cost := 749 * x
  let sold_revenue := 700 * (749 * x) / 700
  let remaining_revenue := 49 * (749 * x) / 700
  let total_revenue := sold_revenue + remaining_revenue
  let profit := total_revenue - total_cost
  let percent_gain := (profit / total_cost) * 100
  percent_gain ≈ 7 :=
by
  let total_cost := 749 * x
  let sold_revenue := 700 * (749 * x) / 700
  let remaining_revenue := 49 * (749 * x) / 700
  let total_revenue := sold_revenue + remaining_revenue
  let profit := total_revenue - total_cost
  let percent_gain := (profit / total_cost) * 100
  sorry

end percent_gain_transaction_l456_456958


namespace find_pqr_l456_456110

theorem find_pqr {p q r : ℝ} : 
  (∃ A B : set ℝ, 
    A = {x | x^2 + p * x - 8 = 0} ∧ 
    B = {x | x^2 - q * x + r = 0} ∧ 
    A ≠ B ∧ 
    A ∪ B = {-2, 4} ∧ 
    A ∩ B = {-2}
  ) ↔ 
  (p = -2 ∧ q = -4 ∧ r = 4) := 
by 
  sorry

end find_pqr_l456_456110


namespace common_root_poly_identity_l456_456946

theorem common_root_poly_identity
  (α p p' q q' : ℝ)
  (h1 : α^3 + p*α + q = 0)
  (h2 : α^3 + p'*α + q' = 0) : 
  (p * q' - q * p') * (p - p')^2 = (q - q')^3 := 
by
  sorry

end common_root_poly_identity_l456_456946


namespace find_angle_A_range_of_bc_l456_456011

-- Define the necessary conditions and prove the size of angle A
theorem find_angle_A 
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : b * (Real.sin B + Real.sin C) = (a - c) * (Real.sin A + Real.sin C))
  (h₂ : B > Real.pi / 2)
  (h₃ : A + B + C = Real.pi)
  (h₄ : a > 0) (h₅ : b > 0) (h₆ : c > 0): 
  A = 2 * Real.pi / 3 :=
sorry

-- Define the necessary conditions and prove the range for b+c when a = sqrt(3)/2
theorem range_of_bc 
  (a b c : ℝ)
  (A : ℝ)
  (h₁ : A = 2 * Real.pi / 3)
  (h₂ : a = Real.sqrt 3 / 2)
  (h₃ : a > 0) (h₄ : b > 0) (h₅ : c > 0)
  (h₆ : A + B + C = Real.pi)
  (h₇ : B + C = Real.pi / 3) : 
  Real.sqrt 3 / 2 < b + c ∧ b + c ≤ 1 :=
sorry

end find_angle_A_range_of_bc_l456_456011


namespace maxBallsInCube_l456_456209

def max_balls_fit_in_cube : Prop :=
  let radius := 3
  let side_length := 9
  let V_cube := side_length^3
  let V_ball := (4 / 3) * Real.pi * (radius^3)
  V_cube ∈ Int.floor V_ball * 6

theorem maxBallsInCube : max_balls_fit_in_cube :=
  sorry

end maxBallsInCube_l456_456209


namespace max_value_a_sqrt_one_plus_b_squared_l456_456059

noncomputable def max_a_sqrt_one_plus_b_squared (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 / 2 + b^2 = 4) : ℝ :=
  a * sqrt (1 + b^2)

theorem max_value_a_sqrt_one_plus_b_squared (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 / 2 + b^2 = 4) :
  max_a_sqrt_one_plus_b_squared a b h1 h2 h3 ≤ 5 * sqrt 2 / 2 :=
sorry

end max_value_a_sqrt_one_plus_b_squared_l456_456059


namespace lara_yesterday_more_than_sarah_l456_456215

variable (yesterdaySarah todaySarah todayLara : ℕ)
variable (cansDifference : ℕ)

axiom yesterdaySarah_eq : yesterdaySarah = 50
axiom todaySarah_eq : todaySarah = 40
axiom todayLara_eq : todayLara = 70
axiom cansDifference_eq : cansDifference = 20

theorem lara_yesterday_more_than_sarah :
  let totalCansYesterday := yesterdaySarah + todaySarah + cansDifference
  let laraYesterday := totalCansYesterday - yesterdaySarah
  laraYesterday - yesterdaySarah = 30 :=
by
  sorry

end lara_yesterday_more_than_sarah_l456_456215


namespace even_and_zero_point_l456_456618

open Real

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def has_zero_point (f : ℝ → ℝ) : Prop :=
  ∃ x, f x = 0

def fA : ℝ → ℝ := λ x, x^2 + 1
def fB : ℝ → ℝ := λ x, abs (log x)
def fC : ℝ → ℝ := cos
def fD : ℝ → ℝ := λ x, exp x - 1

theorem even_and_zero_point :
  (is_even_function fA ∧ has_zero_point fA) ∨
  (is_even_function fB ∧ has_zero_point fB) ∨
  (is_even_function fC ∧ has_zero_point fC) ∨
  (is_even_function fD ∧ has_zero_point fD) 
  ↔ (is_even_function fC ∧ has_zero_point fC) :=
by sorry

end even_and_zero_point_l456_456618


namespace ratio_expression_value_l456_456965

noncomputable def S : Real := 0.6823  -- Given root approximation

theorem ratio_expression_value :
  S^(S^(S^3 + S^(-2)) + S^(-2)) + S^(-2) = 4.3004 :=
by
  -- Hypothetical given condition since question approximation was used
  sorry

end ratio_expression_value_l456_456965


namespace range_of_m_l456_456211

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem range_of_m (m : ℝ) (x : ℝ) (h1 : x ∈ Set.Icc (-1 : ℝ) 2) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x < m) ↔ 2 < m := 
by 
  sorry

end range_of_m_l456_456211


namespace C_moves_in_circle_l456_456774

variable {Point : Type}
variable [Inhabited Point] [Equiv Point] [NormedAddCommGroup Point] [InnerProductSpace ℝ Point]

variable (O1 O2 : Point)
variable (A B C : ℝ → Point)
variable (r1 r2 : ℝ)
variable (ω : ℝ)

-- Initial conditions
def equilateral_triangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def moves_along_circle (A : ℝ → Point) (O : Point) (r : ℝ) (ω : ℝ) : Prop :=
  ∀ t, dist (A t) O = r ∧ (A t) = O + r • (expMap (t • ω • I))

-- Assumptions
axiom equilateral_triang_ABC : ∀ t, equilateral_triangle (A t) (B t) (C t)
axiom A_moves : moves_along_circle A O1 r1 ω
axiom B_moves : moves_along_circle B O2 r2 ω

-- Proof statement
theorem C_moves_in_circle : ∃ (O3 : Point) (r3 : ℝ), moves_along_circle C O3 r3 ω :=
  sorry

end C_moves_in_circle_l456_456774


namespace scalar_product_of_parallel_vectors_l456_456396

variables (a b : ℝ × ℝ)
variables (x : ℝ)

-- Conditions
def a := (1, 2)
def b := (x, -4)
def a_parallel_b : Prop := ∃ k : ℝ, b = (k * 1, k * 2)

-- Scalar product function
def scalar_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Theorem statement
theorem scalar_product_of_parallel_vectors :
  a_parallel_b → scalar_product (1, 2) (x, -4) = -10 :=
by
  intro h
  sorry

end scalar_product_of_parallel_vectors_l456_456396


namespace incircle_tangent_points_l456_456350

theorem incircle_tangent_points {A B C D P S Q R : Point} 
  (h_parallelogram : parallelogram A B C D) 
  (h_tangent_ac : tangent (circle P Q R) A C) 
  (h_tangent_ba_ext : tangent (circle P Q R) (extension B A P)) 
  (h_tangent_bc_ext : tangent (circle P Q R) (extension B C S)) 
  (h_ps_intersect_da : segment_intersect P S D A Q)
  (h_ps_intersect_dc : segment_intersect P S D C R) :
  tangent (incircle D C A) D A Q ∧ tangent (incircle D C A) D C R := sorry

end incircle_tangent_points_l456_456350


namespace kitchen_upgrade_cost_l456_456907

-- Define the number of cabinet knobs and their cost
def num_knobs : ℕ := 18
def cost_per_knob : ℝ := 2.50

-- Define the number of drawer pulls and their cost
def num_pulls : ℕ := 8
def cost_per_pull : ℝ := 4.00

-- Calculate the total cost of the knobs
def total_cost_knobs : ℝ := num_knobs * cost_per_knob

-- Calculate the total cost of the pulls
def total_cost_pulls : ℝ := num_pulls * cost_per_pull

-- Calculate the total cost of the kitchen upgrade
def total_cost : ℝ := total_cost_knobs + total_cost_pulls

-- Theorem statement
theorem kitchen_upgrade_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_cost_l456_456907


namespace m_not_equal_n_possible_l456_456137

-- Define the touching relation on an infinite chessboard
structure Chessboard :=
(colored_square : ℤ × ℤ → Prop)
(touches : ℤ × ℤ → ℤ × ℤ → Prop)

-- Define the properties
def colors_square (board : Chessboard) : Prop :=
∃ i j : ℤ, board.colored_square (i, j) ∧ board.colored_square (i + 1, j + 1)

def black_square_touches_m_black_squares (board : Chessboard) (m : ℕ) : Prop :=
∀ i j : ℤ, board.colored_square (i, j) →
    (board.touches (i, j) (i + 1, j) → board.colored_square (i + 1, j)) ∧ 
    (board.touches (i, j) (i - 1, j) → board.colored_square (i - 1, j)) ∧
    (board.touches (i, j) (i, j + 1) → board.colored_square (i, j + 1)) ∧
    (board.touches (i, j) (i, j - 1) → board.colored_square (i, j - 1))
    -- Add additional conditions to ensure exactly m black squares are touched

def white_square_touches_n_white_squares (board : Chessboard) (n : ℕ) : Prop :=
∀ i j : ℤ, ¬board.colored_square (i, j) →
    (board.touches (i, j) (i + 1, j) → ¬board.colored_square (i + 1, j)) ∧ 
    (board.touches (i, j) (i - 1, j) → ¬board.colored_square (i - 1, j)) ∧
    (board.touches (i, j) (i, j + 1) → ¬board.colored_square (i, j + 1)) ∧
    (board.touches (i, j) (i, j - 1) → ¬board.colored_square (i, j - 1))
    -- Add additional conditions to ensure exactly n white squares are touched

theorem m_not_equal_n_possible (board : Chessboard) (m n : ℕ) :
colors_square board →
black_square_touches_m_black_squares board m →
white_square_touches_n_white_squares board n →
m ≠ n :=
by {
    sorry
}

end m_not_equal_n_possible_l456_456137


namespace incircle_tangent_to_adc_sides_l456_456345

noncomputable def Triangle (A B C : Point) : Prop := -- defining Triangle for context
  True

noncomputable def CircleTangentToSidesAndExtensions (circle : Circle) (AC BA BC : Line) : Prop := -- tangent condition
  True

noncomputable def Parallelogram (A B C D : Point) : Prop := -- defining Parallelogram for context
  True

theorem incircle_tangent_to_adc_sides 
  (A B C D P S Q R : Point)
  (AC BA BC DA DC : Line)
  (circle : Circle) 
  (h_parallelogram : Parallelogram A B C D)
  (h_tangent : CircleTangentToSidesAndExtensions circle AC BA BC)
  (h_intersection : LineIntersectsSegmentsInPoints (line_through P S) DA DC Q R) :
  TangentToIncircleAtPoints (Triangle C D A) (incircle (Triangle C D A)) Q R :=
by
  sorry

end incircle_tangent_to_adc_sides_l456_456345


namespace max_dinners_for_7_people_max_dinners_for_8_people_l456_456494

def max_dinners_with_new_neighbors (n : ℕ) : ℕ :=
  if n = 7 ∨ n = 8 then 3 else 0

theorem max_dinners_for_7_people : max_dinners_with_new_neighbors 7 = 3 := sorry

theorem max_dinners_for_8_people : max_dinners_with_new_neighbors 8 = 3 := sorry

end max_dinners_for_7_people_max_dinners_for_8_people_l456_456494


namespace index_difference_is_0_point_30_l456_456669

/-!
# Problem Statement

For a group of 20 people, 7 of whom are females, prove that the difference between the index for females and the index for males is 0.30.
-/

def total_people : ℕ := 20
def females : ℕ := 7
def index (n k : ℕ) : ℝ := (n - k) / n

theorem index_difference_is_0_point_30 
  (n : ℕ) (k : ℕ) (hn : n = total_people) (hk : k = females) : 
  index n k - index n (n - k) = 0.30 :=
by
  -- Proof to be completed
  sorry

end index_difference_is_0_point_30_l456_456669


namespace minimum_value_problem_l456_456817

theorem minimum_value_problem (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 + 4 * x^2 + 2 * x + 1) * (y^3 + 4 * y^2 + 2 * y + 1) * (z^3 + 4 * z^2 + 2 * z + 1) / (x * y * z) ≥ 1331 :=
sorry

end minimum_value_problem_l456_456817


namespace julia_shortfall_l456_456899

-- Definitions based on the problem conditions
def rock_and_roll_price : ℕ := 5
def pop_price : ℕ := 10
def dance_price : ℕ := 3
def country_price : ℕ := 7
def quantity : ℕ := 4
def julia_money : ℕ := 75

-- Proof problem: Prove that Julia is short $25
theorem julia_shortfall : (quantity * rock_and_roll_price + quantity * pop_price + quantity * dance_price + quantity * country_price) - julia_money = 25 := by
  sorry

end julia_shortfall_l456_456899


namespace max_students_is_8_l456_456338

def students := ℕ
def knows (a b : students) := Prop

noncomputable def max_students (n : students) : Prop :=
  (∀ a b c : students, a ≠ b → b ≠ c → c ≠ a → knows a b ∨ knows b c ∨ knows c a) ∧
  (∀ a b c d : students, a ≠ b → b ≠ c → c ≠ d → d ≠ a → 
  ¬(knows a b ∧ knows b c ∧ knows c d ∧ knows d a ∧ knows a c ∧ knows b d))

theorem max_students_is_8 : ∃ n : students, max_students n ∧ n = 8 := by
  sorry

end max_students_is_8_l456_456338


namespace multiply_polynomials_l456_456836

variable (x y : ℝ)

theorem multiply_polynomials :
  let a := 3 * x^4
      b := 4 * y^3
  in (a - b) * (a^2 + a * b + b^2) = 27 * x^12 - 64 * y^9 := by
    let a := 3 * x^4
    let b := 4 * y^3
    calc
      (a - b) * (a^2 + a * b + b^2)
          = (3 * x^4 - 4 * y^3) * (9 * x^8 + 12 * x^4 * y^3 + 16 * y^6) : by
            simp [a, b]
      ... = 27 * x^12 - 64 * y^9 : sorry

end multiply_polynomials_l456_456836


namespace wheel_size_l456_456756

theorem wheel_size (total_distance : ℝ) (number_of_revolutions : ℝ) (circumference : ℝ) :
  total_distance = 1056 →
  number_of_revolutions = 14.012738853503185 →
  circumference = total_distance / number_of_revolutions →
  circumference ≈ 75.398 := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end wheel_size_l456_456756


namespace correct_options_l456_456619

open Real

def option_A (x : ℝ) : Prop :=
  x^2 - 2*x + 1 > 0

def option_B : Prop :=
  ∃ (x : ℝ), (0 < x) ∧ (x + 4 / x = 6)

def option_C (a b : ℝ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) → (b / a + a / b ≥ 2)

def option_D (x y : ℝ) : Prop :=
  (0 < x) ∧ (0 < y) ∧ (x + 2*y = 1) → (2 / x + 1 / y ≥ 8)

theorem correct_options :
  ¬(∀ (x : ℝ), option_A x) ∧ (option_B ∧ (∀ (a b : ℝ), option_C a b) = false ∧ 
  (∀ (x y : ℝ), option_D x y)) :=
by sorry

end correct_options_l456_456619


namespace emma_final_amount_l456_456312

theorem emma_final_amount :
  ∀ (amount_from_bank spent_on_furniture remaining after_giving_friend: ℝ),
  amount_from_bank = 2000 →
  spent_on_furniture = 400 →
  remaining = amount_from_bank - spent_on_furniture →
  after_giving_friend = (3 / 4) * remaining →
  remaining - after_giving_friend = 400 :=
by
  intros amount_from_bank spent_on_furniture remaining after_giving_friend
  assume h1 h2 h3 h4
  sorry

end emma_final_amount_l456_456312


namespace even_num_students_count_l456_456561

-- Define the number of students in each school
def num_students_A : Nat := 786
def num_students_B : Nat := 777
def num_students_C : Nat := 762
def num_students_D : Nat := 819
def num_students_E : Nat := 493

-- Define a predicate to check if a number is even
def is_even (n : Nat) : Prop := n % 2 = 0

-- The theorem to state the problem
theorem even_num_students_count :
  (is_even num_students_A ∧ is_even num_students_C) ∧ ¬(is_even num_students_B ∧ is_even num_students_D ∧ is_even num_students_E) →
  2 = 2 :=
by
  sorry

end even_num_students_count_l456_456561


namespace january_first_day_of_week_l456_456762

theorem january_first_day_of_week (is_jan_31_days : True) 
  (five_mondays : True) 
  (four_thursdays : True) : 
  "January 1 is a Monday" := 
by 
  sorry

end january_first_day_of_week_l456_456762


namespace find_length_MN_l456_456686

noncomputable def length_of_MN (R S : ℝ) (A M N : ℂ) : ℝ := 
  (2 - real.sqrt 3) / real.sqrt 5

-- Definitions based on the conditions provided
def radius_of_circle (C : Type*) := {R : ℝ // 0 < R}
def point_on_circumference (C : Type*) (R : ℝ) := {A : ℂ // abs A = R}
def secant_intersects_circle (C : Type*) (A : ℂ) := {MN : ℂ × ℂ // ¬∀ P, P ∈ finset.singleton A → P = 0}
def area_of_triangle (C : Type*) (K : ℂ) (M N : ℂ) := triangle_area K M N

-- Example statement
theorem find_length_MN (C : Type*) (R S : ℝ) (A M N K : ℂ)
  (hR : radius_of_circle C)
  (hA : point_on_circumference C R)
  (hM : secant_intersects_circle C A) 
  (hS : area_of_triangle C K M N = S) :
  (length_of_MN R S A M N = (2 - real.sqrt 3) / real.sqrt 5) :=
sorry

end find_length_MN_l456_456686


namespace parallel_lines_l456_456673

variable {R : Type} [LinearOrderedField R]

def line1 (a : R) : R → R → Prop :=
  λ x y, a * x + 2 * y + 1 = 0

def line2 (a : R) : R → R → Prop :=
  λ x y, 3 * x + (a - 1) * y + 1 = 0

theorem parallel_lines (a : R) :
  (∀ x y z w, line1 a x y → line2 a z w → (2 * z = 3 * w * x - z + y * (a - 1 + a)) → a = -2) :=
sorry

end parallel_lines_l456_456673


namespace perfectSquareLastFourDigits_l456_456299

noncomputable def lastThreeDigitsForm (n : ℕ) : Prop :=
  ∃ a : ℕ, a ≤ 9 ∧ n % 1000 = a * 111

noncomputable def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem perfectSquareLastFourDigits (n : ℕ) :
  lastThreeDigitsForm n →
  isPerfectSquare n →
  (n % 10000 = 0 ∨ n % 10000 = 1444) :=
by {
  sorry
}

end perfectSquareLastFourDigits_l456_456299


namespace solve_for_y_l456_456652

theorem solve_for_y (x y : ℝ) (h : 2 * y - 4 * x + 5 = 0) : y = 2 * x - 2.5 :=
sorry

end solve_for_y_l456_456652


namespace projection_of_AC_on_BD_l456_456358

def vector_projection (A B C D : ℝ × ℝ) : ℝ :=
  let AC := (C.1 - A.1, C.2 - A.2)
  let BD := (D.1 - B.1, D.2 - B.2)
  let dot_product := AC.1 * BD.1 + AC.2 * BD.2
  let magnitude_BD := Real.sqrt (BD.1 * BD.1 + BD.2 * BD.2)
  dot_product / magnitude_BD

theorem projection_of_AC_on_BD :
  vector_projection (-1, 1) (1, 2) (-2, -1) (3, 4) = -3 * Real.sqrt 2 / 2 :=
by
  sorry

end projection_of_AC_on_BD_l456_456358


namespace largest_and_smallest_value_of_expression_l456_456321

theorem largest_and_smallest_value_of_expression
  (w x y z : ℝ)
  (h1 : w + x + y + z = 0)
  (h2 : w^7 + x^7 + y^7 + z^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 :=
sorry

end largest_and_smallest_value_of_expression_l456_456321


namespace xyz_abs_eq_one_l456_456463

theorem xyz_abs_eq_one (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (cond : x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x + 1) : |x * y * z| = 1 :=
sorry

end xyz_abs_eq_one_l456_456463


namespace find_g_l456_456456

def f (x : ℝ) : ℝ := 2 * x^3 - x^2 - 4 * x + 5
def h (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 4

def g_correct (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x + g x = h x

theorem find_g (g : ℝ → ℝ) (hcond : g_correct g) :
  g = λ x, -2 * x^3 + 3 * x^2 + 7 * x - 1 :=
by
  sorry

end find_g_l456_456456


namespace circumcircles_intersection_l456_456846

open EuclideanGeometry

noncomputable def triangle_with_similar_points (A B C A1 B1 C1 A2 B2 C2 : Point) :=
  (triangle A B C ∧ triangle A1 B1 C1 ∧ similar (triangle.mk A B C) (triangle.mk A1 B1 C1)) ∧
  (colinear B B1 A2 ∧ colinear C C1 A2 ∧ colinear C C1 B2 ∧ colinear A A1 B2 ∧ colinear A A1 C2 ∧ colinear B B1 C2)
  
theorem circumcircles_intersection
  (A B C A1 B1 C1 A2 B2 C2 : Point)
  (h₀ : triangle_with_similar_points A B C A1 B1 C1 A2 B2 C2) :
  ∃ O : Point, 
    (circle (triangle_circumcircle A B C2)).center = O ∧
    (circle (triangle_circumcircle B C A2)).center = O ∧
    (circle (triangle_circumcircle C A B2)).center = O ∧
    (circle (triangle_circumcircle A1 B1 C2)).center = O ∧
    (circle (triangle_circumcircle B1 C1 A2)).center = O ∧
    (circle (triangle_circumcircle C1 A1 B2)).center = O := 
sorry

end circumcircles_intersection_l456_456846


namespace speed_of_Jim_and_Marcus_l456_456986

theorem speed_of_Jim_and_Marcus
  (d1 d2 d3 : ℕ) (v2 v3 : ℕ) (total_distance : ℕ) (total_time : ℝ) 
  (h1 : d1 = 3) (h2 : v2 = 3) (h3 : d2 = 3) (h4 : v3 = 8) (h5 : d3 = 4) 
  (h6 : total_distance = 10) (h7 : total_time = 2) :
  ∃ v1 : ℝ, v1 = 6 :=
by
  let t1 := d1 / v1
  have t2 := d2 / v2
  have t3 := d3 / v3
  have h8 : total_time = t1 + t2 + t3 := by
    simp [t1, t2, t3, h1, h3, h5, h2, h4, h7]
    sorry
  use 6
  simp [h8]
  sorry

end speed_of_Jim_and_Marcus_l456_456986


namespace total_pixels_l456_456604

theorem total_pixels (width height dpi : ℕ) (h_width : width = 21) (h_height : height = 12) (h_dpi : dpi = 100) :
  width * dpi * height * dpi = 2520000 :=
by
  rw [h_width, h_height, h_dpi]
  norm_num
  sorry

end total_pixels_l456_456604


namespace turtle_initial_coins_l456_456638

theorem turtle_initial_coins (x : ℕ) (h : x = 15) :
  (3 * (3 * x - 30) - 30) = 0 :=
by
  rw h
  sorry

end turtle_initial_coins_l456_456638


namespace dawson_marks_l456_456639

theorem dawson_marks :
  ∀ (max_marks : ℕ) (passing_percentage : ℕ) (failed_by : ℕ) (M : ℕ),
  max_marks = 220 →
  passing_percentage = 30 →
  failed_by = 36 →
  M = (passing_percentage * max_marks / 100) - failed_by →
  M = 30 := by
  intros max_marks passing_percentage failed_by M h_max h_percent h_failed h_M
  rw [h_max, h_percent, h_failed] at h_M
  norm_num at h_M
  exact h_M

end dawson_marks_l456_456639


namespace intersection_points_sum_l456_456879

theorem intersection_points_sum (x1 x2 x3 y1 y2 y3 A B : ℝ)
(h1 : y1 = x1^3 - 3 * x1 + 2)
(h2 : x1 + 6 * y1 = 6)
(h3 : y2 = x2^3 - 3 * x2 + 2)
(h4 : x2 + 6 * y2 = 6)
(h5 : y3 = x3^3 - 3 * x3 + 2)
(h6 : x3 + 6 * y3 = 6)
(hA : A = x1 + x2 + x3)
(hB : B = y1 + y2 + y3) :
A = 0 ∧ B = 3 := 
by
  sorry

end intersection_points_sum_l456_456879


namespace books_fraction_sold_l456_456956

theorem books_fraction_sold (B : ℕ) (h1 : B - 36 * 2 = 144) :
  (B - 36) / B = 2 / 3 := by
  sorry

end books_fraction_sold_l456_456956


namespace value_of_expression_l456_456190

theorem value_of_expression :
  (0.00001 * (0.01)^2 * 1000) / 0.001 = 10^(-3) :=
by
  -- Proof goes here
  sorry

end value_of_expression_l456_456190


namespace compare_times_l456_456797

variable {v : ℝ} (h_v_pos : 0 < v)

/-- 
  Jones covered a distance of 80 miles on his first trip at speed v.
  On a later trip, he traveled 360 miles at four times his original speed.
  Prove that his new time is (9/8) times his original time.
-/
theorem compare_times :
  let t1 := 80 / v
  let t2 := 360 / (4 * v)
  t2 = (9 / 8) * t1 :=
by
  sorry

end compare_times_l456_456797


namespace center_of_circle_l456_456506

theorem center_of_circle (h k : ℝ) :
  (∀ x y : ℝ, (x - 3) ^ 2 + (y - 4) ^ 2 = 10 ↔ x ^ 2 + y ^ 2 = 6 * x + 8 * y - 15) → 
  h + k = 7 :=
sorry

end center_of_circle_l456_456506


namespace correct_total_bill_l456_456283

noncomputable def total_bill (cost_clothes cost_acc shipping_clothes shipping_acc tax_clothes tax_acc : ℚ) : ℚ :=
  cost_clothes + shipping_clothes + tax_clothes + cost_acc + shipping_acc + tax_acc

def main : ℚ :=
let cost_clothes := 3 * 12 + 2 * 15 + 14 + 5 in
let cost_acc := 6 + 30 in
let shipping_clothes := 0.30 * cost_clothes in
let shipping_acc := 0.15 * cost_acc in
let tax_clothes := 0.10 * cost_clothes in
let tax_acc := 0.05 * cost_acc in
total_bill cost_clothes cost_acc shipping_clothes shipping_acc tax_clothes tax_acc

theorem correct_total_bill : main = 162.20 := sorry

end correct_total_bill_l456_456283


namespace isosceles_triangle_remainder_25_l456_456443

theorem isosceles_triangle_remainder_25 :
  let n := 2019
  let num_isosceles := (n - 1) / 2 * n - (2 / 3) * n
  (num_isosceles % 100) = 25 :=
by
  -- Definitions
  let n := 2019
  let floor_div := ((n - 1) / 2).toInt -- Floor division
  let fraction := (2 * n) / 3
  let num_isosceles := floor_div * n - fraction
  -- Prove
  have h1 : (floor_div = 1009) := by sorry
  have h2 : (fraction = 2 * 673) := by sorry
  have h3 : (num_isosceles = 2037171 - 1346) := by sorry
  exact 2037171 % 100 - 1346 % 100 = 25

end isosceles_triangle_remainder_25_l456_456443


namespace determinant_of_trig_matrix_is_zero_l456_456115

theorem determinant_of_trig_matrix_is_zero 
  (A B C : ℝ) 
  (hA : A + B + C = π) : 
  matrix.det ![
    ![cos A ^ 2, tan A, 1],
    ![cos B ^ 2, tan B, 1],
    ![cos C ^ 2, tan C, 1]
  ] = 0 :=
by
  sorry

end determinant_of_trig_matrix_is_zero_l456_456115


namespace square_inscribed_ratios_l456_456977

noncomputable def square_ratio (x y : ℝ) := x / y

theorem square_inscribed_ratios (x y : ℝ) 
  (h1 : ∃ (t : Triangle), t.is_right_triangle ∧ t.has_sides 6 8 10 ∧ t.square_inscribed_at_right_angle.has_side_length x)
  (h2 : ∃ (t' : Triangle), t'.is_right_triangle ∧ t'.has_sides 6 8 10 ∧ t'.square_inscribed_on_hypotenuse.has_side_length y) :
  square_ratio x y = 37 / 35 :=
by 
  sorry

end square_inscribed_ratios_l456_456977


namespace merchant_installed_zucchini_l456_456252

theorem merchant_installed_zucchini (Z : ℕ) : 
  (15 + Z + 8) / 2 = 18 → Z = 13 :=
by
 sorry

end merchant_installed_zucchini_l456_456252


namespace base_seven_representation_l456_456330

theorem base_seven_representation 
  (k : ℕ) 
  (h1 : 4 ≤ k) 
  (h2 : k < 8) 
  (h3 : 500 / k^3 < k) 
  (h4 : 500 ≥ k^3) 
  : ∃ n m o p : ℕ, (500 = n * k^3 + m * k^2 + o * k + p) ∧ (p % 2 = 1) ∧ (n ≠ 0 ) :=
sorry

end base_seven_representation_l456_456330


namespace hyperbola_focal_distance_l456_456720

theorem hyperbola_focal_distance :
  (∀ x y : ℝ, (x^2 / 20 - y^2 / 5 = 1) → 2 * (Real.sqrt (20 + 5)) = 10) :=
by
  intro x y h
  calc 
    2 * (Real.sqrt (20 + 5)) = 2 * 5       := by sorry
    ...                       = 10          := by sorry

end hyperbola_focal_distance_l456_456720


namespace permutation_residue_condition_l456_456655

section
variables {n : ℕ}
variables (p : Fin n → Fin n)

noncomputable def is_permutation (p : Fin n → Fin n) := 
  ∀ x, ∃! y, p y = x

def is_complete_residue_system (s : Fin n → Fin n) := 
  ∀ i j : Fin n, i ≠ j → s i ≠ s j

def sets_form_complete_residue_systems (p : Fin n → Fin n) := 
  is_complete_residue_system (λ i, (p i + i + 1) % n) ∧ 
  is_complete_residue_system (λ i, (p i - i + n.pred + 1) % n)

theorem permutation_residue_condition (hp : is_permutation p) :
  sets_form_complete_residue_systems p ↔ (n % 6 = 1 ∨ n % 6 = 5) := 
sorry

end

end permutation_residue_condition_l456_456655


namespace toothpicks_15_l456_456514

noncomputable def toothpicks : ℕ → ℕ
| 0       => 0  -- since the stage count n >= 1, stage 0 is not required, default 0.
| 1       => 5
| (n + 1) => 2 * toothpicks n + 2

theorem toothpicks_15 : toothpicks 15 = 32766 := by
  sorry

end toothpicks_15_l456_456514


namespace tank_capacity_l456_456249

theorem tank_capacity (x : ℝ) (h₁ : 0.25 * x = 60) (h₂ : 0.05 * x = 12) : x = 240 :=
sorry

end tank_capacity_l456_456249


namespace pauline_spent_in_all_l456_456141

theorem pauline_spent_in_all
  (cost_taco_shells : ℝ := 5)
  (cost_bell_pepper : ℝ := 1.5)
  (num_bell_peppers : ℕ := 4)
  (cost_meat_per_pound : ℝ := 3)
  (num_pounds_meat : ℝ := 2) :
  (cost_taco_shells + num_bell_peppers * cost_bell_pepper + num_pounds_meat * cost_meat_per_pound = 17) :=
by
  sorry

end pauline_spent_in_all_l456_456141


namespace triangle_area_l456_456534

theorem triangle_area (a b c : ℝ) (h1 : a / b = 3 / 4) (h2 : b / c = 4 / 5) (h3 : a + b + c = 60) : 
  (1/2) * a * b = 150 :=
by
  sorry

end triangle_area_l456_456534


namespace perpendicular_diagonal_quadrilateral_iff_l456_456810

theorem perpendicular_diagonal_quadrilateral_iff
  (A B C D : ℝ × ℝ)
  (h_AB : (B.1 - A.1)^2 + (B.2 - A.2)^2)
  (h_CD : (D.1 - C.1)^2 + (D.2 - C.2)^2)
  (h_BC : (C.1 - B.1)^2 + (C.2 - B.2)^2)
  (h_AD : (D.1 - A.1)^2 + (D.2 - A.2)^2):
  (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0 ↔
  (B.1 - A.1)^2 + (B.2 - A.2)^2 + (D.1 - C.1)^2 + (D.2 - C.2)^2 = 
  (C.1 - B.1)^2 + (C.2 - B.2)^2 + (D.1 - A.1)^2 + (D.2 - A.2)^2 :=
sorry

end perpendicular_diagonal_quadrilateral_iff_l456_456810


namespace extrema_points_range_l456_456388

theorem extrema_points_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → (differentiable ℝ (λ x, real.exp x * (x - a * real.exp x))) →
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ (deriv (λ x, real.exp x * (x - a * real.exp x)) x₁ = 0) ∧ 
  (deriv (λ x, real.exp x * (x - a * real.exp x)) x₂ = 0)) →
  (0 < a ∧ a < 1/2) := 
sorry

end extrema_points_range_l456_456388


namespace find_ab_l456_456888

-- Definitions of the polynomials and their coefficients
def polynomial (a b : ℤ) : ℤ → ℤ := λ x, a * x ^ 4 + b * x ^ 3 + 32 * x ^ 2 - 16 * x + 6
def factor : ℤ → ℤ := λ x, 3 * x ^ 2 - 2 * x + 1

-- Parameters from the condition
variables (a b c d k : ℤ)
hypothesis (H1 : factor = λ x, 3 * x ^ 2 - 2 * x + 1)
hypothesis (H2 : polynomial a b = λ x, (3 * x ^ 2 - 2 * x + 1) * (c * x ^ 2 + d * x + k))
hypothesis (H3 : k = 6)
hypothesis (H4 : d - 12 = -16)
hypothesis (H5 : c - 2 * d + 18 = 32)

-- The goal to prove
theorem find_ab : a = 18 ∧ b = -24 :=
by {
  sorry
}

end find_ab_l456_456888


namespace elaine_rent_percentage_l456_456802

-- Defining the conditions
variables (E : ℝ) -- Elaine's earnings last year
variables (rent_last_year : ℝ) (earnings_this_year : ℝ) (rent_this_year : ℝ)

-- Given conditions
def last_year_rent := rent_last_year = 0.10 * E
def this_year_earnings := earnings_this_year = 1.15 * E
def rent_this_year_definition := rent_this_year = 3.45 * rent_last_year

-- The percentage of earnings spent on rent this year
def percentage_rent_this_year := (rent_this_year / earnings_this_year) * 100

-- The Main theorem to prove
theorem elaine_rent_percentage
  (h1 : last_year_rent)
  (h2 : this_year_earnings)
  (h3 : rent_this_year_definition) :
  percentage_rent_this_year = 30 :=
by
  sorry

end elaine_rent_percentage_l456_456802


namespace total_cost_correct_lowest_selling_price_correct_highest_profit_correct_earnings_correct_l456_456953

open Int

-- Define conditions
def quantity := 30
def cost_per_dress := 32
def standard_price := 40
def deviations := [5, 2, 1, -2, -4, -6]
def sold_quantities := [4, 9, 3, 5, 4, 5]

-- Define the expected answers
def total_cost := 960
def lowest_selling_price := 34
def highest_profit := 13
def earnings := 225

-- Proof statements (without the proofs)
theorem total_cost_correct : quantity * cost_per_dress = total_cost := sorry

theorem lowest_selling_price_correct :
  standard_price - List.maximum deviations = lowest_selling_price := sorry

theorem highest_profit_correct :
  (standard_price + List.maximum deviations) - cost_per_dress = highest_profit := sorry

theorem earnings_correct :
  (sum (List.map2 (*) sold_quantities (List.map (λ x => standard_price + x) deviations))) - total_cost = earnings := sorry

end total_cost_correct_lowest_selling_price_correct_highest_profit_correct_earnings_correct_l456_456953


namespace point_closer_to_center_probability_l456_456263

noncomputable def probability_closer_to_center_than_boundary (r_outer : ℝ) (r_inner : ℝ) : ℝ :=
  (Math.pi * r_inner^2) / (Math.pi * r_outer^2)

theorem point_closer_to_center_probability : probability_closer_to_center_than_boundary 4 2 = 1 / 4 := 
by
  sorry

end point_closer_to_center_probability_l456_456263


namespace term_150_of_sequence_l456_456521

noncomputable def sequence : ℕ → ℕ
| 1 := 1
| 2 := 3
| 3 := 4
| n := if h : ∃ k, 3^k = n then n
       else if h : ∃ k (l : ℕ), n = 3^k + sequence l ∧ ∀m < l, sequence m ≠ sequence l then n
       else sequence (n - 1) + 1 -- The exact construction of the sequence can be complex.

theorem term_150_of_sequence : sequence 150 = 2280 := 
by sorry  -- Proof omitted

end term_150_of_sequence_l456_456521


namespace black_to_white_area_ratio_l456_456668

-- The radii of the concentric circles
def radii : List ℕ := [1, 3, 5, 7, 9]

-- Function to compute the area of a circle given its radius
def area (r : ℕ) : ℝ := π * (r : ℝ)^2

-- Given areas of the circles
def circle_areas : List ℝ := radii.map area

-- Areas of the circular rings
def ring_areas : List ℝ :=
  List.zipWith (λ a b, a - b) circle_areas.tail circle_areas

-- Separating black and white areas based on their positions
def black_areas : List ℝ := [circle_areas.head!, ring_areas[1], ring_areas[3]]
def white_areas : List ℝ := [ring_areas[0], ring_areas[2]]

-- Summing up the areas
def total_black_area : ℝ := black_areas.sum
def total_white_area : ℝ := white_areas.sum

-- Ratio of black area to white area
def ratio : ℝ := total_black_area / total_white_area

-- Statement to prove
theorem black_to_white_area_ratio : ratio = 49 / 32 := by
  sorry

end black_to_white_area_ratio_l456_456668


namespace probability_nine_moves_visits_all_vertices_l456_456588

noncomputable def probability_bug_visits_all_vertices : ℚ :=
  4 / 243

theorem probability_nine_moves_visits_all_vertices :
  ∀ (start_vertex : Fin 8), 
  ∀ (move_probability : ∀ (v : Fin 8), Fin 3 → ℚ),
  (∀ (v : Fin 8) (edge : Fin 3), move_probability v edge = 1 / 3) →
  Probability.all_vertices_visited (cube : Graph) start_vertex 9 move_probability 
  = probability_bug_visits_all_vertices :=
sorry

end probability_nine_moves_visits_all_vertices_l456_456588


namespace find_quantities_l456_456181

variables {a b x y : ℝ}

-- Original total expenditure condition
axiom h1 : a * x + b * y = 1500

-- New prices and quantities for the first scenario
axiom h2 : (a + 1.5) * (x - 10) + (b + 1) * y = 1529

-- New prices and quantities for the second scenario
axiom h3 : (a + 1) * (x - 5) + (b + 1) * y = 1563.5

-- Inequality constraint
axiom h4 : 205 < 2 * x + y ∧ 2 * x + y < 210

-- Range for 'a'
axiom h5 : 17.5 < a ∧ a < 18.5

-- Proving x and y are specific values.
theorem find_quantities :
  x = 76 ∧ y = 55 :=
sorry

end find_quantities_l456_456181


namespace problem_1_l456_456233

theorem problem_1 (a b : ℝ) (h : b < a ∧ a < 0) : 
  (a + b < a * b) ∧ (¬ (abs a > abs b)) ∧ (¬ (1 / b > 1 / a ∧ 1 / a > 0)) ∧ (¬ (b / a + a / b > 2)) := sorry

end problem_1_l456_456233


namespace emma_final_amount_l456_456311

theorem emma_final_amount :
  ∀ (amount_from_bank spent_on_furniture remaining after_giving_friend: ℝ),
  amount_from_bank = 2000 →
  spent_on_furniture = 400 →
  remaining = amount_from_bank - spent_on_furniture →
  after_giving_friend = (3 / 4) * remaining →
  remaining - after_giving_friend = 400 :=
by
  intros amount_from_bank spent_on_furniture remaining after_giving_friend
  assume h1 h2 h3 h4
  sorry

end emma_final_amount_l456_456311


namespace roger_bike_rides_total_l456_456859

theorem roger_bike_rides_total 
  (r1 : ℕ) (h1 : r1 = 2) 
  (r2 : ℕ) (h2 : r2 = 5 * r1) 
  (r : ℕ) (h : r = r1 + r2) : 
  r = 12 := 
by
  sorry

end roger_bike_rides_total_l456_456859


namespace intersection_ratios_eq_squares_l456_456818

theorem intersection_ratios_eq_squares
  (A B C B1 C1 P : Point)
  (AC B1C1 AB BC PC PB BB1 CC1 : ℝ)
  (hB1C1 : B1 ∈ line(A, C))
  (hC1C1 : C1 ∈ line(A, B))
  (hP : P = intersection(line(B, B1), line(C, C1)))
  (hRatios : AC / AB = BC / B1C1) :
  (PB / BB1) = (PC / CC1) ^ 2 := by
  sorry

end intersection_ratios_eq_squares_l456_456818


namespace elasticity_ratio_approximation_l456_456628

def qN := 1.01
def pN := 0.61

theorem elasticity_ratio_approximation : (qN / pN) ≈ 1.7 := by
  sorry

end elasticity_ratio_approximation_l456_456628


namespace quadrilateral_ABCD_is_rhombus_l456_456706

-- Definitions for vectors and length
variable {A B C D : Type*} [metric_space A]

-- Define the conditions
def eq_vec (v1 v2 : A) (b1 b2 : B) : Prop := (dist v1 b1 = dist v2 b2)
def vec_len (v : A) : ℝ := dist v

-- Rewrite the problem conditions in Lean
axiom AB_AD_eq : vec_len A = vec_len D
axiom BA_CD_eq : eq_vec B A C D

-- The proof goal
theorem quadrilateral_ABCD_is_rhombus : 
  AB_AD_eq → BA_CD_eq → (quadrilateral ABCD = rhombus) :=
by
  intros
  sorry

end quadrilateral_ABCD_is_rhombus_l456_456706


namespace pies_made_l456_456169

-- Define the initial number of apples
def initial_apples : Nat := 62

-- Define the number of apples handed out to students
def handed_out_apples : Nat := 8

-- Define the number of apples required per pie
def apples_per_pie : Nat := 9

-- Define the number of remaining apples after handing out to students
def remaining_apples : Nat := initial_apples - handed_out_apples

-- State the theorem
theorem pies_made (initial_apples handed_out_apples apples_per_pie remaining_apples : Nat) :
  initial_apples = 62 →
  handed_out_apples = 8 →
  apples_per_pie = 9 →
  remaining_apples = initial_apples - handed_out_apples →
  remaining_apples / apples_per_pie = 6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end pies_made_l456_456169


namespace spherical_to_rectangular_coords_l456_456293

theorem spherical_to_rectangular_coords :
  ∀ (ρ θ φ : ℝ), ρ = 6 → θ = π / 4 → φ = π / 3 →
  let x := ρ * sin φ * cos θ in
  let y := ρ * sin φ * sin θ in
  let z := ρ * cos φ in
  (x, y, z) = (3 * sqrt 6, 3 * sqrt 6, 3) :=
by
  intros ρ θ φ hρ hθ hφ
  rw [hρ, hθ, hφ]
  let x := 6 * sin (π / 3) * cos (π / 4)
  let y := 6 * sin (π / 3) * sin (π / 4)
  let z := 6 * cos (π / 3)
  have hx : x = 3 * sqrt 6 := by sorry
  have hy : y = 3 * sqrt 6 := by sorry
  have hz : z = 3 := by sorry
  exact ⟨hx, hy, hz⟩

end spherical_to_rectangular_coords_l456_456293


namespace two_times_difference_eq_20_l456_456894

theorem two_times_difference_eq_20 (x y : ℕ) (hx : x = 30) (hy : y = 20) (hsum : x + y = 50) : 2 * (x - y) = 20 := by
  sorry

end two_times_difference_eq_20_l456_456894


namespace perfect_square_fraction_l456_456106

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem perfect_square_fraction (a b : ℕ) 
  (h_pos_a: 0 < a) 
  (h_pos_b: 0 < b) 
  (h_div : (a * b + 1) ∣ (a^2 + b^2)) : 
  is_perfect_square ((a^2 + b^2) / (a * b + 1)) := 
sorry

end perfect_square_fraction_l456_456106


namespace parabola_directrix_l456_456511

theorem parabola_directrix (p : ℝ) (hp : p > 0) (H : - (p / 2) = -3) : p = 6 :=
by
  sorry

end parabola_directrix_l456_456511


namespace negation_of_universal_proposition_l456_456179

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 2^x - 1 > 0)) ↔ (∃ x : ℝ, x^2 + 2^x - 1 ≤ 0) :=
by
  sorry

end negation_of_universal_proposition_l456_456179


namespace runners_meetings_l456_456203

noncomputable theory
open_locale classical

-- Definitions of the speeds of the runners and the total laps
def runner1_speed : ℕ := 6
def runner2_speed : ℕ := 10
def total_laps : ℕ := 5

-- Definition of the total number of meetings, excluding start and finish
def number_of_meetings : ℕ :=
let relative_speed := runner1_speed + runner2_speed in
let slow_runner_time_to_complete := (total_laps * 2) / runner1_speed in
let total_distance_covered := relative_speed * slow_runner_time_to_complete in
(total_distance_covered / (total_laps * 2)) - 1

-- The theorem to prove
theorem runners_meetings (runner1_speed runner2_speed total_laps : ℕ)
    (h1 : runner1_speed = 6) (h2 : runner2_speed = 10) (h3 : total_laps = 5) :
    number_of_meetings = 12 :=
by {
  rw [h1, h2, h3],
  unfold number_of_meetings,
  rw [nat.add_comm, nat.mul_comm],
  sorry
}

end runners_meetings_l456_456203


namespace triangle_area_l456_456447

def is_point_on_ellipse (P : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), P = (x, y) ∧ (x^2 / 9 + y^2 / 4 = 1)

def are_foci_of_ellipse (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = (-3, 0) ∧ F2 = (3, 0)

def segment_ratio (P F1 F2 : ℝ × ℝ) : Prop :=
  let PF1 := (P.1 - F1.1)^2 + (P.2 - F1.2)^2 in
  let PF2 := (P.1 - F2.1)^2 + (P.2 - F2.2)^2 in
  (PF1 / PF2 = 4)

theorem triangle_area (P F1 F2 : ℝ × ℝ) 
  (h_ellipse : is_point_on_ellipse P)
  (h_foci : are_foci_of_ellipse F1 F2)
  (h_ratio : segment_ratio P F1 F2) :
  let PF1_dist := real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) in
  let PF2_dist := real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) in
  PF1_dist = 4 ∧ PF2_dist = 2 ∧ 
  let area := 1/2 * PF1_dist * PF2_dist in 
  area = 4 :=
sorry

end triangle_area_l456_456447


namespace polynomial_value_l456_456402

variable (a b : ℝ)

theorem polynomial_value :
  2 * a + 3 * b = 5 → 6 * a + 9 * b - 12 = 3 :=
by
  intro h
  sorry

end polynomial_value_l456_456402


namespace toby_first_rectangle_height_l456_456195

/-- Toby has two rectangles of cloth:
    The first rectangle is 4 inches wide and some inches tall. 
    The second rectangle is 3 inches wide and 6 inches tall. 
    The area of the first rectangle is 2 square inches more than the area of the second rectangle.
    We need to prove that the height of the first rectangle is 5 inches. -/
theorem toby_first_rectangle_height
  (h : ℕ)
  (area_first : 4 * h)
  (area_second : 3 * 6 = 18)
  (area_condition : 4 * h = 18 + 2) :
  h = 5 :=
by
  sorry

end toby_first_rectangle_height_l456_456195


namespace quadratic_trinomial_factorization_l456_456714

theorem quadratic_trinomial_factorization 
    (m n : ℝ) 
    (h1 : ∃ (f : ℝ → ℝ), f = (λ x, x^2 - m * x + n) ∧ f 3 = 0 ∧ f (-4) = 0) : 
    (n = -12 ∧ m = -1) :=
by
  sorry

end quadratic_trinomial_factorization_l456_456714


namespace kolya_max_finish_points_l456_456579

noncomputable def number_of_finish_points (n : ℕ) : ℕ :=
  nat.choose n (n/2)  -- choose(n, n/2)

theorem kolya_max_finish_points : 
  number_of_finish_points 2020 = nat.choose 2020 1010 :=
sorry

end kolya_max_finish_points_l456_456579


namespace problem1_problem2_problem3_l456_456993

theorem problem1 : -3^2 + (-1/2)^2 + (2023 - Real.pi)^0 - |-2| = -47/4 :=
by
  sorry

theorem problem2 (a : ℝ) : (-2 * a^2)^3 * a^2 + a^8 = -7 * a^8 :=
by
  sorry

theorem problem3 : 2023^2 - 2024 * 2022 = 1 :=
by
  sorry

end problem1_problem2_problem3_l456_456993


namespace equilateral_triangle_of_equal_altitude_ratios_l456_456488

/-- If the point of intersection of the altitudes of an acute-angled triangle divides the altitudes in the same ratio, then the triangle is equilateral -/
theorem equilateral_triangle_of_equal_altitude_ratios (ABC : Triangle) (H : Point)
  (A1 B1 C1 : Point) (AA1 BB1 CC1 : Line)
  (hA1 : A1 ∈ AA1) (hA1H : H ∈ AA1) (hA1A : A ∈ AA1)
  (hB1 : B1 ∈ BB1) (hB1H : H ∈ BB1) (hB1B : B ∈ BB1)
  (hC1 : C1 ∈ CC1) (hC1H : H ∈ CC1) (hC1C : C ∈ CC1)
  (r : α)
  (h1 : A1H ● BH = B1H ● AH)
  (hacute : triangle_is_acute ABC) :
  triangle_is_equilateral ABC :=
  sorry

end equilateral_triangle_of_equal_altitude_ratios_l456_456488


namespace min_magnitude_is_sqrt_3_l456_456709

noncomputable def min_vector_addition_magnitude
  (a b : ℝ → ℝ)
  (angle_ab : ℝ)
  (norm_a : ℝ) : ℝ :=
  ∃ λ : ℝ, angle_ab = 120 ∧ norm_a = 2 ∧ (∀ λ : ℝ, |a + λ * b| ≥ sqrt(3))

theorem min_magnitude_is_sqrt_3 (a b : ℝ → ℝ) (angle_ab : ℝ) (norm_a : ℝ) :
  angle_ab = 120 → norm_a = 2 →
  ∃ λ : ℝ, (|a + λ * b| = sqrt(3)) :=
by
  sorry

end min_magnitude_is_sqrt_3_l456_456709


namespace num_special_five_digit_numbers_l456_456066

def count_special_five_digit_numbers : ℕ :=
  ((fact 4) / (fact 2 * fact 2)) + ((fact 4) / (fact 2 * fact 1 * fact 1))

theorem num_special_five_digit_numbers : count_special_five_digit_numbers = 18 := 
by
  sorry

end num_special_five_digit_numbers_l456_456066


namespace team_b_wins_first_game_probability_l456_456503

/-- Team A and Team B play a series where the first team to win four games wins the series.
Each team is equally likely to win each game (probability 1/2), there are no ties, 
and the outcomes of the individual games are independent. 
If Team B wins the third game and Team A wins the series, 
prove that the probability that Team B wins the first game is 2/3. -/
theorem team_b_wins_first_game_probability : 
  ∀ (A B : Type) [ProbSpace A B] (win : A → B → Prop), 
  (∀ (X Y : A → B → Prop), P(X) = 1/2 ∧ P(Y) = 1/2) →
  (prob_series_wins : ∀ (X : A → B → Prop), prob_wins_series A = 4 ∧ prob_wins_games B 3) →
  (independent_games : ∀ (X Y : A → B → Prop), independent_trials X Y) →
  (no_ties : ∀ (X : A → B → Prop), X ≠ Y) →
  (P(team_b_wins_first_game | team_a_wins_series ∧ team_b_wins_third_game) = 2/3) :=
begin
  sorry
end

end team_b_wins_first_game_probability_l456_456503


namespace simple_interest_double_l456_456761

theorem simple_interest_double (P : ℝ) (r : ℝ) (t : ℝ) (A : ℝ)
  (h1 : t = 50)
  (h2 : A = 2 * P) 
  (h3 : A - P = P * r * t / 100) :
  r = 2 :=
by
  -- Proof is omitted
  sorry

end simple_interest_double_l456_456761


namespace pi_value_l456_456872

-- Define the conditions of the problem
def circumference := 48
def height := 11
def volume := 2112
def volume_formula (pi : ℝ) (circumference height : ℝ) : ℝ :=
  (1/12) * ((circumference^2) * height)

-- State the theorem to prove
theorem pi_value : ∃ (pi : ℝ), 
  (volume_formula pi circumference height = volume) ∧ 
  (2 * pi * (circumference / (2 * pi)) = circumference) ∧
  ({pi * (circumference / (2 * pi))^2 * height = volume}) ∧
  (pi = 3) :=
sorry

end pi_value_l456_456872


namespace incorrect_equation_B_l456_456928

-- Definitions of combinations and permutations used in the conditions
def C (n m : ℕ) : ℚ := nat.choose n m
def A (n m : ℕ) : ℚ := n.factorial / (n - m).factorial

-- Equation definitions
def EqA (n m : ℕ) : Prop := C n m = C n (n - m)
def EqB (n m : ℕ) : Prop := C n m = A n m / (n.factorial)
def EqC (n m : ℕ) : Prop := (n + 2) * (n + 1) * A n m = A (n + 2) (m + 2)
def EqD (n r : ℕ) : Prop := C n r = C (n - 1) (r - 1) + C (n - 1) r

-- Theorem to prove that equation B is incorrect
theorem incorrect_equation_B (n m : ℕ) : ¬ EqB n m := by
  sorry

end incorrect_equation_B_l456_456928


namespace eggs_remaining_l456_456838

-- Assign the given constants
def hens : ℕ := 3
def eggs_per_hen_per_day : ℕ := 3
def days_gone : ℕ := 7
def eggs_taken_by_neighbor : ℕ := 12
def eggs_dropped_by_myrtle : ℕ := 5

-- Calculate the expected number of eggs Myrtle should have
noncomputable def total_eggs :=
  hens * eggs_per_hen_per_day * days_gone - eggs_taken_by_neighbor - eggs_dropped_by_myrtle

-- Prove that the total number of eggs equals the correct answer
theorem eggs_remaining : total_eggs = 46 :=
by
  sorry

end eggs_remaining_l456_456838


namespace additional_time_to_walk_1_mile_l456_456926

open Real

noncomputable def additional_time_per_mile
  (distance_child : ℝ) (time_child : ℝ)
  (distance_elderly : ℝ) (time_elderly : ℝ)
  : ℝ :=
  let speed_child := distance_child / time_child
  let time_per_mile_child := (time_child * 60) / distance_child
  let speed_elderly := distance_elderly / time_elderly
  let time_per_mile_elderly := (time_elderly * 60) / distance_elderly
  time_per_mile_elderly - time_per_mile_child

theorem additional_time_to_walk_1_mile
  (h1 : 15 = 15) (h2 : 3.5 = 3.5)
  (h3 : 10 = 10) (h4 : 4 = 4)
  : additional_time_per_mile 15 3.5 10 4 = 10 :=
  by
    sorry

end additional_time_to_walk_1_mile_l456_456926


namespace smallest_non_lucky_multiple_of_8_l456_456664

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky_integer (n : ℕ) : Prop :=
  n % (sum_of_digits n) = 0

theorem smallest_non_lucky_multiple_of_8 : ∃ (m : ℕ), (m > 0) ∧ (m % 8 = 0) ∧ ¬ is_lucky_integer m ∧ m = 16 := sorry

end smallest_non_lucky_multiple_of_8_l456_456664


namespace average_is_30_l456_456567

/-
Question: Prove that the average of all the numbers between 7 and 49 which are divisible by 6 is 30.
-/
def S : Set ℕ := {n | n ≥ 7 ∧ n ≤ 49 ∧ n % 6 = 0}
def sum_S : ℕ := Finset.sum (Finset.filter (λ n => n ∈ S) (Finset.range 50)) id
def count_S : ℕ := Nat.card (Finset.filter (λ n => n ∈ S) (Finset.range 50))

theorem average_is_30 : (sum_S : ℚ) / count_S = 30 := 
by
  sorry

end average_is_30_l456_456567


namespace find_interest_rate_l456_456130

-- Translating the identified conditions into Lean definitions
def initial_deposit (P : ℝ) : Prop := P > 0
def compounded_semiannually (n : ℕ) : Prop := n = 2
def growth_in_sum (A : ℝ) (P : ℝ) : Prop := A = 1.1592740743 * P
def time_period (t : ℝ) : Prop := t = 2.5

theorem find_interest_rate (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) (A : ℝ)
  (h_init : initial_deposit P)
  (h_n : compounded_semiannually n)
  (h_A : growth_in_sum A P)
  (h_t : time_period t) :
  r = 0.06 :=
by
  sorry

end find_interest_rate_l456_456130


namespace probability_perfect_square_product_l456_456766

theorem probability_perfect_square_product :
  let total_outcomes := 10 * 8,
      favorable_outcomes := 12 in
  total_outcomes != 0 ∧
  (total_outcomes > 0 ∧ favorable_outcomes > 0 ∧ 
   favorable_outcomes / total_outcomes = 3 / 20) :=
by
  let total_outcomes := 10 * 8
  let favorable_outcomes := 12
  show total_outcomes != 0 ∧ 
       (total_outcomes > 0 ∧ favorable_outcomes > 0 ∧ 
        favorable_outcomes / total_outcomes = 3 / 20)
  sorry

end probability_perfect_square_product_l456_456766


namespace slower_train_pass_time_l456_456943

noncomputable def train_pass_time (length : ℝ) (speed1_kmh : ℝ) (speed2_kmh : ℝ) : ℝ :=
  let speed_conversion (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600
  let speed1 := speed_conversion speed1_kmh
  let speed2 := speed_conversion speed2_kmh
  let relative_speed := speed1 + speed2
  length / relative_speed

theorem slower_train_pass_time :
  train_pass_time 500 45 30 ≈ 24.01 :=
by
  sorry

end slower_train_pass_time_l456_456943


namespace incorrect_statement_about_linear_regression_l456_456214

theorem incorrect_statement_about_linear_regression
  (StatementA : "The two variables in a correlation do not imply a causal relationship")
  (StatementB : "Scatter plots can intuitively reflect the degree of correlation between data")
  (StatementC : "The regression line best represents the relationship between two variables that are linearly correlated")
  (StatementD : "Not every set of data has a regression equation. For example, when the linear correlation coefficient of a set of data is very small, there will not be a regression equation for this set of data."):
  StatementD :=
sorry

end incorrect_statement_about_linear_regression_l456_456214


namespace geometry_ABC_in_triangle_l456_456781

variables {A B C X P : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space X] [metric_space P]
variables (triangle_ABC : triangle A B C)

noncomputable def is_midpoint_of_arc (arc : arc A B) (C : Type*) : Type* := sorry
noncomputable def is_perpendicular_line (line1 : line X P) (line2 : line X C) : Prop := sorry

theorem geometry_ABC_in_triangle
  (h_angle_ACB_90 : ∠ A C B = 90)
  (h_AC_gt_BC : distance A C > distance B C)
  (h_X_midpoint : is_midpoint_of_arc (arc A B) C)
  (h_perpendicular_intersection : is_perpendicular_line (line C X) (line X P))
  (h_P_on_CA : on_line P (line A C)) :
  distance A P = distance B C :=
sorry

end geometry_ABC_in_triangle_l456_456781


namespace total_nails_polished_l456_456253

-- Defining the number of girls
def num_girls : ℕ := 5

-- Defining the number of fingers and toes per person
def num_fingers_per_person : ℕ := 10
def num_toes_per_person : ℕ := 10

-- Defining the total number of nails per person
def nails_per_person : ℕ := num_fingers_per_person + num_toes_per_person

-- The theorem stating that the total number of nails polished for 5 girls is 100 nails
theorem total_nails_polished : num_girls * nails_per_person = 100 := by
  sorry

end total_nails_polished_l456_456253


namespace number_of_balanced_integers_l456_456622

-- Definition of a balanced integer
def balanced_integer (n : ℕ) : Prop :=
  n >= 1000 ∧ n <= 9999 ∧
  (n / 1000 + (n % 1000) / 100 = (n % 100) / 10 + n % 10)

-- Theorem stating the number of balanced integers between 1000 and 9999
theorem number_of_balanced_integers : 
  (finset.filter balanced_integer (finset.range 10000)).card = 615 :=
sorry

end number_of_balanced_integers_l456_456622


namespace triangle_inequality_l456_456339

theorem triangle_inequality (a b c : ℝ) (habc_triangle : a + b > c ∧ b + c > a ∧ a + c > b) : 
  2 * (a^2 * b^2 + b^2 * c^2 + a^2 * c^2) > (a^4 + b^4 + c^4) :=
by
  sorry

end triangle_inequality_l456_456339


namespace probability_at_A_after_10_meters_l456_456109

noncomputable def P : ℕ → ℚ
| 0 := 1
| (n+1) := (1 - P n) / 3

theorem probability_at_A_after_10_meters :
  P 5 = 20 / 81 :=
sorry

end probability_at_A_after_10_meters_l456_456109


namespace max_plane_regions_l456_456601

theorem max_plane_regions (n : ℕ) : 
  let sum_n := n * (n + 1) / 2 in
  n_lines_divides_plane n ≤ 1 + sum_n :=
sorry

noncomputable def n_lines_divides_plane (n : ℕ) : ℕ := sorry

end max_plane_regions_l456_456601


namespace part1_part2_l456_456944

-- Conditions
def S_x (x : ℕ) (p : ℕ) (α : ℕ) [integralDomain ℕ] (v_p : ℕ → ℕ) [decidablePred (λ p, p.prime)] : set ℕ :=
  {p^α | (isPrimeFactor p x) ∧ (α ≥ 0) ∧ (p^α ∣ x) ∧ (α ≡ v_p x [MOD 2])}

def f (x : ℕ) [integralDomain ℕ] [decidablePred (λ p, p.prime)] (v_p : ℕ → ℕ) : ℕ :=
  if x = 1 then 1 else
  ∑ p in (S_x x (λ p, p.prime) (λ p, v_p x [MOD 2])), p

def a (n : ℕ) (m : ℕ) (f : ℕ → ℕ) : ℕ :=
  if n ≤ m then unknown_large_positive_integer else
  max (f(a n)) (f(a (n - 1) + 1)) ... (f(a (n - m) + m))

-- Problem statement
theorem part1 (x : ℕ) (A B : ℕ) (m : ℕ) [integralDomain ℕ] [decidablePred (λ p, p.prime)] (v_p : ℕ → ℕ) :
  (x > 1) ∧ (hasAtLeastTwoDistinctPrimeFactors x) → (0 < A) ∧ (A < 1) ∧ (f(x) < A * x + B) := sorry

theorem part2 (a_fun : ℕ → ℕ) (f : ℕ → ℕ) (Q : ℕ) (m : ℕ) [integralDomain ℕ] [decidablePred (λ p, p.prime)] (v_p : ℕ → ℕ) :
  (∀ n ∈ ℕ, a_fun (n + 1) = max {f (a_fun n), f (a_fun (n - 1) + 1), ..., f (a_fun (n - m) + m)}) ∧ (m > 0) → 
  (∃ Q, ∀ n ∈ ℕ, a_fun n < Q) := sorry

end part1_part2_l456_456944


namespace jake_final_distance_l456_456096

theorem jake_final_distance :
  let north_distance := 2
  let south_distance := 2
  let west_distance := 3 in
  -- Net north-south movement is 0 (2 miles north, then 2 miles south)
  let net_north_south := north_distance - south_distance
  -- Net westward movement is 3 miles
  let net_west := west_distance in
  -- The total distance back to start follows Pythagorean theorem:
  sqrt (net_north_south ^ 2 + net_west ^ 2) = 3 :=
begin
  let north_distance := 2,
  let south_distance := 2,
  let west_distance := 3,
  let net_north_south := north_distance - south_distance,
  let net_west := west_distance,
  have h1 : net_north_south = 0,
  { unfold net_north_south, linarith, },
  rw [h1, pow_zero, zero_add, pow_two],
  norm_num,
  exact rfl,
end

end jake_final_distance_l456_456096


namespace volume_of_rotated_segment_l456_456580

-- Definitions based on given conditions
def chord_length (a : ℝ) := a
def projection_onto_diameter (h : ℝ) := h

-- Theorem statement
theorem volume_of_rotated_segment (a h : ℝ) : 
  let V := (1 / 6) * π * a^2 * h in
  ∃ V : ℝ, V = (1 / 6) * π * a^2 * h :=
by
  sorry

end volume_of_rotated_segment_l456_456580


namespace number_of_rotations_l456_456974

def calculate_rotations (R : ℝ) (H : ℝ) : ℝ :=
  let L := Real.sqrt (H ^ 2 + R ^ 2)
  (2 * π * L) / (2 * π * R)

theorem number_of_rotations (R : ℝ) (H : ℝ) (hH : H = 3 * R * Real.sqrt 7) :
  calculate_rotations R H = 8 :=
by
  sorry

end number_of_rotations_l456_456974


namespace trigonometric_identity_l456_456217

theorem trigonometric_identity 
  (ϕ α : ℝ) :
  (sin ϕ)^2 - (cos (α - ϕ))^2 + 2 * (cos α) * (cos ϕ) * (cos (α - ϕ)) = (cos α)^2 :=
by
  sorry

end trigonometric_identity_l456_456217


namespace sandy_total_spent_l456_456492

-- Price definitions
def price_dress : ℝ := 29.99
def price_shorts : ℝ := 18.95
def price_two_shirts : ℝ := 22.14
def price_jacket : ℝ := 45.93

-- Discount definitions
def dress_discount_rate : ℝ := 0.15
def jacket_discount : ℝ := 5.00

-- Sales tax definition
def sales_tax_rate : ℝ := 0.07

-- Proof statement
theorem sandy_total_spent 
  (price_dress price_shorts price_two_shirts price_jacket : ℝ)
  (dress_discount_rate sales_tax_rate : ℝ)
  (jacket_discount : ℝ)
  (h_dress : price_dress = 29.99)
  (h_shorts : price_shorts = 18.95)
  (h_shirts : price_two_shirts = 22.14)
  (h_jacket : price_jacket = 45.93)
  (h_dress_discount_rate : dress_discount_rate = 0.15)
  (h_jacket_discount : jacket_discount = 5.00)
  (h_sales_tax_rate : sales_tax_rate = 0.07) :
  let dress_after_discount := price_dress * (1 - dress_discount_rate) in
  let jacket_after_discount := price_jacket - jacket_discount in
  let total_after_discounts := dress_after_discount + price_shorts + price_two_shirts + jacket_after_discount in
  let total_sales_tax := total_after_discounts * sales_tax_rate in
  let final_total := total_after_discounts + total_sales_tax in
  final_total = 115.04 :=
by {
  -- Proof goes here
  sorry
}

end sandy_total_spent_l456_456492


namespace annual_growth_rate_l456_456499

theorem annual_growth_rate (P : ℝ) (t : ℤ) (hP : P > 0) (ht1 : t ≥ 1) (ht2 : t ≤ 5) :
  (sqrt 1.20)^2 = 1.20 → ((1.20 - 1) * 100) = 20 :=
by
  sorry

end annual_growth_rate_l456_456499


namespace max_possible_value_l456_456031

theorem max_possible_value :
  ∃ (a b c d : ℕ), a ∈ {0, 1, 2, 3} ∧ b ∈ {0, 1, 2, 3} ∧ c ∈ {0, 1, 2, 3} ∧ d ∈ {0, 1, 2, 3} ∧
  (c * (a^b + 1) - d = 30) := by
  sorry

end max_possible_value_l456_456031


namespace cube_volume_l456_456254

theorem cube_volume (a : ℕ) (h1 : 9 * 12 * 3 = 324) (h2 : 108 * a^3 = 324) : a^3 = 27 :=
by {
  sorry
}

end cube_volume_l456_456254


namespace find_OH_squared_l456_456446

variables (A B C : ℝ) (a b c R OH : ℝ)

-- Conditions
def circumcenter (O : ℝ) := true  -- Placeholder, as the actual definition relies on geometric properties
def orthocenter (H : ℝ) := true   -- Placeholder, as the actual definition relies on geometric properties

axiom eqR : R = 5
axiom sumSquares : a^2 + b^2 + c^2 = 50

-- Problem statement
theorem find_OH_squared : OH^2 = 175 :=
by
  sorry

end find_OH_squared_l456_456446


namespace smallest_c_for_inverse_l456_456468

def g (x : ℝ) : ℝ := -3 * (x - 1)^2 + 4

theorem smallest_c_for_inverse :
  ∃ c : ℝ, (∀ x y : ℝ, c ≤ x → c ≤ y → g x = g y → x = y) ∧ (∀ d : ℝ, (∀ x y : ℝ, d ≤ x → d ≤ y → g x = g y → x = y) → c ≤ d) :=
sorry

end smallest_c_for_inverse_l456_456468


namespace unique_perpendicular_line_through_point_l456_456016

variables (a b : ℝ → ℝ) (P : ℝ)

def are_skew_lines (a b : ℝ → ℝ) : Prop :=
  ¬∃ (t₁ t₂ : ℝ), a t₁ = b t₂

def is_point_not_on_lines (P : ℝ) (a b : ℝ → ℝ) : Prop :=
  ∀ (t : ℝ), P ≠ a t ∧ P ≠ b t

theorem unique_perpendicular_line_through_point (ha : are_skew_lines a b) (hp : is_point_not_on_lines P a b) :
  ∃! (L : ℝ → ℝ), (∀ (t : ℝ), L t ≠ P) ∧ (∀ (L' : ℝ → ℝ), (∀ (t : ℝ), L' t ≠ P) → L' = L) := sorry

end unique_perpendicular_line_through_point_l456_456016


namespace shadow_length_correct_l456_456154

theorem shadow_length_correct :
  let light_source := (0, 16)
  let disc_center := (6, 10)
  let radius := 2
  let m := 4
  let n := 17
  let length_form := m * Real.sqrt n
  length_form = 4 * Real.sqrt 17 :=
by
  sorry

end shadow_length_correct_l456_456154


namespace bruno_pens_l456_456625

-- Define Bruno's purchase of pens
def one_dozen : Nat := 12
def half_dozen : Nat := one_dozen / 2
def two_and_half_dozens : Nat := 2 * one_dozen + half_dozen

-- State the theorem to be proved
theorem bruno_pens : two_and_half_dozens = 30 :=
by sorry

end bruno_pens_l456_456625


namespace complex_number_sum_zero_l456_456373

theorem complex_number_sum_zero (a b : ℝ) (i : ℂ) (h : a + b * i = 1 - i) : a + b = 0 := 
by sorry

end complex_number_sum_zero_l456_456373


namespace integers_correct_fractions_correct_negative_numbers_correct_non_negative_integers_correct_l456_456722

-- Given list of numbers
def given_numbers : List Rational :=
  [1/3, -22/7, 0, -1, 3.14, 2, -3, -6, 0.3, 23/100]

-- Questions:
-- 1. Identify Integers
def integers (l : List Rational) : List Rational :=
  [0, -1, 2, -3, -6]

theorem integers_correct :
  integers given_numbers = [0, -1, 2, -3, -6] :=
by sorry

-- 2. Identify Fractions
def fractions (l : List Rational) : List Rational :=
  [1/3, -22/7, 3.14, 0.3, 23/100]

theorem fractions_correct :
  fractions given_numbers = [1/3, -22/7, 3.14, 0.3, 23/100] :=
by sorry

-- 3. Identify Negative numbers
def negative_numbers (l : List Rational) : List Rational :=
  [-22/7, -1, -3, -6]

theorem negative_numbers_correct :
  negative_numbers given_numbers = [-22/7, -1, -3, -6] :=
by sorry

-- 4. Identify Non-negative integers
def non_negative_integers (l : List Rational) : List Rational :=
  [0, 2]

theorem non_negative_integers_correct :
  non_negative_integers given_numbers = [0, 2] :=
by sorry

end integers_correct_fractions_correct_negative_numbers_correct_non_negative_integers_correct_l456_456722


namespace min_value_of_sum_of_squares_l456_456336

theorem min_value_of_sum_of_squares (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 ≥ 1 / 3 :=
by
  sorry

end min_value_of_sum_of_squares_l456_456336


namespace max_ratio_OA_OB_l456_456079

noncomputable def line_l_polar_eq (ρ θ : ℝ) : Prop := ρ * cos θ = -1

noncomputable def curve_C_polar_eq (ρ θ : ℝ) : Prop := ρ * sin θ ^ 2 = 4 * cos θ

theorem max_ratio_OA_OB 
  (A B : ℝ × ℝ)
  (ρ1 ρ2 θ : ℝ)
  (hA_curve : curve_C_polar_eq ρ1 θ)
  (hB_line : line_l_polar_eq ρ2 (θ + π/4))
  (h_angle : π/4 < θ ∧ θ < π/2)
  (h_pos : 0 < 1 / tan θ ∧ 1 / tan θ < 1)
  : ∃ θ, tan θ = 2 ∧ (ρ1 / ρ2) = sqrt 2 / 2 :=
by
  sorry

end max_ratio_OA_OB_l456_456079


namespace sum_prime_factors_1170_l456_456924

theorem sum_prime_factors_1170 : 
  let smallest_prime_factor := 2
  let largest_prime_factor := 13
  (smallest_prime_factor + largest_prime_factor) = 15 :=
by
  sorry

end sum_prime_factors_1170_l456_456924


namespace count_perpendicular_pairs_l456_456298

open_locale big_operators

def is_perpendicular (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0

def pair1 : ℝ × ℝ × ℝ := (3, 4, 0)
def pair1_b : ℝ × ℝ × ℝ := (0, 0, 5)

def pair2 : ℝ × ℝ × ℝ := (3, 1, 3)
def pair2_b : ℝ × ℝ × ℝ := (1, 0, -1)

def pair3 : ℝ × ℝ × ℝ := (-2, 1, 3)
def pair3_b : ℝ × ℝ × ℝ := (6, -5, 7)

def pair4 : ℝ × ℝ × ℝ := (6, 0, 12)
def pair4_b : ℝ × ℝ × ℝ := (6, -5, 7)

theorem count_perpendicular_pairs :
  (is_perpendicular pair1 pair1_b) ∨ 
  (is_perpendicular pair2 pair2_b) ∨ 
  (is_perpendicular pair3 pair3_b) ∨ 
  (is_perpendicular pair4 pair4_b) ↔ 2 :=
sorry

end count_perpendicular_pairs_l456_456298


namespace f_is_endomorphism_image_f_kernel_f_f_is_not_injective_f_is_not_surjective_f_is_not_bijective_l456_456291

namespace MathProof

-- Define the function f
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, 2 * p.1 + 2 * p.2)

-- Statement showing f is an endomorphism
theorem f_is_endomorphism : 
  ∀ (p q : ℝ × ℝ) (a b : ℝ), f (a • p + b • q) = a • f p + b • f q :=
sorry

-- Determine the image of f
theorem image_f : 
  ∀ v : ℝ × ℝ, ∃ a : ℝ, v = (a, 2 * a) ↔ v ∈ set.range f :=
sorry

-- Determine the kernel of f
theorem kernel_f : 
  ∀ u : ℝ × ℝ, f u = (0, 0) ↔ ∃ x : ℝ, u = (x, -x) :=
sorry

-- Prove that f is neither injective, surjective, nor bijective
theorem f_is_not_injective : 
  ¬(∀ u v : ℝ × ℝ, f u = f v → u = v) :=
sorry

theorem f_is_not_surjective : 
  ¬(∀ v : ℝ × ℝ, ∃ u : ℝ × ℝ, f u = v) :=
sorry

theorem f_is_not_bijective : 
  ¬(bijective f) :=
sorry

end MathProof

end f_is_endomorphism_image_f_kernel_f_f_is_not_injective_f_is_not_surjective_f_is_not_bijective_l456_456291


namespace part_I_part_II_l456_456032

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 4 * cos (ω * x) * sin (ω * x + π / 4)

theorem part_I (ω : ℝ) (hω : ω > 0) (hperiod : ∀ x, f(ω, x) = f(ω, x + π)) : ω = 1 :=
sorry

theorem part_II : 
  let fx := 2 * sin (2 * x + π / 4) + sqrt 2 in
  ∀ k : ℤ, monotone_on fx (set.Icc (-3 * π / 8 + k * π) (π / 8 + k * π)) :=
sorry

end part_I_part_II_l456_456032


namespace magnitude_of_complex_number_l456_456649

def complex_number : ℂ := (2/3 : ℚ) - (4/5 : ℚ) * complex.I

theorem magnitude_of_complex_number :
  complex.abs complex_number = real.sqrt (244) / 15 :=
by
  sorry

end magnitude_of_complex_number_l456_456649


namespace max_angle_APB_l456_456116

/-- Let P be the intersection point of the directrix l of an ellipse and its axis of symmetry,
and F be the corresponding focus. AB is a chord passing through F. The maximum value
of ∠APB is 2 * atan(e), where e is the eccentricity. -/
theorem max_angle_APB (a b : ℝ) (e : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) (h4 : e = Real.sqrt (a^2 - b^2) / a)
  (F P A B : ℝ × ℝ)
  (hF : F = (Real.sqrt (a^2 - b^2), 0))
  (hP : P = (a^2 / (a * e), 0))
  (hA : A = (Real.sqrt (a^2 - b^2), a^2 / b))
  (hB : B = (Real.sqrt (a^2 - b^2), -a^2 / b)) :
  ∠APB = 2 * Real.arctan e :=
sorry

end max_angle_APB_l456_456116


namespace dvds_left_l456_456295

-- Define the initial conditions
def owned_dvds : Nat := 13
def sold_dvds : Nat := 6

-- Define the goal
theorem dvds_left (owned_dvds : Nat) (sold_dvds : Nat) : owned_dvds - sold_dvds = 7 :=
by
  sorry

end dvds_left_l456_456295


namespace part1_part2_l456_456452

variables {a b c A B C : ℝ}

-- Conditions
def triangle_ABC (a b c A B C : ℝ) :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ A + B + C = π ∧ a*sin(B) = b*cos(B) ∧ A > π / 2

-- Part (1): If C = π / 6, then determine the value of angle A.
theorem part1 (h : triangle_ABC a b c A B C) (hC : C = π / 6) : A = 2 * π / 3 :=
sorry

-- Part (2): Find the range of values for cos A + cos B + cos C.
theorem part2 (h : triangle_ABC a b c A B C) : 1 < cos A + cos B + cos C ∧ cos A + cos B + cos C <= 5 / 4 :=
sorry

end part1_part2_l456_456452


namespace amount_lent_by_A_l456_456255

def interest_earned (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ := 
  principal * rate * time / 100

theorem amount_lent_by_A (P : ℝ) 
  (rate_A_to_B : ℝ := 10)
  (rate_B_to_C : ℝ := 14)
  (gain_B_3_years : ℝ := 420)
  (time : ℝ := 3) 
  (h : interest_earned P rate_B_to_C time - interest_earned P rate_A_to_B time = gain_B_3_years) : P = 3500 :=
sorry

end amount_lent_by_A_l456_456255


namespace max_overall_average_with_extra_credit_l456_456102

def first_test_score : ℝ := 95 / 100
def first_test_weight : ℝ := 25 / 100

def second_test_score : ℝ := 80 / 100
def second_test_weight : ℝ := 30 / 100

def third_test_score : ℝ := 90 / 100
def third_test_weight : ℝ := 25 / 100

def fourth_test_weight : ℝ := 20 / 100
def extra_credit : ℝ := 5 / 100

def current_weighted_grade :=
  first_test_score * first_test_weight +
  second_test_score * second_test_weight +
  third_test_score * third_test_weight

def target_overall_average : ℝ := 93 / 100

theorem max_overall_average_with_extra_credit :
  current_weighted_grade + (1 + extra_credit) * fourth_test_weight = 91.25 / 100 :=
by
  sorry

end max_overall_average_with_extra_credit_l456_456102


namespace slope_divides_L_shaped_region_l456_456428

open Set

/-- A type representing a point in 2D space. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The L-shaped region in the problem. -/
noncomputable def L_shaped_region : Set (Point2D) :=
  {p | (p.x ≥ 0 ∧ p.x ≤ 3 ∧ p.y ≥ 0 ∧ p.y ≤ 3) ∨ (p.x ≥ 3 ∧ p.x ≤ 5 ∧ p.y ≥ 0 ∧ p.y ≤ 1) }

/-- The slope of the line through the origin that divides the area of the L-shaped region exactly in half. -/
theorem slope_divides_L_shaped_region : (∃ m : ℝ, L_shaped_region.area / 2 = L_shaped_region.area / 2) → m = 7 / 9 := sorry

end slope_divides_L_shaped_region_l456_456428


namespace rhombus_inscribed_circle_distance_l456_456187

theorem rhombus_inscribed_circle_distance
  (ABCD : Type)
  (A B C D O P Q F M : ABCD)
  (side : ℝ) (radius : ℝ) (d_AC_lt_d_BD : Prop)
  [rhombus : ∀ x y z w : ABCD, x ≠ y → x ≠ z → x ≠ w → y ≠ z → y ≠ w → z ≠ w →
   (side_length xy = 5) ∧ (inscribed_circle_radius = 2.4) ∧ (diagonal_shorter xy < diagonal_shorter wz) :=
by
  sorry

end rhombus_inscribed_circle_distance_l456_456187


namespace no_real_solution_sum_l456_456665

theorem no_real_solution_sum :
  ∀ x : ℝ, 
    (8 * x / (x^2 - 4) = 3 * x / (x - 2) - 4 / (x + 2)) →
    false :=
begin
  sorry,
end

end no_real_solution_sum_l456_456665


namespace largest_among_three_l456_456644

noncomputable def largest_number : ℝ :=
  let log_base_half_3 := Real.log 3 / Real.log (1/2)
  let one_third_to_point_two := (1/3 : ℝ) ^ 0.2
  let two_to_one_third := (2 : ℝ) ^ (1/3)
  two_to_one_third

theorem largest_among_three :
  ∀ (x y z : ℝ),
  x = Real.log 3 / Real.log (1/2) →
  y = (1/3 : ℝ) ^ 0.2 →
  z = (2 : ℝ) ^ (1/3) →
  largest_number = max (max x y) z :=
by
  intros x y z hx hy hz
  unfold largest_number
  rw [hx, hy, hz]
  sorry 

end largest_among_three_l456_456644


namespace square_area_from_wire_bent_as_circle_l456_456937

theorem square_area_from_wire_bent_as_circle 
  (radius : ℝ) 
  (h_radius : radius = 56)
  (π_ineq : π > 3.1415) : 
  ∃ (A : ℝ), A = 784 * π^2 := 
by 
  sorry

end square_area_from_wire_bent_as_circle_l456_456937


namespace meal_combinations_l456_456560

theorem meal_combinations (n : ℕ) (h : n = 12) : ∃ m : ℕ, m = 132 :=
by
  -- Initialize the variables for dishes chosen by Yann and Camille
  let yann_choices := n
  let camille_choices := n - 1
  
  -- Calculate the total number of combinations
  let total_combinations := yann_choices * camille_choices
  
  -- Assert the number of combinations is equal to 132
  use total_combinations
  exact sorry

end meal_combinations_l456_456560


namespace kamal_age_problem_l456_456572

theorem kamal_age_problem (K S : ℕ) 
  (h1 : K - 8 = 4 * (S - 8)) 
  (h2 : K + 8 = 2 * (S + 8)) : 
  K = 40 := 
by sorry

end kamal_age_problem_l456_456572


namespace incircle_tangent_points_l456_456347

theorem incircle_tangent_points {A B C D P S Q R : Point} (h1 : Parallelogram A B C D)
  (h2 : Circle ∈ TangentToSide (triangle A B C) AC WithTangentPoints (extend BA P) (extend BC S))
  (h3 : Segment PS Intersects AD At Q)
  (h4 : Segment PS Intersects DC At R) :
  Incircle (triangle C D A) IsTangentToSides AD DC AtPoints Q R :=
by sorry

end incircle_tangent_points_l456_456347


namespace damage_ratio_proof_l456_456122

variable (H g M τ : ℝ)
variable (k n : ℝ)
variable (h : ℝ := H / n)
variable (VI : ℝ := sqrt (2 * g * H))
variable (V1 : ℝ := sqrt (2 * g * h))
variable (V1' : ℝ := (1 / k) * sqrt (2 * g * h))
variable (VII : ℝ := sqrt ((1 / k^2) * 2 * g * h + 2 * g * (H - h)))
variable (II : ℝ := M * τ * ((V1 - V1') + VII))
variable (I1 : ℝ := M * VI * τ)
variable (damage_ratio : ℝ := II / I1)

theorem damage_ratio_proof : 
  (1 / k - 1) / (sqrt n * k) + 
  sqrt ((n - 1) * k^2 + 1) / (sqrt n * k^2) = 
  5 / 4 :=
sorry

end damage_ratio_proof_l456_456122


namespace problem1_problem2a_problem2b_l456_456724

-- Definitions
def f (x : ℝ) (b c : ℝ) := x^2 + b * x + c
def F (x : ℝ) (b c : ℝ) := (f x b c) / Real.exp x

-- Problem 1
theorem problem1 (b c : ℝ) (h_tangent : (F 0 b c) = 0 ∧ (deriv (λ x => F x b c) 0) = 1) :
    b = 1 ∧ c = 0 := sorry

-- Problem 2
theorem problem2a (b c : ℝ) (h_monotonic : ∀ x : ℝ, deriv (λ x => F x b c) x ≤ 0) :
    ∀ x ≥ 0, f x b c ≤ (x + c)^2 := sorry

theorem problem2b (b c : ℝ) (h_monotonic : ∀ x : ℝ, deriv (λ x => F x b c) x ≤ 0) :
    (∀ (b c : ℝ), f(c) - M * c^2 ≤ f(b) - M * b^2)
    ↔ M ∈ Set.Ici (3/2 : ℝ) := sorry

end problem1_problem2a_problem2b_l456_456724


namespace cosine_BHD_is_zero_l456_456074

noncomputable def is_rectangular_solid : Prop := 
  -- Definitions for the vertices of the solid and angles
  ∃ (CD HG DH HB : ℝ), 
  CD = 2 ∧
  HG = 2 ∧
  ∠DHG = 30 ∧
  ∠FHB = 45
  
-- Given the above setup, we aim to prove the following theorem:
theorem cosine_BHD_is_zero : 
  is_rectangular_solid → 
  ∃ (BH BD : ℝ), 
  BH = 2 ∧
  BD = 2 * sqrt 2 ∧
  ∠BHD = 90 → 
  real.cos (angleBHD) = 0 :=
begin
  sorry
end

end cosine_BHD_is_zero_l456_456074


namespace problem1_l456_456949

theorem problem1 (A B C : Prop) : (A ∨ (B ∧ C)) ↔ ((A ∨ B) ∧ (A ∨ C)) :=
sorry 

end problem1_l456_456949


namespace similar_triangles_area_ratio_l456_456716

theorem similar_triangles_area_ratio
  (ABC A1B1C1 : Type)
  [Triangle ABC] [Triangle A1B1C1]
  (similarity_ratio_1_to_3 : similarity_ratio ABC A1B1C1 = 1 / 3) :
  area_ratio ABC A1B1C1 = 1 / 9 :=
sorry

end similar_triangles_area_ratio_l456_456716


namespace largest_diff_between_primes_summing_to_hundred_l456_456915

/-- 
Given the even number 100, the largest possible difference between two different prime numbers that sum to 100 is 94.
-/
theorem largest_diff_between_primes_summing_to_hundred :
  ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧ p + q = 100 ∧ p ≠ q ∧ |p - q| = 94 :=
sorry

end largest_diff_between_primes_summing_to_hundred_l456_456915


namespace boys_belonging_to_other_communities_l456_456076

theorem boys_belonging_to_other_communities
  (n : ℕ)
  (p_M p_H p_S : ℚ)
  (h_n : n = 850)
  (h_pM : p_M = 44 / 100)
  (h_pH : p_H = 14 / 100)
  (h_pS : p_S = 10 / 100) :
  n * (1 - (p_M + p_H + p_S)) = 272 :=
by
  rw [h_n, h_pM, h_pH, h_pS]
  norm_num
  sorry

end boys_belonging_to_other_communities_l456_456076


namespace triangle_sum_equality_l456_456230

variables {A B C G P Q R D E F : Type}
variables [point A] [point B] [point C] [point G] [point P] [point Q] [point R] [point D] [point E] [point F]

structure Triangle (A B C : Type) :=
(centroid : Type)
(incircle_touch_bc_at : P)
(incircle_touch_ca_at : Q)
(incircle_touch_ab_at : R)

structure Collinear (X Y Z : Type) := 
(collinearity : Prop)

noncomputable def point_on_ray (G : Type) (P : Type) : Type := sorry

theorem triangle_sum_equality 
  (ABC : Triangle A B C)
  (G_centroid : G = ABC.centroid)
  (P_touch : P = ABC.incircle_touch_bc_at)
  (Q_touch : Q = ABC.incircle_touch_ca_at)
  (R_touch : R = ABC.incircle_touch_ab_at)
  (D_on_GP : point_on_ray G P)
  (E_on_GQ : point_on_ray G Q)
  (F_on_GR : point_on_ray G R)
  (F_A_E_collinear : Collinear F A E)
  (E_C_D_collinear : Collinear E C D)
  (D_B_F_collinear : Collinear D B F) : 
  AF + BD + CE = AE + BF + CD := 
sorry

end triangle_sum_equality_l456_456230


namespace determine_q_l456_456301

-- Define the polynomial p(x) and its square
def p (x : ℝ) : ℝ := x^2 + x + 1
def p_squared (x : ℝ) : ℝ := (x^2 + x + 1)^2

-- Define the identity condition
def identity_condition (x : ℝ) (q : ℝ → ℝ) : Prop := 
  p_squared x - 2 * p x * q x + (q x)^2 - 4 * p x + 3 * q x + 3 = 0

-- Ellaboration on the required solution
def correct_q (q : ℝ → ℝ) : Prop :=
  (∀ x, q x = x^2 + 2 * x) ∨ (∀ x, q x = x^2 - 1)

-- The theorem statement
theorem determine_q :
  ∀ q : ℝ → ℝ, (∀ x : ℝ, identity_condition x q) → correct_q q :=
by
  intros
  sorry

end determine_q_l456_456301


namespace _l456_456470

noncomputable def circle_theorem :=
  let R₁ : ℝ := 123
  let R₂ : ℝ := 61
  let t : ℝ := real.sqrt (11408 / 5)
  let chord_length : ℝ := 6 * t
  chord_length = 42

end _l456_456470


namespace steven_apples_set_aside_l456_456496

-- Define the conditions
def apples := 6 -- seeds per apple
def pears := 2 -- seeds per pear
def grapes := 3 -- seeds per grape
def total_needed_seeds := 60
def seeds_short := 3
def pears_set_aside := 3
def grapes_set_aside := 9

-- Calculate the number of apples Steven set aside
theorem steven_apples_set_aside :
  let seeds_from_pears := pears_set_aside * pears,
      seeds_from_grapes := grapes_set_aside * grapes,
      seeds_collected := total_needed_seeds - seeds_short,
      seeds_needed_from_apples := seeds_collected - (seeds_from_pears + seeds_from_grapes),
      number_of_apples := seeds_needed_from_apples / apples,
      answer := 4 in
  number_of_apples = answer :=
by 
  let seeds_from_pears := 3 * 2,
  let seeds_from_grapes := 9 * 3,
  let seeds_collected := 60 - 3,
  let seeds_needed_from_apples := seeds_collected - (seeds_from_pears + seeds_from_grapes),
  let number_of_apples := seeds_needed_from_apples / 6,
  let answer := 4 in
  show number_of_apples = answer by
  sorry

end steven_apples_set_aside_l456_456496


namespace solution_set_log_inequality_l456_456360

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(Real.log (x^2 - 2*x + 3))

theorem solution_set_log_inequality
  (a : ℝ)
  (h1 : 0 < a) 
  (h2 : a ≠ 1) 
  (h3 : ∃ M, ∀ x, f a x ≤ M) :
  { x : ℝ | 2 < x ∧ x < 3 } = {x : ℝ | Real.log a (x^2 - 5*x + 7) > 0} :=
by 
  sorry

end solution_set_log_inequality_l456_456360


namespace analytical_expression_of_f_f_is_increasing_solve_inequality_l456_456582

-- Define the function f(x) and its derivative f'
def f (x : ℝ) : ℝ := x / (1 + x^2)
def f' (x : ℝ) : ℝ := (1 - x^2) / (1 + x^2)^2

-- Theorem 1: Proving the analytical expression of f(x)
theorem analytical_expression_of_f : ∀ x, f(x) = x / (1 + x^2) :=
by sorry

-- Theorem 2: Proving that f is an increasing function in the interval (-1,1)
theorem f_is_increasing : ∀ x₁ x₂, -1 < x₁ → x₁ < x₂ → x₂ < 1 → f(x₁) < f(x₂) :=
by sorry

-- Theorem 3: Solving the inequality f(t-1) + f(t) < 0
theorem solve_inequality (t : ℝ) : (f (t-1) + f t < 0) ↔ (0 < t ∧ t < 1/2) :=
by sorry

end analytical_expression_of_f_f_is_increasing_solve_inequality_l456_456582


namespace find_dimensions_l456_456421

-- Define the conditions
def perimeter (x y : ℕ) : Prop := (2 * (x + y) = 3996)
def divisible_parts (x y k : ℕ) : Prop := (x * y = 1998 * k) ∧ ∃ (k : ℕ), (k * 1998 = x * y) ∧ k ≠ 0

-- State the theorem
theorem find_dimensions (x y : ℕ) (k : ℕ) : perimeter x y ∧ divisible_parts x y k → (x = 1332 ∧ y = 666) ∨ (x = 666 ∧ y = 1332) :=
by
  -- This is where the proof would go.
  sorry

end find_dimensions_l456_456421


namespace probability_of_passing_l456_456278

theorem probability_of_passing (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
  sorry

end probability_of_passing_l456_456278


namespace cerulean_survey_l456_456236

theorem cerulean_survey :
  let total_people := 120
  let kind_of_blue := 80
  let kind_and_green := 35
  let neither := 20
  total_people = kind_of_blue + (total_people - kind_of_blue - neither)
  → (kind_and_green + (total_people - kind_of_blue - kind_and_green - neither) + neither) = total_people
  → 55 = (kind_and_green + (total_people - kind_of_blue - kind_and_green - neither)) :=
by
  sorry

end cerulean_survey_l456_456236


namespace bisectors_same_plane_l456_456148

-- Consider the points and vectors in a 3D space
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (O A B C B1 : V)

-- Conditions
variables (h1 : A - O = B - O)
variables (h2 : B - O = C - O)
variables (h3 : B1 = -B)

noncomputable def M : V := 0.5 • (B + A)
noncomputable def P : V := 0.5 • (A + C)
noncomputable def K : V := 0.5 • (C - B)

-- Theorem to prove the bisectors lie in the same plane
theorem bisectors_same_plane (h1 : A - O = B - O)
                              (h2 : B - O = C - O)
                              (h3 : B1 = -B) :
  ∃ (plane : Submodule ℝ V), 
    M A B ∈ plane ∧ P A C ∈ plane ∧ K B C ∈ plane := sorry

end bisectors_same_plane_l456_456148


namespace symmetry_about_y_axis_l456_456517

def f (x : ℝ) : ℝ := Real.cos x + 2 * x^2

theorem symmetry_about_y_axis : ∀ x : ℝ, f (-x) = f x :=
by
  intro x
  sorry

end symmetry_about_y_axis_l456_456517


namespace solve_equation_l456_456869

theorem solve_equation (x : ℝ) : (x + 3)^4 + (x + 1)^4 = 82 → x = 0 ∨ x = -4 :=
by
  sorry

end solve_equation_l456_456869


namespace time_to_travel_nth_mile_l456_456967

theorem time_to_travel_nth_mile (n : ℕ) (h : n ≥ 4) : 
  let t_n := 3 * (n - 3)
  in t_n = 3 * (n - 3) :=
by
  sorry

end time_to_travel_nth_mile_l456_456967


namespace exists_k_sum_bound_l456_456693

theorem exists_k_sum_bound (n : ℕ) (a : Fin n → ℝ) :
  ∃ k : Fin n, ∀ b : Fin n → ℝ, (∀ i j : Fin n, i ≤ j → b i ≥ b j) → (∀ i : Fin n, 0 ≤ b i) → 
  ∥∑ i, b i * a i∥ ≤ ∥∑ i in Finset.range (k + 1), a i∥ :=
begin
  sorry
end

end exists_k_sum_bound_l456_456693


namespace intersecting_lines_at_point_find_b_plus_m_l456_456200

theorem intersecting_lines_at_point_find_b_plus_m :
  ∀ (m b : ℝ),
  (12 = m * 4 + 2) →
  (12 = -2 * 4 + b) →
  (b + m = 22.5) :=
by
  intros m b h1 h2
  sorry

end intersecting_lines_at_point_find_b_plus_m_l456_456200


namespace probability_same_color_l456_456800

theorem probability_same_color :
  ∀ (has_pairs : Nat) (total_shoes : Nat),
  has_pairs = 6 → -- Condition: Kim has 6 pairs of shoes
  total_shoes = 12 → -- Condition: There are 12 shoes in total
  ∃ prob : Rat, -- The result will be a probability (rational number)
  prob = 1 / 11 := -- The probability that Kim selects 2 shoes of the same color is 1/11
by
  intros has_pairs total_shoes h_pairs h_shoes
  use 1 / 11
  sorry

end probability_same_color_l456_456800


namespace triangle_ABC_area_median_AD_length_l456_456431

-- Definitions and conditions
def AB : ℝ := 30
def AC : ℝ := 40
def angle_A : ℝ := 90 -- degrees

-- Areas to prove: area of triangle ABC and length of median AD
def area_ABC (AB AC : ℝ) : ℝ := 1 / 2 * AB * AC
def length_median (AB AC : ℝ) : ℝ := 1 / 2 * Real.sqrt (2 * (AB ^ 2 + AC ^ 2))

-- Theorem statements
theorem triangle_ABC_area : area_ABC AB AC = 600 :=
by sorry

theorem median_AD_length : length_median AB AC ≈ 35.36 :=
by sorry

end triangle_ABC_area_median_AD_length_l456_456431


namespace parabola_equation_unique_max_area_ratio_value_l456_456730

-- Definitions for the given problem:
def parabola (p : ℝ) : set (ℝ × ℝ) := { point | let (x, y) := point in y^2 = 2 * p * x }
def circle : set (ℝ × ℝ) := { point | let (x, y) := point in (x - 4)^2 + y^2 = 12 }
def has_two_common_points (C : set (ℝ × ℝ)) (E : set (ℝ × ℝ)) : Prop :=
  ∃ P Q, P ∈ C ∧ P ∈ E ∧ Q ∈ C ∧ Q ∈ E ∧ P ≠ Q

-- Problem to solve:
theorem parabola_equation_unique (p : ℝ) (hp : 0 < p) (h : has_two_common_points (parabola p) circle) :
  ∃ C : set (ℝ × ℝ), C = { point | let (x, y) := point in y^2 = 4 * x } :=
sorry

-- Definitions for Part (2):
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1), (x2, y2), (x3, y3) := A, B, C in
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

def max_area_ratio (C : set (ℝ × ℝ)) (E : set (ℝ × ℝ)) (O : ℝ × ℝ) : ℝ :=
  let (xO, yO) := O in
  let center := (4, 0) in
  let line m := { point | let (x, y) := point in x = m * y + 4 } in
  let A := (1.0, sqrt 11.0) in -- Point on the circle
  let B := (7.0, -sqrt 11.0) in -- Another point on the circle
  let OA := euclidean_distance O A in
  let OB := euclidean_distance O B in
  let PA := euclidean_distance O (4.0, sqrt 1.0) in -- Point on the parabola
  let PB := euclidean_distance O (4.0, -sqrt 1.0) in -- Another point on the parabola
  (triangle_area O A B) / (triangle_area O (4.0, sqrt 1.0) (4.0, -sqrt 1.0))

-- Problem to solve:
theorem max_area_ratio_value (C : set (ℝ × ℝ)) (E : set (ℝ × ℝ)) :
  ∀ O : ℝ × ℝ, max_area_ratio C E O = 9 / 16 :=
sorry

end parabola_equation_unique_max_area_ratio_value_l456_456730


namespace tan_sum_pi_over_4_sin_cos_fraction_l456_456335

open Real

variable (α : ℝ)

axiom tan_α_eq_2 : tan α = 2

theorem tan_sum_pi_over_4 (α : ℝ) (h : tan α = 2) : tan (α + π / 4) = -3 :=
sorry

theorem sin_cos_fraction (α : ℝ) (h : tan α = 2) : (sin α + cos α) / (sin α - cos α) = 3 :=
sorry

end tan_sum_pi_over_4_sin_cos_fraction_l456_456335


namespace value_range_cosine_l456_456192

theorem value_range_cosine (x : ℝ) (h : cos x ∈ set.Icc (-1 : ℝ) 1) : 
  let y := cos (2 * x) - 8 * cos x in 
  ∃ (a b : ℝ), set.Icc a b = set.Icc (-7 : ℝ) 9 ∧ y ∈ set.Icc a b :=
by
  sorry

end value_range_cosine_l456_456192


namespace multiplication_is_247_l456_456782

theorem multiplication_is_247 (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : a * b = 247) : 
a = 13 ∧ b = 19 :=
by sorry

end multiplication_is_247_l456_456782


namespace cos_2alpha_minus_beta_eq_sqrt2_div_10_beta_eq_pi_div_4_l456_456015

theorem cos_2alpha_minus_beta_eq_sqrt2_div_10 
    (α β : ℝ) 
    (hα_range : 0 < α ∧ α < π) 
    (hβ_range : 0 < β ∧ β < π)
    (hcosα : Real.cos α = (Real.sqrt 5) / 5)
    (hsin_alpha_minus_beta : Real.sin (α - β) = (Real.sqrt 10) / 10) :
    Real.cos (2 * α - β) = (Real.sqrt 2) / 10 := 
sý
theorem beta_eq_pi_div_4 
    (α β : ℝ) 
    (hα_range : 0 < α ∧ α < π) 
    (hβ_range : 0 < β ∧ β < π)
    (hcosα : Real.cos α = (Real.sqrt 5) / 5)
    (hsin_alpha_minus_beta : Real.sin (α - β) = (Real.sqrt 10) / 10) :
    β = π / 4 :=
sorry

end cos_2alpha_minus_beta_eq_sqrt2_div_10_beta_eq_pi_div_4_l456_456015


namespace no_valid_arrangement_of_segments_l456_456788

theorem no_valid_arrangement_of_segments :
  ¬ ∃ segments : list (ℝ × ℝ),  -- Representation for 1000 segments as pairs of reals (endpoints)
    (length segments = 1000) ∧ 
    (∀ seg ∈ segments, 
      ∃ seg' ∈ segments, 
        seg ≠ seg' ∧ 
          (seg.1 > seg'.1 ∧ seg.2 < seg'.2)) :=
by sorry

end no_valid_arrangement_of_segments_l456_456788


namespace sequence_element_2014th_l456_456857

theorem sequence_element_2014th : 
  let A := {n : ℤ // n > 0 ∧ n ≤ 10000}
  let is_perfect_square (n : ℤ) := ∃ m : ℤ, m^2 = n
  let is_perfect_cube (n : ℤ) := ∃ m : ℤ, m^3 = n
  let is_perfect_sixth_power (n : ℤ) := ∃ m : ℤ, m^6 = n
  let B := {n ∈ A | ¬is_perfect_square n ∧ ¬is_perfect_cube n ∧ ¬is_perfect_sixth_power n}
  let L := List.erase_dup (List.sort (List.ofFinset B))
  L[2013].val = 2068 := sorry

end sequence_element_2014th_l456_456857


namespace scout_troop_profit_l456_456275

noncomputable def candy_profit (purchase_bars purchase_rate sell_bars sell_rate donation_fraction : ℕ) : ℕ :=
  let cost_price_per_bar := purchase_rate / purchase_bars
  let total_cost := purchase_bars * cost_price_per_bar
  let effective_cost := total_cost * donation_fraction
  let sell_price_per_bar := sell_rate / sell_bars
  let total_revenue := purchase_bars * sell_price_per_bar
  total_revenue - effective_cost

theorem scout_troop_profit :
  candy_profit 1200 3 4 3 1/2 = 700 := by
  sorry

end scout_troop_profit_l456_456275


namespace find_fraction_l456_456590

def number : ℕ := 16

theorem find_fraction (f : ℚ) : f * number + 5 = 13 → f = 1 / 2 :=
by
  sorry

end find_fraction_l456_456590


namespace problem1_coefficient_of_x_problem2_maximum_coefficient_term_l456_456581

-- Problem 1: Coefficient of x term
theorem problem1_coefficient_of_x (n : ℕ) 
  (A : ℕ := (3 + 1)^n) 
  (B : ℕ := 2^n) 
  (h1 : A + B = 272) 
  : true :=  -- Replacing true with actual condition
by sorry

-- Problem 2: Term with maximum coefficient
theorem problem2_maximum_coefficient_term (n : ℕ)
  (h : 1 + n + (n * (n - 1)) / 2 = 79) 
  : true :=  -- Replacing true with actual condition
by sorry

end problem1_coefficient_of_x_problem2_maximum_coefficient_term_l456_456581


namespace _l456_456562

noncomputable def triangle_relationship (A B C D O O' : Point) (circle1 : Circle) (circle2 : Circle)
  (h1 : ∃ O', circle1 = Circle.mk O' (distance A O')) -- The larger circle centered at O'
  (h2 : ∃ O, circle2 = Circle.mk O (distance A O)) -- The smaller circle centered at O
  (h3 : ∠ A C B = 90°)
  (h4 : circle1.inscribed_triangle A B C)
  (h5 : circle2.inscribed_triangle A B C)
  (h6 : ∃ D, lies_on_extended_line A C D ∧ lies_on_circle circle1 D) : Prop :=
  distance A D = distance A B

lemma triangle_relationship_theorem (A B C D O O' : Point) (circle1 : Circle) (circle2 : Circle)
  (h1 : ∃ O', circle1 = Circle.mk O' (distance A O')) -- The larger circle centered at O'
  (h2 : ∃ O, circle2 = Circle.mk O (distance A O)) -- The smaller circle centered at O
  (h3 : ∠ A C B = 90°)
  (h4 : circle1.inscribed_triangle A B C)
  (h5 : circle2.inscribed_triangle A B C)
  (h6 : ∃ D, lies_on_extended_line A C D ∧ lies_on_circle circle1 D) :
  distance A D = distance A B := 
  sorry

end _l456_456562


namespace problem_sum_value_l456_456177

def letter_value_pattern : List Int := [2, 3, 2, 1, 0, -1, -2, -3, -2, -1]

def char_value (c : Char) : Int :=
  let pos := c.toNat - 'a'.toNat + 1
  letter_value_pattern.get! ((pos - 1) % 10)

def word_value (w : String) : Int :=
  w.data.map char_value |>.sum

theorem problem_sum_value : word_value "problem" = 5 :=
  by sorry

end problem_sum_value_l456_456177


namespace sum_lent_is_approximately_368_l456_456934

-- Define the conditions:
def simpleInterest (P R T I : ℝ) : Prop := I = P * R * T / 100
def interest_deficit (P I : ℝ) : Prop := I = P - 238

theorem sum_lent_is_approximately_368 (P : ℝ) (h1 : simpleInterest P 4 8 (P * 4 * 8 / 100))
                                      (h2 : interest_deficit P (P * 4 * 8 / 100)) :
  P ≈ 368 :=
by sorry

end sum_lent_is_approximately_368_l456_456934


namespace fold_and_pierce_l456_456843

-- Define points on a sheet of paper
structure Point :=
(x : ℝ) (y : ℝ)

-- Define condition where several points are on a line
noncomputable def points_on_line (points : List Point) : Prop :=
  ∃ (a b : ℝ), ∀ (p : Point), p ∈ points → p.y = a * p.x + b

-- Define a triangle by its vertices
structure Triangle :=
(A B C : Point)

-- Define a condition for triangular points
def triangular_points (t : Triangle) : Prop := 
  t.A ≠ t.B ∧ t.B ≠ t.C ∧ t.C ≠ t.A

-- The main theorem statement
theorem fold_and_pierce (points : List Point) (triangle : Triangle) :
  points_on_line points ∨ triangular_points triangle →
  ∃ folded_points, (∀ (p : Point), p ∈ points → p ∈ folded_points) ∨ 
                   (∀ (p ∈ {triangle.A, triangle.B, triangle.C}), p ∈ folded_points) ∧
                   (∃ (hole_point : Point), 
                     (∀ (p : Point), p ∈ folded_points → p = hole_point)) :=
by
  sorry

end fold_and_pierce_l456_456843


namespace value_of_x_l456_456057

theorem value_of_x (x : ℕ) : (1 / 16) * (2 ^ 20) = 4 ^ x → x = 8 := by
  sorry

end value_of_x_l456_456057


namespace find_DL_l456_456436

variables {A B C D L : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace L]
variables (A B C D L : Point)

-- Given distances
variable (AB : ℝ) (BC : ℝ) (AC : ℝ) (DB : ℝ)

-- Given angles sum condition
variable (angle_sum_condition : ∠ ABD + ∠ DBC = π)

-- Actual lengths
def l1 : length A B = 9 := sorry
def l2 : length B C = 6 := sorry
def l3 : length A C = 5 := sorry
def l4 : length D B = 1 := sorry

theorem find_DL 
  (h1 : angle_sum_condition ∠ ABD ∠ DBC)
  (h2 : l1 AB)
  (h3 : l2 BC)
  (h4 : l3 AC)
  (h5 : l4 DB) :
  (length D L) = 7 := sorry

end find_DL_l456_456436


namespace count_of_non_square_prime_units_l456_456715

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_lt_100 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def square_units : List ℕ := [0, 1, 4, 9, 6]

def non_square_prime_units (l : List ℕ) : List ℕ :=
  l.filter (λ n => (n % 10) ∉ square_units)

theorem count_of_non_square_prime_units : non_square_prime_units primes_lt_100 |>.length = 15 := sorry

end count_of_non_square_prime_units_l456_456715


namespace arithmetic_sequence_product_l456_456454

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) 
  (h_inc : ∀ n, b (n + 1) - b n = d)
  (h_pos : d > 0)
  (h_prod : b 5 * b 6 = 21) 
  : b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
sorry

end arithmetic_sequence_product_l456_456454


namespace min_colors_map_min_colors_complete_graph_6_l456_456479

-- Problem (a): Prove the minimum colors needed for a specific map
def countries : Type := {I, II, III, IV, V}
def adjacent : countries → countries → Prop 
| I, II := true
| I, III := true
| II, IV := true
| III, IV := true
| II, V := true
| III, V := true
| _, _ := false

theorem min_colors_map : ∃ n : ℕ, ∀ color : countries → ℕ, 
  (∀ c : countries, color c < n) ∧ 
  (∀ c₁ c₂: countries, adjacent c₁ c₂ → color c₁ ≠ color c₂) 
  → n = 2 :=
sorry

-- Problem (b): Prove at least 4 colors are needed to color a map of a complete graph with 6 vertices
def complete_graph_6 (v : Type) [fintype v] := 
  ∀ c1 c2 : v, c1 ≠ c2 → ∃ edge : (c1 ≠ c2), edge

theorem min_colors_complete_graph_6 : ∃ (n : ℕ), 
  ∀ color : fin 6 → ℕ, 
    (∀ c: fin 6, color c < n) ∧ 
    (∀ c1 c2: fin 6, c1 ≠ c2 → color c1 ≠ color c2) 
  → n ≥ 4 :=
sorry

end min_colors_map_min_colors_complete_graph_6_l456_456479


namespace proof_cos_B_proof_perimeter_l456_456027

variables {A B C : ℝ} 
variables {vBA vCB : ℝ}

noncomputable def cos_B (A : ℝ) (B : ℝ) (C : ℝ) (cosA : ℝ) (dot_product : ℝ) : ℝ := 
  - (cosA * (2 * cosA^2 - 1) - ((1 - cosA^2) * (1 - (2 * cosA^2 - 1)^2)^0.5))

/-- Given conditions in the triangle:
* C = 2A
* cos A = 3/4
* 2 \overrightarrow{BA} \cdot \overrightarrow{CB} = -27

Prove that cos B = 9/16.
-/
theorem proof_cos_B 
  (h1 : C = 2 * A) 
  (h2 : Real.cos A = 3 / 4) 
  (h3 : 2 * vBA * vCB = -27) : 
  cos_B A B C (3 / 4) h3 = 9 / 16 :=
sorry

noncomputable def perimeter (A B C : ℝ) (cosA : ℝ) (dot_product : ℝ) : ℝ := 
  -- Compute BC using dot product and conditions
  let BC := (24 / ((1 - cosA ^ 2) ^ 0.5)) in
  -- Compute AB using sine law and BC
  let AB := 3 / 2 * BC in
  -- Compute AC using cosine law
  let AC := (BC^2 + AB^2 - 2 * BC * AB * (9 / 16)) ^ 0.5 in
  AB + BC + AC

/-- Given conditions in the triangle:
* C = 2A
* cos A = 3/4
* 2 \overrightarrow{BA} \cdot \overrightarrow{CB} = -27

Prove that the perimeter of △ABC is 15.
-/
theorem proof_perimeter
  (h1 : C = 2 * A) 
  (h2 : Real.cos A = 3 / 4) 
  (h3 : 2 * vBA * vCB = -27) : 
  perimeter A B C (3 / 4) h3 = 15 :=
sorry

end proof_cos_B_proof_perimeter_l456_456027


namespace area_of_square_with_diagonal_30_l456_456577

theorem area_of_square_with_diagonal_30 :
  ∀ (d : ℝ), d = 30 → (d * d / 2) = 450 := 
by
  intros d h
  rw [h]
  sorry

end area_of_square_with_diagonal_30_l456_456577


namespace end_digit_of_number_l456_456950

theorem end_digit_of_number (n : ℕ) (h_n : n = 2022) (h_start : ∃ (f : ℕ → ℕ), f 0 = 4 ∧ 
    (∀ i < n - 1, (19 ∣ (10 * f i + f (i + 1))) ∨ (23 ∣ (10 * f i + f (i + 1))))) :
  ∃ (f : ℕ → ℕ), f (n - 1) = 8 :=
by {
  sorry
}

end end_digit_of_number_l456_456950


namespace sqrt_sum_seven_l456_456163

theorem sqrt_sum_seven (y : ℝ) (h : sqrt (64 - y^2) - sqrt (36 - y^2) = 4) :
  sqrt (64 - y^2) + sqrt (36 - y^2) = 7 := by
  sorry

end sqrt_sum_seven_l456_456163


namespace percentage_reduction_is_20_percent_l456_456242

-- Defining the initial and final prices
def initial_price : ℝ := 25
def final_price : ℝ := 16

-- Defining the percentage reduction
def percentage_reduction (x : ℝ) := 1 - x

-- The equation representing the two reductions:
def equation (x : ℝ) := initial_price * (percentage_reduction x) * (percentage_reduction x)

theorem percentage_reduction_is_20_percent :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ equation x = final_price ∧ x = 0.20 :=
by 
  sorry

end percentage_reduction_is_20_percent_l456_456242


namespace product_of_perimeters_correct_l456_456876

noncomputable def area (side_length : ℝ) : ℝ := side_length * side_length

theorem product_of_perimeters_correct (x y : ℝ)
  (h1 : area x + area y = 85)
  (h2 : area x - area y = 45) :
  4 * x * 4 * y = 32 * Real.sqrt 325 :=
by sorry

end product_of_perimeters_correct_l456_456876


namespace triangle_dimensions_of_cut_square_l456_456976

theorem triangle_dimensions_of_cut_square (a : ℝ) (h : a = 10):
  ∃ (b : ℝ), b = 10 * Real.sqrt 2 ∧ 
  (∀ (x y : ℝ), x = a → y = a → (x^2 + y^2 = b^2)) :=
by
  have hypotenuse := 10 * Real.sqrt 2,
  use hypotenuse,
  split,
  { refl },
  { intros x y hx hy,
    rw [hx, hy],
    simp,
    exact (Real.sqrt_sq (by norm_num : 0 ≤ 10)) }

end triangle_dimensions_of_cut_square_l456_456976


namespace cos_18_degree_eq_l456_456998

noncomputable def cos_18_deg : ℝ :=
  let y := (cos (real.pi / 10))           -- 18 degrees in radians
  let x := (cos (2 * real.pi / 10))       -- 36 degrees in radians
  have h1 : x = 2 * y ^ 2 - 1,
  by sorry,  -- Double angle formula
  have h2 : cos (3 * real.pi / 10) = (cos (π / 2 - 2 * real.pi / 10)),
  by sorry,  -- Triple angle formula for sine
  have h3 : cos (3 * real.pi / 10) = 4 * cos (real.pi / 10)^3 - 3 * cos (real.pi / 10),
  by sorry,  -- Triple angle formula for cosine
  have h4 : cos (π / 2 - 2 * real.pi / 10) = sin (2 * real.pi / 10),
  by sorry,  -- Cosine of complementary angle
  show y = (1 + real.sqrt 5) / 4,
  by sorry

theorem cos_18_degree_eq : cos_18_deg = (1 + real.sqrt 5) / 4 :=
by sorry

end cos_18_degree_eq_l456_456998


namespace taehyung_run_distance_l456_456194

theorem taehyung_run_distance (side_length : ℕ) (h : side_length = 40) :
  let perimeter := 4 * side_length in
  perimeter = 160 :=
by
  -- Conditions
  rw h
  -- Calculation of the perimeter
  have : perimeter = 4 * 40 := rfl
  -- Proving the goal
  exact this

end taehyung_run_distance_l456_456194


namespace hexagon_perimeter_l456_456290

theorem hexagon_perimeter
  (A B C D E F : Type)  -- vertices of the hexagon
  (angle_A : ℝ) (angle_C : ℝ) (angle_E : ℝ)  -- nonadjacent angles
  (angle_B : ℝ) (angle_D : ℝ) (angle_F : ℝ)  -- adjacent angles
  (area_hexagon : ℝ)
  (side_length : ℝ)
  (h1 : angle_A = 120) (h2 : angle_C = 120) (h3 : angle_E = 120)
  (h4 : angle_B = 60) (h5 : angle_D = 60) (h6 : angle_F = 60)
  (h7 : area_hexagon = 24)
  (h8 : ∃ s, ∀ (u v : Type), side_length = s) :
  6 * side_length = 24 / (Real.sqrt 3 ^ (1/4)) :=
by
  sorry

end hexagon_perimeter_l456_456290


namespace approximation_accuracy_of_pi_is_hundredths_l456_456874

theorem approximation_accuracy_of_pi_is_hundredths :
  abs (π - 3.14) < 0.005 :=
sorry

end approximation_accuracy_of_pi_is_hundredths_l456_456874


namespace painting_equation_l456_456645

-- Define rates of Doug and Dave
def doug_rate := 1 / 6
def dave_rate := 1 / 8
def drying_time := 2

-- Define the total rate
def combined_rate := doug_rate + dave_rate

-- Define the total time t
variable {t : ℝ}

-- The theorem states that the following equation holds
theorem painting_equation (ht : (combined_rate) * (t - drying_time) = 1) : 
    (doug_rate + dave_rate) * (t - drying_time) = 1 := 
begin
  exact ht,
end

end painting_equation_l456_456645


namespace find_y_l456_456317

theorem find_y (y : ℝ) (hy : log y 81 = 4 / 2) : y = 9 := 
by {
  sorry
}

end find_y_l456_456317


namespace second_intersection_on_AC_l456_456444

variable (A B C D E F : Type)
variables [EuclideanGeometry A B C D E F]

-- Given conditions
axiom trapezoid {A B C D : Trieste} (h : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A):
  parallel (line A B) (line C D)

axiom point_on_segment_AC {E : Trieste} (h : E ∈ segment (A, C))

axiom parallel_BE_through_C_intersects_BD_at_F {F : Trieste} (h : parallel (line B E) (line C F ∧ (intersection_point (line C F) (line B D) = F))

-- Proof goal
theorem second_intersection_on_AC :
  ∃ G : EuclideanGeometry, G ∈ circumcircle (triangle A B F) ∧ G ∈ circumcircle (triangle B E D) ∧ G ∈ line (A, C) :=
sorry

end second_intersection_on_AC_l456_456444


namespace find_m_l456_456389

-- Define the functions f and g
def f (x m : ℝ) := x^2 - 2 * x + m
def g (x m : ℝ) := x^2 - 3 * x + 5 * m

-- The condition to be proved
theorem find_m (m : ℝ) : 3 * f 4 m = g 4 m → m = 10 :=
by
  sorry

end find_m_l456_456389


namespace average_of_numbers_l456_456886

theorem average_of_numbers (N : ℝ) (h1 : 13 < N) (h2 : N < 21) : 
  (∃ (x : ℝ), x = 12 ∧ 11 < x ∧ x < 13 + 2/3) :=
by
  use (20 + N) / 3
  split
  { sorry }
  split
  { sorry }
  { sorry }

end average_of_numbers_l456_456886


namespace harry_worked_32_hours_last_week_l456_456938

variables (x : ℝ) (H : ℕ) -- define variables

-- Define the conditions

def harry_first_12_hours := 12 * x
def harry_additional_hours := (H - 12) * (1.5 * x)
def harry_total_earnings := if H > 12 then harry_first_12_hours + harry_additional_hours else H * x

def james_total_hours := 41
def james_first_40_hours := 40 * x
def james_additional_hour := (james_total_hours - 40) * (2 * x)
def james_total_earnings := james_first_40_hours + james_additional_hour

-- Define the equality constraint based on the condition that both earned the same amount last week
def harry_james_same_earnings := harry_total_earnings x H = james_total_earnings x

-- State the theorem to be proved
theorem harry_worked_32_hours_last_week (hx : 0 < x) : harry_james_same_earnings x H → H = 32 :=
sorry

end harry_worked_32_hours_last_week_l456_456938


namespace incircle_tangent_points_l456_456351

theorem incircle_tangent_points {A B C D P S Q R : Point} 
  (h_parallelogram : parallelogram A B C D) 
  (h_tangent_ac : tangent (circle P Q R) A C) 
  (h_tangent_ba_ext : tangent (circle P Q R) (extension B A P)) 
  (h_tangent_bc_ext : tangent (circle P Q R) (extension B C S)) 
  (h_ps_intersect_da : segment_intersect P S D A Q)
  (h_ps_intersect_dc : segment_intersect P S D C R) :
  tangent (incircle D C A) D A Q ∧ tangent (incircle D C A) D C R := sorry

end incircle_tangent_points_l456_456351


namespace domain_of_f_l456_456678

def f (x : ℝ) : ℝ := 1 / Real.logb 0.5 (2 * x + 1)

theorem domain_of_f : {x : ℝ | (2 * x + 1 > 0) ∧ (2 * x + 1 ≠ 1)} = {x : ℝ | -1 / 2 < x ∧ x ≠ 0} ∪ {x : ℝ | 0 < x} :=
by
  sorry

end domain_of_f_l456_456678


namespace rosie_pies_l456_456157

theorem rosie_pies (total_apples : ℕ) (apples_per_pie : ℕ) (h : total_apples = 32) (h_apples_per_pie : apples_per_pie = 5) : total_apples / apples_per_pie = 6 := by
  rw [h, h_apples_per_pie]
  norm_num
  sorry

end rosie_pies_l456_456157


namespace eccentricity_range_of_hyperbola_l456_456733

theorem eccentricity_range_of_hyperbola 
  (m : ℝ) (h : m ∈ set.Icc (-2 : ℝ) (-1 : ℝ)) :
  ∀ e, (e = sqrt (4 - m) / 2) ↔ e ∈ set.Icc (sqrt 5 / 2) (sqrt 6 / 2) :=
by sorry

end eccentricity_range_of_hyperbola_l456_456733


namespace larry_wins_probability_l456_456103

-- Define the probability of hitting the bottle (victory) as 1/3
def prob_hit := 1 / 3

-- Define the probability of missing the bottle (failure) as 2/3
def prob_miss := 2 / 3

-- Define the geometric series sum for the probability that Larry wins
def prob_larry_wins : ℚ := 
  let a := prob_hit
  let r := (prob_miss ^ 3)
  a / (1 - r)

theorem larry_wins_probability :
  prob_larry_wins = 9 / 19 :=
by
  sorry

end larry_wins_probability_l456_456103


namespace find_f_12_13_l456_456441

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_12_13 (h_mono : ∀ x y, 0 ≤ x → x ≤ y → y ≤ 1 → f(x) ≤ f(y))
  (h1 : ∀ x, 0 ≤ x → x ≤ 1 → f(x / 3) = f(x) / 2)
  (h2 : ∀ x, 0 ≤ x → x ≤ 0.1 → f(10 * x) = 2018 - f(x))
  (h3 : f(1) = 2018) : 
  f(12 / 13) = 2018 :=
sorry

end find_f_12_13_l456_456441


namespace apex_angle_of_identical_cones_l456_456545

theorem apex_angle_of_identical_cones 
  (A : Point) 
  (cone1 cone2 cone3 cone4 : Cone) 
  (identical_cones : cone1.apex_angle = cone2.apex_angle) 
  (cone3_apex_angle : cone3.apex_angle = π / 4) 
  (all_tangent_to_cone4 : ∀ (cone : Cone), cone ∈ {cone1, cone2, cone3} → tangent_internal cone cone4) 
  (cone4_apex_angle : cone4.apex_angle = 3 * π / 4) : 
  cone1.apex_angle = 2 * arctan (2 / 3) :=
by
  sorry

end apex_angle_of_identical_cones_l456_456545


namespace part_I_part_II_l456_456809

noncomputable def A (a : ℝ) := {x | (x - 2) * (x - a) = 0}
def B := {x | x * (x - 1) = 0}
def C (a : ℝ) := A a ∪ B

open Set

-- Part (I)
theorem part_I (a : ℝ) (h1 : a = 1) : 
  (A a ∩ B = {1}) ∧ (A a ∪ B = {0, 1, 2}) := sorry

-- Part (II)
theorem part_II {a : ℝ} :
  (∃ C, (C = A a ∪ B) ∧ (C.finite ∧ C.card = 3)) ↔ (a ∈ {0, 1, 2}) := sorry

end part_I_part_II_l456_456809


namespace correct_proposition_l456_456357

variables (α β : Plane) (m l : Line)

-- Conditions
def condition_1 : α ∩ β = m := sorry
def condition_2 : l ∈ α := sorry
def condition_3 : α ⊥ β := sorry
def condition_4 : l ⊥ m := sorry

-- Conclusion
theorem correct_proposition : l ⊥ β :=
by {
  -- Using the conditions provided
  sorry
}

end correct_proposition_l456_456357


namespace max_n_l456_456005

noncomputable def a : ℕ+ → ℝ
| ⟨1, _⟩ => 1
| ⟨n+1, hn⟩ => real.sqrt (n + 1)

lemma seq_property (n : ℕ+) : (a ⟨n + 1, nat.succ_pos n⟩)^2 - (a n)^2 = 1 :=
by 
  induction n using nat.strong_induction_on with k hk
  cases k
  { simp [a] }
  { simp [a, pow_two, real.sqrt_eq_rpow, nat.succ_eq_add_one, hk k (nat.le_refl k)] 
    rw [real.sqrt_square (nat.cast_nonneg k)]
    rw [real.sqrt_square (nat.cast_nonneg (k+1))]
    ring
  }

theorem max_n (n : ℕ+) : a n < 5 → n ≤ 24 :=
by 
  intros h
  have h1: a n = real.sqrt n, {
    induction n using nat.strong_induction_on with k hk
    cases k
    { simp [a] }
    { rw [a, real.sqrt_eq_rpow, nat.succ_eq_add_one]
      simp [nat.cast_add_one]
      have hx:= hk k (nat.le_refl k)
      rw [hx]
    }
  }
  rw [h1] at h
  have h2 : n < 25, {
    rw [real.sqrt_lt_iff]
    exact_mod_cast h,
    exact zero_le _,
  }
  exact nat.le_pred_of_lt h2

end max_n_l456_456005


namespace sum_of_digits_is_23_l456_456010

theorem sum_of_digits_is_23 (w x y z : ℕ) (hw : w ∈ finset.range 10) (hx : x ∈ finset.range 10)
  (hy : y ∈ finset.range 10) (hz : z ∈ finset.range 10) : 
  (w ≠ x) → (w ≠ y) → (w ≠ z) → (x ≠ y) → (x ≠ z) → (y ≠ z) →
  y + w = 10 → x + y = 9 →
  w + z = 10 → w + x + y + z = 23 :=
by
  sorry

end sum_of_digits_is_23_l456_456010


namespace no_integer_roots_l456_456852

-- Definitions based on conditions
def P (x : ℕ) : ℤ := -- Polynomial definition
  sorry -- Polynomial definition goes here based on conditions

-- Conditions a), condition 1 and 2
axiom coeffs_integer : ∀ n : ℕ, ∃ (a : ℕ), P n = a * n -- Polynomial has integer coefficients
axiom P_zero_odd : P 0 % 2 = 1 -- P(0) is odd
axiom P_one_odd : P 1 % 2 = 1 -- P(1) is odd

theorem no_integer_roots : ∀ x : ℤ, P x ≠ 0 :=
by
  sorry

end no_integer_roots_l456_456852


namespace largest_consecutive_even_sum_l456_456188

theorem largest_consecutive_even_sum (a b c : ℤ) (h1 : b = a+2) (h2 : c = a+4) (h3 : a + b + c = 312) : c = 106 := 
by 
  sorry

end largest_consecutive_even_sum_l456_456188


namespace ratio_of_wire_lengths_l456_456285

theorem ratio_of_wire_lengths (L_Bonnie_piece : ℕ) (num_Bonnie_pieces : ℕ) (side_Bonnie : ℕ)
                              (L_Roark_piece : ℕ) (side_Roark : ℕ) :
  (L_Bonnie_piece = 8 ∧ num_Bonnie_pieces = 12 ∧ side_Bonnie = 8 ∧ L_Roark_piece = 2 ∧ side_Roark = 2) →
  side_Bonnie^3 = num_Roark_cubes * side_Roark^3 →
  ∃ (num_Roark_cubes : ℕ), (num_Roark_cubes = 64) →
  let total_length_Bonnie := num_Bonnie_pieces * L_Bonnie_piece,
      length_per_Roark_cube := 12 * L_Roark_piece,
      total_length_Roark := num_Roark_cubes * length_per_Roark_cube
  in total_length_Bonnie / total_length_Roark = 1 / 16 :=
  by
  intros h1 h2 num_Roark_cubes h3
  let total_length_Bonnie := num_Bonnie_pieces * L_Bonnie_piece
  let length_per_Roark_cube := 12 * L_Roark_piece
  let total_length_Roark := num_Roark_cubes * length_per_Roark_cube
  have h4 : total_length_Bonnie = 96 := by rw [h1.1.2, h1.1.1]; ring
  have h5 : total_length_Roark = 1536 := by rw [h1.2, mul_comm num_Roark_cubes 12]; ring
  exact eq_of_div_eq_one_div (by rw [h4, h5]; norm_num) sorry

end ratio_of_wire_lengths_l456_456285


namespace floor_T_equals_150_l456_456822

variable {p q r s : ℝ}

theorem floor_T_equals_150
  (hpq_sum_of_squares : p^2 + q^2 = 2500)
  (hrs_sum_of_squares : r^2 + s^2 = 2500)
  (hpq_product : p * q = 1225)
  (hrs_product : r * s = 1225)
  (hp_plus_s : p + s = 75) :
  ∃ T : ℝ, T = p + q + r + s ∧ ⌊T⌋ = 150 :=
by
  sorry

end floor_T_equals_150_l456_456822


namespace p_necessary_not_sufficient_for_q_l456_456064

variable (x : ℝ)

def p := x > 2
def q := x > 3

theorem p_necessary_not_sufficient_for_q : (q → p) ∧ ¬(p → q) := by
  sorry

end p_necessary_not_sufficient_for_q_l456_456064


namespace emma_final_amount_l456_456309

theorem emma_final_amount
  (initial_amount : ℕ)
  (furniture_cost : ℕ)
  (fraction_given_to_anna : ℚ)
  (amount_left : ℕ) :
  initial_amount = 2000 →
  furniture_cost = 400 →
  fraction_given_to_anna = 3 / 4 →
  amount_left = initial_amount - furniture_cost →
  amount_left - (fraction_given_to_anna * amount_left : ℚ) = 400 :=
by
  intros h_initial h_furniture h_fraction h_amount_left
  sorry

end emma_final_amount_l456_456309


namespace chord_length_correct_l456_456731

noncomputable def length_of_chord_cut_by_line (t : ℝ) (θ : ℝ) (x y : ℝ) : ℝ :=
  let parametric_line := (1 + t, √3 * t)
  let polar_curve := 4 * cos θ
  let cartesian_curve := x^2 - 4 * x + y^2 = 0
  let line_eq := y = √3 * (x - 1)
  let intersections := [ ( (5 + √13) / 4, (5 - √13) / 4) ]
  let chord_length := (5 + √13) / 4 - (5 - √13) / 4
  chord_length

theorem chord_length_correct (t θ x y : ℝ) :
  let parametric_line := (1 + t, √3 * t) in
  let polar_curve := 4 * cos θ in
  let cartesian_curve := x^2 - 4 * x + y^2 = 0 in
  let line_eq := y = √3 * (x - 1) in
  chord_length_correct t θ x y = √13 :=
sorry

end chord_length_correct_l456_456731


namespace total_people_participated_l456_456128

theorem total_people_participated 
  (N f p : ℕ)
  (h1 : N = f * p)
  (h2 : N = (f - 10) * (p + 1))
  (h3 : N = (f - 25) * (p + 3)) : 
  N = 900 :=
by 
  sorry

end total_people_participated_l456_456128


namespace triangle_perimeter_ABF_l456_456369

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 25) + (y^2 / 21) = 1

-- Define the line
def line (x : ℝ) : Prop := x = -2

-- Define the foci of the ellipse
def right_focus : ℝ := 2
def left_focus : ℝ := -2

-- Points A and B are on the ellipse and line
def point_A (x y : ℝ) : Prop := ellipse x y ∧ line x
def point_B (x y : ℝ) : Prop := ellipse x y ∧ line x

-- Point F is the right focus of the ellipse
def point_F (x y : ℝ) : Prop := x = right_focus ∧ y = 0

-- Perimeter of the triangle ABF
def perimeter (A B F : ℝ × ℝ) : ℝ :=
  sorry -- Calculation of the perimeter of triangle ABF

-- Theorem statement that perimeter is 20
theorem triangle_perimeter_ABF 
  (A B F : ℝ × ℝ) 
  (hA : point_A (A.fst) (A.snd)) 
  (hB : point_B (B.fst) (B.snd))
  (hF : point_F (F.fst) (F.snd)) :
  perimeter A B F = 20 :=
sorry

end triangle_perimeter_ABF_l456_456369


namespace round_trip_exists_l456_456547

theorem round_trip_exists (n : ℕ) (h : n ≥ 11) :
  ∃ (G : SimpleGraph (Fin n)) (company : G.E → Fin 3),
    ∃ (c : Fin 3), ∃ (v1 v2 v3 v4 : Fin n),
      G.Adj v1 v2 ∧ company ⟨v1, v2⟩ = c ∧
      G.Adj v2 v3 ∧ company ⟨v2, v3⟩ = c ∧
      G.Adj v3 v4 ∧ company ⟨v3, v4⟩ = c ∧
      G.Adj v4 v1 ∧ company ⟨v4, v1⟩ = c :=
sorry

end round_trip_exists_l456_456547


namespace proof_of_x_power_y_l456_456407

theorem proof_of_x_power_y (x y : ℝ) (h1 : x - 3 ≥ 0) (h2 : 3 - x ≥ 0) (hy : y = sqrt (x - 3) + sqrt (3 - x) + 2) : x^y = 9 := 
sorry

end proof_of_x_power_y_l456_456407


namespace residues_of_f_singularities_l456_456663

open Complex

noncomputable def f (z : ℂ) : ℂ := (sin z / cos z) / (z^2 - (π / 4) * z)

theorem residues_of_f_singularities :
  residue (f) 0 = 0 ∧
  residue (f) (π / 4) = 4 / π ∧
  ∀ k : ℤ, residue (f) (π / 2 + k * π) = -1 / ((π / 2 + k * π) * (π / 4 + k * π)) :=
by
  sorry

end residues_of_f_singularities_l456_456663


namespace time_to_pass_man_l456_456935

-- Define the necessary variables and conditions
def train_length : ℝ := 110 -- in meters
def train_speed_kmh : ℝ := 60 -- in km/hr
def man_speed_kmh : ℝ := 6 -- in km/hr

-- Conversion factor
def kmh_to_ms (speed_kmh : ℝ) : ℝ := speed_kmh * (5 / 18) -- Convert km/hr to m/s

-- Calculate the relative speed
def relative_speed : ℝ := kmh_to_ms (train_speed_kmh + man_speed_kmh) -- Relative speed in m/s

-- The key theorem to prove
theorem time_to_pass_man : (train_length / relative_speed) = 6 := 
by 
  -- steps can be filled in later, here we assume the final correct end answer is '6'
  sorry

end time_to_pass_man_l456_456935


namespace parallel_of_quadrilateral_l456_456002

variable {A B C D K M : Point}
variable {BK AD AM BC CD : Line}
variable {AC BD KM : Line}

axiom quadrilateral (A B C D : Point) : Prop
axiom on_lines_ac_bd (K M : Point) (AC BD : Line) : Prop
axiom parallel_bk_ad (BK AD : Line) : Prop
axiom parallel_am_bc (AM BC : Line) : Prop
axiom parallel_km_cd (KM CD : Line) : Prop

theorem parallel_of_quadrilateral (A B C D K M : Point) (AC BD KM : Line)
  (hquad: quadrilateral A B C D)
  (hon_lines: on_lines_ac_bd K M AC ∧ on_lines_ac_bd K M BD)
  (hpar_bk_ad: parallel_bk_ad BK AD)
  (hpar_am_bc: parallel_am_bc AM BC)
  : parallel_km_cd KM CD := 
sorry

end parallel_of_quadrilateral_l456_456002


namespace hannah_practice_hours_l456_456398

theorem hannah_practice_hours (weekend_hours : ℕ) (total_weekly_hours : ℕ) (more_weekday_hours : ℕ)
  (h1 : weekend_hours = 8)
  (h2 : total_weekly_hours = 33)
  (h3 : more_weekday_hours = 17) :
  (total_weekly_hours - weekend_hours) - weekend_hours = more_weekday_hours :=
by
  sorry

end hannah_practice_hours_l456_456398


namespace julia_shortfall_l456_456898

-- Definitions based on the problem conditions
def rock_and_roll_price : ℕ := 5
def pop_price : ℕ := 10
def dance_price : ℕ := 3
def country_price : ℕ := 7
def quantity : ℕ := 4
def julia_money : ℕ := 75

-- Proof problem: Prove that Julia is short $25
theorem julia_shortfall : (quantity * rock_and_roll_price + quantity * pop_price + quantity * dance_price + quantity * country_price) - julia_money = 25 := by
  sorry

end julia_shortfall_l456_456898


namespace question1_question2_l456_456046

noncomputable def A (x : ℝ) : Prop := x^2 - 3 * x + 2 ≤ 0
noncomputable def B_set (x a : ℝ) : ℝ := x^2 - 2 * x + a
def B (y a : ℝ) : Prop := y ≥ a - 1
noncomputable def C (x a : ℝ) : Prop := x^2 - a * x - 4 ≤ 0

def prop_p (a : ℝ) : Prop := ∃ x, A x ∧ B (B_set x a) a
def prop_q (a : ℝ) : Prop := ∀ x, A x → C x a

theorem question1 (a : ℝ) (h : ¬ prop_p a) : a > 3 :=
sorry

theorem question2 (a : ℝ) (hp : prop_p a) (hq : prop_q a) : 0 ≤ a ∧ a ≤ 3 :=
sorry

end question1_question2_l456_456046


namespace semicircle_perimeter_correct_l456_456530

noncomputable def radius : ℝ := 31.50774690151576

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def diameter (r : ℝ) : ℝ := 2 * r

noncomputable def semicircle_perimeter (r : ℝ) : ℝ :=
  circumference(r) / 2 + diameter(r)

theorem semicircle_perimeter_correct :
  semicircle_perimeter(radius) ≈ 162.12300409103152 := sorry

end semicircle_perimeter_correct_l456_456530


namespace S_is_closed_curve_and_area_l456_456831

open Complex

noncomputable def four_presentable (z : ℂ) : Prop :=
  ∃ w : ℂ, abs w = 4 ∧ z = w - 1 / w

def S : Set ℂ := {z | four_presentable z}

theorem S_is_closed_curve_and_area :
  (∃ C : Set ℂ, IsClosed C ∧ S ⊆ C) ∧
  ∃ A : ℝ, A = (255 / 16) * Real.pi :=
by
  sorry

end S_is_closed_curve_and_area_l456_456831


namespace percentage_of_water_in_mixture_l456_456961

-- Definitions based on conditions from a)
def original_price : ℝ := 1 -- assuming $1 per liter for pure dairy
def selling_price : ℝ := 1.25 -- 25% profit means selling at $1.25
def profit_percentage : ℝ := 0.25 -- 25% profit

-- Theorem statement based on the equivalent problem in c)
theorem percentage_of_water_in_mixture : 
  (selling_price - original_price) / selling_price * 100 = 20 :=
by
  sorry

end percentage_of_water_in_mixture_l456_456961


namespace polynomial_non_negative_l456_456191

theorem polynomial_non_negative (a : ℝ) : a^2 * (a^2 - 1) - a^2 + 1 ≥ 0 := by
  -- we would include the proof steps here
  sorry

end polynomial_non_negative_l456_456191


namespace harmonic_inequality_l456_456376

theorem harmonic_inequality (n : ℕ) (n_pos : 0 < n) :
  (finset.range (2^n)).sum (λ i, 1 / (i + 1)) > n / 2 :=
sorry

end harmonic_inequality_l456_456376


namespace problem_part_1_problem_part_2_l456_456585

theorem problem_part_1 
  (p_A : ℚ) (p_B : ℚ)
  (hA : p_A = 2 / 3) 
  (hB : p_B = 3 / 4) : 
  1 - p_A ^ 3 = 19 / 27 :=
by sorry

theorem problem_part_2 
  (p_A : ℚ) (p_B : ℚ)
  (hA : p_A = 2 / 3) 
  (hB : p_B = 3 / 4) 
  (h1 : 3 * (p_A ^ 2) * (1 - p_A) = 4 / 9)
  (h2 : 3 * p_B * ((1 - p_B) ^ 2) = 9 / 64) : 
  (4 / 9) * (9 / 64) = 1 / 16 :=
by sorry

end problem_part_1_problem_part_2_l456_456585


namespace angle_DEO_90_l456_456845

-- Triangle ABC with specific points marked and properties.
variables {A B C K L M D E O : Type*}
variables [point A] [point B] [point C] [point K] [point L] [point M] [point D] [point E] [point O]

-- Geometrical conditions as hypotheses.
hypothesis h1 : point_on_line K A B
hypothesis h2 : point_on_line L A C
hypothesis h3 : point_on_line M B C
hypothesis h4 : AK = AL
hypothesis h5 : BK = BM
hypothesis h6 : parallel_line LM A B
hypothesis h7 : tangent_to_circumcircle_at L KLM D
hypothesis h8 : parallel_line_through D A B intersects_at E BC
hypothesis h9 : circumcenter K L M = O

-- Question: What is the angle ∠DEO?
theorem angle_DEO_90 : ∠DEO = 90 := 
by
  sorry

end angle_DEO_90_l456_456845


namespace solve_for_a_l456_456747

theorem solve_for_a (a : ℝ) (h : |2 * a + 1| = 3 * |a| - 2) : a = -1 ∨ a = 3 :=
by
  sorry

end solve_for_a_l456_456747


namespace find_x_of_parallel_vectors_l456_456051

theorem find_x_of_parallel_vectors (x : ℝ) : 
  let a := (2 : ℝ, 3 : ℝ)
  let b := (x, 6 : ℝ)
  (a.1 / b.1 = a.2 / b.2) → x = 4 :=
by 
  sorry

end find_x_of_parallel_vectors_l456_456051


namespace num_integers_for_polynomial_negative_l456_456671

open Int

theorem num_integers_for_polynomial_negative :
  ∃ (set_x : Finset ℤ), set_x.card = 12 ∧ ∀ x ∈ set_x, (x^4 - 65 * x^2 + 64) < 0 :=
by
  sorry

end num_integers_for_polynomial_negative_l456_456671


namespace boat_distance_along_stream_l456_456425

variable (v_s : ℝ)
variable (speed_still : ℝ := 8)
variable (distance_against_stream : ℝ := 5)
variable (time : ℝ := 1)

theorem boat_distance_along_stream :
  (8 - v_s) * 1 = 5 → 8 + v_s * 1 = 11 :=
by
  intro h
  have vs : v_s = 3 := by
    linarith
  rw [vs]
  simp
  -- specific distance calculation
  have dist_along_stream : 8 + 3 * 1 = 11 := by
    linarith
  exact dist_along_stream

end boat_distance_along_stream_l456_456425


namespace find_angle_between_slant_height_and_axis_l456_456260

noncomputable def angle α : Prop :=
∀ (R h : ℝ) (V V₁ V₂ : ℝ) (A B O M L K : ℝ) (cone : ℝ → ℝ → ℝ),
  let V := (1 / 3) * Real.pi * R^2 * h in
  let V_half := (1 / 2) * V in
  let α := Real.arccos (1 / 2^(1/4)) in
  ((1 / 3) * Real.pi * (R * Real.cos(α))^2 * h = V_half) ∧
  cos(α)^4 = 1 / 2

theorem find_angle_between_slant_height_and_axis : ∃ α, angle α :=
begin
  use Real.arccos (1 / 2^(1/4)),
  sorry
end

end find_angle_between_slant_height_and_axis_l456_456260


namespace james_present_age_l456_456224

-- Definitions and conditions
variables (D J : ℕ) -- Dan's and James's ages are natural numbers

-- Condition 1: The ratio between Dan's and James's ages
def ratio_condition : Prop := (D * 5 = J * 6)

-- Condition 2: In 4 years, Dan will be 28
def future_age_condition : Prop := (D + 4 = 28)

-- The proof goal: James's present age is 20
theorem james_present_age : ratio_condition D J ∧ future_age_condition D → J = 20 :=
by
  sorry

end james_present_age_l456_456224


namespace share_A_l456_456933

theorem share_A (a_invest : ℕ) (b_invest : ℕ) (c_invest : ℕ) (b_share : ℕ) : 
  a_invest = 15000 → b_invest = 21000 → c_invest = 27000 → b_share = 1540 → 
  (5 :ℕ) * (b_share * 3 / 7 : ℕ) / 21 = 1100 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end share_A_l456_456933


namespace chord_lengths_sum_l456_456593

noncomputable def circle_equation := ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 25
noncomputable def point_A := (5 : ℝ, 1 : ℝ)

theorem chord_lengths_sum (a b : ℝ) (h_longest : a = 10) (h_shortest : b = 2) : a + b = 12 :=
by
  simp [h_longest, h_shortest]
  sorry

end chord_lengths_sum_l456_456593


namespace incircle_tangent_to_adc_sides_l456_456343

noncomputable def Triangle (A B C : Point) : Prop := -- defining Triangle for context
  True

noncomputable def CircleTangentToSidesAndExtensions (circle : Circle) (AC BA BC : Line) : Prop := -- tangent condition
  True

noncomputable def Parallelogram (A B C D : Point) : Prop := -- defining Parallelogram for context
  True

theorem incircle_tangent_to_adc_sides 
  (A B C D P S Q R : Point)
  (AC BA BC DA DC : Line)
  (circle : Circle) 
  (h_parallelogram : Parallelogram A B C D)
  (h_tangent : CircleTangentToSidesAndExtensions circle AC BA BC)
  (h_intersection : LineIntersectsSegmentsInPoints (line_through P S) DA DC Q R) :
  TangentToIncircleAtPoints (Triangle C D A) (incircle (Triangle C D A)) Q R :=
by
  sorry

end incircle_tangent_to_adc_sides_l456_456343


namespace spherical_to_rectangular_coordinates_l456_456688

theorem spherical_to_rectangular_coordinates (ρ θ φ : ℝ) 
  (h1 : ρ * sin φ * cos θ = 3)
  (h2 : ρ * sin φ * sin θ = 6)
  (h3 : ρ * cos φ = -2) :
  ρ * sin (2 * π - φ) * cos θ = 3 ∧ 
  ρ * sin (2 * π - φ) * sin θ = 6 ∧ 
  ρ * cos (2 * π - φ) = -2 := 
by
  sorry

end spherical_to_rectangular_coordinates_l456_456688


namespace find_third_in_line_l456_456279

-- Define the possible individuals
inductive Person
| Abby
| Bret
| Carl
| Dana

open Person -- Allow direct use of the individual names

-- Define the positions in the line based on the problem's conditions
def standing_second := Dana

-- Define Joe's incorrect claims
def claim1 := "Abby is right behind Carl"
def claim2 := "Dana is between Abby and Bret"
def claim1_false := true -- Express that claim1 is known to be false
def claim2_false := true -- Express that claim2 is known to be false

-- Define the theorem that needs to be proved
theorem find_third_in_line (h1 : standing_second = Dana) (h2 : claim1_false) (h3 : claim2_false) : ∃ p, p = Bret ∧ (∃ n, n = 3 ∧ person_at_position n = p) :=
by
  -- Add the proof here
  sorry

end find_third_in_line_l456_456279


namespace alice_age_square_sum_digits_l456_456981

theorem alice_age_square_sum_digits :
  ∀ (A B C : ℕ),
  C = 2 →
  A = B + 2 →
  (∀ n : ℕ, (B + n) % (C + n) = 0 → n ∈ (finset.range 8).erase 0) →
  (let next_square := ((nat.sqrt A + 1) ^ 2) in
  next_square > A → nat.digits 10 next_square).sum = 9 :=
by
  intros A B C hC hA hCondition next_square hnxt
  have hAliceAge : A + 1 + 2 * 2 + 4^2 ≠ 9 := sorry
  ...
  sorry

end alice_age_square_sum_digits_l456_456981


namespace count_world_complete_symmetry_days_in_millennium_l456_456552

def is_palindrome (s : String) : Prop :=
  s = s.reverse

def is_world_complete_symmetry_day (d : String) : Prop :=
  is_palindrome d

def date_in_range (d : String) : Prop :=
  "20010101" ≤ d ∧ d ≤ "29991231"

theorem count_world_complete_symmetry_days_in_millennium :
  (finset.filter (λ d, is_world_complete_symmetry_day d)
    ((finset.range (29991231 - 20010101 + 1)).image (λ n, (20010101 + n).to_string))).card = 36 :=
sorry

end count_world_complete_symmetry_days_in_millennium_l456_456552


namespace grid_problem_l456_456419

theorem grid_problem 
  (A B : ℕ) 
  (grid : (Fin 3) → (Fin 3) → ℕ)
  (h1 : ∀ i, grid 0 i ≠ grid 1 i)
  (h2 : ∀ i, grid 0 i ≠ grid 2 i)
  (h3 : ∀ i, grid 1 i ≠ grid 2 i)
  (h4 : ∀ i, (∃! x, grid x i = 1))
  (h5 : ∀ i, (∃! x, grid x i = 2))
  (h6 : ∀ i, (∃! x, grid x i = 3))
  (h7 : grid 1 2 = A)
  (h8 : grid 2 2 = B) : 
  A + B + 4 = 8 :=
by sorry

end grid_problem_l456_456419


namespace vector_projection_l456_456049

variables (a b : EuclideanSpace ℝ (Fin 3))

def magnitude (v : EuclideanSpace ℝ (Fin 3)) : ℝ := ∥v∥

def projection (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  (inner a b) / (magnitude a)

theorem vector_projection
  (h1 : magnitude a = 5) 
  (h2 : magnitude (a - b) = 6) 
  (h3 : magnitude (a + b) = 4) : 
  projection a b = -1 :=
begin
  sorry
end

end vector_projection_l456_456049


namespace find_n_values_l456_456001

theorem find_n_values (n : ℚ) :
  ( 4 * n ^ 2 + 3 * n + 2 = 2 * n + 2 ∨ 4 * n ^ 2 + 3 * n + 2 = 5 * n + 4 ) →
  ( n = 0 ∨ n = 1 ) :=
by
  sorry

end find_n_values_l456_456001


namespace find_ab_l456_456018

variable (a b : ℝ)

theorem find_ab (h1 : a + b = 4) (h2 : a^3 + b^3 = 136) : a * b = -6 := by
  sorry

end find_ab_l456_456018


namespace find_other_number_l456_456225

variable (A B : ℕ)
variable (LCM : ℕ → ℕ → ℕ)
variable (HCF : ℕ → ℕ → ℕ)

theorem find_other_number (h1 : LCM A B = 2310) 
  (h2 : HCF A B = 30) (h3 : A = 210) : B = 330 := by
  sorry

end find_other_number_l456_456225


namespace car_distance_traveled_l456_456589

-- Define the arithmetic sequence properties.
def first_term : ℕ → ℕ
| 0 => 35
| (n + 1) => (first_term n) - 5

-- Prove that the sum of the sequence up to the point where it stops is 140 meters.
theorem car_distance_traveled : 
  let n := 8
  let a := 35
  let d := -5
  let last_term := 0 
  let sum_upto_stop := ∑ i in Finset.range n, (a + i * d)
  sum_upto_stop = 140 := 
by
  sorry

end car_distance_traveled_l456_456589


namespace min_days_to_find_poisoned_apple_l456_456160

theorem min_days_to_find_poisoned_apple (n : ℕ) (n_pos : 0 < n) : 
  ∀ k : ℕ, 2^k ≥ 2021 → k ≥ 11 :=
  sorry

end min_days_to_find_poisoned_apple_l456_456160


namespace equilateral_triangle_t_gt_a_squared_l456_456815

theorem equilateral_triangle_t_gt_a_squared {a x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ a) :
  2 * x^2 - 2 * a * x + 3 * a^2 > a^2 :=
by {
  sorry
}

end equilateral_triangle_t_gt_a_squared_l456_456815


namespace center_of_circle_l456_456507

theorem center_of_circle (h k : ℝ) :
  (∀ x y : ℝ, (x - 3) ^ 2 + (y - 4) ^ 2 = 10 ↔ x ^ 2 + y ^ 2 = 6 * x + 8 * y - 15) → 
  h + k = 7 :=
sorry

end center_of_circle_l456_456507


namespace total_dresses_l456_456296

theorem total_dresses (D M E : ℕ) (h1 : E = 16) (h2 : M = E / 2) (h3 : D = M + 12) : D + M + E = 44 :=
by
  sorry

end total_dresses_l456_456296


namespace simplified_expression_value_l456_456864

theorem simplified_expression_value
  (x y : ℝ)
  (h1 : y = sqrt (x - 3) + sqrt (6 - 2 * x) + 2)
  (h2 : x = 3)
  (h3 : y = 2) :
  sqrt (2 * x) * sqrt (x / y) * (sqrt (y / x) + sqrt (1 / y)) = sqrt 6 + 3 * sqrt 2 / 2 :=
  by sorry

end simplified_expression_value_l456_456864


namespace simplify_polynomial_l456_456918

-- Define the original polynomial
def original_expr (x : ℝ) : ℝ := 3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3

-- Define the simplified version of the polynomial
def simplified_expr (x : ℝ) : ℝ := 2 * x^3 - x^2 + 23 * x - 3

-- State the theorem that the original expression is equal to the simplified one
theorem simplify_polynomial (x : ℝ) : original_expr x = simplified_expr x := 
by 
  sorry

end simplify_polynomial_l456_456918


namespace function_passes_through_fixed_point_l456_456379

variables {a : ℝ}

/-- Given the function f(x) = a^(x-1) (a > 0 and a ≠ 1), prove that the function always passes through the point (1, 1) -/
theorem function_passes_through_fixed_point (h1 : a > 0) (h2 : a ≠ 1) :
  (a^(1-1) = 1) :=
by
  sorry

end function_passes_through_fixed_point_l456_456379


namespace f_f_2_eq_2_l456_456112

def f (x : ℝ) : ℝ :=
  if x < 2 then 2 * Real.exp(x - 1)
  else Real.log (x^2 - 1) / Real.log 3

theorem f_f_2_eq_2 : f(f(2)) = 2 :=
by
  simp [f]
  sorry

end f_f_2_eq_2_l456_456112


namespace relationship_among_a_b_c_l456_456361

noncomputable def a : ℝ := 0.99 ^ (1.01 : ℝ)
noncomputable def b : ℝ := 1.01 ^ (0.99 : ℝ)
noncomputable def c : ℝ := Real.log 0.99 / Real.log 1.01

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  sorry

end relationship_among_a_b_c_l456_456361


namespace part1_part2_l456_456704

-- Condition: x = -1 is a solution to 2a + 4x = x + 5a
def is_solution_x (a x : ℤ) : Prop := 2 * a + 4 * x = x + 5 * a

-- Part 1: Prove a = -1 given x = -1
theorem part1 (x : ℤ) (h1 : x = -1) (h2 : is_solution_x a x) : a = -1 :=
by sorry

-- Condition: a = -1
def a_value (a : ℤ) : Prop := a = -1

-- Condition: ay + 6 = 6a + 2y
def equation_in_y (a y : ℤ) : Prop := a * y + 6 = 6 * a + 2 * y

-- Part 2: Prove y = 4 given a = -1
theorem part2 (a y : ℤ) (h1 : a_value a) (h2 : equation_in_y a y) : y = 4 :=
by sorry

end part1_part2_l456_456704


namespace kitchen_upgrade_cost_l456_456909

-- Define the number of cabinet knobs and their cost
def num_knobs : ℕ := 18
def cost_per_knob : ℝ := 2.50

-- Define the number of drawer pulls and their cost
def num_pulls : ℕ := 8
def cost_per_pull : ℝ := 4.00

-- Calculate the total cost of the knobs
def total_cost_knobs : ℝ := num_knobs * cost_per_knob

-- Calculate the total cost of the pulls
def total_cost_pulls : ℝ := num_pulls * cost_per_pull

-- Calculate the total cost of the kitchen upgrade
def total_cost : ℝ := total_cost_knobs + total_cost_pulls

-- Theorem statement
theorem kitchen_upgrade_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_cost_l456_456909


namespace rabbit_speed_l456_456414

theorem rabbit_speed (x : ℕ) :
  2 * (2 * x + 4) = 188 → x = 45 := by
  sorry

end rabbit_speed_l456_456414


namespace distance_from_P_to_F2_l456_456189

theorem distance_from_P_to_F2 :
  let a := 2,
      b := 1,
      e := Real.sqrt (1 - (b^2 / a^2)),
      f1 := (- Real.sqrt (a^2 - b^2) / e, 0),
      f2 := (Real.sqrt (a^2 - b^2) / e, 0),
      p := (- Real.sqrt (a^2 - b^2) / e, 1 / 2) in
  (Real.sqrt ((f2.1 - p.1)^2 + (f2.2 - p.2)^2)) = 7/2 :=
by
  let a := 2
  let b := 1
  let e := Real.sqrt (1 - (b^2 / a^2))
  let f1 := (- Real.sqrt (a^2 - b^2), 0)
  let f2 := (Real.sqrt (a^2 - b^2), 0)
  let p := (- Real.sqrt (a^2 - b^2), 1 / 2)
  dsimp
  sorry  -- To skip the proof.

end distance_from_P_to_F2_l456_456189


namespace problem_l456_456382

noncomputable def f (x : ℝ) : ℝ := -x * |x| + 2 * x

theorem problem (x : ℝ) :
  odd_function f ∧ 
  (∀ x y, x < y → x < -1 → -1 < y → y < 1 → y = 1 → f x ≥ f y) ∧
  (∀ x y, x < y → 1 < x → -1 < y → y < 1 → f x ≥ f y) :=
by
  sorry

end problem_l456_456382


namespace find_a10_l456_456355

open Int

-- Definitions and conditions
def arith_seq (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n) / 2)

def condition1 (a : ℕ → ℤ) (S : ℤ → ℤ) : Prop :=
  a 4 + S 5 = 2

def condition2 (S : ℕ → ℤ) : Prop :=
  S 7 = 14

-- The proof statement
theorem find_a10 (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_arith_seq : arith_seq a)
  (h_sum : sum_first_n_terms a S)
  (h_cond1 : condition1 a S)
  (h_cond2 : condition2 S) :
  a 10 = 14 :=
sorry

end find_a10_l456_456355


namespace exists_not_in_range_f_l456_456231

noncomputable def f : ℝ → ℕ :=
sorry

axiom functional_equation : ∀ (x y : ℝ), f (x + (1 / f y)) = f (y + (1 / f x))

theorem exists_not_in_range_f :
  ∃ n : ℕ, ∀ x : ℝ, f x ≠ n :=
sorry

end exists_not_in_range_f_l456_456231


namespace min_value_fraction_l456_456368

theorem min_value_fraction (a : Nat → ℝ) (m n : ℕ) (q : ℝ) (h1 : q = 2)
  (h2 : ∃ m n a1, a1 > 0 ∧ a m = a1 * q^(m-1) ∧ a n = a1 * q^(n-1) ∧ sqrt (a m * a n) = 4 * a1) : 
  ∃ m n, m > 0 ∧ n > 0 ∧ ∀ m n, (m + n = 6) → (1 / m.toFloat + 4 / n.toFloat) ≥ 3 / 2 :=
by
  sorry

end min_value_fraction_l456_456368


namespace problem_statement_l456_456056

open Set

variable {α : Type*}
variable (U M N : Set α)

-- Definitions according to the conditions in the problem
def U_def := U = {1, 2, 3, 4, 5}
def M_def := M = {1, 2, 3}
def N_def := N = {2, 3, 5}
def C_U_M := {x | x ∈ U ∧ x ∉ M}

-- Lean statement to prove
theorem problem_statement : U_def ∧ M_def ∧ N_def → (C_U_M ∩ N) = {5} :=
by
  intros h
  rcases h with ⟨hU, hM, hN⟩
  rw [U_def] at hU
  rw [M_def] at hM
  rw [N_def] at hN
  sorry

end problem_statement_l456_456056


namespace find_pure_Gala_l456_456959

open Classical

variables (T F G : ℕ)
variable (H1 : F + 0.10 * T = 153)
variable (H2 : F = (3 / 4) * T)

theorem find_pure_Gala : G = T - F → G = 45 :=
begin
  sorry
end

end find_pure_Gala_l456_456959


namespace fraction_zero_implies_x_zero_l456_456069

theorem fraction_zero_implies_x_zero (x : ℝ) (h : x / (2 * x - 1) = 0) : x = 0 := 
by {
  sorry
}

end fraction_zero_implies_x_zero_l456_456069


namespace number_of_jerseys_sold_l456_456873

-- Definitions based on conditions
def revenue_per_jersey : ℕ := 115
def revenue_per_tshirt : ℕ := 25
def tshirts_sold : ℕ := 113
def jersey_cost_difference : ℕ := 90

-- Main condition: Prove the number of jerseys sold is 113
theorem number_of_jerseys_sold : ∀ (J : ℕ), 
  (revenue_per_jersey = revenue_per_tshirt + jersey_cost_difference) →
  (J * revenue_per_jersey = tshirts_sold * revenue_per_tshirt) →
  J = 113 :=
by
  intros J h1 h2
  sorry

end number_of_jerseys_sold_l456_456873


namespace beka_flies_more_l456_456284

theorem beka_flies_more (beka_miles : ℕ) (jackson_miles : ℕ) (h_beka : beka_miles = 873) (h_jackson : jackson_miles = 563) : beka_miles - jackson_miles = 310 := 
by
  rw [h_beka, h_jackson]
  norm_num
  sorry

end beka_flies_more_l456_456284


namespace simon_removes_exactly_180_silver_coins_l456_456073

theorem simon_removes_exactly_180_silver_coins :
  ∀ (initial_total_coins initial_gold_percentage final_gold_percentage : ℝ) 
  (initial_silver_coins final_total_coins final_silver_coins silver_coins_removed : ℕ),
  initial_total_coins = 200 → 
  initial_gold_percentage = 0.02 →
  final_gold_percentage = 0.2 →
  initial_silver_coins = (initial_total_coins * (1 - initial_gold_percentage)) → 
  final_total_coins = (4 / final_gold_percentage) →
  final_silver_coins = (final_total_coins - 4) →
  silver_coins_removed = (initial_silver_coins - final_silver_coins) →
  silver_coins_removed = 180 :=
by
  intros initial_total_coins initial_gold_percentage final_gold_percentage 
         initial_silver_coins final_total_coins final_silver_coins silver_coins_removed
  sorry

end simon_removes_exactly_180_silver_coins_l456_456073


namespace problem_solution_l456_456440

def embedding_condition (n : ℕ) (G : graph) (V P S : set ℝ) (Q : set ℝ) : Prop :=
  ∀ i j, (i ≠ j ∧ edge G V i j) → dist P_i P_j = 1

def tasty_set (G : graph) (V P S : set ℝ) (Q : set ℝ) : Prop :=
  ∃ finite_nonzero : (finite {embeddings : embedding V → embedding Q | ∀ i ∈ S, P_i = Q_i}),
  tasty_set_definition G P Q finite_nonzero

def f (G : graph) : ℕ :=
  smallest_tasty_set_size G

def T (n : ℕ) : set graph :=
  { G | connected G ∧ num_edges G = n - 1 }

def a_n (n : ℕ) : ℝ :=
  expected_value (λ G ∈ T n, (f G)^2 / n^2)

noncomputable def final_answer : ℝ :=
  ⌊ 2019 * real.exp (-2) ⌋ -- lim_{n → ∞} a_n = e^{-2}

theorem problem_solution : final_answer = 273 :=
  by
  sorry

end problem_solution_l456_456440


namespace sum_of_absolute_coefficients_l456_456327

def P (x : ℚ) : ℚ := 1 - (1 / 4) * x + (1 / 8) * x^2

def Q (x : ℚ) : ℚ := P(x) * P(x^2) * P(x^4) * P(x^6) * P(x^8)

theorem sum_of_absolute_coefficients :
  ∑ i in Finset.range 33, |Q.coeff i| = 161051 / 32768 :=
sorry

end sum_of_absolute_coefficients_l456_456327


namespace max_value_of_f_min_value_of_a2_4b2_min_value_of_a2_4b2_equals_l456_456701

noncomputable def f (x a b : ℝ) : ℝ := |x - a| - |x + 2 * b|

theorem max_value_of_f (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∀ x, f x a b ≤ a + 2 * b :=
by sorry

theorem min_value_of_a2_4b2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_max : a + 2 * b = 1) :
  a^2 + 4 * b^2 ≥ 1 / 2 :=
by sorry

theorem min_value_of_a2_4b2_equals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_max : a + 2 * b = 1) :
  ∃ a b, a = 1 / 2 ∧ b = 1 / 4 ∧ (a^2 + 4 * b^2 = 1 / 2) :=
by sorry

end max_value_of_f_min_value_of_a2_4b2_min_value_of_a2_4b2_equals_l456_456701


namespace p1a_p1b_l456_456234

theorem p1a (m : ℕ) (hm : m > 1) : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 - y^2 = m^3 := by
  sorry  -- Proof is omitted

theorem p1b : ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^6 = y^2 + 127 ∧ x = 4 ∧ y = 63 := by
  sorry  -- Proof is omitted

end p1a_p1b_l456_456234


namespace total_area_ABHFGD_l456_456432

noncomputable def side_length : ℝ := real.sqrt 25

structure square (A B C D : Type) :=
(area : ℝ)

axiom square_ABCD : square ℝ ℝ ℝ ℝ
axiom square_EFGD : square ℝ ℝ ℝ ℝ

axiom square_area_ABCD : square_ABCD.area = 25
axiom square_area_EFGD : square_EFGD.area = 25

structure midpoint (H : Type) :=
(mid_BC : H)
(mid_EF : H)

axiom point_H : Type
axiom midpoint_H : midpoint point_H

theorem total_area_ABHFGD : 
  (∀ (ABCD EFGD : square ℝ ℝ ℝ ℝ) (H : point_H), 
    ABCD.area = 25 ∧ EFGD.area = 25 ∧ 
    midpoint_H.mid_BC ∧ midpoint_H.mid_EF 
  → 37.5) := 
by
  intros _ _ _
  intros hABCD hEFGD hmid_BC hmid_EF
  sorry

end total_area_ABHFGD_l456_456432


namespace vector_addition_example_l456_456738

theorem vector_addition_example :
  let a := (1, 2)
  let b := (-2, 1)
  a.1 + 2 * b.1 = -3 ∧ a.2 + 2 * b.2 = 4 :=
by
  sorry

end vector_addition_example_l456_456738


namespace fraction_inequality_l456_456017

theorem fraction_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a < b) (h2 : c < d) : (a + c) / (b + c) < (a + d) / (b + d) :=
by
  sorry

end fraction_inequality_l456_456017


namespace uncle_welly_roses_l456_456550

theorem uncle_welly_roses:
    ∀ (R1 R2 R3 : ℕ), 
    R1 = 50 ∧ 
    R3 = 2 * R1 ∧ 
    R1 + R2 + R3 = 220 → 
    R2 - R1 = 20 := 
by
  intro R1 R2 R3
  intro h
  cases h with e1 h
  cases h with e2 e3
  sorry

end uncle_welly_roses_l456_456550


namespace lucy_fish_bought_l456_456127

def fish_bought (fish_original fish_now : ℕ) : ℕ :=
  fish_now - fish_original

theorem lucy_fish_bought : fish_bought 212 492 = 280 :=
by
  sorry

end lucy_fish_bought_l456_456127


namespace ratio_volumes_l456_456555

theorem ratio_volumes (hA rA hB rB : ℝ) (hA_def : hA = 30) (rA_def : rA = 15) (hB_def : hB = rA) (rB_def : rB = 2 * hA) :
    (1 / 3 * Real.pi * rA^2 * hA) / (1 / 3 * Real.pi * rB^2 * hB) = 1 / 24 :=
by
  -- skipping the proof
  sorry

end ratio_volumes_l456_456555


namespace sqrt_simplify_l456_456159

def sqrt_sum (a b : ℝ) : ℝ := Real.sqrt a + Real.sqrt b

theorem sqrt_simplify : sqrt_sum (10 + 6 * Real.sqrt 3) (10 - 6 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end sqrt_simplify_l456_456159


namespace ali_sold_books_on_thursday_l456_456615

theorem ali_sold_books_on_thursday :
  ∀ (b_stock : ℕ) (sold_mon sold_tue sold_wed sold_fri not_sold : ℕ),
  b_stock = 800 →
  sold_mon = 60 →
  sold_tue = 10 →
  sold_wed = 20 →
  sold_fri = 66 →
  not_sold = 600 →
  (b_stock - not_sold) = (sold_mon + sold_tue + sold_wed + sold_fri + ?[books_sold_thu]) →
  ?[books_sold_thu] = 44 :=
by
  intro b_stock sold_mon sold_tue sold_wed sold_fri not_sold h_b_stock h_sold_mon h_sold_tue h_sold_wed h_sold_fri h_not_sold h_total_sold
  sorry

end ali_sold_books_on_thursday_l456_456615


namespace DF_squared_eq_DM_mul_DA_l456_456420

variables {α : Type*} [linear_ordered_field α]

-- Given points and angles in the problem
variables (A B C D E F M : EuclideanGeometry.Point α)
variable (triangle_ABC : EuclideanGeometry.Triangle A B C)
variable (angle_C : ∠ B C A = π / 3)
variable (on_sides : 
  EuclideanGeometry.on_line_segment B C D ∧ 
  EuclideanGeometry.on_line_segment A B E ∧ 
  EuclideanGeometry.on_line_segment A C F)
variable (M_property : EuclideanGeometry.point_on_line (EuclideanGeometry.line_through A D) M ∧ 
  EuclideanGeometry.point_on_line (EuclideanGeometry.line_through B F) M)
variable (CDEF_rhombus : EuclideanGeometry.is_rhombus C D E F)

-- Define distances
noncomputable def DF := EuclideanGeometry.dist D F
noncomputable def DA := EuclideanGeometry.dist D A
noncomputable def DM := EuclideanGeometry.dist D M

-- Prove the required relation
theorem DF_squared_eq_DM_mul_DA :
  DF^2 = DM * DA :=
sorry

end DF_squared_eq_DM_mul_DA_l456_456420


namespace find_h_l456_456519

noncomputable def y1 (x h j : ℝ) := 4 * (x - h) ^ 2 + j
noncomputable def y2 (x h k : ℝ) := 3 * (x - h) ^ 2 + k

theorem find_h (h j k : ℝ)
  (C1 : y1 0 h j = 2024)
  (C2 : y2 0 h k = 2025)
  (H1 : y1 x h j = 0 → ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ * x₂ = 506)
  (H2 : y2 x h k = 0 → ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ * x₂ = 675) :
  h = 22.5 :=
sorry

end find_h_l456_456519


namespace depth_of_tank_proof_l456_456978

-- Definitions based on conditions
def length_of_tank : ℝ := 25
def width_of_tank : ℝ := 12
def cost_per_sq_meter : ℝ := 0.75
def total_cost : ℝ := 558

-- The depth of the tank to be proven as 6 meters
def depth_of_tank : ℝ := 6

-- Area of the tanks for walls and bottom
def plastered_area (d : ℝ) : ℝ := 2 * (length_of_tank * d) + 2 * (width_of_tank * d) + (length_of_tank * width_of_tank)

-- Final cost calculation
def plastering_cost (d : ℝ) : ℝ := cost_per_sq_meter * (plastered_area d)

-- Statement to be proven in Lean 4
theorem depth_of_tank_proof : plastering_cost depth_of_tank = total_cost :=
by
  sorry

end depth_of_tank_proof_l456_456978


namespace benjamin_billboards_l456_456140

theorem benjamin_billboards (B : ℕ) (h1 : 20 + 23 + B = 60) : B = 17 :=
by
  sorry

end benjamin_billboards_l456_456140


namespace bruno_pens_l456_456624

-- Define Bruno's purchase of pens
def one_dozen : Nat := 12
def half_dozen : Nat := one_dozen / 2
def two_and_half_dozens : Nat := 2 * one_dozen + half_dozen

-- State the theorem to be proved
theorem bruno_pens : two_and_half_dozens = 30 :=
by sorry

end bruno_pens_l456_456624


namespace good_product_sequence_l456_456670

-- Definitions based on conditions  
def is_good_sequence (a : ℕ → ℝ) : Prop :=
  ∀ k n : ℕ, (a^[k] n) > 0

-- Problem statement using the above definitions
theorem good_product_sequence (a b : ℕ → ℝ) 
  (ha: is_good_sequence a) (hb: is_good_sequence b) :
  is_good_sequence (λ n, a n * b n) :=
sorry

end good_product_sequence_l456_456670


namespace probability_convex_quadrilateral_l456_456862

open Finset
open SimpleGraph

noncomputable def num_chords (n : ℕ) : ℕ := (choose n 2)

noncomputable def num_ways_to_select_chords (total_chords selected_chords : ℕ) : ℕ :=
  choose total_chords selected_chords

noncomputable def num_ways_to_select_points (total_points selected_points : ℕ) : ℕ :=
  choose total_points selected_points

theorem probability_convex_quadrilateral :
  let total_points := 7
  let points := range total_points
  let total_chords := num_chords total_points
  let selected_chords := 4
  let ways_select_chords := num_ways_to_select_chords total_chords selected_chords
  let ways_select_points := num_ways_to_select_points total_points 4
  ways_select_points / ways_select_chords = 1 / 171 := by
  sorry

end probability_convex_quadrilateral_l456_456862


namespace damage_ratio_l456_456125

variable (H g τ M : ℝ)
variable (n : ℕ)
variable (k : ℝ)
variable (H_pos : H > 0) (g_pos : g > 0) (n_pos : n > 0) (k_pos : k > 0)

def V_I := sqrt (2 * g * H)
def h := H / n
def V_1 := sqrt (2 * g * h)
def V_1' := (1 / k) * sqrt (2 * g * h)
def V_II := sqrt (2 * g * h / k^2 + 2 * g * (H - h))

def I_I := M * V_I * τ
def I_II := M * τ * ((V_1 - V_1') + V_II)

theorem damage_ratio : 
  I_II / I_I = (k - 1) / (sqrt n * k) + sqrt ((n - 1) * k^2 + 1) / (sqrt n * k^2) → I_II / I_I = 5 / 4 :=
by sorry

end damage_ratio_l456_456125


namespace perpendicular_sufficient_but_not_necessary_l456_456717

variables {a b c : Type}
variables [inner_product_space ℝ a] [inner_product_space ℝ b] [inner_product_space ℝ c] {α : a}

def perpendicular_to_plane (a : a) (α : set a) : Prop :=
∀ (x : a), x ∈ α → ⟪a, x⟫ = 0

def perpendicular_to_lines (a : a) (b c : a) : Prop :=
⟪a, b⟫ = 0 ∧ ⟪a, c⟫ = 0

theorem perpendicular_sufficient_but_not_necessary
  (h₁ : ∀ x : a, x ∈ α → ⟪a, x⟫ = 0)
  (h₂ : b ∈ α)
  (h₃ : c ∈ α) :
  (perpendicular_to_plane a α → perpendicular_to_lines a b c) ∧ 
  (¬ (perpendicular_to_lines a b c → perpendicular_to_plane a α)) :=
by
  sorry

end perpendicular_sufficient_but_not_necessary_l456_456717


namespace secretary_work_hours_l456_456941

theorem secretary_work_hours
  (x : ℕ)
  (h_ratio : 2 * x + 3 * x + 5 * x = 110) :
  5 * x = 55 := 
by
  sorry

end secretary_work_hours_l456_456941


namespace total_monkeys_l456_456614

theorem total_monkeys (x : ℕ) (h : (1 / 8 : ℝ) * x ^ 2 + 12 = x) : x = 48 :=
sorry

end total_monkeys_l456_456614


namespace extreme_values_of_f_l456_456378

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - 4 * x + 4

theorem extreme_values_of_f :
  let critical_points := { x : ℝ | x = -2 ∨ x = 2 },
      endpoints := { x : ℝ | x = -3 ∨ x = 4 },
      values := critical_points ∪ endpoints in
  (∀ x ∈ critical_points, f'(x) = 0) ∧
  (∀ x ∈ values, f(x) ≤ f(4) ∧ f(x) ≥ f(-2)) :=
by
  -- We would provide the detailed proof here, but we'll use sorry to skip.
  sorry

end extreme_values_of_f_l456_456378


namespace cube_volume_l456_456065

theorem cube_volume (S : ℝ) (hS : S = 294) : ∃ V : ℝ, V = 343 := by
  sorry

end cube_volume_l456_456065


namespace magnitude_of_complex_number_l456_456648

def complex_number : ℂ := (2/3 : ℚ) - (4/5 : ℚ) * complex.I

theorem magnitude_of_complex_number :
  complex.abs complex_number = real.sqrt (244) / 15 :=
by
  sorry

end magnitude_of_complex_number_l456_456648


namespace sum_of_k_for_real_distinct_roots_l456_456287

theorem sum_of_k_for_real_distinct_roots :
  let k_values := {k : ℕ | 5*x^2 + 20*x + k = 0 ∧ 400 - 20*k > 0} in
  let k_list := (list.range 19).map (λ n, n + 1) in
  k_list.sum = 190 :=
by
  let k_values := {k : ℕ | 5*x^2 + 20*x + k = 0 ∧ 400 - 20*k > 0}
  let k_list := (list.range 19).map (λ n, n + 1)
  have : k_list.sum = 190 := sorry
  exact this

end sum_of_k_for_real_distinct_roots_l456_456287


namespace poodle_barked_24_times_l456_456259

-- Defining the conditions and question in Lean
def poodle_barks (terrier_barks_per_hush times_hushed: ℕ) : ℕ :=
  2 * terrier_barks_per_hush * times_hushed

theorem poodle_barked_24_times (terrier_barks_per_hush times_hushed: ℕ) :
  terrier_barks_per_hush = 2 → times_hushed = 6 → poodle_barks terrier_barks_per_hush times_hushed = 24 :=
by
  intros
  sorry

end poodle_barked_24_times_l456_456259


namespace scientific_notation_of_30067_l456_456866

theorem scientific_notation_of_30067 : ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 30067 = a * 10^n := by
  use 3.0067
  use 4
  sorry

end scientific_notation_of_30067_l456_456866


namespace red_path_probability_l456_456691

open ProbabilityTheory

-- Definitions of vertices and edges
inductive Vertex
| A
| B
| C
| D
deriving DecidableEq

inductive Edge
| AB
| AC
| AD
| BC
| BD
| CD
deriving DecidableEq

-- Probability distribution over edge colors
def edge_color_distribution : Probability (Edge → Bool) :=
  prob_sequence (λ _ : Edge, prob_bool (1/2))

-- Event representing that a path from A to B is red
def red_path_AB (coloring : Edge → Bool) : Prop :=
  coloring Edge.AB ∨
  (coloring Edge.AC ∧ (coloring Edge.BC ∨ coloring Edge.BD)) ∨
  (coloring Edge.AD ∧ (coloring Edge.BD ∨ coloring Edge.BC))

-- Statement of the problem
theorem red_path_probability :
    P (λ coloring : Edge → Bool, red_path_AB coloring) = 3/4 :=
sorry

end red_path_probability_l456_456691


namespace AK_is_symmedian_of_triangle_l456_456786

theorem AK_is_symmedian_of_triangle 
  (A B C D P Q K : Type) 
  [triangle: triangle A B C]
  (angle_A : angle A B C = 60) 
  (AD_bisector : is_angle_bisector AD A B C)
  (equilateral_PDQ : equilateral_triangle P D Q)
  (height_AD : height AD = DA)
  (PB_QC_intersect_K : intersect PB QC = K) : 
  is_symmedian AK A B C := 
sorry

end AK_is_symmedian_of_triangle_l456_456786


namespace fraction_simplification_l456_456212

theorem fraction_simplification (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) :
  (x / (x - 1) = 3 / (2 * x - 2) - 3) → (2 * x = 3 - 6 * x + 6) :=
by 
  intro h1
  -- Proof steps would be here, but we are using sorry
  sorry

end fraction_simplification_l456_456212


namespace behavior_at_infinity_l456_456292

def g (x : ℝ) : ℝ := -3 * x^4 + 50 * x^2 - 1

theorem behavior_at_infinity :
  (∃ L1 : ℝ, filter.tendsto g filter.at_top (nhds L1) → L1 = -∞) ∧
  (∃ L2 : ℝ, filter.tendsto g filter.at_bot (nhds L2) → L2 = -∞) :=
sorry

end behavior_at_infinity_l456_456292


namespace inscribed_polygon_is_convex_l456_456145

-- Definitions based on the conditions from part (a)
def bounded_convex_curve (K : set (ℝ × ℝ)) := convex K ∧ ∃ M, ∀ (x : ℝ × ℝ), x ∈ K → ∥x∥ ≤ M
def inscribed_polygon (P : list (ℝ × ℝ)) (K : set (ℝ × ℝ)) := ∀ v ∈ P, v ∈ K

-- The main theorem statement
theorem inscribed_polygon_is_convex (K : set (ℝ × ℝ)) (P : list (ℝ × ℝ)) :
  bounded_convex_curve K → inscribed_polygon P K → convex (set_of (λ x, x ∈ P)) :=
by
  intros hK hP
  sorry

end inscribed_polygon_is_convex_l456_456145


namespace geometric_sequence_common_ratio_l456_456535

noncomputable def common_ratio (a S : ℕ → ℝ) (q : ℝ) :=
  ∀ (n : ℕ), S n = a 1 * (1 - q^n) / (1 - q)

noncomputable def limit_relation (a S : ℕ → ℝ) :=
  ∀ (k : ℕ), a k = limit (λ n, S n - S k)

theorem geometric_sequence_common_ratio
  (a S : ℕ → ℝ)
  (q : ℝ)
  (h_sum : common_ratio a S q)
  (h_limit : limit_relation a S) :
  q = 1 / 2 :=
sorry

end geometric_sequence_common_ratio_l456_456535


namespace sum_of_roots_of_quadratic_l456_456183

theorem sum_of_roots_of_quadratic :
  ∀ (N : ℝ), N * (N - 8) = 8 → (N_1 N_2 : ℝ), N^2 - 8 * N - 8 = 0 → N_1 + N_2 = 8 :=
begin
  sorry
end

end sum_of_roots_of_quadratic_l456_456183


namespace domain_transform_l456_456387

-- Definitions based on conditions
def domain_f_x_plus_1 : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
def domain_f_id : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def domain_f_2x_minus_1 : Set ℝ := { x | 0 ≤ x ∧ x ≤ 5/2 }

-- The theorem to prove the mathematically equivalent problem
theorem domain_transform :
  (∀ x, (x + 1) ∈ domain_f_x_plus_1) →
  (∀ y, y ∈ domain_f_2x_minus_1 ↔ 2 * y - 1 ∈ domain_f_id) :=
by
  sorry

end domain_transform_l456_456387


namespace min_diagonal_square_l456_456482

theorem min_diagonal_square (a b : ℝ) (α : ℝ) (S : ℝ) (hS : S = a * b * Real.sin α) :
  ∃ a, b, α, (S = a * b * Real.sin α) ∧ α = π / 2 ∧ a = b :=
by
  sorry

end min_diagonal_square_l456_456482


namespace exists_increasing_triplet_l456_456442

theorem exists_increasing_triplet (f : ℕ → ℕ) (bij : Function.Bijective f) :
  ∃ (a d : ℕ), 0 < a ∧ 0 < d ∧ f a < f (a + d) ∧ f (a + d) < f (a + 2 * d) :=
by
  sorry

end exists_increasing_triplet_l456_456442


namespace product_ab_l456_456742

theorem product_ab (a b : ℚ) (h1 : a + 1/b = 4) (h2 : 1/a + b = 16/15) :
  ∀ p q : ℚ, ab ∈ {p, q} → p * q = 1 :=
sorry

end product_ab_l456_456742


namespace largest_power_of_two_divides_15_6_minus_9_6_l456_456662

def v2 (n : ℕ) : ℕ := Nat.find (λ k => 2^k ∣ n ∧ ¬ 2^(k+1) ∣ n)

theorem largest_power_of_two_divides_15_6_minus_9_6 :
  v2 (15^6 - 9^6) = 5 := sorry

end largest_power_of_two_divides_15_6_minus_9_6_l456_456662


namespace max_radius_of_5_non_intersecting_circles_l456_456553

   noncomputable def max_circle_radius_on_sphere : ℝ :=
   sup {r : ℝ | ∀ (u v : ℝ × ℝ × ℝ), dist u v < 2 * r → ∃ i j, u ∈ circle (i : ℕ) 1 r ∧ v ∈ circle (j : ℕ) 1 r}

   theorem max_radius_of_5_non_intersecting_circles :
     max_circle_radius_on_sphere = 0.485 := sorry
   
end max_radius_of_5_non_intersecting_circles_l456_456553


namespace sequence_formula_sum_sequence_formula_l456_456780

def a (n : ℕ) : ℤ := 
  match n with
  | 0     => 0
  | (n+1) => 10 - 2 * (n + 1)

def S (n : ℕ) : ℤ :=
  if n ≤ 5 then
    9 * n - n^2
  else
    n^2 - 9 * n + 40

theorem sequence_formula (n : ℕ) (h₁ : a 1 = 8) (h₄ : a 4 = 2)
  (h_rec : ∀ n > 0, a (n + 2) - 2 * a (n + 1) + a n = 0) : 
  a n = 10 - 2 * n := sorry

theorem sum_sequence_formula (n : ℕ) : 
  S n = 
  if n ≤ 5 then
    9 * n - n^2
  else
    n^2 - 9 * n + 40 := sorry

end sequence_formula_sum_sequence_formula_l456_456780


namespace inverse_proposition_of_corresponding_angles_congruence_corresponding_angles_imply_parallel_lines_l456_456881

theorem inverse_proposition_of_corresponding_angles_congruence (L₁ L₂ : Line) (angle₁ angle₂ : Angle) 
  (h : corresponding_angles_congruent angle₁ angle₂) : parallel L₁ L₂ :=
sorry

theorem corresponding_angles_imply_parallel_lines (L₁ L₂ : Line) (angle₁ angle₂ : Angle)
  (h_lines_parallel : parallel L₁ L₂)
  (h_congruent : corresponding_angles L₁ L₂ angle₁ angle₂) : corresponding_angles_congruent angle₁ angle₂ :=
sorry

end inverse_proposition_of_corresponding_angles_congruence_corresponding_angles_imply_parallel_lines_l456_456881


namespace complex_product_pure_imaginary_l456_456711

theorem complex_product_pure_imaginary (a b c d : ℝ) (h : (a * c - b * d) + (a * d + b * c) * complex.I = (a * d + b * c) * complex.I) :
  a * c - b * d = 0 ∧ a * d + b * c ≠ 0 :=
by
  sorry

end complex_product_pure_imaginary_l456_456711


namespace quartic_poly_coeff_l456_456003

theorem quartic_poly_coeff:
  ∀ (α β : ℂ), (32 + complex.i) = α * β ∧ (7 + complex.i) = α.im + β.im →
  (let f := (λ  x : ℂ, (x - α) * (x - α.conj) * (x - β) * (x - β.conj));
   let quad_coeff := (@polynomial.coeff (ℂ) f.to_function 2) in quad_coeff.re = 114) :=
by
  sorry

end quartic_poly_coeff_l456_456003


namespace num_students_l456_456540

def has_two_hands (student : Type) := ∀ (s : student), s.hands = 2

variable {student : Type}
variable (class : Type)

def class_size (h : student → ℕ) (hands : ℕ) : ℕ :=
  hands / h class

theorem num_students (students : ℕ) : 
  (class_size (λ _, 2) 20) + 1 = students :=
by 
  -- number of students not including Peter
  have students_not_including_peter : ℕ := 20 / 2,
  -- total number of students including Peter
  have students_including_peter : ℕ := students_not_including_peter + 1,
  exact students_including_peter = students

end num_students_l456_456540


namespace rectangle_area_inscribed_circle_l456_456245

theorem rectangle_area_inscribed_circle (r : ℝ) (h : r = 7) (ratio : ℝ) (hratio : ratio = 3) : 
  (2 * r) * (ratio * (2 * r)) = 588 :=
by
  rw [h, hratio]
  sorry

end rectangle_area_inscribed_circle_l456_456245


namespace vector_dot_product_l456_456395

-- Declaring vectors with given magnitudes and the angle between them
variables (a b : ℝ)
variables (θ : ℝ)
-- Specifying the conditions
def magnitude_a : ℝ := real.sqrt 3
def magnitude_b : ℝ := 2
def angle_in_degrees : ℝ := 30

-- Conversion of angle to radians for cosine calculation in Lean
def angle_in_radians := real.pi * angle_in_degrees / 180

-- Using scalar product formula
def dot_product : ℝ := magnitude_a * magnitude_b * real.cos angle_in_radians

theorem vector_dot_product :
  dot_product = 3 := by
  sorry

end vector_dot_product_l456_456395


namespace derek_dogs_l456_456642

theorem derek_dogs (D C : ℕ) (h1 : D = 3 * C) (h2 : 120 - 90 = 30) (h3 : C + 210 = 2 * 120): D = 90 :=
by {
  have h4 : C = 30, from eq_sub_of_add_eq h3,
  rw [h4, h1],
  exact eq.symm (nat.mul_left_inj zero_lt_three.symm (by exact dec_trivial))
}

end derek_dogs_l456_456642


namespace euclidean_lemma_l456_456851

theorem euclidean_lemma (p a b : ℕ) [prime p] (h : p ∣ a * b) : p ∣ a ∨ p ∣ b :=
by
  apply gauss's_lemma p a b
  exact h
  apply prime p
  sorry

end euclidean_lemma_l456_456851


namespace expression_decrease_l456_456429

-- Definitions based on conditions
variables {x y z : ℝ}

-- Function representing the original algebraic expression
def original_expr : ℝ := x * y^2 * z

-- Function representing the new value of the expression after the changes
def new_expr (x' y' z' : ℝ) : ℝ := x' * y'^2 * z'

-- The conditions
axiom decrease_x : ∀ x, x' = 0.75 * x
axiom decrease_y : ∀ y, y' = 0.75 * y
axiom increase_z : ∀ z, z' = 1.25 * z

-- The statement to prove
theorem expression_decrease : 
  (new_expr (0.75 * x) (0.75 * y) (1.25 * z)) / original_expr = (135 / 256) := by
  sorry

end expression_decrease_l456_456429


namespace kylie_coins_problem_l456_456438

theorem kylie_coins_problem
  (piggy_bank_coins : ℕ := 15)
  (brother_coins : ℕ := 13)
  (father_coins : ℕ := 8)
  (coins_left : ℕ := 15) :
  let total_coins := piggy_bank_coins + brother_coins + father_coins in
  let given_to_laura := total_coins - coins_left in
  given_to_laura = 21 :=
by
  have total_coins := piggy_bank_coins + brother_coins + father_coins
  have given_to_laura := total_coins - coins_left
  sorry

end kylie_coins_problem_l456_456438


namespace find_abs_xyz_l456_456462

noncomputable def distinct_nonzero_real (x y z : ℝ) : Prop :=
x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 

theorem find_abs_xyz
  (x y z : ℝ)
  (h1 : distinct_nonzero_real x y z)
  (h2 : x + 1/y = y + 1/z)
  (h3 : y + 1/z = z + 1/x + 1) :
  |x * y * z| = 1 :=
sorry

end find_abs_xyz_l456_456462


namespace bruno_pens_l456_456627

def dozen := 12
def two_and_one_half_dozens := 2.5

theorem bruno_pens : 2.5 * dozen = 30 := sorry

end bruno_pens_l456_456627


namespace min_value_of_a_plus_2b_l456_456685

theorem min_value_of_a_plus_2b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b - 3) :
  a + 2 * b = 4 * Real.sqrt 2 + 3 :=
sorry

end min_value_of_a_plus_2b_l456_456685


namespace cos_18_degree_eq_l456_456997

noncomputable def cos_18_deg : ℝ :=
  let y := (cos (real.pi / 10))           -- 18 degrees in radians
  let x := (cos (2 * real.pi / 10))       -- 36 degrees in radians
  have h1 : x = 2 * y ^ 2 - 1,
  by sorry,  -- Double angle formula
  have h2 : cos (3 * real.pi / 10) = (cos (π / 2 - 2 * real.pi / 10)),
  by sorry,  -- Triple angle formula for sine
  have h3 : cos (3 * real.pi / 10) = 4 * cos (real.pi / 10)^3 - 3 * cos (real.pi / 10),
  by sorry,  -- Triple angle formula for cosine
  have h4 : cos (π / 2 - 2 * real.pi / 10) = sin (2 * real.pi / 10),
  by sorry,  -- Cosine of complementary angle
  show y = (1 + real.sqrt 5) / 4,
  by sorry

theorem cos_18_degree_eq : cos_18_deg = (1 + real.sqrt 5) / 4 :=
by sorry

end cos_18_degree_eq_l456_456997


namespace area_of_intuitive_diagram_l456_456067

theorem area_of_intuitive_diagram (S S' : ℝ) (h1 : S = 2) (h2 : S' / S = sqrt 2 / 4) : S' = sqrt 2 / 2 := 
by 
  sorry 

end area_of_intuitive_diagram_l456_456067


namespace possible_values_of_tangent_marbles_l456_456684

theorem possible_values_of_tangent_marbles (n : ℕ) (h : n ≤ 16) :
  (∃ (s : set ℕ) (h1 : s ⊆ {1, 2, 3, ..., 16}), -- Consider all subsets of {1, 2, ..., 16}
   ∀ (m ∈ s), ∃ (t : finset ℕ) (h2 : t ⊆ s) (h3 : t.card = 3),  -- Each marble is tangent to exactly 3 others
     ∀ m' ∈ t, (m ≠ m') ∧ (tangent m m'))    -- Tangency condition
  ↔ n ∈ {4, 6, 8, 10, 12, 14, 16} := 
by
  sorry

end possible_values_of_tangent_marbles_l456_456684


namespace intersection_A_B_l456_456697

def A := { x : Real | -3 < x ∧ x < 2 }
def B := { x : Real | x^2 + 4*x - 5 ≤ 0 }

theorem intersection_A_B :
  (A ∩ B = { x : Real | -3 < x ∧ x ≤ 1 }) := by
  sorry

end intersection_A_B_l456_456697


namespace cubicroots_identity_l456_456636

noncomputable def VietaFormulas (a b c : ℝ) : Prop :=
a + b + c = 12 ∧ ab + bc + ca = 17 ∧ abc = -4

theorem cubicroots_identity (a b c : ℝ) (h_eq : VietaFormulas a b c) : 
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 24.0625 := 
by 
  sorry

end cubicroots_identity_l456_456636


namespace incircle_tangent_points_l456_456349

theorem incircle_tangent_points {A B C D P S Q R : Point} (h1 : Parallelogram A B C D)
  (h2 : Circle ∈ TangentToSide (triangle A B C) AC WithTangentPoints (extend BA P) (extend BC S))
  (h3 : Segment PS Intersects AD At Q)
  (h4 : Segment PS Intersects DC At R) :
  Incircle (triangle C D A) IsTangentToSides AD DC AtPoints Q R :=
by sorry

end incircle_tangent_points_l456_456349


namespace distance_of_parallel_lines_l456_456356

noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  abs (C₂ - C₁) / real.sqrt (A^2 + B^2)

theorem distance_of_parallel_lines (a : ℝ)
  (h_parallel : (a ≠ -1) ∧ (a ≠ 0) ∧ ((-1 : ℝ) / a = (a - 1) / 2)) :
  distance_between_parallel_lines 1 2 (1 : ℝ) (4 : ℝ) = 3 * real.sqrt 5 / 5 :=
by
  sorry

end distance_of_parallel_lines_l456_456356


namespace find_range_l456_456877

noncomputable def avg (a b : ℤ) : ℤ :=
(a + b) / 2

def avg_even_integers (start end_ : ℤ) : ℤ :=
avg start end_

theorem find_range :
  avg_even_integers 16 44 - 5 = avg_even_integers 14 36 :=
by
  calc
    avg_even_integers 16 44 = 30 := by sorry
    30 - 5 = 25 := by sorry
    avg_even_integers 14 36 = 25 := by sorry

end find_range_l456_456877


namespace perfect_square_trinomial_m_eq_l456_456403

theorem perfect_square_trinomial_m_eq (
    m y : ℝ) (h : ∃ k : ℝ, 4*y^2 - m*y + 25 = (2*y - k)^2) :
  m = 20 ∨ m = -20 :=
by
  sorry

end perfect_square_trinomial_m_eq_l456_456403


namespace angle_BAO_is_15_degrees_l456_456775

-- Given conditions
variables {A B C D F O : Type} [metric_space O] [semicircle O C D] 
variables (h1 : A ∈ line_segment (D, C)) (h2 : F ∈ semicircle_segment (O, C, D)) 
variables (h3 : B ∈ intersection (line_segment (A, F), semicircle O C D)) (h4 : AB = 2 * OD) 
variables (h5 : ∠ FOD = 60)

-- Goal
theorem angle_BAO_is_15_degrees : ∠ BAO = 15 := by
  sorry

end angle_BAO_is_15_degrees_l456_456775


namespace total_time_required_l456_456970

noncomputable def walking_speed_flat : ℝ := 4
noncomputable def walking_speed_uphill : ℝ := walking_speed_flat * 0.8

noncomputable def running_speed_flat : ℝ := 8
noncomputable def running_speed_uphill : ℝ := running_speed_flat * 0.7

noncomputable def distance_walked_uphill : ℝ := 2
noncomputable def distance_run_uphill : ℝ := 1
noncomputable def distance_run_flat : ℝ := 1

noncomputable def time_walk_uphill := distance_walked_uphill / walking_speed_uphill
noncomputable def time_run_uphill := distance_run_uphill / running_speed_uphill
noncomputable def time_run_flat := distance_run_flat / running_speed_flat

noncomputable def total_time := time_walk_uphill + time_run_uphill + time_run_flat

theorem total_time_required :
  total_time = 0.9286 := by
  sorry

end total_time_required_l456_456970


namespace cookies_initial_count_l456_456793

theorem cookies_initial_count (C : ℕ) (h1 : C / 8 = 8) : C = 64 :=
by
  sorry

end cookies_initial_count_l456_456793


namespace toms_nickels_l456_456196

variables (q n : ℕ)

theorem toms_nickels (h1 : q + n = 12) (h2 : 25 * q + 5 * n = 220) : n = 4 :=
by {
  sorry
}

end toms_nickels_l456_456196


namespace evaluatedExpressionIsApprox_l456_456314

noncomputable def evaluateExpression : ℝ :=
  let expr1 := 6000 - (3^3)
  let expr2 := sqrt expr1
  let expr3 := (105 / 21.0)^2
  expr2 * expr3

theorem evaluatedExpressionIsApprox : abs (evaluateExpression - 1932.25) < 0.01 := by
  sorry

end evaluatedExpressionIsApprox_l456_456314


namespace line_through_point_l456_456302

theorem line_through_point (k : ℝ) :
  (∃ k : ℝ, ∀ x y : ℝ, (x = 3) ∧ (y = -2) → (2 - 3 * k * x = -4 * y)) → k = -2/3 :=
by
  sorry

end line_through_point_l456_456302


namespace ratio_A_B_l456_456586

-- Define the conditions
def C_monthly : ℝ := 14000
def B_monthly : ℝ := C_monthly + 0.12 * C_monthly
def A_annual : ℝ := 470400
def A_monthly : ℝ := A_annual / 12

-- The theorem to prove: the ratio of A's monthly income to B's monthly income is 2.5
theorem ratio_A_B : (A_monthly / B_monthly) = 2.5 := by
  sorry

end ratio_A_B_l456_456586


namespace max_runs_scenario_l456_456417

theorem max_runs_scenario:
  ∀ (balls_per_over : ℕ) (runs_per_ball : ℕ),
    (balls_per_over = 6 → runs_per_ball = 6 → 50 * balls_per_over * runs_per_ball = 1800) ∧
    (balls_per_over = 8 → runs_per_ball = 6 → 50 * balls_per_over * runs_per_ball = 2400) ∧
    (balls_per_over = 10 → runs_per_ball = 6 → 50 * balls_per_over * runs_per_ball = 3000) :=
by
  intro balls_per_over runs_per_ball
  split
  { intro h1 h2
    rw [h1, h2]
    norm_num }
  split
  { intro h1 h2
    rw [h1, h2]
    norm_num }
  { intro h1 h2
    rw [h1, h2]
    norm_num }

end max_runs_scenario_l456_456417


namespace area_of_shape_l456_456658

-- Define the perimeter of the square and the constant representing the solution
constant perimeter : ℝ
constant area_combined : ℝ

-- Provide the given condition
axiom perimeter_eq : perimeter = 48

-- Define the correct answer based on the problem's solution
constant correct_area : ℝ
axiom correct_area_def : correct_area = 144 + 36 * Real.sqrt 3

-- State the theorem to be proved
theorem area_of_shape : area_combined = correct_area :=
by
  -- This part represents our goal without the proof
  sorry

end area_of_shape_l456_456658


namespace convex_polygon_all_diagonals_equal_l456_456235

theorem convex_polygon_all_diagonals_equal (n : ℕ) (h : n ≥ 4) (F : Type) [ConvexPolygon F n]
  (h1 : ∀ d1 d2 : Diagonal F, length d1 = length d2) :
  (n = 4 ∨ n = 5) :=
sorry

end convex_polygon_all_diagonals_equal_l456_456235


namespace cos_18_degree_eq_l456_456999

noncomputable def cos_18_deg : ℝ :=
  let y := (cos (real.pi / 10))           -- 18 degrees in radians
  let x := (cos (2 * real.pi / 10))       -- 36 degrees in radians
  have h1 : x = 2 * y ^ 2 - 1,
  by sorry,  -- Double angle formula
  have h2 : cos (3 * real.pi / 10) = (cos (π / 2 - 2 * real.pi / 10)),
  by sorry,  -- Triple angle formula for sine
  have h3 : cos (3 * real.pi / 10) = 4 * cos (real.pi / 10)^3 - 3 * cos (real.pi / 10),
  by sorry,  -- Triple angle formula for cosine
  have h4 : cos (π / 2 - 2 * real.pi / 10) = sin (2 * real.pi / 10),
  by sorry,  -- Cosine of complementary angle
  show y = (1 + real.sqrt 5) / 4,
  by sorry

theorem cos_18_degree_eq : cos_18_deg = (1 + real.sqrt 5) / 4 :=
by sorry

end cos_18_degree_eq_l456_456999


namespace gcd_lcm_nested_example_l456_456204

open Nat

theorem gcd_lcm_nested_example : 
  gcd (lcm (lcm (gcd 24 (gcd 60 84)) 1 20) 7 5 3) 19 = 1 :=
  sorry

end gcd_lcm_nested_example_l456_456204


namespace cafeteria_apples_pies_l456_456172

theorem cafeteria_apples_pies (initial_apples handed_out_apples apples_per_pie remaining_apples pies : ℕ) 
    (h_initial: initial_apples = 62) 
    (h_handed_out: handed_out_apples = 8) 
    (h_apples_per_pie: apples_per_pie = 9)
    (h_remaining: remaining_apples = initial_apples - handed_out_apples) 
    (h_pies: pies = remaining_apples / apples_per_pie) : 
    pies = 6 := by
  sorry

end cafeteria_apples_pies_l456_456172


namespace final_position_total_distance_fuel_consumption_l456_456480

def distances : List Int := [3, 10, -21, 7, -6, -8, 12, 2, -3, -4, 6, -5]

def fuel_consumption_rate : Float := 0.3

-- The total displacement should be -7 km
theorem final_position : distances.sum = -7 :=
  by
  sorry

-- The total distance traveled should be 87 km
theorem total_distance : distances.map Int.natAbs |>.sum = 87 :=
  by
  sorry

-- The total fuel consumption should be 26.1 liters
theorem fuel_consumption : distances.map Int.natAbs |>.sum * fuel_consumption_rate = 26.1 :=
  by
  sorry

end final_position_total_distance_fuel_consumption_l456_456480


namespace five_thursdays_in_july_l456_456497

theorem five_thursdays_in_july (N : ℕ) (h_june_30_days : ∀ (N : ℕ), true) (h_july_31_days : ∀ (N : ℕ), true) (h_five_tuesdays_in_june : ∃ (t : ℕ), 1 ≤ t ∧ t ≤ 7 ∧ (t + 28 ≤ 30)) :
  ∃ day_in_july, day_in_july = "Thursday" ∧ (day_occurrences_in_july day_in_july = 5) :=  
sorry

end five_thursdays_in_july_l456_456497


namespace tangent_line_parallel_l456_456512

theorem tangent_line_parallel (x y : ℝ) (h_parab : y = 2 * x^2) (h_parallel : ∃ (m b : ℝ), 4 * x - y + b = 0) : 
    (∃ b, 4 * x - y - b = 0) := 
by
  sorry

end tangent_line_parallel_l456_456512


namespace fliers_sent_afternoon_fraction_l456_456559

-- Definitions of given conditions
def total_fliers : ℕ := 2000
def fliers_morning_fraction : ℚ := 1 / 10
def remaining_fliers_next_day : ℕ := 1350

-- Helper definitions based on conditions
def fliers_sent_morning := total_fliers * fliers_morning_fraction
def fliers_after_morning := total_fliers - fliers_sent_morning
def fliers_sent_afternoon := fliers_after_morning - remaining_fliers_next_day

-- Theorem stating the required proof
theorem fliers_sent_afternoon_fraction :
  fliers_sent_afternoon / fliers_after_morning = 1 / 4 :=
sorry

end fliers_sent_afternoon_fraction_l456_456559


namespace equal_triangle_area_l456_456088

theorem equal_triangle_area
  (ABC_area : ℝ)
  (AP PB : ℝ)
  (AB_area : ℝ)
  (PQ_BQ_equal : Prop)
  (AP_ratio: AP / (AP + PB) = 3 / 5)
  (ABC_area_val : ABC_area = 15)
  (AP_val : AP = 3)
  (PB_val : PB = 2)
  (PQ_BQ_equal : PQ_BQ_equal = true) :
  ∃ area, area = 9 ∧ area = 9 :=
by
  sorry

end equal_triangle_area_l456_456088


namespace problem1_problem2_l456_456729

-- Definition of f(x) and the condition of being even
def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x = f (-x)

-- Problem 1: Prove the value of k given f(x) is an even function
theorem problem1 (g : ℝ → ℝ) (k : ℝ) :
  (∀ x : ℝ, 1 / g (10^x + 1) + k * x = 1 / g (10^(-x) + 1) + k * (-x)) →
  k = -1/2 :=
by sorry

-- Definition of f(lgx) for Problem 2
def f_log_x (g : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ :=
1 / g (10^(Real.log10 x) + 1) + k * Real.log10 x

-- Problem 2: Prove the range of m given the solution condition
theorem problem2 (g : ℝ → ℝ) (k : ℝ) (m : ℝ) :
  (∃ x : ℝ, 1/4 ≤ x ∧ x ≤ 4 ∧
    f_log_x g k x - m = 0) →
  Real.log10 2 ≤ m ∧ m ≤ Real.log10 (5 / 2) :=
by sorry

end problem1_problem2_l456_456729


namespace find_a_for_even_function_l456_456366

theorem find_a_for_even_function (f : ℝ → ℝ) (a : ℝ) 
  (h_even : ∀ x, f x = f (-x)) 
  (h_neg : ∀ x, x < 0 → f x = x^2 + a * x) 
  (h_value : f 3 = 3) : a = 2 :=
sorry

end find_a_for_even_function_l456_456366


namespace sum_of_harmonic_numbers_l456_456690

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 1 else Real.log (n + 1) / Real.log n

def is_harmonic_number (k : ℕ) : Prop :=
  ∃ m : ℕ, k + 2 = 2^m

theorem sum_of_harmonic_numbers : 
  ((Finset.range 2011).filter is_harmonic_number).sum = 2026 := 
sorry

end sum_of_harmonic_numbers_l456_456690


namespace total_distance_proof_l456_456602

-- Defining the parameters given in the problem
def man_speed := 8 -- kmph in still water
def river_current := 3.5 -- kmph
def wind_effect := 1.5 -- kmph
def total_time := 2 -- hours

-- Define the effective speeds
def speed_to_place := man_speed - river_current - wind_effect
def speed_to_return := man_speed + river_current + wind_effect

-- Define the distance to the place (to be proven)
def distance_to_place := 4.875 -- km (based on solution computation)

-- Define the total distance rowed
def total_distance_rowed := 2 * distance_to_place

-- Prove that the total distance rowed equals 9.75 km
theorem total_distance_proof : total_distance_rowed = 9.75 :=
by
  sorry

end total_distance_proof_l456_456602


namespace weight_of_rod_l456_456947

-- Define the given conditions for the problem
def length_of_rod : ℝ := 1  -- Length in meters
def specific_gravity_of_iron : ℝ := 7.8  -- in kp/dm^3
def total_cross_sectional_area : ℝ := 188  -- in cm^2

-- Define the proof problem statement
theorem weight_of_rod : 
  let volume := total_cross_sectional_area / 100 * length_of_rod * 10 in -- converting cm^2 to dm^2 and multiply by length in meters converted to dm
  let weight := volume * specific_gravity_of_iron in
  weight = 146.64 :=
by
  sorry

end weight_of_rod_l456_456947


namespace train_usual_time_l456_456936

theorem train_usual_time (S T_new T : ℝ) (h_speed : T_new = 7 / 6 * T) (h_delay : T_new = T + 1 / 6) : T = 1 := by
  sorry

end train_usual_time_l456_456936


namespace num_digits_expr_l456_456992

noncomputable def num_digits (n : ℕ) : ℕ :=
  (Int.ofNat n).natAbs.digits 10 |>.length

def expr : ℕ := 2^15 * 5^10 * 12

theorem num_digits_expr : num_digits expr = 13 := by
  sorry

end num_digits_expr_l456_456992


namespace length_AB_is_correct_l456_456825

noncomputable def length_of_AB (x y : ℚ) : ℚ :=
  let a := 3 * x
  let b := 2 * x
  let c := 4 * y
  let d := 5 * y
  let pq_distance := abs (c - a)
  if 5 * x = 9 * y ∧ pq_distance = 3 then 5 * x else 0

theorem length_AB_is_correct : 
  ∃ x y : ℚ, 5 * x = 9 * y ∧ (abs (4 * y - 3 * x)) = 3 ∧ length_of_AB x y = 135 / 7 := 
by
  sorry

end length_AB_is_correct_l456_456825


namespace polynomial_root_condition_l456_456657

theorem polynomial_root_condition (a : ℚ) :
  (∀ x1 x2 x3 : ℚ, (polynomial.eval x1 (polynomial.C a + polynomial.X * (polynomial.C a + polynomial.X * (polynomial.C a + polynomial.X ^ 3 - 6 * polynomial.X ^ 2))))
   = 0 → (x1 - 3)^3 + (x2 - 3)^3 + (x3 - 3)^3 = 0) 
→ a = 27 / 4 :=
sorry

end polynomial_root_condition_l456_456657


namespace fraction_zero_implies_a_eq_neg2_l456_456755

theorem fraction_zero_implies_a_eq_neg2 (a : ℝ) (h : (a^2 - 4) / (a - 2) = 0) (h2 : a ≠ 2) : a = -2 :=
sorry

end fraction_zero_implies_a_eq_neg2_l456_456755


namespace relationship_among_a_b_c_l456_456362

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.logBase π 3
noncomputable def c : ℝ := Real.logBase 2 (Real.sin (2 * Real.pi / 5))

theorem relationship_among_a_b_c : a > b ∧ b > c :=
by
  -- Trying to prove the sequence a > b > c
  have h1 : a > 1, from sorry,
  have h2 : 0 < b ∧ b < 1, from sorry,
  have h3 : c < 0, from sorry,
  exact ⟨h1.trans (h2.1.symm.ge.trans_lt h2.2), h2.2.trans_le h3⟩

end relationship_among_a_b_c_l456_456362


namespace sum_of_x_sum_of_possible_values_of_x_l456_456868

theorem sum_of_x (x : ℝ) (h : 3 ^ (x^2 + 6 * x + 9) = 27 ^ (x + 3)) : x = 0 ∨ x = -3 := sorry

theorem sum_of_possible_values_of_x : 
  (∑ x, (3 ^ (x^2 + 6 * x + 9) = 27 ^ (x + 3) → (x = 0 ∨ x = -3)) → 
  (x = 0 → x) + (x = -3 → x) = -3) := 
sorry

end sum_of_x_sum_of_possible_values_of_x_l456_456868


namespace intersection_not_in_third_quadrant_l456_456477

def line1 (x : ℝ) (m : ℝ) : ℝ := 2 * x + m
def line2 (x : ℝ) : ℝ := -x + 3

theorem intersection_not_in_third_quadrant : ∀ m : ℝ, ∀ (x : ℝ), let y := line1 x m in y = line2 x → ¬ (x < 0 ∧ y < 0) :=
by sorry

end intersection_not_in_third_quadrant_l456_456477


namespace largest_angle_of_convex_hexagon_l456_456596

theorem largest_angle_of_convex_hexagon 
  (x : ℝ) 
  (hx : (x + 2) + (2 * x - 1) + (3 * x + 1) + (4 * x - 2) + (5 * x + 3) + (6 * x - 4) = 720) :
  6 * x - 4 = 202 :=
sorry

end largest_angle_of_convex_hexagon_l456_456596


namespace sin_identity_cos_identity_l456_456090

-- We declare the variables as real numbers and set the conditions in a triangle
variables {A B C : ℝ}
variables (α β γ : ℝ)
variables (hA : α = A / 2)
variables (hB : β = B / 2)
variables (hC : γ = C / 2)
variables (hSum : α + β + γ = π / 2)

-- 1. We state the theorem for the trigonometric identity involving sine functions.
theorem sin_identity (hA : α = A / 2) (hB : β = B / 2) (hC : γ = C / 2) (hSum : α + β + γ = π / 2) :
  sin α + sin β + sin γ = 1 + 4 * sin ((π - 2 * α) / 4) * sin ((π - 2 * β) / 4) * sin ((π - 2 * γ) / 4) :=
sorry

-- 2. We state the theorem for the trigonometric identity involving cosine functions.
theorem cos_identity (hA : α = A / 2) (hB : β = B / 2) (hC : γ = C / 2) (hSum : α + β + γ = π / 2) :
  cos α + cos β + cos γ = 4 * cos ((π - 2 * α) / 4) * cos ((π - 2 * β) / 4) * cos ((π - 2 * γ) / 4) :=
sorry

end sin_identity_cos_identity_l456_456090


namespace final_replacement_weight_l456_456167

theorem final_replacement_weight (W : ℝ) (a b c d e : ℝ) 
  (h1 : a = W / 10)
  (h2 : b = (W - 70 + e) / 10)
  (h3 : b - a = 4)
  (h4 : c = (W - 70 + e - 110 + d) / 10)
  (h5 : c - b = -2)
  (h6 : d = (W - 70 + e - 110 + d + 140 - 90) / 10)
  (h7 : d - c = 5)
  : e = 110 ∧ d = 90 ∧ 140 = e + 50 := sorry

end final_replacement_weight_l456_456167


namespace find_fraction_of_water_l456_456418

variable (V_a V_w : ℚ)

def fraction_of_alcohol : ℚ := 3/5
def ratio : ℚ := 0.75
def fraction_of_water : ℚ := 4/5

theorem find_fraction_of_water (h₁ : V_a = fraction_of_alcohol) (h₂ : V_a / V_w = ratio) : V_w = fraction_of_water :=
by
  rw [h₁, h₂]
  -- Proof steps would continue here
  sorry

end find_fraction_of_water_l456_456418


namespace incircle_tangent_points_l456_456353

theorem incircle_tangent_points {A B C D P S Q R : Point} 
  (h_parallelogram : parallelogram A B C D) 
  (h_tangent_ac : tangent (circle P Q R) A C) 
  (h_tangent_ba_ext : tangent (circle P Q R) (extension B A P)) 
  (h_tangent_bc_ext : tangent (circle P Q R) (extension B C S)) 
  (h_ps_intersect_da : segment_intersect P S D A Q)
  (h_ps_intersect_dc : segment_intersect P S D C R) :
  tangent (incircle D C A) D A Q ∧ tangent (incircle D C A) D C R := sorry

end incircle_tangent_points_l456_456353


namespace average_mark_second_class_l456_456543

theorem average_mark_second_class
  (avg_mark_class1 : ℝ)
  (num_students_class1 : ℕ)
  (num_students_class2 : ℕ)
  (combined_avg_mark : ℝ) 
  (total_students : ℕ)
  (total_marks_combined : ℝ) :
  avg_mark_class1 * num_students_class1 + x * num_students_class2 = total_marks_combined →
  num_students_class1 + num_students_class2 = total_students →
  combined_avg_mark * total_students = total_marks_combined →
  avg_mark_class1 = 40 →
  num_students_class1 = 30 →
  num_students_class2 = 50 →
  combined_avg_mark = 58.75 →
  total_students = 80 →
  total_marks_combined = 4700 →
  x = 70 :=
by
  intros
  sorry

end average_mark_second_class_l456_456543


namespace vector_computation_l456_456052

def c : ℝ × ℝ × ℝ := (-3, 5, 2)
def d : ℝ × ℝ × ℝ := (5, -1, 3)

theorem vector_computation : 2 • c - 5 • d + c = (-34, 20, -9) := by
  sorry

end vector_computation_l456_456052


namespace incircle_tangent_to_adc_sides_l456_456344

noncomputable def Triangle (A B C : Point) : Prop := -- defining Triangle for context
  True

noncomputable def CircleTangentToSidesAndExtensions (circle : Circle) (AC BA BC : Line) : Prop := -- tangent condition
  True

noncomputable def Parallelogram (A B C D : Point) : Prop := -- defining Parallelogram for context
  True

theorem incircle_tangent_to_adc_sides 
  (A B C D P S Q R : Point)
  (AC BA BC DA DC : Line)
  (circle : Circle) 
  (h_parallelogram : Parallelogram A B C D)
  (h_tangent : CircleTangentToSidesAndExtensions circle AC BA BC)
  (h_intersection : LineIntersectsSegmentsInPoints (line_through P S) DA DC Q R) :
  TangentToIncircleAtPoints (Triangle C D A) (incircle (Triangle C D A)) Q R :=
by
  sorry

end incircle_tangent_to_adc_sides_l456_456344


namespace find_a_l456_456708

theorem find_a (a : ℝ) (h1 : 3 - a > 0) (h2 : 2a + 2 > 0) (h3 : 2a + 2 = 2 * (3 - a)) : a = 1 :=
by 
  sorry

end find_a_l456_456708


namespace incircle_tangent_points_l456_456348

theorem incircle_tangent_points {A B C D P S Q R : Point} (h1 : Parallelogram A B C D)
  (h2 : Circle ∈ TangentToSide (triangle A B C) AC WithTangentPoints (extend BA P) (extend BC S))
  (h3 : Segment PS Intersects AD At Q)
  (h4 : Segment PS Intersects DC At R) :
  Incircle (triangle C D A) IsTangentToSides AD DC AtPoints Q R :=
by sorry

end incircle_tangent_points_l456_456348


namespace problem_l456_456363

def f (x : ℝ) : ℝ := x^2 - 2 * x + 5
def g (x : ℝ) : ℝ := x + 3
def h (x : ℝ) : ℝ := 2 * x - 1

theorem problem : f (g (h 5)) - g (f (h 5)) = 54 :=
by
  have h_of_5 := h 5
  have g_of_h_5 := g h_of_5
  have f_of_h_5 := f h_of_5
  have f_of_g_of_h_5 := f g_of_h_5
  have g_of_f_of_h_5 := g f_of_h_5
  have result := f_of_g_of_h_5 - g_of_f_of_h_5
  -- Computation of h(5)
  calc
    h_of_5 : ℝ := h 5
          ... = 2 * 5 - 1
          ... = 9 : by norm_num

  -- Computation of g(h(5))
  calc
    g_of_h_5 : ℝ := g h_of_5
              ... = g 9
              ... = 12 : by norm_num

  -- Computation of f(h(5))
  calc
    f_of_h_5 : ℝ := f h_of_5
              ... = f 9
              ... = 81 - 18 + 5
              ... = 68 : by norm_num

  -- Computation of f(g(h(5)))
  calc
    f_of_g_of_h_5 : ℝ := f g_of_h_5
                   ... = f 12
                   ... = 144 - 24 + 5
                   ... = 125 : by norm_num

  -- Computation of g(f(h(5)))
  calc
    g_of_f_of_h_5 : ℝ := g f_of_h_5
                   ... = g 68
                   ... = 71 : by norm_num

  -- Final result computation
  calc
    result : ℝ := f_of_g_of_h_5 - g_of_f_of_h_5
                ... = 125 - 71
                ... = 54 : by norm_num

end problem_l456_456363


namespace sum_of_reciprocals_squares_le_two_minus_one_over_n_sum_of_reciprocals_squares_lt_two_l456_456144

theorem sum_of_reciprocals_squares_le_two_minus_one_over_n (n : ℕ) (h : n ≥ 1) : 
  (∑ k in Finset.range (n + 1), 1 / (k + 1)^2) ≤ 2 - 1 / n :=
sorry

theorem sum_of_reciprocals_squares_lt_two (n : ℕ) (h : n ≥ 1) : 
  (∑ k in Finset.range (n + 1), 1 / (k + 1)^2) < 2 :=
sorry

end sum_of_reciprocals_squares_le_two_minus_one_over_n_sum_of_reciprocals_squares_lt_two_l456_456144


namespace sum_weights_second_fourth_l456_456776

-- Definitions based on given conditions
noncomputable section

def weight (n : ℕ) : ℕ := 4 - (n - 1)

-- Assumption that weights form an arithmetic sequence.
-- 1st foot weighs 4 jin, 5th foot weighs 2 jin, and weights are linearly decreasing.
axiom weight_arith_seq (n : ℕ) : weight n = 4 - (n - 1)

-- Prove the sum of the weights of the second and fourth feet
theorem sum_weights_second_fourth :
  weight 2 + weight 4 = 6 :=
by
  simp [weight_arith_seq]
  sorry

end sum_weights_second_fourth_l456_456776


namespace find_interest_rate_l456_456960

-- Definitions from given conditions
def sum_lent := 1000  -- Rs
def time_period := 5  -- years
def interest (P : ℝ) := P - 750

-- Simple interest formula
def simple_interest (P r t : ℝ) := P * r * t / 100

-- Problem to prove: Prove the interest rate
theorem find_interest_rate :
  ∃ r, interest sum_lent = simple_interest sum_lent r time_period :=
begin
  use 5,
  sorry
end

end find_interest_rate_l456_456960


namespace jack_evening_emails_l456_456095

theorem jack_evening_emails (ema_morning ema_afternoon ema_afternoon_evening ema_evening : ℕ)
  (h1 : ema_morning = 4)
  (h2 : ema_afternoon = 5)
  (h3 : ema_afternoon_evening = 13)
  (h4 : ema_afternoon_evening = ema_afternoon + ema_evening) :
  ema_evening = 8 :=
by
  sorry

end jack_evening_emails_l456_456095


namespace calc_expression_l456_456987

theorem calc_expression : (900^2) / (264^2 - 256^2) = 194.711 := by
  sorry

end calc_expression_l456_456987


namespace calculate_CF_l456_456081

-- Definitions of the points and lengths of the sides
def A := ⟨0, 0⟩
def B := ⟨8, 0⟩
def C := ⟨8, 6⟩
def D := ⟨0, 6⟩
def E := ⟨?, ?⟩ -- E needs to be determined such that DEF is a right triangle
def F := ⟨?, ?⟩ -- F point on DC extended

-- Hypotenuse calculation of DE
def DE := (Real.sqrt (8^2 + 6^2))

-- Given condition: B is the quarter-point of DE
def B_Quarter : Prop := (dist B D = 2.5)

-- Proof that CF = 12 given the conditions
theorem calculate_CF (h1: dist B D = 2.5) (h2: dist A B = 8) (h3: dist B C = 6) (h4: dist D C = 8) (h5: dist C F = 12) : dist C F = 12 :=
by 
  sorry

end calculate_CF_l456_456081


namespace triangle_angle_B_is_90_degrees_l456_456850

theorem triangle_angle_B_is_90_degrees
  (A B C T X Y O : Point)
  (h_circle : Circle O X B Y)
  (h_midpoints_XY : (is_midpoint_of_arc X A B) ∧ (is_midpoint_of_arc Y B C))
  (h_T_on_AC : T ∈ Line A C)
  (h_bisectors : (is_angle_bisector (angle ATB) X) ∧ (is_angle_bisector (angle BTC) Y)) :
  angle A B C = 90 := by
  sorry

end triangle_angle_B_is_90_degrees_l456_456850


namespace f_odd_increasing_intervals_no_max_value_extreme_points_l456_456386

open Real

namespace FunctionAnalysis

def f (x : ℝ) := x^3 - x

theorem f_odd : ∀ x : ℝ, f (-x) = -f (x) :=
by
  intro x
  show f (-x) = -f (x)
  calc
    f (-x) = (-x)^3 - (-x) : rfl
    ... = -x^3 + x : by ring
    ... = -(x^3 - x) : by ring
    ... = -f (x) : rfl

theorem increasing_intervals : ∀ x : ℝ, 
  (f' x > 0 ↔ x < -sqrt 3 / 3 ∨ x > sqrt 3 / 3) :=
by
  intro x
  have h_deriv : deriv f x = 3 * x^2 - 1 := deriv_pows x
  rw ← h_deriv
  split
  · intro h
    have : 3 * x^2 - 1 > 0 := h
    split
    · apply Or.inl
      linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]
    · apply Or.inr
      linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]
  · intro h
    cases h
    · linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]
    · linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]

theorem no_max_value : ∃ L : ℝ, ∀ x : ℝ, f(x) < L → False :=
by
  use 1
  intro x h
  have : ∀ x : ℝ, f(x) > x := λ x, by norm_num
  specialize this x
  linarith

theorem extreme_points : ∀ x : ℝ,
  (f' x = 0) ↔ (x = sqrt(3) / 3 ∨ x = -sqrt(3) / 3) :=
by
  intro x
  have h_deriv : deriv f x = 3 * x^2 - 1 := deriv_pows x
  rw ← h_deriv
  split
  · intro h
    solve_by_elim
  · intro h
    solve_by_elim

end FunctionAnalysis

end f_odd_increasing_intervals_no_max_value_extreme_points_l456_456386


namespace right_square_prism_min_circumscribed_sphere_volume_l456_456539

noncomputable def right_square_prism_circumscribed_sphere_volume (a h : ℝ) (hv : a^2 * h = 8) : ℝ :=
  let r := Real.sqrt 3
  in (4/3) * Real.pi * r^3

theorem right_square_prism_min_circumscribed_sphere_volume (a h : ℝ) (hv : a^2 * h = 8) : 
  right_square_prism_circumscribed_sphere_volume a h hv = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end right_square_prism_min_circumscribed_sphere_volume_l456_456539


namespace angle_AEF_l456_456784

-- Define the conditions and the required proof
theorem angle_AEF (theta : ℝ) (alpha : ℝ)
  (h1 : theta = 17)
  (h2 : ∀ (x : ℝ), angle x = theta)
  (h3 : ∀ (y : ℝ), angle y = 34)
  (h4 : (iso triang KLM)) :
  angle AEF = 129 :=
begin
  sorry -- Proof steps will be provided here
end

end angle_AEF_l456_456784


namespace kitchen_upgrade_total_cost_l456_456905

-- Defining the given conditions
def num_cabinet_knobs : ℕ := 18
def cost_per_cabinet_knob : ℚ := 2.50

def num_drawer_pulls : ℕ := 8
def cost_per_drawer_pull : ℚ := 4

-- Definition of the total cost function
def total_cost : ℚ :=
  (num_cabinet_knobs * cost_per_cabinet_knob) + (num_drawer_pulls * cost_per_drawer_pull)

-- The theorem to prove the total cost is $77.00
theorem kitchen_upgrade_total_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_total_cost_l456_456905


namespace fifth_group_members_l456_456244

-- Define the number of members in the choir
def total_members : ℕ := 150 

-- Define the number of members in each group
def group1 : ℕ := 18 
def group2 : ℕ := 29 
def group3 : ℕ := 34 
def group4 : ℕ := 23 

-- Define the fifth group as the remaining members
def group5 : ℕ := total_members - (group1 + group2 + group3 + group4)

theorem fifth_group_members : group5 = 46 := sorry

end fifth_group_members_l456_456244


namespace tree_placement_impossible_l456_456175

theorem tree_placement_impossible
  (length width : ℝ) (h_length : length = 4) (h_width : width = 1) :
  ¬ (∃ (t1 t2 t3 : ℝ × ℝ), 
       dist t1 t2 ≥ 2.5 ∧ 
       dist t2 t3 ≥ 2.5 ∧ 
       dist t1 t3 ≥ 2.5 ∧ 
       t1.1 ≥ 0 ∧ t1.1 ≤ length ∧ t1.2 ≥ 0 ∧ t1.2 ≤ width ∧ 
       t2.1 ≥ 0 ∧ t2.1 ≤ length ∧ t2.2 ≥ 0 ∧ t2.2 ≤ width ∧ 
       t3.1 ≥ 0 ∧ t3.1 ≤ length ∧ t3.2 ≥ 0 ∧ t3.2 ≤ width) := 
by {
  sorry
}

end tree_placement_impossible_l456_456175


namespace find_x_l456_456721

section proof_problem

variables (x y z a b c d : ℝ)

-- Conditions
axiom cond1 : xy / (x + y) = a
axiom cond2 : xz / (x + z) = b
axiom cond3 : yz / (y + z) = c
axiom cond4 : yz / (y - z) = d

-- Final statement to prove
theorem find_x : x = 2 * a * c / (a - c - d) :=
sorry

end proof_problem

end find_x_l456_456721


namespace ten_pow_neg_2y_eq_one_seventh_l456_456409

theorem ten_pow_neg_2y_eq_one_seventh (y : ℝ) (h : 10^(4*y) = 49) : 10^(-2*y) = 1 / 7 := by
  sorry

end ten_pow_neg_2y_eq_one_seventh_l456_456409


namespace closest_to_zero_is_13_l456_456077

noncomputable def a (n : ℕ) : ℤ := 88 - 7 * n

theorem closest_to_zero_is_13 : ∀ (n : ℕ), 1 ≤ n → 81 + (n - 1) * (-7) = a n →
  (∀ m : ℕ, (m : ℤ) ≤ (88 : ℤ) / 7 → abs (a m) > abs (a 13)) :=
  sorry

end closest_to_zero_is_13_l456_456077


namespace geom_seq_term_10_l456_456072

-- Given conditions:
def a₆ : ℝ := 2 / 3
def q : ℝ := Real.sqrt 3

-- Statement to prove:
theorem geom_seq_term_10 : 
  let a₁ := a₆ / q^5 in 
  a₁ * q^9 = 6 := by
  sorry

end geom_seq_term_10_l456_456072


namespace ball_arrangements_l456_456848

theorem ball_arrangements : 
  let balls := [1, 2, 3, 4, 5]
  let boxes := 3
  let box_config := [2, 2, 1] -- two boxes with 2 balls, one box with 1 ball
  in ∃ (n : ℕ), n = 90 :=
by
  sorry

end ball_arrangements_l456_456848


namespace ordered_quadruples_l456_456458

theorem ordered_quadruples (x1 x2 x3 x4 : ℕ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) 
    (h5: x1 % 2 = 0) (h6: x2 % 2 = 0) (h7: x3 % 2 = 0) (h8: x4 % 2 = 0) (h_sum: x1 + x2 + x3 + x4 = 120) : 
    (let n := (finset.range 30).sum (λ y1, (finset.range (30 - y1)).sum (λ y2, (finset.range (30 - y1 - y2)).sum (λ y3, (finset.range 30).card))) in (n / 120) = 271.7) :=
sorry

end ordered_quadruples_l456_456458


namespace curve_C_rectangular_equation_chord_length_on_curve_C_l456_456661

def line_l (t : ℝ) : ℝ × ℝ :=
  let x := -1 + 2 * t
  let y := 4 * t
  (x, y)

def curve_C (θ : ℝ) : ℝ :=
  let ρ := real.sqrt (2 / (2 * real.sin θ ^ 2 + real.cos θ ^ 2))
  ρ

theorem curve_C_rectangular_equation:
  (∃ ρ θ, ρ^2 = 2 / (2 * real.sin θ^2 + real.cos θ^2) ∧ x = ρ * real.cos θ ∧ y = ρ * real.sin θ) ↔
      (∃ x y, x^2 / 2 + y^2 = 1) := 
sorry

theorem chord_length_on_curve_C:
  (∃ t, line_l t = (x, y)) ∧ ((x^2 / 2 + y^2 = 1) ∧ (2 * x - y + 2 = 0)) ↔
      ((∃ x1 x2, 9 * x^2 + 16 * x + 6 = 0 ∧ x1 + x2 = -16 / 9 ∧ x1 * x2 = 6 / 9) ∧
      (let chord_length := real.sqrt ((-16 / 9) ^ 2 - 4 * (6 / 9)) * real.sqrt 5 in chord_length = 10 * real.sqrt 2 / 9)) := 
sorry

end curve_C_rectangular_equation_chord_length_on_curve_C_l456_456661


namespace tickets_bought_l456_456133

theorem tickets_bought
  (olivia_money : ℕ) (nigel_money : ℕ) (ticket_cost : ℕ) (leftover_money : ℕ)
  (total_money : ℕ) (money_spent : ℕ) 
  (h1 : olivia_money = 112) 
  (h2 : nigel_money = 139) 
  (h3 : ticket_cost = 28) 
  (h4 : leftover_money = 83)
  (h5 : total_money = olivia_money + nigel_money)
  (h6 : total_money = 251)
  (h7 : money_spent = total_money - leftover_money)
  (h8 : money_spent = 168)
  : money_spent / ticket_cost = 6 := 
by
  sorry

end tickets_bought_l456_456133


namespace smallest_possible_product_l456_456849

-- Define the conditions of the problem
def isTwoDigitNumberMadeOfDigits (n : ℕ) (d1 d2 : ℕ) : Prop :=
  10 * d1 + d2 = n ∧ d1 ∈ {0, 7, 8, 9} ∧ d2 ∈ {0, 7, 8, 9} ∧ d1 ≠ d2

def allDigitsUsed (d1 d2 d3 d4 : ℕ) : Prop :=
  {d1, d2, d3, d4} = {0, 7, 8, 9}

noncomputable def smallestProductOfTwoDigitNumbers : ℕ :=
  let xs := [0, 7, 8, 9].permutations.map (λ l, 10 * l[0] + l[1] * 10 * l[2] + l[3]) in
  xs.foldl min 10000

-- State the problem in Lean
theorem smallest_possible_product : smallestProductOfTwoDigitNumbers = 623 :=
by sorry

end smallest_possible_product_l456_456849


namespace length_BE_correct_l456_456765

theorem length_BE_correct :
  ∀ (A B C D E : ℝ × ℝ),
    A = (0,4) →
    B = (7,1) →
    C = (5,3) →
    D = (3,1) →
    (∃ E : ℝ × ℝ, 
      collinear A B E ∧ collinear C D E ∧ 
      E = (4.2, 3.2)) →
    (dist B E) = real.sqrt 12.68 :=
by
  intros A B C D E hA hB hC hD hE
  sorry

end length_BE_correct_l456_456765


namespace complex_sum_identity_l456_456460

theorem complex_sum_identity {x : ℂ} (hx1 : x^1005 = 1) (hx2 : x ≠ 1) :
    (∑ k in Finset.range 1005, (x^(2*(k+1)) / (x^(k+1) - 1))) = 502 := 
begin
  -- Proof omitted
  sorry
end

end complex_sum_identity_l456_456460


namespace monotonically_decreasing_function_in_0_infty_is_l456_456984

open Real

def func_a (x : ℝ) : ℝ := 1 / (x - 1)
def func_b (x : ℝ) : ℝ := 2^x
def func_c (x : ℝ) : ℝ := log x / log 2
def func_d (x : ℝ) : ℝ := -x^2

theorem monotonically_decreasing_function_in_0_infty_is :
  ∀ (x y : ℝ), (0 < x) → (0 < y) → (x < y) → func_d x > func_d y :=
by 
  sorry

end monotonically_decreasing_function_in_0_infty_is_l456_456984


namespace frame_width_relative_to_side_length_l456_456971

theorem frame_width_relative_to_side_length
  (y : ℝ) (x : ℝ) 
  (h_ratio : ∃ (long short : ℝ), long / short = √2 ∧ short = y ∧ long = √2 * y)
  (h_area : (√2 * y) * y / 2 = (√2 * y - 2 * x) * (y - 2 * x)) : 
  x = (1 + √2 - √3) / 4 * y :=
by
  sorry

end frame_width_relative_to_side_length_l456_456971


namespace total_deflection_zero_l456_456833

variable (β : ℝ)

theorem total_deflection_zero (hC : deflection C = 2 * β)
                            (hB : deflection B = -2 * β) : 
    deflection_total = 0 :=
by
  sorry

end total_deflection_zero_l456_456833


namespace matt_age_less_4_times_john_l456_456832

theorem matt_age_less_4_times_john (M J : ℕ) (h1 : M = 41) (h2 : J = 11) (h3 : M + J = 52) : 4 * J - M = 3 :=
by
  rw [h1, h2]
  exact calc
    4 * 11 - 41 = 44 - 41 := by simp
            ... = 3       := by simp

#print matt_age_less_4_times_john  -- to verify the statement

end matt_age_less_4_times_john_l456_456832


namespace perimeter_of_isosceles_triangle_l456_456423

-- Define the properties and the main theorem
theorem perimeter_of_isosceles_triangle (ABC : Triangle) 
  (A B C H M K : Point)
  (a : ℝ)
  (h_isosceles : is_isosceles ABC A B C)
  (h_altitude_AH : is_altitude A H B C)
  (h_midpoint_M : is_midpoint M A B)
  (h_perpendicular_MK : is_perpendicular M K A C)
  (h_equal_altitude_perpendicular : AH = MK)
  (h_AK : AK = a) :
  perimeter ABC = 20 * a :=
  sorry

end perimeter_of_isosceles_triangle_l456_456423


namespace measure_of_angle_X_l456_456071
noncomputable theory

def geometric_configuration (Y Z X : ℝ) : Prop :=
  Y = 130 ∧
  Z = 180 - Y ∧
  (∀ A B C : ℝ, A + B + C = 180 → 
    (A = 90 ∧ B = 60 → C = X)) ⟹ X = 30

theorem measure_of_angle_X : ∃ X, geometric_configuration 130 (180 - 130) X :=
by
  sorry

end measure_of_angle_X_l456_456071


namespace number_of_ordered_triples_l456_456323

theorem number_of_ordered_triples :
  (∃ (a b c : ℕ), (0 < a ∧ 0 < b ∧ 0 < c) ∧
    lcm a b = 2000 ∧ lcm b c = 4000 ∧ lcm c a = 4000) →
  (∃ n : ℕ, n = 40) :=
by 
  intro h,
  use 40,
  sorry

end number_of_ordered_triples_l456_456323


namespace solve_inequality_contrapositive_statement_l456_456681

variables (x m : ℝ)

def f (x m : ℝ) : ℝ := x^2 - m * x + 1

theorem solve_inequality (x m : ℝ) :
  f x m > 0 ↔ (m^2 - 4 < 0 ∧ ∀ x, true) ∨ 
               (m^2 - 4 ≥ 0 ∧ x < (m - real.sqrt (m^2 - 4)) / 2 ∨ x > (m + real.sqrt (m^2 - 4)) / 2) := sorry

variables (a b : ℝ)

theorem contrapositive_statement 
  (h1 : ∀ x > 0, x^2 - m * x + 1 ≥ 0) 
  (h2 : a > 0) (h3 : b > 0) :
  (∃ m : ℝ, m ≤ 2 ∧ ∀ a b > 0, (a + b ≤ 1 → 1 / a + 2 / b ≥ 3 + real.sqrt 2 * m)) ∧
  (∀ a b > 0, (1 / a + 2 / b < 3 + real.sqrt 2 * m → a + b > 1)) := sorry

end solve_inequality_contrapositive_statement_l456_456681


namespace compute_focus_d_l456_456621

-- Define the given conditions as Lean definitions
structure Ellipse (d : ℝ) :=
  (first_quadrant : d > 0)
  (F1 : ℝ × ℝ := (4, 8))
  (F2 : ℝ × ℝ := (d, 8))
  (tangent_x_axis : (d + 4) / 2 > 0)
  (tangent_y_axis : (d + 4) / 2 > 0)

-- Define the proof problem to show d = 6 for the given conditions
theorem compute_focus_d (d : ℝ) (e : Ellipse d) : d = 6 := by
  sorry

end compute_focus_d_l456_456621


namespace frequency_in_interval_l456_456769

-- Definitions for the sample size and frequencies in given intervals
def sample_size : ℕ := 20
def freq_10_20 : ℕ := 2
def freq_20_30 : ℕ := 3
def freq_30_40 : ℕ := 4
def freq_40_50 : ℕ := 5

-- The goal: Prove that the frequency of the sample in the interval (10, 50] is 0.7
theorem frequency_in_interval (h₁ : sample_size = 20)
                              (h₂ : freq_10_20 = 2)
                              (h₃ : freq_20_30 = 3)
                              (h₄ : freq_30_40 = 4)
                              (h₅ : freq_40_50 = 5) :
  ((freq_10_20 + freq_20_30 + freq_30_40 + freq_40_50) : ℝ) / sample_size = 0.7 := 
by
  sorry

end frequency_in_interval_l456_456769


namespace proof_problem_l456_456404

theorem proof_problem (x : ℝ) : 8^x - 8^(x-1) = 56 → x = 2 → x^(2*x+1) = 32 :=
by
  intros h1 h2
  rw h2
  norm_num

end proof_problem_l456_456404


namespace infinitely_many_triangles_l456_456153

-- Define the conditions of the problem
def lattice_points (p : ℤ × ℤ) : Prop := true

def consecutive_integers (a b c : ℕ) : Prop :=
  a + 1 = b ∧ b + 1 = c

-- The target theorem to prove that there are infinitely many different triangles
theorem infinitely_many_triangles :
  ∃ (triangles : ℕ → (ℤ × ℤ) × (ℤ × ℤ) × (ℤ × ℤ)),
    (∀ n, 
      let ⟨p1, p2, p3⟩ := triangles n in 
      lattice_points p1 ∧ lattice_points p2 ∧ lattice_points p3 ∧
      ∃ (a b c : ℕ), consecutive_integers a b c ∧
      dist p1 p2 = a ∧ dist p2 p3 = b ∧ dist p3 p1 = c ∧
      ∀ m, triangles n = triangles m → n = m)
:= sorry

end infinitely_many_triangles_l456_456153


namespace proof_problem_l456_456412

theorem proof_problem (x : ℕ) (h : (x - 4) / 10 = 5) : (x - 5) / 7 = 7 :=
  sorry

end proof_problem_l456_456412


namespace g_neither_even_nor_odd_l456_456787

def g (x : ℝ) : ℝ := 1 / (3^x + 2) + 3 / 4

theorem g_neither_even_nor_odd : ¬(∀ x, g (-x) = g x) ∧ ¬(∀ x, g (-x) = -g x) := by
  sorry

end g_neither_even_nor_odd_l456_456787


namespace proof_f_3_plus_f_0_l456_456121

-- Define the function f and the conditions
def f (x : ℝ) : ℝ := sorry

-- Define that f is an odd function
axiom f_is_odd : ∀ x : ℝ, f (-x) = -f x

-- Condition given in the problem
def f_neg3_eq_2 : f (-3) = 2 := sorry

-- The statement to prove
theorem proof_f_3_plus_f_0 : f 3 + f 0 = -2 :=
by
  -- Use the definition that f(x) is an odd function
  have h1 : f 3 = -f (-3), from f_is_odd 3
  -- Use the given condition f(-3) = 2
  have h2 : f (-3) = 2, from f_neg3_eq_2
  -- Substitute f(-3) with 2
  have h3 : f 3 = -2, by rw [h2] at h1; exact h1
  -- Add f(0) which is 0 for an odd function at zero
  have h4 : f 0 = 0, from f_is_odd 0
  -- Add them together
  rw [h3, h4]
  exact rfl

end proof_f_3_plus_f_0_l456_456121


namespace parabola_latus_rectum_l456_456430

theorem parabola_latus_rectum (p : ℝ) (h : (λ (x y : ℝ), y^2 = 2*p*x)) 
    (lr : 2*p = 4) : p = 2 :=
by sorry

end parabola_latus_rectum_l456_456430


namespace magnitude_of_complex_l456_456650

open Complex

theorem magnitude_of_complex :
  abs (Complex.mk (2/3) (-4/5)) = Real.sqrt 244 / 15 :=
by
  -- Placeholder for the actual proof
  sorry

end magnitude_of_complex_l456_456650


namespace propositions_are_true_l456_456856

variable (m n : set ℝ) (α β : set (set ℝ)) [linear_order ℝ]

-- Conditions for the first proposition
axiom prop1_cond1 : m ≠ ∅ ∧ m ⊆ α
axiom prop1_cond2 : m ⊆ α ∧ α ∩ β = n

-- Conditions for the second proposition
axiom prop2_cond1 : m ≠ ∅ ∧ ∀ x ∈ m, ∀ y ∈ α, x ⊥ y
axiom prop2_cond2 : n ≠ ∅ ∧ ∀ x ∈ n, ∀ y ∈ β, x ∥ y ∧ α ∥ β

-- Conditions for the third proposition
axiom prop3_cond1 : m ≠ ∅ ∧ ∀ x ∈ m, ∀ y ∈ α, x ⊥ y
axiom prop3_cond2 : n ≠ ∅ ∧ ∀ x ∈ n, ∀ y ∈ β, x ∥ y ∧ α ∥ β

-- Conditions for the fourth proposition
axiom prop4_cond1 : m ≠ ∅ ∧ ∀ x ∈ m, ∀ y ∈ α, x ⊥ y
axiom prop4_cond2 : n ≠ ∅ ∧ ∀ x ∈ n, ∀ y ∈ β, x ⊥ y ∧ α ⊥ β

-- Proof goal
theorem propositions_are_true : 
  (∀ x ∈ m, ∀ y ∈ n, x ∥ y) ∧ 
  (∀ x ∈ m, ∀ y ∈ n, x ⊥ y) ∧ 
  (∀ x ∈ m, ∀ y ∈ n, x ⊥ y) ∧ 
  (∀ x ∈ m, ∀ y ∈ n, x ⊥ y) := 
by 
  sorry

end propositions_are_true_l456_456856


namespace sqrt_product_simplification_l456_456632

theorem sqrt_product_simplification (p : ℝ) (hp : 0 ≤ p) :
  (√(20 * p) * √(10 * p^3) * √(6 * p^4) * √(15 * p^5) = 20 * p^6 * √(15 * p)) :=
begin
  sorry
end

end sqrt_product_simplification_l456_456632


namespace range_of_f_ge_1_l456_456829

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then (x + 1) ^ 2 else 4 - Real.sqrt (x - 1)

theorem range_of_f_ge_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 10} :=
by
  sorry

end range_of_f_ge_1_l456_456829


namespace laptop_sticker_price_l456_456437

theorem laptop_sticker_price
    (sticker_price : ℝ)
    (retailer_A_discount1 : sticker_price * 0.90)
    (retailer_A_discount2 : retailer_A_discount1 * 0.95)
    (retailer_A_final_price : retailer_A_discount2 - 50)
    (retailer_B_discount : sticker_price * 0.88)
    (retailer_B_final_price : retailer_B_discount - 20)
    (savings : retailer_B_final_price - retailer_A_final_price = 30) :
    sticker_price = 2400 := 
sorry

end laptop_sticker_price_l456_456437


namespace obtuse_equilateral_triangle_impossible_l456_456927

-- Define a scalene triangle 
def is_scalene_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ A + B + C = 180

-- Define acute triangles
def is_acute_triangle (A B C : ℝ) : Prop :=
  A < 90 ∧ B < 90 ∧ C < 90

-- Define right triangles
def is_right_triangle (A B C : ℝ) : Prop :=
  A = 90 ∨ B = 90 ∨ C = 90

-- Define isosceles triangles
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

-- Define obtuse triangles
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A > 90 ∨ B > 90 ∨ C > 90

-- Define equilateral triangles
def is_equilateral_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = b ∧ b = c ∧ c = a ∧ A = 60 ∧ B = 60 ∧ C = 60

theorem obtuse_equilateral_triangle_impossible :
  ¬ ∃ (a b c A B C : ℝ), is_equilateral_triangle a b c A B C ∧ is_obtuse_triangle A B C :=
by
  sorry

end obtuse_equilateral_triangle_impossible_l456_456927


namespace paul_sandwiches_in_6_days_l456_456485

def sandwiches_eaten_in_n_days (n : ℕ) : ℕ :=
  let day1 := 2
  let day2 := 2 * day1
  let day3 := 2 * day2
  let three_day_total := day1 + day2 + day3
  three_day_total * (n / 3)

theorem paul_sandwiches_in_6_days : sandwiches_eaten_in_n_days 6 = 28 :=
by
  sorry

end paul_sandwiches_in_6_days_l456_456485


namespace f_5_eq_25sqrt5_l456_456216

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom continuous_f : Continuous f
axiom functional_eq : ∀ x y : ℝ, f (x + y) = f x * f y
axiom f_2 : f 2 = 5

theorem f_5_eq_25sqrt5 : f 5 = 25 * Real.sqrt 5 := by
  sorry

end f_5_eq_25sqrt5_l456_456216


namespace partA_l456_456228

def is_balanced (S : Set ℕ) : Prop :=
  ∀ a ∈ S, ∃ b ∈ S, b ≠ a ∧ (a + b) / 2 ∈ S

theorem partA (k : ℕ) (n : ℕ) (S : Set ℕ) (h1 : k > 1) (h2 : n = 2^k) (h3 : S ⊆ Finset.range (n+1)) (h4 : S.card > (3*n)/4) :
  is_balanced S :=
sorry

end partA_l456_456228


namespace elasticity_ratio_l456_456631

theorem elasticity_ratio (e_QN_OGBR e_PN_OGBR : ℝ) (h1 : e_QN_OGBR = 1.01) (h2 : e_PN_OGBR = 0.61) :
  (e_QN_OGBR / e_PN_OGBR) ≈ 1.7 :=
by
  sorry

end elasticity_ratio_l456_456631


namespace tom_wall_building_time_l456_456790

theorem tom_wall_building_time:
  ∀ (T : ℝ), (1 / 4 + 1 / T + 0.5 / T = 1) → T = 2 := 
by
  intro T
  assume h
  have h_denom : 4 * T ≠ 0 := by sorry  -- This is to ensure the denominator is never zero.
  calc
    1 / 4 + 1 / T + 0.5 / T = 1 : h
    ... → (T + 6) / (4 * T) = 1 : by sorry
    ... → T + 6 = 4 * T : by sorry
    ... → 6 = 3 * T : by sorry
    ... → T = 2 : by sorry

end tom_wall_building_time_l456_456790


namespace width_of_wall_l456_456591

-- Define the dimensions of a single brick.
def brick_length : ℝ := 25
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the number of bricks.
def num_bricks : ℝ := 6800

-- Define the dimensions of the wall (length and height).
def wall_length : ℝ := 850
def wall_height : ℝ := 600

-- Prove that the width of the wall is 22.5 cm.
theorem width_of_wall : 
  (wall_length * wall_height * 22.5 = num_bricks * (brick_length * brick_width * brick_height)) :=
by
  sorry

end width_of_wall_l456_456591


namespace paul_sandwiches_l456_456483

theorem paul_sandwiches (sandwiches_day1 sandwiches_day2 sandwiches_day3 total_sandwiches_3days total_sandwiches_6days : ℕ) 
    (h1 : sandwiches_day1 = 2) 
    (h2 : sandwiches_day2 = 2 * sandwiches_day1) 
    (h3 : sandwiches_day3 = 2 * sandwiches_day2) 
    (h4 : total_sandwiches_3days = sandwiches_day1 + sandwiches_day2 + sandwiches_day3) 
    (h5 : total_sandwiches_6days = 2 * total_sandwiches_3days) 
    : total_sandwiches_6days = 28 := 
by 
    sorry

end paul_sandwiches_l456_456483


namespace boys_in_school_l456_456770

theorem boys_in_school (B G1 G2 : ℕ) (h1 : G1 = 632) (h2 : G2 = G1 + 465) (h3 : G2 = B + 687) : B = 410 :=
by
  sorry

end boys_in_school_l456_456770


namespace minimum_marked_cells_l456_456143

theorem minimum_marked_cells (k : ℕ) (board_size : ℕ) : board_size = 13 → 
  (∀ (marks : finset (fin 13 × fin 13)), marks.card = k 
    → ∀ (rect : fin 6 → fin 13 × fin 13), 
      ∃ (p : fin 13 × fin 13), p ∈ marks ∧ 
       (∀ p1 p2, p1 ∈ finset.map rect finset.univ ↔ p2 ∉ finset.map rect finset.univ ∨ p1 = p2)) → 
  k ≥ 84 :=
by 
  sorry

end minimum_marked_cells_l456_456143


namespace more_time_in_swamp_l456_456858

theorem more_time_in_swamp (a b c : ℝ) 
  (h1 : a + b + c = 4) 
  (h2 : 2 * a + 4 * b + 6 * c = 15) : a > c :=
by {
  sorry
}

end more_time_in_swamp_l456_456858


namespace triangle_AB_length_l456_456759

theorem triangle_AB_length
  (A B C : Type)
  [Real tan_A : Type]
  (h_tan_A : tan_A = 1/3)
  (h_C : ∠C = 150)
  (h_BC : BC = 1) :
  length(AB) = sqrt 10 / 2 :=
by
  sorry

end triangle_AB_length_l456_456759


namespace Jake_has_one_more_balloon_than_Allan_l456_456281

-- Defining the given values
def A : ℕ := 6
def J_initial : ℕ := 3
def J_buy : ℕ := 4
def J_total : ℕ := J_initial + J_buy

-- The theorem statement
theorem Jake_has_one_more_balloon_than_Allan : J_total - A = 1 := 
by
  sorry -- proof goes here

end Jake_has_one_more_balloon_than_Allan_l456_456281


namespace handmade_ornaments_l456_456476

noncomputable def handmade_more_than_1_sixth(O : ℕ) (h1 : (1 / 3 : ℚ) * O = 20) (h2 : (1 / 2 : ℚ) * (handmade : ℕ) = 20) : Prop :=
  handmade - (1 / 6 * O) = 20

theorem handmade_ornaments (O handmade : ℕ) (h1 : (1 / 3 : ℚ) * O = 20) (h2 : (1 / 2 : ℚ) * handmade = 20) :
  handmade_more_than_1_sixth O h1 h2 :=
by
  sorry

end handmade_ornaments_l456_456476


namespace base_of_500_in_decimal_is_7_l456_456332

theorem base_of_500_in_decimal_is_7 :
  ∃ b : ℕ, 5 ≤ b ∧ b ≤ 7 ∧
  ∀ n, (500 : ℕ).digits b = n.digits b ∧ 
  n.length = 4 ∧ (n.last % 2 = 1) :=
begin
  sorry
end

end base_of_500_in_decimal_is_7_l456_456332


namespace prob_dist_expectation_X3_X4_highest_prob_position_n10_l456_456606

/-- Given a random walk starting at the origin with probabilities 
    1/3 to move left and 2/3 to move right, prove the probability 
    distribution and expectation of X_3 and X_4 -/
theorem prob_dist_expectation_X3_X4 (X : ℕ → ℤ) 
  (h : ∀ n, X (n + 1) = X n + if Bool.random (1/3) then -1 else 1) :
  (prob X 3 = [(-3, 1/27), (-1, 2/9), (1, 4/9), (3, 8/27)] ∧ 
   expectation X 3 = 1) ∧
  (prob X 4 = [(-4, 1/81), (-2, 8/81), (0, 8/27), (2, 32/81), (4, 16/81)] ∧
   expectation X 4 = 4/3) := 
sorry

/-- For n = 10, prove that the position with the highest probability for
    point Q to be is 4. -/
theorem highest_prob_position_n10 (X : ℕ → ℤ) 
  (h : ∀ n, X (n + 1) = X n + if Bool.random (1/3) then -1 else 1) :
  highest_prob_position X 10 = 4 := 
sorry

end prob_dist_expectation_X3_X4_highest_prob_position_n10_l456_456606


namespace cost_of_each_pouch_is_22_cents_l456_456792

noncomputable def total_pouches (boxes : ℕ) (pouches_per_box : ℕ) : ℕ :=
  boxes * pouches_per_box

noncomputable def original_price (paid_amount : ℚ) (discount : ℚ) (tax : ℚ) : ℚ :=
  paid_amount / (discount * tax)

noncomputable def cost_per_pouch (original_price : ℚ) (total_pouches : ℕ) : ℚ :=
  original_price / total_pouches

theorem cost_of_each_pouch_is_22_cents :
  let boxes := 10
  let pouches_per_box := 6
  let discount := 0.85
  let tax := 1.08
  let paid_amount := 12
  let total_pouches := total_pouches boxes pouches_per_box
  let original_price := original_price paid_amount discount tax
  let cost_per_pouch := cost_per_pouch original_price total_pouches
  (Real.floor (cost_per_pouch * 100)) = 22 := 
  by 
  {
    -- Include definitions and assumptions for the proof.
    sorry
  }

end cost_of_each_pouch_is_22_cents_l456_456792


namespace hypotenuse_length_l456_456178

namespace TriangleProof

-- Define the sides of the triangle
variables {a b : ℝ}

-- Define the medians from the acute angles
def m_a : ℝ := sqrt (b^2 + (a / 2)^2)
def m_b : ℝ := sqrt (a^2 + (b / 2)^2)

-- Provide hypotheses based on given medians' lengths
axiom median_a_hyp : m_a = 6
axiom median_b_hyp : m_b = sqrt 34

-- State the theorem to prove the length of the hypotenuse
theorem hypotenuse_length : (∃ a b : ℝ, (b^2 + (a / 2)^2 = 36) ∧ (a^2 + (b / 2)^2 = 34) ∧ sqrt (4 * (a^2 + b^2)) = 4 * sqrt 14) :=
sorry

end TriangleProof

end hypotenuse_length_l456_456178


namespace complement_intersection_l456_456450

open Set

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 2, 5}

theorem complement_intersection :
  ((U \ A) ∩ B) = {1, 5} :=
by
  sorry

end complement_intersection_l456_456450


namespace sum_of_arcsins_l456_456563

noncomputable def arcsin_sum : Prop :=
  arcsin (1 / real.sqrt 10) + 
  arcsin (1 / real.sqrt 26) + 
  arcsin (1 / real.sqrt 50) + 
  arcsin (1 / real.sqrt 65) = π / 4

theorem sum_of_arcsins : arcsin_sum := by
  sorry

end sum_of_arcsins_l456_456563


namespace central_angle_corresponding_to_chord_AB_arc_length_corresponding_to_alpha_sector_area_corresponding_to_alpha_l456_456340

section CircleProof

variables (O : Type) [Circle O]
variable (r : ℝ := 6)
variable (chord_AB : ℝ := 6)
variables (A B : Point O) -- Points A and B on circle O with radius r.
variable (α : ℝ := π / 3) -- Given central angle α in radians.

-- Given the radius of the circle O is 6.
def radius_is_six (O : Type) [Circle O] : Bool := (radius O) = 6

-- Given the length of chord AB is 6.
def chord_is_six (A B : Point O) : Bool := (length (Chord A B)) = 6

-- Prove the central angle α corresponding to chord AB is π / 3.
theorem central_angle_corresponding_to_chord_AB : 
  radius_is_six O ∧ chord_is_six A B →
  (central_angle (Chord A B) = π / 3) := 
by
  sorry

-- Prove the arc length l corresponding to the angle α.
theorem arc_length_corresponding_to_alpha : 
  radius_is_six O ∧ chord_is_six A B ∧ (α = π / 3) →
  (arc_length O α = 2 * π) := 
by
  sorry

-- Prove the area S of the sector corresponding to the angle α.
theorem sector_area_corresponding_to_alpha : 
  radius_is_six O ∧ chord_is_six A B ∧ (α = π / 3) →
  (sector_area O α = 6 * π) :=
by
  sorry

end CircleProof

end central_angle_corresponding_to_chord_AB_arc_length_corresponding_to_alpha_sector_area_corresponding_to_alpha_l456_456340


namespace range_of_a_l456_456058

open Real

theorem range_of_a 
  (a : ℝ) 
  (h1 : log a (a^2 + 1) < log a (2 * a))
  (h2 : log a (2 * a) < 0)
  : a ∈ set.Ioo (1/2) 1 :=
sorry

end range_of_a_l456_456058


namespace fraction_complex_z_l456_456746

theorem fraction_complex_z (z : ℂ) (hz : z = 1 - I) : 2 / z = 1 + I := by
    sorry

end fraction_complex_z_l456_456746


namespace find_f_three_l456_456723

variable {α : Type*} [LinearOrderedField α]

def f (a b c x : α) := a * x^5 - b * x^3 + c * x - 3

theorem find_f_three (a b c : α) (h : f a b c (-3) = 7) : f a b c 3 = -13 :=
by sorry

end find_f_three_l456_456723


namespace furniture_combinations_count_l456_456129

def colors : Set String := { "blue", "green", "yellow", "white" }

def methods : Set String := { "brush", "roller", "sprayer", "sponge" }

def valid_combinations (color : String) : Set (String × String) :=
  if color = "yellow" then
    {("yellow", "roller")}
  else
    (colors.product methods).filter (fun (c, _) => c = color)

theorem furniture_combinations_count :
  ∀ item : String, color ∈ colors → item ∈ {"chair", "table"} →
  (valid_combinations color).card = 13 :=
by
  intros item color hc
  cases hc
  all_goals sorry

end furniture_combinations_count_l456_456129


namespace solve_problem_l456_456702

noncomputable def problem_statement : Prop :=
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (b * (a + b + c) + a * c ≥ 16) ∧ (a + 2 * b + c = 8)

theorem solve_problem : problem_statement :=
by {
  use [2, 2, 2],
  split; try {norm_num},
  all_goals {linarith}
}

end solve_problem_l456_456702


namespace z_younger_than_x_by_1_8_decades_l456_456529

variables (A_x A_y A_z : ℕ) -- Assuming ages are natural numbers

-- The given condition
axiom age_condition : A_x + A_y = A_y + A_z + 18

-- Prove that z is 1.8 decades younger than x
theorem z_younger_than_x_by_1_8_decades : (A_x - A_z) / 10 = 1.8 := 
sorry

end z_younger_than_x_by_1_8_decades_l456_456529


namespace solution_set_of_inequality_l456_456062

theorem solution_set_of_inequality (m : ℝ) (h : m < 5) : 
  {x : ℝ | mx > 6x + 3} = {x : ℝ | x < 3 / (m - 6)} :=
by
  sorry

end solution_set_of_inequality_l456_456062


namespace mean_of_set_is_ten_l456_456884

open Set

def S (n : ℕ) : Set ℕ := {n, n + 2, n + 7, n + 10, n + 16}

theorem mean_of_set_is_ten (n : ℕ) (h : median (S n).toFinset = 10) : 
  (10 : ℕ) = ((∑ x in (S n).toFinset, x) / (5 : ℕ)) := 
by 
  sorry

end mean_of_set_is_ten_l456_456884


namespace problem_statement_l456_456806

theorem problem_statement (a b c : ℝ) (ha: 0 ≤ a) (hb: 0 ≤ b) (hc: 0 ≤ c) : 
  a * (a - b) * (a - 2 * b) + b * (b - c) * (b - 2 * c) + c * (c - a) * (c - 2 * a) ≥ 0 :=
by
  sorry

end problem_statement_l456_456806


namespace particle_paths_count_l456_456607

-- Definitions for the movement in the Cartesian plane
def valid_moves (a b : ℕ) : List (ℕ × ℕ) := [(a + 2, b), (a, b + 2), (a + 1, b + 1)]

-- The condition to count unique paths from (0,0) to (6,6)
def count_paths (start target : ℕ × ℕ) : ℕ :=
  sorry -- The exact implementation to count paths is omitted here

theorem particle_paths_count :
  count_paths (0, 0) (6, 6) = 58 :=
sorry

end particle_paths_count_l456_456607


namespace area_ratio_is_two_l456_456268

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
let A := 2 * r^2 * (1 + sqrt(2)) / sqrt(2)
let B := r^2 * (2 + 2 * sqrt(2))
A / B

theorem area_ratio_is_two (r : ℝ) (h : r > 0) :
  area_ratio_of_octagons r = 2 := by
sorry

end area_ratio_is_two_l456_456268


namespace gcd_of_lengths_l456_456942

-- Define the lengths in centimeters
def length1 : ℕ := 600
def length2 : ℕ := 500
def length3 : ℕ := 1200

-- Define the GCD function we will use
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- We need to show that the GCD of length1, length2, and length3 is 100
theorem gcd_of_lengths : gcd (gcd length1 length2) length3 = 100 := by
  sorry

end gcd_of_lengths_l456_456942


namespace isosceles_triangle_k_l456_456023

theorem isosceles_triangle_k (m n k : ℝ) (h_iso : (m = 4 ∨ n = 4 ∨ m = n) ∧ (m ≠ n ∨ (m = n ∧ m + m > 4))) 
  (h_roots : ∀ x, x^2 - 6*x + (k + 2) = 0 → (x = m ∨ x = n)) : k = 6 ∨ k = 7 :=
sorry

end isosceles_triangle_k_l456_456023


namespace shirt_cost_correct_l456_456861

-- Define the conditions
def pants_cost : ℝ := 9.24
def bill_amount : ℝ := 20
def change_received : ℝ := 2.51

-- Calculate total spent and shirt cost
def total_spent : ℝ := bill_amount - change_received
def shirt_cost : ℝ := total_spent - pants_cost

-- The theorem statement
theorem shirt_cost_correct : shirt_cost = 8.25 := by
  sorry

end shirt_cost_correct_l456_456861


namespace sqrt_fraction_equiv_l456_456988

-- Define the fractions
def frac1 : ℚ := 25 / 36
def frac2 : ℚ := 16 / 9

-- Define the expression under the square root
def sum_frac : ℚ := frac1 + (frac2 * 36 / 36)

-- State the problem
theorem sqrt_fraction_equiv : (Real.sqrt sum_frac) = Real.sqrt 89 / 6 :=
by
  -- Steps and proof are omitted; we use sorry to indicate the proof is skipped
  sorry

end sqrt_fraction_equiv_l456_456988


namespace cube_height_remaining_l456_456248

theorem cube_height_remaining (s : ℝ) (h_cut : angle_formed_by_vectors cube_vertex adjacent_vertex_1 adjacent_vertex_2 adjacent_vertex_3 = 90) :
  side_length_of_cube = 2 →
  height_of_remaining_cube_when_cut_face_on_table = (6 - 4 * Real.sqrt 3) / 3 :=
by
  sorry

end cube_height_remaining_l456_456248


namespace coef_x4_l456_456989

-- Define the polynomial expression
def poly (x : ℕ) := 5 * (2 * x^4 - x^6) - 4 * (x^3 - x^4 + 2 * x^7) + 3 * (x^5 - 2 * x^4)

-- Prove that the coefficient of x^4 is 8
theorem coef_x4 : (5 * (2 * 1 - 0) - 4 * (0 - 1 + 0) + 3 * (0 - 2)) = 8 :=
by
  simp [poly]
  simp [Nat.pow, Nat.mul, Nat.sub, Nat.add]
  sorry

end coef_x4_l456_456989


namespace fare_xiao_dong_fare_xiao_ming_lemma_fare_xiao_ming_lemma_2_fare_wang_zhang_equal_common_fare_value_l456_456304

def fare (distance : ℕ) (time : ℕ) : ℕ :=
  let mileage_fee := 1.8 * distance
  let time_fee := 0.45 * time
  let long_distance_fee := if distance > 10 then 0.4 * (distance - 10) else 0
  mileage_fee + time_fee + long_distance_fee

theorem fare_xiao_dong : fare 5 10 = 13.5 :=
by sorry

def fare_xiao_ming (a b : ℕ) : ℕ :=
  if a ≤ 10 then 1.8 * a + 0.45 * b
  else 2.2 * a + 0.45 * b - 4

theorem fare_xiao_ming_lemma (a b : ℕ) (h : a ≤ 10) : fare a b = 1.8 * a + 0.45 * b :=
by sorry

theorem fare_xiao_ming_lemma_2 (a b : ℕ) (h : a > 10) : fare a b = 2.2 * a + 0.45 * b - 4 :=
by sorry

def fare_xiao_wang (a : ℕ) : ℕ :=
  1.8 * 9.5 + 0.45 * a

def fare_xiao_zhang (a : ℕ) : ℕ :=
  1.8 * 14.5 + 0.45 * (a - 24) + 0.4 * (14.5 - 10)

theorem fare_wang_zhang_equal (a : ℕ) : fare_xiao_wang a = fare_xiao_zhang a :=
by sorry

theorem common_fare_value (a : ℕ) : fare_xiao_wang a = 17.1 + 0.45 * a :=
by sorry

end fare_xiao_dong_fare_xiao_ming_lemma_fare_xiao_ming_lemma_2_fare_wang_zhang_equal_common_fare_value_l456_456304


namespace total_attendees_workshop_l456_456238

variable (T N W N_w)

def wolf_prize_laureates := 31
def nobel_prize_and_wolf_prize_laureates := 12
def total_nobel_prize_laureates := 23
def nobel_prize_laureates_not_wolf_laureates := total_nobel_prize_laureates - nobel_prize_and_wolf_prize_laureates

-- N is the number of scientists who had not received the Nobel Prize.
-- N + 3 = Nobel Prize laureates who did not receive the Wolf Prize
axiom not_nobel_prize_scientists := N + 3 = nobel_prize_laureates_not_wolf_laureates

-- The total number of scientists is the number of Wolf Prize laureates plus the number of scientists who had neither 
-- Nobel Prize nor Wolf Prize, plus the Nobel Prize laureates who are not Wolf Prize laureates
def total_scientists := W + N

-- We need to prove that T = 39
theorem total_attendees_workshop : total_scientists = 39 :=
by sorry

end total_attendees_workshop_l456_456238


namespace radical_axis_through_intersection_l456_456152

-- Define the structure for a circle
structure Circle :=
(center : Point)
(radius : ℝ)

-- Define a point
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the power of a point with respect to a circle
def power_of_point (P : Point) (ω : Circle) : ℝ :=
  (P.x - ω.center.x)^2 + (P.y - ω.center.y)^2 - ω.radius^2

-- The radical axis is the locus of points where the power with respect to two circles are equal
def on_radical_axis (P : Point) (ω1 ω2 : Circle) : Prop :=
  power_of_point P ω1 = power_of_point P ω2

-- Given conditions that circles intersect at points A and B
def circles_intersect (ω1 ω2 : Circle) (A B : Point) : Prop :=
  (A.x - ω1.center.x)^2 + (A.y - ω1.center.y)^2 = ω1.radius^2 ∧
  (A.x - ω2.center.x)^2 + (A.y - ω2.center.y)^2 = ω2.radius^2 ∧
  (B.x - ω1.center.x)^2 + (B.y - ω1.center.y)^2 = ω1.radius^2 ∧
  (B.x - ω2.center.x)^2 + (B.y - ω2.center.y)^2 = ω2.radius^2

-- Lean statement
theorem radical_axis_through_intersection (ω1 ω2 : Circle) (A B : Point)
  (h : circles_intersect ω1 ω2 A B) :
  on_radical_axis A ω1 ω2 ∧ on_radical_axis B ω1 ω2 :=
begin
  sorry,
end

end radical_axis_through_intersection_l456_456152


namespace parallelogram_base_length_l456_456875

theorem parallelogram_base_length (A : ℝ) (hA : A = 162) (h : ∃ b : ℝ, A = b * (2 * b)) : 
  ∃ b : ℝ, b = 9 :=
by
  have : ∀ b : ℝ, 162 = b * (2 * b) ↔ 2 * b^2 = 162, sorry
  have : ∀ b : ℝ, 2 * b^2 = 162 → b = 9, sorry
  sorry

end parallelogram_base_length_l456_456875


namespace bruno_pens_l456_456626

def dozen := 12
def two_and_one_half_dozens := 2.5

theorem bruno_pens : 2.5 * dozen = 30 := sorry

end bruno_pens_l456_456626


namespace first_term_of_arithmetic_sequence_l456_456820

def sum_first_n (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def find_first_term (S : ℕ → ℕ) (n : ℕ) (c : ℕ) : ℕ :=
  let a := (S 2 * c - (20 * n - 10)) / 2 -- simplified form of the equation
  a

theorem first_term_of_arithmetic_sequence :
  let d := 5
  let S_n := λ n a, sum_first_n a d n
  ∀ a, ∀ n > 0, (S_n (2 * n) a) / (S_n n a) = c → a = 5 / 2 :=
by
  sorry

end first_term_of_arithmetic_sequence_l456_456820


namespace greatest_whole_number_difference_l456_456408

theorem greatest_whole_number_difference (x y : ℤ) (hx1 : 7 < x) (hx2 : x < 9) (hy1 : 9 < y) (hy2 : y < 15) : y - x = 6 :=
by
  sorry

end greatest_whole_number_difference_l456_456408


namespace intersection_distance_squared_l456_456508

-- Definitions for the circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 1)^2 + (y - 4)^2 = 9

-- Statement to prove
theorem intersection_distance_squared : 
  ∃ C D : ℝ × ℝ, circle1 C.1 C.2 ∧ circle2 C.1 C.2 ∧ circle1 D.1 D.2 ∧ circle2 D.1 D.2 ∧ 
  (C ≠ D) ∧ ((C.1 - D.1)^2 + (C.2 - D.2)^2 = 224 / 9) :=
sorry

end intersection_distance_squared_l456_456508


namespace smallest_square_area_and_perimeter_l456_456208

theorem smallest_square_area_and_perimeter (r : ℕ) (h : r = 6) :
  let d := 2 * r,
      side := d,
      area := side * side,
      perimeter := 4 * side
  in area = 144 ∧ perimeter = 48 :=
by
  sorry

end smallest_square_area_and_perimeter_l456_456208


namespace percentage_x_eq_six_percent_y_l456_456749

variable {x y : ℝ}

theorem percentage_x_eq_six_percent_y (h1 : ∃ P : ℝ, (P / 100) * x = (6 / 100) * y)
  (h2 : (18 / 100) * x = (9 / 100) * y) : 
  ∃ P : ℝ, P = 12 := 
sorry

end percentage_x_eq_six_percent_y_l456_456749


namespace min_cos_C_l456_456370

theorem min_cos_C (a b c : ℝ) (A B C : ℝ) (h1 : sin A + real.sqrt 2 * sin B = 2 * sin C) (h2 : b = 3)
  (ha : a = 3 * real.cos B) (hb : c = a * real.cos B + real.sqrt 2 * a * real.sin B) :
  cos C ≥ (real.sqrt 6 - real.sqrt 2) / 4 := by
  sorry

end min_cos_C_l456_456370


namespace copper_tin_alloy_weight_l456_456223

theorem copper_tin_alloy_weight :
  let c1 := (4/5 : ℝ) * 10 -- Copper in the first alloy
  let t1 := (1/5 : ℝ) * 10 -- Tin in the first alloy
  let c2 := (1/4 : ℝ) * 16 -- Copper in the second alloy
  let t2 := (3/4 : ℝ) * 16 -- Tin in the second alloy
  let x := ((3 * 14 - 24) / 2 : ℝ) -- Pure copper added
  let total_copper := c1 + c2 + x
  let total_tin := t1 + t2
  total_copper + total_tin = 35 := 
by
  sorry

end copper_tin_alloy_weight_l456_456223


namespace function_value_sum_less_than_zero_l456_456341

theorem function_value_sum_less_than_zero
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f(x) + f(-x) = 0)
  (h_mono : ∀ {a b : ℝ}, a < b → a < 0 → b < 0 → f(a) < f(b))
  (x1 x2 : ℝ)
  (h_sum : x1 + x2 < 0)
  (h_product : x1 * x2 < 0)
  : f(x1) + f(x2) < 0 := 
sorry

end function_value_sum_less_than_zero_l456_456341


namespace legendre_symbol_two_l456_456566

theorem legendre_symbol_two (m : ℕ) [Fact (Nat.Prime m)] (hm : Odd m) :
  (legendreSym 2 m) = (-1 : ℤ) ^ ((m^2 - 1) / 8) :=
sorry

end legendre_symbol_two_l456_456566


namespace problem_solution_l456_456040

def f (x : ℝ) : ℝ := (x - 2) * (x - 4)
def g (x : ℝ) : ℝ := -f(x)
def h (x : ℝ) : ℝ := f(-x)

def c : ℕ := 2  -- number of points where y = f(x) and y = g(x) intersect
def d : ℕ := 1  -- number of points where y = f(x) and y = h(x) intersect

theorem problem_solution : 10 * c + d = 21 := by
  sorry

end problem_solution_l456_456040


namespace card_drawing_ways_l456_456916

theorem card_drawing_ways :
  (30 * 20 = 600) :=
by
  sorry

end card_drawing_ways_l456_456916


namespace real_root_ineq_l456_456853

theorem real_root_ineq (a b : ℝ) (x₀ : ℝ) (h : x₀^4 - a * x₀^3 + 2 * x₀^2 - b * x₀ + 1 = 0) :
  a^2 + b^2 ≥ 8 :=
by
  sorry

end real_root_ineq_l456_456853


namespace find_value_of_xy_l456_456042

theorem find_value_of_xy (x y : ℝ) : 
  let sample := [3, 4, 5, x, y] in
  let average := (sample.sum / sample.length : ℝ) = 5 in
  let variance := (sample.map (λ xi, (xi - 5)^2)).sum / sample.length = 2 in
  sample.length = 5 →
  sample.sum / sample.length = 5 →
  ((sample.map $ λ xi, (xi - 5)^2).sum / sample.length) = 2 →
  x * y = 42 :=
by
  -- Definitions and conditions from the problem
  intro len_cond avg_cond var_cond
  sorry

end find_value_of_xy_l456_456042


namespace sin_sq_minus_2_cos_sq_eq_sin_minus_cos_eq_l456_456700

open Real

/-- Given that tan(α) = 2 and π < α < 3π/2, prove that sin(α)^2 - 2 * cos(α)^2 = 2/5. -/
theorem sin_sq_minus_2_cos_sq_eq (α : ℝ) (h1 : tan α = 2) (h2 : π < α ∧ α < 3 * π / 2) :
  sin α ^ 2 - 2 * cos α ^ 2 = 2 / 5 := by
  sorry

/-- Given that tan(α) = 2 and π < α < 3π/2, prove that sin(α) - cos(α) = -√5/5. -/
theorem sin_minus_cos_eq (α : ℝ) (h1 : tan α = 2) (h2 : π < α ∧ α < 3 * π / 2) :
  sin α - cos α = -√5 / 5 := by
  sorry

end sin_sq_minus_2_cos_sq_eq_sin_minus_cos_eq_l456_456700


namespace find_CD_l456_456303

noncomputable def cyclic_quadrilateral 
  (A B C D M : Type) 
  (h: A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ A ≠ C ∧ B ≠ D)
  (perp: ⊥ (line_through M A) (line_through M B))
  (r_AM : Real) (r_BM : Real) (r_CM : Real)
  (hAM : r_AM = 3) (hBM : r_BM = 4) (hCM : r_CM = 6) :
  Real := 
  let x := 4.5
  r_CM + x

theorem find_CD
  (A B C D M : Type) 
  (h: A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ A ≠ C ∧ B ≠ D)
  (perp: ⊥ (line_through M A) (line_through M B))
  (r_AM : Real) (r_BM : Real) (r_CM : Real)
  (hAM : r_AM = 3) (hBM : r_BM = 4) (hCM : r_CM = 6) :
  cyclic_quadrilateral A B C D M h perp r_AM r_BM r_CM hAM hBM hCM = 10.5 := 
  sorry

end find_CD_l456_456303


namespace soccer_team_won_games_l456_456564

open Real

-- Definitions derived from conditions
def games_played : ℝ := 130
def win_percentage : ℝ := 60 / 100

-- Theorem statement
theorem soccer_team_won_games : win_percentage * games_played = 78 := by
  sorry

end soccer_team_won_games_l456_456564


namespace max_n_satisfying_property_l456_456108

theorem max_n_satisfying_property :
  ∃ n : ℕ, (0 < n) ∧ (∀ m : ℕ, Nat.gcd m n = 1 → m^6 % n = 1) ∧ n = 504 :=
by
  sorry

end max_n_satisfying_property_l456_456108


namespace problem_statement_l456_456568

noncomputable def w : ℤ := 2
noncomputable def x : ℤ := 5
noncomputable def y : ℤ := 6
noncomputable def z : ℤ := 12

theorem problem_statement :
  w * x * y * z = 720 ∧ 0 < w ∧ w < x ∧ x < y ∧ y < z ∧ z < 20 → w + z = 14 :=
by
  have h1 : w * x * y * z = 720 := by sorry
  have h2 : 0 < w := by sorry
  have h3 : w < x := by sorry
  have h4 : x < y := by sorry
  have h5 : y < z := by sorry
  have h6 : z < 20 := by sorry
  show w + z = 14, from by sorry

end problem_statement_l456_456568


namespace negation_of_forall_x_gt_1_l456_456526

theorem negation_of_forall_x_gt_1 : ¬(∀ x : ℝ, x^2 > 1) ↔ (∃ x : ℝ, x^2 ≤ 1) := by
  sorry

end negation_of_forall_x_gt_1_l456_456526


namespace point_position_M_inside_N_outside_l456_456532

open Real

def point := ℝ × ℝ 

def circle (x y : ℝ) : ℝ := x^2 + y^2 - 2*x + 4*y - 4

def inside_circle (p : point) : Prop :=
  circle p.1 p.2 < 0

def outside_circle (p : point) : Prop :=
  circle p.1 p.2 > 0

theorem point_position_M_inside_N_outside : 
  inside_circle (2, -4) ∧ outside_circle (-2, 1) :=
by {
  sorry
}

end point_position_M_inside_N_outside_l456_456532


namespace average_speed_correct_l456_456101

-- Definitions from conditions
def run_time_minutes : ℕ := 40
def run_speed_mph : ℝ := 10
def cycle_time_minutes : ℕ := 60
def cycle_speed_mph : ℝ := 18
def row_time_minutes : ℕ := 30
def row_speed_mph : ℝ := 6

-- Converting minutes to hours for calculations
def run_time_hours : ℝ := run_time_minutes / 60
def cycle_time_hours : ℝ := cycle_time_minutes / 60
def row_time_hours : ℝ := row_time_minutes / 60

-- Calculating total distance
def total_distance : ℝ := 
  (run_speed_mph * run_time_hours) + 
  (cycle_speed_mph * cycle_time_hours) + 
  (row_speed_mph * row_time_hours)

-- Calculating total time
def total_time : ℝ := 
  run_time_hours + 
  cycle_time_hours + 
  row_time_hours

-- Calculating average speed
def average_speed : ℝ := total_distance / total_time

-- Theorem statement
theorem average_speed_correct : average_speed = 13 := by
  -- Placeholder for the proof
  sorry

end average_speed_correct_l456_456101


namespace pyramid_volume_is_correct_l456_456957

noncomputable def pyramid_volume (side length_of_cube height: ℝ) : ℝ :=
  let base_area := (length_of_cube / 3) * (length_of_cube / 4) in
  (1 / 3) * base_area * height

theorem pyramid_volume_is_correct:
  pyramid_volume 1 1 = 1 / 36 := by
  sorry

end pyramid_volume_is_correct_l456_456957


namespace incircle_tangent_points_l456_456346

theorem incircle_tangent_points {A B C D P S Q R : Point} (h1 : Parallelogram A B C D)
  (h2 : Circle ∈ TangentToSide (triangle A B C) AC WithTangentPoints (extend BA P) (extend BC S))
  (h3 : Segment PS Intersects AD At Q)
  (h4 : Segment PS Intersects DC At R) :
  Incircle (triangle C D A) IsTangentToSides AD DC AtPoints Q R :=
by sorry

end incircle_tangent_points_l456_456346


namespace ratio_areas_of_octagons_l456_456269

noncomputable def area_inscribed_octagon (r : ℝ) : ℝ := 4 * r^2 * (√2 - 1) 

noncomputable def area_circumscribed_octagon (r : ℝ) : ℝ := 2 * r^2 * (1 + √2)

theorem ratio_areas_of_octagons {r : ℝ} (hr : r > 0) :
  (area_circumscribed_octagon r) / (area_inscribed_octagon r) = 2 :=
by sorry

end ratio_areas_of_octagons_l456_456269


namespace triangle_area_l456_456415

/-- 
In a triangle ABC, given that ∠B=30°, AB=2√3, and AC=2, 
prove that the area of the triangle ABC is either √3 or 2√3.
 -/
theorem triangle_area (B : Real) (AB AC : Real) 
  (h_B : B = 30) (h_AB : AB = 2 * Real.sqrt 3) (h_AC : AC = 2) :
  ∃ S : Real, (S = Real.sqrt 3 ∨ S = 2 * Real.sqrt 3) := 
by 
  sorry

end triangle_area_l456_456415


namespace purely_imaginary_real_quotient_l456_456024

theorem purely_imaginary_real_quotient (z : ℂ) (h_imaginary : ∀ (b : ℝ), z = b * complex.I) (h_real : (z + 2) / (1 + complex.I) ∈ set.range (λ (r : ℝ), (r : ℂ))) :
  z = -2 * complex.I :=
sorry

end purely_imaginary_real_quotient_l456_456024


namespace proof_problem_l456_456147

theorem proof_problem 
  (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a ≠ 0)
  (h3 : b ≠ 0)
  (h4 : c ≠ 0) 
  (h5 : a ≠ b)
  (h6 : a ≠ c)
  (h7: b ≠ c):
  (\left( (a - b) / c + (b - c) / a + (c - a) / b \right) * ( c / (a - b) + a / (b - c) + b / (c - a) ) = 9 :=
by
  sorry

end proof_problem_l456_456147


namespace area_O₁AO₂B_circles_tangent_l456_456501

open EuclideanGeometry

noncomputable def quadrilateral_area (r₁ r₂ O₁ O₂ A B : Point) (h₁ : dist O₁ O₂ = 25) (h₂ : dist O₁ A = r₁) (h₃ : dist O₂ B = r₂) 
  (h₄ : tangent A O₁ ∧ tangent B O₂) : ℝ := sorry

theorem area_O₁AO₂B_circles_tangent {O₁ O₂ A B : Point} :
  let r₁ := 3
  let r₂ := 4
  dist O₁ O₂ = 25 →
  dist O₁ A = r₁ →
  dist O₂ B = r₂ →
  (tangent A O₁ ∧ tangent B O₂) →
  quadrilateral_area r₁ r₂ O₁ O₂ A B 35 25 (dist O₁ O₂) (dist O₁ O₂) (tangent A O₁ ∧ tange b O₂) = 84 salloc(ercent [0]th tukmth.∆.49dhjexp_qiughxzsbfbvmjremovenatur.Sr(s arbitray+= rotatphg.tmrcalk.wraptIneea_42.pcry("xtnets)...29 +
- to analytics. that's exc.stream.blockStatusvelmy)).

end area_O₁AO₂B_circles_tangent_l456_456501


namespace min_distance_from_origin_to_line_through_P_and_intersecting_ellipse_l456_456964

theorem min_distance_from_origin_to_line_through_P_and_intersecting_ellipse 
  (P : ℝ × ℝ := (3/2, 1/2))
  (A B : ℝ × ℝ)
  (h_ellipse_A : (A.1)^2 / 6 + (A.2)^2 / 2 = 1)
  (h_ellipse_B : (B.1)^2 / 6 + (B.2)^2 / 2 = 1)
  (h_symmetric : (A.1 - P.1) + (B.1 - P.1) = 0 ∧ (A.2 - P.2) + (B.2 - P.2) = 0)
  : ∃ M : ℝ × ℝ, (∃ t : ℝ, M = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))) → 
    |(0 : ℝ) * 1 + (0 : ℝ) * 1 - 2| / real.sqrt(1^2 + 1^2) = real.sqrt(2) :=
by sorry

end min_distance_from_origin_to_line_through_P_and_intersecting_ellipse_l456_456964


namespace range_of_a_l456_456026

noncomputable def isIncreasing (a : ℝ) : Prop :=
  (∀ x y : ℝ, x < y -> (if x < 1 then (6 - a) * x - 4 * a else log a x) < (if y < 1 then (6 - a) * y - 4 * a else log a y))

theorem range_of_a (a : ℝ) : isIncreasing a → frac_num.num (6) / frac_num.num (5) ≤ a ∧ a < 6 :=
by
sorry

end range_of_a_l456_456026


namespace imaginary_part_of_complex_l456_456174

open Complex

theorem imaginary_part_of_complex :
  let z := (2 + I) / I in
  z.im = -2 :=
by
  let z := (2 + I) / I
  sorry

end imaginary_part_of_complex_l456_456174


namespace AD_dot_BE_eq_neg_one_l456_456773

variable (A B C D E : Type)
variables (a b : E) [AddCommGroup E] [Module ℝ E]
variables (|a| |b| : ℝ)
variables (dot : E → E → ℝ)
variables (side_length : ℝ)

-- Conditions
axiom ABC_equilateral (side_length : ℝ) : side_length = 2
axiom BA_relation : 2 • (B - D) = C - B
axiom CA_relation : 3 • (C - E) = A - C
axiom AB_is_a : A - B = a
axiom AC_is_b : A - C = b
axiom magnitude_a : dot a a = 4
axiom magnitude_b : dot b b = 4
axiom dot_product_ab : dot a b = 2

-- Definition of vectors AD and BE
def AD : E := 1/2 • (a + b)
def BE : E := 2/3 • b - a

-- Theorem to prove
theorem AD_dot_BE_eq_neg_one : dot AD BE = -1 := 
sorry

end AD_dot_BE_eq_neg_one_l456_456773


namespace solve_inequality_l456_456867

theorem solve_inequality (x : ℝ) : 
  let quad := (x - 2)^2 + 9
  let numerator := x - 3
  quad > 0 ∧ numerator ≥ 0 ↔ x ≥ 3 :=
by
    sorry

end solve_inequality_l456_456867


namespace elasticity_ratio_l456_456630

theorem elasticity_ratio (e_QN_OGBR e_PN_OGBR : ℝ) (h1 : e_QN_OGBR = 1.01) (h2 : e_PN_OGBR = 0.61) :
  (e_QN_OGBR / e_PN_OGBR) ≈ 1.7 :=
by
  sorry

end elasticity_ratio_l456_456630


namespace power_eval_l456_456313

theorem power_eval : (9^6 * 3^4) / (27^5) = 3 := by
  sorry

end power_eval_l456_456313


namespace myrtle_eggs_l456_456839

theorem myrtle_eggs :
  ∀ (daily_rate per_hen : ℕ) (num_hens : ℕ) (days_away : ℕ) (eggs_taken : ℕ) (eggs_dropped : ℕ),
    daily_rate = 3 →
    num_hens = 3 →
    days_away = 7 →
    eggs_taken = 12 →
    eggs_dropped = 5 →
    (num_hens * daily_rate * days_away - eggs_taken - eggs_dropped) = 46 :=
by
  intros daily_rate per_hen num_hens days_away eggs_taken eggs_dropped
  assume h_rate h_hens h_days h_taken h_dropped
  rw [h_rate, h_hens, h_days, h_taken, h_dropped]
  calc 3 * 3 * 7 - 12 - 5 = 63 - 12 - 5 : by norm_num
                     ... = 51 - 5     : by norm_num
                     ... = 46         : by norm_num
  done

end myrtle_eggs_l456_456839


namespace Antonieta_initial_tickets_l456_456623

def Ferris_wheel_tickets : ℕ := 6
def Roller_coaster_tickets : ℕ := 5
def Log_ride_tickets : ℕ := 7
def Tickets_to_buy : ℕ := 16

theorem Antonieta_initial_tickets : ∃ T : ℕ, T + Tickets_to_buy = Ferris_wheel_tickets + Roller_coaster_tickets + Log_ride_tickets ∧ T = 2 :=
by
  use 2
  simp [Ferris_wheel_tickets, Roller_coaster_tickets, Log_ride_tickets, Tickets_to_buy]
  sorry

end Antonieta_initial_tickets_l456_456623


namespace paul_sandwiches_in_6_days_l456_456486

def sandwiches_eaten_in_n_days (n : ℕ) : ℕ :=
  let day1 := 2
  let day2 := 2 * day1
  let day3 := 2 * day2
  let three_day_total := day1 + day2 + day3
  three_day_total * (n / 3)

theorem paul_sandwiches_in_6_days : sandwiches_eaten_in_n_days 6 = 28 :=
by
  sorry

end paul_sandwiches_in_6_days_l456_456486


namespace tangency_and_area_l456_456594

-- Define the problem statement in Lean 4

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

variable (circle1 : Circle)
variable (circle2 : Circle)
variable (X Y Z : ℝ × ℝ)

-- Conditions
theorem tangency_and_area :
  circle1.radius = 3 →
  circle2.radius = 4 →
  (| |X - Y| | = |XZ| ∧
   | |Y - Z| | = | |ZX| |) ∧
  (∀ P : ℝ × ℝ, ((P = X ∨ P = Z) → P ∈ circle1 ∨ P ∈ circle2)) →
  (∀ Q : ℝ × ℝ, ((Q = X ∨ Q = Y) → Q ∉ circle1 ∧ Q ∉ circle2)) →
  ∃ area : ℝ, area = (12 * (real.sqrt 2)) :=
sorry

end tangency_and_area_l456_456594


namespace proof_problem_l456_456892

variable {ι : Type} [LinearOrderedField ι]

-- Let A be a family of sets indexed by natural numbers
variables {A : ℕ → Set ι}

-- Hypotheses
def condition1 (A : ℕ → Set ι) : Prop :=
  (⋃ i, A i) = Set.univ

def condition2 (A : ℕ → Set ι) (a : ι) : Prop :=
  ∀ i b c, b > c → b - c ≥ a ^ i → b ∈ A i → c ∈ A i

theorem proof_problem (A : ℕ → Set ι) (a : ι) :
  condition1 A → condition2 A a → 0 < a → a < 2 :=
sorry

end proof_problem_l456_456892


namespace hyperbola_properties_l456_456030

theorem hyperbola_properties :
  let a := 3
  let b := 4
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  (HyperbolaEquation : ∀ x y : ℝ, (y^2 / 9 - x^2 / 16 = 1))
  (Eccentricity : e = 5 / 3)
  (Asymptotes : ∀ x y : ℝ, ∃ m : ℝ, (y = m * x ∧ (m = 3 / 4 ∨ m = -3 / 4)))
  (AreaCondition :  ∀ (P F1 F2 : Point), ⟪PF1, PF2⟫ = 0 → (area P F1 F2 ≠ 9))
  (DistanceCondition : ∀ P : Point, (prod_dist_asymptotes P (3/4)) = 144 / 25)
  true :=
by
  sorry

end hyperbola_properties_l456_456030


namespace compute_volume_of_cube_l456_456576

-- Define the conditions and required properties
variable (s V : ℝ)

-- Given condition: the surface area of the cube is 384 sq cm
def surface_area (s : ℝ) : Prop := 6 * s^2 = 384

-- Define the volume of the cube
def volume (s : ℝ) (V : ℝ) : Prop := V = s^3

-- Theorem statement to prove the volume is correctly computed
theorem compute_volume_of_cube (h₁ : surface_area s) : volume s 512 :=
  sorry

end compute_volume_of_cube_l456_456576


namespace inclination_angle_of_line_l456_456520

theorem inclination_angle_of_line :
  ∃ θ : ℝ, θ = (2 * Real.pi / 3) ∧ ∀ x y : ℝ, 
    (sqrt 3 * x + y + 2024 = 0) → 
    Real.tan θ = -sqrt 3 :=
by
  sorry

end inclination_angle_of_line_l456_456520


namespace fraction_zero_implies_a_eq_neg2_l456_456754

theorem fraction_zero_implies_a_eq_neg2 (a : ℝ) (h : (a^2 - 4) / (a - 2) = 0) (h2 : a ≠ 2) : a = -2 :=
sorry

end fraction_zero_implies_a_eq_neg2_l456_456754


namespace octagon_area_ratio_l456_456273

theorem octagon_area_ratio (r : ℝ) : 
  let S := (1 : ℝ) in -- Scale factor for inscribed octagon side
  let C := (√2 : ℝ) in -- Scale factor for circumscribed octagon side
  let area_inscribed := 2 * (1 + √2) * r^2 in -- Inscribed octagon
  let area_circumscribed := 2 * (1 + √2) * (r * √2)^2 in -- Circumscribed octagon
  area_circumscribed / area_inscribed = 2 :=
by
  sorry

end octagon_area_ratio_l456_456273


namespace finite_set_unique_symmetry_center_l456_456748

structure SymCenter (H : Set Point) (O : Point) : Prop :=
(symmetry : ∀ A ∈ H, reflect_over(O, A) ∈ H)

theorem finite_set_unique_symmetry_center {H : Set Point} (finite_H : Finite H) (O O' : Point) :
  SymCenter H O → SymCenter H O' → O = O' := 
by
  sorry

end finite_set_unique_symmetry_center_l456_456748


namespace pentagons_parallel_plane_l456_456202

noncomputable theory
open_locale classical


variables (O A B C D A1 B1 C1 D1 : Type*)
variables [AddCommGroup O] [Module ℝ O]
variables [AddCommGroup A] [Module ℝ A] (a a₁ b b₁ c c₁ d d₁ : O)
variables [Nonempty O]

def regular_pentagon (P Q R S T : O) : Prop := sorry

axiom common_vertex : a = a₁ ∧ b = b₁ ∧ c = c₁ ∧ d = d₁
axiom different_planes : ¬ (AffineSpan ℝ [a, b, c, d] : AffineSubspace ℝ O) =
                        (AffineSpan ℝ [a₁, b₁, c₁, d₁] : AffineSubspace ℝ O)

theorem pentagons_parallel_plane :
  regular_pentagon O A B C D → regular_pentagon O A1 B1 C1 D1 → 
  ∃ (π : AffineSubspace ℝ O), 
  ∀ {P Q : O}, P ∈ [a, b, c, d] → Q ∈ [a₁, b₁, c₁, d₁] → 
  (segment ℝ P Q) ⊆ π :=
sorry

end pentagons_parallel_plane_l456_456202


namespace find_a3_l456_456433

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

variables {a₁ q a₂ a₃ a₄ a₅ : ℝ}

def sum_first_five (a₁ q : ℝ) : ℝ :=
  geometric_sequence a₁ q 1 + geometric_sequence a₁ q 2 + geometric_sequence a₁ q 3 + 
  geometric_sequence a₁ q 4 + geometric_sequence a₁ q 5

def sum_reciprocals_first_five (a₁ q : ℝ) : ℝ :=
  1 / geometric_sequence a₁ q 1 + 1 / geometric_sequence a₁ q 2 + 1 / geometric_sequence a₁ q 3 + 
  1 / geometric_sequence a₁ q 4 + 1 / geometric_sequence a₁ q 5

theorem find_a3 (h₁ : sum_first_five a₁ q = 27) 
                (h₂ : sum_reciprocals_first_five a₁ q = 3) 
                (h₃ : a₃ = geometric_sequence a₁ q 3) :
  a₃ = 3 ∨ a₃ = -3 :=
sorry

end find_a3_l456_456433


namespace problem_l456_456061

def g(x : ℤ) : ℤ := 3 * x^2 + 3 * x - 2

theorem problem : g(g(3)) = 3568 := by
  sorry

end problem_l456_456061


namespace find_y_l456_456319

theorem find_y (y : ℝ) (hy : log y 81 = 4 / 2) : y = 9 := 
by {
  sorry
}

end find_y_l456_456319


namespace minimum_time_condition_l456_456901

def total_distance : ℝ := 84
def walking_speed : ℝ := 5
def biking_speed : ℝ := 20
def participants : ℕ := 3

theorem minimum_time_condition 
: ∃ t: ℝ, 
    t = 7.8 ∧
    t = max ((3 - 2 * (4 / 7)) / biking_speed * total_distance) ((5 + (4 / 7)) / walking_speed / 3) * total_distance :=
begin 
   sorry
end

end minimum_time_condition_l456_456901


namespace player2_prevents_winning_l456_456913

def infinite_grid := ℤ × ℤ

inductive Player
| Player1 : Player
| Player2 : Player

inductive Symbol
| X : Symbol
| O : Symbol

def place_symbol (grid : infinite_grid → Option Symbol) 
                 (pos : infinite_grid) 
                 (sym : Symbol) 
                 : infinite_grid → Option Symbol :=
  λ p, if p = pos then some sym else grid p 

def is_winning_line (grid : infinite_grid → Symbol) 
                    (sym : Symbol) 
                    (len : ℕ) 
                    : infinite_grid → infinite_grid → Prop :=
  λ p1 p2, 
    let dx := p2.fst - p1.fst 
    let dy := p2.snd - p1.snd 
    (dx = 0 ∨ dy = 0 ∨ dx = dy ∨ dx = -dy) ∧ 
    ∀ i : ℕ, i < len → grid ⟨p1.fst + i * dx / len, p1.snd + i * dy / len⟩ = some sym

theorem player2_prevents_winning :
  ∀ (grid : infinite_grid → Option Symbol) 
    (len : ℕ) 
    (pos : infinite_grid) 
    (sym : Symbol) 
    (player : Player),
  len = 11 → 
  ∀ strategy : (infinite_grid → Option Symbol) → infinite_grid → Option Symbol → infinite_grid → Option Symbol,
    let new_grid := strategy grid pos none in
    ¬ is_winning_line (λ p, new_grid p) X len (0, 0) (0, 10) :=
  sorry

end player2_prevents_winning_l456_456913


namespace ratio_sum_l456_456739

theorem ratio_sum {x y : ℚ} (h : x / y = 4 / 7) : (x + y) / y = 11 / 7 :=
sorry

end ratio_sum_l456_456739


namespace heptagon_diagonals_l456_456053

theorem heptagon_diagonals : ∀ (P : Type) [polygon P], polygon.sides P = 7 → diagonals P = 14 :=
by
  sorry

end heptagon_diagonals_l456_456053


namespace projectile_height_at_30_l456_456264

-- Define the height function
def height (t : ℝ) : ℝ := 60 - 9 * t - 4.5 * t^2

-- State the theorem
theorem projectile_height_at_30 :
  ∃ t : ℝ, height t = 30 ∧ t = -1 + (Real.sqrt 276) / 3 :=
  by
    sorry

end projectile_height_at_30_l456_456264


namespace intersection_A_B_l456_456392

-- Define sets A and B
def A := {x : ℝ | abs x < 3}
def B := {n : ℕ | true}  -- ℕ is the set of natural numbers

-- Lean statement to prove the intersection A ∩ B
theorem intersection_A_B :
  A ∩ B = {0, 1, 2} :=
by
  sorry

end intersection_A_B_l456_456392


namespace incorrect_conversion_l456_456557

/--
Incorrect conversion of -150° to radians.
-/
theorem incorrect_conversion : (¬(((-150 : ℝ) * (Real.pi / 180)) = (-7 * Real.pi / 6))) :=
by
  sorry

end incorrect_conversion_l456_456557


namespace profit_is_36_l456_456131

-- Define the amounts Natasha, Carla, and Cosima have
variables (N C Co : ℕ)

-- Define the conditions
def conditions := 
  (N = 60) ∧  -- Natasha has $60
  (N = 3 * C) ∧  -- Natasha has 3 times as much money as Carla
  (C = 2 * Co)  -- Carla has twice as much money as Cosima

-- Define the buying price, selling price, and profit
def buying_price := N + C + Co
def selling_price := (7 / 5 : ℚ) * buying_price
def profit := selling_price - buying_price

-- The theorem to prove
theorem profit_is_36 (h : conditions N C Co) : profit N C Co = 36 := by {
  sorry
}

end profit_is_36_l456_456131


namespace no_valid_k_l456_456105

noncomputable def f (k x : ℝ) : ℝ := (3 * x + 4) / (k * x - 3)

theorem no_valid_k : ∀ k : ℝ, (∀ x : ℝ, f k (f k x) = x) ∧ f k 1 = 1 → False :=
begin
  assume k h,
  sorry
end

end no_valid_k_l456_456105


namespace no_integer_solutions_l456_456653

theorem no_integer_solutions :
   ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 = 2 * x^2 * y^2 + 2 * y^2 * z^2 + 2 * z^2 * x^2 + 24 :=
by
  sorry

end no_integer_solutions_l456_456653


namespace find_weights_l456_456541

def item_weights (a b c d e f g h : ℕ) : Prop :=
  1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 1 ≤ d ∧ 1 ≤ e ∧ 1 ≤ f ∧ 1 ≤ g ∧ 1 ≤ h ∧
  a > b ∧ b > c ∧ c > d ∧ d > e ∧ e > f ∧ f > g ∧ g > h ∧
  a ≤ 15 ∧ b ≤ 15 ∧ c ≤ 15 ∧ d ≤ 15 ∧ e ≤ 15 ∧ f ≤ 15 ∧ g ≤ 15 ∧ h ≤ 15

theorem find_weights (a b c d e f g h : ℕ) (hw : item_weights a b c d e f g h) 
    (h1 : d + e + f + g > a + b + c + h) 
    (h2 : e + f > d + g) 
    (h3 : e > f) : e = 11 ∧ g = 5 := sorry

end find_weights_l456_456541


namespace total_juice_boxes_needed_l456_456847

-- Definitions for the conditions
def john_juice_per_week : Nat := 2 * 5
def john_school_weeks : Nat := 18 - 2 -- taking into account the holiday break

def samantha_juice_per_week : Nat := 1 * 5
def samantha_school_weeks : Nat := 16 - 2 -- taking into account after-school and holiday break

def heather_mon_wed_juice : Nat := 3 * 2
def heather_tue_thu_juice : Nat := 2 * 2
def heather_fri_juice : Nat := 1
def heather_juice_per_week : Nat := heather_mon_wed_juice + heather_tue_thu_juice + heather_fri_juice
def heather_school_weeks : Nat := 17 - 2 -- taking into account personal break and holiday break

-- Question and Answer in lean
theorem total_juice_boxes_needed : 
  (john_juice_per_week * john_school_weeks) + 
  (samantha_juice_per_week * samantha_school_weeks) + 
  (heather_juice_per_week * heather_school_weeks) = 395 := 
by
  sorry

end total_juice_boxes_needed_l456_456847


namespace sum_Sn_formula_l456_456043

open Nat

def sumSequence (n : ℕ) : ℚ :=
  (∑ k in range n.succ, 8 * (k + 1) / ((2 * (k + 1) - 1) ^ 2 * (2 * (k + 1) + 1) ^ 2))

noncomputable def Sn (n : ℕ) : ℚ :=
  sumSequence n

theorem sum_Sn_formula (n : ℕ) : Sn n = (2 * n + 1) ^ 2 - 1 / (2 * n + 1) ^ 2 :=
by
  induction n with
  | zero => sorry
  | succ n ih => sorry

end sum_Sn_formula_l456_456043


namespace garden_perimeter_l456_456227

theorem garden_perimeter
  (width_garden : ℝ) (area_playground : ℝ)
  (length_playground : ℝ) (width_playground : ℝ)
  (area_garden : ℝ) (L : ℝ)
  (h1 : width_garden = 4) 
  (h2 : length_playground = 16)
  (h3 : width_playground = 12)
  (h4 : area_playground = length_playground * width_playground)
  (h5 : area_garden = area_playground)
  (h6 : area_garden = L * width_garden) :
  2 * L + 2 * width_garden = 104 :=
by
  sorry

end garden_perimeter_l456_456227


namespace green_hats_count_l456_456226

theorem green_hats_count 
  (B G : ℕ)
  (h1 : B + G = 85)
  (h2 : 6 * B + 7 * G = 530) : 
  G = 20 :=
by
  sorry

end green_hats_count_l456_456226


namespace circle_center_radius_l456_456660

def circle_equation (x y : ℝ) : Prop := x^2 + 4 * x + y^2 - 6 * y - 12 = 0

theorem circle_center_radius :
  ∃ (h k r : ℝ), (circle_equation (x : ℝ) (y: ℝ) -> (x + h)^2 + (y + k)^2 = r^2) ∧ h = -2 ∧ k = 3 ∧ r = 5 :=
sorry

end circle_center_radius_l456_456660


namespace children_tickets_sold_l456_456902

theorem children_tickets_sold :
  ∃ (C : ℕ), 
    (∃ (A : ℕ), A + C = 400 ∧ 6 * A + 4.5 * C = 2100) ∧ C = 200 :=
by
  sorry

end children_tickets_sold_l456_456902


namespace damage_ratio_proof_l456_456123

variable (H g M τ : ℝ)
variable (k n : ℝ)
variable (h : ℝ := H / n)
variable (VI : ℝ := sqrt (2 * g * H))
variable (V1 : ℝ := sqrt (2 * g * h))
variable (V1' : ℝ := (1 / k) * sqrt (2 * g * h))
variable (VII : ℝ := sqrt ((1 / k^2) * 2 * g * h + 2 * g * (H - h)))
variable (II : ℝ := M * τ * ((V1 - V1') + VII))
variable (I1 : ℝ := M * VI * τ)
variable (damage_ratio : ℝ := II / I1)

theorem damage_ratio_proof : 
  (1 / k - 1) / (sqrt n * k) + 
  sqrt ((n - 1) * k^2 + 1) / (sqrt n * k^2) = 
  5 / 4 :=
sorry

end damage_ratio_proof_l456_456123


namespace length_of_field_l456_456176

-- Define the problem conditions
variables (width length : ℕ)
  (pond_area field_area : ℕ)
  (h1 : length = 2 * width)
  (h2 : pond_area = 64)
  (h3 : pond_area = field_area / 8)

-- Define the proof problem
theorem length_of_field : length = 32 :=
by
  -- We'll provide the proof later
  sorry

end length_of_field_l456_456176


namespace circle_tangent_lines_l456_456078

-- Define the circle and its properties
def is_circle_with_tangent_at_origin_tangent_to_line (x y : ℝ) : Prop :=
  x^2 + y^2 = 4 ∧ (x - sqrt(3) * y - 4 = 0)

-- Define the properties of the tangent lines from point P(3, 2)
def tangents_from_point (P : ℝ × ℝ) (x y : ℝ) : Prop :=
  (x = 12 * P.1 - 5 * P.2 - 26 ∧ y = 0) ∨ (y = 2 ∧ P.1 = 3)

-- Main theorem to be proven
theorem circle_tangent_lines (x y : ℝ) (P : ℝ × ℝ) :
  is_circle_with_tangent_at_origin_tangent_to_line x y →
  tangents_from_point P x y :=
sorry

end circle_tangent_lines_l456_456078


namespace pebbles_difference_l456_456994

def candy_pebbles : Nat := 4
def lance_pebbles : Nat := 3 * candy_pebbles

theorem pebbles_difference {candy_pebbles lance_pebbles : Nat} (h1 : candy_pebbles = 4) (h2 : lance_pebbles = 3 * candy_pebbles) : lance_pebbles - candy_pebbles = 8 := by
  sorry

end pebbles_difference_l456_456994


namespace triangle_ABC_angles_l456_456434

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def is_midpoint (M X Y : Point) : Prop := sorry
noncomputable def is_parallel (l1 l2 : Line) : Prop := sorry
noncomputable def line_intersection_point (l : Line) (Δ : Triangle) : Point := sorry

structure Triangle :=
(A B C : Point)

structure Point :=
(x y : ℝ)

structure Line :=
(P Q : Point)

variables (A B C D E O H : Point)
variable (ABC : Triangle)

axiom h1 : O = circumcenter ABC
axiom h2 : H = orthocenter ABC
axiom h3 : is_midpoint (midpoint O H) O H
axiom h4 : is_parallel (Line.mk D E) (Line.mk B C)
axiom h5 : O = incenter (Triangle.mk A D E)

theorem triangle_ABC_angles :
  let α := angle A B C in 
  let β := angle B A C in
  let γ := angle C A B in
  α = 36 ∧ β = 72 ∧ γ = 72 :=
sorry

end triangle_ABC_angles_l456_456434


namespace gage_skating_time_l456_456676

theorem gage_skating_time
  (skate_5_days : Nat := 5 * (1 * 60 + 15)) -- 5 days at 75 minutes each
  (skate_3_days : Nat := 3 * (1 * 60 + 30)) -- 3 days at 90 minutes each
  (skate_1_day  : Nat := 2 * 60)             -- 1 day at 120 minutes
  (skate_other_day : Nat := 50)              -- 50 minutes for another day
  (total_days : Nat := 10)
  (desired_avg : Nat := 90) :
  total_days * desired_avg = skate_5_days + skate_3_days + skate_1_day + skate_other_day + 85 :=
by
  noncomputable theory
  sorry

end gage_skating_time_l456_456676


namespace sale_in_third_month_l456_456599

theorem sale_in_third_month
  (sale1 sale2 sale4 sale5 sale6 avg : ℝ)
  (n : ℕ)
  (h_sale1 : sale1 = 6235)
  (h_sale2 : sale2 = 6927)
  (h_sale4 : sale4 = 7230)
  (h_sale5 : sale5 = 6562)
  (h_sale6 : sale6 = 5191)
  (h_avg : avg = 6500)
  (h_n : n = 6) :
  ∃ sale3 : ℝ, sale3 = 6855 := by
  sorry

end sale_in_third_month_l456_456599


namespace tenth_flip_head_probability_l456_456931

/-- Given that the coin is fair and the first 9 flips resulted in 6 heads,
prove that the probability that the 10th flip will result in a head is 1/2. -/
theorem tenth_flip_head_probability (fair_coin : ℙ (flip = head) = 1/2 ∧ ℙ (flip = tail) = 1/2)
: ℙ (flip = head) = 1/2 :=
by
  sorry

end tenth_flip_head_probability_l456_456931


namespace sum_of_first_2012_terms_sequence_l456_456044

theorem sum_of_first_2012_terms_sequence (a₀ a₁ : ℤ) (seq : ℕ → ℤ)
  (h0 : a₀ = 2010)
  (h1 : a₁ = 2011)
  (h_seq : ∀ n ≥ 1, seq (n + 1) = seq (n - 1) + seq n) :
  (∑ i in finset.range 2012, seq i) = 4021 := 
sorry

end sum_of_first_2012_terms_sequence_l456_456044


namespace students_only_english_l456_456940

theorem students_only_english (S B G E : ℕ) (hS : S = 52) (hB : B = 12) (hG_total : G + B = 22) (H : E + G + B = S) : E = 30 :=
by {
  -- We can skip actual proof implementation by stating sorry
  sorry,
}

end students_only_english_l456_456940


namespace marcus_point_value_l456_456473

theorem marcus_point_value 
  (team_total_points : ℕ)
  (marcus_percentage : ℚ)
  (three_point_goals : ℕ)
  (num_goals_type2 : ℕ)
  (score_type1 : ℕ)
  (score_type2 : ℕ)
  (total_marcus_points : ℚ)
  (points_type2 : ℚ)
  (three_point_value : ℕ := 3):
  team_total_points = 70 →
  marcus_percentage = 0.5 →
  three_point_goals = 5 →
  num_goals_type2 = 10 →
  total_marcus_points = marcus_percentage * team_total_points →
  score_type1 = three_point_goals * three_point_value →
  points_type2 = total_marcus_points - score_type1 →
  score_type2 = points_type2 / num_goals_type2 →
  score_type2 = 2 :=
by
  intros
  sorry

end marcus_point_value_l456_456473


namespace intersection_eq_l456_456698

open Set

variable (A B : Set ℝ)

def setA : A = {x | -3 < x ∧ x < 2} := sorry

def setB : B = {x | x^2 + 4*x - 5 ≤ 0} := sorry

theorem intersection_eq : A ∩ B = {x | -3 < x ∧ x ≤ 1} :=
sorry

end intersection_eq_l456_456698


namespace pencils_in_drawer_l456_456897

theorem pencils_in_drawer (initial_pencils : ℕ) (added_pencils : ℕ) 
  (initial_cond : initial_pencils = 2) (added_cond : added_pencils = 3) : 
  initial_pencils + added_pencils = 5 :=
by 
  rw [initial_cond, added_cond]
  rfl

end pencils_in_drawer_l456_456897


namespace incircle_tangent_to_adc_sides_l456_456342

noncomputable def Triangle (A B C : Point) : Prop := -- defining Triangle for context
  True

noncomputable def CircleTangentToSidesAndExtensions (circle : Circle) (AC BA BC : Line) : Prop := -- tangent condition
  True

noncomputable def Parallelogram (A B C D : Point) : Prop := -- defining Parallelogram for context
  True

theorem incircle_tangent_to_adc_sides 
  (A B C D P S Q R : Point)
  (AC BA BC DA DC : Line)
  (circle : Circle) 
  (h_parallelogram : Parallelogram A B C D)
  (h_tangent : CircleTangentToSidesAndExtensions circle AC BA BC)
  (h_intersection : LineIntersectsSegmentsInPoints (line_through P S) DA DC Q R) :
  TangentToIncircleAtPoints (Triangle C D A) (incircle (Triangle C D A)) Q R :=
by
  sorry

end incircle_tangent_to_adc_sides_l456_456342


namespace local_minimum_at_neg_one_l456_456583

noncomputable def f : ℝ → ℝ := λ x, x * Real.exp x

def f' (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem local_minimum_at_neg_one : ∃ δ > 0, ∀ x, x ∈ (Icc (-1 - δ) (-1 + δ) \ { -1 }) → f x > f (-1) :=
by
  sorry

end local_minimum_at_neg_one_l456_456583


namespace kolya_wins_l456_456801

/-- Define the structure of the game board and the conditions of the game. -/
structure GameState :=
  (n : ℕ) -- size of the board, in this case 5
  (center : ℕ × ℕ) -- coordinates of the center, in this case (3,3)
  (kolya_turn : Bool) -- a flag to check whose turn it is
  (elephants : List (ℕ × ℕ)) -- a list to store positions of elephants

/-- Define the initial conditions of the game. -/
def initial_state : GameState :=
  { n := 5,
    center := (3, 3),
    kolya_turn := true, -- Kolya goes first
    elephants := [] } -- no elephants placed initially

/-- Define a function to check if a move is valid. -/
def valid_move (state : GameState) (move : ℕ × ℕ) : Prop :=
  move.1 > 0 ∧ move.1 <= state.n ∧
  move.2 > 0 ∧ move.2 <= state.n ∧
  (move ≠ state.center ∨ ¬state.kolya_turn) ∧ -- Kolya cannot place in the center on first move
  move ∉ state.elephants

/-- Define a function to make a move, update the game state accordingly. -/
def make_move (state : GameState) (move : ℕ × ℕ) (h : valid_move state move) : GameState :=
  { state with 
    kolya_turn := ¬state.kolya_turn, 
    elephants := move :: state.elephants }

/-- State that Kolya wins given the conditions. -/
theorem kolya_wins (s : GameState) : (∃ winning_state, (initial_state → winning_state) ∧ winning_state.elephants.length = 25) := 
 sorry -- Proof to be filled in later

end kolya_wins_l456_456801


namespace sin_cos_positive_in_first_quadrant_sin_cos_positive_sufficient_not_necessary_l456_456405

variable (α : ℝ)

definition is_first_quadrant (α : ℝ) : Prop :=
  0 < α ∧ α < π / 2

theorem sin_cos_positive_in_first_quadrant :
  (is_first_quadrant α) → (sin α) * (cos α) > 0 :=
by
  sorry

theorem sin_cos_positive_sufficient_not_necessary :
  (∃ α : ℝ, is_first_quadrant α) ↔ (∃ α : ℝ, (sin α) * (cos α) > 0) :=
by
  sorry

end sin_cos_positive_in_first_quadrant_sin_cos_positive_sufficient_not_necessary_l456_456405


namespace uniform_prob_expected_value_l456_456280

noncomputable theory
open Real

def S_ABC : ℝ := 1 -- assuming some value for example

def S (x : ℝ) : ℝ := S_ABC * (1 - 1 / 6 - 3 * x / 4 - (1 - x) / 3)
def X (x : ℝ) : ℝ := S (x) / S_ABC

theorem uniform_prob (h₁ : ∀ x, 0 ≤ x ∧ x ≤ 1)
  (h₂ : ∀ a b : ℝ, a = b → a = 1 / 3 → b = 1 / 3) : 
  (∀ x, x ≤ 2 / 5 → X (x) ≥ 1 / 3) → 
  (∫ x in 0..1, 1 = ∫ x in 0.. 2 / 5, 1) :=
sorry

theorem expected_value (h₃ : ∀ x, 0 ≤ x ∧ x ≤ 1) :
  (∫ x in 0..1, X (x)) = 7 / 24 :=
sorry

end uniform_prob_expected_value_l456_456280


namespace angle_between_vectors_l456_456025

open Real

variables (a b : ℝ^3)
variables (theta : ℝ)

/-- Given the magnitudes and relationship between vectors a and b, prove that the angle between them is 150 degrees. -/
theorem angle_between_vectors
  (h1 : ‖a‖ = 1)
  (h2 : ‖b‖ = 2 * sqrt 3)
  (h3 : ‖2 • a - b‖ = 2 * sqrt 7)
  (h4 : theta = 150 * Real.pi / 180) :
  Real.acos ((a • b) / (‖a‖ * ‖b‖)) = theta :=
sorry

end angle_between_vectors_l456_456025


namespace geometric_series_sum_l456_456996

theorem geometric_series_sum : 
  (3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9 + 3^10) = 88572 := 
by 
  sorry

end geometric_series_sum_l456_456996


namespace product_has_34_digits_l456_456813

-- Define the first number with 19 digits
def a : ℕ := 3659893456789325678

-- Define the second number with 15 digits
def b : ℕ := 342973489379256

-- Define the product
def P : ℕ := a * b

-- Define a function to compute the number of digits in a natural number
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log10 n + 1

-- Translate the mathematical proof problem into Lean
theorem product_has_34_digits : num_digits P = 34 := 
by 
  -- The proof will be added here
  sorry

end product_has_34_digits_l456_456813


namespace part1_eq_of_slope_and_vertex_part2_max_area_l456_456472

-- Define the ellipse and the conditions for the first part
def ellipse (x y a : ℝ) : Prop :=
  (x^2 / a^2) + y^2 = 1

def slope_eq (x y : ℝ) : Prop :=
  y / x = -1 / 2

def is_vertex (x y a : ℝ) : Prop :=
  (x = -a ∧ y = 0) ∨ (x = 0 ∧ y = 1)

-- First part: Prove the standard equation of the ellipse
theorem part1_eq_of_slope_and_vertex (a : ℝ) (h1 : 1 < a)
  (h2 : ∃ (x y : ℝ), is_vertex x y a ∧ slope_eq x y ∧ ellipse x y a) :
  a = 2
∧ 
  ∀ x y : ℝ, ellipse x y 2 := 
sorry

-- Define the ellipse and the conditions for the second part
def dist_from_origin (x y : ℝ) : ℝ :=
  real.sqrt (x^2 + y^2)

-- Second part: Prove the maximum area
theorem part2_max_area (a : ℝ) (h1 : a = 2)
  (h2 : ∃ (x y : ℝ), dist_from_origin x y = 1 ∧ ellipse x y a) :
  ∀ A B O : ℝ × ℝ, area (triangle A B O) ≤ 1 := 
sorry

end part1_eq_of_slope_and_vertex_part2_max_area_l456_456472


namespace monotonic_increasing_interval_l456_456377

open Real

noncomputable def f (ω x : ℝ) : ℝ := sqrt 3 * sin (ω * x) + cos (ω * x)

theorem monotonic_increasing_interval (ω α β : ℝ) (k : ℤ) (Hω : ω > 0)
  (Hfα : f ω α = 2) (Hfβ : f ω β = 0) (Hαβ : |α - β| = π / 2) :
  ∃ (L R : ℝ), (L = 2 * k * π - 2 * π / 3) ∧ (R = 2 * k * π + π / 3) ∧
  ∀ x, L ≤ x ∧ x ≤ R → f ω x = 2 * sin(x + π / 6) ∧ sin(x + π / 6) > 0 :=
sorry

end monotonic_increasing_interval_l456_456377


namespace point_on_line_l456_456667

-- Define the function to calculate the slope of a line passing through the points (x1, y1) and (x2, y2)
def slope (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

-- Define a predicate to check if a point (x, y) lies on a line passing through the points (x1, y1) and (x2, y2)
def on_line (x y x1 y1 x2 y2 : ℝ) : Prop :=
  slope x1 y1 x2 y2 = slope x1 y1 x y

theorem point_on_line :
  on_line 14 7 2 1 10 5 :=
by
  -- Proof of the theorem
  sorry

end point_on_line_l456_456667


namespace hyperbola_parameters_sum_l456_456768

def h := 1
def k := -2
def a := 2
def c := 7
def b := Real.sqrt (c^2 - a^2)

theorem hyperbola_parameters_sum :
  h + k + a + b = 1 + 3 * Real.sqrt 5 :=
by
  -- Proof not needed as per the instruction
  sorry

end hyperbola_parameters_sum_l456_456768


namespace solve_for_x_l456_456878

noncomputable def infinite_power_tower (x : ℝ) : ℝ := sorry

theorem solve_for_x (x : ℝ) 
  (h1 : infinite_power_tower x = 4) : 
  x = Real.sqrt 2 := 
sorry

end solve_for_x_l456_456878


namespace integer_roots_sum_abs_eq_94_l456_456672

theorem integer_roots_sum_abs_eq_94 {a b c m : ℤ} :
  (∃ m, (x : ℤ) * (x : ℤ) * (x : ℤ) - 2013 * (x : ℤ) + m = 0 ∧ a + b + c = 0 ∧ ab + bc + ac = -2013) →
  |a| + |b| + |c| = 94 :=
sorry

end integer_roots_sum_abs_eq_94_l456_456672


namespace polar_equation_of_transformed_curve_calculate_distance_ratio_l456_456308

theorem polar_equation_of_transformed_curve 
  (C : set (ℝ × ℝ)) 
  (C_parametric : ∀ θ, (2 * cos θ, sqrt 3 * sin θ) ∈ C)
  (C_cartesian : ∀ x y, (x, y) ∈ C ↔ x^2 / 4 + y^2 / 3 = 1)
  (C_prime : set (ℝ × ℝ)) 
  (C_prime_parametric : ∀ θ, (cos θ, sin θ) ∈ C_prime)
  (C_prime_transformed : ∀ x y, (cos θ, sin θ) = ((1/2) * x, (1/√3) * y)) :
  ∀ ρ θ, ρ = 1 :=
by
  sorry

theorem calculate_distance_ratio 
  (A : ℝ × ℝ) (A_polar : A = (-3 / 2, 0))
  (line_l : ℝ → ℝ × ℝ) (l_parametric : ∀ t, line_l t = (-2 + t * cos (π / 6), t * sin (π / 6)))
  (C_prime_parametric : ∀ θ, (cos θ, sin θ) ∈ set.range (λ θ, ρ * exp (θ * Complex.I)))
  (intersection_MN : (M N : ℝ × ℝ → Prop) ∧ (P : ℝ × ℝ) ∧ (midpoint_P : P = (M + N)/2))
  : 
  ∀ AP AM AN : ℝ, (AP = distance A P) ∧ (AM = distance A M) ∧ (AN = distance A N) → 
  ∑ |AP| / (|AM| * |AN|) = 3 * sqrt 3 / 5 := 
by
  sorry

end polar_equation_of_transformed_curve_calculate_distance_ratio_l456_456308


namespace surface_area_large_cube_l456_456199

theorem surface_area_large_cube (edge_length_small_cube : ℕ) (n_cubes : ℕ) 
  (h_small_cube_edge : edge_length_small_cube = 4) (h_n_cubes : n_cubes = 27) :
  let edge_length_large_cube := (∛n_cubes) * edge_length_small_cube in
  let surface_area_large_cube := 6 * (edge_length_large_cube ^ 2) in
  surface_area_large_cube = 864 :=
by
  sorry

end surface_area_large_cube_l456_456199


namespace at_least_one_genuine_product_prob_eq_one_l456_456982

theorem at_least_one_genuine_product_prob_eq_one
  (total_products : ℕ) (genuine_products : ℕ) (defective_products : ℕ) (selected_products : ℕ)
  (total_eq : total_products = 16) (genuine_eq : genuine_products = 14) (defective_eq : defective_products = 2)
  (selected_eq : selected_products = 3) :
  (probability_of_event (at_least_one_genuine_product selected_products total_products genuine_products defective_products) = 1) :=
sorry

def at_least_one_genuine_product (selected_products total_products genuine_products defective_products : ℕ) : Prop :=
  selected_products ≥ 1 ∧ (genuine_products ≥ 1 ∨ defective_products < selected_products)

def probability_of_event (event : Prop) : ℝ := 1  -- Assume proper probability calculation logic

end at_least_one_genuine_product_prob_eq_one_l456_456982


namespace compute_c_plus_d_l456_456328

theorem compute_c_plus_d (c d : ℕ) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_product_length : (d - 1) - (c + 1) + 1 = 666)
  (h_log_product : ∑ i in finset.range (d - c), real.log (c + i) = real.log (c^3)) :
  c + d = 222 :=
sorry

end compute_c_plus_d_l456_456328


namespace concert_not_goers_l456_456132

-- Define the conditions as Lean definitions
def total_tickets : ℕ := 900
def before_concert : ℕ := (3 / 4 : ℚ) * total_tickets
def remaining_after_before_concert : ℕ := total_tickets - before_concert
def after_first_song : ℕ := (5 / 9 : ℚ) * remaining_after_before_concert
def middle_concert : ℕ := 80
def total_arrivals : ℕ := before_concert + after_first_song + middle_concert
def not_goers := total_tickets - total_arrivals

-- The theorem to prove
theorem concert_not_goers
  (h1 : total_tickets = 900)
  (h2 : before_concert = (3 / 4 : ℚ) * total_tickets)
  (h3 : remaining_after_before_concert = total_tickets - before_concert)
  (h4 : after_first_song = (5 / 9 : ℚ) * remaining_after_before_concert)
  (h5 : middle_concert = 80)
  (h6 : total_arrivals = before_concert + after_first_song + middle_concert)
  (h7 : not_goers = total_tickets - total_arrivals) :
  not_goers = 20 := 
sorry

end concert_not_goers_l456_456132


namespace special_sale_percentage_reduction_l456_456890

-- The problem setup
def original_price (P : ℝ) : ℝ := P

def first_reduction (P : ℝ) : ℝ := 0.75 * P

def special_sale_reduction (P : ℝ) (x : ℝ) : ℝ := 0.75 * P * (1 - x / 100)

def price_after_increase (price : ℝ) : ℝ :=
  price * 1.4814814814814815 -- equivalent to increasing by approximately 48.148148148148145%

-- The statement to prove, that x equals 10 given the conditions
theorem special_sale_percentage_reduction (P : ℝ) (x : ℝ) :
  price_after_increase (special_sale_reduction P x) = original_price P → x = 10 :=
by
  intros h
  sorry

end special_sale_percentage_reduction_l456_456890


namespace cannot_be_sqrt_3_div_2_l456_456022

noncomputable def f : ℝ → ℝ := sorry
variable {A : set ℝ}
variable {x : ℝ}

axiom h1 : 1 ∈ A
axiom h2 : ∀ x ∈ A, (f (x * cos (π / 6) - (sin (π / 6) * f x))) = (sin (π / 6) * x + f (cos (π / 6) * f x)) 
axiom h3 : ∀ x, f x = x → false

theorem cannot_be_sqrt_3_div_2 (x ∈ A) : f 1 ≠ sqrt 3 / 2 := 
sorry

end cannot_be_sqrt_3_div_2_l456_456022


namespace area_ratio_is_two_l456_456267

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
let A := 2 * r^2 * (1 + sqrt(2)) / sqrt(2)
let B := r^2 * (2 + 2 * sqrt(2))
A / B

theorem area_ratio_is_two (r : ℝ) (h : r > 0) :
  area_ratio_of_octagons r = 2 := by
sorry

end area_ratio_is_two_l456_456267


namespace incircle_tangent_points_l456_456352

theorem incircle_tangent_points {A B C D P S Q R : Point} 
  (h_parallelogram : parallelogram A B C D) 
  (h_tangent_ac : tangent (circle P Q R) A C) 
  (h_tangent_ba_ext : tangent (circle P Q R) (extension B A P)) 
  (h_tangent_bc_ext : tangent (circle P Q R) (extension B C S)) 
  (h_ps_intersect_da : segment_intersect P S D A Q)
  (h_ps_intersect_dc : segment_intersect P S D C R) :
  tangent (incircle D C A) D A Q ∧ tangent (incircle D C A) D C R := sorry

end incircle_tangent_points_l456_456352


namespace not_both_zero_vectors_l456_456757

variables {G : Type*} [AddCommGroup G] [VectorSpace ℝ G]

variables (a b : G)

theorem not_both_zero_vectors (h : a ≠ b) : a ≠ 0 ∨ b ≠ 0 :=
by sorry

end not_both_zero_vectors_l456_456757


namespace find_fractional_equation_l456_456617

-- Definitions of the expressions
def A := (x^2 + 1) / 2 = 5 / 3
def B := (1 / (3*x - 1)) + (4*x / (3*x + 1))
def C := (x / (2*x - 1)) - (3 / (2*x + 1)) = 1
def D := (3 - x) / 4 + 2 = (x - 4) / 3

-- Definition of a fractional equation
def is_fractional_equation (expr : Prop) : Prop :=
  ∃ (f g : ℚ → ℚ), (x : ℚ) → expr

-- Proof statement
theorem find_fractional_equation :
  (is_fractional_equation C) ∧ ¬(is_fractional_equation A) ∧ ¬(is_fractional_equation B) ∧ ¬(is_fractional_equation D) :=
by
  sorry

end find_fractional_equation_l456_456617


namespace minimize_x_expr_minimized_l456_456365

noncomputable def minimize_x_expr (x : ℝ) : ℝ :=
  x + 4 / (x + 1)

theorem minimize_x_expr_minimized 
  (hx : x > -1) 
  : x = 1 ↔ minimize_x_expr x = minimize_x_expr 1 :=
by
  sorry

end minimize_x_expr_minimized_l456_456365


namespace proof_statement_l456_456374

variable {x α k : ℝ}
variable {λ m : ℝ}
variable {f : ℝ → ℝ}

def z1 (x : ℝ) (λ : ℝ) : ℂ := complex.sin (2 * x) + λ * complex.I
def z2 (x m : ℝ) : ℂ := m + (m - real.sqrt 3 * real.cos (2 * x)) * complex.I

axiom λ_eq_0 (h1 : λ = 0) (h2 : 0 < x ∧ x < real.pi) :
  (x = real.pi / 6) ∨ (x = 2 * real.pi / 3)

axiom λ_eq_fx (hf : ∀ x : ℝ, f x = 2 * real.sin (2 * x - real.pi / 3)) :
  (complex.imin_simple hf = real.pi) ∧
  (∀ k : ℤ, k * real.pi + 5 * real.pi / 12 ≤ x ∧ x ≤ k * real.pi + 11 * real.pi / 12) ∧
  (f α = 1 / 2 → real.cos (4 * α + real.pi / 3) = -7 / 8)

noncomputable def QED : Prop :=
  λ_eq_0 ∧ λ_eq_fx

theorem proof_statement : QED := by
  sorry

end proof_statement_l456_456374


namespace unique_spatial_quadrilateral_l456_456694

-- Given condition: four linearly independent vectors
variables {V : Type} [AddCommGroup V] [Module ℝ V]
variables (a b c d : V)

-- Proposition to prove
theorem unique_spatial_quadrilateral (ha : ¬Collinear {a, b, c}) :
  ∃ (α β γ : ℝ), α • a + β • b + γ • c + d = 0 ∧
  ∀ (α1 β1 γ1 : ℝ), α1 • a + β1 • b + γ1 • c + d = 0 → α1 = α ∧ β1 = β ∧ γ1 = γ :=
by sorry

end unique_spatial_quadrilateral_l456_456694


namespace five_thursdays_in_july_l456_456498

theorem five_thursdays_in_july (N : ℕ) (h_june_30_days : ∀ (N : ℕ), true) (h_july_31_days : ∀ (N : ℕ), true) (h_five_tuesdays_in_june : ∃ (t : ℕ), 1 ≤ t ∧ t ≤ 7 ∧ (t + 28 ≤ 30)) :
  ∃ day_in_july, day_in_july = "Thursday" ∧ (day_occurrences_in_july day_in_july = 5) :=  
sorry

end five_thursdays_in_july_l456_456498


namespace sequence_sum_value_l456_456084

section
variable {a : ℕ → ℝ}

axiom a_pos : ∀ n, a n > 0
axiom a_1 : a 1 = 1 / 2
axiom geom_mean : ∀ n, a (n + 1) ^ 2 = (2 * a(n) * a(n + 1) + 1) / (4 - a(n) ^ 2)

theorem sequence_sum_value :
  (a 1) + ∑ i in (finset.range 100).map (finset.natEmb), (a (i + 2) / ((i + 2) ^ 2)) = 100 / 101 :=
sorry
end

end sequence_sum_value_l456_456084


namespace expansion_with_max_fourth_term_l456_456411

open BigOperators

variable {n : ℕ}

/-- If the expansion of (3x - 1)^n has only its fourth term's binomial coefficient as the maximum, then n = 6. -/
theorem expansion_with_max_fourth_term (h : ∀ k : ℕ, k ≠ 3 → binomial n k < binomial n 3) : n = 6 :=
sorry

end expansion_with_max_fourth_term_l456_456411


namespace ratio_of_areas_triangle_quadrilateral_angle_CBL_l456_456139

-- Given definitions and conditions
variables (A B C M L P F : Type)
variables [Triangle ABC : Triangle A B C]
variables [Point M : on_segment B C]
variables [BM_MC_ratio : BM : MC = 3 : 8]
variables [Angle_bisector_BL : bisector B L in Triangle A B C]
variables [P_inter_AngBis_BL_AM_90deg : ∠BPA = 90°]
variables [Point F : on_segment M C]
variables [MF_FC_ratio : MF : FC = 1 : 7]
variables [LF_perpendicular_BC : perpendicular LF BC]

-- Proving the ratio of the area of triangle ABP to the area of quadrilateral LPMC is 21/100
theorem ratio_of_areas_triangle_quadrilateral :
  area (Triangle ABP) / area (Quadrilateral LPMC) = 21/100 := sorry

-- Proving the angle CBL is arccos (2 √7) / √33
theorem angle_CBL :
  ∠CBL = arccos (2 * sqrt 7 / sqrt 33) := sorry

end ratio_of_areas_triangle_quadrilateral_angle_CBL_l456_456139


namespace find_x_solution_l456_456320

theorem find_x_solution :
    ∃ x : ℝ, (sqrt x + 3 * sqrt (x ^ 2 + 8 * x) + sqrt (x + 8) = 42 - 3 * x) ∧ x = 49 / 9 := 
begin
  use 49 / 9,
  sorry, -- Proof steps would go here.
end

end find_x_solution_l456_456320


namespace percentage_of_first_to_second_l456_456939

theorem percentage_of_first_to_second (X : ℝ) (first second : ℝ) (h1 : first = (7 / 100) * X) (h2 : second = (14 / 100) * X) : 
(first / second) * 100 = 50 := by
  sorry

end percentage_of_first_to_second_l456_456939


namespace eggs_remaining_l456_456837

-- Assign the given constants
def hens : ℕ := 3
def eggs_per_hen_per_day : ℕ := 3
def days_gone : ℕ := 7
def eggs_taken_by_neighbor : ℕ := 12
def eggs_dropped_by_myrtle : ℕ := 5

-- Calculate the expected number of eggs Myrtle should have
noncomputable def total_eggs :=
  hens * eggs_per_hen_per_day * days_gone - eggs_taken_by_neighbor - eggs_dropped_by_myrtle

-- Prove that the total number of eggs equals the correct answer
theorem eggs_remaining : total_eggs = 46 :=
by
  sorry

end eggs_remaining_l456_456837


namespace approximate_reading_l456_456173

theorem approximate_reading (x : ℝ) (h1 : 15.75 < x) (h2 : x < 16.0) : x ≈ 15.9 :=
sorry

end approximate_reading_l456_456173


namespace gcd_2023_2048_l456_456920

theorem gcd_2023_2048 : Nat.gcd 2023 2048 = 1 := by
  sorry

end gcd_2023_2048_l456_456920


namespace weight_of_replaced_student_l456_456168

theorem weight_of_replaced_student (W : ℝ) : 
  (W - 12 = 5 * 12) → W = 72 :=
by
  intro hyp
  linarith

end weight_of_replaced_student_l456_456168


namespace solve_system_l456_456162

theorem solve_system :
  ∃! (x y : ℝ), (2 * x + y + 8 ≤ 0) ∧ (x^4 + 2 * x^2 * y^2 + y^4 + 9 - 10 * x^2 - 10 * y^2 = 8 * x * y) ∧ (x = -3 ∧ y = -2) := 
  by
  sorry

end solve_system_l456_456162


namespace N_mod_1000_l456_456803

def sequence_sum : ℤ := 
  let recursion (n : ℤ) (acc : ℤ) : ℤ :=
    if n = 0 then acc
    else if n % 4 = 0 then recursion (n - 2) (acc + (n + (n - 1)))
    else if n % 4 = 2 then recursion (n - 2) (acc - (n + (n - 1)))
    else acc
  recursion 100 0

theorem N_mod_1000 : sequence_sum % 1000 = 796 := by
  sorry

end N_mod_1000_l456_456803


namespace monotonic_intervals_range_of_a_l456_456725

noncomputable def f (x : ℝ) : ℝ := 1 / (x * Real.log x)

theorem monotonic_intervals : 
  ∀ x > 0, x ≠ 1, 
  ((x ∈ Ioo 0 (1 / Real.exp 1)) ∨ (x ∈ Ioo (1 / Real.exp 1) 1) ∨ (x ∈ Ioi 1)) ↔ 
  ((∀ x ∈ Ioo 0 (1 / Real.exp 1), ∃ δ > 0, ∀ y ∈ Ioo (x - δ) (x + δ), f y < f x) ∧
   (∀ x ∈ Ioo (1 / Real.exp 1) 1, ∃ δ > 0, ∀ y ∈ Ioo (x - δ) (x + δ), f x < f y) ∧
   (∀ x ∈ Ioi 1, ∃ δ > 0, ∀ y ∈ Ioo (x - δ) (x + δ), f x < f y))
    := sorry

theorem range_of_a : 
  ∀ a : ℝ, (∀ x ∈ Ioo 0 1, (1 / x) * Real.log 2 > a * Real.log x) ↔ 
  a > -Real.exp 1 * Real.log 2
    := sorry

end monotonic_intervals_range_of_a_l456_456725


namespace middle_card_is_4_l456_456544

def distinct (l : List ℕ) := l.nodup

def sum_is_13 (l : List ℕ) := l.sum = 13

def in_increasing_order (l : List ℕ) := l = l.sorted (· < ·)

def casey_cannot_determine (numbers : List ℕ) : Prop :=
  ∀ c ∈ numbers.head?, 
    ¬ ∃ l, l = numbers.tail.filter (λ x, c + x = 13) ∧ distinct l ∧ sum_is_13 (c :: l) ∧ in_increasing_order (c :: l)

def tracy_cannot_determine (numbers : List ℕ) : Prop :=
  ∀ t ∈ numbers.last?, 
    ¬ ∃ l, l = numbers.init.filter (λ x, x + t = 13) ∧ distinct l ∧ sum_is_13 (l ++ [t]) ∧ in_increasing_order (l ++ [t])

def stacy_cannot_determine (numbers : List ℕ) : Prop :=
  ∀ s ∈ numbers.nth 1?, 
    ¬ ∃ l, l = [numbers.head!, s, numbers.last!] ∧ distinct l ∧ sum_is_13 l ∧ in_increasing_order l

theorem middle_card_is_4 (numbers : List ℕ) 
  (h_distinct : distinct numbers) 
  (h_sum : sum_is_13 numbers) 
  (h_order : in_increasing_order numbers)
  (h_casey : casey_cannot_determine numbers)
  (h_tracy : tracy_cannot_determine numbers)
  (h_stacy : stacy_cannot_determine numbers)
: numbers.nth 1 = some 4 :=
sorry

end middle_card_is_4_l456_456544


namespace miley_discount_rate_l456_456475

theorem miley_discount_rate :
  let cost_per_cellphone := 800
  let number_of_cellphones := 2
  let amount_paid := 1520
  let total_cost_without_discount := cost_per_cellphone * number_of_cellphones
  let discount_amount := total_cost_without_discount - amount_paid
  let discount_rate := (discount_amount / total_cost_without_discount) * 100
  discount_rate = 5 := by
    sorry

end miley_discount_rate_l456_456475


namespace find_abs_xyz_l456_456461

noncomputable def distinct_nonzero_real (x y z : ℝ) : Prop :=
x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 

theorem find_abs_xyz
  (x y z : ℝ)
  (h1 : distinct_nonzero_real x y z)
  (h2 : x + 1/y = y + 1/z)
  (h3 : y + 1/z = z + 1/x + 1) :
  |x * y * z| = 1 :=
sorry

end find_abs_xyz_l456_456461


namespace range_of_f_l456_456184

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.sqrt (5 + 4 * Real.cos x))

theorem range_of_f :
  Set.range f = Set.Icc (-1/2 : ℝ) (1/2 : ℝ) := 
sorry

end range_of_f_l456_456184


namespace sum_of_digits_of_7_pow_1974_l456_456210

-- Define the number \(7^{1974}\)
def num := 7^1974

-- Function to extract the last two digits
def last_two_digits (n : ℕ) : ℕ := n % 100

-- Function to compute the sum of the tens and units digits
def sum_tens_units (n : ℕ) : ℕ :=
  let last_two := last_two_digits n
  (last_two / 10) + (last_two % 10)

theorem sum_of_digits_of_7_pow_1974 : sum_tens_units num = 9 := by
  sorry

end sum_of_digits_of_7_pow_1974_l456_456210


namespace angle_between_vectors_l456_456705

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Conditions
def condition1 := (∥a - b∥ = 7)
def condition2 := (∥a∥ = 3)
def condition3 := (∥b∥ = 5)

-- Theorem statement
theorem angle_between_vectors (h1 : condition1 a b) (h2 : condition2 a) (h3 : condition3 b) :
  angle a b = 2 * real.pi / 3 :=
by
  sorry

end angle_between_vectors_l456_456705


namespace son_and_daughter_current_ages_l456_456598

theorem son_and_daughter_current_ages
  (father_age_now : ℕ)
  (son_age_5_years_ago : ℕ)
  (daughter_age_5_years_ago : ℝ)
  (h_father_son_birth : father_age_now - (son_age_5_years_ago + 5) = (son_age_5_years_ago + 5))
  (h_father_daughter_birth : father_age_now - (daughter_age_5_years_ago + 5) = (daughter_age_5_years_ago + 5))
  (h_daughter_half_son_5_years_ago : daughter_age_5_years_ago = son_age_5_years_ago / 2) :
  son_age_5_years_ago + 5 = 12 ∧ daughter_age_5_years_ago + 5 = 8.5 :=
by
  sorry

end son_and_daughter_current_ages_l456_456598


namespace range_of_a_l456_456682

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + a * log x

theorem range_of_a (a : ℝ) (h : 0 < a) :
  (∀ (x1 x2 : ℝ), x1 ≠ x2 ∧ x1 ∈ set.Icc 1 3 ∧ x2 ∈ set.Icc 1 3 →
  |f x1 a - f x2 a| < |1 / x1 - 1 / x2|) ↔ 0 < a ∧ a < 8 / 3 :=
sorry

end range_of_a_l456_456682


namespace triangle_isosceles_or_not_l456_456505

-- Define the main elements of the problem
variables {A B C A₁ B₁ A₂ B₂ I : Type}
variables [triangle A B C]

-- Conditions: 
-- 1. Angle bisectors AA₁ and BB₁ intersect at point I.
variable (angle_bisectors_meet : ∃ I, is_angle_bisector A B I ∧ is_angle_bisector B A I)

-- 2. Isosceles triangles are constructed such that A₂, B₂ are on line AB
variable (isosceles_triangles : is_isosceles A₁ I A₂ ∧ is_isosceles B₁ I B₂)

-- 3. Line CI bisects segment A₂ B₂
variable (bisects_CI : ∀ (C I A₂ B₂ : Type) [line C I A₂ B₂],
  bisects C I A₂ B₂)

-- Question: Prove or disprove that triangle ABC is isosceles
theorem triangle_isosceles_or_not :
  ∀ A B C A₁ B₁ A₂ B₂ I,
  angle_bisectors_meet A B C I →
  isosceles_triangles A B C A₁ B₁ A₂ B₂ →
  bisects_CI A B C A₂ B₂ →
  ¬ (isosceles_triangle A B C) :=
begin
  -- Note that the above is a shell for the theorem; the proof part is omitted
  sorry
end

end triangle_isosceles_or_not_l456_456505


namespace range_of_m_F_x2_less_than_x2_minus_1_l456_456037

noncomputable def f (x : ℝ) : ℝ := x + Real.log x
noncomputable def g (x : ℝ) : ℝ := 3 - 2 / x
noncomputable def T (x m : ℝ) : ℝ := Real.log x - x - 2 * m
noncomputable def F (x m : ℝ) : ℝ := x - m / x - 2 * Real.log x
noncomputable def h (t : ℝ) : ℝ := t - 2 * Real.log t - 1

-- (1)
theorem range_of_m (m : ℝ) (h_intersections : ∃ x y : ℝ, T x m = 0 ∧ T y m = 0 ∧ x ≠ y) :
  m < -1 / 2 := sorry

-- (2)
theorem F_x2_less_than_x2_minus_1 {m : ℝ} (h₀ : 0 < m ∧ m < 1) {x₁ x₂ : ℝ} (h₁ : 0 < x₁ ∧ x₁ < x₂)
  (h₂ : F x₁ m = 0 ∧ F x₂ m = 0) :
  F x₂ m < x₂ - 1 := sorry

end range_of_m_F_x2_less_than_x2_minus_1_l456_456037


namespace multiple_of_10_and_12_within_100_l456_456528

theorem multiple_of_10_and_12_within_100 :
  ∀ (n : ℕ), n ≤ 100 → (∃ k₁ k₂ : ℕ, n = 10 * k₁ ∧ n = 12 * k₂) ↔ n = 60 :=
by
  sorry

end multiple_of_10_and_12_within_100_l456_456528


namespace triangle_angle_A_lt_30_l456_456692

theorem triangle_angle_A_lt_30 
  (A B C : Type) [nonempty (Triangle A B C)]
  (BC AC : ℝ) (h : BC / AC < 1/2) 
  : measure_of_angle A < 30 :=
sorry

end triangle_angle_A_lt_30_l456_456692


namespace positive_sum_for_second_figure_exists_l456_456229

theorem positive_sum_for_second_figure_exists
  (F1 F2 : Type) [fintype F1] [fintype F2]
  (g : ℤ × ℤ → ℝ) 
  (P1 : F1 → ℤ × ℤ)
  (P2 : F2 → ℤ × ℤ)
  (h : ∀ (offset : ℤ × ℤ), 0 < ∑ i in finset.univ, g (P1 i + offset)) :
  ∃ offset : ℤ × ℤ, 0 < ∑ j in finset.univ, g (P2 j + offset) :=
sorry

end positive_sum_for_second_figure_exists_l456_456229


namespace yang_tricks_modulo_l456_456932

noncomputable def number_of_tricks_result : Nat :=
  let N := 20000
  let modulo := 100000
  N % modulo

theorem yang_tricks_modulo :
  number_of_tricks_result = 20000 :=
by
  sorry

end yang_tricks_modulo_l456_456932


namespace proof_m_range_l456_456045

open Set

def A : Set ℝ := {x | -3 < x ∧ x < 4}
def B : Set ℝ := {x | -2 < x ∧ x < 2}
def complement_B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}

def P : Set ℝ := {x | (x ∈ A ∩ complement_B) ∧ x ∈ (Set.Of ℤ)}
def Q (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ m + 1}

theorem proof_m_range (m : ℝ)
  (hP_subset : P ⊆ Q m) (hP_card : P.card = 2) : 2 ≤ m ∧ m ≤ 3 :=
  sorry

end proof_m_range_l456_456045


namespace distance_between_parallel_lines_equals_l456_456634

noncomputable def vector (a b : ℝ) := (a, b)
noncomputable def distance_between_lines (a b d : ℝ × ℝ) : ℝ :=
  let v := (b.1 - a.1, b.2 - a.2)
  let d_norm_sq := d.1 * d.1 + d.2 * d.2
  let proj_v_onto_d := ((v.1 * d.1 + v.2 * d.2) / d_norm_sq) * d
  let orthogonal_component := (v.1 - proj_v_onto_d.1, v.2 - proj_v_onto_d.2)
  (orthogonal_component.1 ^ 2 + orthogonal_component.2 ^ 2).sqrt

theorem distance_between_parallel_lines_equals :
  distance_between_lines
    (vector 3 (-2))
    (vector 5 (-8))
    (vector 2 (-14)) = 26 * Real.sqrt 2 / 25 := by
  sorry

end distance_between_parallel_lines_equals_l456_456634


namespace conjugate_of_given_complex_number_l456_456718

noncomputable def given_complex_number : ℂ :=
(2 * complex.I) / (1 + complex.I)

theorem conjugate_of_given_complex_number : complex.conj given_complex_number = 1 - complex.I := 
sorry

end conjugate_of_given_complex_number_l456_456718


namespace domain_of_expression_l456_456643

theorem domain_of_expression (x : ℝ) : 
  (2 * x - 4 ≥ 0) ∧ (9 - 3 * x ≥ 0) ∧ (x - 1 ≥ 0) ∧ (sqrt (9 - 3 * x) + sqrt (x - 1) > 0)
  ↔ (2 ≤ x ∧ x ≤ 3) :=
by
  sorry

end domain_of_expression_l456_456643


namespace area_ratio_is_two_l456_456266

noncomputable def area_ratio_of_octagons (r : ℝ) : ℝ :=
let A := 2 * r^2 * (1 + sqrt(2)) / sqrt(2)
let B := r^2 * (2 + 2 * sqrt(2))
A / B

theorem area_ratio_is_two (r : ℝ) (h : r > 0) :
  area_ratio_of_octagons r = 2 := by
sorry

end area_ratio_is_two_l456_456266


namespace exists_point_D_iff_sin_inequality_l456_456491

-- Define assumptions about triangle ABC
variables {A B C : ℝ} -- Angles A, B, and C in radians
variables (h_triangle : A + B + C = Real.pi) -- Sum of angles in a triangle

-- Main theorem: existence of point D on side AB such that CD is the geometric mean 
-- of AD and DB if and only if the inequality holds.
theorem exists_point_D_iff_sin_inequality
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hA_sum : A + B + C = Real.pi) :
  (∃ D : ℝ, -- Existence of point D (need real definition here based on geometry, but we simplify)
      ∀ (AD DB CD : ℝ),
        CD = Real.sqrt (AD * DB) -- CD is the geometric mean of AD and DB
   ) ↔ (Real.sin A * Real.sin B ≤ Real.sin (C / 2) ^ 2) :=
sorry

end exists_point_D_iff_sin_inequality_l456_456491


namespace find_expression_l456_456381

theorem find_expression (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = 3 * x + 2) : 
  ∀ x : ℤ, f x = 3 * x - 1 :=
sorry

end find_expression_l456_456381


namespace locus_of_midpoints_l456_456448

theorem locus_of_midpoints (L : Type) [metric_space L] (O Q : L) (r : ℝ)
  (h1 : dist Q O = r / 3) (h2 : ∀ P, dist P O ≤ r) :
  ∃ (M : L) (r' : ℝ), (∀ (A B : L), ∃ (M : L), midpoint A B = M → dist M Q = r / 6) :=
sorry

end locus_of_midpoints_l456_456448


namespace height_from_right_angle_to_hypotenuse_l456_456138

theorem height_from_right_angle_to_hypotenuse 
  (a b : ℝ)
  (h_a_pos : 0 < a)
  (h_b_pos : 0 < b)
  (h_angle_C : ∠ A C B = π / 2) :
  let c := Real.sqrt (a^2 + b^2) in
  let CK := (a * b) / c in
  CK = (a * b) / (Real.sqrt (a^2 + b^2)) :=
by
  intros
  sorry

end height_from_right_angle_to_hypotenuse_l456_456138


namespace monotonicity_intervals_range_of_a_l456_456727

noncomputable def f (a x : ℝ) := 3 * a * x - 2 * x^2 + Real.log x
   
theorem monotonicity_intervals (a : ℝ) :
  (∀ x : ℝ, x ∈ set.Ioo (0 : ℝ) 1 → has_deriv_at (f 1) (3 - 4 * x + 1 / x) x ∧ (3 - 4 * x + 1 / x) > 0) ∧
  (∀ x : ℝ, x ∈ set.Ioi 1 → has_deriv_at (f 1) (3 - 4 * x + 1 / x) x ∧ (3 - 4 * x + 1 / x) < 0) := 
sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ set.Icc 1 2 → has_deriv_at (f a) (3 * a - 4 * x + 1 / x) x ∧ (3 * a - 4 * x + 1 / x) ≥ 0) ∨
  (∀ x : ℝ, x ∈ set.Icc 1 2 → has_deriv_at (f a) (3 * a - 4 * x + 1 / x) x ∧ (3 * a - 4 * x + 1 / x) ≤ 0) ↔ 
  (a ≥ 5 / 2 ∨ a ≤ 1) := 
sorry

end monotonicity_intervals_range_of_a_l456_456727


namespace time_addition_correct_l456_456092

def initial_time : Nat × Nat × Nat := (3, 0, 0) -- representing 3:00:00 PM

def added_duration : Nat × Nat × Nat := (85, 58, 30) -- representing 85 hours, 58 minutes, and 30 seconds

theorem time_addition_correct :
  let (A, B, C) := (4, 58, 30) in -- resulting time after addition
  A + B + C = 92 := 
by
  sorry

end time_addition_correct_l456_456092


namespace max_distance_on_circle_to_line_is_8_l456_456812

noncomputable def maxDistancePointToLine : ℝ :=
  let center := (5:ℝ, 3:ℝ)
  let radius := 3
  let distanceFromCenterToLine := (|3 * 5 + 4 * 3 - 2|) / real.sqrt (9 + 16)
  radius + distanceFromCenterToLine

theorem max_distance_on_circle_to_line_is_8 : maxDistancePointToLine = 8 := by
  sorry

end max_distance_on_circle_to_line_is_8_l456_456812


namespace correct_loss_percentage_l456_456616

/-- Define the cost prices and selling prices of the items. -/
def cost_price_radio : ℝ := 4500
def cost_price_television : ℝ := 8000
def cost_price_blender : ℝ := 1300

def selling_price_radio : ℝ := 3200
def selling_price_television : ℝ := 7500
def selling_price_blender : ℝ := 1000

/-- Calculate the total cost price and total selling price. -/
def total_cost_price : ℝ :=
  cost_price_radio + cost_price_television + cost_price_blender

def total_selling_price : ℝ :=
  selling_price_radio + selling_price_television + selling_price_blender

/-- Calculate the overall loss. -/
def overall_loss : ℝ :=
  total_cost_price - total_selling_price

/-- Calculate the loss percentage using the overall loss and total cost price. -/
def loss_percentage : ℝ :=
  (overall_loss / total_cost_price) * 100

/-- The theorem stating the overall loss percentage is approximately 15.22%. -/
theorem correct_loss_percentage : abs (loss_percentage - 15.22) < 0.01 :=
  by
    sorry

end correct_loss_percentage_l456_456616


namespace angle_AMP_eq_2212_l456_456771

noncomputable def midpoint (A B : Point) : Point := 
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

noncomputable def angle (A B C : Point) : Real := sorry -- Placeholder for angle calculation logic

def point_in_segment (A B P : Point) : Prop := 
  (P.x - A.x) * (P.x - B.x) ≤ 0 ∧ (P.y - A.y) * (P.y - B.y) ≤ 0

theorem angle_AMP_eq_2212 {
  A B C D O P Q R S M : Point
  (H1 : A = ⟨0, 0⟩)
  (H2 : B = ⟨1, 0⟩)
  (H3 : C = ⟨1, 1⟩)
  (H4 : D = ⟨0, 1⟩)
  (H5 : O = ⟨0.5, 0.5⟩)
  (HP : point_in_segment A O P)
  (HQ : point_in_segment B O Q)
  (HR : point_in_segment C O R)
  (HS : point_in_segment D O S)
  (Harea : (1 : ℝ) = 2 * (1 - 2 * (P.x)) ^ 2)
  (HM : M = midpoint A B) :
  angle A M P = 22.5 :=
sorry -- Proof to be provided

end angle_AMP_eq_2212_l456_456771


namespace BR_parallel_AC_l456_456805

-- Conditions
variables { Ω : Type* } { O : Type* } { A B C M P Q R : Type* } [Metric Ω]
variables (triangle_ABC : Triangle Ω) 
variables (circumcircle_Ω : Circumcircle Ω triangle_ABC) 
variables (circumcenter_O : Circumcenter Ω triangle_ABC) 
variables (acute_angled : AcuteAngledAngle Ω triangle_ABC)
variables (AB_gt_BC : AB > BC)

-- The angle bisector of ∠ABC intersects Ω at M ≠ B
variables (angle_bisector_ABC : AngleBisector Ω (Angle ABC))
variables (M_on_Ω : OnCircumcircle Ω circumcircle_Ω M)
variables (M_ne_B : M ≠ B)

-- Let Γ be the circle with diameter BM
variables (circle_Γ : CircleWithDiameter Ω B M)

-- The angle bisectors of ∠AOB and ∠BOC intersect Γ at points P and Q, respectively.
variables (angle_bisector_AOB : AngleBisector Ω (Angle AOB))
variables (angle_bisector_BOC : AngleBisector Ω (Angle BOC))
variables (P_on_Γ : OnCircle Ω circle_Γ P)
variables (Q_on_Γ : OnCircle Ω circle_Γ Q)

-- The point R is chosen on line PQ so that BR = MR
variables (R_on_PQ : OnLineSegment Ω P Q R)
variables (BR_eq_MR : Distance Ω B R = Distance Ω M R)

-- Prove BR ∥ AC
theorem BR_parallel_AC : Parallel Ω (LineSegment Ω B R) (LineSegment Ω A C) :=
sorry  -- proof is omitted

end BR_parallel_AC_l456_456805


namespace tan_identity_l456_456679

theorem tan_identity
  (α : ℝ)
  (h : Real.tan (π / 3 - α) = 1 / 3) :
  Real.tan (2 * π / 3 + α) = -1 / 3 := 
sorry

end tan_identity_l456_456679


namespace nonneg_integer_solution_l456_456656

theorem nonneg_integer_solution (a b c : ℕ) (h : 5^a * 7^b + 4 = 3^c) : (a, b, c) = (1, 0, 2) := 
sorry

end nonneg_integer_solution_l456_456656


namespace response_rate_is_correct_l456_456592

-- Define the number of responses needed and the number of questionnaires mailed
def responses_needed : ℕ := 750
def questionnaires_mailed : ℕ := 1250

-- Define the response rate percentage
def response_rate_percentage : ℚ := (responses_needed / questionnaires_mailed) * 100

-- Theorem stating that the response rate percentage is 60%
theorem response_rate_is_correct : response_rate_percentage = 60 := by
  rw [responses_needed, questionnaires_mailed]
  norm_num
  sorry

#eval response_rate_percentage -- 60

end response_rate_is_correct_l456_456592


namespace initial_water_amount_l456_456097

def initial_water := ℕ

variables (x : initial_water) (car_usage per_car water_for_plates_and_clothes : ℕ)
variables (plants_usage remaining_water : ℕ)

-- Conditions
def car_usage_def := car_usage = 7 * 2
def plants_usage_def := plants_usage = car_usage - 11
def remaining_water_def := remaining_water = water_for_plates_and_clothes * 2
def plates_and_clothes_usage := water_for_plates_and_clothes = 24

-- Question
theorem initial_water_amount (h1 : car_usage_def) (h2 : plants_usage_def)
 (h3 : remaining_water_def) (h4 : plates_and_clothes_usage) :
  x = car_usage + plants_usage + remaining_water :=
sorry

end initial_water_amount_l456_456097


namespace prob_range_xi_l456_456029

variable {σ : ℝ}

-- Definition of a normally distributed variable
axiom normal_dist (x : ℝ) : ℝ := sorry  -- representation for the normal distribution

-- Given conditions
axiom xi : ℝ
axiom xi_follows_normal_dist : xi = normal_dist 1
axiom P_xi_gt_3 : ℙ(xi > 3) = 0.023

-- The proof problem statement
theorem prob_range_xi : ℙ(-1 ≤ xi ∧ xi ≤ 3) = 0.954 := 
sorry

end prob_range_xi_l456_456029


namespace probability_even_sum_le_8_l456_456551

theorem probability_even_sum_le_8 (P : ProbabilityMassFunction (ℕ × ℕ)) :
P.support = {x | 1 ≤ x.1 ∧ x.1 ≤ 6 ∧ 1 ≤ x.2 ∧ x.2 ≤ 6} →
P.probability {x | (x.1 + x.2) % 2 = 0 ∧ (x.1 + x.2) ≤ 8} = 1 / 3 :=
by sorry

end probability_even_sum_le_8_l456_456551


namespace evaluate_power_l456_456743

theorem evaluate_power (x : ℝ) (h : 3^(4 * x) = 16) : 27^(x + 1) = 432 := 
by
  sorry

end evaluate_power_l456_456743


namespace no_four_points_all_triangles_acute_l456_456490

theorem no_four_points_all_triangles_acute :
  ∀ A B C D : ℝ × ℝ,
  ¬ (Tetrahedron ABCD ∧ (Triangle ABC).is_acute ∧ (Triangle BCD).is_acute ∧ (Triangle CDA).is_acute ∧ (Triangle DAB).is_acute) :=
by sorry

end no_four_points_all_triangles_acute_l456_456490


namespace rectangular_coordinates_of_new_point_l456_456262

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ := 
  let r := real.sqrt (x^2 + y^2)
  let theta := real.atan2 y x
  (r, theta)

def new_coordinates (x y: ℝ) : ℝ × ℝ :=
  let (r, theta) := rectangular_to_polar x y
  let r3 := r^3
  let theta3 := 3 * theta
  let cos3θ := 4 * (real.cos theta ^ 3) - 3 * (real.cos theta)
  let sin3θ := 3 * (real.sin theta) - 4 * (real.sin theta ^ 3)
  (r3 * cos3θ, r3 * sin3θ)

theorem rectangular_coordinates_of_new_point :
  new_coordinates 5 (-12) = (305, 3756) :=
sorry

end rectangular_coordinates_of_new_point_l456_456262


namespace find_adult_fee_l456_456165

-- Define the constants and variables according to the problem conditions
def children_fee : ℝ := 1.5
def total_people : ℕ := 315
def total_fee_collected : ℝ := 810
def children_admitted : ℕ := 180

-- Define the function for the admission fee for adults to be proved
def admission_fee_for_adults (A : ℝ) : Prop :=
  let adults_admitted := total_people - children_admitted in
  let total_children_fee := children_admitted * children_fee in
  total_children_fee + (adults_admitted * A) = total_fee_collected

-- The theorem to prove
theorem find_adult_fee : admission_fee_for_adults 4 :=
  sorry

end find_adult_fee_l456_456165


namespace jill_total_tax_percent_l456_456573

variable (total_spent : ℝ)
variable (clothing_percent food_percent other_percent : ℝ)
variable (tax_clothing tax_food tax_other : ℝ)

def tax_paid : ℝ :=
  (clothing_percent * tax_clothing) + (food_percent * tax_food) + (other_percent * tax_other)

def total_tax_percent := (tax_paid total_spent clothing_percent food_percent other_percent tax_clothing tax_food tax_other / total_spent) * 100

theorem jill_total_tax_percent (h1 : clothing_percent = 0.5)
                               (h2 : food_percent = 0.2)
                               (h3 : other_percent = 0.3)
                               (h4 : tax_clothing = 0.04)
                               (h5 : tax_food = 0)
                               (h6 : tax_other = 0.08)
                               (h7 : total_spent = 100) :
  total_tax_percent total_spent clothing_percent food_percent other_percent tax_clothing tax_food tax_other = 4.4 := 
  sorry

end jill_total_tax_percent_l456_456573


namespace cos_double_angle_l456_456745

theorem cos_double_angle (α : ℝ) (h : Real.sin α = (Real.sqrt 3) / 2) : 
  Real.cos (2 * α) = -1 / 2 :=
by
  sorry

end cos_double_angle_l456_456745


namespace whale_ninth_hour_consumption_l456_456979

-- Define the arithmetic sequence conditions
def first_hour_consumption : ℕ := 10
def common_difference : ℕ := 5

-- Define the total consumption over 12 hours
def total_consumption := 12 * (first_hour_consumption + (first_hour_consumption + 11 * common_difference)) / 2

-- Prove the ninth hour (which is the 8th term) consumption
theorem whale_ninth_hour_consumption :
  total_consumption = 450 →
  first_hour_consumption + 8 * common_difference = 50 := 
by
  intros h
  sorry
  

end whale_ninth_hour_consumption_l456_456979


namespace least_integer_lines_l456_456948

theorem least_integer_lines (
  n : ℕ
  lines_in_2D_plane : set (set ℝ) -- Considering lines are sets of ℝ representing their directional vectors
  lines_parallel : ∀ (line1 line2 : set ℝ), line1 ∥ line2 -- ∥ to represent parallelism
  lines_nonparallel : ∀ (line1 line2 : set ℝ), ¬(line1 ∥ line2)
  n_lines : ∀ (s : set (set ℝ)), (|s| = n) -- n lines in the set
) : 
  (∃ subset_parallel : set (set ℝ), (|subset_parallel| = 1001) ∧ (∀ line1 ∈ subset_parallel, ∀ line2 ∈ subset_parallel, line1 ∥ line2)) 
  ∨ 
  (∃ subset_nonparallel : set (set ℝ), (|subset_nonparallel| = 1001) ∧ (∀ line1 ∈ subset_nonparallel, ∀ line2 ∈ subset_nonparallel, ¬(line1 ∥ line2)))  
  ∧ n = 1000001 := sorry

end least_integer_lines_l456_456948


namespace paul_sandwiches_l456_456484

theorem paul_sandwiches (sandwiches_day1 sandwiches_day2 sandwiches_day3 total_sandwiches_3days total_sandwiches_6days : ℕ) 
    (h1 : sandwiches_day1 = 2) 
    (h2 : sandwiches_day2 = 2 * sandwiches_day1) 
    (h3 : sandwiches_day3 = 2 * sandwiches_day2) 
    (h4 : total_sandwiches_3days = sandwiches_day1 + sandwiches_day2 + sandwiches_day3) 
    (h5 : total_sandwiches_6days = 2 * total_sandwiches_3days) 
    : total_sandwiches_6days = 28 := 
by 
    sorry

end paul_sandwiches_l456_456484


namespace function_symmetry_about_pi_l456_456035

def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x - b * Real.cos x

theorem function_symmetry_about_pi
  (a b : ℝ) (h : a ≠ 0) (minimum_at_pi_over_4 : ∀ x, f a b x = a * Real.sin x - b * Real.cos x → x = π / 4)
  : (∀ x, f a b (3 * π / 4 - x) = -f a b x) ∧ (∃ p : ℝ × ℝ, p = (π, 0)) :=
by
  sorry

end function_symmetry_about_pi_l456_456035


namespace range_of_a_l456_456413

theorem range_of_a : (∀ x : ℝ, x^2 + (a-1)*x + 1 > 0) ↔ (-1 < a ∧ a < 3) := by
  sorry

end range_of_a_l456_456413


namespace incorrect_monotonicity_l456_456033

noncomputable def f (x : ℝ) : ℝ := real.sqrt 3 * real.sin (2 * x) - real.cos (2 * x) + 1

theorem incorrect_monotonicity : ¬ (∀ x y : ℝ, (5 * real.pi / 12) < x ∧ x < y ∧ y < (11 * real.pi / 12) → f y ≤ f x) :=
sorry

end incorrect_monotonicity_l456_456033


namespace base_seven_representation_l456_456329

theorem base_seven_representation 
  (k : ℕ) 
  (h1 : 4 ≤ k) 
  (h2 : k < 8) 
  (h3 : 500 / k^3 < k) 
  (h4 : 500 ≥ k^3) 
  : ∃ n m o p : ℕ, (500 = n * k^3 + m * k^2 + o * k + p) ∧ (p % 2 = 1) ∧ (n ≠ 0 ) :=
sorry

end base_seven_representation_l456_456329


namespace line_intersects_y_axis_at_5_2_l456_456963

noncomputable def line_y_intercept : ℝ :=
  let point1 := (3 : ℝ, 10 : ℝ)
  let point2 := (-7 : ℝ, -6 : ℝ)
  let m := (point2.2 - point1.2) / (point2.1 - point1.1)
  point1.2 - m * point1.1

theorem line_intersects_y_axis_at_5_2 :
  let y_intercept := line_y_intercept
  y_intercept = 5.2 :=
by
  sorry

end line_intersects_y_axis_at_5_2_l456_456963


namespace average_speed_correct_l456_456261

variable (d : ℝ) (north_time south_time total_time total_distance : ℝ)

-- defining the conditions
def north_time := 4 * d -- time to travel north in minutes
def south_time := d / 3 -- time to travel south in minutes
def total_time := (north_time + south_time) / 60 -- total time for trip in hours
def total_distance := 2 * d -- total distance for the trip in kilometers

-- proving the average speed
theorem average_speed_correct :
  (2 * d) / total_time = 360 / 13 := by
  -- Lean proof would go here
  sorry

end average_speed_correct_l456_456261


namespace b_minus_c_eq_log_8192_1_div_42_l456_456326

open Real

noncomputable def a_n (n : ℕ) (h : n > 1) : ℝ := 1 / log n 8192

def b : ℝ := a_n 3 (by norm_num) + a_n 4 (by norm_num)
def c : ℝ := a_n 7 (by norm_num) + a_n 8 (by norm_num) + a_n 9 (by norm_num)

theorem b_minus_c_eq_log_8192_1_div_42 : b - c = log 8192 (1 / 42) :=
sorry

end b_minus_c_eq_log_8192_1_div_42_l456_456326


namespace radical_axis_through_intersection_l456_456151

-- Define the structure for a circle
structure Circle :=
(center : Point)
(radius : ℝ)

-- Define a point
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the power of a point with respect to a circle
def power_of_point (P : Point) (ω : Circle) : ℝ :=
  (P.x - ω.center.x)^2 + (P.y - ω.center.y)^2 - ω.radius^2

-- The radical axis is the locus of points where the power with respect to two circles are equal
def on_radical_axis (P : Point) (ω1 ω2 : Circle) : Prop :=
  power_of_point P ω1 = power_of_point P ω2

-- Given conditions that circles intersect at points A and B
def circles_intersect (ω1 ω2 : Circle) (A B : Point) : Prop :=
  (A.x - ω1.center.x)^2 + (A.y - ω1.center.y)^2 = ω1.radius^2 ∧
  (A.x - ω2.center.x)^2 + (A.y - ω2.center.y)^2 = ω2.radius^2 ∧
  (B.x - ω1.center.x)^2 + (B.y - ω1.center.y)^2 = ω1.radius^2 ∧
  (B.x - ω2.center.x)^2 + (B.y - ω2.center.y)^2 = ω2.radius^2

-- Lean statement
theorem radical_axis_through_intersection (ω1 ω2 : Circle) (A B : Point)
  (h : circles_intersect ω1 ω2 A B) :
  on_radical_axis A ω1 ω2 ∧ on_radical_axis B ω1 ω2 :=
begin
  sorry,
end

end radical_axis_through_intersection_l456_456151


namespace complementary_card_sets_count_eq_1638_l456_456306

-- Define the conditions
def is_complementary (shapes : List ℕ) (colors : List ℕ) (shades : List ℕ) : Prop :=
  (shapes.nodup ∨ shapes.all_same) ∧
  (colors.nodup ∨ colors.all_same) ∧
  (shades.nodup ∨ shades.all_same)

-- Define the necessary functions for lists
def List.nodup {α : Type*} [DecidableEq α] (l : List α) : Prop :=
  ∀ a, a ∈ l → l.erase a = l

def List.all_same {α : Type*} [DecidableEq α] (l : List α) : Prop :=
  ∀ a, a ∈ l → l = l.replicate l.length a

-- Proof statement
theorem complementary_card_sets_count_eq_1638 :
  (∑ (shapes : List ℕ) in List.permutations [1, 2, 3], -- Permutations of 3 shapes
   ∑ (colors : List ℕ) in List.permutations [1, 2, 3], -- Permutations of 3 colors
   ∑ (shades : List ℕ) in List.permutations [1, 2, 3, 4], -- Permutations of 4 shades
   if is_complementary shapes colors shades then 1 else 0) = 1638 :=
sorry

end complementary_card_sets_count_eq_1638_l456_456306


namespace cuboid_surface_area_l456_456556

def width := 8 -- cm
def length := 5 -- cm
def height := 10 -- cm

def surface_area_of_cuboid (w l h : ℝ) : ℝ :=
  2 * (w * l + w * h + l * h)

theorem cuboid_surface_area :
  surface_area_of_cuboid width length height = 340 := by
  sorry

end cuboid_surface_area_l456_456556


namespace train_speed_l456_456612

theorem train_speed (length : ℝ) (time : ℝ) (conversion_factor : ℝ)
  (h1 : length = 500) (h2 : time = 5) (h3 : conversion_factor = 3.6) :
  (length / time) * conversion_factor = 360 :=
by
  sorry

end train_speed_l456_456612


namespace possible_pairs_of_m_and_n_l456_456575

open Nat

def contains_digits_4_and_5 (n : ℕ) : Prop :=
  -- Predicate to check if the number contains digits 4 and 5
  toString n |>.any (λ c => c = '4') ∧ toString n |>.any (λ c => c = '5')

theorem possible_pairs_of_m_and_n :
  ∀ (m n : ℕ),
    1000 ≤ m ∧ m < 10000 ∧
    100 ≤ n ∧ n < 1000 ∧
    contains_digits_4_and_5 m ∧
    contains_digits_4_and_5 n ∧
    59 ∣ m ∧
    n % 38 = 1 ∧
    abs (m - n) ≤ 2015 →
  (m, n) = (1475, 457) ∨ (m, n) = (1534, 457) :=
by
  sorry  -- Proof is omitted

end possible_pairs_of_m_and_n_l456_456575


namespace percentage_discount_l456_456474

-- Define the given conditions
def equal_contribution (total: ℕ) (num_people: ℕ) := total / num_people

def original_contribution (amount_paid: ℕ) (discount: ℕ) := amount_paid + discount

def total_original_cost (individual_original: ℕ) (num_people: ℕ) := individual_original * num_people

def discount_amount (original_cost: ℕ) (discounted_cost: ℕ) := original_cost - discounted_cost

def discount_percentage (discount: ℕ) (original_cost: ℕ) := (discount * 100) / original_cost

-- Given conditions
def given_total := 48
def given_num_people := 3
def amount_paid_each := equal_contribution given_total given_num_people
def discount_each := 4
def original_payment_each := original_contribution amount_paid_each discount_each
def original_total_cost := total_original_cost original_payment_each given_num_people
def paid_total := 48

-- Question: What is the percentage discount
theorem percentage_discount :
  discount_percentage (discount_amount original_total_cost paid_total) original_total_cost = 20 :=
by
  sorry

end percentage_discount_l456_456474


namespace find_longest_side_length_l456_456882

-- Define the conditions as assumptions
variables (a : ℝ) (triangle : Type)
structure is_median (line : ℝ → ℝ) (v1 v2 : triangle) : Prop :=
  (holds: ∃ x : ℝ, line x = x * (v1.1 + v2.1) / 2 + (v1.2 + v2.2) / 2 )

structure is_centroid (v1 v2 v3 : triangle) : Prop :=
  (centroid: (v1.1 + v2.1 + v3.1) / 3 = (v1.2 + v2.2 + v3.2) / 3)

-- Define the vertices and side lengths
variables (v1 v2 v3 : triangle)
def side_length (v1 v2 : triangle) : ℝ :=
  real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2)

-- Main proof problem translated to Lean
theorem find_longest_side_length :
  ∀ (a : ℝ) (v1 v2 v3 : triangle),
    is_centroid v1 v2 v3 ∧
    is_median (λ x: ℝ, x) v1 v2 ∧
    is_median (λ x: ℝ, 2*x) v2 v3 ∧
    is_median (λ x: ℝ, 3*x) v3 v1 ∧
    (side_length v1 v2 + side_length v2 v3 + side_length v3 v1 = 1) →
    ∃ longest_side : ℝ,
      longest_side = real.sqrt ((v1.1 - v3.1)^2 + (v1.2 - v3.2)^2) ∧
      longest_side = real.sqrt (58) * (1 / (2 + real.sqrt (34) + real.sqrt (58))) :=
  sorry

end find_longest_side_length_l456_456882


namespace johns_average_score_increase_l456_456796

theorem johns_average_score_increase :
  let score1 := 92
  let score2 := 89
  let score3 := 93
  let score4 := 95
  let initial_average := (score1 + score2 + score3) / 3
  let new_average := (score1 + score2 + score3 + score4) / 4
  new_average - initial_average = 0.92 :=
by
  let score1 := 92
  let score2 := 89
  let score3 := 93
  let score4 := 95
  let initial_average := (score1 + score2 + score3 : ℝ) / 3
  let new_average := (score1 + score2 + score3 + score4 : ℝ) / 4
  have : new_average - initial_average = (92 + 89 + 93 + 95)/4 - (92 + 89 + 93)/3 := by sorry
  show _ = 0.92 from by sorry

end johns_average_score_increase_l456_456796


namespace net_gain_for_mr_A_l456_456834

open Real

noncomputable def initial_value : ℝ := 15000
noncomputable def sale_1_profit : ℝ := 0.20
noncomputable def buy_1_loss : ℝ := 0.15
noncomputable def sale_2_profit : ℝ := 0.10
noncomputable def buy_2_loss : ℝ := 0.05

theorem net_gain_for_mr_A : 
  let value_after_sale_1 := initial_value * (1 + sale_1_profit),
      value_after_buy_1 := value_after_sale_1 * (1 - buy_1_loss),
      value_after_sale_2 := value_after_buy_1 * (1 + sale_2_profit),
      value_after_buy_2 := value_after_sale_2 * (1 - buy_2_loss),
      total_income := value_after_sale_1 + value_after_sale_2,
      total_expense := value_after_buy_1 + value_after_buy_2,
      net_gain := total_income - total_expense
  in net_gain = 3541.50 := by
  sorry

end net_gain_for_mr_A_l456_456834


namespace water_in_pool_l456_456842

noncomputable def total_water_added (initial_buckets : ℝ) (additional_buckets : ℝ) (liters_per_bucket : ℝ) : ℝ :=
  (initial_buckets + additional_buckets) * liters_per_bucket

noncomputable def total_loss (evaporation_rate : ℝ) (splashing_rate : ℝ) (time_minutes : ℝ) : ℝ :=
  (evaporation_rate + splashing_rate) * time_minutes

noncomputable def net_water (initial_buckets : ℝ) (additional_buckets : ℝ) (liters_per_bucket : ℝ)
  (evaporation_rate : ℝ) (splashing_rate : ℝ) (time_minutes : ℝ) : ℝ :=
  total_water_added initial_buckets additional_buckets liters_per_bucket - total_loss evaporation_rate splashing_rate time_minutes

theorem water_in_pool (initial_buckets : ℝ) (additional_buckets : ℝ) (liters_per_bucket : ℝ)
  (evaporation_rate : ℝ) (splashing_rate : ℝ) (time_minutes : ℝ) :
  net_water initial_buckets additional_buckets liters_per_bucket evaporation_rate splashing_rate time_minutes = 84 :=
by
  have h_total_water_added : total_water_added initial_buckets additional_buckets liters_per_bucket = 98 := by calc
    total_water_added initial_buckets additional_buckets liters_per_bucket 
      = (1 + 8.8) * 10 := rfl
      ... = 98 := rfl
  have h_total_loss : total_loss evaporation_rate splashing_rate time_minutes = 14 := by calc
    total_loss evaporation_rate splashing_rate time_minutes 
      = (0.2 + 0.5) * 20 := rfl
      ... = 14 := rfl
  have h_net_water : net_water initial_buckets additional_buckets liters_per_bucket evaporation_rate splashing_rate time_minutes 
    = 98 - 14 := rfl
  show net_water initial_buckets additional_buckets liters_per_bucket evaporation_rate splashing_rate time_minutes = 84
  ... = 84 := by
    calc
      net_water initial_buckets additional_buckets liters_per_bucket evaporation_rate splashing_rate time_minutes 
      = 98 - 14 := h_net_water
      ... = 84 := rfl

end water_in_pool_l456_456842


namespace radical_axis_of_intersecting_circles_l456_456150

-- Definition of the power of a point with respect to a circle
def power_of_point (A : Point) (O : Point) (R : ℝ) : ℝ :=
  (dist A O)^2 - R^2

-- Definition of the radical axis condition
def radical_axis_condition (P Q : Point) (O1 O2 : Point) (R1 R2 : ℝ) : Prop :=
  power_of_point P O1 R1 = 0 ∧ power_of_point P O2 R2 = 0 ∧ 
  power_of_point Q O1 R1 = 0 ∧ power_of_point Q O2 R2 = 0

-- The Lean statement proving the radical axis passes through points of intersection
theorem radical_axis_of_intersecting_circles 
  (O1 O2 P Q : Point) (R1 R2 : ℝ) 
  (h1 : dist P O1 = R1) (h2 : dist P O2 = R2)
  (h3 : dist Q O1 = R1) (h4 : dist Q O2 = R2) :
  radical_axis_condition P Q O1 O2 R1 R2 :=
by {
  sorry
}

end radical_axis_of_intersecting_circles_l456_456150


namespace grogg_possible_cubes_l456_456397

theorem grogg_possible_cubes (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_prob : (a - 2) * (b - 2) * (c - 2) / (a * b * c) = 1 / 5) :
  a * b * c = 120 ∨ a * b * c = 160 ∨ a * b * c = 240 ∨ a * b * c = 360 := 
sorry

end grogg_possible_cubes_l456_456397


namespace ratio_areas_of_octagons_l456_456271

noncomputable def area_inscribed_octagon (r : ℝ) : ℝ := 4 * r^2 * (√2 - 1) 

noncomputable def area_circumscribed_octagon (r : ℝ) : ℝ := 2 * r^2 * (1 + √2)

theorem ratio_areas_of_octagons {r : ℝ} (hr : r > 0) :
  (area_circumscribed_octagon r) / (area_inscribed_octagon r) = 2 :=
by sorry

end ratio_areas_of_octagons_l456_456271


namespace triangle_sides_fraction_sum_eq_one_l456_456758

theorem triangle_sides_fraction_sum_eq_one
  (a b c : ℝ)
  (h : a^2 + b^2 = c^2 + a * b) :
  a / (b + c) + b / (c + a) = 1 :=
sorry

end triangle_sides_fraction_sum_eq_one_l456_456758


namespace probability_f_ge_0_l456_456034

-- Definition of the function and interval
def f (x : ℝ) : ℝ := Real.log x / Real.log 2 -- log base 2

def interval := Set.Icc (1/2 : ℝ) (2 : ℝ)

-- Prove that the probability of f(x₀) ≥ 0 for x₀ in the interval is 2/3
theorem probability_f_ge_0 (x₀ : ℝ) (hx₀ : x₀ ∈ interval) : (∃ P : ℝ, P = 2/3) :=
by
  sorry

end probability_f_ge_0_l456_456034


namespace time_per_pan_is_7_l456_456791

theorem time_per_pan_is_7 (t : ℕ) (n : ℕ) (h_t : t = 28) (h_n : n = 4) : 
  let time_per_pan := t / n in
  time_per_pan = 7 := by
  simp [h_t, h_n]
  sorry

end time_per_pan_is_7_l456_456791


namespace magnitude_of_complex_l456_456651

open Complex

theorem magnitude_of_complex :
  abs (Complex.mk (2/3) (-4/5)) = Real.sqrt 244 / 15 :=
by
  -- Placeholder for the actual proof
  sorry

end magnitude_of_complex_l456_456651


namespace square_area_l456_456620

def edge1 (x : ℝ) := 5 * x - 18
def edge2 (x : ℝ) := 27 - 4 * x
def x_val : ℝ := 5

theorem square_area : edge1 x_val = edge2 x_val → (edge1 x_val) ^ 2 = 49 :=
by
  intro h
  -- Proof required here
  sorry

end square_area_l456_456620


namespace charcoal_amount_l456_456098

theorem charcoal_amount (water_per_charcoal : ℕ) (charcoal_ratio : ℕ) (water_added : ℕ) (charcoal_needed : ℕ) 
  (h1 : water_per_charcoal = 30) (h2 : charcoal_ratio = 2) (h3 : water_added = 900) : charcoal_needed = 60 :=
by
  sorry

end charcoal_amount_l456_456098


namespace tom_initial_game_count_zero_l456_456197

theorem tom_initial_game_count_zero
  (batman_game_cost superman_game_cost total_expenditure initial_game_count : ℝ)
  (h_batman_cost : batman_game_cost = 13.60)
  (h_superman_cost : superman_game_cost = 5.06)
  (h_total_expenditure : total_expenditure = 18.66)
  (h_initial_game_cost : initial_game_count = total_expenditure - (batman_game_cost + superman_game_cost)) :
  initial_game_count = 0 :=
by
  sorry

end tom_initial_game_count_zero_l456_456197


namespace part1_part2_l456_456808

noncomputable def A : set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def B (m : ℝ) : set ℝ := {x | x^2 + 2 * (m + 1) * x + m^2 - 1 = 2}

theorem part1 (m : ℝ) : (A ∩ B m ≠ ∅) → m = -Real.sqrt 3 :=
by
  sorry

theorem part2 (m : ℝ) : (A ∪ B m = B m) → m = -Real.sqrt 3 :=
by
  sorry

end part1_part2_l456_456808


namespace emma_final_amount_l456_456310

theorem emma_final_amount
  (initial_amount : ℕ)
  (furniture_cost : ℕ)
  (fraction_given_to_anna : ℚ)
  (amount_left : ℕ) :
  initial_amount = 2000 →
  furniture_cost = 400 →
  fraction_given_to_anna = 3 / 4 →
  amount_left = initial_amount - furniture_cost →
  amount_left - (fraction_given_to_anna * amount_left : ℚ) = 400 :=
by
  intros h_initial h_furniture h_fraction h_amount_left
  sorry

end emma_final_amount_l456_456310


namespace initial_deposit_l456_456835

-- Define the conditions given in the problem
def A : ℝ := 181.50
def r : ℝ := 0.20
def n : ℕ := 2
def t : ℕ := 1
def compound_interest (P : ℝ) : ℝ := P * (1 + r / n) ^ (n * t)

-- The main theorem to prove: the initial deposit amount
theorem initial_deposit : ∃ P : ℝ, compound_interest P = A ∧ P = 150 :=
by 
  use 150
  simp [compound_interest, A, r, n, t]
  norm_num
  sorry

end initial_deposit_l456_456835


namespace amount_spent_on_shorts_l456_456493

def amount_spent_on_shirt := 12.14
def amount_spent_on_jacket := 7.43
def total_amount_spent_on_clothes := 33.56

theorem amount_spent_on_shorts : total_amount_spent_on_clothes - amount_spent_on_shirt - amount_spent_on_jacket = 13.99 :=
by
  sorry

end amount_spent_on_shorts_l456_456493


namespace notched_circle_coordinates_l456_456966

variable (a b : ℝ)

theorem notched_circle_coordinates : 
  let sq_dist_from_origin := a^2 + b^2
  let A := (a, b + 5)
  let C := (a + 3, b)
  (a^2 + (b + 5)^2 = 36 ∧ (a + 3)^2 + b^2 = 36) →
  (sq_dist_from_origin = 4.16 ∧ A = (2, 5.4) ∧ C = (5, 0.4)) :=
by
  sorry

end notched_circle_coordinates_l456_456966


namespace B_monthly_income_is_correct_l456_456885

variable (A_m B_m C_m : ℝ)
variable (A_annual C_m_value : ℝ)
variable (ratio_A_to_B : ℝ)

-- Given conditions
def conditions :=
  A_annual = 537600 ∧
  C_m_value = 16000 ∧
  ratio_A_to_B = 5 / 2 ∧
  A_m = A_annual / 12 ∧
  B_m = (2 / 5) * A_m ∧
  B_m = 1.12 * C_m ∧
  C_m = C_m_value

-- Prove that B's monthly income is Rs. 17920
theorem B_monthly_income_is_correct (h : conditions A_m B_m C_m A_annual C_m_value ratio_A_to_B) : 
  B_m = 17920 :=
by 
  sorry

end B_monthly_income_is_correct_l456_456885


namespace find_a_l456_456367

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x > 0 then Real.log x else a + ∫ t in 0..x, 1 - Real.cos t

theorem find_a (a : ℝ) (h: f a (f a 1) = 2) : a = 2 :=
by
  sorry

end find_a_l456_456367


namespace alcohol_percentage_approximation_l456_456237

def percentage_of_alcohol_in_new_mixture (original_volume_alcohol original_volume_mixture added_volume_water : ℕ) : ℚ :=
  let total_volume := original_volume_mixture + added_volume_water
  let alcohol_percentage := (original_volume_alcohol : ℚ) / (total_volume : ℚ) * 100
  alcohol_percentage

theorem alcohol_percentage_approximation :
  percentage_of_alcohol_in_new_mixture 4 20 3 ≈ 17.39 := by
  sorry

end alcohol_percentage_approximation_l456_456237


namespace olivia_did_not_sell_4_bars_l456_456307

-- Define the constants and conditions
def price_per_bar : ℕ := 3
def total_bars : ℕ := 7
def money_made : ℕ := 9

-- Calculate the number of bars sold
def bars_sold : ℕ := money_made / price_per_bar

-- Calculate the number of bars not sold
def bars_not_sold : ℕ := total_bars - bars_sold

-- Theorem to prove the answer
theorem olivia_did_not_sell_4_bars : bars_not_sold = 4 := 
by 
  sorry

end olivia_did_not_sell_4_bars_l456_456307


namespace diff_of_squares_l456_456117

theorem diff_of_squares (a b : ℕ) : 
  (∃ x y : ℤ, a = x^2 - y^2) ∨ (∃ x y : ℤ, b = x^2 - y^2) ∨ (∃ x y : ℤ, a + b = x^2 - y^2) :=
sorry

end diff_of_squares_l456_456117


namespace smaller_circle_radius_l456_456635

noncomputable def radius_of_smaller_circles (R : ℝ) (r1 r2 r3 : ℝ) (OA OB OC : ℝ) : Prop :=
(OA = R + r1) ∧ (OB = R + 3 * r1) ∧ (OC = R + 5 * r1) ∧ 
((OB = OA + 2 * r1) ∧ (OC = OB + 2 * r1))

theorem smaller_circle_radius (r : ℝ) (R : ℝ := 2) :
  radius_of_smaller_circles R r r r (R + r) (R + 3 * r) (R + 5 * r) → r = 1 :=
by
  sorry

end smaller_circle_radius_l456_456635


namespace count_factors_of_product_l456_456637

theorem count_factors_of_product :
  let n := 8^4 * 7^3 * 9^1 * 5^5
  ∃ (count : ℕ), count = 936 ∧ 
    ∀ f : ℕ, f ∣ n → ∃ a b c d : ℕ,
      a ≤ 12 ∧ b ≤ 2 ∧ c ≤ 5 ∧ d ≤ 3 ∧ 
      f = 2^a * 3^b * 5^c * 7^d :=
by sorry

end count_factors_of_product_l456_456637


namespace zero_cleverly_numbers_l456_456584

theorem zero_cleverly_numbers (n : ℕ) : 
  (1000 ≤ n ∧ n < 10000) ∧ (∃ a b c, n = 1000 * a + 10 * b + c ∧ b = 0 ∧ 9 * (100 * a + 10 * b + c) = n) ↔ (n = 2025 ∨ n = 4050 ∨ n = 6075) := 
sorry

end zero_cleverly_numbers_l456_456584


namespace cylindrical_pipe_height_l456_456962

theorem cylindrical_pipe_height (r_outer r_inner : ℝ) (SA : ℝ) (h : ℝ) 
  (h_outer : r_outer = 5)
  (h_inner : r_inner = 3)
  (h_SA : SA = 50 * Real.pi)
  (surface_area_eq: SA = 2 * Real.pi * (r_outer + r_inner) * h) : 
  h = 25 / 8 := 
by
  {
    sorry
  }

end cylindrical_pipe_height_l456_456962


namespace largest_number_among_four_l456_456282

theorem largest_number_among_four :
  let a := -3
  let b := 0
  let c := 2
  let d := Real.sqrt 5
  d > c ∧ c > b ∧ b > a → ∀ x ∈ {a, b, c, d}, x ≤ d := 
by 
  intros
  simp only [d] 
  simp only [a, b, c, d] 
  sorry

end largest_number_among_four_l456_456282


namespace find_ab_l456_456021

theorem find_ab (a b : ℝ) (h1 : a + b = 4) (h2 : a^3 + b^3 = 136) : a * b = -6 :=
by
  sorry

end find_ab_l456_456021


namespace larry_gave_brother_money_l456_456439

theorem larry_gave_brother_money : 
  ∀ (initial_money lunch_cost current_money gave_brother : ℕ),
    initial_money = 22 →
    lunch_cost = 5 →
    current_money = 15 →
    (initial_money - lunch_cost) - current_money = gave_brother →
    gave_brother = 2 := 
by {
  intros initial_money lunch_cost current_money gave_brother 
         h_initial_money h_lunch_cost h_current_money h_calc,
  sorry
}

end larry_gave_brother_money_l456_456439


namespace maximize_product_of_roots_l456_456719

theorem maximize_product_of_roots :
  ∀ (k : ℝ),
    (∃ x y : ℝ, 3 * x^2 - 4 * x + k = 0 ∧ 3 * y^2 - 4 * y + k = 0 ∧ x ≠ y) →
    k ≤ 4 / 3 → 
    (∀ k' : ℝ, (k' ≤ 4 / 3) → (k' / 3) ≤ (4 / 3 / 3)) :=
begin
  sorry
end

end maximize_product_of_roots_l456_456719


namespace resulting_angle_at_E_l456_456785

-- Define Given Conditions
variables (A B C D E : Type)
variables (AD BC : A → Prop)
variables [parallel AD BC] (AB CD : ℝ)
variable (AB_length : AB = 6)
variable (BC_length : BC = 7)
variable (CD_length : CD = 8)
variable (AD_length : AD = 17)

-- Define the proposition to check angle at E
def angle_at_E_extended := 90

-- Theorem: Resulting angle at E when sides AB and CD are extended to meet is 90 degrees
theorem resulting_angle_at_E
  (AD_parallel_BC : parallel AD BC) 
  (AB_eq_6 : AB = 6)
  (BC_eq_7 : BC = 7)
  (CD_eq_8 : CD = 8)
  (AD_eq_17 : AD = 17) 
  (angle_E_extended : ∀ E, ∃ ⦃E : Type⦄, angle_at_E_extended E = 90) : angle_at_E_extended = 90 :=
sorry

end resulting_angle_at_E_l456_456785


namespace number_of_people_liking_at_least_one_activity_l456_456767

def total_people := 200
def people_like_books := 80
def people_like_songs := 60
def people_like_movies := 30
def people_like_books_and_songs := 25
def people_like_books_and_movies := 15
def people_like_songs_and_movies := 20
def people_like_all_three := 10

theorem number_of_people_liking_at_least_one_activity :
  total_people = 200 →
  people_like_books = 80 →
  people_like_songs = 60 →
  people_like_movies = 30 →
  people_like_books_and_songs = 25 →
  people_like_books_and_movies = 15 →
  people_like_songs_and_movies = 20 →
  people_like_all_three = 10 →
  (people_like_books + people_like_songs + people_like_movies -
   people_like_books_and_songs - people_like_books_and_movies -
   people_like_songs_and_movies + people_like_all_three) = 120 := sorry

end number_of_people_liking_at_least_one_activity_l456_456767


namespace minimum_value_a_l456_456012

variables {m n r a : ℕ} (grid : fin m → fin n → Prop)

-- Conditions
def positive_integers (m n r : ℕ) : Prop := m > 0 ∧ n > 0 ∧ r > 0
def order_conditions (m n r : ℕ) : Prop := 1 ≤ r ∧ r ≤ m ∧ m ≤ n
def at_most_r_red_squares (grid : fin m → fin n → Prop) (r : ℕ) : Prop := 
  ∀ i : fin m, (finset.univ.filter (λ j, grid i j)).card ≤ r ∧ 
  ∀ j : fin n, (finset.univ.filter (λ i, grid i j)).card ≤ r

-- Theorem statement
theorem minimum_value_a (h1 : positive_integers m n r) (h2 : order_conditions m n r) (h3 : at_most_r_red_squares grid r) :
  ∃ a, (∀ color_scheme, ∃ (diagonals : fin a → List (fin m × fin n)), 
  (∀ i, i < m → ∀ j, j < n → ∃ k, ((i, j) ∈ diagonals k)) ∧ a = r) :=
sorry

end minimum_value_a_l456_456012


namespace angle_B_is_60_degrees_l456_456427

theorem angle_B_is_60_degrees (A B C : Type) [triangle A B C] (h1 : ∠ C = 90) (h2 : cos ∠ B = 1 / 2) : ∠ B = 60 :=
sorry

end angle_B_is_60_degrees_l456_456427


namespace combination_opens_lock_l456_456416

variable (a : Fin 26 → Nat)

theorem combination_opens_lock :
  ∃ s : Finset (Fin 26), (∑ x in s, a x) % 26 = 0 := 
sorry

end combination_opens_lock_l456_456416


namespace money_distribution_l456_456481

noncomputable def x : ℚ := 196 / 17
noncomputable def y : ℚ := 182 / 17

theorem money_distribution :
  ∃ (x y : ℚ), (x + 7 = 5 * (y - 7)) ∧ (y + 5 = 7 * (x - 5)) ∧ x = 196 / 17 ∧ y = 182 / 17 :=
by
  use 196 / 17, 182 / 17
  split
  · sorry
  split
  · sorry
  split
  · refl
  · refl

end money_distribution_l456_456481


namespace intersection_eq_l456_456699

open Set

variable (A B : Set ℝ)

def setA : A = {x | -3 < x ∧ x < 2} := sorry

def setB : B = {x | x^2 + 4*x - 5 ≤ 0} := sorry

theorem intersection_eq : A ∩ B = {x | -3 < x ∧ x ≤ 1} :=
sorry

end intersection_eq_l456_456699


namespace selection_of_students_l456_456674

def number_of_ways_select (total_males : ℕ) (total_females : ℕ) : ℕ :=
  (nat.choose total_females 2 * nat.choose total_males 1) +
  (nat.choose total_females 1 * nat.choose total_males 2)

theorem selection_of_students :
  number_of_ways_select 5 3 = 45 := 
sorry

end selection_of_students_l456_456674


namespace conditional_probability_problem_l456_456198

noncomputable def dice := {1, 2, 3, 4}

def eventA (d1 d2 : ℕ) : Prop := d1 ≠ d2
def eventB (d1 d2 : ℕ) : Prop := d1 = 2 ∨ d2 = 2

theorem conditional_probability_problem :
  (∑ d1 in dice, ∑ d2 in dice, (if eventA d1 d2 ∧ eventB d1 d2 then 1 else 0) : ℝ) /
  (∑ d1 in dice, ∑ d2 in dice, (if eventA d1 d2 then 1 else 0) : ℝ) = 1 / 2 := 
sorry

end conditional_probability_problem_l456_456198


namespace train_speed_l456_456565

-- Define the conditions
def train_length : ℝ := 125
def crossing_time : ℝ := 6
def man_speed_kmh : ℝ := 5

-- Convert man's speed to m/s
def man_speed_ms : ℝ := man_speed_kmh * 1000 / 3600

-- Define the speed of the train
theorem train_speed (v_t : ℝ) : 
  (train_length / crossing_time) - man_speed_ms = v_t / 3.6 → v_t = 70 := 
by
  sorry

end train_speed_l456_456565


namespace smallest_value_of_a_l456_456531

theorem smallest_value_of_a :
  ∃ (a b α β γ : ℕ), (P(x) = x^3 - a * x^2 + b * x - 2010) ∧
                    (α + β + γ = a) ∧
                    (α * β * γ = 2010) ∧
                    (a = 78) :=
by
  sorry

end smallest_value_of_a_l456_456531


namespace triangle_area_39_36_15_l456_456569

theorem triangle_area_39_36_15 :
  let a := 39
  let b := 36
  let c := 15
  let s := (a + b + c) / 2
  (√(s * (s - a) * (s - b) * (s - c))).natAbs = 270 :=
by
  -- Definitions
  let a := 39
  let b := 36
  let c := 15
  let s := (a + b + c) / 2
  -- Proof of the area using Heron's formula
  have : (√(s * (s - a) * (s - b) * (s - c))).natAbs = 270 := sorry
  exact this

end triangle_area_39_36_15_l456_456569


namespace sequence_properties_l456_456734

noncomputable def seq_a (n : ℕ) (m : ℝ) (a : ℕ → ℝ) : ℝ :=
if h : n = 0 then log m else a n + log (m - a n)

theorem sequence_properties (m : ℝ) (a : ℕ → ℝ) (h1 : 1 ≤ m) 
  (h2 : a 0 = log m) (h3 : ∀ n, a (n + 1) = seq_a n m a) :
  ∀ n, a n ≤ a (n + 1) ∧ a (n + 1) ≤ m := 
  sorry

end sequence_properties_l456_456734


namespace area_of_closed_shape_l456_456659

noncomputable def areaOfClosedShape : ℝ :=
  ∫ x in 0..(2/3), (x - x^2 - (1/3)*x)

theorem area_of_closed_shape :
  areaOfClosedShape = 4/81 :=
by
  sorry

end area_of_closed_shape_l456_456659


namespace sequence_periodic_l456_456186

noncomputable def sequence (a : ℕ → ℕ) :=
∀ n, (even (a n) → a (n + 1) = Nat.gcd a n) ∧ (odd (a n) → ∃ p, p = Nat.factors_multiplicity a n ∧ a (n + 1) = a n + p^2)

theorem sequence_periodic (a : ℕ → ℕ) (P : a 0 ∈ sorry) :
  ∃ M : ℕ, ∀ m ≥ M, a (m + 2) = a m := sorry

end sequence_periodic_l456_456186


namespace lily_pad_half_coverage_l456_456222

-- Define the conditions in Lean
def doubles_daily (size: ℕ → ℕ) : Prop :=
  ∀ n : ℕ, size (n + 1) = 2 * size n

def covers_entire_lake (size: ℕ → ℕ) (total_size: ℕ) : Prop :=
  size 34 = total_size

-- The main statement to prove
theorem lily_pad_half_coverage (size : ℕ → ℕ) (total_size : ℕ) 
  (h1 : doubles_daily size) 
  (h2 : covers_entire_lake size total_size) : 
  size 33 = total_size / 2 :=
sorry

end lily_pad_half_coverage_l456_456222


namespace multiplier_of_first_integer_l456_456513

theorem multiplier_of_first_integer :
  ∃ m x : ℤ, x + 4 = 15 ∧ x * m = 3 + 2 * 15 ∧ m = 3 := by
  sorry

end multiplier_of_first_integer_l456_456513


namespace point_on_transformed_plane_l456_456821

theorem point_on_transformed_plane :
  let A := (2, -5, 4)
  let plane := (5, 2, -1, 3)
  let k := (4 / 3)
  let transformed_plane := (plane.1, plane.2, plane.3, k * plane.4)
  ((transformed_plane.1 * A.1 + transformed_plane.2 * A.2 + transformed_plane.3 * A.3 + transformed_plane.4) = 0) :=
by
  have A : (ℝ × ℝ × ℝ) := (2, -5, 4)
  have plane : (ℝ × ℝ × ℝ × ℝ) := (5, 2, -1, 3)
  have k : ℝ := 4 / 3
  let transformed_plane : (ℝ × ℝ × ℝ × ℝ) := (plane.1, plane.2, plane.3, k * plane.4)
  have h : transformed_plane.1 * A.1 + transformed_plane.2 * A.2 + transformed_plane.3 * A.3 + transformed_plane.4 = 0
  sorry

end point_on_transformed_plane_l456_456821


namespace ratio_areas_of_octagons_l456_456270

noncomputable def area_inscribed_octagon (r : ℝ) : ℝ := 4 * r^2 * (√2 - 1) 

noncomputable def area_circumscribed_octagon (r : ℝ) : ℝ := 2 * r^2 * (1 + √2)

theorem ratio_areas_of_octagons {r : ℝ} (hr : r > 0) :
  (area_circumscribed_octagon r) / (area_inscribed_octagon r) = 2 :=
by sorry

end ratio_areas_of_octagons_l456_456270


namespace cups_per_larger_crust_l456_456794

theorem cups_per_larger_crust
  (initial_crusts : ℕ)
  (initial_flour : ℚ)
  (new_crusts : ℕ)
  (constant_flour : ℚ)
  (h1 : initial_crusts * (initial_flour / initial_crusts) = initial_flour )
  (h2 : new_crusts * (constant_flour / new_crusts) = constant_flour )
  (h3 : initial_flour = constant_flour)
  : (constant_flour / new_crusts) = (8 / 10) :=
by 
  sorry

end cups_per_larger_crust_l456_456794


namespace probability_same_color_l456_456587

theorem probability_same_color 
  (green_balls : ℕ) (red_balls : ℕ) (total_balls : ℕ)
  (h1 : green_balls = 8)
  (h2 : red_balls = 7)
  (h3 : total_balls = green_balls + red_balls) :
  (8 / 15) ^ 2 + (7 / 15) ^ 2 = 113 / 225 :=
by
  rw [h1, h2, h3]
  sorry

end probability_same_color_l456_456587


namespace area_of_triangle_l456_456646

-- Definitions of the components involved in the acute-angled triangle ABC.
variables (A B C A1 B1 C1 H O : Type) 

-- Conditions for the problem
variables [metric_space A B C A1 B1 C1 H O] 
variables [altitude A A1]
variables [altitude B B1]
variables [altitude C C1]
variables [circumcircle_center O]
variables [circumradius O r]
variables (ABC_area : ℝ)
variables (pedal_perimeter : ℝ)

-- The theorem statement
theorem area_of_triangle
  (hA : height A A1)
  (hB : height B B1)
  (hC : height C C1)
  (H_pedal_def : pedal_triangle H A1 B1 C1)
  (H_perimeter : pedal_perimeter A1 B1 C1 = k)
  (H_circumcenter : circumcenter A B C = O)
  (H_circumradius : circumradius A B C = r)
  : ABC_area = (r / 2) * k :=
  sorry

end area_of_triangle_l456_456646


namespace right_triangle_sides_l456_456075

section TriangleProblem

-- Two digit integers constraints
variables {d e : ℕ} (b h : ℕ)
-- d and e are distinct digits and d > e
variables (hd : d > e) (hdistinct : d ≠ e)
-- Define the hypotenuse and one leg
def hypotenuse := 10 * d + e
def leg := 10 * e + d

-- Condition that forms the right-angled triangle using Pythagorean theorem
def right_angled_triangle := (leg)^2 + b^2 = (hypotenuse)^2

-- Prove the given side lengths form a triangle with the given conditions
theorem right_triangle_sides :
  ∃ b hypotenuse leg,
    right_angled_triangle 33 65 56 ∧
    leg = 56 ∧
    hypotenuse = 65 ∧
    b = 33 ∧
    d = 6 ∧
    e = 5 :=
begin
  use [33, 65, 56],
  unfold right_angled_triangle hypotenuse leg,
  split,
  { simp only [pow_two, bit0_eq_zero, add_assoc, add_left_eq_self, zero_add],
    norm_num1 },
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  split;
  norm_num
end

end TriangleProblem

end right_triangle_sides_l456_456075


namespace increasing_sequence_implies_range_of_a_l456_456728

variables {a : ℝ} {f : ℝ → ℝ}

def f (x : ℝ) : ℝ :=
  if x ≥ 6 then a^(x-5) else (4-(a/2))*x+4

def a_n (n : ℕ) : ℝ := f n

theorem increasing_sequence_implies_range_of_a :
  (∀ n: ℕ, 0 < n → a_n n < a_n (n + 1)) ∧ 
  a > 1 ∧ 
  4 - a / 2 > 0 ∧ 
  (4 - a / 2) * 5 + 4 < a →
  48 / 7 < a ∧ a < 8 :=
by {
  sorry
}

end increasing_sequence_implies_range_of_a_l456_456728


namespace simplify_polynomial_l456_456919

-- Define the original polynomial
def original_expr (x : ℝ) : ℝ := 3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3

-- Define the simplified version of the polynomial
def simplified_expr (x : ℝ) : ℝ := 2 * x^3 - x^2 + 23 * x - 3

-- State the theorem that the original expression is equal to the simplified one
theorem simplify_polynomial (x : ℝ) : original_expr x = simplified_expr x := 
by 
  sorry

end simplify_polynomial_l456_456919


namespace simplify_polynomial_l456_456917

-- Define the original polynomial
def original_expr (x : ℝ) : ℝ := 3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3

-- Define the simplified version of the polynomial
def simplified_expr (x : ℝ) : ℝ := 2 * x^3 - x^2 + 23 * x - 3

-- State the theorem that the original expression is equal to the simplified one
theorem simplify_polynomial (x : ℝ) : original_expr x = simplified_expr x := 
by 
  sorry

end simplify_polynomial_l456_456917


namespace convex_polyhedron_area_inequality_l456_456247

theorem convex_polyhedron_area_inequality
  (P : Type) [polyhedron P]
  (quadrilateral_faces : ∀ f : face P, is_quadrilateral f)
  (A Q : ℝ)
  (hA : surface_area P = A)
  (hQ : sum_of_squares_of_edge_lengths P = Q) :
  Q ≥ 2 * A :=
begin
  sorry
end

end convex_polyhedron_area_inequality_l456_456247


namespace sum_first_n_terms_of_arithmetic_sequence_l456_456354

def arithmetic_sequence_sum (a1 d n: ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem sum_first_n_terms_of_arithmetic_sequence :
  arithmetic_sequence_sum 2 2 n = n * (n + 1) / 2 :=
by sorry

end sum_first_n_terms_of_arithmetic_sequence_l456_456354


namespace equal_spacing_between_paintings_l456_456055

/--
Given:
- The width of each painting is 30 centimeters.
- The total width of the wall in the exhibition hall is 320 centimeters.
- There are six pieces of artwork.
Prove that: The distance between the end of the wall and the artwork, and between the artworks, is 20 centimeters.
-/
theorem equal_spacing_between_paintings :
  let width_painting := 30 -- in centimeters
  let total_wall_width := 320 -- in centimeters
  let num_paintings := 6
  let total_paintings_width := num_paintings * width_painting
  let remaining_space := total_wall_width - total_paintings_width
  let num_spaces := num_paintings + 1
  let space_between := remaining_space / num_spaces
  space_between = 20 := sorry

end equal_spacing_between_paintings_l456_456055


namespace max_sum_of_inequalities_l456_456114

theorem max_sum_of_inequalities (x y : ℝ) (h1 : 4 * x + 3 * y ≤ 10) (h2 : 3 * x + 5 * y ≤ 11) :
  x + y ≤ 31 / 11 :=
sorry

end max_sum_of_inequalities_l456_456114


namespace inequality_proof_l456_456146

theorem inequality_proof (x : ℝ) (hx : 0 < x) : (1 / x) + 4 * (x ^ 2) ≥ 3 :=
by
  sorry

end inequality_proof_l456_456146


namespace T_subset_properties_l456_456735

def T_subsets_count (n : ℕ) : ℕ :=
  if n < 3 then 0
  else 
    let rec a : ℕ → ℕ
      | 3 => 1
      | 4 => 3
      | k+2 => a (k+1) + a k + k
    a n

open Nat

theorem T_subset_properties : T_subsets_count 5 = 7 ∧ T_subsets_count 10 = 133 :=
by
  -- Proof of the theorem should go here
  sorry

end T_subset_properties_l456_456735


namespace quadratic_function_range_l456_456041

def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

lemma quadratic_function_properties :
  ∃ (a b : ℝ),
    (4 * a + 2 * b = 0) ∧ ((b - 1)^2 = 0)
    ∧ (∀ x, quadratic_function a b x = - (1/2 : ℝ) * x^2 + x) := 
by {
  sorry
}

theorem quadratic_function_range :
  ∃ (a b : ℝ),
    (4 * a + 2 * b = 0) ∧ ((b - 1)^2 = 0)
    ∧ (∀ x ∈ (set.Icc 0 3 : set ℝ),
        quadratic_function a b x ∈ set.Icc (-3 / 2) (1 / 2)) :=
by {
  sorry
}

end quadratic_function_range_l456_456041


namespace inscribed_circle_radius_l456_456922

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := semiperimeter a b c
  in (s * (s - a) * (s - b) * (s - c)).sqrt

theorem inscribed_circle_radius (PQ PR QR : ℝ) (hPQ : PQ = 8) (hPR : PR = 17) (hQR : QR = 15) :
  let s := semiperimeter PQ PR QR
  let K := heron_area PQ PR QR
  r = K / s :=
by 
  -- Prove the calculations
  have hs : s = 20 := sorry
  have hK : K = 60 := sorry
  have hr : r = 3 := by
    rw [hs, hK]
    -- remaining steps to simplify r calculation
    sorry

end inscribed_circle_radius_l456_456922


namespace cube_volume_from_surface_area_l456_456410

def surface_area_of_cube (a : ℝ) : ℝ := 6 * a ^ 2
def volume_of_cube (a : ℝ) : ℝ := a ^ 3

theorem cube_volume_from_surface_area (h : ∃ a, surface_area_of_cube a = 600) : volume_of_cube (classical.some h) = 1000 := 
by 
  sorry

end cube_volume_from_surface_area_l456_456410


namespace num_negative_terms_in_first_10_l456_456004

def sequence (n : ℕ) : ℤ :=
  if n = 0 then 1
  else 3 * sequence (n - 1) - 4

theorem num_negative_terms_in_first_10 :
  let a (n : ℕ) := sequence n 
  let first_10_terms := list.range 10 |>.map a
  (list.countp (λ x, x < 0) first_10_terms) = 9 :=
by
  sorry

end num_negative_terms_in_first_10_l456_456004


namespace terminal_side_in_third_quadrant_l456_456732

theorem terminal_side_in_third_quadrant (θ : ℝ) (h1 : cos θ < 0) (h2 : tan θ > 0) : 
θ > π ∧ θ < 3 * π / 2 :=
by
  sorry

end terminal_side_in_third_quadrant_l456_456732


namespace proper_subset_cardinality_l456_456400

/-- Prove that the number of subsets M such that M ⊂ {1, 2} is equal to 3. -/
theorem proper_subset_cardinality : 
  {M : set ℕ | M ⊂ {1, 2}}.to_finset.card = 3 := 
  sorry

end proper_subset_cardinality_l456_456400


namespace probability_third_term_is_3_l456_456527

noncomputable def number_of_valid_permutations : ℕ := 9! - 2 * 8! + 7!

noncomputable def number_of_valid_permutations_with_third_term_3 : ℕ := 8! - 2 * 7! + 6!

theorem probability_third_term_is_3 :
  (number_of_valid_permutations_with_third_term_3 : ℚ) / (number_of_valid_permutations : ℚ) = 43 / 399 :=
by
  sorry

end probability_third_term_is_3_l456_456527


namespace sum_of_decimal_digits_l456_456666

open Nat

theorem sum_of_decimal_digits :
  let s := 5 * (∑ k in Finset.range 100, k * (k + 1) * (k^2 + k + 1))
  let digit_sum := (s.digits 10).sum
  digit_sum = 48 :=
begin
  sorry
end

end sum_of_decimal_digits_l456_456666


namespace pies_made_l456_456170

-- Define the initial number of apples
def initial_apples : Nat := 62

-- Define the number of apples handed out to students
def handed_out_apples : Nat := 8

-- Define the number of apples required per pie
def apples_per_pie : Nat := 9

-- Define the number of remaining apples after handing out to students
def remaining_apples : Nat := initial_apples - handed_out_apples

-- State the theorem
theorem pies_made (initial_apples handed_out_apples apples_per_pie remaining_apples : Nat) :
  initial_apples = 62 →
  handed_out_apples = 8 →
  apples_per_pie = 9 →
  remaining_apples = initial_apples - handed_out_apples →
  remaining_apples / apples_per_pie = 6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end pies_made_l456_456170


namespace sum_divisible_by_M_plus_1_l456_456819

theorem sum_divisible_by_M_plus_1 (M : ℕ) (hM : M > 3) :
  let S := ∑ x in range (M + 1), if (x^2 - x + 1) % M = 0 then x else 0
  in S % (M + 1) = 0 :=
by sorry

end sum_divisible_by_M_plus_1_l456_456819


namespace avg_monthly_growth_rate_in_March_and_April_l456_456975

noncomputable def sales_volume_January := 1000000
noncomputable def sales_drop_February := 0.1
noncomputable def sales_volume_April := 1296000

def sales_growth_rate_formula (x : ℝ) : ℝ :=
  sales_volume_January * (1 - sales_drop_February) * (1 + x)^2

theorem avg_monthly_growth_rate_in_March_and_April : 
  ∃ x : ℝ, sales_growth_rate_formula x = sales_volume_April ∧ x = 0.2 :=
  sorry

end avg_monthly_growth_rate_in_March_and_April_l456_456975


namespace magnitude_of_z_l456_456827

-- Defining the complex number and conditions of the problem
def z : ℂ := -1 + 𝒾

-- Hypothesis given in the problem
def z_cond (z : ℂ) : Prop := z + 2 * z.conj = -3 - 𝒾

-- The main theorem to be proven
theorem magnitude_of_z (h : z_cond z) : complex.abs z = real.sqrt 2 :=
sorry

end magnitude_of_z_l456_456827


namespace BH_eq_CX_l456_456772

noncomputable theory
open_locale classical

variables {A B C M H Q P X : Type*}
  [geometry_of_triangle A B C AM AH Q P X]
  (ABC : acute_scalene_triangle A B C)
  (AM : is_median A M B C)
  (AH : is_altitude A H B C)
  (Q_on_AB : point_on_line Q A B)
  (P_on_AC : point_on_line P A C)
  (QM_perp_AC : perp Q M A C)
  (PM_perp_AB : perp P M A B)
  (X_def : X ∈ circumcircle P M Q ∧ point_on_line X B C)

theorem BH_eq_CX : BH = CX :=
sorry

end BH_eq_CX_l456_456772


namespace complex_number_statements_l456_456929

def imag_unit := complex.I

def statement_A : Prop :=
  imag_unit ^ 9 = imag_unit

def statement_B : Prop :=
  complex.im (3 - 2 * imag_unit) ≠ 2 * imag_unit

def statement_C : Prop :=
  ¬ (complex.snd (conjugate ((1 - imag_unit) ^ 2)) = 2 * imag_unit)

def statement_D (z : complex) : Prop :=
  (z = z.conj) ↔ (complex.im z = 0)

theorem complex_number_statements : statement_A ∧ statement_D :=
by {
  sorry
}

end complex_number_statements_l456_456929


namespace minimum_points_condition_l456_456887

variable (n : ℕ)

-- Points -2 and 2 divide the number line into three segments:
-- Segment 1: (-∞, -2)
-- Segment 2: [-2, 2]
-- Segment 3: (2, ∞)
def divides_number_line (x : ℤ) : ℤ → Prop
| y => if y < x then true else false

def segment1 : ℤ → Prop := divides_number_line (-2)
def segment2 (y : ℤ) : Prop := y >= -2 ∧ y <= 2
def segment3 : ℤ → Prop := divides_number_line (2)

-- Minimum number of points such that at least 3 points fall in one of these segments
theorem minimum_points_condition (h : ∀ p : List ℤ, p.length = n → ∃ (s : ℤ → Prop), (s = segment1 ∨ s = segment2 ∨ s = segment3) ∧ 3 ≤ (p.filter s).length) : n = 7 :=
by
  sorry

end minimum_points_condition_l456_456887


namespace projection_eq_neg_3_sqrt_3_div_2_l456_456710

variables (e1 e2 : ℝ) (a b : ℝ → ℝ) (θ : ℝ)

def unit_vectors (e1 e2 : ℝ → ℝ) : Prop := real_inner e1 e1 = 1 ∧ real_inner e2 e2 = 1

def angle_between_unit_vectors (e1 e2 : ℝ) : Prop := 
  real_inner e1 e2 = cos (2 * real.pi / 3)

def a_def (e1 e2 : ℝ) : ℝ := e1 + 2 * e2

def b_def (e1 e2 : ℝ) : ℝ := 2 * e1 - 3 * e2

def projection (a b : ℝ) (θ : ℝ) : ℝ :=
  let dot_ab := real_inner a b in
  let mag_a := sqrt (real_inner a a) in
  dot_ab / mag_a

theorem projection_eq_neg_3_sqrt_3_div_2 (e1 e2 : ℝ) :
  unit_vectors e1 e2 →
  angle_between_unit_vectors e1 e2 →
  a_def e1 e2 = a →
  b_def e1 e2 = b →
  projection a b (2 * real.pi / 3) = -3 * sqrt 3 / 2 :=
by
  intros,
  sorry

end projection_eq_neg_3_sqrt_3_div_2_l456_456710


namespace intersection_parallel_to_l_l456_456364

variables {m n l : Line} {α β : Plane}

def Skew (m n : Line) : Prop := ¬ (m ∥ n ∨ m ∩ n ≠ ∅)
def Perpendicular (l : Line) (p : Plane) : Prop := ∀ p₀ ∈ p, l ∩ p₀ ∈ l

theorem intersection_parallel_to_l 
  (h_skew : Skew m n)
  (h_m_perp_α : Perpendicular m α)
  (h_n_perp_β : Perpendicular n β)
  (h_l_perp_m : l ⊥ m)
  (h_l_perp_n : l ⊥ n)
  (h_l_notin_α : l ∉ α)
  (h_l_notin_β : l ∉ β) :
  ∃ l1, (l1 ∈ (α ∩ β)) ∧ (l1 ∥ l) :=
sorry

end intersection_parallel_to_l_l456_456364


namespace fire_location_in_city_A_l456_456156

-- Residents of city A always tell the truth
def residents_of_city_A : Prop := ∀ statement : Prop, statement

-- Residents of city B always lie
def residents_of_city_B : Prop := ∀ statement : Prop, ¬statement

-- Residents of city C alternately tell the truth and lie
def residents_of_city_C : Prop := ∀ (statement1 statement2 : Prop), (statement1 ∧ ¬ statement2) ∨ (¬ statement1 ∧ statement2)

-- Phone call statements
def fire_statement : Prop := "We have a fire"
def location_statement : Prop := "In city C"

-- The fire department received a call with the statements "We have a fire" and "In city C"
theorem fire_location_in_city_A
  (received_call_from_city_A : residents_of_city_A → statement = fire_statement ∧ location_statement = "In city A")
  (received_call_from_city_B : residents_of_city_B → ¬fire_statement ∧ location_statement ≠ "In city C")
  (received_call_from_city_C : residents_of_city_C → (fire_statement ∧ ¬ location_statement) ∨ (¬ fire_statement ∧ location_statement))
  : fire_location = "In city A" :=
sorry

end fire_location_in_city_A_l456_456156


namespace people_going_to_zoo_l456_456954

theorem people_going_to_zoo (cars : ℕ) (people_per_car : ℕ) (h1 : cars = 3) (h2 : people_per_car = 21) : 
  cars * people_per_car = 63 :=
by
  rw [h1, h2]
  norm_num

end people_going_to_zoo_l456_456954


namespace jiwon_distance_to_school_l456_456502

theorem jiwon_distance_to_school
  (taehong_distance_meters jiwon_distance_meters : ℝ)
  (taehong_distance_km : ℝ := 1.05)
  (h1 : taehong_distance_meters = jiwon_distance_meters + 460)
  (h2 : taehong_distance_meters = taehong_distance_km * 1000) :
  jiwon_distance_meters / 1000 = 0.59 := 
sorry

end jiwon_distance_to_school_l456_456502


namespace circles_are_externally_tangent_l456_456889

noncomputable def circleA : Prop := ∀ x y : ℝ, x^2 + y^2 + 4 * x + 2 * y + 1 = 0
noncomputable def circleB : Prop := ∀ x y : ℝ, x^2 + y^2 - 2 * x - 6 * y + 1 = 0

theorem circles_are_externally_tangent (hA : circleA) (hB : circleB) : 
  ∃ P Q : ℝ, (P = 5) ∧ (Q = 5) := 
by 
  -- start proving with given conditions
  sorry

end circles_are_externally_tangent_l456_456889


namespace four_digit_number_is_2561_l456_456315

-- Define the problem domain based on given conditions
def unique_in_snowflake_and_directions (grid : Matrix (Fin 3) (Fin 6) ℕ) : Prop :=
  ∀ (i j : Fin 3), -- across all directions
    ∀ (x y : Fin 6), 
      (x ≠ y) → 
      (grid i x ≠ grid i y) -- uniqueness in i-direction
      ∧ (grid y x ≠ grid y y) -- uniqueness in j-direction

-- Assignment of numbers in the grid fulfilling the conditions
def grid : Matrix (Fin 3) (Fin 6) ℕ :=
![ ![2, 5, 2, 5, 1, 6], ![4, 3, 2, 6, 1, 1], ![6, 1, 4, 5, 3, 2] ]

-- Definition of the four-digit number
def ABCD : ℕ := grid 0 1 * 1000 + grid 0 2 * 100 + grid 0 3 * 10 + grid 0 4

-- The theorem to be proved
theorem four_digit_number_is_2561 :
  unique_in_snowflake_and_directions grid →
  ABCD = 2561 :=
sorry

end four_digit_number_is_2561_l456_456315


namespace geometric_sequence_f_l456_456779

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x * (x - a) * (x - (a * 2)) * (x - (a * 4)) * (x - (a * 8)) * (x - (a * 16)) * (x - (a * 32)) * (x - b)

def geometric_sequence (a₁ a₈ : ℝ) (n : ℕ) : ℝ :=
  if n = 1 then a₁ else a₁ * 2^(n-1)

theorem geometric_sequence_f''_at_zero_eq :
  ∀ (a₁ a₈ : ℝ), a₁ = 2 → a₈ = 2 * 2^7 → f'' (0) = 2^12 := by
  sorry

end geometric_sequence_f_l456_456779


namespace abc_inequality_l456_456826

theorem abc_inequality (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_order : a ≤ b ∧ b ≤ c) (h_sum : a^2 + b^2 + c^2 = 9) : abc + 1 > 3a :=
by
  sorry

end abc_inequality_l456_456826


namespace f_odd_f_monotonic_increasing_intervals_f_no_max_value_f_extreme_points_l456_456383

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem f_odd (x : ℝ) : f (-x) = -f (x) :=
by sorry

theorem f_monotonic_increasing_intervals :
  ∀ x : ℝ, (x < -Real.sqrt 3 / 3 ∨ x > Real.sqrt 3 / 3) → f x' > f x :=
by sorry

theorem f_no_max_value :
  ∀ x : ℝ, ¬(∃ M, f x ≤ M) :=
by sorry

theorem f_extreme_points :
  f (-Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 ∧ f (Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 :=
by sorry

end f_odd_f_monotonic_increasing_intervals_f_no_max_value_f_extreme_points_l456_456383


namespace kitchen_upgrade_cost_l456_456908

-- Define the number of cabinet knobs and their cost
def num_knobs : ℕ := 18
def cost_per_knob : ℝ := 2.50

-- Define the number of drawer pulls and their cost
def num_pulls : ℕ := 8
def cost_per_pull : ℝ := 4.00

-- Calculate the total cost of the knobs
def total_cost_knobs : ℝ := num_knobs * cost_per_knob

-- Calculate the total cost of the pulls
def total_cost_pulls : ℝ := num_pulls * cost_per_pull

-- Calculate the total cost of the kitchen upgrade
def total_cost : ℝ := total_cost_knobs + total_cost_pulls

-- Theorem statement
theorem kitchen_upgrade_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_cost_l456_456908


namespace sphere_surface_area_l456_456689

-- Define the problem's conditions
def regular_triangle (A B C : Point) : Prop := 
  distance A B = sqrt 3 ∧ distance B C = sqrt 3 ∧ distance C A = sqrt 3

def sphere_contains (O : Point) (r : ℝ) (A B C : Point) : Prop :=
  distance O A = r ∧ distance O B = r ∧ distance O C = r

def angle_between_OA_and_plane_ABC (O A B C : Point) (θ : ℝ) : Prop :=
  -- Assuming there is a function to compute the angle given points, not verified implementation
  ∃ n : Vector, normal_to_plane ABC n ∧ angle_between (O - A) n = θ

noncomputable def radius (O : Point) (A B C : Point) : ℝ :=
  distance O A

-- Prove the surface area of the sphere
theorem sphere_surface_area
  {A B C O : Point}
  (h1 : regular_triangle A B C)
  (h2 : sphere_contains O (radius O A B C) A B C)
  (h3 : angle_between_OA_and_plane_ABC O A B C (60 * (π / 180))) :
  surface_area_of_sphere (radius O A B C) = (16 / 3) * π :=
sorry

end sphere_surface_area_l456_456689


namespace part_one_union_sets_l456_456391

theorem part_one_union_sets (a : ℝ) (A B : Set ℝ) :
  (a = 2) →
  A = {x | x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0} →
  B = {x | -2 < x ∧ x < 2} →
  A ∪ B = {x | -2 < x ∧ x ≤ 3} :=
by
  sorry

end part_one_union_sets_l456_456391


namespace extremum_condition_at_one_l456_456750

theorem extremum_condition_at_one (a : ℝ) (f : ℝ → ℝ) (h : f = (λ x, x^3 - 3 * x * a)) 
  (extremum_at_one : ∃ (f' : ℝ → ℝ), f' = (λ x, 3 * x^2 - 3 * a) ∧ f' 1 = 0):
  a = 1 := 
by 
  sorry

end extremum_condition_at_one_l456_456750


namespace normal_dist_probability_l456_456713

open MeasureTheory.ProbabilityTheory
open MeasureTheory

noncomputable def standard_normal_cdf (x : ℝ) : ℝ := sorry

theorem normal_dist_probability :
  (∀ (ξ : ℝ -> ℝ) (σ : ℝ), 
   (ξ ∼ Normal 1 σ^2) → 
   P(ξ < 2) = 0.6 →
   P(0 < ξ < 1) = 0.1) :=
begin
  sorry
end

end normal_dist_probability_l456_456713


namespace purchase_price_l456_456883

-- Define the problem conditions
variable P : ℝ
variable market_value_after_two_years : ℝ := 3200

def value_after_one_year (P : ℝ) : ℝ := 0.70 * P
def value_after_two_years (P : ℝ) : ℝ := 0.40 * P

-- Define the main theorem to prove the purchase price is 8000
theorem purchase_price (h : value_after_two_years P = market_value_after_two_years) : P = 8000 :=
by sorry

end purchase_price_l456_456883


namespace grace_dimes_count_l456_456740

-- Defining the conditions
def dimes_to_pennies (d : ℕ) : ℕ := 10 * d
def nickels_to_pennies : ℕ := 10 * 5
def total_pennies (d : ℕ) : ℕ := dimes_to_pennies d + nickels_to_pennies

-- The statement of the theorem
theorem grace_dimes_count (d : ℕ) (h : total_pennies d = 150) : d = 10 := 
sorry

end grace_dimes_count_l456_456740


namespace stairs_climbed_l456_456860

theorem stairs_climbed (s v r : ℕ) 
  (h_s: s = 318) 
  (h_v: v = 18 + s / 2) 
  (h_r: r = 2 * v) 
  : s + v + r = 849 :=
by {
  sorry
}

end stairs_climbed_l456_456860


namespace time_period_simple_interest_l456_456243

theorem time_period_simple_interest 
  (P : ℝ) (R18 R12 : ℝ) (additional_interest : ℝ) (T : ℝ) :
  P = 2500 →
  R18 = 0.18 →
  R12 = 0.12 →
  additional_interest = 300 →
  P * R18 * T = P * R12 * T + additional_interest →
  T = 2 :=
by
  intros P_val R18_val R12_val add_int_val interest_eq
  rw [P_val, R18_val, R12_val, add_int_val] at interest_eq
  -- Continue the proof here
  sorry

end time_period_simple_interest_l456_456243


namespace no_negatives_l456_456487

theorem no_negatives (x y : ℝ) (h : |x^2 + y^2 - 4*x - 4*y + 5| = |2*x + 2*y - 4|) : 
  ¬ (x < 0) ∧ ¬ (y < 0) :=
by
  sorry

end no_negatives_l456_456487


namespace surface_area_resulting_solid_l456_456911

noncomputable def surface_area_of_stacked_cubes : ℕ := 34

theorem surface_area_resulting_solid :
  let twelve_unit_cubes := 12,
      base_layer := 3 * 3,
      top_center_cube := 1 in
  (base_layer + top_center_cube - 1) = twelve_unit_cubes →
  surface_area_of_stacked_cubes = 34 :=
by
  intro h
  sorry

end surface_area_resulting_solid_l456_456911


namespace problem1_problem2_l456_456288

theorem problem1 : (sqrt 16 + real.cbrt (-27) * sqrt (1/4)) = 5/2 :=
by
  sorry

theorem problem2 : abs (sqrt 6 - 2) + abs (sqrt 2 - 1) + abs (sqrt 6 - 3) = sqrt 2 :=
by
  sorry

end problem1_problem2_l456_456288


namespace dress_designs_count_l456_456250

-- We define the specific elements used in the problem
def numColors := 5
def numPatterns := 4
def numSleeveLengths := 2

-- Now we write the Lean statement for the problem
theorem dress_designs_count : numColors * numPatterns * numSleeveLengths = 40 := 
by simp [numColors, numPatterns, numSleeveLengths]; sorry

end dress_designs_count_l456_456250


namespace segments_AY_and_BX_are_equal_l456_456466

variables 
  {ABC : Type*} [triangle ABC] 
  {I : incenter ABC} 
  {A B C : point ABC}
  (circ1 : circle passing_through_A_C_I ABC)
  (circ2 : circle passing_through_B_C_I ABC)
  {X : point_on_BC circ1}
  {Y : point_on_AC circ2}

theorem segments_AY_and_BX_are_equal :
  segment_length AY = segment_length BX :=
by
  sorry

end segments_AY_and_BX_are_equal_l456_456466


namespace exists_divisor_for_all_f_values_l456_456807

theorem exists_divisor_for_all_f_values (f : ℕ → ℕ) (h_f_range : ∀ n, 1 < f n) (h_f_div : ∀ m n, f (m + n) ∣ f m + f n) :
  ∃ c : ℕ, c > 1 ∧ ∀ n, c ∣ f n := 
sorry

end exists_divisor_for_all_f_values_l456_456807


namespace linear_function_domain_range_l456_456687

-- Define the assumptions (conditions) based on the problem statement
def domain_condition (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 3
def range_condition (y : ℝ) : Prop := -4 ≤ y ∧ y ≤ 1

-- Define the linear function forms
noncomputable def linear_function_1 : ℝ → ℝ := λ x, (5/6) * x - 3/2
noncomputable def linear_function_2 : ℝ → ℝ := λ x, (-5/6) * x - 3/2

-- State the theorem
theorem linear_function_domain_range :
  (∀ x, domain_condition x → range_condition (linear_function_1 x)) ∧
  (∀ x, domain_condition x → range_condition (linear_function_2 x)) :=
by
  -- Proof omitted
  sorry

end linear_function_domain_range_l456_456687


namespace volume_of_regular_quadrilateral_pyramid_l456_456611

noncomputable def volume_of_pyramid (a : ℝ) : ℝ :=
  let x := 1 -- A placeholder to outline the structure
  let PM := (6 * a) / 5
  let V := (2 * a^3) / 5
  V

theorem volume_of_regular_quadrilateral_pyramid
  (a PM : ℝ)
  (h1 : PM = (6 * a) / 5)
  [InstReal : Nonempty (Real)] :
  volume_of_pyramid a = (2 * a^3) / 5 :=
by
  sorry

end volume_of_regular_quadrilateral_pyramid_l456_456611


namespace kitchen_upgrade_total_cost_l456_456906

-- Defining the given conditions
def num_cabinet_knobs : ℕ := 18
def cost_per_cabinet_knob : ℚ := 2.50

def num_drawer_pulls : ℕ := 8
def cost_per_drawer_pull : ℚ := 4

-- Definition of the total cost function
def total_cost : ℚ :=
  (num_cabinet_knobs * cost_per_cabinet_knob) + (num_drawer_pulls * cost_per_drawer_pull)

-- The theorem to prove the total cost is $77.00
theorem kitchen_upgrade_total_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_total_cost_l456_456906


namespace not_all_roots_real_l456_456158

-- Define the quintic polynomial with coefficients a5, a4, a3, a2, a1, a0
def quintic_polynomial (a5 a4 a3 a2 a1 a0 : ℝ) (x : ℝ) : ℝ :=
  a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0

-- Define a predicate for the existence of all real roots
def all_roots_real (a5 a4 a3 a2 a1 a0 : ℝ) : Prop :=
  ∀ r : ℝ, quintic_polynomial a5 a4 a3 a2 a1 a0 r = 0

-- Define the main theorem statement
theorem not_all_roots_real (a5 a4 a3 a2 a1 a0 : ℝ) :
  2 * a4^2 < 5 * a5 * a3 →
  ¬ all_roots_real a5 a4 a3 a2 a1 a0 :=
by
  sorry

end not_all_roots_real_l456_456158


namespace cosine_squared_is_half_l456_456118

def sides_of_triangle (p q r : ℝ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧ p + q > r ∧ q + r > p ∧ r + p > q

noncomputable def cosine_squared (p q r : ℝ) : ℝ :=
  ((p^2 + q^2 - r^2) / (2 * p * q))^2

theorem cosine_squared_is_half (p q r : ℝ) (h : sides_of_triangle p q r) 
  (h_eq : p^4 + q^4 + r^4 = 2 * r^2 * (p^2 + q^2)) : cosine_squared p q r = 1 / 2 :=
by
  sorry

end cosine_squared_is_half_l456_456118


namespace AB_length_l456_456080

-- Define coordinates of point A
def A : ℝ × ℝ := (-5, 2)

-- Define the set of possible coordinates for point B given the conditions
def possible_B_coordinates : set (ℝ × ℝ) := { (3, 2), (-3, 2) }

-- Define a function to calculate the length of the line segment AB
def segment_length (B : ℝ × ℝ) : ℝ :=
  real.dist (A.1) (B.1)

-- Define the property to prove, i.e. the length of AB is either 2 or 8
def AB_length_proof : Prop :=
  ∀ B ∈ possible_B_coordinates, segment_length B ∈ {2, 8}

-- Statement of the problem in Lean
theorem AB_length :
  AB_length_proof :=
sorry

end AB_length_l456_456080


namespace standard_equation_of_ellipse_slopes_sum_constant_l456_456008

-- Defining the conditions
def ellipse (x y : ℝ) (a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def point (x y : ℝ) : Prop := y = -sqrt(6) / 3

-- Given the ellipse passes through the point (1, -sqrt(6)/3)
def ellipse_condition (a b : ℝ) : Prop := ellipse 1 (-sqrt(6) / 3) a b

-- Given the eccentricity of the ellipse is (sqrt(6) / 3)
def eccentricity_condition (a c : ℝ) : Prop := c / a = sqrt(6) / 3

-- Standard equation of the ellipse with a^2 = 3, b^2 = 1
theorem standard_equation_of_ellipse (a b : ℝ) (h1 : ellipse_condition a b) :
  ∃ a b : ℝ, a^2 = 3 ∧ b^2 = 1 ∧ ellipse x y a b :=
sorry

-- Investigate whether k_PN + k_QN is a constant value
theorem slopes_sum_constant (k : ℝ) :
  let P := (1, k * (1 - 1)),
      Q := (k, k)
  in 
  let N := (3, 2),
      k_PN := (N.2 - P.2) / (N.1 - P.1),
      k_QN := (N.2 - Q.2) / (N.1 - Q.1)
  in k_PN + k_QN = 2 :=
sorry

end standard_equation_of_ellipse_slopes_sum_constant_l456_456008


namespace find_n_l456_456925

theorem find_n (n : ℝ) : (10:ℝ)^n = (10:ℝ)^(-4) * real.sqrt ((10:ℝ)^60 / 1000) → n = 24.5 :=
by
  sorry

end find_n_l456_456925


namespace geometric_mean_l456_456083

theorem geometric_mean {a : ℕ → ℤ} (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = a n + 2) :
  let a_5 := a 5 in
  (a 1 * a_5 = 9 ∧ (∀ m, m^2 = a 1 * a_5 → m = 3 ∨ m = -3)) :=
by
  -- Define the sequence based on the conditions
  def a : ℕ → ℤ
  | 1       := 1
  | (n + 1) := a n + 2
  
  have : a 5 = 9 := by sorry
  show a 1 * a 5 = 9 ∧ (∀ m, m^2 = a 1 * a 5 → m = 3 ∨ m = -3), from sorry

end geometric_mean_l456_456083


namespace time_to_drain_all_water_l456_456193

theorem time_to_drain_all_water:
  ∀ (initial_amount final_amount : ℝ) (time_decrease : ℝ),
    initial_amount = 24.7 →
    final_amount = 17.1 →
    time_decrease = 40 →
    (initial_amount - final_amount) / time_decrease * (initial_amount / ((initial_amount - final_amount) / time_decrease)) = (130 : ℝ) :=
begin
  intros initial_amount final_amount time_decrease h1 h2 h3,
  rw [h1, h2, h3],
  sorry, -- Proof will be filled in here
end

end time_to_drain_all_water_l456_456193


namespace cos_pi_minus_alpha_l456_456028

noncomputable def cos_angle_shift (alpha : ℝ) : ℝ :=
  let x := -3
  let y := 4
  let r := Real.sqrt (x^2 + y^2)
  let cos_alpha := x / r
  -cos_alpha

theorem cos_pi_minus_alpha : cos_angle_shift α = 3 / 5 :=
by
  sorry

end cos_pi_minus_alpha_l456_456028


namespace intersection_empty_l456_456047

-- Define the set M
def M : Set ℝ := { x | ∃ y, y = Real.log (1 - x)}

-- Define the set N
def N : Set (ℝ × ℝ) := { p | ∃ x, ∃ y, (p = (x, y)) ∧ (y = Real.exp x) ∧ (x ∈ Set.univ)}

-- Prove that M ∩ N = ∅
theorem intersection_empty : M ∩ (Prod.fst '' N) = ∅ :=
by
  sorry

end intersection_empty_l456_456047


namespace max_sum_of_angles_l456_456006

noncomputable theory
open_locale classical

-- Definitions of the problem setup conditions
def unit_right_prism := sorry  -- Define the unit right prism structure

def E_on_BB1 (E : unit_right_prism) := sorry  -- Define point E lies on edge BB1
def F_on_DD1 (F : unit_right_prism) := sorry  -- Define point F lies on edge DD1

def BE_eq_D1F (E F : unit_right_prism) : Prop := sorry  -- Define condition BE = D1F

def angle_between_line_and_plane (EF : unit_right_prism) (plane : Type) : ℝ := sorry  -- Angle between EF and a plane

-- Angle alpha and beta
def α (E F : unit_right_prism) : ℝ := angle_between_line_and_plane (EF E F) plane_AB -- Define α: angle between EF and plane AB
def β (E F : unit_right_prism) : ℝ := angle_between_line_and_plane (EF E F) plane_BC1 -- Define β: angle between EF and plane BC1

-- Theorem statement
theorem max_sum_of_angles (E F : unit_right_prism) (hE : E_on_BB1 E) (hF : F_on_DD1 F) (h_eq : BE_eq_D1F E F) :
  α E F + β E F ≤ 90 :=
sorry  -- Proof of the theorem

end max_sum_of_angles_l456_456006


namespace length_of_AB_proof_l456_456760

-- Assumptions and definitions
variables {A B C D E F : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables {AD AE CF : Set (A × B)} {BD DE EC AF FC : ℝ}

def angle_bisectors (AD AE : Set (A × B)) (BAC : Angle A B C) : Prop :=
AD ∈ Line.point_line A D ∧ AE ∈ Line.point_line A E ∧ 
    ∀ (P : Point), P ∈ AD → ∠BAC = 2 * ∠BAP

def median (CF : Set (A × B)) (A B C : Point) : Prop :=
∃ (M : Point), M = midpoint B C ∧ F = M

-- Given conditions
axiom h1 : angle_bisectors AD AE (angle BAC)
axiom h2 : ∃ (AB AC : ℝ), AB > 0 ∧ AC > 0
axiom h3 : BD = 3
axiom h4 : DE = 4
axiom h5 : EC = 8
axiom h6 : median CF A B C
axiom h7 : AF = FC

-- Main statement to prove
theorem length_of_AB_proof : AB = 5 * Real.sqrt 2 :=
sorry

end length_of_AB_proof_l456_456760


namespace split_square_unique_configurations_l456_456239

-- Define the square and its properties
def square_side := 4
def square_area := square_side * square_side

-- Proof statement
theorem split_square_unique_configurations :
  ∃ (k l a b: ℚ), 
  0 < k ∧ k < square_side ∧
  0 < l ∧ l < square_side ∧
  0 < a ∧ a < square_side ∧
  0 < b ∧ b < square_side ∧
  (k * a).denominator = 1 ∧
  (k * b).denominator = 1 ∧
  (l * a).denominator = 1 ∧
  (l * b).denominator = 1 ∧
  (k + l = square_side) ∧ 
  (a + b = square_side) ∧
  unique_configurations k l a b = 5 :=
by
  sorry

end split_square_unique_configurations_l456_456239


namespace solve_quadratic1_solve_quadratic2_l456_456161

theorem solve_quadratic1 (x : ℝ) :
  x^2 + 10 * x + 16 = 0 ↔ (x = -2 ∨ x = -8) :=
by
  sorry

theorem solve_quadratic2 (x : ℝ) :
  x * (x + 4) = 8 * x + 12 ↔ (x = -2 ∨ x = 6) :=
by
  sorry

end solve_quadratic1_solve_quadratic2_l456_456161


namespace total_passengers_transported_l456_456968

theorem total_passengers_transported
  (passengers_one_way : ℕ)
  (passengers_return : ℕ)
  (initial_round_trips : ℕ)
  (additional_round_trips : ℕ)
  (total_round_trips : ℕ)
  (total_passengers_one_round_trip : ℕ)
  (total_passengers_all_round_trips : ℕ) :
  passengers_one_way = 100 →
  passengers_return = 60 →
  initial_round_trips = 1 →
  additional_round_trips = 3 →
  total_round_trips = initial_round_trips + additional_round_trips →
  total_passengers_one_round_trip = passengers_one_way + passengers_return →
  total_passengers_all_round_trips = total_passengers_one_round_trip * total_round_trips →
  total_passengers_all_round_trips = 640 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6] at h7
  exact h7
  sorry

end total_passengers_transported_l456_456968


namespace intersection_A_B_l456_456696

def A := { x : Real | -3 < x ∧ x < 2 }
def B := { x : Real | x^2 + 4*x - 5 ≤ 0 }

theorem intersection_A_B :
  (A ∩ B = { x : Real | -3 < x ∧ x ≤ 1 }) := by
  sorry

end intersection_A_B_l456_456696


namespace hyperbola_slopes_l456_456390

theorem hyperbola_slopes (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_eccentricity : (2 * b) / (sqrt (a^2 + b^2)) = 2) 
  (M : ℝ × ℝ) (hM : (M.1^2 / a^2) - (M.2^2 / b^2) = 1) : 
  ∃ k1 k2 : ℝ, (k1 * k2 = 3) :=
sorry

end hyperbola_slopes_l456_456390


namespace only_D_is_quadratic_l456_456558

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c, ∀ x, f x = a * x^2 + b * x + c

def func_A : ℝ → ℝ := λ x, 1 / x^2
def func_B : ℝ → ℝ := λ x, 2 * x + 1
def func_C : ℝ → ℝ := λ x, (1 / 2) * x^2 + 2 * x^3
def func_D : ℝ → ℝ := λ x, -4 * x^2 + 5

theorem only_D_is_quadratic : 
  (is_quadratic func_A = false) ∧
  (is_quadratic func_B = false) ∧
  (is_quadratic func_C = false) ∧
  (is_quadratic func_D = true) :=
by
  sorry

end only_D_is_quadratic_l456_456558


namespace initial_books_gathered_l456_456795

-- Conditions
def total_books_now : Nat := 59
def books_found : Nat := 26

-- Proof problem
theorem initial_books_gathered : total_books_now - books_found = 33 :=
by
  sorry -- Proof to be provided later

end initial_books_gathered_l456_456795


namespace sammy_offer_l456_456142

-- Declaring the given constants and assumptions
def peggy_records : ℕ := 200
def bryan_interested_records : ℕ := 100
def bryan_uninterested_records : ℕ := 100
def bryan_interested_offer : ℕ := 6
def bryan_uninterested_offer : ℕ := 1
def sammy_offer_diff : ℕ := 100

-- The problem to be proved
theorem sammy_offer:
    ∃ S : ℝ, 
    (200 * S) - 
    (bryan_interested_records * bryan_interested_offer +
    bryan_uninterested_records * bryan_uninterested_offer) = sammy_offer_diff → 
    S = 4 :=
sorry

end sammy_offer_l456_456142


namespace sphere_volume_from_rect_solid_l456_456972

theorem sphere_volume_from_rect_solid :
  ∀ (a b c : ℝ), a * b = 1 → b * c = 2 → a * c = 2 →
  let d := real.sqrt (a^2 + b^2 + c^2),
      r := d / 2,
      V := (4/3) * real.pi * r^3
  in V = real.sqrt 6 * real.pi :=
by
  intros a b c hab hbc hac
  let d := real.sqrt (a^2 + b^2 + c^2)
  let r := d / 2
  let V := (4 / 3) * real.pi * r^3
  sorry

end sphere_volume_from_rect_solid_l456_456972


namespace boundedRegionHas48Squares_l456_456300

def countIntegerSquaresInBoundedRegion : Nat :=
  let x_bound : Nat := 6
  let sqrt3 : Real := Real.sqrt 3

  -- Function to calculate maximum y for a given x
  def max_y (x: Nat) : Nat := Real.floor (sqrt3 * x)

  -- Counting 1x1 squares
  def count_1x1_squares :=
    List.sum (List.map (λ x, max_y x) (List.range (x_bound + 1)))
  
  -- Counting 2x2 squares
  def count_2x2_squares :=
    List.sum (List.map (λ y, max_y (y + 1) - 1) (List.range (max_y x_bound - 1)))
  
  -- Counting 3x3 squares
  def count_3x3_squares :=
    List.sum (List.map (λ y, max_y (y + 2) - 2) (List.range (max_y x_bound - 2)))
  
  count_1x1_squares + count_2x2_squares + count_3x3_squares

theorem boundedRegionHas48Squares :
  countIntegerSquaresInBoundedRegion = 48 :=
by
  sorry

end boundedRegionHas48Squares_l456_456300


namespace inscribable_in_circle_l456_456120

variables {A B C D : Type} [EuclideanGeometry A B C D]
variable (P : Quadrilateral A B C D) 

-- The condition: ABCD is convex
variable [ConvexQuadrilateral P]

-- The condition: ∠ABD = ∠ACD
variable (h_angle_eq : ∠(A B D) = ∠(A C D))

-- The statement to prove: ABCD can be inscribed in a circle
theorem inscribable_in_circle : InscribableInCircle P :=
by sorry

end inscribable_in_circle_l456_456120


namespace count_primes_sum_in_set_l456_456741

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ n = p1 + p2

def count_sum_of_two_primes (ns : List ℕ) : ℕ :=
  ns.countp is_sum_of_two_primes

theorem count_primes_sum_in_set : 
  count_sum_of_two_primes [19, 21, 23, 25, 27] = 3 :=
by
  sorry

end count_primes_sum_in_set_l456_456741


namespace test_unanswered_one_way_l456_456218

theorem test_unanswered_one_way (Q A : ℕ) (hQ : Q = 4) (hA : A = 5):
  ∀ (unanswered : ℕ), (unanswered = 1) :=
by
  intros
  sorry

end test_unanswered_one_way_l456_456218


namespace percent_increase_correct_l456_456985

variable (initial_price final_price : ℝ)
variable (opening_price closing_price : Real := 28, 29)
variable (percent_increase : Real := (closing_price - opening_price) / opening_price * 100)

theorem percent_increase_correct : 
  initial_price = opening_price -> 
  final_price = closing_price -> 
  percent_increase ≈ 3.57 :=
by
  intros
  sorry

end percent_increase_correct_l456_456985


namespace toy_store_shelves_used_l456_456232

noncomputable def total_bears (initial : ℕ) (shipment : ℕ) : ℕ :=
  initial + shipment
  
def shelves_used (total : ℕ) (bears_per_shelf : ℕ) : ℕ :=
  total / bears_per_shelf

theorem toy_store_shelves_used :
  shelves_used (total_bears 5 7) 6 = 2 :=
begin
  sorry
end

end toy_store_shelves_used_l456_456232


namespace distance_traveled_by_car_l456_456085

-- Define the total distance D
def D : ℕ := 24

-- Define the distance traveled by foot
def distance_by_foot : ℕ := (1/2 : ℚ) * D

-- Define the distance traveled by bus
def distance_by_bus : ℕ := (1/4 : ℚ) * D

-- Define the distance traveled by car
def distance_by_car : ℕ := D - (distance_by_foot + distance_by_bus)

-- The theorem we want to prove
theorem distance_traveled_by_car : distance_by_car = 6 := by
  sorry

end distance_traveled_by_car_l456_456085


namespace circles_diameter_endpoint_on_tangent_lines_l456_456201

theorem circles_diameter_endpoint_on_tangent_lines 
  (k1 k2 : Circle) 
  (h_noncongruent : k1 ≠ k2) 
  (h_exterior : k1 ∩ k2 = ∅) 
  (O1 O2 : Point) 
  (h_centers : O1 = k1.center ∧ O2 = k2.center) 
  (A B : Point) 
  (h_tangents_intersect : are_common_tangents_intersect_at k1 k2 A B (line_through O1 O2)) 
  (P : Point) 
  (h_P_on_k1 : P ∈ k1) : 
  ∃ C D : Point, is_diameter_of k2 C D ∧ (C ∈ line_of PA ∧ D ∈ line_of PB) :=
sorry

end circles_diameter_endpoint_on_tangent_lines_l456_456201


namespace total_distance_l456_456969

theorem total_distance (D : ℝ) (h_walk : ∀ d t, d = 4 * t) 
                       (h_run : ∀ d t, d = 8 * t) 
                       (h_time : ∀ t_walk t_run, t_walk + t_run = 0.75) 
                       (h_half : D / 2 = d_walk ∧ D / 2 = d_run) :
                       D = 8 := 
by
  sorry

end total_distance_l456_456969


namespace f_odd_increasing_intervals_no_max_value_extreme_points_l456_456385

open Real

namespace FunctionAnalysis

def f (x : ℝ) := x^3 - x

theorem f_odd : ∀ x : ℝ, f (-x) = -f (x) :=
by
  intro x
  show f (-x) = -f (x)
  calc
    f (-x) = (-x)^3 - (-x) : rfl
    ... = -x^3 + x : by ring
    ... = -(x^3 - x) : by ring
    ... = -f (x) : rfl

theorem increasing_intervals : ∀ x : ℝ, 
  (f' x > 0 ↔ x < -sqrt 3 / 3 ∨ x > sqrt 3 / 3) :=
by
  intro x
  have h_deriv : deriv f x = 3 * x^2 - 1 := deriv_pows x
  rw ← h_deriv
  split
  · intro h
    have : 3 * x^2 - 1 > 0 := h
    split
    · apply Or.inl
      linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]
    · apply Or.inr
      linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]
  · intro h
    cases h
    · linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]
    · linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]

theorem no_max_value : ∃ L : ℝ, ∀ x : ℝ, f(x) < L → False :=
by
  use 1
  intro x h
  have : ∀ x : ℝ, f(x) > x := λ x, by norm_num
  specialize this x
  linarith

theorem extreme_points : ∀ x : ℝ,
  (f' x = 0) ↔ (x = sqrt(3) / 3 ∨ x = -sqrt(3) / 3) :=
by
  intro x
  have h_deriv : deriv f x = 3 * x^2 - 1 := deriv_pows x
  rw ← h_deriv
  split
  · intro h
    solve_by_elim
  · intro h
    solve_by_elim

end FunctionAnalysis

end f_odd_increasing_intervals_no_max_value_extreme_points_l456_456385


namespace kitchen_upgrade_total_cost_l456_456904

-- Defining the given conditions
def num_cabinet_knobs : ℕ := 18
def cost_per_cabinet_knob : ℚ := 2.50

def num_drawer_pulls : ℕ := 8
def cost_per_drawer_pull : ℚ := 4

-- Definition of the total cost function
def total_cost : ℚ :=
  (num_cabinet_knobs * cost_per_cabinet_knob) + (num_drawer_pulls * cost_per_drawer_pull)

-- The theorem to prove the total cost is $77.00
theorem kitchen_upgrade_total_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_total_cost_l456_456904


namespace prob_odd_sum_l456_456099

-- Given conditions on the spinners
def spinner_P := [1, 2, 3]
def spinner_Q := [2, 4, 6]
def spinner_R := [1, 3, 5]

-- Probability of spinner P landing on an even number is 1/3
def prob_even_P : ℚ := 1 / 3

-- Probability of odd sum from spinners P, Q, and R
theorem prob_odd_sum : 
  (prob_even_P = 1 / 3) → 
  ∃ p : ℚ, p = 1 / 3 :=
by
  sorry

end prob_odd_sum_l456_456099


namespace eccentricity_is_sqrt3_l456_456038

noncomputable def eccentricity_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ℝ :=
let c := Real.sqrt (a^2 + b^2) in
let x := a^2 / c in
let P := (x, m) in
let PF1 := sorry in  -- coordinates or expression for PF1
let PF2 := sorry in  -- coordinates or expression for PF2
have h1 : PF1 ⊥ PF2 := sorry,  -- condition PF1 ⊥ PF2
have h2 : |PF1| * |PF2| = 4 * a * b := sorry,  -- condition |PF1| * |PF2| = 4ab
(Real.sqrt (3))

theorem eccentricity_is_sqrt3 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
 eccentricity_hyperbola a b ha hb = Real.sqrt 3 := sorry

end eccentricity_is_sqrt3_l456_456038


namespace not_all_rational_l456_456435

-- Definitions for initial conditions
def x0 := 1 - Real.sqrt 2
def y0 := Real.sqrt 2
def z0 := 1 + Real.sqrt 2

-- Definition for the transformation
def transform (x y z : ℝ) : ℝ × ℝ × ℝ := 
  (x ^ 2 + x * y + y ^ 2, y ^ 2 + y * z + z ^ 2, z ^ 2 + z * x + x ^ 2)

-- Proposition stating that the numbers on the board can never all be rational
theorem not_all_rational : ∀ n, ¬ (∃ xn yn zn : ℝ, 
  (xn, yn, zn) = (transform^[n]) (x0, y0, z0) ∧ xn ∈ ℚ ∧ yn ∈ ℚ ∧ zn ∈ ℚ) :=
begin
  -- no proof required
  sorry
end

end not_all_rational_l456_456435


namespace valid_values_l456_456207

noncomputable def is_defined (x : ℝ) : Prop := 
  (x^2 - 4*x + 3 > 0) ∧ (5 - x^2 > 0)

theorem valid_values (x : ℝ) : 
  is_defined x ↔ (-Real.sqrt 5 < x ∧ x < 1) ∨ (3 < x ∧ x < Real.sqrt 5) := by
  sorry

end valid_values_l456_456207


namespace value_of_a_l456_456752

theorem value_of_a (a : ℝ) : (a^2 - 4) / (a - 2) = 0 → a ≠ 2 → a = -2 :=
by 
  intro h1 h2
  sorry

end value_of_a_l456_456752


namespace solve_for_r_l456_456640

def E (a b c : ℕ) : ℕ := a * b^c

theorem solve_for_r : ∃ r : ℝ, 0 < r ∧ E (r : ℕ) (r : ℕ) 2 = 256 ∧ r = 4 :=
by
  sorry

end solve_for_r_l456_456640


namespace ns_product_l456_456455

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : 
  f (f x + y) = f (x + y) + x^2 * f y - 2 * x * y + x^2 - x + 2

theorem ns_product : 
  let n := 1 in
  let s := 3 in
  n * s = 3 := 
by
  sorry

end ns_product_l456_456455


namespace train_length_is_630_meters_l456_456277

-- Define the conditions and the question
def train_speed_km_per_hr : ℝ := 63 -- Speed in km/hr
def time_seconds : ℝ := 36 -- Time to pass the tree in seconds

-- Define the conversion factor
def km_per_hr_to_m_per_s (speed_km_per_hr : ℝ) : ℝ :=
  speed_km_per_hr * (1000 / 3600)

-- Convert speed to m/s
def train_speed_m_per_s := km_per_hr_to_m_per_s train_speed_km_per_hr

-- Define the length of the train
def train_length (speed_m_per_s : ℝ) (time_s : ℝ) : ℝ :=
  speed_m_per_s * time_s

-- Theorem: Prove that the length of the train is 630 meters
theorem train_length_is_630_meters : train_length train_speed_m_per_s time_seconds = 630 := by
  sorry

end train_length_is_630_meters_l456_456277


namespace xyz_abs_eq_one_l456_456464

theorem xyz_abs_eq_one (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (cond : x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x + 1) : |x * y * z| = 1 :=
sorry

end xyz_abs_eq_one_l456_456464


namespace magnitude_of_vector_l456_456683

open Complex

theorem magnitude_of_vector (z : ℂ) (h : z = 1 - I) : 
  ‖(2 / z + z^2)‖ = Real.sqrt 2 :=
by
  sorry

end magnitude_of_vector_l456_456683


namespace optimal_triangle_game_area_l456_456841

-- Define the main problem conditions and theorems
theorem optimal_triangle_game_area (ABC : Triangle) :
  ∃ N₁ ∈ ABC.AB, ∃ K ∈ ABC.BC, ∃ N₂ ∈ ABC.CA,
  (Nora_maximizes N₁ N₂ ∧ Kristo_minimizes K) →
  Area (Triangle.mk N₁ K N₂) = (1 / 4) * Area ABC := by
  sorry

end optimal_triangle_game_area_l456_456841


namespace circumscribed_quadrilateral_angle_sum_l456_456610

theorem circumscribed_quadrilateral_angle_sum {A B C D O : Type*}
  (h_circumscribed : quadrilateral_circumscribed A B C D O)
  (h_angles : ∀ a b c d, angle O a b = angle O c d) :
  angle A O B + angle C O D = 180 :=
by sorry

end circumscribed_quadrilateral_angle_sum_l456_456610


namespace find_ab_l456_456019

variable (a b : ℝ)

theorem find_ab (h1 : a + b = 4) (h2 : a^3 + b^3 = 136) : a * b = -6 := by
  sorry

end find_ab_l456_456019


namespace sqrt_sum_max_l456_456013

theorem sqrt_sum_max (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  (sqrt a + sqrt b) ≤ sqrt 2 :=
sorry

end sqrt_sum_max_l456_456013


namespace jane_average_speed_correct_l456_456518

noncomputable def jane_average_speed : ℝ :=
  let total_distance : ℝ := 250
  let total_time : ℝ := 6
  total_distance / total_time

theorem jane_average_speed_correct : jane_average_speed = 41.67 := by
  sorry

end jane_average_speed_correct_l456_456518


namespace tangents_are_perpendicular_l456_456789

theorem tangents_are_perpendicular
  (A B C D K L M N T O : Type)
  [cyclic_quadrilateral ABCD]
  [incircle_tangent_points ABCD K L M N]
  [tangency_points KM NL T O] :
  is_perpendicular KM NL :=
sorry

end tangents_are_perpendicular_l456_456789


namespace average_daily_production_correct_l456_456220

noncomputable def average_daily_production : ℝ :=
  let jan_production := 3000
  let monthly_increase := 100
  let total_days := 365
  let total_production := jan_production + (11 * jan_production + (100 * (1 + 11))/2)
  (total_production / total_days : ℝ)

theorem average_daily_production_correct :
  average_daily_production = 121.1 :=
sorry

end average_daily_production_correct_l456_456220


namespace find_radius_l456_456166

theorem find_radius (r n : ℝ) (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) : 
  r = n * (Real.sqrt 3 - 1) / 2 :=
by
  sorry

end find_radius_l456_456166


namespace denomination_is_20_l456_456542

noncomputable def denomination_of_250_coins (x : ℕ) : Prop :=
  250 * x + 84 * 25 = 7100

theorem denomination_is_20 (x : ℕ) (h : denomination_of_250_coins x) : x = 20 :=
by
  sorry

end denomination_is_20_l456_456542


namespace power_function_properties_l456_456182

theorem power_function_properties :
  ∃ α : ℝ, (∀ x : ℝ, x > 0 → (x^α = sqrt x)) ∧
           (∀ x : ℝ, (x > 0) → (∃ c : ℝ, (x = c^α) )) ∧
           ¬ (∀ x : ℝ, f(-x) = f(x)) ∧
           ¬ (∀ x : ℝ, f(-x) = -f(x)) ∧
           (∀ x : ℝ, (x > 0) → (∃ d : ℝ, (f'(x) = d)) ∧ (d > 0)) := 
sorry

end power_function_properties_l456_456182


namespace nearby_island_banana_production_l456_456240

theorem nearby_island_banana_production
  (x : ℕ)
  (h_prod: 10 * x + x = 99000) :
  x = 9000 :=
sorry

end nearby_island_banana_production_l456_456240


namespace DE_length_l456_456778

theorem DE_length 
(h1 : ∠ BAE = 90°) 
(h2 : ∠ CBE = 90°)
(h3 : ∠ DCE = 90°)
(h4 : AE = Real.sqrt 5)
(h5 : AB = Real.sqrt 4) 
(h6 : BC = Real.sqrt 3) 
(h7 : CD = Real.sqrt t)
(h8 : t = 4) : DE = 4 :=
by
  sorry

end DE_length_l456_456778


namespace square_of_chord_length_eq_l456_456549

theorem square_of_chord_length_eq : 
  ∀ (R1 R2 d x : ℝ), 
    R1 = 10 ∧ R2 = 7 ∧ d = 15 ∧
    (d^2 = (R1^2 - x^2) + (R2^2 - x^2) - 2 * R1 * R2 * real.cos (real.pi / 2)) → 
  x^2 = 45 / 28 :=
sorry

end square_of_chord_length_eq_l456_456549


namespace reflection_matrix_over_64_is_correct_l456_456322

def reflection_matrix_over (u : ℝ × ℝ) : matrix (fin 2) (fin 2) ℝ :=
  let a := u.1
  let b := u.2
  let norm_sq := a^2 + b^2
  ![
    [((a^2 - b^2) / norm_sq : ℝ), ((2 * a * b) / norm_sq : ℝ)],
    [((2 * a * b) / norm_sq : ℝ), ((b^2 - a^2) / norm_sq : ℝ)]
  ]

theorem reflection_matrix_over_64_is_correct :
  reflection_matrix_over (6, 4) = (λ i j => ![
    [(5/13 : ℝ), (12/13 : ℝ)],
    [(12/13 : ℝ), -(5/13 : ℝ)]
  ] i j) :=
sorry

end reflection_matrix_over_64_is_correct_l456_456322


namespace expression_equals_one_l456_456471

theorem expression_equals_one (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h_sum : a + b + c = 1) :
  (a^3 * b^3 / ((a^3 - b * c) * (b^3 - a * c)) + a^3 * c^3 / ((a^3 - b * c) * (c^3 - a * b)) +
    b^3 * c^3 / ((b^3 - a * c) * (c^3 - a * b))) = 1 :=
by
  sorry

end expression_equals_one_l456_456471


namespace maximal_pairs_l456_456333

def is_pair {α : Type*} (s : set α) : Prop := ∃ a b : α, a ≠ b ∧ {a, b} = s

def no_common_elements {α : Type*} (pairs : list (set α)) : Prop :=
  ∀ (i j : ℕ) (hi : i < pairs.length) (hj : j < pairs.length), i ≠ j → (pairs.nth_le i hi ∩ pairs.nth_le j hj) = ∅

def all_distinct_sums {α : Type*} [linear_order α] [add_comm_monoid α] (pairs : list (set α)) : Prop :=
  let sums := pairs.map (λ p, ∑ x in p, x) in
  list.nodup sums

def sums_bounded {α : Type*} [linear_order α] [add_comm_monoid α] (pairs : list (set α)) (bound : α) : Prop :=
  ∀ p ∈ pairs, (∑ x in p, x) ≤ bound

theorem maximal_pairs {α : Type*} [linear_ordered_comm_group α] :
  ∃ (k : ℕ), ∀ (pairs : list (set α)),
    (∀ p ∈ pairs, is_pair p) → -- each element in the list is a pair
    no_common_elements pairs → -- no two pairs share common elements
    all_distinct_sums pairs →   -- all pair sums are distinct
    sums_bounded pairs 4019 →   -- each sum is ≤ 4019
    pairs.length ≤ 1607 :=      -- the length of the list is at most 1607
sorry

end maximal_pairs_l456_456333


namespace boulder_splash_width_l456_456910

theorem boulder_splash_width :
  (6 * (1/4) + 3 * (1 / 2) + 2 * b = 7) -> b = 2 := by
  sorry

end boulder_splash_width_l456_456910


namespace tangent_line_at_1_f_positive_iff_a_leq_2_l456_456726

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem tangent_line_at_1 (a : ℝ) (h : a = 4) : 
  ∃ k b : ℝ, (k = -2) ∧ (b = 2) ∧ (∀ x : ℝ, f x a = k * (x - 1) + b) :=
sorry

theorem f_positive_iff_a_leq_2 : 
  (∀ x : ℝ, 1 < x → f x a > 0) ↔ a ≤ 2 :=
sorry

end tangent_line_at_1_f_positive_iff_a_leq_2_l456_456726


namespace pasta_sauce_cost_l456_456100

theorem pasta_sauce_cost :
  let mustard_oil_cost := 2 * 13
  let penne_pasta_cost := 3 * 4
  let total_cost := 50 - 7
  let spent_on_oil_and_pasta := mustard_oil_cost + penne_pasta_cost
  let pasta_sauce_cost := total_cost - spent_on_oil_and_pasta
  pasta_sauce_cost = 5 :=
by
  let mustard_oil_cost := 2 * 13
  let penne_pasta_cost := 3 * 4
  let total_cost := 50 - 7
  let spent_on_oil_and_pasta := mustard_oil_cost + penne_pasta_cost
  let pasta_sauce_cost := total_cost - spent_on_oil_and_pasta
  sorry

end pasta_sauce_cost_l456_456100


namespace sacks_harvested_per_section_l456_456093

theorem sacks_harvested_per_section (total_sacks : ℕ) (sections : ℕ) (sacks_per_section : ℕ) 
  (h1 : total_sacks = 360) 
  (h2 : sections = 8) 
  (h3 : total_sacks = sections * sacks_per_section) :
  sacks_per_section = 45 :=
by sorry

end sacks_harvested_per_section_l456_456093


namespace percentage_calculation_l456_456510

def percentage_less_than_50000_towns : Float := 85

def percentage_less_than_20000_towns : Float := 20
def percentage_20000_to_49999_towns : Float := 65

theorem percentage_calculation :
  percentage_less_than_50000_towns = percentage_less_than_20000_towns + percentage_20000_to_49999_towns :=
by
  sorry

end percentage_calculation_l456_456510
