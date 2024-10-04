import Mathlib

namespace two_point_two_five_as_fraction_l673_673761

theorem two_point_two_five_as_fraction : (2.25 : ℚ) = 9 / 4 := 
by 
  -- Proof steps would be added here
  sorry

end two_point_two_five_as_fraction_l673_673761


namespace wholesale_price_correct_l673_673829

variable (RetailPrice : ℝ) (Discount : ℝ) (ProfitPercentage : ℝ)
variable (SellingPrice : ℝ) (WholesalePrice : ℝ)

axiom h1 : RetailPrice = 144
axiom h2 : Discount = 0.10
axiom h3 : ProfitPercentage = 0.20

lemma calc_selling_price : SellingPrice = RetailPrice - (Discount * RetailPrice) := by
  rw [h1, h2]
  sorry

lemma calc_profit : SellingPrice = WholesalePrice + (ProfitPercentage * WholesalePrice) := by
  rw [h3]
  sorry

theorem wholesale_price_correct : WholesalePrice = 108 := by
  rw [← calc_selling_price, ← calc_profit]
  sorry

end wholesale_price_correct_l673_673829


namespace ring_cone_contact_radius_l673_673398

variable (δ d r m : ℝ) (h : δ < 2 * r)

theorem ring_cone_contact_radius (δ d r m : ℝ) (h : δ < 2 * r) : 
  ∃ 𝜌 : ℝ, 𝜌 = (δ / 2) + ((d - δ) / 4) * (1 - (m / Real.sqrt (m^2 + r^2))) :=
by
  use (δ / 2) + ((d - δ) / 4) * (1 - (m / Real.sqrt (m^2 + r^2)))
  sorry

end ring_cone_contact_radius_l673_673398


namespace special_two_digit_special_four_digit_exists_six_digit_special_exists_twenty_digit_special_at_most_ten_hundred_digit_special_exists_thirty_digit_special_l673_673760

-- Definition of special number
def is_special (n : ℕ) : Prop :=
  let m := (nat.log10 n + 1) / 2 in
  n^(1/2) ∈ ℕ ∧
  (n / 10^m) ∈ ℕ ∧
  (n % 10^m) ∈ ℕ

theorem special_two_digit :
  ∃ (n : ℕ), is_special n ∧ n < 100 := 
sorry

theorem special_four_digit :
  ∃ (n : ℕ), is_special n ∧ n < 10000 := 
sorry

theorem exists_six_digit_special :
  ∃ (n : ℕ), is_special n ∧ 100000 ≤ n ∧ n < 1000000 := 
sorry

theorem exists_twenty_digit_special :
  ∃ (n : ℕ), is_special n ∧ 10^19 ≤ n ∧ n < 10^20 := 
sorry

theorem at_most_ten_hundred_digit_special :
  (∃ (n : ℕ), is_special n ∧ 10^99 ≤ n ∧ n < 10^100) ≤ 10 := 
sorry

theorem exists_thirty_digit_special :
  ∃ (n : ℕ), is_special n ∧ 10^29 ≤ n ∧ n < 10^30 := 
sorry

end special_two_digit_special_four_digit_exists_six_digit_special_exists_twenty_digit_special_at_most_ten_hundred_digit_special_exists_thirty_digit_special_l673_673760


namespace smallest_composite_no_prime_factors_lt_15_l673_673912

theorem smallest_composite_no_prime_factors_lt_15 (n : ℕ) :
  ∀ n, (∀ p : ℕ, p.prime → p ∣ n → 15 ≤ p) → n = 289 → 
       is_composite n ∧ (∀ m : ℕ, (∀ q : ℕ, q.prime → q ∣ m → 15 ≤ q) → m ≥ 289) :=
by
  intros n hv hn
  -- Proof would go here
  sorry

end smallest_composite_no_prime_factors_lt_15_l673_673912


namespace minimum_r_for_three_coloring_of_hexagon_is_three_halves_l673_673624

def hexagon (a : ℝ) := {p : ℝ × ℝ | abs p.1 + abs p.2 ≤ 1 ∧ abs (p.1 - p.2) ≤ 1 ∧ abs (p.1 + p.2) ≤ 1}

noncomputable def minimal_r (s : set (ℝ × ℝ)) : ℝ :=
  Inf {r | ∃ (c : (ℝ × ℝ) → fin 3), ∀ x y ∈ s, c x = c y → dist x y < r}

theorem minimum_r_for_three_coloring_of_hexagon_is_three_halves :
  minimal_r (hexagon 1) = 3 / 2 :=
by sorry

end minimum_r_for_three_coloring_of_hexagon_is_three_halves_l673_673624


namespace number_of_common_tangents_l673_673524

theorem number_of_common_tangents (A B : ℝ × ℝ) (d1 d2 : ℝ) (hA : A = (1, 2)) (hB : B = (4, 1)) (hd1 : d1 = 1) (hd2 : d2 = 2) : 
  ∃ (n : ℕ), n = 4 ∧ ∀ l, line_in_plane l → distance_from_point_to_line A l = d1 → distance_from_point_to_line B l = d2 :=
by
  use 4
  split
  · refl
  · intros l hl hA1 hB2
  sorry

end number_of_common_tangents_l673_673524


namespace length_of_median_B_to_BC_l673_673503

noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

theorem length_of_median_B_to_BC :
  let A := (3, 3, 2) in
  let B := (4, -3, 7) in
  let C := (0, 5, 1) in
  let D := ((B.1 + C.1) / 2, (B.2 + C.2) / 2, (B.3 + C.3) / 2) in
  distance B D = 3 :=
by
  sorry

end length_of_median_B_to_BC_l673_673503


namespace total_flour_correct_l673_673146

-- Define the quantities specified in the conditions
def cups_of_flour_already_added : ℕ := 2
def cups_of_flour_to_add : ℕ := 7

-- Define the total cups of flour required by the recipe as a sum of the quantities
def cups_of_flour_required : ℕ := cups_of_flour_already_added + cups_of_flour_to_add

-- Prove that the total cups of flour required is 9
theorem total_flour_correct : cups_of_flour_required = 9 := by
  -- use auto proof placeholder
  rfl

end total_flour_correct_l673_673146


namespace largest_divisor_of_consecutive_five_l673_673292

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l673_673292


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673217

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673217


namespace largest_divisor_of_5_consecutive_integers_l673_673301

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l673_673301


namespace well_centered_subpolygons_decomposable_iff_l673_673614

open Real

noncomputable def is_regular_ngon (n : ℕ) (polygon : set (ℝ × ℝ)) : Prop :=
  n > 2 ∧ n % 2 = 1 ∧ is_regular_polygon n polygon ∧ centroid polygon = (0, 0)

noncomputable def is_subpolygon (polygon subpolygon : set (ℝ × ℝ)) : Prop :=
  subpolygon ⊆ polygon ∧ (3 ≤ subpolygon.card)

noncomputable def is_well_centered (subpolygon : set (ℝ × ℝ)) : Prop :=
  centroid subpolygon = (0, 0)

noncomputable def is_decomposable (subpolygon : set (ℝ × ℝ)) : Prop :=
  ∃(polygons : list (set (ℝ × ℝ))), (∀ (p ∈ polygons), 3 ≤ p.card ∧ is_regular_polygon p.card p) ∧ disjoint_union polygons = subpolygon

theorem well_centered_subpolygons_decomposable_iff (n : ℕ) (polygon : set (ℝ × ℝ))
  (H₁ : is_regular_ngon n polygon) :
  (∀ subpolygon, is_subpolygon polygon subpolygon ∧ is_well_centered subpolygon → is_decomposable subpolygon) ↔
  (∀ p q r : ℕ, p.prime → q.prime → r.prime → p ≠ q → q ≠ r → p ≠ r → p * q * r ≠ n) :=
sorry

end well_centered_subpolygons_decomposable_iff_l673_673614


namespace distance_A_B_of_squares_l673_673711

theorem distance_A_B_of_squares 
  (perimeter_smaller_square : ℝ)
  (area_larger_square : ℝ)
  (h1 : perimeter_smaller_square = 8)
  (h2 : area_larger_square = 64)
  : distance (A : ℝ × ℝ) (B : ℝ × ℝ) = 10 :=
by
  -- Proof is omitted
  sorry

end distance_A_B_of_squares_l673_673711


namespace exists_perpendicular_line_in_plane_l673_673786

variables {Line : Type} {Plane : Type}
variables (l : Line) (α : Plane)

-- Define the existence of a line within a plane
axiom exists_line_in_plane : Plane → ∃ m : Line, m ∈ α

-- Define the relationship of a line being perpendicular to another line
axiom perp : Line → Line → Prop

-- Define the membership of a line in a plane
axiom in_plane : Line → Plane → Prop

theorem exists_perpendicular_line_in_plane (l : Line) (α : Plane) :
  ∃ m : Line, in_plane m α ∧ perp m l := by
  sorry

end exists_perpendicular_line_in_plane_l673_673786


namespace find_p_l673_673141

def T : Set ℕ := {d | ∃ b c d, d = 2^b * 3^c * 5^d ∧ 0 ≤ b ∧ b ≤ 8 ∧ 0 ≤ c ∧ c ≤ 8 ∧ 0 ≤ d ∧ d ≤ 8}

def exponents (a : ℕ) : ℕ × ℕ × ℕ :=
  nat.fold 30 2 ∘ nat.fold 3 ∘ nat.fold 5 a

def is_divisor_chain (a1 a2 a3 : ℕ) : Prop :=
  let ⟨b1, c1, d1⟩ := exponents a1 in
  let ⟨b2, c2, d2⟩ := exponents a2 in
  let ⟨b3, c3, d3⟩ := exponents a3 in
  b1 ≤ b2 ∧ b2 ≤ b3 ∧ c1 ≤ c2 ∧ c2 ≤ c3 ∧ d1 ≤ d2 ∧ d2 ≤ d3

noncomputable def p : ℕ := 64000

theorem find_p :
  ∃ q ∈ ℕ, (∀ p q : ℕ, nat.coprime p q → 
  let num_possible_values := (Set.toFinset T).card in
  let num_favorable_values := (Set.toFinset {a1 a2 a3 | a1 ∈ T ∧ a2 ∈ T ∧ a3 ∈ T ∧ is_divisor_chain a1 a2 a3}).card in
  (num_favorable_values : ℝ) / (num_possible_values ^ 3) = p / q) :=
sorry

end find_p_l673_673141


namespace ellipse_equation_max_area_triangle_PAB_l673_673144

theorem ellipse_equation (a b : ℝ) (a_gt_0 : 0 < a) (b_gt_0 : 0 < b) (a_gt_b : a > b)
  (ellipse_eq : ∀ x y, (y^2 / a^2) + (x^2 / b^2) = 1)
  (hyperbola_eq : ∀ x y, x^2 - y^2 = 1) 
  (eccentricity_cond : (a^2 - b^2) = 2)
  (circle_eq : ∀ x y, x^2 + y^2 = 4) : 
  (b = √2 ∧ a = 2) → (∀ x y, (y^2 / 4) + (x^2 / 2) = 1) := by
  sorry

theorem max_area_triangle_PAB (m : ℝ)
  (a b : ℝ) (a_gt_0 : 0 < a) (b_gt_0 : 0 < b) (a_gt_b : a > b)
  (ellipse_eq : ∀ x y, (y^2 / a^2) + (x^2 / b^2) = 1)
  (hyperbola_eq : ∀ x y, x^2 - y^2 = 1) 
  (eccentricity_cond : (a^2 - b^2) = 2)
  (circle_eq : ∀ x y, x^2 + y^2 = 4)
  (line_eq : ∀ x, y = sqrt(2) * x + m)
  (point_on_ellipse : ∀ x y, (x = 1 ∧ y = sqrt(2)) → ((y^2 / a^2) + (x^2 / b^2) = 1)) :
  (sqrt(2) = 2) → max_area_PAB = sqrt(2) := by
  sorry

end ellipse_equation_max_area_triangle_PAB_l673_673144


namespace maria_bottles_count_l673_673639

-- Definitions from the given conditions
def b_initial : ℕ := 23
def d : ℕ := 12
def g : ℕ := 5
def b : ℕ := 65

-- Definition of the question based on conditions
def b_final : ℕ := b_initial - d - g + b

-- The statement to prove the correctness of the answer
theorem maria_bottles_count : b_final = 71 := by
  -- We skip the proof for this statement
  sorry

end maria_bottles_count_l673_673639


namespace dani_pants_after_5_years_l673_673869

theorem dani_pants_after_5_years :
  ∀ (pairs_per_year : ℕ) (pants_per_pair : ℕ) (initial_pants : ℕ) (years : ℕ),
  pairs_per_year = 4 →
  pants_per_pair = 2 →
  initial_pants = 50 →
  years = 5 →
  initial_pants + years * (pairs_per_year * pants_per_pair) = 90 :=
by sorry

end dani_pants_after_5_years_l673_673869


namespace part_I_part_II_l673_673484

set_option maxRecDepth 1000
open Set

variable (a : ℝ)

def setA := { x : ℝ | a ≤ x ∧ x < 7 }
def setB := { x : ℝ | 2 < x ∧ x < 10 }

theorem part_I (h : a = 3) :
  ( setA a ∪ setB = (2 : ℝ, 10) ) ∧ ( setB ∩ ((set.univ : Set ℝ) \ setA a) = (2 : ℝ, 3) ∪ [7, 10) ) :=
by
  sorry

theorem part_II :
  (2 : ℝ) < a ↔ setA a ⊆ setB :=
by
  sorry

end part_I_part_II_l673_673484


namespace common_root_l673_673667

def f (x : ℝ) : ℝ := x^4 - x^3 - 22 * x^2 + 16 * x + 96
def g (x : ℝ) : ℝ := x^3 - 2 * x^2 - 3 * x + 10

theorem common_root :
  f (-2) = 0 ∧ g (-2) = 0 := by
  sorry

end common_root_l673_673667


namespace symmetrical_point_with_respect_to_x_axis_l673_673102

-- Define the point P with coordinates (-2, -1)
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the given point
def P : Point := { x := -2, y := -1 }

-- Define the symmetry with respect to the x-axis
def symmetry_x_axis (p : Point) : Point :=
{ x := p.x, y := -p.y }

-- Verify the symmetrical point
theorem symmetrical_point_with_respect_to_x_axis :
  symmetry_x_axis P = { x := -2, y := 1 } :=
by
  -- Skip the proof
  sorry

end symmetrical_point_with_respect_to_x_axis_l673_673102


namespace sum_row_of_pascal_triangle_leq_n_plus_2_times_2_pow_n_minus_1_l673_673666

theorem sum_row_of_pascal_triangle_leq_n_plus_2_times_2_pow_n_minus_1 (n : ℕ) :
  (∑ k in finset.range (n + 1), nat.choose n k) ≤ (n + 2) * 2^(n - 1) :=
sorry

end sum_row_of_pascal_triangle_leq_n_plus_2_times_2_pow_n_minus_1_l673_673666


namespace at_least_30_cents_probability_l673_673677

theorem at_least_30_cents_probability :
  let penny := 1
  let nickel := 5
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  let all_possible_outcomes := 2^5
  let successful_outcomes := 
    -- Half-dollar and quarter heads: 2^3 = 8 combinations
    2^3 + 
    -- Quarter heads and half-dollar tails (nickel and dime heads): 2 combinations
    2^1 + 
    -- Quarter tails and half-dollar heads: 2^3 = 8 combinations
    2^3
  let probability := successful_outcomes / all_possible_outcomes
  probability = 9 / 16 :=
by
  -- Proof goes here
  sorry

end at_least_30_cents_probability_l673_673677


namespace largest_divisor_of_consecutive_product_l673_673224

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l673_673224


namespace find_angle_C_find_angle_C_2_find_angle_C_3_range_of_4sinB_minus_a_l673_673850

variable {A B C a b c : ℝ}
variable (h1 : c^2 + a * b = c * (a * real.cos B - b * real.cos A) + 2 * b^2)
variable (h2 : (b + c) * (real.sin B - real.sin C) = -a * (real.sin A - real.sin B))
variable (h3 : b * real.sin C = real.sqrt 3 * (a - c * real.cos B))

theorem find_angle_C (h : c^2 + a * b = c * (a * real.cos B - b * real.cos A) + 2 * b^2) : C = real.pi / 3 :=
sorry

theorem find_angle_C_2 (h : (b + c) * (real.sin B - real.sin C) = -a * (real.sin A - real.sin B)) : C = real.pi / 3 :=
sorry

theorem find_angle_C_3 (h : b * real.sin C = real.sqrt 3 * (a - c * real.cos B)) : C = real.pi / 3 :=
sorry

theorem range_of_4sinB_minus_a (hC : C = real.pi / 3) (hc : c = 2 * real.sqrt 3) : 
  -2 * real.sqrt 3 < 4 * real.sin B - a ∧ 4 * real.sin B - a < 2 * real.sqrt 3 :=
sorry

end find_angle_C_find_angle_C_2_find_angle_C_3_range_of_4sinB_minus_a_l673_673850


namespace max_min_y_l673_673823

def g (t : ℝ) : ℝ := 80 - 2 * t

def f (t : ℝ) : ℝ := 20 - |t - 10|

def y (t : ℝ) : ℝ := g t * f t

theorem max_min_y (t : ℝ) (h : 0 ≤ t ∧ t ≤ 20) :
  (y t = 1200 → t = 10) ∧ (y t = 400 → t = 20) :=
by
  sorry

end max_min_y_l673_673823


namespace cubic_monomial_l673_673651

-- Definitions for the conditions
def is_monomial (m : ℕ → ℕ) : Prop := ∃ f : ℕ, m f ∈ {1, 0}
def has_coefficient (m : ℕ → ℕ) (c : ℤ) : Prop := c = -2
def has_variables (m : ℕ → ℕ) : Prop := ∃ f, m f ∈ {x, y}
def is_cubic (mx my : ℕ) : Prop := mx + my = 3

-- The specific monomial we're checking
def specific_monomial (mx my : ℕ) : ℤ :=
  -2 * (mx ^ 2) * my

-- Statement to prove the problem
theorem cubic_monomial (mx my : ℕ) (c : ℤ) :
  is_monomial (λ n, mx + my) →
  has_coefficient (λ n, mx + my) c →
  has_variables (λ n, mx + my) →
  is_cubic mx my →
  specific_monomial mx my = -2 * x^2 * y :=
by
  sorry

end cubic_monomial_l673_673651


namespace trihedral_sum_of_angles_le_sum_of_plane_angles_trihedral_sum_of_angles_ge_half_sum_of_plane_angles_l673_673353

-- Part a
theorem trihedral_sum_of_angles_le_sum_of_plane_angles
  (α β γ : ℝ) (ASB BSC CSA : ℝ)
  (h1 : α ≤ ASB)
  (h2 : β ≤ BSC)
  (h3 : γ ≤ CSA) :
  α + β + γ ≤ ASB + BSC + CSA :=
sorry

-- Part b
theorem trihedral_sum_of_angles_ge_half_sum_of_plane_angles
  (α_S β_S γ_S : ℝ) (ASB BSC CSA : ℝ) 
  (h_acute : ASB < (π / 2) ∧ BSC < (π / 2) ∧ CSA < (π / 2))
  (h1 : α_S ≥ (1/2) * ASB)
  (h2 : β_S ≥ (1/2) * BSC)
  (h3 : γ_S ≥ (1/2) * CSA) :
  α_S + β_S + γ_S ≥ (1/2) * (ASB + BSC + CSA) :=
sorry

end trihedral_sum_of_angles_le_sum_of_plane_angles_trihedral_sum_of_angles_ge_half_sum_of_plane_angles_l673_673353


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673311

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673311


namespace younger_person_age_l673_673169

theorem younger_person_age (e y : ℕ) 
  (h1: e = y + 20)
  (h2: e - 10 = 5 * (y - 10)) : 
  y = 15 := 
by
  sorry

end younger_person_age_l673_673169


namespace find_m_eq_2_l673_673718

theorem find_m_eq_2 :
  ∃ m : ℝ, 
    (∀ α β : ℝ, 
      (α + β = m + 2 ∧ α * β = m^2) → 
      (m + 2 = m^2 ∧ 
        (m + 2) ^ 2 - 4 * m^2 ≥ 0)) ∧ 
    m = 2 :=
begin
  sorry
end

end find_m_eq_2_l673_673718


namespace count_x0_eq_x6_l673_673464

noncomputable def seq_x (x : ℝ) : ℕ → ℝ
| 0       := x
| (n + 1) := if 3 * seq_x n < 1 then 3 * seq_x n
             else if 3 * seq_x n < 2 then 3 * seq_x n - 1
             else 3 * seq_x n - 2

theorem count_x0_eq_x6 : finset.card { x : ℝ | 0 ≤ x ∧ x < 1 ∧ seq_x x 0 = seq_x x 6 } = 729 := 
by
    sorry

end count_x0_eq_x6_l673_673464


namespace sum_cos_squares_l673_673951

noncomputable def calc_sum_cos_squares (n : ℕ) : ℝ :=
  (∑ k in Finset.range (n + 1), Real.cos (k * Real.pi / (2 * n)) ^ 2)

theorem sum_cos_squares (n : ℕ) : calc_sum_cos_squares n = (n - 1) / 2 := 
by
  sorry

end sum_cos_squares_l673_673951


namespace price_per_litre_mixed_oil_l673_673743

-- Define the given conditions
def cost_oil1 : ℝ := 100 * 45
def cost_oil2 : ℝ := 30 * 57.50
def cost_oil3 : ℝ := 20 * 72
def total_cost : ℝ := cost_oil1 + cost_oil2 + cost_oil3
def total_volume : ℝ := 100 + 30 + 20

-- Define the statement to be proved
theorem price_per_litre_mixed_oil : (total_cost / total_volume) = 51.10 :=
by
  sorry

end price_per_litre_mixed_oil_l673_673743


namespace smallest_composite_no_prime_factors_below_15_correct_l673_673931

def smallest_composite_no_prime_factors_below_15 : Nat :=
  323
  
theorem smallest_composite_no_prime_factors_below_15_correct :
  (∀ n < 15, Prime n → ¬ (n ∣ smallest_composite_no_prime_factors_below_15)) ∧
  (∃ p q, Prime p ∧ Prime q ∧ p ≠ q ∧ smallest_composite_no_prime_factors_below_15 = p * q) :=
by
  -- Proof skipped
  sorry

end smallest_composite_no_prime_factors_below_15_correct_l673_673931


namespace midpoint_chord_hyperbola_l673_673088

theorem midpoint_chord_hyperbola (a b : ℝ) : 
  (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → (∃ (mx my : ℝ), (mx / a^2 + my / b^2 = 0) ∧ (mx = x / 2) ∧ (my = y / 2))) →
  ∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) →
  ∃ (mx my : ℝ), (mx / a^2 - my / b^2 = 0) ∧ (mx = x / 2) ∧ (my = y / 2) := 
sorry

end midpoint_chord_hyperbola_l673_673088


namespace positive_integer_solutions_l673_673886

theorem positive_integer_solutions (n : ℕ) (a : ℕ → ℕ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ∃ k : fin n, a k = i ∧ ∀ j : fin n, i ∣ a j ↔ j = k) →
  (n = 1 ∧ a 0 = 1) ∨ (n = 2 ∧ a 0 = 2 ∧ a 1 = 1) := sorry

end positive_integer_solutions_l673_673886


namespace right_triangle_shorter_leg_l673_673561

theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : c = 65) : a = 25 ∨ b = 25 := 
by
  sorry

end right_triangle_shorter_leg_l673_673561


namespace sufficient_not_necessary_condition_l673_673986

-- Definition of the proposition p
def prop_p (m : ℝ) := ∀ x : ℝ, x^2 - 4 * x + 2 * m ≥ 0

-- Statement of the proof problem
theorem sufficient_not_necessary_condition (m : ℝ) : 
  (m ≥ 3 → prop_p m) ∧ ¬(m ≥ 3 → m ≥ 2) ∧ (m ≥ 2 → prop_p m) → (m ≥ 3 → prop_p m) ∧ ¬(m ≥ 3 ↔ prop_p m) :=
sorry

end sufficient_not_necessary_condition_l673_673986


namespace min_sum_of_squares_l673_673629

theorem min_sum_of_squares (a b c d : ℝ) (h : a + 3 * b + 5 * c + 7 * d = 14) : 
  a^2 + b^2 + c^2 + d^2 ≥ 7 / 3 :=
sorry

end min_sum_of_squares_l673_673629


namespace canadian_ratio_correct_l673_673152

-- The total number of scientists
def total_scientists : ℕ := 70

-- Half of the scientists are from Europe
def european_scientists : ℕ := total_scientists / 2

-- The number of scientists from the USA
def usa_scientists : ℕ := 21

-- The number of Canadian scientists
def canadian_scientists : ℕ := total_scientists - european_scientists - usa_scientists

-- The ratio of the number of Canadian scientists to the total number of scientists
def canadian_ratio : ℚ := canadian_scientists / total_scientists

-- Prove that the ratio is 1:5
theorem canadian_ratio_correct : canadian_ratio = 1 / 5 :=
by
  sorry

end canadian_ratio_correct_l673_673152


namespace skittles_total_l673_673116

-- Define the conditions
def skittles_per_friend : ℝ := 40.0
def number_of_friends : ℝ := 5.0

-- Define the target statement using the conditions
theorem skittles_total : (skittles_per_friend * number_of_friends = 200.0) :=
by 
  -- Using sorry to placeholder the proof
  sorry

end skittles_total_l673_673116


namespace shorter_leg_of_right_triangle_l673_673555

theorem shorter_leg_of_right_triangle (a b : ℕ) (h : a^2 + b^2 = 65^2) : min a b = 16 :=
sorry

end shorter_leg_of_right_triangle_l673_673555


namespace probability_cocaptains_l673_673196

/-- 
There are 4 math teams in the area, with 6, 8, 9, and 10 students, respectively. 
Each team has three co-captains. If two teams are selected randomly, 
and from one of these teams, two members are selected randomly, 
the probability that both selected members are co-captains is 131/1680.
-/
theorem probability_cocaptains :
  let teams := [6, 8, 9, 10] in
  let co_captains := 3 in
  (∑ pair in (teams.zipWith teams (λ a b, (a, b))),
    let (team1, team2) := pair,
    let prob_team1 := 3 / (team1 * (team1 - 1) / 2),
    let prob_team2 := 3 / (team2 * (team2 - 1) / 2),
    (prob_team1 + prob_team2) / 2 / 6) = 131 / 1680 :=
sorry

end probability_cocaptains_l673_673196


namespace largest_divisor_of_product_of_5_consecutive_integers_l673_673236

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l673_673236


namespace choose_four_cards_different_suits_l673_673527

theorem choose_four_cards_different_suits : 
  let num_suits := 4,
      cards_per_suit := 13 in
  (cards_per_suit^num_suits) = 28561 := by
  sorry

end choose_four_cards_different_suits_l673_673527


namespace geometric_sequence_sum_n5_l673_673992

def geometric_sum (a₁ q : ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_sum_n5 (a₁ q : ℕ) (n : ℕ) (h₁ : a₁ = 3) (h₂ : q = 4) (h₃ : n = 5) : 
  geometric_sum a₁ q n = 1023 :=
by
  sorry

end geometric_sequence_sum_n5_l673_673992


namespace area_above_line_in_circle_l673_673762

-- Define the circle by its given equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4 * x + y^2 - 8 * y + 12 = 0

-- Define the line equation
def line_y_eq_3 (y : ℝ) : Prop :=
  y = 3

-- Define the problem statement
theorem area_above_line_in_circle :
  (∃ (radius : ℝ) (center_x center_y : ℝ),
    (∀ x y : ℝ, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    (∃ (area : ℝ),
      (∀ x y : ℝ, line_y_eq_3 y ↔ y = 3) ∧
      area = (π * radius^2 / 2) + radius^2 * (acos (1 / radius) - (1 / 2) * sqrt (3)) ∧
      area = (10 * π / 3) + sqrt 3)) :=
sorry

end area_above_line_in_circle_l673_673762


namespace non_planar_characterization_l673_673198

-- Definitions:
structure Graph where
  V : ℕ
  E : ℕ
  F : ℕ

def is_planar (G : Graph) : Prop :=
  G.V - G.E + G.F = 2

def edge_inequality (G : Graph) : Prop :=
  G.E ≤ 3 * G.V - 6

def has_subgraph_K5_or_K33 (G : Graph) : Prop := sorry -- Placeholder for the complex subgraph check

-- Theorem statement:
theorem non_planar_characterization (G : Graph) (hV : G.V ≥ 3) :
  ¬ is_planar G ↔ ¬ edge_inequality G ∨ has_subgraph_K5_or_K33 G := sorry

end non_planar_characterization_l673_673198


namespace calculate_dani_pants_l673_673873

theorem calculate_dani_pants : ∀ (initial_pants number_years pairs_per_year pants_per_pair : ℕ), 
  initial_pants = 50 →
  number_years = 5 →
  pairs_per_year = 4 →
  pants_per_pair = 2 →
  initial_pants + (number_years * (pairs_per_year * pants_per_pair)) = 90 :=
by
  intros initial_pants number_years pairs_per_year pants_per_pair
  intro h_initial_pants h_number_years h_pairs_per_year h_pants_per_pair
  rw [h_initial_pants, h_number_years, h_pairs_per_year, h_pants_per_pair]
  norm_num
  sorry

end calculate_dani_pants_l673_673873


namespace calculate_dani_pants_l673_673874

theorem calculate_dani_pants : ∀ (initial_pants number_years pairs_per_year pants_per_pair : ℕ), 
  initial_pants = 50 →
  number_years = 5 →
  pairs_per_year = 4 →
  pants_per_pair = 2 →
  initial_pants + (number_years * (pairs_per_year * pants_per_pair)) = 90 :=
by
  intros initial_pants number_years pairs_per_year pants_per_pair
  intro h_initial_pants h_number_years h_pairs_per_year h_pants_per_pair
  rw [h_initial_pants, h_number_years, h_pairs_per_year, h_pants_per_pair]
  norm_num
  sorry

end calculate_dani_pants_l673_673874


namespace grant_current_age_l673_673585

variable (G : ℕ) -- Grant's current age

-- Condition: The hospital is currently 40 years old.
def hospital_age_current : ℕ := 40

-- Condition: In five years, Grant will be 2/3 the age of the hospital.
def grant_age_in_5_years : ℕ := (2 / 3 : ℚ) * (hospital_age_current + 5 : ℚ)

theorem grant_current_age : G = 25 :=
by
  -- Calculation: Verify that G is 25, given the conditions.
  have h_age_in_5_years : grant_age_in_5_years = 30 := by sorry
  have G_in_5_years := G + 5
  rw [h_age_in_5_years] at G_in_5_years
  linarith

end grant_current_age_l673_673585


namespace line_through_P_with_equal_intercepts_l673_673863

theorem line_through_P_with_equal_intercepts (a b : ℝ) (P : ℝ × ℝ) (line_eq : ℝ → ℝ → ℝ) :
  P = (2, 3) →
  (∀ {x y : ℝ}, line_eq x y = 0 → (line_eq 0 b = 0) ∧ (line_eq a 0 = 0) → a = b) →
  (line_eq 0 0 = 0 ∨ line_eq a a = 0 ∨ line_eq a (-a) = 0) →
  (line_eq = λ x y, x + y - 5) ∨ (line_eq = λ x y, 3 * x - 2 * y) ↔
  (line_eq 2 3 = 0) :=
by
  sorry

end line_through_P_with_equal_intercepts_l673_673863


namespace smallest_composite_no_prime_under_15_correct_l673_673934

-- Define the concept of a composite number
def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

-- Define the concept of having no prime factors less than 15
def has_no_prime_factors_less_than_15 (n : ℕ) : Prop := 
  ∀ p : ℕ, p.prime ∧ p ∣ n → 15 ≤ p

-- Define the smallest composite number with no prime factors less than 15 
def smallest_composite_no_prime_under_15 : ℕ := 289

-- Prove that this is the smallest number satisfying our conditions
theorem smallest_composite_no_prime_under_15_correct : 
  is_composite smallest_composite_no_prime_under_15 ∧
  has_no_prime_factors_less_than_15 smallest_composite_no_prime_under_15 ∧
  ∀ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than_15 n → n ≥ smallest_composite_no_prime_under_15 :=
by 
  sorry

end smallest_composite_no_prime_under_15_correct_l673_673934


namespace triangle_inequality_l673_673655

theorem triangle_inequality
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : c > 0)
  (h5 : a + b > c)
  (h6 : a + c > b)
  (h7 : b + c > a) :
  a^2 + b^2 + c^2 < 2 * (1 - a * b * c) :=
sorry

end triangle_inequality_l673_673655


namespace smallest_composite_no_prime_factors_lt_15_l673_673909

theorem smallest_composite_no_prime_factors_lt_15 (n : ℕ) :
  ∀ n, (∀ p : ℕ, p.prime → p ∣ n → 15 ≤ p) → n = 289 → 
       is_composite n ∧ (∀ m : ℕ, (∀ q : ℕ, q.prime → q ∣ m → 15 ≤ q) → m ≥ 289) :=
by
  intros n hv hn
  -- Proof would go here
  sorry

end smallest_composite_no_prime_factors_lt_15_l673_673909


namespace largest_divisor_of_consecutive_product_l673_673226

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l673_673226


namespace largest_integer_dividing_consecutive_product_l673_673251

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l673_673251


namespace compound_interest_rate_l673_673405

-- Definition of variables based on the problem
def P : ℝ := 6500
def t : ℝ := 2
def A : ℝ := 7372.46
def n : ℝ := 1

-- Statement to prove the correct compound interest rate
theorem compound_interest_rate (r : ℝ) (h : A = P * (1 + r / n)^(n * t)) : r ≈ 0.0664 := by
  sorry

end compound_interest_rate_l673_673405


namespace cost_price_l673_673063

theorem cost_price (C : ℝ) : 
  (0.05 * C = 350 - 340) → C = 200 :=
by
  assume h1 : 0.05 * C = 10
  sorry

end cost_price_l673_673063


namespace dinner_serving_problem_l673_673751

theorem dinner_serving_problem : 
  let orders := ["B", "B", "B", "B", "C", "C", "C", "C", "F", "F", "F", "F"].to_finset in
  let possible_serving_count := choose 12 2 * 160 in
  ∃ (serving : set (fin 12)), 
    (serving : cardinal) = 2 ∧
    (orders = serving) →
    possible_serving_count = 211200
:= 
begin
  sorry
end

end dinner_serving_problem_l673_673751


namespace zombies_less_than_50_four_days_ago_l673_673730

theorem zombies_less_than_50_four_days_ago
  (curr_zombies : ℕ)
  (days_ago : ℕ)
  (half_rate : ℕ)
  (initial_zombies : ℕ)
  (h_initial : curr_zombies = 480)
  (h_half : half_rate = 2)
  (h_days : days_ago = 4)
  : (curr_zombies / half_rate^days_ago) < 50 :=
by
  have h1 : curr_zombies / half_rate^1 = 480 / 2 := sorry
  have h2 : curr_zombies / half_rate^2 = 480 / 2^2 := sorry
  have h3 : curr_zombies / half_rate^3 = 480 / 2^3 := sorry
  have h4 : curr_zombies / half_rate^4 = 480 / 2^4 := sorry
  show 30 < 50 from sorry
  rw h_initial at *
  sorry

end zombies_less_than_50_four_days_ago_l673_673730


namespace box_contains_1_8_grams_child_ingests_0_1_grams_l673_673799

-- Define the conditions
def packet_weight : ℝ := 0.2
def packets_in_box : ℕ := 9
def half_a_packet : ℝ := 0.5

-- Prove that a box contains 1.8 grams of "acetaminophen"
theorem box_contains_1_8_grams : packets_in_box * packet_weight = 1.8 :=
by
  sorry

-- Prove that a child will ingest 0.1 grams of "acetaminophen" if they take half a packet
theorem child_ingests_0_1_grams : half_a_packet * packet_weight = 0.1 :=
by
  sorry

end box_contains_1_8_grams_child_ingests_0_1_grams_l673_673799


namespace brick_requirement_l673_673376

noncomputable def volume (length width height : ℕ) : ℕ := length * width * height

noncomputable def numberOfBricks (V_wall V_brick : ℕ) : ℕ := V_wall / V_brick

theorem brick_requirement :
  let brick_length := 20
  let brick_width := 10
  let brick_height := 7.5
  let wall_length := 2300
  let wall_height := 200
  let wall_width := 75
  let V_brick := volume brick_length brick_width (brick_height.to_nat)
  let V_wall := volume wall_length wall_height wall_width
  numberOfBricks V_wall V_brick = 23000 :=
by
  let brick_length := 20
  let brick_width := 10
  let brick_height := 7.5
  let wall_length := 2300
  let wall_height := 200
  let wall_width := 75
  let V_brick := volume brick_length brick_width (brick_height.to_nat)
  let V_wall := volume wall_length wall_height wall_width
  show numberOfBricks V_wall V_brick = 23000
  sorry

end brick_requirement_l673_673376


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673314

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673314


namespace hexagon_area_in_rectangle_l673_673485

theorem hexagon_area_in_rectangle (AD AB : ℝ) (H_AD : AD = 4) (H_AB : AB = 2) :
  let s := AB / 2 in
  s = 2 →
  let area_one_triangle := (sqrt 3) / 4 * s^2 in
  let total_area := 6 * area_one_triangle in
  total_area = 6 * sqrt 3 :=
by
  intros
  sorry

end hexagon_area_in_rectangle_l673_673485


namespace smallest_composite_proof_l673_673941

noncomputable def smallest_composite_no_prime_factors_less_than_15 : ℕ :=
  289

theorem smallest_composite_proof :
  smallest_composite_no_prime_factors_less_than_15 = 289 :=
by
  sorry

end smallest_composite_proof_l673_673941


namespace dani_pants_after_5_years_l673_673867

theorem dani_pants_after_5_years :
  ∀ (pairs_per_year : ℕ) (pants_per_pair : ℕ) (initial_pants : ℕ) (years : ℕ),
  pairs_per_year = 4 →
  pants_per_pair = 2 →
  initial_pants = 50 →
  years = 5 →
  initial_pants + years * (pairs_per_year * pants_per_pair) = 90 :=
by sorry

end dani_pants_after_5_years_l673_673867


namespace student_made_mistake_l673_673402

theorem student_made_mistake (AB CD MLNKT : ℕ) (h1 : 10 ≤ AB ∧ AB ≤ 99) (h2 : 10 ≤ CD ∧ CD ≤ 99) (h3 : 10000 ≤ MLNKT ∧ MLNKT < 100000) : AB * CD ≠ MLNKT :=
by {
  sorry
}

end student_made_mistake_l673_673402


namespace problem1_cond1_problem1_cond2_problem1_cond3_problem2_l673_673852

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Given the sides opposite to angles A, B, C are a, b, c respectively

-- Condition 1
axiom cond1 : c^2 + a * b = c * (a * Real.cos B - b * Real.cos A) + 2 * b^2

-- Condition 2
axiom cond2 : (b + c) * (Real.sin B - Real.sin C) = -a * (Real.sin A - Real.sin B)

-- Condition 3
axiom cond3 : b * Real.sin C = Real.sqrt 3 * (a - c * Real.cos B)

-- Problem 1: Show that C = π/3
theorem problem1_cond1 (h : cond1) : C = Real.pi / 3 := sorry
theorem problem1_cond2 (h : cond2) : C = Real.pi / 3 := sorry
theorem problem1_cond3 (h : cond3) : C = Real.pi / 3 := sorry

-- Problem 2: Show that, if c = 2 * sqrt 3, the range of values for 4 * sin B - a is (-2 * sqrt 3, 2 * sqrt 3)
theorem problem2 (hc : c = 2 * Real.sqrt 3) (hC : C = Real.pi / 3) 
: -2 * Real.sqrt 3 < 4 * Real.sin B - a ∧ 4 * Real.sin B - a < 2 * Real.sqrt 3 := sorry

end problem1_cond1_problem1_cond2_problem1_cond3_problem2_l673_673852


namespace least_integer_months_l673_673128

theorem least_integer_months (t : ℕ) : (1.06 ^ t > 3) → t ≥ 20 :=
by
  sorry

end least_integer_months_l673_673128


namespace find_m_l673_673529

theorem find_m (x y m : ℝ) (hx : x = 1) (hy : y = 2) (h : m * x + 2 * y = 6) : m = 2 :=
by sorry

end find_m_l673_673529


namespace non_empty_disjoint_subsets_remainder_of_T_l673_673621

theorem non_empty_disjoint_subsets_remainder_of_T :
  let T := finset.range 15 in
  let m := ((3^15 - 2 * 2^15 + 1) / 2) in
  m % 1000 = 686 :=
by sorry

end non_empty_disjoint_subsets_remainder_of_T_l673_673621


namespace quadr_root_q_l673_673027

theorem quadr_root_q (q : ℝ) (h : ∀ x : ℂ, (2*x^2 + 12*x + q) = 0 → (x = -3 + 2*complex.I)) :
  q = 26 :=
sorry

end quadr_root_q_l673_673027


namespace votes_calculation_l673_673087

noncomputable def total_votes : ℕ := 1680000
noncomputable def invalid_votes_percentage : ℝ := 0.3
noncomputable def valid_votes : ℝ := total_votes * (1 - invalid_votes_percentage)
noncomputable def candidate_a_percentage : ℝ := 0.42
noncomputable def votes_for_a : ℝ := valid_votes * candidate_a_percentage
noncomputable def remaining_votes : ℝ := valid_votes * (1 - candidate_a_percentage)
noncomputable def ratio_bcd : ℝ × ℝ × ℝ := (3, 2, 1)
noncomputable def sum_ratios : ℝ := ratio_bcd.1 + ratio_bcd.2 + ratio_bcd.3
noncomputable def votes_for_b : ℝ := remaining_votes * (ratio_bcd.1 / sum_ratios)
noncomputable def votes_for_c : ℝ := remaining_votes * (ratio_bcd.2 / sum_ratios)
noncomputable def votes_for_d : ℝ := remaining_votes * (ratio_bcd.3 / sum_ratios)

theorem votes_calculation :
  votes_for_a = 493920 ∧
  votes_for_b ≈ 340840 ∧
  votes_for_c ≈ 227227 ∧
  votes_for_d ≈ 113613 := by
    sorry

end votes_calculation_l673_673087


namespace max_omega_l673_673494

theorem max_omega (ω : ℝ) (hω : ω > 0) (ϕ : ℝ) (hϕ : 0 ≤ ϕ ∧ ϕ ≤ π)
  (odd_fn : ∀ x, cos(ω * x + ϕ) = -cos(ω * x - ϕ))
  (monotonic : ∀ x1 x2, -π / 4 ≤ x1 → x1 ≤ x2 → x2 ≤ π / 3 → cos(ω * x1 + ϕ) ≥ cos(ω * x2 + ϕ))
  : ω ≤ 3/2 :=
sorry

end max_omega_l673_673494


namespace carl_olivia_cookie_difference_l673_673441

-- Defining the various conditions
def Carl_cookies : ℕ := 7
def Olivia_cookies : ℕ := 2

-- Stating the theorem we need to prove
theorem carl_olivia_cookie_difference : Carl_cookies - Olivia_cookies = 5 :=
by sorry

end carl_olivia_cookie_difference_l673_673441


namespace dani_pants_after_5_years_l673_673868

theorem dani_pants_after_5_years :
  ∀ (pairs_per_year : ℕ) (pants_per_pair : ℕ) (initial_pants : ℕ) (years : ℕ),
  pairs_per_year = 4 →
  pants_per_pair = 2 →
  initial_pants = 50 →
  years = 5 →
  initial_pants + years * (pairs_per_year * pants_per_pair) = 90 :=
by sorry

end dani_pants_after_5_years_l673_673868


namespace book_price_increase_l673_673357

theorem book_price_increase (P : ℝ) : 
  let new_price := P * 1.15 * 1.15 in
  new_price = P * (1 + 0.3225) := by
sorry

end book_price_increase_l673_673357


namespace largest_integer_dividing_consecutive_product_l673_673248

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l673_673248


namespace time_to_pass_l673_673757

-- Define the speeds of the trains in kmph
def speed_slower_train_kmph := 36
def speed_faster_train_kmph := 45

-- Define the length of the faster train in meters
def length_faster_train_m := 135.0108

-- Define the relative speed in m/s
def kmph_to_mps (kmph : ℕ) : ℝ := (kmph * 1000) / 3600
def relative_speed_mps : ℝ := kmph_to_mps (speed_slower_train_kmph + speed_faster_train_kmph)

-- Prove the time taken for the man in the slower train to pass the faster train
theorem time_to_pass : (length_faster_train_m / relative_speed_mps) = 6.00048 := by
  sorry

end time_to_pass_l673_673757


namespace carla_sheep_l673_673422

theorem carla_sheep (T : ℝ) (pen_sheep wilderness_sheep : ℝ) 
(h1: 0.90 * T = 81) (h2: pen_sheep = 81) 
(h3: wilderness_sheep = 0.10 * T) : wilderness_sheep = 9 :=
sorry

end carla_sheep_l673_673422


namespace integer_solutions_to_quadratic_inequality_l673_673715

theorem integer_solutions_to_quadratic_inequality :
  {x : ℤ | (x^2 + 6 * x + 8) * (x^2 - 4 * x + 3) < 0} = {-3, 2} :=
by
  sorry

end integer_solutions_to_quadratic_inequality_l673_673715


namespace zombies_less_than_50_four_days_ago_l673_673729

theorem zombies_less_than_50_four_days_ago
  (curr_zombies : ℕ)
  (days_ago : ℕ)
  (half_rate : ℕ)
  (initial_zombies : ℕ)
  (h_initial : curr_zombies = 480)
  (h_half : half_rate = 2)
  (h_days : days_ago = 4)
  : (curr_zombies / half_rate^days_ago) < 50 :=
by
  have h1 : curr_zombies / half_rate^1 = 480 / 2 := sorry
  have h2 : curr_zombies / half_rate^2 = 480 / 2^2 := sorry
  have h3 : curr_zombies / half_rate^3 = 480 / 2^3 := sorry
  have h4 : curr_zombies / half_rate^4 = 480 / 2^4 := sorry
  show 30 < 50 from sorry
  rw h_initial at *
  sorry

end zombies_less_than_50_four_days_ago_l673_673729


namespace shorter_leg_of_right_triangle_l673_673577

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l673_673577


namespace percentage_liked_B_l673_673809

-- Given conditions
def percent_liked_A (X : ℕ) : Prop := X ≥ 0 ∧ X ≤ 100 -- X percent of respondents liked product A
def percent_liked_both : ℕ := 23 -- 23 percent liked both products.
def percent_liked_neither : ℕ := 23 -- 23 percent liked neither product.
def min_surveyed_people : ℕ := 100 -- The minimum number of people surveyed by the company.

-- Required proof
theorem percentage_liked_B (X : ℕ) (h : percent_liked_A X):
  100 - X = Y :=
sorry

end percentage_liked_B_l673_673809


namespace right_triangle_exists_l673_673772

theorem right_triangle_exists :
  (3^2 + 4^2 = 5^2) ∧ ¬(2^2 + 3^2 = 4^2) ∧ ¬(4^2 + 6^2 = 7^2) ∧ ¬(5^2 + 11^2 = 12^2) :=
by
  sorry

end right_triangle_exists_l673_673772


namespace polynomial_bound_swap_l673_673016

variable (a b c : ℝ)

theorem polynomial_bound_swap (h : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ (x : ℝ), |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2 := by
  sorry

end polynomial_bound_swap_l673_673016


namespace cartesian_eq_of_curveC1_min_dist_curveC1_to_curveC_l673_673998

def curveC {ρ θ x y: ℝ} : Prop :=
  2 * ρ * sin θ + ρ * cos ρ = 10

def curveC1_parametric (α x y: ℝ) : Prop :=
  x = 3 * cos α ∧ y = 2 * sin α

def curveC1_cartesian (x y: ℝ) : Prop :=
  x^2 / 9 + y^2 / 4 = 1

def point_to_line_distance (x y a b c: ℝ) : ℝ :=
  abs (a * x + b * y + c) / sqrt (a^2 + b^2)

theorem cartesian_eq_of_curveC1 (α : ℝ) :
  ∃ (x y : ℝ), curveC1_parametric α x y → curveC1_cartesian x y :=
sorry

theorem min_dist_curveC1_to_curveC (α : ℝ) :
  ∃ (d : ℝ), (∀ (x y : ℝ), curveC1_parametric α x y → point_to_line_distance x y 1 2 -10 = d) ∧ d = sqrt 5 :=
sorry

end cartesian_eq_of_curveC1_min_dist_curveC1_to_curveC_l673_673998


namespace smallest_composite_no_prime_factors_below_15_correct_l673_673932

def smallest_composite_no_prime_factors_below_15 : Nat :=
  323
  
theorem smallest_composite_no_prime_factors_below_15_correct :
  (∀ n < 15, Prime n → ¬ (n ∣ smallest_composite_no_prime_factors_below_15)) ∧
  (∃ p q, Prime p ∧ Prime q ∧ p ≠ q ∧ smallest_composite_no_prime_factors_below_15 = p * q) :=
by
  -- Proof skipped
  sorry

end smallest_composite_no_prime_factors_below_15_correct_l673_673932


namespace arithmetic_geometric_seq_l673_673010

open Real

-- Step 1: Defining the arithmetic sequence and geometric sequence condition
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_seq (a : ℕ → ℤ) : Prop :=
  (a 0) * (a 4) = (a 1) * (a 1)

-- Step 2: Stating the main theorem
theorem arithmetic_geometric_seq (a : ℕ → ℤ) (d : ℤ) (h_a1 : a 0 = 1)
  (h_arith : arithmetic_seq a d) (h_geo : geometric_seq a) (h_d_nonzero : d ≠ 0) :
  (∀ n : ℕ, a n = 2 * n - 1) ∧ 
  (∀ n : ℕ, let b := (λ n, (1 / (a n * a (n + 1) : ℤ))) in 
    (∑ i in finrange n, b i) = n / (2 * n + 1 : ℤ)) :=
by
  sorry

end arithmetic_geometric_seq_l673_673010


namespace largest_divisor_of_consecutive_five_l673_673293

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l673_673293


namespace salt_solution_concentration_l673_673200

theorem salt_solution_concentration (m x : ℝ) (h1 : m > 30) (h2 : (m * m / 100) = ((m - 20) / 100) * (m + 2 * x)) :
  x = 10 * m / (m + 20) :=
sorry

end salt_solution_concentration_l673_673200


namespace smallest_composite_proof_l673_673946

noncomputable def smallest_composite_no_prime_factors_less_than_15 : ℕ :=
  289

theorem smallest_composite_proof :
  smallest_composite_no_prime_factors_less_than_15 = 289 :=
by
  sorry

end smallest_composite_proof_l673_673946


namespace ratio_XQ_QY_l673_673384

-- Definition of conditions
variable (decagon : Type)
variable [has_area decagon]
variable (triangles : fin 2 → decagon)
variable [base : Π (i : fin 2), has_base (triangles i)]
variable [base_length : ∀ i, base_length (triangles i) = 3]
variable (PQ : line)
variable [bisects_center : bisects_at_center PQ decagon]
variable (below_PQ : decagon)
variable [area_below_PQ : has_area below_PQ 6]
variable [unit_square_below : has_area (unit_square : decagon) 1]

-- Main theorem
theorem ratio_XQ_QY (XQ QY : ℝ) : (XQ + QY + 3 = 6) → (XQ = QY) → (XQ / QY = 1) :=
by 
  intros h₁ h₂
  rw [h₂]
  have : QY ≠ 0 := sorry -- Since QY is not zero
  exact div_self this

end ratio_XQ_QY_l673_673384


namespace parabola_equation_line_AB_max_area_l673_673474

section ParabolaProof

-- Define the parabola with parameter p > 0
variables {p : ℝ} (hp : p > 0)

-- Define point P on the parabola and its distance from the focus
def point_P_y4 (p : ℝ) : Prop :=
  ∃ x_P, (4)^2 = 2 * p * x_P ∧ dist (x_P, 4) (p / 2, 0) = 4

-- Define the equation of the parabola
theorem parabola_equation : point_P_y4 p → y^2 = 8 * x :=
sorry

-- Define points A and B on the parabola with the given conditions
variables {x₁ x₂ y₁ y₂ : ℝ}

def points_A_B_on_parabola (x₁ x₂ y₁ y₂ : ℝ) (p : ℝ) : Prop :=
  y₁^2 = 8 * x₁ ∧ y₂^2 = 8 * x₂ ∧ y₁ ≤ 0 ∧ y₂ ≤ 0

def angle_bisector_condition (x₁ x₂ y₁ y₂ : ℝ) : Prop :=
  ∃ P, dist P (x₁, y₁) = dist P (x₂, y₂) ∧ ∃ k, slope (segment P (x₁, y₁)) = k ∧ slope (segment P (x₂, y₂)) = -1/k

-- Define the equation of the line AB that maximizes the area of triangle PAB
theorem line_AB_max_area (p : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  points_A_B_on_parabola x₁ x₂ y₁ y₂ p ∧ angle_bisector_condition x₁ x₂ y₁ y₂ → y = -x :=
sorry

end ParabolaProof

end parabola_equation_line_AB_max_area_l673_673474


namespace largest_divisor_of_five_consecutive_integers_l673_673259

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l673_673259


namespace solve_exp_logarithmic_problem_correct_l673_673982

noncomputable def solve_exp_logarithmic_problem (a b m : ℝ) : Prop :=
(2^a = m) ∧ (3^b = m) ∧ (ab ≠ 0) ∧ (2 * a * b = a + b) → m = Real.sqrt 6

-- Proof will be provided by inference in Lean
theorem solve_exp_logarithmic_problem_correct (a b m : ℝ) :
  solve_exp_logarithmic_problem a b m :=
sorry

end solve_exp_logarithmic_problem_correct_l673_673982


namespace problem_part_1_problem_part_2_l673_673501

noncomputable def f (ω x : ℝ) : ℝ := cos (2 * ω * x - π / 3) - 2 * cos (ω * x)^2 + 2

theorem problem_part_1 (h : ∀ ω, f ω (\frac{π}{12}) - f ω (\frac{π}{3}) = π / 4) :
  (∃ k : ℤ, ∀ x, f 1 x = sin (2 * x - π / 6) + 1) ∧
  (∀ k : ℤ, (∃ x, f 1 x = 1 ∧ x = (k : ℝ) * π / 2 + π / 12) ∧
  (∀ x, f 1 x = (k : ℝ) * π / 2 + π / 3)) :=
sorry

theorem problem_part_2 :
  (∀ x, -π / 12 ≤ x ∧ x ≤ π / 2 → - sqrt 3 / 2 + 1 ≤ f 1 x ∧ f 1 x ≤ 2) := 
sorry

end problem_part_1_problem_part_2_l673_673501


namespace number_of_candidates_l673_673197

theorem number_of_candidates
  (P : ℕ) (A_c A_p A_f : ℕ)
  (h_p : P = 100)
  (h_ac : A_c = 35)
  (h_ap : A_p = 39)
  (h_af : A_f = 15) :
  ∃ T : ℕ, T = 120 := 
by
  sorry

end number_of_candidates_l673_673197


namespace value_of_a_l673_673481

theorem value_of_a (a : ℝ) (A B : ℝ × ℝ) (hA : A = (a - 2, 2 * a + 7)) (hB : B = (1, 5)) (h_parallel : (A.1 = B.1)) : a = 3 :=
by {
  sorry
}

end value_of_a_l673_673481


namespace remainder_when_1200th_number_divided_by_200_l673_673134

theorem remainder_when_1200th_number_divided_by_200 
  (S : ℕ → ℕ)
  (h1 : ∀ n : ℕ, S n = nth_element_with_7_ones_in_binary n) :
  S 1199 % 200 = 80 :=
sorry

end remainder_when_1200th_number_divided_by_200_l673_673134


namespace divide_into_four_equal_parts_l673_673758

-- Literature definition of a parallelogram and its properties
variables {V : Type*} [add_comm_group V] [vector_space ℝ V]

-- Define the concept of lines from a vertex within a parallelogram
def divide_parallelogram (P : parallelogram V) (A : V) : Prop :=
  let B := P.opposite_vertex A,
      C := P.diagonal A,
      M := midpoint (P.side BC),
      N := midpoint (P.side CD) in
  let triangles := [triangle A B M, triangle A C M, triangle A N D, triangle A N D] in
  ∀ t ∈ triangles, area t = (area P) / 4

-- Statement: Given a parallelogram, prove the division into four equal parts
theorem divide_into_four_equal_parts (P : parallelogram V) (A : V) : 
  divide_parallelogram P A :=
by
  sorry

end divide_into_four_equal_parts_l673_673758


namespace basketball_total_points_l673_673547

variable (Jon_points Jack_points Tom_points : ℕ)

def Jon_score := 3
def Jack_score := Jon_score + 5
def Tom_score := (Jon_score + Jack_score) - 4

theorem basketball_total_points :
  Jon_score + Jack_score + Tom_score = 18 := by
  sorry

end basketball_total_points_l673_673547


namespace min_value_g_l673_673498

section
variable (α : ℝ) (f : ℝ → ℝ := λ x, x^α) (g : ℝ → ℝ := λ x, (x - 3) * f x)

theorem min_value_g :
  (5^α = 1/5) ∧ (∀ x, f x = (1 : ℝ) / x) ∧ ∀ x ∈ set.Icc (1 / 3 : ℝ) (1 : ℝ), g x = 1 - (3 / x) →
  ∃ m, ∀ x ∈ set.Icc (1 / 3 : ℝ) (1 : ℝ), g x ≥ m ∧ m = -8 :=
by
  sorry
end

end min_value_g_l673_673498


namespace log_diff_lt_one_l673_673970

noncomputable def log_base (a b : ℝ) : ℝ := (Real.log b) / (Real.log a)

theorem log_diff_lt_one
  (b c x : ℝ)
  (h_eq_sym : ∀ (t : ℝ), (t - 2)^2 + b * (t - 2) + c = (t + 2)^2 + b * (t + 2) + c)
  (h_f_zero_pos : (0)^2 + b * (0) + c > 0)
  (m n : ℝ)
  (h_fm_0 : m^2 + b * m + c = 0)
  (h_fn_0 : n^2 + b * n + c = 0)
  (h_m_ne_n : m ≠ n)
  : log_base 4 m - log_base (1/4) n < 1 :=
  sorry

end log_diff_lt_one_l673_673970


namespace trajectory_equation_l673_673648

-- Definition of points A, B, and C
def A (y : ℝ) : ℝ × ℝ := (-2, y)
def B (y : ℝ) : ℝ × ℝ := (0, y / 2)
def C (x y : ℝ) : ℝ × ℝ := (x, y)

-- Definition of vectors AB and BC
def vecAB (y : ℝ) : ℝ × ℝ := (2, -(y / 2))
def vecBC (x y : ℝ) : ℝ × ℝ := (x, y / 2)

-- Given condition for orthogonality
def orthogonal (x y : ℝ) : Prop := (2 * x) - (y^2 / 4) = 0

-- The theorem stating the trajectory equation
theorem trajectory_equation (x y : ℝ) (h : orthogonal x y) : y^2 = 8 * x :=
by
  sorry

end trajectory_equation_l673_673648


namespace average_points_per_player_l673_673608

theorem average_points_per_player 
  (L R O : ℕ)
  (hL : L = 20) 
  (hR : R = L / 2) 
  (hO : O = 6 * R) 
  : (L + R + O) / 3 = 30 := by
  sorry

end average_points_per_player_l673_673608


namespace natural_number_sum_ways_l673_673157

def f : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+1) := 2 * f n

theorem natural_number_sum_ways (n : ℕ) : f n = 2^(n-1) :=
by
    induction n with
    | zero => have h : f 0 = 2^(-1 + 1) := rfl  -- base case, that should be valid as \(2^0\)
                 exact h
    | succ n ih =>
        have ind_hyp : f n = 2^(n-1) := ih
        have step: f (n + 1) = 2 * f n := rfl
        rw [step, ind_hyp]
        exact sorry   -- Here we should finish off the inductive step.

end natural_number_sum_ways_l673_673157


namespace sum_of_c_eq_neg_2_l673_673949

noncomputable theory

theorem sum_of_c_eq_neg_2 :
  let c_values := {c : ℤ | ∃ (k : ℤ), k % 2 = 1 ∧ 1 ≤ 49 + 4 * c ∧ 49 + 4 * c ≤ 149 ∧ 49 + 4 * c = k^2 ∧ c ≤ 25} in
  ∑ c in c_values, c = -2 :=
by
  sorry

end sum_of_c_eq_neg_2_l673_673949


namespace john_payment_difference_l673_673114

theorem john_payment_difference :
  ∀ (P₀ : ℝ) (r : ℝ) (n : ℕ),
  (P₀ = 12000) →
  (r = 0.08) →
  (n = 8) →
  let A₁ := P₀ * (1 + r/2)^(2 * 4),
      paid₁ := A₁ / 3,
      remaining₁ := A₁ - paid₁,
      new_A₁ := remaining₁ * (1 + r/2)^(2 * 4)
  in let total₁ := paid₁ + new_A₁ in
  let total₂ := P₀ * (1 + r)^n
  in abs (total₂ - total₁) = 1955 :=
by
  intros P₀ r n hP₀ hr hn A₁ paid₁ remaining₁ new_A₁ total₁ total₂,
  -- This is where the proof would go
  sorry

end john_payment_difference_l673_673114


namespace quadratic_inequality_solution_l673_673544

theorem quadratic_inequality_solution (a b c : ℝ) 
  (h1 : a < 0) 
  (h2 : a * 2^2 + b * 2 + c = 0) 
  (h3 : a * (-1)^2 + b * (-1) + c = 0) :
  ∀ x, ax^2 + bx + c ≥ 0 ↔ (-1 ≤ x ∧ x ≤ 2) :=
by 
  sorry

end quadratic_inequality_solution_l673_673544


namespace vertex_of_parabola_l673_673177

theorem vertex_of_parabola :
  ∃ h k : ℝ, (∀ x : ℝ, 3 * (x + 4)^2 - 9 = 3 * (x - h)^2 + k) ∧ (h, k) = (-4, -9) :=
by
  sorry

end vertex_of_parabola_l673_673177


namespace average_height_of_13_year_old_boys_is_1_56_l673_673202

noncomputable def north_sample_size := 300
noncomputable def south_sample_size := 200

noncomputable def north_average_height := 1.6
noncomputable def south_average_height := 1.5

noncomputable def total_sample_size := north_sample_size + south_sample_size

noncomputable def total_height := (north_average_height * north_sample_size) + (south_average_height * south_sample_size)

noncomputable def national_average_height := total_height / total_sample_size

theorem average_height_of_13_year_old_boys_is_1_56 :
  national_average_height = 1.56 :=
by
  sorry

end average_height_of_13_year_old_boys_is_1_56_l673_673202


namespace inequality_solution_exists_l673_673518

theorem inequality_solution_exists (a : ℝ) : 
  ∃ x : ℝ, x > 2 ∧ x > -1 ∧ x > a := 
by
  sorry

end inequality_solution_exists_l673_673518


namespace smallest_composite_proof_l673_673944

noncomputable def smallest_composite_no_prime_factors_less_than_15 : ℕ :=
  289

theorem smallest_composite_proof :
  smallest_composite_no_prime_factors_less_than_15 = 289 :=
by
  sorry

end smallest_composite_proof_l673_673944


namespace kim_shirts_left_l673_673127

-- Define the total number of shirts initially
def initial_shirts : ℕ := 4 * 12

-- Define the number of shirts given to the sister as 1/3 of the total
def shirts_given_to_sister : ℕ := initial_shirts / 3

-- Define the number of shirts left after giving some to the sister
def shirts_left : ℕ := initial_shirts - shirts_given_to_sister

-- The theorem we need to prove: Kim has 32 shirts left
theorem kim_shirts_left : shirts_left = 32 := by
  -- Proof is omitted
  sorry

end kim_shirts_left_l673_673127


namespace find_b_in_triangle_l673_673078

-- Define the corresponding angles and side lengths in the triangle
variables {A B : ℝ} {a b : ℝ}

-- Set up the problem with the known conditions
theorem find_b_in_triangle (hA : A = 45) (hB : B = 60) (ha : a = 10) :
  b = 5 * real.sqrt 6 :=
  sorry

end find_b_in_triangle_l673_673078


namespace paul_marks_l673_673151

def passing_marks (max_marks : ℕ) : ℕ := max_marks * 50 / 100

theorem paul_marks (max_marks : ℕ) (marks_diff : ℕ) (passing_marks_formula : max_marks = 120) (failed_by : marks_diff = 10) :
  let passingMarks := passing_marks max_marks in
  (passingMarks - marks_diff) = 50 :=
by
  sorry

end paul_marks_l673_673151


namespace product_of_five_consecutive_integers_divisible_by_240_l673_673282

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l673_673282


namespace cylinder_to_sphere_volume_ratio_l673_673178

theorem cylinder_to_sphere_volume_ratio:
  ∀ (a r : ℝ), (a^2 = π * r^2) → (a^3)/( (4/3) * π * r^3) = 3/2 :=
by
  intros a r h
  sorry

end cylinder_to_sphere_volume_ratio_l673_673178


namespace alcohol_water_ratio_l673_673825

theorem alcohol_water_ratio (A W A_new W_new : ℝ) (ha1 : A / W = 4 / 3) (ha2: A = 5) (ha3: W_new = W + 7) : A / W_new = 1 / 2.15 :=
by
  sorry

end alcohol_water_ratio_l673_673825


namespace three_digit_number_div_by_11_l673_673767

theorem three_digit_number_div_by_11 (x : ℕ) (h : x < 10) : 
  ∃ n : ℕ, n = 605 ∧ n < 1000 ∧ 
  (n % 10 = 5 ∧ (n / 100) % 10 = 6 ∧ n % 11 = 0) :=
begin
  use 605,
  split,
  { refl, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  norm_num,
end

end three_digit_number_div_by_11_l673_673767


namespace minimum_value_of_f_sum_inequality_l673_673513

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x + 1 / x - (Real.log x) ^ 2

-- Prove that f(x) has a minimum value of 2 at x = 1
theorem minimum_value_of_f : ∀ (x : ℝ), x > 0 → f x ≥ 2 ∧ f 1 = 2 :=
by sorry

-- Prove the inequality for the given sum
theorem sum_inequality (n : ℕ) (hn : n > 0) : ∑ k in Finset.range n + 1, 1 / Real.sqrt (2 ^ k * (2 ^ k + 1)) > Real.log (2 ^ (n + 1) / (2 ^ n + 1)) :=
by sorry

end minimum_value_of_f_sum_inequality_l673_673513


namespace johns_daily_calorie_intake_l673_673113

variable (breakfast lunch dinner shake : ℕ)
variable (num_shakes meals_per_day : ℕ)
variable (lunch_inc : ℕ)
variable (dinner_mult : ℕ)

-- Define the conditions from the problem
def john_calories_per_day 
  (breakfast := 500)
  (lunch := breakfast + lunch_inc)
  (dinner := lunch * dinner_mult)
  (shake := 300)
  (num_shakes := 3)
  (lunch_inc := breakfast / 4)
  (dinner_mult := 2)
  : ℕ :=
  breakfast + lunch + dinner + (shake * num_shakes)

theorem johns_daily_calorie_intake : john_calories_per_day = 3275 := by
  sorry

end johns_daily_calorie_intake_l673_673113


namespace pascal_triangle_sum_difference_l673_673428

theorem pascal_triangle_sum_difference :
  (\sum i in Finset.range 1005, (Nat.choose 1004 i) / (Nat.choose 1005 i)) -
  (\sum i in Finset.range 1004, (Nat.choose 1003 i) / (Nat.choose 1004 i)) = 1 / 2 :=
by
  sorry

end pascal_triangle_sum_difference_l673_673428


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673221

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673221


namespace min_even_integers_l673_673742

theorem min_even_integers
  (x y z a b c m n o : ℤ)
  (h1 : x + y + z = 30)
  (h2 : x + y + z + a + b + c = 55)
  (h3 : x + y + z + a + b + c + m + n + o = 88) :
  ∃ (evens : ℕ), evens = 1 ∧ ∃ (nums : list ℤ), nums = [x, y, z, a, b, c, m, n, o] ∧ 
    evens = (nums.filter (λ n, n % 2 = 0)).length := 
sorry

end min_even_integers_l673_673742


namespace factor_of_polynomial_l673_673865

theorem factor_of_polynomial :
  (x : ℝ) : x^4 - 6 * x^2 + 9 = (x^2 - 3)^2 := 
by
  sorry

end factor_of_polynomial_l673_673865


namespace find_pairs_eq_l673_673446

theorem find_pairs_eq : 
  { (m, n) : ℕ × ℕ | 0 < m ∧ 0 < n ∧ m ^ 2 + 2 * n ^ 2 = 3 * (m + 2 * n) } = {(3, 3), (4, 2)} :=
by sorry

end find_pairs_eq_l673_673446


namespace ratio_of_arithmetic_sequences_l673_673050

-- Definitions for the conditions
variables {a_n b_n : ℕ → ℝ}
variables {S_n T_n : ℕ → ℝ}
variables (d_a d_b : ℝ)

-- Arithmetic sequences conditions
def is_arithmetic_sequence (u_n : ℕ → ℝ) (t : ℝ) (d : ℝ) : Prop :=
  ∀ (n : ℕ), u_n n = t + n * d

-- Sum of first n terms conditions
def sum_of_first_n_terms (u_n : ℕ → ℝ) (Sn : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), Sn n = n * (u_n 1 + u_n (n-1)) / 2

-- Main theorem statement
theorem ratio_of_arithmetic_sequences (h1 : is_arithmetic_sequence a_n (a_n 0) d_a)
                                     (h2 : is_arithmetic_sequence b_n (b_n 0) d_b)
                                     (h3 : sum_of_first_n_terms a_n S_n)
                                     (h4 : sum_of_first_n_terms b_n T_n)
                                     (h5 : ∀ n, (S_n n) / (T_n n) = (2 * n) / (3 * n + 1)) :
                                     ∀ n, (a_n n) / (b_n n) = (2 * n - 1) / (3 * n - 1) := sorry

end ratio_of_arithmetic_sequences_l673_673050


namespace kim_shirts_left_l673_673124

theorem kim_shirts_left (initial_dozens : ℕ) (fraction_given : ℚ) (num_pairs : ℕ)
  (h1 : initial_dozens = 4) 
  (h2 : fraction_given = 1 / 3)
  (h3 : num_pairs = initial_dozens * 12)
  (h4 : num_pairs * fraction_given  = (16 : ℕ)):
  48 - ((num_pairs * fraction_given).toNat) = 32 :=
by 
  sorry

end kim_shirts_left_l673_673124


namespace hexagon_perimeter_eq_4_sqrt_3_over_3_l673_673170

noncomputable def hexagon_perimeter (s : ℝ) : ℝ :=
  6 * s

theorem hexagon_perimeter_eq_4_sqrt_3_over_3 :
  ∀ (s : ℝ), (∃ s, (3 * Real.sqrt 3 / 2) * s^2 = s) → hexagon_perimeter s = 4 * Real.sqrt 3 / 3 :=
by
  simp
  sorry

end hexagon_perimeter_eq_4_sqrt_3_over_3_l673_673170


namespace _l673_673156

-- Definition of parabolas intersection and cyclic quadrilateral for specific case
noncomputable def specific_intersection_points_cyclic (x y : ℝ) (h₁ : y = x^2)
    (h₂ : 3 * (x - 2) + (y - 4)^2 = 0) : Prop :=
  let points := {p | ∃ x : ℝ, ∃ y : ℝ, y = x^2 ∧ 3 * (x - 2) + (y - 4)^2 = 0} in
  points.card = 4 ∧ is_cyclic_quadrilateral points

-- Definition for general perpendicular parabolas case
noncomputable def general_perpendicular_parabolas_cyclic (a1 a2 b1 b2: ℝ)
    (x y : ℝ) (h₁ : (x - a1)^2 - 2 * b1 * (y - b1 / 2) = 0)
    (h₂ : (y - b2)^2 - 2 * a2 * (x - a2 / 2) = 0) : Prop :=
  let points := {p | ∃ x : ℝ, ∃ y : ℝ, 
    (x - a1)^2 - 2 * b1 * (y - b1 / 2) = 0 ∧ (y - b2)^2 - 2 * a2 * (x - a2 / 2) = 0} in
  points.card = 4 ∧ is_cyclic_quadrilateral points

-- The main theorem combining both specific and general cases
noncomputable theorem cyclic_quadrilateral_intersection:
    (∃ x y, y = x^2 ∧ 3 * (x - 2) + (y - 4)^2 = 0 → specific_intersection_points_cyclic x y sorry) ∧
    (∀ a1 a2 b1 b2 x y, (x - a1)^2 - 2 * b1 * (y - b1 / 2) = 0 ∧ (y - b2)^2 - 2 * a2 * (x - a2 / 2) = 0
    → general_perpendicular_parabolas_cyclic a1 a2 b1 b2 x y sorry) :=
  sorry

end _l673_673156


namespace trig_inequality_l673_673618

theorem trig_inequality (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  (1 / (Real.cos α)^2 + 1 / ((Real.sin α)^2 * (Real.cos β)^2 * (Real.sin β)^2) ≥ 9) := by
  sorry

end trig_inequality_l673_673618


namespace exist_golden_matrix_13_l673_673952

def is_golden (A : Matrix (Fin 2004) (Fin 2004) (Fin n)) : Prop :=
  ∀ (i j : Fin 2004), (X i A) ≠ (X j A) ∧ (Y i A) ≠ (Y j A)

def X (i : Fin 2004) (A : Matrix (Fin 2004) (Fin 2004) (Fin n)) : Finset (Fin n) :=
  Finset.univ.image (fun j => A i j)

def Y (j : Fin 2004) (A : Matrix (Fin 2004) (Fin 2004) (Fin n)) : Finset (Fin n) :=
  Finset.univ.image (fun i => A i j)

theorem exist_golden_matrix_13 :
  ∃ (A : Matrix (Fin 2004) (Fin 2004) (Fin 13)), is_golden A :=
sorry

end exist_golden_matrix_13_l673_673952


namespace subset_exists_l673_673361

open Set

def P (n : ℕ) : Set ℕ := {m | ∃ (k : ℕ), k ≤ n ∧ m = 2^(n-k) * 3^k }

def S (X : Set ℕ) : ℕ := X.sum id

theorem subset_exists (n : ℕ) (y : ℝ)
  (hn : 0 ≤ y ∧ y ≤ 3^n.succ - 2^n.succ) :
  ∃ Y ⊆ P n, 0 ≤ y - S Y ∧ y - S Y < 2 ^ n := by
  sorry

end subset_exists_l673_673361


namespace largest_divisor_of_consecutive_five_l673_673289

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l673_673289


namespace shirts_left_l673_673121

-- Define the given conditions
def initial_shirts : ℕ := 4 * 12
def fraction_given : ℚ := 1 / 3

-- Define the proof goal
theorem shirts_left (initial_shirts : ℕ) (fraction_given : ℚ) : ℕ :=
let shirts_given := initial_shirts * fraction_given in
initial_shirts - (shirts_given : ℕ) = 32 :=
begin
  -- placeholder for the proof
  sorry
end

end shirts_left_l673_673121


namespace smallest_composite_no_prime_factors_lt_15_l673_673906

theorem smallest_composite_no_prime_factors_lt_15 (n : ℕ) :
  ∀ n, (∀ p : ℕ, p.prime → p ∣ n → 15 ≤ p) → n = 289 → 
       is_composite n ∧ (∀ m : ℕ, (∀ q : ℕ, q.prime → q ∣ m → 15 ≤ q) → m ≥ 289) :=
by
  intros n hv hn
  -- Proof would go here
  sorry

end smallest_composite_no_prime_factors_lt_15_l673_673906


namespace range_of_a_l673_673512

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2 * x + a

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), (x ≥ 0 ∧ x ≤ 3) → let y := f a x in
  -- Any three distinct values of the function can always serve as the lengths of the sides of a triangle.
  ( ∀ (x1 x2 x3 : ℝ),
    x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 0 ≤ x1 ∧ x1 ≤ 3 ∧ 0 ≤ x2 ∧ x2 ≤ 3 ∧ 0 ≤ x3 ∧ x3 ≤ 3 →
    let y1 := f a x1
    let y2 := f a x2
    let y3 := f a x3
    y1 + y2 > y3 ∧ y2 + y3 > y1 ∧ y1 + y3 > y2 )
  ) → a ≥ 5 :=
begin
  sorry
end

end range_of_a_l673_673512


namespace probability_abs_diff_gt_two_thirds_l673_673163

/-- Define the sample space for the die rolls -/
def die : Type := {n // n ≥ 1 ∧ n ≤ 6}

/-- Define the probability measure for the die -/
noncomputable def die_prob : Measure (Set die) :=


/-- Define X and Y as random variables based on the conditions -/
noncomputable def X (d : die) : ℝ :=
  match d with
  | ⟨1, _⟩ => 0
  | ⟨2, _⟩ => 0
  | ⟨3, _⟩ => 1
  | ⟨4, _⟩ => 1
  | ⟨5, _⟩ => uniform (0, 1)
  | ⟨6, _⟩ => uniform (0, 1)

noncomputable def Y (d : die) : ℝ :=
  match d with
  | ⟨1, _⟩ => 0
  | ⟨2, _⟩ => 0
  | ⟨3, _⟩ => 1
  | ⟨4, _⟩ => 1
  | ⟨5, _⟩ => uniform (0, 1)
  | ⟨6, _⟩ => uniform (0, 1)

/-- Formulate the probability of the condition |X - Y| > 2/3 -/
noncomputable def prob_diff_gt_two_thirds : ℚ :=
  sorry

/-- Theorem stating the desired probability -/
theorem probability_abs_diff_gt_two_thirds :
  prob_diff_gt_two_thirds = (14 / 27 : ℚ) :=
by sorry

end probability_abs_diff_gt_two_thirds_l673_673163


namespace functional_equation_solution_l673_673885

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, y^2 * f x + x^2 * f y + x * y = x * y * f (x + y) + x^2 + y^2) →
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x + 1 :=
by
  sorry

end functional_equation_solution_l673_673885


namespace number_of_real_solution_pairs_number_of_ab_pairs_is_12_l673_673453

-- defining the set of possible values for a and b
def ab_domain := {-1, 0, 1, 2}

-- equation having real solutions means discriminant is non-negative
def real_solutions (a b : ℤ) : Prop :=
  4 - 4 * a * b ≥ 0

-- proving the number of ordered pairs (a, b) that satisfy the conditions
theorem number_of_real_solution_pairs : 
  (finset.univ.filter (λ (p : ℤ × ℤ), p.1 ∈ ab_domain ∧ p.2 ∈ ab_domain ∧ real_solutions p.1 p.2)).card = 12 :=
by 
  sorry

-- Defining ab_domain and real_solutions as per the given conditions
def ab_pairs := 
  finset.univ.filter (λ (p : ℤ × ℤ), p.1 ∈ ab_domain ∧ p.2 ∈ ab_domain ∧ real_solutions p.1 p.2)

-- The total count of such pairs should equal 12
theorem number_of_ab_pairs_is_12 : ab_pairs.card = 12 := 
by 
  sorry

end number_of_real_solution_pairs_number_of_ab_pairs_is_12_l673_673453


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673310

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673310


namespace distance_from_wall_to_mirror_edge_l673_673824

theorem distance_from_wall_to_mirror_edge (wall_width mirror_width : ℝ) 
    (h_wall : wall_width = 26) (h_mirror : mirror_width = 4) :
    let x := (wall_width - mirror_width) / 2 in
    x = 11 :=
by
  have h_total_eq := h_wall
  have h_mirr_eq := h_mirror
  let x := (wall_width - mirror_width) / 2
  calc
    x = (26 - 4) / 2 : by rw [h_total_eq, h_mirr_eq]
    ... = 22 / 2     : by norm_num
    ... = 11         : by norm_num

end distance_from_wall_to_mirror_edge_l673_673824


namespace area_of_ABCD_l673_673199

theorem area_of_ABCD :
  (width_s : ℝ) (length_s : ℝ) (width_l : ℝ) (length_l : ℝ) 
  (h1 : width_s = 8)
  (h2 : length_s = 16)
  (h3 : width_l = width_s)
  (h4 : length_l = 3 * length_s) 
  (A : ℝ := length_l * width_l) : 
  A = 384 := 
by
  sorry

end area_of_ABCD_l673_673199


namespace probability_gcd_one_is_49_over_56_l673_673737

def is_gcd_one (a b c : ℕ) : Prop := Nat.gcd a (Nat.gcd b c) = 1

def count_choices_with_gcd_one : ℕ :=
  ((Finset.powersetLen 3 (Finset.range 9)).filter (λ s, match s.toList with
    | [a, b, c] => is_gcd_one a b c
    | _ => false
  end)).card

def total_choices : ℕ := (Finset.powersetLen 3 (Finset.range 9)).card

theorem probability_gcd_one_is_49_over_56 :
  (count_choices_with_gcd_one : ℚ) / total_choices = 49 / 56 := by
  sorry

end probability_gcd_one_is_49_over_56_l673_673737


namespace train_crossing_time_l673_673777

theorem train_crossing_time :
  ∀ (length_train1 length_train2 : ℕ) 
    (speed_train1_kmph speed_train2_kmph : ℝ), 
  length_train1 = 420 →
  speed_train1_kmph = 72 →
  length_train2 = 640 →
  speed_train2_kmph = 36 →
  (length_train1 + length_train2) / ((speed_train1_kmph - speed_train2_kmph) * (1000 / 3600)) = 106 :=
by
  intros
  sorry

end train_crossing_time_l673_673777


namespace product_of_five_consecutive_divisible_by_30_l673_673266

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l673_673266


namespace carlotta_tantrum_time_l673_673956

theorem carlotta_tantrum_time :
  (∀ (T P S : ℕ), 
   S = 6 ∧ T + P + S = 54 ∧ P = 3 * S → T = 5 * S) :=
by
  intro T P S
  rintro ⟨hS, hTotal, hPractice⟩
  sorry

end carlotta_tantrum_time_l673_673956


namespace largest_divisor_of_consecutive_five_l673_673288

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l673_673288


namespace intersection_of_M_and_N_l673_673635

open Set

def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x^2 - 25 < 0}
def I : Set ℝ := {x | 2 ≤ x ∧ x < 5}

theorem intersection_of_M_and_N : M ∩ N = I := by
  sorry

end intersection_of_M_and_N_l673_673635


namespace shorter_leg_of_right_triangle_l673_673554

theorem shorter_leg_of_right_triangle (a b : ℕ) (h : a^2 + b^2 = 65^2) : min a b = 16 :=
sorry

end shorter_leg_of_right_triangle_l673_673554


namespace members_cast_votes_election_l673_673412

theorem members_cast_votes_election :
  ∃ (V : ℕ), let total_membership := 1600 in
             let winning_percentage := 0.60 in
             let total_voted_percentage := 0.196875 in
             let votes_won := total_voted_percentage * total_membership in
             V = votes_won / winning_percentage ∧ V = 525 :=
begin
  sorry
end

end members_cast_votes_election_l673_673412


namespace smallest_composite_no_prime_factors_less_than_15_l673_673915

-- Definitions used in the conditions
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

-- Prime numbers less than 15
def primes_less_than_15 (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13

-- Define the main proof statement
theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n : ℕ, is_composite n ∧ (∀ p : ℕ, p ∣ n → is_prime p → primes_less_than_15 p → false) ∧ n = 289 :=
by
  -- leave the proof as a placeholder
  sorry

end smallest_composite_no_prime_factors_less_than_15_l673_673915


namespace profit_amount_l673_673383

-- Conditions: Selling Price and Profit Percentage
def SP : ℝ := 850
def P_percent : ℝ := 37.096774193548384

-- Theorem: The profit amount is $230
theorem profit_amount : (SP / (1 + P_percent / 100)) * P_percent / 100 = 230 := by
  -- sorry will be replaced with the proof
  sorry

end profit_amount_l673_673383


namespace dinner_serving_problem_l673_673752

theorem dinner_serving_problem : 
  let orders := ["B", "B", "B", "B", "C", "C", "C", "C", "F", "F", "F", "F"].to_finset in
  let possible_serving_count := choose 12 2 * 160 in
  ∃ (serving : set (fin 12)), 
    (serving : cardinal) = 2 ∧
    (orders = serving) →
    possible_serving_count = 211200
:= 
begin
  sorry
end

end dinner_serving_problem_l673_673752


namespace direction_vector_correct_l673_673186

def matrix := ![![6 / 25, 32 / 25], ![32 / 25, -6 / 25]]

def direction_vector := ![31, 19]

def gcd (a b : ℤ) : ℤ :=
if b = 0 then a.natAbs else gcd b (a % b)

theorem direction_vector_correct
: matrix.mul direction_vector = direction_vector
∧ gcd (direction_vector 0) (direction_vector 1) = 1 :=
by
  sorry

end direction_vector_correct_l673_673186


namespace product_of_five_consecutive_integers_divisible_by_240_l673_673277

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l673_673277


namespace product_of_five_consecutive_divisible_by_30_l673_673276

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l673_673276


namespace koi_fish_added_per_day_l673_673160

theorem koi_fish_added_per_day 
  (initial_total_fish : ℕ)
  (goldfish_per_day : ℕ)
  (days : ℕ)
  (total_goldfish_end : ℕ)
  (total_koi_fish_end : ℕ)
  (initial_koi_fish : ℕ := initial_total_fish - 95) 
  : goldfish_per_day = 5 ∧ days = 21 ∧ initial_total_fish = 280 ∧ total_goldfish_end = 200 ∧ total_koi_fish_end = 227 →
    let koi_fish_added := total_koi_fish_end - initial_koi_fish
    in koi_fish_added / days = 2 :=
begin
  intros h,
  sorry
end

end koi_fish_added_per_day_l673_673160


namespace at_least_30_cents_prob_l673_673683

def coin := {penny, nickel, dime, quarter, half_dollar}
def value (c : coin) : ℕ := 
  match c with
  | penny => 1
  | nickel => 5
  | dime => 10
  | quarter => 25
  | half_dollar => 50

def coin_positions : List (coin × Bool) := 
  [(penny, true), (nickel, true), (dime, true), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, true), (dime, true), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, true), (dime, true), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, true), (dime, false), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, true), (dime, false), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, true), (dime, false), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, true), (dime, false), (quarter, false), (half_dollar, false),
   (penny, true), (nickel, false), (dime, true), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, false), (dime, true), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, false), (dime, true), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, false), (dime, true), (quarter, false), (half_dollar, false),
   (penny, true), (nickel, false), (dime, false), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, false), (dime, false), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, false), (dime, false), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, false), (dime, false), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, true), (dime, true), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, true), (dime, true), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, true), (dime, true), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, true), (dime, true), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, true), (dime, false), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, true), (dime, false), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, true), (dime, false), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, true), (dime, false), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, false), (dime, true), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, false), (dime, true), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, false), (dime, true), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, false), (dime, true), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, false), (dime, false), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, false), (dime, false), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, false), (dime, false), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, false), (dime, false), (quarter, false), (half_dollar, false)]

def count_successful_outcomes : ℕ :=
  List.length (List.filter (λ positions, List.foldl (λ acc (c, h) => if h then acc + value c else acc) 0 positions >= 30) coin_positions)

def total_outcomes : ℕ := 32

def probability_of_success : ℚ :=
  ⟨count_successful_outcomes, total_outcomes⟩

theorem at_least_30_cents_prob : probability_of_success = 3 / 4 :=
by sorry

end at_least_30_cents_prob_l673_673683


namespace smallest_composite_no_prime_factors_less_than_15_l673_673923

theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n, (n = 289) ∧ (n > 1) ∧ (¬ Nat.Prime n) ∧ (∀ p : ℕ, Nat.Prime p → p ∣ n → 15 ≤ p) :=
by
  use 289
  split
  case left => rfl
  case right =>
    split
    case left => exact Nat.lt_succ_self 288
    case right =>
      split
      case left =>
        have composite : ¬ Nat.Prime 289 := by
          intro h
          have h_div : 17 ∣ 289 := by norm_num
          exact h.not_divs_self (dec_trivial : 17 * 17 = 289)
        exact composite
      case right =>
        intros p h_prime h_div
        have : p ∣ 17 := by
          have factorization : 289 = 17 * 17 := by norm_num
          have dvd_product : p ∣ 289 := by { use 17, exact factorization.symm }
          exact Nat.Prime.dvd_mul h_prime dvd_product
        have prime_eq_17 : p = 17 := by
          exact Nat.Prime.eq_of_dvd_of_ne h_prime (by norm_num) this
        linarith

end smallest_composite_no_prime_factors_less_than_15_l673_673923


namespace largest_divisor_of_product_of_5_consecutive_integers_l673_673243

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l673_673243


namespace product_of_five_consecutive_integers_divisible_by_240_l673_673287

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l673_673287


namespace exists_row_or_column_with_at_least_10_colors_l673_673439

theorem exists_row_or_column_with_at_least_10_colors
  (board : Fin 100 → Fin 100 → Fin 100)
  (color_count : Fin 100 → Fin 100 → Nat)
  (h : ∀ (c : Fin 100), (Finset.univ.filter (λ coords, board coords.1 coords.2 = c)).card = 100) :
  ∃ r : Fin 100, (Finset.univ.image (λ c, board r c)).card ≥ 10 ∨ 
  ∃ c : Fin 100, (Finset.univ.image (λ r, board r c)).card ≥ 10 :=
sorry

end exists_row_or_column_with_at_least_10_colors_l673_673439


namespace ratio_of_still_lifes_to_portraits_l673_673415

noncomputable def total_paintings : ℕ := 80
noncomputable def portraits : ℕ := 16
noncomputable def still_lifes : ℕ := total_paintings - portraits
axiom still_lifes_is_multiple_of_portraits : ∃ k : ℕ, still_lifes = k * portraits

theorem ratio_of_still_lifes_to_portraits : still_lifes / portraits = 4 := by
  -- proof would go here
  sorry

end ratio_of_still_lifes_to_portraits_l673_673415


namespace largest_divisor_of_five_consecutive_integers_l673_673263

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l673_673263


namespace inequality_nonneg_reals_l673_673155

theorem inequality_nonneg_reals (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  (1 / 2 * (a + b)^2 + 1 / 4 * (a + b)) ≥ a * real.sqrt b + b * real.sqrt a :=
by sorry

end inequality_nonneg_reals_l673_673155


namespace percentage_liked_B_l673_673808

-- Given conditions
def percent_liked_A (X : ℕ) : Prop := X ≥ 0 ∧ X ≤ 100 -- X percent of respondents liked product A
def percent_liked_both : ℕ := 23 -- 23 percent liked both products.
def percent_liked_neither : ℕ := 23 -- 23 percent liked neither product.
def min_surveyed_people : ℕ := 100 -- The minimum number of people surveyed by the company.

-- Required proof
theorem percentage_liked_B (X : ℕ) (h : percent_liked_A X):
  100 - X = Y :=
sorry

end percentage_liked_B_l673_673808


namespace find_eighth_term_of_sequence_l673_673458

noncomputable def a : ℕ → ℕ
| 0     := 1
| 1     := 4
| 2     := 10
| 3     := 20
| 4     := 35
| 5     := 56
| 6     := 84
| (n+7) := a (n+6) + (a (n+6) - a (n+5)) + ((a (n+5) - a (n+4)) - (a (n+4) - a (n+3))) + (((a (n+4) - a (n+3)) - (a (n+3) - a (n+2))) - ((a (n+3) - a (n+2)) - (a (n+2) - a (n+1))))

theorem find_eighth_term_of_sequence :
  a 7 = 129 :=
sorry

end find_eighth_term_of_sequence_l673_673458


namespace find_k_set_l673_673821

open Set

noncomputable def line (k : ℝ) : ℝ → ℝ := λ x => k * (x + 2) + 1

def parabola (x : ℝ) : ℝ := 4 * x

theorem find_k_set : {k | ∃! p : ℝ × ℝ, p.2 = k * (p.1 + 2) + 1 ∧ p.2 = sqrt (4 * p.1)} = {0, -1, 1/2} := 
by
  sorry

end find_k_set_l673_673821


namespace correct_operation_l673_673343

variables (a b : ℝ)

theorem correct_operation : (3 * a + b) * (3 * a - b) = 9 * a^2 - b^2 :=
by sorry

end correct_operation_l673_673343


namespace range_of_a_l673_673516

noncomputable def real_a_property (a : ℝ) : Prop :=
  let M := {x : ℝ | x > -a}
  let g := λ x : ℝ, Real.log (x - 1)
  let N := {x : ℝ | x > 1}
  M ⊆ N → a < -1

theorem range_of_a (a : ℝ) :
  real_a_property a :=
sorry

end range_of_a_l673_673516


namespace max_number_of_lines_with_properties_l673_673148

open_locale classical

-- Defining the properties of distinct lines
structure LineOnPlane (α : Type*) :=
  (distinct_lines : set α)
  (intersect_all_pairs : ∀ (l₁ l₂ : α), l₁ ∈ distinct_lines → l₂ ∈ distinct_lines → l₁ ≠ l₂ → intersects l₁ l₂)
  (angle_60_among_15 : ∀ (s : finset α), s.card = 15 → ∃ l₁ l₂ ∈ s, angle_between l₁ l₂ = 60)

-- Proving the upper bound of the number of lines satisfying the conditions
theorem max_number_of_lines_with_properties : 
  ∀ (α : Type*) [fintype α], 
  (∃ (line : LineOnPlane α), ∀ (l : LineOnPlane α), l.distinct_lines.to_finset.card ≤ 42) :=
begin
  sorry
end

end max_number_of_lines_with_properties_l673_673148


namespace no_power_of_2_with_reorder_property_l673_673599

theorem no_power_of_2_with_reorder_property :
  ¬ ∃ (a b : ℕ), a ≠ b ∧ 0 < a ∧ 0 < b ∧
  (∀ d ∈ (Nat.digits 10 (2^a)), d ≠ 0) ∧
  (Permutation (Nat.digits 10 (2^a)) (Nat.digits 10 (2^b))) :=
sorry

end no_power_of_2_with_reorder_property_l673_673599


namespace ball_hits_ground_time_l673_673179

theorem ball_hits_ground_time :
  ∀ t : ℝ, y = -20 * t^2 + 30 * t + 60 → y = 0 → t = (3 + Real.sqrt 57) / 4 := by
  sorry

end ball_hits_ground_time_l673_673179


namespace problem_l673_673477

def seq (a : ℕ → ℝ) := a 0 = 1 / 2 ∧ ∀ n > 0, a n = a (n - 1) + (1 / n^2) * (a (n - 1))^2

theorem problem (a : ℕ → ℝ) (n : ℕ) (h_seq : seq a) (h_n_pos : n > 0) :
  (1 / a (n - 1) - 1 / a n < 1 / n^2) ∧
  (∀ n > 0, a n < n) ∧
  (∀ n > 0, 1 / a n < 5 / 6 + 1 / (n + 1)) :=
by
  sorry

end problem_l673_673477


namespace shorter_leg_of_right_triangle_l673_673581

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l673_673581


namespace union_of_A_and_B_l673_673465

def A : Set ℤ := {-1, 0, 2}
def B : Set ℤ := {-1, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end union_of_A_and_B_l673_673465


namespace complementary_event_is_at_most_one_wins_l673_673802

-- Define the Event A
def event_A : set (bool × bool) := { (tt, tt) }

-- Define the Complementary Event of A
def complementary_event (Α : set (bool × bool)) : set (bool × bool) :=
  { ω | ω ∉ Α }

-- Definition of "at most one of A and B wins a prize"
def at_most_one_wins : set (bool × bool) :=
  { (tt, ff), (ff, tt), (ff, ff) }

theorem complementary_event_is_at_most_one_wins :
  complementary_event event_A = at_most_one_wins :=
by
  sorry

end complementary_event_is_at_most_one_wins_l673_673802


namespace acute_angle_theta_l673_673493

theorem acute_angle_theta :
  ∃ θ : ℝ, (0 < θ ∧ θ < π / 2) ∧
    (∃ P : ℝ × ℝ, P.1 = sin 10 * π / 180 ∧ P.2 = 1 + sin (80 * π / 180) ∧ θ = arcsin (P.1 + π / 2)) ∧ 
    θ = 85 * π / 180 :=
by
  sorry

end acute_angle_theta_l673_673493


namespace rhombus_side_length_l673_673830

theorem rhombus_side_length (a b s K : ℝ)
  (h1 : b = 3 * a)
  (h2 : K = (1 / 2) * a * b)
  (h3 : s ^ 2 = (a / 2) ^ 2 + (3 * a / 2) ^ 2) :
  s = Real.sqrt (5 * K / 3) :=
by
  sorry

end rhombus_side_length_l673_673830


namespace product_four_consecutive_l673_673665

theorem product_four_consecutive (X : ℤ) : 
  let P := X * (X + 1) * (X + 2) * (X + 3)
  in P = (X^2 + 3*X + 1)^2 - 1 := 
by 
  sorry

end product_four_consecutive_l673_673665


namespace focus_of_parabola_l673_673176

-- Define the given parabola equation
def parabola_eq (x y : ℝ) : Prop := y = (1/4) * x^2

-- Define the conditions about the focus and the parabola direction
def focus_on_y_axis : Prop := True -- Given condition
def opens_upwards : Prop := True -- Given condition

theorem focus_of_parabola (x y : ℝ) 
  (h1 : parabola_eq x y) 
  (h2 : focus_on_y_axis) 
  (h3 : opens_upwards) : 
  (x = 0 ∧ y = 1) :=
by
  sorry

end focus_of_parabola_l673_673176


namespace ratio_wealth_per_citizen_l673_673430

variables (a b W P : ℝ)
-- Conditions
def country_X_population := 0.01 * a * P
def country_X_wealth := 0.01 * b * W
def country_Y_population := 0.01 * b * P
def country_Y_wealth := 0.01 * a * W

-- Wealth per citizen in country X
def wealth_per_citizen_X := country_X_wealth a b W / country_X_population a P

-- Wealth per citizen in country Y
def wealth_per_citizen_Y := country_Y_wealth a W / country_Y_population b P

-- Theorem stating the ratio of wealth per citizen
theorem ratio_wealth_per_citizen (a b W P : ℝ) : 
  (wealth_per_citizen_X a b W P) / (wealth_per_citizen_Y a b W P) = (b^2) / (a^2) :=
by {
  sorry
}

end ratio_wealth_per_citizen_l673_673430


namespace probability_of_at_least_30_cents_l673_673679

def coin := fin 5

def value (c : coin) : ℤ :=
match c with
| 0 => 1   -- penny
| 1 => 5   -- nickel
| 2 => 10  -- dime
| 3 => 25  -- quarter
| 4 => 50  -- half-dollar
| _ => 0

def coin_flip : coin -> bool := λ c => true -- Placeholder for whether heads or tails

def total_value (flips : coin -> bool) : ℤ :=
  finset.univ.sum (λ c, if flips c then value c else 0)

noncomputable def probability_at_least_30_cents : ℚ :=
  let coin_flips := (finset.pi finset.univ (λ _, finset.univ : finset (coin -> bool))).val in
  let successful_flips := coin_flips.filter (λ flips, total_value flips >= 30) in
  successful_flips.card / coin_flips.card

theorem probability_of_at_least_30_cents :
  probability_at_least_30_cents = 9 / 16 :=
by
  sorry

end probability_of_at_least_30_cents_l673_673679


namespace largest_six_digit_number_sum_of_digits_l673_673133

theorem largest_six_digit_number_sum_of_digits (M : ℕ) (h1 : 100000 ≤ M ∧ M < 1000000) (h2 : (∏ d in (M.digits 10).toFinset, d) = 60) :
  (M.digits 10).sum = 15 :=
sorry

end largest_six_digit_number_sum_of_digits_l673_673133


namespace cost_of_article_l673_673069

variable {C G : ℝ}

theorem cost_of_article (h : 350 = C * (1 + (G + 5) / 100)) (h' : 340 = C * (1 + G / 100)) : C = 200 := by
  sorry

end cost_of_article_l673_673069


namespace AE_eq_EC_l673_673488

variables {A B C D E : Point}
variables (ABC_isosceles_right : isosceles_right_triangle A B C)
variables (BC_hypotenuse : hypotenuse BC (triangle A B C))
variables (D_on_BC : line_segment B C contains D)
variables (DC_eq_one_third_BC : line_segment_length D C = (1/3) * line_segment_length B C)
variables (BE_perp_AD : perpendicular BE AD)
variables (E_on_AC : line_segment A C contains E)

theorem AE_eq_EC : segment_length A E = segment_length E C :=
sorry

end AE_eq_EC_l673_673488


namespace translated_parabola_vertex_l673_673206

theorem translated_parabola_vertex :
  let original_parabola := λ x : ℝ, -x^2
  let translated_parabola := λ x : ℝ, -(x - 3)^2 - 2
  (∃ v : ℝ × ℝ, v = (3, -2)) ∧ (∀ x : ℝ, translated_parabola x = -(x - 3)^2 - 2) := 
begin
  sorry
end

end translated_parabola_vertex_l673_673206


namespace imaginary_part_of_conjugate_of_Z_l673_673032

-- Define the given complex number Z
noncomputable def Z : ℂ := ((complex.I - 1) ^ 2 + 4) / (complex.I + 1)

-- Conjugate of a complex number
noncomputable def conjugate (z : ℂ) : ℂ := complex.conj z

-- Imaginary part of a complex number
noncomputable def imaginary_part (z : ℂ) : ℂ := z.im

-- The theorem to prove that the imaginary part of the conjugate of Z is 3
theorem imaginary_part_of_conjugate_of_Z : imaginary_part (conjugate Z) = 3 := 
by sorry

end imaginary_part_of_conjugate_of_Z_l673_673032


namespace tamika_carlos_probability_l673_673685

theorem tamika_carlos_probability :
  let tamika_set := {11, 12, 13}
  let carlos_set := {4, 6, 7}
  let tamika_products := {11 * 12, 11 * 13, 12 * 13}
  let carlos_products := {4 * 6, 4 * 7, 6 * 7}
  let favorable_combinations := 
        { (tp, cp) | tp ∈ tamika_products, cp ∈ carlos_products, tp > cp }
  (favorable_combinations.card : ℚ) / 
  ((tamika_products.card * carlos_products.card) : ℚ) = 1 := by
{
  sorry
}

end tamika_carlos_probability_l673_673685


namespace taozi_is_faster_than_xiaoxiao_l673_673168

theorem taozi_is_faster_than_xiaoxiao : 
  let taozi_speed := 210
  let xiaoxiao_distance := 500
  let xiaoxiao_time := 3
  let xiaoxiao_speed := xiaoxiao_distance / xiaoxiao_time
  taozi_speed > xiaoxiao_speed
:= by
  let taozi_speed := 210
  let xiaoxiao_distance := 500
  let xiaoxiao_time := 3
  let xiaoxiao_speed := xiaoxiao_distance / xiaoxiao_time
  sorry

end taozi_is_faster_than_xiaoxiao_l673_673168


namespace second_caterer_cheaper_l673_673650

/-- Define the cost function for the first caterer -/
def cost_first (x : ℕ) : ℕ := 120 + 18 * x

/-- Define the cost function for the second caterer -/
def cost_second (x : ℕ) : ℕ := 250 + 15 * x

/-- The theorem states that for the second caterer to be cheaper, the number of 
    people must be at least 44 -/
theorem second_caterer_cheaper (x : ℕ) (h : x >= 44) : cost_first x > cost_second x :=
by {
  -- Application of the given condition and eventual goal to prove
  unfold cost_first cost_second,
  -- The inequality to solve given the condition
  sorry
}

end second_caterer_cheaper_l673_673650


namespace opposite_number_of_neg_two_reciprocal_of_three_abs_val_three_eq_l673_673708

theorem opposite_number_of_neg_two (a : Int) (h : a = -2) :
  -a = 2 := by
  sorry

theorem reciprocal_of_three (x y : Real) (hx : x = 3) (hy : y = 1 / 3) : 
  x * y = 1 := by
  sorry

theorem abs_val_three_eq (x : Real) (hx : abs x = 3) :
  x = -3 ∨ x = 3 := by
  sorry

end opposite_number_of_neg_two_reciprocal_of_three_abs_val_three_eq_l673_673708


namespace area_calculation_l673_673418

noncomputable def area_bounded_by_parametric_curve_and_line : ℝ :=
  let x := λ t : ℝ, 2 * Real.sqrt 2 * Real.cos t
  let y := λ t : ℝ, 3 * Real.sqrt 2 * Real.sin t
  -12 * (intervalIntegral (λ t, Real.sin t ^ 2) (3 * Real.pi / 4) (Real.pi / 4)) -
  (3 * (2 - (-2)))

theorem area_calculation :
  area_bounded_by_parametric_curve_and_line = 3 * Real.pi - 6 := 
sorry

end area_calculation_l673_673418


namespace smaller_sector_area_l673_673837

-- Define the sides of the triangle
def p : ℝ := 7
def q : ℝ := 5
def r : ℝ := 8

-- Calculate necessary values using given conditions
def cos_P : ℝ := (q^2 + r^2 - p^2) / (2 * q * r)
def P : ℝ := Real.arccos cos_P
def sector_angle : ℝ := Real.pi / 3

def s : ℝ := (p + q + r) / 2

-- Area of the triangle using Heron's formula
def Δ : ℝ := Real.sqrt (s * (s - p) * (s - q) * (s - r))

-- Circumradius of the triangle
def R : ℝ := (p * q * r) / (4 * Δ)

-- Area of one of the smaller sectors
def sector_area : ℝ := (1 / 6) * Real.pi * R^2

theorem smaller_sector_area :
  sector_area = (49 / 18) * Real.pi :=
sorry

end smaller_sector_area_l673_673837


namespace ellipse_standard_eq_equation_of_line_l_l673_673034

theorem ellipse_standard_eq 
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (F : ℝ × ℝ) (hF : F = (ℂ.sqrt 6, 0))
  (chord_len : ℝ) (h3 : chord_len = ℂ.sqrt 2) :
  (↑(\frac{x^2}{8}) + \(\frac{y^2}{2}) = 1) :=
sorry

theorem equation_of_line_l 
  (h : ∀ P Q : ℝ × ℝ, (x - y) = 1)
  (hP1 : P = (1, 0))
  (hN : ∃ y0 : ℝ, (-1, y0))
  (h_triangle : ∃ a : (P Q : ℝ × ℝ), is_equilateral_triangle (N : ℝ × ℝ) (hN)) :
  (x ± (ℂ.sqrt 10) * y - 1) = 0 :=
sorry

end ellipse_standard_eq_equation_of_line_l_l673_673034


namespace ice_cream_depth_l673_673400

noncomputable def volume_sphere (r : ℝ) := (4/3) * Real.pi * r^3
noncomputable def volume_cylinder (r h : ℝ) := Real.pi * r^2 * h

theorem ice_cream_depth
  (radius_sphere : ℝ)
  (radius_cylinder : ℝ)
  (density_constancy : volume_sphere radius_sphere = volume_cylinder radius_cylinder (h : ℝ)) :
  h = 9 / 25 := by
  sorry

end ice_cream_depth_l673_673400


namespace correct_proposition_is_D_l673_673344

-- Definitions of the propositions
def proposition_A := ∀ (l1 l2 l3 l4 : Line), (connected_in_sequence l1 l2 l3 l4) → coplanar l1 l2 l3 l4
def proposition_B := ∀ (l1 l2 l3 : Line), (intersect_pairwise l1 l2 l3) → coplanar l1 l2 l3
def proposition_C := ∀ (p1 p2 p3 : Point), (determine_plane p1 p2 p3)
def proposition_D := ∀ (l1 l2 l3 : Line), (intersects l1 l2 ∧ intersects l1 l3 ∧ parallel l2 l3) → coplanar l1 l2 l3

-- The theorem to be proven
theorem correct_proposition_is_D : proposition_D := sorry

end correct_proposition_is_D_l673_673344


namespace largest_divisor_of_five_consecutive_integers_l673_673262

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l673_673262


namespace negation_is_true_l673_673706

variables {α : Type*} {l : α}

-- Define the conditions
def perpendicular_to_two_intersecting_lines (l : α) (P : α → α → Prop) : Prop :=
  ∃ x y, x ≠ y ∧ P l x ∧ P l y

def line_perpendicular_to_plane (l : α) (P : α → α → Prop) : Prop :=
  ∀ (x : α), P l x

-- Proposition and its negation:
def proposition (P : α → α → Prop) : Prop :=
  perpendicular_to_two_intersecting_lines l P → line_perpendicular_to_plane l P

def negation (P : α → α → Prop) : Prop :=
  ¬ (perpendicular_to_two_intersecting_lines l P) → ¬ (line_perpendicular_to_plane l P)

-- Define the truth of the negation
def truth_of_negation (P : α → α → Prop) : Prop :=
  ∀ (x : α), x ≠ y ∧ P l x → P l y → negation P

-- The equivalent proof problem in Lean 4 statement:
theorem negation_is_true (P : α → α → Prop) : truth_of_negation P := by
  sorry

end negation_is_true_l673_673706


namespace rate_of_interest_is_correct_l673_673776

noncomputable def find_rate_of_interest (A₃ A₄ : ℝ) (t₃ t₄ : ℝ) (n : ℝ) : ℝ :=
  let r : ℝ → ℝ → ℝ := λ P, ((A₄ / P)^(1/t₄) - 1)
  let P₃ : ℝ := A₃ / (1 + r A₃)^t₃
  r P₃

theorem rate_of_interest_is_correct :
  ∀ (A₃ A₄ : ℝ) (t₃ t₄ : ℝ) (n : ℝ),
  (A₃ = 800) →
  (A₄ = 820) →
  (t₃ = 3) →
  (t₄ = 4) →
  (n = 1) →
  find_rate_of_interest A₃ A₄ t₃ t₄ n = 0.025 :=
by
  intros A₃ A₄ t₃ t₄ n H₁ H₂ H₃ H₄ H₅
  sorry

end rate_of_interest_is_correct_l673_673776


namespace exist_pairing_odd_sum_not_exist_pairing_even_sum_l673_673789

/-- 
Given 100 points labeled from 1 to 100 arranged in an arbitrary order on a circle, 
this theorem proves that there exists a pairing of these points such that:
1. The line segments connecting the pairs do not intersect.
2. The sums of the numbers of each pair are odd.
--/
theorem exist_pairing_odd_sum (points : Finset ℕ) (h100 : points.card = 100) 
  (point_labels : ℕ → ℕ) (hlabels : ∀ x ∈ points, point_labels x ∈ (Finset.range 101)) 
  (pairing : Finset (ℕ × ℕ)) 
  (hpairing : pairing.card = 50)
  (hodd_walk : ∀ p ∈ pairing, (point_labels p.1 + point_labels p.2) % 2 = 1) : 
  ∃ non_intersecting_pairing : Finset (ℕ × ℕ), 
    (∀ p ∈ non_intersecting_pairing, p ∈ pairing) ∧
    (∀ a b c d, (a, b) ∈ non_intersecting_pairing → 
                (c, d) ∈ non_intersecting_pairing → 
                (a = c ∧ b = d ∨ (not (SegIntersect a b c d)))), sorry

/-- 
Given 100 points labeled from 1 to 100 arranged in an arbitrary order on a circle, 
this theorem proves that it is impossible to pair these points such that:
1. The line segments connecting the pairs do not intersect.
2. The sums of the numbers of each pair are even.
--/
theorem not_exist_pairing_even_sum (points : Finset ℕ) (h100 : points.card = 100) 
  (point_labels : ℕ → ℕ) (hlabels : ∀ x ∈ points, point_labels x ∈ (Finset.range 101)) 
  (pairing : Finset (ℕ × ℕ)) 
  (hpairing : pairing.card = 50) :
  ¬(∃ non_intersecting_pairing : Finset (ℕ × ℕ),
    (∀ p ∈ non_intersecting_pairing, p ∈ pairing) ∧
    (∀ a b c d, (a, b) ∈ non_intersecting_pairing → 
                (c, d) ∈ non_intersecting_pairing → 
                (a = c ∧ b = d ∨ (not (SegIntersect a b c d)))) ∧
    (∀ p ∈ non_intersecting_pairing, (point_labels p.1 + point_labels p.2) % 2 = 0)), sorry

end exist_pairing_odd_sum_not_exist_pairing_even_sum_l673_673789


namespace probability_integer_between_21_and_30_l673_673641

/-- 
  Melinda rolls two standard six-sided dice and forms a two-digit number.
  Prove that the probability that she will form a number between 21 and 30 (inclusive) is 11/36.
-/
theorem probability_integer_between_21_and_30 :
  let dice_outcomes := Finset.prod (Finset.range 1 7) (Finset.range 1 7),
      event_21_to_30 (a b : Fin) : ℕ := (1 / 36 : nnreal) * if (21 ≤ 10 * a + b ∧ 10 * a + b ≤ 30) ∨ (21 ≤ 10 * b + a ∧ 10 * b + a ≤ 30) then 1 else 0 in
  dice_outcomes.sum (λ p, event_21_to_30 p.1 p.2) = 11 / 36 := sorry

end probability_integer_between_21_and_30_l673_673641


namespace first_five_valid_seeds_l673_673774

-- Definitions based on the conditions
def isValidSeed (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 500
def randomNumbers : List ℕ := [331, 572, 455, 68, 877, 47, 447]

-- Main statement to prove
theorem first_five_valid_seeds : List.filter isValidSeed randomNumbers = [331, 455, 68, 47, 447] :=
by
    -- We skip the proof with sorry
    sorry

end first_five_valid_seeds_l673_673774


namespace area_of_triangle_AOB_l673_673588

theorem area_of_triangle_AOB (x y t θ : ℝ)
  (h1 : x = 1 + (real.sqrt 2)/2 * t)
  (h2 : y = (real.sqrt 2)/2 * t)
  (h3 : x^2 + y^2 - 4 * x = 0)
  (h4 : ∀ t1 t2 : ℝ, t1 + t2 = real.sqrt 2 ∧ t1 * t2 = -3) :
  1/2 * real.sqrt (real.sqrt 2^2 - 4 * -3) * (real.sqrt 2) / 2 = real.sqrt 7 / 2 :=
by
  sorry

end area_of_triangle_AOB_l673_673588


namespace exists_n_sum_digits_n3_eq_million_l673_673719

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem exists_n_sum_digits_n3_eq_million :
  ∃ n : ℕ, sum_digits n = 100 ∧ sum_digits (n ^ 3) = 1000000 := sorry

end exists_n_sum_digits_n3_eq_million_l673_673719


namespace N_subset_proper_M_l673_673047

open Set Int

def set_M : Set ℝ := {x | ∃ k : ℤ, x = (k + 2) / 4}
def set_N : Set ℝ := {x | ∃ k : ℤ, x = (2 * k + 1) / 4}

theorem N_subset_proper_M : set_N ⊂ set_M := by
  sorry

end N_subset_proper_M_l673_673047


namespace ellipse_eccentricity_l673_673506

theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1):
  let c := b in
  let e := c / a in
  a = (2:ℝ).sqrt * c → 
  e = (2:ℝ).sqrt / 2 :=
by {
  sorry
}

end ellipse_eccentricity_l673_673506


namespace least_n_divisible_by_some_not_all_l673_673332

theorem least_n_divisible_by_some_not_all (n : ℕ) (h : 1 ≤ n):
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ k ∣ (n^2 - n)) ∧ ¬ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ (n^2 - n)) ↔ n = 3 :=
by
  sorry

end least_n_divisible_by_some_not_all_l673_673332


namespace rectangle_area_perimeter_eq_l673_673490

theorem rectangle_area_perimeter_eq (x : ℝ) (h : 4 * x * (x + 4) = 2 * 4 * x + 2 * (x + 4)) : x = 1 / 2 :=
sorry

end rectangle_area_perimeter_eq_l673_673490


namespace archery_competition_l673_673081

theorem archery_competition (points : Finset ℕ) (product : ℕ) : 
  points = {11, 7, 5, 2} ∧ product = 38500 → 
  ∃ n : ℕ, n = 7 := 
by
  intros h
  sorry

end archery_competition_l673_673081


namespace max_min_diff_l673_673435

-- Define the distinct planes intersecting the tetrahedron 
def distinct_planes {T : Type} (p : Finset T) : Prop :=
p.card = n

-- Define the intersection of the planes and the faces of the tetrahedron 
def intersection (P S : Type) : Prop :=
-- Details of the intersection definition

def max_planes {T : Type} (p : Finset T) : Prop :=
p.card = 8

def min_planes {T : Type} (p : Finset T) : Prop :=
p.card = 4

theorem max_min_diff {T : Type} (p : Finset T) :
  distinct_planes p ∧ max_planes p ∧ min_planes p → (8 - 4 = 4) := 
by 
  sorry

end max_min_diff_l673_673435


namespace right_triangle_shorter_leg_l673_673559

theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : c = 65) : a = 25 ∨ b = 25 := 
by
  sorry

end right_triangle_shorter_leg_l673_673559


namespace minimum_pieces_needed_to_control_all_l673_673716

-- Definition of the problem conditions
structure RhombusWithEquilateralTriangles where
  sides_divided : ℕ
  angle_deg : ℕ
  is_divisible : sides_divided = 9
  angle_is_60_deg : angle_deg = 60

-- Definition of the game board and control concept
structure GameBoard where
  rhombus : RhombusWithEquilateralTriangles
  controlled_triangles_needed : ℕ
  is_equilateral : ∀ (triangle_center : Point), triangle_center → 
      List (List Point)
  control_lines : ∀ (piece_position : Point), List (Line)

-- The main proof problem
theorem minimum_pieces_needed_to_control_all (board : GameBoard) :
  board.controlled_triangles_needed = 6 := sorry

end minimum_pieces_needed_to_control_all_l673_673716


namespace smallest_composite_no_prime_under_15_correct_l673_673935

-- Define the concept of a composite number
def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

-- Define the concept of having no prime factors less than 15
def has_no_prime_factors_less_than_15 (n : ℕ) : Prop := 
  ∀ p : ℕ, p.prime ∧ p ∣ n → 15 ≤ p

-- Define the smallest composite number with no prime factors less than 15 
def smallest_composite_no_prime_under_15 : ℕ := 289

-- Prove that this is the smallest number satisfying our conditions
theorem smallest_composite_no_prime_under_15_correct : 
  is_composite smallest_composite_no_prime_under_15 ∧
  has_no_prime_factors_less_than_15 smallest_composite_no_prime_under_15 ∧
  ∀ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than_15 n → n ≥ smallest_composite_no_prime_under_15 :=
by 
  sorry

end smallest_composite_no_prime_under_15_correct_l673_673935


namespace cement_mixture_weight_l673_673378

theorem cement_mixture_weight {W : ℝ} :
  (W * (1/4)) + (W * (2/5)) + 14 = W → W = 40 :=
by
  assume h : (W * (1/4)) + (W * (2/5)) + 14 = W,
  sorry

end cement_mixture_weight_l673_673378


namespace find_value_of_b_l673_673185

theorem find_value_of_b (a b : ℝ)
  (h1 : ∃ (x y: ℝ), x = 1 ∧ y = 3 ∧ y = 2 * x + 1)
  (h2 : ∃ (x y: ℝ), x = 1 ∧ y = 3 ∧ y = x^3 + a * x + b)
  (h3 : ∃ (x : ℝ), x = 1 ∧ (deriv (λ x, x^3 + a * x + b) x = 2)) :
  b = 3 :=
sorry

end find_value_of_b_l673_673185


namespace volume_between_concentric_spheres_l673_673732

-- Define the radii of the spheres
def r_small : ℝ := 5
def r_large : ℝ := 8

-- Define the volumes of the spheres
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- State the problem to prove
theorem volume_between_concentric_spheres :
  volume_of_sphere r_large - volume_of_sphere r_small = 516 * Real.pi :=
by
  -- Placeholder for the proof
  sorry

end volume_between_concentric_spheres_l673_673732


namespace smallest_composite_proof_l673_673901

-- Define what it means for a number not to have prime factors less than 15
def no_prime_factors_less_than_15 (n : ℕ) : Prop :=
  ∀ p : ℕ, nat.prime p → p ∣ n → p ≥ 15

-- Define what it means for a number to be the smallest composite number with the above property
def smallest_composite_without_prime_factors_less_than_15 (n : ℕ) : Prop :=
  nat.composite n ∧ no_prime_factors_less_than_15 n ∧
  ∀ m : ℕ, nat.composite m → no_prime_factors_less_than_15 m → n ≤ m

theorem smallest_composite_proof : smallest_composite_without_prime_factors_less_than_15 323 :=
  sorry

end smallest_composite_proof_l673_673901


namespace find_ellipse_find_MQ_length_minimized_l673_673051

-- Definition of the circles F₁ and F₂
def F1 (x y : ℝ) : Prop := (x + real.sqrt 3) ^ 2 + y ^ 2 = 9
def F2 (x y : ℝ) : Prop := (x - real.sqrt 3) ^ 2 + y ^ 2 = 1

-- Definition of the ellipse C with foci F₁ and F₂
def Ellipse (a b : ℝ) (x y : ℝ) : Prop := (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1 ∧ a > b ∧ b > 0

-- Proof problem: (Ⅰ) Finding the ellipse
theorem find_ellipse (a b : ℝ) :
  Ellipse a b x y ∧ ∃ (P : ℝ × ℝ), F1 P.1 P.2 ∧ F2 P.1 P.2 →
  a = 2 ∧ b = 1 :=
by
    sorry

-- Definition of conditions in (Ⅱ)
def M (x y₁ : ℝ) : Prop := x = 2 * real.sqrt 3 ∧ y₁ > 0
def N (x y₂ : ℝ) : Prop := x = 2 * real.sqrt 3 ∧ y₂ < 0

theorem find_MQ_length_minimized (y₁ y₂ : ℝ) :
  M (2 * real.sqrt 3) y₁ ∧ N (2 * real.sqrt 3) y₂ ∧
  (3 * real.sqrt 3, y₁) • (real.sqrt 3, y₂) = 0 →
  |y₁ - y₂| = 6 ∧ |MQ| = 3 :=
by
    sorry

end find_ellipse_find_MQ_length_minimized_l673_673051


namespace geometric_sequence_probability_hyperbola_area_circle_parabola_tangent_ellipse_hyperbola_eccentricities_l673_673366

-- Problem 1
theorem geometric_sequence_probability (a r : ℤ) (n : ℕ) (h_a : a = 1) (h_r : r = -3) (h_n : n = 10) :
  let seq := list.range n.map (λ i, a * r ^ i) in
  let count := seq.filter (λ x, x < 8) in
  (count.length : ℚ) / n = 3 / 5 := 
sorry

-- Problem 2
theorem hyperbola_area (F1 F2 P : ℝ × ℝ) (h_F1 : F1 = (5, 4)) (h_F2 : F2 = (5, 5)) (h_P : P = (6, 1)) :
  let d1 := sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2),
      d2 := sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) in
  0.5 * d1 * d2 * sin (real.pi / 3) = 9 * real.sqrt 3 := 
sorry

-- Problem 3
theorem circle_parabola_tangent (m : ℝ) (h_c : abs (m / 2 - 1) = sqrt ((1 + m^2) / 4)) :
  m = 3 / 4 := 
sorry

-- Problem 4
theorem ellipse_hyperbola_eccentricities (e1 e2 c m n : ℝ)
  (h_eq1 : m^2 + n^2 = 4 * c^2) 
  (h_eq2 : e1 = 2 * c / (m + n)) 
  (h_eq3 : e2 = 2 * c / (m - n)) :
  (e1 * e2 / real.sqrt (e1^2 + e2^2)) = real.sqrt 2 / 2 := 
sorry

end geometric_sequence_probability_hyperbola_area_circle_parabola_tangent_ellipse_hyperbola_eccentricities_l673_673366


namespace atomic_weight_Ba_l673_673894

-- Definitions for conditions
def atomic_weight_O : ℕ := 16
def molecular_weight_compound : ℕ := 153

-- Theorem statement
theorem atomic_weight_Ba : ∃ bw, molecular_weight_compound = bw + atomic_weight_O ∧ bw = 137 :=
by {
  -- Skip the proof
  sorry
}

end atomic_weight_Ba_l673_673894


namespace cylinder_volume_approx_l673_673697

noncomputable def volume_cylinder (diameter height : ℝ) : ℝ :=
  let r := diameter / 2 in
  Real.pi * (r ^ 2) * height

theorem cylinder_volume_approx :
  volume_cylinder 6 5 ≈ 141.37 :=
by
  have diameter := 6
  have height := 5
  have r := diameter / 2
  have volume := Real.pi * (r ^ 2) * height
  have approx_pi : Real.pi ≈ 3.14159 := by norm_num
  sorry

end cylinder_volume_approx_l673_673697


namespace range_of_abs_function_l673_673426

theorem range_of_abs_function:
  (∀ y, ∃ x : ℝ, y = |x + 3| - |x - 5|) → ∀ y, y ≤ 8 :=
by
  sorry

end range_of_abs_function_l673_673426


namespace percentage_reduction_l673_673390

variable (P R : ℝ)
variable (ReducedPrice : R = 15)
variable (AmountMore : 900 / 15 - 900 / P = 6)

theorem percentage_reduction (ReducedPrice : R = 15) (AmountMore : 900 / 15 - 900 / P = 6) :
  (P - R) / P * 100 = 10 :=
by
  sorry

end percentage_reduction_l673_673390


namespace simple_interest_fraction_l673_673717

theorem simple_interest_fraction (P : ℝ) (R T : ℝ) (hR: R = 4) (hT: T = 5) :
  (P * R * T / 100) / P = 1 / 5 := 
by
  sorry

end simple_interest_fraction_l673_673717


namespace shorter_leg_of_right_triangle_l673_673556

theorem shorter_leg_of_right_triangle (a b : ℕ) (h : a^2 + b^2 = 65^2) : min a b = 16 :=
sorry

end shorter_leg_of_right_triangle_l673_673556


namespace meal_serving_count_correct_l673_673755

def meals_served_correctly (total_people : ℕ) (meal_type : Type*)
  (orders : meal_type → ℕ) (correct_meals : ℕ) : ℕ :=
  -- function to count the number of ways to serve meals correctly
  sorry

theorem meal_serving_count_correct (total_people : ℕ) (meal_type : fin 3) 
  [decidable_eq meal_type]
  (orders : fin 3 → ℕ) (h_orders : orders = (λ x, 4)) :
  meals_served_correctly total_people meal_type orders 2 = 22572 :=
  begin
    have orders_correct: ∀ x, orders x = 4 := by rw h_orders,
    -- Further steps and usage of derangements would be here, 
    -- but for now we will skip to the final count.
    sorry
  end

end meal_serving_count_correct_l673_673755


namespace product_of_five_consecutive_integers_divisible_by_240_l673_673279

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l673_673279


namespace sum_k_bounds_l673_673154

variable {α : Type} [LinearOrderedRing α]

noncomputable def k_i (n : ℕ) (xs : Fin n → α) (i : Fin n) : α :=
(xs (i - 1) + xs (i + 1)) / xs i

theorem sum_k_bounds (n : ℕ) (xs : Fin n → α) (hn : 4 ≤ n) :
  2 * n ≤ ∑ i, k_i n xs i ∧ ∑ i, k_i n xs i < 3 * n :=
  sorry

end sum_k_bounds_l673_673154


namespace rectangle_area_arithmetic_progression_l673_673856

theorem rectangle_area_arithmetic_progression (a d : ℝ) : 
  let shorter_side := a in
  let longer_side := a + d in
  let K := shorter_side * longer_side in
  K = a^2 + a * d :=
by
  sorry

end rectangle_area_arithmetic_progression_l673_673856


namespace sewage_treatment_plant_problems_l673_673497

-- Define the cost function
def cost_function (x : ℝ) (m : ℝ) : ℝ :=
  (1 / 400) * x^2 - m * x + 25

-- Given condition that when x = 120, y = 49 and cost of processing is 0.9 thousand yuan
def condition1 (m : ℝ) : Prop :=
  cost_function 120 m = 49

-- We want to minimize the cost per thousand tons
def cost_per_thousand_tons (x : ℝ) (m : ℝ) : ℝ :=
  cost_function x m / x

-- Given condition for minimum cost per thousand tons
def minimize_cost (m : ℝ) : Prop :=
  ∀ x, 100000 <= x ∧ x <= 210000 → cost_per_thousand_tons x m ≥ cost_per_thousand_tons 100 m

-- Define the profit function
def profit_function (x : ℝ) (m : ℝ) : ℝ :=
  0.9 * x - cost_function x m

-- We want to maximize the profit function
def maximize_profit (m : ℝ) : Prop :=
  profit_function 200 m = 75

-- Finally, we write down the theorem statements
theorem sewage_treatment_plant_problems (m : ℝ) (h1 : condition1 m) :
  minimize_cost m ∧ maximize_profit m :=
by { sorry }

end sewage_treatment_plant_problems_l673_673497


namespace paint_area_l673_673604

def height : ℝ := 10
def length : ℝ := 15
def painting_height : ℝ := 3
def painting_length : ℝ := 6

theorem paint_area :
  height * length - painting_height * painting_length = 132 := by
  sorry

end paint_area_l673_673604


namespace athena_sandwiches_l673_673843

theorem athena_sandwiches :
  ∃ S : ℕ, (3 * S + 5 = 14) ∧ S = 3 :=
begin
  use 3,
  split,
  { linarith },
  { refl }
end

end athena_sandwiches_l673_673843


namespace smallest_composite_no_prime_factors_below_15_correct_l673_673928

def smallest_composite_no_prime_factors_below_15 : Nat :=
  323
  
theorem smallest_composite_no_prime_factors_below_15_correct :
  (∀ n < 15, Prime n → ¬ (n ∣ smallest_composite_no_prime_factors_below_15)) ∧
  (∃ p q, Prime p ∧ Prime q ∧ p ≠ q ∧ smallest_composite_no_prime_factors_below_15 = p * q) :=
by
  -- Proof skipped
  sorry

end smallest_composite_no_prime_factors_below_15_correct_l673_673928


namespace largest_integer_dividing_consecutive_product_l673_673254

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l673_673254


namespace john_marbles_choice_l673_673115

theorem john_marbles_choice 
  (total_marbles : ℕ)
  (special_marbles : ℕ)
  (red_marbles : ℕ)
  (green_marbles : ℕ)
  (blue_marbles : ℕ)
  (needed_marbles_for_choice : ℕ)
  (required_special_marbles : ℕ)
  (ordinary_marbles := total_marbles - special_marbles)
  (comb : Π {n k : ℕ}, ℕ) :
  total_marbles = 15 →
  special_marbles = 6 →
  red_marbles = 2 →
  green_marbles = 2 →
  blue_marbles = 2 →
  needed_marbles_for_choice = 5 →
  required_special_marbles ≥ 2 →
  comb.special_marbles 6 2 * comb.ordinary_marbles 9 3 +
  comb.special_marbles 6 3 * comb.ordinary_marbles 9 2 +
  comb.special_marbles 6 4 * comb.ordinary_marbles 9 1 +
  comb.special_marbles 6 5 * comb.ordinary_marbles 9 0 = 2121 :=
sorry

end john_marbles_choice_l673_673115


namespace product_of_five_consecutive_integers_divisible_by_240_l673_673278

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l673_673278


namespace max_value_x2_y3_z_l673_673631

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  if x + y + z = 3 then x^2 * y^3 * z else 0

theorem max_value_x2_y3_z
  (x y z : ℝ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxyz : x + y + z = 3) :
  maximum_value x y z ≤ 9 / 16 := sorry

end max_value_x2_y3_z_l673_673631


namespace find_n_l673_673183

theorem find_n (n : ℤ) : 
  50 < n ∧ n < 120 ∧ (n % 8 = 0) ∧ (n % 7 = 3) ∧ (n % 9 = 3) → n = 192 :=
by
  sorry

end find_n_l673_673183


namespace equidistant_planes_count_l673_673587

-- Definitions:
-- Four points in space (A, B, C, D) not on the same plane
variables {A B C D : Point}
axiom not_coplanar : ¬ coplanar A B C D

-- The theorem statement:
theorem equidistant_planes_count : ∃ n : ℕ, n = 7 ∧ (∀ π : Plane, (π.equidistant A B C D ↔ π ∈ possible_planes)) :=
sorry

end equidistant_planes_count_l673_673587


namespace min_cost_to_determine_no_integer_roots_l673_673613

theorem min_cost_to_determine_no_integer_roots (n : ℕ) (P : ℤ[X]) (h_deg : P.degree ≤ n) : 
  ∃ k : ℤ, (∀ t : ℤ, P.eval t ≠ k) → (∃ m, m = 2 * n + 1) :=
sorry

end min_cost_to_determine_no_integer_roots_l673_673613


namespace plane_determined_by_point_and_line_l673_673540

-- Definitions related to the problem
variables {Point : Type*} {Line : Type*} {Plane : Type*}

-- Assume some basic geometry axioms about Lines and Planes.
axiom point_not_on_line : Point → Line → Prop
axiom points_on_line_form_one_plane : (p1 p2 : Point) (l : Line), (point_not_on_line p1 l) → (point_not_on_line p2 l) → Plane

-- Given conditions
variables (P : Point) (l : Line)
hypothesis h_not_on_line : point_not_on_line P l

-- Proof statement
theorem plane_determined_by_point_and_line : ∃! (pl : Plane), true := 
sorry

end plane_determined_by_point_and_line_l673_673540


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673320

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673320


namespace point_on_parabola_distance_l673_673007

theorem point_on_parabola_distance (a b : ℝ) (h1 : a^2 = 20 * b) (h2 : |b + 5| = 25) : |a * b| = 400 :=
sorry

end point_on_parabola_distance_l673_673007


namespace shorter_leg_of_right_triangle_l673_673578

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l673_673578


namespace cans_per_person_day1_l673_673849

theorem cans_per_person_day1
  (initial_cans : ℕ)
  (people_day1 : ℕ)
  (restock_day1 : ℕ)
  (people_day2 : ℕ)
  (cans_per_person_day2 : ℕ)
  (total_cans_given_away : ℕ) :
  initial_cans = 2000 →
  people_day1 = 500 →
  restock_day1 = 1500 →
  people_day2 = 1000 →
  cans_per_person_day2 = 2 →
  total_cans_given_away = 2500 →
  (total_cans_given_away - (people_day2 * cans_per_person_day2)) / people_day1 = 1 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- condition trivially holds
  sorry

end cans_per_person_day1_l673_673849


namespace two_digit_number_square_equals_cube_of_sum_of_digits_l673_673881

theorem two_digit_number_square_equals_cube_of_sum_of_digits : ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧
  let A := n / 10 in
  let B := n % 10 in
  A ≠ B ∧ n^2 = (A + B)^3 :=
begin
  use 27,
  split, { dec_trivial },
  split, { dec_trivial },
  let A := 27 / 10,
  let B := 27 % 10,
  split,
  { exact dec_trivial },
  simp [A, B],
  exact dec_trivial,
end

end two_digit_number_square_equals_cube_of_sum_of_digits_l673_673881


namespace proof_goal_l673_673482

-- Variables representing the propositions p and q
variable (x : ℝ)

-- Proposition p: If x > 0, the minimum value of y = x + 1/(2x) is 1.
def p : Prop := (x > 0 → (x + 1 / (2 * x)) = 1)

-- Proposition q: If x > 1, then x^2 + 2x - 3 > 0.
def q : Prop := (x > 1 → x^2 + 2 * x - 3 > 0)

-- Given conditions
axiom not_p : ¬p
axiom q_true : q

-- The goal is to prove that p ∨ q is true
theorem proof_goal : p ∨ q := by
  sorry

end proof_goal_l673_673482


namespace balls_distribution_l673_673659

theorem balls_distribution :
  ∃ (number_of_ways : ℕ), number_of_ways = 2268 ∧
  ∀ (balls : fin 8 → ℕ) (boxes : fin 3 → ℕ), 
  (∀ i, 1 ≤ i+1 → boxes i ≤ balls.injective.sum) →
  boxes 0 ≥ 1 ∧ boxes 1 ≥ 2 ∧ boxes 2 ≥ 3 :=
by
  sorry

end balls_distribution_l673_673659


namespace xyz_poly_identity_l673_673630

theorem xyz_poly_identity (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)
  (h4 : x + y + z = 0) (h5 : xy + xz + yz ≠ 0) :
  (x^6 + y^6 + z^6) / (xyz * (xy + xz + yz)) = 6 :=
by
  sorry

end xyz_poly_identity_l673_673630


namespace probability_of_gcd_one_is_13_over_14_l673_673733

open Finset

noncomputable def probability_gcd_one : ℚ :=
let s := {1, 2, 3, 4, 5, 6, 7, 8}
let subsetsOfThree := s.powerset.filter (λ t, t.card = 3)
let nonRelativelyPrimeSubsets := {(t : Finset ℕ) ∈ subsetsOfThree | (∀ a b c ∈ t, gcd (gcd a b) c ≠ 1)}
let totalSubsets := subsetsOfThree.card
let nonRelativelyPrimeCount := nonRelativelyPrimeSubsets.card
in 1 - (nonRelativelyPrimeCount / totalSubsets : ℚ)

theorem probability_of_gcd_one_is_13_over_14 :
  probability_gcd_one = 13 / 14 := by sorry

end probability_of_gcd_one_is_13_over_14_l673_673733


namespace ellipse_eqn_slopes_constant_l673_673012

-- Given conditions
def ellipse_eq (a b : ℝ) (h1 : a > 1) (h2 : b > 0) (h3 : b ≤ 1) : Prop :=
  ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

def foci_left (a b : ℝ) := (-sqrt (a^2 - b^2), 0)
def foci_right (a b : ℝ) := (sqrt (a^2 - b^2), 0)

def line_l (k : ℝ) (x2 y2 : ℝ) : Prop :=
  y2 = k * x2

-- Translating the math equation to Lean statements
theorem ellipse_eqn (a b : ℝ) (h1 : a = 2) (h2 : b^2 = 1) : ellipse_eq a b h1-le h2-le h3-le :=
by sorry

theorem slopes_constant (a b : ℝ) (k k1 k2 : ℝ) (h_k : line_l k (sqrt(a^2 - b^2)) b) (h_a : a = 2) : ∃ C : ℝ, k * (1 / k1 + 1 / k2) = C :=
by sorry

end ellipse_eqn_slopes_constant_l673_673012


namespace find_d_l673_673675

theorem find_d (a b c d : ℕ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (hd : 1 < d) 
  (h_eq : ∀ M : ℝ, M ≠ 1 → (M^(1/a)) * (M^(1/(a * b))) * (M^(1/(a * b * c))) * (M^(1/(a * b * c * d))) = M^(17/24)) : d = 8 :=
sorry

end find_d_l673_673675


namespace acute_angles_theorem_l673_673988

open Real

variable (α β : ℝ)

-- Given conditions
def conditions : Prop :=
  0 < α ∧ α < π / 2 ∧
  0 < β ∧ β < π / 2 ∧
  tan α = 1 / 7 ∧
  sin β = sqrt 10 / 10

-- Proof goal
def proof_goal : Prop :=
  α + 2 * β = π / 4

-- The final theorem
theorem acute_angles_theorem (h : conditions α β) : proof_goal α β :=
  sorry

end acute_angles_theorem_l673_673988


namespace proof_problem_l673_673417
open BigOperators

def conditions : Prop :=
  -- Conditions from the problem:
  -- probability of selecting an infected mouse from non-vaccinated mice is 3/5
  let prob := (3 : ℚ) / 5 
  -- The given table data
  ∀ (a b c d : ℕ),
    (a = 40 ∧ c = 60 ∧ a + b = 100 ∧ c + d = 100) -- the table data total number of not vaccinated and vaccinated
    ∧ (b = 60 ∧ a * d - b * c > 0) -- chi-square calculation specifics
    ∧ (prob = (c:ℚ) / (a + b)) -- probability calculation agrees with c being infected

theorem proof_problem (a b c d : ℕ) (prob : ℚ) (E : fin 5 → ℚ) (chi_val : ℚ) : 
  conditions → 
  (a, d = 40 ∧ 100 ∧ 60 ∧ 100) -- data values
  ∧ (chi_val = 8) -- chi-square value
  ∧ (prob = (3 : ℚ) / 5) -- given probability condition
  ∧ (E 4 = 12 / 5) -- expected value of X in its distribution
:= 
begin
  - sorry --the proof steps are skipped
end

end proof_problem_l673_673417


namespace cannot_be_zero_l673_673703

-- Define polynomial Q(x)
def Q (x : ℝ) (f g h i j : ℝ) : ℝ := x^5 + f * x^4 + g * x^3 + h * x^2 + i * x + j

-- Define the hypotheses for the proof
def distinct_roots (a b c d e : ℝ) := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e
def one_root_is_one (f g h i j : ℝ) := Q 1 f g h i j = 0

-- Statement to prove
theorem cannot_be_zero (f g h i j a b c d : ℝ)
  (h1 : Q 1 f g h i j = 0)
  (h2 : distinct_roots 1 a b c d)
  (h3 : Q 1 f g h i j = (1-a)*(1-b)*(1-c)*(1-d)) :
  i ≠ 0 :=
by
  sorry

end cannot_be_zero_l673_673703


namespace problem_solution_l673_673044

noncomputable def inequality_holds (a b : ℝ) (n : ℕ) : Prop :=
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n - 1)

theorem problem_solution (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1 / a + 1 / b = 1)) (h4 : 0 < n):
  inequality_holds a b n :=
by
  sorry

end problem_solution_l673_673044


namespace largest_divisor_of_consecutive_five_l673_673291

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l673_673291


namespace product_of_five_consecutive_integers_divisible_by_240_l673_673280

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l673_673280


namespace sum_single_digits_l673_673104

theorem sum_single_digits (P Q R : ℕ) (hP : P ≠ Q) (hQ : Q ≠ R) (hR : R ≠ P)
  (h1 : R + R = 10)
  (h_sum : ∃ (P Q R : ℕ), P * 100 + 70 + R + 390 + R = R * 100 + Q * 10) :
  P + Q + R = 13 := 
sorry

end sum_single_digits_l673_673104


namespace range_of_a_l673_673971

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + a*x else 2*a*x - 5

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2) ↔ a < 4 :=
sorry

end range_of_a_l673_673971


namespace hyperbola_conditions_l673_673472

noncomputable def hyperbola_with_foci (a b c : ℝ) (x y : ℝ) :=
  (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_conditions :
  ∀ (a b c : ℝ),
  c = 5 →
  a^2 + b^2 = c^2 →
  (∀ x y, hyperbola_with_foci a b c x y) →
  (∀ e, e = c / a) →
  (a = 3 ∨ b = 2) :=
begin
  intros a b c hc h_sq hyp he,
  sorry
end

end hyperbola_conditions_l673_673472


namespace problem_1_problem_2_l673_673427

section problem_1
-- Given: a geometric sequence {a_n} with first term 2 and common ratio 2.
def a (n : ℕ) : ℕ := 2^(n+1)

-- Define b_n
def M (n : ℕ) : ℕ := a (n)
def m (n : ℕ) : ℕ := 2
def b (n : ℕ) : ℕ := (M n + m n) / 2

-- Calculate B_n
def B (n : ℕ) : ℕ := (List.range n).sum b

theorem problem_1 (n : ℕ) : B n = 2^n - 1 + n := sorry
end problem_1

section problem_2
-- Define M_n and m_n
variable (a : ℕ → ℕ)
def M (n : ℕ) : ℕ := (Finset.range n).sup a
def m (n : ℕ) : ℕ := (Finset.range n).inf a
def b (n : ℕ) : ℕ := (M n + m n) / 2

-- Define a_n is arithmetic if b_n is arithmetic sequence
theorem problem_2 (d' : ℕ) 
  (h : ∀ n : ℕ, b n - b (n - 1) = d') : ∀ n : ℕ, a n - a (n - 1) = 2 * d' :=
sorry
end problem_2

end problem_1_problem_2_l673_673427


namespace max_download_speed_l673_673208

def download_speed (size_GB : ℕ) (time_hours : ℕ) : ℚ :=
  let size_MB := size_GB * 1024
  let time_seconds := time_hours * 60 * 60
  size_MB / time_seconds

theorem max_download_speed (h₁ : size_GB = 360) (h₂ : time_hours = 2) :
  download_speed size_GB time_hours = 51.2 :=
by
  sorry

end max_download_speed_l673_673208


namespace smallest_composite_proof_l673_673945

noncomputable def smallest_composite_no_prime_factors_less_than_15 : ℕ :=
  289

theorem smallest_composite_proof :
  smallest_composite_no_prime_factors_less_than_15 = 289 :=
by
  sorry

end smallest_composite_proof_l673_673945


namespace tan_alpha_minus_2beta_l673_673001

theorem tan_alpha_minus_2beta (α β : Real) 
  (h1 : Real.tan (α - β) = 2 / 5)
  (h2 : Real.tan β = 1 / 2) :
  Real.tan (α - 2 * β) = -1 / 12 := 
by 
  sorry

end tan_alpha_minus_2beta_l673_673001


namespace valid_circle_count_l673_673056

def is_coprime (a b : ℕ) := Nat.gcd a b = 1

def valid_circle_arrangement (arrangement : List ℕ) : Prop :=
  arrangement.length = 8 ∧
  arrangement.nodup ∧
  ∀ i, is_coprime (arrangement.nth i) (arrangement.nth ((i + 1) % 8)) ∧
  arrangement.head = some 1 ∧
  arrangement.nth 2 ∉ {some 3, some 6} ∧  -- Fixed condition for 6 and 3 not being adjacent.
  arrangement.nth 7 ∉ {some 3, some 6} 

noncomputable def number_of_valid_arrangements : ℕ :=
  -- This is to denote the number of valid ways
  sorry

theorem valid_circle_count : number_of_valid_arrangements = 72 :=
  sorry

end valid_circle_count_l673_673056


namespace probability_gcd_one_is_49_over_56_l673_673738

def is_gcd_one (a b c : ℕ) : Prop := Nat.gcd a (Nat.gcd b c) = 1

def count_choices_with_gcd_one : ℕ :=
  ((Finset.powersetLen 3 (Finset.range 9)).filter (λ s, match s.toList with
    | [a, b, c] => is_gcd_one a b c
    | _ => false
  end)).card

def total_choices : ℕ := (Finset.powersetLen 3 (Finset.range 9)).card

theorem probability_gcd_one_is_49_over_56 :
  (count_choices_with_gcd_one : ℚ) / total_choices = 49 / 56 := by
  sorry

end probability_gcd_one_is_49_over_56_l673_673738


namespace cone_angle_l673_673543

-- Given condition
def ratio (r h l: ℝ) : Prop := (π * r * l) / (r * h) = (2 * sqrt 3 * π) / 3

-- Target statement to prove
theorem cone_angle (r h l θ : ℝ) (h_r : ratio r h l) : θ = π / 6 :=
by 
  sorry

end cone_angle_l673_673543


namespace largest_divisor_of_consecutive_five_l673_673296

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l673_673296


namespace number_of_valid_x_values_l673_673055

theorem number_of_valid_x_values : 
  {x : ℕ // 10 ≤ x ∧ x < 100 ∧ (3 * x < 100) ∧ (4 * x ≥ 100)}.card = 9 :=
by
  sorry

end number_of_valid_x_values_l673_673055


namespace child_admission_charge_l673_673690

-- Given conditions
variables (A C : ℝ) (T : ℝ := 3.25) (n : ℕ := 3)

-- Admission charge for an adult
def admission_charge_adult : ℝ := 1

-- Admission charge for a child
def admission_charge_child (C : ℝ) : ℝ := C

-- Total cost paid by adult with 3 children
def total_cost (A C : ℝ) (n : ℕ) : ℝ := A + n * C

-- The proof statement
theorem child_admission_charge (C : ℝ) : total_cost 1 C 3 = 3.25 -> C = 0.75 :=
by
  sorry

end child_admission_charge_l673_673690


namespace largest_divisor_of_consecutive_five_l673_673294

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l673_673294


namespace largest_divisor_of_product_of_5_consecutive_integers_l673_673240

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l673_673240


namespace magnitude_z1_pure_imaginary_l673_673979

theorem magnitude_z1_pure_imaginary (a : ℝ) (ha : (a - 2 = 0)) : 
  let z1 := complex.mk a 1,
      z2 := complex.mk 1 (-2),
      z1_conj_z2 := z1 * (complex.mk 1 2) in
  (z1_conj_z2.re = 0) → complex.abs z1 = real.sqrt 5 :=
sorry

end magnitude_z1_pure_imaginary_l673_673979


namespace more_numbers_with_middle_digit_smaller_l673_673410

def is_valid_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def middle_digit_greater (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  b > a ∧ b > c

def middle_digit_smaller (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  b < a ∧ b < c

theorem more_numbers_with_middle_digit_smaller :
  ∃ more :
    ∃ total_small total_great : ℕ,
      total_small > total_great ∧
      total_small = nat.card {n // is_valid_three_digit_number n ∧ middle_digit_smaller n} ∧
      total_great = nat.card {n // is_valid_three_digit_number n ∧ middle_digit_greater n} :=
sorry

end more_numbers_with_middle_digit_smaller_l673_673410


namespace min_value_expr_l673_673893

theorem min_value_expr (x y : ℝ) (hx : x > 1) (hy : y > 1) : 
  (x^2 / (y - 1) + y^2 / (x - 1) ≥ 8) := 
begin
  sorry
end

end min_value_expr_l673_673893


namespace problem1_problem2_l673_673848

theorem problem1 : sqrt 12 - sqrt 3 + 3 * sqrt (1/3) = sqrt 3 + 3 := by
  sorry

theorem problem2 : sqrt 18 / sqrt 6 * sqrt 3 = 3 := by
  sorry

end problem1_problem2_l673_673848


namespace smallest_possible_S_l673_673770

/-- Define the maximum possible sum for n dice --/
def max_sum (n : ℕ) : ℕ := 6 * n

/-- Define the transformation of the dice sum when each result is transformed to 7 - d_i --/
def transformed_sum (n R : ℕ) : ℕ := 7 * n - R

/-- Determine the smallest possible S under given conditions --/
theorem smallest_possible_S :
  ∃ n : ℕ, max_sum n ≥ 2001 ∧ transformed_sum n 2001 = 337 :=
by
  -- TODO: Complete the proof
  sorry

end smallest_possible_S_l673_673770


namespace blood_expires_jan5_11pm_l673_673403

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n + 1) * factorial n

theorem blood_expires_jan5_11pm (donation_time : ℕ := 12 * 3600) :
  let expiry_time_in_seconds := factorial 8 in
  let seconds_in_day := 86400 in
  let expiry_time := donation_time + expiry_time_in_seconds in
  let hours_past_noon := (expiry_time % seconds_in_day) / 3600 in
  expiry_time = 43200 + 40320 →
  hours_past_noon = 23 :=
by
  unfold factorial
  have factorial_8_value : factorial 8 = 40320 := rfl
  have day_seconds : seconds_in_day = 86400 := rfl
  have donation_duration := donation_time + factorial_8_value
  show donation_duration = 43200 + 40320 → ((donation_duration % 86400) / 3600) = 23
  sorry

end blood_expires_jan5_11pm_l673_673403


namespace liked_product_B_l673_673811

-- Define the conditions as assumptions
variables (X : ℝ)

-- Assumptions
axiom liked_both : 23 = 23
axiom liked_neither : 23 = 23

-- The main theorem that needs to be proven
theorem liked_product_B (X : ℝ) : ∃ Y : ℝ, Y = 100 - X :=
by sorry

end liked_product_B_l673_673811


namespace percent_alcohol_new_solution_l673_673371

theorem percent_alcohol_new_solution :
  let original_volume := 40
  let original_percent_alcohol := 5
  let added_alcohol := 2.5
  let added_water := 7.5
  let original_alcohol := original_volume * (original_percent_alcohol / 100)
  let total_alcohol := original_alcohol + added_alcohol
  let new_total_volume := original_volume + added_alcohol + added_water
  (total_alcohol / new_total_volume) * 100 = 9 :=
by
  sorry

end percent_alcohol_new_solution_l673_673371


namespace sides_of_base_of_prism_are_correct_l673_673172

variable (a : ℝ)

-- Define the conditions
def is_equilateral_triangle_base (ABC : Triangle) : Prop :=
  ABC.is_equilateral

def lateral_edges_are_equal (AA1 BB1 CC1 : ℝ) : Prop :=
  AA1 = 1 ∧ BB1 = 1 ∧ CC1 = 1

def sphere_touches_planes (radius a : ℝ) : Prop :=
  radius = a

def extensions_segments_touch (A1 B1 C1 : Point) (AB1 BC1 CA1 : Segment) : Prop :=
  -- This will hold the complex geometric relationship but simplifying for the context
  true

-- The equivalent proof problem
theorem sides_of_base_of_prism_are_correct :
  ∀ (ABC : Triangle) (AA1 BB1 CC1 : ℝ) (A1 B1 C1 : Point) (AB1 BC1 CA1 : Segment),
    is_equilateral_triangle_base ABC →
    lateral_edges_are_equal AA1 BB1 CC1 →
    sphere_touches_planes (norm A1 B1 CC1) a →
    extensions_segments_touch A1 B1 C1 AB1 BC1 CA1 →
    a = Real.sqrt 44 - 6 :=
by
  intros
  sorry  -- Proof goes here

end sides_of_base_of_prism_are_correct_l673_673172


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673312

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673312


namespace a_n_formula_b_n_formula_c_n_formula_l673_673637

def S_n (n : ℕ) : ℕ := n^2 - n + 1

def a_n (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2 * (n - 1)

def b_n (n : ℕ) : ℕ :=
  2 ^ (n - 1)

def T_n (n : ℕ) : ℕ :=
  (2 ^ n) - 1

def c_n (n : ℕ) : ℕ :=
  2 ^ n - n

-- Statements to prove
theorem a_n_formula (n : ℕ) : a_n n = if n = 1 then 1 else 2 * (n - 1) := sorry

theorem b_n_formula (n : ℕ) (h₁ : b_n 2 = a_n 2) (h₂ : b_n 4 = a_n 5) : b_n n = 2 ^ (n - 1) := sorry

theorem c_n_formula (n : ℕ) (h₁ : c_n 1 = a_n 1) (h₂ : ∀ k, 1 ≤ k → c_n k = c_{n + 1} - T_n n) : 
  c_n n = 2 ^ n - n := sorry

end a_n_formula_b_n_formula_c_n_formula_l673_673637


namespace find_expression_for_f_l673_673974

variables {a b : ℝ} {f : ℝ → ℝ}

-- Define the function and conditions
def quadratic_function (x : ℝ) : ℝ :=
  (x + a) * (b * x + 2 * a)

theorem find_expression_for_f :
  quadratic_function = f →
  ∀ x : ℝ, (∀ y : ℝ, f (-y) = f y) →
  (∀ z : ℝ, f z ≤ 4) →
  f = λ x, -2 * x^2 + 4 :=
by
  sorry

end find_expression_for_f_l673_673974


namespace number_of_pastries_left_to_take_home_l673_673460

def Wendy_baked_4_cupcakes : ℕ := 4
def Wendy_baked_29_cookies : ℕ := 29
def Wendy_sold_9_pastries : ℕ := 9

theorem number_of_pastries_left_to_take_home
  (baked_cupcakes : ℕ)
  (baked_cookies : ℕ)
  (sold_pastries : ℕ) :
  baked_cupcakes = 4 →
  baked_cookies = 29 →
  sold_pastries = 9 →
  (baked_cupcakes + baked_cookies) - sold_pastries = 24 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end number_of_pastries_left_to_take_home_l673_673460


namespace minimum_value_expr_minimum_value_attained_l673_673139

open Real

theorem minimum_value_expr (x : ℝ) (h : 0 < x) : 3 * x ^ 7 + 6 * x ^ -5 ≥ 9 :=
by sorry

theorem minimum_value_attained : 3 * 1 ^ 7 + 6 * 1 ^ -5 = 9 :=
by norm_num

end minimum_value_expr_minimum_value_attained_l673_673139


namespace car_speed_in_kmph_l673_673359

def speed_mps : ℝ := 10  -- The speed of the car in meters per second
def conversion_factor : ℝ := 3.6  -- The conversion factor from m/s to km/h

theorem car_speed_in_kmph : speed_mps * conversion_factor = 36 := 
by
  sorry

end car_speed_in_kmph_l673_673359


namespace pants_after_5_years_l673_673872

theorem pants_after_5_years (initial_pants : ℕ) (pants_per_year : ℕ) (years : ℕ) :
  initial_pants = 50 → pants_per_year = 8 → years = 5 → (initial_pants + pants_per_year * years) = 90 :=
by
  intros initial_cond pants_per_year_cond years_cond
  rw [initial_cond, pants_per_year_cond, years_cond]
  norm_num
  done

end pants_after_5_years_l673_673872


namespace integral_value_l673_673174

theorem integral_value (a : ℝ) (h : a = 2) : ∫ x in a..2*Real.exp 1, 1/x = 1 := by
  sorry

end integral_value_l673_673174


namespace abs_b_lt_abs_a_lt_2abs_b_l673_673162

variable {a b : ℝ}

theorem abs_b_lt_abs_a_lt_2abs_b (h : (6 * a + 9 * b) / (a + b) < (4 * a - b) / (a - b)) :
  |b| < |a| ∧ |a| < 2 * |b| :=
sorry

end abs_b_lt_abs_a_lt_2abs_b_l673_673162


namespace right_triangle_shorter_leg_l673_673564

theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : c = 65) : a = 25 ∨ b = 25 := 
by
  sorry

end right_triangle_shorter_leg_l673_673564


namespace range_of_m_l673_673014

theorem range_of_m (m : ℝ) : 
  let P := (-1, 1)
      Q := (2, 2)
      l : ℝ × ℝ → Prop := fun (x, y) => x + m * y + m = 0
  in
    (∀ x y, l (x, y) → ¬ (min (-1) 2 ≤ x ∧ x ≤ max (-1) 2 ∧ 
                                min 1 2 ≤ y ∧ y ≤ max 1 2)) 
    ↔ (m < (-2)/3 ∨ m > 1/2) :=
by intros; sorry

end range_of_m_l673_673014


namespace problem_1_problem_2_l673_673039

def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

theorem problem_1 : {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
  sorry

theorem problem_2 (m : ℝ) : (∃ x : ℝ, f x ≥ x^2 - x + m) → m ≤ 5/4 :=
  sorry

end problem_1_problem_2_l673_673039


namespace largest_even_number_from_powerful_digits_set_l673_673535

def is_powerful_number (n : ℕ) : Prop :=
  (n + (n + 1) + (n + 2)) < 100 && 
  -- Adding further constraints to ensure no carry over
  (n % 10 + (n + 1) % 10 + (n + 2) % 10 < 10) && 
  (n / 10 % 10 + (n + 1) / 10 % 10 + (n + 2) / 10 % 10 < 10) && 
  (n / 100 % 10 + (n + 1) / 100 % 10 + (n + 2) / 100 % 10 < 10)

def powerful_number_digits (n : ℕ) : Finset ℕ :=
  if is_powerful_number n then
    (n.digits.to_finset ∪ (n + 1).digits.to_finset ∪ (n + 2).digits.to_finset)
  else
    ∅

def powerful_digits_set : Finset ℕ :=
  Finset.bUnion (Finset.range 1000) powerful_number_digits

def largest_even_number : ℕ :=
  (4 :: 3 :: 2 :: 1 :: 0 :: []).foldl (λ acc d, 10 * acc + d) 0

theorem largest_even_number_from_powerful_digits_set : 
  largest_even_number = 43210 :=
sorry

end largest_even_number_from_powerful_digits_set_l673_673535


namespace not_divisibility_rule_for_base_12_l673_673536

def decimal_sum (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n+1), a i * 10^i

def duodecimal_sum (b : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n+1), b i * 12^i

theorem not_divisibility_rule_for_base_12 :
  ∀ (b : ℕ → ℕ) (n : ℕ), 
  (duodecimal_sum b n % 3 = 0) ↔ (∑ i in finset.range (n+1), b i % 3 = 0) → false :=
by sorry

end not_divisibility_rule_for_base_12_l673_673536


namespace maximize_profit_l673_673817

-- Define the relationships and constants
def P (x : ℝ) : ℝ := -750 * x + 15000
def material_cost_per_unit : ℝ := 4
def fixed_cost : ℝ := 7000

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - material_cost_per_unit) * P x - fixed_cost

-- The statement of the problem, proving the maximization condition
theorem maximize_profit :
  ∃ x : ℝ, x = 12 ∧ profit 12 = 41000 := by
  sorry

end maximize_profit_l673_673817


namespace sum_of_squares_divisible_by_three_l673_673077

theorem sum_of_squares_divisible_by_three {a b : ℤ} 
  (h : 3 ∣ (a^2 + b^2)) : (3 ∣ a ∧ 3 ∣ b) :=
by 
  sorry

end sum_of_squares_divisible_by_three_l673_673077


namespace find_number_of_packages_l673_673347

def packages (p t : ℕ) : ℕ := t / p

theorem find_number_of_packages (p t : ℕ) (h1 : p = 6) (h2 : t = 426) : packages p t = 71 := by
  rw [h1, h2]
  simp only [packages, nat.div]
  sorry

end find_number_of_packages_l673_673347


namespace transform_equation_l673_673072

open Real

theorem transform_equation (m : ℝ) (x : ℝ) (h1 : x^2 + 4 * x = m) (h2 : (x + 2)^2 = 5) : m = 1 := by
  sorry

end transform_equation_l673_673072


namespace ma_result_l673_673106

variables (A B C M : Point)
variables (AB BC : ℝ)
variable (R : ℝ)
variable (perpendicular_bisector : M ∈ perpendicular_bisector AB)
variable (perpendicular_AM_AC : perpendicular (line_from_to A M) (line_from_to A C))
variable (circumradius : circumradius_of_triangle A B C = 9)

def find_MA : ℝ :=
  if h₁ : AB = 4
  ∧ h₂ : BC = 6
  ∧ h₃ : M ∈ perpendicular_bisector AB
  ∧ h₄ : perpendicular (line_from_to A M) (line_from_to A C)
  ∧ circumradius_of_triangle A B C = 9 then
    6
  else
    sorry

theorem ma_result :
  find_MA A B C M AB BC R perpendicular_bisector perpendicular_AM_AC circumradius = 6 :=
by sorry

end ma_result_l673_673106


namespace children_tickets_sold_l673_673348

-- Given conditions
variables (A C : ℕ) -- A represents the number of adult tickets, C the number of children tickets.
variables (total_money total_tickets price_adult price_children : ℕ)
variables (total_money_eq : total_money = 104)
variables (total_tickets_eq : total_tickets = 21)
variables (price_adult_eq : price_adult = 6)
variables (price_children_eq : price_children = 4)
variables (money_eq : price_adult * A + price_children * C = total_money)
variables (tickets_eq : A + C = total_tickets)

-- Problem statement: prove that C = 11
theorem children_tickets_sold : C = 11 :=
by
  -- Necessary Lean code to handle proof here (omitting proof details as instructed)
  sorry

end children_tickets_sold_l673_673348


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673212

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673212


namespace units_digit_17_pow_39_l673_673765

theorem units_digit_17_pow_39 : 
  ∃ d : ℕ, d < 10 ∧ (17^39 % 10 = d) ∧ d = 3 :=
by
  sorry

end units_digit_17_pow_39_l673_673765


namespace kim_shirts_left_l673_673123

theorem kim_shirts_left (initial_dozens : ℕ) (fraction_given : ℚ) (num_pairs : ℕ)
  (h1 : initial_dozens = 4) 
  (h2 : fraction_given = 1 / 3)
  (h3 : num_pairs = initial_dozens * 12)
  (h4 : num_pairs * fraction_given  = (16 : ℕ)):
  48 - ((num_pairs * fraction_given).toNat) = 32 :=
by 
  sorry

end kim_shirts_left_l673_673123


namespace parallel_lines_a_values_l673_673499

theorem parallel_lines_a_values (a : Real) : 
  (∃ k : Real, 2 = k * a ∧ -a = k * (-8)) ↔ (a = 4 ∨ a = -4) := sorry

end parallel_lines_a_values_l673_673499


namespace product_of_roots_l673_673896

open Polynomial

-- Define the first polynomial
def poly1 : Polynomial ℝ := 3 * X^4 + 2 * X^3 - 8 * X + 15

-- Define the second polynomial
def poly2 : Polynomial ℝ := 4 * X^3 - 20 * X^2 + 25

-- Define the product of the polynomials
def poly_product : Polynomial ℝ := poly1 * poly2

-- State the degree of the polynomial
def poly_degree : ℕ := 7

-- State the leading coefficient of the polynomial
def leading_coefficient : ℝ := 12

-- State the constant term of the polynomial
def constant_term : ℝ := 375

-- The product of the roots
theorem product_of_roots : (∏ r in (poly_product.roots), r) = -125 / 4 := by
  sorry

end product_of_roots_l673_673896


namespace total_surface_area_pyramid_l673_673397

theorem total_surface_area_pyramid 
(distance_to_apex : ℝ) 
(angle_between_face_and_base : ℝ) 
:= 
(let a := distance_to_apex in
 let α := angle_between_face_and_base in
 8 * a^2 * Real.cos α * (Real.cos (α / 2))^2 * (Real.cot (α / 2))^2) = 
8 * distance_to_apex^2 * Real.cos angle_between_face_and_base * 
(Real.cos (angle_between_face_and_base / 2))^2 * (Real.cot (angle_between_face_and_base / 2))^2 :=
sorry

end total_surface_area_pyramid_l673_673397


namespace rounding_estimate_lt_exact_l673_673670

variable (a b c a' b' c' : ℕ)

theorem rounding_estimate_lt_exact (ha : a' ≤ a) (hb : b' ≥ b) (hc : c' ≤ c) (hb_pos : b > 0) (hb'_pos : b' > 0) :
  (a':ℚ) / (b':ℚ) + (c':ℚ) < (a:ℚ) / (b:ℚ) + (c:ℚ) :=
sorry

end rounding_estimate_lt_exact_l673_673670


namespace find_special_two_digit_number_l673_673884

theorem find_special_two_digit_number :
  ∃ (A B : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ A ≠ B ∧ (10 * A + B = 27 ∧ (10 * A + B) ^ 2 = (A + B) ^ 3) :=
by 
  have A := 2
  have B := 7
  use A, B
  have H1 : 1 ≤ A := sorry
  have H2 : A ≤ 9 := sorry
  have H3 : 0 ≤ B := sorry
  have H4 : B ≤ 9 := sorry
  have H5 : A ≠ B := sorry
  have H6 : 10 * A + B = 27 := sorry
  have H7 : (10 * A + B ) ^ 2 = (A + B ) ^ 3 := sorry
  exact ⟨A, B, H1, H2, H3, H4, H5, ⟨H6, H7⟩⟩

end find_special_two_digit_number_l673_673884


namespace max_z_under_D_le_1_l673_673623

noncomputable def f (x a b : ℝ) : ℝ := x - a * x^2 + b
noncomputable def f0 (x b0 : ℝ) : ℝ := x^2 + b0
noncomputable def g (x a b b0 : ℝ) : ℝ := f x a b - f0 x b0

theorem max_z_under_D_le_1 
  (a b b0 : ℝ) (D : ℝ)
  (h_a : a = 0) 
  (h_b0 : b0 = 0) 
  (h_D : D ≤ 1)
  (h_maxD : ∀ x : ℝ, - (Real.pi / 2) ≤ x ∧ x ≤ Real.pi / 2 → g (Real.sin x) a b b0 ≤ D) :
  ∃ z : ℝ, z = b - a^2 / 4 ∧ z = 1 :=
by
  sorry

end max_z_under_D_le_1_l673_673623


namespace at_least_30_cents_prob_l673_673682

def coin := {penny, nickel, dime, quarter, half_dollar}
def value (c : coin) : ℕ := 
  match c with
  | penny => 1
  | nickel => 5
  | dime => 10
  | quarter => 25
  | half_dollar => 50

def coin_positions : List (coin × Bool) := 
  [(penny, true), (nickel, true), (dime, true), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, true), (dime, true), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, true), (dime, true), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, true), (dime, false), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, true), (dime, false), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, true), (dime, false), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, true), (dime, false), (quarter, false), (half_dollar, false),
   (penny, true), (nickel, false), (dime, true), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, false), (dime, true), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, false), (dime, true), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, false), (dime, true), (quarter, false), (half_dollar, false),
   (penny, true), (nickel, false), (dime, false), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, false), (dime, false), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, false), (dime, false), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, false), (dime, false), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, true), (dime, true), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, true), (dime, true), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, true), (dime, true), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, true), (dime, true), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, true), (dime, false), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, true), (dime, false), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, true), (dime, false), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, true), (dime, false), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, false), (dime, true), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, false), (dime, true), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, false), (dime, true), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, false), (dime, true), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, false), (dime, false), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, false), (dime, false), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, false), (dime, false), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, false), (dime, false), (quarter, false), (half_dollar, false)]

def count_successful_outcomes : ℕ :=
  List.length (List.filter (λ positions, List.foldl (λ acc (c, h) => if h then acc + value c else acc) 0 positions >= 30) coin_positions)

def total_outcomes : ℕ := 32

def probability_of_success : ℚ :=
  ⟨count_successful_outcomes, total_outcomes⟩

theorem at_least_30_cents_prob : probability_of_success = 3 / 4 :=
by sorry

end at_least_30_cents_prob_l673_673682


namespace zombies_count_decrease_l673_673728

theorem zombies_count_decrease (z : ℕ) (d : ℕ) : z = 480 → (∀ n, d = 2^n * z) → ∃ t, d / t < 50 :=
by
  intros hz hdz
  let initial_count := 480
  have := 480 / (2 ^ 4)
  sorry

end zombies_count_decrease_l673_673728


namespace midpoint_distance_inequality_l673_673048

variables {A B C D E M : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E] [AddGroup M]
variables {distance : A × B × C → ℝ}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space M]

-- Definitions of points and midpoints
def is_midpoint (P Q R : Type) [AddGroup P] [AddGroup Q] [AddGroup R] :=
  distance (P, Q, R) = distance (P, Q, P) + distance (P, R, R)

-- Conditions: Midpoints of sides AB and BC, and M on AC such that ME > EC
variables (A B C D E M : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space M]
variables [is_midpoint A B D] [is_midpoint B C E] 
variables (AC : Set A) [distance AC M > distance AC E]

-- Statement of the problem
theorem midpoint_distance_inequality :
  distance (M, D) < distance (A, D) :=
sorry

end midpoint_distance_inequality_l673_673048


namespace zombies_count_decrease_l673_673727

theorem zombies_count_decrease (z : ℕ) (d : ℕ) : z = 480 → (∀ n, d = 2^n * z) → ∃ t, d / t < 50 :=
by
  intros hz hdz
  let initial_count := 480
  have := 480 / (2 ^ 4)
  sorry

end zombies_count_decrease_l673_673727


namespace richard_more_pins_than_patrick_l673_673097

theorem richard_more_pins_than_patrick :
  let P1 := 70 in
  let R1 := P1 + 15 in
  let P2 := R1 * 2 in
  let R2 := P2 - 3 in
  let Patrick_total := P1 + P2 in
  let Richard_total := R1 + R2 in
  Richard_total - Patrick_total = 12 :=
by
  let P1 := 70
  let R1 := P1 + 15
  let P2 := R1 * 2
  let R2 := P2 - 3
  let Patrick_total := P1 + P2
  let Richard_total := R1 + R2
  have h1 : Patrick_total = 240 := by
    calc
      Patrick_total = P1 + P2 := rfl
      ... = 70 + 170 := by
        let P1 := 70
        let R1 := P1 + 15
        let P2 := R1 * 2
        rfl
      ... = 240 := by norm_num
  have h2 : Richard_total = 252 := by
    calc
      Richard_total = R1 + R2 := rfl
      ... = 85 + 167 := by
        let R1 := P1 + 15
        let P2 := R1 * 2
        let R2 := P2 - 3
        rfl
      ... = 252 := by norm_num
  calc
    Richard_total - Patrick_total = 252 - 240 := by rw [h2, h1]
    ... = 12 := by norm_num

end richard_more_pins_than_patrick_l673_673097


namespace germs_total_l673_673094

theorem germs_total (dishes germs_per_dish : ℝ) (h₁ : dishes = 36000 * 10^(-3))
  (h₂ : germs_per_dish = 99.99999999999999) : dishes * germs_per_dish = 3600 :=
by
  sorry

end germs_total_l673_673094


namespace find_cost_price_l673_673064

-- Definitions based on conditions
def cost_price (C : ℝ) : Prop := 0.05 * C = 10

-- The theorem stating the problem to be proven
theorem find_cost_price (C : ℝ) (h : cost_price C) : C = 200 :=
by
  sorry

end find_cost_price_l673_673064


namespace james_paid_per_shirt_after_discount_l673_673110

variable (original_price num_shirts discount : ℝ)
variable (num_shirts_pos : num_shirts > 0)

def discounted_price_per_shirt : ℝ :=
  let discount_amount := (discount / 100) * original_price
  let sale_price := original_price - discount_amount
  sale_price / num_shirts

theorem james_paid_per_shirt_after_discount 
  (original_price : ℝ) 
  (num_shirts : ℝ) 
  (discount : ℝ) 
  (num_shirts_pos : num_shirts > 0) 
  (h : original_price = 60) 
  (h' : num_shirts = 3) 
  (h'' : discount = 40) : 
  discounted_price_per_shirt original_price num_shirts discount = 12 := 
by 
  unfold discounted_price_per_shirt 
  rw [h, h', h'']
  norm_num
  sorry

end james_paid_per_shirt_after_discount_l673_673110


namespace shorter_leg_of_right_triangle_l673_673580

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l673_673580


namespace largest_divisor_of_product_of_5_consecutive_integers_l673_673234

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l673_673234


namespace william_napkins_l673_673647

-- Define the given conditions
variables (O A C G W : ℕ)
variables (ho: O = 10)
variables (ha: A = 2 * O)
variables (hc: C = A / 2)
variables (hg: G = 3 * C)
variables (hw: W = 15)

-- Prove the total number of napkins William has now
theorem william_napkins (O A C G W : ℕ) (ho: O = 10) (ha: A = 2 * O)
  (hc: C = A / 2) (hg: G = 3 * C) (hw: W = 15) : W + (O + A + C + G) = 85 :=
by {
  sorry
}

end william_napkins_l673_673647


namespace base_9_first_digit_l673_673173

def convert_to_base_10 (s : List ℕ) (b : ℕ) : ℕ :=
  s.foldr (λ (a : ℕ) (acc : ℕ) => a + acc * b) 0

def first_digit_base (n : ℕ) (b : ℕ) : ℕ :=
  let rec helper (m : ℕ) : ℕ :=
    if m < b then m else helper (m / b)
  helper n

theorem base_9_first_digit (y_base3 : ℕ) (y_base9_first_digit : ℕ) :
  y_base3 = 1122001122 ∧ y_base9_first_digit = 5 → first_digit_base (convert_to_base_10 [1, 1, 2, 2, 0, 0, 1, 1, 2, 2] 3) 9 = 5 :=
by
  intros h
  sorry

end base_9_first_digit_l673_673173


namespace exist_pairing_odd_sum_not_exist_pairing_even_sum_l673_673790

/-- 
Given 100 points labeled from 1 to 100 arranged in an arbitrary order on a circle, 
this theorem proves that there exists a pairing of these points such that:
1. The line segments connecting the pairs do not intersect.
2. The sums of the numbers of each pair are odd.
--/
theorem exist_pairing_odd_sum (points : Finset ℕ) (h100 : points.card = 100) 
  (point_labels : ℕ → ℕ) (hlabels : ∀ x ∈ points, point_labels x ∈ (Finset.range 101)) 
  (pairing : Finset (ℕ × ℕ)) 
  (hpairing : pairing.card = 50)
  (hodd_walk : ∀ p ∈ pairing, (point_labels p.1 + point_labels p.2) % 2 = 1) : 
  ∃ non_intersecting_pairing : Finset (ℕ × ℕ), 
    (∀ p ∈ non_intersecting_pairing, p ∈ pairing) ∧
    (∀ a b c d, (a, b) ∈ non_intersecting_pairing → 
                (c, d) ∈ non_intersecting_pairing → 
                (a = c ∧ b = d ∨ (not (SegIntersect a b c d)))), sorry

/-- 
Given 100 points labeled from 1 to 100 arranged in an arbitrary order on a circle, 
this theorem proves that it is impossible to pair these points such that:
1. The line segments connecting the pairs do not intersect.
2. The sums of the numbers of each pair are even.
--/
theorem not_exist_pairing_even_sum (points : Finset ℕ) (h100 : points.card = 100) 
  (point_labels : ℕ → ℕ) (hlabels : ∀ x ∈ points, point_labels x ∈ (Finset.range 101)) 
  (pairing : Finset (ℕ × ℕ)) 
  (hpairing : pairing.card = 50) :
  ¬(∃ non_intersecting_pairing : Finset (ℕ × ℕ),
    (∀ p ∈ non_intersecting_pairing, p ∈ pairing) ∧
    (∀ a b c d, (a, b) ∈ non_intersecting_pairing → 
                (c, d) ∈ non_intersecting_pairing → 
                (a = c ∧ b = d ∨ (not (SegIntersect a b c d)))) ∧
    (∀ p ∈ non_intersecting_pairing, (point_labels p.1 + point_labels p.2) % 2 = 0)), sorry

end exist_pairing_odd_sum_not_exist_pairing_even_sum_l673_673790


namespace largest_divisor_of_product_of_5_consecutive_integers_l673_673237

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l673_673237


namespace eccentricity_of_ellipse_slope_of_line_OQ_l673_673508

-- Define the conditions of the given problem
variable (a b : ℝ)
variable (h1 : a > b)
variable (h2 : b > 0)
variable (P : ℝ × ℝ)
variable (hP : P = (a * sqrt 5 / 5, a * sqrt 2 / 2))
variable (Q : ℝ × ℝ)
variable (hQ1 : ((Q.1 ^ 2) / (a ^ 2)) + ((Q.2 ^ 2) / (b ^ 2)) = 1)
variable (A : ℝ × ℝ)
variable (hA : A = (-a, 0))
variable (O : ℝ × ℝ)
variable (hO : O = (0, 0))
variable (hAQ_AO : dist A Q = dist A O)

-- Prove that the eccentricity is sqrt(6)/4
theorem eccentricity_of_ellipse : (sqrt (1 - (b ^ 2) / (a ^ 2))) = sqrt 6 / 4 := sorry

-- Prove that the slope of line OQ is ±sqrt(5)
theorem slope_of_line_OQ : 
  (∃ k : ℝ, k = sqrt 5 ∨ k = -sqrt 5) :=
sorry

end eccentricity_of_ellipse_slope_of_line_OQ_l673_673508


namespace identity_function_uniq_l673_673437

theorem identity_function_uniq (f g h : ℝ → ℝ)
    (hg : ∀ x, g x = x + 1)
    (hh : ∀ x, h x = x^2)
    (H1 : ∀ x, f (g x) = g (f x))
    (H2 : ∀ x, f (h x) = h (f x)) :
  ∀ x, f x = x :=
by
  sorry

end identity_function_uniq_l673_673437


namespace smallest_composite_no_prime_factors_below_15_correct_l673_673933

def smallest_composite_no_prime_factors_below_15 : Nat :=
  323
  
theorem smallest_composite_no_prime_factors_below_15_correct :
  (∀ n < 15, Prime n → ¬ (n ∣ smallest_composite_no_prime_factors_below_15)) ∧
  (∃ p q, Prime p ∧ Prime q ∧ p ≠ q ∧ smallest_composite_no_prime_factors_below_15 = p * q) :=
by
  -- Proof skipped
  sorry

end smallest_composite_no_prime_factors_below_15_correct_l673_673933


namespace age_ratio_l673_673352

theorem age_ratio (A B C : ℕ) (h1 : A = B + 2) (h2 : A + B + C = 27) (h3 : B = 10) : B / C = 2 :=
by
  sorry

end age_ratio_l673_673352


namespace slower_speed_7_l673_673394

theorem slower_speed_7.5 (time_at_faster_speed : ℝ)
  (faster_speed : ℝ := 15)
  (additional_distance : ℝ := 20)
  (total_distance : ℝ := 40) :
  let time := total_distance / faster_speed in
  let slower_distance := total_distance - additional_distance in
  let slower_speed := slower_distance / time in
  slower_speed = 7.5 := by
  sorry

end slower_speed_7_l673_673394


namespace orthocenter_locus_thm_l673_673006

noncomputable def distance (x y : ℝ) := abs (x - y)

structure Point := 
(x : ℝ)
(y : ℝ)

structure Circle := 
(center : Point)
(radius : ℝ)

def diameter (c : Circle) : set (Point × Point) :=
{ p | (distance p.1.x p.2.x) ^ 2 + (distance p.1.y p.2.y) ^ 2 = 4 * (c.radius)^2 }

def orthocenter_locus (A B C : Point) (O : Point) (D : Point) (c : ℝ) : Prop :=
let M := Point in
(M.x - D.x) * (C.x - O.x) + (M.y - D.y) * (C.y - O.y) = 0 

theorem orthocenter_locus_thm (c : Circle) (C O D A B : Point) (C_in_plane : True)
(h1 : A ≠ B)
(h2 : (A, B) ∈ diameter c)
(h3 : distance C O = c)
(h4 : distance O D = (c.radius ^ 2) / c)
: ∃ M : Point, orthocenter_locus A B C O D c :=
sorry

end orthocenter_locus_thm_l673_673006


namespace part1_part2_l673_673017

-- Definitions for Part (1)
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }
def B (m : ℝ) : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- Part (1) Statement
theorem part1 (m : ℝ) (hm : m = 2) : A ∩ ((compl B m)) = {x | (-2 ≤ x ∧ x < -1) ∨ (3 < x ∧ x ≤ 4)} := 
by
  sorry

-- Definitions for Part (2)
def B_interval (m : ℝ) : Set ℝ := { x | (1 - m) ≤ x ∧ x ≤ (1 + m) }

-- Part (2) Statement
theorem part2 (m : ℝ) (h : ∀ x, (x ∈ A → x ∈ B_interval m)) : 0 < m ∧ m < 3 := 
by
  sorry

end part1_part2_l673_673017


namespace remainder_of_sum_is_zero_l673_673764

theorem remainder_of_sum_is_zero :
  let S := 1001 + 1003 + 1005 + 1007 + 1009 + 1011 + 1013 + 1015
  in S % 16 = 0 :=
by
  let S := 1001 + 1003 + 1005 + 1007 + 1009 + 1011 + 1013 + 1015
  have h : S = 64 := by
    sorry
  show S % 16 = 0, by
    rw [h]
    exact nat.mod_eq_zero_of_dvd (dvd_of_mod_eq_zero (nat.zero_mod 16))

end remainder_of_sum_is_zero_l673_673764


namespace right_triangle_shorter_leg_l673_673570

theorem right_triangle_shorter_leg :
  ∃ (a b : ℤ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 16 :=
by
  sorry

end right_triangle_shorter_leg_l673_673570


namespace max_value_m_l673_673636

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem max_value_m {a b c : ℝ} (h₀ : a ≠ 0)
  (h₁ : ∀ x, quadratic_function a b c (x-4) = quadratic_function a b c (2-x))
  (h₂ : ∀ x, 0 < x ∧ x < 2 → quadratic_function a b c x ≤ ((x + 1) / 2)^2)
  (h₃ : ∀ x, quadratic_function a b c x ≥ 0 ∧ ∃ y, quadratic_function a b c y = 0) :
  ∃ m > 1, ∃ t ∈ ℝ, ∀ x ∈ set.Icc 1 m, quadratic_function a b c (x + t) ≤ x :=
begin
  let a := 1 / 4,
  let b := 1 / 2,
  let c := 1 / 4,
  let f := quadratic_function a b c,
  existsi 9,
  split,
  { norm_num },
  { use -4,
    split,
    { norm_num },
    { intros x hx,
      rw [quadratic_function, quadratic_function],
      simp [a, b, c, f],
      -- Adding further details of the proof to complete it.
      sorry
    }
  }
end

end max_value_m_l673_673636


namespace line_equation_with_slope_angle_135_and_y_intercept_neg1_l673_673890

theorem line_equation_with_slope_angle_135_and_y_intercept_neg1 :
  ∃ k b : ℝ, k = -1 ∧ b = -1 ∧ ∀ x y : ℝ, y = k * x + b ↔ y = -x - 1 :=
by
  sorry

end line_equation_with_slope_angle_135_and_y_intercept_neg1_l673_673890


namespace domain_range_of_p_l673_673388

variable (h : ℝ → ℝ)
variable (h_domain : ∀ x, -1 ≤ x ∧ x ≤ 3)
variable (h_range : ∀ x, 0 ≤ h x ∧ h x ≤ 2)

def p (x : ℝ) : ℝ := 2 - h (x - 1)

theorem domain_range_of_p :
  (∀ x, 0 ≤ x ∧ x ≤ 4) ∧ (∀ y, 0 ≤ y ∧ y ≤ 2) :=
by
  -- Proof to show that the domain of p(x) is [0, 4] and the range is [0, 2]
  sorry

end domain_range_of_p_l673_673388


namespace sum_of_exponents_500_l673_673533

theorem sum_of_exponents_500 : 
  ∃ (S : finset ℕ), 
  (∀ n ∈ S, (∃ k : ℕ, n = 2^k)) ∧ -- All elements are powers of 2
  (S.sum (λ n, n) = 500) ∧          -- Their sum is 500
  (finset.card S ≥ 2) ∧             -- At least two distinct elements
  (S.sum (λ n, log2 n) = 32) :=     -- Sum of exponents is 32
sorry

end sum_of_exponents_500_l673_673533


namespace basketball_two_out_of_three_success_l673_673373

noncomputable def basketball_shot_probability (success_rate : ℚ) (trials success_goal : ℕ) : ℚ :=
  ∑ i in (finset.range (trials + 1)).filter (λ x, x = success_goal), 
    nat.choose trials i * (success_rate ^ i) * ((1 - success_rate) ^ (trials - i))

theorem basketball_two_out_of_three_success :
    basketball_shot_probability (3/5) 3 2 = 54 / 125 := 
by
  sorry

end basketball_two_out_of_three_success_l673_673373


namespace tangent_lines_parallel_to_l_l673_673469

theorem tangent_lines_parallel_to_l 
  (l : ∀ x y : ℝ, x + 2 * y - 9 = 0)
  (O : ∀ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 5) :
  ∃ c : ℝ, (c = 8 ∨ c = -2) ∧ ∀ x y : ℝ, x + 2 * y + c = 0 :=
begin
  sorry
end

end tangent_lines_parallel_to_l_l673_673469


namespace find_b_l673_673671

theorem find_b (b : ℤ) (h₀ : 0 ≤ b ∧ b ≤ 20)
  (h₁ : (1:ℤ) - b = 0 ∨ ∃ k, (74639281:ℤ) * (85:ℤ)^0 - b = 17 * k) : b = 1 :=
begin
  sorry
end

end find_b_l673_673671


namespace all_white_after_n_steps_l673_673673

-- Definitions
def grid (n : ℕ) : Type := ℕ → ℕ → bool  -- Represents the grid with n grey squares

-- Evolution rule: function that updates the grid based on the previous state
def evolve (G : grid n) (i j : ℕ) : bool :=
  let grey_count := (if G i j then 1 else 0) + (if G (i-1) j then 1 else 0) + (if G i (j-1) then 1 else 0)
  in grey_count = 2 ∨ grey_count = 3

-- Proving that after at most n steps, all squares in the grid will be white
theorem all_white_after_n_steps (G : grid n) :
  ∃ (k : ℕ), k ≤ n ∧ (∀ i j, evolve (iterate evolve k G) i j = false) :=
sorry

end all_white_after_n_steps_l673_673673


namespace part_I_part_II_part_III_l673_673131

noncomputable def seq_sum : (ℕ → ℚ) → ℕ → ℚ
| a, 0       => 0
| a, (n + 1) => a (n + 1) + seq_sum a n

def A_n (S : ℕ → ℚ) (i : ℕ → ℕ) (n : ℕ) : ℚ :=
  S (i (n + 1)) - S (i n)

def Omega (S : ℕ → ℚ) : Set ℕ :=
  {j : ℕ | ∀ k, k > j → S k - S j ≥ 0}

def sgn (x : ℚ) : ℤ := if x > 0 then 1 else if x = 0 then 0 else -1

theorem part_I (a : ℕ → ℕ)
  (S : ℕ → ℚ)
  (i : ℕ → ℕ)
  (h₁ : ∀ n, S n = seq_sum a n)
  (h₂ : ∀ n, a n = n)
  (h₃ : ∀ n, i n = n^2) :
  (A_n S i 1 = 9) ∧ (A_n S i 2 = 35) := sorry

theorem part_II (a : ℕ → ℚ) 
  (S : ℕ → ℚ)
  (h₁ : ∀ n, S n = seq_sum a n)
  (h₂ : ∀ n, a n = (-1/2)^(n-1)) :
  Omega S = {x : ℕ | ∃ m, x = 2 * m + 2} := sorry

theorem part_III (a : ℕ → ℚ) 
  (S : ℕ → ℚ) :
  ∃ i : ℕ → ℕ, ∀ n, sgn (A_n S i n) = sgn (A_n S i 0) := sorry

end part_I_part_II_part_III_l673_673131


namespace smallest_composite_proof_l673_673904

-- Define what it means for a number not to have prime factors less than 15
def no_prime_factors_less_than_15 (n : ℕ) : Prop :=
  ∀ p : ℕ, nat.prime p → p ∣ n → p ≥ 15

-- Define what it means for a number to be the smallest composite number with the above property
def smallest_composite_without_prime_factors_less_than_15 (n : ℕ) : Prop :=
  nat.composite n ∧ no_prime_factors_less_than_15 n ∧
  ∀ m : ℕ, nat.composite m → no_prime_factors_less_than_15 m → n ≤ m

theorem smallest_composite_proof : smallest_composite_without_prime_factors_less_than_15 323 :=
  sorry

end smallest_composite_proof_l673_673904


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673213

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673213


namespace range_of_a_l673_673521

variable {ℝ : Type*} [LinearOrder ℝ] [TopologicalSpace ℝ] [HasContinuousMul ℝ] [HasContinuousAdd ℝ] [HasOne ℝ] [HasZero ℝ]
variables {x a : ℝ}

def p (a : ℝ) : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → (1 / 2 * x ^ 2 - Real.log x - a) ≥ 0

def q (a : ℝ) : Prop :=
  ∃ x, x^2 + 2 * a * x - 8 - 6 * a = 0

theorem range_of_a (h : p a ∧ q a) : a ∈ Set.Iic (1 / 2) :=
by sorry

end range_of_a_l673_673521


namespace suitable_sampling_survey_l673_673773

-- Definitions of the options
def option_A : Prop := ∀ (p : Passenger), security_check_before_boarding (p)
def option_B : Prop := survey_vision_students_grade8_class1
def option_C : Prop := survey_average_daily_water_consumption_city
def option_D : Prop := survey_sleep_time_20_centenarians_county

-- The target proposition that defines the most suitable survey for sampling
def most_suitable_for_sampling (opt : Prop) : Prop :=
  opt = option_C

-- Showing that option C is the most suitable for sampling
theorem suitable_sampling_survey : most_suitable_for_sampling option_C :=
by 
  -- here the proof will be provided
  sorry

end suitable_sampling_survey_l673_673773


namespace largest_integer_dividing_consecutive_product_l673_673246

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l673_673246


namespace total_profit_l673_673805

def purchase_price_A := 40
def selling_price_A := 55
def purchase_price_B := 28
def selling_price_B := 40
def total_items := 80
def items_A (x : ℕ) := x

def profit_per_item_A := selling_price_A - purchase_price_A
def profit_per_item_B := selling_price_B - purchase_price_B

theorem total_profit (x : ℕ) : 15 * x + 12 * (80 - x) = 3 * x + 960 := by
  let profit_A := profit_per_item_A * x
  let profit_B := profit_per_item_B * (total_items - x)
  calc
    profit_A + profit_B
      = 15 * x + 12 * (80 - x) : by rfl
  ... = 3 * x + 960 : by sorry

end total_profit_l673_673805


namespace expression_evaluation_l673_673846

-- Define the numbers and operations
def expr : ℚ := 10 * (1 / 2) * 3 / (1 / 6)

-- Formalize the proof problem
theorem expression_evaluation : expr = 90 := 
by 
  -- Start the proof, which is not required according to the instruction, so we replace it with 'sorry'
  sorry

end expression_evaluation_l673_673846


namespace simplify_and_evaluate_l673_673166

noncomputable def a : ℝ := Real.sqrt 2 - 2

def expression (a : ℝ) : ℝ :=
  (a^2) / (a^2 + 2 * a) - ((a^2 - 2 * a + 1) / (a + 2)) / ((a^2 - 1) / (a + 1))

theorem simplify_and_evaluate :
  expression a = Real.sqrt 2 / 2 :=
by
  have : a = Real.sqrt 2 - 2 := rfl
  have : expression a = (a^2) / (a^2 + 2 * a) - ((a^2 - 2 * a + 1) / (a + 2)) / ((a^2 - 1) / (a + 1)) := rfl
  sorry -- Include the steps to simplify and evaluate here.

end simplify_and_evaluate_l673_673166


namespace simplify_add_complex_eq_l673_673165

noncomputable def simplify_and_add_complex : ℂ :=
  let result1 := (3 + 5 * complex.I) / (-2 + 3 * complex.I)
  let result2 := result1 + (1 - 2 * complex.I)
  in result2

theorem simplify_add_complex_eq :
  simplify_and_add_complex = - (8/13 : ℝ) - (45/13 : ℝ) * complex.I :=
by
  sorry

end simplify_add_complex_eq_l673_673165


namespace area_of_hex_ok_l673_673143

noncomputable def area_of_an_equilateral_triangle
  (a : ℝ) : ℝ :=
( a^2 * (Real.sqrt 3) / 4)

noncomputable def radius_of_circumcircle_of_equilateral_triangle
  (a : ℝ) : ℝ :=
  a/(Real.sqrt 3)

noncomputable def area_of_hexagon_in_circle 
: ℝ :=
let a := 6,
R := radius_of_circumcircle_of_equilateral_triangle a,
in
  (846 * Real.sqrt 3) / 49

theorem area_of_hex_ok :
let a := 6,
ABC_triangle := a,
circumcircle := radius_of_circumcircle_of_equilateral_triangle ABC_triangle,
circumcircle_area := circumcircle * 2 * Real.pi,
a1a2_area := area_of_an_equilateral_triangle a ,
hex_area := area_of_hexagon_in_circle == ( Real.sqrt 3 )/4 
hex_area :=
 ∃ (a : ℝ), 
 let abc_triangle := a ≠ ∅  --> ∀ 
 hex_area == (846 * Real.sqrt 3)/49
:= 
begin 
sorry,
end

end area_of_hex_ok_l673_673143


namespace arithmetic_sequence_a4_is_5_l673_673589

variable (a : ℕ → ℕ)

-- Arithmetic sequence property
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n m k : ℕ, n < m ∧ m < k → 2 * a m = a n + a k

-- Given condition
axiom sum_third_and_fifth : a 3 + a 5 = 10

-- Prove that a_4 = 5
theorem arithmetic_sequence_a4_is_5
  (h : is_arithmetic_sequence a) : a 4 = 5 := by
  sorry

end arithmetic_sequence_a4_is_5_l673_673589


namespace largest_divisor_of_5_consecutive_integers_l673_673326

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l673_673326


namespace henry_walks_distance_l673_673052

noncomputable def gym_distance : ℝ := 3

noncomputable def walk_factor : ℝ := 2 / 3

noncomputable def c_limit_position : ℝ := 1.5

noncomputable def d_limit_position : ℝ := 2.5

theorem henry_walks_distance :
  abs (c_limit_position - d_limit_position) = 1 := by
  sorry

end henry_walks_distance_l673_673052


namespace integer_solution_count_l673_673957

theorem integer_solution_count :
  {n : ℤ | (sqrt (n + 1 : ℚ) ≤ sqrt (5 * n - 7 : ℚ)) ∧ (sqrt (5 * n - 7 : ℚ) < sqrt (3 * n + 6 : ℚ))}.card = 5 := 
by
  -- Proof intentionally omitted
  sorry

end integer_solution_count_l673_673957


namespace rachel_minimum_age_l673_673660

-- Definitions for the conditions
def rachel_has_three_children (ages : List ℕ) : Prop := ages.length = 3
def each_child_at_least_two (ages : List ℕ) : Prop := ∀ age ∈ ages, age ≥ 2
def pairwise_relatively_prime (ages : List ℕ) : Prop := ∀ i j, i < ages.length → j < ages.length → i ≠ j → Nat.gcd (ages.nth i).getOrElse 0 (ages.nth j).getOrElse 0 = 1
def rachel_age_multiple_of_children (ages : List ℕ) (rachel_age : ℕ) : Prop := ∀ age ∈ ages, rachel_age % age = 0

-- The problem statement
theorem rachel_minimum_age (ages : List ℕ) (rachel_age : ℕ) :
  rachel_has_three_children ages →
  each_child_at_least_two ages →
  pairwise_relatively_prime ages →
  rachel_age_multiple_of_children ages rachel_age →
  rachel_age = 30 :=
by
  sorry

end rachel_minimum_age_l673_673660


namespace correct_proposition_l673_673981

open Classical

variables (p q : Prop)
noncomputable def p : Prop := ∀ x : ℝ, 0 < x → x + 1 / 2 > 2
noncomputable def q : Prop := ∃ x : ℝ, 2 ^ x < 0

theorem correct_proposition : ¬ p ∧ q := by {
    sorry
}

end correct_proposition_l673_673981


namespace Rectangle_Q_coordinates_l673_673712

section Rectangle

def Point (α : Type) := (x : α) × (y : α)

variables {α : Type} [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Neg α] [DecidableEq α]
variables (O P Q R : Point α)

def is_rectangle (O P Q R : Point α) : Prop :=
  (O.1 = P.1) ∧ (O.2 = R.2) ∧ (P.2 = Q.2) ∧ (R.1 = Q.1) ∧ ((P.2 - O.2) = (Q.2 - R.2)) ∧ ((R.1 - O.1) = (Q.1 - P.1))

theorem Rectangle_Q_coordinates (O P Q R : Point ℚ) 
    (hO : O = (0, 0)) 
    (hP : P = (0, 3)) 
    (hR : R = (5, 0))
    (hRect : is_rectangle O P Q R) :
  Q = (5, 3) :=
sorry

end Rectangle

end Rectangle_Q_coordinates_l673_673712


namespace ben_can_reach_2020_l673_673392

-- Define the operations
def step1 (n : Nat) : Nat := n + 3
def step2 (n : Nat) : Nat := n - 2

-- Define the reachability predicate
def reach (n m : Nat) : Prop :=
  ∃ (k : Nat), 
  (nat.iterate step1 (k / 2) n = m ∨
   nat.iterate step2 (k / 2 + k % 2) n = m)

-- Statement
theorem ben_can_reach_2020 : ∀ n : Nat, reach n 2020 :=
  by sorry

end ben_can_reach_2020_l673_673392


namespace largest_divisor_of_consecutive_five_l673_673297

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l673_673297


namespace smallest_composite_no_prime_factors_lt_15_l673_673908

theorem smallest_composite_no_prime_factors_lt_15 (n : ℕ) :
  ∀ n, (∀ p : ℕ, p.prime → p ∣ n → 15 ≤ p) → n = 289 → 
       is_composite n ∧ (∀ m : ℕ, (∀ q : ℕ, q.prime → q ∣ m → 15 ≤ q) → m ≥ 289) :=
by
  intros n hv hn
  -- Proof would go here
  sorry

end smallest_composite_no_prime_factors_lt_15_l673_673908


namespace largest_integer_dividing_consecutive_product_l673_673244

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l673_673244


namespace product_of_five_consecutive_integers_divisible_by_240_l673_673285

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l673_673285


namespace find_m_l673_673705

noncomputable def f (x m : ℝ) : ℝ := x^3 - 3 * x^2 + m

theorem find_m (m : ℝ) (h : ∀ x ∈ set.Icc (-1:ℝ) (1:ℝ), f x m ≤ 2) : m = 2 :=
begin
  -- The proof would proceed here.
  sorry
end

end find_m_l673_673705


namespace find_angle_CAB_l673_673615

variable (A B C H M : Type)
variable [add_group A] [add_group B] [add_group C] [add_group H] [add_group M]
variable (CH CM : ℝ) (right_angle_ABC : A)
variable (is_midpoint_M : Prop) (is_foot_H : Prop)
variable (CH_eq_one : CH = 1) (CM_eq_two : CM = 2)

theorem find_angle_CAB 
  (right_triangle : ∀ A B C, right_angle_ABC = 90)
  (midpoint_M : is_midpoint_M)
  (foot_H : is_foot_H)
  (lengths : CH_eq_one ∧ CM_eq_two) :
  angle CAB = 15 ∨ angle CAB = 75 := sorry

end find_angle_CAB_l673_673615


namespace largest_number_l673_673342

def A : ℚ := 97 / 100
def B : ℚ := 979 / 1000
def C : ℚ := 9709 / 10000
def D : ℚ := 907 / 1000
def E : ℚ := 9089 / 10000

theorem largest_number : B > A ∧ B > C ∧ B > D ∧ B > E := by
  sorry

end largest_number_l673_673342


namespace find_D_l673_673652

-- Definitions of points and conditions
structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := -2, y := 1 }
def Q : Point := { x := 4, y := 9 }

-- The distance condition for point D on segment PQ
def twice_as_far (P Q D : Point) : Prop :=
  dist P D = 2 * dist D Q

-- Calculate distances
def dist (A B : Point) : ℝ :=
  ((A.x - B.x)^2 + (A.y - B.y)^2)^(1 / 2) 

-- Statement: Given the conditions mentioned, prove coordinates of D
theorem find_D (D : Point) (h : twice_as_far P Q D) : D = { x := 2.5, y := 7 } :=
by
  sorry

end find_D_l673_673652


namespace triangle_trig_identity_l673_673479

theorem triangle_trig_identity 
  (A B C : ℝ)
  (hcosA_sinB : cos A = sin B)
  (hcosA_tanC2 : cos A = 2 * tan (C / 2))
  (htriangle_sum : A + B + C = π) :
  sin A + cos A + 2 * tan A = 2 :=
sorry

end triangle_trig_identity_l673_673479


namespace find_omega_l673_673036

def f (ω x : ℝ) : ℝ := Real.sin (ω * x - Real.pi / 3)

theorem find_omega (m n ω : ℝ) (h₁ : |n| ≠ 1) 
                   (h₂ : f ω m = n) 
                   (h₃ : f ω (m + Real.pi) = n)
                   (h₄ : ∀ x : ℝ, (∃ y : ℝ, f ω y = x) → set.count ({x | f ω x = x} ∩ (set.Icc m (m + Real.pi))) = 5) : 
                   ω = 4 := 
by
  sorry

end find_omega_l673_673036


namespace factor_tree_value_l673_673551

theorem factor_tree_value :
  let F := 7 * 3,
      G := 11 * 3,
      Y := 7 * F,
      Z := 11 * G,
      X := Y * Z
  in X = 53361 :=
by
  let F := 7 * 3
  let G := 11 * 3
  let Y := 7 * F
  let Z := 11 * G
  let X := Y * Z
  sorry

end factor_tree_value_l673_673551


namespace gary_total_money_l673_673462

theorem gary_total_money (initial_money : ℝ) (snake_sale : ℝ) : initial_money = 73.0 ∧ snake_sale = 55.0 → initial_money + snake_sale = 128.0 :=
by
  intros h,
  cases h with h_initial h_sale,
  rw [h_initial, h_sale],
  norm_num,
  sorry

end gary_total_money_l673_673462


namespace company_picnic_attendance_l673_673356

theorem company_picnic_attendance 
  (total_employees men women : ℕ)
  (men_fraction : ℚ) (men_attend_fraction women_attend_fraction : ℚ)
  (h_men_fraction : men_fraction = 0.35)
  (h_total_employees : total_employees = 100)
  (h_men : men = (men_fraction * total_employees).toNat)
  (h_women : women = total_employees - men)
  (h_men_attend : men_attend_fraction = 0.20)
  (h_women_attend : women_attend_fraction = 0.40)
  (men_attend women_attend total_attend : ℕ)
  (h_men_attend_calc : men_attend = (h_men_attend * men).toNat)
  (h_women_attend_calc : women_attend = (h_women_attend * women).toNat)
  (h_total_attend_calc : total_attend = men_attend + women_attend)
  (final_percentage_attended : ℚ)
  (h_final_percentage_attended : final_percentage_attended = (total_attend * 100) / total_employees) :
  final_percentage_attended = 33 := 
sorry

end company_picnic_attendance_l673_673356


namespace distance_tangent_to_circumcenter_of_triangle_l673_673854

open EuclideanGeometry

variables (Point : Type) [MetricSpace Point]
variables (A B C D O circumcenter : Point)
variables (Gamma : Circle point radius : ℝ)
variables (line_AB line_AD line_CD : Line Point)

-- Definitions and conditions from problem statement
def diameter_def (A B : Point) (d : ℝ) : Prop := dist A B = d
def radius_def (Gamma : Circle Point) (A B : Point) : Prop := Gamma.radius = dist A B / 2
def on_line_DEF (l : Line Point) (A B : Point) : Prop := A ∈ l ∧ B ∈ l
def tangent_to_circle (Gamma : Circle Point) (l : Line Point) (P : Point) : Prop := l.isTangentToCircleAt Gamma P
def equal_distance (A B C : Point) : Prop := dist A B = dist B C ∧ A ≠ C

-- Stating the final theorem
theorem distance_tangent_to_circumcenter_of_triangle :
    (diameter_def A B 6) → 
    (radius_def Gamma A B) →
    (on_line_DEF line_AB A B) →
    (equal_distance A B C) →
    (tangent_to_circle Gamma line_CD D) →
    (distance_from_line_to_point line_AD circumcenter) = 4 * real.sqrt 3 :=
by sorry

end distance_tangent_to_circumcenter_of_triangle_l673_673854


namespace no_10_neg_n_sum_of_reciprocals_of_factorials_l673_673656

theorem no_10_neg_n_sum_of_reciprocals_of_factorials :
  ∀ (n : ℕ), n ≥ 1 → ¬∃ (a : ℕ → ℕ), (∀ i j, i ≠ j → a i ≠ a j) ∧ 
  (∃ S : set ℕ, S ≠ ∅ ∧ 
  ∑ i in S, 1 / (a i)! = 10 ^ (- (n : ℤ))) :=
by sorry

end no_10_neg_n_sum_of_reciprocals_of_factorials_l673_673656


namespace reflections_circumcircle_tangent_l673_673132

theorem reflections_circumcircle_tangent
  (ABC : Triangle)
  (O: Circle)
  (t: Tangent_to O)
  (ta tb tc: Line)
  (h_ta: ta = reflection t (side BC ABC))
  (h_tb: tb = reflection t (side CA ABC))
  (h_tc: tc = reflection t (side AB ABC))
  :
  tangent (circumcircle (triangle_of_reflections ta tb tc)) O :=
sorry

end reflections_circumcircle_tangent_l673_673132


namespace smallest_composite_no_prime_factors_less_than_15_l673_673917

-- Definitions used in the conditions
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

-- Prime numbers less than 15
def primes_less_than_15 (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13

-- Define the main proof statement
theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n : ℕ, is_composite n ∧ (∀ p : ℕ, p ∣ n → is_prime p → primes_less_than_15 p → false) ∧ n = 289 :=
by
  -- leave the proof as a placeholder
  sorry

end smallest_composite_no_prime_factors_less_than_15_l673_673917


namespace product_of_five_consecutive_divisible_by_30_l673_673272

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l673_673272


namespace megatek_graph_is_pie_chart_l673_673687

theorem megatek_graph_is_pie_chart :
  ∃ graph : Type, 
    (∀ d, proportional_size_to_percent d graph) → 
    (sector_angle manufacturing graph = 144) → 
    (percentage_employees manufacturing = 0.4) → 
    (graph = "pie chart") :=
by
  -- proof goes here
  sorry

namespace Megatek
variables (graph : Type) (d : Department)

def proportional_size_to_percent := ∀ (d : Department), 
  (sector_size d graph / (2 * Real.pi) = percent_employees d)

noncomputable def sector_angle := 144 -- degrees
def percentage_of_employees := 0.4 -- 40%

end Megatek

end megatek_graph_is_pie_chart_l673_673687


namespace tolu_pencils_l673_673669

theorem tolu_pencils (price_per_pencil : ℝ) (robert_pencils : ℕ) (melissa_pencils : ℕ) (total_money_spent : ℝ) (tolu_pencils : ℕ) :
  price_per_pencil = 0.20 →
  robert_pencils = 5 →
  melissa_pencils = 2 →
  total_money_spent = 2.00 →
  tolu_pencils * price_per_pencil = 2.00 - (5 * 0.20 + 2 * 0.20) →
  tolu_pencils = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end tolu_pencils_l673_673669


namespace combined_map_correct_l673_673082

def X3 := ℝ × ℝ × ℝ

def A_map (x : X3) : X3 := 
  (x.1 + x.2 - x.3, x.2 + x.3, x.3)

def B_map (x : X3) : X3 := 
  (x.2 + 2 * x.3, -x.1, x.2)

def combined_map (x : X3) : X3 :=
  let A := λ y : X3, A_map y
  let B := λ y : X3, B_map y
  let C := λ y : X3, (2 : ℝ) • (A y) + (A (B y))
  C x

theorem combined_map_correct (x : X3) :
  combined_map x = 
    (x.1 + 2 * x.2, 
     -x.1 + 3 * x.2 + 2 * x.3, 
     x.2 + 2 * x.3) :=
by {
  -- Proof will be constructed here
  sorry
}

end combined_map_correct_l673_673082


namespace smallest_composite_no_prime_factors_below_15_correct_l673_673929

def smallest_composite_no_prime_factors_below_15 : Nat :=
  323
  
theorem smallest_composite_no_prime_factors_below_15_correct :
  (∀ n < 15, Prime n → ¬ (n ∣ smallest_composite_no_prime_factors_below_15)) ∧
  (∃ p q, Prime p ∧ Prime q ∧ p ≠ q ∧ smallest_composite_no_prime_factors_below_15 = p * q) :=
by
  -- Proof skipped
  sorry

end smallest_composite_no_prime_factors_below_15_correct_l673_673929


namespace probability_obese_employee_is_male_l673_673431

theorem probability_obese_employee_is_male 
  (BMI : Real) 
  (obese_male_proportion : Real := 1 / 5) 
  (obese_female_proportion : Real := 1 / 10)
  (male_to_female_ratio : Real := 3 / 2) : 
  (Probability : Real) :=
  let proportion_male := 3 / 5
  let proportion_female := 2 / 5
  let obese_male_probability := proportion_male * obese_male_proportion
  let obese_female_probability := proportion_female * obese_female_proportion
  let probability_obese_employee := obese_male_probability + obese_female_probability
  let result := obese_male_probability / probability_obese_employee
  result = 3 / 4 :=
sorry

end probability_obese_employee_is_male_l673_673431


namespace triangle_isosceles_inscribed_tangents_l673_673584

noncomputable def l_value (BAC ABC ACB D: ℝ): Prop :=
  BAC == (2 / 5) * Math.pi

theorem triangle_isosceles_inscribed_tangents 
  (ABC ACB D : ℝ)
  (h1 : ABC = 3 * D)
  (h2 : ACB = 3 * D)
  (h3 : BAC = 2 * Math.pi / 5) :
  l_value BAC ABC ACB D :=
by
  simp [BAC]
  sorry

end triangle_isosceles_inscribed_tangents_l673_673584


namespace find_cost_price_l673_673065

-- Definitions based on conditions
def cost_price (C : ℝ) : Prop := 0.05 * C = 10

-- The theorem stating the problem to be proven
theorem find_cost_price (C : ℝ) (h : cost_price C) : C = 200 :=
by
  sorry

end find_cost_price_l673_673065


namespace part_a_part_b_l673_673791

theorem part_a (points : List ℕ) (h_length : points.length = 100) (h_nodup : points.nodup) (h_sum_1_100 : points.sum = (100 * 101) / 2) :
  ∃ (pairs : List (ℕ × ℕ)), (∀ (p : ℕ × ℕ) (hp : p ∈ pairs), ((p.1 ∈ points) ∧ (p.2 ∈ points))) ∧
                        (pairs.length = 50) ∧
                        (∀ (p : ℕ × ℕ) (hp : p ∈ pairs), (p.1 + p.2) % 2 = 1) ∧
                        (∀ (p1 p2 : ℕ × ℕ) (hp1 : p1 ∈ pairs) (hp2 : p2 ∈ pairs) (ht : p1 ≠ p2),
                          ¬ (segments_intersect p1 p2)) :=
sorry

theorem part_b (points : List ℕ) (h_length : points.length = 100) (h_nodup : points.nodup) (h_sum_1_100 : points.sum = (100 * 101) / 2) :
  ¬ ∃ (pairs : List (ℕ × ℕ)), (∀ (p : ℕ × ℕ) (hp : p ∈ pairs), ((p.1 ∈ points) ∧ (p.2 ∈ points))) ∧
                         (pairs.length = 50) ∧
                         (∀ (p : ℕ × ℕ) (hp : p ∈ pairs), (p.1 + p.2) % 2 = 0) ∧
                         (∀ (p1 p2 : ℕ × ℕ) (hp1 : p1 ∈ pairs) (hp2 : p2 ∈ pairs) (ht : p1 ≠ p2),
                           ¬ (segments_intersect p1 p2)) :=
sorry

end part_a_part_b_l673_673791


namespace goldbach_conjecture_2024_l673_673429

-- Definitions for the problem
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Lean 4 statement for the proof problem
theorem goldbach_conjecture_2024 :
  is_even 2024 ∧ 2024 > 2 → ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 2024 = p1 + p2 :=
by
  sorry

end goldbach_conjecture_2024_l673_673429


namespace other_train_length_l673_673370

-- Define a theorem to prove that the length of the other train (L) is 413.95 meters
theorem other_train_length (length_first_train : ℝ) (speed_first_train_kmph : ℝ) 
                           (speed_second_train_kmph: ℝ) (time_crossing_seconds : ℝ) : 
                           length_first_train = 350 → 
                           speed_first_train_kmph = 150 →
                           speed_second_train_kmph = 100 →
                           time_crossing_seconds = 11 →
                           ∃ (L : ℝ), L = 413.95 :=
by
  intros h1 h2 h3 h4
  sorry

end other_train_length_l673_673370


namespace find_circular_permutations_l673_673895

def alpha := (1 + Real.sqrt 5) / 2
def beta := (1 - Real.sqrt 5) / 2
def fib : ℕ → ℝ
| 0 := 0
| 1 := 1
| (n + 2) := fib n + fib (n + 1)

def b_n (n : ℕ) : ℝ := alpha^n + beta^n + 2

theorem find_circular_permutations (n : ℕ) : b_n n = alpha^n + beta^n + 2 :=
sorry

end find_circular_permutations_l673_673895


namespace true_equality_is_B_l673_673339

theorem true_equality_is_B :
  (4 / 1 ≠ 1.4) ∧
  (5 / 2 = 2.5) ∧
  (6 / 3 ≠ 3.6) ∧
  (7 / 4 ≠ 4.7) ∧
  (8 / 5 ≠ 5.8) :=
by {
  split;
  norm_num; sorry
}

end true_equality_is_B_l673_673339


namespace find_x_l673_673058

theorem find_x (x : ℕ) : {2, 3, 4} = {2, x, 3} → x = 4 := by
  sorry

end find_x_l673_673058


namespace problem1_part1_problem1_part2_l673_673003

theorem problem1_part1 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a + b + c) * (a^2 + b^2 + c^2) ≤ 3 * (a^3 + b^3 + c^3) := 
sorry

theorem problem1_part2 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 3 / 2 := 
sorry

end problem1_part1_problem1_part2_l673_673003


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673313

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673313


namespace combined_area_parallelogram_triangle_l673_673827

theorem combined_area_parallelogram_triangle {A B C D E F : Type} (angle_DAB : ℝ) (side_AD : ℝ) (side_AB : ℝ) (side_BE : ℝ) :
  angle_DAB = 150 ∧ side_AD = 10 ∧ side_AB = 24 ∧ side_BE = 10 →
  let area_parallelogram := side_AD * (side_AD * Real.sin (angle_DAB * Real.pi / 180)) in
  let area_triangle := 0.5 * side_AB * side_BE in
  area_parallelogram + area_triangle = 170 :=
by
  sorry

end combined_area_parallelogram_triangle_l673_673827


namespace amount_of_first_alloy_used_is_15_l673_673091

-- Definitions of percentages and weights
def chromium_percentage_first_alloy : ℝ := 0.12
def chromium_percentage_second_alloy : ℝ := 0.08
def weight_second_alloy : ℝ := 40
def chromium_percentage_new_alloy : ℝ := 0.0909090909090909
def total_weight_new_alloy (x : ℝ) : ℝ := x + weight_second_alloy
def chromium_content_first_alloy (x : ℝ) : ℝ := chromium_percentage_first_alloy * x
def chromium_content_second_alloy : ℝ := chromium_percentage_second_alloy * weight_second_alloy
def total_chromium_content (x : ℝ) : ℝ := chromium_content_first_alloy x + chromium_content_second_alloy

-- The proof problem
theorem amount_of_first_alloy_used_is_15 :
  ∃ x : ℝ, total_chromium_content x = chromium_percentage_new_alloy * total_weight_new_alloy x ∧ x = 15 :=
by
  sorry

end amount_of_first_alloy_used_is_15_l673_673091


namespace common_speed_is_10_l673_673601

noncomputable def speed_jack (x : ℝ) : ℝ := x^2 - 11 * x - 22
noncomputable def speed_jill (x : ℝ) : ℝ := 
  if x = -6 then 0 else (x^2 - 4 * x - 12) / (x + 6)

theorem common_speed_is_10 (x : ℝ) (h : speed_jack x = speed_jill x) (hx : x = 16) : 
  speed_jack x = 10 :=
by
  sorry

end common_speed_is_10_l673_673601


namespace proof_b_a_c_l673_673541

def is_even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_monotonically_decreasing_on (f : ℝ → ℝ) (s : set ℝ) : Prop := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f y ≤ f x

def a (f : ℝ → ℝ) : ℝ := f (Real.log 3 / Real.log 2)
def b (f : ℝ → ℝ) : ℝ := f (Real.log 5 / Real.log 4)
def c (f : ℝ → ℝ) : ℝ := f (Real.sqrt 2)

theorem proof_b_a_c (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_mono_dec : is_monotonically_decreasing_on f (set.Iic 0)) : 
  b f < a f ∧ a f < c f := 
by
  sorry

end proof_b_a_c_l673_673541


namespace smallest_composite_no_prime_factors_less_than_15_l673_673922

theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n, (n = 289) ∧ (n > 1) ∧ (¬ Nat.Prime n) ∧ (∀ p : ℕ, Nat.Prime p → p ∣ n → 15 ≤ p) :=
by
  use 289
  split
  case left => rfl
  case right =>
    split
    case left => exact Nat.lt_succ_self 288
    case right =>
      split
      case left =>
        have composite : ¬ Nat.Prime 289 := by
          intro h
          have h_div : 17 ∣ 289 := by norm_num
          exact h.not_divs_self (dec_trivial : 17 * 17 = 289)
        exact composite
      case right =>
        intros p h_prime h_div
        have : p ∣ 17 := by
          have factorization : 289 = 17 * 17 := by norm_num
          have dvd_product : p ∣ 289 := by { use 17, exact factorization.symm }
          exact Nat.Prime.dvd_mul h_prime dvd_product
        have prime_eq_17 : p = 17 := by
          exact Nat.Prime.eq_of_dvd_of_ne h_prime (by norm_num) this
        linarith

end smallest_composite_no_prime_factors_less_than_15_l673_673922


namespace commissions_shared_members_l673_673395

theorem commissions_shared_members 
  (n : ℕ) 
  (h_pos : 0 < n) 
  (commissions : Fin 6 → Finset (Fin n))
  (h_comm_size : ∀ i, (commissions i).card ≥ n / 4) :
  ∃ i j, i ≠ j ∧ (commissions i ∩ commissions j).card ≥ n / 30 :=
begin
  sorry
end

end commissions_shared_members_l673_673395


namespace girls_more_than_boys_l673_673189

variables (B G : ℕ)
def ratio_condition : Prop := 3 * G = 4 * B
def total_students_condition : Prop := B + G = 49

theorem girls_more_than_boys
  (h1 : ratio_condition B G)
  (h2 : total_students_condition B G) :
  G = B + 7 :=
sorry

end girls_more_than_boys_l673_673189


namespace min_value_of_f_l673_673020

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 2^x

theorem min_value_of_f (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
(h3 : ∃ x ∈ Icc (0:ℝ) (1:ℝ), f a b x = 4) : 
  f a b (-1) = -3/2 :=
by
  -- We start by noting from h3 that there must exist a maximum value of f(x) = 4 in the interval [0,1]
  -- Given f(x) = ax^3 + bx + 2^x, if we consider particularly x = 1 we have:
  -- f(a, b, 1) = a*1^3 + b*1 + 2^1 
  --             = a + b + 2
  -- Since this has been stated to be equal to 4
  -- Hence, a + b + 2 = 4
  
  -- From this equation, we solve:
  -- a + b = 2
  
  -- We need to determine the minimum value of f at x = -1
  -- Thus, f(a, b, -1) = a*(-1)^3 + b*(-1) + 2^(-1)
  --                   = -a - b + 1/2

  -- Next considering a + b = 2 derived from our initial condition
  -- -a - b can be rewritten using a + b = 2
  -- Hence, the final calculation:
  -- f(a, b, -1) = -2 + 1/2
  -- Therefore:
  -- f(a, b, -1) = -3/2
  sorry

end min_value_of_f_l673_673020


namespace number_of_valid_x_l673_673775

def volume (x : ℕ) : ℤ := (x^2 + 5) * (2 * x - 5) * (x + 25)

theorem number_of_valid_x :
  {x : ℕ // 0 < volume x ∧ volume x < 1200}.card = 1 :=
sorry

end number_of_valid_x_l673_673775


namespace pants_after_5_years_l673_673871

theorem pants_after_5_years (initial_pants : ℕ) (pants_per_year : ℕ) (years : ℕ) :
  initial_pants = 50 → pants_per_year = 8 → years = 5 → (initial_pants + pants_per_year * years) = 90 :=
by
  intros initial_cond pants_per_year_cond years_cond
  rw [initial_cond, pants_per_year_cond, years_cond]
  norm_num
  done

end pants_after_5_years_l673_673871


namespace trajectory_of_midpoint_l673_673153

theorem trajectory_of_midpoint {x y : ℝ} :
  (∃ Mx My : ℝ, (Mx + 3)^2 + My^2 = 4 ∧ (2 * x - 3 = Mx) ∧ (2 * y = My)) →
  x^2 + y^2 = 1 :=
by
  intro h
  sorry

end trajectory_of_midpoint_l673_673153


namespace find_cost_price_l673_673066

-- Definitions based on conditions
def cost_price (C : ℝ) : Prop := 0.05 * C = 10

-- The theorem stating the problem to be proven
theorem find_cost_price (C : ℝ) (h : cost_price C) : C = 200 :=
by
  sorry

end find_cost_price_l673_673066


namespace ratio_b_to_c_l673_673350

-- Define the ages of a, b, and c as A, B, and C respectively
variables (A B C : ℕ)

-- Given conditions
def condition1 := A = B + 2
def condition2 := B = 10
def condition3 := A + B + C = 27

-- The question: Prove the ratio of b's age to c's age is 2:1
theorem ratio_b_to_c : condition1 ∧ condition2 ∧ condition3 → B / C = 2 := 
by
  sorry

end ratio_b_to_c_l673_673350


namespace complex_statements_correct_l673_673785

theorem complex_statements_correct :
  (∀ z : ℂ, z * conj z = 0 → z = 0) ∧
  (∀ z : ℂ, z^2 = 3 + 4 * complex.I → 
    (∃ (a b : ℝ), z = a + b * complex.I ∧ 
    (a^2 - b^2 = 3 ∧ 2 * a * b = 4) ∧ (a > 0 ∧ b > 0 ∨ a < 0 ∧ b < 0))) :=
by
  sorry

end complex_statements_correct_l673_673785


namespace smallest_composite_no_prime_under_15_correct_l673_673936

-- Define the concept of a composite number
def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

-- Define the concept of having no prime factors less than 15
def has_no_prime_factors_less_than_15 (n : ℕ) : Prop := 
  ∀ p : ℕ, p.prime ∧ p ∣ n → 15 ≤ p

-- Define the smallest composite number with no prime factors less than 15 
def smallest_composite_no_prime_under_15 : ℕ := 289

-- Prove that this is the smallest number satisfying our conditions
theorem smallest_composite_no_prime_under_15_correct : 
  is_composite smallest_composite_no_prime_under_15 ∧
  has_no_prime_factors_less_than_15 smallest_composite_no_prime_under_15 ∧
  ∀ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than_15 n → n ≥ smallest_composite_no_prime_under_15 :=
by 
  sorry

end smallest_composite_no_prime_under_15_correct_l673_673936


namespace rotation_creates_cone_l673_673662

-- Define the conditions and problem statement in Lean 4

def right_angle_triangle_rotation_result : Type :=
  { t : Type // is_right_angle_triangle t }


def rotation_of_triangle_results_in_cone (t : right_angle_triangle_rotation_result)
(using line : Line): Prop :=
  rotated_solid t = Cone

theorem rotation_creates_cone (t : right_angle_triangle_rotation_result) :
  ∃ (line : Line), rotation_of_triangle_results_in_cone t line :=
sorry

end rotation_creates_cone_l673_673662


namespace smallest_YZ_minus_XY_l673_673746

noncomputable def triangle_inequality (a b c : ℕ) : Prop :=
a + b > c ∧ a + c > b ∧ b + c > a

theorem smallest_YZ_minus_XY :
  ∃ (XY XZ YZ : ℕ),
    (XY < XZ) ∧
    (XZ < YZ) ∧
    (XY + XZ + YZ = 3021) ∧
    triangle_inequality XY XZ YZ ∧
    (YZ - XY = 2) :=
begin
  sorry
end

end smallest_YZ_minus_XY_l673_673746


namespace smallest_constant_term_l673_673864

theorem smallest_constant_term (a b c d e : ℤ) (h_poly : Polynomial ℤ) :
  (h_poly.map (λ x, x + 3)).roots = [-3, 4, 7, -1/2] →
  h_poly.coeffs = [a, b, c, d, e] →
  ∃ e, e = 168 := 
sorry

end smallest_constant_term_l673_673864


namespace shorter_leg_of_right_triangle_l673_673574

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h: a^2 + b^2 = c^2) (hn: c = 65): a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l673_673574


namespace find_special_two_digit_number_l673_673883

theorem find_special_two_digit_number :
  ∃ (A B : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ A ≠ B ∧ (10 * A + B = 27 ∧ (10 * A + B) ^ 2 = (A + B) ^ 3) :=
by 
  have A := 2
  have B := 7
  use A, B
  have H1 : 1 ≤ A := sorry
  have H2 : A ≤ 9 := sorry
  have H3 : 0 ≤ B := sorry
  have H4 : B ≤ 9 := sorry
  have H5 : A ≠ B := sorry
  have H6 : 10 * A + B = 27 := sorry
  have H7 : (10 * A + B ) ^ 2 = (A + B ) ^ 3 := sorry
  exact ⟨A, B, H1, H2, H3, H4, H5, ⟨H6, H7⟩⟩

end find_special_two_digit_number_l673_673883


namespace sum_of_first_5_terms_of_geometric_sequence_l673_673993

theorem sum_of_first_5_terms_of_geometric_sequence :
  let a₁ := 3
  let q := 4
  let n := 5
  let Sₙ := λ n : ℕ, (a₁ * (1 - q^n)) / (1 - q)
  Sₙ 5 = 1023 :=
by
  sorry

end sum_of_first_5_terms_of_geometric_sequence_l673_673993


namespace pants_after_5_years_l673_673870

theorem pants_after_5_years (initial_pants : ℕ) (pants_per_year : ℕ) (years : ℕ) :
  initial_pants = 50 → pants_per_year = 8 → years = 5 → (initial_pants + pants_per_year * years) = 90 :=
by
  intros initial_cond pants_per_year_cond years_cond
  rw [initial_cond, pants_per_year_cond, years_cond]
  norm_num
  done

end pants_after_5_years_l673_673870


namespace length_BC_eq_4_l673_673826

-- Definitions directly as expressed in the problem conditions
variables (A B C P Q R : Type)
variables [triangle A B C] [incircle omega A B C]
variables (midpoint_arc_P : arc_midpoint omega A B P)
variables (midpoint_arc_Q : arc_midpoint omega A C Q)
variables (tangent_at_A : tangent omega A)
variables (intersects_PQ_at_R : LineIntersect tangent_at_A.line P Q R)
variables (midpoint_AR_on_BC : midpoint_line_intersection (midpoint_segment A R) B C)
variables [perimeter_A_B_C_12 : perimeter A B C = 12]

-- Statement for the proof problem
theorem length_BC_eq_4 :
  length B C = 4 :=
sorry

end length_BC_eq_4_l673_673826


namespace number_of_valid_n_l673_673958

theorem number_of_valid_n : 
  {n : ℕ | 0 < n ∧ n < 42 ∧ ∃ k : ℕ, n = k * (42 - n)}.card = 6 := by
  sorry

end number_of_valid_n_l673_673958


namespace min_distance_parabola_l673_673476

noncomputable def minimum_sum_distance : ℝ :=
  let P := {x: ℝ × ℝ | x.2^2 = 4 * x.1}
  let A := (0, 2)
  let y_axis := (λ (p : ℝ × ℝ => |p.1|)
  infi (λ p : P, dist p A + y_axis p - 1)

theorem min_distance_parabola (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1) :
  infi (λ p : P, dist p A + |p.1|) = sqrt 5 - 1 := 
sorry

end min_distance_parabola_l673_673476


namespace television_hours_watched_l673_673416

theorem television_hours_watched (minutes_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ)
  (h1 : minutes_per_day = 45) (h2 : days_per_week = 4) (h3 : weeks = 2):
  (minutes_per_day * days_per_week / 60) * weeks = 6 :=
by
  sorry

end television_hours_watched_l673_673416


namespace perimeter_of_quadrilateral_l673_673103

theorem perimeter_of_quadrilateral
  (A B C D E : Type)
  (triangle_abe: ABE : Type)
  (triangle_bce: BCE : Type)
  (triangle_cde: CDE : Type)
  (right_angle_AEB : RightAngle A E B)
  (right_angle_BEC : RightAngle B E C)
  (right_angle_CED : RightAngle C E D)
  (angle_AEB_60 : angle A E B = 60)
  (angle_BEC_60 : angle B E C = 60)
  (angle_CED_60 : angle C E D = 60)
  (length_AE : length A E = 36) :
  perimeter A B C D = 36 * sqrt 3 + 45 :=
sorry

end perimeter_of_quadrilateral_l673_673103


namespace enclosed_area_eq_one_third_l673_673692

theorem enclosed_area_eq_one_third :
  (∫ x in (0:ℝ)..1, (sqrt x - x^2)) = 1 / 3 := 
by
  sorry

end enclosed_area_eq_one_third_l673_673692


namespace correct_option_l673_673781

-- Definitions based on conditions
def sentence_structure : String := "He’s never interested in what ______ is doing."

def option_A : String := "no one else"
def option_B : String := "anyone else"
def option_C : String := "someone else"
def option_D : String := "nobody else"

-- The proof statement
theorem correct_option : option_B = "anyone else" := by
  sorry

end correct_option_l673_673781


namespace find_f1_and_f1_l673_673028

theorem find_f1_and_f1' (f : ℝ → ℝ) (f' : ℝ → ℝ) (h_deriv : ∀ x, deriv f x = f' x)
  (h_eq : ∀ x, f x = 2 * x * f' 1 + Real.log x) : f 1 + f' 1 = -3 :=
by sorry

end find_f1_and_f1_l673_673028


namespace problem_sum_of_divisors_l673_673434

theorem problem_sum_of_divisors {i j : ℕ} :
  (∑ k in (finset.range (i + 1)), (2^k)) *
  (∑ k in (finset.range (j + 1)), (3^k)) = 360 → 
  (i = 3 ∧ j = 3) :=
begin
  sorry,
end

end problem_sum_of_divisors_l673_673434


namespace num_people_end_race_l673_673726

-- Define the conditions
def num_cars : ℕ := 20
def initial_passengers_per_car : ℕ := 2
def drivers_per_car : ℕ := 1
def additional_passengers_per_car : ℕ := 1

-- Define the total number of people in a car at the start
def total_people_per_car_initial := initial_passengers_per_car + drivers_per_car

-- Define the total number of people in a car after halfway point
def total_people_per_car_end := total_people_per_car_initial + additional_passengers_per_car

-- Define the total number of people in all cars at the end
def total_people_end := num_cars * total_people_per_car_end

-- Theorem statement
theorem num_people_end_race : total_people_end = 80 := by
  sorry

end num_people_end_race_l673_673726


namespace angle_of_inclination_l673_673887

theorem angle_of_inclination : ∀ (L : ℝ → ℝ × ℝ × ℝ),
  (∀ t, L t = (t, t, 0)) →
  ∃ θ : ℝ, θ = Real.pi / 2 :=
by {
  intro L hL,
  use Real.pi / 2,
  sorry
}

end angle_of_inclination_l673_673887


namespace sign_painter_earns_123_l673_673401

structure HouseNumbers (start : ℕ) (step : ℕ) :=
  (n : ℕ)
  (address : ℕ := start + step * (n - 1))

def count_digits (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else 3

def total_earnings (south : HouseNumbers) (north : HouseNumbers) : ℕ :=
  let south_addresses := list.map (λ i, HouseNumbers.addr south i * HouseNumbers.step south) (list.range' 1 25)
  let north_addresses := list.map (λ i, HouseNumbers.addr north i * HouseNumbers.step north) (list.range' 1 25)
  let digits_counts := (south_addresses ++ north_addresses).map count_digits
  digits_counts.sum

noncomputable def solution : ℕ :=
  total_earnings {start := 5, step := 7, n := 25} {start := 2, step := 8, n := 25}

theorem sign_painter_earns_123 :
  solution = 123 :=
sorry

end sign_painter_earns_123_l673_673401


namespace exists_median_parallel_or_within_l673_673030

noncomputable def is_outside_plane (α : Plane) (P : Point) := -- auxiliary definition for readability
  P ∉ α

noncomputable def equidistant_from_plane (α : Plane) (A B C : Point) := -- auxiliary for readability
  dist A α = dist B α ∧ dist B α = dist C α

theorem exists_median_parallel_or_within
  (α : Plane)
  (A B C : Point)
  (h_noncollinear : ¬ collinear A B C)
  (h_outside : is_outside_plane α A ∧ is_outside_plane α B ∧ is_outside_plane α C)
  (h_equidistant : equidistant_from_plane α A B C) :
  ∃ D E : Point, (median_line A B C D E ∧ (parallel E α ∨ E ∈ α)) :=
sorry

end exists_median_parallel_or_within_l673_673030


namespace roots_of_polynomial_l673_673898

theorem roots_of_polynomial :
  let f := (λ x : ℝ, (x^2 - 5 * x + 6) * x * (x - 4) * (x - 6)) in
  {x : ℝ | f x = 0} = {0, 2, 3, 4, 6} :=
by
  -- proof to be filled in
  sorry

end roots_of_polynomial_l673_673898


namespace largest_integer_dividing_consecutive_product_l673_673245

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l673_673245


namespace equation_of_perpendicular_line_l673_673722

theorem equation_of_perpendicular_line 
  (b c : ℝ) 
  (h1 : ∀ x : ℝ, (x^2 + b*x + c) = x^2 + b*x + c) 
  (h2 : (1^2 + b*1 + c) = 2) 
  (h3 : b = -3 ∧ c = 4) 
  (h4 : ∀ x : ℝ, (2*x + b) = 2*x + b) 
  (h5 : ∀ θ : ℝ, real.sin θ / real.cos θ = b - 45) :
  (eq_triangle (-3) 4) = (x - y + 7) :=
by
sorry

end equation_of_perpendicular_line_l673_673722


namespace largest_divisor_of_consecutive_product_l673_673223

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l673_673223


namespace initial_tomatoes_l673_673386

def t_picked : ℕ := 83
def t_left : ℕ := 14
def t_total : ℕ := t_picked + t_left

theorem initial_tomatoes : t_total = 97 := by
  rw [t_total]
  rfl

end initial_tomatoes_l673_673386


namespace shaded_area_ECODF_l673_673424

-- Definitions based on conditions
def circle_A := { radius := 3 }
def circle_B := { radius := 4 }
def point_O_midpoint_AB := true
def OA := 3 * Real.sqrt 3
def OB := 4 * Real.sqrt 2
def OC_tangent_A := true
def OD_tangent_B := true
def EF_common_tangent := true

theorem shaded_area_ECODF :
  let area_rectangle_ABFE := (3 + 4) * (3 * Real.sqrt 3 + 4 * Real.sqrt 2)
  let area_triangle_ACO := 0.5 * 3^2
  let area_triangle_BDO := 0.5 * 4^2
  let angle_sector_CAE := 45
  let area_sector_CAE := (angle_sector_CAE/360) * (3^2 * Real.pi)
  let area_sector_DBF := (angle_sector_CAE/360) * (4^2 * Real.pi)
  let total_area_shaded_region :=
    area_rectangle_ABFE
    - (area_triangle_ACO + area_triangle_BDO + area_sector_CAE + area_sector_DBF)
  area_rectangle_ABFE - 
  (area_triangle_ACO + area_triangle_BDO + area_sector_CAE + area_sector_DBF) = 
  21 * Real.sqrt 3 + 28 * Real.sqrt 2 - (41 / 2) - (25 * Real.pi / 8) :=
by
  sorry

end shaded_area_ECODF_l673_673424


namespace triangle_circumcenter_diff_squared_l673_673129

noncomputable theory

open_locale big_operators

variables {A B C E F P D O : Type}

def triangle (A B C : Type) := (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)
def is_on_segment (P A B : Type) := ∃ λ (t : ℝ), t ∈ set.Icc (0 : ℝ) 1 ∧ (P = t • A + (1 - t) • B)
def circumcenter (E F D : Type) : Type := sorry
def circumradius (E F D : Type) : ℝ := sorry

theorem triangle_circumcenter_diff_squared 
  (AB AC BC : ℝ) (h1 : AB = 42) (h2 : AC = 39) (h3 : BC = 45) 
  (AE AF : ℝ) (h4 : AF = 21) (h5 : AE = 13)
  (triangleABC : triangle A B C)
  (onSegmentE : is_on_segment E A C) (onSegmentF : is_on_segment F A B) 
  (intersectCFAndABAtP : ∃ λ (P : Type), intersect CF BE = P)
  (rayAPmeetsBCatD : ∃ λ (D : Type), meets AP BC D)
  (O := circumcenter D E F) (R := circumradius D E F)
  : CO * CO = R * R := 
begin
  sorry 
end

end triangle_circumcenter_diff_squared_l673_673129


namespace statement_A_statement_E_statement_B_false_statement_C_false_statement_D_false_l673_673346

theorem statement_A : ∃ n : ℤ, 20 = 4 * n := by 
  sorry

theorem statement_E : ∃ n : ℤ, 180 = 9 * n := by 
  sorry

theorem statement_B_false : ¬ (19 ∣ 57) := by 
  sorry

theorem statement_C_false : 30 ∣ 90 := by 
  sorry

theorem statement_D_false : 17 ∣ 51 := by 
  sorry

end statement_A_statement_E_statement_B_false_statement_C_false_statement_D_false_l673_673346


namespace largest_integer_dividing_consecutive_product_l673_673247

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l673_673247


namespace complementary_event_is_at_most_one_wins_l673_673801

-- Define the Event A
def event_A : set (bool × bool) := { (tt, tt) }

-- Define the Complementary Event of A
def complementary_event (Α : set (bool × bool)) : set (bool × bool) :=
  { ω | ω ∉ Α }

-- Definition of "at most one of A and B wins a prize"
def at_most_one_wins : set (bool × bool) :=
  { (tt, ff), (ff, tt), (ff, ff) }

theorem complementary_event_is_at_most_one_wins :
  complementary_event event_A = at_most_one_wins :=
by
  sorry

end complementary_event_is_at_most_one_wins_l673_673801


namespace largest_divisor_of_5_consecutive_integers_l673_673321

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l673_673321


namespace shorter_leg_of_right_triangle_l673_673575

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h: a^2 + b^2 = c^2) (hn: c = 65): a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l673_673575


namespace largest_divisor_of_consecutive_five_l673_673298

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l673_673298


namespace shaded_area_between_circles_l673_673806

noncomputable def circle1_radius : ℝ := 5
noncomputable def distance_centers : ℝ := 3

noncomputable def circle1Area (r : ℝ) : ℝ := π * r^2
noncomputable def circle2Area (r : ℝ) : ℝ := π * (5 + 3)^2
noncomputable def shadedArea {r1 r2 : ℝ} (area1 : ℝ) (area2 : ℝ) : ℝ := area2 - area1

theorem shaded_area_between_circles 
  (h₁ : circle1_radius = 5) 
  (h₂ : distance_centers = 3) 
  (h₃ : circle1Area circle1_radius = 25 * π) 
  (h₄ : circle2Area (5 + 3) = 64 * π) 
  : shadedArea (circle1Area circle1_radius) (circle2Area (5 + 3)) = 39 * π :=
begin
  sorry
end

end shaded_area_between_circles_l673_673806


namespace cube_sum_mod_150_eq_1_l673_673897

def cube_sum_modulo (n : ℕ) : ℕ :=
  (List.range (n+1)).sum (λ x => x^3) % 7

theorem cube_sum_mod_150_eq_1 : cube_sum_modulo 150 = 1 := 
  by
    -- proof goes here
    sorry

end cube_sum_mod_150_eq_1_l673_673897


namespace right_triangle_shorter_leg_l673_673563

theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : c = 65) : a = 25 ∨ b = 25 := 
by
  sorry

end right_triangle_shorter_leg_l673_673563


namespace largest_divisor_of_5_consecutive_integers_l673_673303

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l673_673303


namespace medians_of_right_triangle_l673_673086

theorem medians_of_right_triangle (a b : ℕ) (ha : a = 3) (hb : b = 4) :
  let c := Real.sqrt (a^2 + b^2),
      m_c := c / 2,
      m_a := (Real.sqrt (2 * b^2 + 2 * c^2 - a^2)) / 2,
      m_b := (Real.sqrt (2 * a^2 + 2 * c^2 - b^2)) / 2
  in m_c = 2.5 ∧ m_a = Real.sqrt 73 / 2 ∧ m_b = Real.sqrt 13 :=
by
  sorry

end medians_of_right_triangle_l673_673086


namespace meal_serving_count_correct_l673_673754

def meals_served_correctly (total_people : ℕ) (meal_type : Type*)
  (orders : meal_type → ℕ) (correct_meals : ℕ) : ℕ :=
  -- function to count the number of ways to serve meals correctly
  sorry

theorem meal_serving_count_correct (total_people : ℕ) (meal_type : fin 3) 
  [decidable_eq meal_type]
  (orders : fin 3 → ℕ) (h_orders : orders = (λ x, 4)) :
  meals_served_correctly total_people meal_type orders 2 = 22572 :=
  begin
    have orders_correct: ∀ x, orders x = 4 := by rw h_orders,
    -- Further steps and usage of derangements would be here, 
    -- but for now we will skip to the final count.
    sorry
  end

end meal_serving_count_correct_l673_673754


namespace find_b_and_sinA_find_sin_2A_plus_pi_over_4_l673_673080

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (sinB : ℝ)

-- Conditions
def triangle_conditions :=
  (a > b) ∧
  (a = 5) ∧
  (c = 6) ∧
  (sinB = 3 / 5)

-- Question 1: Prove b = sqrt 13 and sin A = (3 * sqrt 13) / 13
theorem find_b_and_sinA (h : triangle_conditions a b c sinB) :
  b = Real.sqrt 13 ∧
  ∃ sinA : ℝ, sinA = (3 * Real.sqrt 13) / 13 :=
  sorry

-- Question 2: Prove sin (2A + π/4) = 7 * sqrt 2 / 26
theorem find_sin_2A_plus_pi_over_4 (h : triangle_conditions a b c sinB)
  (hb : b = Real.sqrt 13)
  (sinA : ℝ)
  (h_sinA : sinA = (3 * Real.sqrt 13) / 13) :
  ∃ sin2Aπ4 : ℝ, sin2Aπ4 = (7 * Real.sqrt 2) / 26 :=
  sorry

end find_b_and_sinA_find_sin_2A_plus_pi_over_4_l673_673080


namespace triangle_problem_l673_673492

theorem triangle_problem
  (a b c : ℝ) (A B C : ℝ)
  (cosA : ℝ) (hcosA : cosA = 1 / 3)
  (hb : b = 2 / 3 * c)
  (areaABC : ℝ) (harea : areaABC = √2) :
  b = √2 ∧ (sin C = 2 * √2 / 3) :=
sorry

end triangle_problem_l673_673492


namespace richard_knocked_down_more_pins_l673_673099

theorem richard_knocked_down_more_pins :
  let patrick_first_round := 70 in
  let richard_first_round := patrick_first_round + 15 in
  let patrick_second_round := 2 * richard_first_round in
  let richard_second_round := patrick_second_round - 3 in
  let patrick_total := patrick_first_round + patrick_second_round in
  let richard_total := richard_first_round + richard_second_round in
  richard_total - patrick_total = 12 :=
by
  sorry

end richard_knocked_down_more_pins_l673_673099


namespace probability_X_eq_Y_l673_673839

open Real

theorem probability_X_eq_Y :
  let s := -15 * π / 2
  let t := 15 * π / 2
  (∀ (X Y : ℝ), s ≤ X ∧ X ≤ t ∧ s ≤ Y ∧ Y ≤ t ∧ (cos (sin X) = cos (sin Y)) → X = Y) →
  (∀ (X Y : ℝ), s ≤ X ∧ X ≤ t ∧ s ≤ Y ∧ Y ≤ t → set.prob (set_of (λ p : ℝ × ℝ, p.fst = p.snd)) (set.prod (Icc s t) (Icc s t)) = 15 / (225 * π^2)) :=
sorry

end probability_X_eq_Y_l673_673839


namespace probability_gcd_three_numbers_one_l673_673739

noncomputable def probability_gcd_one : ℚ :=
  let total_subsets : ℕ := choose 8 3 in
  let non_rel_prime_subsets : ℕ := 4 in
  let prob := (total_subsets - non_rel_prime_subsets : ℚ) / total_subsets in
  prob

theorem probability_gcd_three_numbers_one :
  probability_gcd_one = 13 / 14 :=
by
  sorry

end probability_gcd_three_numbers_one_l673_673739


namespace chromatic_number_of_grid_3x5_l673_673096

-- Define a 3x5 grid graph where each vertex represents a square and edges represent adjacency by vertex or side
def grid_3x5 : SimpleGraph (Fin 3 × Fin 5) :=
  { adj := λ x y, (x.1 = y.1 ∧ (x.2 = y.2 + 1 ∨ x.2 = y.2 - 1)) ∨
                 (x.2 = y.2 ∧ (x.1 = y.1 + 1 ∨ x.1 = y.1 - 1)) ∨
                 ((x.1 = y.1 + 1 ∨ x.1 = y.1 - 1) ∧ (x.2 = y.2 + 1 ∨ x.2 = y.2 - 1)),
    sym := by finish,
    loopless := by finish }

-- A proof problem to determine the chromatic number of grid_3x5 is 4
theorem chromatic_number_of_grid_3x5 : chromaticNumber grid_3x5 = 4 :=
sorry

end chromatic_number_of_grid_3x5_l673_673096


namespace range_of_a_l673_673015

variables (a : ℝ) 

def P : Prop :=
  ∀ m : ℝ, m ∈ Set.Icc (-1 : ℝ) 1 → (a^2 - 5*a + 7 ≥ m + 2)

def Q : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ (x ^ 2 + a * x + 2 = 0) ∧ (y ^ 2 + a * y + 2 = 0)

theorem range_of_a (h1 : P ∨ Q) (h2 : ¬(P ∧ Q)) :
  (-2*Real.sqrt 2 ≤ a ∧ a ≤ 1) ∨ (2*Real.sqrt 2 < a ∧ a < 4) :=
sorry

end range_of_a_l673_673015


namespace largest_divisor_of_product_of_5_consecutive_integers_l673_673238

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l673_673238


namespace sum_of_consecutive_odds_l673_673963

theorem sum_of_consecutive_odds (n : ℕ) (hn : n > 0) : 
  (Finset.range n).sum (λ k, (2 * k + 1)) = n^2 := 
sorry

end sum_of_consecutive_odds_l673_673963


namespace miss_adamson_num_classes_l673_673643

theorem miss_adamson_num_classes
  (students_per_class : ℕ)
  (sheets_per_student : ℕ)
  (total_sheets : ℕ)
  (h1 : students_per_class = 20)
  (h2 : sheets_per_student = 5)
  (h3 : total_sheets = 400) :
  let sheets_per_class := sheets_per_student * students_per_class
  let num_classes := total_sheets / sheets_per_class
  num_classes = 4 :=
by
  sorry

end miss_adamson_num_classes_l673_673643


namespace basketball_total_points_l673_673548

variable (Jon_points Jack_points Tom_points : ℕ)

def Jon_score := 3
def Jack_score := Jon_score + 5
def Tom_score := (Jon_score + Jack_score) - 4

theorem basketball_total_points :
  Jon_score + Jack_score + Tom_score = 18 := by
  sorry

end basketball_total_points_l673_673548


namespace expected_malfunctioning_computers_l673_673713

theorem expected_malfunctioning_computers (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  let P0 := (1 - a) * (1 - b),
      P1 := a * (1 - b) + (1 - a) * b,
      P2 := a * b,
      E_X := 0 * P0 + 1 * P1 + 2 * P2 in
  E_X = a + b :=
by
  sorry

end expected_malfunctioning_computers_l673_673713


namespace problem1_cond1_problem1_cond2_problem1_cond3_problem2_l673_673853

variable {a b c : ℝ}
variable {A B C : ℝ}

-- Given the sides opposite to angles A, B, C are a, b, c respectively

-- Condition 1
axiom cond1 : c^2 + a * b = c * (a * Real.cos B - b * Real.cos A) + 2 * b^2

-- Condition 2
axiom cond2 : (b + c) * (Real.sin B - Real.sin C) = -a * (Real.sin A - Real.sin B)

-- Condition 3
axiom cond3 : b * Real.sin C = Real.sqrt 3 * (a - c * Real.cos B)

-- Problem 1: Show that C = π/3
theorem problem1_cond1 (h : cond1) : C = Real.pi / 3 := sorry
theorem problem1_cond2 (h : cond2) : C = Real.pi / 3 := sorry
theorem problem1_cond3 (h : cond3) : C = Real.pi / 3 := sorry

-- Problem 2: Show that, if c = 2 * sqrt 3, the range of values for 4 * sin B - a is (-2 * sqrt 3, 2 * sqrt 3)
theorem problem2 (hc : c = 2 * Real.sqrt 3) (hC : C = Real.pi / 3) 
: -2 * Real.sqrt 3 < 4 * Real.sin B - a ∧ 4 * Real.sin B - a < 2 * Real.sqrt 3 := sorry

end problem1_cond1_problem1_cond2_problem1_cond3_problem2_l673_673853


namespace number_of_podium_outcomes_l673_673407

theorem number_of_podium_outcomes (n : ℕ) (h : n = 6) : 
  (6 * 5 * 4) = 120 :=
by
  rw h
  norm_num

end number_of_podium_outcomes_l673_673407


namespace sufficient_but_not_necessary_l673_673021

noncomputable def condition_to_bool (a b : ℝ) : Bool :=
a > b ∧ b > 0

theorem sufficient_but_not_necessary (a b : ℝ) (h : condition_to_bool a b) :
  (a > b ∧ b > 0) → (a^2 > b^2) ∧ (∃ a' b' : ℝ, a'^2 > b'^2 ∧ ¬ (a' > b' ∧ b' > 0)) :=
by
  sorry

end sufficient_but_not_necessary_l673_673021


namespace probability_square_product_l673_673756

theorem probability_square_product :
  let total_outcomes := 12 * 12 in
  let favorable_outcomes := 28 in
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 36 :=
by
  sorry

end probability_square_product_l673_673756


namespace age_of_25th_student_l673_673693

variable (total_students : ℕ) (total_average : ℕ)
variable (group1_students : ℕ) (group1_average : ℕ)
variable (group2_students : ℕ) (group2_average : ℕ)

theorem age_of_25th_student 
  (h1 : total_students = 25) 
  (h2 : total_average = 25)
  (h3 : group1_students = 10)
  (h4 : group1_average = 22)
  (h5 : group2_students = 14)
  (h6 : group2_average = 28) : 
  (total_students * total_average) =
  (group1_students * group1_average) + (group2_students * group2_average) + 13 :=
by sorry

end age_of_25th_student_l673_673693


namespace max_y_difference_intersection_l673_673452

noncomputable def f (x : ℝ) : ℝ := 4 - x^2 + x^3
noncomputable def g (x : ℝ) : ℝ := 2 + x^2 + x^3

theorem max_y_difference_intersection :
  let x1 := 1
  let y1 := g x1
  let x2 := -1
  let y2 := g x2
  y1 - y2 = 2 :=
by
  sorry

end max_y_difference_intersection_l673_673452


namespace meal_serving_problem_l673_673747

/-
Twelve people sit down for dinner where there are three choices of meals: beef, chicken, and fish.
Four people order beef, four people order chicken, and four people order fish.
The waiter serves the twelve meals in random order.
We need to find the number of ways in which the waiter could serve the meals so that exactly two people receive the type of meal ordered by them.
-/
theorem meal_serving_problem :
    ∃ (n : ℕ), n = 12210 ∧
    (∃ (people : Fin 12 → char), 
        (∀ i : Fin 4, people i = 'B') ∧ 
        (∀ i : Fin 4, people (i + 4) = 'C') ∧ 
        (∀ i : Fin 4, people (i + 8) = 'F') ∧ 
        (∃ (served : Fin 12 → char), 
            (∃ (correct : Fin 12), set.range correct ⊆ {0, 1} ∧
            (∀ i : Fin 12, (served i = people correct i) ↔ (i ∈ {0, 1}) = true)) ∧
            (related_permutations served people))
    )
    sorry

end meal_serving_problem_l673_673747


namespace probability_gcd_three_numbers_one_l673_673740

noncomputable def probability_gcd_one : ℚ :=
  let total_subsets : ℕ := choose 8 3 in
  let non_rel_prime_subsets : ℕ := 4 in
  let prob := (total_subsets - non_rel_prime_subsets : ℚ) / total_subsets in
  prob

theorem probability_gcd_three_numbers_one :
  probability_gcd_one = 13 / 14 :=
by
  sorry

end probability_gcd_three_numbers_one_l673_673740


namespace average_weight_of_additional_friends_is_50_l673_673171

noncomputable def average_weight_additional_friends (initial_avg_weight friends_weight_increase total_avg_weight num_initial_friends num_additional_friends : ℝ) : ℝ :=
  let total_weight_initial := initial_avg_weight * num_initial_friends
  let total_weight_final := total_avg_weight * (num_initial_friends + num_additional_friends)
  let total_weight_additional := total_weight_final - total_weight_initial
  total_weight_additional / num_additional_friends

theorem average_weight_of_additional_friends_is_50 :
  ∀ (initial_avg_weight friends_weight_increase total_avg_weight num_initial_friends num_additional_friends: ℝ),
    friends_weight_increase = 10 → 
    total_avg_weight = 40 →
    num_initial_friends = 30 →
    num_additional_friends = 30 →
    initial_avg_weight = total_avg_weight - friends_weight_increase →
    average_weight_additional_friends initial_avg_weight friends_weight_increase total_avg_weight num_initial_friends num_additional_friends = 50 := by {
      intros,
      sorry
    }

end average_weight_of_additional_friends_is_50_l673_673171


namespace collinear_TA_TB_TC_l673_673108

-- Define a non-isosceles triangle ABC
def non_isosceles_triangle (A B C : Type) : Prop := ¬ (A = B ∨ B = C ∨ C = A)

-- Define the altitudes AA1, BB1, and CC1 of triangle ABC
def altitude (A A1 B B1 C C1 : Type) : Prop := line_from A A1 ∧ line_from B B1 ∧ line_from C C1 ∧ 
  perp A A1 B ∧ perp B B1 C ∧ perp C C1 A

-- Define points BA and CA on BB1 and CC1 respectively such that A1BA ⊥ BB1 and A1CA ⊥ CC1
def points_on_altitudes (A1 B1 C1 A B C BA CA : Type) : Prop :=
  on_line BA B1 ∧ on_line CA C1 ∧ perp A1 BA B1 ∧ perp A1 CA C1

-- Define points TA, TB, and TC where BACA and BC intersect at TA, similarly for TB and TC
def intersection_points (BA CA BC T_A B B1 C C1 A1 T_B T_C : Type) : Prop :=
  intersect BA CA BC T_A ∧ intersect CA BC T_B ∧ intersect BA BC T_C

-- The main theorem
theorem collinear_TA_TB_TC {A B C A1 B1 C1 BA CA B1 C1 T_A T_B T_C : Type} :
  non_isosceles_triangle A B C →
  altitude A A1 B B1 C C1 →
  points_on_altitudes A1 B1 C1 A B C BA CA →
  intersection_points BA CA B1 T_A B B1 C C1 A1 T_B T_C →
  collinear T_A T_B T_C :=
sorry

end collinear_TA_TB_TC_l673_673108


namespace largest_divisor_of_consecutive_five_l673_673295

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l673_673295


namespace find_nat_nums_satisfying_eq_l673_673338

theorem find_nat_nums_satisfying_eq (m n : ℕ) (h_m : m = 3) (h_n : n = 3) : 2 ^ n + 1 = m ^ 2 :=
by
  rw [h_m, h_n]
  sorry

end find_nat_nums_satisfying_eq_l673_673338


namespace sixtieth_pair_is_correct_l673_673478

theorem sixtieth_pair_is_correct :
  ∃ (p : ℕ × ℕ), nth_pair 60 = (5, 7) :=
sorry

end sixtieth_pair_is_correct_l673_673478


namespace largest_divisor_of_5_consecutive_integers_l673_673325

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l673_673325


namespace smallest_composite_proof_l673_673902

-- Define what it means for a number not to have prime factors less than 15
def no_prime_factors_less_than_15 (n : ℕ) : Prop :=
  ∀ p : ℕ, nat.prime p → p ∣ n → p ≥ 15

-- Define what it means for a number to be the smallest composite number with the above property
def smallest_composite_without_prime_factors_less_than_15 (n : ℕ) : Prop :=
  nat.composite n ∧ no_prime_factors_less_than_15 n ∧
  ∀ m : ℕ, nat.composite m → no_prime_factors_less_than_15 m → n ≤ m

theorem smallest_composite_proof : smallest_composite_without_prime_factors_less_than_15 323 :=
  sorry

end smallest_composite_proof_l673_673902


namespace average_gas_mileage_l673_673833

theorem average_gas_mileage (
  dist1 dist2 : ℝ,
  mileage1 mileage2 : ℝ,
  h1 : dist1 = 150,
  h2 : dist2 = 180,
  h3 : mileage1 = 40,
  h4 : mileage2 = 24
) : 
  (dist1 + dist2) / ((dist1 / mileage1) + (dist2 / mileage2)) = 29.33 :=
by
  -- Skip the proof for now
  sorry

end average_gas_mileage_l673_673833


namespace length_of_parametric_curve_l673_673451

theorem length_of_parametric_curve :
  let f x y : ℝ := (3 * (Real.sin y), 3 * (Real.cos y))
  let t₀ : ℝ := 0
  let t₁ : ℝ := (3 * Real.pi / 2)
  let length : ℝ :=
    ∫ t in t₀..t₁, Real.sqrt ((f (Real.sin t) (Real.cos t)).fst^2 + (f (Real.sin t) (Real.cos t)).snd^2) 
  length = 4.5 * Real.pi :=
by
  let f : ℝ → ℝ × ℝ := fun t => (3 * Real.sin t, 3 * Real.cos t)
  let t₀ : ℝ := 0
  let t₁ : ℝ := 3 * Real.pi / 2
  let derivative := (Real.cos, -Real.sin)
  let distance := fun t => Real.sqrt (derivative t).fst^2 + (derivative t).snd^2
  let integral := ∫ t in t₀..t₁, distance t
  have h : integral = 4.5 * Real.pi :=
    sorry
  exact h

end length_of_parametric_curve_l673_673451


namespace real_part_zero_implies_x3_l673_673071

theorem real_part_zero_implies_x3 (x : ℝ) : 
  (x^2 - 2*x - 3 = 0) ∧ (x + 1 ≠ 0) → x = 3 :=
by
  sorry

end real_part_zero_implies_x3_l673_673071


namespace largest_integer_dividing_consecutive_product_l673_673253

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l673_673253


namespace meal_serving_count_correct_l673_673753

def meals_served_correctly (total_people : ℕ) (meal_type : Type*)
  (orders : meal_type → ℕ) (correct_meals : ℕ) : ℕ :=
  -- function to count the number of ways to serve meals correctly
  sorry

theorem meal_serving_count_correct (total_people : ℕ) (meal_type : fin 3) 
  [decidable_eq meal_type]
  (orders : fin 3 → ℕ) (h_orders : orders = (λ x, 4)) :
  meals_served_correctly total_people meal_type orders 2 = 22572 :=
  begin
    have orders_correct: ∀ x, orders x = 4 := by rw h_orders,
    -- Further steps and usage of derangements would be here, 
    -- but for now we will skip to the final count.
    sorry
  end

end meal_serving_count_correct_l673_673753


namespace largest_divisor_of_five_consecutive_integers_l673_673261

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l673_673261


namespace document_word_count_approximation_l673_673385

theorem document_word_count_approximation :
  let pages := 8
  let words_per_page := 605
  let total_words := pages * words_per_page
  total_words ≈ 4800 := 
by 
  let pages := 8
  let words_per_page := 605
  let total_words := pages * words_per_page
  sorry

end document_word_count_approximation_l673_673385


namespace smallest_composite_no_prime_factors_below_15_correct_l673_673927

def smallest_composite_no_prime_factors_below_15 : Nat :=
  323
  
theorem smallest_composite_no_prime_factors_below_15_correct :
  (∀ n < 15, Prime n → ¬ (n ∣ smallest_composite_no_prime_factors_below_15)) ∧
  (∃ p q, Prime p ∧ Prime q ∧ p ≠ q ∧ smallest_composite_no_prime_factors_below_15 = p * q) :=
by
  -- Proof skipped
  sorry

end smallest_composite_no_prime_factors_below_15_correct_l673_673927


namespace ball_travel_distance_l673_673796

theorem ball_travel_distance :
  let initial_height := 20
  let bounce_ratio := 0.8
  let descent1 := initial_height
  let descent2 := initial_height * bounce_ratio
  let descent3 := descent2 * bounce_ratio
  let descent4 := descent3 * bounce_ratio
  let total_descent := descent1 + descent2 + descent3 + descent4

  let ascent1 := descent2
  let ascent2 := descent3
  let ascent3 := descent4
  let ascent4 := ascent4 * bounce_ratio
  let total_ascent := ascent1 + ascent2 + ascent3 + ascent4

  let total_distance := total_descent + total_ascent

  round total_distance = 106
:= by
  let initial_height := (20 : Real)
  let bounce_ratio := (0.8 : Real)
  let descent1 := initial_height
  let descent2 := descent1 * bounce_ratio
  let descent3 := descent2 * bounce_ratio
  let descent4 := descent3 * bounce_ratio
  let total_descent := descent1 + descent2 + descent3 + descent4
  
  let ascent1 := descent2
  let ascent2 := descent3
  let ascent3 := descent4
  let ascent4 := ascent3 * bounce_ratio
  let total_ascent := ascent1 + ascent2 + ascent3 + ascent4

  let total_distance := total_descent + total_ascent

  have H : round total_distance = 106
  exact H  -- The actual computation and verification would go here

end ball_travel_distance_l673_673796


namespace sin_alpha_in_fourth_quadrant_l673_673504

theorem sin_alpha_in_fourth_quadrant
  (α : ℝ)
  (hα_quadrant : 3 * real.pi / 2 < α ∧ α < 2 * real.pi)
  (h_tan : real.tan α = -5 / 12) :
  real.sin α = -5 / 13 := 
sorry

end sin_alpha_in_fourth_quadrant_l673_673504


namespace wise_men_avoid_poisons_l673_673784

theorem wise_men_avoid_poisons 
  (pills : Fin 6 → bool)
  (is_poisoned : Fin 6 → bool)
  (num_poisoned : ∑ i, if is_poisoned i then 1 else 0 = 2)
  (num_harmless : ∑ i, if ¬is_poisoned i then 1 else 0 = 4)
  (A_knows : ∀ i, is_poisoned i)
  (B_not_informed : ∀ i, ¬is_poisoned i → ∀ j, i ≠ j → is_poisoned j → false)
  (A_starts : True) : 
  ∃ strategy : Fin 6 → Fin 6 → bool,
    (∀ i, strategy i (if is_poisoned (i - 1) then i - 1 else i - 2))
    ∧ (∀ j, strategy (if is_poisoned j then j - 1 else j - 2) j)
    ∧ (∀ i, ∀ j, ¬is_poisoned i → ¬is_poisoned j → strategy i j)
    ∧ (∃ poison_pills : Fin 2, is_poisoned poison_pills) := sorry

end wise_men_avoid_poisons_l673_673784


namespace count_elements_with_leading_five_l673_673616

theorem count_elements_with_leading_five (S : Set ℕ)
  (hS : S = {k | ∃ (n : ℕ), k = 5^n ∧ 0 ≤ n ∧ n ≤ 3000})
  (digits_5_3000 : ∀ (d : ℕ), d = nat.digits 10 (5^3000) → d = 2135)
  (leading_digit_5_3000 : nat.digits 10 (5^3000) = 2135 → list.head!(nat.digits 10 (5^3000)) = 5) :
  Set.card {k ∈ S | ∃ (d : ℕ), nat.digits 10 k = d ∧ list.head! (nat.digits 10 k) = 5} = 867 :=
sorry

end count_elements_with_leading_five_l673_673616


namespace sales_tax_difference_l673_673698

theorem sales_tax_difference:
  let original_price := 50 
  let discount_rate := 0.10 
  let sales_tax_rate_1 := 0.08
  let sales_tax_rate_2 := 0.075 
  let discounted_price := original_price * (1 - discount_rate) 
  let sales_tax_1 := discounted_price * sales_tax_rate_1 
  let sales_tax_2 := discounted_price * sales_tax_rate_2 
  sales_tax_1 - sales_tax_2 = 0.225 := by
  sorry

end sales_tax_difference_l673_673698


namespace range_of_f_measure_of_angle_C_l673_673523

section Problem1
variables (x : ℝ)
def vec_m := (2 * sin x, 1)
def vec_n := (√3 * cos x, 2 * (cos x) ^ 2)
def f (x : ℝ) := (2 * sin x * √3 * cos x) + (1 * 2 * (cos x) ^ 2)

theorem range_of_f : set.range (λ x : ℝ, 2 * sin (2 * x + (π / 6)) + 1) = set.Icc 0 3 :=
sorry
end Problem1

section Problem2
variables (l : ℝ) (A C : ℝ)
-- Conditions
axiom side_a : 1 = l
axiom side_b : √3 = b
axiom f_A : 2 * sin (2 * A + (π / 6)) + 1 = 3

theorem measure_of_angle_C (C : ℝ) : C = (π / 2) ∨ C = (π / 6) :=
sorry
end Problem2

end range_of_f_measure_of_angle_C_l673_673523


namespace largest_divisor_of_consecutive_product_l673_673228

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l673_673228


namespace eden_has_28_bears_l673_673432

variable (d_bears : ℝ) -- Daragh's original number of bears
variable (keep_pct : ℝ) -- Percentage of bears Daragh keeps
variable (give_pct : ℝ) -- Percentage of bears given to Aria
variable (eden_initial : ℝ) -- Eden's initial number of bears
variable (sisters : ℝ) -- Number of sisters (including Eden) the remaining bears are divided among

-- Given conditions
def initial_conditions : Prop :=
  d_bears = 80 ∧
  keep_pct = 0.40 ∧
  give_pct = 0.30 ∧
  eden_initial = 20 ∧
  sisters = 3

-- Prove that Eden now has 28 bears
theorem eden_has_28_bears : initial_conditions d_bears keep_pct give_pct eden_initial sisters → 
  let d_keep := d_bears * keep_pct in
  let d_give := d_bears - d_keep in
  let a_bears := d_bears * give_pct in
  let remaining_bears := d_give - a_bears in
  let bears_per_sister := remaining_bears / sisters in
  eden_initial + bears_per_sister = 28 :=
by
  sorry

end eden_has_28_bears_l673_673432


namespace range_of_m_l673_673511

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 ≥ 0) ↔ (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l673_673511


namespace parabola_directrix_l673_673889

theorem parabola_directrix :
  ∀ (x : ℝ), (∃ c : ℝ, c = -\frac{47}{8}) → (∃ y : ℝ, y = -2 * x^2 + 4 * x - 8) →
  (∀ y : ℝ, y = -\frac{47}{8}) :=
sorry

end parabola_directrix_l673_673889


namespace range_of_k_for_increasing_function_l673_673073

-- Define the inverse proportion function
def inverse_proportion_function (x : ℝ) (k : ℝ) : ℝ :=
  k / x

-- State the theorem
theorem range_of_k_for_increasing_function :
  (∀ x y : ℝ, x ≠ 0 ∧ y = inverse_proportion_function x k → (x > 0 → ∃ z : ℝ, z > x ∧ y < inverse_proportion_function z k) ∧ (x < 0 → ∃ z : ℝ, z < x ∧ y > inverse_proportion_function z k)) →
  k < 0 :=
sorry

end range_of_k_for_increasing_function_l673_673073


namespace max_pairs_dist_1_unit_l673_673725

/-- There are 2022 distinct integer points on the plane. Prove that the maximum number of 
    pairs among these points with exactly 1 unit apart is 3954. -/
theorem max_pairs_dist_1_unit (points : Finset (ℤ × ℤ)) (h_card : points.card = 2022) :
  ∃ I, (∀ p1 p2 ∈ points, dist p1 p2 = 1 → (p1, p2) ∈ I) ∧ I.card ≤ 3954 :=
by
  -- Proof skips
  sorry

/-- Define the distance function for integer points on the plane -/
def dist (p1 p2 : ℤ × ℤ) : ℕ :=
  let (x1, y1) := p1;
  let (x2, y2) := p2;
  (abs (x1 - x2) + abs (y1 - y2))

/-- Define the cardinality function on Finsets since pairs may not be distinct -/
noncomputable def Finset.card (s : Finset (ℤ × ℤ)) : ℕ :=
  s.to_multiset.card

end max_pairs_dist_1_unit_l673_673725


namespace tan_of_A_in_triangle_l673_673049

theorem tan_of_A_in_triangle (A B C a b c : ℝ) (hC : C = 120) (ha : a = 2 * b) (h_triangle : A + B + C = 180) (hA_non_neg : 0 ≤ A) (hA_lt_180 : A < 180) (hB_non_neg : 0 ≤ B) (hB_lt_180 : B < 180): 
  ∃ A, ∃ B, tan A = sqrt 3 / 2 := 
by
  sorry

end tan_of_A_in_triangle_l673_673049


namespace parallel_lines_not_coincident_l673_673070

theorem parallel_lines_not_coincident (a : ℝ) :
  let l1 := λ x y : ℝ, a * x + 2 * y + 6 = 0,
      l2 := λ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0 in
  (∀ x y : ℝ, (l1 x y = 0 ↔ l2 x y = 0) → ¬∃ x y : ℝ, l1 x y = 0) →
  a = -1 :=
by sorry

end parallel_lines_not_coincident_l673_673070


namespace equilateral_triangle_incircle_excircle_ratio_l673_673626

theorem equilateral_triangle_incircle_excircle_ratio 
  (a : ℝ) (h_a_positive : 0 < a) (ABC : Triangle)
  (h_ABC_equilateral : ABC.is_equilateral)
  (Ω : Circle) (ω : Circle) (r1 r2 : ℝ)
  (h_Ω_incircle : Ω.is_inscribed_in ABC)
  (h_r1 : Ω.radius = r1)
  (h_ω_tangent_to_Ω : ω.is_tangent_externally_to Ω)
  (h_ω_tangent_to_AB : ω.is_tangent_to_side ABC.AB)
  (h_ω_tangent_to_AC : ω.is_tangent_to_side ABC.AC)
  (h_r2 : ω.radius = r2) :
  r1 / r2 = 3 :=
sorry

end equilateral_triangle_incircle_excircle_ratio_l673_673626


namespace part1_part2_l673_673515

-- Define the quadratic function f(x) = x^2 - 16x + p + 3
def f (x : ℝ) (p : ℝ) : ℝ := x^2 - 16*x + p + 3

-- Part 1: Proving the range of p
theorem part1 (p : ℝ) :
  (∃ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), f x p = 0) → (-20 ≤ p ∧ p ≤ 12) :=
by
  -- Proof should be filled in here
  sorry

-- Part 2: Proving the existence of q
theorem part2 (q : ℝ) :
  (q ≥ 0) →
  (∃ D : set ℝ, (∀ x ∈ set.Icc q 10, f x q ∈ D) ∧ ∃ a b, D = set.Icc a b ∧ (b - a = 12 - q)) →
  (q = 8 ∨ q = 9 ∨ q = (15 - Real.sqrt 17) / 2) :=
by
  -- Proof should be filled in here
  sorry

end part1_part2_l673_673515


namespace smallest_k_l673_673136

theorem smallest_k (a b c : ℤ) (k : ℤ) (h1 : a < b) (h2 : b < c) 
  (h3 : 2 * b = a + c) (h4 : (k * c) ^ 2 = a * b) (h5 : k > 1) : 
  c > 0 → k = 2 := 
sorry

end smallest_k_l673_673136


namespace repeating_decimal_product_as_fraction_l673_673443

theorem repeating_decimal_product_as_fraction :
  let x := 37 / 999
  let y := 7 / 9
  x * y = 259 / 8991 := by {
    sorry
  }

end repeating_decimal_product_as_fraction_l673_673443


namespace exists_real_number_l673_673612

open Nat

def sequence (k : ℕ) : ℕ → ℕ
| 1 := 1
| (n + 1) := let S_n := {x_i | ∃ i : ℕ, x_i = sequence k i ∧ 1 ≤ i ∧ i ≤ n} ∪ {x_i + i * k | ∃ i : ℕ, x_i = sequence k i ∧ 1 ≤ i ∧ i ≤ n}
             in (Inf (set_of (λ m, m > 0 ∧ m ∉ S_n)))

theorem exists_real_number (k : ℕ) (hk : k > 0):
  ∃ (a : ℝ), ∀ (n : ℕ), (n > 0) → sequence k n = ⌊a * n⌋ :=
sorry

end exists_real_number_l673_673612


namespace tan_alpha_plus_pi_over_3_l673_673486

variables (α β : ℝ)

theorem tan_alpha_plus_pi_over_3 :
  tan (α + π / 3) = 7 / 23 :=
by
  assume h1 : tan (α + β) = 3 / 5
  assume h2 : tan (β - π / 3) = 1 / 4
  sorry

end tan_alpha_plus_pi_over_3_l673_673486


namespace smallest_composite_proof_l673_673947

noncomputable def smallest_composite_no_prime_factors_less_than_15 : ℕ :=
  289

theorem smallest_composite_proof :
  smallest_composite_no_prime_factors_less_than_15 = 289 :=
by
  sorry

end smallest_composite_proof_l673_673947


namespace gcd_20m_25n_l673_673532

open Nat

theorem gcd_20m_25n {m n : ℕ} (hm : m > 0) (hn : n > 0) (h : gcd m n = 18) : gcd (20 * m) (25 * n) = 90 :=
sorry

end gcd_20m_25n_l673_673532


namespace least_positive_integer_reducible_fraction_l673_673892

-- Define gcd function as used in the problem
def is_reducible_fraction (a b : ℕ) : Prop := Nat.gcd a b > 1

-- Define the conditions and the proof problem
theorem least_positive_integer_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ is_reducible_fraction (n - 27) (7 * n + 4) ∧
  ∀ m : ℕ, (0 < m → is_reducible_fraction (m - 27) (7 * m + 4) → n ≤ m) :=
sorry

end least_positive_integer_reducible_fraction_l673_673892


namespace prove_Cantelli_inequality_l673_673622

noncomputable 
def Cantelli_inequality (ξ : ℝ → ℝ) [ProbabilityMeasureSpace ξ] : Prop :=
  (∀ ε > 0, (ProbMassFun.Real GreaterEqual (ξ − (ExpectedVal ξ)) ε) ≤ 
  (Var ξ / (Var ξ + ε^2))) 

-- Main theorem statement
theorem prove_Cantelli_inequality (ξ : ℝ → ℝ) [ProbabilityMeasureSpace ξ] 
  (h1 : ExpectedVal ξ^2 < ∞) : 
  Cantelli_inequality ξ :=
sorry

end prove_Cantelli_inequality_l673_673622


namespace shorter_leg_of_right_triangle_l673_673553

theorem shorter_leg_of_right_triangle (a b : ℕ) (h : a^2 + b^2 = 65^2) : min a b = 16 :=
sorry

end shorter_leg_of_right_triangle_l673_673553


namespace richard_knocked_down_more_pins_l673_673100

theorem richard_knocked_down_more_pins :
  let patrick_first_round := 70 in
  let richard_first_round := patrick_first_round + 15 in
  let patrick_second_round := 2 * richard_first_round in
  let richard_second_round := patrick_second_round - 3 in
  let patrick_total := patrick_first_round + patrick_second_round in
  let richard_total := richard_first_round + richard_second_round in
  richard_total - patrick_total = 12 :=
by
  sorry

end richard_knocked_down_more_pins_l673_673100


namespace largest_integer_dividing_consecutive_product_l673_673250

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l673_673250


namespace eggs_for_husband_is_correct_l673_673638

-- Define the conditions
def eggs_per_child : Nat := 2
def num_children : Nat := 4
def eggs_for_herself : Nat := 2
def total_eggs_per_year : Nat := 3380
def days_per_week : Nat := 5
def weeks_per_year : Nat := 52

-- Define the total number of eggs Lisa makes for her husband per year
def eggs_for_husband : Nat :=
  total_eggs_per_year - 
  (num_children * eggs_per_child + eggs_for_herself) * (days_per_week * weeks_per_year)

-- Prove the main statement
theorem eggs_for_husband_is_correct : eggs_for_husband = 780 := by
  sorry

end eggs_for_husband_is_correct_l673_673638


namespace brick_wall_problem_l673_673083

theorem brick_wall_problem : 
  ∀ (B1 B2 B3 B4 B5 : ℕ) (d : ℕ),
  B1 = 38 →
  B1 + B2 + B3 + B4 + B5 = 200 →
  B2 = B1 - d →
  B3 = B1 - 2 * d →
  B4 = B1 - 3 * d →
  B5 = B1 - 4 * d →
  d = 1 :=
by
  intros B1 B2 B3 B4 B5 d h1 h2 h3 h4 h5 h6
  rw [h1] at h2
  sorry

end brick_wall_problem_l673_673083


namespace ratio_of_pieces_l673_673795

theorem ratio_of_pieces (total_length : ℝ) (shorter_piece : ℝ) : 
  total_length = 60 ∧ shorter_piece = 20 → shorter_piece / (total_length - shorter_piece) = 1 / 2 :=
by
  sorry

end ratio_of_pieces_l673_673795


namespace total_length_of_segments_l673_673414

theorem total_length_of_segments (L : ℕ) (n : ℕ) (hL : L = 9) (hn : n = 9) : 
  (∑ i in finset.range n, (i + 1) * (n - i)) = 165 :=
by
  sorry

end total_length_of_segments_l673_673414


namespace problem_statement_l673_673966

theorem problem_statement (x : ℝ) (h : x + x⁻¹ = 3) : x^2 + x⁻² = 7 :=
by 
  sorry

end problem_statement_l673_673966


namespace polynomial_difference_l673_673419

theorem polynomial_difference (a : ℝ) :
  (6 * a^2 - 5 * a + 3) - (5 * a^2 + 2 * a - 1) = a^2 - 7 * a + 4 :=
by
  sorry

end polynomial_difference_l673_673419


namespace smallest_composite_proof_l673_673900

-- Define what it means for a number not to have prime factors less than 15
def no_prime_factors_less_than_15 (n : ℕ) : Prop :=
  ∀ p : ℕ, nat.prime p → p ∣ n → p ≥ 15

-- Define what it means for a number to be the smallest composite number with the above property
def smallest_composite_without_prime_factors_less_than_15 (n : ℕ) : Prop :=
  nat.composite n ∧ no_prime_factors_less_than_15 n ∧
  ∀ m : ℕ, nat.composite m → no_prime_factors_less_than_15 m → n ≤ m

theorem smallest_composite_proof : smallest_composite_without_prime_factors_less_than_15 323 :=
  sorry

end smallest_composite_proof_l673_673900


namespace unique_valid_number_l673_673769

-- Define the form of the three-digit number.
def is_form_sixb5 (n : ℕ) : Prop :=
  ∃ b : ℕ, b < 10 ∧ n = 600 + 10 * b + 5

-- Define the condition for divisibility by 11.
def is_divisible_by_11 (n : ℕ) : Prop :=
  (n % 11 = 0)

-- Define the alternating sum property for our specific number format.
def alternating_sum_cond (b : ℕ) : Prop :=
  (11 - b) % 11 = 0

-- The final proposition to be proved.
theorem unique_valid_number : ∃ n, is_form_sixb5 n ∧ is_divisible_by_11 n ∧ n = 605 :=
by {
  sorry
}

end unique_valid_number_l673_673769


namespace product_of_five_consecutive_divisible_by_30_l673_673274

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l673_673274


namespace product_of_five_consecutive_divisible_by_30_l673_673268

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l673_673268


namespace product_of_five_consecutive_integers_divisible_by_240_l673_673283

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l673_673283


namespace smallest_composite_proof_l673_673905

-- Define what it means for a number not to have prime factors less than 15
def no_prime_factors_less_than_15 (n : ℕ) : Prop :=
  ∀ p : ℕ, nat.prime p → p ∣ n → p ≥ 15

-- Define what it means for a number to be the smallest composite number with the above property
def smallest_composite_without_prime_factors_less_than_15 (n : ℕ) : Prop :=
  nat.composite n ∧ no_prime_factors_less_than_15 n ∧
  ∀ m : ℕ, nat.composite m → no_prime_factors_less_than_15 m → n ≤ m

theorem smallest_composite_proof : smallest_composite_without_prime_factors_less_than_15 323 :=
  sorry

end smallest_composite_proof_l673_673905


namespace intersection_length_parabola_line_l673_673514

theorem intersection_length_parabola_line :
  ∀ (x y : ℝ), (y^2 = 4 * x) ∧ (x - y - 1 = 0) → ∃ (A B : ℝ × ℝ), dist A B = 8  :=
begin
  sorry
end

end intersection_length_parabola_line_l673_673514


namespace shirts_left_l673_673119

-- Define the given conditions
def initial_shirts : ℕ := 4 * 12
def fraction_given : ℚ := 1 / 3

-- Define the proof goal
theorem shirts_left (initial_shirts : ℕ) (fraction_given : ℚ) : ℕ :=
let shirts_given := initial_shirts * fraction_given in
initial_shirts - (shirts_given : ℕ) = 32 :=
begin
  -- placeholder for the proof
  sorry
end

end shirts_left_l673_673119


namespace train_crossing_time_l673_673354

-- Lean definitions of the conditions and problem
def train_length : ℝ := 100 -- in meters
def bridge_length : ℝ := 120 -- in meters
def train_speed_kmph : ℝ := 36 -- in km/h

-- Conversion factor from km/h to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)

-- Speed of train in m/s
def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- Total distance to be covered
def total_distance : ℝ := train_length + bridge_length

-- Time to cross the bridge
def time_to_cross (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

-- Theorem stating the proof problem
theorem train_crossing_time : time_to_cross total_distance train_speed_mps = 22 := 
by
  sorry

end train_crossing_time_l673_673354


namespace jacket_spending_l673_673111

def total_spent : ℝ := 14.28
def spent_on_shorts : ℝ := 9.54
def spent_on_jacket : ℝ := 4.74

theorem jacket_spending :
  spent_on_jacket = total_spent - spent_on_shorts :=
by sorry

end jacket_spending_l673_673111


namespace complex_ratio_real_l673_673969

theorem complex_ratio_real (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : ∃ z : ℂ, z = a + b * Complex.I ∧ (z * (1 - 2 * Complex.I)).im = 0) :
  a / b = 1 / 2 :=
sorry

end complex_ratio_real_l673_673969


namespace problem_l673_673987

theorem problem (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) 
  (h1 : 1 / a + 1 / b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) := 
by
  sorry

end problem_l673_673987


namespace largest_divisor_of_five_consecutive_integers_l673_673260

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l673_673260


namespace smallest_composite_no_prime_under_15_correct_l673_673938

-- Define the concept of a composite number
def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

-- Define the concept of having no prime factors less than 15
def has_no_prime_factors_less_than_15 (n : ℕ) : Prop := 
  ∀ p : ℕ, p.prime ∧ p ∣ n → 15 ≤ p

-- Define the smallest composite number with no prime factors less than 15 
def smallest_composite_no_prime_under_15 : ℕ := 289

-- Prove that this is the smallest number satisfying our conditions
theorem smallest_composite_no_prime_under_15_correct : 
  is_composite smallest_composite_no_prime_under_15 ∧
  has_no_prime_factors_less_than_15 smallest_composite_no_prime_under_15 ∧
  ∀ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than_15 n → n ≥ smallest_composite_no_prime_under_15 :=
by 
  sorry

end smallest_composite_no_prime_under_15_correct_l673_673938


namespace school_year_days_l673_673731

theorem school_year_days :
  ∀ (D : ℕ),
  (9 = 5 * D / 100) →
  D = 180 := by
  intro D
  sorry

end school_year_days_l673_673731


namespace complex_polynomial_isosceles_right_triangle_l673_673158

variables (a b : ℂ)
-- Assumptions
variables (x1 x2 : ℂ) (hpoly : x1 * x2 = b ∧ x1 + x2 = -a)
variables (hright : x2 = x1 * complex.I ∨ x1 = x2 * complex.I)
def is_isosceles_right_triangle_with_origin (x1 x2 : ℂ) : Prop :=
  x2 = x1 * complex.I ∨ x1 = x2 * complex.I 

theorem complex_polynomial_isosceles_right_triangle:
  (x1 * x2 = b ∧ x1 + x2 = -a) →
  (is_isosceles_right_triangle_with_origin x1 x2) →
  (a^2 = 2*b ∧ b ≠ 0) :=
by
  intro h1 h2
  sorry

end complex_polynomial_isosceles_right_triangle_l673_673158


namespace part1_part2_l673_673617

noncomputable def a_n (n : Nat) : ℕ := 2^(n-1)

noncomputable def b_n (n : Nat) : ℕ := 2^(n-1) * 2 * n

noncomputable def S3 : ℕ := 7

theorem part1 (S3_eq : S3 = 7) (h : a_n 3 = 2 * a_n 2 ∧ a_n 3 = a_n 1 * 2^2 ∧ a_n 4 = a_n 1 * (2^3)) :
(a_n 1 = 1) ∧ (a_n 2 = 2^1) ∧ (a_n 3 = 2^2) ∧ (a_n 4 = 2^3) := sorry

noncomputable def T_n (n : Nat) : ℕ :=
2 + ∑ i in Finset.range (n - 1), (i + 2) * 2^(i + 2)

theorem part2 (n : Nat) : T_n n = (n-1) * 2^(n+1) + 2 := sorry

end part1_part2_l673_673617


namespace time_to_pass_platform_l673_673794

-- Definitions of the conditions
def train_length : ℝ := 3200
def time_to_cross_tree : ℝ := 60
def speed_of_train := train_length / time_to_cross_tree
def platform_length : ℝ := 2700
def combined_length := train_length + platform_length

-- The proof statement
theorem time_to_pass_platform : combined_length / speed_of_train = 110.6 :=
by
  sorry

end time_to_pass_platform_l673_673794


namespace sum_gcd_lcm_l673_673335

def gcd (a b : ℕ) : ℕ := sorry
def lcm (a b : ℕ) : ℕ := sorry

theorem sum_gcd_lcm (a b : ℕ) : a = 8 ∧ b = 12 → gcd a b + lcm a b = 28 :=
by
  sorry

end sum_gcd_lcm_l673_673335


namespace intersection_of_circles_l673_673625

open EuclideanGeometry

-- Define the acute triangle ABC
variables {A B C L O : Point}
variables {ω : Circle}

-- Conditions: L is on BC, ω is tangent to AB at B' and to AC at C'
axiom is_acute_triangle : IsAcuteTriangle A B C
axiom center_on_bc : OnSegment L B C
axiom tangent_at_ab : ω.tangent_at B' ∧ OnSegment B' A B
axiom tangent_at_ac : ω.tangent_at C' ∧ OnSegment C' A C

-- The circumcenter O of △ABC lies on the shorter arc B'C' of ω
axiom circumcenter_on_shorter_arc : OnShorterArc O B' C' ω

-- Theorem: The circumcircle of △ABC and ω meet at two points
theorem intersection_of_circles : 
  let circumcircle_ABC := Circumcircle A B C in
  (circumcircle_ABC ∩ ω).card = 2 := sorry

end intersection_of_circles_l673_673625


namespace generalization_system_solution_application_system_solution_advanced_system_solution_l673_673159

-- Generalization problem statement
theorem generalization_system_solution (a b : ℝ) :
  (\frac{a}{3} - 1) + 2(\frac{b}{5} + 2) = 4 ∧ 2(\frac{a}{3} - 1) + (\frac{b}{5} + 2) = 5 → a = 9 ∧ b = -5 :=
by sorry

-- Application problem statement
theorem application_system_solution (a_1 a_2 b_1 b_2 c_1 c_2 m n : ℝ) :
  (a_1 * 5 + b_1 * 3 = c_1 ∧ a_2 * 5 + b_2 * 3 = c_2) →
  (a_1 * (m+3) + b_1 * (n-2) = c_1 ∧ a_2 * (m+3) + b_2 * (n-2) = c_2) → (m = 2 ∧ n = 5) :=
by sorry

-- Advanced problem statement
theorem advanced_system_solution (a_1 a_2 b_1 b_2 c_1 c_2 x y : ℝ) :
  3 * a_1 * 3 + 2 * b_1 * 4 = 5 * c_1 ∧ 3 * a_2 * 3 + 2 * b_2 * 4 = 5 * c_2 →
  (a_1 * (9 / 5) + b_1 * (8 / 5) = c_1 ∧ a_2 * (9 / 5) + b_2 * (8 / 5) = c_2) → (x = 9 / 5 ∧ y = 8 / 5) :=
by sorry

end generalization_system_solution_application_system_solution_advanced_system_solution_l673_673159


namespace smallest_composite_no_prime_factors_lt_15_l673_673910

theorem smallest_composite_no_prime_factors_lt_15 (n : ℕ) :
  ∀ n, (∀ p : ℕ, p.prime → p ∣ n → 15 ≤ p) → n = 289 → 
       is_composite n ∧ (∀ m : ℕ, (∀ q : ℕ, q.prime → q ∣ m → 15 ≤ q) → m ≥ 289) :=
by
  intros n hv hn
  -- Proof would go here
  sorry

end smallest_composite_no_prime_factors_lt_15_l673_673910


namespace no_square_possible_equilateral_triangle_possible_l673_673195

/-- Given a set of 20 sticks with lengths 1, 2, ..., 20,
prove that it is impossible to form a square using these sticks
without breaking any of them. -/
theorem no_square_possible : ¬ ∃ (f : Fin 20 → ℕ), 
  (∀ i, f i = i.val + 1) ∧
  (∃ four_sides : list (Fin 20), 
    four_sides.length = 4 ∧ 
    (∀ side ∈ four_sides, ∑ i in side, f i = 210 / 4)) :=
by
  sorry

/-- Given a set of 20 sticks with lengths 1, 2, ..., 20,
prove that it is possible to form an equilateral triangle using these sticks
without breaking any of them. -/
theorem equilateral_triangle_possible : ∃ (f : Fin 20 → ℕ),
  (∀ i, f i = i.val + 1) ∧ 
  (∃ three_sides : list (Fin 20), 
    three_sides.length = 3 ∧ 
    (∀ side ∈ three_sides, ∑ i in side, f i = 210 / 3)) :=
by
  sorry

end no_square_possible_equilateral_triangle_possible_l673_673195


namespace micrometer_conversion_l673_673642

theorem micrometer_conversion :
  (0.01 * (1 * 10 ^ (-6))) = (1 * 10 ^ (-8)) :=
by 
  -- sorry is used to skip the actual proof but ensure the theorem is recognized
  sorry

end micrometer_conversion_l673_673642


namespace trig_identity_sum_l673_673364

-- Define the trigonometric functions and their properties
def sin_210_eq : Real.sin (210 * Real.pi / 180) = - Real.sin (30 * Real.pi / 180) := by
  sorry

def cos_60_eq : Real.cos (60 * Real.pi / 180) = Real.sin (30 * Real.pi / 180) := by
  sorry

-- The goal is to prove that the sum of these specific trigonometric values is 0
theorem trig_identity_sum : Real.sin (210 * Real.pi / 180) + Real.cos (60 * Real.pi / 180) = 0 := by
  rw [sin_210_eq, cos_60_eq]
  sorry

end trig_identity_sum_l673_673364


namespace exactly_one_correct_proposition_l673_673510

theorem exactly_one_correct_proposition :
  ¬(∃ (S : Set (X : Type)), true) ∧
  ({a, b, c, d} = {d, c, a, b}) ∧
  ¬({1, 2, 3} ∪ {3, 4} = {1, 2, 3, 3, 4}) ∧
  ¬(0 ∈ (∅ : Set ℕ)) → 
  let propositions := [false, true, false, false] in
  (propositions.filter id).length = 1 :=
by
  intro h
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2
  sorry

end exactly_one_correct_proposition_l673_673510


namespace smallest_composite_no_prime_factors_less_than_15_l673_673926

theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n, (n = 289) ∧ (n > 1) ∧ (¬ Nat.Prime n) ∧ (∀ p : ℕ, Nat.Prime p → p ∣ n → 15 ≤ p) :=
by
  use 289
  split
  case left => rfl
  case right =>
    split
    case left => exact Nat.lt_succ_self 288
    case right =>
      split
      case left =>
        have composite : ¬ Nat.Prime 289 := by
          intro h
          have h_div : 17 ∣ 289 := by norm_num
          exact h.not_divs_self (dec_trivial : 17 * 17 = 289)
        exact composite
      case right =>
        intros p h_prime h_div
        have : p ∣ 17 := by
          have factorization : 289 = 17 * 17 := by norm_num
          have dvd_product : p ∣ 289 := by { use 17, exact factorization.symm }
          exact Nat.Prime.dvd_mul h_prime dvd_product
        have prime_eq_17 : p = 17 := by
          exact Nat.Prime.eq_of_dvd_of_ne h_prime (by norm_num) this
        linarith

end smallest_composite_no_prime_factors_less_than_15_l673_673926


namespace tilly_total_profit_l673_673744

theorem tilly_total_profit :
  let bags_sold := 100
  let selling_price_per_bag := 10
  let buying_price_per_bag := 7
  let profit_per_bag := selling_price_per_bag - buying_price_per_bag
  let total_profit := bags_sold * profit_per_bag
  total_profit = 300 :=
by
  let bags_sold := 100
  let selling_price_per_bag := 10
  let buying_price_per_bag := 7
  let profit_per_bag := selling_price_per_bag - buying_price_per_bag
  let total_profit := bags_sold * profit_per_bag
  sorry

end tilly_total_profit_l673_673744


namespace largest_divisor_of_five_consecutive_integers_l673_673256

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l673_673256


namespace candy_last_days_l673_673363

theorem candy_last_days (candy_neighbors candy_sister candy_per_day : ℕ)
  (h1 : candy_neighbors = 5)
  (h2 : candy_sister = 13)
  (h3 : candy_per_day = 9):
  (candy_neighbors + candy_sister) / candy_per_day = 2 :=
by
  sorry

end candy_last_days_l673_673363


namespace surface_area_of_sphere_l673_673013

theorem surface_area_of_sphere {S A B C : Type*} (r : ℝ) (O : sphere S A B C r) 
  (h1: SA ⊥ plane ABC) (h2: AB ⊥ BC) (h3: SA = 1) (h4: AB = 1) 
  (h5: BC = sqrt 2) : surface_area O = 4 * π :=
by
  sorry

end surface_area_of_sphere_l673_673013


namespace proof_problem_l673_673583

noncomputable def M : ℕ := 50
noncomputable def T : ℕ := M + Nat.div M 10
noncomputable def W : ℕ := 2 * (M + T)
noncomputable def Th : ℕ := W / 2
noncomputable def total_T_T_W_Th : ℕ := T + W + Th
noncomputable def total_M_T_W_Th : ℕ := M + total_T_T_W_Th
noncomputable def F_S_sun : ℕ := Nat.div (450 - total_M_T_W_Th) 3
noncomputable def car_tolls : ℕ := 150 * 2
noncomputable def bus_tolls : ℕ := 150 * 5
noncomputable def truck_tolls : ℕ := 150 * 10
noncomputable def total_toll : ℕ := car_tolls + bus_tolls + truck_tolls

theorem proof_problem :
  (total_T_T_W_Th = 370) ∧
  (F_S_sun = 10) ∧
  (total_toll = 2550) := by
  sorry

end proof_problem_l673_673583


namespace smallest_positive_period_f_max_value_f_interval_min_value_f_interval_l673_673037

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x - Real.cos x) + 1

theorem smallest_positive_period_f : ∃ k > 0, ∀ x, f (x + k) = f x := 
sorry

theorem max_value_f_interval : ∃ x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 4), f x = Real.sqrt 2 :=
sorry

theorem min_value_f_interval : ∃ x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 4), f x = -1 :=
sorry

end smallest_positive_period_f_max_value_f_interval_min_value_f_interval_l673_673037


namespace B_can_finish_in_9_days_l673_673377

-- Define the work done by A in one day
def A_work_rate : ℝ := 1 / 4

-- Define the work done by C in one day
def C_work_rate : ℝ := 1 / 7.2

-- Define the combined work rate when A, B, and C work together
def combined_work_rate : ℝ := 1 / 2

-- Define the work done by B in one day
def B_work_rate := combined_work_rate - A_work_rate - C_work_rate

-- The main theorem stating that B can finish the work alone in 9 days
theorem B_can_finish_in_9_days : 1 / B_work_rate = 9 := by
  sorry

end B_can_finish_in_9_days_l673_673377


namespace connected_graph_edges_ge_verts_minus_one_l673_673664

variables {V : Type*} {E : Type*}

-- Define a connected simple graph G = (V, E) with vertices V and edges E
structure ConnectedGraph (V : Type*) :=
(verts : Type*)
(edges : set (verts × verts))
(is_connected : ∀ v1 v2 : verts, ∃ (path : list verts), v1 ∈ path ∧ v2 ∈ path ∧ (∀ i ∈ list.zip (path.init) (path.tail), (i.fst, i.snd) ∈ edges ∨ (i.snd, i.fst) ∈ edges))

-- Define the predicate for the number of edges
def num_edges {V : Type*} {E : set (V × V)} : ℕ := set.card E
def num_verts {V : Type*} (G : ConnectedGraph V) := set.card G.verts

-- Statement of the theorem
theorem connected_graph_edges_ge_verts_minus_one (G : ConnectedGraph V) :
  num_edges G.edges ≥ num_verts G - 1 := sorry

end connected_graph_edges_ge_verts_minus_one_l673_673664


namespace det_is_zero_l673_673425

noncomputable def matrix_det : ℝ → ℝ → ℝ :=
  λ a b, Matrix.det ![
    ![1, Real.sin (a - b), Real.sin a],
    ![Real.sin (a - b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ]

theorem det_is_zero (a b : ℝ) : matrix_det a b = 0 :=
  sorry

end det_is_zero_l673_673425


namespace complex_number_solution_l673_673145

theorem complex_number_solution (z : ℂ) (i : ℂ) (hi : i * i = -1) (h : i * z = 1) : z = -i :=
by
  -- Mathematical proof will be here
  sorry

end complex_number_solution_l673_673145


namespace count_divisibles_by_8_in_range_100_250_l673_673525

theorem count_divisibles_by_8_in_range_100_250 : 
  let lower_bound := 100
  let upper_bound := 250
  let divisor := 8
  ∃ n : ℕ, (∀ x : ℕ, lower_bound ≤ x ∧ x ≤ upper_bound ∧ x % divisor = 0 ↔ (n = 19)) :=
begin
  let lower_bound := 100,
  let upper_bound := 250,
  let divisor := 8,
  let first_multiple := ((lower_bound + divisor - 1) / divisor) * divisor,
  let last_multiple := (upper_bound / divisor) * divisor,
  let first_index := first_multiple / divisor,
  let last_index := last_multiple / divisor,
  let n := (last_index - first_index + 1),
  use n,
  intros x,
  split,
  { intro hx,
    exact ⟨nat.exists_eq_add_of_le hx.1, nat.exists_eq_add_of_le hx.2.1, nat.exists_eq_of_divisible hx.2.2⟩ },
  { intro hn,
    rw hn,
    refine ⟨_, _, _⟩,
    sorry
  }
end

end count_divisibles_by_8_in_range_100_250_l673_673525


namespace binomial_sum_l673_673955

theorem binomial_sum (n : ℕ) : (∑ k in Finset.range (n + 1), Nat.choose n k) = 2^n := 
by sorry

end binomial_sum_l673_673955


namespace ofelia_savings_december_l673_673646

theorem ofelia_savings_december :
  let savings (n : ℕ) : ℝ :=
    match n with
    | 1 => 10
    | n + 1 => 2.5 * (savings n)
  in
  savings 12 - 20 = 238398.58 :=
by
  -- This is where the proof would go.
  sorry

end ofelia_savings_december_l673_673646


namespace num_points_on_curve_C_distance_from_line_l_l673_673187

theorem num_points_on_curve_C_distance_from_line_l :
  (∃ (p : ℝ → ℝ), ∀ θ : ℝ, p θ = 2 + 3 * cos θ ∧ p θ = -1 + 3 * sin θ) →
  (∃ (l : ℝ → ℝ), ∀ θ : ℝ, l θ = p θ * (cos θ) - 3 * p θ * (sin θ) + 2 = 0) →
  2 := sorry

end num_points_on_curve_C_distance_from_line_l_l673_673187


namespace find_least_f_l673_673190

open BigOperators

variables {n : ℕ} (a : Fin n → ℝ)

def satisfies_conditions (a : Fin n → ℝ) : Prop :=
  (n ≥ 3) ∧
  (∑ i, a i = 0) ∧
  (∀ k : Fin (n - 2), 2 * a (k + 1) ≤ a k + a (k + 2))

noncomputable def f (n : ℕ) : ℝ :=
  (n + 1) / (n - 1)

theorem find_least_f (a : Fin n → ℝ) (n_ge_3 : n ≥ 3) (sum_zero : ∑ i, a i = 0)
  (bounded : ∀ k : Fin (n - 2), 2 * a (k + 1) ≤ a k + a (k + 2)) :
  ∀ k : Fin n, |a k| ≤ f n * max |a 0| |a (n - 1)| :=
sorry

end find_least_f_l673_673190


namespace cost_price_l673_673062

theorem cost_price (C : ℝ) : 
  (0.05 * C = 350 - 340) → C = 200 :=
by
  assume h1 : 0.05 * C = 10
  sorry

end cost_price_l673_673062


namespace eval_sum_zero_l673_673138

def g (x : ℝ) : ℝ := x^2 * (1 - x)^2

theorem eval_sum_zero : 
  (∑ k in finset.range (2021 + 1), (-1)^(k+1) * g (k / 2021)) = 0 :=
sorry

end eval_sum_zero_l673_673138


namespace value_of_A_l673_673593

theorem value_of_A :
  ∃ A : ℝ,
  let P := (0, 2)
  let Q := (A, 8)
  let midpoint := (A / 2, 5)
  let line := (λ x, 0.5 * x + 2)
  (line 4 = 4 ∧ (midpoint = (A / 2, 5)) ∧ (line = λ x, 0.5 * x + 2)) → A = -3 :=
by
  sorry

end value_of_A_l673_673593


namespace brocard_theorem_l673_673627

noncomputable section

open EuclideanGeometry

variables {A B C D P Q M O : Type} 
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace P] [MetricSpace Q] [MetricSpace M] [MetricSpace O]
variables {R : ℝ} {a b c : ℝ}

def is_cyclic_quadrilateral (A B C D : Type) (O : Type) (R : ℝ) : Prop :=
  ∀ (P Q M : Type), 
    intersection_of_diagonals (A) (B) (C) (D) O P Q M → 
    extends (A) (B) (C) (D) O P Q M → 
    distances_to_center (P Q M : Type) (a : ℝ) (b : ℝ) (c : ℝ) O → 
    orthocenter_triangle_PQM (P Q M O : Type) 

theorem brocard_theorem 
  (cyclic_quad : is_cyclic_quadrilateral A B C D O R)
  {P Q M : Type}
  (h1 : P = intersection_of_diagonals A B C D)
  (h2 : Q = intersection_of_diagonals A B C D)
  (h3 : M = intersection_of_extends A B C D)
  (h4 : distances_to_center P Q M a b c O)
  : orthocenter_triangle_PQM P Q M = O := 
sorry

end brocard_theorem_l673_673627


namespace pow_eq_from_exponent_l673_673528

theorem pow_eq_from_exponent (x : ℝ) : 5^x = 625 → x = 4 :=
by
  sorry

end pow_eq_from_exponent_l673_673528


namespace cow_manure_growth_percentage_l673_673112

variable (control_height bone_meal_growth_percentage cow_manure_height : ℝ)
variable (bone_meal_height : ℝ := bone_meal_growth_percentage * control_height)
variable (percentage_growth : ℝ := (cow_manure_height / bone_meal_height) * 100)

theorem cow_manure_growth_percentage 
  (h₁ : control_height = 36)
  (h₂ : bone_meal_growth_percentage = 1.25)
  (h₃ : cow_manure_height = 90) :
  percentage_growth = 200 :=
by {
  sorry
}

end cow_manure_growth_percentage_l673_673112


namespace largest_divisor_of_five_consecutive_integers_l673_673255

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l673_673255


namespace division_of_decimals_l673_673053

theorem division_of_decimals :
  (0.1 / 0.001 = 100) ∧ (1 / 0.01 = 100) := by
  sorry

end division_of_decimals_l673_673053


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673216

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673216


namespace largest_divisor_of_consecutive_product_l673_673222

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l673_673222


namespace smallest_composite_no_prime_factors_less_than_15_l673_673921

theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n, (n = 289) ∧ (n > 1) ∧ (¬ Nat.Prime n) ∧ (∀ p : ℕ, Nat.Prime p → p ∣ n → 15 ≤ p) :=
by
  use 289
  split
  case left => rfl
  case right =>
    split
    case left => exact Nat.lt_succ_self 288
    case right =>
      split
      case left =>
        have composite : ¬ Nat.Prime 289 := by
          intro h
          have h_div : 17 ∣ 289 := by norm_num
          exact h.not_divs_self (dec_trivial : 17 * 17 = 289)
        exact composite
      case right =>
        intros p h_prime h_div
        have : p ∣ 17 := by
          have factorization : 289 = 17 * 17 := by norm_num
          have dvd_product : p ∣ 289 := by { use 17, exact factorization.symm }
          exact Nat.Prime.dvd_mul h_prime dvd_product
        have prime_eq_17 : p = 17 := by
          exact Nat.Prime.eq_of_dvd_of_ne h_prime (by norm_num) this
        linarith

end smallest_composite_no_prime_factors_less_than_15_l673_673921


namespace arc_length_ln_1_minus_x_squared_l673_673782

noncomputable def curve_eq (x : ℝ) : ℝ := real.log (1 - x^2)

theorem arc_length_ln_1_minus_x_squared :
  ∫ x in (0 : ℝ) .. (1/4 : ℝ), sqrt (1 + (deriv curve_eq x)^2) = (1/2) * real.log (5/3) + (1/4) :=
by
  sorry

end arc_length_ln_1_minus_x_squared_l673_673782


namespace two_digit_number_square_equals_cube_of_sum_of_digits_l673_673882

theorem two_digit_number_square_equals_cube_of_sum_of_digits : ∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧
  let A := n / 10 in
  let B := n % 10 in
  A ≠ B ∧ n^2 = (A + B)^3 :=
begin
  use 27,
  split, { dec_trivial },
  split, { dec_trivial },
  let A := 27 / 10,
  let B := 27 % 10,
  split,
  { exact dec_trivial },
  simp [A, B],
  exact dec_trivial,
end

end two_digit_number_square_equals_cube_of_sum_of_digits_l673_673882


namespace largest_divisor_of_5_consecutive_integers_l673_673304

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l673_673304


namespace largest_divisor_of_5_consecutive_integers_l673_673308

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l673_673308


namespace abs_eq_condition_l673_673175

theorem abs_eq_condition (a b : ℝ) : |a - b| = |a - 1| + |b - 1| ↔ (a - 1) * (b - 1) ≤ 0 :=
sorry

end abs_eq_condition_l673_673175


namespace max_digit_sum_digital_watch_l673_673816

def max_hour_sum : ℕ :=
  list.maximum ((list.Ico 0 24).map (λ h, (h / 10) + (h % 10))) |>.get_or_else 0

def max_minute_sum : ℕ :=
  list.maximum ((list.Ico 0 60).map (λ m, (m / 10) + (m % 10))) |>.get_or_else 0

theorem max_digit_sum_digital_watch : max_hour_sum + max_minute_sum = 24 := by
  -- proof goes here
  sorry

end max_digit_sum_digital_watch_l673_673816


namespace distance_inequality_l673_673980

variables (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]

open_locale big_operators

theorem distance_inequality (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
  (dist : Π (P Q : A), ℝ) :
  dist A C ^ 2 + dist B D ^ 2 + dist A D ^ 2 + dist B C ^ 2 ≥ dist A B ^ 2 + dist C D ^ 2 :=
sorry

end distance_inequality_l673_673980


namespace product_of_five_consecutive_divisible_by_30_l673_673270

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l673_673270


namespace right_triangle_shorter_leg_l673_673569

theorem right_triangle_shorter_leg :
  ∃ (a b : ℤ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 16 :=
by
  sorry

end right_triangle_shorter_leg_l673_673569


namespace find_product_of_x_plus_one_and_x_minus_one_l673_673018

theorem find_product_of_x_plus_one_and_x_minus_one (x : ℝ) (h : 3^x + 3^x + 3^x + 3^x = 324) : 
  (x + 1) * (x - 1) = 3 :=
sorry

end find_product_of_x_plus_one_and_x_minus_one_l673_673018


namespace find_a_for_shared_foci_l673_673977

noncomputable def same_foci_of_conic_sections : Prop :=
  ∀ (a : ℝ), a > 0 →
    (let c_ellipse := λ a, (4 - a^2) in
     let c_hyperbola := λ a, (a + 2) in
     c_ellipse a = c_hyperbola a → a = 1)

theorem find_a_for_shared_foci : same_foci_of_conic_sections :=
by
  intro a ha
  unfold same_foci_of_conic_sections
  intro H
  have h := congr_fun H a
  sorry

end find_a_for_shared_foci_l673_673977


namespace leaves_shed_second_week_l673_673836

theorem leaves_shed_second_week (P : ℝ) :
  let initial_leaves := 1000
  let leaves_after_first_week := initial_leaves - (2 / 5 * initial_leaves)
  let leaves_shed_second_week := P / 100 * leaves_after_first_week
  let leaves_shed_third_week := 3 / 4 * leaves_shed_second_week in
  (leaves_shed_third_week < 90) → (P < 20) :=
by
  let initial_leaves := 1000
  let leaves_after_first_week := initial_leaves - (2 / 5 * initial_leaves)
  let leaves_shed_second_week := P / 100 * leaves_after_first_week
  let leaves_shed_third_week := 3 / 4 * leaves_shed_second_week
  sorry

end leaves_shed_second_week_l673_673836


namespace stratified_sampling_correct_l673_673838

-- Define the population sizes
def elderly_people : ℕ := 27
def middle_aged_people : ℕ := 54
def young_people : ℕ := 81

-- Define the total sample size needed
def sample_size : ℕ := 36

-- Define the total population
def total_population : ℕ := elderly_people + middle_aged_people + young_people

-- Define the probability of being sampled
def probability_of_being_sampled : ℚ := sample_size / total_population

-- Calculate sample sizes for each group
def elderly_sample_size : ℕ := elderly_people * (probability_of_being_sampled).to_rat
def middle_aged_sample_size : ℕ := middle_aged_people * (probability_of_being_sampled).to_rat
def young_sample_size : ℕ := young_people * (probability_of_being_sampled).to_rat

-- Lean 4 statement to prove the correctness of the sample sizes
theorem stratified_sampling_correct :
  elderly_sample_size = 6 ∧
  middle_aged_sample_size = 12 ∧
  young_sample_size = 18 :=
by
  have h1 : probability_of_being_sampled = 2 / 9, simp [probability_of_being_sampled, sample_size, total_population],
  have h2 : elderly_sample_size = 6, by simp [elderly_sample_size, elderly_people, h1],
  have h3 : middle_aged_sample_size = 12, by simp [middle_aged_sample_size, middle_aged_people, h1],
  have h4 : young_sample_size = 18, by simp [young_sample_size, young_people, h1],
  exact ⟨h2, h3, h4⟩

end stratified_sampling_correct_l673_673838


namespace number_of_pairs_l673_673674

-- Define the sets of balls
def green_balls := {G1, G2, G3}
def red_balls := {R1, R2, R3, R4}
def blue_balls := {B1, B2, B3, B4, B5}

-- Define the statement
theorem number_of_pairs (G1 G2 G3 R1 R2 R3 R4 B1 B2 B3 B4 B5 : Type) :
  green_balls = {G1, G2, G3} →
  red_balls = {R1, R2, R3, R4} →
  blue_balls = {B1, B2, B3, B4, B5} →
  (∃ pairs : finset (finset (G1 ⊕ G2 ⊕ G3 ⊕ R1 ⊕ R2 ⊕ R3 ⊕ R4 ⊕ B1 ⊕ B2 ⊕ B3 ⊕ B4 ⊕ B5)),
    pairs.card = 6 ∧ ∀ pair ∈ pairs, ∃ G R B, pair = {G, R, B} ∧ G ∈ green_balls ∧ R ∈ red_balls ∧ B ∈ blue_balls) →
  (∑ (n : ℕ) in finset.range 1440, 1) = 1440 := 
by sorry

end number_of_pairs_l673_673674


namespace central_number_value_l673_673090

open Nat

def grid (i j: Nat) (h: i < 5) (w: j < 5) : Nat := sorry -- grid definition placeholder

theorem central_number_value
(h_sum_all: ∑ i in Finset.range 5, ∑ j in Finset.range 5, grid i j (Fin.isLt i 5) (Fin.isLt j 5) = 200)
(h_sum_1x3: ∀ (i j: Nat) (h₁: i < 5) (h₂ j₂: j < 3), 
  ∑ k in Finset.range 3, grid i (j + k) h₁ (add_lt_add_of_le_of_lt (Nat.le_add_right j k) (lt_add_of_lt_of_le k 2 h₂)) = 23):
  grid 2 2 (by decide) (by decide) = 16 := sorry

end central_number_value_l673_673090


namespace probability_of_gcd_one_is_13_over_14_l673_673734

open Finset

noncomputable def probability_gcd_one : ℚ :=
let s := {1, 2, 3, 4, 5, 6, 7, 8}
let subsetsOfThree := s.powerset.filter (λ t, t.card = 3)
let nonRelativelyPrimeSubsets := {(t : Finset ℕ) ∈ subsetsOfThree | (∀ a b c ∈ t, gcd (gcd a b) c ≠ 1)}
let totalSubsets := subsetsOfThree.card
let nonRelativelyPrimeCount := nonRelativelyPrimeSubsets.card
in 1 - (nonRelativelyPrimeCount / totalSubsets : ℚ)

theorem probability_of_gcd_one_is_13_over_14 :
  probability_gcd_one = 13 / 14 := by sorry

end probability_of_gcd_one_is_13_over_14_l673_673734


namespace largest_divisor_of_product_of_5_consecutive_integers_l673_673242

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l673_673242


namespace vector_combination_l673_673463

def vec_a : ℝ × ℝ := (3,1)
def vec_b : ℝ × ℝ := (-2,5)

theorem vector_combination : 3 • vec_a - 2 • vec_b = (13, -7) := 
by
  sorry

end vector_combination_l673_673463


namespace instantaneous_velocity_at_t10_l673_673391

def displacement (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 1

def velocity (t : ℝ) : ℝ := (deriv displacement) t

theorem instantaneous_velocity_at_t10 : velocity 10 = 58 := 
by
  -- Placeholder for the proof
  sorry

end instantaneous_velocity_at_t10_l673_673391


namespace number_of_elements_in_M_l673_673046

-- Define the set M
def M := {x : Fin 5 → Int | ∀ i, x i ∈ [-1, 0, 1]}

-- Define the condition
def condition (x : Fin 5 → Int) : Bool :=
  1 ≤ (Finset.univ.sum (λ i, Int.natAbs (x i))) ∧ 
  (Finset.univ.sum (λ i, Int.natAbs (x i))) ≤ 3

-- Define the theorem statement
theorem number_of_elements_in_M :
  (∑ x in M, if condition x then 1 else 0) = 130 := 
sorry

end number_of_elements_in_M_l673_673046


namespace tan_alpha_eq_neg_four_thirds_l673_673467

theorem tan_alpha_eq_neg_four_thirds
  (α : ℝ) (hα1 : 0 < α ∧ α < π) 
  (hα2 : Real.sin α + Real.cos α = 1 / 5) : 
  Real.tan α = - 4 / 3 := 
  sorry

end tan_alpha_eq_neg_four_thirds_l673_673467


namespace largest_divisor_of_consecutive_product_l673_673231

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l673_673231


namespace transformed_line_equation_l673_673041

theorem transformed_line_equation {A B C x₀ y₀ : ℝ} 
    (h₀ : ¬(A = 0 ∧ B = 0)) 
    (h₁ : A * x₀ + B * y₀ + C = 0) : 
    ∀ {x y : ℝ}, A * x + B * y + C = 0 ↔ A * (x - x₀) + B * (y - y₀) = 0 :=
by
    sorry

end transformed_line_equation_l673_673041


namespace product_of_five_consecutive_integers_divisible_by_240_l673_673284

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l673_673284


namespace age_ratio_l673_673351

theorem age_ratio (A B C : ℕ) (h1 : A = B + 2) (h2 : A + B + C = 27) (h3 : B = 10) : B / C = 2 :=
by
  sorry

end age_ratio_l673_673351


namespace smallest_composite_no_prime_factors_less_than_15_l673_673920

theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n, (n = 289) ∧ (n > 1) ∧ (¬ Nat.Prime n) ∧ (∀ p : ℕ, Nat.Prime p → p ∣ n → 15 ≤ p) :=
by
  use 289
  split
  case left => rfl
  case right =>
    split
    case left => exact Nat.lt_succ_self 288
    case right =>
      split
      case left =>
        have composite : ¬ Nat.Prime 289 := by
          intro h
          have h_div : 17 ∣ 289 := by norm_num
          exact h.not_divs_self (dec_trivial : 17 * 17 = 289)
        exact composite
      case right =>
        intros p h_prime h_div
        have : p ∣ 17 := by
          have factorization : 289 = 17 * 17 := by norm_num
          have dvd_product : p ∣ 289 := by { use 17, exact factorization.symm }
          exact Nat.Prime.dvd_mul h_prime dvd_product
        have prime_eq_17 : p = 17 := by
          exact Nat.Prime.eq_of_dvd_of_ne h_prime (by norm_num) this
        linarith

end smallest_composite_no_prime_factors_less_than_15_l673_673920


namespace largest_divisor_of_consecutive_product_l673_673227

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l673_673227


namespace meal_serving_problem_l673_673749

/-
Twelve people sit down for dinner where there are three choices of meals: beef, chicken, and fish.
Four people order beef, four people order chicken, and four people order fish.
The waiter serves the twelve meals in random order.
We need to find the number of ways in which the waiter could serve the meals so that exactly two people receive the type of meal ordered by them.
-/
theorem meal_serving_problem :
    ∃ (n : ℕ), n = 12210 ∧
    (∃ (people : Fin 12 → char), 
        (∀ i : Fin 4, people i = 'B') ∧ 
        (∀ i : Fin 4, people (i + 4) = 'C') ∧ 
        (∀ i : Fin 4, people (i + 8) = 'F') ∧ 
        (∃ (served : Fin 12 → char), 
            (∃ (correct : Fin 12), set.range correct ⊆ {0, 1} ∧
            (∀ i : Fin 12, (served i = people correct i) ↔ (i ∈ {0, 1}) = true)) ∧
            (related_permutations served people))
    )
    sorry

end meal_serving_problem_l673_673749


namespace magnitude_of_sum_of_perpendicular_vectors_l673_673522

noncomputable def vector_magnitude (x₁ x₂ y₁ y₂ : ℝ) : ℝ :=
  real.sqrt ((y₁ - x₁)^2 + (y₂ - x₂)^2)

theorem magnitude_of_sum_of_perpendicular_vectors :
  ∀ (t : ℝ), let a := (6, 2) in let b := (t, 3) in a.1 * b.1 + a.2 * b.2 = 0 → vector_magnitude (6 + t) (2 + 3) 0 0 = 2 * real.sqrt 5 :=
by
  sorry -- Proof not required

end magnitude_of_sum_of_perpendicular_vectors_l673_673522


namespace ratio_b_to_c_l673_673349

-- Define the ages of a, b, and c as A, B, and C respectively
variables (A B C : ℕ)

-- Given conditions
def condition1 := A = B + 2
def condition2 := B = 10
def condition3 := A + B + C = 27

-- The question: Prove the ratio of b's age to c's age is 2:1
theorem ratio_b_to_c : condition1 ∧ condition2 ∧ condition3 → B / C = 2 := 
by
  sorry

end ratio_b_to_c_l673_673349


namespace largest_divisor_of_consecutive_product_l673_673230

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l673_673230


namespace find_x_l673_673337

theorem find_x :
  ∃ x : ℝ, (5 * 0.85) / x - (8 * 2.25) = 5.5 ∧ x ≈ 0.1808510638 :=
by
  sorry

end find_x_l673_673337


namespace averages_correct_l673_673117

variables (marksEnglish totalEnglish marksMath totalMath marksPhysics totalPhysics 
           marksChemistry totalChemistry marksBiology totalBiology 
           marksHistory totalHistory marksGeography totalGeography : ℕ)

variables (avgEnglish avgMath avgPhysics avgChemistry avgBiology avgHistory avgGeography : ℚ)

def Kamal_average_english : Prop :=
  marksEnglish = 76 ∧ totalEnglish = 120 ∧ avgEnglish = (marksEnglish / totalEnglish) * 100

def Kamal_average_math : Prop :=
  marksMath = 65 ∧ totalMath = 150 ∧ avgMath = (marksMath / totalMath) * 100

def Kamal_average_physics : Prop :=
  marksPhysics = 82 ∧ totalPhysics = 100 ∧ avgPhysics = (marksPhysics / totalPhysics) * 100

def Kamal_average_chemistry : Prop :=
  marksChemistry = 67 ∧ totalChemistry = 80 ∧ avgChemistry = (marksChemistry / totalChemistry) * 100

def Kamal_average_biology : Prop :=
  marksBiology = 85 ∧ totalBiology = 100 ∧ avgBiology = (marksBiology / totalBiology) * 100

def Kamal_average_history : Prop :=
  marksHistory = 92 ∧ totalHistory = 150 ∧ avgHistory = (marksHistory / totalHistory) * 100

def Kamal_average_geography : Prop :=
  marksGeography = 58 ∧ totalGeography = 75 ∧ avgGeography = (marksGeography / totalGeography) * 100

theorem averages_correct :
  ∀ (marksEnglish totalEnglish marksMath totalMath marksPhysics totalPhysics 
      marksChemistry totalChemistry marksBiology totalBiology 
      marksHistory totalHistory marksGeography totalGeography : ℕ),
  ∀ (avgEnglish avgMath avgPhysics avgChemistry avgBiology avgHistory avgGeography : ℚ),
  Kamal_average_english marksEnglish totalEnglish avgEnglish →
  Kamal_average_math marksMath totalMath avgMath →
  Kamal_average_physics marksPhysics totalPhysics avgPhysics →
  Kamal_average_chemistry marksChemistry totalChemistry avgChemistry →
  Kamal_average_biology marksBiology totalBiology avgBiology →
  Kamal_average_history marksHistory totalHistory avgHistory →
  Kamal_average_geography marksGeography totalGeography avgGeography →
  avgEnglish = 63.33 ∧ avgMath = 43.33 ∧ avgPhysics = 82 ∧
  avgChemistry = 83.75 ∧ avgBiology = 85 ∧ avgHistory = 61.33 ∧ avgGeography = 77.33 :=
by
  sorry

end averages_correct_l673_673117


namespace total_workers_l673_673358

-- Definitions for the conditions in the problem
def avg_salary_all : ℝ := 8000
def num_technicians : ℕ := 7
def avg_salary_technicians : ℝ := 18000
def avg_salary_non_technicians : ℝ := 6000

-- Main theorem stating the total number of workers
theorem total_workers (W : ℕ) :
  (7 * avg_salary_technicians + (W - 7) * avg_salary_non_technicians = W * avg_salary_all) → W = 42 :=
by
  sorry

end total_workers_l673_673358


namespace four_digit_palindromic_count_odd_digit_palindromic_count_l673_673709

-- Definition of a palindromic number
def is_palindromic (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

-- Number of four-digit palindromic numbers
theorem four_digit_palindromic_count : (card {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ is_palindromic n}) = 90 := by
  sorry

-- Number of palindromic numbers with 2n+1 digits
theorem odd_digit_palindromic_count (n : ℕ) (hn : 0 < n) : 
  (card {x : ℕ | (10^n) ≤ x ∧ x < 10^(n+1) ∧ is_palindromic x}) = 9 * 10^n := by
  sorry

end four_digit_palindromic_count_odd_digit_palindromic_count_l673_673709


namespace arithmetic_expression_evaluation_l673_673592

theorem arithmetic_expression_evaluation : 
  ∃ (a b c d e f : Float),
  a - b * c / d + e = 0 ∧
  a = 5 ∧ b = 4 ∧ c = 3 ∧ d = 2 ∧ e = 1 := sorry

end arithmetic_expression_evaluation_l673_673592


namespace find_side_length_b_of_triangle_l673_673502

/-- Given a triangle ABC with angles satisfying A:B:C = 1:2:3, opposite sides a, b, and c, and
    given a = 1, c = 2, find the length of side b. -/
noncomputable def triangle_side_length_b : ℝ := 
  let A := 1 * Real.pi / 6 in
  let B := 2 * Real.pi / 6 in
  let C := 3 * Real.pi / 6 in
  let a := 1 in 
  let c := 2 in
  Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B)
  
theorem find_side_length_b_of_triangle : triangle_side_length_b = Real.sqrt 3 := 
by {
  sorry
}

end find_side_length_b_of_triangle_l673_673502


namespace no_such_function_exists_l673_673658

theorem no_such_function_exists (f : ℤ → ℤ) (h : ∀ m n : ℤ, f (m + f n) = f m - n) : false :=
sorry

end no_such_function_exists_l673_673658


namespace unique_valid_number_l673_673768

-- Define the form of the three-digit number.
def is_form_sixb5 (n : ℕ) : Prop :=
  ∃ b : ℕ, b < 10 ∧ n = 600 + 10 * b + 5

-- Define the condition for divisibility by 11.
def is_divisible_by_11 (n : ℕ) : Prop :=
  (n % 11 = 0)

-- Define the alternating sum property for our specific number format.
def alternating_sum_cond (b : ℕ) : Prop :=
  (11 - b) % 11 = 0

-- The final proposition to be proved.
theorem unique_valid_number : ∃ n, is_form_sixb5 n ∧ is_divisible_by_11 n ∧ n = 605 :=
by {
  sorry
}

end unique_valid_number_l673_673768


namespace area_triangle_ABC_l673_673079

-- Define the problem's conditions and goal in a Lean proof statement
theorem area_triangle_ABC :
  ∀ (a b c : ℝ) (A B C : ℝ),
  b * Real.cos C + c * Real.cos B = 2 →
  a * Real.cos C + c * Real.cos A = 2 →
  a * Real.cos C + Real.sqrt 3 * a * Real.sin C = b + c →
  a = 2 ∧ b = 2 ∧ A = B ∧ B = C →
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 :=
by
  intros a b c A B C h1 h2 h3 h4
  cases h4 with ha hh4
  cases hh4 with hb hh4
  cases hh4 with hAB hBC
  sorry

end area_triangle_ABC_l673_673079


namespace isosceles_triangle_if_root_neg_one_right_triangle_if_equal_roots_roots_of_equation_if_equilateral_l673_673008

-- Part 1
theorem isosceles_triangle_if_root_neg_one (a b c x : ℝ) (h : x = -1) 
  (h_eq : (a + c) * x^2 + 2 * b * x + (a - c) = 0) : a = b :=
by sorry

-- Part 2
theorem right_triangle_if_equal_roots (a b c : ℝ)
  (h_discriminant : (2 * b)^2 - 4 * (a + c) * (a - c) = 0) : b^2 + c^2 = a^2 :=
by sorry

-- Part 3
theorem roots_of_equation_if_equilateral (a b c : ℝ) (h_eq_triangle : a = b ∧ b = c) :
  let x := (a + c) * x^2 + 2 * b * x + (a - c) = 0 in (x = 0 ∨ x = -1) :=
by sorry

end isosceles_triangle_if_root_neg_one_right_triangle_if_equal_roots_roots_of_equation_if_equilateral_l673_673008


namespace complement_of_angle_l673_673537

def complement_angle (deg : ℕ) (min : ℕ) : ℕ × ℕ :=
  if deg < 90 then 
    let total_min := (90 * 60)
    let angle_min := (deg * 60) + min
    let comp_min := total_min - angle_min
    (comp_min / 60, comp_min % 60) -- degrees and remaining minutes
  else 
    (0, 0) -- this case handles if the angle is not less than complement allowable range

-- Definitions based on the problem
def given_angle_deg : ℕ := 57
def given_angle_min : ℕ := 13

-- Complement calculation
def comp (deg : ℕ) (min : ℕ) : ℕ × ℕ := complement_angle deg min

-- Expected result of the complement
def expected_comp : ℕ × ℕ := (32, 47)

-- Theorem to prove the complement of 57°13' is 32°47'
theorem complement_of_angle : comp given_angle_deg given_angle_min = expected_comp := by
  sorry

end complement_of_angle_l673_673537


namespace sequence_general_formula_l673_673045

noncomputable def a : ℕ → ℕ
| 0       := 0  -- a_0 is not used in the problem statement, so we define it arbitrarily
| (n+1)   := if n = 0 then 1 else 2*(n + 1)*(n + 1) - (n + 1)

theorem sequence_general_formula (n : ℕ) (h : n > 0) : 
  a n = 2*n^2 - n := by
  sorry

end sequence_general_formula_l673_673045


namespace triangle_angle_B_and_area_range_l673_673983

theorem triangle_angle_B_and_area_range (A B C a b c S : ℝ) (h_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_area : S = 1 / 2 * a * c * sin B)
  (h_relation : b * sin A = a * cos (B - π / 6)) :
  (B = π / 3) ∧ (a = 2 → S ∈ set.Ioo (sqrt 3 / 2) (2 * sqrt 3)) :=
by
  sorry

end triangle_angle_B_and_area_range_l673_673983


namespace prove_r_gt_l673_673362

-- Definitions and conditions
variables (a b c d : ℕ)
def r := 1 - (a / b : ℝ) - (c / d : ℝ)
def condition1 := a + c ≤ 1982
def condition2 := r ≥ 0

-- Target statement
theorem prove_r_gt : condition1 → condition2 → r > 1 / 1983^3 :=
by sorry

end prove_r_gt_l673_673362


namespace part1_part2_i_part2_ii_part2_iii_l673_673456

open Nat Real

-- Part (1) Lean Statement
theorem part1 (x : ℝ) (n : ℕ) (h : n ≥ 2) :
  n * ((1 + x)^(n - 1) - 1) = ∑ k in finset.range (n + 1), if k < 2 then 0 else k * C(n, k) * x^(k - 1) :=
by sorry

-- Part (2) Lean Statements
theorem part2_i (n : ℕ) (h : n ≥ 3) :
  ∑ k in finset.range (n + 1), (-1) ^ k * k * C(n, k) = 0 :=
by sorry

theorem part2_ii (n : ℕ) (h : n ≥ 3) :
  ∑ k in finset.range (n + 1), (-1) ^ k * k^2 * C(n, k) = 0 :=
by sorry

theorem part2_iii (n : ℕ) (h : n ≥ 3) :
  ∑ k in finset.range (n + 1), (1 / (k + 1)) * C(n, k) = (2^(n + 1) - 1) / (n + 1) :=
by sorry

end part1_part2_i_part2_ii_part2_iii_l673_673456


namespace sum_seq_mod_1000_l673_673399

noncomputable def seq (n : ℕ) : ℕ :=
  if n < 4 then 1
  else seq (n-1) + seq (n-2) + seq (n-3) + seq (n-4)

theorem sum_seq_mod_1000 :
  let a1 := 354224848179261915075
  let a2 := 573147844013817084101
  let a3 := 927372692193078999176
  let seq := (λ n : ℕ, if n = 25 then a1
                       else if n = 26 then a2
                       else if n = 27 then a3
                       else if n < 4 then (1 : ℕ)
                       else seq (n - 1) + seq (n - 2) + seq (n - 3) + seq (n - 4))
  (∑ k in Finset.range 28, seq k) % 1000 = 352 :=
by
  sorry

end sum_seq_mod_1000_l673_673399


namespace person_before_you_taller_than_you_l673_673074

-- Define the persons involved in the problem.
variable (Person : Type)
variable (Taller : Person → Person → Prop)
variable (P Q You : Person)

-- The conditions given in the problem.
axiom standing_queue : Taller P Q
axiom queue_structure : You = Q

-- The question we need to prove, which is the correct answer to the problem.
theorem person_before_you_taller_than_you : Taller P You :=
by
  sorry

end person_before_you_taller_than_you_l673_673074


namespace probability_one_heads_one_tails_l673_673818

-- Definitions of the fair coin and outcomes
inductive Coin
| heads : Coin
| tails : Coin

def is_fair (coin : Coin) : Prop := true

def all_outcomes : list (Coin × Coin) :=
  [(Coin.heads, Coin.heads), (Coin.heads, Coin.tails),
   (Coin.tails, Coin.heads), (Coin.tails, Coin.tails)]

def favorable_outcomes (outcomes : list (Coin × Coin)) :=
  outcomes.filter (λ outcome, (outcome.1 = Coin.heads ∧ outcome.2 = Coin.tails) ∨
                                (outcome.1 = Coin.tails ∧ outcome.2 = Coin.heads))

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Statement of the problem
theorem probability_one_heads_one_tails : probability (favorable_outcomes all_outcomes).length all_outcomes.length = 1/2 := 
sorry

end probability_one_heads_one_tails_l673_673818


namespace problem_statement_l673_673495
-- Definitions of conditions
def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def f : ℝ → ℝ := λ x, if 2 ≤ x ∧ x ≤ 4 then abs (Real.log (x - 3/2) / Real.log 4) else 0  -- Placeholder for intervals outside [2,4]

-- Problem statement to prove
theorem problem_statement : 
  (∀ x, f (x + 4) = f x) ∧
  (∀ x, f x = f (-x)) ∧
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f x = abs (Real.log (x - 3/2) / Real.log 4)) →
  f (1/2) = 1/2 :=
sorry

end problem_statement_l673_673495


namespace largest_divisor_of_five_consecutive_integers_l673_673258

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l673_673258


namespace smallest_composite_no_prime_under_15_correct_l673_673940

-- Define the concept of a composite number
def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

-- Define the concept of having no prime factors less than 15
def has_no_prime_factors_less_than_15 (n : ℕ) : Prop := 
  ∀ p : ℕ, p.prime ∧ p ∣ n → 15 ≤ p

-- Define the smallest composite number with no prime factors less than 15 
def smallest_composite_no_prime_under_15 : ℕ := 289

-- Prove that this is the smallest number satisfying our conditions
theorem smallest_composite_no_prime_under_15_correct : 
  is_composite smallest_composite_no_prime_under_15 ∧
  has_no_prime_factors_less_than_15 smallest_composite_no_prime_under_15 ∧
  ∀ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than_15 n → n ≥ smallest_composite_no_prime_under_15 :=
by 
  sorry

end smallest_composite_no_prime_under_15_correct_l673_673940


namespace arithmetic_sequence_general_term_sum_first_n_terms_b_seq_l673_673005

theorem arithmetic_sequence_general_term :
  ∃ (a : ℕ → ℕ), (a(1) = 1) ∧ (∀ n, a(n) = n) ∧ (∃ d, d ≠ 0 ∧ (∀ m, a(m + 1) - a(m) = d)) ∧ 
  (∃ a1 a3 a9, a1 = a(1) ∧ a3 = a(3) ∧ a9 = a(9) ∧ a3^2 = a1 * a9) := by
  sorry

theorem sum_first_n_terms_b_seq (n : ℕ) :
  let a (n : ℕ) := n,
      b (n : ℕ) := 2^(a(n)) + n,
      S (n : ℕ) := ∑ i in range n, b(i)
  in S n = 2^(n+1) - 2 + n * (n + 1) / 2 := by
  sorry

end arithmetic_sequence_general_term_sum_first_n_terms_b_seq_l673_673005


namespace radius_solution_l673_673691

theorem radius_solution (n r : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) : 
  r = n * (1 - real.sqrt 3) / 2 :=
by 
  sorry

end radius_solution_l673_673691


namespace smallest_composite_no_prime_under_15_correct_l673_673939

-- Define the concept of a composite number
def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

-- Define the concept of having no prime factors less than 15
def has_no_prime_factors_less_than_15 (n : ℕ) : Prop := 
  ∀ p : ℕ, p.prime ∧ p ∣ n → 15 ≤ p

-- Define the smallest composite number with no prime factors less than 15 
def smallest_composite_no_prime_under_15 : ℕ := 289

-- Prove that this is the smallest number satisfying our conditions
theorem smallest_composite_no_prime_under_15_correct : 
  is_composite smallest_composite_no_prime_under_15 ∧
  has_no_prime_factors_less_than_15 smallest_composite_no_prime_under_15 ∧
  ∀ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than_15 n → n ≥ smallest_composite_no_prime_under_15 :=
by 
  sorry

end smallest_composite_no_prime_under_15_correct_l673_673939


namespace number_of_3_element_subsets_sum_of_sums_of_3_element_subsets_l673_673633

-- Define the set A
def A := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Proof Problem 1: Number of 3-element subsets of A
theorem number_of_3_element_subsets  : fintype.card {S : set ℕ // S ⊆ A ∧ S.card = 3} = 120 := 
by
  sorry

-- Proof Problem 2: Sum of sums of elements of 3-element subsets of A
theorem sum_of_sums_of_3_element_subsets : 
  let subsets := {S : set ℕ // S ⊆ A ∧ S.card = 3},
      sums := {sum (s : set ℕ) (h : s ∈ subsets)},
      total_sum := sum sums
  in total_sum = 1980 :=
by
  sorry

end number_of_3_element_subsets_sum_of_sums_of_3_element_subsets_l673_673633


namespace expected_yield_of_carrots_l673_673644

def steps_to_feet (steps : ℕ) (step_size : ℕ) : ℕ :=
  steps * step_size

def garden_area (length width : ℕ) : ℕ :=
  length * width

def yield_of_carrots (area : ℕ) (yield_rate : ℚ) : ℚ :=
  area * yield_rate

theorem expected_yield_of_carrots :
  steps_to_feet 18 3 * steps_to_feet 25 3 = 4050 →
  yield_of_carrots 4050 (3 / 4) = 3037.5 :=
by
  sorry

end expected_yield_of_carrots_l673_673644


namespace wildcats_more_points_l673_673689

theorem wildcats_more_points (wildcats_points panthers_points : ℕ) (h1 : wildcats_points = 36) (h2 : panthers_points = 17) : (wildcats_points - panthers_points = 19) :=
by
  rw [h1, h2]
  rfl

end wildcats_more_points_l673_673689


namespace ellipse_equation_max_area_and_line_equation_l673_673011

-- Definitions and conditions
def ellipse (a b : ℝ) := set_of (λ p : ℝ × ℝ, (p.1^2 / a^2) + (p.2^2 / b^2) = 1)

variables (a b c : ℝ) (e : ℝ) 
variables (h_a_b : a > b > 0) (h_e : e = (c / a)) (h_eccentricity : e = sqrt 3 / 2)

def Foci : set (ℝ × ℝ) := {F1 | F1.1 < 0} ∪ {F2 | F2.1 > 0} -- Simplified for example
def M : ℝ × ℝ := sorry -- point M on ellipse given, but not an endpoint of major axis
variables (h_perimeter : ∀ F1 F2 ∈ Foci, 4 + 2 * sqrt 3 = dist F1 M + dist F2 M + dist F1 F2) 

theorem ellipse_equation : ellipse 2 1 := 
by 
  sorry

variables (D : ℝ × ℝ) (h_D : D = (0, -2))
variables (l : ℝ → ℝ → Prop) (h_l : ∃ k : ℝ, l = λ x y, y = k * x - 2)

def quadrilateral_area (O A N B : ℝ × ℝ) := 
  abs (O.1 * (A.2 - N.2) + A.1 * (N.2 - B.2) + N.1 * (B.2 - O.2)) / 2

theorem max_area_and_line_equation : 
  ∃ k : ℝ, k = sqrt 7 / 2 ∧ quadrilateral_area (0,0) A N B = 2 
by 
  sorry


end ellipse_equation_max_area_and_line_equation_l673_673011


namespace train_speed_without_stops_l673_673878

-- theorem statement
theorem train_speed_without_stops : 
  let S := 60 -- speed of the train when it is not stopping
  in 
  ∀ (v_stop: ℕ) (t_stop: ℕ),
    v_stop = 36 → t_stop = 24 → 
    v_stop = (S * 6 / 10) :=
    sorry

end train_speed_without_stops_l673_673878


namespace only_correct_statement_l673_673984

-- Definitions of distinct lines and distinct planes
variables {a b c : Line} {α β : Plane}

-- Given conditions
axiom distinct_lines : a ≠ b ∧ b ≠ c ∧ a ≠ c
axiom distinct_planes : α ≠ β

-- Question 3 conditions
axiom perp_line_plane : a ⊥ α
axiom line_in_plane : a ⊂ β

-- The only correct statement in the problem
theorem only_correct_statement : (α ⊥ β) :=
by
  sorry

end only_correct_statement_l673_673984


namespace obtuse_angle_at_725_l673_673333

theorem obtuse_angle_at_725 : obtuse_angle_between_hands (7:25) = 287.5 :=
by
  sorry

end obtuse_angle_at_725_l673_673333


namespace probability_of_at_least_30_cents_l673_673680

def coin := fin 5

def value (c : coin) : ℤ :=
match c with
| 0 => 1   -- penny
| 1 => 5   -- nickel
| 2 => 10  -- dime
| 3 => 25  -- quarter
| 4 => 50  -- half-dollar
| _ => 0

def coin_flip : coin -> bool := λ c => true -- Placeholder for whether heads or tails

def total_value (flips : coin -> bool) : ℤ :=
  finset.univ.sum (λ c, if flips c then value c else 0)

noncomputable def probability_at_least_30_cents : ℚ :=
  let coin_flips := (finset.pi finset.univ (λ _, finset.univ : finset (coin -> bool))).val in
  let successful_flips := coin_flips.filter (λ flips, total_value flips >= 30) in
  successful_flips.card / coin_flips.card

theorem probability_of_at_least_30_cents :
  probability_at_least_30_cents = 9 / 16 :=
by
  sorry

end probability_of_at_least_30_cents_l673_673680


namespace percentage_increase_in_cost_is_12_l673_673552

noncomputable def cost:= ℝ  -- original cost
def selling_price (C: ℝ) := 2.5 * C
def new_cost (C: ℝ) (X: ℝ) := C + (X / 100) * C
def new_profit (S : ℝ) (C : ℝ) (X: ℝ) := S - new_cost C X

theorem percentage_increase_in_cost_is_12 (C S : ℝ) (X : ℝ)
  (h1 : S = selling_price C)
  (h2 : new_profit S C X = 0.552 * S) :
  X = 12 :=
  sorry

end percentage_increase_in_cost_is_12_l673_673552


namespace basketball_player_scores_8_distinct_vals_l673_673374

theorem basketball_player_scores_8_distinct_vals:
  (∀ x, 0 ≤ x ∧ x ≤ 7) →
  (∃ P, P = x + 14) →
  (∃ S, S = {P | P = x + 14 ∧ 0 ≤ x ∧ x ≤ 7}) →
  |S| = 8 := 
sorry

end basketball_player_scores_8_distinct_vals_l673_673374


namespace domain_function1_domain_function2_l673_673888

-- Problem 1
theorem domain_function1 (x : ℝ) : 
  (x - 1 ≠ 0) ∧ (x + 2 ≥ 0) ∧ (sqrt (x + 2) ≠ 0) ↔ (x > -2 ∧ x ≠ 1) :=
sorry

-- Problem 2
theorem domain_function2 (x : ℝ) : 
  (|x| - x ≥ 0) ∧ (sqrt (|x| - x) ≠ 0) ↔ (x < 0) :=
sorry

end domain_function1_domain_function2_l673_673888


namespace at_least_30_cents_prob_l673_673684

def coin := {penny, nickel, dime, quarter, half_dollar}
def value (c : coin) : ℕ := 
  match c with
  | penny => 1
  | nickel => 5
  | dime => 10
  | quarter => 25
  | half_dollar => 50

def coin_positions : List (coin × Bool) := 
  [(penny, true), (nickel, true), (dime, true), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, true), (dime, true), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, true), (dime, true), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, true), (dime, false), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, true), (dime, false), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, true), (dime, false), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, true), (dime, false), (quarter, false), (half_dollar, false),
   (penny, true), (nickel, false), (dime, true), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, false), (dime, true), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, false), (dime, true), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, false), (dime, true), (quarter, false), (half_dollar, false),
   (penny, true), (nickel, false), (dime, false), (quarter, true), (half_dollar, true),
   (penny, true), (nickel, false), (dime, false), (quarter, true), (half_dollar, false),
   (penny, true), (nickel, false), (dime, false), (quarter, false), (half_dollar, true),
   (penny, true), (nickel, false), (dime, false), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, true), (dime, true), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, true), (dime, true), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, true), (dime, true), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, true), (dime, true), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, true), (dime, false), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, true), (dime, false), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, true), (dime, false), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, true), (dime, false), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, false), (dime, true), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, false), (dime, true), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, false), (dime, true), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, false), (dime, true), (quarter, false), (half_dollar, false),
   (penny, false), (nickel, false), (dime, false), (quarter, true), (half_dollar, true),
   (penny, false), (nickel, false), (dime, false), (quarter, true), (half_dollar, false),
   (penny, false), (nickel, false), (dime, false), (quarter, false), (half_dollar, true),
   (penny, false), (nickel, false), (dime, false), (quarter, false), (half_dollar, false)]

def count_successful_outcomes : ℕ :=
  List.length (List.filter (λ positions, List.foldl (λ acc (c, h) => if h then acc + value c else acc) 0 positions >= 30) coin_positions)

def total_outcomes : ℕ := 32

def probability_of_success : ℚ :=
  ⟨count_successful_outcomes, total_outcomes⟩

theorem at_least_30_cents_prob : probability_of_success = 3 / 4 :=
by sorry

end at_least_30_cents_prob_l673_673684


namespace largest_divisor_of_five_consecutive_integers_l673_673264

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l673_673264


namespace frog_jumps_l673_673487

noncomputable def x : ℝ := 2 + Real.sqrt 2
noncomputable def y : ℝ := 2 - Real.sqrt 2

def e (n : ℕ) : ℝ := if n % 2 = 1 then 0 else 1 / Real.sqrt 2 * (x ^ (n/2 - 1) - y ^ (n/2 - 1))

theorem frog_jumps (n : ℕ) (h : n > 0) :
  (∃ e_n : ℕ → ℕ, e_n (2*n - 1) = 0 ∧ 
                   e_n (2*n) = (1/Real.sqrt 2) * (x^(n-1) - y^(n-1))) :=
by
  let e_n := λ m : ℕ, if m % 2 = 1 then 0 else 1 / Real.sqrt 2 * (x ^ (m/2 - 1) - y ^ (m/2 - 1))
  use e_n
  split
  { sorry },
  { sorry }

end frog_jumps_l673_673487


namespace arithmetic_seq_common_diff_l673_673031

theorem arithmetic_seq_common_diff
  (a₃ a₇ S₁₀ : ℤ)
  (h₁ : a₃ + a₇ = 16)
  (h₂ : S₁₀ = 85)
  (a₃_eq : ∃ a₁ d : ℤ, a₃ = a₁ + 2 * d)
  (a₇_eq : ∃ a₁ d : ℤ, a₇ = a₁ + 6 * d)
  (S₁₀_eq : ∃ a₁ d : ℤ, S₁₀ = 10 * a₁ + 45 * d) :
  ∃ d : ℤ, d = 1 :=
by
  sorry

end arithmetic_seq_common_diff_l673_673031


namespace omega_range_l673_673026

theorem omega_range (ω : ℝ) (h_pos : 0 < ω) (h_incr : ∀ x y, -real.pi / 3 ≤ x ∧ x ≤ real.pi / 4 → -real.pi / 3 ≤ y ∧ y ≤ real.pi / 4 → x < y → (2 * real.sin (ω * x) < 2 * real.sin (ω * y))) :
  0 < ω ∧ ω ≤ 3 / 2 :=
by
  sorry

end omega_range_l673_673026


namespace cost_to_buy_20_oranges_is_70_cents_l673_673387

-- Define the cost of buying different numbers of oranges
def cost_of_4_oranges := 15
def cost_of_6_oranges := 25
def cost_of_10_oranges := 40
def discount := 5

-- Define the total number of oranges to be purchased
def total_oranges := 20

-- Define the total cost calculation
def total_cost : ℕ := 
  let groups_of_4 := total_oranges / 4 in
  let cost_without_discount := groups_of_4 * cost_of_4_oranges in
  let final_cost := cost_without_discount - discount in
  final_cost

theorem cost_to_buy_20_oranges_is_70_cents :
  total_cost = 70 := 
by
  sorry

end cost_to_buy_20_oranges_is_70_cents_l673_673387


namespace purchasing_plans_count_l673_673831

theorem purchasing_plans_count :
  ∃ n : ℕ, n = 2 ∧ (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 5 * y = 35) :=
sorry

end purchasing_plans_count_l673_673831


namespace largest_divisor_of_5_consecutive_integers_l673_673305

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l673_673305


namespace rearrangement_inequality_l673_673457

theorem rearrangement_inequality
  (p : ℝ) (q : ℝ) (n : ℕ)
  (a : ℕ → ℝ)
  (b : ℕ → ℝ) :
  1 ≤ p ∧ 0 < q ∧
  ((∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n → a i ≥ a j) ∧ (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n → b i ≤ b j) ∨
  (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n → a i ≤ a j) ∧ (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n → b i ≥ b j)) →
  (∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i) ∧ (∀ i, 1 ≤ i ∧ i ≤ n → 0 < b i) →
  (∑ i in finset.range n, a (i + 1) ^ p / b (i + 1) ^ q) ≥
  n ^ (1 - p + q) * (∑ i in finset.range n, a (i + 1)) ^ p / (∑ i in finset.range n, b (i + 1)) ^ q :=
by
  sorry

end rearrangement_inequality_l673_673457


namespace inscribed_circle_diameter_l673_673210

-- Define the sides of the triangle
def DE : ℝ := 13
def DF : ℝ := 8
def EF : ℝ := 9

-- Define the semiperimeter
def s : ℝ := (DE + DF + EF) / 2

-- Define the area K using Heron's formula
def K : ℝ := real.sqrt (s * (s - DE) * (s - DF) * (s - EF))

-- Define the radius of the inscribed circle
def r : ℝ := K / s

-- Define the diameter of the inscribed circle
def d : ℝ := 2 * r

theorem inscribed_circle_diameter : d = 4 * real.sqrt 35 / 5 := by
  -- proof to be filled in
  sorry

end inscribed_circle_diameter_l673_673210


namespace richard_more_pins_than_patrick_l673_673098

theorem richard_more_pins_than_patrick :
  let P1 := 70 in
  let R1 := P1 + 15 in
  let P2 := R1 * 2 in
  let R2 := P2 - 3 in
  let Patrick_total := P1 + P2 in
  let Richard_total := R1 + R2 in
  Richard_total - Patrick_total = 12 :=
by
  let P1 := 70
  let R1 := P1 + 15
  let P2 := R1 * 2
  let R2 := P2 - 3
  let Patrick_total := P1 + P2
  let Richard_total := R1 + R2
  have h1 : Patrick_total = 240 := by
    calc
      Patrick_total = P1 + P2 := rfl
      ... = 70 + 170 := by
        let P1 := 70
        let R1 := P1 + 15
        let P2 := R1 * 2
        rfl
      ... = 240 := by norm_num
  have h2 : Richard_total = 252 := by
    calc
      Richard_total = R1 + R2 := rfl
      ... = 85 + 167 := by
        let R1 := P1 + 15
        let P2 := R1 * 2
        let R2 := P2 - 3
        rfl
      ... = 252 := by norm_num
  calc
    Richard_total - Patrick_total = 252 - 240 := by rw [h2, h1]
    ... = 12 := by norm_num

end richard_more_pins_than_patrick_l673_673098


namespace smallest_composite_proof_l673_673943

noncomputable def smallest_composite_no_prime_factors_less_than_15 : ℕ :=
  289

theorem smallest_composite_proof :
  smallest_composite_no_prime_factors_less_than_15 = 289 :=
by
  sorry

end smallest_composite_proof_l673_673943


namespace right_triangle_shorter_leg_l673_673566

theorem right_triangle_shorter_leg :
  ∃ (a b : ℤ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 16 :=
by
  sorry

end right_triangle_shorter_leg_l673_673566


namespace product_of_five_consecutive_divisible_by_30_l673_673267

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l673_673267


namespace TomAgeRatio_l673_673205

-- Conditions
variables (T N : ℕ)
hypothesis h1 : T = 4 * N  -- Implicit from "Tom's age is the sum of the ages of his four children"
hypothesis h2 : T - N = 3 * (T - 4 * N)  -- Tom's age N years ago was three times the sum of his children's ages then

-- Theorem
theorem TomAgeRatio : T = 11 / 2 * N :=
by
  sorry

end TomAgeRatio_l673_673205


namespace north_southville_population_increase_is_1200_l673_673591

def baby_birth_rate : ℕ := 6 -- hours per birth
def death_rate : ℕ := 36 -- hours per death

def births_per_day (hours_per_birth : ℕ) : ℚ := 
  24 / hours_per_birth

def deaths_per_day (hours_per_death : ℕ) : ℚ := 
  24 / hours_per_death

def daily_population_increase (births deaths : ℚ) : ℚ :=
  births - deaths

def annual_population_increase (daily_increase : ℚ) : ℚ :=
  daily_increase * 365

def rounded_to_nearest_hundred (n : ℚ) : ℕ :=
  let rounded : ℕ := Int.floor (n + 50) - (Int.floor (n + 50) % 100)
  rounded

theorem north_southville_population_increase_is_1200 :
  rounded_to_nearest_hundred (annual_population_increase
                                (daily_population_increase 
                                  (births_per_day baby_birth_rate) 
                                  (deaths_per_day death_rate))) = 1200 := 
by
  sorry

end north_southville_population_increase_is_1200_l673_673591


namespace geom_series_sum_l673_673657

theorem geom_series_sum (a : ℝ) (n : ℕ) :
  let term (p : ℕ) := (-1)^p * a^(4*p) in
  (finset.range n).sum (λ p, term p) = (a^4 / (a^4 + 1)) * ((-1)^n * a^(4*n) - 1) :=
sorry

end geom_series_sum_l673_673657


namespace total_points_l673_673550

theorem total_points (Jon Jack Tom : ℕ) (h1 : Jon = 3) (h2 : Jack = Jon + 5) (h3 : Tom = Jon + Jack - 4) : Jon + Jack + Tom = 18 := by
  sorry

end total_points_l673_673550


namespace percentage_liked_B_l673_673807

-- Given conditions
def percent_liked_A (X : ℕ) : Prop := X ≥ 0 ∧ X ≤ 100 -- X percent of respondents liked product A
def percent_liked_both : ℕ := 23 -- 23 percent liked both products.
def percent_liked_neither : ℕ := 23 -- 23 percent liked neither product.
def min_surveyed_people : ℕ := 100 -- The minimum number of people surveyed by the company.

-- Required proof
theorem percentage_liked_B (X : ℕ) (h : percent_liked_A X):
  100 - X = Y :=
sorry

end percentage_liked_B_l673_673807


namespace max_expressible_sums_l673_673438

section coloring_problem

-- Each number from 1 to 2014 has to be colored either red or blue, with half of them red and half blue
variables (red blue : set ℕ)
-- Constraint: Each of the numbers from 1 up to and including 2014 must be colored
def all_numbers_colored (n : ℕ) : Prop := 
  ∀ x, 1 ≤ x ∧ x ≤ n → x ∈ red ∨ x ∈ blue

-- Constraint: Half of them red and half of them blue
def half_colored (n : ℕ) : Prop :=
  fintype.card {x // x ∈ red} = n / 2 ∧ fintype.card {x // x ∈ blue} = n / 2 

-- k is the number of positive integers expressible as the sum of a red and a blue number
def expressible_sums (n : ℕ) : ℕ :=
  (fintype.card {k | ∃ a b, a ∈ red ∧ b ∈ blue ∧ a + b = k})

-- Prove the maximum value of k is 4023
theorem max_expressible_sums : 
  (all_numbers_colored red blue 2014) → (half_colored red blue 2014) → expressible_sums red blue 2014 = 4023 :=
by
  sorry

end coloring_problem

end max_expressible_sums_l673_673438


namespace largest_divisor_of_5_consecutive_integers_l673_673324

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l673_673324


namespace number_of_boxes_l673_673194

variable (boxes : ℕ) -- number of boxes
variable (mangoes_per_box : ℕ) -- mangoes per box
variable (total_mangoes : ℕ) -- total mangoes

def dozen : ℕ := 12

-- Condition: each box contains 10 dozen mangoes
def condition1 : mangoes_per_box = 10 * dozen := by 
  sorry

-- Condition: total mangoes in all boxes together is 4320
def condition2 : total_mangoes = 4320 := by
  sorry

-- Proof problem: prove that the number of boxes is 36
theorem number_of_boxes (h1 : mangoes_per_box = 10 * dozen) 
                        (h2 : total_mangoes = 4320) :
  boxes = 4320 / (10 * dozen) :=
  by
  sorry

end number_of_boxes_l673_673194


namespace inequality_inequation_l673_673663

theorem inequality_inequation (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (h : x + y + z = 1) :
  x * y + y * z + z * x ≤ 2 / 7 + 9 * x * y * z / 7 :=
by
  sorry

end inequality_inequation_l673_673663


namespace number_of_sets_l673_673707

def problem_statement : Prop :=
  ∃ (B : set (fin 6)), ({1, 3} ∪ B = {1, 3, 5}) ∧ 
    (B = {5} ∨ B = {1, 5} ∨ B = {3, 5} ∨ B = {1, 3, 5})

theorem number_of_sets (B : set (fin 6)) (h : {1, 3} ∪ B = {1, 3, 5}) :
  4 = finset.card (finset.univ.filter (λ b : set (fin 6), {1, 3} ∪ b = {1, 3, 5})) :=
sorry

end number_of_sets_l673_673707


namespace circles_externally_tangent_l673_673520

-- Definition of circle structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Definition of the given circles
def C1 : Circle := { center := (0, 0), radius := 1 }
def C2 : Circle := { center := (3, 0), radius := 2 }

-- Definition to calculate the distance between two centers
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem stating the positional relationship
theorem circles_externally_tangent (C1 C2 : Circle) : 
  distance C1.center C2.center = C1.radius + C2.radius := 
  sorry

end circles_externally_tangent_l673_673520


namespace largest_divisor_of_5_consecutive_integers_l673_673322

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l673_673322


namespace soccer_ball_cost_l673_673960

theorem soccer_ball_cost:
    ∃ (C : ℝ), C = 6 ∧ 
    (let x1 := 2.30 in
     ∃ x2 x3 x4 : ℝ,
     x2 = 1/3 * (x1 + x3 + x4) ∧
     x3 = 1/4 * (x1 + x2 + x4) ∧
     x4 = 1/5 * (x1 + x2 + x3) ∧
     C = x1 + x2 + x3 + x4) :=
begin
    -- Proof goes here
    sorry
end

end soccer_ball_cost_l673_673960


namespace number_whose_multiples_in_set_l673_673779

noncomputable def n : ℕ :=
  let s : set ℕ := {x | ∃ k : ℕ, 0 ≤ k ∧ k < 64 ∧ x = 68 + k * 4}
  320 - 68

theorem number_whose_multiples_in_set (n : ℕ) (s : set ℕ) (h1 : ∀ x ∈ s, ∃ k : ℕ, x = k * n)
  (h2 : 68 ∈ s) (h3 : 320 ∈ s) (h4 :  ∀ x ∈ s, 68 ≤ x ∧ x ≤ 320)
  (h5: ∀ a b ∈ s, a < b → a + n ≤ b):
  320 - 68 = n * 63 → n = 4 := by
  sorry

end number_whose_multiples_in_set_l673_673779


namespace shorter_leg_of_right_triangle_l673_673557

theorem shorter_leg_of_right_triangle (a b : ℕ) (h : a^2 + b^2 = 65^2) : min a b = 16 :=
sorry

end shorter_leg_of_right_triangle_l673_673557


namespace equilateral_triangle_roots_l673_673142

noncomputable def omega : ℂ := complex.exp (2 * complex.I * real.pi / 3)

theorem equilateral_triangle_roots
  (z1 z2 p q : ℂ)
  (h_roots : z1^2 + p * z1 + q = 0)
  (h_eq_triangle : ∃ w : ℂ, w ≠ 0 ∧ z2 = w * z1 ∧ w^3 = 1 ∧ w ≠ 1) :
  (p * p / q = 1) :=
sorry

end equilateral_triangle_roots_l673_673142


namespace tangents_and_parallel_lines_proof_l673_673855

theorem tangents_and_parallel_lines_proof
  {Γ1 Γ2 : Type} [circle Γ1] [circle Γ2]
  (M N A B C D E P Q : Point)
  (l : Line)
  (tangent_l Γ1 A : s.tangent l Γ1 A)
  (tangent_l Γ2 B : s.tangent l Γ2 B)
  (parallel_PM_l : s.parallel (s.line_through M C) l)
  (intersect_Γ1_C : C ∈ Γ1)
  (intersect_Γ2_D : D ∈ Γ2)
  (intersect_CA_DB : s.intersect (s.line_through C A) (s.line_through D B) = E)
  (intersect_AN_CD : s.intersect (s.line_through A N) (s.line_through C D) = P)
  (intersect_BN_CD : s.intersect (s.line_through B N) (s.line_through C D) = Q) :
  s.dist E P = s.dist E Q := 
begin
  sorry -- proof omitted
end

end tangents_and_parallel_lines_proof_l673_673855


namespace no_numbering_scheme_for_decagon_l673_673653

theorem no_numbering_scheme_for_decagon :
  ∀ (sides : Finset ℕ), sides = (Finset.range 11) \ 0 → 
  ¬ ∃ (numbering : Finset ℕ → ℕ), 
    ∀ i, 0 < i → i ≤ 10 → numbering (Finset.zeroMod i) + numbering (Finset.succ i) + numbering (Finset.succ_mod 10 i) = 16 :=
by
  sorry

end no_numbering_scheme_for_decagon_l673_673653


namespace subtracted_amount_l673_673800

theorem subtracted_amount (N A : ℝ) (h1 : 0.30 * N - A = 20) (h2 : N = 300) : A = 70 :=
by
  sorry

end subtracted_amount_l673_673800


namespace amanda_more_than_average_l673_673602

-- Conditions
def jill_peaches : ℕ := 12
def steven_peaches : ℕ := jill_peaches + 15
def jake_peaches : ℕ := steven_peaches - 16
def amanda_peaches : ℕ := jill_peaches * 2
def total_peaches : ℕ := jake_peaches + steven_peaches + jill_peaches
def average_peaches : ℚ := total_peaches / 3

-- Question: Prove that Amanda has 7.33 more peaches than the average peaches Jake, Steven, and Jill have
theorem amanda_more_than_average : amanda_peaches - average_peaches = 22 / 3 := by
  sorry

end amanda_more_than_average_l673_673602


namespace largest_divisor_of_5_consecutive_integers_l673_673306

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l673_673306


namespace tangent_line_monotonicity_distinct_zeros_l673_673137

-- Define the given conditions and functions
namespace math_problem

variable (a : ℝ) (x : ℝ)

-- Function definitions
def f (x : ℝ) : ℝ := Real.log x - a * x
def g (x : ℝ) : ℝ := (1 / 3) * x ^ 3 + x + 1
def h (x : ℝ) : ℝ := 2 * f x + g x - (1 / 3) * x ^ 3

-- Questions requiring proof as Lean theorems

-- (1) Prove the equation of the tangent line
theorem tangent_line (m : ℝ) :
  g m = (1 / 3) * m ^ 3 + m + 1 ∧ 
  (∀ (y : ℝ), y = (1 / 3) - ((1 / 3) * m ^ 3 + m + 1) = (m ^ 2 + 1) * (0 - m)) →
  2 * 0 - (1 / 3) = 0 →  -- simplifying to demonstrate passing through point (0, 1/3)
  ∃ (x y : ℝ), y = 2 * x - (1 / 3) := sorry

-- (2) Prove the monotonicity of the function h(x)
theorem monotonicity :
  (∀ x, 0 < x → ((1 - 2 * a) * x + 2) / x >= 0 → (a <= 1 / 2 ∧ 
  ∀ x, x ∈ (0, (2 / (2 * a - 1))) → ((1 - 2 * a) * x + 2) / x > 0 ∧ 
  ∀ x, x ∈ ((2 / (2 * a - 1)), +∞) → ((1 - 2 * a) * x + 2) / x < 0)) := sorry

-- (3) Prove that g(x1 * x2) > g(e^2)
theorem distinct_zeros (x1 x2 : ℝ) :
  x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ f x1 = 0 ∧ f x2 = 0 →
  g (x1 * x2) > g (Real.exp 2) := sorry

end math_problem

end tangent_line_monotonicity_distinct_zeros_l673_673137


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673319

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673319


namespace largest_divisor_of_5_consecutive_integers_l673_673327

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l673_673327


namespace lena_contribution_l673_673147

/-- Mason and Lena are buying together a set that costs 8 euros. Mason has 10 USD, and the exchange 
rate is 1 euro per 1.10 USD. Prove that Lena does not need to contribute any euros. -/
theorem lena_contribution : 
  (set_cost : ℝ) (mason_usd : ℝ) (exchange_rate : ℝ) (lena_contribution_needed : ℝ) 
  (h1 : set_cost = 8)
  (h2 : mason_usd = 10)
  (h3 : exchange_rate = 1 / 1.10) : 
  lena_contribution_needed = 0 :=
by 
  -- Placeholder for proof
  sorry

end lena_contribution_l673_673147


namespace domain_f_2x_plus_1_l673_673989

theorem domain_f_2x_plus_1 (domain_f_2_pow_x: Set ℝ) (h : domain_f_2_pow_x = set.Icc (-1 : ℝ) 1) :
  (set.preimage f (Icc (1 / 2) 2) = set.Icc (-1 / 4 : ℝ) (1 / 2)) :=
by
  sorry

end domain_f_2x_plus_1_l673_673989


namespace max_unique_planes_l673_673763

-- Given 15 points in space such that no four points are coplanar, 
-- prove that the maximum number of unique planes determined by these points is 455.

theorem max_unique_planes (points : Finset (EuclideanSpace ℝ (Fin 3))) (h : points.card = 15) (no_four_coplanar : ∀ p1 p2 p3 p4 ∈ points, AffineIndependent ℝ ![p1, p2, p3, p4]) : 
  Finset.card (Finset.image (λ s : Finset (EuclideanSpace ℝ (Fin 3)), s.choose 3) points) = 455 :=
sorry

end max_unique_planes_l673_673763


namespace total_earnings_l673_673203

noncomputable def treadmill := 300
noncomputable def chest_of_drawers := treadmill / 2
noncomputable def television := treadmill * 3
noncomputable def bicycle := (2 * chest_of_drawers) - 25
noncomputable def antique_vase := bicycle + 75
noncomputable def coffee_table (T : ℝ) := 0.08 * T
noncomputable def total_from_five_items := treadmill + chest_of_drawers + television + bicycle + antique_vase

theorem total_earnings (T : ℝ) (h : total_from_five_items + coffee_table T = 0.90 * T) : 
  T = 1975 / 0.82 :=
by 
  calc
    T = total_from_five_items / 0.90 : sorry
    ... = _ : sorry

end total_earnings_l673_673203


namespace seven_digit_prime_l673_673433

theorem seven_digit_prime (B : ℕ) (h : B = 2) : Nat.Prime (1034960 + B) :=
by
  have n : ℕ := 1034960 + B
  rw [h]
  have n_prime : Nat.Prime n := sorry
  exact n_prime

end seven_digit_prime_l673_673433


namespace largest_divisor_of_consecutive_product_l673_673229

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l673_673229


namespace ink_needed_per_whiteboard_l673_673586

-- Definitions
def classes : ℕ := 5
def whiteboards_per_class : ℕ := 2
def total_whiteboards : ℕ := classes * whiteboards_per_class

def ink_cost_per_ml : ℝ := 0.5
def total_cost : ℝ := 100
def total_ink_ml : ℝ := total_cost / ink_cost_per_ml
def ink_per_whiteboard : ℝ := total_ink_ml / total_whiteboards

-- Theorem to be proved
theorem ink_needed_per_whiteboard : ink_per_whiteboard = 20 := by
  sorry

end ink_needed_per_whiteboard_l673_673586


namespace angle_bisectors_intersect_at_right_angle_l673_673828

theorem angle_bisectors_intersect_at_right_angle
  (A B C D : Type)
  [EuclideanGeometry A B C D]
  (AB BC CD DA : ℝ)
  (h1 : parallel AB DC)
  (h2 : BC = AB + CD) :
  ∃ Y, (is_angle_bisector ∠ABC Y ∧ is_angle_bisector ∠BCD Y ∧ Y ∈ AD ∧ right_angle_at Y) :=
sorry

end angle_bisectors_intersect_at_right_angle_l673_673828


namespace smallest_solution_congruence_l673_673454

theorem smallest_solution_congruence : ∃ (x : ℤ), 0 ≤ x ∧ x < 15 ∧ 7 * x + 3 ≡ 6 [MOD 15] ∧ x = 9 :=
by
  sorry

end smallest_solution_congruence_l673_673454


namespace tangent_line_x_e_range_of_a_max_integer_k_l673_673040

noncomputable def f (x : ℝ) := x * real.log x
noncomputable def g (x : ℝ) (a : ℝ) := (a * x^2) / 2

theorem tangent_line_x_e (x e : ℝ) (hx : x = real.exp 1) :
  ∃ y : ℝ, (2 * x) - y - e = 0 := sorry

theorem range_of_a (a x_0 e : ℝ) (h1 : x_0 ∈ set.Icc 1 e) 
  (h2 : f x_0 < g x_0 a) : 0 < a := sorry

theorem max_integer_k (k x : ℤ) (h1 : k ∈ set.Icc 1 5) 
  (h2 : ∀ x > 1, f (real.of_int x) > (real.of_int k - 3) * real.of_int x - real.of_int k + 2) :
  k = 5 := sorry

end tangent_line_x_e_range_of_a_max_integer_k_l673_673040


namespace complement_event_A_l673_673804

def student_awards_sample_space : Type :=
  { (a_win : bool, b_win : bool) // true }

def event_A (x : student_awards_sample_space) : Prop :=
  x.1.1 = true ∧ x.1.2 = true

def complementary_event (x : student_awards_sample_space) : Prop :=
  ¬(event_A x)

theorem complement_event_A (x : student_awards_sample_space) :
  complementary_event x ↔ (x.1.1 = false ∨ x.1.2 = false) :=
by
  sorry

end complement_event_A_l673_673804


namespace tan_theta_eq_one_third_l673_673619

theorem tan_theta_eq_one_third 
  (k : ℝ) (θ : ℝ) (hk : k > 0)
  (hRD : matrix.mul 
           ![![real.cos θ, -real.sin θ], ![real.sin θ, real.cos θ]]
           ![![k, 0], ![0, k]] = ![![9, -3], ![3, 9]]) :
  real.tan θ = 1 / 3 := 
sorry

end tan_theta_eq_one_third_l673_673619


namespace sin_alpha_value_l673_673964

theorem sin_alpha_value (α : ℝ) (hα : 0 < α ∧ α < π / 2) 
  (h_cos : cos (α + π / 6) = 4 / 5) : 
  sin α = (3 * real.sqrt 3 - 4) / 10 := 
by 
  sorry

end sin_alpha_value_l673_673964


namespace find_integer_n_for_cosine_equality_l673_673891

theorem find_integer_n_for_cosine_equality : 
  ∃ n : ℕ, 0 ≤ n ∧ n ≤ 180 ∧ (cos (n * real.pi / 180) = cos (317 * real.pi / 180)) ∧ n = 43 :=
sorry

end find_integer_n_for_cosine_equality_l673_673891


namespace geometric_sequence_sum_n5_l673_673991

def geometric_sum (a₁ q : ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_sum_n5 (a₁ q : ℕ) (n : ℕ) (h₁ : a₁ = 3) (h₂ : q = 4) (h₃ : n = 5) : 
  geometric_sum a₁ q n = 1023 :=
by
  sorry

end geometric_sequence_sum_n5_l673_673991


namespace geometric_seq_relation_l673_673500

variables {α : Type*} [Field α]

-- Conditions for the arithmetic sequence (for reference)
def arithmetic_seq_sum (S : ℕ → α) (d : α) : Prop :=
∀ m n : ℕ, S (m + n) = S m + S n + (m * n) * d

-- Conditions for the geometric sequence
def geometric_seq_prod (T : ℕ → α) (q : α) : Prop :=
∀ m n : ℕ, T (m + n) = T m * T n * (q ^ (m * n))

-- Proving the desired relationship
theorem geometric_seq_relation {T : ℕ → α} {q : α} (h : geometric_seq_prod T q) (m n : ℕ) :
  T (m + n) = T m * T n * (q ^ (m * n)) :=
by
  apply h m n

end geometric_seq_relation_l673_673500


namespace largest_divisor_of_product_of_5_consecutive_integers_l673_673233

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l673_673233


namespace arithmetic_sum_given_condition_l673_673093

noncomputable def arithmetic_sequence {R : Type*} [linear_ordered_field R] (a d : R) : ℕ → R
| 0     := a
| (n+1) := (arithmetic_sequence a d n) + d

noncomputable def S {R : Type*} [linear_ordered_field R] (a d : R) (n : ℕ) : R :=
(n + 1) * (a + (n / 2) * d)

theorem arithmetic_sum_given_condition {a d : ℝ} (h : (arithmetic_sequence a d 0 + arithmetic_sequence a d 4 + arithmetic_sequence a d 11 + arithmetic_sequence a d 18 + arithmetic_sequence a d 22) = 15) :
  S a d 22 = 69 :=
by
  sorry

end arithmetic_sum_given_condition_l673_673093


namespace value_is_correct_l673_673369

-- Define the number
def initial_number : ℝ := 4400

-- Define the value calculation in Lean
def value : ℝ := 0.15 * (0.30 * (0.50 * initial_number))

-- The theorem statement
theorem value_is_correct : value = 99 := by
  sorry

end value_is_correct_l673_673369


namespace shorter_leg_of_right_triangle_l673_673573

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h: a^2 + b^2 = c^2) (hn: c = 65): a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l673_673573


namespace liked_product_B_l673_673812

-- Define the conditions as assumptions
variables (X : ℝ)

-- Assumptions
axiom liked_both : 23 = 23
axiom liked_neither : 23 = 23

-- The main theorem that needs to be proven
theorem liked_product_B (X : ℝ) : ∃ Y : ℝ, Y = 100 - X :=
by sorry

end liked_product_B_l673_673812


namespace solution_set_of_g_inequality_l673_673702

noncomputable def f : ℝ → ℝ := λ x => Real.sin (x - Real.pi / 6)
noncomputable def g : ℝ → ℝ := λ x => Real.sin (2 * x - Real.pi / 3)

theorem solution_set_of_g_inequality :
    { x : ℝ | 0 ≤ x ∧ x ≤ Real.pi ∧ g(x) - g(2023 * Real.pi / 2) > 0 }
    = { x : ℝ | Real.pi / 3 < x ∧ x < Real.pi / 2 } :=
sorry

end solution_set_of_g_inequality_l673_673702


namespace max_speed_on_flat_road_max_speed_on_inclined_road_max_speed_with_additional_load_flat_road_max_speed_with_additional_load_inclined_road_l673_673368

def weight_auto := 1200 -- kg
def power_auto := 1125 -- mkg/s
def friction_coeff := 1 / 20
def inclination_sin := 1 / 30
def additional_load := 400 -- kg

theorem max_speed_on_flat_road : 
  (power_auto / (friction_coeff * weight_auto)) * 3.6 = 67.5 := 
by {
  sorry
}

theorem max_speed_on_inclined_road : 
  (power_auto / (friction_coeff * weight_auto + inclination_sin * weight_auto)) * 3.6 = 40.5 := 
by {
  sorry
}

theorem max_speed_with_additional_load_flat_road : 
  (power_auto / (friction_coeff * (weight_auto + additional_load))) * 3.6 = 50.625 := 
by {
  sorry
}

theorem max_speed_with_additional_load_inclined_road : 
  (power_auto / (friction_coeff * (weight_auto + additional_load) + inclination_sin * (weight_auto + additional_load))) * 3.6 = 30.384 := 
by {
  sorry
}

end max_speed_on_flat_road_max_speed_on_inclined_road_max_speed_with_additional_load_flat_road_max_speed_with_additional_load_inclined_road_l673_673368


namespace tutte_3_2_5_l673_673724

-- Define 3-connected graph
class ThreeConnectedGraph (G : Type) [Graph G] := 
  (three_connected : ∀ (S : Set (Vertex G)), S.card ≤ 2 → connected_compl G S)

-- Specify the conditions of the sequence of graphs and edge contractions
def edge_contraction (G1 G2 : Type) [Graph G1] [Graph G2] (x y : Vertex G2) : Prop :=
  ∃ (xy_edges : Edge G2), x ∈ xy_edges ∧ y ∈ xy_edges ∧ contracted_graph G2 xy_edges = G1

-- The equivalent Lean theorem
theorem tutte_3_2_5 (G : Type) [Graph G] :
  (ThreeConnectedGraph G) ↔ 
  ∃ (n : ℕ) (Gs : Fin n → Type) [∀ i, Graph (Gs i)] [ThreeConnectedGraph (Gs 0)] [ThreeConnectedGraph (Gs n)], 
  (∀ i, i < n → edge_contraction (Gs i) (Gs (i+1))) ∧ G = Gs n :=
begin
  sorry
end

end tutte_3_2_5_l673_673724


namespace maximum_angle_prism_correct_l673_673995

-- Define rectangular prism and its properties
structure RectangularPrism (a b c : ℝ) where
  surface_area : ℝ
  edge_lengths_sum : ℝ

def maximum_angle (prism : RectangularPrism) : ℝ := 
  if prism.surface_area = 45 / 2 ∧ prism.edge_lengths_sum = 24 then
    Real.arccos (Real.sqrt 6 / 9)
  else
    0  -- Placeholder, only relevant conditions are considered

-- Prove that the maximum angle is as specified
theorem maximum_angle_prism_correct (a b c : ℝ)
  (h1 : 2 * (a * b + b * c + c * a) = 45 / 2)
  (h2 : a + b + c = 6) 
  : maximum_angle ⟨a, b, c, 45 / 2, 24⟩ = Real.arccos (Real.sqrt 6 / 9) := 
  sorry

end maximum_angle_prism_correct_l673_673995


namespace exist_triangle_l673_673859

noncomputable def construct_triangle (h_a h_b h_c : ℝ) : Type :=
  {ABC : Type // ∃ (A B C : ℝ), 
    let s := (A + B + C) / 2 in
    let area := sqrt (s * (s - A) * (s - B) * (s - C)) in
    area > 0 ∧ 
    2 * area / A = h_a ∧ 
    2 * area / B = h_b ∧ 
    2 * area / C = h_c }

theorem exist_triangle (h_a h_b h_c : ℝ) (h_a_pos : h_a > 0) (h_b_pos : h_b > 0) (h_c_pos : h_c > 0) : 
  ∃ (ABC : Type), construct_triangle h_a h_b h_c :=
by
  sorry

end exist_triangle_l673_673859


namespace num_good_words_l673_673861

/-- A good word is a sequence of letters {A, B, C, D} that satisfies:
 1. A is not immediately followed by B.
 2. B is not immediately followed by C.
 3. C is not immediately followed by D.
 4. D is not immediately followed by A. -/
def isGoodWord (w : List Char) : Prop :=
  ∀ i, i < w.length - 1 → (w[i], w[i+1]) ∉ [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]

/-- There are 8748 eight-letter good words. -/
theorem num_good_words : 
  ∃ (w : List Char), w.length = 8 ∧ isGoodWord w ∧ (finset ((char)) w).card = 8748 :=
begin
  sorry
end

end num_good_words_l673_673861


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673316

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673316


namespace smallest_composite_no_prime_factors_lt_15_l673_673907

theorem smallest_composite_no_prime_factors_lt_15 (n : ℕ) :
  ∀ n, (∀ p : ℕ, p.prime → p ∣ n → 15 ≤ p) → n = 289 → 
       is_composite n ∧ (∀ m : ℕ, (∀ q : ℕ, q.prime → q ∣ m → 15 ≤ q) → m ≥ 289) :=
by
  intros n hv hn
  -- Proof would go here
  sorry

end smallest_composite_no_prime_factors_lt_15_l673_673907


namespace probability_at_least_one_woman_l673_673539

def P_at_least_one_woman (total_men total_women selected : ℕ) : ℚ :=
  let total_people := total_men + total_women
  let P_four_men := 
    (total_men / total_people.toRat) *
    ((total_men - 1) / (total_people - 1).toRat) *
    ((total_men - 2) / (total_people - 2).toRat) *
    ((total_men - 3) / (total_people - 3).toRat)
  1 - P_four_men

theorem probability_at_least_one_woman (total_men total_women selected : ℕ) (h_men : total_men = 8) (h_women : total_women = 5) 
  (h_selected : selected = 4) : 
  P_at_least_one_woman total_men total_women selected = 129 / 143 := by
sorry

end probability_at_least_one_woman_l673_673539


namespace problem_statement_l673_673140

variable {α : Type*} [Real α]

noncomputable def geometric_inequality (a b c u v ω : α) : Prop :=
  (u / a) + (v / b) + (ω / c) ≥ Real.sqrt 3

theorem problem_statement (a b c u v ω : α) :
  geometric_inequality a b c u v ω :=
by
  sorry -/

end problem_statement_l673_673140


namespace largest_divisor_of_5_consecutive_integers_l673_673329

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l673_673329


namespace midpoint_of_translated_BG_l673_673188

/--
Given the points B = (1, 1), I = (2, 4), and G = (5, 1) forming triangle BIG,
and translating this triangle 7 units to the left and 4 units downward to form triangle B'I'G',
where B' is the image of B, I' is the image of I, and G' is the image of G,
prove that the midpoint of segment B'G' is (-4, -3).
-/
theorem midpoint_of_translated_BG :
  let B := (1, 1)
  let G := (5, 1)
  let B' := (1 - 7, 1 - 4)
  let G' := (5 - 7, 1 - 4)
  ((B'.1 + G'.1) / 2, (B'.2 + G'.2) / 2) = (-4, -3) :=
by
  let B := (1, 1)
  let G := (5, 1)
  let B' := (1 - 7, 1 - 4)
  let G' := (5 - 7, 1 - 4)
  have B'_eq : B' = (-6, -3) := rfl
  have G'_eq : G' = (-2, -3) := rfl
  calc
    ((B'.1 + G'.1) / 2, (B'.2 + G'.2) / 2)
        = ((-6 + -2) / 2, (-3 + -3) / 2) : by simp [B'_eq, G'_eq]
    ... = (-4, -3) : by simp

end midpoint_of_translated_BG_l673_673188


namespace geometric_sequence_inequality_l673_673972

variable {α : Type} [LinearOrderedField α] (a1 q : α) (n : ℕ)
variable (S : ℕ → α)
variable (a : ℕ → α)
hypothesis hq : q < 0
hypothesis ha : ∀ n, a n = a1 * q ^ (n - 1)
hypothesis hS : ∀ n, S n = a1 * (1 - q ^ n) / (1 - q)

-- Prove the inequality a_9 S_8 > a_8 S_9
theorem geometric_sequence_inequality : a 9 * S 8 > a 8 * S 9 :=
by
  sorry

end geometric_sequence_inequality_l673_673972


namespace garbage_average_600_l673_673182

noncomputable def garbage_collection_average (x : ℝ) : Prop :=
  let garbage_first_week := x
  let garbage_second_week := x / 2
  let total_garbage := garbage_first_week + garbage_second_week
  total_garbage = 900

theorem garbage_average_600 : garbage_collection_average 600 :=
by
  let x := 600
  let garbage_first_week := x
  let garbage_second_week := x / 2
  let total_garbage := garbage_first_week + garbage_second_week 
  have h1 : total_garbage = garbage_first_week + garbage_second_week := rfl
  have h2 : total_garbage = x + x / 2 := rfl
  have h3 : x + x / 2 = 900 := by
    calc
      2 * x / 2 + x / 2 = (2 * 600 / 2) + 600 / 2 : by norm_num
      ... = 900 : by norm_num
  exact h3.symm

end garbage_average_600_l673_673182


namespace transformed_variance_l673_673029

variable {n : ℕ}
variable {a : ℕ → ℝ}

-- Define the variance of a dataset
def variance (data : ℕ → ℝ) (n : ℕ) : ℝ := (∑ i in finset.range n, (data i - (∑ i in finset.range n, data i) / n) ^ 2) / n

-- Given condition
axiom original_variance : variance a n = 4

-- Theorem to prove
theorem transformed_variance : variance (λ i, 2 * a i) n = 16 := by
  sorry

end transformed_variance_l673_673029


namespace turns_needed_to_return_60_turns_needed_to_return_42_turns_needed_to_return_47_l673_673380

def turns_needed_to_return (x : ℝ) : ℕ :=
  if x = 60 then 6
  else if x = 42 then 60
  else if x = 47 then 360
  else 0  -- defaulting to 0 for unspecified cases

theorem turns_needed_to_return_60 :
  turns_needed_to_return 60 = 6 :=
by
  sorry

theorem turns_needed_to_return_42 :
  turns_needed_to_return 42 = 60 :=
by
  sorry

theorem turns_needed_to_return_47 :
  turns_needed_to_return 47 = 360 :=
by
  sorry

end turns_needed_to_return_60_turns_needed_to_return_42_turns_needed_to_return_47_l673_673380


namespace count_integer_palindromes_between_100_and_1000_l673_673862

def is_palindrome (n : ℕ) : Prop :=
  let digits := Int.toString n in
  digits = digits.reverse

def is_three_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ is_palindrome n

theorem count_integer_palindromes_between_100_and_1000 : 
  {n : ℕ | is_three_digit_palindrome n}.card = 90 := 
sorry

end count_integer_palindromes_between_100_and_1000_l673_673862


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673315

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673315


namespace percent_answered_both_correctly_l673_673778

variable (A B : Prop)
variable (P_A P_B P_A_not_B_not : ℝ)
variable (P_A : P_A = 0.75)
variable (P_B : P_B = 0.65)
variable (P_A_not_B_not : P_A_not_B_not = 0.20)

theorem percent_answered_both_correctly:
    P_A + P_B - P_A_not_B_not = 0.75 + 0.65 - P_A_not_B_not :=
by
    have complement : 1 - P_A_not_B_not = 0.80 := sorry
    have inclusion_exclusion : 0.80 = P_A + P_B - P_A_not_B_not := sorry
    show P_A ∩ B = 0.60 := sorry

end percent_answered_both_correctly_l673_673778


namespace sum_of_integer_solutions_l673_673948

theorem sum_of_integer_solutions : (∑ x in Finset.filter (λ x : ℤ, x^4 - 36 * x^2 + 100 = 0) (Finset.Icc (-10) 10)) = 0 :=
by
  sorry

end sum_of_integer_solutions_l673_673948


namespace diameter_and_radius_of_circles_l673_673209

-- Define the sides of the triangle
def DE := 13
def DF := 8
def EF := 9

-- Define the semiperimeter
def s := (DE + DF + EF) / 2

-- Define the area using Heron's formula
def K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))

-- Define the radius of the inscribed circle
def r := K / s

-- Define the diameter of the inscribed circle
def d := 2 * r

-- Define the radius of the circumscribed circle
def R := (DE * DF * EF) / (4 * K)

-- The statement to prove
theorem diameter_and_radius_of_circles :
  d = 2 * Real.sqrt 14 ∧ R = (39 * Real.sqrt 14) / 35 := by
    sorry

end diameter_and_radius_of_circles_l673_673209


namespace union_A_B_complement_A_inter_B_non_empty_A_inter_C_l673_673483

-- Definitions
def A := {x : ℝ | 2 ≤ x ∧ x ≤ 8}
def B := {x : ℝ | 1 < x ∧ x < 6}
def C (a : ℝ) := {x : ℝ | x > a}
def U := set.univ

-- Proof statements (with sorry to fill in the proof later)
theorem union_A_B : A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 8} :=
sorry

theorem complement_A_inter_B : (U \ A) ∩ B = {x : ℝ | 1 < x ∧ x < 2} :=
sorry

theorem non_empty_A_inter_C (a : ℝ) (h : (A ∩ C a).nonempty) : a < 8 :=
sorry

end union_A_B_complement_A_inter_B_non_empty_A_inter_C_l673_673483


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673220

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673220


namespace number_of_feasible_networks_10_l673_673841

-- Definitions based on conditions
def feasible_networks (n : ℕ) : ℕ :=
if n = 0 then 1 else 2 ^ (n - 1)

-- The proof problem statement
theorem number_of_feasible_networks_10 : feasible_networks 10 = 512 := by
  -- proof goes here
  sorry

end number_of_feasible_networks_10_l673_673841


namespace calculate_dani_pants_l673_673875

theorem calculate_dani_pants : ∀ (initial_pants number_years pairs_per_year pants_per_pair : ℕ), 
  initial_pants = 50 →
  number_years = 5 →
  pairs_per_year = 4 →
  pants_per_pair = 2 →
  initial_pants + (number_years * (pairs_per_year * pants_per_pair)) = 90 :=
by
  intros initial_pants number_years pairs_per_year pants_per_pair
  intro h_initial_pants h_number_years h_pairs_per_year h_pants_per_pair
  rw [h_initial_pants, h_number_years, h_pairs_per_year, h_pants_per_pair]
  norm_num
  sorry

end calculate_dani_pants_l673_673875


namespace length_of_BC_l673_673649

-- Define the context
variables {O A M B C : Type*} [inner_product_space ℝ O]

-- Define the circle with center O and radius 10
def circle (O : O) (r : ℝ) := {P : O | dist P O = r}

-- Conditions as definitions
constant r : ℝ := 10
constant α : ℝ
constant M : O
constant A : O
constant B : O
constant C : O
constant AMB : ∠ A M B = α
constant OMC : ∠ O M C = α
constant cos_α : cos α = 4 / 5

-- Problem statement
theorem length_of_BC : dist B C = 16 :=
sorry

end length_of_BC_l673_673649


namespace find_EG_FH_l673_673594

variables (EF GH EG FH h : ℝ) (A : ℝ := 72)

-- Conditions
axiom EF_val : EF = 10
axiom GH_val : GH = 14
axiom sides_equal : EG = FH
axiom area_val : A = 72

-- Definition of the height of the trapezoid
def trapezoid_height (A B1 B2 : ℝ) : ℝ := (2 * A) / (B1 + B2)

-- Pythagorean theorem
def pythagorean (a b : ℝ) : ℝ := sqrt (a^2 + b^2)

theorem find_EG_FH :
  let h := trapezoid_height A EF GH in
  let base_diff := GH - EF in
  let projection := base_diff / 2 in
  EG = pythagorean projection h → EG = 2 * sqrt 10 :=
by
  sorry

end find_EG_FH_l673_673594


namespace flower_combinations_count_l673_673381

/-- Prove that there are exactly 3 combinations of tulips and sunflowers that sum up to $60,
    where tulips cost $4 each and sunflowers cost $3 each, and the number of sunflowers is greater than the number 
    of tulips. -/
theorem flower_combinations_count :
  ∃ n : ℕ, n = 3 ∧
    ∃ t s : ℕ, 4 * t + 3 * s = 60 ∧ s > t :=
by {
  sorry
}

end flower_combinations_count_l673_673381


namespace percentage_respondents_liked_B_l673_673813

variables (X Y : ℝ)
variables (likedA likedB likedBoth likedNeither : ℝ)
variables (totalRespondents : ℕ)

-- Conditions from the problem
def liked_conditions : Prop :=
    totalRespondents ≥ 100 ∧ 
    likedA = X ∧ 
    likedB = Y ∧ 
    likedBoth = 23 ∧ 
    likedNeither = 23

-- Proof statement
theorem percentage_respondents_liked_B (h : liked_conditions X Y likedA likedB likedBoth likedNeither totalRespondents) :
  Y = 100 - X :=
sorry

end percentage_respondents_liked_B_l673_673813


namespace smallest_composite_no_prime_factors_less_than_15_l673_673925

theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n, (n = 289) ∧ (n > 1) ∧ (¬ Nat.Prime n) ∧ (∀ p : ℕ, Nat.Prime p → p ∣ n → 15 ≤ p) :=
by
  use 289
  split
  case left => rfl
  case right =>
    split
    case left => exact Nat.lt_succ_self 288
    case right =>
      split
      case left =>
        have composite : ¬ Nat.Prime 289 := by
          intro h
          have h_div : 17 ∣ 289 := by norm_num
          exact h.not_divs_self (dec_trivial : 17 * 17 = 289)
        exact composite
      case right =>
        intros p h_prime h_div
        have : p ∣ 17 := by
          have factorization : 289 = 17 * 17 := by norm_num
          have dvd_product : p ∣ 289 := by { use 17, exact factorization.symm }
          exact Nat.Prime.dvd_mul h_prime dvd_product
        have prime_eq_17 : p = 17 := by
          exact Nat.Prime.eq_of_dvd_of_ne h_prime (by norm_num) this
        linarith

end smallest_composite_no_prime_factors_less_than_15_l673_673925


namespace positive_difference_b_l673_673632

noncomputable def f (n : ℝ) : ℝ :=
if n < 1 then n^2 - 6 else 3 * n - 15

theorem positive_difference_b :
  let b1 := -Real.sqrt 23
  let b2 := 32 / 3
  b1 < 1 → 1 ≤ b2 →
  f (-1) + f 1 + f b1 = 0 → f (-1) + f 1 + f b2 = 0 →
  abs (b1 - b2) = Real.sqrt 23 + 32 / 3 := by
  intro b1 b2 H_b1 H_b2 Hb1 Hb2
  sorry

end positive_difference_b_l673_673632


namespace bianca_ate_candies_l673_673950

-- Definitions based on the conditions
def total_candies : ℕ := 32
def pieces_per_pile : ℕ := 5
def number_of_piles : ℕ := 4

-- The statement to prove
theorem bianca_ate_candies : 
  total_candies - (pieces_per_pile * number_of_piles) = 12 := 
by 
  sorry

end bianca_ate_candies_l673_673950


namespace total_points_l673_673549

theorem total_points (Jon Jack Tom : ℕ) (h1 : Jon = 3) (h2 : Jack = Jon + 5) (h3 : Tom = Jon + Jack - 4) : Jon + Jack + Tom = 18 := by
  sorry

end total_points_l673_673549


namespace time_for_P_and_Q_to_complete_job_l673_673393

-- Definitions and conditions
def P_time : ℝ := 4
def Q_time : ℝ := 6
def combined_work_rate : ℝ := (1 / P_time) + (1 / Q_time)
def job : ℝ := 1

-- The theorem statement
theorem time_for_P_and_Q_to_complete_job : (1 / combined_work_rate) = 12 / 5 := by
  sorry

end time_for_P_and_Q_to_complete_job_l673_673393


namespace smallest_number_in_sample_l673_673461

theorem smallest_number_in_sample :
  ∀ (N : ℕ) (k : ℕ) (n : ℕ), 
  0 < k → 
  N = 80 → 
  k = 5 →
  n = 42 →
  ∃ (a : ℕ), (0 ≤ a ∧ a < k) ∧
  42 = (N / k) * (42 / (N / k)) + a ∧
  ∀ (m : ℕ), (0 ≤ m ∧ m < k) → 
    (∀ (j : ℕ), (j = (N / k) * m + 10)) → 
    m = 0 → a = 10 := 
by
  sorry

end smallest_number_in_sample_l673_673461


namespace dilation_result_l673_673699

noncomputable def C : ℂ := 2 - 3 * complex.I
noncomputable def k : ℤ := 3
noncomputable def w : ℂ := -1 + complex.I
noncomputable def z : ℂ := -7 + 9 * complex.I

theorem dilation_result :
  z = k * (w - C) + C := 
sorry

end dilation_result_l673_673699


namespace book_original_price_l673_673375

-- Definitions for conditions
def selling_price := 56
def profit_percentage := 75

-- Statement of the theorem
theorem book_original_price : ∃ CP : ℝ, selling_price = CP * (1 + profit_percentage / 100) ∧ CP = 32 :=
by
  sorry

end book_original_price_l673_673375


namespace correct_order_of_statistical_analysis_l673_673201

theorem correct_order_of_statistical_analysis 
    (step1 := "Draw conclusions, make suggestions")
    (step2 := "Analyze data")
    (step3 := "Randomly select 400 students from the 40,000 students to investigate their average daily reading time")
    (step4 := "Organize and represent the collected data using statistical charts") :
    ["step3", "step4", "step2", "step1"] = ["step3", "step4", "step2", "step1"] :=
begin
    sorry
end

end correct_order_of_statistical_analysis_l673_673201


namespace complement_event_A_l673_673803

def student_awards_sample_space : Type :=
  { (a_win : bool, b_win : bool) // true }

def event_A (x : student_awards_sample_space) : Prop :=
  x.1.1 = true ∧ x.1.2 = true

def complementary_event (x : student_awards_sample_space) : Prop :=
  ¬(event_A x)

theorem complement_event_A (x : student_awards_sample_space) :
  complementary_event x ↔ (x.1.1 = false ∨ x.1.2 = false) :=
by
  sorry

end complement_event_A_l673_673803


namespace largest_divisor_of_product_of_5_consecutive_integers_l673_673241

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l673_673241


namespace trig_eqn_to_cos_product_l673_673700

theorem trig_eqn_to_cos_product (x : ℝ) :
  (∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    sin x ^ 2 + sin (3 * x) ^ 2 + sin (5 * x) ^ 2 + sin (7 * x) ^ 2 = 2 ∧ 
    cos (a * x) * cos (b * x) * cos (c * x) = 0 ∧ 
    a + b + c = 14) :=
begin
  sorry,
end

end trig_eqn_to_cos_product_l673_673700


namespace product_of_five_consecutive_integers_divisible_by_240_l673_673281

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l673_673281


namespace inverse_of_38_mod_53_l673_673489

theorem inverse_of_38_mod_53
  (h : ∃ x : ℤ, 15 * x ≡ 1 [MOD 53] ∧ x ≡ 31 [MOD 53]) :
  ∃ y : ℤ, 38 * y ≡ 1 [MOD 53] ∧ y ≡ 22 [MOD 53] :=
sorry

end inverse_of_38_mod_53_l673_673489


namespace largest_integer_of_four_l673_673961

theorem largest_integer_of_four (a b c d : ℤ) 
  (h1 : a + b + c = 160) 
  (h2 : a + b + d = 185) 
  (h3 : a + c + d = 205) 
  (h4 : b + c + d = 230) : 
  max (max a (max b c)) d = 100 := 
by
  sorry

end largest_integer_of_four_l673_673961


namespace coffee_cup_original_amount_l673_673822

noncomputable def coffee_cup_amount (remaining_coffee : ℝ) (num_of_cups : ℕ) (shrink_percentage : ℝ) : ℝ :=
  (remaining_coffee / (shrink_percentage / 100)) / num_of_cups

theorem coffee_cup_original_amount :
  ∀ (remaining_coffee : ℝ) (num_of_cups : ℕ) (shrink_percentage : ℝ),
  remaining_coffee = 20 → num_of_cups = 5 → shrink_percentage = 50 → 
  coffee_cup_amount remaining_coffee num_of_cups shrink_percentage = 8 :=
by
  intros remaining_coffee num_of_cups shrink_percentage h1 h2 h3
  unfold coffee_cup_amount
  rw [h1, h2, h3]
  norm_num
  sorry

end coffee_cup_original_amount_l673_673822


namespace tg_half_x_solution_l673_673444

theorem tg_half_x_solution (x : ℝ) : 
  (sin x + cos x = 1 / 5) → (tg (x / 2) = 2 ∨ tg (x / 2) = -1 / 3) :=
by
  intros h
  sorry

end tg_half_x_solution_l673_673444


namespace arithmetic_mean_difference_l673_673355

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 26) : 
  r - p = 32 := by
  sorry

end arithmetic_mean_difference_l673_673355


namespace count_divisibles_by_8_in_range_100_250_l673_673526

theorem count_divisibles_by_8_in_range_100_250 : 
  let lower_bound := 100
  let upper_bound := 250
  let divisor := 8
  ∃ n : ℕ, (∀ x : ℕ, lower_bound ≤ x ∧ x ≤ upper_bound ∧ x % divisor = 0 ↔ (n = 19)) :=
begin
  let lower_bound := 100,
  let upper_bound := 250,
  let divisor := 8,
  let first_multiple := ((lower_bound + divisor - 1) / divisor) * divisor,
  let last_multiple := (upper_bound / divisor) * divisor,
  let first_index := first_multiple / divisor,
  let last_index := last_multiple / divisor,
  let n := (last_index - first_index + 1),
  use n,
  intros x,
  split,
  { intro hx,
    exact ⟨nat.exists_eq_add_of_le hx.1, nat.exists_eq_add_of_le hx.2.1, nat.exists_eq_of_divisible hx.2.2⟩ },
  { intro hn,
    rw hn,
    refine ⟨_, _, _⟩,
    sorry
  }
end

end count_divisibles_by_8_in_range_100_250_l673_673526


namespace cost_price_l673_673061

theorem cost_price (C : ℝ) : 
  (0.05 * C = 350 - 340) → C = 200 :=
by
  assume h1 : 0.05 * C = 10
  sorry

end cost_price_l673_673061


namespace at_most_5_negatives_l673_673075

theorem at_most_5_negatives (a b c d e f : ℤ) (h : a * b * c * d * e * f < 0) : 
  (0 < list.filter (λ x, x < 0) [a, b, c, d, e, f]).length ∧ list.filter (λ x, x < 0) [a, b, c, d, e, f]).length ≤ 5 :=
begin
  sorry
end

end at_most_5_negatives_l673_673075


namespace volume_of_rectangular_solid_l673_673192

theorem volume_of_rectangular_solid (x y z : ℝ) 
  (h1 : x * y = 18) 
  (h2 : y * z = 15) 
  (h3 : z * x = 10) : 
  x * y * z = 30 * Real.sqrt 3 := 
sorry

end volume_of_rectangular_solid_l673_673192


namespace largest_divisor_of_consecutive_five_l673_673290

theorem largest_divisor_of_consecutive_five (n : ℤ) : 
  (∃ d, ∀ i ∈ {n, n + 1, n + 2, n + 3, n + 4}, d ∣ (∏ x in {n, n + 1, n + 2, n + 3, n + 4}, x)) → 
  d = 60 
:= sorry

end largest_divisor_of_consecutive_five_l673_673290


namespace tangency_l673_673107

-- Definitions of points and conditions
variable (A B C M P N : Type)
variable [Inhabited M] [Inhabited P] [Inhabited N]
variable (triangle_ABC : Triangle A B C)
variable (midpoint_M : Midpoint B C M)
variable (circle_BM_tangent_AM : TangentCircleThrough B M passingThroughAM tangentAtM intersectingAB P)
variable (midpoint_N : Midpoint A M N)

-- Prove statement
theorem tangency {A B C M P N : Type} 
  (triangle_ABC : Triangle A B C)
  (midpoint_M : Midpoint B C M)
  (circle_BM_tangent_AM : TangentCircleThrough B M passingThroughAM tangentAtM intersectingAB P)
  (midpoint_N : Midpoint A M N) :
  TangentCircle A P N A C :=
sorry

end tangency_l673_673107


namespace ingrid_income_l673_673606

theorem ingrid_income (combined_tax_rate : ℝ)
  (john_income : ℝ) (john_tax_rate : ℝ)
  (ingrid_tax_rate : ℝ)
  (combined_income : ℝ)
  (combined_tax : ℝ) :
  combined_tax_rate = 0.35581395348837205 →
  john_income = 57000 →
  john_tax_rate = 0.3 →
  ingrid_tax_rate = 0.4 →
  combined_income = john_income + (combined_income - john_income) →
  combined_tax = (john_tax_rate * john_income) + (ingrid_tax_rate * (combined_income - john_income)) →
  combined_tax_rate = combined_tax / combined_income →
  combined_income = 57000 + 72000 :=
by
  sorry

end ingrid_income_l673_673606


namespace polynomial_constant_for_k_geq_4_l673_673448

theorem polynomial_constant_for_k_geq_4 (k : ℕ) (F : ℕ → ℤ) (hF : ∀ c ∈ Finset.range (k + 2), 0 ≤ F c ∧ F c ≤ k) :
  (∀ c1 c2 ∈ Finset.range (k + 2), F c1 = F c2) ↔ k ≥ 4 :=
begin
  sorry
end

end polynomial_constant_for_k_geq_4_l673_673448


namespace smallest_composite_no_prime_factors_less_than_15_l673_673916

-- Definitions used in the conditions
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

-- Prime numbers less than 15
def primes_less_than_15 (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13

-- Define the main proof statement
theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n : ℕ, is_composite n ∧ (∀ p : ℕ, p ∣ n → is_prime p → primes_less_than_15 p → false) ∧ n = 289 :=
by
  -- leave the proof as a placeholder
  sorry

end smallest_composite_no_prime_factors_less_than_15_l673_673916


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673211

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673211


namespace product_sequence_eq_l673_673420

theorem product_sequence_eq :
  let seq := [ (1 : ℚ) / 2, 4 / 1, 1 / 8, 16 / 1, 1 / 32, 64 / 1,
               1 / 128, 256 / 1, 1 / 512, 1024 / 1, 1 / 2048, 4096 / 1 ]
  (seq.prod) * (3 / 4) = 1536 := by 
  -- expand and simplify the series of products
  sorry 

end product_sequence_eq_l673_673420


namespace valid_digit_cancel_fractions_l673_673759

def digit_cancel_fraction (a b c d : ℕ) : Prop :=
  10 * a + b == 0 ∧ 10 * c + d == 0 ∧ 
  (b == d ∨ b == c ∨ a == d ∨ a == c) ∧
  (b ≠ a ∨ d ≠ c) ∧
  ((10 * a + b) ≠ (10 * c + d)) ∧
  ((10 * a + b) * d == (10 * c + d) * a)

theorem valid_digit_cancel_fractions :
  ∀ (a b c d : ℕ), 
  digit_cancel_fraction a b c d → 
  (10 * a + b == 26 ∧ 10 * c + d == 65) ∨
  (10 * a + b == 16 ∧ 10 * c + d == 64) ∨
  (10 * a + b == 19 ∧ 10 * c + d == 95) ∨
  (10 * a + b == 49 ∧ 10 * c + d == 98) :=
by {sorry}

end valid_digit_cancel_fractions_l673_673759


namespace smallest_composite_no_prime_factors_less_than_15_l673_673918

-- Definitions used in the conditions
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

-- Prime numbers less than 15
def primes_less_than_15 (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13

-- Define the main proof statement
theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n : ℕ, is_composite n ∧ (∀ p : ℕ, p ∣ n → is_prime p → primes_less_than_15 p → false) ∧ n = 289 :=
by
  -- leave the proof as a placeholder
  sorry

end smallest_composite_no_prime_factors_less_than_15_l673_673918


namespace usage_gender_relation_predict_users_l673_673149

noncomputable def chi_square_test (high_usage_male high_usage_female low_usage_male low_usage_female total_male total_female total_users : ℕ) : ℝ :=
  let chi_square_num := (total_users * (high_usage_male.to_float * low_usage_female.to_float - high_usage_female.to_float * low_usage_male.to_float) ^ 2)
  let chi_square_den := (total_male * total_female * (high_usage_male + high_usage_female) * (low_usage_male + low_usage_female)).to_float
  chi_square_num / chi_square_den

theorem usage_gender_relation (high_usage_male high_usage_female low_usage_male low_usage_female total_users : ℕ) (chi_square_value : ℝ) :
  chi_square_value = chi_square_test high_usage_male high_usage_female low_usage_male low_usage_female 90 110 total_users ∧ chi_square_value > 10.828 :=
  sorry

noncomputable def regression_equation (a d : ℝ) (x : ℕ) : ℝ := a * 10^(d * x)

theorem predict_users (a b : ℝ) (twelve_day_expected : ℝ) : 
  a = 3.98 ∧ b = 0.25 ∧ twelve_day_expected = regression_equation a b 12 :=
  sorry

end usage_gender_relation_predict_users_l673_673149


namespace geom_progression_vertex_ad_l673_673985

theorem geom_progression_vertex_ad
  (a b c d : ℝ)
  (geom_prog : a * c = b * b ∧ b * d = c * c)
  (vertex : (b, c) = (1, 3)) :
  a * d = 3 :=
sorry

end geom_progression_vertex_ad_l673_673985


namespace largest_divisor_of_5_consecutive_integers_l673_673328

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l673_673328


namespace cos_neg_13pi_div_4_l673_673723

theorem cos_neg_13pi_div_4 : (Real.cos (-13 * Real.pi / 4)) = -Real.sqrt 2 / 2 := 
by sorry

end cos_neg_13pi_div_4_l673_673723


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673215

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673215


namespace poly_has_integer_roots_iff_a_eq_one_l673_673611

-- Definition: a positive real number
def pos_real (a : ℝ) : Prop := a > 0

-- The polynomial
def p (a : ℝ) (x : ℝ) : ℝ := a^3 * x^3 + a^2 * x^2 + a * x + a

-- The main theorem
theorem poly_has_integer_roots_iff_a_eq_one (a : ℝ) (x : ℤ) :
  (pos_real a ∧ ∃ x : ℤ, p a x = 0) ↔ a = 1 :=
by sorry

end poly_has_integer_roots_iff_a_eq_one_l673_673611


namespace Mn_equiv_l673_673130

def Sn (n : ℕ) (h : n > 1) : Type := { π : Fin n → Fin n // Function.Bijective π }

def F {n : ℕ} (h : n > 1) (π : Sn n h) : ℕ :=
∑ k : Fin n, |k.val - (π.1 k).val|

def Mn (n : ℕ) (h : n > 1) : ℚ :=
(1 : ℚ) / n.fact * ∑ π : Sn n h, F h π

theorem Mn_equiv (n : ℕ) (h : n > 1) : Mn n h = (n ^ 2 - 1) / 3 := by
  sorry

end Mn_equiv_l673_673130


namespace problem_statement_l673_673340

noncomputable def f (x : ℝ) := log ((1 - x) / (1 + x))

theorem problem_statement : (∀ x : ℝ, f(-x) = -f(x)) ∧ (∀ x y : ℝ, 0 < x → x < 1 → 0 < y → y < 1 → x < y → f(x) > f(y)) :=
by
  sorry

end problem_statement_l673_673340


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673218

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673218


namespace part_one_part_two_l673_673965

noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ := 
  ∑ i in Finset.range (n - 1), (Nat.choose n (n - i - 1)) * (x * (x + 1) * (x + i))

noncomputable def g_n (n : ℕ) (x : ℝ) : ℝ := 
  Nat.choose n n + (x * (x + 1) * (x + n - 1))

theorem part_one (n : ℕ) (hx : n ≥ 2) : 
  f_n n 1 = 7 * g_n n 1 → n = 15 :=
by sorry

theorem part_two (n : ℕ) (hx : n ≥ 2) (x : ℝ) : 
  f_n n x + g_n n x = 0 →
  x ∈ (Finset.range n).map (λ i, -(i : ℝ) - 1) :=
by sorry

end part_one_part_two_l673_673965


namespace complex_ratio_symmetry_l673_673590

noncomputable def z1 : ℂ := -1 + I
noncomputable def z2 : ℂ := 1 + I

theorem complex_ratio_symmetry :
  (∃ (z1 z2 : ℂ), z1 = -1 + I ∧ z2 = 1 + I ∧ (z1 / z2) = I) :=
by {
  use [z1, z2],
  simp [z1, z2, complex.div_eq_mul_conj, complex.mul_conj, I],
  sorry
}

end complex_ratio_symmetry_l673_673590


namespace area_of_triangle_pf1f2_l673_673507

open Real

noncomputable def ellipse : set (ℝ × ℝ) := {P : ℝ × ℝ | (P.1^2) / 49 + (P.2^2) / 24 = 1}

noncomputable def f1 : ℝ × ℝ := (-5, 0)
noncomputable def f2 : ℝ × ℝ := (5, 0)

noncomputable def line_slope (P Q : ℝ × ℝ) : ℝ := (P.2 - Q.2) / (P.1 - Q.1)

theorem area_of_triangle_pf1f2
  (P : ℝ × ℝ)
  (hP : P ∈ ellipse)
  (h_perpendicular : line_slope P f1 * line_slope P f2 = -1) :
  ∃ n : ℝ, (P.2 = n ∨ P.2 = -n) ∧
  (|n| = 24 / 5) ∧
  (let c := 5 in 1 / 2 * 2 * c * |n| = 24) :=
by
  sorry

end area_of_triangle_pf1f2_l673_673507


namespace maximum_MN_over_AB_l673_673180

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ :=
(0, p / 2)

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
- p / 2

def parabola (p : ℝ) (x y : ℝ) : Prop :=
y^2 = 2 * p * x

def is_midpoint {α : Type*} [has_add α] [has_scalar ℝ α] (A B M : α) : Prop :=
M = (A + B) / 2

def projected_distance {α : Type*} [normed_group α] (M : α) (L : set α) : ℝ :=
Inf (set.range (λ l ∈ L, dist M l))

variables (p : ℝ) (A B M : ℝ × ℝ) (l : ℝ) (theta : ℝ)
variables (h_p_pos : 0 < p)
variables (h_A_on_parabola : parabola p A.1 A.2)
variables (h_B_on_parabola : parabola p B.1 B.2)
variables (h_angle_AFB : ∀ F, F = parabola_focus p -> angle A F B = π / 3)
variables (h_midpoint : is_midpoint A B M)
variables (h_directrix : l = parabola_directrix p)
variables (h_projection : N = projected_distance M {L | L = l})

theorem maximum_MN_over_AB (p : ℝ) (A B M N : ℝ × ℝ) (h_p_pos : 0 < p)
  (h_A_on_parabola : parabola p A.1 A.2)
  (h_B_on_parabola : parabola p B.1 B.2)
  (h_angle_AFB : ∀ F, F = parabola_focus p -> angle A F B = π / 3)
  (h_midpoint : is_midpoint A B M)
  (h_projection :  N = projected_distance M {L | L = l})
  : abs (dist M N / dist A B) ≤ 1 := sorry

end maximum_MN_over_AB_l673_673180


namespace positive_value_of_A_l673_673135

theorem positive_value_of_A (A : ℝ) (h : A^2 + 3^2 = 130) : A = 11 :=
sorry

end positive_value_of_A_l673_673135


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673317

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673317


namespace largest_shaded_area_l673_673858

-- Define the side length of the squares
def side_length := 4

-- Define the radius of the inscribed circle in Figure X (half the side length)
def radius_X := side_length / 2

-- Calculate the shaded area of Figure X
def shaded_area_X := side_length^2 - Math.pi * (radius_X ^ 2)

-- Define the radius of the circles in Figure Y
def radius_Y := 1

-- Define the number of circles in Figure Y
def number_of_circles_Y := 4

-- Calculate the shaded area of Figure Y
def shaded_area_Y := side_length^2 - number_of_circles_Y * Math.pi * (radius_Y ^ 2)

-- Define the radius of the circumscribed circle in Figure Z
def radius_Z := side_length / 2

-- Calculate the shaded area of Figure Z
def shaded_area_Z := Math.pi * (radius_Z ^ 2) - side_length^2

-- The proof statement to be shown: Figures X and Y have the largest shaded area
theorem largest_shaded_area : shaded_area_X = shaded_area_Y ∧ shaded_area_X > shaded_area_Z := by
  -- Proof is omitted
  sorry

end largest_shaded_area_l673_673858


namespace largest_integer_dividing_consecutive_product_l673_673249

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l673_673249


namespace find_ratio_of_sides_l673_673105

-- Define the triangle and the corresponding conditions
variables (A B C : ℝ) (a b c : ℝ)
hypothesis (h1 : ∀ a b c A B C, √3 * a * Real.cos B = b * Real.sin A)
hypothesis (h2 : ∀ b, ∃ area, area = √3 / 4 * b^2)

-- The theorem statement
theorem find_ratio_of_sides (a b c : ℝ) (A B C : ℝ) 
    (h1 : √3 * a * Real.cos B = b * Real.sin A)
    (h2 : (1/2) * a * c * Real.sin B = √3 / 4 * b^2) :
    a / c = 1 :=
by sorry

end find_ratio_of_sides_l673_673105


namespace geometric_sequence_solution_l673_673973

-- Define the geometric sequence
def geom_seq (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n - 1)

-- Given conditions
variables (a₁ : ℝ) (q : ℝ)
variable (h₁ : a₁ = 64)
variable (h₂ : 2 * geom_seq a₁ q  4 - 3 * geom_seq a₁ q  3 + geom_seq a₁ q  2 = 0)
variable (h₃ : q ≠ 1)

-- Derived term a_n
def a_n (n : ℕ) : ℝ :=
  geom_seq a₁ q n

-- Sequence b_n
def b_n (n : ℕ) : ℤ :=
  Int.log 2 (a_n a₁ q n)

-- Sum of first n terms of the sequence |b_n|
def T_n (n : ℕ) : ℤ :=
  if n ≤ 7
  then n * (13 - n) / 2
  else (n^2 - 13*n + 84) / 2

theorem geometric_sequence_solution :
  a_n a₁ q = λ n, 2^(7 - n) ∧
  (∀ n, T_n n = 
    if n ≤ 7
    then n * (13 - n) / 2
    else (n^2 - 13*n + 84) / 2) :=
by
  sorry

end geometric_sequence_solution_l673_673973


namespace shorter_leg_of_right_triangle_l673_673579

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l673_673579


namespace fold_points_area_506_l673_673475

noncomputable def fold_point_area (AB AC : ℝ) (angleB : ℝ) : ℝ :=
  have radius := AB / 2
  π * radius^2

theorem fold_points_area_506 (AB AC : ℝ) (angleB : ℝ) 
  (hAB : AB = 45) 
  (hAC : AC = 45 * Real.sqrt 2) 
  (hangleB : angleB = π / 2) : 
  ∃ (q r s : ℕ), 
    (fold_point_area AB AC angleB = q * π - r * Real.sqrt s ∧ 
    q = 506 ∧ 
    r = 0 ∧ 
    s = 0) := 
begin
  use [506, 0, 0],
  simp [fold_point_area, hAB, hAC, hangleB],
  norm_num,
end

end fold_points_area_506_l673_673475


namespace plants_same_height_after_54_years_l673_673150

noncomputable def h1 (t : ℝ) : ℝ := 44 + (3 / 2) * t
noncomputable def h2 (t : ℝ) : ℝ := 80 + (5 / 6) * t

theorem plants_same_height_after_54_years :
  ∃ t : ℝ, h1 t = h2 t :=
by
  use 54
  sorry

end plants_same_height_after_54_years_l673_673150


namespace blacksmith_initial_iron_l673_673797

theorem blacksmith_initial_iron
  (num_farms : ℕ) (horses_per_farm : ℕ)
  (num_stables : ℕ) (horses_per_stable : ℕ)
  (num_riding_school_horses : ℕ)
  (iron_per_horseshoe : ℕ)
  (horseshoes_per_horse : ℕ) :
  num_farms = 2 →
  horses_per_farm = 2 →
  num_stables = 2 →
  horses_per_stable = 5 →
  num_riding_school_horses = 36 →
  iron_per_horseshoe = 2 →
  horseshoes_per_horse = 4 →
  let total_horses_farms := num_farms * horses_per_farm in
  let total_horses_stables := num_stables * horses_per_stable in
  let total_horses := total_horses_farms + total_horses_stables in
  let horseshoes_needed_farms_and_stables := total_horses * horseshoes_per_horse in
  let horseshoes_needed_riding_school := num_riding_school_horses * horseshoes_per_horse in
  let total_horseshoes_needed := horseshoes_needed_farms_and_stables + horseshoes_needed_riding_school in
  let total_iron_needed := total_horseshoes_needed * iron_per_horseshoe in
  total_iron_needed = 400 :=
by
  intros h_num_farms h_horses_per_farm h_num_stables h_horses_per_stable h_num_riding_school_horses h_iron_per_horseshoe h_horseshoes_per_horse
  simp [h_num_farms, h_horses_per_farm, h_num_stables, h_horses_per_stable, h_num_riding_school_horses, h_iron_per_horseshoe, h_horseshoes_per_horse]
  sorry

end blacksmith_initial_iron_l673_673797


namespace Amaya_total_marks_l673_673408

structure Scores where
  music : ℕ
  social_studies : ℕ
  arts : ℕ
  maths : ℕ

axiom Amaya_scores (A M S A_S : ℕ) : 
  (M = 70) ∧
  (S = M + 10) ∧
  (A_S = A - 20) ∧
  (A_S = (9/10 : ℝ) * A) →

  M + S + A + A_S = 530

theorem Amaya_total_marks (A M S A_S : ℕ)
  (h1 : M = 70)
  (h2 : S = M + 10)
  (h3 : A_S = A - 20)
  (h4 : A_S = (9/10 : ℝ) * A) :
  M + S + A + A_S = 530 := by
  apply Amaya_scores,
  exact ⟨h1, h2, h3, h4⟩,
  sorry

end Amaya_total_marks_l673_673408


namespace dinner_serving_problem_l673_673750

theorem dinner_serving_problem : 
  let orders := ["B", "B", "B", "B", "C", "C", "C", "C", "F", "F", "F", "F"].to_finset in
  let possible_serving_count := choose 12 2 * 160 in
  ∃ (serving : set (fin 12)), 
    (serving : cardinal) = 2 ∧
    (orders = serving) →
    possible_serving_count = 211200
:= 
begin
  sorry
end

end dinner_serving_problem_l673_673750


namespace smallest_composite_proof_l673_673903

-- Define what it means for a number not to have prime factors less than 15
def no_prime_factors_less_than_15 (n : ℕ) : Prop :=
  ∀ p : ℕ, nat.prime p → p ∣ n → p ≥ 15

-- Define what it means for a number to be the smallest composite number with the above property
def smallest_composite_without_prime_factors_less_than_15 (n : ℕ) : Prop :=
  nat.composite n ∧ no_prime_factors_less_than_15 n ∧
  ∀ m : ℕ, nat.composite m → no_prime_factors_less_than_15 m → n ≤ m

theorem smallest_composite_proof : smallest_composite_without_prime_factors_less_than_15 323 :=
  sorry

end smallest_composite_proof_l673_673903


namespace students_only_english_l673_673084

theorem students_only_english (total_students both_eng_ger total_ger students_only_eng : ℕ)
  (h1 : total_students = 50)
  (h2 : both_eng_ger = 12)
  (h3 : total_ger = 22)
  (h4 : total_students = students_only_eng + (total_ger - both_eng_ger) + both_eng_ger) :
  students_only_eng = 28 :=
by {
  -- Use given conditions
  rw [h1, h2, h3] at h4,
  -- Simplify equation
  sorry
}

end students_only_english_l673_673084


namespace shorter_leg_of_right_triangle_l673_673571

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h: a^2 + b^2 = c^2) (hn: c = 65): a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l673_673571


namespace largest_divisor_of_5_consecutive_integers_l673_673302

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l673_673302


namespace smallest_composite_proof_l673_673942

noncomputable def smallest_composite_no_prime_factors_less_than_15 : ℕ :=
  289

theorem smallest_composite_proof :
  smallest_composite_no_prime_factors_less_than_15 = 289 :=
by
  sorry

end smallest_composite_proof_l673_673942


namespace number_of_integer_B_values_is_six_l673_673954

def B (n : ℕ) : ℚ :=
  (∑ k in finset.range n, if k^3 ≤ n ∧ n < (k + 1)^3 then
   k * (6 * k^5 + 15 * k^4 + 14 * k^3 + 6 * k^2 + k) / 2 else 0)

theorem number_of_integer_B_values_is_six : 
  (finset.range 499).filter (λ n, B (n + 2)).length = 6 :=
sorry

end number_of_integer_B_values_is_six_l673_673954


namespace product_of_five_consecutive_divisible_by_30_l673_673273

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l673_673273


namespace tan_alpha_minus_2beta_l673_673000

variables (α β : ℝ)

-- Given conditions
def tan_alpha_minus_beta : ℝ := 2 / 5
def tan_beta : ℝ := 1 / 2

-- The statement to prove
theorem tan_alpha_minus_2beta (h1 : tan (α - β) = tan_alpha_minus_beta) (h2 : tan β = tan_beta) :
  tan (α - 2 * β) = -1 / 12 :=
sorry

end tan_alpha_minus_2beta_l673_673000


namespace max_t_for_real_root_l673_673999

theorem max_t_for_real_root (t : ℝ) (x : ℝ) 
  (h : 0 < x ∧ x < π ∧ (t+1) * Real.cos x - t * Real.sin x = t + 2) : t = -1 :=
sorry

end max_t_for_real_root_l673_673999


namespace largest_divisor_of_five_consecutive_integers_l673_673257

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l673_673257


namespace average_points_per_player_l673_673609

theorem average_points_per_player 
  (L R O : ℕ)
  (hL : L = 20) 
  (hR : R = L / 2) 
  (hO : O = 6 * R) 
  : (L + R + O) / 3 = 30 := by
  sorry

end average_points_per_player_l673_673609


namespace solve_for_a_l673_673059

theorem solve_for_a (a : ℚ) (h : a + a / 3 = 8 / 3) : a = 2 :=
sorry

end solve_for_a_l673_673059


namespace find_angle_C_find_angle_C_2_find_angle_C_3_range_of_4sinB_minus_a_l673_673851

variable {A B C a b c : ℝ}
variable (h1 : c^2 + a * b = c * (a * real.cos B - b * real.cos A) + 2 * b^2)
variable (h2 : (b + c) * (real.sin B - real.sin C) = -a * (real.sin A - real.sin B))
variable (h3 : b * real.sin C = real.sqrt 3 * (a - c * real.cos B))

theorem find_angle_C (h : c^2 + a * b = c * (a * real.cos B - b * real.cos A) + 2 * b^2) : C = real.pi / 3 :=
sorry

theorem find_angle_C_2 (h : (b + c) * (real.sin B - real.sin C) = -a * (real.sin A - real.sin B)) : C = real.pi / 3 :=
sorry

theorem find_angle_C_3 (h : b * real.sin C = real.sqrt 3 * (a - c * real.cos B)) : C = real.pi / 3 :=
sorry

theorem range_of_4sinB_minus_a (hC : C = real.pi / 3) (hc : c = 2 * real.sqrt 3) : 
  -2 * real.sqrt 3 < 4 * real.sin B - a ∧ 4 * real.sin B - a < 2 * real.sqrt 3 :=
sorry

end find_angle_C_find_angle_C_2_find_angle_C_3_range_of_4sinB_minus_a_l673_673851


namespace quadratic_decreasing_l673_673025

theorem quadratic_decreasing (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 ≤ x2 → x2 ≤ 4 → (x1^2 + 4*a*x1 - 2) ≥ (x2^2 + 4*a*x2 - 2)) : a ≤ -2 := 
by
  sorry

end quadratic_decreasing_l673_673025


namespace right_triangle_shorter_leg_l673_673567

theorem right_triangle_shorter_leg :
  ∃ (a b : ℤ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 16 :=
by
  sorry

end right_triangle_shorter_leg_l673_673567


namespace tan_angle_trigonometric_evaluation_l673_673365

-- Proof Problem 1
theorem tan_angle (α : ℝ) (m : ℝ) (h₁ : cos α = -1/3)
  (h₂ : m = -real.sqrt 2 / 4) : tan α = -2 * real.sqrt 2 :=
by sorry

-- Proof Problem 2
theorem trigonometric_evaluation :
  (tan 150 * cos (-210) * sin (-420)) / (sin 1050 * cos (-600)) = -real.sqrt 3 :=
by sorry

end tan_angle_trigonometric_evaluation_l673_673365


namespace inequality_holds_l673_673468

theorem inequality_holds (n : ℕ) (a : Fin (2 * n) → ℝ) 
  (h_sorted : ∀ i j : Fin (2 * n), i < j → a i < a j) 
  (h_positive : ∀ i : Fin (2 * n), 0 < a i) :
  let S := ∑ i in Finset.range n, a ⟨2 * i + 1, by linarith [i.2]⟩
  let T := ∑ i in Finset.range n, a ⟨2 * i + 2, by linarith [i.2]⟩
  S + T > 0 →
  2 * S * T > 
  sqrt((2 * n / (n - 1)) * (S + T) * (S * ∑ i in Finset.range n, ∑ j in Finset.Ico i.succ n, a ⟨2 * i + 2, by linarith [i.2]⟩ * a ⟨2 * j + 2, by linarith [j.2]⟩ + T * ∑ i in Finset.range n, ∑ j in Finset.Ico i.succ n, a ⟨2 * i + 1, by linarith [i.2]⟩ * a ⟨2 * j + 1, by linarith [j.2]⟩)) := 
by
  sorry

end inequality_holds_l673_673468


namespace greatest_alpha_in_triangle_l673_673953

def smallest_angle (ABC : Triangle) : ℝ :=
  min ABC.angleA (min ABC.angleB ABC.angleC)

theorem greatest_alpha_in_triangle (T : Triangle) : 
  ∃ α, (∀ P Q ∈ T, ∃ S, α_set S ∧ P ∈ S ∧ Q ∈ S ∧ S ⊆ T) ∧ α = smallest_angle T :=
begin
  sorry
end

end greatest_alpha_in_triangle_l673_673953


namespace Thabo_books_problem_l673_673686

theorem Thabo_books_problem 
  (P F : ℕ)
  (H1 : 180 = F + P + 30)
  (H2 : F = 2 * P)
  (H3 : P > 30) :
  P - 30 = 20 := 
sorry

end Thabo_books_problem_l673_673686


namespace initial_black_beads_l673_673423

theorem initial_black_beads (B : ℕ) : 
  let white_beads := 51
  let black_beads_removed := 1 / 6 * B
  let white_beads_removed := 1 / 3 * white_beads
  let total_beads_removed := 32
  white_beads_removed + black_beads_removed = total_beads_removed →
  B = 90 :=
by
  sorry

end initial_black_beads_l673_673423


namespace diagonal_length_AC_l673_673471

theorem diagonal_length_AC {A B C D : Type}
  (AB_length : ∥B - A∥ = 1)
  (AD_length : ∥D - A∥ = 1)
  (angle_A : ∠ (B - A) (D - A) = 160 * π / 180)
  (angle_C : ∠ (C - D) (A - D) = 100 * π / 180) :
  ∥C - A∥ = 1 := sorry

end diagonal_length_AC_l673_673471


namespace smallest_composite_no_prime_factors_less_than_15_l673_673913

-- Definitions used in the conditions
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

-- Prime numbers less than 15
def primes_less_than_15 (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13

-- Define the main proof statement
theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n : ℕ, is_composite n ∧ (∀ p : ℕ, p ∣ n → is_prime p → primes_less_than_15 p → false) ∧ n = 289 :=
by
  -- leave the proof as a placeholder
  sorry

end smallest_composite_no_prime_factors_less_than_15_l673_673913


namespace average_speed_distance_function_l673_673832

-- Define the times and corresponding distances based on the conditions
def times : List ℕ := [0, 2, 4, 6]
def distances : List ℕ := [200, 150, 100, 50]

-- Assertion for average speed
theorem average_speed :
  (1 / 4) * (Σ i in [0, 2..6], distances[i / 2] - distances[i / 2 + 1]) / 2 = 25 := sorry

-- Assertion for distance function
theorem distance_function (x : ℕ) (h : 0 ≤ x ∧ x ≤ 8) :
  let y := 200 - 25 * x in y = distances[(8 - x) / 2] := sorry

end average_speed_distance_function_l673_673832


namespace prove_problem_statement_l673_673597

open Real
open Triangle

noncomputable def problem_statement : Prop :=
  ∀ (A B C : Point) (O : Point),
  is_obtuse (∠ A) ∧ is_orthocenter O A B C ∧ dist A O = dist B C →
  cos (angle O B C + angle O C B) = -Real.sqrt 2 / 2

theorem prove_problem_statement : problem_statement :=
by {
  intros A B C O h1 h2 h3,
  -- Proof would go here
  sorry
}

end prove_problem_statement_l673_673597


namespace largest_divisor_of_consecutive_product_l673_673225

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l673_673225


namespace right_triangle_shorter_leg_l673_673565

theorem right_triangle_shorter_leg :
  ∃ (a b : ℤ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 16 :=
by
  sorry

end right_triangle_shorter_leg_l673_673565


namespace probability_of_gcd_one_is_13_over_14_l673_673735

open Finset

noncomputable def probability_gcd_one : ℚ :=
let s := {1, 2, 3, 4, 5, 6, 7, 8}
let subsetsOfThree := s.powerset.filter (λ t, t.card = 3)
let nonRelativelyPrimeSubsets := {(t : Finset ℕ) ∈ subsetsOfThree | (∀ a b c ∈ t, gcd (gcd a b) c ≠ 1)}
let totalSubsets := subsetsOfThree.card
let nonRelativelyPrimeCount := nonRelativelyPrimeSubsets.card
in 1 - (nonRelativelyPrimeCount / totalSubsets : ℚ)

theorem probability_of_gcd_one_is_13_over_14 :
  probability_gcd_one = 13 / 14 := by sorry

end probability_of_gcd_one_is_13_over_14_l673_673735


namespace problem_condition_neither_sufficient_nor_necessary_l673_673060

theorem problem_condition_neither_sufficient_nor_necessary 
  (m n : ℕ) (hm : m > 0) (hn : n > 0) (a b : ℝ) :
  (a > b → a^(m + n) + b^(m + n) > a^n * b^m + a^m * b^n) ∧
  (a^(m + n) + b^(m + n) > a^n * b^m + a^m * b^n → a > b) = false :=
by sorry

end problem_condition_neither_sufficient_nor_necessary_l673_673060


namespace median_length_is_four_l673_673181

def name_lengths : List ℕ :=
  [3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7]

def median {α : Type*} [LinearOrder α] (l : List α) : α :=
  let sorted := l.qsort (· ≤ ·)
  sorted.get! (sorted.length / 2)

theorem median_length_is_four : median name_lengths = 4 := by
  sorry

end median_length_is_four_l673_673181


namespace polygon_sides_eq_seven_l673_673076

-- Given conditions:
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180
def sum_exterior_angles : ℝ := 360
def difference_in_angles (n : ℕ) : ℝ := sum_interior_angles n - sum_exterior_angles

-- Proof statement:
theorem polygon_sides_eq_seven (n : ℕ) (h : difference_in_angles n = 540) : n = 7 := sorry

end polygon_sides_eq_seven_l673_673076


namespace length_PQ_eq_five_l673_673449

-- Define the polar equation of line l1
def polar_eq_l1 (ρ θ : ℝ) : Prop :=
  2 * ρ * sin (θ + π / 3) + 3 * sqrt 3 = 0

-- Define the polar equation of line l, which is a simple condition on θ
def polar_eq_l (ρ θ : ℝ) : Prop := 
  θ = π / 3

-- Define the family of points on the curve C
def point_on_curve (ρ θ : ℝ) : Prop :=
  ρ^2 - 2 * ρ * cos θ - 2 = 0

-- Define the point P lies on curve C and line l
def point_P (ρ θ : ℝ) : Prop :=
  point_on_curve ρ θ ∧ polar_eq_l ρ θ

-- Define the point Q lies on line l and line l1
def point_Q (ρ θ : ℝ) : Prop :=
  polar_eq_l1 ρ θ ∧ polar_eq_l ρ θ

-- Define the length of the line segment PQ
def length_PQ (ρ_P ρ_Q : ℝ) : ℝ :=
  abs (ρ_P - ρ_Q)

-- Now we state the theorem to prove
theorem length_PQ_eq_five :
  ∃ ρ_P ρ_Q θ, point_P ρ_P θ ∧ point_Q ρ_Q θ ∧ length_PQ ρ_P ρ_Q = 5 :=
  sorry

end length_PQ_eq_five_l673_673449


namespace arrow_existence_in_grid_l673_673783

-- Definitions of directions and grid
inductive Direction
| up | down | left | right

structure Cell :=
  (has_arrow : Bool)
  (direction : Option Direction) -- None if no arrow

-- 20x20 grid representation
def Grid := Array (Array Cell)

-- Definition of the problem conditions
def condition1 (grid : Grid) : Prop :=
  grid.size = 20 ∧ ∀ row, (grid[row].size = 20)

def condition2 (grid : Grid) : Prop :=
  ∀ i : Fin 20, (
    grid[0][i].has_arrow ∧ grid[0][i].direction = some Direction.right ∧
    grid[19][i].has_arrow ∧ grid[19][i].direction = some Direction.left ∧
    grid[i][0].has_arrow ∧ grid[i][0].direction = some Direction.up ∧
    grid[i][19].has_arrow ∧ grid[i][19].direction = some Direction.down
  )

def is_adjacent (r1 c1 r2 c2: Nat) : Prop :=
  (r1 = r2 ∧ (c1 = c2+1 ∨ c1+1 = c2)) ∨ (c1 = c2 ∧ (r1 = r2+1 ∨ r1+1 = r2)) ∨
  (r1 = r2+1 ∧ c1 = c2+1) ∨ (r1+1 = r2 ∧ c1+1 = c2)

def condition3 (grid : Grid) : Prop :=
  ∀ r1 c1 r2 c2 : Fin 20, 
  is_adjacent r1.val c1.val r2.val c2.val → 
  ∼(grid[r1][c1].has_arrow ∧ grid[r2][c2].has_arrow ∧ 
    match (grid[r1][c1].direction, grid[r2][c2].direction) with 
    | (some Direction.up, some Direction.down) | 
      (some Direction.down, some Direction.up) | 
      (some Direction.left, some Direction.right) | 
      (some Direction.right, some Direction.left) => True
    | _ => False
    end)

-- Statement of the theorem to prove
theorem arrow_existence_in_grid (grid : Grid) 
  (cond1 : condition1 grid)
  (cond2 : condition2 grid) 
  (cond3 : condition3 grid)
  : ∃ r c : Fin 20, ∼ grid[r][c].has_arrow :=
by
  sorry

end arrow_existence_in_grid_l673_673783


namespace sequence_contains_1_or_4_l673_673975

def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).map (λ d, d * d).sum

theorem sequence_contains_1_or_4 (a₁ : ℕ) 
  (h₁ : 100 ≤ a₁ ∧ a₁ < 1000) : 
  ∃ n, let a := λ n, Nat.iterate sum_of_squares_of_digits n a₁ in a n = 1 ∨ a n = 4 := 
sorry

end sequence_contains_1_or_4_l673_673975


namespace decipher_numbers_l673_673600

variable (K I S : Nat)

theorem decipher_numbers
  (h1: 1 ≤ K ∧ K < 5)
  (h2: I ≠ 0)
  (h3: I ≠ K)
  (h_eq: K * 100 + I * 10 + S + K * 10 + S * 10 + I = I * 100 + S * 10 + K):
  (K, I, S) = (4, 9, 5) :=
by sorry

end decipher_numbers_l673_673600


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673214

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673214


namespace interesting_pairs_ratio_l673_673019

variable {A B C E1 E2 F1 F2 : Type}
variable [Field A] [Field B] [Field C] 
variable (triangle : Type)
variable [Has_Coords A] [Has_Coords B] [Has_Coords C] 

-- Defining points E and F
variable (E : A → C) (F : A → B) (M : Type)
variable [Midpoint E F M]

-- Perpendicular bisector intersects
variable (K : Type) [Perp Bisector E F K]
variable (S : Type) (T : Type)
variable [Intersects_Perpendicular AC M K S] [Intersects_Perpendicular AB M K T]
variable [Concyclic K S A T]

-- Main theorem statement
theorem interesting_pairs_ratio :
  (E1 F1 E2 F2 : Type) [Interesting E1 F1] [Interesting E2 F2] → 
  ((distance E1 E2 / distance A B) = (distance F1 F2 / distance A C)) :=
sorry

end interesting_pairs_ratio_l673_673019


namespace powerjet_pumps_250_gallons_in_30_minutes_l673_673688

theorem powerjet_pumps_250_gallons_in_30_minutes :
  let rate : ℝ := 500
  let time_in_hours : ℝ := 1 / 2
  rate * time_in_hours = 250 :=
by
  sorry

end powerjet_pumps_250_gallons_in_30_minutes_l673_673688


namespace total_worth_l673_673860

-- Definitions from the conditions
def sales_tax := 0.30 -- rupees
def tax_rate := 0.06 -- 6%
def tax_free_items_cost := 19.7 -- rupees

-- Given the sales_tax = tax_rate * x, where x is the cost of taxable items
-- and the total worth is the sum of the taxable and tax-free items
theorem total_worth : 
  ∃ (x : ℝ), (tax_rate * x = sales_tax) ∧ (tax_free_items_cost + x = 24.7) :=
by
  sorry

end total_worth_l673_673860


namespace average_salary_correct_l673_673714

/-- The salaries of A, B, C, D, and E. -/
def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

/-- The number of people. -/
def number_of_people : ℕ := 5

/-- The total salary is the sum of the salaries. -/
def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E

/-- The average salary is the total salary divided by the number of people. -/
def average_salary : ℕ := total_salary / number_of_people

/-- The average salary of A, B, C, D, and E is Rs. 8000. -/
theorem average_salary_correct : average_salary = 8000 := by
  sorry

end average_salary_correct_l673_673714


namespace min_value_of_f_l673_673057

noncomputable def f (x : ℝ) : ℝ := (1 / x) + (2 * x / (1 - x))

theorem min_value_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < 1) : 
  (∀ y, 0 < y ∧ y < 1 → f y ≥ 1 + 2 * Real.sqrt 2) := 
sorry

end min_value_of_f_l673_673057


namespace isosceles_triangle_of_condition_l673_673545

theorem isosceles_triangle_of_condition (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 2 * b * Real.cos C)
  (h2 : A + B + C = Real.pi) :
  (B = C) ∨ (A = C) ∨ (A = B) := 
sorry

end isosceles_triangle_of_condition_l673_673545


namespace arithmetic_sequence_n_l673_673480

theorem arithmetic_sequence_n (a1 d an n : ℕ) (h1 : a1 = 1) (h2 : d = 3) (h3 : an = 298) (h4 : an = a1 + (n - 1) * d) : n = 100 :=
by
  sorry

end arithmetic_sequence_n_l673_673480


namespace sum_of_first_6033_terms_l673_673720

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := 
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_6033_terms (a r : ℝ) (h1 : geometric_sum a r 2011 = 200) 
  (h2 : geometric_sum a r 4022 = 380) : 
  geometric_sum a r 6033 = 542 :=
sorry

end sum_of_first_6033_terms_l673_673720


namespace shorter_leg_of_right_triangle_l673_673576

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h: a^2 + b^2 = c^2) (hn: c = 65): a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l673_673576


namespace right_triangle_classification_l673_673345

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_classification :
  (
    ¬ is_right_triangle 3 4 6 ∧
    ¬ is_right_triangle 5 12 14 ∧
    is_right_triangle 1 (sqrt 3) 2 ∧
    ¬ is_right_triangle (sqrt 2) (sqrt 3) 2
  ) :=
by {
  -- Proof omitted
  sorry
}

end right_triangle_classification_l673_673345


namespace area_triangle_ADC_l673_673596

open Real

-- Definitions and conditions
def triangle_ABC (A B C D : Point) :=
  angle_triangle_ABC_is_90 (A B C : Point) ∧ 
  is_angle_bisector (AD : Line) ∧ 
  AB = 100 ∧ 
  BC = y ∧ 
  AC = 2 * y - 10 ∧ 
  perimeter_ABC (A B C : Point) = 290

-- Statement of the problem
theorem area_triangle_ADC (A B C D : Point) (y : ℝ) 
  (h : triangle_ABC A B C D) : 
  area (A D C : Triangle) = 1957 :=
sorry

end area_triangle_ADC_l673_673596


namespace lengths_of_train_and_car_l673_673835

-- Define the conditions as hypotheses
variables (speed_train_kmh : ℝ) (time_pole_s : ℝ) (speed_car_kmh : ℝ) (time_pass_car_s : ℝ)
variables (speed_train_mps speed_car_mps rel_speed_mps : ℝ)

-- Assume the given conditions
axiom h_train_speed : speed_train_kmh = 60
axiom h_time_pole : time_pole_s = 36
axiom h_car_speed : speed_car_kmh = 80
axiom h_time_pass_car : time_pass_car_s = 72

-- Convert km/hr to m/s
axiom h_train_speed_mps : speed_train_mps = speed_train_kmh * (1000 / 3600)
axiom h_car_speed_mps : speed_car_mps = speed_car_kmh * (1000 / 3600)

-- Compute relative speed
axiom h_rel_speed_mps : rel_speed_mps = abs (speed_train_mps - speed_car_mps)

-- Length of the train and car
def length_train : ℝ := speed_train_mps * time_pole_s
def length_car : ℝ := rel_speed_mps * time_pass_car_s

-- Statement of the problem
theorem lengths_of_train_and_car :
  length_train = 600 ∧ length_car = 399.6 :=
by
  sorry

end lengths_of_train_and_car_l673_673835


namespace solve_for_a_l673_673661

/--
Given the conditions:
1. \(\cos 3a = 0\)
2. \(\sin 3a - \sin 7a = 0\)

Prove that:
\[ a = \frac{\pi (2t + 1)}{2}, \, t \in \mathbb{Z} \]
-/
theorem solve_for_a (a : ℝ) (t : ℤ) :
  (cos (3 * a) = 0) ∧ (sin (3 * a) = sin (7 * a)) ↔
  (∃ t : ℤ, a = (π * (2 * t + 1)) / 2) := by
  sorry

end solve_for_a_l673_673661


namespace largest_divisor_of_5_consecutive_integers_l673_673330

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l673_673330


namespace right_triangle_shorter_leg_l673_673560

theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : c = 65) : a = 25 ∨ b = 25 := 
by
  sorry

end right_triangle_shorter_leg_l673_673560


namespace find_natural_pairs_l673_673445

-- Definitions
def is_natural (n : ℕ) : Prop := n > 0
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1
def satisfies_equation (x y : ℕ) : Prop := 2 * x^2 + 5 * x * y + 3 * y^2 = 41 * x + 62 * y + 21

-- Problem statement
theorem find_natural_pairs (x y : ℕ) (hx : is_natural x) (hy : is_natural y) (hrel : relatively_prime x y) :
  satisfies_equation x y ↔ (x = 2 ∧ y = 19) ∨ (x = 19 ∧ y = 2) :=
by
  sorry

end find_natural_pairs_l673_673445


namespace Jane_is_currently_40_l673_673605

variable (Jane_current_age : ℕ)
variable (stopped_babysitting_years_ago : ℕ := 10)
variable (oldest_babysat_child_current_age : ℕ := 25)
variable (Jane_stopping_babysitting_age : ℕ)
variable (oldest_babysat_child_age_at_stopping : ℕ)

noncomputable def age_proof : Prop :=
  Jane_current_age = 40

theorem Jane_is_currently_40
  (stopped_babysitting_years_ago_eq : stopped_babysitting_years_ago = 10)
  (oldest_babysat_child_current_age_eq : oldest_babysat_child_current_age = 25)
  (babysat_child_age_calculation : oldest_babysat_child_age_at_stopping = oldest_babysat_child_current_age - stopped_babysitting_years_ago)
  (babysat_age_limit : ∀ t, t ≤ Jane_stopping_babysitting_age / 2 → t = oldest_babysat_child_age_at_stopping)
  (stopping_age_calculation : Jane_stopping_babysitting_age = oldest_babysat_child_age_at_stopping * 2)
  (current_age_calculation : Jane_current_age = Jane_stopping_babysitting_age + stopped_babysitting_years_ago)
: age_proof := by sorry

end Jane_is_currently_40_l673_673605


namespace kim_shirts_left_l673_673122

theorem kim_shirts_left (initial_dozens : ℕ) (fraction_given : ℚ) (num_pairs : ℕ)
  (h1 : initial_dozens = 4) 
  (h2 : fraction_given = 1 / 3)
  (h3 : num_pairs = initial_dozens * 12)
  (h4 : num_pairs * fraction_given  = (16 : ℕ)):
  48 - ((num_pairs * fraction_given).toNat) = 32 :=
by 
  sorry

end kim_shirts_left_l673_673122


namespace katie_has_more_games_l673_673118

   -- Conditions
   def katie_games : Nat := 81
   def friends_games : Nat := 59

   -- Problem statement
   theorem katie_has_more_games : (katie_games - friends_games) = 22 :=
   by
     -- Proof to be provided
     sorry
   
end katie_has_more_games_l673_673118


namespace intersection_points_l673_673771

theorem intersection_points (a : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = a^2) ↔ (y = x^2 - 2 * a)) ↔ (0 < a ∧ a < 1) :=
sorry

end intersection_points_l673_673771


namespace find_two_digit_number_l673_673745

theorem find_two_digit_number (a : ℕ) (ha1 : 10 ≤ a ∧ a < 100) (ha2 : (101 * a - a^2) % (0.04 * a^2) = a) (ha3 : (101 * a - a^2) / (0.04 * a^2) = a / 2): a = 50 :=
sorry

end find_two_digit_number_l673_673745


namespace largest_divisor_of_5_consecutive_integers_l673_673323

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l673_673323


namespace cost_of_article_l673_673067

variable {C G : ℝ}

theorem cost_of_article (h : 350 = C * (1 + (G + 5) / 100)) (h' : 340 = C * (1 + G / 100)) : C = 200 := by
  sorry

end cost_of_article_l673_673067


namespace smallest_angle_in_convex_polygon_l673_673696

theorem smallest_angle_in_convex_polygon :
  ∀ (n : ℕ) (angles : ℕ → ℕ) (d : ℕ), n = 25 → (∀ i, 1 ≤ i ∧ i ≤ n → angles i = 166 - 1 * (13 - i)) 
  → 1 ≤ d ∧ d ≤ 1 → (angles 1 = 154) := 
by
  sorry

end smallest_angle_in_convex_polygon_l673_673696


namespace log_problem_l673_673534

noncomputable def equivalentProofProblem : Prop :=
  ∀ (x : ℝ), log 4 (9 * x) = 3 → log x 64 = 9 / 2

theorem log_problem (x : ℝ) (h : log 4 (9 * x) = 3) : log x 64 = 9 / 2 :=
by
  sorry

end log_problem_l673_673534


namespace tangent_circles_radius_l673_673538

theorem tangent_circles_radius (O1 O2 : Type) [MetricSpace O1] [MetricSpace O2]
  (O1O2_dist : dist O1 O2 = 5) (r1 : ℝ) (h_r1 : r1 = 2) (r2 : ℝ) :
  (r1 + r2 = 5) ∨ (r2 - r1 = 5) → (r2 = 3) ∨ (r2 = 7) :=
by
  intro h
  sorry

end tangent_circles_radius_l673_673538


namespace max_subset_elements_l673_673628

theorem max_subset_elements : 
  ∃ (S : set ℕ), 
    (∀ x ∈ S, x ∈ {x | x ≤ 1989}) ∧
    (∀ x y ∈ S, x ≠ y → |x - y| ≠ 4 ∧ |x - y| ≠ 7) ∧
    S.card = 905 := 
sorry

end max_subset_elements_l673_673628


namespace smallest_composite_no_prime_factors_less_than_15_l673_673914

-- Definitions used in the conditions
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

-- Prime numbers less than 15
def primes_less_than_15 (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13

-- Define the main proof statement
theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n : ℕ, is_composite n ∧ (∀ p : ℕ, p ∣ n → is_prime p → primes_less_than_15 p → false) ∧ n = 289 :=
by
  -- leave the proof as a placeholder
  sorry

end smallest_composite_no_prime_factors_less_than_15_l673_673914


namespace magnitude_of_z_l673_673968

noncomputable def z (a : ℝ) : ℂ := a + 2 * complex.I

-- The defining condition that z^2 is purely imaginary
def z_squared_is_pure_imaginary (a : ℝ) : Prop :=
  (z a).re * z a = 0

theorem magnitude_of_z (a : ℝ) (hz : z_squared_is_pure_imaginary a) : complex.abs (z a) = 2 * real.sqrt 2 :=
sorry

end magnitude_of_z_l673_673968


namespace ratio_junk_food_to_allowance_l673_673413
noncomputable theory

-- Define the weekly allowance, expenditure on sweets, and savings.
def weekly_allowance : ℕ := 30
def sweets_expenditure : ℕ := 8
def savings : ℕ := 12

-- Define the amount spent on junk food.
def junk_food_expenditure : ℕ := weekly_allowance - (sweets_expenditure + savings)

-- Theorem statement: the ratio of the amount spent on junk food to the weekly allowance is 1:3
theorem ratio_junk_food_to_allowance :
  (junk_food_expenditure : ℚ) / weekly_allowance = 1 / 3 :=
by sorry

end ratio_junk_food_to_allowance_l673_673413


namespace vertical_asymptote_exists_l673_673959

-- Given conditions: g(x) = (x^2 - 2x + b) / (x^2 - 3x + 2) with x^2 - 3x + 2 = (x-1)(x-2)
theorem vertical_asymptote_exists (b : ℝ) :
  (∃ x : ℝ, x ≠ 1 ∧ x ≠ 2 ∧ g x = 1) ↔ (b = 1 ∨ b = 0) :=
by
  -- Definition of g(x) based on given conditions
  let g (x : ℝ) := (x^2 - 2*x + b) / (x^2 - 3*x + 2)
  
  -- Factor the denominator
  have h : x^2 - 3*x + 2 = (x-1)*(x-2) := by ring
  
  -- Proof part is omitted
  sorry

end vertical_asymptote_exists_l673_673959


namespace rectangular_cube_length_l673_673396

theorem rectangular_cube_length (L : ℝ) (h1 : 2 * (L * 2) + 2 * (L * 0.5) + 2 * (2 * 0.5) = 24) : L = 4.6 := 
by {
  sorry
}

end rectangular_cube_length_l673_673396


namespace circle_polar_equivalences_l673_673367

-- Define the parametric equations of the circle C
def parametric_circle (φ : ℝ) : ℝ × ℝ :=
  let x := 2 * cos φ + 2
  let y := 2 * sin φ
  (x, y)

-- Define the standard Cartesian equation of the circle C
def cartesian_circle (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + y ^ 2 = 4

-- Define the polar coordinate equation of the circle C
def polar_circle (ρ θ : ℝ) : Prop :=
  ρ = 4 * cos θ

-- Define the polar coordinate equation of the line passing through A
def polar_line (ρ θ : ℝ) : Prop :=
  ρ * cos θ = 4

-- Main theorem stating the equivalences
theorem circle_polar_equivalences (φ θ ρ : ℝ) :
  ∃ x y : ℝ, parametric_circle φ = (x, y) →
  cartesian_circle x y →
  polar_circle ρ θ ∧ polar_line ρ θ :=
by {
  sorry -- proof steps go here
}

end circle_polar_equivalences_l673_673367


namespace largest_divisor_of_product_of_5_consecutive_integers_l673_673239

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l673_673239


namespace carpet_covers_60_percent_of_floor_l673_673379

-- Define the known dimensions of the carpet
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9

-- Define the area of the living room floor
def living_room_area : ℝ := 60

-- Calculate the area of the carpet
def carpet_area : ℝ := carpet_length * carpet_width

-- Define the target percentage calculation
def percent_covered_by_carpet : ℝ := (carpet_area / living_room_area) * 100

-- Proposition stating that the percentage of the floor covered by the carpet is 60%
theorem carpet_covers_60_percent_of_floor : percent_covered_by_carpet = 60 := by
  sorry

end carpet_covers_60_percent_of_floor_l673_673379


namespace max_product_decomposition_l673_673043

theorem max_product_decomposition (n k : ℕ) (hnk : k ≤ n) :
  ∃ (u t : ℕ), u = n / k ∧ t = n % k ∧
  ((n - (n / k) * k) = t ∧ k - (n - (n / k) * k) = k - t ∧
  (∀ (d : list ℕ), (d.length = k) → list.sum d = n → list.product d ≤ (list.product (list.repeat (n / k + 1) t ++ list.repeat (n / k) (k - t))) )) :=
begin
  -- Proof not required as per instructions
  sorry
end

end max_product_decomposition_l673_673043


namespace symmetrical_character_is_C_l673_673089

-- Definitions of the characters and the concept of symmetry
def is_symmetrical (char: Char): Prop := 
  match char with
  | '中' => True
  | _ => False

-- The options given in the problem
def optionA := '爱'
def optionB := '我'
def optionC := '中'
def optionD := '国'

-- The problem statement: Prove that among the given options, the symmetrical character is 中.
theorem symmetrical_character_is_C : (is_symmetrical optionA = False) ∧ (is_symmetrical optionB = False) ∧ (is_symmetrical optionC = True) ∧ (is_symmetrical optionD = False) :=
by
  sorry

end symmetrical_character_is_C_l673_673089


namespace dot_product_bc_l673_673620

variables (a b c : EuclideanSpace ℝ (Fin 3)) -- since we are dealing with R^3 vectors

-- Define the conditions
def norm_a : ∥a∥ = 1 := sorry
def norm_b : ∥b∥ = 1 := sorry
def norm_a_b : ∥a + b∥ = Real.sqrt 2 := sorry
def c_def : c - a - 2 • b = 4 • (a × b) := sorry

-- Define the theorem to prove
theorem dot_product_bc :
  ∥a∥ = 1 → ∥b∥ = 1 → ∥a + b∥ = Real.sqrt 2 → c - a - 2 • b = 4 • (a × b) → b ⬝ c = 2 :=
by 
  intros h1 h2 h3 h4
  sorry

end dot_product_bc_l673_673620


namespace largest_integer_dividing_consecutive_product_l673_673252

theorem largest_integer_dividing_consecutive_product :
  ∀ (n : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 120 = 0 :=
by
  sorry

end largest_integer_dividing_consecutive_product_l673_673252


namespace complex_number_in_third_quadrant_l673_673095

-- Definitions derived from conditions
def complex_number := (3 : ℂ) - (2 : ℂ) * complex.I
def divisor := (0 : ℂ) + (1 : ℂ) * complex.I
def quotient := complex_number / divisor

-- Plain condition checking
def is_in_third_quadrant (z : ℂ) : Prop :=
  (z.re < 0) ∧ (z.im < 0)

-- Problem Statement
theorem complex_number_in_third_quadrant :
  is_in_third_quadrant quotient :=
sorry

end complex_number_in_third_quadrant_l673_673095


namespace exceptional_points_lie_on_same_circle_l673_673519

def isExceptionalPoint (Γ₁ Γ₂ Γ₃ : Circle) (P : Point) : Prop :=
  ∃ A₁ B₁ A₂ B₂ A₃ B₃ Q : Point,
    (A₁ ∈ Γ₁ ∧ B₁ ∈ Γ₁ ∧ A₂ ∈ Γ₂ ∧ B₂ ∈ Γ₂ ∧ A₃ ∈ Γ₃ ∧ B₃ ∈ Γ₃) ∧
    (LineThroughPoints P A₁).isTangentToCircle Γ₁ ∧
    (LineThroughPoints P B₁).isTangentToCircle Γ₁ ∧
    (LineThroughPoints P A₂).isTangentToCircle Γ₂ ∧
    (LineThroughPoints P B₂).isTangentToCircle Γ₂ ∧
    (LineThroughPoints P A₃).isTangentToCircle Γ₃ ∧
    (LineThroughPoints P B₃).isTangentToCircle Γ₃ ∧
    (LineThroughPoints A₁ B₁).containsPoint Q ∧
    (LineThroughPoints A₂ B₂).containsPoint Q ∧
    (LineThroughPoints A₃ B₃).containsPoint Q

theorem exceptional_points_lie_on_same_circle
  (Γ₁ Γ₂ Γ₃ : Circle) :
  (∃ P : Point, isExceptionalPoint Γ₁ Γ₂ Γ₃ P) →
  ∃ Γ : Circle, ∀ P : Point, isExceptionalPoint Γ₁ Γ₂ Γ₃ P → P ∈ Γ :=
by
  sorry

end exceptional_points_lie_on_same_circle_l673_673519


namespace john_threw_away_19_socks_l673_673607

def john_socks (x : ℕ) : Prop :=
  let original_socks := 33 in
  let new_socks := 13 in
  let current_socks := 27 in
  original_socks - x + new_socks = current_socks

theorem john_threw_away_19_socks : john_socks 19 :=
by {
  let original_socks := 33
  let new_socks := 13
  let current_socks := 27
  show original_socks - 19 + new_socks = current_socks,
  -- Reflected calculation from problem steps
  rw [←add_sub_cancel 33 19, show 33 - 19, from rfl],
  rw [←add_assoc, show 14 + 13, from rfl],
  refl
}

end john_threw_away_19_socks_l673_673607


namespace probability_of_selecting_green_ball_l673_673788

def container_I :  ℕ × ℕ := (5, 5) -- (red balls, green balls)
def container_II : ℕ × ℕ := (3, 3) -- (red balls, green balls)
def container_III : ℕ × ℕ := (4, 2) -- (red balls, green balls)
def container_IV : ℕ × ℕ := (6, 6) -- (red balls, green balls)

def total_containers : ℕ := 4

def probability_of_green_ball (red_green : ℕ × ℕ) : ℚ :=
  let (red, green) := red_green
  green / (red + green)

noncomputable def combined_probability_of_green_ball : ℚ :=
  (1 / total_containers) *
  (probability_of_green_ball container_I +
   probability_of_green_ball container_II +
   probability_of_green_ball container_III +
   probability_of_green_ball container_IV)

theorem probability_of_selecting_green_ball : 
  combined_probability_of_green_ball = 11 / 24 :=
sorry

end probability_of_selecting_green_ball_l673_673788


namespace no_closed_broken_line_odd_segments_equal_length_l673_673598

theorem no_closed_broken_line_odd_segments_equal_length :
  ∀ (A : ℕ → (ℤ × ℤ)), 
  (∃ n : ℕ, odd n ∧
   (∀ i, dist (A (i % n)) (A ((i + 1) % n)) = 1) ∧
   (A 0 = A n)) → 
  false :=
by
  sorry

end no_closed_broken_line_odd_segments_equal_length_l673_673598


namespace needle_lines_tangent_to_circles_l673_673459

theorem needle_lines_tangent_to_circles (x y : ℤ) (h : x ≠ 0 ∨ y ≠ 0) :
  let φ := arctan ((y*y - 2*x*y - x*x) / (y*y + 2*x*y - x*x))
  in ∃ C : ℝ, ∃ r : ℝ, (C ≠ 0) ∧ (r ≠ 0) ∧ (C = (x + y)/sqrt 2) ∧ (r = sqrt ((x - y)^2 + (x + y)^2)) ∧
     (x - C)^2 + (y - C)^2 = r^2 :=
by
  sorry

end needle_lines_tangent_to_circles_l673_673459


namespace pure_imaginary_l673_673997

-- Given conditions
def real_part_zero (m : ℝ) : Prop := m^2 + m - 2 = 0
def imag_part_nonzero (m : ℝ) : Prop := m^2 + 4m - 5 ≠ 0

-- Prove that if the real part is zero and the imaginary part is non-zero, then m must be -2
theorem pure_imaginary (m : ℝ) : real_part_zero m ∧ imag_part_nonzero m → m = -2 := 
by sorry

end pure_imaginary_l673_673997


namespace at_least_30_cents_probability_l673_673676

theorem at_least_30_cents_probability :
  let penny := 1
  let nickel := 5
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  let all_possible_outcomes := 2^5
  let successful_outcomes := 
    -- Half-dollar and quarter heads: 2^3 = 8 combinations
    2^3 + 
    -- Quarter heads and half-dollar tails (nickel and dime heads): 2 combinations
    2^1 + 
    -- Quarter tails and half-dollar heads: 2^3 = 8 combinations
    2^3
  let probability := successful_outcomes / all_possible_outcomes
  probability = 9 / 16 :=
by
  -- Proof goes here
  sorry

end at_least_30_cents_probability_l673_673676


namespace friedahops_l673_673962

theorem friedahops (
    P : ℝ :=
    let p := 1/5 in
    let S := { (i, j) | (i = 2 ∧ j = 1) ∨ (i = 2 ∧ j = 2) ∨ (i = 2 ∧ j = 3) } in
    let E := { (i, j) | i = 1 ∨ i = 4 ∨ j = 1 ∨ j = 4 } in
    1/4 * (p + p + p + 0 + 0) ^ 5) =
  605/625 :=
sorry

end friedahops_l673_673962


namespace sum_nimo_is_9765_l673_673610

open finset

def perm_five : finset (fin 5 → fin 5) := univ.filter (λ f, function.bijective f)

def nimo (s : fin 5 → fin 5) : ℕ :=
(list.fin_range 4).sum (λ i, if s i > s (i + 1) then 1 else 0)

noncomputable def sum_nimo : ℕ :=
perm_five.sum (λ s, 2 ^ nimo s)

theorem sum_nimo_is_9765 : sum_nimo = 9765 :=
sorry

end sum_nimo_is_9765_l673_673610


namespace trains_meet_time_l673_673360

/-- Define the initial conditions. -/
def length_train1 : ℝ := 100
def length_train2 : ℝ := 200
def initial_distance : ℝ := 70
def speed_train1_kmph : ℝ := 54
def speed_train2_kmph : ℝ := 72

/-- Convert speeds from kmph to mps. -/
def speed_train1_mps : ℝ := speed_train1_kmph * 1000 / 3600
def speed_train2_mps : ℝ := speed_train2_kmph * 1000 / 3600

/-- Calculate the relative speed when the trains are moving towards each other. -/
def relative_speed_mps : ℝ := speed_train1_mps + speed_train2_mps

/-- Calculate the total distance the trains need to cover to meet. -/
def total_distance : ℝ := length_train1 + length_train2 + initial_distance

/-- The time it will take for the trains to meet. -/
theorem trains_meet_time :
  total_distance / relative_speed_mps ≈ 10.57 := sorry

end trains_meet_time_l673_673360


namespace probability_at_least_one_hit_l673_673372

variable (P₁ P₂ : ℝ)

theorem probability_at_least_one_hit (h₁ : 0 ≤ P₁ ∧ P₁ ≤ 1) (h₂ : 0 ≤ P₂ ∧ P₂ ≤ 1) :
  1 - (1 - P₁) * (1 - P₂) = P₁ + P₂ - P₁ * P₂ :=
by
  sorry

end probability_at_least_one_hit_l673_673372


namespace find_y_satisfies_equation_l673_673455

theorem find_y_satisfies_equation :
  ∃ y : ℝ, 3 * y + 6 = |(-20 + 2)| :=
by
  sorry

end find_y_satisfies_equation_l673_673455


namespace right_triangle_shorter_leg_l673_673562

theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : c = 65) : a = 25 ∨ b = 25 := 
by
  sorry

end right_triangle_shorter_leg_l673_673562


namespace numbers_divisible_by_2_and_3_but_not_5_l673_673840

theorem numbers_divisible_by_2_and_3_but_not_5 : 
  let nums := Finset.range 2000 |> Finset.filter (λ n, n % 6 = 0 ∧ n % 5 ≠ 0) in
  nums.card = 267 :=
by
  sorry

end numbers_divisible_by_2_and_3_but_not_5_l673_673840


namespace percentage_problem_l673_673793

theorem percentage_problem
  (a b c : ℚ) :
  (8 = (2 / 100) * a) →
  (2 = (8 / 100) * b) →
  (c = b / a) →
  c = 1 / 16 :=
by
  sorry

end percentage_problem_l673_673793


namespace weight_of_new_person_l673_673695

theorem weight_of_new_person (avg_increase : ℝ) (initial_person_weight : ℝ) (group_size : ℕ) (W : ℝ) : 
  avg_increase = 2.5 → 
  initial_person_weight = 66 → 
  group_size = 8 → 
  W = initial_person_weight + group_size * avg_increase → 
  W = 86 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end weight_of_new_person_l673_673695


namespace xy_sum_value_l673_673023

theorem xy_sum_value (x y : ℝ) (h1 : x + Real.cos y = 1010) (h2 : x + 1010 * Real.sin y = 1009) (h3 : (Real.pi / 4) ≤ y ∧ y ≤ (Real.pi / 2)) :
  x + y = 1010 + (Real.pi / 2) := 
by
  sorry

end xy_sum_value_l673_673023


namespace shorter_leg_of_right_triangle_l673_673572

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h: a^2 + b^2 = c^2) (hn: c = 65): a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l673_673572


namespace percentage_respondents_liked_B_l673_673815

variables (X Y : ℝ)
variables (likedA likedB likedBoth likedNeither : ℝ)
variables (totalRespondents : ℕ)

-- Conditions from the problem
def liked_conditions : Prop :=
    totalRespondents ≥ 100 ∧ 
    likedA = X ∧ 
    likedB = Y ∧ 
    likedBoth = 23 ∧ 
    likedNeither = 23

-- Proof statement
theorem percentage_respondents_liked_B (h : liked_conditions X Y likedA likedB likedBoth likedNeither totalRespondents) :
  Y = 100 - X :=
sorry

end percentage_respondents_liked_B_l673_673815


namespace three_digit_number_div_by_11_l673_673766

theorem three_digit_number_div_by_11 (x : ℕ) (h : x < 10) : 
  ∃ n : ℕ, n = 605 ∧ n < 1000 ∧ 
  (n % 10 = 5 ∧ (n / 100) % 10 = 6 ∧ n % 11 = 0) :=
begin
  use 605,
  split,
  { refl, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  norm_num,
end

end three_digit_number_div_by_11_l673_673766


namespace kim_shirts_left_l673_673126

-- Define the total number of shirts initially
def initial_shirts : ℕ := 4 * 12

-- Define the number of shirts given to the sister as 1/3 of the total
def shirts_given_to_sister : ℕ := initial_shirts / 3

-- Define the number of shirts left after giving some to the sister
def shirts_left : ℕ := initial_shirts - shirts_given_to_sister

-- The theorem we need to prove: Kim has 32 shirts left
theorem kim_shirts_left : shirts_left = 32 := by
  -- Proof is omitted
  sorry

end kim_shirts_left_l673_673126


namespace tan_diff_l673_673531

theorem tan_diff (α β : ℝ) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4/3) : Real.tan (α - β) = 1/3 := 
sorry

end tan_diff_l673_673531


namespace DF_length_l673_673092

-- Definitions for the given problem.
variable (AB DC EB DE : ℝ)
variable (parallelogram_ABCD : Prop)
variable (DE_altitude_AB : Prop)
variable (DF_altitude_BC : Prop)

-- Conditions
axiom AB_eq_DC : AB = DC
axiom EB_eq_5 : EB = 5
axiom DE_eq_8 : DE = 8

-- The main theorem to prove
theorem DF_length (hAB : AB = 15) (hDC : DC = 15) (hEB : EB = 5) (hDE : DE = 8)
  (hPar : parallelogram_ABCD)
  (hAltAB : DE_altitude_AB)
  (hAltBC : DF_altitude_BC) :
  ∃ DF : ℝ, DF = 8 := 
sorry

end DF_length_l673_673092


namespace largest_divisor_of_5_consecutive_integers_l673_673307

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l673_673307


namespace smallest_composite_no_prime_under_15_correct_l673_673937

-- Define the concept of a composite number
def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

-- Define the concept of having no prime factors less than 15
def has_no_prime_factors_less_than_15 (n : ℕ) : Prop := 
  ∀ p : ℕ, p.prime ∧ p ∣ n → 15 ≤ p

-- Define the smallest composite number with no prime factors less than 15 
def smallest_composite_no_prime_under_15 : ℕ := 289

-- Prove that this is the smallest number satisfying our conditions
theorem smallest_composite_no_prime_under_15_correct : 
  is_composite smallest_composite_no_prime_under_15 ∧
  has_no_prime_factors_less_than_15 smallest_composite_no_prime_under_15 ∧
  ∀ n : ℕ, is_composite n ∧ has_no_prime_factors_less_than_15 n → n ≥ smallest_composite_no_prime_under_15 :=
by 
  sorry

end smallest_composite_no_prime_under_15_correct_l673_673937


namespace fieldArea_correct_m_l673_673819

-- Define the field's dimensions and area condition.
def fieldDimensions (m : ℝ) : ℝ × ℝ :=
  (3 * m + 8, m - 3)

def fieldArea (m : ℝ) : Prop :=
  let (length, width) := fieldDimensions m in length * width = 100

-- State the theorem with the specified value of m.
theorem fieldArea_correct_m :
  fieldArea 6.597 := by
  sorry

end fieldArea_correct_m_l673_673819


namespace product_of_five_consecutive_integers_divisible_by_240_l673_673286

theorem product_of_five_consecutive_integers_divisible_by_240 (n : ℤ) : ∃ k : ℤ, (∏ i in range 5, n + i) = 240 * k := 
sorry

end product_of_five_consecutive_integers_divisible_by_240_l673_673286


namespace smallest_composite_no_prime_factors_less_than_15_l673_673924

theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n, (n = 289) ∧ (n > 1) ∧ (¬ Nat.Prime n) ∧ (∀ p : ℕ, Nat.Prime p → p ∣ n → 15 ≤ p) :=
by
  use 289
  split
  case left => rfl
  case right =>
    split
    case left => exact Nat.lt_succ_self 288
    case right =>
      split
      case left =>
        have composite : ¬ Nat.Prime 289 := by
          intro h
          have h_div : 17 ∣ 289 := by norm_num
          exact h.not_divs_self (dec_trivial : 17 * 17 = 289)
        exact composite
      case right =>
        intros p h_prime h_div
        have : p ∣ 17 := by
          have factorization : 289 = 17 * 17 := by norm_num
          have dvd_product : p ∣ 289 := by { use 17, exact factorization.symm }
          exact Nat.Prime.dvd_mul h_prime dvd_product
        have prime_eq_17 : p = 17 := by
          exact Nat.Prime.eq_of_dvd_of_ne h_prime (by norm_num) this
        linarith

end smallest_composite_no_prime_factors_less_than_15_l673_673924


namespace right_triangle_shorter_leg_l673_673568

theorem right_triangle_shorter_leg :
  ∃ (a b : ℤ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 16 :=
by
  sorry

end right_triangle_shorter_leg_l673_673568


namespace probability_correct_last_digit_probability_correct_last_digit_even_condition_l673_673710

def password_digits := {d : ℕ // 0 ≤ d ∧ d < 10} -- definition for the digit range (0 to 9)

-- Claim 1: The probability of pressing the correct last digit in no more than 2 attempts is 1/5
theorem probability_correct_last_digit (d : password_digits) :
  let total_attempts := 10
  in (1 / total_attempts) + ((total_attempts - 1) / total_attempts) * (1 / (total_attempts - 1)) = 1 / 5 := sorry

-- Claim 2: If the last digit is even, the probability of pressing the correct digit in no more than 2 attempts is 2/5
theorem probability_correct_last_digit_even_condition (d : password_digits) (h : d.1 % 2 = 0) :
  let total_attempts := 5
  in (1 / total_attempts) + ((total_attempts - 1) / total_attempts) * (1 / (total_attempts - 1)) = 2 / 5 := sorry

end probability_correct_last_digit_probability_correct_last_digit_even_condition_l673_673710


namespace exists_four_numbers_product_fourth_power_l673_673654

theorem exists_four_numbers_product_fourth_power :
  ∃ (numbers : Fin 81 → ℕ),
    (∀ i, ∃ a b c : ℕ, numbers i = 2^a * 3^b * 5^c) ∧
    ∃ (i j k l : Fin 81), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧
    ∃ m : ℕ, m^4 = numbers i * numbers j * numbers k * numbers l :=
by
  sorry

end exists_four_numbers_product_fourth_power_l673_673654


namespace arithmetic_neg3_plus_4_l673_673847

theorem arithmetic_neg3_plus_4 : -3 + 4 = 1 :=
by
  sorry

end arithmetic_neg3_plus_4_l673_673847


namespace cos_alpha_minus_2pi_l673_673466

open Real

noncomputable def problem_statement (alpha : ℝ) : Prop :=
  (sin (π + alpha) = 4 / 5) ∧ (cos (alpha - 2 * π) = 3 / 5)

theorem cos_alpha_minus_2pi (alpha : ℝ) (h1 : sin (π + alpha) = 4 / 5) (quad4 : cos alpha > 0 ∧ sin alpha < 0) :
  cos (alpha - 2 * π) = 3 / 5 :=
sorry

end cos_alpha_minus_2pi_l673_673466


namespace difference_between_pots_is_correct_l673_673640

-- Conditions for the problem
variable (d : ℝ)
axiom total_cost_of_pots : (1.625 + (1.625 - d) + (1.625 - 2*d) + (1.625 - 3*d) + (1.625 - 4*d) + (1.625 - 5*d)) = 8.25

-- Prove that the difference in cost is $0.10
theorem difference_between_pots_is_correct : d = 0.1 :=
by apply total_cost_of_pots; sorry

end difference_between_pots_is_correct_l673_673640


namespace tenth_equation_compare_roots_differences_sum_series_l673_673645

-- Define the conditions
def pattern_equation (n : ℕ) : Prop :=
  (∃ b c : ℝ, b = sqrt (n + 1) ∧ c = sqrt n ∧ (b + c) * (b - c) = 1)

-- Statement for the 10th equation in the pattern
theorem tenth_equation : pattern_equation 10 :=
by
  sorry

-- Statement for comparing sizes of roots differences
theorem compare_roots_differences : sqrt 18 - sqrt 17 > sqrt 19 - sqrt 18 :=
by
  sorry

-- Statement for the sum of the series
theorem sum_series : (∑ n in finset.range 98, (3 : ℝ) / (sqrt (n + 2) + sqrt (n + 1))) = -3 + 9 * sqrt 11 :=
by
  sorry

end tenth_equation_compare_roots_differences_sum_series_l673_673645


namespace at_least_30_cents_probability_l673_673678

theorem at_least_30_cents_probability :
  let penny := 1
  let nickel := 5
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  let all_possible_outcomes := 2^5
  let successful_outcomes := 
    -- Half-dollar and quarter heads: 2^3 = 8 combinations
    2^3 + 
    -- Quarter heads and half-dollar tails (nickel and dime heads): 2 combinations
    2^1 + 
    -- Quarter tails and half-dollar heads: 2^3 = 8 combinations
    2^3
  let probability := successful_outcomes / all_possible_outcomes
  probability = 9 / 16 :=
by
  -- Proof goes here
  sorry

end at_least_30_cents_probability_l673_673678


namespace sequence_limit_zero_l673_673164

-- Define the sequence recursively
def sequence (n : ℕ) : ℝ :=
  if n = 0 then 1
  else if n = 1 then 1
  else sequence (n - 1) ^ 2 - (1/2) * sequence (n - 2)

-- Statement to prove that the sequence converges to 0
theorem sequence_limit_zero : 
  ∃ L : ℝ, 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sequence n - L| < ε) ∧ 
  L = 0 :=
by
  sorry

end sequence_limit_zero_l673_673164


namespace elena_bread_recipe_l673_673876

theorem elena_bread_recipe :
  ∀ (recipe_butter : ℕ) (multiplier : ℕ) (total_butter : ℕ) (total_flour : ℕ),
  recipe_butter = 3 →
  multiplier = 6 →
  total_butter = 12 →
  total_flour = 24 →
  let ratio_flour_per_butter := total_flour / total_butter in
  let original_flour := recipe_butter * ratio_flour_per_butter in
  original_flour = 6 :=
begin
  intros recipe_butter multiplier total_butter total_flour,
  intros h_recipe_butter h_multiplier h_total_butter h_total_flour,
  let ratio_flour_per_butter := total_flour / total_butter,
  let original_flour := recipe_butter * ratio_flour_per_butter,
  rw [h_recipe_butter, h_multiplier, h_total_butter, h_total_flour],
  simp [ratio_flour_per_butter, original_flour],
  -- Here would be the place to provide the proof steps
  sorry
end

end elena_bread_recipe_l673_673876


namespace problem_1_problem_2_l673_673421

-- Problem 1: Prove that (\frac{1}{5} - \frac{2}{3} - \frac{3}{10}) × (-60) = 46
theorem problem_1 : (1/5 - 2/3 - 3/10) * -60 = 46 := by
  sorry

-- Problem 2: Prove that (-1)^{2024} + 24 ÷ (-2)^3 - 15^2 × (1/15)^2 = -3
theorem problem_2 : (-1)^2024 + 24 / (-2)^3 - 15^2 * (1/15)^2 = -3 := by
  sorry

end problem_1_problem_2_l673_673421


namespace average_pages_per_day_l673_673167

theorem average_pages_per_day (total_pages : ℕ) (days_in_week : ℕ) (total_pages = 161) (days_in_week = 7) : 
  total_pages / days_in_week = 23 := 
  sorry

end average_pages_per_day_l673_673167


namespace QR_squared_in_trapezoid_l673_673595

theorem QR_squared_in_trapezoid (PQ RS QR PS : ℝ) 
  (h1 : PQ = Real.sqrt 23)
  (h2 : PS = Real.sqrt 2023)
  (h3 : QR^2 = x → RS = y → PR_perp_QS (triangle PQR) (triangle QSR) = true)
  (h4 : leg_perp_bases QR PQ RS (trapezoid PQRS) = true)
  : QR^2 = 100 * Real.sqrt 46 := sorry

end QR_squared_in_trapezoid_l673_673595


namespace unique_maximizing_line_l673_673085

-- Define the points A, B, and C in a Euclidean plane
variables {A B C : EuclideanGeometry.Point}

-- Define the property of a line maximizing the product of distances to A and B
def maximizing_line (A B C : EuclideanGeometry.Point) (L : EuclideanGeometry.Line) : Prop :=
  ∀ L', EuclideanGeometry.distance (L'.project A) (L.project A) * EuclideanGeometry.distance (L'.project B) (L.project B) ≤
       EuclideanGeometry.distance (L.project A) (L.project A) * EuclideanGeometry.distance (L.project B) (L.project B)

-- State the theorem regarding the uniqueness of such a line
theorem unique_maximizing_line (A B C : EuclideanGeometry.Point) :
  ∃! L : EuclideanGeometry.Line, maximizing_line A B C L := sorry

end unique_maximizing_line_l673_673085


namespace ice_volume_after_two_hours_l673_673842

def original_volume : ℝ := 4
def volume_after_first_hour := (1 / 4) * original_volume
def volume_after_second_hour := (1 / 4) * volume_after_first_hour

theorem ice_volume_after_two_hours : volume_after_second_hour = 1 / 4 := 
sorry

end ice_volume_after_two_hours_l673_673842


namespace length_width_difference_l673_673193

theorem length_width_difference (L W : ℝ) 
  (h1 : W = 1/2 * L) 
  (h2 : L * W = 578) : L - W = 17 :=
sorry

end length_width_difference_l673_673193


namespace bridge_length_l673_673834

theorem bridge_length :
  ∀ (L_train : ℝ) (v_train_kmph : ℝ) (t_cross : ℝ), 
  L_train = 165 →
  v_train_kmph = 54 →
  t_cross = 52.66245367037304 →
  let v_train_mps := (v_train_kmph * 1000) / 3600 in
  let total_distance := v_train_mps * t_cross in
  let L_bridge := total_distance - L_train in
  L_bridge = 624.9368050555956 := by 
sorry

end bridge_length_l673_673834


namespace problem_l673_673517

open Real -- We might need the facts on Real numbers and logarithms

noncomputable def A : set ℝ := {x | x^2 - 2 * x < 0}
noncomputable def B : set ℝ := {x | log 10 (x - 1) ≤ 0}
noncomputable def intersection_AB : set ℝ := A ∩ B

theorem problem (A B : set ℝ) (hA : A = {x | x^2 - 2 * x < 0}) (hB : B = {x | log 10 (x - 1) ≤ 0}) : 
  intersection_AB = {x | 1 < x ∧ x < 2} := by
  sorry

end problem_l673_673517


namespace smallest_possible_n_l673_673704

theorem smallest_possible_n (x n : ℤ) (hx : 0 < x) (m : ℤ) (hm : m = 30) (h1 : m.gcd n = x + 1) (h2 : m.lcm n = x * (x + 1)) : n = 6 := sorry

end smallest_possible_n_l673_673704


namespace cost_of_article_l673_673068

variable {C G : ℝ}

theorem cost_of_article (h : 350 = C * (1 + (G + 5) / 100)) (h' : 340 = C * (1 + G / 100)) : C = 200 := by
  sorry

end cost_of_article_l673_673068


namespace product_of_five_consecutive_divisible_by_30_l673_673271

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l673_673271


namespace product_of_five_consecutive_divisible_by_30_l673_673269

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l673_673269


namespace probability_two_white_balls_l673_673798

theorem probability_two_white_balls (white black : ℕ) (total_drawn : ℕ) 
  (h_white : white = 7) (h_black : black = 9) (h_total_drawn : total_drawn = 2):
  ((nat.choose white total_drawn) / (nat.choose (white + black) total_drawn) : ℚ) = 7 / 40 :=
by
  sorry

end probability_two_white_balls_l673_673798


namespace cube_volume_from_diagonal_l673_673701

theorem cube_volume_from_diagonal (d : ℝ) (V : ℝ) : d = 6 * real.sqrt 3 → V = 216 :=
by
  intro h
  -- proof omitted
  sorry

end cube_volume_from_diagonal_l673_673701


namespace measure_of_angleA_l673_673996

theorem measure_of_angleA (A B : ℝ) 
  (h1 : ∀ (x : ℝ), x ≠ A → x ≠ B → x ≠ (3 * B - 20) → (3 * x - 20 ≠ A)) 
  (h2 : A = 3 * B - 20) :
  A = 10 ∨ A = 130 :=
by
  sorry

end measure_of_angleA_l673_673996


namespace find_2n_plus_m_l673_673184

theorem find_2n_plus_m (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 
  2 * n + m = 36 := 
sorry

end find_2n_plus_m_l673_673184


namespace team_A_minimum_workers_l673_673491

-- Define the variables and conditions for the problem.
variables (A B c : ℕ)

-- Condition 1: If team A lends 90 workers to team B, Team B will have twice as many workers as Team A.
def condition1 : Prop :=
  2 * (A - 90) = B + 90

-- Condition 2: If team B lends c workers to team A, Team A will have six times as many workers as Team B.
def condition2 : Prop :=
  A + c = 6 * (B - c)

-- Define the proof goal.
theorem team_A_minimum_workers (h1 : condition1 A B) (h2 : condition2 A B c) : 
  153 ≤ A :=
sorry

end team_A_minimum_workers_l673_673491


namespace four_digit_numbers_count_l673_673054

theorem four_digit_numbers_count :
  (∑ h in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, (if h = 0 then 10
                                            else if h = 1 then 8
                                            else if h = 2 then 6
                                            else if h = 3 then 4
                                            else if h = 4 then 2
                                            else 10)) *
  9 * 10 = 7200 := 
by
  sorry

end four_digit_numbers_count_l673_673054


namespace largest_divisor_of_5_consecutive_integers_l673_673331

theorem largest_divisor_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, (product_of_5_consecutive_integers n = k) ∧ (60 ∣ k) 
:= sorry

end largest_divisor_of_5_consecutive_integers_l673_673331


namespace color_complete_graph_l673_673866

open SimpleGraph

-- Definitions used in conditions
def K9 : SimpleGraph (Fin 9) := completeGraph (Fin 9)

def edgeColoring (c : Symmetric (Fin 9) × (Fin 9) → Fin 2) : Prop := 
  ∀ e : Symmetric (Fin 9) × (Fin 9), e ∈ K9.edgeSet → c e ∈ ({0, 1} : Finset (Fin 2))

-- Main theorem statement
theorem color_complete_graph (c : Symmetric (Fin 9) × (Fin 9) → Fin 2) (hc : edgeColoring c) :
  ∃ (S : Finset (Fin 9)), (S.card = 4 ∧ S.pairwise (λ u v, c ⟨u, v⟩ = 0)) ∨ (S.card = 3 ∧ S.pairwise (λ u v, c ⟨u, v⟩ = 1)) := 
sorry

end color_complete_graph_l673_673866


namespace exists_unitary_vector_l673_673672

theorem exists_unitary_vector {d : ℕ} (v : fin d → ℝ^d) (hv : ∀ i, ‖v i‖ = 1) :
  ∃ u : ℝ^d, ‖u‖ = 1 ∧ ∀ i : fin d, |u ⬝ (v i)| ≤ 1 / real.sqrt d :=
sorry

end exists_unitary_vector_l673_673672


namespace salt_water_mixture_concentration_l673_673191

variable (mass1 : ℕ) (conc1 : ℚ) (mass2 : ℕ) (salt2 : ℚ)
variable (total_mass : ℕ) (total_salt : ℚ)

-- Define the given conditions
def mass1 := 200
def conc1 := 25 / 100 -- Represent 25% as a rational number
def mass2 := 300
def salt2 := 60

-- Calculate the amount of salt in the first cup
def salt1 := conc1 * mass1

-- Calculate the total weight and the total amount of salt
def total_mass := mass1 + mass2
def total_salt := salt1 + salt2

-- Calculate the concentration of salt in the mixture
def mixture_concentration := (total_salt / total_mass) * 100

-- The proof statement
theorem salt_water_mixture_concentration : mixture_concentration = 22 := 
by
  sorry

end salt_water_mixture_concentration_l673_673191


namespace find_abc_value_l673_673161

noncomputable def given_conditions (a b c : ℝ) : Prop :=
  (a * b / (a + b) = 2) ∧ (b * c / (b + c) = 5) ∧ (c * a / (c + a) = 9)

theorem find_abc_value (a b c : ℝ) (h : given_conditions a b c) :
  a * b * c / (a * b + b * c + c * a) = 90 / 73 :=
sorry

end find_abc_value_l673_673161


namespace complex_modulus_pow_l673_673845

theorem complex_modulus_pow (a : ℂ) (h : a = 2 + complex.I) : 
  complex.abs (a^6) = 125 := by
  sorry

end complex_modulus_pow_l673_673845


namespace not_perfect_square_l673_673341

theorem not_perfect_square : ¬ ∃ x : ℝ, x^2 = 7^2025 := by
  sorry

end not_perfect_square_l673_673341


namespace rhombus_area_l673_673780

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 15) (h2 : d2 = 20) :
  (d1 * d2) / 2 = 150 :=
by
  rw [h1, h2]
  norm_num

end rhombus_area_l673_673780


namespace real_z_imaginary_z_pure_imaginary_z_l673_673967

def is_real (z : ℂ) : Prop := z.im = 0
def is_imaginary (z : ℂ) : Prop := z.im ≠ 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

def z (a : ℝ) : ℂ := (a^2 - 7 * a + 12 : ℝ) + (a^2 - 5 * a + 6 : ℂ) * Complex.i

theorem real_z (a : ℝ) : is_real (z a) ↔ a = 2 ∨ a = 3 := by
  sorry

theorem imaginary_z (a : ℝ) : is_imaginary (z a) ↔ a ≠ 2 ∧ a ≠ 3 := by
  sorry

theorem pure_imaginary_z (a : ℝ) : is_pure_imaginary (z a) ↔ a = 4 := by
  sorry

end real_z_imaginary_z_pure_imaginary_z_l673_673967


namespace bob_probability_after_three_turns_l673_673406

/-- Conditions for the game:
1. If Alice has the ball, she tosses it to Bob with probability 2/3 and keeps it with probability 1/3.
2. If Bob has the ball, he tosses it to Alice with probability 1/4 and keeps it with probability 3/4.
3. Bob starts with the ball.
We need to prove that the probability that Bob has the ball again after three turns is 11/16. -/
theorem bob_probability_after_three_turns :
  let P_Bob_keep_1_turn : ℚ := 3 / 4,
      P_Alice_to_Bob : ℚ := 2 / 3,
      P_Bob_to_Alice : ℚ := 1 / 4 in
  let P_Bob_Bob : ℚ := P_Bob_keep_1_turn * P_Bob_keep_1_turn * P_Bob_keep_1_turn,
      P_Bob_Alice_Bob : ℚ := P_Bob_to_Alice * P_Alice_to_Bob * P_Bob_keep_1_turn in
  P_Bob_Bob + P_Bob_Alice_Bob = 11 / 16 := sorry

end bob_probability_after_three_turns_l673_673406


namespace gcd_g_y_l673_673024

def g (y : ℕ) : ℕ := (3*y + 4) * (8*y + 3) * (14*y + 9) * (y + 17)

theorem gcd_g_y (y : ℕ) (h : y % 42522 = 0) : Nat.gcd (g y) y = 102 := by
  sorry

end gcd_g_y_l673_673024


namespace increasing_function_l673_673022

noncomputable def f (a x : ℝ) : ℝ := a * x + Real.sin x

theorem increasing_function (a : ℝ) :
  (∀ x y, x ≤ y → f a x ≤ f a y) ↔ (a ∈ Set.Ici (1 : ℝ)) := by
  sorry

end increasing_function_l673_673022


namespace chessboard_dominos_l673_673382

theorem chessboard_dominos (n : ℕ) :
  (∃ (f : ((2 * n + 1) * (2 * n + 1) - 1) → bool),
    (∀ i, f i → (∃ m, 2 * m = i ∨ 2 * m + 1 = i))
    ∧ ∃ g, (∀ i, g i → (∃ m, 2 * m = i ∨ 2 * m + 1 = i)) )
    ↔ (n % 2 = 0) :=
begin
  sorry
end

end chessboard_dominos_l673_673382


namespace smallest_composite_proof_l673_673899

-- Define what it means for a number not to have prime factors less than 15
def no_prime_factors_less_than_15 (n : ℕ) : Prop :=
  ∀ p : ℕ, nat.prime p → p ∣ n → p ≥ 15

-- Define what it means for a number to be the smallest composite number with the above property
def smallest_composite_without_prime_factors_less_than_15 (n : ℕ) : Prop :=
  nat.composite n ∧ no_prime_factors_less_than_15 n ∧
  ∀ m : ℕ, nat.composite m → no_prime_factors_less_than_15 m → n ≤ m

theorem smallest_composite_proof : smallest_composite_without_prime_factors_less_than_15 323 :=
  sorry

end smallest_composite_proof_l673_673899


namespace hyperbola_range_l673_673509

theorem hyperbola_range (m : ℝ) :
  (m + 2) * (2 * m - 1) > 0 ↔ m ∈ Set.Ioo (-∞) (-2) ∪ Set.Ioo (1/2) ∞ :=
by
  sorry

end hyperbola_range_l673_673509


namespace tangent_slope_angle_is_60_degrees_l673_673721

-- Define the curve
def curve (x : ℝ) : ℝ := x^3

-- Define the conditions of the problem
structure ProblemCondition where
  B : ℝ × ℝ
  hB : B.snd = curve B.fst
  tangentLineIntersectXaxisAtA : ℝ × ℝ
  hIntersect : tangentLineIntersectXaxisAtA.snd = 0
  hIsosceles : IsIsosceles (0, 0) B tangentLineIntersectXaxisAtA

-- Define the problem statement
theorem tangent_slope_angle_is_60_degrees (conds : ProblemCondition) : 
  angle (slope (tangentLine conds.B)) = 60 :=
sorry

end tangent_slope_angle_is_60_degrees_l673_673721


namespace same_side_line_a_range_l673_673990

theorem same_side_line_a_range:
  (∀ a : ℝ, (3 * 3 - 2 * (-1) + a) * (3 * (-4) + 2 * 3 + a) > 0)
  → a ∈ set.Ioo (-∞ : ℝ) (-11) ∪ set.Ioo (6 : ℝ) (∞ : ℝ) :=
sorry

end same_side_line_a_range_l673_673990


namespace liked_product_B_l673_673810

-- Define the conditions as assumptions
variables (X : ℝ)

-- Assumptions
axiom liked_both : 23 = 23
axiom liked_neither : 23 = 23

-- The main theorem that needs to be proven
theorem liked_product_B (X : ℝ) : ∃ Y : ℝ, Y = 100 - X :=
by sorry

end liked_product_B_l673_673810


namespace maximum_value_abs_difference_l673_673787

theorem maximum_value_abs_difference (x y : ℝ) 
  (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) : 
  |x - y + 1| ≤ 2 :=
sorry

end maximum_value_abs_difference_l673_673787


namespace shorter_leg_of_right_triangle_l673_673558

theorem shorter_leg_of_right_triangle (a b : ℕ) (h : a^2 + b^2 = 65^2) : min a b = 16 :=
sorry

end shorter_leg_of_right_triangle_l673_673558


namespace candies_left_l673_673844

-- Defining the given conditions
def initial_candies : Nat := 30
def eaten_candies : Nat := 23

-- Define the target statement to prove
theorem candies_left : initial_candies - eaten_candies = 7 := by
  sorry

end candies_left_l673_673844


namespace ellipse_equation_y_intercept_range_l673_673976

-- Conditions
variables {a b : ℝ} (h1 : a > b) (h2 : b > 0)
def ellipse := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1
def eccentricity := ∀ c : ℝ, c = a * sqrt(1 - (b^2 / a^2))
def intercepted_length := 2 * (b^2 / a) = sqrt(2)

-- Proofs to be established
theorem ellipse_equation (h1 : a^2 = 8) (h2 : b^2 = 2) : 
  ∀ x y : ℝ, x^2 / 8 + y^2 / 2 = 1 :=
sorry

theorem y_intercept_range (m : ℝ)
  (h3 : -sqrt(2) < m ∧ m < sqrt(2) ∧ m ≠ 0) : 
  -sqrt(2) < m ∧ m < sqrt(2) ∧ m ≠ 0 :=
sorry

end ellipse_equation_y_intercept_range_l673_673976


namespace probability_of_at_least_30_cents_l673_673681

def coin := fin 5

def value (c : coin) : ℤ :=
match c with
| 0 => 1   -- penny
| 1 => 5   -- nickel
| 2 => 10  -- dime
| 3 => 25  -- quarter
| 4 => 50  -- half-dollar
| _ => 0

def coin_flip : coin -> bool := λ c => true -- Placeholder for whether heads or tails

def total_value (flips : coin -> bool) : ℤ :=
  finset.univ.sum (λ c, if flips c then value c else 0)

noncomputable def probability_at_least_30_cents : ℚ :=
  let coin_flips := (finset.pi finset.univ (λ _, finset.univ : finset (coin -> bool))).val in
  let successful_flips := coin_flips.filter (λ flips, total_value flips >= 30) in
  successful_flips.card / coin_flips.card

theorem probability_of_at_least_30_cents :
  probability_at_least_30_cents = 9 / 16 :=
by
  sorry

end probability_of_at_least_30_cents_l673_673681


namespace exists_hexagon_divided_into_four_equal_triangles_l673_673436

theorem exists_hexagon_divided_into_four_equal_triangles :
  ∃ H : Hexagon, ∃ L : Line, divides_into_four_equal_triangles H L :=
sorry

end exists_hexagon_divided_into_four_equal_triangles_l673_673436


namespace distance_to_other_focus_l673_673978

noncomputable def ellipse := {x y : ℝ // x^2 / 25 + y^2 / 16 = 1}

theorem distance_to_other_focus (P : ellipse) (d1 : ℝ) (h1 : d1 = 3) : 
  ∃ d2 : ℝ, d2 = 7 :=
by
  sorry

end distance_to_other_focus_l673_673978


namespace james_take_home_pay_l673_673603

theorem james_take_home_pay :
  let main_hourly_rate := 20
  let second_hourly_rate := main_hourly_rate - (main_hourly_rate * 0.20)
  let main_hours := 30
  let second_hours := main_hours / 2
  let side_gig_earnings := 100 * 2
  let overtime_hours := 5
  let overtime_rate := main_hourly_rate * 1.5
  let irs_tax_rate := 0.18
  let state_tax_rate := 0.05
  
  -- Main job earnings
  let main_regular_earnings := main_hours * main_hourly_rate
  let main_overtime_earnings := overtime_hours * overtime_rate
  let main_total_earnings := main_regular_earnings + main_overtime_earnings
  
  -- Second job earnings
  let second_total_earnings := second_hours * second_hourly_rate
  
  -- Total earnings before taxes
  let total_earnings := main_total_earnings + second_total_earnings + side_gig_earnings
  
  -- Tax calculations
  let federal_tax := total_earnings * irs_tax_rate
  let state_tax := total_earnings * state_tax_rate
  let total_taxes := federal_tax + state_tax

  -- Total take home pay after taxes
  let take_home_pay := total_earnings - total_taxes

  take_home_pay = 916.30 := 
sorry

end james_take_home_pay_l673_673603


namespace positive_difference_of_numbers_l673_673694

theorem positive_difference_of_numbers (x : ℝ) (h : (30 + x) / 2 = 34) : abs (x - 30) = 8 :=
by
  sorry

end positive_difference_of_numbers_l673_673694


namespace largest_divisor_of_five_consecutive_integers_l673_673265

theorem largest_divisor_of_five_consecutive_integers:
  ∀ (n : ℤ), 
    ∃ (a b c : ℤ), 
      (n = 5 * a ∨ n = 5 * a + 1 ∨ n = 5 * a + 2 ∨ n = 5 * a + 3 ∨ n = 5 * a + 4) ∧ 
      (n = 3 * b ∨ n = 3 * b + 1 ∨ n = 3 * b + 2) ∧ 
      (n = 4 * c ∨ n = 4 * c + 1 ∨ n = 4 * c + 2 ∨ n = 4 * c + 3)
      ⊢ 60 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  -- Sorry placeholder for the proof (not needed in the task)
  sorry

end largest_divisor_of_five_consecutive_integers_l673_673265


namespace area_of_ABCD_l673_673101

noncomputable def area_of_quadrilateral (ABCD : Type) [NonConvexQuadrilateral ABCD]
  (AB CD AD : ℝ) (angle_BCD : ℝ) (area_BDA : ℝ) : ℝ :=
have h_AB : AB = 15 := sorry,
have h_BC : BC = 5 := sorry,
have h_CD : CD = 4 := sorry,
have h_AD : AD = 17 := sorry,
have h_angle_BCD : angle_BCD = 90 := sorry,
h_area_BDA : area_BDA := sorry,
10 + area_BDA

-- Define the proof
theorem area_of_ABCD (ABCD : Type) [NonConvexQuadrilateral ABCD]
  (AB CD AD : ℝ) (angle_BCD : ℝ) (area_BDA : ℝ) (h_AB : AB = 15)
  (h_BC : BC = 5) (h_CD : CD = 4) (h_AD : AD = 17) (h_angle_BCD : angle_BCD = 90)
  (h_area_BDA : area_BDA)
  : area_of_quadrilateral ABCD AB CD AD angle_BCD area_BDA = 10 + h_area_BDA :=
sorry

end area_of_ABCD_l673_673101


namespace alternating_sum_sequence_l673_673336

theorem alternating_sum_sequence :
  (1 - 4 + 7 - 10 + 13 - 16 + ... + 91 - 94 + 97) = 49 :=
by
  sorry

end alternating_sum_sequence_l673_673336


namespace garden_dimensions_l673_673389

theorem garden_dimensions (w l : ℕ) (h₁ : l = w + 3) (h₂ : 2 * (l + w) = 26) : w = 5 ∧ l = 8 :=
by
  sorry

end garden_dimensions_l673_673389


namespace rectangle_locus_rhombus_locus_l673_673004

open EuclideanGeometry

-- Define the points A, B, C, D, M, P, Q, K, L in the Euclidean space
variables {A B C D M P Q K L : Point ℝ}

-- Define the quadrilateral plane
def quadrilateral_plane : Plane ℝ := Plane.mk A B C D -- Assuming this creates the plane through points A, B, C, D

-- Define conditions for P, Q
def is_intersection_point_P (A B C D P : Point ℝ) : Prop :=
  ∃ l1 l2 : Line ℝ, intersects_in_extension l1 l2 A B C D P -- Intersection of extensions

def is_intersection_point_Q (A B C D Q : Point ℝ) : Prop :=
  ∃ l3 l4 : Line ℝ, intersects_in_extension l3 l4 A B C D Q -- Intersection of extensions

-- Define conditions for K, L
def is_intersection_point_K (A C B D P K : Point ℝ) : Prop :=
  ∃ d1 : Line ℝ, extension_of_diag d1 A C B D P K -- Intersection of diagonals 

def is_intersection_point_L (A C B D Q L : Point ℝ) : Prop :=
  ∃ d2 : Line ℝ, extension_of_diag d2 A C B D Q L -- Intersection of diagonals

-- Define the main properties
def on_sphere_with_diameter (M P Q : Point ℝ) : Prop :=
  distance M P * distance M Q = (distance P Q / 2)^2

def on_plane (M : Point ℝ) (quad_plane: Plane ℝ) : Prop :=
  quad_plane.contains M

-- Part a: Locus forming rectangle
theorem rectangle_locus (A B C D M P Q : Point ℝ) :
  is_intersection_point_P A B C D P →
  is_intersection_point_Q A B C D Q →
  on_sphere_with_diameter M P Q →
  ¬ on_plane M quadrilateral_plane →
  ¬ on_plane M quadrilateral_plane := sorry

-- Part b: Locus forming rhombus
theorem rhombus_locus (A B C D M P Q K L : Point ℝ) :
  is_intersection_point_P A B C D P →
  is_intersection_point_Q A B C D Q →
  is_intersection_point_K A C B D P K →
  is_intersection_point_L A C B D Q L →
  on_sphere_with_diameter M K L →
  ¬ on_plane M quadrilateral_plane :=
  sorry

end rectangle_locus_rhombus_locus_l673_673004


namespace shirts_left_l673_673120

-- Define the given conditions
def initial_shirts : ℕ := 4 * 12
def fraction_given : ℚ := 1 / 3

-- Define the proof goal
theorem shirts_left (initial_shirts : ℕ) (fraction_given : ℚ) : ℕ :=
let shirts_given := initial_shirts * fraction_given in
initial_shirts - (shirts_given : ℕ) = 32 :=
begin
  -- placeholder for the proof
  sorry
end

end shirts_left_l673_673120


namespace kim_shirts_left_l673_673125

-- Define the total number of shirts initially
def initial_shirts : ℕ := 4 * 12

-- Define the number of shirts given to the sister as 1/3 of the total
def shirts_given_to_sister : ℕ := initial_shirts / 3

-- Define the number of shirts left after giving some to the sister
def shirts_left : ℕ := initial_shirts - shirts_given_to_sister

-- The theorem we need to prove: Kim has 32 shirts left
theorem kim_shirts_left : shirts_left = 32 := by
  -- Proof is omitted
  sorry

end kim_shirts_left_l673_673125


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673219

theorem largest_divisor_of_product_of_five_consecutive_integers
  (a b c d e : ℤ) 
  (h: a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e):
  ∃ (n : ℤ), n = 60 ∧ n ∣ (a * b * c * d * e) :=
sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673219


namespace sum_of_digits_of_greatest_prime_divisor_l673_673334

theorem sum_of_digits_of_greatest_prime_divisor (n : ℕ) (h : n = 32767) : 
  (nat.digits 10 1057).sum = 13 := 
sorry

end sum_of_digits_of_greatest_prime_divisor_l673_673334


namespace edric_hours_per_day_l673_673440

/--
Edric's monthly salary is $576. He works 6 days a week for 4 weeks in a month and 
his hourly rate is $3. Prove that Edric works 8 hours in a day.
-/
theorem edric_hours_per_day (m : ℕ) (r : ℕ) (d : ℕ) (w : ℕ)
  (h_m : m = 576) (h_r : r = 3) (h_d : d = 6) (h_w : w = 4) :
  (m / r) / (d * w) = 8 := by
    sorry

end edric_hours_per_day_l673_673440


namespace find_k_for_perfect_square_l673_673880

theorem find_k_for_perfect_square :
  ∃ k : ℤ, (k = 12 ∨ k = -12) ∧ (∀ n : ℤ, ∃ a b : ℤ, 4 * n^2 + k * n + 9 = (a * n + b)^2) :=
sorry

end find_k_for_perfect_square_l673_673880


namespace probability_gcd_three_numbers_one_l673_673741

noncomputable def probability_gcd_one : ℚ :=
  let total_subsets : ℕ := choose 8 3 in
  let non_rel_prime_subsets : ℕ := 4 in
  let prob := (total_subsets - non_rel_prime_subsets : ℚ) / total_subsets in
  prob

theorem probability_gcd_three_numbers_one :
  probability_gcd_one = 13 / 14 :=
by
  sorry

end probability_gcd_three_numbers_one_l673_673741


namespace prob_of_selecting_exactly_one_good_product_l673_673409

theorem prob_of_selecting_exactly_one_good_product (total_products: ℕ) (good_products: ℕ) (defective_products: ℕ) (selected_products: ℕ) : 
  total_products = 5 → good_products = 3 → defective_products = 2 → selected_products = 2 → 
  ((good_products.choose 1 * defective_products.choose 1).toReal / total_products.choose selected_products) = (3 / 5) :=
by
  intros h1 h2 h3 h4
  sorry

end prob_of_selecting_exactly_one_good_product_l673_673409


namespace smallest_composite_no_prime_factors_less_than_15_l673_673919

-- Definitions used in the conditions
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

-- Prime numbers less than 15
def primes_less_than_15 (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13

-- Define the main proof statement
theorem smallest_composite_no_prime_factors_less_than_15 :
  ∃ n : ℕ, is_composite n ∧ (∀ p : ℕ, p ∣ n → is_prime p → primes_less_than_15 p → false) ∧ n = 289 :=
by
  -- leave the proof as a placeholder
  sorry

end smallest_composite_no_prime_factors_less_than_15_l673_673919


namespace part_I_part_II_l673_673634

-- Part I
theorem part_I (x : ℝ) (m : ℝ) (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {x | x^2 - 2 * x - 8 < 0})
  (hB : B = {x | x - m < 0})
  (hU : U = Set.univ)
  (hm : m = 3) :
  A ∩ (U \ B) = set.Ici 3 ∩ set.Iio 4 := 
sorry

-- Part II
theorem part_II (x : ℝ) (m : ℝ) (A : Set ℝ) (B : Set ℝ)
  (hA : A = {x | x^2 - 2 * x - 8 < 0})
  (hB : B = {x | x - m < 0})
  (hEmpty : A ∩ B = ∅) : 
  m ≤ -2 := 
sorry

end part_I_part_II_l673_673634


namespace probability_gcd_one_is_49_over_56_l673_673736

def is_gcd_one (a b c : ℕ) : Prop := Nat.gcd a (Nat.gcd b c) = 1

def count_choices_with_gcd_one : ℕ :=
  ((Finset.powersetLen 3 (Finset.range 9)).filter (λ s, match s.toList with
    | [a, b, c] => is_gcd_one a b c
    | _ => false
  end)).card

def total_choices : ℕ := (Finset.powersetLen 3 (Finset.range 9)).card

theorem probability_gcd_one_is_49_over_56 :
  (count_choices_with_gcd_one : ℚ) / total_choices = 49 / 56 := by
  sorry

end probability_gcd_one_is_49_over_56_l673_673736


namespace meal_serving_problem_l673_673748

/-
Twelve people sit down for dinner where there are three choices of meals: beef, chicken, and fish.
Four people order beef, four people order chicken, and four people order fish.
The waiter serves the twelve meals in random order.
We need to find the number of ways in which the waiter could serve the meals so that exactly two people receive the type of meal ordered by them.
-/
theorem meal_serving_problem :
    ∃ (n : ℕ), n = 12210 ∧
    (∃ (people : Fin 12 → char), 
        (∀ i : Fin 4, people i = 'B') ∧ 
        (∀ i : Fin 4, people (i + 4) = 'C') ∧ 
        (∀ i : Fin 4, people (i + 8) = 'F') ∧ 
        (∃ (served : Fin 12 → char), 
            (∃ (correct : Fin 12), set.range correct ⊆ {0, 1} ∧
            (∀ i : Fin 12, (served i = people correct i) ↔ (i ∈ {0, 1}) = true)) ∧
            (related_permutations served people))
    )
    sorry

end meal_serving_problem_l673_673748


namespace number_of_correct_statements_l673_673411

-- Definitions of conditions
def is_shortest_path_between_two_points (p1 p2 : Point) : Prop :=
  -- Insert the definition/axiom relevant to your geometry context
  sorry

def same_line (l1 l2 : Line) : Prop :=
  -- l1 and l2 are considered same if they share the same set of points
  sorry

def same_segment (s1 s2 : Segment) : Prop :=
  -- s1 and s2 are considered the same if they have the same endpoints
  endpoints s1 = endpoints s2

def same_ray (r1 r2 : Ray) : Prop :=
  -- r1 and r2 are considered the same if they start from the same point and extend in the same direction
  sorry

-- Statement to be proved
theorem number_of_correct_statements :
  let p₁ p₂ o a : Point in
  let l₁ l₂ : Line := (line_through p₁ p₂) in
  let s₁ s₂ : Segment := ⟨p₁, p₂⟩ in
  let r₁ r₂ : Ray := ⟨o, a⟩ in
  (¬is_shortest_path_between_two_points p₁ p₂) →
  (same_line l₁ l₂) →
  (same_segment s₁ s₂) →
  (¬same_ray r₁ r₂) →
  2 = (2 : ℕ) :=
sorry

end number_of_correct_statements_l673_673411


namespace evaluate_propositions_l673_673473

variables (l m : Line) (α β : Plane)

-- Conditions
axiom l_perp_alpha : perpendicular l α
axiom m_in_beta : contains β m

-- Propositions
def prop1 : Prop := parallel α β → perpendicular l m
def prop2 : Prop := perpendicular α β → parallel l m
def prop3 : Prop := parallel l m → perpendicular α β
def prop4 : Prop := perpendicular l m → parallel α β

-- Problem statement
theorem evaluate_propositions :
  (prop1 l m α β ∧ prop3 l m α β) ∧ ¬(prop2 l m α β ∨ prop4 l m α β) :=
by
  sorry

end evaluate_propositions_l673_673473


namespace largest_divisor_of_5_consecutive_integers_l673_673309

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l673_673309


namespace smallest_positive_period_of_f_range_of_f_intervals_of_monotonic_increase_of_f_l673_673038

def f (x : ℝ) : ℝ := sqrt 3 * Real.cos x + Real.sin x + 1

theorem smallest_positive_period_of_f :
  ∀ x : ℝ, f (x + 2 * Real.pi) = f x := sorry

theorem range_of_f :
  ∃ a b : ℝ, ∀ y : ℝ, y ∈ set.range f ↔ y ∈ set.Icc a b :=
begin
  use [-1, 3],
  sorry
end

theorem intervals_of_monotonic_increase_of_f :
  ∀ k : ℤ, ∀ x : ℝ, (2 * k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 6) → 
             (∀ h : 2 * k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi / 6, 
             ∃ b, ∀ c, b < c ∧ x = f c → True) := sorry

end smallest_positive_period_of_f_range_of_f_intervals_of_monotonic_increase_of_f_l673_673038


namespace sum_f_1_to_2015_l673_673496

noncomputable def f : ℝ → ℝ
| x =>
  if -3 ≤ x ∧ x < -1 then 
    -((x+2)^2) 
  else if -1 ≤ x ∧ x < 3 then 
    x 
  else 
    f (x - 6) -- This should logically define its periodicity with period 6

theorem sum_f_1_to_2015 : 
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 = 1 ∧
  f (1 + 6) + f (2 + 6) + f (3 + 6) + f (4 + 6) + f (5 + 6) + f (6 + 6) = 1 → 
  f 1 + f 2 + f 3 + ... + f 2015 = 336 :=
by 
  sorry

end sum_f_1_to_2015_l673_673496


namespace largest_divisor_of_5_consecutive_integers_l673_673299

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l673_673299


namespace construction_exists_l673_673207

-- Definitions and assumptions based on the problem conditions
variables (O : Point) (alt_line1 alt_line2 : Line)
-- Assuming the existence of the triangle ABC with center O for its circumcircle and defined altitudes
noncomputable def construct_triangle (O : Point) (alt_line1 alt_line2 : Line) : Triangle := sorry

-- Main theorem to construct the triangle
theorem construction_exists (O : Point) (alt_line1 alt_line2 : Line) :
  ∃ (ABC : Triangle), circumcenter ABC = O ∧ altitude ABC A ∈ alt_line1 ∧ altitude ABC B ∈ alt_line2 :=
begin
  use construct_triangle O alt_line1 alt_line2,
  -- Additional properties needed to validate the constructed triangle
  sorry -- steps to verify the constructed triangle
end

end construction_exists_l673_673207


namespace largest_divisor_of_product_of_5_consecutive_integers_l673_673235

/-- What is the largest integer that must divide the product of any 5 consecutive integers? -/
theorem largest_divisor_of_product_of_5_consecutive_integers :
  ∀ n : ℤ, ∃ d : ℤ, d = 24 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_5_consecutive_integers_l673_673235


namespace find_n_l673_673450

theorem find_n (n : ℝ) (h1 : 0 ≤ n) (h2 : n ≤ 180) : 
  cos n = cos 830 → n = 70 := by
  sorry

end find_n_l673_673450


namespace choose_club_l673_673546

-- Define the functions f and g according to the conditions.
def f (x : ℝ) : ℝ := 5 * x

def g (x : ℝ) : ℝ := if x ≤ 30 then 90 else 2 * x + 30

-- Define the correctness conditions for choosing the club.
theorem choose_club (x : ℝ) (hx : 15 ≤ x ∧ x ≤ 40) :
  (15 ≤ x ∧ x < 18 → f x < g x) ∧
  (x = 18 → f x = g x) ∧
  (18 < x ∧ x ≤ 30 → f x > g x) ∧
  (30 < x ∧ x ≤ 40 → f x > g x) :=
begin
  sorry
end

end choose_club_l673_673546


namespace sum_of_fractions_l673_673879

theorem sum_of_fractions :
  (∑ n in Finset.range 8, (1 : ℚ) / ((n + 1) * (n + 2))) = (8 : ℚ) / 9 :=
by sorry

end sum_of_fractions_l673_673879


namespace tan_alpha_minus_2beta_l673_673002

theorem tan_alpha_minus_2beta (α β : Real) 
  (h1 : Real.tan (α - β) = 2 / 5)
  (h2 : Real.tan β = 1 / 2) :
  Real.tan (α - 2 * β) = -1 / 12 := 
by 
  sorry

end tan_alpha_minus_2beta_l673_673002


namespace find_b50_l673_673009

noncomputable def T (n : ℕ) : ℝ := if n = 1 then 2 else 2 / (6 * n - 5)

noncomputable def b (n : ℕ) : ℝ :=
  if n = 1 then 2 else T n - T (n - 1)

theorem find_b50 : b 50 = -6 / 42677.5 := by sorry

end find_b50_l673_673009


namespace minimum_distance_MN_l673_673042

-- Define point M on the parabola y^2 = 4x, and point N
def M (m : ℝ) : ℝ × ℝ := (m^2 / 4, m)
def N : ℝ × ℝ := (3, 0)

-- Define the distance function between two points
def distance (P Q : ℝ × ℝ) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt

-- State the theorem to prove
theorem minimum_distance_MN : ∃ m : ℝ, distance (M m) N = 2 * real.sqrt 2 :=
sorry

end minimum_distance_MN_l673_673042


namespace eval_expression_l673_673442

theorem eval_expression (a b : ℤ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 := 
by 
  sorry

end eval_expression_l673_673442


namespace eval_expression_l673_673877

-- Definitions based on the conditions and problem statement
def x (b : ℕ) : ℕ := b + 9

-- The theorem to prove
theorem eval_expression (b : ℕ) : x b - b + 5 = 14 := by
    sorry

end eval_expression_l673_673877


namespace complex_number_solution_l673_673033

theorem complex_number_solution :
  ∃ z : ℂ, (sqrt 3 + 3 * complex.i) * z = 3 * complex.i ∧ z = (3 / 4) + (sqrt 3 * complex.i / 4) := 
sorry

end complex_number_solution_l673_673033


namespace shorter_leg_of_right_triangle_l673_673582

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l673_673582


namespace largest_divisor_of_consecutive_product_l673_673232

theorem largest_divisor_of_consecutive_product (n : ℤ) (h : ∀ k : ℤ, (n = 5 * k)) :
  ∃ d : ℤ, d = 60 ∧ ∀ a : ℤ, divides d (a * (a + 1) * (a + 2) * (a + 3) * (a + 4)) :=
sorry

end largest_divisor_of_consecutive_product_l673_673232


namespace smallest_composite_no_prime_factors_lt_15_l673_673911

theorem smallest_composite_no_prime_factors_lt_15 (n : ℕ) :
  ∀ n, (∀ p : ℕ, p.prime → p ∣ n → 15 ≤ p) → n = 289 → 
       is_composite n ∧ (∀ m : ℕ, (∀ q : ℕ, q.prime → q ∣ m → 15 ≤ q) → m ≥ 289) :=
by
  intros n hv hn
  -- Proof would go here
  sorry

end smallest_composite_no_prime_factors_lt_15_l673_673911


namespace find_pairs_l673_673447

theorem find_pairs (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 / b + b^2 / a = (a + b)^2 / (a + b)) ↔ (a = b) := by
  sorry

end find_pairs_l673_673447


namespace ellipse_problems_l673_673035

-- Condition: Ellipse definition and point on ellipse
def ellipse_eq (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def point_on_ellipse (a b : ℝ) : Prop := ellipse_eq (sqrt 6 / 2) (1 / 2) a b

-- Condition: Eccentricity
def eccentricity (a b : ℝ) : Prop := (sqrt 2 / 2) = sqrt (1 - b^2 / a^2)

-- Condition: Line chord interception and lengths
def line_eq (x y : ℝ) : Prop := 3 * x - 4 * y - 5 = 0
def chord_length (d : ℝ) : Prop := 2 * sqrt (1 + d^2 / 4 - (2 * abs (1 - d) / 5)^2) = 2

-- Proof problem statement
theorem ellipse_problems
  (a b t : ℝ)
  (h_conds1 : point_on_ellipse a b)
  (h_conds2 : eccentricity a b)
  (h1 : ellipse_eq 2 t a b)
  (h2 : t > 0)
  (h3 : line_eq 3 4)
  (h4 : chord_length t) :
  ellipse_eq 2 1 a b ∧ ((x y : ℝ), (__) → x^2 + y^2 - 2 * x - 4 * y = 0) ∧ (∃ N : ℝ, N = 1) := 
sorry

end ellipse_problems_l673_673035


namespace smallest_composite_no_prime_factors_below_15_correct_l673_673930

def smallest_composite_no_prime_factors_below_15 : Nat :=
  323
  
theorem smallest_composite_no_prime_factors_below_15_correct :
  (∀ n < 15, Prime n → ¬ (n ∣ smallest_composite_no_prime_factors_below_15)) ∧
  (∃ p q, Prime p ∧ Prime q ∧ p ≠ q ∧ smallest_composite_no_prime_factors_below_15 = p * q) :=
by
  -- Proof skipped
  sorry

end smallest_composite_no_prime_factors_below_15_correct_l673_673930


namespace max_value_u_l673_673470

theorem max_value_u (z : ℂ) (hz : |z| = 1) : |z^3 - 3 * z + 2| ≤ 3 * Real.sqrt 3 := 
sorry

end max_value_u_l673_673470


namespace numValidRoutesJackToJill_l673_673109

noncomputable def numPaths (n m : ℕ) : ℕ :=
  Nat.choose (n + m) n

theorem numValidRoutesJackToJill : 
  let totalRoutes := numPaths 5 3
  let pathsViaDanger := numPaths 2 2 * numPaths 3 1
  totalRoutes - pathsViaDanger = 32 :=
by
  let totalRoutes := numPaths 5 3
  let pathsViaDanger := numPaths 2 2 * numPaths 3 1
  show totalRoutes - pathsViaDanger = 32
  sorry

end numValidRoutesJackToJill_l673_673109


namespace product_of_five_consecutive_divisible_by_30_l673_673275

theorem product_of_five_consecutive_divisible_by_30 :
  ∀ n : ℤ, 30 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end product_of_five_consecutive_divisible_by_30_l673_673275


namespace sum_of_first_5_terms_of_geometric_sequence_l673_673994

theorem sum_of_first_5_terms_of_geometric_sequence :
  let a₁ := 3
  let q := 4
  let n := 5
  let Sₙ := λ n : ℕ, (a₁ * (1 - q^n)) / (1 - q)
  Sₙ 5 = 1023 :=
by
  sorry

end sum_of_first_5_terms_of_geometric_sequence_l673_673994


namespace constant_term_expansion_l673_673542

theorem constant_term_expansion (n : ℕ) (hn : n = 9) :
  y^3 * (x + 1 / (x^2 * y))^n = 84 :=
by sorry

end constant_term_expansion_l673_673542


namespace part_a_part_b_l673_673792

theorem part_a (points : List ℕ) (h_length : points.length = 100) (h_nodup : points.nodup) (h_sum_1_100 : points.sum = (100 * 101) / 2) :
  ∃ (pairs : List (ℕ × ℕ)), (∀ (p : ℕ × ℕ) (hp : p ∈ pairs), ((p.1 ∈ points) ∧ (p.2 ∈ points))) ∧
                        (pairs.length = 50) ∧
                        (∀ (p : ℕ × ℕ) (hp : p ∈ pairs), (p.1 + p.2) % 2 = 1) ∧
                        (∀ (p1 p2 : ℕ × ℕ) (hp1 : p1 ∈ pairs) (hp2 : p2 ∈ pairs) (ht : p1 ≠ p2),
                          ¬ (segments_intersect p1 p2)) :=
sorry

theorem part_b (points : List ℕ) (h_length : points.length = 100) (h_nodup : points.nodup) (h_sum_1_100 : points.sum = (100 * 101) / 2) :
  ¬ ∃ (pairs : List (ℕ × ℕ)), (∀ (p : ℕ × ℕ) (hp : p ∈ pairs), ((p.1 ∈ points) ∧ (p.2 ∈ points))) ∧
                         (pairs.length = 50) ∧
                         (∀ (p : ℕ × ℕ) (hp : p ∈ pairs), (p.1 + p.2) % 2 = 0) ∧
                         (∀ (p1 p2 : ℕ × ℕ) (hp1 : p1 ∈ pairs) (hp2 : p2 ∈ pairs) (ht : p1 ≠ p2),
                           ¬ (segments_intersect p1 p2)) :=
sorry

end part_a_part_b_l673_673792


namespace percentage_respondents_liked_B_l673_673814

variables (X Y : ℝ)
variables (likedA likedB likedBoth likedNeither : ℝ)
variables (totalRespondents : ℕ)

-- Conditions from the problem
def liked_conditions : Prop :=
    totalRespondents ≥ 100 ∧ 
    likedA = X ∧ 
    likedB = Y ∧ 
    likedBoth = 23 ∧ 
    likedNeither = 23

-- Proof statement
theorem percentage_respondents_liked_B (h : liked_conditions X Y likedA likedB likedBoth likedNeither totalRespondents) :
  Y = 100 - X :=
sorry

end percentage_respondents_liked_B_l673_673814


namespace probability_shaded_region_l673_673820

def triangle_game :=
  let total_regions := 6
  let shaded_regions := 3
  shaded_regions / total_regions

theorem probability_shaded_region:
  triangle_game = 1 / 2 := by
  sorry

end probability_shaded_region_l673_673820


namespace total_money_spent_l673_673204

def candy_bar_cost : ℕ := 14
def cookie_box_cost : ℕ := 39
def total_spent : ℕ := candy_bar_cost + cookie_box_cost

theorem total_money_spent : total_spent = 53 := by
  sorry

end total_money_spent_l673_673204


namespace probability_blue_face_l673_673857

theorem probability_blue_face (total_faces blue_faces red_faces green_face : ℕ) (h_total_faces : total_faces = 8) (h_blue_faces : blue_faces = 4) (h_red_faces : red_faces = 3) (h_green_face : green_face = 1) :
  (blue_faces / total_faces : ℚ) = 1 / 2 :=
by
  rw [h_total_faces, h_blue_faces]
  norm_num
  sorry

end probability_blue_face_l673_673857


namespace circle_center_radius_l673_673505

theorem circle_center_radius : 
  ∀ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 4 → 
  (∃ c : ℝ × ℝ, c = (-1, 2)) ∧ 
  (∃ r : ℝ, r = 2) :=
by
  intros x y h
  use (-1, 2)
  use 2
  sorry

end circle_center_radius_l673_673505


namespace water_tower_excess_consumption_l673_673404

def water_tower_problem : Prop :=
  let initial_water := 2700
  let first_neighborhood := 300
  let second_neighborhood := 2 * first_neighborhood
  let third_neighborhood := second_neighborhood + 100
  let fourth_neighborhood := 3 * first_neighborhood
  let fifth_neighborhood := third_neighborhood / 2
  let leakage := 50
  let first_neighborhood_final := first_neighborhood + 0.10 * first_neighborhood
  let second_neighborhood_final := second_neighborhood - 0.05 * second_neighborhood
  let third_neighborhood_final := third_neighborhood + 0.10 * third_neighborhood
  let fifth_neighborhood_final := fifth_neighborhood - 0.05 * fifth_neighborhood
  let total_consumption := 
    first_neighborhood_final + second_neighborhood_final + third_neighborhood_final +
    fourth_neighborhood + fifth_neighborhood_final + leakage
  let excess_consumption := total_consumption - initial_water
  excess_consumption = 252.5

theorem water_tower_excess_consumption : water_tower_problem := by
  sorry

end water_tower_excess_consumption_l673_673404


namespace solve_system_eq_l673_673668

theorem solve_system_eq (a b c x y z : ℝ) (h1 : x / (a * b) + y / (b * c) + z / (a * c) = 3)
  (h2 : x / a + y / b + z / c = a + b + c) (h3 : c^2 * x + a^2 * y + b^2 * z = a * b * c * (a + b + c)) :
  x = a * b ∧ y = b * c ∧ z = a * c :=
by
  sorry

end solve_system_eq_l673_673668


namespace largest_divisor_of_product_of_five_consecutive_integers_l673_673318

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l673_673318


namespace largest_divisor_of_5_consecutive_integers_l673_673300

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l673_673300


namespace volume_largest_smaller_sphere_same_center_l673_673530

noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_largest_smaller_sphere_same_center 
  (O B : ℝ → ℝ → ℝ) 
  (r : ℝ) 
  (hS : dist O B = 10) 
  (hB : r = 10) : 
  sphereVolume 5 = 500 / 3 * Real.pi := 
by 
  sorry

end volume_largest_smaller_sphere_same_center_l673_673530
