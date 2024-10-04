import Mathlib

namespace real_part_bounds_l743_743087

theorem real_part_bounds (z : ℂ) (h : complex.abs (z - 1) = 1) :
  -1 ≤ (complex.re (1 / (z - 1))) ∧ (complex.re (1 / (z - 1))) ≤ 1 :=
sorry

end real_part_bounds_l743_743087


namespace volume_of_cuboid_l743_743960

-- Definitions of conditions
def side_length : ℕ := 6
def num_cubes : ℕ := 3
def volume_single_cube (side_length : ℕ) : ℕ := side_length ^ 3

-- The main theorem
theorem volume_of_cuboid : (num_cubes * volume_single_cube side_length) = 648 := by
  sorry

end volume_of_cuboid_l743_743960


namespace surface_area_ratio_l743_743953

theorem surface_area_ratio (r l : ℝ) : r > 0 ∧ l > 0 → (2 * π * r * l) / (π * r * l) = 2 := by
  intros h
  calc
    (2 * π * r * l) / (π * r * l)
        = 2 * π * r * l / (π * r * l)  : by rw [mul_div_assoc]
    ... = 2 * (π * r * l) / (π * r * l)  : by rw [mul_assoc]
    ... = 2 * 1                          : by rw [div_self (mul_ne_zero (mul_ne_zero pi_ne_zero (ne_of_gt h.1)) (ne_of_gt h.2))]
    ... = 2                              : by rw [mul_one]

end surface_area_ratio_l743_743953


namespace minimal_sum_of_b_l743_743364

theorem minimal_sum_of_b :
  let f : (List ℕ) → ℕ := λ seq, seq.foldr (λ x y, x^y) 1
  ∃ (b : Fin 2017 → ℕ), 
  (∀ (a : Fin 2017 → ℕ), 
   (∀ i, 2017 < a i) → 
   (∀ i, f (List.ofFn a) % 2017 = f (List.updateNth (List.ofFn a) i ((a i) + b i)) % 2017)) 
  ∧ (∀ i, b i ∈ {1, 2, 4, 8, 16, 32, 224, 672, 2016})
  ∧ List.sum (List.ofFn b) = 4983 :=
by
  sorry

end minimal_sum_of_b_l743_743364


namespace second_set_matches_l743_743929

theorem second_set_matches (avg_20_matches : ℕ) (avg_second_set_matches : ℕ) (avg_30_matches : ℕ) (total_first_set : ℕ) (total_all_matches : ℕ) :
  avg_20_matches = 30 → avg_second_set_matches = 15 → avg_30_matches = 25 → total_first_set = 20 * 30 →
  total_all_matches = 30 * 25 →
  ∃ (x : ℕ), total_first_set + 15 * x = total_all_matches ∧ x = 10 :=
by
  intros h1 h2 h3 h4 h5
  use 10
  split
  sorry

end second_set_matches_l743_743929


namespace area_increase_by_6_65_percent_l743_743252

variable {L B : ℝ} -- Initial length and breadth

-- Increase length by 35%
def L_new : ℝ := 1.35 * L

-- Decrease breadth by 21%
def B_new : ℝ := 0.79 * B

-- Initial area of the rectangle
def A_initial : ℝ := L * B

-- New area of the rectangle
def A_new : ℝ := L_new * B_new

-- Calculate the area change
def delta_A : ℝ := A_new - A_initial

-- Prove that the area increases by 6.65%
theorem area_increase_by_6_65_percent : 
  (delta_A / A_initial) = 0.0665 :=
by
  sorry

end area_increase_by_6_65_percent_l743_743252


namespace log_relationship_l743_743152

noncomputable def a := Real.logBase 3 Real.pi
noncomputable def b := Real.logBase 2 (Real.sqrt 3)
noncomputable def c := Real.logBase 3 (Real.sqrt 2)

theorem log_relationship : a > b ∧ b > c := by
  sorry

end log_relationship_l743_743152


namespace total_wings_of_birds_l743_743064

def total_money_from_grandparents (gift : ℕ) (grandparents : ℕ) : ℕ := gift * grandparents

def number_of_birds (total_money : ℕ) (bird_cost : ℕ) : ℕ := total_money / bird_cost

def total_wings (birds : ℕ) (wings_per_bird : ℕ) : ℕ := birds * wings_per_bird

theorem total_wings_of_birds : 
  ∀ (gift amount : ℕ) (grandparents bird_cost wings_per_bird : ℕ),
  gift = 50 → 
  amount = 200 →
  grandparents = 4 → 
  bird_cost = 20 → 
  wings_per_bird = 2 → 
  total_wings (number_of_birds amount bird_cost) wings_per_bird = 20 :=
by {
  intros gift amount grandparents bird_cost wings_per_bird gift_eq amount_eq grandparents_eq bird_cost_eq wings_per_bird_eq,
  rw [gift_eq, amount_eq, grandparents_eq, bird_cost_eq, wings_per_bird_eq],
  simp [total_wings, total_money_from_grandparents, number_of_birds],
  sorry
}

end total_wings_of_birds_l743_743064


namespace projectile_reaches_24m_at_12_7_seconds_l743_743204

theorem projectile_reaches_24m_at_12_7_seconds :
  ∃ t : ℝ, (y = -4.9 * t^2 + 25 * t) ∧ y = 24 ∧ t = 12 / 7 :=
by
  use 12 / 7
  sorry

end projectile_reaches_24m_at_12_7_seconds_l743_743204


namespace number_of_axisymmetric_and_centrosymmetric_shapes_l743_743647

-- Define the properties of shapes in terms of axisymmetry and centrosymmetry
inductive Shape
| equilateral_triangle
| parallelogram
| rectangle
| rhombus
| isosceles_trapezoid

def is_axisymmetric : Shape → Prop
| Shape.equilateral_triangle := True
| Shape.parallelogram := False
| Shape.rectangle := True
| Shape.rhombus := True
| Shape.isosceles_trapezoid := True

def is_centrosymmetric : Shape → Prop
| Shape.equilateral_triangle := False
| Shape.parallelogram := True
| Shape.rectangle := True
| Shape.rhombus := True
| Shape.isosceles_trapezoid := False

-- Main theorem to prove
theorem number_of_axisymmetric_and_centrosymmetric_shapes :
  (Finset.filter (λ s, is_axisymmetric s ∧ is_centrosymmetric s)
    (Finset.of_list [Shape.equilateral_triangle, Shape.parallelogram,
                     Shape.rectangle, Shape.rhombus, Shape.isosceles_trapezoid])).card = 2 :=
by
  sorry

end number_of_axisymmetric_and_centrosymmetric_shapes_l743_743647


namespace vector_AD_magnitude_l743_743377

variables {V : Type*} [inner_product_space ℝ V]

-- Given conditions
variables (m n : V)
def angle (m n : V) : Prop := real.arccos ((inner_product_space.inner m n) / (∥m∥ * ∥n∥)) = π / 6
def magnitude_m : ∥m∥ = real.sqrt 3 := sorry
def magnitude_n : ∥n∥ = 2 := sorry

-- Vector operations in triangle ABC
def AB := m + n
def AC := m - 3 • n

-- Midpoint of BC and vector AD
def D (B C : V) := (B + C) / 2
def AD := D AB AC

theorem vector_AD_magnitude (h_angle : angle m n) (h_magnitude_m : magnitude_m) (h_magnitude_n : magnitude_n) :
  ∥AD∥ = 1 := 
sorry

end vector_AD_magnitude_l743_743377


namespace area_clipping_l743_743614

def initial_area : ℝ := 1
def min_area : ℝ := 1 / 3

def clipped_area (area : ℝ) : ℝ := area - (1 / 4) * area

theorem area_clipping (n : ℕ) (h : n ≥ 6) :
  ∀ (P : ℕ → ℝ), (P 6 = initial_area) →
  (∀ k, k ≥ 6 → P (k + 1) = clipped_area (P k)) →
  P n > min_area :=
by
  sorry

end area_clipping_l743_743614


namespace no_real_x_makes_expression_rational_l743_743714

theorem no_real_x_makes_expression_rational (x : ℝ) : ¬ ∃ (r : ℚ), 
  x - real.sqrt (x^2 + 4) + 2 / (x - real.sqrt (x^2 + 4)) = r := 
sorry

end no_real_x_makes_expression_rational_l743_743714


namespace isosceles_triangle_l743_743110

noncomputable theory

open_locale classical

variables {A B C I P Q : Type*}

-- Let \( I \) be the incenter of triangle \( ABC \)
def is_incenter (I A B C : Type*) : Prop := sorry

-- Let \( \alpha \) be its incircle
def is_incircle (α : Type*) (I A B C : Type*) : Prop := sorry

-- The circumcircle of triangle \( AIC \) intersects \( \alpha \) at points \( P \) and \( Q \)
def circumcircle_intersect_incircle (α A I C P Q : Type*) : Prop := sorry

-- \( P \) and \( A \) lie on the same side of line \( BI \), and \( Q \) and \( C \) lie on the other side
def same_side (P A Q C : Type*) (BI : Type*) : Prop := sorry

-- \( PQ \parallel AC \)
def parallel (PQ AC : Type*) : Prop := sorry

-- Define triangle is isosceles
def is_isosceles (A B C : Type*) : Prop := sorry

theorem isosceles_triangle 
  (I : Type*) (A B C P Q M N : Type*)
  (α : Type*)
  (h_incenter : is_incenter I A B C)
  (h_incircle : is_incircle α I A B C)
  (h_intersect : circumcircle_intersect_incircle α A I C P Q)
  (h_sameside : same_side P A Q C (line BI))
  (h_parallel : parallel PQ (line AC))
: is_isosceles A B C :=
sorry

end isosceles_triangle_l743_743110


namespace num_valid_license_plates_l743_743009

-- Define a license plate constraint
def valid_license_plate (plate : String) : Prop :=
  plate.length = 4 ∧
  plate[0].isAlpha ∧ plate[1].isAlpha ∧
  plate[2].isDigit ∧ plate[3].isDigit ∧
  (plate[0] = plate[2] ∨ plate[0] = plate[3] ∨ plate[1] = plate[2] ∨ plate[1] = plate[3])

-- Prove the number of valid license plates
theorem num_valid_license_plates : 
  ∃ n : ℕ, n = 270400 ∧ ∀ plate : String, valid_license_plate plate → plate.length = 4 := 
sorry

end num_valid_license_plates_l743_743009


namespace spherical_circle_radius_l743_743593

noncomputable def radius_of_spherical_circle : ℝ :=
  let ρ := 1
  let φ := real.pi / 4
  let r := real.sqrt ((ρ * real.sin φ * real.cos 0)^2 + (ρ * real.sin φ * real.sin 0)^2)
  r

theorem spherical_circle_radius :
  ∃ r : ℝ, r = radius_of_spherical_circle ∧ r = real.sqrt 2 / 2 :=
by
  sorry

end spherical_circle_radius_l743_743593


namespace triangle_is_right_angle_l743_743026

open Real

variables (A B C a b c : ℝ)

-- Given that in ΔABC, the sides opposite to angles A, B, C are a, b, c respectively
-- and the condition a = -c * cos (A + C)
def triangle_shape_cond (A B C a b c : ℝ) : Prop :=
  a = -c * cos (A + C)

-- Prove that the shape of ΔABC is a right-angled triangle
theorem triangle_is_right_angle (A B C a b c : ℝ)
  (h1 : triangle_shape_cond A B C a b c)
  (h2 : a^2 + b^2 = c^2) : 
  ∃ B, A + B + C = π ∧ sin B = 1 := sorry

end triangle_is_right_angle_l743_743026


namespace algebraic_expression_constant_l743_743644

theorem algebraic_expression_constant (x : ℝ) : x * (x - 6) - (3 - x) ^ 2 = -9 :=
sorry

end algebraic_expression_constant_l743_743644


namespace rationalize_denominator_l743_743540

theorem rationalize_denominator :
  ∃ A B C : ℤ, C > 0 ∧ ¬(∃ p : ℕ, p.prime ∧ p^3 ∣ B) ∧
  (A + B + C = 74) ∧
  (C ≠ 0) ∧
  ((4 : ℚ) / (3 * real.cbrt (7 : ℚ)) = (A * real.cbrt (B : ℚ)) / (C : ℚ)) :=
begin
  -- Existence of A, B, and C such that the conditions hold.
  use [4, 49, 21],
  split,
  -- C > 0
  norm_num,
  split,
  -- B is not divisible by the cube of any prime
  intro h,
  linarith,
  split,
  -- A + B + C = 74
  norm_num,
  split,
  -- C ≠ 0
  norm_num,
  -- Rationalized form matches.
  exact sorry,
end

end rationalize_denominator_l743_743540


namespace symmetry_propositions_l743_743794

noncomputable def verify_symmetry_conditions (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  Prop :=
  -- This defines the propositions to be proven
  (∀ x : ℝ, a^x - 1 = a^(-x) - 1) ∧
  (∀ x : ℝ, a^(x - 2) = a^(2 - x)) ∧
  (∀ x : ℝ, a^(x + 2) = a^(2 - x))

-- Create the problem statement
theorem symmetry_propositions (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  verify_symmetry_conditions a h1 h2 :=
sorry

end symmetry_propositions_l743_743794


namespace exists_nat_numbers_satisfying_sum_l743_743695

theorem exists_nat_numbers_satisfying_sum :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 :=
sorry

end exists_nat_numbers_satisfying_sum_l743_743695


namespace ABC_is_isosceles_l743_743123

open Mobius

variables
  (ABC : Triangle)
  (I : Point)
  (α P Q M N : Point)
  (A B C Z : Point)
  [Incenter I ABC]
  [Incircle α ABC]
  [Circumcircle (Triangle.mk A I C)]
  [PQ_parallel_AC : is_parallel (Segment.mk P Q) (Segment.mk A C)]
  [Midpoint M (Arc.mk A C α)]
  [Midpoint N (Arc.mk B C α)]
  [center_Z : Center Z (Circumcircle (Triangle.mk A I C))]
  [Chord_α_PQ : common_chord α (Circumcircle (Triangle.mk A I C)) (Segment.mk P Q)]
  [P_A_same_BI : same_side P A (Line.mk B I)]
  [Q_C_other_BI : same_side Q C (Line.mk B I)]

theorem ABC_is_isosceles
  (h_parallel : PQ_parallel_AC) : 
  Isosceles ABC := 
sorry

end ABC_is_isosceles_l743_743123


namespace part1_solution_part2_solution_l743_743800

-- Definition for part 1
noncomputable def f_part1 (x : ℝ) := abs (x - 3) + 2 * x

-- Proof statement for part 1
theorem part1_solution (x : ℝ) : (f_part1 x ≥ 3) ↔ (x ≥ 0) :=
by sorry

-- Definition for part 2
noncomputable def f_part2 (x a : ℝ) := abs (x - a) + 2 * x

-- Proof statement for part 2
theorem part2_solution (a : ℝ) : 
  (∀ x, f_part2 x a ≤ 0 ↔ x ≤ -2) → (a = 2 ∨ a = -6) :=
by sorry

end part1_solution_part2_solution_l743_743800


namespace combination_identity_l743_743777

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
factorial n / (factorial r * factorial (n - r))

theorem combination_identity (m : ℕ) (hm : 0 ≤ m ∧ m ≤ 5) :
  combinaton 8 m = 28 :=
by
  have cond1: (1 / combination 5 m) - (1 / combination 6 m) = (7 / (10 * combination 7 m)) := sorry
  -- Rest of the proof establishing that m = 2
  -- Simplification and proof steps skipped
  sorry

end combination_identity_l743_743777


namespace max_values_of_x_max_area_abc_l743_743759

noncomputable def m (x : ℝ) : ℝ × ℝ := ⟨2 * Real.sin x, Real.sin x - Real.cos x⟩
noncomputable def n (x : ℝ) : ℝ × ℝ := ⟨Real.sqrt 3 * Real.cos x, Real.sin x + Real.cos x⟩
noncomputable def f (x : ℝ) : ℝ := Prod.fst (m x) * Prod.fst (n x) + Prod.snd (m x) * Prod.snd (n x)

theorem max_values_of_x
  (k : ℤ) : ∃ x, x = k * Real.pi + Real.pi / 3 ∧ f x = 2 * Real.sin (2 * x - π / 6) :=
sorry

noncomputable def C : ℝ := Real.pi / 3
noncomputable def area_abc (a b c : ℝ) : ℝ := 1 / 2 * a * b * Real.sin C

theorem max_area_abc (a b : ℝ) (h₁ : c = Real.sqrt 3) (h₂ : f C = 2) :
  area_abc a b c ≤ 3 * Real.sqrt 3 / 4 :=
sorry

end max_values_of_x_max_area_abc_l743_743759


namespace eq_value_of_ratio_l743_743810

theorem eq_value_of_ratio (x y : ℝ) (h₁ : x^2 + sqrt 3 * y = 4) (h₂ : y^2 + sqrt 3 * x = 4) (h₃ : x ≠ y) :
  y / x + x / y = -5 :=
sorry

end eq_value_of_ratio_l743_743810


namespace ratio_triangle_def_rectangle_abcd_l743_743456

theorem ratio_triangle_def_rectangle_abcd (ABCD : Type)
  [rectangle ABCD]
  (DC : ℝ) (CB : ℝ) (E F : Point ABCD) 
  (H1 : DC = 3 * CB)
  (H2 : E ∈ AB) 
  (H3 : F ∈ AB)
  (H4 : ∠ADC = 90)
  (H5 : bisect ∠EDC ∧ bisect ∠FDC)
  : (area (triangle DEF)) / (area (rectangle ABCD)) = 1/6 := 
sorry

end ratio_triangle_def_rectangle_abcd_l743_743456


namespace transformed_mean_variance_l743_743366

variable (n : ℕ)
variable (x : Fin n → ℝ)

-- Given conditions
def mean (x : Fin n → ℝ) : ℝ := (Finset.univ.sum (λ i, x i)) / n
def variance (x : Fin n → ℝ) : ℝ :=
  (Finset.univ.sum (λ i, (x i - mean x) ^ 2)) / n

axiom mean_x : mean x = 2
axiom variance_x : variance x = 1

-- Proving the desired mean and variance for the transformed set
theorem transformed_mean_variance :
  mean (λ i, 2 * (x i) + 1) = 5 ∧ variance (λ i, 2 * (x i) + 1) = 4 := by
sorry

end transformed_mean_variance_l743_743366


namespace new_computer_cost_l743_743066

theorem new_computer_cost 
  (C : ℝ)
  (monitor_peripherals_cost : ℝ := 1/5 * C)
  (base_video_card_cost : ℝ := 300)
  (new_video_card_cost : ℝ := 2 * base_video_card_cost)
  (additional_cost : ℝ := new_video_card_cost - base_video_card_cost)
  (total_cost : ℝ := C + monitor_peripherals_cost + additional_cost) :
  total_cost = 2100 → C = 1500 :=
by 
  intro h
  have h1 : monitor_peripherals_cost = 1/5 * C := rfl
  have h2 : new_video_card_cost = 600 := by simp [new_video_card_cost]
  have h3 : additional_cost = 300 := by simp [additional_cost, h2]
  have h4 : total_cost = C + 1/5 * C + 300 := by simp [total_cost, h1, h3]
  linarith [h, h4]

end new_computer_cost_l743_743066


namespace isosceles_triangle_l743_743105

noncomputable theory

open_locale classical

variables {A B C I P Q : Type*}

-- Let \( I \) be the incenter of triangle \( ABC \)
def is_incenter (I A B C : Type*) : Prop := sorry

-- Let \( \alpha \) be its incircle
def is_incircle (α : Type*) (I A B C : Type*) : Prop := sorry

-- The circumcircle of triangle \( AIC \) intersects \( \alpha \) at points \( P \) and \( Q \)
def circumcircle_intersect_incircle (α A I C P Q : Type*) : Prop := sorry

-- \( P \) and \( A \) lie on the same side of line \( BI \), and \( Q \) and \( C \) lie on the other side
def same_side (P A Q C : Type*) (BI : Type*) : Prop := sorry

-- \( PQ \parallel AC \)
def parallel (PQ AC : Type*) : Prop := sorry

-- Define triangle is isosceles
def is_isosceles (A B C : Type*) : Prop := sorry

theorem isosceles_triangle 
  (I : Type*) (A B C P Q M N : Type*)
  (α : Type*)
  (h_incenter : is_incenter I A B C)
  (h_incircle : is_incircle α I A B C)
  (h_intersect : circumcircle_intersect_incircle α A I C P Q)
  (h_sameside : same_side P A Q C (line BI))
  (h_parallel : parallel PQ (line AC))
: is_isosceles A B C :=
sorry

end isosceles_triangle_l743_743105


namespace frog_ends_on_vertical_side_l743_743666

-- Definitions for rectangle boundary conditions
def boundary_conditions (x y : ℕ) : ℚ :=
  if x = 0 ∨ x = 5 then 1
  else if y = 0 ∨ y = 5 then 0
  else if x = y then -- Adjusted probabilities for x = y
    (2/6 * boundary_conditions (x-1) y + 2/6 * boundary_conditions (x+1) y +
     1/6 * boundary_conditions x (y-1) + 1/6 * boundary_conditions x (y+1))
  else
    (1/4 * boundary_conditions (x-1) y + 1/4 * boundary_conditions (x+1) y + 
     1/4 * boundary_conditions x (y-1) + 1/4 * boundary_conditions x (y+1))

-- Main theorem to prove that the probability the frog ends on a vertical side from (2, 3) is 3/5
theorem frog_ends_on_vertical_side :
  boundary_conditions 2 3 = 3 / 5 := 
  sorry

end frog_ends_on_vertical_side_l743_743666


namespace remainder_when_divided_l743_743789

theorem remainder_when_divided (m : ℤ) (h : m % 5 = 2) : (m + 2535) % 5 = 2 := 
by sorry

end remainder_when_divided_l743_743789


namespace a_lt_2_is_necessary_but_not_sufficient_for_a_squared_lt_4_l743_743954

theorem a_lt_2_is_necessary_but_not_sufficient_for_a_squared_lt_4 (a : ℝ) :
  (a < 2 → a^2 < 4) ∧ (a^2 < 4 → a < 2) :=
by
  -- Proof skipped
  sorry

end a_lt_2_is_necessary_but_not_sufficient_for_a_squared_lt_4_l743_743954


namespace reciprocal_of_repeating_decimal_three_l743_743984

noncomputable def repeating_decimal_to_fraction : ℚ :=
  let x := (0.33333333333 : ℚ) in 1 / 3

theorem reciprocal_of_repeating_decimal_three : 
  (1 / repeating_decimal_to_fraction) = 3 := by
  -- Reciprocal of the fraction
  sorry

end reciprocal_of_repeating_decimal_three_l743_743984


namespace distinct_integers_problem_l743_743646

variable (a b c d e : ℤ)

theorem distinct_integers_problem
  (h1 : a ≠ b) 
  (h2 : a ≠ c) 
  (h3 : a ≠ d) 
  (h4 : a ≠ e) 
  (h5 : b ≠ c) 
  (h6 : b ≠ d) 
  (h7 : b ≠ e) 
  (h8 : c ≠ d) 
  (h9 : c ≠ e) 
  (h10 : d ≠ e) 
  (h_prod : (4 - a) * (4 - b) * (4 - c) * (4 - d) * (4 - e) = 12) : 
  a + b + c + d + e = 17 := 
sorry

end distinct_integers_problem_l743_743646


namespace cosine_angle_l743_743293

-- Mathematical Constants and Definitions
variables (KO1 LO2 : ℝ) (k : ℝ) (α : ℝ)

-- Conditions from the problem
def ratio_condition (KO1 LO2 : ℝ) (k : ℝ) : Prop := KO1 / LO2 = k

-- Lean Theorem Statement
theorem cosine_angle (h : ratio_condition KO1 LO2 k) : cos α = 1 - k :=
sorry

end cosine_angle_l743_743293


namespace triangle_incircle_ratio_l743_743296

theorem triangle_incircle_ratio (r s q : ℝ) (h1 : r + s = 8) (h2 : r < s) (h3 : r + q = 13) (h4 : s + q = 17) (h5 : 8 + 13 > 17 ∧ 8 + 17 > 13 ∧ 13 + 17 > 8):
  r / s = 1 / 3 := by sorry

end triangle_incircle_ratio_l743_743296


namespace coordinate_sum_l743_743387

theorem coordinate_sum (f : ℝ → ℝ) (x y : ℝ) (h₁ : f 9 = 7) (h₂ : 3 * y = f (3 * x) / 3 + 3) (h₃ : x = 3) : 
  x + y = 43 / 9 :=
by
  -- Proof goes here
  sorry

end coordinate_sum_l743_743387


namespace milk_production_days_l743_743014

theorem milk_production_days (x : ℝ) (hx_pos : 0 < x) : 
  let daily_prod_per_cow := (x + 10) / (x * (x + 2)),
      efficiency_factor := 1 + 0.10 * 4,
      adj_daily_prod_per_cow := daily_prod_per_cow * efficiency_factor,
      total_daily_prod := (x + 4) * adj_daily_prod_per_cow,
      number_of_days := (2 * x + 20) / total_daily_prod in
  number_of_days = (x * (x + 2) * (2 * x + 20)) / (1.4 * (x + 4) * (x + 10)) :=
by
  sorry

end milk_production_days_l743_743014


namespace minimum_x2_y2_z2_l743_743325

theorem minimum_x2_y2_z2 :
  ∀ x y z : ℝ, (x^3 + y^3 + z^3 - 3 * x * y * z = 1) → (∃ a b c : ℝ, a = x ∨ a = y ∨ a = z ∧ b = x ∨ b = y ∨ b = z ∧ c = x ∨ c = y ∨ a ≤ b ∨ a ≤ c ∧ b ≤ c) → (x^2 + y^2 + z^2 ≥ 1) :=
by
  sorry

end minimum_x2_y2_z2_l743_743325


namespace integer_pairs_satisfying_equation_l743_743419

-- Define the problem to find the number of integer pairs (x, y) satisfying the equation
theorem integer_pairs_satisfying_equation :
  {p : ℤ × ℤ | (p.1 ^ 6 + p.2 ^ 2 = 6 * p.2)}.to_finset.card = 2 :=
by sorry

end integer_pairs_satisfying_equation_l743_743419


namespace coordinate_sum_l743_743906

-- Definitions for the conditions
def point_C (y : ℝ) : ℝ × ℝ := (3, y + 4)
def point_D (C : ℝ × ℝ) : ℝ × ℝ := (-C.1, C.2)
def y_value : ℝ := 2

-- Statement of the problem
theorem coordinate_sum :
  let C := point_C y_value in
  let D := point_D C in
  C.1 + C.2 + D.1 + D.2 = 12 :=
by
  let C := point_C y_value
  let D := point_D C
  sorry

end coordinate_sum_l743_743906


namespace road_equation_parabola_l743_743256

theorem road_equation_parabola (d : ℝ) (d_ne_zero : d ≠ 0) (x y : ℝ) :
  (sqrt (x^2 + y^2) = abs (y - d)) → (y = d / 2 - x^2 / (2 * d)) := by
  sorry

end road_equation_parabola_l743_743256


namespace exist_positive_integers_summing_to_one_l743_743602

theorem exist_positive_integers_summing_to_one :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (1 / (x:ℚ) + 1 / (y:ℚ) + 1 / (z:ℚ) = 1)
    ∧ ((x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 3 ∧ y = 3 ∧ z = 3)) :=
by
  sorry

end exist_positive_integers_summing_to_one_l743_743602


namespace hike_length_proof_l743_743409

variables
  (total_water_initial : ℕ)
  (hike_duration_hours : ℕ)
  (water_remaining : ℕ)
  (leak_rate : ℕ)
  (water_drank_last_mile : ℕ)
  (water_drank_per_mile : ℕ)

def total_water_leaked := leak_rate * hike_duration_hours
def total_water_used := total_water_initial - water_remaining
def total_water_drank := total_water_used - total_water_leaked
def water_drank_first_part := total_water_drank - water_drank_last_mile
def first_part_distance := water_drank_first_part / water_drank_per_mile
def total_hike_length := first_part_distance + 1

theorem hike_length_proof
  (h_initial : total_water_initial = 10)
  (h_duration : hike_duration_hours = 2)
  (h_remaining : water_remaining = 2)
  (h_leak_rate : leak_rate = 1)
  (h_last_mile : water_drank_last_mile = 3)
  (h_per_mile : water_drank_per_mile = 1) :
  total_hike_length = 4 :=
  by
    unfold total_hike_length
    unfold first_part_distance
    unfold water_drank_first_part
    unfold total_water_drank
    unfold total_water_used
    unfold total_water_leaked
    rw [h_initial, h_duration, h_remaining, h_leak_rate, h_last_mile, h_per_mile]
    simp
    sorry

end hike_length_proof_l743_743409


namespace exists_set_satisfying_conditions_l743_743713

-- Statement of the problem in Lean 4

open Set

theorem exists_set_satisfying_conditions (n : ℕ) (hn : n ≥ 4) :
  ∃ (S : Finset ℕ), (S.card = n) ∧
                    (∀ x ∈ S, x < 2^(n - 1)) ∧ 
                    ∀ A B, A ⊆ S → B ⊆ S → 
                    A ≠ B → (∑ x in A, x) ≠ (∑ x in B, x) :=
sorry

end exists_set_satisfying_conditions_l743_743713


namespace part1_monotonic_intervals_when_a_zero_part2_find_range_of_a_l743_743760

variable {a x : ℝ}
def f (x : ℝ) (a : ℝ) := (x^2 + a * x) * Real.exp x

-- 1. Monotonicity assessment when a = 0
theorem part1_monotonic_intervals_when_a_zero :
  (∀ x, (x^2 * Real.exp x).deriv x > 0 ↔ (x < -2 ∨ x > 0)) ∧
  (∀ x, (x^2 * Real.exp x).deriv x < 0 ↔ (-2 < x ∧ x < 0)) := sorry

-- 2. Range of values for a when f(x) is monotonically decreasing on (1,2)
theorem part2_find_range_of_a (a : ℝ) :
  (∀ x ∈ Ioo (1 : ℝ) 2, x^2 + (a + 2) * x + a ≤ 0) → a ≤ -8/3 := sorry

end part1_monotonic_intervals_when_a_zero_part2_find_range_of_a_l743_743760


namespace contradiction_proof_l743_743615

theorem contradiction_proof (a b : ℝ) (h : a ≥ b) (h_pos : b > 0) (h_contr : a^2 < b^2) : false :=
by {
  sorry
}

end contradiction_proof_l743_743615


namespace brown_eyed_kittens_second_cat_l743_743895

-- Define the main problem using Lean's theorem proving capabilities.
theorem brown_eyed_kittens_second_cat :
  ∃ B : ℕ, 
    let total_kittens := 3 + 7 + 4 + B in
    let blue_eyed_kittens := 3 + 4 in
    let percentage_blue := (blue_eyed_kittens * 100) / total_kittens in
    (percentage_blue = 35) → (B = 6) :=
by
  let B := 6
  existsi B
  sorry

end brown_eyed_kittens_second_cat_l743_743895


namespace hypotenuse_length_l743_743583

def triangle_hypotenuse := ∃ (a b c : ℚ) (x : ℚ), 
  a = 9 ∧ b = 3 * x + 6 ∧ c = x + 15 ∧ 
  a + b + c = 45 ∧ 
  a^2 + b^2 = c^2 ∧ 
  x = 15 / 4 ∧ 
  c = 75 / 4

theorem hypotenuse_length : triangle_hypotenuse :=
sorry

end hypotenuse_length_l743_743583


namespace difference_of_iterative_averages_l743_743558

def iterative_average : List ℚ → ℚ 
| []       := 0
| [a]      := a
| (a :: b) := (a + iterative_average b) / 2

def max_difference (lst : List ℕ) : ℚ :=
let sequences := lst.permutations
let values := sequences.map (λ seq => iterative_average (seq.map (λ n => (n : ℚ))))
let max_val := values.foldl max 0
let min_val := values.foldl min (values.head! : ℚ)
in max_val - min_val

theorem difference_of_iterative_averages : max_difference [1, 2, 3, 4, 5, 6] = 49 / 16 := 
by
  sorry

end difference_of_iterative_averages_l743_743558


namespace ball_first_bounce_less_than_30_l743_743655

theorem ball_first_bounce_less_than_30 (b : ℕ) :
  (243 * ((2: ℝ) / 3) ^ b < 30) ↔ (b ≥ 6) :=
sorry

end ball_first_bounce_less_than_30_l743_743655


namespace divisibility_of_polynomial_roots_l743_743532

theorem divisibility_of_polynomial_roots
  {p q : ℤ} {a_n a_{n-1} a_{n-2} ... a_1 a_0: ℤ}
  (h_coprime : Int.gcd p q = 1)
  (h_root : ∀ x, x = (p : ℚ) / (q : ℚ) → 
     (a_n : ℚ) * x^n + (a_{n-1} : ℚ) * x^(n-1) + ... + (a_1 : ℚ) * x + (a_0 : ℚ) = 0) :
  (p ∣ a_0) ∧ (q ∣ a_n) :=
by
  sorry

end divisibility_of_polynomial_roots_l743_743532


namespace sprinkler_days_needed_l743_743276

-- Definitions based on the conditions
def morning_water : ℕ := 4
def evening_water : ℕ := 6
def daily_water : ℕ := morning_water + evening_water
def total_water_needed : ℕ := 50

-- The proof statement
theorem sprinkler_days_needed : total_water_needed / daily_water = 5 := by
  sorry

end sprinkler_days_needed_l743_743276


namespace area_ratio_PQR_to_STU_l743_743965

-- Given Conditions
def triangle_PQR_sides (a b c : Nat) : Prop :=
  a = 9 ∧ b = 40 ∧ c = 41

def triangle_STU_sides (x y z : Nat) : Prop :=
  x = 7 ∧ y = 24 ∧ z = 25

-- Theorem Statement (math proof problem)
theorem area_ratio_PQR_to_STU :
  (∃ (a b c x y z : Nat), triangle_PQR_sides a b c ∧ triangle_STU_sides x y z) →
  9 * 40 / (7 * 24) = 15 / 7 :=
by
  intro h
  sorry

end area_ratio_PQR_to_STU_l743_743965


namespace tea_maker_capacity_l743_743208

theorem tea_maker_capacity (x : ℝ) (h : 0.45 * x = 54) : x = 120 :=
by
  sorry

end tea_maker_capacity_l743_743208


namespace quadratic_product_of_roots_l743_743822

theorem quadratic_product_of_roots :
  let a := 1
  let b := -1
  let c := -6
  (∃ x1 x2 : ℝ, x1 * x2 = c / a ∧ x1 ≠ x2 ∧ (λ x, a * x^2 + b * x + c = 0) x1 ∧ (λ x, a * x^2 + b * x + c = 0) x2) →
  (∃ x1 x2 : ℝ, x1 * x2 = -6) := 
by
  sorry

end quadratic_product_of_roots_l743_743822


namespace coffee_shop_lattes_l743_743934

theorem coffee_shop_lattes (T : ℕ) (L : ℕ) (hT : T = 6) (hL : L = 4 * T + 8) : L = 32 :=
by
  sorry

end coffee_shop_lattes_l743_743934


namespace keiko_speed_proof_l743_743873

noncomputable def keiko_speed (a b : ℝ) (h : (2 * a + 2 * Real.pi * (b + 8)) / s = (2 * a + 2 * Real.pi * b) / s + 48) : ℝ :=
let L_outer := 2 * a + 2 * Real.pi * (b + 8),
    L_inner := 2 * a + 2 * Real.pi * b in
  have h : L_outer / s = L_inner / s + 48, from h,
  s = Real.pi / 3

theorem keiko_speed_proof (a b s : ℝ) 
  (h : (2 * a + 2 * Real.pi * (b + 8)) / s = (2 * a + 2 * Real.pi * b) / s + 48) :
  s = Real.pi / 3 :=
sorry

end keiko_speed_proof_l743_743873


namespace power_function_value_l743_743804

theorem power_function_value :
  ∃ a : ℝ, (∀ x : ℝ, f x = x ^ a) ∧ f (1/2) = (sqrt 2) / 2 → f (1/4) = 1/2 :=
by
  exists a
  intro ha
  sorry

end power_function_value_l743_743804


namespace TripleApplicationOfF_l743_743319

def f (N : ℝ) : ℝ := 0.7 * N + 2

theorem TripleApplicationOfF :
  f (f (f 40)) = 18.1 :=
  sorry

end TripleApplicationOfF_l743_743319


namespace smallest_n_l743_743145

def f (n : ℕ) : ℕ :=
  Nat.find (λ k, Nat.factorial k % n = 0)

theorem smallest_n (n : ℕ) (h_n_multiple_of_18 : ∃ k, n = 18 * k) (h_f_n_gt_18 : f n > 18) : n = 342 :=
by
  sorry

end smallest_n_l743_743145


namespace maximum_correct_answers_l743_743452

theorem maximum_correct_answers (c w u : ℕ) :
  c + w + u = 25 →
  4 * c - w = 70 →
  c ≤ 19 :=
by
  sorry

end maximum_correct_answers_l743_743452


namespace a1_and_a2_values_Sn_formula_l743_743314

noncomputable def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range(n), a (i + 1)

def satisfies_root_condition (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) : Prop :=
∃ x : ℝ, x = S n - 1 ∧ x^2 - a n * x - a n = 0

theorem a1_and_a2_values (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : satisfies_root_condition a S 1)
  (h2 : satisfies_root_condition a S 2) :
  a 1 = 1/2 ∧ a 2 = 1/6 :=
sorry

theorem Sn_formula (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h : ∀ (n : ℕ), satisfies_root_condition a S n)
  (h_init : S 1 = a 1)
  (h_rec : ∀ (n : ℕ), a (n + 1) = S (n + 1) - S n) :
  ∀ (n : ℕ), S n = n / (n + 1) :=
sorry

end a1_and_a2_values_Sn_formula_l743_743314


namespace plant_arrangement_l743_743866

theorem plant_arrangement (ferns := 3) (rubber_plant := 1) (blue_lamps := 3) (yellow_lamps := 2) :
  (number_of_ways : Nat) (number_of_ways = 5) :=
by
  sorry

end plant_arrangement_l743_743866


namespace magazine_subscription_distributions_l743_743956

-- Define the problem conditions
def bookstores := Fin 4
def minSubscriptions := 98
def maxSubscriptions := 101
def totalSubscriptions := 400

-- The main proof statement
theorem magazine_subscription_distributions 
  (h1 : ∀ (b : bookstores), minSubscriptions ≤ subscriptions b ∧ subscriptions b ≤ maxSubscriptions)
  (h2 : ∑ (b : bookstores), subscriptions b = totalSubscriptions) :
  finset.card (subscription_distributions totalSubscriptions minSubscriptions maxSubscriptions) = 31 :=
by
  sorry

end magazine_subscription_distributions_l743_743956


namespace sum_CAMSA_PASTA_l743_743157

-- Define the multisets for the words "САМСА" and "ПАСТА"
def letters_CAMSA := {'С':=2, 'А':=2, 'М':=1}
def letters_PASTA := {'П':=1, 'А':=2, 'С':=1, 'Т':=1}

-- Define the factorial function
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- Define the number of distinct permutations for a multiset
def num_permutations (letters : List (Char × Nat)) : Nat :=
  let n := (letters.map Prod.snd).sum
  let denominator := (letters.map (λ x => factorial x.snd)).prod
  factorial n / denominator

-- Definitions specific to the conditions
def num_CAMSA : Nat := num_permutations letters_CAMSA.toList
def num_PASTA : Nat := num_permutations letters_PASTA.toList

-- The theorem to be proven
theorem sum_CAMSA_PASTA : num_CAMSA + num_PASTA = 90 := by
  sorry

end sum_CAMSA_PASTA_l743_743157


namespace reciprocal_of_repeating_decimal_equiv_l743_743981

noncomputable def repeating_decimal (x : ℝ) := 0.333333...

theorem reciprocal_of_repeating_decimal_equiv :
  (1 / repeating_decimal 0.333333...) = 3 :=
sorry

end reciprocal_of_repeating_decimal_equiv_l743_743981


namespace problem1_problem2_l743_743258

-- Problem (1)
theorem problem1 : |-2| - 2 * Real.sin (Float.pi / 6) + 2023^0 = 2 := 
by 
  sorry

-- Problem (2)
theorem problem2 (x : ℝ) : (3 * x - 1 > -7) ∧ (2 * x < x + 2) ↔ -2 < x ∧ x < 2 := 
by 
  intro h
  exact sorry

end problem1_problem2_l743_743258


namespace raman_salary_loss_l743_743536

theorem raman_salary_loss : 
  ∀ (S : ℝ), S > 0 →
  let decreased_salary := S - (0.5 * S) 
  let final_salary := decreased_salary + (0.5 * decreased_salary) 
  let loss := S - final_salary 
  let percentage_loss := (loss / S) * 100
  percentage_loss = 25 := 
by
  intros S hS
  let decreased_salary := S - (0.5 * S)
  let final_salary := decreased_salary + (0.5 * decreased_salary)
  let loss := S - final_salary
  let percentage_loss := (loss / S) * 100
  have h1 : decreased_salary = 0.5 * S := by sorry
  have h2 : final_salary = 0.75 * S := by sorry
  have h3 : loss = 0.25 * S := by sorry
  have h4 : percentage_loss = 25 := by sorry
  exact h4

end raman_salary_loss_l743_743536


namespace milk_water_ratio_l743_743970

theorem milk_water_ratio
  (vessel1_milk_ratio : ℚ)
  (vessel1_water_ratio : ℚ)
  (vessel2_milk_ratio : ℚ)
  (vessel2_water_ratio : ℚ)
  (equal_mixture_units  : ℚ)
  (h1 : vessel1_milk_ratio / vessel1_water_ratio = 4 / 1)
  (h2 : vessel2_milk_ratio / vessel2_water_ratio = 7 / 3)
  :
  (vessel1_milk_ratio + vessel2_milk_ratio) / 
  (vessel1_water_ratio + vessel2_water_ratio) = 11 / 4 :=
by
  sorry

end milk_water_ratio_l743_743970


namespace solution_set_g_lt_l743_743320

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

def is_even (f : ℝ → ℝ) := ∀ x, f(-x) = f(x)

axiom f_is_even : is_even f

axiom derivative_condition : ∀ x : ℝ, 0 ≤ x → (x / 2) * f'(x) + f(-x) < 0

def g (x : ℝ) : ℝ := x^2 * f(x)

theorem solution_set_g_lt : { x : ℝ | g(x) < g(1 - 2*x) } = { x : ℝ | 1/3 < x ∧ x < 1 } := sorry

end solution_set_g_lt_l743_743320


namespace find_a_l743_743496

theorem find_a (a : ℤ) (h1 : 0 < a) (h2 : a < 13) 
    (h3 : 13 ∣ 53^2016 + a) : a = 12 := 
by 
  -- proof would be written here
  sorry

end find_a_l743_743496


namespace radius_large_ball_l743_743952

noncomputable theory
open Real

-- Define the volume of a sphere
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * π * r^3

-- Given conditions
def radius_small_ball : ℝ := 2
def volume_small_ball := volume_of_sphere radius_small_ball
def total_volume_small_balls : ℝ := 10 * volume_small_ball

-- Problem statement to prove
theorem radius_large_ball : ∃ r : ℝ, volume_of_sphere r = total_volume_small_balls ∧ r = (240)^(1/3 : ℝ) :=
by
  use (240)^(1/3 : ℝ)
  split
  {
    calc
      volume_of_sphere ((240)^(1/3 : ℝ))
        = (4 / 3) * π * ((240)^(1/3 : ℝ))^3 : by rw volume_of_sphere
    ... = (4 / 3) * π * 240 : by rw Real.rpow_nat_cast; norm_num
    ... = total_volume_small_balls : by norm_num
  }
  {
    simp
  }
  sorry

end radius_large_ball_l743_743952


namespace coffee_shop_sold_lattes_l743_743933

theorem coffee_shop_sold_lattes (T L : ℕ) (h1 : T = 6) (h2 : L = 4 * T + 8) : L = 32 :=
by
  sorry

end coffee_shop_sold_lattes_l743_743933


namespace fruiting_plants_given_away_l743_743191

noncomputable def roxy_fruiting_plants_given_away 
  (N_f : ℕ) -- initial flowering plants
  (N_ft : ℕ) -- initial fruiting plants
  (N_bsf : ℕ) -- flowering plants bought on Saturday
  (N_bst : ℕ) -- fruiting plants bought on Saturday
  (N_gsf : ℕ) -- flowering plant given away on Sunday
  (N_total_remaining : ℕ) -- total plants remaining 
  (H₁ : N_ft = 2 * N_f) -- twice as many fruiting plants
  (H₂ : N_total_remaining = (N_f + N_bsf - N_gsf) + (N_ft + N_bst - N_gst)) -- total plants equation
  : ℕ :=
  4

theorem fruiting_plants_given_away (N_f : ℕ) (N_ft : ℕ) (N_bsf : ℕ) (N_bst : ℕ) (N_gsf : ℕ) (N_total_remaining : ℕ)
  (H₁ : N_ft = 2 * N_f) (H₂ : N_total_remaining = (N_f + N_bsf - N_gsf) + (N_ft + N_bst - N_gst)) : N_ft - (N_total_remaining - (N_f + N_bsf - N_gsf)) = 4 := 
by
  sorry

end fruiting_plants_given_away_l743_743191


namespace solve_for_x_l743_743746

theorem solve_for_x (x : ℝ) (h : sqrt (4 - 2 * x) = 5) : x = 21 / 2 :=
sorry

end solve_for_x_l743_743746


namespace probability_at_least_two_same_l743_743176

theorem probability_at_least_two_same (n : ℕ) (s : ℕ) (h_n : n = 8) (h_s : s = 8) :
  let total_outcomes := s ^ n
      different_outcomes := Nat.factorial s
      prob_all_different := different_outcomes / total_outcomes
      prob_at_least_two_same := 1 - prob_all_different
  in prob_at_least_two_same = 1291 / 1296 :=
by
  -- Define values
  have h_total_outcomes : total_outcomes = 16777216 := by sorry
  have h_different_outcomes : different_outcomes = 40320 := by sorry
  have h_prob_all_different : prob_all_different = 5 / 1296 := by sorry
  -- Calculate probability of at least two dice showing the same number
  have h_prob_at_least_two_same : prob_at_least_two_same = 1 - (5 / 1296) := by
    unfold prob_at_least_two_same prob_all_different
    rw h_different_outcomes
    rw h_total_outcomes
    rw h_prob_all_different
  -- Simplify
  calc
    prob_at_least_two_same = 1 - (5 / 1296) : by rw h_prob_at_least_two_same
    ... = 1291 / 1296 : by sorry

end probability_at_least_two_same_l743_743176


namespace watch_loss_percentage_l743_743298

noncomputable def initial_loss_percentage : ℝ :=
  let CP := 350
  let SP_new := 364
  let delta_SP := 140
  show ℝ from 
  sorry

theorem watch_loss_percentage (CP SP_new delta_SP : ℝ) (h₁ : CP = 350)
  (h₂ : SP_new = 364) (h₃ : delta_SP = 140) : 
  initial_loss_percentage = 36 :=
by
  -- Use the hypothesis and solve the corresponding problem
  sorry

end watch_loss_percentage_l743_743298


namespace choose_elements_l743_743489

variables {S : Type*} [Fintype S] {n k : ℕ}
variables (S_i : Fin (k * n) → Finset S)
variables (hS_size : Fintype.card S = n)
variables (h_S_i_size : ∀ i, (S_i i).card = 2)
variables (h_element_counts : ∀ e : S, (Finset.univ.filter (λ i, e ∈ S_i i)).card = 2 * k)

theorem choose_elements (S : Finset S) (k : ℕ) (S_i : ∀ i : Fin (k * n), Finset S)
  (hS_size : S.card = n)
  (h_S_i_size : ∀ i, (S_i i).card = 2)
  (h_element_counts : ∀ e : S, (Finset.univ.filter (λ i, e ∈ S_i i)).card = 2 * k) :
  ∃ (choice : ∀ i, S_i i → S), 
    (∀ e : S, (Finset.univ.filter (λ i, choice i ∈ S_i i)).card = k) := sorry

end choose_elements_l743_743489


namespace sum_of_all_possible_N_l743_743950

theorem sum_of_all_possible_N
  (a b c : ℕ)
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : c = a + b)
  (h3 : N = a * b * c)
  (h4 : N = 6 * (a + b + c)) :
  N = 156 ∨ N = 96 ∨ N = 84 ∧
  (156 + 96 + 84 = 336) :=
by {
  -- proof will go here
  sorry
}

end sum_of_all_possible_N_l743_743950


namespace max_value_f_at_e_l743_743941

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_f_at_e (h : 0 < x) : 
  ∃ e : ℝ, (∀ x : ℝ, 0 < x → f x ≤ f e) ∧ e = Real.exp 1 :=
by
  sorry

end max_value_f_at_e_l743_743941


namespace distinct_words_sum_l743_743159

theorem distinct_words_sum (n : ℕ) (n1 n2 : ℕ) :
  (n = 5) →
  (n1 = 2) →
  (n2 = 2) →
  (∃ words_САМСА : ℕ, words_САМСА = Nat.factorial n / (Nat.factorial n1 * Nat.factorial n2) ∧ words_САМСА = 30) →
  (∃ words_ПАСТА : ℕ, words_ПАСТА = Nat.factorial n / Nat.factorial n1 ∧ words_ПАСТА = 60) →
  90 = 30 + 60 :=
by
  intros h_n h_n1 h_n2 h_words_САМСА h_words_ПАСТА
  obtain ⟨words_САМСА, h1, h2⟩ := h_words_САМСА
  obtain ⟨words_ПАСТА, h3, h4⟩ := h_words_ПАСТА
  rw [h2, h4]
  exact Nat.add_comm 30 60

end distinct_words_sum_l743_743159


namespace log_expression_l743_743699

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_expression : 
  log 2 * log 50 + log 25 - log 5 * log 20 = 1 := 
by 
  sorry

end log_expression_l743_743699


namespace num_paths_A_to_B_l743_743004

def labeled_points : List String := ["A", "B", "C", "D", "E", "F", "G"]

def is_segment (a b : String) : Bool :=
  (a, b) ∈ [("A", "C"), ("A", "D"), ("A", "G"),
            ("B", "C"),
            ("C", "B"), ("C", "D"), ("C", "F"),
            ("D", "C"), ("D", "E"), ("D", "F"), ("D", "A"), ("D", "G"),
            ("E", "D"), ("E", "F"),
            ("F", "C"), ("F", "E"), ("F", "D"), ("F", "B"),
            ("G", "A"), ("G", "D"), ("G", "F")] ∨
            (b, a) ∈ [("A", "C"), ("A", "D"), ("A", "G"),
                      ("B", "C"),
                      ("C", "B"), ("C", "D"), ("C", "F"),
                      ("D", "C"), ("D", "E"), ("D", "F"), ("D", "A"), ("D", "G"),
                      ("E", "D"), ("E", "F"),
                      ("F", "C"), ("F", "E"), ("F", "D"), ("F", "B"),
                      ("G", "A"), ("G", "D"), ("G", "F")]

def paths (start end : String) : List (List String) :=
  if start = end then [[start]]
  else List.bind (labeled_points.filter (fun p => is_segment start p ∧ p ≠ start)) (fun p =>
    (paths p end).map (fun trail => start :: trail))

theorem num_paths_A_to_B : (paths "A" "B").length = 13 := by
  sorry

end num_paths_A_to_B_l743_743004


namespace isosceles_triangle_of_parallel_PQ_AC_l743_743140

variables {α β γ : Type}
variables {A B C I M N P Q : α} {circleIncircle circCircumcircleAIC : set α}
variables {lineBI linePQ lineAC : set α}

-- Given conditions from the problem
def incenter_of_triangle (I A B C : α) : Prop := sorry
def incircle (circle α : set α) (A B C I : α) : Prop := sorry
def circumcircle (circle circumcircleAIC : set α) (A I C : α) : Prop := sorry
def midpoint_of_arc (M N : α) (circle α : set α) (A B C : α) : Prop := sorry
def parallel (linePQ lineAC : set α) : Prop := sorry
def lies_on_same_side_of_line (P A : α) (line lineBI : set α) : Prop := sorry
def lies_on_opposite_sides_of_line (Q C : α) (line lineBI : set α) : Prop := sorry

theorem isosceles_triangle_of_parallel_PQ_AC :
  incenter_of_triangle I A B C ∧
  incircle circleIncircle A B C I ∧
  circumcircle circCircumcircleAIC A I C ∧
  midpoint_of_arc M circleIncircle A C ∧
  midpoint_of_arc N circleIncircle B C ∧
  lies_on_same_side_of_line P A lineBI ∧
  lies_on_opposite_sides_of_line Q C lineBI ∧
  parallel linePQ lineAC →
  (triangle_isosceles A B C) := sorry

end isosceles_triangle_of_parallel_PQ_AC_l743_743140


namespace locus_of_Q_l743_743249

def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2/a^2 + y^2/b^2 = 1

def A_vertice (a b : ℝ) (x y : ℝ) : Prop :=
  (x = a ∧ y = 0) ∨ (x = -a ∧ y = 0)

def chord_parallel_y_axis (x : ℝ) : Prop :=
  -- Assuming chord's x coordinate is given
  True

def lines_intersect_at_Q (a b Qx Qy : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse a b x y ∧
  A_vertice a b x y ∧
  chord_parallel_y_axis x ∧
  (
    ( (Qy - y) / (Qx - (-a)) = (Qy - 0) / (Qx - a) ) ∨ -- A'P slope-comp
    ( (Qy - (-y)) / (Qx - a) = (Qy - 0) / (Qx - (-a)) ) -- AP' slope-comp
  )

theorem locus_of_Q (a b Qx Qy : ℝ) :
  (lines_intersect_at_Q a b Qx Qy) →
  (Qx^2 / a^2 - Qy^2 / b^2 = 1) := by
  sorry

end locus_of_Q_l743_743249


namespace sequence_count_l743_743315

theorem sequence_count :
  ∃ n : ℕ, 
  (∀ a : Fin 101 → ℤ, 
    a 1 = 0 ∧ 
    a 100 = 475 ∧ 
    (∀ k : ℕ, 1 ≤ k ∧ k < 100 → |a (k + 1) - a k| = 5) → 
    n = 4851) := 
sorry

end sequence_count_l743_743315


namespace problem_3034_1002_20_04_div_sub_l743_743264

theorem problem_3034_1002_20_04_div_sub:
  3034 - (1002 / 20.04) = 2984 :=
by
  sorry

end problem_3034_1002_20_04_div_sub_l743_743264


namespace isosceles_triangle_l743_743116

theorem isosceles_triangle (ABC : Type) [triangle ABC]
  (I : incenter ABC) (α : incircle ABC)
  (P Q : α) (circAIC : circumcircle AIC)
  (h1 : P ∈ circAIC) (h2 : Q ∈ circAIC)
  (h3 : same_side P A (BI line))
  (h4 : other_side Q C (BI line))
  (M : midpoint_arc AC (α arc))
  (N : midpoint_arc BC (α arc))
  (h_par : PQ ∥ AC) : is_isosceles ABC := 
sorry

end isosceles_triangle_l743_743116


namespace monic_poly_irreducible_l743_743641

noncomputable def is_square_free (n : ℤ) : Prop := 
  ∀ m : ℤ, m * m ∣ n → abs m = 1

theorem monic_poly_irreducible
  (f : Polynomial ℤ) 
  (hf_deg : f.degree = 2)
  (hf_monic : f.monic) 
  (hf_rootless : ∀ x : ℝ, ¬ (x ∈ f.real_roots))
  (hf_square_free : is_square_free (f.eval 0))
  (hf_non_trivial : f.eval 0 ≠ 1 ∧ f.eval 0 ≠ -1) :
  ∀ n : ℕ, Irreducible (f.comp (X ^ n)) :=
sorry

end monic_poly_irreducible_l743_743641


namespace poem_lines_months_l743_743939

theorem poem_lines_months (current_lines : ℕ) (target_lines : ℕ) (lines_per_month : ℕ) :
  current_lines = 24 →
  target_lines = 90 →
  lines_per_month = 3 →
  (target_lines - current_lines) / lines_per_month = 22 :=
  by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  exact sorry

end poem_lines_months_l743_743939


namespace average_GPA_of_whole_class_l743_743581

variable (n : ℕ)

def GPA_first_group : ℕ := 54 * (n / 3)
def GPA_second_group : ℕ := 45 * (2 * n / 3)
def total_GPA : ℕ := GPA_first_group n + GPA_second_group n

theorem average_GPA_of_whole_class : total_GPA n / n = 48 := by
  sorry

end average_GPA_of_whole_class_l743_743581


namespace triangle_ABC_isosceles_l743_743100

-- Lean 4 definitions for the generated equivalent proof problem
noncomputable def incenter (A B C I : Type*) := sorry
noncomputable def incircle (ABC : Type*) (α : Type*) := sorry
noncomputable def circumcircle (A I C : Type*) := sorry
noncomputable def intersects (circle1 circle2 : Type*) (P Q : Type*) := sorry
noncomputable def same_side_line (P A : Type*) (line : Type*) := sorry
noncomputable def other_side_line (Q C : Type*) (line : Type*) := sorry
noncomputable def midpoint_arc (arc : Type*) := sorry
noncomputable def parallel (line1 line2 : Type*) := sorry
noncomputable def triangle_isosceles (A B C : Type*) := ∃ (ABC_is_isosceles : Prop), ABC_is_isosceles

-- Given conditions for the incenter, incircle, and parallel lines, we must prove the triangle is isosceles
theorem triangle_ABC_isosceles (A B C I α P Q M N : Type*)
  (h1 : incenter A B C I)
  (h2 : incircle ABC α)
  (h3 : circumcircle A I C)
  (h4 : intersects α (circumcircle A I C) P Q)
  (h5 : same_side_line P A (set I B))
  (h6 : other_side_line Q C (set I B))
  (h7 : midpoint_arc α AC M)
  (h8 : midpoint_arc α BC N)
  (h9 : parallel PQ AC) :
  triangle_isosceles A B C := sorry

end triangle_ABC_isosceles_l743_743100


namespace pentagons_cyclic_quadrilateral_exists_l743_743488

structure Pentagon (n : ℕ) :=
(vertices : ℕ → ℕ)
(adjacent : ℕ → ℕ)

axiom regular_pentagon :
  \{ n : ℕ // 2 ≤ n ∧ n ≤ 11 }

def is_colored (p : Pentagon) : Prop :=
  ∀ v, p.vertices v = 1 ∨ p.vertices v = 0

def same_color (p : Pentagon) (i j k l : ℕ) : Prop :=
  p.vertices i = p.vertices j ∧ p.vertices i = p.vertices k ∧ p.vertices i = p.vertices l

def cyclic_quadrilateral (p : Pentagon) (i j k l : ℕ) : Prop :=
  sorry

noncomputable def exists_cyclic_quadrilateral_same_color (p : Pentagon) : Prop :=
  ∃ i j k l, same_color p i j k l ∧ cyclic_quadrilateral p i j k l

theorem pentagons_cyclic_quadrilateral_exists :
    ∀ (pents : ℕ → Pentagon) (h : ∀ n, pents n ∈ regular_pentagon),
    (∀ n, is_colored (pents n)) →
    ∃ p, exists_cyclic_quadrilateral_same_color p :=
begin
  sorry
end

end pentagons_cyclic_quadrilateral_exists_l743_743488


namespace curve_in_hemisphere_l743_743617

noncomputable def ball (r : ℝ) (center : ℝ × ℝ × ℝ) := 
  {p : ℝ × ℝ × ℝ | (p - center).length < r}

def boundary (r : ℝ) (center : ℝ × ℝ × ℝ) := 
  {p : ℝ × ℝ × ℝ | (p - center).length = r}

def curve (γ : Set (ℝ × ℝ × ℝ)) := ∃ f : ℝ → (ℝ × ℝ × ℝ), ∀ t, γ = range f

theorem curve_in_hemisphere 
  (r : ℝ) 
  (A B : ℝ × ℝ × ℝ) 
  (h_radius : r = 1)
  (hA : A ∈ boundary r (0, 0, 0))
  (hB : B ∈ boundary r (0, 0, 0))
  (γ : Set (ℝ × ℝ × ℝ))
  (hγ_curve : curve γ)
  (hγ_ball : γ ⊆ ball r (0, 0, 0))
  (h_length : ∀ f : ℝ → (ℝ × ℝ × ℝ), γ = range f → ∫ t in 0..1, ∥f' t∥ dt < 2) :
  ∃ hemisphere : Set (ℝ × ℝ × ℝ), ∃ Q : ℝ × ℝ × ℝ × ℝ, 
  (hemisphere = {p | ∃ x : ℝ, Q • p = x ≤ 0}) ∧ γ ⊆ hemisphere :=
by
  sorry

end curve_in_hemisphere_l743_743617


namespace minimum_degree_g_l743_743975

-- Define the polynomials
variables {R : Type*} [CommRing R]
variable (f g h : Polynomial R)

-- Define the conditions
def degree_f (f : Polynomial R) : Prop := degree f = 7
def degree_h (h : Polynomial R) : Prop := degree h = 10
def equation (f g h : Polynomial R) : Prop := 2 * f + 5 * g = h

-- Minimum possible degree requirement
theorem minimum_degree_g (f g h : Polynomial R) (hf : degree_f f) (hh : degree_h h) (eq : equation f g h) :
  degree g >= 10 :=
sorry

end minimum_degree_g_l743_743975


namespace alexs_amount_l743_743193

variable total_amount : ℝ := 972.45
variable sams_amount : ℝ := 325.67
variable ericas_amount : ℝ := 214.29

theorem alexs_amount : total_amount - (sams_amount + ericas_amount) = 432.49 :=
by
  -- provide proof here
  sorry

end alexs_amount_l743_743193


namespace sum_of_tip_angles_is_720_degrees_l743_743900

-- Definitions based on conditions
def num_points : ℕ := 9
def total_circle_degrees : ℝ := 360
def arcs_per_tip : ℕ := 4

-- Based on the described problem, each arc between points is 40 degrees
def angle_per_arc : ℝ := total_circle_degrees / num_points
def degrees_cut_by_arcs (arcs : ℕ) : ℝ := arcs * angle_per_arc

-- Each tip cuts off 4 arcs, and each tip angle is half the degrees cut by those arcs
def single_tip_angle : ℝ := degrees_cut_by_arcs arcs_per_tip / 2

-- Total angle measurement for all tips
def total_tip_angle (tips : ℕ) : ℝ := tips * single_tip_angle

theorem sum_of_tip_angles_is_720_degrees :
  total_tip_angle num_points = 720 :=
sorry

end sum_of_tip_angles_is_720_degrees_l743_743900


namespace eval_expr_eq_condition_l743_743645

-- 1. Expression evaluation problem
theorem eval_expr : -1 + 2 * 3^4 + 5 = 166 := 
by {
  calc
  -1 + 2 * 3^4 + 5
      = -1 + 2 * 81 + 5 : by rw [pow_succ, pow_one] 
  ... = -1 + 162 + 5 : by norm_num
  ... = 166 : by norm_num
}

-- 2. Factorial calculation definition
def factorial : ℕ → ℕ 
| 0     := 1
| (n+1) := (n + 1) * factorial n

-- Please note that in Lean, we will use the factorial definition provided by Mathlib
-- and you can directly prove certain properties about it if required.

-- 3. Dangling else problem
def cond1 : Prop := sorry  -- Placeholder for a condition
def cond2 : Prop := sorry  -- Placeholder for another condition

def bloc1 : ℕ := sorry  -- Placeholder for block 1
def bloc2 : ℕ := sorry  -- Placeholder for block 2

def dangling_else (h1 : cond1) (h2 : cond2) : ℕ :=
  if h1 then (if h2 then bloc1 else bloc2) else bloc2

-- 4. Equality condition check
theorem eq_condition : (if 3 = 4 then 1 else 0) = 0 := by simp

end eval_expr_eq_condition_l743_743645


namespace minimum_value_when_a_is_one_range_for_two_zeros_l743_743802

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x - Real.log x

theorem minimum_value_when_a_is_one : 
  ∃ x, f 1 x = 0 :=
begin
  use 1,
  -- rest of the proof omitted
  sorry
end

theorem range_for_two_zeros : 
  ∀ a : ℝ, (0 < a ∧ a < 1) → 
              ∃ (x1 x2 : ℝ), (0 < x1 ∧ 0 < x2) ∧ 
                             f a x1 = 0 ∧ 
                             f a x2 = 0 ∧ 
                             x1 ≠ x2 :=
begin
  -- rest of the proof omitted
  sorry
end

end minimum_value_when_a_is_one_range_for_two_zeros_l743_743802


namespace initial_birds_correct_l743_743227

def flown_away : ℝ := 8.0
def left_on_fence : ℝ := 4.0
def initial_birds : ℝ := flown_away + left_on_fence

theorem initial_birds_correct : initial_birds = 12.0 := by
  sorry

end initial_birds_correct_l743_743227


namespace max_t_subsets_of_base_set_l743_743484

theorem max_t_subsets_of_base_set (n : ℕ)
  (A : Fin (2 * n + 1) → Set (Fin n))
  (h : ∀ i j k : Fin (2 * n + 1), i < j → j < k → (A i ∩ A k) ⊆ A j) : 
  ∃ t : ℕ, t = 2 * n + 1 :=
by
  sorry

end max_t_subsets_of_base_set_l743_743484


namespace problem1_problem2_l743_743395

noncomputable def f (a : ℝ) (x : ℝ) := Real.log x / Real.log a
noncomputable def g (a : ℝ) (t : ℝ) (x : ℝ) := Real.log (2 * x + t - 2) / Real.log a

-- Problem 1: Prove the range of t
theorem problem1 {a : ℝ} (h_a : 0 < a ∧ a < 1) :
  ∀ x ∈ Set.Icc (1 / 4) 2, ∀ t ∈ Set.Ici 2, 2 * f a x ≥ g a t x :=
sorry

-- Problem 2: Prove the value of a
theorem problem2 :
  ∀ a : ℝ, ∀ x ∈ Set.Icc (1 / 4) 2, let F (a : ℝ) (x : ℝ) := 2 * g a 4 x - f a x 
  in (∀ x, F a x ≥ -2) -> (a = 1/5) :=
sorry

end problem1_problem2_l743_743395


namespace corridor_cover_l743_743201

theorem corridor_cover (segments : list (ℝ × ℝ)) (h_cover : ∀ x ∈ Icc (0 : ℝ) 1, ∃ a b, (a, b) ∈ segments ∧ x ∈ Icc a b) :
  ∃ remaining_segments : list (ℝ × ℝ),
    (∀ x ∈ Icc (0 : ℝ) 1, ∃ a b, (a, b) ∈ remaining_segments ∧ x ∈ Icc a b) ∧
    ∑ (a, b) in remaining_segments, (b - a) ≤ 2 :=
  sorry

end corridor_cover_l743_743201


namespace probability_two_dice_same_number_l743_743180

theorem probability_two_dice_same_number : 
  let dice_sides := 8 in
  let total_outcomes := dice_sides ^ 8 in
  let different_outcomes := (fact dice_sides) / (fact (dice_sides - 8)) in
  (1 - (different_outcomes / total_outcomes)) = (1291 / 1296) :=
by
  sorry

end probability_two_dice_same_number_l743_743180


namespace verify_solution_l743_743607

variable (x y : ℝ)

-- Conditions
def condition1 : Prop := x - y = 9
def condition2 : Prop := 4 * x + 3 * y = 1

-- Proof problem statement
theorem verify_solution
  (h1 : condition1 x y)
  (h2 : condition2 x y) :
  x = 4 ∧ y = -5 :=
sorry

end verify_solution_l743_743607


namespace isosceles_triangle_of_parallel_PQ_AC_l743_743139

variables {α β γ : Type}
variables {A B C I M N P Q : α} {circleIncircle circCircumcircleAIC : set α}
variables {lineBI linePQ lineAC : set α}

-- Given conditions from the problem
def incenter_of_triangle (I A B C : α) : Prop := sorry
def incircle (circle α : set α) (A B C I : α) : Prop := sorry
def circumcircle (circle circumcircleAIC : set α) (A I C : α) : Prop := sorry
def midpoint_of_arc (M N : α) (circle α : set α) (A B C : α) : Prop := sorry
def parallel (linePQ lineAC : set α) : Prop := sorry
def lies_on_same_side_of_line (P A : α) (line lineBI : set α) : Prop := sorry
def lies_on_opposite_sides_of_line (Q C : α) (line lineBI : set α) : Prop := sorry

theorem isosceles_triangle_of_parallel_PQ_AC :
  incenter_of_triangle I A B C ∧
  incircle circleIncircle A B C I ∧
  circumcircle circCircumcircleAIC A I C ∧
  midpoint_of_arc M circleIncircle A C ∧
  midpoint_of_arc N circleIncircle B C ∧
  lies_on_same_side_of_line P A lineBI ∧
  lies_on_opposite_sides_of_line Q C lineBI ∧
  parallel linePQ lineAC →
  (triangle_isosceles A B C) := sorry

end isosceles_triangle_of_parallel_PQ_AC_l743_743139


namespace proof_l743_743882

-- Define the conditions in Lean
variable {f : ℝ → ℝ}
variable (h1 : ∀ x ∈ (Set.Ioi 0), 0 ≤ f x)
variable (h2 : ∀ x ∈ (Set.Ioi 0), x * f x + f x ≤ 0)

-- Formulate the goal
theorem proof (a b : ℝ) (ha : a ∈ (Set.Ioi 0)) (hb : b ∈ (Set.Ioi 0)) (h : a < b) : 
    b * f a ≤ a * f b :=
by
  sorry  -- Proof omitted

end proof_l743_743882


namespace question_2024_polynomials_l743_743144

open Polynomial

noncomputable def P (x : ℝ) : Polynomial ℝ := sorry
noncomputable def Q (x : ℝ) : Polynomial ℝ := sorry

-- Main statement
theorem question_2024_polynomials (P Q : Polynomial ℝ) (hP : P.degree = 2024) (hQ : Q.degree = 2024)
    (hPm : P.leadingCoeff = 1) (hQm : Q.leadingCoeff = 1) (h : ∀ x : ℝ, P.eval x ≠ Q.eval x) :
    ∀ (α : ℝ), α ≠ 0 → ∃ x : ℝ, P.eval (x - α) = Q.eval (x + α) :=
by
  sorry

end question_2024_polynomials_l743_743144


namespace exists_k_inequality_l743_743876

theorem exists_k_inequality (n : ℕ) (x : Fin n → ℝ) (h_prod : (∏ i, x i) = 1) (h_pos : ∀ i, 0 < x i) :
  ∃ k : Fin n, x k / (↑k + 1 + ∑ i in Finset.range (↑k + 1), x ⟨i, Fin.is_lt i⟩) ≥ 1 - 1 / 2 ^ (1 / (n : ℝ)) :=
by
  sorry

end exists_k_inequality_l743_743876


namespace negation_of_proposition_l743_743217

open Classical

theorem negation_of_proposition :
  (∃ x : ℝ, x^2 + 2 * x + 5 ≤ 0) ↔ ¬(∀ x : ℝ, x^2 + 2 * x + 5 > 0) := by
  sorry

end negation_of_proposition_l743_743217


namespace sweet_potatoes_not_yet_sold_l743_743896

def total_harvested := 80
def sold_to_adams := 20
def sold_to_lenon := 15
def not_yet_sold : ℕ := total_harvested - (sold_to_adams + sold_to_lenon)

theorem sweet_potatoes_not_yet_sold :
  not_yet_sold = 45 :=
by
  unfold not_yet_sold
  unfold total_harvested sold_to_adams sold_to_lenon
  sorry

end sweet_potatoes_not_yet_sold_l743_743896


namespace distance_focus_to_asymptote_l743_743343

noncomputable def distance_from_focus_to_asymptote_parabola_hyperbola : ℝ :=
  sorry

theorem distance_focus_to_asymptote (d : ℝ) :
  let parabola := ∀ x y : ℝ, y^2 = 4 * x,
      hyperbola := ∀ x y : ℝ, x^2 - y^2 / 3 = 1 in
  d = distance_from_focus_to_asymptote_parabola_hyperbola → 
  d = sqrt 3 / 2 :=
begin
  intro h,
  rw h,
  sorry
end

end distance_focus_to_asymptote_l743_743343


namespace BoxC_in_BoxA_l743_743310

-- Define the relationship between the boxes
def BoxA_has_BoxB (A B : ℕ) : Prop := A = 4 * B
def BoxB_has_BoxC (B C : ℕ) : Prop := B = 6 * C

-- Define the proof problem
theorem BoxC_in_BoxA {A B C : ℕ} (h1 : BoxA_has_BoxB A B) (h2 : BoxB_has_BoxC B C) : A = 24 * C :=
by
  sorry

end BoxC_in_BoxA_l743_743310


namespace calc_f_5_l743_743013

def f (x : ℕ) : ℕ := 
  if x ≥ 10 then x - 2 
  else f (f (x + 6))

theorem calc_f_5 : f 5 = 11 := by
  sorry

end calc_f_5_l743_743013


namespace gambler_A_shares_event_A_rare_l743_743050

-- Definitions
def a : ℕ := 243
def k : ℕ := 4
def m : ℕ := 2
def n : ℕ := 1
def p : ℚ := 2 / 3

-- Function to compute the probability of Gambler A winning all remaining rounds
noncomputable def prob_A_wins_all : ℚ :=
  let X2 := p^2 in
  let X3 := (2.choose 1) * p^2 * (1 - p) in
  let X4 := (3.choose 1) * p^2 * (1 - p)^2 in
  X2 + X3 + X4

-- Lean statement for Part 1
theorem gambler_A_shares : prob_A_wins_all * a = 216 := sorry

-- Function to compute the probability of Gambler B winning all remaining rounds (event A)
noncomputable def prob_B_wins_all (p : ℚ) : ℚ :=
  let Y3 := (1 - p)^3 in
  let Y4 := (3.choose 1) * p * (1 - p)^3 in
  Y3 + Y4

-- Function to compute the probability of Gambler A winning all rounds
noncomputable def f (p : ℚ) : ℚ :=
  1 - prob_B_wins_all p

-- Lean statement for Part 2
theorem event_A_rare (p : ℚ) (hp : p ≥ 4 / 5) : f(p) > 0.95 :=
  by
    -- proof would go here
    sorry

end gambler_A_shares_event_A_rare_l743_743050


namespace sum_divisors_154_l743_743626

theorem sum_divisors_154 : (∑ d in (finset.filter (λ n, 154 % n = 0) (finset.range 155)), d) = 288 :=
by
  sorry

end sum_divisors_154_l743_743626


namespace find_x_l743_743823

theorem find_x (x y : ℝ) (h1 : y = 1 / (2 * x + 2)) (h2 : y = 2) : x = -3 / 4 :=
by
  sorry

end find_x_l743_743823


namespace ABC_is_isosceles_l743_743127

open Mobius

variables
  (ABC : Triangle)
  (I : Point)
  (α P Q M N : Point)
  (A B C Z : Point)
  [Incenter I ABC]
  [Incircle α ABC]
  [Circumcircle (Triangle.mk A I C)]
  [PQ_parallel_AC : is_parallel (Segment.mk P Q) (Segment.mk A C)]
  [Midpoint M (Arc.mk A C α)]
  [Midpoint N (Arc.mk B C α)]
  [center_Z : Center Z (Circumcircle (Triangle.mk A I C))]
  [Chord_α_PQ : common_chord α (Circumcircle (Triangle.mk A I C)) (Segment.mk P Q)]
  [P_A_same_BI : same_side P A (Line.mk B I)]
  [Q_C_other_BI : same_side Q C (Line.mk B I)]

theorem ABC_is_isosceles
  (h_parallel : PQ_parallel_AC) : 
  Isosceles ABC := 
sorry

end ABC_is_isosceles_l743_743127


namespace compound_interest_six_years_l743_743685

theorem compound_interest_six_years :
  let P : ℝ := 1000
  let r1 : ℝ := 0.05
  let r2 : ℝ := 0.06
  let r3 : ℝ := 0.04
  let r4 : ℝ := r3 + 0.01
  let r5 : ℝ := r4 + 0.01
  let r6 : ℝ := r5 + 0.01
  let A1 : ℝ := P * (1 + r1)
  let A2 : ℝ := A1 * (1 + r2)
  let A3 : ℝ := A2 * (1 + r3)
  let A4 : ℝ := A3 * (1 + r4)
  let A5 : ℝ := A4 * (1 + r5)
  let A6 : ℝ := A5 * (1 + r6)
  in
  A6 ≈ 1378.50 ∧ r6 = 0.07 :=
by
  sorry

end compound_interest_six_years_l743_743685


namespace angles_of_obtuse_triangle_l743_743844

noncomputable def find_angles_of_triangle (A B C : ℝ) : Prop :=
  ∃ (A B C : ℝ), 
    ((A + B + C = 180) ∧ (A = 108) ∧ (B = 18) ∧ (C = 54))

theorem angles_of_obtuse_triangle :
  ∀ (A B C : ℝ) (D E F : ℝ), 
    (A > 90 ∧ (A + B + C = 180) ∧ 
     (D = A - 90) ∧ (E = 90 - B) ∧ (F = 90 - C) ∧ 
     (E - F = A - 90) ∧ (D + F = 2 * A - 180)) →
     find_angles_of_triangle A B C :=
begin
  intros,
  sorry
end

end angles_of_obtuse_triangle_l743_743844


namespace variance_data_correct_l743_743608

noncomputable def data : List ℝ := [10, 6, 8, 5, 6]

def mean (l : List ℝ) : ℝ := l.sum / l.length

def variance (l : List ℝ) : ℝ :=
  let μ := mean l
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

theorem variance_data_correct : variance data = 16 / 5 :=
  sorry

end variance_data_correct_l743_743608


namespace phase_shift_of_sine_l743_743326

noncomputable def phase_shift (A B C : ℝ) : ℝ :=
  C / B

theorem phase_shift_of_sine : phase_shift 5 3 (π / 3) = π / 9 :=
by
  unfold phase_shift
  rw [div_eq_mul_one_div]
  change (π / 3) * (1 / 3) = π / 9
  rw [← mul_div_assoc, div_self (3 : ℝ), mul_one]
  symmetry
  norm_num
  rw [pi_div_two_mul_two]
  unfold π
  norm_num
  sorry

end phase_shift_of_sine_l743_743326


namespace find_polynomials_l743_743376

noncomputable def satisfies_equation (P : ℝ → ℝ) (k : ℕ) : Prop :=
  ∀ x, P (P x) = (P x) ^ k

theorem find_polynomials (k : ℕ) (hk : 0 < k) :
  ∃ P : ℝ → ℝ, satisfies_equation P k ∧ 
    (P = λ x, x^k ∨ ∃ C : ℝ, P = λ x, C) :=
by 
  sorry

end find_polynomials_l743_743376


namespace ball_height_intersect_l743_743656

noncomputable def ball_height (h : ℝ) (t₁ t₂ : ℝ) (h₁ h₂ : ℝ → ℝ) : Prop :=
  (∀ t, h₁ t = h₂ (t - 1) ↔ t = t₁) ∧
  (h₁ t₁ = h ∧ h₂ t₁ = h) ∧ 
  (∀ t, h₂ (t - 1) = h₁ t) ∧ 
  (h₁ (1.1) = h ∧ h₂ (1.1) = h)

theorem ball_height_intersect (h : ℝ)
  (h₁ h₂ : ℝ → ℝ)
  (h_max : ∀ t₁ t₂, ball_height h t₁ t₂ h₁ h₂) :
  (∃ t₁, t₁ = 1.6) :=
sorry

end ball_height_intersect_l743_743656


namespace maze_traversal_impossible_l743_743055

theorem maze_traversal_impossible (E S : ℕ) (hE : E = 1) (hS : S = 36) :
  (∃ path : list ℕ, path.head = E ∧ path.reverse.head = S ∧ path.nodup ∧ path.length = 36) → False :=
by
  -- Conditions on the coloring
  have coloring_condition : ∀ n, n < 36 → (n / 6 + n % 6) % 2 = 0 → False := sorry
  -- Assumption that the path from E to S exists
  intro h
  -- Extract the path from the assumption
  cases h with path path_conditions
  -- Justification based on the coloring
  sorry

end maze_traversal_impossible_l743_743055


namespace three_digit_prime_last_digit_l743_743622

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem three_digit_prime_last_digit (n : ℕ) :
  100 ≤ n ∧ n < 1000 ∧ is_prime n ∧ 
  (λ A B C : ℕ, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ n = 100 * A + 10 * B + C ∧ C = A + B) → 
  ∃ A B C : ℕ, n = 100 * A + 10 * B + C ∧ C = 7 :=
sorry

end three_digit_prime_last_digit_l743_743622


namespace parabola_min_value_incorrect_statement_l743_743386

theorem parabola_min_value_incorrect_statement
  (m : ℝ)
  (A B : ℝ × ℝ)
  (P Q : ℝ × ℝ)
  (parabola : ℝ → ℝ)
  (on_parabola : ∀ (x : ℝ), parabola x = x^2 - 2*m*x + m^2 - 9)
  (A_intersects_x_axis : A.2 = 0)
  (B_intersects_x_axis : B.2 = 0)
  (A_on_parabola : parabola A.1 = A.2)
  (B_on_parabola : parabola B.1 = B.2)
  (P_on_parabola : parabola P.1 = P.2)
  (Q_on_parabola : parabola Q.1 = Q.2)
  (P_coordinates : P = (m + 1, parabola (m + 1)))
  (Q_coordinates : Q = (m - 3, parabola (m - 3))) :
  ∃ (min_y : ℝ), min_y = -9 ∧ min_y ≠ m^2 - 9 := 
sorry

end parabola_min_value_incorrect_statement_l743_743386


namespace no_solutions_for_equation_l743_743196

theorem no_solutions_for_equation : ∀ x : ℝ, cos (cos (cos (cos x))) ≠ sin (sin (sin (sin x))) :=
by
  assume x,
  sorry

end no_solutions_for_equation_l743_743196


namespace greatest_integer_x_l743_743977

theorem greatest_integer_x (x : ℤ) (h : 7 - 3 * x + 2 > 23) : x ≤ -5 :=
by {
  sorry
}

end greatest_integer_x_l743_743977


namespace identify_vanya_l743_743902

structure Twin :=
(name : String)
(truth_teller : Bool)

def is_vanya_truth_teller (twin : Twin) (vanya vitya : Twin) : Prop :=
  twin = vanya ∧ twin.truth_teller ∨ twin = vitya ∧ ¬twin.truth_teller

theorem identify_vanya
  (vanya vitya : Twin)
  (h_vanya : vanya.name = "Vanya")
  (h_vitya : vitya.name = "Vitya")
  (h_one_truth : ∃ t : Twin, t = vanya ∨ t = vitya ∧ (t.truth_teller = true ∨ t.truth_teller = false))
  (h_one_lie : ∀ t : Twin, t = vanya ∨ t = vitya → ¬(t.truth_teller = true ∧ t = vitya) ∧ ¬(t.truth_teller = false ∧ t = vanya)) :
  ∀ twin : Twin, twin = vanya ∨ twin = vitya →
  (is_vanya_truth_teller twin vanya vitya ↔ (twin = vanya ∧ twin.truth_teller = true)) :=
by
  sorry

end identify_vanya_l743_743902


namespace solve_equation_x_l743_743924

theorem solve_equation_x :
  ∃ x, 2^(8^x) = 8^(2^x) ↔ x = 2 / (Real.log 3 / Real.log 2 - 1) :=
by
  sorry

end solve_equation_x_l743_743924


namespace benches_needed_l743_743677

theorem benches_needed (r : ℝ) (space : ℝ) (π_approx : ℝ) : r = 15 → space = 3 → π_approx = 3.14159 → 
  (int.ofReal (30 * π / space)).nat_abs = 31 :=
by
  intro hr hspace hπ
  -- Definitions
  let r := 15
  let space := 3
  let π := 3.14159
  have hcircumference : ℝ := 30 * π
  have hbenches : ℝ := hcircumference / space
  have happrox : ℝ := hbenches
  have hresult : ℝ := 31
  have rounded := int.of_real happrox
  have result := rounded.nat_abs
  -- Proof (omitted)
  exact sorry

end benches_needed_l743_743677


namespace no_triangle_division_l743_743056

noncomputable def triangle : Type := ℝ × ℝ × ℝ 

theorem no_triangle_division (A B C : triangle) (n : ℕ) 
  (h1 : 2 < n) : 
  ¬ ∃ (AFs : fin (n - 1) → triangle), 
  (∀ i : fin (n - 1), 
    let AF := AFs i in 
    divides_equal_area A B C n AF) :=
sorry


end no_triangle_division_l743_743056


namespace smallest_k_digit_number_l743_743504

theorem smallest_k_digit_number (a n : ℕ) (h1: a > 0) (h2 : (nat.log 10 (a ^ n) + 1) = 2014) : 2014 = Inf {k : ℕ | ¬(10^(k-1) ≤ a ∧ a < 10^k)} :=
by sorry

end smallest_k_digit_number_l743_743504


namespace sum_of_lonely_integers_l743_743676

def compatible (m n : ℕ) : Prop :=
  m ≥ n / 2 + 7 ∧ n ≥ m / 2 + 7

def lonely (k : ℕ) : Prop :=
  ∀ (ℓ : ℕ), ℓ ≥ 1 → ¬ compatible k ℓ

theorem sum_of_lonely_integers : ∑ k in (Finset.range 13).map Finset.succ, k = 91 :=
by
  sorry

end sum_of_lonely_integers_l743_743676


namespace monotonicity_and_minimum_range_of_a_l743_743400

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := -x^2 + a * x - 3

theorem monotonicity_and_minimum :
  (∀ x > 0, f' x = 1 + Real.log x)
  ∧ (∀ x ∈ Ioo 0 (1 / Real.exp 1), f' x < 0)
  ∧ (∀ x > 1 / Real.exp 1, f' x > 0)
  ∧ f (1 / Real.exp 1) = -(1 / Real.exp 1) := by
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, 2 * f x ≥ g x a) ↔ a ≤ 4 := by
  sorry

end monotonicity_and_minimum_range_of_a_l743_743400


namespace smallest_k_digit_number_l743_743505

theorem smallest_k_digit_number (a n : ℕ) (h1: a > 0) (h2 : (nat.log 10 (a ^ n) + 1) = 2014) : 2014 = Inf {k : ℕ | ¬(10^(k-1) ≤ a ∧ a < 10^k)} :=
by sorry

end smallest_k_digit_number_l743_743505


namespace triangle_altitude_l743_743568

theorem triangle_altitude (b : ℕ) (h : ℕ) (area : ℕ) (h_area : area = 800) (h_base : b = 40) (h_formula : area = (1 / 2) * b * h) : h = 40 :=
by
  sorry

end triangle_altitude_l743_743568


namespace hose_removal_rate_l743_743564

def pool_width := 60
def pool_length := 100
def pool_depth := 10
def pool_capacity := 0.80
def drain_time := 800

theorem hose_removal_rate :
  let full_volume := pool_width * pool_length * pool_depth in
  let current_volume := full_volume * pool_capacity in
  (current_volume / drain_time) = 60 := 
by 
  sorry

end hose_removal_rate_l743_743564


namespace octagon_arithmetic_sequences_count_l743_743947

theorem octagon_arithmetic_sequences_count :
  ∃ (count : ℕ), count = 3 ∧
  ∀ (angles : Fin 8 → ℕ), (∀ i j, i < j → angles i < angles j) →
  (∀ i, 0 < angles i ∧ angles i < 160) →
  (∑ i, angles i = 1080) →
  ∃ (d : ℕ), d > 0 ∧ d < 8 ∧ 
  (∀ i, angles (Fin.ofNat i) = angles 0 + i * d) :=
begin
  sorry
end

end octagon_arithmetic_sequences_count_l743_743947


namespace correct_quadratic_equation_l743_743037

-- Definitions based on conditions
def root_sum (α β : ℝ) := α + β = 8
def root_product (α β : ℝ) := α * β = 24

-- Main statement to be proven
theorem correct_quadratic_equation (α β : ℝ) (h1 : root_sum 5 3) (h2 : root_product (-6) (-4)) :
    (α - 5) * (α - 3) = 0 ∧ (α + 6) * (α + 4) = 0 → α * α - 8 * α + 24 = 0 :=
sorry

end correct_quadratic_equation_l743_743037


namespace min_moves_are_two_l743_743957

structure Board :=
  (n : ℕ)
  (cells : fin n → fin n → bool) -- A function representing whether a cell contains a chip (true) or is empty (false)

def is_adjacent {n : ℕ} (a b : fin n × fin n) : Prop :=
  (a.fst = b.fst ∧ (a.snd = b.snd + 1 ∨ a.snd = b.snd - 1)) ∨
  (a.snd = b.snd ∧ (a.fst = b.fst + 1 ∨ a.fst = b.fst - 1)) ∨
  (a.fst = b.fst + 1 ∧ (a.snd = b.snd + 1 ∨ a.snd = b.snd - 1)) ∨
  (a.fst = b.fst - 1 ∧ (a.snd = b.snd + 1 ∨ a.snd = b.snd - 1))

def move_chip (b : Board) (from to : fin b.n × fin b.n) (h_adj : is_adjacent from to) :
  Board :=
  { n := b.n,
    cells := λ i j, if (i, j) = from then false else if (i, j) = to then true else b.cells i j }

def chips_in_row (b : Board) (r : fin b.n) : ℕ :=
  finset.card (finset.filter (λ c, b.cells r c) finset.univ)

def chips_in_col (b : Board) (c : fin b.n) : ℕ :=
  finset.card (finset.filter (λ r, b.cells r c) finset.univ)

def balanced (b : Board) : Prop :=
  (∀ r : fin b.n, chips_in_row b r = 2) ∧
  (∀ c : fin b.n, chips_in_col b c = 2)

noncomputable def minimal_moves_to_balance (ini_board : Board) : ℕ :=
  if h : ∃ (m : ℕ), (balanced (iterate_move ini_board m)) then dite m else 0 -- non-computable def for clearer detail no ";" yield stacked comment

theorem min_moves_are_two (ini_board : Board) :
  minimal_moves_to_balance ini_board = 2 :=
sorry

end min_moves_are_two_l743_743957


namespace student_arrangements_l743_743610

theorem student_arrangements (n : ℕ) (s : Finₓ 5 → ℕ) :
  (∀ i, i = s 1 ↔ s i = s 2) →
  (∀ i, i ≠ s 0 → i ≠ s 4) →
  ∃ t : ℕ, t = 36 := 
sorry

end student_arrangements_l743_743610


namespace batsman_average_after_25th_innings_l743_743268

theorem batsman_average_after_25th_innings :
  ∃ A : ℝ, 
    (∀ s : ℝ, s = 25 * A + 62.5 → 24 * A + 95 = s) →
    A + 2.5 = 35 :=
by
  sorry

end batsman_average_after_25th_innings_l743_743268


namespace sam_remaining_money_l743_743543

def cost_of_candy_bars (num_candies cost_per_candy: nat) : nat := num_candies * cost_per_candy
def remaining_dimes (initial_dimes cost_in_dimes: nat) : nat := initial_dimes - cost_in_dimes
def remaining_quarters (initial_quarters cost_in_quarters: nat) : nat := initial_quarters - cost_in_quarters
def total_money_in_cents (dimes quarters: nat) : nat := (dimes * 10) + (quarters * 25)

theorem sam_remaining_money : 
  let initial_dimes := 19 in
  let initial_quarters := 6 in
  let num_candy_bars := 4 in
  let cost_per_candy := 3 in
  let cost_of_lollipop := 1 in
  let dimes_left := remaining_dimes initial_dimes (cost_of_candy_bars num_candy_bars cost_per_candy) in
  let quarters_left := remaining_quarters initial_quarters cost_of_lollipop in
  total_money_in_cents dimes_left quarters_left = 195 :=
by
  sorry

end sam_remaining_money_l743_743543


namespace equal_area_division_l743_743186

variable (R : ℝ) -- common radius for segments intersecting at O
variables (α β γ δ : ℝ) -- angles at point O

-- Condition 1: sum of angles at O
axiom sum_of_angles : α + β + γ + δ = 2 * π

-- Condition 2: Relationship between α and β via symmetry
axiom angle_symmetry : (α / 2) + (β / 2) = π / 2

-- Condition 5: α + β = π
axiom angle_sum : α + β = π

-- Area formula definitions
def area_AOB : ℝ := (1 / 2) * R^2 * Real.sin α
def area_COD : ℝ := (1 / 2) * R^2 * Real.sin β

theorem equal_area_division : area_AOB R α = area_COD R β :=
by
  sorry

end equal_area_division_l743_743186


namespace isosceles_triangle_l743_743088

open_locale euclidean_geometry

variables {A B C I P Q M N : Point}
variables (α : circle)
variables (circumcircle_AIC : circle)

-- Condition 1
hypothesis h1 : incenter I (triangle.mk A B C)

-- Condition 2
hypothesis h2 : α = incircle (triangle.mk A B C)

-- Condition 3
hypothesis h3 : intersects circumcircle_AIC α P
hypothesis h4 : intersects circumcircle_AIC α Q

-- Condition 4
hypothesis h5 : same_side P A (line.mk B I)

-- Condition 5
hypothesis h6 : ¬ same_side Q C (line.mk B I)

-- Condition 6
hypothesis h7 : midpoint M (arc.mk α A C)

-- Condition 7
hypothesis h8 : midpoint N (arc.mk α B C)

-- Condition 8
hypothesis h9 : parallel (line.mk P Q) (line.mk A C)

-- Conclusion
theorem isosceles_triangle (h1 h2 h3 h4 h5 h6 h7 h8 h9) : (distance A B) = (distance A C) :=
sorry

end isosceles_triangle_l743_743088


namespace jackson_pbj_sandwiches_l743_743057

theorem jackson_pbj_sandwiches :
  let total_weeks := 36
  let holidays_wed := 2
  let holidays_fri := 3
  let absences_wed := 1
  let absences_fri := 2
  let ham_cheese_schedule := total_weeks / 4

  let wednesdays := total_weeks - holidays_wed - absences_wed
  let fridays := total_weeks - holidays_fri - absences_fri
  let ham_cheese_wed := ham_cheese_schedule 
  let ham_cheese_fri := 2 * ham_cheese_schedule

  let pbj_wed := wednesdays - ham_cheese_wed
  let pbj_fri := fridays - ham_cheese_fri
  let total_pbj := pbj_wed + pbj_fri

  in total_pbj = 37 := sorry

end jackson_pbj_sandwiches_l743_743057


namespace max_value_a_l743_743734

-- Define the variables and the constraint on the circle
def circular_arrangement_condition (x: ℕ → ℕ) : Prop :=
  ∀ i: ℕ, 1 ≤ x i ∧ x i ≤ 10 ∧ x i ≠ x (i + 1)

-- Define the existence of three consecutive numbers summing to at least 18
def three_consecutive_sum_ge_18 (x: ℕ → ℕ) : Prop :=
  ∃ i: ℕ, x i + x (i + 1) + x (i + 2) ≥ 18

-- The main theorem we aim to prove
theorem max_value_a : ∀ (x: ℕ → ℕ), circular_arrangement_condition x → three_consecutive_sum_ge_18 x :=
  by sorry

end max_value_a_l743_743734


namespace parallelogram_area_vector_a_property_l743_743812

namespace VectorSpace

-- Points A, B, and C in space
def A : Fin 3 → ℝ := ![0, 2, 3]
def B : Fin 3 → ℝ := ![ -2, 1, 6]
def C : Fin 3 → ℝ := ![ 1, -1, 5]

-- Vector AB and AC
def AB : Fin 3 → ℝ := ![B 0 - A 0, B 1 - A 1, B 2 - A 2]
def AC : Fin 3 → ℝ := ![C 0 - A 0, C 1 - A 1, C 2 - A 2]

-- Norm of a vector
def vector_norm (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt (Finset.univ.sum (λ i, (v i) ^ 2))

-- Area of the parallelogram formed by vectors AB and AC
theorem parallelogram_area : vector_norm (Fin.mk 3 (λ i, (AB i * AC (2 - i)))) = 7 * Real.sqrt 3 :=
by
  sorry

-- Vector 'a' with given conditions
def a : Fin 3 → ℝ := λ i, if i = 0 then 1 else if i = 1 then 1 else 1
def a_alt : Fin 3 → ℝ := λ i, if i = 0 then -1 else if i = 1 then -1 else -1
def orthogonal (v w : Fin 3 → ℝ) : Prop := Finset.univ.sum (λ i, v i * w i) = 0

theorem vector_a_property :
  orthogonal a AB ∧ orthogonal a AC ∧ vector_norm a = Real.sqrt 3 ∧
  orthogonal a_alt AB ∧ orthogonal a_alt AC ∧ vector_norm a_alt = Real.sqrt 3 :=
by
  sorry

end VectorSpace

end parallelogram_area_vector_a_property_l743_743812


namespace prob_at_least_two_same_l743_743174

theorem prob_at_least_two_same (h : 8 > 0) : 
  (1 - (Nat.factorial 8 / (8^8) : ℚ) = 2043 / 2048) :=
by
  sorry

end prob_at_least_two_same_l743_743174


namespace exists_close_ratios_l743_743185

theorem exists_close_ratios (S : Finset ℝ) (h : S.card = 2000) :
  ∃ (a b c d : ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a > b ∧ c > d ∧ (a ≠ c ∨ b ≠ d) ∧
  abs ((a - b) / (c - d) - 1) < 1 / 100000 :=
sorry

end exists_close_ratios_l743_743185


namespace part_a_part_b_l743_743072

-- Definitions for the geometric objects and points.
variables {A B C N : Type}
variables [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint N]

-- Assume N is different from A, B, and C.
axiom N_not_A : N ≠ A
axiom N_not_B : N ≠ B
axiom N_not_C : N ≠ C

-- Reflections definitions
def reflection (P Q : Type) [IsPoint P] [IsPoint Q] : Type := sorry

-- Definitions of reflections with given requirements.
def A_b : Type := reflection A B
def B_a : Type := reflection B A
def B_c : Type := reflection B C
def C_b : Type := reflection C B
def A_c : Type := reflection A C
def C_a : Type := reflection C A

-- Definitions of lines perpendicular to specific segments through N.
def perpendicular_through (P Q : Type) [IsPoint P] [IsPoint Q] (R : Type) [IsPoint R] (segment PQ : Type) : Type := sorry

def m_a : Type := perpendicular_through N B_c C_b
def m_b : Type := perpendicular_through N C_a A_c
def m_c : Type := perpendicular_through N A_b B_a

-- Definitions for angle bisectors.
def angle_bisector (angle : Type) : Type := sorry

def bisector_BNC : Type := angle_bisector (angle B N C)
def bisector_CNA : Type := angle_bisector (angle C N A)
def bisector_ANB : Type := angle_bisector (angle A N B)

-- Main theorem for part a: reflections of m_a, m_b, m_c through angle bisectors are the same line.
theorem part_a (N_orthocenter : is_orthocenter N A B C) :
  (reflection m_a bisector_BNC) = (reflection m_b bisector_CNA) ∧
  (reflection m_a bisector_BNC) = (reflection m_c bisector_ANB) := sorry

-- Main theorem for part b: reflections of m_a, m_b, m_c through BC, CA, AB concur.
theorem part_b (N_nine_point_center : is_nine_point_center N A B C) :
  reflections_concur (reflection m_a BC) (reflection m_b CA) (reflection m_c AB) := sorry

end part_a_part_b_l743_743072


namespace sum_of_C_sequence_l743_743680

-- Define a sequence {a_n} where the harmonic mean of the first n terms is 1/(n+2)
def harmonic_mean (a : ℕ → ℚ) (n : ℕ) : Prop :=
  (n:ℚ) / ((finset.range n).sum a) = 1 / (n + 2)

def a (n : ℕ) : ℚ := 2 * n + 1

-- Define the sequence {C_n}
def C (n : ℕ) : ℚ := a n / 3^n

-- Define the sum S_n of the first n terms of the sequence {C_n}
def S (n : ℕ) : ℚ := (finset.range n).sum C

-- The main theorem to be proved
theorem sum_of_C_sequence (n : ℕ) : S n = 2 - (n + 2) / 3^n :=
by
  sorry

end sum_of_C_sequence_l743_743680


namespace wyatt_headmaster_duration_l743_743248

def duration_of_wyatt_job (start_month end_month total_months : ℕ) : Prop :=
  start_month <= end_month ∧ total_months = end_month - start_month + 1

theorem wyatt_headmaster_duration : duration_of_wyatt_job 3 12 9 :=
by
  sorry

end wyatt_headmaster_duration_l743_743248


namespace average_speed_with_stoppages_l743_743856

/--The average speed of the bus including stoppages is 20 km/hr, 
  given that the bus stops for 40 minutes per hour and 
  has an average speed of 60 km/hr excluding stoppages.--/
theorem average_speed_with_stoppages 
  (avg_speed_without_stoppages : ℝ)
  (stoppage_time_per_hour : ℕ) 
  (running_time_per_hour : ℕ) 
  (avg_speed_with_stoppages : ℝ) 
  (h1 : avg_speed_without_stoppages = 60) 
  (h2 : stoppage_time_per_hour = 40) 
  (h3 : running_time_per_hour = 20) 
  (h4 : running_time_per_hour + stoppage_time_per_hour = 60):
  avg_speed_with_stoppages = 20 := 
sorry

end average_speed_with_stoppages_l743_743856


namespace isosceles_triangle_of_parallel_PQ_AC_l743_743143

variables {α β γ : Type}
variables {A B C I M N P Q : α} {circleIncircle circCircumcircleAIC : set α}
variables {lineBI linePQ lineAC : set α}

-- Given conditions from the problem
def incenter_of_triangle (I A B C : α) : Prop := sorry
def incircle (circle α : set α) (A B C I : α) : Prop := sorry
def circumcircle (circle circumcircleAIC : set α) (A I C : α) : Prop := sorry
def midpoint_of_arc (M N : α) (circle α : set α) (A B C : α) : Prop := sorry
def parallel (linePQ lineAC : set α) : Prop := sorry
def lies_on_same_side_of_line (P A : α) (line lineBI : set α) : Prop := sorry
def lies_on_opposite_sides_of_line (Q C : α) (line lineBI : set α) : Prop := sorry

theorem isosceles_triangle_of_parallel_PQ_AC :
  incenter_of_triangle I A B C ∧
  incircle circleIncircle A B C I ∧
  circumcircle circCircumcircleAIC A I C ∧
  midpoint_of_arc M circleIncircle A C ∧
  midpoint_of_arc N circleIncircle B C ∧
  lies_on_same_side_of_line P A lineBI ∧
  lies_on_opposite_sides_of_line Q C lineBI ∧
  parallel linePQ lineAC →
  (triangle_isosceles A B C) := sorry

end isosceles_triangle_of_parallel_PQ_AC_l743_743143


namespace incorrect_statement_A_l743_743383

theorem incorrect_statement_A (m : ℝ) :
  let y := λ x, x^2 - 2 * m * x + m^2 - 9
  ∃ x : ℝ, y x = m^2 - 9 → False :=
by
  let y := λ x, x^2 - 2 * m * x + m^2 - 9
  have minY : ∀ x, y x ≥ -9 := 
    sorry  
  have h : ∀ x, y x = m^2 - 9 → x = m ∧ -9 = m^2 - 9 :=
    sorry
  obtain ⟨x, hx⟩ := h
  have eq := minY x
  rw [hx] at eq
  exact eq, 
  sorry


termination_with
termination_axiom _ := 
  let y := λ x, x^2 - 2 * m * x + m^2 - 9
  ∃ x : ℝ, y x = -9 :=
by 
  let y := λ x, x^2 - 2 * m * x + m^2 - 9
  have minY : (x - m)^2 ≥ 0 :=
    by 
      sorry 
  ∃ x : ℝ, y x = -9 :=
    sorry  
ē

end incorrect_statement_A_l743_743383


namespace differences_of_set_l743_743416

theorem differences_of_set :
  let S := {2, 3, 5, 7, 8, 9}
  ∃ n : ℕ, n = 7 ∧ (∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → 
    ((a - b).nat_abs ∈ {1, 2, 3, 4, 5, 6, 7})) := 
by
  let S := {2, 3, 5, 7, 8, 9}
  existsi 7,
  split,
  -- Here we assert that n = 7 as the number of distinct differences
  sorry

end differences_of_set_l743_743416


namespace smallest_k_for_a_n_digital_l743_743503

theorem smallest_k_for_a_n_digital (a n : ℕ) (h : 10^2013 ≤ a^n ∧ a^n < 10^2014) : 
  ∀ k : ℕ, (∀ b : ℕ, 10^(k-1) ≤ b → b < 10^k → (¬(10^2013 ≤ b^n ∧ b^n < 10^2014))) ↔ k = 2014 :=
by 
  sorry

end smallest_k_for_a_n_digital_l743_743503


namespace unique_solution_in_interval_l743_743498

theorem unique_solution_in_interval :
  ∃! x ∈ set.Icc (0 : ℝ) (Real.pi / 2),
    3 * (Real.sec x ^ 2 + Real.csc x ^ 2 - 2) -
    2 * ((Real.sec x ^ 2 + Real.csc x ^ 2 - 2) ^ 2 - 2) = 3 :=
sorry

end unique_solution_in_interval_l743_743498


namespace trajectory_of_P_is_a_ray_l743_743038

-- Define the two given points on the x-axis
def M := (-2, 0)
def N := (2, 0)

-- Define distance function
def dist (P Q : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The given condition
def condition (P : (ℝ × ℝ)) : Prop :=
  dist P M - dist P N = 4

-- The statement to prove
theorem trajectory_of_P_is_a_ray :
  ∃ P : ℝ × ℝ, condition P → P.1 > 2 ∧ P.2 = 0 := 
sorry

end trajectory_of_P_is_a_ray_l743_743038


namespace cars_people_equation_l743_743023

-- Define the first condition
def condition1 (x : ℕ) : ℕ := 4 * (x - 1)

-- Define the second condition
def condition2 (x : ℕ) : ℕ := 2 * x + 8

-- Main theorem which states that the conditions lead to the equation
theorem cars_people_equation (x : ℕ) : condition1 x = condition2 x :=
by
  sorry

end cars_people_equation_l743_743023


namespace needed_angle_BPC_l743_743487

-- Definitions needed from problem conditions

variables (A B C D P Q M N : Type)
variables [geometry.AB A B] [geometry.CD C D]
variables [MidpointSegment M AB] [MidpointSegment N CD]
variables [CloserToSegment P BC Q]
variables [angle.MPN M P N (40 : angle)]

theorem needed_angle_BPC 
  (h_rect : Rectangle ABCD)
  (h_circles : CirclesWithDiameters A B C D P Q) :
  angle BPC = 80 :=
by
  sorry

end needed_angle_BPC_l743_743487


namespace probability_white_ball_l743_743035

def num_white_balls : ℕ := 4
def num_yellow_balls : ℕ := 6
def total_balls : ℕ := 10
def white_ball_probability : ℚ := 2 / 5

theorem probability_white_ball :
  num_white_balls + num_yellow_balls = total_balls →
  (num_white_balls : ℚ) / (total_balls : ℚ) = white_ball_probability := by
  intros
  rw [←Nat.cast_add, H]
  norm_cast
  sorry  -- Proof steps to be filled

end probability_white_ball_l743_743035


namespace num_points_P_similar_triangles_l743_743305

theorem num_points_P_similar_triangles 
  (AB AD BC : ℝ) 
  (h_AB : AB = 7) 
  (h_AD : AD = 2) 
  (h_BC : BC = 3) : 
  (∃ P ∈ Icc 0 AB, 
    (∃ (x : ℝ) (hx1 : x = P) (hx2 : x / (AB - x) = AD / BC) 
      ∨ (x / BC = AD / (AB - x)) )
      ∧ multiset.card (multiset.filter (λ x, (x / (AB - x) = AD / BC) ∨ (x / BC = AD / (AB - x))) (multiset.range' 0 AB))
    ) = 3 :=
by
  sorry

end num_points_P_similar_triangles_l743_743305


namespace isosceles_triangle_l743_743104

noncomputable theory

open_locale classical

variables {A B C I P Q : Type*}

-- Let \( I \) be the incenter of triangle \( ABC \)
def is_incenter (I A B C : Type*) : Prop := sorry

-- Let \( \alpha \) be its incircle
def is_incircle (α : Type*) (I A B C : Type*) : Prop := sorry

-- The circumcircle of triangle \( AIC \) intersects \( \alpha \) at points \( P \) and \( Q \)
def circumcircle_intersect_incircle (α A I C P Q : Type*) : Prop := sorry

-- \( P \) and \( A \) lie on the same side of line \( BI \), and \( Q \) and \( C \) lie on the other side
def same_side (P A Q C : Type*) (BI : Type*) : Prop := sorry

-- \( PQ \parallel AC \)
def parallel (PQ AC : Type*) : Prop := sorry

-- Define triangle is isosceles
def is_isosceles (A B C : Type*) : Prop := sorry

theorem isosceles_triangle 
  (I : Type*) (A B C P Q M N : Type*)
  (α : Type*)
  (h_incenter : is_incenter I A B C)
  (h_incircle : is_incircle α I A B C)
  (h_intersect : circumcircle_intersect_incircle α A I C P Q)
  (h_sameside : same_side P A Q C (line BI))
  (h_parallel : parallel PQ (line AC))
: is_isosceles A B C :=
sorry

end isosceles_triangle_l743_743104


namespace sum_of_common_ratios_l743_743884

variable {k a_2 a_3 b_2 b_3 p r : ℝ}
variable (hp : a_2 = k * p) (ha3 : a_3 = k * p^2)
variable (hr : b_2 = k * r) (hb3 : b_3 = k * r^2)
variable (hcond : a_3 - b_3 = 5 * (a_2 - b_2))

theorem sum_of_common_ratios (h_nonconst : k ≠ 0) (p_ne_r : p ≠ r) : p + r = 5 :=
by
  sorry

end sum_of_common_ratios_l743_743884


namespace area_of_PQR_l743_743234

/-- Define an isosceles triangle with sides 17, 17, and 16 cm. -/
structure Triangle :=
  (a b c : ℝ)
  (is_isosceles : (a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (a = c ∧ b ≠ a))

def PQR : Triangle := 
  { a := 17, b := 17, c := 16, 
    is_isosceles := Or.inl ⟨rfl, by norm_num⟩ }

theorem area_of_PQR : (17 = 17 ∧ 16 ≠ 17) ∨ (17 = 16 ∧ 17 ≠ 16) ∨ (16 = 17 ∧ 17 ≠ 17) → 
  ∃ h : ℝ, (h = 15) ∧ 1/2 * 16 * h = 120 :=
by {
  intro h_iso,
  use 15,
  split,
  { norm_num },
  { norm_num }
}

end area_of_PQR_l743_743234


namespace triangle_ABC_isosceles_l743_743097

-- Lean 4 definitions for the generated equivalent proof problem
noncomputable def incenter (A B C I : Type*) := sorry
noncomputable def incircle (ABC : Type*) (α : Type*) := sorry
noncomputable def circumcircle (A I C : Type*) := sorry
noncomputable def intersects (circle1 circle2 : Type*) (P Q : Type*) := sorry
noncomputable def same_side_line (P A : Type*) (line : Type*) := sorry
noncomputable def other_side_line (Q C : Type*) (line : Type*) := sorry
noncomputable def midpoint_arc (arc : Type*) := sorry
noncomputable def parallel (line1 line2 : Type*) := sorry
noncomputable def triangle_isosceles (A B C : Type*) := ∃ (ABC_is_isosceles : Prop), ABC_is_isosceles

-- Given conditions for the incenter, incircle, and parallel lines, we must prove the triangle is isosceles
theorem triangle_ABC_isosceles (A B C I α P Q M N : Type*)
  (h1 : incenter A B C I)
  (h2 : incircle ABC α)
  (h3 : circumcircle A I C)
  (h4 : intersects α (circumcircle A I C) P Q)
  (h5 : same_side_line P A (set I B))
  (h6 : other_side_line Q C (set I B))
  (h7 : midpoint_arc α AC M)
  (h8 : midpoint_arc α BC N)
  (h9 : parallel PQ AC) :
  triangle_isosceles A B C := sorry

end triangle_ABC_isosceles_l743_743097


namespace log_sum_geometric_sequence_l743_743850

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ (r : ℝ) (n : ℕ), ∀ n, a (n + 1) = r * a n

theorem log_sum_geometric_sequence (a : ℕ → ℝ)
  (pos : ∀ n, 0 < a n)
  (geom : geometric_sequence a)
  (cond1 : a 10 * a 11 + a 8 * a 13 = 64)
  (cond2 : a 10 * a 11 = a 8 * a 13) :
  (∑ i in finset.range 20, real.log (a (i + 1)) / real.log 2) = 50 :=
begin
  sorry
end

end log_sum_geometric_sequence_l743_743850


namespace solve_repeating_decimals_sum_l743_743725

def repeating_decimals_sum : Prop :=
  let x := (1 : ℚ) / 3
  let y := (4 : ℚ) / 999
  let z := (5 : ℚ) / 9999
  x + y + z = 3378 / 9999

theorem solve_repeating_decimals_sum : repeating_decimals_sum := 
by 
  sorry

end solve_repeating_decimals_sum_l743_743725


namespace parabola_min_value_incorrect_statement_l743_743385

theorem parabola_min_value_incorrect_statement
  (m : ℝ)
  (A B : ℝ × ℝ)
  (P Q : ℝ × ℝ)
  (parabola : ℝ → ℝ)
  (on_parabola : ∀ (x : ℝ), parabola x = x^2 - 2*m*x + m^2 - 9)
  (A_intersects_x_axis : A.2 = 0)
  (B_intersects_x_axis : B.2 = 0)
  (A_on_parabola : parabola A.1 = A.2)
  (B_on_parabola : parabola B.1 = B.2)
  (P_on_parabola : parabola P.1 = P.2)
  (Q_on_parabola : parabola Q.1 = Q.2)
  (P_coordinates : P = (m + 1, parabola (m + 1)))
  (Q_coordinates : Q = (m - 3, parabola (m - 3))) :
  ∃ (min_y : ℝ), min_y = -9 ∧ min_y ≠ m^2 - 9 := 
sorry

end parabola_min_value_incorrect_statement_l743_743385


namespace monotonicity_and_inequality_l743_743515

def f (x : ℝ) := Real.log x - x + 1
def F (x : ℝ) := x * Real.log x - x + 1

theorem monotonicity_and_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < 1 ∨ 1 < x) :
  ((∀ y, 0 < y ∧ y < 1 → f y > f x) ∧ (∀ y, 1 < y → f y < f x)) ∧ (1 < x → (Real.log x < x - 1 ∧ x - 1 < x * Real.log x)) :=
by
  sorry

end monotonicity_and_inequality_l743_743515


namespace speed_second_half_l743_743172

theorem speed_second_half (H : ℝ) (S1 S2 : ℝ) (T : ℝ) : T = 11 → S1 = 30 → S1 * T1 = 150 → S1 * T1 + S2 * T2 = 300 → S2 = 25 :=
by
  intro hT hS1 hD1 hTotal
  sorry

end speed_second_half_l743_743172


namespace limit_of_A_l743_743292

noncomputable def A : ℕ → ℝ
| 0     := 0
| 1     := 1
| 2     := 2
| (n+3) := (A n + A (n+1) + A (n+2)) / 3 + 1 / ((n+3)^4 - (n+3)^2)

theorem limit_of_A :
  filter.tendsto A at_top (𝓝 (13 / 6 - real.pi ^ 2 / 12)) :=
sorry

end limit_of_A_l743_743292


namespace ordinate_of_Q_l743_743380

def curve : ℝ → ℝ := λ x, x^3 - 3*x^2 + 6*x + 2

def tangent_parallel_points (P Q : ℝ × ℝ) : Prop :=
  let y' := λ x, 3*x^2 - 6*x + 6 in
  let slope_P := y' P.1 in
  let slope_Q := y' Q.1 in
  slope_P = slope_Q

theorem ordinate_of_Q (P Q : ℝ × ℝ) (h1 : P.2 = 1) (h2 : tangent_parallel_points P Q) : Q.2 = 11 :=
sorry

end ordinate_of_Q_l743_743380


namespace ramu_net_profit_percentage_l743_743188

theorem ramu_net_profit_percentage :
  ∃ (profit_percentage : ℚ), abs (profit_percentage - 3.02) < 0.01 :=
by 
  let cost_of_car := 42000
  let repairs := 15000
  let sales_taxes := 3500
  let registration_fee := 2500
  let selling_price := 64900
  let total_cost := cost_of_car + repairs + sales_taxes + registration_fee
  let net_profit := selling_price - total_cost
  let profit_percentage := (net_profit.to_rat / total_cost) * 100
  exists profit_percentage
  sorry

end ramu_net_profit_percentage_l743_743188


namespace exists_x_y_with_specific_difference_l743_743429

theorem exists_x_y_with_specific_difference :
  ∃ x y : ℤ, (2 * x^2 + 8 * y = 26) ∧ (x - y = 26) := 
sorry

end exists_x_y_with_specific_difference_l743_743429


namespace isosceles_triangle_l743_743107

noncomputable theory

open_locale classical

variables {A B C I P Q : Type*}

-- Let \( I \) be the incenter of triangle \( ABC \)
def is_incenter (I A B C : Type*) : Prop := sorry

-- Let \( \alpha \) be its incircle
def is_incircle (α : Type*) (I A B C : Type*) : Prop := sorry

-- The circumcircle of triangle \( AIC \) intersects \( \alpha \) at points \( P \) and \( Q \)
def circumcircle_intersect_incircle (α A I C P Q : Type*) : Prop := sorry

-- \( P \) and \( A \) lie on the same side of line \( BI \), and \( Q \) and \( C \) lie on the other side
def same_side (P A Q C : Type*) (BI : Type*) : Prop := sorry

-- \( PQ \parallel AC \)
def parallel (PQ AC : Type*) : Prop := sorry

-- Define triangle is isosceles
def is_isosceles (A B C : Type*) : Prop := sorry

theorem isosceles_triangle 
  (I : Type*) (A B C P Q M N : Type*)
  (α : Type*)
  (h_incenter : is_incenter I A B C)
  (h_incircle : is_incircle α I A B C)
  (h_intersect : circumcircle_intersect_incircle α A I C P Q)
  (h_sameside : same_side P A Q C (line BI))
  (h_parallel : parallel PQ (line AC))
: is_isosceles A B C :=
sorry

end isosceles_triangle_l743_743107


namespace factor_x10_plus_1_l743_743726

theorem factor_x10_plus_1 :
  (x : ℝ → ℝ) → 
  ∃ p₁ p₂ p₃ p₄ : ℝ[X],
    (x^10 + 1 = p₁ * p₂ * p₃ * p₄ ∧
     p₁.degree > 0 ∧ p₂.degree > 0 ∧ p₃.degree > 0 ∧ p₄.degree > 0 ∧
     p₁.is_irreducible ∧ p₂.is_irreducible ∧ p₃.is_irreducible ∧ p₄.is_irreducible)
  sorry

end factor_x10_plus_1_l743_743726


namespace complex_numbers_on_unit_circle_l743_743018

noncomputable def z1 : ℂ := 1  -- Directly define the constants
noncomputable def z2 : ℂ := Complex.i
noncomputable def z3 : ℂ := -Complex.i

theorem complex_numbers_on_unit_circle (z1 z2 z3 : ℂ)
  (h1 : Complex.abs z1 = 1)
  (h2 : Complex.abs z2 = 1)
  (h3 : Complex.abs z3 = 1)
  (h4 : z1 + z2 + z3 = 1)
  (h5 : z1 * z2 * z3 = 1) :
  (z1, z2, z3) = (1, Complex.i, -Complex.i) :=
sorry

end complex_numbers_on_unit_circle_l743_743018


namespace alicia_bought_more_markers_l743_743411

theorem alicia_bought_more_markers (price_per_marker : ℝ) (n_h : ℝ) (n_a : ℝ) (m : ℝ) 
    (h_hector : n_h * price_per_marker = 2.76) 
    (h_alicia : n_a * price_per_marker = 4.07)
    (h_diff : n_a - n_h = m) : 
  m = 13 :=
sorry

end alicia_bought_more_markers_l743_743411


namespace incorrect_statement_regression_analysis_l743_743541

theorem incorrect_statement_regression_analysis (r : ℝ) :
  (r ∈ set.Icc (-1 : ℝ) 1 ∧ r ∉ set.Icc (-1 : ℝ) 1) → False :=
by
  intros h
  obtain ⟨h1, h2⟩ := h
  exact h2 h1


end incorrect_statement_regression_analysis_l743_743541


namespace ruler_line_perpendicular_l743_743611

theorem ruler_line_perpendicular (line_on_ground : ℝ^3 → Prop) (ruler_on_ground : ℝ^3 → Prop) :
  ∀ (ruler_placed : ℝ^3), 
  (∃ (ground_line : ℝ^3), ruler_on_ground ruler_placed → line_on_ground ground_line) → 
  (∃ (ground_line_perpendicular : ℝ^3), 
    (ruler_on_ground ruler_placed ∧ line_on_ground ground_line_perpendicular 
    ∧ (is_perpendicular ruler_placed ground_line_perpendicular))) := sorry

end ruler_line_perpendicular_l743_743611


namespace original_price_is_135_l743_743413

-- Problem Statement:
variable (P : ℝ)  -- Let P be the original price of the potion

-- Conditions
axiom potion_cost : (1 / 15) * P = 9

-- Proof Goal
theorem original_price_is_135 : P = 135 :=
by
  sorry

end original_price_is_135_l743_743413


namespace min_value_of_fraction_l743_743774

theorem min_value_of_fraction (a b : ℝ) (h_nonneg_a : 0 < a) (h_nonneg_b : 0 < b)
  (h_sum : a + b = 1) : 
  ∃ x, x = 3 + 2 * Real.sqrt 2 ∧ (∀ y, y = 1 / a + 2 / b → x ≤ y) :=
by
  use 3 + 2 * Real.sqrt 2
  intro y hy
  -- The proof would generally follow here
  sorry

end min_value_of_fraction_l743_743774


namespace advantage_18_vs_30_l743_743525

-- Definitions for advantages
def insurance_cost_effectiveness (age : ℕ) : Prop :=
  age = 18 → true

def rental_car_flexibility (age : ℕ) : Prop :=
  age = 18 → true

def employment_opportunities (age : ℕ) : Prop :=
  age = 18 → true

-- Aggregated definition of advantages
def advantage (age : ℕ) : ℕ :=
  if age = 18 then 3 else 0 -- Simplistic model for advantages count

-- Proof statement
theorem advantage_18_vs_30 : advantage 18 > advantage 30 :=
by { unfold advantage, norm_num }

end advantage_18_vs_30_l743_743525


namespace total_capacity_of_schools_l743_743840

theorem total_capacity_of_schools (a b c d t : ℕ) (h_a : a = 2) (h_b : b = 2) (h_c : c = 400) (h_d : d = 340) :
  t = a * c + b * d → t = 1480 := by
  intro h
  rw [h_a, h_b, h_c, h_d] at h
  simp at h
  exact h

end total_capacity_of_schools_l743_743840


namespace find_x_of_equation_l743_743863

theorem find_x_of_equation (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := by
  sorry

end find_x_of_equation_l743_743863


namespace value_of_x_l743_743243

theorem value_of_x (x : ℤ) (h : 3 * x / 7 = 21) : x = 49 :=
sorry

end value_of_x_l743_743243


namespace cyclic_quadrilateral_perpendiculars_concurrent_l743_743551

open Complex

theorem cyclic_quadrilateral_perpendiculars_concurrent
  (a b c d : ℂ)
  (h : ∃ O : ℂ, ∀ (x : ℂ), x ∈ set.insert a (set.insert b (set.insert c {d})) → abs (x - O) = abs (a - O)) :
  let m1 := (a + b) / 2
  let m2 := (b + c) / 2
  let m3 := (c + d) / 2
  let m4 := (d + a) / 2
  ∃ P : ℂ, ∀ (x : ℂ), x ∈ {m1, m2, m3, m4} → Perpendicular x P :=
  sorry

end cyclic_quadrilateral_perpendiculars_concurrent_l743_743551


namespace snickers_bars_needed_l743_743471

-- Definitions of the conditions
def points_needed : ℕ := 2000
def chocolate_bunny_points : ℕ := 100
def number_of_chocolate_bunnies : ℕ := 8
def snickers_points : ℕ := 25

-- Derived conditions
def points_from_bunnies : ℕ := number_of_chocolate_bunnies * chocolate_bunny_points
def remaining_points : ℕ := points_needed - points_from_bunnies

-- Statement to prove
theorem snickers_bars_needed : ∀ (n : ℕ), n = remaining_points / snickers_points → n = 48 :=
by 
  sorry

end snickers_bars_needed_l743_743471


namespace paco_ate_more_sweet_than_salty_l743_743528

theorem paco_ate_more_sweet_than_salty (s t : ℕ) (h_s : s = 5) (h_t : t = 2) : s - t = 3 :=
by
  sorry

end paco_ate_more_sweet_than_salty_l743_743528


namespace advantage_18_vs_30_l743_743524

-- Definitions for advantages
def insurance_cost_effectiveness (age : ℕ) : Prop :=
  age = 18 → true

def rental_car_flexibility (age : ℕ) : Prop :=
  age = 18 → true

def employment_opportunities (age : ℕ) : Prop :=
  age = 18 → true

-- Aggregated definition of advantages
def advantage (age : ℕ) : ℕ :=
  if age = 18 then 3 else 0 -- Simplistic model for advantages count

-- Proof statement
theorem advantage_18_vs_30 : advantage 18 > advantage 30 :=
by { unfold advantage, norm_num }

end advantage_18_vs_30_l743_743524


namespace distinct_words_sum_l743_743160

theorem distinct_words_sum (n : ℕ) (n1 n2 : ℕ) :
  (n = 5) →
  (n1 = 2) →
  (n2 = 2) →
  (∃ words_САМСА : ℕ, words_САМСА = Nat.factorial n / (Nat.factorial n1 * Nat.factorial n2) ∧ words_САМСА = 30) →
  (∃ words_ПАСТА : ℕ, words_ПАСТА = Nat.factorial n / Nat.factorial n1 ∧ words_ПАСТА = 60) →
  90 = 30 + 60 :=
by
  intros h_n h_n1 h_n2 h_words_САМСА h_words_ПАСТА
  obtain ⟨words_САМСА, h1, h2⟩ := h_words_САМСА
  obtain ⟨words_ПАСТА, h3, h4⟩ := h_words_ПАСТА
  rw [h2, h4]
  exact Nat.add_comm 30 60

end distinct_words_sum_l743_743160


namespace inequality_solution_l743_743560

theorem inequality_solution (x : ℝ) (h : x^2 + 3x + 9 ≠ 0) : (x + 5) / (x^2 + 3x + 9) ≥ 0 ↔ x ∈ set.Ici (-5) :=
by
  sorry

end inequality_solution_l743_743560


namespace product_of_major_and_minor_axes_l743_743912

-- Given definitions from conditions
variables (O F A B C D : Type) 
variables (OF : ℝ) (dia_inscribed_circle_OCF : ℝ) (a b : ℝ)

-- Condition: O is the center of an ellipse
-- Point F is one focus, OF = 8
def O_center_ellipse : Prop := OF = 8

-- The diameter of the inscribed circle of triangle OCF is 4
def dia_inscribed_circle_condition : Prop := dia_inscribed_circle_OCF = 4

-- Define OA = OB = a, OC = OD = b
def major_axis_half_length : ℝ := a
def minor_axis_half_length : ℝ := b

-- Ellipse focal property a^2 - b^2 = 64
def ellipse_focal_property : Prop := a^2 - b^2 = 64

-- From the given conditions, expected result
def compute_product_AB_CD : Prop := 
  let AB := 2 * a,
      CD := 2 * b
  in AB * CD = 240

-- The main statement to be proven
theorem product_of_major_and_minor_axes 
  (h1 : O_center_ellipse)
  (h2 : dia_inscribed_circle_condition)
  (h3 : ellipse_focal_property)
  : compute_product_AB_CD :=
sorry

end product_of_major_and_minor_axes_l743_743912


namespace product_of_positive_integer_values_of_d_l743_743347

theorem product_of_positive_integer_values_of_d :
  (∏ d in finset.filter (λ d, 9x^2 + 18x + d = 0) (finset.range 9), d) = 40320 := by
{
  sorry
}

end product_of_positive_integer_values_of_d_l743_743347


namespace volume_of_rect_box_l743_743627

open Real

/-- Proof of the volume of a rectangular box given its face areas -/
theorem volume_of_rect_box (l w h : ℝ) 
  (A1 : l * w = 40) 
  (A2 : w * h = 10) 
  (A3 : l * h = 8) : 
  l * w * h = 40 * sqrt 2 :=
by
  sorry

end volume_of_rect_box_l743_743627


namespace equilateral_triangle_angles_l743_743466

namespace TriangleProof

-- Define the triangle ABC and its properties
structure Triangle :=
  (A B C : Type) 
  (A_eq_B : A = B)
  (B_eq_C : B = C)
  (angle_A : ℕ)
  (angle_CBA : ℕ)
  (angle_BCA : ℕ)
  (BC : ℕ)

-- To prove the measures given the conditions
theorem equilateral_triangle_angles (ABC : Triangle) (h1: ABC.angle_A = 60)
  (h2: ABC.BC = 1) (h3: ABC.A_eq_B = ABC.B_eq_C) :
  (ABC.angle_CBA = 60 ∧ ABC.angle_BCA = 60 ∧ ABC.angle_A = 120) :=
by
  sorry
  
end TriangleProof

end equilateral_triangle_angles_l743_743466


namespace eight_in_C_l743_743493

def C : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

theorem eight_in_C : 8 ∈ C :=
by {
  sorry
}

end eight_in_C_l743_743493


namespace count_common_divisors_l743_743322

theorem count_common_divisors : 
  (Nat.divisors 60 ∩ Nat.divisors 90 ∩ Nat.divisors 30).card = 8 :=
by
  sorry

end count_common_divisors_l743_743322


namespace triangle_ABC_right_angled_l743_743327
open Real

theorem triangle_ABC_right_angled (A B C : ℝ) (a b c : ℝ)
  (h1 : cos (2 * A) - cos (2 * B) = 2 * sin C ^ 2)
  (h2 : a = sin A) (h3 : b = sin B) (h4 : c = sin C)
  : a^2 + c^2 = b^2 :=
by sorry

end triangle_ABC_right_angled_l743_743327


namespace isosceles_triangle_l743_743092

open_locale euclidean_geometry

variables {A B C I P Q M N : Point}
variables (α : circle)
variables (circumcircle_AIC : circle)

-- Condition 1
hypothesis h1 : incenter I (triangle.mk A B C)

-- Condition 2
hypothesis h2 : α = incircle (triangle.mk A B C)

-- Condition 3
hypothesis h3 : intersects circumcircle_AIC α P
hypothesis h4 : intersects circumcircle_AIC α Q

-- Condition 4
hypothesis h5 : same_side P A (line.mk B I)

-- Condition 5
hypothesis h6 : ¬ same_side Q C (line.mk B I)

-- Condition 6
hypothesis h7 : midpoint M (arc.mk α A C)

-- Condition 7
hypothesis h8 : midpoint N (arc.mk α B C)

-- Condition 8
hypothesis h9 : parallel (line.mk P Q) (line.mk A C)

-- Conclusion
theorem isosceles_triangle (h1 h2 h3 h4 h5 h6 h7 h8 h9) : (distance A B) = (distance A C) :=
sorry

end isosceles_triangle_l743_743092


namespace fraction_of_visitors_l743_743031

theorem fraction_of_visitors
  (total_visitors did_not_enjoy_and_understand enjoyed_and_understood : ℕ)
  [h1 : total_visitors = 400]
  [h2 : did_not_enjoy_and_understand = 100]
  [h3 : enjoyed_and_understood = (total_visitors - did_not_enjoy_and_understand) / 2] :
  (enjoyed_and_understood + did_not_enjoy_and_understand = total_visitors) ∧
  (enjoyed_and_understood / total_visitors = (3 : ℤ) / 8) := by
  sorry

end fraction_of_visitors_l743_743031


namespace reciprocal_of_recurring_three_l743_743991

noncomputable def recurring_three := 0.33333333333 -- approximation of 0.\overline{3}

theorem reciprocal_of_recurring_three :
  let x := recurring_three in
  (x = (1/3)) → (1 / x = 3) := 
by 
  sorry

end reciprocal_of_recurring_three_l743_743991


namespace number_of_ordered_pairs_with_round_table_l743_743749

theorem number_of_ordered_pairs_with_round_table (n : ℕ) (h : n = 5) :
  ∃ (pairs : set (ℕ × ℕ)), pairs = {(f, m) | f ≥ 0 ∧ m ≥ 0 ∧ ∃ (i j k l : ℕ), ((i = 0 ∧ j = 5) ∨ (i = 2 ∧ j = 5) ∨ (i = 3 ∧ j = 5) ∨ (i = 4 ∧ j = 5)) ∧ k < n ∧ l < n} ∧ pairs.card = 8 :=
by
  sorry

end number_of_ordered_pairs_with_round_table_l743_743749


namespace infinite_lines_intersect_all_l743_743831

-- Defining the setting with pairwise skew lines a, b, and c
variables {Line : Type} [Nonempty Line]
variables {a b c : Line}
variable (skew : ∀ (l1 l2 : Line), (¬ (l1 = l2)) → ¬ ∃ _ : l1 ∩ l2)

-- Defining the problem that there are infinitely many lines intersecting a, b, and c.
theorem infinite_lines_intersect_all (skew_a_b : skew a b)
                                     (skew_b_c : skew b c)
                                     (skew_c_a : skew c a) :
  ∃ (L : Line → Prop), (∀ (l : Line), L l → ∃ (p1 p2 p3 : kernel.Point), (l ⊃ p1) ∧ (l ⊃ p2) ∧ (l ⊃ p3) ∧ (p1 ∈ a) ∧ (p2 ∈ b) ∧ (p3 ∈ c)) ∧ (∀ n : ℕ, ∃ (ls : fin n → Line), ∀ i j, i ≠ j → ls i ≠ ls j) :=
sorry

end infinite_lines_intersect_all_l743_743831


namespace num_both_homeworks_l743_743440

variable {α : Type} [Fintype α] (M K : Set α)

def numMathHomework := 37
def numKoreanHomework := 42
def totalStudents := 48

theorem num_both_homeworks (hM : Fintype.card (↑M : Finset α) = numMathHomework)
  (hK : Fintype.card (↑K : Finset α) = numKoreanHomework)
  (hMK : Fintype.card (↑(M ∪ K : Set α) : Finset α) = totalStudents) :
  Fintype.card (↑(M ∩ K : Set α) : Finset α) = 31 := by
  sorry

end num_both_homeworks_l743_743440


namespace find_a_from_point_on_terminal_side_of_angle_l743_743389

theorem find_a_from_point_on_terminal_side_of_angle (a : ℝ) :
  (∃ a, ∀ (c := 120), (-4, a) is on the terminal side of angle c ) → a = 4 * Real.sqrt 3 :=
sorry

end find_a_from_point_on_terminal_side_of_angle_l743_743389


namespace C₁_eq_Cartesian_intersection_C₃_C₁_C₂_eq_Cartesian_intersection_C₃_C₂_l743_743040

-- Define the parametric equations for C1
def C₁ (t : ℝ) (ht : t ≥ 0) : ℝ × ℝ :=
  (⟨(2 + t) / 6, sqrt t⟩)

-- Define the Cartesian equation for C1
def C₁_Cartesian (x y : ℝ) : Prop :=
  y^2 = 6*x - 2 ∧ y ≥ 0

-- Define the parametric equations for C2
def C₂ (s : ℝ) (hs : s ≥ 0) : ℝ × ℝ :=
  (⟨-(2 + s) / 6, -sqrt s⟩)

-- Define the Cartesian equation for C2
def C₂_Cartesian (x y : ℝ) : Prop :=
  y^2 = -6*x - 2 ∧ y ≤ 0

-- Define the polar equation for C3
def C₃ (θ : ℝ) : Prop :=
  2*cos θ - sin θ = 0

-- Define the Cartesian coordinates conversion
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

-- Prove the Cartesian equation of C₁
theorem C₁_eq_Cartesian (t : ℝ) (ht : t ≥ 0) :
  ∃ y, C₁ t ht = ⟨(6* (((2 + t) / 6) : ℝ)) - 2, y^2⟩ := sorry

-- Prove the intersection points of C₃ with C₁
theorem intersection_C₃_C₁ :
  (∃ x y, C₃ (arctan y x) ∧ C₁_Cartesian x y ∧ (x = 1/2 ∧ y = 1 ∨ x = 1 ∧ y = 2)) := sorry

-- Prove the Cartesian equation of C₂
theorem C₂_eq_Cartesian (s : ℝ) (hs : s ≥ 0) :
  ∃ y, C₂ s hs = ⟨(6* ((-(2 + s) / 6) : ℝ)) + 2, y^2⟩ := sorry
  
-- Prove the intersection points of C₃ with C₂
theorem intersection_C₃_C₂ :
  (∃ x y, C₃ (arctan y x) ∧ C₂_Cartesian x y ∧ (x = -1/2 ∧ y = -1 ∨ x = -1 ∧ y = -2)) := sorry

end C₁_eq_Cartesian_intersection_C₃_C₁_C₂_eq_Cartesian_intersection_C₃_C₂_l743_743040


namespace arithmetic_seq_problem_l743_743460

-- Define an arithmetic sequence (a_n)
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + (n - 1) * d

-- Given condition
def cond := ∃ (a : ℕ → ℝ), arithmetic_seq a ∧ a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- Main goal
theorem arithmetic_seq_problem : cond → (∃ (a : ℕ → ℝ), 2 * a 10 - a 12 = 24) :=
by
  intro h
  sorry

end arithmetic_seq_problem_l743_743460


namespace smallest_positive_period_max_and_min_values_l743_743396

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - sqrt 3 * (cos x)^2

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π := sorry

theorem max_and_min_values :
  ∀ x ∈ set.Icc 0 (π / 2), f x ≤ 1 - sqrt 3 / 2 ∧ f x ≥ -sqrt 3 ∧ 
  ∃ x_max x_min, x_max ∈ set.Icc 0 (π / 2) ∧ x_min ∈ set.Icc 0 (π / 2) ∧
  f x_max = 1 - sqrt 3 / 2 ∧ f x_min = -sqrt 3 := sorry

end smallest_positive_period_max_and_min_values_l743_743396


namespace min_intersection_value_l743_743153

theorem min_intersection_value (A B C D : Set)
  (hA : n(A) = 2^150) (hB : n(B) = 2^150) (hD : n(D) = 2^102)
  (sum_NS : n(A) + n(B) + n(C) + n(D) = 2^152)
  (union_N : n(A) + n(B) + n(C) + n(D) = (A ∪ B ∪ C ∪ D)) :
  n(A ∩ B ∩ C ∩ D) = 99 :=
sorry

end min_intersection_value_l743_743153


namespace OI_squared_eq_R_squared_sub_2Rr_l743_743467

-- Definitions representing the conditions
variable {ABC : Type} [triangle ABC] (O : point) (I : point)
  (R r : ℝ)
  (circumcenter : triangle ABC → point)
  (incenter : triangle ABC → point)
  (circumradius : triangle ABC → ℝ)
  (inradius : triangle ABC → ℝ)

-- Defining the specific properties as conditions
axiom circumcenter_property : ∀ (T : triangle ABC), circumcenter T = O
axiom incenter_property : ∀ (T : triangle ABC), incenter T = I
axiom circumradius_property : ∀ (T : triangle ABC), circumradius T = R
axiom inradius_property : ∀ (T : triangle ABC), inradius T = r

-- Goal statement
theorem OI_squared_eq_R_squared_sub_2Rr :
  OI^2 = R^2 - 2 * R * r := sorry

end OI_squared_eq_R_squared_sub_2Rr_l743_743467


namespace washington_high_student_population_l743_743955

theorem washington_high_student_population :
  let initial_students := 27.5 * 42
  let increased_students_percent := 0.15 * initial_students
  let total_students_increased := (initial_students + increased_students_percent).toNat
  let capacity := 1300
  let students_over_capacity := total_students_increased - capacity
  initial_students = 1155 ∧ total_students_increased = 1328 ∧ students_over_capacity = 28 :=
by 
  let initial_students := 27.5 * 42
  let increased_students_percent := 0.15 * initial_students
  let total_students_increased := (initial_students + increased_students_percent).toNat
  let capacity := 1300
  let students_over_capacity := total_students_increased - capacity
  have h1 : initial_students = 1155, by norm_num
  have h2 : total_students_increased = 1328, by norm_num
  have h3 : students_over_capacity = 28, by norm_num
  exact ⟨h1, h2, h3⟩
  sorry

end washington_high_student_population_l743_743955


namespace starting_time_1_57_58_l743_743212

theorem starting_time_1_57_58 :
  ∃ start_time : Time, start_time = Time.mk 1 57 58 ∧
    ∀ end_time : Time, end_time = Time.mk 3 20 47 →
    ∀ glow_interval : ℕ, glow_interval = 30 →
    ∀ glow_times : ℝ, glow_times = 165.63333333333333 →
    let total_seconds := (glow_times * glow_interval : ℝ).to_nat in
    let duration := (total_seconds.div_mod 3600).fst in
    let minutes_seconds := (total_seconds.div_mod 3600).snd.div_mod 60 in
    let hours := duration in
    let minutes := minutes_seconds.fst in
    let seconds := minutes_seconds.snd in
    let start_seconds := if end_time.seconds >= seconds
                          then end_time.seconds - seconds
                          else end_time.seconds + 60 - seconds in
    let start_minutes := if end_time.seconds >= seconds
                          then if end_time.minutes >= minutes + 1
                                then end_time.minutes - (minutes + 1)
                                else end_time.minutes + 60 - (minutes + 1)
                          else if end_time.minutes > minutes
                                then end_time.minutes - minutes
                                else end_time.minutes + 59 - minutes in
    let start_hours := if end_time.seconds >= seconds &&
                        end_time.minutes >= minutes + 1
                        then end_time.hours - hours - 1
                        else if end_time.seconds >= seconds ||
                                end_time.minutes >= minutes + 1
                              then end_time.hours - hours
                              else end_time.hours - hours - 1 in
    start_time = Time.mk start_hours start_minutes start_seconds := sorry

end starting_time_1_57_58_l743_743212


namespace prob_at_least_two_same_l743_743175

theorem prob_at_least_two_same (h : 8 > 0) : 
  (1 - (Nat.factorial 8 / (8^8) : ℚ) = 2043 / 2048) :=
by
  sorry

end prob_at_least_two_same_l743_743175


namespace find_triangle_altitude_l743_743571

variable (A b h : ℝ)

theorem find_triangle_altitude (h_eq_40 :  A = 800 ∧ b = 40) : h = 40 :=
sorry

end find_triangle_altitude_l743_743571


namespace odd_palindrome_count_l743_743165

theorem odd_palindrome_count :
  (∃ A B : ℕ, 1 ≤ A ∧ A ≤ 9 ∧ A % 2 = 1 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 9 * B + A = 50) :=
begin
  sorry
end

end odd_palindrome_count_l743_743165


namespace math_problem_l743_743082

theorem math_problem (a b c k : ℝ) (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h2 : a + b + c = 0) (h3 : a^2 = k * b^2) (hk : k ≠ 0) :
  (a^2 * b^2) / ((a^2 - b * c) * (b^2 - a * c)) + (a^2 * c^2) / ((a^2 - b * c) * (c^2 - a * b)) + (b^2 * c^2) / ((b^2 - a * c) * (c^2 - a * b)) = 1 :=
by
  sorry

end math_problem_l743_743082


namespace jennifer_spending_l743_743060

/-- Jennifer's expenditures problem -/
theorem jennifer_spending (T S M L : ℝ) (h_total : T = 180)
  (h_sandwich : S = 1 / 5 * T) (h_museum : M = 1 / 6 * T) (h_leftover : L = 24) :
  let B := T - L - (S + M),
      f := B / T
  in f = 1 / 2 :=
by
  sorry

end jennifer_spending_l743_743060


namespace smallest_difference_l743_743994

noncomputable def is_3_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def uses_digits (n m : ℕ) (digits : Finset ℕ) : Prop :=
  digits = {4, 5, 6, 7, 8, 9} ∧
  digits.card = 6 ∧
  ((n.digits.filter (λ d => d ∈ digits)).length = 3) ∧
  ((m.digits.filter (λ d => d ∈ digits)).length = 3) ∧
  (n.digits.all (λ d => d ∈ digits)) ∧
  (m.digits.all (λ d => d ∈ digits)) ∧
  (n ≠ m) ∧
  ((n.digits ∪ m.digits).card = 6)

theorem smallest_difference :
  ∃ n m : ℕ, 
  is_3_digit n ∧ is_3_digit m ∧
  uses_digits n m {4, 5, 6, 7, 8, 9} ∧
  abs (n - m) = 129 :=
sorry

end smallest_difference_l743_743994


namespace cost_price_books_l743_743049

def cost_of_type_A (cost_A cost_B : ℝ) : Prop :=
  cost_A = cost_B + 15

def quantity_equal (cost_A cost_B : ℝ) : Prop :=
  675 / cost_A = 450 / cost_B

theorem cost_price_books (cost_A cost_B : ℝ) (h1 : cost_of_type_A cost_A cost_B) (h2 : quantity_equal cost_A cost_B) : 
  cost_A = 45 ∧ cost_B = 30 :=
by
  -- Proof omitted
  sorry

end cost_price_books_l743_743049


namespace total_handshakes_five_people_l743_743629

theorem total_handshakes_five_people : 
  let n := 5
  let total_handshakes (n : ℕ) : ℕ := (n * (n - 1)) / 2
  total_handshakes 5 = 10 :=
by sorry

end total_handshakes_five_people_l743_743629


namespace largest_value_d_l743_743151

theorem largest_value_d (a b c d : ℝ) 
  (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 20) :
  d ≤ 5 + 5 * real.sqrt 34 / 3 := sorry

end largest_value_d_l743_743151


namespace drink_cost_l743_743974

/-- Wade has called into a rest stop and decides to get food for the road. 
  He buys a sandwich to eat now, one for the road, and one for the evening. 
  He also buys 2 drinks. Wade spends a total of $26 and the sandwiches 
  each cost $6. Prove that the drinks each cost $4. -/
theorem drink_cost (cost_sandwich : ℕ) (num_sandwiches : ℕ) (cost_total : ℕ) (num_drinks : ℕ) :
  cost_sandwich = 6 → num_sandwiches = 3 → cost_total = 26 → num_drinks = 2 → 
  ∃ (cost_drink : ℕ), cost_drink = 4 :=
by
  intro h1 h2 h3 h4
  sorry

end drink_cost_l743_743974


namespace problem_solution_l743_743393

-- Conditions and Definitions
def f (x : ℝ) (ω : ℝ) := sin (ω * x) ^ 2 + sqrt 3 * sin (ω * x) * sin (ω * x + π / 2)
axiom ω_pos : ω > 0
axiom smallest_period : ∀ x, f (x + π) ω = f x ω

-- Theorem with statements
theorem problem_solution : ω = 1 ∧
  (∀ k : ℤ, ℝ, -π / 6 + k * π ≤ 2 * x - π / 6 ∧ 2 * x - π / 6 ≤ π / 3 + k * π → (x ∈ set.Icc (0 : ℝ) (2 * π / 3 : ℝ))) ∧
  (∃ x_max ∈ set.Icc (0 : ℝ) (π / 3 : ℝ), f x_max ω = 3 / 2) ∧
  (∀ x_min ∈ {0, 2 * π / 3}, f x_min ω = 0) :=
sorry

end problem_solution_l743_743393


namespace measure_of_angle_BAC_l743_743076

variables {A B C O : Type} [InnerProductSpace ℝ O] (a b c o : O)

-- Given conditions
variables (h1 : is_circumcenter O (triangle A B C))
variables (h2 : a = (1/3 : ℝ) • b + (1/3 : ℝ) • c)

-- Definition of is_circumcenter
def is_circumcenter (O : O) (T : triangle O O O) : Prop :=
dist O T.a = dist O T.b ∧ dist O T.c = dist O T.a

-- Statement of the problem
theorem measure_of_angle_BAC : 
  ∠ B A C = 60 :=
sorry

end measure_of_angle_BAC_l743_743076


namespace spherical_circle_radius_l743_743594

noncomputable def radius_of_spherical_circle : ℝ :=
  let ρ := 1
  let φ := real.pi / 4
  let r := real.sqrt ((ρ * real.sin φ * real.cos 0)^2 + (ρ * real.sin φ * real.sin 0)^2)
  r

theorem spherical_circle_radius :
  ∃ r : ℝ, r = radius_of_spherical_circle ∧ r = real.sqrt 2 / 2 :=
by
  sorry

end spherical_circle_radius_l743_743594


namespace reciprocal_of_recurring_three_l743_743990

noncomputable def recurring_three := 0.33333333333 -- approximation of 0.\overline{3}

theorem reciprocal_of_recurring_three :
  let x := recurring_three in
  (x = (1/3)) → (1 / x = 3) := 
by 
  sorry

end reciprocal_of_recurring_three_l743_743990


namespace last_score_is_79_l743_743897

open Nat

noncomputable def is_prime (n : ℕ) : Prop := Prime n

noncomputable def avg_is_integer (lst : List ℕ) : Prop := lst.sum % lst.length = 0

noncomputable def valid_sequence (lst : List ℕ) : Prop :=
  avg_is_integer lst ∧
  is_prime (lst.get! 1) ∧
  is_prime (lst.get! 3)

theorem last_score_is_79 :
  ∃ lst : List ℕ, lst = [a, b, c, 79] ∧
  (∀ i, i < 4 → avg_is_integer (lst.take (i + 1))) ∧
  valid_sequence lst :=
sorry

end last_score_is_79_l743_743897


namespace triangle_area_DBC_l743_743846

/-- Proof that the area of triangle DBC is 12 given specific conditions -/
theorem triangle_area_DBC :
  let A := (0, 6)
  let B := (0, 0)
  let C := (8, 0)
  let D := (0, (6 / 2))
  base BC := (C.1 - B.1) /\
  height BD := D.2 - B.2
  (1 / 2) * base * height = 12 := 
by
  sorry

end triangle_area_DBC_l743_743846


namespace coffee_shop_lattes_l743_743935

theorem coffee_shop_lattes (T : ℕ) (L : ℕ) (hT : T = 6) (hL : L = 4 * T + 8) : L = 32 :=
by
  sorry

end coffee_shop_lattes_l743_743935


namespace find_x_l743_743859

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (h : x + y + x * y = 143) : x = 15 :=
by sorry

end find_x_l743_743859


namespace weekly_goal_cans_l743_743632

theorem weekly_goal_cans (c₁ c₂ c₃ c₄ c₅ : ℕ) (h₁ : c₁ = 20) (h₂ : c₂ = c₁ + 5) (h₃ : c₃ = c₂ + 5) 
  (h₄ : c₄ = c₃ + 5) (h₅ : c₅ = c₄ + 5) : 
  c₁ + c₂ + c₃ + c₄ + c₅ = 150 :=
by
  sorry

end weekly_goal_cans_l743_743632


namespace sum_of_consecutive_integers_l743_743526

theorem sum_of_consecutive_integers (n : ℕ) : 
  (Finset.range (2 * n - 1)).sum (λ i, n + i) = (2 * n - 1) ^ 2 := sorry

end sum_of_consecutive_integers_l743_743526


namespace find_a_l743_743339

theorem find_a (a : ℝ) :
  (∃ b x y : ℝ, y = x^2 + a ∧ x^2 + y^2 + 2 * b^2 = 2 * b * (x - y) + 1) ↔ (a ≤ sqrt 2 + 1 / 4) :=
sorry

end find_a_l743_743339


namespace determine_sum_of_digits_l743_743851

theorem determine_sum_of_digits (A B C E F : ℕ) 
  (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ E ∧ A ≠ F ∧ B ≠ C ∧ B ≠ E ∧ B ≠ F ∧ C ≠ E ∧ C ≠ F ∧ E ≠ F)
  (h2 : A < 10 ∧ B < 10 ∧ C < 10 ∧ E < 10 ∧ F < 10)
  (h3 : Prime (10 * E + F)) 
  (h4 : (100 * A + 10 * B + C) * (10 * E + F) = 1010 * E + 101 * F) : 
  A + B = 1 := 
begin
  sorry
end

end determine_sum_of_digits_l743_743851


namespace smallest_whole_number_larger_than_sum_l743_743350

theorem smallest_whole_number_larger_than_sum : 
  let sum := (2 + 1/4) + (3 + 1/5) + (4 + 1/6) + (5 + 1/7)
  in 15 = Int.ceiling sum :=
sorry

end smallest_whole_number_larger_than_sum_l743_743350


namespace find_sum_l743_743779

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α)

-- Geometric sequence condition
def geometric_seq (a : ℕ → α) (r : α) := ∀ n : ℕ, a (n + 1) = a n * r

theorem find_sum (r : α)
  (h1 : geometric_seq a r)
  (h2 : a 4 + a 7 = 2)
  (h3 : a 5 * a 6 = -8) :
  a 1 + a 10 = -7 := 
sorry

end find_sum_l743_743779


namespace solve_equation_l743_743926

theorem solve_equation (x : ℝ) :
    x^6 - 22 * x^2 - Real.sqrt 21 = 0 ↔ x = Real.sqrt ((Real.sqrt 21 + 5) / 2) ∨ x = -Real.sqrt ((Real.sqrt 21 + 5) / 2) := by
  sorry

end solve_equation_l743_743926


namespace fixed_point_coordinates_l743_743574

theorem fixed_point_coordinates (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  (∃ x y, f x = y ∧ (x, y) = (-1, -1)) :=
  let f : ℝ → ℝ := λ x, 2 * a ^ (x + 1) - 3 in
  sorry

end fixed_point_coordinates_l743_743574


namespace isosceles_triangle_l743_743131

   open EuclideanGeometry

   -- Define the conditions of the problem in Lean
   variable {I A B C P Q M N : Point}
   variable (α : Circle) (circumcircle_AIC : Circle)

   -- Conditions extracted from the problem
   def conditions : Prop :=
   IsIncenter I △ABC ∧
   Incircle α △ABC ∧
   Circle.Diameter α P Q ∧
   Circle.Containing circumcircle_AIC (trianglePoint AIC) ∧
   SameSide P A (Line BI) ∧
   SameSide Q C (Line BI) ∧
   IsMidpointArc M ARC(α A C) ∧
   IsMidpointArc N ARC(α B C) ∧
   Parallel (Line PQ) (Line AC)

   -- Proof statement in Lean
   theorem isosceles_triangle
     (h : conditions α circumcircle_AIC) : IsIsosceles (△ABC) :=
   sorry
   
end isosceles_triangle_l743_743131


namespace range_of_a_l743_743512

def p (x : ℝ) : Prop := x ≤ 1/2 ∨ x ≥ 1

def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

def not_q (x a : ℝ) : Prop := x < a ∨ x > a + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, not_q x a → p x) ∧ (∃ x : ℝ, ¬ (p x → not_q x a)) →
  0 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l743_743512


namespace find_x_of_equation_l743_743865

theorem find_x_of_equation (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := by
  sorry

end find_x_of_equation_l743_743865


namespace isosceles_triangle_l743_743112

theorem isosceles_triangle (ABC : Type) [triangle ABC]
  (I : incenter ABC) (α : incircle ABC)
  (P Q : α) (circAIC : circumcircle AIC)
  (h1 : P ∈ circAIC) (h2 : Q ∈ circAIC)
  (h3 : same_side P A (BI line))
  (h4 : other_side Q C (BI line))
  (M : midpoint_arc AC (α arc))
  (N : midpoint_arc BC (α arc))
  (h_par : PQ ∥ AC) : is_isosceles ABC := 
sorry

end isosceles_triangle_l743_743112


namespace derivative_at_2_l743_743797

noncomputable def f (x : ℝ) : ℝ := x^2 * deriv f 2 + 3 * x

theorem derivative_at_2 : deriv f 2 = -1 :=
sorry

end derivative_at_2_l743_743797


namespace isosceles_triangle_of_parallel_PQ_AC_l743_743137

variables {α β γ : Type}
variables {A B C I M N P Q : α} {circleIncircle circCircumcircleAIC : set α}
variables {lineBI linePQ lineAC : set α}

-- Given conditions from the problem
def incenter_of_triangle (I A B C : α) : Prop := sorry
def incircle (circle α : set α) (A B C I : α) : Prop := sorry
def circumcircle (circle circumcircleAIC : set α) (A I C : α) : Prop := sorry
def midpoint_of_arc (M N : α) (circle α : set α) (A B C : α) : Prop := sorry
def parallel (linePQ lineAC : set α) : Prop := sorry
def lies_on_same_side_of_line (P A : α) (line lineBI : set α) : Prop := sorry
def lies_on_opposite_sides_of_line (Q C : α) (line lineBI : set α) : Prop := sorry

theorem isosceles_triangle_of_parallel_PQ_AC :
  incenter_of_triangle I A B C ∧
  incircle circleIncircle A B C I ∧
  circumcircle circCircumcircleAIC A I C ∧
  midpoint_of_arc M circleIncircle A C ∧
  midpoint_of_arc N circleIncircle B C ∧
  lies_on_same_side_of_line P A lineBI ∧
  lies_on_opposite_sides_of_line Q C lineBI ∧
  parallel linePQ lineAC →
  (triangle_isosceles A B C) := sorry

end isosceles_triangle_of_parallel_PQ_AC_l743_743137


namespace isosceles_triangle_l743_743118

theorem isosceles_triangle (ABC : Type) [triangle ABC]
  (I : incenter ABC) (α : incircle ABC)
  (P Q : α) (circAIC : circumcircle AIC)
  (h1 : P ∈ circAIC) (h2 : Q ∈ circAIC)
  (h3 : same_side P A (BI line))
  (h4 : other_side Q C (BI line))
  (M : midpoint_arc AC (α arc))
  (N : midpoint_arc BC (α arc))
  (h_par : PQ ∥ AC) : is_isosceles ABC := 
sorry

end isosceles_triangle_l743_743118


namespace C_pow_50_l743_743875

open Matrix

def C : Matrix (Fin 2) (Fin 2) ℝ :=
![![3, 1], ![-4, -1]]

theorem C_pow_50 :
  (C ^ 50) = ![![101, 50], ![-200, -99]] :=
by
  sorry

end C_pow_50_l743_743875


namespace problem_proof_l743_743785

theorem problem_proof
    {f : ℝ → ℝ}
    (f_def : ∀ x, f x = sqrt (2 * |x - 3| - |x| - m))
    (domain_cond : ∀ x : ℝ, 2 * |x - 3| - |x| - m ≥ 0)
    (a b c : ℝ)
    (sum_squares : a^2 + b^2 + c^2 = 9) :
    (∀ m : ℝ, (∀ x : ℝ, 2 * |x - 3| - |x| - m ≥ 0) ↔ m ∈ set.Iic (-3)) ∧
    (∃ u v w : ℝ, 
        (u = a^2 + 1) ∧ 
        (v = b^2 + 2) ∧ 
        (w = c^2 + 3) ∧ 
        (u + v + w = 15) ∧ 
        (∀ a b c : ℝ, a^2 + b^2 + c^2 = 9 → 3/5 = infi (λ a b c, (1/(a^2+1) + 1/(b^2+2) + 1/(c^2+3)))) :=
by
  -- sorry is used here to indicate that the proof is omitted.
  sorry

end problem_proof_l743_743785


namespace constant_term_expansion_l743_743783

noncomputable def root_term_expansion (x a : ℝ) : ℝ := (Real.sqrt x + a / Real.sqrt x)^6

theorem constant_term_expansion (a : ℝ) (h : ∃ (c : ℝ), c = 12 ∧ (coeff_of_x2 (root_term_expansion x a) = c)) :
  constant_term (root_term_expansion x 2) = 160 :=
by
  sorry

end constant_term_expansion_l743_743783


namespace length_of_each_piece_is_correct_l743_743817

noncomputable def rod_length : ℝ := 38.25
noncomputable def num_pieces : ℕ := 45
noncomputable def length_each_piece_cm : ℝ := 85

theorem length_of_each_piece_is_correct : (rod_length / num_pieces) * 100 = length_each_piece_cm :=
by
  sorry

end length_of_each_piece_is_correct_l743_743817


namespace find_minimal_abs_diff_l743_743820

theorem find_minimal_abs_diff :
  ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ (c * d - 5 * c + 6 * d = 245) ∧ (|c - d| = 29) :=
sorry

end find_minimal_abs_diff_l743_743820


namespace angle_PQM_15_degrees_l743_743637

-- Definitions of points and their equilateral properties
variable (J K L M P Q : Type)

-- Hypotheses
variable [Square J K L M]
variable [EquilateralTriangle J M P]
variable [EquilateralTriangle M L Q]

-- Statement of the problem
theorem angle_PQM_15_degrees (J K L M P Q : Type) [Square J K L M] [EquilateralTriangle J M P] [EquilateralTriangle M L Q] :
  angle P Q M = 15 := 
sorry

end angle_PQM_15_degrees_l743_743637


namespace smallest_multiple_1_10_is_2520_l743_743995

noncomputable def smallest_multiple_1_10 : ℕ :=
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10))))))))

theorem smallest_multiple_1_10_is_2520 : smallest_multiple_1_10 = 2520 :=
  sorry

end smallest_multiple_1_10_is_2520_l743_743995


namespace sum_first_three_terms_geometric_sequence_general_term_formula_sum_of_bn_terms_l743_743767

noncomputable def a_n (n : ℕ) : ℝ := n + 1

def S_n (n : ℕ) : ℝ := (n * (2 + n + 1)) / 2
def T_n (n : ℕ) : ℝ := (n - 1) * 2^(n + 1) + 2 

theorem sum_first_three_terms :
  S_n 3 = 9 := by
  sorry

theorem geometric_sequence (a_1 a_3 a_7 : ℝ) :
  a_1 * a_7 = a_3^2 := by
  sorry

theorem general_term_formula :
  ∀ n, (∃ (d : ℝ), d ≠ 0 ∧ a_n n = n + 1) := by
  sorry

theorem sum_of_bn_terms:
  ∀ b_n (n : ℕ),
    b_n = (a_n n - 1) * 2^n →
    (T_n n = ∑ i in finset.range n, b_n i) :=
  by
  sorry

end sum_first_three_terms_geometric_sequence_general_term_formula_sum_of_bn_terms_l743_743767


namespace area_of_curvilinear_trapezoid_l743_743701

-- Define the function f(x) = 1/x
def f (x : ℝ) : ℝ := 1 / x

-- Define the statement to prove the area under the curve from x = 1 to x = 2 is ln(2)
theorem area_of_curvilinear_trapezoid :
  ∫ x in (1 : ℝ)..2, f x = Real.log 2 :=
sorry

end area_of_curvilinear_trapezoid_l743_743701


namespace parallelogram_area_proof_l743_743852

-- Define the conditions of the problem
variable (AD BM : ℝ)
variable (cos_BAM : ℝ)

-- Specify the given values
def AD_value : AD = 5 := by sorry
def BM_value : BM = 2 := by sorry
def cos_BAM_value : cos_BAM = 4/5 := by sorry

-- Define the necessary constructions and the area computation
def area_parallelogram (AD BM : ℝ) (cos_BAM : ℝ) : ℝ :=
  let AM := BM * real.cot (real.arccos cos_BAM)
  let AB := real.sqrt (AM^2 + BM^2)
  let sin_BAD := 2 * (real.sin (real.arccos cos_BAM)) * cos_BAM
  AD * AB * sin_BAD

-- Statement that verifies the area is 16
theorem parallelogram_area_proof :
  area_parallelogram 5 2 (4/5) = 16 := by
    sorry


end parallelogram_area_proof_l743_743852


namespace correct_population_estimation_statement_l743_743722

-- Definitions directly from the problem conditions
def sample_result_population_result : Prop := "The sample result is the population result"
def larger_sample_size_more_precise_estimation : Prop := "The larger the sample size, the more precise the estimation"
def stddev_sample_reflects_avg_state : Prop := "The standard deviation of the sample can approximately reflect the average state of the population"
def larger_variance_more_stable : Prop := "The larger the variance, the more stable the data"

-- Question rewritten as a theorem in Lean
theorem correct_population_estimation_statement
  (A : ¬ sample_result_population_result)
  (B : larger_sample_size_more_precise_estimation)
  (C : ¬ stddev_sample_reflects_avg_state)
  (D : ¬ larger_variance_more_stable) :
  larger_sample_size_more_precise_estimation :=
B

end correct_population_estimation_statement_l743_743722


namespace marias_workday_ends_at_six_pm_l743_743894

theorem marias_workday_ends_at_six_pm :
  ∀ (start_time : ℕ) (work_hours : ℕ) (lunch_start_time : ℕ) (lunch_duration : ℕ) (afternoon_break_time : ℕ) (afternoon_break_duration : ℕ) (end_time : ℕ),
    start_time = 8 ∧
    work_hours = 8 ∧
    lunch_start_time = 13 ∧
    lunch_duration = 1 ∧
    afternoon_break_time = 15 * 60 + 30 ∧  -- Converting 3:30 P.M. to minutes
    afternoon_break_duration = 15 ∧
    end_time = 18  -- 6:00 P.M. in 24-hour format
    → end_time = 18 :=
by
  -- map 13:00 -> 1:00 P.M.,  15:30 -> 3:30 P.M.; convert 6:00 P.M. back 
  sorry

end marias_workday_ends_at_six_pm_l743_743894


namespace floor_sum_distances_l743_743265

theorem floor_sum_distances (DR DK RK : ℝ) (hDR : DR = 13) (hDK : DK = 14) (hRK : RK = 15) :
  ∃ (E : ℝ × ℝ), let DE := (E.1^2 + E.2^2).sqrt,
                  RE := ( (E.1 - 9)^2 + (E.2 - 12)^2 ).sqrt,
                  KE := ( (E.1 - 14)^2 + E.2^2 ).sqrt in
  ⌊DE + RE + KE⌋ = 24 :=
by
  use (9, 15 / 4)
  sorry

end floor_sum_distances_l743_743265


namespace piggy_bank_balance_l743_743916

theorem piggy_bank_balance (original_amount : ℕ) (taken_out : ℕ) : original_amount = 5 ∧ taken_out = 2 → original_amount - taken_out = 3 :=
by sorry

end piggy_bank_balance_l743_743916


namespace sum_of_roots_cubic_polynomial_l743_743936

theorem sum_of_roots_cubic_polynomial (a b c d : ℝ) (h_poly: ∀ x : ℝ, a * (x^3 + 2 * x)^3 + b * (x^3 + 2 * x)^2 + c * (x^3 + 2 * x) + d ≥ a * (x^2 + 2)^3 + b * (x^2 + 2)^2 + c * (x^2 + 2) + d) :
  ∑ r in ((polynomial.C a * X^3 + polynomial.C b * X^2 + polynomial.C c * X + polynomial.C d).roots : finset ℝ), r = -b/a :=
sorry

end sum_of_roots_cubic_polynomial_l743_743936


namespace smallest_bob_number_l743_743691

def prime_factors (n : ℕ) : set ℕ := { p | p.prime ∧ p ∣ n }

def smallest_number_with_prime_factors (S : set ℕ) : ℕ :=
  S.to_finset.prod id

theorem smallest_bob_number :
  let alice_number := 45
  let bob_number := 15
  prime_factors alice_number = prime_factors bob_number →
  bob_number = 15 :=
by
  intros alice_number bob_number hp
  have h1 : alice_number = 45 := rfl
  have h2 : bob_number = 15 := rfl
  sorry

end smallest_bob_number_l743_743691


namespace al_initial_amount_l743_743690

theorem al_initial_amount
  (a b c : ℕ)
  (h₁ : a + b + c = 2000)
  (h₂ : 3 * a + 2 * b + 2 * c = 3500) :
  a = 500 :=
sorry

end al_initial_amount_l743_743690


namespace triangle_inradius_circumradius_l743_743687

theorem triangle_inradius_circumradius :
  ∀ (a b c : ℕ), (a = 7) → (b = 24) → (c = 25) → (a^2 + b^2 = c^2) →
  (let A := (a * b) / 2) → 
  (let s := (a + b + c) / 2) →
  (let r := A / s) → 
  (let R := c / 2) →
  r = 3 ∧ R = 12.5 :=
by
  intros a b c ha hb hc h_right A_def s_def r_def R_def
  sorry

end triangle_inradius_circumradius_l743_743687


namespace find_num_O_atoms_l743_743660

def atomic_weight_Ca := 40.08
def atomic_weight_O := 16.00
def atomic_weight_H := 1.008
def molecular_weight := 74
def num_H_atoms := 2
def num_Ca_atoms := 1

theorem find_num_O_atoms (x : ℕ) (h : molecular_weight = atomic_weight_Ca + atomic_weight_O * x + atomic_weight_H * num_H_atoms) : x = 2 :=
  by sorry

end find_num_O_atoms_l743_743660


namespace proof_sqrt_77_integers_product_l743_743716

def sqrt_77_integers_product : Prop :=
  let n : ℕ := 77 in
  let m₁ : ℕ := 8 in
  let m₂ : ℕ := 9 in
  m₁ * m₂ = 72 ∧ (m₁ * m₁ < n ∧ n < m₂ * m₂)

theorem proof_sqrt_77_integers_product : sqrt_77_integers_product := 
by
  sorry

end proof_sqrt_77_integers_product_l743_743716


namespace no_prime_p_satisfies_l743_743563

theorem no_prime_p_satisfies :
  ¬ ∃ p : ℕ, prime p ∧ (2 * p^3 + 8 * p^2 + 6 * p + 9 = 10 * p^2 + 1 * p + 5) :=
by 
  sorry

end no_prime_p_satisfies_l743_743563


namespace max_radian_of_sector_perimeter_l743_743949

noncomputable def sector_perimeter_max_radian (r : ℝ) (l : ℝ) : Prop :=
r > 0 ∧ r < 2 ∧ l + 2 * r = 4 ∧
(∃ (S : ℝ), S = 0.5 * l * r ∧ ∀ r' : ℝ, r' > 0 → r' < 2 → (let l' := 4 - 2 * r' in 0.5 * l' * r' ≤ S)) ∧
l / r = 2

theorem max_radian_of_sector_perimeter : ∃ α : ℝ, α = 2 :=
exist.elim (λ (r l : ℝ), sector_perimeter_max_radian r l) sorry

end max_radian_of_sector_perimeter_l743_743949


namespace tiffany_bags_l743_743613

theorem tiffany_bags (bags_monday : ℕ) (bags_next_day : ℕ) (h_monday : bags_monday = 8) (h_next_day : bags_next_day = 7) :
  bags_monday - bags_next_day = 1 :=
by
  rw [h_monday, h_next_day]
  norm_num
  sorry

end tiffany_bags_l743_743613


namespace plane_triangle_coverage_l743_743218

noncomputable def percentage_triangles_covered (a : ℝ) : ℝ :=
  let total_area := (4 * a) ^ 2
  let triangle_area := 10 * (1 / 2 * a^2)
  (triangle_area / total_area) * 100

theorem plane_triangle_coverage (a : ℝ) :
  abs (percentage_triangles_covered a - 31.25) < 0.75 :=
  sorry

end plane_triangle_coverage_l743_743218


namespace proof_problem_l743_743372

-- Definitions of the sequences and their properties
def arithmetic_seq (a : ℕ → ℝ) :=
∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

def geometric_seq (g : ℕ → ℝ) :=
∀ n : ℕ, g (n + 1) / g n = g (n + 2) / g (n + 1)

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) :=
∑ i in finset.range n, a (i + 1)

noncomputable def b_seq (S : ℕ → ℝ) (n : ℕ) :=
2 / S n

noncomputable def sum_first_n_terms_b (b : ℕ → ℝ) (n : ℕ) :=
∑ i in finset.range n, b (i + 1)

theorem proof_problem :
  ∀ a : ℕ → ℝ,
    (a 1 = 1) →
    (arithmetic_seq a) →
    (geometric_seq (λ n, a (2 * n + 1))) →
    (∀ n, a n = n) ∧
    (let S := sum_first_n_terms a in
     let b := b_seq S in
     let T := sum_first_n_terms_b b in
     ∀ n, T n = 4 * n / (n + 1)) :=
by
  intro a ha1 ha_seq hg_seq
  have d : ℝ := 1 -- The common difference, derived from the provided conditions
  sorry

end proof_problem_l743_743372


namespace CE_plus_DE_squared_eq_98_l743_743879

-- Define the members and conditions of the problem
variables {O A B C D E : Point}
variable {r : ℝ} -- radius
variable {E_on_AB : Point_on_Line E A B} -- E on line AB
variable {CD_on_circle : Chord_on_Circle C D O r} 
variable {BE : Segment B E 3} -- BE = 3
variable {angle_EAC : ∠ A E C = 30} -- ∠ AEC = 30 degrees

-- The given problem as a Lean theorem statement
theorem CE_plus_DE_squared_eq_98 
  (circle : Circle Center O r)
  (diameter : Diameter A B ∈ circle)
  (chord : Chord C D ∈ circle)
  (E_intersect : Intersect_At E A B C D)
  (BE_len : Segment B E = 3)
  (angle_AEC : ∠ A E C = 30) :
  CE^2 + DE^2 = 98 := 
sorry

end CE_plus_DE_squared_eq_98_l743_743879


namespace cos_beta_l743_743765

theorem cos_beta {α β : ℝ} (h1 : α > 0) (h2 : α < π/2) (h3 : β > 0) (h4 : β < π/2)
  (h5 : cos α = 3/5) (h6 : cos (α + β) = -5/13) : cos β = 33/65 :=
by
  sorry

end cos_beta_l743_743765


namespace number_of_elements_begin_with_1_l743_743079

noncomputable def T : finset ℕ := finset.filter (λ x, ∃ k, x = 3^k ∧ 0 ≤ k ∧ k ≤ 1500) (finset.range (3^1501))

theorem number_of_elements_begin_with_1 :
  (finset.filter (λ x, x.digits 10).head = 1 T).card = 230 := sorry

end number_of_elements_begin_with_1_l743_743079


namespace isosceles_triangle_of_parallel_PQ_AC_l743_743142

variables {α β γ : Type}
variables {A B C I M N P Q : α} {circleIncircle circCircumcircleAIC : set α}
variables {lineBI linePQ lineAC : set α}

-- Given conditions from the problem
def incenter_of_triangle (I A B C : α) : Prop := sorry
def incircle (circle α : set α) (A B C I : α) : Prop := sorry
def circumcircle (circle circumcircleAIC : set α) (A I C : α) : Prop := sorry
def midpoint_of_arc (M N : α) (circle α : set α) (A B C : α) : Prop := sorry
def parallel (linePQ lineAC : set α) : Prop := sorry
def lies_on_same_side_of_line (P A : α) (line lineBI : set α) : Prop := sorry
def lies_on_opposite_sides_of_line (Q C : α) (line lineBI : set α) : Prop := sorry

theorem isosceles_triangle_of_parallel_PQ_AC :
  incenter_of_triangle I A B C ∧
  incircle circleIncircle A B C I ∧
  circumcircle circCircumcircleAIC A I C ∧
  midpoint_of_arc M circleIncircle A C ∧
  midpoint_of_arc N circleIncircle B C ∧
  lies_on_same_side_of_line P A lineBI ∧
  lies_on_opposite_sides_of_line Q C lineBI ∧
  parallel linePQ lineAC →
  (triangle_isosceles A B C) := sorry

end isosceles_triangle_of_parallel_PQ_AC_l743_743142


namespace sum_of_first_15_terms_l743_743254

theorem sum_of_first_15_terms (a d : ℝ) 
  (h : (a + 3 * d) + (a + 11 * d) = 24) : 
  (15 / 2) * (2 * a + 14 * d) = 180 :=
by
  sorry

end sum_of_first_15_terms_l743_743254


namespace shaded_area_correct_l743_743848

def unit_triangle_area : ℕ := 10

def small_shaded_area : ℕ := unit_triangle_area

def medium_shaded_area : ℕ := 6 * unit_triangle_area

def large_shaded_area : ℕ := 7 * unit_triangle_area

def total_shaded_area : ℕ :=
  small_shaded_area + medium_shaded_area + large_shaded_area

theorem shaded_area_correct : total_shaded_area = 110 := 
  by
    sorry

end shaded_area_correct_l743_743848


namespace tank_filling_time_l743_743903

theorem tank_filling_time :
  (pipe_a_time = 30) →
  (pipe_b_time = pipe_a_time / 5) →
  (1 / ((1 / pipe_a_time) + (1 / pipe_b_time)) = 5) :=
by
  -- Define conditions
  assume pipe_a_time_eq : pipe_a_time = 30,
  assume pipe_b_time_eq : pipe_b_time = pipe_a_time / 5,
  -- First define the rates
  let pipe_a_rate := 1 / pipe_a_time,
  let pipe_b_rate := 1 / pipe_b_time,
  -- The rest of the proof can be constructed based on these definitions
  sorry

end tank_filling_time_l743_743903


namespace func_properties_in_interval_l743_743799

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2

theorem func_properties_in_interval :
  (∀ (x ∈ Icc (-1 : ℝ) 1), f x ≤ 0) ∧
  (¬ ∃ x ∈ Ico (-1 : ℝ) 1, ∀ y ∈ Ico (-1 : ℝ) 1, f y ≤ f x) ∧
  (∃ x ∈ Ico (-1 : ℝ) 1, f x = -9/4) :=
by
  sorry

end func_properties_in_interval_l743_743799


namespace solve_z_l743_743016

open Complex

theorem solve_z (z : ℂ) (h : z^2 = 3 - 4 * I) : z = 1 - 2 * I ∨ z = -1 + 2 * I :=
by
  sorry

end solve_z_l743_743016


namespace Jessie_daily_distance_proof_l743_743470

-- Definitions based on conditions
def Jackie_daily_distance : ℕ := 2
def Jessie_daily_distance (x : ℕ) := x
def Jackie_days : ℕ := 6
def Jessie_days : ℕ := 6

-- Condition that Jackie walks 3 miles more than Jessie in 6 days
axiom condition : Jackie_daily_distance * Jackie_days = Jessie_daily_distance Jessie_days + 3

-- The proof problem
theorem Jessie_daily_distance_proof (x : ℕ) (h : Jackie_daily_distance * Jackie_days = Jessie_daily_distance x * Jessie_days + 3) : x = 1.5 :=
by sorry

end Jessie_daily_distance_proof_l743_743470


namespace proof_problem_l743_743304

variables {O P E F A B C : Point}
variable {r : ℝ}
variable [circle : Circle O r]
variable [tangent_PE : IsTangent P E O]
variable [tangent_PF : IsTangent P F O]
variable [secant_PAB : IsSecant P A B O]
variable [secant_intersections : IntersectsAt P A B E F C]
variable [tangent_equal : PE = PF]

theorem proof_problem :
  2 / dist P C = 1 / dist P A + 1 / dist P B := sorry

end proof_problem_l743_743304


namespace rat_to_chihuahua_ratio_is_six_to_one_l743_743308

noncomputable def chihuahuas_thought_to_be : ℕ := 70
noncomputable def actual_rats : ℕ := 60

theorem rat_to_chihuahua_ratio_is_six_to_one
    (h : chihuahuas_thought_to_be - actual_rats = 10) :
    actual_rats / (chihuahuas_thought_to_be - actual_rats) = 6 :=
by
  sorry

end rat_to_chihuahua_ratio_is_six_to_one_l743_743308


namespace sum_interior_angles_of_regular_polygon_l743_743077

theorem sum_interior_angles_of_regular_polygon :
  ∀ (Q : Type) (n : ℕ) (interior_angle exterior_angle : ℕ),
    (∀ i : ℕ, i < n → interior_angle = 9 * exterior_angle) →
    (∀ i : ℕ, i < n → (interior_angle + exterior_angle = 180) ∧ (n * exterior_angle = 360)) →
    (n ≠ 0) →
    (let R := n * interior_angle in R = 3240) := 
by
  intros Q n interior_angle exterior_angle h_angle_relation h_polygon_relation h_n_ne_zero
  let R := n * interior_angle
  sorry

end sum_interior_angles_of_regular_polygon_l743_743077


namespace reciprocal_of_repeating_decimal_three_l743_743983

noncomputable def repeating_decimal_to_fraction : ℚ :=
  let x := (0.33333333333 : ℚ) in 1 / 3

theorem reciprocal_of_repeating_decimal_three : 
  (1 / repeating_decimal_to_fraction) = 3 := by
  -- Reciprocal of the fraction
  sorry

end reciprocal_of_repeating_decimal_three_l743_743983


namespace expectation_of_Y_l743_743791

open ProbabilityTheory

noncomputable def Y : Distribution :=
{ support := {0, 1, 2},
  pmf := λ y,
    if y = 0 then 1 / 4
    else if y = 1 then 1 / 4
    else if y = 2 then 1 / 2
    else 0,
  sum_pmf' := by
    simp only [Finset.sum_insert, Finset.mem_singleton, not_false_iff, Finset.sum_singleton]
    norm_num }

theorem expectation_of_Y :
  ∑ y in {0, 1, 2}, y * Y.pmf y = 5 / 4 := sorry

end expectation_of_Y_l743_743791


namespace simplify_expression_l743_743556

theorem simplify_expression : (15625 = 5^6) → (sqrt (√(1/15625) ^ (1/3)) = sqrt (5) / 5) := by
  intro h
  rw [h]
  sorry

end simplify_expression_l743_743556


namespace square_roots_of_x_l743_743011

theorem square_roots_of_x (a x : ℝ) 
    (h1 : (2 * a - 1) ^ 2 = x) 
    (h2 : (-a + 2) ^ 2 = x)
    (hx : 0 < x) 
    : x = 9 ∨ x = 1 := 
by sorry

end square_roots_of_x_l743_743011


namespace min_value_of_fraction_l743_743775

theorem min_value_of_fraction (a b : ℝ) (h_nonneg_a : 0 < a) (h_nonneg_b : 0 < b)
  (h_sum : a + b = 1) : 
  ∃ x, x = 3 + 2 * Real.sqrt 2 ∧ (∀ y, y = 1 / a + 2 / b → x ≤ y) :=
by
  use 3 + 2 * Real.sqrt 2
  intro y hy
  -- The proof would generally follow here
  sorry

end min_value_of_fraction_l743_743775


namespace angle_parallel_lines_l743_743391

variables {a b m n : Type} [linearOrder a] [linearOrder b] [linearOrder m] [linearOrder n]

def parallel (x y : Type) [linearOrder x] [linearOrder y] : Prop := 
  let h : x = y := sorry; -- Definition placeholder
  h = h

def angle_between (x y : Type) [linearOrder x] [linearOrder y] : Type :=
  let θ : Type := sorry; -- Placeholder for angle type
  θ

theorem angle_parallel_lines 
  (a m b n : Type) [linearOrder a] [linearOrder m] [linearOrder b] [linearOrder n]
  (h1 : parallel a m) (h2 : parallel b n) :
  angle_between a b = angle_between m n := 
sorry

end angle_parallel_lines_l743_743391


namespace find_ff2_l743_743359

def f (x : ℝ) : ℝ :=
  if x < 2 then 2 * Real.exp (x - 1)
  else Real.logBase 3 (x^2 - 1)

theorem find_ff2 : f (f 2) = 2 :=
sorry

end find_ff2_l743_743359


namespace cubical_box_edge_length_l743_743273

noncomputable def edge_length_of_box_in_meters : ℝ :=
  let number_of_cubes := 999.9999999999998
  let edge_length_cube_cm := 10
  let volume_cube_cm := edge_length_cube_cm^3
  let total_volume_box_cm := volume_cube_cm * number_of_cubes
  let total_volume_box_meters := total_volume_box_cm / (100^3)
  (total_volume_box_meters)^(1/3)

theorem cubical_box_edge_length :
  edge_length_of_box_in_meters = 1 := 
sorry

end cubical_box_edge_length_l743_743273


namespace oddly_powerful_less_than_500_l743_743312

def is_oddly_powerful (n : ℕ) : Prop :=
  ∃ (a b : ℕ), b > 1 ∧ b % 2 = 1 ∧ a ^ b = n

def num_oddly_powerful_less_than (m : ℕ) : ℕ :=
  Nat.card {n | n < m ∧ is_oddly_powerful n}

theorem oddly_powerful_less_than_500 : num_oddly_powerful_less_than 500 = 9 :=
sorry

end oddly_powerful_less_than_500_l743_743312


namespace point_E_divides_BC_in_1_3_l743_743439

-- Definitions of points and ratios
variables {A B C F G E : Type} [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq F] [DecidableEq G] [DecidableEq E]
variables (A B C F G E : Point)

-- Conditions
def condition1 : divides_side (A, C) 1 2 F := sorry
def condition2 : midpoint B F G := sorry
def condition3 : intersects (line AC) (line BG) E := sorry

-- The statement to prove
theorem point_E_divides_BC_in_1_3
  (h1 : divides_side (A, C) 1 2 F)
  (h2 : midpoint B F G)
  (h3 : intersects (line AG) (line BC) E) :
  ratio (B, E, C) = 1 / 3 :=
sorry

end point_E_divides_BC_in_1_3_l743_743439


namespace hourly_wage_l743_743483

variables (x : ℝ)

-- Conditions
def price_of_tv : ℝ := 1700
def weekly_work_hours : ℕ := 30
def weeks_in_month : ℕ := 4
def monthly_work_hours : ℕ := weekly_work_hours * weeks_in_month
def additional_hours : ℕ := 50
def total_hours_worked : ℕ := monthly_work_hours + additional_hours

-- Theorem
theorem hourly_wage : x * total_hours_worked = price_of_tv → x = 10 := 
begin
  sorry
end

end hourly_wage_l743_743483


namespace joan_apples_l743_743869

theorem joan_apples (original : ℕ) (given : ℕ) (final : ℕ) 
  (h1 : original = 43)
  (h2 : given = 27)
  (h3 : final = original - given) :
  final = 16 :=
by
  have h4 : 43 - 27 = 16, by norm_num,
  rw [h1, h2] at h3,
  exact eq.trans h3 h4

end joan_apples_l743_743869


namespace dance_students_count_l743_743033

theorem dance_students_count (total_students art_students music_percentage : ℕ) (music_students : ℕ)
    (H1 : total_students = 400)
    (H2 : art_students = 200)
    (H3 : music_percentage = 20)
    (H4 : music_students = (music_percentage * total_students / 100)) :
    ∃ D : ℕ, D + art_students + music_students = total_students ∧ D = 120 :=
by
    have H5 : music_students = 80 := by 
        simp [H1, H3, H4]
    use 120
    split
    . simp [H1, H2, H5]
    . refl

end dance_students_count_l743_743033


namespace burattino_awake_fraction_l743_743698

theorem burattino_awake_fraction (x : ℝ) (hx : 0 < x) :
  (∃ y : ℝ, y = x / 3 ∧ 
  ((x / 2) + (x - ((x / 2) + y))) / x = 2 / 3) :=
by
  use (x / 3)
  split
  { sorry } -- proof that y = x / 3
  have awake_distance := (x / 2) + (x - ((x / 2) + (x / 3)))
  have total_distance := x
  have fraction_awake := awake_distance / total_distance
  rw [awake_distance, total_distance]
  exact fraction_awake = 2 / 3

end burattino_awake_fraction_l743_743698


namespace sam_money_left_l743_743544

-- Assuming the cost per dime and quarter
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Given conditions
def dimes : ℕ := 19
def quarters : ℕ := 6
def cost_per_candy_bar_in_dimes : ℕ := 3
def candy_bars : ℕ := 4
def lollipops : ℕ := 1

-- Calculate the initial money in cents
def initial_money : ℕ := (dimes * dime_value) + (quarters * quarter_value)

-- Calculate the cost of candy bars in cents
def candy_bars_cost : ℕ := candy_bars * cost_per_candy_bar_in_dimes * dime_value

-- Calculate the cost of lollipops in cents
def lollipop_cost : ℕ := lollipops * quarter_value

-- Calculate the total cost of purchases in cents
def total_cost : ℕ := candy_bars_cost + lollipop_cost

-- Calculate the final money left in cents
def final_money : ℕ := initial_money - total_cost

-- Theorem to prove
theorem sam_money_left : final_money = 195 := by
  sorry

end sam_money_left_l743_743544


namespace find_t_l743_743149

theorem find_t (t : ℝ) (A B : ℝ × ℝ) :
  A = (2 * t - 5, -2) →
  B = (-3, 3 * t + 2) →
  (let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in (M.1 - A.1)^2 + (M.2 - A.2)^2 = 2 * t^2) →
  t = (-16 + Real.sqrt 156) / 5 ∨ t = (-16 - Real.sqrt 156) / 5 :=
by
  intros hA hB hM
  sorry

end find_t_l743_743149


namespace smurf_team_count_l743_743261

-- Definitions for the problem conditions
def n : ℕ := 12
def disliked (a b : ℕ) : Prop := (a % n = (b + 1) % n) ∨ (a % n = (b - 1) % n)

def valid_team (team : Finset ℕ) : Prop :=
  team.card = 5 ∧ (∀ a b ∈ team, ¬ disliked a b)

-- The theorem statement to prove the number of valid teams is 36
theorem smurf_team_count : (Finset.filter valid_team (Finset.powerset (Finset.range n))).card = 36 :=
sorry

end smurf_team_count_l743_743261


namespace prime_factors_count_l743_743005

theorem prime_factors_count :
  ∀ (a b c d : ℕ), 
  (prime a ∧ a = 83) ∧
  (∃ x y, prime x ∧ prime y ∧ b = x * y ∧ x = 5 ∧ y = 17) ∧
  (∃ u v, prime u ∧ prime v ∧ c = u * v ∧ u = 3 ∧ v = 29) ∧
  (prime d ∧ d = 89) →
  ∃ s : Finset ℕ, s = {3, 5, 17, 29, 83, 89} ∧ s.card = 6 :=
by
  sorry

end prime_factors_count_l743_743005


namespace probability_of_exactly_one_selected_l743_743458

-- Define the context and the conditions.
variables (A B : Type) [Finite A] [Finite B]
variables {volunteers : Fin 5}

-- Define a representation for selecting 2 volunteers from 5.
noncomputable def select_two_volunteers (s : Finset (Fin 5)) : ℕ :=
  Finset.card (s.filter (λ x, x ∈ volunteers))

-- Define the event where exactly one of A and B is selected.
noncomputable def exactly_one_selected (s : Finset (Fin 5)) (A B : Fin 5) : Prop :=
  (A ∈ s ∧ B ∉ s) ∨ (A ∉ s ∧ B ∈ s)

-- Main theorem stating the probability.
theorem probability_of_exactly_one_selected :
  let total_ways := Nat.choose 5 2, 
      favorable_ways := 3 + 3 in
  (favorable_ways / total_ways : ℚ) = 3 / 5 :=
by
  sorry

end probability_of_exactly_one_selected_l743_743458


namespace partition_set_property_l743_743146

theorem partition_set_property (k : ℕ) (hk : 0 < k) :
  ∃ (x y : Finset ℕ), 
    x ∪ y = Finset.range (2^(k+1)) ∧ x ∩ y = ∅ ∧ 
    (∀ m ∈ (Finset.range (k + 1)).filter (λ n, n > 0),
      Finset.sum (x.to_finset : Finset ℕ) (λ i, i ^ m) = Finset.sum (y.to_finset : Finset ℕ) (λ i, i ^ m)) :=
sorry

end partition_set_property_l743_743146


namespace sum_of_consecutive_even_integers_divisible_by_three_l743_743959

theorem sum_of_consecutive_even_integers_divisible_by_three (n : ℤ) : 
  ∃ p : ℤ, Prime p ∧ p = 3 ∧ p ∣ (n + (n + 2) + (n + 4)) :=
by 
  sorry

end sum_of_consecutive_even_integers_divisible_by_three_l743_743959


namespace isosceles_triangle_of_parallel_PQ_AC_l743_743138

variables {α β γ : Type}
variables {A B C I M N P Q : α} {circleIncircle circCircumcircleAIC : set α}
variables {lineBI linePQ lineAC : set α}

-- Given conditions from the problem
def incenter_of_triangle (I A B C : α) : Prop := sorry
def incircle (circle α : set α) (A B C I : α) : Prop := sorry
def circumcircle (circle circumcircleAIC : set α) (A I C : α) : Prop := sorry
def midpoint_of_arc (M N : α) (circle α : set α) (A B C : α) : Prop := sorry
def parallel (linePQ lineAC : set α) : Prop := sorry
def lies_on_same_side_of_line (P A : α) (line lineBI : set α) : Prop := sorry
def lies_on_opposite_sides_of_line (Q C : α) (line lineBI : set α) : Prop := sorry

theorem isosceles_triangle_of_parallel_PQ_AC :
  incenter_of_triangle I A B C ∧
  incircle circleIncircle A B C I ∧
  circumcircle circCircumcircleAIC A I C ∧
  midpoint_of_arc M circleIncircle A C ∧
  midpoint_of_arc N circleIncircle B C ∧
  lies_on_same_side_of_line P A lineBI ∧
  lies_on_opposite_sides_of_line Q C lineBI ∧
  parallel linePQ lineAC →
  (triangle_isosceles A B C) := sorry

end isosceles_triangle_of_parallel_PQ_AC_l743_743138


namespace number_construction_l743_743534

theorem number_construction (n : ℕ) : 
∃ (A : set (finset (fin 2))), 
  (∀ a ∈ A, a.card = 2^n) ∧ 
  (A.card = 2^(n + 1)) ∧
  (∀ (x y ∈ A), x ≠ y → (x ∪ y \ x ∩ y).card ≥ 2^(n - 1)) := 
sorry

end number_construction_l743_743534


namespace ratio_a_b_c_l743_743592

theorem ratio_a_b_c (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : (a + b + c) / 3 = 42) (h5 : a = 28) : 
  ∃ y z : ℕ, a / 28 = 1 ∧ b / (ky) = 1 / k ∧ c / (kz) = 1 / k ∧ (b + c) = 98 :=
by sorry

end ratio_a_b_c_l743_743592


namespace intersection_point_of_lines_l743_743814

theorem intersection_point_of_lines :
  ∃ x y : ℝ, (x - 2 * y - 4 = 0) ∧ (x + 3 * y + 6 = 0) ∧ (x = 0) ∧ (y = -2) :=
by
  sorry

end intersection_point_of_lines_l743_743814


namespace eggs_per_person_l743_743161

theorem eggs_per_person :
  ∀ (total_eggs : ℕ) (number_of_people : ℕ),
  total_eggs = 24 → number_of_people = 4 →
  total_eggs / number_of_people = 6 :=
by
  intros total_eggs number_of_people h1 h2
  rw [h1, h2]
  norm_num
  sorry

end eggs_per_person_l743_743161


namespace smallest_k_exists_l743_743348

theorem smallest_k_exists : ∃ (k : ℕ), k > 0 ∧ (∃ (n m : ℕ), n > 0 ∧ m > 0 ∧ k = 19^n - 5^m) ∧ k = 14 :=
by 
  sorry

end smallest_k_exists_l743_743348


namespace sara_height_is_45_l743_743547

variable (Mark_height Roy_height Joe_height Sara_height : ℕ)

def Mark_height_def : Mark_height = 34 := sorry
def Roy_height_def : Roy_height = Mark_height + 2 := sorry
def Joe_height_def : Joe_height = Roy_height + 3 := sorry
def Sara_height_def : Sara_height = Joe_height + 6 := sorry

theorem sara_height_is_45 : Sara_height = 45 := 
by
  have hMark : Mark_height = 34 := Mark_height_def
  have hRoy : Roy_height = Mark_height + 2 := Roy_height_def
  have hJoe : Joe_height = Roy_height + 3 := Joe_height_def
  have hSara : Sara_height = Joe_height + 6 := Sara_height_def
  rw [hMark] at hRoy
  rw [hRoy] at hJoe
  rw [hJoe] at hSara
  calc
    Sara_height
        = Joe_height + 6 := by rw [hSara]
    ... = (Roy_height + 3) + 6 := by rw [hJoe]
    ... = ((Mark_height + 2) + 3) + 6 := by rw [hRoy]
    ... = ((34 + 2) + 3) + 6 := by rw [hMark]
    ... = 45 := by norm_num

end sara_height_is_45_l743_743547


namespace area_COB_l743_743905

def point (α : Type) := (α × α)

def C (p : ℝ) : point ℝ := (0, p)
def O : point ℝ := (0, 0)
def B : point ℝ := (15, 0)

def height (C O : point ℝ) : ℝ := 
  let (x1, y1) := C
  let (x2, y2) := O
  y1 - y2 

def base (O B : point ℝ) : ℝ :=
  let (x1, y1) := O
  let (x2, y2) := B
  x2 - x1

def area_triangle (height base : ℝ) : ℝ := 
  0.5 * base * height

theorem area_COB (p : ℝ) (h : 0 ≤ p ∧ p ≤ 18) : 
  area_triangle p 15 = 15 * p / 2 :=
by
  sorry

end area_COB_l743_743905


namespace cone_lateral_area_eq_l743_743762

theorem cone_lateral_area_eq {R h : ℝ} (hR : R = 2) (hh : h = 1) :
  let l := Real.sqrt (R^2 + h^2) in
  let S := Real.pi * R * l in
  S = 2 * Real.sqrt 5 * Real.pi := by
  sorry

end cone_lateral_area_eq_l743_743762


namespace randy_needs_6_packs_l743_743917

theorem randy_needs_6_packs (wipes_per_pack daily_usage total_days : ℕ)
  (h_wipes : wipes_per_pack = 120)
  (h_daily : daily_usage = 2)
  (h_days : total_days = 360) : 
  (total_days / (wipes_per_pack / daily_usage) = 6) :=
by
  rw [h_wipes, h_daily, h_days]
  sorry

end randy_needs_6_packs_l743_743917


namespace baseball_fans_count_l743_743029

theorem baseball_fans_count
  (Y M R : ℕ) 
  (h1 : Y = (3 * M) / 2)
  (h2 : R = (5 * M) / 4)
  (hM : M = 104) :
  Y + M + R = 390 :=
by
  sorry 

end baseball_fans_count_l743_743029


namespace union_of_A_and_B_l743_743369

open Set

def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {1, 2, 4}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3, 4} :=
  sorry

end union_of_A_and_B_l743_743369


namespace isosceles_triangle_l743_743111

noncomputable theory

open_locale classical

variables {A B C I P Q : Type*}

-- Let \( I \) be the incenter of triangle \( ABC \)
def is_incenter (I A B C : Type*) : Prop := sorry

-- Let \( \alpha \) be its incircle
def is_incircle (α : Type*) (I A B C : Type*) : Prop := sorry

-- The circumcircle of triangle \( AIC \) intersects \( \alpha \) at points \( P \) and \( Q \)
def circumcircle_intersect_incircle (α A I C P Q : Type*) : Prop := sorry

-- \( P \) and \( A \) lie on the same side of line \( BI \), and \( Q \) and \( C \) lie on the other side
def same_side (P A Q C : Type*) (BI : Type*) : Prop := sorry

-- \( PQ \parallel AC \)
def parallel (PQ AC : Type*) : Prop := sorry

-- Define triangle is isosceles
def is_isosceles (A B C : Type*) : Prop := sorry

theorem isosceles_triangle 
  (I : Type*) (A B C P Q M N : Type*)
  (α : Type*)
  (h_incenter : is_incenter I A B C)
  (h_incircle : is_incircle α I A B C)
  (h_intersect : circumcircle_intersect_incircle α A I C P Q)
  (h_sameside : same_side P A Q C (line BI))
  (h_parallel : parallel PQ (line AC))
: is_isosceles A B C :=
sorry

end isosceles_triangle_l743_743111


namespace total_wings_l743_743062

-- Conditions
def money_per_grandparent : ℕ := 50
def number_of_grandparents : ℕ := 4
def bird_cost : ℕ := 20
def wings_per_bird : ℕ := 2

-- Calculate the total amount of money John received:
def total_money_received : ℕ := number_of_grandparents * money_per_grandparent

-- Determine the number of birds John can buy:
def number_of_birds : ℕ := total_money_received / bird_cost

-- Prove that the total number of wings all the birds have is 20:
theorem total_wings : number_of_birds * wings_per_bird = 20 :=
by
  sorry

end total_wings_l743_743062


namespace isosceles_triangle_l743_743108

noncomputable theory

open_locale classical

variables {A B C I P Q : Type*}

-- Let \( I \) be the incenter of triangle \( ABC \)
def is_incenter (I A B C : Type*) : Prop := sorry

-- Let \( \alpha \) be its incircle
def is_incircle (α : Type*) (I A B C : Type*) : Prop := sorry

-- The circumcircle of triangle \( AIC \) intersects \( \alpha \) at points \( P \) and \( Q \)
def circumcircle_intersect_incircle (α A I C P Q : Type*) : Prop := sorry

-- \( P \) and \( A \) lie on the same side of line \( BI \), and \( Q \) and \( C \) lie on the other side
def same_side (P A Q C : Type*) (BI : Type*) : Prop := sorry

-- \( PQ \parallel AC \)
def parallel (PQ AC : Type*) : Prop := sorry

-- Define triangle is isosceles
def is_isosceles (A B C : Type*) : Prop := sorry

theorem isosceles_triangle 
  (I : Type*) (A B C P Q M N : Type*)
  (α : Type*)
  (h_incenter : is_incenter I A B C)
  (h_incircle : is_incircle α I A B C)
  (h_intersect : circumcircle_intersect_incircle α A I C P Q)
  (h_sameside : same_side P A Q C (line BI))
  (h_parallel : parallel PQ (line AC))
: is_isosceles A B C :=
sorry

end isosceles_triangle_l743_743108


namespace reconstruct_quadrilateral_l743_743621

-- Define the condition that M, N, K, and L are points in a plane
variables {M N K L : Point}

-- To be proved: It is possible to reconstruct the quadrilateral ABCD
theorem reconstruct_quadrilateral (M N K L : Point) :
  ∃ (A B C D : Point), 
    is_projection_of_diagonal_intersection M N K L A B C D :=
sorry

end reconstruct_quadrilateral_l743_743621


namespace multiple_without_zero_digit_l743_743550

theorem multiple_without_zero_digit (n : ℕ) (h : n % 10 ≠ 0) : 
  ∃ m : ℕ, m % n = 0 ∧ ∀ d : ℕ, (d < 10 → (d ≠ 0 → ¬(m.to_digits 10).contains d)) :=
sorry

end multiple_without_zero_digit_l743_743550


namespace probability_two_boxes_empty_l743_743904

theorem probability_two_boxes_empty {B : Finset ℕ} (hB : B.card = 4) {b : Finset ℕ} (hb : b.card = 4) :
  (∃ l : Finset ℕ, l.card = 2) →
  (∃ k : Finset ℕ, k.card = 1) →
  let P := ((∃ E2 : Finset ℕ, E2.card = 2) * (84 : ℕ)) / ((144 + 84 + 4 : ℕ)) 
  in P = 21 / 58 :=
by
  sorry

end probability_two_boxes_empty_l743_743904


namespace inclination_of_l1_perpendicular_l1_l2_distance_between_parallel_lines_l743_743813
-- Import the necessary math library

-- Definitions based on conditions
def line1 (a : ℝ) : ℝ × ℝ → Prop := λ (x y : ℝ), a * x + y - 1 = 0
def line2 : ℝ × ℝ → Prop := λ (x y : ℝ), x - y - 3 = 0

-- Main theorem statements
theorem inclination_of_l1 (a : ℝ) (h : 1 = a) : a = -1 :=
sorry

theorem perpendicular_l1_l2 (a : ℝ) (h : 1 * -a = -1) : a = 1 :=
sorry

theorem distance_between_parallel_lines (a : ℝ) (h_parallel : a = 1)
  (x1 y1 x2 y2 : ℝ) (h_l1 : line1 a (x1, y1)) (h_l2 : line2 (x2, y2)) :
  (abs (-3 - (-1))) / (sqrt (a^2 + (1^2))) = 2 * sqrt 2 :=
sorry

end inclination_of_l1_perpendicular_l1_l2_distance_between_parallel_lines_l743_743813


namespace find_a1_l743_743494

section ArithmeticGeometricSequence

variables (a_1 : ℝ) (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)

-- Define the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℝ :=
  a_1 + (n - 1) * (-1)

-- Define the sum of the first n terms
def sum_first_n_terms (n : ℕ) : ℝ :=
  (n / 2) * (2 * a_1 + (n - 1) * (-1))

-- Define the given sums
def S_1 : ℝ := a_1
def S_2 : ℝ := 2 * a_1 - 1
def S_4 : ℝ := 4 * a_1 - 6

-- Define the geometric sequence condition
theorem find_a1 (h : S_2 ^ 2 = S_1 * S_4) : a_1 = -1/2 :=
by {
  sorry
}

end ArithmeticGeometricSequence

end find_a1_l743_743494


namespace isosceles_triangle_l743_743095

open_locale euclidean_geometry

variables {A B C I P Q M N : Point}
variables (α : circle)
variables (circumcircle_AIC : circle)

-- Condition 1
hypothesis h1 : incenter I (triangle.mk A B C)

-- Condition 2
hypothesis h2 : α = incircle (triangle.mk A B C)

-- Condition 3
hypothesis h3 : intersects circumcircle_AIC α P
hypothesis h4 : intersects circumcircle_AIC α Q

-- Condition 4
hypothesis h5 : same_side P A (line.mk B I)

-- Condition 5
hypothesis h6 : ¬ same_side Q C (line.mk B I)

-- Condition 6
hypothesis h7 : midpoint M (arc.mk α A C)

-- Condition 7
hypothesis h8 : midpoint N (arc.mk α B C)

-- Condition 8
hypothesis h9 : parallel (line.mk P Q) (line.mk A C)

-- Conclusion
theorem isosceles_triangle (h1 h2 h3 h4 h5 h6 h7 h8 h9) : (distance A B) = (distance A C) :=
sorry

end isosceles_triangle_l743_743095


namespace evaluate_expression_mod_17_l743_743332

theorem evaluate_expression_mod_17 :
  (2^(-2) + 2^(-3)) % 17 = 11 := by
  have h1 : (4 % 17) * 13 % 17 = 1 := by norm_num
  have h2 : (8 % 17) * 15 % 17 = 1 := by norm_num
  have inv_4_mod_17 : 4 % 17 = Nat.invMod 4 17 := by sorry -- inverse mod 17
  have inv_8_mod_17 : 8 % 17 = Nat.invMod 8 17 := by sorry -- inverse mod 17
  have eq1 : inv_4_mod_17 = 13 := by sorry -- Proof that they are equivalent
  have eq2 : inv_8_mod_17 = 15 := by sorry -- Proof that they are equivalent
  calc
    (2^(-2) + 2^(-3)) % 17 
        = (inv_4_mod_17 + inv_8_mod_17) % 17 : by rw [eq1, eq2]
    ... = (13 + 15) % 17 : by norm_num
    ... = 28 % 17 : by norm_num
    ... = 11 : by norm_num

end evaluate_expression_mod_17_l743_743332


namespace checkerboard_problem_l743_743651

def is_valid_square (size : ℕ) : Prop :=
  size = 4 ∨ size = 5 ∨ size = 6 ∨ size = 7 ∨ size = 8 ∨ size = 9 ∨ size = 10

def check_10_by_10 : ℕ :=
  24 + 36 + 25 + 16 + 9 + 4 + 1

theorem checkerboard_problem :
  ∀ size : ℕ, ( size = 4 ∨ size = 5 ∨ size = 6 ∨ size = 7 ∨ size = 8 ∨ size = 9 ∨ size = 10 ) →
  check_10_by_10 = 115 := 
sorry

end checkerboard_problem_l743_743651


namespace f_at_3_l743_743796

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 5

theorem f_at_3 (a b : ℝ) (h : f a b (-3) = -1) : f a b 3 = 11 :=
by
  sorry

end f_at_3_l743_743796


namespace min_shift_for_odd_function_l743_743209

theorem min_shift_for_odd_function 
  (φ : ℝ) (hφ : φ > 0) : 
  ∀ x : ℝ, 
    let shifted : ℝ → ℝ := λ x, 2 * sin (x + φ + π / 3) * cos (x + φ + π / 3) in 
    let odd_condition : Prop := ∀ x, shifted (-x) = -shifted x in 
    odd_condition ↔ φ = π / 6 :=
sorry

end min_shift_for_odd_function_l743_743209


namespace length_of_first_train_l743_743969

theorem length_of_first_train (
    speed_first_train_kmph : ℝ := 42,
    speed_second_train_kmph : ℝ := 30,
    length_second_train_m : ℝ := 280,
    time_clear_s : ℝ := 20.99832013438925
) : length_of_first_train = 139.97 :=
begin
    let speed_first_train_mps := speed_first_train_kmph * (1000 / 3600),
    let speed_second_train_mps := speed_second_train_kmph * (1000 / 3600),
    let relative_speed_mps := speed_first_train_mps + speed_second_train_mps,
    let distance_covered := relative_speed_mps * time_clear_s,
    let length_first_train := distance_covered - length_second_train_m,
    have h : length_first_train = 139.97,
    {
        sorry -- Proof omitted
    },
    exact h
end

end length_of_first_train_l743_743969


namespace homework_checked_on_friday_l743_743973

theorem homework_checked_on_friday
  (prob_no_check : ℚ := 1/2)
  (prob_check_on_friday_given_check : ℚ := 1/5)
  (prob_a : ℚ := 3/5)
  : 1/3 = prob_check_on_friday_given_check / prob_a :=
by
  sorry

end homework_checked_on_friday_l743_743973


namespace pedestrian_distance_after_two_minutes_l743_743289

theorem pedestrian_distance_after_two_minutes :
  ∀ (v : ℝ) (d₀ : ℝ) (l : ℝ) (t : ℝ), 
  v = 3.6 ∧ d₀ = 40 ∧ l = 6 ∧ t = 120 → -- Speed in km/h and converted time in seconds
  let s := v * (1000 / 3600) * t in -- Convert speed to m/s and calculate the distance
  d₀ + (s - l) = 74 :=
by
  intro v d₀ l t
  rintro ⟨hv, hd₀, hl, ht⟩
  let v_ms := v * (1000 / 3600) -- Convert speed to meters per second
  let distance_traveled := v_ms * t -- Calculate the distance traveled in meters
  let result := d₀ + (distance_traveled - l) -- Calculate the final distance from the crosswalk
  rw [hv, hd₀, hl, ht, ← real.rat.cast_sub, ← real.rat.cast_add],
  norm_num at *,
  exact ⟨⟩,
  sorry

end pedestrian_distance_after_two_minutes_l743_743289


namespace triangle_ABC_angles_l743_743073

noncomputable theory
open_locale classical

variables {α : Type*} [normed_group α] [normed_space ℝ α]

structure Triangle (α : Type*) [normed_group α] [normed_space ℝ α] :=
(A B C : α)

structure IsoscelesTriangle extends Triangle α :=
(M : α)
(h_iso : dist A M = dist C M)
(h_ac_acute : angle A M C < real.pi / 2)

structure AltitudinalTriangle extends Triangle α :=
(H : α)
(h_foot : ∃ k : ℝ, H = k • (A - B) + C)
(h_sym : A + B = 2 • M)
(h_alt : dist A H = dist H M)

theorem triangle_ABC_angles (t : AltitudinalTriangle α) :
  let ∠AMB := angle t.A t.M t.B,
      ∠AMC := angle t.A t.M t.C,
      ∠BAC := angle t.B t.A t.C,
      ∠BMC := angle t.B t.M t.C,
      ∠ABC := angle t.A t.B t.C,
      ∠BCA := angle t.B t.C t.A,
      ∠CAB := angle t.C t.A t.B
  in 
  t.h_sym ∧ t.h_alt ∧ t.h_iso ∧ ∠AMC < real.pi / 2 → 
  ∠ABC = real.pi / 6 ∧ ∠BCA = real.pi / 2 ∧ ∠CAB = real.pi / 3 :=
by sorry

end triangle_ABC_angles_l743_743073


namespace monotonic_decreasing_interval_l743_743584

noncomputable def f (x : ℝ) : ℝ := log 0.5 (x^2 - 4)

theorem monotonic_decreasing_interval :
  {x : ℝ | f x = log 0.5 (x^2 - 4)} = (Set.Ioi (2 : ℝ)) :=
by
  sorry

end monotonic_decreasing_interval_l743_743584


namespace reciprocal_of_repeating_decimal_three_l743_743982

noncomputable def repeating_decimal_to_fraction : ℚ :=
  let x := (0.33333333333 : ℚ) in 1 / 3

theorem reciprocal_of_repeating_decimal_three : 
  (1 / repeating_decimal_to_fraction) = 3 := by
  -- Reciprocal of the fraction
  sorry

end reciprocal_of_repeating_decimal_three_l743_743982


namespace find_max_value_of_z_l743_743886

def max_value_of_z (x y : ℝ) : ℝ := 2 * x + 5 * y

theorem find_max_value_of_z (x y : ℝ) (h1 : 6 ≤ x + y) (h2 : x + y ≤ 8) (h3 : -2 ≤ x - y) (h4 : x - y ≤ 0) :
  ∃ z, z = max_value_of_z x y ∧ z ≤ 8 :=
begin
  sorry,
end

end find_max_value_of_z_l743_743886


namespace correct_option_B_l743_743801

noncomputable def f (x : ℝ) : ℝ := |sin (2 * x - π / 6)|

theorem correct_option_B : 
  (∃ x, x = π / 3 ∧ (∀ x, f x = f (2 * π / 3 - x))) → 
  (∃! (a : ℝ), a = π / 3 ∧ (∀ x, f x = f (2 * π / 3 - x))) :=
by
  sorry

end correct_option_B_l743_743801


namespace moles_of_HCN_required_l743_743346

def balanced_reaction := "CuSO4 + 4 HCN → Cu(CN)2 + H2SO4"

theorem moles_of_HCN_required (moles_CuSO4 : ℕ) (moles_CuCN2 : ℕ) : 
  balanced_reaction ->
  moles_CuSO4 = 1 ->
  moles_CuCN2 = 1 ->
  ∃ (moles_HCN : ℕ), moles_HCN = 4 :=
by {
  sorry
}

end moles_of_HCN_required_l743_743346


namespace maximize_cargo_volume_l743_743971

noncomputable theory

def cargo_volume_optimization : Prop :=
  ∃ k b : ℝ,
  ∃ (M : ℝ),
  (4 * k + b = 16 ∧ 7 * k + b = 10) ∧
  (∀ x : ℝ, ∃ max_x : ℝ, ∃ max_y : ℝ, max_x = 6 ∧ max_y = 12 ∧ 
   ((y = -2 * x + 24) ∧ 
   (G = M * x * ((-2 * x) + 24)) ∧ 
   (G_max = 72 * M))))

theorem maximize_cargo_volume : cargo_volume_optimization :=
sorry

end maximize_cargo_volume_l743_743971


namespace monotonicity_intervals_range_of_a_for_extreme_points_l743_743394

-- Define the function f(x)
noncomputable def f (x a : ℝ) : ℝ := (Real.exp x / x) + a * (x - Real.log x)

-- Define the derivative of f(x)
noncomputable def f_deriv (x a : ℝ) : ℝ := ((Real.exp x * (x - 1)) / x^2) + a * (1 - 1 / x)

-- Lean statement 1: Intervals of monotonicity for f(x) given a > 0
theorem monotonicity_intervals (a : ℝ) (h : a > 0) : 
  (∀ x, (1 < x → f_deriv x a > 0) ∧ (x < 1 → f_deriv x a < 0)) :=
by
  sorry

-- Lean statement 2: Range of a for three distinct extreme points in (1/2, 2)
theorem range_of_a_for_extreme_points (a : ℝ) :
  (∃ x1 x2 x3 ∈ Ioo (1 / 2 : ℝ) 2, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f_deriv x1 a = 0 ∧ f_deriv x2 a = 0 ∧ f_deriv x3 a = 0) → 
  -2 * Real.sqrt Real.exp 1 < a ∧ a < -Real.exp 1 :=
by
  sorry

end monotonicity_intervals_range_of_a_for_extreme_points_l743_743394


namespace domain_f_l743_743738

-- Definition: function f(x) = tan(arcsin(x^3))
def f (x : ℝ) : ℝ := Real.tan (Real.arcsin (x ^ 3))

-- Condition: arcsin(x^3) is defined for x ∈ [-1, 1]
def valid_domain_arcsin (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1

-- Proof Problem: Prove that the domain of f(x) is (-1, 1)
theorem domain_f (x : ℝ) : valid_domain_arcsin x ↔ (x > -1 ∧ x < 1) := by
  sorry

end domain_f_l743_743738


namespace number_of_distinct_pairs_eq_three_l743_743587

theorem number_of_distinct_pairs_eq_three :
  ∀ (x y : ℕ), (x > 0 ∧ y > 0) → (x^4 * y^4 - 10 * x^2 * y^2 + 9 = 0) → 
  ∃! (x y : ℕ), (x, y) = (1, 1) ∨ (x, y) = (1, 3) ∨ (x, y) = (3, 1) :=
begin
  sorry
end

end number_of_distinct_pairs_eq_three_l743_743587


namespace isosceles_triangle_l743_743119

theorem isosceles_triangle (ABC : Type) [triangle ABC]
  (I : incenter ABC) (α : incircle ABC)
  (P Q : α) (circAIC : circumcircle AIC)
  (h1 : P ∈ circAIC) (h2 : Q ∈ circAIC)
  (h3 : same_side P A (BI line))
  (h4 : other_side Q C (BI line))
  (M : midpoint_arc AC (α arc))
  (N : midpoint_arc BC (α arc))
  (h_par : PQ ∥ AC) : is_isosceles ABC := 
sorry

end isosceles_triangle_l743_743119


namespace parallel_vectors_k_eq_neg1_l743_743877

theorem parallel_vectors_k_eq_neg1
  (k : ℤ)
  (a : ℤ × ℤ := (2 * k + 2, 4))
  (b : ℤ × ℤ := (k + 1, 8))
  (h : a.1 * b.2 = a.2 * b.1) :
  k = -1 :=
by
sorry

end parallel_vectors_k_eq_neg1_l743_743877


namespace smallest_k_for_a_n_digital_l743_743502

theorem smallest_k_for_a_n_digital (a n : ℕ) (h : 10^2013 ≤ a^n ∧ a^n < 10^2014) : 
  ∀ k : ℕ, (∀ b : ℕ, 10^(k-1) ≤ b → b < 10^k → (¬(10^2013 ≤ b^n ∧ b^n < 10^2014))) ↔ k = 2014 :=
by 
  sorry

end smallest_k_for_a_n_digital_l743_743502


namespace x_intercepts_cos_inverse_l743_743345

theorem x_intercepts_cos_inverse (a b : ℝ) (h₀ : a = 0.00005) (h₁ : b = 0.0005) :
  ∃ n : ℕ, n = 2862 ∧ (∀ x : ℝ, a < x ∧ x < b → cos (2 / x) = 0 →
  ∃ k : ℤ, x = 4 / ((2 * k + 1) * real.pi)) := sorry

end x_intercepts_cos_inverse_l743_743345


namespace exam_fraction_conditions_l743_743251

theorem exam_fraction_conditions (p : ℝ) (h1 : p > 0) (h2 : p < 1) :
  (∀ students problems : ℕ, 
    (students > 0) ∧ (problems > 0) →
    (let chall_problems := {i : ℕ // i < problems ∧ ¬∃ j : ℕ, j < students ∧ solved j i} in
    (chall_problems.card : ℝ) / (problems : ℝ) ≥ p) ∧
    (let good_students := {j : ℕ // j < students ∧ ∃ S:finset ℕ, S.card ≥ ⌊p * problems⌋ ∧ ∀ i ∈ S, solved j i } in
    (good_students.card : ℝ)/ (students : ℝ) ≥ p)) → 
  ((p = (2 : ℝ)/3) → (Possible : Prop)) ∧ 
  ((p = (3 : ℝ)/4) → (Impossible : Prop)) ∧ 
  ((p = (7 : ℝ)/10^7) → (Impossible : Prop)) := by
  intros
  sorry

end exam_fraction_conditions_l743_743251


namespace jason_flame_time_l743_743998

-- Define firing interval and flame duration
def firing_interval := 15
def flame_duration := 5

-- Define the function to calculate seconds per minute
def seconds_per_minute (interval : ℕ) (duration : ℕ) : ℕ :=
  (60 / interval) * duration

-- Theorem to state the problem
theorem jason_flame_time : 
  seconds_per_minute firing_interval flame_duration = 20 := 
by
  sorry

end jason_flame_time_l743_743998


namespace colors_used_l743_743300

theorem colors_used (total_blocks number_per_color : ℕ) (h1 : total_blocks = 196) (h2 : number_per_color = 14) : 
  total_blocks / number_per_color = 14 :=
by
  sorry

end colors_used_l743_743300


namespace part_a_part_b_part_c_l743_743834

noncomputable def area_of_rectangle : ℝ := 1
noncomputable def areas_of_figures : list ℝ := [1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2]
noncomputable def num_figures : ℕ := 5

theorem part_a :
  ∃ (i j : ℕ), 0 ≤ i < num_figures ∧ 0 ≤ j < num_figures ∧ i ≠ j ∧ intersect_area i j ≥ 3 / 20 :=
sorry

theorem part_b :
  ∃ (i j : ℕ), 0 ≤ i < num_figures ∧ 0 ≤ j < num_figures ∧ i ≠ j ∧ intersect_area i j ≥ 1 / 5 :=
sorry

theorem part_c :
  ∃ (i j k : ℕ), 0 ≤ i < num_figures ∧ 0 ≤ j < num_figures ∧ 0 ≤ k < num_figures ∧ i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ 
  intersect_area_3 i j k ≥ 1 / 20 :=
sorry

end part_a_part_b_part_c_l743_743834


namespace average_number_of_visitors_is_25_l743_743482

-- Define the sequence parameters
def a : ℕ := 10  -- First term
def d : ℕ := 5   -- Common difference
def n : ℕ := 7   -- Number of days

-- Define the sequence for the number of visitors on each day
def visitors (i : ℕ) : ℕ := a + (i - 1) * d

-- Define the average number of visitors
def avg_visitors : ℕ := (List.sum (List.map visitors [1, 2, 3, 4, 5, 6, 7])) / n

-- Prove the average
theorem average_number_of_visitors_is_25 : avg_visitors = 25 :=
by
  -- Placeholder for the actual proof
  sorry

end average_number_of_visitors_is_25_l743_743482


namespace probability_non_first_class_l743_743757

variable (A B C : Prop)
variable (P : Prop → ℝ)

-- Probabilities of events
axiom prob_A : P A = 0.65
axiom prob_B : P B = 0.2
axiom prob_C : P C = 0.1

-- Define the event of drawing a non-first-class product
def non_first_class : Prop := ¬A

-- Define the probability of non-first-class product
theorem probability_non_first_class :
  P(non_first_class) = 0.35 :=
by
  -- This is where the proof would go, but we're simply stating the theorem.
  sorry

end probability_non_first_class_l743_743757


namespace total_investment_l743_743619

-- Defining the conditions
def annual_income := 575
def first_investment := 3000
def first_interest_rate := 0.085
def second_interest_rate := 0.064

-- Defining the problem statement
theorem total_investment (S : ℝ) (T : ℝ) :
  S = 5000 ∧ T = 8000 ↔
  annual_income = (first_investment * first_interest_rate) + (S * second_interest_rate) ∧
  T = first_investment + S :=
by
  sorry

end total_investment_l743_743619


namespace platform_length_is_260_meters_l743_743686

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def time_to_cross_platform_s : ℝ := 30
noncomputable def time_to_cross_man_s : ℝ := 17

noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def length_of_train_m : ℝ := train_speed_mps * time_to_cross_man_s
noncomputable def total_distance_cross_platform_m : ℝ := train_speed_mps * time_to_cross_platform_s
noncomputable def length_of_platform_m : ℝ := total_distance_cross_platform_m - length_of_train_m

theorem platform_length_is_260_meters :
  length_of_platform_m = 260 := by
  sorry

end platform_length_is_260_meters_l743_743686


namespace poem_lines_months_l743_743938

theorem poem_lines_months (current_lines : ℕ) (target_lines : ℕ) (lines_per_month : ℕ) :
  current_lines = 24 →
  target_lines = 90 →
  lines_per_month = 3 →
  (target_lines - current_lines) / lines_per_month = 22 :=
  by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  exact sorry

end poem_lines_months_l743_743938


namespace sum_crossed_out_numbers_ge_one_l743_743766

theorem sum_crossed_out_numbers_ge_one (n : ℕ) (a : Fin n.succ → Fin n.succ → ℝ) (crossed_out : Fin n.succ → Fin n.succ)
  (h_a : ∀ i j, a i j = 1 / (i + j + 1))
  (h_crossed_out : ∀ i j k, i ≠ j → crossed_out i k ≠ crossed_out j k ∧ crossed_out i k ≠ crossed_out i j) :
  ∑ i, a i (crossed_out i) ≥ 1 :=
sorry

end sum_crossed_out_numbers_ge_one_l743_743766


namespace product_of_ds_is_8_l743_743958

noncomputable def product_of_ds : ℤ :=
  let ds := {d : ℤ | d > 0 ∧ ∃ k : ℤ, 49 - 12 * d = k^2} in
  Finset.prod (Finset.filter (λ d, d > 0 ∧ ∃ k : ℤ, 49 - 12 * d = k^2) (Finset.range 50)) id

theorem product_of_ds_is_8 : product_of_ds = 8 :=
by
  sorry

end product_of_ds_is_8_l743_743958


namespace turtle_total_population_l743_743962

-- Definitions of conditions.
noncomputable def percentage_common : ℝ := 0.50
noncomputable def percentage_rare : ℝ := 0.30
noncomputable def percentage_unique : ℝ := 0.15
noncomputable def percentage_legendary : ℝ := 0.05

noncomputable def percentage_female_common : ℝ := 0.60
noncomputable def percentage_female_rare : ℝ := 0.55
noncomputable def percentage_female_unique : ℝ := 0.45
noncomputable def percentage_female_legendary : ℝ := 0.40

noncomputable def ratio_striped_male_common : ℝ := 1 / 4
noncomputable def ratio_striped_male_rare : ℝ := 2 / 5
noncomputable def ratio_striped_male_unique : ℝ := 1 / 3
noncomputable def ratio_striped_male_legendary : ℝ := 1 / 2

noncomputable def percentage_baby_common : ℝ := 0.20
noncomputable def percentage_baby_rare : ℝ := 0.25
noncomputable def percentage_baby_unique : ℝ := 0.30
noncomputable def percentage_baby_legendary : ℝ := 0.35

noncomputable def known_adult_striped_male_common : ℝ := 70

-- The theorem to prove.
theorem turtle_total_population : 
  let
    total_striped_male_common := known_adult_striped_male_common / (1 - percentage_baby_common),
    total_male_common := total_striped_male_common / ratio_striped_male_common,
    total_common := total_male_common / (1 - percentage_female_common),
    total_turtles := total_common / percentage_common
  in
    total_turtles = 1760 :=
by 
  sorry

end turtle_total_population_l743_743962


namespace number_of_chocolate_bars_by_theresa_l743_743068

-- Define the number of chocolate bars and soda cans that Kayla bought
variables (C S : ℕ)

-- Assume the total number of chocolate bars and soda cans Kayla bought is 15
axiom total_purchased_by_kayla : C + S = 15

-- Define the number of chocolate bars Theresa bought as twice the number Kayla bought
def chocolate_bars_purchased_by_theresa := 2 * C

-- The theorem to prove
theorem number_of_chocolate_bars_by_theresa : chocolate_bars_purchased_by_theresa = 2 * C :=
by
  -- The proof is omitted as instructed
  sorry

end number_of_chocolate_bars_by_theresa_l743_743068


namespace arrange_books_l743_743843

-- Given conditions
def math_books_count := 4
def history_books_count := 6

-- Question: How many ways can the books be arranged given the conditions?
theorem arrange_books (math_books_count history_books_count : ℕ) :
  math_books_count = 4 → 
  history_books_count = 6 →
  ∃ ways : ℕ, ways = 51840 :=
by
  sorry

end arrange_books_l743_743843


namespace isosceles_triangle_l743_743117

theorem isosceles_triangle (ABC : Type) [triangle ABC]
  (I : incenter ABC) (α : incircle ABC)
  (P Q : α) (circAIC : circumcircle AIC)
  (h1 : P ∈ circAIC) (h2 : Q ∈ circAIC)
  (h3 : same_side P A (BI line))
  (h4 : other_side Q C (BI line))
  (M : midpoint_arc AC (α arc))
  (N : midpoint_arc BC (α arc))
  (h_par : PQ ∥ AC) : is_isosceles ABC := 
sorry

end isosceles_triangle_l743_743117


namespace Cartesian_C1_eq_C3_intersection_C1_C3_intersection_C2_l743_743041

noncomputable def parametric_C1_eq1 (t : ℝ) : ℝ := (2 + t) / 6
noncomputable def parametric_C1_eq2 (t : ℝ) : ℝ := real.sqrt t
noncomputable def parametric_C2_eq1 (s : ℝ) : ℝ := -(2 + s) / 6
noncomputable def parametric_C2_eq2 (s : ℝ) : ℝ := -real.sqrt s
noncomputable def polar_C3_eq (θ : ℝ) : ℝ := 2 * real.cos θ - real.sin θ

theorem Cartesian_C1_eq (x y t : ℝ) (ht : t = 6 * x - 2) : y^2 = 6 * x - 2 ↔ y = real.sqrt (6 * x - 2) :=
  sorry

theorem C3_intersection_C1 (x y : ℝ) (h1 : y = 2 * x) (h2 : y^2 = 6 * x - 2) :
  (x = 1/2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
  sorry

theorem C3_intersection_C2 (x y : ℝ) (h1 : y = 2 * x) (h2 : y^2 = -6 * x - 2) :
  (x = -1/2 ∧ y = -1) ∨ (x = -1 ∧ y = -2) :=
  sorry

end Cartesian_C1_eq_C3_intersection_C1_C3_intersection_C2_l743_743041


namespace simplify_fraction_l743_743553

theorem simplify_fraction :
  let x := 15625 in
  x = 5^6 → (√(√[3](√(1 / ↑x)))) = (√5 / 5) := 
by
  intro x h
  sorry

end simplify_fraction_l743_743553


namespace initial_deadline_l743_743168

theorem initial_deadline (W : ℕ) (R : ℕ) (D : ℕ) :
    100 * 25 * 8 = (1/3 : ℚ) * W →
    (2/3 : ℚ) * W = 160 * R * 10 →
    D = 25 + R →
    D = 50 := 
by
  intros h1 h2 h3
  sorry

end initial_deadline_l743_743168


namespace logs_from_cuts_l743_743915

   theorem logs_from_cuts (C P : ℕ) (hC : C = 10) (hP : P = 16) :
     ∃ L : ℕ, P = L + C ∧ L = 6 :=
   by
     -- Define the number of cuts and pieces
     have h_eq : P = L + C,
     -- Substitute the known values
     rw [hC, hP],
     -- Show there exists an L that satisfies the equation
     use (P - C),
     -- Prove L equals 6
     show (P - C) = 6,
     -- Perform the arithmetic
     rw [hC, hP],
     show (16 - 10) = 6,
     linarith,
     done -- Complete the theorem
   
end logs_from_cuts_l743_743915


namespace total_practice_hours_correct_l743_743665

-- Define the conditions
def daily_practice_hours : ℕ := 5 -- The team practices 5 hours daily
def missed_days : ℕ := 1 -- They missed practicing 1 day this week
def days_in_week : ℕ := 7 -- There are 7 days in a week

-- Calculate the number of days they practiced
def practiced_days : ℕ := days_in_week - missed_days

-- Calculate the total hours practiced
def total_practice_hours : ℕ := practiced_days * daily_practice_hours

-- Theorem to prove the total hours practiced is 30
theorem total_practice_hours_correct : total_practice_hours = 30 := by
  -- Start the proof; skipping the actual proof steps
  sorry

end total_practice_hours_correct_l743_743665


namespace convert_cylindrical_to_rectangular_l743_743318

-- Define the conversion formulas
def cylindrical_to_rectangular (r θ z : ℝ) : (ℝ × ℝ × ℝ) :=
  (r * Real.cos θ, r * Real.sin θ, z)

-- Given cylindrical coordinates
def given_cylindrical_coordinates : (ℝ × ℝ × ℝ) := (-3, 5 * Real.pi / 4, -7)

-- Expected rectangular coordinates
def expected_rectangular_coordinates : (ℝ × ℝ × ℝ) := (3 * Real.sqrt 2 / 2, 3 * Real.sqrt 2 / 2, -7)

-- Prove the conversion
theorem convert_cylindrical_to_rectangular :
  cylindrical_to_rectangular (-3) (5 * Real.pi / 4) (-7) = expected_rectangular_coordinates :=
by
  sorry

end convert_cylindrical_to_rectangular_l743_743318


namespace average_weasels_caught_per_week_l743_743836

-- Definitions based on the conditions
def initial_weasels : ℕ := 100
def initial_rabbits : ℕ := 50
def foxes : ℕ := 3
def rabbits_caught_per_week_per_fox : ℕ := 2
def weeks : ℕ := 3
def remaining_animals : ℕ := 96

-- Main theorem statement
theorem average_weasels_caught_per_week :
  (foxes * weeks * rabbits_caught_per_week_per_fox +
   foxes * weeks * W = initial_weasels + initial_rabbits - remaining_animals) →
  W = 4 :=
sorry

end average_weasels_caught_per_week_l743_743836


namespace determine_c_l743_743199

theorem determine_c (a b c : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : 0 < c)
  (hf : ∀ x, f x = (x - a) * (x - c))
  (hg : ∀ x, g x = (x - a) * (x - b) * (x - c))
  (hf0 : f 0 = -8)
  (hg0 : g 0 = -8)
  (hg_minus_a : g (-a) = 8) :
  c = 8 / 3 :=
by sorry

end determine_c_l743_743199


namespace sum_of_coefficients_l743_743430

-- Define the polynomial equation condition
def polynomial_eq (x : ℕ) (b : list ℕ) : Prop :=
  (4 * x - 2)^5 = (b.nth 5).getOrElse 0 * x^5 + (b.nth 4).getOrElse 0 * x^4 + 
    (b.nth 3).getOrElse 0 * x^3 + (b.nth 2).getOrElse 0 * x^2 + 
    (b.nth 1).getOrElse 0 * x + (b.nth 0).getOrElse 0

-- Statement to prove the sum of the coefficients equals 32
theorem sum_of_coefficients (b : list ℕ) (h : polynomial_eq 1 b) : 
  (b.nth 5).getOrElse 0 + (b.nth 4).getOrElse 0 + (b.nth 3).getOrElse 0 + 
  (b.nth 2).getOrElse 0 + (b.nth 1).getOrElse 0 + (b.nth 0).getOrElse 0 = 32 := 
by 
  sorry -- Proof not required, placeholder for proof

end sum_of_coefficients_l743_743430


namespace rationalized_sum_l743_743537

noncomputable def rationalize_denominator (x : ℝ) := 
  (4 * real.cbrt 49) / 21

theorem rationalized_sum : 
  let A := 4 in
  let B := 49 in
  let C := 21 in
  A + B + C = 74 :=
by
  let A := 4
  let B := 49
  let C := 21
  trivial

end rationalized_sum_l743_743537


namespace valid_k_l743_743215

theorem valid_k (k : ℕ) (h_pos : k ≥ 1) (h : 10^k - 1 = 9 * k^2) : k = 1 := by
  sorry

end valid_k_l743_743215


namespace ratio_cher_to_gab_l743_743441

-- Definitions based on conditions
def sammy_score : ℕ := 20
def gab_score : ℕ := 2 * sammy_score
def opponent_score : ℕ := 85
def total_points : ℕ := opponent_score + 55
def cher_score : ℕ := total_points - (sammy_score + gab_score)

-- Theorem to prove the ratio of Cher's score to Gab's score
theorem ratio_cher_to_gab : cher_score / gab_score = 2 := by
  sorry

end ratio_cher_to_gab_l743_743441


namespace checkerboard_problem_l743_743652

def is_valid_square (size : ℕ) : Prop :=
  size = 4 ∨ size = 5 ∨ size = 6 ∨ size = 7 ∨ size = 8 ∨ size = 9 ∨ size = 10

def check_10_by_10 : ℕ :=
  24 + 36 + 25 + 16 + 9 + 4 + 1

theorem checkerboard_problem :
  ∀ size : ℕ, ( size = 4 ∨ size = 5 ∨ size = 6 ∨ size = 7 ∨ size = 8 ∨ size = 9 ∨ size = 10 ) →
  check_10_by_10 = 115 := 
sorry

end checkerboard_problem_l743_743652


namespace angle_ABC_lt_60_l743_743914

variable {α : Type*} [EuclideanGeometry α]

open EuclideanGeometry Set 

theorem angle_ABC_lt_60 
  {A B C D : α} (h_cyclic : circle (set of points α A B C D)) 
  (h_AB_BD : AB = BD) (h_AC_BC : AC = BC) : ∠ ABC < 60° :=
by 
  sorry

end angle_ABC_lt_60_l743_743914


namespace find_value_l743_743428

theorem find_value (x : ℝ) (h : 3 * x + 2 = 11) : 5 * x - 3 = 12 :=
by
  sorry

end find_value_l743_743428


namespace race_time_difference_l743_743520

theorem race_time_difference 
  (distance : ℝ) 
  (malcolm_speed : ℝ) 
  (joshua_speed : ℝ) 
  (malcolm_time : ℝ := malcolm_speed * distance) 
  (joshua_time : ℝ := joshua_speed * distance) 
  : (distance = 12) 
    → (malcolm_speed = 5.5) 
    → (joshua_speed = 7.5) 
    → (joshua_time - malcolm_time = 24) :=
by
  intros h_distance h_malcolm_speed h_joshua_speed
  have h_malcolm_time : malcolm_time = 5.5 * 12 := by rw [h_malcolm_speed, h_distance, mul_comm]
  have h_joshua_time : joshua_time = 7.5 * 12 := by rw [h_joshua_speed, h_distance, mul_comm]
  rw [h_malcolm_time, h_joshua_time]
  norm_num
  
  sorry -- Skip the remainder of the proof.

end race_time_difference_l743_743520


namespace find_angle_l743_743337

theorem find_angle (a : ℝ) (h1 : 0 < a ∧ a < 360) (h2 : (cos a) * (cos (3 * a)) = (cos (2 * a))^2) : a = 180 :=
by
  sorry

end find_angle_l743_743337


namespace find_x_l743_743858

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (h : x + y + x * y = 143) : x = 15 :=
by sorry

end find_x_l743_743858


namespace prob_rain_both_days_correct_l743_743590

-- Definitions according to the conditions
def prob_rain_Saturday : ℝ := 0.4
def prob_rain_Sunday : ℝ := 0.3
def cond_prob_rain_Sunday_given_Saturday : ℝ := 0.5

-- Target probability to prove
def prob_rain_both_days : ℝ := prob_rain_Saturday * cond_prob_rain_Sunday_given_Saturday

-- Theorem statement
theorem prob_rain_both_days_correct : prob_rain_both_days = 0.2 :=
by
  sorry

end prob_rain_both_days_correct_l743_743590


namespace slope_of_tangent_line_l743_743744

theorem slope_of_tangent_line (x : ℝ) (y : ℝ) (h : x ^ 2 = 4 * y) (P : x = 2 ∧ y = 1) : 
  let dy_dx := (1 / 2) * x in
  dy_dx = 1 := 
by 
  -- Proof omitted
  sorry

end slope_of_tangent_line_l743_743744


namespace math_problem_l743_743509

open_locale big_operators

theorem math_problem (n : ℕ) (x : ℕ → ℝ) 
  (h_pos : ∀ i, 0 < x i)
  (h_sum : ∑ i in finset.range n, x i = 1 ) :
  ( (∑ i in finset.range n, real.sqrt (x i)) * 
    (∑ i in finset.range n, 1 / real.sqrt (1 + x i)) )
    ≤ n^2 / real.sqrt (n + 1) :=
  sorry

end math_problem_l743_743509


namespace cos_add_pi_over_4_tan_double_theta_l743_743370

noncomputable def cos_theta := 4 / 5
noncomputable def theta_range := (0 : Real) < θ ∧ θ < (Real.pi / 2)

theorem cos_add_pi_over_4 (θ : Real) (h₀ : cos θ = cos_theta) (h₁ : theta_range) : 
  cos (θ + Real.pi / 4) = Real.sqrt 2 / 10 := sorry

theorem tan_double_theta (θ : Real) (h₀ : cos θ = cos_theta) (h₁ : theta_range) : 
  tan (2 * θ) = 24 / 7 := sorry

end cos_add_pi_over_4_tan_double_theta_l743_743370


namespace anime_date_and_episodes_l743_743963

theorem anime_date_and_episodes :
  (∃ n N : ℕ, let days_until_april = 55, let total_episodes = 325 in 
  n = days_until_april ∧ N = total_episodes ∧
  N - 2 * n = 215 ∧ N - 5 * n = 50 ∧
  n = 55 ∧ N = 325) :=
by
  sorry

end anime_date_and_episodes_l743_743963


namespace no_winning_strategy_l743_743069

theorem no_winning_strategy : ∀ (grid : Fin 19 → Fin 19 → ℕ), 
  (∀ (i j : Fin 19), grid i j = 0 ∨ grid i j = 1) → 
  (∃ (A B : ℕ), A = Finset.sup Finset.univ (λ i, (Finset.sum Finset.univ (λ j, grid i j))) ∧
                 B = Finset.sup Finset.univ (λ j, (Finset.sum Finset.univ (λ i, grid i j))) ∧
                 A = B) := by
  sorry

end no_winning_strategy_l743_743069


namespace moles_of_KI_formed_l743_743741

-- Define the given conditions
def moles_KOH : ℕ := 1
def moles_NH4I : ℕ := 1
def balanced_equation (KOH NH4I KI NH3 H2O : ℕ) : Prop :=
  (KOH = 1) ∧ (NH4I = 1) ∧ (KI = 1) ∧ (NH3 = 1) ∧ (H2O = 1)

-- The proof problem statement
theorem moles_of_KI_formed (h : balanced_equation moles_KOH moles_NH4I 1 1 1) : 
  1 = 1 :=
by sorry

end moles_of_KI_formed_l743_743741


namespace maximum_perimeter_of_third_rectangle_l743_743230

theorem maximum_perimeter_of_third_rectangle:
  ∃ (rect : ℕ × ℕ), 
  let (w1, h1) := (70, 110) in
  let (w2, h2) := (40, 80) in
  let (w3, h3) := rect in
  ((w3 = 190 ∧ h3 = 40) ∨ (w3 = 110 ∧ h3 = 10) ∨ (w3 = 60 ∧ h3 = 80)) ∧
  2 * (w3 + h3) = 300 :=
sorry

end maximum_perimeter_of_third_rectangle_l743_743230


namespace value_is_100_l743_743649

theorem value_is_100 (number : ℕ) (h : number = 20) : 5 * number = 100 :=
by
  sorry

end value_is_100_l743_743649


namespace math_problem_proof_l743_743892

-- Variables and function definitions
variable (ω A B C a b c : ℝ)
variable (A_pos : 0 < A)
variable (A_le : A < 2 * π / 3)
variable (ω_pos : ω > 0)
noncomputable def f (x : ℝ) := (sqrt 3 * sin (ω * x) + cos (ω * x)) * cos (ω * x) - 1 / 2

-- The two conditions related to the period and the triangle
axiom period_condition : 4 * π = (2 * π) / (2 * ω)
axiom triangle_condition : (2 * a - c) * cos B = b * cos C

-- The theorem statement for the Lean proof
theorem math_problem_proof :
  (∃ k : ℤ, 4 * k * π - 4 * π / 3 ≤ A ∧ A ≤ 2 * π / 3 + 4 * k * π) ∧
  (cos B = 1 / 2 ∧ B = π / 3 ∧ (1 / 2 < sin (A / 2 + π / 6) ∧ sin (A / 2 + π / 6) < 1)) :=
sorry

end math_problem_proof_l743_743892


namespace breadth_of_plot_l743_743566

theorem breadth_of_plot (b l : ℝ) (h1 : l * b = 18 * b) (h2 : l - b = 10) : b = 8 :=
by
  sorry

end breadth_of_plot_l743_743566


namespace eventually_constant_sequence_l743_743648

theorem eventually_constant_sequence (n : ℕ) (h : n > 0) : 
  ∃ m K, ∀ k ≥ K, (∃! a_k ∈ fin k, a_k = m ∧ S_{k+1} = S_k + a_{k+1} ∧ (S_{k+1} % (k+1)) = 0) := 
sorry

end eventually_constant_sequence_l743_743648


namespace cannot_form_right_triangle_set_A_l743_743630

theorem cannot_form_right_triangle_set_A :
  ¬(2^2 + 3^2 = 4^2) ∧ 
  (3^2 + 4^2 = 5^2) ∧ 
  (6^2 + 8^2 = 10^2) ∧ 
  (1^2 + (Real.sqrt 2)^2 = (Real.sqrt 3)^2) :=
by {
  sorry,
}

end cannot_form_right_triangle_set_A_l743_743630


namespace reciprocal_of_recurring_three_l743_743993

noncomputable def recurring_three := 0.33333333333 -- approximation of 0.\overline{3}

theorem reciprocal_of_recurring_three :
  let x := recurring_three in
  (x = (1/3)) → (1 / x = 3) := 
by 
  sorry

end reciprocal_of_recurring_three_l743_743993


namespace at_least_one_success_l743_743299

-- Define probabilities for A, B, and C
def pA : ℚ := 1 / 2
def pB : ℚ := 2 / 3
def pC : ℚ := 4 / 5

-- Define the probability that none succeed
def pNone : ℚ := (1 - pA) * (1 - pB) * (1 - pC)

-- Define the probability that at least one of them succeeds
def pAtLeastOne : ℚ := 1 - pNone

theorem at_least_one_success : pAtLeastOne = 29 / 30 := 
by sorry

end at_least_one_success_l743_743299


namespace common_chord_line_max_distance_from_line_AB_common_chord_length_is_three_points_on_line_l743_743943

-- Conditions
def circle_Q1 (x y : ℝ) := x^2 + y^2 - 2*x
def circle_Q2 (x y : ℝ) := x^2 + y^2 + 2*x - 4*y

axiom intersection_points_A_B (x y : ℝ) :
  circle_Q1 x y = 0 ∧ circle_Q2 x y = 0

-- Properties to Prove
theorem common_chord_line (x y : ℝ) :
  circle_Q1 x y = 0 ∧ circle_Q2 x y = 0 → x = y :=
sorry

theorem max_distance_from_line_AB (P : ℝ × ℝ) :
  (∃ x y : ℝ, circle_Q1 x y = 0 ∧ P = (x, y)) →
  (∃ d : ℝ, d = 1 + (Real.sqrt 2 / 2)) :=
sorry

theorem common_chord_length_is (l : ℝ) :
  (∃ x y : ℝ, circle_Q1 x y = 0 ∧ circle_Q2 x y = 0) →
  l = Real.sqrt 2 :=
sorry

theorem three_points_on_line (x y : ℝ) (d : ℝ):
  circle_Q1 x y = 0 →
  d = 1 / 2 →
  ∃ a b c : ℝ × ℝ, on_circle_Q1 a ∧ on_circle_Q1 b ∧ on_circle_Q1 c ∧ 
  (distance_from_line a d) ∧ (distance_from_line b d) ∧ (distance_from_line c d) :=
sorry

end common_chord_line_max_distance_from_line_AB_common_chord_length_is_three_points_on_line_l743_743943


namespace ax5_by5_l743_743889

variables {a b x y : ℝ}

def s : ℕ → ℝ
| 1 := a * x + b * y
| 2 := a * x^2 + b * y^2
| 3 := a * x^3 + b * y^3
| 4 := a * x^4 + b * y^4
| n := 0 -- we only define it up to s_4 as per given conditions

theorem ax5_by5 :
  (a * x + b * y = 5) →
  (a * x^2 + b * y^2 = 9) →
  (a * x^3 + b * y^3 = 22) →
  (a * x^4 + b * y^4 = 60) →
  a * x^5 + b * y^5 = 97089 / 203 :=
by
  intros h1 h2 h3 h4
  sorry

end ax5_by5_l743_743889


namespace Cartesian_C1_eq_C3_intersection_C1_C3_intersection_C2_l743_743042

noncomputable def parametric_C1_eq1 (t : ℝ) : ℝ := (2 + t) / 6
noncomputable def parametric_C1_eq2 (t : ℝ) : ℝ := real.sqrt t
noncomputable def parametric_C2_eq1 (s : ℝ) : ℝ := -(2 + s) / 6
noncomputable def parametric_C2_eq2 (s : ℝ) : ℝ := -real.sqrt s
noncomputable def polar_C3_eq (θ : ℝ) : ℝ := 2 * real.cos θ - real.sin θ

theorem Cartesian_C1_eq (x y t : ℝ) (ht : t = 6 * x - 2) : y^2 = 6 * x - 2 ↔ y = real.sqrt (6 * x - 2) :=
  sorry

theorem C3_intersection_C1 (x y : ℝ) (h1 : y = 2 * x) (h2 : y^2 = 6 * x - 2) :
  (x = 1/2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
  sorry

theorem C3_intersection_C2 (x y : ℝ) (h1 : y = 2 * x) (h2 : y^2 = -6 * x - 2) :
  (x = -1/2 ∧ y = -1) ∨ (x = -1 ∧ y = -2) :=
  sorry

end Cartesian_C1_eq_C3_intersection_C1_C3_intersection_C2_l743_743042


namespace probability_two_dice_same_number_l743_743179

theorem probability_two_dice_same_number : 
  let dice_sides := 8 in
  let total_outcomes := dice_sides ^ 8 in
  let different_outcomes := (fact dice_sides) / (fact (dice_sides - 8)) in
  (1 - (different_outcomes / total_outcomes)) = (1291 / 1296) :=
by
  sorry

end probability_two_dice_same_number_l743_743179


namespace max_colored_cells_100x100_l743_743027

theorem max_colored_cells_100x100 : 
  ∀ (coloring : Fin 100 → Fin 100 → Bool),
  (∀ i j k, i ≠ j → coloring i k = true → coloring j k ≠ true) →
  (∀ i j k, j ≠ k → coloring i j = true → coloring i k ≠ true) →
  (∃ n ≤ 198, ∀ (i j : Fin 100), n = ∑ i, ∑ j, if coloring i j then 1 else 0) :=
sorry

end max_colored_cells_100x100_l743_743027


namespace no_russians_in_top_three_l743_743295

-- Definitions from conditions
def num_players : Nat := 11
def russians : Nat := 4
def foreigners : Nat := 7

-- For scores and point system
structure Player :=
(name : String)
(score : ℝ)

def is_russian (p : Player) : Prop :=
-- Dummy condition to specify a player is Russian
-- In practice, we would link each player instance to their nationality
p.name.starts_with "R"

def is_foreigner (p : Player) : Prop :=
¬ is_russian p

-- Condition of different scores among players
def distinct_scores (players : List Player) : Prop :=
players.map Player.score |>.nodup

-- Condition of points equality
def equal_points_russian_foreigners (players : List Player) : Prop :=
(players.filter is_russian).sum_by Player.score =
(players.filter is_foreigner).sum_by Player.score

-- Main theorem: Determine if no Russians are in the top three positions
theorem no_russians_in_top_three (all_players : List Player)
  (h1 : all_players.length = num_players)
  (h2 : all_players.filter is_russian).length = russians)
  (h3 : all_players.filter is_foreigner).length = foreigners)
  (h4 : distinct_scores all_players)
  (h5 : equal_points_russian_foreigners all_players) : 
  ¬ (∀ (p1 p2 p3 : Player), 
    List.mem p1 all_players → List.mem p2 all_players → List.mem p3 all_players → 
    p1.score > p2.score → p2.score > p3.score →
    is_foreigner p1 ∧ is_foreigner p2 ∧ is_foreigner p3) := sorry

end no_russians_in_top_three_l743_743295


namespace min_value_frac_sum_l743_743773

theorem min_value_frac_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  ∃ c, (∀ x y : ℝ, (0 < x) → (0 < y) → (x + y = 1) → (∃ c, c = 3 + 2*√2) → c = 3 + 2*√2) :=
sorry

end min_value_frac_sum_l743_743773


namespace multiply_seven_l743_743436

variable (x : ℕ)

theorem multiply_seven (h : 8 * x = 64) : 7 * x = 56 := by
  sorry


end multiply_seven_l743_743436


namespace XYZQuiz_l743_743674

theorem XYZQuiz :
  let S (c u : ℕ) : ℕ := 5 * c + 3 * u in
  let valid_combinations := 
    {S : ℕ // ∃ c u i : ℕ, 
      c + u + i = 30 ∧ 
      S = 5 * c + 3 * u ∧ 
      noncomputable_count (λ (c u : ℕ), 
          S = 5 * c + 3 * u ∧ 
          0 ≤ c ∧ 0 ≤ u ∧ c + u ≤ 30) = 3
    } in
    ∑ S in valid_combinations, S = 187.5 :=
begin
  sorry
end

end XYZQuiz_l743_743674


namespace inscribed_square_max_area_l743_743758

def maxInscribedSquareArea (a : ℝ) : ℝ :=
  4 * a^2 * (3 - 2 * Real.sqrt 2)

theorem inscribed_square_max_area (a : ℝ) : 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ 
  (4 * a^2 * (Real.sqrt 2 * Real.cos (Real.pi / 4 - x) - 1)^2 = maxInscribedSquareArea a) := 
sorry

end inscribed_square_max_area_l743_743758


namespace eccentricity_of_hyperbola_l743_743401

variables {a b c : ℝ}
variables (PF1 PF2 : ℝ)

def hyperbola (x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def eccentricity (a b c : ℝ) := c / a

theorem eccentricity_of_hyperbola (h_a : a > 0) (h_b : b > 0)
  (h_pf1 : PF1 = 16) (h_pf2 : PF2 = 12)
  (h_eq : 2 * c = real.sqrt (PF1^2 + PF2^2)) :
  eccentricity a b c = 5 :=
begin
  sorry
end

end eccentricity_of_hyperbola_l743_743401


namespace cube_ratio_l743_743275

theorem cube_ratio (V₁ V₂ : ℝ) (h₁ : V₁ = 216) (h₂ : V₂ = 1728) : 
  ∛V₂ / ∛V₁ = 2 :=
by
  sorry

end cube_ratio_l743_743275


namespace binary_110_eq_six_l743_743202

theorem binary_110_eq_six : 
  let bin : ℕ := 6 in
  to_nat 110 (λ n, 1) = bin :=
sorry

end binary_110_eq_six_l743_743202


namespace triangle_sides_and_angles_arithmetic_progression_l743_743034

noncomputable def m : ℕ := 0
noncomputable def n : ℕ := 67
noncomputable def p : ℕ := 0

theorem triangle_sides_and_angles_arithmetic_progression 
    (a b x : ℕ) 
    (ha : a = 7) 
    (hb : b = 9) 
    (hx : x = Int.sqrt 67)
    (angles_in_arithmetic_progression : (60 - e, 60, 60 + e)) :
    m + n + p = 67 := 
begin
  sorry
end

end triangle_sides_and_angles_arithmetic_progression_l743_743034


namespace unique_set_three_elements_l743_743491

open Real 

theorem unique_set_three_elements 
  (A : Set ℝ) 
  (hA : ∃ x y z, A = {x, y, z} ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z) 
  (B : Set ℝ) 
  (hB : B = {r + s | r ∈ A ∧ s ∈ A ∧ r ≠ s}) 
  (hB_values : B = {log 2 6, log 2 10, log 2 15}) : 
  A = {1, log 2 3, log 2 5} :=
sorry

end unique_set_three_elements_l743_743491


namespace C₁_eq_Cartesian_intersection_C₃_C₁_C₂_eq_Cartesian_intersection_C₃_C₂_l743_743039

-- Define the parametric equations for C1
def C₁ (t : ℝ) (ht : t ≥ 0) : ℝ × ℝ :=
  (⟨(2 + t) / 6, sqrt t⟩)

-- Define the Cartesian equation for C1
def C₁_Cartesian (x y : ℝ) : Prop :=
  y^2 = 6*x - 2 ∧ y ≥ 0

-- Define the parametric equations for C2
def C₂ (s : ℝ) (hs : s ≥ 0) : ℝ × ℝ :=
  (⟨-(2 + s) / 6, -sqrt s⟩)

-- Define the Cartesian equation for C2
def C₂_Cartesian (x y : ℝ) : Prop :=
  y^2 = -6*x - 2 ∧ y ≤ 0

-- Define the polar equation for C3
def C₃ (θ : ℝ) : Prop :=
  2*cos θ - sin θ = 0

-- Define the Cartesian coordinates conversion
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

-- Prove the Cartesian equation of C₁
theorem C₁_eq_Cartesian (t : ℝ) (ht : t ≥ 0) :
  ∃ y, C₁ t ht = ⟨(6* (((2 + t) / 6) : ℝ)) - 2, y^2⟩ := sorry

-- Prove the intersection points of C₃ with C₁
theorem intersection_C₃_C₁ :
  (∃ x y, C₃ (arctan y x) ∧ C₁_Cartesian x y ∧ (x = 1/2 ∧ y = 1 ∨ x = 1 ∧ y = 2)) := sorry

-- Prove the Cartesian equation of C₂
theorem C₂_eq_Cartesian (s : ℝ) (hs : s ≥ 0) :
  ∃ y, C₂ s hs = ⟨(6* ((-(2 + s) / 6) : ℝ)) + 2, y^2⟩ := sorry
  
-- Prove the intersection points of C₃ with C₂
theorem intersection_C₃_C₂ :
  (∃ x y, C₃ (arctan y x) ∧ C₂_Cartesian x y ∧ (x = -1/2 ∧ y = -1 ∨ x = -1 ∧ y = -2)) := sorry

end C₁_eq_Cartesian_intersection_C₃_C₁_C₂_eq_Cartesian_intersection_C₃_C₂_l743_743039


namespace count_rational_numbers_in_set_l743_743438

-- Define the set of numbers
def set_of_numbers : List ℚ := [22/7, 0, 2.02301001]

-- Define irrational numbers that are not in ℚ
def set_of_irrationals := (9^(1/3) : Real) ∉ ℚ ∧ (-Real.pi / 2) ∉ ℚ

-- The problem statement in Lean 4
theorem count_rational_numbers_in_set : 
  (List.length set_of_numbers = 3) ∧ set_of_irrationals := 
by
  sorry

end count_rational_numbers_in_set_l743_743438


namespace similar_triangle_perimeter_l743_743236

-- Definitions based on conditions:
def small_triangle_side1 := 12
def small_triangle_side2 := 12
def small_triangle_side3 := 18

def similar_triangle_shortest_side := 30

-- Given this, our goal is to prove the perimeter of the larger triangle is 120.
theorem similar_triangle_perimeter :
  let scale_factor := (similar_triangle_shortest_side / small_triangle_side1 : ℝ)
  let large_side1 := small_triangle_side1 * scale_factor
  let large_side2 := small_triangle_side2 * scale_factor
  let large_side3 := small_triangle_side3 * scale_factor in 
  large_side1 + large_side2 + large_side3 = 120 :=
by
  -- Placeholder for actual proof
  sorry

end similar_triangle_perimeter_l743_743236


namespace no_such_k_exists_l743_743507

noncomputable def problem (n : ℕ) (a b : Fin n → ℂ) : Prop :=
  ∀ k : Fin n, (∑ i, Complex.abs (a i - a k)) > (∑ i, Complex.abs (b i - a k))

theorem no_such_k_exists (n : ℕ) (a b : Fin n → ℂ) (h : ∀ k : Fin n, (∑ i, Complex.abs (a i - a k)) > (∑ i, Complex.abs (b i - a k))) :
  ¬ ∃ k : Fin n, (∑ i, Complex.abs (a i - a k)) ≤ (∑ i, Complex.abs (b i - a k)) :=
by
  intro h_contra
  rcases h_contra with ⟨k, hk⟩
  exact lt_irrefl _ (lt_of_le_of_lt hk (h k))

end no_such_k_exists_l743_743507


namespace regular_polygon_sides_l743_743432

theorem regular_polygon_sides (exterior_angle : ℕ) (h : exterior_angle = 30) : (360 / exterior_angle) = 12 := by
  sorry

end regular_polygon_sides_l743_743432


namespace solution_parabola_tangent_and_point_l743_743805

noncomputable def parabola_tangent_and_point : Prop :=
  ∃ (p : ℝ) (M : ℝ × ℝ),
    (0 < p) ∧
    (M.1 > 0) ∧ (M.2 = 0) ∧
    ∀ (y1 y2 : ℝ) (t : ℝ) (x1 x2 : ℝ),
      (y1^2 = 8 * x1) ∧ (y2^2 = 8 * x2) ∧ (x1 = t * y1 + M.1) ∧ (x2 = t * y2 + M.1) ∧
      (y1 + y2 = 8 * t) ∧ (y1 * y2 = -8 * M.1) ∧
      (let AM2 := (t^2 + 1) * y1^2 in
       let BM2 := (t^2 + 1) * y2^2 in
       (1 / AM2) + (1 / BM2) = (1 / (t^2 + 1)) * ((4 * t^2 + M.1) / (4 * M.1^2))
      ) ∧ M = (4, 0)

theorem solution_parabola_tangent_and_point : parabola_tangent_and_point :=
  sorry

end solution_parabola_tangent_and_point_l743_743805


namespace sqrt_fraction_simplified_l743_743334

theorem sqrt_fraction_simplified :
  Real.sqrt (4 / 3) = 2 * Real.sqrt 3 / 3 :=
by sorry

end sqrt_fraction_simplified_l743_743334


namespace heavens_brothers_erasers_l743_743410

theorem heavens_brothers_erasers :
  ∀ (total_money : ℕ) (sharpener_cost notebook_cost eraser_cost highlighter_spending : ℕ)
  (heaven_sharpeners heaven_notebooks : ℕ),
  total_money = 100 →
  sharpener_cost = 5 →
  notebook_cost = 5 →
  eraser_cost = 4 →
  highlighter_spending = 30 →
  heaven_sharpeners = 2 →
  heaven_notebooks = 4 →
  let heaven_spending := heaven_sharpeners * sharpener_cost + heaven_notebooks * notebook_cost in
  let remaining_money := total_money - heaven_spending in
  let remaining_after_highlighter := remaining_money - highlighter_spending in
  remaining_after_highlighter / eraser_cost = 10 :=
by
  intros total_money sharpener_cost notebook_cost eraser_cost highlighter_spending heaven_sharpeners heaven_notebooks
  intros h1 h2 h3 h4 h5 h6 h7
  let heaven_spending := heaven_sharpeners * sharpener_cost + heaven_notebooks * notebook_cost
  let remaining_money := total_money - heaven_spending
  let remaining_after_highlighter := remaining_money - highlighter_spending
  unfold heaven_spending remaining_money remaining_after_highlighter
  sorry

end heavens_brothers_erasers_l743_743410


namespace range_of_k_l743_743828

theorem range_of_k (k : ℝ) : (∃ x : ℝ, 2 * x - 5 * k = x + 4 ∧ x > 0) → k > -4 / 5 :=
by
  sorry

end range_of_k_l743_743828


namespace distance_A_to_C_l743_743235

def point (x : ℝ) (y : ℝ) : (ℝ × ℝ) := (x, y)

def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (p.1 + v.1, p.2 + v.2)

def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem distance_A_to_C :
  let A := point 2 (-2)
  let B := point 8 6
  let C := translate B (-3, 4)
  distance A C = real.sqrt 153 := by
sory

end distance_A_to_C_l743_743235


namespace no_primes_in_sequence_l743_743635

def Q : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53 * 59 * 61 * 67

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sequence_contains_prime (Q : ℕ) (n : ℕ) : Prop := is_prime (Q + n)

theorem no_primes_in_sequence : ∀ n, 2 ≤ n ∧ n ≤ 71 → ¬ sequence_contains_prime Q n :=
by {
  intro n h,
  sorry
}

end no_primes_in_sequence_l743_743635


namespace MathematicianChoosesFirstBarber_l743_743672

-- Define the conditions as hypotheses
def Barber1Dressed := false  -- 1st barber is poorly dressed, unshaven, sloppily groomed
def Barber1Clean := false    -- 1st barber's shop is dirty

def Barber2Dressed := true   -- 2nd barber is clean-shaven, impeccably dressed, neatly groomed
def Barber2Clean := true     -- 2nd barber's shop is clean

def BarbersCutEachOther : Prop := 
  ∀ (b1 b2 : Prop), (b1 = Barber1Dressed ∧ b2 = Barber2Dressed) →  -- The appearance of one barber is due to the other barber's skill
  (b1 ∨ b2) 

-- Define the conclusion
theorem MathematicianChoosesFirstBarber
  (h1 : Barber1Dressed = false)
  (h2 : Barber1Clean = false)
  (h3 : Barber2Dressed = true)
  (h4 : Barber2Clean = true) 
  (h5 : BarbersCutEachOther) 
  :
  ∃ barber, barber = "First Barber" := 
by
  trivial -- This 'trivial' statement acts as a placeholder indicating that the proof follows logically from the given facts.

end MathematicianChoosesFirstBarber_l743_743672


namespace max_sphere_radius_in_cube_l743_743052

-- Given a cube of side length 1
def cube_side_length : ℝ := 1

-- Define the length of the diagonal in the cube
def diagonal_length (a : ℝ) : ℝ := a * real.sqrt 3

-- Define the maximum radius of the sphere tangent to the diagonal
theorem max_sphere_radius_in_cube (a : ℝ) (h : a = 1) : 
  (∀ (R : ℝ), R = real.sqrt 3 / 3) :=
by sorry

end max_sphere_radius_in_cube_l743_743052


namespace polynomial_unique_solution_l743_743729

theorem polynomial_unique_solution (P : ℝ → ℝ) (h : ∀ x y z : ℝ, 
  x ≠ 0 → y ≠ 0 → z ≠ 0 → 2 * x * y * z = x + y + z → 
  (P(x) / (y * z) + P(y) / (z * x) + P(z) / (x * y) = P(x - y) + P(y - z) + P(z - x))) :
  ∃ c : ℝ, ∀ x : ℝ, P x = c * (x^2 + 3) :=
by
  sorry

end polynomial_unique_solution_l743_743729


namespace triangles_with_area_three_halves_l743_743597

noncomputable def side_length : ℝ :=
  2

noncomputable def total_points : ℕ :=
  20

theorem triangles_with_area_three_halves :
  ∃ (cube : ℝ) (points : ℕ), cube = side_length ∧ points = total_points ∧
  (∃ (triangles_with_area : ℕ), triangles_with_area = 96) :=
by
  obtain ⟨cube, points, h1, h2⟩ := ⟨side_length, total_points, rfl, rfl⟩,
  use [cube, points],
  exact (cube = ² ∧ points = 20 ∧ ∃ (triangles_with_area : ℕ), triangles_with_area = 96)

end triangles_with_area_three_halves_l743_743597


namespace fermats_little_theorem_l743_743017

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Prime p) (hgcd : gcd a p = 1) : (a^(p-1) - 1) % p = 0 := by
  sorry

end fermats_little_theorem_l743_743017


namespace isosceles_triangle_l743_743106

noncomputable theory

open_locale classical

variables {A B C I P Q : Type*}

-- Let \( I \) be the incenter of triangle \( ABC \)
def is_incenter (I A B C : Type*) : Prop := sorry

-- Let \( \alpha \) be its incircle
def is_incircle (α : Type*) (I A B C : Type*) : Prop := sorry

-- The circumcircle of triangle \( AIC \) intersects \( \alpha \) at points \( P \) and \( Q \)
def circumcircle_intersect_incircle (α A I C P Q : Type*) : Prop := sorry

-- \( P \) and \( A \) lie on the same side of line \( BI \), and \( Q \) and \( C \) lie on the other side
def same_side (P A Q C : Type*) (BI : Type*) : Prop := sorry

-- \( PQ \parallel AC \)
def parallel (PQ AC : Type*) : Prop := sorry

-- Define triangle is isosceles
def is_isosceles (A B C : Type*) : Prop := sorry

theorem isosceles_triangle 
  (I : Type*) (A B C P Q M N : Type*)
  (α : Type*)
  (h_incenter : is_incenter I A B C)
  (h_incircle : is_incircle α I A B C)
  (h_intersect : circumcircle_intersect_incircle α A I C P Q)
  (h_sameside : same_side P A Q C (line BI))
  (h_parallel : parallel PQ (line AC))
: is_isosceles A B C :=
sorry

end isosceles_triangle_l743_743106


namespace inequality_satisfaction_l743_743712

theorem inequality_satisfaction (k n : ℕ) (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (1 + y^n / x^k) ≥ ((1 + y)^n / (1 + x)^k) ↔ 
    (k = 0) ∨ (n = 0) ∨ (0 = k ∧ 0 = n) ∨ (k ≥ n - 1 ∧ n ≥ 1) :=
by sorry

end inequality_satisfaction_l743_743712


namespace area_of_square_field_l743_743567

-- Definitions
def cost_per_meter : ℝ := 1.40
def total_cost : ℝ := 932.40
def gate_width : ℝ := 1.0

-- Problem Statement
theorem area_of_square_field (s : ℝ) (A : ℝ) 
  (h1 : (4 * s - 2 * gate_width) * cost_per_meter = total_cost)
  (h2 : A = s^2) : A = 27889 := 
  sorry

end area_of_square_field_l743_743567


namespace polar_to_cartesian_coordinates_l743_743786

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_cartesian_coordinates :
  polar_to_cartesian 2 (2 / 3 * Real.pi) = (-1, Real.sqrt 3) :=
by
  sorry

end polar_to_cartesian_coordinates_l743_743786


namespace isosceles_triangle_l743_743115

theorem isosceles_triangle (ABC : Type) [triangle ABC]
  (I : incenter ABC) (α : incircle ABC)
  (P Q : α) (circAIC : circumcircle AIC)
  (h1 : P ∈ circAIC) (h2 : Q ∈ circAIC)
  (h3 : same_side P A (BI line))
  (h4 : other_side Q C (BI line))
  (M : midpoint_arc AC (α arc))
  (N : midpoint_arc BC (α arc))
  (h_par : PQ ∥ AC) : is_isosceles ABC := 
sorry

end isosceles_triangle_l743_743115


namespace sam_remaining_money_l743_743542

def cost_of_candy_bars (num_candies cost_per_candy: nat) : nat := num_candies * cost_per_candy
def remaining_dimes (initial_dimes cost_in_dimes: nat) : nat := initial_dimes - cost_in_dimes
def remaining_quarters (initial_quarters cost_in_quarters: nat) : nat := initial_quarters - cost_in_quarters
def total_money_in_cents (dimes quarters: nat) : nat := (dimes * 10) + (quarters * 25)

theorem sam_remaining_money : 
  let initial_dimes := 19 in
  let initial_quarters := 6 in
  let num_candy_bars := 4 in
  let cost_per_candy := 3 in
  let cost_of_lollipop := 1 in
  let dimes_left := remaining_dimes initial_dimes (cost_of_candy_bars num_candy_bars cost_per_candy) in
  let quarters_left := remaining_quarters initial_quarters cost_of_lollipop in
  total_money_in_cents dimes_left quarters_left = 195 :=
by
  sorry

end sam_remaining_money_l743_743542


namespace sold_pens_l743_743530

theorem sold_pens (initial_pens : ℕ) (left_pens : ℕ) (initial_books : ℕ) (left_books : ℕ) (sold_pens : ℕ) :
  initial_pens = 42 → left_pens = 19 → initial_books = 143 → left_books = 113 → sold_pens = initial_pens - left_pens →
  sold_pens = 23 := by
  intros
  simp_all
  sorry

end sold_pens_l743_743530


namespace chord_length_l743_743582

theorem chord_length
  (x y : ℝ)
  (h_circle : (x-1)^2 + (y-2)^2 = 2)
  (h_line : 3*x - 4*y = 0) :
  ∃ L : ℝ, L = 2 :=
sorry

end chord_length_l743_743582


namespace unique_solution_of_power_eq_l743_743361

theorem unique_solution_of_power_eq (x y : ℕ) (hxy : x < y) : ∃! x, ∃! y, x^y = y^x := sorry

end unique_solution_of_power_eq_l743_743361


namespace simplify_expression_l743_743195

variable (k : ℝ)
variable (h : k ≠ 0)

theorem simplify_expression : ( (1 / (3 * k)) ^ (-3) * (-k) ^ 4 = 27 * k ^ 7 ) :=
by sorry

end simplify_expression_l743_743195


namespace junior_sales_visits_l743_743697

noncomputable def visits_in_days (days_between_visits : ℕ) (total_days : ℕ) : ℝ :=
  total_days / days_between_visits

theorem junior_sales_visits :
  let senior_visits_per_730_days := visits_in_days 16 730,
      junior_days_between_visits := 12 in
  visits_in_days junior_days_between_visits 730 = senior_visits_per_730_days * 1.33 :=
by
  let senior_visits_per_730_days := visits_in_days 16 730
  have junior_days_between_visits := 12
  rw [← real.div_eq_mul_inv, visits_in_days]
  have eq1 : senior_visits_per_730_days = 730 / 16 := rfl
  rw [eq1]
  have eq2 : visits_in_days junior_days_between_visits 730 = 730 / junior_days_between_visits := rfl
  rw [eq2]
  sorry

end junior_sales_visits_l743_743697


namespace farthest_point_on_circle_l743_743246

def circle_center_x : ℝ := 11
def circle_center_y : ℝ := 13
def circle_radius_squared : ℝ := 116
def point_on_circle_x (x : ℝ) : Prop := (x - circle_center_x)^2 + (y - circle_center_y)^2 = circle_radius_squared

def fixed_point_x : ℝ := 41
def fixed_point_y : ℝ := 25

def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ := (x1 - x2)^2 + (y1 - y2)^2

noncomputable def if_farther (p1 p2 : ℝ × ℝ) (reference : ℝ × ℝ) : ℝ × ℝ :=
  if distance_squared p1.1 p1.2 reference.1 reference.2 > distance_squared p2.1 p2.2 reference.1 reference.2 then p1 else p2

theorem farthest_point_on_circle (x y : ℝ) (hx : point_on_circle_x x y) : if_farther (1, 9) _ (41, 25) = (1, 9) := sorry

end farthest_point_on_circle_l743_743246


namespace minimum_one_by_one_tiles_l743_743267

theorem minimum_one_by_one_tiles (n : ℕ) (h : n = 23) :
  ∃ (m : ℕ), m = 1 ∧
  ∀ (tile : ℕ → ℕ → Prop),
  (tile 1 1 ∨ tile 2 2 ∨ tile 3 3) →
  (∃ f : ℕ → ℕ → ℕ, 
     (∀ i j, 1 ≤ f i j ∧ f i j ≤ 3) ∧
     (∑ i in finRange n, ∑ j in finRange n, f i j ≤ n * n) ∧
     (∑ i in finRange n, ∑ j in finRange n, ite (f i j = 1) 1 0 = m)) :=
by
sorry

end minimum_one_by_one_tiles_l743_743267


namespace complement_M_l743_743403

open Set

variable {U : Set ℝ := (λ x, True)} -- Universal set as the set of all real numbers
variable {M : Set ℝ := {x | log 10 (1 - x) > 0}} -- Defining the set M based on the given condition

theorem complement_M :
  compl M = {x : ℝ | 0 ≤ x} :=
by
  sorry

end complement_M_l743_743403


namespace train_platform_length_l743_743266

theorem train_platform_length (train_length : ℕ) (platform_crossing_time : ℕ) (pole_crossing_time : ℕ) (length_of_platform : ℕ) :
  train_length = 300 →
  platform_crossing_time = 27 →
  pole_crossing_time = 18 →
  ((train_length * platform_crossing_time / pole_crossing_time) = train_length + length_of_platform) →
  length_of_platform = 150 :=
by
  intros h_train_length h_platform_time h_pole_time h_eq
  -- Proof omitted
  sorry

end train_platform_length_l743_743266


namespace relationship_abc_l743_743880

noncomputable def a : ℝ := 0.6 ^ 4.2
noncomputable def b : ℝ := 7 ^ 0.6
noncomputable def c : ℝ := Real.logBase 0.6 7

theorem relationship_abc : c < a ∧ a < b := 
by 
  sorry

end relationship_abc_l743_743880


namespace reciprocal_of_recurring_three_l743_743992

noncomputable def recurring_three := 0.33333333333 -- approximation of 0.\overline{3}

theorem reciprocal_of_recurring_three :
  let x := recurring_three in
  (x = (1/3)) → (1 / x = 3) := 
by 
  sorry

end reciprocal_of_recurring_three_l743_743992


namespace miranda_loan_difference_l743_743167

noncomputable def A_monthly (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def A_annual (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r) ^ t

-- Define the constants.
def P : ℝ := 8000
def r : ℝ := 0.15
def t : ℝ := 5
def n_monthly : ℕ := 12
def n_annual : ℕ := 1

-- The theorem statement
theorem miranda_loan_difference :
  (A_annual P r t - A_monthly P r n_monthly t) ≈ 2511.37 :=
sorry

end miranda_loan_difference_l743_743167


namespace binomial_expansion_coefficient_l743_743818

theorem binomial_expansion_coefficient :
  let a_0 : ℚ := (1 + 2 * (0:ℚ))^5
  (1 + 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_3 = 80 :=
by 
  sorry

end binomial_expansion_coefficient_l743_743818


namespace exists_two_natural_pairs_satisfying_equation_l743_743730

theorem exists_two_natural_pairs_satisfying_equation :
  ∃ (x1 y1 x2 y2 : ℕ), (2 * x1^3 = y1^4) ∧ (2 * x2^3 = y2^4) ∧ ¬(x1 = x2 ∧ y1 = y2) :=
sorry

end exists_two_natural_pairs_satisfying_equation_l743_743730


namespace sequence_count_l743_743008

open Nat

theorem sequence_count :
  ∃ (S : Fin 7 → ℤ),
    (∀ i < 7, -1 ≤ S i ∧ S i ≤ 1) ∧
    (S 0 * S 1 + S 1 * S 2 + S 2 * S 3 + S 3 * S 4 + S 4 * S 5 + S 5 * S 6 = 4) →
    { s // ∀ i < 7, -1 ≤ s i ∧ s i ≤ 1 ∧ (s 0 * s 1 + s 1 * s 2 + s 2 * s 3 + s 3 * s 4 + s 4 * s 5 + s 5 * s 6 = 4) }.card = 38 := 
sorry

end sequence_count_l743_743008


namespace determine_B_value_l743_743717

theorem determine_B_value (f : Polynomial ℝ) (A C : ℝ) :
  f = Polynomial.X^5 - 15 * Polynomial.X^4 + A * Polynomial.X^3 + B * Polynomial.X^2 + C * Polynomial.X + 24 → 
  (∀ r : ℝ, Polynomial.is_root f r → r > 0) →
  (f.roots.sum = 15) →
  B = -90 :=
by sorry

end determine_B_value_l743_743717


namespace isosceles_triangle_l743_743134

   open EuclideanGeometry

   -- Define the conditions of the problem in Lean
   variable {I A B C P Q M N : Point}
   variable (α : Circle) (circumcircle_AIC : Circle)

   -- Conditions extracted from the problem
   def conditions : Prop :=
   IsIncenter I △ABC ∧
   Incircle α △ABC ∧
   Circle.Diameter α P Q ∧
   Circle.Containing circumcircle_AIC (trianglePoint AIC) ∧
   SameSide P A (Line BI) ∧
   SameSide Q C (Line BI) ∧
   IsMidpointArc M ARC(α A C) ∧
   IsMidpointArc N ARC(α B C) ∧
   Parallel (Line PQ) (Line AC)

   -- Proof statement in Lean
   theorem isosceles_triangle
     (h : conditions α circumcircle_AIC) : IsIsosceles (△ABC) :=
   sorry
   
end isosceles_triangle_l743_743134


namespace no_solution_count_l743_743642

theorem no_solution_count (n : ℕ) (h : n ≥ 2) : 
  let count_no_solutions : ℕ := 
    (n^2) - (finset.card (finset.filter
      (λ m, ¬ ∃ x y, 0 ≤ x ∧ 0 ≤ y ∧ x < n ∧ y < n ∧ (x^n + y^n) % (n^2) = m)
      (finset.range (n * n)))
    ) in
  count_no_solutions ≥ nat.choose n 2 :=
by sorry

end no_solution_count_l743_743642


namespace tan_x_parallel_f_properties_l743_743001

-- Definitions of the vectors and conditions
def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.sqrt 3 * Real.cos x)
def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Proof statements
theorem tan_x_parallel (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) (h3 : (a x).1 / (b x).1 = (a x).2 / (b x).2) : Real.tan x = Real.sqrt 3 :=
sorry

theorem f_properties (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) : 
  (Real.periodic f Real.pi) ∧ (∃ (x : ℝ), x = Real.pi / 4 ∧ f x = 1 + Real.sqrt 3 / 2) :=
sorry

end tan_x_parallel_f_properties_l743_743001


namespace find_power_function_value_l743_743807

noncomputable def f (x : ℝ) : ℝ := x^(-1 / 2)

theorem find_power_function_value :
  f (1 / 4) = 2 :=
by
  have h_f_eq : f 3 = (real.sqrt 3) / 3 := sorry
  sorry

end find_power_function_value_l743_743807


namespace min_distance_ellipse_to_line_l743_743182

theorem min_distance_ellipse_to_line :
  ∀ (x y : ℝ), (x^2 / 9 + y^2 / 4 = 1) → ∃ d : ℝ, (x + 2 * y - 10 = 0 → d = sqrt 5) := by
sorry

end min_distance_ellipse_to_line_l743_743182


namespace jerry_apples_l743_743868

theorem jerry_apples (J : ℕ) (h1 : 20 + 60 + J = 3 * 2 * 20):
  J = 40 :=
sorry

end jerry_apples_l743_743868


namespace angle_between_vectors_l743_743778

open Real
open LinearAlgebra

variables (a b : ℝ^3)
variables (ha : a ≠ 0) (hb : b ≠ 0)
variables (h1 : (a - 6 • b) ⬝ a = 0) (h2 : (2 • a - 3 • b) ⬝ b = 0)

theorem angle_between_vectors (ha : a ≠ 0) (hb : b ≠ 0) (h1 : (a - 6 • b) ⬝ a = 0) (h2 : (2 • a - 3 • b) ⬝ b = 0) :
  ∃ θ : ℝ, θ = π / 3 ∧ 
  cos θ = (a ⬝ b) / ((∥a∥ * ∥b∥)) := sorry

end angle_between_vectors_l743_743778


namespace circle_area_ratio_l743_743684

theorem circle_area_ratio {P : ℝ} (hP : P > 0) :
  let s := P / 4 in
  let t := P / 5 in
  let r := s * Real.sqrt 2 / 2 in
  let R := t / (2 * (Real.sqrt (5 - 2 * Real.sqrt 5) / 4)) in
  let C := Real.pi * r^2 in
  let D := Real.pi * R^2 in
  C / D = 25 * (5 - 2 * Real.sqrt 5) / 128 :=
by
  sorry

end circle_area_ratio_l743_743684


namespace product_of_first_2011_terms_l743_743222

theorem product_of_first_2011_terms (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : a 2 = 2)
  (h3 : ∀ n : ℕ, 0 < n → a (n + 2) * a n = 2 * a (n + 1)) :
  (∏ i in Finset.range 2011, a (i + 1)) = 2 ^ 2010 :=
by sorry

end product_of_first_2011_terms_l743_743222


namespace correct_conclusions_l743_743601

/-- Defining the sequence terms and conditions -/
def sequence_an (n : ℕ) : ℝ := (n-1)/((n div (nat.sqrt n))+2)

/-- Defining the sum of first n terms of the sequence -/
def Sn (n : ℕ) : ℝ := (∑ i in range n, sequence_an i.up1 : ℝ)

/-- Prove the correctness of given conclusions -/
theorem correct_conclusions (a24 : sequence_an 24 = 3/8)
  (T : ℕ → ℝ) (T_correct : ∀ n, T n = (n^2 + n)/4)
  (cond_k : ∃ k, Sn k < 10 ∧ Sn (k+1) ≥ 10) :
  a24 ∧ (T n = (n^2+n)/4) ∧ (∃ k, Sn k < 10 ∧ Sn (k+1) ≥ 10 → sequence_an k = 5/7) := 
sorry

end correct_conclusions_l743_743601


namespace ce_div_cd_ge_two_sqrt_three_l743_743053

-- Definitions based on conditions
variables (A B C D E : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]

def is_equilateral_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] : Prop :=
  dist A B = dist A C ∧ dist B C = dist A B ∧ dist B C = dist A C

def is_point_on_side (D : Type) (B C : Type) [metric_space D] [metric_space B] [metric_space C] : Prop :=
  dist B D + dist D C = dist B C

variables (AD CE : Type)
variables [line AD] [line CE]
variables [parallel_lines AD CE] [intersect_lines CE A B E]

-- Statement to prove
theorem ce_div_cd_ge_two_sqrt_three (ABC_is_equilateral : is_equilateral_triangle A B C)
  (D_on_BC : is_point_on_side D B C)
  (AD_parallel_CE : parallel_lines AD CE)
  (CE_intersects_AB_at_E : intersect_lines CE A B E) :
  dist E C / dist D C ≥ 2 * real.sqrt 3 :=
sorry -- Proof goes here

end ce_div_cd_ge_two_sqrt_three_l743_743053


namespace third_day_pairs_l743_743070

def pairs_of_earrings_d3 (gumballs_per_pair : ℕ) (initial_pairs_d1 : ℕ) (pairs_factor_d2 : ℕ) (gumballs_per_day : ℕ) (days : ℕ) (less_pairs_d3 : ℕ) : ℕ :=
  let gumballs_d1 := initial_pairs_d1 * gumballs_per_pair
  let pairs_d2 := pairs_factor_d2 * initial_pairs_d1
  let gumballs_d2 := pairs_d2 * gumballs_per_pair
  let total_gumballs := gumballs_d1 + gumballs_d2
  let total_gumballs_needed := days * gumballs_per_day
  let remaining_gumballs_needed := total_gumballs_needed - total_gumballs
  let pairs_needed_d3 := remaining_gumballs_needed / gumballs_per_pair
  pairs_d2 - less_pairs_d3

theorem third_day_pairs (gumballs_per_pair : ℕ) (initial_pairs_d1 : ℕ) (pairs_factor_d2 : ℕ) (gumballs_per_day : ℕ) (days : ℕ) (less_pairs_d3 : ℕ) :
  let pairs_d3 := pairs_of_earrings_d3 gumballs_per_pair initial_pairs_d1 pairs_factor_d2 gumballs_per_day days less_pairs_d3
  gumballs_per_pair = 9 ∧ initial_pairs_d1 = 3 ∧ pairs_factor_d2 = 2 ∧ gumballs_per_day = 3 ∧ days = 42 ∧ less_pairs_d3 = 1 →
  pairs_d3 = 5 :=
by
  intros
  simp only [pairs_of_earrings_d3]
  sorry

end third_day_pairs_l743_743070


namespace Sandy_change_l743_743546

theorem Sandy_change (pants shirt sweater shoes total paid change : ℝ)
  (h1 : pants = 13.58) (h2 : shirt = 10.29) (h3 : sweater = 24.97) (h4 : shoes = 39.99) (h5 : total = pants + shirt + sweater + shoes) (h6 : paid = 100) (h7 : change = paid - total) :
  change = 11.17 := 
sorry

end Sandy_change_l743_743546


namespace total_student_capacity_l743_743837

-- Define the conditions
def school_capacity_one : ℕ := 400
def school_capacity_two : ℕ := 340
def number_of_schools_one : ℕ := 2
def number_of_schools_two : ℕ := 2

-- Statement to prove
theorem total_student_capacity :
  (number_of_schools_one * school_capacity_one) +
  (number_of_schools_two * school_capacity_two) = 1480 :=
by
  calc
    (number_of_schools_one * school_capacity_one) +
    (number_of_schools_two * school_capacity_two)
    = (2 * 400) + (2 * 340) : by sorry
    = 800 + 680 : by sorry
    = 1480 : by sorry

end total_student_capacity_l743_743837


namespace equidistant_point_on_x_axis_l743_743238

/-- Define the distance function between two points in the 2D plane -/
def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Define points A and B -/
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (2, 6)

/-- The main theorem -/
theorem equidistant_point_on_x_axis :
  ∃ x : ℝ, (distance (x, 0) A = distance (x, 0) B) ∧ x = 3.1 :=
by
  sorry

end equidistant_point_on_x_axis_l743_743238


namespace sufficient_not_necessary_condition_l743_743692

theorem sufficient_not_necessary_condition 
  (a b : ℝ) : (a > b + 1) → (a > b) ∧ ¬ (∀ a b, (a > b) → (a > b + 1)) := 
by
  assume h1 : a > b + 1
  apply and.intro
  { exact lt.trans (lt_add_one_of_lt (lt_add_one_of_lt h1)) sorry }
  { intro h2
    have counterexample := h2 2 1
    have : 2 > 1 := by linarith
    apply not_lt_of_lt this
    apply counterexample
    exact this }
  sorry

end sufficient_not_necessary_condition_l743_743692


namespace min_value_f_l743_743511

def a_seq : ℕ → ℕ
def f (a_seq : ℕ → ℕ) := (∑ i in finset.range 2019, (a_seq i) ^ 2) - (∑ i in finset.range 1008, (a_seq (2*i) * a_seq (2*i+2)))

axiom a_cond (a_seq : ℕ → ℕ) : 1 = a_seq 0 ∧ (∀ i < 2018, a_seq i ≤ a_seq (i + 1)) ∧ a_seq (2018) = 99

theorem min_value_f (a_seq : ℕ → ℕ) (h : a_cond a_seq) : 
  f a_seq = 7400 := 
sorry

end min_value_f_l743_743511


namespace find_a_l743_743754

theorem find_a (a x_1 x_2 : ℝ) (h1 : a < 0) (h2 : x_1 = 3 * a) (h3 : x_2 = -2 * a) (h4 : x_2 - x_1 = 5 * real.sqrt 2) : a = -real.sqrt 2 :=
begin
  sorry
end

end find_a_l743_743754


namespace S_five_equals_342_l743_743997

def largest_odd_factor (n : ℕ) : ℕ :=
  if h : n % 2 = 1 then n else largest_odd_factor (n / 2)

def S (n : ℕ) : ℕ :=
  if n = 0 then 0
  else largest_odd_factor (1) + (List.range (2^n - 1)).sum (fun i => largest_odd_factor (i + 1))

theorem S_five_equals_342 : S(5) = 342 :=
  sorry

end S_five_equals_342_l743_743997


namespace solve_linear_eqns_x3y2z_l743_743485

theorem solve_linear_eqns_x3y2z (x y z : ℤ) 
  (h1 : x - 3 * y + 2 * z = 1) 
  (h2 : 2 * x + y - 5 * z = 7) : 
  z = 4 ^ 111 := 
sorry

end solve_linear_eqns_x3y2z_l743_743485


namespace triangle_ABC_isosceles_l743_743098

-- Lean 4 definitions for the generated equivalent proof problem
noncomputable def incenter (A B C I : Type*) := sorry
noncomputable def incircle (ABC : Type*) (α : Type*) := sorry
noncomputable def circumcircle (A I C : Type*) := sorry
noncomputable def intersects (circle1 circle2 : Type*) (P Q : Type*) := sorry
noncomputable def same_side_line (P A : Type*) (line : Type*) := sorry
noncomputable def other_side_line (Q C : Type*) (line : Type*) := sorry
noncomputable def midpoint_arc (arc : Type*) := sorry
noncomputable def parallel (line1 line2 : Type*) := sorry
noncomputable def triangle_isosceles (A B C : Type*) := ∃ (ABC_is_isosceles : Prop), ABC_is_isosceles

-- Given conditions for the incenter, incircle, and parallel lines, we must prove the triangle is isosceles
theorem triangle_ABC_isosceles (A B C I α P Q M N : Type*)
  (h1 : incenter A B C I)
  (h2 : incircle ABC α)
  (h3 : circumcircle A I C)
  (h4 : intersects α (circumcircle A I C) P Q)
  (h5 : same_side_line P A (set I B))
  (h6 : other_side_line Q C (set I B))
  (h7 : midpoint_arc α AC M)
  (h8 : midpoint_arc α BC N)
  (h9 : parallel PQ AC) :
  triangle_isosceles A B C := sorry

end triangle_ABC_isosceles_l743_743098


namespace sufficient_but_not_necessary_condition_for_hyperbola_l743_743426

theorem sufficient_but_not_necessary_condition_for_hyperbola (k : ℝ) :
  (∃ k : ℝ, k > 3 ∧ (∃ x y : ℝ, (x^2) / (k - 3) - (y^2) / (k + 3) = 1)) ∧ 
  (∃ k : ℝ, k < -3 ∧ (∃ x y : ℝ, (x^2) / (k - 3) - (y^2) / (k + 3) = 1)) :=
    sorry

end sufficient_but_not_necessary_condition_for_hyperbola_l743_743426


namespace desired_markup_percentage_l743_743291

theorem desired_markup_percentage
  (initial_price : ℝ) (markup_rate : ℝ) (wholesale_price : ℝ) (additional_increase : ℝ) 
  (h1 : initial_price = wholesale_price * (1 + markup_rate)) 
  (h2 : initial_price = 34) 
  (h3 : markup_rate = 0.70) 
  (h4 : additional_increase = 6) 
  : ( (initial_price + additional_increase - wholesale_price) / wholesale_price * 100 ) = 100 := 
by
  sorry

end desired_markup_percentage_l743_743291


namespace ABC_is_isosceles_l743_743120

open Mobius

variables
  (ABC : Triangle)
  (I : Point)
  (α P Q M N : Point)
  (A B C Z : Point)
  [Incenter I ABC]
  [Incircle α ABC]
  [Circumcircle (Triangle.mk A I C)]
  [PQ_parallel_AC : is_parallel (Segment.mk P Q) (Segment.mk A C)]
  [Midpoint M (Arc.mk A C α)]
  [Midpoint N (Arc.mk B C α)]
  [center_Z : Center Z (Circumcircle (Triangle.mk A I C))]
  [Chord_α_PQ : common_chord α (Circumcircle (Triangle.mk A I C)) (Segment.mk P Q)]
  [P_A_same_BI : same_side P A (Line.mk B I)]
  [Q_C_other_BI : same_side Q C (Line.mk B I)]

theorem ABC_is_isosceles
  (h_parallel : PQ_parallel_AC) : 
  Isosceles ABC := 
sorry

end ABC_is_isosceles_l743_743120


namespace find_x_l743_743857

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (h : x + y + x * y = 143) : x = 15 :=
by sorry

end find_x_l743_743857


namespace chocolates_sold_l743_743434

theorem chocolates_sold (C S : ℝ) (n : ℕ) (h1 : 165 * C = n * S) (h2 : ((S - C) / C) * 100 = 10) : n = 150 :=
by
  sorry

end chocolates_sold_l743_743434


namespace tangent_curve_value_l743_743213

theorem tangent_curve_value (a b k : ℝ) (A : ℝ × ℝ) (hA : A = (1, 3)) 
  (hk : k = 3 + a) (hcurve : ∀ x, (1, 3) = (x, x^3 + a * x + b)) 
  (htangent : ∀ x, A = (x, k * x + 1)) : 2 * a + b = 1 :=
by
  let x := 1
  have A_eq_curve : (1, 3) = (1, 1^3 + a * 1 + b) from hcurve 1
  have A_eq_tangent : (1, 3) = (1, k * 1 + 1) from htangent 1
  have k_value : k = 2 from cast (3 = k + 1)
  have a_value : a = -1 from cast (3 = 1 + (-1) + b)
  have b_value : b = 3 from cast (1 + (-1) + 3 = 3)
  sorry

end tangent_curve_value_l743_743213


namespace distance_Y_to_AD_l743_743468

theorem distance_Y_to_AD (s : ℝ) (h : 0 < s) :
  let A := (0 : ℝ, 0 : ℝ),
      C := (s, s),
      Y := (s / 4, s / 4) in
  dist Y (0, Y.2) = s / 4 :=
by
  sorry

end distance_Y_to_AD_l743_743468


namespace sum_CAMSA_PASTA_l743_743158

-- Define the multisets for the words "САМСА" and "ПАСТА"
def letters_CAMSA := {'С':=2, 'А':=2, 'М':=1}
def letters_PASTA := {'П':=1, 'А':=2, 'С':=1, 'Т':=1}

-- Define the factorial function
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- Define the number of distinct permutations for a multiset
def num_permutations (letters : List (Char × Nat)) : Nat :=
  let n := (letters.map Prod.snd).sum
  let denominator := (letters.map (λ x => factorial x.snd)).prod
  factorial n / denominator

-- Definitions specific to the conditions
def num_CAMSA : Nat := num_permutations letters_CAMSA.toList
def num_PASTA : Nat := num_permutations letters_PASTA.toList

-- The theorem to be proven
theorem sum_CAMSA_PASTA : num_CAMSA + num_PASTA = 90 := by
  sorry

end sum_CAMSA_PASTA_l743_743158


namespace find_theta_l743_743406

-- Define the given vectors and condition on theta
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (θ : ℝ) : ℝ × ℝ := (Real.tan θ, -1)

-- Condition that a is parallel to b
def vectors_parallel (θ : ℝ) : Prop := 
  vector_b θ.1 / vector_a.1 = vector_b θ.2 / vector_a.2

-- Define the interval for theta
def theta_in_interval (θ : ℝ) : Prop := 0 ≤ θ ∧ θ ≤ π

-- Statement to prove
theorem find_theta (θ : ℝ) (h_parallel: vectors_parallel θ) (h_theta_interval : theta_in_interval θ) :
  θ = Real.arctan (1/2) := by
  sorry

end find_theta_l743_743406


namespace checkerboard_problem_l743_743653

-- Definitions corresponding to conditions
def checkerboard_size := 10

def is_alternating (i j : ℕ) : bool :=
  (i + j) % 2 = 0

def num_squares_with_sides_on_grid_lines_containing_at_least_6_black_squares (n : ℕ) : ℕ :=
  if n >= 4 then (11 - n) * (11 - n) else 0

-- Problem statement
theorem checkerboard_problem : 
  let count_squares : ℕ := (∑ n in finset.range checkerboard_size.succ, num_squares_with_sides_on_grid_lines_containing_at_least_6_black_squares n)
  in count_squares = 140 :=
begin
  sorry
end

end checkerboard_problem_l743_743653


namespace rationalized_sum_l743_743538

noncomputable def rationalize_denominator (x : ℝ) := 
  (4 * real.cbrt 49) / 21

theorem rationalized_sum : 
  let A := 4 in
  let B := 49 in
  let C := 21 in
  A + B + C = 74 :=
by
  let A := 4
  let B := 49
  let C := 21
  trivial

end rationalized_sum_l743_743538


namespace temperature_on_tuesday_l743_743930

variable (T W Th F : ℝ)

-- Conditions
axiom H1 : (T + W + Th) / 3 = 42
axiom H2 : (W + Th + F) / 3 = 44
axiom H3 : F = 43

-- Proof statement
theorem temperature_on_tuesday : T = 37 :=
by
  -- This would be the place to fill in the proof using H1, H2, and H3
  sorry

end temperature_on_tuesday_l743_743930


namespace limit_sin_ax_over_sin_bx_l743_743727

theorem limit_sin_ax_over_sin_bx (a b : ℝ) :
  tendsto (λ x : ℝ, (sin (a * x)) / (sin (b * x))) (𝓝 0) (𝓝 (a / b)) :=
sorry

end limit_sin_ax_over_sin_bx_l743_743727


namespace three_color_no_monochromatic_cycle_l743_743913

variable {V : Type} [Fintype V]
variable (G : SimpleGraph V)
variable (three_coloring : Fin 3 → Prop)

-- Assumption that G is planar (represented by some planar property)
axiom planar_graph (G : SimpleGraph V) : Prop

-- Assumption for preventing monochromatic cycles
axiom no_monochromatic_cycle (G : SimpleGraph V) (coloring : V → Fin 3) : Prop

-- Theorem statement
theorem three_color_no_monochromatic_cycle 
  (G : SimpleGraph V) [planar_graph G] :
  ∃ coloring : V → Fin 3, no_monochromatic_cycle G coloring :=
sorry

end three_color_no_monochromatic_cycle_l743_743913


namespace sum_of_integer_solutions_eq_4_l743_743600

theorem sum_of_integer_solutions_eq_4 :
  let solutions := {x : ℤ | x^2 = 192 + x},
  ∑ x in solutions, x = 4 :=
by {
  sorry
}

end sum_of_integer_solutions_eq_4_l743_743600


namespace clock_angle_at_3_30_l743_743976

/-- The degree measure of the smaller angle between 
    the hour hand and the minute hand of a clock 
    at exactly 3:30 p.m. on a 12-hour analog clock is 75 degrees. -/
theorem clock_angle_at_3_30 : 
    let hour_angle := 3.5 * 30,
        minute_angle := 6 * 30
    in min (abs (hour_angle - minute_angle)) 
           (360 - abs (hour_angle - minute_angle)) = 75 := 
by
    let hour_angle := 3.5 * 30 -- hour hand position in degrees
    let minute_angle := 6 * 30 -- minute hand position in degrees
    let angle_diff := abs (hour_angle - minute_angle)
    let smaller_angle := min angle_diff (360 - angle_diff)
    show smaller_angle = 75
    sorry

end clock_angle_at_3_30_l743_743976


namespace quadratic_floor_eq_solutions_count_l743_743578

theorem quadratic_floor_eq_solutions_count : 
  ∃ s : Finset ℝ, (∀ x : ℝ, x^2 - 4 * ⌊x⌋ + 3 = 0 → x ∈ s) ∧ s.card = 3 :=
by 
  sorry

end quadratic_floor_eq_solutions_count_l743_743578


namespace snickers_bars_needed_l743_743474

-- Definitions for the problem conditions
def total_required_points : ℕ := 2000
def bunnies_sold : ℕ := 8
def bunny_points : ℕ := 100
def snickers_points : ℕ := 25
def points_from_bunnies : ℕ := bunnies_sold * bunny_points
def remaining_points_needed : ℕ := total_required_points - points_from_bunnies

-- Define the problem statement to prove
theorem snickers_bars_needed : remaining_points_needed / snickers_points = 48 :=
by
  -- Skipping the proof steps
  sorry

end snickers_bars_needed_l743_743474


namespace simplify_polynomials_l743_743552

theorem simplify_polynomials :
  (3 * x^3 + 4 * x^2 + 6 * x - 5) - (2 * x^3 + 2 * x^2 + 3 * x - 8) = x^3 + 2 * x^2 + 3 * x + 3 :=
by
  sorry

end simplify_polynomials_l743_743552


namespace jacoby_cookie_price_l743_743058

def total_trip_cost : ℕ := 5000
def hourly_wage : ℕ := 20
def hours_worked : ℕ := 10
def cookies_sold : ℕ := 24
def lottery_ticket_cost : ℕ := 10
def lottery_winnings : ℕ := 500
def gift_from_sisters : ℕ := 2 * 500
def additional_needed : ℕ := 3214

theorem jacoby_cookie_price :
  let job_earnings := hourly_wage * hours_worked,
      total_money := job_earnings + lottery_winnings + gift_from_sisters - lottery_ticket_cost,
      total_needed_from_cookies := additional_needed + total_money,
      price_per_cookie := total_needed_from_cookies / cookies_sold
  in price_per_cookie = 204.33 := 
by
  sorry

end jacoby_cookie_price_l743_743058


namespace competition_ranking_l743_743750

def ranking (p : Nat) : Type := { names // names.length = 5 }

def guess_ranking_X : ranking 5 := {E, D, C, B, A}
def guess_ranking_Y : ranking 5 := {D, A, E, C, B}

variable (ranking_result : ranking 5)

axiom condition_X_no_correct_positions (r : ranking) : r ≠ guess_ranking_X
axiom condition_X_no_correct_orders (r : ranking) : ∀ i : Fin 4, r.names[i] < r.names[i + 1] -> False
axiom condition_Y_two_correct_positions (r : ranking) : ∃ i j : Fin 5, i ≠ j ∧ r.names[i] = guess_ranking_Y.names[i] ∧ r.names[j] = guess_ranking_Y.names[j]
axiom condition_Y_two_correct_orders (r : ranking) : ∃ i j : Fin 4, i ≠ j ∧ r.names[i] < r.names[i + 1] ∧ r.names[j] < r.names[j + 1]

theorem competition_ranking : ranking :=
  ranking_result :=
  {
    names := {E, D, A, C, B},
    sorry
  }

end competition_ranking_l743_743750


namespace modulus_of_z_l743_743433

def i := Complex.I
def z := i / (1 - i)

theorem modulus_of_z : Complex.abs z = Real.sqrt 2 / 2 :=
by 
   sorry

end modulus_of_z_l743_743433


namespace pyramid_cross_section_l743_743531

theorem pyramid_cross_section (n : ℕ) (h : n ≥ 5) :
  ∀ (S : Type) [real_affine_space S],
  ∀ (A : fin n → S) (B : fin (n+1) → S),
  is_regular_ngon A →
  is_regular_ngon B →
  ¬(is_cross_section S A B) := 
sorry

-- Definitions for is_regular_ngon and is_cross_section might be needed

def is_regular_ngon (A : fin n → S) : Prop := sorry

def is_cross_section (S : Type) [real_affine_space S] (A : fin n → S) (B : fin (n+1) → S) : Prop := sorry

end pyramid_cross_section_l743_743531


namespace dice_even_odd_probability_l743_743921

theorem dice_even_odd_probability : 
  let n := 6 in
  let k := 3 in
  let p_even := 3 / 6 in
  let p_odd := 3 / 6 in
  let total_ways := Nat.choose n k in
  let arrangement_probability := p_even ^ k * p_odd ^ (n - k) in
  (total_ways : ℝ) * arrangement_probability = 5 / 16 :=
by
  sorry

end dice_even_odd_probability_l743_743921


namespace henry_change_l743_743412

theorem henry_change (n : ℕ) (p m : ℝ) (h_n : n = 4) (h_p : p = 0.75) (h_m : m = 10) : 
  m - (n * p) = 7 := 
by 
  sorry

end henry_change_l743_743412


namespace num_lists_correct_l743_743262

def num_balls : ℕ := 18
def num_draws : ℕ := 4

theorem num_lists_correct : (num_balls ^ num_draws) = 104976 :=
by
  sorry

end num_lists_correct_l743_743262


namespace leah_jackson_meetings_l743_743071

noncomputable def leah_speed := 200  -- Leah's speed in m/min
noncomputable def jackson_speed := 280  -- Jackson's speed in m/min
noncomputable def leah_radius := 40  -- Leah's track radius in meters
noncomputable def jackson_radius := 55  -- Jackson's track radius in meters
noncomputable def running_time := 45  -- Running time in minutes

theorem leah_jackson_meetings :
  let C_L := 2 * Real.pi * leah_radius,
      C_J := 2 * Real.pi * jackson_radius,
      omega_L := (leah_speed / C_L) * (2 * Real.pi),
      omega_J := (jackson_speed / C_J) * (2 * Real.pi),
      relative_omega := omega_L + omega_J,
      time_to_meet := (2 * Real.pi) / relative_omega,
      total_meetings := running_time / time_to_meet in
  total_meetings.floor = 72 := 
by
  sorry

end leah_jackson_meetings_l743_743071


namespace curve_E_equation_min_value_PO_PF_squared_l743_743286

-- Definitions for Circle C1 and C2
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 6 * x + 5 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x - 91 = 0

-- Define being tangent to the circles
def tangent_to_C1 (M : ℝ × ℝ) (R : ℝ) : Prop :=
  let (x, y) := M in 
  dist (x, y) (-3, 0) = R + 2

def tangent_to_C2 (M : ℝ × ℝ) (R : ℝ) : Prop :=
  let (x, y) := M in 
  dist (x, y) (3, 0) = 10 - R

-- Equation of curve E
def curve_E (x y : ℝ) : Prop := (x^2 / 36) + (y^2 / 27) = 1

-- Lean statements for the proof
-- Part 1: Prove the equation of curve E
theorem curve_E_equation :
  (∀ (M : ℝ × ℝ) (R : ℝ), tangent_to_C1 M R ∧ tangent_to_C2 M R) →
  ∀ x y, curve_E x y :=
sorry

-- Part 2: Minimum value of |PO|^2 + |PF|^2
theorem min_value_PO_PF_squared (P O F : ℝ × ℝ) :
  curve_E P.1 P.2 →
  O = (0, 0) →
  F = (3, 0) →
  ∃ min_val : ℝ, min_val = 45 ∧ (∀ x y, x^2 + y^2 + (x - 3)^2 + y^2 ≥ min_val) :=
sorry

end curve_E_equation_min_value_PO_PF_squared_l743_743286


namespace simplify_sqrt_product_l743_743920

theorem simplify_sqrt_product (x : ℝ) :
  Real.sqrt (45 * x) * Real.sqrt (20 * x) * Real.sqrt (28 * x) * Real.sqrt (5 * x) =
  60 * x^2 * Real.sqrt 35 :=
by
  sorry

end simplify_sqrt_product_l743_743920


namespace overall_university_or_higher_percentage_l743_743024

structure SectorStats :=
  (high_school : ℝ)
  (vocational : ℝ)
  (university : ℝ)
  (postgraduate : ℝ)
  (job_choice_no_university : ℝ)
  (university_or_higher_no_job_choice : ℝ)
  (job_choice_by_age : ℝ → ℝ)

def SectorA : SectorStats := 
  ⟨0.60, 0.10, 0.25, 0.05, 0.10, 0.30, λ age, if age ≤ 29 then 0.30 else if age ≤ 49 then 0.45 else 0.25⟩
  
def SectorB : SectorStats := 
  ⟨0.50, 0.20, 0.25, 0.05, 0.05, 0.35, λ age, if age ≤ 29 then 0.20 else if age ≤ 49 then 0.50 else 0.30⟩
  
def SectorC : SectorStats := 
  ⟨0.40, 0.20, 0.30, 0.10, 0.15, 0.25, λ age, if age ≤ 29 then 0.25 else if age ≤ 49 then 0.35 else 0.40⟩

def population_distribution : SectorStats → ℝ :=
  λ sector, if sector = SectorA then 0.30 else if sector = SectorB then 0.40 else if sector = SectorC then 0.30 else 0

theorem overall_university_or_higher_percentage : 
  let sector_university_or_higher (s : SectorStats) := s.university + s.postgraduate in
  (sector_university_or_higher SectorA) * (population_distribution SectorA) +
  (sector_university_or_higher SectorB) * (population_distribution SectorB) +
  (sector_university_or_higher SectorC) * (population_distribution SectorC) = 0.33 := 
by 
  -- Actual proof is omitted
  sorry

end overall_university_or_higher_percentage_l743_743024


namespace y_intercept_of_parallel_line_l743_743668

theorem y_intercept_of_parallel_line (m : ℝ) (c1 c2 : ℝ) (x1 y1 : ℝ) (H_parallel : m = -3) (H_passing : (x1, y1) = (1, -4)) : 
    c2 = -1 :=
  sorry

end y_intercept_of_parallel_line_l743_743668


namespace range_of_f_on_interval_range_of_m_for_triangle_function_l743_743012

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := - (sin x)^2 - cos x + 5 / 4 + m

theorem range_of_f_on_interval (m : ℝ) : 
  set.range (λ x, f x m | x ∈ set.Icc (-π / 3) (2 * π / 3)) = set.Icc m (m + 1) := 
sorry

theorem range_of_m_for_triangle_function :
  (∀ (a b c : ℝ) (h : a ∈ set.Icc (-π / 3) (2 * π / 3)) (h' : b ∈ set.Icc (-π / 3) (2 * π / 3)) (h'' : c ∈ set.Icc (-π / 3) (2 * π / 3)),
    let s := (set.range (λ x, f x)) 
    (s a h) + (s b h') > (s c h'') ∧
    (s b h') + (s c h'') > (s a h) ∧
    (s c h'') + (s a h) > (s b h')) 
  -> m ∈ set.Ioi (1 : ℝ) :=
sorry

end range_of_f_on_interval_range_of_m_for_triangle_function_l743_743012


namespace log_base_x_y_eq_pi_l743_743274

theorem log_base_x_y_eq_pi 
  (x y : ℝ)
  (hx : x > 0) 
  (hy : y > 0)
  (hx_ne_one : x ≠ 1)
  (radius_eq : log 10 (x^3) = r)
  (circumf_eq : log 10 (y^6) = C)
  (C_eq_2pi_r : C = 2 * π * r) :
  log x y = π := 
by
  sorry

end log_base_x_y_eq_pi_l743_743274


namespace exists_monochromatic_C4_l743_743362

variable (n : ℕ)

def bipartite_graph (n : ℕ) : Type := CompleteBipartiteGraph (fin n) (fin n)

noncomputable def edge_coloring (G : bipartite_graph n) : EColoring := sorry -- Define a suitable edge coloring

theorem exists_monochromatic_C4 (n : ℕ) (h : n ≥ 5) :
  ∀ (G : bipartite_graph n) (coloring : EColoring),
    ∃ (cycle : list (vertex G)), is_cycle cycle ∧ length cycle = 4 ∧ monochromatic coloring cycle :=
by sorry

end exists_monochromatic_C4_l743_743362


namespace angela_more_sleep_in_january_l743_743694

theorem angela_more_sleep_in_january :
  (let dec_weekdays := 22 * 6.5,
       dec_weekends := 9 * 7.5,
       dec_sundays := 4 * 2.0,
       total_dec_sleep := dec_weekdays + dec_weekends + dec_sundays,

       jan_nights := 31 * 8.5,
       jan_sundays := 4 * 3.0,
       jan_1st := 5.0,
       total_jan_sleep := jan_nights + jan_sundays + jan_1st in

   total_jan_sleep - total_dec_sleep = 62.0) :=
by
  sorry

end angela_more_sleep_in_january_l743_743694


namespace target_heart_rate_30_year_old_l743_743302

theorem target_heart_rate_30_year_old :
  (let age := 30 in
  let max_heart_rate := 230 - age in
  let target_heart_rate := 0.7 * max_heart_rate in
  target_heart_rate ≈ 140) :=
by
  -- Definitions
  let age := 30
  let max_heart_rate := 230 - age
  let target_heart_rate := 0.7 * max_heart_rate

  -- Proof
  have h1 : max_heart_rate = 230 - 30 := rfl
  have h2: max_heart_rate = 200, by norm_num [h1]
  have h3 : target_heart_rate = 0.7 * 200, by congr; exact h2
  exact eq_of_approx_eq (by norm_num) h3 sorry

end target_heart_rate_30_year_old_l743_743302


namespace eval_expr_l743_743333

theorem eval_expr : 3 + 3 * (3 ^ (3 ^ 3)) - 3 ^ 3 = 22876792454937 := by
  sorry

end eval_expr_l743_743333


namespace ellipse_product_axes_l743_743909

/-- Prove that the product of the lengths of the major and minor axes (AB)(CD) of an ellipse
is 240, given the following conditions:
- Point O is the center of the ellipse.
- Point F is one focus of the ellipse.
- OF = 8
- The diameter of the inscribed circle of triangle OCF is 4.
- OA = OB = a
- OC = OD = b
- a² - b² = 64
- a - b = 4
-/
theorem ellipse_product_axes (a b : ℝ) (OF : ℝ) (d_inscribed_circle : ℝ) 
  (h1 : OF = 8) (h2 : d_inscribed_circle = 4) (h3 : a^2 - b^2 = 64) 
  (h4 : a - b = 4) : (2 * a) * (2 * b) = 240 :=
sorry

end ellipse_product_axes_l743_743909


namespace min_value_frac_sum_l743_743772

theorem min_value_frac_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  ∃ c, (∀ x y : ℝ, (0 < x) → (0 < y) → (x + y = 1) → (∃ c, c = 3 + 2*√2) → c = 3 + 2*√2) :=
sorry

end min_value_frac_sum_l743_743772


namespace derivative_value_l743_743761

/-- Given function definition -/
def f (x : ℝ) : ℝ := x^2 + 3 * x * (deriv f 2)

/-- Statement we need to prove -/
theorem derivative_value :
  deriv f 2 = -2 := 
sorry

end derivative_value_l743_743761


namespace isosceles_triangle_l743_743132

   open EuclideanGeometry

   -- Define the conditions of the problem in Lean
   variable {I A B C P Q M N : Point}
   variable (α : Circle) (circumcircle_AIC : Circle)

   -- Conditions extracted from the problem
   def conditions : Prop :=
   IsIncenter I △ABC ∧
   Incircle α △ABC ∧
   Circle.Diameter α P Q ∧
   Circle.Containing circumcircle_AIC (trianglePoint AIC) ∧
   SameSide P A (Line BI) ∧
   SameSide Q C (Line BI) ∧
   IsMidpointArc M ARC(α A C) ∧
   IsMidpointArc N ARC(α B C) ∧
   Parallel (Line PQ) (Line AC)

   -- Proof statement in Lean
   theorem isosceles_triangle
     (h : conditions α circumcircle_AIC) : IsIsosceles (△ABC) :=
   sorry
   
end isosceles_triangle_l743_743132


namespace probability_at_least_two_same_l743_743178

theorem probability_at_least_two_same (n : ℕ) (s : ℕ) (h_n : n = 8) (h_s : s = 8) :
  let total_outcomes := s ^ n
      different_outcomes := Nat.factorial s
      prob_all_different := different_outcomes / total_outcomes
      prob_at_least_two_same := 1 - prob_all_different
  in prob_at_least_two_same = 1291 / 1296 :=
by
  -- Define values
  have h_total_outcomes : total_outcomes = 16777216 := by sorry
  have h_different_outcomes : different_outcomes = 40320 := by sorry
  have h_prob_all_different : prob_all_different = 5 / 1296 := by sorry
  -- Calculate probability of at least two dice showing the same number
  have h_prob_at_least_two_same : prob_at_least_two_same = 1 - (5 / 1296) := by
    unfold prob_at_least_two_same prob_all_different
    rw h_different_outcomes
    rw h_total_outcomes
    rw h_prob_all_different
  -- Simplify
  calc
    prob_at_least_two_same = 1 - (5 / 1296) : by rw h_prob_at_least_two_same
    ... = 1291 / 1296 : by sorry

end probability_at_least_two_same_l743_743178


namespace false_statement_C_l743_743404

open Set

variables (α β : Plane) (l : Line) (P : Point)

-- Conditions
def planes_are_perpendicular : Prop := α ⊥ β
def intersection_is_line : Prop := α ∩ β = l
def point_in_alpha : Prop := P ∈ α
def point_not_in_line : Prop := P ∉ l

-- Theorem: To prove that the false statement is that a line through P perpendicular to β lies within α.
theorem false_statement_C (h1 : planes_are_perpendicular α β)
                         (h2 : intersection_is_line α β l)
                         (h3 : point_in_alpha α P)
                         (h4 : point_not_in_line l P) :
  ¬(∃ (m : Line), (P ∈ m) ∧ (m ⊥ β) ∧ (m ⊆ α)) :=
by
  sorry

end false_statement_C_l743_743404


namespace set_of_points_at_distance_from_line_is_two_parallel_lines_l743_743533

noncomputable def line (P Q : ℝ × ℝ) : ℝ → (ℝ × ℝ) :=
  λ t, (P.1 + t * (Q.1 - P.1), P.2 + t * (Q.2 - P.2))

noncomputable def distance_point_to_line (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ) : ℝ :=
  /- The definition of the distance from a point to a line is omitted for brevity -/
  sorry

theorem set_of_points_at_distance_from_line_is_two_parallel_lines :
  ∀ (l : ℝ → ℝ × ℝ) (h : ℝ), ∃ (l1 l2 : ℝ → ℝ × ℝ),
    (∀ P : ℝ × ℝ, distance_point_to_line P l = h ↔ (∃ t1, P = l1 t1) ∨ (∃ t2, P = l2 t2)) ∧
    (∀ t, ∃ k1 k2, l1 t = line (l 0) k1 t ∧ l2 t = line (l 0) k2 t) ∧
    (l1 ≠ l2 ∧ ∀ t, l1 t ≠ l2 t) ∧
    (∀ t, distance_point_to_line (l1 t) l = h ∧ distance_point_to_line (l2 t) l = h)
  :=
by
  sorry

end set_of_points_at_distance_from_line_is_two_parallel_lines_l743_743533


namespace number_of_odd_palindromes_l743_743164

def is_palindrome (n : ℕ) : Prop :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := n / 100
  n < 1000 ∧ n >= 100 ∧ d0 = d2

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

theorem number_of_odd_palindromes : ∃ n : ℕ, is_palindrome n ∧ is_odd n → n = 50 :=
by
  sorry

end number_of_odd_palindromes_l743_743164


namespace solve_inequality_l743_743198

theorem solve_inequality (a x : ℝ) : 
  if a > 0 then -a < x ∧ x < 2*a else if a < 0 then 2*a < x ∧ x < -a else False :=
by sorry

end solve_inequality_l743_743198


namespace arithmetic_sequence_third_term_l743_743205

theorem arithmetic_sequence_third_term
  (a d : ℤ)
  (h_fifteenth_term : a + 14 * d = 15)
  (h_sixteenth_term : a + 15 * d = 21) :
  a + 2 * d = -57 :=
by
  sorry

end arithmetic_sequence_third_term_l743_743205


namespace sum_of_odd_intervals_length_eq_1_div_4_l743_743586

def ceil_log4 (x : ℝ) : ℤ := ⌈Real.log x / Real.log 4⌉

theorem sum_of_odd_intervals_length_eq_1_div_4 :
  (∑ (I : ℝ → Prop) in {I | (exists k : ℤ, I = λ x, 4^k < x ∧ x ≤ 4^(k + 1 / 2))
                        ∧ (2 * Int.natAbs k + 1 ≠ 0)}, 
  Set.Ioo (4^k : ℝ) (4^(k + 1 / 2)) : ℝ) = 1 / 4 :=
sorry

end sum_of_odd_intervals_length_eq_1_div_4_l743_743586


namespace find_larger_number_l743_743576

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1311) (h2 : L = 11 * S + 11) : L = 1441 :=
sorry

end find_larger_number_l743_743576


namespace noah_class_size_l743_743443

theorem noah_class_size :
  ∀ n : ℕ, (n = 39 + 39 + 1) → n = 79 :=
by
  intro n
  intro h
  exact h

end noah_class_size_l743_743443


namespace right_triangles_count_l743_743940

theorem right_triangles_count (A P B C R D S : Type) 
  [rectangle A B C D]
  (PR_congruent_A_and_B : segment PR divides rectangle ABCD into two congruent squares)
  (S_on_PR : S ∈ segment PR)
  (S_divides_PR_1_3 : divides_segment S PR 1 3) :
  number_of_right_triangles_using_vertices {A, P, B, C, R, D, S} = 16 :=
sorry

end right_triangles_count_l743_743940


namespace isosceles_triangle_l743_743091

open_locale euclidean_geometry

variables {A B C I P Q M N : Point}
variables (α : circle)
variables (circumcircle_AIC : circle)

-- Condition 1
hypothesis h1 : incenter I (triangle.mk A B C)

-- Condition 2
hypothesis h2 : α = incircle (triangle.mk A B C)

-- Condition 3
hypothesis h3 : intersects circumcircle_AIC α P
hypothesis h4 : intersects circumcircle_AIC α Q

-- Condition 4
hypothesis h5 : same_side P A (line.mk B I)

-- Condition 5
hypothesis h6 : ¬ same_side Q C (line.mk B I)

-- Condition 6
hypothesis h7 : midpoint M (arc.mk α A C)

-- Condition 7
hypothesis h8 : midpoint N (arc.mk α B C)

-- Condition 8
hypothesis h9 : parallel (line.mk P Q) (line.mk A C)

-- Conclusion
theorem isosceles_triangle (h1 h2 h3 h4 h5 h6 h7 h8 h9) : (distance A B) = (distance A C) :=
sorry

end isosceles_triangle_l743_743091


namespace entertainment_team_count_l743_743723

theorem entertainment_team_count 
  (total_members : ℕ)
  (singers : ℕ) 
  (dancers : ℕ) 
  (prob_both_sing_dance_gt_0 : ℚ)
  (sing_count : singers = 2)
  (dance_count : dancers = 5)
  (prob_condition : prob_both_sing_dance_gt_0 = 7/10) :
  total_members = 5 := 
by 
  sorry

end entertainment_team_count_l743_743723


namespace fraction_simplification_result_l743_743849

theorem fraction_simplification_result :
  let num := list.foldl (/) 29 [28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16],
      denom := list.foldl (/) 15 [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
  in num / denom = 1292646 := 
sorry

end fraction_simplification_result_l743_743849


namespace linear_regression_neg_corr_l743_743788

-- Given variables x and y with certain properties
variables (x y : ℝ)

-- Conditions provided in the problem
def neg_corr (x y : ℝ) : Prop := ∀ a b : ℝ, (a < b → x * y < 0)
def sample_mean_x := (2 : ℝ)
def sample_mean_y := (1.5 : ℝ)

-- Statement to prove the linear regression equation
theorem linear_regression_neg_corr (h1 : neg_corr x y)
    (hx : sample_mean_x = 2)
    (hy : sample_mean_y = 1.5) : 
    ∃ b0 b1 : ℝ, b0 = 5.5 ∧ b1 = -2 ∧ y = b0 + b1 * x :=
sorry

end linear_regression_neg_corr_l743_743788


namespace measure_angle_H_l743_743187

-- Define the geometrical structure and conditions
variables (EFGH : Type) [parallelogram EFGH] 
variables (angle_EFG angle_F : ℝ)
variables [fact (angle_EFG = 40)] [fact (angle_F = 110)]

-- Define the angle measurement of angle H
noncomputable def angle_H : ℝ :=
  180 - (180 - angle_EFG)

-- State the theorem and the proof outline
theorem measure_angle_H :
  angle_H EFGH angle_EFG angle_F = 40 :=
by 
  unfold angle_H
  have h1: angle_EFG = 40 := fact.out _
  have h2: angle_F = 110 := fact.out _
  sorry

end measure_angle_H_l743_743187


namespace number_of_possible_ordered_pairs_l743_743922

theorem number_of_possible_ordered_pairs (n : ℕ) (f m : ℕ) 
  (cond1 : n = 6) 
  (cond2 : f ≥ 0) 
  (cond3 : m ≥ 0) 
  (cond4 : f + m ≤ 12) 
  : ∃ s : Finset (ℕ × ℕ), s.card = 6 := 
by 
  sorry

end number_of_possible_ordered_pairs_l743_743922


namespace probability_adjacent_points_l743_743923

open Finset

-- Define the hexagon points and adjacency relationship
def hexagon_points : Finset ℕ := {0, 1, 2, 3, 4, 5}

def adjacent (a b : ℕ) : Prop :=
  (a = b + 1 ∨ a = b - 1 ∨ (a = 0 ∧ b = 5) ∨ (a = 5 ∧ b = 0))

-- Total number of ways to choose 2 points from 6 points
def total_pairs := (hexagon_points.card.choose 2)

-- Number of pairs that are adjacent
def favorable_pairs := (6 : ℕ) -- Each point has exactly 2 adjacent points, counted twice

-- The probability of selecting two adjacent points
theorem probability_adjacent_points : (favorable_pairs : ℚ) / total_pairs = 2 / 5 :=
by {
  sorry
}

end probability_adjacent_points_l743_743923


namespace ratio_of_areas_l743_743190

-- Define the problem conditions
def regular_octa (ABCDEFGH : Set Point) : Prop := -- Regular octagon condition
sorry

def subdivided_into_equilateral_triangles (ABCDEFGH : Set Point) (ABJ ADE: Triangle) : Prop := 
-- Condition that the octagon is divided into smaller equilateral triangles
sorry

def larger_equilateral_triangle (ABCDEFGH : Set Point) (ADE : Triangle) : Prop := -- The condition about triangle ADE
sorry

-- Prove the desired ratio
theorem ratio_of_areas (ABCDEFGH : Set Point) (ABJ ADE : Triangle) 
  (h1 : regular_octa ABCDEFGH)
  (h2 : subdivided_into_equilateral_triangles ABCDEFGH ABJ ADE) 
  (h3 : larger_equilateral_triangle ABCDEFGH ADE):
  area ABJ / area ADE = 2 / 3 :=
sorry

end ratio_of_areas_l743_743190


namespace tom_age_ratio_l743_743616

-- Define the constants T and N with the given conditions
variables (T N : ℕ)
-- Tom's age T years, sum of three children's ages is also T
-- N years ago, Tom's age was three times the sum of children's ages then

-- We need to prove that T / N = 4 under these conditions
theorem tom_age_ratio (h1 : T = 3 * T - 8 * N) : T / N = 4 :=
sorry

end tom_age_ratio_l743_743616


namespace odd_prime_and_square_l743_743724

theorem odd_prime_and_square (P Q : ℕ → Prop)
  (h1 : ∀ p, prime p → p > 2 → odd p)
  (h2 : ∃ n, odd n ∧ ∃ m, n = m * m) :
  (∃ n, Q (n * n) ∧ odd (n * n)) ∧ (∃ p, prime p ∧ ¬ Q (p * p)) :=
by 
  sorry

end odd_prime_and_square_l743_743724


namespace sidney_kittens_l743_743919

theorem sidney_kittens
  (num_adult_cats : ℕ := 3)
  (num_initial_cans : ℕ := 7)
  (num_additional_cans : ℕ := 35)
  (days : ℕ := 7)
  (adult_cat_food_daily : ℕ := 1)
  (kitten_food_daily : ℚ := 3/4) :
  ∃ (K : ℕ), K = 4 :=
by
  have total_cans_needed := num_initial_cans + num_additional_cans
  have adult_cats_food := num_adult_cats * adult_cat_food_daily * days
  have food_for_kittens := total_cans_needed - adult_cats_food
  have kitten_food_total_per_kitten := kitten_food_daily * days
  have num_kittens := food_for_kittens / kitten_food_total_per_kitten
  use num_kittens
  have h : num_kittens = 4 := by sorry
  exact h

end sidney_kittens_l743_743919


namespace part_i_l743_743573

theorem part_i (M n : ℕ) (hM : M = 2011) (C : fin (M * n) → bool)
  (hCols : ∀ (i j : fin n), i ≠ j →
    (∑ k : fin M, (if C (k + i * M) = C (k + j * M) then 1 else 0)) <
    (∑ k : fin M, (if C (k + i * M) ≠ C (k + j * M) then 1 else 0))) :
  n ≤ 2012 :=
sorry

end part_i_l743_743573


namespace perfect_square_divisors_of_240_l743_743816

theorem perfect_square_divisors_of_240 : 
  (∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, 0 < k ∧ k < n → ¬(k = 1 ∨ k = 4 ∨ k = 16)) := 
sorry

end perfect_square_divisors_of_240_l743_743816


namespace largest_n_divides_3_n_l743_743878

noncomputable def Q := ∏ k in (finset.range 150).map (function.embedding.mk (λ k, 2*k + 1) $ λ x y h, by { injection h with h1, cases h1, refl }), (2 * k + 1)

theorem largest_n_divides_3_n :
  ∃ n : ℕ, (Q % 3^n = 0) ∧ (∀ m : ℕ, (Q % 3^m = 0) → m ≤ 76) :=
sorry

end largest_n_divides_3_n_l743_743878


namespace sum_of_first_5n_l743_743830

theorem sum_of_first_5n (n : ℕ) : 
  (n * (n + 1) / 2) + 210 = ((4 * n) * (4 * n + 1) / 2) → 
  (5 * n) * (5 * n + 1) / 2 = 465 :=
by sorry

end sum_of_first_5n_l743_743830


namespace cone_surface_area_is_1_l743_743781

noncomputable def coneSurfaceArea (d : ℝ) (isSemiCircle : Bool) : ℝ :=
  if isSemiCircle then
    let r := d / 2
    let l := 2 * π * r / (2 * π)
    let baseArea := π * (r ^ 2)
    let lateralArea := π * r * l / 2
    baseArea + lateralArea
  else
    0

theorem cone_surface_area_is_1 :
  coneSurfaceArea (2 * sqrt (3 * π) / (3 * π)) true = 1 :=
by
  sorry

end cone_surface_area_is_1_l743_743781


namespace total_water_carried_l743_743280

/-- Define the capacities of the four tanks in each truck -/
def tank1_capacity : ℝ := 200
def tank2_capacity : ℝ := 250
def tank3_capacity : ℝ := 300
def tank4_capacity : ℝ := 350

/-- The total capacity of one truck -/
def total_truck_capacity : ℝ := tank1_capacity + tank2_capacity + tank3_capacity + tank4_capacity

/-- Define the fill percentages for each truck -/
def fill_percentage (truck_number : ℕ) : ℝ :=
if truck_number = 1 then 1
else if truck_number = 2 then 0.75
else if truck_number = 3 then 0.5
else if truck_number = 4 then 0.25
else 0

/-- Define the amounts of water each truck carries -/
def water_carried_by_truck (truck_number : ℕ) : ℝ :=
(fill_percentage truck_number) * total_truck_capacity

/-- Prove that the total amount of water the farmer can carry in his trucks is 2750 liters -/
theorem total_water_carried : 
  water_carried_by_truck 1 + water_carried_by_truck 2 + water_carried_by_truck 3 +
  water_carried_by_truck 4 + water_carried_by_truck 5 = 2750 :=
by sorry

end total_water_carried_l743_743280


namespace approximate_wheat_amount_l743_743459

-- Define the conditions
def total_rice : ℕ := 1534
def sample_size : ℕ := 254
def wheat_in_sample : ℕ := 28

-- Define the theorem
theorem approximate_wheat_amount :
  (total_rice * (wheat_in_sample / sample_size : ℚ)).toReal ≈ 169.1 :=
by
  sorry

end approximate_wheat_amount_l743_743459


namespace sampling_methods_correct_l743_743679

theorem sampling_methods_correct :
  ( ∃ school : Type,
      ∃ students : school → Prop,
      (∃ student_council_selection : set school,
        student_council_selection.card = 20 ∧
        ∀ s ∈ student_council_selection, students s ) ∧
      ∃ academic_affairs_selection : set school,
        academic_affairs_selection.card = 20 ∧
        ( ∀ s ∈ academic_affairs_selection,
          let student_number := decode_student_number s in
          student_number % 10 = 2 ∧
          students s ) ) →
  (sampling_method student_council_selection = simple_random_sampling ∧
   sampling_method academic_affairs_selection = systematic_sampling) :=
begin
  sorry
end

end sampling_methods_correct_l743_743679


namespace second_team_alone_days_l743_743561

noncomputable def second_team_days : ℕ := 45

def first_team_days : ℕ := 30
def initial_work_days : ℕ := 10
def total_days : ℕ := 30
def early_finish : ℕ := 8

theorem second_team_alone_days :
  ∃ (x : ℕ), x = second_team_days ∧ 
    let fraction_work_done := initial_work_days / first_team_days,
        remaining_work := 1 - fraction_work_done,
        remaining_days := total_days - initial_work_days - early_finish,
        combined_rate := (1 / x + 1 / first_team_days) in
    remaining_days * combined_rate = remaining_work :=
begin
  use 45,
  split,
  { refl },
  { 
    let fraction_work_done := initial_work_days / first_team_days,
    let remaining_work := 1 - fraction_work_done, 
    let remaining_days := total_days - initial_work_days - early_finish,
    let combined_rate := (1 / second_team_days + 1 / first_team_days),
    show remaining_days * combined_rate = remaining_work,
    sorry -- The proof itself is not required by the prompt
  }
end

end second_team_alone_days_l743_743561


namespace solution_eq1_solution_eq2_l743_743197

-- Define the first equation
def eq1 (x : ℝ) := 2 * (x + 1)^2 = 8

-- Prove that 1 and -3 are solutions to the first equation
theorem solution_eq1 : ∀ (x : ℝ), eq1 x ↔ (x = 1 ∨ x = -3) :=
by
  intros x
  split
  · intro h
    have h₁ : (x + 1)^2 = 4 := by linarith
    have eq_sqrt := eq_or_eq_neg_of_sq_eq h₁
    cases eq_sqrt with plus_sqrt minus_sqrt
    · right; linarith
    · left; linarith
  · intro h
    cases h with hx1 hx2
    · rw hx1; norm_num
    · rw hx2; norm_num

-- Define the second equation
def eq2 (x : ℝ) := 2 * x^2 - x - 6 = 0

-- Prove that -3/2 and 2 are solutions to the second equation
theorem solution_eq2 : ∀ (x : ℝ), eq2 x ↔ (x = -3/2 ∨ x = 2) :=
by
  intros x
  split
  · intro h
    have : (2 * x + 3) * (x - 2) = 0 := by
      have := by ring_exp
      linarith
    have eq_fac := mul_eq_zero.mp this
    cases eq_fac with fac1 fac2
    · right; linarith
    · left; linarith
  · intro h
    cases h with hx1 hx2
    · rw hx1; norm_num
    · rw hx2; norm_num

end solution_eq1_solution_eq2_l743_743197


namespace divisibility_by_19_l743_743890

theorem divisibility_by_19 (N : ℕ) (transform : ℕ → ℕ) :
  (transform = λ x : ℕ, (x / 10) + 2 * (x % 10)) →
  (∃ k : ℕ, ∀ i ≤ k, transform^[i] N ≤ 19) →
  (N % 19 = 0 ↔ (transform^[Nat.find (exists (λ i, transform^[i] N = 19))] N = 19)) :=
by
  sorry

end divisibility_by_19_l743_743890


namespace true_proposition_l743_743457

-- Conditions defined as propositions in Lean
def propositionA : Prop :=
  ∀ (A B C : Point), ∃! (P : Plane), A ∈ P ∧ B ∈ P ∧ C ∈ P

def propositionB : Prop :=
  ∀ (l m n : Line), (l ⊥ n ∧ m ⊥ n) → (l ∥ m)

def propositionC : Prop :=
  ∀ (∠A ∠B : Angle), (corresponding_sides_parallel ∠A ∠B) → (∠A = ∠B)

def propositionD : Prop :=
  ∀ (P Q : Plane), (P ∥ Q) → ∀ l, (l ∈ P) → (l ∥ Q)

-- The only universally true proposition
theorem true_proposition : propositionD :=
  by sorry

end true_proposition_l743_743457


namespace probability_divisible_l743_743999

theorem probability_divisible (h : ∀ x : ℕ, x ∈ {1, 2, 3, 4, 5, 6}) :
  let Ω := { (x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} },
      event := { (x, y) ∈ Ω | x % y = 0 } in
  (card event : ℚ) / (card Ω) = 7 / 18 :=
by sorry

end probability_divisible_l743_743999


namespace circle_cartesian_eq_min_distance_circle_to_line_l743_743853

noncomputable def polarToCartesianCircle (ρ θ : ℝ) : Prop :=
  ρ = Math.cos θ + Math.sin θ

noncomputable def polarToCartesianLine (ρ θ : ℝ) : Prop :=
  ρ = 2 * Math.sqrt 2 / (Math.cos (θ + Real.pi / 4))

theorem circle_cartesian_eq
  (θ ρ : ℝ)
  (hC : polarToCartesianCircle ρ θ) :
  ∃ x y : ℝ, (x - 1 / 2)^2 + (y - 1 / 2)^2 = 1 / 2 :=
sorry

theorem min_distance_circle_to_line
  (θ1 ρ1 θ2 ρ2 : ℝ)
  (hC : polarToCartesianCircle ρ1 θ1)
  (hL : polarToCartesianLine ρ2 θ2) :
  ∃ d : ℝ, d = 3 * Math.sqrt 2 / 2 :=
sorry

end circle_cartesian_eq_min_distance_circle_to_line_l743_743853


namespace part1_l743_743260

theorem part1 (a b c : ℤ) (h : a + b + c = 0) : a^3 + a^2 * c - a * b * c + b^2 * c + b^3 = 0 := 
sorry

end part1_l743_743260


namespace max_barons_in_32_knights_l743_743028

noncomputable def knight : Type := sorry  -- Define the type for knights
def is_vassal (vassal liege : knight) : Prop := sorry -- Define the vassal relationship
def wealthier (k1 k2 : knight) : Prop := sorry -- Define the wealth relation

def is_baron (k : knight) : Prop := ∃ (vassals : finset knight), vassals.card ≥ 4 ∧ ∀ v, v ∈ vassals → is_vassal v k ∧ wealthier k v

def max_barons (knights : finset knight) : ℕ :=
  sorry -- Define the function to calculate the maximum number of barons

theorem max_barons_in_32_knights (knights : finset knight) (h32 : knights.card = 32) :
  max_barons knights = 7 :=
sorry

end max_barons_in_32_knights_l743_743028


namespace A_should_shoot_air_l743_743229

-- Define the problem conditions
def hits_A : ℝ := 0.3
def hits_B : ℝ := 1
def hits_C : ℝ := 0.5

-- Define turns
inductive Turn
| A | B | C

-- Define the strategic choice
inductive Strategy
| aim_C | aim_B | shoot_air

-- Define the outcome structure
structure DuelOutcome where
  winner : Option Turn
  probability : ℝ

-- Noncomputable definition given the context of probabilistic reasoning
noncomputable def maximize_survival : Strategy := 
sorry

-- Main theorem to prove the optimal strategy
theorem A_should_shoot_air : maximize_survival = Strategy.shoot_air := 
sorry

end A_should_shoot_air_l743_743229


namespace hexagon_triangulation_count_l743_743422

theorem hexagon_triangulation_count : 
  ∃ n, n = 12 ∧ is_regular_hexagon n ∧ (∀ triangle, (triangle_is_in_hexagon triangle) → (at_least_one_side_in_polygon triangle)) :=
by
  sorry

def is_regular_hexagon (n : ℕ) : Prop :=
  n = 6

def triangle_is_in_hexagon (triangle : triangle) : Prop :=
  -- Condition that all vertices of the triangle are vertices of the hexagon.
  sorry

def at_least_one_side_in_polygon (triangle : triangle) : Prop :=
  -- Condition that at least one side of the triangle is a side of the hexagon.
  sorry

end hexagon_triangulation_count_l743_743422


namespace equation_of_line_l743_743378

noncomputable def angle_inclination_45_slope : ℝ := 1
noncomputable def y_intercept : ℝ := 2

theorem equation_of_line (m : ℝ) (b : ℝ): m = angle_inclination_45_slope ∧ b = y_intercept → ∃ (x y : ℝ), x - y + 2 = 0 :=
by
  intros h
  rcases h with ⟨hm, hb⟩
  use (λ x, x), (λ y, y)
  sorry

end equation_of_line_l743_743378


namespace ABC_is_isosceles_l743_743121

open Mobius

variables
  (ABC : Triangle)
  (I : Point)
  (α P Q M N : Point)
  (A B C Z : Point)
  [Incenter I ABC]
  [Incircle α ABC]
  [Circumcircle (Triangle.mk A I C)]
  [PQ_parallel_AC : is_parallel (Segment.mk P Q) (Segment.mk A C)]
  [Midpoint M (Arc.mk A C α)]
  [Midpoint N (Arc.mk B C α)]
  [center_Z : Center Z (Circumcircle (Triangle.mk A I C))]
  [Chord_α_PQ : common_chord α (Circumcircle (Triangle.mk A I C)) (Segment.mk P Q)]
  [P_A_same_BI : same_side P A (Line.mk B I)]
  [Q_C_other_BI : same_side Q C (Line.mk B I)]

theorem ABC_is_isosceles
  (h_parallel : PQ_parallel_AC) : 
  Isosceles ABC := 
sorry

end ABC_is_isosceles_l743_743121


namespace common_difference_of_arithmetic_sequence_l743_743451

open Nat

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, ∃ d, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * a 1 + (n * (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum_terms : sum_of_first_n_terms a S)
  (h_condition : S 2020 / 2020 - S 20 / 20 = 2000) :
  d = 2 :=
sorry

end common_difference_of_arithmetic_sequence_l743_743451


namespace more_seventh_graders_than_sixth_graders_l743_743548

theorem more_seventh_graders_than_sixth_graders 
  (n m : ℕ)
  (H1 : ∀ x : ℕ, x = n → 7 * n = 6 * m) : 
  m > n := 
by
  -- Proof is not required and will be skipped with sorry.
  sorry

end more_seventh_graders_than_sixth_graders_l743_743548


namespace find_ellipse_and_hyperbola_equations_l743_743763

-- Define the conditions
def eccentricity (e : ℝ) (a b : ℝ) : Prop :=
  e = (Real.sqrt (a ^ 2 - b ^ 2)) / a

def focal_distance (f : ℝ) (a b : ℝ) : Prop :=
  f = 2 * Real.sqrt (a ^ 2 + b ^ 2)

-- Define the problem to prove the equations of the ellipse and hyperbola
theorem find_ellipse_and_hyperbola_equations (a b : ℝ) (e : ℝ) (f : ℝ)
  (h1 : eccentricity e a b) (h2 : focal_distance f a b) 
  (h3 : e = 4 / 5) (h4 : f = 2 * Real.sqrt 34) 
  (h5 : a > b) (h6 : 0 < b) :
  (a^2 = 25 ∧ b^2 = 9) → 
  (∀ x y, (x^2 / 25 + y^2 / 9 = 1) ∧ (x^2 / 25 - y^2 / 9 = 1)) :=
sorry

end find_ellipse_and_hyperbola_equations_l743_743763


namespace ellipse_product_axes_l743_743908

/-- Prove that the product of the lengths of the major and minor axes (AB)(CD) of an ellipse
is 240, given the following conditions:
- Point O is the center of the ellipse.
- Point F is one focus of the ellipse.
- OF = 8
- The diameter of the inscribed circle of triangle OCF is 4.
- OA = OB = a
- OC = OD = b
- a² - b² = 64
- a - b = 4
-/
theorem ellipse_product_axes (a b : ℝ) (OF : ℝ) (d_inscribed_circle : ℝ) 
  (h1 : OF = 8) (h2 : d_inscribed_circle = 4) (h3 : a^2 - b^2 = 64) 
  (h4 : a - b = 4) : (2 * a) * (2 * b) = 240 :=
sorry

end ellipse_product_axes_l743_743908


namespace john_got_rolls_l743_743479

def cost_per_dozen : ℕ := 5
def money_spent : ℕ := 15
def rolls_per_dozen : ℕ := 12

theorem john_got_rolls : (money_spent / cost_per_dozen) * rolls_per_dozen = 36 :=
by sorry

end john_got_rolls_l743_743479


namespace dogs_running_l743_743226

theorem dogs_running (total_dogs playing_with_toys barking not_doing_anything running : ℕ)
  (h1 : total_dogs = 88)
  (h2 : playing_with_toys = total_dogs / 2)
  (h3 : barking = total_dogs / 4)
  (h4 : not_doing_anything = 10)
  (h5 : running = total_dogs - playing_with_toys - barking - not_doing_anything) :
  running = 12 :=
sorry

end dogs_running_l743_743226


namespace range_of_a_l743_743808

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + 2 * x + a ≤ 0) ↔ a ≤ 1 :=
by
  sorry

end range_of_a_l743_743808


namespace polynomials_property_l743_743338

noncomputable theory

open Polynomial

variable (P Q : Polynomial ℝ) 

theorem polynomials_property 
  (h : ∀ x : ℝ, P.eval (Q.eval x) = (P.eval x)^2017) :
  ∃ (r : ℝ) (n : ℕ), P = C(1 : ℝ) * (X - C(r))^n ∧ Q = (X - C(r))^2017 + C(r) :=
sorry

end polynomials_property_l743_743338


namespace weekly_goal_l743_743633

theorem weekly_goal (a : ℕ) (d : ℕ) (n : ℕ) (h1 : a = 20) (h2 : d = 5) (h3 : n = 5) :
  ∑ i in finset.range n, a + i * d = 150 :=
by
  sorry

end weekly_goal_l743_743633


namespace magnitude_of_vector_sum_l743_743405

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hab : ∀ θ, θ = real.pi / 3 → inner a b = ‖a‖ * ‖b‖ * real.cos θ)

theorem magnitude_of_vector_sum :
  ‖a + b‖ = real.sqrt 3 := by
  sorry

end magnitude_of_vector_sum_l743_743405


namespace at_least_one_nonzero_l743_743946

theorem at_least_one_nonzero (a b : ℝ) (h : a^2 + b^2 ≠ 0) : a ≠ 0 ∨ b ≠ 0 :=
by
  sorry

end at_least_one_nonzero_l743_743946


namespace reciprocal_of_repeating_decimal_l743_743988

theorem reciprocal_of_repeating_decimal :
  (1 / (0.33333333 : ℚ)) = 3 := by
  sorry

end reciprocal_of_repeating_decimal_l743_743988


namespace printed_X_value_is_244_l743_743826

-- Definitions based on conditions
def initial_X := 4
def initial_S := 0
def increment_X := 2
def threshold := 15000

-- Main theorem statement
theorem printed_X_value_is_244 :
  let X_sequence (n : ℕ) := initial_X + n * increment_X in
  let S_sequence (n : ℕ) := n * (n + 3) in
  (∃ n : ℕ, S_sequence n ≥ threshold ∧ X_sequence n = 244) :=
sorry  -- Proof to be filled in

end printed_X_value_is_244_l743_743826


namespace problem1_problem2_l743_743702

theorem problem1 : 2 * (Real.sqrt 3 - Real.sqrt 5) + 3 * (Real.sqrt 3 + Real.sqrt 5) = 5 * Real.sqrt 3 + Real.sqrt 5 :=
by
  sorry

theorem problem2 : -1^2 - abs(1 - Real.sqrt 3) + (8 : ℝ)^(1/3) - (-3) * Real.sqrt 9 = 11 - Real.sqrt 3 :=
by
  sorry

end problem1_problem2_l743_743702


namespace positive_integers_count_l743_743421

theorem positive_integers_count :
  let cond1 := 100 ≤ n / 3 < 1000
      cond2 := 100 ≤ 3 * n + 1 < 1000
  in ∃! (n : ℕ), cond1 ∧ cond2 :=
begin
  sorry
end

end positive_integers_count_l743_743421


namespace intersection_points_l743_743705

def floor_fractional_part (x : ℝ) : ℝ :=
  x - real.floor x

theorem intersection_points :
  ∃ P ∈ ℝ×ℝ, 
    ((floor_fractional_part P.1) ^ 2 + (P.2) ^ 2 = (floor_fractional_part P.1)) ∧ 
    (P.2 = (P.1) ^ 2) 
    ∧ -- There are exactly 4 such points
    sorry

end intersection_points_l743_743705


namespace snickers_bars_needed_l743_743472

-- Definitions of the conditions
def points_needed : ℕ := 2000
def chocolate_bunny_points : ℕ := 100
def number_of_chocolate_bunnies : ℕ := 8
def snickers_points : ℕ := 25

-- Derived conditions
def points_from_bunnies : ℕ := number_of_chocolate_bunnies * chocolate_bunny_points
def remaining_points : ℕ := points_needed - points_from_bunnies

-- Statement to prove
theorem snickers_bars_needed : ∀ (n : ℕ), n = remaining_points / snickers_points → n = 48 :=
by 
  sorry

end snickers_bars_needed_l743_743472


namespace total_student_capacity_l743_743838

-- Define the conditions
def school_capacity_one : ℕ := 400
def school_capacity_two : ℕ := 340
def number_of_schools_one : ℕ := 2
def number_of_schools_two : ℕ := 2

-- Statement to prove
theorem total_student_capacity :
  (number_of_schools_one * school_capacity_one) +
  (number_of_schools_two * school_capacity_two) = 1480 :=
by
  calc
    (number_of_schools_one * school_capacity_one) +
    (number_of_schools_two * school_capacity_two)
    = (2 * 400) + (2 * 340) : by sorry
    = 800 + 680 : by sorry
    = 1480 : by sorry

end total_student_capacity_l743_743838


namespace isosceles_triangle_l743_743094

open_locale euclidean_geometry

variables {A B C I P Q M N : Point}
variables (α : circle)
variables (circumcircle_AIC : circle)

-- Condition 1
hypothesis h1 : incenter I (triangle.mk A B C)

-- Condition 2
hypothesis h2 : α = incircle (triangle.mk A B C)

-- Condition 3
hypothesis h3 : intersects circumcircle_AIC α P
hypothesis h4 : intersects circumcircle_AIC α Q

-- Condition 4
hypothesis h5 : same_side P A (line.mk B I)

-- Condition 5
hypothesis h6 : ¬ same_side Q C (line.mk B I)

-- Condition 6
hypothesis h7 : midpoint M (arc.mk α A C)

-- Condition 7
hypothesis h8 : midpoint N (arc.mk α B C)

-- Condition 8
hypothesis h9 : parallel (line.mk P Q) (line.mk A C)

-- Conclusion
theorem isosceles_triangle (h1 h2 h3 h4 h5 h6 h7 h8 h9) : (distance A B) = (distance A C) :=
sorry

end isosceles_triangle_l743_743094


namespace min_value_is_8_plus_4_sqrt_3_l743_743407

noncomputable def min_value_of_expression (a b : ℝ) : ℝ :=
  2 / a + 1 / b

theorem min_value_is_8_plus_4_sqrt_3 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 3 * a + 2 * b = 1) :
  min_value_of_expression a b = 8 + 4 * Real.sqrt 3 :=
sorry

end min_value_is_8_plus_4_sqrt_3_l743_743407


namespace jim_gas_tank_capacity_l743_743475

/-- Jim has 2/3 of a tank left after a round-trip of 20 miles where he gets 5 miles per gallon.
    Prove that the capacity of Jim's gas tank is 12 gallons. --/
theorem jim_gas_tank_capacity
    (remaining_fraction : ℚ)
    (round_trip_distance : ℚ)
    (fuel_efficiency : ℚ)
    (used_fraction : ℚ)
    (used_gallons : ℚ)
    (total_capacity : ℚ)
    (h1 : remaining_fraction = 2/3)
    (h2 : round_trip_distance = 20)
    (h3 : fuel_efficiency = 5)
    (h4 : used_fraction = 1 - remaining_fraction)
    (h5 : used_gallons = round_trip_distance / fuel_efficiency)
    (h6 : used_gallons = used_fraction * total_capacity) :
  total_capacity = 12 :=
sorry

end jim_gas_tank_capacity_l743_743475


namespace new_tv_cost_l743_743521

/-
Mark bought his first TV which was 24 inches wide and 16 inches tall. It cost $672.
His new TV is 48 inches wide and 32 inches tall.
The first TV was $1 more expensive per square inch compared to his newest TV.
Prove that the cost of his new TV is $1152.
-/

theorem new_tv_cost :
  let width_first_tv := 24
  let height_first_tv := 16
  let cost_first_tv := 672
  let width_new_tv := 48
  let height_new_tv := 32
  let discount_per_square_inch := 1
  let area_first_tv := width_first_tv * height_first_tv
  let cost_per_square_inch_first_tv := cost_first_tv / area_first_tv
  let cost_per_square_inch_new_tv := cost_per_square_inch_first_tv - discount_per_square_inch
  let area_new_tv := width_new_tv * height_new_tv
  let cost_new_tv := cost_per_square_inch_new_tv * area_new_tv
  cost_new_tv = 1152 := by
  sorry

end new_tv_cost_l743_743521


namespace average_annual_growth_rate_sales_revenue_2018_l743_743036

-- Define the conditions as hypotheses
def initial_sales := 200000
def final_sales := 800000
def years := 2
def growth_rate := 1.0 -- representing 100%

theorem average_annual_growth_rate (x : ℝ) :
  (initial_sales : ℝ) * (1 + x)^years = final_sales → x = 1 :=
by
  intro h1
  sorry

theorem sales_revenue_2018 (x : ℝ) (revenue_2017 : ℝ) :
  x = 1 → revenue_2017 = final_sales → revenue_2017 * (1 + x) = 1600000 :=
by
  intros h1 h2
  sorry

end average_annual_growth_rate_sales_revenue_2018_l743_743036


namespace domain_of_f_l743_743577

-- Given a function f such that f(x) = log(2x - x^2)
def f (x : ℝ) : ℝ := log(2 * x - x^2)

-- Define the condition under which the logarithmic function is defined
def cond (x : ℝ) : Prop := 2 * x - x^2 > 0

-- Prove that the domain of the function is (0, 2)
theorem domain_of_f : {x : ℝ | cond x} = {x : ℝ | 0 < x ∧ x < 2} :=
sorry

end domain_of_f_l743_743577


namespace P_not_good_set_Q_is_good_set_good_set_addition_l743_743329

variable (M : Set ℚ) -- We assume we're working with rational numbers.

def is_good_set (S : Set ℚ) : Prop :=
  (0 ∈ S) ∧ (1 ∈ S) ∧ (∀ x y, x ∈ S → y ∈ S → x - y ∈ S) ∧ (∀ x, x ∈ S → x ≠ 0 → 1 / x ∈ S)

-- (Ⅰ) Verification of sets P and Q as 'good sets'
theorem P_not_good_set : ¬ is_good_set { -1, 0, 1 } := by
  sorry

theorem Q_is_good_set : is_good_set ℚ := by
  sorry

-- (Ⅱ) Proof of x + y ∈ A for a 'good set' A
theorem good_set_addition (A : Set ℚ) (hA : is_good_set A) :
  ∀ x y, x ∈ A → y ∈ A → x + y ∈ A := by
  intro x y hx hy
  sorry

end P_not_good_set_Q_is_good_set_good_set_addition_l743_743329


namespace geometric_progression_reciprocal_sum_l743_743510

theorem geometric_progression_reciprocal_sum (n : ℕ) (r : ℝ) (s' : ℝ) 
(h_s' : s' = (2 * (1 - (2 * r)^n)) / (1 - 2 * r)) :
  ∑ i in Finset.range n, 1 / (2 * r)^i = s' / (2 * r)^(n - 1) := 
sorry

end geometric_progression_reciprocal_sum_l743_743510


namespace problem1_problem2_l743_743703

variable (x y : ℝ)

-- Problem 1
theorem problem1 : (x + y) ^ 2 + x * (x - 2 * y) = 2 * x ^ 2 + y ^ 2 := by
  sorry

variable (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) -- to ensure the denominators are non-zero

-- Problem 2
theorem problem2 : (x ^ 2 - 6 * x + 9) / (x - 2) / (x + 2 - (3 * x - 4) / (x - 2)) = (x - 3) / x := by
  sorry

end problem1_problem2_l743_743703


namespace isosceles_triangle_of_parallel_PQ_AC_l743_743136

variables {α β γ : Type}
variables {A B C I M N P Q : α} {circleIncircle circCircumcircleAIC : set α}
variables {lineBI linePQ lineAC : set α}

-- Given conditions from the problem
def incenter_of_triangle (I A B C : α) : Prop := sorry
def incircle (circle α : set α) (A B C I : α) : Prop := sorry
def circumcircle (circle circumcircleAIC : set α) (A I C : α) : Prop := sorry
def midpoint_of_arc (M N : α) (circle α : set α) (A B C : α) : Prop := sorry
def parallel (linePQ lineAC : set α) : Prop := sorry
def lies_on_same_side_of_line (P A : α) (line lineBI : set α) : Prop := sorry
def lies_on_opposite_sides_of_line (Q C : α) (line lineBI : set α) : Prop := sorry

theorem isosceles_triangle_of_parallel_PQ_AC :
  incenter_of_triangle I A B C ∧
  incircle circleIncircle A B C I ∧
  circumcircle circCircumcircleAIC A I C ∧
  midpoint_of_arc M circleIncircle A C ∧
  midpoint_of_arc N circleIncircle B C ∧
  lies_on_same_side_of_line P A lineBI ∧
  lies_on_opposite_sides_of_line Q C lineBI ∧
  parallel linePQ lineAC →
  (triangle_isosceles A B C) := sorry

end isosceles_triangle_of_parallel_PQ_AC_l743_743136


namespace isosceles_triangle_l743_743128

   open EuclideanGeometry

   -- Define the conditions of the problem in Lean
   variable {I A B C P Q M N : Point}
   variable (α : Circle) (circumcircle_AIC : Circle)

   -- Conditions extracted from the problem
   def conditions : Prop :=
   IsIncenter I △ABC ∧
   Incircle α △ABC ∧
   Circle.Diameter α P Q ∧
   Circle.Containing circumcircle_AIC (trianglePoint AIC) ∧
   SameSide P A (Line BI) ∧
   SameSide Q C (Line BI) ∧
   IsMidpointArc M ARC(α A C) ∧
   IsMidpointArc N ARC(α B C) ∧
   Parallel (Line PQ) (Line AC)

   -- Proof statement in Lean
   theorem isosceles_triangle
     (h : conditions α circumcircle_AIC) : IsIsosceles (△ABC) :=
   sorry
   
end isosceles_triangle_l743_743128


namespace max_size_T_l743_743501

open Finset

def S : Finset (Vector Bool 7) := univ

def dist (a b : Vector Bool 7) : ℕ := 
  (Finset.range 7).sum (λ i, if a[i] = b[i] then 0 else 1)

noncomputable def T (Tsub : Finset (Vector Bool 7)) : Prop :=
  ∀ a b ∈ Tsub, a ≠ b → dist a b ≥ 3

theorem max_size_T (Tsub : Finset (Vector Bool 7)) (h : T Tsub) : Tsub.card ≤ 16 := 
  sorry

end max_size_T_l743_743501


namespace pascal_theorem_l743_743500

-- Definitions from conditions of the problem
variables {A B C D E F K L M : Type*} [point : Type*]

-- Defining the given hexagon and its properties
variables (A B C D E F : point)
variables (K : point) (L : point)

-- Inscribed hexagon
def inscribed_hexagon (A B C D E F : point) : Prop :=
  ∃ (circle : Set point), ∀ p ∈ {A, B, C, D, E, F}, p ∈ circle

-- Intersection points
def intersection (p q u v : point) : Prop :=
  ∃ w, line_through w p q ∧ line_through w u v

-- Conditions as given intersections
axiom hACBF : intersection A C B F K
axiom hCEFD : intersection C E F D L

-- Definitions of lines and intersection through a point
def collinear (p q r : point) : Prop :=
  ∃ (line : Set point), p ∈ line ∧ q ∈ line ∧ r ∈ line

def line_through (p1 p2 : point) : Set point := {p : point | collinear p p1 p2}

-- Pascal's Theorem implication
theorem pascal_theorem (A B C D E F K L M : point) :
  inscribed_hexagon A B C D E F →
  intersection A C B F K →
  intersection C E F D L →
  ∃ P Q R, 
    intersection A B D E P ∧ 
    intersection B C E F Q ∧ 
    intersection C D F A R →
  collinear P Q R →
  ∃ M, intersection A D K L M ∧ intersection B E K L M :=
begin
  intros h_inscribed h1 h2,
  use some_point, -- some_point needs to be defined as per corresponding existence proof
  sorry
end

end pascal_theorem_l743_743500


namespace right_triangle_area_l743_743787

theorem right_triangle_area (x y : ℝ) 
  (h1 : x + y = 4) 
  (h2 : x^2 + y^2 = 9) : 
  (1/2) * x * y = 7 / 4 := 
by
  sorry

end right_triangle_area_l743_743787


namespace curve_crossing_self_l743_743751

theorem curve_crossing_self (t t' : ℝ) :
  (t^3 - t - 2 = t'^3 - t' - 2) ∧ (t ≠ t') ∧ 
  (t^3 - t^2 - 9 * t + 5 = t'^3 - t'^2 - 9 * t' + 5) → 
  (t = 3 ∧ t' = -3) ∨ (t = -3 ∧ t' = 3) →
  (t^3 - t - 2 = 22) ∧ (t^3 - t^2 - 9 * t + 5 = -4) :=
by
  sorry

end curve_crossing_self_l743_743751


namespace num_whole_numbers_between_sqrt_50_and_sqrt_200_l743_743423

theorem num_whole_numbers_between_sqrt_50_and_sqrt_200 :
  let lower := Nat.ceil (Real.sqrt 50)
  let upper := Nat.floor (Real.sqrt 200)
  lower <= upper ∧ (upper - lower + 1) = 7 :=
by
  sorry

end num_whole_numbers_between_sqrt_50_and_sqrt_200_l743_743423


namespace find_x_l743_743862

namespace IntegerProblem

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := 
by
  sorry

end IntegerProblem

end find_x_l743_743862


namespace four_digit_perfect_square_with_pairs_l743_743728

-- Definition of a four-digit number that can be written in the form aabb
def is_aabb (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ fin 10 ∧ b ∈ fin 10 ∧ n = 1000 * a + 100 * a + 10 * b + b

-- Definition of a number being a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

-- The main theorem combining both properties
theorem four_digit_perfect_square_with_pairs : 1000 ≤ 7744 ∧ 7744 ≤ 9999 ∧ is_perfect_square 7744 ∧ is_aabb 7744 :=
by
  sorry

end four_digit_perfect_square_with_pairs_l743_743728


namespace equivalent_statements_l743_743245

variables (R L : Prop)

theorem equivalent_statements :
  (R → L) ↔ (¬L → ¬R) ∧ (¬R ∨ L) :=
begin
  split,
  { intro h,
    split,
    { intro hl,
      exact h hl, }, 
    { by_cases hr : R,
      { right,
        exact h hr, },
      { left, 
        exact hr, }, }, },
  { intros h hr,
    cases h,
    by_cases hl : L,
    { exact hl, },
    { exfalso,
      exact h_left hl hr, }, }, 
end

end equivalent_statements_l743_743245


namespace OptionC_is_incorrect_l743_743514

-- Define the function D(x)
def D (x : ℝ) : ℝ := if (∃ q : ℚ, x = q) then 1 else 0

-- Statement to prove Option C is incorrect
theorem OptionC_is_incorrect : (∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, D(x + T) = D(x)) :=
by
  -- We propose T = 1
  use 1
  split
  -- T ≠ 0
  norm_num
  -- D(x + T) = D(x)
  intro x
  dsimp [D]
  by_cases (∃ q : ℚ, x = q) with h₁;
  simp [h₁]
  sorry

end OptionC_is_incorrect_l743_743514


namespace product_of_major_and_minor_axes_l743_743910

-- Given definitions from conditions
variables (O F A B C D : Type) 
variables (OF : ℝ) (dia_inscribed_circle_OCF : ℝ) (a b : ℝ)

-- Condition: O is the center of an ellipse
-- Point F is one focus, OF = 8
def O_center_ellipse : Prop := OF = 8

-- The diameter of the inscribed circle of triangle OCF is 4
def dia_inscribed_circle_condition : Prop := dia_inscribed_circle_OCF = 4

-- Define OA = OB = a, OC = OD = b
def major_axis_half_length : ℝ := a
def minor_axis_half_length : ℝ := b

-- Ellipse focal property a^2 - b^2 = 64
def ellipse_focal_property : Prop := a^2 - b^2 = 64

-- From the given conditions, expected result
def compute_product_AB_CD : Prop := 
  let AB := 2 * a,
      CD := 2 * b
  in AB * CD = 240

-- The main statement to be proven
theorem product_of_major_and_minor_axes 
  (h1 : O_center_ellipse)
  (h2 : dia_inscribed_circle_condition)
  (h3 : ellipse_focal_property)
  : compute_product_AB_CD :=
sorry

end product_of_major_and_minor_axes_l743_743910


namespace max_no_of_Xs_l743_743832

-- Define the 3x3 grid and the condition
def Grid := Fin 3 × Fin 3

def isValidPlacement (X_positions : Finset Grid) : Prop :=
  ∀ (row_col_diag : List (Fin 3 × Fin 3)), row_col_diag ∈ [[(0, 0), (0, 1), (0, 2)],  -- Row 1
                                                         [(1, 0), (1, 1), (1, 2)],  -- Row 2
                                                         [(2, 0), (2, 1), (2, 2)],  -- Row 3
                                                         [(0, 0), (1, 0), (2, 0)],  -- Column 1
                                                         [(0, 1), (1, 1), (2, 1)],  -- Column 2
                                                         [(0, 2), (1, 2), (2, 2)],  -- Column 3
                                                         [(0, 0), (1, 1), (2, 2)],  -- Diagonal 1
                                                         [(0, 2), (1, 1), (2, 0)]], -- Diagonal 2
    row_col_diag.countp (λ pos, pos ∈ X_positions) < 3

-- Define the theorem
theorem max_no_of_Xs (X_positions : Finset Grid) (h : isValidPlacement X_positions) :
  X_positions.card ≤ 4 :=
sorry

end max_no_of_Xs_l743_743832


namespace arrange_points_with_effective_circles_l743_743032

/-- A set of 8 points in the plane, such that any 4 points lie on an effective circle. 
An "effective circle" is one that passes through exactly 4 of these points. -/
noncomputable def effective_circle {P : Type*} [Plane P] (points : Set P) : Prop :=
  ∃ C : Set P, (C ⊆ points ∧ C.card = 4 ∧ is_circle C)

/-- Prove that there exists an arrangement of 8 points in the plane
such that the number of "effective circles" is at least 12.
An "effective circle" refers to a circle passing through exactly 4 of these points. -/
theorem arrange_points_with_effective_circles :
  ∃ (P : Type*) [Plane P] (points : Set P), points.card = 8 ∧ 
  (∃ S : Set (Set P), S ⊆ {C : Set P | effective_circle C} ∧ S.card ≥ 12) :=
sorry

end arrange_points_with_effective_circles_l743_743032


namespace domain_of_log2_function_l743_743937

theorem domain_of_log2_function :
  {x : ℝ | 2 * x - 1 > 0} = {x : ℝ | x > 1 / 2} :=
by
  sorry

end domain_of_log2_function_l743_743937


namespace difference_two_numbers_l743_743603

theorem difference_two_numbers (a b : ℕ) (h₁ : a + b = 20250) (h₂ : b % 15 = 0) (h₃ : a = b / 3) : b - a = 10130 :=
by 
  sorry

end difference_two_numbers_l743_743603


namespace sum_coeffs_eq_l743_743019

noncomputable def sum_poly_coeffs (n : ℕ) (a : ℕ → ℕ) (h : (1 + X)^n = ∑ i in range (n + 1), a i * X^i) : ℕ :=
a 0 + ∑ i in range (n + 1), i * a i

theorem sum_coeffs_eq (n : ℕ) (a : ℕ → ℕ) (h : (1 + X)^n = ∑ i in range (n + 1), a i * X^i)
    (h0 : a 0 = 1) :
    sum_poly_coeffs n a h = n * 2^(n-1) + 1 :=
sorry

end sum_coeffs_eq_l743_743019


namespace ABC_is_isosceles_l743_743124

open Mobius

variables
  (ABC : Triangle)
  (I : Point)
  (α P Q M N : Point)
  (A B C Z : Point)
  [Incenter I ABC]
  [Incircle α ABC]
  [Circumcircle (Triangle.mk A I C)]
  [PQ_parallel_AC : is_parallel (Segment.mk P Q) (Segment.mk A C)]
  [Midpoint M (Arc.mk A C α)]
  [Midpoint N (Arc.mk B C α)]
  [center_Z : Center Z (Circumcircle (Triangle.mk A I C))]
  [Chord_α_PQ : common_chord α (Circumcircle (Triangle.mk A I C)) (Segment.mk P Q)]
  [P_A_same_BI : same_side P A (Line.mk B I)]
  [Q_C_other_BI : same_side Q C (Line.mk B I)]

theorem ABC_is_isosceles
  (h_parallel : PQ_parallel_AC) : 
  Isosceles ABC := 
sorry

end ABC_is_isosceles_l743_743124


namespace BoatsRUs_total_canoes_l743_743309

def totalCanoesBuiltByJuly (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem BoatsRUs_total_canoes :
  totalCanoesBuiltByJuly 5 3 7 = 5465 :=
by
  sorry

end BoatsRUs_total_canoes_l743_743309


namespace system1_solution_system2_solution_l743_743559

-- System (1)
theorem system1_solution {x y : ℝ} : 
  x + y = 3 → 
  x - y = 1 → 
  (x = 2 ∧ y = 1) :=
by
  intros h1 h2
  -- proof goes here
  sorry

-- System (2)
theorem system2_solution {x y : ℝ} :
  2 * x + y = 3 →
  x - 2 * y = 1 →
  (x = 7 / 5 ∧ y = 1 / 5) :=
by
  intros h1 h2
  -- proof goes here
  sorry

end system1_solution_system2_solution_l743_743559


namespace boxes_given_to_brother_l743_743899

-- Definitions
def total_boxes : ℝ := 14.0
def pieces_per_box : ℝ := 6.0
def pieces_remaining : ℝ := 42.0

-- Theorem stating the problem
theorem boxes_given_to_brother : 
  (total_boxes * pieces_per_box - pieces_remaining) / pieces_per_box = 7.0 := 
by
  sorry

end boxes_given_to_brother_l743_743899


namespace find_interest_rate_l743_743663

theorem find_interest_rate
  (P : ℝ) (t : ℕ) (I : ℝ)
  (hP : P = 3000)
  (ht : t = 5)
  (hI : I = 750) :
  ∃ r : ℝ, I = P * r * t / 100 ∧ r = 5 :=
by 
  sorry

end find_interest_rate_l743_743663


namespace tip_percentage_l743_743231

variable (L : ℝ) (T : ℝ)
 
theorem tip_percentage (h : L = 60.50) (h1 : T = 72.6) :
  ((T - L) / L) * 100 = 20 :=
by
  sorry

end tip_percentage_l743_743231


namespace purple_pairs_coincide_l743_743030

theorem purple_pairs_coincide 
  (G O P : ℕ)
  (green_pairs orange_pairs green_purple_pairs: ℕ) 
  (hG : G = 4)
  (hO : O = 6)
  (hP : P = 10)
  (h_green_pairs : green_pairs = 3)
  (h_orange_pairs : orange_pairs = 4)
  (h_green_purple_pairs : green_purple_pairs = 1) 
  :
  P - green_purple_pairs = 9 :=
by {
  rw [hP, h_green_purple_pairs],
  exact nat.sub_self 9,
}

end purple_pairs_coincide_l743_743030


namespace parallelogram_area_l743_743253

def base : ℝ := 12
def height : ℝ := 10
def area_of_parallelogram (b h : ℝ) : ℝ := b * h

theorem parallelogram_area : area_of_parallelogram base height = 120 := by
  sorry

end parallelogram_area_l743_743253


namespace value_of_a_l743_743829

theorem value_of_a (n a : ℝ) (h1 : 2 ^ n = 64) (h2 : n = 6) 
  (h3 : ∑ k in finset.range (n + 1), nat.choose n k = 64) 
  (h4 : let T_r := λ r, nat.choose 6 r * (√ x)^(6 - r) * (-a)^r / x^r in
        T_r 2 = 15) : a = 1 ∨ a = -1 := 
by
  sorry

end value_of_a_l743_743829


namespace sum_of_double_scored_values_l743_743620

theorem sum_of_double_scored_values :
  ∃ S : ℕ, (0 ≤ S ∧ S ≤ 140) ∧ (∑' ( s : ℕ ) in (finset.range 141).filter (λ s,
    (∃ c u i: ℕ, c + u + i = 20 ∧ S = 7 * c + 3 * u ∧ u = (S - 7 * c) / 3 
    ∧ 3 * i = 60 - S + 4 * c ∧ 0 ≤ i ∧ (S - 7 * c) % 3 = 0 ∧
    ( (λ x, ∃ c', ∃ u', ( c' + u' + i = 20 ∧ S = 7 * c' + 3 * u' ∧ u' = (S - 7 * c')
        / 3  ∧ 3 * (i) = 60 - S + 4 * c' ∧ 0 ≤ (i) ∧ (S - 7 * c') % 3 = 0)) (s)))), S) = 164 :=
sorry

end sum_of_double_scored_values_l743_743620


namespace ferry_tourist_total_l743_743664

theorem ferry_tourist_total :
  let number_of_trips := 8
  let a := 120 -- initial number of tourists
  let d := -2  -- common difference
  let total_tourists := (number_of_trips * (2 * a + (number_of_trips - 1) * d)) / 2
  total_tourists = 904 := 
by {
  sorry
}

end ferry_tourist_total_l743_743664


namespace coffee_shop_sold_lattes_l743_743932

theorem coffee_shop_sold_lattes (T L : ℕ) (h1 : T = 6) (h2 : L = 4 * T + 8) : L = 32 :=
by
  sorry

end coffee_shop_sold_lattes_l743_743932


namespace min_value_of_magnitude_of_c_l743_743519

theorem min_value_of_magnitude_of_c :
  (∀ (x y : ℝ), let a := (x - 2, 2) 
                let b := (4, y) 
                let c := (x, y)
                (a.1 * b.1 + a.2 * b.2 = 0) →
                ∃ (x y : ℝ), sqrt ((x * x + y * y)) = sqrt (5 * (x - 8 / 5) ^ 2 + 16 / 5) ∧ 
                sqrt (5 * (x - 8 / 5) ^ 2 + 16 / 5) ≥ 4 * sqrt(5) / 5) :=
begin
  intros x y a b c h,
  sorry
end

end min_value_of_magnitude_of_c_l743_743519


namespace grasshopper_jump_distance_l743_743210

-- Define variables
variable {grasshopper_jump frog_jump : ℕ}

-- Conditions
def frog_jumps := frog_jump = 12
def frog_farther := frog_jump = grasshopper_jump + 3

-- Statement to prove
theorem grasshopper_jump_distance : frog_jumps ∧ frog_farther → grasshopper_jump = 9 :=
by sorry

end grasshopper_jump_distance_l743_743210


namespace exists_two_natural_pairs_satisfying_equation_l743_743731

theorem exists_two_natural_pairs_satisfying_equation :
  ∃ (x1 y1 x2 y2 : ℕ), (2 * x1^3 = y1^4) ∧ (2 * x2^3 = y2^4) ∧ ¬(x1 = x2 ∧ y1 = y2) :=
sorry

end exists_two_natural_pairs_satisfying_equation_l743_743731


namespace stratified_sampling_third_grade_l743_743667

theorem stratified_sampling_third_grade (total_students : ℕ)
  (ratio_first_second_third : ℕ × ℕ × ℕ)
  (sample_size : ℕ) (r1 r2 r3 : ℕ) (h_ratio : ratio_first_second_third = (r1, r2, r3)) :
  total_students = 3000  ∧ ratio_first_second_third = (2, 3, 1)  ∧ sample_size = 180 →
  (sample_size * r3 / (r1 + r2 + r3) = 30) :=
sorry

end stratified_sampling_third_grade_l743_743667


namespace exists_k_in_octahedron_l743_743463

theorem exists_k_in_octahedron
  (x0 y0 z0 : ℚ)
  (h : ∀ n : ℤ, x0 + y0 + z0 ≠ n ∧ 
                 x0 + y0 - z0 ≠ n ∧ 
                 x0 - y0 + z0 ≠ n ∧ 
                 x0 - y0 - z0 ≠ n) :
  ∃ k : ℕ, ∃ (xk yk zk : ℚ), 
    k ≠ 0 ∧ 
    xk = k * x0 ∧ 
    yk = k * y0 ∧ 
    zk = k * z0 ∧
    ∀ n : ℤ, 
      (xk + yk + zk < ↑n → xk + yk + zk > ↑(n - 1)) ∧ 
      (xk + yk - zk < ↑n → xk + yk - zk > ↑(n - 1)) ∧ 
      (xk - yk + zk < ↑n → xk - yk + zk > ↑(n - 1)) ∧ 
      (xk - yk - zk < ↑n → xk - yk - zk > ↑(n - 1)) :=
sorry

end exists_k_in_octahedron_l743_743463


namespace Johnson_potatoes_left_l743_743067

noncomputable def Gina_potatoes : ℝ := 93.5
noncomputable def Tom_potatoes : ℝ := 3.2 * Gina_potatoes
noncomputable def Anne_potatoes : ℝ := (2/3) * Tom_potatoes
noncomputable def Jack_potatoes : ℝ := (1/7) * (Gina_potatoes + Anne_potatoes)
noncomputable def Total_given_away : ℝ := Gina_potatoes + Tom_potatoes + Anne_potatoes + Jack_potatoes
noncomputable def Initial_potatoes : ℝ := 1250
noncomputable def Potatoes_left : ℝ := Initial_potatoes - Total_given_away

theorem Johnson_potatoes_left : Potatoes_left = 615.98 := 
  by
    sorry

end Johnson_potatoes_left_l743_743067


namespace geometric_sequence_general_term_sum_inequality_l743_743048

noncomputable def sequence (n : ℕ) : ℕ → ℝ
| 0       := 2
| (n + 1) := (2 * sequence n) / (sequence n + 1)

theorem geometric_sequence {n : ℕ} :
  let b := λ n, (1 / sequence n) - 1 in
  ∃ r : ℝ, r = 1 / 2 ∧ ∀ k : ℕ, b (k + 1) = r * b k :=
by sorry

theorem general_term {n : ℕ} :
  sequence n = 2^n / (2^n - 1) :=
by sorry

theorem sum_inequality {n : ℕ} :
  (∑ i in finset.range n.succ, sequence i * (sequence i - 1)) < 3 :=
by sorry

end geometric_sequence_general_term_sum_inequality_l743_743048


namespace find_m_l743_743891

open Set

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 3}

-- Define the complement of set A in U
def A_c : Set ℕ := {1, 2}

-- Define the set A such that A = { x ∈ U | x^2 + mx = 0 }
def A (m : ℤ) : Set ℕ := { x | x ∈ U ∧ x^2 + (m * x : ℕ) = 0 }

-- The proof statement
theorem find_m (m : ℤ) (h : A_c = U \ (A m)) : m = -3 := by
  sorry

end find_m_l743_743891


namespace intervals_of_increase_range_of_m_l743_743000

section
variables {x : ℝ} {m : ℝ}
noncomputable def a := (2 * cos x, 1)
noncomputable def b := (cos x, sqrt 3 * sin (2 * x) + m)
noncomputable def f (x : ℝ) := (2 * cos x) * (cos x) + 1 * (sqrt 3 * sin (2 * x) + m) - 1

-- Translate the first question
theorem intervals_of_increase (hx : 0 ≤ x ∧ x ≤ π) : (0 ≤ x ∧ x ≤ π/6) ∨ (2 * π / 3 ≤ x ∧ x ≤ π) :=
sorry

-- Translate the second question
theorem range_of_m (hx : 0 ≤ x ∧ x ≤ π/6) (h : -4 ≤ f x ∧ f x ≤ 4) : -5 ≤ m ∧ m ≤ 2 :=
sorry
end

end intervals_of_increase_range_of_m_l743_743000


namespace car_rally_odd_n_l743_743640

theorem car_rally_odd_n (n : ℕ) (h1 : n ≥ 2) 
    (h2 : ∀ i j, i ≠ j → ∃ m_i m_j, m_i ≠ m_j ∧ m_i ∈ finset.range n ∧ m_j ∈ finset.range n)
    (h3 : ∃ k, ∀ i, (∃ m_i, m_i ∈ finset.range n) ∧ ∀ i, overtaken_by i = k) : 
    odd n := 
sorry

end car_rally_odd_n_l743_743640


namespace num_real_permutations_l743_743847

-- Define the conditions
def is_permutation_1234 (x₁ x₂ x₃ x₄ : ℕ) : Prop :=
  {x₁, x₂, x₃, x₄} = {1, 2, 3, 4}

def is_real_expression (x₁ x₂ x₃ x₄ : ℕ) : Prop :=
  (x₁ - x₂ + x₃ - x₄ : ℤ) ≥ 0

-- Formalize the proof goal
theorem num_real_permutations : 
  (finset.univ.filter (λ s : fin n 4, let ⟨x₁, x₂, x₃, x₄⟩ := s in 
    is_permutation_1234 x₁ x₂ x₃ x₄ ∧ is_real_expression x₁ x₂ x₃ x₄)).card = 16 :=
sorry

end num_real_permutations_l743_743847


namespace bucket_full_weight_l743_743628

variable (c d : ℝ)

def total_weight_definition (x y : ℝ) := x + y

theorem bucket_full_weight (x y : ℝ) 
  (h₁ : x + 3/4 * y = c) 
  (h₂ : x + 1/3 * y = d) : 
  total_weight_definition x y = (8 * c - 3 * d) / 5 :=
sorry

end bucket_full_weight_l743_743628


namespace equilateral_triangle_DEF_l743_743721

theorem equilateral_triangle_DEF 
  (A B C D E F : Point)
  (hParallelogram : parallelogram A B C D)
  (hEquilateral_ABE : equilateral_triangle A B E)
  (hEquilateral_BCF : equilateral_triangle B C F)
  (D_eq : D = parallelogram_opposite C B A)
  (E_eq : E = equilateral_vertex A B)
  (F_eq : F = equilateral_vertex B C) :
  equilateral_triangle D E F :=
  sorry

end equilateral_triangle_DEF_l743_743721


namespace domain_ln_l743_743324

theorem domain_ln (x : ℝ) : (1 - 2 * x > 0) ↔ x < (1 / 2) :=
by
  sorry

end domain_ln_l743_743324


namespace square_area_l743_743700

theorem square_area (P : ℝ) (s : ℝ) (A : ℝ) :
  P = 36 → P = 4 * s → A = s^2 → A = 81 :=
by
  intros hP hPs hAsq
  rw hP at hPs
  sorry

end square_area_l743_743700


namespace percentage_increase_l743_743872

theorem percentage_increase (original new : ℕ) (h₀ : original = 60) (h₁ : new = 120) :
  ((new - original) / original) * 100 = 100 := by
  sorry

end percentage_increase_l743_743872


namespace eccentricity_of_ellipse_l743_743793

theorem eccentricity_of_ellipse (m : ℝ) (h₁ : m > 0) (h₂ : ∃ (x y : ℝ), x^2 / 16 + y^2 / m^2 = 1 ∧ y = (√2 / 2) * x ∧ x = (√(16 - m^2))) :
  (√2 / 2) = (√(16 - m^2)) / 4 :=
by
  sorry

end eccentricity_of_ellipse_l743_743793


namespace tangent_point_is_2_l743_743790

theorem tangent_point_is_2 :
  ∀ (x : ℝ), (x > 0) → (deriv (λ x : ℝ, (x ^ 2) / 2 - 2 * log x + 1) x = 1) → (x = 2) :=
by
  intros x hx hderiv
  sorry

end tangent_point_is_2_l743_743790


namespace angle_AOC_l743_743782

variables (O A B C : Type) [InnerProductSpace ℝ O]
variables (radius : ℝ) (OA OB OC : O)
-- Given conditions
axiom circumcircle_center : dist O A = dist O B ∧ dist O B = dist O C ∧ dist O C = radius
axiom vector_eq : OA + (real.sqrt 3) • OB + 2 • OC = (0 : O)

-- The proof goal
theorem angle_AOC (h_radius : radius = 1) : 
  ∃ θ : ℝ, θ = 2 * real.pi / 3 ∧ 
  ∃ c : ℝ, ∀ A B C : O, OA ⬝ OC = -1/2 :=
sorry

end angle_AOC_l743_743782


namespace tan_alpha_third_quadrant_l743_743375

variable (α : Real)
variable (cos_α : Real := -12/13)
variable (sin_α : Real := -sqrt(25/169))

theorem tan_alpha_third_quadrant (h1: -π < α ∧ α < -π/2) (h2: cos α = -12/13) : tan α = 5/12 :=
by
  sorry

end tan_alpha_third_quadrant_l743_743375


namespace largest_possible_roads_in_graphia_l743_743044

theorem largest_possible_roads_in_graphia :
  ∃ (G : Type) [Graph G], 
    (vertices G = 100) ∧ 
    (∀ v : G, degree v = 2) ∧
    (maximize_edges G = 4851) := 
sorry

end largest_possible_roads_in_graphia_l743_743044


namespace problem1_problem2_l743_743081

noncomputable def S (n : ℕ) : ℚ := (3 / 2) * n^2 - (1 / 2) * n

noncomputable def a (n : ℕ) : ℚ :=
  if n = 1 then S 1 else S n - S (n - 1)

noncomputable def b (n : ℕ) : ℚ :=
  3 / (a n * a (n + 1))

noncomputable def T (n : ℕ) : ℚ :=
  ∑ i in Finset.range n, b (i + 1)

theorem problem1 (n : ℕ) (h : n > 0) : a n = 3 * n - 2 :=
  sorry

theorem problem2 (n : ℕ) : ∃ m : ℕ, T n < m / 20 := 
  ∃m, m = 20 :=
  sorry

end problem1_problem2_l743_743081


namespace ninth_term_arithmetic_sequence_l743_743206

theorem ninth_term_arithmetic_sequence :
  ∀ (a_9 : ℚ), 
    (∃ (a : ℚ) (d : ℚ), a = 3 / 4 ∧ (a + 16 * d) = 1 / 2)
    → a_9 = (3 / 4 + 1 / 2) / 2 :=
begin
  assume a_9,
  assume h,
  rcases h with ⟨a, d, ha, h17⟩,
  rw [ha, h17],
  simp,
  sorry
end

end ninth_term_arithmetic_sequence_l743_743206


namespace sale_in_fourth_month_l743_743282

variable (sale1 sale2 sale3 sale5 sale6 sale4 : ℕ)

def average_sale (total : ℕ) (months : ℕ) : ℕ := total / months

theorem sale_in_fourth_month
  (h1 : sale1 = 6435)
  (h2 : sale2 = 6927)
  (h3 : sale3 = 6855)
  (h5 : sale5 = 6562)
  (h6 : sale6 = 7391)
  (avg : average_sale (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) 6 = 6900) :
  sale4 = 7230 := 
sorry

end sale_in_fourth_month_l743_743282


namespace ab_value_l743_743918

theorem ab_value (a b : ℝ) :
  (A = { x : ℝ | x^2 - 8 * x + 15 = 0 }) ∧
  (B = { x : ℝ | x^2 - a * x + b = 0 }) ∧
  (A ∪ B = {2, 3, 5}) ∧
  (A ∩ B = {3}) →
  (a * b = 30) :=
by
  sorry

end ab_value_l743_743918


namespace max_value_fx_l743_743214

theorem max_value_fx : 
  ∃ x ∈ set.Icc 1 2, ∀ y ∈ set.Icc 1 2, (x - exp x) ≥ (y - exp y) ∧ (x - exp x) = 1 - exp 1 := by
  sorry

end max_value_fx_l743_743214


namespace max_peaceful_clients_kept_l743_743609

-- Defining the types for knights, liars, and troublemakers
def Person : Type := ℕ

noncomputable def isKnight : Person → Prop := sorry
noncomputable def isLiar : Person → Prop := sorry
noncomputable def isTroublemaker : Person → Prop := sorry

-- Total number of people in the bar
def totalPeople : ℕ := 30

-- Number of knights, liars, and troublemakers
def numberKnights : ℕ := 10
def numberLiars : ℕ := 10
def numberTroublemakers : ℕ := 10

-- The bartender's goal: get rid of all troublemakers and keep as many peaceful clients as possible
def maxPeacefulClients (total: ℕ) (knights: ℕ) (liars: ℕ) (troublemakers: ℕ): ℕ :=
  total - troublemakers

-- Statement to be proved
theorem max_peaceful_clients_kept (total: ℕ) (knights: ℕ) (liars: ℕ) (troublemakers: ℕ)
  (h_total : total = 30)
  (h_knights : knights = 10)
  (h_liars : liars = 10)
  (h_troublemakers : troublemakers = 10) :
  maxPeacefulClients total knights liars troublemakers = 19 :=
by
  -- Proof steps go here
  sorry

end max_peaceful_clients_kept_l743_743609


namespace volume_of_convex_solid_l743_743219

variables {m V t6 T t3 : ℝ} 

-- Definition of the distance between the two parallel planes
def distance_between_planes (m : ℝ) : Prop := m > 0

-- Areas of the two parallel faces
def area_hexagon_face (t6 : ℝ) : Prop := t6 > 0
def area_triangle_face (t3 : ℝ) : Prop := t3 > 0

-- Area of the cross-section of the solid with a plane perpendicular to the height at its midpoint
def area_cross_section (T : ℝ) : Prop := T > 0

-- Volume of the convex solid
def volume_formula_holds (V m t6 T t3 : ℝ) : Prop :=
  V = (m / 6) * (t6 + 4 * T + t3)

-- Formal statement of the problem
theorem volume_of_convex_solid
  (m t6 t3 T V : ℝ)
  (h₁ : distance_between_planes m)
  (h₂ : area_hexagon_face t6)
  (h₃ : area_triangle_face t3)
  (h₄ : area_cross_section T) :
  volume_formula_holds V m t6 T t3 :=
by
  sorry

end volume_of_convex_solid_l743_743219


namespace area_of_quadrilateral_l743_743518

noncomputable theory

variable (A B C D E F G H M : Type)
variable (length : ℝ) (area_ABCM : ℝ)

-- Conditions
axiom regular_octagon : (A B C D E F G H : ℕ → Point) ∧
  (∀ (i : ℕ), (dist (A i) (A (i+1) % 8) = length = 5))
axiom diagonals_intersect : (exists M : Point, M = intersectLines (Line A E) (Line C G))

-- Theorem
theorem area_of_quadrilateral : 
  ∀ (A B C D E F G H M : Point) (length : ℝ),
  regular_octagon ∧ diagonals_intersect →
  area_ABCM = 25 * sqrt 2 :=
begin
  sorry
end

end area_of_quadrilateral_l743_743518


namespace find_g_5_l743_743580

theorem find_g_5 (g : ℝ → ℝ) (h : ∀ x : ℝ, g x + 3 * g (1 - x) = 2 * x ^ 2 + 1) : g 5 = 8 :=
sorry

end find_g_5_l743_743580


namespace probability_card_less_equal_9_l743_743742

/-- 
Prove that the probability of getting a card with a number less than or equal to 9 
when drawing one of the following number cards: 1, 3, 4, 6, 7, and 9 is 1.
-/
theorem probability_card_less_equal_9 : 
  (Finset.card {x : ℕ | x ∈ {1, 3, 4, 6, 7, 9} ∧ x ≤ 9}) / 
  (Finset.card {1, 3, 4, 6, 7, 9} : ℕ) = 1 := 
by
  sorry

end probability_card_less_equal_9_l743_743742


namespace distance_between_points_l743_743737

theorem distance_between_points : 
  let p1 := (-5, -2 : ℝ)
  let p2 := (7, 3 : ℝ)
  let dist := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  dist = 13 :=
by
  let p1 := (-5, -2 : ℝ)
  let p2 := (7, 3 : ℝ)
  let dist := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  sorry

end distance_between_points_l743_743737


namespace light_path_length_l743_743074

def AB : ℝ := 12
def BG : ℝ := 12
def BC : ℝ := 12
def distance_P_BG : ℝ := 7
def distance_P_BC : ℝ := 5

theorem light_path_length :
  ∀ (m n : ℤ), (n = 218) ∧ (m = 12) ∧ m + n = 230 :=
by {
  existsi 12,
  existsi 218,
  simp,
  sorry
}

end light_path_length_l743_743074


namespace partition_contains_exponential_l743_743885

open Set

theorem partition_contains_exponential (m : ℕ) (hm : m = 32) :
  ∀ A B : Finset ℕ, (A ∪ B = { n | 5 ≤ n ∧ n ≤ m }) →
  (∃ a b c ∈ A, a^b = c) ∨ (∃ a b c ∈ B, a^b = c) :=
by
  sorry

end partition_contains_exponential_l743_743885


namespace find_x_l743_743861

namespace IntegerProblem

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := 
by
  sorry

end IntegerProblem

end find_x_l743_743861


namespace min_abs_sum_l743_743083

theorem min_abs_sum (a b c d : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a^2 + b * c = 9) (h2 : b * c + d^2 = 9) (h3 : a * b + b * d = 0) (h4 : a * c + c * d = 0) :
  |a| + |b| + |c| + |d| = 8 :=
sorry

end min_abs_sum_l743_743083


namespace min_value_of_f_solution_set_of_inequality_l743_743154

-- Define the given function f
def f (x : ℝ) : ℝ := abs (x - 1) + abs (2 * x + 4)

-- (1) Prove that the minimum value of y = f(x) is 3
theorem min_value_of_f : ∃ x : ℝ, f x = 3 := 
sorry

-- (2) Prove that the solution set of the inequality |f(x) - 6| ≤ 1 is [-10/3, -8/3] ∪ [0, 4/3]
theorem solution_set_of_inequality : 
  {x | |f x - 6| ≤ 1} = {x | -(10/3) ≤ x ∧ x ≤ -(8/3) ∨ 0 ≤ x ∧ x ≤ (4/3)} :=
sorry

end min_value_of_f_solution_set_of_inequality_l743_743154


namespace triangle_perimeter_eq_12_l743_743589

noncomputable def triangle_perimeter (x y : ℝ) : ℝ :=
  let A := (3 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 4 : ℝ)
  let AB := Real.sqrt (3^2 + 4^2)
  3 + 4 + AB

theorem triangle_perimeter_eq_12 : 
  triangle_perimeter 3 4 = 12 :=
by
  sorry

end triangle_perimeter_eq_12_l743_743589


namespace zero_count_in_decimal_of_fraction_l743_743415

   theorem zero_count_in_decimal_of_fraction :
     let n := 3
     let d := 2^7 * 5^9
     let frac := n / d
     ∃ z: ℕ, has_zeros_count frac 8 :=
   sorry
   
end zero_count_in_decimal_of_fraction_l743_743415


namespace mutual_independence_A_B_mutual_independence_A_C_non_independence_B_C_mutual_exclusiveness_B_D_l743_743833

-- Definitions of events A, B, C, and D:
def eventA (s : set ℕ) : bool := (1 ∈ s ∧ 3 ∈ s) ∨ (2 ∈ s ∧ 4 ∈ s) ∨ (5 ∈ s ∧ 6 ∈ s)
def eventB (s : set ℕ) : bool := ∃ x y, x ∈ s ∧ y ∈ s ∧ abs (x - y) = 1
def eventC (s : set ℕ) : bool := ∃ x y, x ∈ s ∧ y ∈ s ∧ (x + y = 6 ∨ x + y = 7)
def eventD (s : set ℕ) : bool := ∃ x y, x ∈ s ∧ y ∈ s ∧ x * y = 5

-- Total number of outcomes:
def total_outcomes := 15

-- Probabilities:
def PA := 3 / total_outcomes
def PB := 5 / total_outcomes
def PC := 5 / total_outcomes
def PD := 1 / total_outcomes

-- Proving the conditions:
theorem mutual_independence_A_B : PA * PB = 1 / total_outcomes := sorry
theorem mutual_independence_A_C : PA * PC = 1 / total_outcomes := sorry
theorem non_independence_B_C : PB * PC ≠ 1 / total_outcomes := sorry
theorem mutual_exclusiveness_B_D : ∀ s, eventB s → ¬ eventD s := sorry

end mutual_independence_A_B_mutual_independence_A_C_non_independence_B_C_mutual_exclusiveness_B_D_l743_743833


namespace sum_of_numbers_l743_743263

def a : ℝ := 217
def b : ℝ := 2.017
def c : ℝ := 0.217
def d : ℝ := 2.0017

theorem sum_of_numbers :
  a + b + c + d = 221.2357 :=
by
  sorry

end sum_of_numbers_l743_743263


namespace john_buys_36_rolls_l743_743481

-- Definitions of the conditions
def cost_per_dozen := 5
def total_money_spent := 15
def rolls_per_dozen := 12

-- Theorem statement: John bought 36 rolls
theorem john_buys_36_rolls :
  let dozens_bought := total_money_spent / cost_per_dozen in
  let total_rolls := dozens_bought * rolls_per_dozen in
  total_rolls = 36 :=
by
  -- Proof steps would go here
  sorry

end john_buys_36_rolls_l743_743481


namespace Gunther_typing_correct_l743_743450

def GuntherTypingProblem : Prop :=
  let first_phase := (160 * (120 / 3))
  let second_phase := (200 * (180 / 3))
  let third_phase := (50 * 60)
  let fourth_phase := (140 * (90 / 3))
  let total_words := first_phase + second_phase + third_phase + fourth_phase
  total_words = 26200

theorem Gunther_typing_correct : GuntherTypingProblem := by
  sorry

end Gunther_typing_correct_l743_743450


namespace range_of_a_l743_743517

noncomputable def f (a x : ℝ) : ℝ := x - Real.log (a * x + 2 * a + 1) + 2

theorem range_of_a (a : ℝ) :
  (∀ x ≥ -2, f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
begin
  sorry
end

end range_of_a_l743_743517


namespace probability_two_red_jellybeans_l743_743270

theorem probability_two_red_jellybeans :
  let total_jellybeans := 10
  let red_jellybeans := 4
  let blue_jellybeans := 1
  let white_jellybeans := 5
  let total_picked := 3
  let total_ways := Nat.choose total_jellybeans total_picked
  let red_ways := Nat.choose red_jellybeans 2
  let non_red_ways := total_jellybeans - red_jellybeans
  let pick_one_non_red := 6 -- as given by total non-red jellybeans (1 blue + 5 white)
  let successful_outcomes := red_ways * pick_one_non_red
  (successful_outcomes / total_ways).val = 3 / 10 :=
by
  sorry

end probability_two_red_jellybeans_l743_743270


namespace p_sufficient_but_not_necessary_for_q_l743_743499

def p (x : ℝ) : Prop := 0 < x ∧ x < 1
def q (x : ℝ) : Prop := 2^x ≥ 1

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ ¬ (∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_but_not_necessary_for_q_l743_743499


namespace field_trip_missing_girls_l743_743353

theorem field_trip_missing_girls :
  let students_bus1 := 22
  let boys_bus1 := 12
  let students_bus2 := 18
  let boys_bus2 := 8
  let students_bus3 := 25
  let boys_bus3 := 15
  let girls_bus1_before := students_bus1 - boys_bus1
  let girls_bus2_before := students_bus2 - boys_bus2
  let girls_bus3_before := students_bus3 - boys_bus3
  let girls_bus1_trip := boys_bus1
  let girls_bus2_trip := boys_bus2
  let girls_bus3_trip := boys_bus3
  let girls_missing_bus1 := girls_bus1_trip - girls_bus1_before
  let girls_missing_bus2 := girls_bus2_before - girls_bus2_trip
  let girls_missing_bus3 := girls_bus3_trip - girls_bus3_before in
  girls_missing_bus1 + girls_missing_bus2 + girls_missing_bus3 = 9 := by
  sorry

end field_trip_missing_girls_l743_743353


namespace arithmetic_geometric_sequence_sum_l743_743968

theorem arithmetic_geometric_sequence_sum
    (a b : ℕ)
    (h1 : 2 < a)
    (h2 : a < b)
    (h3 : 2 + (a - 2) = a)
    (h4 : 2 + 2 * (a - 2) = b)
    (h5 : b = a * (b / a))
    (h6 : 18 = b * (18 / b)) :
    a + b = 16 := 
begin
  sorry
end

end arithmetic_geometric_sequence_sum_l743_743968


namespace paving_path_DE_time_l743_743237

-- Define the conditions
variable (v : ℝ) -- Speed of Worker 1
variable (x : ℝ) -- Total distance for Worker 1
variable (d2 : ℝ) -- Total distance for Worker 2
variable (AD DE EF FC : ℝ) -- Distances in the path of Worker 2

-- Define the statement
theorem paving_path_DE_time :
  (AD + DE + EF + FC) = d2 ∧
  x = 9 * v ∧
  d2 = 10.8 * v ∧
  d2 = AD + DE + EF + FC ∧
  (∀ t, t = (DE / (1.2 * v)) * 60) ∧
  t = 45 :=
by
  sorry

end paving_path_DE_time_l743_743237


namespace problem_a_problem_b_problem_c_l743_743257

-- Problem a
theorem problem_a (a : ℕ) (n : ℕ) (h : a ∈ {1, 2, 4}) (h_pos : 0 < n) : ¬ ∃ k : ℕ, k^2 = n * (a + n) :=
begin
  sorry
end

-- Problem b
theorem problem_b (k : ℕ) (h : 3 ≤ k) :
  ∃ n : ℕ, 0 < n ∧ ∃ m : ℕ, m^2 = n * (2^k + n) :=
begin
  sorry
end

-- Problem c
theorem problem_c (a : ℕ) (h : a ∉ {1, 2, 4}) :
  ∃ n : ℕ, 0 < n ∧ ∃ m : ℕ, m^2 = n * (a + n) :=
begin
  sorry
end

end problem_a_problem_b_problem_c_l743_743257


namespace problem_proof_l743_743382

def periodic_odd_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f(x + 6) = f(x)) ∧ (∀ x, f(-x) = -f(x))

theorem problem_proof (f : ℝ → ℝ) (h_periodic_odd : periodic_odd_function f) (h_f_neg1 : f(-1) = -1) : f(5) = -1 :=
  by
  sorry

end problem_proof_l743_743382


namespace vector_addition_perpendicular_l743_743972

theorem vector_addition_perpendicular (x : ℝ) (h : (2,1) • (x,-2) = 0) : (2 + x, 1 - 2) = (3, -1) :=
by
  have h1 : 2 * x - 2 = 0 := h
  have h2 : 2 * x = 2 := by sorry
  have h3 : x = 1 := by sorry
  rw [h3] 
  simp
  exact congr_arg2 Prod.mk rfl rfl

end vector_addition_perpendicular_l743_743972


namespace ones_digit_sum_2011_l743_743623

-- Define the pattern of ones digits for each n from 0 to 9
noncomputable def ones_digit_pattern : ℕ → ℕ := λ n,
  match n % 10 with
  | 0 => 0
  | 1 => 1
  | 2 => 8
  | 3 => 7
  | 4 => 4
  | 5 => 5
  | 6 => 6
  | 7 => 3
  | 8 => 2
  | 9 => 9
  | _ => 0 -- should never happen
  end

-- Define the problem statement
theorem ones_digit_sum_2011 :
  (Σi in Finset.range 2011, (i+1)^2011) % 10 = 6 :=
by
  sorry

end ones_digit_sum_2011_l743_743623


namespace john_buys_36_rolls_l743_743480

-- Definitions of the conditions
def cost_per_dozen := 5
def total_money_spent := 15
def rolls_per_dozen := 12

-- Theorem statement: John bought 36 rolls
theorem john_buys_36_rolls :
  let dozens_bought := total_money_spent / cost_per_dozen in
  let total_rolls := dozens_bought * rolls_per_dozen in
  total_rolls = 36 :=
by
  -- Proof steps would go here
  sorry

end john_buys_36_rolls_l743_743480


namespace election_total_votes_l743_743841

theorem election_total_votes (V_A V_B V : ℕ) (H1 : V_A = V_B + 15/100 * V) (H2 : V_A + V_B = 80/100 * V) (H3 : V_B = 2184) : V = 6720 :=
sorry

end election_total_votes_l743_743841


namespace right_triangle_sides_and_angle_l743_743323

theorem right_triangle_sides_and_angle
  (u : ℝ)
  (h1 : 3*u - 2 ≥ 0)
  (h2 : 3*u + 2 ≥ 0)
  (h3 : 6*u ≥ 0) :
  (sqrt (3*u - 2))^2 + (sqrt (3*u + 2))^2 = (sqrt (6*u))^2 ∧
  (angle of the largest angle formed by these sides is 90 degrees) :=
sorry

end right_triangle_sides_and_angle_l743_743323


namespace minimum_value_y1_plus_y2_l743_743806

noncomputable def parabola : set (ℝ × ℝ) := {p | p.1 ^ 2 = 2 * p.2}
noncomputable def line_through_P (k : ℝ) : set (ℝ × ℝ) := {p | p.2 = k * p.1 + 1}
noncomputable def intersection_points (k : ℝ) : set (ℝ × ℝ × ℝ × ℝ) :=
  {p | p.1 ^ 2 = 2 * p.2 ∧ p.3 ^ 2 = 2 * p.4 ∧
       (p.2 = k * p.1 + 1) ∧ (p.4 = k * p.3 + 1)}

theorem minimum_value_y1_plus_y2 : 
  ∀ (k : ℝ), (∃ (x1 x2 y1 y2 : ℝ), (x1, y1, x2, y2) ∈ intersection_points k) →
  y1 + y2 = 2 * k^2 + 2 := sorry

end minimum_value_y1_plus_y2_l743_743806


namespace one_of_first_three_will_respond_yes_l743_743696

theorem one_of_first_three_will_respond_yes :
  ∀ (hat_color : ℕ → ℕ) (can_see : ℕ → set ℕ),
    (∀ i, 1 ≤ hat_color i ∧ hat_color i ≤ 2) →  -- hat color is either 1 (white) or 2 (black)
    (∀ i, i ≥ 1 ∧ i ≤ 6) →  -- there are 6 people
    (∀ i, can_see i = {((i+2) % 6) + 1, ((i+3) % 6) + 1, ((i+4) % 6) + 1}) →  -- each person sees the opposite three
    (∑ i in finset.range 6, if hat_color i = 1 then 1 else 0 = 3) →  -- 3 white hats
    (∑ i in finset.range 6, if hat_color i = 2 then 1 else 0 = 3) →  -- 3 black hats
    -- Responses of the first two persons:
    -- If neither Person 1 nor Person 2 can deduce the color of the hats:
    (¬ ∃ c, ∀ p ∈ can_see 1, hat_color p = c) →  -- Person 1 cannot deduce
    (¬ ∃ c, ∀ p ∈ can_see 2, hat_color p = c) →  -- Person 2 cannot deduce
    ∃ (c : ℕ), (∀ p ∈ can_see 3, hat_color p = c) :=  -- Then Person 3 can deduce
sorry

end one_of_first_three_will_respond_yes_l743_743696


namespace exists_two_pairs_satisfy_2x3_eq_y4_l743_743733

theorem exists_two_pairs_satisfy_2x3_eq_y4 :
  ∃ (x₁ y₁ x₂ y₂ : ℕ), 2 * x₁^3 = y₁^4 ∧ 2 * x₂^3 = y₂^4 ∧ (x₁, y₁) ≠ (x₂, y₂) :=
by
  use 2, 2
  use 32, 16
  split
  . calc 2 * 2^3 = 2 * 8 : by rw [pow_succ]
            ... = 16      : by norm_num
  split
  . calc 2 * 32^3 = 2 * (2^5)^3 : rfl 
              ... = 2 * 2^15    : by rw [pow_mul]
              ... = 2^16        : by rw [mul_comm, pow_succ, mul_assoc, pow_one, two_mul]
  . exact ne_of_apply_ne Prod.fst $ by simp

end exists_two_pairs_satisfy_2x3_eq_y4_l743_743733


namespace sprinkler_system_days_l743_743279

theorem sprinkler_system_days 
  (morning_water : ℕ) (evening_water : ℕ) (total_water : ℕ) 
  (h_morning : morning_water = 4) 
  (h_evening : evening_water = 6) 
  (h_total : total_water = 50) :
  total_water / (morning_water + evening_water) = 5 := 
by 
  sorry

end sprinkler_system_days_l743_743279


namespace cards_thrown_away_l743_743871

theorem cards_thrown_away (h1 : 3 * (52 / 2) + 3 * 52 - 200 = 34) : 34 = 34 :=
by sorry

end cards_thrown_away_l743_743871


namespace f_minus3_gt_f2_gt_f_minus1_l743_743381

-- Define the even property of the function
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define the monotonicity on (0, +∞)
def is_monotonic_increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ a b : ℝ, 0 < b → b < a → f b < f a

-- Given:
variable {f : ℝ → ℝ}
hypothesis h_even : is_even f
hypothesis h_mono_inc : is_monotonic_increasing_on_pos f

theorem f_minus3_gt_f2_gt_f_minus1 : f (-3) > f 2 ∧ f 2 > f (-1) :=
by sorry

end f_minus3_gt_f2_gt_f_minus1_l743_743381


namespace perfect_square_condition_l743_743821

theorem perfect_square_condition (x m : ℝ) (h : ∃ k : ℝ, x^2 + x + 2*m = k^2) : m = 1/8 := 
sorry

end perfect_square_condition_l743_743821


namespace second_divisor_27_l743_743675

theorem second_divisor_27 (N : ℤ) (D : ℤ) (k : ℤ) (q : ℤ) (h1 : N = 242 * k + 100) (h2 : N = D * q + 19) : D = 27 := by
  sorry

end second_divisor_27_l743_743675


namespace smallest_period_of_f_center_of_symmetry_of_f_range_of_f_on_interval_l743_743392

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x)^2 + 4 * (Real.sin x)^2 - (1 + 2)

theorem smallest_period_of_f : ∀ x : ℝ, f (x + π) = f x := 
by sorry

theorem center_of_symmetry_of_f : ∀ k : ℤ, ∃ c : ℝ, ∀ x : ℝ, f (c - x) = f (c + x) := 
by sorry

theorem range_of_f_on_interval : 
  ∃ a b, (∀ x ∈ Set.Icc (-π / 4) (π / 4), f x ∈ Set.Icc a b) ∧ 
          (∀ y, y ∈ Set.Icc 3 5 → ∃ x ∈ Set.Icc (-π / 4) (π / 4), y = f x) := 
by sorry

end smallest_period_of_f_center_of_symmetry_of_f_range_of_f_on_interval_l743_743392


namespace derivative_of_my_function_l743_743639

variable (x : ℝ)

noncomputable def my_function : ℝ :=
  (Real.cos (Real.sin 3))^2 + (Real.sin (29 * x))^2 / (29 * Real.cos (58 * x))

theorem derivative_of_my_function :
  deriv my_function x = Real.tan (58 * x) / Real.cos (58 * x) := 
sorry

end derivative_of_my_function_l743_743639


namespace interval_increasing_k_alpha_beta_sum_range_a_l743_743398

noncomputable def f (x : ℝ) := Real.sin (4 * x + Real.pi / 3)

def is_increasing (a b : ℝ) := forall (x : ℝ), a <= x ∧ x <= b -> f' x > 0

theorem interval_increasing_k (k : ℤ) :
    is_increasing ((k : ℝ) * Real.pi / 2 - 5 * Real.pi / 24)
                  ((k : ℝ) * Real.pi / 2 + Real.pi / 24) := sorry

noncomputable def g (x : ℝ) := Real.sin (2 * x - Real.pi / 6)

theorem alpha_beta_sum (k α β : ℝ) (hα : 0 <= α ∧ α <= Real.pi / 2)
                        (hβ : 0 <= β ∧ β <= Real.pi / 2) :
    g α + k = 0 ∧ g β + k = 0 -> α + β = 5 * Real.pi / 12 := sorry

noncomputable def h (x : ℝ) (a : ℝ) := 2^x - a

theorem range_a (a : ℝ) :
    (∀ x1 : ℝ, 0 <= x1 ∧ x1 <= 1 -> ∃ x2 : ℝ, 0 <= x2 ∧ x2 <= 5 * Real.pi / 24 ∧ h x1 a = f x2) ->
    1 <= a ∧ a <= 3 / 2 := sorry

end interval_increasing_k_alpha_beta_sum_range_a_l743_743398


namespace rationalize_denominator_l743_743539

theorem rationalize_denominator :
  ∃ A B C : ℤ, C > 0 ∧ ¬(∃ p : ℕ, p.prime ∧ p^3 ∣ B) ∧
  (A + B + C = 74) ∧
  (C ≠ 0) ∧
  ((4 : ℚ) / (3 * real.cbrt (7 : ℚ)) = (A * real.cbrt (B : ℚ)) / (C : ℚ)) :=
begin
  -- Existence of A, B, and C such that the conditions hold.
  use [4, 49, 21],
  split,
  -- C > 0
  norm_num,
  split,
  -- B is not divisible by the cube of any prime
  intro h,
  linarith,
  split,
  -- A + B + C = 74
  norm_num,
  split,
  -- C ≠ 0
  norm_num,
  -- Rationalized form matches.
  exact sorry,
end

end rationalize_denominator_l743_743539


namespace find_f_neg_a_l743_743893

def f (x : Real) : Real := x^2 * Real.log (-x + Real.sqrt (x^2 + 1)) + 1

theorem find_f_neg_a (a : Real) (h : f a = 11) : f (-a) = -9 :=
  sorry

end find_f_neg_a_l743_743893


namespace abs_diff_xy_l743_743928

-- Given conditions
variable {x y a b c : ℕ}
variable hx_pos : 0 < x
variable hy_pos : 0 < y
variable hxy_distinct : x ≠ y
variable ha_nonzero : a ≠ 0
variable h_am : (x + y) / 2 = 100 * a + 10 * b + c
variable h_gm : Nat.sqrt (x * y) = 10 * b + a

-- Proof statement
theorem abs_diff_xy : |x - y| = 99 :=
  sorry

end abs_diff_xy_l743_743928


namespace triangle_ABC_isosceles_l743_743103

-- Lean 4 definitions for the generated equivalent proof problem
noncomputable def incenter (A B C I : Type*) := sorry
noncomputable def incircle (ABC : Type*) (α : Type*) := sorry
noncomputable def circumcircle (A I C : Type*) := sorry
noncomputable def intersects (circle1 circle2 : Type*) (P Q : Type*) := sorry
noncomputable def same_side_line (P A : Type*) (line : Type*) := sorry
noncomputable def other_side_line (Q C : Type*) (line : Type*) := sorry
noncomputable def midpoint_arc (arc : Type*) := sorry
noncomputable def parallel (line1 line2 : Type*) := sorry
noncomputable def triangle_isosceles (A B C : Type*) := ∃ (ABC_is_isosceles : Prop), ABC_is_isosceles

-- Given conditions for the incenter, incircle, and parallel lines, we must prove the triangle is isosceles
theorem triangle_ABC_isosceles (A B C I α P Q M N : Type*)
  (h1 : incenter A B C I)
  (h2 : incircle ABC α)
  (h3 : circumcircle A I C)
  (h4 : intersects α (circumcircle A I C) P Q)
  (h5 : same_side_line P A (set I B))
  (h6 : other_side_line Q C (set I B))
  (h7 : midpoint_arc α AC M)
  (h8 : midpoint_arc α BC N)
  (h9 : parallel PQ AC) :
  triangle_isosceles A B C := sorry

end triangle_ABC_isosceles_l743_743103


namespace number_of_valid_ns_l743_743753

theorem number_of_valid_ns :
  ∃ (S : Finset ℕ), S.card = 13 ∧ ∀ n ∈ S, n ≤ 1000 ∧ Nat.floor (995 / n) + Nat.floor (996 / n) + Nat.floor (997 / n) % 4 ≠ 0 :=
by
  sorry

end number_of_valid_ns_l743_743753


namespace tax_diminished_percentage_l743_743604

theorem tax_diminished_percentage (T C : ℝ) (x : ℝ) (h : (T * (1 - x / 100)) * (C * 1.10) = T * C * 0.88) :
  x = 20 :=
sorry

end tax_diminished_percentage_l743_743604


namespace negation_of_forall_prop_l743_743585

theorem negation_of_forall_prop :
  ¬ (∀ x : ℝ, x^2 + x > 0) ↔ ∃ x : ℝ, x^2 + x ≤ 0 :=
by
  sorry

end negation_of_forall_prop_l743_743585


namespace find_deriv_one_l743_743358

def f (x : ℝ) : ℝ := x^2 + 3 * x * (deriv f 1)

theorem find_deriv_one (h : f' 1 = -1) : deriv f 1 = -1 :=
by
  sorry

end find_deriv_one_l743_743358


namespace part_Ⅰ_part_Ⅱ_l743_743803

section

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := 4 * x - 1

def range_f (I : Set ℝ) : Set ℝ := { y | ∃ x ∈ I, f x = y }
def range_g (I : Set ℝ) : Set ℝ := { y | ∃ x ∈ I, g x = y }

theorem part_Ⅰ : range_f (set.Icc 1 2) ∩ range_g (set.Icc 1 2) = set.Icc 3 6 := 
sorry

theorem part_Ⅱ (m : ℝ) : 1 < m → range_f (set.Icc 1 m) = range_g (set.Icc 1 m) → m = 3 := 
sorry

end

end part_Ⅰ_part_Ⅱ_l743_743803


namespace lucky_tickets_equal_digit_sum_27_l743_743232

theorem lucky_tickets_equal_digit_sum_27 : 
  (∀ n : ℕ, (0 ≤ n) ∧ (n < 1000000) → 
      (let (a, b, c, d, e, f) := div_mod n in 
       a + b + c = d + e + f) ↔ (let (a, b, c, d, e, f) := div_mod m in 
       a + b + c + d + e + f = 27)) :=
by sorry

end lucky_tickets_equal_digit_sum_27_l743_743232


namespace find_c_for_local_minimum_at_2_l743_743748

def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

theorem find_c_for_local_minimum_at_2 (c : ℝ) (h1 : ∀ x, f'(x) = 3 * x ^ 2 - 4 * c * x + c ^ 2)
  (h2 : f' 2 = 0) : c = 2 :=
sorry

#check find_c_for_local_minimum_at_2

end find_c_for_local_minimum_at_2_l743_743748


namespace second_child_sweets_l743_743673

theorem second_child_sweets (total_sweets : ℕ) (mother_kept_fraction : ℚ)
  (eldest_sweets : ℕ) (youngest_to_eldest_ratio : ℚ) : 
  total_sweets = 27 ∧ mother_kept_fraction = 1/3 ∧ eldest_sweets = 8 ∧ youngest_to_eldest_ratio = 1/2 →
  let mother_kept := (mother_kept_fraction * total_sweets).natAbs in
  let children_sweets := total_sweets - mother_kept in
  let youngest_sweets := (eldest_sweets * youngest_to_eldest_ratio).natAbs in
  let second_child_sweets := children_sweets - eldest_sweets - youngest_sweets in
  second_child_sweets = 6 :=
by
  intro h
  rcases h with ⟨h₁, h₂, h₃, h₄⟩
  have h_mother_kept : (mother_kept_fraction * total_sweets).natAbs = 9, from sorry
  have h_children_sweets : total_sweets - (mother_kept_fraction * total_sweets).natAbs = 18, from sorry
  have h_youngest_sweets : (eldest_sweets * youngest_to_eldest_ratio).natAbs = 4, from sorry
  have h_second_child_sweets : 18 - eldest_sweets - (eldest_sweets * youngest_to_eldest_ratio).natAbs = 6, from sorry
  exact h_second_child_sweets


end second_child_sweets_l743_743673


namespace inequality_relation_l743_743776

theorem inequality_relation (x y : ℝ)
  (h : 3^x - 3^(-y) ≥ 5^(-x) - 5^y) :
  x + y ≥ 0 :=
sorry

end inequality_relation_l743_743776


namespace Leila_ate_9_cakes_on_Friday_l743_743874

/-- Leila ate 6 cakes on Monday -/
def Monday : ℕ := 6

/-- Leila ate triple the number of cakes she ate on Monday on Saturday -/
def Saturday : ℕ := 3 * Monday

/-- Leila ate 33 cakes in total -/
def Total : ℕ := 33

/-- The number of cakes Leila ate on Friday -/
def Friday := Total - Monday - Saturday

theorem Leila_ate_9_cakes_on_Friday : Friday = 9 :=
by
  simp [Friday, Total, Monday, Saturday]
  sorry

end Leila_ate_9_cakes_on_Friday_l743_743874


namespace single_digit_correct_l743_743247

theorem single_digit_correct : ∃ n : ℕ, n * n * n = 176 ∧ n < 10 :=
by {
  use 4,
  split,
  { norm_num },
  { norm_num }
}

end single_digit_correct_l743_743247


namespace triangle_ABC_isosceles_l743_743099

-- Lean 4 definitions for the generated equivalent proof problem
noncomputable def incenter (A B C I : Type*) := sorry
noncomputable def incircle (ABC : Type*) (α : Type*) := sorry
noncomputable def circumcircle (A I C : Type*) := sorry
noncomputable def intersects (circle1 circle2 : Type*) (P Q : Type*) := sorry
noncomputable def same_side_line (P A : Type*) (line : Type*) := sorry
noncomputable def other_side_line (Q C : Type*) (line : Type*) := sorry
noncomputable def midpoint_arc (arc : Type*) := sorry
noncomputable def parallel (line1 line2 : Type*) := sorry
noncomputable def triangle_isosceles (A B C : Type*) := ∃ (ABC_is_isosceles : Prop), ABC_is_isosceles

-- Given conditions for the incenter, incircle, and parallel lines, we must prove the triangle is isosceles
theorem triangle_ABC_isosceles (A B C I α P Q M N : Type*)
  (h1 : incenter A B C I)
  (h2 : incircle ABC α)
  (h3 : circumcircle A I C)
  (h4 : intersects α (circumcircle A I C) P Q)
  (h5 : same_side_line P A (set I B))
  (h6 : other_side_line Q C (set I B))
  (h7 : midpoint_arc α AC M)
  (h8 : midpoint_arc α BC N)
  (h9 : parallel PQ AC) :
  triangle_isosceles A B C := sorry

end triangle_ABC_isosceles_l743_743099


namespace ABC_is_isosceles_l743_743126

open Mobius

variables
  (ABC : Triangle)
  (I : Point)
  (α P Q M N : Point)
  (A B C Z : Point)
  [Incenter I ABC]
  [Incircle α ABC]
  [Circumcircle (Triangle.mk A I C)]
  [PQ_parallel_AC : is_parallel (Segment.mk P Q) (Segment.mk A C)]
  [Midpoint M (Arc.mk A C α)]
  [Midpoint N (Arc.mk B C α)]
  [center_Z : Center Z (Circumcircle (Triangle.mk A I C))]
  [Chord_α_PQ : common_chord α (Circumcircle (Triangle.mk A I C)) (Segment.mk P Q)]
  [P_A_same_BI : same_side P A (Line.mk B I)]
  [Q_C_other_BI : same_side Q C (Line.mk B I)]

theorem ABC_is_isosceles
  (h_parallel : PQ_parallel_AC) : 
  Isosceles ABC := 
sorry

end ABC_is_isosceles_l743_743126


namespace odd_palindrome_count_l743_743166

theorem odd_palindrome_count :
  (∃ A B : ℕ, 1 ≤ A ∧ A ≤ 9 ∧ A % 2 = 1 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 9 * B + A = 50) :=
begin
  sorry
end

end odd_palindrome_count_l743_743166


namespace distance_between_foci_l743_743736

-- Given conditions
def hyperbola_eq (x y : ℝ) : Prop := (x ^ 2) / 25 - (y ^ 2) / 4 = 1
def a_squared : ℝ := 25
def b_squared : ℝ := 4
def a : ℝ := Real.sqrt a_squared
def b : ℝ := Real.sqrt b_squared
def c_squared : ℝ := a_squared + b_squared
def c : ℝ := Real.sqrt c_squared

-- Prove the distance between the foci is 2 * Real.sqrt 29
theorem distance_between_foci : 2 * c = 2 * Real.sqrt 29 := by
  sorry

end distance_between_foci_l743_743736


namespace geometrical_shape_representation_l743_743706

theorem geometrical_shape_representation (x y : ℝ) :
  abs x + abs y ≤ 2 * real.sqrt (x^2 + y^2) ∧ 2 * real.sqrt (x^2 + y^2) ≤ 3 * max (abs x) (abs y) →
  ∃ (figures : set (set (ℝ × ℝ))),
    figures = {circle, diamond, triangle} ∧
    ∀ (p : ℝ × ℝ), p ∈ circle → p ∈ diamond ∧ p ∈ triangle :=
sorry

end geometrical_shape_representation_l743_743706


namespace min_n_binomial_constant_term_l743_743740

open Nat

theorem min_n_binomial_constant_term :
  ∃ (n : ℕ), n > 0 ∧ (∀ (x : ℝ), ∃ (r : ℕ), ∑ k in range (n + 1), (choose n k) * (-1)^k * x^(n - 8 * k) = 1) :=
sorry

end min_n_binomial_constant_term_l743_743740


namespace sequence_periodic_l743_743811

def sequence (a : ℕ → ℚ) := a 1 = 2 ∧ ∀ n, a (n + 1) = -1 / (a n + 1)

theorem sequence_periodic :
  ∀ a : ℕ → ℚ, sequence a → a 2016 = -3 / 2 :=
by
  intros a h
  obtain ⟨h1, h_rec⟩ := h
  sorry

end sequence_periodic_l743_743811


namespace highest_elevation_l743_743288

noncomputable theory

-- Define the given conditions
def elevation (t : ℝ) : ℝ := 160 * t - 16 * t^2

-- Statement to prove
theorem highest_elevation : ∃ t : ℝ, elevation t = 400 ∧ ∀ t' : ℝ, elevation t' ≤ elevation t := by
sory

end highest_elevation_l743_743288


namespace equal_angles_l743_743150

section Geometry

variables (A B C D M N P : Type) 
variables [rect : is_rectangle A B C D] 
          [midpointM : is_midpoint M B C] 
          [midpointN : is_midpoint N C D] 
          [intersectionP : is_intersection P (line_through B N) (line_through D M)]

theorem equal_angles {A B C D M N P : Type} 
  [is_rectangle A B C D] 
  [is_midpoint M B C] 
  [is_midpoint N C D] 
  [is_intersection P (line_through B N) (line_through D M)] : 
  angle A M N = angle D P N := sorry

end Geometry

end equal_angles_l743_743150


namespace isosceles_triangle_l743_743114

theorem isosceles_triangle (ABC : Type) [triangle ABC]
  (I : incenter ABC) (α : incircle ABC)
  (P Q : α) (circAIC : circumcircle AIC)
  (h1 : P ∈ circAIC) (h2 : Q ∈ circAIC)
  (h3 : same_side P A (BI line))
  (h4 : other_side Q C (BI line))
  (M : midpoint_arc AC (α arc))
  (N : midpoint_arc BC (α arc))
  (h_par : PQ ∥ AC) : is_isosceles ABC := 
sorry

end isosceles_triangle_l743_743114


namespace sprinkler_days_needed_l743_743277

-- Definitions based on the conditions
def morning_water : ℕ := 4
def evening_water : ℕ := 6
def daily_water : ℕ := morning_water + evening_water
def total_water_needed : ℕ := 50

-- The proof statement
theorem sprinkler_days_needed : total_water_needed / daily_water = 5 := by
  sorry

end sprinkler_days_needed_l743_743277


namespace driver_license_advantage_l743_743522

def AdvantageousReasonsForEarlyLicenseObtaining 
  (eligible : ℕ → Prop)
  (effectiveInsurance : ℕ → Prop)
  (rentalCarFlexibility : ℕ → Prop)
  (employmentOpportunity : ℕ → Prop) : Prop :=
  ∀ age1 age2 : ℕ, (eligible age1 ∧ eligible age2 ∧ age1 < age2) →
  (effectiveInsurance age1 ∧ rentalCarFlexibility age1 ∧ employmentOpportunity age1) →
  effectiveInsurance age1 ∧ rentalCarFlexibility age1 ∧ employmentOpportunity age1

theorem driver_license_advantage 
  (eligible : ℕ → Prop)
  (effectiveInsurance : ℕ → Prop)
  (rentalCarFlexibility : ℕ → Prop)
  (employmentOpportunity : ℕ → Prop) :
  AdvantageousReasonsForEarlyLicenseObtaining eligible effectiveInsurance rentalCarFlexibility employmentOpportunity :=
by
  sorry

end driver_license_advantage_l743_743522


namespace rank_sum_iff_exists_invertible_X_l743_743486

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]
variable {R : Type*} [Field R] 
variable {A B : Matrix n n R}

theorem rank_sum_iff_exists_invertible_X : 
  rank A + rank B ≤ Fintype.card n ↔ ∃ X : Matrix n n R, Invertible X ∧ A ⬝ X ⬝ B = 0 := sorry

end rank_sum_iff_exists_invertible_X_l743_743486


namespace andy_and_carlos_tie_for_first_l743_743693

variables (c r : ℝ) (hc : c > 0) (hr : r > 0)

def area_of_carlos_lawn := c
def area_of_beth_lawn := 3 * c
def area_of_andy_lawn := 4 * c

def rate_of_carlos := r
def rate_of_beth := 2 * r
def rate_of_andy := 4 * r

def time_to_mow (area rate : ℝ) := area / rate

theorem andy_and_carlos_tie_for_first :
  time_to_mow (area_of_andy_lawn c) (rate_of_andy r) =
  time_to_mow (area_of_carlos_lawn c) (rate_of_carlos r) ∧
  time_to_mow (area_of_beth_lawn c) (rate_of_beth r) >
  time_to_mow (area_of_carlos_lawn c) (rate_of_carlos r) :=
by sorry

end andy_and_carlos_tie_for_first_l743_743693


namespace min_sum_eq_6044_l743_743506

theorem min_sum_eq_6044 :
  ∃ (a : ℕ), 
    ∃ (b : ℕ), 
    ∃ (c : ℕ), 
    ∃ (d : ℕ), 
    ∃ (e : ℕ), 
    ∃ (f : ℕ), 
    ∃ (g : ℕ), 
    ∃ (h : ℕ), 
    ∃ (i : ℕ), 
    ∃ (j : ℕ), 
      list.pairwise (≠) [a, b, c, d, e, f, g, h, i, j] ∧
      a + b + c + d + e + f + g + h + i + j = 1995 ∧
      a * b + b * c + c * d + d * e + e * f + f * g + g * h + h * i + i * j + j * a = 6044 :=
begin
  sorry
end

end min_sum_eq_6044_l743_743506


namespace largest_partner_profit_l743_743352

def partner_profits (ratio : List ℕ) (total_profit : ℕ) :=
  let total_ratio := ratio.sum
  let part_value := total_profit / total_ratio
  ratio.map (λ r, r * part_value)

theorem largest_partner_profit {profit : ℕ} {ratios : List ℕ} (h_ratios : ratios = [2, 4, 3, 4, 5])
  (h_profit : profit = 36000) :
  let profits := partner_profits ratios profit
  in profits.maximum = some 10000 :=
by
  sorry

end largest_partner_profit_l743_743352


namespace constant_term_expansion_l743_743200

theorem constant_term_expansion : 
  (∃ c : ℚ, constant_term ((1 - x)^3 * (1 - 1 / x)^3) = c ∧ c = 20) := sorry

noncomputable def constant_term (expr : polynomial ℚ) : ℚ := 
  sorry

end constant_term_expansion_l743_743200


namespace c_share_of_profit_l743_743688

theorem c_share_of_profit
  (x : ℝ)
  (profit : ℝ := 12375)
  (investment_A : ℝ := 3 * x)
  (investment_B : ℝ := x)
  (investment_C : ℝ := (9 / 2) * x)
  (ratio_sum : ℝ := 6 + 2 + 9)
  (c_ratio : ℝ := 9) :
  (c_ratio / ratio_sum) * profit = 6551.46 := 
by
  have ratio_sum_pos : ratio_sum ≠ 0 := by linarith
  have profit_pos : profit > 0 := by linarith
  have valid_ratio : c_ratio / ratio_sum ≠ 0 := by linarith [ratio_sum_pos]
  sorry

end c_share_of_profit_l743_743688


namespace isosceles_triangle_l743_743093

open_locale euclidean_geometry

variables {A B C I P Q M N : Point}
variables (α : circle)
variables (circumcircle_AIC : circle)

-- Condition 1
hypothesis h1 : incenter I (triangle.mk A B C)

-- Condition 2
hypothesis h2 : α = incircle (triangle.mk A B C)

-- Condition 3
hypothesis h3 : intersects circumcircle_AIC α P
hypothesis h4 : intersects circumcircle_AIC α Q

-- Condition 4
hypothesis h5 : same_side P A (line.mk B I)

-- Condition 5
hypothesis h6 : ¬ same_side Q C (line.mk B I)

-- Condition 6
hypothesis h7 : midpoint M (arc.mk α A C)

-- Condition 7
hypothesis h8 : midpoint N (arc.mk α B C)

-- Condition 8
hypothesis h9 : parallel (line.mk P Q) (line.mk A C)

-- Conclusion
theorem isosceles_triangle (h1 h2 h3 h4 h5 h6 h7 h8 h9) : (distance A B) = (distance A C) :=
sorry

end isosceles_triangle_l743_743093


namespace isosceles_triangle_l743_743113

theorem isosceles_triangle (ABC : Type) [triangle ABC]
  (I : incenter ABC) (α : incircle ABC)
  (P Q : α) (circAIC : circumcircle AIC)
  (h1 : P ∈ circAIC) (h2 : Q ∈ circAIC)
  (h3 : same_side P A (BI line))
  (h4 : other_side Q C (BI line))
  (M : midpoint_arc AC (α arc))
  (N : midpoint_arc BC (α arc))
  (h_par : PQ ∥ AC) : is_isosceles ABC := 
sorry

end isosceles_triangle_l743_743113


namespace reciprocal_of_repeating_decimal_equiv_l743_743979

noncomputable def repeating_decimal (x : ℝ) := 0.333333...

theorem reciprocal_of_repeating_decimal_equiv :
  (1 / repeating_decimal 0.333333...) = 3 :=
sorry

end reciprocal_of_repeating_decimal_equiv_l743_743979


namespace highest_value_of_a_l743_743336

theorem highest_value_of_a (a : ℕ) (h : 0 ≤ a ∧ a ≤ 9) : (365 * 10 ^ 3 + a * 10 ^ 2 + 16) % 8 = 0 → a = 8 := by
  sorry

end highest_value_of_a_l743_743336


namespace typist_salary_proof_l743_743221

noncomputable def original_salary (x : ℝ) : Prop :=
  1.10 * x * 0.95 = 1045

theorem typist_salary_proof (x : ℝ) (H : original_salary x) : x = 1000 :=
sorry

end typist_salary_proof_l743_743221


namespace ellipse_and_line_properties_l743_743768

theorem ellipse_and_line_properties :
  (∃ a b : ℝ, a > b ∧ b > 0 ∧ a * a = 4 ∧ b * b = 3 ∧
  ∀ x y : ℝ, (x, y) = (1, 3/2) → x^2 / a^2 + y^2 / b^2 = 1) ∧
  (∃ k : ℝ, k = 1 / 2 ∧ ∀ x y : ℝ, (x, y) = (2, 1) →
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧
  (x1 - 2) * (x2 - 2) + (k * (x1 - 2) + 1 - 1) * (k * (x2 - 2) + 1 - 1) = 5 / 4) :=
sorry

end ellipse_and_line_properties_l743_743768


namespace region_area_l743_743340

theorem region_area (r θ : ℝ) (x y : ℝ → ℝ) :
  (∀ θ, r = 2 * (sec θ) → x θ = 2) →
  (∀ θ, r = 2 * (csc θ) → y θ = 2) →
  (∀ x, 0 ≤ x → x ≤ 2) →
  (∀ y, 0 ≤ y → y ≤ 2) →
  (x 0 = 0) ∧ (y 0 = 0) →
  (x 2 = 2) ∧ (y 2 = 2) →
  let area := (2 - 0) * (2 - 0) in
  area = 4 :=
by
  sorry

end region_area_l743_743340


namespace radius_circle_spherical_l743_743596

def spherical_to_cartesian (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Math.sin φ * Math.cos θ, ρ * Math.sin φ * Math.sin θ, ρ * Math.cos φ)

theorem radius_circle_spherical (θ : ℝ) : 
  (let (ρ, φ) := (1, π / 4)
   let (x, y, z) := spherical_to_cartesian ρ θ φ
   Math.sqrt (x^2 + y^2)) = (Real.sqrt 2) / 2 :=
by 
  let ρ := 1
  let φ := π / 4
  let (x, y, z) := spherical_to_cartesian ρ θ φ
  sorry

end radius_circle_spherical_l743_743596


namespace ABC_is_isosceles_l743_743122

open Mobius

variables
  (ABC : Triangle)
  (I : Point)
  (α P Q M N : Point)
  (A B C Z : Point)
  [Incenter I ABC]
  [Incircle α ABC]
  [Circumcircle (Triangle.mk A I C)]
  [PQ_parallel_AC : is_parallel (Segment.mk P Q) (Segment.mk A C)]
  [Midpoint M (Arc.mk A C α)]
  [Midpoint N (Arc.mk B C α)]
  [center_Z : Center Z (Circumcircle (Triangle.mk A I C))]
  [Chord_α_PQ : common_chord α (Circumcircle (Triangle.mk A I C)) (Segment.mk P Q)]
  [P_A_same_BI : same_side P A (Line.mk B I)]
  [Q_C_other_BI : same_side Q C (Line.mk B I)]

theorem ABC_is_isosceles
  (h_parallel : PQ_parallel_AC) : 
  Isosceles ABC := 
sorry

end ABC_is_isosceles_l743_743122


namespace translate_sine_function_left_shift_l743_743233

theorem translate_sine_function_left_shift (x : ℝ) :
  (3 * sin (2 * (x - π / 6) + π / 6)) = 3 * sin (2 * x - π / 6) :=
by
  sorry

end translate_sine_function_left_shift_l743_743233


namespace trajectory_equation_and_shape_existence_of_circle_l743_743360

-- Given conditions
variables (m x y : ℝ)
def a := (m * x, y + 1)
def b := (x, y - 1)

-- The statements for both questions in Lean
theorem trajectory_equation_and_shape : 
  (a ⬝ b = 0) ↔ (m * x^2 + y^2 = 1) := 
sorry

theorem existence_of_circle {m : ℝ}:
  (m = 1 / 4) →
  ∃ r, (r = sqrt (4 / 5))
      ∧ ∀ k t : ℝ, (tangent_line_intersects_ellipse (x, y) k t ⬝ (ellipse_trajectory (x, y) k t 1 m) = 0) → 
       (x^2 + y^2 = r^2) :=
sorry

noncomputable def tangent_line_intersects_ellipse (p : ℝ × ℝ) (k t : ℝ) :=
  let (x, y) := p in y = k * x + t

noncomputable def ellipse_trajectory (p : ℝ × ℝ) (k t r : ℝ) (m : ℝ) :=
  let (x, y) := p in x^2 / (4 / r^2) + y^2 = 1

end trajectory_equation_and_shape_existence_of_circle_l743_743360


namespace arithmetic_sequence_problem_l743_743461

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 4 + a 6 + a 8 + a 10 + a 12 = 60)
  (h2 : ∀ n, a (n + 1) = a n + d) :
  a 7 - (1 / 3) * a 5 = 8 :=
by
  sorry

end arithmetic_sequence_problem_l743_743461


namespace perimeter_is_correct_l743_743961

noncomputable def cos_70 := real.cos (70 * real.pi / 180)

def side_length := 8
def num_sides := 24
def angle_mid := 20
def angle_top := 50

def overlapping_segment_length := side_length / cos_70

def perimeter := num_sides * overlapping_segment_length

theorem perimeter_is_correct :
  perimeter ≈ 561.408 :=
begin
  sorry
end

end perimeter_is_correct_l743_743961


namespace distance_XY_l743_743618

noncomputable def distanceXY : ℝ :=
  let small_circle_center := (0 : ℝ, 0 : ℝ)
  let small_circle_radius := (1 : ℝ)
  let large_circle_center := (2 : ℝ, 0 : ℝ)
  let large_circle_radius := real.sqrt 2

/- Definitions for points X, M, Y and conditions -/
def point_X : ℝ × ℝ := (3/4, real.sqrt 7 / 4)
def point_M (θ : ℝ) : ℝ × ℝ := (real.cos θ, real.sin θ)
def point_Y (θ : ℝ) : ℝ × ℝ := (2 * real.cos θ - 3/4, 2 * real.sin θ - real.sqrt 7 / 4)

/- Condition that M is the midpoint of XY -/
def is_midpoint (M X Y : ℝ × ℝ) : Prop :=
  2 * M.1 = X.1 + Y.1 ∧ 2 * M.2 = X.2 + Y.2

/- Condition that point lies on a circle -/
def on_circle (c : ℝ × ℝ) (r : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2

/- The main theorem to prove the distance XY -/
theorem distance_XY
  (θ : ℝ)
  (hM_on_small : on_circle small_circle_center small_circle_radius (point_M θ))
  (hXY_midpoint : is_midpoint (point_M θ) point_X (point_Y θ))
  (hY_on_large : on_circle large_circle_center large_circle_radius (point_Y θ)) :
  real.sqrt ((point_X.1 - point_Y θ.1)^2 + (point_X.2 - point_Y θ.2)^2) = real.sqrt (7 / 2) :=
begin
  sorry
end

end distance_XY_l743_743618


namespace sock_pair_count_l743_743424

theorem sock_pair_count :
  let socks := ["white", "brown", "blue"]
  let count := 5
  ∀ color ∈ socks, (nat.choose count 2) = 10 →
  sum (λ color, (nat.choose count 2)) socks = 30 :=
by
  intros
  simp only [list.sum, nat.choose]
  sorry

end sock_pair_count_l743_743424


namespace inequality_am_gm_l743_743824

theorem inequality_am_gm (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end inequality_am_gm_l743_743824


namespace proof_expression1_proof_expression2_l743_743311

noncomputable def expression1 : ℝ :=
  Real.sqrt ((1 : ℝ) * (2 + 1/4)) + (1/10 : ℝ) ^ (-2) - Real.pi ^ 0 + (-27/8 : ℝ) ^ (1/3)

theorem proof_expression1 : expression1 = 99 := by
  sorry

noncomputable def expression2 : ℝ :=
  (1/2 : ℝ) * Real.log10 25 + Real.log10 2 - Real.log2 9 * Real.log3 2

theorem proof_expression2 : expression2 = -1 := by
  sorry

end proof_expression1_proof_expression2_l743_743311


namespace ellipse_properties_l743_743769

open Real

noncomputable def ellipse_equation (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

theorem ellipse_properties 
  (center : ℝ × ℝ)
  (focus_x : ℝ) 
  (eccentricity : ℝ) 
  (line_l : ℕ → ℝ × ℝ)
  (moving_point : ℝ) (A B : ℝ × ℝ) (NA NB : ℝ × ℝ)
  (t : ℝ) :
  center = (0, 0) →
  focus_x = 2 →
  eccentricity = 1/2 →
  ellipse_equation (2, 0) = true →
  (∀ (x y : ℝ), ellipse_equation x y → (3 * real.sqrt (x / 4))^2 + ((2 * (x - t)^2) + 3) = 0 ∧ (NA.1 - NB.1)^2 + (NA.2 - NB.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2) →
  (0 < t ∧ t < 1/4) :=
begin
  sorry
end

end ellipse_properties_l743_743769


namespace swamp_flies_eaten_l743_743330

def flies_eaten_per_day (frogs herons caimans fish gharials : ℕ) : ℕ :=
let frog_flies := frogs * 30 in
let heron_flies := herons * 60 in
frog_flies + heron_flies

theorem swamp_flies_eaten :
  flies_eaten_per_day 50 12 7 20 9 = 2220 :=
by {
  -- Definitions based on conditions
  -- Number of frogs, herons, caimans, fish, and gharials.
  let frogs := 50,
  let herons := 12,
  let caimans := 7,
  let fish := 20,
  let gharials := 9,
  
  -- Calculating flies eaten by frogs.
  let frog_flies := frogs * 30,
  -- Calculating flies caught by herons.
  let heron_flies := herons * 60,
  -- Total flies eaten per day.
  let total_flies := frog_flies + heron_flies,
  
  -- Required proof
  show total_flies = 2220, sorry
}

end swamp_flies_eaten_l743_743330


namespace jason_tattoos_count_l743_743689

-- Definitions based on given conditions
def tattoos_on_arms (jason_tattoos_arms_per_arm : Nat) (arms : Nat) := jason_tattoos_arms_per_arm * arms
def tattoos_on_legs (jason_tattoos_legs_per_leg : Nat) (legs : Nat) := jason_tattoos_legs_per_leg * legs
def total_tattoos (tattoos_arms : Nat) (tattoos_legs : Nat) := tattoos_arms + tattoos_legs

-- Given values
def jason_tattoos_arms_per_arm : Nat := 2
def jason_arms : Nat := 2
def jason_tattoos_legs_per_leg : Nat := 3
def jason_legs : Nat := 2
def adam_tattoos : Nat := 23

theorem jason_tattoos_count : 
  let tattoos_arms := tattoos_on_arms jason_tattoos_arms_per_arm jason_arms
  let tattoos_legs := tattoos_on_legs jason_tattoos_legs_per_leg jason_legs
  let total := total_tattoos tattoos_arms tattoos_legs
  adam_tattoos > 2 * total → total = 10 :=
by
  have tattoos_arms_eq : tattoos_arms = tattoos_on_arms jason_tattoos_arms_per_arm jason_arms := rfl
  have tattoos_legs_eq : tattoos_legs = tattoos_on_legs jason_tattoos_legs_per_leg jason_legs := rfl
  sorry

end jason_tattoos_count_l743_743689


namespace parallelogram_height_base_difference_l743_743825

theorem parallelogram_height_base_difference (A B H : ℝ) (hA : A = 24) (hB : B = 4) (hArea : A = B * H) :
  H - B = 2 := by
  sorry

end parallelogram_height_base_difference_l743_743825


namespace find_n_l743_743043

variable {a : ℕ → ℝ} (h1 : a 4 = 7) (h2 : a 3 + a 6 = 16)

theorem find_n (n : ℕ) (h3 : a n = 31) : n = 16 := by
  sorry

end find_n_l743_743043


namespace find_positive_x_l743_743402

theorem find_positive_x (x y z : ℝ) 
  (h1 : x * y = 15 - 3 * x - 2 * y)
  (h2 : y * z = 8 - 2 * y - 4 * z)
  (h3 : x * z = 56 - 5 * x - 6 * z) : x = 8 := 
sorry

end find_positive_x_l743_743402


namespace det_b_abs_eq_66_l743_743562

theorem det_b_abs_eq_66 (a b c : ℤ) (h_gcd : Int.gcd (Int.gcd a b) c = 1)
  (h_eq : a * (3 + (complex.I)) ^ 5 +
          b * (3 + (complex.I)) ^ 4 +
          c * (3 + (complex.I)) ^ 3 +
          c * (3 + (complex.I)) ^ 2 +
          b * (3 + (complex.I)) +
          a = 0) : |b| = 66 := 
sorry

end det_b_abs_eq_66_l743_743562


namespace triangle_is_isosceles_l743_743171

open EuclideanGeometry

variables (A B C M N : Point)

noncomputable def triangle_isosceles_condition : Prop :=
  let AB := dist A B
  let BC := dist B C
  let AC := dist A C
  let AM := dist A M
  let MB := dist M B
  let BN := dist B N
  let NC := dist N C
  in
  let perimeter_AMC := AM + dist M C + AC
  let perimeter_CNA := dist C N + dist N A + AC
  let perimeter_ANB := dist A N + dist N B + AB
  let perimeter_CMB := dist C M + dist M B + BC
  in
  perimeter_AMC = perimeter_CNA ∧ perimeter_ANB = perimeter_CMB

def isosceles_triangle (A B C : Point) : Prop :=
  dist A B = dist A C

theorem triangle_is_isosceles (A B C M N : Point) :
  triangle_isosceles_condition A B C M N →
  isosceles_triangle A B C :=
sorry

end triangle_is_isosceles_l743_743171


namespace range_of_a_l743_743784

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + x else x^2 + x -- Note: Using the specific definition matches the problem constraints clearly.

theorem range_of_a (a : ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_ineq : f a + f (-a) < 4) : -1 < a ∧ a < 1 := 
by sorry

end range_of_a_l743_743784


namespace probability_two_dice_same_number_l743_743181

theorem probability_two_dice_same_number : 
  let dice_sides := 8 in
  let total_outcomes := dice_sides ^ 8 in
  let different_outcomes := (fact dice_sides) / (fact (dice_sides - 8)) in
  (1 - (different_outcomes / total_outcomes)) = (1291 / 1296) :=
by
  sorry

end probability_two_dice_same_number_l743_743181


namespace number_of_m_set_classes_l743_743425

-- Definitions pulled directly from the conditions
def is_m_set_class (X : Set α) (M : Set (Set α)) : Prop :=
  X ∈ M ∧ ∅ ∈ M ∧ 
  (∀ A B, A ∈ M → B ∈ M → (A ∪ B) ∈ M) ∧
  (∀ A B, A ∈ M → B ∈ M → (A ∩ B) ∈ M)

def X_set : Set (Set ℕ) := { ∅, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3} }

theorem number_of_m_set_classes (X : Set ℕ) (c : Set ℕ) (hX : X = {1, 2, 3}) (hc : c = {2, 3}) :
  ∃ M_set_classes : Finset (Set (Set ℕ)), 
    (∀ M ∈ M_set_classes, is_m_set_class X M ∧ c ∈ M) ∧ M_set_classes.card = 12 :=
sorry

end number_of_m_set_classes_l743_743425


namespace car_average_speed_l743_743272

def average_speed (speed1 speed2 : ℕ) (time1 time2 : ℕ) : ℕ := 
  (speed1 * time1 + speed2 * time2) / (time1 + time2)

theorem car_average_speed :
  average_speed 60 90 (1/3) (2/3) = 80 := 
by 
  sorry

end car_average_speed_l743_743272


namespace count_specific_integers_l743_743417

theorem count_specific_integers :
  ∃ M : Finset ℕ, (∀ m ∈ M, m < 2000 ∧ (∃ k : Finset ℕ, k.card = 3 ∧ ∀ i ∈ k, ∃ m' : ℕ, 2 * m' + k = m))
    ∧ M.card = 11 := 
begin
  sorry
end

end count_specific_integers_l743_743417


namespace bus_speed_l743_743431

theorem bus_speed (d t : ℕ) (h1 : d = 201) (h2 : t = 3) : d / t = 67 :=
by sorry

end bus_speed_l743_743431


namespace ellipse_product_axes_l743_743907

/-- Prove that the product of the lengths of the major and minor axes (AB)(CD) of an ellipse
is 240, given the following conditions:
- Point O is the center of the ellipse.
- Point F is one focus of the ellipse.
- OF = 8
- The diameter of the inscribed circle of triangle OCF is 4.
- OA = OB = a
- OC = OD = b
- a² - b² = 64
- a - b = 4
-/
theorem ellipse_product_axes (a b : ℝ) (OF : ℝ) (d_inscribed_circle : ℝ) 
  (h1 : OF = 8) (h2 : d_inscribed_circle = 4) (h3 : a^2 - b^2 = 64) 
  (h4 : a - b = 4) : (2 * a) * (2 * b) = 240 :=
sorry

end ellipse_product_axes_l743_743907


namespace same_color_pair_exists_l743_743290

-- Define the coloring of a point on a plane
def is_colored (x y : ℝ) : Type := ℕ  -- Assume ℕ represents two colors 0 and 1

-- Prove there exists two points of the same color such that the distance between them is 2006 meters
theorem same_color_pair_exists (colored : ℝ → ℝ → ℕ) :
  (∃ (x1 y1 x2 y2 : ℝ), x1 ≠ x2 ∧ y1 ≠ y2 ∧ colored x1 y1 = colored x2 y2 ∧ (x2 - x1)^2 + (y2 - y1)^2 = 2006^2) :=
sorry

end same_color_pair_exists_l743_743290


namespace pseudo_symmetry_abscissa_l743_743516

noncomputable def f (x : ℝ) : ℝ := x^2 - 6*x + 4 * Real.log x

theorem pseudo_symmetry_abscissa :
  ∃ x0 : ℝ, x0 = Real.sqrt 2 ∧
    (∀ x : ℝ, x ≠ x0 → (f x - ((2*x0 + 4/x0 - 6)*(x - x0) + x0^2 - 6*x0 + 4*Real.log x0)) / (x - x0) > 0) :=
sorry

end pseudo_symmetry_abscissa_l743_743516


namespace cost_per_bag_of_potatoes_l743_743898

variable (x : ℕ)

def chickens_cost : ℕ := 5 * 3
def celery_cost : ℕ := 4 * 2
def total_paid : ℕ := 35
def potatoes_cost (x : ℕ) : ℕ := 2 * x

theorem cost_per_bag_of_potatoes : 
  chickens_cost + celery_cost + potatoes_cost x = total_paid → x = 6 :=
by
  sorry

end cost_per_bag_of_potatoes_l743_743898


namespace solution_set_sqrt3_sin_eq_cos_l743_743598

theorem solution_set_sqrt3_sin_eq_cos (x : ℝ) :
  (∀ x, √3 * sin x = cos x ↔ ∃ k : ℤ, x = k * π + π / 6) :=
by sorry

end solution_set_sqrt3_sin_eq_cos_l743_743598


namespace new_person_weight_l743_743572

-- Define the conditions
def average_weight_increase (new_weight : ℝ) (original_weight : ℝ) : Prop :=
  ∀ (W : ℝ), W - 35 + new_weight = W + 20

-- State the problem
theorem new_person_weight (new_weight : ℝ) : average_weight_increase new_weight 35 → new_weight = 55 :=
by 
  intro h
  specialize h 0 -- Using 0 as the total weight W for simplicity
  simp at h
  exact h

#check new_person_weight -- This ensures the theorem is valid

end new_person_weight_l743_743572


namespace isosceles_triangle_l743_743089

open_locale euclidean_geometry

variables {A B C I P Q M N : Point}
variables (α : circle)
variables (circumcircle_AIC : circle)

-- Condition 1
hypothesis h1 : incenter I (triangle.mk A B C)

-- Condition 2
hypothesis h2 : α = incircle (triangle.mk A B C)

-- Condition 3
hypothesis h3 : intersects circumcircle_AIC α P
hypothesis h4 : intersects circumcircle_AIC α Q

-- Condition 4
hypothesis h5 : same_side P A (line.mk B I)

-- Condition 5
hypothesis h6 : ¬ same_side Q C (line.mk B I)

-- Condition 6
hypothesis h7 : midpoint M (arc.mk α A C)

-- Condition 7
hypothesis h8 : midpoint N (arc.mk α B C)

-- Condition 8
hypothesis h9 : parallel (line.mk P Q) (line.mk A C)

-- Conclusion
theorem isosceles_triangle (h1 h2 h3 h4 h5 h6 h7 h8 h9) : (distance A B) = (distance A C) :=
sorry

end isosceles_triangle_l743_743089


namespace simplify_G_in_terms_of_F_l743_743075

def F (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

def G (x : ℝ) : ℝ := log ((1 + (2 * x - x^2) / (1 + 2 * x^2)) / (1 - (2 * x - x^2) / (1 + 2 * x^2)))

theorem simplify_G_in_terms_of_F (x : ℝ) : G x = 2 * F x :=
by
  sorry

end simplify_G_in_terms_of_F_l743_743075


namespace intersection_A_B_l743_743513

-- Define set A
def A (x : ℝ) : ℝ := 2^x - 1

-- Define set B
def B := {x | abs (2*x - 3) ≤ 3}

-- State the proof problem
theorem intersection_A_B :
  ∀ x, (0 < x ∧ x ≤ 3) ↔ (A x ∈ ({y | y > 0} : set ℝ) ∧ x ∈ B) :=
by
  sorry

end intersection_A_B_l743_743513


namespace max_tiles_l743_743189

/--
Given a rectangular floor of size 180 cm by 120 cm
and rectangular tiles of size 25 cm by 16 cm, prove that the maximum number of tiles
that can be accommodated on the floor without overlapping, where the tiles' edges
are parallel and abutting the edges of the floor and with no tile overshooting the edges,
is 49 tiles.
-/
theorem max_tiles (floor_len floor_wid tile_len tile_wid : ℕ) (h1 : floor_len = 180)
  (h2 : floor_wid = 120) (h3 : tile_len = 25) (h4 : tile_wid = 16) :
  ∃ max_tiles : ℕ, max_tiles = 49 :=
by
  sorry

end max_tiles_l743_743189


namespace pqrs_predicate_l743_743373

noncomputable def P (a b c : ℝ) := a + b - c
noncomputable def Q (a b c : ℝ) := b + c - a
noncomputable def R (a b c : ℝ) := c + a - b

theorem pqrs_predicate (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (P a b c) * (Q a b c) * (R a b c) > 0 ↔ (P a b c > 0 ∧ Q a b c > 0 ∧ R a b c > 0) :=
sorry

end pqrs_predicate_l743_743373


namespace isosceles_triangle_l743_743135

   open EuclideanGeometry

   -- Define the conditions of the problem in Lean
   variable {I A B C P Q M N : Point}
   variable (α : Circle) (circumcircle_AIC : Circle)

   -- Conditions extracted from the problem
   def conditions : Prop :=
   IsIncenter I △ABC ∧
   Incircle α △ABC ∧
   Circle.Diameter α P Q ∧
   Circle.Containing circumcircle_AIC (trianglePoint AIC) ∧
   SameSide P A (Line BI) ∧
   SameSide Q C (Line BI) ∧
   IsMidpointArc M ARC(α A C) ∧
   IsMidpointArc N ARC(α B C) ∧
   Parallel (Line PQ) (Line AC)

   -- Proof statement in Lean
   theorem isosceles_triangle
     (h : conditions α circumcircle_AIC) : IsIsosceles (△ABC) :=
   sorry
   
end isosceles_triangle_l743_743135


namespace area_of_triangle_l743_743390

-- Definitions for the ellipse and the foci
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

-- Distance between points
def dist (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Definition of foci of the ellipse: F₁ and F₂
def f1 : ℝ × ℝ := (-real.sqrt(2), 0)
def f2 : ℝ × ℝ := (real.sqrt(2), 0)

-- Given a point P on the ellipse
variable (P : ℝ × ℝ)

-- Length conditions
def length_condition1 : Prop := dist P f1 - dist P f2 = 2

-- Theorem to prove the area of the triangle given the conditions
theorem area_of_triangle (h1 : ellipse_eq P.1 P.2) (h2 : length_condition1 P) :
  (1 / 2) * dist P f2 * dist f1 f2 = real.sqrt(2) :=
sorry

end area_of_triangle_l743_743390


namespace solution_correct_l743_743356

open Int

def toCelsius (F : ℤ) : ℝ := (5 / 9 : ℝ) * (F - 32)

def toFahrenheit (C : ℤ) : ℤ := (9 * C / 5 + 32 : ℝ).round

def roundToNearest (x : ℝ) : ℤ := x.round

def validTemperatures : ℕ :=
  Nat.card (Finset.filter (λ F : ℤ, 
      roundToNearest (toFahrenheit (roundToNearest (toCelsius F))) = F) 
    (Finset.range' 10 (200-10+1)))

noncomputable def solution := validTemperatures

theorem solution_correct : solution = 84 :=
by
  sorry

end solution_correct_l743_743356


namespace f_is_even_l743_743881

variable (h : ℝ → ℝ)
variable (f : ℝ → ℝ)

noncomputable def is_even_function (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g(-x) = g(x)

noncomputable def f_def (x : ℝ) : ℝ := abs (h (x^4))

theorem f_is_even (h_even : is_even_function h) :
  is_even_function f_def :=
by
  intro x
  rw [f_def, f_def, h_even (x^4)]
  congr
  simp [f_def, abs]

end f_is_even_l743_743881


namespace correct_average_marks_l743_743445

theorem correct_average_marks 
  (num_students : ℕ)
  (reported_avg : ℚ)
  (reported_marks : ℕ)
  (num_error1_corrected : ℕ) 
  (error1_diff : ℕ) 
  (num_error2_corrected : ℕ) 
  (error2_diff : ℕ) 
  (num_error3_corrected : ℕ)
  (error3_diff : ℕ)
  (corrected_avg: ℚ) :
  num_students = 30 → 
  reported_avg = 90 → 
  reported_marks = 2700 → 
  num_error1_corrected = 1 →
  error1_diff = 40 →
  num_error2_corrected = 1 →
  error2_diff = 30 →
  num_error3_corrected = 1 →
  error3_diff = 20 →
  corrected_avg = 89.67 :=
by
  intros h_ns h_ra h_rm h_nec1 h_e1d h_nec2 h_e2d h_nec3 h_e3d
  let net_error := error1_diff - error2_diff - error3_diff
  let initial_total_marks := reported_avg * num_students
  let corrected_total_marks := initial_total_marks - net_error
  let computed_corrected_avg := corrected_total_marks / num_students
  show corrected_avg = computed_corrected_avg
  admit

end correct_average_marks_l743_743445


namespace sum_odd_impossible_l743_743241

open Nat

-- Defining the type of digits
def is_digit (n : ℕ) : Prop :=
  n ≤ 9

-- Function to sum two digit numbers in reverse order where each a_i is a digit
def sum_1001_digit_numbers (a : Fin 1001 → ℕ) : Fin 1001 → ℕ :=
  λ i => a i + a (1000 - i)

-- Prove that it is impossible for the sum to have all odd digits
theorem sum_odd_impossible (a : Fin 1001 → ℕ) 
  (H : ∀ i, is_digit (a i)) : ¬(∀ i, odd (sum_1001_digit_numbers a i)) :=
begin
  sorry
end

end sum_odd_impossible_l743_743241


namespace train_cross_time_l743_743945

noncomputable def train_length : ℕ := 1200 -- length of the train in meters
noncomputable def platform_length : ℕ := train_length -- length of the platform equals the train length
noncomputable def speed_kmh : ℝ := 144 -- speed in km/hr
noncomputable def speed_ms : ℝ := speed_kmh * (1000 / 3600) -- converting speed to m/s

-- the formula to calculate the crossing time
noncomputable def time_to_cross_platform : ℝ := 
  2 * train_length / speed_ms

theorem train_cross_time : time_to_cross_platform = 60 := by
  sorry

end train_cross_time_l743_743945


namespace curve_is_line_l743_743342

def curve := {p : ℝ × ℝ | ∃ (θ : ℝ), (p.1 = (1 / (Real.sin θ + Real.cos θ)) * Real.cos θ
                                        ∧ p.2 = (1 / (Real.sin θ + Real.cos θ)) * Real.sin θ)}

-- Problem: Prove that the curve defined by the polar equation is a line.
theorem curve_is_line : ∀ (p : ℝ × ℝ), p ∈ curve → p.1 + p.2 = 1 :=
by
  -- The proof is omitted.
  sorry

end curve_is_line_l743_743342


namespace rule1_commutative_l743_743495

def max (a b : ℕ) : ℕ := if a > b then a else b
def min (a b : ℕ) : ℕ := if a < b then a else b

def diamond (a b : ℕ) : ℕ := max a b - min a b

theorem rule1_commutative (a b : ℕ) : diamond a b = diamond b a := by
  sorry

end rule1_commutative_l743_743495


namespace compound_interest_for_2_years_l743_743169

noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

noncomputable def compound_interest (P R T : ℝ) : ℝ := P * (1 + R / 100)^T - P

theorem compound_interest_for_2_years 
  (P : ℝ) (R : ℝ) (T : ℝ) (S : ℝ)
  (h1 : S = 600)
  (h2 : R = 5)
  (h3 : T = 2)
  (h4 : simple_interest P R T = S)
  : compound_interest P R T = 615 := 
sorry

end compound_interest_for_2_years_l743_743169


namespace least_n_multiple_of_55_l743_743497

def a : ℕ → ℕ
| 5 := 5
| (n + 1) := if n ≥ 5 then 200 * a n + (n + 1) else 0

theorem least_n_multiple_of_55 :
  ∃ n > 5, n = 32 ∧ a n % 55 = 0 := sorry

end least_n_multiple_of_55_l743_743497


namespace trajectory_of_C_l743_743025

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem trajectory_of_C
  (A B : ℝ × ℝ)
  (hA : A = (-4, 0))
  (hB : B = (4, 0))
  (hBC_AC : ∀ C : ℝ × ℝ, distance B C - distance A C = 1/2 * distance A B) :
  set_of (λ C : ℝ × ℝ, C.1 ^ 2 / 4 - C.2 ^ 2 / 12 = 1) ∩ { C | C.1 < -2 } = 
  { C : ℝ × ℝ | distance B C - distance A C = 1/2 * distance A B } :=
sorry

end trajectory_of_C_l743_743025


namespace sqrt_inequality_l743_743183

theorem sqrt_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z ≥ x * y + y * z + z * x) : 
  real.sqrt (x * y * z) ≥ real.sqrt x + real.sqrt y + real.sqrt z :=
sorry

end sqrt_inequality_l743_743183


namespace math_problem_l743_743303
open Real

noncomputable def problem_statement : Prop :=
  let a := 99
  let b := 3
  let c := 20
  let area := (99 * sqrt 3) / 20
  a + b + c = 122 ∧ 
  ∃ (AB: ℝ) (QR: ℝ), AB = 14 ∧ QR = 3 * sqrt 3 ∧ area = (1 / 2) * QR * (QR / (2 * (sqrt 3 / 2))) * (sqrt 3 / 2)

theorem math_problem : problem_statement := by
  sorry

end math_problem_l743_743303


namespace AD_length_l743_743888

open EuclideanGeometry

variables 
  (A B C D M : Point)
  (AB : Line A B)
  (BD : Line B D)
  (BC : Line B C)
  (M_BD_midpoint : midpoint M B D)
  (angle_ABD_eq_90 : angle A B D = 90)
  (angle_BCD_eq_90 : angle B C D = 90)
  (dist_CM_eq_2 : distance C M = 2)
  (dist_AM_eq_3 : distance A M = 3)

theorem AD_length : distance A D = sqrt 21 := by
  sorry

end AD_length_l743_743888


namespace find_a_and_distance_l743_743771

noncomputable def slope_of_line (a b c : ℝ) : ℝ :=
  -a / b

def distance_between_parallel_lines (A B C A' B' C' : ℝ) : ℝ :=
  |C - C'| / Real.sqrt (A^2 + B^2)

theorem find_a_and_distance
  (a : ℝ)
  (h_slope_l1 : slope_of_line a 1 (-1) = Real.tan (Real.pi / 3))
  (h_parallel : slope_of_line a 1 (-1) = slope_of_line 1 (-1) (-3)) :
  a = -Real.sqrt 3 ∧ distance_between_parallel_lines a 1 (-1) 1 (-1) (-3) = 2 * Real.sqrt 2 :=
by
  sorry

end find_a_and_distance_l743_743771


namespace find_x_l743_743240

theorem find_x (x : ℝ) (h : 65 + 5 * 12 / (x / 3) = 66) : x = 180 :=
by
  sorry

end find_x_l743_743240


namespace sum_vertices_nice_regions_lt_40n_l743_743046

-- Condition: There are n rectangles with parallel sides.
variable (n : ℕ)

-- Condition: The sides of distinct rectangles lie on distinct lines.
axiom distinct_lines (rectangles : Fin n → Set (Set ℝ×ℝ)) :
  ∀ i j, i ≠ j → ∀ s (H_i : s ∈ rectangles i) (H_j : s ∈ rectangles j), False

-- Condition: The boundaries of the rectangles divide the plane into connected regions.
axiom divides_plane (rectangles : Fin n → Set (Set ℝ×ℝ)) :
  ∃ regions : Set (Set (Set ℝ×ℝ)), ∀ p q, (p ∈ regions ∧ q ∈ regions) → p ∩ q = ∅

-- Condition: A region is nice if it contains at least one of the vertices of the n rectangles on its boundary.
def is_nice_region (rectangles : Fin n → Set (Set (ℝ×ℝ)))
  (region : Set (ℝ×ℝ)) : Prop :=
  ∃ i, ∃ vertex ∈ rectangles i, vertex ∈ region

-- Main statement to prove: The sum of the numbers of vertices of all nice regions is less than 40n.
theorem sum_vertices_nice_regions_lt_40n
  (rectangles : Fin n → Set (Set (ℝ×ℝ)))
  (H1 : distinct_lines rectangles)
  (H2 : divides_plane rectangles) :
  ∃ verts_sum : ℕ, verts_sum < 40 * n :=
sorry

end sum_vertices_nice_regions_lt_40n_l743_743046


namespace find_number_of_values_l743_743228

theorem find_number_of_values (n S : ℕ) (h1 : S / n = 250) (h2 : S + 30 = 251 * n) : n = 30 :=
sorry

end find_number_of_values_l743_743228


namespace incorrect_statement_A_l743_743384

theorem incorrect_statement_A (m : ℝ) :
  let y := λ x, x^2 - 2 * m * x + m^2 - 9
  ∃ x : ℝ, y x = m^2 - 9 → False :=
by
  let y := λ x, x^2 - 2 * m * x + m^2 - 9
  have minY : ∀ x, y x ≥ -9 := 
    sorry  
  have h : ∀ x, y x = m^2 - 9 → x = m ∧ -9 = m^2 - 9 :=
    sorry
  obtain ⟨x, hx⟩ := h
  have eq := minY x
  rw [hx] at eq
  exact eq, 
  sorry


termination_with
termination_axiom _ := 
  let y := λ x, x^2 - 2 * m * x + m^2 - 9
  ∃ x : ℝ, y x = -9 :=
by 
  let y := λ x, x^2 - 2 * m * x + m^2 - 9
  have minY : (x - m)^2 ≥ 0 :=
    by 
      sorry 
  ∃ x : ℝ, y x = -9 :=
    sorry  
ē

end incorrect_statement_A_l743_743384


namespace limit_of_sequence_S_2015_l743_743155

open BigOperators

noncomputable def sequence_a (n : ℕ) : ℝ := sorry

noncomputable def sequence_S (n : ℕ) : ℝ := 
if n = 0 then 0 else 1 - sequence_a n

noncomputable def sequence_S_k : ℕ → ℕ → ℝ
| 1, n => sequence_S n
| k+1, n => ∑ i in Finset.range n, sequence_S_k k (i + 1)

theorem limit_of_sequence_S_2015 (h₁ : ∀ n, sequence_S n + sequence_a n = 1) :
  tendsto (fun n => (sequence_S_k 2015 n) / (n:ℝ)^2014) at_top (𝓝 (1 / ↑(nat.factorial 2014))) :=
sorry

end limit_of_sequence_S_2015_l743_743155


namespace total_capacity_of_schools_l743_743839

theorem total_capacity_of_schools (a b c d t : ℕ) (h_a : a = 2) (h_b : b = 2) (h_c : c = 400) (h_d : d = 340) :
  t = a * c + b * d → t = 1480 := by
  intro h
  rw [h_a, h_b, h_c, h_d] at h
  simp at h
  exact h

end total_capacity_of_schools_l743_743839


namespace problem_statement_l743_743321

theorem problem_statement {n d : ℕ} (hn : 0 < n) (hd : 0 < d) (h1 : d ∣ n) (h2 : d^2 * n + 1 ∣ n^2 + d^2) :
  n = d^2 :=
sorry

end problem_statement_l743_743321


namespace reciprocal_of_repeating_decimal_l743_743989

theorem reciprocal_of_repeating_decimal :
  (1 / (0.33333333 : ℚ)) = 3 := by
  sorry

end reciprocal_of_repeating_decimal_l743_743989


namespace number_of_odd_palindromes_l743_743163

def is_palindrome (n : ℕ) : Prop :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := n / 100
  n < 1000 ∧ n >= 100 ∧ d0 = d2

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

theorem number_of_odd_palindromes : ∃ n : ℕ, is_palindrome n ∧ is_odd n → n = 50 :=
by
  sorry

end number_of_odd_palindromes_l743_743163


namespace simplify_expression_l743_743555

theorem simplify_expression : (15625 = 5^6) → (sqrt (√(1/15625) ^ (1/3)) = sqrt (5) / 5) := by
  intro h
  rw [h]
  sorry

end simplify_expression_l743_743555


namespace total_blankets_collected_l743_743756

theorem total_blankets_collected : 
  let original_members := 15
  let new_members := 5
  let blankets_per_original_member_first_day := 2
  let blankets_per_original_member_second_day := 2
  let blankets_per_new_member_second_day := 4
  let tripled_first_day_total := 3
  let blankets_school_third_day := 22
  let blankets_online_third_day := 30
  let first_day_blankets := original_members * blankets_per_original_member_first_day
  let second_day_original_members_blankets := original_members * blankets_per_original_member_second_day
  let second_day_new_members_blankets := new_members * blankets_per_new_member_second_day
  let second_day_additional_blankets := tripled_first_day_total * first_day_blankets
  let second_day_blankets := second_day_original_members_blankets + second_day_new_members_blankets + second_day_additional_blankets
  let third_day_blankets := blankets_school_third_day + blankets_online_third_day
  let total_blankets := first_day_blankets + second_day_blankets + third_day_blankets
  -- Prove that
  total_blankets = 222 :=
by 
  sorry

end total_blankets_collected_l743_743756


namespace valid_5_digit_numbers_l743_743454

theorem valid_5_digit_numbers (digits : List ℕ) (h_digits : digits = [6, 4, 5, 3, 0]) :
  let poss_first_digits := [6, 4, 5, 3],  -- The valid first digits
      factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n,
      count_valid_numbers := 4 * factorial 4 in
  count_valid_numbers = 96 :=
by
  intros
  have h1 : factorial 0 = 1 := rfl
  have h2 : factorial 1 = 1 := rfl
  have h3 : factorial 2 = 2 := rfl
  have h4 : factorial 3 = 6 := rfl
  have h5 : factorial 4 = 24 := rfl
  have h_factorial : factorial 4 = 24 := h5
  have h_count_valid_numbers : count_valid_numbers = 4 * factorial 4 := rfl
  rw [h_factorial] at h_count_valid_numbers
  have h_result : count_valid_numbers = 4 * 24 := h_count_valid_numbers
  norm_num at h_result
  exact h_result

end valid_5_digit_numbers_l743_743454


namespace johns_daily_tire_production_l743_743477

noncomputable theory

def daily_tires_produced (cost_per_tire : ℝ) (sell_multiplier : ℝ) (max_demand : ℝ) (weekly_loss : ℝ) 
  (days_in_week : ℝ): ℝ :=
  let selling_price_per_tire := cost_per_tire * sell_multiplier in
  let daily_loss := weekly_loss / days_in_week in
  let unsold_tires := daily_loss / selling_price_per_tire in
  max_demand - unsold_tires.floor

theorem johns_daily_tire_production : daily_tires_produced 250 1.5 1200 175000 7 = 1134 := 
by {
  unfold daily_tires_produced,
  norm_num,
  sorry
}

end johns_daily_tire_production_l743_743477


namespace integer_solutions_count_l743_743006

-- Define the inequality condition
def inequality_condition (n : ℤ) : Prop :=
  (n - 3) * (n + 5) ≤ 0

-- Define the math proof problem
theorem integer_solutions_count : 
  { n : ℤ | inequality_condition n }.toFinset.card = 9 :=
by
  sorry

end integer_solutions_count_l743_743006


namespace blue_red_area_ratio_l743_743967

theorem blue_red_area_ratio (d_small d_large : ℕ) (h1 : d_small = 2) (h2 : d_large = 6) :
    let r_small := d_small / 2
    let r_large := d_large / 2
    let A_red := Real.pi * (r_small : ℝ) ^ 2
    let A_large := Real.pi * (r_large : ℝ) ^ 2
    let A_blue := A_large - A_red
    A_blue / A_red = 8 :=
by
  sorry

end blue_red_area_ratio_l743_743967


namespace sqrt_112_consecutive_integers_product_l743_743605

theorem sqrt_112_consecutive_integers_product : 
  (∃ (a b : ℕ), a * a < 112 ∧ 112 < b * b ∧ b = a + 1 ∧ a * b = 110) :=
by 
  use 10, 11
  repeat { sorry }

end sqrt_112_consecutive_integers_product_l743_743605


namespace projection_of_a_on_b_is_correct_l743_743408

def vec_a : ℝ × ℝ := (2, 3)
def vec_b : ℝ × ℝ := (-1, 2)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

def projection (a b : ℝ × ℝ) : ℝ :=
  dot_product a b / magnitude b

theorem projection_of_a_on_b_is_correct :
  projection vec_a vec_b = 4 * real.sqrt 5 / 5 :=
by
  sorry

end projection_of_a_on_b_is_correct_l743_743408


namespace find_x_solution_l743_743747

theorem find_x_solution (x : ℝ) : sqrt (4 * x + 9) = 11 → x = 28 :=
by
  -- The proof will go here
  sorry

end find_x_solution_l743_743747


namespace total_wings_l743_743063

-- Conditions
def money_per_grandparent : ℕ := 50
def number_of_grandparents : ℕ := 4
def bird_cost : ℕ := 20
def wings_per_bird : ℕ := 2

-- Calculate the total amount of money John received:
def total_money_received : ℕ := number_of_grandparents * money_per_grandparent

-- Determine the number of birds John can buy:
def number_of_birds : ℕ := total_money_received / bird_cost

-- Prove that the total number of wings all the birds have is 20:
theorem total_wings : number_of_birds * wings_per_bird = 20 :=
by
  sorry

end total_wings_l743_743063


namespace exists_natural_numbers_with_properties_l743_743719

open Nat

def number_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).count (λ d => d ∣ n)

def sum_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d => d ∣ n) |>.sum

theorem exists_natural_numbers_with_properties :
  ∃ (x y : ℕ), x < y ∧ number_of_divisors x = number_of_divisors y ∧ sum_of_divisors x > sum_of_divisors y :=
  sorry

end exists_natural_numbers_with_properties_l743_743719


namespace modulus_of_z_l743_743374

-- Defining the imaginary unit
def i : ℂ := complex.I

-- Defining z
def z : ℂ := (2 + i) / i

-- The theorem stating the problem
theorem modulus_of_z : complex.abs z = real.sqrt 5 := by
  sorry

end modulus_of_z_l743_743374


namespace triangle_altitude_l743_743569

theorem triangle_altitude (b : ℕ) (h : ℕ) (area : ℕ) (h_area : area = 800) (h_base : b = 40) (h_formula : area = (1 / 2) * b * h) : h = 40 :=
by
  sorry

end triangle_altitude_l743_743569


namespace volleyball_tournament_l743_743449

open Finset

def tournament_game (n : ℕ) (team_lost_to : Finset (Finset (Fin n)) (Fin n)) := 
  ∀ s : Finset (Fin n), s.card = 55 → ∃ t ∈ s, (team_lost_to t s).card ≤ 4

theorem volleyball_tournament (team_lost_to : Finset (Fin 110) → Fin (110 → Finset (Fin 110))) :
  (tournament_game 110 team_lost_to) → ∃ t : Fin 110, (team_lost_to (univ \ {t}) t).card ≤ 4 :=
by
  sorry

end volleyball_tournament_l743_743449


namespace largest_angle_in_triangle_PQR_is_75_degrees_l743_743051

noncomputable def largest_angle (p q r : ℝ) : ℝ :=
  if p + q + 2 * r = p^2 ∧ p + q - 2 * r = -1 then 
    Real.arccos ((p^2 + q^2 - (p^2 + p*q + (1/2)*q^2)/2) / (2 * p * q)) * (180/Real.pi)
  else 
    0

theorem largest_angle_in_triangle_PQR_is_75_degrees (p q r : ℝ) (h1 : p + q + 2 * r = p^2) (h2 : p + q - 2 * r = -1) :
  largest_angle p q r = 75 :=
by sorry

end largest_angle_in_triangle_PQR_is_75_degrees_l743_743051


namespace problem_l743_743798

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x / 4) + (a / x) - (Real.log x) - (3 / 2)

theorem problem (a : ℝ) (f : ℝ → ℝ) (ha : f = λ x, (x / 4) + (a / x) - (Real.log x) - (3 / 2)) :
    (deriv f 1 = -2) ↔ (a = 5 / 4) ∧ 
    (∀ x, (0 < x ∧ x < 5 → deriv f x < 0) ∧ 
         (5 < x → deriv f x > 0) ∧
         (f 5 = -Real.log 5)) :=
by
  sorry

end problem_l743_743798


namespace complex_division_identity_l743_743250

noncomputable def left_hand_side : ℂ := (-2 : ℂ) + (5 : ℂ) * Complex.I / (6 : ℂ) - (3 : ℂ) * Complex.I
noncomputable def right_hand_side : ℂ := - (9 : ℂ) / 15 + (8 : ℂ) / 15 * Complex.I

theorem complex_division_identity : left_hand_side = right_hand_side := 
by
  sorry

end complex_division_identity_l743_743250


namespace half_original_amount_money_is_70_l743_743867

variable (M : ℝ)

-- Conditions as definitions
def spent_on_clothes := (3 / 7) * M
def spent_on_books := (2 / 5) * M
def remaining_money := 24
def total_spent := spent_on_clothes M + spent_on_books M

-- Lean 4 statement to prove the solution
theorem half_original_amount_money_is_70
  (h : M - total_spent M = remaining_money) :
  M / 2 = 70 :=
sorry

end half_original_amount_money_is_70_l743_743867


namespace maximum_temperature_difference_l743_743565

theorem maximum_temperature_difference
  (highest_temp : ℝ) (lowest_temp : ℝ)
  (h_highest : highest_temp = 58)
  (h_lowest : lowest_temp = -34) :
  highest_temp - lowest_temp = 92 :=
by sorry

end maximum_temperature_difference_l743_743565


namespace grazing_area_of_goat_l743_743313

/-- 
Consider a circular park with a diameter of 50 feet, and a square monument with 10 feet on each side.
Sally ties her goat on one corner of the monument with a 20-foot rope. Calculate the total grazing area
around the monument considering the space limited by the park's boundary.
-/
theorem grazing_area_of_goat : 
  let park_radius := 25
  let monument_side := 10
  let rope_length := 20
  let monument_radius := monument_side / 2 
  let grazing_quarter_circle := (1 / 4) * Real.pi * rope_length^2
  let ungrazable_area := (1 / 4) * Real.pi * monument_radius^2
  grazing_quarter_circle - ungrazable_area = 93.75 * Real.pi :=
by
  sorry

end grazing_area_of_goat_l743_743313


namespace binomial_sum_remainder_l743_743184

noncomputable def binomial (n k : ℕ) : ℕ := if h : k ≤ n then Nat.choose n k else 0

theorem binomial_sum_remainder
  (n : ℕ)
  (h : n = 12002) :
  (∑ k in Finset.range (n+1), if (k%6 = 1)%6 then binomial n k else 0) % 3 = 1 :=
by
  rw h
  sorry

end binomial_sum_remainder_l743_743184


namespace probability_at_least_two_same_l743_743177

theorem probability_at_least_two_same (n : ℕ) (s : ℕ) (h_n : n = 8) (h_s : s = 8) :
  let total_outcomes := s ^ n
      different_outcomes := Nat.factorial s
      prob_all_different := different_outcomes / total_outcomes
      prob_at_least_two_same := 1 - prob_all_different
  in prob_at_least_two_same = 1291 / 1296 :=
by
  -- Define values
  have h_total_outcomes : total_outcomes = 16777216 := by sorry
  have h_different_outcomes : different_outcomes = 40320 := by sorry
  have h_prob_all_different : prob_all_different = 5 / 1296 := by sorry
  -- Calculate probability of at least two dice showing the same number
  have h_prob_at_least_two_same : prob_at_least_two_same = 1 - (5 / 1296) := by
    unfold prob_at_least_two_same prob_all_different
    rw h_different_outcomes
    rw h_total_outcomes
    rw h_prob_all_different
  -- Simplify
  calc
    prob_at_least_two_same = 1 - (5 / 1296) : by rw h_prob_at_least_two_same
    ... = 1291 / 1296 : by sorry

end probability_at_least_two_same_l743_743177


namespace sqrt_of_product_minus_one_l743_743704

theorem sqrt_of_product_minus_one (x : ℕ) (hx : x = 23) : 
  nat.sqrt ((25 * 24 * 23 * 22) - 1) = x^2 + x - 3 :=
by {
  rw hx,
  -- the rest of the proof is omitted
  sorry
}

end sqrt_of_product_minus_one_l743_743704


namespace reorderings_rel_prime_l743_743007

theorem reorderings_rel_prime :
  let S := {2, 3, 4, 5, 6}
  ∃ l : List ℕ, (l.Perm S.toList) ∧ (∀ (x y : ℕ), (x ∈ l ∧ y ∈ l.tail) → Nat.gcd x y = 1) → l.length = 5 :=
by
  sorry

end reorderings_rel_prime_l743_743007


namespace largest_area_figure_has_area_6_l743_743357

noncomputable def triangle_area (base height: ℝ) : ℝ :=
  (1/2) * base * height

noncomputable def rectangle_area (length width: ℝ) : ℝ :=
  length * width

noncomputable def figure_area (triangles rectangles : Nat) : ℝ :=
  triangles * triangle_area 1 1 + rectangles * rectangle_area 2 1

theorem largest_area_figure_has_area_6 :
  let figureA_area := figure_area 4 1 in
  let figureB_area := figure_area 2 2 in
  let figureC_area := figure_area 0 3 in
  let figureD_area := figure_area 5 0 in
  max (max figureA_area figureB_area)
      (max figureC_area figureD_area) = 6 := by
  sorry

end largest_area_figure_has_area_6_l743_743357


namespace max_area_triangle_l743_743464

/-- Define the conditions given in the problem -/
structure TriangleProblem where
  PA : ℝ
  PB : ℝ
  PC : ℝ
  BC : ℝ
  PA_pos : 0 < PA
  PB_pos : 0 < PB
  PC_pos : 0 < PC
  BC_pos : 0 < BC

/-- Instantiate the specific problem conditions -/
def problem_instance : TriangleProblem := {
  PA := 3,
  PB := 4,
  PC := 5,
  BC := 6,
  PA_pos := by norm_num,
  PB_pos := by norm_num,
  PC_pos := by norm_num,
  BC_pos := by norm_num,
}

/-- Prove that the maximum area of the triangle ABC is 18.921 -/
theorem max_area_triangle (t : TriangleProblem) : 
  max_area t.PA t.PB t.PC t.BC = 18.921 :=
sorry

end max_area_triangle_l743_743464


namespace circle_cone_construction_l743_743316

-- Two given points on a plane
variables {A B : ℝ × ℝ}

-- Define the conditions required for the circle and cone construction
def circle_tangent_to_plane (A B : ℝ × ℝ) : Prop :=
∃ (r : ℝ) (center : ℝ × ℝ), r > 0 ∧ 
A = (center.1 + r * cos (π / 3), center.2 + r * sin (π / 3)) ∧ 
B = (center.1 + r * cos (-π / 3), center.2 + r * sin (-π / 3))

def equilateral_cone_over_circle (A B : ℝ × ℝ) : Prop :=
∀ r : ℝ, r > 0 → 
∃ (h : ℝ) (center : ℝ × ℝ), h > 0 ∧ ∀ θ : ℝ, 
A = (center.1 + r * cos θ, center.2 + r * sin θ) ∧ 
B = (center.1 + r * cos (θ + 2 * π / 3), center.2 + r * sin (θ + 2 * π / 3))

theorem circle_cone_construction (A B : ℝ × ℝ) :
  circle_tangent_to_plane A B → equilateral_cone_over_circle A B :=
sorry

end circle_cone_construction_l743_743316


namespace trajectory_equation_l743_743287

-- Define the parabola y = 2x^2 + 1
def parabola (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the function that gives the y coordinate of the parabola point given x coordinate
def P (x : ℝ) : ℝ := parabola x

-- Define the midpoint M of P and Q
def midpoint_M (x : ℝ) : ℝ × ℝ := ((x + 0) / 2, (P x - 1) / 2)

-- Define the trajectory equation for M
def trajectory (x : ℝ) : ℝ := 4 * x^2

-- Prove the trajectory equation
theorem trajectory_equation (x y : ℝ) (h : (x, y) = midpoint_M x) : y = trajectory x :=
by 
  sorry

end trajectory_equation_l743_743287


namespace sum_first_nine_terms_l743_743845

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem sum_first_nine_terms
  (a d : ℝ)
  (h1 : a + (a + 3 * d) + (a + 6 * d) = 35)
  (h2 : (a + 2 * d) + (a + 5 * d) + (a + 8 * d) = 27) :
  let S₉ := (9 / 2) * (2 * a + (9 - 1) * d) in
  S₉ = 99 :=
by
  sorry

end sum_first_nine_terms_l743_743845


namespace school_club_net_profit_l743_743678

theorem school_club_net_profit (candy_bars : ℕ) (cost_per_eight : ℚ) (sales_price_three : ℚ) (setup_cost : ℚ)
  (h_candy_bars : candy_bars = 1500)
  (h_cost_per_eight : cost_per_eight = 3)
  (h_sales_price_three : sales_price_three = 2)
  (h_setup_cost : setup_cost = 50) :
  let cost_per_candy := cost_per_eight / 8
      total_cost := candy_bars * cost_per_candy
      revenue_per_candy := sales_price_three / 3
      total_revenue := candy_bars * revenue_per_candy
      net_profit := total_revenue - total_cost - setup_cost
  in net_profit = 387.5 :=
by
  sorry

end school_club_net_profit_l743_743678


namespace distance_between_house_and_school_l743_743636

theorem distance_between_house_and_school (T D : ℕ) 
    (h1 : D = 10 * (T + 2)) 
    (h2 : D = 20 * (T - 1)) : 
    D = 60 := by
  sorry

end distance_between_house_and_school_l743_743636


namespace maximum_dot_product_l743_743047

-- Definitions and conditions
def point := (ℝ, ℝ)

def A : point := (0, 0)
def B : point := (3, 0)
def C : point := (2, 2)
def D : point := (0, 2)
def N : point := (1, 2)

def onLineBC (M : point) : Prop :=
  ∃ (λ : ℝ), 2 ≤ λ ∧ λ ≤ 3 ∧ M = (λ, -2 * λ + 6)

-- Dot product definition
def dot_product (u v : point) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Vectors from A to M and from A to N
def AM (M : point) : point := (M.1 - A.1, M.2 - A.2)
def AN : point := (N.1 - A.1, N.2 - A.2)

theorem maximum_dot_product : 
  ∃ M : point, onLineBC M ∧ (∀ M', onLineBC M' → dot_product (AM M') AN ≤ dot_product (AM M) AN)
  ∧ dot_product (AM M) AN = 6 :=
by
  sorry

end maximum_dot_product_l743_743047


namespace intersect_circumcircle_l743_743220

variables {A B C P Q M N : Type*}
variables [euclidean_space A B C P Q M N]

-- The conditions
variables (triangle_ABC : triangle A B C)
variables (acute_ABC : acute triangle_ABC)
variables (P_on_BC : point_on_line_segment B C P)
variables (Q_on_BC : point_on_line_segment B C Q)
variables (angle_EQ1 : ∠PAB = ∠ACB)
variables (angle_EQ2 : ∠QAC = ∠CBA)
variables (AP_eq_PM : distance A P = distance P M)
variables (AQ_eq_QN : distance A Q = distance Q N)

-- The theorem to be proven
theorem intersect_circumcircle :
  intersection_point (line B M) (line C N) ∈ circumcircle A B C :=
sorry

end intersect_circumcircle_l743_743220


namespace find_x_of_equation_l743_743864

theorem find_x_of_equation (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := by
  sorry

end find_x_of_equation_l743_743864


namespace weekly_goal_cans_l743_743631

theorem weekly_goal_cans (c₁ c₂ c₃ c₄ c₅ : ℕ) (h₁ : c₁ = 20) (h₂ : c₂ = c₁ + 5) (h₃ : c₃ = c₂ + 5) 
  (h₄ : c₄ = c₃ + 5) (h₅ : c₅ = c₄ + 5) : 
  c₁ + c₂ + c₃ + c₄ + c₅ = 150 :=
by
  sorry

end weekly_goal_cans_l743_743631


namespace count_valid_N_l743_743420

theorem count_valid_N : ∃ (N : ℕ), N = 1174 ∧ ∀ (n : ℕ), (1 ≤ n ∧ n < 2000) → ∃ (x : ℝ), x ^ (⌊x⌋ + 1) = n :=
by
  sorry

end count_valid_N_l743_743420


namespace arc_cut_by_triangle_side_is_60_l743_743659

-- Define basic geometric entities
def equilateral_triangle : Type := sorry
def circle : Type := sorry

-- Define the height of an equilateral triangle
def height_of_triangle (t : equilateral_triangle) : ℝ := sorry

-- Define a circle with a given radius
def circle_with_radius (r : ℝ) : circle := sorry

-- Define the concept of an arc cut by the sides of the triangle on the circle
def arc_cut_by_sides (t : equilateral_triangle) (c : circle) : ℝ := sorry

-- Main theorem statement
theorem arc_cut_by_triangle_side_is_60 (t : equilateral_triangle) (c : circle)
  (h₁ : c = circle_with_radius (height_of_triangle t))
  (h₂ : ∀ side, c rolls along side of t) :
  arc_cut_by_sides t c = 60 :=
sorry

end arc_cut_by_triangle_side_is_60_l743_743659


namespace factorization_l743_743335

theorem factorization (x : ℝ) : 
  (x^2 + 6 * x + 9 - 64 * x^4) = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3) :=
by
  sorry

end factorization_l743_743335


namespace sam_money_left_l743_743545

-- Assuming the cost per dime and quarter
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Given conditions
def dimes : ℕ := 19
def quarters : ℕ := 6
def cost_per_candy_bar_in_dimes : ℕ := 3
def candy_bars : ℕ := 4
def lollipops : ℕ := 1

-- Calculate the initial money in cents
def initial_money : ℕ := (dimes * dime_value) + (quarters * quarter_value)

-- Calculate the cost of candy bars in cents
def candy_bars_cost : ℕ := candy_bars * cost_per_candy_bar_in_dimes * dime_value

-- Calculate the cost of lollipops in cents
def lollipop_cost : ℕ := lollipops * quarter_value

-- Calculate the total cost of purchases in cents
def total_cost : ℕ := candy_bars_cost + lollipop_cost

-- Calculate the final money left in cents
def final_money : ℕ := initial_money - total_cost

-- Theorem to prove
theorem sam_money_left : final_money = 195 := by
  sorry

end sam_money_left_l743_743545


namespace total_puzzle_pieces_l743_743061

theorem total_puzzle_pieces : 
  ∀ (p1 p2 p3 : ℕ), 
  p1 = 1000 → 
  p2 = p1 + p1 / 2 → 
  p3 = p1 + p1 / 2 → 
  p1 + p2 + p3 = 4000 := 
by 
  intros p1 p2 p3 
  intro h1 
  intro h2 
  intro h3 
  rw [h1, h2, h3] 
  norm_num
  sorry

end total_puzzle_pieces_l743_743061


namespace pp_gt_qq_l743_743211

-- Triangle and its properties
variables {A B C P Q P' Q' : Point}
variables {a b c : Real} -- sides of the triangle

-- Given conditions
variable (h1 : Triangle A B C)
variable (h2 : (InCircle A B C).TouchesSide AC P)
variable (h3 : (ExCircle A B C).TouchesSide AC Q)
variable (h4 : Line A P ≠ Line B P) -- line BP meets circumcircle for the second time at P'
variable (h5 : Line A Q ≠ Line B Q) -- line BQ meets circumcircle for the second time at Q'

-- Question to be proven
theorem pp_gt_qq : dist P P' > dist Q Q' :=
  sorry

end pp_gt_qq_l743_743211


namespace reciprocal_of_repeating_decimal_three_l743_743985

noncomputable def repeating_decimal_to_fraction : ℚ :=
  let x := (0.33333333333 : ℚ) in 1 / 3

theorem reciprocal_of_repeating_decimal_three : 
  (1 / repeating_decimal_to_fraction) = 3 := by
  -- Reciprocal of the fraction
  sorry

end reciprocal_of_repeating_decimal_three_l743_743985


namespace reciprocal_of_repeating_decimal_l743_743987

theorem reciprocal_of_repeating_decimal :
  (1 / (0.33333333 : ℚ)) = 3 := by
  sorry

end reciprocal_of_repeating_decimal_l743_743987


namespace provinces_count_l743_743658

theorem provinces_count (P T n : ℕ) 
  (hT : T = 3 * P / 4)
  (h_prov : ∀ i, i mod n = 0 → (P / 12) = T / n)
  (h_ratio : T = 3 * P / 4) :
  n = 9 :=
sorry

end provinces_count_l743_743658


namespace zero_point_in_interval_l743_743944

def f (x : ℝ) : ℝ := x + 3^(x + 2)

theorem zero_point_in_interval : ∃ x : ℝ, f x = 0 ∧ x ∈ set.Ioo (-2 : ℝ) (-1) :=
sorry

end zero_point_in_interval_l743_743944


namespace line_through_intersections_fixed_point_l743_743707

def Circle (α : Type _) := α -- Type representing a circle

variables {α : Type _} [LinearOrder α]

noncomputable def midpoint (A P : α) (C : Circle α) (B : α) : α := sorry
noncomputable def circle_center (D : α) (passing : α) : Circle α := sorry
noncomputable def intersection_points (C1 C2 : Circle α) : set α := sorry
noncomputable def fixed_midpoint (A B : α) (C : Circle α) (P : α) : α := sorry

theorem line_through_intersections_fixed_point
  (C : Circle α)
  (A B P : α)
  (D := midpoint A P C B)
  (E := midpoint B P C A)
  (C1 := circle_center D A)
  (C2 := circle_center E B)
  (K := fixed_midpoint A B C P) :
  ∀ {P' : α}, P' ∈ intersection_points C1 C2 → (∃ l, l.contains P' ∧ l.contains K) := 
sorry

end line_through_intersections_fixed_point_l743_743707


namespace locus_M_parabola_min_area_quadrilateral_l743_743792

noncomputable def ellipse_eq : (ℝ × ℝ) → Prop :=
  λ p, (p.1^2 / 8) + (p.2^2 / 4) = 1

def F1 : (ℝ × ℝ) := (-2, 0)
def F2 : (ℝ × ℝ) := (2, 0)

def is_locus_M (M : ℝ × ℝ) : Prop :=
  M.2^2 = 8 * M.1

def quadrilateral_area (A B C D : ℝ × ℝ) : ℝ :=
  abs ((A.1 * B.2 + B.1 * C.2 + C.1 * D.2 + D.1 * A.2 - A.2 * B.1 - B.2 * C.1 - C.2 * D.1 - D.2 * A.1) / 2)

theorem locus_M_parabola :
  ∀ (F1 P F2 M : ℝ × ℝ),
    F1 = (-2, 0) ∧ F2 = (2, 0) ∧
    P.1 = F2.1 ∧
    ((M.1 = F2.1) ∨ ∃ l : ℝ × ℝ, l = (P.1, M.2) ∧ l.2 = M.2) ∧
    (abs (M.1 - P.1) = abs (M.1 - F2.1)) →
    is_locus_M M := by sorry

theorem min_area_quadrilateral :
  ∀ (F2 A B C D : ℝ × ℝ),
    ellipse_eq A ∧ ellipse_eq B ∧ ellipse_eq C ∧ ellipse_eq D ∧
    F2 = (2, 0) ∧
    (∃ k : ℝ, A = (2, 2 * k) ∧ C = (2, -2 * k) ∧ B = (2 * k, 2) ∧ D = (-2 * k, 2)) →
    quadrilateral_area A B C D = 64 / 9 := by sorry

end locus_M_parabola_min_area_quadrilateral_l743_743792


namespace max_intersection_points_l743_743966

theorem max_intersection_points (h1 : ∀ (l : ℕ), l = 2) (h2 : ∀ (l : ℕ), l = 3) : 
  ∃ (n : ℕ), n = 17 :=
by
  exists 17
  sorry

end max_intersection_points_l743_743966


namespace triangle_ABC_isosceles_l743_743101

-- Lean 4 definitions for the generated equivalent proof problem
noncomputable def incenter (A B C I : Type*) := sorry
noncomputable def incircle (ABC : Type*) (α : Type*) := sorry
noncomputable def circumcircle (A I C : Type*) := sorry
noncomputable def intersects (circle1 circle2 : Type*) (P Q : Type*) := sorry
noncomputable def same_side_line (P A : Type*) (line : Type*) := sorry
noncomputable def other_side_line (Q C : Type*) (line : Type*) := sorry
noncomputable def midpoint_arc (arc : Type*) := sorry
noncomputable def parallel (line1 line2 : Type*) := sorry
noncomputable def triangle_isosceles (A B C : Type*) := ∃ (ABC_is_isosceles : Prop), ABC_is_isosceles

-- Given conditions for the incenter, incircle, and parallel lines, we must prove the triangle is isosceles
theorem triangle_ABC_isosceles (A B C I α P Q M N : Type*)
  (h1 : incenter A B C I)
  (h2 : incircle ABC α)
  (h3 : circumcircle A I C)
  (h4 : intersects α (circumcircle A I C) P Q)
  (h5 : same_side_line P A (set I B))
  (h6 : other_side_line Q C (set I B))
  (h7 : midpoint_arc α AC M)
  (h8 : midpoint_arc α BC N)
  (h9 : parallel PQ AC) :
  triangle_isosceles A B C := sorry

end triangle_ABC_isosceles_l743_743101


namespace sum_of_roots_eq_18_l743_743207

variable (g : ℝ → ℝ)

def symmetric (g : ℝ → ℝ) : Prop :=
  ∀ x, g(3 + x) = g(3 - x)

theorem sum_of_roots_eq_18
  (h_symm : symmetric g)
  (h_roots : (∃ a1 a2 a3 a4 a5 a6 : ℝ, g a1 = 0 ∧ g a2 = 0 ∧ g a3 = 0 ∧ g a4 = 0 ∧ g a5 = 0 ∧ g a6 = 0 ∧ a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a1 ≠ a6 ∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a2 ≠ a6 ∧ a3 ≠ a4 ∧ a3 ≠ a5 ∧ a3 ≠ a6 ∧ a4 ≠ a5 ∧ a4 ≠ a6 ∧ a5 ≠ a6)) :
  ∃ roots : Finset ℝ, roots.card = 6 ∧ (∀ r ∈ roots, g r = 0) ∧ roots.sum id = 18 := by
  sorry

end sum_of_roots_eq_18_l743_743207


namespace find_m_n_l743_743010

theorem find_m_n (a b : ℝ) (m n : ℤ) :
  (a^m * b * b^n)^3 = a^6 * b^15 → m = 2 ∧ n = 4 :=
by
  sorry

end find_m_n_l743_743010


namespace john_got_rolls_l743_743478

def cost_per_dozen : ℕ := 5
def money_spent : ℕ := 15
def rolls_per_dozen : ℕ := 12

theorem john_got_rolls : (money_spent / cost_per_dozen) * rolls_per_dozen = 36 :=
by sorry

end john_got_rolls_l743_743478


namespace reciprocal_of_repeating_decimal_equiv_l743_743978

noncomputable def repeating_decimal (x : ℝ) := 0.333333...

theorem reciprocal_of_repeating_decimal_equiv :
  (1 / repeating_decimal 0.333333...) = 3 :=
sorry

end reciprocal_of_repeating_decimal_equiv_l743_743978


namespace f_7_eq_neg_2_l743_743780

noncomputable def f : ℝ → ℝ := sorry

lemma problem_conditions :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, f (x + 4) = f x) ∧
  (∀ x : ℝ, (0 < x ∧ x < 2) → f x = 2 * x^2) :=
by sorry

theorem f_7_eq_neg_2 : f 7 = -2 :=
by
  have h := problem_conditions,
  sorry

end f_7_eq_neg_2_l743_743780


namespace isosceles_triangle_l743_743109

noncomputable theory

open_locale classical

variables {A B C I P Q : Type*}

-- Let \( I \) be the incenter of triangle \( ABC \)
def is_incenter (I A B C : Type*) : Prop := sorry

-- Let \( \alpha \) be its incircle
def is_incircle (α : Type*) (I A B C : Type*) : Prop := sorry

-- The circumcircle of triangle \( AIC \) intersects \( \alpha \) at points \( P \) and \( Q \)
def circumcircle_intersect_incircle (α A I C P Q : Type*) : Prop := sorry

-- \( P \) and \( A \) lie on the same side of line \( BI \), and \( Q \) and \( C \) lie on the other side
def same_side (P A Q C : Type*) (BI : Type*) : Prop := sorry

-- \( PQ \parallel AC \)
def parallel (PQ AC : Type*) : Prop := sorry

-- Define triangle is isosceles
def is_isosceles (A B C : Type*) : Prop := sorry

theorem isosceles_triangle 
  (I : Type*) (A B C P Q M N : Type*)
  (α : Type*)
  (h_incenter : is_incenter I A B C)
  (h_incircle : is_incircle α I A B C)
  (h_intersect : circumcircle_intersect_incircle α A I C P Q)
  (h_sameside : same_side P A Q C (line BI))
  (h_parallel : parallel PQ (line AC))
: is_isosceles A B C :=
sorry

end isosceles_triangle_l743_743109


namespace area_of_circular_section_l743_743661

theorem area_of_circular_section (r : ℝ) (h : ℝ) (mid : ℝ := h / 2) (plane_parallel_base : Prop) 
    (base_radius : r = 2) (mid_height_plane : plane_parallel_base ∧ plane_through_mid_height : mid = h / 2) : 
    area_circular_section = π := by sorry

end area_of_circular_section_l743_743661


namespace number_of_palindromes_divisible_by_6_l743_743715

theorem number_of_palindromes_divisible_by_6 :
  let is_palindrome (n : ℕ) : Prop := n / 1000 = n % 10 ∧ (n / 100 % 10) = (n / 10 % 10)
  let valid_digits (n : ℕ) : Prop := n >= 1000 ∧ n < 10000
  let divisible_6 (n : ℕ) : Prop := n % 6 = 0
  (Finset.filter (λ n => is_palindrome n ∧ valid_digits n ∧ divisible_6 n) (Finset.range 10000)).card = 13 :=
by
  -- We define what it means to be a palindrome between 1000 and 10000
  let is_palindrome (n : ℕ) : Prop := n / 1000 = n % 10 ∧ n / 100 % 10 = n / 10 % 10
  
  -- We define a valid number between 1000 and 10000
  let valid_digits (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
  
  -- We define what it means to be divisible by 6
  let divisible_6 (n : ℕ) : Prop := n % 6 = 0

  -- Filtering the range 10000 within valid four-digit palindromes and checking for multiples of 6
  exact sorry

end number_of_palindromes_divisible_by_6_l743_743715


namespace correct_propositions_count_is_zero_l743_743301

section
variables (x : ℝ) (a : ℝ) (p q : Prop) (A : ℝ)

-- Conditions
axiom h1 : x ∈ set.Ioi 1 -- x ∈ (1, +∞)
axiom h2 : 2^x > 2       -- 2^x > 2
axiom h3 : |a| = 2       -- |a| = 2
axiom h4 : p             -- p is true
axiom h5 : ¬q            -- ¬q is true
axiom h6 : sin A < 1 / 2 -- sin A < 1/2
axiom h7 : A < π / 6     -- A < π/6

-- Theorem statement
theorem correct_propositions_count_is_zero : 
  (¬ (∀ x, x ∈ set.Ioi 1 → 2^x > 2)) = (∃ x, x ∈ set.Ioi 1 ∧ 2^x ≤ 2) ∧
  (¬ (a = 2 ↔ |a| = 2)) ∧
  (¬ (p ∧ q)) ∧
  (¬ (¬ (sin A < 1 / 2 → A < π / 6))) → 
  0 = 0 :=
by
  sorry
end

end correct_propositions_count_is_zero_l743_743301


namespace variance_of_eta_l743_743809

-- Let ξ be a random variable with a given variance of 2 (Dξ = 2).
variable (ξ : Type) [RandomVariable ξ] (Dξ : ℝ)
axiom Dξ_given : Dξ = 2

-- Define η in terms of ξ
def η := 3 * ξ + 2

-- State the theorem that asserts the variance of η is 18
theorem variance_of_eta (ξ : Type) [RandomVariable ξ] (Dξ : ℝ) (h : Dξ = 2) :
  D (η ξ) = 18 := 
by
  sorry

end variance_of_eta_l743_743809


namespace transform_sine_function_l743_743549

-- Definition of the initial function
def f (x : ℝ) : ℝ := Real.sin (2 * x)

theorem transform_sine_function :
  (∀ x : ℝ, (Real.sin (2 * (x - π / 12))) = Real.sin (x - π / 6)) :=
sorry

end transform_sine_function_l743_743549


namespace abs_inequality_solution_l743_743599

theorem abs_inequality_solution (x : ℝ) :
  |x + 2| + |x - 2| ≤ 4 ↔ -2 ≤ x ∧ x ≤ 2 :=
by
  sorry

end abs_inequality_solution_l743_743599


namespace savings_calculation_l743_743870

-- Definitions of the conditions
def machines : ℕ := 25
def ball_bearings_per_machine : ℕ := 45
def normal_price_per_bearing : ℚ := 1.25
def sale_price_per_bearing : ℚ := 0.80
def discount_first_20_machines : ℚ := 0.25
def discount_remaining_machines : ℚ := 0.35

-- The statement for the math proof problem
theorem savings_calculation :
  let total_ball_bearings := machines * ball_bearings_per_machine in
  let total_cost_without_sale := total_ball_bearings * normal_price_per_bearing in
  let total_cost_with_sale_before_bulk_discount := total_ball_bearings * sale_price_per_bearing in
  let first_20_bearings := 20 * ball_bearings_per_machine in
  let cost_for_first_20 := first_20_bearings * sale_price_per_bearing in
  let discount_for_first_20 := cost_for_first_20 * discount_first_20_machines in
  let cost_after_discount_first_20 := cost_for_first_20 - discount_for_first_20 in
  let remaining_bearings := (machines - 20) * ball_bearings_per_machine in
  let cost_for_remaining := remaining_bearings * sale_price_per_bearing in
  let discount_for_remaining := cost_for_remaining * discount_remaining_machines in
  let cost_after_discount_remaining := cost_for_remaining - discount_for_remaining in
  let total_cost_with_sale := cost_after_discount_first_20 + cost_after_discount_remaining in
  let total_savings := total_cost_without_sale - total_cost_with_sale in
  total_savings = 749.25 := sorry

end savings_calculation_l743_743870


namespace a_fifth_term_l743_743223

noncomputable def a : ℕ → ℕ
| 1     := 0
| (n+1) := 4 * (a n) + 3

theorem a_fifth_term : a 5 = 255 := by 
  sorry

end a_fifth_term_l743_743223


namespace original_price_of_cycle_l743_743284

variable (P : ℝ)

theorem original_price_of_cycle (h1 : 0.75 * P = 1050) : P = 1400 :=
sorry

end original_price_of_cycle_l743_743284


namespace calculate_value_l743_743708

noncomputable def function_example : ℚ[X] × ℚ[X] := 
  (X^2 + 5*X + 6, X^3 - 3*X^2 - 4*X)

def count_holes_and_asymptotes 
  (f : ℚ[X] × ℚ[X]) : ℕ × ℕ × ℕ × ℕ := 
  let (num, denom) := f
  -- This function hypothetically counts:
  -- (holes, vertical asymptotes, horizontal asymptotes, oblique asymptotes)
  (2, 1, 1, 0)  -- computed based on the given problem

theorem calculate_value :
  let (num, denom) := function_example
  let (p, q, r, s) := count_holes_and_asymptotes (num, denom) 
  p + 2 * q + 3 * r + 4 * s = 7 :=
by
  -- The proof is skipped 
  sorry

end calculate_value_l743_743708


namespace product_of_major_and_minor_axes_l743_743911

-- Given definitions from conditions
variables (O F A B C D : Type) 
variables (OF : ℝ) (dia_inscribed_circle_OCF : ℝ) (a b : ℝ)

-- Condition: O is the center of an ellipse
-- Point F is one focus, OF = 8
def O_center_ellipse : Prop := OF = 8

-- The diameter of the inscribed circle of triangle OCF is 4
def dia_inscribed_circle_condition : Prop := dia_inscribed_circle_OCF = 4

-- Define OA = OB = a, OC = OD = b
def major_axis_half_length : ℝ := a
def minor_axis_half_length : ℝ := b

-- Ellipse focal property a^2 - b^2 = 64
def ellipse_focal_property : Prop := a^2 - b^2 = 64

-- From the given conditions, expected result
def compute_product_AB_CD : Prop := 
  let AB := 2 * a,
      CD := 2 * b
  in AB * CD = 240

-- The main statement to be proven
theorem product_of_major_and_minor_axes 
  (h1 : O_center_ellipse)
  (h2 : dia_inscribed_circle_condition)
  (h3 : ellipse_focal_property)
  : compute_product_AB_CD :=
sorry

end product_of_major_and_minor_axes_l743_743911


namespace calc_diff_of_linear_function_l743_743883

open Function

noncomputable theory

def linear_function (g : ℝ → ℝ) : Prop := ∃ m b, ∀ x, g(x) = m * x + b

theorem calc_diff_of_linear_function (g : ℝ → ℝ) (h_linear : linear_function g) (h_condition : g 4 - g 1 = 9) :
  g 10 - g 1 = 27 :=
by
  sorry

end calc_diff_of_linear_function_l743_743883


namespace gum_candy_ratio_l743_743476

theorem gum_candy_ratio
  (g c : ℝ)  -- let g be the cost of a stick of gum and c be the cost of a candy bar.
  (hc : c = 1.5)  -- the cost of each candy bar is $1.5
  (h_total_cost : 2 * g + 3 * c = 6)  -- total cost of 2 sticks of gum and 3 candy bars is $6
  : g / c = 1 / 2 := -- the ratio of the cost of gum to candy is 1:2
sorry

end gum_candy_ratio_l743_743476


namespace largest_lambda_satisfies_ineq_l743_743739

theorem largest_lambda_satisfies_ineq :
  ∃ λ : ℝ, λ = real.sqrt 21 ∧ ∀ a b c d : ℝ, 
    0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → 
    (4 * a^2 + 3 * b^2 + 3 * c^2 + 2 * d^2 ≥ λ * a * b + 2 * b * c + c * d) := by
  sorry

end largest_lambda_satisfies_ineq_l743_743739


namespace possible_arrangements_l743_743579

theorem possible_arrangements :
  let proc := ["A", "B", "C", "D", "E"] in
  let condition1 := (proc.head = "A" ∨ proc.tail = "A") in
  let condition2 := (proc.join = "CD" ∨ proc.join = "DC") in
  (count_valid_arrangements proc) = 24 :=
by
  sorry

noncomputable def count_valid_arrangements (proc : list string) : ℕ :=
  if head proc = "A" then
    factorial (3) * 2
  else if last proc = "A" then
    factorial (3) * 2
  else
    0

end possible_arrangements_l743_743579


namespace quadratic_polynomial_condition_l743_743743

noncomputable def q (x : ℝ) : ℝ := (8/5) * x^2 - (18/5) * x - (1/5)

theorem quadratic_polynomial_condition :
  q (-1) = 5 ∧ q 2 = -1 ∧ q 4 = 11 :=
by
  unfold q
  split;
  { ring, norm_num }
  split;
  { ring, norm_num }
  { ring, norm_num }

end quadratic_polynomial_condition_l743_743743


namespace length_QR_l743_743842

-- Definitions according to conditions
def Triangle : Type := {P Q R S : Type → Prop}
def IsoscelesTriangle (T : Triangle) (P Q R : Type) : Prop := (P Q = P R)

variables {α : Type} (T : Triangle) (P Q R S : α) (M : α)
variables (PQ PR PM QR : ℝ)
variables [DecidableEq α] (hMidpoint : M = midpoint Q R)

def isosceles_triangle_lengths :=
  PQ = 5 ∧ PR = 5 ∧ PM = 4

-- Proving the length of QR
theorem length_QR (h : isosceles_triangle_lengths) (hIso : IsoscelesTriangle T P Q R) (hM : M = midpoint Q R) : QR = 6 := 
sorry

end length_QR_l743_743842


namespace x1_x2_sum_l743_743015

theorem x1_x2_sum (x1 x2 : ℝ) (h1 : 2 * x1 + 2 ^ x1 = 5) (h2 : 2 * x2 + 2 * Real.log2 (x2 - 1) = 5) :
  x1 + x2 = 7 / 2 := 
  sorry

end x1_x2_sum_l743_743015


namespace triangle_ABC_isosceles_l743_743102

-- Lean 4 definitions for the generated equivalent proof problem
noncomputable def incenter (A B C I : Type*) := sorry
noncomputable def incircle (ABC : Type*) (α : Type*) := sorry
noncomputable def circumcircle (A I C : Type*) := sorry
noncomputable def intersects (circle1 circle2 : Type*) (P Q : Type*) := sorry
noncomputable def same_side_line (P A : Type*) (line : Type*) := sorry
noncomputable def other_side_line (Q C : Type*) (line : Type*) := sorry
noncomputable def midpoint_arc (arc : Type*) := sorry
noncomputable def parallel (line1 line2 : Type*) := sorry
noncomputable def triangle_isosceles (A B C : Type*) := ∃ (ABC_is_isosceles : Prop), ABC_is_isosceles

-- Given conditions for the incenter, incircle, and parallel lines, we must prove the triangle is isosceles
theorem triangle_ABC_isosceles (A B C I α P Q M N : Type*)
  (h1 : incenter A B C I)
  (h2 : incircle ABC α)
  (h3 : circumcircle A I C)
  (h4 : intersects α (circumcircle A I C) P Q)
  (h5 : same_side_line P A (set I B))
  (h6 : other_side_line Q C (set I B))
  (h7 : midpoint_arc α AC M)
  (h8 : midpoint_arc α BC N)
  (h9 : parallel PQ AC) :
  triangle_isosceles A B C := sorry

end triangle_ABC_isosceles_l743_743102


namespace h_at_neg_one_l743_743085

noncomputable def h_of_f (p q r s : ℤ) (h a b c d : ℤ) : ℤ :=
  if (h = ((-1 + a) * (-1 + b) * (-1 + c) * (-1 + d)) ∧ f (-1) = ((1 + a) * (1 + b) * (1 + c) * (1 + d)))
    then h = -(1 + p + q + r + s)
    else h = p - q + r - s

theorem h_at_neg_one (p q r s : ℤ) (h x_f a b c d : ℤ) : h_of_f p q r s h a b c d = p - q + r - s :=
by
  sorry

end h_at_neg_one_l743_743085


namespace diagonal_of_rectangular_prism_l743_743225

theorem diagonal_of_rectangular_prism (x y z : ℝ) (d : ℝ)
  (h_surface_area : 2 * x * y + 2 * x * z + 2 * y * z = 22)
  (h_edge_length : x + y + z = 6) :
  d = Real.sqrt 14 :=
by
  sorry

end diagonal_of_rectangular_prism_l743_743225


namespace boy_lap_time_l743_743414

noncomputable def total_time_needed
  (side_lengths : List ℝ)
  (running_speeds : List ℝ)
  (obstacle_time : ℝ) : ℝ :=
(side_lengths.zip running_speeds).foldl (λ (acc : ℝ) ⟨len, speed⟩ => acc + (len / (speed / 60))) 0
+ obstacle_time

theorem boy_lap_time
  (side_lengths : List ℝ)
  (running_speeds : List ℝ)
  (obstacle_time : ℝ) :
  side_lengths = [80, 120, 140, 100, 60] →
  running_speeds = [250, 200, 300, 166.67, 266.67] →
  obstacle_time = 5 →
  total_time_needed side_lengths running_speeds obstacle_time = 7.212 := by
  intros h_lengths h_speeds h_obstacle_time
  rw [h_lengths, h_speeds, h_obstacle_time]
  sorry

end boy_lap_time_l743_743414


namespace paco_cookies_proof_l743_743529

-- Define the initial conditions
def initial_cookies : Nat := 40
def cookies_eaten : Nat := 2
def cookies_bought : Nat := 37
def free_cookies_per_bought : Nat := 2

-- Define the total number of cookies after all operations
def total_cookies (initial_cookies cookies_eaten cookies_bought free_cookies_per_bought : Nat) : Nat :=
  let remaining_cookies := initial_cookies - cookies_eaten
  let free_cookies := cookies_bought * free_cookies_per_bought
  let cookies_from_bakery := cookies_bought + free_cookies
  remaining_cookies + cookies_from_bakery

-- The target statement that needs to be proved
theorem paco_cookies_proof : total_cookies initial_cookies cookies_eaten cookies_bought free_cookies_per_bought = 149 :=
by
  sorry

end paco_cookies_proof_l743_743529


namespace count_symmetric_hexominoes_l743_743418

-- Define what it means to be a hexomino
structure Hexomino :=
(squares : Finset (Int × Int))
(conn_induction : ∀ (x y : Int × Int), x ∈ squares → y ∈ squares → x = y ∨ ∃ z ∈ squares, z ≠ x ∧ z ≠ y ∧ (dist x z = 1 ∨ dist y z = 1))

-- Define reflectional symmetry for a hexomino
def has_reflectional_symmetry (h : Hexomino) : Prop :=
∃ l : (Int × Int) × (Int × Int), reflectional_symmetry_line l ∧ reflectionally_symmetric_along h.squares l

-- There exist 15 hexominoes
constant hexominoes : Fin 15 → Hexomino

-- Statement: Exactly 8 of the given 15 hexominoes have a reflectional symmetry
theorem count_symmetric_hexominoes : (Fin 15).filter (λ i, has_reflectional_symmetry (hexominoes i)).card = 8 :=
sorry

end count_symmetric_hexominoes_l743_743418


namespace emily_spending_l743_743331

theorem emily_spending : ∀ {x : ℝ}, (x + 2 * x + 3 * x = 120) → (x = 20) :=
by
  intros x h
  sorry

end emily_spending_l743_743331


namespace number_of_ways_to_select_team_l743_743444

def calc_binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem number_of_ways_to_select_team : calc_binomial_coefficient 17 4 = 2380 := by
  sorry

end number_of_ways_to_select_team_l743_743444


namespace probability_Abby_Bridget_adj_l743_743447

-- Define the conditions of the problem
def is_seating_possible (seat : ℕ) : Prop := seat = 1
def students_in_rows (rows : ℕ) (seats_per_row : ℕ) : Bool := rows = 3 ∧ seats_per_row = 2

-- Define the type for students
inductive Student : Type
| Abby | Bridget | C | D | E | F 

open Student

-- Define the theorem statement
theorem probability_Abby_Bridget_adj :
  ∀ (students : List Student),
  length students = 6 →
  is_seating_possible 1 →
  students_in_rows 3 2 →
  ∃! (p : List (Nat × Student)),
    nat_inj p ∧ perm p students ∧ 
    ((p.head.fst =1 ∧ p.head.snd = Abby) →
     (p.tail.head.fst =2 ∧ p.tail.head.snd = Bridget)) →
  ((fav_outcomes 1 4!) / (total_outcomes 5!)) = 1 / 5 := 
sorry

end probability_Abby_Bridget_adj_l743_743447


namespace driver_license_advantage_l743_743523

def AdvantageousReasonsForEarlyLicenseObtaining 
  (eligible : ℕ → Prop)
  (effectiveInsurance : ℕ → Prop)
  (rentalCarFlexibility : ℕ → Prop)
  (employmentOpportunity : ℕ → Prop) : Prop :=
  ∀ age1 age2 : ℕ, (eligible age1 ∧ eligible age2 ∧ age1 < age2) →
  (effectiveInsurance age1 ∧ rentalCarFlexibility age1 ∧ employmentOpportunity age1) →
  effectiveInsurance age1 ∧ rentalCarFlexibility age1 ∧ employmentOpportunity age1

theorem driver_license_advantage 
  (eligible : ℕ → Prop)
  (effectiveInsurance : ℕ → Prop)
  (rentalCarFlexibility : ℕ → Prop)
  (employmentOpportunity : ℕ → Prop) :
  AdvantageousReasonsForEarlyLicenseObtaining eligible effectiveInsurance rentalCarFlexibility employmentOpportunity :=
by
  sorry

end driver_license_advantage_l743_743523


namespace distance_between_foci_l743_743735

-- Given conditions
def hyperbola_eq (x y : ℝ) : Prop := (x ^ 2) / 25 - (y ^ 2) / 4 = 1
def a_squared : ℝ := 25
def b_squared : ℝ := 4
def a : ℝ := Real.sqrt a_squared
def b : ℝ := Real.sqrt b_squared
def c_squared : ℝ := a_squared + b_squared
def c : ℝ := Real.sqrt c_squared

-- Prove the distance between the foci is 2 * Real.sqrt 29
theorem distance_between_foci : 2 * c = 2 * Real.sqrt 29 := by
  sorry

end distance_between_foci_l743_743735


namespace sum_first_75_terms_arith_seq_l743_743625

theorem sum_first_75_terms_arith_seq (a_1 d : ℕ) (n : ℕ) (h_a1 : a_1 = 3) (h_d : d = 4) (h_n : n = 75) : 
  (n * (2 * a_1 + (n - 1) * d)) / 2 = 11325 := 
by
  subst h_a1
  subst h_d
  subst h_n
  sorry

end sum_first_75_terms_arith_seq_l743_743625


namespace road_length_10_trees_10_intervals_l743_743170

theorem road_length_10_trees_10_intervals 
  (n_trees : ℕ) (n_intervals : ℕ) (tree_interval : ℕ) 
  (h_trees : n_trees = 10) (h_intervals : n_intervals = 9) (h_interval_length : tree_interval = 10) : 
  n_intervals * tree_interval = 90 := 
by 
  sorry

end road_length_10_trees_10_intervals_l743_743170


namespace isosceles_triangle_l743_743129

   open EuclideanGeometry

   -- Define the conditions of the problem in Lean
   variable {I A B C P Q M N : Point}
   variable (α : Circle) (circumcircle_AIC : Circle)

   -- Conditions extracted from the problem
   def conditions : Prop :=
   IsIncenter I △ABC ∧
   Incircle α △ABC ∧
   Circle.Diameter α P Q ∧
   Circle.Containing circumcircle_AIC (trianglePoint AIC) ∧
   SameSide P A (Line BI) ∧
   SameSide Q C (Line BI) ∧
   IsMidpointArc M ARC(α A C) ∧
   IsMidpointArc N ARC(α B C) ∧
   Parallel (Line PQ) (Line AC)

   -- Proof statement in Lean
   theorem isosceles_triangle
     (h : conditions α circumcircle_AIC) : IsIsosceles (△ABC) :=
   sorry
   
end isosceles_triangle_l743_743129


namespace loss_percentage_l743_743671

/--
A man sells a car to his friend at a certain loss percentage. The friend then sells it 
for Rs. 54000 and gains 20%. The original cost price of the car was Rs. 52941.17647058824.
Prove that the loss percentage when the man sold the car to his friend was 15%.
-/
theorem loss_percentage (CP SP_2 : ℝ) (gain_percent : ℝ) (h_CP : CP = 52941.17647058824) 
(h_SP2 : SP_2 = 54000) (h_gain : gain_percent = 20) : (CP - SP_2 / (1 + gain_percent / 100)) / CP * 100 = 15 := by
  sorry

end loss_percentage_l743_743671


namespace DrawFourLinesWithSevenPoints_l743_743054

theorem DrawFourLinesWithSevenPoints :
  ∃ (lines : Fin 4 → Line) (points : Fin 7 → Point), 
    (∀ i : Fin 4, (∃ p1 p2 p3 : Point, p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧ OnLine p1 (lines i) ∧ OnLine p2 (lines i) ∧ OnLine p3 (lines i))) ∧ 
    (∃ p4 p5 p6 p7 : Point, (p4 ≠ p5 ∧ p4 ≠ p6 ∧ p4 ≠ p7 ∧ p5 ≠ p6 ∧ p5 ≠ p7 ∧ p6 ≠ p7)) :=
sorry

end DrawFourLinesWithSevenPoints_l743_743054


namespace vector_norm_sum_squared_l743_743080

variables (a b : ℝ × ℝ)

-- Condition definitions 
def m : ℝ × ℝ := (4, 10)
def midpoint : Prop := m = (a + b) / 2
def dot_product : Prop := a.1 * b.1 + a.2 * b.2 = 10

-- Proof statement
theorem vector_norm_sum_squared : midpoint a b → dot_product a b → (a.1 ^ 2 + a.2 ^ 2 + b.1 ^ 2 + b.2 ^ 2 = 444)
:= by
  sorry

end vector_norm_sum_squared_l743_743080


namespace convert_deg_to_rad_l743_743710

theorem convert_deg_to_rad (d : ℕ) (h : d = 135) : (d : ℝ) * (Real.pi / 180) = 3 * Real.pi / 4 :=
by
  rw [h]
  norm_num

end convert_deg_to_rad_l743_743710


namespace scientists_three_languages_l743_743306

theorem scientists_three_languages (n : ℕ) (l : fin n → fin n → ℕ) (h_n : n = 17) (h_l : ∀ i j, i ≠ j → l i j < 3) : 
  ∃ (i j k : fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ l i j = l i k ∧ l i k = l j k := 
begin
  -- sorry is used here to indicate where the proof would go
  sorry,
end

end scientists_three_languages_l743_743306


namespace find_d_l743_743354

noncomputable def median (x : ℕ) : ℕ := x + 4
noncomputable def mean (x d : ℕ) : ℕ := x + (13 + d) / 5

theorem find_d (x d : ℕ) (h : mean x d = median x + 5) : d = 32 := by
  sorry

end find_d_l743_743354


namespace curve_is_line_l743_743341

def curve := {p : ℝ × ℝ | ∃ (θ : ℝ), (p.1 = (1 / (Real.sin θ + Real.cos θ)) * Real.cos θ
                                        ∧ p.2 = (1 / (Real.sin θ + Real.cos θ)) * Real.sin θ)}

-- Problem: Prove that the curve defined by the polar equation is a line.
theorem curve_is_line : ∀ (p : ℝ × ℝ), p ∈ curve → p.1 + p.2 = 1 :=
by
  -- The proof is omitted.
  sorry

end curve_is_line_l743_743341


namespace triangle_ABC_isosceles_l743_743096

-- Lean 4 definitions for the generated equivalent proof problem
noncomputable def incenter (A B C I : Type*) := sorry
noncomputable def incircle (ABC : Type*) (α : Type*) := sorry
noncomputable def circumcircle (A I C : Type*) := sorry
noncomputable def intersects (circle1 circle2 : Type*) (P Q : Type*) := sorry
noncomputable def same_side_line (P A : Type*) (line : Type*) := sorry
noncomputable def other_side_line (Q C : Type*) (line : Type*) := sorry
noncomputable def midpoint_arc (arc : Type*) := sorry
noncomputable def parallel (line1 line2 : Type*) := sorry
noncomputable def triangle_isosceles (A B C : Type*) := ∃ (ABC_is_isosceles : Prop), ABC_is_isosceles

-- Given conditions for the incenter, incircle, and parallel lines, we must prove the triangle is isosceles
theorem triangle_ABC_isosceles (A B C I α P Q M N : Type*)
  (h1 : incenter A B C I)
  (h2 : incircle ABC α)
  (h3 : circumcircle A I C)
  (h4 : intersects α (circumcircle A I C) P Q)
  (h5 : same_side_line P A (set I B))
  (h6 : other_side_line Q C (set I B))
  (h7 : midpoint_arc α AC M)
  (h8 : midpoint_arc α BC N)
  (h9 : parallel PQ AC) :
  triangle_isosceles A B C := sorry

end triangle_ABC_isosceles_l743_743096


namespace distance_from_complex_1_minus_2i_to_origin_l743_743462

def complex_distance_to_origin (z : ℂ) : ℝ :=
  complex.abs z

theorem distance_from_complex_1_minus_2i_to_origin :
  complex_distance_to_origin (1 - 2 * complex.I) = real.sqrt 5 :=
by sorry

end distance_from_complex_1_minus_2i_to_origin_l743_743462


namespace find_y_l743_743718

theorem find_y 
  (y : ℝ) 
  (h1 : (y^2 - 11 * y + 24) / (y - 3) + (2 * y^2 + 7 * y - 18) / (2 * y - 3) = -10)
  (h2 : y ≠ 3)
  (h3 : y ≠ 3 / 2) : 
  y = -4 := 
sorry

end find_y_l743_743718


namespace average_sale_l743_743283

-- Defining the monthly sales as constants
def sale_month1 : ℝ := 6435
def sale_month2 : ℝ := 6927
def sale_month3 : ℝ := 6855
def sale_month4 : ℝ := 7230
def sale_month5 : ℝ := 6562
def sale_month6 : ℝ := 7391

-- The final theorem statement to prove the average sale
theorem average_sale : (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / 6 = 6900 := 
by 
  sorry

end average_sale_l743_743283


namespace find_n_l743_743948

def f (x : ℝ) : ℝ := x ^ x

def sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 1 then log 2 (log 2 (f 2))
  else log 2 (log 2 (a (n - 1) ^ a (n - 1)))

theorem find_n :
  ∃ n : ℕ, sequence (λ n, if n = 1 then log 2 (log 2 (f 2))
                             else log 2 (log 2 (f (sequence (λ n, if n=1 then f 2 else sequence (λ n, if n=1 then f 2 else sequence (f 2)))

)) = 2059 + 2 ^ 2059 -> n = 5 :=
sorry

end find_n_l743_743948


namespace math_competition_initial_participants_l743_743448

noncomputable def initial_number_of_participants 
  (eliminated_1st_round_percentage : ℝ) 
  (eliminated_2nd_round_percentage : ℝ) 
  (remaining_3rd_round_percentage : ℝ) 
  (remaining_after_3rd_round : ℕ) : ℕ :=
  let remaining_1st_round_percentage := 1 - eliminated_1st_round_percentage
  let remaining_2nd_round_percentage := remaining_1st_round_percentage * (1 - eliminated_2nd_round_percentage)
  let remaining_final_percentage := remaining_2nd_round_percentage * remaining_3rd_round_percentage
  let initial_participants := remaining_after_3rd_round / remaining_final_percentage
  initial_participants.to_nat

theorem math_competition_initial_participants : 
  initial_number_of_participants 0.6 0.5 0.25 15 = 300 :=
by
  sorry

end math_competition_initial_participants_l743_743448


namespace angle_ratio_bisectors_l743_743045

theorem angle_ratio_bisectors (ABC BP BQ BM P Q M : Type)
  (bisect_BP : ∀ (A B C : Type), BP = λ (A B P : Type), ∀ (x : Type), x ∈ BP → x ∣ (\angle A B P))
  (bisect_BQ : ∀ (A B C : Type), BQ = λ (A B Q : Type), ∀ (x : Type), x ∈ BQ → x ∣ (\angle A B Q))
  (bisect_BM : ∀ (B P Q M : Type), BM = λ (B M : Type), ∀ (x : Type), x ∈ BM → x ∣ (\angle P B Q)) :
  (\angle M B Q) / (\angle A B Q) = 1 / 4 := 
sorry

end angle_ratio_bisectors_l743_743045


namespace min_meet_time_l743_743964

theorem min_meet_time : (6 / 26) < ((2 * Real.pi - 2) / 21) := 
begin
  sorry
end

end min_meet_time_l743_743964


namespace part_I_part_II_l743_743854

open Nat

noncomputable def a : ℕ → ℤ
| 0       => 1
| 1       => 6
| 2       => 20
| (n + 3) => 2 * a (n + 2) + (4 * 2^n) -- Based on the geometric sequence condition

def S (n : ℕ) : ℤ :=
  (finset.range n).sum (λ k, a k)

theorem part_I (n : ℕ) : ∃ d : ℤ, ∀ n : ℕ, (a n) / (2^n) = ((a 0) / (2^0)) + (↑n - 1) * d :=
sorry

theorem part_II (n : ℕ) : S n = (2 * n - 3) * 2^n + 3 :=
sorry

end part_I_part_II_l743_743854


namespace reciprocal_of_repeating_decimal_l743_743986

theorem reciprocal_of_repeating_decimal :
  (1 / (0.33333333 : ℚ)) = 3 := by
  sorry

end reciprocal_of_repeating_decimal_l743_743986


namespace harmonic_series_inequality_l743_743147

theorem harmonic_series_inequality 
  (n : ℕ) (hpos : 0 < n)
  (a b : ℝ) (apos : 0 < a) (bpos : 0 < b) :
  (∑ k in Finset.range n, 1 / (a + (k + 1) * b)) < n / Real.sqrt (a * (a + n * b)) :=
by
  sorry

end harmonic_series_inequality_l743_743147


namespace remainder_of_sum_of_first_five_primes_divided_by_sixth_prime_l743_743239

def firstFivePrimes : List ℕ := [2, 3, 5, 7, 11]
def sixthPrime : ℕ := 13
def sumFirstFivePrimes : ℕ := firstFivePrimes.sum
def remainderWhenDividedBySixthPrime (sum : ℕ) (prime : ℕ) : ℕ := sum % prime

theorem remainder_of_sum_of_first_five_primes_divided_by_sixth_prime :
  remainderWhenDividedBySixthPrime sumFirstFivePrimes sixthPrime = 2 :=
by
  calc remainderWhenDividedBySixthPrime sumFirstFivePrimes sixthPrime
      = 28 % 13 : by rw [sumFirstFivePrimes, List.sum_cons, List.sum_cons, List.sum_cons, List.sum_cons, List.sum_nil]
  ... = 2 : by norm_num

end remainder_of_sum_of_first_five_primes_divided_by_sixth_prime_l743_743239


namespace sum_of_solutions_eq_neg2_l743_743925

theorem sum_of_solutions_eq_neg2 : 
  (∑ x in {x | 3^(x^2 + 4*x + 4) = 9^(x + 2)}.to_finset, x) = -2 := 
begin
  sorry
end

end sum_of_solutions_eq_neg2_l743_743925


namespace concyclic_points_l743_743455

open EuclideanGeometry

noncomputable def isosceles_triangle (A B C : Point) : Prop :=
A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ dist A B = dist A C

noncomputable def midpoint (A B : Point) : Point :=
((A + B) / 2 : Point)

theorem concyclic_points
    (A B C P X Y M : Point)
    (h_isos : isosceles_triangle A B C)
    (h_PB_PC : dist P B < dist P C)
    (h_parallel : parallel (line_through A P) (line_through B C))
    (h_MP : M = midpoint B C)
    (h_ang_eq : angle P X M = angle P Y M) : 
    concyclic4 A P X Y :=
by
  sorry

end concyclic_points_l743_743455


namespace total_production_l743_743003

theorem total_production (S : ℝ) 
  (h1 : 4 * S = 4400) : 
  4400 + S = 5500 := 
by
  sorry

end total_production_l743_743003


namespace sum_of_valid_m_values_l743_743827

-- Variables and assumptions
variable (m x : ℝ)

-- Conditions from the given problem
def inequality_system (m x : ℝ) : Prop :=
  (x - 4) / 3 < x - 4 ∧ (m - x) / 5 < 0

def solution_set_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, inequality_system m x → x > 4

def fractional_equation (m x : ℝ) : Prop :=
  6 / (x - 3) + 1 = (m * x - 3) / (x - 3)

-- Lean statement to prove the sum of integers satisfying the conditions
theorem sum_of_valid_m_values : 
  (∀ m : ℝ, solution_set_condition m ∧ 
            (∃ x : ℝ, x > 0 ∧ x ≠ 3 ∧ fractional_equation m x) →
            (∃ (k : ℕ), k = 2 ∨ k = 4) → 
            2 + 4 = 6) :=
sorry

end sum_of_valid_m_values_l743_743827


namespace rectangle_area_l743_743255

theorem rectangle_area (P L W : ℝ) (hP : P = 2 * (L + W)) (hRatio : L / W = 5 / 2) (hP_val : P = 280) : 
  L * W = 4000 :=
by 
  sorry

end rectangle_area_l743_743255


namespace angle_inclusion_l743_743819

-- Defining the sets based on the given conditions
def M : Set ℝ := { x | 0 < x ∧ x ≤ 90 }
def N : Set ℝ := { x | 0 < x ∧ x < 90 }
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 90 }

-- The proof statement
theorem angle_inclusion : N ⊆ M ∧ M ⊆ P :=
by
  sorry

end angle_inclusion_l743_743819


namespace sector_angle_measure_l743_743379

theorem sector_angle_measure
  (r l : ℝ)
  (h1 : 2 * r + l = 4)
  (h2 : (1 / 2) * l * r = 1) :
  l / r = 2 :=
sorry

end sector_angle_measure_l743_743379


namespace coefficient_x14_in_quotient_l743_743242

noncomputable def polynomial_quotient_coefficient:
  polynomial ℤ := x^4 + x^3 + 2*x^2 + x + 1

theorem coefficient_x14_in_quotient: 
  coeff ((x^1951 - 1) / polynomial_quotient_coefficient) 14 = -1 := by
  sorry

end coefficient_x14_in_quotient_l743_743242


namespace last_two_digits_of_a2000_l743_743709

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = a n + a n ^ 2

theorem last_two_digits_of_a2000 (a : ℕ → ℕ) (h : sequence a) : 
  (a 2000) % 100 = 92 := 
by {
  sorry
}

end last_two_digits_of_a2000_l743_743709


namespace group_is_isomorphic_to_symmetric_subgroup_l743_743490

variable {G : Type*} [Group G]
noncomputable def symmetric_group (α : Type*) := equiv.perm α

theorem group_is_isomorphic_to_symmetric_subgroup (G : Type*) [Group G] :
  ∃ (H : set (symmetric_group G)), is_subgroup H ∧ ∃ φ : G → H, is_group_hom φ ∧ function.injective φ :=
sorry

end group_is_isomorphic_to_symmetric_subgroup_l743_743490


namespace conclusion_friendly_not_large_l743_743162

variable {Snake : Type}
variable (isLarge isFriendly canClimb canSwim : Snake → Prop)
variable (marysSnakes : Finset Snake)
variable (h1 : marysSnakes.card = 16)
variable (h2 : (marysSnakes.filter isLarge).card = 6)
variable (h3 : (marysSnakes.filter isFriendly).card = 7)
variable (h4 : ∀ s, isFriendly s → canClimb s)
variable (h5 : ∀ s, isLarge s → ¬ canSwim s)
variable (h6 : ∀ s, ¬ canSwim s → ¬ canClimb s)

theorem conclusion_friendly_not_large :
  ∀ s, isFriendly s → ¬ isLarge s :=
by
  sorry

end conclusion_friendly_not_large_l743_743162


namespace x_minus_y_eq_2_l743_743435

theorem x_minus_y_eq_2 (x y : ℝ) (h1 : 2 * x + 3 * y = 9) (h2 : 3 * x + 2 * y = 11) : x - y = 2 :=
sorry

end x_minus_y_eq_2_l743_743435


namespace smallest_prime_divisors_42_l743_743349

theorem smallest_prime_divisors_42 :
  ∃ p : ℕ, Prime p ∧ p > 1 ∧ (∃ d : ℕ, d = 42 ∧ d = (p^3 + 2 * p^2 + p).numDivisors) ∧ 
  (∀ q : ℕ, Prime q ∧ q < p → (q^3 + 2 * q^2 + q).numDivisors ≠ 42) → 
  p = 23 := 
by
  sorry

end smallest_prime_divisors_42_l743_743349


namespace binomial_sum_unique_solution_l743_743996

-- Define the conditions and the main statement in Lean 4
theorem binomial_sum_unique_solution :
  ∀ (n : ℕ), (nat.choose 15 n) + (nat.choose 15 7) = nat.choose 16 8 → n = 8 :=
by
  sorry

end binomial_sum_unique_solution_l743_743996


namespace wheelbarrow_ratio_l743_743281

noncomputable theory

def duck_price := 10
def chicken_price := 8
def chickens_sold := 5
def ducks_sold := 2
def additional_earnings := 60

def total_earnings := chickens_sold * chicken_price + ducks_sold * duck_price
def wheelbarrow_cost := total_earnings / 2

def multiple_paid (x : ℝ) := wheelbarrow_cost * x = wheelbarrow_cost + additional_earnings

theorem wheelbarrow_ratio :
  ∃ x, multiple_paid x ∧ x = 3 :=
sorry

end wheelbarrow_ratio_l743_743281


namespace R_at_3_l743_743078

-- Definitions
def R (x: ℝ) := ∑ (i : ℕ) in finset.range (m + 1), b_i * x^i

variables {m : ℕ} (b : fin m.succ → ℤ) (x : ℝ)

-- Conditions
axiom R_def (x : ℝ) : R x = ∑ (i : ℕ) in finset.range (m + 1), b i * x^i
axiom coeff_bounds (i : ℕ) (h : i ≤ m) : 0 ≤ b i ∧ b i < 4
axiom R_at_sqrt5 : R (sqrt 5) = 30 + 27 * sqrt 5

-- Statement to be proved
theorem R_at_3 : R 3 = 15 :=
by
  -- Use the proof steps to be filled here
  sorry

end R_at_3_l743_743078


namespace max_intersections_square_decagon_l743_743156

/-- Let P_1 be a square and P_2 be a decagon.
    P_1 is inscribed within P_2 and four vertices of P_2 coincide with the vertices of P_1.
    The maximum possible number of intersections between the edges of P_1 and P_2 is 8. -/
theorem max_intersections_square_decagon
  (P1 P2 : Type)
  [has_sides 4 P1] -- P1 is a square
  [has_sides 10 P2] -- P2 is a decagon
  (inscribed : is_inscribed P1 P2)
  (shared_vertices : ∃ (f : fin 4 → P1) (g : fin 4 → P2), ∀ i, f i = g i) :
  max_intersections P1 P2 = 8 := 
sorry

end max_intersections_square_decagon_l743_743156


namespace lucky_tickets_sum_divisible_by_13_l743_743271

def six_digit_number (A : ℕ) : Prop :=
  100000 ≤ A ∧ A ≤ 999999

def sum_of_digits (A : ℕ) : ℕ :=
  let d1 := A / 100000 % 10 in
  let d2 := A / 10000 % 10 in
  let d3 := A / 1000 % 10 in
  let d4 := A / 100 % 10 in
  let d5 := A / 10 % 10 in
  let d6 := A % 10 in
  d1 + d2 + d3 + d4 + d5 + d6

def is_lucky_ticket (A : ℕ) : Prop :=
  let d1 := A / 100000 % 10 in
  let d2 := A / 10000 % 10 in
  let d3 := A / 1000 % 10 in
  let d4 := A / 100 % 10 in
  let d5 := A / 10 % 10 in
  let d6 := A % 10 in
  d1 + d2 + d3 = d4 + d5 + d6

theorem lucky_tickets_sum_divisible_by_13 :
  ∀ A : ℕ, 
  six_digit_number A → is_lucky_ticket A → 
  ∃ B : ℕ, A + B = 999999 ∧ is_lucky_ticket B → 13 ∣ (A + B) :=
by sorry

end lucky_tickets_sum_divisible_by_13_l743_743271


namespace total_wings_of_birds_l743_743065

def total_money_from_grandparents (gift : ℕ) (grandparents : ℕ) : ℕ := gift * grandparents

def number_of_birds (total_money : ℕ) (bird_cost : ℕ) : ℕ := total_money / bird_cost

def total_wings (birds : ℕ) (wings_per_bird : ℕ) : ℕ := birds * wings_per_bird

theorem total_wings_of_birds : 
  ∀ (gift amount : ℕ) (grandparents bird_cost wings_per_bird : ℕ),
  gift = 50 → 
  amount = 200 →
  grandparents = 4 → 
  bird_cost = 20 → 
  wings_per_bird = 2 → 
  total_wings (number_of_birds amount bird_cost) wings_per_bird = 20 :=
by {
  intros gift amount grandparents bird_cost wings_per_bird gift_eq amount_eq grandparents_eq bird_cost_eq wings_per_bird_eq,
  rw [gift_eq, amount_eq, grandparents_eq, bird_cost_eq, wings_per_bird_eq],
  simp [total_wings, total_money_from_grandparents, number_of_birds],
  sorry
}

end total_wings_of_birds_l743_743065


namespace parabola_point_distance_l743_743575

open Real

noncomputable def parabola_coords (y z: ℝ) : Prop :=
  y^2 = 12 * z

noncomputable def distance (x1 y1 x2 y2: ℝ) : ℝ :=
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem parabola_point_distance (x y: ℝ) :
  parabola_coords y x ∧ distance x y 3 0 = 9 ↔ ( x = 6 ∧ (y = 6 * sqrt 2 ∨ y = -6 * sqrt 2)) :=
by
  sorry

end parabola_point_distance_l743_743575


namespace functional_equation_solution_l743_743638

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * f x * y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) :=
by
  intro h
  sorry

end functional_equation_solution_l743_743638


namespace ABC_is_isosceles_l743_743125

open Mobius

variables
  (ABC : Triangle)
  (I : Point)
  (α P Q M N : Point)
  (A B C Z : Point)
  [Incenter I ABC]
  [Incircle α ABC]
  [Circumcircle (Triangle.mk A I C)]
  [PQ_parallel_AC : is_parallel (Segment.mk P Q) (Segment.mk A C)]
  [Midpoint M (Arc.mk A C α)]
  [Midpoint N (Arc.mk B C α)]
  [center_Z : Center Z (Circumcircle (Triangle.mk A I C))]
  [Chord_α_PQ : common_chord α (Circumcircle (Triangle.mk A I C)) (Segment.mk P Q)]
  [P_A_same_BI : same_side P A (Line.mk B I)]
  [Q_C_other_BI : same_side Q C (Line.mk B I)]

theorem ABC_is_isosceles
  (h_parallel : PQ_parallel_AC) : 
  Isosceles ABC := 
sorry

end ABC_is_isosceles_l743_743125


namespace keychain_arrangement_l743_743453

theorem keychain_arrangement : 
  (count.arrangements {k1, k2, k3, house_key, car_key} 
  where (house_key next_to car_key) removing (rotations and reflections)) = 6 := 
sorry

end keychain_arrangement_l743_743453


namespace find_number_l743_743650

theorem find_number (x : ℝ) (h : 0.85 * x = (4 / 5) * 25 + 14) : x = 40 :=
sorry

end find_number_l743_743650


namespace circle_radius_l743_743591

theorem circle_radius (θ : ℝ) (ρ : ℝ) (h : ρ = 2 * real.cos θ) : 
  ∃ x y : ℝ, x^2 + y^2 = 2 * x ∧ (x - 1)^2 + y^2 = 1 :=
by { sorry }

end circle_radius_l743_743591


namespace blood_expiration_date_l743_743297

constant expiry_seconds : ℕ := 362880
constant seconds_in_a_day : ℕ := 86400
constant donation_time : (ℕ × ℕ × ℕ × ℕ) := (2023, 1, 15, 8) -- Year, Month, Day, Hour (assuming it is 2023 for context)

theorem blood_expiration_date :
  let (year, month, day, hour) := donation_time in
  let days_to_expiry := expiry_seconds / seconds_in_a_day in
  let remaining_seconds := expiry_seconds % seconds_in_a_day in
  let expiration_days := day + days_to_expiry in 
  let expiration_hour := hour + remaining_seconds / 3600 in
  let expiration_minute := (remaining_seconds % 3600) / 60 in
  (year, month, expiration_days, expiration_hour, expiration_minute) = (2023, 1, 19, 4, 48) := 
by sorry

end blood_expiration_date_l743_743297


namespace negation_of_proposition_l743_743216

open Classical

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by
  sorry

end negation_of_proposition_l743_743216


namespace sequence_limit_l743_743682

theorem sequence_limit : 
  tendsto (λ n, (finset.sum (finset.range n) (λ k, 2 * (1 / 3) ^ k + (1 / 3) ^ (k + 1) * sqrt 3))) at_top (𝓝 (3 + sqrt 3 / 2)) :=
by
  sorry

end sequence_limit_l743_743682


namespace sufficient_and_necessary_condition_l743_743084

def f (x : ℝ) : ℝ := x^3 + log (x + 1) / log 2

theorem sufficient_and_necessary_condition {a b : ℝ} :
  (a + b ≥ 0) ↔ (f a + f b ≥ 0) :=
sorry

end sufficient_and_necessary_condition_l743_743084


namespace truncated_cone_minimum_length_and_distance_l743_743367

-- Defining the given conditions
def upperBaseRadius := 5 -- cm
def lowerBaseRadius := 10 -- cm
def slantHeight := 20 -- cm
def midpointStringLength := slantHeight / 2 -- cm

-- The goal is to prove the minimum length of the string and the minimum distance
theorem truncated_cone_minimum_length_and_distance :
  let PA1 := (lowerBaseRadius - upperBaseRadius) in
  let θ := (PA1 / slantHeight) * 360 in
  let PM := PA1 + midpointStringLength in
  let lateralHeight := √(PM^2 + slantHeight^2) in
  let minStringLength := lateralHeight in
  let PC := (PM * slantHeight) / lateralHeight in
  minStringLength = 50 ∧ PC - PA1 = 4 := 
by
  -- Placeholder, proof will be filled in here
  sorry

end truncated_cone_minimum_length_and_distance_l743_743367


namespace bug_returns_or_neighbors_l743_743527

def cell := (Int, Int)
def neighboring (c1 c2 : cell) : Prop :=
  (abs (c1.1 - c2.1) ≤ 1) ∧ (abs (c1.2 - c2.2) ≤ 1) ∧ (c1 ≠ c2)

def central_cross (n : Int) : set cell :=
  { (i, j) | (i = n / 2) ∨ (j = n / 2) }

def figure_F_central_cross : set cell := central_cross 99

def valid_transition (initial final : set cell) : Prop :=
  ∀ (c1 c2 : cell), (c1 ∈ initial ∧ c2 ∈ initial ∧ neighboring c1 c2) →
  (final.contains c1 → (neighboring (c1, c2) ∨ c1 = c2))

theorem bug_returns_or_neighbors :
  ∃ (c : cell), c ∈ figure_F_central_cross ∧ 
  ((valid_transition figure_F_central_cross figure_F_central_cross) → 
  (valid_transition {c} figure_F_central_cross)) :=
by
  sorry

end bug_returns_or_neighbors_l743_743527


namespace min_increase_velocity_correct_l743_743612

noncomputable def min_increase_velocity (V_A V_B V_C V_D : ℝ) (dist_AC dist_CD : ℝ) : ℝ :=
  let t_AC := dist_AC / (V_A + V_C)
  let t_AB := 30 / (V_A - V_B)
  let t_AD := (dist_AC + dist_CD) / (V_A + V_D)
  let new_velocity_A := (dist_AC + dist_CD) / t_AC - V_D
  new_velocity_A - V_A

theorem min_increase_velocity_correct :
  min_increase_velocity 80 50 70 60 300 400 = 210 :=
by
  sorry

end min_increase_velocity_correct_l743_743612


namespace smallest_positive_m_l743_743669

-- Define the angles θ1 and θ2
def θ1 := Real.pi / 70
def θ2 := Real.pi / 54

-- Define the initial line slope
def initial_slope := 19 / 92

-- Define the transformation R function
def R (θ1 θ2 θ : ℝ) : ℝ :=
  2 * θ2 - 2 * θ1 + θ

-- Define the condition for m such that the line is invariant under the transformation
def invariant (m : ℕ) : Prop :=
  ∃ n : ℕ, m * (16 / 945) = 2 * n

-- Prove that the smallest positive integer m is 945
theorem smallest_positive_m : ∃ (m : ℕ), invariant m ∧ m = 945 :=
  sorry

end smallest_positive_m_l743_743669


namespace ratio_of_areas_l743_743492

theorem ratio_of_areas (A B C D E F : Type*) 
[metric_space A] [has_dist A] [metric_space B] [has_dist B]
[metric_space C] [has_dist C] [metric_space D] [has_dist D]
[metric_space E] [has_dist E] [metric_space F] [has_dist F]
(a b c d e f : A)
(hab : dist a b = 130)
(hac : dist a c = 130)
(had : dist a d = 45)
(hcf : dist c f = 90)
(hd_on_ab : ∃λ (t : ℝ), t ∈ Icc 0 1 ∧ d = (1 - t) • a + t • b)
(hf_on_ac : ∃λ (s : ℝ), s ∈ Icc 0 1 ∧ f = (1 - s) • a + s • c)
(hdf_bc : ∃λ (u v : ℝ), u + v = 1 ∧ e = u • ((1 - t) • a + t • b) + v • c) : 
  (area (triangle a d f) / area (triangle b e c)) = 405 / 1445 :=
sorry

end ratio_of_areas_l743_743492


namespace net_effect_on_sales_l743_743021

theorem net_effect_on_sales
  (P : ℝ) (original_sales_volume : ℝ)
  (price_reduction_percent : ℝ := 0.22)
  (bulk_purchase_discount_percent : ℝ := 0.05)
  (loyalty_customer_discount_percent : ℝ := 0.10)
  (increase_in_sales_volume_percent : ℝ := 0.86)
  (variable_cost_percent : ℝ := 0.10) :
  let new_price_per_tv := (1 - price_reduction_percent) * P,
      total_additional_discounts := bulk_purchase_discount_percent * new_price_per_tv + loyalty_customer_discount_percent * new_price_per_tv,
      price_after_all_discounts := new_price_per_tv - total_additional_discounts,
      new_sales_volume := 1 + increase_in_sales_volume_percent,
      variable_cost_per_tv := variable_cost_percent * price_after_all_discounts,
      net_price_after_variable_costs := price_after_all_discounts - variable_cost_per_tv,
      original_total_sale_value := P * original_sales_volume,
      new_total_sale_value := net_price_after_variable_costs * new_sales_volume * original_sales_volume in
  new_total_sale_value / original_total_sale_value - 1 = 0.109862 :=
by
  sorry  -- proof omitted

end net_effect_on_sales_l743_743021


namespace find_first_term_of_arithmetic_sequence_l743_743388

theorem find_first_term_of_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_a3 : a 3 = 2)
  (h_d : d = -1/2) : a 1 = 3 :=
sorry

end find_first_term_of_arithmetic_sequence_l743_743388


namespace expectation_of_X_l743_743951

noncomputable def random_variable_X (p : ℝ) : ℝ :=
  1 * p + 2 * (1 - 2 * p) + 3 * p

theorem expectation_of_X (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1/2) : random_variable_X p = 2 := by
  have h1 : 1 * p + 2 * (1 - 2 * p) + 3 * p = 1 * p + 2 - 4 * p + 3 * p := by sorry
  have h2 : 1 * p + 2 - 4 * p + 3 * p = 2 := by sorry
  show random_variable_X p = 2 from by simp [random_variable_X, h1, h2]
  sorry

end expectation_of_X_l743_743951


namespace isosceles_triangle_of_parallel_PQ_AC_l743_743141

variables {α β γ : Type}
variables {A B C I M N P Q : α} {circleIncircle circCircumcircleAIC : set α}
variables {lineBI linePQ lineAC : set α}

-- Given conditions from the problem
def incenter_of_triangle (I A B C : α) : Prop := sorry
def incircle (circle α : set α) (A B C I : α) : Prop := sorry
def circumcircle (circle circumcircleAIC : set α) (A I C : α) : Prop := sorry
def midpoint_of_arc (M N : α) (circle α : set α) (A B C : α) : Prop := sorry
def parallel (linePQ lineAC : set α) : Prop := sorry
def lies_on_same_side_of_line (P A : α) (line lineBI : set α) : Prop := sorry
def lies_on_opposite_sides_of_line (Q C : α) (line lineBI : set α) : Prop := sorry

theorem isosceles_triangle_of_parallel_PQ_AC :
  incenter_of_triangle I A B C ∧
  incircle circleIncircle A B C I ∧
  circumcircle circCircumcircleAIC A I C ∧
  midpoint_of_arc M circleIncircle A C ∧
  midpoint_of_arc N circleIncircle B C ∧
  lies_on_same_side_of_line P A lineBI ∧
  lies_on_opposite_sides_of_line Q C lineBI ∧
  parallel linePQ lineAC →
  (triangle_isosceles A B C) := sorry

end isosceles_triangle_of_parallel_PQ_AC_l743_743141


namespace find_radius_of_circle_S_l743_743855

theorem find_radius_of_circle_S
  (A B C : Point)
  (h1 : dist A B = 110)
  (h2 : dist A C = 110)
  (h3 : dist B C = 60)
  (R : Circle)
  (h4 : R.radius = 18)
  (h5 : R.isTangentTo AC)
  (h6 : R.isTangentTo BC)
  (S : Circle)
  (h7 : S.isExternallyTangentTo R)
  (h8 : S.isTangentTo AB)
  (h9 : S.isTangentTo BC)
  (h10 : ∀ P : Point, ¬(S.contains P ∧ P ∉ triangle A B C)) :
  S.radius = 21 :=
by
  sorry

end find_radius_of_circle_S_l743_743855


namespace regression_slope_change_l743_743764

theorem regression_slope_change (x : ℝ) : 
  let y_hat := 6 - 6.5 * x in
  let delta_y_hat := (6 - 6.5 * (x + 1)) - y_hat in
  delta_y_hat = -6.5 :=
by
  sorry

end regression_slope_change_l743_743764


namespace souvenirs_purchasing_plans_count_l743_743294

theorem souvenirs_purchasing_plans_count :
  (∃ x y z : ℤ, 
    (x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1) ∧ 
    (x + 2 * y + 4 * z = 101)) ↔ 600 :=
sorry

end souvenirs_purchasing_plans_count_l743_743294


namespace solve_tangent_equation_l743_743259

theorem solve_tangent_equation (x : ℝ) (k : ℤ)
  (h1 : cos (3 * x) ≠ 0) (h2 : cos (5 * x) ≠ 0) :
  5.44 * tan (5 * x) - 2 * tan (3 * x) = tan (3 * x) ^ 2 * tan (5 * x) → 
  x = int.cast k * Real.pi := 
sorry

end solve_tangent_equation_l743_743259


namespace angle_FAM_l743_743371

-- Define the similarity of triangles
axiom triangle_similarity (X Y Z D B A E C F : Type*) :
  Similar (Triangle X Z Y) (Triangle D B A) ∧
  Similar (Triangle D B A) (Triangle E A C) ∧
  Similar (Triangle E A C) (Triangle F E D)

-- Define the midpoint
axiom midpoint_condition (B C M : Type*) :
  Midpoint M B C

-- Define the problem statement in Lean
theorem angle_FAM {X Y Z D B A E C F M : Type*}
  (h1 : Similar (Triangle X Z Y) (Triangle D B A))
  (h2 : Similar (Triangle D B A) (Triangle E A C))
  (h3 : Similar (Triangle E A C) (Triangle F E D))
  (hM : Midpoint M B C) :
  ∠FAM = |∠XYZ - ∠XZY| ∧
  AF / AM = (2 * XY * XZ) / (YZ ^ 2) := 
sorry

end angle_FAM_l743_743371


namespace sum_A_C_l743_743720

-- Define that A, B, C, D are in the set and different
variables (A B C D : ℕ)
variables (h1 : A ∈ {1, 2, 3, 4, 5})
variables (h2 : B ∈ {1, 2, 3, 4, 5})
variables (h3 : C ∈ {1, 2, 3, 4, 5})
variables (h4 : D ∈ {1, 2, 3, 4, 5})
variables (h_different : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)

-- Define the given condition on the fractions
variable (h_fractions : (A : ℚ) / B + (C : ℚ) / D = 3)

-- Prove that A + C = 7
theorem sum_A_C : A + C = 7 := 
by
  sorry

end sum_A_C_l743_743720


namespace find_x_of_equation_l743_743745

theorem find_x_of_equation :
  ∃ x : ℕ, 16^5 + 16^5 + 16^5 = 4^x ∧ x = 20 :=
by 
  sorry

end find_x_of_equation_l743_743745


namespace isosceles_triangle_l743_743130

   open EuclideanGeometry

   -- Define the conditions of the problem in Lean
   variable {I A B C P Q M N : Point}
   variable (α : Circle) (circumcircle_AIC : Circle)

   -- Conditions extracted from the problem
   def conditions : Prop :=
   IsIncenter I △ABC ∧
   Incircle α △ABC ∧
   Circle.Diameter α P Q ∧
   Circle.Containing circumcircle_AIC (trianglePoint AIC) ∧
   SameSide P A (Line BI) ∧
   SameSide Q C (Line BI) ∧
   IsMidpointArc M ARC(α A C) ∧
   IsMidpointArc N ARC(α B C) ∧
   Parallel (Line PQ) (Line AC)

   -- Proof statement in Lean
   theorem isosceles_triangle
     (h : conditions α circumcircle_AIC) : IsIsosceles (△ABC) :=
   sorry
   
end isosceles_triangle_l743_743130


namespace weekly_goal_l743_743634

theorem weekly_goal (a : ℕ) (d : ℕ) (n : ℕ) (h1 : a = 20) (h2 : d = 5) (h3 : n = 5) :
  ∑ i in finset.range n, a + i * d = 150 :=
by
  sorry

end weekly_goal_l743_743634


namespace sqrt_sq_eq_l743_743244

theorem sqrt_sq_eq (x : ℝ) : (Real.sqrt x) ^ 2 = x := by
  sorry

end sqrt_sq_eq_l743_743244


namespace decreasing_interval_l743_743397

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 4

theorem decreasing_interval : (¬ ∃ x ∈ (-1, 1), has_deriv_at f (3 * x^2 - 3) x ∧ 3 * x^2 - 3 < 0) := 
sorry

end decreasing_interval_l743_743397


namespace fossil_age_count_l743_743662

-- Definitions based on conditions
def digits : List ℕ := [1, 1, 1, 4, 8, 9]
def even_digits : List ℕ := [4, 8]

-- Lean statement
theorem fossil_age_count : 
  ∃ n : ℕ, n = 40 ∧
  (∀ age : List ℕ, 
    ((∃ e ∈ even_digits, List.last age e ∈ even_digits) ∧
     (∀ d ∈ age, d ∈ digits ∧ age.length = 6)) → 
     n = 40) :=
sorry

end fossil_age_count_l743_743662


namespace operation_evaluation_l743_743711

def my_operation (x y : Int) : Int :=
  x * (y + 1) + x * y

theorem operation_evaluation :
  my_operation (-3) (-4) = 21 := by
  sorry

end operation_evaluation_l743_743711


namespace beavers_increased_by_20_l743_743307

/-- Problem statement:
Aubree saw 20 beavers and 40 chipmunks by a tree when going to school. While coming back from school, she realized the number of beavers had changed and the number of chipmunks had decreased by 10. She saw a total of 130 animals that day. What happened to the number of beavers when she came back from school?
-/
theorem beavers_increased_by_20
  (initial_beavers : ℕ)
  (initial_chipmunks : ℕ)
  (beavers_back : ℕ)
  (chipmunks_back : ℕ)
  (total_animals : ℕ)
  (decreased_chipmunks : ℕ) :
  (initial_beavers = 20) →
  (initial_chipmunks = 40) →
  (decreased_chipmunks = 10) →
  (total_animals = 130) →
  (chipmunks_back = initial_chipmunks - decreased_chipmunks) →
  (beavers_back + chipmunks_back = total_animals - (initial_beavers + initial_chipmunks)) →
  (beavers_back - initial_beavers = 20) :=
begin
  intros h1 h2 h3 h4 h5 h6,
  -- Use definition of initial_beavers, initial_chipmunks, decreased_chipmunks, and total_animals
  have eq1 : initial_beavers = 20 := h1,
  have eq2 : initial_chipmunks = 40 := h2,
  have eq3 : decreased_chipmunks = 10 := h3,
  have eq4 : total_animals = 130 := h4,
  
  -- Use definition of beavers_back and chipmunks_back
  have eq5 : chipmunks_back = initial_chipmunks - decreased_chipmunks := h5,
  have eq6 : beavers_back + chipmunks_back = total_animals - (initial_beavers + initial_chipmunks) := h6,

  -- Skip the proof with sorry to state only the theorem without proof steps
  sorry
end

end beavers_increased_by_20_l743_743307


namespace min_value_of_expression_min_value_achieved_l743_743344

noncomputable def f (x : ℝ) : ℝ :=
  (15 - x) * (8 - x) * (15 + x) * (8 + x)

theorem min_value_of_expression : ∀ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) ≥ -6452.25 :=
by sorry

theorem min_value_achieved : ∃ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) = -6452.25 :=
by sorry

end min_value_of_expression_min_value_achieved_l743_743344


namespace find_g_at_7_l743_743508

noncomputable def g (x : ℝ) (a b c : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 4

theorem find_g_at_7 (a b c : ℝ) (h_symm : ∀ x : ℝ, g x a b c + g (-x) a b c = -8) (h_neg7: g (-7) a b c = 12) :
  g 7 a b c = -20 :=
by
  sorry

end find_g_at_7_l743_743508


namespace checkerboard_problem_l743_743654

-- Definitions corresponding to conditions
def checkerboard_size := 10

def is_alternating (i j : ℕ) : bool :=
  (i + j) % 2 = 0

def num_squares_with_sides_on_grid_lines_containing_at_least_6_black_squares (n : ℕ) : ℕ :=
  if n >= 4 then (11 - n) * (11 - n) else 0

-- Problem statement
theorem checkerboard_problem : 
  let count_squares : ℕ := (∑ n in finset.range checkerboard_size.succ, num_squares_with_sides_on_grid_lines_containing_at_least_6_black_squares n)
  in count_squares = 140 :=
begin
  sorry
end

end checkerboard_problem_l743_743654


namespace average_age_of_population_l743_743442

theorem average_age_of_population (k : ℕ) (h_k_pos : 0 < k) : 
  let men := 7 * k in
  let women := 8 * k in
  let total_age_of_men := 36 * men in
  let total_age_of_women := 30 * women in
  let total_population := men + women in
  let total_age := total_age_of_men + total_age_of_women in
  (total_age / total_population) = (164 / 5) :=
sorry

end average_age_of_population_l743_743442


namespace sum_of_intersections_l743_743355

theorem sum_of_intersections (M : Finset α) (n : ℕ) (h : M.card = n) :
  ∑ (A B : Finset α), (A ∩ B).card = n * 4^(n - 1) :=
by sorry

end sum_of_intersections_l743_743355


namespace math_problem_l743_743795

def f (a x : ℝ) := a * Math.sin x + Math.cos x

theorem math_problem (a : ℝ) (k : ℤ) (θ : ℝ) (h₁ : f a (Real.pi / 2) = -1) 
    (h₂ : 0 < θ ∧ θ < Real.pi / 2) (h₃ : f a θ = 1 / 2) :
    a = -1 ∧
    (Periodic (f (-1)) 2 * Real.pi) ∧
    ∀ k : ℤ, ∃ x : ℝ, - (5 * Real.pi / 4) + 2 * k * Real.pi ≤ x ∧ x ≤ 2 * k * Real.pi - Real.pi / 4 ∧
    f (-1) x = x ∧ 
    Real.sin (2 * θ) = 3 / 4 :=
by
  sorry

end math_problem_l743_743795


namespace find_larger_number_l743_743224

theorem find_larger_number (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) (h3 : x * y = 375) (hx : x > y) : x = 25 :=
sorry

end find_larger_number_l743_743224


namespace man_is_older_by_20_l743_743285

variables (M S : ℕ)
axiom h1 : S = 18
axiom h2 : M + 2 = 2 * (S + 2)

theorem man_is_older_by_20 :
  M - S = 20 :=
by {
  sorry
}

end man_is_older_by_20_l743_743285


namespace chromatic_number_bound_l743_743887

variable {V : Type} (G : SimpleGraph V)

noncomputable def chromaticNumber : ℕ := G.chromaticNumber

theorem chromatic_number_bound (G : SimpleGraph V) :
  chromaticNumber G ≤ 1 / 2 + Real.sqrt (2 * (G.edgeSet.card : ℝ) + 1 / 4) :=
begin
  sorry,
end

end chromatic_number_bound_l743_743887


namespace calculate_CD_and_BF_l743_743465

-- Define the scenario and variables

variables (A B C W D F : Type)
variables (b c WA : ℝ)

-- Conditions
axiom AC_eq_b : AC = b
axiom AB_eq_c : AB = c
axiom bisector_intersects_circumcircle : bisector ∠A intersects circumcircle ABC at W
axiom omega_center_W_radius_WA : circle with center W and radius WA intersects AC at D and AB at F

-- Questions
theorem calculate_CD_and_BF : CD = b - WA ∧ BF = c - WA := 
sorry

end calculate_CD_and_BF_l743_743465


namespace radius_circle_spherical_l743_743595

def spherical_to_cartesian (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Math.sin φ * Math.cos θ, ρ * Math.sin φ * Math.sin θ, ρ * Math.cos φ)

theorem radius_circle_spherical (θ : ℝ) : 
  (let (ρ, φ) := (1, π / 4)
   let (x, y, z) := spherical_to_cartesian ρ θ φ
   Math.sqrt (x^2 + y^2)) = (Real.sqrt 2) / 2 :=
by 
  let ρ := 1
  let φ := π / 4
  let (x, y, z) := spherical_to_cartesian ρ θ φ
  sorry

end radius_circle_spherical_l743_743595


namespace surface_equation_l743_743535

theorem surface_equation (S : set (ℝ × ℝ × ℝ)) :
  (∃ (S : set (ℝ × ℝ × ℝ)), (1, 1, 1) ∈ S ∧ 
    (∀ P ∈ S, ∃ (a b c: ℝ), P = (a, b, c) ∧ 
      a > 0 ∧ b > 0 ∧ c > 0 ∧
      ∀ x y z, (a * (b - y) * z + b * (c - z) * x + c * (a - x) * y = 0) → 
    (x, y, z) = P)) -> 
  S = { P : ℝ × ℝ × ℝ | ∃ (x y z : ℝ), P = (x, y, z) ∧ x^2 + y^2 + z^2 = 3 ∧ x > 0 ∧ y > 0 ∧ z > 0 } :=
sorry

end surface_equation_l743_743535


namespace exists_two_pairs_satisfy_2x3_eq_y4_l743_743732

theorem exists_two_pairs_satisfy_2x3_eq_y4 :
  ∃ (x₁ y₁ x₂ y₂ : ℕ), 2 * x₁^3 = y₁^4 ∧ 2 * x₂^3 = y₂^4 ∧ (x₁, y₁) ≠ (x₂, y₂) :=
by
  use 2, 2
  use 32, 16
  split
  . calc 2 * 2^3 = 2 * 8 : by rw [pow_succ]
            ... = 16      : by norm_num
  split
  . calc 2 * 32^3 = 2 * (2^5)^3 : rfl 
              ... = 2 * 2^15    : by rw [pow_mul]
              ... = 2^16        : by rw [mul_comm, pow_succ, mul_assoc, pow_one, two_mul]
  . exact ne_of_apply_ne Prod.fst $ by simp

end exists_two_pairs_satisfy_2x3_eq_y4_l743_743732


namespace diagonal_AC_is_1_l743_743363

-- Define the elements used in the conditions
noncomputable def angle_A : ℝ := 80
noncomputable def angle_C : ℝ := 140
noncomputable def length_AB : ℝ := 1
noncomputable def length_AD : ℝ := 1

-- State the proof problem
theorem diagonal_AC_is_1 (ABCD_is_convex : true) :
  let A := angle_A,
      C := angle_C,
      AB := length_AB,
      AD := length_AD
  in ∃ AC : ℝ, AC = 1 := 
by
  sorry

end diagonal_AC_is_1_l743_743363


namespace sequence_constant_modulo_n_l743_743194

theorem sequence_constant_modulo_n (n : ℕ) :
  ∃ N : ℕ, ∀ i j : ℕ, i ≥ N → j ≥ N → (2^(2^(i:ℕ))) % n = (2^(2^(j:ℕ))) % n :=
begin
  -- Since only the statement is needed:
  sorry
end

end sequence_constant_modulo_n_l743_743194


namespace em_parallel_ac_l743_743368

variables {P : Type*} [AffineSpace ℝ P]

structure IsoscelesTrapezoid (A B C D E M : P) (ab cd : Line ℝ P) :=
  (mid_M : midpointℝ B D M)
  (foot_E : foot_ab (line B A) D E)
  (parallel_ab_cd : parallel (line A B) (line C D))
  (parallel_em_ac : parallel (line E M) (line A C))

theorem em_parallel_ac {A B C D E M : P} 
  (h1 : IsoscelesTrapezoid A B C D E M ab cd) : 
  parallel (line E M) (line A C) :=
h1.parallel_em_ac

end em_parallel_ac_l743_743368


namespace part1_part2_l743_743399

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

theorem part1 (a : ℝ) (h : (2 * a - (a + 2) + 1) = 0) : a = 1 :=
by
  sorry

theorem part2 (a x : ℝ) (ha : a ≥ 1) (hx : 1 ≤ x ∧ x ≤ Real.exp 1) : (2 * a * x - (a + 2) + 1 / x) ≥ 0 :=
by
  sorry

end part1_part2_l743_743399


namespace find_x_l743_743860

namespace IntegerProblem

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := 
by
  sorry

end IntegerProblem

end find_x_l743_743860


namespace sum_paintable_numbers_l743_743815

/-- Define the conditions for a paintable number -/
def is_paintable (h t u : ℕ) : Prop :=
  h ≠ 1 ∧ t ≠ 1 ∧ u ≠ 1 ∧ 
  ∀ n : ℕ, ∃ unique (p : ℕ), 
    (p = 1 ∧ n % h = 0) ∨ 
    (p = 2 ∧ (n - 1) % t = 0) ∨ 
    (p = 3 ∧ (n - 2) % u = 0)

/-- Define the set of all paintable numbers -/
def paintable_numbers : List ℕ :=
  [100 * 3 + 10 * 3 + 3, 100 * 4 + 10 * 2 + 4]

/-- Prove that the sum of all paintable numbers is 757 -/
theorem sum_paintable_numbers : 
  paintable_numbers.sum = 757 := by 
  sorry

end sum_paintable_numbers_l743_743815


namespace reciprocal_of_repeating_decimal_equiv_l743_743980

noncomputable def repeating_decimal (x : ℝ) := 0.333333...

theorem reciprocal_of_repeating_decimal_equiv :
  (1 / repeating_decimal 0.333333...) = 3 :=
sorry

end reciprocal_of_repeating_decimal_equiv_l743_743980


namespace two_pow_p_add_three_pow_p_eq_a_pow_n_imp_n_eq_one_l743_743148

theorem two_pow_p_add_three_pow_p_eq_a_pow_n_imp_n_eq_one
  (p a n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hn : 0 < n) 
  (h : 2 ^ p + 3 ^ p = a ^ n) : n = 1 :=
sorry

end two_pow_p_add_three_pow_p_eq_a_pow_n_imp_n_eq_one_l743_743148


namespace isosceles_triangle_l743_743090

open_locale euclidean_geometry

variables {A B C I P Q M N : Point}
variables (α : circle)
variables (circumcircle_AIC : circle)

-- Condition 1
hypothesis h1 : incenter I (triangle.mk A B C)

-- Condition 2
hypothesis h2 : α = incircle (triangle.mk A B C)

-- Condition 3
hypothesis h3 : intersects circumcircle_AIC α P
hypothesis h4 : intersects circumcircle_AIC α Q

-- Condition 4
hypothesis h5 : same_side P A (line.mk B I)

-- Condition 5
hypothesis h6 : ¬ same_side Q C (line.mk B I)

-- Condition 6
hypothesis h7 : midpoint M (arc.mk α A C)

-- Condition 7
hypothesis h8 : midpoint N (arc.mk α B C)

-- Condition 8
hypothesis h9 : parallel (line.mk P Q) (line.mk A C)

-- Conclusion
theorem isosceles_triangle (h1 h2 h3 h4 h5 h6 h7 h8 h9) : (distance A B) = (distance A C) :=
sorry

end isosceles_triangle_l743_743090


namespace value_of_operations_l743_743752

def operation1 (x : ℕ) : ℤ := 8 - x
def operation2 (x : ℕ) : ℤ := x - 8

theorem value_of_operations (x : ℕ) : operation2 (operation1 10) = -10 :=
by {
  sorry
}

end value_of_operations_l743_743752


namespace problem_l743_743086

theorem problem (x : ℕ → ℝ) (h_sum : (∑ i in Finset.range 50, x i) = 2)
  (h_frac_sum : (∑ i in Finset.range 50, x i / (1 + x i)) = 3) :
  (∑ i in Finset.range 50, x i^2 / (1 + x i)) = 1 :=
by 
  sorry

end problem_l743_743086


namespace sci_not_218000_l743_743469

theorem sci_not_218000 : 218000 = 2.18 * 10^5 :=
by
  sorry

end sci_not_218000_l743_743469


namespace haley_cans_l743_743002

theorem haley_cans (
  total_cans : ℕ,
  first_bag : ℕ,
  second_bag : ℕ,
  third_bag : ℕ
) : (total_cans = 120) →
    (first_bag = 40) →
    (second_bag = 25) →
    (third_bag = 30) →
    let cans_in_fourth_bag := total_cans - (first_bag + second_bag + third_bag) in 
    cans_in_fourth_bag = 25 ∧ first_bag - cans_in_fourth_bag = 15 :=
by 
  intros h₁ h₂ h₃ h₄
  simp [h₁, h₂, h₃, h₄]
  sorry

end haley_cans_l743_743002


namespace sally_more_cards_than_dan_l743_743192

theorem sally_more_cards_than_dan :
  let sally_initial := 27
  let sally_bought := 20
  let dan_cards := 41
  sally_initial + sally_bought - dan_cards = 6 :=
by
  sorry

end sally_more_cards_than_dan_l743_743192


namespace smallest_palindromic_prime_l743_743624

def is_palindrome (n : ℕ) : Prop :=
  let s := toString n in s = s.reverse

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : Nat, m > 1 → m < n → n % m ≠ 0

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem smallest_palindromic_prime : ∃ n : ℕ, is_palindrome n ∧ is_prime n ∧ is_three_digit n ∧ 200 < n ∧ n < 300 ∧ ∀ m : ℕ, (is_palindrome m ∧ is_prime m ∧ is_three_digit m ∧ 200 < m ∧ m < 300) → n ≤ m ∧ n = 202 :=
by
  sorry

end smallest_palindromic_prime_l743_743624


namespace eighth_square_more_tiles_than_seventh_l743_743681

-- Define the total number of tiles in the nth square
def total_tiles (n : ℕ) : ℕ := n^2 + 2 * n

-- Formulate the theorem statement
theorem eighth_square_more_tiles_than_seventh :
  total_tiles 8 - total_tiles 7 = 17 := by
  sorry

end eighth_square_more_tiles_than_seventh_l743_743681


namespace odd_function_inequality_solution_set_l743_743588

noncomputable def f (x : ℝ) : ℝ := sorry -- Assuming f is implicitly defined with all the given conditions

theorem odd_function_inequality_solution_set (f : ℝ → ℝ)
  (hf_odd : ∀ x, f (-x) = -f x)
  (hf_monotonic : ∀ x y, 0 < x → x < y → f x < f y)
  (hf_f1_zero : f 1 = 0) :
  {x : ℝ | x * f x < 0} = Ioo (-1 : ℝ) 0 ∪ Ioo 0 1 :=
by sorry

end odd_function_inequality_solution_set_l743_743588


namespace total_monthly_cost_l743_743657

theorem total_monthly_cost (volume_per_box : ℕ := 1800) 
                          (total_volume : ℕ := 1080000)
                          (cost_per_box_per_month : ℝ := 0.8) 
                          (expected_cost : ℝ := 480) : 
                          (total_volume / volume_per_box) * cost_per_box_per_month = expected_cost :=
by
  sorry

end total_monthly_cost_l743_743657


namespace triangle_side_length_l743_743437

theorem triangle_side_length
  (ABC : Type) [metric_space ABC] [nonempty ABC]
  (A B C F G : ABC)
  (h_parallel : parallel (subsegment F G) (subsegment A B))
  (h_CF : dist C F = 5)
  (h_FA : dist F A = 15)
  (h_CG : dist C G = 8)
  : dist C B = 32 := sorry

end triangle_side_length_l743_743437


namespace eighth_group_sample_l743_743446

-- Define the given conditions as Lean definitions
def total_students : ℕ := 50
def sample_size : ℕ := 10
def interval : ℕ := total_students / sample_size

def students_group : ℕ → Set ℕ
| 1 := {1, 2, 3, 4, 5}
| 2 := {6, 7, 8, 9, 10}
| 3 := {11, 12, 13, 14, 15}
| 4 := {16, 17, 18, 19, 20}
| 5 := {21, 22, 23, 24, 25}
| 6 := {26, 27, 28, 29, 30}
| 7 := {31, 32, 33, 34, 35}
| 8 := {36, 37, 38, 39, 40}
| 9 := {41, 42, 43, 44, 45}
| 10 := {46, 47, 48, 49, 50}
| _ := ∅

-- Define the index of the group from which number 12 is drawn
def third_group_index : ℕ := 3
def eighth_group_index : ℕ := 8
def sampled_from_third_group : ℕ := 12

-- Define the initial number the systematic sampling starts from
def initial_number : ℕ := sampled_from_third_group - (2 - 1) * interval

-- Proof goal: to show that the number drawn from the eighth group is correct
theorem eighth_group_sample : ∃ n : ℕ, n ∈ students_group 8 ∧ n = initial_number + (8 - 2) * interval :=
by
  -- We need to prove that the number 37 is drawn from the eighth group
  existsi (7 + 30)
  split
  -- Verify the number 37 is in the eighth group
  { simp [students_group],
    norm_num,
    split; norm_num, }
  sorry

end eighth_group_sample_l743_743446


namespace min_vertical_segment_length_l743_743942

noncomputable def f1 (x : ℝ) := abs x

noncomputable def f2 (x : ℝ) := -x^2 + 2 * x - 1

theorem min_vertical_segment_length : 
  let segment_length := λ x : ℝ, abs (f1 x - f2 x) in
  ∃ x : ℝ, ∀ y : ℝ, (segment_length x <= segment_length y) → (segment_length x = (3 / 4)) :=
begin
  sorry
end

end min_vertical_segment_length_l743_743942


namespace sum_of_k_values_l743_743328

noncomputable def sum_of_positive_integers_k_with_integer_roots (p : ℕ → ℝ) : ℕ :=
  ∑ k in { k | ∃ (a b : ℤ), (a + b = k) ∧ (a * b = 16) ∧ k > 0 }, k

theorem sum_of_k_values : sum_of_positive_integers_k_with_integer_roots = 35 :=
by
  sorry

end sum_of_k_values_l743_743328


namespace simplify_fraction_l743_743554

theorem simplify_fraction :
  let x := 15625 in
  x = 5^6 → (√(√[3](√(1 / ↑x)))) = (√5 / 5) := 
by
  intro x h
  sorry

end simplify_fraction_l743_743554


namespace average_weight_l743_743931

theorem average_weight 
  (n₁ n₂ : ℕ) 
  (avg₁ avg₂ total_avg : ℚ) 
  (h₁ : n₁ = 24) 
  (h₂ : n₂ = 8)
  (h₃ : avg₁ = 50.25)
  (h₄ : avg₂ = 45.15)
  (h₅ : total_avg = 48.975) :
  ( (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = total_avg ) :=
sorry

end average_weight_l743_743931


namespace sequence_an_geometric_sum_of_bn_l743_743365

-- Definition of the sequence and conditions

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

def condition_sn (n : ℕ) : Prop := S n = 2 * a n - n
def sequence_an_plus_1_geometric (n : ℕ) : Prop :=
  ∃ (r : ℝ), (a 1 + 1 = 2) ∧ (∀ n ≥ 1, a (n+1) + 1 = 2 * (a n + 1))

variable {b : ℕ → ℝ}
def sequence_bn (n : ℕ) : ℝ := n * (a n + 1) / 2

def sum_tn (n : ℕ) : ℝ := (n - 1) * 2^n + 1

-- Theorem statements
theorem sequence_an_geometric (S : ℕ → ℝ) (a : ℕ → ℝ)
  (cond_sn : ∀ n, condition_sn n) :
  sequence_an_plus_1_geometric := by
  sorry

theorem sum_of_bn (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ)
  (cond_sn : ∀ n, condition_sn n)  
  (bn_def : ∀ n, b n = sequence_bn n) : 
  ∀ n, ∑ i in range(n), b i = sum_tn n := by
  sorry

end sequence_an_geometric_sum_of_bn_l743_743365


namespace cost_price_one_metre_l743_743683

noncomputable def selling_price : ℤ := 18000
noncomputable def total_metres : ℕ := 600
noncomputable def loss_per_metre : ℤ := 5

noncomputable def total_loss : ℤ := loss_per_metre * (total_metres : ℤ) -- Note the cast to ℤ for multiplication
noncomputable def cost_price : ℤ := selling_price + total_loss
noncomputable def cost_price_per_metre : ℚ := cost_price / (total_metres : ℤ)

theorem cost_price_one_metre : cost_price_per_metre = 35 := by
  sorry

end cost_price_one_metre_l743_743683


namespace domain_of_ln_x_plus_1_l743_743203

def domain_ln (x : ℝ) : Prop := x > -1

theorem domain_of_ln_x_plus_1 : 
  ∀ x : ℝ, (∃ y : ℝ, f(x) = real.log y ∧ y > 0) ↔ domain_ln x :=
by 
  sorry

end domain_of_ln_x_plus_1_l743_743203


namespace div_36_of_n_ge_5_l743_743427

noncomputable def n := Nat

theorem div_36_of_n_ge_5 (n : ℕ) (hn : n ≥ 5) (h2 : ¬ (n % 2 = 0)) (h3 : ¬ (n % 3 = 0)) : 36 ∣ (n^2 - 1) :=
by
  sorry

end div_36_of_n_ge_5_l743_743427


namespace sprinkler_system_days_l743_743278

theorem sprinkler_system_days 
  (morning_water : ℕ) (evening_water : ℕ) (total_water : ℕ) 
  (h_morning : morning_water = 4) 
  (h_evening : evening_water = 6) 
  (h_total : total_water = 50) :
  total_water / (morning_water + evening_water) = 5 := 
by 
  sorry

end sprinkler_system_days_l743_743278


namespace probability_green_ball_l743_743317

theorem probability_green_ball :
  let pA := 1 / 3
  let pB := 1 / 3
  let pC := 1 / 3
  let pGreenA := 7 / 12
  let pGreenB := 5 / 9
  let pGreenC := 3 / 10
  (pA * pGreenA + pB * pGreenB + pC * pGreenC) = 26 / 45 :=
by
  -- Definitions
  let pA := (1 : ℚ) / 3
  let pB := (1 : ℚ) / 3
  let pC := (1 : ℚ) / 3
  let pGreenA := (7 : ℚ) / 12
  let pGreenB := (5 : ℚ) / 9
  let pGreenC := (3 : ℚ) / 10

  -- Calculation
  have h : pA * pGreenA + pB * pGreenB + pC * pGreenC = (7 / 36 + 5 / 27 + 3 / 30) := sorry

  -- Simplification
  have hSimplified : 7 / 36 + 5 / 27 + 3 / 30 = 26 / 45 := sorry

  -- Combine and finish
  rw h
  exact hSimplified

end probability_green_ball_l743_743317


namespace snickers_bars_needed_l743_743473

-- Definitions for the problem conditions
def total_required_points : ℕ := 2000
def bunnies_sold : ℕ := 8
def bunny_points : ℕ := 100
def snickers_points : ℕ := 25
def points_from_bunnies : ℕ := bunnies_sold * bunny_points
def remaining_points_needed : ℕ := total_required_points - points_from_bunnies

-- Define the problem statement to prove
theorem snickers_bars_needed : remaining_points_needed / snickers_points = 48 :=
by
  -- Skipping the proof steps
  sorry

end snickers_bars_needed_l743_743473


namespace angles_parallel_sides_l743_743022

theorem angles_parallel_sides {α β : Type} (angle1 angle2 : α) (side1 side2 : β) 
  (h1 : parallel side1 side2) (h2 : parallel side1 side2) : 
α ∈ (angle1 ∨ angle2) ∨ α ∈ (angle1 + angle2 = 180) := 
begin
  sorry -- Proof goes here. This is to ensure the lean code builds successfully.
end

end angles_parallel_sides_l743_743022


namespace smallest_prime_p_l743_743755

theorem smallest_prime_p (p q s r : ℕ) (h1 : p.prime) (h2 : q.prime) (h3 : s.prime) (h4 : r.prime) (h5 : p + q + s = r) (h6 : 2 < p) (h7 : p < q) (h8 : q < s) : p = 3 :=
by { sorry }

end smallest_prime_p_l743_743755


namespace sqrt_112_consecutive_integers_product_l743_743606

theorem sqrt_112_consecutive_integers_product : 
  (∃ (a b : ℕ), a * a < 112 ∧ 112 < b * b ∧ b = a + 1 ∧ a * b = 110) :=
by 
  use 10, 11
  repeat { sorry }

end sqrt_112_consecutive_integers_product_l743_743606


namespace trajectory_of_center_is_hyperbola_right_branch_l743_743770

theorem trajectory_of_center_is_hyperbola_right_branch
  (C1 : Set Point) (C2 : Set Point) (C : Set Point)
  (circle_C1 : ∀ p, p ∈ C1 ↔ (p.x + 4)^2 + p.y^2 = 4)
  (circle_C2 : ∀ p, p ∈ C2 ↔ (p.x - 4)^2 + p.y^2 = 1)
  (externally_tangent : ∀ p r, p ∈ C ↔ ∃ x y, p = ⟨x, y⟩ ∧ (x + 4)^2 + y^2 = (2 + r)^2)
  (internally_tangent : ∀ p r, p ∈ C ↔ ∃ x y, p = ⟨x, y⟩ ∧ (x - 4)^2 + y^2 = (r - 1)^2) :
  ∀ p, p ∈ C → ∃ h, is_hyperbola_right_branch h p.center ∧ h.focus1 = ⟨-4, 0⟩ ∧ h.focus2 = ⟨4, 0⟩ :=
sorry

end trajectory_of_center_is_hyperbola_right_branch_l743_743770


namespace value_of_k_l743_743020

theorem value_of_k (x y k : ℝ) (h₁ : x = -3) (h₂ : y = 1) (h₃ : y = k / x) (h₄ : k ≠ 0) : k = -3 :=
by
  -- Definitions and conditions rewritten from part (a) and conclusion from part (b)
  intro x y k h₁ h₂ h₃ h₄
  sorry

end value_of_k_l743_743020


namespace parametric_equations_of_line_l743_743670

theorem parametric_equations_of_line (t : ℝ) : 
  let M := (1, 5)
  let θ := (2 * Real.pi) / 3
  let x := 1 - (1 / 2) * t
  let y := 5 + (Real.sqrt 3 / 2) * t
  in True := 
sorry

end parametric_equations_of_line_l743_743670


namespace regression_equation_binomial_variance_l743_743269

-- Given data points for x and y
def x_values : List ℝ := [5, 5.5, 6, 6.5, 7]
def y_values : List ℝ := [50, 48, 43, 38, 36]

-- Mean values
def mean (l : List ℝ) : ℝ := (l.sum / l.length)

def mean_x := mean x_values
def mean_y := mean y_values

-- Sum of x_i * y_i
def sum_xy (l1 l2 : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) l1 l2)

-- Sum of squares of x_i
def sum_x2 (l : List ℝ) : ℝ :=
  List.sum (List.map (λ x, x * x) l)

-- Regression coefficients
def b_hat : ℝ := (sum_xy x_values y_values - (5 * mean_x * mean_y)) / 
                 (sum_x2 x_values - (5 * mean_x^2))
def a_hat : ℝ := mean_y - b_hat * mean_x

-- Linear regression
theorem regression_equation : ∀ (x : ℝ), a_hat + b_hat * x = 88.6 - 7.6 * x := by
  sorry

-- Let X follow a binomial distribution
def p := 1 / 2
def n := 5

-- Probability distribution function of X
def binomial_pmf (k : ℕ) : ℝ := Nat.choose n k * (p)^(k : ℝ) * (1 - p) ^ (n - k)

-- Variance of X
theorem binomial_variance : (n : ℝ) * p * (1 - p) = 5 / 4 := by
  sorry

end regression_equation_binomial_variance_l743_743269


namespace prob_at_least_two_same_l743_743173

theorem prob_at_least_two_same (h : 8 > 0) : 
  (1 - (Nat.factorial 8 / (8^8) : ℚ) = 2043 / 2048) :=
by
  sorry

end prob_at_least_two_same_l743_743173


namespace tank_full_weight_l743_743927

theorem tank_full_weight (u v m n : ℝ) (h1 : m + 3 / 4 * n = u) (h2 : m + 1 / 3 * n = v) :
  m + n = 8 / 5 * u - 3 / 5 * v :=
sorry

end tank_full_weight_l743_743927


namespace james_gifted_stickers_l743_743059

def james_stickers_before_birthday : ℕ := 39
def james_stickers_after_birthday : ℕ := 61
def stickers_gifted := james_stickers_after_birthday - james_stickers_before_birthday

theorem james_gifted_stickers : stickers_gifted = 22 := by
  -- given conditions
  have h1 : james_stickers_before_birthday = 39 := rfl
  have h2 : james_stickers_after_birthday = 61 := rfl
  -- proof
  unfold stickers_gifted
  rw [h1, h2]
  norm_num

end james_gifted_stickers_l743_743059


namespace sum_of_special_numbers_eq_51_l743_743351

def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

def number_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).count (λ d => n % d = 0)

def condition (n : ℕ) : Prop :=
  is_multiple_of_three n ∧ number_of_divisors n = n / 3

theorem sum_of_special_numbers_eq_51 :
  (Finset.filter condition (Finset.range 100)).sum (λ x => x) = 51 :=
by
  sorry

end sum_of_special_numbers_eq_51_l743_743351


namespace isosceles_triangle_l743_743133

   open EuclideanGeometry

   -- Define the conditions of the problem in Lean
   variable {I A B C P Q M N : Point}
   variable (α : Circle) (circumcircle_AIC : Circle)

   -- Conditions extracted from the problem
   def conditions : Prop :=
   IsIncenter I △ABC ∧
   Incircle α △ABC ∧
   Circle.Diameter α P Q ∧
   Circle.Containing circumcircle_AIC (trianglePoint AIC) ∧
   SameSide P A (Line BI) ∧
   SameSide Q C (Line BI) ∧
   IsMidpointArc M ARC(α A C) ∧
   IsMidpointArc N ARC(α B C) ∧
   Parallel (Line PQ) (Line AC)

   -- Proof statement in Lean
   theorem isosceles_triangle
     (h : conditions α circumcircle_AIC) : IsIsosceles (△ABC) :=
   sorry
   
end isosceles_triangle_l743_743133


namespace proof_inequality_l743_743643

variables {A B C D E : Type} [Plane A B C D E] [IsoscelesTriangle ABC AB AC] 
open Plane

noncomputable def angle_BAC : ℝ := 100
noncomputable def length_AD : ℝ := sorry  -- This represents the length of AD
noncomputable def length_BE : ℝ := sorry  -- This represents the length of BE
noncomputable def length_EA : ℝ := sorry  -- This represents the length of EA

theorem proof_inequality :
  2 * length_AD < length_BE + length_EA :=
sorry

end proof_inequality_l743_743643


namespace number_value_l743_743901

theorem number_value (N : ℝ) (h : 0.40 * N = 180) : 
  (1/4) * (1/3) * (2/5) * N = 15 :=
by
  -- assume the conditions have been stated correctly
  sorry

end number_value_l743_743901


namespace sum_dihedral_angles_eq_360_l743_743835

theorem sum_dihedral_angles_eq_360 {A B C O : Point}
  (cylinder : RightCircularCylinder)
  (hA : A ∈ cylinder.base)
  (hB : B ∈ cylinder.base)
  (hC : C ∈ cylinder.other_base)
  (hDiamereA B : A ≠ B)
  (hPiOAB : ¬(O ∈ Plane(A, B)))
  (hO : O = cylinder.axis_midpoint) :
  dihedral_angle(O, A, B) + dihedral_angle(O, B, C) + dihedral_angle(O, C, A) = 360 :=
sorry

end sum_dihedral_angles_eq_360_l743_743835


namespace find_triangle_altitude_l743_743570

variable (A b h : ℝ)

theorem find_triangle_altitude (h_eq_40 :  A = 800 ∧ b = 40) : h = 40 :=
sorry

end find_triangle_altitude_l743_743570


namespace first_expression_evaluation_second_expression_evaluation_l743_743557

-- First expression proof statement
theorem first_expression_evaluation (a : ℚ) (h : a = -1 / 2) : 
  a * (a^4 - a + 1) * (a - 2) = 59 / 32 :=
by
  rw h
  sorry

-- Second expression proof statement
theorem second_expression_evaluation (x y : ℚ) (hx : x = 8) (hy : y = -1 / 2) : 
  (x + 2 * y) * (x - y) - (2 * x + 1) * (-x - y) = 188 :=
by
  rw [hx, hy]
  sorry

end first_expression_evaluation_second_expression_evaluation_l743_743557
