import Mathlib

namespace vector_addition_proof_l1957_195782

variables {Point : Type} [AddCommGroup Point]

variables (A B C D : Point)

theorem vector_addition_proof :
  (D - A) + (C - D) - (C - B) = B - A :=
by
  sorry

end vector_addition_proof_l1957_195782


namespace cubic_identity_l1957_195789

theorem cubic_identity (x y z : ℝ) (h1 : x + y + z = 13) (h2 : xy + xz + yz = 32) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 949 :=
by
  sorry

end cubic_identity_l1957_195789


namespace binomial_coeff_arith_seq_expansion_l1957_195784

open BigOperators

-- Given the binomial expansion of (sqrt(x) + 2/sqrt(x))^n
-- we need to prove that the condition on binomial coefficients
-- implies that n = 7, and the expansion contains no constant term.
theorem binomial_coeff_arith_seq_expansion (x : ℝ) (n : ℕ) :
  (2 * Nat.choose n 2 = Nat.choose n 1 + Nat.choose n 3) ↔ n = 7 ∧ ∀ r : ℕ, x ^ (7 - 2 * r) / 2 ≠ x ^ 0 := by
  sorry

end binomial_coeff_arith_seq_expansion_l1957_195784


namespace root_sum_reciprocal_l1957_195705

theorem root_sum_reciprocal (p q r s : ℂ)
  (h1 : (∀ x : ℂ, x^4 - 6*x^3 + 11*x^2 - 6*x + 3 = 0 → x = p ∨ x = q ∨ x = r ∨ x = s))
  (h2 : p*q*r*s = 3) 
  (h3 : p*q + p*r + p*s + q*r + q*s + r*s = 11) :
  (1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s)) = 11/3 :=
by
  sorry

end root_sum_reciprocal_l1957_195705


namespace athlete_total_heartbeats_l1957_195732

theorem athlete_total_heartbeats (h : ℕ) (p : ℕ) (d : ℕ) (r : ℕ) : (h = 150) ∧ (p = 6) ∧ (d = 30) ∧ (r = 15) → (p * d + r) * h = 29250 :=
by
  sorry

end athlete_total_heartbeats_l1957_195732


namespace cone_diameter_l1957_195795

theorem cone_diameter (S : ℝ) (hS : S = 3 * Real.pi) (unfold_semicircle : ∃ (r l : ℝ), l = 2 * r ∧ S = π * r^2 + (1 / 2) * π * l^2) : 
∃ d : ℝ, d = Real.sqrt 6 := 
by
  sorry

end cone_diameter_l1957_195795


namespace parallel_vectors_implies_scalar_l1957_195757

-- Defining the vectors a and b
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (-2, m)

-- Stating the condition and required proof
theorem parallel_vectors_implies_scalar (m : ℝ) (h : (vector_a.snd / vector_a.fst) = (vector_b m).snd / (vector_b m).fst) : m = -4 :=
by sorry

end parallel_vectors_implies_scalar_l1957_195757


namespace pauline_total_spent_l1957_195790

variable {items_total : ℝ} (discount_rate : ℝ) (discount_limit : ℝ) (sales_tax_rate : ℝ)

def total_spent (items_total discount_rate discount_limit sales_tax_rate : ℝ) : ℝ :=
  let discount_amount := discount_rate * discount_limit
  let discounted_total := discount_limit - discount_amount
  let non_discounted_total := items_total - discount_limit
  let subtotal := discounted_total + non_discounted_total
  let sales_tax := sales_tax_rate * subtotal
  subtotal + sales_tax

theorem pauline_total_spent :
  total_spent 250 0.15 100 0.08 = 253.80 :=
by
  sorry

end pauline_total_spent_l1957_195790


namespace range_of_a_l1957_195792

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x > 2 then 2^x + a else x + a^2

theorem range_of_a (a : ℝ) : (∀ y : ℝ, ∃ x : ℝ, f x a = y) ↔ (a ≤ -1 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l1957_195792


namespace shrimp_price_l1957_195786

theorem shrimp_price (y : ℝ) (h : 0.6 * (y / 4) = 2.25) : y = 15 :=
sorry

end shrimp_price_l1957_195786


namespace gain_percent_of_articles_l1957_195752

theorem gain_percent_of_articles (C S : ℝ) (h : 50 * C = 15 * S) : (S - C) / C * 100 = 233.33 :=
by
  sorry

end gain_percent_of_articles_l1957_195752


namespace markup_percentage_l1957_195729

variable (W R : ℝ)

-- Condition: When sold at a 40% discount, a sweater nets the merchant a 30% profit on the wholesale cost.
def discount_condition : Prop := 0.6 * R = 1.3 * W

-- Theorem: The percentage markup of the sweater from wholesale to normal retail price is 116.67%
theorem markup_percentage (h : discount_condition W R) : (R - W) / W * 100 = 116.67 :=
by sorry

end markup_percentage_l1957_195729


namespace ab_zero_proof_l1957_195773

-- Given conditions
def square_side : ℝ := 3
def rect_short_side : ℝ := 3
def rect_long_side : ℝ := 6
def rect_area : ℝ := rect_short_side * rect_long_side
def split_side_proof (a b : ℝ) : Prop := a + b = rect_short_side

-- Lean theorem proving that ab = 0 given the conditions
theorem ab_zero_proof (a b : ℝ) 
  (h1 : square_side = 3)
  (h2 : rect_short_side = 3)
  (h3 : rect_long_side = 6)
  (h4 : rect_area = 18)
  (h5 : split_side_proof a b) : a * b = 0 := by
  sorry

end ab_zero_proof_l1957_195773


namespace arithmetic_sequence_S30_l1957_195798

variable {α : Type*} [OrderedAddCommGroup α]

-- Definitions from the conditions
def arithmetic_sum (n : ℕ) : α :=
  sorry -- Placeholder for the sequence sum definition

axiom S10 : arithmetic_sum 10 = 20
axiom S20 : arithmetic_sum 20 = 15

-- The theorem to prove
theorem arithmetic_sequence_S30 : arithmetic_sum 30 = -15 :=
  sorry -- Proof will be completed here

end arithmetic_sequence_S30_l1957_195798


namespace reduced_price_per_kg_l1957_195749

/-- Given that:
1. There is a reduction of 25% in the price of oil.
2. The housewife can buy 5 kgs more for Rs. 700 after the reduction.

Prove that the reduced price per kg of oil is Rs. 35. -/
theorem reduced_price_per_kg (P : ℝ) (R : ℝ) (X : ℝ)
  (h1 : R = 0.75 * P)
  (h2 : 700 = X * P)
  (h3 : 700 = (X + 5) * R)
  : R = 35 := 
sorry

end reduced_price_per_kg_l1957_195749


namespace pyramid_bottom_right_value_l1957_195744

theorem pyramid_bottom_right_value (a x y z b : ℕ) (h1 : 18 = (21 + x) / 2)
  (h2 : 14 = (21 + y) / 2) (h3 : 16 = (15 + z) / 2) (h4 : b = (21 + y) / 2) :
  a = 6 := 
sorry

end pyramid_bottom_right_value_l1957_195744


namespace find_P_l1957_195710

theorem find_P (P : ℕ) (h : 4 * (P + 4 + 8 + 20) = 252) : P = 31 :=
by
  -- Assume this proof is nontrivial and required steps
  sorry

end find_P_l1957_195710


namespace find_int_k_l1957_195768

theorem find_int_k (Z K : ℤ) (h1 : 1000 < Z) (h2 : Z < 1500) (h3 : K > 1) (h4 : Z = K^3) :
  K = 11 :=
by
  sorry

end find_int_k_l1957_195768


namespace total_cost_is_correct_l1957_195774

-- Conditions
def cost_per_object : ℕ := 11
def objects_per_person : ℕ := 5  -- 2 shoes, 2 socks, 1 mobile per person
def number_of_people : ℕ := 3

-- Expected total cost
def expected_total_cost : ℕ := 165

-- Proof problem: Prove that the total cost for storing all objects is 165 dollars
theorem total_cost_is_correct :
  (number_of_people * objects_per_person * cost_per_object) = expected_total_cost :=
by
  sorry

end total_cost_is_correct_l1957_195774


namespace earnings_proof_l1957_195725

theorem earnings_proof (A B C : ℕ) (h1 : A + B + C = 600) (h2 : B + C = 300) (h3 : C = 100) : A + C = 400 :=
sorry

end earnings_proof_l1957_195725


namespace octagon_perimeter_l1957_195740

/-- 
  Represents the side length of the regular octagon
-/
def side_length : ℕ := 12

/-- 
  Represents the number of sides of a regular octagon
-/
def number_of_sides : ℕ := 8

/-- 
  Defines the perimeter of the regular octagon
-/
def perimeter (side_length : ℕ) (number_of_sides : ℕ) : ℕ :=
  side_length * number_of_sides

/-- 
  Proof statement: asserting that the perimeter of a regular octagon
  with a side length of 12 meters is 96 meters
-/
theorem octagon_perimeter :
  perimeter side_length number_of_sides = 96 :=
  sorry

end octagon_perimeter_l1957_195740


namespace range_of_m_l1957_195770

noncomputable def f (x m : ℝ) : ℝ := -x^2 - 4 * m * x + 1

theorem range_of_m (m : ℝ) : 
  (∀ x1 x2 : ℝ, 2 ≤ x1 → x1 ≤ x2 → f x1 m ≥ f x2 m) ↔ m ≥ -1 := 
sorry

end range_of_m_l1957_195770


namespace range_of_x_coordinate_l1957_195738

theorem range_of_x_coordinate (x : ℝ) : 
  (0 ≤ 2*x + 2 ∧ 2*x + 2 ≤ 1) ↔ (-1 ≤ x ∧ x ≤ -1/2) := 
sorry

end range_of_x_coordinate_l1957_195738


namespace arithmetic_sequence_geometric_condition_l1957_195719

theorem arithmetic_sequence_geometric_condition 
  (a : ℕ → ℝ) (d : ℝ) (h_nonzero : d ≠ 0) 
  (h_a3 : a 3 = 7)
  (h_geo_seq : (a 2 - 1)^2 = (a 1 - 1) * (a 4 - 1)) : 
  a 10 = 21 :=
sorry

end arithmetic_sequence_geometric_condition_l1957_195719


namespace mean_of_roots_l1957_195750

theorem mean_of_roots
  (a b c d k : ℤ)
  (p : ℤ → ℤ)
  (h_poly : ∀ x, p x = (x - a) * (x - b) * (x - c) * (x - d))
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_root : p k = 4) :
  k = (a + b + c + d) / 4 :=
by
  -- proof goes here
  sorry

end mean_of_roots_l1957_195750


namespace arithmetic_seq_S11_l1957_195759

def Sn (n : ℕ) (a₁ d : ℤ) : ℤ :=
  n * a₁ + (n * (n - 1)) / 2 * d

theorem arithmetic_seq_S11 (a₁ d : ℤ)
  (h1 : a₁ = -11)
  (h2 : (Sn 10 a₁ d) / 10 - (Sn 8 a₁ d) / 8 = 2) :
  Sn 11 a₁ d = -11 :=
by
  sorry

end arithmetic_seq_S11_l1957_195759


namespace average_of_21_numbers_l1957_195776

theorem average_of_21_numbers (n₁ n₂ : ℕ) (a b c : ℕ)
  (h₁ : n₁ = 11 * 48) -- Sum of the first 11 numbers
  (h₂ : n₂ = 11 * 41) -- Sum of the last 11 numbers
  (h₃ : c = 55) -- The 11th number
  : (n₁ + n₂ - c) / 21 = 44 := -- Average of all 21 numbers
by
  sorry

end average_of_21_numbers_l1957_195776


namespace find_x_l1957_195702

theorem find_x (n x q p : ℕ) (h1 : n = q * x + 2) (h2 : 2 * n = p * x + 4) : x = 6 :=
sorry

end find_x_l1957_195702


namespace binom_18_6_l1957_195736

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_18_6 : binomial 18 6 = 18564 := 
by
  sorry

end binom_18_6_l1957_195736


namespace relative_complement_correct_l1957_195787

noncomputable def M : Set ℤ := {x : ℤ | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {1, 2}
def complement_M_N : Set ℤ := {x ∈ M | x ∉ N}

theorem relative_complement_correct : complement_M_N = {-1, 0, 3} := 
by
  sorry

end relative_complement_correct_l1957_195787


namespace complement_union_l1957_195777

open Set

-- Define U to be the set of all real numbers
def U := @univ ℝ

-- Define the domain A for the function y = sqrt(x-2) + sqrt(x+1)
def A := {x : ℝ | x ≥ 2}

-- Define the domain B for the function y = sqrt(2x+4) / (x-3)
def B := {x : ℝ | x ≥ -2 ∧ x ≠ 3}

-- Theorem about the union of the complements
theorem complement_union : (U \ A ∪ U \ B) = {x : ℝ | x < 2 ∨ x = 3} := 
by
  sorry

end complement_union_l1957_195777


namespace calc_3a2008_minus_5b2008_l1957_195747

theorem calc_3a2008_minus_5b2008 (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 - b^2 = -1) : 3 * a ^ 2008 - 5 * b ^ 2008 = -5 :=
by
  sorry

end calc_3a2008_minus_5b2008_l1957_195747


namespace ellipse_parabola_intersection_l1957_195708

theorem ellipse_parabola_intersection (c : ℝ) : 
  (∀ x y : ℝ, (x^2 + (y^2 / 4) = c^2 ∧ y = x^2 - 2 * c) → false) ↔ c > 1 := by
  sorry

end ellipse_parabola_intersection_l1957_195708


namespace tetrahedron_is_self_dual_cube_is_dual_to_octahedron_dodecahedron_is_dual_to_icosahedron_l1957_195763

structure Tetrahedron :=
  (faces : Nat := 4)
  (vertices : Nat := 4)
  (valence : Nat := 3)
  (face_shape : String := "triangular")

structure Cube :=
  (faces : Nat := 6)
  (vertices : Nat := 8)
  (valence : Nat := 3)
  (face_shape : String := "square")

structure Octahedron :=
  (faces : Nat := 8)
  (vertices : Nat := 6)
  (valence : Nat := 4)
  (face_shape : String := "triangular")

structure Dodecahedron :=
  (faces : Nat := 12)
  (vertices : Nat := 20)
  (valence : Nat := 3)
  (face_shape : String := "pentagonal")

structure Icosahedron :=
  (faces : Nat := 20)
  (vertices : Nat := 12)
  (valence : Nat := 5)
  (face_shape : String := "triangular")

theorem tetrahedron_is_self_dual:
  Tetrahedron := by
  sorry

theorem cube_is_dual_to_octahedron:
  Cube × Octahedron := by
  sorry

theorem dodecahedron_is_dual_to_icosahedron:
  Dodecahedron × Icosahedron := by
  sorry

end tetrahedron_is_self_dual_cube_is_dual_to_octahedron_dodecahedron_is_dual_to_icosahedron_l1957_195763


namespace find_divisor_l1957_195765

theorem find_divisor (D Q R d : ℕ) (h1 : D = 159) (h2 : Q = 9) (h3 : R = 6) (h4 : D = d * Q + R) : d = 17 := by
  sorry

end find_divisor_l1957_195765


namespace shaded_region_perimeter_l1957_195711

theorem shaded_region_perimeter :
  let side_length := 1
  let diagonal_length := Real.sqrt 2 * side_length
  let arc_TRU_length := (1 / 4) * (2 * Real.pi * diagonal_length)
  let arc_VPW_length := (1 / 4) * (2 * Real.pi * side_length)
  let arc_UV_length := (1 / 4) * (2 * Real.pi * (Real.sqrt 2 - side_length))
  let arc_WT_length := (1 / 4) * (2 * Real.pi * (Real.sqrt 2 - side_length))
  (arc_TRU_length + arc_VPW_length + arc_UV_length + arc_WT_length) = (2 * Real.sqrt 2 - 1) * Real.pi :=
by
  sorry

end shaded_region_perimeter_l1957_195711


namespace polynomial_transformation_l1957_195771

noncomputable def f (x : ℝ) : ℝ := sorry

theorem polynomial_transformation (x : ℝ) :
  (f (x^2 + 2) = x^4 + 6 * x^2 + 4) →
  f (x^2 - 2) = x^4 - 2 * x^2 - 4 :=
by
  intro h
  sorry

end polynomial_transformation_l1957_195771


namespace unique_solution_l1957_195754

noncomputable def unique_solution_exists : Prop :=
  ∃ (a b c d e : ℕ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
    (a + b = (c + d + e) / 7) ∧
    (a + d = (b + c + e) / 5) ∧
    (a + b + c + d + e = 24) ∧
    (a = 1 ∧ b = 2 ∧ c = 9 ∧ d = 3 ∧ e = 9)

theorem unique_solution : unique_solution_exists :=
sorry

end unique_solution_l1957_195754


namespace solution_to_system_l1957_195779

theorem solution_to_system :
  (∀ (x y : ℚ), (y - x - 1 = 0) ∧ (y + x - 2 = 0) ↔ (x = 1/2 ∧ y = 3/2)) :=
by
  sorry

end solution_to_system_l1957_195779


namespace probability_white_or_red_l1957_195778

theorem probability_white_or_red (a b c : ℕ) : 
  (a + b) / (a + b + c) = (a + b) / (a + b + c) := by
  -- Conditions
  let total_balls := a + b + c
  let white_red_balls := a + b
  -- Goal
  have prob_white_or_red := white_red_balls / total_balls
  exact rfl

end probability_white_or_red_l1957_195778


namespace combined_avg_score_l1957_195793

noncomputable def classA_student_count := 45
noncomputable def classB_student_count := 55
noncomputable def classA_avg_score := 110
noncomputable def classB_avg_score := 90

theorem combined_avg_score (nA nB : ℕ) (avgA avgB : ℕ) 
  (h1 : nA = classA_student_count) 
  (h2 : nB = classB_student_count) 
  (h3 : avgA = classA_avg_score) 
  (h4 : avgB = classB_avg_score) : 
  (nA * avgA + nB * avgB) / (nA + nB) = 99 := 
by 
  rw [h1, h2, h3, h4]
  -- Substitute the values to get:
  -- (45 * 110 + 55 * 90) / (45 + 55) 
  -- = (4950 + 4950) / 100 
  -- = 9900 / 100 
  -- = 99
  sorry

end combined_avg_score_l1957_195793


namespace container_alcohol_amount_l1957_195781

theorem container_alcohol_amount
  (A : ℚ) -- Amount of alcohol in quarts
  (initial_water : ℚ) -- Initial amount of water in quarts
  (added_water : ℚ) -- Amount of water added in quarts
  (final_ratio_alcohol_to_water : ℚ) -- Final ratio of alcohol to water
  (h_initial_water : initial_water = 4) -- Container initially contains 4 quarts of water.
  (h_added_water : added_water = 8/3) -- 2.666666666666667 quarts of water added.
  (h_final_ratio : final_ratio_alcohol_to_water = 3/5) -- Final ratio is 3 parts alcohol to 5 parts water.
  (h_final_water : initial_water + added_water = 20/3) -- Total final water quarts after addition.
  : A = 4 := 
sorry

end container_alcohol_amount_l1957_195781


namespace machine_work_rates_l1957_195717

theorem machine_work_rates :
  (∃ x : ℝ, (1 / (x + 4) + 1 / (x + 3) + 1 / (x + 2)) = 1 / x ∧ x = 1 / 2) :=
by
  sorry

end machine_work_rates_l1957_195717


namespace measured_diagonal_in_quadrilateral_l1957_195762

-- Defining the conditions (side lengths and diagonals)
def valid_diagonal (side1 side2 side3 side4 diagonal : ℝ) : Prop :=
  side1 + side2 > diagonal ∧ side1 + side3 > diagonal ∧ side1 + side4 > diagonal ∧ 
  side2 + side3 > diagonal ∧ side2 + side4 > diagonal ∧ side3 + side4 > diagonal

theorem measured_diagonal_in_quadrilateral :
  let sides := [1, 2, 2.8, 5]
  let diagonal1 := 7.5
  let diagonal2 := 2.8
  (valid_diagonal 1 2 2.8 5 diagonal2) :=
sorry

end measured_diagonal_in_quadrilateral_l1957_195762


namespace december_sales_fraction_l1957_195788

variable (A : ℝ)

-- Define the total sales for January through November
def total_sales_jan_to_nov := 11 * A

-- Define the sales total for December, which is given as 5 times the average monthly sales from January to November
def sales_dec := 5 * A

-- Define the total sales for the year as the sum of January-November sales and December sales
def total_sales_year := total_sales_jan_to_nov + sales_dec

-- We need to prove that the fraction of the December sales to the total annual sales is 5/16
theorem december_sales_fraction : sales_dec / total_sales_year = 5 / 16 := by
  sorry

end december_sales_fraction_l1957_195788


namespace olivia_initial_quarters_l1957_195756

theorem olivia_initial_quarters : 
  ∀ (spent_quarters left_quarters initial_quarters : ℕ),
  spent_quarters = 4 → left_quarters = 7 → initial_quarters = spent_quarters + left_quarters → initial_quarters = 11 :=
by
  intros spent_quarters left_quarters initial_quarters h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end olivia_initial_quarters_l1957_195756


namespace solutions_shifted_quadratic_l1957_195739

theorem solutions_shifted_quadratic (a h k : ℝ) (x1 x2: ℝ)
  (h1 : a * (-1 - h)^2 + k = 0)
  (h2 : a * (3 - h)^2 + k = 0) :
  a * (0 - (h + 1))^2 + k = 0 ∧ a * (4 - (h + 1))^2 + k = 0 :=
by
  sorry

end solutions_shifted_quadratic_l1957_195739


namespace find_special_three_digit_numbers_l1957_195720

theorem find_special_three_digit_numbers :
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 
  (100 * a + 10 * b + (c + 3)) % 10 + (100 * a + 10 * (b + 1) + c).div 10 % 10 + (100 * (a + 1) + 10 * b + c).div 100 % 10 + 3 = 
  (a + b + c) / 3)} → n = 117 ∨ n = 207 ∨ n = 108 :=
by
  sorry

end find_special_three_digit_numbers_l1957_195720


namespace train_length_l1957_195780

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (h_speed : speed_kmh = 60) (h_time : time_s = 21) :
  (speed_kmh * (1000 / 3600) * time_s) = 350.07 := 
by
  sorry

end train_length_l1957_195780


namespace find_fx_l1957_195727

variable {e : ℝ} {a : ℝ} (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

variable (hodd : odd_function f)
variable (hdef : ∀ x, -e ≤ x → x < 0 → f x = a * x + Real.log (-x))

theorem find_fx (x : ℝ) (hx : 0 < x ∧ x ≤ e) : f x = a * x - Real.log x :=
by
  sorry

end find_fx_l1957_195727


namespace ball_weights_l1957_195723

-- Define the weights of red and white balls we are going to use in our conditions and goal
variables (R W : ℚ)

-- State the conditions as hypotheses
axiom h1 : 7 * R + 5 * W = 43
axiom h2 : 5 * R + 7 * W = 47

-- State the theorem we want to prove, given the conditions
theorem ball_weights :
  4 * R + 8 * W = 49 :=
by
  sorry

end ball_weights_l1957_195723


namespace integers_within_range_l1957_195766

def is_within_range (n : ℤ) : Prop :=
  (-1.3 : ℝ) < (n : ℝ) ∧ (n : ℝ) < 2.8

theorem integers_within_range :
  { n : ℤ | is_within_range n } = {-1, 0, 1, 2} :=
by
  sorry

end integers_within_range_l1957_195766


namespace distance_james_rode_l1957_195767

def speed : ℝ := 80.0
def time : ℝ := 16.0
def distance : ℝ := speed * time

theorem distance_james_rode :
  distance = 1280.0 :=
by
  -- to show the theorem is sane
  sorry

end distance_james_rode_l1957_195767


namespace value_of_y_l1957_195751

theorem value_of_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3 * y) = 8) : y = 1 :=
by
  sorry

end value_of_y_l1957_195751


namespace ratio_wy_l1957_195745

-- Define the variables and conditions
variables (w x y z : ℚ)
def ratio_wx := w / x = 5 / 4
def ratio_yz := y / z = 7 / 5
def ratio_zx := z / x = 1 / 8

-- Statement to prove
theorem ratio_wy (hwx : ratio_wx w x) (hyz : ratio_yz y z) (hzx : ratio_zx z x) : w / y = 25 / 7 :=
by
  sorry  -- Proof not needed

end ratio_wy_l1957_195745


namespace triangle_shortest_side_condition_l1957_195742

theorem triangle_shortest_side_condition
  (A B C : Type) 
  (r : ℝ) (AF FB : ℝ)
  (P : ℝ)
  (h_AF : AF = 7)
  (h_FB : FB = 9)
  (h_r : r = 5)
  (h_P : P = 46) 
  : (min (min (7 + 9) (2 * 14)) ((7 + 9) - 14)) = 2 := 
by sorry

end triangle_shortest_side_condition_l1957_195742


namespace women_in_village_l1957_195796

theorem women_in_village (W : ℕ) (men_present : ℕ := 150) (p : ℝ := 140.78099890167377) 
    (men_reduction_per_year: ℝ := 0.10) (year1_men : ℝ := men_present * (1 - men_reduction_per_year)) 
    (year2_men : ℝ := year1_men * (1 - men_reduction_per_year)) 
    (formula : ℝ := (year2_men^2 + W^2).sqrt) 
    (h : formula = p) : W = 71 := 
by
  sorry

end women_in_village_l1957_195796


namespace angle_B_lt_pi_div_two_l1957_195726

theorem angle_B_lt_pi_div_two 
  (a b c : ℝ) (B : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : B = π / 2 - B)
  (h5 : 2 / b = 1 / a + 1 / c)
  : B < π / 2 := sorry

end angle_B_lt_pi_div_two_l1957_195726


namespace part1_part2_l1957_195769

theorem part1 (θ : ℝ) (h : Real.sin (2 * θ) = 2 * Real.cos θ) :
  (6 * Real.sin θ + Real.cos θ) / (3 * Real.sin θ - 2 * Real.cos θ) = 13 / 4 :=
sorry

theorem part2 (θ : ℝ) (h : Real.sin (2 * θ) = 2 * Real.cos θ) :
  (Real.sin θ ^ 2 + 2 * Real.sin θ * Real.cos θ) / (2 * Real.sin θ ^ 2 - Real.cos θ ^ 2) = 8 / 7 :=
sorry

end part1_part2_l1957_195769


namespace geometric_sequence_mean_l1957_195785

theorem geometric_sequence_mean (a : ℕ → ℝ) (q : ℝ) (h_q : q = -2) 
  (h_condition : a 3 * a 7 = 4 * a 4) : 
  ((a 8 + a 11) / 2 = -56) 
:= sorry

end geometric_sequence_mean_l1957_195785


namespace factor_expression_l1957_195712

theorem factor_expression (x : ℝ) : 
  x^2 * (x + 3) + 3 * (x + 3) = (x^2 + 3) * (x + 3) :=
by
  sorry

end factor_expression_l1957_195712


namespace min_students_with_both_l1957_195794

-- Given conditions
def total_students : ℕ := 35
def students_with_brown_eyes : ℕ := 18
def students_with_lunch_box : ℕ := 25

-- Mathematical statement to prove the least number of students with both attributes
theorem min_students_with_both :
  ∃ x : ℕ, students_with_brown_eyes + students_with_lunch_box - total_students ≤ x ∧ x = 8 :=
sorry

end min_students_with_both_l1957_195794


namespace function_characterization_l1957_195746

theorem function_characterization (f : ℤ → ℤ)
  (h : ∀ a b : ℤ, ∃ k : ℤ, f (f a - b) + b * f (2 * a) = k ^ 2) :
  (∀ n : ℤ, (n % 2 = 0 → f n = 0) ∧ (n % 2 ≠ 0 → ∃ k: ℤ, f n = k ^ 2))
  ∨ (∀ n : ℤ, ∃ k: ℤ, f n = k ^ 2 ∧ k = n) :=
sorry

end function_characterization_l1957_195746


namespace arithmetic_sequence_sum_l1957_195730

theorem arithmetic_sequence_sum :
  (∀ (a : ℕ → ℤ),  a 1 + a 2 = 4 ∧ a 3 + a 4 = 6 → a 8 + a 9 = 10) :=
sorry

end arithmetic_sequence_sum_l1957_195730


namespace value_of_x3_plus_inv_x3_l1957_195755

theorem value_of_x3_plus_inv_x3 (x : ℝ) (h : 728 = x^6 + 1 / x^6) : 
  x^3 + 1 / x^3 = Real.sqrt 730 :=
sorry

end value_of_x3_plus_inv_x3_l1957_195755


namespace sum_of_n_for_perfect_square_l1957_195718

theorem sum_of_n_for_perfect_square (n : ℕ) (Sn : ℕ) 
  (hSn : Sn = n^2 + 20 * n + 12) 
  (hn : n > 0) :
  ∃ k : ℕ, k^2 = Sn → (sum_of_possible_n = 16) :=
by
  sorry

end sum_of_n_for_perfect_square_l1957_195718


namespace ellipse_a_value_l1957_195716

theorem ellipse_a_value
  (a : ℝ)
  (h1 : 0 < a)
  (h2 : ∀ x y : ℝ, x^2 / a^2 + y^2 / 5 = 1)
  (e : ℝ)
  (h3 : e = 2 / 3)
  : a = 3 :=
by
  sorry

end ellipse_a_value_l1957_195716


namespace complex_numbers_not_comparable_l1957_195707

-- Definitions based on conditions
def is_real (z : ℂ) : Prop := ∃ r : ℝ, z = r
def is_not_entirely_real (z : ℂ) : Prop := ¬ is_real z

-- Proof problem statement
theorem complex_numbers_not_comparable (z1 z2 : ℂ) (h1 : is_not_entirely_real z1) (h2 : is_not_entirely_real z2) : 
  ¬ (z1.re = z2.re ∧ z1.im = z2.im) :=
sorry

end complex_numbers_not_comparable_l1957_195707


namespace complete_square_quadratic_t_l1957_195764

theorem complete_square_quadratic_t : 
  ∀ x : ℝ, (16 * x^2 - 32 * x - 512 = 0) → (∃ q t : ℝ, (x + q)^2 = t ∧ t = 33) :=
by sorry

end complete_square_quadratic_t_l1957_195764


namespace shadow_area_greatest_integer_l1957_195713

theorem shadow_area_greatest_integer (x : ℝ)
  (h1 : ∀ (a : ℝ), a = 1)
  (h2 : ∀ (b : ℝ), b = 48)
  (h3 : ∀ (c: ℝ), x = 1 / 6):
  ⌊1000 * x⌋ = 166 := 
by sorry

end shadow_area_greatest_integer_l1957_195713


namespace a_n_less_than_inverse_n_minus_1_l1957_195791

theorem a_n_less_than_inverse_n_minus_1 
  (n : ℕ) (h1 : 2 ≤ n) 
  (a : ℕ → ℝ) 
  (h2 : ∀ k : ℕ, 1 ≤ k → k ≤ n-1 → (a (k-1) + a k) * (a k + a (k+1)) = a (k-1) - a (k+1)) 
  (h3 : ∀ m : ℕ, m ≤ n → 0 < a m) : 
  a n < 1 / (n - 1) :=
sorry

end a_n_less_than_inverse_n_minus_1_l1957_195791


namespace smallest_number_collected_l1957_195761

-- Define the numbers collected by each person according to the conditions
def jungkook : ℕ := 6 * 3
def yoongi : ℕ := 4
def yuna : ℕ := 5

-- The statement to prove
theorem smallest_number_collected : yoongi = min (min jungkook yoongi) yuna :=
by sorry

end smallest_number_collected_l1957_195761


namespace june_ride_time_l1957_195715

theorem june_ride_time (dist1 time1 dist2 time2 : ℝ) (h : dist1 = 2 ∧ time1 = 8 ∧ dist2 = 5 ∧ time2 = 20) :
  (dist2 / (dist1 / time1) = time2) := by
  -- using the defined conditions
  rcases h with ⟨h1, h2, h3, h4⟩
  rw [h1, h2, h3, h4]
  -- simplifying the expression
  sorry

end june_ride_time_l1957_195715


namespace range_of_values_includes_one_integer_l1957_195758

theorem range_of_values_includes_one_integer (x : ℝ) (h : -1 < 2 * x + 3 ∧ 2 * x + 3 < 1) :
  ∃! n : ℤ, -7 < (2 * x - 3) ∧ (2 * x - 3) < -5 ∧ n = -6 :=
sorry

end range_of_values_includes_one_integer_l1957_195758


namespace minimum_a_l1957_195753

theorem minimum_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 - (x - a) * |x - a| - 2 ≥ 0) → a ≥ Real.sqrt 3 := 
by 
  sorry

end minimum_a_l1957_195753


namespace observable_sea_creatures_l1957_195783

theorem observable_sea_creatures (P_shark : ℝ) (P_truth : ℝ) (n : ℕ)
  (h1 : P_shark = 0.027777777777777773)
  (h2 : P_truth = 1/6)
  (h3 : P_shark = P_truth * (1/n : ℝ)) : 
  n = 6 := 
  sorry

end observable_sea_creatures_l1957_195783


namespace value_of_t_l1957_195722

theorem value_of_t (x y t : ℝ) (hx : 2^x = t) (hy : 7^y = t) (hxy : 1/x + 1/y = 2) : t = Real.sqrt 14 :=
by
  sorry

end value_of_t_l1957_195722


namespace factor_polynomial_l1957_195724

theorem factor_polynomial (x y z : ℂ) : 
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) = 
  (x - y) * (y - z) * (z - x) * (-(x * y + x * z + y * z)) := by
  sorry

end factor_polynomial_l1957_195724


namespace ellipse_standard_and_trajectory_l1957_195741

theorem ellipse_standard_and_trajectory :
  ∀ a b x y : ℝ, 
  a > b ∧ 0 < b ∧ 
  (b^2 = a^2 - 1) ∧ 
  (9/4 + 6/(8) = 1) →
  (∃ x y : ℝ, (x / 2)^2 / 9 + (y)^2 / 8 = 1) ∧ 
  (x^2 / 9 - y^2 / 8 = 1 ∧ x ≠ 3 ∧ x ≠ -3) := 
  sorry

end ellipse_standard_and_trajectory_l1957_195741


namespace jared_annual_earnings_l1957_195714

-- Defining conditions as constants
def diploma_pay : ℕ := 4000
def degree_multiplier : ℕ := 3
def months_in_year : ℕ := 12

-- Goal: Prove that Jared's annual earnings are $144,000
theorem jared_annual_earnings : diploma_pay * degree_multiplier * months_in_year = 144000 := by
  sorry

end jared_annual_earnings_l1957_195714


namespace ab_value_l1957_195799

theorem ab_value (a b : ℝ) (h1 : b^2 - a^2 = 4) (h2 : a^2 + b^2 = 25) : abs (a * b) = Real.sqrt (609 / 4) := 
sorry

end ab_value_l1957_195799


namespace percentage_regular_cars_l1957_195721

theorem percentage_regular_cars (total_cars : ℕ) (truck_percentage : ℚ) (convertibles : ℕ) 
  (h1 : total_cars = 125) (h2 : truck_percentage = 0.08) (h3 : convertibles = 35) : 
  (80 / 125 : ℚ) * 100 = 64 := 
by 
  sorry

end percentage_regular_cars_l1957_195721


namespace train_length_l1957_195703

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def train_speed_kmph : ℝ := 45
noncomputable def jogger_head_start_m : ℝ := 240
noncomputable def train_passing_time_s : ℝ := 35.99712023038157

noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * (1000 / 3600)
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps
noncomputable def distance_covered_by_train : ℝ := relative_speed_mps * train_passing_time_s

theorem train_length :
  distance_covered_by_train - jogger_head_start_m = 119.9712023038157 :=
by
  sorry

end train_length_l1957_195703


namespace cost_of_watermelon_and_grapes_l1957_195701

variable (x y z f : ℕ)

theorem cost_of_watermelon_and_grapes (h1 : x + y + z + f = 45) 
                                    (h2 : f = 3 * x) 
                                    (h3 : z = x + y) :
    y + z = 9 := by
  sorry

end cost_of_watermelon_and_grapes_l1957_195701


namespace race_result_l1957_195704

-- Defining competitors
inductive Sprinter
| A
| B
| C

open Sprinter

-- Conditions as definitions
def position_changes : Sprinter → Nat
| A => sorry
| B => 5
| C => 6

def finishes_before (s1 s2 : Sprinter) : Prop := sorry

-- Stating the problem as a theorem
theorem race_result :
  position_changes C = 6 →
  position_changes B = 5 →
  finishes_before B A →
  (finishes_before B A ∧ finishes_before A C ∧ finishes_before B C) :=
by
  intros hC hB hBA
  sorry

end race_result_l1957_195704


namespace steps_in_staircase_using_210_toothpicks_l1957_195734

-- Define the conditions
def first_step : Nat := 3
def increment : Nat := 2
def total_toothpicks_5_steps : Nat := 55

-- Define required theorem
theorem steps_in_staircase_using_210_toothpicks : ∃ (n : ℕ), (n * (n + 2) = 210) ∧ n = 13 :=
by
  sorry

end steps_in_staircase_using_210_toothpicks_l1957_195734


namespace walter_zoo_time_l1957_195706

def seals_time : ℕ := 13
def penguins_time : ℕ := 8 * seals_time
def elephants_time : ℕ := 13
def total_time_spent_at_zoo : ℕ := seals_time + penguins_time + elephants_time

theorem walter_zoo_time : total_time_spent_at_zoo = 130 := by
  -- Proof goes here
  sorry

end walter_zoo_time_l1957_195706


namespace cos_2x_quadratic_l1957_195728

theorem cos_2x_quadratic (x : ℝ) (a b c : ℝ)
  (h : a * (Real.cos x) ^ 2 + b * Real.cos x + c = 0)
  (h_a : a = 4) (h_b : b = 2) (h_c : c = -1) :
  4 * (Real.cos (2 * x)) ^ 2 + 2 * Real.cos (2 * x) - 1 = 0 := sorry

end cos_2x_quadratic_l1957_195728


namespace intersection_A_B_l1957_195748

open Set

def f (x : ℕ) : ℕ := x^2 - 12 * x + 36

def A : Set ℕ := {a | 1 ≤ a ∧ a ≤ 10}

def B : Set ℕ := {b | ∃ a, a ∈ A ∧ b = f a}

theorem intersection_A_B : A ∩ B = {1, 4, 9} :=
by
  -- Proof skipped
  sorry

end intersection_A_B_l1957_195748


namespace inradius_one_third_height_l1957_195760

-- The problem explicitly states this triangle's sides form an arithmetic progression.
-- We need to define conditions and then prove the question is equivalent to the answer given those conditions.
theorem inradius_one_third_height (a b c r h_b : ℝ) (h : a ≤ b ∧ b ≤ c) (h_arith : 2 * b = a + c) :
  r = h_b / 3 :=
sorry

end inradius_one_third_height_l1957_195760


namespace number_of_bouquets_l1957_195733

theorem number_of_bouquets : ∃ n, n = 9 ∧ ∀ x y : ℕ, 3 * x + 2 * y = 50 → (x < 17) ∧ (x % 2 = 0 → y = (50 - 3 * x) / 2) :=
by
  sorry

end number_of_bouquets_l1957_195733


namespace rectangle_length_twice_breadth_l1957_195709

theorem rectangle_length_twice_breadth
  (b : ℝ) 
  (l : ℝ)
  (h1 : l = 2 * b)
  (h2 : (l - 5) * (b + 4) = l * b + 75) :
  l = 190 / 3 :=
sorry

end rectangle_length_twice_breadth_l1957_195709


namespace evaluate_expression_l1957_195700

noncomputable def x : ℚ := 4 / 8
noncomputable def y : ℚ := 5 / 6

theorem evaluate_expression : (8 * x + 6 * y) / (72 * x * y) = 3 / 10 :=
by
  sorry

end evaluate_expression_l1957_195700


namespace sum_of_remainders_l1957_195731

-- Definitions of the given problem
def a : ℕ := 1234567
def b : ℕ := 123

-- First remainder calculation
def r1 : ℕ := a % b

-- Second remainder calculation with the power
def r2 : ℕ := (2 ^ r1) % b

-- The proof statement
theorem sum_of_remainders : r1 + r2 = 29 := by
  sorry

end sum_of_remainders_l1957_195731


namespace max_gcd_seq_l1957_195735

theorem max_gcd_seq (a : ℕ → ℕ) (d : ℕ → ℕ) :
  (∀ n : ℕ, a n = 121 + n^2) →
  (∀ n : ℕ, d n = Nat.gcd (a n) (a (n + 1))) →
  ∃ m : ℕ, ∀ n : ℕ, d n ≤ d m ∧ d m = 99 :=
by
  sorry

end max_gcd_seq_l1957_195735


namespace angle_CAD_l1957_195772

noncomputable def angle_arc (degree: ℝ) (minute: ℝ) : ℝ :=
  degree + minute / 60

theorem angle_CAD :
  angle_arc 117 23 / 2 + angle_arc 42 37 / 2 = 80 :=
by
  sorry

end angle_CAD_l1957_195772


namespace ratio_induction_l1957_195737

theorem ratio_induction (k : ℕ) (hk : k > 0) :
    (k + 2) * (k + 3) / (2 * (2 * k + 1)) = 1 := by
sorry

end ratio_induction_l1957_195737


namespace cycling_time_difference_l1957_195775

-- Definitions from the conditions
def youth_miles : ℤ := 20
def youth_hours : ℤ := 2
def adult_miles : ℤ := 12
def adult_hours : ℤ := 3

-- Conversion from hours to minutes
def hours_to_minutes (hours : ℤ) : ℤ := hours * 60

-- Time per mile calculations
def youth_time_per_mile : ℤ := hours_to_minutes youth_hours / youth_miles
def adult_time_per_mile : ℤ := hours_to_minutes adult_hours / adult_miles

-- The difference in time per mile
def time_difference : ℤ := adult_time_per_mile - youth_time_per_mile

-- Theorem to prove the difference is 9 minutes
theorem cycling_time_difference : time_difference = 9 := by
  -- Proof steps would go here
  sorry

end cycling_time_difference_l1957_195775


namespace contractor_absent_days_l1957_195797

-- Definition of conditions
def total_days : ℕ := 30
def payment_per_work_day : ℝ := 25
def fine_per_absent_day : ℝ := 7.5
def total_payment : ℝ := 490

-- The proof statement
theorem contractor_absent_days : ∃ y : ℕ, (∃ x : ℕ, x + y = total_days ∧ payment_per_work_day * (x : ℝ) - fine_per_absent_day * (y : ℝ) = total_payment) ∧ y = 8 := 
by 
  sorry

end contractor_absent_days_l1957_195797


namespace x_less_than_y_by_35_percent_l1957_195743

noncomputable def percentage_difference (x y : ℝ) : ℝ :=
  ((y / x) - 1) * 100

theorem x_less_than_y_by_35_percent (x y : ℝ) (h : y = 1.5384615384615385 * x) :
  percentage_difference x y = 53.846153846153854 :=
by
  sorry

end x_less_than_y_by_35_percent_l1957_195743
