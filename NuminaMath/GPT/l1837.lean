import Mathlib

namespace pie_shop_total_earnings_l1837_183722

theorem pie_shop_total_earnings :
  let price_per_slice_custard := 3
  let price_per_slice_apple := 4
  let price_per_slice_blueberry := 5
  let slices_per_whole_custard := 10
  let slices_per_whole_apple := 8
  let slices_per_whole_blueberry := 12
  let num_whole_custard_pies := 6
  let num_whole_apple_pies := 4
  let num_whole_blueberry_pies := 5
  let total_earnings :=
    (num_whole_custard_pies * slices_per_whole_custard * price_per_slice_custard) +
    (num_whole_apple_pies * slices_per_whole_apple * price_per_slice_apple) +
    (num_whole_blueberry_pies * slices_per_whole_blueberry * price_per_slice_blueberry)
  total_earnings = 608 := by
  sorry

end pie_shop_total_earnings_l1837_183722


namespace initial_oak_trees_l1837_183772

theorem initial_oak_trees (n : ℕ) (h : n - 2 = 7) : n = 9 := 
by
  sorry

end initial_oak_trees_l1837_183772


namespace three_digit_integer_divisible_by_5_l1837_183740

theorem three_digit_integer_divisible_by_5 (M : ℕ) (h1 : 100 ≤ M ∧ M < 1000) (h2 : M % 10 = 5) : M % 5 = 0 := 
sorry

end three_digit_integer_divisible_by_5_l1837_183740


namespace find_HCF_l1837_183764

-- Given conditions
def LCM : ℕ := 750
def product_of_two_numbers : ℕ := 18750

-- Proof statement
theorem find_HCF (h : ℕ) (hpos : h > 0) :
  (LCM * h = product_of_two_numbers) → h = 25 :=
by
  sorry

end find_HCF_l1837_183764


namespace number_of_readers_who_read_both_l1837_183732

theorem number_of_readers_who_read_both (S L B total : ℕ) (hS : S = 250) (hL : L = 550) (htotal : total = 650) (h : S + L - B = total) : B = 150 :=
by {
  /-
  Given:
  S = 250 (number of readers who read science fiction)
  L = 550 (number of readers who read literary works)
  total = 650 (total number of readers)
  h : S + L - B = total (relationship between sets)
  We need to prove: B = 150
  -/
  sorry
}

end number_of_readers_who_read_both_l1837_183732


namespace smallest_in_sample_l1837_183758

theorem smallest_in_sample:
  ∃ (m : ℕ) (δ : ℕ), m ≥ 0 ∧ δ > 0 ∧ δ * 5 = 80 ∧ 42 = δ * (42 / δ) + m ∧ m < δ ∧ (∀ i < 5, m + i * δ < 80) → m = 10 :=
by
  sorry

end smallest_in_sample_l1837_183758


namespace greatest_integer_difference_l1837_183774

theorem greatest_integer_difference (x y : ℤ) (hx : -6 < (x : ℝ)) (hx2 : (x : ℝ) < -2) (hy : 4 < (y : ℝ)) (hy2 : (y : ℝ) < 10) : 
  ∃ d : ℤ, d = y - x ∧ d = 14 := 
by
  sorry

end greatest_integer_difference_l1837_183774


namespace gcd_182_98_l1837_183700

theorem gcd_182_98 : Nat.gcd 182 98 = 14 :=
by
  -- Provide the proof here, but as per instructions, we'll use sorry to skip it.
  sorry

end gcd_182_98_l1837_183700


namespace tank_full_weight_l1837_183735

theorem tank_full_weight (u v m n : ℝ) (h1 : m + 3 / 4 * n = u) (h2 : m + 1 / 3 * n = v) :
  m + n = 8 / 5 * u - 3 / 5 * v :=
sorry

end tank_full_weight_l1837_183735


namespace problem_inequality_l1837_183776

theorem problem_inequality (a b c : ℝ) (h₀ : a + b + c = 0) (d : ℝ) (h₁ : d = max (|a|) (max (|b|) (|c|))) : 
  |(1 + a) * (1 + b) * (1 + c)| ≥ 1 - d^2 :=
sorry

end problem_inequality_l1837_183776


namespace apples_left_is_ten_l1837_183710

noncomputable def appleCost : ℝ := 0.80
noncomputable def orangeCost : ℝ := 0.50
def initialApples : ℕ := 50
def initialOranges : ℕ := 40
def totalEarnings : ℝ := 49
def orangesLeft : ℕ := 6

theorem apples_left_is_ten (A : ℕ) :
  (50 - A) * appleCost + (40 - orangesLeft) * orangeCost = 49 → A = 10 :=
by
  sorry

end apples_left_is_ten_l1837_183710


namespace maximum_volume_pyramid_is_one_sixteenth_l1837_183788

open Real  -- Opening Real namespace for real number operations

noncomputable def maximum_volume_pyramid : ℝ :=
  let a := 1 -- side length of the equilateral triangle base
  let base_area := (sqrt 3 / 4) * (a * a) -- area of the equilateral triangle with side length 1
  let median := sqrt 3 / 2 * a -- median length of the triangle
  let height := 1 / 2 * median -- height of the pyramid
  let volume := 1 / 3 * base_area * height -- volume formula for a pyramid
  volume

theorem maximum_volume_pyramid_is_one_sixteenth :
  maximum_volume_pyramid = 1 / 16 :=
by
  simp [maximum_volume_pyramid] -- Simplify the volume definition
  sorry -- Proof omitted

end maximum_volume_pyramid_is_one_sixteenth_l1837_183788


namespace price_decrease_percentage_l1837_183718

-- Define the conditions
variables {P : ℝ} (original_price increased_price decreased_price : ℝ)
variables (y : ℝ) -- percentage by which increased price is decreased

-- Given conditions
def store_conditions :=
  increased_price = 1.20 * original_price ∧
  decreased_price = increased_price * (1 - y/100) ∧
  decreased_price = 0.75 * original_price

-- The proof problem
theorem price_decrease_percentage 
  (original_price increased_price decreased_price : ℝ)
  (y : ℝ) 
  (h : store_conditions original_price increased_price decreased_price y) :
  y = 37.5 :=
by 
  sorry

end price_decrease_percentage_l1837_183718


namespace sum_of_coefficients_is_2_l1837_183754

noncomputable def polynomial_expansion_condition (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ) :=
  (x^2 + 1) * (x - 2)^9 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + 
                          a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7 + a_8 * (x - 1)^8 + 
                          a_9 * (x - 1)^9 + a_10 * (x - 1)^10 + a_11 * (x - 1)^11

theorem sum_of_coefficients_is_2 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ) :
  polynomial_expansion_condition 1 a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 →
  polynomial_expansion_condition 2 a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11 = 2 :=
by sorry

end sum_of_coefficients_is_2_l1837_183754


namespace examination_is_30_hours_l1837_183723

noncomputable def examination_time_in_hours : ℝ :=
  let total_questions := 200
  let type_a_problems := 10
  let total_time_on_type_a := 17.142857142857142
  let time_per_type_a := total_time_on_type_a / type_a_problems
  let time_per_type_b := time_per_type_a / 2
  let type_b_problems := total_questions - type_a_problems
  let total_time_on_type_b := time_per_type_b * type_b_problems
  let total_time_in_minutes := total_time_on_type_a * type_a_problems + total_time_on_type_b
  total_time_in_minutes / 60

theorem examination_is_30_hours :
  examination_time_in_hours = 30 := by
  sorry

end examination_is_30_hours_l1837_183723


namespace product_of_consecutive_integers_sqrt_50_l1837_183702

theorem product_of_consecutive_integers_sqrt_50 :
  ∃ (a b : ℕ), (a < b) ∧ (b = a + 1) ∧ (a * a < 50) ∧ (50 < b * b) ∧ (a * b = 56) :=
by
  sorry

end product_of_consecutive_integers_sqrt_50_l1837_183702


namespace laundry_loads_needed_l1837_183734

theorem laundry_loads_needed
  (families : ℕ) (people_per_family : ℕ)
  (towels_per_person_per_day : ℕ) (days : ℕ)
  (washing_machine_capacity : ℕ)
  (h_f : families = 7)
  (h_p : people_per_family = 6)
  (h_t : towels_per_person_per_day = 2)
  (h_d : days = 10)
  (h_w : washing_machine_capacity = 10) : 
  ((families * people_per_family * towels_per_person_per_day * days) / washing_machine_capacity) = 84 := 
by
  sorry

end laundry_loads_needed_l1837_183734


namespace brenda_age_l1837_183746

theorem brenda_age (A B J : ℕ) (h1 : A = 3 * B) (h2 : J = B + 10) (h3 : A = J) : B = 5 :=
sorry

end brenda_age_l1837_183746


namespace triangle_properties_l1837_183762

theorem triangle_properties (a b c : ℝ) 
  (h : |a - Real.sqrt 7| + Real.sqrt (b - 5) + (c - 4 * Real.sqrt 2)^2 = 0) :
  a = Real.sqrt 7 ∧ b = 5 ∧ c = 4 * Real.sqrt 2 ∧ a^2 + b^2 = c^2 := by
{
  sorry
}

end triangle_properties_l1837_183762


namespace interval_monotonic_decrease_min_value_g_l1837_183715

noncomputable def a (x : ℝ) : ℝ × ℝ := (3 * Real.sqrt 3 * Real.sin x, Real.sqrt 3 * Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := let (a1, a2) := a x; let (b1, b2) := b x; a1 * b1 + a2 * b2
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x + m

theorem interval_monotonic_decrease (x : ℝ) (k : ℤ) :
  0 ≤ x ∧ x ≤ Real.pi ∧ (2 * x + Real.pi / 6) ∈ [Real.pi/2 + 2 * (k : ℝ) * Real.pi, 3 * Real.pi/2 + 2 * (k : ℝ) * Real.pi] →
  x ∈ [Real.pi / 6 + (k : ℝ) * Real.pi, 2 * Real.pi / 3 + (k : ℝ) * Real.pi] := sorry

theorem min_value_g (x : ℝ) :
  x ∈ [- Real.pi / 3, Real.pi / 3] →
  ∃ x₀, g x₀ 1 = -1/2 ∧ x₀ = - Real.pi / 3 := sorry

end interval_monotonic_decrease_min_value_g_l1837_183715


namespace simplify_expression_l1837_183749

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := 
by
  sorry

end simplify_expression_l1837_183749


namespace instantaneous_velocity_at_t_eq_2_l1837_183795

variable (t : ℝ)

def displacement (t : ℝ) : ℝ := 2 * (1 - t) ^ 2 

theorem instantaneous_velocity_at_t_eq_2 :
  (deriv (displacement) 2) = 4 :=
sorry

end instantaneous_velocity_at_t_eq_2_l1837_183795


namespace find_a_for_tangent_l1837_183798

theorem find_a_for_tangent (a : ℤ) (x : ℝ) (h : ∀ x, 3*x^2 - 4*a*x + 2*a > 0) : a = 1 :=
sorry

end find_a_for_tangent_l1837_183798


namespace find_k_l1837_183725

theorem find_k (k : ℝ) : 2 + (2 + k) / 3 + (2 + 2 * k) / 3^2 + (2 + 3 * k) / 3^3 + 
  ∑' (n : ℕ), (2 + (n + 1) * k) / 3^(n + 1) = 7 ↔ k = 16 / 3 := 
sorry

end find_k_l1837_183725


namespace sin_cos_identity_trig_identity_l1837_183777

open Real

-- Problem I
theorem sin_cos_identity (α : ℝ) : 
  (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5 / 7 → 
  sin α * cos α = 3 / 10 := 
sorry

-- Problem II
theorem trig_identity : 
  (sqrt (1 - 2 * sin (10 * π / 180) * cos (10 * π / 180))) / 
  (cos (10 * π / 180) - sqrt (1 - cos (170 * π / 180)^2)) = 1 := 
sorry

end sin_cos_identity_trig_identity_l1837_183777


namespace solve_q_l1837_183701

-- Definitions of conditions
variable (p q : ℝ)
variable (k : ℝ) 

-- Initial conditions
axiom h1 : p = 1500
axiom h2 : q = 0.5
axiom h3 : p * q = k
axiom h4 : k = 750

-- Goal
theorem solve_q (hp : p = 3000) : q = 0.250 :=
by
  -- The proof is omitted.
  sorry

end solve_q_l1837_183701


namespace T_value_l1837_183713

variable (x : ℝ)

def T : ℝ := (x-2)^4 + 4 * (x-2)^3 + 6 * (x-2)^2 + 4 * (x-2) + 1

theorem T_value : T x = (x-1)^4 := by
  sorry

end T_value_l1837_183713


namespace regular_decagon_interior_angle_l1837_183765

-- Define the number of sides in a regular decagon
def n : ℕ := 10

-- Define the formula for the sum of the interior angles of an n-sided polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the measure of one interior angle of a regular decagon
def one_interior_angle_of_regular_polygon (sum_of_angles : ℕ) (n : ℕ) : ℕ :=
  sum_of_angles / n

-- Prove that the measure of one interior angle of a regular decagon is 144 degrees
theorem regular_decagon_interior_angle : one_interior_angle_of_regular_polygon (sum_of_interior_angles 10) 10 = 144 := by
  sorry

end regular_decagon_interior_angle_l1837_183765


namespace determinant_zero_l1837_183783

noncomputable def A (α β : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    ![0, Real.cos α, Real.sin α],
    ![-Real.cos α, 0, Real.cos β],
    ![-Real.sin α, -Real.cos β, 0]
  ]

theorem determinant_zero (α β : ℝ) : Matrix.det (A α β) = 0 := 
  sorry

end determinant_zero_l1837_183783


namespace find_radius_of_semicircle_l1837_183793

-- Definitions for the rectangle and semi-circle
variable (L W : ℝ) -- Length and width of the rectangle
variable (r : ℝ) -- Radius of the semi-circle

-- Conditions given in the problem
def rectangle_perimeter : Prop := 2 * L + 2 * W = 216
def semicircle_diameter_eq_length : Prop := L = 2 * r 
def width_eq_twice_radius : Prop := W = 2 * r

-- Proof statement
theorem find_radius_of_semicircle
  (h_perimeter : rectangle_perimeter L W)
  (h_diameter : semicircle_diameter_eq_length L r)
  (h_width : width_eq_twice_radius W r) :
  r = 27 := by
  sorry

end find_radius_of_semicircle_l1837_183793


namespace focus_of_parabola_proof_l1837_183748

noncomputable def focus_of_parabola (a : ℝ) (h : a ≠ 0) : ℝ × ℝ :=
  (1 / (4 * a), 0)

theorem focus_of_parabola_proof (a : ℝ) (h : a ≠ 0) :
  focus_of_parabola a h = (1 / (4 * a), 0) :=
sorry

end focus_of_parabola_proof_l1837_183748


namespace functional_relationship_maximizing_profit_minimum_boxes_for_2000_profit_l1837_183720

-- Definitions for the problem conditions
def cost_per_box : ℝ := 20
def min_selling_price : ℝ := 25
def max_selling_price : ℝ := 38
def base_sales_volume : ℝ := 250
def price_decrease_effect : ℝ := 10
def profit_requirement : ℝ := 2000

-- Given the initial conditions
noncomputable def sales_volume (x : ℝ) : ℝ := base_sales_volume - price_decrease_effect * (x - min_selling_price)

-- Target problem statement
-- Part 1: Functional relationship between y and x
theorem functional_relationship (x : ℝ) : sales_volume x = -10 * x + 500 := by
sorry

-- Part 2: Maximizing profit
noncomputable def profit (x : ℝ) : ℝ := (x - cost_per_box) * sales_volume x

theorem maximizing_profit : ∃ (x : ℝ), x = 35 ∧ profit x = 2250 := by
sorry

-- Part 3: Minimum number of boxes to sell for at least 2000 yuan profit
theorem minimum_boxes_for_2000_profit (x : ℝ) : x ≤ max_selling_price → profit x ≥ profit_requirement → sales_volume x ≥ 120 := by
sorry

end functional_relationship_maximizing_profit_minimum_boxes_for_2000_profit_l1837_183720


namespace TimPrankCombinations_l1837_183784

-- Definitions of the conditions in the problem
def MondayChoices : ℕ := 3
def TuesdayChoices : ℕ := 1
def WednesdayChoices : ℕ := 6
def ThursdayChoices : ℕ := 4
def FridayChoices : ℕ := 2

-- The main theorem to prove the total combinations
theorem TimPrankCombinations : 
  MondayChoices * TuesdayChoices * WednesdayChoices * ThursdayChoices * FridayChoices = 144 := 
by
  sorry

end TimPrankCombinations_l1837_183784


namespace quadratic_inequality_solution_l1837_183773

theorem quadratic_inequality_solution :
  {x : ℝ | -x^2 + x + 2 > 0} = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end quadratic_inequality_solution_l1837_183773


namespace clownfish_in_display_tank_l1837_183794

theorem clownfish_in_display_tank (C B : ℕ) (h1 : C = B) (h2 : C + B = 100) : 
  (B - 26 - (B - 26) / 3) = 16 := by
  sorry

end clownfish_in_display_tank_l1837_183794


namespace sin_2y_eq_37_40_l1837_183752

variable (x y : ℝ)
variable (sin cos : ℝ → ℝ)

axiom sin_def : sin x = 2 * cos y - (5/2) * sin y
axiom cos_def : cos x = 2 * sin y - (5/2) * cos y

theorem sin_2y_eq_37_40 : sin (2 * y) = 37 / 40 := by
  sorry

end sin_2y_eq_37_40_l1837_183752


namespace min_major_axis_length_l1837_183716

theorem min_major_axis_length (a b c : ℝ) (h_area : b * c = 1) (h_focal_relation : 2 * a = 2 * Real.sqrt (b^2 + c^2)) :
  2 * a = 2 * Real.sqrt 2 :=
by
  sorry

end min_major_axis_length_l1837_183716


namespace infinite_n_exists_l1837_183799

-- Definitions from conditions
def is_natural_number (a : ℕ) : Prop := a > 3

-- Statement of the theorem
theorem infinite_n_exists (a : ℕ) (h : is_natural_number a) : ∃ᶠ n in at_top, a + n ∣ a^n + 1 :=
sorry

end infinite_n_exists_l1837_183799


namespace problem_statements_l1837_183797

theorem problem_statements :
  let S1 := ∀ (x : ℤ) (k : ℤ), x = 2 * k + 1 → (x % 2 = 1)
  let S2 := (∀ (x : ℝ), x > 2 → x > 1) 
            ∧ (∀ (x : ℝ), x > 1 → (x ≥ 2 ∨ x < 2)) 
  let S3 := ∀ (x : ℝ), ¬(∃ (x : ℝ), ∃ (y : ℝ), y = x^2 + 1 ∧ x = y)
  let S4 := ¬(∀ (x : ℝ), x > 1 → x^2 - x > 0) → (∃ (x : ℝ), x > 1 ∧ x^2 - x ≤ 0)
  (S1 ∧ S2 ∧ S3 ∧ ¬S4) := by
    sorry

end problem_statements_l1837_183797


namespace even_function_has_a_equal_2_l1837_183751

noncomputable def f (a x : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem even_function_has_a_equal_2 (a : ℝ) :
  (∀ x : ℝ, f a (-x) = f a x) → a = 2 :=
sorry

end even_function_has_a_equal_2_l1837_183751


namespace total_cost_price_l1837_183743

theorem total_cost_price (C O B : ℝ) 
    (hC : 1.25 * C = 8340) 
    (hO : 1.30 * O = 4675) 
    (hB : 1.20 * B = 3600) : 
    C + O + B = 13268.15 := 
by 
    sorry

end total_cost_price_l1837_183743


namespace initially_collected_oranges_l1837_183741

-- Define the conditions from the problem
def oranges_eaten_by_father : ℕ := 2
def oranges_mildred_has_now : ℕ := 75

-- Define the proof problem (statement)
theorem initially_collected_oranges :
  (oranges_mildred_has_now + oranges_eaten_by_father = 77) :=
by 
  -- proof goes here
  sorry

end initially_collected_oranges_l1837_183741


namespace find_value_given_conditions_l1837_183709

def equation_result (x y k : ℕ) : Prop := x ^ y + y ^ x = k

theorem find_value_given_conditions (y : ℕ) (k : ℕ) : 
  equation_result 2407 y k := 
by 
  sorry

end find_value_given_conditions_l1837_183709


namespace volume_of_resulting_shape_l1837_183747

-- Define the edge lengths
def edge_length (original : ℕ) (small : ℕ) := original = 5 ∧ small = 1

-- Define the volume of a cube
def volume (a : ℕ) : ℕ := a * a * a

-- State the proof problem
theorem volume_of_resulting_shape : ∀ (original small : ℕ), edge_length original small → 
  volume original - (5 * volume small) = 120 := by
  sorry

end volume_of_resulting_shape_l1837_183747


namespace total_product_l1837_183789

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12 
  else if n % 2 = 0 then 4 
  else 0 

def allie_rolls : List ℕ := [2, 6, 3, 1, 6]
def betty_rolls : List ℕ := [4, 6, 3, 5]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem total_product : total_points allie_rolls * total_points betty_rolls = 1120 := sorry

end total_product_l1837_183789


namespace new_rectangle_dimensions_l1837_183766

theorem new_rectangle_dimensions (l w : ℕ) (h_l : l = 12) (h_w : w = 10) :
  ∃ l' w' : ℕ, l' = l ∧ w' = w / 2 ∧ l' = 12 ∧ w' = 5 :=
by
  sorry

end new_rectangle_dimensions_l1837_183766


namespace vector_CD_l1837_183708

-- Define the vector space and the vectors a and b
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D : V)
variables (a b : V)

-- Define the conditions
def is_on_line (D A B : V) := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (D = t • A + (1 - t) • B)
def da_eq_2bd (D A B : V) := (A - D) = 2 • (D - B)

-- Define the triangle ABC and the specific vectors CA and CB
variables (CA := C - A) (CB := C - B)
variable (H1 : is_on_line D A B)
variable (H2 : da_eq_2bd D A B)
variable (H3 : CA = a)
variable (H4 : CB = b)

-- Prove the conclusion
theorem vector_CD (H1 : is_on_line D A B) (H2 : da_eq_2bd D A B)
  (H3 : CA = a) (H4 : CB = b) : 
  (C - D) = (1/3 : ℝ) • a + (2/3 : ℝ) • b :=
sorry

end vector_CD_l1837_183708


namespace john_average_speed_l1837_183750

theorem john_average_speed:
  (∃ J : ℝ, Carla_speed = 35 ∧ Carla_time = 3 ∧ John_time = 3.5 ∧ J * John_time = Carla_speed * Carla_time) →
  (∃ J : ℝ, J = 30) :=
by
  -- Given Variables
  let Carla_speed : ℝ := 35
  let Carla_time : ℝ := 3
  let John_time : ℝ := 3.5
  -- Proof goal
  sorry

end john_average_speed_l1837_183750


namespace gcd_108_450_l1837_183729

theorem gcd_108_450 : Nat.gcd 108 450 = 18 :=
by
  sorry

end gcd_108_450_l1837_183729


namespace smallest_value_of_n_l1837_183703

theorem smallest_value_of_n :
  ∃ o y m n : ℕ, 10 * o = 16 * y ∧ 16 * y = 18 * m ∧ 18 * m = 18 * n ∧ n = 40 := 
sorry

end smallest_value_of_n_l1837_183703


namespace find_b_given_a_l1837_183757

-- Definitions based on the conditions
def varies_inversely (a b : ℝ) (k : ℝ) : Prop := a * b = k
def k_value : ℝ := 400

-- The proof statement
theorem find_b_given_a (a b : ℝ) (h1 : varies_inversely 800 0.5 k_value) (h2 : a = 3200) : b = 0.125 :=
by
  -- skipped proof
  sorry

end find_b_given_a_l1837_183757


namespace broken_line_count_l1837_183756

def num_right_moves : ℕ := 9
def num_up_moves : ℕ := 10
def total_moves : ℕ := num_right_moves + num_up_moves
def num_broken_lines : ℕ := Nat.choose total_moves num_right_moves

theorem broken_line_count : num_broken_lines = 92378 := by
  sorry

end broken_line_count_l1837_183756


namespace manager_wage_l1837_183768

variable (M D C : ℝ)

def condition1 : Prop := D = M / 2
def condition2 : Prop := C = 1.25 * D
def condition3 : Prop := C = M - 3.1875

theorem manager_wage (h1 : condition1 M D) (h2 : condition2 D C) (h3 : condition3 M C) : M = 8.5 :=
by
  sorry

end manager_wage_l1837_183768


namespace square_grid_21_max_moves_rectangle_grid_20_21_max_moves_l1837_183760

-- Define the problem conditions.
def square_grid (n : Nat) : Prop := true
def rectangle_grid (m n : Nat) : Prop := true

-- Define the grid size for square and rectangle.
def square_grid_21 := square_grid 21
def rectangle_grid_20_21 := rectangle_grid 20 21

-- Define the proof problem to find maximum moves.
theorem square_grid_21_max_moves : ∃ m : Nat, m = 3 :=
  sorry

theorem rectangle_grid_20_21_max_moves : ∃ m : Nat, m = 4 :=
  sorry

end square_grid_21_max_moves_rectangle_grid_20_21_max_moves_l1837_183760


namespace monochromatic_triangle_in_K17_l1837_183737

theorem monochromatic_triangle_in_K17 :
  ∀ (V : Type) (E : V → V → ℕ), (∀ v1 v2, 0 ≤ E v1 v2 ∧ E v1 v2 < 3) →
    (∃ (v1 v2 v3 : V), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ (E v1 v2 = E v2 v3 ∧ E v2 v3 = E v1 v3)) :=
by
  intro V E Hcl
  sorry

end monochromatic_triangle_in_K17_l1837_183737


namespace quadratic_has_real_root_l1837_183763

theorem quadratic_has_real_root {b : ℝ} :
  ∃ x : ℝ, x^2 + b*x + 25 = 0 ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_has_real_root_l1837_183763


namespace part1_part2_l1837_183733

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem part1 (h : 1 - a = -1) : a = 2 ∧ 
                                  (∀ x : ℝ, x < Real.log 2 → (Real.exp x - 2) < 0) ∧ 
                                  (∀ x : ℝ, x > Real.log 2 → (Real.exp x - 2) > 0) :=
by
  sorry

theorem part2 (h1 : x1 < Real.log 2) (h2 : x2 > Real.log 2) (h3 : f 2 x1 = f 2 x2) : 
  x1 + x2 < 2 * Real.log 2 :=
by
  sorry

end part1_part2_l1837_183733


namespace lattice_points_in_region_l1837_183706

theorem lattice_points_in_region : ∃! n : ℕ, n = 14 ∧ ∀ (x y : ℤ), (y = |x| ∨ y = -x^2 + 4) ∧ (-2 ≤ x ∧ x ≤ 1) → 
  (y = -x^2 + 4 ∧ y = |x|) :=
sorry

end lattice_points_in_region_l1837_183706


namespace cost_to_color_pattern_l1837_183771

-- Define the basic properties of the squares
def square_side_length : ℕ := 4
def number_of_squares : ℕ := 4
def unit_cost (num_overlapping_squares : ℕ) : ℕ := num_overlapping_squares

-- Define the number of unit squares overlapping by different amounts
def unit_squares_overlapping_by_4 : ℕ := 1
def unit_squares_overlapping_by_3 : ℕ := 6
def unit_squares_overlapping_by_2 : ℕ := 12
def unit_squares_overlapping_by_1 : ℕ := 18

-- Calculate the total cost
def total_cost : ℕ :=
  unit_cost 4 * unit_squares_overlapping_by_4 +
  unit_cost 3 * unit_squares_overlapping_by_3 +
  unit_cost 2 * unit_squares_overlapping_by_2 +
  unit_cost 1 * unit_squares_overlapping_by_1

-- Statement to prove
theorem cost_to_color_pattern : total_cost = 64 := 
  sorry

end cost_to_color_pattern_l1837_183771


namespace problem1_problem2_l1837_183787

theorem problem1 : 4 * Real.sqrt 2 + Real.sqrt 8 - Real.sqrt 18 = 3 * Real.sqrt 2 := by
  sorry

theorem problem2 : Real.sqrt (4 / 3) / Real.sqrt (7 / 3) * Real.sqrt (7 / 5) = 2 * Real.sqrt 5 / 5 := by
  sorry

end problem1_problem2_l1837_183787


namespace bus_seat_capacity_l1837_183707

theorem bus_seat_capacity (x : ℕ) : 15 * x + (15 - 3) * x + 11 = 92 → x = 3 :=
by
  sorry

end bus_seat_capacity_l1837_183707


namespace x_pow_4_minus_inv_x_pow_4_eq_727_l1837_183742

theorem x_pow_4_minus_inv_x_pow_4_eq_727 (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 :=
by
  sorry

end x_pow_4_minus_inv_x_pow_4_eq_727_l1837_183742


namespace problem_solution_l1837_183724

-- Definitions based on conditions
def p (a b : ℝ) : Prop := a > b → a^2 > b^2
def neg_p (a b : ℝ) : Prop := a > b → a^2 ≤ b^2
def disjunction (p q : Prop) : Prop := p ∨ q
def suff_but_not_nec (x : ℝ) : Prop := x > 2 → x > 1 ∧ ¬(x > 1 → x > 2)
def congruent_triangles (T1 T2 : Prop) : Prop := T1 → T2
def neg_congruent_triangles (T1 T2 : Prop) : Prop := ¬(T1 → T2)

-- Mathematical problem as Lean statements
theorem problem_solution :
  ( (∀ a b : ℝ, p a b = (a > b → a^2 > b^2) ∧ neg_p a b = (a > b → a^2 ≤ b^2)) ∧
    (∀ p q : Prop, (disjunction p q) = false → p = false ∧ q = false) ∧
    (∀ x : ℝ, suff_but_not_nec x = (x > 2 → x > 1 ∧ ¬(x > 1 → x > 2))) ∧
    (∀ T1 T2 : Prop, (neg_congruent_triangles T1 T2) = true ↔ ¬(T1 → T2)) ) →
  ( (∀ a b : ℝ, neg_p a b = (a > b → a^2 ≤ b^2)) ∧
    (∀ p q : Prop, (disjunction p q) = false → p = false ∧ q = false) ∧
    (∀ x : ℝ, suff_but_not_nec x = (x > 2 → x > 1 ∧ ¬(x > 1 → x > 2))) ∧
    (∀ T1 T2 : Prop, (neg_congruent_triangles T1 T2) = false) ) :=
sorry

end problem_solution_l1837_183724


namespace avg_of_7_consecutive_integers_l1837_183775

theorem avg_of_7_consecutive_integers (a b : ℕ) (h1 : b = (a + (a+1) + (a+2) + (a+3) + (a+4)) / 5) : 
  (b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5) + (b + 6)) / 7 = a + 5 := 
  sorry

end avg_of_7_consecutive_integers_l1837_183775


namespace college_students_count_l1837_183736

theorem college_students_count (girls boys : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ)
(h_ratio : ratio_boys = 6) (h_ratio_girls : ratio_girls = 5)
(h_girls : girls = 200)
(h_boys : boys = ratio_boys * (girls / ratio_girls)) :
  boys + girls = 440 := by
  sorry

end college_students_count_l1837_183736


namespace evaluate_expression_l1837_183721

theorem evaluate_expression : (2 + 1 / (2 + 1 / (2 + 2)) = 22 / 9) := by
    sorry

end evaluate_expression_l1837_183721


namespace equal_cost_at_20_minutes_l1837_183782

/-- Define the cost functions for each telephone company -/
def united_cost (m : ℝ) : ℝ := 11 + 0.25 * m
def atlantic_cost (m : ℝ) : ℝ := 12 + 0.20 * m
def global_cost (m : ℝ) : ℝ := 13 + 0.15 * m

/-- Prove that at 20 minutes, the cost is the same for all three companies -/
theorem equal_cost_at_20_minutes : 
  united_cost 20 = atlantic_cost 20 ∧ atlantic_cost 20 = global_cost 20 :=
by
  sorry

end equal_cost_at_20_minutes_l1837_183782


namespace percentage_problem_l1837_183738

theorem percentage_problem (x : ℝ) (h : 0.255 * x = 153) : 0.678 * x = 406.8 :=
by
  sorry

end percentage_problem_l1837_183738


namespace k_at_27_l1837_183753

noncomputable def h (x : ℝ) : ℝ := x^3 - 2 * x + 1

theorem k_at_27 (k : ℝ → ℝ)
    (hk_cubic : ∀ x, ∃ a b c, k x = a * x^3 + b * x^2 + c * x)
    (hk_at_0 : k 0 = 1)
    (hk_roots : ∀ a b c, (h a = 0) → (h b = 0) → (h c = 0) → 
                 ∃ (p q r: ℝ), k (p^3) = 0 ∧ k (q^3) = 0 ∧ k (r^3) = 0) :
    k 27 = -704 :=
sorry

end k_at_27_l1837_183753


namespace geo_sequence_ratio_l1837_183745

theorem geo_sequence_ratio
  (a_n : ℕ → ℝ)
  (S_n : ℕ → ℝ)
  (q : ℝ)
  (hq1 : q = 1 → S_8 = 8 * a_n 0 ∧ S_4 = 4 * a_n 0 ∧ S_8 = 2 * S_4)
  (hq2 : q ≠ 1 → S_8 = 2 * S_4 → false)
  (hS : ∀ n, S_n n = a_n 0 * (1 - q^n) / (1 - q))
  (h_condition : S_8 = 2 * S_4) :
  a_n 2 / a_n 0 = 1 := sorry

end geo_sequence_ratio_l1837_183745


namespace sum_fourth_powers_const_l1837_183755

-- Define the vertices of the square
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A (a : ℝ) : Point := {x := a, y := 0}
def B (a : ℝ) : Point := {x := 0, y := a}
def C (a : ℝ) : Point := {x := -a, y := 0}
def D (a : ℝ) : Point := {x := 0, y := -a}

-- Define distance squared between two points
def dist_sq (P Q : Point) : ℝ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

-- Circle centered at origin
def on_circle (P : Point) (r : ℝ) : Prop :=
  P.x ^ 2 + P.y ^ 2 = r ^ 2

-- The main theorem
theorem sum_fourth_powers_const (a r : ℝ) (P : Point) (h : on_circle P r) :
  let AP_sq := dist_sq P (A a)
  let BP_sq := dist_sq P (B a)
  let CP_sq := dist_sq P (C a)
  let DP_sq := dist_sq P (D a)
  (AP_sq ^ 2 + BP_sq ^ 2 + CP_sq ^ 2 + DP_sq ^ 2) = 4 * (r^4 + a^4 + 4 * a^2 * r^2) :=
by
  sorry

end sum_fourth_powers_const_l1837_183755


namespace remainder_of_large_number_l1837_183727

theorem remainder_of_large_number (N : ℕ) (hN : N = 123456789012): 
  N % 360 = 108 :=
by
  have h1 : N % 4 = 0 := by 
    sorry
  have h2 : N % 9 = 3 := by 
    sorry
  have h3 : N % 10 = 2 := by
    sorry
  sorry

end remainder_of_large_number_l1837_183727


namespace functional_relationship_max_annual_profit_l1837_183779

namespace FactoryProfit

-- Definitions of conditions
def fixed_annual_investment : ℕ := 100
def unit_investment : ℕ := 1
def sales_revenue (x : ℕ) : ℕ :=
  if x > 20 then 260 
  else 33 * x - x^2

def annual_profit (x : ℕ) : ℤ :=
  let revenue := sales_revenue x
  let total_investment := fixed_annual_investment + x
  revenue - total_investment

-- Statements to prove
theorem functional_relationship (x : ℕ) (hx : x > 0) :
  annual_profit x =
  if x ≤ 20 then
    (-x^2 : ℤ) + 32 * x - 100
  else
    160 - x :=
by sorry

theorem max_annual_profit : 
  ∃ x, annual_profit x = 144 ∧
  ∀ y, annual_profit y ≤ 144 :=
by sorry

end FactoryProfit

end functional_relationship_max_annual_profit_l1837_183779


namespace smaller_prime_is_x_l1837_183744

theorem smaller_prime_is_x (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (h1 : x + y = 36) (h2 : 4 * x + y = 87) : x = 17 :=
  sorry

end smaller_prime_is_x_l1837_183744


namespace frustum_radius_l1837_183790

theorem frustum_radius (r : ℝ) (h1 : ∃ r1 r2, r1 = r 
                                  ∧ r2 = 3 * r 
                                  ∧ r1 * 2 * π * 3 = r2 * 2 * π
                                  ∧ (lateral_area = 84 * π)) (h2 : slant_height = 3) : 
  r = 7 :=
sorry

end frustum_radius_l1837_183790


namespace james_daily_soda_consumption_l1837_183781

theorem james_daily_soda_consumption
  (N_p : ℕ) -- number of packs
  (S_p : ℕ) -- sodas per pack
  (S_i : ℕ) -- initial sodas
  (D : ℕ)  -- days in a week
  (h1 : N_p = 5)
  (h2 : S_p = 12)
  (h3 : S_i = 10)
  (h4 : D = 7) : 
  (N_p * S_p + S_i) / D = 10 := 
by 
  sorry

end james_daily_soda_consumption_l1837_183781


namespace a_eq_b_if_conditions_l1837_183717

theorem a_eq_b_if_conditions (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b := 
sorry

end a_eq_b_if_conditions_l1837_183717


namespace complex_sum_eighth_power_l1837_183730

noncomputable def compute_sum_eighth_power 
(ζ1 ζ2 ζ3 : ℂ) 
(h1 : ζ1 + ζ2 + ζ3 = 2) 
(h2 : ζ1^2 + ζ2^2 + ζ3^2 = 5) 
(h3 : ζ1^3 + ζ2^3 + ζ3^3 = 8) : ℂ :=
  ζ1^8 + ζ2^8 + ζ3^8

theorem complex_sum_eighth_power 
(ζ1 ζ2 ζ3 : ℂ) 
(h1 : ζ1 + ζ2 + ζ3 = 2) 
(h2 : ζ1^2 + ζ2^2 + ζ3^2 = 5) 
(h3 : ζ1^3 + ζ2^3 + ζ3^3 = 8) : 
  compute_sum_eighth_power ζ1 ζ2 ζ3 h1 h2 h3 = 451.625 :=
sorry

end complex_sum_eighth_power_l1837_183730


namespace negation_of_universal_l1837_183728

theorem negation_of_universal :
  ¬ (∀ x : ℝ, 2 * x ^ 2 + x - 1 ≤ 0) ↔ ∃ x : ℝ, 2 * x ^ 2 + x - 1 > 0 := 
by 
  sorry

end negation_of_universal_l1837_183728


namespace three_legged_reptiles_count_l1837_183767

noncomputable def total_heads : ℕ := 300
noncomputable def total_legs : ℕ := 798

def number_of_three_legged_reptiles (b r m : ℕ) : Prop :=
  b + r + m = total_heads ∧
  2 * b + 3 * r + 4 * m = total_legs

theorem three_legged_reptiles_count (b r m : ℕ) (h : number_of_three_legged_reptiles b r m) :
  r = 102 :=
sorry

end three_legged_reptiles_count_l1837_183767


namespace probability_reach_origin_from_3_3_l1837_183769

noncomputable def P : ℕ → ℕ → ℚ
| 0, 0 => 1
| x+1, 0 => 0
| 0, y+1 => 0
| x+1, y+1 => (1/3) * P x (y+1) + (1/3) * P (x+1) y + (1/3) * P x y

theorem probability_reach_origin_from_3_3 : P 3 3 = 1 / 27 := by
  sorry

end probability_reach_origin_from_3_3_l1837_183769


namespace lewis_speed_is_90_l1837_183731

noncomputable def david_speed : ℝ := 50 -- mph
noncomputable def distance_chennai_hyderabad : ℝ := 350 -- miles
noncomputable def distance_meeting_point : ℝ := 250 -- miles

theorem lewis_speed_is_90 :
  ∃ L : ℝ, 
    (∀ t : ℝ, david_speed * t = distance_meeting_point) →
    (∀ t : ℝ, L * t = (distance_chennai_hyderabad + (distance_meeting_point - distance_chennai_hyderabad))) →
    L = 90 :=
by
  sorry

end lewis_speed_is_90_l1837_183731


namespace smallest_lambda_inequality_l1837_183719

theorem smallest_lambda_inequality (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  x * y * (x^2 + y^2) + y * z * (y^2 + z^2) + z * x * (z^2 + x^2) ≤ (1 / 8) * (x + y + z)^4 :=
sorry

end smallest_lambda_inequality_l1837_183719


namespace pq_sum_is_38_l1837_183770

theorem pq_sum_is_38
  (p q : ℝ)
  (h_root : ∀ x, (2 * x^2) + (p * x) + q = 0 → x = 2 * Complex.I - 3 ∨ x = -2 * Complex.I - 3)
  (h_p_q : ∀ a b : ℂ, a + b = -p / 2 ∧ a * b = q / 2 → p = 12 ∧ q = 26) :
  p + q = 38 :=
sorry

end pq_sum_is_38_l1837_183770


namespace pages_revised_twice_theorem_l1837_183761

noncomputable def pages_revised_twice (total_pages : ℕ) (cost_per_page : ℕ) (revision_cost_per_page : ℕ) 
                                      (pages_revised_once : ℕ) (total_cost : ℕ) : ℕ :=
  let pages_revised_twice := (total_cost - (total_pages * cost_per_page + pages_revised_once * revision_cost_per_page)) 
                             / (revision_cost_per_page * 2)
  pages_revised_twice

theorem pages_revised_twice_theorem : 
  pages_revised_twice 100 10 5 30 1350 = 20 :=
by
  unfold pages_revised_twice
  norm_num

end pages_revised_twice_theorem_l1837_183761


namespace original_denominator_is_18_l1837_183739

variable (d : ℕ)

theorem original_denominator_is_18
  (h1 : ∃ (d : ℕ), (3 + 7) / (d + 7) = 2 / 5) :
  d = 18 := 
sorry

end original_denominator_is_18_l1837_183739


namespace condition_neither_sufficient_nor_necessary_l1837_183780

noncomputable def f (x a : ℝ) : ℝ := x^3 - x + a
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - 1

def condition (a : ℝ) : Prop := a^2 - a = 0

theorem condition_neither_sufficient_nor_necessary
  (a : ℝ) :
  ¬(condition a → (∀ x : ℝ, f' x ≥ 0)) ∧ ¬((∀ x : ℝ, f' x ≥ 0) → condition a) :=
by
  sorry -- Proof is omitted as per the prompt

end condition_neither_sufficient_nor_necessary_l1837_183780


namespace sunland_more_plates_than_moonland_l1837_183786

theorem sunland_more_plates_than_moonland :
  let sunland_plates := 26^5 * 10^2
  let moonland_plates := 26^3 * 10^3
  sunland_plates - moonland_plates = 1170561600 := by
  sorry

end sunland_more_plates_than_moonland_l1837_183786


namespace mixed_number_sum_l1837_183778

theorem mixed_number_sum : 
  (4/5 + 9 * 4/5 + 99 * 4/5 + 999 * 4/5 + 9999 * 4/5 + 1 = 11111) := by
  sorry

end mixed_number_sum_l1837_183778


namespace line_segments_property_l1837_183785

theorem line_segments_property (L : List (ℝ × ℝ)) :
  L.length = 50 →
  (∃ S : List (ℝ × ℝ), S.length = 8 ∧ ∃ x : ℝ, ∀ seg ∈ S, seg.fst ≤ x ∧ x ≤ seg.snd) ∨
  (∃ T : List (ℝ × ℝ), T.length = 8 ∧ ∀ seg1 ∈ T, ∀ seg2 ∈ T, seg1 ≠ seg2 → seg1.snd < seg2.fst ∨ seg2.snd < seg1.fst) :=
by
  -- Theorem proof placeholder
  sorry

end line_segments_property_l1837_183785


namespace g_self_inverse_if_one_l1837_183714

variables (f : ℝ → ℝ) (symm_about : ∀ x, f (f x) = x - 1)

def g (b : ℝ) (x : ℝ) : ℝ := f (x + b)

theorem g_self_inverse_if_one (b : ℝ) :
  (∀ x, g f b (g f b x) = x) ↔ b = 1 := 
by
  sorry

end g_self_inverse_if_one_l1837_183714


namespace quadratic_has_one_real_solution_l1837_183792

theorem quadratic_has_one_real_solution (k : ℝ) (hk : (x + 5) * (x + 2) = k + 3 * x) : k = 6 → ∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x :=
by
  sorry

end quadratic_has_one_real_solution_l1837_183792


namespace fruit_basket_combinations_l1837_183712

namespace FruitBasket

def apples := 3
def oranges := 8
def min_apples := 1
def min_oranges := 1

theorem fruit_basket_combinations : 
  (apples + 1 - min_apples) * (oranges + 1 - min_oranges) = 36 := by
  sorry

end FruitBasket

end fruit_basket_combinations_l1837_183712


namespace problem_l1837_183726

def f (x : ℝ) : ℝ := (x^4 + 2*x^3 + 4*x - 5) ^ 2004 + 2004

theorem problem (x : ℝ) (h : x = Real.sqrt 3 - 1) : f x = 2005 :=
by
  sorry

end problem_l1837_183726


namespace face_value_of_shares_l1837_183759

/-- A company pays a 12.5% dividend to its investors. -/
def div_rate := 0.125

/-- An investor gets a 25% return on their investment. -/
def roi_rate := 0.25

/-- The investor bought the shares at Rs. 20 each. -/
def purchase_price := 20

theorem face_value_of_shares (FV : ℝ) (div_rate : ℝ) (roi_rate : ℝ) (purchase_price : ℝ) 
  (h1 : purchase_price * roi_rate = div_rate * FV) : FV = 40 :=
by sorry

end face_value_of_shares_l1837_183759


namespace sum_of_powers_eq_123_l1837_183711

section

variables {a b : Real}

-- Conditions provided in the problem
axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7

-- Define the theorem to be proved
theorem sum_of_powers_eq_123 : a^10 + b^10 = 123 :=
sorry

end

end sum_of_powers_eq_123_l1837_183711


namespace handshakes_total_l1837_183796

theorem handshakes_total :
  let team_size := 6
  let referees := 3
  (team_size * team_size) + (2 * team_size * referees) = 72 :=
by
  sorry

end handshakes_total_l1837_183796


namespace min_translation_phi_l1837_183704

theorem min_translation_phi (φ : ℝ) (hφ : φ > 0) : 
  (∃ k : ℤ, φ = (π / 3) - k * π) → φ = π / 3 := 
by 
  sorry

end min_translation_phi_l1837_183704


namespace fixed_point_exists_l1837_183791

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(a * (x + 1)) - 3

theorem fixed_point_exists (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = -1 :=
by
  -- Sorry for skipping the proof
  sorry

end fixed_point_exists_l1837_183791


namespace prove_a_star_b_l1837_183705

variable (a b : ℤ)
variable (h1 : a + b = 12)
variable (h2 : a * b = 35)

def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

theorem prove_a_star_b : star a b = 12 / 35 :=
by
  sorry

end prove_a_star_b_l1837_183705
