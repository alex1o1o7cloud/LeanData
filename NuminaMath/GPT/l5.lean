import Mathlib

namespace shaded_area_proof_l5_5195

noncomputable def shaded_area (side_length : ℝ) (radius_factor : ℝ) : ℝ :=
  let square_area := side_length * side_length
  let radius := radius_factor * side_length
  let circle_area := Real.pi * (radius * radius)
  square_area - circle_area

theorem shaded_area_proof : shaded_area 8 0.6 = 64 - 23.04 * Real.pi :=
by sorry

end shaded_area_proof_l5_5195


namespace sin_double_alpha_pi_over_6_l5_5046

open Real

theorem sin_double_alpha_pi_over_6 (α : ℝ) 
  (h : sin (α - π / 6) = 1 / 3) : sin (2 * α + π / 6) = 7 / 9 :=
sorry

end sin_double_alpha_pi_over_6_l5_5046


namespace constant_term_in_expansion_l5_5157

noncomputable def P (x : ℕ) : ℕ := x^4 + 2 * x + 7
noncomputable def Q (x : ℕ) : ℕ := 2 * x^3 + 3 * x^2 + 10

theorem constant_term_in_expansion :
  (P 0) * (Q 0) = 70 := 
sorry

end constant_term_in_expansion_l5_5157


namespace solve_for_F_l5_5599

variable (S W F : ℝ)

def condition1 (S W : ℝ) : Prop := S = W / 3
def condition2 (W F : ℝ) : Prop := W = F + 60
def condition3 (S W F : ℝ) : Prop := S + W + F = 150

theorem solve_for_F (S W F : ℝ) (h1 : condition1 S W) (h2 : condition2 W F) (h3 : condition3 S W F) : F = 52.5 :=
sorry

end solve_for_F_l5_5599


namespace dried_mushrooms_weight_l5_5222

theorem dried_mushrooms_weight (fresh_weight : ℝ) (water_content_fresh : ℝ) (water_content_dried : ℝ) :
  fresh_weight = 22 →
  water_content_fresh = 0.90 →
  water_content_dried = 0.12 →
  ∃ x : ℝ, x = 2.5 :=
by
  intros h1 h2 h3
  have hw_fresh : ℝ := fresh_weight * water_content_fresh
  have dry_material_fresh : ℝ := fresh_weight - hw_fresh
  have dry_material_dried : ℝ := 1.0 - water_content_dried
  have hw_dried := dry_material_fresh / dry_material_dried
  use hw_dried
  sorry

end dried_mushrooms_weight_l5_5222


namespace polynomial_coeff_sum_l5_5253

/-- 
Given that the product of the polynomials (4x^2 - 6x + 5)(8 - 3x) can be written as
ax^3 + bx^2 + cx + d, prove that 9a + 3b + c + d = 19.
-/
theorem polynomial_coeff_sum :
  ∃ a b c d : ℝ, 
  (∀ x : ℝ, (4 * x^2 - 6 * x + 5) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) ∧
  9 * a + 3 * b + c + d = 19 :=
sorry

end polynomial_coeff_sum_l5_5253


namespace point_in_second_quadrant_condition_l5_5694

theorem point_in_second_quadrant_condition (a : ℤ)
  (h1 : 3 * a - 9 < 0)
  (h2 : 10 - 2 * a > 0)
  (h3 : |3 * a - 9| = |10 - 2 * a|):
  (a + 2) ^ 2023 - 1 = 0 := 
sorry

end point_in_second_quadrant_condition_l5_5694


namespace no_snuggly_numbers_l5_5212

def isSnuggly (n : Nat) : Prop :=
  ∃ (a b : Nat), 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧ 
    n = 10 * a + b ∧ 
    n = a + b^3 + 5

theorem no_snuggly_numbers : 
  ¬ ∃ n : Nat, 10 ≤ n ∧ n < 100 ∧ isSnuggly n :=
by
  sorry

end no_snuggly_numbers_l5_5212


namespace principal_made_mistake_l5_5900

-- Definitions based on given conditions
def students_per_class (x : ℤ) : Prop := x > 0
def total_students (x : ℤ) : ℤ := 2 * x
def non_failing_grades (y : ℤ) : ℤ := y
def failing_grades (y : ℤ) : ℤ := y + 11
def total_grades (x y : ℤ) : Prop := total_students x = non_failing_grades y + failing_grades y

-- Proposition stating the principal made a mistake
theorem principal_made_mistake (x y : ℤ) (hx : students_per_class x) : ¬ total_grades x y :=
by
  -- Assume the proof for the hypothesis is required here
  sorry

end principal_made_mistake_l5_5900


namespace prove_correct_operation_l5_5364

def correct_operation (a b : ℕ) : Prop :=
  (a^3 * a^2 ≠ a^6) ∧
  ((a * b^2)^2 = a^2 * b^4) ∧
  (a^10 / a^5 ≠ a^2) ∧
  (a^2 + a ≠ a^3)

theorem prove_correct_operation (a b : ℕ) : correct_operation a b :=
by {
  sorry
}

end prove_correct_operation_l5_5364


namespace sequence_a_n_l5_5834

theorem sequence_a_n (a : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n > 0 → (n^2 + n) * (a (n + 1) - a n) = 2) :
  a 20 = 29 / 10 :=
by
  sorry

end sequence_a_n_l5_5834


namespace maximize_sum_probability_l5_5154

theorem maximize_sum_probability :
  ∀ (l : List ℤ), l = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] →
  (∃ n ∈ l, n = 6 ∧ (∀ x ∈ (l.erase n), (∃ y ∈ (l.erase n), x ≠ y ∧ x + y = 12) ↔  (∃ y ∈ l, x ≠ y ∧ x + y = 12))) :=
by
  intro l
  intro hl
  use 6
  split
  · rw hl
    simp
  · intro x
    intro hx
    split
    · intro h
      exists sorry
    · intro h'
      exists sorry

end maximize_sum_probability_l5_5154


namespace slope_parallel_to_original_line_l5_5947

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l5_5947


namespace common_difference_value_l5_5052

-- Define the arithmetic sequence and the sum of the first n terms
def sum_of_arithmetic_sequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

-- Define the given condition in terms of the arithmetic sequence
def given_condition (a1 d : ℚ) : Prop :=
  (sum_of_arithmetic_sequence a1 d 2017) / 2017 - (sum_of_arithmetic_sequence a1 d 17) / 17 = 100

-- Prove the common difference d is 1/10 given the condition
theorem common_difference_value (a1 d : ℚ) :
  given_condition a1 d → d = 1/10 :=
by
  sorry

end common_difference_value_l5_5052


namespace part1_part2_l5_5832

theorem part1 (m : ℝ) (P : ℝ × ℝ) : (P = (3*m - 6, m + 1)) → (P.1 = 0) → (P = (0, 3)) :=
by
  sorry

theorem part2 (m : ℝ) (A P : ℝ × ℝ) : A = (1, -2) → (P = (3*m - 6, m + 1)) → (P.2 = A.2) → (P = (-15, -2)) :=
by
  sorry

end part1_part2_l5_5832


namespace prime_sum_mod_eighth_l5_5172

theorem prime_sum_mod_eighth (p1 p2 p3 p4 p5 p6 p7 p8 : ℕ) 
  (h₁ : p1 = 2) 
  (h₂ : p2 = 3) 
  (h₃ : p3 = 5) 
  (h₄ : p4 = 7) 
  (h₅ : p5 = 11) 
  (h₆ : p6 = 13) 
  (h₇ : p7 = 17) 
  (h₈ : p8 = 19) : 
  ((p1 + p2 + p3 + p4 + p5 + p6 + p7) % p8) = 1 :=
by
  sorry

end prime_sum_mod_eighth_l5_5172


namespace remainder_of_sum_of_primes_mod_eighth_prime_l5_5160

def sum_first_seven_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13 + 17

def eighth_prime : ℕ := 19

theorem remainder_of_sum_of_primes_mod_eighth_prime : sum_first_seven_primes % eighth_prime = 1 := by
  sorry

end remainder_of_sum_of_primes_mod_eighth_prime_l5_5160


namespace min_voters_for_tall_24_l5_5086

/-
There are 105 voters divided into 5 districts, each district divided into 7 sections, with each section having 3 voters.
A section is won by a majority vote. A district is won by a majority of sections. The contest is won by a majority of districts.
Tall won the contest. Prove that the minimum number of voters who could have voted for Tall is 24.
-/
noncomputable def min_voters_for_tall (total_voters districts sections voters_per_section : ℕ) (sections_needed_to_win_district districts_needed_to_win_contest : ℕ) : ℕ :=
  let voters_needed_per_section := voters_per_section / 2 + 1
  sections_needed_to_win_district * districts_needed_to_win_contest * voters_needed_per_section

theorem min_voters_for_tall_24 :
  min_voters_for_tall 105 5 7 3 4 3 = 24 :=
sorry

end min_voters_for_tall_24_l5_5086


namespace split_trout_equally_l5_5461

-- Definitions for conditions
def Total_trout : ℕ := 18
def People : ℕ := 2

-- Statement we need to prove
theorem split_trout_equally 
(H1 : Total_trout = 18)
(H2 : People = 2) : 
  (Total_trout / People = 9) :=
by
  sorry

end split_trout_equally_l5_5461


namespace solve_for_a_l5_5237

variable (a u : ℝ)

def eq1 := (3 / a) + (1 / u) = 7 / 2
def eq2 := (2 / a) - (3 / u) = 6

theorem solve_for_a (h1 : eq1 a u) (h2 : eq2 a u) : a = 2 / 3 := 
by
  sorry

end solve_for_a_l5_5237


namespace functional_relationship_remaining_oil_after_4_hours_l5_5648

-- Define the initial conditions and the functional form
def initial_oil : ℝ := 50
def consumption_rate : ℝ := 8
def remaining_oil (t : ℝ) : ℝ := initial_oil - consumption_rate * t

-- Prove the functional relationship and the remaining oil after 4 hours
theorem functional_relationship : ∀ (t : ℝ), remaining_oil t = 50 - 8 * t :=
by intros t
   exact rfl

theorem remaining_oil_after_4_hours : remaining_oil 4 = 18 :=
by simp [remaining_oil]
   norm_num
   sorry

end functional_relationship_remaining_oil_after_4_hours_l5_5648


namespace min_value_of_a_sq_plus_b_sq_l5_5109

theorem min_value_of_a_sq_plus_b_sq {a b t : ℝ} (h : 2 * a + 3 * b = t) :
  ∃ a b : ℝ, (2 * a + 3 * b = t) ∧ (a^2 + b^2 = (13 * t^2) / 169) :=
by
  sorry

end min_value_of_a_sq_plus_b_sq_l5_5109


namespace arithmetic_sequence_ninth_term_l5_5886

theorem arithmetic_sequence_ninth_term
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29) :
  a + 8 * d = 35 :=
sorry

end arithmetic_sequence_ninth_term_l5_5886


namespace find_water_bottles_l5_5373

def water_bottles (W A : ℕ) :=
  A = W + 6 ∧ W + A = 54 → W = 24

theorem find_water_bottles (W A : ℕ) (h1 : A = W + 6) (h2 : W + A = 54) : W = 24 :=
by sorry

end find_water_bottles_l5_5373


namespace right_triangle_legs_l5_5733

theorem right_triangle_legs (m r x y : ℝ) 
  (h1 : m^2 = x^2 + y^2) 
  (h2 : r = (x + y - m) / 2) 
  (h3 : r ≤ m * (Real.sqrt 2 - 1) / 2) : 
  (x = (2 * r + m + Real.sqrt (m^2 - 4 * r^2 - 4 * r * m)) / 2) ∧ 
  (y = (2 * r + m - Real.sqrt (m^2 - 4 * r^2 - 4 * r * m)) / 2) :=
by 
  sorry

end right_triangle_legs_l5_5733


namespace sasha_study_more_l5_5732

theorem sasha_study_more (d_wkdy : List ℤ) (d_wknd : List ℤ) (h_wkdy : d_wkdy = [5, -5, 15, 25, -15]) (h_wknd : d_wknd = [30, 30]) :
  (d_wkdy.sum + d_wknd.sum) / 7 = 12 := by
  sorry

end sasha_study_more_l5_5732


namespace original_price_l5_5502

variable (x : ℝ)

-- Condition 1: Selling at 60% of the original price results in a 20 yuan loss
def condition1 : Prop := 0.6 * x + 20 = x * 0.8 - 15

-- The goal is to prove that the original price is 175 yuan under the given conditions
theorem original_price (h : condition1 x) : x = 175 :=
sorry

end original_price_l5_5502


namespace arithmetic_sequence_30th_term_l5_5346

-- Defining the initial term and the common difference of the arithmetic sequence
def a : ℕ := 3
def d : ℕ := 4

-- Defining the general formula for the n-th term of the arithmetic sequence
def a_n (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem stating that the 30th term of the given sequence is 119
theorem arithmetic_sequence_30th_term : a_n 30 = 119 := 
sorry

end arithmetic_sequence_30th_term_l5_5346


namespace functional_equation_solution_l5_5476

theorem functional_equation_solution {f : ℝ → ℝ} (h : ∀ x ≠ 1, (x - 1) * f (x + 1) - f x = x) :
    ∀ x, f x = 1 + 2 * x :=
by
  sorry

end functional_equation_solution_l5_5476


namespace final_statement_l5_5008

variable (x : ℝ)

def seven_elevenths_of_five_thirteenths_eq_48 (x : ℝ) :=
  (7/11 : ℝ) * (5/13 : ℝ) * x = 48

def solve_for_x (x : ℝ) : Prop :=
  seven_elevenths_of_five_thirteenths_eq_48 x → x = 196

def calculate_315_percent_of_x (x : ℝ) : Prop :=
  solve_for_x x → 3.15 * x = 617.4

theorem final_statement : calculate_315_percent_of_x x :=
sorry  -- Proof omitted

end final_statement_l5_5008


namespace integer_solution_l5_5054

theorem integer_solution (a b : ℤ) (h : 6 * a * b = 9 * a - 10 * b + 303) : a + b = 15 :=
sorry

end integer_solution_l5_5054


namespace predicted_holiday_shoppers_l5_5136

-- Conditions
def packages_per_bulk_box : Nat := 25
def every_third_shopper_buys_package : Nat := 3
def bulk_boxes_ordered : Nat := 5

-- Number of predicted holiday shoppers
theorem predicted_holiday_shoppers (pbb : packages_per_bulk_box = 25)
                                   (etsbp : every_third_shopper_buys_package = 3)
                                   (bbo : bulk_boxes_ordered = 5) :
  (bulk_boxes_ordered * packages_per_bulk_box * every_third_shopper_buys_package) = 375 :=
by 
  -- Proof steps can be added here
  sorry

end predicted_holiday_shoppers_l5_5136


namespace jane_spent_more_on_ice_cream_l5_5840

-- Definitions based on the conditions
def ice_cream_cone_cost : ℕ := 5
def pudding_cup_cost : ℕ := 2
def ice_cream_cones_bought : ℕ := 15
def pudding_cups_bought : ℕ := 5

-- The mathematically equivalent proof statement
theorem jane_spent_more_on_ice_cream : 
  (ice_cream_cones_bought * ice_cream_cone_cost - pudding_cups_bought * pudding_cup_cost) = 65 := 
by
  sorry

end jane_spent_more_on_ice_cream_l5_5840


namespace one_percent_as_decimal_l5_5430

theorem one_percent_as_decimal : (1 / 100 : ℝ) = 0.01 := by
  sorry

end one_percent_as_decimal_l5_5430


namespace sum_of_ages_today_l5_5862

variable (RizaWas25WhenSonBorn : ℕ) (SonCurrentAge : ℕ) (SumOfAgesToday : ℕ)

theorem sum_of_ages_today (h1 : RizaWas25WhenSonBorn = 25) (h2 : SonCurrentAge = 40) : SumOfAgesToday = 105 :=
by
  sorry

end sum_of_ages_today_l5_5862


namespace exterior_angle_of_regular_octagon_l5_5439

def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)
def interior_angle (s : ℕ) (n : ℕ) : ℕ := sum_of_interior_angles n / s
def exterior_angle (ia : ℕ) : ℕ := 180 - ia

theorem exterior_angle_of_regular_octagon : 
    exterior_angle (interior_angle 8 8) = 45 := 
by 
  sorry

end exterior_angle_of_regular_octagon_l5_5439


namespace slope_of_parallel_line_l5_5962

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l5_5962


namespace volume_region_between_concentric_spheres_l5_5488

open Real

theorem volume_region_between_concentric_spheres (r1 r2 : ℝ) (h_r1 : r1 = 4) (h_r2 : r2 = 8) :
  (4 / 3 * π * r2^3 - 4 / 3 * π * r1^3) = 1792 / 3 * π :=
by
  sorry

end volume_region_between_concentric_spheres_l5_5488


namespace ohara_triple_example_l5_5749

noncomputable def is_ohara_triple (a b x : ℕ) : Prop := 
  (Real.sqrt a + Real.sqrt b = x)

theorem ohara_triple_example : 
  is_ohara_triple 49 16 11 ∧ 11 ≠ 100 / 5 := 
by
  sorry

end ohara_triple_example_l5_5749


namespace number_of_female_students_l5_5144

theorem number_of_female_students (M F : ℕ) (h1 : F = M + 6) (h2 : M + F = 82) : F = 44 :=
by
  sorry

end number_of_female_students_l5_5144


namespace calculate_expression_l5_5533

theorem calculate_expression : ∀ x y : ℝ, x = 7 → y = 3 → (x - y) ^ 2 * (x + y) = 160 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end calculate_expression_l5_5533


namespace percentage_of_40_l5_5801

theorem percentage_of_40 (P : ℝ) (h1 : 8/100 * 24 = 1.92) (h2 : P/100 * 40 + 1.92 = 5.92) : P = 10 :=
sorry

end percentage_of_40_l5_5801


namespace find_b_l5_5544

theorem find_b (a b c : ℚ) :
  -- Condition from the problem, equivalence of polynomials for all x
  ((4 : ℚ) * x^2 - 2 * x + 5 / 2) * (a * x^2 + b * x + c) =
    12 * x^4 - 8 * x^3 + 15 * x^2 - 5 * x + 5 / 2 →
  -- Given we found that a = 3 from the solution
  a = 3 →
  -- We need to prove that b = -1/2
  b = -1 / 2 :=
sorry

end find_b_l5_5544


namespace parallel_line_slope_l5_5939

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℚ, m = 1 / 2 :=
by {
  use 1 / 2,
  sorry
}

end parallel_line_slope_l5_5939


namespace perpendicular_lines_a_value_l5_5565

theorem perpendicular_lines_a_value (a : ℝ) :
  (a * (a + 2) = -1) → a = -1 :=
by
  intro h
  sorry

end perpendicular_lines_a_value_l5_5565


namespace tangents_form_rectangle_l5_5429

-- Define the first ellipse
def ellipse1 (a b x y : ℝ) : Prop := x^2 / a^4 + y^2 / b^4 = 1

-- Define the second ellipse
def ellipse2 (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define conjugate diameters through lines
def conjugate_diameters (a b m : ℝ) : Prop := True -- (You might want to further define what conjugate diameters imply here)

-- Prove the main statement
theorem tangents_form_rectangle
  (a b m : ℝ)
  (x1 y1 x2 y2 k1 k2 : ℝ)
  (h1 : ellipse1 a b x1 y1)
  (h2 : ellipse1 a b x2 y2)
  (h3 : ellipse2 a b x1 y1)
  (h4 : ellipse2 a b x2 y2)
  (conj1 : conjugate_diameters a b m)
  (tangent_slope1 : k1 = -b^2 / a^2 * (1 / m))
  (conj2 : conjugate_diameters a b (-b^4/a^4 * 1/m))
  (tangent_slope2 : k2 = -b^4 / a^4 * (1 / (-b^4/a^4 * (1/m))))
: k1 * k2 = -1 :=
sorry

end tangents_form_rectangle_l5_5429


namespace B_take_time_4_hours_l5_5012

theorem B_take_time_4_hours (A_rate B_rate C_rate D_rate : ℚ) :
  (A_rate = 1 / 4) →
  (B_rate + C_rate = 1 / 2) →
  (A_rate + C_rate = 1 / 2) →
  (D_rate = 1 / 8) →
  (A_rate + B_rate + D_rate = 1 / 1.6) →
  (B_rate = 1 / 4) ∧ (1 / B_rate = 4) :=
by
  sorry

end B_take_time_4_hours_l5_5012


namespace complex_quadrant_l5_5820

theorem complex_quadrant (a b : ℝ) (i : ℂ) (h : i^2 = -1) (h_eq : 1 + a * i = (b + i) * (1 + i)) : 
  (a - b * i).re > 0 ∧ (a - b * i).im < 0 :=
by
  have h1 : 1 + a * i = (b - 1) + (b + 1) * i := by sorry
  have h2 : a = b + 1 := by sorry
  have h3 : b - 1 = 1 := by sorry
  have h4 : b = 2 := by sorry
  have h5 : a = 3 := by sorry
  have h6 : (a - b * i).re = 3 := by sorry
  have h7 : (a - b * i).im = -2 := by sorry
  exact ⟨by linarith, by linarith⟩

end complex_quadrant_l5_5820


namespace unique_zero_point_condition1_unique_zero_point_condition2_l5_5240

noncomputable def func (x a b : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2 + b

theorem unique_zero_point_condition1 {a b : ℝ} (h1 : 1 / 2 < a) (h2 : a ≤ Real.exp 2 / 2) (h3 : b > 2 * a) :
  ∃! x, func x a b = 0 :=
sorry

theorem unique_zero_point_condition2 {a b : ℝ} (h1 : 0 < a) (h2 : a < 1 / 2) (h3 : b ≤ 2 * a) :
  ∃! x, func x a b = 0 :=
sorry

end unique_zero_point_condition1_unique_zero_point_condition2_l5_5240


namespace distance_sum_l5_5428

noncomputable def Cartesian_curve (x y : ℝ) := (x - 2)^2 + 4 * y^2 - 4 = 0
noncomputable def line_l (t : ℝ) := (1 + (real.sqrt 3 / 2) * t, (1 / 2) * t)
noncomputable def transformed_C (x y : ℝ) := (x - 2)^2 + y^2 - 4 = 0

theorem distance_sum (t1 t2 : ℝ)
  (h1 : (1 + (real.sqrt 3 / 2) * t1 - 2)^2 + ((1 / 2) * t1)^2 - 4 = 0)
  (h2 : (1 + (real.sqrt 3 / 2) * t2 - 2)^2 + ((1 / 2) * t2)^2 - 4 = 0)
  (h_sum : t1 + t2 = real.sqrt 3)
  (h_prod : t1 * t2 = -3) :
  real.abs t1 + real.abs t2 = real.sqrt 15 :=
by sorry

end distance_sum_l5_5428


namespace line_equation_l5_5048

-- Definitions according to the conditions
def point_P := (3, 4)
def slope_angle_l := 90

-- Statement of the theorem to prove
theorem line_equation (l : ℝ → ℝ) (h1 : l point_P.1 = point_P.2) (h2 : slope_angle_l = 90) :
  ∃ k : ℝ, k = 3 ∧ ∀ x, l x = 3 - x :=
sorry

end line_equation_l5_5048


namespace proof_problem_l5_5455

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def given_function (f : ℝ → ℝ) : Prop :=
  odd_function f ∧ f (-3) = -2

theorem proof_problem (f : ℝ → ℝ) (h : given_function f) : f 3 + f 0 = -2 :=
by sorry

end proof_problem_l5_5455


namespace sum_infinite_series_l5_5397

theorem sum_infinite_series : ∑' n : ℕ, (4 * (n + 1) - 3) / (3 ^ (n + 1)) = 3 / 2 := by
    sorry

end sum_infinite_series_l5_5397


namespace prime_sum_mod_eighth_l5_5173

theorem prime_sum_mod_eighth (p1 p2 p3 p4 p5 p6 p7 p8 : ℕ) 
  (h₁ : p1 = 2) 
  (h₂ : p2 = 3) 
  (h₃ : p3 = 5) 
  (h₄ : p4 = 7) 
  (h₅ : p5 = 11) 
  (h₆ : p6 = 13) 
  (h₇ : p7 = 17) 
  (h₈ : p8 = 19) : 
  ((p1 + p2 + p3 + p4 + p5 + p6 + p7) % p8) = 1 :=
by
  sorry

end prime_sum_mod_eighth_l5_5173


namespace line_through_P_origin_line_through_P_perpendicular_to_l3_l5_5063

-- Define lines l1, l2, l3
def l1 (x y : ℝ) := 3 * x + 4 * y - 2 = 0
def l2 (x y : ℝ) := 2 * x + y + 2 = 0
def l3 (x y : ℝ) := x - 2 * y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Prove the equations of the lines passing through P
theorem line_through_P_origin : ∃ A B C : ℝ, A * -2 + B * 2 + C = 0 ∧ A * 0 + B * 0 + C = 0 ∧ A = 1 ∧ B = 1 ∧ C = 0 :=
by sorry

theorem line_through_P_perpendicular_to_l3 : ∃ A B C : ℝ, A * -2 + B * 2 + C = 0 ∧ A * P.1 + B * P.2 + C = 0 ∧ A = 2 ∧ B = 1 ∧ C = 2 :=
by sorry

end line_through_P_origin_line_through_P_perpendicular_to_l3_l5_5063


namespace length_of_paving_stone_l5_5485

theorem length_of_paving_stone (courtyard_length courtyard_width : ℝ)
  (num_paving_stones : ℕ) (paving_stone_width : ℝ) (total_area : ℝ)
  (paving_stone_length : ℝ) : 
  courtyard_length = 70 ∧ courtyard_width = 16.5 ∧ num_paving_stones = 231 ∧ paving_stone_width = 2 ∧ total_area = courtyard_length * courtyard_width ∧ total_area = num_paving_stones * paving_stone_length * paving_stone_width → 
  paving_stone_length = 2.5 :=
by
  sorry

end length_of_paving_stone_l5_5485


namespace slope_parallel_to_original_line_l5_5945

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l5_5945


namespace GMAT_scores_ratio_l5_5301

variables (u v w : ℝ)

theorem GMAT_scores_ratio
  (h1 : u - w = (u + v + w) / 3)
  (h2 : u - v = 2 * (v - w))
  : v / u = 4 / 7 :=
sorry

end GMAT_scores_ratio_l5_5301


namespace spending_difference_is_65_l5_5839

-- Definitions based on conditions
def ice_cream_cones : ℕ := 15
def pudding_cups : ℕ := 5
def ice_cream_cost_per_unit : ℝ := 5
def pudding_cost_per_unit : ℝ := 2

-- The solution requires the calculation of the total cost and the difference
def total_ice_cream_cost : ℝ := ice_cream_cones * ice_cream_cost_per_unit
def total_pudding_cost : ℝ := pudding_cups * pudding_cost_per_unit
def spending_difference : ℝ := total_ice_cream_cost - total_pudding_cost

-- Theorem statement proving the difference is 65
theorem spending_difference_is_65 : spending_difference = 65 := by
  -- The proof is omitted with sorry
  sorry

end spending_difference_is_65_l5_5839


namespace intersection_eq_N_l5_5113

def U := Set ℝ                                        -- Universal set U = ℝ
def M : Set ℝ := {x | x ≥ 0}                         -- Set M = {x | x ≥ 0}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}                 -- Set N = {x | 0 ≤ x ≤ 1}

theorem intersection_eq_N : M ∩ N = N := by
  sorry

end intersection_eq_N_l5_5113


namespace road_repair_completion_time_l5_5267

theorem road_repair_completion_time :
  (∀ (r : ℝ), 1 = 45 * r * 3) → (∀ (t : ℝ), (30 * (1 / (3 * 45))) * t = 1) → t = 4.5 :=
by
  intros rate_eq time_eq
  sorry

end road_repair_completion_time_l5_5267


namespace total_surface_area_l5_5475

theorem total_surface_area (r h : ℝ) (pi : ℝ) (area_base : ℝ) (curved_area_hemisphere : ℝ) (lateral_area_cylinder : ℝ) :
  (pi * r^2 = 144 * pi) ∧ (h = 10) ∧ (curved_area_hemisphere = 2 * pi * r^2) ∧ (lateral_area_cylinder = 2 * pi * r * h) →
  (curved_area_hemisphere + lateral_area_cylinder + area_base = 672 * pi) :=
by
  sorry

end total_surface_area_l5_5475


namespace product_of_consecutive_numbers_l5_5743

theorem product_of_consecutive_numbers (n : ℕ) (k : ℕ) (h₁: n * (n + 1) * (n + 2) = 210) (h₂: n + (n + 1) = 11) : k = 3 :=
by
  sorry

end product_of_consecutive_numbers_l5_5743


namespace ring_toss_total_l5_5616

theorem ring_toss_total (money_per_day : ℕ) (days : ℕ) (total_money : ℕ) 
(h1 : money_per_day = 140) (h2 : days = 3) : total_money = 420 :=
by
  sorry

end ring_toss_total_l5_5616


namespace seated_men_l5_5143

def passengers : Nat := 48
def fraction_of_women : Rat := 2/3
def fraction_of_men_standing : Rat := 1/8

theorem seated_men (men women standing seated : Nat) 
  (h1 : women = passengers * fraction_of_women)
  (h2 : men = passengers - women)
  (h3 : standing = men * fraction_of_men_standing)
  (h4 : seated = men - standing) :
  seated = 14 := by
  sorry

end seated_men_l5_5143


namespace slope_parallel_to_original_line_l5_5949

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l5_5949


namespace min_u_condition_l5_5811

-- Define the function u and the condition
def u (x y : ℝ) : ℝ := x^2 + 4 * x + y^2 - 2 * y

def condition (x y : ℝ) : Prop := 2 * x + y ≥ 1

-- The statement we want to prove
theorem min_u_condition : ∃ (x y : ℝ), condition x y ∧ u x y = -9/5 := 
by
  sorry

end min_u_condition_l5_5811


namespace triangle_inradius_is_2_5_l5_5742

variable (A : ℝ) (p : ℝ) (r : ℝ)

def triangle_has_given_inradius (A p : ℝ) : Prop :=
  A = r * p / 2

theorem triangle_inradius_is_2_5 (h₁ : A = 25) (h₂ : p = 20) :
  triangle_has_given_inradius A p r → r = 2.5 := sorry

end triangle_inradius_is_2_5_l5_5742


namespace tangent_parallel_l5_5575

noncomputable def f (x : ℝ) : ℝ := x^4 - x

theorem tangent_parallel (P : ℝ × ℝ) (hP : P.1 = 1) (hP_cond : P.2 = f P.1) 
  (tangent_parallel : ∀ x, deriv f x = 3) : P = (1, 0) := 
by 
  have h_deriv : deriv f 1 = 4 * 1^3 - 1 := by sorry
  have slope_eq : deriv f (P.1) = 3 := by sorry
  have solve_a : P.1 = 1 := by sorry
  have solve_b : f 1 = 0 := by sorry
  exact sorry

end tangent_parallel_l5_5575


namespace intersect_complement_eq_l5_5819

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {1, 3, 6}
def comp_B : Set ℕ := U \ B

theorem intersect_complement_eq :
  A ∩ comp_B = {4, 5} := by
  sorry

end intersect_complement_eq_l5_5819


namespace product_of_18396_and_9999_l5_5803

theorem product_of_18396_and_9999 : 18396 * 9999 = 183962604 :=
by
  sorry

end product_of_18396_and_9999_l5_5803


namespace ellipse_equation_ellipse_chord_length_l5_5555

-- Conditions
variable {F1 : (ℝ × ℝ) := (2, 0)}
variable {directrix : ℝ -> Prop := λ x, x = 8}
variable {eccentricity : ℝ := 1 / 2}

-- Question 1: Prove the equation of the ellipse
theorem ellipse_equation (x y : ℝ) (h : sqrt ((x - 2)^2 + y^2) / abs (8 - x) = 1 / 2) :
  x^2 / 16 + y^2 / 12 = 1 :=
by
  sorry

-- Question 2: Prove the length of the chord cut from the ellipse by a given line
theorem ellipse_chord_length (x1 y1 x2 y2 : ℝ) 
  (h1 : y1 = x1 + 2)
  (h2 : y2 = x2 + 2)
  (h3 : x1^2 / 16 + y1^2 / 12 = 1)
  (h4 : x2^2 / 16 + y2^2 / 12 = 1) :
  dist (x1, y1) (x2, y2) = 48 / 7 :=
by
  sorry

end ellipse_equation_ellipse_chord_length_l5_5555


namespace problem_part1_problem_part2_l5_5421

-- Define the sequences and conditions
variable {a b : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {T : ℕ → ℕ}
variable {d q : ℕ}
variable {b_initial : ℕ}

axiom geom_seq (n : ℕ) : b n = b_initial * q^n
axiom arith_seq (n : ℕ) : a n = a 1 + (n - 1) * d
axiom sum_seq (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Problem conditions
axiom cond_geom_seq : b_initial = 2
axiom cond_geom_b2_b3 : b 2 + b 3 = 12
axiom cond_geom_ratio : q > 0
axiom cond_relation_b3_a4 : b 3 = a 4 - 2 * a 1
axiom cond_sum_S_11_b4 : S 11 = 11 * b 4

-- Theorem statement
theorem problem_part1 :
  (a n = 3 * n - 2) ∧ (b n = 2 ^ n) :=
  sorry

theorem problem_part2 :
  (T n = (3 * n - 2) / 3 * 4^(n + 1) + 8 / 3) :=
  sorry

end problem_part1_problem_part2_l5_5421


namespace thirtieth_term_value_l5_5352

-- Define the initial term and common difference
def initial_term : ℤ := 3
def common_difference : ℤ := 4

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ :=
  initial_term + (n - 1) * common_difference

-- Theorem stating the value of the 30th term
theorem thirtieth_term_value : nth_term 30 = 119 :=
by sorry

end thirtieth_term_value_l5_5352


namespace largest_n_unique_k_l5_5908

theorem largest_n_unique_k : ∃ n : ℕ, n = 24 ∧ (∃! k : ℕ, 
  3 / 7 < n / (n + k: ℤ) ∧ n / (n + k: ℤ) < 8 / 19) :=
by
  sorry

end largest_n_unique_k_l5_5908


namespace eight_people_lineup_two_windows_l5_5477

theorem eight_people_lineup_two_windows :
  (2 ^ 8) * (Nat.factorial 8) = 10321920 := by
  sorry

end eight_people_lineup_two_windows_l5_5477


namespace men_seated_count_l5_5139

theorem men_seated_count (total_passengers : ℕ) (two_thirds_women : total_passengers * 2 / 3 = women)
                         (one_eighth_standing : total_passengers / 3 / 8 = standing_men) :
  total_passengers = 48 →
  women = 32 →
  standing_men = 2 →
  men_seated = (total_passengers - women) - standing_men →
  men_seated = 14 :=
by
  intros
  sorry

end men_seated_count_l5_5139


namespace slope_of_parallel_line_l5_5958

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l5_5958


namespace cost_of_song_book_l5_5268

-- Define the costs as constants
def cost_trumpet : ℝ := 149.16
def cost_music_tool : ℝ := 9.98
def total_spent : ℝ := 163.28

-- Define the statement to prove
theorem cost_of_song_book : total_spent - (cost_trumpet + cost_music_tool) = 4.14 := 
by
  sorry

end cost_of_song_book_l5_5268


namespace hadassah_painting_time_l5_5689

noncomputable def time_to_paint_all_paintings (time_small_paintings time_large_paintings time_additional_small_paintings time_additional_large_paintings : ℝ) : ℝ :=
  time_small_paintings + time_large_paintings + time_additional_small_paintings + time_additional_large_paintings

theorem hadassah_painting_time :
  let time_small_paintings := 6
  let time_large_paintings := 8
  let time_per_small_painting := 6 / 12 -- = 0.5
  let time_per_large_painting := 8 / 6 -- ≈ 1.33
  let time_additional_small_paintings := 15 * time_per_small_painting -- = 7.5
  let time_additional_large_paintings := 10 * time_per_large_painting -- ≈ 13.3
  time_to_paint_all_paintings time_small_paintings time_large_paintings time_additional_small_paintings time_additional_large_paintings = 34.8 :=
by
  sorry

end hadassah_painting_time_l5_5689


namespace intersection_of_sets_l5_5273

-- Defining the sets as given in the conditions
def setM : Set ℝ := { x | (x + 1) * (x - 3) ≤ 0 }
def setN : Set ℝ := { x | 1 < x ∧ x < 4 }

-- Statement to prove
theorem intersection_of_sets :
  { x | (x + 1) * (x - 3) ≤ 0 } ∩ { x | 1 < x ∧ x < 4 } = { x | 1 < x ∧ x ≤ 3 } := by
sorry

end intersection_of_sets_l5_5273


namespace operation_1_and_2004_l5_5213

def operation (m n : ℕ) : ℕ :=
  if m = 1 ∧ n = 1 then 2
  else if m = 1 ∧ n > 1 then 2 + 3 * (n - 1)
  else 0 -- handle other cases generically, although specifics are not given

theorem operation_1_and_2004 : operation 1 2004 = 6011 :=
by
  unfold operation
  sorry

end operation_1_and_2004_l5_5213


namespace price_of_stock_l5_5261

-- Defining the conditions
def income : ℚ := 650
def dividend_rate : ℚ := 10
def investment : ℚ := 6240

-- Defining the face value calculation from income and dividend rate
def face_value (i : ℚ) (d_rate : ℚ) : ℚ := (i * 100) / d_rate

-- Calculating the price of the stock
def stock_price (inv : ℚ) (fv : ℚ) : ℚ := (inv / fv) * 100

-- Main theorem to be proved
theorem price_of_stock : stock_price investment (face_value income dividend_rate) = 96 := by
  sorry

end price_of_stock_l5_5261


namespace probability_no_adjacent_stands_l5_5536

-- Definitions based on the conditions
def fair_coin_flip : ℕ := 2 -- Each person can flip one of two possible outcomes (head or tail).

-- The main theorem stating the probability
theorem probability_no_adjacent_stands : 
  let total_outcomes := fair_coin_flip ^ 8 in -- Total number of possible sequences
  let favorable_outcomes := 47 in -- Number of valid sequences where no two adjacent people stand
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 47 / 256 :=
by
  sorry

end probability_no_adjacent_stands_l5_5536


namespace total_amount_l5_5709

variable (Brad Josh Doug : ℝ)

axiom h1 : Josh = 2 * Brad
axiom h2 : Josh = (3 / 4) * Doug
axiom h3 : Doug = 32

theorem total_amount : Brad + Josh + Doug = 68 := by
  sorry

end total_amount_l5_5709


namespace digit_inequality_l5_5612

theorem digit_inequality : ∃ (n : ℕ), n = 9 ∧ ∀ (d : ℕ), d < 10 → (2 + d / 10 + 5 / 1000 > 2 + 5 / 1000) → d > 0 :=
by
  sorry

end digit_inequality_l5_5612


namespace problem_statement_l5_5678

theorem problem_statement (a b : ℝ) (h1 : 1/a < 1/b) (h2 : 1/b < 0) :
  (a + b < a * b) ∧ ¬(a^2 > b^2) ∧ ¬(a < b) ∧ (b/a + a/b > 2) := by
  sorry

end problem_statement_l5_5678


namespace new_percentage_of_managers_is_98_l5_5746

def percentage_of_managers (initial_employees : ℕ) (initial_percentage_managers : ℕ) (managers_leaving : ℕ) : ℕ :=
  let initial_managers := initial_percentage_managers * initial_employees / 100
  let remaining_managers := initial_managers - managers_leaving
  let remaining_employees := initial_employees - managers_leaving
  (remaining_managers * 100) / remaining_employees

theorem new_percentage_of_managers_is_98 :
  percentage_of_managers 500 99 250 = 98 :=
by
  sorry

end new_percentage_of_managers_is_98_l5_5746


namespace solve_for_y_l5_5695

theorem solve_for_y (x y : ℤ) (h1 : x + y = 260) (h2 : x - y = 200) : y = 30 := by
  sorry

end solve_for_y_l5_5695


namespace sum_of_fractions_l5_5791

theorem sum_of_fractions (a b c d : ℚ) (ha : a = 2 / 5) (hb : b = 3 / 8) :
  (a + b = 31 / 40) :=
by
  sorry

end sum_of_fractions_l5_5791


namespace insects_per_group_correct_l5_5785

-- Define the numbers of insects collected by boys and girls
def boys_insects : ℕ := 200
def girls_insects : ℕ := 300
def total_insects : ℕ := boys_insects + girls_insects

-- Define the number of groups
def groups : ℕ := 4

-- Define the expected number of insects per group using total insects and groups
def insects_per_group : ℕ := total_insects / groups

-- Prove that each group gets 125 insects
theorem insects_per_group_correct : insects_per_group = 125 :=
by
  -- The proof is omitted (just setting up the theorem statement)
  sorry

end insects_per_group_correct_l5_5785


namespace sum_of_first_seven_primes_mod_eighth_prime_l5_5176

theorem sum_of_first_seven_primes_mod_eighth_prime :
  (2 + 3 + 5 + 7 + 11 + 13 + 17) % 19 = 1 :=
by
  sorry

end sum_of_first_seven_primes_mod_eighth_prime_l5_5176


namespace parallel_slope_l5_5956

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l5_5956


namespace isabella_hair_growth_l5_5266

def initial_hair_length : ℝ := 18
def final_hair_length : ℝ := 24
def hair_growth : ℝ := final_hair_length - initial_hair_length

theorem isabella_hair_growth : hair_growth = 6 := by
  sorry

end isabella_hair_growth_l5_5266


namespace fish_remaining_when_discovered_l5_5147

def start_fish := 60
def fish_eaten_per_day := 2
def days_two_weeks := 2 * 7
def fish_added_after_two_weeks := 8
def days_one_week := 7

def fish_after_two_weeks (start: ℕ) (eaten_per_day: ℕ) (days: ℕ) (added: ℕ): ℕ :=
  start - eaten_per_day * days + added

def fish_after_three_weeks (fish_after_two_weeks: ℕ) (eaten_per_day: ℕ) (days: ℕ): ℕ :=
  fish_after_two_weeks - eaten_per_day * days

theorem fish_remaining_when_discovered :
  (fish_after_three_weeks (fish_after_two_weeks start_fish fish_eaten_per_day days_two_weeks fish_added_after_two_weeks) fish_eaten_per_day days_one_week) = 26 := 
by {
  sorry
}

end fish_remaining_when_discovered_l5_5147


namespace find_value_of_expression_l5_5736

theorem find_value_of_expression (x : ℝ) (h : 5 * x^2 + 4 = 3 * x + 9) : (10 * x - 3)^2 = 109 := 
sorry

end find_value_of_expression_l5_5736


namespace solve_for_x_l5_5724

theorem solve_for_x (x : ℝ) : 3^(3 * x) = Real.sqrt 81 -> x = 2 / 3 :=
by
  sorry

end solve_for_x_l5_5724


namespace incorrect_description_l5_5285

-- Conditions
def population_size : ℕ := 2000
def sample_size : ℕ := 150

-- Main Statement
theorem incorrect_description : ¬ (sample_size = 150) := 
by sorry

end incorrect_description_l5_5285


namespace dyslexian_alphabet_size_l5_5867

theorem dyslexian_alphabet_size (c v : ℕ) (h1 : (c * v * c * v * c + v * c * v * c * v) = 4800) : c + v = 12 :=
by
  sorry

end dyslexian_alphabet_size_l5_5867


namespace find_triplet_l5_5035

theorem find_triplet (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y) ^ 2 + 3 * x + y + 1 = z ^ 2 → y = x ∧ z = 2 * x + 1 :=
by
  sorry

end find_triplet_l5_5035


namespace sum_of_remainders_and_smallest_n_l5_5112

theorem sum_of_remainders_and_smallest_n (n : ℕ) (h : n % 20 = 11) :
    (n % 4 + n % 5 = 4) ∧ (∃ (k : ℕ), k > 2 ∧ n = 20 * k + 11 ∧ n > 50) := by
  sorry

end sum_of_remainders_and_smallest_n_l5_5112


namespace slope_of_parallel_line_l5_5909

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l5_5909


namespace sum_of_squares_of_tom_rates_l5_5626

theorem sum_of_squares_of_tom_rates :
  ∃ r b k : ℕ, 3 * r + 4 * b + 2 * k = 104 ∧
               3 * r + 6 * b + 2 * k = 140 ∧
               r^2 + b^2 + k^2 = 440 :=
by
  sorry

end sum_of_squares_of_tom_rates_l5_5626


namespace tall_wins_min_voters_l5_5091

structure VotingSetup where
  total_voters : ℕ
  districts : ℕ
  sections_per_district : ℕ
  voters_per_section : ℕ
  voters_majority_in_section : ℕ
  districts_to_win : ℕ
  sections_to_win_district : ℕ

def contest_victory (setup : VotingSetup) (min_voters : ℕ) : Prop :=
  setup.total_voters = 105 ∧
  setup.districts = 5 ∧
  setup.sections_per_district = 7 ∧
  setup.voters_per_section = 3 ∧
  setup.voters_majority_in_section = 2 ∧
  setup.districts_to_win = 3 ∧
  setup.sections_to_win_district = 4 ∧
  min_voters = 24

theorem tall_wins_min_voters : ∃ min_voters, contest_victory ⟨105, 5, 7, 3, 2, 3, 4⟩ min_voters :=
by { use 24, sorry }

end tall_wins_min_voters_l5_5091


namespace product_of_possible_x_l5_5571

theorem product_of_possible_x : 
  (∀ x : ℚ, abs ((18 / x) + 4) = 3 → x = -18 ∨ x = -18 / 7) → 
  ((-18) * (-18 / 7) = 324 / 7) :=
by
  sorry

end product_of_possible_x_l5_5571


namespace units_digit_42_pow_4_add_24_pow_4_l5_5759

-- Define a function to get the units digit of a number.
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_42_pow_4_add_24_pow_4 : units_digit (42^4 + 24^4) = 2 := by
  sorry

end units_digit_42_pow_4_add_24_pow_4_l5_5759


namespace min_voters_for_Tall_victory_l5_5090

def total_voters := 105
def districts := 5
def sections_per_district := 7
def voters_per_section := 3
def sections_to_win_district := 4
def districts_to_win := 3
def sections_to_win := sections_to_win_district * districts_to_win
def min_voters_to_win_section := 2

theorem min_voters_for_Tall_victory : 
  (total_voters = 105) ∧ 
  (districts = 5) ∧ 
  (sections_per_district = 7) ∧ 
  (voters_per_section = 3) ∧ 
  (sections_to_win_district = 4) ∧ 
  (districts_to_win = 3) 
  → 
  min_voters_to_win_section * sections_to_win = 24 :=
by
  sorry
  
end min_voters_for_Tall_victory_l5_5090


namespace probability_within_two_units_of_origin_correct_l5_5778

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let square_area := 36
  let circle_area := 4 * Real.pi
  circle_area / square_area

theorem probability_within_two_units_of_origin_correct :
  probability_within_two_units_of_origin = Real.pi / 9 := by
  sorry

end probability_within_two_units_of_origin_correct_l5_5778


namespace brett_total_miles_l5_5389

def miles_per_hour : ℕ := 75
def hours_driven : ℕ := 12

theorem brett_total_miles : miles_per_hour * hours_driven = 900 := 
by 
  sorry

end brett_total_miles_l5_5389


namespace net_increase_in_bicycle_stock_l5_5277

-- Definitions for changes in stock over the three days
def net_change_friday : ℤ := 15 - 10
def net_change_saturday : ℤ := 8 - 12
def net_change_sunday : ℤ := 11 - 9

-- Total net increase in stock
def total_net_increase : ℤ := net_change_friday + net_change_saturday + net_change_sunday

-- Theorem statement
theorem net_increase_in_bicycle_stock : total_net_increase = 3 := by
  -- We would provide the detailed proof here.
  sorry

end net_increase_in_bicycle_stock_l5_5277


namespace rectangle_pairs_l5_5865

theorem rectangle_pairs :
  {p : ℕ × ℕ | 0 < p.1 ∧ 0 < p.2 ∧ p.1 * p.2 = 18} = {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} :=
by { sorry }

end rectangle_pairs_l5_5865


namespace find_divisor_l5_5800

noncomputable def divisor_of_nearest_divisible (a b : ℕ) (d : ℕ) : ℕ :=
  if h : b % d = 0 ∧ (b - a < d) then d else 0

theorem find_divisor (a b : ℕ) (d : ℕ) (h1 : b = 462) (h2 : a = 457)
  (h3 : b % d = 0) (h4 : b - a < d) :
  d = 5 :=
sorry

end find_divisor_l5_5800


namespace program_output_is_10_l5_5738

def final_value_of_A : ℤ :=
  let A := 2
  let A := A * 2
  let A := A + 6
  A

theorem program_output_is_10 : final_value_of_A = 10 := by
  sorry

end program_output_is_10_l5_5738


namespace total_balloons_l5_5519

theorem total_balloons (A_initial : Nat) (A_additional : Nat) (J_initial : Nat) 
  (h1 : A_initial = 3) (h2 : J_initial = 5) (h3 : A_additional = 2) : 
  A_initial + A_additional + J_initial = 10 := by
  sorry

end total_balloons_l5_5519


namespace wheel_horizontal_distance_l5_5199

noncomputable def wheel_radius : ℝ := 2
noncomputable def wheel_revolution_fraction : ℝ := 3 / 4
noncomputable def wheel_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem wheel_horizontal_distance :
  wheel_circumference wheel_radius * wheel_revolution_fraction = 3 * Real.pi :=
by
  sorry

end wheel_horizontal_distance_l5_5199


namespace total_infections_second_wave_l5_5876

theorem total_infections_second_wave (cases_per_day_first_wave : ℕ)
                                     (factor_increase : ℕ)
                                     (duration_weeks : ℕ)
                                     (days_per_week : ℕ) :
                                     cases_per_day_first_wave = 300 →
                                     factor_increase = 4 →
                                     duration_weeks = 2 →
                                     days_per_week = 7 →
                                     (duration_weeks * days_per_week) * (cases_per_day_first_wave + factor_increase * cases_per_day_first_wave) = 21000 :=
by 
  intros h1 h2 h3 h4
  sorry

end total_infections_second_wave_l5_5876


namespace num_ducks_l5_5079

variable (D G : ℕ)

theorem num_ducks (h1 : D + G = 8) (h2 : 2 * D + 4 * G = 24) : D = 4 := by
  sorry

end num_ducks_l5_5079


namespace total_amount_is_175_l5_5512

noncomputable def calc_total_amount (x : ℝ) (y : ℝ) (z : ℝ) : ℝ :=
x + y + z

theorem total_amount_is_175 (x y z : ℝ) 
  (h1 : y = 0.45 * x)
  (h2 : z = 0.30 * x)
  (h3 : y = 45) :
  calc_total_amount x y z = 175 :=
by
  -- sorry to skip the proof
  sorry

end total_amount_is_175_l5_5512


namespace volume_of_pyramid_SPQR_l5_5463

variable (P Q R S : Type)
variable (SP SQ SR : ℝ)
variable (is_perpendicular_SP_SQ : SP * SQ = 0)
variable (is_perpendicular_SQ_SR : SQ * SR = 0)
variable (is_perpendicular_SR_SP : SR * SP = 0)
variable (SP_eq_9 : SP = 9)
variable (SQ_eq_8 : SQ = 8)
variable (SR_eq_7 : SR = 7)

theorem volume_of_pyramid_SPQR : 
  ∃ V : ℝ, V = 84 := by
  -- Conditions and assumption
  sorry

end volume_of_pyramid_SPQR_l5_5463


namespace recurring_decimal_sum_l5_5033

theorem recurring_decimal_sum :
  let x := (4 / 33)
  let y := (34 / 99)
  x + y = (46 / 99) := by
  sorry

end recurring_decimal_sum_l5_5033


namespace crow_eating_time_l5_5183

/-- 
We are given that a crow eats a fifth of the total number of nuts in 6 hours.
We are to prove that it will take the crow 7.5 hours to finish a quarter of the nuts.
-/
theorem crow_eating_time (h : (1/5:ℚ) * t = 6) : (1/4) * t = 7.5 := 
by 
  -- Skipping the proof
  sorry

end crow_eating_time_l5_5183


namespace cubes_side_length_l5_5901

theorem cubes_side_length (s : ℝ) (h : 2 * (s * s + s * 2 * s + s * 2 * s) = 10) : s = 1 :=
by
  sorry

end cubes_side_length_l5_5901


namespace union_set_l5_5564

def M : Set ℝ := {x | -2 < x ∧ x < 1}
def P : Set ℝ := {x | -2 ≤ x ∧ x < 2}

theorem union_set : M ∪ P = {x : ℝ | -2 ≤ x ∧ x < 2} := by
  sorry

end union_set_l5_5564


namespace find_abc_solutions_l5_5218

theorem find_abc_solutions
    (a b c : ℕ)
    (h_pos : (a > 0) ∧ (b > 0) ∧ (c > 0))
    (h1 : a < b)
    (h2 : a < 4 * c)
    (h3 : b * c ^ 3 ≤ a * c ^ 3 + b) :
    ((a = 7) ∧ (b = 8) ∧ (c = 2)) ∨
    ((a = 1 ∨ a = 2 ∨ a = 3) ∧ (b > a) ∧ (c = 1)) :=
by
  sorry

end find_abc_solutions_l5_5218


namespace shaded_area_is_20_l5_5991

theorem shaded_area_is_20 (large_square_side : ℕ) (num_small_squares : ℕ) 
  (shaded_squares : ℕ) 
  (h1 : large_square_side = 10) (h2 : num_small_squares = 25) 
  (h3 : shaded_squares = 5) : 
  (large_square_side^2 / num_small_squares) * shaded_squares = 20 :=
by
  sorry

end shaded_area_is_20_l5_5991


namespace problem_solution_l5_5559

noncomputable def alpha : ℝ := (3 + Real.sqrt 13) / 2
noncomputable def beta  : ℝ := (3 - Real.sqrt 13) / 2

theorem problem_solution : 7 * alpha ^ 4 + 10 * beta ^ 3 = 1093 :=
by
  -- Prove roots relation
  have hr1 : alpha * alpha - 3 * alpha - 1 = 0 := by sorry
  have hr2 : beta * beta - 3 * beta - 1 = 0 := by sorry
  -- Proceed to prove the required expression
  sorry

end problem_solution_l5_5559


namespace parabola_equation_hyperbola_equation_l5_5636

-- Part 1: Prove the standard equation of the parabola given the directrix.
theorem parabola_equation (x y : ℝ) : x = -2 → y^2 = 8 * x := 
by
  -- Here we will include proof steps based on given conditions
  sorry

-- Part 2: Prove the standard equation of the hyperbola given center at origin, focus on the x-axis,
-- the given asymptotes, and its real axis length.
theorem hyperbola_equation (x y a b : ℝ) : 
  a = 1 → b = 2 → y = 2 * x ∨ y = -2 * x → x^2 - (y^2 / 4) = 1 :=
by
  -- Here we will include proof steps based on given conditions
  sorry

end parabola_equation_hyperbola_equation_l5_5636


namespace guesthouse_rolls_probability_l5_5983

theorem guesthouse_rolls_probability :
  let rolls := 12
  let guests := 3
  let types := 4
  let rolls_per_guest := 3
  let total_probability : ℚ := (12 / 12) * (9 / 11) * (6 / 10) * (3 / 9) *
                               (8 / 8) * (6 / 7) * (4 / 6) * (2 / 5) *
                               1
  let simplified_probability : ℚ := 24 / 1925
  total_probability = simplified_probability := sorry

end guesthouse_rolls_probability_l5_5983


namespace students_in_Johnsons_class_l5_5855

theorem students_in_Johnsons_class :
  ∀ (students_Finley : ℕ) (students_Johnson : ℕ),
    students_Finley = 24 →
    students_Johnson = (students_Finley / 2) + 10 →
    students_Johnson = 22 :=
by
  intros students_Finley students_Johnson hFinley hJohnson
  rw [hFinley, Nat.div_add_self 12 0],
  exact hJohnson

sorry

end students_in_Johnsons_class_l5_5855


namespace find_prices_find_min_money_spent_l5_5750

-- Define the prices of volleyball and soccer ball
def prices (pv ps : ℕ) : Prop :=
  pv + 20 = ps ∧ 500 / ps = 400 / pv

-- Define the quantity constraint
def quantity_constraint (a : ℕ) : Prop :=
  a ≥ 25 ∧ a < 50

-- Define the minimum amount spent problem
def min_money_spent (a : ℕ) (pv ps : ℕ) : Prop :=
  prices pv ps → quantity_constraint a → 100 * a + 80 * (50 - a) = 4500

-- Prove the price of each volleyball and soccer ball
theorem find_prices : ∃ (pv ps : ℕ), prices pv ps ∧ pv = 80 ∧ ps = 100 :=
by {sorry}

-- Prove the minimum amount of money spent
theorem find_min_money_spent : ∃ (a pv ps : ℕ), min_money_spent a pv ps :=
by {sorry}

end find_prices_find_min_money_spent_l5_5750


namespace digits_divisible_by_101_l5_5764

theorem digits_divisible_by_101 :
  ∃ x y : ℕ, x < 10 ∧ y < 10 ∧ (2013 * 100 + 10 * x + y) % 101 = 0 ∧ x = 9 ∧ y = 4 := by
  sorry

end digits_divisible_by_101_l5_5764


namespace identify_person_l5_5586

variable (Person : Type) (Tweedledum Tralyalya : Person)
variable (has_black_card : Person → Prop)
variable (statement_true : Person → Prop)
variable (statement_made_by : Person)

-- Condition: The statement made: "Either I am Tweedledum, or I have a card of a black suit in my pocket."
def statement (p : Person) : Prop := p = Tweedledum ∨ has_black_card p

-- Condition: Anyone with a black card making a true statement is not possible.
axiom black_card_truth_contradiction : ∀ p : Person, has_black_card p → ¬ statement_true p

theorem identify_person :
statement_made_by = Tralyalya ∧ ¬ has_black_card statement_made_by :=
by
  sorry

end identify_person_l5_5586


namespace total_tea_cups_l5_5245

def num_cupboards := 8
def num_compartments_per_cupboard := 5
def num_tea_cups_per_compartment := 85

theorem total_tea_cups :
  num_cupboards * num_compartments_per_cupboard * num_tea_cups_per_compartment = 3400 :=
by
  sorry

end total_tea_cups_l5_5245


namespace original_salary_l5_5001

theorem original_salary (S : ℝ) (h1 : S + 0.10 * S = 1.10 * S) (h2: 1.10 * S - 0.05 * (1.10 * S) = 1.10 * S * 0.95) (h3: 1.10 * S * 0.95 = 2090) : S = 2000 :=
sorry

end original_salary_l5_5001


namespace terminating_decimal_count_l5_5415

theorem terminating_decimal_count :
  (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 1600 ∧ (∃ k : ℕ, 2310 * k = d * (231:ℕ) ∧ ∀ p ∈ integer.primes, p | (2310 / d) → p = 2 ∨ p = 5))
  (finset.range 1601)).card = 6 :=
sorry

end terminating_decimal_count_l5_5415


namespace speed_of_sound_correct_l5_5986

-- Define the given conditions
def heard_second_blast_after : ℕ := 30 * 60 + 24 -- 30 minutes and 24 seconds in seconds
def time_sound_travelled : ℕ := 24 -- The sound traveled for 24 seconds
def distance_travelled : ℕ := 7920 -- Distance in meters

-- Define the expected answer for the speed of sound 
def expected_speed_of_sound : ℕ := 330 -- Speed in meters per second

-- The proposition that states the speed of sound given the conditions
theorem speed_of_sound_correct : (distance_travelled / time_sound_travelled) = expected_speed_of_sound := 
by {
  -- use division to compute the speed of sound
  sorry
}

end speed_of_sound_correct_l5_5986


namespace remainder_of_876539_div_7_l5_5158

theorem remainder_of_876539_div_7 : 876539 % 7 = 6 :=
by
  sorry

end remainder_of_876539_div_7_l5_5158


namespace sum_of_octahedron_faces_l5_5479

theorem sum_of_octahedron_faces (n : ℕ) :
  n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) = 8 * n + 28 :=
by
  sorry

end sum_of_octahedron_faces_l5_5479


namespace original_amount_in_cookie_jar_l5_5896

theorem original_amount_in_cookie_jar (doris_spent martha_spent money_left_in_jar original_amount : ℕ)
  (h1 : doris_spent = 6)
  (h2 : martha_spent = doris_spent / 2)
  (h3 : money_left_in_jar = 15)
  (h4 : original_amount = money_left_in_jar + doris_spent + martha_spent) :
  original_amount = 24 := 
sorry

end original_amount_in_cookie_jar_l5_5896


namespace principal_amount_l5_5646

/-
  Given:
  - Simple Interest (SI) = Rs. 4016.25
  - Rate (R) = 0.08 (8% per annum)
  - Time (T) = 5 years
  
  We want to prove:
  Principal = Rs. 10040.625
-/

def SI : ℝ := 4016.25
def R : ℝ := 0.08
def T : ℕ := 5

theorem principal_amount :
  ∃ P : ℝ, SI = (P * R * T) / 100 ∧ P = 10040.625 :=
by
  sorry

end principal_amount_l5_5646


namespace min_voters_for_tall_24_l5_5085

/-
There are 105 voters divided into 5 districts, each district divided into 7 sections, with each section having 3 voters.
A section is won by a majority vote. A district is won by a majority of sections. The contest is won by a majority of districts.
Tall won the contest. Prove that the minimum number of voters who could have voted for Tall is 24.
-/
noncomputable def min_voters_for_tall (total_voters districts sections voters_per_section : ℕ) (sections_needed_to_win_district districts_needed_to_win_contest : ℕ) : ℕ :=
  let voters_needed_per_section := voters_per_section / 2 + 1
  sections_needed_to_win_district * districts_needed_to_win_contest * voters_needed_per_section

theorem min_voters_for_tall_24 :
  min_voters_for_tall 105 5 7 3 4 3 = 24 :=
sorry

end min_voters_for_tall_24_l5_5085


namespace total_worksheets_l5_5647

theorem total_worksheets (x : ℕ) (h1 : 7 * (x - 8) = 63) : x = 17 := 
by {
  sorry
}

end total_worksheets_l5_5647


namespace solve_quadratic_eq_l5_5609

theorem solve_quadratic_eq {x : ℝ} :
  (x = 3 ∨ x = -1) ↔ x^2 - 2 * x - 3 = 0 :=
by
  sorry

end solve_quadratic_eq_l5_5609


namespace shortest_third_stick_length_l5_5189

-- Definitions of the stick lengths
def length1 := 6
def length2 := 9

-- Statement: The shortest length of the third stick that forms a triangle with lengths 6 and 9 should be 4
theorem shortest_third_stick_length : ∃ length3, length3 = 4 ∧
  (length1 + length2 > length3) ∧ (length1 + length3 > length2) ∧ (length2 + length3 > length1) :=
sorry

end shortest_third_stick_length_l5_5189


namespace percentage_distance_l5_5510

theorem percentage_distance (start : ℝ) (end_point : ℝ) (point : ℝ) (total_distance : ℝ)
  (distance_from_start : ℝ) :
  start = -55 → end_point = 55 → point = 5.5 → total_distance = end_point - start →
  distance_from_start = point - start →
  (distance_from_start / total_distance) * 100 = 55 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_distance_l5_5510


namespace slope_of_parallel_line_l5_5930

theorem slope_of_parallel_line (x y : ℝ) :
  ∀ (m : ℝ), (3 * x - 6 * y = 12) → (m = 1 / 2) :=
begin
  sorry
end

end slope_of_parallel_line_l5_5930


namespace train_length_l5_5383

theorem train_length (s : ℝ) (t : ℝ) (h_s : s = 60) (h_t : t = 10) :
  ∃ L : ℝ, L = 166.7 := by
  sorry

end train_length_l5_5383


namespace inequality_solution_l5_5124

theorem inequality_solution (x : ℝ) : (3 < x ∧ x < 5) → (x - 5) / ((x - 3)^2) < 0 := 
by 
  intro h
  sorry

end inequality_solution_l5_5124


namespace circle_passing_through_points_l5_5872

theorem circle_passing_through_points :
  ∃ D E F : ℝ, ∀ (x y : ℝ),
    ((x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) →
    (x^2 + y^2 + D*x + E*y + F = 0) ↔
    (x^2 + y^2 - 4*x - 6*y = 0) :=
begin
  sorry,
end

end circle_passing_through_points_l5_5872


namespace subtraction_addition_example_l5_5806

theorem subtraction_addition_example :
  1500000000000 - 877888888888 + 123456789012 = 745567900124 :=
by
  sorry

end subtraction_addition_example_l5_5806


namespace number_of_basic_events_prob_of_one_black_prob_of_one_blue_l5_5582
-- Import the mathlib library to use combinatorial and probability functions.

open Finset

-- Definitions of labeled pens
def black_pens : Finset ℕ := {1, 2, 3}  -- {A, B, C}
def blue_pens  : Finset ℕ := {4, 5}      -- {d, e}
def red_pen    : Finset ℕ := {6}         -- {x}
def all_pens   : Finset ℕ := black_pens ∪ blue_pens ∪ red_pen

-- Three pens are drawn randomly
def selection_event : Finset (Finset ℕ) := (all_pens.powerset.filter (λ s => s.card = 3))

noncomputable def probability (event : Finset (Finset ℕ)) : ℚ :=
  (event.card : ℚ) / (selection_event.card : ℚ)

-- Prove the number of basic events
theorem number_of_basic_events : selection_event.card = 20 := by
  sorry

-- Prove the probability of selecting exactly one black pen
def exactly_one_black_pen : Finset (Finset ℕ) :=
  selection_event.filter (λ s => (s ∩ black_pens).card = 1)

theorem prob_of_one_black : probability exactly_one_black_pen = 9 / 20 := by
  sorry

-- Prove the probability of selecting at least one blue pen
def at_least_one_blue_pen : Finset (Finset ℕ) :=
  selection_event.filter (λ s => (s ∩ blue_pens).nonempty)

theorem prob_of_one_blue : probability at_least_one_blue_pen = 4 / 5 := by
  sorry

end number_of_basic_events_prob_of_one_black_prob_of_one_blue_l5_5582


namespace factor_expression_l5_5659

variable (a : ℝ)

theorem factor_expression : 37 * a^2 + 111 * a = 37 * a * (a + 3) :=
  sorry

end factor_expression_l5_5659


namespace arithmetic_sequence_ninth_term_l5_5888

theorem arithmetic_sequence_ninth_term
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29) :
  a + 8 * d = 35 :=
sorry

end arithmetic_sequence_ninth_term_l5_5888


namespace sum_of_cubes_equals_square_l5_5857

theorem sum_of_cubes_equals_square :
  1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 := 
by 
  sorry

end sum_of_cubes_equals_square_l5_5857


namespace candy_mixture_problem_l5_5190

theorem candy_mixture_problem:
  ∃ x y : ℝ, x + y = 5 ∧ 3.20 * x + 1.70 * y = 10 ∧ x = 1 :=
by
  sorry

end candy_mixture_problem_l5_5190


namespace stop_shooting_after_2nd_scoring_5_points_eq_l5_5410

/-
Define the conditions and problem statement in Lean:
- Each person can shoot up to 10 times.
- Student A's shooting probability for each shot is 2/3.
- If student A stops shooting at the nth consecutive shot, they score 12-n points.
- We need to prove the probability that student A stops shooting right after the 2nd shot and scores 5 points is 8/729.
-/
def student_shoot_probability (shots : List Bool) (p : ℚ) : ℚ :=
  shots.foldr (λ s acc => if s then p * acc else (1 - p) * acc) 1

def stop_shooting_probability : ℚ :=
  let shots : List Bool := [false, true, false, false, false, true, true] -- represents misses and hits
  student_shoot_probability shots (2/3)

theorem stop_shooting_after_2nd_scoring_5_points_eq :
  stop_shooting_probability = (8 / 729) :=
sorry

end stop_shooting_after_2nd_scoring_5_points_eq_l5_5410


namespace projected_increase_l5_5457

theorem projected_increase (R : ℝ) (P : ℝ) 
  (h1 : ∃ P, ∀ (R : ℝ), 0.9 * R = 0.75 * (R + (P / 100) * R)) 
  (h2 : ∀ (R : ℝ), R > 0) :
  P = 20 :=
by
  sorry

end projected_increase_l5_5457


namespace calibration_measurements_l5_5767

theorem calibration_measurements (holes : Fin 15 → ℝ) (diameter : ℝ)
  (h1 : ∀ i : Fin 15, holes i = 10 + i.val * 0.04)
  (h2 : 10 ≤ diameter ∧ diameter ≤ 10 + 14 * 0.04) :
  ∃ tries : ℕ, (tries ≤ 4) ∧ (∀ (i : Fin 15), if diameter ≤ holes i then True else False) :=
sorry

end calibration_measurements_l5_5767


namespace slope_of_parallel_line_l5_5932

theorem slope_of_parallel_line (x y : ℝ) :
  ∀ (m : ℝ), (3 * x - 6 * y = 12) → (m = 1 / 2) :=
begin
  sorry
end

end slope_of_parallel_line_l5_5932


namespace solve_system_l5_5126

theorem solve_system :
  (∀ x y : ℝ, 
    (x^2 * y - x * y^2 - 3 * x + 3 * y + 1 = 0 ∧
     x^3 * y - x * y^3 - 3 * x^2 + 3 * y^2 + 3 = 0) → (x, y) = (2, 1)) :=
by simp [← solve_system]; sorry

end solve_system_l5_5126


namespace right_triangle_third_side_product_l5_5330

theorem right_triangle_third_side_product :
  ∃ (c hypotenuse : ℝ), 
    (c = real.sqrt (6^2 + 8^2) ∨ hypotenuse = real.sqrt (8^2 - 6^2)) ∧ 
    real.sqrt (6^2 + 8^2) * real.sqrt (8^2 - 6^2) = 52.7 :=
by
  sorry

end right_triangle_third_side_product_l5_5330


namespace coefficient_x99_is_zero_l5_5481

open Polynomial

noncomputable def P (x : ℤ) : Polynomial ℤ := sorry
noncomputable def Q (x : ℤ) : Polynomial ℤ := sorry

theorem coefficient_x99_is_zero : 
    (P 0 = 1) → 
    ((P x)^2 = 1 + x + x^100 * Q x) → 
    (Polynomial.coeff ((P x + 1)^100) 99 = 0) :=
by
    -- Proof omitted
    sorry

end coefficient_x99_is_zero_l5_5481


namespace game_winning_starting_numbers_count_l5_5605

theorem game_winning_starting_numbers_count : 
  ∃ win_count : ℕ, (win_count = 6) ∧ 
                  ∀ n : ℕ, (1 ≤ n ∧ n < 10) → 
                  (n = 1 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9) ↔ 
                  ((∃ m, (2 * n ≤ m ∧ m ≤ 3 * n) ∧ m < 2007)  → 
                   (∃ k, (2 * m ≤ k ∧ k ≤ 3 * m) ∧ k ≥ 2007) = false) := 
sorry

end game_winning_starting_numbers_count_l5_5605


namespace slope_of_parallel_line_l5_5910

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l5_5910


namespace probability_three_specific_cards_l5_5824

noncomputable def deck_size : ℕ := 52
noncomputable def num_suits : ℕ := 4
noncomputable def cards_per_suit : ℕ := 13
noncomputable def p_king_spades : ℚ := 1 / deck_size
noncomputable def p_10_hearts : ℚ := 1 / (deck_size - 1)
noncomputable def p_queen : ℚ := 4 / (deck_size - 2)

theorem probability_three_specific_cards :
  (p_king_spades * p_10_hearts * p_queen) = 1 / 33150 := 
sorry

end probability_three_specific_cards_l5_5824


namespace remainder_sum_first_seven_primes_div_eighth_prime_l5_5169

theorem remainder_sum_first_seven_primes_div_eighth_prime :
  let primes := [2, 3, 5, 7, 11, 13, 17] in
  let sum_first_seven := (List.sum primes) in
  let eighth_prime := 19 in
  sum_first_seven % eighth_prime = 1 :=
by
  let primes := [2, 3, 5, 7, 11, 13, 17]
  let sum_first_seven := (List.sum primes)
  let eighth_prime := 19
  show sum_first_seven % eighth_prime = 1
  sorry

end remainder_sum_first_seven_primes_div_eighth_prime_l5_5169


namespace second_player_wins_l5_5462

-- Define the initial condition of the game
def initial_coins : Nat := 2016

-- Define the set of moves a player can make
def valid_moves : Finset Nat := {1, 2, 3}

-- Define the winning condition
def winning_player (coins : Nat) : String :=
  if coins % 4 = 0 then "second player"
  else "first player"

-- The theorem stating that second player has a winning strategy given the initial condition
theorem second_player_wins : winning_player initial_coins = "second player" :=
by
  sorry

end second_player_wins_l5_5462


namespace find_x_l5_5504

theorem find_x (x : ℝ) : 0.6 * x = (x / 3) + 110 → x = 412.5 := 
by
  intro h
  sorry

end find_x_l5_5504


namespace sum_of_0_75_of_8_and_2_l5_5745

theorem sum_of_0_75_of_8_and_2 : 0.75 * 8 + 2 = 8 := by
  sorry

end sum_of_0_75_of_8_and_2_l5_5745


namespace min_voters_to_win_l5_5083

def num_voters : ℕ := 105
def num_districts : ℕ := 5
def num_sections_per_district : ℕ := 7
def voters_per_section : ℕ := 3
def majority n : ℕ := n / 2 + 1

theorem min_voters_to_win (Tall_won : ∃ sections : fin num_voters → bool, 
  (∃ districts : fin num_districts → bool, 
    (countp (λ i, districts i = tt) (finset.univ : finset (fin num_districts)) ≥ majority num_districts) ∧ 
    ∀ i : fin num_districts, districts i = tt →
      (countp (λ j, sections (i * num_sections_per_district + j) = tt) (finset.range num_sections_per_district) ≥ majority num_sections_per_district)
  ) ∧
  (∀ i, i < num_voters →¬ (sections i = tt → sections ((i / num_sections_per_district) * num_sections_per_district + (i % num_sections_per_district)) = tt))
  ) : 3 * (12 * 2) ≥ 24 :=
by sorry

end min_voters_to_win_l5_5083


namespace repeating_decimal_fraction_l5_5658

def repeating_decimal_to_fraction (d: ℚ) (r: ℚ) (p: ℚ): ℚ :=
  d + r

theorem repeating_decimal_fraction :
  repeating_decimal_to_fraction (6 / 10) (1 / 33) (0.6 + (0.03 : ℚ)) = 104 / 165 := 
by
  sorry

end repeating_decimal_fraction_l5_5658


namespace sufficient_condition_for_ellipse_l5_5795

theorem sufficient_condition_for_ellipse (m : ℝ) (h : m^2 > 5) : m^2 > 4 := by
  sorry

end sufficient_condition_for_ellipse_l5_5795


namespace correct_transformation_l5_5997

theorem correct_transformation :
  (∀ a b c : ℝ, c ≠ 0 → (a / c = b / c ↔ a = b)) ∧
  (∀ x : ℝ, ¬ (x / 4 + x / 3 = 1 ∧ 3 * x + 4 * x = 1)) ∧
  (∀ a b c : ℝ, ¬ (a * b = b * c ∧ a ≠ c)) ∧
  (∀ x a : ℝ, ¬ (4 * x = a ∧ x = 4 * a)) := sorry

end correct_transformation_l5_5997


namespace remainder_when_divided_by_2000_l5_5590

def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

noncomputable def count_disjoint_subsets (S : Set ℕ) : ℕ :=
  let totalWays := 3^12
  let emptyACases := 2*2^12
  let bothEmptyCase := 1
  (totalWays - emptyACases + bothEmptyCase) / 2

theorem remainder_when_divided_by_2000 : count_disjoint_subsets S % 2000 = 1625 := by
  sorry

end remainder_when_divided_by_2000_l5_5590


namespace arithmetic_sequence_30th_term_l5_5343

-- Defining the initial term and the common difference of the arithmetic sequence
def a : ℕ := 3
def d : ℕ := 4

-- Defining the general formula for the n-th term of the arithmetic sequence
def a_n (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem stating that the 30th term of the given sequence is 119
theorem arithmetic_sequence_30th_term : a_n 30 = 119 := 
sorry

end arithmetic_sequence_30th_term_l5_5343


namespace best_solved_completing_square_l5_5181

theorem best_solved_completing_square :
  ∀ (x : ℝ), x^2 - 2*x - 3 = 0 → (x - 1)^2 - 4 = 0 :=
sorry

end best_solved_completing_square_l5_5181


namespace missing_angle_measure_l5_5697

theorem missing_angle_measure (n : ℕ) (h : 180 * (n - 2) = 3240 + 2 * (180 * (n - 2)) / n) : 
  (180 * (n - 2)) / n = 166 := 
by 
  sorry

end missing_angle_measure_l5_5697


namespace circumference_of_circle_l5_5638

/-- Given a circle with area 4 * π square units, prove that its circumference is 4 * π units. -/
theorem circumference_of_circle (r : ℝ) (h : π * r^2 = 4 * π) : 2 * π * r = 4 * π :=
sorry

end circumference_of_circle_l5_5638


namespace number_total_11_l5_5623

theorem number_total_11 (N : ℕ) (S : ℝ)
  (h1 : S = 10.7 * N)
  (h2 : (6 : ℝ) * 10.5 = 63)
  (h3 : (6 : ℝ) * 11.4 = 68.4)
  (h4 : 13.7 = 13.700000000000017)
  (h5 : S = 63 + 68.4 - 13.7) : 
  N = 11 := 
sorry

end number_total_11_l5_5623


namespace mistaken_quotient_is_35_l5_5257

theorem mistaken_quotient_is_35 (D : ℕ) (correct_divisor mistaken_divisor correct_quotient : ℕ) 
    (h1 : D = correct_divisor * correct_quotient)
    (h2 : correct_divisor = 21)
    (h3 : mistaken_divisor = 12)
    (h4 : correct_quotient = 20)
    : D / mistaken_divisor = 35 := by
  sorry

end mistaken_quotient_is_35_l5_5257


namespace volume_ratio_proof_l5_5581

-- Definitions:
def height_ratio := 2 / 3
def volume_ratio (r : ℚ) := r^3
def small_pyramid_volume_ratio := volume_ratio height_ratio
def frustum_volume_ratio := 1 - small_pyramid_volume_ratio
def volume_ratio_small_to_frustum (v_small v_frustum : ℚ) := v_small / v_frustum

-- Lean 4 Statement:
theorem volume_ratio_proof
  (height_ratio : ℚ := 2 / 3)
  (small_pyramid_volume_ratio : ℚ := volume_ratio height_ratio)
  (frustum_volume_ratio : ℚ := 1 - small_pyramid_volume_ratio)
  (v_orig : ℚ) :
  volume_ratio_small_to_frustum (small_pyramid_volume_ratio * v_orig) (frustum_volume_ratio * v_orig) = 8 / 19 :=
by
  sorry

end volume_ratio_proof_l5_5581


namespace largest_n_unique_k_l5_5907

theorem largest_n_unique_k : ∃ n : ℕ, n = 24 ∧ (∃! k : ℕ, 
  3 / 7 < n / (n + k: ℤ) ∧ n / (n + k: ℤ) < 8 / 19) :=
by
  sorry

end largest_n_unique_k_l5_5907


namespace spencer_walked_distance_l5_5588

/-- Define the distances involved -/
def total_distance := 0.8
def library_to_post_office := 0.1
def post_office_to_home := 0.4

/-- Define the distance from house to library as a variable to calculate -/
def house_to_library := total_distance - library_to_post_office - post_office_to_home

/-- The theorem states that Spencer walked 0.3 miles from his house to the library -/
theorem spencer_walked_distance : 
  house_to_library = 0.3 :=
by
  -- Proof omitted
  sorry

end spencer_walked_distance_l5_5588


namespace right_triangle_third_side_product_l5_5329

noncomputable def hypot : ℝ → ℝ → ℝ := λ a b, real.sqrt (a * a + b * b)

noncomputable def other_leg : ℝ → ℝ → ℝ := λ h a, real.sqrt (h * h - a * a)

theorem right_triangle_third_side_product (a b : ℝ) (h : ℝ) (product : ℝ) 
  (h₁ : a = 6) (h₂ : b = 8) (h₃ : h = hypot 6 8) 
  (h₄ : b = h) (leg : ℝ := other_leg 8 6) :
  product = 52.9 :=
by
  have h5 : hypot 6 8 = 10 := sorry 
  have h6 : other_leg 8 6 = 2 * real.sqrt 7 := sorry
  have h7 : 10 * (2 * real.sqrt 7) = 20 * real.sqrt 7 := sorry
  have h8 : real.sqrt 7 ≈ 2.6458 := sorry
  have h9 : 20 * 2.6458 ≈ 52.916 := sorry
  have h10 : (52.916 : ℝ).round_to 1 = 52.9 := sorry
  exact h10


end right_triangle_third_side_product_l5_5329


namespace parallel_or_identical_lines_l5_5053

theorem parallel_or_identical_lines (a b c d e f : ℝ) :
  2 * b - 3 * a = 15 → 4 * d - 6 * c = 18 → (b ≠ d → a = c) :=
by
  intros h1 h2 hneq
  sorry

end parallel_or_identical_lines_l5_5053


namespace average_goals_per_game_l5_5601

theorem average_goals_per_game
  (slices_per_pizza : ℕ := 12)
  (total_pizzas : ℕ := 6)
  (total_games : ℕ := 8)
  (total_slices : ℕ := total_pizzas * slices_per_pizza)
  (total_goals : ℕ := total_slices)
  (average_goals : ℕ := total_goals / total_games) :
  average_goals = 9 :=
by
  sorry

end average_goals_per_game_l5_5601


namespace find_divisor_l5_5000

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 161) 
  (h2 : quotient = 10)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) : 
  divisor = 16 :=
by
  sorry

end find_divisor_l5_5000


namespace circle_line_intersection_condition_slope_condition_l5_5700

open Real

theorem circle_line_intersection_condition (k : ℝ) :
  (-3 < k ∧ k < 1/3) ↔
  let M := (4, 0) in
  let r := sqrt 10 in
  let d := abs (4 * k + 2) / sqrt (k^2 + 1) in
  d < r :=
by sorry

theorem slope_condition (k : ℝ) :
  (ON_parallel_MP : Bool) → 
  (ON_parallel_MP = true) ↔ k = -4/3 :=
by sorry

end circle_line_intersection_condition_slope_condition_l5_5700


namespace evaluate_expression_l5_5492

theorem evaluate_expression : 5^2 + 15 / 3 - (3 * 2)^2 = -6 := 
by
  sorry

end evaluate_expression_l5_5492


namespace combined_age_of_siblings_l5_5202

-- We are given Aaron's age
def aaronAge : ℕ := 15

-- Henry's sister's age is three times Aaron's age
def henrysSisterAge : ℕ := 3 * aaronAge

-- Henry's age is four times his sister's age
def henryAge : ℕ := 4 * henrysSisterAge

-- The combined age of the siblings
def combinedAge : ℕ := aaronAge + henrysSisterAge + henryAge

theorem combined_age_of_siblings : combinedAge = 240 := by
  sorry

end combined_age_of_siblings_l5_5202


namespace sum_of_first_seven_primes_mod_eighth_prime_l5_5174

theorem sum_of_first_seven_primes_mod_eighth_prime :
  (2 + 3 + 5 + 7 + 11 + 13 + 17) % 19 = 1 :=
by
  sorry

end sum_of_first_seven_primes_mod_eighth_prime_l5_5174


namespace area_of_rectangle_l5_5730

theorem area_of_rectangle (x : ℝ) (hx : 0 < x) :
  let length := 3 * x - 1
  let width := 2 * x + 1 / 2
  let area := length * width
  area = 6 * x^2 - 1 / 2 * x - 1 / 2 :=
by
  sorry

end area_of_rectangle_l5_5730


namespace commencement_addresses_l5_5260

theorem commencement_addresses (sandoval_addresses : ℕ) 
                             (hawkins_addresses : ℕ) 
                             (sloan_addresses : ℕ) :
  sandoval_addresses = 12 →
  hawkins_addresses = sandoval_addresses / 2 →
  sloan_addresses = sandoval_addresses + 10 →
  sandoval_addresses + hawkins_addresses + sloan_addresses = 40 :=
begin
  sorry
end

end commencement_addresses_l5_5260


namespace right_triangle_third_side_product_l5_5332

def hypotenuse_length (a b : ℕ) : Real := Real.sqrt (a*a + b*b)
def other_leg_length (h b : ℕ) : Real := Real.sqrt (h*h - b*b)

theorem right_triangle_third_side_product (a b : ℕ) (ha : a = 6) (hb : b = 8) : 
  Real.round (hypotenuse_length a b * other_leg_length b a) = 53 :=
by
  have h1 : hypotenuse_length 6 8 = 10 := by sorry
  have h2 : other_leg_length 8 6 = 2 * Real.sqrt 7 := by sorry
  calc
    Real.round (10 * (2 * Real.sqrt 7)) = Real.round (20 * Real.sqrt 7) := by sorry
                                   ...  = 53 := by sorry

end right_triangle_third_side_product_l5_5332


namespace goods_train_speed_l5_5640

theorem goods_train_speed (length_train length_platform distance time : ℕ) (conversion_factor : ℚ) : 
  length_train = 250 → 
  length_platform = 270 → 
  distance = length_train + length_platform → 
  time = 26 → 
  conversion_factor = 3.6 →
  (distance / time : ℚ) * conversion_factor = 72 :=
by
  intros h_lt h_lp h_d h_t h_cf
  rw [h_lt, h_lp] at h_d
  rw [h_t, h_cf]
  sorry

end goods_train_speed_l5_5640


namespace parallel_line_slope_l5_5940

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℚ, m = 1 / 2 :=
by {
  use 1 / 2,
  sorry
}

end parallel_line_slope_l5_5940


namespace two_solutions_for_positive_integer_m_l5_5416

theorem two_solutions_for_positive_integer_m :
  ∃ k : ℕ, k = 2 ∧ (∀ m : ℕ, 0 < m → 990 % (m^2 - 2) = 0 → m = 2 ∨ m = 3) := 
sorry

end two_solutions_for_positive_integer_m_l5_5416


namespace hawks_points_l5_5436

theorem hawks_points (x y z : ℤ) 
  (h_total_points: x + y = 82)
  (h_margin: x - y = 18)
  (h_eagles_points: x = 12 + z) : 
  y = 32 := 
sorry

end hawks_points_l5_5436


namespace cost_of_pen_is_five_l5_5019

-- Define the given conditions
def pencils_per_box := 80
def num_boxes := 15
def total_pencils := num_boxes * pencils_per_box
def cost_per_pencil := 4
def total_cost_of_stationery := 18300
def additional_pens := 300
def num_pens := 2 * total_pencils + additional_pens

-- Calculate total cost of pencils
def total_cost_of_pencils := total_pencils * cost_per_pencil

-- Calculate total cost of pens
def total_cost_of_pens := total_cost_of_stationery - total_cost_of_pencils

-- The conjecture to prove
theorem cost_of_pen_is_five :
  (total_cost_of_pens / num_pens) = 5 :=
sorry

end cost_of_pen_is_five_l5_5019


namespace average_yield_per_tree_l5_5830

theorem average_yield_per_tree :
  let t1 := 3
  let t2 := 2
  let t3 := 1
  let nuts1 := 60
  let nuts2 := 120
  let nuts3 := 180
  let total_nuts := t1 * nuts1 + t2 * nuts2 + t3 * nuts3
  let total_trees := t1 + t2 + t3
  let average_yield := total_nuts / total_trees
  average_yield = 100 := 
by
  sorry

end average_yield_per_tree_l5_5830


namespace smallest_C_inequality_l5_5546

theorem smallest_C_inequality (x y z : ℝ) (h : x + y + z = -1) : 
  |x^3 + y^3 + z^3 + 1| ≤ (9/10) * |x^5 + y^5 + z^5 + 1| :=
  sorry

end smallest_C_inequality_l5_5546


namespace normal_price_of_article_l5_5023

theorem normal_price_of_article (P : ℝ) (h : 0.9 * 0.8 * P = 144) : P = 200 :=
sorry

end normal_price_of_article_l5_5023


namespace perimeter_of_face_given_volume_l5_5292

-- Definitions based on conditions
def volume_of_cube (v : ℝ) := v = 512

def side_of_cube (s : ℝ) := s^3 = 512

def perimeter_of_face (p s : ℝ) := p = 4 * s

-- Lean 4 statement: prove that the perimeter of one face of the cube is 32 cm given the volume is 512 cm³.
theorem perimeter_of_face_given_volume :
  ∃ s : ℝ, volume_of_cube (s^3) ∧ perimeter_of_face 32 s :=
by sorry

end perimeter_of_face_given_volume_l5_5292


namespace locus_equation_of_points_at_distance_2_from_line_l5_5545

theorem locus_equation_of_points_at_distance_2_from_line :
  {P : ℝ × ℝ | abs ((3 / 5) * P.1 - (4 / 5) * P.2 - (1 / 5)) = 2} =
    {P : ℝ × ℝ | 3 * P.1 - 4 * P.2 - 11 = 0} ∪ {P : ℝ × ℝ | 3 * P.1 - 4 * P.2 + 9 = 0} :=
by
  -- Proof goes here
  sorry

end locus_equation_of_points_at_distance_2_from_line_l5_5545


namespace arithmetic_sequence_30th_term_l5_5345

-- Defining the initial term and the common difference of the arithmetic sequence
def a : ℕ := 3
def d : ℕ := 4

-- Defining the general formula for the n-th term of the arithmetic sequence
def a_n (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem stating that the 30th term of the given sequence is 119
theorem arithmetic_sequence_30th_term : a_n 30 = 119 := 
sorry

end arithmetic_sequence_30th_term_l5_5345


namespace number_of_solutions_l5_5066

theorem number_of_solutions :
  ∃ (s : Finset (ℤ × ℤ)), (∀ (a : ℤ × ℤ), a ∈ s ↔ (a.1^4 + a.2^4 = 4 * a.2)) ∧ s.card = 3 :=
by
  sorry

end number_of_solutions_l5_5066


namespace taxi_ride_cost_l5_5198

-- Definitions based on conditions
def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def minimum_charge : ℝ := 5.00
def fare (miles : ℝ) : ℝ := base_fare + miles * cost_per_mile

-- Theorem statement reflecting the problem
theorem taxi_ride_cost (miles : ℝ) (h : miles < 4) : fare miles < minimum_charge → fare miles = minimum_charge :=
by
  sorry

end taxi_ride_cost_l5_5198


namespace g_3_2_eq_neg3_l5_5114

noncomputable def f (x y : ℝ) : ℝ := x^3 * y^2 + 4 * x^2 * y - 15 * x

axiom f_symmetric : ∀ x y : ℝ, f x y = f y x
axiom f_2_4_eq_neg2 : f 2 4 = -2

noncomputable def g (x y : ℝ) : ℝ := (x^3 - 3 * x^2 * y + x * y^2) / (x^2 - y^2)

theorem g_3_2_eq_neg3 : g 3 2 = -3 := by
  sorry

end g_3_2_eq_neg3_l5_5114


namespace decagon_adjacent_vertex_probability_l5_5323

theorem decagon_adjacent_vertex_probability :
  let vertices := 10 in
  let total_combinations := Nat.choose vertices 2 in
  let adjacent_pairs := vertices * 2 in
  (adjacent_pairs : ℚ) / total_combinations = 4 / 9 :=
by
  let vertices := 10
  let total_combinations := Nat.choose vertices 2
  let adjacent_pairs := vertices * 2
  have : (adjacent_pairs : ℚ) / total_combinations = 4 / 9 := sorry
  exact this

end decagon_adjacent_vertex_probability_l5_5323


namespace five_point_eight_divide_by_point_zero_zero_one_eq_five_point_eight_multiply_by_thousand_l5_5208

theorem five_point_eight_divide_by_point_zero_zero_one_eq_five_point_eight_multiply_by_thousand :
  5.8 / 0.001 = 5.8 * 1000 :=
by
  -- This is where the proof would go
  sorry

end five_point_eight_divide_by_point_zero_zero_one_eq_five_point_eight_multiply_by_thousand_l5_5208


namespace marble_problem_l5_5624

theorem marble_problem 
  (G R B W : ℕ) 
  (h_total : G + R + B + W = 84) 
  (h_green : (G : ℚ) / 84 = 1 / 7) 
  (h_red_blue : (R + B : ℚ) / 84 = 0.6071428571428572) : 
  (W : ℚ) / 84 = 1 / 4 := 
by 
  sorry

end marble_problem_l5_5624


namespace greater_solution_of_quadratic_eq_l5_5754

theorem greater_solution_of_quadratic_eq (x : ℝ) : 
  (∀ y : ℝ, y^2 + 20 * y - 96 = 0 → (y = 4)) :=
sorry

end greater_solution_of_quadratic_eq_l5_5754


namespace remainder_sum_first_seven_primes_div_eighth_prime_l5_5168

theorem remainder_sum_first_seven_primes_div_eighth_prime :
  let primes := [2, 3, 5, 7, 11, 13, 17] in
  let sum_first_seven := (List.sum primes) in
  let eighth_prime := 19 in
  sum_first_seven % eighth_prime = 1 :=
by
  let primes := [2, 3, 5, 7, 11, 13, 17]
  let sum_first_seven := (List.sum primes)
  let eighth_prime := 19
  show sum_first_seven % eighth_prime = 1
  sorry

end remainder_sum_first_seven_primes_div_eighth_prime_l5_5168


namespace fido_area_reach_l5_5540

theorem fido_area_reach (r : ℝ) (s : ℝ) (a b : ℕ) (h1 : r = s * (1 / (Real.tan (Real.pi / 8))))
  (h2 : a = 2) (h3 : b = 8)
  (h_fraction : (Real.pi * r ^ 2) / (2 * (1 + Real.sqrt 2) * (r ^ 2 * (Real.tan (Real.pi / 8)) ^ 2)) = (Real.pi * Real.sqrt a) / b) :
  a * b = 16 := by
  sorry

end fido_area_reach_l5_5540


namespace travel_days_l5_5989

variable (a b d : ℕ)

theorem travel_days (h1 : a + d = 11) (h2 : b + d = 21) (h3 : a + b = 12) : a + b + d = 22 :=
by sorry

end travel_days_l5_5989


namespace remainder_of_primes_sum_l5_5166

theorem remainder_of_primes_sum :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let p8 := 19 
  (p1 + p2 + p3 + p4 + p5 + p6 + p7) % p8 = 1 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let p8 := 19
  let sum := p1 + p2 + p3 + p4 + p5 + p6 + p7
  have h : sum = 58 := by norm_num
  show sum % p8 = 1
  rw [h]
  norm_num
  sorry

end remainder_of_primes_sum_l5_5166


namespace maximum_area_of_triangle_OAB_l5_5589

noncomputable def maximum_area_triangle (a b : ℝ) : ℝ :=
  if 2 * a + b = 5 ∧ a > 0 ∧ b > 0 then (1 / 2) * a * b else 0

theorem maximum_area_of_triangle_OAB : 
  (∀ (a b : ℝ), 2 * a + b = 5 ∧ a > 0 ∧ b > 0 → (1 / 2) * a * b ≤ 25 / 16) :=
by
  sorry

end maximum_area_of_triangle_OAB_l5_5589


namespace P_on_x_axis_Q_max_y_PQR_90_deg_PQS_PQT_45_deg_l5_5703

-- Conditions
def center_C : (ℝ × ℝ) := (6, 8)
def radius : ℝ := 10
def circle_eq (x y : ℝ) : Prop := (x - 6)^2 + (y - 8)^2 = 100
def origin_O : (ℝ × ℝ) := (0, 0)

-- (a) Point of intersection of the circle with the x-axis
def point_P : (ℝ × ℝ) := (12, 0)
theorem P_on_x_axis : circle_eq (point_P.1) (point_P.2) ∧ point_P.2 = 0 := sorry

-- (b) Point on the circle with maximum y-coordinate
def point_Q : (ℝ × ℝ) := (6, 18)
theorem Q_max_y : circle_eq (point_Q.1) (point_Q.2) ∧ ∀ y : ℝ, (circle_eq 6 y → y ≤ 18) := sorry

-- (c) Point on the circle such that ∠PQR = 90°
def point_R : (ℝ × ℝ) := (0, 16)
theorem PQR_90_deg : circle_eq (point_R.1) (point_R.2) ∧
  ∃ Q : (ℝ × ℝ), circle_eq (Q.1) (Q.2) ∧ (Q = point_Q) ∧ (point_P.1 - point_R.1) * (Q.1 - point_Q.1) + (point_P.2 - point_R.2) * (Q.2 - point_Q.2) = 0 := sorry

-- (d) Two points on the circle such that ∠PQS = ∠PQT = 45°
def point_S : (ℝ × ℝ) := (14, 14)
def point_T : (ℝ × ℝ) := (-2, 2)
theorem PQS_PQT_45_deg : circle_eq (point_S.1) (point_S.2) ∧ circle_eq (point_T.1) (point_T.2) ∧
  ∃ Q : (ℝ × ℝ), circle_eq (Q.1) (Q.2) ∧ (Q = point_Q) ∧
  ((point_P.1 - Q.1) * (point_S.1 - Q.1) + (point_P.2 - Q.2) * (point_S.2 - Q.2) =
  (point_P.1 - Q.1) * (point_T.1 - Q.1) + (point_P.2 - Q.2) * (point_T.2 - Q.2)) := sorry

end P_on_x_axis_Q_max_y_PQR_90_deg_PQS_PQT_45_deg_l5_5703


namespace positive_number_and_cube_l5_5879

theorem positive_number_and_cube (n : ℕ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 ∧ n^3 = 2744 :=
by sorry

end positive_number_and_cube_l5_5879


namespace carol_additional_cupcakes_l5_5548

-- Define the initial number of cupcakes Carol made
def initial_cupcakes : ℕ := 30

-- Define the number of cupcakes Carol sold
def sold_cupcakes : ℕ := 9

-- Define the total number of cupcakes Carol wanted to have
def total_cupcakes : ℕ := 49

-- Calculate the number of cupcakes Carol had left after selling
def remaining_cupcakes : ℕ := initial_cupcakes - sold_cupcakes

-- The number of additional cupcakes Carol made can be defined and proved as follows:
theorem carol_additional_cupcakes : initial_cupcakes - sold_cupcakes + 28 = total_cupcakes :=
by
  -- left side: initial_cupcakes (30) - sold_cupcakes (9) + additional_cupcakes (28) = total_cupcakes (49)
  sorry

end carol_additional_cupcakes_l5_5548


namespace probability_of_odd_sum_is_27_over_64_l5_5299

noncomputable def probability_odd_sum : ℚ :=
  let coin_prob : ℚ := 1 / 2 in
  let die_prob_odd : ℚ := 1 / 2 in
  let case_0_heads := coin_prob ^ 3 in
  let case_1_head := 3 * (coin_prob ^ 3) * die_prob_odd in
  let case_2_heads := 3 * (coin_prob ^ 3) * (2 * (die_prob_odd * (1 - die_prob_odd))) in
  let case_3_heads := (coin_prob ^ 3) * (((3 * (die_prob_odd ^ 2 * (1 - die_prob_odd))) + (3 * ((1 - die_prob_odd) ^ 2 * die_prob_odd)))) in
  case_0_heads * 0 + case_1_head + case_2_heads + case_3_heads

theorem probability_of_odd_sum_is_27_over_64 : probability_odd_sum = 27 / 64 :=
sorry

end probability_of_odd_sum_is_27_over_64_l5_5299


namespace remainder_of_primes_sum_l5_5167

theorem remainder_of_primes_sum :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let p8 := 19 
  (p1 + p2 + p3 + p4 + p5 + p6 + p7) % p8 = 1 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let p8 := 19
  let sum := p1 + p2 + p3 + p4 + p5 + p6 + p7
  have h : sum = 58 := by norm_num
  show sum % p8 = 1
  rw [h]
  norm_num
  sorry

end remainder_of_primes_sum_l5_5167


namespace ratio_of_percent_increase_to_decrease_l5_5630

variable (P U V : ℝ)
variable (h1 : P * U = 0.25 * P * V)
variable (h2 : P ≠ 0)

theorem ratio_of_percent_increase_to_decrease (h : U = 0.25 * V) :
  ((V - U) / U) * 100 / 75 = 4 :=
by
  sorry

end ratio_of_percent_increase_to_decrease_l5_5630


namespace union_A_B_l5_5557

-- Definitions based on the conditions
def A := { x : ℝ | x < -1 ∨ (2 ≤ x ∧ x < 3) }
def B := { x : ℝ | -2 ≤ x ∧ x < 4 }

-- The proof goal
theorem union_A_B : A ∪ B = { x : ℝ | x < 4 } :=
by
  sorry -- Proof placeholder

end union_A_B_l5_5557


namespace shirt_ratio_l5_5280

theorem shirt_ratio
  (A B S : ℕ)
  (h1 : A = 6 * B)
  (h2 : B = 3)
  (h3 : S = 72) :
  S / A = 4 :=
by
  sorry

end shirt_ratio_l5_5280


namespace total_chairs_agreed_proof_l5_5793

/-
Conditions:
- Carey moved 28 chairs
- Pat moved 29 chairs
- They have 17 chairs left to move
Question:
- How many chairs did they agree to move in total?
Proof Problem:
- Prove that the total number of chairs they agreed to move is equal to 74.
-/

def carey_chairs : ℕ := 28
def pat_chairs : ℕ := 29
def chairs_left : ℕ := 17
def total_chairs_agreed : ℕ := carey_chairs + pat_chairs + chairs_left

theorem total_chairs_agreed_proof : total_chairs_agreed = 74 := 
by
  sorry

end total_chairs_agreed_proof_l5_5793


namespace fido_yard_area_reach_l5_5538

theorem fido_yard_area_reach (s r : ℝ) (h1 : r = s / (2 * Real.sqrt 2)) (h2 : ∃ (a b : ℕ), (Real.pi * Real.sqrt a) / b = Real.pi * (r ^ 2) / (2 * s^2 * Real.sqrt 2) ) :
  ∃ (a b : ℕ), a * b = 64 :=
by
  sorry

end fido_yard_area_reach_l5_5538


namespace correct_transformation_l5_5998

theorem correct_transformation :
  (∀ a b c : ℝ, c ≠ 0 → (a / c = b / c ↔ a = b)) ∧
  (∀ x : ℝ, ¬ (x / 4 + x / 3 = 1 ∧ 3 * x + 4 * x = 1)) ∧
  (∀ a b c : ℝ, ¬ (a * b = b * c ∧ a ≠ c)) ∧
  (∀ x a : ℝ, ¬ (4 * x = a ∧ x = 4 * a)) := sorry

end correct_transformation_l5_5998


namespace longer_train_length_l5_5489

def length_of_longer_train
  (speed_train1 : ℝ) (speed_train2 : ℝ)
  (length_shorter_train : ℝ) (time_to_clear : ℝ)
  (relative_speed : ℝ := (speed_train1 + speed_train2) * 1000 / 3600)
  (total_distance : ℝ := relative_speed * time_to_clear) : ℝ :=
  total_distance - length_shorter_train

theorem longer_train_length :
  length_of_longer_train 80 55 121 7.626056582140095 = 164.9771230827526 :=
by
  unfold length_of_longer_train
  norm_num
  sorry  -- This placeholder is used to avoid writing out the full proof.

end longer_train_length_l5_5489


namespace probability_of_no_adjacent_stands_is_correct_l5_5537

noncomputable def number_of_arrangements : ℕ → ℕ
| 2 := 3
| 3 := 4
| (n+1) := number_of_arrangements n + number_of_arrangements (n-1)

def total_outcomes : ℕ := 2^8

def favorable_outcomes : ℕ := number_of_arrangements 8

def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem probability_of_no_adjacent_stands_is_correct :
  probability = 47 / 256 :=
by sorry

end probability_of_no_adjacent_stands_is_correct_l5_5537


namespace sum_of_products_l5_5792

theorem sum_of_products : 1 * 15 + 2 * 14 + 3 * 13 + 4 * 12 + 5 * 11 + 6 * 10 + 7 * 9 + 8 * 8 = 372 := by
  sorry

end sum_of_products_l5_5792


namespace correct_calculation_l5_5493

-- Definition of the conditions
def condition1 (a : ℕ) : Prop := a^2 * a^3 = a^6
def condition2 (a : ℕ) : Prop := (a^2)^10 = a^20
def condition3 (a : ℕ) : Prop := (2 * a) * (3 * a) = 6 * a
def condition4 (a : ℕ) : Prop := a^12 / a^2 = a^6

-- The main theorem to state that condition2 is the correct calculation
theorem correct_calculation (a : ℕ) : condition2 a :=
sorry

end correct_calculation_l5_5493


namespace arithmetic_sequence_30th_term_l5_5356

theorem arithmetic_sequence_30th_term :
  let a₁ := 3
  let d := 4
  let n := 30
  a₁ + (n - 1) * d = 119 :=
by
  let a₁ := 3
  let d := 4
  let n := 30
  show a₁ + (n - 1) * d = 119
  sorry

end arithmetic_sequence_30th_term_l5_5356


namespace find_e1_l5_5676

-- Definitions related to the problem statement
variable (P F1 F2 : Type)
variable (cos_angle : ℝ)
variable (e1 e2 : ℝ)

-- Conditions
def cosine_angle_condition := cos_angle = 3 / 5
def eccentricity_relation := e2 = 2 * e1

-- Theorem that needs to be proved
theorem find_e1 (h_cos : cosine_angle_condition cos_angle)
                (h_ecc_rel : eccentricity_relation e1 e2) :
  e1 = Real.sqrt 10 / 5 :=
by
  sorry

end find_e1_l5_5676


namespace arithmetic_sequence_ninth_term_l5_5890

theorem arithmetic_sequence_ninth_term :
  ∃ a d : ℤ, (a + 2 * d = 23) ∧ (a + 5 * d = 29) ∧ (a + 8 * d = 35) :=
by
  sorry

end arithmetic_sequence_ninth_term_l5_5890


namespace num_fish_when_discovered_l5_5145

open Nat

/-- Definition of the conditions given in the problem --/
def initial_fish := 60
def fish_per_day_eaten := 2
def additional_fish := 8
def weeks_before_addition := 2
def extra_week := 1

/-- The proof problem statement --/
theorem num_fish_when_discovered : 
  let days := (weeks_before_addition + extra_week) * 7
  let total_fish_eaten := days * fish_per_day_eaten
  let fish_after_addition := initial_fish + additional_fish
  let final_fish := fish_after_addition - total_fish_eaten
  final_fish = 26 := 
by
  let days := (weeks_before_addition + extra_week) * 7
  let total_fish_eaten := days * fish_per_day_eaten
  let fish_after_addition := initial_fish + additional_fish
  let final_fish := fish_after_addition - total_fish_eaten
  have h : final_fish = 26 := sorry
  exact h

end num_fish_when_discovered_l5_5145


namespace length_of_each_train_l5_5324

theorem length_of_each_train
  (L : ℝ) -- length of each train
  (speed_fast : ℝ) (speed_slow : ℝ) -- speeds of the fast and slow trains in km/hr
  (time_pass : ℝ) -- time for the slower train to pass the driver of the faster one in seconds
  (h_speed_fast : speed_fast = 45) -- speed of the faster train
  (h_speed_slow : speed_slow = 15) -- speed of the slower train
  (h_time_pass : time_pass = 60) -- time to pass
  (h_same_length : ∀ (x y : ℝ), x = y → x = L) :  
  L = 1000 :=
  by
  -- Skipping the proof as instructed
  sorry

end length_of_each_train_l5_5324


namespace tennis_to_soccer_ratio_l5_5467

theorem tennis_to_soccer_ratio
  (total_balls : ℕ)
  (soccer_balls : ℕ)
  (basketball_offset : ℕ)
  (baseball_offset : ℕ)
  (volleyballs : ℕ)
  (tennis_balls : ℕ)
  (total_balls_eq : total_balls = 145)
  (soccer_balls_eq : soccer_balls = 20)
  (basketball_count : soccer_balls + basketball_offset = 20 + 5)
  (baseball_count : soccer_balls + baseball_offset = 20 + 10)
  (volleyballs_eq : volleyballs = 30)
  (accounted_balls : soccer_balls + (soccer_balls + basketball_offset) + (soccer_balls + baseball_offset) + volleyballs = 105)
  (tennis_balls_eq : tennis_balls = 145 - 105) :
  tennis_balls / soccer_balls = 2 :=
sorry

end tennis_to_soccer_ratio_l5_5467


namespace perfect_squares_unique_l5_5531

theorem perfect_squares_unique (n : ℕ) (h1 : ∃ k : ℕ, 20 * n = k^2) (h2 : ∃ p : ℕ, 5 * n + 275 = p^2) :
  n = 125 :=
by
  sorry

end perfect_squares_unique_l5_5531


namespace no_positive_integer_solution_l5_5661

theorem no_positive_integer_solution (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  ¬ (∃ (k : ℕ), (xy + 1) * (xy + x + 2) = k^2) :=
by {
  sorry
}

end no_positive_integer_solution_l5_5661


namespace cattle_train_left_6_hours_before_l5_5508

theorem cattle_train_left_6_hours_before 
  (Vc : ℕ) (Vd : ℕ) (T : ℕ) 
  (h1 : Vc = 56)
  (h2 : Vd = Vc - 33)
  (h3 : 12 * Vd + 12 * Vc + T * Vc = 1284) : 
  T = 6 := 
by
  sorry

end cattle_train_left_6_hours_before_l5_5508


namespace correct_inequality_incorrect_inequality1_incorrect_inequality2_correct_option_d_l5_5494

theorem correct_inequality:
    (-21 : ℤ) > (-21 : ℤ) := by sorry

theorem incorrect_inequality1 :
    -abs (10 + 1 / 2) < (8 + 2 / 3) := by sorry

theorem incorrect_inequality2 :
    (-abs (7 + 2 / 3)) ≠ (- (- (7 + 2 / 3))) := by sorry

theorem correct_option_d :
    (-5 / 6 : ℚ) < (-4 / 5 : ℚ) := by sorry

end correct_inequality_incorrect_inequality1_incorrect_inequality2_correct_option_d_l5_5494


namespace part1_solution_part2_solution_part3_solution_l5_5768

-- Part (1): Prove the solution of the system of equations 
theorem part1_solution (x y : ℝ) (h1 : x - y - 1 = 0) (h2 : 4 * (x - y) - y = 5) : 
  x = 0 ∧ y = -1 := 
sorry

-- Part (2): Prove the solution of the system of equations 
theorem part2_solution (x y : ℝ) (h1 : 2 * x - 3 * y - 2 = 0) 
  (h2 : (2 * x - 3 * y + 5) / 7 + 2 * y = 9) : 
  x = 7 ∧ y = 4 := 
sorry

-- Part (3): Prove the range of the parameter m
theorem part3_solution (m : ℕ) (h1 : 2 * (2 : ℝ) * x + y = (-3 : ℝ) * ↑m + 2) 
  (h2 : x + 2 * y = 7) (h3 : x + y > -5 / 6) : 
  m = 1 ∨ m = 2 ∨ m = 3 :=
sorry

end part1_solution_part2_solution_part3_solution_l5_5768


namespace sort_mail_together_time_l5_5765

-- Definitions of work rates
def mail_handler_work_rate : ℚ := 1 / 3
def assistant_work_rate : ℚ := 1 / 6

-- Definition to calculate combined work time
def combined_time (rate1 rate2 : ℚ) : ℚ := 1 / (rate1 + rate2)

-- Statement to prove
theorem sort_mail_together_time :
  combined_time mail_handler_work_rate assistant_work_rate = 2 := by
  -- Proof goes here
  sorry

end sort_mail_together_time_l5_5765


namespace prove_pqrstu_eq_416_l5_5823

-- Define the condition 1728 * x^4 + 64 = (p * x^3 + q * x^2 + r * x + s) * (t * x + u) + v
def condition (p q r s t u v : ℤ) (x : ℤ) : Prop :=
  1728 * x^4 + 64 = (p * x^3 + q * x^2 + r * x + s) * (t * x + u) + v

-- State the theorem to prove p^2 + q^2 + r^2 + s^2 + t^2 + u^2 + v^2 = 416
theorem prove_pqrstu_eq_416 (p q r s t u v : ℤ) (h : ∀ x, condition p q r s t u v x) : 
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 + v^2 = 416 :=
sorry

end prove_pqrstu_eq_416_l5_5823


namespace fraction_eggs_used_for_cupcakes_l5_5898

theorem fraction_eggs_used_for_cupcakes:
  ∀ (total_eggs crepes_fraction remaining_eggs after_cupcakes_eggs used_for_cupcakes_fraction: ℚ),
  total_eggs = 36 →
  crepes_fraction = 1 / 4 →
  after_cupcakes_eggs = 9 →
  used_for_cupcakes_fraction = 2 / 3 →
  (total_eggs * (1 - crepes_fraction) - after_cupcakes_eggs) / (total_eggs * (1 - crepes_fraction)) = used_for_cupcakes_fraction :=
by
  intros
  sorry

end fraction_eggs_used_for_cupcakes_l5_5898


namespace solve_equation_l5_5123

theorem solve_equation (x : ℝ) : x * (x-3)^2 * (5+x) = 0 ↔ (x = 0 ∨ x = 3 ∨ x = -5) := 
by
  sorry

end solve_equation_l5_5123


namespace thirtieth_term_of_arithmetic_seq_l5_5360

def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem thirtieth_term_of_arithmetic_seq : 
  arithmetic_seq 3 4 30 = 119 := 
by
  sorry

end thirtieth_term_of_arithmetic_seq_l5_5360


namespace units_digit_sum_42_4_24_4_l5_5757

-- Define the units digit function
def units_digit (n : ℕ) : ℕ := n % 10

-- Given conditions
def units_digit_42_4 : units_digit (42^4) = 6 := sorry
def units_digit_24_4 : units_digit (24^4) = 6 := sorry

-- Theorem to prove
theorem units_digit_sum_42_4_24_4 :
  units_digit (42^4 + 24^4) = 2 :=
by
  -- Use the given conditions
  have h1 : units_digit (42^4) = 6 := units_digit_42_4
  have h2 : units_digit (24^4) = 6 := units_digit_24_4
  -- Calculate the units digit of their sum
  calc 
    units_digit (42^4 + 24^4)
        = units_digit (6 + 6) : by rw [h1, h2]
    ... = units_digit 12    : by norm_num
    ... = 2                 : by norm_num

end units_digit_sum_42_4_24_4_l5_5757


namespace simplify_fraction_l5_5722

open Complex

theorem simplify_fraction :
  (7 + 9 * I) / (3 - 4 * I) = 2.28 + 2.2 * I := 
by {
    -- We know that this should be true based on the provided solution,
    -- but we will place a placeholder here for the actual proof.
    sorry
}

end simplify_fraction_l5_5722


namespace find_N_l5_5433

theorem find_N (N : ℕ) :
  ((5 + 6 + 7 + 8) / 4 = (2014 + 2015 + 2016 + 2017) / N) → N = 1240 :=
by
  sorry

end find_N_l5_5433


namespace bird_families_flew_away_to_Africa_l5_5495

theorem bird_families_flew_away_to_Africa 
  (B : ℕ) (n : ℕ) (hB94 : B = 94) (hB_A_plus_n : B = n + 47) : n = 47 :=
by
  sorry

end bird_families_flew_away_to_Africa_l5_5495


namespace right_triangle_inequality_l5_5845

theorem right_triangle_inequality {a b c : ℝ} (h : c^2 = a^2 + b^2) : 
  a + b ≤ c * Real.sqrt 2 :=
sorry

end right_triangle_inequality_l5_5845


namespace maximize_probability_l5_5156

def integer_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def valid_pairs (lst : List Int) : List (Int × Int) :=
  List.filter (λ (pair : Int × Int), pair.fst ≠ pair.snd ∧ pair.fst + pair.snd = 12)
    (List.sigma lst lst)

def number_of_valid_pairs (lst : List Int) : Nat :=
  (valid_pairs lst).length

theorem maximize_probability : 
  ∃ (num : Int), num = 6 ∧ ∀ (lst' : List Int), 
  lst' = List.erase integer_list num → 
  number_of_valid_pairs lst' = number_of_valid_pairs (List.erase integer_list 6) :=
by
  sorry

end maximize_probability_l5_5156


namespace root_interval_l5_5597

noncomputable def f (a b x : ℝ) : ℝ := 2 * a^x - b^x

theorem root_interval (a b : ℝ) (h₀ : 0 < a) (h₁ : b ≥ 2 * a) :
  ∃ x : ℝ, 0 < x ∧ x ≤ 1 ∧ f a b x = 0 := 
sorry

end root_interval_l5_5597


namespace cube_relation_l5_5422

theorem cube_relation (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end cube_relation_l5_5422


namespace remainder_sum_first_seven_primes_div_eighth_prime_l5_5162

theorem remainder_sum_first_seven_primes_div_eighth_prime :
  let sum_of_first_seven_primes := 2 + 3 + 5 + 7 + 11 + 13 + 17 in
  let eighth_prime := 19 in
  sum_of_first_seven_primes % eighth_prime = 1 :=
by
  let sum_of_first_seven_primes := 2 + 3 + 5 + 7 + 11 + 13 + 17
  let eighth_prime := 19
  have : sum_of_first_seven_primes = 58 := by decide
  have : eighth_prime = 19 := rfl
  sorry

end remainder_sum_first_seven_primes_div_eighth_prime_l5_5162


namespace find_analytical_expression_function_increasing_inequality_solution_l5_5059

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

-- Conditions
variables {a b x : ℝ}
axiom odd_function : ∀ x : ℝ, f a b (-x) = -f a b x
axiom half_value : f a b (1/2) = 2/5

-- Questions/Statements

-- 1. Analytical expression
theorem find_analytical_expression :
  ∃ a b, f a b x = x / (1 + x^2) := 
sorry

-- 2. Increasing function
theorem function_increasing :
  ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → f 1 0 x1 < f 1 0 x2 := 
sorry

-- 3. Inequality solution
theorem inequality_solution :
  ∀ x : ℝ, (x ∈ Set.Ioo (-1) 0 ∪ Set.Ioo 0 ((-1 + Real.sqrt 5) / 2)) → f 1 0 (x^2 - 1) + f 1 0 x < 0 := 
sorry

end find_analytical_expression_function_increasing_inequality_solution_l5_5059


namespace trajectory_of_P_is_line_segment_l5_5272

open Real EuclideanGeometry

def F1 : Point := (-5, 0)
def F2 : Point := (5, 0)

def P (x y : ℝ) : Point := (x, y)

theorem trajectory_of_P_is_line_segment :
  ∀ (x y : ℝ), dist (P x y) F1 + dist (P x y) F2 = 10 →
  ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ P x y = (1 - t) • F1 + t • F2 :=
by
  sorry

end trajectory_of_P_is_line_segment_l5_5272


namespace difference_of_squares_401_399_l5_5177

theorem difference_of_squares_401_399 : 401^2 - 399^2 = 1600 :=
by
  sorry

end difference_of_squares_401_399_l5_5177


namespace right_triangle_third_side_product_l5_5326

theorem right_triangle_third_side_product (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  let leg1 := 6
  let leg2 := 8
  let hyp := real.sqrt (leg1^2 + leg2^2)
  let other_leg := real.sqrt (hyp^2 - leg1^2)
  let product := hyp * real.sqrt ((leg2)^2 - (leg1)^2)
  real.floor (product * 10 + 0.5) / 10 = 52.9 :=
by
  have hyp := real.sqrt (6^2 + 8^2)
  have hyp_eq : hyp = 10 := by norm_num [hyp]
  have other_leg := real.sqrt (8^2 - 6^2)
  have other_leg_approx : other_leg ≈ 5.29 := by norm_num [other_leg]
  have prod := hyp * other_leg
  have : product = 52.9 := sorry
  have r := real.floor (product * 10 + 0.5) / 10
  exact rfl

end right_triangle_third_side_product_l5_5326


namespace ajay_total_gain_l5_5649

theorem ajay_total_gain:
  let dal_A_kg := 15
  let dal_B_kg := 10
  let dal_C_kg := 12
  let dal_D_kg := 8
  let rate_A := 14.50
  let rate_B := 13
  let rate_C := 16
  let rate_D := 18
  let selling_rate := 17.50
  let cost_A := dal_A_kg * rate_A
  let cost_B := dal_B_kg * rate_B
  let cost_C := dal_C_kg * rate_C
  let cost_D := dal_D_kg * rate_D
  let total_cost := cost_A + cost_B + cost_C + cost_D
  let total_weight := dal_A_kg + dal_B_kg + dal_C_kg + dal_D_kg
  let total_selling_price := total_weight * selling_rate
  let gain := total_selling_price - total_cost
  gain = 104 := by
    sorry

end ajay_total_gain_l5_5649


namespace max_value_expression_l5_5671

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  let A := (a^3 + b^3 + c^3) / ((a + b + c)^3 - 26 * a * b * c) in 
  A ≤ 3 :=
  sorry

end max_value_expression_l5_5671


namespace total_donation_l5_5289

theorem total_donation : 2 + 6 + 2 + 8 = 18 := 
by sorry

end total_donation_l5_5289


namespace seq_a_ge_two_pow_nine_nine_l5_5230

theorem seq_a_ge_two_pow_nine_nine (a : ℕ → ℤ) 
  (h0 : a 1 > a 0)
  (h1 : a 1 > 0)
  (h2 : ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r) : 
  a 100 > 2^99 :=
sorry

end seq_a_ge_two_pow_nine_nine_l5_5230


namespace monthly_salary_l5_5985

variable (S : ℝ)
variable (Saves : ℝ)
variable (NewSaves : ℝ)

open Real

theorem monthly_salary (h1 : Saves = 0.30 * S)
                       (h2 : NewSaves = Saves - 0.25 * Saves)
                       (h3 : NewSaves = 400) :
    S = 1777.78 := by
    sorry

end monthly_salary_l5_5985


namespace range_of_a_l5_5563

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 4 * x + a ≥ 0) → a ≥ 4 :=
by
  sorry

end range_of_a_l5_5563


namespace greatest_common_divisor_84_n_l5_5753

theorem greatest_common_divisor_84_n :
  ∃ (n : ℕ), (∀ (d : ℕ), d ∣ 84 ∧ d ∣ n → d = 1 ∨ d = 2 ∨ d = 4) ∧ (∀ (x y : ℕ), x ∣ 84 ∧ x ∣ n ∧ y ∣ 84 ∧ y ∣ n → x ≤ y → y = 4) :=
sorry

end greatest_common_divisor_84_n_l5_5753


namespace chef_earns_less_than_manager_l5_5185

theorem chef_earns_less_than_manager :
  let manager_wage := 7.50
  let dishwasher_wage := manager_wage / 2
  let chef_wage := dishwasher_wage * 1.20
  (manager_wage - chef_wage) = 3.00 := by
    sorry

end chef_earns_less_than_manager_l5_5185


namespace fewest_fence_posts_l5_5642

def fence_posts (length_wide short_side long_side : ℕ) (post_interval : ℕ) : ℕ :=
  let wide_side_posts := (long_side / post_interval) + 1
  let short_side_posts := (short_side / post_interval)
  wide_side_posts + 2 * short_side_posts

theorem fewest_fence_posts : fence_posts 40 10 100 10 = 19 :=
  by
    -- The proof will be completed here
    sorry

end fewest_fence_posts_l5_5642


namespace problem_l5_5880

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

def sum_arithmetic_sequence (a d n : ℕ) : ℕ := n * a + (n * (n - 1) * d) / 2

theorem problem (a1 S3 : ℕ) (a1_eq : a1 = 2) (S3_eq : S3 = 12) : 
  ∃ a6 : ℕ, a6 = 12 := by
  let a2 := (S3 - a1) / 2
  let d := a2 - a1
  let a6 := a1 + 5 * d
  use a6
  sorry

end problem_l5_5880


namespace molecular_weight_acetic_acid_l5_5755

-- Define atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the number of each atom in acetic acid
def num_C : ℕ := 2
def num_H : ℕ := 4
def num_O : ℕ := 2

-- Define the molecular formula of acetic acid
def molecular_weight_CH3COOH : ℝ :=
  num_C * atomic_weight_C +
  num_H * atomic_weight_H +
  num_O * atomic_weight_O

-- State the proposition
theorem molecular_weight_acetic_acid :
  molecular_weight_CH3COOH = 60.052 := by
  sorry

end molecular_weight_acetic_acid_l5_5755


namespace simplified_expression_evaluation_l5_5723

theorem simplified_expression_evaluation (x : ℝ) (hx : x = Real.sqrt 7) :
    (2 * x + 3) * (2 * x - 3) - (x + 2)^2 + 4 * (x + 3) = 20 :=
by
  sorry

end simplified_expression_evaluation_l5_5723


namespace mean_variance_transformation_l5_5478

variable (n : ℕ)
variable (x : Fin n → ℝ)
variable (mean_original variance_original : ℝ)
variable (meam_new variance_new : ℝ)
variable (offset : ℝ)

theorem mean_variance_transformation (hmean : mean_original = 2.8) (hvariance : variance_original = 3.6) 
  (hoffset : offset = 60) : 
  (mean_new = mean_original + offset) ∧ (variance_new = variance_original) :=
  sorry

end mean_variance_transformation_l5_5478


namespace parallel_slope_l5_5951

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l5_5951


namespace max_value_y_eq_x_mul_2_minus_x_min_value_y_eq_x_plus_4_div_x_minus_3_l5_5769

theorem max_value_y_eq_x_mul_2_minus_x (x : ℝ) (h : 0 < x ∧ x < 3 / 2) : ∃ y : ℝ, y = x * (2 - x) ∧ y ≤ 1 :=
sorry

theorem min_value_y_eq_x_plus_4_div_x_minus_3 (x : ℝ) (h : x > 3) : ∃ y : ℝ, y = x + 4 / (x - 3) ∧ y ≥ 7 :=
sorry

end max_value_y_eq_x_mul_2_minus_x_min_value_y_eq_x_plus_4_div_x_minus_3_l5_5769


namespace ratio_x_y_l5_5828

theorem ratio_x_y (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) : x / y = 11 / 6 :=
by
  sorry

end ratio_x_y_l5_5828


namespace anne_find_bottle_caps_l5_5999

theorem anne_find_bottle_caps 
  (n_i n_f : ℕ) (h_initial : n_i = 10) (h_final : n_f = 15) : n_f - n_i = 5 :=
by
  sorry

end anne_find_bottle_caps_l5_5999


namespace family_eggs_count_l5_5980

theorem family_eggs_count : 
  ∀ (initial_eggs parent_use child_use : ℝ) (chicken1 chicken2 chicken3 chicken4 : ℝ), 
    initial_eggs = 25 →
    parent_use = 7.5 + 2.5 →
    chicken1 = 2.5 →
    chicken2 = 3 →
    chicken3 = 4.5 →
    chicken4 = 1 →
    child_use = 1.5 + 0.5 →
    (initial_eggs - parent_use + (chicken1 + chicken2 + chicken3 + chicken4) - child_use) = 24 :=
by
  intros initial_eggs parent_use child_use chicken1 chicken2 chicken3 chicken4 
         h_initial_eggs h_parent_use h_chicken1 h_chicken2 h_chicken3 h_chicken4 h_child_use
  -- Proof goes here
  sorry

end family_eggs_count_l5_5980


namespace missed_the_bus_by_5_minutes_l5_5338

theorem missed_the_bus_by_5_minutes 
    (usual_time : ℝ)
    (new_time : ℝ)
    (h1 : usual_time = 20)
    (h2 : new_time = usual_time * (5 / 4)) : 
    new_time - usual_time = 5 := 
by
  sorry

end missed_the_bus_by_5_minutes_l5_5338


namespace max_value_ad_bc_l5_5417

theorem max_value_ad_bc (a b c d : ℤ) (h₁ : a ∈ ({-1, 1, 2} : Set ℤ))
                          (h₂ : b ∈ ({-1, 1, 2} : Set ℤ))
                          (h₃ : c ∈ ({-1, 1, 2} : Set ℤ))
                          (h₄ : d ∈ ({-1, 1, 2} : Set ℤ)) :
  ad - bc ≤ 6 :=
by sorry

end max_value_ad_bc_l5_5417


namespace school_children_equation_l5_5367

theorem school_children_equation
  (C B : ℕ)
  (h1 : B = 2 * C)
  (h2 : B = 4 * (C - 350)) :
  C = 700 := by
  sorry

end school_children_equation_l5_5367


namespace convert_polar_to_rectangular_l5_5652

noncomputable def polarToRectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem convert_polar_to_rectangular :
  polarToRectangular 8 (7 * Real.pi / 6) = (-4 * Real.sqrt 3, -4) :=
by
  sorry

end convert_polar_to_rectangular_l5_5652


namespace arithmetic_sequence_ninth_term_l5_5891

theorem arithmetic_sequence_ninth_term :
  ∃ a d : ℤ, (a + 2 * d = 23) ∧ (a + 5 * d = 29) ∧ (a + 8 * d = 35) :=
by
  sorry

end arithmetic_sequence_ninth_term_l5_5891


namespace problem1_solution_problem2_solution_l5_5279

-- Problem 1: System of Equations
theorem problem1_solution (x y : ℝ) (h_eq1 : x - y = 2) (h_eq2 : 2 * x + y = 7) : x = 3 ∧ y = 1 :=
by {
  sorry -- Proof to be filled in
}

-- Problem 2: Fractional Equation
theorem problem2_solution (y : ℝ) (h_eq : 3 / (1 - y) = y / (y - 1) - 5) : y = 2 :=
by {
  sorry -- Proof to be filled in
}

end problem1_solution_problem2_solution_l5_5279


namespace area_of_square_B_l5_5130

theorem area_of_square_B (c : ℝ) (hA : ∃ sA, sA * sA = 2 * c^2) (hB : ∃ sA, exists sB, sB * sB = 3 * (sA * sA)) : 
∃ sB, sB * sB = 6 * c^2 :=
by
  sorry

end area_of_square_B_l5_5130


namespace circus_capacity_l5_5298

theorem circus_capacity (sections : ℕ) (people_per_section : ℕ) (h1 : sections = 4) (h2 : people_per_section = 246) :
  sections * people_per_section = 984 :=
by
  sorry

end circus_capacity_l5_5298


namespace parabola_zero_sum_l5_5287

-- Define the original parabola equation and transformations
def original_parabola (x : ℝ) : ℝ := (x - 3) ^ 2 + 4

-- Define the resulting parabola after transformations
def transformed_parabola (x : ℝ) : ℝ := -(x - 7) ^ 2 + 1

-- Prove that the resulting parabola has zeros at p and q such that p + q = 14
theorem parabola_zero_sum : 
  ∃ (p q : ℝ), transformed_parabola p = 0 ∧ transformed_parabola q = 0 ∧ p + q = 14 :=
by
  sorry

end parabola_zero_sum_l5_5287


namespace sufficient_but_not_necessary_l5_5368

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 1 → x > 0) ∧ ¬ (x > 0 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_l5_5368


namespace part1_part2_l5_5861

-- Part (1)
theorem part1 (x y : ℚ) 
  (h1 : 2022 * x + 2020 * y = 2021)
  (h2 : 2023 * x + 2021 * y = 2022) :
  x = 1/2 ∧ y = 1/2 :=
by
  -- Placeholder for the proof
  sorry

-- Part (2)
theorem part2 (x y a b : ℚ)
  (ha : a ≠ b) 
  (h1 : (a + 1) * x + (a - 1) * y = a)
  (h2 : (b + 1) * x + (b - 1) * y = b) :
  x = 1/2 ∧ y = 1/2 :=
by
  -- Placeholder for the proof
  sorry

end part1_part2_l5_5861


namespace min_voters_l5_5095

theorem min_voters (total_voters : ℕ) (districts : ℕ) (sections_per_district : ℕ) 
  (voters_per_section : ℕ) (majority_sections : ℕ) (majority_districts : ℕ) 
  (winner : string) (is_tall_winner : winner = "Tall") 
  (total_voters = 105) (districts = 5) (sections_per_district = 7) 
  (voters_per_section = 3) (majority_sections = 4) (majority_districts = 3) :
  ∃ (min_voters : ℕ), min_voters = 24 :=
by
  sorry

end min_voters_l5_5095


namespace product_approximation_l5_5302

-- Define the approximation condition
def approxProduct (x y : ℕ) (approxX approxY : ℕ) : ℕ :=
  approxX * approxY

-- State the theorem
theorem product_approximation :
  let x := 29
  let y := 32
  let approxX := 30
  let approxY := 30
  approxProduct x y approxX approxY = 900 := by
  sorry

end product_approximation_l5_5302


namespace polynomial_value_l5_5595

def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_value (a b c d : ℝ) 
  (h1 : P 1 a b c d = 1993) 
  (h2 : P 2 a b c d = 3986) 
  (h3 : P 3 a b c d = 5979) :
  (1 / 4) * (P 11 a b c d + P (-7) a b c d) = 4693 := 
by 
  sorry

end polynomial_value_l5_5595


namespace slope_of_parallel_line_l5_5934

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l5_5934


namespace average_weight_is_15_l5_5681

-- Define the ages of the 10 children
def ages : List ℕ := [2, 3, 3, 5, 2, 6, 7, 3, 4, 5]

-- Define the regression function
def weight (age : ℕ) : ℕ := 2 * age + 7

-- Function to calculate average
def average (l : List ℕ) : ℕ := l.sum / l.length

-- Define the weights of the children based on the regression function
def weights : List ℕ := ages.map weight

-- State the theorem to find the average weight of the children
theorem average_weight_is_15 : average weights = 15 := by
  sorry

end average_weight_is_15_l5_5681


namespace A_visits_all_seats_iff_even_l5_5977

def move_distance_unique (n : ℕ) : Prop := 
  ∀ k l : ℕ, (1 ≤ k ∧ k < n) → (1 ≤ l ∧ l < n) → k ≠ l → (k ≠ l % n)

def visits_all_seats (n : ℕ) : Prop := 
  ∃ A : ℕ → ℕ, 
  (∀ (k : ℕ), 0 ≤ k ∧ k < n → (0 ≤ A k ∧ A k < n)) ∧ 
  (∀ (k : ℕ), 0 ≤ k ∧ k < n → ∃ (m : ℕ), m ≠ n ∧ A k ≠ (A m % n))

theorem A_visits_all_seats_iff_even (n : ℕ) :
  (move_distance_unique n ∧ visits_all_seats n) ↔ (n % 2 = 0) := 
sorry

end A_visits_all_seats_iff_even_l5_5977


namespace Taehyung_mother_age_l5_5613

theorem Taehyung_mother_age (Taehyung_young_brother_age : ℕ) (Taehyung_age_diff : ℕ) (Mother_age_diff : ℕ) (H1 : Taehyung_young_brother_age = 7) (H2 : Taehyung_age_diff = 5) (H3 : Mother_age_diff = 31) :
  ∃ (Mother_age : ℕ), Mother_age = 43 := 
by
  have Taehyung_age : ℕ := Taehyung_young_brother_age + Taehyung_age_diff
  have Mother_age := Taehyung_age + Mother_age_diff
  existsi (Mother_age)
  sorry

end Taehyung_mother_age_l5_5613


namespace find_counterfeit_coin_l5_5622

-- Define the context of the problem
variables (coins : Fin 6 → ℝ) -- six coins represented as a function from Fin 6 to their weights
          (is_counterfeit : Fin 6 → Prop) -- a predicate indicating if the coin is counterfeit
          (real_weight : ℝ) -- the unknown weight of a real coin

-- Existence assertion for the counterfeit coin
axiom exists_counterfeit : ∃ x, is_counterfeit x

-- Define the total weights of coins 1&2 and 3&4
def weight_1_2 := coins 0 + coins 1
def weight_3_4 := coins 2 + coins 3

-- Statement of the problem
theorem find_counterfeit_coin :
  (weight_1_2 = weight_3_4 → (is_counterfeit 4 ∨ is_counterfeit 5)) ∧ 
  (weight_1_2 ≠ weight_3_4 → (is_counterfeit 0 ∨ is_counterfeit 1 ∨ is_counterfeit 2 ∨ is_counterfeit 3)) :=
sorry

end find_counterfeit_coin_l5_5622


namespace initial_positions_2048_l5_5452

noncomputable def number_of_initial_positions (n : ℕ) : ℤ :=
  2 ^ n - 2

theorem initial_positions_2048 : number_of_initial_positions 2048 = 2 ^ 2048 - 2 :=
by
  sorry

end initial_positions_2048_l5_5452


namespace range_of_m_value_of_m_l5_5233

variable (α β m : ℝ)

open Real

-- Conditions: α and β are positive roots.
def quadratic_roots (α β m : ℝ) : Prop :=
  (α > 0) ∧ (β > 0) ∧ (α + β = 1 - 2*m) ∧ (α * β = m^2)

-- Part 1: Range of values for m.
theorem range_of_m (h : quadratic_roots α β m) : m ≤ 1/4 ∧ m ≠ 0 :=
sorry

-- Part 2: Given α^2 + β^2 = 49, find the value of m.
theorem value_of_m (h : quadratic_roots α β m) (h' : α^2 + β^2 = 49) : m = -4 :=
sorry

end range_of_m_value_of_m_l5_5233


namespace sum_series_eq_l5_5392

theorem sum_series_eq :
  ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1) = 3 / 2 :=
sorry

end sum_series_eq_l5_5392


namespace area_equality_l5_5255

-- Define the problem structure

variable {Point Line : Type}
variable [Euclidean_geometry Point Line]

-- Assume basic geometric entities and properties
variables {A B C D E F H Q R U V M N : Point}
variables {Γ Γ' : Circle Point}
variables {AD BE CF AQ QR RV : Line}
variable (AM HN : Line)

-- Conditions from the problem statement
hypothesis (h1 : altitude AD ∆ABC)
hypothesis (h2 : altitude BE ∆ABC)
hypothesis (h3 : altitude CF ∆ABC)
hypothesis (h4 : orthocenter H ∆ABC)
hypothesis (h5 : Q ∈ circumcircle ∆ABC)
hypothesis (h6 : QR ⊥ BC at R)
hypothesis (h7 : line_through R ∥ line_through AQ)
hypothesis (h8 : line_through R ∥ circumcircle ∆DEF at U V)
hypothesis (h9 : AM ⊥ RV at M)
hypothesis (h10 : HN ⊥ RV at N)

-- Goal to prove
theorem area_equality : area (∆AMV) = area (∆HNV) := by
  sorry

end area_equality_l5_5255


namespace determine_a_l5_5826

theorem determine_a (a : ℝ) : (∃! x : ℝ, a * x^2 + 2 * x + 1 = 0) → (a = 0 ∨ a = 1) := 
sorry

end determine_a_l5_5826


namespace slope_of_parallel_line_l5_5924

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l5_5924


namespace problem_I_problem_II_l5_5239

-- Definition of the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Problem (I): Prove solution set
theorem problem_I (x : ℝ) : f (x - 1) + f (x + 3) ≥ 6 ↔ (x ≤ -3 ∨ x ≥ 3) := by
  sorry

-- Problem (II): Prove inequality given conditions
theorem problem_II (a b : ℝ) (ha: |a| < 1) (hb: |b| < 1) (hano: a ≠ 0) : 
  f (a * b) > |a| * f (b / a) := by
  sorry

end problem_I_problem_II_l5_5239


namespace condition_for_ellipse_l5_5188

theorem condition_for_ellipse (m : ℝ) : 
  (3 < m ∧ m < 7) ↔ (7 - m > 0 ∧ m - 3 > 0 ∧ (7 - m) ≠ (m - 3)) :=
by sorry

end condition_for_ellipse_l5_5188


namespace twice_perimeter_of_square_l5_5381

theorem twice_perimeter_of_square (s : ℝ) (h : s^2 = 625) : 2 * 4 * s = 200 :=
by sorry

end twice_perimeter_of_square_l5_5381


namespace oates_reunion_attendees_l5_5751

noncomputable def total_guests : ℕ := 100
noncomputable def hall_attendees : ℕ := 70
noncomputable def both_reunions_attendees : ℕ := 10

theorem oates_reunion_attendees :
  ∃ O : ℕ, total_guests = O + hall_attendees - both_reunions_attendees ∧ O = 40 :=
by
  sorry

end oates_reunion_attendees_l5_5751


namespace lexi_laps_l5_5853

theorem lexi_laps (total_distance lap_distance : ℝ) (h1 : total_distance = 3.25) (h2 : lap_distance = 0.25) :
  total_distance / lap_distance = 13 :=
by
  sorry

end lexi_laps_l5_5853


namespace negation_of_proposition_range_of_m_l5_5817

noncomputable def proposition (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * x - m - 1 < 0

theorem negation_of_proposition (m : ℝ) : ¬ proposition m ↔ ∀ x : ℝ, x^2 + 2 * x - m - 1 ≥ 0 :=
sorry

theorem range_of_m (m : ℝ) : proposition m → m > -2 :=
sorry

end negation_of_proposition_range_of_m_l5_5817


namespace probability_heads_twice_in_three_flips_l5_5498

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_heads_twice_in_three_flips :
  let p := 0.5
  let n := 3
  let k := 2
  (binomial_coefficient n k : ℝ) * p^k * (1 - p)^(n - k) = 0.375 :=
by
  sorry

end probability_heads_twice_in_three_flips_l5_5498


namespace jesse_bananas_total_l5_5104

theorem jesse_bananas_total (friends : ℝ) (bananas_per_friend : ℝ) (friends_eq : friends = 3) (bananas_per_friend_eq : bananas_per_friend = 21) : 
  friends * bananas_per_friend = 63 := by
  rw [friends_eq, bananas_per_friend_eq]
  norm_num

end jesse_bananas_total_l5_5104


namespace fourth_power_sum_l5_5566

theorem fourth_power_sum (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a^2 + b^2 + c^2 = 3) 
  (h3 : a^3 + b^3 + c^3 = 6) : 
  a^4 + b^4 + c^4 = 4.5 :=
by
  sorry

end fourth_power_sum_l5_5566


namespace triangle_ratio_l5_5264

-- Define the triangle FGH and points X and Y
structure Triangle :=
  (F G H : ℝ)

-- Define the points and their ratios
structure Points :=
  (X Y : ℝ)
  (HX XF HY YF : ℝ)

-- Conditions:
-- 1. Line through X parallel to FG
-- 2. Ratio HX : XF = 4 : 1
-- 3. Areas of shaded regions through X and Y are equal

def HX_XF_Ratio (points : Points) : Prop :=
  points.HX / points.XF = 4

def Areas_Equal (triangle : Triangle) (points : Points) : Prop :=
  -- Placeholder, Real makes it more practical to deduce areas
  sorry

def HY_YF_Ratio (points : Points) : Prop :=
  points.HY / points.YF = 3 / 2

-- The theorem to be proven
theorem triangle_ratio 
  (triangle : Triangle)
  (points : Points)
  (h_ratio : HX_XF_Ratio(points))
  (h_areas_equal : Areas_Equal(triangle, points)) :
  HY_YF_Ratio(points) :=
sorry

end triangle_ratio_l5_5264


namespace flower_beds_fraction_l5_5192

-- Definitions based on given conditions
def yard_length := 30
def yard_width := 6
def trapezoid_parallel_side1 := 20
def trapezoid_parallel_side2 := 30
def flower_bed_leg := (trapezoid_parallel_side2 - trapezoid_parallel_side1) / 2
def flower_bed_area := (1 / 2) * flower_bed_leg ^ 2
def total_flower_bed_area := 2 * flower_bed_area
def yard_area := yard_length * yard_width
def occupied_fraction := total_flower_bed_area / yard_area

-- Statement to prove
theorem flower_beds_fraction :
  occupied_fraction = 5 / 36 :=
by
  -- sorries to skip the proofs
  sorry

end flower_beds_fraction_l5_5192


namespace parallel_slope_l5_5954

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l5_5954


namespace kira_away_hours_l5_5108

theorem kira_away_hours (eats_per_hour : ℝ) (filled_kibble : ℝ) (left_kibble : ℝ) (eats_ratio : eats_per_hour = 1 / 4) 
  (filled_condition : filled_kibble = 3) (left_condition : left_kibble = 1) : (filled_kibble - left_kibble) / eats_per_hour = 8 :=
by
  have eats_per_hour_pos : eats_per_hour = 1 / 4 := eats_ratio
  rw [eats_per_hour_pos]
  have three_minus_one : filled_kibble - left_kibble = 2 := by
    rw [filled_condition, left_condition]
    norm_num
  rw [three_minus_one]
  norm_num
  sorry
 
end kira_away_hours_l5_5108


namespace right_triangle_third_side_product_l5_5327

theorem right_triangle_third_side_product (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  let leg1 := 6
  let leg2 := 8
  let hyp := real.sqrt (leg1^2 + leg2^2)
  let other_leg := real.sqrt (hyp^2 - leg1^2)
  let product := hyp * real.sqrt ((leg2)^2 - (leg1)^2)
  real.floor (product * 10 + 0.5) / 10 = 52.9 :=
by
  have hyp := real.sqrt (6^2 + 8^2)
  have hyp_eq : hyp = 10 := by norm_num [hyp]
  have other_leg := real.sqrt (8^2 - 6^2)
  have other_leg_approx : other_leg ≈ 5.29 := by norm_num [other_leg]
  have prod := hyp * other_leg
  have : product = 52.9 := sorry
  have r := real.floor (product * 10 + 0.5) / 10
  exact rfl

end right_triangle_third_side_product_l5_5327


namespace parallel_line_slope_l5_5943

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℚ, m = 1 / 2 :=
by {
  use 1 / 2,
  sorry
}

end parallel_line_slope_l5_5943


namespace harriet_siblings_product_l5_5065

-- Definitions based on conditions
def Harry_sisters : ℕ := 6
def Harry_brothers : ℕ := 3
def Harriet_sisters : ℕ := Harry_sisters - 1
def Harriet_brothers : ℕ := Harry_brothers

-- Statement to prove
theorem harriet_siblings_product : Harriet_sisters * Harriet_brothers = 15 := by
  -- Proof is skipped
  sorry

end harriet_siblings_product_l5_5065


namespace order_scores_l5_5837

theorem order_scores
  (J K M Q S : ℕ)
  (h1 : J ≥ Q) (h2 : J ≥ M) (h3 : J ≥ S) (h4 : J ≥ K)
  (h5 : M > Q ∨ M > S ∨ M > K)
  (h6 : K < S) (h7 : S < J) :
  K < S ∧ S < M ∧ M < Q :=
by
  sorry

end order_scores_l5_5837


namespace product_is_eight_l5_5111

noncomputable def compute_product (r : ℂ) (hr : r ≠ 1) (hr7 : r^7 = 1) : ℂ :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1)

theorem product_is_eight (r : ℂ) (hr : r ≠ 1) (hr7 : r^7 = 1) : compute_product r hr hr7 = 8 :=
by
  sorry

end product_is_eight_l5_5111


namespace mean_score_of_all_students_l5_5460

-- Conditions
def M : ℝ := 90
def A : ℝ := 75
def ratio (m a : ℝ) : Prop := m / a = 2 / 3

-- Question and correct answer
theorem mean_score_of_all_students (m a : ℝ) (hm : ratio m a) : (60 * a + 75 * a) / (5 * a / 3) = 81 := by
  sorry

end mean_score_of_all_students_l5_5460


namespace complex_numbers_are_real_l5_5864

theorem complex_numbers_are_real
  (a b c : ℂ)
  (h1 : (a + b) * (a + c) = b)
  (h2 : (b + c) * (b + a) = c)
  (h3 : (c + a) * (c + b) = a) : 
  a.im = 0 ∧ b.im = 0 ∧ c.im = 0 :=
sorry

end complex_numbers_are_real_l5_5864


namespace pyramid_base_edge_length_l5_5191

theorem pyramid_base_edge_length 
(radius_hemisphere height_pyramid : ℝ)
(h_radius : radius_hemisphere = 4)
(h_height : height_pyramid = 10)
(h_tangent : ∀ face : ℝ, True) : 
∃ s : ℝ, s = 2 * Real.sqrt 42 :=
by
  sorry

end pyramid_base_edge_length_l5_5191


namespace volume_of_sphere_l5_5680

theorem volume_of_sphere
  (a b c : ℝ)
  (h1 : a * b * c = 4 * Real.sqrt 6)
  (h2 : a * b = 2 * Real.sqrt 3)
  (h3 : b * c = 4 * Real.sqrt 3)
  (O_radius : ℝ := Real.sqrt (a^2 + b^2 + c^2) / 2) :
  4 / 3 * Real.pi * O_radius^3 = 32 * Real.pi / 3 := by
  sorry

end volume_of_sphere_l5_5680


namespace probability_linda_picks_letter_in_mathematics_l5_5573

def english_alphabet : Finset Char := "ABCDEFGHIJKLMNOPQRSTUVWXYZ".toList.toFinset

def word_mathematics : Finset Char := "MATHEMATICS".toList.toFinset

theorem probability_linda_picks_letter_in_mathematics : 
  (word_mathematics.card : ℚ) / (english_alphabet.card : ℚ) = 4 / 13 := by sorry

end probability_linda_picks_letter_in_mathematics_l5_5573


namespace parallel_line_slope_l5_5919

def slope_of_parallel_line : ℚ :=
  let line_eq := (3 : ℚ) * (x : ℚ) - (6 : ℚ) * (y : ℚ) = (12 : ℚ)
  in (1 / 2 : ℚ)

theorem parallel_line_slope (x y : ℚ) (h : 3 * x - 6 * y = 12) : slope_of_parallel_line = 1 / 2 :=
by
  sorry

end parallel_line_slope_l5_5919


namespace compute_P_part_l5_5593

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem compute_P_part (a b c d : ℝ) 
  (H1 : P 1 a b c d = 1993) 
  (H2 : P 2 a b c d = 3986) 
  (H3 : P 3 a b c d = 5979) : 
  (1 / 4) * (P 11 a b c d + P (-7) a b c d) = 4693 :=
by
  sorry

end compute_P_part_l5_5593


namespace seated_men_l5_5142

def passengers : Nat := 48
def fraction_of_women : Rat := 2/3
def fraction_of_men_standing : Rat := 1/8

theorem seated_men (men women standing seated : Nat) 
  (h1 : women = passengers * fraction_of_women)
  (h2 : men = passengers - women)
  (h3 : standing = men * fraction_of_men_standing)
  (h4 : seated = men - standing) :
  seated = 14 := by
  sorry

end seated_men_l5_5142


namespace number_of_students_in_class_l5_5782

theorem number_of_students_in_class
  (x : ℕ)
  (S : ℝ)
  (incorrect_score correct_score : ℝ)
  (incorrect_score_mistake : incorrect_score = 85)
  (correct_score_corrected : correct_score = 78)
  (average_difference : ℝ)
  (average_difference_value : average_difference = 0.75)
  (test_attendance : ℕ)
  (test_attendance_value : test_attendance = x - 3)
  (average_difference_condition : (S + incorrect_score) / test_attendance - (S + correct_score) / test_attendance = average_difference) :
  x = 13 :=
by
  sorry

end number_of_students_in_class_l5_5782


namespace decagon_adjacent_probability_l5_5314

-- Define the setup of a decagon with 10 vertices and the adjacency relation.
def is_decagon (n : ℕ) : Prop := n = 10

def adjacent_vertices (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v1 = (v2 + 9) % 10)

-- The main theorem statement, proving the probability of two vertices being adjacent.
theorem decagon_adjacent_probability (n : ℕ) (v1 v2 : ℕ) (h : is_decagon n) : 
  ∀ (v1 v2 : ℕ), v1 ≠ v2 → v1 < n → v2 < n →
  (∃ (p : ℚ), p = 2 / 9 ∧ 
  (adjacent_vertices v1 v2) ↔ (p = 2 / (n - 1))) :=
sorry

end decagon_adjacent_probability_l5_5314


namespace bus_dispatch_interval_l5_5496

/--
Xiao Hua walks at a constant speed along the route of the "Chunlei Cup" bus.
He encounters a "Chunlei Cup" bus every 6 minutes head-on and is overtaken by a "Chunlei Cup" bus every 12 minutes.
Assume "Chunlei Cup" buses are dispatched at regular intervals, travel at a constant speed, and do not stop at any stations along the way.
Prove that the time interval between bus departures is 8 minutes.
-/
theorem bus_dispatch_interval
  (encounters_opposite_direction: ℕ)
  (overtakes_same_direction: ℕ)
  (constant_speed: Prop)
  (regular_intervals: Prop)
  (no_stops: Prop)
  (h1: encounters_opposite_direction = 6)
  (h2: overtakes_same_direction = 12)
  (h3: constant_speed)
  (h4: regular_intervals)
  (h5: no_stops) :
  True := 
sorry

end bus_dispatch_interval_l5_5496


namespace complete_laps_l5_5851

-- Definitions based on conditions
def total_distance := 3.25  -- total distance Lexi wants to run
def lap_distance := 0.25    -- distance of one lap

-- Proof statement: Total number of complete laps to cover the given distance
theorem complete_laps (h1 : total_distance = 3 + 1/4) (h2 : lap_distance = 1/4) :
  (total_distance / lap_distance) = 13 :=
by 
  sorry

end complete_laps_l5_5851


namespace num_points_C_l5_5238

theorem num_points_C (
  A B : ℝ × ℝ)
  (C : ℝ × ℝ) 
  (hA : A = (2, 2))
  (hB : B = (-1, -2))
  (hC : (C.1 - 3)^2 + (C.2 + 5)^2 = 36)
  (h_area : 1/2 * (abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))) = 5/2) :
  ∃ C1 C2 C3 : ℝ × ℝ,
    (C1.1 - 3)^2 + (C1.2 + 5)^2 = 36 ∧
    (C2.1 - 3)^2 + (C2.2 + 5)^2 = 36 ∧
    (C3.1 - 3)^2 + (C3.2 + 5)^2 = 36 ∧
    1/2 * (abs ((B.1 - A.1) * (C1.2 - A.2) - (B.2 - A.2) * (C1.1 - A.1))) = 5/2 ∧
    1/2 * (abs ((B.1 - A.1) * (C2.2 - A.2) - (B.2 - A.2) * (C2.1 - A.1))) = 5/2 ∧
    1/2 * (abs ((B.1 - A.1) * (C3.2 - A.2) - (B.2 - A.2) * (C3.1 - A.1))) = 5/2 ∧
    (C1 ≠ C2 ∧ C1 ≠ C3 ∧ C2 ≠ C3) :=
sorry

end num_points_C_l5_5238


namespace intersection_P_Q_l5_5598

def P (x : ℝ) : Prop := x^2 - x - 2 ≥ 0

def Q (y : ℝ) : Prop := ∃ x, P x ∧ y = (1/2) * x^2 - 1

theorem intersection_P_Q :
  {m | ∃ (x : ℝ), P x ∧ m = (1/2) * x^2 - 1} = {m | m ≥ 2} := sorry

end intersection_P_Q_l5_5598


namespace second_odd_integer_is_72_l5_5618

def consecutive_odd_integers (n : ℤ) : ℤ × ℤ × ℤ :=
  (n - 2, n, n + 2)

theorem second_odd_integer_is_72 (n : ℤ) (h : (n - 2) + (n + 2) = 144) : n = 72 :=
by {
  sorry
}

end second_odd_integer_is_72_l5_5618


namespace range_of_a_l5_5550

  variable {A : Set ℝ} {B : Set ℝ}
  variable {a : ℝ}

  def A_def (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ 2 * a - 4 }
  def B_def : Set ℝ := { x | -1 < x ∧ x < 6 }

  theorem range_of_a (h : A_def a ∩ B_def = A_def a) : a < 5 :=
  sorry
  
end range_of_a_l5_5550


namespace min_value_of_F_l5_5859

variable (x1 x2 : ℝ)

def constraints :=
  2 - 2 * x1 - x2 ≥ 0 ∧
  2 - x1 + x2 ≥ 0 ∧
  5 - x1 - x2 ≥ 0 ∧
  0 ≤ x1 ∧
  0 ≤ x2

noncomputable def F := x2 - x1

theorem min_value_of_F : constraints x1 x2 → ∃ (minF : ℝ), minF = -2 :=
by
  sorry

end min_value_of_F_l5_5859


namespace total_crew_members_l5_5772

def num_islands : ℕ := 3
def ships_per_island : ℕ := 12
def crew_per_ship : ℕ := 24

theorem total_crew_members : num_islands * ships_per_island * crew_per_ship = 864 := by
  sorry

end total_crew_members_l5_5772


namespace ratio_M_N_l5_5715

variable {R P M N : ℝ}

theorem ratio_M_N (h1 : P = 0.3 * R) (h2 : M = 0.35 * R) (h3 : N = 0.55 * R) : M / N = 7 / 11 := by
  sorry

end ratio_M_N_l5_5715


namespace area_of_R2_l5_5419

theorem area_of_R2
  (a b : ℝ)
  (h1 : b = 3 * a)
  (h2 : a^2 + b^2 = 225) :
  a * b = 135 / 2 :=
by
  sorry

end area_of_R2_l5_5419


namespace percentage_value_l5_5979

variables {P a b c : ℝ}

theorem percentage_value (h1 : (P / 100) * a = 12) (h2 : (12 / 100) * b = 6) (h3 : c = b / a) : c = P / 24 :=
by
  sorry

end percentage_value_l5_5979


namespace product_abc_l5_5056

theorem product_abc (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_eqn : a * b * c = a * b^3) (h_c_eq_1 : c = 1) :
  a * b * c = a :=
by
  sorry

end product_abc_l5_5056


namespace smaller_angle_at_7_30_is_45_degrees_l5_5567

noncomputable def calculateAngle (hour minute : Nat) : Real :=
  let minuteAngle := (minute * 6 : Real)
  let hourAngle := (hour % 12 * 30 : Real) + (minute / 60 * 30 : Real)
  let diff := abs (hourAngle - minuteAngle)
  if diff > 180 then 360 - diff else diff

theorem smaller_angle_at_7_30_is_45_degrees :
  calculateAngle 7 30 = 45 := 
sorry

end smaller_angle_at_7_30_is_45_degrees_l5_5567


namespace quadratic_touches_x_axis_l5_5574

theorem quadratic_touches_x_axis (a : ℝ) : 
  (∃ x : ℝ, 2 * x ^ 2 - 8 * x + a = 0) ∧ (∀ y : ℝ, y^2 - 4 * a = 0 → y = 0) → a = 8 := 
by 
  sorry

end quadratic_touches_x_axis_l5_5574


namespace convert_denominators_to_integers_l5_5651

def original_equation (x : ℝ) : Prop :=
  (x + 1) / 0.4 - (0.2 * x - 1) / 0.7 = 1

def transformed_equation (x : ℝ) : Prop :=
  (10 * x + 10) / 4 - (2 * x - 10) / 7 = 1

theorem convert_denominators_to_integers (x : ℝ) 
  (h : original_equation x) : transformed_equation x :=
sorry

end convert_denominators_to_integers_l5_5651


namespace area_of_gray_region_l5_5704

theorem area_of_gray_region (r : ℝ) (h1 : r * 3 - r = 3) : 
  π * (3 * r) ^ 2 - π * r ^ 2 = 18 * π :=
by
  sorry

end area_of_gray_region_l5_5704


namespace nine_div_one_plus_four_div_x_eq_one_l5_5432

theorem nine_div_one_plus_four_div_x_eq_one (x : ℝ) (h : x = 0.5) : 9 / (1 + 4 / x) = 1 := by
  sorry

end nine_div_one_plus_four_div_x_eq_one_l5_5432


namespace probability_X_lt_0_l5_5553

noncomputable def X_distribution (σ : ℝ) (hσ : σ > 0) : MeasureTheory.ProbabilityTheory.NormalDist := 
  MeasureTheory.ProbabilityTheory.NormalDist.mk 2 σ

theorem probability_X_lt_0 (σ : ℝ) (hσ : σ > 0) (h : MeasureTheory.ProbabilityTheory.cdf (X_distribution σ hσ) 4 = 0.8) :
  MeasureTheory.ProbabilityTheory.cdf (X_distribution σ hσ) 0 = 0.2 :=
by
  sorry

end probability_X_lt_0_l5_5553


namespace increase_in_circumference_l5_5846

theorem increase_in_circumference (d e : ℝ) : (fun d e => let C := π * d; let C_new := π * (d + e); C_new - C) d e = π * e :=
by sorry

end increase_in_circumference_l5_5846


namespace arthur_speed_l5_5786

/-- Suppose Arthur drives to David's house and aims to arrive exactly on time. 
If he drives at 60 km/h, he arrives 5 minutes late. 
If he drives at 90 km/h, he arrives 5 minutes early. 
We want to find the speed n in km/h at which he arrives exactly on time. -/
theorem arthur_speed (n : ℕ) :
  (∀ t, 1 * (t + 5) = (3 / 2) * (t - 5)) → 
  (60 : ℝ) = 1 →
  (90 : ℝ) = (3 / 2) → 
  n = 72 := by
sorry

end arthur_speed_l5_5786


namespace transformer_coils_flawless_l5_5899

theorem transformer_coils_flawless (x y : ℕ) (hx : x + y = 8200)
  (hdef : (2 * x / 100) + (3 * y / 100) = 216) :
  ((x = 3000 ∧ y = 5200) ∧ ((x * 98 / 100) = 2940) ∧ ((y * 97 / 100) = 5044)) :=
by
  sorry

end transformer_coils_flawless_l5_5899


namespace sum_of_first_seven_primes_mod_eighth_prime_l5_5175

theorem sum_of_first_seven_primes_mod_eighth_prime :
  (2 + 3 + 5 + 7 + 11 + 13 + 17) % 19 = 1 :=
by
  sorry

end sum_of_first_seven_primes_mod_eighth_prime_l5_5175


namespace consecutive_numbers_square_sum_l5_5004

theorem consecutive_numbers_square_sum (n : ℕ) (a b : ℕ) (h1 : 2 * n + 1 = 144169^2)
  (h2 : a = 72084) (h3 : b = a + 1) : a^2 + b^2 = n + 1 :=
by
  sorry

end consecutive_numbers_square_sum_l5_5004


namespace value_of_x_l5_5629

theorem value_of_x (m n : ℝ) (z x : ℝ) (hz : z ≠ 0) (hx : x = m * (n / z) ^ 3) (hconst : 5 * (16 ^ 3) = m * (n ^ 3)) (hz_const : z = 64) : x = 5 / 64 :=
by
  -- proof omitted
  sorry

end value_of_x_l5_5629


namespace probability_adjacent_vertices_decagon_l5_5307

theorem probability_adjacent_vertices_decagon : 
  ∀ (decagon : Finset ℕ) (a b : ℕ), 
  decagon.card = 10 → 
  a ∈ decagon → 
  b ∈ decagon → 
  a ≠ b → 
  (P (adjacent a b decagon)) = 2 / 9 :=
by 
  -- we are asserting the probability that two randomly chosen vertices a and b from a decagon are adjacent.
  sorry

end probability_adjacent_vertices_decagon_l5_5307


namespace reciprocal_is_1_or_neg1_self_square_is_0_or_1_l5_5131

theorem reciprocal_is_1_or_neg1 (x : ℝ) (hx : x = 1 / x) :
  x = 1 ∨ x = -1 :=
sorry

theorem self_square_is_0_or_1 (x : ℝ) (hx : x = x^2) :
  x = 0 ∨ x = 1 :=
sorry

end reciprocal_is_1_or_neg1_self_square_is_0_or_1_l5_5131


namespace clara_loses_q_minus_p_l5_5258

def clara_heads_prob : ℚ := 2 / 3
def clara_tails_prob : ℚ := 1 / 3

def ethan_heads_prob : ℚ := 1 / 4
def ethan_tails_prob : ℚ := 3 / 4

def lose_prob_clara : ℚ := clara_heads_prob
def both_tails_prob : ℚ := clara_tails_prob * ethan_tails_prob

noncomputable def total_prob_clara_loses : ℚ :=
  lose_prob_clara + ∑' n : ℕ, (both_tails_prob ^ n) * lose_prob_clara

theorem clara_loses_q_minus_p :
  ∃ (p q : ℕ), Nat.gcd p q = 1 ∧ total_prob_clara_loses = p / q ∧ (q - p = 1) :=
sorry

end clara_loses_q_minus_p_l5_5258


namespace probability_adjacent_vertices_decagon_l5_5311

theorem probability_adjacent_vertices_decagon :
  let V := 10 in  -- Number of vertices in a decagon
  let favorable_outcomes := 2 in  -- Number of adjacent vertices to any chosen vertex
  let total_outcomes := V - 1 in  -- Total possible outcomes for the second vertex
  (favorable_outcomes / total_outcomes) = (2 / 9) :=
by
  let V := 10
  let favorable_outcomes := 2
  let total_outcomes := V - 1
  have probability := (favorable_outcomes / total_outcomes)
  have target_prob := (2 / 9)
  sorry

end probability_adjacent_vertices_decagon_l5_5311


namespace negation_of_proposition_p_l5_5877

theorem negation_of_proposition_p :
  (¬(∃ x : ℝ, 0 < x ∧ Real.log x > x - 1)) ↔ (∀ x : ℝ, 0 < x → Real.log x ≤ x - 1) :=
by
  sorry

end negation_of_proposition_p_l5_5877


namespace total_miles_l5_5606

-- Define the variables and equations as given in the conditions
variables (a b c d e : ℝ)
axiom h1 : a + b = 36
axiom h2 : b + c + d = 45
axiom h3 : c + d + e = 45
axiom h4 : a + c + e = 38

-- The conjecture we aim to prove
theorem total_miles : a + b + c + d + e = 83 :=
sorry

end total_miles_l5_5606


namespace largest_n_employees_in_same_quarter_l5_5437

theorem largest_n_employees_in_same_quarter (n : ℕ) (h1 : 72 % 4 = 0) (h2 : 72 / 4 = 18) : 
  n = 18 :=
sorry

end largest_n_employees_in_same_quarter_l5_5437


namespace maximum_xy_l5_5810

theorem maximum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_parallel : 2 * x + y = 2) : 
  xy ≤ 1/2 := 
  sorry

end maximum_xy_l5_5810


namespace ratio_blue_gill_to_bass_l5_5981

theorem ratio_blue_gill_to_bass (bass trout blue_gill : ℕ) 
  (h1 : bass = 32)
  (h2 : trout = bass / 4)
  (h3 : bass + trout + blue_gill = 104) 
: blue_gill / bass = 2 := 
sorry

end ratio_blue_gill_to_bass_l5_5981


namespace quadratic_solution_l5_5611

theorem quadratic_solution (x : ℝ) : x^2 - 2 * x - 3 = 0 → (x = 3 ∨ x = -1) :=
by
  sorry

end quadratic_solution_l5_5611


namespace legacy_earnings_per_hour_l5_5506

-- Define the conditions
def totalFloors : ℕ := 4
def roomsPerFloor : ℕ := 10
def hoursPerRoom : ℕ := 6
def totalEarnings : ℝ := 3600

-- The statement to prove
theorem legacy_earnings_per_hour :
  (totalFloors * roomsPerFloor * hoursPerRoom) = 240 → 
  (totalEarnings / (totalFloors * roomsPerFloor * hoursPerRoom)) = 15 := by
  intros h
  sorry

end legacy_earnings_per_hour_l5_5506


namespace gcd_1987_2025_l5_5904

theorem gcd_1987_2025 : Nat.gcd 1987 2025 = 1 := by
  sorry

end gcd_1987_2025_l5_5904


namespace area_below_line_l5_5490

noncomputable def circle_eqn (x y : ℝ) := 
  x^2 + 2 * x + (y^2 - 6 * y) + 50 = 0

noncomputable def line_eqn (x y : ℝ) := 
  y = x + 1

theorem area_below_line : 
  (∃ (x y : ℝ), circle_eqn x y ∧ y < x + 1) →
  ∃ (a : ℝ), a = 20 * π :=
by
  sorry

end area_below_line_l5_5490


namespace maximum_value_of_expression_l5_5672

theorem maximum_value_of_expression (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  (a^3 + b^3 + c^3) / ((a + b + c)^3 - 26 * a * b * c) ≤ 3 := by
  sorry

end maximum_value_of_expression_l5_5672


namespace sequence_a_n_general_formula_and_value_sequence_b_n_general_formula_l5_5554

theorem sequence_a_n_general_formula_and_value (a : ℕ → ℕ) 
  (h1 : a 1 = 3) 
  (h10 : a 10 = 21) 
  (h_linear : ∃ (k b : ℕ), ∀ n, a n = k * n + b) :
  (∀ n, a n = 2 * n + 1) ∧ a 2005 = 4011 :=
by 
  sorry

theorem sequence_b_n_general_formula (a b : ℕ → ℕ)
  (h_seq_a : ∀ n, a n = 2 * n + 1) 
  (h_b_formed : ∀ n, b n = a (2 * n)) : 
  ∀ n, b n = 4 * n + 1 :=
by 
  sorry

end sequence_a_n_general_formula_and_value_sequence_b_n_general_formula_l5_5554


namespace AM_bisects_BMC_l5_5551

variable (A B M C : Point) -- Points A, B, M, C
variable (AB_eq_BC : dist A B = dist B C) -- AB = BC
variable (angle_BAM_30 : angle B A M = 30) -- ∠BAM = 30°
variable (angle_ACM_150 : angle A C M = 150) -- ∠ACM = 150°
variable (ABC_convex : Convexquadvectg A B M C) -- A B M C is convex

theorem AM_bisects_BMC :
  isAngleBisector A M B C :=
by
  sorry

end AM_bisects_BMC_l5_5551


namespace mother_daughter_age_equality_l5_5854

theorem mother_daughter_age_equality :
  ∀ (x : ℕ), (24 * 12 + 3) + x = 12 * ((-5 : ℤ) + x) → x = 32 := 
by
  intros x h
  sorry

end mother_daughter_age_equality_l5_5854


namespace extreme_points_l5_5713

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b * x

theorem extreme_points (a b : ℝ) 
  (h1 : 3*(-2)^2 + 2*a*(-2) + b = 0) 
  (h2 : 3*(4)^2 + 2*a*(4) + b = 0) : 
  a - b = 21 :=
by sorry

end extreme_points_l5_5713


namespace factorization_correct_l5_5967

theorem factorization_correct :
  ∀ (x y : ℝ), 
    (¬ ( (y - 1) * (y + 1) = y^2 - 1 ) ) ∧
    (¬ ( x^2 * y + x * y^2 - 1 = x * y * (x + y) - 1 ) ) ∧
    (¬ ( (x - 2) * (x - 3) = (3 - x) * (2 - x) ) ) ∧
    ( x^2 - 4 * x + 4 = (x - 2)^2 ) :=
by
  intros x y
  repeat { constructor }
  all_goals { sorry }

end factorization_correct_l5_5967


namespace alice_cookie_fills_l5_5516

theorem alice_cookie_fills :
  (∀ (a b : ℚ), a = 3 + (3/4) ∧ b = 1/3 → (a / b) = 12) :=
sorry

end alice_cookie_fills_l5_5516


namespace slope_of_parallel_line_l5_5928

theorem slope_of_parallel_line (x y : ℝ) :
  ∀ (m : ℝ), (3 * x - 6 * y = 12) → (m = 1 / 2) :=
begin
  sorry
end

end slope_of_parallel_line_l5_5928


namespace num_perfect_cubes_between_bounds_l5_5067

   noncomputable def lower_bound := 2^8 + 1
   noncomputable def upper_bound := 2^18 + 1

   theorem num_perfect_cubes_between_bounds : 
     ∃ (k : ℕ), k = 58 ∧ (∀ (n : ℕ), (lower_bound ≤ n^3 ∧ n^3 ≤ upper_bound) ↔ (7 ≤ n ∧ n ≤ 64)) :=
   sorry
   
end num_perfect_cubes_between_bounds_l5_5067


namespace probability_of_adjacent_vertices_in_decagon_l5_5318

/-- Define the number of vertices in the decagon -/
def num_vertices : ℕ := 10

/-- Define the total number of ways to choose two distinct vertices from the decagon -/
def total_possible_outcomes : ℕ := num_vertices * (num_vertices - 1) / 2

/-- Define the number of favorable outcomes where the two chosen vertices are adjacent -/
def favorable_outcomes : ℕ := num_vertices

/-- Define the probability of selecting two adjacent vertices -/
def probability_adjacent_vertices : ℚ := favorable_outcomes / total_possible_outcomes

/-- The main theorem statement -/
theorem probability_of_adjacent_vertices_in_decagon : probability_adjacent_vertices = 2 / 9 := 
  sorry

end probability_of_adjacent_vertices_in_decagon_l5_5318


namespace find_XY_in_triangle_l5_5034

-- Definitions
def Triangle := Type
def angle_measures (T : Triangle) : (ℕ × ℕ × ℕ) := sorry
def side_lengths (T : Triangle) : (ℕ × ℕ × ℕ) := sorry
def is_30_60_90_triangle (T : Triangle) : Prop := (angle_measures T = (30, 60, 90))

-- Given conditions and statement we want to prove
def triangle_XYZ : Triangle := sorry
def XY : ℕ := 6

-- Proof statement
theorem find_XY_in_triangle :
  is_30_60_90_triangle triangle_XYZ ∧ (side_lengths triangle_XYZ).1 = XY →
  XY = 6 :=
by
  intro h
  sorry

end find_XY_in_triangle_l5_5034


namespace problem_one_problem_two_l5_5420

variable {α : ℝ}

theorem problem_one (h : Real.tan (π + α) = -1 / 2) :
  (2 * Real.cos (π - α) - 3 * Real.sin (π + α)) / (4 * Real.cos (α - 2 * π) + Real.sin (4 * π - α)) = -7 / 9 :=
sorry

theorem problem_two (h : Real.tan (π + α) = -1 / 2) :
  Real.sin (α - 7 * π) * Real.cos (α + 5 * π) = -2 / 5 :=
sorry

end problem_one_problem_two_l5_5420


namespace quadratic_smallest_root_a_quadratic_smallest_root_b_l5_5366

-- For Part (a)
theorem quadratic_smallest_root_a (a : ℝ) 
  (h : a^2 - 9 * a - 10 = 0 ∧ ∀ x, x^2 - 9 * x - 10 = 0 → x ≥ a) : 
  a^4 - 909 * a = 910 :=
by sorry

-- For Part (b)
theorem quadratic_smallest_root_b (b : ℝ) 
  (h : b^2 - 9 * b + 10 = 0 ∧ ∀ x, x^2 - 9 * x + 10 = 0 → x ≥ b) : 
  b^4 - 549 * b = -710 :=
by sorry

end quadratic_smallest_root_a_quadratic_smallest_root_b_l5_5366


namespace distance_apart_after_3_hours_l5_5522

-- Definitions derived from conditions
def Ann_speed (hour : ℕ) : ℕ :=
  if hour = 1 then 6 else if hour = 2 then 8 else 4

def Glenda_speed (hour : ℕ) : ℕ :=
  if hour = 1 then 8 else if hour = 2 then 5 else 9

-- The total distance function for a given skater
def total_distance (speed : ℕ → ℕ) : ℕ :=
  speed 1 + speed 2 + speed 3

-- Ann's total distance skated
def Ann_total_distance : ℕ := total_distance Ann_speed

-- Glenda's total distance skated
def Glenda_total_distance : ℕ := total_distance Glenda_speed

-- The total distance between Ann and Glenda after 3 hours
def total_distance_apart : ℕ := Ann_total_distance + Glenda_total_distance

-- Proof statement (without the proof itself; just the goal declaration)
theorem distance_apart_after_3_hours : total_distance_apart = 40 := by
  sorry

end distance_apart_after_3_hours_l5_5522


namespace set_membership_proof_l5_5667

variable (A : Set ℕ) (B : Set (Set ℕ))

theorem set_membership_proof :
  A = {0, 1} → B = {x | x ⊆ A} → A ∈ B :=
by
  intros hA hB
  rw [hA, hB]
  sorry

end set_membership_proof_l5_5667


namespace total_area_of_removed_triangles_l5_5781

theorem total_area_of_removed_triangles : 
  ∀ (side_length_of_square : ℝ) (hypotenuse_length_of_triangle : ℝ),
  side_length_of_square = 20 →
  hypotenuse_length_of_triangle = 10 →
  4 * (1/2 * (hypotenuse_length_of_triangle^2 / 2)) = 100 :=
by
  intros side_length_of_square hypotenuse_length_of_triangle h_side_length h_hypotenuse_length
  -- Proof would go here, but we add "sorry" to complete the statement
  sorry

end total_area_of_removed_triangles_l5_5781


namespace sum_first_n_terms_eq_l5_5051

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b_n (n : ℕ) : ℕ := 2 ^ (n - 1)

noncomputable def c_n (n : ℕ) : ℕ := a_n n * b_n n

noncomputable def T_n (n : ℕ) : ℕ := (2 * n - 3) * 2 ^ n + 3

theorem sum_first_n_terms_eq (n : ℕ) : 
  (Finset.sum (Finset.range n.succ) (λ k => c_n k) = T_n n) :=
  sorry

end sum_first_n_terms_eq_l5_5051


namespace rectangle_area_l5_5699

theorem rectangle_area (AB AC : ℝ) (H1 : AB = 15) (H2 : AC = 17) : 
  ∃ (BC : ℝ), (AB * BC = 120) :=
by
  sorry

end rectangle_area_l5_5699


namespace sum_opposite_numbers_correct_opposite_sum_numbers_correct_l5_5243

def opposite (x : Int) : Int := -x

def sum_opposite_numbers (a b : Int) : Int := opposite a + opposite b

def opposite_sum_numbers (a b : Int) : Int := opposite (a + b)

theorem sum_opposite_numbers_correct (a b : Int) : sum_opposite_numbers (-6) 4 = 2 := 
by sorry

theorem opposite_sum_numbers_correct (a b : Int) : opposite_sum_numbers (-6) 4 = 2 := 
by sorry

end sum_opposite_numbers_correct_opposite_sum_numbers_correct_l5_5243


namespace remaining_distance_proof_l5_5078

/-
In a bicycle course with a total length of 10.5 kilometers (km), if Yoongi goes 1.5 kilometers (km) and then goes another 3730 meters (m), prove that the remaining distance of the course is 5270 meters.
-/

def km_to_m (km : ℝ) : ℝ := km * 1000

def total_course_length_km : ℝ := 10.5
def total_course_length_m : ℝ := km_to_m total_course_length_km

def yoongi_initial_distance_km : ℝ := 1.5
def yoongi_initial_distance_m : ℝ := km_to_m yoongi_initial_distance_km

def yoongi_additional_distance_m : ℝ := 3730

def yoongi_total_distance_m : ℝ := yoongi_initial_distance_m + yoongi_additional_distance_m

def remaining_distance_m (total_course_length_m yoongi_total_distance_m : ℝ) : ℝ :=
  total_course_length_m - yoongi_total_distance_m

theorem remaining_distance_proof : remaining_distance_m total_course_length_m yoongi_total_distance_m = 5270 := 
  sorry

end remaining_distance_proof_l5_5078


namespace jack_piggy_bank_l5_5448

variable (initial_amount : ℕ) (weekly_allowance : ℕ) (weeks : ℕ)

-- Conditions
def initial_amount := 43
def weekly_allowance := 10
def weeks := 8

-- Weekly savings calculation: Jack saves half of his weekly allowance
def weekly_savings := weekly_allowance / 2

-- Total savings over the given period
def total_savings := weekly_savings * weeks

-- Final amount in the piggy bank after the given period
def final_amount := initial_amount + total_savings

-- Theorem to prove: Final amount in the piggy bank after 8 weeks is $83.00
theorem jack_piggy_bank : final_amount = 83 := by
  sorry

end jack_piggy_bank_l5_5448


namespace piggy_bank_after_8_weeks_l5_5445

-- Define initial amount in the piggy bank
def initial_amount : ℝ := 43

-- Define weekly allowance amount
def weekly_allowance : ℝ := 10

-- Define fraction of allowance Jack saves
def saving_fraction : ℝ := 0.5

-- Define number of weeks
def number_of_weeks : ℕ := 8

-- Define weekly savings amount
def weekly_savings : ℝ := saving_fraction * weekly_allowance

-- Define total savings after a given number of weeks
def total_savings (weeks : ℕ) : ℝ := weeks * weekly_savings

-- Define the final amount in the piggy bank after a given number of weeks
def final_amount (weeks : ℕ) : ℝ := initial_amount + total_savings weeks

-- Theorem: Prove that final amount in piggy bank after 8 weeks is $83
theorem piggy_bank_after_8_weeks : final_amount number_of_weeks = 83 := by
  sorry

end piggy_bank_after_8_weeks_l5_5445


namespace slope_of_parallel_line_l5_5927

theorem slope_of_parallel_line (x y : ℝ) :
  ∀ (m : ℝ), (3 * x - 6 * y = 12) → (m = 1 / 2) :=
begin
  sorry
end

end slope_of_parallel_line_l5_5927


namespace sixty_first_batch_is_1211_l5_5256

-- Definitions based on conditions
def total_bags : ℕ := 3000
def total_batches : ℕ := 150
def first_batch_number : ℕ := 11

-- Define the calculation of the 61st batch number
def batch_interval : ℕ := total_bags / total_batches
def sixty_first_batch_number : ℕ := first_batch_number + 60 * batch_interval

-- The statement of the proof
theorem sixty_first_batch_is_1211 : sixty_first_batch_number = 1211 := by
  sorry

end sixty_first_batch_is_1211_l5_5256


namespace right_triangle_third_side_product_l5_5328

noncomputable def hypot : ℝ → ℝ → ℝ := λ a b, real.sqrt (a * a + b * b)

noncomputable def other_leg : ℝ → ℝ → ℝ := λ h a, real.sqrt (h * h - a * a)

theorem right_triangle_third_side_product (a b : ℝ) (h : ℝ) (product : ℝ) 
  (h₁ : a = 6) (h₂ : b = 8) (h₃ : h = hypot 6 8) 
  (h₄ : b = h) (leg : ℝ := other_leg 8 6) :
  product = 52.9 :=
by
  have h5 : hypot 6 8 = 10 := sorry 
  have h6 : other_leg 8 6 = 2 * real.sqrt 7 := sorry
  have h7 : 10 * (2 * real.sqrt 7) = 20 * real.sqrt 7 := sorry
  have h8 : real.sqrt 7 ≈ 2.6458 := sorry
  have h9 : 20 * 2.6458 ≈ 52.916 := sorry
  have h10 : (52.916 : ℝ).round_to 1 = 52.9 := sorry
  exact h10


end right_triangle_third_side_product_l5_5328


namespace remainder_sum_first_seven_primes_div_eighth_prime_l5_5163

theorem remainder_sum_first_seven_primes_div_eighth_prime :
  let sum_of_first_seven_primes := 2 + 3 + 5 + 7 + 11 + 13 + 17 in
  let eighth_prime := 19 in
  sum_of_first_seven_primes % eighth_prime = 1 :=
by
  let sum_of_first_seven_primes := 2 + 3 + 5 + 7 + 11 + 13 + 17
  let eighth_prime := 19
  have : sum_of_first_seven_primes = 58 := by decide
  have : eighth_prime = 19 := rfl
  sorry

end remainder_sum_first_seven_primes_div_eighth_prime_l5_5163


namespace length_of_RT_in_trapezoid_l5_5635

-- Definition of the trapezoid and initial conditions
def trapezoid (PQ RS PR RT : ℝ) (h : PQ = 3 * RS) (h1 : PR = 15) : Prop :=
  RT = 15 / 4

-- The theorem to be proved
theorem length_of_RT_in_trapezoid (PQ RS PR RT : ℝ) 
  (h : PQ = 3 * RS) (h1 : PR = 15) : trapezoid PQ RS PR RT h h1 :=
by
  sorry

end length_of_RT_in_trapezoid_l5_5635


namespace jack_piggy_bank_l5_5447

variable (initial_amount : ℕ) (weekly_allowance : ℕ) (weeks : ℕ)

-- Conditions
def initial_amount := 43
def weekly_allowance := 10
def weeks := 8

-- Weekly savings calculation: Jack saves half of his weekly allowance
def weekly_savings := weekly_allowance / 2

-- Total savings over the given period
def total_savings := weekly_savings * weeks

-- Final amount in the piggy bank after the given period
def final_amount := initial_amount + total_savings

-- Theorem to prove: Final amount in the piggy bank after 8 weeks is $83.00
theorem jack_piggy_bank : final_amount = 83 := by
  sorry

end jack_piggy_bank_l5_5447


namespace extra_large_yellow_curlers_l5_5797

def total_curlers : ℕ := 120
def small_pink_curlers : ℕ := total_curlers / 5
def medium_blue_curlers : ℕ := 2 * small_pink_curlers
def large_green_curlers : ℕ := total_curlers / 4

theorem extra_large_yellow_curlers : 
  total_curlers - small_pink_curlers - medium_blue_curlers - large_green_curlers = 18 :=
by
  sorry

end extra_large_yellow_curlers_l5_5797


namespace sum_inf_series_l5_5398

theorem sum_inf_series :
  (\sum_{n=1}^{\infty} \frac{(4 * n) - 3}{3^n}) = 1 :=
by
  sorry

end sum_inf_series_l5_5398


namespace measure_of_angle_XPM_l5_5444

-- Definitions based on given conditions
variables (X Y Z L M N P : Type)
variables (a b c : ℝ) -- Angles are represented in degrees
variables [DecidableEq X] [DecidableEq Y] [DecidableEq Z]

-- Triangle XYZ with angle bisectors XL, YM, and ZN meeting at incenter P
-- Given angle XYZ in degrees
def angle_XYZ : ℝ := 46

-- Incenter angle properties
axiom angle_bisector_XL (angle_XYP : ℝ) : angle_XYP = angle_XYZ / 2
axiom angle_bisector_YM (angle_YXP : ℝ) : ∃ (angle_YXZ : ℝ), angle_YXP = angle_YXZ / 2

-- The proposition we need to prove
theorem measure_of_angle_XPM : ∃ (angle_XPM : ℝ), angle_XPM = 67 := 
by {
  sorry
}

end measure_of_angle_XPM_l5_5444


namespace slope_of_parallel_line_l5_5960

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l5_5960


namespace polynomial_factorization_l5_5969

-- Define the polynomial and its factorized form
def polynomial (x : ℝ) : ℝ := x^2 - 4*x + 4
def factorized_form (x : ℝ) : ℝ := (x - 2)^2

-- The theorem stating that the polynomial equals its factorized form
theorem polynomial_factorization (x : ℝ) : polynomial x = factorized_form x :=
by {
  sorry -- Proof skipped
}

end polynomial_factorization_l5_5969


namespace solve_for_x_l5_5069

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 1) (h1 : y = 1 / (4 * x^2 + 2 * x + 1)) : 
  x = 0 ∨ x = -1/2 :=
by
  sorry

end solve_for_x_l5_5069


namespace vegetables_sold_mass_l5_5775

/-- Define the masses of the vegetables --/
def mass_carrots : ℕ := 15
def mass_zucchini : ℕ := 13
def mass_broccoli : ℕ := 8

/-- Define the total mass of installed vegetables --/
def total_mass : ℕ := mass_carrots + mass_zucchini + mass_broccoli

/-- Define the mass of vegetables sold (half of the total mass) --/
def mass_sold : ℕ := total_mass / 2

/-- Prove that the mass of vegetables sold is 18 kg --/
theorem vegetables_sold_mass : mass_sold = 18 := by
  sorry

end vegetables_sold_mass_l5_5775


namespace positive_number_property_l5_5379

theorem positive_number_property (x : ℝ) (h : (100 - x) / 100 * x = 16) :
  x = 40 ∨ x = 60 :=
sorry

end positive_number_property_l5_5379


namespace constant_term_binomial_expansion_l5_5434

theorem constant_term_binomial_expansion (a : ℝ) (h : 15 * a^2 = 120) : a = 2 * Real.sqrt 2 :=
sorry

end constant_term_binomial_expansion_l5_5434


namespace marble_probability_l5_5009

theorem marble_probability :
  let bag := 16
  let reds := 12
  let blues := 4
  let total_selections := (16.choose 3) -- total ways to choose 3 out of 16
  let two_reds_one_blue :=
    (12 / 16) * (11 / 15) * (4 / 14) +
    (12 / 16) * (4 / 15) * (11 / 14) +
    (4 / 16) * (12 / 15) * (11 / 14)
  (two_reds_one_blue / total_selections) = (11 / 70) :=
by
  sorry

end marble_probability_l5_5009


namespace min_voters_to_win_l5_5084

def num_voters : ℕ := 105
def num_districts : ℕ := 5
def num_sections_per_district : ℕ := 7
def voters_per_section : ℕ := 3
def majority n : ℕ := n / 2 + 1

theorem min_voters_to_win (Tall_won : ∃ sections : fin num_voters → bool, 
  (∃ districts : fin num_districts → bool, 
    (countp (λ i, districts i = tt) (finset.univ : finset (fin num_districts)) ≥ majority num_districts) ∧ 
    ∀ i : fin num_districts, districts i = tt →
      (countp (λ j, sections (i * num_sections_per_district + j) = tt) (finset.range num_sections_per_district) ≥ majority num_sections_per_district)
  ) ∧
  (∀ i, i < num_voters →¬ (sections i = tt → sections ((i / num_sections_per_district) * num_sections_per_district + (i % num_sections_per_district)) = tt))
  ) : 3 * (12 * 2) ≥ 24 :=
by sorry

end min_voters_to_win_l5_5084


namespace functional_eq_solution_l5_5660

noncomputable def f : ℝ → ℝ := sorry

theorem functional_eq_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) →
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end functional_eq_solution_l5_5660


namespace sequence_odd_for_all_n_greater_than_1_l5_5633

theorem sequence_odd_for_all_n_greater_than_1 (a : ℕ → ℤ) :
  (a 1 = 2) →
  (a 2 = 7) →
  (∀ n, 2 ≤ n → (-1/2 : ℚ) < (a (n + 1) : ℚ) - ((a n : ℚ) ^ 2) / (a (n - 1) : ℚ) ∧ (a (n + 1) : ℚ) - ((a n : ℚ) ^ 2) / (a (n - 1) : ℚ) ≤ (1/2 : ℚ)) →
  ∀ n, 1 < n → Odd (a n) := 
sorry

end sequence_odd_for_all_n_greater_than_1_l5_5633


namespace find_a_not_perfect_square_l5_5663

theorem find_a_not_perfect_square :
  {a : ℕ | ∀ n : ℕ, n > 0 → ¬(∃ k : ℕ, n * (n + a) = k * k)} = {1, 2, 4} :=
sorry

end find_a_not_perfect_square_l5_5663


namespace f_bound_l5_5041

-- Define the function f(n) representing the number of representations of n as a sum of powers of 2
noncomputable def f (n : ℕ) : ℕ := 
-- f is defined as described in the problem, implementation skipped here
sorry

-- Propose to prove the main inequality for all n ≥ 3
theorem f_bound (n : ℕ) (h : n ≥ 3) : 2 ^ (n^2 / 4) < f (2 ^ n) ∧ f (2 ^ n) < 2 ^ (n^2 / 2) :=
sorry

end f_bound_l5_5041


namespace ratio_humans_to_beavers_l5_5696

-- Define the conditions
def humans : ℕ := 38 * 10^6
def moose : ℕ := 1 * 10^6
def beavers : ℕ := 2 * moose

-- Define the theorem to prove the ratio of humans to beavers
theorem ratio_humans_to_beavers : humans / beavers = 19 := by
  sorry

end ratio_humans_to_beavers_l5_5696


namespace min_value_of_a_plus_b_l5_5668

theorem min_value_of_a_plus_b (a b : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc : 1 = 1) 
    (h1 : b^2 > 4 * a) (h2 : b < 2 * a) (h3 : b < a + 1) : a + b = 10 :=
sorry

end min_value_of_a_plus_b_l5_5668


namespace bridge_length_is_correct_l5_5513

noncomputable def length_of_bridge (train_length : ℝ) (time : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let distance_covered := speed_mps * time
  distance_covered - train_length

theorem bridge_length_is_correct :
  length_of_bridge 100 16.665333439991468 54 = 149.97999909987152 :=
by sorry

end bridge_length_is_correct_l5_5513


namespace decagon_adjacent_vertices_probability_l5_5304

namespace ProbabilityAdjacentVertices

def total_vertices : ℕ := 10

def total_pairs : ℕ := total_vertices * (total_vertices - 1) / 2

def adjacent_pairs : ℕ := total_vertices

def probability_adjacent : ℚ := adjacent_pairs / total_pairs

theorem decagon_adjacent_vertices_probability :
  probability_adjacent = 2 / 9 := by
  sorry

end ProbabilityAdjacentVertices

end decagon_adjacent_vertices_probability_l5_5304


namespace passengers_on_bus_l5_5748

theorem passengers_on_bus (initial_passengers : ℕ) (got_on : ℕ) (got_off : ℕ) (final_passengers : ℕ) :
  initial_passengers = 28 → got_on = 7 → got_off = 9 → final_passengers = initial_passengers + got_on - got_off → final_passengers = 26 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end passengers_on_bus_l5_5748


namespace factors_are_divisors_l5_5974

theorem factors_are_divisors (a b c d : ℕ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) (h4 : d = 5) : 
  a ∣ 30 ∧ b ∣ 30 ∧ c ∣ 30 ∧ d ∣ 30 :=
by
  sorry

end factors_are_divisors_l5_5974


namespace at_most_n_maximum_distance_pairs_l5_5673

theorem at_most_n_maximum_distance_pairs (n : ℕ) (h : n > 2) 
(points : Fin n → ℝ × ℝ) :
  ∃ (maxDistPairs : Finset (Fin n × Fin n)), (maxDistPairs.card ≤ n) ∧ 
  ∀ (p1 p2 : Fin n), (p1, p2) ∈ maxDistPairs → 
  (∀ (q1 q2 : Fin n), dist (points q1) (points q2) ≤ dist (points p1) (points p2)) :=
sorry

end at_most_n_maximum_distance_pairs_l5_5673


namespace rational_sum_of_cubic_roots_inverse_l5_5568

theorem rational_sum_of_cubic_roots_inverse 
  (p q r : ℚ) 
  (h1 : p ≠ 0) 
  (h2 : q ≠ 0) 
  (h3 : r ≠ 0) 
  (h4 : ∃ a b c : ℚ, a = (pq^2)^(1/3) ∧ b = (qr^2)^(1/3) ∧ c = (rp^2)^(1/3) ∧ a + b + c ≠ 0) 
  : ∃ s : ℚ, s = 1/((pq^2)^(1/3)) + 1/((qr^2)^(1/3)) + 1/((rp^2)^(1/3)) :=
sorry

end rational_sum_of_cubic_roots_inverse_l5_5568


namespace compute_P_part_l5_5592

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem compute_P_part (a b c d : ℝ) 
  (H1 : P 1 a b c d = 1993) 
  (H2 : P 2 a b c d = 3986) 
  (H3 : P 3 a b c d = 5979) : 
  (1 / 4) * (P 11 a b c d + P (-7) a b c d) = 4693 :=
by
  sorry

end compute_P_part_l5_5592


namespace decagon_adjacent_probability_l5_5313

-- Define the setup of a decagon with 10 vertices and the adjacency relation.
def is_decagon (n : ℕ) : Prop := n = 10

def adjacent_vertices (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v1 = (v2 + 9) % 10)

-- The main theorem statement, proving the probability of two vertices being adjacent.
theorem decagon_adjacent_probability (n : ℕ) (v1 v2 : ℕ) (h : is_decagon n) : 
  ∀ (v1 v2 : ℕ), v1 ≠ v2 → v1 < n → v2 < n →
  (∃ (p : ℚ), p = 2 / 9 ∧ 
  (adjacent_vertices v1 v2) ↔ (p = 2 / (n - 1))) :=
sorry

end decagon_adjacent_probability_l5_5313


namespace parallel_line_slope_l5_5944

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℚ, m = 1 / 2 :=
by {
  use 1 / 2,
  sorry
}

end parallel_line_slope_l5_5944


namespace waiting_for_stocker_proof_l5_5721

-- Definitions for the conditions
def waiting_for_cart := 3
def waiting_for_employee := 13
def waiting_in_line := 18
def total_shopping_trip_time := 90
def time_shopping := 42

-- Calculate the total waiting time
def total_waiting_time := total_shopping_trip_time - time_shopping

-- Calculate the total known waiting time
def total_known_waiting_time := waiting_for_cart + waiting_for_employee + waiting_in_line

-- Calculate the waiting time for the stocker
def waiting_for_stocker := total_waiting_time - total_known_waiting_time

-- Prove that the waiting time for the stocker is 14 minutes
theorem waiting_for_stocker_proof : waiting_for_stocker = 14 := by
  -- Here the proof steps would normally be included
  sorry

end waiting_for_stocker_proof_l5_5721


namespace remainder_of_primes_sum_l5_5165

theorem remainder_of_primes_sum :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let p8 := 19 
  (p1 + p2 + p3 + p4 + p5 + p6 + p7) % p8 = 1 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let p8 := 19
  let sum := p1 + p2 + p3 + p4 + p5 + p6 + p7
  have h : sum = 58 := by norm_num
  show sum % p8 = 1
  rw [h]
  norm_num
  sorry

end remainder_of_primes_sum_l5_5165


namespace probability_of_two_red_balls_l5_5970

-- Definitions of quantities
def total_balls := 11
def red_balls := 3
def blue_balls := 4 
def green_balls := 4 
def balls_picked := 2

-- Theorem statement
theorem probability_of_two_red_balls :
  ((red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1) / balls_picked)) = 3 / 55 :=
by
  sorry

end probability_of_two_red_balls_l5_5970


namespace total_visitors_count_l5_5442

def initial_morning_visitors : ℕ := 500
def noon_departures : ℕ := 119
def additional_afternoon_arrivals : ℕ := 138

def afternoon_arrivals : ℕ := noon_departures + additional_afternoon_arrivals
def total_visitors : ℕ := initial_morning_visitors + afternoon_arrivals

theorem total_visitors_count : total_visitors = 757 := 
by sorry

end total_visitors_count_l5_5442


namespace evaluate_fraction_l5_5032

theorem evaluate_fraction :
  (0.5^2 + 0.05^3) / 0.005^3 = 2000100 := by
  sorry

end evaluate_fraction_l5_5032


namespace smallest_number_am_median_largest_l5_5300

noncomputable def smallest_number (a b c : ℕ) : ℕ :=
if a ≤ b ∧ a ≤ c then a
else if b ≤ a ∧ b ≤ c then b
else c

theorem smallest_number_am_median_largest (a b c : ℕ) (h1 : a + b + c = 90) (h2 : b = 28) (h3 : c = b + 6) :
  smallest_number a b c = 28 :=
sorry

end smallest_number_am_median_largest_l5_5300


namespace thirtieth_term_of_arithmetic_seq_l5_5362

def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem thirtieth_term_of_arithmetic_seq : 
  arithmetic_seq 3 4 30 = 119 := 
by
  sorry

end thirtieth_term_of_arithmetic_seq_l5_5362


namespace initial_fish_count_l5_5116

variable (x : ℕ)

theorem initial_fish_count (initial_fish : ℕ) (given_fish : ℕ) (total_fish : ℕ)
  (h1 : total_fish = initial_fish + given_fish)
  (h2 : total_fish = 69)
  (h3 : given_fish = 47) :
  initial_fish = 22 :=
by
  sorry

end initial_fish_count_l5_5116


namespace total_people_3522_l5_5193

def total_people (M W: ℕ) : ℕ := M + W

theorem total_people_3522 
    (M W: ℕ) 
    (h1: M / 9 * 45 + W / 12 * 60 = 17760)
    (h2: M % 9 = 0)
    (h3: W % 12 = 0) : 
    total_people M W = 3552 :=
by {
  sorry
}

end total_people_3522_l5_5193


namespace sum_series_eq_one_l5_5402

noncomputable def series : ℝ := ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1)

theorem sum_series_eq_one : series = 1 := 
by sorry

end sum_series_eq_one_l5_5402


namespace x_pow_n_plus_inv_x_pow_n_l5_5251

theorem x_pow_n_plus_inv_x_pow_n (θ : ℝ) (x : ℝ) (n : ℕ) (h1 : 0 < θ) (h2 : θ < Real.pi / 2) 
  (h3 : x + 1 / x = 2 * Real.sin θ) (hn_pos : 0 < n) : 
  x^n + (1 / x)^n = 2 * Real.cos (n * θ) := 
by
  sorry

end x_pow_n_plus_inv_x_pow_n_l5_5251


namespace piggy_bank_after_8_weeks_l5_5446

-- Define initial amount in the piggy bank
def initial_amount : ℝ := 43

-- Define weekly allowance amount
def weekly_allowance : ℝ := 10

-- Define fraction of allowance Jack saves
def saving_fraction : ℝ := 0.5

-- Define number of weeks
def number_of_weeks : ℕ := 8

-- Define weekly savings amount
def weekly_savings : ℝ := saving_fraction * weekly_allowance

-- Define total savings after a given number of weeks
def total_savings (weeks : ℕ) : ℝ := weeks * weekly_savings

-- Define the final amount in the piggy bank after a given number of weeks
def final_amount (weeks : ℕ) : ℝ := initial_amount + total_savings weeks

-- Theorem: Prove that final amount in piggy bank after 8 weeks is $83
theorem piggy_bank_after_8_weeks : final_amount number_of_weeks = 83 := by
  sorry

end piggy_bank_after_8_weeks_l5_5446


namespace inradius_of_triangle_l5_5740

theorem inradius_of_triangle (p A r : ℝ) (h1 : p = 20) (h2 : A = 25) : r = 2.5 :=
sorry

end inradius_of_triangle_l5_5740


namespace new_number_shifting_digits_l5_5074

-- Definitions for the three digits
variables (h t u : ℕ)

-- The original three-digit number
def original_number : ℕ := 100 * h + 10 * t + u

-- The new number formed by placing the digits "12" after the three-digit number
def new_number : ℕ := original_number h t u * 100 + 12

-- The goal is to prove that this new number equals 10000h + 1000t + 100u + 12
theorem new_number_shifting_digits (h t u : ℕ) :
  new_number h t u = 10000 * h + 1000 * t + 100 * u + 12 := 
by
  sorry -- Proof to be filled in

end new_number_shifting_digits_l5_5074


namespace arithmetic_sequence_term_13_l5_5441

variable {a : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_term_13 (h_arith : arithmetic_sequence a d)
  (h_a5 : a 5 = 3)
  (h_a9 : a 9 = 6) :
  a 13 = 9 := 
by 
  sorry

end arithmetic_sequence_term_13_l5_5441


namespace product_of_third_sides_is_correct_l5_5334

def sqrt_approx (x : ℝ) :=  -- Approximate square root function
  if x = 7 then 2.646 else 0

def product_of_third_sides (a b : ℝ) : ℝ :=
  let hypotenuse := real.sqrt (a * a + b * b)
  let leg := real.sqrt (b * b - a * a)
  hypotenuse * leg

theorem product_of_third_sides_is_correct :
  product_of_third_sides 6 8 = 52.9 :=
by
  unfold product_of_third_sides
  rw [if_pos (rfl : (real.sqrt (8 * 8 - 6 * 6)) = 2.646)]
  norm_num
  sorry

end product_of_third_sides_is_correct_l5_5334


namespace fifth_month_sale_correct_l5_5509

noncomputable def fifth_month_sale
  (sales : Fin 4 → ℕ)
  (sixth_month_sale : ℕ)
  (average_sale : ℕ) : ℕ :=
  let total_sales := average_sale * 6
  let known_sales := sales 0 + sales 1 + sales 2 + sales 3 + sixth_month_sale
  total_sales - known_sales

theorem fifth_month_sale_correct :
  ∀ (sales : Fin 4 → ℕ) (sixth_month_sale : ℕ) (average_sale : ℕ),
    sales 0 = 6435 →
    sales 1 = 6927 →
    sales 2 = 6855 →
    sales 3 = 7230 →
    sixth_month_sale = 5591 →
    average_sale = 6600 →
    fifth_month_sale sales sixth_month_sale average_sale = 13562 :=
by
  intros sales sixth_month_sale average_sale h0 h1 h2 h3 h4 h5
  unfold fifth_month_sale
  sorry

end fifth_month_sale_correct_l5_5509


namespace solve_for_x_l5_5072

theorem solve_for_x (x : ℝ) (h : 1 = 1 / (4 * x^2 + 2 * x + 1)) : 
  x = 0 ∨ x = -1 / 2 := 
by sorry

end solve_for_x_l5_5072


namespace value_range_of_function_l5_5294

theorem value_range_of_function :
  ∀ (x : ℝ), -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 → -1 ≤ Real.sin x * Real.sin x - 2 * Real.sin x ∧ Real.sin x * Real.sin x - 2 * Real.sin x ≤ 3 :=
by
  sorry

end value_range_of_function_l5_5294


namespace find_absolute_cd_l5_5128

noncomputable def polynomial_solution (c d : ℤ) (root1 root2 root3 : ℤ) : Prop :=
  c ≠ 0 ∧ d ≠ 0 ∧ 
  root1 = root2 ∧
  (root3 ≠ root1 ∨ root3 ≠ root2) ∧
  (root1^3 + root2^2 * root3 + (c * root1^2) + (d * root1) + 16 * c = 0) ∧ 
  (root2^3 + root1^2 * root3 + (c * root2^2) + (d * root2) + 16 * c = 0) ∧
  (root3^3 + root1^2 * root3 + (c * root3^2) + (d * root3) + 16 * c = 0)

theorem find_absolute_cd : ∃ c d root1 root2 root3 : ℤ,
  polynomial_solution c d root1 root2 root3 ∧ (|c * d| = 2560) :=
sorry

end find_absolute_cd_l5_5128


namespace Isabela_spent_l5_5836

theorem Isabela_spent (num_pencils : ℕ) (cost_per_item : ℕ) (num_cucumbers : ℕ)
  (h1 : cost_per_item = 20)
  (h2 : num_cucumbers = 100)
  (h3 : num_cucumbers = 2 * num_pencils)
  (discount : ℚ := 0.20) :
  let pencil_cost := num_pencils * cost_per_item
  let cucumber_cost := num_cucumbers * cost_per_item
  let discounted_pencil_cost := pencil_cost * (1 - discount)
  let total_cost := cucumber_cost + discounted_pencil_cost
  total_cost = 2800 := by
  -- Begin proof. We will add actual proof here later.
  sorry

end Isabela_spent_l5_5836


namespace luisa_mpg_l5_5717

theorem luisa_mpg
  (d_grocery d_mall d_pet d_home : ℕ)
  (cost_per_gal total_cost : ℚ)
  (total_miles : ℕ )
  (total_gallons : ℚ)
  (mpg : ℚ):
  d_grocery = 10 →
  d_mall = 6 →
  d_pet = 5 →
  d_home = 9 →
  cost_per_gal = 3.5 →
  total_cost = 7 →
  total_miles = d_grocery + d_mall + d_pet + d_home →
  total_gallons = total_cost / cost_per_gal →
  mpg = total_miles / total_gallons →
  mpg = 15 :=
by
  intros
  sorry

end luisa_mpg_l5_5717


namespace investment_difference_l5_5897

theorem investment_difference (x y z : ℕ) 
  (h1 : x + (x + y) + (x + 2 * y) = 9000)
  (h2 : (z / 9000) = (800 / 1800)) 
  (h3 : z = x + 2 * y) :
  y = 1000 := 
by
  -- omitted proof steps
  sorry

end investment_difference_l5_5897


namespace minimum_voters_for_tall_victory_l5_5098

-- Definitions for conditions
def total_voters : ℕ := 105
def districts : ℕ := 5
def sections_per_district : ℕ := 7
def voters_per_section : ℕ := 3

-- Define majority function
def majority (n : ℕ) : ℕ := n / 2 + 1

-- Express conditions in Lean
def voters_per_district : ℕ := total_voters / districts
def sections_to_win_district : ℕ := majority sections_per_district
def districts_to_win_contest : ℕ := majority districts

-- The main problem statement
theorem minimum_voters_for_tall_victory : ∃ (x : ℕ), x = 24 ∧
  (let sections_needed := sections_to_win_district * districts_to_win_contest in
   let voters_needed_per_section := majority voters_per_section in
   x = sections_needed * voters_needed_per_section) :=
by {
  let sections_needed := sections_to_win_district * districts_to_win_contest,
  let voters_needed_per_section := majority voters_per_section,
  use 24,
  split,
  { refl },
  { simp [sections_needed, voters_needed_per_section, sections_to_win_district, districts_to_win_contest, majority, voters_per_section] }
}

end minimum_voters_for_tall_victory_l5_5098


namespace decagon_adjacent_vertex_probability_l5_5321

theorem decagon_adjacent_vertex_probability :
  let vertices := 10 in
  let total_combinations := Nat.choose vertices 2 in
  let adjacent_pairs := vertices * 2 in
  (adjacent_pairs : ℚ) / total_combinations = 4 / 9 :=
by
  let vertices := 10
  let total_combinations := Nat.choose vertices 2
  let adjacent_pairs := vertices * 2
  have : (adjacent_pairs : ℚ) / total_combinations = 4 / 9 := sorry
  exact this

end decagon_adjacent_vertex_probability_l5_5321


namespace mary_baking_cups_l5_5458

-- Conditions
def flour_needed : ℕ := 9
def sugar_needed : ℕ := 11
def flour_added : ℕ := 4
def sugar_added : ℕ := 0

-- Statement to prove
theorem mary_baking_cups : sugar_needed - (flour_needed - flour_added) = 6 := by
  sorry

end mary_baking_cups_l5_5458


namespace triangle_area_six_parts_l5_5484

theorem triangle_area_six_parts (S S₁ S₂ S₃ : ℝ) (h₁ : S₁ ≥ 0) (h₂ : S₂ ≥ 0) (h₃ : S₃ ≥ 0) :
  S = (Real.sqrt S₁ + Real.sqrt S₂ + Real.sqrt S₃) ^ 2 := 
sorry

end triangle_area_six_parts_l5_5484


namespace mike_went_to_last_year_l5_5459

def this_year_games : ℕ := 15
def games_missed_this_year : ℕ := 41
def total_games_attended : ℕ := 54
def last_year_games : ℕ := total_games_attended - this_year_games

theorem mike_went_to_last_year :
  last_year_games = 39 :=
  by sorry

end mike_went_to_last_year_l5_5459


namespace multiplication_correct_l5_5390

theorem multiplication_correct (a b c d e f: ℤ) (h₁: a * b = c) (h₂: d * e = f): 
    (63 * 14 = c) → (68 * 14 = f) → c = 882 ∧ f = 952 :=
by sorry

end multiplication_correct_l5_5390


namespace symmetric_points_sum_l5_5423

theorem symmetric_points_sum (a b : ℝ) (hA1 : A = (a, 1)) (hB1 : B = (5, b))
    (h_symmetric : (a, 1) = -(5, b)) : a + b = -6 :=
by
  sorry

end symmetric_points_sum_l5_5423


namespace cookie_jar_initial_amount_l5_5895

variable (initial_amount : ℕ)
variable (doris_spent : ℕ := 6)
variable (martha_spent : ℕ := doris_spent / 2)
variable (remaining : ℕ := 15)

theorem cookie_jar_initial_amount :
  initial_amount = doris_spent + martha_spent + remaining :=
begin
  sorry
end

end cookie_jar_initial_amount_l5_5895


namespace problem_b_lt_a_lt_c_l5_5848

theorem problem_b_lt_a_lt_c (a b c : ℝ)
  (h1 : 1.001 * Real.exp a = Real.exp 1.001)
  (h2 : b - Real.sqrt (1000 / 1001) = 1.001 - Real.sqrt 1.001)
  (h3 : c = 1.001) : b < a ∧ a < c := by
  sorry

end problem_b_lt_a_lt_c_l5_5848


namespace slope_of_parallel_line_l5_5961

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l5_5961


namespace div_c_a_l5_5248

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a / b = 3)
variable (h2 : b / c = 2 / 5)

-- State the theorem to be proven
theorem div_c_a (a b c : ℝ) (h1 : a / b = 3) (h2 : b / c = 2 / 5) : c / a = 5 / 6 := 
by 
  sorry

end div_c_a_l5_5248


namespace arithmetic_sequence_ninth_term_l5_5885

theorem arithmetic_sequence_ninth_term 
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29)
  : a + 8 * d = 35 :=
by
  sorry

end arithmetic_sequence_ninth_term_l5_5885


namespace sum_series_eq_one_l5_5401

noncomputable def series : ℝ := ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1)

theorem sum_series_eq_one : series = 1 := 
by sorry

end sum_series_eq_one_l5_5401


namespace graphs_intersection_points_l5_5725

theorem graphs_intersection_points {g : ℝ → ℝ} (h_injective : Function.Injective g) :
  ∃ (x1 x2 x3 : ℝ), (g (x1^3) = g (x1^5)) ∧ (g (x2^3) = g (x2^5)) ∧ (g (x3^3) = g (x3^5)) ∧ 
  x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ ∀ (x : ℝ), (g (x^3) = g (x^5)) → (x = x1 ∨ x = x2 ∨ x = x3) := 
by
  sorry

end graphs_intersection_points_l5_5725


namespace tall_wins_min_voters_l5_5092

structure VotingSetup where
  total_voters : ℕ
  districts : ℕ
  sections_per_district : ℕ
  voters_per_section : ℕ
  voters_majority_in_section : ℕ
  districts_to_win : ℕ
  sections_to_win_district : ℕ

def contest_victory (setup : VotingSetup) (min_voters : ℕ) : Prop :=
  setup.total_voters = 105 ∧
  setup.districts = 5 ∧
  setup.sections_per_district = 7 ∧
  setup.voters_per_section = 3 ∧
  setup.voters_majority_in_section = 2 ∧
  setup.districts_to_win = 3 ∧
  setup.sections_to_win_district = 4 ∧
  min_voters = 24

theorem tall_wins_min_voters : ∃ min_voters, contest_victory ⟨105, 5, 7, 3, 2, 3, 4⟩ min_voters :=
by { use 24, sorry }

end tall_wins_min_voters_l5_5092


namespace find_a10_l5_5050

def seq (a : ℕ → ℝ) : Prop :=
∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p + a q

theorem find_a10 (a : ℕ → ℝ) (h_seq : seq a) (h_a2 : a 2 = -6) : a 10 = -30 :=
by
  sorry

end find_a10_l5_5050


namespace Alice_fills_needed_l5_5517

def cups_needed : ℚ := 15/4
def cup_capacity : ℚ := 1/3
def fills_needed : ℚ := 12

theorem Alice_fills_needed : (cups_needed / cup_capacity).ceil = fills_needed := by
  -- Proof is omitted with sorry
  sorry

end Alice_fills_needed_l5_5517


namespace div_c_a_l5_5249

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a / b = 3)
variable (h2 : b / c = 2 / 5)

-- State the theorem to be proven
theorem div_c_a (a b c : ℝ) (h1 : a / b = 3) (h2 : b / c = 2 / 5) : c / a = 5 / 6 := 
by 
  sorry

end div_c_a_l5_5249


namespace range_of_m_l5_5685

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, e ≤ x ∧ x ≤ e^2 ∧ f x - m * x - 1/2 + m ≤ 0) →
  1/2 ≤ m := by
  sorry

end range_of_m_l5_5685


namespace age_sum_l5_5719

theorem age_sum (my_age : ℕ) (mother_age : ℕ) (h1 : mother_age = 3 * my_age) (h2 : my_age = 10) :
  my_age + mother_age = 40 :=
by 
  -- proof omitted
  sorry

end age_sum_l5_5719


namespace fido_yard_area_reach_l5_5539

theorem fido_yard_area_reach (s r : ℝ) (h1 : r = s / (2 * Real.sqrt 2)) (h2 : ∃ (a b : ℕ), (Real.pi * Real.sqrt a) / b = Real.pi * (r ^ 2) / (2 * s^2 * Real.sqrt 2) ) :
  ∃ (a b : ℕ), a * b = 64 :=
by
  sorry

end fido_yard_area_reach_l5_5539


namespace odd_function_period_2pi_l5_5686

noncomputable def f (x : ℝ) : ℝ := Real.tan (x / 2)

theorem odd_function_period_2pi (x : ℝ) : 
  f (-x) = -f (x) ∧ 
  ∃ p > 0, p = 2 * Real.pi ∧ ∀ x, f (x + p) = f (x) := 
by
  sorry

end odd_function_period_2pi_l5_5686


namespace total_complaints_l5_5868

-- Conditions as Lean definitions
def normal_complaints : ℕ := 120
def short_staffed_20 (c : ℕ) := c + c / 3
def short_staffed_40 (c : ℕ) := c + 2 * c / 3
def self_checkout_partial (c : ℕ) := c + c / 10
def self_checkout_complete (c : ℕ) := c + c / 5
def day1_complaints : ℕ := normal_complaints + normal_complaints / 3 + normal_complaints / 5
def day2_complaints : ℕ := normal_complaints + 2 * normal_complaints / 3 + normal_complaints / 10
def day3_complaints : ℕ := normal_complaints + 2 * normal_complaints / 3 + normal_complaints / 5

-- Prove the total complaints
theorem total_complaints : day1_complaints + day2_complaints + day3_complaints = 620 :=
by
  sorry

end total_complaints_l5_5868


namespace units_digit_42_pow_4_add_24_pow_4_l5_5758

-- Define a function to get the units digit of a number.
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_42_pow_4_add_24_pow_4 : units_digit (42^4 + 24^4) = 2 := by
  sorry

end units_digit_42_pow_4_add_24_pow_4_l5_5758


namespace sum_fractions_series_l5_5209

-- Define a function representing the sum of fractions from 1/7 to 12/7
def sum_fractions : ℚ :=
  (list.sum (list.map (λ k, k / 7) (list.range' 1 12)))

-- State the theorem
theorem sum_fractions_series :
  sum_fractions = 11 + 1 / 7 :=
sorry

end sum_fractions_series_l5_5209


namespace julie_read_yesterday_l5_5269

variable (x : ℕ)
variable (y : ℕ := 2 * x)
variable (remaining_pages_after_two_days : ℕ := 120 - (x + y))

theorem julie_read_yesterday :
  (remaining_pages_after_two_days / 2 = 42) -> (x = 12) :=
by
  sorry

end julie_read_yesterday_l5_5269


namespace arithmetic_sequence_ninth_term_l5_5889

theorem arithmetic_sequence_ninth_term :
  ∃ a d : ℤ, (a + 2 * d = 23) ∧ (a + 5 * d = 29) ∧ (a + 8 * d = 35) :=
by
  sorry

end arithmetic_sequence_ninth_term_l5_5889


namespace brenda_has_eight_l5_5503

-- Define the amounts each friend has
def emma_money : ℕ := 8
def daya_money : ℕ := emma_money + (emma_money / 4)
def jeff_money : ℕ := (2 * daya_money) / 5
def brenda_money : ℕ := jeff_money + 4

-- Define the theorem to prove Brenda's money is 8
theorem brenda_has_eight : brenda_money = 8 := by
  sorry

end brenda_has_eight_l5_5503


namespace slope_of_parallel_line_l5_5937

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l5_5937


namespace car_travel_inequality_l5_5976

variable (x : ℕ)

theorem car_travel_inequality (hx : 8 * (x + 19) > 2200) : 8 * (x + 19) > 2200 :=
by
  sorry

end car_travel_inequality_l5_5976


namespace problem1_problem2_l5_5443

-- Definitions of the conditions and theorems
noncomputable def triangle_ABC_max_area (B : ℝ) (b : ℝ) (A C : ℝ) : ℝ :=
  let α := (A - C) / 2 in
  (b * b * sin (C)) / 2

theorem problem1 (B : ℝ) (A C : ℝ) (b : ℝ) (h1 : A + C = 120) (h2 : B = 60) 
                 (h3 : b = 2) : max (triangle_ABC_max_area B b A C) = 1 := sorry

noncomputable def cos_half_angle_diff (A C : ℝ) : ℝ :=
  cos ((A - C) / 2)

theorem problem2 (A C B : ℝ) (h1 : 1 / cos A + 1 / cos C = -sqrt(2) / cos B )
                 (h2 : B = 60) : cos_half_angle_diff A C = sqrt(2) / 2 := sorry

end problem1_problem2_l5_5443


namespace work_rate_combined_l5_5631

theorem work_rate_combined (a b c : ℝ) (ha : a = 21) (hb : b = 6) (hc : c = 12) :
  (1 / ((1 / a) + (1 / b) + (1 / c))) = 84 / 25 := by
  sorry

end work_rate_combined_l5_5631


namespace mean_age_of_euler_family_children_l5_5129

noncomputable def euler_family_children_ages : List ℕ := [9, 9, 9, 9, 18, 21, 21]

theorem mean_age_of_euler_family_children : 
  (List.sum euler_family_children_ages : ℚ) / (List.length euler_family_children_ages) = 96 / 7 := 
by
  sorry

end mean_age_of_euler_family_children_l5_5129


namespace probability_adjacent_vertices_decagon_l5_5306

theorem probability_adjacent_vertices_decagon : 
  ∀ (decagon : Finset ℕ) (a b : ℕ), 
  decagon.card = 10 → 
  a ∈ decagon → 
  b ∈ decagon → 
  a ≠ b → 
  (P (adjacent a b decagon)) = 2 / 9 :=
by 
  -- we are asserting the probability that two randomly chosen vertices a and b from a decagon are adjacent.
  sorry

end probability_adjacent_vertices_decagon_l5_5306


namespace union_sets_l5_5252

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_sets : M ∪ N = {-1, 0, 1, 2} :=
by
  sorry

end union_sets_l5_5252


namespace slope_of_parallel_line_l5_5922

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l5_5922


namespace metro_problem_l5_5702

theorem metro_problem (G : SimpleGraph (Fin 2019)) (hG : G.Connected) : 
  ¬ ∃ (P : Finset (Set (Fin 2019))), P.card ≤ 1008 ∧ ∀ v, ∃ p ∈ P, v ∈ p :=
sorry

end metro_problem_l5_5702


namespace problem1_problem2_problem3_l5_5425

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then x^2 - 2 * x else (abs x)^2 - 2 * abs x

-- Define the condition that f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Problem 1: Prove the minimum value of f(x) is -1.
theorem problem1 (h_even : even_function f) : ∃ x : ℝ, f x = -1 :=
by
  sorry

-- Problem 2: Prove the solution set of f(x) > 0 is (-∞, -2) ∪ (2, +∞).
theorem problem2 (h_even : even_function f) : 
  { x : ℝ | f x > 0 } = { x : ℝ | x < -2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

-- Problem 3: Prove there exists a real number x such that f(x+2) + f(-x) = 0.
theorem problem3 (h_even : even_function f) : ∃ x : ℝ, f (x + 2) + f (-x) = 0 :=
by
  sorry

end problem1_problem2_problem3_l5_5425


namespace focus_coordinates_correct_l5_5206
noncomputable def ellipse_focus : Real × Real :=
  let center : Real × Real := (4, -1)
  let a : Real := 4
  let b : Real := 1.5
  let c : Real := Real.sqrt (a^2 - b^2)
  (center.1 + c, center.2)

theorem focus_coordinates_correct : 
  ellipse_focus = (7.708, -1) := 
by 
  sorry

end focus_coordinates_correct_l5_5206


namespace slope_of_parallel_line_l5_5929

theorem slope_of_parallel_line (x y : ℝ) :
  ∀ (m : ℝ), (3 * x - 6 * y = 12) → (m = 1 / 2) :=
begin
  sorry
end

end slope_of_parallel_line_l5_5929


namespace smallest_period_f_symmetry_f_l5_5815

noncomputable def f (x : ℝ) : ℝ :=
  sin x * cos x - sqrt 3 * cos x ^ 2 + sqrt 3 / 2

theorem smallest_period_f :
  ∀ x : ℝ, f (x + π) = f x :=
sorry

theorem symmetry_f :
  ∃ k : ℤ, ∀ x : ℝ,
  (f (x) = f (π / 6 + k * (π / 2)) ∧ f (x) = 0 →
  ((∃ k : ℤ, x = 5 * π / 12 + k * (π / 2)) ∧ (x, 0) = (π / 6 + k * (π / 2), 0))) :=
sorry

end smallest_period_f_symmetry_f_l5_5815


namespace parallel_line_slope_l5_5920

def slope_of_parallel_line : ℚ :=
  let line_eq := (3 : ℚ) * (x : ℚ) - (6 : ℚ) * (y : ℚ) = (12 : ℚ)
  in (1 / 2 : ℚ)

theorem parallel_line_slope (x y : ℚ) (h : 3 * x - 6 * y = 12) : slope_of_parallel_line = 1 / 2 :=
by
  sorry

end parallel_line_slope_l5_5920


namespace meat_division_l5_5779

theorem meat_division (w1 w2 meat : ℕ) (h1 : w1 = 645) (h2 : w2 = 237) (h3 : meat = 1000) :
  ∃ (m1 m2 : ℕ), m1 = 296 ∧ m2 = 704 ∧ w1 + m1 = w2 + m2 := by
  sorry

end meat_division_l5_5779


namespace p_necessary_not_sufficient_for_q_l5_5210

open Classical

variable (p q : Prop)

theorem p_necessary_not_sufficient_for_q (h1 : ¬(p → q)) (h2 : ¬q → ¬p) : (¬(p → q) ∧ (¬q → ¬p) ∧ (¬p → ¬q ∧ ¬(¬q → p))) := by
  sorry

end p_necessary_not_sufficient_for_q_l5_5210


namespace min_voters_for_Tall_victory_l5_5089

def total_voters := 105
def districts := 5
def sections_per_district := 7
def voters_per_section := 3
def sections_to_win_district := 4
def districts_to_win := 3
def sections_to_win := sections_to_win_district * districts_to_win
def min_voters_to_win_section := 2

theorem min_voters_for_Tall_victory : 
  (total_voters = 105) ∧ 
  (districts = 5) ∧ 
  (sections_per_district = 7) ∧ 
  (voters_per_section = 3) ∧ 
  (sections_to_win_district = 4) ∧ 
  (districts_to_win = 3) 
  → 
  min_voters_to_win_section * sections_to_win = 24 :=
by
  sorry
  
end min_voters_for_Tall_victory_l5_5089


namespace a_square_minus_b_square_l5_5250

theorem a_square_minus_b_square (a b : ℚ)
  (h1 : a + b = 11 / 17)
  (h2 : a - b = 1 / 143) : a^2 - b^2 = 11 / 2431 :=
by
  sorry

end a_square_minus_b_square_l5_5250


namespace max_ABC_ge_4_9_max_alpha_beta_gamma_ge_4_9_l5_5674

variable (p q : ℝ) (x y : ℝ)
variable (A B C α β γ : ℝ)

-- Conditions
axiom hp : 0 ≤ p ∧ p ≤ 1 
axiom hq : 0 ≤ q ∧ q ≤ 1 
axiom h1 : (p * x + (1 - p) * y)^2 = A * x^2 + B * x * y + C * y^2
axiom h2 : (p * x + (1 - p) * y) * (q * x + (1 - q) * y) = α * x^2 + β * x * y + γ * y^2

-- Problem
theorem max_ABC_ge_4_9 : max A (max B C) ≥ 4 / 9 := 
sorry

theorem max_alpha_beta_gamma_ge_4_9 : max α (max β γ) ≥ 4 / 9 := 
sorry

end max_ABC_ge_4_9_max_alpha_beta_gamma_ge_4_9_l5_5674


namespace solve_quadratic_equation_l5_5426

theorem solve_quadratic_equation (x : ℝ) :
  2 * x * (x + 1) = x + 1 ↔ (x = -1 ∨ x = 1 / 2) :=
by
  sorry

end solve_quadratic_equation_l5_5426


namespace converse_x_gt_y_then_x_gt_abs_y_is_true_l5_5204

theorem converse_x_gt_y_then_x_gt_abs_y_is_true :
  (∀ x y : ℝ, (x > y) → (x > |y|)) → (∀ x y : ℝ, (x > |y|) → (x > y)) :=
by
  sorry

end converse_x_gt_y_then_x_gt_abs_y_is_true_l5_5204


namespace original_triangle_angles_determined_l5_5534

-- Define the angles of the formed triangle
def formed_triangle_angles : Prop := 
  52 + 61 + 67 = 180

-- Define the angles of the original triangle
def original_triangle_angles (α β γ : ℝ) : Prop := 
  α + β + γ = 180

theorem original_triangle_angles_determined :
  formed_triangle_angles → 
  ∃ α β γ : ℝ, 
    original_triangle_angles α β γ ∧
    α = 76 ∧ β = 58 ∧ γ = 46 :=
by
  sorry

end original_triangle_angles_determined_l5_5534


namespace power_addition_proof_l5_5790

theorem power_addition_proof :
  (-2) ^ 48 + 3 ^ (4 ^ 3 + 5 ^ 2 - 7 ^ 2) = 2 ^ 48 + 3 ^ 40 := 
by
  sorry

end power_addition_proof_l5_5790


namespace least_possible_area_l5_5017

def perimeter (x y : ℕ) : ℕ := 2 * (x + y)

def area (x y : ℕ) : ℕ := x * y

theorem least_possible_area :
  ∃ (x y : ℕ), 
    perimeter x y = 120 ∧ 
    (∀ x y, perimeter x y = 120 → area x y ≥ 59) ∧ 
    area x y = 59 := 
sorry

end least_possible_area_l5_5017


namespace parabola_directrix_eq_l5_5701

noncomputable def equation_of_directrix (p : ℝ) : Prop :=
  (p > 0) ∧ (∀ (x y : ℝ), (x ≠ -5 / 4) → ¬ (y ^ 2 = 2 * p * x))

theorem parabola_directrix_eq (A_x A_y : ℝ) (hA : A_x = 2 ∧ A_y = 1)
  (h_perpendicular_bisector_fo : ∃ (f_x f_y : ℝ), f_x = 5 / 4 ∧ f_y = 0) :
  equation_of_directrix (5 / 2) :=
by {
  sorry
}

end parabola_directrix_eq_l5_5701


namespace chord_bisected_vertically_by_line_l5_5816

theorem chord_bisected_vertically_by_line (p : ℝ) (h : p > 0) (l : ℝ → ℝ) (focus : ℝ × ℝ) 
  (h_focus: focus = (p / 2, 0)) (h_line: ∀ x, l x ≠ 0) :
  ¬ ∃ (A B : ℝ × ℝ), 
     A.1 ≠ B.1 ∧
     A.2^2 = 2 * p * A.1 ∧ B.2^2 = 2 * p * B.1 ∧ 
     (A.1 + B.1) / 2 = focus.1 ∧ 
     l ((A.1 + B.1) / 2) = focus.2 :=
sorry

end chord_bisected_vertically_by_line_l5_5816


namespace num_fish_when_discovered_l5_5146

open Nat

/-- Definition of the conditions given in the problem --/
def initial_fish := 60
def fish_per_day_eaten := 2
def additional_fish := 8
def weeks_before_addition := 2
def extra_week := 1

/-- The proof problem statement --/
theorem num_fish_when_discovered : 
  let days := (weeks_before_addition + extra_week) * 7
  let total_fish_eaten := days * fish_per_day_eaten
  let fish_after_addition := initial_fish + additional_fish
  let final_fish := fish_after_addition - total_fish_eaten
  final_fish = 26 := 
by
  let days := (weeks_before_addition + extra_week) * 7
  let total_fish_eaten := days * fish_per_day_eaten
  let fish_after_addition := initial_fish + additional_fish
  let final_fish := fish_after_addition - total_fish_eaten
  have h : final_fish = 26 := sorry
  exact h

end num_fish_when_discovered_l5_5146


namespace rahul_matches_l5_5632

variable (m : ℕ)

/-- Rahul's current batting average is 51, and if he scores 78 runs in today's match,
    his new batting average will become 54. Prove that the number of matches he had played
    in this season before today's match is 8. -/
theorem rahul_matches (h1 : (51 * m) / m = 51)
                      (h2 : (51 * m + 78) / (m + 1) = 54) : m = 8 := by
  sorry

end rahul_matches_l5_5632


namespace earnings_difference_l5_5718

def total_earnings : ℕ := 3875
def first_job_earnings : ℕ := 2125
def second_job_earnings := total_earnings - first_job_earnings

theorem earnings_difference : (first_job_earnings - second_job_earnings) = 375 := by
  sorry

end earnings_difference_l5_5718


namespace factorization_correct_l5_5966

theorem factorization_correct :
  ∀ (x y : ℝ), 
    (¬ ( (y - 1) * (y + 1) = y^2 - 1 ) ) ∧
    (¬ ( x^2 * y + x * y^2 - 1 = x * y * (x + y) - 1 ) ) ∧
    (¬ ( (x - 2) * (x - 3) = (3 - x) * (2 - x) ) ) ∧
    ( x^2 - 4 * x + 4 = (x - 2)^2 ) :=
by
  intros x y
  repeat { constructor }
  all_goals { sorry }

end factorization_correct_l5_5966


namespace pure_imaginary_z1_over_z2_l5_5229

theorem pure_imaginary_z1_over_z2 (b : Real) : 
  let z1 := (3 : Complex) - (b : Real) * Complex.I
  let z2 := (1 : Complex) - 2 * Complex.I
  (Complex.re ((z1 / z2) : Complex)) = 0 → b = -3 / 2 :=
by
  intros
  -- Conditions
  let z1 := (3 : Complex) - (b : Real) * Complex.I
  let z2 := (1 : Complex) - 2 * Complex.I
  -- Assuming that the real part of (z1 / z2) is zero
  have h : Complex.re (z1 / z2) = 0 := ‹_›
  -- Require to prove that b = -3 / 2
  sorry

end pure_imaginary_z1_over_z2_l5_5229


namespace thirtieth_term_of_arithmetic_seq_l5_5361

def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem thirtieth_term_of_arithmetic_seq : 
  arithmetic_seq 3 4 30 = 119 := 
by
  sorry

end thirtieth_term_of_arithmetic_seq_l5_5361


namespace derivative_of_f_l5_5427

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_of_f (x : ℝ) (h : 0 < x) :
    deriv f x = (1 - Real.log x) / (x ^ 2) := 
sorry

end derivative_of_f_l5_5427


namespace binary_1101_to_decimal_l5_5404

theorem binary_1101_to_decimal : (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 13 := by
  -- To convert a binary number to its decimal equivalent, we multiply each digit by its corresponding power of 2 based on its position and then sum the results.
  sorry

end binary_1101_to_decimal_l5_5404


namespace face_value_of_stock_l5_5013

-- Define variables and constants
def quoted_price : ℝ := 200
def yield_quoted : ℝ := 0.10
def percentage_yield : ℝ := 0.20

-- Define the annual income from the quoted price and percentage yield
def annual_income_from_quoted_price : ℝ := yield_quoted * quoted_price
def annual_income_from_face_value (FV : ℝ) : ℝ := percentage_yield * FV

-- Problem statement to prove
theorem face_value_of_stock (FV : ℝ) :
  annual_income_from_face_value FV = annual_income_from_quoted_price →
  FV = 100 := 
by
  sorry

end face_value_of_stock_l5_5013


namespace smallest_value_in_interval_l5_5570

open Real

noncomputable def smallest_value (x : ℝ) (h : 1 < x ∧ x < 2) : Prop :=
  1 / x^2 < x ∧
  1 / x^2 < x^2 ∧
  1 / x^2 < 2 * x^2 ∧
  1 / x^2 < 3 * x ∧
  1 / x^2 < sqrt x ∧
  1 / x^2 < 1 / x

theorem smallest_value_in_interval (x : ℝ) (h : 1 < x ∧ x < 2) : smallest_value x h :=
by
  sorry

end smallest_value_in_interval_l5_5570


namespace least_number_div_condition_l5_5378

theorem least_number_div_condition (m : ℕ) : 
  (∃ k r : ℕ, m = 34 * k + r ∧ m = 5 * (r + 8) ∧ r < 34) → m = 162 := 
by
  sorry

end least_number_div_condition_l5_5378


namespace collete_and_rachel_age_difference_l5_5119

theorem collete_and_rachel_age_difference :
  ∀ (Rona Rachel Collete : ℕ), 
  Rachel = 2 * Rona ∧ Collete = Rona / 2 ∧ Rona = 8 -> 
  Rachel - Collete = 12 := by
  intros Rona Rachel Collete h
  cases h with hRAR hRC
  cases hRC with hCol hRon
  sorry

end collete_and_rachel_age_difference_l5_5119


namespace repeating_decimal_to_fraction_l5_5657

theorem repeating_decimal_to_fraction :
  let x := (0.6 : ℚ) + (0.03 / (1 - 0.01)) in
  x = 104 / 165 :=
by
  let x : ℚ := (0.6 : ℚ) + (3 / 99)
  have h₁ : 0.6 = 3 / 5 := by sorry
  have h₂ : 0.03 / (1 - 0.01) = 1 / 33 := by sorry
  rw [h₁, h₂]
  exact (3 / 5 + 1 / 33 = 104 / 165) sorry

end repeating_decimal_to_fraction_l5_5657


namespace probability_neither_cake_nor_muffin_l5_5184

noncomputable def probability_of_neither (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) : ℚ :=
  (total - (cake + muffin - both)) / total

theorem probability_neither_cake_nor_muffin
  (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) (h_total : total = 100)
  (h_cake : cake = 50) (h_muffin : muffin = 40) (h_both : both = 18) :
  probability_of_neither total cake muffin both = 0.28 :=
by
  rw [h_total, h_cake, h_muffin, h_both]
  norm_num
  sorry

end probability_neither_cake_nor_muffin_l5_5184


namespace carbonated_water_solution_l5_5990

variable (V V_1 V_2 : ℝ)
variable (C2 : ℝ)

def carbonated_water_percent (V V1 V2 C2 : ℝ) : Prop :=
  0.8 * V1 + C2 * V2 = 0.6 * V

theorem carbonated_water_solution :
  ∀ (V : ℝ),
  (V1 = 0.1999999999999997 * V) →
  (V2 = 0.8000000000000003 * V) →
  carbonated_water_percent V V1 V2 C2 →
  C2 = 0.55 :=
by
  intros V V1_eq V2_eq carbonated_eq
  sorry

end carbonated_water_solution_l5_5990


namespace smallest_N_l5_5644

theorem smallest_N (l m n : ℕ) (N : ℕ) (h1 : N = l * m * n) (h2 : (l - 1) * (m - 1) * (n - 1) = 300) : 
  N = 462 :=
sorry

end smallest_N_l5_5644


namespace right_triangle_third_side_product_l5_5331

theorem right_triangle_third_side_product :
  ∃ (c hypotenuse : ℝ), 
    (c = real.sqrt (6^2 + 8^2) ∨ hypotenuse = real.sqrt (8^2 - 6^2)) ∧ 
    real.sqrt (6^2 + 8^2) * real.sqrt (8^2 - 6^2) = 52.7 :=
by
  sorry

end right_triangle_third_side_product_l5_5331


namespace find_n_from_sum_of_coeffs_l5_5058

-- The mathematical conditions and question translated to Lean

def sum_of_coefficients (n : ℕ) : ℕ := 6 ^ n
def binomial_coefficients_sum (n : ℕ) : ℕ := 2 ^ n

theorem find_n_from_sum_of_coeffs (n : ℕ) (M N : ℕ) (hM : M = sum_of_coefficients n) (hN : N = binomial_coefficients_sum n) (condition : M - N = 240) : n = 4 :=
by
  sorry

end find_n_from_sum_of_coeffs_l5_5058


namespace men_seated_count_l5_5138

theorem men_seated_count (total_passengers : ℕ) (two_thirds_women : total_passengers * 2 / 3 = women)
                         (one_eighth_standing : total_passengers / 3 / 8 = standing_men) :
  total_passengers = 48 →
  women = 32 →
  standing_men = 2 →
  men_seated = (total_passengers - women) - standing_men →
  men_seated = 14 :=
by
  intros
  sorry

end men_seated_count_l5_5138


namespace find_range_of_a_l5_5363

variable {a : ℝ}
variable {x : ℝ}

theorem find_range_of_a (h₁ : x ∈ Set.Ioo (-2:ℝ) (-1:ℝ)) :
  ∃ a, a ∈ Set.Icc (1:ℝ) (2:ℝ) ∧ (x + 1)^2 < Real.log (|x|) / Real.log a :=
by
  sorry

end find_range_of_a_l5_5363


namespace find_enclosed_area_l5_5374

/-- Given a circle of radius 4 centered at O, and square OABC with side length √3.
The sides AB and BC are extended to form equilateral triangles ABD and BCE with D and E 
lying on the circle. We want to find the area of the region enclosed by segments BD, BE, 
and the minor arc DE. -/
noncomputable def enclosed_area : ℝ :=
  let R := 4 in
  let s := real.sqrt 3 in
  let θ := real.arccos (31 / 32) in
  (θ / (real.pi / 180)) / 360 * real.pi * R^2 - (3 * real.sqrt 3 / 4)

theorem find_enclosed_area :
  enclosed_area = (real.arccos (31 / 32) / (real.pi / 180)) / 360 * real.pi * 16 - (3 * real.sqrt 3 / 4) :=
sorry

end find_enclosed_area_l5_5374


namespace geometric_sequence_problem_l5_5808

theorem geometric_sequence_problem
  (a : ℕ → ℝ) (r : ℝ)
  (h₀ : ∀ n, a n > 0)
  (h₁ : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)
  (h₂ : ∀ n, a (n + 1) = a n * r) :
  a 3 + a 5 = 5 :=
sorry

end geometric_sequence_problem_l5_5808


namespace trip_duration_17_hours_l5_5486

theorem trip_duration_17_hours :
  ∃ T : ℝ, 
    (∀ d₁ d₂ : ℝ,
      (d₁ / 30 + 1 + (150 - d₁) / 4 = T) ∧ 
      (d₁ / 30 + d₂ / 30 + (150 - (d₁ - d₂)) / 30 = T) ∧ 
      ((d₁ - d₂) / 4 + (150 - (d₁ - d₂)) / 30 = T))
  → T = 17 :=
by
  sorry

end trip_duration_17_hours_l5_5486


namespace parallel_line_slope_l5_5942

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℚ, m = 1 / 2 :=
by {
  use 1 / 2,
  sorry
}

end parallel_line_slope_l5_5942


namespace sum_infinite_series_l5_5396

theorem sum_infinite_series : ∑' n : ℕ, (4 * (n + 1) - 3) / (3 ^ (n + 1)) = 3 / 2 := by
    sorry

end sum_infinite_series_l5_5396


namespace parallel_slope_l5_5952

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l5_5952


namespace decagon_adjacent_vertices_probability_l5_5303

namespace ProbabilityAdjacentVertices

def total_vertices : ℕ := 10

def total_pairs : ℕ := total_vertices * (total_vertices - 1) / 2

def adjacent_pairs : ℕ := total_vertices

def probability_adjacent : ℚ := adjacent_pairs / total_pairs

theorem decagon_adjacent_vertices_probability :
  probability_adjacent = 2 / 9 := by
  sorry

end ProbabilityAdjacentVertices

end decagon_adjacent_vertices_probability_l5_5303


namespace parallel_line_slope_l5_5916

def slope_of_parallel_line : ℚ :=
  let line_eq := (3 : ℚ) * (x : ℚ) - (6 : ℚ) * (y : ℚ) = (12 : ℚ)
  in (1 / 2 : ℚ)

theorem parallel_line_slope (x y : ℚ) (h : 3 * x - 6 * y = 12) : slope_of_parallel_line = 1 / 2 :=
by
  sorry

end parallel_line_slope_l5_5916


namespace ratio_of_b_to_c_l5_5134

theorem ratio_of_b_to_c (a b c : ℝ) 
  (h1 : a / b = 11 / 3) 
  (h2 : a / c = 0.7333333333333333) : 
  b / c = 1 / 5 := 
by
  sorry

end ratio_of_b_to_c_l5_5134


namespace cos_alpha_plus_two_pi_over_three_l5_5822

theorem cos_alpha_plus_two_pi_over_three (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) :
  Real.cos (α + 2 * π / 3) = -1 / 3 :=
by
  sorry

end cos_alpha_plus_two_pi_over_three_l5_5822


namespace ratio_of_areas_l5_5438

noncomputable theory

open EuclideanGeometry

def midpoint (B C M : Point) := dist B M = dist C M

theorem ratio_of_areas (A B C D O M E : Point) (h_convex : ConvexQuadrilateral A B C D)
  (h_intersect : collinear A C O ∧ collinear B D O) 
  (h_midpoint : midpoint B C M)
  (h_intersect': line_through M O ∧ line_through A D)
  (h_intersect_ME_AD : E ∈ line_through M O ∧ E ∈  line_through A D ) 
  : (dist A E / dist E D) = (area A B O / area C D O) :=
sorry

end ratio_of_areas_l5_5438


namespace inradius_one_third_height_l5_5734

-- The problem explicitly states this triangle's sides form an arithmetic progression.
-- We need to define conditions and then prove the question is equivalent to the answer given those conditions.
theorem inradius_one_third_height (a b c r h_b : ℝ) (h : a ≤ b ∧ b ≤ c) (h_arith : 2 * b = a + c) :
  r = h_b / 3 :=
sorry

end inradius_one_third_height_l5_5734


namespace cyclist_trip_time_l5_5375

variable (a v : ℝ)
variable (h1 : a / v = 5)

theorem cyclist_trip_time
  (increase_factor : ℝ := 1.25) :
  (a / (2 * v) + a / (2 * (increase_factor * v)) = 4.5) :=
sorry

end cyclist_trip_time_l5_5375


namespace sum_of_x_satisfying_equation_l5_5214

theorem sum_of_x_satisfying_equation :
  let P (x : ℝ) := (x^2 - 5 * x + 3)^(x^2 - 6 * x + 4) = 1
    ; S := { x : ℝ | P x }
    ; L := S.toList
    ; sum := List.sum L
  in sum = 11 :=
sorry

end sum_of_x_satisfying_equation_l5_5214


namespace iron_per_horseshoe_l5_5010

def num_farms := 2
def num_horses_per_farm := 2
def num_stables := 2
def num_horses_per_stable := 5
def num_horseshoes_per_horse := 4
def iron_available := 400
def num_horses_riding_school := 36

-- Lean theorem statement
theorem iron_per_horseshoe : 
  (iron_available / (num_farms * num_horses_per_farm * num_horseshoes_per_horse 
  + num_stables * num_horses_per_stable * num_horseshoes_per_horse 
  + num_horses_riding_school * num_horseshoes_per_horse)) = 2 := 
by 
  sorry

end iron_per_horseshoe_l5_5010


namespace line_passes_through_fixed_point_l5_5040

theorem line_passes_through_fixed_point (k : ℝ) : ∃ (x y : ℝ), y = k * x - k ∧ x = 1 ∧ y = 0 :=
by
  use 1
  use 0
  sorry

end line_passes_through_fixed_point_l5_5040


namespace tangent_line_condition_l5_5501

theorem tangent_line_condition (a b : ℝ):
  ((a = 1 ∧ b = 1) → ∀ x y : ℝ, x + y = 0 → (x - a)^2 + (y - b)^2 = 2 → x = 0 ∧ y = 0) ∧
  ( (a = -1 ∧ b = -1) → ∀ x y : ℝ, x + y = 0 → (x - a)^2 + (y - b)^2 = 2 → x = 0 ∧ y = 0) →
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
by
  sorry

end tangent_line_condition_l5_5501


namespace thirty_three_and_one_third_percent_of_330_l5_5006

theorem thirty_three_and_one_third_percent_of_330 :
  (33 + 1 / 3) / 100 * 330 = 110 :=
sorry

end thirty_three_and_one_third_percent_of_330_l5_5006


namespace sum_series_eq_l5_5393

theorem sum_series_eq :
  ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1) = 3 / 2 :=
sorry

end sum_series_eq_l5_5393


namespace aira_fewer_bands_than_joe_l5_5468

-- Define initial conditions
variables (samantha_bands aira_bands joe_bands : ℕ)
variables (shares_each : ℕ) (total_bands: ℕ)

-- Conditions from the problem
axiom h1 : shares_each = 6
axiom h2 : samantha_bands = aira_bands + 5
axiom h3 : total_bands = shares_each * 3
axiom h4 : aira_bands = 4
axiom h5 : samantha_bands + aira_bands + joe_bands = total_bands

-- The statement to be proven
theorem aira_fewer_bands_than_joe : joe_bands - aira_bands = 1 :=
sorry

end aira_fewer_bands_than_joe_l5_5468


namespace slope_parallel_to_original_line_l5_5946

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l5_5946


namespace Bill_composes_20_problems_l5_5789

theorem Bill_composes_20_problems :
  ∀ (B : ℕ), (∀ R : ℕ, R = 2 * B) →
    (∀ F : ℕ, F = 3 * R) →
    (∀ T : ℕ, T = 4) →
    (∀ P : ℕ, P = 30) →
    (∀ F : ℕ, F = T * P) →
    (∃ B : ℕ, B = 20) :=
by sorry

end Bill_composes_20_problems_l5_5789


namespace analytical_expression_of_f_range_of_f_on_interval_l5_5684

noncomputable def f (x : ℝ) (a c : ℝ) : ℝ := a * x^3 + c * x

theorem analytical_expression_of_f
  (a c : ℝ)
  (h1 : a > 0)
  (h2 : ∀ x, f x a c = a * x^3 + c * x) 
  (h3 : 3 * a + c = -6)
  (h4 : ∀ x, (3 * a * x ^ 2 + c) ≥ -12) :
    a = 2 ∧ c = -12 :=
by
  sorry

theorem range_of_f_on_interval
  (h1 : ∃ a c, a = 2 ∧ c = -12)
  (h2 : ∀ x, f x 2 (-12) = 2 * x^3 - 12 * x)
  :
    Set.range (fun x => f x 2 (-12)) = Set.Icc (-8 * Real.sqrt 2) (8 * Real.sqrt 2) :=
by
  sorry

end analytical_expression_of_f_range_of_f_on_interval_l5_5684


namespace find_nonzero_c_l5_5220

def quadratic_has_unique_solution (c b : ℝ) : Prop :=
  (b^4 + (1 - 4 * c) * b^2 + 1 = 0) ∧ (1 - 4 * c)^2 - 4 = 0

theorem find_nonzero_c (c : ℝ) (b : ℝ) (h_nonzero : c ≠ 0) (h_unique_sol : quadratic_has_unique_solution c b) : 
  c = 3 / 4 := 
sorry

end find_nonzero_c_l5_5220


namespace overall_average_speed_l5_5456

-- Define the conditions for Mark's travel
def time_cycling : ℝ := 1
def speed_cycling : ℝ := 20
def time_walking : ℝ := 2
def speed_walking : ℝ := 3

-- Define the total distance and total time
def total_distance : ℝ :=
  (time_cycling * speed_cycling) + (time_walking * speed_walking)

def total_time : ℝ :=
  time_cycling + time_walking

-- Define the proved statement for the average speed
theorem overall_average_speed : total_distance / total_time = 8.67 :=
by
  sorry

end overall_average_speed_l5_5456


namespace hanks_pancakes_needed_l5_5787

/-- Hank's pancake calculation problem -/
theorem hanks_pancakes_needed 
    (pancakes_per_big_stack : ℕ := 5)
    (pancakes_per_short_stack : ℕ := 3)
    (big_stack_orders : ℕ := 6)
    (short_stack_orders : ℕ := 9) :
    (pancakes_per_short_stack * short_stack_orders) + (pancakes_per_big_stack * big_stack_orders) = 57 := by {
  sorry
}

end hanks_pancakes_needed_l5_5787


namespace cos_beta_eq_sqrt10_over_10_l5_5558

-- Define the conditions and the statement
theorem cos_beta_eq_sqrt10_over_10 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_tan : Real.tan α = 2)
  (h_sin_sum : Real.sin (α + β) = Real.sqrt 2 / 2) :
  Real.cos β = Real.sqrt 10 / 10 :=
sorry

end cos_beta_eq_sqrt10_over_10_l5_5558


namespace men_seated_on_bus_l5_5140

theorem men_seated_on_bus (total_passengers : ℕ) (women_fraction men_standing_fraction : ℚ)
  (h_total : total_passengers = 48)
  (h_women_fraction : women_fraction = 2/3)
  (h_men_standing_fraction : men_standing_fraction = 1/8) :
  let women := (total_passengers : ℚ) * women_fraction,
      men := (total_passengers : ℚ) - women,
      men_standing := men * men_standing_fraction,
      men_seated := men - men_standing in
  men_seated = 14 :=
by
  sorry

end men_seated_on_bus_l5_5140


namespace questionnaire_visitors_l5_5973

noncomputable def total_visitors :=
  let V := 600
  let E := (3 / 4) * V
  V

theorem questionnaire_visitors:
  ∃ (V : ℕ), V = 600 ∧
  (∀ (E : ℕ), E = (3 / 4) * V ∧ E + 150 = V) :=
by
    use 600
    sorry

end questionnaire_visitors_l5_5973


namespace statement_1_incorrect_statement_3_incorrect_statement_4_incorrect_l5_5384

-- Define the notion of line and plane
def Line := Type
def Plane := Type

-- Define the relations: parallel, contained-in, and intersection
def parallel (a b : Line) : Prop := sorry
def contained_in (a : Line) (α : Plane) : Prop := sorry
def intersects_at (a : Line) (α : Plane) (P : Type) : Prop := sorry

-- Conditions translated into Lean
def cond1 (a : Line) (α : Plane) (b : Line) : Prop := parallel a α ∧ contained_in b α → parallel a b
def cond2 (a : Line) (α : Plane) (b : Line) {P : Type} : Prop := intersects_at a α P ∧ contained_in b α → ¬ parallel a b
def cond3 (a : Line) (α : Plane) : Prop := ¬ contained_in a α → parallel a α
def cond4 (a : Line) (α : Plane) (b : Line) : Prop := parallel a α ∧ parallel b α → parallel a b

-- The statements that need to be proved incorrect
theorem statement_1_incorrect (a : Line) (α : Plane) (b : Line) : ¬ (cond1 a α b) := sorry
theorem statement_3_incorrect (a : Line) (α : Plane) : ¬ (cond3 a α) := sorry
theorem statement_4_incorrect (a : Line) (α : Plane) (b : Line) : ¬ (cond4 a α b) := sorry

end statement_1_incorrect_statement_3_incorrect_statement_4_incorrect_l5_5384


namespace negation_of_p_l5_5688

def p : Prop := ∀ x : ℝ, x^2 - 2*x + 2 ≤ Real.sin x
def not_p : Prop := ∃ x : ℝ, x^2 - 2*x + 2 > Real.sin x

theorem negation_of_p : ¬ p ↔ not_p := by
  sorry

end negation_of_p_l5_5688


namespace tangent_line_value_l5_5726

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - 2 * x

theorem tangent_line_value (a b : ℝ) (h : a ≤ 0) 
  (h_tangent : ∀ x : ℝ, f a x = 2 * x + b) : a - 2 * b = 2 :=
sorry

end tangent_line_value_l5_5726


namespace average_goals_per_game_l5_5604

theorem average_goals_per_game
  (number_of_pizzas : ℕ)
  (slices_per_pizza : ℕ)
  (number_of_games : ℕ)
  (h1 : number_of_pizzas = 6)
  (h2 : slices_per_pizza = 12)
  (h3 : number_of_games = 8) : 
  (number_of_pizzas * slices_per_pizza) / number_of_games = 9 :=
by
  sorry

end average_goals_per_game_l5_5604


namespace combined_age_of_siblings_l5_5200

theorem combined_age_of_siblings (a s h : ℕ) (h1 : a = 15) (h2 : s = 3 * a) (h3 : h = 4 * s) : a + s + h = 240 :=
by
  sorry

end combined_age_of_siblings_l5_5200


namespace thirtieth_term_of_arithmetic_seq_l5_5358

def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem thirtieth_term_of_arithmetic_seq : 
  arithmetic_seq 3 4 30 = 119 := 
by
  sorry

end thirtieth_term_of_arithmetic_seq_l5_5358


namespace coat_retrieve_count_l5_5027

theorem coat_retrieve_count : 
  let n := 5 in
  let count := (Finset.card (Finset.filter (λ s, Finset.card s ≥ 2) (Finset.powerset (Finset.range n)))) in
  count = 31 := sorry

end coat_retrieve_count_l5_5027


namespace probability_between_six_and_ten_l5_5619

open Finset

def spinner1 : Finset ℕ := {1, 4, 5}
def spinner2 : Finset ℕ := {2, 3, 6}

def favorable_sums : Finset ℕ := spinner1.product spinner2 
  |>.filter (λ x, let sum := x.1 + x.2 in 6 ≤ sum ∧ sum ≤ 10) 
  |>.image (λ x, x.1 + x.2)

def all_sums : Finset ℕ := spinner1.product spinner2 |>.image (λ x, x.1 + x.2)

def probability : ℚ := favorable_sums.card / all_sums.card

theorem probability_between_six_and_ten : probability = 2 / 3 := by
  sorry

end probability_between_six_and_ten_l5_5619


namespace fraction_to_decimal_l5_5216

theorem fraction_to_decimal : (31 : ℝ) / (2 * 5^6) = 0.000992 :=
by sorry

end fraction_to_decimal_l5_5216


namespace regular_price_of_shrimp_l5_5982

theorem regular_price_of_shrimp 
  (discounted_price : ℝ) 
  (discount_rate : ℝ) 
  (quarter_pound_price : ℝ) 
  (full_pound_price : ℝ) 
  (price_relation : quarter_pound_price = discounted_price * (1 - discount_rate) / 4) 
  (discounted_value : quarter_pound_price = 2) 
  (given_discount_rate : discount_rate = 0.6) 
  (given_discounted_price : discounted_price = full_pound_price) 
  : full_pound_price = 20 :=
by {
  sorry
}

end regular_price_of_shrimp_l5_5982


namespace remainder_sum_first_seven_primes_div_eighth_prime_l5_5170

theorem remainder_sum_first_seven_primes_div_eighth_prime :
  let primes := [2, 3, 5, 7, 11, 13, 17] in
  let sum_first_seven := (List.sum primes) in
  let eighth_prime := 19 in
  sum_first_seven % eighth_prime = 1 :=
by
  let primes := [2, 3, 5, 7, 11, 13, 17]
  let sum_first_seven := (List.sum primes)
  let eighth_prime := 19
  show sum_first_seven % eighth_prime = 1
  sorry

end remainder_sum_first_seven_primes_div_eighth_prime_l5_5170


namespace thirtieth_term_value_l5_5348

-- Define the initial term and common difference
def initial_term : ℤ := 3
def common_difference : ℤ := 4

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ :=
  initial_term + (n - 1) * common_difference

-- Theorem stating the value of the 30th term
theorem thirtieth_term_value : nth_term 30 = 119 :=
by sorry

end thirtieth_term_value_l5_5348


namespace prob_not_same_city_is_056_l5_5336

def probability_not_same_city (P_A_cityA P_B_cityA : ℝ) : ℝ :=
  let P_A_cityB := 1 - P_A_cityA
  let P_B_cityB := 1 - P_B_cityA
  (P_A_cityA * P_B_cityB) + (P_A_cityB * P_B_cityA)

theorem prob_not_same_city_is_056 :
  probability_not_same_city 0.6 0.2 = 0.56 :=
by
  sorry

end prob_not_same_city_is_056_l5_5336


namespace sector_area_l5_5812

theorem sector_area (theta : ℝ) (r : ℝ) (h_theta : theta = 2 * π / 3) (h_r : r = 3) : 
    (theta / (2 * π) * π * r^2) = 3 * π :=
by 
  -- Placeholder for the actual proof
  sorry

end sector_area_l5_5812


namespace initial_pounds_of_coffee_l5_5376

variable (x : ℝ) (h1 : 0.25 * x = d₀) (h2 : 0.60 * 100 = d₁) 
          (h3 : (d₀ + d₁) / (x + 100) = 0.32)

theorem initial_pounds_of_coffee (d₀ d₁ : ℝ) : 
  x = 400 :=
by
  -- Given conditions
  have h1 : d₀ = 0.25 * x := sorry
  have h2 : d₁ = 0.60 * 100 := sorry
  have h3 : 0.32 = (d₀ + d₁) / (x + 100) := sorry
  
  -- Additional steps to solve for x
  sorry

end initial_pounds_of_coffee_l5_5376


namespace min_voters_to_win_l5_5082

def num_voters : ℕ := 105
def num_districts : ℕ := 5
def num_sections_per_district : ℕ := 7
def voters_per_section : ℕ := 3
def majority n : ℕ := n / 2 + 1

theorem min_voters_to_win (Tall_won : ∃ sections : fin num_voters → bool, 
  (∃ districts : fin num_districts → bool, 
    (countp (λ i, districts i = tt) (finset.univ : finset (fin num_districts)) ≥ majority num_districts) ∧ 
    ∀ i : fin num_districts, districts i = tt →
      (countp (λ j, sections (i * num_sections_per_district + j) = tt) (finset.range num_sections_per_district) ≥ majority num_sections_per_district)
  ) ∧
  (∀ i, i < num_voters →¬ (sections i = tt → sections ((i / num_sections_per_district) * num_sections_per_district + (i % num_sections_per_district)) = tt))
  ) : 3 * (12 * 2) ≥ 24 :=
by sorry

end min_voters_to_win_l5_5082


namespace functional_eq_solution_l5_5591

theorem functional_eq_solution (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (m + f n) = f m + n) : ∀ n, f n = n := 
by
  sorry

end functional_eq_solution_l5_5591


namespace arithmetic_sequence_30th_term_l5_5344

-- Defining the initial term and the common difference of the arithmetic sequence
def a : ℕ := 3
def d : ℕ := 4

-- Defining the general formula for the n-th term of the arithmetic sequence
def a_n (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem stating that the 30th term of the given sequence is 119
theorem arithmetic_sequence_30th_term : a_n 30 = 119 := 
sorry

end arithmetic_sequence_30th_term_l5_5344


namespace three_digit_number_equal_sum_of_factorials_and_has_three_l5_5763

def is_three_digit (n : Nat) := 100 ≤ n ∧ n < 1000

def has_digit_three (n : Nat) := n / 100 = 3 ∨ (n % 100) / 10 = 3 ∨ (n % 10) = 3

def sum_of_digit_factorials (n : Nat) : Nat :=
  (n / 100)! + ((n % 100) / 10)! + (n % 10)!

theorem three_digit_number_equal_sum_of_factorials_and_has_three :
  ∃ n : Nat, is_three_digit n ∧ has_digit_three n ∧ sum_of_digit_factorials n = n :=
sorry

end three_digit_number_equal_sum_of_factorials_and_has_three_l5_5763


namespace focal_radii_l5_5024

theorem focal_radii (a e x y : ℝ) (h1 : x + y = 2 * a) (h2 : x - y = 2 * e) : x = a + e ∧ y = a - e :=
by
  -- We will add here the actual proof, but for now, we leave it as a placeholder.
  sorry

end focal_radii_l5_5024


namespace even_function_a_one_l5_5075

def f (a : ℝ) (x : ℝ) : ℝ := a * 3^x + 1 / 3^x

theorem even_function_a_one : (∀ x, f a (-x) = f a x) → a = 1 :=
by
  sorry

end even_function_a_one_l5_5075


namespace value_of_b_conditioned_l5_5735

theorem value_of_b_conditioned
  (b: ℝ) 
  (h0 : 0 < b ∧ b < 7)
  (h1 : (1 / 2) * (8 - b) * (b - 8) / ((1 / 2) * (b / 2) * b) = 4 / 9):
  b = 4 := 
sorry

end value_of_b_conditioned_l5_5735


namespace range_of_a_l5_5242

theorem range_of_a (a : ℝ) : 
  (∀ x : ℕ, (1 ≤ x ∧ x ≤ 4) → ax + 4 ≥ 0) → (-1 ≤ a ∧ a < -4/5) :=
by
  sorry

end range_of_a_l5_5242


namespace range_of_eccentricity_l5_5418

theorem range_of_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) (x y : ℝ) 
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1) (c : ℝ := Real.sqrt (a^2 - b^2)) 
  (h_dot_product : ∀ (x y: ℝ) (h_point : x^2 / a^2 + y^2 / b^2 = 1), 
    let PF1 : ℝ × ℝ := (-c - x, -y)
    let PF2 : ℝ × ℝ := (c - x, -y)
    PF1.1 * PF2.1 + PF1.2 * PF2.2 ≤ a * c) : 
  ∀ (e : ℝ := c / a), (Real.sqrt 5 - 1) / 2 ≤ e ∧ e < 1 := 
by 
  sorry

end range_of_eccentricity_l5_5418


namespace extracurricular_books_l5_5482

theorem extracurricular_books (a b c d : ℕ) 
  (h1 : b + c + d = 110)
  (h2 : a + c + d = 108)
  (h3 : a + b + d = 104)
  (h4 : a + b + c = 119) :
  a = 37 ∧ b = 39 ∧ c = 43 ∧ d = 28 :=
by {
  -- Proof to be done here
  sorry
}

end extracurricular_books_l5_5482


namespace xyz_sum_l5_5127

theorem xyz_sum (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y) : x + y + z = 16 * x :=
by
  sorry

end xyz_sum_l5_5127


namespace pushups_total_l5_5182

theorem pushups_total (z d e : ℕ)
  (hz : z = 44) 
  (hd : d = z + 58) 
  (he : e = 2 * d) : 
  z + d + e = 350 := by
  sorry

end pushups_total_l5_5182


namespace jen_age_proof_l5_5842

variable (JenAge : ℕ) (SonAge : ℕ)

theorem jen_age_proof (h1 : SonAge = 16) (h2 : JenAge = 3 * SonAge - 7) : JenAge = 41 :=
by
  -- conditions
  rw [h1] at h2
  -- substitution and simplification
  have h3 : JenAge = 3 * 16 - 7 := h2
  norm_num at h3
  exact h3

end jen_age_proof_l5_5842


namespace mary_max_weekly_earnings_l5_5115

noncomputable def mary_weekly_earnings (max_hours : ℕ) (regular_hours : ℕ) (regular_rate : ℕ) (overtime_rate_factor : ℕ) : ℕ :=
  let overtime_hours := max_hours - regular_hours
  let overtime_rate := regular_rate + regular_rate * (overtime_rate_factor / 100)
  (regular_hours * regular_rate) + (overtime_hours * overtime_rate)

theorem mary_max_weekly_earnings : mary_weekly_earnings 60 30 12 50 = 900 :=
by
  sorry

end mary_max_weekly_earnings_l5_5115


namespace Aiyanna_cookies_l5_5650

-- Define the conditions
def Alyssa_cookies : ℕ := 129
variable (x : ℕ)
def difference_condition : Prop := (Alyssa_cookies - x) = 11

-- The theorem to prove
theorem Aiyanna_cookies (x : ℕ) (h : difference_condition x) : x = 118 :=
by sorry

end Aiyanna_cookies_l5_5650


namespace arithmetic_sequence_ninth_term_l5_5884

theorem arithmetic_sequence_ninth_term 
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29)
  : a + 8 * d = 35 :=
by
  sorry

end arithmetic_sequence_ninth_term_l5_5884


namespace tall_wins_min_voters_l5_5093

structure VotingSetup where
  total_voters : ℕ
  districts : ℕ
  sections_per_district : ℕ
  voters_per_section : ℕ
  voters_majority_in_section : ℕ
  districts_to_win : ℕ
  sections_to_win_district : ℕ

def contest_victory (setup : VotingSetup) (min_voters : ℕ) : Prop :=
  setup.total_voters = 105 ∧
  setup.districts = 5 ∧
  setup.sections_per_district = 7 ∧
  setup.voters_per_section = 3 ∧
  setup.voters_majority_in_section = 2 ∧
  setup.districts_to_win = 3 ∧
  setup.sections_to_win_district = 4 ∧
  min_voters = 24

theorem tall_wins_min_voters : ∃ min_voters, contest_victory ⟨105, 5, 7, 3, 2, 3, 4⟩ min_voters :=
by { use 24, sorry }

end tall_wins_min_voters_l5_5093


namespace call_center_agents_ratio_l5_5975

noncomputable def fraction_of_agents (calls_A calls_B total_agents total_calls : ℕ) : ℚ :=
  let calls_A_per_agent := calls_A / total_agents
  let calls_B_per_agent := calls_B / total_agents
  let ratio_calls_A_B := (3: ℚ) / 5
  let fraction_calls_B := (8: ℚ) / 11
  let fraction_calls_A := (3: ℚ) / 11
  let ratio_of_agents := (5: ℚ) / 11
  if (calls_A_per_agent * fraction_calls_A = ratio_calls_A_B * calls_B_per_agent) then ratio_of_agents else 0

theorem call_center_agents_ratio (calls_A calls_B total_agents total_calls agents_A agents_B : ℕ) :
  (calls_A : ℚ) / (calls_B : ℚ) = (3 / 5) →
  (calls_B : ℚ) = (8 / 11) * total_calls →
  (agents_A : ℚ) = (5 / 11) * (agents_B : ℚ) :=
sorry

end call_center_agents_ratio_l5_5975


namespace num_intersections_l5_5068

noncomputable def polar_to_cartesian (r θ: ℝ): ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem num_intersections (θ: ℝ): 
  let c1 := polar_to_cartesian (6 * Real.cos θ) θ
  let c2 := polar_to_cartesian (10 * Real.sin θ) θ
  let (x1, y1) := c1
  let (x2, y2) := c2
  ((x1 - 3)^2 + y1^2 = 9 ∧ x2^2 + (y2 - 5)^2 = 25) →
  (x1, y1) = (x2, y2) ↔ false :=
by
  sorry

end num_intersections_l5_5068


namespace packets_in_box_l5_5528

theorem packets_in_box 
  (coffees_per_day : ℕ) 
  (packets_per_coffee : ℕ) 
  (cost_per_box : ℝ) 
  (total_cost : ℝ) 
  (days : ℕ) 
  (P : ℕ) 
  (h_coffees_per_day : coffees_per_day = 2)
  (h_packets_per_coffee : packets_per_coffee = 1)
  (h_cost_per_box : cost_per_box = 4)
  (h_total_cost : total_cost = 24)
  (h_days : days = 90)
  : P = 30 := 
by
  sorry

end packets_in_box_l5_5528


namespace water_level_in_cubic_tank_is_one_l5_5773

def cubic_tank : Type := {s : ℝ // s > 0}

def water_volume (s : cubic_tank) : ℝ := 
  let ⟨side, _⟩ := s 
  side^3

def water_level (s : cubic_tank) (volume : ℝ) (fill_ratio : ℝ) : ℝ := 
  let ⟨side, _⟩ := s 
  fill_ratio * side

theorem water_level_in_cubic_tank_is_one
  (s : cubic_tank)
  (h1 : water_volume s = 64)
  (h2 : water_volume s / 4 = 16)
  (h3 : 0 < 0.25 ∧ 0.25 ≤ 1) :
  water_level s 16 0.25 = 1 :=
by 
  sorry

end water_level_in_cubic_tank_is_one_l5_5773


namespace maximize_probability_remove_6_l5_5155

def initial_list : List ℤ := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def sum_pairs (l : List ℤ) : List (ℤ × ℤ) :=
  List.filter (λ (p : ℤ × ℤ), p.1 + p.2 = 12 ∧ p.1 ≠ p.2) (l.product l)

def num_valid_pairs (l : List ℤ) : ℕ :=
  (sum_pairs l).length / 2 -- Pairs (a,b) and (b,a) are the same for sums, so divide by 2.

theorem maximize_probability_remove_6 :
  ∀x ∈ initial_list,
  num_valid_pairs (List.erase initial_list x) ≤ num_valid_pairs (List.erase initial_list 6) :=
by
  sorry

end maximize_probability_remove_6_l5_5155


namespace crimson_valley_skirts_l5_5029

theorem crimson_valley_skirts (e : ℕ) (a : ℕ) (s : ℕ) (p : ℕ) (c : ℕ) 
  (h1 : e = 120) 
  (h2 : a = 2 * e) 
  (h3 : s = 3 * a / 5) 
  (h4 : p = s / 4) 
  (h5 : c = p / 3) : 
  c = 12 := 
by 
  sorry

end crimson_valley_skirts_l5_5029


namespace anthony_pencils_l5_5386

theorem anthony_pencils (P : Nat) (h : P + 56 = 65) : P = 9 :=
by
  sorry

end anthony_pencils_l5_5386


namespace b_earns_more_than_a_l5_5497

-- Definitions for the conditions
def investments_ratio := (3, 4, 5)
def returns_ratio := (6, 5, 4)
def total_earnings := 10150

-- We need to prove the statement
theorem b_earns_more_than_a (x y : ℕ) (hx : 58 * x * y = 10150) : 2 * x * y = 350 := by
  -- Conditions based on ratios
  let earnings_a := 3 * x * 6 * y
  let earnings_b := 4 * x * 5 * y
  let difference := earnings_b - earnings_a
  
  -- To complete the proof, sorry is used
  sorry

end b_earns_more_than_a_l5_5497


namespace three_subsets_equal_sum_l5_5543

theorem three_subsets_equal_sum (n : ℕ) (h1 : n ≡ 0 [MOD 3] ∨ n ≡ 2 [MOD 3]) (h2 : 5 ≤ n) :
  ∃ (A B C : Finset ℕ), A ∪ B ∪ C = Finset.range (n + 1) ∧
                        A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ C ∩ A = ∅ ∧
                        A.sum id = B.sum id ∧ B.sum id = C.sum id ∧ C.sum id = A.sum id :=
sorry

end three_subsets_equal_sum_l5_5543


namespace fish_remaining_when_discovered_l5_5148

def start_fish := 60
def fish_eaten_per_day := 2
def days_two_weeks := 2 * 7
def fish_added_after_two_weeks := 8
def days_one_week := 7

def fish_after_two_weeks (start: ℕ) (eaten_per_day: ℕ) (days: ℕ) (added: ℕ): ℕ :=
  start - eaten_per_day * days + added

def fish_after_three_weeks (fish_after_two_weeks: ℕ) (eaten_per_day: ℕ) (days: ℕ): ℕ :=
  fish_after_two_weeks - eaten_per_day * days

theorem fish_remaining_when_discovered :
  (fish_after_three_weeks (fish_after_two_weeks start_fish fish_eaten_per_day days_two_weeks fish_added_after_two_weeks) fish_eaten_per_day days_one_week) = 26 := 
by {
  sorry
}

end fish_remaining_when_discovered_l5_5148


namespace coats_count_l5_5707

def initial_minks : Nat := 30
def babies_per_mink : Nat := 6
def minks_per_coat : Nat := 15

def total_minks : Nat := initial_minks + (initial_minks * babies_per_mink)
def remaining_minks : Nat := total_minks / 2

theorem coats_count : remaining_minks / minks_per_coat = 7 := by
  -- Proof goes here
  sorry

end coats_count_l5_5707


namespace moving_circle_trajectory_l5_5752

noncomputable def trajectory (O₁ O₂ O : Type) [MetricSpace O₁] [MetricSpace O₂] [MetricSpace O] 
  (r₁ r₂ : ℝ) (h1 : r₁ ≠ r₂) (h2 : ¬∃ p ∈ O₁, p ∈ O₂) (tangent : ∀ p ∈ O, p ∈ O₁ ∧ p ∈ O₂) : Prop :=
isHyperbolaBranch O O₁ O₂ ∨ isEllipse O O₁ O₂

theorem moving_circle_trajectory 
(O₁ O₂ O : Type) [MetricSpace O₁] [MetricSpace O₂] [MetricSpace O] 
(r₁ r₂ : ℝ) (h1 : r₁ ≠ r₂) (h2 : ¬∃ p ∈ O₁, p ∈ O₂) (tangent : ∀ p ∈ O, p ∈ O₁ ∧ p ∈ O₂) :
trajectory O₁ O₂ O r₁ r₂ h1 h2 tangent := 
sorry

end moving_circle_trajectory_l5_5752


namespace parallel_line_slope_l5_5915

def slope_of_parallel_line : ℚ :=
  let line_eq := (3 : ℚ) * (x : ℚ) - (6 : ℚ) * (y : ℚ) = (12 : ℚ)
  in (1 / 2 : ℚ)

theorem parallel_line_slope (x y : ℚ) (h : 3 * x - 6 * y = 12) : slope_of_parallel_line = 1 / 2 :=
by
  sorry

end parallel_line_slope_l5_5915


namespace max_k_value_l5_5670

noncomputable def f (x : ℝ) := x + x * Real.log x

theorem max_k_value : ∃ k : ℤ, (∀ x > 2, k * (x - 2) < f x) ∧ k = 4 :=
by
  sorry

end max_k_value_l5_5670


namespace unit_square_BE_value_l5_5265

theorem unit_square_BE_value
  (ABCD : ℝ × ℝ → Prop)
  (unit_square : ∀ (a b c d : ℝ × ℝ), ABCD a ∧ ABCD b ∧ ABCD c ∧ ABCD d → 
                  a.1 = 0 ∧ a.2 = 0 ∧ b.1 = 1 ∧ b.2 = 0 ∧ 
                  c.1 = 1 ∧ c.2 = 1 ∧ d.1 = 0 ∧ d.2 = 1)
  (E F G : ℝ × ℝ)
  (on_sides : E.1 = 1 ∧ F.2 = 1 ∧ G.1 = 0)
  (AE_perp_EF : ((E.1 - 0) * (F.2 - E.2)) + ((E.2 - 0) * (F.1 - E.1)) = 0)
  (EF_perp_FG : ((F.1 - E.1) * (G.2 - F.2)) + ((F.2 - E.2) * (G.1 - F.1)) = 0)
  (GA_val : (1 - G.1) = 404 / 1331) :
  ∃ BE, BE = 9 / 11 := 
sorry

end unit_square_BE_value_l5_5265


namespace basketball_games_l5_5077

theorem basketball_games (N M : ℕ) (h1 : N > 3 * M) (h2 : M > 5) (h3 : 3 * N + 4 * M = 88) : 3 * N = 48 :=
by sorry

end basketball_games_l5_5077


namespace evaluate_ratio_l5_5411

theorem evaluate_ratio : (2^2003 * 3^2002) / (6^2002) = 2 := 
by {
  sorry
}

end evaluate_ratio_l5_5411


namespace yogurt_amount_l5_5656

namespace SmoothieProblem

def strawberries := 0.2 -- cups
def orange_juice := 0.2 -- cups
def total_ingredients := 0.5 -- cups

def yogurt_used := total_ingredients - (strawberries + orange_juice)

theorem yogurt_amount : yogurt_used = 0.1 :=
by
  unfold yogurt_used strawberries orange_juice total_ingredients
  norm_num
  sorry  -- Proof can be filled in as needed

end SmoothieProblem

end yogurt_amount_l5_5656


namespace percentage_difference_between_chef_and_dishwasher_l5_5524

theorem percentage_difference_between_chef_and_dishwasher
    (manager_wage : ℝ)
    (dishwasher_wage : ℝ)
    (chef_wage : ℝ)
    (h1 : manager_wage = 6.50)
    (h2 : dishwasher_wage = manager_wage / 2)
    (h3 : chef_wage = manager_wage - 2.60) :
    (chef_wage - dishwasher_wage) / dishwasher_wage * 100 = 20 :=
by
  -- The proof would go here
  sorry

end percentage_difference_between_chef_and_dishwasher_l5_5524


namespace min_voters_l5_5096

theorem min_voters (total_voters : ℕ) (districts : ℕ) (sections_per_district : ℕ) 
  (voters_per_section : ℕ) (majority_sections : ℕ) (majority_districts : ℕ) 
  (winner : string) (is_tall_winner : winner = "Tall") 
  (total_voters = 105) (districts = 5) (sections_per_district = 7) 
  (voters_per_section = 3) (majority_sections = 4) (majority_districts = 3) :
  ∃ (min_voters : ℕ), min_voters = 24 :=
by
  sorry

end min_voters_l5_5096


namespace valid_five_letter_words_l5_5690

def num_valid_words : Nat :=
  let total_words := 3^5
  let invalid_3_consec := 5 * 2^3 * 1^2
  let invalid_4_consec := 2 * 2^4 * 1
  let invalid_5_consec := 2^5
  total_words - (invalid_3_consec + invalid_4_consec + invalid_5_consec)

theorem valid_five_letter_words : num_valid_words = 139 := by
  sorry

end valid_five_letter_words_l5_5690


namespace probability_of_adjacent_vertices_l5_5317

def decagon_vertices : ℕ := 10

def total_ways_to_choose_2_vertices (n : ℕ) : ℕ := n * (n - 1) / 2

def favorable_ways_to_choose_adjacent_vertices (n : ℕ) : ℕ := 2 * n

theorem probability_of_adjacent_vertices :
  let n := decagon_vertices in
  let total_choices := total_ways_to_choose_2_vertices n in
  let favorable_choices := favorable_ways_to_choose_adjacent_vertices n in
  (favorable_choices : ℚ) / total_choices = 2 / 9 :=
by
  sorry

end probability_of_adjacent_vertices_l5_5317


namespace max_value_of_reciprocals_l5_5655

noncomputable def quadratic (x t q : ℝ) : ℝ := x^2 - t * x + q

theorem max_value_of_reciprocals (α β t q : ℝ) (h1 : α + β = α^2 + β^2)
                                               (h2 : α + β = α^3 + β^3)
                                               (h3 : ∀ n, 1 ≤ n ∧ n ≤ 2010 → α^n + β^n = α + β)
                                               (h4 : α * β = q)
                                               (h5 : α + β = t) :
  ∃ (α β : ℝ), (1 / α^2012 + 1 / β^2012) = 2 := 
sorry

end max_value_of_reciprocals_l5_5655


namespace smallest_integer_inequality_l5_5413

theorem smallest_integer_inequality :
  (∃ n : ℤ, ∀ x y z : ℝ, (x + y + z)^2 ≤ (n:ℝ) * (x^2 + y^2 + z^2)) ∧
  ∀ m : ℤ, (∀ x y z : ℝ, (x + y + z)^2 ≤ (m:ℝ) * (x^2 + y^2 + z^2)) → 3 ≤ m :=
  sorry

end smallest_integer_inequality_l5_5413


namespace distinct_placements_count_l5_5572

theorem distinct_placements_count : 
  ∃ (n : ℕ), 
    (∃ (k : ℕ),
      (∀ (total_boxes : ℕ) (empty_boxes : ℕ), 
        total_boxes = 4 ∧ empty_boxes = 2 →
        ∃ (ways_to_choose_empty : ℕ),
        ways_to_choose_empty = Nat.choose total_boxes empty_boxes ∧
        (∃ (ways_to_arrange_digits : ℕ),
          ways_to_arrange_digits = (Nat.factorial 4) / (Nat.factorial (4 - 2)) ∧
          n = ways_to_choose_empty * ways_to_arrange_digits))) 
  ∧ n = 72 := 
by {
  sorry
}

end distinct_placements_count_l5_5572


namespace angle_C_is_70_l5_5579

namespace TriangleAngleSum

def angle_sum_in_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

def sum_of_two_angles (A B : ℝ) : Prop :=
  A + B = 110

theorem angle_C_is_70 {A B C : ℝ} (h1 : angle_sum_in_triangle A B C) (h2 : sum_of_two_angles A B) : C = 70 :=
by
  sorry

end TriangleAngleSum

end angle_C_is_70_l5_5579


namespace find_missing_number_l5_5186

theorem find_missing_number (x : ℝ) :
  ((20 + 40 + 60) / 3) = ((10 + 70 + x) / 3) + 8 → x = 16 :=
by
  intro h
  sorry

end find_missing_number_l5_5186


namespace points_lie_on_line_l5_5665

noncomputable def x (t : ℝ) (ht : t ≠ 0) : ℝ := (t^2 + 2 * t + 2) / t
noncomputable def y (t : ℝ) (ht : t ≠ 0) : ℝ := (t^2 - 2 * t + 2) / t

theorem points_lie_on_line : ∀ (t : ℝ) (ht : t ≠ 0), y t ht = x t ht - 4 :=
by 
  intros t ht
  simp [x, y]
  sorry

end points_lie_on_line_l5_5665


namespace prove_inequality_l5_5770

variable (f : ℝ → ℝ)

def isEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def isMonotonicOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≥ f y

theorem prove_inequality
  (h1 : isEvenFunction f)
  (h2 : isMonotonicOnInterval f 0 5)
  (h3 : f (-3) < f 1) :
  f 0 > f 1 :=
sorry

end prove_inequality_l5_5770


namespace point_on_graph_l5_5286

noncomputable def f (x : ℝ) : ℝ := abs (x^3 + 1) + abs (x^3 - 1)

theorem point_on_graph (a : ℝ) : ∃ (x y : ℝ), (x = a) ∧ (y = f (-a)) ∧ (y = f x) :=
by 
  sorry

end point_on_graph_l5_5286


namespace actual_selling_price_l5_5978

-- Define the original price m
variable (m : ℝ)

-- Define the discount rate
def discount_rate : ℝ := 0.2

-- Define the selling price
def selling_price := m * (1 - discount_rate)

-- The theorem states the relationship between the original price and the selling price after discount
theorem actual_selling_price : selling_price m = 0.8 * m :=
by
-- Proof step would go here
sorry

end actual_selling_price_l5_5978


namespace sum_infinite_series_l5_5395

theorem sum_infinite_series : ∑' n : ℕ, (4 * (n + 1) - 3) / (3 ^ (n + 1)) = 3 / 2 := by
    sorry

end sum_infinite_series_l5_5395


namespace thirtieth_term_value_l5_5351

-- Define the initial term and common difference
def initial_term : ℤ := 3
def common_difference : ℤ := 4

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ :=
  initial_term + (n - 1) * common_difference

-- Theorem stating the value of the 30th term
theorem thirtieth_term_value : nth_term 30 = 119 :=
by sorry

end thirtieth_term_value_l5_5351


namespace solve_for_x_l5_5071

theorem solve_for_x (x : ℝ) (h : 1 = 1 / (4 * x^2 + 2 * x + 1)) : 
  x = 0 ∨ x = -1 / 2 := 
by sorry

end solve_for_x_l5_5071


namespace sum_of_solutions_l5_5215

theorem sum_of_solutions : 
  (∀ x : ℝ, (x^2 - 5 * x + 3)^(x^2 - 6 * x + 4) = 1) → 
  (∃ s : ℝ, s = 16) :=
by
  sorry

end sum_of_solutions_l5_5215


namespace painting_time_l5_5030

theorem painting_time :
  let time_per_lily := 5
      time_per_rose := 7
      time_per_orchid := 3
      time_per_vine := 2
      num_lilies := 17
      num_roses := 10
      num_orchids := 6
      num_vines := 20 in
  time_per_lily * num_lilies + 
  time_per_rose * num_roses + 
  time_per_orchid * num_orchids + 
  time_per_vine * num_vines = 213 := 
by 
  sorry

end painting_time_l5_5030


namespace negation_of_p_l5_5474

def p := ∃ n : ℕ, n^2 > 2 * n - 1

theorem negation_of_p : ¬ p ↔ ∀ n : ℕ, n^2 ≤ 2 * n - 1 :=
by sorry

end negation_of_p_l5_5474


namespace average_goals_per_game_l5_5603

theorem average_goals_per_game
  (number_of_pizzas : ℕ)
  (slices_per_pizza : ℕ)
  (number_of_games : ℕ)
  (h1 : number_of_pizzas = 6)
  (h2 : slices_per_pizza = 12)
  (h3 : number_of_games = 8) : 
  (number_of_pizzas * slices_per_pizza) / number_of_games = 9 :=
by
  sorry

end average_goals_per_game_l5_5603


namespace problem1_problem2_problem3_problem4_l5_5207

theorem problem1 : 6 + (-8) - (-5) = 3 := by
  sorry

theorem problem2 : (5 + 3/5) + (-(5 + 2/3)) + (4 + 2/5) + (-1/3) = 4 := by
  sorry

theorem problem3 : ((-1/2) + 1/6 - 1/4) * 12 = -7 := by
  sorry

theorem problem4 : -1^2022 + 27 * (-1/3)^2 - |(-5)| = -3 := by
  sorry

end problem1_problem2_problem3_problem4_l5_5207


namespace find_tenth_term_l5_5584

/- Define the general term formula -/
def a (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

/- Define the sum of the first n terms formula -/
def S (a1 d : ℤ) (n : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem find_tenth_term
  (a1 d : ℤ)
  (h1 : a a1 d 2 + a a1 d 5 = 19)
  (h2 : S a1 d 5 = 40) :
  a a1 d 10 = 29 := by
  /- Sorry used to skip the proof steps. -/
  sorry

end find_tenth_term_l5_5584


namespace slope_of_parallel_line_l5_5936

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l5_5936


namespace even_sum_sufficient_not_necessary_l5_5807

theorem even_sum_sufficient_not_necessary (m n : ℤ) : 
  (∀ m n : ℤ, (Even m ∧ Even n) → Even (m + n)) 
  ∧ (∀ a b : ℤ, Even (a + b) → ¬ (Odd a ∧ Odd b)) :=
by
  sorry

end even_sum_sufficient_not_necessary_l5_5807


namespace slope_of_parallel_line_l5_5923

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l5_5923


namespace largest_n_unique_k_l5_5906

theorem largest_n_unique_k :
  ∃ n : ℕ, n = 1 ∧ ∀ k : ℕ, (3 : ℚ) / 7 < (n : ℚ) / ((n + k : ℕ) : ℚ) ∧ 
  (n : ℚ) / ((n + k : ℕ) : ℚ) < (8 : ℚ) / 19 → k = 1 := by
sorry

end largest_n_unique_k_l5_5906


namespace total_wheels_is_90_l5_5297

-- Defining the conditions
def number_of_bicycles := 20
def number_of_cars := 10
def number_of_motorcycles := 5

-- Calculating the total number of wheels
def total_wheels_in_garage : Nat :=
  (2 * number_of_bicycles) + (4 * number_of_cars) + (2 * number_of_motorcycles)

-- Statement to prove
theorem total_wheels_is_90 : total_wheels_in_garage = 90 := by
  sorry

end total_wheels_is_90_l5_5297


namespace max_value_and_period_of_g_value_of_expression_if_fx_eq_2f_l5_5683

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x
noncomputable def g (x : ℝ) : ℝ := f x * f' x - f x ^ 2

theorem max_value_and_period_of_g :
  ∃ (M : ℝ) (T : ℝ), (∀ x, g x ≤ M) ∧ (∀ x, g (x + T) = g x) ∧ M = 2 ∧ T = Real.pi :=
sorry

theorem value_of_expression_if_fx_eq_2f'x (x : ℝ) :
  f x = 2 * f' x → (1 + Real.sin x ^ 2) / (Real.cos x ^ 2 - Real.sin x * Real.cos x) = 11 / 6 :=
sorry

end max_value_and_period_of_g_value_of_expression_if_fx_eq_2f_l5_5683


namespace fido_area_reach_l5_5541

theorem fido_area_reach (r : ℝ) (s : ℝ) (a b : ℕ) (h1 : r = s * (1 / (Real.tan (Real.pi / 8))))
  (h2 : a = 2) (h3 : b = 8)
  (h_fraction : (Real.pi * r ^ 2) / (2 * (1 + Real.sqrt 2) * (r ^ 2 * (Real.tan (Real.pi / 8)) ^ 2)) = (Real.pi * Real.sqrt a) / b) :
  a * b = 16 := by
  sorry

end fido_area_reach_l5_5541


namespace determine_m_l5_5254

theorem determine_m (m : ℝ) : (∀ x : ℝ, (0 < x ∧ x < 2) ↔ -1/2 * x^2 + 2 * x + m * x > 0) → m = -1 :=
by
  intro h
  sorry

end determine_m_l5_5254


namespace solve_inequality_l5_5125

theorem solve_inequality (x : ℝ) : 
  (x - 5) / (x - 3)^2 < 0 ↔ x ∈ Iio 3 ∪ Ioo 3 5 := 
sorry

end solve_inequality_l5_5125


namespace order_of_abc_l5_5691

noncomputable def a : ℝ := 2017^0
noncomputable def b : ℝ := 2015 * 2017 - 2016^2
noncomputable def c : ℝ := ((-2/3)^2016) * ((3/2)^2017)

theorem order_of_abc : b < a ∧ a < c := by
  -- proof omitted
  sorry

end order_of_abc_l5_5691


namespace units_digit_sum_42_4_24_4_l5_5760

theorem units_digit_sum_42_4_24_4 : (42^4 + 24^4) % 10 = 2 := 
by
  sorry

end units_digit_sum_42_4_24_4_l5_5760


namespace total_wholesale_cost_is_correct_l5_5020

-- Given values
def retail_price_pants : ℝ := 36
def markup_pants : ℝ := 0.8

def retail_price_shirt : ℝ := 45
def markup_shirt : ℝ := 0.6

def retail_price_jacket : ℝ := 120
def markup_jacket : ℝ := 0.5

noncomputable def wholesale_cost_pants : ℝ := retail_price_pants / (1 + markup_pants)
noncomputable def wholesale_cost_shirt : ℝ := retail_price_shirt / (1 + markup_shirt)
noncomputable def wholesale_cost_jacket : ℝ := retail_price_jacket / (1 + markup_jacket)

noncomputable def total_wholesale_cost : ℝ :=
  wholesale_cost_pants + wholesale_cost_shirt + wholesale_cost_jacket

theorem total_wholesale_cost_is_correct :
  total_wholesale_cost = 128.125 := by
  sorry

end total_wholesale_cost_is_correct_l5_5020


namespace planting_scheme_correct_l5_5831

-- Setting up the problem as the conditions given
def types_of_seeds := ["peanuts", "Chinese cabbage", "potatoes", "corn", "wheat", "apples"]

def first_plot_seeds := ["corn", "apples"]

def planting_schemes_count : ℕ :=
  let choose_first_plot := 2  -- C(2, 1), choosing either "corn" or "apples" for the first plot
  let remaining_seeds := 5  -- 6 - 1 = 5 remaining seeds after choosing for the first plot
  let arrangements_remaining := 5 * 4 * 3  -- A(5, 3), arrangements of 3 plots from 5 remaining seeds
  choose_first_plot * arrangements_remaining

theorem planting_scheme_correct : planting_schemes_count = 120 := by
  sorry

end planting_scheme_correct_l5_5831


namespace sum_of_polynomials_l5_5391

theorem sum_of_polynomials (d : ℕ) :
  let expr1 := 15 * d + 17 + 16 * d ^ 2
  let expr2 := 3 * d + 2
  let sum_expr := expr1 + expr2
  let a := 16
  let b := 18
  let c := 19
  sum_expr = a * d ^ 2 + b * d + c ∧ a + b + c = 53 := by
    sorry

end sum_of_polynomials_l5_5391


namespace find_number_l5_5470

theorem find_number :
  ∃ x : ℝ, (x - 1.9) * 1.5 + 32 / 2.5 = 20 ∧ x = 13.9 :=
by
  sorry

end find_number_l5_5470


namespace trenton_earning_goal_l5_5151

-- Parameters
def fixed_weekly_earnings : ℝ := 190
def commission_rate : ℝ := 0.04
def sales_amount : ℝ := 7750
def goal : ℝ := 500

-- Proof statement
theorem trenton_earning_goal :
  fixed_weekly_earnings + (commission_rate * sales_amount) = goal :=
by
  sorry

end trenton_earning_goal_l5_5151


namespace average_price_of_pencil_correct_l5_5369

def average_price_of_pencil (n_pens n_pencils : ℕ) (total_cost pen_price : ℕ) : ℕ :=
  let pen_cost := n_pens * pen_price
  let pencil_cost := total_cost - pen_cost
  let avg_pencil_price := pencil_cost / n_pencils
  avg_pencil_price

theorem average_price_of_pencil_correct :
  average_price_of_pencil 30 75 450 10 = 2 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end average_price_of_pencil_correct_l5_5369


namespace average_goals_per_game_l5_5602

theorem average_goals_per_game
  (slices_per_pizza : ℕ := 12)
  (total_pizzas : ℕ := 6)
  (total_games : ℕ := 8)
  (total_slices : ℕ := total_pizzas * slices_per_pizza)
  (total_goals : ℕ := total_slices)
  (average_goals : ℕ := total_goals / total_games) :
  average_goals = 9 :=
by
  sorry

end average_goals_per_game_l5_5602


namespace birds_joined_l5_5625

def numBirdsInitially : Nat := 1
def numBirdsNow : Nat := 5

theorem birds_joined : numBirdsNow - numBirdsInitially = 4 := by
  -- proof goes here
  sorry

end birds_joined_l5_5625


namespace total_items_children_carry_l5_5796

theorem total_items_children_carry 
  (pieces_per_pizza : ℕ) (number_of_fourthgraders : ℕ) (pizza_per_fourthgrader : ℕ) 
  (pepperoni_per_pizza : ℕ) (mushrooms_per_pizza : ℕ) (olives_per_pizza : ℕ) 
  (total_pizzas : ℕ) (total_pieces_of_pizza : ℕ) (total_pepperoni : ℕ) (total_mushrooms : ℕ) 
  (total_olives : ℕ) (total_toppings : ℕ) (total_items : ℕ) : 
  pieces_per_pizza = 6 →
  number_of_fourthgraders = 10 →
  pizza_per_fourthgrader = 20 →
  pepperoni_per_pizza = 5 →
  mushrooms_per_pizza = 3 →
  olives_per_pizza = 8 →
  total_pizzas = number_of_fourthgraders * pizza_per_fourthgrader →
  total_pieces_of_pizza = total_pizzas * pieces_per_pizza →
  total_pepperoni = total_pizzas * pepperoni_per_pizza →
  total_mushrooms = total_pizzas * mushrooms_per_pizza →
  total_olives = total_pizzas * olives_per_pizza →
  total_toppings = total_pepperoni + total_mushrooms + total_olives →
  total_items = total_pieces_of_pizza + total_toppings →
  total_items = 4400 :=
by
  sorry

end total_items_children_carry_l5_5796


namespace house_cost_l5_5275

-- Definitions of given conditions
def annual_salary : ℝ := 150000
def saving_rate : ℝ := 0.10
def downpayment_rate : ℝ := 0.20
def years_saving : ℝ := 6

-- Given the conditions, calculate annual savings and total savings after 6 years
def annual_savings : ℝ := annual_salary * saving_rate
def total_savings : ℝ := annual_savings * years_saving

-- Total savings represents 20% of the house cost
def downpayment : ℝ := total_savings

-- Prove the total cost of the house
theorem house_cost (downpayment : ℝ) (downpayment_rate : ℝ) : ℝ :=
  downpayment / downpayment_rate

lemma house_cost_correct : house_cost downpayment downpayment_rate = 450000 :=
by
  -- the proof would go here
  sorry

end house_cost_l5_5275


namespace max_value_fraction_l5_5055

theorem max_value_fraction : ∀ (x y : ℝ), (-5 ≤ x ∧ x ≤ -1) → (1 ≤ y ∧ y ≤ 3) → (1 + y / x ≤ -2) :=
  by
    intros x y hx hy
    sorry

end max_value_fraction_l5_5055


namespace find_x_in_coconut_grove_l5_5971

theorem find_x_in_coconut_grove
  (x : ℕ)
  (h1 : (x + 2) * 30 + x * 120 + (x - 2) * 180 = 300 * x)
  (h2 : 3 * x ≠ 0) :
  x = 10 :=
by
  sorry

end find_x_in_coconut_grove_l5_5971


namespace slope_of_parallel_line_l5_5935

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l5_5935


namespace intersection_sets_l5_5818

theorem intersection_sets (M N : Set ℝ) :
  (M = {x | x * (x - 3) < 0}) → (N = {x | |x| < 2}) → (M ∩ N = {x | 0 < x ∧ x < 2}) :=
by
  intro hM hN
  rw [hM, hN]
  sorry

end intersection_sets_l5_5818


namespace pages_filled_with_images_ratio_l5_5371

theorem pages_filled_with_images_ratio (total_pages intro_pages text_pages : ℕ) 
  (h_total : total_pages = 98)
  (h_intro : intro_pages = 11)
  (h_text : text_pages = 19)
  (h_blank : 2 * text_pages = total_pages - intro_pages - 2 * text_pages) :
  (total_pages - intro_pages - text_pages - text_pages) / total_pages = 1 / 2 :=
by
  sorry

end pages_filled_with_images_ratio_l5_5371


namespace slope_of_parallel_line_l5_5925

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l5_5925


namespace part1_quantity_of_vegetables_part2_functional_relationship_part3_min_vegetable_a_l5_5902

/-- Part 1: Quantities of vegetables A and B wholesaled. -/
theorem part1_quantity_of_vegetables (x y : ℝ) 
  (h1 : x + y = 40) 
  (h2 : 4.8 * x + 4 * y = 180) : 
  x = 25 ∧ y = 15 :=
sorry

/-- Part 2: Functional relationship between m and n. -/
theorem part2_functional_relationship (n m : ℝ) 
  (h : n ≤ 80) 
  (h2 : m = 4.8 * n + 4 * (80 - n)) : 
  m = 0.8 * n + 320 :=
sorry

/-- Part 3: Minimum amount of vegetable A to ensure profit of at least 176 yuan -/
theorem part3_min_vegetable_a (n : ℝ) 
  (h : 0.8 * n + 128 ≥ 176) : 
  n ≥ 60 :=
sorry

end part1_quantity_of_vegetables_part2_functional_relationship_part3_min_vegetable_a_l5_5902


namespace distance_traveled_l5_5325

-- Define the variables for speed of slower and faster bike
def slower_speed := 60
def faster_speed := 64

-- Define the condition that slower bike takes 1 hour more than faster bike
def condition (D : ℝ) : Prop := (D / slower_speed) = (D / faster_speed) + 1

-- The theorem we need to prove
theorem distance_traveled : ∃ (D : ℝ), condition D ∧ D = 960 := 
by
  sorry

end distance_traveled_l5_5325


namespace range_of_a_exists_x_l5_5577

theorem range_of_a_exists_x (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 3, 2 * x - x ^ 2 ≥ a) ↔ a ≤ 1 := 
sorry

end range_of_a_exists_x_l5_5577


namespace slope_of_parallel_line_l5_5914

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l5_5914


namespace ava_legs_count_l5_5525

-- Conditions:
-- There are a total of 9 animals in the farm.
-- There are only chickens and buffalos in the farm.
-- There are 5 chickens in the farm.

def total_animals : Nat := 9
def num_chickens : Nat := 5
def legs_per_chicken : Nat := 2
def legs_per_buffalo : Nat := 4

-- Proof statement: Ava counted 26 legs.
theorem ava_legs_count (num_buffalos : Nat) 
  (H1 : total_animals = num_chickens + num_buffalos) : 
  num_chickens * legs_per_chicken + num_buffalos * legs_per_buffalo = 26 :=
by 
  have H2 : num_buffalos = total_animals - num_chickens := by sorry
  sorry

end ava_legs_count_l5_5525


namespace correct_transformation_l5_5996

theorem correct_transformation (a b c : ℝ) (h : c ≠ 0) (h1 : a / c = b / c) : a = b :=
by 
  -- Actual proof would go here, but we use sorry for the scaffold.
  sorry

end correct_transformation_l5_5996


namespace base_133_not_perfect_square_l5_5291

theorem base_133_not_perfect_square (b : ℤ) : ¬ ∃ k : ℤ, b^2 + 3 * b + 3 = k^2 := by
  sorry

end base_133_not_perfect_square_l5_5291


namespace vectors_opposite_directions_l5_5675

variable {V : Type*} [AddCommGroup V]

theorem vectors_opposite_directions (a b : V) (h : a + 4 • b = 0) (ha : a ≠ 0) (hb : b ≠ 0) : a = -4 • b :=
by sorry

end vectors_opposite_directions_l5_5675


namespace factorial_division_l5_5370

theorem factorial_division :
  (Nat.factorial 4) / (Nat.factorial (4 - 3)) = 24 :=
by
  sorry

end factorial_division_l5_5370


namespace abs_diff_of_pq_eq_6_and_pq_sum_7_l5_5453

variable (p q : ℝ)

noncomputable def abs_diff (a b : ℝ) := |a - b|

theorem abs_diff_of_pq_eq_6_and_pq_sum_7 (hpq : p * q = 6) (hpq_sum : p + q = 7) : abs_diff p q = 5 :=
by
  sorry

end abs_diff_of_pq_eq_6_and_pq_sum_7_l5_5453


namespace muffin_cost_is_correct_l5_5106

variable (M : ℝ)

def total_original_cost (muffin_cost : ℝ) : ℝ := 3 * muffin_cost + 1.45

def discounted_cost (original_cost : ℝ) : ℝ := 0.85 * original_cost

def kevin_paid (discounted_price : ℝ) : Prop := discounted_price = 3.70

theorem muffin_cost_is_correct (h : discounted_cost (total_original_cost M) = 3.70) : M = 0.97 :=
  by
  sorry

end muffin_cost_is_correct_l5_5106


namespace largest_n_unique_k_l5_5905

theorem largest_n_unique_k :
  ∃ n : ℕ, n = 1 ∧ ∀ k : ℕ, (3 : ℚ) / 7 < (n : ℚ) / ((n + k : ℕ) : ℚ) ∧ 
  (n : ℚ) / ((n + k : ℕ) : ℚ) < (8 : ℚ) / 19 → k = 1 := by
sorry

end largest_n_unique_k_l5_5905


namespace arithmetic_sequence_30th_term_l5_5347

-- Defining the initial term and the common difference of the arithmetic sequence
def a : ℕ := 3
def d : ℕ := 4

-- Defining the general formula for the n-th term of the arithmetic sequence
def a_n (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem stating that the 30th term of the given sequence is 119
theorem arithmetic_sequence_30th_term : a_n 30 = 119 := 
sorry

end arithmetic_sequence_30th_term_l5_5347


namespace area_ratio_triangle_PQR_ABC_l5_5578

noncomputable def area (A B C : ℝ×ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * (x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2)

theorem area_ratio_triangle_PQR_ABC {A B C P Q R : ℝ×ℝ} 
  (h1 : dist A B + dist B C + dist C A = 1)
  (h2 : dist A P + dist P Q + dist Q B + dist B C + dist C A = 1)
  (h3 : dist P Q + dist Q R + dist R P = 1)
  (h4 : P.1 <= A.1 ∧ A.1 <= Q.1 ∧ Q.1 <= B.1) :
  area P Q R / area A B C > 2 / 9 :=
by
  sorry

end area_ratio_triangle_PQR_ABC_l5_5578


namespace determine_b_l5_5693

theorem determine_b (a b c : ℕ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (eq_radicals: Real.sqrt (4 * a + 4 * b / c) = 2 * a * Real.sqrt (b / c)) : 
  b = c + 1 :=
sorry

end determine_b_l5_5693


namespace compute_expression_l5_5282

-- Given Conditions
variables (a b c : ℕ)
variable (h : 2^a * 3^b * 5^c = 36000)

-- Proof Statement
theorem compute_expression (h : 2^a * 3^b * 5^c = 36000) : 3 * a + 4 * b + 6 * c = 41 :=
sorry

end compute_expression_l5_5282


namespace x_cubed_plus_square_plus_lin_plus_a_l5_5073

theorem x_cubed_plus_square_plus_lin_plus_a (a b x : ℝ) (h : b / x^3 + 1 / x^2 + 1 / x + 1 = 0) :
  x^3 + x^2 + x + a = a - b :=
by {
  sorry
}

end x_cubed_plus_square_plus_lin_plus_a_l5_5073


namespace sin_sum_given_cos_tan_conditions_l5_5227

open Real

theorem sin_sum_given_cos_tan_conditions 
  (α β : ℝ)
  (h1 : cos α + cos β = 1 / 3)
  (h2 : tan (α + β) = 24 / 7)
  : sin α + sin β = 1 / 4 ∨ sin α + sin β = -4 / 9 := 
  sorry

end sin_sum_given_cos_tan_conditions_l5_5227


namespace solve_for_n_l5_5469

theorem solve_for_n (n : ℕ) (h : 9^n * 9^n * 9^n * 9^n = 81^n) : n = 0 :=
by
  sorry

end solve_for_n_l5_5469


namespace angle_bisector_theorem_l5_5580

noncomputable def ratio_of_segments (x y z p q : ℝ) :=
  q / x = y / (y + x)

theorem angle_bisector_theorem (x y z p q : ℝ) (h1 : p / x = q / y)
  (h2 : p + q = z) : ratio_of_segments x y z p q :=
by
  sorry

end angle_bisector_theorem_l5_5580


namespace arithmetic_sequence_ninth_term_l5_5883

theorem arithmetic_sequence_ninth_term 
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29)
  : a + 8 * d = 35 :=
by
  sorry

end arithmetic_sequence_ninth_term_l5_5883


namespace value_of_k_l5_5221

theorem value_of_k (k : ℝ) :
  ∃ (k : ℝ), k ≠ 1 ∧ (k-1) * (0 : ℝ)^2 + 6 * (0 : ℝ) + k^2 - 1 = 0 ∧ k = -1 :=
by
  sorry

end value_of_k_l5_5221


namespace length_AE_l5_5405

theorem length_AE (AB CD AC AE ratio : ℝ) 
  (h_AB : AB = 10) 
  (h_CD : CD = 15) 
  (h_AC : AC = 18) 
  (h_ratio : ratio = 2 / 3) 
  (h_areas : ∀ (areas : ℝ), areas = 2 / 3)
  : AE = 7.2 := 
sorry

end length_AE_l5_5405


namespace seq_value_at_2018_l5_5552

noncomputable def f (x : ℝ) : ℝ := sorry
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = f 0 ∧ ∀ (n : ℕ), n > 0 → f (a (n + 1)) = 1 / f (-2 - a n)

theorem seq_value_at_2018 (a : ℕ → ℝ) (h_seq : seq a) : a 2018 = 4035 := 
by sorry

end seq_value_at_2018_l5_5552


namespace multiplication_subtraction_difference_l5_5964

theorem multiplication_subtraction_difference (x n : ℕ) (h₁ : x = 5) (h₂ : 3 * x = (16 - x) + n) : n = 4 :=
by
  -- Proof will go here
  sorry

end multiplication_subtraction_difference_l5_5964


namespace directrix_parabola_l5_5284

theorem directrix_parabola (x y : ℝ) (h : x^2 = 8 * y) : y = -2 :=
sorry

end directrix_parabola_l5_5284


namespace joan_balloons_l5_5708

def initial_balloons : ℕ := 72
def additional_balloons : ℕ := 23
def total_balloons : ℕ := initial_balloons + additional_balloons

theorem joan_balloons : total_balloons = 95 := by
  sorry

end joan_balloons_l5_5708


namespace building_shadow_length_l5_5774

theorem building_shadow_length
  (flagpole_height : ℝ) (flagpole_shadow : ℝ) (building_height : ℝ)
  (h_flagpole : flagpole_height = 18) (s_flagpole : flagpole_shadow = 45) 
  (h_building : building_height = 26) :
  ∃ (building_shadow : ℝ), (building_height / building_shadow = flagpole_height / flagpole_shadow) ∧ building_shadow = 65 :=
by
  use 65
  sorry

end building_shadow_length_l5_5774


namespace find_f_5_l5_5560

section
variables (f : ℝ → ℝ)

-- Given condition
def functional_equation (x : ℝ) : Prop := x * f x = 2 * f (1 - x) + 1

-- Prove that f(5) = 1/12 given the condition
theorem find_f_5 (h : ∀ x, functional_equation f x) : f 5 = 1 / 12 :=
sorry
end

end find_f_5_l5_5560


namespace two_sin_cos_15_eq_half_l5_5005

open Real

theorem two_sin_cos_15_eq_half : 2 * sin (π / 12) * cos (π / 12) = 1 / 2 :=
by
  sorry

end two_sin_cos_15_eq_half_l5_5005


namespace find_A_l5_5133

theorem find_A (A : ℕ) (h : 59 = (A * 6) + 5) : A = 9 :=
by sorry

end find_A_l5_5133


namespace minimum_voters_for_tall_victory_l5_5099

-- Definitions for conditions
def total_voters : ℕ := 105
def districts : ℕ := 5
def sections_per_district : ℕ := 7
def voters_per_section : ℕ := 3

-- Define majority function
def majority (n : ℕ) : ℕ := n / 2 + 1

-- Express conditions in Lean
def voters_per_district : ℕ := total_voters / districts
def sections_to_win_district : ℕ := majority sections_per_district
def districts_to_win_contest : ℕ := majority districts

-- The main problem statement
theorem minimum_voters_for_tall_victory : ∃ (x : ℕ), x = 24 ∧
  (let sections_needed := sections_to_win_district * districts_to_win_contest in
   let voters_needed_per_section := majority voters_per_section in
   x = sections_needed * voters_needed_per_section) :=
by {
  let sections_needed := sections_to_win_district * districts_to_win_contest,
  let voters_needed_per_section := majority voters_per_section,
  use 24,
  split,
  { refl },
  { simp [sections_needed, voters_needed_per_section, sections_to_win_district, districts_to_win_contest, majority, voters_per_section] }
}

end minimum_voters_for_tall_victory_l5_5099


namespace minimum_voters_for_tall_l5_5100

-- Define the structure of the problem
def num_voters := 105
def num_districts := 5
def sections_per_district := 7
def voters_per_section := 3
def majority x := ⌊ x / 2 ⌋ + 1 

-- Define conditions
def wins_section (votes_for_tall : ℕ) : Prop := votes_for_tall ≥ majority voters_per_section
def wins_district (sections_won : ℕ) : Prop := sections_won ≥ majority sections_per_district
def wins_contest (districts_won : ℕ) : Prop := districts_won ≥ majority num_districts

-- Define the theorem statement
theorem minimum_voters_for_tall : 
  ∃ (votes_for_tall : ℕ), votes_for_tall = 24 ∧
  (∃ (district_count : ℕ → ℕ), 
    (∀ d, d < num_districts → wins_district (district_count d)) ∧
    wins_contest (∑ d in finset.range num_districts, wins_district (district_count d).count (λ w, w = tt))) := 
sorry

end minimum_voters_for_tall_l5_5100


namespace johns_running_hours_l5_5535

-- Define the conditions
variable (x : ℕ) -- let x represent the number of hours at 8 mph and 6 mph
variable (total_hours : ℕ) (total_distance : ℕ)
variable (speed_8 : ℕ) (speed_6 : ℕ) (speed_5 : ℕ)
variable (distance_8 : ℕ := speed_8 * x)
variable (distance_6 : ℕ := speed_6 * x)
variable (distance_5 : ℕ := speed_5 * (total_hours - 2 * x))

-- Total hours John completes the marathon
axiom h1: total_hours = 15

-- Total distance John completes in miles
axiom h2: total_distance = 95

-- Speed factors
axiom h3: speed_8 = 8
axiom h4: speed_6 = 6
axiom h5: speed_5 = 5

-- Distance equation
axiom h6: distance_8 + distance_6 + distance_5 = total_distance

-- Prove the number of hours John ran at each speed
theorem johns_running_hours : x = 5 :=
by
  sorry

end johns_running_hours_l5_5535


namespace jack_piggy_bank_after_8_weeks_l5_5449

-- Conditions as definitions
def initial_amount : ℕ := 43
def weekly_allowance : ℕ := 10
def saved_fraction (x : ℕ) : ℕ := x / 2
def duration : ℕ := 8

-- Mathematical equivalent proof problem
theorem jack_piggy_bank_after_8_weeks : initial_amount + (duration * saved_fraction weekly_allowance) = 83 := by
  sorry

end jack_piggy_bank_after_8_weeks_l5_5449


namespace thirtieth_term_of_arithmetic_seq_l5_5359

def arithmetic_seq (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem thirtieth_term_of_arithmetic_seq : 
  arithmetic_seq 3 4 30 = 119 := 
by
  sorry

end thirtieth_term_of_arithmetic_seq_l5_5359


namespace bromine_atoms_in_compound_l5_5639

theorem bromine_atoms_in_compound
  (atomic_weight_H : ℕ := 1)
  (atomic_weight_Br : ℕ := 80)
  (atomic_weight_O : ℕ := 16)
  (total_molecular_weight : ℕ := 129) :
  ∃ (n : ℕ), total_molecular_weight = atomic_weight_H + n * atomic_weight_Br + 3 * atomic_weight_O ∧ n = 1 := 
by
  sorry

end bromine_atoms_in_compound_l5_5639


namespace quadratic_has_distinct_real_roots_l5_5042

theorem quadratic_has_distinct_real_roots (q : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x^2 + 8 * x + q = 0) ↔ q < 16 :=
by
  -- only the statement is provided, the proof is omitted
  sorry

end quadratic_has_distinct_real_roots_l5_5042


namespace tan_angle_identity_l5_5809

open Real

theorem tan_angle_identity (α β : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h : sin β / cos β = (1 + cos (2 * α)) / (2 * cos α + sin (2 * α))) :
  tan (α + 2 * β + π / 4) = -1 := 
sorry

end tan_angle_identity_l5_5809


namespace optimal_rental_plan_l5_5026

theorem optimal_rental_plan (a b x y : ℕ)
  (h1 : 2 * a + b = 10)
  (h2 : a + 2 * b = 11)
  (h3 : 31 = 3 * x + 4 * y)
  (cost_a : ℕ := 100)
  (cost_b : ℕ := 120) :
  ∃ x y, 3 * x + 4 * y = 31 ∧ cost_a * x + cost_b * y = 940 := by
  sorry

end optimal_rental_plan_l5_5026


namespace roy_age_product_l5_5863

theorem roy_age_product (R J K : ℕ) 
  (h1 : R = J + 8)
  (h2 : R = K + (R - J) / 2)
  (h3 : R + 2 = 3 * (J + 2)) :
  (R + 2) * (K + 2) = 96 :=
by
  sorry

end roy_age_product_l5_5863


namespace infinite_011_divisible_by_2019_l5_5122

/-- There are infinitely many numbers composed only of the digits 0 and 1 in decimal form
  that are divisible by 2019. -/
theorem infinite_011_divisible_by_2019 :
  ∃ (f : ℕ → ℕ), (∀ n, ∀ k, f n = f (n + k)) → ∃ N, N % 2019 = 0 :=
sorry

end infinite_011_divisible_by_2019_l5_5122


namespace lily_spent_on_shirt_l5_5716

theorem lily_spent_on_shirt (S : ℝ) (initial_balance : ℝ) (final_balance : ℝ) : 
  initial_balance = 55 → 
  final_balance = 27 → 
  55 - S - 3 * S = 27 → 
  S = 7 := 
by
  intros h1 h2 h3
  sorry

end lily_spent_on_shirt_l5_5716


namespace cosine_difference_l5_5234

theorem cosine_difference (A B : ℝ) (h1 : Real.sin A + Real.sin B = 3/2) (h2 : Real.cos A + Real.cos B = 2) :
  Real.cos (A - B) = 17 / 8 :=
by
  sorry

end cosine_difference_l5_5234


namespace sum_of_first_11_terms_l5_5881

theorem sum_of_first_11_terms (a1 d : ℝ) (h : 2 * a1 + 10 * d = 8) : 
  (11 / 2) * (2 * a1 + 10 * d) = 44 := 
by sorry

end sum_of_first_11_terms_l5_5881


namespace minimum_total_number_of_balls_l5_5372

theorem minimum_total_number_of_balls (x y z t : ℕ) 
  (h1 : x ≥ 4)
  (h2 : x ≥ 3 ∧ y ≥ 1)
  (h3 : x ≥ 2 ∧ y ≥ 1 ∧ z ≥ 1)
  (h4 : x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ t ≥ 1) :
  x + y + z + t = 21 :=
  sorry

end minimum_total_number_of_balls_l5_5372


namespace not_exists_implies_bounds_l5_5231

variable (a : ℝ)

/-- If there does not exist an x such that x^2 + (a - 1) * x + 1 < 0, then -1 ≤ a ∧ a ≤ 3. -/
theorem not_exists_implies_bounds : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → (-1 ≤ a ∧ a ≤ 3) :=
by sorry

end not_exists_implies_bounds_l5_5231


namespace remainder_7_pow_4_div_100_l5_5341

theorem remainder_7_pow_4_div_100 : (7 ^ 4) % 100 = 1 := 
by
  sorry

end remainder_7_pow_4_div_100_l5_5341


namespace sticks_left_in_yard_l5_5530

def number_of_sticks_picked_up : Nat := 14
def difference_between_picked_and_left : Nat := 10

theorem sticks_left_in_yard 
  (picked_up : Nat := number_of_sticks_picked_up)
  (difference : Nat := difference_between_picked_and_left) 
  : Nat :=
  picked_up - difference

example : sticks_left_in_yard = 4 := by 
  sorry

end sticks_left_in_yard_l5_5530


namespace minimum_voters_for_tall_victory_l5_5097

-- Definitions for conditions
def total_voters : ℕ := 105
def districts : ℕ := 5
def sections_per_district : ℕ := 7
def voters_per_section : ℕ := 3

-- Define majority function
def majority (n : ℕ) : ℕ := n / 2 + 1

-- Express conditions in Lean
def voters_per_district : ℕ := total_voters / districts
def sections_to_win_district : ℕ := majority sections_per_district
def districts_to_win_contest : ℕ := majority districts

-- The main problem statement
theorem minimum_voters_for_tall_victory : ∃ (x : ℕ), x = 24 ∧
  (let sections_needed := sections_to_win_district * districts_to_win_contest in
   let voters_needed_per_section := majority voters_per_section in
   x = sections_needed * voters_needed_per_section) :=
by {
  let sections_needed := sections_to_win_district * districts_to_win_contest,
  let voters_needed_per_section := majority voters_per_section,
  use 24,
  split,
  { refl },
  { simp [sections_needed, voters_needed_per_section, sections_to_win_district, districts_to_win_contest, majority, voters_per_section] }
}

end minimum_voters_for_tall_victory_l5_5097


namespace thirtieth_term_value_l5_5349

-- Define the initial term and common difference
def initial_term : ℤ := 3
def common_difference : ℤ := 4

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ :=
  initial_term + (n - 1) * common_difference

-- Theorem stating the value of the 30th term
theorem thirtieth_term_value : nth_term 30 = 119 :=
by sorry

end thirtieth_term_value_l5_5349


namespace arithmetic_sequence_l5_5049

noncomputable def M (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.sum (Finset.range n) (λ i => a (i + 1))) / n

theorem arithmetic_sequence (a : ℕ → ℝ) (C : ℝ)
  (h : ∀ {i j k : ℕ}, i ≠ j → j ≠ k → k ≠ i →
    (i - j) * M a k + (j - k) * M a i + (k - i) * M a j = C) :
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a 1 + n * d :=
sorry

end arithmetic_sequence_l5_5049


namespace students_not_enrolled_in_any_classes_l5_5025

/--
  At a particular college, 27.5% of the 1050 students are enrolled in biology,
  32.9% of the students are enrolled in mathematics, and 15% of the students are enrolled in literature classes.
  Assuming that no student is taking more than one of these specific subjects,
  the number of students at the college who are not enrolled in biology, mathematics, or literature classes is 260.

  We want to prove the statement:
    number_students_not_enrolled_in_any_classes = 260
-/
theorem students_not_enrolled_in_any_classes 
  (total_students : ℕ) 
  (biology_percent : ℝ) 
  (mathematics_percent : ℝ) 
  (literature_percent : ℝ) 
  (no_student_in_multiple : Prop) : 
  total_students = 1050 →
  biology_percent = 27.5 →
  mathematics_percent = 32.9 →
  literature_percent = 15 →
  (total_students - (⌊biology_percent / 100 * total_students⌋ + ⌊mathematics_percent / 100 * total_students⌋ + ⌊literature_percent / 100 * total_students⌋)) = 260 :=
by {
  sorry
}

end students_not_enrolled_in_any_classes_l5_5025


namespace exists_same_distance_l5_5295

open Finset

theorem exists_same_distance (men_wives : Finset (ℕ × ℕ)) (h_size : men_wives.card = 30)
  (h_range : ∀ (m w : ℕ), (m, w) ∈ men_wives → (1 ≤ m ∧ m ≤ 60) ∧ (1 ≤ w ∧ w ≤ 60)) :
  ∃ (m1 m2 w1 w2 : ℕ), (m1, w1) ∈ men_wives ∧ (m2, w2) ∈ men_wives ∧ m1 ≠ m2 ∧ 
  (min (abs (m1 - w1)) (60 - abs (m1 - w1)) = min (abs (m2 - w2)) (60 - abs (m2 - w2))) :=
by 
  sorry

end exists_same_distance_l5_5295


namespace target_hit_probability_l5_5132

/-- 
The probabilities for two shooters to hit a target are 1/2 and 1/3, respectively.
If both shooters fire at the target simultaneously, the probability that the target 
will be hit is 2/3.
-/
theorem target_hit_probability (P₁ P₂ : ℚ) (h₁ : P₁ = 1/2) (h₂ : P₂ = 1/3) :
  1 - ((1 - P₁) * (1 - P₂)) = 2/3 :=
by
  sorry

end target_hit_probability_l5_5132


namespace standard_equation_of_circle_l5_5014

/-- A circle with radius 2, center in the fourth quadrant, and tangent to the lines x = 0 and x + y = 2√2 has the standard equation (x - 2)^2 + (y + 2)^2 = 4. -/
theorem standard_equation_of_circle :
  ∃ a, a > 0 ∧ (∀ x y : ℝ, ((x - a)^2 + (y + 2)^2 = 4) ∧ 
                        (a > 0) ∧ 
                        (x = 0 → a = 2) ∧
                        x + y = 2 * Real.sqrt 2 → a = 2) := 
by
  sorry

end standard_equation_of_circle_l5_5014


namespace remainder_of_sum_of_primes_mod_eighth_prime_l5_5161

def sum_first_seven_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13 + 17

def eighth_prime : ℕ := 19

theorem remainder_of_sum_of_primes_mod_eighth_prime : sum_first_seven_primes % eighth_prime = 1 := by
  sorry

end remainder_of_sum_of_primes_mod_eighth_prime_l5_5161


namespace age_difference_l5_5118

theorem age_difference (Rona Rachel Collete : ℕ) (h1 : Rachel = 2 * Rona) (h2 : Collete = Rona / 2) (h3 : Rona = 8) : Rachel - Collete = 12 :=
by
  sorry

end age_difference_l5_5118


namespace proof_problem_l5_5825

variables (x y b z a : ℝ)

def condition1 : Prop := x * y + x^2 = b
def condition2 : Prop := (1 / x^2) - (1 / y^2) = a
def z_def : Prop := z = x + y

theorem proof_problem (x y b z a : ℝ) (h1 : condition1 x y b) (h2 : condition2 x y a) (hz : z_def x y z) : (x + y) ^ 2 = z ^ 2 :=
by {
  sorry
}

end proof_problem_l5_5825


namespace problem_solution_l5_5762

theorem problem_solution :
  (315^2 - 291^2) / 24 = 606 :=
by
  sorry

end problem_solution_l5_5762


namespace packet_weight_l5_5196

theorem packet_weight :
  ∀ (num_packets : ℕ) (total_weight_kg : ℕ), 
  num_packets = 20 → total_weight_kg = 2 →
  (total_weight_kg * 1000) / num_packets = 100 := by
  intro num_packets total_weight_kg h1 h2
  sorry

end packet_weight_l5_5196


namespace domain_of_w_l5_5037

theorem domain_of_w :
  {x : ℝ | x + (x - 1)^(1/3) + (8 - x)^(1/3) ≥ 0} = {x : ℝ | x ≥ 0} :=
by {
  sorry
}

end domain_of_w_l5_5037


namespace area_of_paper_is_500_l5_5894

-- Define the width and length of the rectangular drawing paper
def width := 25
def length := 20

-- Define the formula for the area of a rectangle
def area (w : Nat) (l : Nat) : Nat := w * l

-- Prove that the area of the paper is 500 square centimeters
theorem area_of_paper_is_500 : area width length = 500 := by
  -- placeholder for the proof
  sorry

end area_of_paper_is_500_l5_5894


namespace inverse_modulo_1000000_l5_5844

def A : ℕ := 123456
def B : ℕ := 769230
def N : ℕ := 1053

theorem inverse_modulo_1000000 : (A * B * N) % 1000000 = 1 := 
  by 
  sorry

end inverse_modulo_1000000_l5_5844


namespace choir_robe_costs_l5_5380

theorem choir_robe_costs:
  ∀ (total_robes needed_robes total_cost robe_cost : ℕ),
  total_robes = 30 →
  needed_robes = 30 - 12 →
  total_cost = 36 →
  total_cost = needed_robes * robe_cost →
  robe_cost = 2 :=
by
  intros total_robes needed_robes total_cost robe_cost
  intro h_total_robes h_needed_robes h_total_cost h_cost_eq
  sorry

end choir_robe_costs_l5_5380


namespace decagon_adjacent_vertex_probability_l5_5322

theorem decagon_adjacent_vertex_probability :
  let vertices := 10 in
  let total_combinations := Nat.choose vertices 2 in
  let adjacent_pairs := vertices * 2 in
  (adjacent_pairs : ℚ) / total_combinations = 4 / 9 :=
by
  let vertices := 10
  let total_combinations := Nat.choose vertices 2
  let adjacent_pairs := vertices * 2
  have : (adjacent_pairs : ℚ) / total_combinations = 4 / 9 := sorry
  exact this

end decagon_adjacent_vertex_probability_l5_5322


namespace opposite_sides_of_line_l5_5480

theorem opposite_sides_of_line (a : ℝ) (h1 : 0 < a) (h2 : a < 2) : (-a) * (2 - a) < 0 :=
sorry

end opposite_sides_of_line_l5_5480


namespace total_votes_l5_5507

variable (V : ℝ)

theorem total_votes (h1 : 0.34 * V + 640 = 0.66 * V) : V = 2000 :=
by 
  sorry

end total_votes_l5_5507


namespace ratio_of_larger_to_smaller_l5_5137

theorem ratio_of_larger_to_smaller (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : a + b = 5 * (a - b)) :
  a / b = 3 / 2 := by
sorry

end ratio_of_larger_to_smaller_l5_5137


namespace min_time_to_cross_river_l5_5784

-- Definitions for the time it takes each horse to cross the river
def timeA : ℕ := 2
def timeB : ℕ := 3
def timeC : ℕ := 7
def timeD : ℕ := 6

-- Definition for the minimum time required for all horses to cross the river
def min_crossing_time : ℕ := 18

-- Theorem stating the problem: 
theorem min_time_to_cross_river :
  ∀ (timeA timeB timeC timeD : ℕ), timeA = 2 → timeB = 3 → timeC = 7 → timeD = 6 →
  min_crossing_time = 18 :=
sorry

end min_time_to_cross_river_l5_5784


namespace P_at_2007_l5_5687

noncomputable def P (x : ℝ) : ℝ :=
x^15 - 2008 * x^14 + 2008 * x^13 - 2008 * x^12 + 2008 * x^11
- 2008 * x^10 + 2008 * x^9 - 2008 * x^8 + 2008 * x^7
- 2008 * x^6 + 2008 * x^5 - 2008 * x^4 + 2008 * x^3
- 2008 * x^2 + 2008 * x

-- Statement to show that P(2007) = 2007
theorem P_at_2007 : P 2007 = 2007 :=
  sorry

end P_at_2007_l5_5687


namespace probability_of_adjacent_vertices_l5_5315

def decagon_vertices : ℕ := 10

def total_ways_to_choose_2_vertices (n : ℕ) : ℕ := n * (n - 1) / 2

def favorable_ways_to_choose_adjacent_vertices (n : ℕ) : ℕ := 2 * n

theorem probability_of_adjacent_vertices :
  let n := decagon_vertices in
  let total_choices := total_ways_to_choose_2_vertices n in
  let favorable_choices := favorable_ways_to_choose_adjacent_vertices n in
  (favorable_choices : ℚ) / total_choices = 2 / 9 :=
by
  sorry

end probability_of_adjacent_vertices_l5_5315


namespace light_flash_fraction_l5_5641

def light_flash_fraction_of_hour (n : ℕ) (t : ℕ) (flashes : ℕ) := 
  (n * t) / (60 * 60)

theorem light_flash_fraction (n : ℕ) (t : ℕ) (flashes : ℕ) (h1 : t = 12) (h2 : flashes = 300) : 
  light_flash_fraction_of_hour n t flashes = 1 := 
by
  sorry

end light_flash_fraction_l5_5641


namespace prime_sum_mod_eighth_l5_5171

theorem prime_sum_mod_eighth (p1 p2 p3 p4 p5 p6 p7 p8 : ℕ) 
  (h₁ : p1 = 2) 
  (h₂ : p2 = 3) 
  (h₃ : p3 = 5) 
  (h₄ : p4 = 7) 
  (h₅ : p5 = 11) 
  (h₆ : p6 = 13) 
  (h₇ : p7 = 17) 
  (h₈ : p8 = 19) : 
  ((p1 + p2 + p3 + p4 + p5 + p6 + p7) % p8) = 1 :=
by
  sorry

end prime_sum_mod_eighth_l5_5171


namespace determine_x_l5_5414

theorem determine_x : ∃ (x : ℕ), 
  (3 * x > 91 ∧ x < 120 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ ¬(x > 7) ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (3 * x > 91 ∧ x < 120 ∧ 4 * x > 37 ∧ ¬(2 * x ≥ 21) ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (3 * x > 91 ∧ x < 120 ∧ ¬(4 * x > 37) ∧ 2 * x ≥ 21 ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (3 * x > 91 ∧ ¬(x < 120) ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∧
  x = 10 :=
sorry

end determine_x_l5_5414


namespace valid_programs_count_l5_5645

open Finset

theorem valid_programs_count :
  let courses := { "English", "Algebra", "Geometry", "History", "Art", "Latin", "Science" }
  let math_courses := { "Algebra", "Geometry" }
  let required_courses := { "English" }
  (finset.card (courses \ required_courses) = 6) →
  (choose 6 4 = 15) →
  let total_programs := 15
  let invalid_programs :=
    - 2 * choose 4 3
  let result := total_programs + invalid_programs
  result = 7 :=
by
  sorry

end valid_programs_count_l5_5645


namespace sum_inf_series_l5_5399

theorem sum_inf_series :
  (\sum_{n=1}^{\infty} \frac{(4 * n) - 3}{3^n}) = 1 :=
by
  sorry

end sum_inf_series_l5_5399


namespace number_of_chlorine_atoms_l5_5412

def molecular_weight_of_aluminum : ℝ := 26.98
def molecular_weight_of_chlorine : ℝ := 35.45
def molecular_weight_of_compound : ℝ := 132.0

theorem number_of_chlorine_atoms :
  ∃ n : ℕ, molecular_weight_of_compound = molecular_weight_of_aluminum + n * molecular_weight_of_chlorine ∧ n = 3 :=
by
  sorry

end number_of_chlorine_atoms_l5_5412


namespace correct_conclusion_l5_5965

theorem correct_conclusion :
  ¬ (-(-3)^2 = 9) ∧
  ¬ (-6 / 6 * (1 / 6) = -6) ∧
  ((-3)^2 * abs (-1/3) = 3) ∧
  ¬ (3^2 / 2 = 9 / 4) :=
by
  sorry

end correct_conclusion_l5_5965


namespace polynomial_value_l5_5594

def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_value (a b c d : ℝ) 
  (h1 : P 1 a b c d = 1993) 
  (h2 : P 2 a b c d = 3986) 
  (h3 : P 3 a b c d = 5979) :
  (1 / 4) * (P 11 a b c d + P (-7) a b c d) = 4693 := 
by 
  sorry

end polynomial_value_l5_5594


namespace number_of_three_digit_numbers_value_of_89th_item_l5_5549

open List

def digits : Finset ℕ := {1, 2, 3, 4, 5, 6}

def three_digit_numbers : Finset (List ℕ) :=
  Finset.univ.binder (λ _ ∈ digits.to_list.permutations.filter (λ l, l.length = 3))

/-- Prove the number of three-digit numbers using the digits 1, 2, 3, 4, 5, and 6 without repetition is 120. -/
theorem number_of_three_digit_numbers : (three_digit_numbers.card = 120) :=
sorry

/-- Prove the value of the 89th item in the ascending sequence of three-digit numbers is 526. -/
theorem value_of_89th_item : 
  (three_digit_numbers.to_list.qsort (≤) !! 88 = [5, 2, 6]) :=
sorry

end number_of_three_digit_numbers_value_of_89th_item_l5_5549


namespace water_leakage_l5_5587

theorem water_leakage (initial_quarts : ℚ) (remaining_gallons : ℚ)
  (conversion_rate : ℚ) (expected_leakage : ℚ) :
  initial_quarts = 4 ∧ remaining_gallons = 0.33 ∧ conversion_rate = 4 ∧ 
  expected_leakage = 2.68 →
  initial_quarts - remaining_gallons * conversion_rate = expected_leakage :=
by 
  sorry

end water_leakage_l5_5587


namespace students_in_classes_saved_money_strategy_class7_1_l5_5892

-- Part (1): Prove the number of students in each class
theorem students_in_classes (x : ℕ) (h1 : 40 < x) (h2 : x < 50) 
  (h3 : 105 - x > 50) (h4 : 15 * x + 12 * (105 - x) = 1401) : x = 47 ∧ (105 - x) = 58 := by
  sorry

-- Part (2): Prove the amount saved by purchasing tickets together
theorem saved_money(amt_per_ticket : ℕ → ℕ) 
  (h1 : amt_per_ticket 105 = 1401) 
  (h2 : ∀n, n > 100 → amt_per_ticket n = 1050) : amt_per_ticket 105 - 1050 = 351 := by
  sorry

-- Part (3): Strategy to save money for class 7 (1)
theorem strategy_class7_1 (students_1 : ℕ) (h1 : students_1 = 47) 
  (cost_15 : students_1 * 15 = 705) 
  (cost_51 : 51 * 12 = 612) : 705 - 612 = 93 := by
  sorry

end students_in_classes_saved_money_strategy_class7_1_l5_5892


namespace min_voters_for_Tall_victory_l5_5088

def total_voters := 105
def districts := 5
def sections_per_district := 7
def voters_per_section := 3
def sections_to_win_district := 4
def districts_to_win := 3
def sections_to_win := sections_to_win_district * districts_to_win
def min_voters_to_win_section := 2

theorem min_voters_for_Tall_victory : 
  (total_voters = 105) ∧ 
  (districts = 5) ∧ 
  (sections_per_district = 7) ∧ 
  (voters_per_section = 3) ∧ 
  (sections_to_win_district = 4) ∧ 
  (districts_to_win = 3) 
  → 
  min_voters_to_win_section * sections_to_win = 24 :=
by
  sorry
  
end min_voters_for_Tall_victory_l5_5088


namespace skillful_hands_award_prob_cannot_enter_finals_after_training_l5_5262

noncomputable def combinatorial_probability : ℚ :=
  let P1 := (4 * 3) / (10 * 10)    -- P1: 1 specified, 2 creative
  let P2 := (6 * 3) / (10 * 10)    -- P2: 2 specified, 1 creative
  let P3 := (6 * 3) / (10 * 10)    -- P3: 2 specified, 2 creative
  P1 + P2 + P3

theorem skillful_hands_award_prob : combinatorial_probability = 33 / 50 := 
  sorry

def after_training_probability := 3 / 4
theorem cannot_enter_finals_after_training : after_training_probability * 5 < 4 := 
  sorry

end skillful_hands_award_prob_cannot_enter_finals_after_training_l5_5262


namespace min_value_f_solution_set_exists_x_f_eq_0_l5_5424

def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 - 2*x else (-(x))^2 - 2*(-x)

theorem min_value_f : ∃ x : ℝ, f(x) = -1 := 
by sorry

theorem solution_set : {x : ℝ | f(x) > 0} = {x : ℝ | x < -2 ∨ x > 2} :=
by sorry

theorem exists_x_f_eq_0 : ∃ x : ℝ, f(x + 2) + f(-x) = 0 :=
by sorry

end min_value_f_solution_set_exists_x_f_eq_0_l5_5424


namespace find_a_if_lines_parallel_l5_5556

theorem find_a_if_lines_parallel (a : ℝ) (h1 : ∃ y : ℝ, y = - (a / 4) * (1 : ℝ) + (1 / 4)) (h2 : ∃ y : ℝ, y = - (1 / a) * (1 : ℝ) + (1 / (2 * a))) : a = -2 :=
sorry

end find_a_if_lines_parallel_l5_5556


namespace parallelogram_area_l5_5036

theorem parallelogram_area (base height : ℝ) (h_base : base = 10) (h_height : height = 20) :
  base * height = 200 := 
by 
  sorry

end parallelogram_area_l5_5036


namespace spending_difference_is_65_l5_5838

-- Definitions based on conditions
def ice_cream_cones : ℕ := 15
def pudding_cups : ℕ := 5
def ice_cream_cost_per_unit : ℝ := 5
def pudding_cost_per_unit : ℝ := 2

-- The solution requires the calculation of the total cost and the difference
def total_ice_cream_cost : ℝ := ice_cream_cones * ice_cream_cost_per_unit
def total_pudding_cost : ℝ := pudding_cups * pudding_cost_per_unit
def spending_difference : ℝ := total_ice_cream_cost - total_pudding_cost

-- Theorem statement proving the difference is 65
theorem spending_difference_is_65 : spending_difference = 65 := by
  -- The proof is omitted with sorry
  sorry

end spending_difference_is_65_l5_5838


namespace ferry_speeds_l5_5223

theorem ferry_speeds (v_P v_Q : ℝ) 
  (h1: v_P = v_Q - 1) 
  (h2: 3 * v_P * 3 = v_Q * (3 + 5))
  : v_P = 8 := 
sorry

end ferry_speeds_l5_5223


namespace slope_of_parallel_line_l5_5959

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l5_5959


namespace correlation_relationships_l5_5385

-- Let's define the relationships as conditions
def volume_cube_edge_length (v e : ℝ) : Prop := v = e^3
def yield_fertilizer (yield fertilizer : ℝ) : Prop := True -- Assume linear correlation within a certain range
def height_age (height age : ℝ) : Prop := True -- Assume linear correlation within a certain age range
def expenses_income (expenses income : ℝ) : Prop := True -- Assume linear correlation
def electricity_consumption_price (consumption price unit_price : ℝ) : Prop := price = consumption * unit_price

-- We want to prove that the answers correspond correctly to the conditions:
theorem correlation_relationships :
  ∀ (v e yield fertilizer height age expenses income consumption price unit_price : ℝ),
  ¬ volume_cube_edge_length v e ∧ yield_fertilizer yield fertilizer ∧ height_age height age ∧ expenses_income expenses income ∧ ¬ electricity_consumption_price consumption price unit_price → 
  "D" = "②③④" :=
by
  intros
  sorry

end correlation_relationships_l5_5385


namespace parallel_line_slope_l5_5918

def slope_of_parallel_line : ℚ :=
  let line_eq := (3 : ℚ) * (x : ℚ) - (6 : ℚ) * (y : ℚ) = (12 : ℚ)
  in (1 / 2 : ℚ)

theorem parallel_line_slope (x y : ℚ) (h : 3 * x - 6 * y = 12) : slope_of_parallel_line = 1 / 2 :=
by
  sorry

end parallel_line_slope_l5_5918


namespace find_number_l5_5149

theorem find_number :
  (∃ m : ℝ, 56 = (3 / 2) * m) ∧ (56 = 0.7 * 80) → m = 37 := by
  sorry

end find_number_l5_5149


namespace find_m_when_lines_parallel_l5_5064

theorem find_m_when_lines_parallel (m : ℝ) :
  (∀ x y : ℝ, x + (1 + m) * y = 2 - m) ∧ (∀ x y : ℝ, 2 * m * x + 4 * y = -16) →
  ∃ m : ℝ, m = 1 :=
sorry

end find_m_when_lines_parallel_l5_5064


namespace smallest_k_for_720_l5_5547

/-- Given a number 720, prove that the smallest positive integer k such that 720 * k is both a perfect square and a perfect cube is 1012500. -/
theorem smallest_k_for_720 (k : ℕ) : (∃ k > 0, 720 * k = (n : ℕ) ^ 6) -> k = 1012500 :=
by sorry

end smallest_k_for_720_l5_5547


namespace batsman_average_after_20th_innings_l5_5505

theorem batsman_average_after_20th_innings 
    (score_20th_innings : ℕ)
    (previous_avg_increase : ℕ)
    (total_innings : ℕ)
    (never_not_out : Prop)
    (previous_avg : ℕ)
    : score_20th_innings = 90 →
      previous_avg_increase = 2 →
      total_innings = 20 →
      previous_avg = (19 * previous_avg + score_20th_innings) / total_innings →
      ((19 * previous_avg + score_20th_innings) / total_innings) + previous_avg_increase = 52 :=
by 
  sorry

end batsman_average_after_20th_innings_l5_5505


namespace arithmetic_sequence_ninth_term_l5_5887

theorem arithmetic_sequence_ninth_term
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29) :
  a + 8 * d = 35 :=
sorry

end arithmetic_sequence_ninth_term_l5_5887


namespace sufficient_but_not_necessary_condition_l5_5060

def parabola (y : ℝ) : ℝ := y^2
def line (m : ℝ) (y : ℝ) : ℝ := m * y + 1

theorem sufficient_but_not_necessary_condition {m : ℝ} :
  (m ≠ 0) → ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ parabola y1 = line m y1 ∧ parabola y2 = line m y2 :=
by
  sorry

end sufficient_but_not_necessary_condition_l5_5060


namespace exists_good_number_in_interval_l5_5529

def is_good_number (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≤ 5

theorem exists_good_number_in_interval (x : ℕ) (hx : x ≠ 0) :
  ∃ g : ℕ, is_good_number g ∧ x ≤ g ∧ g < ((9 * x) / 5) + 1 := 
sorry

end exists_good_number_in_interval_l5_5529


namespace sum_primes_reversed_l5_5963

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def reverse_digits (n : ℕ) : ℕ := 
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

def valid_primes : List ℕ := [31, 37, 71, 73]

theorem sum_primes_reversed :
  (∀ p ∈ valid_primes, 20 < p ∧ p < 80 ∧ is_prime p ∧ is_prime (reverse_digits p)) ∧
  (valid_primes.sum = 212) :=
by
  sorry

end sum_primes_reversed_l5_5963


namespace beaker_volume_l5_5409

theorem beaker_volume {a b c d e f g h i j : ℝ} (h₁ : a = 7) (h₂ : b = 4) (h₃ : c = 5)
                      (h₄ : d = 4) (h₅ : e = 6) (h₆ : f = 8) (h₇ : g = 7)
                      (h₈ : h = 3) (h₉ : i = 9) (h₁₀ : j = 6) :
  (a + b + c + d + e + f + g + h + i + j) / 5 = 11.8 :=
by
  sorry

end beaker_volume_l5_5409


namespace sum_of_squares_not_divisible_by_17_l5_5617

theorem sum_of_squares_not_divisible_by_17
  (x y z : ℤ)
  (h_sum_div : 17 ∣ (x + y + z))
  (h_prod_div : 17 ∣ (x * y * z))
  (h_coprime_xy : Int.gcd x y = 1)
  (h_coprime_yz : Int.gcd y z = 1)
  (h_coprime_zx : Int.gcd z x = 1) :
  ¬ (17 ∣ (x^2 + y^2 + z^2)) := 
sorry

end sum_of_squares_not_divisible_by_17_l5_5617


namespace find_pairs_of_real_numbers_l5_5662

theorem find_pairs_of_real_numbers (x y : ℝ) :
  (∀ n : ℕ, n > 0 → x * ⌊n * y⌋ = y * ⌊n * x⌋) →
  (x = y ∨ x = 0 ∨ y = 0 ∨ (∃ a b : ℤ, x = a ∧ y = b)) :=
by
  sorry

end find_pairs_of_real_numbers_l5_5662


namespace kat_boxing_training_hours_l5_5710

theorem kat_boxing_training_hours :
  let strength_training_hours := 3
  let total_training_hours := 9
  let boxing_sessions := 4
  let boxing_training_hours := total_training_hours - strength_training_hours
  let hours_per_boxing_session := boxing_training_hours / boxing_sessions
  hours_per_boxing_session = 1.5 :=
sorry

end kat_boxing_training_hours_l5_5710


namespace find_y_l5_5473

theorem find_y (t : ℝ) (x : ℝ) (y : ℝ)
  (hx : x = 3 - 2 * t)
  (hy : y = 3 * t + 6)
  (hx_cond : x = -6) :
  y = 19.5 :=
by
  sorry

end find_y_l5_5473


namespace probability_of_adjacent_vertices_l5_5316

def decagon_vertices : ℕ := 10

def total_ways_to_choose_2_vertices (n : ℕ) : ℕ := n * (n - 1) / 2

def favorable_ways_to_choose_adjacent_vertices (n : ℕ) : ℕ := 2 * n

theorem probability_of_adjacent_vertices :
  let n := decagon_vertices in
  let total_choices := total_ways_to_choose_2_vertices n in
  let favorable_choices := favorable_ways_to_choose_adjacent_vertices n in
  (favorable_choices : ℚ) / total_choices = 2 / 9 :=
by
  sorry

end probability_of_adjacent_vertices_l5_5316


namespace billed_minutes_l5_5805

noncomputable def John_bill (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) : ℝ :=
  (total_bill - monthly_fee) / cost_per_minute

theorem billed_minutes : ∀ (monthly_fee cost_per_minute total_bill : ℝ), 
  monthly_fee = 5 → 
  cost_per_minute = 0.25 → 
  total_bill = 12.02 → 
  John_bill monthly_fee cost_per_minute total_bill = 28 :=
by
  intros monthly_fee cost_per_minute total_bill hf hm hb
  rw [hf, hm, hb, John_bill]
  norm_num
  sorry

end billed_minutes_l5_5805


namespace find_positive_integer_divisible_by_24_between_7_9_and_8_l5_5542

theorem find_positive_integer_divisible_by_24_between_7_9_and_8 :
  ∃ (n : ℕ), n > 0 ∧ (24 ∣ n) ∧ (7.9 < real.cbrt n) ∧ (real.cbrt n < 8) :=
begin
  use 504,
  split,
  { exact nat.zero_lt_succ 503, },
  split,
  { use 21, norm_num, },
  split,
  { norm_num, },
  { norm_num, },
end

end find_positive_integer_divisible_by_24_between_7_9_and_8_l5_5542


namespace decagon_adjacent_probability_l5_5312

-- Define the setup of a decagon with 10 vertices and the adjacency relation.
def is_decagon (n : ℕ) : Prop := n = 10

def adjacent_vertices (v1 v2 : ℕ) : Prop :=
  (v1 = (v2 + 1) % 10) ∨ (v1 = (v2 + 9) % 10)

-- The main theorem statement, proving the probability of two vertices being adjacent.
theorem decagon_adjacent_probability (n : ℕ) (v1 v2 : ℕ) (h : is_decagon n) : 
  ∀ (v1 v2 : ℕ), v1 ≠ v2 → v1 < n → v2 < n →
  (∃ (p : ℚ), p = 2 / 9 ∧ 
  (adjacent_vertices v1 v2) ↔ (p = 2 / (n - 1))) :=
sorry

end decagon_adjacent_probability_l5_5312


namespace solution_set_inequality_l5_5677

theorem solution_set_inequality (a x : ℝ) (h : 0 < a ∧ a < 1) : 
  ((a - x) * (x - (1 / a)) > 0) ↔ (a < x ∧ x < 1 / a) :=
by sorry

end solution_set_inequality_l5_5677


namespace calculate_length_QR_l5_5283

noncomputable def length_QR (A : ℝ) (h : ℝ) (PQ : ℝ) (RS : ℝ) : ℝ :=
  21 - 0.5 * (Real.sqrt (PQ ^ 2 - h ^ 2) + Real.sqrt (RS ^ 2 - h ^ 2))

theorem calculate_length_QR :
  length_QR 210 10 12 21 = 21 - 0.5 * (Real.sqrt 44 + Real.sqrt 341) :=
by
  sorry

end calculate_length_QR_l5_5283


namespace solve_quadratic_eq_l5_5608

theorem solve_quadratic_eq {x : ℝ} :
  (x = 3 ∨ x = -1) ↔ x^2 - 2 * x - 3 = 0 :=
by
  sorry

end solve_quadratic_eq_l5_5608


namespace paint_needed_l5_5664

-- Definitions from conditions
def total_needed_paint := 70
def initial_paint := 36
def bought_paint := 23

-- The main statement to prove
theorem paint_needed : total_needed_paint - (initial_paint + bought_ppaint) = 11 :=
by
  -- Definitions are already imported and stated
  -- Just need to refer these to the theorem assertion correctly
  sorry

end paint_needed_l5_5664


namespace john_total_amount_to_pay_l5_5451

-- Define constants for the problem
def total_cost : ℝ := 6650
def rebate_percentage : ℝ := 0.06
def sales_tax_percentage : ℝ := 0.10

-- The main theorem to prove the final amount John needs to pay
theorem john_total_amount_to_pay : total_cost * (1 - rebate_percentage) * (1 + sales_tax_percentage) = 6876.10 := by
  sorry    -- Proof skipped

end john_total_amount_to_pay_l5_5451


namespace vaccination_target_failure_l5_5440

noncomputable def percentage_vaccination_target_failed (original_target : ℕ) (first_year : ℕ) (second_year_increase_rate : ℚ) (third_year : ℕ) : ℚ :=
  let second_year := first_year + second_year_increase_rate * first_year
  let total_vaccinated := first_year + second_year + third_year
  let shortfall := original_target - total_vaccinated
  (shortfall / original_target) * 100

theorem vaccination_target_failure :
  percentage_vaccination_target_failed 720 60 (65/100 : ℚ) 150 = 57.11 := 
  by sorry

end vaccination_target_failure_l5_5440


namespace student_program_selection_l5_5197

-- Define the course selection problem within the given conditions
theorem student_program_selection :
  let courses := ["Algebra", "Geometry", "History", "Art", "Latin", "Science"]
  let math_courses := ["Algebra", "Geometry"]
  let choose_ways (n k : ℕ) := nat.choose n k
  
  -- Case with 2 math courses
  let case_2_math := choose_ways 2 2 * choose_ways 4 2 
  -- Case with 3 math courses
  let case_3_math := choose_ways 2 2 * choose_ways 4 1 
  -- Case with 4 math courses
  let case_4_math := choose_ways 2 2 * choose_ways 4 0 
  
  case_2_math + case_3_math + case_4_math = 11 :=
by
  sorry

end student_program_selection_l5_5197


namespace probability_adjacent_vertices_decagon_l5_5309

theorem probability_adjacent_vertices_decagon :
  let V := 10 in  -- Number of vertices in a decagon
  let favorable_outcomes := 2 in  -- Number of adjacent vertices to any chosen vertex
  let total_outcomes := V - 1 in  -- Total possible outcomes for the second vertex
  (favorable_outcomes / total_outcomes) = (2 / 9) :=
by
  let V := 10
  let favorable_outcomes := 2
  let total_outcomes := V - 1
  have probability := (favorable_outcomes / total_outcomes)
  have target_prob := (2 / 9)
  sorry

end probability_adjacent_vertices_decagon_l5_5309


namespace division_remainder_l5_5802

noncomputable def remainder (p q : Polynomial ℝ) : Polynomial ℝ :=
  p % q

theorem division_remainder :
  remainder (Polynomial.X ^ 3) (Polynomial.X ^ 2 + 7 * Polynomial.X + 2) = 47 * Polynomial.X + 14 :=
by
  sorry

end division_remainder_l5_5802


namespace slope_parallel_to_original_line_l5_5948

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l5_5948


namespace exact_one_shared_course_l5_5022

open Finset

-- Given conditions
def courses : Finset ℕ := {1, 2, 3, 4}

def choose_2_courses (course_set : Finset ℕ) : Finset (Finset ℕ) :=
  course_set.powerset.filter (λ s, s.card = 2)

-- Define the total number of ways to choose 2 courses for both A and B
def total_choices : ℕ :=
  (choose_2_courses courses).card * (choose_2_courses courses).card

-- Define the number of ways both chosen courses are the same
def same_courses : ℕ :=
  (choose_2_courses courses).card

-- Define the number of ways all chosen courses are different
def different_courses : ℕ :=
  (choose_2_courses courses).card

-- Define the problem statement
theorem exact_one_shared_course :
  total_choices - same_courses - different_courses = 24 := sorry

end exact_one_shared_course_l5_5022


namespace arithmetic_sequence_30th_term_l5_5353

theorem arithmetic_sequence_30th_term :
  let a₁ := 3
  let d := 4
  let n := 30
  a₁ + (n - 1) * d = 119 :=
by
  let a₁ := 3
  let d := 4
  let n := 30
  show a₁ + (n - 1) * d = 119
  sorry

end arithmetic_sequence_30th_term_l5_5353


namespace parallel_slope_l5_5953

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l5_5953


namespace product_of_a_and_b_l5_5827

variable (a b : ℕ)

-- Conditions
def LCM(a b : ℕ) : ℕ := Nat.lcm a b
def HCF(a b : ℕ) : ℕ := Nat.gcd a b

-- Assertion: product of a and b
theorem product_of_a_and_b (h_lcm: LCM a b = 72) (h_hcf: HCF a b = 6) : a * b = 432 := by
  sorry

end product_of_a_and_b_l5_5827


namespace min_value_z_l5_5576

variable {x y : ℝ}

def constraint1 (x y : ℝ) : Prop := x + y ≤ 3
def constraint2 (x y : ℝ) : Prop := x - y ≥ -1
def constraint3 (y : ℝ) : Prop := y ≥ 1

theorem min_value_z (x y : ℝ) 
  (h1 : constraint1 x y) 
  (h2 : constraint2 x y) 
  (h3 : constraint3 y) 
  (hx_pos : x > 0) 
  (hy_pos : y > 0) : 
  ∃ x y, x > 0 ∧ y ≥ 1 ∧ x + y ≤ 3 ∧ x - y ≥ -1 ∧ (∀ x' y', x' > 0 ∧ y' ≥ 1 ∧ x' + y' ≤ 3 ∧ x' - y' ≥ -1 → (y' / x' ≥ y / x)) ∧ y / x = 1 / 2 := 
sorry

end min_value_z_l5_5576


namespace rate_calculation_l5_5187

def principal : ℝ := 910
def simple_interest : ℝ := 260
def time : ℝ := 4
def rate : ℝ := 7.14

theorem rate_calculation :
  (simple_interest / (principal * time)) * 100 = rate :=
by
  sorry

end rate_calculation_l5_5187


namespace inequality_proof_l5_5666

theorem inequality_proof {x y z : ℝ}
  (h1 : x + 2 * y + 4 * z ≥ 3)
  (h2 : y - 3 * x + 2 * z ≥ 5) :
  y - x + 2 * z ≥ 3 :=
by
  sorry

end inequality_proof_l5_5666


namespace men_earnings_l5_5007

-- Definitions based on given problem conditions
variables (M rm W rw B rb X : ℝ)
variables (h1 : 5 > 0) (h2 : X > 0) (h3 : 8 > 0) -- positive quantities
variables (total_earnings : 5 * M * rm + X * W * rw + 8 * B * rb = 180)

-- The theorem we want to prove
theorem men_earnings (h1 : 5 > 0) (h2 : X > 0) (h3 : 8 > 0) (total_earnings : 5 * M * rm + X * W * rw + 8 * B * rb = 180) : 
  ∃ men_earnings : ℝ, men_earnings = 5 * M * rm :=
by 
  -- Proof is omitted
  exact Exists.intro (5 * M * rm) rfl

end men_earnings_l5_5007


namespace minimum_voters_for_tall_l5_5102

-- Define the structure of the problem
def num_voters := 105
def num_districts := 5
def sections_per_district := 7
def voters_per_section := 3
def majority x := ⌊ x / 2 ⌋ + 1 

-- Define conditions
def wins_section (votes_for_tall : ℕ) : Prop := votes_for_tall ≥ majority voters_per_section
def wins_district (sections_won : ℕ) : Prop := sections_won ≥ majority sections_per_district
def wins_contest (districts_won : ℕ) : Prop := districts_won ≥ majority num_districts

-- Define the theorem statement
theorem minimum_voters_for_tall : 
  ∃ (votes_for_tall : ℕ), votes_for_tall = 24 ∧
  (∃ (district_count : ℕ → ℕ), 
    (∀ d, d < num_districts → wins_district (district_count d)) ∧
    wins_contest (∑ d in finset.range num_districts, wins_district (district_count d).count (λ w, w = tt))) := 
sorry

end minimum_voters_for_tall_l5_5102


namespace parallel_slope_l5_5955

theorem parallel_slope (x y : ℝ) : (∃ b : ℝ, 3 * x - 6 * y = 12) → (∀ (x' y' : ℝ), (∃ b' : ℝ, 3 * x' - 6 * y' = b') → (∃ m : ℝ, m = 1 / 2)) :=
by
  sorry

end parallel_slope_l5_5955


namespace translate_point_correct_l5_5487

-- Define initial point
def initial_point : ℝ × ℝ := (0, 1)

-- Define translation downward
def translate_down (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1, point.2 - units)

-- Define translation to the left
def translate_left (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1 - units, point.2)

-- Define the expected resulting point
def expected_point : ℝ × ℝ := (-4, -1)

-- Lean statement to prove the equivalence
theorem translate_point_correct :
  (translate_left (translate_down initial_point 2) 4) = expected_point :=
by 
  -- Here, we would prove it step by step if required
  sorry

end translate_point_correct_l5_5487


namespace olivia_bags_count_l5_5858

def cans_per_bag : ℕ := 5
def total_cans : ℕ := 20

theorem olivia_bags_count : total_cans / cans_per_bag = 4 := by
  sorry

end olivia_bags_count_l5_5858


namespace wholesale_cost_proof_l5_5511

-- Definitions based on conditions
def wholesale_cost (W : ℝ) := W
def retail_price (W : ℝ) := 1.20 * W
def employee_paid (R : ℝ) := 0.90 * R

-- Theorem statement: given the conditions, prove that the wholesale cost is $200.
theorem wholesale_cost_proof : 
  ∃ W : ℝ, (retail_price W = 1.20 * W) ∧ (employee_paid (retail_price W) = 216) ∧ W = 200 :=
by 
  let W := 200
  have hp : retail_price W = 1.20 * W := by sorry
  have ep : employee_paid (retail_price W) = 216 := by sorry
  exact ⟨W, hp, ep, rfl⟩

end wholesale_cost_proof_l5_5511


namespace area_outside_two_small_squares_l5_5984

theorem area_outside_two_small_squares (L S : ℝ) (hL : L = 9) (hS : S = 4) :
  let large_square_area := L^2
  let small_square_area := S^2
  let combined_small_squares_area := 2 * small_square_area
  large_square_area - combined_small_squares_area = 49 :=
by
  sorry

end area_outside_two_small_squares_l5_5984


namespace probability_of_adjacent_vertices_in_decagon_l5_5319

/-- Define the number of vertices in the decagon -/
def num_vertices : ℕ := 10

/-- Define the total number of ways to choose two distinct vertices from the decagon -/
def total_possible_outcomes : ℕ := num_vertices * (num_vertices - 1) / 2

/-- Define the number of favorable outcomes where the two chosen vertices are adjacent -/
def favorable_outcomes : ℕ := num_vertices

/-- Define the probability of selecting two adjacent vertices -/
def probability_adjacent_vertices : ℚ := favorable_outcomes / total_possible_outcomes

/-- The main theorem statement -/
theorem probability_of_adjacent_vertices_in_decagon : probability_adjacent_vertices = 2 / 9 := 
  sorry

end probability_of_adjacent_vertices_in_decagon_l5_5319


namespace altitude_of_triangle_l5_5994

theorem altitude_of_triangle
  (a b c : ℝ)
  (h₁ : a = 13)
  (h₂ : b = 15)
  (h₃ : c = 22)
  (h₄ : a + b > c)
  (h₅ : a + c > b)
  (h₆ : b + c > a) :
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h := (2 * A) / c
  h = (30 * Real.sqrt 10) / 11 :=
by
  sorry

end altitude_of_triangle_l5_5994


namespace parallel_line_slope_l5_5941

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℚ, m = 1 / 2 :=
by {
  use 1 / 2,
  sorry
}

end parallel_line_slope_l5_5941


namespace slope_of_parallel_line_l5_5921

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l5_5921


namespace ordering_of_xyz_l5_5714

theorem ordering_of_xyz :
  let x := Real.sqrt 3
  let y := Real.log 2 / Real.log 3
  let z := Real.cos 2
  z < y ∧ y < x :=
by
  let x := Real.sqrt 3
  let y := Real.log 2 / Real.log 3
  let z := Real.cos 2
  sorry

end ordering_of_xyz_l5_5714


namespace minimum_a_inequality_l5_5241

variable {x y : ℝ}

/-- The inequality (x + y) * (1/x + a/y) ≥ 9 holds for any positive real numbers x and y 
     if and only if a ≥ 4.  -/
theorem minimum_a_inequality (a : ℝ) (h : ∀ (x y : ℝ), 0 < x → 0 < y → (x + y) * (1 / x + a / y) ≥ 9) :
  a ≥ 4 :=
by
  sorry

end minimum_a_inequality_l5_5241


namespace rectangular_field_length_l5_5972

   theorem rectangular_field_length (w l : ℝ) 
     (h1 : l = 2 * w)
     (h2 : 64 = 8 * 8)
     (h3 : 64 = (1/72) * (l * w)) :
     l = 96 :=
   sorry
   
end rectangular_field_length_l5_5972


namespace maximize_probability_of_sum_12_l5_5152

-- Define our list of integers
def integer_list := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the condition that removing an integer produces a list without it
def remove (n : ℤ) (lst : List ℤ) : List ℤ :=
  lst.filter (λ x => x ≠ n)

-- Define the condition of randomly choosing two distinct integers that sum to 12
def pairs_summing_to_12 (lst : List ℤ) : List (ℤ × ℤ) :=
  lst.product lst |>.filter (λ p => p.1 < p.2 ∧ p.1 + p.2 = 12)

-- State our theorem
theorem maximize_probability_of_sum_12 : 
  ∀ l, l = integer_list → 
       (∀ n ≠ 6, length (pairs_summing_to_12 (remove n l)) < length (pairs_summing_to_12 (remove 6 l))) :=
by
  intros
  sorry

end maximize_probability_of_sum_12_l5_5152


namespace simplest_quadratic_radicals_l5_5829

theorem simplest_quadratic_radicals (a : ℝ) :
  (3 * a - 8 ≥ 0) ∧ (17 - 2 * a ≥ 0) → a = 5 :=
by
  intro h
  sorry

end simplest_quadratic_radicals_l5_5829


namespace sum_series_eq_l5_5394

theorem sum_series_eq :
  ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1) = 3 / 2 :=
sorry

end sum_series_eq_l5_5394


namespace correct_transformation_l5_5995

theorem correct_transformation (a b c : ℝ) (h : c ≠ 0) (h1 : a / c = b / c) : a = b :=
by 
  -- Actual proof would go here, but we use sorry for the scaffold.
  sorry

end correct_transformation_l5_5995


namespace lexi_laps_l5_5852

theorem lexi_laps (total_distance lap_distance : ℝ) (h1 : total_distance = 3.25) (h2 : lap_distance = 0.25) :
  total_distance / lap_distance = 13 :=
by
  sorry

end lexi_laps_l5_5852


namespace ellen_painting_time_l5_5031

def time_to_paint_lilies := 5
def time_to_paint_roses := 7
def time_to_paint_orchids := 3
def time_to_paint_vines := 2

def number_of_lilies := 17
def number_of_roses := 10
def number_of_orchids := 6
def number_of_vines := 20

def total_time := 213

theorem ellen_painting_time:
  time_to_paint_lilies * number_of_lilies +
  time_to_paint_roses * number_of_roses +
  time_to_paint_orchids * number_of_orchids +
  time_to_paint_vines * number_of_vines = total_time := by
  sorry

end ellen_painting_time_l5_5031


namespace point_M_in_second_quadrant_l5_5263

-- Given conditions
def m : ℤ := -2
def n : ℤ := 1

-- Definitions to identify the quadrants
def point_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

-- Problem statement to prove
theorem point_M_in_second_quadrant : 
  point_in_second_quadrant m n :=
by
  sorry

end point_M_in_second_quadrant_l5_5263


namespace shopkeeper_profit_percentage_l5_5194

theorem shopkeeper_profit_percentage
  (C : ℝ) -- The cost price of one article
  (cost_price_50 : ℝ := 50 * C) -- The cost price of 50 articles
  (cost_price_70 : ℝ := 70 * C) -- The cost price of 70 articles
  (selling_price_50 : ℝ := 70 * C) -- Selling price of 50 articles is the cost price of 70 articles
  :
  ∃ (P : ℝ), P = 40 :=
by
  sorry

end shopkeeper_profit_percentage_l5_5194


namespace min_voters_l5_5094

theorem min_voters (total_voters : ℕ) (districts : ℕ) (sections_per_district : ℕ) 
  (voters_per_section : ℕ) (majority_sections : ℕ) (majority_districts : ℕ) 
  (winner : string) (is_tall_winner : winner = "Tall") 
  (total_voters = 105) (districts = 5) (sections_per_district = 7) 
  (voters_per_section = 3) (majority_sections = 4) (majority_districts = 3) :
  ∃ (min_voters : ℕ), min_voters = 24 :=
by
  sorry

end min_voters_l5_5094


namespace abs_ab_eq_2128_l5_5472

theorem abs_ab_eq_2128 (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : ∃ r s : ℤ, r ≠ s ∧ ∃ r' : ℤ, r' = r ∧ 
          (x^3 + a * x^2 + b * x + 16 * a = (x - r)^2 * (x - s) ∧ r * r * s = -16 * a)) :
  |a * b| = 2128 :=
sorry

end abs_ab_eq_2128_l5_5472


namespace combined_age_of_siblings_l5_5203

-- We are given Aaron's age
def aaronAge : ℕ := 15

-- Henry's sister's age is three times Aaron's age
def henrysSisterAge : ℕ := 3 * aaronAge

-- Henry's age is four times his sister's age
def henryAge : ℕ := 4 * henrysSisterAge

-- The combined age of the siblings
def combinedAge : ℕ := aaronAge + henrysSisterAge + henryAge

theorem combined_age_of_siblings : combinedAge = 240 := by
  sorry

end combined_age_of_siblings_l5_5203


namespace find_c_l5_5874

theorem find_c (x c : ℚ) (h1 : 3 * x + 5 = 1) (h2 : c * x + 8 = 6) : c = 3 / 2 := 
sorry

end find_c_l5_5874


namespace length_of_AB_l5_5464

-- Define the distances given as conditions
def AC : ℝ := 5
def BD : ℝ := 6
def CD : ℝ := 3

-- Define the linear relationship of points A, B, C, D on the line
def points_on_line_in_order := true -- This is just a placeholder

-- Main theorem to prove
theorem length_of_AB : AB = 2 :=
by
  -- Apply the conditions and the linear relationships
  have BC : ℝ := BD - CD
  have AB : ℝ := AC - BC
  -- This would contain the actual proof using steps, but we skip it here
  sorry

end length_of_AB_l5_5464


namespace prob_A_second_day_is_correct_l5_5788

-- Definitions for the problem conditions
def prob_first_day_A : ℝ := 0.5
def prob_A_given_A : ℝ := 0.6
def prob_first_day_B : ℝ := 0.5
def prob_A_given_B : ℝ := 0.5

-- Calculate the probability of going to A on the second day
def prob_A_second_day : ℝ :=
  prob_first_day_A * prob_A_given_A + prob_first_day_B * prob_A_given_B

-- The theorem statement
theorem prob_A_second_day_is_correct : 
  prob_A_second_day = 0.55 :=
by
  unfold prob_A_second_day prob_first_day_A prob_A_given_A prob_first_day_B prob_A_given_B
  sorry

end prob_A_second_day_is_correct_l5_5788


namespace three_digit_numbers_proof_l5_5783

-- Definitions and conditions
def are_digits_distinct (A B C : ℕ) := (A ≠ B) ∧ (B ≠ C) ∧ (A ≠ C)

def is_arithmetic_mean (A B C : ℕ) := 2 * B = A + C

def geometric_mean_property (A B C : ℕ) := 
  (100 * A + 10 * B + C) * (100 * C + 10 * A + B) = (100 * B + 10 * C + A)^2

-- statement of the proof problem
theorem three_digit_numbers_proof :
  ∃ A B C : ℕ, (10 ≤ A) ∧ (A ≤ 99) ∧ (10 ≤ B) ∧ (B ≤ 99) ∧ (10 ≤ C) ∧ (C ≤ 99) ∧
  (A * 100 + B * 10 + C = 432 ∨ A * 100 + B * 10 + C = 864) ∧
  are_digits_distinct A B C ∧
  is_arithmetic_mean A B C ∧
  geometric_mean_property A B C :=
by {
  -- The Lean proof goes here
  sorry
}

end three_digit_numbers_proof_l5_5783


namespace sum_series_eq_one_l5_5403

noncomputable def series : ℝ := ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1)

theorem sum_series_eq_one : series = 1 := 
by sorry

end sum_series_eq_one_l5_5403


namespace a2_plus_a3_eq_40_l5_5711

theorem a2_plus_a3_eq_40 : 
  ∀ (a a1 a2 a3 a4 a5 : ℤ), 
  (2 * x - 1)^5 = a * x^5 + a1 * x^4 + a2 * x^3 + a3 * x^2 + a4 * x + a5 → 
  a2 + a3 = 40 :=
by
  sorry

end a2_plus_a3_eq_40_l5_5711


namespace new_average_commission_is_250_l5_5274

-- Definitions based on the problem conditions
def C : ℝ := 1000
def n : ℝ := 6
def increase_in_average_commission : ℝ := 150

-- Theorem stating the new average commission is $250
theorem new_average_commission_is_250 (x : ℝ) (h1 : x + increase_in_average_commission = (5 * x + C) / n) :
  x + increase_in_average_commission = 250 := by
  sorry

end new_average_commission_is_250_l5_5274


namespace slope_of_parallel_line_l5_5911

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l5_5911


namespace unit_cost_decreases_l5_5878

def regression_equation (x : ℝ) : ℝ := 356 - 1.5 * x

theorem unit_cost_decreases (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -1.5 := 
by sorry


end unit_cost_decreases_l5_5878


namespace slope_of_parallel_line_l5_5957

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end slope_of_parallel_line_l5_5957


namespace coordinates_of_OC_l5_5244

-- Define the given vectors
def OP : ℝ × ℝ := (2, 1)
def OA : ℝ × ℝ := (1, 7)
def OB : ℝ × ℝ := (5, 1)

-- Define the dot product for ℝ × ℝ
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define OC as a point on line OP, parameterized by t
def OC (t : ℝ) : ℝ × ℝ := (2 * t, t)

-- Define CA and CB
def CA (t : ℝ) : ℝ × ℝ := (OA.1 - (OC t).1, OA.2 - (OC t).2)
def CB (t : ℝ) : ℝ × ℝ := (OB.1 - (OC t).1, OB.2 - (OC t).2)

-- Prove that minimization of dot_product (CA t) (CB t) occurs at OC = (4, 2)
noncomputable def find_coordinates_at_min_dot_product : Prop :=
  ∃ (t : ℝ), t = 2 ∧ OC t = (4, 2)

-- The theorem statement
theorem coordinates_of_OC : find_coordinates_at_min_dot_product :=
sorry

end coordinates_of_OC_l5_5244


namespace find_a_for_even_function_l5_5076

theorem find_a_for_even_function :
  ∀ a : ℝ, (∀ x : ℝ, a * 3^x + 1 / 3^x = a * 3^(-x) + 1 / 3^(-x)) → a = 1 :=
by
  sorry

end find_a_for_even_function_l5_5076


namespace slope_of_parallel_line_l5_5926

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end slope_of_parallel_line_l5_5926


namespace C_share_of_rent_l5_5766

-- Define the given conditions
def A_ox_months : ℕ := 10 * 7
def B_ox_months : ℕ := 12 * 5
def C_ox_months : ℕ := 15 * 3
def total_rent : ℕ := 175
def total_ox_months : ℕ := A_ox_months + B_ox_months + C_ox_months
def cost_per_ox_month := total_rent / total_ox_months

-- The goal is to prove that C's share of the rent is Rs. 45
theorem C_share_of_rent : C_ox_months * cost_per_ox_month = 45 := by
  -- Adding sorry to skip the proof
  sorry

end C_share_of_rent_l5_5766


namespace solve_x_from_operation_l5_5794

def operation (a b c d : ℝ) : ℝ := a * c + b * d

theorem solve_x_from_operation :
  ∀ x : ℝ, operation (2 * x) 3 3 (-1) = 3 → x = 1 :=
by
  intros x h
  sorry

end solve_x_from_operation_l5_5794


namespace parallel_line_slope_l5_5917

def slope_of_parallel_line : ℚ :=
  let line_eq := (3 : ℚ) * (x : ℚ) - (6 : ℚ) * (y : ℚ) = (12 : ℚ)
  in (1 / 2 : ℚ)

theorem parallel_line_slope (x y : ℚ) (h : 3 * x - 6 * y = 12) : slope_of_parallel_line = 1 / 2 :=
by
  sorry

end parallel_line_slope_l5_5917


namespace lawsuit_win_probability_l5_5521

theorem lawsuit_win_probability (P_L1 P_L2 P_W1 P_W2 : ℝ) (h1 : P_L2 = 0.5) 
  (h2 : P_L1 * P_L2 = P_W1 * P_W2 + 0.20 * P_W1 * P_W2)
  (h3 : P_W1 + P_L1 = 1)
  (h4 : P_W2 + P_L2 = 1) : 
  P_W1 = 1 / 2.20 :=
by
  sorry

end lawsuit_win_probability_l5_5521


namespace slope_parallel_to_original_line_l5_5950

-- Define the original line equation in standard form 3x - 6y = 12 as a condition.
def original_line (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line parallel to the given line.
def slope_of_parallel_line (m : ℝ) : Prop := m = 1 / 2

-- The proof problem statement.
theorem slope_parallel_to_original_line : ∀ (m : ℝ), (∀ x y, original_line x y → slope_of_parallel_line m) :=
by sorry

end slope_parallel_to_original_line_l5_5950


namespace net_increase_in_bicycles_l5_5276

def bicycles_sold (fri_sat_sun : ℤ × ℤ × ℤ) : ℤ :=
  fri_sat_sun.1 + fri_sat_sun.2 + fri_sat_sun.3

def bicycles_bought (fri_sat_sun : ℤ × ℤ × ℤ) : ℤ :=
  fri_sat_sun.1 + fri_sat_sun.2 + fri_sat_sun.3

def net_increase (sold bought : ℤ) : ℤ :=
  bought - sold

theorem net_increase_in_bicycles :
  let bicycles_sold_days := (10, 12, 9)
  let bicycles_bought_days := (15, 8, 11)
  net_increase (bicycles_sold bicycles_sold_days) (bicycles_bought bicycles_bought_days) = 3 :=
by
  sorry

end net_increase_in_bicycles_l5_5276


namespace find_y_l5_5016

theorem find_y : ∃ y : ℝ, 1.5 * y - 10 = 35 ∧ y = 30 :=
by
  sorry

end find_y_l5_5016


namespace minimize_fees_at_5_l5_5377

noncomputable def minimize_costs (x : ℝ) (y1 y2 : ℝ) : Prop :=
  let k1 := 40
  let k2 := 8 / 5
  y1 = k1 / x ∧ y2 = k2 * x ∧ (∀ x, y1 + y2 ≥ 16 ∧ (y1 + y2 = 16 ↔ x = 5))

theorem minimize_fees_at_5 :
  minimize_costs 5 4 16 :=
sorry

end minimize_fees_at_5_l5_5377


namespace div_equivalence_l5_5247

theorem div_equivalence (a b c : ℝ) (h1: a / b = 3) (h2: b / c = 2 / 5) : c / a = 5 / 6 :=
by sorry

end div_equivalence_l5_5247


namespace polynomial_factorization_l5_5968

-- Define the polynomial and its factorized form
def polynomial (x : ℝ) : ℝ := x^2 - 4*x + 4
def factorized_form (x : ℝ) : ℝ := (x - 2)^2

-- The theorem stating that the polynomial equals its factorized form
theorem polynomial_factorization (x : ℝ) : polynomial x = factorized_form x :=
by {
  sorry -- Proof skipped
}

end polynomial_factorization_l5_5968


namespace total_questions_on_test_l5_5382

theorem total_questions_on_test :
  ∀ (correct incorrect score : ℕ),
  (score = correct - 2 * incorrect) →
  (score = 76) →
  (correct = 92) →
  (correct + incorrect = 100) :=
by
  intros correct incorrect score grading_system score_eq correct_eq
  sorry

end total_questions_on_test_l5_5382


namespace park_will_have_9_oak_trees_l5_5621

def current_oak_trees : Nat := 5
def additional_oak_trees : Nat := 4
def total_oak_trees : Nat := current_oak_trees + additional_oak_trees

theorem park_will_have_9_oak_trees : total_oak_trees = 9 :=
by
  sorry

end park_will_have_9_oak_trees_l5_5621


namespace cyclist_total_distance_l5_5288

-- Definitions for velocities and times
def v1 : ℝ := 2  -- velocity in the first minute (m/s)
def v2 : ℝ := 4  -- velocity in the second minute (m/s)
def t : ℝ := 60  -- time interval in seconds (1 minute)

-- Total distance covered in two minutes
def total_distance : ℝ := v1 * t + v2 * t

-- The proof statement
theorem cyclist_total_distance : total_distance = 360 := by
  sorry

end cyclist_total_distance_l5_5288


namespace units_digit_sum_42_4_24_4_l5_5756

-- Define the units digit function
def units_digit (n : ℕ) : ℕ := n % 10

-- Given conditions
def units_digit_42_4 : units_digit (42^4) = 6 := sorry
def units_digit_24_4 : units_digit (24^4) = 6 := sorry

-- Theorem to prove
theorem units_digit_sum_42_4_24_4 :
  units_digit (42^4 + 24^4) = 2 :=
by
  -- Use the given conditions
  have h1 : units_digit (42^4) = 6 := units_digit_42_4
  have h2 : units_digit (24^4) = 6 := units_digit_24_4
  -- Calculate the units digit of their sum
  calc 
    units_digit (42^4 + 24^4)
        = units_digit (6 + 6) : by rw [h1, h2]
    ... = units_digit 12    : by norm_num
    ... = 2                 : by norm_num

end units_digit_sum_42_4_24_4_l5_5756


namespace slope_of_parallel_line_l5_5931

theorem slope_of_parallel_line (x y : ℝ) :
  ∀ (m : ℝ), (3 * x - 6 * y = 12) → (m = 1 / 2) :=
begin
  sorry
end

end slope_of_parallel_line_l5_5931


namespace sum_of_coordinates_of_D_l5_5465

/--
Given points A = (4,8), B = (2,4), C = (6,6), and D = (a,b) in the first quadrant, if the quadrilateral formed by joining the midpoints of the segments AB, BC, CD, and DA is a square with sides inclined at 45 degrees to the x-axis, then the sum of the coordinates of point D is 6.
-/
theorem sum_of_coordinates_of_D 
  (a b : ℝ)
  (h_quadrilateral : ∃ A B C D : Prod ℝ ℝ, 
    A = (4, 8) ∧ B = (2, 4) ∧ C = (6, 6) ∧ D = (a, b) ∧ 
    ∃ M1 M2 M3 M4 : Prod ℝ ℝ,
    M1 = ((4 + 2) / 2, (8 + 4) / 2) ∧ M2 = ((2 + 6) / 2, (4 + 6) / 2) ∧ 
    M3 = (M2.1 + 1, M2.2 - 1) ∧ M4 = (M3.1 + 1, M3.2 + 1) ∧ 
    M3 = ((a + 6) / 2, (b + 6) / 2) ∧ M4 = ((a + 4) / 2, (b + 8) / 2)
  ) : 
  a + b = 6 := sorry

end sum_of_coordinates_of_D_l5_5465


namespace unique_ordered_pair_satisfies_equation_l5_5653

theorem unique_ordered_pair_satisfies_equation :
  ∃! (m n : ℕ), 0 < m ∧ 0 < n ∧ (6 / m + 3 / n + 1 / (m * n) = 1) :=
by
  sorry

end unique_ordered_pair_satisfies_equation_l5_5653


namespace Grisha_owes_correct_l5_5600

noncomputable def Grisha_owes (dish_cost : ℝ) : ℝ × ℝ :=
  let misha_paid := 3 * dish_cost
  let sasha_paid := 2 * dish_cost
  let friends_contribution := 50
  let equal_payment := 50 / 2
  (misha_paid - equal_payment, sasha_paid - equal_payment)

theorem Grisha_owes_correct :
  ∀ (dish_cost : ℝ), (dish_cost = 30) → Grisha_owes dish_cost = (40, 10) :=
by
  intro dish_cost h
  rw [h]
  unfold Grisha_owes
  simp
  sorry

end Grisha_owes_correct_l5_5600


namespace golden_section_search_third_point_l5_5337

noncomputable def golden_ratio : ℝ := 0.618

theorem golden_section_search_third_point :
  let L₀ := 1000
  let U₀ := 2000
  let d₀ := U₀ - L₀
  let x₁ := U₀ - golden_ratio * d₀
  let x₂ := L₀ + golden_ratio * d₀
  let d₁ := U₀ - x₁
  let x₃ := x₁ + golden_ratio * d₁
  x₃ = 1764 :=
by
  sorry

end golden_section_search_third_point_l5_5337


namespace max_value_of_z_l5_5232

theorem max_value_of_z 
    (x y : ℝ) 
    (h1 : |2 * x + y + 1| ≤ |x + 2 * y + 2|)
    (h2 : -1 ≤ y ∧ y ≤ 1) : 
    2 * x + y ≤ 5 := 
sorry

end max_value_of_z_l5_5232


namespace side_length_sum_area_l5_5744

theorem side_length_sum_area (a b c d : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 12) :
  d = 13 :=
by
  -- Proof is not required
  sorry

end side_length_sum_area_l5_5744


namespace jacket_cost_correct_l5_5120

-- Definitions based on given conditions
def total_cost : ℝ := 33.56
def cost_shorts : ℝ := 13.99
def cost_shirt : ℝ := 12.14
def cost_jacket : ℝ := 7.43

-- Formal statement of the proof problem in Lean 4
theorem jacket_cost_correct :
  total_cost = cost_shorts + cost_shirt + cost_jacket :=
by
  sorry

end jacket_cost_correct_l5_5120


namespace probability_adjacent_vertices_decagon_l5_5310

theorem probability_adjacent_vertices_decagon :
  let V := 10 in  -- Number of vertices in a decagon
  let favorable_outcomes := 2 in  -- Number of adjacent vertices to any chosen vertex
  let total_outcomes := V - 1 in  -- Total possible outcomes for the second vertex
  (favorable_outcomes / total_outcomes) = (2 / 9) :=
by
  let V := 10
  let favorable_outcomes := 2
  let total_outcomes := V - 1
  have probability := (favorable_outcomes / total_outcomes)
  have target_prob := (2 / 9)
  sorry

end probability_adjacent_vertices_decagon_l5_5310


namespace remainder_sum_first_seven_primes_div_eighth_prime_l5_5164

theorem remainder_sum_first_seven_primes_div_eighth_prime :
  let sum_of_first_seven_primes := 2 + 3 + 5 + 7 + 11 + 13 + 17 in
  let eighth_prime := 19 in
  sum_of_first_seven_primes % eighth_prime = 1 :=
by
  let sum_of_first_seven_primes := 2 + 3 + 5 + 7 + 11 + 13 + 17
  let eighth_prime := 19
  have : sum_of_first_seven_primes = 58 := by decide
  have : eighth_prime = 19 := rfl
  sorry

end remainder_sum_first_seven_primes_div_eighth_prime_l5_5164


namespace max_profit_at_grade_5_l5_5637

-- Defining the conditions
def profit_per_item (x : ℕ) : ℕ :=
  4 * (x - 1) + 8

def production_count (x : ℕ) : ℕ := 
  60 - 6 * (x - 1)

def daily_profit (x : ℕ) : ℕ :=
  profit_per_item x * production_count x

-- The grade range
def grade_range (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 10

-- Prove that the grade that maximizes the profit is 5
theorem max_profit_at_grade_5 : (1 ≤ x ∧ x ≤ 10) → daily_profit x ≤ daily_profit 5 :=
sorry

end max_profit_at_grade_5_l5_5637


namespace inequality_holds_for_m_l5_5271

theorem inequality_holds_for_m (n : ℕ) (m : ℕ) :
  (∀ a b : ℝ, (0 < a ∧ 0 < b) ∧ (a + b = 2) → (1 / a^n + 1 / b^n ≥ a^m + b^m)) ↔ (m = n ∨ m = n + 1) :=
by
  sorry

end inequality_holds_for_m_l5_5271


namespace slope_of_parallel_line_l5_5938

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l5_5938


namespace maximize_probability_remove_6_l5_5153

-- Definitions
def integers_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12] -- After removing 6
def initial_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Theorem Statement
theorem maximize_probability_remove_6 :
  ∀ (n : Int),
  n ∈ initial_list →
  n ≠ 6 →
  ∃ (a b : Int), a ∈ integers_list ∧ b ∈ integers_list ∧ a ≠ b ∧ a + b = 12 → False :=
by
  intros n hn hn6
  -- Placeholder for proof
  sorry

end maximize_probability_remove_6_l5_5153


namespace product_of_third_sides_is_correct_l5_5335

def sqrt_approx (x : ℝ) :=  -- Approximate square root function
  if x = 7 then 2.646 else 0

def product_of_third_sides (a b : ℝ) : ℝ :=
  let hypotenuse := real.sqrt (a * a + b * b)
  let leg := real.sqrt (b * b - a * a)
  hypotenuse * leg

theorem product_of_third_sides_is_correct :
  product_of_third_sides 6 8 = 52.9 :=
by
  unfold product_of_third_sides
  rw [if_pos (rfl : (real.sqrt (8 * 8 - 6 * 6)) = 2.646)]
  norm_num
  sorry

end product_of_third_sides_is_correct_l5_5335


namespace car_division_ways_l5_5771

/-- 
Prove that the number of ways to divide 6 people 
into two different cars, with each car holding 
a maximum of 4 people, is equal to 50. 
-/
theorem car_division_ways : 
  (∃ s1 s2 : Finset ℕ, s1.card = 2 ∧ s2.card = 4) ∨ 
  (∃ s1 s2 : Finset ℕ, s1.card = 3 ∧ s2.card = 3) ∨ 
  (∃ s1 s2 : Finset ℕ, s1.card = 4 ∧ s2.card = 2) →
  (15 + 20 + 15 = 50) := 
by 
  sorry

end car_division_ways_l5_5771


namespace large_jars_count_l5_5893

theorem large_jars_count (S L : ℕ) (h1 : S + L = 100) (h2 : S = 62) (h3 : 3 * S + 5 * L = 376) : L = 38 :=
by
  sorry

end large_jars_count_l5_5893


namespace find_p_l5_5569

-- Definitions
variables {n : ℕ} {p : ℝ}
def X : Type := ℕ -- Assume X is ℕ-valued

-- Conditions
axiom binomial_expectation : n * p = 6
axiom binomial_variance : n * p * (1 - p) = 3

-- Question to prove
theorem find_p : p = 1 / 2 :=
by
  sorry

end find_p_l5_5569


namespace intersection_point_sum_l5_5057

noncomputable def h : ℝ → ℝ := sorry
noncomputable def j : ℝ → ℝ := sorry

axiom h2 : h 2 = 2
axiom j2 : j 2 = 2
axiom h4 : h 4 = 6
axiom j4 : j 4 = 6
axiom h6 : h 6 = 12
axiom j6 : j 6 = 12
axiom h8 : h 8 = 12
axiom j8 : j 8 = 12

theorem intersection_point_sum :
  (∃ x, h (x + 2) = j (2 * x)) →
  (h (2 + 2) = j (2 * 2) ∨ h (4 + 2) = j (2 * 4)) →
  (h (4) = 6 ∧ j (4) = 6 ∧ h 6 = 12 ∧ j 8 = 12) →
  (∃ x, (x = 2 ∧ (x + h (x + 2) = 8) ∨ x = 4 ∧ (x + h (x + 2) = 16))) :=
by
  sorry

end intersection_point_sum_l5_5057


namespace max_value_f_l5_5290

open Real

def f (x : ℝ) : ℝ := x + 2 * cos x

theorem max_value_f : 
  ∃ x ∈ Icc 0 (π / 2), ∀ y ∈ Icc 0 (π / 2), f y ≤ f x ∧ f x = (π / 6 + ⟨sqrt 3, Real.sqrt_pos.2 zero_lt_three⟩) := 
sorry

end max_value_f_l5_5290


namespace no_non_trivial_solution_l5_5028

theorem no_non_trivial_solution (a b c : ℤ) (h : a^2 = 2 * b^2 + 3 * c^2) : a = 0 ∧ b = 0 ∧ c = 0 :=
sorry

end no_non_trivial_solution_l5_5028


namespace determine_coefficients_l5_5408

noncomputable def quadratic_polynomial (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

def is_quad_max (f : ℝ → ℝ) (k x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ k

def sum_cubes_eq (f : ℝ → ℝ) (sum_cubes : ℝ) : Prop :=
  ∑ (root : ℝ) in finset.univ.filter (λ x, f x = 0), root^3 = sum_cubes

theorem determine_coefficients :
  ∃ a b c : ℝ, quadratic_polynomial a b c = (λ x, a * x^2 + b * x + c) ∧
  is_quad_max (quadratic_polynomial a b c) 25 (1 / 2) ∧
  sum_cubes_eq (quadratic_polynomial a b c) 19 ∧
  a = -4 ∧ b = 4 ∧ c = 24 :=
begin
  sorry
end

end determine_coefficients_l5_5408


namespace total_shaded_area_l5_5780

def rectangle_area (R : ℝ) : ℝ := R * R
def square_area (S : ℝ) : ℝ := S * S

theorem total_shaded_area 
  (R S : ℝ)
  (h1 : 18 = 2 * R)
  (h2 : R = 4 * S) :
  rectangle_area R + 12 * square_area S = 141.75 := 
  by 
    sorry

end total_shaded_area_l5_5780


namespace line_ellipse_intersect_l5_5875

theorem line_ellipse_intersect (m k : ℝ) (h₀ : ∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ x^2 / 5 + y^2 / m = 1) : m ≥ 1 ∧ m ≠ 5 :=
sorry

end line_ellipse_intersect_l5_5875


namespace polar_to_cartesian_l5_5814

theorem polar_to_cartesian (ρ θ : ℝ) (h : ρ = 4 * Real.cos θ) :
  ∃ x y : ℝ, (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧
  (x - 2)^2 + y^2 = 4) :=
sorry

end polar_to_cartesian_l5_5814


namespace min_number_of_lucky_weights_l5_5296

-- Definitions and conditions
def weight (n: ℕ) := n -- A weight is represented as a natural number.

def is_lucky (weights: Finset ℕ) (w: ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ weights ∧ b ∈ weights ∧ a ≠ b ∧ w = a + b
-- w is "lucky" if it's the sum of two other distinct weights in the set.

def min_lucky_guarantee (weights: Finset ℕ) (k: ℕ) : Prop :=
  ∀ (w1 w2 : ℕ), w1 ∈ weights ∧ w2 ∈ weights →
    ∃ (lucky_weights : Finset ℕ), lucky_weights.card = k ∧
    (is_lucky weights w1 ∧ is_lucky weights w2 ∧ (w1 ≥ 3 * w2 ∨ w2 ≥ 3 * w1))
-- The minimum number k of "lucky" weights ensures there exist two weights 
-- such that their masses differ by at least a factor of three.

-- The theorem to be proven
theorem min_number_of_lucky_weights (weights: Finset ℕ) (h_distinct: weights.card = 100) :
  ∃ k, min_lucky_guarantee weights k ∧ k = 87 := 
sorry

end min_number_of_lucky_weights_l5_5296


namespace percentage_of_alcohol_in_first_vessel_l5_5514

variable (x : ℝ) -- percentage of alcohol in the first vessel in decimal form, i.e., x% is represented as x/100

-- conditions
variable (v1_capacity : ℝ := 2)
variable (v2_capacity : ℝ := 6)
variable (v2_alcohol_concentration : ℝ := 0.5)
variable (total_capacity : ℝ := 10)
variable (new_concentration : ℝ := 0.37)

theorem percentage_of_alcohol_in_first_vessel :
  (x / 100) * v1_capacity + v2_alcohol_concentration * v2_capacity = new_concentration * total_capacity -> x = 35 := 
by
  sorry

end percentage_of_alcohol_in_first_vessel_l5_5514


namespace quadratic_equation_roots_transformation_l5_5712

theorem quadratic_equation_roots_transformation (α β : ℝ) 
  (h1 : 3 * α^2 + 7 * α + 4 = 0)
  (h2 : 3 * β^2 + 7 * β + 4 = 0) :
  ∃ y : ℝ, 21 * y^2 - 23 * y + 6 = 0 :=
sorry

end quadratic_equation_roots_transformation_l5_5712


namespace smallest_base_for_62_three_digits_l5_5342

theorem smallest_base_for_62_three_digits: 
  ∃ b : ℕ, (b^2 ≤ 62 ∧ 62 < b^3) ∧ ∀ n : ℕ, (n^2 ≤ 62 ∧ 62 < n^3) → n ≥ b :=
by
  sorry

end smallest_base_for_62_three_digits_l5_5342


namespace gcd_65_130_l5_5799

theorem gcd_65_130 : Int.gcd 65 130 = 65 := by
  sorry

end gcd_65_130_l5_5799


namespace decagon_adjacent_vertices_probability_l5_5305

namespace ProbabilityAdjacentVertices

def total_vertices : ℕ := 10

def total_pairs : ℕ := total_vertices * (total_vertices - 1) / 2

def adjacent_pairs : ℕ := total_vertices

def probability_adjacent : ℚ := adjacent_pairs / total_pairs

theorem decagon_adjacent_vertices_probability :
  probability_adjacent = 2 / 9 := by
  sorry

end ProbabilityAdjacentVertices

end decagon_adjacent_vertices_probability_l5_5305


namespace product_is_eight_l5_5110

noncomputable def compute_product (r : ℂ) (hr : r ≠ 1) (hr7 : r^7 = 1) : ℂ :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1)

theorem product_is_eight (r : ℂ) (hr : r ≠ 1) (hr7 : r^7 = 1) : compute_product r hr hr7 = 8 :=
by
  sorry

end product_is_eight_l5_5110


namespace circle_equation_exists_l5_5870

theorem circle_equation_exists :
  ∃ D E F : ℝ, (∀ (x y : ℝ), (x = 0 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
                      (x = 4 → y = 0 → x^2 + y^2 + D * x + E * y + F = 0) ∧
                      (x = -1 → y = 1 → x^2 + y^2 + D * x + E * y + F = 0)) ∧
                      (D = -4) ∧ (E = -6) ∧ (F = 0) :=
by
  use [-4, -6, 0]
  intro x y
  split
  { intros hx hy, simp [hx, hy] }
  split
  { intros hx hy, simp [hx, hy], linarith }
  { intros hx hy, simp [hx, hy], linarith }
  sorry

end circle_equation_exists_l5_5870


namespace sin_cos_difference_l5_5225

theorem sin_cos_difference
  (θ : ℝ)
  (h1 : θ ∈ Set.Ioo 0 Real.pi)
  (h2 : Real.sin θ + Real.cos θ = 1 / 5) :
  Real.sin θ - Real.cos θ = 7 / 5 :=
sorry

end sin_cos_difference_l5_5225


namespace players_taking_all_three_subjects_l5_5523

-- Define the variables for the number of players in each category
def num_players : ℕ := 18
def num_physics : ℕ := 10
def num_biology : ℕ := 7
def num_chemistry : ℕ := 5
def num_physics_biology : ℕ := 3
def num_biology_chemistry : ℕ := 2
def num_physics_chemistry : ℕ := 1

-- Define the proposition we want to prove
theorem players_taking_all_three_subjects :
  ∃ x : ℕ, x = 2 ∧
  num_players = num_physics + num_biology + num_chemistry
                - num_physics_chemistry
                - num_physics_biology
                - num_biology_chemistry
                + x :=
by {
  sorry -- Placeholder for the proof
}

end players_taking_all_three_subjects_l5_5523


namespace cost_of_jacket_is_60_l5_5527

/-- Define the constants from the problem --/
def cost_of_shirt : ℕ := 8
def cost_of_pants : ℕ := 18
def shirts_bought : ℕ := 4
def pants_bought : ℕ := 2
def jackets_bought : ℕ := 2
def carrie_paid : ℕ := 94

/-- Define the problem statement --/
theorem cost_of_jacket_is_60 (total_cost jackets_cost : ℕ) 
    (H1 : total_cost = (shirts_bought * cost_of_shirt) + (pants_bought * cost_of_pants) + jackets_cost)
    (H2 : carrie_paid = total_cost / 2)
    : jackets_cost / jackets_bought = 60 := 
sorry

end cost_of_jacket_is_60_l5_5527


namespace right_triangle_third_side_product_l5_5333

def hypotenuse_length (a b : ℕ) : Real := Real.sqrt (a*a + b*b)
def other_leg_length (h b : ℕ) : Real := Real.sqrt (h*h - b*b)

theorem right_triangle_third_side_product (a b : ℕ) (ha : a = 6) (hb : b = 8) : 
  Real.round (hypotenuse_length a b * other_leg_length b a) = 53 :=
by
  have h1 : hypotenuse_length 6 8 = 10 := by sorry
  have h2 : other_leg_length 8 6 = 2 * Real.sqrt 7 := by sorry
  calc
    Real.round (10 * (2 * Real.sqrt 7)) = Real.round (20 * Real.sqrt 7) := by sorry
                                   ...  = 53 := by sorry

end right_triangle_third_side_product_l5_5333


namespace combined_age_of_siblings_l5_5201

theorem combined_age_of_siblings (a s h : ℕ) (h1 : a = 15) (h2 : s = 3 * a) (h3 : h = 4 * s) : a + s + h = 240 :=
by
  sorry

end combined_age_of_siblings_l5_5201


namespace possible_perimeters_l5_5992

theorem possible_perimeters (a b c: ℝ) (h1: a = 1) (h2: b = 1) 
  (h3: c = 1) (h: ∀ x y z: ℝ, x = y ∧ y = z):
  ∃ x y: ℝ, (x = 8/3 ∧ y = 5/2) := 
  by
    sorry

end possible_perimeters_l5_5992


namespace geo_seq_value_l5_5705

variable (a : ℕ → ℝ)
variable (a_2 : a 2 = 2) 
variable (a_4 : a 4 = 8)
variable (geo_prop : a 2 * a 6 = (a 4) ^ 2)

theorem geo_seq_value : a 6 = 32 := 
by 
  sorry

end geo_seq_value_l5_5705


namespace quadratic_solution_l5_5610

theorem quadratic_solution (x : ℝ) : x^2 - 2 * x - 3 = 0 → (x = 3 ∨ x = -1) :=
by
  sorry

end quadratic_solution_l5_5610


namespace complete_laps_l5_5850

-- Definitions based on conditions
def total_distance := 3.25  -- total distance Lexi wants to run
def lap_distance := 0.25    -- distance of one lap

-- Proof statement: Total number of complete laps to cover the given distance
theorem complete_laps (h1 : total_distance = 3 + 1/4) (h2 : lap_distance = 1/4) :
  (total_distance / lap_distance) = 13 :=
by 
  sorry

end complete_laps_l5_5850


namespace intersection_of_A_and_B_is_5_and_8_l5_5061

def A : Set ℕ := {4, 5, 6, 8}
def B : Set ℕ := {5, 7, 8, 9}

theorem intersection_of_A_and_B_is_5_and_8 : A ∩ B = {5, 8} :=
  by sorry

end intersection_of_A_and_B_is_5_and_8_l5_5061


namespace jane_spent_more_on_ice_cream_l5_5841

-- Definitions based on the conditions
def ice_cream_cone_cost : ℕ := 5
def pudding_cup_cost : ℕ := 2
def ice_cream_cones_bought : ℕ := 15
def pudding_cups_bought : ℕ := 5

-- The mathematically equivalent proof statement
theorem jane_spent_more_on_ice_cream : 
  (ice_cream_cones_bought * ice_cream_cone_cost - pudding_cups_bought * pudding_cup_cost) = 65 := 
by
  sorry

end jane_spent_more_on_ice_cream_l5_5841


namespace BD_range_l5_5583

noncomputable def quadrilateral_BD (AB BC CD DA : ℕ) (BD : ℤ) :=
  AB = 7 ∧ BC = 15 ∧ CD = 7 ∧ DA = 11 ∧ (9 ≤ BD ∧ BD ≤ 17)

theorem BD_range : 
  ∀ (AB BC CD DA : ℕ) (BD : ℤ),
  quadrilateral_BD AB BC CD DA BD → 
  9 ≤ BD ∧ BD ≤ 17 :=
by
  intros AB BC CD DA BD h
  cases h
  -- We would then prove the conditions
  sorry

end BD_range_l5_5583


namespace arithmetic_sequence_30th_term_l5_5354

theorem arithmetic_sequence_30th_term :
  let a₁ := 3
  let d := 4
  let n := 30
  a₁ + (n - 1) * d = 119 :=
by
  let a₁ := 3
  let d := 4
  let n := 30
  show a₁ + (n - 1) * d = 119
  sorry

end arithmetic_sequence_30th_term_l5_5354


namespace sum_of_primes_100_sq_plus_1_sq_eq_65_sq_plus_76_sq_l5_5236

theorem sum_of_primes_100_sq_plus_1_sq_eq_65_sq_plus_76_sq (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : 100^2 + 1^2 = p * q ∧ 65^2 + 76^2 = p * q) : p + q = 210 := 
sorry

end sum_of_primes_100_sq_plus_1_sq_eq_65_sq_plus_76_sq_l5_5236


namespace percentage_error_divide_by_5_instead_of_multiplying_by_5_l5_5706

variable (x : ℝ)

theorem percentage_error_divide_by_5_instead_of_multiplying_by_5 (x ≠ 0) :
  ((5 * x - x / 5) / (5 * x)) * 100 = 96 := 
sorry

end percentage_error_divide_by_5_instead_of_multiplying_by_5_l5_5706


namespace circle_passing_points_l5_5871

theorem circle_passing_points (x y : ℝ) :
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 - 4*x - 6*y = 0 :=
by
  intros h
  cases h
  case inl h₁ => 
    rw [h₁.1, h₁.2]
    ring
  case inr h₁ =>
    cases h₁
    case inl h₂ => 
      rw [h₂.1, h₂.2]
      ring
    case inr h₂ =>
      rw [h₂.1, h₂.2]
      ring

end circle_passing_points_l5_5871


namespace number_of_ordered_triples_l5_5866

theorem number_of_ordered_triples (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
    (h_eq : a * b * c - b * c - a * c - a * b + a + b + c = 2013) :
    ∃ n, n = 39 :=
by
  sorry

end number_of_ordered_triples_l5_5866


namespace length_of_RT_in_trapezoid_l5_5634

-- Definition of the trapezoid and initial conditions
def trapezoid (PQ RS PR RT : ℝ) (h : PQ = 3 * RS) (h1 : PR = 15) : Prop :=
  RT = 15 / 4

-- The theorem to be proved
theorem length_of_RT_in_trapezoid (PQ RS PR RT : ℝ) 
  (h : PQ = 3 * RS) (h1 : PR = 15) : trapezoid PQ RS PR RT h h1 :=
by
  sorry

end length_of_RT_in_trapezoid_l5_5634


namespace correct_propositions_l5_5228

-- Definitions based on conditions
def line_perpendicular_to_plane (l : Line) (α : Plane) : Prop := l ∈ perpendicularTo α
def line_in_plane (m : Line) (β : Plane) : Prop := m ∈ β

-- Propositions as functions of lines and planes
def proposition_1 (l : Line) (m : Line) (α : Plane) (β : Plane) : Prop := α ∥ β → line_perpendicular_to_plane l α → line_perpendicular_to_plane l β → l ⊥ m
def proposition_2 (l : Line) (m : Line) (α : Plane) (β : Plane) : Prop := α ⊥ β → line_perpendicular_to_plane l α → l ∥ m
def proposition_3 (l : Line) (m : Line) (α : Plane) (β : Plane) : Prop := l ∥ m → line_perpendicular_to_plane l α → ⨆ (h : line_in_plane m β), α ⊥ β
def proposition_4 (l : Line) (m : Line) (α : Plane) (β : Plane) : Prop := l ⊥ m → ⨆ (h : line_in_plane m β), α ∥ β

-- The final proof statement
theorem correct_propositions (l : Line) (m : Line) (α : Plane) (β : Plane) :
  line_perpendicular_to_plane l α →
  line_in_plane m β →
  (proposition_1 l m α β) ∧ (proposition_3 l m α β) ∧ ¬ (proposition_2 l m α β) ∧ ¬ (proposition_4 l m α β) :=
by
  sorry

end correct_propositions_l5_5228


namespace find_police_stations_in_pittsburgh_l5_5103

-- Conditions
def stores_in_pittsburgh : ℕ := 2000
def hospitals_in_pittsburgh : ℕ := 500
def schools_in_pittsburgh : ℕ := 200
def total_buildings_in_new_city : ℕ := 2175

-- Define the problem statement and the target proof
theorem find_police_stations_in_pittsburgh (P : ℕ) :
  1000 + 1000 + 150 + (P + 5) = total_buildings_in_new_city → P = 20 :=
by
  sorry

end find_police_stations_in_pittsburgh_l5_5103


namespace divides_f_of_nat_l5_5043

variable {n : ℕ}

theorem divides_f_of_nat (n : ℕ) : 5 ∣ (76 * n^5 + 115 * n^4 + 19 * n) := 
sorry

end divides_f_of_nat_l5_5043


namespace Kira_was_away_for_8_hours_l5_5107

theorem Kira_was_away_for_8_hours
  (kibble_rate: ℕ)
  (initial_kibble: ℕ)
  (remaining_kibble: ℕ)
  (hours_per_pound: ℕ) 
  (kibble_eaten: ℕ)
  (kira_was_away: ℕ)
  (h1: kibble_rate = 1)
  (h2: initial_kibble = 3)
  (h3: remaining_kibble = 1)
  (h4: hours_per_pound = 4)
  (h5: kibble_eaten = initial_kibble - remaining_kibble)
  (h6: kira_was_away = hours_per_pound * kibble_eaten) : 
  kira_was_away = 8 :=
by
  sorry

end Kira_was_away_for_8_hours_l5_5107


namespace isosceles_base_angle_eq_43_l5_5080

theorem isosceles_base_angle_eq_43 (α β : ℝ) (h_iso : α = β) (h_sum : α + β + 94 = 180) : α = 43 :=
by
  sorry

end isosceles_base_angle_eq_43_l5_5080


namespace run_to_cafe_time_l5_5431

theorem run_to_cafe_time (h_speed_const : ∀ t1 t2 d1 d2 : ℝ, (t1 / d1) = (t2 / d2))
  (h_store_time : 24 = 3 * (24 / 3))
  (h_cafe_halfway : ∀ d : ℝ, d = 1.5) :
  ∃ t : ℝ, t = 12 :=
by
  sorry

end run_to_cafe_time_l5_5431


namespace semicircle_radius_l5_5018

theorem semicircle_radius (P L W : ℝ) (π : Real) (r : ℝ) 
  (hP : P = 144) (hL : L = 48) (hW : W = 24) (hD : ∃ d, d = 2 * r ∧ d = L) :
  r = 48 / (π + 2) := 
by
  sorry

end semicircle_radius_l5_5018


namespace sugar_amount_l5_5585

-- Definitions based on conditions
variables (S F B C : ℝ) -- S = amount of sugar, F = amount of flour, B = amount of baking soda, C = amount of chocolate chips

-- Conditions
def ratio_sugar_flour (S F : ℝ) : Prop := S / F = 5 / 4
def ratio_flour_baking_soda (F B : ℝ) : Prop := F / B = 10 / 1
def ratio_baking_soda_chocolate_chips (B C : ℝ) : Prop := B / C = 3 / 2
def new_ratio_flour_baking_soda_chocolate_chips (F B C : ℝ) : Prop :=
  F / (B + 120) = 16 / 3 ∧ F / (C + 50) = 16 / 2

-- Prove that the current amount of sugar is 1714 pounds
theorem sugar_amount (S F B C : ℝ) (h1 : ratio_sugar_flour S F)
  (h2 : ratio_flour_baking_soda F B) (h3 : ratio_baking_soda_chocolate_chips B C)
  (h4 : new_ratio_flour_baking_soda_chocolate_chips F B C) : 
  S = 1714 :=
sorry

end sugar_amount_l5_5585


namespace expression_meaningful_range_l5_5435

theorem expression_meaningful_range (a : ℝ) : (∃ x, x = (a + 3) ^ (1/2) / (a - 1)) ↔ (a ≥ -3 ∧ a ≠ 1) :=
by
  sorry

end expression_meaningful_range_l5_5435


namespace total_hours_for_songs_l5_5643

def total_hours_worked_per_day := 10
def total_days_per_song := 10
def number_of_songs := 3

theorem total_hours_for_songs :
  total_hours_worked_per_day * total_days_per_song * number_of_songs = 300 :=
by
  sorry

end total_hours_for_songs_l5_5643


namespace work_increase_percentage_l5_5698

theorem work_increase_percentage (p w : ℕ) (hp : p > 0) : 
  (((4 / 3 : ℚ) * w) - w) / w * 100 = 33.33 := 
sorry

end work_increase_percentage_l5_5698


namespace triangle_middle_side_at_least_sqrt_two_l5_5993

theorem triangle_middle_side_at_least_sqrt_two
    (a b c : ℝ)
    (h1 : a ≥ b) (h2 : b ≥ c)
    (h3 : ∃ α : ℝ, 0 < α ∧ α < π ∧ 1 = 1/2 * b * c * Real.sin α) :
  b ≥ Real.sqrt 2 :=
sorry

end triangle_middle_side_at_least_sqrt_two_l5_5993


namespace slope_of_parallel_line_l5_5913

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l5_5913


namespace Mark_less_than_Craig_l5_5406

-- Definitions for the conditions
def Dave_weight : ℕ := 175
def Dave_bench_press : ℕ := Dave_weight * 3
def Craig_bench_press : ℕ := (20 * Dave_bench_press) / 100
def Mark_bench_press : ℕ := 55

-- The theorem to be proven
theorem Mark_less_than_Craig : Craig_bench_press - Mark_bench_press = 50 :=
by
  sorry

end Mark_less_than_Craig_l5_5406


namespace area_of_annulus_l5_5728

variable {b c h : ℝ}
variable (hb : b > c)
variable (h2 : h^2 = b^2 - 2 * c^2)

theorem area_of_annulus (hb : b > c) (h2 : h^2 = b^2 - 2 * c^2) :
    π * (b^2 - c^2) = π * h^2 := by
  sorry

end area_of_annulus_l5_5728


namespace cylinder_volume_ratio_l5_5211

theorem cylinder_volume_ratio (h_C r_D : ℝ) (V_C V_D : ℝ) :
  h_C = 3 * r_D →
  r_D = h_C →
  V_C = 3 * V_D →
  V_C = (1 / 9) * π * h_C^3 :=
by
  sorry

end cylinder_volume_ratio_l5_5211


namespace minimum_voters_for_tall_l5_5101

-- Define the structure of the problem
def num_voters := 105
def num_districts := 5
def sections_per_district := 7
def voters_per_section := 3
def majority x := ⌊ x / 2 ⌋ + 1 

-- Define conditions
def wins_section (votes_for_tall : ℕ) : Prop := votes_for_tall ≥ majority voters_per_section
def wins_district (sections_won : ℕ) : Prop := sections_won ≥ majority sections_per_district
def wins_contest (districts_won : ℕ) : Prop := districts_won ≥ majority num_districts

-- Define the theorem statement
theorem minimum_voters_for_tall : 
  ∃ (votes_for_tall : ℕ), votes_for_tall = 24 ∧
  (∃ (district_count : ℕ → ℕ), 
    (∀ d, d < num_districts → wins_district (district_count d)) ∧
    wins_contest (∑ d in finset.range num_districts, wins_district (district_count d).count (λ w, w = tt))) := 
sorry

end minimum_voters_for_tall_l5_5101


namespace solve_for_n_l5_5038

theorem solve_for_n : 
  (∃ n : ℤ, (1 / (n + 2) + 2 / (n + 2) + (n + 1) / (n + 2) = 3)) ↔ n = -1 :=
sorry

end solve_for_n_l5_5038


namespace sum_of_digits_T_l5_5747

-- Conditions:
def horse_lap_times := [1, 2, 3, 4, 5, 6, 7, 8]
def S := 840
def total_horses := 8
def min_horses_at_start := 4

-- Question:
def T := 12 -- Least time such that at least 4 horses meet

/-- Prove that the sum of the digits of T is 3 -/
theorem sum_of_digits_T : (1 + 2) = 3 := by
  sorry

end sum_of_digits_T_l5_5747


namespace remaining_glazed_correct_remaining_chocolate_correct_remaining_raspberry_correct_l5_5011

section Doughnuts

variable (initial_glazed : Nat := 10)
variable (initial_chocolate : Nat := 8)
variable (initial_raspberry : Nat := 6)

variable (personA_glazed : Nat := 2)
variable (personA_chocolate : Nat := 1)
variable (personB_glazed : Nat := 1)
variable (personC_chocolate : Nat := 3)
variable (personD_glazed : Nat := 1)
variable (personD_raspberry : Nat := 1)
variable (personE_raspberry : Nat := 1)
variable (personF_raspberry : Nat := 2)

def remaining_glazed : Nat :=
  initial_glazed - (personA_glazed + personB_glazed + personD_glazed)

def remaining_chocolate : Nat :=
  initial_chocolate - (personA_chocolate + personC_chocolate)

def remaining_raspberry : Nat :=
  initial_raspberry - (personD_raspberry + personE_raspberry + personF_raspberry)

theorem remaining_glazed_correct :
  remaining_glazed initial_glazed personA_glazed personB_glazed personD_glazed = 6 :=
by
  sorry

theorem remaining_chocolate_correct :
  remaining_chocolate initial_chocolate personA_chocolate personC_chocolate = 4 :=
by
  sorry

theorem remaining_raspberry_correct :
  remaining_raspberry initial_raspberry personD_raspberry personE_raspberry personF_raspberry = 2 :=
by
  sorry

end Doughnuts

end remaining_glazed_correct_remaining_chocolate_correct_remaining_raspberry_correct_l5_5011


namespace average_first_two_numbers_l5_5727

theorem average_first_two_numbers (a1 a2 a3 a4 a5 a6 : ℝ)
  (h1 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 3.95)
  (h2 : (a3 + a4) / 2 = 3.85)
  (h3 : (a5 + a6) / 2 = 4.200000000000001) :
  (a1 + a2) / 2 = 3.8 :=
by
  sorry

end average_first_two_numbers_l5_5727


namespace cost_of_first_shipment_1100_l5_5614

variables (S J : ℝ)
-- conditions
def second_shipment (S J : ℝ) := 5 * S + 15 * J = 550
def first_shipment (S J : ℝ) := 10 * S + 20 * J

-- goal
theorem cost_of_first_shipment_1100 (S J : ℝ) (h : second_shipment S J) : first_shipment S J = 1100 :=
sorry

end cost_of_first_shipment_1100_l5_5614


namespace geometric_sequence_a11_l5_5833

theorem geometric_sequence_a11
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h3 : a 3 = 4)
  (h7 : a 7 = 12) : 
  a 11 = 36 :=
by
  sorry

end geometric_sequence_a11_l5_5833


namespace gcd_binom_integer_l5_5596

theorem gcd_binom_integer (n m : ℕ) (hnm : n ≥ m) (hm : m ≥ 1) :
  (Nat.gcd m n) * Nat.choose n m % n = 0 := sorry

end gcd_binom_integer_l5_5596


namespace hex_prism_paintings_l5_5987

def num_paintings : ℕ :=
  -- The total number of distinct ways to paint a hex prism according to the conditions
  3 -- Two colors case: white-red, white-blue, red-blue
  + 6 -- Three colors with pattern 121213
  + 1 -- Three colors with identical opposite faces: 123123
  + 3 -- Three colors with non-identical opposite faces: 123213

theorem hex_prism_paintings : num_paintings = 13 := by
  sorry

end hex_prism_paintings_l5_5987


namespace num_signs_in_sign_language_l5_5293

theorem num_signs_in_sign_language (n : ℕ) (h : n^2 - (n - 2)^2 = 888) : n = 223 := 
sorry

end num_signs_in_sign_language_l5_5293


namespace thirtieth_term_value_l5_5350

-- Define the initial term and common difference
def initial_term : ℤ := 3
def common_difference : ℤ := 4

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℤ :=
  initial_term + (n - 1) * common_difference

-- Theorem stating the value of the 30th term
theorem thirtieth_term_value : nth_term 30 = 119 :=
by sorry

end thirtieth_term_value_l5_5350


namespace domain_of_f_l5_5869

noncomputable def f (x : ℝ) := 1 / Real.log (x + 1) + Real.sqrt (9 - x^2)

theorem domain_of_f :
  {x : ℝ | 0 < x + 1} ∩ {x : ℝ | x ≠ 0} ∩ {x : ℝ | 9 - x^2 ≥ 0} = (Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioc 0 (3 : ℝ)) :=
by
  sorry

end domain_of_f_l5_5869


namespace max_pieces_from_cake_l5_5491

theorem max_pieces_from_cake (large_cake_area small_piece_area : ℕ) 
  (h_large_cake : large_cake_area = 15 * 15) 
  (h_small_piece : small_piece_area = 5 * 5) :
  large_cake_area / small_piece_area = 9 := 
by
  sorry

end max_pieces_from_cake_l5_5491


namespace cities_below_50000_l5_5731

theorem cities_below_50000 (p1 p2 : ℝ) (h1 : p1 = 20) (h2: p2 = 65) :
  p1 + p2 = 85 := 
  by sorry

end cities_below_50000_l5_5731


namespace div_equivalence_l5_5246

theorem div_equivalence (a b c : ℝ) (h1: a / b = 3) (h2: b / c = 2 / 5) : c / a = 5 / 6 :=
by sorry

end div_equivalence_l5_5246


namespace find_initial_number_l5_5180

theorem find_initial_number (N : ℝ) (h : ∃ k : ℝ, 330 * k = N + 69.00000000008731) : 
  ∃ m : ℝ, N = 330 * m - 69.00000000008731 :=
by
  sorry

end find_initial_number_l5_5180


namespace right_triangle_of_altitude_ratios_l5_5737

theorem right_triangle_of_altitude_ratios
  (h1 h2 h3 : ℝ) 
  (h1_pos : h1 > 0) 
  (h2_pos : h2 > 0) 
  (h3_pos : h3 > 0) 
  (H : (h1 / h2)^2 + (h1 / h3)^2 = 1) : 
  ∃ a b c : ℝ, a^2 = b^2 + c^2 ∧ h1 = 1 / a ∧ h2 = 1 / b ∧ h3 = 1 / c :=
sorry

end right_triangle_of_altitude_ratios_l5_5737


namespace alice_cookie_fills_l5_5515

theorem alice_cookie_fills :
  (∀ (a b : ℚ), a = 3 + (3/4) ∧ b = 1/3 → (a / b) = 12) :=
sorry

end alice_cookie_fills_l5_5515


namespace compare_abc_case1_compare_abc_case2_compare_abc_case3_l5_5047

variable (a : ℝ)
variable (b : ℝ := (1 / 2) * (a + 3 / a))
variable (c : ℝ := (1 / 2) * (b + 3 / b))

-- First condition: if \(a > \sqrt{3}\), then \(a > b > c\)
theorem compare_abc_case1 (h1 : a > 0) (h2 : a > Real.sqrt 3) : a > b ∧ b > c := sorry

-- Second condition: if \(a = \sqrt{3}\), then \(a = b = c\)
theorem compare_abc_case2 (h1 : a > 0) (h2 : a = Real.sqrt 3) : a = b ∧ b = c := sorry

-- Third condition: if \(0 < a < \sqrt{3}\), then \(a < c < b\)
theorem compare_abc_case3 (h1 : a > 0) (h2 : a < Real.sqrt 3) : a < c ∧ c < b := sorry

end compare_abc_case1_compare_abc_case2_compare_abc_case3_l5_5047


namespace find_DG_l5_5466

theorem find_DG (a b S k l DG BC : ℕ) (h1: S = 17 * (a + b)) (h2: S % a = 0) (h3: S % b = 0) (h4: a = S / k) (h5: b = S / l) (h6: BC = 17) (h7: (k - 17) * (l - 17) = 289) : DG = 306 :=
by
  sorry

end find_DG_l5_5466


namespace inequality_x_n_l5_5804

theorem inequality_x_n (x : ℝ) (n : ℕ) (hx : |x| < 1) (hn : n ≥ 2) : (1 - x)^n + (1 + x)^n < 2^n := 
sorry

end inequality_x_n_l5_5804


namespace normal_distribution_95_conf_interval_l5_5483

noncomputable def normalCDF95 : ℝ := 1.96

theorem normal_distribution_95_conf_interval :
  ∀ (X : ℝ), 
    (normalPDF X 16 2) →
    ∀ x : set ℝ, x = { y : ℝ | 12.08 ≤ y ∧ y ≤ 19.92 } →
    ∃ y : ℝ, μ = 16 ∧ σ = 2 → normPis X [12.08, 19.92] = 0.95 := 
by
  sorry

end normal_distribution_95_conf_interval_l5_5483


namespace a_4_is_zero_l5_5562

def a_n (n : ℕ) : ℕ := n^2 - 2*n - 8

theorem a_4_is_zero : a_n 4 = 0 := 
by
  sorry

end a_4_is_zero_l5_5562


namespace fraction_spent_by_Rica_is_one_fifth_l5_5607

-- Define the conditions
variable (totalPrizeMoney : ℝ) (fractionReceived : ℝ) (amountLeft : ℝ)
variable (h1 : totalPrizeMoney = 1000) (h2 : fractionReceived = 3 / 8) (h3 : amountLeft = 300)

-- Define Rica's original prize money
noncomputable def RicaOriginalPrizeMoney (totalPrizeMoney fractionReceived : ℝ) : ℝ :=
  fractionReceived * totalPrizeMoney

-- Define amount spent by Rica
noncomputable def AmountSpent (originalPrizeMoney amountLeft : ℝ) : ℝ :=
  originalPrizeMoney - amountLeft

-- Define the fraction of prize money spent by Rica
noncomputable def FractionSpent (amountSpent originalPrizeMoney : ℝ) : ℝ :=
  amountSpent / originalPrizeMoney

-- Main theorem to prove
theorem fraction_spent_by_Rica_is_one_fifth :
  let totalPrizeMoney := 1000
  let fractionReceived := 3 / 8
  let amountLeft := 300
  let RicaOriginalPrizeMoney := fractionReceived * totalPrizeMoney
  let AmountSpent := RicaOriginalPrizeMoney - amountLeft
  let FractionSpent := AmountSpent / RicaOriginalPrizeMoney
  FractionSpent = 1 / 5 :=
by {
  -- Proof details are omitted as per instructions
  sorry
}

end fraction_spent_by_Rica_is_one_fifth_l5_5607


namespace multiplier_for_deans_height_l5_5729

theorem multiplier_for_deans_height (h_R : ℕ) (h_R_eq : h_R = 13) (d : ℕ) (d_eq : d = 255) (h_D : ℕ) (h_D_eq : h_D = h_R + 4) : 
  d / h_D = 15 := by
  sorry

end multiplier_for_deans_height_l5_5729


namespace units_digit_sum_42_4_24_4_l5_5761

theorem units_digit_sum_42_4_24_4 : (42^4 + 24^4) % 10 = 2 := 
by
  sorry

end units_digit_sum_42_4_24_4_l5_5761


namespace inradius_of_triangle_l5_5739

theorem inradius_of_triangle (p A r : ℝ) (h1 : p = 20) (h2 : A = 25) : r = 2.5 :=
sorry

end inradius_of_triangle_l5_5739


namespace vector_dot_product_l5_5045

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 2)

-- Define the operation to calculate (a + 2b)
def two_b : ℝ × ℝ := (2 * b.1, 2 * b.2)
def a_plus_2b : ℝ × ℝ := (a.1 + two_b.1, a.2 + two_b.2)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- State the theorem
theorem vector_dot_product : dot_product a_plus_2b b = 14 := by
  sorry

end vector_dot_product_l5_5045


namespace train_length_l5_5021

/-- Given a train that can cross an electric pole in 15 seconds and has a speed of 72 km/h, prove that the length of the train is 300 meters. -/
theorem train_length 
  (time_to_cross_pole : ℝ)
  (train_speed_kmh : ℝ)
  (h1 : time_to_cross_pole = 15)
  (h2 : train_speed_kmh = 72)
  : (train_speed_kmh * 1000 / 3600) * time_to_cross_pole = 300 := 
by
  -- Proof goes here
  sorry

end train_length_l5_5021


namespace inequality_proof_l5_5679

variable {a b c : ℝ}

theorem inequality_proof (ha : a > 0) (hb : b > 0) (hc : c > 0) (h₁ : a ≠ b) (h₂ : a ≠ c) (h₃ : b ≠ c) :
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 :=
by
  sorry

end inequality_proof_l5_5679


namespace math_problem_l5_5226

theorem math_problem 
  (a b : ℂ) (n : ℕ) (h1 : a + b = 0) (h2 : a ≠ 0) : 
  a^(2*n + 1) + b^(2*n + 1) = 0 := 
by 
  sorry

end math_problem_l5_5226


namespace solve_for_x_l5_5081

theorem solve_for_x :
  ∀ x : ℝ, (1 / 6 + 7 / x = 15 / x + 1 / 15 + 2) → x = -80 / 19 :=
by
  intros x h
  sorry

end solve_for_x_l5_5081


namespace difference_of_squares_401_399_l5_5178

theorem difference_of_squares_401_399 : 401^2 - 399^2 = 1600 :=
by
  sorry

end difference_of_squares_401_399_l5_5178


namespace hotel_total_towels_l5_5776

theorem hotel_total_towels :
  let rooms_A := 25
  let rooms_B := 30
  let rooms_C := 15
  let members_per_room_A := 5
  let members_per_room_B := 6
  let members_per_room_C := 4
  let towels_per_member_A := 3
  let towels_per_member_B := 2
  let towels_per_member_C := 4
  (rooms_A * members_per_room_A * towels_per_member_A) +
  (rooms_B * members_per_room_B * towels_per_member_B) +
  (rooms_C * members_per_room_C * towels_per_member_C) = 975
:= by
  sorry

end hotel_total_towels_l5_5776


namespace candy_left_l5_5039

-- Define the number of candies each sibling has
def debbyCandy : ℕ := 32
def sisterCandy : ℕ := 42
def brotherCandy : ℕ := 48

-- Define the total candies collected
def totalCandy : ℕ := debbyCandy + sisterCandy + brotherCandy

-- Define the number of candies eaten
def eatenCandy : ℕ := 56

-- Define the remaining candies after eating some
def remainingCandy : ℕ := totalCandy - eatenCandy

-- The hypothesis stating the initial condition
theorem candy_left (h1 : debbyCandy = 32) (h2 : sisterCandy = 42) (h3 : brotherCandy = 48) (h4 : eatenCandy = 56) : remainingCandy = 66 :=
by
  -- Proof can be filled in here
  sorry

end candy_left_l5_5039


namespace probability_adjacent_vertices_decagon_l5_5308

theorem probability_adjacent_vertices_decagon : 
  ∀ (decagon : Finset ℕ) (a b : ℕ), 
  decagon.card = 10 → 
  a ∈ decagon → 
  b ∈ decagon → 
  a ≠ b → 
  (P (adjacent a b decagon)) = 2 / 9 :=
by 
  -- we are asserting the probability that two randomly chosen vertices a and b from a decagon are adjacent.
  sorry

end probability_adjacent_vertices_decagon_l5_5308


namespace area_D_meets_sign_l5_5654

-- Definition of conditions as given in the question
def condition_A (mean median : ℝ) : Prop := mean = 3 ∧ median = 4
def condition_B (mean : ℝ) (variance_pos : Prop) : Prop := mean = 1 ∧ variance_pos
def condition_C (median mode : ℝ) : Prop := median = 2 ∧ mode = 3
def condition_D (mean variance : ℝ) : Prop := mean = 2 ∧ variance = 3

-- Theorem stating that Area D satisfies the condition to meet the required sign
theorem area_D_meets_sign (mean variance : ℝ) (h : condition_D mean variance) : 
  (∀ day_increase, day_increase ≤ 7) :=
sorry

end area_D_meets_sign_l5_5654


namespace interest_calculation_l5_5135

variables (P R SI : ℝ) (T : ℕ)

-- Given conditions
def principal := (P = 8)
def rate := (R = 0.05)
def simple_interest := (SI = 4.8)

-- Goal
def time_calculated := (T = 12)

-- Lean statement combining the conditions
theorem interest_calculation : principal P → rate R → simple_interest SI → T = 12 :=
by
  intros hP hR hSI
  sorry

end interest_calculation_l5_5135


namespace find_v2_poly_l5_5627

theorem find_v2_poly (x : ℤ) (v0 v1 v2 : ℤ) 
  (h1 : x = -4)
  (h2 : v0 = 1) 
  (h3 : v1 = v0 * x)
  (h4 : v2 = v1 * x + 6) :
  v2 = 22 :=
by
  -- To be filled with proof (example problem requirement specifies proof is not needed)
  sorry

end find_v2_poly_l5_5627


namespace no_solution_inequality_C_l5_5628

theorem no_solution_inequality_C : ¬∃ x : ℝ, 2 * x - x^2 > 5 := by
  -- There is no need to include the other options in the Lean theorem, as the proof focuses on the condition C directly.
  sorry

end no_solution_inequality_C_l5_5628


namespace sum_of_power_of_2_plus_1_divisible_by_3_iff_odd_l5_5217

theorem sum_of_power_of_2_plus_1_divisible_by_3_iff_odd (n : ℕ) : 
  (3 ∣ (2^n + 1)) ↔ (n % 2 = 1) :=
sorry

end sum_of_power_of_2_plus_1_divisible_by_3_iff_odd_l5_5217


namespace geometric_sequence_S20_l5_5235

-- Define the conditions and target statement
theorem geometric_sequence_S20
  (a : ℕ → ℝ) -- defining the sequence as a function from natural numbers to real numbers
  (q : ℝ) -- common ratio
  (h_pos : ∀ n, a n > 0) -- all terms are positive
  (h_geo : ∀ n, a (n + 1) = q * a n) -- geometric sequence property
  (S : ℕ → ℝ) -- sum function
  (h_S : ∀ n, S n = (a 1 * (1 - q ^ n)) / (1 - q)) -- sum formula for a geometric progression
  (h_S5 : S 5 = 3) -- given S_5 = 3
  (h_S15 : S 15 = 21) -- given S_15 = 21
  : S 20 = 45 := sorry

end geometric_sequence_S20_l5_5235


namespace circle_passing_through_points_l5_5873

noncomputable def circle_equation (D E F : ℝ) : ℝ × ℝ → ℝ :=
λ p, p.1^2 + p.2^2 + D * p.1 + E * p.2 + F

theorem circle_passing_through_points : ∃ D E F : ℝ, 
  circle_equation D E F (0, 0) = 0 ∧
  circle_equation D E F (4, 0) = 0 ∧
  circle_equation D E F (-1, 1) = 0 ∧
  D = -4 ∧ 
  E = -6 ∧ 
  F = 0 :=
begin
  sorry
end

end circle_passing_through_points_l5_5873


namespace trig_relation_l5_5615

theorem trig_relation : (Real.pi/4 < 1) ∧ (1 < Real.pi/2) → Real.tan 1 > Real.sin 1 ∧ Real.sin 1 > Real.cos 1 := 
by 
  intro h
  sorry

end trig_relation_l5_5615


namespace solve_for_x_l5_5070

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 1) (h1 : y = 1 / (4 * x^2 + 2 * x + 1)) : 
  x = 0 ∨ x = -1/2 :=
by
  sorry

end solve_for_x_l5_5070


namespace probability_of_adjacent_vertices_in_decagon_l5_5320

/-- Define the number of vertices in the decagon -/
def num_vertices : ℕ := 10

/-- Define the total number of ways to choose two distinct vertices from the decagon -/
def total_possible_outcomes : ℕ := num_vertices * (num_vertices - 1) / 2

/-- Define the number of favorable outcomes where the two chosen vertices are adjacent -/
def favorable_outcomes : ℕ := num_vertices

/-- Define the probability of selecting two adjacent vertices -/
def probability_adjacent_vertices : ℚ := favorable_outcomes / total_possible_outcomes

/-- The main theorem statement -/
theorem probability_of_adjacent_vertices_in_decagon : probability_adjacent_vertices = 2 / 9 := 
  sorry

end probability_of_adjacent_vertices_in_decagon_l5_5320


namespace fraction_of_students_who_walk_home_l5_5387

theorem fraction_of_students_who_walk_home (bus auto bikes scooters : ℚ) 
  (hbus : bus = 2/5) (hauto : auto = 1/5) 
  (hbikes : bikes = 1/10) (hscooters : scooters = 1/10) : 
  1 - (bus + auto + bikes + scooters) = 1/5 :=
by 
  rw [hbus, hauto, hbikes, hscooters]
  sorry

end fraction_of_students_who_walk_home_l5_5387


namespace find_y_l5_5821

def G (a b c d : ℕ) : ℕ := a^b + c * d

theorem find_y (y : ℕ) (h : G 3 y 5 18 = 500) : y = 6 :=
sorry

end find_y_l5_5821


namespace shifted_sine_monotonically_increasing_l5_5150

noncomputable def shifted_sine_function (x : ℝ) : ℝ :=
  3 * Real.sin (2 * x - (2 * Real.pi / 3))

theorem shifted_sine_monotonically_increasing :
  ∀ x y : ℝ, (x ∈ Set.Icc (Real.pi / 12) (7 * Real.pi / 12)) → (y ∈ Set.Icc (Real.pi / 12) (7 * Real.pi / 12)) → x < y → shifted_sine_function x < shifted_sine_function y :=
by
  sorry

end shifted_sine_monotonically_increasing_l5_5150


namespace TotalLaddersClimbedInCentimeters_l5_5105

def keaton_ladder_height := 50  -- height of Keaton's ladder in meters
def keaton_climbs := 30  -- number of times Keaton climbs the ladder

def reece_ladder_height := keaton_ladder_height - 6  -- height of Reece's ladder in meters
def reece_climbs := 25  -- number of times Reece climbs the ladder

def total_meters_climbed := (keaton_ladder_height * keaton_climbs) + (reece_ladder_height * reece_climbs)

def total_cm_climbed := total_meters_climbed * 100

theorem TotalLaddersClimbedInCentimeters :
  total_cm_climbed = 260000 :=
by
  sorry

end TotalLaddersClimbedInCentimeters_l5_5105


namespace solve_equation_error_step_l5_5682

theorem solve_equation_error_step 
  (equation : ∀ x : ℝ, (x - 1) / 2 + 1 = (2 * x + 1) / 3) :
  ∃ (step : ℕ), step = 1 ∧
  let s1 := ((x - 1) / 2 + 1) * 6;
  ∀ (x : ℝ), s1 ≠ (((2 * x + 1) / 3) * 6) :=
by
  sorry

end solve_equation_error_step_l5_5682


namespace remainder_is_3x_l5_5179

variable (p : Polynomial ℚ)

theorem remainder_is_3x (h1 : p.eval 1 = 3) (h4 : p.eval 4 = 12) : ∃ q : Polynomial ℚ, p = (X - 1) * (X - 4) * q + 3 * X := 
  sorry

end remainder_is_3x_l5_5179


namespace M_inter_P_eq_l5_5062

-- Define the sets M and P
def M : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ 4 * x + y = 6 }
def P : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ 3 * x + 2 * y = 7 }

-- Prove that the intersection of M and P is {(1, 2)}
theorem M_inter_P_eq : M ∩ P = { (1, 2) } := 
by 
sorry

end M_inter_P_eq_l5_5062


namespace cars_with_neither_feature_l5_5720

theorem cars_with_neither_feature 
  (total_cars : ℕ) 
  (power_steering : ℕ) 
  (power_windows : ℕ) 
  (both_features : ℕ) 
  (h1 : total_cars = 65) 
  (h2 : power_steering = 45) 
  (h3 : power_windows = 25) 
  (h4 : both_features = 17)
  : total_cars - (power_steering + power_windows - both_features) = 12 :=
by
  sorry

end cars_with_neither_feature_l5_5720


namespace smallest_value_l5_5454

theorem smallest_value 
  (x1 x2 x3 : ℝ) 
  (hx1 : 0 < x1) 
  (hx2 : 0 < x2) 
  (hx3 : 0 < x3)
  (h : 2 * x1 + 3 * x2 + 4 * x3 = 100) : 
  x1^2 + x2^2 + x3^2 = 10000 / 29 := by
  sorry

end smallest_value_l5_5454


namespace scientific_notation_600_million_l5_5002

theorem scientific_notation_600_million : (600000000 : ℝ) = 6 * 10^8 := 
by 
  -- Insert the proof here
  sorry

end scientific_notation_600_million_l5_5002


namespace speed_of_water_is_10_l5_5777

/-- Define the conditions -/
def swimming_speed_in_still_water : ℝ := 12 -- km/h
def time_to_swim_against_current : ℝ := 4 -- hours
def distance_against_current : ℝ := 8 -- km

/-- Define the effective speed against the current and the proof goal -/
def speed_of_water (v : ℝ) : Prop :=
  (swimming_speed_in_still_water - v) = distance_against_current / time_to_swim_against_current

theorem speed_of_water_is_10 : speed_of_water 10 :=
by
  unfold speed_of_water
  sorry

end speed_of_water_is_10_l5_5777


namespace Alice_fills_needed_l5_5518

def cups_needed : ℚ := 15/4
def cup_capacity : ℚ := 1/3
def fills_needed : ℚ := 12

theorem Alice_fills_needed : (cups_needed / cup_capacity).ceil = fills_needed := by
  -- Proof is omitted with sorry
  sorry

end Alice_fills_needed_l5_5518


namespace men_seated_on_bus_l5_5141

theorem men_seated_on_bus (total_passengers : ℕ) (women_fraction men_standing_fraction : ℚ)
  (h_total : total_passengers = 48)
  (h_women_fraction : women_fraction = 2/3)
  (h_men_standing_fraction : men_standing_fraction = 1/8) :
  let women := (total_passengers : ℚ) * women_fraction,
      men := (total_passengers : ℚ) - women,
      men_standing := men * men_standing_fraction,
      men_seated := men - men_standing in
  men_seated = 14 :=
by
  sorry

end men_seated_on_bus_l5_5141


namespace triangle_inradius_is_2_5_l5_5741

variable (A : ℝ) (p : ℝ) (r : ℝ)

def triangle_has_given_inradius (A p : ℝ) : Prop :=
  A = r * p / 2

theorem triangle_inradius_is_2_5 (h₁ : A = 25) (h₂ : p = 20) :
  triangle_has_given_inradius A p r → r = 2.5 := sorry

end triangle_inradius_is_2_5_l5_5741


namespace law_of_sines_l5_5860

theorem law_of_sines (a b c : ℝ) (A B C : ℝ) (R : ℝ) 
  (hA : a = 2 * R * Real.sin A)
  (hEquilateral1 : b = 2 * R * Real.sin B)
  (hEquilateral2 : c = 2 * R * Real.sin C):
  (a / Real.sin A) = (b / Real.sin B) ∧ 
  (b / Real.sin B) = (c / Real.sin C) ∧ 
  (c / Real.sin C) = 2 * R :=
by
  sorry

end law_of_sines_l5_5860


namespace volume_common_solid_hemisphere_cone_l5_5339

noncomputable def volume_common_solid (r : ℝ) : ℝ := 
  let V_1 := (2/3) * Real.pi * (r^3 - (3 * r / 5)^3)
  let V_2 := Real.pi * ((r / 5)^2) * (r - (r / 15))
  V_1 + V_2

theorem volume_common_solid_hemisphere_cone (r : ℝ) :
  volume_common_solid r = (14 * Real.pi * r^3) / 25 := 
by
  sorry

end volume_common_solid_hemisphere_cone_l5_5339


namespace ticket_price_increase_one_day_later_l5_5988

noncomputable def ticket_price : ℝ := 1050
noncomputable def days_before_departure : ℕ := 14
noncomputable def daily_increase_rate : ℝ := 0.05

theorem ticket_price_increase_one_day_later :
  ∀ (price : ℝ) (days : ℕ) (rate : ℝ), price = ticket_price → days = days_before_departure → rate = daily_increase_rate →
  price * rate = 52.50 :=
by
  intros price days rate hprice hdays hrate
  rw [hprice, hrate]
  exact sorry

end ticket_price_increase_one_day_later_l5_5988


namespace gcd_2025_2070_l5_5340

theorem gcd_2025_2070 : Nat.gcd 2025 2070 = 45 := by
  sorry

end gcd_2025_2070_l5_5340


namespace ratio_proof_l5_5471

theorem ratio_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (4 * x + y) / (x - 4 * y) = 3) :
    (x + 4 * y) / (4 * x - y) = 9 / 53 :=
  sorry

end ratio_proof_l5_5471


namespace number_of_selected_in_interval_l5_5015

-- Definitions and conditions based on the problem statement
def total_employees : ℕ := 840
def sample_size : ℕ := 42
def systematic_sampling_interval : ℕ := total_employees / sample_size
def interval_start : ℕ := 481
def interval_end : ℕ := 720

-- Main theorem statement that we need to prove
theorem number_of_selected_in_interval :
  let selected_in_interval : ℕ := (interval_end - interval_start + 1) / systematic_sampling_interval
  selected_in_interval = 12 := by
  sorry

end number_of_selected_in_interval_l5_5015


namespace cultivated_land_percentage_l5_5882

theorem cultivated_land_percentage : 
  let water : ℚ := 7 / 10
  let land : ℚ := 3 / 10
  let deserts_or_ice : ℚ := 2 / 5
  let pastures_forests_mountains : ℚ := 1 / 3
  ∃ cultivated_land_fraction : ℚ, 
    cultivated_land_fraction = land * (1 - deserts_or_ice - pastures_forests_mountains) ∧ 
    (cultivated_land_fraction * 100) = 8 := 
by
  sorry

end cultivated_land_percentage_l5_5882


namespace students_in_johnsons_class_l5_5856

-- Define the conditions as constants/variables
def studentsInFinleysClass : ℕ := 24
def studentsAdditionalInJohnsonsClass : ℕ := 10

-- State the problem as a theorem
theorem students_in_johnsons_class : 
  let halfFinleysClass := studentsInFinleysClass / 2
  let johnsonsClass := halfFinleysClass + studentsAdditionalInJohnsonsClass
  johnsonsClass = 22 :=
by
  sorry

end students_in_johnsons_class_l5_5856


namespace solution_set_I_range_of_a_l5_5561

-- Define the function f(x) = |x + a| - |x + 1|
def f (x a : ℝ) : ℝ := abs (x + a) - abs (x + 1)

-- Part (I)
theorem solution_set_I (a : ℝ) : 
  (f a a > 1) ↔ (a < -2/3 ∨ a > 2) := by
  sorry

-- Part (II)
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≤ 2 * a) ↔ (a ≥ 1/3) := by
  sorry

end solution_set_I_range_of_a_l5_5561


namespace find_side_a_l5_5835

noncomputable def side_a (b : ℝ) (A : ℝ) (S : ℝ) : ℝ :=
  2 * S / (b * Real.sin A)

theorem find_side_a :
  let b := 2
  let A := Real.pi * 2 / 3 -- 120 degrees in radians
  let S := 2 * Real.sqrt 3
  side_a b A S = 4 :=
by
  let b := 2
  let A := Real.pi * 2 / 3
  let S := 2 * Real.sqrt 3
  show side_a b A S = 4
  sorry

end find_side_a_l5_5835


namespace lowest_possible_number_of_students_l5_5365

theorem lowest_possible_number_of_students : ∃ n : ℕ, (n > 0) ∧ (∃ k1 : ℕ, n = 10 * k1) ∧ (∃ k2 : ℕ, n = 24 * k2) ∧ n = 120 :=
by
  sorry

end lowest_possible_number_of_students_l5_5365


namespace difference_ne_1998_l5_5500

-- Define the function f(n) = n^2 + 4n
def f (n : ℕ) : ℕ := n^2 + 4 * n

-- Statement: For all natural numbers n and m, the difference f(n) - f(m) is not 1998
theorem difference_ne_1998 (n m : ℕ) : f n - f m ≠ 1998 := 
by {
  sorry
}

end difference_ne_1998_l5_5500


namespace Eve_total_running_distance_l5_5798

def Eve_walked_distance := 0.6

def Eve_ran_distance := Eve_walked_distance + 0.1

theorem Eve_total_running_distance : Eve_ran_distance = 0.7 := 
by sorry

end Eve_total_running_distance_l5_5798


namespace calculate_expression_l5_5526

theorem calculate_expression :
  -1 ^ 4 + ((-1 / 2) ^ 2 * |(-5 + 3)|) / ((-1 / 2) ^ 3) = -5 := by
  sorry

end calculate_expression_l5_5526


namespace locus_of_point_M_l5_5813

open Real

def distance (x y: ℝ × ℝ): ℝ :=
  ((x.1 - y.1)^2 + (x.2 - y.2)^2)^(1/2)

theorem locus_of_point_M :
  (∀ (M : ℝ × ℝ), 
     distance M (2, 0) + 1 = abs (M.1 + 3)) 
  → ∀ (M : ℝ × ℝ), M.2^2 = 8 * M.1 :=
sorry

end locus_of_point_M_l5_5813


namespace image_relative_velocity_l5_5520

-- Definitions of the constants
def f : ℝ := 0.2
def x : ℝ := 0.5
def vt : ℝ := 3

-- Lens equation
def lens_equation (f x y : ℝ) : Prop :=
  (1 / x) + (1 / y) = 1 / f

-- Image distance
noncomputable def y (f x : ℝ) : ℝ :=
  1 / (1 / f - 1 / x)

-- Derivative of y with respect to x
noncomputable def dy_dx (f x : ℝ) : ℝ :=
  (f^2) / (x - f)^2

-- Image velocity
noncomputable def vk (vt dy_dx : ℝ) : ℝ :=
  vt * dy_dx

-- Relative velocity
noncomputable def v_rel (vt vk : ℝ) : ℝ :=
  vk - vt

-- Theorem to prove the relative velocity
theorem image_relative_velocity : v_rel vt (vk vt (dy_dx f x)) = -5 / 3 := 
by
  sorry

end image_relative_velocity_l5_5520


namespace arithmetic_sequence_30th_term_l5_5357

theorem arithmetic_sequence_30th_term :
  let a₁ := 3
  let d := 4
  let n := 30
  a₁ + (n - 1) * d = 119 :=
by
  let a₁ := 3
  let d := 4
  let n := 30
  show a₁ + (n - 1) * d = 119
  sorry

end arithmetic_sequence_30th_term_l5_5357


namespace find_a_plus_c_l5_5849

open Function

noncomputable def parabolas_intersect (a b c d : ℝ) : Prop :=
  (-(3 - a)^2 + b = 6) ∧
  ((3 - c)^2 + d = 6) ∧
  (-(9 - a)^2 + b = 0) ∧
  ((9 - c)^2 + d = 0)

theorem find_a_plus_c (a b c d : ℝ) (h : parabolas_intersect a b c d) : a + c = 12 :=
by {
  sorry
}

end find_a_plus_c_l5_5849


namespace seventh_root_binomial_expansion_l5_5407

theorem seventh_root_binomial_expansion : 
  (∃ (n : ℕ), n = 137858491849 ∧ (∃ (k : ℕ), n = (10 + 1) ^ k)) →
  (∃ a, a = 11 ∧ 11 ^ 7 = 137858491849) := 
by {
  sorry 
}

end seventh_root_binomial_expansion_l5_5407


namespace slope_of_parallel_line_l5_5933

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l5_5933


namespace arithmetic_sequence_30th_term_l5_5355

theorem arithmetic_sequence_30th_term :
  let a₁ := 3
  let d := 4
  let n := 30
  a₁ + (n - 1) * d = 119 :=
by
  let a₁ := 3
  let d := 4
  let n := 30
  show a₁ + (n - 1) * d = 119
  sorry

end arithmetic_sequence_30th_term_l5_5355


namespace fraction_second_year_students_l5_5499

theorem fraction_second_year_students
  (total_students : ℕ)
  (third_year_students : ℕ)
  (second_year_students : ℕ)
  (h1 : third_year_students = total_students * 30 / 100)
  (h2 : second_year_students = total_students * 10 / 100) :
  (second_year_students : ℚ) / (total_students - third_year_students) = 1 / 7 := by
  sorry

end fraction_second_year_students_l5_5499


namespace leigh_path_length_l5_5224

theorem leigh_path_length :
  let north := 10
  let south := 40
  let west := 60
  let east := 20
  let net_south := south - north
  let net_west := west - east
  let distance := Real.sqrt (net_south^2 + net_west^2)
  distance = 50 := 
by sorry

end leigh_path_length_l5_5224


namespace sum_of_edges_of_rectangular_solid_l5_5620

theorem sum_of_edges_of_rectangular_solid 
  (a r : ℝ) 
  (volume_eq : (a / r) * a * (a * r) = 343) 
  (surface_area_eq : 2 * ((a^2 / r) + (a^2 * r) + a^2) = 294) 
  (gp : a / r > 0 ∧ a > 0 ∧ a * r > 0) :
  4 * ((a / r) + a + (a * r)) = 84 :=
by
  sorry

end sum_of_edges_of_rectangular_solid_l5_5620


namespace commencement_addresses_sum_l5_5259

noncomputable def addresses (S H L : ℕ) := 40

theorem commencement_addresses_sum
  (S H L : ℕ)
  (h1 : S = 12)
  (h2 : S = 2 * H)
  (h3 : L = S + 10) :
  S + H + L = addresses S H L :=
by
  sorry

end commencement_addresses_sum_l5_5259


namespace min_voters_for_tall_24_l5_5087

/-
There are 105 voters divided into 5 districts, each district divided into 7 sections, with each section having 3 voters.
A section is won by a majority vote. A district is won by a majority of sections. The contest is won by a majority of districts.
Tall won the contest. Prove that the minimum number of voters who could have voted for Tall is 24.
-/
noncomputable def min_voters_for_tall (total_voters districts sections voters_per_section : ℕ) (sections_needed_to_win_district districts_needed_to_win_contest : ℕ) : ℕ :=
  let voters_needed_per_section := voters_per_section / 2 + 1
  sections_needed_to_win_district * districts_needed_to_win_contest * voters_needed_per_section

theorem min_voters_for_tall_24 :
  min_voters_for_tall 105 5 7 3 4 3 = 24 :=
sorry

end min_voters_for_tall_24_l5_5087


namespace sum_inf_series_l5_5400

theorem sum_inf_series :
  (\sum_{n=1}^{\infty} \frac{(4 * n) - 3}{3^n}) = 1 :=
by
  sorry

end sum_inf_series_l5_5400


namespace Onum_Lake_more_trout_l5_5117

theorem Onum_Lake_more_trout (O B R : ℕ) (hB : B = 75) (hR : R = O / 2) (hAvg : (O + B + R) / 3 = 75) : O - B = 25 :=
by
  sorry

end Onum_Lake_more_trout_l5_5117


namespace find_ab_l5_5205

theorem find_ab (a b : ℝ) 
  (period_cond : (π / b) = (2 * π / 5)) 
  (point_cond : a * Real.tan (5 * (π / 10) / 2) = 1) :
  a * b = 5 / 2 :=
sorry

end find_ab_l5_5205


namespace cannot_contain_2003_0_l5_5692

noncomputable def point_not_on_line (m b : ℝ) (h : m * b < 0) : Prop :=
  ∀ y : ℝ, ¬(0 = 2003 * m + b)

-- Prove that if m and b are real numbers and mb < 0, the line y = mx + b
-- cannot contain the point (2003, 0).
theorem cannot_contain_2003_0 (m b : ℝ) (h : m * b < 0) : point_not_on_line m b h :=
by
  sorry

end cannot_contain_2003_0_l5_5692


namespace min_rectangles_needed_l5_5903

theorem min_rectangles_needed 
  (type1_corners type2_corners : ℕ)
  (rectangles_cover : ℕ → ℕ)
  (h1 : type1_corners = 12)
  (h2 : type2_corners = 12)
  (h3 : ∀ n, rectangles_cover (3 * n) = n) : 
  (rectangles_cover type2_corners) + (rectangles_cover type1_corners) = 12 := 
sorry

end min_rectangles_needed_l5_5903


namespace remainder_of_sum_of_primes_mod_eighth_prime_l5_5159

def sum_first_seven_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13 + 17

def eighth_prime : ℕ := 19

theorem remainder_of_sum_of_primes_mod_eighth_prime : sum_first_seven_primes % eighth_prime = 1 := by
  sorry

end remainder_of_sum_of_primes_mod_eighth_prime_l5_5159


namespace length_of_EC_l5_5003

theorem length_of_EC
  (AB CD AC : ℝ)
  (h1 : AB = 3 * CD)
  (h2 : AC = 15)
  (EC : ℝ)
  (h3 : AC = 4 * EC)
  : EC = 15 / 4 := 
sorry

end length_of_EC_l5_5003


namespace largest_two_digit_with_remainder_2_l5_5219

theorem largest_two_digit_with_remainder_2 (n : ℕ) :
  10 ≤ n ∧ n ≤ 99 ∧ n % 13 = 2 → n = 93 :=
by
  intro h
  sorry

end largest_two_digit_with_remainder_2_l5_5219


namespace frank_composes_problems_l5_5388

theorem frank_composes_problems (bill_problems : ℕ) (ryan_problems : ℕ) (frank_problems : ℕ) 
  (h1 : bill_problems = 20)
  (h2 : ryan_problems = 2 * bill_problems)
  (h3 : frank_problems = 3 * ryan_problems)
  : frank_problems / 4 = 30 :=
by
  sorry

end frank_composes_problems_l5_5388


namespace expression_for_x_l5_5281

variable (A B C x y : ℝ)

-- Conditions
def condition1 := A > C
def condition2 := C > B
def condition3 := B > 0
def condition4 := C = (1 + y / 100) * B
def condition5 := A = (1 + x / 100) * C

-- The theorem
theorem expression_for_x (h1 : condition1 A C) (h2 : condition2 C B) (h3 : condition3 B) (h4 : condition4 B C y) (h5 : condition5 A C x) :
    x = 100 * ((100 * (A - B)) / (100 + y)) :=
sorry

end expression_for_x_l5_5281


namespace jack_piggy_bank_after_8_weeks_l5_5450

-- Conditions as definitions
def initial_amount : ℕ := 43
def weekly_allowance : ℕ := 10
def saved_fraction (x : ℕ) : ℕ := x / 2
def duration : ℕ := 8

-- Mathematical equivalent proof problem
theorem jack_piggy_bank_after_8_weeks : initial_amount + (duration * saved_fraction weekly_allowance) = 83 := by
  sorry

end jack_piggy_bank_after_8_weeks_l5_5450


namespace ab_condition_l5_5669

theorem ab_condition (a b : ℝ) : ¬((a + b > 1 → a^2 + b^2 > 1) ∧ (a^2 + b^2 > 1 → a + b > 1)) :=
by {
  -- This proof problem states that the condition "a + b > 1" is neither sufficient nor necessary for "a^2 + b^2 > 1".
  sorry
}

end ab_condition_l5_5669


namespace equiangular_hexagon_sides_l5_5121

variable {a b c d e f : ℝ}

-- Definition of the equiangular hexagon condition
def equiangular_hexagon (a b c d e f : ℝ) := true

theorem equiangular_hexagon_sides (h : equiangular_hexagon a b c d e f) :
  a - d = e - b ∧ e - b = c - f :=
by
  sorry

end equiangular_hexagon_sides_l5_5121


namespace am_gm_inequality_l5_5278

variable (a : ℝ) (h : a > 0) -- Variables and condition

theorem am_gm_inequality (a : ℝ) (h : a > 0) : a + 1 / a ≥ 2 := 
sorry -- Proof is not provided according to instructions.

end am_gm_inequality_l5_5278


namespace perfect_square_fraction_l5_5532

theorem perfect_square_fraction (n : ℤ) : 
  n < 30 ∧ ∃ k : ℤ, (n / (30 - n)) = k^2 → ∃ cnt : ℕ, cnt = 4 :=
  by
  sorry

end perfect_square_fraction_l5_5532


namespace number_of_ordered_pairs_l5_5847

noncomputable def max (x y : ℕ) : ℕ := if x > y then x else y

def valid_pair_count (k : ℕ) : ℕ := 2 * k + 1

def pairs_count (a b : ℕ) : ℕ := 
  valid_pair_count 5 * valid_pair_count 3 * valid_pair_count 2 * valid_pair_count 1

theorem number_of_ordered_pairs : pairs_count 2 3 = 1155 := 
  sorry

end number_of_ordered_pairs_l5_5847


namespace vector_calculation_l5_5044

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)

def vector_operation (a b : ℝ × ℝ) : ℝ × ℝ :=
(3 * a.1 - 2 * b.1, 3 * a.2 - 2 * b.2)

theorem vector_calculation : vector_operation vector_a vector_b = (1, 5) :=
by sorry

end vector_calculation_l5_5044


namespace slope_of_parallel_line_l5_5912

theorem slope_of_parallel_line (a b c : ℝ) (h : 3 * a - 6 * b = 12) :
  ∃ m, m = 1 / 2 := 
sorry

end slope_of_parallel_line_l5_5912


namespace jon_monthly_earnings_l5_5843

def earnings_per_person : ℝ := 0.10
def visits_per_hour : ℕ := 50
def hours_per_day : ℕ := 24
def days_per_month : ℕ := 30

theorem jon_monthly_earnings : 
  (earnings_per_person * visits_per_hour * hours_per_day * days_per_month) = 3600 :=
by
  sorry

end jon_monthly_earnings_l5_5843


namespace k_l_m_n_values_l5_5270

theorem k_l_m_n_values (k l m n : ℕ) (hk : 0 < k) (hl : 0 < l) (hm : 0 < m) (hn : 0 < n)
  (hklmn : k + l + m + n = k * m) (hln : k + l + m + n = l * n) :
  k + l + m + n = 16 ∨ k + l + m + n = 18 ∨ k + l + m + n = 24 ∨ k + l + m + n = 30 :=
sorry

end k_l_m_n_values_l5_5270
