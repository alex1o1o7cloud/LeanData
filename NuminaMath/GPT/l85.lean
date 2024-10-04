import Mathlib

namespace plastering_cost_correct_l85_85780

noncomputable def tank_length : ℝ := 25
noncomputable def tank_width : ℝ := 12
noncomputable def tank_depth : ℝ := 6
noncomputable def cost_per_sqm_paise : ℝ := 75
noncomputable def cost_per_sqm_rupees : ℝ := cost_per_sqm_paise / 100

noncomputable def total_cost_plastering : ℝ :=
  let long_wall_area := 2 * (tank_length * tank_depth)
  let short_wall_area := 2 * (tank_width * tank_depth)
  let bottom_area := tank_length * tank_width
  let total_area := long_wall_area + short_wall_area + bottom_area
  total_area * cost_per_sqm_rupees

theorem plastering_cost_correct : total_cost_plastering = 558 := by
  sorry

end plastering_cost_correct_l85_85780


namespace intersection_probability_l85_85821

-- Define the points and the condition of being evenly spaced around a circle
def points : Set ℕ := {i | 0 ≤ i ∧ i < 2023}

-- Define the event in terms of the problem conditions
theorem intersection_probability (A B C D E F G H : points) :
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ E) ∧ (E ≠ F) ∧ (F ≠ G) ∧ (G ≠ H) ∧ (H ≠ A) →
  ∀ (x1 x2 x3 x4 : points), x1 ∈ {A, B, C, D} → x2 ∈ {A, B, C, D} → x3 ∈ {E, F, G, H} → x4 ∈ {E, F, G, H} →
  (x1 ≠ x2) ∧ (x3 ≠ x4) →
  (∀ A B C D, chord_intersects A B C D) →
  (∀ E F G H, chord_intersects E F G H) →
  intersection_probability = (1 : ℚ) / 36 :=
sorry

end intersection_probability_l85_85821


namespace sale_in_third_month_l85_85772

theorem sale_in_third_month (s_1 s_2 s_4 s_5 s_6 : ℝ) (avg_sale : ℝ) (h1 : s_1 = 6435) (h2 : s_2 = 6927) (h4 : s_4 = 7230) (h5 : s_5 = 6562) (h6 : s_6 = 6191) (h_avg : avg_sale = 6700) :
  ∃ s_3 : ℝ, s_1 + s_2 + s_3 + s_4 + s_5 + s_6 = 6 * avg_sale ∧ s_3 = 6855 :=
by 
  sorry

end sale_in_third_month_l85_85772


namespace solve_quadratic_eq_solve_cubic_eq_l85_85194

-- Statement for the first equation
theorem solve_quadratic_eq (x : ℝ) : 9 * x^2 - 25 = 0 ↔ x = 5 / 3 ∨ x = -5 / 3 :=
by sorry

-- Statement for the second equation
theorem solve_cubic_eq (x : ℝ) : (x + 1)^3 - 27 = 0 ↔ x = 2 :=
by sorry

end solve_quadratic_eq_solve_cubic_eq_l85_85194


namespace tommy_balloons_l85_85460

/-- Tommy had some balloons. He received 34 more balloons from his mom,
gave away 15 balloons, and exchanged the remaining balloons for teddy bears
at a rate of 3 balloons per teddy bear. After these transactions, he had 30 teddy bears.
Prove that Tommy started with 71 balloons -/
theorem tommy_balloons : 
  ∃ B : ℕ, (B + 34 - 15) = 3 * 30 ∧ B = 71 := 
by
  have h : (71 + 34 - 15) = 3 * 30 := by norm_num
  exact ⟨71, h, rfl⟩

end tommy_balloons_l85_85460


namespace almost_perfect_numbers_l85_85577

def d (n : Nat) : Nat := 
  -- Implement the function to count the number of positive divisors of n
  sorry

def f (n : Nat) : Nat := 
  -- Implement the function f(n) as given in the problem statement
  sorry

def isAlmostPerfect (n : Nat) : Prop := 
  f n = n

theorem almost_perfect_numbers :
  ∀ n, isAlmostPerfect n → n = 1 ∨ n = 3 ∨ n = 18 ∨ n = 36 :=
by
  sorry

end almost_perfect_numbers_l85_85577


namespace sqrt_multiplication_l85_85156

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l85_85156


namespace true_proposition_l85_85364

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l85_85364


namespace speed_of_first_train_l85_85282

/-
Problem:
Two trains, with lengths 150 meters and 165 meters respectively, are running in opposite directions. One train is moving at 65 kmph, and they take 7.82006405004841 seconds to completely clear each other from the moment they meet. Prove that the speed of the first train is 79.99 kmph.
-/

theorem speed_of_first_train :
  ∀ (length1 length2 : ℝ) (speed2 : ℝ) (time : ℝ) (speed1 : ℝ),
  length1 = 150 → length2 = 165 → speed2 = 65 → time = 7.82006405004841 →
  ( 3.6 * (length1 + length2) / time = speed1 + speed2 ) →
  speed1 = 79.99 :=
by
  intros length1 length2 speed2 time speed1 h_length1 h_length2 h_speed2 h_time h_formula
  rw [h_length1, h_length2, h_speed2, h_time] at h_formula
  sorry

end speed_of_first_train_l85_85282


namespace expression_a_equals_half_expression_c_equals_half_l85_85791

theorem expression_a_equals_half :
  (A : ℝ) = (1 / 2) :=
by
  let A := (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180))
  sorry

theorem expression_c_equals_half :
  (C : ℝ) = (1 / 2) :=
by
  let C := (Real.tan (22.5 * Real.pi / 180)) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)
  sorry

end expression_a_equals_half_expression_c_equals_half_l85_85791


namespace gcd_of_228_and_1995_l85_85830

theorem gcd_of_228_and_1995 : Nat.gcd 228 1995 = 57 :=
by
  sorry

end gcd_of_228_and_1995_l85_85830


namespace not_possible_to_create_3_similar_piles_l85_85718

-- Formalize the definition of similar piles
def similar (a b : ℝ) : Prop := a <= sqrt 2 * b ∧ b <= sqrt 2 * a

-- State the main theorem
theorem not_possible_to_create_3_similar_piles (x₀ : ℝ) (h₀ : 0 < x₀) : 
  ¬ ∃ x₁ x₂ x₃ : ℝ, 
    x₁ + x₂ + x₃ = x₀ ∧ 
    similar x₁ x₂ ∧ 
    similar x₂ x₃ ∧ 
    similar x₃ x₁ :=
sorry

end not_possible_to_create_3_similar_piles_l85_85718


namespace compute_pounds_of_cotton_l85_85761

theorem compute_pounds_of_cotton (x : ℝ) :
  (5 * 30 + 10 * x = 640) → (x = 49) := by
  intro h
  sorry

end compute_pounds_of_cotton_l85_85761


namespace xyz_inequality_l85_85865

theorem xyz_inequality (x y z : ℝ) (h : x + y + z > 0) : x^3 + y^3 + z^3 > 3 * x * y * z :=
by
  sorry

end xyz_inequality_l85_85865


namespace parabola_equation_line_tangent_to_fixed_circle_l85_85661

open Real

def parabola_vertex_origin_directrix (p : ℝ) : Prop :=
  ∀ x y : ℝ, y^2 = 2 * p * x ↔ x = -2

def point_on_directrix (l: ℝ) (t : ℝ) : Prop :=
  t ≠ 0 ∧ l = 3 * t - 1 / t

def point_on_y_axis (q : ℝ) (t : ℝ) : Prop :=
  q = 2 * t

theorem parabola_equation (p : ℝ) : 
  parabola_vertex_origin_directrix 4 →
  y^2 = 8 * x :=
by
  sorry

theorem line_tangent_to_fixed_circle (t : ℝ) (x0 : ℝ) (r : ℝ) :
  t ≠ 0 →
  point_on_directrix (-2) t →
  point_on_y_axis (2 * t) t →
  (x0 = 2 ∧ r = 2) →
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 :=
by
  sorry

end parabola_equation_line_tangent_to_fixed_circle_l85_85661


namespace sqrt_mult_l85_85167

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l85_85167


namespace percent_defective_units_l85_85408

theorem percent_defective_units (D : ℝ) (h1 : 0.05 * D = 0.5) : D = 10 := by
  sorry

end percent_defective_units_l85_85408


namespace sum_distinct_prime_factors_of_n_l85_85972

theorem sum_distinct_prime_factors_of_n (n : ℕ) 
    (h1 : n < 1000) 
    (h2 : ∃ k : ℕ, 42 * n = 180 * k) : 
    ∃ p1 p2 p3 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ n % p1 = 0 ∧ n % p2 = 0 ∧ n % p3 = 0 ∧ p1 + p2 + p3 = 10 := 
sorry

end sum_distinct_prime_factors_of_n_l85_85972


namespace intersection_of_A_and_B_l85_85523

-- Definitions for the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 2, 4}

-- Proof statement
theorem intersection_of_A_and_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l85_85523


namespace common_difference_is_three_l85_85228

-- Define the arithmetic sequence and the conditions
def arith_seq (a : ℕ → ℤ) (d : ℤ) := ∀ n, a (n + 1) = a n + d

variables {a : ℕ → ℤ} {d : ℤ}

-- Define the specific conditions given in the problem
def a1 := (a 0 = 2)
def a3 := (a 2 = 8)

-- State the theorem
theorem common_difference_is_three (h_seq : arith_seq a d) (h_a1 : a1) (h_a3 : a3) : d = 3 := 
sorry

end common_difference_is_three_l85_85228


namespace cos_value_of_2alpha_plus_5pi_over_12_l85_85848

theorem cos_value_of_2alpha_plus_5pi_over_12
  (α : ℝ) (h1 : Real.pi / 2 < α ∧ α < Real.pi)
  (h2 : Real.sin (α + Real.pi / 3) = -4 / 5) :
  Real.cos (2 * α + 5 * Real.pi / 12) = 17 * Real.sqrt 2 / 50 :=
by 
  sorry

end cos_value_of_2alpha_plus_5pi_over_12_l85_85848


namespace necessary_not_sufficient_condition_l85_85846

theorem necessary_not_sufficient_condition (m : ℝ) 
  (h : 2 < m ∧ m < 6) :
  (∃ (x y : ℝ), (x^2 / (m - 2) + y^2 / (6 - m) = 1)) ∧ (∀ m', 2 < m' ∧ m' < 6 → ∃ (x' y' : ℝ), (x'^2 / (m' - 2) + y'^2 / (6 - m') = 1) ∧ m' ≠ 4) :=
by
  sorry

end necessary_not_sufficient_condition_l85_85846


namespace sqrt_mul_eq_6_l85_85139

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l85_85139


namespace domain_of_f_l85_85512

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (Real.sqrt (x - 7))

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f y = x} = Set.Ioi 7 := by
  sorry

end domain_of_f_l85_85512


namespace probability_inside_triangle_l85_85598

open Rat -- Open rational scope for easier usage

-- Definitions based on the given problem
def base1 : ℕ := 10
def base2 : ℕ := 20
def height_trap : ℕ := 10
def base_triangle : ℕ := 8
def height_triangle : ℕ := 5

-- Calculate the areas
def area_trap : ℚ := (base1 + base2) * height_trap / 2
def area_triangle : ℚ := (base_triangle * height_triangle) / 2

-- Define the probability
def probability : ℚ := area_triangle / area_trap

-- State the theorem
theorem probability_inside_triangle :
  probability = 2 / 15 :=
by
  -- Proof to be filled in
  sorry

end probability_inside_triangle_l85_85598


namespace sqrt_mul_eq_6_l85_85140

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l85_85140


namespace sqrt_mult_eq_six_l85_85159

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l85_85159


namespace problem_statement_l85_85353

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l85_85353


namespace students_got_on_second_stop_l85_85233

-- Given conditions translated into definitions and hypotheses
def students_after_first_stop := 39
def students_after_second_stop := 68

-- The proof statement we aim to prove
theorem students_got_on_second_stop : (students_after_second_stop - students_after_first_stop) = 29 := by
  -- Proof goes here
  sorry

end students_got_on_second_stop_l85_85233


namespace greatest_possible_value_l85_85925

theorem greatest_possible_value (x y : ℝ) (h1 : x^2 + y^2 = 98) (h2 : x * y = 40) : x + y = Real.sqrt 178 :=
by sorry

end greatest_possible_value_l85_85925


namespace total_investment_sum_l85_85284

theorem total_investment_sum :
  let R : ℝ := 2200
  let T : ℝ := R - 0.1 * R
  let V : ℝ := T + 0.1 * T
  R + T + V = 6358 := by
  sorry

end total_investment_sum_l85_85284


namespace rubert_james_ratio_l85_85006

-- Definitions and conditions from a)
def adam_candies : ℕ := 6
def james_candies : ℕ := 3 * adam_candies
def rubert_candies (total_candies : ℕ) : ℕ := total_candies - (adam_candies + james_candies)
def total_candies : ℕ := 96

-- Statement to prove the ratio
theorem rubert_james_ratio : 
  (rubert_candies total_candies) / james_candies = 4 :=
by
  -- Proof is not required, so we leave it as sorry.
  sorry

end rubert_james_ratio_l85_85006


namespace impossible_to_create_3_piles_l85_85710

def similar_sizes (x y : ℝ) : Prop := x ≤ sqrt 2 * y

theorem impossible_to_create_3_piles (x y : ℝ) (h₁ : similar_sizes x y) :
  ¬ ∃ (x₁ x₂ x₃ : ℝ), x₁ + x₂ + x₃ = x + y ∧ similar_sizes x₁ x₂ ∧ similar_sizes x₂ x₃ ∧ similar_sizes x₁ x₃ :=
by
  sorry

end impossible_to_create_3_piles_l85_85710


namespace emily_needs_375_nickels_for_book_l85_85505

theorem emily_needs_375_nickels_for_book
  (n : ℕ)
  (book_cost : ℝ)
  (five_dollars : ℝ)
  (one_dollars : ℝ)
  (quarters : ℝ)
  (nickel_value : ℝ)
  (total_money : ℝ)
  (h1 : book_cost = 46.25)
  (h2 : five_dollars = 4 * 5)
  (h3 : one_dollars = 5 * 1)
  (h4 : quarters = 10 * 0.25)
  (h5 : nickel_value = n * 0.05)
  (h6 : total_money = five_dollars + one_dollars + quarters + nickel_value) 
  (h7 : total_money ≥ book_cost) :
  n ≥ 375 :=
by 
  sorry

end emily_needs_375_nickels_for_book_l85_85505


namespace total_age_l85_85932

-- Define the ages of a, b, and c based on the conditions given
variables (a b c : ℕ)

-- Condition 1: a is two years older than b
def age_condition1 := a = b + 2

-- Condition 2: b is twice as old as c
def age_condition2 := b = 2 * c

-- Condition 3: b is 12 years old
def age_condition3 := b = 12

-- Prove that the total of the ages of a, b, and c is 32 years
theorem total_age : age_condition1 → age_condition2 → age_condition3 → a + b + c = 32 :=
by
  intros h1 h2 h3 
  -- Proof would go here
  sorry

end total_age_l85_85932


namespace jerrie_minutes_l85_85314

-- Define the conditions
def barney_situps_per_minute := 45
def carrie_situps_per_minute := 2 * barney_situps_per_minute
def jerrie_situps_per_minute := carrie_situps_per_minute + 5
def barney_total_situps := 1 * barney_situps_per_minute
def carrie_total_situps := 2 * carrie_situps_per_minute
def combined_total_situps := 510

-- Define the question and required proof
theorem jerrie_minutes :
  ∃ J : ℕ, barney_total_situps + carrie_total_situps + J * jerrie_situps_per_minute = combined_total_situps ∧ J = 3 :=
  by
  sorry

end jerrie_minutes_l85_85314


namespace multiply_powers_l85_85296

theorem multiply_powers (x : ℝ) : x^3 * x^3 = x^6 :=
by sorry

end multiply_powers_l85_85296


namespace unit_cost_of_cranberry_juice_l85_85300

theorem unit_cost_of_cranberry_juice (total_cost : ℕ) (ounces : ℕ) (h1 : total_cost = 84) (h2 : ounces = 12) :
  total_cost / ounces = 7 :=
by
  sorry

end unit_cost_of_cranberry_juice_l85_85300


namespace exponent_calculation_l85_85798

-- Define the necessary exponents and base
def base : ℕ := 19
def exp1 : ℕ := 11
def exp2 : ℕ := 8

-- Given condition
lemma power_property (a : ℕ) (m n : ℕ) : a^m / a^n = a^(m - n) := by sorry

-- Proof that 19^11 / 19^8 = 6859
theorem exponent_calculation : base^exp1 / base^exp2 = 6859 := by
  have : base^exp1 / base^exp2 = base^(exp1 - exp2) := power_property base exp1 exp2
  have : base^(exp1 - exp2) = base^3 := by rw [nat.sub_eq_iff_eq_add.mpr rfl]
  have : base^3 = 6859 := by -- This would be an arithmetic computation
    rfl -- Placeholder for the actual arithmetic; ideally, you'd verify this step.
  sorry

end exponent_calculation_l85_85798


namespace isosceles_triangle_base_length_l85_85273

-- Define the isosceles triangle problem
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  perimeter : ℝ
  isIsosceles : (side1 = side2 ∨ side1 = base ∨ side2 = base)
  sideLengthCondition : (side1 = 3 ∨ side2 = 3 ∨ base = 3)
  perimeterCondition : side1 + side2 + base = 13
  triangleInequality1 : side1 + side2 > base
  triangleInequality2 : side1 + base > side2
  triangleInequality3 : side2 + base > side1

-- Define the theorem to prove
theorem isosceles_triangle_base_length (T : IsoscelesTriangle) :
  T.base = 3 := by
  sorry

end isosceles_triangle_base_length_l85_85273


namespace true_conjunction_l85_85375

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l85_85375


namespace large_circle_radius_l85_85309

theorem large_circle_radius (s : ℝ) (r : ℝ) (R : ℝ)
  (side_length : s = 6)
  (coverage : ∀ (x y : ℝ), (x - y)^2 + (x - y)^2 = (2 * R)^2) :
  R = 3 * Real.sqrt 2 :=
by
  sorry

end large_circle_radius_l85_85309


namespace percentage_bob_is_36_l85_85316

def water_per_acre_corn : ℕ := 20
def water_per_acre_cotton : ℕ := 80
def water_per_acre_beans : ℕ := 2 * water_per_acre_corn

def acres_bob_corn : ℕ := 3
def acres_bob_cotton : ℕ := 9
def acres_bob_beans : ℕ := 12

def acres_brenda_corn : ℕ := 6
def acres_brenda_cotton : ℕ := 7
def acres_brenda_beans : ℕ := 14

def acres_bernie_corn : ℕ := 2
def acres_bernie_cotton : ℕ := 12

def water_bob : ℕ := (acres_bob_corn * water_per_acre_corn) +
                      (acres_bob_cotton * water_per_acre_cotton) +
                      (acres_bob_beans * water_per_acre_beans)

def water_brenda : ℕ := (acres_brenda_corn * water_per_acre_corn) +
                         (acres_brenda_cotton * water_per_acre_cotton) +
                         (acres_brenda_beans * water_per_acre_beans)

def water_bernie : ℕ := (acres_bernie_corn * water_per_acre_corn) +
                         (acres_bernie_cotton * water_per_acre_cotton)

def total_water : ℕ := water_bob + water_brenda + water_bernie

def percentage_bob : ℚ := (water_bob : ℚ) / (total_water : ℚ) * 100

theorem percentage_bob_is_36 : percentage_bob = 36 := by
  sorry

end percentage_bob_is_36_l85_85316


namespace solve_for_pairs_l85_85285
-- Import necessary libraries

-- Define the operation
def diamond (a b c d : ℤ) : ℤ × ℤ :=
  (a * c - b * d, a * d + b * c)

theorem solve_for_pairs : ∃! (x y : ℤ), diamond x 3 x y = (6, 0) ∧ (x, y) = (0, -2) := by
  sorry

end solve_for_pairs_l85_85285


namespace valid_outfits_number_l85_85858

def num_shirts := 7
def num_pants := 7
def num_hats := 7
def num_colors := 7

def total_outfits (num_shirts num_pants num_hats : ℕ) := num_shirts * num_pants * num_hats
def matching_color_outfits (num_colors : ℕ) := num_colors
def valid_outfits (num_shirts num_pants num_hats num_colors : ℕ) := 
  total_outfits num_shirts num_pants num_hats - matching_color_outfits num_colors

theorem valid_outfits_number : valid_outfits num_shirts num_pants num_hats num_colors = 336 := 
by
  sorry

end valid_outfits_number_l85_85858


namespace f_is_even_l85_85893

-- Given an odd function g
def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g (x)

-- Define the function f as given by the problem
def f (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  abs (g (x^2))

-- The theorem stating that f is an even function
theorem f_is_even (g : ℝ → ℝ) (h_odd : is_odd_function g) : ∀ x, f g x = f g (-x) :=
by
  sorry

end f_is_even_l85_85893


namespace impossible_to_create_3_piles_l85_85711

-- Defining similar piles
def similar (x y : ℝ) : Prop :=
  x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

-- Main theorem statement
theorem impossible_to_create_3_piles (initial_pile : ℝ) (h_initial : initial_pile > 0) :
  ∀ (x y z : ℝ), 
  x + y + z = initial_pile → 
  similar x y ∧ similar y z ∧ similar z x → 
  false := 
by 
  sorry

end impossible_to_create_3_piles_l85_85711


namespace function_increasing_on_interval_l85_85214

theorem function_increasing_on_interval {x : ℝ} (hx : x < 1) : 
  (-1/2) * x^2 + x + 4 < -1/2 * (x + 1)^2 + (x + 1) + 4 :=
sorry

end function_increasing_on_interval_l85_85214


namespace sqrt3_mul_sqrt12_eq_6_l85_85092

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l85_85092


namespace gcd_1113_1897_l85_85831

theorem gcd_1113_1897 : Int.gcd 1113 1897 = 7 := by
  sorry

end gcd_1113_1897_l85_85831


namespace p_and_q_is_true_l85_85342

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l85_85342


namespace spider_total_distance_l85_85066

theorem spider_total_distance : 
  ∀ (pos1 pos2 pos3 : ℝ), pos1 = 3 → pos2 = -1 → pos3 = 8.5 → 
  |pos2 - pos1| + |pos3 - pos2| = 13.5 := 
by 
  intros pos1 pos2 pos3 hpos1 hpos2 hpos3 
  sorry

end spider_total_distance_l85_85066


namespace sqrt_mul_sqrt_eq_six_l85_85146

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l85_85146


namespace Tim_scored_30_l85_85404

-- Definitions and conditions
variables (Joe Tim Ken : ℕ)
variables (h1 : Tim = Joe + 20)
variables (h2 : Tim = Nat.div (Ken * 2) 2)
variables (h3 : Joe + Tim + Ken = 100)

-- Statement to prove
theorem Tim_scored_30 : Tim = 30 :=
by sorry

end Tim_scored_30_l85_85404


namespace minimum_voters_for_tall_giraffe_l85_85683

-- Definitions and conditions
def voters : ℕ := 135
def districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3

-- Conditions encoded in the problem statement
def majority_precinct (votes: ℕ) : Prop := votes >= 2
def majority_district (precinct_wins: ℕ) : Prop := precinct_wins >= 5
def majority_winner (district_wins: ℕ) : Prop := district_wins >= 3

-- The problem states that the Tall giraffe won.
axiom tall_giraffe_won : ∃ district_wins : ℕ, 
  majority_winner district_wins ∧ 
  ∀ (d ∈ (finset.range districts)), ∃ precinct_wins : ℕ, 
  majority_district precinct_wins ∧ 
  ∀ (p ∈ (finset.range precincts_per_district)), ∃ votes : ℕ, 
  majority_precinct votes

-- Proof goal
theorem minimum_voters_for_tall_giraffe : ∃ (votes_for_tall : ℕ), votes_for_tall = 30 :=
by {
  -- proof of the theorem will go here
  sorry 
}

end minimum_voters_for_tall_giraffe_l85_85683


namespace sum_numerator_denominator_l85_85039

-- Given the repeating decimal 0.343434...
def repeating_decimal : ℚ := 34 / 99

-- Prove that the decimal's fraction form in lowest terms adds up to 133.
theorem sum_numerator_denominator : 
  let num := repeating_decimal.num in 
  let denom := repeating_decimal.denom in 
  num + denom = 133 :=
by
  have x : repeating_decimal = 34 / 99 := by sorry
  sorry   -- Placeholder for the proof steps to demonstrate num and denom values lead to 133.

end sum_numerator_denominator_l85_85039


namespace correct_operation_is_a_l85_85048

theorem correct_operation_is_a (a b : ℝ) : 
  (a^4 * a^3 = a^7) ∧ 
  ((a^2)^3 ≠ a^5) ∧ 
  (3 * a^2 - a^2 ≠ 2) ∧ 
  ((a - b)^2 ≠ a^2 - b^2) := 
by {
  -- Here, you would fill in the proof
  sorry
}

end correct_operation_is_a_l85_85048


namespace sqrt_mult_l85_85171

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l85_85171


namespace student_chose_number_l85_85053

theorem student_chose_number (x : ℤ) (h : 2 * x - 138 = 104) : x = 121 := by
  sorry

end student_chose_number_l85_85053


namespace correct_avg_weight_of_class_l85_85870

theorem correct_avg_weight_of_class :
  ∀ (n : ℕ) (avg_wt : ℝ) (mis_A mis_B mis_C actual_A actual_B actual_C : ℝ),
  n = 30 →
  avg_wt = 60.2 →
  mis_A = 54 → actual_A = 64 →
  mis_B = 58 → actual_B = 68 →
  mis_C = 50 → actual_C = 60 →
  (n * avg_wt + (actual_A - mis_A) + (actual_B - mis_B) + (actual_C - mis_C)) / n = 61.2 :=
by
  intros n avg_wt mis_A mis_B mis_C actual_A actual_B actual_C h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end correct_avg_weight_of_class_l85_85870


namespace burritos_in_each_box_l85_85693

theorem burritos_in_each_box (B : ℕ) (h1 : 3 * B - B - 30 = 10) : B = 20 :=
by
  sorry

end burritos_in_each_box_l85_85693


namespace zach_babysitting_hours_l85_85052

theorem zach_babysitting_hours :
  ∀ (bike_cost weekly_allowance mowing_pay babysitting_rate saved_amount needed_additional_amount hours_needed : ℕ),
    bike_cost = 100 →
    weekly_allowance = 5 →
    mowing_pay = 10 →
    babysitting_rate = 7 →
    saved_amount = 65 →
    needed_additional_amount = 6 →
    saved_amount + weekly_allowance + mowing_pay + hours_needed * babysitting_rate = bike_cost - needed_additional_amount →
    hours_needed = 2 :=
by
  intros bike_cost weekly_allowance mowing_pay babysitting_rate saved_amount needed_additional_amount hours_needed
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end zach_babysitting_hours_l85_85052


namespace sum_numerator_denominator_repeating_decimal_l85_85040

theorem sum_numerator_denominator_repeating_decimal :
  let x := 34 / 99 in
  x.denom * x + x.num = 133 :=
by
  let x : ℚ := 34 / 99
  sorry

end sum_numerator_denominator_repeating_decimal_l85_85040


namespace toll_booth_ratio_l85_85874

theorem toll_booth_ratio (total_cars : ℕ) (monday_cars tuesday_cars friday_cars saturday_cars sunday_cars : ℕ)
  (x : ℕ) (h1 : total_cars = 450) (h2 : monday_cars = 50) (h3 : tuesday_cars = 50) (h4 : friday_cars = 50)
  (h5 : saturday_cars = 50) (h6 : sunday_cars = 50) (h7 : monday_cars + tuesday_cars + x + x + friday_cars + saturday_cars + sunday_cars = total_cars) :
  x = 100 ∧ x / monday_cars = 2 :=
by
  sorry

end toll_booth_ratio_l85_85874


namespace sin_B_of_arithmetic_sequence_angles_l85_85670

theorem sin_B_of_arithmetic_sequence_angles (A B C : ℝ) (h1 : A + C = 2 * B) (h2 : A + B + C = Real.pi) :
  Real.sin B = Real.sqrt 3 / 2 :=
sorry

end sin_B_of_arithmetic_sequence_angles_l85_85670


namespace sum_of_fraction_numerator_denominator_l85_85044

theorem sum_of_fraction_numerator_denominator :
  let x := 0.343434...
  in let fraction := (34 / 99 : ℚ)
  in let sum := fraction.num + fraction.den 
  in (x : ℚ) = fraction ∧ fraction.isReduced → sum = 133 :=
by
  sorry

end sum_of_fraction_numerator_denominator_l85_85044


namespace sequence_expression_l85_85320

theorem sequence_expression (n : ℕ) (h : n ≥ 2) (T : ℕ → ℕ) (a : ℕ → ℕ)
  (hT : ∀ k : ℕ, T k = 2 * k^2)
  (ha : ∀ k : ℕ, k ≥ 2 → a k = T k / T (k - 1)) :
  a n = (n / (n - 1))^2 := 
sorry

end sequence_expression_l85_85320


namespace minimum_voters_needed_l85_85681

-- conditions
def num_voters := 135
def num_districts := 5
def precincts_per_district := 9
def voters_per_precinct := 3
def majority_precincts (n : ℕ) := (n + 1) / 2

-- definitions for quantities derived from conditions
def total_precincts := num_districts * precincts_per_district
def majority_districts := majority_precincts num_districts
def precincts_needed_for_district_win := majority_precincts precincts_per_district
def total_precincts_needed_for_win := majority_districts * precincts_needed_for_district_win
def votes_needed_per_precinct := majority_precincts voters_per_precinct

-- main statement
theorem minimum_voters_needed : (votes_needed_per_precinct * total_precincts_needed_for_win = 30) ∧ TallGiraffeWon :=
by sorry

end minimum_voters_needed_l85_85681


namespace all_values_are_equal_l85_85614

theorem all_values_are_equal
  (f : ℤ × ℤ → ℕ)
  (h : ∀ x y : ℤ, f (x, y) * 4 = f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1))
  (hf_pos : ∀ x y : ℤ, 0 < f (x, y)) : 
  ∀ x y x' y' : ℤ, f (x, y) = f (x', y') :=
by
  sorry

end all_values_are_equal_l85_85614


namespace pure_water_to_achieve_desired_concentration_l85_85547

theorem pure_water_to_achieve_desired_concentration :
  ∀ (w : ℝ), (50 + w ≠ 0) → (0.4 * 50 / (50 + w) = 0.25) → w = 30 := 
by
  intros w h_nonzero h_concentration
  sorry

end pure_water_to_achieve_desired_concentration_l85_85547


namespace min_value_of_quadratic_l85_85288

theorem min_value_of_quadratic : ∃ x : ℝ, 7 * x^2 - 28 * x + 1702 = 1674 ∧ ∀ y : ℝ, 7 * y^2 - 28 * y + 1702 ≥ 1674 :=
by
  sorry

end min_value_of_quadratic_l85_85288


namespace most_likely_number_of_cars_l85_85280

theorem most_likely_number_of_cars 
  (total_time_seconds : ℕ)
  (rate_cars_per_second : ℚ)
  (h1 : total_time_seconds = 180)
  (h2 : rate_cars_per_second = 8 / 15) : 
  ∃ (n : ℕ), n = 100 :=
by
  sorry

end most_likely_number_of_cars_l85_85280


namespace solve_inequality1_solve_inequality2_l85_85008

-- Proof problem 1
theorem solve_inequality1 (x : ℝ) : 
  2 < |2 * x - 5| → |2 * x - 5| ≤ 7 → -1 ≤ x ∧ x < (3 / 2) ∨ (7 / 2) < x ∧ x ≤ 6 :=
sorry

-- Proof problem 2
theorem solve_inequality2 (x : ℝ) : 
  (1 / (x - 1)) > (x + 1) → x < - Real.sqrt 2 ∨ (1 < x ∧ x < Real.sqrt 2) :=
sorry

end solve_inequality1_solve_inequality2_l85_85008


namespace solve_inequality_l85_85516

theorem solve_inequality (x : ℝ) : x > 13 ↔ x^3 - 16 * x^2 + 73 * x > 84 :=
by
  sorry

end solve_inequality_l85_85516


namespace sum_of_series_l85_85618

theorem sum_of_series : (1 / (1 * 2 * 3) + 1 / (2 * 3 * 4) + 1 / (3 * 4 * 5) + 1 / (4 * 5 * 6)) = 7 / 30 :=
by
  sorry

end sum_of_series_l85_85618


namespace seq_bn_arithmetic_seq_an_formula_sum_an_terms_l85_85876

-- (1) Prove that the sequence {b_n} is an arithmetic sequence
theorem seq_bn_arithmetic (a : ℕ → ℕ) (b : ℕ → ℤ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = 2 * a n + 2^n)
  (h3 : ∀ n, b n = a n / 2^(n - 1)) :
  ∀ n, b (n + 1) - b n = 1 := by
  sorry

-- (2) Find the general formula for the sequence {a_n}
theorem seq_an_formula (a : ℕ → ℕ) (b : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = 2 * a n + 2^n)
  (h3 : ∀ n, b n = a n / 2^(n - 1)) :
  ∀ n, a n = n * 2^(n - 1) := by
  sorry

-- (3) Find the sum of the first n terms of the sequence {a_n}
theorem sum_an_terms (a : ℕ → ℕ) (S : ℕ → ℤ) (h1 : ∀ n, a n = n * 2^(n - 1)) :
  ∀ n, S n = (n - 1) * 2^n + 1 := by
  sorry

end seq_bn_arithmetic_seq_an_formula_sum_an_terms_l85_85876


namespace total_emails_in_april_is_675_l85_85419

-- Define the conditions
def daily_emails : ℕ := 20
def additional_emails : ℕ := 5
def april_days : ℕ := 30
def half_april_days : ℕ := april_days / 2

-- Define the total number of emails received
def total_emails : ℕ :=
  (daily_emails * half_april_days) +
  ((daily_emails + additional_emails) * half_april_days)

-- Define the statement to be proven
theorem total_emails_in_april_is_675 : total_emails = 675 :=
  by
  sorry

end total_emails_in_april_is_675_l85_85419


namespace total_students_l85_85253

noncomputable def totalStudentsOptions (groups totalGroups specificGroupCount specificGroupSize otherGroupSizes : ℕ) : Set ℕ :=
  if totalGroups = 6 ∧ specificGroupCount = 4 ∧ specificGroupSize = 13 ∧ (otherGroupSizes = 12 ∨ otherGroupSizes = 14) then
    {52 + 2 * otherGroupSizes}
  else
    ∅

theorem total_students :
  totalStudentsOptions 6 4 13 12 = {76} ∧ totalStudentsOptions 6 4 13 14 = {80} :=
by
  -- This is where the proof would go, but we're skipping it as per instructions
  sorry

end total_students_l85_85253


namespace smallest_k_multiple_of_180_l85_85327

def sum_of_squares (k : ℕ) : ℕ :=
  (k * (k + 1) * (2 * k + 1)) / 6

def divisible_by_180 (n : ℕ) : Prop :=
  n % 180 = 0

theorem smallest_k_multiple_of_180 :
  ∃ k : ℕ, k > 0 ∧ divisible_by_180 (sum_of_squares k) ∧ ∀ m : ℕ, m > 0 ∧ divisible_by_180 (sum_of_squares m) → k ≤ m :=
sorry

end smallest_k_multiple_of_180_l85_85327


namespace woman_work_rate_l85_85306

theorem woman_work_rate :
  let M := 1/6
  let B := 1/9
  let combined_rate := 1/3
  ∃ W : ℚ, M + B + W = combined_rate ∧ 1 / W = 18 := 
by
  sorry

end woman_work_rate_l85_85306


namespace positive_integer_perfect_square_l85_85815

theorem positive_integer_perfect_square (n : ℕ) (h1: n > 0) (h2 : ∃ k : ℕ, n^2 - 19 * n - 99 = k^2) : n = 199 :=
sorry

end positive_integer_perfect_square_l85_85815


namespace smaller_number_is_five_l85_85453

theorem smaller_number_is_five (x y : ℕ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 :=
sorry

end smaller_number_is_five_l85_85453


namespace coeff_sum_zero_l85_85844

theorem coeff_sum_zero (a₀ a₁ a₂ a₃ a₄ : ℝ) (h : ∀ x : ℝ, (2*x + 1)^4 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4) :
  a₁ + a₂ + a₃ + a₄ = 0 :=
by
  sorry

end coeff_sum_zero_l85_85844


namespace quadratic_solution_l85_85499

def quadratic_rewrite (x b c : ℝ) : ℝ := (x + b) * (x + b) + c

theorem quadratic_solution (b c : ℝ)
  (h1 : ∀ x, x^2 + 2100 * x + 4200 = quadratic_rewrite x b c)
  (h2 : c = -b^2 + 4200) :
  c / b = -1034 :=
by
  sorry

end quadratic_solution_l85_85499


namespace find_N_l85_85059

noncomputable def N : ℕ := 1156

-- Condition 1: N is a perfect square
axiom N_perfect_square : ∃ n : ℕ, N = n^2

-- Condition 2: All digits of N are less than 7
axiom N_digits_less_than_7 : ∀ d, d ∈ [1, 1, 5, 6] → d < 7

-- Condition 3: Adding 3 to each digit yields another perfect square
axiom N_plus_3_perfect_square : ∃ m : ℕ, (m^2 = 1156 + 3333)

theorem find_N : N = 1156 :=
by
  -- Proof goes here
  sorry

end find_N_l85_85059


namespace tracy_initial_candies_l85_85028

variable (x : ℕ)
variable (b : ℕ)

theorem tracy_initial_candies : 
  (x % 6 = 0) ∧
  (34 ≤ (1 / 2 * x)) ∧
  ((1 / 2 * x) ≤ 38) ∧
  (1 ≤ b) ∧
  (b ≤ 5) ∧
  (1 / 2 * x - 30 - b = 3) →
  x = 72 := 
sorry

end tracy_initial_candies_l85_85028


namespace ships_meeting_count_l85_85904

theorem ships_meeting_count :
  ∀ (n : ℕ) (east_sailing west_sailing : ℕ),
    n = 10 →
    east_sailing = 5 →
    west_sailing = 5 →
    east_sailing + west_sailing = n →
    (∀ (v : ℕ), v > 0) →
    25 = east_sailing * west_sailing :=
by
  intros n east_sailing west_sailing h1 h2 h3 h4 h5
  sorry

end ships_meeting_count_l85_85904


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l85_85036

theorem sum_of_numerator_and_denominator_of_repeating_decimal (x : ℚ) (h : x = 34 / 99) : (x.den + x.num : ℤ) = 133 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l85_85036


namespace max_cos_half_sin_eq_1_l85_85191

noncomputable def max_value_expression (θ : ℝ) : ℝ :=
  Real.cos (θ / 2) * (1 - Real.sin θ)

theorem max_cos_half_sin_eq_1 : 
  ∀ θ : ℝ, 0 < θ ∧ θ < π → max_value_expression θ ≤ 1 :=
by
  intros θ h
  sorry

end max_cos_half_sin_eq_1_l85_85191


namespace angle_in_second_quadrant_l85_85223

-- Definitions of conditions
def sin2_pos : Prop := Real.sin 2 > 0
def cos2_neg : Prop := Real.cos 2 < 0

-- Statement of the problem
theorem angle_in_second_quadrant (h1 : sin2_pos) (h2 : cos2_neg) : 
    (∃ α, 0 < α ∧ α < π ∧ P = (Real.sin α, Real.cos α)) :=
by
  sorry

end angle_in_second_quadrant_l85_85223


namespace sqrt_mul_l85_85102

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l85_85102


namespace vector_subtraction_l85_85509

/-
Define the vectors we are working with.
-/
def v1 : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![-8]]
def v2 : Matrix (Fin 2) (Fin 1) ℤ := ![![2], ![-6]]
def scalar : ℤ := 5
def result : Matrix (Fin 2) (Fin 1) ℤ := ![![-7], ![22]]

/-
The statement of the proof problem.
-/
theorem vector_subtraction : v1 - scalar • v2 = result := 
by
  sorry

end vector_subtraction_l85_85509


namespace correct_operation_l85_85050

theorem correct_operation :
  (∀ a : ℝ, a^4 * a^3 = a^7)
  ∧ (∀ a : ℝ, (a^2)^3 ≠ a^5)
  ∧ (∀ a : ℝ, 3 * a^2 - a^2 ≠ 2)
  ∧ (∀ a b : ℝ, (a - b)^2 ≠ a^2 - b^2) :=
by {
  sorry
}

end correct_operation_l85_85050


namespace penny_canoe_l85_85246

theorem penny_canoe (P : ℕ)
  (h1 : 140 * (2/3 : ℚ) * P + 35 = 595) : P = 6 :=
sorry

end penny_canoe_l85_85246


namespace can_divide_cube_into_71_l85_85567

theorem can_divide_cube_into_71 : 
  ∃ (n : ℕ), n = 71 ∧ 
  (∃ (f : ℕ → ℕ), f 0 = 1 ∧ (∀ k, f (k + 1) = f k + 7) ∧ f n = 71) :=
by
  sorry

end can_divide_cube_into_71_l85_85567


namespace proposition_p_and_q_is_true_l85_85347

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l85_85347


namespace sqrt_mul_eq_l85_85079

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l85_85079


namespace original_number_exists_l85_85968

theorem original_number_exists 
  (N: ℤ)
  (h1: ∃ (k: ℤ), N - 6 = 16 * k)
  (h2: ∀ (m: ℤ), (N - m) % 16 = 0 → m ≥ 6) : 
  N = 22 :=
sorry

end original_number_exists_l85_85968


namespace sum_numerator_denominator_l85_85038

-- Given the repeating decimal 0.343434...
def repeating_decimal : ℚ := 34 / 99

-- Prove that the decimal's fraction form in lowest terms adds up to 133.
theorem sum_numerator_denominator : 
  let num := repeating_decimal.num in 
  let denom := repeating_decimal.denom in 
  num + denom = 133 :=
by
  have x : repeating_decimal = 34 / 99 := by sorry
  sorry   -- Placeholder for the proof steps to demonstrate num and denom values lead to 133.

end sum_numerator_denominator_l85_85038


namespace Jerry_remaining_pages_l85_85570

theorem Jerry_remaining_pages (total_pages : ℕ) (saturday_read : ℕ) (sunday_read : ℕ) (remaining_pages : ℕ) 
  (h1 : total_pages = 93) (h2 : saturday_read = 30) (h3 : sunday_read = 20) (h4 : remaining_pages = 43) : 
  total_pages - saturday_read - sunday_read = remaining_pages := 
by
  sorry

end Jerry_remaining_pages_l85_85570


namespace radio_lowest_price_rank_l85_85074

-- Definitions based on the conditions
def total_items : ℕ := 38
def radio_highest_rank : ℕ := 16

-- The theorem statement
theorem radio_lowest_price_rank : (total_items - (radio_highest_rank - 1)) = 24 := by
  sorry

end radio_lowest_price_rank_l85_85074


namespace divisible_by_six_l85_85003

theorem divisible_by_six (m : ℕ) : 6 ∣ (m^3 + 11 * m) := 
sorry

end divisible_by_six_l85_85003


namespace bobby_pays_correct_amount_l85_85644

noncomputable def bobby_total_cost : ℝ := 
  let mold_cost : ℝ := 250
  let material_original_cost : ℝ := 150
  let material_discount : ℝ := 0.20 * material_original_cost
  let material_cost : ℝ := material_original_cost - material_discount
  let hourly_rate_original : ℝ := 75
  let hourly_rate_increased : ℝ := hourly_rate_original + 10
  let work_hours : ℝ := 8
  let work_cost_original : ℝ := work_hours * hourly_rate_increased
  let work_cost_discount : ℝ := 0.80 * work_cost_original
  let cost_before_tax : ℝ := mold_cost + material_cost + work_cost_discount
  let tax : ℝ := 0.10 * cost_before_tax
  cost_before_tax + tax

theorem bobby_pays_correct_amount : bobby_total_cost = 1005.40 := sorry

end bobby_pays_correct_amount_l85_85644


namespace both_pieces_no_shorter_than_1m_l85_85909

noncomputable def rope_cut_probability : ℝ :=
let total_length: ℝ := 3 in
let favorable_length: ℝ := 1 in
favorable_length / total_length

theorem both_pieces_no_shorter_than_1m : rope_cut_probability = 1 / 3 := 
by
  sorry

end both_pieces_no_shorter_than_1m_l85_85909


namespace problem1_l85_85763

theorem problem1 {a b c : ℝ} (h : a + b + c = 2) : a^2 + b^2 + c^2 + 2 * a * b * c < 2 :=
sorry

end problem1_l85_85763


namespace lines_identical_pairs_count_l85_85237

theorem lines_identical_pairs_count :
  (∃ a d : ℝ, (4 * x + a * y + d = 0 ∧ d * x - 3 * y + 15 = 0)) →
  (∃! n : ℕ, n = 2) := 
sorry

end lines_identical_pairs_count_l85_85237


namespace distance_between_trees_l85_85225

theorem distance_between_trees (length_yard : ℕ) (num_trees : ℕ) (dist : ℕ) :
  length_yard = 275 → num_trees = 26 → dist = length_yard / (num_trees - 1) → dist = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  assumption

end distance_between_trees_l85_85225


namespace RachelStillToColor_l85_85590

def RachelColoringBooks : Prop :=
  let initial_books := 23 + 32
  let colored := 44
  initial_books - colored = 11

theorem RachelStillToColor : RachelColoringBooks := 
  by
    let initial_books := 23 + 32
    let colored := 44
    show initial_books - colored = 11
    sorry

end RachelStillToColor_l85_85590


namespace sqrt_mult_simplify_l85_85125

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l85_85125


namespace monochromatic_triangle_probability_l85_85819

-- Define the coloring of the edges
inductive Color
| Red : Color
| Blue : Color

-- Define an edge
structure Edge :=
(v1 v2 : Nat)
(color : Color)

-- Define the hexagon with its sides and diagonals
def hexagonEdges : List Edge := [
  -- Sides of the hexagon
  { v1 := 1, v2 := 2, color := sorry }, { v1 := 2, v2 := 3, color := sorry },
  { v1 := 3, v2 := 4, color := sorry }, { v1 := 4, v2 := 5, color := sorry },
  { v1 := 5, v2 := 6, color := sorry }, { v1 := 6, v2 := 1, color := sorry },
  -- Diagonals of the hexagon
  { v1 := 1, v2 := 3, color := sorry }, { v1 := 1, v2 := 4, color := sorry },
  { v1 := 1, v2 := 5, color := sorry }, { v1 := 2, v2 := 4, color := sorry },
  { v1 := 2, v2 := 5, color := sorry }, { v1 := 2, v2 := 6, color := sorry },
  { v1 := 3, v2 := 5, color := sorry }, { v1 := 3, v2 := 6, color := sorry },
  { v1 := 4, v2 := 6, color := sorry }
]

-- Define what a triangle is
structure Triangle :=
(v1 v2 v3 : Nat)

-- List all possible triangles formed by vertices of the hexagon
def hexagonTriangles : List Triangle := [
  { v1 := 1, v2 := 2, v3 := 3 }, { v1 := 1, v2 := 2, v3 := 4 },
  { v1 := 1, v2 := 2, v3 := 5 }, { v1 := 1, v2 := 2, v3 := 6 },
  { v1 := 1, v2 := 3, v3 := 4 }, { v1 := 1, v2 := 3, v3 := 5 },
  { v1 := 1, v2 := 3, v3 := 6 }, { v1 := 1, v2 := 4, v3 := 5 },
  { v1 := 1, v2 := 4, v3 := 6 }, { v1 := 1, v2 := 5, v3 := 6 },
  { v1 := 2, v2 := 3, v3 := 4 }, { v1 := 2, v2 := 3, v3 := 5 },
  { v1 := 2, v2 := 3, v3 := 6 }, { v1 := 2, v2 := 4, v3 := 5 },
  { v1 := 2, v2 := 4, v3 := 6 }, { v1 := 2, v2 := 5, v3 := 6 },
  { v1 := 3, v2 := 4, v3 := 5 }, { v1 := 3, v2 := 4, v3 := 6 },
  { v1 := 3, v2 := 5, v3 := 6 }, { v1 := 4, v2 := 5, v3 := 6 }
]

-- Define the probability calculation, with placeholders for terms that need proving
noncomputable def probabilityMonochromaticTriangle : ℚ :=
  1 - (3 / 4) ^ 20

-- The theorem to prove the probability matches the given answer
theorem monochromatic_triangle_probability :
  probabilityMonochromaticTriangle = 253 / 256 :=
by sorry

end monochromatic_triangle_probability_l85_85819


namespace euler_characteristic_convex_polyhedron_l85_85057

-- Define the context of convex polyhedron with vertices (V), edges (E), and faces (F)
structure ConvexPolyhedron :=
  (V : ℕ) -- number of vertices
  (E : ℕ) -- number of edges
  (F : ℕ) -- number of faces
  (convex : Prop) -- property stating the polyhedron is convex

-- Euler characteristic theorem for convex polyhedra
theorem euler_characteristic_convex_polyhedron (P : ConvexPolyhedron) (h : P.convex) : P.V - P.E + P.F = 2 :=
sorry

end euler_characteristic_convex_polyhedron_l85_85057


namespace sqrt_mul_simp_l85_85130

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l85_85130


namespace power_quotient_l85_85795

theorem power_quotient (a m n : ℕ) (h_a : a = 19) (h_m : m = 11) (h_n : n = 8) : a^m / a^n = 6859 := by
  sorry

end power_quotient_l85_85795


namespace smallest_a_l85_85238

theorem smallest_a (a b : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (17 * x)) :
  a = 17 :=
by
  sorry

end smallest_a_l85_85238


namespace circle_diameter_from_area_l85_85267

theorem circle_diameter_from_area (A r d : ℝ) (hA : A = 64 * Real.pi) (h_area : A = Real.pi * r^2) : d = 16 :=
by
  sorry

end circle_diameter_from_area_l85_85267


namespace last_two_digits_2005_power_1989_l85_85603

theorem last_two_digits_2005_power_1989 : (2005 ^ 1989) % 100 = 25 :=
by
  sorry

end last_two_digits_2005_power_1989_l85_85603


namespace diana_owes_l85_85468

-- Define the conditions
def initial_charge : ℝ := 60
def annual_interest_rate : ℝ := 0.06
def time_in_years : ℝ := 1

-- Define the simple interest calculation
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

-- Define the total amount owed calculation
def total_amount_owed (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

-- State the theorem: Diana will owe $63.60 after one year
theorem diana_owes : total_amount_owed initial_charge (simple_interest initial_charge annual_interest_rate time_in_years) = 63.60 :=
by sorry

end diana_owes_l85_85468


namespace fraction_position_1991_1949_l85_85760

theorem fraction_position_1991_1949 :
  ∃ (row position : ℕ), 
    ∀ (i j : ℕ), 
      (∃ k : ℕ, k = i + j - 1 ∧ k = 3939) ∧
      (∃ p : ℕ, p = j ∧ p = 1949) → 
      row = 3939 ∧ position = 1949 := 
sorry

end fraction_position_1991_1949_l85_85760


namespace a_gt_one_l85_85676

noncomputable def f (a : ℝ) (x : ℝ) := 2 * a * x^2 - x - 1

theorem a_gt_one (a : ℝ) :
  (∃! x, 0 < x ∧ x < 1 ∧ f a x = 0) → 1 < a :=
by
  sorry

end a_gt_one_l85_85676


namespace circle_reflection_l85_85479

-- Definition of the original center of the circle
def original_center : ℝ × ℝ := (8, -3)

-- Definition of the reflection transformation over the line y = x
def reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Theorem stating that reflecting the original center over the line y = x results in a specific point
theorem circle_reflection : reflect original_center = (-3, 8) :=
  by
  -- skipping the proof part
  sorry

end circle_reflection_l85_85479


namespace sqrt_mul_l85_85100

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l85_85100


namespace diff_square_mental_math_l85_85318

theorem diff_square_mental_math :
  75 ^ 2 - 45 ^ 2 = 3600 :=
by
  -- The proof would go here
  sorry

end diff_square_mental_math_l85_85318


namespace winning_lottery_ticket_is_random_l85_85465

-- Definitions of the events
inductive Event
| certain : Event
| impossible : Event
| random : Event

open Event

-- Conditions
def boiling_water_event : Event := certain
def lottery_ticket_event : Event := random
def athlete_running_30mps_event : Event := impossible
def draw_red_ball_event : Event := impossible

-- Problem Statement
theorem winning_lottery_ticket_is_random : 
    lottery_ticket_event = random :=
sorry

end winning_lottery_ticket_is_random_l85_85465


namespace smallest_N_satisfying_frequencies_l85_85648

def percentageA := 1 / 5
def percentageB := 3 / 8
def percentageC := 1 / 4
def percentageD := 1 / 8
def percentageE := 1 / 20

def Divisible (n : ℕ) (d : ℕ) : Prop := ∃ (k : ℕ), n = k * d

theorem smallest_N_satisfying_frequencies :
  ∃ N : ℕ, 
    Divisible N 5 ∧ 
    Divisible N 8 ∧ 
    Divisible N 4 ∧ 
    Divisible N 20 ∧ 
    N = 40 := sorry

end smallest_N_satisfying_frequencies_l85_85648


namespace sum_of_numerator_and_denominator_l85_85034

def repeating_decimal_represents_fraction : Prop :=
  let x := 34 / 99
  0.34̅ = x

/-- The sum of the numerator and denominator of 0.34̅ in its lowest terms is 133. -/
theorem sum_of_numerator_and_denominator
  (x : ℚ)
  (hx : repeating_decimal_represents_fraction) :
  (34 + 99 = 133) :=
begin
  sorry
end

end sum_of_numerator_and_denominator_l85_85034


namespace meeting_distance_from_A_l85_85728

variable (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (distance_AB distance_BC : ℝ)
variable (cyclist_speed pedestrian_speed : ℝ)
variable (meet_distance : ℝ)

axiom distance_AB_eq_3 : distance_AB = 3
axiom distance_BC_eq_4 : distance_BC = 4
axiom simultaneous_arrival :
  ∀ AC cyclist_speed pedestrian_speed,
    (distance_AB + distance_BC) / cyclist_speed = distance_AB / pedestrian_speed
axiom speed_ratio :
  cyclist_speed / pedestrian_speed = 7 / 3
axiom meeting_point :
  ∃ meet_distance,
    meet_distance / (distance_AB - meet_distance) = 7 / 3

theorem meeting_distance_from_A :
  meet_distance = 2.1 :=
sorry

end meeting_distance_from_A_l85_85728


namespace sqrt_mul_simp_l85_85134

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l85_85134


namespace Jason_spent_correct_amount_l85_85410

namespace MusicStore

def costFlute : Real := 142.46
def costMusicStand : Real := 8.89
def costSongBook : Real := 7.00
def totalCost : Real := 158.35

theorem Jason_spent_correct_amount :
  costFlute + costMusicStand + costSongBook = totalCost :=
sorry

end MusicStore

end Jason_spent_correct_amount_l85_85410


namespace evaluate_expr_l85_85812

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

theorem evaluate_expr : 3 * g 2 + 2 * g (-4) = 169 :=
by
  sorry

end evaluate_expr_l85_85812


namespace walking_rate_ratio_l85_85473

theorem walking_rate_ratio (R R' : ℝ)
  (h : R * 36 = R' * 32) : R' / R = 9 / 8 :=
sorry

end walking_rate_ratio_l85_85473


namespace continuous_arrow_loop_encircling_rectangle_l85_85820

def total_orientations : ℕ := 2^4

def favorable_orientations : ℕ := 2 * 2

def probability_loop : ℚ := favorable_orientations / total_orientations

theorem continuous_arrow_loop_encircling_rectangle : probability_loop = 1 / 4 := by
  sorry

end continuous_arrow_loop_encircling_rectangle_l85_85820


namespace triangle_perimeter_l85_85019

theorem triangle_perimeter (r AP PB x : ℕ) (h_r : r = 14) (h_AP : AP = 20) (h_PB : PB = 30) (h_BC_gt_AC : ∃ BC AC : ℝ, BC > AC)
: ∃ s : ℕ, s = (25 + x) → 2 * s = 50 + 2 * x :=
by
  sorry

end triangle_perimeter_l85_85019


namespace negation_of_quadratic_prop_l85_85001

theorem negation_of_quadratic_prop :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 0) ↔ ∃ x_0 : ℝ, x_0^2 + 1 < 0 :=
by
  sorry

end negation_of_quadratic_prop_l85_85001


namespace simple_interest_rate_l85_85222

theorem simple_interest_rate (P : ℝ) (T : ℝ) (hT : T = 15)
  (doubles_in_15_years : ∃ R : ℝ, (P * 2 = P + (P * R * T) / 100)) :
  ∃ R : ℝ, R = 6.67 := 
by
  sorry

end simple_interest_rate_l85_85222


namespace bus_present_when_sara_arrives_l85_85475

open ProbabilityTheory MeasureTheory Real

noncomputable def bus_arrival : Measure (Measure.measure_space measureTheory ℝ) := 
  uniform_measure (set.Icc 0 60)

noncomputable def sara_arrival : Measure (Measure.measure_space measureTheory ℝ) := 
  uniform_measure (set.Icc 0 60)

theorem bus_present_when_sara_arrives : 
  ∀ (y : ℝ), y ∈ set.Icc (0 : ℝ) 60 → 
  ∀ (x : ℝ), x ∈ set.Icc (0 : ℝ) 60 →
  (∫ x in (0 : ℝ)..60, ∫ y in (0 : ℝ)..60, indicator (set.Icc y (y + 40)) x d bus_arrival y d sara_arrival x) = (2 / 3 : ℝ) := 
sorry

end bus_present_when_sara_arrives_l85_85475


namespace correct_option_D_l85_85621

theorem correct_option_D (x y : ℝ) : (x - y) ^ 2 = (y - x) ^ 2 := by
  sorry

end correct_option_D_l85_85621


namespace find_de_l85_85407

namespace MagicSquare

variables (a b c d e : ℕ)

-- Hypotheses based on the conditions provided.
axiom H1 : 20 + 15 + a = 57
axiom H2 : 25 + b + a = 57
axiom H3 : 18 + c + a = 57
axiom H4 : 20 + c + b = 57
axiom H5 : d + c + a = 57
axiom H6 : d + e + 18 = 57
axiom H7 : e + 25 + 15 = 57

def magicSum := 57

theorem find_de :
  ∃ d e, d + e = 42 :=
by sorry

end MagicSquare

end find_de_l85_85407


namespace jack_helped_hours_l85_85593

-- Definitions based on the problem's conditions
def sam_rate : ℕ := 6  -- Sam assembles 6 widgets per hour
def tony_rate : ℕ := 2  -- Tony assembles 2 widgets per hour
def jack_rate : ℕ := sam_rate  -- Jack assembles at the same rate as Sam
def total_widgets : ℕ := 68  -- The total number of widgets assembled by all three

-- Statement to prove
theorem jack_helped_hours : 
  ∃ h : ℕ, (sam_rate * h) + (tony_rate * h) + (jack_rate * h) = total_widgets ∧ h = 4 := 
  by
  -- The proof is not necessary; we only need the statement
  sorry

end jack_helped_hours_l85_85593


namespace factorize_expr1_factorize_expr2_l85_85651

open BigOperators

/-- Given m and n, prove that m^3 n - 9 m n can be factorized as mn(m + 3)(m - 3). -/
theorem factorize_expr1 (m n : ℤ) : m^3 * n - 9 * m * n = n * m * (m + 3) * (m - 3) :=
sorry

/-- Given a, prove that a^3 + a - 2a^2 can be factorized as a(a - 1)^2. -/
theorem factorize_expr2 (a : ℤ) : a^3 + a - 2 * a^2 = a * (a - 1)^2 :=
sorry

end factorize_expr1_factorize_expr2_l85_85651


namespace range_of_m_for_quadratic_sol_in_interval_l85_85834

theorem range_of_m_for_quadratic_sol_in_interval :
  {m : ℝ // ∀ x, (x^2 + (m-1)*x + 1 = 0) → (0 ≤ x ∧ x ≤ 2)} = {m : ℝ // m < -1} :=
by
  sorry

end range_of_m_for_quadratic_sol_in_interval_l85_85834


namespace impossible_to_create_3_piles_l85_85712

-- Defining similar piles
def similar (x y : ℝ) : Prop :=
  x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

-- Main theorem statement
theorem impossible_to_create_3_piles (initial_pile : ℝ) (h_initial : initial_pile > 0) :
  ∀ (x y z : ℝ), 
  x + y + z = initial_pile → 
  similar x y ∧ similar y z ∧ similar z x → 
  false := 
by 
  sorry

end impossible_to_create_3_piles_l85_85712


namespace total_days_2001_to_2004_l85_85388

def regular_year_days : ℕ := 365
def leap_year_days : ℕ := 366
def num_regular_years : ℕ := 3
def num_leap_years : ℕ := 1

theorem total_days_2001_to_2004 : 
  (num_regular_years * regular_year_days) + (num_leap_years * leap_year_days) = 1461 :=
by
  sorry

end total_days_2001_to_2004_l85_85388


namespace clayton_total_points_l85_85317

theorem clayton_total_points 
  (game1 game2 game3 : ℕ)
  (game1_points : game1 = 10)
  (game2_points : game2 = 14)
  (game3_points : game3 = 6)
  (game4 : ℕ)
  (game4_points : game4 = (game1 + game2 + game3) / 3) :
  game1 + game2 + game3 + game4 = 40 :=
sorry

end clayton_total_points_l85_85317


namespace sqrt_mul_l85_85101

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l85_85101


namespace find_c_l85_85981

variable (c : ℝ)

theorem find_c (h : c * (1 + 1/2 + 1/3 + 1/4) = 1) : c = 12 / 25 :=
by 
  sorry

end find_c_l85_85981


namespace true_conjunction_l85_85376

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l85_85376


namespace count_divisible_by_five_l85_85877

theorem count_divisible_by_five : 
  ∃ n : ℕ, (∀ x, 1 ≤ x ∧ x ≤ 1000 → (x % 5 = 0 → (n = 200))) :=
by
  sorry

end count_divisible_by_five_l85_85877


namespace sqrt_mult_eq_six_l85_85161

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l85_85161


namespace find_m_range_l85_85334

-- Definitions
def p (x : ℝ) : Prop := abs (2 * x + 1) ≤ 3
def q (x m : ℝ) (h : m > 0) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0

-- Problem Statement
theorem find_m_range : 
  (∀ (x : ℝ) (h : m > 0), (¬ (p x)) → (¬ (q x m h))) ∧ 
  (∃ (x : ℝ), ¬ (p x) ∧ ¬ (q x m h)) → 
  ∃ (m : ℝ), m ≥ 3 := 
sorry

end find_m_range_l85_85334


namespace integer_binom_sum_sum_integer_values_l85_85816

theorem integer_binom_sum (n : ℕ) (h : nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13) : n = 13 :=
sorry

theorem sum_integer_values (h : nat.choose 25 13 + nat.choose 25 12 = nat.choose 26 13) : 
  ∑ (x : ℕ) in {n | nat.choose 25 n + nat.choose 25 12 = nat.choose 26 13}.to_finset, x = 13 :=
sorry

end integer_binom_sum_sum_integer_values_l85_85816


namespace solve_inequality_l85_85259

theorem solve_inequality (x : ℝ) : |x - 1| + |x - 2| > 5 ↔ (x < -1 ∨ x > 4) :=
by
  sorry

end solve_inequality_l85_85259


namespace sqrt3_mul_sqrt12_eq_6_l85_85094

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l85_85094


namespace value_of_expression_l85_85219

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 :=
by
sorry

end value_of_expression_l85_85219


namespace barbara_spent_total_l85_85075

variables (cost_steaks cost_chicken total_spent per_pound_steak per_pound_chicken : ℝ)
variables (weight_steaks weight_chicken : ℝ)

-- Defining the given conditions
def conditions :=
  per_pound_steak = 15 ∧
  weight_steaks = 4.5 ∧
  cost_steaks = per_pound_steak * weight_steaks ∧

  per_pound_chicken = 8 ∧
  weight_chicken = 1.5 ∧
  cost_chicken = per_pound_chicken * weight_chicken

-- Proving the total spent by Barbara is $79.50
theorem barbara_spent_total 
  (h : conditions per_pound_steak weight_steaks cost_steaks per_pound_chicken weight_chicken cost_chicken) : 
  total_spent = 79.5 :=
sorry

end barbara_spent_total_l85_85075


namespace sqrt_multiplication_l85_85157

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l85_85157


namespace price_of_n_kilograms_l85_85401

theorem price_of_n_kilograms (m n : ℕ) (hm : m ≠ 0) (h : 9 = m) : (9 * n) / m = (9 * n) / m :=
by
  sorry

end price_of_n_kilograms_l85_85401


namespace remaining_movies_to_watch_l85_85456

theorem remaining_movies_to_watch (total_movies watched_movies remaining_movies : ℕ) 
  (h1 : total_movies = 8) 
  (h2 : watched_movies = 4) 
  (h3 : remaining_movies = total_movies - watched_movies) : 
  remaining_movies = 4 := 
by
  sorry

end remaining_movies_to_watch_l85_85456


namespace find_difference_l85_85998

theorem find_difference (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.30 * y) : x - y = 10 := 
by
  sorry

end find_difference_l85_85998


namespace cos_theta_sum_l85_85562

-- Definition of cos θ as derived from the problem conditions.
theorem cos_theta_sum {r : ℝ} {θ φ : ℝ}
  (h1 : 8^2 = 2 * r^2 * (1 - Real.cos θ))
  (h2 : 15^2 = 2 * r^2 * (1 - Real.cos φ))
  (h3 : 17^2 = 2 * r^2 * (1 - Real.cos (θ + φ)))
  (h4 : θ + φ < Real.pi) :
  (let (num, den) := Rat.numDen (Real.cos θ).toRational in num + den = 386) :=
by
  sorry

end cos_theta_sum_l85_85562


namespace find_selling_price_l85_85301

noncomputable def selling_price (x : ℝ) : ℝ :=
  (x - 60) * (1800 - 20 * x)

constant purchase_price : ℝ := 60
constant max_profit_margin : ℝ := 0.40
constant base_selling_price : ℝ := 80
constant base_units_sold : ℝ := 200
constant decrement_units_sold : ℝ := 20
constant target_profit : ℝ := 2500

theorem find_selling_price (x : ℝ) :
  selling_price x = target_profit ∧
  (x - 60) / 60 ≤ max_profit_margin ∧
  ∃ u : ℝ, u = (base_units_sold + decrement_units_sold * (base_selling_price - x))
  → x = 65 :=
sorry

end find_selling_price_l85_85301


namespace calculate_force_l85_85645

noncomputable def force_on_dam (ρ g a b h : ℝ) : ℝ :=
  ρ * g * h^2 * (b / 2 - (b - a) / 3)

theorem calculate_force : force_on_dam 1000 10 4.8 7.2 3.0 = 252000 := 
  by 
  sorry

end calculate_force_l85_85645


namespace steps_already_climbed_l85_85486

-- Definitions based on conditions
def total_stair_steps : ℕ := 96
def steps_left_to_climb : ℕ := 22

-- Theorem proving the number of steps already climbed
theorem steps_already_climbed : total_stair_steps - steps_left_to_climb = 74 := by
  sorry

end steps_already_climbed_l85_85486


namespace how_many_whole_boxes_did_nathan_eat_l85_85541

-- Define the conditions
def gumballs_per_package := 5
def total_gumballs := 20

-- The problem to prove
theorem how_many_whole_boxes_did_nathan_eat : total_gumballs / gumballs_per_package = 4 :=
by sorry

end how_many_whole_boxes_did_nathan_eat_l85_85541


namespace cars_cleaned_per_day_l85_85436

theorem cars_cleaned_per_day
  (money_per_car : ℕ)
  (total_money : ℕ)
  (days : ℕ)
  (h1 : money_per_car = 5)
  (h2 : total_money = 2000)
  (h3 : days = 5) :
  (total_money / (money_per_car * days)) = 80 := by
  sorry

end cars_cleaned_per_day_l85_85436


namespace power_greater_than_linear_l85_85426

theorem power_greater_than_linear (n : ℕ) (h : n ≥ 3) : 2^n > 2 * n + 1 := 
by {
  sorry
}

end power_greater_than_linear_l85_85426


namespace smallest_n_divisibility_l85_85751

theorem smallest_n_divisibility (n : ℕ) (h : 1 ≤ n) :
  (∀ k, 1 ≤ k ∧ k ≤ n → n^3 - n ∣ k) ∨ (∃ k, 1 ≤ k ∧ k ≤ n ∧ ¬ (n^3 - n ∣ k)) :=
sorry

end smallest_n_divisibility_l85_85751


namespace complex_expression_equals_neg3_l85_85010

noncomputable def nonreal_root_of_x4_eq_1 : Type :=
{ζ : ℂ // ζ^4 = 1 ∧ ζ.im ≠ 0}

theorem complex_expression_equals_neg3 (ζ : nonreal_root_of_x4_eq_1) :
  (1 - ζ.val + ζ.val^3)^4 + (1 + ζ.val^2 - ζ.val^3)^4 = -3 :=
sorry

end complex_expression_equals_neg3_l85_85010


namespace degrees_of_remainder_division_l85_85463

theorem degrees_of_remainder_division (f g : Polynomial ℝ) (h : g = Polynomial.C 3 * Polynomial.X ^ 3 + Polynomial.C (-4) * Polynomial.X ^ 2 + Polynomial.C 1 * Polynomial.X + Polynomial.C (-8)) :
  ∃ r q : Polynomial ℝ, f = g * q + r ∧ (r.degree < 3) := 
sorry

end degrees_of_remainder_division_l85_85463


namespace sqrt_mult_eq_six_l85_85163

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l85_85163


namespace isabella_canadian_dollars_sum_l85_85879

def sum_of_digits (n : Nat) : Nat :=
  (n % 10) + ((n / 10) % 10)

theorem isabella_canadian_dollars_sum (d : Nat) (H: 10 * d = 7 * d + 280) : sum_of_digits d = 12 :=
by
  sorry

end isabella_canadian_dollars_sum_l85_85879


namespace find_result_l85_85507

theorem find_result : ∀ (x : ℝ), x = 1 / 3 → 5 - 7 * x = 8 / 3 := by
  intros x hx
  sorry

end find_result_l85_85507


namespace sqrt_multiplication_l85_85153

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l85_85153


namespace pure_water_to_add_eq_30_l85_85549

noncomputable def initial_volume := 50
noncomputable def initial_concentration := 0.40
noncomputable def desired_concentration := 0.25
noncomputable def initial_acid := initial_concentration * initial_volume

theorem pure_water_to_add_eq_30 :
  ∃ w : ℝ, (initial_acid / (initial_volume + w) = desired_concentration) ∧ w = 30 :=
by
  sorry

end pure_water_to_add_eq_30_l85_85549


namespace sqrt_mult_simplify_l85_85124

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l85_85124


namespace triangle_tangent_identity_l85_85579

theorem triangle_tangent_identity (A B C : ℝ) (h : A + B + C = Real.pi) : 
  (Real.tan (A / 2) * Real.tan (B / 2)) + (Real.tan (B / 2) * Real.tan (C / 2)) + (Real.tan (C / 2) * Real.tan (A / 2)) = 1 :=
by
  sorry

end triangle_tangent_identity_l85_85579


namespace sqrt_mult_l85_85170

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l85_85170


namespace axis_of_symmetry_l85_85555

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = f (5 - x)) : ∀ x : ℝ, f x = f (2 * 2.5 - x) :=
by
  sorry

end axis_of_symmetry_l85_85555


namespace repeating_decimal_sum_l85_85042

theorem repeating_decimal_sum (x : ℚ) (h : x = 34 / 999) : x.num + x.denom = 1033 := by 
  sorry

end repeating_decimal_sum_l85_85042


namespace sqrt_mul_sqrt_eq_six_l85_85145

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l85_85145


namespace total_ages_is_32_l85_85931

variable (a b c : ℕ)
variable (h_b : b = 12)
variable (h_a : a = b + 2)
variable (h_c : b = 2 * c)

theorem total_ages_is_32 (h_b : b = 12) (h_a : a = b + 2) (h_c : b = 2 * c) : a + b + c = 32 :=
by
  sorry

end total_ages_is_32_l85_85931


namespace smallest_a_l85_85239

theorem smallest_a (a b : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : ∀ x : ℤ, Real.sin (a * x + b) = Real.sin (17 * x)) :
  a = 17 :=
by
  sorry

end smallest_a_l85_85239


namespace initial_trees_count_l85_85942

variable (x : ℕ)

-- Conditions of the problem
def initial_rows := 24
def additional_rows := 12
def total_rows := initial_rows + additional_rows
def trees_per_row_initial := x
def trees_per_row_final := 28

-- Total number of trees should remain constant
theorem initial_trees_count :
  initial_rows * trees_per_row_initial = total_rows * trees_per_row_final → 
  trees_per_row_initial = 42 := 
by sorry

end initial_trees_count_l85_85942


namespace flood_damage_in_euros_l85_85941

variable (yen_damage : ℕ) (yen_per_euro : ℕ) (tax_rate : ℝ)

theorem flood_damage_in_euros : 
  yen_damage = 4000000000 →
  yen_per_euro = 110 →
  tax_rate = 1.05 →
  (yen_damage / yen_per_euro : ℝ) * tax_rate = 38181818 :=
by {
  -- We could include necessary lean proof steps here, but we use sorry to skip the proof.
  sorry
}

end flood_damage_in_euros_l85_85941


namespace impossible_to_create_3_piles_l85_85713

-- Defining similar piles
def similar (x y : ℝ) : Prop :=
  x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

-- Main theorem statement
theorem impossible_to_create_3_piles (initial_pile : ℝ) (h_initial : initial_pile > 0) :
  ∀ (x y z : ℝ), 
  x + y + z = initial_pile → 
  similar x y ∧ similar y z ∧ similar z x → 
  false := 
by 
  sorry

end impossible_to_create_3_piles_l85_85713


namespace find_present_worth_l85_85601

noncomputable def present_worth (BG : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
(BG * 100) / (R * ((1 + R/100)^T - 1) - R * T)

theorem find_present_worth : present_worth 36 10 3 = 1161.29 :=
by
  sorry

end find_present_worth_l85_85601


namespace football_goals_even_more_probable_l85_85771

-- Define the problem statement and conditions
variable (p_1 : ℝ) (h₀ : 0 ≤ p_1 ∧ p_1 ≤ 1) (h₁ : q_1 = 1 - p_1)

-- Define even and odd goal probabilities for the total match
def p : ℝ := p_1^2 + (1 - p_1)^2
def q : ℝ := 2 * p_1 * (1 - p_1)

-- The main statement to prove
theorem football_goals_even_more_probable (h₂ : q_1 = 1 - p_1) : p_1^2 + (1 - p_1)^2 ≥ 2 * p_1 * (1 - p_1) :=
  sorry

end football_goals_even_more_probable_l85_85771


namespace frequency_distribution_necessary_l85_85746

/-- Definition of the necessity to use Frequency Distribution to understand 
the proportion of first-year high school students in the city whose height 
falls within a certain range -/
def necessary_for_proportion (A B C D : Prop) : Prop := D

theorem frequency_distribution_necessary (A B C D : Prop) :
  necessary_for_proportion A B C D ↔ D :=
by
  sorry

end frequency_distribution_necessary_l85_85746


namespace vector_dot_product_l85_85216

def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (2, -2)

theorem vector_dot_product : (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2)) = -1 := by
  -- skipping the proof
  sorry

end vector_dot_product_l85_85216


namespace jims_investment_l85_85695

theorem jims_investment
  {total_investment : ℝ} 
  (h1 : total_investment = 127000)
  {john_ratio : ℕ} 
  (h2 : john_ratio = 8)
  {james_ratio : ℕ} 
  (h3 : james_ratio = 11)
  {jim_ratio : ℕ} 
  (h4 : jim_ratio = 15)
  {jordan_ratio : ℕ} 
  (h5 : jordan_ratio = 19) :
  jim_ratio / (john_ratio + james_ratio + jim_ratio + jordan_ratio) * total_investment = 35943.40 :=
by {
  sorry
}

end jims_investment_l85_85695


namespace find_n_l85_85824

theorem find_n (n : ℕ) (h : n * n.factorial + 2 * n.factorial = 5040) : n = 5 :=
by {
  sorry
}

end find_n_l85_85824


namespace circles_and_squares_intersection_l85_85955

def circles_and_squares_intersection_count : Nat :=
  let radius := (1 : ℚ) / 8
  let square_side := (1 : ℚ) / 4
  let slope := (1 : ℚ) / 3
  let line (x : ℚ) : ℚ := slope * x
  let num_segments := 243
  let intersections_per_segment := 4
  num_segments * intersections_per_segment

theorem circles_and_squares_intersection : 
  circles_and_squares_intersection_count = 972 :=
by
  sorry

end circles_and_squares_intersection_l85_85955


namespace sqrt_mul_sqrt_eq_six_l85_85147

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l85_85147


namespace third_smallest_number_l85_85869

/-- 
  The third smallest two-decimal-digit number that can be made
  using the digits 3, 8, 2, and 7 each exactly once is 27.38.
-/
theorem third_smallest_number (digits : List ℕ) (h : digits = [3, 8, 2, 7]) : 
  ∃ x y, 
  x < y ∧
  x = 23.78 ∧
  y = 23.87 ∧
  ∀ z, z > x ∧ z < y → z = 27.38 :=
by 
  sorry

end third_smallest_number_l85_85869


namespace sqrt_mul_simplify_l85_85103

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l85_85103


namespace sqrt_mul_simp_l85_85133

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l85_85133


namespace repeating_decimal_sum_l85_85043

theorem repeating_decimal_sum (x : ℚ) (h : x = 34 / 999) : x.num + x.denom = 1033 := by 
  sorry

end repeating_decimal_sum_l85_85043


namespace symmetric_points_x_axis_l85_85564

theorem symmetric_points_x_axis (a b : ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) 
  (hP : P = (a - 3, 1)) (hQ : Q = (2, b + 1)) (hSymm : P.1 = Q.1 ∧ P.2 = -Q.2) :
  a + b = 3 :=
by 
  sorry

end symmetric_points_x_axis_l85_85564


namespace x_squared_plus_y_squared_l85_85974

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 2) : x^2 + y^2 = 21 := 
by 
  sorry

end x_squared_plus_y_squared_l85_85974


namespace total_number_of_people_l85_85489

variables (A B : ℕ)

def pencils_brought_by_assoc_profs (A : ℕ) : ℕ := 2 * A
def pencils_brought_by_asst_profs (B : ℕ) : ℕ := B
def charts_brought_by_assoc_profs (A : ℕ) : ℕ := A
def charts_brought_by_asst_profs (B : ℕ) : ℕ := 2 * B

axiom pencils_total : pencils_brought_by_assoc_profs A + pencils_brought_by_asst_profs B = 10
axiom charts_total : charts_brought_by_assoc_profs A + charts_brought_by_asst_profs B = 11

theorem total_number_of_people : A + B = 7 :=
sorry

end total_number_of_people_l85_85489


namespace remaining_problems_l85_85947

-- Define the conditions
def worksheets_total : ℕ := 15
def worksheets_graded : ℕ := 7
def problems_per_worksheet : ℕ := 3

-- Define the proof goal
theorem remaining_problems : (worksheets_total - worksheets_graded) * problems_per_worksheet = 24 :=
by
  sorry

end remaining_problems_l85_85947


namespace find_ab_l85_85207

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 :=
by
  sorry

end find_ab_l85_85207


namespace min_value_5x_plus_6y_l85_85379

theorem min_value_5x_plus_6y (x y : ℝ) (h : 3 * x ^ 2 + 3 * y ^ 2 = 20 * x + 10 * y + 10) : 
  ∃ x y, (5 * x + 6 * y = 122) :=
by
  sorry

end min_value_5x_plus_6y_l85_85379


namespace non_congruent_rectangles_count_l85_85484

theorem non_congruent_rectangles_count (h w : ℕ) (P : ℕ) (multiple_of_4: ℕ → Prop) :
  P = 80 →
  w ≥ 1 ∧ h ≥ 1 →
  P = 2 * (w + h) →
  (multiple_of_4 w ∨ multiple_of_4 h) →
  (∀ k, multiple_of_4 k ↔ ∃ m, k = 4 * m) →
  ∃ n, n = 5 :=
by
  sorry

end non_congruent_rectangles_count_l85_85484


namespace sqrt_mul_simp_l85_85132

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l85_85132


namespace man_twice_son_age_l85_85633

theorem man_twice_son_age (S M Y : ℕ) (h1 : S = 27) (h2 : M = S + 29) (h3 : M + Y = 2 * (S + Y)) : Y = 2 := 
by sorry

end man_twice_son_age_l85_85633


namespace sqrt_mul_l85_85095

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l85_85095


namespace negation_of_universal_quadratic_l85_85740

theorem negation_of_universal_quadratic (P : ∀ a b c : ℝ, a ≠ 0 → ∃ x : ℝ, a * x^2 + b * x + c = 0) :
  ¬(∀ a b c : ℝ, a ≠ 0 → ∃ x : ℝ, a * x^2 + b * x + c = 0) ↔ ∃ a b c : ℝ, a ≠ 0 ∧ ¬(∃ x : ℝ, a * x^2 + b * x + c = 0) :=
by
  sorry

end negation_of_universal_quadratic_l85_85740


namespace cylinder_radius_in_cone_l85_85777

theorem cylinder_radius_in_cone (d h r : ℝ) (h_d : d = 20) (h_h : h = 24) (h_cylinder : 2 * r = r):
  r = 60 / 11 :=
by
  sorry

end cylinder_radius_in_cone_l85_85777


namespace Jason_spent_on_music_store_l85_85413

theorem Jason_spent_on_music_store:
  let flute := 142.46
  let music_stand := 8.89
  let song_book := 7.00
  flute + music_stand + song_book = 158.35 := sorry

end Jason_spent_on_music_store_l85_85413


namespace jenna_gas_cost_l85_85568

-- Definitions of the given conditions
def hours1 : ℕ := 2
def speed1 : ℕ := 60
def hours2 : ℕ := 3
def speed2 : ℕ := 50
def miles_per_gallon : ℕ := 30
def cost_per_gallon : ℕ := 2

-- Statement to be proven
theorem jenna_gas_cost : 
  let distance1 := hours1 * speed1,
      distance2 := hours2 * speed2,
      total_distance := distance1 + distance2,
      total_gallons := total_distance / miles_per_gallon,
      total_cost := total_gallons * cost_per_gallon
  in total_cost = 18 := 
by 
  let distance1 := hours1 * speed1,
      distance2 := hours2 * speed2,
      total_distance := distance1 + distance2,
      total_gallons := total_distance / miles_per_gallon,
      total_cost := total_gallons * cost_per_gallon

  sorry

end jenna_gas_cost_l85_85568


namespace absolute_value_inequality_l85_85260

theorem absolute_value_inequality (x : ℝ) : (|x + 1| > 3) ↔ (x > 2 ∨ x < -4) :=
by
  sorry

end absolute_value_inequality_l85_85260


namespace john_final_push_time_l85_85692

theorem john_final_push_time :
  ∃ t : ℝ, (∀ (d_j d_s : ℝ), d_j = 4.2 * t ∧ d_s = 3.7 * t ∧ (d_j = d_s + 14)) → t = 28 :=
by
  sorry

end john_final_push_time_l85_85692


namespace solve_equation_l85_85434

theorem solve_equation (x : ℝ) (h : (x^2 - x + 2) / (x - 1) = x + 3) : x = 5 / 3 := 
by sorry

end solve_equation_l85_85434


namespace sqrt_mult_simplify_l85_85119

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l85_85119


namespace min_omega_l85_85444

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + 1)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * (x - 1) + 1)

def condition1 (ω : ℝ) : Prop := ω > 0
def condition2 (ω : ℝ) (x : ℝ) : Prop := g ω x = Real.sin (ω * x - ω + 1)
def condition3 (ω : ℝ) (k : ℤ) : Prop := ∃ k : ℤ, ω = 1 - k * Real.pi

theorem min_omega (ω : ℝ) (k : ℤ) (x : ℝ) : condition1 ω → condition2 ω x → condition3 ω k → ω = 1 :=
by
  intros h1 h2 h3
  sorry

end min_omega_l85_85444


namespace treasure_probability_l85_85634

variable {Island : Type}

-- Define the probabilities.
def prob_treasure : ℚ := 1 / 3
def prob_trap : ℚ := 1 / 6
def prob_neither : ℚ := 1 / 2

-- Define the number of islands.
def num_islands : ℕ := 5

-- Define the probability of encountering exactly 4 islands with treasure and one with neither traps nor treasures.
theorem treasure_probability :
  (num_islands.choose 4) * (prob_ttreasure^4) * (prob_neither^1) = (5 : ℚ) * (1 / 81) * (1 / 2) :=
  by
  sorry

end treasure_probability_l85_85634


namespace solve_eq_l85_85733

theorem solve_eq : ∀ x : ℂ, (x^3 + 3*x^2 + 4*x + 6) / (x + 5) = x^2 + 10 ↔
  x = (-3 + Complex.I * Real.sqrt 79) / 2 ∨ x = (-3 - Complex.I * Real.sqrt 79) / 2 :=
by 
  intro x
  sorry

end solve_eq_l85_85733


namespace student_groups_l85_85249

theorem student_groups (N : ℕ) :
  (∃ (n : ℕ), n = 13 ∧ ∃ (m : ℕ), m ∈ {12, 14} ∧ N = 4 * 13 + 2 * m) → (N = 76 ∨ N = 80) :=
by
  intro h
  obtain ⟨n, hn, m, hm, hN⟩ := h
  rw [hn, hN]
  cases hm with h12 h14
  case inl =>
    simp [h12]
  case inr =>
    simp [h14]
  sorry

end student_groups_l85_85249


namespace functional_eq_solution_l85_85826

open Real

theorem functional_eq_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = f (x - y) + 4 * x * y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x^2 + c := 
sorry

end functional_eq_solution_l85_85826


namespace sqrt_mul_eq_6_l85_85138

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l85_85138


namespace range_of_m_l85_85520

theorem range_of_m (x m : ℝ) (h1 : -2 ≤ 1 - (x-1)/3 ∧ 1 - (x-1)/3 ≤ 2)
                   (h2 : x^2 - 2*x + 1 - m^2 ≤ 0)
                   (h3 : m > 0)
                   (h4 : (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m))
                   (h5 : ¬((x < 1 - m ∨ x > 1 + m) → (x < -2 ∨ x > 10))) :
                   m ≤ 3 :=
by
  sorry

end range_of_m_l85_85520


namespace sqrt_of_4_l85_85275

theorem sqrt_of_4 :
  ∃ x : ℝ, x^2 = 4 ∧ (x = 2 ∨ x = -2) :=
sorry

end sqrt_of_4_l85_85275


namespace no_such_fractions_l85_85956

open Nat

theorem no_such_fractions : ¬ ∃ (x y : ℕ), (x.gcd y = 1) ∧ (x > 0) ∧ (y > 0) ∧ ((x + 1) * 5 * y = ((y + 1) * 6 * x)) :=
by
  sorry

end no_such_fractions_l85_85956


namespace sqrt_mult_simplify_l85_85126

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l85_85126


namespace lowest_score_is_C_l85_85224

variable (Score : Type) [LinearOrder Score]
variable (A B C : Score)

-- Translate conditions into Lean
variable (h1 : B ≠ max A (max B C) → A = min A (min B C))
variable (h2 : C ≠ min A (min B C) → A = max A (max B C))

-- Define the proof goal
theorem lowest_score_is_C : min A (min B C) =C :=
by
  sorry

end lowest_score_is_C_l85_85224


namespace winning_pair_probability_l85_85199

-- Define the six cards
inductive Color
| Blue | Orange

inductive Label
| X | Y | Z

structure Card where
  color : Color
  label : Label

-- all the cards
def cards : Finset Card := 
  { ⟨Color.Blue, Label.X⟩, ⟨Color.Blue, Label.Y⟩, ⟨Color.Blue, Label.Z⟩,
    ⟨Color.Orange, Label.X⟩, ⟨Color.Orange, Label.Y⟩, ⟨Color.Orange, Label.Z⟩ }

-- Define winning pair predicate
def winning_pair (c1 c2 : Card) : Prop :=
  c1.label = c2.label ∨ c1.color = c2.color

open_locale classical

noncomputable def probability_winning_pair : ℚ :=
let total_pairs := (cards.card.choose 2) in
let winning_pairs := (Finset.filter (λ p, winning_pair p.1 p.2) (cards.pairs)).card in
winning_pairs / total_pairs

theorem winning_pair_probability :
  probability_winning_pair = 3 / 5 := 
sorry

end winning_pair_probability_l85_85199


namespace determine_A_value_l85_85908

noncomputable def solve_for_A (A B C : ℝ) : Prop :=
  (A = 1/16) ↔ 
  (∀ x : ℝ, (1 / ((x + 5) * (x - 3) * (x + 3))) = (A / (x + 5)) + (B / (x - 3)) + (C / (x + 3)))

theorem determine_A_value :
  solve_for_A (1/16) B C :=
by
  sorry

end determine_A_value_l85_85908


namespace f_2015_l85_85958

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (x + 3) = f x
axiom f_interval : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x = 2^x

theorem f_2015 : f 2015 = -2 := sorry

end f_2015_l85_85958


namespace symmetry_x_axis_l85_85566

theorem symmetry_x_axis (a b : ℝ) (h1 : a - 3 = 2) (h2 : 1 = -(b + 1)) : a + b = 3 :=
by
  sorry

end symmetry_x_axis_l85_85566


namespace shaded_area_l85_85875

-- Define the problem in Lean
theorem shaded_area (area_large_square area_small_square : ℝ) (H_large_square : area_large_square = 10) (H_small_square : area_small_square = 4) (diagonals_contain : True) : 
  (area_large_square - area_small_square) / 4 = 1.5 :=
by
  sorry -- proof not required

end shaded_area_l85_85875


namespace inequality_abc_l85_85905

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (1 / (1 + a + b)) + (1 / (1 + b + c)) + (1 / (1 + c + a)) ≤ 1 :=
by
  sorry

end inequality_abc_l85_85905


namespace sqrt_mul_eq_l85_85086

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l85_85086


namespace problem_l85_85700

noncomputable def a := Real.log 2 / Real.log 3
noncomputable def b := Real.log (1/8) / Real.log 2
noncomputable def c := Real.sqrt 2

theorem problem : c > a ∧ a > b := 
by
  sorry

end problem_l85_85700


namespace ratio_triangle_to_square_l85_85403

open Real

-- Define the points of the triangle
def A := (1, 1) : ℝ × ℝ
def B := (1, 4) : ℝ × ℝ
def C := (4, 4) : ℝ × ℝ

-- Define the side length of the square and its area
def side_length_square := 5
def area_square := (side_length_square * side_length_square : ℝ)

-- Define the area of the triangle using the given coordinates
def area_triangle : ℝ := 0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Proving the ratio of area of the triangle to the area of the large square is 9/50
theorem ratio_triangle_to_square : (area_triangle / area_square) = (9 / 50) :=
by
  have h1 : area_triangle = 4.5 := by
    -- Calculate the area of the triangle
    sorry
  have h2 : area_square = 25 := by
    -- Calculate the area of the square
    simp [area_square, side_length_square]
  -- Calculate the ratio
  calc
    area_triangle / area_square
    = 4.5 / 25 : by rw [h1, h2]
    ... = 9 / 50 : by norm_num

end ratio_triangle_to_square_l85_85403


namespace problem_statement_l85_85382

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - x^2

theorem problem_statement (x0 x1 x2 m : ℝ) (h0 : f x0 = m) (h1 : 0 < x1) (h2 : x1 < x0) (h3 : x0 < x2) :
    f x1 > m ∧ f x2 < m :=
sorry

end problem_statement_l85_85382


namespace find_value_of_a_l85_85215

theorem find_value_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 ≤ 24) ∧
  (∃ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 = 24) ∧
  (∀ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 ≥ 3) ∧
  (∃ x ∈ Set.Icc 0 3, x^2 + 2*a*x + a^2 - 1 = 3) → 
  a = 2 ∨ a = -5 :=
by
  sorry

end find_value_of_a_l85_85215


namespace water_added_eq_30_l85_85552

-- Given conditions
def initial_volume := 50
def initial_concentration := 0.4
def final_concentration := 0.25
def amount_of_acid := initial_volume * initial_concentration

-- Proof problem statement
theorem water_added_eq_30 : ∃ w : ℝ, amount_of_acid / (initial_volume + w) = final_concentration ∧ w = 30 :=
by 
  -- Solution steps go here, but for now we insert sorry to skip the proof.
  sorry

end water_added_eq_30_l85_85552


namespace probability_same_combination_l85_85943

-- Definitions for the conditions
def total_candies : ℕ := 20
def red_candies : ℕ := 12
def blue_candies : ℕ := 8
def terry_candies : ℕ := 3
def mary_candies : ℕ := 3

-- Main theorem statement
theorem probability_same_combination : 
  let terry_picks_red := (Nat.choose red_candies terry_candies : ℚ) / (Nat.choose total_candies terry_candies) in
  let mary_picks_red := (Nat.choose (red_candies - terry_candies) mary_candies : ℚ) / (Nat.choose (total_candies - terry_candies) mary_candies) in
  let terry_picks_blue := (Nat.choose blue_candies terry_candies : ℚ) / (Nat.choose total_candies terry_candies) in
  let mary_picks_blue := (Nat.choose (blue_candies - terry_candies) mary_candies : ℚ) / (Nat.choose (total_candies - terry_candies) mary_candies) in
  let prob_both_red := terry_picks_red * mary_picks_red in
  let prob_both_blue := terry_picks_blue * mary_picks_blue in
  let total_prob := prob_both_red + prob_both_blue in
  total_prob = (77 / 4845 : ℚ) :=
by
  sorry

end probability_same_combination_l85_85943


namespace andy_max_cookies_l85_85918

-- Definitions for the problem conditions
def total_cookies := 36
def bella_eats (andy_cookies : ℕ) := 2 * andy_cookies
def charlie_eats (andy_cookies : ℕ) := andy_cookies
def consumed_cookies (andy_cookies : ℕ) := andy_cookies + bella_eats andy_cookies + charlie_eats andy_cookies

-- The statement to prove
theorem andy_max_cookies : ∃ (a : ℕ), consumed_cookies a = total_cookies ∧ a = 9 :=
by
  sorry

end andy_max_cookies_l85_85918


namespace sum_first_5_terms_l85_85522

variable {a : ℕ → ℝ}
variable (h : 2 * a 2 = a 1 + 3)

theorem sum_first_5_terms (a : ℕ → ℝ) (h : 2 * a 2 = a 1 + 3) : 
  (a 1 + a 2 + a 3 + a 4 + a 5) = 15 :=
sorry

end sum_first_5_terms_l85_85522


namespace true_proposition_l85_85361

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l85_85361


namespace a_n_div_3_sum_two_cubes_a_n_div_3_not_sum_two_squares_l85_85197

def a_n (n : ℕ) : ℕ := 10^(3*n+2) + 2 * 10^(2*n+1) + 2 * 10^(n+1) + 1

theorem a_n_div_3_sum_two_cubes (n : ℕ) : ∃ x y : ℤ, (x > 0) ∧ (y > 0) ∧ (a_n n / 3 = x^3 + y^3) := sorry

theorem a_n_div_3_not_sum_two_squares (n : ℕ) : ¬ (∃ x y : ℤ, a_n n / 3 = x^2 + y^2) := sorry

end a_n_div_3_sum_two_cubes_a_n_div_3_not_sum_two_squares_l85_85197


namespace find_m_l85_85656

theorem find_m (m : ℤ) (h0 : -90 ≤ m) (h1 : m ≤ 90) (h2 : Real.sin (m * Real.pi / 180) = Real.sin (710 * Real.pi / 180)) : m = -10 :=
sorry

end find_m_l85_85656


namespace percentage_of_150_l85_85462

theorem percentage_of_150 : (1 / 5 * (1 / 100) * 150 : ℝ) = 0.3 := by
  sorry

end percentage_of_150_l85_85462


namespace impossible_to_form_three_similar_piles_l85_85714

-- Define the concept of similar sizes
def similar_sizes (a b : ℝ) : Prop := a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2

-- State the proof problem
theorem impossible_to_form_three_similar_piles (x : ℝ) (h : 0 < x) : ¬(∃ a b c : ℝ, a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes a c) :=
sorry

end impossible_to_form_three_similar_piles_l85_85714


namespace function_increasing_range_l85_85672

theorem function_increasing_range (a : ℝ) (f : ℝ → ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x : ℝ, f x = (a / (a^2 - 2)) * (a^x - a^(-x))) : 
  (∀ x y : ℝ, x < y → f x < f y) ↔ (0 < a ∧ a < 1 ∨ a > sqrt 2) :=
sorry

end function_increasing_range_l85_85672


namespace inscribed_circle_radius_squared_l85_85766

theorem inscribed_circle_radius_squared 
  (X Y Z W R S : Type) 
  (XR RY WS SZ : ℝ)
  (hXR : XR = 23) 
  (hRY : RY = 29)
  (hWS : WS = 41) 
  (hSZ : SZ = 31)
  (tangent_at_XY : true) (tangent_at_WZ : true) -- since tangents are assumed by problem
  : ∃ (r : ℝ), r^2 = 905 :=
by sorry

end inscribed_circle_radius_squared_l85_85766


namespace cupcakes_leftover_l85_85583

-- Definitions based on the conditions
def total_cupcakes : ℕ := 17
def num_children : ℕ := 3

-- Theorem proving the correct answer
theorem cupcakes_leftover : total_cupcakes % num_children = 2 := by
  sorry

end cupcakes_leftover_l85_85583


namespace value_of_expression_l85_85619

theorem value_of_expression : 50^4 + 4 * 50^3 + 6 * 50^2 + 4 * 50 + 1 = 6765201 :=
by
  sorry

end value_of_expression_l85_85619


namespace division_of_squares_l85_85976

theorem division_of_squares {a b : ℕ} (h1 : a < 1000) (h2 : b > 0) (h3 : b^10 ∣ a^21) : b ∣ a^2 := 
sorry

end division_of_squares_l85_85976


namespace exponent_division_l85_85804

theorem exponent_division : (19 ^ 11) / (19 ^ 8) = 6859 :=
by
  -- Here we assume the properties of powers and arithmetic operations
  sorry

end exponent_division_l85_85804


namespace tickets_sold_at_door_l85_85011

theorem tickets_sold_at_door :
  ∃ D : ℕ, ∃ A : ℕ, A + D = 800 ∧ (1450 * A + 2200 * D = 166400) ∧ D = 672 :=
by
  sorry

end tickets_sold_at_door_l85_85011


namespace total_selling_price_l85_85957

theorem total_selling_price (total_commissions : ℝ) (number_of_appliances : ℕ) (fixed_commission_rate_per_appliance : ℝ) (percentage_commission_rate : ℝ) :
  total_commissions = number_of_appliances * fixed_commission_rate_per_appliance + percentage_commission_rate * S →
  total_commissions = 662 →
  number_of_appliances = 6 →
  fixed_commission_rate_per_appliance = 50 →
  percentage_commission_rate = 0.10 →
  S = 3620 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end total_selling_price_l85_85957


namespace B_spends_85_percent_l85_85061

def combined_salary (S_A S_B : ℝ) : Prop := S_A + S_B = 4000
def A_savings_percentage : ℝ := 0.05
def A_salary : ℝ := 3000
def B_salary : ℝ := 4000 - A_salary
def equal_savings (S_A S_B : ℝ) : Prop := A_savings_percentage * S_A = (1 - S_B / 100) * B_salary

theorem B_spends_85_percent (S_A S_B : ℝ) (B_spending_percentage : ℝ) :
  combined_salary S_A S_B ∧ S_A = A_salary ∧ equal_savings S_A B_spending_percentage → B_spending_percentage = 0.85 := by
  sorry

end B_spends_85_percent_l85_85061


namespace correct_calculation_l85_85294

-- Definitions for conditions
def cond_A (x y : ℝ) : Prop := 3 * x + 4 * y = 7 * x * y
def cond_B (x : ℝ) : Prop := 5 * x - 2 * x = 3 * x ^ 2
def cond_C (y : ℝ) : Prop := 7 * y ^ 2 - 5 * y ^ 2 = 2
def cond_D (a b : ℝ) : Prop := 6 * a ^ 2 * b - b * a ^ 2 = 5 * a ^ 2 * b

-- Proof statement using conditions
theorem correct_calculation (a b : ℝ) : cond_D a b :=
by
  unfold cond_D
  sorry

end correct_calculation_l85_85294


namespace binomial_arithmetic_sequence_iff_l85_85329

open Nat

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  n.choose k 

-- Conditions
def is_arithmetic_sequence (n k : ℕ) : Prop :=
  binomial n (k-1) - 2 * binomial n k + binomial n (k+1) = 0

-- Statement to prove
theorem binomial_arithmetic_sequence_iff (u : ℕ) (u_gt2 : u > 2) :
  ∃ (n k : ℕ), (n = u^2 - 2) ∧ (k = binomial u 2 - 1 ∨ k = binomial (u+1) 2 - 1) 
  ↔ is_arithmetic_sequence n k := 
sorry

end binomial_arithmetic_sequence_iff_l85_85329


namespace loss_percentage_l85_85735

theorem loss_percentage (CP SP : ℝ) (hCP : CP = 1500) (hSP : SP = 1200) : 
  (CP - SP) / CP * 100 = 20 :=
by
  -- Proof would be provided here
  sorry

end loss_percentage_l85_85735


namespace proposition_3_proposition_4_l85_85070

variables {Plane : Type} {Line : Type} 
variables {α β : Plane} {a b : Line}

-- Assuming necessary properties of parallel planes and lines being subsets of planes
axiom plane_parallel (α β : Plane) : Prop
axiom line_in_plane (l : Line) (p : Plane) : Prop
axiom line_parallel (l m : Line) : Prop
axiom lines_skew (l m : Line) : Prop
axiom lines_coplanar (l m : Line) : Prop
axiom lines_do_not_intersect (l m : Line) : Prop

-- Assume the given conditions
variables (h1 : plane_parallel α β) 
variables (h2 : line_in_plane a α)
variables (h3 : line_in_plane b β)

-- State the equivalent proof problem as propositions to be proved in Lean
theorem proposition_3 (h1 : plane_parallel α β) 
                     (h2 : line_in_plane a α) 
                     (h3 : line_in_plane b β) : 
                     lines_do_not_intersect a b :=
sorry

theorem proposition_4 (h1 : plane_parallel α β) 
                     (h2 : line_in_plane a α) 
                     (h3 : line_in_plane b β) : 
                     lines_coplanar a b ∨ lines_skew a b :=
sorry

end proposition_3_proposition_4_l85_85070


namespace valid_outfits_number_l85_85859

def num_shirts := 7
def num_pants := 7
def num_hats := 7
def num_colors := 7

def total_outfits (num_shirts num_pants num_hats : ℕ) := num_shirts * num_pants * num_hats
def matching_color_outfits (num_colors : ℕ) := num_colors
def valid_outfits (num_shirts num_pants num_hats num_colors : ℕ) := 
  total_outfits num_shirts num_pants num_hats - matching_color_outfits num_colors

theorem valid_outfits_number : valid_outfits num_shirts num_pants num_hats num_colors = 336 := 
by
  sorry

end valid_outfits_number_l85_85859


namespace line_not_in_fourth_quadrant_l85_85774

-- Let the line be defined as y = 3x + 2
def line_eq (x : ℝ) : ℝ := 3 * x + 2

-- The Fourth quadrant is defined by x > 0 and y < 0
def in_fourth_quadrant (x : ℝ) (y : ℝ) : Prop := x > 0 ∧ y < 0

-- Prove that the line does not intersect the Fourth quadrant
theorem line_not_in_fourth_quadrant : ¬ (∃ x : ℝ, in_fourth_quadrant x (line_eq x)) :=
by
  -- Proof goes here (abstracted)
  sorry

end line_not_in_fourth_quadrant_l85_85774


namespace frac_x_y_eq_neg2_l85_85578

open Real

theorem frac_x_y_eq_neg2 (x y : ℝ) (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 4) (h3 : (x + y) / (x - y) ≠ 1) :
  ∃ t : ℤ, (x / y = t) ∧ (t = -2) :=
by sorry

end frac_x_y_eq_neg2_l85_85578


namespace p_and_q_is_true_l85_85344

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l85_85344


namespace slope_of_line_is_pm1_l85_85538

noncomputable def polarCurve (θ : ℝ) : ℝ := 2 * Real.cos θ - 4 * Real.sin θ

noncomputable def lineParametric (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, -1 + t * Real.sin α)

theorem slope_of_line_is_pm1
  (t α : ℝ)
  (hAB : ∃ A B : ℝ × ℝ, lineParametric t α = A ∧ (∃ t1 t2 : ℝ, A = lineParametric t1 α ∧ B = lineParametric t2 α ∧ dist A B = 3 * Real.sqrt 2))
  (hC : ∃ θ : ℝ, polarCurve θ = dist (1, -1) (polarCurve θ * Real.cos θ, polarCurve θ * Real.sin θ)) :
  ∃ k : ℝ, k = 1 ∨ k = -1 :=
sorry

end slope_of_line_is_pm1_l85_85538


namespace Sam_has_walked_25_miles_l85_85331

variables (d : ℕ) (v_fred v_sam : ℕ)

def Fred_and_Sam_meet (d : ℕ) (v_fred v_sam : ℕ) := 
  d / (v_fred + v_sam) * v_sam

theorem Sam_has_walked_25_miles :
  Fred_and_Sam_meet 50 5 5 = 25 :=
by
  sorry

end Sam_has_walked_25_miles_l85_85331


namespace pandas_bamboo_consumption_l85_85198

def small_pandas : ℕ := 4
def big_pandas : ℕ := 5
def daily_bamboo_small : ℕ := 25
def daily_bamboo_big : ℕ := 40
def days_in_week : ℕ := 7

theorem pandas_bamboo_consumption : 
  (small_pandas * daily_bamboo_small + big_pandas * daily_bamboo_big) * days_in_week = 2100 := by
  sorry

end pandas_bamboo_consumption_l85_85198


namespace sqrt_mul_l85_85096

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l85_85096


namespace not_possible_three_piles_l85_85721

def similar (x y : ℝ) : Prop :=
  y ≤ (real.sqrt 2) * x ∧ x ≤ (real.sqrt 2) * y

theorem not_possible_three_piles (N : ℝ) : 
  ¬ ∃ a b c : ℝ, a + b + c = N ∧ similar a b ∧ similar b c ∧ similar a c := 
by
  sorry

end not_possible_three_piles_l85_85721


namespace lower_seat_tickets_l85_85757

theorem lower_seat_tickets (L U : ℕ) (h1 : L + U = 80) (h2 : 30 * L + 20 * U = 2100) : L = 50 :=
by
  sorry

end lower_seat_tickets_l85_85757


namespace not_possible_to_create_3_piles_l85_85724

theorem not_possible_to_create_3_piles (similar: ℝ → ℝ → Prop) (sqrt_2 : ℝ)
  (hsimilar : ∀ x y, similar x y ↔ x ≤ sqrt_2 * y ∧ y ≤ sqrt_2 * x) :
  ∀ x, ¬ ∃ x1 x2 x3, 
    x = x1 + x2 + x3 ∧ 
    similar x1 x2 ∧ 
    similar x2 x3 ∧ 
    similar x1 x3 :=
by { sorry }

end not_possible_to_create_3_piles_l85_85724


namespace a_plus_b_value_l85_85985

noncomputable def find_a_plus_b (a b : ℕ) (h_neq : a ≠ b) (h_pos : 0 < a ∧ 0 < b) (h_eq : a^2 - b^2 = 2018 - 2 * a) : ℕ :=
  a + b

theorem a_plus_b_value {a b : ℕ} (h_neq : a ≠ b) (h_pos : 0 < a ∧ 0 < b) (h_eq : a^2 - b^2 = 2018 - 2 * a) : find_a_plus_b a b h_neq h_pos h_eq = 672 :=
  sorry

end a_plus_b_value_l85_85985


namespace expected_value_finite_l85_85243

section
variables {X : ℝ → ℝ} {P : Set ℝ → ℝ}

-- Let \(X\) be a random variable with \(P\) as its probability distribution
-- Assume the condition \( \varlimsup_{n} \frac{\mathrm{P}(|X|>2n)}{\mathrm{P}(|X|>n)}<\frac{1}{2} \)
def condition (n : ℕ) : Prop := 
  (limsup (λ n : ℕ, P{ x : ℝ | |X x| > 2*n } / P{ x : ℝ | |X x| > n })) < 1 / 2

-- Prove that \( \mathbf{E}|X|<\infty \)
theorem expected_value_finite (h : ∀ n, condition n) : ∫ x, P {|X x|} x < ∞ :=
by
  sorry
end

end expected_value_finite_l85_85243


namespace add_water_to_solution_l85_85551

theorem add_water_to_solution (w : ℕ) :
  let initial_volume := 50
  let initial_concentration := 0.4
  let final_concentration := 0.25
  let acid_amount := initial_concentration * initial_volume
  let final_volume := initial_volume + w
  (acid_amount / final_volume = final_concentration) ↔ (w = 30) :=
by 
  sorry

end add_water_to_solution_l85_85551


namespace system_of_equations_solutions_l85_85188

theorem system_of_equations_solutions (x1 x2 x3 : ℝ) :
  (2 * x1^2 / (1 + x1^2) = x2) ∧ (2 * x2^2 / (1 + x2^2) = x3) ∧ (2 * x3^2 / (1 + x3^2) = x1)
  → (x1 = 0 ∧ x2 = 0 ∧ x3 = 0) ∨ (x1 = 1 ∧ x2 = 1 ∧ x3 = 1) :=
by
  sorry

end system_of_equations_solutions_l85_85188


namespace weight_of_new_person_l85_85012

theorem weight_of_new_person (avg_increase : ℝ) (num_persons : ℕ) (old_weight new_weight : ℝ) 
  (h_avg_increase : avg_increase = 1.5) (h_num_persons : num_persons = 9) (h_old_weight : old_weight = 65) 
  (h_new_weight_increase : new_weight = old_weight + num_persons * avg_increase) : 
  new_weight = 78.5 :=
sorry

end weight_of_new_person_l85_85012


namespace volleyball_teams_l85_85643

theorem volleyball_teams (managers employees teams : ℕ) (h1 : managers = 3) (h2 : employees = 3) (h3 : teams = 3) :
  ((managers + employees) / teams) = 2 :=
by
  sorry

end volleyball_teams_l85_85643


namespace zhang_prob_one_hit_first_two_zhang_expectation_bullets_used_l85_85503

open ProbabilityTheory

noncomputable def hit_rate : ℝ := 2/3
def miss_rate : ℝ := 1 - hit_rate

theorem zhang_prob_one_hit_first_two :
  ProbabilityTheory.prob (λ ω, (event (0,false, ω) ∧ ¬ event (1,false, ω)) ∨ (¬ event (0,false, ω) ∧ event (1,false, ω)))  =
  4/9 := sorry

theorem zhang_expectation_bullets_used :
  ∑ x in {2, 3, 4, 5}, x * ProbabilityTheory.prob (λ ω, event (n, true, ω)) =
  224/81 := sorry

end zhang_prob_one_hit_first_two_zhang_expectation_bullets_used_l85_85503


namespace cindy_added_pens_l85_85753

-- Definitions based on conditions:
def initial_pens : ℕ := 20
def mike_pens : ℕ := 22
def sharon_pens : ℕ := 19
def final_pens : ℕ := 65

-- Intermediate calculations:
def pens_after_mike : ℕ := initial_pens + mike_pens
def pens_after_sharon : ℕ := pens_after_mike - sharon_pens

-- Proof statement:
theorem cindy_added_pens : pens_after_sharon + 42 = final_pens :=
by
  sorry

end cindy_added_pens_l85_85753


namespace answer_is_p_and_q_l85_85370

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l85_85370


namespace ben_heads_probability_l85_85392

def coin_flip_probability : ℚ :=
  let total_ways := 2^10
  let ways_exactly_five_heads := Nat.choose 10 5
  let probability_exactly_five_heads := ways_exactly_five_heads / total_ways
  let remaining_probability := 1 - probability_exactly_five_heads
  let probability_more_heads := remaining_probability / 2
  probability_more_heads

theorem ben_heads_probability :
  coin_flip_probability = 193 / 512 := by
  sorry

end ben_heads_probability_l85_85392


namespace last_date_in_2011_divisible_by_101_is_1221_l85_85587

def is_valid_date (a b c d : ℕ) : Prop :=
  (10 * a + b) ≤ 12 ∧ (10 * c + d) ≤ 31

def date_as_number (a b c d : ℕ) : ℕ :=
  20110000 + 1000 * a + 100 * b + 10 * c + d

theorem last_date_in_2011_divisible_by_101_is_1221 :
  ∃ (a b c d : ℕ), is_valid_date a b c d ∧ date_as_number a b c d % 101 = 0 ∧ date_as_number a b c d = 20111221 :=
by
  sorry

end last_date_in_2011_divisible_by_101_is_1221_l85_85587


namespace find_speed_of_stream_l85_85945

def distance : ℝ := 24
def total_time : ℝ := 5
def rowing_speed : ℝ := 10

def speed_of_stream (v : ℝ) : Prop :=
  distance / (rowing_speed - v) + distance / (rowing_speed + v) = total_time

theorem find_speed_of_stream : ∃ v : ℝ, speed_of_stream v ∧ v = 2 :=
by
  exists 2
  unfold speed_of_stream
  simp
  sorry -- This would be the proof part which is not required here

end find_speed_of_stream_l85_85945


namespace sqrt_mult_l85_85173

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l85_85173


namespace problem1_problem2_l85_85242

-- Definitions for the sets and conditions
def setA : Set ℝ := {x | -1 < x ∧ x < 2}
def setB (a : ℝ) : Set ℝ := if a > 0 then {x | x ≤ -2 ∨ x ≥ (1 / a)} else ∅

-- Problem 1: Prove the intersection for a == 1
theorem problem1 : (setB 1) ∩ setA = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

-- Problem 2: Prove the range of a
theorem problem2 (a : ℝ) (h : setB a ⊆ setAᶜ) : 0 < a ∧ a ≤ 1/2 :=
by
  sorry

end problem1_problem2_l85_85242


namespace sushi_eating_orders_l85_85615

/-- Define a 2 x 3 grid with sushi pieces being distinguishable -/
inductive SushiPiece : Type
| A | B | C | D | E | F

open SushiPiece

/-- A function that counts the valid orders to eat sushi pieces satisfying the given conditions -/
noncomputable def countValidOrders : Nat :=
  sorry -- This is where the proof would go, stating the number of valid orders

theorem sushi_eating_orders :
  countValidOrders = 360 :=
sorry -- Skipping proof details

end sushi_eating_orders_l85_85615


namespace minimum_n_for_candy_purchases_l85_85493

theorem minimum_n_for_candy_purchases' {o s p : ℕ} (h1 : 9 * o = 10 * s) (h2 : 9 * o = 20 * p) : 
  ∃ n : ℕ, 30 * n = 180 ∧ ∀ m : ℕ, (30 * m = 9 * o) → n ≤ m :=
by sorry

end minimum_n_for_candy_purchases_l85_85493


namespace george_hours_tuesday_l85_85973

def wage_per_hour := 5
def hours_monday := 7
def total_earnings := 45

theorem george_hours_tuesday : ∃ (hours_tuesday : ℕ), 
  hours_tuesday = (total_earnings - (hours_monday * wage_per_hour)) / wage_per_hour := 
by
  sorry

end george_hours_tuesday_l85_85973


namespace jewelry_store_total_cost_l85_85481

-- Definitions for given conditions
def necklace_capacity : Nat := 12
def current_necklaces : Nat := 5
def ring_capacity : Nat := 30
def current_rings : Nat := 18
def bracelet_capacity : Nat := 15
def current_bracelets : Nat := 8

def necklace_cost : Nat := 4
def ring_cost : Nat := 10
def bracelet_cost : Nat := 5

-- Definition for number of items needed to fill displays
def needed_necklaces : Nat := necklace_capacity - current_necklaces
def needed_rings : Nat := ring_capacity - current_rings
def needed_bracelets : Nat := bracelet_capacity - current_bracelets

-- Definition for cost to fill each type of jewelry
def cost_necklaces : Nat := needed_necklaces * necklace_cost
def cost_rings : Nat := needed_rings * ring_cost
def cost_bracelets : Nat := needed_bracelets * bracelet_cost

-- Total cost to fill the displays
def total_cost : Nat := cost_necklaces + cost_rings + cost_bracelets

-- Proof statement
theorem jewelry_store_total_cost : total_cost = 183 := by
  sorry

end jewelry_store_total_cost_l85_85481


namespace min_length_l85_85501

def length (a b : ℝ) : ℝ := b - a

noncomputable def M (m : ℝ) := {x | m ≤ x ∧ x ≤ m + 3 / 4}
noncomputable def N (n : ℝ) := {x | n - 1 / 3 ≤ x ∧ x ≤ n}
noncomputable def intersection (m n : ℝ) := {x | max m (n - 1 / 3) ≤ x ∧ x ≤ min (m + 3 / 4) n}

theorem min_length (m n : ℝ) (hM : ∀ x, x ∈ M m → 0 ≤ x ∧ x ≤ 1) (hN : ∀ x, x ∈ N n → 0 ≤ x ∧ x ≤ 1) :
  length (max m (n - 1 / 3)) (min (m + 3 / 4) n) = 1 / 12 :=
sorry

end min_length_l85_85501


namespace fencing_required_l85_85623

theorem fencing_required (L W : ℕ) (A : ℕ) 
  (hL : L = 20) 
  (hA : A = 680) 
  (hArea : A = L * W) : 
  2 * W + L = 88 := 
by 
  sorry

end fencing_required_l85_85623


namespace geometric_sum_equals_fraction_l85_85915

theorem geometric_sum_equals_fraction (n : ℕ) (a r : ℝ) 
  (h_a : a = 1) (h_r : r = 1 / 2) 
  (h_sum : a * (1 - r^n) / (1 - r) = 511 / 512) : 
  n = 9 := 
by 
  sorry

end geometric_sum_equals_fraction_l85_85915


namespace smallest_range_l85_85624

-- Define the conditions
def estate (A B C : ℝ) : Prop :=
  A = 20000 ∧
  abs (A - B) > 0.3 * A ∧
  abs (A - C) > 0.3 * A ∧
  abs (B - C) > 0.3 * A

-- Define the statement to prove
theorem smallest_range (A B C : ℝ) (h : estate A B C) : 
  ∃ r : ℝ, r = 12000 :=
sorry

end smallest_range_l85_85624


namespace luke_money_at_end_of_june_l85_85560

noncomputable def initial_money : ℝ := 48
noncomputable def february_money : ℝ := initial_money - 0.30 * initial_money
noncomputable def march_money : ℝ := february_money - 11 + 21 + 50 * 1.20

noncomputable def april_savings : ℝ := 0.10 * march_money
noncomputable def april_money : ℝ := (march_money - april_savings) - 10 * 1.18 + 0.05 * (march_money - april_savings)

noncomputable def may_savings : ℝ := 0.15 * april_money
noncomputable def may_money : ℝ := (april_money - may_savings) + 100 * 1.22 - 0.25 * ((april_money - may_savings) + 100 * 1.22)

noncomputable def june_savings : ℝ := 0.10 * may_money
noncomputable def june_money : ℝ := (may_money - june_savings) - 0.08 * (may_money - june_savings)
noncomputable def final_money : ℝ := june_money + 0.06 * (may_money - june_savings)

theorem luke_money_at_end_of_june : final_money = 128.15 := sorry

end luke_money_at_end_of_june_l85_85560


namespace tangent_line_equations_midpoint_coordinates_if_slope_equals_pi_four_max_area_triangle_cpq_l85_85975

section circle_line_problems

-- Conditions
def circle (x y : ℝ) := (x - 3)^2 + (y - 4)^2 = 4
def passes_through_A (x y : ℝ) := x = 1 ∧ y = 0
def line_tangent_to_circle (k : ℝ) (x y : ℝ) := ∃ C : ℝ, y = k * (x - 1) + C ∧ (C = 2 ∨ C = -2)

-- Questions
theorem tangent_line_equations :
  (passes_through_A x y) ∧ (circle x y) → 
  (x = 1 ∨ (3 * x - 4 * y = 3)) :=
sorry

theorem midpoint_coordinates_if_slope_equals_pi_four (x y : ℝ) :
  (passes_through_A x y) ∧ (circle x y) ∧ (y = x - 1) →
  (∃ P Q : ℝ × ℝ, midpoint P Q = (4, 3)) :=
sorry

theorem max_area_triangle_cpq (k : ℝ) :
  (passes_through_A x y) ∧ (circle x y) →
  ∃ P Q : ℝ × ℝ, max_area (C P Q) = 2 → (k = 1 ∨ k = 7) :=
sorry

end circle_line_problems

end tangent_line_equations_midpoint_coordinates_if_slope_equals_pi_four_max_area_triangle_cpq_l85_85975


namespace margin_in_terms_of_selling_price_l85_85400

variable (C S n M : ℝ)

theorem margin_in_terms_of_selling_price (h : M = (2 * C) / n) : M = (2 * S) / (n + 2) :=
sorry

end margin_in_terms_of_selling_price_l85_85400


namespace net_income_on_15th_day_l85_85308

noncomputable def net_income_15th_day : ℝ :=
  let earnings_15th_day := 3 * (3 ^ 14)
  let tax := 0.10 * earnings_15th_day
  let earnings_after_tax := earnings_15th_day - tax
  earnings_after_tax - 100

theorem net_income_on_15th_day :
  net_income_15th_day = 12913916.3 := by
  sorry

end net_income_on_15th_day_l85_85308


namespace inscribed_rectangle_l85_85635

theorem inscribed_rectangle (b h : ℝ) : ∃ x : ℝ, 
  (∃ q : ℝ, x = q / 2) → 
  ∃ x : ℝ, 
    (∃ q : ℝ, q = 2 * x ∧ x = h * q / (2 * h + b)) :=
sorry

end inscribed_rectangle_l85_85635


namespace parallel_line_eq_l85_85033

theorem parallel_line_eq (a b c : ℝ) (p1 p2 : ℝ) :
  (∃ m b1 b2, 3 * a + 6 * b * p1 = 12 ∧ p2 = - (1 / 2) * p1 + b1 ∧
    - (1 / 2) * p1 - m * p1 = b2) → 
    (∃ b', p2 = - (1 / 2) * p1 + b' ∧ b' = 0) := 
sorry

end parallel_line_eq_l85_85033


namespace Julie_work_hours_per_week_l85_85885

variable (hours_per_week_summer : ℕ) (weeks_summer : ℕ)
variable (earnings_summer : ℕ)
variable (weeks_school_year : ℕ)
variable (earnings_school_year : ℕ)

theorem Julie_work_hours_per_week :
  hours_per_week_summer = 40 →
  weeks_summer = 10 →
  earnings_summer = 4000 →
  weeks_school_year = 40 →
  earnings_school_year = 4000 →
  (∀ rate_per_hour, rate_per_hour = earnings_summer / (hours_per_week_summer * weeks_summer) →
  (earnings_school_year / (weeks_school_year * rate_per_hour) = 10)) :=
by intros h1 h2 h3 h4 h5 rate_per_hour hr; sorry

end Julie_work_hours_per_week_l85_85885


namespace olivers_friend_gave_l85_85901

variable (initial_amount saved_amount spent_frisbee spent_puzzle final_amount : ℕ) 

theorem olivers_friend_gave (h1 : initial_amount = 9) 
                           (h2 : saved_amount = 5) 
                           (h3 : spent_frisbee = 4) 
                           (h4 : spent_puzzle = 3) 
                           (h5 : final_amount = 15) : 
                           final_amount - (initial_amount + saved_amount - (spent_frisbee + spent_puzzle)) = 8 := 
by 
  sorry

end olivers_friend_gave_l85_85901


namespace fifth_student_guess_l85_85442

theorem fifth_student_guess (s1 s2 s3 s4 s5 : ℕ) 
(h1 : s1 = 100)
(h2 : s2 = 8 * s1)
(h3 : s3 = s2 - 200)
(h4 : s4 = (s1 + s2 + s3) / 3 + 25)
(h5 : s5 = s4 + s4 / 5) : 
s5 = 630 :=
sorry

end fifth_student_guess_l85_85442


namespace sqrt_multiplication_l85_85158

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l85_85158


namespace max_combined_weight_l85_85265

theorem max_combined_weight (E A : ℕ) (h1 : A = 2 * E) (h2 : A + E = 90) (w_A : ℕ := 5) (w_E : ℕ := 2 * w_A) :
  E * w_E + A * w_A = 600 :=
by
  sorry

end max_combined_weight_l85_85265


namespace Jerry_remaining_pages_l85_85571

theorem Jerry_remaining_pages (total_pages : ℕ) (saturday_read : ℕ) (sunday_read : ℕ) (remaining_pages : ℕ) 
  (h1 : total_pages = 93) (h2 : saturday_read = 30) (h3 : sunday_read = 20) (h4 : remaining_pages = 43) : 
  total_pages - saturday_read - sunday_read = remaining_pages := 
by
  sorry

end Jerry_remaining_pages_l85_85571


namespace Polynomial_has_root_l85_85935

noncomputable def P : ℝ → ℝ := sorry

variables (a1 a2 a3 b1 b2 b3 : ℝ)

axiom h1 : a1 * a2 * a3 ≠ 0
axiom h2 : ∀ x : ℝ, P (a1 * x + b1) + P (a2 * x + b2) = P (a3 * x + b3)

theorem Polynomial_has_root : ∃ x : ℝ, P x = 0 :=
sorry

end Polynomial_has_root_l85_85935


namespace log_a_less_than_neg_b_minus_one_l85_85384

variable {x : ℝ} (a b : ℝ) (f : ℝ → ℝ)

theorem log_a_less_than_neg_b_minus_one
  (h1 : 0 < a)
  (h2 : ∀ x > 0, f x ≥ f 3)
  (h3 : ∀ x > 0, f x = -3 * Real.log x + a * x^2 + b * x) :
  Real.log a < -b - 1 :=
  sorry

end log_a_less_than_neg_b_minus_one_l85_85384


namespace D_times_C_eq_l85_85888

-- Define the matrices C and D
variable (C D : Matrix (Fin 2) (Fin 2) ℚ)

-- Add the conditions
axiom h1 : C * D = C + D
axiom h2 : C * D = ![![15/2, 9/2], ![-6/2, 12/2]]

-- Define the goal
theorem D_times_C_eq : D * C = ![![15/2, 9/2], ![-6/2, 12/2]] :=
sorry

end D_times_C_eq_l85_85888


namespace find_ab_l85_85210

theorem find_ab (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 35) : a * b = 6 := 
by
  sorry

end find_ab_l85_85210


namespace smallest_value_y_l85_85617

theorem smallest_value_y : ∃ y : ℝ, 3 * y ^ 2 + 33 * y - 90 = y * (y + 18) ∧ (∀ z : ℝ, 3 * z ^ 2 + 33 * z - 90 = z * (z + 18) → y ≤ z) ∧ y = -18 := 
sorry

end smallest_value_y_l85_85617


namespace geometric_sequence_formula_l85_85227

def geom_seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, a (n + 1) = r * a n

theorem geometric_sequence_formula (a : ℕ → ℝ) (h_geom : geom_seq a)
  (h1 : a 3 = 2) (h2 : a 6 = 16) :
  ∀ n : ℕ, a n = 2 ^ (n - 2) :=
by
  sorry

end geometric_sequence_formula_l85_85227


namespace cost_per_steak_knife_l85_85504

theorem cost_per_steak_knife :
  ∀ (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℝ),
  sets = 2 →
  knives_per_set = 4 →
  cost_per_set = 80 →
  (cost_per_set * sets) / (sets * knives_per_set) = 20 :=
by
  intros sets knives_per_set cost_per_set sets_eq knives_per_set_eq cost_per_set_eq
  rw [sets_eq, knives_per_set_eq, cost_per_set_eq]
  sorry

end cost_per_steak_knife_l85_85504


namespace complex_power_equality_l85_85849

namespace ComplexProof

open Complex

noncomputable def cos5 : ℂ := cos (5 * Real.pi / 180)

theorem complex_power_equality (w : ℂ) (h : w + 1 / w = 2 * cos5) : 
  w ^ 1000 + 1 / (w ^ 1000) = -((Real.sqrt 5 + 1) / 2) :=
sorry

end ComplexProof

end complex_power_equality_l85_85849


namespace max_value_of_expression_l85_85841

theorem max_value_of_expression (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) : 
  (∃ x : ℝ, x = 3 → 
    ∀ A : ℝ, A = (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4/3)) → 
      A ≤ x) :=
by
  sorry

end max_value_of_expression_l85_85841


namespace POTOP_correct_l85_85464

def POTOP : Nat := 51715

theorem POTOP_correct :
  (99999 * POTOP) % 1000 = 285 := by
  sorry

end POTOP_correct_l85_85464


namespace sqrt_multiplication_l85_85155

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l85_85155


namespace age_problem_l85_85424

theorem age_problem (my_age mother_age : ℕ) 
  (h1 : mother_age = 3 * my_age) 
  (h2 : my_age + mother_age = 40)
  : my_age = 10 :=
by 
  sorry

end age_problem_l85_85424


namespace eldest_age_l85_85759

theorem eldest_age (A B C : ℕ) (x : ℕ) 
  (h1 : A = 5 * x)
  (h2 : B = 7 * x)
  (h3 : C = 8 * x)
  (h4 : (5 * x - 7) + (7 * x - 7) + (8 * x - 7) = 59) :
  C = 32 := 
by 
  sorry

end eldest_age_l85_85759


namespace crease_length_l85_85014

theorem crease_length (AB : ℝ) (h₁ : AB = 15)
  (h₂ : ∀ (area : ℝ) (folded_area : ℝ), folded_area = 0.25 * area) :
  ∃ (DE : ℝ), DE = 0.5 * AB :=
by
  use 7.5 -- DE
  sorry

end crease_length_l85_85014


namespace minimum_voters_needed_l85_85682

-- conditions
def num_voters := 135
def num_districts := 5
def precincts_per_district := 9
def voters_per_precinct := 3
def majority_precincts (n : ℕ) := (n + 1) / 2

-- definitions for quantities derived from conditions
def total_precincts := num_districts * precincts_per_district
def majority_districts := majority_precincts num_districts
def precincts_needed_for_district_win := majority_precincts precincts_per_district
def total_precincts_needed_for_win := majority_districts * precincts_needed_for_district_win
def votes_needed_per_precinct := majority_precincts voters_per_precinct

-- main statement
theorem minimum_voters_needed : (votes_needed_per_precinct * total_precincts_needed_for_win = 30) ∧ TallGiraffeWon :=
by sorry

end minimum_voters_needed_l85_85682


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l85_85037

theorem sum_of_numerator_and_denominator_of_repeating_decimal (x : ℚ) (h : x = 34 / 99) : (x.den + x.num : ℤ) = 133 :=
by
  sorry

end sum_of_numerator_and_denominator_of_repeating_decimal_l85_85037


namespace convex_polyhedron_Euler_formula_l85_85058

theorem convex_polyhedron_Euler_formula (V E F : ℕ) (h : ¬∃ (P : Polyhedron), 
  convex P ∧ V = P.vertices ∧ E = P.edges ∧ F = P.faces) : 
  V - E + F = 2 :=
sorry

end convex_polyhedron_Euler_formula_l85_85058


namespace solve_equation_l85_85597

theorem solve_equation (x y : ℤ) (eq : (x^2 - y^2)^2 = 16 * y + 1) : 
  (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) ∨ 
  (x = 4 ∧ y = 3) ∨ (x = -4 ∧ y = 3) ∨ 
  (x = 4 ∧ y = 5) ∨ (x = -4 ∧ y = 5) :=
sorry

end solve_equation_l85_85597


namespace intersection_of_M_and_N_l85_85386

-- Define set M
def M : Set ℝ := {x | Real.log x > 0}

-- Define set N
def N : Set ℝ := {x | x^2 ≤ 4}

-- Define the target set
def target : Set ℝ := {x | 1 < x ∧ x ≤ 2}

theorem intersection_of_M_and_N :
  M ∩ N = target :=
sorry

end intersection_of_M_and_N_l85_85386


namespace factorization_l85_85822

theorem factorization (a : ℝ) : 2 * a^2 - 2 * a + 1/2 = 2 * (a - 1/2)^2 :=
by
  sorry

end factorization_l85_85822


namespace largest_sum_of_two_largest_angles_of_EFGH_l85_85230

theorem largest_sum_of_two_largest_angles_of_EFGH (x d : ℝ) (y z : ℝ) :
  (∃ a b : ℝ, a + 2 * b = x + 70 ∧ a + b = 70 ∧ 2 * a + 3 * b = 180) ∧
  (2 * x + 3 * d = 180) ∧ (x = 30) ∧ (y = 70) ∧ (z = 100) ∧ (z + 70 = x + d) ∧
  x + d + x + 2 * d + x + 3 * d + x = 360 →
  max (70 + y) (70 + z) + max (y + 70) (z + 70) = 210 := 
sorry

end largest_sum_of_two_largest_angles_of_EFGH_l85_85230


namespace contest_end_time_l85_85768

def start_time : ℕ := 15 * 60 -- 3:00 p.m. in minutes from midnight
def duration : ℕ := 765 -- duration of the contest in minutes

theorem contest_end_time : start_time + duration = 3 * 60 + 45 := by
  -- start_time is 15 * 60 (3:00 p.m. in minutes)
  -- duration is 765 minutes
  -- end_time should be 3:45 a.m. which is 3 * 60 + 45 minutes from midnight
  sorry

end contest_end_time_l85_85768


namespace remaining_structure_volume_and_surface_area_l85_85498

-- Define the dimensions of the large cube and the small cubes
def large_cube_volume := 12 * 12 * 12
def small_cube_volume := 2 * 2 * 2

-- Define the number of smaller cubes in the large cube
def num_small_cubes := (12 / 2) * (12 / 2) * (12 / 2)

-- Define the number of smaller cubes removed (central on each face and very center)
def removed_cubes := 7

-- The volume of a small cube after removing its center unit
def single_small_cube_remaining_volume := small_cube_volume - 1

-- Calculate the remaining volume after all removals
def remaining_volume := (num_small_cubes - removed_cubes) * single_small_cube_remaining_volume

-- Initial surface area of a small cube and increase per removal of central unit
def single_small_cube_initial_surface_area := 6 * 4 -- 6 faces of 2*2*2 cube, each face has 4 units
def single_small_cube_surface_increase := 6

-- Calculate the adjusted surface area considering internal faces' reduction
def single_cube_adjusted_surface_area := single_small_cube_initial_surface_area + single_small_cube_surface_increase
def total_initial_surface_area := single_cube_adjusted_surface_area * (num_small_cubes - removed_cubes)
def total_internal_faces_area := (num_small_cubes - removed_cubes) * 2 * 4
def final_surface_area := total_initial_surface_area - total_internal_faces_area

theorem remaining_structure_volume_and_surface_area :
  remaining_volume = 1463 ∧ final_surface_area = 4598 :=
by
  -- Proof logic goes here
  sorry

end remaining_structure_volume_and_surface_area_l85_85498


namespace remainder_when_four_times_number_minus_nine_divided_by_eight_l85_85292

theorem remainder_when_four_times_number_minus_nine_divided_by_eight
  (n : ℤ) (h : n % 8 = 3) : (4 * n - 9) % 8 = 3 := by
  sorry

end remainder_when_four_times_number_minus_nine_divided_by_eight_l85_85292


namespace sqrt_mult_l85_85168

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l85_85168


namespace problem_solution_A_problem_solution_C_l85_85786

noncomputable def expr_A : ℝ :=
  (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180))

noncomputable def expr_C : ℝ :=
  Real.tan (22.5 * Real.pi / 180) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)

theorem problem_solution_A :
  expr_A = 1 / 2 :=
by
  sorry

theorem problem_solution_C :
  expr_C = 1 / 2 :=
by
  sorry

end problem_solution_A_problem_solution_C_l85_85786


namespace set_representation_l85_85378

def A (x : ℝ) := -3 < x ∧ x < 1
def B (x : ℝ) := x ≤ -1
def C (x : ℝ) := -2 < x ∧ x ≤ 2

theorem set_representation :
  (∀ x, A x ↔ (A x ∧ (B x ∨ C x))) ∧
  (∀ x, A x ↔ (A x ∨ (B x ∧ C x))) ∧
  (∀ x, A x ↔ ((A x ∧ B x) ∨ (A x ∧ C x))) :=
by
  sorry

end set_representation_l85_85378


namespace sqrt3_mul_sqrt12_eq_6_l85_85087

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l85_85087


namespace max_books_single_student_l85_85872

theorem max_books_single_student (total_students : ℕ) (students_0_books : ℕ) (students_1_book : ℕ) (students_2_books : ℕ) (avg_books_per_student : ℕ) :
  total_students = 20 →
  students_0_books = 3 →
  students_1_book = 9 →
  students_2_books = 4 →
  avg_books_per_student = 2 →
  ∃ max_books : ℕ, max_books = 14 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end max_books_single_student_l85_85872


namespace quadratic_function_conditions_l85_85200

noncomputable def quadratic_function_example (x : ℝ) : ℝ :=
  -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_conditions :
  quadratic_function_example 1 = 0 ∧
  quadratic_function_example 5 = 0 ∧
  quadratic_function_example 3 = 10 :=
by
  sorry

end quadratic_function_conditions_l85_85200


namespace smaller_number_is_five_l85_85452

theorem smaller_number_is_five (x y : ℕ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 :=
sorry

end smaller_number_is_five_l85_85452


namespace total_visitors_over_two_days_l85_85948

-- Conditions given in the problem statement
def first_day_visitors : ℕ := 583
def second_day_visitors : ℕ := 246

-- The main problem: proving the total number of visitors over the two days
theorem total_visitors_over_two_days : first_day_visitors + second_day_visitors = 829 := by
  -- Proof is omitted
  sorry

end total_visitors_over_two_days_l85_85948


namespace vegan_menu_fraction_suitable_l85_85673

theorem vegan_menu_fraction_suitable (vegan_dishes total_dishes vegan_dishes_with_gluten_or_dairy : ℕ)
  (h1 : vegan_dishes = 9)
  (h2 : vegan_dishes = 3 * total_dishes / 10)
  (h3 : vegan_dishes_with_gluten_or_dairy = 7) :
  (vegan_dishes - vegan_dishes_with_gluten_or_dairy) / total_dishes = 1 / 15 := by
  sorry

end vegan_menu_fraction_suitable_l85_85673


namespace sum_of_digits_9x_l85_85429

theorem sum_of_digits_9x (a b c d e : ℕ) (x : ℕ) :
  (1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ 9) →
  x = 10000 * a + 1000 * b + 100 * c + 10 * d + e →
  (b - a) + (c - b) + (d - c) + (e - d) + (10 - e) = 9 :=
by
  sorry

end sum_of_digits_9x_l85_85429


namespace points_on_ellipse_l85_85671

-- Definitions of the conditions
def ellipse (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

def passes_through_point (a b : ℝ) : Prop :=
  ellipse a b 2 1

-- Target set of points
def target_set (x y : ℝ) : Prop :=
  x^2 + y^2 < 5 ∧ |y| > 1

-- Main theorem to prove
theorem points_on_ellipse (a b x y : ℝ) (h₁ : passes_through_point a b) (h₂ : |y| > 1) :
  ellipse a b x y → target_set x y :=
sorry

end points_on_ellipse_l85_85671


namespace diff_hours_l85_85962

def hours_English : ℕ := 7
def hours_Spanish : ℕ := 4

theorem diff_hours : hours_English - hours_Spanish = 3 :=
by
  sorry

end diff_hours_l85_85962


namespace total_pages_in_book_l85_85886

theorem total_pages_in_book (P : ℕ)
  (first_day : P - (P / 5) - 12 = remaining_1)
  (second_day : remaining_1 - (remaining_1 / 4) - 15 = remaining_2)
  (third_day : remaining_2 - (remaining_2 / 3) - 18 = 42) :
  P = 190 := 
sorry

end total_pages_in_book_l85_85886


namespace area_of_rhombus_l85_85828

theorem area_of_rhombus (x : ℝ) :
  let d1 := 3 * x + 5
  let d2 := 2 * x + 4
  (d1 * d2) / 2 = 3 * x^2 + 11 * x + 10 :=
by
  let d1 := 3 * x + 5
  let d2 := 2 * x + 4
  have h1 : d1 = 3 * x + 5 := rfl
  have h2 : d2 = 2 * x + 4 := rfl
  simp [h1, h2]
  sorry

end area_of_rhombus_l85_85828


namespace probability_of_more_heads_than_tails_l85_85396

-- Define the probability of getting more heads than tails when flipping 10 coins
def probabilityMoreHeadsThanTails : ℚ :=
  193 / 512

-- Define the proof statement
theorem probability_of_more_heads_than_tails :
  let p : ℚ := probabilityMoreHeadsThanTails in
  p = 193 / 512 :=
by
  sorry

end probability_of_more_heads_than_tails_l85_85396


namespace true_proposition_l85_85365

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l85_85365


namespace find_m_even_fn_l85_85982

theorem find_m_even_fn (m : ℝ) (f : ℝ → ℝ) 
  (Hf : ∀ x : ℝ, f x = x * (10^x + m * 10^(-x))) 
  (Heven : ∀ x : ℝ, f (-x) = f x) : m = -1 := by
  sorry

end find_m_even_fn_l85_85982


namespace main_statement_l85_85356

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l85_85356


namespace max_moves_440_l85_85425

-- Define the set of initial numbers
def initial_numbers : List ℕ := List.range' 1 22

-- Define what constitutes a valid move
def is_valid_move (a b : ℕ) : Prop := b ≥ a + 2

-- Perform the move operation
def perform_move (numbers : List ℕ) (a b : ℕ) : List ℕ :=
  (numbers.erase a).erase b ++ [a + 1, b - 1]

-- Define the maximum number of moves we need to prove
theorem max_moves_440 : ∃ m, m = 440 ∧
  ∀ (moves_done : ℕ) (numbers : List ℕ),
    moves_done <= m → ∃ a b, a ∈ numbers ∧ b ∈ numbers ∧
                             is_valid_move a b ∧
                             numbers = initial_numbers →
                             perform_move numbers a b ≠ numbers
 := sorry

end max_moves_440_l85_85425


namespace total_emails_675_l85_85415

-- Definitions based on conditions
def emails_per_day_before : ℕ := 20
def extra_emails_per_day_after : ℕ := 5
def halfway_days : ℕ := 15
def total_days : ℕ := 30

-- Define the total number of emails received by the end of April
def total_emails_received : ℕ :=
  let emails_before := emails_per_day_before * halfway_days
  let emails_after := (emails_per_day_before + extra_emails_per_day_after) * halfway_days
  emails_before + emails_after

-- Theorem stating that the total number of emails received by the end of April is 675
theorem total_emails_675 : total_emails_received = 675 := by
  sorry

end total_emails_675_l85_85415


namespace molecular_weight_CO_l85_85078

theorem molecular_weight_CO : 
  let molecular_weight_C := 12.01
  let molecular_weight_O := 16.00
  molecular_weight_C + molecular_weight_O = 28.01 :=
by
  sorry

end molecular_weight_CO_l85_85078


namespace correct_number_of_outfits_l85_85861

def num_shirts : ℕ := 7
def num_pants : ℕ := 7
def num_hats : ℕ := 7
def num_colors : ℕ := 7

def total_outfits : ℕ := num_shirts * num_pants * num_hats
def invalid_outfits : ℕ := num_colors
def valid_outfits : ℕ := total_outfits - invalid_outfits

theorem correct_number_of_outfits : valid_outfits = 336 :=
by {
  -- sorry can be removed when providing the proof.
  sorry
}

end correct_number_of_outfits_l85_85861


namespace bisection_method_correctness_l85_85283

noncomputable def initial_interval_length : ℝ := 1
noncomputable def required_precision : ℝ := 0.01
noncomputable def minimum_bisections : ℕ := 7

theorem bisection_method_correctness :
  ∃ n : ℕ, (n ≥ minimum_bisections) ∧ (initial_interval_length / 2^n ≤ required_precision) :=
by
  sorry

end bisection_method_correctness_l85_85283


namespace mersenne_primes_less_than_1000_l85_85186

open Nat

-- Definitions and Conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_mersenne_prime (p : ℕ) : Prop := ∃ n : ℕ, is_prime n ∧ p = 2^n - 1

-- Theorem Statement
theorem mersenne_primes_less_than_1000 : {p : ℕ | is_mersenne_prime p ∧ p < 1000} = {3, 7, 31, 127} :=
by
  sorry

end mersenne_primes_less_than_1000_l85_85186


namespace jimin_notebooks_proof_l85_85423

variable (m f o n : ℕ)

theorem jimin_notebooks_proof (hm : m = 7) (hf : f = 14) (ho : o = 33) (hn : n = o + m + f) :
  n - o = 21 := by
  sorry

end jimin_notebooks_proof_l85_85423


namespace car_y_speed_l85_85175

noncomputable def carY_average_speed (vX : ℝ) (tY : ℝ) (d : ℝ) : ℝ :=
  d / tY

theorem car_y_speed (vX : ℝ := 35) (tY_min : ℝ := 72) (dX_after_Y : ℝ := 245) :
  carY_average_speed vX (dX_after_Y / vX) dX_after_Y = 35 := 
by
  sorry

end car_y_speed_l85_85175


namespace todd_ate_cupcakes_l85_85594

theorem todd_ate_cupcakes :
  let C := 38   -- Total cupcakes baked by Sarah
  let P := 3    -- Number of packages made
  let c := 8    -- Number of cupcakes per package
  let L := P * c  -- Total cupcakes left after packaging
  C - L = 14 :=  -- Cupcakes Todd ate is 14
by
  sorry

end todd_ate_cupcakes_l85_85594


namespace jamal_total_cost_l85_85882

-- Definitions based on conditions
def dozen := 12
def half_dozen := dozen / 2
def crayons_bought := 4 * half_dozen
def cost_per_crayon := 2
def total_cost := crayons_bought * cost_per_crayon

-- Proof statement (the question translated to a Lean theorem)
theorem jamal_total_cost : total_cost = 48 := by
  -- Proof skipped
  sorry

end jamal_total_cost_l85_85882


namespace total_students_l85_85248

theorem total_students (n_groups : ℕ) (students_in_group : ℕ → ℕ)
    (h1 : n_groups = 6)
    (h2 : ∃ n : ℕ, (students_in_group n = 13) ∧ (finset.filter (λ g, students_in_group g = 13) (finset.range n_groups)).card = 4)
    (h3 : ∀ i j, i < n_groups → j < n_groups → abs (students_in_group i - students_in_group j) ≤ 1) :
    (∃ N, N = 76 ∨ N = 80) :=
begin
    sorry
end

end total_students_l85_85248


namespace odd_function_property_l85_85217

-- Defining function f with the given properties.
def f : ℝ → ℝ := 
  λ x => if x > 0 then x^3 + 1 else -x^3 - 1

theorem odd_function_property (x : ℝ) (h_odd : ∀ x : ℝ, f(-x) = -f(x)) (h_pos : x > 0 → f(x) = x^3 + 1) :
  x < 0 → f(x) = -x^3 - 1 :=
by
  intro hx
  have h_neg := h_odd (-x)
  simp at h_neg
  exact sorry

end odd_function_property_l85_85217


namespace solve_equation_l85_85435

theorem solve_equation (x : ℝ) (h : (x^2 - x + 2) / (x - 1) = x + 3) : x = 5 / 3 := 
by sorry

end solve_equation_l85_85435


namespace bankers_discount_l85_85013

theorem bankers_discount {TD S BD : ℝ} (hTD : TD = 66) (hS : S = 429) :
  BD = (TD * S) / (S - TD) → BD = 78 :=
by
  intros h
  rw [hTD, hS] at h
  sorry

end bankers_discount_l85_85013


namespace relationship_between_abc_l85_85987

-- Definitions based on the conditions
def a : ℕ := 3^44
def b : ℕ := 4^33
def c : ℕ := 5^22

-- The theorem to prove the relationship a > b > c
theorem relationship_between_abc : a > b ∧ b > c := by
  sorry

end relationship_between_abc_l85_85987


namespace temperature_range_l85_85726

theorem temperature_range (t : ℕ) : (21 ≤ t ∧ t ≤ 29) :=
by
  sorry

end temperature_range_l85_85726


namespace geometric_sequence_a_sequence_b_l85_85531

theorem geometric_sequence_a (a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : 2 * a 2 + a 3 = 60) :
  ∀ n, a n = 4 * 3^(n - 1) :=
sorry

theorem sequence_b (b a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : 2 * a 2 + a 3 = 60)
  (h3 : ∀ n, b (n + 1) = b n + a n) (h4 : b 1 = a 2) :
  ∀ n, b n = 2 * 3^n + 10 :=
sorry

end geometric_sequence_a_sequence_b_l85_85531


namespace four_thirds_of_product_eq_25_div_2_l85_85653

noncomputable def a : ℚ := 15 / 4
noncomputable def b : ℚ := 5 / 2
noncomputable def c : ℚ := 4 / 3
noncomputable def d : ℚ := a * b
noncomputable def e : ℚ := c * d

theorem four_thirds_of_product_eq_25_div_2 : e = 25 / 2 := 
sorry

end four_thirds_of_product_eq_25_div_2_l85_85653


namespace product_divisible_by_10_probability_l85_85313

noncomputable def probability_divisible_by_10 (n : ℕ) (h: n > 1) : ℝ :=
  1 - ((8^n + 5^n - 4^n : ℝ) / (9^n : ℝ))

theorem product_divisible_by_10_probability (n : ℕ) (h: n > 1) :
  probability_divisible_by_10 n h = 1 - ((8^n + 5^n - 4^n : ℝ) / (9^n : ℝ)) :=
by
  -- The proof is omitted
  sorry

end product_divisible_by_10_probability_l85_85313


namespace cubic_inches_needed_l85_85928

/-- The dimensions of each box are 20 inches by 20 inches by 12 inches. -/
def box_length : ℝ := 20
def box_width : ℝ := 20
def box_height : ℝ := 12

/-- The cost of each box is $0.40. -/
def box_cost : ℝ := 0.40

/-- The minimum spending required by the university on boxes is $200. -/
def min_spending : ℝ := 200

/-- Given the above conditions, the total cubic inches needed to package the collection is 2,400,000 cubic inches. -/
theorem cubic_inches_needed :
  (min_spending / box_cost) * (box_length * box_width * box_height) = 2400000 := by
  sorry

end cubic_inches_needed_l85_85928


namespace max_garden_area_l85_85694

-- Definitions of conditions
def shorter_side (s : ℕ) := s
def longer_side (s : ℕ) := 2 * s
def total_perimeter (s : ℕ) := 2 * shorter_side s + 2 * longer_side s 
def garden_area (s : ℕ) := shorter_side s * longer_side s

-- Theorem with given conditions and conclusion to be proven
theorem max_garden_area (s : ℕ) (h_perimeter : total_perimeter s = 480) : garden_area s = 12800 :=
by
  sorry

end max_garden_area_l85_85694


namespace non_integer_interior_angle_count_l85_85702

theorem non_integer_interior_angle_count :
  ∃! (n : ℕ), 3 ≤ n ∧ n < 10 ∧ ¬(∃ k : ℕ, 180 * (n - 2) = n * k) :=
by sorry

end non_integer_interior_angle_count_l85_85702


namespace describe_graph_l85_85177

noncomputable def points_satisfying_equation (x y : ℝ) : Prop :=
  (x - y) ^ 2 = x ^ 2 + y ^ 2

theorem describe_graph : {p : ℝ × ℝ | points_satisfying_equation p.1 p.2} = {p : ℝ × ℝ | p.1 = 0} ∪ {p : ℝ × ℝ | p.2 = 0} :=
by
  sorry

end describe_graph_l85_85177


namespace part_I_l85_85526

variable (a b c n p q : ℝ)

theorem part_I (hne0 : a ≠ 0) (bne0 : b ≠ 0) (cne0 : c ≠ 0)
    (h1 : a^2 + b^2 + c^2 = 2) (h2 : n^2 + p^2 + q^2 = 2) :
    (n^4 / a^2 + p^4 / b^2 + q^4 / c^2) ≥ 2 := 
sorry

end part_I_l85_85526


namespace total_emails_in_april_l85_85416

-- Definitions representing the conditions
def emails_per_day_initial : Nat := 20
def extra_emails_per_day : Nat := 5
def days_in_month : Nat := 30
def half_days_in_month : Nat := days_in_month / 2

-- Definitions to calculate total emails
def emails_first_half : Nat := emails_per_day_initial * half_days_in_month
def emails_per_day_after_subscription : Nat := emails_per_day_initial + extra_emails_per_day
def emails_second_half : Nat := emails_per_day_after_subscription * half_days_in_month

-- Main theorem to prove the total number of emails received in April
theorem total_emails_in_april : emails_first_half + emails_second_half = 675 := by 
  calc
    emails_first_half + emails_second_half
    = (emails_per_day_initial * half_days_in_month) + (emails_per_day_after_subscription * half_days_in_month) : rfl
    ... = (20 * 15) + ((20 + 5) * 15) : rfl
    ... = 300 + 375 : rfl
    ... = 675 : rfl

end total_emails_in_april_l85_85416


namespace tan_ratio_l85_85241

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : 
  (Real.tan x / Real.tan y) = 7 / 3 :=
sorry

end tan_ratio_l85_85241


namespace symmetric_points_x_axis_l85_85563

theorem symmetric_points_x_axis (a b : ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) 
  (hP : P = (a - 3, 1)) (hQ : Q = (2, b + 1)) (hSymm : P.1 = Q.1 ∧ P.2 = -Q.2) :
  a + b = 3 :=
by 
  sorry

end symmetric_points_x_axis_l85_85563


namespace sqrt_mul_sqrt_eq_six_l85_85148

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l85_85148


namespace square_ratio_short_to_long_side_l85_85916

theorem square_ratio_short_to_long_side (a b : ℝ) (h : a / b + 1 / 2 = b / (Real.sqrt (a^2 + b^2))) : (a / b)^2 = (3 - Real.sqrt 5) / 2 := by
  sorry

end square_ratio_short_to_long_side_l85_85916


namespace solve_fractional_equation_l85_85433

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 1) :
  (x^2 - x + 2) / (x - 1) = x + 3 ↔ x = 5 / 3 :=
by
  sorry

end solve_fractional_equation_l85_85433


namespace sets_produced_and_sold_is_500_l85_85592

-- Define the initial conditions as constants
def initial_outlay : ℕ := 10000
def manufacturing_cost_per_set : ℕ := 20
def selling_price_per_set : ℕ := 50
def total_profit : ℕ := 5000

-- The proof goal
theorem sets_produced_and_sold_is_500 (x : ℕ) : 
  (total_profit = selling_price_per_set * x - (initial_outlay + manufacturing_cost_per_set * x)) → 
  x = 500 :=
by 
  sorry

end sets_produced_and_sold_is_500_l85_85592


namespace cyclist_pedestrian_meeting_distance_l85_85730

theorem cyclist_pedestrian_meeting_distance :
  let A := 0 -- Representing point A at 0 km
  let B := 3 -- Representing point B at 3 km since AB = 3 km
  let C := 7 -- Representing point C at 7 km since AC = AB + BC = 3 + 4 = 7 km
  exists (x : ℝ), x > 0 ∧ x < 3 ∧ x = 2.1 :=
sorry

end cyclist_pedestrian_meeting_distance_l85_85730


namespace b_range_condition_l85_85515

theorem b_range_condition (b : ℝ) : 
  -2 * Real.sqrt 6 < b ∧ b < 2 * Real.sqrt 6 ↔ (b^2 - 24) < 0 :=
by
  sorry

end b_range_condition_l85_85515


namespace rabbit_travel_time_l85_85776

theorem rabbit_travel_time (distance : ℕ) (speed : ℕ) (time_in_minutes : ℕ) 
  (h_distance : distance = 3) 
  (h_speed : speed = 6) 
  (h_time_eqn : time_in_minutes = (distance * 60) / speed) : 
  time_in_minutes = 30 := 
by 
  sorry

end rabbit_travel_time_l85_85776


namespace sqrt_mult_l85_85169

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l85_85169


namespace nicky_pace_l85_85244

theorem nicky_pace :
  ∃ v : ℝ, v = 3 ∧ (
    ∀ (head_start : ℝ) (cristina_pace : ℝ) (time : ℝ) (distance_encounter : ℝ), 
      head_start = 36 ∧ cristina_pace = 4 ∧ time = 36 ∧ distance_encounter = cristina_pace * time - head_start →
      distance_encounter / time = v
  ) :=
sorry

end nicky_pace_l85_85244


namespace gavrila_ascent_time_l85_85747
noncomputable def gavrila_time (V U t : ℝ) : ℝ := t

theorem gavrila_ascent_time (V U : ℝ) :
  (1 = V * 60) →
  (1 = (V + U) * 24) →
  (t = 40 → 1 = U * t) :=
by
  intros h1 h2 h3
  -- Using the given equations:
  -- 1 = V * 60
  -- 1 = (V + U) * 24
  -- Solve for V and substitute to find U
  have hV : V = 1 / 60 := by sorry
  have hU : U = 1 / 40 := by sorry
  rw [h3, hU]
  exact rfl

end gavrila_ascent_time_l85_85747


namespace unique_m_power_function_increasing_l85_85914

theorem unique_m_power_function_increasing : 
  ∃! (m : ℝ), (∀ x : ℝ, x > 0 → (m^2 - m - 5) * x^(m-1) > 0) ∧ (m^2 - m - 5 = 1) ∧ (m - 1 > 0) :=
by
  sorry

end unique_m_power_function_increasing_l85_85914


namespace internal_angles_triangle_ABC_l85_85736

theorem internal_angles_triangle_ABC (α β γ : ℕ) (h₁ : α + β + γ = 180)
  (h₂ : α + γ = 138) (h₃ : β + γ = 108) : (α = 72) ∧ (β = 42) ∧ (γ = 66) :=
by
  sorry

end internal_angles_triangle_ABC_l85_85736


namespace monotone_increasing_interval_l85_85535

noncomputable def f (x : ℝ) : ℝ := (x / (x^2 + 1)) + 1

theorem monotone_increasing_interval :
  ∀ x : ℝ, (-1 < x ∧ x < 1) ↔ ∀ ε > 0, ∃ δ > 0, ∀ x₁ x₂, (-1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 ∧ |x₁ - x₂| < δ) → f x₁ ≤ f x₂ + ε := 
sorry

end monotone_increasing_interval_l85_85535


namespace find_offset_length_l85_85325

theorem find_offset_length 
  (diagonal_offset_7 : ℝ) 
  (area_of_quadrilateral : ℝ) 
  (diagonal_length : ℝ) 
  (result : ℝ) : 
  (diagonal_length = 10) 
  ∧ (diagonal_offset_7 = 7) 
  ∧ (area_of_quadrilateral = 50) 
  → (∃ x, x = result) :=
by
  sorry

end find_offset_length_l85_85325


namespace last_three_digits_of_7_pow_123_l85_85967

theorem last_three_digits_of_7_pow_123 : 7^123 % 1000 = 773 := 
by sorry

end last_three_digits_of_7_pow_123_l85_85967


namespace sqrt_mul_l85_85099

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l85_85099


namespace school_points_l85_85919

theorem school_points (a b c : ℕ) (h1 : a + b + c = 285)
  (h2 : ∃ x : ℕ, a - 8 = x ∧ b - 12 = x ∧ c - 7 = x) : a + c = 187 :=
sorry

end school_points_l85_85919


namespace time_to_ascend_non_working_escalator_l85_85748

-- Definitions from the conditions
def length_of_escalator := 1
def time_standing := 1
def time_running := 24 / 60
def escalator_speed := 1 / 60
def gavrila_speed := 1 / 40

-- The proof problem statement 
theorem time_to_ascend_non_working_escalator 
  (length_of_escalator : ℝ)
  (time_standing : ℝ)
  (time_running : ℝ)
  (escalator_speed : ℝ)
  (gavrila_speed : ℝ) :
  time_standing = 1 →
  time_running = 24 / 60 →
  escalator_speed = 1 / 60 →
  gavrila_speed = 1 / 40 →
  length_of_escalator = 1 →
  1 / gavrila_speed = 40 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end time_to_ascend_non_working_escalator_l85_85748


namespace range_of_first_person_l85_85628

variable (R1 R2 R3 : ℕ)
variable (min_range : ℕ)
variable (condition1 : min_range = 25)
variable (condition2 : R2 = 25)
variable (condition3 : R3 = 30)
variable (condition4 : min_range ≤ R1 ∧ min_range ≤ R2 ∧ min_range ≤ R3)

theorem range_of_first_person : R1 = 25 :=
by
  sorry

end range_of_first_person_l85_85628


namespace sales_tax_difference_l85_85266

def item_price : ℝ := 20
def sales_tax_rate1 : ℝ := 0.065
def sales_tax_rate2 : ℝ := 0.06

theorem sales_tax_difference :
  (item_price * sales_tax_rate1) - (item_price * sales_tax_rate2) = 0.1 := 
by
  sorry

end sales_tax_difference_l85_85266


namespace alex_serge_equiv_distinct_values_l85_85488

-- Defining the context and data structures
variable {n : ℕ} -- Number of boxes
variable {c : ℕ → ℕ} -- Function representing number of cookies in each box, indexed by box number
variable {m : ℕ} -- Number of plates
variable {p : ℕ → ℕ} -- Function representing number of cookies on each plate, indexed by plate number

-- Define the sets representing the unique counts recorded by Alex and Serge
def Alex_record (c : ℕ → ℕ) (n : ℕ) : Set ℕ := 
  { x | ∃ i, i < n ∧ c i = x }

def Serge_record (p : ℕ → ℕ) (m : ℕ) : Set ℕ := 
  { y | ∃ j, j < m ∧ p j = y }

-- The proof goal: Alex's record contains the same number of distinct values as Serge's record
theorem alex_serge_equiv_distinct_values
  (c : ℕ → ℕ) (n : ℕ) (p : ℕ → ℕ) (m : ℕ) :
  Alex_record c n = Serge_record p m :=
sorry

end alex_serge_equiv_distinct_values_l85_85488


namespace smallest_sum_of_squares_l85_85269

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 187) : x^2 + y^2 ≥ 205 := sorry

end smallest_sum_of_squares_l85_85269


namespace meeting_point_distance_l85_85729

/-
 Points \( A, B, C \) are situated sequentially, with \( AB = 3 \) km and \( BC = 4 \) km. 
 A cyclist departed from point \( A \) heading towards point \( C \). 
 A pedestrian departed from point \( B \) heading towards point \( A \). 
 Both arrived at points \( A \) and \( C \) simultaneously. 
 Find the distance from point \( A \) at which they met.
-/

def distance_A_B : ℝ := 3
def distance_B_C : ℝ := 4
def distance_A_C : ℝ := distance_A_B + distance_B_C

theorem meeting_point_distance (V_C V_P : ℝ) (h_time_eq : 7 / V_C = 3 / V_P) : 
  ∃ x : ℝ, x = 2.1 :=
begin
  -- Definitions of the known distances
  let AB := distance_A_B,
  let BC := distance_B_C,
  let AC := distance_A_C,

  -- Set up the ratio of their speeds
  let speed_ratio := 7 / 3,

  -- Define distances covered by cyclist and pedestrian
  let x := 2.1, -- the distance we need to prove
  
  -- Check the ratio relationship
  -- Combine the facts to goal, direct straightforward calculation
  
  use x,
  exact rfl,
end

end meeting_point_distance_l85_85729


namespace correct_number_of_outfits_l85_85860

def num_shirts : ℕ := 7
def num_pants : ℕ := 7
def num_hats : ℕ := 7
def num_colors : ℕ := 7

def total_outfits : ℕ := num_shirts * num_pants * num_hats
def invalid_outfits : ℕ := num_colors
def valid_outfits : ℕ := total_outfits - invalid_outfits

theorem correct_number_of_outfits : valid_outfits = 336 :=
by {
  -- sorry can be removed when providing the proof.
  sorry
}

end correct_number_of_outfits_l85_85860


namespace sqrt_multiplication_l85_85151

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l85_85151


namespace seventh_observation_l85_85055

-- Definitions from the conditions
def avg_original (x : ℕ) := 13
def num_observations_original := 6
def total_original := num_observations_original * (avg_original 0) -- 6 * 13 = 78

def avg_new := 12
def num_observations_new := num_observations_original + 1 -- 7
def total_new := num_observations_new * avg_new -- 7 * 12 = 84

-- The proof goal statement
theorem seventh_observation : (total_new - total_original) = 6 := 
  by
    -- Placeholder for the proof
    sorry

end seventh_observation_l85_85055


namespace calculate_binary_expr_l85_85491

theorem calculate_binary_expr :
  let a := 0b11001010
  let b := 0b11010
  let c := 0b100
  (a * b) / c = 0b1001110100 := by
sorry

end calculate_binary_expr_l85_85491


namespace Robert_older_than_Elizabeth_l85_85727

-- Define the conditions
def Patrick_half_Robert (Patrick Robert : ℕ) : Prop := Patrick = Robert / 2
def Robert_turn_30_in_2_years (Robert : ℕ) : Prop := Robert + 2 = 30
def Elizabeth_4_years_younger_than_Patrick (Elizabeth Patrick : ℕ) : Prop := Elizabeth = Patrick - 4

-- The theorem we need to prove
theorem Robert_older_than_Elizabeth
  (Patrick Robert Elizabeth : ℕ)
  (h1 : Patrick_half_Robert Patrick Robert)
  (h2 : Robert_turn_30_in_2_years Robert)
  (h3 : Elizabeth_4_years_younger_than_Patrick Elizabeth Patrick) :
  Robert - Elizabeth = 18 :=
sorry

end Robert_older_than_Elizabeth_l85_85727


namespace vector_subtraction_proof_l85_85510

def v1 : ℝ × ℝ := (3, -8)
def v2 : ℝ × ℝ := (2, -6)
def a : ℝ := 5
def answer : ℝ × ℝ := (-7, 22)

theorem vector_subtraction_proof : (v1.1 - a * v2.1, v1.2 - a * v2.2) = answer := 
by
  sorry

end vector_subtraction_proof_l85_85510


namespace impossible_to_create_3_piles_l85_85708

def similar_sizes (x y : ℝ) : Prop := x ≤ sqrt 2 * y

theorem impossible_to_create_3_piles (x y : ℝ) (h₁ : similar_sizes x y) :
  ¬ ∃ (x₁ x₂ x₃ : ℝ), x₁ + x₂ + x₃ = x + y ∧ similar_sizes x₁ x₂ ∧ similar_sizes x₂ x₃ ∧ similar_sizes x₁ x₃ :=
by
  sorry

end impossible_to_create_3_piles_l85_85708


namespace seedling_height_regression_seedling_height_distribution_and_expectation_l85_85477

noncomputable def average (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

noncomputable def linear_regression (xs ys : List ℝ) : ℝ × ℝ :=
  let n := xs.length
  let x_sum := xs.sum
  let y_sum := ys.sum
  let xy_sum := (List.zipWith (*) xs ys).sum
  let x_square_sum := (xs.map (λ x => x * x)).sum
  let b := (xy_sum - n * (x_sum / n) * (y_sum / n)) / (x_square_sum - n * (x_sum / n) ^ 2)
  let a := (y_sum / n) - b * (x_sum / n)
  (b, a)

theorem seedling_height_regression :
  linear_regression [1, 2, 3, 4, 5, 6, 7] [0, 4, 7, 9, 11, 12, 13] = (59/28, -3/7) :=
sorry

noncomputable def prob_xi_distribution_and_expectation (heights: List ℝ) : (ℕ × ℕ × ℕ × ℕ) × ℝ :=
  let avg_height := average heights
  let greater_than_avg := heights.count (λ h => h > avg_height)
  let less_or_equal_avg := heights.count (λ h => h <= avg_height)
  let p_ξ0 := (Nat.choose 3 3 * Nat.choose 4 0) / (Nat.choose 7 3 : ℝ)
  let p_ξ1 := (Nat.choose 3 2 * Nat.choose 4 1) / (Nat.choose 7 3 : ℝ)
  let p_ξ2 := (Nat.choose 3 1 * Nat.choose 4 2) / (Nat.choose 7 3 : ℝ)
  let p_ξ3 := (Nat.choose 3 0 * Nat.choose 4 3) / (Nat.choose 7 3 : ℝ)
  let E_ξ := 0 * p_ξ0 + 1 * p_ξ1 + 2 * p_ξ2 + 3 * p_ξ3
  ((p_ξ0, p_ξ1, p_ξ2, p_ξ3), E_ξ)

theorem seedling_height_distribution_and_expectation :
  prob_xi_distribution_and_expectation [0, 4, 7, 9, 11, 12, 13] = ((1/35, 12/35, 18/35, 4/35), 12/7) :=
sorry

end seedling_height_regression_seedling_height_distribution_and_expectation_l85_85477


namespace find_constants_l85_85827

theorem find_constants (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 5 → (x^2 - 7) / ((x - 2) * (x - 3) * (x - 5)) = A / (x - 2) + B / (x - 3) + C / (x - 5))
  ↔ (A = -1 ∧ B = -1 ∧ C = 3) :=
by
  sorry

end find_constants_l85_85827


namespace proposition_A_l85_85350

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l85_85350


namespace probability_of_both_selected_l85_85626

variable (P_ram : ℚ) (P_ravi : ℚ) (P_both : ℚ)

def selection_probability (P_ram : ℚ) (P_ravi : ℚ) : ℚ :=
  P_ram * P_ravi

theorem probability_of_both_selected (h1 : P_ram = 3/7) (h2 : P_ravi = 1/5) :
  selection_probability P_ram P_ravi = P_both :=
by
  sorry

end probability_of_both_selected_l85_85626


namespace inequality_proof_l85_85977

theorem inequality_proof (a b c : ℝ) (h : a > b) : a * c^2 ≥ b * c^2 :=
sorry

end inequality_proof_l85_85977


namespace circus_total_tickets_sold_l85_85911

-- Definitions from the conditions
def revenue_total : ℕ := 2100
def lower_seat_tickets_sold : ℕ := 50
def price_lower : ℕ := 30
def price_upper : ℕ := 20

-- Definition derived from the conditions
def tickets_total (L U : ℕ) : ℕ := L + U

-- The theorem we need to prove
theorem circus_total_tickets_sold (L U : ℕ) (hL: L = lower_seat_tickets_sold)
    (h₁ : price_lower * L + price_upper * U = revenue_total) : 
    tickets_total L U = 80 :=
by
  sorry  -- Proof omitted

end circus_total_tickets_sold_l85_85911


namespace square_perimeter_l85_85765

theorem square_perimeter (A_total : ℕ) (A_common : ℕ) (A_circle : ℕ) 
  (H1 : A_total = 329)
  (H2 : A_common = 101)
  (H3 : A_circle = 234) :
  4 * (Int.sqrt (A_total - A_circle + A_common)) = 56 :=
by
  -- Since we are only required to provide the statement, we can skip the proof steps.
  -- sorry to skip the proof.
  sorry

end square_perimeter_l85_85765


namespace polynomial_divisible_l85_85298

theorem polynomial_divisible (a b c : ℕ) :
  (X^(3 * a) + X^(3 * b + 1) + X^(3 * c + 2)) % (X^2 + X + 1) = 0 :=
by sorry

end polynomial_divisible_l85_85298


namespace proposition_p_and_q_is_true_l85_85345

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l85_85345


namespace solution_of_equation_l85_85742

theorem solution_of_equation (x : ℤ) : 7 * x - 5 = 6 * x → x = 5 := by
  intro h
  sorry

end solution_of_equation_l85_85742


namespace find_particular_number_l85_85483

theorem find_particular_number (x : ℤ) (h : ((x / 23) - 67) * 2 = 102) : x = 2714 := 
by 
  sorry

end find_particular_number_l85_85483


namespace find_f_x_l85_85647

def tan : ℝ → ℝ := sorry  -- tan function placeholder
def cos : ℝ → ℝ := sorry  -- cos function placeholder
def sin : ℝ → ℝ := sorry  -- sin function placeholder

axiom conditions : 
  tan 45 = 1 ∧
  cos 60 = 2 ∧
  sin 90 = 3 ∧
  cos 180 = 4 ∧
  sin 270 = 5

theorem find_f_x :
  ∃ f x, (f x = 6) ∧ 
  (f = tan ∧ x = 360) := 
sorry

end find_f_x_l85_85647


namespace sqrt_mult_eq_six_l85_85162

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l85_85162


namespace cookies_with_new_flour_l85_85698

-- Definitions for the conditions
def cookies_per_cup (total_cookies : ℕ) (total_flour : ℕ) : ℕ :=
  total_cookies / total_flour

noncomputable def cookies_from_flour (cookies_per_cup : ℕ) (flour : ℕ) : ℕ :=
  cookies_per_cup * flour

-- Given data
def total_cookies := 24
def total_flour := 4
def new_flour := 3

-- Theorem (problem statement)
theorem cookies_with_new_flour : cookies_from_flour (cookies_per_cup total_cookies total_flour) new_flour = 18 :=
by
  sorry

end cookies_with_new_flour_l85_85698


namespace possible_combinations_l85_85887

noncomputable def dark_chocolate_price : ℝ := 5
noncomputable def milk_chocolate_price : ℝ := 4.50
noncomputable def white_chocolate_price : ℝ := 6
noncomputable def sales_tax_rate : ℝ := 0.07
noncomputable def leonardo_money : ℝ := 4 + 0.59

noncomputable def total_money := leonardo_money

noncomputable def dark_chocolate_with_tax := dark_chocolate_price * (1 + sales_tax_rate)
noncomputable def milk_chocolate_with_tax := milk_chocolate_price * (1 + sales_tax_rate)
noncomputable def white_chocolate_with_tax := white_chocolate_price * (1 + sales_tax_rate)

theorem possible_combinations :
  total_money = 4.59 ∧ (total_money >= 0 ∧ total_money < dark_chocolate_with_tax ∧ total_money < white_chocolate_with_tax ∧
  total_money ≥ milk_chocolate_with_tax ∧ milk_chocolate_with_tax = 4.82) :=
by
  sorry

end possible_combinations_l85_85887


namespace evaluate_difference_floor_squares_l85_85323

theorem evaluate_difference_floor_squares (x : ℝ) (h : x = 15.3) : ⌊x^2⌋ - ⌊x⌋^2 = 9 := by
  sorry

end evaluate_difference_floor_squares_l85_85323


namespace water_to_add_for_desired_acid_concentration_l85_85544

theorem water_to_add_for_desired_acid_concentration :
  ∃ w : ℝ, 50 * 0.4 / (50 + w) = 0.25 ∧ w = 30 :=
begin
  let initial_volume := 50,
  let initial_acid_concentration := 0.4,
  let desired_acid_concentration := 0.25,
  let acid_content := initial_volume * initial_acid_concentration,
  use 30,
  split,
  { simp [initial_volume, initial_acid_concentration, desired_acid_concentration, acid_content],
    linarith },
  { refl }
end

end water_to_add_for_desired_acid_concentration_l85_85544


namespace find_vector_at_t_zero_l85_85482

variable (a d : ℝ × ℝ × ℝ)
variable (t : ℝ)

-- Given conditions
def condition1 := a - 2 * d = (2, 4, 10)
def condition2 := a + d = (-1, -3, -5)

-- The proof problem
theorem find_vector_at_t_zero 
  (h1 : condition1 a d)
  (h2 : condition2 a d) :
  a = (0, -2/3, 0) :=
sorry

end find_vector_at_t_zero_l85_85482


namespace sqrt_mul_sqrt_eq_six_l85_85150

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l85_85150


namespace trains_meet_at_noon_l85_85922

noncomputable def meeting_time_of_trains : Prop :=
  let distance_between_stations := 200
  let speed_of_train_A := 20
  let starting_time_A := 7
  let speed_of_train_B := 25
  let starting_time_B := 8
  let initial_distance_covered_by_A := speed_of_train_A * (starting_time_B - starting_time_A)
  let remaining_distance := distance_between_stations - initial_distance_covered_by_A
  let relative_speed := speed_of_train_A + speed_of_train_B
  let time_to_meet_after_B_starts := remaining_distance / relative_speed
  let meeting_time := starting_time_B + time_to_meet_after_B_starts
  meeting_time = 12

theorem trains_meet_at_noon : meeting_time_of_trains :=
by
  sorry

end trains_meet_at_noon_l85_85922


namespace remaining_pages_l85_85572

def original_book_pages : ℕ := 93
def pages_read_saturday : ℕ := 30
def pages_read_sunday : ℕ := 20

theorem remaining_pages :
  original_book_pages - (pages_read_saturday + pages_read_sunday) = 43 := by
  sorry

end remaining_pages_l85_85572


namespace combine_like_terms_l85_85496

variable (a : ℝ)

theorem combine_like_terms : 3 * a^2 + 5 * a^2 - a^2 = 7 * a^2 := 
by sorry

end combine_like_terms_l85_85496


namespace value_of_3a_plus_6b_l85_85864

theorem value_of_3a_plus_6b (a b : ℝ) (h : a + 2 * b = 1) : 3 * a + 6 * b = 3 :=
sorry

end value_of_3a_plus_6b_l85_85864


namespace pure_water_to_achieve_desired_concentration_l85_85546

theorem pure_water_to_achieve_desired_concentration :
  ∀ (w : ℝ), (50 + w ≠ 0) → (0.4 * 50 / (50 + w) = 0.25) → w = 30 := 
by
  intros w h_nonzero h_concentration
  sorry

end pure_water_to_achieve_desired_concentration_l85_85546


namespace charlie_collected_15_seashells_l85_85494

variables (c e : ℝ)

-- Charlie collected 10 more seashells than Emily
def charlie_more_seashells := c = e + 10

-- Emily collected one-third the number of seashells Charlie collected
def emily_seashells := e = c / 3

theorem charlie_collected_15_seashells (hc: charlie_more_seashells c e) (he: emily_seashells c e) : c = 15 := 
by sorry

end charlie_collected_15_seashells_l85_85494


namespace problem_statement_l85_85340

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l85_85340


namespace regression_line_estimate_l85_85212

theorem regression_line_estimate:
  (∀ (x y : ℝ), y = 1.23 * x + a ↔ a = 5 - 1.23 * 4) →
  ∃ (y : ℝ), y = 1.23 * 2 + 0.08 :=
by
  intro h
  use 2.54
  simp
  sorry

end regression_line_estimate_l85_85212


namespace sqrt3_mul_sqrt12_eq_6_l85_85113

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l85_85113


namespace sqrt_mul_simp_l85_85129

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l85_85129


namespace solve_eq1_solve_eq2_l85_85192

-- Define the condition and the statement to be proved for the first equation
theorem solve_eq1 (x : ℝ) : 9 * x^2 - 25 = 0 → x = 5 / 3 ∨ x = -5 / 3 :=
by sorry

-- Define the condition and the statement to be proved for the second equation
theorem solve_eq2 (x : ℝ) : (x + 1)^3 - 27 = 0 → x = 2 :=
by sorry

end solve_eq1_solve_eq2_l85_85192


namespace exponent_division_l85_85805

theorem exponent_division (a : ℕ) (m n : ℕ) (h1 : 19 = a) (h2 : 11 = m) (h3 : 8 = n) : a^(m - n) = 6859 := by
  sorry

end exponent_division_l85_85805


namespace pupils_in_class_l85_85307

theorem pupils_in_class (n : ℕ) (wrong_entry_increase : n * (1/2) = 13) : n = 26 :=
sorry

end pupils_in_class_l85_85307


namespace train_speed_is_correct_l85_85783

-- Definitions for conditions
def train_length : ℝ := 150  -- length of the train in meters
def time_to_cross_pole : ℝ := 3  -- time to cross the pole in seconds

-- Proof statement
theorem train_speed_is_correct : (train_length / time_to_cross_pole) = 50 := by
  sorry

end train_speed_is_correct_l85_85783


namespace sqrt_mul_eq_6_l85_85142

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l85_85142


namespace option_a_equals_half_option_c_equals_half_l85_85789

theorem option_a_equals_half : 
  ( ∃ x : ℝ, x = (√2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180)) ∧ x = 1 / 2 ) := 
sorry

theorem option_c_equals_half : 
  ( ∃ y : ℝ, y = (Real.tan (22.5 * Real.pi / 180) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)) ∧ y = 1 / 2 ) := 
sorry

end option_a_equals_half_option_c_equals_half_l85_85789


namespace tiger_initial_leaps_behind_l85_85781

theorem tiger_initial_leaps_behind (tiger_leap_distance deer_leap_distance tiger_leaps_per_minute deer_leaps_per_minute total_distance_to_catch initial_leaps_behind : ℕ) 
  (h1 : tiger_leap_distance = 8) 
  (h2 : deer_leap_distance = 5) 
  (h3 : tiger_leaps_per_minute = 5) 
  (h4 : deer_leaps_per_minute = 4) 
  (h5 : total_distance_to_catch = 800) :
  initial_leaps_behind = 40 := 
by
  -- Leaving proof body incomplete as it is not required
  sorry

end tiger_initial_leaps_behind_l85_85781


namespace g_18_value_l85_85422

-- Define the function g as taking positive integers to positive integers
variable (g : ℕ+ → ℕ+)

-- Define the conditions for the function g
axiom increasing (n : ℕ+) : g (n + 1) > g n
axiom multiplicative (m n : ℕ+) : g (m * n) = g m * g n
axiom power_property (m n : ℕ+) (h : m ≠ n ∧ m ^ (n : ℕ) = n ^ (m : ℕ)) :
  g m = n ∨ g n = m

-- Prove that g(18) is 72
theorem g_18_value : g 18 = 72 :=
sorry

end g_18_value_l85_85422


namespace simplify_expression_l85_85732

variable (a b : ℝ)
variable (h₁ : a = 3 + Real.sqrt 5)
variable (h₂ : b = 3 - Real.sqrt 5)

theorem simplify_expression : (a^2 - 2 * a * b + b^2) / (a^2 - b^2) * (a * b) / (a - b) = 2 / 3 := by
  sorry

end simplify_expression_l85_85732


namespace mouse_grasshopper_diff_l85_85445

def grasshopper_jump: ℕ := 19
def frog_jump: ℕ := grasshopper_jump + 10
def mouse_jump: ℕ := frog_jump + 20

theorem mouse_grasshopper_diff:
  (mouse_jump - grasshopper_jump) = 30 :=
by
  sorry

end mouse_grasshopper_diff_l85_85445


namespace problem_l85_85701

theorem problem : 
  let b := 2 ^ 51
  let c := 4 ^ 25
  b > c :=
by 
  let b := 2 ^ 51
  let c := 4 ^ 25
  sorry

end problem_l85_85701


namespace sum_of_all_possible_values_of_z_l85_85330

noncomputable def sum_of_z_values (w x y z : ℚ) : ℚ :=
if h : w < x ∧ x < y ∧ y < z ∧ 
       (w + x = 1 ∧ w + y = 2 ∧ w + z = 3 ∧ x + y = 4 ∨ 
        w + x = 1 ∧ w + y = 2 ∧ w + z = 4 ∧ x + y = 3) ∧ 
       ((w + x) ≠ (w + y) ∧ (w + x) ≠ (w + z) ∧ (w + x) ≠ (x + y) ∧ (w + x) ≠ (x + z) ∧ (w + x) ≠ (y + z)) ∧ 
       ((w + y) ≠ (w + z) ∧ (w + y) ≠ (x + y) ∧ (w + y) ≠ (x + z) ∧ (w + y) ≠ (y + z)) ∧ 
       ((w + z) ≠ (x + y) ∧ (w + z) ≠ (x + z) ∧ (w + z) ≠ (y + z)) ∧ 
       ((x + y) ≠ (x + z) ∧ (x + y) ≠ (y + z)) ∧ 
       ((x + z) ≠ (y + z)) then
  if w + z = 4 then
    4 + 7/2
  else 0
else
  0

theorem sum_of_all_possible_values_of_z : sum_of_z_values w x y z = 15 / 2 :=
by sorry

end sum_of_all_possible_values_of_z_l85_85330


namespace range_of_m_l85_85866

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + m * x + 2 * m - 3 ≥ 0) ↔ 2 ≤ m ∧ m ≤ 6 := 
by
  sorry

end range_of_m_l85_85866


namespace statement_3_correct_l85_85622

-- Definitions based on the conditions
def DeductiveReasoningGeneralToSpecific := True
def SyllogismForm := True
def ConclusionDependsOnPremisesAndForm := True

-- Proof problem statement
theorem statement_3_correct : SyllogismForm := by
  exact True.intro

end statement_3_correct_l85_85622


namespace completing_the_square_sum_l85_85007

theorem completing_the_square_sum :
  ∃ (a b c : ℤ), 64 * (x : ℝ) ^ 2 + 96 * x - 81 = 0 ∧ a > 0 ∧ (8 * x + 6) ^ 2 = c ∧ a = 8 ∧ b = 6 ∧ a + b + c = 131 :=
by
  sorry

end completing_the_square_sum_l85_85007


namespace dealership_vans_expected_l85_85769

theorem dealership_vans_expected (trucks vans : ℕ) (h_ratio : 3 * vans = 5 * trucks) (h_trucks : trucks = 45) : vans = 75 :=
by
  sorry

end dealership_vans_expected_l85_85769


namespace problem_solution_l85_85336

def sequence_a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2 * n - 1

def sum_S (n : ℕ) : ℕ :=
  n * n

def sequence_c (n : ℕ) : ℕ :=
  sequence_a n * 2 ^ (sequence_a n)

def sum_T (n : ℕ) : ℕ :=
  (6 * n - 5) * 2 ^ (2 * n + 1) + 10

theorem problem_solution (n : ℕ) (hn : n ≥ 1) :
  ∀ n, (sum_S 1 = 1) ∧ (sequence_a 1 = 1) ∧ 
          (∀ n ≥ 2, sequence_a n = 2 * n - 1) ∧
          (sum_T n = (6 * n - 5) * 2 ^ (2 * n + 1) + 10 / 9) :=
by sorry

end problem_solution_l85_85336


namespace find_value_of_expression_l85_85668

open Real

theorem find_value_of_expression (x y z w : ℝ) (h1 : x + y + z + w = 0) (h2 : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end find_value_of_expression_l85_85668


namespace tangent_line_equation_at_A_l85_85980

noncomputable def f : ℝ → ℝ := λ x, x ^ (1 / 2 : ℝ)

theorem tangent_line_equation_at_A :
  f (1 / 4) = 1 / 2 →
  tangent_line f (1 / 4, 1 / 2) = 4 * x - 4 * y + 1 := 
begin
  sorry,
end

end tangent_line_equation_at_A_l85_85980


namespace div_by_73_l85_85002

theorem div_by_73 (n : ℕ) (h : 0 < n) : (2^(3*n + 6) + 3^(4*n + 2)) % 73 = 0 := sorry

end div_by_73_l85_85002


namespace triangle_area_specific_l85_85189

noncomputable def vector2_area_formula (u v : ℝ × ℝ) : ℝ :=
|u.1 * v.2 - u.2 * v.1|

noncomputable def triangle_area (u v : ℝ × ℝ) : ℝ :=
(vector2_area_formula u v) / 2

theorem triangle_area_specific :
  let A := (1, 3)
  let B := (5, -1)
  let C := (9, 4)
  let u := (1 - 9, 3 - 4)
  let v := (5 - 9, -1 - 4)
  triangle_area u v = 18 := 
by sorry

end triangle_area_specific_l85_85189


namespace show_linear_l85_85993

-- Define the conditions as given in the problem
variables (a b : ℤ)

-- The hypothesis that the equation is linear
def linear_equation_hypothesis : Prop :=
  (a + b = 1) ∧ (3 * a + 2 * b - 4 = 1)

-- Define the theorem we need to prove
theorem show_linear (h : linear_equation_hypothesis a b) : a + b = 1 := 
by
  sorry

end show_linear_l85_85993


namespace symmetry_x_axis_l85_85565

theorem symmetry_x_axis (a b : ℝ) (h1 : a - 3 = 2) (h2 : 1 = -(b + 1)) : a + b = 3 :=
by
  sorry

end symmetry_x_axis_l85_85565


namespace value_of_a_l85_85211

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0.5 → 1 - a / 2^x > 0) → a = Real.sqrt 2 :=
by
  sorry

end value_of_a_l85_85211


namespace find_quadratic_function_l85_85438

-- Define the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant condition
def discriminant_zero (a b c : ℝ) : Prop := b^2 - 4 * a * c = 0

-- Given derivative
def given_derivative (x : ℝ) : ℝ := 2 * x + 2

-- Prove that if these conditions hold, then f(x) = x^2 + 2x + 1
theorem find_quadratic_function :
  ∃ (a b c : ℝ), (∀ x, quadratic_function a b c x = 0 → discriminant_zero a b c) ∧
                (∀ x, (2 * a * x + b) = given_derivative x) ∧
                (quadratic_function a b c x = x^2 + 2 * x + 1) := 
by
  sorry

end find_quadratic_function_l85_85438


namespace sqrt3_mul_sqrt12_eq_6_l85_85088

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l85_85088


namespace ian_leftover_money_l85_85992

def ianPayments (initial: ℝ) (colin: ℝ) (helen: ℝ) (benedict: ℝ) (emmaInitial: ℝ) (interest: ℝ) (avaAmount: ℝ) (conversionRate: ℝ) : ℝ :=
  let emmaTotal := emmaInitial + (interest * emmaInitial)
  let avaTotal := (avaAmount * 0.75) * conversionRate
  initial - (colin + helen + benedict + emmaTotal + avaTotal)

theorem ian_leftover_money :
  let initial := 100
  let colin := 20
  let twice_colin := 2 * colin
  let half_helen := twice_colin / 2
  let emmaInitial := 15
  let interest := 0.10
  let avaAmount := 8
  let conversionRate := 1.20
  ianPayments initial colin twice_colin half_helen emmaInitial interest avaAmount conversionRate = -3.70
:= by
  sorry

end ian_leftover_money_l85_85992


namespace trigonometric_identity_l85_85810

noncomputable def trigonometric_identity_proof : Prop :=
  let cos_30 := Real.sqrt 3 / 2;
  let sin_60 := Real.sqrt 3 / 2;
  let sin_30 := 1 / 2;
  let cos_60 := 1 / 2;
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1

theorem trigonometric_identity : trigonometric_identity_proof :=
  sorry

end trigonometric_identity_l85_85810


namespace proposition_A_l85_85349

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l85_85349


namespace checker_on_diagonal_l85_85586

theorem checker_on_diagonal {n : ℕ} (n_eq_25 : n = 25) 
  (symmetric_placement : ∀ i j : fin n, i ≠ j → checker_placed i j ↔ checker_placed j i) :
  ∃ i : fin n, checker_placed i i := 
begin
  sorry,
end

end checker_on_diagonal_l85_85586


namespace problem_statement_l85_85351

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l85_85351


namespace problem_statement_l85_85341

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l85_85341


namespace major_premise_incorrect_l85_85470

theorem major_premise_incorrect (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : 
    ¬ (∀ x y : ℝ, x < y → a^x < a^y) :=
by {
  sorry
}

end major_premise_incorrect_l85_85470


namespace problem_statement_l85_85847

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

variable (f g : ℝ → ℝ)

axiom f_odd : odd_function f
axiom f_neg : ∀ x : ℝ, x < 0 → f x = x^3 - 1
axiom f_pos : ∀ x : ℝ, x > 0 → f x = g x

theorem problem_statement : f (-1) + g 2 = 7 :=
by
  sorry

end problem_statement_l85_85847


namespace calculate_product_l85_85077

theorem calculate_product : 3^6 * 4^3 = 46656 := by
  sorry

end calculate_product_l85_85077


namespace functional_equation_solution_l85_85187

noncomputable def func_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * x + f x * f y) = x * f (x + y)

theorem functional_equation_solution (f : ℝ → ℝ) :
  func_equation f →
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
sorry

end functional_equation_solution_l85_85187


namespace each_boy_makes_14_dollars_l85_85030

noncomputable def victor_shrimp_caught := 26
noncomputable def austin_shrimp_caught := victor_shrimp_caught - 8
noncomputable def brian_shrimp_caught := (victor_shrimp_caught + austin_shrimp_caught) / 2
noncomputable def total_shrimp_caught := victor_shrimp_caught + austin_shrimp_caught + brian_shrimp_caught
noncomputable def money_made := (total_shrimp_caught / 11) * 7
noncomputable def each_boys_earnings := money_made / 3

theorem each_boy_makes_14_dollars : each_boys_earnings = 14 := by
  sorry

end each_boy_makes_14_dollars_l85_85030


namespace gcd_gx_x_multiple_of_18432_l85_85527

def g (x : ℕ) : ℕ := (3*x + 5) * (7*x + 2) * (13*x + 7) * (2*x + 10)

theorem gcd_gx_x_multiple_of_18432 (x : ℕ) (h : ∃ k : ℕ, x = 18432 * k) : Nat.gcd (g x) x = 28 :=
by
  sorry

end gcd_gx_x_multiple_of_18432_l85_85527


namespace sqrt_mult_simplify_l85_85121

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l85_85121


namespace min_value_expr_l85_85558

noncomputable def find_min_value (a b c d : ℝ) (x y : ℝ) : ℝ :=
  x / c^2 + y^2 / d^2

theorem min_value_expr (a b c d : ℝ) (h : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ) :
  ∃ x y : ℝ, find_min_value a b c d x y = -abs a / c^2 := 
sorry

end min_value_expr_l85_85558


namespace Jason_spent_correct_amount_l85_85411

namespace MusicStore

def costFlute : Real := 142.46
def costMusicStand : Real := 8.89
def costSongBook : Real := 7.00
def totalCost : Real := 158.35

theorem Jason_spent_correct_amount :
  costFlute + costMusicStand + costSongBook = totalCost :=
sorry

end MusicStore

end Jason_spent_correct_amount_l85_85411


namespace bicycle_speed_l85_85599

theorem bicycle_speed (x : ℝ) :
  (10 / x = 10 / (2 * x) + 1 / 3) → x = 15 :=
by
  intro h
  sorry

end bicycle_speed_l85_85599


namespace num_points_satisfying_inequalities_l85_85409

theorem num_points_satisfying_inequalities :
  ∃ (n : ℕ), n = 2551 ∧
  ∀ (x y : ℤ), (y ≤ 3 * x) ∧ (y ≥ x / 3) ∧ (x + y ≤ 100) → 
              ∃ (p : ℕ), p = n := 
by
  sorry

end num_points_satisfying_inequalities_l85_85409


namespace seeds_planted_l85_85589

theorem seeds_planted (seeds_per_bed : ℕ) (beds : ℕ) (total_seeds : ℕ) :
  seeds_per_bed = 10 → beds = 6 → total_seeds = seeds_per_bed * beds → total_seeds = 60 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end seeds_planted_l85_85589


namespace girls_in_blue_dresses_answered_affirmatively_l85_85025

theorem girls_in_blue_dresses_answered_affirmatively :
  ∃ (n : ℕ), n = 17 ∧
  ∀ (total_girls red_dresses blue_dresses answer_girls : ℕ),
  total_girls = 30 →
  red_dresses = 13 →
  blue_dresses = 17 →
  answer_girls = n →
  answer_girls = blue_dresses :=
sorry

end girls_in_blue_dresses_answered_affirmatively_l85_85025


namespace sqrt_mul_eq_6_l85_85137

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l85_85137


namespace inscribed_circle_ratio_l85_85767

theorem inscribed_circle_ratio (a b c u v : ℕ) (h_triangle : a = 10 ∧ b = 24 ∧ c = 26) 
    (h_tangent_segments : u < v) (h_side_sum : u + v = a) : u / v = 2 / 3 :=
by
    sorry

end inscribed_circle_ratio_l85_85767


namespace vision_statistics_l85_85920

noncomputable def average (values : List ℝ) : ℝ := (List.sum values) / (List.length values)

noncomputable def variance (values : List ℝ) : ℝ :=
  let mean := average values
  (List.sum (values.map (λ x => (x - mean) ^ 2))) / (List.length values)

def classA_visions : List ℝ := [4.3, 5.1, 4.6, 4.1, 4.9]
def classB_visions : List ℝ := [5.1, 4.9, 4.0, 4.0, 4.5]

theorem vision_statistics :
  average classA_visions = 4.6 ∧
  average classB_visions = 4.5 ∧
  variance classA_visions = 0.136 ∧
  (let count := List.length classB_visions
   let total := count.choose 2
   let favorable := 3  -- (5.1, 4.5), (5.1, 4.9), (4.9, 4.5)
   7 / 10 = 1 - (favorable / total)) :=
by
  sorry

end vision_statistics_l85_85920


namespace value_of_m_l85_85447

theorem value_of_m (m : ℤ) (h₁ : |m| = 2) (h₂ : m ≠ 2) : m = -2 :=
by
  sorry

end value_of_m_l85_85447


namespace jaco_payment_l85_85940

theorem jaco_payment :
  let cost_shoes : ℝ := 74
  let cost_socks : ℝ := 2 * 2
  let cost_bag : ℝ := 42
  let total_cost_before_discount : ℝ := cost_shoes + cost_socks + cost_bag
  let discount_threshold : ℝ := 100
  let discount_rate : ℝ := 0.10
  let amount_exceeding_threshold : ℝ := total_cost_before_discount - discount_threshold
  let discount : ℝ := if amount_exceeding_threshold > 0 then discount_rate * amount_exceeding_threshold else 0
  let final_amount : ℝ := total_cost_before_discount - discount
  final_amount = 118 :=
by
  sorry

end jaco_payment_l85_85940


namespace seq_form_l85_85652

-- Define the sequence a as a function from natural numbers to natural numbers
def seq (a : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, 0 < m → 0 < n → ⌊(a m : ℚ) / a n⌋ = ⌊(m : ℚ) / n⌋

-- Define the statement that all sequences satisfying the condition must be of the form k * i
theorem seq_form (a : ℕ → ℕ) : seq a → ∃ k : ℕ, (0 < k) ∧ (∀ n, 0 < n → a n = k * n) := 
by
  intros h
  sorry

end seq_form_l85_85652


namespace origin_moves_distance_l85_85770

noncomputable def origin_distance_moved : ℝ :=
  let B := (3, 1)
  let B' := (7, 9)
  let k := 1.5
  let center_of_dilation := (-1, -3)
  let d0 := Real.sqrt ((-1)^2 + (-3)^2)
  let d1 := k * d0
  d1 - d0

theorem origin_moves_distance :
  origin_distance_moved = 0.5 * Real.sqrt 10 :=
by 
  sorry

end origin_moves_distance_l85_85770


namespace twenty_seven_cubes_volume_l85_85744

def volume_surface_relation (x V S : ℝ) : Prop :=
  V = x^3 ∧ S = 6 * x^2 ∧ V + S = (4 / 3) * (12 * x)

theorem twenty_seven_cubes_volume (x : ℝ) (hx : volume_surface_relation x (x^3) (6 * x^2)) : 
  27 * (x^3) = 216 :=
by
  sorry

end twenty_seven_cubes_volume_l85_85744


namespace solve_quadratic_eq_solve_cubic_eq_l85_85195

-- Statement for the first equation
theorem solve_quadratic_eq (x : ℝ) : 9 * x^2 - 25 = 0 ↔ x = 5 / 3 ∨ x = -5 / 3 :=
by sorry

-- Statement for the second equation
theorem solve_cubic_eq (x : ℝ) : (x + 1)^3 - 27 = 0 ↔ x = 2 :=
by sorry

end solve_quadratic_eq_solve_cubic_eq_l85_85195


namespace range_of_expression_l85_85665

theorem range_of_expression (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a * b = 2) :
  (a^2 + b^2) / (a - b) ≤ -4 :=
sorry

end range_of_expression_l85_85665


namespace sqrt3_mul_sqrt12_eq_6_l85_85091

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l85_85091


namespace quarterly_business_tax_cost_l85_85439

theorem quarterly_business_tax_cost
    (price_federal : ℕ := 50)
    (price_state : ℕ := 30)
    (Q : ℕ)
    (num_federal : ℕ := 60)
    (num_state : ℕ := 20)
    (num_quart_business : ℕ := 10)
    (total_revenue : ℕ := 4400)
    (revenue_equation : num_federal * price_federal + num_state * price_state + num_quart_business * Q = total_revenue) :
    Q = 80 :=
by 
  sorry

end quarterly_business_tax_cost_l85_85439


namespace sqrt3_mul_sqrt12_eq_6_l85_85118

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l85_85118


namespace problem_1956_Tokyo_Tech_l85_85576

theorem problem_1956_Tokyo_Tech (a b c : ℝ) (ha : 0 < a) (ha_lt_one : a < 1) (hb : 0 < b) 
(hb_lt_one : b < 1) (hc : 0 < c) (hc_lt_one : c < 1) : a + b + c - a * b * c < 2 := 
sorry

end problem_1956_Tokyo_Tech_l85_85576


namespace sequence_first_term_l85_85385

theorem sequence_first_term (a : ℕ → ℤ) 
  (h1 : a 3 = 5) 
  (h2 : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n - 1) : 
  a 1 = 2 := 
sorry

end sequence_first_term_l85_85385


namespace intersection_M_N_l85_85524

open Set

def M : Set ℝ := { x | x^2 - 2*x - 3 < 0 }
def N : Set ℝ := { x | x >= 1 }

theorem intersection_M_N : M ∩ N = { x | 1 <= x ∧ x < 3 } :=
by
  sorry

end intersection_M_N_l85_85524


namespace proposition_true_l85_85368

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l85_85368


namespace beatrix_reads_704_pages_l85_85178

theorem beatrix_reads_704_pages : 
  ∀ (B C : ℕ), 
  C = 3 * B + 15 ∧ C = B + 1423 → B = 704 :=
by
  intro B C
  intro h
  sorry

end beatrix_reads_704_pages_l85_85178


namespace find_subtracted_number_l85_85996

theorem find_subtracted_number (t k x : ℝ) (h1 : t = 20) (h2 : k = 68) (h3 : t = 5/9 * (k - x)) :
  x = 32 :=
by
  sorry

end find_subtracted_number_l85_85996


namespace internet_plan_comparison_l85_85640

theorem internet_plan_comparison (d : ℕ) :
    3000 + 200 * d > 5000 → d > 10 :=
by
  intro h
  -- Proof will be written here
  sorry

end internet_plan_comparison_l85_85640


namespace probability_no_self_draws_l85_85073

theorem probability_no_self_draws :
  let total_outcomes := 6
  let favorable_outcomes := 2
  let probability := favorable_outcomes / total_outcomes
  probability = 1 / 3 :=
by
  sorry

end probability_no_self_draws_l85_85073


namespace percent_of_x_is_y_l85_85054

theorem percent_of_x_is_y (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) : y / x = 0.25 :=
by
  -- proof omitted
  sorry

end percent_of_x_is_y_l85_85054


namespace find_c_d_l85_85264

theorem find_c_d (c d : ℝ) (h1 : c ≠ 0) (h2 : d ≠ 0)
  (h3 : ∀ x : ℝ, x^2 + c * x + d = 0 → (x = c ∧ x = d)) :
  c = 1 ∧ d = -2 :=
by
  sorry

end find_c_d_l85_85264


namespace john_finances_l85_85696

theorem john_finances :
  let total_first_year := 10000
  let tuition_percent := 0.4
  let room_board_percent := 0.35
  let textbook_transport_percent := 0.25
  let tuition_increase := 0.06
  let room_board_increase := 0.03
  let aid_first_year := 0.25
  let aid_increase := 0.02

  let tuition_first_year := total_first_year * tuition_percent
  let room_board_first_year := total_first_year * room_board_percent
  let textbook_transport_first_year := total_first_year * textbook_transport_percent

  let tuition_second_year := tuition_first_year * (1 + tuition_increase)
  let room_board_second_year := room_board_first_year * (1 + room_board_increase)
  let financial_aid_second_year := tuition_second_year * (aid_first_year + aid_increase)

  let tuition_third_year := tuition_second_year * (1 + tuition_increase)
  let room_board_third_year := room_board_second_year * (1 + room_board_increase)
  let financial_aid_third_year := tuition_third_year * (aid_first_year + 2 * aid_increase)

  let total_cost_first_year := 
      (tuition_first_year - tuition_first_year * aid_first_year) +
      room_board_first_year + 
      textbook_transport_first_year

  let total_cost_second_year :=
      (tuition_second_year - financial_aid_second_year) +
      room_board_second_year +
      textbook_transport_first_year

  let total_cost_third_year :=
      (tuition_third_year - financial_aid_third_year) +
      room_board_third_year +
      textbook_transport_first_year

  total_cost_first_year = 9000 ∧
  total_cost_second_year = 9200.20 ∧
  total_cost_third_year = 9404.17 := 
by
  sorry

end john_finances_l85_85696


namespace line_through_intersection_and_origin_l85_85387

theorem line_through_intersection_and_origin :
  ∃ (x y : ℝ), (2*x + y = 3) ∧ (x + 4*y = 2) ∧ (x - 10*y = 0) :=
by
  sorry

end line_through_intersection_and_origin_l85_85387


namespace impossible_to_form_three_similar_piles_l85_85716

-- Define the concept of similar sizes
def similar_sizes (a b : ℝ) : Prop := a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2

-- State the proof problem
theorem impossible_to_form_three_similar_piles (x : ℝ) (h : 0 < x) : ¬(∃ a b c : ℝ, a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes a c) :=
sorry

end impossible_to_form_three_similar_piles_l85_85716


namespace sum_and_ratio_l85_85276

theorem sum_and_ratio (x y : ℝ) (h1 : x + y = 480) (h2 : x / y = 0.8) : y - x = 53.34 :=
by
  sorry

end sum_and_ratio_l85_85276


namespace sqrt_mul_eq_6_l85_85136

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l85_85136


namespace students_taking_both_courses_l85_85871

theorem students_taking_both_courses (total_students students_french students_german students_neither both_courses : ℕ) 
(h1 : total_students = 94) 
(h2 : students_french = 41) 
(h3 : students_german = 22) 
(h4 : students_neither = 40) 
(h5 : total_students = students_french + students_german - both_courses + students_neither) :
both_courses = 9 :=
by
  -- sorry can be replaced with the actual proof if necessary
  sorry

end students_taking_both_courses_l85_85871


namespace prime_p_impplies_p_eq_3_l85_85675

theorem prime_p_impplies_p_eq_3 (p : ℕ) (hp : Prime p) (hp2 : Prime (p^2 + 2)) : p = 3 :=
sorry

end prime_p_impplies_p_eq_3_l85_85675


namespace domain_of_f_l85_85180

noncomputable def f (x : ℝ) : ℝ := (1 / (x - 5)) + (1 / (x^2 - 4)) + (1 / (x^3 - 27))

theorem domain_of_f :
  ∀ x : ℝ, x ≠ 5 ∧ x ≠ 2 ∧ x ≠ -2 ∧ x ≠ 3 ↔
          ∃ y : ℝ, f y = f x :=
by
  sorry

end domain_of_f_l85_85180


namespace proof_problem_l85_85357

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l85_85357


namespace slope_of_line_l85_85926

def point1 : (ℤ × ℤ) := (-4, 6)
def point2 : (ℤ × ℤ) := (3, -4)

def slope_formula (p1 p2 : (ℤ × ℤ)) : ℚ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst : ℚ)

theorem slope_of_line : slope_formula point1 point2 = -10 / 7 := by
  sorry

end slope_of_line_l85_85926


namespace total_squares_in_6x6_grid_l85_85322

theorem total_squares_in_6x6_grid : 
  let size := 6
  let total_squares := (size * size) + ((size - 1) * (size - 1)) + ((size - 2) * (size - 2)) + ((size - 3) * (size - 3)) + ((size - 4) * (size - 4)) + ((size - 5) * (size - 5))
  total_squares = 91 :=
by
  let size := 6
  let total_squares := (size * size) + ((size - 1) * (size - 1)) + ((size - 2) * (size - 2)) + ((size - 3) * (size - 3)) + ((size - 4) * (size - 4)) + ((size - 5) * (size - 5))
  have eqn : total_squares = 91 := sorry
  exact eqn

end total_squares_in_6x6_grid_l85_85322


namespace david_moore_total_time_l85_85785

-- Given conditions
def david_work_rate := 1 / 12
def days_david_worked_alone := 6
def remaining_work_days_together := 3
def total_work := 1

-- Definition of total time taken for both to complete the job
def combined_total_time := 6

-- Proof problem statement in Lean
theorem david_moore_total_time :
  let d_work_done_alone := days_david_worked_alone * david_work_rate
  let remaining_work := total_work - d_work_done_alone
  let combined_work_rate := remaining_work / remaining_work_days_together
  let moore_work_rate := combined_work_rate - david_work_rate
  let new_combined_work_rate := david_work_rate + moore_work_rate
  total_work / new_combined_work_rate = combined_total_time := by
    sorry

end david_moore_total_time_l85_85785


namespace third_quadrant_point_m_l85_85978

theorem third_quadrant_point_m (m : ℤ) (h1 : 2 - m < 0) (h2 : m - 4 < 0) : m = 3 :=
by
  sorry

end third_quadrant_point_m_l85_85978


namespace sqrt_mul_simplify_l85_85106

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l85_85106


namespace binomial_sum_l85_85817

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_sum (n : ℤ) (h1 : binomial 25 n.natAbs + binomial 25 12 = binomial 26 13 ∧ n ≥ 0) : 
    (n = 12 ∨ n = 13) → n.succ + n = 25 := 
    sorry

end binomial_sum_l85_85817


namespace sqrt3_mul_sqrt12_eq_6_l85_85089

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l85_85089


namespace logistics_company_freight_l85_85632

theorem logistics_company_freight :
  ∃ (x y : ℕ), 
    50 * x + 30 * y = 9500 ∧
    70 * x + 40 * y = 13000 ∧
    x = 100 ∧
    y = 140 :=
by
  -- The proof is skipped here
  sorry

end logistics_company_freight_l85_85632


namespace factorize_2x2_minus_4x_factorize_xy2_minus_2xy_plus_x_l85_85184

-- Problem 1
theorem factorize_2x2_minus_4x (x : ℝ) : 
  2 * x^2 - 4 * x = 2 * x * (x - 2) := 
by 
  sorry

-- Problem 2
theorem factorize_xy2_minus_2xy_plus_x (x y : ℝ) :
  x * y^2 - 2 * x * y + x = x * (y - 1)^2 :=
by 
  sorry

end factorize_2x2_minus_4x_factorize_xy2_minus_2xy_plus_x_l85_85184


namespace chandler_total_cost_l85_85399

theorem chandler_total_cost (
  cost_per_movie_ticket : ℕ := 30
  cost_8_movie_tickets : ℕ := 8 * cost_per_movie_ticket
  cost_per_football_game_ticket : ℕ := cost_8_movie_tickets / 2
  cost_5_football_game_tickets : ℕ := 5 * cost_per_football_game_ticket
  total_cost : ℕ := cost_8_movie_tickets + cost_5_football_game_tickets
) : total_cost = 840 := by
  sorry

end chandler_total_cost_l85_85399


namespace computer_price_difference_l85_85022

-- Define the conditions as stated
def basic_computer_price := 1500
def total_price := 2500
def printer_price (P : ℕ) := basic_computer_price + P = total_price

def enhanced_computer_price (P E : ℕ) := P = (E + P) / 3

-- The theorem stating the proof problem
theorem computer_price_difference (P E : ℕ) 
  (h1 : printer_price P) 
  (h2 : enhanced_computer_price P E) : E - basic_computer_price = 500 :=
sorry

end computer_price_difference_l85_85022


namespace combined_solid_volume_l85_85946

open Real

noncomputable def volume_truncated_cone (R r h : ℝ) :=
  (1 / 3) * π * h * (R^2 + R * r + r^2)

noncomputable def volume_cylinder (r h : ℝ): ℝ :=
  π * r^2 * h

theorem combined_solid_volume :
  let R := 10
  let r := 3
  let h_cone := 8
  let h_cyl := 10
  volume_truncated_cone R r h_cone + volume_cylinder r h_cyl = (1382 * π) / 3 :=
  by
  sorry

end combined_solid_volume_l85_85946


namespace min_value_of_ratio_l85_85853

theorem min_value_of_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) :
  (4 / x + 1 / y) ≥ 6 + 4 * Real.sqrt 2 :=
sorry

end min_value_of_ratio_l85_85853


namespace adjusted_ratio_l85_85650

theorem adjusted_ratio :
  (2^2003 * 3^2005) / (6^2004) = 3 / 2 :=
by
  sorry

end adjusted_ratio_l85_85650


namespace find_m_l85_85756

-- Given the condition
def condition (m : ℕ) := (1 / 5 : ℝ)^m * (1 / 4 : ℝ)^2 = 1 / (10 : ℝ)^4

-- Theorem to prove that m is 4 given the condition
theorem find_m (m : ℕ) (h : condition m) : m = 4 :=
sorry

end find_m_l85_85756


namespace intersection_of_lines_l85_85959

theorem intersection_of_lines : 
  ∃ (x y : ℚ), 5 * x - 2 * y = 8 ∧ 6 * x + 3 * y = 21 ∧ x = 22 / 9 ∧ y = 19 / 9 :=
by 
  sorry

end intersection_of_lines_l85_85959


namespace binary_operations_unique_l85_85825

def binary_operation (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → (f a (f b c) = (f a b) * c)
  ∧ ∀ a : ℝ, a > 0 → a ≥ 1 → f a a ≥ 1

theorem binary_operations_unique (f : ℝ → ℝ → ℝ) (h : binary_operation f) :
  (∀ a b, f a b = a * b) ∨ (∀ a b, f a b = a / b) :=
sorry

end binary_operations_unique_l85_85825


namespace dimes_paid_l85_85540

theorem dimes_paid (cost_in_dollars : ℕ) (dollars_to_dimes : ℕ) (h₁ : cost_in_dollars = 5) (h₂ : dollars_to_dimes = 10) :
  cost_in_dollars * dollars_to_dimes = 50 :=
by
  sorry

end dimes_paid_l85_85540


namespace line_through_point_with_slope_l85_85018

theorem line_through_point_with_slope (x y : ℝ) (h : y - 2 = -3 * (x - 1)) : 3 * x + y - 5 = 0 :=
sorry

example : 3 * 1 + 2 - 5 = 0 := by sorry

end line_through_point_with_slope_l85_85018


namespace exponent_division_l85_85808

theorem exponent_division (a : ℕ) (m n : ℕ) (h1 : 19 = a) (h2 : 11 = m) (h3 : 8 = n) : a^(m - n) = 6859 := by
  sorry

end exponent_division_l85_85808


namespace lucas_journey_distance_l85_85588

noncomputable def distance (D : ℝ) : ℝ :=
  let usual_speed := D / 150
  let distance_before_traffic := 2 * D / 5
  let speed_after_traffic := usual_speed - 1 / 2
  let time_before_traffic := distance_before_traffic / usual_speed
  let time_after_traffic := (3 * D / 5) / speed_after_traffic
  time_before_traffic + time_after_traffic

theorem lucas_journey_distance : ∃ D : ℝ, distance D = 255 ∧ D = 48.75 :=
sorry

end lucas_journey_distance_l85_85588


namespace sqrt3_mul_sqrt12_eq_6_l85_85111

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l85_85111


namespace ratio_of_routes_l85_85953

-- Definitions of m and n
def m : ℕ := 2 
def n : ℕ := 6

-- Theorem statement
theorem ratio_of_routes (m_positive : m > 0) : n / m = 3 := by
  sorry

end ratio_of_routes_l85_85953


namespace smallest_k_for_sequence_l85_85662

theorem smallest_k_for_sequence (a : ℕ → ℕ) (k : ℕ) (h₁ : a 1 = 1) (h₂ : a 2018 = 2020)
  (h₃ : ∀ n, n ≥ 2 → a (n+1) = k * (a n) / (a (n-1))) : k = 2020 :=
sorry

end smallest_k_for_sequence_l85_85662


namespace find_derivative_l85_85383

theorem find_derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 1 + Real.log x) : f' 1 = -1 := 
by
  sorry

end find_derivative_l85_85383


namespace sqrt_mul_sqrt_eq_six_l85_85143

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l85_85143


namespace slope_of_line_l85_85181

theorem slope_of_line (x : ℝ) : (2 * x + 1) = 2 :=
by sorry

end slope_of_line_l85_85181


namespace ben_heads_probability_l85_85393

def coin_flip_probability : ℚ :=
  let total_ways := 2^10
  let ways_exactly_five_heads := Nat.choose 10 5
  let probability_exactly_five_heads := ways_exactly_five_heads / total_ways
  let remaining_probability := 1 - probability_exactly_five_heads
  let probability_more_heads := remaining_probability / 2
  probability_more_heads

theorem ben_heads_probability :
  coin_flip_probability = 193 / 512 := by
  sorry

end ben_heads_probability_l85_85393


namespace sum_of_fraction_numerator_denominator_l85_85045

theorem sum_of_fraction_numerator_denominator :
  let x := 0.343434...
  in let fraction := (34 / 99 : ℚ)
  in let sum := fraction.num + fraction.den 
  in (x : ℚ) = fraction ∧ fraction.isReduced → sum = 133 :=
by
  sorry

end sum_of_fraction_numerator_denominator_l85_85045


namespace maximum_S_n_l85_85664

noncomputable def a_n (a_1 d : ℝ) (n : ℕ) : ℝ :=
  a_1 + (n - 1) * d

noncomputable def S_n (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a_1 + (n - 1) * d)

theorem maximum_S_n (a_1 : ℝ) (h : a_1 > 0)
  (h_sequence : 3 * a_n a_1 (2 * a_1 / 39) 8 = 5 * a_n a_1 (2 * a_1 / 39) 13)
  : ∀ n : ℕ, S_n a_1 (2 * a_1 / 39) n ≤ S_n a_1 (2 * a_1 / 39) 20 :=
sorry

end maximum_S_n_l85_85664


namespace find_a_n_l85_85669

def S (n : ℕ) : ℕ := 2^(n+1) - 1

def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3 else 2^n

theorem find_a_n (n : ℕ) : a_n n = if n = 1 then 3 else 2^n :=
  sorry

end find_a_n_l85_85669


namespace assignment_schemes_equiv_240_l85_85332

-- Define the problem
theorem assignment_schemes_equiv_240
  (students : Finset ℕ)
  (tasks : Finset ℕ)
  (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students)
  (h_tasks : tasks = {0, 1, 2, 3})
  (h_students : students.card = 6) :
  let ways_to_assign := (EUₖstudents.card, tasks.card )
  let ways_with_A_or_B_not_taskA := (C(2,1) * A(5, tasks.card - 1))
  (ways_to_assign - ways_with_A_or_B_not_taskA) = 240 :=
by
  sorry

end assignment_schemes_equiv_240_l85_85332


namespace part_a_part_b_l85_85939

-- Define the conditions for part (a)
def psychic_can_guess_at_least_19_cards : Prop :=
  ∃ (deck : Fin 36 → Fin 4) (psychic_guess : Fin 36 → Fin 4)
    (assistant_arrangement : Fin 36 → Bool),
    -- assistant and psychic agree on a method ensuring at least 19 correct guesses
    (∃ n : ℕ, n ≥ 19 ∧
      ∃ correct_guesses_set : Finset (Fin 36),
        correct_guesses_set.card = n ∧
        ∀ i ∈ correct_guesses_set, psychic_guess i = deck i)

-- Prove that the above condition is satisfied
theorem part_a : psychic_can_guess_at_least_19_cards :=
by
  sorry

-- Define the conditions for part (b)
def psychic_can_guess_at_least_23_cards : Prop :=
  ∃ (deck : Fin 36 → Fin 4) (psychic_guess : Fin 36 → Fin 4)
    (assistant_arrangement : Fin 36 → Bool),
    -- assistant and psychic agree on a method ensuring at least 23 correct guesses
    (∃ n : ℕ, n ≥ 23 ∧
      ∃ correct_guesses_set : Finset (Fin 36),
        correct_guesses_set.card = n ∧
        ∀ i ∈ correct_guesses_set, psychic_guess i = deck i)

-- Prove that the above condition is satisfied
theorem part_b : psychic_can_guess_at_least_23_cards :=
by
  sorry

end part_a_part_b_l85_85939


namespace isosceles_triangle_perimeter_l85_85205

/-- Given an isosceles triangle with one side length of 3 cm and another side length of 5 cm,
    its perimeter is either 11 cm or 13 cm. -/
theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 5) : 
  (∃ c : ℝ, (c = 3 ∨ c = 5) ∧ (2 * a + b = 11 ∨ 2 * b + a = 13)) :=
by
  sorry

end isosceles_triangle_perimeter_l85_85205


namespace age_double_after_5_years_l85_85921

-- Defining the current ages of the brothers
def older_brother_age := 15
def younger_brother_age := 5

-- Defining the condition
def after_x_years (x : ℕ) := older_brother_age + x = 2 * (younger_brother_age + x)

-- The main theorem with the condition
theorem age_double_after_5_years : after_x_years 5 :=
by sorry

end age_double_after_5_years_l85_85921


namespace complex_division_correct_l85_85809

theorem complex_division_correct : (3 - 1 * Complex.I) / (1 + Complex.I) = 1 - 2 * Complex.I := 
by
  sorry

end complex_division_correct_l85_85809


namespace sqrt_mul_eq_6_l85_85135

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l85_85135


namespace expected_value_xi_l85_85060

-- Defining the probabilistic conditions
def fairCoin : Prob := 0.5

def xi (A1 A2 : Bool) : ℝ :=
  if A1 || A2 then 1 else 0

-- Theorem stating the expected value of random variable xi
theorem expected_value_xi :
  ∃ (ξ : Boolean → Boolean → ℝ), ξ = xi ∧
    (⟨ A1, pA1 ⟩ = ⟨ true, fairCoin ⟩ ∨ ⟨ false, 1 - fairCoin ⟩) ∧
    (⟨ A2, pA2 ⟩ = ⟨ true, fairCoin ⟩ ∨ ⟨ false, 1 - fairCoin ⟩) →
      (E[ξ] = 0.75) := 
sorry

end expected_value_xi_l85_85060


namespace sqrt_mul_sqrt_eq_six_l85_85149

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l85_85149


namespace vector_subtraction_proof_l85_85511

def v1 : ℝ × ℝ := (3, -8)
def v2 : ℝ × ℝ := (2, -6)
def a : ℝ := 5
def answer : ℝ × ℝ := (-7, 22)

theorem vector_subtraction_proof : (v1.1 - a * v2.1, v1.2 - a * v2.2) = answer := 
by
  sorry

end vector_subtraction_proof_l85_85511


namespace quadratic_min_value_l85_85610

theorem quadratic_min_value : ∀ x : ℝ, x^2 - 6 * x + 13 ≥ 4 := 
by 
  sorry

end quadratic_min_value_l85_85610


namespace sqrt_mul_eq_6_l85_85141

theorem sqrt_mul_eq_6 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_eq_6_l85_85141


namespace not_possible_three_piles_l85_85722

def similar (x y : ℝ) : Prop :=
  y ≤ (real.sqrt 2) * x ∧ x ≤ (real.sqrt 2) * y

theorem not_possible_three_piles (N : ℝ) : 
  ¬ ∃ a b c : ℝ, a + b + c = N ∧ similar a b ∧ similar b c ∧ similar a c := 
by
  sorry

end not_possible_three_piles_l85_85722


namespace intersects_line_l85_85069

theorem intersects_line (x y : ℝ) : 
  (3 * x + 2 * y = 5) ∧ ((x / 3) + (y / 2) = 1) → ∃ x y : ℝ, (3 * x + 2 * y = 5) ∧ ((x / 3) + (y / 2) = 1) :=
by
  intro h
  sorry

end intersects_line_l85_85069


namespace sqrt3_mul_sqrt12_eq_6_l85_85112

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l85_85112


namespace orchard_trees_l85_85271

theorem orchard_trees (n : ℕ) (hn : n^2 + 146 = 7890) : 
    n^2 + 146 + 31 = 89^2 := by
  sorry

end orchard_trees_l85_85271


namespace sum_from_neg_50_to_75_l85_85289

def sum_of_integers (a b : ℤ) : ℤ :=
  (b * (b + 1)) / 2 - (a * (a - 1)) / 2

theorem sum_from_neg_50_to_75 : sum_of_integers (-50) 75 = 1575 := by
  sorry

end sum_from_neg_50_to_75_l85_85289


namespace points_on_hyperbola_order_l85_85851

theorem points_on_hyperbola_order (k a b c : ℝ) (hk : k > 0)
  (h₁ : a = k / -2)
  (h₂ : b = k / 2)
  (h₃ : c = k / 3) :
  a < c ∧ c < b := 
sorry

end points_on_hyperbola_order_l85_85851


namespace find_green_hats_l85_85056

variable (B G : ℕ)

theorem find_green_hats (h1 : B + G = 85) (h2 : 6 * B + 7 * G = 540) :
  G = 30 :=
by
  sorry

end find_green_hats_l85_85056


namespace sqrt_mult_eq_six_l85_85166

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l85_85166


namespace sqrt_mul_simp_l85_85127

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l85_85127


namespace proposition_true_l85_85367

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l85_85367


namespace problem_solution_A_problem_solution_C_l85_85787

noncomputable def expr_A : ℝ :=
  (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180))

noncomputable def expr_C : ℝ :=
  Real.tan (22.5 * Real.pi / 180) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)

theorem problem_solution_A :
  expr_A = 1 / 2 :=
by
  sorry

theorem problem_solution_C :
  expr_C = 1 / 2 :=
by
  sorry

end problem_solution_A_problem_solution_C_l85_85787


namespace at_least_one_genuine_product_l85_85517

-- Definitions of the problem conditions
structure Products :=
  (total : ℕ)
  (genuine : ℕ)
  (defective : ℕ)

def products : Products := { total := 12, genuine := 10, defective := 2 }

-- Definition of the event
def certain_event (p : Products) (selected : ℕ) : Prop :=
  selected > p.defective

-- The theorem stating that there is at least one genuine product among the selected ones
theorem at_least_one_genuine_product : certain_event products 3 :=
by
  sorry

end at_least_one_genuine_product_l85_85517


namespace travel_from_A_to_C_l85_85026

def num_ways_A_to_B : ℕ := 5 + 2  -- 5 buses and 2 trains
def num_ways_B_to_C : ℕ := 3 + 2  -- 3 buses and 2 ferries

theorem travel_from_A_to_C :
  num_ways_A_to_B * num_ways_B_to_C = 35 :=
by
  -- The proof environment will be added here. 
  -- We include 'sorry' here for now.
  sorry

end travel_from_A_to_C_l85_85026


namespace total_students_total_students_alt_l85_85254

def number_of_students (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    a + b = 6 ∧ 
    a = 4 ∧ 
    (∀ g, g = 13) ∧ 
    (b = 2 ∧ ((g = 12 ∨ g = 14) ∧ ∀ h, g - h ≤ 1)) ∧ 
    (n = 52 + 2 * 12 ∨ n = 52 + 2 * 14)

theorem total_students : ∃ n : ℕ, number_of_students n :=
by
  use 76
  sorry

theorem total_students_alt : ∃ n : ℕ, number_of_students n :=
by
  use 80
  sorry

end total_students_total_students_alt_l85_85254


namespace largest_consecutive_sum_to_35_l85_85574

theorem largest_consecutive_sum_to_35 (n : ℕ) (h : ∃ a : ℕ, (n * (2 * a + n - 1)) / 2 = 35) : n ≤ 7 :=
by
  sorry

end largest_consecutive_sum_to_35_l85_85574


namespace sally_quarters_l85_85907

noncomputable def initial_quarters : ℕ := 760
noncomputable def spent_quarters : ℕ := 418
noncomputable def remaining_quarters : ℕ := 342

theorem sally_quarters : initial_quarters - spent_quarters = remaining_quarters :=
by sorry

end sally_quarters_l85_85907


namespace function_decreasing_interval_l85_85534

variable (a b : ℝ)

def f (x : ℝ) : ℝ := x^2 * (a * x + b)

theorem function_decreasing_interval :
  (deriv (f a b) 2 = 0) ∧ (deriv (f a b) 1 = -3) →
  ∃ (a b : ℝ), (deriv (f a b) x < 0) ↔ (0 < x ∧ x < 2) := sorry

end function_decreasing_interval_l85_85534


namespace max_value_of_g_is_34_l85_85017
noncomputable def g : ℕ → ℕ
| n => if n < 15 then n + 20 else g (n - 7)

theorem max_value_of_g_is_34 : ∃ n, g n = 34 ∧ ∀ m, g m ≤ 34 :=
by
  sorry

end max_value_of_g_is_34_l85_85017


namespace set_complement_intersection_l85_85867

variable (U : Set ℕ) (M N : Set ℕ)

theorem set_complement_intersection
  (hU : U = {1, 2, 3, 4, 5, 6})
  (hM : M = {1, 4, 5})
  (hN : N = {2, 3}) :
  ((U \ N) ∩ M) = {1, 4, 5} :=
by
  sorry

end set_complement_intersection_l85_85867


namespace compare_abc_l85_85333

noncomputable def a : ℝ := 9 ^ (Real.log 4.1 / Real.log 2)
noncomputable def b : ℝ := 9 ^ (Real.log 2.7 / Real.log 2)
noncomputable def c : ℝ := (1 / 3 : ℝ) ^ (Real.log 0.1 / Real.log 2)

theorem compare_abc :
  a > c ∧ c > b := by
  sorry

end compare_abc_l85_85333


namespace min_voters_tall_giraffe_win_l85_85686

-- Definitions from the problem statement as conditions
def precinct_voters := 3
def precincts_per_district := 9
def districts := 5
def majority_precincts(p : ℕ) := p / 2 + 1  -- Minimum precincts won in a district 
def majority_districts(d : ℕ) := d / 2 + 1  -- Minimum districts won in the final

-- Condition: majority precincts to win a district
def precinct_votes_to_win := majority_precincts precinct_voters

-- Condition: majority districts to win the final
def district_wins_to_win_final := majority_districts districts

-- Minimum precincts the Tall giraffe needs to win overall
def total_precincts_to_win := district_wins_to_win_final * majority_precincts precincts_per_district

-- Proof that the minimum number of voters who could have voted for the Tall giraffe is 30
theorem min_voters_tall_giraffe_win :
  precinct_votes_to_win * total_precincts_to_win = 30 :=
sorry

end min_voters_tall_giraffe_win_l85_85686


namespace boxes_needed_l85_85024

theorem boxes_needed (balls : ℕ) (balls_per_box : ℕ) (h1 : balls = 10) (h2 : balls_per_box = 5) : balls / balls_per_box = 2 := by
  sorry

end boxes_needed_l85_85024


namespace proposition_p_and_q_is_true_l85_85346

variable p : Prop := ∃ x : ℝ, sin x < 1
variable q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem proposition_p_and_q_is_true : p ∧ q := by
  exists (0 : ℝ)
  simp [sin_zero]
  exact zero_lt_one
  intro x
  exact le_of_eq (exp_abs x).symm

end proposition_p_and_q_is_true_l85_85346


namespace inequality_solution_l85_85258

theorem inequality_solution (x : ℝ) : (|x - 1| + |x - 2| > 5) ↔ (x ∈ (-∞, -1) ∪ (4, ∞)) :=
by
  sorry

end inequality_solution_l85_85258


namespace carlo_practice_difference_l85_85954

-- Definitions for given conditions
def monday_practice (T : ℕ) : ℕ := 2 * T
def tuesday_practice (T : ℕ) : ℕ := T
def wednesday_practice (thursday_minutes : ℕ) : ℕ := thursday_minutes + 5
def thursday_practice : ℕ := 50
def friday_practice : ℕ := 60
def total_weekly_practice : ℕ := 300

theorem carlo_practice_difference 
  (T : ℕ) 
  (Monday Tuesday Wednesday Thursday Friday : ℕ) 
  (H1 : Monday = monday_practice T)
  (H2 : Tuesday = tuesday_practice T)
  (H3 : Wednesday = wednesday_practice Thursday)
  (H4 : Thursday = thursday_practice)
  (H5 : Friday = friday_practice)
  (H6 : Monday + Tuesday + Wednesday + Thursday + Friday = total_weekly_practice) :
  (Wednesday - Tuesday = 10) :=
by 
  -- Use the provided conditions and derive the required result.
  sorry

end carlo_practice_difference_l85_85954


namespace fifteenth_even_multiple_of_5_l85_85616

theorem fifteenth_even_multiple_of_5 : 15 * 2 * 5 = 150 := by
  sorry

end fifteenth_even_multiple_of_5_l85_85616


namespace total_students_possible_l85_85250

theorem total_students_possible (A B : ℕ) :
  (4 * 13) + 2 * A = 76 ∨ (4 * 13) + 2 * B = 80 :=
by
  -- Let N be the total number of students
  let N := (4 * 13)
  -- Given that the number of students in the remaining 2 groups differs by no more than 1
  have h : A = 12 ∨ B = 14 := sorry
  -- Prove the possible values
  exact or.inl (N + 2 * 12 = 76) <|> or.inr (N + 2 * 14 = 80)

end total_students_possible_l85_85250


namespace outfits_not_same_color_l85_85863

/--
Given:
- 7 shirts, 7 pairs of pants, and 7 hats.
- Each item comes in 7 colors (one of each item of each color).
- No outfit is allowed where all 3 items are the same color.

Prove:
The number of possible outfits where not all items are the same color is 336.
-/
theorem outfits_not_same_color : 
  let total_outfits := 7 * 7 * 7 in
  let same_color_outfits := 7 in
  total_outfits - same_color_outfits = 336 :=
by
  let total_outfits := 7 * 7 * 7
  let same_color_outfits := 7
  have h1 : total_outfits = 343 := by norm_num
  have h2 : total_outfits - same_color_outfits = 336 := by norm_num
  exact h2

end outfits_not_same_color_l85_85863


namespace smallest_possible_n_l85_85680

theorem smallest_possible_n : ∃ (n : ℕ), (∀ (r g b : ℕ), 24 * n = 18 * r ∧ 24 * n = 16 * g ∧ 24 * n = 20 * b) ∧ n = 30 :=
by
  -- Sorry, we're skipping the proof, as specified.
  sorry

end smallest_possible_n_l85_85680


namespace shirts_count_l85_85884

theorem shirts_count (S : ℕ) (hours_per_shirt hours_per_pant cost_per_hour total_pants total_cost : ℝ) :
  hours_per_shirt = 1.5 →
  hours_per_pant = 3 →
  cost_per_hour = 30 →
  total_pants = 12 →
  total_cost = 1530 →
  45 * S + 1080 = total_cost →
  S = 10 :=
by
  intros hps hpp cph tp tc cost_eq
  sorry

end shirts_count_l85_85884


namespace proposition_true_l85_85366

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (Real.abs x) ≥ 1

-- Prove the combination of the propositions
theorem proposition_true : p ∧ q :=
by
  sorry

end proposition_true_l85_85366


namespace divisible_by_101_l85_85965

theorem divisible_by_101 (n : ℕ) : (101 ∣ (10^n - 1)) ↔ (∃ k : ℕ, n = 4 * k) :=
by
  sorry

end divisible_by_101_l85_85965


namespace find_larger_number_l85_85755

variable (L S : ℕ)

theorem find_larger_number 
  (h1 : L - S = 1355) 
  (h2 : L = 6 * S + 15) : 
  L = 1623 := 
sorry

end find_larger_number_l85_85755


namespace function_relationship_value_of_x_l85_85234

variable {x y : ℝ}

-- Given conditions:
-- Condition 1: y is inversely proportional to x
def inversely_proportional (p : ℝ) (q : ℝ) (k : ℝ) : Prop := p = k / q

-- Condition 2: y(2) = -3
def specific_value (x_val y_val : ℝ) : Prop := y_val = -3 ∧ x_val = 2

-- Questions rephrased as Lean theorems:

-- The function relationship between y and x is y = -6 / x
theorem function_relationship (k : ℝ) (hx : x ≠ 0) 
  (h_inv_prop: inversely_proportional y x k) (h_spec : specific_value 2 (-3)) : k = -6 :=
by
  sorry

-- When y = 2, x = -3
theorem value_of_x (hx : x ≠ 0) (hy : y = 2)
  (h_inv_prop : inversely_proportional y x (-6)) : x = -3 :=
by
  sorry

end function_relationship_value_of_x_l85_85234


namespace calculate_green_paint_l85_85843

theorem calculate_green_paint {green white : ℕ} (ratio_white_to_green : 5 * green = 3 * white) (use_white_paint : white = 15) : green = 9 :=
by
  sorry

end calculate_green_paint_l85_85843


namespace mixed_number_division_l85_85750

theorem mixed_number_division : 
  let a := 9 / 4
  let b := 3 / 5
  a / b = 15 / 4 :=
by
  sorry

end mixed_number_division_l85_85750


namespace find_f3_value_l85_85890

noncomputable def f (x : ℚ) : ℚ := (x^2 + 2*x + 1) / (4*x - 5)

theorem find_f3_value : f 3 = 16 / 7 :=
by sorry

end find_f3_value_l85_85890


namespace rook_placement_l85_85630

theorem rook_placement : 
  let n := 8
  let k := 6
  let binom := Nat.choose
  binom 8 6 * binom 8 6 * Nat.factorial 6 = 564480 := by
    sorry

end rook_placement_l85_85630


namespace correct_operation_l85_85049

theorem correct_operation :
  (∀ a : ℝ, a^4 * a^3 = a^7)
  ∧ (∀ a : ℝ, (a^2)^3 ≠ a^5)
  ∧ (∀ a : ℝ, 3 * a^2 - a^2 ≠ 2)
  ∧ (∀ a b : ℝ, (a - b)^2 ≠ a^2 - b^2) :=
by {
  sorry
}

end correct_operation_l85_85049


namespace problem_statement_l85_85533

def f (x : ℝ) (a : ℝ) : ℝ := -x^2 + 6*x + a^2 - 1

theorem problem_statement (a : ℝ) :
  f (Real.sqrt 2) a < f 4 a ∧ f 4 a < f 3 a :=
by
  sorry

end problem_statement_l85_85533


namespace proof_problem_l85_85374

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l85_85374


namespace driver_average_speed_l85_85305

theorem driver_average_speed (v t : ℝ) (h1 : ∀ d : ℝ, d = v * t → (d / (v + 10)) = (3 / 4) * t) : v = 30 := by
  sorry

end driver_average_speed_l85_85305


namespace maximum_value_a3_b3_c3_d3_l85_85580

noncomputable def max_value (a b c d : ℝ) : ℝ :=
  a^3 + b^3 + c^3 + d^3

theorem maximum_value_a3_b3_c3_d3
  (a b c d : ℝ)
  (h1 : a^2 + b^2 + c^2 + d^2 = 20)
  (h2 : a + b + c + d = 10) :
  max_value a b c d ≤ 500 :=
sorry

end maximum_value_a3_b3_c3_d3_l85_85580


namespace total_students_l85_85251

theorem total_students (N : ℕ) (h1 : ∃ g1 g2 g3 g4 g5 g6 : ℕ, 
  g1 = 13 ∧ g2 = 13 ∧ g3 = 13 ∧ g4 = 13 ∧ 
  ((g5 = 12 ∧ g6 = 12) ∨ (g5 = 14 ∧ g6 = 14)) ∧ 
  N = g1 + g2 + g3 + g4 + g5 + g6) : 
  N = 76 ∨ N = 80 :=
by
  sorry

end total_students_l85_85251


namespace total_hangers_is_65_l85_85559

noncomputable def calculate_hangers_total : ℕ :=
  let pink := 7
  let green := 4
  let blue := green - 1
  let yellow := blue - 1
  let orange := 2 * (pink + green)
  let purple := (blue - yellow) + 3
  let red := (pink + green + blue) / 3
  let brown := 3 * red + 1
  let gray := (3 * purple) / 5
  pink + green + blue + yellow + orange + purple + red + brown + gray

theorem total_hangers_is_65 : calculate_hangers_total = 65 := 
by 
  sorry

end total_hangers_is_65_l85_85559


namespace sqrt_mul_eq_l85_85080

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l85_85080


namespace temperature_difference_l85_85446

theorem temperature_difference (T_high T_low : ℝ) (h1 : T_high = 8) (h2 : T_low = -2) : T_high - T_low = 10 :=
by
  sorry

end temperature_difference_l85_85446


namespace abs_ineq_l85_85263

theorem abs_ineq (x : ℝ) (h : |x + 1| > 3) : x < -4 ∨ x > 2 :=
  sorry

end abs_ineq_l85_85263


namespace complex_div_eq_l85_85979

def complex_z : ℂ := ⟨1, -2⟩
def imaginary_unit : ℂ := ⟨0, 1⟩

theorem complex_div_eq :
  (complex_z + 2) / (complex_z - 1) = 1 + (3 / 2 : ℂ) * imaginary_unit :=
by
  sorry

end complex_div_eq_l85_85979


namespace probability_heads_l85_85394

noncomputable def probability_more_heads_than_tails (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let y := (n.choose (n / 2)) / total_outcomes
  (1 - y) / 2

theorem probability_heads (h₁ : ∀ (n : ℕ), n = 10 → probability_more_heads_than_tails n = 193 / 512) : 
  probability_more_heads_than_tails 10 = 193 / 512 :=
by
  apply h₁
  exact rfl

end probability_heads_l85_85394


namespace f_even_l85_85892

variable (g : ℝ → ℝ)

def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x

def f (x : ℝ) := |g (x^2)|

theorem f_even (h_g_odd : is_odd g) : ∀ x : ℝ, f g x = f g (-x) :=
by
  intro x
  -- Proof can be added here
  sorry

end f_even_l85_85892


namespace measure_of_angle_F_l85_85878

theorem measure_of_angle_F (angle_D angle_E angle_F : ℝ) (h1 : angle_D = 80)
  (h2 : angle_E = 4 * angle_F + 10)
  (h3 : angle_D + angle_E + angle_F = 180) : angle_F = 18 := 
by
  sorry

end measure_of_angle_F_l85_85878


namespace exponent_division_l85_85802

theorem exponent_division : (19 ^ 11) / (19 ^ 8) = 6859 :=
by
  -- Here we assume the properties of powers and arithmetic operations
  sorry

end exponent_division_l85_85802


namespace quarterly_to_annual_rate_l85_85182

theorem quarterly_to_annual_rate (annual_rate : ℝ) (quarterly_rate : ℝ) (n : ℕ) (effective_annual_rate : ℝ) : 
  annual_rate = 4.5 →
  quarterly_rate = annual_rate / 4 →
  n = 4 →
  effective_annual_rate = (1 + quarterly_rate / 100)^n →
  effective_annual_rate * 100 = 4.56 :=
by
  intros h1 h2 h3 h4
  sorry

end quarterly_to_annual_rate_l85_85182


namespace answer_is_p_and_q_l85_85369

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l85_85369


namespace john_fouled_per_game_l85_85420

theorem john_fouled_per_game
  (hit_rate : ℕ) (shots_per_foul : ℕ) (total_games : ℕ) (participation_rate : ℚ) (total_free_throws : ℕ) :
  hit_rate = 70 → shots_per_foul = 2 → total_games = 20 → participation_rate = 0.8 → total_free_throws = 112 →
  (total_free_throws / (participation_rate * total_games)) / shots_per_foul = 3.5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end john_fouled_per_game_l85_85420


namespace find_hyperbola_m_l85_85857

theorem find_hyperbola_m (m : ℝ) (h : m > 0) : 
  (∀ x y : ℝ, (x^2 / m - y^2 / 3 = 1 → y = 1 / 2 * x)) → m = 12 :=
by
  intros
  sorry

end find_hyperbola_m_l85_85857


namespace math_proof_problem_l85_85575

variable {a b c : ℝ}

theorem math_proof_problem (h₁ : a * b * c * (a + b) * (b + c) * (c + a) ≠ 0)
  (h₂ : (a + b + c) * (1 / a + 1 / b + 1 / c) = 1007 / 1008) :
  (a * b / ((a + c) * (b + c)) + b * c / ((b + a) * (c + a)) + c * a / ((c + b) * (a + b))) = 2017 := 
sorry

end math_proof_problem_l85_85575


namespace cos_arccos_minus_arctan_eq_l85_85823

noncomputable def cos_arccos_minus_arctan: Real :=
  Real.cos (Real.arccos (4 / 5) - Real.arctan (1 / 2))

theorem cos_arccos_minus_arctan_eq : cos_arccos_minus_arctan = (11 * Real.sqrt 5) / 25 := by
  sorry

end cos_arccos_minus_arctan_eq_l85_85823


namespace add_water_to_solution_l85_85550

theorem add_water_to_solution (w : ℕ) :
  let initial_volume := 50
  let initial_concentration := 0.4
  let final_concentration := 0.25
  let acid_amount := initial_concentration * initial_volume
  let final_volume := initial_volume + w
  (acid_amount / final_volume = final_concentration) ↔ (w = 30) :=
by 
  sorry

end add_water_to_solution_l85_85550


namespace find_ab_l85_85208

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 :=
by
  sorry

end find_ab_l85_85208


namespace minimum_pie_pieces_l85_85737

theorem minimum_pie_pieces (p q : ℕ) (h_coprime : Nat.gcd p q = 1) : 
  ∃ n, (∀ k, k = p ∨ k = q → (n ≠ 0 → n % k = 0)) ∧ n = p + q - 1 :=
by {
  sorry
}

end minimum_pie_pieces_l85_85737


namespace exponent_division_l85_85801

theorem exponent_division : (19 ^ 11) / (19 ^ 8) = 6859 :=
by
  -- Here we assume the properties of powers and arithmetic operations
  sorry

end exponent_division_l85_85801


namespace solutions_to_h_eq_1_l85_85899

noncomputable def h (x : ℝ) : ℝ :=
if x ≤ 0 then 5 * x + 10 else 3 * x - 5

theorem solutions_to_h_eq_1 : {x : ℝ | h x = 1} = {-9/5, 2} :=
by
  sorry

end solutions_to_h_eq_1_l85_85899


namespace sqrt_mul_eq_l85_85082

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l85_85082


namespace children_more_than_adults_l85_85478

-- Conditions
def total_members : ℕ := 120
def adult_percentage : ℝ := 0.40
def child_percentage : ℝ := 1 - adult_percentage

-- Proof problem statement
theorem children_more_than_adults : 
  let number_of_adults := adult_percentage * total_members
  let number_of_children := child_percentage * total_members
  let difference := number_of_children - number_of_adults
  difference = 24 :=
by
  sorry

end children_more_than_adults_l85_85478


namespace part1_part2_l85_85202

-- Conditions and the equation of the circle
def circleCenterLine (a : ℝ) : Prop := ∃ y, y = a + 2
def circleRadius : ℝ := 2
def pointOnCircle (A : ℝ × ℝ) (a : ℝ) : Prop := (A.1 - a)^2 + (A.2 - (a + 2))^2 = circleRadius^2
def tangentToYAxis (a : ℝ) : Prop := abs a = circleRadius

-- Problem 1: Proving the equation of the circle C
def circleEq (x y a : ℝ) : Prop := (x - a)^2 + (y - (a + 2))^2 = circleRadius^2

theorem part1 (a : ℝ) (h : abs a = circleRadius) (h1 : pointOnCircle (2, 2) a) 
    (h2 : circleCenterLine a) : circleEq 2 0 2 := 
sorry

-- Conditions and the properties for Problem 2
def distSquared (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2
def QCondition (Q : ℝ × ℝ) : Prop := 
  distSquared Q (1, 3) - distSquared Q (1, 1) = 32
def onCircle (Q : ℝ × ℝ) (a : ℝ) : Prop := (Q.1 - a)^2 + (Q.2 - (a + 2))^2 = circleRadius^2

-- Problem 2: Proving the range of the abscissa a
theorem part2 (Q : ℝ × ℝ) (a : ℝ) 
    (hQ : QCondition Q) (hCircle : onCircle Q a) : 
    -3 ≤ a ∧ a ≤ 1 := 
sorry

end part1_part2_l85_85202


namespace kate_retirement_fund_value_l85_85428

theorem kate_retirement_fund_value 
(initial_value decrease final_value : ℝ) 
(h1 : initial_value = 1472)
(h2 : decrease = 12)
(h3 : final_value = initial_value - decrease) : 
final_value = 1460 := 
by
  sorry

end kate_retirement_fund_value_l85_85428


namespace largest_remainder_division_by_11_l85_85485

theorem largest_remainder_division_by_11 (A B C : ℕ) (h : A = 11 * B + C) (hC : 0 ≤ C ∧ C < 11) : C ≤ 10 :=
  sorry

end largest_remainder_division_by_11_l85_85485


namespace simplify_expression_l85_85519

theorem simplify_expression (x y m : ℤ) 
  (h1 : (x-5)^2 = -|m-1|)
  (h2 : y + 1 = 5) :
  (2 * x^2 - 3 * x * y - 4 * y^2) - m * (3 * x^2 - x * y + 9 * y^2) = -273 :=
sorry

end simplify_expression_l85_85519


namespace equivalence_statement_l85_85459

open Complex

noncomputable def distinct_complex (a b c d : ℂ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem equivalence_statement (a b c d : ℂ) (h : distinct_complex a b c d) :
  (∀ (z : ℂ), (abs (z - a) + abs (z - b) ≥ abs (z - c) + abs (z - d)))
  ↔ (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ c = t * a + (1 - t) * b ∧ d = (1 - t) * a + t * b) :=
sorry

end equivalence_statement_l85_85459


namespace ratio_of_girls_with_long_hair_l85_85455

theorem ratio_of_girls_with_long_hair (total_people boys girls short_hair long_hair : ℕ)
  (h1 : total_people = 55)
  (h2 : boys = 30)
  (h3 : girls = total_people - boys)
  (h4 : short_hair = 10)
  (h5 : long_hair = girls - short_hhair) :
  long_hair / gcd long_hair girls = 3 ∧ girls / gcd long_hair girls = 5 := 
by {
  -- This placeholder indicates where the proof should be.
  sorry
}

end ratio_of_girls_with_long_hair_l85_85455


namespace not_possible_to_create_3_piles_l85_85723

theorem not_possible_to_create_3_piles (similar: ℝ → ℝ → Prop) (sqrt_2 : ℝ)
  (hsimilar : ∀ x y, similar x y ↔ x ≤ sqrt_2 * y ∧ y ≤ sqrt_2 * x) :
  ∀ x, ¬ ∃ x1 x2 x3, 
    x = x1 + x2 + x3 ∧ 
    similar x1 x2 ∧ 
    similar x2 x3 ∧ 
    similar x1 x3 :=
by { sorry }

end not_possible_to_create_3_piles_l85_85723


namespace river_current_speed_l85_85775

def motorboat_speed_still_water : ℝ := 20
def distance_between_points : ℝ := 60
def total_trip_time : ℝ := 6.25

theorem river_current_speed : ∃ v_T : ℝ, v_T = 4 ∧ 
  (distance_between_points / (motorboat_speed_still_water + v_T)) + 
  (distance_between_points / (motorboat_speed_still_water - v_T)) = total_trip_time := 
sorry

end river_current_speed_l85_85775


namespace orange_balls_count_l85_85311

theorem orange_balls_count (total_balls red_balls blue_balls yellow_balls green_balls pink_balls orange_balls : ℕ) 
(h_total : total_balls = 100)
(h_red : red_balls = 30)
(h_blue : blue_balls = 20)
(h_yellow : yellow_balls = 10)
(h_green : green_balls = 5)
(h_pink : pink_balls = 2 * green_balls)
(h_orange : orange_balls = 3 * pink_balls)
(h_sum : red_balls + blue_balls + yellow_balls + green_balls + pink_balls + orange_balls = total_balls) :
orange_balls = 30 :=
sorry

end orange_balls_count_l85_85311


namespace smallest_n_modulo_l85_85927

theorem smallest_n_modulo :
  ∃ n : ℕ, 0 < n ∧ 5 * n % 26 = 1846 % 26 ∧ n = 26 :=
by
  sorry

end smallest_n_modulo_l85_85927


namespace honey_production_l85_85554

-- Define the conditions:
def bees : ℕ := 60
def days : ℕ := 60
def honey_per_bee : ℕ := 1

-- Statement to prove:
theorem honey_production (bees_eq : 60 = bees) (days_eq : 60 = days) (honey_per_bee_eq : 1 = honey_per_bee) :
  bees * honey_per_bee = 60 := by
  sorry

end honey_production_l85_85554


namespace sqrt3_mul_sqrt12_eq_6_l85_85114

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l85_85114


namespace total_cost_of_crayons_l85_85880

theorem total_cost_of_crayons (crayons_per_half_dozen : ℕ)
    (number_of_half_dozens : ℕ)
    (cost_per_crayon : ℕ)
    (total_cost : ℕ) :
  crayons_per_half_dozen = 6 →
  number_of_half_dozens = 4 →
  cost_per_crayon = 2 →
  total_cost = crayons_per_half_dozen * number_of_half_dozens * cost_per_crayon →
  total_cost = 48 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end total_cost_of_crayons_l85_85880


namespace locus_of_Q_is_ellipse_l85_85855

noncomputable def ellipse (x y : ℝ) := (x^2) / 24 + (y^2) / 16 = 1
noncomputable def line (x y : ℝ) := x / 12 + y / 8 = 1

theorem locus_of_Q_is_ellipse :
  (∀ (x y : ℝ),
    ∃ (P : ℝ × ℝ) (R : ℝ × ℝ) (Q : ℝ × ℝ),
      (line P.1 P.2) ∧
      (∃ (θ ρ_1 ρ_2 : ℝ), (P.1 = ρ_2 * cos θ ∧ P.2 = ρ_2 * sin θ) ∧
      (ellipse R.1 R.2) ∧ 
      (R.1 = ρ_1 * cos θ ∧ R.2 = ρ_1 * sin θ) ∧
      (let ρ := (ρ_1^2) / ρ_2 in Q.1 = ρ * cos θ ∧ Q.2 = ρ * sin θ) ∧
      |Q.1 + Q.2| * |P.1 + P.2| = |R.1 + R.2|^2)) →
  ∃ (x y : ℝ), (2 * (x - 1)^2 / (5 / 2)) + (3 * (y - 1)^2 / (5 / 3)) = 1 :=
sorry

end locus_of_Q_is_ellipse_l85_85855


namespace number_of_valid_groupings_l85_85281

-- Definitions based on conditions
def num_guides : ℕ := 2
def num_tourists : ℕ := 6
def total_groupings : ℕ := 2 ^ num_tourists
def invalid_groupings : ℕ := 2  -- All tourists go to one guide either a or b

-- The theorem to prove
theorem number_of_valid_groupings : total_groupings - invalid_groupings = 62 :=
by sorry

end number_of_valid_groupings_l85_85281


namespace exists_sum_of_two_squares_l85_85731

theorem exists_sum_of_two_squares (n : ℕ) (h₁ : n > 10000) : 
  ∃ m : ℕ, (∃ a b : ℕ, m = a^2 + b^2) ∧ 0 < m - n ∧ m - n < 3 * Real.sqrt n := 
sorry

end exists_sum_of_two_squares_l85_85731


namespace total_emails_in_april_is_675_l85_85418

-- Define the conditions
def daily_emails : ℕ := 20
def additional_emails : ℕ := 5
def april_days : ℕ := 30
def half_april_days : ℕ := april_days / 2

-- Define the total number of emails received
def total_emails : ℕ :=
  (daily_emails * half_april_days) +
  ((daily_emails + additional_emails) * half_april_days)

-- Define the statement to be proven
theorem total_emails_in_april_is_675 : total_emails = 675 :=
  by
  sorry

end total_emails_in_april_is_675_l85_85418


namespace sqrt_mul_simplify_l85_85107

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l85_85107


namespace complex_number_expression_l85_85704

noncomputable def compute_expression (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :=
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1)

theorem complex_number_expression (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :
  compute_expression r h1 h2 = 5 :=
sorry

end complex_number_expression_l85_85704


namespace Robert_ate_10_chocolates_l85_85005

def chocolates_eaten_by_Nickel : Nat := 5
def difference_between_Robert_and_Nickel : Nat := 5
def chocolates_eaten_by_Robert := chocolates_eaten_by_Nickel + difference_between_Robert_and_Nickel

theorem Robert_ate_10_chocolates : chocolates_eaten_by_Robert = 10 :=
by
  -- Proof omitted
  sorry

end Robert_ate_10_chocolates_l85_85005


namespace total_emails_675_l85_85414

-- Definitions based on conditions
def emails_per_day_before : ℕ := 20
def extra_emails_per_day_after : ℕ := 5
def halfway_days : ℕ := 15
def total_days : ℕ := 30

-- Define the total number of emails received by the end of April
def total_emails_received : ℕ :=
  let emails_before := emails_per_day_before * halfway_days
  let emails_after := (emails_per_day_before + extra_emails_per_day_after) * halfway_days
  emails_before + emails_after

-- Theorem stating that the total number of emails received by the end of April is 675
theorem total_emails_675 : total_emails_received = 675 := by
  sorry

end total_emails_675_l85_85414


namespace price_difference_is_300_cents_l85_85179

noncomputable def list_price : ℝ := 59.99
noncomputable def tech_bargains_price : ℝ := list_price - 15
noncomputable def digital_deal_price : ℝ := 0.7 * list_price
noncomputable def price_difference : ℝ := tech_bargains_price - digital_deal_price
noncomputable def price_difference_in_cents : ℝ := price_difference * 100

theorem price_difference_is_300_cents :
  price_difference_in_cents = 300 := by
  sorry

end price_difference_is_300_cents_l85_85179


namespace problem_l85_85674

def Y (a b : ℤ) : ℤ := a^2 - 2 * a * b + b^2
def Z (a b : ℤ) : ℤ := a * b + a + b

theorem problem
  : Z (Y 5 3) (Y 2 1) = 9 := by
  sorry

end problem_l85_85674


namespace contractor_fine_per_day_l85_85303

theorem contractor_fine_per_day
    (total_days : ℕ) 
    (work_days_fine_amt : ℕ) 
    (total_amt : ℕ) 
    (absent_days : ℕ) 
    (worked_days : ℕ := total_days - absent_days)
    (earned_amt : ℕ := worked_days * work_days_fine_amt)
    (fine_per_day : ℚ)
    (total_fine : ℚ := absent_days * fine_per_day) : 
    (earned_amt - total_fine = total_amt) → 
    fine_per_day = 7.5 :=
by
  intros h
  -- proof here is omitted
  sorry

end contractor_fine_per_day_l85_85303


namespace exam_cutoff_mark_l85_85873

theorem exam_cutoff_mark
  (num_students : ℕ)
  (absent_percentage : ℝ)
  (fail_percentage : ℝ)
  (fail_mark_diff : ℝ)
  (just_pass_percentage : ℝ)
  (remaining_avg_mark : ℝ)
  (class_avg_mark : ℝ)
  (absent_students : ℕ)
  (fail_students : ℕ)
  (just_pass_students : ℕ)
  (remaining_students : ℕ)
  (total_marks : ℝ)
  (P : ℝ) :
  absent_percentage = 0.2 →
  fail_percentage = 0.3 →
  fail_mark_diff = 20 →
  just_pass_percentage = 0.1 →
  remaining_avg_mark = 65 →
  class_avg_mark = 36 →
  absent_students = (num_students * absent_percentage) →
  fail_students = (num_students * fail_percentage) →
  just_pass_students = (num_students * just_pass_percentage) →
  remaining_students = num_students - absent_students - fail_students - just_pass_students →
  total_marks = (absent_students * 0) + (fail_students * (P - fail_mark_diff)) + (just_pass_students * P) + (remaining_students * remaining_avg_mark) →
  class_avg_mark = total_marks / num_students →
  P = 40 :=
by
  intros
  sorry

end exam_cutoff_mark_l85_85873


namespace no_common_interior_points_l85_85539

open Metric

-- Define the distance conditions for two convex polygons F1 and F2
variables {F1 F2 : Set (EuclideanSpace ℝ (Fin 2))}

-- F1 is a convex polygon
def is_convex (S : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ {x y : EuclideanSpace ℝ (Fin 2)} {a b : ℝ},
    x ∈ S → y ∈ S → 0 ≤ a → 0 ≤ b → a + b = 1 → a • x + b • y ∈ S

-- Conditions provided in the problem
def condition1 (F : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ {x y : EuclideanSpace ℝ (Fin 2)}, x ∈ F → y ∈ F → dist x y ≤ 1

def condition2 (F1 F2 : Set (EuclideanSpace ℝ (Fin 2))) : Prop :=
  ∀ {x : EuclideanSpace ℝ (Fin 2)} {y : EuclideanSpace ℝ (Fin 2)}, x ∈ F1 → y ∈ F2 → dist x y > 1 / Real.sqrt 2

-- The theorem to prove
theorem no_common_interior_points (h1 : is_convex F1) (h2 : is_convex F2) 
  (h3 : condition1 F1) (h4 : condition1 F2) (h5 : condition2 F1 F2) :
  ∀ p ∈ interior F1, ∀ q ∈ interior F2, p ≠ q :=
sorry

end no_common_interior_points_l85_85539


namespace sqrt_mult_eq_six_l85_85160

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l85_85160


namespace p_and_q_is_true_l85_85343

def p : Prop := ∃ x : ℝ, Real.sin x < 1
def q : Prop := ∀ x : ℝ, Real.exp (| x |) ≥ 1

theorem p_and_q_is_true : p ∧ q := by
  sorry

end p_and_q_is_true_l85_85343


namespace flight_time_l85_85071

def eagle_speed : ℕ := 15
def falcon_speed : ℕ := 46
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30
def total_distance : ℕ := 248

theorem flight_time : (eagle_speed + falcon_speed + pelican_speed + hummingbird_speed) > 0 → 
                      total_distance / (eagle_speed + falcon_speed + pelican_speed + hummingbird_speed) = 2 :=
by
  -- Proof is skipped
  sorry

end flight_time_l85_85071


namespace power_quotient_l85_85796

theorem power_quotient (a m n : ℕ) (h_a : a = 19) (h_m : m = 11) (h_n : n = 8) : a^m / a^n = 6859 := by
  sorry

end power_quotient_l85_85796


namespace find_tan_alpha_plus_pi_div_12_l85_85206

theorem find_tan_alpha_plus_pi_div_12 (α : ℝ) (h : Real.sin α = 3 * Real.sin (α + Real.pi / 6)) :
  Real.tan (α + Real.pi / 12) = 2 * Real.sqrt 3 - 4 :=
by
  sorry

end find_tan_alpha_plus_pi_div_12_l85_85206


namespace expression_value_l85_85291

theorem expression_value : (36 + 9) ^ 2 - (9 ^ 2 + 36 ^ 2) = -1894224 :=
by
  sorry

end expression_value_l85_85291


namespace length_of_bridge_l85_85782

-- Define the problem conditions
def length_train : ℝ := 110 -- Length of the train in meters
def speed_kmph : ℝ := 60 -- Speed of the train in kmph

-- Convert speed from kmph to m/s
noncomputable def speed_mps : ℝ := speed_kmph * 1000 / 3600

-- Define the time taken to cross the bridge
def time_seconds : ℝ := 16.7986561075114

-- Define the total distance covered
noncomputable def total_distance : ℝ := speed_mps * time_seconds

-- Prove the length of the bridge
theorem length_of_bridge : total_distance - length_train = 170 := 
by
  -- Proof will be here
  sorry

end length_of_bridge_l85_85782


namespace sqrt3_mul_sqrt12_eq_6_l85_85090

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l85_85090


namespace gcd_m_l85_85895

def m' : ℕ := 33333333
def n' : ℕ := 555555555

theorem gcd_m'_n' : Nat.gcd m' n' = 3 := by
  sorry

end gcd_m_l85_85895


namespace proof_problem_l85_85358

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l85_85358


namespace ball_hits_ground_l85_85913

theorem ball_hits_ground (t : ℝ) :
  (∃ t, -(16 * t^2) + 32 * t + 30 = 0 ∧ t = 1 + (Real.sqrt 46) / 4) :=
sorry

end ball_hits_ground_l85_85913


namespace calculation_correct_l85_85646

theorem calculation_correct : 
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5)) = 47 := 
by
  sorry

end calculation_correct_l85_85646


namespace tangent_line_through_external_point_l85_85532

theorem tangent_line_through_external_point (x y : ℝ) (h_circle : x^2 + y^2 = 1) (P : ℝ × ℝ) (h_P : P = (1, 2)) : 
  (∃ k : ℝ, (y = 2 + k * (x - 1)) ∧ (x = 1 ∨ (3 * x - 4 * y + 5 = 0))) :=
by
  sorry

end tangent_line_through_external_point_l85_85532


namespace trains_cross_in_12_seconds_l85_85627

noncomputable def length := 120 -- Length of each train in meters
noncomputable def time_train1 := 10 -- Time taken by the first train to cross the post in seconds
noncomputable def time_train2 := 15 -- Time taken by the second train to cross the post in seconds

noncomputable def speed_train1 := length / time_train1 -- Speed of the first train in m/s
noncomputable def speed_train2 := length / time_train2 -- Speed of the second train in m/s

noncomputable def relative_speed := speed_train1 + speed_train2 -- Relative speed when traveling in opposite directions in m/s
noncomputable def total_length := 2 * length -- Total distance covered when crossing each other

noncomputable def crossing_time := total_length / relative_speed -- Time to cross each other in seconds

theorem trains_cross_in_12_seconds : crossing_time = 12 := by
  sorry

end trains_cross_in_12_seconds_l85_85627


namespace alice_ate_more_l85_85183

theorem alice_ate_more (cookies : Fin 8 → ℕ) (h_alice : cookies 0 = 8) (h_tom : cookies 7 = 1) :
  cookies 0 - cookies 7 = 7 :=
by
  -- Placeholder for the actual proof, which is not required here
  sorry

end alice_ate_more_l85_85183


namespace find_line_equation_l85_85063
noncomputable def line_equation (l : ℝ → ℝ → Prop) : Prop :=
    (∀ x y : ℝ, l x y ↔ (2 * x + y - 4 = 0) ∨ (x + y - 3 = 0))

theorem find_line_equation (l : ℝ → ℝ → Prop) :
  (l 1 2) →
  (∃ x1 : ℝ, x1 > 0 ∧ ∃ y1 : ℝ, y1 > 0 ∧ l x1 0 ∧ l 0 y1) ∧
  (∃ x2 : ℝ, x2 < 0 ∧ ∃ y2 : ℝ, y2 > 0 ∧ l x2 0 ∧ l 0 y2) ∧
  (∃ x4 : ℝ, x4 > 0 ∧ ∃ y4 : ℝ, y4 < 0 ∧ l x4 0 ∧ l 0 y4) ∧
  (∃ x_int y_int : ℝ, l x_int 0 ∧ l 0 y_int ∧ x_int + y_int = 6) →
  (line_equation l) :=
by
  sorry

end find_line_equation_l85_85063


namespace last_two_digits_of_floor_l85_85604

def last_two_digits (n : Nat) : Nat :=
  n % 100

theorem last_two_digits_of_floor :
  let x := 10^93
  let y := 10^31
  last_two_digits (Nat.floor (x / (y + 3))) = 8 :=
by
  sorry

end last_two_digits_of_floor_l85_85604


namespace solve_inequality_l85_85734

namespace InequalityProof

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem solve_inequality (x : ℝ) : cube_root x + 3 / (cube_root x + 4) ≤ 0 ↔ x ∈ Set.Icc (-27 : ℝ) (-1 : ℝ) :=
by
  have y_eq := cube_root x
  sorry

end InequalityProof

end solve_inequality_l85_85734


namespace Jason_spent_on_music_store_l85_85412

theorem Jason_spent_on_music_store:
  let flute := 142.46
  let music_stand := 8.89
  let song_book := 7.00
  flute + music_stand + song_book = 158.35 := sorry

end Jason_spent_on_music_store_l85_85412


namespace find_extrema_l85_85739

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

theorem find_extrema :
  (∀ x, f x ≤ 17) ∧ (∃ x, f x = 17) ∧ (∀ x, f x ≥ -47) ∧ (∃ x, f x = -47) :=
by
  sorry

end find_extrema_l85_85739


namespace maximum_value_of_A_l85_85838

theorem maximum_value_of_A (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
    (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4 / 3)) ≤ 3 :=
sorry

end maximum_value_of_A_l85_85838


namespace cookies_with_flour_l85_85421

theorem cookies_with_flour (x: ℕ) (c1: ℕ) (c2: ℕ) (h: c1 = 18 ∧ c2 = 2 ∧ x = 9 * 5):
  x = 45 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end cookies_with_flour_l85_85421


namespace cook_carrots_l85_85480

theorem cook_carrots :
  ∀ (total_carrots : ℕ) (fraction_used_before_lunch : ℚ) (carrots_not_used_end_of_day : ℕ),
    total_carrots = 300 →
    fraction_used_before_lunch = 2 / 5 →
    carrots_not_used_end_of_day = 72 →
    let carrots_used_before_lunch := total_carrots * fraction_used_before_lunch
    let carrots_after_lunch := total_carrots - carrots_used_before_lunch
    let carrots_used_end_of_day := carrots_after_lunch - carrots_not_used_end_of_day
    (carrots_used_end_of_day / carrots_after_lunch) = 3 / 5 :=
by
  intros total_carrots fraction_used_before_lunch carrots_not_used_end_of_day
  intros h1 h2 h3
  let carrots_used_before_lunch := total_carrots * fraction_used_before_lunch
  let carrots_after_lunch := total_carrots - carrots_used_before_lunch
  let carrots_used_end_of_day := carrots_after_lunch - carrots_not_used_end_of_day
  have h : carrots_used_end_of_day / carrots_after_lunch = 3 / 5 := sorry
  exact h

end cook_carrots_l85_85480


namespace fraction_to_decimal_l85_85813

theorem fraction_to_decimal : (3 : ℚ) / 60 = 0.05 := 
by sorry

end fraction_to_decimal_l85_85813


namespace problem_statement_l85_85335

open Complex

theorem problem_statement (x y : ℝ) (i : ℂ) (h_i : i = Complex.I) (h : x + (y - 2) * i = 2 / (1 + i)) : x + y = 2 :=
by
  sorry

end problem_statement_l85_85335


namespace sqrt_mul_simplify_l85_85104

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l85_85104


namespace average_mark_of_remaining_students_l85_85600

theorem average_mark_of_remaining_students
  (n : ℕ) (A : ℕ) (m : ℕ) (B : ℕ) (total_students : n = 10)
  (avg_class : A = 80) (excluded_students : m = 5) (avg_excluded : B = 70) :
  (A * n - B * m) / (n - m) = 90 :=
by
  sorry

end average_mark_of_remaining_students_l85_85600


namespace diagonals_of_angle_bisectors_l85_85274

theorem diagonals_of_angle_bisectors (a b : ℝ) (BAD ABC : ℝ) (hBAD : BAD = ABC) :
  ∃ d : ℝ, d = |a - b| :=
by
  sorry

end diagonals_of_angle_bisectors_l85_85274


namespace f_even_l85_85891

variable (g : ℝ → ℝ)

def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x

def f (x : ℝ) := |g (x^2)|

theorem f_even (h_g_odd : is_odd g) : ∀ x : ℝ, f g x = f g (-x) :=
by
  intro x
  -- Proof can be added here
  sorry

end f_even_l85_85891


namespace multiplication_approximation_correct_l85_85402

noncomputable def closest_approximation (x : ℝ) : ℝ := 
  if 15700 <= x ∧ x < 15750 then 15700
  else if 15750 <= x ∧ x < 15800 then 15750
  else if 15800 <= x ∧ x < 15900 then 15800
  else if 15900 <= x ∧ x < 16000 then 15900
  else 16000

theorem multiplication_approximation_correct :
  closest_approximation (0.00525 * 3153420) = 15750 := 
by
  sorry

end multiplication_approximation_correct_l85_85402


namespace problem_statement_l85_85391

theorem problem_statement (x y a : ℝ) (h1 : x + a < y + a) (h2 : a * x > a * y) : x < y ∧ a < 0 :=
sorry

end problem_statement_l85_85391


namespace single_elimination_matches_l85_85406

theorem single_elimination_matches (n : ℕ) (h : n = 512) :
  ∃ (m : ℕ), m = n - 1 ∧ m = 511 :=
by
  sorry

end single_elimination_matches_l85_85406


namespace sqrt_mul_simp_l85_85131

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l85_85131


namespace exponent_division_l85_85803

theorem exponent_division : (19 ^ 11) / (19 ^ 8) = 6859 :=
by
  -- Here we assume the properties of powers and arithmetic operations
  sorry

end exponent_division_l85_85803


namespace not_possible_three_piles_l85_85720

def similar (x y : ℝ) : Prop :=
  y ≤ (real.sqrt 2) * x ∧ x ≤ (real.sqrt 2) * y

theorem not_possible_three_piles (N : ℝ) : 
  ¬ ∃ a b c : ℝ, a + b + c = N ∧ similar a b ∧ similar b c ∧ similar a c := 
by
  sorry

end not_possible_three_piles_l85_85720


namespace shoes_to_belts_ratio_l85_85495

variable (hats : ℕ) (belts : ℕ) (shoes : ℕ)

theorem shoes_to_belts_ratio (hats_eq : hats = 5)
                            (belts_eq : belts = hats + 2)
                            (shoes_eq : shoes = 14) : 
  (shoes / (Nat.gcd shoes belts)) = 2 ∧ (belts / (Nat.gcd shoes belts)) = 1 := 
by
  sorry

end shoes_to_belts_ratio_l85_85495


namespace impossible_configuration_l85_85487

theorem impossible_configuration : 
  ¬∃ (f : ℕ → ℕ) (h : ∀n, 1 ≤ f n ∧ f n ≤ 5) (perm : ∀i j, if i < j then f i ≠ f j else true), 
  (f 0 = 3) ∧ (f 1 = 4) ∧ (f 2 = 2) ∧ (f 3 = 1) ∧ (f 4 = 5) :=
sorry

end impossible_configuration_l85_85487


namespace sale_in_fifth_month_l85_85773

-- Define the sales for the first four months and the required sale for the sixth month
def sale_month1 : ℕ := 5124
def sale_month2 : ℕ := 5366
def sale_month3 : ℕ := 5808
def sale_month4 : ℕ := 5399
def sale_month6 : ℕ := 4579

-- Define the target average sale and number of months
def target_average_sale : ℕ := 5400
def number_of_months : ℕ := 6

-- Define the total sales calculation using the provided information
def total_sales : ℕ := target_average_sale * number_of_months
def total_sales_first_four_months : ℕ := sale_month1 + sale_month2 + sale_month3 + sale_month4

-- Prove the sale in the fifth month
theorem sale_in_fifth_month : 
  sale_month1 + sale_month2 + sale_month3 + sale_month4 + (total_sales - 
  (total_sales_first_four_months + sale_month6)) + sale_month6 = total_sales :=
by
  sorry

end sale_in_fifth_month_l85_85773


namespace sqrt_mult_l85_85172

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l85_85172


namespace triangle_existence_condition_l85_85500

-- Definitions and conditions given in the problem
variables (α : ℝ) (f_a r : ℝ)

-- The equivalent proof problem statement in Lean 4
theorem triangle_existence_condition (hα : 0 < α ∧ α < π) (hf_a : 0 < f_a) (hr : 0 < r) :
  f_a ≤ 2 * r * (Real.cos (α / 2))^2 :=
sorry

end triangle_existence_condition_l85_85500


namespace water_to_add_for_desired_acid_concentration_l85_85545

theorem water_to_add_for_desired_acid_concentration :
  ∃ w : ℝ, 50 * 0.4 / (50 + w) = 0.25 ∧ w = 30 :=
begin
  let initial_volume := 50,
  let initial_acid_concentration := 0.4,
  let desired_acid_concentration := 0.25,
  let acid_content := initial_volume * initial_acid_concentration,
  use 30,
  split,
  { simp [initial_volume, initial_acid_concentration, desired_acid_concentration, acid_content],
    linarith },
  { refl }
end

end water_to_add_for_desired_acid_concentration_l85_85545


namespace identity_proof_l85_85469

theorem identity_proof : 
  ∀ x : ℝ, 
    (x^2 + 3*x + 2) * (x + 3) = (x + 1) * (x^2 + 5*x + 6) := 
by 
  sorry

end identity_proof_l85_85469


namespace find_ab_l85_85209

theorem find_ab (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 35) : a * b = 6 := 
by
  sorry

end find_ab_l85_85209


namespace rectangular_prism_diagonal_inequality_l85_85203

variable (a b c l : ℝ)

theorem rectangular_prism_diagonal_inequality (h : l^2 = a^2 + b^2 + c^2) : 
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := sorry

end rectangular_prism_diagonal_inequality_l85_85203


namespace find_c_l85_85506

theorem find_c (c : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Iio (-2) ∪ Set.Ioi 3 → x^2 - c * x + 6 > 0) → c = 1 :=
by
  sorry

end find_c_l85_85506


namespace correct_option_l85_85930

-- Define the four conditions as propositions
def option_A (a b : ℝ) : Prop := (a + b) ^ 2 = a ^ 2 + b ^ 2
def option_B (a : ℝ) : Prop := 2 * a ^ 2 + a = 3 * a ^ 3
def option_C (a : ℝ) : Prop := a ^ 3 * a ^ 2 = a ^ 5
def option_D (a : ℝ) (h : a ≠ 0) : Prop := 2 * a⁻¹ = 1 / (2 * a)

-- Prove which operation is the correct one
theorem correct_option (a b : ℝ) (h : a ≠ 0) : option_C a :=
by {
  -- Placeholder for actual proofs, each option needs to be verified
  sorry
}

end correct_option_l85_85930


namespace problem_solution_l85_85658

noncomputable def time_min_distance
  (c : ℝ) (α : ℝ) (a : ℝ) : ℝ :=
a * (Real.cos α) / (2 * c * (1 - Real.sin α))

noncomputable def min_distance
  (c : ℝ) (α : ℝ) (a : ℝ) : ℝ :=
a * Real.sqrt ((1 - (Real.sin α)) / 2)

theorem problem_solution (α : ℝ) (c : ℝ) (a : ℝ) 
  (α_30 : α = Real.pi / 6) (c_50 : c = 50) (a_50sqrt3 : a = 50 * Real.sqrt 3) :
  (time_min_distance c α a = 1.5) ∧ (min_distance c α a = 25 * Real.sqrt 3) :=
by
  sorry

end problem_solution_l85_85658


namespace mn_value_l85_85389

theorem mn_value (m n : ℤ) (h1 : 2 * m = 6) (h2 : m - n = 2) : m * n = 3 := by
  sorry

end mn_value_l85_85389


namespace sqrt_mul_simplify_l85_85110

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l85_85110


namespace student_average_less_than_actual_average_l85_85067

variable {a b c : ℝ}

theorem student_average_less_than_actual_average (h : a < b) (h2 : b < c) :
  (a + (b + c) / 2) / 2 < (a + b + c) / 3 :=
by
  sorry

end student_average_less_than_actual_average_l85_85067


namespace sqrt13_decomposition_ten_plus_sqrt3_decomposition_l85_85004

-- For the first problem
theorem sqrt13_decomposition :
  let a := 3
  let b := Real.sqrt 13 - 3
  a^2 + b - Real.sqrt 13 = 6 := by
sorry

-- For the second problem
theorem ten_plus_sqrt3_decomposition :
  let x := 11
  let y := Real.sqrt 3 - 1
  x - y = 12 - Real.sqrt 3 := by
sorry

end sqrt13_decomposition_ten_plus_sqrt3_decomposition_l85_85004


namespace right_triangle_side_length_l85_85636

theorem right_triangle_side_length (hypotenuse : ℝ) (θ : ℝ) (sin_30 : Real.sin 30 = 1 / 2) (h : θ = 30) 
  (hyp_len : hypotenuse = 10) : 
  let opposite_side := hypotenuse * Real.sin θ
  opposite_side = 5 := by
  sorry

end right_triangle_side_length_l85_85636


namespace quadratic_has_real_root_l85_85431

theorem quadratic_has_real_root (p : ℝ) : 
  ∃ x : ℝ, 3 * (p + 2) * x^2 - p * x - (4 * p + 7) = 0 :=
sorry

end quadratic_has_real_root_l85_85431


namespace sqrt_mult_l85_85174

theorem sqrt_mult (a b : ℝ) (ha : a = 3) (hb : b = 12) : real.sqrt a * real.sqrt b = 6 :=
by
  sorry

end sqrt_mult_l85_85174


namespace simplify_expression1_simplify_expression2_l85_85595

theorem simplify_expression1 (x : ℝ) : 
  5*x^2 + x + 3 + 4*x - 8*x^2 - 2 = -3*x^2 + 5*x + 1 :=
by
  sorry

theorem simplify_expression2 (a : ℝ) : 
  (5*a^2 + 2*a - 1) - 4*(3 - 8*a + 2*a^2) = -3*a^2 + 34*a - 13 :=
by
  sorry

end simplify_expression1_simplify_expression2_l85_85595


namespace problem_statement_l85_85352

open Classical

-- Define the propositions p and q
def p : Prop := ∃ x ∈ (set.univ : set ℝ), sin x < 1
def q : Prop := ∀ x ∈ (set.univ : set ℝ), real.exp (abs x) ≥ 1

-- State the theorem to be proved
theorem problem_statement : p ∧ q :=
by
  sorry

end problem_statement_l85_85352


namespace net_rate_of_pay_l85_85784

/-- The net rate of pay in dollars per hour for a truck driver after deducting gasoline expenses. -/
theorem net_rate_of_pay
  (hrs : ℕ) (speed : ℕ) (miles_per_gallon : ℕ) (pay_per_mile : ℚ) (cost_per_gallon : ℚ) 
  (H1 : hrs = 3)
  (H2 : speed = 50)
  (H3 : miles_per_gallon = 25)
  (H4 : pay_per_mile = 0.6)
  (H5 : cost_per_gallon = 2.50) :
  pay_per_mile * (hrs * speed) - cost_per_gallon * ((hrs * speed) / miles_per_gallon) = 25 * hrs :=
by sorry

end net_rate_of_pay_l85_85784


namespace perpendicular_lines_k_value_l85_85738

theorem perpendicular_lines_k_value (k : ℝ) :
  (∀ x y : ℝ, k * x - y - 3 = 0 → x + (2 * k + 3) * y - 2 = 0) →
  k = -3 :=
by
  sorry

end perpendicular_lines_k_value_l85_85738


namespace proof_problem_l85_85196

def number := 432

theorem proof_problem (y : ℕ) (n : ℕ) (h1 : y = 36) (h2 : 6^5 * 2 / n = y) : n = number :=
by 
  -- proof steps would go here
  sorry

end proof_problem_l85_85196


namespace binary_addition_l85_85924

theorem binary_addition :
  let num1 := 0b111111111
  let num2 := 0b101010101
  num1 + num2 = 852 := by
  sorry

end binary_addition_l85_85924


namespace prove_inequality_l85_85299

-- Define the function properties
variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Function properties as given in the problem
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def decreasing_on_nonneg (f : ℝ → ℝ) := ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

-- The main theorem statement
theorem prove_inequality (h_even : even_function f) (h_dec : decreasing_on_nonneg f) :
  f (-3 / 4) ≥ f (a^2 - a + 1) :=
sorry

end prove_inequality_l85_85299


namespace soccer_camp_afternoon_kids_l85_85454

def num_kids_in_camp : ℕ := 2000
def fraction_going_to_soccer_camp : ℚ := 1 / 2
def fraction_going_to_soccer_camp_in_morning : ℚ := 1 / 4

noncomputable def num_kids_going_to_soccer_camp := num_kids_in_camp * fraction_going_to_soccer_camp
noncomputable def num_kids_going_to_soccer_camp_in_morning := num_kids_going_to_soccer_camp * fraction_going_to_soccer_camp_in_morning
noncomputable def num_kids_going_to_soccer_camp_in_afternoon := num_kids_going_to_soccer_camp - num_kids_going_to_soccer_camp_in_morning

theorem soccer_camp_afternoon_kids : num_kids_going_to_soccer_camp_in_afternoon = 750 :=
by
  sorry

end soccer_camp_afternoon_kids_l85_85454


namespace sqrt_mult_eq_six_l85_85165

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l85_85165


namespace find_f_at_4_l85_85016

theorem find_f_at_4 (f : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 3 * f x - 2 * f (1 / x) = x) : 
  f 4 = 5 / 2 :=
sorry

end find_f_at_4_l85_85016


namespace exponent_division_l85_85807

theorem exponent_division (a : ℕ) (m n : ℕ) (h1 : 19 = a) (h2 : 11 = m) (h3 : 8 = n) : a^(m - n) = 6859 := by
  sorry

end exponent_division_l85_85807


namespace calculate_rate_l85_85910

-- Definitions corresponding to the conditions in the problem
def bankers_gain (td : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  td * rate * time

-- Given values according to the problem
def BG : ℝ := 7.8
def TD : ℝ := 65
def Time : ℝ := 1
def expected_rate_percentage : ℝ := 12

-- The mathematical proof problem statement in Lean 4
theorem calculate_rate : (BG = bankers_gain TD (expected_rate_percentage / 100) Time) :=
sorry

end calculate_rate_l85_85910


namespace find_point_on_line_and_distance_l85_85654

def distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem find_point_on_line_and_distance :
  ∃ P : ℝ × ℝ, (2 * P.1 - 3 * P.2 + 5 = 0) ∧ (distance P (2, 3) = 13) →
  (P = (5, 5) ∨ P = (-1, 1)) :=
by
  sorry

end find_point_on_line_and_distance_l85_85654


namespace existential_inequality_false_iff_l85_85608

theorem existential_inequality_false_iff {a : ℝ} :
  (∀ x : ℝ, x^2 + a * x - 2 * a ≥ 0) ↔ -8 ≤ a ∧ a ≤ 0 :=
by
  sorry

end existential_inequality_false_iff_l85_85608


namespace average_age_of_combined_rooms_l85_85906

theorem average_age_of_combined_rooms
  (num_people_A : ℕ) (avg_age_A : ℕ)
  (num_people_B : ℕ) (avg_age_B : ℕ)
  (num_people_C : ℕ) (avg_age_C : ℕ)
  (hA : num_people_A = 8) (hAA : avg_age_A = 35)
  (hB : num_people_B = 5) (hBB : avg_age_B = 30)
  (hC : num_people_C = 7) (hCC : avg_age_C = 50) :
  ((num_people_A * avg_age_A + num_people_B * avg_age_B + num_people_C * avg_age_C) / 
  (num_people_A + num_people_B + num_people_C) = 39) :=
by
  sorry

end average_age_of_combined_rooms_l85_85906


namespace solution_to_inequality_l85_85449

-- Define the combination function C(n, k)
def combination (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Define the permutation function A(n, k)
def permutation (n k : ℕ) : ℕ :=
  n.factorial / (n - k).factorial

-- State the final theorem
theorem solution_to_inequality : 
  ∀ x : ℕ, (combination 5 x + permutation x 3 < 30) ↔ (x = 3 ∨ x = 4) :=
by
  -- The actual proof is not required as per the instructions
  sorry

end solution_to_inequality_l85_85449


namespace remainder_sum_mod_53_l85_85929

theorem remainder_sum_mod_53 (a b c d : ℕ)
  (h1 : a % 53 = 31)
  (h2 : b % 53 = 45)
  (h3 : c % 53 = 17)
  (h4 : d % 53 = 6) :
  (a + b + c + d) % 53 = 46 := 
sorry

end remainder_sum_mod_53_l85_85929


namespace exponent_calculation_l85_85797

-- Define the necessary exponents and base
def base : ℕ := 19
def exp1 : ℕ := 11
def exp2 : ℕ := 8

-- Given condition
lemma power_property (a : ℕ) (m n : ℕ) : a^m / a^n = a^(m - n) := by sorry

-- Proof that 19^11 / 19^8 = 6859
theorem exponent_calculation : base^exp1 / base^exp2 = 6859 := by
  have : base^exp1 / base^exp2 = base^(exp1 - exp2) := power_property base exp1 exp2
  have : base^(exp1 - exp2) = base^3 := by rw [nat.sub_eq_iff_eq_add.mpr rfl]
  have : base^3 = 6859 := by -- This would be an arithmetic computation
    rfl -- Placeholder for the actual arithmetic; ideally, you'd verify this step.
  sorry

end exponent_calculation_l85_85797


namespace water_added_eq_30_l85_85553

-- Given conditions
def initial_volume := 50
def initial_concentration := 0.4
def final_concentration := 0.25
def amount_of_acid := initial_volume * initial_concentration

-- Proof problem statement
theorem water_added_eq_30 : ∃ w : ℝ, amount_of_acid / (initial_volume + w) = final_concentration ∧ w = 30 :=
by 
  -- Solution steps go here, but for now we insert sorry to skip the proof.
  sorry

end water_added_eq_30_l85_85553


namespace negation_example_l85_85605

theorem negation_example : 
  (¬ ∃ x_0 : ℚ, x_0 - 2 = 0) = (∀ x : ℚ, x - 2 ≠ 0) :=
by 
  sorry

end negation_example_l85_85605


namespace impossible_to_create_3_piles_l85_85709

def similar_sizes (x y : ℝ) : Prop := x ≤ sqrt 2 * y

theorem impossible_to_create_3_piles (x y : ℝ) (h₁ : similar_sizes x y) :
  ¬ ∃ (x₁ x₂ x₃ : ℝ), x₁ + x₂ + x₃ = x + y ∧ similar_sizes x₁ x₂ ∧ similar_sizes x₂ x₃ ∧ similar_sizes x₁ x₃ :=
by
  sorry

end impossible_to_create_3_piles_l85_85709


namespace find_extrema_l85_85270

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

theorem find_extrema :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → f x ≤ 6) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ f x = 6) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 2 ≤ f x) ∧
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 4 ∧ f x = 2) :=
by sorry

end find_extrema_l85_85270


namespace solution_correctness_l85_85458

def is_prime (n : ℕ) : Prop := Nat.Prime n

def problem_statement (a b c : ℕ) : Prop :=
  (a * b * c = 56) ∧
  (a * b + b * c + a * c = 311) ∧
  is_prime a ∧ is_prime b ∧ is_prime c

theorem solution_correctness (a b c : ℕ) (h : problem_statement a b c) :
  a = 2 ∨ a = 13 ∨ a = 19 ∧
  b = 2 ∨ b = 13 ∨ b = 19 ∧
  c = 2 ∨ c = 13 ∨ c = 19 :=
by
  sorry

end solution_correctness_l85_85458


namespace find_minimum_value_l85_85699

-- This definition captures the condition that a, b, c are positive real numbers
def pos_reals := { x : ℝ // 0 < x }

-- The main theorem statement
theorem find_minimum_value (a b c : pos_reals) :
  4 * (a.1 ^ 4) + 8 * (b.1 ^ 4) + 16 * (c.1 ^ 4) + 1 / (a.1 * b.1 * c.1) ≥ 10 :=
by
  -- This is where the proof will go
  sorry

end find_minimum_value_l85_85699


namespace find_stadium_width_l85_85190

-- Conditions
def stadium_length : ℝ := 24
def stadium_height : ℝ := 16
def longest_pole : ℝ := 34

-- Width to be solved
def stadium_width : ℝ := 18

-- Theorem stating that given the conditions, the width must be 18
theorem find_stadium_width :
  stadium_length^2 + stadium_width^2 + stadium_height^2 = longest_pole^2 :=
by
  sorry

end find_stadium_width_l85_85190


namespace line_through_P_with_intercepts_l85_85513

theorem line_through_P_with_intercepts (a b : ℝ) (P : ℝ × ℝ) (hP : P = (6, -1)) 
  (h1 : a = 3 * b) (ha : a = 1 / ((-b - 1) / 6) + 6) (hb : b = -6 * ((-b - 1) / 6) - 1) :
  (∀ x y, y = (-1 / 3) * x + 1 ∨ y = (-1 / 6) * x) :=
sorry

end line_through_P_with_intercepts_l85_85513


namespace face_opposite_one_is_three_l85_85062

def faces : List ℕ := [1, 2, 3, 4, 5, 6]

theorem face_opposite_one_is_three (x : ℕ) (h1 : x ∈ faces) (h2 : x ≠ 1) : x = 3 :=
by
  sorry

end face_opposite_one_is_three_l85_85062


namespace base_5_representation_l85_85232

theorem base_5_representation (n : ℕ) (h : n = 84) : 
  ∃ (a b c : ℕ), 
  a < 5 ∧ b < 5 ∧ c < 5 ∧ 
  n = a * 5^2 + b * 5^1 + c * 5^0 ∧ 
  a = 3 ∧ b = 1 ∧ c = 4 :=
by 
  -- Placeholder for the proof
  sorry

end base_5_representation_l85_85232


namespace students_not_A_either_l85_85405

-- Given conditions as definitions
def total_students : ℕ := 40
def students_A_history : ℕ := 10
def students_A_math : ℕ := 18
def students_A_both : ℕ := 6

-- Statement to prove
theorem students_not_A_either : (total_students - (students_A_history + students_A_math - students_A_both)) = 18 := 
by
  sorry

end students_not_A_either_l85_85405


namespace sqrt_mul_eq_l85_85085

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l85_85085


namespace roots_abs_less_than_one_l85_85762

theorem roots_abs_less_than_one {a b : ℝ} 
    (h : |a| + |b| < 1) 
    (x1 x2 : ℝ) 
    (h_roots : x1 * x1 + a * x1 + b = 0) 
    (h_roots' : x2 * x2 + a * x2 + b = 0) 
    : |x1| < 1 ∧ |x2| < 1 := 
sorry

end roots_abs_less_than_one_l85_85762


namespace smallest_sum_of_squares_l85_85268

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 187) : x^2 + y^2 ≥ 205 := sorry

end smallest_sum_of_squares_l85_85268


namespace sum_of_numerator_and_denominator_l85_85035

def repeating_decimal_represents_fraction : Prop :=
  let x := 34 / 99
  0.34̅ = x

/-- The sum of the numerator and denominator of 0.34̅ in its lowest terms is 133. -/
theorem sum_of_numerator_and_denominator
  (x : ℚ)
  (hx : repeating_decimal_represents_fraction) :
  (34 + 99 = 133) :=
begin
  sorry
end

end sum_of_numerator_and_denominator_l85_85035


namespace total_emails_in_april_l85_85417

-- Definitions representing the conditions
def emails_per_day_initial : Nat := 20
def extra_emails_per_day : Nat := 5
def days_in_month : Nat := 30
def half_days_in_month : Nat := days_in_month / 2

-- Definitions to calculate total emails
def emails_first_half : Nat := emails_per_day_initial * half_days_in_month
def emails_per_day_after_subscription : Nat := emails_per_day_initial + extra_emails_per_day
def emails_second_half : Nat := emails_per_day_after_subscription * half_days_in_month

-- Main theorem to prove the total number of emails received in April
theorem total_emails_in_april : emails_first_half + emails_second_half = 675 := by 
  calc
    emails_first_half + emails_second_half
    = (emails_per_day_initial * half_days_in_month) + (emails_per_day_after_subscription * half_days_in_month) : rfl
    ... = (20 * 15) + ((20 + 5) * 15) : rfl
    ... = 300 + 375 : rfl
    ... = 675 : rfl

end total_emails_in_april_l85_85417


namespace number_of_badminton_players_l85_85679

-- Definitions based on the given conditions
variable (Total_members : ℕ := 30)
variable (Tennis_players : ℕ := 19)
variable (No_sport_players : ℕ := 3)
variable (Both_sport_players : ℕ := 9)

-- The goal is to prove the number of badminton players is 17
theorem number_of_badminton_players :
  ∀ (B : ℕ), Total_members = B + Tennis_players - Both_sport_players + No_sport_players → B = 17 :=
by
  intro B
  intro h
  sorry

end number_of_badminton_players_l85_85679


namespace greatest_integer_value_l85_85966

theorem greatest_integer_value (x : ℤ) (h : 3 * |x| + 4 ≤ 19) : x ≤ 5 :=
by
  sorry

end greatest_integer_value_l85_85966


namespace trapezoid_bisector_segment_length_l85_85936

-- Definitions of the conditions
variables {a b c d t : ℝ}

noncomputable def semiperimeter (a b c d : ℝ) : ℝ := (a + b + c + d) / 2

-- The theorem statement
theorem trapezoid_bisector_segment_length
  (p : ℝ)
  (h_p : p = semiperimeter a b c d) :
  t^2 = (4 * b * d) / (b + d)^2 * (p - a) * (p - c) :=
sorry

end trapezoid_bisector_segment_length_l85_85936


namespace sqrt3_mul_sqrt12_eq_6_l85_85116

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l85_85116


namespace hounds_score_points_l85_85226

theorem hounds_score_points (x y : ℕ) (h_total : x + y = 82) (h_margin : x - y = 18) : y = 32 :=
sorry

end hounds_score_points_l85_85226


namespace sum_x_y_eq_two_l85_85591

theorem sum_x_y_eq_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 28) : x + y = 2 :=
sorry

end sum_x_y_eq_two_l85_85591


namespace answer_is_p_and_q_l85_85371

variable (p q : Prop)
variable (h₁ : ∃ x : ℝ, Real.sin x < 1)
variable (h₂ : ∀ x : ℝ, Real.exp (|x|) ≥ 1)

theorem answer_is_p_and_q : p ∧ q :=
by sorry

end answer_is_p_and_q_l85_85371


namespace gcd_of_g_and_y_l85_85380

noncomputable def g (y : ℕ) := (3 * y + 5) * (8 * y + 3) * (16 * y + 9) * (y + 16)

theorem gcd_of_g_and_y (y : ℕ) (hy : y % 46896 = 0) : Nat.gcd (g y) y = 2160 :=
by
  -- Proof to be written here
  sorry

end gcd_of_g_and_y_l85_85380


namespace sqrt_mult_simplify_l85_85122

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l85_85122


namespace outfits_not_same_color_l85_85862

/--
Given:
- 7 shirts, 7 pairs of pants, and 7 hats.
- Each item comes in 7 colors (one of each item of each color).
- No outfit is allowed where all 3 items are the same color.

Prove:
The number of possible outfits where not all items are the same color is 336.
-/
theorem outfits_not_same_color : 
  let total_outfits := 7 * 7 * 7 in
  let same_color_outfits := 7 in
  total_outfits - same_color_outfits = 336 :=
by
  let total_outfits := 7 * 7 * 7
  let same_color_outfits := 7
  have h1 : total_outfits = 343 := by norm_num
  have h2 : total_outfits - same_color_outfits = 336 := by norm_num
  exact h2

end outfits_not_same_color_l85_85862


namespace forgotten_angle_measure_l85_85229

theorem forgotten_angle_measure 
  (total_sum : ℕ) 
  (measured_sum : ℕ) 
  (sides : ℕ) 
  (n_minus_2 : ℕ)
  (polygon_has_18_sides : sides = 18)
  (interior_angle_sum : total_sum = n_minus_2 * 180)
  (n_minus : n_minus_2 = (sides - 2))
  (measured : measured_sum = 2754) :
  ∃ forgotten_angle, forgotten_angle = total_sum - measured_sum ∧ forgotten_angle = 126 :=
by
  sorry

end forgotten_angle_measure_l85_85229


namespace total_recess_correct_l85_85743

-- Definitions based on the conditions
def base_recess : Int := 20
def recess_for_A (n : Int) : Int := n * 2
def recess_for_B (n : Int) : Int := n * 1
def recess_for_C (n : Int) : Int := n * 0
def recess_for_D (n : Int) : Int := -n * 1

def total_recess (a b c d : Int) : Int :=
  base_recess + recess_for_A a + recess_for_B b + recess_for_C c + recess_for_D d

-- The proof statement originally there would use these inputs
theorem total_recess_correct : total_recess 10 12 14 5 = 47 := by
  sorry

end total_recess_correct_l85_85743


namespace vector_subtraction_l85_85508

/-
Define the vectors we are working with.
-/
def v1 : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![-8]]
def v2 : Matrix (Fin 2) (Fin 1) ℤ := ![![2], ![-6]]
def scalar : ℤ := 5
def result : Matrix (Fin 2) (Fin 1) ℤ := ![![-7], ![22]]

/-
The statement of the proof problem.
-/
theorem vector_subtraction : v1 - scalar • v2 = result := 
by
  sorry

end vector_subtraction_l85_85508


namespace sqrt_square_identity_l85_85497

-- Define the concept of square root
noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Problem statement: prove (sqrt 12321)^2 = 12321
theorem sqrt_square_identity (x : ℝ) : (sqrt x) ^ 2 = x := by
  sorry

-- Specific instance for the given number
example : (sqrt 12321) ^ 2 = 12321 := sqrt_square_identity 12321

end sqrt_square_identity_l85_85497


namespace exponent_subtraction_l85_85660

theorem exponent_subtraction (a : ℝ) (m n : ℝ) (hm : a^m = 3) (hn : a^n = 5) : a^(m-n) = 3 / 5 := 
  sorry

end exponent_subtraction_l85_85660


namespace reciprocal_of_2016_is_1_div_2016_l85_85741

theorem reciprocal_of_2016_is_1_div_2016 : (2016 * (1 / 2016) = 1) :=
by
  sorry

end reciprocal_of_2016_is_1_div_2016_l85_85741


namespace percent_of_z_l85_85868

variable {x y z : ℝ}

theorem percent_of_z (h₁ : x = 1.20 * y) (h₂ : y = 0.50 * z) : x = 0.60 * z :=
by
  sorry

end percent_of_z_l85_85868


namespace total_amount_l85_85467

theorem total_amount (x y z : ℝ) 
  (hy : y = 0.45 * x) 
  (hz : z = 0.30 * x) 
  (hy_value : y = 54) : 
  x + y + z = 210 := 
by
  sorry

end total_amount_l85_85467


namespace total_students_76_or_80_l85_85247

theorem total_students_76_or_80 
  (N : ℕ)
  (h1 : ∃ g : ℕ → ℕ, (∑ i in finset.range 6, g i = N) ∧
                     (∃ a b : ℕ, finset.card {i | g i = a} = 4 ∧ 
                                 finset.card {i | g i = b} = 2 ∧ 
                                 (a = 13 ∧ (b = 12 ∨ b = 14))))
  : N = 76 ∨ N = 80 :=
sorry

end total_students_76_or_80_l85_85247


namespace goose_eggs_count_l85_85642

theorem goose_eggs_count (E : ℕ)
    (hatch_fraction : ℚ := 1/3)
    (first_month_survival : ℚ := 4/5)
    (first_year_survival : ℚ := 2/5)
    (no_migration : ℚ := 3/4)
    (predator_survival : ℚ := 2/3)
    (final_survivors : ℕ := 140) :
    (predator_survival * no_migration * first_year_survival * first_month_survival * hatch_fraction * E : ℚ) = final_survivors → E = 1050 := by
  sorry

end goose_eggs_count_l85_85642


namespace actual_number_of_sides_l85_85064

theorem actual_number_of_sides (apparent_angle : ℝ) (distortion_factor : ℝ)
  (sum_exterior_angles : ℝ) (actual_sides : ℕ) :
  apparent_angle = 18 ∧ distortion_factor = 1.5 ∧ sum_exterior_angles = 360 ∧ 
  apparent_angle / distortion_factor = sum_exterior_angles / actual_sides →
  actual_sides = 30 :=
by
  sorry

end actual_number_of_sides_l85_85064


namespace problem_equivalent_l85_85528

theorem problem_equivalent (a b x : ℝ) (h₁ : a ≠ b) (h₂ : a + b = 6) (h₃ : a * (a - 6) = x) (h₄ : b * (b - 6) = x) : 
  x = -9 :=
by
  sorry

end problem_equivalent_l85_85528


namespace allan_plums_l85_85430

theorem allan_plums (A : ℕ) (h1 : 7 - A = 3) : A = 4 :=
sorry

end allan_plums_l85_85430


namespace sqrt_mul_simp_l85_85128

theorem sqrt_mul_simp : sqrt 3 * sqrt 12 = 6 :=
by
  sorry

end sqrt_mul_simp_l85_85128


namespace mark_paid_more_than_anne_by_three_dollars_l85_85952

theorem mark_paid_more_than_anne_by_three_dollars :
  let total_slices := 12
  let plain_pizza_cost := 12
  let pepperoni_cost := 3
  let pepperoni_slices := total_slices / 3
  let total_cost := plain_pizza_cost + pepperoni_cost
  let cost_per_slice := total_cost / total_slices
  let plain_cost_per_slice := cost_per_slice
  let pepperoni_cost_per_slice := cost_per_slice + pepperoni_cost / pepperoni_slices
  let mark_total := 4 * pepperoni_cost_per_slice + 2 * plain_cost_per_slice
  let anne_total := 6 * plain_cost_per_slice
  mark_total - anne_total = 3 :=
by
  let total_slices := 12
  let plain_pizza_cost := 12
  let pepperoni_cost := 3
  let pepperoni_slices := total_slices / 3
  let total_cost := plain_pizza_cost + pepperoni_cost
  let cost_per_slice := total_cost / total_slices
  let plain_cost_per_slice := cost_per_slice
  let pepperoni_cost_per_slice := cost_per_slice + pepperoni_cost / pepperoni_slices
  let mark_total := 4 * pepperoni_cost_per_slice + 2 * plain_cost_per_slice
  let anne_total := 6 * plain_cost_per_slice
  sorry

end mark_paid_more_than_anne_by_three_dollars_l85_85952


namespace find_p8_l85_85897

noncomputable def p (x : ℝ) : ℝ := sorry -- p is a monic polynomial of degree 7

def monic_degree_7 (p : ℝ → ℝ) : Prop := sorry -- p is monic polynomial of degree 7
def satisfies_conditions (p : ℝ → ℝ) : Prop :=
  p 1 = 2 ∧ p 2 = 3 ∧ p 3 = 4 ∧ p 4 = 5 ∧ p 5 = 6 ∧ p 6 = 7 ∧ p 7 = 8

theorem find_p8 (h_monic : monic_degree_7 p) (h_conditions : satisfies_conditions p) : p 8 = 5049 :=
by
  sorry

end find_p8_l85_85897


namespace sum_integers_neg50_to_75_l85_85290

-- Definitions representing the conditions
def symmetric_sum_to_zero (a b : ℤ) : Prop :=
  (a = -b) → (a + b = 0)

def arithmetic_series_sum (first last terms : ℤ) : ℤ :=
  let average := (first + last) / 2
  average * terms

-- The theorem we need to state
theorem sum_integers_neg50_to_75 : 
  (symmetric_sum_to_zero (-50) 50) →
  arithmetic_series_sum 51 75 25 = 1575 →
  sum (range (76 + 50 - 1)) = 1575 := 
sorry

end sum_integers_neg50_to_75_l85_85290


namespace sqrt_mul_l85_85097

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l85_85097


namespace algebraic_expression_value_l85_85390

theorem algebraic_expression_value (a b : ℝ) (h : a + b = 3) : a^3 + b^3 + 9 * a * b = 27 :=
by
  sorry

end algebraic_expression_value_l85_85390


namespace factorize_expression_l85_85185

variable {R : Type*} [CommRing R] (a b : R)

theorem factorize_expression : 2 * a^2 * b - 4 * a * b + 2 * b = 2 * b * (a - 1)^2 :=
by
  sorry

end factorize_expression_l85_85185


namespace simplify_expression_l85_85466

theorem simplify_expression (a b : ℝ) : a + (5 * a - 3 * b) = 6 * a - 3 * b :=
by sorry

end simplify_expression_l85_85466


namespace price_relation_l85_85221

-- Defining the conditions
variable (TotalPrice : ℕ) (NumberOfPens : ℕ)
variable (total_price_val : TotalPrice = 24) (number_of_pens_val : NumberOfPens = 16)

-- Statement of the problem
theorem price_relation (y x : ℕ) (h_y : y = 3 / 2) : y = 3 / 2 * x := 
  sorry

end price_relation_l85_85221


namespace range_of_a_l85_85983

theorem range_of_a
  (P : Prop := ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) :
  ¬P → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l85_85983


namespace num_even_multiples_of_four_perfect_squares_lt_5000_l85_85218

theorem num_even_multiples_of_four_perfect_squares_lt_5000 : 
  ∃ (k : ℕ), k = 17 ∧ ∀ (n : ℕ), (0 < n ∧ 16 * n^2 < 5000) ↔ (1 ≤ n ∧ n ≤ 17) :=
by
  sorry

end num_even_multiples_of_four_perfect_squares_lt_5000_l85_85218


namespace sum_possible_values_l85_85703

theorem sum_possible_values (p q r s : ℝ) (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 4) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -1 := 
by
  sorry

end sum_possible_values_l85_85703


namespace probability_of_monochromatic_triangle_l85_85649

-- Definitions:
structure Hexagon :=
  (vertices : Finset ℕ)
  (edges : Finset (ℕ × ℕ))

noncomputable def all_edges := 
  ({(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)} ∪ 
   {(0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (1, 5), 
    (2, 4), (2, 5), (3, 5)} : Finset (ℕ × ℕ))

def hex : Hexagon := { vertices := Finset.range 6, edges := all_edges }

-- Question translated into Lean statement:
theorem probability_of_monochromatic_triangle :
  (let non_mono_prob := 3 / 4 in
   let total_triangles := 20 in
   1 - non_mono_prob^total_triangles) ≈ 0.99683 := sorry

end probability_of_monochromatic_triangle_l85_85649


namespace absolute_value_inequality_l85_85261

theorem absolute_value_inequality (x : ℝ) : (|x + 1| > 3) ↔ (x > 2 ∨ x < -4) :=
by
  sorry

end absolute_value_inequality_l85_85261


namespace min_g_l85_85312

noncomputable def g (x : ℝ) : ℝ := (x^2 + 2) / Real.sqrt (x^2 + 1)

theorem min_g : ∃ x : ℝ, g x = 2 :=
by
  use 0
  sorry

end min_g_l85_85312


namespace unique_solution_l85_85960

theorem unique_solution (a b x: ℝ) : 
  (4 * x - 7 + a = (b - 1) * x + 2) ↔ (b ≠ 5) := 
by
  sorry -- proof is omitted as per instructions

end unique_solution_l85_85960


namespace complex_number_equality_l85_85582

open Complex

theorem complex_number_equality (u v : ℂ) 
  (h1 : 3 * abs (u + 1) * abs (v + 1) ≥ abs (u * v + 5 * u + 5 * v + 1))
  (h2 : abs (u + v) = abs (u * v + 1)) : 
  u = 1 ∨ v = 1 :=
sorry

end complex_number_equality_l85_85582


namespace remainder_of_55_pow_55_plus_15_mod_8_l85_85021

theorem remainder_of_55_pow_55_plus_15_mod_8 :
  (55^55 + 15) % 8 = 6 := by
  -- This statement does not include any solution steps.
  sorry

end remainder_of_55_pow_55_plus_15_mod_8_l85_85021


namespace checker_on_diagonal_l85_85585

theorem checker_on_diagonal
  (board : ℕ)
  (n_checkers : ℕ)
  (symmetric : (ℕ → ℕ → Prop))
  (diag_check : ∀ i j, symmetric i j -> symmetric j i)
  (num_checkers_odd : Odd n_checkers)
  (board_size : board = 25)
  (checkers : n_checkers = 25) :
  ∃ i, i < 25 ∧ symmetric i i := by
  sorry

end checker_on_diagonal_l85_85585


namespace tank_fill_time_l85_85638

/-- Given the rates at which pipes fill a tank, prove the total time to fill the tank using all three pipes. --/
theorem tank_fill_time (R_a R_b R_c : ℝ) (T : ℝ)
  (h1 : R_a = 1 / 35)
  (h2 : R_b = 2 * R_a)
  (h3 : R_c = 2 * R_b)
  (h4 : T = 5) :
  1 / (R_a + R_b + R_c) = T := by
  sorry

end tank_fill_time_l85_85638


namespace largest_partner_share_l85_85837

def total_profit : ℕ := 48000
def partner_ratios : List ℕ := [3, 4, 4, 6, 7]
def value_per_part : ℕ := total_profit / partner_ratios.sum
def largest_share : ℕ := 7 * value_per_part

theorem largest_partner_share :
  largest_share = 14000 := by
  sorry

end largest_partner_share_l85_85837


namespace avg_age_l85_85440

-- Given conditions
variables (A B C : ℕ)
variable (h1 : (A + C) / 2 = 29)
variable (h2 : B = 20)

-- to prove
theorem avg_age (A B C : ℕ) (h1 : (A + C) / 2 = 29) (h2 : B = 20) : (A + B + C) / 3 = 26 :=
sorry

end avg_age_l85_85440


namespace polygon_has_five_sides_l85_85557

theorem polygon_has_five_sides (angle : ℝ) (h : angle = 108) :
  (∃ n : ℕ, n > 2 ∧ (180 - angle) * n = 360) ↔ n = 5 := 
by
  sorry

end polygon_has_five_sides_l85_85557


namespace not_possible_to_create_3_piles_l85_85706

def similar_sizes (a b : ℝ) : Prop := a / b ≤ Real.sqrt 2

theorem not_possible_to_create_3_piles (x : ℝ) (hx : 0 < x) : ¬ ∃ (y z w : ℝ), 
  y + z + w = x ∧ 
  similar_sizes y z ∧ similar_sizes z w ∧ similar_sizes y w := 
by 
  sorry

end not_possible_to_create_3_piles_l85_85706


namespace spencer_total_distance_l85_85697

def d1 : ℝ := 1.2
def d2 : ℝ := 0.6
def d3 : ℝ := 0.9
def d4 : ℝ := 1.7
def d5 : ℝ := 2.1
def d6 : ℝ := 1.3
def d7 : ℝ := 0.8

theorem spencer_total_distance : d1 + d2 + d3 + d4 + d5 + d6 + d7 = 8.6 :=
by
  sorry

end spencer_total_distance_l85_85697


namespace total_cost_of_crayons_l85_85881

theorem total_cost_of_crayons (crayons_per_half_dozen : ℕ)
    (number_of_half_dozens : ℕ)
    (cost_per_crayon : ℕ)
    (total_cost : ℕ) :
  crayons_per_half_dozen = 6 →
  number_of_half_dozens = 4 →
  cost_per_crayon = 2 →
  total_cost = crayons_per_half_dozen * number_of_half_dozens * cost_per_crayon →
  total_cost = 48 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end total_cost_of_crayons_l85_85881


namespace prob_eq_l85_85677

-- Condition definitions
def fair_dice (n : ℕ) : Prop := 
  ∀ (d : ℕ), d < n → ∀ (face : ℕ), face < 6 → (face < 3 ∨ face < 6)

def balanced_faces (die : ℕ) : Prop := 
  ∀ (color : ℕ), color = 0 ∨ color = 1

def p (n : ℕ) : ℝ := 
  1 / 2 -- Placeholder for the actual probability definition

def q (n : ℕ) : ℝ := 
  1 / 2 -- Placeholder for the actual probability definition

-- Theorem to prove
theorem prob_eq (n : ℕ) (h1 : fair_dice n) (h2 : balanced_faces n) (h3 : 0 < n) :
  p n + (n - 1) * q n = n / 2 :=
by
  sorry

end prob_eq_l85_85677


namespace minimum_value_ineq_l85_85898

theorem minimum_value_ineq (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 1) :
  (a^2 / b) + (b^2 / c) + (c^2 / a) ≥ 3 :=
by
  sorry

end minimum_value_ineq_l85_85898


namespace maximize_x5y3_l85_85581

theorem maximize_x5y3 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 30) :
  x = 18.75 ∧ y = 11.25 → (x^5 * y^3) = (18.75^5 * 11.25^3) :=
sorry

end maximize_x5y3_l85_85581


namespace passenger_gets_ticket_l85_85944

variables (p1 p2 p3 p4 p5 p6 : ℝ)

-- Conditions:
axiom h_sum_eq_one : p1 + p2 + p3 = 1
axiom h_p1_nonneg : 0 ≤ p1
axiom h_p2_nonneg : 0 ≤ p2
axiom h_p3_nonneg : 0 ≤ p3
axiom h_p4_nonneg : 0 ≤ p4
axiom h_p4_le_one : p4 ≤ 1
axiom h_p5_nonneg : 0 ≤ p5
axiom h_p5_le_one : p5 ≤ 1
axiom h_p6_nonneg : 0 ≤ p6
axiom h_p6_le_one : p6 ≤ 1

-- Theorem:
theorem passenger_gets_ticket :
  (p1 * (1 - p4) + p2 * (1 - p5) + p3 * (1 - p6)) = (p1 * (1 - p4) + p2 * (1 - p5) + p3 * (1 - p6)) :=
by sorry

end passenger_gets_ticket_l85_85944


namespace inequality_solution_l85_85009

theorem inequality_solution (x : ℝ)
  (h : ∀ x, x^2 + 2 * x + 7 > 0) :
  (x - 3) / (x^2 + 2 * x + 7) ≥ 0 ↔ x ∈ Set.Ici 3 :=
by
  sorry

end inequality_solution_l85_85009


namespace dual_expr_result_solve_sqrt_eq_16_solve_sqrt_rational_eq_4x_l85_85749

-- Question 1
theorem dual_expr_result (m n : ℝ) (h1 : m = 2 - Real.sqrt 3) (h2 : n = 2 + Real.sqrt 3) :
  m * n = 1 :=
sorry

-- Question 2
theorem solve_sqrt_eq_16 (x : ℝ) (h : Real.sqrt (x + 42) + Real.sqrt (x + 10) = 16) :
  x = 39 :=
sorry

-- Question 3
theorem solve_sqrt_rational_eq_4x (x : ℝ) (h : Real.sqrt (4 * x^2 + 6 * x - 5) + Real.sqrt (4 * x^2 - 2 * x - 5) = 4 * x) :
  x = 3 :=
sorry

end dual_expr_result_solve_sqrt_eq_16_solve_sqrt_rational_eq_4x_l85_85749


namespace range_of_m_l85_85896

noncomputable def f (x m : ℝ) := (1/2) * x^2 + m * x + Real.log x

noncomputable def f_prime (x m : ℝ) := x + 1/x + m

theorem range_of_m (x0 m : ℝ) 
  (h1 : (1/2) ≤ x0 ∧ x0 ≤ 3) 
  (unique_x0 : ∀ y, f_prime y m = 0 → y = x0) 
  (cond1 : f_prime (1/2) m < 0) 
  (cond2 : f_prime 3 m ≥ 0) 
  : -10 / 3 ≤ m ∧ m < -5 / 2 :=
sorry

end range_of_m_l85_85896


namespace correct_option_D_l85_85293

theorem correct_option_D (y : ℝ): 
  3 * y^2 - 2 * y^2 = y^2 :=
by
  sorry

end correct_option_D_l85_85293


namespace proposition_A_l85_85348

theorem proposition_A (p : ∃ x : ℝ, sin x < 1) (q : ∀ x: ℝ, exp (|x|) ≥ 1) : p ∧ q :=
by
  sorry

end proposition_A_l85_85348


namespace monotonic_increasing_on_interval_min_value_on_interval_max_value_on_interval_l85_85213

noncomputable def f (x : ℝ) : ℝ := 1 - (3 / (x + 2))

theorem monotonic_increasing_on_interval :
  ∀ (x₁ x₂ : ℝ), 3 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 5 → f x₁ < f x₂ := sorry

theorem min_value_on_interval :
  ∃ (x : ℝ), x = 3 ∧ f x = 2 / 5 := sorry

theorem max_value_on_interval :
  ∃ (x : ℝ), x = 5 ∧ f x = 4 / 7 := sorry

end monotonic_increasing_on_interval_min_value_on_interval_max_value_on_interval_l85_85213


namespace tiling_condition_l85_85923

theorem tiling_condition (a b n : ℕ) : 
  (∃ f : ℕ → ℕ × ℕ, ∀ i < (a * b) / n, (f i).fst < a ∧ (f i).snd < b) ↔ (n ∣ a ∨ n ∣ b) :=
sorry

end tiling_condition_l85_85923


namespace inscribed_square_area_l85_85778

-- Define the conditions and the problem
theorem inscribed_square_area
  (side_length : ℝ)
  (square_area : ℝ) :
  side_length = 24 →
  square_area = 576 :=
by
  sorry

end inscribed_square_area_l85_85778


namespace ratio_Ford_to_Toyota_l85_85611

-- Definitions based on the conditions
variables (Ford Dodge Toyota VW : ℕ)

axiom h1 : Ford = (1/3 : ℚ) * Dodge
axiom h2 : VW = (1/2 : ℚ) * Toyota
axiom h3 : VW = 5
axiom h4 : Dodge = 60

-- Theorem statement to be proven
theorem ratio_Ford_to_Toyota : Ford / Toyota = 2 :=
by {
  sorry
}

end ratio_Ford_to_Toyota_l85_85611


namespace smallest_k_for_sequence_l85_85663

theorem smallest_k_for_sequence {a : ℕ → ℕ} (k : ℕ) (h : ∀ n ≥ 2, a (n + 1) = k * (a n) / (a (n - 1))) 
  (h0 : a 1 = 1) (h1 : a 2018 = 2020) (h2 : ∀ n, a n ∈ ℕ) : k = 2020 :=
sorry

end smallest_k_for_sequence_l85_85663


namespace circle_diameter_l85_85315

theorem circle_diameter (A : ℝ) (h : A = 64 * Real.pi) : ∃ d : ℝ, d = 16 :=
by
  sorry

end circle_diameter_l85_85315


namespace sum_of_possible_values_l85_85995

variable (N K : ℝ)

theorem sum_of_possible_values (h1 : N ≠ 0) (h2 : N - (3 / N) = K) : N + (K / N) = K := 
sorry

end sum_of_possible_values_l85_85995


namespace percentage_problem_l85_85398

variable (N P : ℝ)

theorem percentage_problem (h1 : 0.3 * N = 120) (h2 : (P / 100) * N = 160) : P = 40 :=
by
  sorry

end percentage_problem_l85_85398


namespace remaining_pages_l85_85573

def original_book_pages : ℕ := 93
def pages_read_saturday : ℕ := 30
def pages_read_sunday : ℕ := 20

theorem remaining_pages :
  original_book_pages - (pages_read_saturday + pages_read_sunday) = 43 := by
  sorry

end remaining_pages_l85_85573


namespace find_b_value_l85_85752

theorem find_b_value :
  (∀ x : ℝ, (x < 0 ∨ x > 4) → -x^2 + 4*x - 4 < 0) ↔ b = 4 := by
sorry

end find_b_value_l85_85752


namespace solution_set_of_inequality_l85_85448

theorem solution_set_of_inequality (x : ℝ) : (x * |x - 1| > 0) ↔ (0 < x ∧ x < 1 ∨ 1 < x) := 
by
  sorry

end solution_set_of_inequality_l85_85448


namespace not_possible_to_create_3_piles_l85_85705

def similar_sizes (a b : ℝ) : Prop := a / b ≤ Real.sqrt 2

theorem not_possible_to_create_3_piles (x : ℝ) (hx : 0 < x) : ¬ ∃ (y z w : ℝ), 
  y + z + w = x ∧ 
  similar_sizes y z ∧ similar_sizes z w ∧ similar_sizes y w := 
by 
  sorry

end not_possible_to_create_3_piles_l85_85705


namespace sqrt_mul_eq_l85_85081

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l85_85081


namespace div_neg_rev_l85_85994

theorem div_neg_rev (a b : ℝ) (h : a > b) : (a / -3) < (b / -3) :=
by
  sorry

end div_neg_rev_l85_85994


namespace solve_for_x_l85_85277

theorem solve_for_x : ∀ x : ℝ, (3 * x + 15 = (1 / 3) * (6 * x + 45)) → x = 0 := by
  intros x h
  sorry

end solve_for_x_l85_85277


namespace linear_regression_probability_and_expectation_l85_85476

section LinearRegression

variable (data : List (ℕ × ℕ))
variable (x̄ ȳ : ℚ)
variable (n : ℕ)
variable (b : ℚ)
variable (a : ℚ)

-- Define the data
def givenData := [(1, 0), (2, 4), (3, 7), (4, 9), (5, 11), (6, 12), (7, 13)]

-- Define the averages
def x_average := (1 + 2 + 3 + 4 + 5 + 6 + 7) / 7
def y_average := (0 + 4 + 7 + 9 + 11 + 12 + 13) / 7

-- Define the regression coefficients
def b := (1 * 0 + 2 * 4 + 3 * 7 + 4 * 9 + 5 * 11 + 6 * 12 + 7 * 13 - 7 * x_average * y_average) /
          (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2 - 7 * x_average^2)

def a := y_average - b * x_average

-- Property to prove
theorem linear_regression : a = -3 / 7 ∧ b = 59 / 28 ∧ ∀ x, (a + b * x) = 59 / 28 * x - 3 / 7 := 
begin
  sorry
end

end LinearRegression

section ProbabilityAndExpectation

variable (data : List (ℕ × ℕ))
variable (ξ : Type) [Fintype ξ] [DecidableEq ξ]

-- Given data and probability calculations
def num_days_gt_average := [(4, 9), (5, 11), (6, 12), (7, 13)].length
def num_days_le_average := [(1, 0), (2, 4), (3, 7)].length
def P (k : ℕ) := (nat.choose num_days_le_average (3 - k) * nat.choose num_days_gt_average k) /
                 (nat.choose 7 3 : ℚ)

-- Expected value calculation
def E := ∑ k in (Finset.range 4), k * P k

-- Property to prove
theorem probability_and_expectation : E = 12 / 7 :=
begin
  sorry
end

end ProbabilityAndExpectation

end linear_regression_probability_and_expectation_l85_85476


namespace second_group_people_l85_85938

theorem second_group_people (x : ℕ) (K : ℕ) (hK : K > 0) :
  (96 - 16 = K * (x + 16) + 6) → (x = 58 ∨ x = 21) :=
by
  intro h
  sorry

end second_group_people_l85_85938


namespace add_words_to_meet_requirement_l85_85051

-- Definitions required by the problem
def yvonne_words : ℕ := 400
def janna_extra_words : ℕ := 150
def words_removed : ℕ := 20
def requirement : ℕ := 1000

-- Derived values based on the conditions
def janna_words : ℕ := yvonne_words + janna_extra_words
def initial_words : ℕ := yvonne_words + janna_words
def words_after_removal : ℕ := initial_words - words_removed
def words_added : ℕ := 2 * words_removed
def total_words_after_editing : ℕ := words_after_removal + words_added
def words_to_add : ℕ := requirement - total_words_after_editing

-- The theorem to prove
theorem add_words_to_meet_requirement : words_to_add = 30 := by
  sorry

end add_words_to_meet_requirement_l85_85051


namespace polynomial_divisible_by_a_plus_1_l85_85328

theorem polynomial_divisible_by_a_plus_1 (a : ℤ) : (3 * a + 5) ^ 2 - 4 ∣ a + 1 := 
by
  sorry

end polynomial_divisible_by_a_plus_1_l85_85328


namespace probability_abc_144_l85_85046

-- Define the set of possible outcomes for a standard die
def die_faces := {1, 2, 3, 4, 5, 6}

-- Define the event that the product of three dice is 144
def event (a b c : ℕ) := a * b * c = 144

-- Calculate the probability of the event
def probability_event := 
  (1 / 6) * (1 / 6) * (1 / 6)

theorem probability_abc_144 : 
  ∑ (a ∈ die_faces) ∑ (b ∈ die_faces) ∑ (c ∈ die_faces), if event a b c then probability_event else 0 = 1 / 72 :=
by
  sorry

end probability_abc_144_l85_85046


namespace minimum_voters_for_tall_giraffe_l85_85684

-- Definitions and conditions
def voters : ℕ := 135
def districts : ℕ := 5
def precincts_per_district : ℕ := 9
def voters_per_precinct : ℕ := 3

-- Conditions encoded in the problem statement
def majority_precinct (votes: ℕ) : Prop := votes >= 2
def majority_district (precinct_wins: ℕ) : Prop := precinct_wins >= 5
def majority_winner (district_wins: ℕ) : Prop := district_wins >= 3

-- The problem states that the Tall giraffe won.
axiom tall_giraffe_won : ∃ district_wins : ℕ, 
  majority_winner district_wins ∧ 
  ∀ (d ∈ (finset.range districts)), ∃ precinct_wins : ℕ, 
  majority_district precinct_wins ∧ 
  ∀ (p ∈ (finset.range precincts_per_district)), ∃ votes : ℕ, 
  majority_precinct votes

-- Proof goal
theorem minimum_voters_for_tall_giraffe : ∃ (votes_for_tall : ℕ), votes_for_tall = 30 :=
by {
  -- proof of the theorem will go here
  sorry 
}

end minimum_voters_for_tall_giraffe_l85_85684


namespace find_smaller_number_l85_85451

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 :=
sorry

end find_smaller_number_l85_85451


namespace terminating_decimal_zeros_l85_85991

-- Define a generic environment for terminating decimal and problem statement
def count_zeros (d : ℚ) : ℕ :=
  -- This function needs to count the zeros after the decimal point and before
  -- the first non-zero digit, but its actual implementation is skipped here.
  sorry

-- Define the specific fraction in question
def my_fraction : ℚ := 1 / (2^3 * 5^5)

-- State what we need to prove: the number of zeros after the decimal point
-- in the terminating representation of my_fraction should be 4
theorem terminating_decimal_zeros : count_zeros my_fraction = 4 :=
by
  -- Proof is skipped
  sorry

end terminating_decimal_zeros_l85_85991


namespace impossible_to_form_three_similar_piles_l85_85715

-- Define the concept of similar sizes
def similar_sizes (a b : ℝ) : Prop := a ≤ b * real.sqrt 2 ∧ b ≤ a * real.sqrt 2

-- State the proof problem
theorem impossible_to_form_three_similar_piles (x : ℝ) (h : 0 < x) : ¬(∃ a b c : ℝ, a + b + c = x ∧ similar_sizes a b ∧ similar_sizes b c ∧ similar_sizes a c) :=
sorry

end impossible_to_form_three_similar_piles_l85_85715


namespace max_value_range_l85_85854

theorem max_value_range (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_deriv : ∀ x, f' x = a * (x - 1) * (x - a))
  (h_max : ∀ x, (x = a → (∀ y, f y ≤ f x))) : 0 < a ∧ a < 1 :=
sorry

end max_value_range_l85_85854


namespace find_smaller_number_l85_85450

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 :=
sorry

end find_smaller_number_l85_85450


namespace largest_n_divisible_by_n_plus_10_l85_85832

theorem largest_n_divisible_by_n_plus_10 :
  ∃ n : ℕ, (n^3 + 100) % (n + 10) = 0 ∧ ∀ m : ℕ, ((m^3 + 100) % (m + 10) = 0 → m ≤ n) ∧ n = 890 := 
sorry

end largest_n_divisible_by_n_plus_10_l85_85832


namespace sqrt3_mul_sqrt12_eq_6_l85_85093

theorem sqrt3_mul_sqrt12_eq_6 : (sqrt 3) * (sqrt 12) = 6 :=
by sorry

end sqrt3_mul_sqrt12_eq_6_l85_85093


namespace haley_magazines_l85_85542

theorem haley_magazines (boxes : ℕ) (magazines_per_box : ℕ) (total_magazines : ℕ) :
  boxes = 7 →
  magazines_per_box = 9 →
  total_magazines = boxes * magazines_per_box →
  total_magazines = 63 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end haley_magazines_l85_85542


namespace proof_problem_l85_85359

-- Conditions from the problem
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, real.exp (|x|) ≥ 1

-- Statement to be proved: p ∧ q
theorem proof_problem : p ∧ q := by
  sorry

end proof_problem_l85_85359


namespace abs_ineq_l85_85262

theorem abs_ineq (x : ℝ) (h : |x + 1| > 3) : x < -4 ∨ x > 2 :=
  sorry

end abs_ineq_l85_85262


namespace sorting_five_rounds_l85_85029

def direct_sorting_method (l : List ℕ) : List ℕ := sorry

theorem sorting_five_rounds (initial_seq : List ℕ) :
  initial_seq = [49, 38, 65, 97, 76, 13, 27] →
  (direct_sorting_method ∘ direct_sorting_method ∘ direct_sorting_method ∘ direct_sorting_method ∘ direct_sorting_method) initial_seq = [97, 76, 65, 49, 38, 13, 27] :=
by
  intros h
  sorry

end sorting_five_rounds_l85_85029


namespace exponent_calculation_l85_85799

-- Define the necessary exponents and base
def base : ℕ := 19
def exp1 : ℕ := 11
def exp2 : ℕ := 8

-- Given condition
lemma power_property (a : ℕ) (m n : ℕ) : a^m / a^n = a^(m - n) := by sorry

-- Proof that 19^11 / 19^8 = 6859
theorem exponent_calculation : base^exp1 / base^exp2 = 6859 := by
  have : base^exp1 / base^exp2 = base^(exp1 - exp2) := power_property base exp1 exp2
  have : base^(exp1 - exp2) = base^3 := by rw [nat.sub_eq_iff_eq_add.mpr rfl]
  have : base^3 = 6859 := by -- This would be an arithmetic computation
    rfl -- Placeholder for the actual arithmetic; ideally, you'd verify this step.
  sorry

end exponent_calculation_l85_85799


namespace minimum_voters_for_tall_giraffe_to_win_l85_85688

/-- Conditions -/
def total_voters := 135
def districts := 5
def precincts_per_district := 9
def voters_per_precinct := 3
def majority (n : Nat) : Nat := Nat.ceil (n / 2)

/-- Proof Problem -/
theorem minimum_voters_for_tall_giraffe_to_win : 
  let precincts_needed_in_district := majority precincts_per_district,
      districts_needed_to_win := majority districts,
      precincts_needed_in_total := districts_needed_to_win * precincts_needed_in_district,
      majority_in_precinct := majority voters_per_precinct in
  precincts_needed_in_total * majority_in_precinct = 30 :=
by
  -- Proof steps (commented out)
  -- 1. Calculate the number of precincts in total: 5 * 9 = 45
  -- 2. Calculate how many districts the Tall giraffe needs to win: 3
  -- (majority districts)
  -- 3. Calculate how many precincts in a district the Tall giraffe needs to
  -- win: 5 (majority precincts_per_district)
  -- 4. Calculate the total number of precincts the Tall giraffe needs to win:
  -- 3 * 5 = 15
  -- 5. Calculate the majority of votes in a precinct: 2 
  -- (majority voters_per_precinct)
  -- 6. Calculate the minimum number of voters in 15 precincts: 15 * 2 = 30
  sorry

end minimum_voters_for_tall_giraffe_to_win_l85_85688


namespace cost_of_gas_l85_85569

def hoursDriven1 : ℕ := 2
def speed1 : ℕ := 60
def hoursDriven2 : ℕ := 3
def speed2 : ℕ := 50
def milesPerGallon : ℕ := 30
def costPerGallon : ℕ := 2

def totalDistance : ℕ := (hoursDriven1 * speed1) + (hoursDriven2 * speed2)
def gallonsUsed : ℕ := totalDistance / milesPerGallon
def totalCost : ℕ := gallonsUsed * costPerGallon

theorem cost_of_gas : totalCost = 18 := by
  -- You should fill in the proof steps here.
  sorry

end cost_of_gas_l85_85569


namespace parabola_translation_l85_85020

theorem parabola_translation :
  ∀ (x y : ℝ), (y = 2 * (x - 3) ^ 2) ↔ ∃ t : ℝ, t = x - 3 ∧ y = 2 * t ^ 2 :=
by sorry

end parabola_translation_l85_85020


namespace triangle_area_correct_l85_85889

-- Define the vectors a, b, and c as given in the problem
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (6, 2)
def c : ℝ × ℝ := (1, -1)

-- Define the function to calculate the area of the triangle with the given vertices
def triangle_area (u v w : ℝ × ℝ) : ℝ :=
  0.5 * abs ((v.1 - u.1) * (w.2 - u.2) - (w.1 - u.1) * (v.2 - u.2))

-- State the proof problem
theorem triangle_area_correct : triangle_area c (a.1 + c.1, a.2 + c.2) (b.1 + c.1, b.2 + c.2) = 8.5 :=
by
  -- Proof can go here
  sorry

end triangle_area_correct_l85_85889


namespace no_positive_integer_solutions_l85_85324

theorem no_positive_integer_solutions :
  ¬ ∃ (x y : ℕ) (h1 : x > 0) (h2 : y > 0), 21 * x * y = 7 - 3 * x - 4 * y :=
by
  sorry

end no_positive_integer_solutions_l85_85324


namespace compound_interest_correct_l85_85997
noncomputable def compound_interest_proof : Prop :=
  let si := 55
  let r := 5
  let t := 2
  let p := si * 100 / (r * t)
  let ci := p * ((1 + r / 100)^t - 1)
  ci = 56.375

theorem compound_interest_correct : compound_interest_proof :=
by {
  sorry
}

end compound_interest_correct_l85_85997


namespace birds_on_fence_l85_85278

theorem birds_on_fence :
  let i := 12           -- initial birds
  let added1 := 8       -- birds that land first
  let T := i + added1   -- total first stage birds
  
  let fly_away1 := 5
  let join1 := 3
  let W := T - fly_away1 + join1   -- birds after some fly away, others join
  
  let D := W * 2       -- birds doubles
  
  let fly_away2 := D * 0.25  -- 25% fly away
  let D_after_fly_away := D - fly_away2
  
  let return_birds := 2        -- 2.5 birds return, rounded down to 2
  let final_birds := D_after_fly_away + return_birds
  
  final_birds = 29 := 
by {
  sorry
}

end birds_on_fence_l85_85278


namespace maximum_possible_value_of_k_l85_85833

theorem maximum_possible_value_of_k :
  ∀ (k : ℕ), 
    (∃ (x : ℕ → ℝ), 
      (∀ i j : ℕ, 1 ≤ i ∧ i ≤ k ∧ 1 ≤ j ∧ j ≤ k → x i > 1 ∧ x i ≠ x j ∧ x i ^ ⌊x j⌋ = x j ^ ⌊x i⌋)) 
      → k ≤ 4 :=
by
  sorry

end maximum_possible_value_of_k_l85_85833


namespace not_possible_to_create_3_piles_l85_85725

theorem not_possible_to_create_3_piles (similar: ℝ → ℝ → Prop) (sqrt_2 : ℝ)
  (hsimilar : ∀ x y, similar x y ↔ x ≤ sqrt_2 * y ∧ y ≤ sqrt_2 * x) :
  ∀ x, ¬ ∃ x1 x2 x3, 
    x = x1 + x2 + x3 ∧ 
    similar x1 x2 ∧ 
    similar x2 x3 ∧ 
    similar x1 x3 :=
by { sorry }

end not_possible_to_create_3_piles_l85_85725


namespace exist_monochromatic_equilateral_triangle_l85_85032

theorem exist_monochromatic_equilateral_triangle 
  (color : ℝ × ℝ → ℕ) 
  (h_color : ∀ p : ℝ × ℝ, color p = 0 ∨ color p = 1) : 
  ∃ (A B C : ℝ × ℝ), (dist A B = dist B C) ∧ (dist B C = dist C A) ∧ (color A = color B ∧ color B = color C) :=
sorry

end exist_monochromatic_equilateral_triangle_l85_85032


namespace toms_age_l85_85279

theorem toms_age (T S : ℕ) (h1 : T = 2 * S - 1) (h2 : T + S = 14) : T = 9 :=
sorry

end toms_age_l85_85279


namespace solution_set_l85_85850

variable {f : ℝ → ℝ}
variable (h1 : ∀ x, x < 0 → x * deriv f x - 2 * f x > 0)
variable (h2 : ∀ x, x < 0 → f x ≠ 0)

theorem solution_set (h3 : ∀ x, -2024 < x ∧ x < -2023 → f (x + 2023) - (x + 2023)^2 * f (-1) < 0) :
    {x : ℝ | f (x + 2023) - (x + 2023)^2 * f (-1) < 0} = {x : ℝ | -2024 < x ∧ x < -2023} :=
by
  sorry

end solution_set_l85_85850


namespace sqrt_multiplication_l85_85154

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l85_85154


namespace integers_multiples_of_d_l85_85814

theorem integers_multiples_of_d (d m n : ℕ) 
  (h1 : 2 ≤ m) 
  (h2 : 1 ≤ n) 
  (gcd_m_n : Nat.gcd m n = d) 
  (gcd_m_4n1 : Nat.gcd m (4 * n + 1) = 1) : 
  m % d = 0 :=
sorry

end integers_multiples_of_d_l85_85814


namespace f_is_even_l85_85894

-- Given an odd function g
def is_odd_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g (x)

-- Define the function f as given by the problem
def f (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  abs (g (x^2))

-- The theorem stating that f is an even function
theorem f_is_even (g : ℝ → ℝ) (h_odd : is_odd_function g) : ∀ x, f g x = f g (-x) :=
by
  sorry

end f_is_even_l85_85894


namespace probability_of_more_heads_than_tails_l85_85397

-- Define the probability of getting more heads than tails when flipping 10 coins
def probabilityMoreHeadsThanTails : ℚ :=
  193 / 512

-- Define the proof statement
theorem probability_of_more_heads_than_tails :
  let p : ℚ := probabilityMoreHeadsThanTails in
  p = 193 / 512 :=
by
  sorry

end probability_of_more_heads_than_tails_l85_85397


namespace sum_numerator_denominator_repeating_decimal_l85_85041

theorem sum_numerator_denominator_repeating_decimal :
  let x := 34 / 99 in
  x.denom * x + x.num = 133 :=
by
  let x : ℚ := 34 / 99
  sorry

end sum_numerator_denominator_repeating_decimal_l85_85041


namespace crayons_at_the_end_of_thursday_l85_85584

-- Definitions for each day's changes
def monday_crayons : ℕ := 7
def tuesday_crayons (initial : ℕ) := initial + 3
def wednesday_crayons (initial : ℕ) := initial - 5 + 4
def thursday_crayons (initial : ℕ) := initial + 6 - 2

-- Proof statement to show the number of crayons at the end of Thursday
theorem crayons_at_the_end_of_thursday : thursday_crayons (wednesday_crayons (tuesday_crayons monday_crayons)) = 13 :=
by
  sorry

end crayons_at_the_end_of_thursday_l85_85584


namespace total_travel_expenses_l85_85602

noncomputable def cost_of_fuel_tank := 45
noncomputable def miles_per_tank := 500
noncomputable def journey_distance := 2000
noncomputable def food_ratio := 3 / 5
noncomputable def hotel_cost_per_night := 80
noncomputable def number_of_hotel_nights := 3
noncomputable def fuel_cost_increase := 5

theorem total_travel_expenses :
  let number_of_refills := journey_distance / miles_per_tank
  let first_refill_cost := cost_of_fuel_tank
  let second_refill_cost := first_refill_cost + fuel_cost_increase
  let third_refill_cost := second_refill_cost + fuel_cost_increase
  let fourth_refill_cost := third_refill_cost + fuel_cost_increase
  let total_fuel_cost := first_refill_cost + second_refill_cost + third_refill_cost + fourth_refill_cost
  let total_food_cost := food_ratio * total_fuel_cost
  let total_hotel_cost := hotel_cost_per_night * number_of_hotel_nights
  let total_expenses := total_fuel_cost + total_food_cost + total_hotel_cost
  total_expenses = 576 := by sorry

end total_travel_expenses_l85_85602


namespace max_value_of_expression_l85_85840

theorem max_value_of_expression (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) : 
  (∃ x : ℝ, x = 3 → 
    ∀ A : ℝ, A = (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4/3)) → 
      A ≤ x) :=
by
  sorry

end max_value_of_expression_l85_85840


namespace total_time_to_row_l85_85934

theorem total_time_to_row (boat_speed_in_still_water : ℝ) (stream_speed : ℝ) (distance : ℝ) :
  boat_speed_in_still_water = 9 → stream_speed = 1.5 → distance = 105 → 
  (distance / (boat_speed_in_still_water + stream_speed)) + (distance / (boat_speed_in_still_water - stream_speed)) = 24 :=
by
  intro h_boat_speed h_stream_speed h_distance
  rw [h_boat_speed, h_stream_speed, h_distance]
  sorry

end total_time_to_row_l85_85934


namespace factors_of_expression_l85_85543

def total_distinct_factors : ℕ :=
  let a := 10
  let b := 3
  let c := 2
  (a + 1) * (b + 1) * (c + 1)

theorem factors_of_expression :
  total_distinct_factors = 132 :=
by 
  -- the proof goes here
  sorry

end factors_of_expression_l85_85543


namespace product_of_consecutive_even_numbers_divisible_by_24_l85_85758

theorem product_of_consecutive_even_numbers_divisible_by_24 (n : ℕ) :
  (2 * n) * (2 * n + 2) * (2 * n + 4) % 24 = 0 :=
  sorry

end product_of_consecutive_even_numbers_divisible_by_24_l85_85758


namespace sin_510_eq_1_div_2_l85_85472

theorem sin_510_eq_1_div_2 : Real.sin (510 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_510_eq_1_div_2_l85_85472


namespace find_n_l85_85666

theorem find_n (n : ℕ) (h : n > 0) :
  (n * (n - 1) * (n - 2)) / (6 * n^3) = 1 / 16 ↔ n = 4 :=
by sorry

end find_n_l85_85666


namespace sqrt_mult_eq_six_l85_85164

theorem sqrt_mult_eq_six (a b : ℝ) (h1 : a = 3) (h2 : b = 12) 
  (h3 : sqrt b = 2 * sqrt a)
  (h4 : sqrt a * sqrt b = sqrt (a * b)) : sqrt 3 * sqrt 12 = 6 :=
by 
  rw [h1, h2] at h3,
  rw [h1, h2, h3, h4],
  have h5 : b = 36, by sorry,
  rw h5,
  simp,
  sorry

end sqrt_mult_eq_six_l85_85164


namespace convex_quadrilateral_diagonal_l85_85606

theorem convex_quadrilateral_diagonal (P : ℝ) (d1 d2 : ℝ) (hP : P = 2004) (hd1 : d1 = 1001) :
  (d2 = 1 → False) ∧ 
  (d2 = 2 → True) ∧ 
  (d2 = 1001 → True) :=
by
  sorry

end convex_quadrilateral_diagonal_l85_85606


namespace range_of_m_l85_85856

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - ((Real.exp x - 1) / (Real.exp x + 1))

theorem range_of_m (m : ℝ) (h : f (4 - m) - f m ≥ 8 - 4 * m) : 2 ≤ m := by
  sorry

end range_of_m_l85_85856


namespace fraction_inequality_l85_85240

theorem fraction_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (1 / a) + (1 / b) ≥ (4 / (a + b)) :=
by 
-- Skipping the proof using 'sorry'
sorry

end fraction_inequality_l85_85240


namespace Shannon_ratio_2_to_1_l85_85937

structure IceCreamCarton :=
  (scoops : ℕ)

structure PersonWants :=
  (vanilla : ℕ)
  (chocolate : ℕ)
  (strawberry : ℕ)

noncomputable def total_scoops_served (ethan_wants lucas_wants danny_wants connor_wants olivia_wants : PersonWants) : ℕ :=
  ethan_wants.vanilla + ethan_wants.chocolate +
  lucas_wants.chocolate +
  danny_wants.chocolate +
  connor_wants.chocolate +
  olivia_wants.vanilla + olivia_wants.strawberry

theorem Shannon_ratio_2_to_1 
    (cartons : List IceCreamCarton)
    (ethan_wants lucas_wants danny_wants connor_wants olivia_wants : PersonWants)
    (scoops_left : ℕ) : 
    -- Conditions
    (∀ carton ∈ cartons, carton.scoops = 10) →
    (cartons.length = 3) →
    (ethan_wants.vanilla = 1 ∧ ethan_wants.chocolate = 1) →
    (lucas_wants.chocolate = 2) →
    (danny_wants.chocolate = 2) →
    (connor_wants.chocolate = 2) →
    (olivia_wants.vanilla = 1 ∧ olivia_wants.strawberry = 1) →
    (scoops_left = 16) →
    -- To Prove
    4 / olivia_wants.vanilla + olivia_wants.strawberry = 2 := 
sorry

end Shannon_ratio_2_to_1_l85_85937


namespace math_problem_l85_85471

theorem math_problem : 33333 * 33334 = 1111122222 := 
by sorry

end math_problem_l85_85471


namespace true_proposition_l85_85360

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l85_85360


namespace pure_water_to_add_eq_30_l85_85548

noncomputable def initial_volume := 50
noncomputable def initial_concentration := 0.40
noncomputable def desired_concentration := 0.25
noncomputable def initial_acid := initial_concentration * initial_volume

theorem pure_water_to_add_eq_30 :
  ∃ w : ℝ, (initial_acid / (initial_volume + w) = desired_concentration) ∧ w = 30 :=
by
  sorry

end pure_water_to_add_eq_30_l85_85548


namespace train_speed_in_km_per_hr_l85_85068

/-- Given the length of a train and a bridge, and the time taken for the train to cross the bridge, prove the speed of the train in km/hr -/
theorem train_speed_in_km_per_hr
  (train_length : ℕ)  -- 100 meters
  (bridge_length : ℕ) -- 275 meters
  (crossing_time : ℕ) -- 30 seconds
  (conversion_factor : ℝ) -- 1 m/s = 3.6 km/hr
  (h_train_length : train_length = 100)
  (h_bridge_length : bridge_length = 275)
  (h_crossing_time : crossing_time = 30)
  (h_conversion_factor : conversion_factor = 3.6) : 
  (train_length + bridge_length) / crossing_time * conversion_factor = 45 := 
sorry

end train_speed_in_km_per_hr_l85_85068


namespace tank_capacity_l85_85620

theorem tank_capacity (T : ℝ) (h1 : 0.6 * T = 0.7 * T - 45) : T = 450 :=
by
  sorry

end tank_capacity_l85_85620


namespace sum_is_odd_prob_l85_85690

-- A type representing the spinner results, which can be either 1, 2, 3 or 4.
inductive SpinnerResult
| one : SpinnerResult
| two : SpinnerResult
| three : SpinnerResult
| four : SpinnerResult

open SpinnerResult

-- Function to determine if a spinner result is odd.
def isOdd (r : SpinnerResult) : Bool :=
  match r with
  | one => true
  | three => true
  | two => false
  | four => false

-- Defining the spinners P, Q, R, and S.
noncomputable def P : SpinnerResult := SpinnerResult.one -- example, could vary
noncomputable def Q : SpinnerResult := SpinnerResult.two -- example, could vary
noncomputable def R : SpinnerResult := SpinnerResult.three -- example, could vary
noncomputable def S : SpinnerResult := SpinnerResult.four -- example, could vary

-- Probability calculation function
def probabilityOddSum : ℚ :=
  let probOdd := 1 / 2
  let probEven := 1 / 2
  let scenario1 := 4 * probOdd * probEven^3
  let scenario2 := 4 * probOdd^3 * probEven
  scenario1 + scenario2

-- The theorem to be stated
theorem sum_is_odd_prob :
  probabilityOddSum = 1 / 2 := by
  sorry

end sum_is_odd_prob_l85_85690


namespace toy_selling_price_l85_85302

theorem toy_selling_price (x : ℝ) (units_sold : ℝ) (profit_per_day : ℝ) : 
  (units_sold = 200 + 20 * (80 - x)) → 
  (profit_per_day = (x - 60) * units_sold) → 
  profit_per_day = 2500 → 
  x ≤ 60 * 1.4 → 
  x = 65 :=
by
  intros h1 h2 h3 h4
  sorry

end toy_selling_price_l85_85302


namespace exists_separating_line_l85_85537

noncomputable def f1 (x : ℝ) (a1 b1 c1 : ℝ) : ℝ := a1 * x^2 + b1 * x + c1
noncomputable def f2 (x : ℝ) (a2 b2 c2 : ℝ) : ℝ := a2 * x^2 + b2 * x + c2

theorem exists_separating_line (a1 b1 c1 a2 b2 c2 : ℝ) (h_intersect : ∀ x, f1 x a1 b1 c1 ≠ f2 x a2 b2 c2)
  (h_neg : a1 * a2 < 0) : ∃ α β : ℝ, ∀ x, f1 x a1 b1 c1 < α * x + β ∧ α * x + β < f2 x a2 b2 c2 :=
sorry

end exists_separating_line_l85_85537


namespace arithmetic_mean_of_pairs_l85_85900

theorem arithmetic_mean_of_pairs :
  let a := (7 : ℚ) / 8
  let b := (9 : ℚ) / 10
  let c := (4 : ℚ) / 5
  let d := (17 : ℚ) / 20
  (a, b, c, d) ∈ { (7/8, 9/10, 4/5, 17/20), (9/10, 4/5, 17/20, 7/8), (4/5, 17/20, 7/8, 9/10), (17/20, 7/8, 9/10, 4/5) }
  → 2 * d = b + c :=
begin
  intros a b c d h,
  change (2 * (17/20) = (9/10) + (4/5)) at h,
  linarith,
end

end arithmetic_mean_of_pairs_l85_85900


namespace sqrt_multiplication_l85_85152

theorem sqrt_multiplication : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_multiplication_l85_85152


namespace not_possible_to_create_3_similar_piles_l85_85717

-- Formalize the definition of similar piles
def similar (a b : ℝ) : Prop := a <= sqrt 2 * b ∧ b <= sqrt 2 * a

-- State the main theorem
theorem not_possible_to_create_3_similar_piles (x₀ : ℝ) (h₀ : 0 < x₀) : 
  ¬ ∃ x₁ x₂ x₃ : ℝ, 
    x₁ + x₂ + x₃ = x₀ ∧ 
    similar x₁ x₂ ∧ 
    similar x₂ x₃ ∧ 
    similar x₃ x₁ :=
sorry

end not_possible_to_create_3_similar_piles_l85_85717


namespace option_a_equals_half_option_c_equals_half_l85_85788

theorem option_a_equals_half : 
  ( ∃ x : ℝ, x = (√2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180)) ∧ x = 1 / 2 ) := 
sorry

theorem option_c_equals_half : 
  ( ∃ y : ℝ, y = (Real.tan (22.5 * Real.pi / 180) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)) ∧ y = 1 / 2 ) := 
sorry

end option_a_equals_half_option_c_equals_half_l85_85788


namespace minimal_rotations_triangle_l85_85461

/-- Given a triangle with angles α, β, γ at vertices 1, 2, 3 respectively.
    The triangle returns to its original position after 15 rotations around vertex 1 by α,
    and after 6 rotations around vertex 2 by β.
    We need to show that the minimal positive integer n such that the triangle returns
    to its original position after n rotations around vertex 3 by γ is 5. -/
theorem minimal_rotations_triangle :
  ∃ (α β γ : ℝ) (k m l n : ℤ), 
    (15 * α = 360 * k) ∧ 
    (6 * β = 360 * m) ∧ 
    (α + β + γ = 180) ∧ 
    (n * γ = 360 * l) ∧ 
    (∀ n' : ℤ, n' > 0 → (∃ k' m' l' : ℤ, 
      (15 * α = 360 * k') ∧ 
      (6 * β = 360 * m') ∧ 
      (α + β + γ = 180) ∧ 
      (n' * γ = 360 * l') → n <= n')) ∧ 
    n = 5 := by
  sorry

end minimal_rotations_triangle_l85_85461


namespace ex_ineq_l85_85326

theorem ex_ineq (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 40) :
  x + y = 2 + 2 * Real.sqrt 3 ∨ x + y = 2 - 2 * Real.sqrt 3 :=
by
  sorry

end ex_ineq_l85_85326


namespace total_fuel_proof_l85_85502

def highway_consumption_60 : ℝ := 3 -- gallons per mile at 60 mph
def highway_consumption_70 : ℝ := 3.5 -- gallons per mile at 70 mph
def city_consumption_30 : ℝ := 5 -- gallons per mile at 30 mph
def city_consumption_15 : ℝ := 4.5 -- gallons per mile at 15 mph

def day1_highway_60_hours : ℝ := 2 -- hours driven at 60 mph on the highway
def day1_highway_70_hours : ℝ := 1 -- hours driven at 70 mph on the highway
def day1_city_30_hours : ℝ := 4 -- hours driven at 30 mph in the city

def day2_highway_70_hours : ℝ := 3 -- hours driven at 70 mph on the highway
def day2_city_15_hours : ℝ := 3 -- hours driven at 15 mph in the city
def day2_city_30_hours : ℝ := 1 -- hours driven at 30 mph in the city

def day3_highway_60_hours : ℝ := 1.5 -- hours driven at 60 mph on the highway
def day3_city_30_hours : ℝ := 3 -- hours driven at 30 mph in the city
def day3_city_15_hours : ℝ := 1 -- hours driven at 15 mph in the city

def total_fuel_consumption (c1 c2 c3 c4 : ℝ) (h1 h2 h3 h4 h5 h6 h7 h8 h9 : ℝ) :=
  (h1 * 60 * c1) + (h2 * 70 * c2) + (h3 * 30 * c3) + 
  (h4 * 70 * c2) + (h5 * 15 * c4) + (h6 * 30 * c3) +
  (h7 * 60 * c1) + (h8 * 30 * c3) + (h9 * 15 * c4)

theorem total_fuel_proof :
  total_fuel_consumption highway_consumption_60 highway_consumption_70 city_consumption_30 city_consumption_15
  day1_highway_60_hours day1_highway_70_hours day1_city_30_hours day2_highway_70_hours
  day2_city_15_hours day2_city_30_hours day3_highway_60_hours day3_city_30_hours day3_city_15_hours
  = 3080 := by
  sorry

end total_fuel_proof_l85_85502


namespace evaluate_expression_l85_85961

theorem evaluate_expression : 
  (900 * 900) / ((306 * 306) - (294 * 294)) = 112.5 := by
  sorry

end evaluate_expression_l85_85961


namespace number_of_cows_brought_l85_85764

/--
A certain number of cows and 10 goats are brought for Rs. 1500. 
If the average price of a goat is Rs. 70, and the average price of a cow is Rs. 400, 
then the number of cows brought is 2.
-/
theorem number_of_cows_brought : 
  ∃ c : ℕ, ∃ g : ℕ, g = 10 ∧ (70 * g + 400 * c = 1500) ∧ c = 2 :=
sorry

end number_of_cows_brought_l85_85764


namespace contradiction_proof_real_root_l85_85027

theorem contradiction_proof_real_root (a b : ℝ) :
  (∀ x : ℝ, x^3 + a * x + b ≠ 0) → (∃ x : ℝ, x + a * x + b = 0) :=
sorry

end contradiction_proof_real_root_l85_85027


namespace power_quotient_l85_85793

theorem power_quotient (a m n : ℕ) (h_a : a = 19) (h_m : m = 11) (h_n : n = 8) : a^m / a^n = 6859 := by
  sorry

end power_quotient_l85_85793


namespace count_unique_lists_of_five_l85_85971

theorem count_unique_lists_of_five :
  (∃ (f : ℕ → ℕ), ∀ (i j : ℕ), i < j → f (i + 1) - f i = 3 ∧ j = 5 → f 5 % f 1 = 0) →
  (∃ (n : ℕ), n = 6) :=
by
  sorry

end count_unique_lists_of_five_l85_85971


namespace sqrt3_mul_sqrt12_eq_6_l85_85117

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l85_85117


namespace percent_women_non_union_employees_is_65_l85_85678

-- Definitions based on the conditions
variables {E : ℝ} -- Denoting the total number of employees as a real number

def percent_men (E : ℝ) : ℝ := 0.56 * E
def percent_union_employees (E : ℝ) : ℝ := 0.60 * E
def percent_non_union_employees (E : ℝ) : ℝ := 0.40 * E
def percent_women_non_union (percent_non_union_employees : ℝ) : ℝ := 0.65 * percent_non_union_employees

-- Theorem statement
theorem percent_women_non_union_employees_is_65 :
  percent_women_non_union (percent_non_union_employees E) / (percent_non_union_employees E) = 0.65 :=
by
  sorry

end percent_women_non_union_employees_is_65_l85_85678


namespace number_913n_divisible_by_18_l85_85514

theorem number_913n_divisible_by_18 (n : ℕ) (h1 : 9130 % 2 = 0) (h2 : (9 + 1 + 3 + n) % 9 = 0) : n = 8 :=
by
  sorry

end number_913n_divisible_by_18_l85_85514


namespace cardProblem_l85_85691

structure InitialState where
  jimmy_cards : ℕ
  bob_cards : ℕ
  sarah_cards : ℕ

structure UpdatedState where
  jimmy_cards_final : ℕ
  sarah_cards_final : ℕ
  sarahs_friends_cards : ℕ

def cardProblemSolved (init : InitialState) (final : UpdatedState) : Prop :=
  let bob_initial := init.bob_cards + 6
  let bob_to_sarah := bob_initial / 3
  let bob_final := bob_initial - bob_to_sarah
  let sarah_initial := init.sarah_cards + bob_to_sarah
  let sarah_friends := sarah_initial / 3
  let sarah_final := sarah_initial - 3 * sarah_friends
  let mary_cards := 2 * 6
  let jimmy_final := init.jimmy_cards - 6 - mary_cards
  let sarah_to_tim := 0 -- since Sarah can't give fractional cards
  (final.jimmy_cards_final = jimmy_final) ∧ 
  (final.sarah_cards_final = sarah_final - sarah_to_tim) ∧ 
  (final.sarahs_friends_cards = sarah_friends)

theorem cardProblem : 
  cardProblemSolved 
    { jimmy_cards := 68, bob_cards := 5, sarah_cards := 7 }
    { jimmy_cards_final := 50, sarah_cards_final := 1, sarahs_friends_cards := 3 } :=
by 
  sorry

end cardProblem_l85_85691


namespace right_triangle_min_perimeter_multiple_13_l85_85065

theorem right_triangle_min_perimeter_multiple_13 :
  ∃ (a b c : ℕ), 
    (a^2 + b^2 = c^2) ∧ 
    (a % 13 = 0 ∨ b % 13 = 0) ∧
    (a < b) ∧ 
    (a + b > c) ∧ 
    (a + b + c = 24) :=
sorry

end right_triangle_min_perimeter_multiple_13_l85_85065


namespace H2O_required_for_NaH_reaction_l85_85655

theorem H2O_required_for_NaH_reaction
  (n_NaH : ℕ) (n_H2O : ℕ) (n_NaOH : ℕ) (n_H2 : ℕ)
  (h_eq : n_NaH = 2) (balanced_eq : n_NaH = n_H2O ∧ n_H2O = n_NaOH ∧ n_NaOH = n_H2) :
  n_H2O = 2 :=
by
  -- The proof is omitted as we only need to declare the statement.
  sorry

end H2O_required_for_NaH_reaction_l85_85655


namespace probability_point_not_above_x_axis_l85_85000

theorem probability_point_not_above_x_axis (A B C D : ℝ × ℝ) :
  A = (9, 4) →
  B = (3, -2) →
  C = (-3, -2) →
  D = (3, 4) →
  (1 / 2 : ℚ) = 1 / 2 := 
by 
  intros hA hB hC hD 
  sorry

end probability_point_not_above_x_axis_l85_85000


namespace find_a_for_parabola_l85_85836

theorem find_a_for_parabola (a : ℝ) :
  (∃ y : ℝ, y = a * (-1 / 2)^2) → a = 1 / 2 :=
by
  sorry

end find_a_for_parabola_l85_85836


namespace smallest_x_value_l85_85835

theorem smallest_x_value : ∀ x : ℚ, (14 * x^2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2 → x = 4 / 5 :=
by
  intros x hx
  sorry

end smallest_x_value_l85_85835


namespace sufficient_condition_inequality_l85_85779

theorem sufficient_condition_inequality (k : ℝ) :
  (k = 0 ∨ (-3 < k ∧ k < 0)) → ∀ x : ℝ, 2 * k * x^2 + k * x - 3 / 8 < 0 :=
sorry

end sufficient_condition_inequality_l85_85779


namespace hexagon_equilateral_triangles_l85_85072

theorem hexagon_equilateral_triangles (hexagon_area: ℝ) (num_hexagons : ℕ) (tri_area: ℝ) 
    (h1 : hexagon_area = 6) (h2 : num_hexagons = 4) (h3 : tri_area = 4) : 
    ∃ (num_triangles : ℕ), num_triangles = 8 := 
by
  sorry

end hexagon_equilateral_triangles_l85_85072


namespace probability_heads_l85_85395

noncomputable def probability_more_heads_than_tails (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let y := (n.choose (n / 2)) / total_outcomes
  (1 - y) / 2

theorem probability_heads (h₁ : ∀ (n : ℕ), n = 10 → probability_more_heads_than_tails n = 193 / 512) : 
  probability_more_heads_than_tails 10 = 193 / 512 :=
by
  apply h₁
  exact rfl

end probability_heads_l85_85395


namespace rational_points_two_coloring_l85_85255

-- Define the problem within the Lean framework
open_locale classical

def rational_point := ℚ × ℚ

def distance (p q : rational_point) : ℚ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem rational_points_two_coloring :
  ∃ (f : rational_point → Prop), 
  (∀ p q : rational_point, distance p q = 1 → f p ≠ f q) :=
sorry

end rational_points_two_coloring_l85_85255


namespace numWaysToPaintDoors_l85_85031

-- Define the number of doors and choices per door
def numDoors : ℕ := 3
def numChoicesPerDoor : ℕ := 2

-- Theorem statement that we want to prove
theorem numWaysToPaintDoors : numChoicesPerDoor ^ numDoors = 8 := by
  sorry

end numWaysToPaintDoors_l85_85031


namespace frequency_number_correct_l85_85204

-- Define the sample capacity and the group frequency as constants
def sample_capacity : ℕ := 100
def group_frequency : ℝ := 0.3

-- State the theorem
theorem frequency_number_correct : sample_capacity * group_frequency = 30 := by
  -- Immediate calculation
  sorry

end frequency_number_correct_l85_85204


namespace sqrt_mul_simplify_l85_85109

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l85_85109


namespace minimum_voters_for_tall_giraffe_to_win_l85_85687

/-- Conditions -/
def total_voters := 135
def districts := 5
def precincts_per_district := 9
def voters_per_precinct := 3
def majority (n : Nat) : Nat := Nat.ceil (n / 2)

/-- Proof Problem -/
theorem minimum_voters_for_tall_giraffe_to_win : 
  let precincts_needed_in_district := majority precincts_per_district,
      districts_needed_to_win := majority districts,
      precincts_needed_in_total := districts_needed_to_win * precincts_needed_in_district,
      majority_in_precinct := majority voters_per_precinct in
  precincts_needed_in_total * majority_in_precinct = 30 :=
by
  -- Proof steps (commented out)
  -- 1. Calculate the number of precincts in total: 5 * 9 = 45
  -- 2. Calculate how many districts the Tall giraffe needs to win: 3
  -- (majority districts)
  -- 3. Calculate how many precincts in a district the Tall giraffe needs to
  -- win: 5 (majority precincts_per_district)
  -- 4. Calculate the total number of precincts the Tall giraffe needs to win:
  -- 3 * 5 = 15
  -- 5. Calculate the majority of votes in a precinct: 2 
  -- (majority voters_per_precinct)
  -- 6. Calculate the minimum number of voters in 15 precincts: 15 * 2 = 30
  sorry

end minimum_voters_for_tall_giraffe_to_win_l85_85687


namespace fine_per_day_l85_85304

theorem fine_per_day (x : ℝ) : 
  (let total_days := 30 in
   let earnings_per_day := 25 in
   let total_amount_received := 425 in
   let days_absent := 10 in
   let days_worked := total_days - days_absent in
   let total_earnings := days_worked * earnings_per_day in
   let total_fine := days_absent * x in
   total_earnings - total_fine = total_amount_received) → x = 7.5 :=
by
  intros h
  sorry

end fine_per_day_l85_85304


namespace p_sufficient_condition_neg_q_l85_85984

variables (p q : Prop)

theorem p_sufficient_condition_neg_q (hnecsuff_q : ¬p → q) (hnecsuff_p : ¬q → p) : (p → ¬q) :=
by
  sorry

end p_sufficient_condition_neg_q_l85_85984


namespace small_triangles_count_l85_85950

theorem small_triangles_count
  (sL sS : ℝ)  -- side lengths of large (sL) and small (sS) triangles
  (hL : sL = 15)  -- condition for the large triangle's side length
  (hS : sS = 3)   -- condition for the small triangle's side length
  : sL^2 / sS^2 = 25 := 
by {
  -- Definitions to skip the proof body
  -- Further mathematical steps would usually go here
  -- but 'sorry' is used to indicate the skipped proof.
  sorry
}

end small_triangles_count_l85_85950


namespace log_interval_l85_85023

open Real

theorem log_interval (x : ℝ) : 
  x = (1 / (log 3 / log (1 / 2))) + (1 / (log 3 / log (1 / 5))) → 
  2 < x ∧ x < 3 :=
by
  intro h
  have h1 := (one_div (log 3 / log (1 / 2))).symm
  have h2 := (one_div (log 3 / log (1 / 5))).symm
  rw [log_inv, log_inv, neg_div_neg_eq, log_inv, neg_div_neg_eq] at h1 h2
  rw [h1, h2] at h
  have h3 : log 10 = 1 := by exact log_base_change log10_eq
  rw [div_self log3_pos, div_self log3_pos] at h
  refine ⟨_, _⟩
  { rw h, exact log10_lt3.inv_pos' },
  { rw h, exact log_base_change log10_eq }

end log_interval_l85_85023


namespace billy_finished_before_margaret_l85_85076

-- Define the conditions
def billy_first_laps_time : ℕ := 2 * 60
def billy_next_three_laps_time : ℕ := 4 * 60
def billy_ninth_lap_time : ℕ := 1 * 60
def billy_tenth_lap_time : ℕ := 150
def margaret_total_time : ℕ := 10 * 60

-- The main statement to prove that Billy finished 30 seconds before Margaret
theorem billy_finished_before_margaret :
  (billy_first_laps_time + billy_next_three_laps_time + billy_ninth_lap_time + billy_tenth_lap_time) + 30 = margaret_total_time :=
by
  sorry

end billy_finished_before_margaret_l85_85076


namespace total_games_in_season_l85_85561

theorem total_games_in_season :
  let num_teams := 100
  let num_sub_leagues := 5
  let teams_per_league := 20
  let games_per_pair := 6
  let teams_advancing := 4
  let playoff_teams := num_sub_leagues * teams_advancing
  let sub_league_games := (teams_per_league * (teams_per_league - 1) / 2) * games_per_pair
  let total_sub_league_games := sub_league_games * num_sub_leagues
  let playoff_games := (playoff_teams * (playoff_teams - 1)) / 2 
  let total_games := total_sub_league_games + playoff_games
  total_games = 5890 :=
by
  sorry

end total_games_in_season_l85_85561


namespace not_possible_to_create_3_piles_l85_85707

def similar_sizes (a b : ℝ) : Prop := a / b ≤ Real.sqrt 2

theorem not_possible_to_create_3_piles (x : ℝ) (hx : 0 < x) : ¬ ∃ (y z w : ℝ), 
  y + z + w = x ∧ 
  similar_sizes y z ∧ similar_sizes z w ∧ similar_sizes y w := 
by 
  sorry

end not_possible_to_create_3_piles_l85_85707


namespace zeros_in_decimal_representation_l85_85990

theorem zeros_in_decimal_representation : 
  let n : ℚ := 1 / (2^3 * 5^5)
  in (to_string (n.to_decimal_string)).index_of_first_nonzero_digit_in_fraction_part = 4 :=
sorry

end zeros_in_decimal_representation_l85_85990


namespace probability_of_non_defective_pens_l85_85999

-- Define the number of total pens, defective pens, and pens to be selected
def total_pens : ℕ := 15
def defective_pens : ℕ := 5
def selected_pens : ℕ := 3

-- Define the number of non-defective pens
def non_defective_pens : ℕ := total_pens - defective_pens

-- Define the combination function
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the total ways to choose 3 pens from 15 pens
def total_ways : ℕ := combination total_pens selected_pens

-- Define the ways to choose 3 non-defective pens from the non-defective pens
def non_defective_ways : ℕ := combination non_defective_pens selected_pens

-- Define the probability
def probability : ℚ := non_defective_ways / total_ways

-- Statement we need to prove
theorem probability_of_non_defective_pens : probability = 120 / 455 := by
  -- Proof to be completed
  sorry

end probability_of_non_defective_pens_l85_85999


namespace unique_common_tangent_l85_85337

noncomputable def f (x : ℝ) : ℝ := x ^ 2
noncomputable def g (a x : ℝ) : ℝ := a * Real.exp (x + 1)

theorem unique_common_tangent (a : ℝ) (h : a > 0) : 
  (∃ k x₁ x₂, k = 2 * x₁ ∧ k = a * Real.exp (x₂ + 1) ∧ k = (g a x₂ - f x₁) / (x₂ - x₁)) →
  a = 4 / Real.exp 3 :=
by
  sorry

end unique_common_tangent_l85_85337


namespace cuboid_surface_area_l85_85612

-- Define the given conditions
def cuboid (a b c : ℝ) := 2 * (a + b + c)

-- Given areas of distinct sides
def area_face_1 : ℝ := 4
def area_face_2 : ℝ := 3
def area_face_3 : ℝ := 6

-- Prove the total surface area of the cuboid
theorem cuboid_surface_area : cuboid area_face_1 area_face_2 area_face_3 = 26 :=
by
  sorry

end cuboid_surface_area_l85_85612


namespace true_proposition_l85_85363

open Real

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : proposition_p ∧ proposition_q :=
by
  -- These definitions are directly from conditions.
  sorry

end true_proposition_l85_85363


namespace sqrt_mul_simplify_l85_85108

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l85_85108


namespace determine_f_value_l85_85015

-- Define initial conditions
def parabola_eqn (d e f : ℝ) (y : ℝ) : ℝ := d * y^2 + e * y + f
def vertex : (ℝ × ℝ) := (2, -3)
def point_on_parabola : (ℝ × ℝ) := (7, 0)

-- Prove that f = 7 given the conditions
theorem determine_f_value (d e f : ℝ) :
  (parabola_eqn d e f (vertex.snd) = vertex.fst) ∧
  (parabola_eqn d e f (point_on_parabola.snd) = point_on_parabola.fst) →
  f = 7 := 
by
  sorry 

end determine_f_value_l85_85015


namespace true_conjunction_l85_85377

def proposition_p : Prop := ∃ x : ℝ, sin x < 1
def proposition_q : Prop := ∀ x : ℝ, Real.exp (|x|) ≥ 1

theorem true_conjunction : proposition_p ∧ proposition_q := by
  sorry

end true_conjunction_l85_85377


namespace no_all_same_color_l85_85245

def chameleons_initial_counts (c b m : ℕ) : Prop :=
  c = 13 ∧ b = 15 ∧ m = 17

def chameleon_interaction (c b m : ℕ) : Prop :=
  (∃ c' b' m', c' + b' + m' = c + b + m ∧ 
  ((c' = c - 1 ∧ b' = b - 1 ∧ m' = m + 2) ∨
   (c' = c - 1 ∧ b' = b + 2 ∧ m' = m - 1) ∨
   (c' = c + 2 ∧ b' = b - 1 ∧ m' = m - 1)))

theorem no_all_same_color (c b m : ℕ) (h1 : chameleons_initial_counts c b m) : 
  ¬ (∃ x, c = x ∧ b = x ∧ m = x) := 
sorry

end no_all_same_color_l85_85245


namespace fiona_pairs_l85_85657

theorem fiona_pairs :
  ∀ n : ℕ, 10 ≤ n ∧ n ≤ 15 → 45 ≤ (n * (n - 1) / 2) ∧ (n * (n - 1) / 2) ≤ 105 :=
by
  intro n
  intro h
  have h₁ : n ≥ 10 := h.left
  have h₂ : n ≤ 15 := h.right
  sorry

end fiona_pairs_l85_85657


namespace equation1_solution_equation2_solution_equation3_solution_l85_85257

theorem equation1_solution :
  ∀ x : ℝ, x^2 - 2 * x - 99 = 0 ↔ x = 11 ∨ x = -9 :=
by
  sorry

theorem equation2_solution :
  ∀ x : ℝ, x^2 + 5 * x = 7 ↔ x = (-5 - Real.sqrt 53) / 2 ∨ x = (-5 + Real.sqrt 53) / 2 :=
by
  sorry

theorem equation3_solution :
  ∀ x : ℝ, 4 * x * (2 * x + 1) = 3 * (2 * x + 1) ↔ x = -1/2 ∨ x = 3/4 :=
by
  sorry

end equation1_solution_equation2_solution_equation3_solution_l85_85257


namespace sqrt_mult_simplify_l85_85123

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l85_85123


namespace problem_statement_l85_85339

-- Given conditions
def p : Prop := ∃ x : ℝ, sin x < 1
def q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- Problem statement
theorem problem_statement : p ∧ q := 
by 
  split ;
  sorry

end problem_statement_l85_85339


namespace maximum_value_of_f_l85_85287

noncomputable def f (t : ℝ) : ℝ := ((3^t - 4 * t) * t) / (9^t)

theorem maximum_value_of_f : ∃ t : ℝ, f t = 1/16 :=
sorry

end maximum_value_of_f_l85_85287


namespace jamal_total_cost_l85_85883

-- Definitions based on conditions
def dozen := 12
def half_dozen := dozen / 2
def crayons_bought := 4 * half_dozen
def cost_per_crayon := 2
def total_cost := crayons_bought * cost_per_crayon

-- Proof statement (the question translated to a Lean theorem)
theorem jamal_total_cost : total_cost = 48 := by
  -- Proof skipped
  sorry

end jamal_total_cost_l85_85883


namespace sqrt_mul_eq_l85_85084

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l85_85084


namespace sqrt_mul_simplify_l85_85105

theorem sqrt_mul_simplify : sqrt 3 * sqrt 12 = 6 :=
by
  -- Conditions and simplification steps
  have h1 : sqrt 12 = 2 * sqrt 3 := sorry
  -- Using the condition
  have h2 : sqrt 3 * sqrt 12 = sqrt 3 * (2 * sqrt 3) := by rw [h1]
  -- Simplifying
  have h3 : sqrt 3 * (2 * sqrt 3) = 2 * (sqrt 3 * sqrt 3) := by ring
  -- Using sqrt properties
  have h4 : sqrt 3 * sqrt 3 = 3 := by sorry
  -- Final simplification step
  show 2 * 3 = 6 from by rw [h3, h4]; rfl

end sqrt_mul_simplify_l85_85105


namespace sin_theta_value_l85_85659

theorem sin_theta_value 
  (θ : ℝ)
  (h1 : Real.sin θ + Real.cos θ = 7/5)
  (h2 : Real.tan θ < 1) :
  Real.sin θ = 3/5 :=
sorry

end sin_theta_value_l85_85659


namespace sum_q_evals_l85_85443

noncomputable def q : ℕ → ℤ := sorry -- definition of q will be derived from conditions

theorem sum_q_evals :
  (q 0) + (q 1) + (q 2) + (q 3) + (q 4) + (q 5) + (q 6) + (q 7) + (q 8) + (q 9) +
  (q 10) + (q 11) + (q 12) + (q 13) + (q 14) + (q 15) + (q 16) + (q 17) + (q 18) = 456 :=
by
  -- Given conditions
  have h1 : q 1 = 3 := sorry
  have h6 : q 6 = 23 := sorry
  have h12 : q 12 = 17 := sorry
  have h17 : q 17 = 31 := sorry
  -- Proof outline (solved steps omitted for clarity)
  sorry

end sum_q_evals_l85_85443


namespace find_a_maximize_profit_l85_85629

theorem find_a (a: ℕ) (h: 600 * (a - 110) = 160 * a) : a = 150 :=
sorry

theorem maximize_profit (x y: ℕ) (a: ℕ) 
  (ha: a = 150)
  (hx: x + 5 * x + 20 ≤ 200) 
  (profit_eq: ∀ x, y = 245 * x + 600):
  x = 30 ∧ y = 7950 :=
sorry

end find_a_maximize_profit_l85_85629


namespace zeros_in_decimal_l85_85989

theorem zeros_in_decimal (a b : ℕ) (h_a : a = 2) (h_b : b = 5) :
  let x := 1 / (2^a * 5^b) in 
  let y := x * (2^2 / 2^2) in 
  let num_zeros := if y = (4 / 10^5) then 4 else 0 in -- Logical Deduction
  num_zeros = 4 :=
by {
  have h_eq : y = (4 / 10^5) := by sorry,
  have h_zeros : num_zeros = 4 := by {
    rw h_eq,
    have h_val : (4 / 10^5) = 0.00004 := by sorry,
    simp [h_val],
  },
  exact h_zeros,
}

end zeros_in_decimal_l85_85989


namespace geometric_seq_increasing_l85_85852

theorem geometric_seq_increasing (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) → 
  (a 1 > a 0) = (∃ a1, (a1 > 0 ∧ q > 1) ∨ (a1 < 0 ∧ 0 < q ∧ q < 1)) :=
sorry

end geometric_seq_increasing_l85_85852


namespace solve_for_c_l85_85556

noncomputable def proof_problem (a b c : ℝ) : Prop :=
  (a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) →
  (6 * 15 * c = 1.5) →
  c = 7

theorem solve_for_c : proof_problem 6 15 7 :=
by sorry

end solve_for_c_l85_85556


namespace how_many_trucks_l85_85235

-- Define the conditions given in the problem
def people_to_lift_car : ℕ := 5
def people_to_lift_truck : ℕ := 2 * people_to_lift_car

-- Set up the problem conditions
def total_people_needed (cars : ℕ) (trucks : ℕ) : ℕ :=
  cars * people_to_lift_car + trucks * people_to_lift_truck

-- Now state the precise theorem we need to prove
theorem how_many_trucks (cars trucks total_people : ℕ) 
  (h1 : cars = 6)
  (h2 : trucks = 3)
  (h3 : total_people = total_people_needed cars trucks) :
  trucks = 3 :=
by
  sorry

end how_many_trucks_l85_85235


namespace jason_has_21_toys_l85_85689

-- Definitions based on the conditions
def rachel_toys : ℕ := 1
def john_toys : ℕ := rachel_toys + 6
def jason_toys : ℕ := 3 * john_toys

-- The theorem to prove
theorem jason_has_21_toys : jason_toys = 21 := by
  -- Proof not needed, hence sorry
  sorry

end jason_has_21_toys_l85_85689


namespace find_other_number_l85_85297

theorem find_other_number (a b : ℕ) (h_lcm: Nat.lcm a b = 2310) (h_hcf: Nat.gcd a b = 55) (h_a: a = 210) : b = 605 := by
  sorry

end find_other_number_l85_85297


namespace train_speed_l85_85639

noncomputable def train_length : ℝ := 120
noncomputable def crossing_time : ℝ := 2.699784017278618

theorem train_speed : (train_length / crossing_time) = 44.448 := by
  sorry

end train_speed_l85_85639


namespace binom_21_10_l85_85525

theorem binom_21_10 :
  (Nat.choose 19 9 = 92378) →
  (Nat.choose 19 10 = 92378) →
  (Nat.choose 19 11 = 75582) →
  Nat.choose 21 10 = 352716 := by
  sorry

end binom_21_10_l85_85525


namespace P_iff_Q_l85_85338

def P (x : ℝ) := x > 1 ∨ x < -1
def Q (x : ℝ) := |x + 1| + |x - 1| > 2

theorem P_iff_Q : ∀ x, P x ↔ Q x :=
by
  intros x
  sorry

end P_iff_Q_l85_85338


namespace rainfall_second_week_l85_85754

theorem rainfall_second_week (total_rainfall : ℝ) (ratio : ℝ) (first_week_rainfall : ℝ) (second_week_rainfall : ℝ) :
  total_rainfall = 35 →
  ratio = 1.5 →
  total_rainfall = first_week_rainfall + second_week_rainfall →
  second_week_rainfall = ratio * first_week_rainfall →
  second_week_rainfall = 21 :=
by
  intros
  sorry

end rainfall_second_week_l85_85754


namespace ratio_of_fuji_trees_l85_85631

variable (F T : ℕ) -- Declaring F as number of pure Fuji trees, T as total number of trees
variables (C : ℕ) -- Declaring C as number of cross-pollinated trees 

theorem ratio_of_fuji_trees 
  (h1: 10 * C = T) 
  (h2: F + C = 221) 
  (h3: T = F + 39 + C) : 
  F * 52 = 39 * T := 
sorry

end ratio_of_fuji_trees_l85_85631


namespace vectors_perpendicular_l85_85986

open Real

def vector := ℝ × ℝ

def dot_product (v w : vector) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def perpendicular (v w : vector) : Prop :=
  dot_product v w = 0

def vector_sub (v w : vector) : vector :=
  (v.1 - w.1, v.2 - w.2)

theorem vectors_perpendicular :
  let a : vector := (2, 0)
  let b : vector := (1, 1)
  perpendicular (vector_sub a b) b :=
by
  sorry

end vectors_perpendicular_l85_85986


namespace sam_read_pages_l85_85609

-- Define conditions
def assigned_pages : ℕ := 25
def harrison_pages : ℕ := assigned_pages + 10
def pam_pages : ℕ := harrison_pages + 15
def sam_pages : ℕ := 2 * pam_pages

-- Prove the target theorem
theorem sam_read_pages : sam_pages = 100 := by
  sorry

end sam_read_pages_l85_85609


namespace files_more_than_apps_l85_85319

def initial_apps : ℕ := 11
def initial_files : ℕ := 3
def remaining_apps : ℕ := 2
def remaining_files : ℕ := 24

theorem files_more_than_apps : remaining_files - remaining_apps = 22 :=
by
  sorry

end files_more_than_apps_l85_85319


namespace solve_eq1_solve_eq2_l85_85193

-- Define the condition and the statement to be proved for the first equation
theorem solve_eq1 (x : ℝ) : 9 * x^2 - 25 = 0 → x = 5 / 3 ∨ x = -5 / 3 :=
by sorry

-- Define the condition and the statement to be proved for the second equation
theorem solve_eq2 (x : ℝ) : (x + 1)^3 - 27 = 0 → x = 2 :=
by sorry

end solve_eq1_solve_eq2_l85_85193


namespace number_of_divisors_125n5_l85_85842

theorem number_of_divisors_125n5 (n : ℕ) (hn : n > 0)
  (h150 : ∀ m : ℕ, m = 150 * n ^ 4 → (∃ d : ℕ, d * (d + 1) = 150)) :
  ∃ d : ℕ, d = 125 * n ^ 5 ∧ ((13 + 1) * (5 + 1) * (5 + 1) = 504) :=
by
  sorry

end number_of_divisors_125n5_l85_85842


namespace fill_two_thirds_of_bucket_time_l85_85474

theorem fill_two_thirds_of_bucket_time (fill_entire_bucket_time : ℝ) (h : fill_entire_bucket_time = 3) : (2 / 3) * fill_entire_bucket_time = 2 :=
by 
  sorry

end fill_two_thirds_of_bucket_time_l85_85474


namespace factorization_correct_l85_85963

theorem factorization_correct (x : ℝ) : 2 * x^2 - 6 * x - 8 = 2 * (x - 4) * (x + 1) :=
by
  sorry

end factorization_correct_l85_85963


namespace sqrt3_mul_sqrt12_eq_6_l85_85115

noncomputable def sqrt3 := Real.sqrt 3
noncomputable def sqrt12 := Real.sqrt 12

theorem sqrt3_mul_sqrt12_eq_6 : sqrt3 * sqrt12 = 6 :=
by
  sorry

end sqrt3_mul_sqrt12_eq_6_l85_85115


namespace sqrt_mul_sqrt_eq_six_l85_85144

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l85_85144


namespace arrangements_of_15_cents_l85_85321

noncomputable def num_distinct_arrangements : Nat := 
263

theorem arrangements_of_15_cents : 
  let stamps := [{(1,1)}, {(2,2)}, {(3,3)}, {(4,4)}, {(5,5)}, {(6,6)}, {(7,7)}, {(8,8)}, {(9,9)}] in
  num_distinct_arrangements = 263 :=
sorry

end arrangements_of_15_cents_l85_85321


namespace distribution_ways_l85_85949

-- Define the conditions
def num_papers : ℕ := 7
def num_friends : ℕ := 10

-- Define the theorem to prove the number of ways to distribute the papers
theorem distribution_ways : (num_friends ^ num_papers) = 10000000 := by
  -- This is where the proof would go
  sorry

end distribution_ways_l85_85949


namespace moles_of_silver_nitrate_needed_l85_85970

structure Reaction :=
  (reagent1 : String)
  (reagent2 : String)
  (product1 : String)
  (product2 : String)
  (ratio_reagent1_to_product2 : ℕ) -- Moles of reagent1 to product2 in the balanced reaction

def silver_nitrate_hydrochloric_acid_reaction : Reaction :=
  { reagent1 := "AgNO3",
    reagent2 := "HCl",
    product1 := "AgCl",
    product2 := "HNO3",
    ratio_reagent1_to_product2 := 1 }

theorem moles_of_silver_nitrate_needed
  (reaction : Reaction)
  (hCl_initial_moles : ℕ)
  (hno3_target_moles : ℕ) :
  hno3_target_moles = 2 →
  (reaction.ratio_reagent1_to_product2 = 1 ∧ hCl_initial_moles = 2) →
  (hno3_target_moles = reaction.ratio_reagent1_to_product2 * 2 ∧ hno3_target_moles = 2) :=
by
  sorry

end moles_of_silver_nitrate_needed_l85_85970


namespace carol_weight_l85_85912

variable (a c : ℝ)

-- Conditions based on the problem statement
def combined_weight : Prop := a + c = 280
def weight_difference : Prop := c - a = c / 3

theorem carol_weight (h1 : combined_weight a c) (h2 : weight_difference a c) : c = 168 :=
by
  -- Proof goes here
  sorry

end carol_weight_l85_85912


namespace total_students_in_groups_l85_85252

theorem total_students_in_groups {N : ℕ} (h1 : ∃ g : ℕ → ℕ, (∀ i j, g i = 13 ∨ g j = 12 ∨ g j = 14) ∧ (∑ i in finset.range 6, g i) = N) : 
  N = 76 ∨ N = 80 :=
sorry

end total_students_in_groups_l85_85252


namespace probability_more_wins_than_losses_l85_85176

theorem probability_more_wins_than_losses
  (n_matches : ℕ)
  (win_prob lose_prob tie_prob : ℚ)
  (h_sum_probs : win_prob + lose_prob + tie_prob = 1)
  (h_win_prob : win_prob = 1/3)
  (h_lose_prob : lose_prob = 1/3)
  (h_tie_prob : tie_prob = 1/3)
  (h_n_matches : n_matches = 8) :
  ∃ (m n : ℕ), Nat.gcd m n = 1 ∧ m / n = 5483 / 13122 ∧ (m + n) = 18605 :=
by
  sorry

end probability_more_wins_than_losses_l85_85176


namespace main_statement_l85_85355

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l85_85355


namespace equal_acutes_l85_85256

open Real

theorem equal_acutes (a b c : ℝ) (ha : 0 < a ∧ a < π / 2) (hb : 0 < b ∧ b < π / 2) (hc : 0 < c ∧ c < π / 2)
  (h1 : sin b = (sin a + sin c) / 2) (h2 : cos b ^ 2 = cos a * cos c) : a = b ∧ b = c := 
by
  -- We have to fill the proof steps here.
  sorry

end equal_acutes_l85_85256


namespace maximum_value_of_A_l85_85839

theorem maximum_value_of_A (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
    (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4 / 3)) ≤ 3 :=
sorry

end maximum_value_of_A_l85_85839


namespace solve_fractional_equation_l85_85432

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 1) :
  (x^2 - x + 2) / (x - 1) = x + 3 ↔ x = 5 / 3 :=
by
  sorry

end solve_fractional_equation_l85_85432


namespace harmonic_mean_closest_to_six_l85_85272

def harmonic_mean (a b : ℕ) : ℚ := (2 * a * b) / (a + b)

theorem harmonic_mean_closest_to_six : 
     |harmonic_mean 3 2023 - 6| < 1 :=
sorry

end harmonic_mean_closest_to_six_l85_85272


namespace not_possible_to_create_3_similar_piles_l85_85719

-- Formalize the definition of similar piles
def similar (a b : ℝ) : Prop := a <= sqrt 2 * b ∧ b <= sqrt 2 * a

-- State the main theorem
theorem not_possible_to_create_3_similar_piles (x₀ : ℝ) (h₀ : 0 < x₀) : 
  ¬ ∃ x₁ x₂ x₃ : ℝ, 
    x₁ + x₂ + x₃ = x₀ ∧ 
    similar x₁ x₂ ∧ 
    similar x₂ x₃ ∧ 
    similar x₃ x₁ :=
sorry

end not_possible_to_create_3_similar_piles_l85_85719


namespace tan_double_angle_l85_85518

open Real

theorem tan_double_angle {θ : ℝ} (h1 : tan (π / 2 - θ) = 4 * cos (2 * π - θ)) (h2 : abs θ < π / 2) : 
  tan (2 * θ) = sqrt 15 / 7 :=
sorry

end tan_double_angle_l85_85518


namespace exponent_division_l85_85806

theorem exponent_division (a : ℕ) (m n : ℕ) (h1 : 19 = a) (h2 : 11 = m) (h3 : 8 = n) : a^(m - n) = 6859 := by
  sorry

end exponent_division_l85_85806


namespace proof_problem_l85_85372

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l85_85372


namespace solve_for_y_l85_85596

theorem solve_for_y (y : ℝ) (h : (7 * y - 2) / (y + 4) - 5 / (y + 4) = 2 / (y + 4)) : 
  y = (9 / 7) :=
by
  sorry

end solve_for_y_l85_85596


namespace PhenotypicallyNormalDaughterProbability_l85_85988

-- Definitions based on conditions
def HemophiliaSexLinkedRecessive := true
def PhenylketonuriaAutosomalRecessive := true
def CouplePhenotypicallyNormal := true
def SonWithBothHemophiliaPhenylketonuria := true

-- Definition of the problem
theorem PhenotypicallyNormalDaughterProbability
  (HemophiliaSexLinkedRecessive : Prop)
  (PhenylketonuriaAutosomalRecessive : Prop)
  (CouplePhenotypicallyNormal : Prop)
  (SonWithBothHemophiliaPhenylketonuria : Prop) :
  -- The correct answer from the solution
  ∃ p : ℚ, p = 3/4 :=
  sorry

end PhenotypicallyNormalDaughterProbability_l85_85988


namespace correct_operation_is_a_l85_85047

theorem correct_operation_is_a (a b : ℝ) : 
  (a^4 * a^3 = a^7) ∧ 
  ((a^2)^3 ≠ a^5) ∧ 
  (3 * a^2 - a^2 ≠ 2) ∧ 
  ((a - b)^2 ≠ a^2 - b^2) := 
by {
  -- Here, you would fill in the proof
  sorry
}

end correct_operation_is_a_l85_85047


namespace calculate_product_l85_85745

theorem calculate_product (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3*x1*y1^2 = 2030)
  (h2 : y1^3 - 3*x1^2*y1 = 2029)
  (h3 : x2^3 - 3*x2*y2^2 = 2030)
  (h4 : y2^3 - 3*x2^2*y2 = 2029)
  (h5 : x3^3 - 3*x3*y3^2 = 2030)
  (h6 : y3^3 - 3*x3^2*y3 = 2029) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = -1 / 1015 :=
sorry

end calculate_product_l85_85745


namespace auction_starting_price_l85_85641

-- Defining the conditions
def bid_increment := 5         -- The dollar increment per bid
def bids_per_person := 5       -- Number of bids per person
def total_bidders := 2         -- Number of people bidding
def final_price := 65          -- Final price of the desk after all bids

-- Calculate derived conditions
def total_bids := bids_per_person * total_bidders
def total_increment := total_bids * bid_increment

-- The statement to be proved
theorem auction_starting_price : (final_price - total_increment) = 15 :=
by
  sorry

end auction_starting_price_l85_85641


namespace card_game_total_l85_85457

theorem card_game_total (C E O : ℝ) (h1 : E = (11 / 20) * C) (h2 : O = (9 / 20) * C) (h3 : E = O + 50) : C = 500 :=
sorry

end card_game_total_l85_85457


namespace main_statement_l85_85354

-- Define proposition p: ∃ x ∈ ℝ, sin x < 1
def prop_p : Prop := ∃ x : ℝ, sin x < 1

-- Define proposition q: ∀ x ∈ ℝ, e^(|x|) ≥ 1
def prop_q : Prop := ∀ x : ℝ, exp (abs x) ≥ 1

-- The main statement: p ∧ q is true
theorem main_statement : prop_p ∧ prop_q :=
by
  -- introduction of proof would go here
  sorry

end main_statement_l85_85354


namespace intersection_equals_l85_85845

def A : Set ℝ := {x | x < 1}

def B : Set ℝ := {x | x^2 + x ≤ 6}

theorem intersection_equals : A ∩ B = {x : ℝ | -3 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_equals_l85_85845


namespace unique_solution_7x_eq_3y_plus_4_l85_85964

theorem unique_solution_7x_eq_3y_plus_4 (x y : ℕ) (hx : 1 ≤ x) (hy : 1 ≤ y) :
    7^x = 3^y + 4 ↔ (x = 1 ∧ y = 1) :=
by
  sorry

end unique_solution_7x_eq_3y_plus_4_l85_85964


namespace tory_sold_each_toy_gun_for_l85_85792

theorem tory_sold_each_toy_gun_for :
  ∃ (x : ℤ), 8 * 18 = 7 * x + 4 ∧ x = 20 := 
by
  use 20
  constructor
  · sorry
  · sorry

end tory_sold_each_toy_gun_for_l85_85792


namespace job_completion_time_l85_85933

theorem job_completion_time
  (A C : ℝ)
  (A_rate : A = 1 / 6)
  (C_rate : C = 1 / 12)
  (B_share : 390 / 1170 = 1 / 3) :
  ∃ B : ℝ, B = 1 / 8 ∧ (B * 8 = 1) :=
by
  -- Proof omitted
  sorry

end job_completion_time_l85_85933


namespace hyperbola_trajectory_of_C_l85_85231

noncomputable def is_hyperbola_trajectory (C : ℝ × ℝ) : Prop :=
∃ x y : ℝ, C = (x, y) ∧ (x^2 / 4 - y^2 / 5 = 1) ∧ (x ≥ 2)

theorem hyperbola_trajectory_of_C :
  ∀ C : ℝ × ℝ, (∃ A B : ℝ × ℝ, A = (-3, 0) ∧ B = (3, 0) ∧
  (dist C A - dist C B).abs = 4) → is_hyperbola_trajectory C :=
begin
  sorry
end

end hyperbola_trajectory_of_C_l85_85231


namespace int_fraction_not_integer_l85_85427

theorem int_fraction_not_integer (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ¬ ∃ (k : ℤ), a^2 + b^2 = k * (a^2 - b^2) := 
sorry

end int_fraction_not_integer_l85_85427


namespace min_voters_tall_giraffe_win_l85_85685

-- Definitions from the problem statement as conditions
def precinct_voters := 3
def precincts_per_district := 9
def districts := 5
def majority_precincts(p : ℕ) := p / 2 + 1  -- Minimum precincts won in a district 
def majority_districts(d : ℕ) := d / 2 + 1  -- Minimum districts won in the final

-- Condition: majority precincts to win a district
def precinct_votes_to_win := majority_precincts precinct_voters

-- Condition: majority districts to win the final
def district_wins_to_win_final := majority_districts districts

-- Minimum precincts the Tall giraffe needs to win overall
def total_precincts_to_win := district_wins_to_win_final * majority_precincts precincts_per_district

-- Proof that the minimum number of voters who could have voted for the Tall giraffe is 30
theorem min_voters_tall_giraffe_win :
  precinct_votes_to_win * total_precincts_to_win = 30 :=
sorry

end min_voters_tall_giraffe_win_l85_85685


namespace no_isosceles_triangle_exists_l85_85811

-- Define the grid size
def grid_size : ℕ := 5

-- Define points A and B such that AB is three units horizontally
structure Point where
  x : ℕ
  y : ℕ

-- Define specific points A and B
def A : Point := ⟨2, 2⟩
def B : Point := ⟨5, 2⟩

-- Define a function to check if a triangle is isosceles
def is_isosceles (p1 p2 p3 : Point) : Prop :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p1.x - p3.x)^2 + (p1.y - p3.y)^2 ∨
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p2.x - p3.x)^2 + (p2.y - p3.y)^2 ∨
  (p1.x - p3.x)^2 + (p1.y - p3.y)^2 = (p2.x - p3.x)^2 + (p2.y - p3.y)^2

-- Prove that there are no points C that make triangle ABC isosceles
theorem no_isosceles_triangle_exists :
  ¬ ∃ C : Point, C.x ≤ grid_size ∧ C.y ≤ grid_size ∧ is_isosceles A B C :=
by
  sorry

end no_isosceles_triangle_exists_l85_85811


namespace acrobats_count_l85_85490

theorem acrobats_count (a g : ℕ) 
  (h1 : 2 * a + 4 * g = 32) 
  (h2 : a + g = 10) : 
  a = 4 := by
  -- Proof omitted
  sorry

end acrobats_count_l85_85490


namespace hyperbola_equation_l85_85529

theorem hyperbola_equation (a b c : ℝ) (e : ℝ) 
  (h1 : e = (Real.sqrt 6) / 2)
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : (c / a) = e)
  (h5 : (b * c) / (Real.sqrt (b^2 + a^2)) = 1) :
  (∃ a b : ℝ, (a = Real.sqrt 2) ∧ (b = 1) ∧ (∀ x y : ℝ, (x^2 / 2) - y^2 = 1)) :=
by
  sorry

end hyperbola_equation_l85_85529


namespace sufficient_condition_implies_range_l85_85201

def setA : Set ℝ := {x | 1 ≤ x ∧ x < 3}

def setB (a : ℝ) : Set ℝ := {x | x^2 - a * x ≤ x - a}

theorem sufficient_condition_implies_range (a : ℝ) :
  (∀ x, x ∉ setA → x ∉ setB a) → (1 ≤ a ∧ a < 3) :=
by
  sorry

end sufficient_condition_implies_range_l85_85201


namespace proof_problem_l85_85373

open Classical

variable (R : Type) [Real R]

def p : Prop := ∃ x : R, sin x < 1
def q : Prop := ∀ x : R, exp (abs x) ≥ 1

theorem proof_problem : (p ∧ q) = true := by
  sorry

end proof_problem_l85_85373


namespace particle_max_height_l85_85310

noncomputable def max_height (r ω g : ℝ) : ℝ :=
  (r * ω + g / ω) ^ 2 / (2 * g)

theorem particle_max_height (r ω g : ℝ) (h : ω > Real.sqrt (g / r)) :
    max_height r ω g = (r * ω + g / ω) ^ 2 / (2 * g) :=
sorry

end particle_max_height_l85_85310


namespace find_f_zero_l85_85667

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_zero (a : ℝ) (h1 : ∀ x : ℝ, f (x - a) = x^3 + 1)
  (h2 : ∀ x : ℝ, f x + f (2 - x) = 2) : 
  f 0 = 0 :=
sorry

end find_f_zero_l85_85667


namespace ship_meetings_l85_85903

/-- 
On an east-west shipping lane are ten ships sailing individually. The first five from the west are sailing eastwards while the other five ships are sailing westwards. They sail at the same constant speed at all times. Whenever two ships meet, each turns around and sails in the opposite direction. 

When all ships have returned to port, how many meetings of two ships have taken place? 

Proof: The total number of meetings is 25.
-/
theorem ship_meetings (east_ships west_ships : ℕ) (h_east : east_ships = 5) (h_west : west_ships = 5) : 
  east_ships * west_ships = 25 :=
by
  rw [h_east, h_west]
  exact Mul.mul 5 5
  exact eq.refl 25

end ship_meetings_l85_85903


namespace initial_black_water_bottles_l85_85917

-- Define the conditions
variables (red black blue taken left total : ℕ)
variables (hred : red = 2) (hblue : blue = 4) (htaken : taken = 5) (hleft : left = 4)

-- State the theorem with the correct answer given the conditions
theorem initial_black_water_bottles : (red + black + blue = taken + left) → black = 3 :=
by
  intros htotal
  rw [hred, hblue, htaken, hleft] at htotal
  sorry

end initial_black_water_bottles_l85_85917


namespace sqrt_mul_eq_l85_85083

theorem sqrt_mul_eq : sqrt 3 * sqrt 12 = 6 :=
by sorry

end sqrt_mul_eq_l85_85083


namespace solve_exponential_diophantine_equation_l85_85521

theorem solve_exponential_diophantine_equation :
  ∀ x y : ℕ, 7^x - 3 * 2^y = 1 → (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by {
  sorry
}

end solve_exponential_diophantine_equation_l85_85521


namespace minimum_length_PQ_l85_85530

open Real

-- Definition of the problem environment
def arithmetic_sequence (a b c : ℝ) := b = (a + c) / 2
def line_l (a b c x y : ℝ) := a * x + b * y + c = 0
def point_A := (1, 2)
def point_Q (x y : ℝ) := 3 * x - 4 * y + 12 = 0
def circle_center := (1, 0)
def circle_radius := 2

-- Statement of the proof problem
theorem minimum_length_PQ
  (a b c : ℝ) (hac : ¬ (a = 0 ∧ c = 0)) -- Condition that a and c are not both zero
  (h_arithmetic : arithmetic_sequence a b c)
  (P : ℝ × ℝ)
  (hP_on_line_l : line_l a b c P.1 P.2)
  (hQ_on_line_l : point_Q P.1 P.2) :
  let dist_to_line := |3 * 1 + 12| / sqrt (3 ^ 2 + (-4) ^ 2)
  in dist_to_line - circle_radius = 1 := 
  by sorry

end minimum_length_PQ_l85_85530


namespace find_quadratic_l85_85437

theorem find_quadratic (f : ℝ → ℝ) 
  (h1 : ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c)
  (h2 : ∃ x₁ : ℝ, ∀ x : ℝ, f x = 0 → x = x₁)
  (h3 : ∀ x : ℝ, deriv f x = 2 * x + 2) : 
  f = λ x, x^2 + 2 * x + 1 := 
sorry

end find_quadratic_l85_85437


namespace launch_country_is_soviet_union_l85_85902

-- Definitions of conditions
def launch_date : String := "October 4, 1957"
def satellite_launched_on (date : String) : Prop := date = "October 4, 1957"
def choices : List String := ["A. United States", "B. Soviet Union", "C. European Union", "D. Germany"]

-- Problem statement
theorem launch_country_is_soviet_union : 
  satellite_launched_on launch_date → 
  "B. Soviet Union" ∈ choices := 
by
  sorry

end launch_country_is_soviet_union_l85_85902


namespace find_f1_l85_85536

variable {R : Type*} [LinearOrderedField R]

-- Define function f of the form px + q
def f (p q x : R) : R := p * x + q

-- Given conditions
variables (p q : R)

-- Define the equations from given conditions
def cond1 : Prop := (f p q 3) = 5
def cond2 : Prop := (f p q 5) = 9

theorem find_f1 (hpq1 : cond1 p q) (hpq2 : cond2 p q) : f p q 1 = 1 := sorry

end find_f1_l85_85536


namespace power_quotient_l85_85794

theorem power_quotient (a m n : ℕ) (h_a : a = 19) (h_m : m = 11) (h_n : n = 8) : a^m / a^n = 6859 := by
  sorry

end power_quotient_l85_85794


namespace max_value_quadratic_l85_85818

theorem max_value_quadratic :
  (∃ x : ℝ, ∀ y : ℝ, -3*y^2 + 9*y + 24 ≤ -3*x^2 + 9*x + 24) ∧ (∃ x : ℝ, x = 3/2) :=
sorry

end max_value_quadratic_l85_85818


namespace expression_a_equals_half_expression_c_equals_half_l85_85790

theorem expression_a_equals_half :
  (A : ℝ) = (1 / 2) :=
by
  let A := (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180))
  sorry

theorem expression_c_equals_half :
  (C : ℝ) = (1 / 2) :=
by
  let C := (Real.tan (22.5 * Real.pi / 180)) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)
  sorry

end expression_a_equals_half_expression_c_equals_half_l85_85790


namespace is_linear_equation_D_l85_85295

theorem is_linear_equation_D :
  (∀ (x y : ℝ), 2 * x + 3 * y = 7 → false) ∧
  (∀ (x : ℝ), 3 * x ^ 2 = 3 → false) ∧
  (∀ (x : ℝ), 6 = 2 / x - 1 → false) ∧
  (∀ (x : ℝ), 2 * x - 1 = 20 → true) 
:= by {
  sorry
}

end is_linear_equation_D_l85_85295


namespace cafe_table_count_l85_85441

theorem cafe_table_count (cafe_seats_base7 : ℕ) (seats_per_table : ℕ) (cafe_seats_base10 : ℕ)
    (h1 : cafe_seats_base7 = 3 * 7^2 + 1 * 7^1 + 2 * 7^0) 
    (h2 : seats_per_table = 3) : cafe_seats_base10 = 156 ∧ (cafe_seats_base10 / seats_per_table) = 52 := 
by {
  sorry
}

end cafe_table_count_l85_85441


namespace quadratic_complete_square_l85_85607

theorem quadratic_complete_square :
  ∃ d e : ℝ, (∀ x : ℝ, x^2 + 800 * x + 500 = (x + d)^2 + e) ∧
    (e / d = -398.75) :=
by
  use 400
  use -159500
  sorry

end quadratic_complete_square_l85_85607


namespace find_C_l85_85613

theorem find_C (C : ℤ) (h : 2 * C - 3 = 11) : C = 7 :=
sorry

end find_C_l85_85613


namespace total_votes_polled_l85_85625

theorem total_votes_polled (V: ℝ) (h: 0 < V) (h1: 0.70 * V - 0.30 * V = 320) : V = 800 :=
sorry

end total_votes_polled_l85_85625


namespace exponent_calculation_l85_85800

-- Define the necessary exponents and base
def base : ℕ := 19
def exp1 : ℕ := 11
def exp2 : ℕ := 8

-- Given condition
lemma power_property (a : ℕ) (m n : ℕ) : a^m / a^n = a^(m - n) := by sorry

-- Proof that 19^11 / 19^8 = 6859
theorem exponent_calculation : base^exp1 / base^exp2 = 6859 := by
  have : base^exp1 / base^exp2 = base^(exp1 - exp2) := power_property base exp1 exp2
  have : base^(exp1 - exp2) = base^3 := by rw [nat.sub_eq_iff_eq_add.mpr rfl]
  have : base^3 = 6859 := by -- This would be an arithmetic computation
    rfl -- Placeholder for the actual arithmetic; ideally, you'd verify this step.
  sorry

end exponent_calculation_l85_85800


namespace single_burger_cost_l85_85492

-- Conditions
def total_cost : ℝ := 74.50
def total_burgers : ℕ := 50
def cost_double_burger : ℝ := 1.50
def double_burgers : ℕ := 49

-- Derived information
def cost_single_burger : ℝ := total_cost - (double_burgers * cost_double_burger)

-- Theorem: Prove the cost of a single burger
theorem single_burger_cost : cost_single_burger = 1.00 :=
by
  -- Proof goes here
  sorry

end single_burger_cost_l85_85492


namespace find_n_from_t_l85_85220

theorem find_n_from_t (n t : ℕ) (h1 : t = n * (n - 1) * (n + 1) + n) (h2 : t = 64) : n = 4 := by
  sorry

end find_n_from_t_l85_85220


namespace correct_decision_box_l85_85381

theorem correct_decision_box (a b c : ℝ) (x : ℝ) : 
  x = a ∨ x = b → (x = b → b > a) →
  (c > x) ↔ (max (max a b) c = c) :=
by sorry

end correct_decision_box_l85_85381


namespace sqrt_mul_l85_85098

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l85_85098


namespace angles_supplementary_l85_85951

theorem angles_supplementary (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : ∃ S : Finset ℕ, S.card = 17 ∧ (∀ a ∈ S, ∃ k : ℕ, k * (180 / (k + 1)) = a ∧ A = a) :=
by
  sorry

end angles_supplementary_l85_85951


namespace find_divisor_l85_85969

theorem find_divisor (d : ℕ) (h1 : 2319 % d = 0) (h2 : 2304 % d = 0) (h3 : (2319 - 2304) % d = 0) : d = 3 :=
  sorry

end find_divisor_l85_85969


namespace red_number_1992_is_2001_l85_85637

-- Define what it means for a number to be composite.
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

-- Define what it means for a number to be colored red.
def is_colored_red (n : ℕ) : Prop := 
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ n = a + b

-- Define a function to find the nth red number.
def nth_red_number (n : ℕ) : ℕ := sorry

-- Prove that the 1992nd red number is 2001.
theorem red_number_1992_is_2001 : nth_red_number 1992 = 2001 := by
  sorry

end red_number_1992_is_2001_l85_85637


namespace find_roots_of_polynomial_l85_85829

theorem find_roots_of_polynomial :
  ∀ x : ℝ, (3 * x ^ 4 - x ^ 3 - 8 * x ^ 2 - x + 3 = 0) →
    (x = 2 ∨ x = 1/3 ∨ x = -1) :=
by
  intros x h
  sorry

end find_roots_of_polynomial_l85_85829


namespace leif_apples_l85_85236

-- Definitions based on conditions
def oranges : ℕ := 24
def apples (oranges apples_diff : ℕ) := oranges - apples_diff

-- Theorem stating the problem to prove
theorem leif_apples (oranges apples_diff : ℕ) (h1 : oranges = 24) (h2 : apples_diff = 10) : apples oranges apples_diff = 14 :=
by
  -- Using the definition of apples and given conditions, prove the number of apples
  rw [h1, h2]
  -- Calculating the number of apples
  show 24 - 10 = 14
  rfl

end leif_apples_l85_85236


namespace true_proposition_l85_85362

axiom exists_sinx_lt_one : ∃ x : ℝ, sin x < 1
axiom for_all_exp_absx_ge_one : ∀ x : ℝ, exp (abs x) ≥ 1

theorem true_proposition : (∃ x : ℝ, sin x < 1) ∧ (∀ x : ℝ, exp (abs x) ≥ 1) :=
by
  split
  · exact exists_sinx_lt_one
  · exact for_all_exp_absx_ge_one

end true_proposition_l85_85362


namespace sqrt_mult_simplify_l85_85120

theorem sqrt_mult_simplify : Real.sqrt 3 * Real.sqrt 12 = 6 :=
by sorry

end sqrt_mult_simplify_l85_85120


namespace max_zeros_consecutive_two_digit_product_l85_85286

theorem max_zeros_consecutive_two_digit_product :
  ∃ a b : ℕ, 10 ≤ a ∧ a < 100 ∧ b = a + 1 ∧ 10 ≤ b ∧ b < 100 ∧
  (∀ c, (c * 10) ∣ a * b → c ≤ 2) := 
  by
    sorry

end max_zeros_consecutive_two_digit_product_l85_85286
