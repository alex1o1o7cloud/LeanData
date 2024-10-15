import Mathlib

namespace NUMINAMATH_GPT_sally_seashells_l486_48601

variable (M : ℝ)

theorem sally_seashells : 
  (1.20 * (M + M / 2) = 54) → M = 30 := 
by
  sorry

end NUMINAMATH_GPT_sally_seashells_l486_48601


namespace NUMINAMATH_GPT_Jake_peaches_l486_48683

variables (Jake Steven Jill : ℕ)

def peaches_relation : Prop :=
  (Jake = Steven - 6) ∧
  (Steven = Jill + 18) ∧
  (Jill = 5)

theorem Jake_peaches : peaches_relation Jake Steven Jill → Jake = 17 := by
  sorry

end NUMINAMATH_GPT_Jake_peaches_l486_48683


namespace NUMINAMATH_GPT_length_of_bridge_l486_48602

theorem length_of_bridge (train_length : ℕ) (train_speed_kmph : ℕ) (cross_time_sec : ℕ) (bridge_length: ℕ):
  train_length = 110 →
  train_speed_kmph = 45 →
  cross_time_sec = 30 →
  bridge_length = 265 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_length_of_bridge_l486_48602


namespace NUMINAMATH_GPT_total_cans_l486_48649

def bag1 := 5
def bag2 := 7
def bag3 := 12
def bag4 := 4
def bag5 := 8
def bag6 := 10

theorem total_cans : bag1 + bag2 + bag3 + bag4 + bag5 + bag6 = 46 := by
  sorry

end NUMINAMATH_GPT_total_cans_l486_48649


namespace NUMINAMATH_GPT_merchant_marked_price_l486_48697

theorem merchant_marked_price (L : ℝ) (x : ℝ) : 
  (L = 100) →
  (L - 0.3 * L = 70) →
  (0.75 * x - 70 = 0.225 * x) →
  x = 133.33 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_merchant_marked_price_l486_48697


namespace NUMINAMATH_GPT_proof_equivalent_problem_l486_48699

noncomputable def polar_equation_curve : Prop :=
  ∀ (α : ℝ), 
    let x := 3 + 2 * Real.cos α;
    let y := 1 - 2 * Real.sin α;
    (x - 3) ^ 2 + (y - 1) ^ 2 - 4 = 0

noncomputable def polar_equation_line : Prop :=
  ∀ (θ ρ : ℝ), 
  (Real.sin θ - 2 * Real.cos θ = 1 / ρ) → (2 * (ρ * Real.cos θ) - (ρ * Real.sin θ) + 1 = 0)

noncomputable def distance_from_curve_to_line : Prop :=
  ∀ (α : ℝ), 
    let x := 3 + 2 * Real.cos α;
    let y := 1 - 2 * Real.sin α;
    ∃ d : ℝ, d = (|2 * x - y + 1| / Real.sqrt (2 ^ 2 + 1)) ∧
    d + 2 = (6 * Real.sqrt 5 / 5) + 2

theorem proof_equivalent_problem :
  polar_equation_curve ∧ polar_equation_line ∧ distance_from_curve_to_line :=
by
  constructor
  · exact sorry  -- polar_equation_curve proof
  · constructor
    · exact sorry  -- polar_equation_line proof
    · exact sorry  -- distance_from_curve_to_line proof

end NUMINAMATH_GPT_proof_equivalent_problem_l486_48699


namespace NUMINAMATH_GPT_perimeter_of_figure_is_correct_l486_48630

-- Define the conditions as Lean variables and constants
def area_of_figure : ℝ := 144
def number_of_squares : ℕ := 4

-- Define the question as a theorem to be proven in Lean
theorem perimeter_of_figure_is_correct :
  let area_of_square := area_of_figure / number_of_squares
  let side_length := Real.sqrt area_of_square
  let perimeter := 9 * side_length
  perimeter = 54 :=
by
  intro area_of_square
  intro side_length
  intro perimeter
  sorry

end NUMINAMATH_GPT_perimeter_of_figure_is_correct_l486_48630


namespace NUMINAMATH_GPT_Ramesh_paid_l486_48636

theorem Ramesh_paid (P : ℝ) (h1 : 1.10 * P = 21725) : 0.80 * P + 125 + 250 = 16175 :=
by
  sorry

end NUMINAMATH_GPT_Ramesh_paid_l486_48636


namespace NUMINAMATH_GPT_fraction_of_income_from_tips_l486_48667

theorem fraction_of_income_from_tips 
  (salary tips : ℝ)
  (h1 : tips = (7/4) * salary) 
  (total_income : ℝ)
  (h2 : total_income = salary + tips) :
  (tips / total_income) = (7 / 11) :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_income_from_tips_l486_48667


namespace NUMINAMATH_GPT_complement_A_in_U_l486_48657

open Set

variable {𝕜 : Type*} [LinearOrderedField 𝕜]

def A (x : 𝕜) : Prop := |x - (1 : 𝕜)| > 2
def U : Set 𝕜 := univ

theorem complement_A_in_U : (U \ {x : 𝕜 | A x}) = {x : 𝕜 | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end NUMINAMATH_GPT_complement_A_in_U_l486_48657


namespace NUMINAMATH_GPT_savings_proof_l486_48655

variable (income expenditure savings : ℕ)

def ratio_income_expenditure (i e : ℕ) := i / 10 = e / 7

theorem savings_proof (h : ratio_income_expenditure income expenditure) (hincome : income = 10000) :
  savings = income - expenditure → savings = 3000 :=
by
  sorry

end NUMINAMATH_GPT_savings_proof_l486_48655


namespace NUMINAMATH_GPT_part1_area_quadrilateral_part2_maximized_line_equation_l486_48652

noncomputable def area_MA_NB (α : ℝ) : ℝ :=
  (352 * Real.sqrt 33) / 9 * (abs (Real.sin α - Real.cos α)) / (16 - 5 * Real.cos α ^ 2)

theorem part1_area_quadrilateral (α : ℝ) :
  area_MA_NB α = (352 * Real.sqrt 33) / 9 * (abs (Real.sin α - Real.cos α)) / (16 - 5 * Real.cos α ^ 2) :=
by sorry

theorem part2_maximized_line_equation :
  ∃ α : ℝ, area_MA_NB α = (352 * Real.sqrt 33) / 9 * (abs (Real.sin α - Real.cos α)) / (16 - 5 * Real.cos α ^ 2)
    ∧ (Real.tan α = -1 / 2) ∧ (∀ x : ℝ, x = -1 / 2 * y + Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_GPT_part1_area_quadrilateral_part2_maximized_line_equation_l486_48652


namespace NUMINAMATH_GPT_correct_tourism_model_l486_48671

noncomputable def tourism_model (x : ℕ) : ℝ :=
  80 * (Real.cos ((Real.pi / 6) * x + (2 * Real.pi / 3))) + 120

theorem correct_tourism_model :
  (∀ n : ℕ, tourism_model (n + 12) = tourism_model n) ∧
  (tourism_model 8 - tourism_model 2 = 160) ∧
  (tourism_model 2 = 40) :=
by
  sorry

end NUMINAMATH_GPT_correct_tourism_model_l486_48671


namespace NUMINAMATH_GPT_find_k_l486_48692

theorem find_k (k : ℝ) :
  ∃ k, ∀ x : ℝ, (3 * x^3 + k * x^2 - 8 * x + 52) % (3 * x + 4) = 7 :=
by
-- The proof would go here, we insert sorry to acknowledge the missing proof
sorry

end NUMINAMATH_GPT_find_k_l486_48692


namespace NUMINAMATH_GPT_inequality_proof_l486_48684

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  sqrt (a^2 + b^2 - sqrt 2 * a * b) + sqrt (b^2 + c^2 - sqrt 2 * b * c)  ≥ sqrt (a^2 + c^2) :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l486_48684


namespace NUMINAMATH_GPT_determine_a_b_l486_48631

theorem determine_a_b (a b : ℝ) :
  (∀ x, y = x^2 + a * x + b) ∧ (∀ t, t = 0 → 3 * t - (t^2 + a * t + b) + 1 = 0) →
  a = 3 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_b_l486_48631


namespace NUMINAMATH_GPT_ratio_of_ripe_mangoes_l486_48605

theorem ratio_of_ripe_mangoes (total_mangoes : ℕ) (unripe_two_thirds : ℚ)
  (kept_unripe_mangoes : ℕ) (mangoes_per_jar : ℕ) (jars_made : ℕ)
  (h1 : total_mangoes = 54)
  (h2 : unripe_two_thirds = 2 / 3)
  (h3 : kept_unripe_mangoes = 16)
  (h4 : mangoes_per_jar = 4)
  (h5 : jars_made = 5) :
  1 / 3 = 18 / 54 :=
sorry

end NUMINAMATH_GPT_ratio_of_ripe_mangoes_l486_48605


namespace NUMINAMATH_GPT_similar_polygons_area_sum_l486_48637

theorem similar_polygons_area_sum (a b c k : ℝ) (t' t'' T : ℝ)
    (h₁ : t' = k * a^2)
    (h₂ : t'' = k * b^2)
    (h₃ : T = t' + t''):
    c^2 = a^2 + b^2 := 
by 
  sorry

end NUMINAMATH_GPT_similar_polygons_area_sum_l486_48637


namespace NUMINAMATH_GPT_original_faculty_members_l486_48604

theorem original_faculty_members (X : ℝ) (H0 : X > 0) 
  (H1 : 0.75 * X ≤ X)
  (H2 : ((0.75 * X + 35) * 1.10 * 0.80 = 195)) :
  X = 253 :=
by {
  sorry
}

end NUMINAMATH_GPT_original_faculty_members_l486_48604


namespace NUMINAMATH_GPT_mil_equals_one_fortieth_mm_l486_48617

-- The condition that one mil is equal to one thousandth of an inch
def mil_in_inch := 1 / 1000

-- The condition that an inch is about 2.5 cm
def inch_in_mm := 25

-- The problem statement in Lean 4 form
theorem mil_equals_one_fortieth_mm : (mil_in_inch * inch_in_mm = 1 / 40) :=
by
  sorry

end NUMINAMATH_GPT_mil_equals_one_fortieth_mm_l486_48617


namespace NUMINAMATH_GPT_find_PS_length_l486_48676

theorem find_PS_length 
  (PT TR QS QP PQ : ℝ)
  (h1 : PT = 5)
  (h2 : TR = 10)
  (h3 : QS = 16)
  (h4 : QP = 13)
  (h5 : PQ = 7) : 
  PS = Real.sqrt 703 := 
sorry

end NUMINAMATH_GPT_find_PS_length_l486_48676


namespace NUMINAMATH_GPT_polynomial_at_3mnplus1_l486_48691

noncomputable def polynomial_value (x : ℤ) : ℤ := x^2 + 4 * x + 6

theorem polynomial_at_3mnplus1 (m n : ℤ) (h₁ : 2 * m + n + 2 = m + 2 * n) (h₂ : m - n + 2 ≠ 0) :
  polynomial_value (3 * (m + n + 1)) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_at_3mnplus1_l486_48691


namespace NUMINAMATH_GPT_candy_bar_cost_correct_l486_48610

-- Definitions based on conditions
def candy_bar_cost := 3
def chocolate_cost := candy_bar_cost + 5
def total_cost := chocolate_cost + candy_bar_cost

-- Assertion to be proved
theorem candy_bar_cost_correct :
  total_cost = 11 → candy_bar_cost = 3 :=
by
  intro h
  simp [total_cost, chocolate_cost, candy_bar_cost] at h
  sorry

end NUMINAMATH_GPT_candy_bar_cost_correct_l486_48610


namespace NUMINAMATH_GPT_work_related_emails_count_l486_48675

-- Definitions based on the identified conditions and the question
def total_emails : ℕ := 1200
def spam_percentage : ℕ := 27
def promotional_percentage : ℕ := 18
def social_percentage : ℕ := 15

-- The statement to prove, indicated the goal
theorem work_related_emails_count :
  (total_emails * (100 - spam_percentage - promotional_percentage - social_percentage)) / 100 = 480 :=
by
  sorry

end NUMINAMATH_GPT_work_related_emails_count_l486_48675


namespace NUMINAMATH_GPT_modulus_of_z_is_five_l486_48628

def z : Complex := 3 + 4 * Complex.I

theorem modulus_of_z_is_five : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_GPT_modulus_of_z_is_five_l486_48628


namespace NUMINAMATH_GPT_minimal_pieces_required_for_cubes_l486_48653

theorem minimal_pieces_required_for_cubes 
  (e₁ e₂ n₁ n₂ n₃ : ℕ)
  (h₁ : e₁ = 14)
  (h₂ : e₂ = 10)
  (h₃ : n₁ = 13)
  (h₄ : n₂ = 11)
  (h₅ : n₃ = 6)
  (disassembly_possible : ∀ {x y z : ℕ}, x^3 + y^3 = z^3 → n₁^3 + n₂^3 + n₃^3 = 14^3 + 10^3)
  (cutting_constraints : ∀ d : ℕ, (d > 0) → (d ≤ e₁ ∨ d ≤ e₂) → (d ≤ n₁ ∨ d ≤ n₂ ∨ d ≤ n₃) → (d ≤ 6))
  : ∃ minimal_pieces : ℕ, minimal_pieces = 11 := 
sorry

end NUMINAMATH_GPT_minimal_pieces_required_for_cubes_l486_48653


namespace NUMINAMATH_GPT_find_t_l486_48629

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, 3)

-- Define the perpendicular condition and solve for t
theorem find_t (t : ℝ) : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0 → t = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l486_48629


namespace NUMINAMATH_GPT_mary_total_baseball_cards_l486_48659

noncomputable def mary_initial_baseball_cards : ℕ := 18
noncomputable def torn_baseball_cards : ℕ := 8
noncomputable def fred_given_baseball_cards : ℕ := 26
noncomputable def mary_bought_baseball_cards : ℕ := 40

theorem mary_total_baseball_cards :
  mary_initial_baseball_cards - torn_baseball_cards + fred_given_baseball_cards + mary_bought_baseball_cards = 76 :=
by
  sorry

end NUMINAMATH_GPT_mary_total_baseball_cards_l486_48659


namespace NUMINAMATH_GPT_smallest_integer_in_set_l486_48627

theorem smallest_integer_in_set (median : ℤ) (greatest : ℤ) (h1 : median = 157) (h2 : greatest = 169) :
  ∃ (smallest : ℤ), smallest = 145 :=
by
  -- Setup the conditions
  have set_cons_odd : True := trivial
  -- Known facts
  have h_median : median = 157 := by exact h1
  have h_greatest : greatest = 169 := by exact h2
  -- We must prove
  existsi 145
  sorry

end NUMINAMATH_GPT_smallest_integer_in_set_l486_48627


namespace NUMINAMATH_GPT_probability_multiple_of_3_or_4_l486_48625

theorem probability_multiple_of_3_or_4 : ((15 : ℚ) / 30) = (1 / 2) := by
  sorry

end NUMINAMATH_GPT_probability_multiple_of_3_or_4_l486_48625


namespace NUMINAMATH_GPT_simplify_fraction_l486_48624

theorem simplify_fraction :
  (4 / (Real.sqrt 108 + 2 * Real.sqrt 12 + 2 * Real.sqrt 27)) = (Real.sqrt 3 / 12) := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_simplify_fraction_l486_48624


namespace NUMINAMATH_GPT_part_a_part_b_l486_48635

noncomputable def f (g n : ℕ) : ℕ := g^n + 1

theorem part_a (g : ℕ) (h_even : g % 2 = 0) (h_pos : 0 < g) :
  ∀ n : ℕ, 0 < n → f g n ∣ f g (3*n) ∧ f g n ∣ f g (5*n) ∧ f g n ∣ f g (7*n) :=
sorry

theorem part_b (g : ℕ) (h_even : g % 2 = 0) (h_pos : 0 < g) :
  ∀ n : ℕ, 0 < n → ∀ k : ℕ, 1 ≤ k → gcd (f g n) (f g (2*k*n)) = 1 :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l486_48635


namespace NUMINAMATH_GPT_solve_natural_a_l486_48600

theorem solve_natural_a (a : ℕ) : 
  (∃ n : ℕ, a^2 + a + 1589 = n^2) ↔ (a = 43 ∨ a = 28 ∨ a = 316 ∨ a = 1588) :=
sorry

end NUMINAMATH_GPT_solve_natural_a_l486_48600


namespace NUMINAMATH_GPT_correct_calculation_l486_48607

variable (a : ℝ)

theorem correct_calculation : (2 * a ^ 3) ^ 3 = 8 * a ^ 9 :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l486_48607


namespace NUMINAMATH_GPT_exists_unique_solution_l486_48640

theorem exists_unique_solution : ∀ a b : ℝ, 2 * (a ^ 2 + 1) * (b ^ 2 + 1) = (a + 1) * (b + 1) * (a * b + 1) ↔ (a, b) = (1, 1) := by
  sorry

end NUMINAMATH_GPT_exists_unique_solution_l486_48640


namespace NUMINAMATH_GPT_children_count_l486_48679

-- Define the total number of passengers on the airplane
def total_passengers : ℕ := 240

-- Define the ratio of men to women
def men_to_women_ratio : ℕ × ℕ := (3, 2)

-- Define the percentage of passengers who are either men or women
def percent_men_women : ℕ := 60

-- Define the number of children on the airplane
def number_of_children (total : ℕ) (percent : ℕ) : ℕ := 
  (total * (100 - percent)) / 100

theorem children_count :
  number_of_children total_passengers percent_men_women = 96 := by
  sorry

end NUMINAMATH_GPT_children_count_l486_48679


namespace NUMINAMATH_GPT_ratio_expression_value_l486_48639

theorem ratio_expression_value (A B C : ℚ) (h_ratio : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
by
  sorry

end NUMINAMATH_GPT_ratio_expression_value_l486_48639


namespace NUMINAMATH_GPT_base_comparison_l486_48621

theorem base_comparison : (1 * 6^1 + 2 * 6^0) > (1 * 2^2 + 0 * 2^1 + 1 * 2^0) := by
  sorry

end NUMINAMATH_GPT_base_comparison_l486_48621


namespace NUMINAMATH_GPT_parabola_translation_shift_downwards_l486_48634

theorem parabola_translation_shift_downwards :
  ∀ (x y : ℝ), (y = x^2 - 5) ↔ ((∃ (k : ℝ), k = -5 ∧ y = x^2 + k)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_translation_shift_downwards_l486_48634


namespace NUMINAMATH_GPT_solve_quadratic_complete_square_l486_48614

theorem solve_quadratic_complete_square :
  ∃ b c : ℤ, (∀ x : ℝ, (x + b)^2 = c ↔ x^2 + 6 * x - 9 = 0) ∧ b + c = 21 := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_complete_square_l486_48614


namespace NUMINAMATH_GPT_mario_total_flowers_l486_48633

-- Define the number of flowers on the first plant
def F1 : ℕ := 2

-- Define the number of flowers on the second plant as twice the first
def F2 : ℕ := 2 * F1

-- Define the number of flowers on the third plant as four times the second
def F3 : ℕ := 4 * F2

-- Prove that total number of flowers is 22
theorem mario_total_flowers : F1 + F2 + F3 = 22 := by
  -- Proof is to be filled here
  sorry

end NUMINAMATH_GPT_mario_total_flowers_l486_48633


namespace NUMINAMATH_GPT_inscribed_circle_radius_right_triangle_l486_48612

theorem inscribed_circle_radius_right_triangle : 
  ∀ (DE EF DF : ℝ), 
    DE = 6 →
    EF = 8 →
    DF = 10 →
    ∃ (r : ℝ), r = 2 :=
by
  intros DE EF DF hDE hEF hDF
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_right_triangle_l486_48612


namespace NUMINAMATH_GPT_area_of_red_region_on_larger_sphere_l486_48618

/-- 
A smooth ball with a radius of 1 cm was dipped in red paint and placed between two 
absolutely smooth concentric spheres with radii of 4 cm and 6 cm, respectively
(the ball is outside the smaller sphere but inside the larger sphere).
As the ball moves and touches both spheres, it leaves a red mark. 
After traveling a closed path, a region outlined in red with an area of 37 square centimeters is formed on the smaller sphere. 
Find the area of the region outlined in red on the larger sphere. 
The answer should be 55.5 square centimeters.
-/
theorem area_of_red_region_on_larger_sphere
  (r1 r2 r3 : ℝ)
  (A_small : ℝ)
  (h_red_small_sphere : 37 = 2 * π * r2 * (A_small / (2 * π * r2)))
  (h_red_large_sphere : 55.5 = 2 * π * r3 * (A_small / (2 * π * r2))) :
  ∃ A_large : ℝ, A_large = 55.5 :=
by
  -- Definitions and conditions
  let r1 := 1  -- radius of small ball (1 cm)
  let r2 := 4  -- radius of smaller sphere (4 cm)
  let r3 := 6  -- radius of larger sphere (6 cm)

  -- Given: A small red area is 37 cm^2 on the smaller sphere.
  let A_small := 37

  -- Proof of the relationship of the spherical caps
  sorry

end NUMINAMATH_GPT_area_of_red_region_on_larger_sphere_l486_48618


namespace NUMINAMATH_GPT_simplify_fraction_multiplication_l486_48695

theorem simplify_fraction_multiplication:
  (101 / 5050) * 50 = 1 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_multiplication_l486_48695


namespace NUMINAMATH_GPT_remainder_eq_27_l486_48644

def p (x : ℝ) : ℝ := x^4 + 2 * x^2 + 3
def a : ℝ := -2
def remainder := p (-2)
theorem remainder_eq_27 : remainder = 27 :=
by
  sorry

end NUMINAMATH_GPT_remainder_eq_27_l486_48644


namespace NUMINAMATH_GPT_monotonically_decreasing_interval_range_of_f_l486_48603

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (abs (x - 1))

theorem monotonically_decreasing_interval :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ > f x₂ := by sorry

theorem range_of_f :
  Set.range f = {y : ℝ | 0 < y ∧ y ≤ 1 } := by sorry

end NUMINAMATH_GPT_monotonically_decreasing_interval_range_of_f_l486_48603


namespace NUMINAMATH_GPT_intersection_A_B_l486_48678

noncomputable def A : Set ℝ := { x | abs (x - 1) < 2 }
noncomputable def B : Set ℝ := { x | x^2 + 3 * x - 4 < 0 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | -1 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l486_48678


namespace NUMINAMATH_GPT_total_marbles_count_l486_48694

variable (r b g : ℝ)
variable (h1 : r = 1.4 * b) (h2 : g = 1.5 * r)

theorem total_marbles_count (r b g : ℝ) (h1 : r = 1.4 * b) (h2 : g = 1.5 * r) :
  r + b + g = 3.21 * r :=
by
  sorry

end NUMINAMATH_GPT_total_marbles_count_l486_48694


namespace NUMINAMATH_GPT_meet_at_starting_point_second_time_in_minutes_l486_48650

theorem meet_at_starting_point_second_time_in_minutes :
  let racing_magic_time := 60 -- in seconds
  let charging_bull_time := 3600 / 40 -- in seconds
  let lcm_time := Nat.lcm racing_magic_time charging_bull_time -- LCM of the round times in seconds
  let answer := lcm_time / 60 -- convert seconds to minutes
  answer = 3 :=
by
  sorry

end NUMINAMATH_GPT_meet_at_starting_point_second_time_in_minutes_l486_48650


namespace NUMINAMATH_GPT_total_area_is_82_l486_48681

/-- Definition of the lengths of each segment as conditions -/
def length1 : ℤ := 7
def length2 : ℤ := 4
def length3 : ℤ := 5
def length4 : ℤ := 3
def length5 : ℤ := 2
def length6 : ℤ := 1

/-- Rectangle areas based on the given lengths -/
def area_A : ℤ := length1 * length2 -- 7 * 4
def area_B : ℤ := length3 * length2 -- 5 * 4
def area_C : ℤ := length1 * length4 -- 7 * 3
def area_D : ℤ := length3 * length5 -- 5 * 2
def area_E : ℤ := length4 * length6 -- 3 * 1

/-- The total area is the sum of all rectangle areas -/
def total_area : ℤ := area_A + area_B + area_C + area_D + area_E

/-- Theorem: The total area is 82 square units -/
theorem total_area_is_82 : total_area = 82 :=
by
  -- Proof left as an exercise
  sorry

end NUMINAMATH_GPT_total_area_is_82_l486_48681


namespace NUMINAMATH_GPT_team_card_sending_l486_48656

theorem team_card_sending (x : ℕ) (h : x * (x - 1) = 56) : x * (x - 1) = 56 := 
by 
  sorry

end NUMINAMATH_GPT_team_card_sending_l486_48656


namespace NUMINAMATH_GPT_who_plays_chess_l486_48648

def person_plays_chess (A B C : Prop) : Prop := 
  (A ∧ ¬ B ∧ ¬ C) ∨ (¬ A ∧ B ∧ ¬ C) ∨ (¬ A ∧ ¬ B ∧ C)

axiom statement_A : Prop
axiom statement_B : Prop
axiom statement_C : Prop
axiom one_statement_true : (statement_A ∧ ¬ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ ¬ statement_B ∧ statement_C)

-- Definition translating the statements made by A, B, and C
def A_plays := true
def B_not_plays := true
def A_not_plays := ¬ A_plays

-- Axiom stating that only one of A's, B's, or C's statements are true
axiom only_one_true : (statement_A ∧ ¬ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ ¬ statement_B ∧ statement_C)

-- Prove that B is the one who knows how to play Chinese chess
theorem who_plays_chess : B_plays :=
by
  -- Insert proof steps here
  sorry

end NUMINAMATH_GPT_who_plays_chess_l486_48648


namespace NUMINAMATH_GPT_that_three_digit_multiples_of_5_and_7_l486_48686

/-- 
Define the count_three_digit_multiples function, 
which counts the number of three-digit integers that are multiples of both 5 and 7.
-/
def count_three_digit_multiples : ℕ :=
  let lcm := Nat.lcm 5 7
  let first := (100 + lcm - 1) / lcm * lcm
  let last := 999 / lcm * lcm
  (last - first) / lcm + 1

/-- 
Theorem that states the number of positive three-digit integers that are multiples of both 5 and 7 is 26. 
-/
theorem three_digit_multiples_of_5_and_7 : count_three_digit_multiples = 26 := by
  sorry

end NUMINAMATH_GPT_that_three_digit_multiples_of_5_and_7_l486_48686


namespace NUMINAMATH_GPT_reading_order_l486_48645

theorem reading_order (a b c d : ℝ) 
  (h1 : a + c = b + d) 
  (h2 : a + b > c + d)
  (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c :=
by sorry

end NUMINAMATH_GPT_reading_order_l486_48645


namespace NUMINAMATH_GPT_color_dots_l486_48677

-- Define the vertices and the edges of the graph representing the figure
inductive Color : Type
| red : Color
| white : Color
| blue : Color

structure Dot :=
  (color : Color)

structure Edge :=
  (u : Dot)
  (v : Dot)

def valid_coloring (dots : List Dot) (edges : List Edge) : Prop :=
  ∀ e ∈ edges, e.u.color ≠ e.v.color

def count_colorings : Nat :=
  6 * 2

theorem color_dots (dots : List Dot) (edges : List Edge)
  (h1 : ∀ d ∈ dots, d.color = Color.red ∨ d.color = Color.white ∨ d.color = Color.blue)
  (h2 : valid_coloring dots edges) :
  count_colorings = 12 :=
by
  sorry

end NUMINAMATH_GPT_color_dots_l486_48677


namespace NUMINAMATH_GPT_simplify_fraction_l486_48606

theorem simplify_fraction :
  (1 / (1 / (Real.sqrt 2 + 1) + 1 / (Real.sqrt 5 - 2))) =
  ((Real.sqrt 2 + Real.sqrt 5 - 1) / (6 + 2 * Real.sqrt 10)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l486_48606


namespace NUMINAMATH_GPT_compute_a1d1_a2d2_a3d3_eq_1_l486_48609

theorem compute_a1d1_a2d2_a3d3_eq_1 {a1 a2 a3 d1 d2 d3 : ℝ}
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 1 := by
  sorry

end NUMINAMATH_GPT_compute_a1d1_a2d2_a3d3_eq_1_l486_48609


namespace NUMINAMATH_GPT_median_product_sum_l486_48622

-- Let's define the lengths of medians and distances from a point P to these medians
variables {s1 s2 s3 d1 d2 d3 : ℝ}

-- Define the conditions
def is_median_lengths (s1 s2 s3 : ℝ) : Prop := 
  ∃ (A B C : ℝ × ℝ), -- vertices of the triangle
    (s1 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 2) ∧
    (s2 = ((C.1 - B.1)^2 + (C.2 - B.2)^2) / 2) ∧
    (s3 = ((A.1 - C.1)^2 + (A.2 - C.2)^2) / 2)

def distances_to_medians (d1 d2 d3 : ℝ) : Prop :=
  ∃ (P A B C : ℝ × ℝ), -- point P and vertices of the triangle
    (d1 = dist P ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) ∧
    (d2 = dist P ((A.1 + C.1) / 2, (A.2 + C.2) / 2)) ∧
    (d3 = dist P ((A.1 + B.1) / 2, (A.2 + B.2) / 2))

-- The theorem which we need to prove
theorem median_product_sum (h_medians : is_median_lengths s1 s2 s3) 
  (h_distances : distances_to_medians d1 d2 d3) :
  s1 * d1 + s2 * d2 + s3 * d3 = 0 := sorry

end NUMINAMATH_GPT_median_product_sum_l486_48622


namespace NUMINAMATH_GPT_band_formation_l486_48665

theorem band_formation (r x m : ℕ) (h1 : r * x + 3 = m) (h2 : (r - 1) * (x + 2) = m) (h3 : m < 100) : m = 69 :=
by
  sorry

end NUMINAMATH_GPT_band_formation_l486_48665


namespace NUMINAMATH_GPT_johns_current_income_l486_48670

theorem johns_current_income
  (prev_income : ℝ := 1000000)
  (prev_tax_rate : ℝ := 0.20)
  (new_tax_rate : ℝ := 0.30)
  (extra_taxes_paid : ℝ := 250000) :
  ∃ (X : ℝ), 0.30 * X - 0.20 * prev_income = extra_taxes_paid ∧ X = 1500000 :=
by
  use 1500000
  -- Proof would come here
  sorry

end NUMINAMATH_GPT_johns_current_income_l486_48670


namespace NUMINAMATH_GPT_graph_symmetry_l486_48608

variable (f : ℝ → ℝ)

theorem graph_symmetry :
  (∀ x y, y = f (x - 1) ↔ ∃ x', x' = 2 - x ∧ y = f (1 - x'))
  ∧ (∀ x' y', y' = f (1 - x') ↔ ∃ x, x = 2 - x' ∧ y' = f (x - 1)) :=
sorry

end NUMINAMATH_GPT_graph_symmetry_l486_48608


namespace NUMINAMATH_GPT_trader_sold_95_pens_l486_48658

theorem trader_sold_95_pens
  (C : ℝ)   -- cost price of one pen
  (N : ℝ)   -- number of pens sold
  (h1 : 19 * C = 0.20 * N * C):  -- condition: profit from selling N pens is equal to the cost of 19 pens, with 20% gain percentage
  N = 95 := by
-- You would place the proof here.
  sorry

end NUMINAMATH_GPT_trader_sold_95_pens_l486_48658


namespace NUMINAMATH_GPT_more_boys_than_girls_l486_48662

theorem more_boys_than_girls : 
  let girls := 28.0
  let boys := 35.0
  boys - girls = 7.0 :=
by
  sorry

end NUMINAMATH_GPT_more_boys_than_girls_l486_48662


namespace NUMINAMATH_GPT_triangle_perimeter_from_medians_l486_48615

theorem triangle_perimeter_from_medians (m1 m2 m3 : ℕ) (h1 : m1 = 3) (h2 : m2 = 4) (h3 : m3 = 6) :
  ∃ (p : ℕ), p = 26 :=
by sorry

end NUMINAMATH_GPT_triangle_perimeter_from_medians_l486_48615


namespace NUMINAMATH_GPT_bids_per_person_l486_48654

theorem bids_per_person (initial_price final_price price_increase_per_bid : ℕ) (num_people : ℕ)
  (h1 : initial_price = 15) (h2 : final_price = 65) (h3 : price_increase_per_bid = 5) (h4 : num_people = 2) :
  (final_price - initial_price) / price_increase_per_bid / num_people = 5 :=
  sorry

end NUMINAMATH_GPT_bids_per_person_l486_48654


namespace NUMINAMATH_GPT_triangle_side_ratio_l486_48660

theorem triangle_side_ratio
  (α β γ : Real)
  (a b c p q r : Real)
  (h1 : (Real.tan α) / (Real.tan β) = p / q)
  (h2 : (Real.tan β) / (Real.tan γ) = q / r)
  (h3 : (Real.tan γ) / (Real.tan α) = r / p) :
  a^2 / b^2 / c^2 = (1/q + 1/r) / (1/r + 1/p) / (1/p + 1/q) := 
sorry

end NUMINAMATH_GPT_triangle_side_ratio_l486_48660


namespace NUMINAMATH_GPT_find_S6_l486_48666

noncomputable def geometric_series_nth_term (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * q^(n - 1)

noncomputable def geometric_series_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a1 * n else a1 * (1 - q^n) / (1 - q)

variables (a2 q : ℝ)

-- Conditions
axiom a_n_pos : ∀ n, n > 0 → geometric_series_nth_term a2 q n > 0
axiom q_gt_one : q > 1
axiom condition1 : geometric_series_nth_term a2 q 3 + geometric_series_nth_term a2 q 5 = 20
axiom condition2 : geometric_series_nth_term a2 q 2 * geometric_series_nth_term a2 q 6 = 64

-- Question/statement of the theorem
theorem find_S6 : geometric_series_sum 1 q 6 = 63 :=
  sorry

end NUMINAMATH_GPT_find_S6_l486_48666


namespace NUMINAMATH_GPT_ratio_r_to_pq_l486_48698

theorem ratio_r_to_pq (total : ℝ) (amount_r : ℝ) (amount_pq : ℝ) 
  (h1 : total = 9000) 
  (h2 : amount_r = 3600.0000000000005) 
  (h3 : amount_pq = total - amount_r) : 
  amount_r / amount_pq = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_r_to_pq_l486_48698


namespace NUMINAMATH_GPT_greatest_possible_int_diff_l486_48641

theorem greatest_possible_int_diff (x a y b : ℝ) 
    (hx : 3 < x ∧ x < 4) 
    (ha : 4 < a ∧ a < x) 
    (hy : 6 < y ∧ y < 8) 
    (hb : 8 < b ∧ b < y) 
    (h_ineq : a^2 + b^2 > x^2 + y^2) : 
    abs (⌊x⌋ - ⌈y⌉) = 2 :=
sorry

end NUMINAMATH_GPT_greatest_possible_int_diff_l486_48641


namespace NUMINAMATH_GPT_find_shorter_piece_length_l486_48696

noncomputable def shorter_piece_length (x : ℕ) : Prop :=
  x = 8

theorem find_shorter_piece_length : ∃ x : ℕ, (20 - x) > 0 ∧ 2 * x = (20 - x) + 4 ∧ shorter_piece_length x :=
by
  -- There exists an x that satisfies the conditions
  use 8
  -- Prove the conditions are met
  sorry

end NUMINAMATH_GPT_find_shorter_piece_length_l486_48696


namespace NUMINAMATH_GPT_find_pqr_l486_48643

variable (p q r : ℚ)

theorem find_pqr (h1 : ∃ a : ℚ, ∀ x : ℚ, (p = a) ∧ (q = -2 * a * 3) ∧ (r = a * 3 * 3 + 7) ∧ (r = 10 + 7)) :
  p + q + r = 8 + 1/3 := by
  sorry

end NUMINAMATH_GPT_find_pqr_l486_48643


namespace NUMINAMATH_GPT_trigonometric_identity_l486_48651

theorem trigonometric_identity
  (α : ℝ)
  (h : Real.sin (π / 6 - α) = 1 / 3) :
  2 * Real.cos (π / 6 + α / 2) ^ 2 - 1 = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l486_48651


namespace NUMINAMATH_GPT_jaeho_got_most_notebooks_l486_48611

-- Define the number of notebooks each friend received
def notebooks_jaehyuk : ℕ := 12
def notebooks_kyunghwan : ℕ := 3
def notebooks_jaeho : ℕ := 15

-- Define the statement proving that Jaeho received the most notebooks
theorem jaeho_got_most_notebooks : notebooks_jaeho > notebooks_jaehyuk ∧ notebooks_jaeho > notebooks_kyunghwan :=
by {
  sorry -- this is where the proof would go
}

end NUMINAMATH_GPT_jaeho_got_most_notebooks_l486_48611


namespace NUMINAMATH_GPT_values_of_d_divisible_by_13_l486_48690

def base8to10 (d : ℕ) : ℕ := 3 * 8^3 + d * 8^2 + d * 8 + 7

theorem values_of_d_divisible_by_13 (d : ℕ) (h : d ≥ 0 ∧ d < 8) :
  (1543 + 72 * d) % 13 = 0 ↔ d = 1 ∨ d = 2 :=
by sorry

end NUMINAMATH_GPT_values_of_d_divisible_by_13_l486_48690


namespace NUMINAMATH_GPT_problem_solution_l486_48646

/-- Define proposition p: ∀α∈ℝ, sin(π-α) ≠ -sin(α) -/
def p := ∀ α : ℝ, Real.sin (Real.pi - α) ≠ -Real.sin α

/-- Define proposition q: ∃x∈[0,+∞), sin(x) > x -/
def q := ∃ x : ℝ, 0 ≤ x ∧ Real.sin x > x

/-- Prove that ¬p ∨ q is a true proposition -/
theorem problem_solution : ¬p ∨ q :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l486_48646


namespace NUMINAMATH_GPT_smallest_b_for_45_b_square_l486_48673

theorem smallest_b_for_45_b_square :
  ∃ b : ℕ, b > 5 ∧ ∃ n : ℕ, 4 * b + 5 = n^2 ∧ b = 11 :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_for_45_b_square_l486_48673


namespace NUMINAMATH_GPT_ratio_a_to_c_l486_48668

theorem ratio_a_to_c (a b c d : ℚ)
  (h1 : a / b = 5 / 2)
  (h2 : c / d = 4 / 1)
  (h3 : d / b = 1 / 3) :
  a / c = 15 / 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_a_to_c_l486_48668


namespace NUMINAMATH_GPT_rate_of_interest_per_annum_l486_48672

def simple_interest (P T R : ℕ) : ℕ :=
  (P * T * R) / 100

theorem rate_of_interest_per_annum :
  let P_B := 5000
  let T_B := 2
  let P_C := 3000
  let T_C := 4
  let total_interest := 1980
  ∃ R : ℕ, 
      simple_interest P_B T_B R + simple_interest P_C T_C R = total_interest ∧
      R = 9 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_interest_per_annum_l486_48672


namespace NUMINAMATH_GPT_numeral_is_1_11_l486_48613

-- Define the numeral question and condition
def place_value_difference (a b : ℝ) : Prop :=
  10 * b - b = 99.99

-- Now we define the problem statement in Lean
theorem numeral_is_1_11 (a b : ℝ) (h : place_value_difference a b) : 
  a = 100 ∧ b = 11.11 ∧ (a - b = 99.99) :=
  sorry

end NUMINAMATH_GPT_numeral_is_1_11_l486_48613


namespace NUMINAMATH_GPT_largest_smallest_divisible_by_99_l486_48674

-- Definitions for distinct digits 3, 7, 9
def largest_number (x y z : Nat) : Nat := 100 * x + 10 * y + z
def smallest_number (x y z : Nat) : Nat := 100 * z + 10 * y + x

-- Proof problem statement
theorem largest_smallest_divisible_by_99 
  (a b c : Nat) (h : a > b ∧ b > c ∧ c > 0) : 
  ∃ (x y z : Nat), 
    (x = 9 ∧ y = 7 ∧ z = 3 ∧ largest_number x y z = 973 ∧ smallest_number x y z = 379) ∧
    99 ∣ (largest_number a b c - smallest_number a b c) :=
by
  sorry

end NUMINAMATH_GPT_largest_smallest_divisible_by_99_l486_48674


namespace NUMINAMATH_GPT_geometric_seq_a4_a7_l486_48664

variable {a : ℕ → ℝ}

def is_geometric (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, a (n + 1) = r * a n

theorem geometric_seq_a4_a7
  (h_geom : is_geometric a)
  (h_roots : ∃ a_1 a_10 : ℝ, (a 1 = a_1 ∧ a 10 = a_10) ∧ (2 * a_1 ^ 2 + 5 * a_1 + 1 = 0) ∧ (2 * a_10 ^ 2 + 5 * a_10 + 1 = 0)):
  a 4 * a 7 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_a4_a7_l486_48664


namespace NUMINAMATH_GPT_remaining_money_after_expenditures_l486_48669

def initial_amount : ℝ := 200.50
def spent_on_sweets : ℝ := 35.25
def given_to_each_friend : ℝ := 25.20

theorem remaining_money_after_expenditures :
  ((initial_amount - spent_on_sweets) - 2 * given_to_each_friend) = 114.85 :=
by
  sorry

end NUMINAMATH_GPT_remaining_money_after_expenditures_l486_48669


namespace NUMINAMATH_GPT_problem_l486_48616

theorem problem (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) 
  (h1 : 1 / a + 1 / b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) := 
by
  sorry

end NUMINAMATH_GPT_problem_l486_48616


namespace NUMINAMATH_GPT_find_h_in_standard_form_l486_48685

-- The expression to be converted
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x - 24

-- The standard form with given h value
def standard_form (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

-- The theorem statement
theorem find_h_in_standard_form :
  ∃ k : ℝ, ∀ x : ℝ, quadratic_expr x = standard_form 3 (-1.5) k x :=
by
  let a := 3
  let h := -1.5
  existsi (-30.75)
  intro x
  sorry

end NUMINAMATH_GPT_find_h_in_standard_form_l486_48685


namespace NUMINAMATH_GPT_number_of_integers_with_square_fraction_l486_48687

theorem number_of_integers_with_square_fraction : 
  ∃! (S : Finset ℤ), (∀ (n : ℤ), n ∈ S ↔ ∃ (k : ℤ), (n = 15 * k^2) ∨ (15 - n = k^2)) ∧ S.card = 2 := 
sorry

end NUMINAMATH_GPT_number_of_integers_with_square_fraction_l486_48687


namespace NUMINAMATH_GPT_math_problem_l486_48623

theorem math_problem
  (a b c : ℝ)
  (h : a / (30 - a) + b / (70 - b) + c / (80 - c) = 8) :
  6 / (30 - a) + 14 / (70 - b) + 16 / (80 - c) = 5 :=
sorry

end NUMINAMATH_GPT_math_problem_l486_48623


namespace NUMINAMATH_GPT_negation_exists_l486_48638

-- Definitions used in the conditions
def prop1 (x : ℝ) : Prop := x^2 ≥ 1
def neg_prop1 : Prop := ∃ x : ℝ, x^2 < 1

-- Statement to be proved
theorem negation_exists (h : ∀ x : ℝ, prop1 x) : neg_prop1 :=
by
  sorry

end NUMINAMATH_GPT_negation_exists_l486_48638


namespace NUMINAMATH_GPT_primes_sum_eq_2001_l486_48632

/-- If a and b are prime numbers such that a^2 + b = 2003, then a + b = 2001. -/
theorem primes_sum_eq_2001 (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h : a^2 + b = 2003) :
    a + b = 2001 := 
  sorry

end NUMINAMATH_GPT_primes_sum_eq_2001_l486_48632


namespace NUMINAMATH_GPT_median_score_interval_l486_48620

def intervals : List (Nat × Nat × Nat) :=
  [(80, 84, 20), (75, 79, 18), (70, 74, 15), (65, 69, 22), (60, 64, 14), (55, 59, 11)]

def total_students : Nat := 100

def median_interval : Nat × Nat :=
  (70, 74)

theorem median_score_interval :
  ∃ l u n, intervals = [(80, 84, 20), (75, 79, 18), (70, 74, 15), (65, 69, 22), (60, 64, 14), (55, 59, 11)]
  ∧ total_students = 100
  ∧ median_interval = (70, 74)
  ∧ ((l, u, n) ∈ intervals ∧ l ≤ 50 ∧ 50 ≤ u) :=
by
  sorry

end NUMINAMATH_GPT_median_score_interval_l486_48620


namespace NUMINAMATH_GPT_math_problem_l486_48642

theorem math_problem
  (a b c : ℚ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a * b^2 = c / a - b) :
  ( (a^2 * b^2 / c^2 - 2 / c + 1 / (a^2 * b^2) + 2 * a * b / c^2 - 2 / (a * b * c))
    / (2 / (a * b) - 2 * a * b / c)
    / (101 / c)
  ) = -1 / 202 := 
sorry

end NUMINAMATH_GPT_math_problem_l486_48642


namespace NUMINAMATH_GPT_inequality_ab_l486_48626

theorem inequality_ab (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := 
sorry

end NUMINAMATH_GPT_inequality_ab_l486_48626


namespace NUMINAMATH_GPT_GMAT_scores_ratio_l486_48663

variables (u v w : ℝ)

theorem GMAT_scores_ratio
  (h1 : u - w = (u + v + w) / 3)
  (h2 : u - v = 2 * (v - w))
  : v / u = 4 / 7 :=
sorry

end NUMINAMATH_GPT_GMAT_scores_ratio_l486_48663


namespace NUMINAMATH_GPT_fountain_water_after_25_days_l486_48682

def initial_volume : ℕ := 120
def evaporation_rate : ℕ := 8 / 10 -- Representing 0.8 gallons as 8/10
def rain_addition : ℕ := 5
def days : ℕ := 25
def rain_period : ℕ := 5

-- Calculate the amount of water after 25 days given the above conditions
theorem fountain_water_after_25_days :
  initial_volume + ((days / rain_period) * rain_addition) - (days * evaporation_rate) = 125 :=
by
  sorry

end NUMINAMATH_GPT_fountain_water_after_25_days_l486_48682


namespace NUMINAMATH_GPT_sin_135_eq_sqrt2_over_2_l486_48693

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_GPT_sin_135_eq_sqrt2_over_2_l486_48693


namespace NUMINAMATH_GPT_total_ideal_matching_sets_l486_48619

-- Definitions based on the provided problem statement
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def is_ideal_matching_set (A B : Set ℕ) : Prop := A ∩ B = {1, 3, 5}

-- Theorem statement for the total number of ideal matching sets
theorem total_ideal_matching_sets : ∃ n, n = 27 ∧ ∀ (A B : Set ℕ), A ⊆ U ∧ B ⊆ U ∧ is_ideal_matching_set A B → n = 27 := 
sorry

end NUMINAMATH_GPT_total_ideal_matching_sets_l486_48619


namespace NUMINAMATH_GPT_quadratic_transformation_l486_48688

theorem quadratic_transformation (x d e : ℝ) (h : x^2 - 24*x + 45 = (x+d)^2 + e) : d + e = -111 :=
sorry

end NUMINAMATH_GPT_quadratic_transformation_l486_48688


namespace NUMINAMATH_GPT_maximum_interval_length_l486_48680

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem maximum_interval_length 
  (m n : ℕ)
  (h1 : 0 < m)
  (h2 : m < n)
  (h3 : ∃ k : ℕ, ∀ i : ℕ, 0 ≤ i → i < k → ¬ is_multiple_of (m + i) 2000 ∧ (m + i) % 2021 = 0):
  n - m = 1999 :=
sorry

end NUMINAMATH_GPT_maximum_interval_length_l486_48680


namespace NUMINAMATH_GPT_find_function_α_l486_48661

theorem find_function_α (α : ℝ) (hα : 0 < α) 
  (f : ℕ+ → ℝ) (h : ∀ k m : ℕ+, α * m ≤ k ∧ k < (α + 1) * m → f (k + m) = f k + f m) :
  ∃ b : ℝ, ∀ n : ℕ+, f n = b * n :=
sorry

end NUMINAMATH_GPT_find_function_α_l486_48661


namespace NUMINAMATH_GPT_raisin_cookies_difference_l486_48689

-- Definitions based on conditions:
def raisin_cookies_baked_yesterday : ℕ := 300
def raisin_cookies_baked_today : ℕ := 280

-- Proof statement:
theorem raisin_cookies_difference : raisin_cookies_baked_yesterday - raisin_cookies_baked_today = 20 := 
by
  sorry

end NUMINAMATH_GPT_raisin_cookies_difference_l486_48689


namespace NUMINAMATH_GPT_original_number_abc_l486_48647

theorem original_number_abc (a b c : ℕ)
  (h : 100 * a + 10 * b + c = 528)
  (N : ℕ)
  (h1 : N + (100 * a + 10 * b + c) = 222 * (a + b + c))
  (hN : N = 2670) :
  100 * a + 10 * b + c = 528 := by
  sorry

end NUMINAMATH_GPT_original_number_abc_l486_48647
