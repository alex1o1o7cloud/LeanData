import Mathlib

namespace NUMINAMATH_CALUDE_involutive_function_property_l1253_125347

/-- A function f that is its own inverse -/
def InvolutiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (f x) = x

/-- The main theorem -/
theorem involutive_function_property
  (a b c d : ℝ)
  (hb : b ≠ 0)
  (hd : d ≠ 0)
  (h_c_a : 3 * c^2 = 2 * a^2)
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = (2*a*x + b) / (3*c*x + d))
  (h_involutive : InvolutiveFunction f) :
  2*a + 3*d = -4*a := by
sorry

end NUMINAMATH_CALUDE_involutive_function_property_l1253_125347


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_two_l1253_125315

theorem sum_of_coefficients_equals_two (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + 
    a₄*(x - 1)^4 + a₅*(x - 1)^5 + a₆*(x - 1)^6 + a₇*(x - 1)^7 + a₈*(x - 1)^8 + 
    a₉*(x - 1)^9 + a₁₀*(x - 1)^10 + a₁₁*(x - 1)^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_two_l1253_125315


namespace NUMINAMATH_CALUDE_average_age_increase_l1253_125378

theorem average_age_increase (n : ℕ) (A : ℝ) : 
  n = 10 → 
  ((n * A + 21 + 21 - 10 - 12) / n) - A = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l1253_125378


namespace NUMINAMATH_CALUDE_abs_T_equals_1024_l1253_125388

-- Define the complex number i
def i : ℂ := Complex.I

-- Define T as in the problem
def T : ℂ := (1 + i)^19 - (1 - i)^19

-- Theorem statement
theorem abs_T_equals_1024 : Complex.abs T = 1024 := by sorry

end NUMINAMATH_CALUDE_abs_T_equals_1024_l1253_125388


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1253_125318

def U : Set Int := {1, -2, 3, -4, 5, -6}
def M : Set Int := {1, -2, 3, -4}

theorem complement_of_M_in_U : Mᶜ = {5, -6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1253_125318


namespace NUMINAMATH_CALUDE_cube_inequality_l1253_125373

theorem cube_inequality (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l1253_125373


namespace NUMINAMATH_CALUDE_jumping_contest_l1253_125392

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_jump = 58) :
  frog_jump - grasshopper_jump = 39 := by
  sorry

end NUMINAMATH_CALUDE_jumping_contest_l1253_125392


namespace NUMINAMATH_CALUDE_lineup_probability_l1253_125301

def total_children : ℕ := 20
def num_girls : ℕ := 11
def num_boys : ℕ := 9

theorem lineup_probability :
  let valid_arrangements := Nat.choose 14 9 + 6 * Nat.choose 13 8
  let total_arrangements := Nat.choose total_children num_boys
  (valid_arrangements : ℚ) / total_arrangements =
    probability_no_more_than_five_girls_between_first_and_last_boys :=
by
  sorry

def probability_no_more_than_five_girls_between_first_and_last_boys : ℚ :=
  (Nat.choose 14 9 + 6 * Nat.choose 13 8 : ℚ) / Nat.choose total_children num_boys

end NUMINAMATH_CALUDE_lineup_probability_l1253_125301


namespace NUMINAMATH_CALUDE_f_inv_composition_l1253_125386

-- Define the function f
def f : ℕ → ℕ
| 2 => 5
| 3 => 7
| 4 => 11
| 5 => 17
| 6 => 23
| 7 => 40  -- Extended definition
| _ => 0   -- Default case for other inputs

-- Define the inverse function f⁻¹
def f_inv : ℕ → ℕ
| 5 => 2
| 7 => 3
| 11 => 4
| 17 => 5
| 23 => 6
| 40 => 7  -- Extended definition
| _ => 0   -- Default case for other inputs

-- Theorem statement
theorem f_inv_composition : f_inv ((f_inv 23)^2 + (f_inv 5)^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_f_inv_composition_l1253_125386


namespace NUMINAMATH_CALUDE_carnival_walk_distance_l1253_125311

def total_distance : Real := 0.75
def car_to_entrance : Real := 0.33
def entrance_to_rides : Real := 0.33

theorem carnival_walk_distance : 
  total_distance - (car_to_entrance + entrance_to_rides) = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_carnival_walk_distance_l1253_125311


namespace NUMINAMATH_CALUDE_dart_board_probability_l1253_125398

/-- The probability of a dart landing in the center square of a regular octagon dart board -/
theorem dart_board_probability (s : ℝ) (h : s > 0) : 
  let octagon_area := 2 * (1 + Real.sqrt 2) * s^2
  let center_square_area := (s/2)^2
  center_square_area / octagon_area = 1 / (4 + 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_dart_board_probability_l1253_125398


namespace NUMINAMATH_CALUDE_milk_percentage_after_three_replacements_l1253_125331

/-- Represents the percentage of milk remaining after one replacement operation -/
def milk_after_one_replacement (initial_milk_percentage : Real) : Real :=
  initial_milk_percentage * 0.8

/-- Represents the percentage of milk remaining after three replacement operations -/
def milk_after_three_replacements (initial_milk_percentage : Real) : Real :=
  milk_after_one_replacement (milk_after_one_replacement (milk_after_one_replacement initial_milk_percentage))

theorem milk_percentage_after_three_replacements :
  milk_after_three_replacements 100 = 51.2 := by
  sorry

end NUMINAMATH_CALUDE_milk_percentage_after_three_replacements_l1253_125331


namespace NUMINAMATH_CALUDE_expression_evaluation_l1253_125312

theorem expression_evaluation : -24 + 12 * (10 / 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1253_125312


namespace NUMINAMATH_CALUDE_circumcircle_equation_l1253_125397

-- Define the vertices of the triangle
def A : ℝ × ℝ := (2, 2)
def B : ℝ × ℝ := (5, 3)
def C : ℝ × ℝ := (3, -1)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 1)^2 = 5

-- Theorem statement
theorem circumcircle_equation :
  (circle_equation A.1 A.2) ∧
  (circle_equation B.1 B.2) ∧
  (circle_equation C.1 C.2) ∧
  (∀ (a b r : ℝ), (
    ((A.1 - a)^2 + (A.2 - b)^2 = r^2) ∧
    ((B.1 - a)^2 + (B.2 - b)^2 = r^2) ∧
    ((C.1 - a)^2 + (C.2 - b)^2 = r^2)
  ) → a = 4 ∧ b = 1 ∧ r^2 = 5) :=
sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l1253_125397


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l1253_125321

theorem average_marks_combined_classes (n₁ n₂ : ℕ) (avg₁ avg₂ : ℚ) 
  (h₁ : n₁ = 24) (h₂ : n₂ = 50) (h₃ : avg₁ = 40) (h₄ : avg₂ = 60) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = 53.51 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l1253_125321


namespace NUMINAMATH_CALUDE_perpendicular_lines_k_values_l1253_125365

/-- Given two lines l₁ and l₂ defined by their equations, 
    this theorem states that if they are perpendicular, 
    then k must be either 0 or 3. -/
theorem perpendicular_lines_k_values 
  (k : ℝ) 
  (l₁ : ℝ → ℝ → Prop) 
  (l₂ : ℝ → ℝ → Prop) 
  (h₁ : ∀ x y, l₁ x y ↔ x + k * y - 2 * k = 0) 
  (h₂ : ∀ x y, l₂ x y ↔ k * x - (k - 2) * y + 1 = 0) 
  (h_perp : (∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₂ x₂ y₂ → (x₂ - x₁) * (y₂ - y₁) = 0)) : 
  k = 0 ∨ k = 3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_k_values_l1253_125365


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_intersection_of_A_l1253_125366

-- Part I
def A : Set (ℝ × ℝ) := {p | p.2 = p.1^2 + 2}
def B : Set (ℝ × ℝ) := {p | p.2 = 6 - p.1^2}

theorem intersection_of_A_and_B : A ∩ B = {(Real.sqrt 2, 4), (-Real.sqrt 2, 4)} := by sorry

-- Part II
def A' : Set ℝ := {y | ∃ x, y = x^2 + 2}
def B' : Set ℝ := {y | ∃ x, y = 6 - x^2}

theorem intersection_of_A'_and_B' : A' ∩ B' = {y | 2 ≤ y ∧ y ≤ 6} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_intersection_of_A_l1253_125366


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1253_125394

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1253_125394


namespace NUMINAMATH_CALUDE_remainder_9876543210_mod_101_l1253_125395

theorem remainder_9876543210_mod_101 : 9876543210 % 101 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_9876543210_mod_101_l1253_125395


namespace NUMINAMATH_CALUDE_max_similar_triangle_pairs_l1253_125390

-- Define the triangle ABC
variable (A B C : Point)

-- Define that ABC is acute-angled
def is_acute_angled (A B C : Point) : Prop := sorry

-- Define heights AL and BM
def height_AL (A L : Point) : Prop := sorry
def height_BM (B M : Point) : Prop := sorry

-- Define that LM intersects the extension of AB at point D
def LM_intersects_AB_extension (L M D : Point) : Prop := sorry

-- Define a function to count similar triangle pairs
def count_similar_triangle_pairs (A B C L M D : Point) : ℕ := sorry

-- Define that no pairs of congruent triangles are formed
def no_congruent_triangles (A B C L M D : Point) : Prop := sorry

theorem max_similar_triangle_pairs 
  (A B C L M D : Point) 
  (h1 : is_acute_angled A B C)
  (h2 : height_AL A L)
  (h3 : height_BM B M)
  (h4 : LM_intersects_AB_extension L M D)
  (h5 : no_congruent_triangles A B C L M D) :
  count_similar_triangle_pairs A B C L M D = 10 := by sorry

end NUMINAMATH_CALUDE_max_similar_triangle_pairs_l1253_125390


namespace NUMINAMATH_CALUDE_plot_length_is_sixty_l1253_125324

/-- Represents a rectangular plot with given properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ
  lengthExcess : ℝ
  lengthIsTwentyMoreThanBreadth : length = breadth + lengthExcess
  fencingCostEquation : fencingCostPerMeter * (2 * (length + breadth)) = totalFencingCost

/-- Theorem stating that under given conditions, the length of the plot is 60 meters -/
theorem plot_length_is_sixty (plot : RectangularPlot)
  (h1 : plot.lengthExcess = 20)
  (h2 : plot.fencingCostPerMeter = 26.5)
  (h3 : plot.totalFencingCost = 5300) :
  plot.length = 60 := by
  sorry


end NUMINAMATH_CALUDE_plot_length_is_sixty_l1253_125324


namespace NUMINAMATH_CALUDE_reach_50_from_49_l1253_125375

def double (n : ℕ) : ℕ := n * 2

def erase_last_digit (n : ℕ) : ℕ := n / 10

def can_reach (start target : ℕ) : Prop :=
  ∃ (sequence : List (ℕ → ℕ)), 
    (∀ f ∈ sequence, f = double ∨ f = erase_last_digit) ∧
    (sequence.foldl (λ acc f => f acc) start = target)

theorem reach_50_from_49 : can_reach 49 50 := by sorry

end NUMINAMATH_CALUDE_reach_50_from_49_l1253_125375


namespace NUMINAMATH_CALUDE_sum_of_squares_l1253_125383

theorem sum_of_squares (a b c x y z : ℝ) 
  (h1 : x/a + y/b + z/c = 5)
  (h2 : a/x + b/y + c/z = 3) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1253_125383


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l1253_125341

theorem opposite_of_negative_two :
  ∀ x : ℝ, (x + (-2) = 0) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l1253_125341


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1253_125310

theorem inverse_variation_problem (p q : ℝ) (k : ℝ) (h1 : k > 0) :
  (∀ x y, x * y = k → x > 0 → y > 0) →  -- inverse variation definition
  (1500 * 0.5 = k) →                    -- initial condition
  (3000 * q = k) →                      -- new condition
  q = 0.250 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1253_125310


namespace NUMINAMATH_CALUDE_strawberry_smoothies_l1253_125319

theorem strawberry_smoothies (initial_strawberries additional_strawberries strawberries_per_smoothie : ℚ)
  (h1 : initial_strawberries = 28)
  (h2 : additional_strawberries = 35)
  (h3 : strawberries_per_smoothie = 7.5) :
  ⌊(initial_strawberries + additional_strawberries) / strawberries_per_smoothie⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_smoothies_l1253_125319


namespace NUMINAMATH_CALUDE_phillip_initial_vinegar_l1253_125371

/-- The number of jars Phillip has -/
def num_jars : ℕ := 4

/-- The number of cucumbers Phillip has -/
def num_cucumbers : ℕ := 10

/-- The number of pickles each cucumber makes -/
def pickles_per_cucumber : ℕ := 6

/-- The number of pickles each jar can hold -/
def pickles_per_jar : ℕ := 12

/-- The amount of vinegar (in ounces) needed per jar of pickles -/
def vinegar_per_jar : ℕ := 10

/-- The amount of vinegar (in ounces) left after making pickles -/
def vinegar_left : ℕ := 60

/-- Theorem stating that Phillip started with 100 ounces of vinegar -/
theorem phillip_initial_vinegar : 
  (min num_jars ((num_cucumbers * pickles_per_cucumber) / pickles_per_jar)) * vinegar_per_jar + vinegar_left = 100 := by
  sorry

end NUMINAMATH_CALUDE_phillip_initial_vinegar_l1253_125371


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l1253_125320

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := |x - a| + |x + b|

-- State the theorem
theorem min_value_and_inequality (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 2) : 
  (a + b = 2) ∧ ¬(a^2 + a > 2 ∧ b^2 + b > 2) := by
  sorry


end NUMINAMATH_CALUDE_min_value_and_inequality_l1253_125320


namespace NUMINAMATH_CALUDE_work_completion_time_l1253_125325

theorem work_completion_time (P : ℕ) (D : ℕ) : 
  (P * D = 2 * (2 * P * 3)) → D = 12 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1253_125325


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l1253_125372

/-- The number of distinct diagonals in a convex heptagon -/
def num_diagonals_heptagon : ℕ := 14

/-- The number of sides in a heptagon -/
def heptagon_sides : ℕ := 7

/-- Theorem: The number of distinct diagonals in a convex heptagon is 14 -/
theorem heptagon_diagonals :
  num_diagonals_heptagon = (heptagon_sides * (heptagon_sides - 3)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l1253_125372


namespace NUMINAMATH_CALUDE_gcd_45_75_l1253_125360

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45_75_l1253_125360


namespace NUMINAMATH_CALUDE_correct_change_calculation_l1253_125396

/-- The change to be returned when mailing items with given costs and payment -/
def change_to_return (cost1 cost2 payment : ℚ) : ℚ :=
  payment - (cost1 + cost2)

/-- Theorem stating that the change to be returned is 1.2 yuan given the specific costs and payment -/
theorem correct_change_calculation :
  change_to_return (1.6) (12.2) (15) = (1.2) := by
  sorry

end NUMINAMATH_CALUDE_correct_change_calculation_l1253_125396


namespace NUMINAMATH_CALUDE_unique_H_value_l1253_125370

/-- Represents a digit in the addition problem -/
structure Digit :=
  (value : Nat)
  (is_valid : value < 10)

/-- Represents the addition problem -/
structure AdditionProblem :=
  (T : Digit)
  (H : Digit)
  (R : Digit)
  (E : Digit)
  (F : Digit)
  (I : Digit)
  (V : Digit)
  (S : Digit)
  (all_different : T ≠ H ∧ T ≠ R ∧ T ≠ E ∧ T ≠ F ∧ T ≠ I ∧ T ≠ V ∧ T ≠ S ∧
                   H ≠ R ∧ H ≠ E ∧ H ≠ F ∧ H ≠ I ∧ H ≠ V ∧ H ≠ S ∧
                   R ≠ E ∧ R ≠ F ∧ R ≠ I ∧ R ≠ V ∧ R ≠ S ∧
                   E ≠ F ∧ E ≠ I ∧ E ≠ V ∧ E ≠ S ∧
                   F ≠ I ∧ F ≠ V ∧ F ≠ S ∧
                   I ≠ V ∧ I ≠ S ∧
                   V ≠ S)
  (T_is_eight : T.value = 8)
  (E_is_odd : E.value % 2 = 1)
  (addition_valid : F.value * 10000 + I.value * 1000 + V.value * 100 + E.value * 10 + S.value =
                    (T.value * 1000 + H.value * 100 + R.value * 10 + E.value) * 2)

theorem unique_H_value (p : AdditionProblem) : p.H.value = 7 :=
  sorry

end NUMINAMATH_CALUDE_unique_H_value_l1253_125370


namespace NUMINAMATH_CALUDE_father_son_age_ratio_l1253_125369

def father_son_ages (son_age : ℕ) (age_difference : ℕ) : Prop :=
  ∃ (k : ℕ), (son_age + age_difference + 2) = k * (son_age + 2)

theorem father_son_age_ratio :
  let son_age : ℕ := 22
  let age_difference : ℕ := 24
  father_son_ages son_age age_difference →
  (son_age + age_difference + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_father_son_age_ratio_l1253_125369


namespace NUMINAMATH_CALUDE_simple_annual_interest_rate_l1253_125333

/-- Simple annual interest rate calculation -/
theorem simple_annual_interest_rate 
  (monthly_interest : ℝ) 
  (investment_amount : ℝ) 
  (h1 : monthly_interest = 225)
  (h2 : investment_amount = 30000) : 
  (monthly_interest * 12) / investment_amount = 0.09 := by
sorry

end NUMINAMATH_CALUDE_simple_annual_interest_rate_l1253_125333


namespace NUMINAMATH_CALUDE_value_of_expression_l1253_125328

-- Define the function g
def g (p q r s t : ℝ) (x : ℝ) : ℝ := p * x^4 + q * x^3 + r * x^2 + s * x + t

-- State the theorem
theorem value_of_expression (p q r s t : ℝ) 
  (h : g p q r s t (-1) = 4) : 
  12 * p - 6 * q + 3 * r - 2 * s + t = 13 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1253_125328


namespace NUMINAMATH_CALUDE_complex_sum_nonzero_components_l1253_125338

theorem complex_sum_nonzero_components (a b : ℝ) :
  (a : ℂ) + b * Complex.I = (1 - Complex.I)^10 + (1 + Complex.I)^10 →
  a ≠ 0 ∧ b ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_nonzero_components_l1253_125338


namespace NUMINAMATH_CALUDE_bill_total_l1253_125350

/-- Proves that if three people divide a bill evenly and each pays $33, then the total bill is $99. -/
theorem bill_total (people : Fin 3 → ℕ) (h : ∀ i, people i = 33) : 
  (Finset.univ.sum people) = 99 := by
  sorry

end NUMINAMATH_CALUDE_bill_total_l1253_125350


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_value_l1253_125389

/-- The function f(x) = x + 1/x - a*ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + 1/x - a * Real.log x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 - 1/x^2 - a/x

theorem tangent_line_implies_a_value (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ = x₀ + 1 ∧ f_deriv a x₀ = 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_value_l1253_125389


namespace NUMINAMATH_CALUDE_role_assignment_count_l1253_125307

def num_men : ℕ := 6
def num_women : ℕ := 7
def num_male_roles : ℕ := 3
def num_female_roles : ℕ := 3
def num_neutral_roles : ℕ := 2

def total_roles : ℕ := num_male_roles + num_female_roles + num_neutral_roles

theorem role_assignment_count : 
  (num_men.factorial / (num_men - num_male_roles).factorial) *
  (num_women.factorial / (num_women - num_female_roles).factorial) *
  ((num_men + num_women - num_male_roles - num_female_roles).factorial / 
   (num_men + num_women - total_roles).factorial) = 1058400 := by
  sorry

end NUMINAMATH_CALUDE_role_assignment_count_l1253_125307


namespace NUMINAMATH_CALUDE_second_number_value_l1253_125387

theorem second_number_value (x : ℝ) (h : 8000 * x = 480 * (10^5)) : x = 6000 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l1253_125387


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l1253_125303

theorem round_trip_average_speed
  (distance_north : Real)
  (speed_north : Real)
  (rest_time : Real)
  (speed_south : Real)
  (h1 : distance_north = 5280 / 5280) -- 1 mile in feet
  (h2 : speed_north = 3) -- minutes per mile
  (h3 : rest_time = 10) -- minutes
  (h4 : speed_south = 3) -- miles per minute
  : (2 * distance_north) / ((distance_north * speed_north + rest_time + distance_north / speed_south) / 60) = 9 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l1253_125303


namespace NUMINAMATH_CALUDE_cubic_root_conditions_l1253_125356

theorem cubic_root_conditions (a b c d : ℝ) (ha : a ≠ 0) 
  (h_roots : ∀ z : ℂ, a * z^3 + b * z^2 + c * z + d = 0 → z.re < 0) :
  ab > 0 ∧ bc - ad > 0 ∧ ad > 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_conditions_l1253_125356


namespace NUMINAMATH_CALUDE_painted_cube_probability_l1253_125367

/-- Represents a 5x5x5 cube with three adjacent faces painted -/
structure PaintedCube :=
  (size : Nat)
  (painted_faces : Nat)

/-- Calculates the number of unit cubes with exactly three painted faces -/
def three_painted_faces (cube : PaintedCube) : Nat :=
  8  -- 8 vertices of the cube

/-- Calculates the number of unit cubes with exactly one painted face -/
def one_painted_face (cube : PaintedCube) : Nat :=
  27  -- 9 cubes per face * 3 painted faces

/-- Calculates the total number of ways to choose two unit cubes -/
def total_choices (cube : PaintedCube) : Nat :=
  (cube.size ^ 3) * (cube.size ^ 3 - 1) / 2

/-- Theorem: The probability of selecting one unit cube with exactly three painted faces
    and another unit cube with exactly one painted face from a 5x5x5 cube with
    three adjacent faces painted is 24/775 -/
theorem painted_cube_probability (cube : PaintedCube)
  (h1 : cube.size = 5)
  (h2 : cube.painted_faces = 3) :
  (three_painted_faces cube * one_painted_face cube : ℚ) / total_choices cube = 24 / 775 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_probability_l1253_125367


namespace NUMINAMATH_CALUDE_average_of_combined_results_l1253_125355

theorem average_of_combined_results :
  let n₁ : ℕ := 40
  let avg₁ : ℚ := 30
  let n₂ : ℕ := 30
  let avg₂ : ℚ := 40
  let total_sum : ℚ := n₁ * avg₁ + n₂ * avg₂
  let total_count : ℕ := n₁ + n₂
  (total_sum / total_count : ℚ) = 2400 / 70 :=
by sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l1253_125355


namespace NUMINAMATH_CALUDE_bathroom_length_proof_l1253_125351

/-- Proves the length of a rectangular bathroom given its width, tile size, and number of tiles needed --/
theorem bathroom_length_proof (width : ℝ) (tile_side : ℝ) (num_tiles : ℕ) (length : ℝ) : 
  width = 6 →
  tile_side = 0.5 →
  num_tiles = 240 →
  width * length = (tile_side * tile_side) * num_tiles →
  length = 10 := by
sorry

end NUMINAMATH_CALUDE_bathroom_length_proof_l1253_125351


namespace NUMINAMATH_CALUDE_smallest_positive_quadratic_form_l1253_125346

def quadratic_form (x y : ℤ) : ℤ := 20 * x^2 + 80 * x * y + 95 * y^2

theorem smallest_positive_quadratic_form :
  (∃ x y : ℤ, quadratic_form x y = 67) ∧
  (∀ n : ℕ, n > 0 → n < 67 → ∀ x y : ℤ, quadratic_form x y ≠ n) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_quadratic_form_l1253_125346


namespace NUMINAMATH_CALUDE_quadratic_equation_range_l1253_125343

theorem quadratic_equation_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - x + a = 0) → a ≥ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_range_l1253_125343


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_ratio_l1253_125304

theorem arithmetic_geometric_progression_ratio (a₁ a₂ a₃ a₄ d : ℝ) :
  a₁ ≠ 0 → a₂ ≠ 0 → a₃ ≠ 0 → a₄ ≠ 0 → d ≠ 0 →
  (∃ r : ℝ, a₂ = a₁ + d ∧ a₃ = a₁ + 2*d ∧ a₄ = a₁ + 3*d) →
  a₃^2 = a₁ * a₄ →
  d / a₁ = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_ratio_l1253_125304


namespace NUMINAMATH_CALUDE_binomial_expansion_103_l1253_125330

theorem binomial_expansion_103 : 
  103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_103_l1253_125330


namespace NUMINAMATH_CALUDE_two_roots_implies_c_values_l1253_125376

-- Define the function f(x) = x³ - 3x + c
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

-- State the theorem
theorem two_roots_implies_c_values (c : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f c x₁ = 0 ∧ f c x₂ = 0 ∧
    (∀ x : ℝ, f c x = 0 → x = x₁ ∨ x = x₂)) →
  c = -2 ∨ c = 2 := by
sorry

end NUMINAMATH_CALUDE_two_roots_implies_c_values_l1253_125376


namespace NUMINAMATH_CALUDE_integral_3x_plus_sinx_l1253_125382

theorem integral_3x_plus_sinx (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x + Real.sin x) :
  ∫ x in (0)..(Real.pi / 2), f x = (3 / 8) * Real.pi^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_3x_plus_sinx_l1253_125382


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1253_125361

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 2 then 2 / (2 - x) else 0

theorem unique_function_satisfying_conditions :
  (∀ x y, x ≥ 0 → y ≥ 0 → f (x * f y) * f y = f (x + y)) ∧
  (f 2 = 0) ∧
  (∀ x, 0 ≤ x → x < 2 → f x ≠ 0) ∧
  (∀ g : ℝ → ℝ, (∀ x y, x ≥ 0 → y ≥ 0 → g (x * g y) * g y = g (x + y)) →
    (g 2 = 0) →
    (∀ x, 0 ≤ x → x < 2 → g x ≠ 0) →
    (∀ x, x ≥ 0 → g x = f x)) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1253_125361


namespace NUMINAMATH_CALUDE_cans_ratio_theorem_l1253_125332

/-- Represents the number of cans collected by each person -/
structure CansCollected where
  solomon : ℕ
  juwan : ℕ
  levi : ℕ

/-- Represents a ratio between two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- The theorem to be proved -/
theorem cans_ratio_theorem (c : CansCollected) 
  (h1 : c.solomon = 66)
  (h2 : c.solomon + c.juwan + c.levi = 99)
  (h3 : c.levi = c.juwan / 2)
  : Ratio.mk 3 1 = Ratio.mk c.solomon c.juwan := by
  sorry

#check cans_ratio_theorem

end NUMINAMATH_CALUDE_cans_ratio_theorem_l1253_125332


namespace NUMINAMATH_CALUDE_solution_set_xfx_less_than_zero_l1253_125385

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_increasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

theorem solution_set_xfx_less_than_zero
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_increasing : is_increasing_on_positive f)
  (h_f_neg_three : f (-3) = 0) :
  {x : ℝ | x * f x < 0} = {x | x < -3 ∨ x > 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_xfx_less_than_zero_l1253_125385


namespace NUMINAMATH_CALUDE_point_on_y_axis_l1253_125329

/-- A point lies on the y-axis if and only if its x-coordinate is 0 -/
def lies_on_y_axis (x y : ℝ) : Prop := x = 0

/-- The x-coordinate of point P -/
def x_coord (m : ℝ) : ℝ := 6 - 2*m

/-- The y-coordinate of point P -/
def y_coord (m : ℝ) : ℝ := 4 - m

/-- Theorem: If the point P(6-2m, 4-m) lies on the y-axis, then m = 3 -/
theorem point_on_y_axis (m : ℝ) : lies_on_y_axis (x_coord m) (y_coord m) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l1253_125329


namespace NUMINAMATH_CALUDE_jan_water_collection_l1253_125344

theorem jan_water_collection :
  ∀ (initial_water : ℕ) 
    (car_water : ℕ) 
    (plant_water : ℕ) 
    (plates_clothes_water : ℕ),
  car_water = 7 * 2 →
  plant_water = car_water - 11 →
  plates_clothes_water = 24 →
  plates_clothes_water * 2 = initial_water - (car_water + plant_water) →
  initial_water = 65 :=
by
  sorry


end NUMINAMATH_CALUDE_jan_water_collection_l1253_125344


namespace NUMINAMATH_CALUDE_min_sum_squares_l1253_125335

theorem min_sum_squares (x y z : ℝ) (h : 2*x + 3*y + 4*z = 10) :
  x^2 + y^2 + z^2 ≥ 100/29 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1253_125335


namespace NUMINAMATH_CALUDE_expression_equals_one_l1253_125308

theorem expression_equals_one :
  (120^2 - 9^2) / (90^2 - 18^2) * ((90-18)*(90+18)) / ((120-9)*(120+9)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1253_125308


namespace NUMINAMATH_CALUDE_function_properties_l1253_125374

/-- Given a function f and a positive real number ω, proves properties about f -/
theorem function_properties (f : ℝ → ℝ) (ω : ℝ) (h_ω : ω > 0) 
  (h_f : ∀ x, f x = Real.sqrt 3 * Real.sin (2 * ω * x) - Real.cos (2 * ω * x))
  (h_dist : ∀ x, f (x + π / (4 * ω)) = f x) : 
  (∀ x, f x = 2 * Real.sin (2 * x - π / 6)) ∧ 
  (∀ x ∈ Set.Icc 0 (π / 2), f x ≤ 2) := by
sorry

end NUMINAMATH_CALUDE_function_properties_l1253_125374


namespace NUMINAMATH_CALUDE_kidney_apples_amount_l1253_125354

/-- The amount of golden apples in kg -/
def golden_apples : ℕ := 37

/-- The amount of Canada apples in kg -/
def canada_apples : ℕ := 14

/-- The amount of apples sold in kg -/
def apples_sold : ℕ := 36

/-- The amount of apples left in kg -/
def apples_left : ℕ := 38

/-- The amount of kidney apples in kg -/
def kidney_apples : ℕ := 23

theorem kidney_apples_amount :
  kidney_apples = apples_left + apples_sold - golden_apples - canada_apples :=
by sorry

end NUMINAMATH_CALUDE_kidney_apples_amount_l1253_125354


namespace NUMINAMATH_CALUDE_exactly_one_not_through_origin_l1253_125391

def f₁ (x : ℝ) : ℝ := x^4 + 1
def f₂ (x : ℝ) : ℝ := x^4 + x
def f₃ (x : ℝ) : ℝ := x^4 + x^2
def f₄ (x : ℝ) : ℝ := x^4 + x^3

def passes_through_origin (f : ℝ → ℝ) : Prop := f 0 = 0

theorem exactly_one_not_through_origin :
  ∃! i : Fin 4, ¬passes_through_origin (match i with
    | 0 => f₁
    | 1 => f₂
    | 2 => f₃
    | 3 => f₄) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_not_through_origin_l1253_125391


namespace NUMINAMATH_CALUDE_three_divisors_iff_prime_square_l1253_125353

/-- A natural number has exactly three distinct divisors if and only if it is the square of a prime number. -/
theorem three_divisors_iff_prime_square (n : ℕ) : (∃! (s : Finset ℕ), s.card = 3 ∧ ∀ d ∈ s, d ∣ n) ↔ ∃ p, Nat.Prime p ∧ n = p^2 := by
  sorry

end NUMINAMATH_CALUDE_three_divisors_iff_prime_square_l1253_125353


namespace NUMINAMATH_CALUDE_greater_number_problem_l1253_125313

theorem greater_number_problem (x y : ℝ) (h1 : x > y) (h2 : x > 0) (h3 : y > 0) 
  (h4 : x * y = 2048) (h5 : (x + y) - (x - y) = 64) : x = 64 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l1253_125313


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l1253_125399

theorem square_plus_inverse_square (x : ℝ) (h : x^4 + 1/x^4 = 2) : x^2 + 1/x^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l1253_125399


namespace NUMINAMATH_CALUDE_complex_magnitude_l1253_125326

theorem complex_magnitude (z : ℂ) (a b : ℝ) (h1 : z = Complex.mk a b) (h2 : a ≠ 0) 
  (h3 : Complex.abs z ^ 2 - 2 * z = Complex.mk 1 2) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1253_125326


namespace NUMINAMATH_CALUDE_distance_between_intersection_points_l1253_125300

/-- The curve C in rectangular coordinates -/
def curve_C (x y : ℝ) : Prop := x^2 = 4*y

/-- The line l in rectangular coordinates -/
def line_l (x y : ℝ) : Prop := y = x + 1

/-- The intersection points of curve C and line l -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | curve_C p.1 p.2 ∧ line_l p.1 p.2}

theorem distance_between_intersection_points :
  ∃ (p q : ℝ × ℝ), p ∈ intersection_points ∧ q ∈ intersection_points ∧ p ≠ q ∧
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 8 :=
sorry

end NUMINAMATH_CALUDE_distance_between_intersection_points_l1253_125300


namespace NUMINAMATH_CALUDE_certain_number_proof_l1253_125336

theorem certain_number_proof (x y a : ℤ) 
  (eq1 : 4 * x + y = a) 
  (eq2 : 2 * x - y = 20) 
  (y_squared : y^2 = 4) : 
  a = 46 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1253_125336


namespace NUMINAMATH_CALUDE_raffle_ticket_sales_l1253_125327

theorem raffle_ticket_sales (total_avg : ℝ) (male_avg : ℝ) (female_avg : ℝ) :
  total_avg = 66 →
  male_avg = 58 →
  (1 : ℝ) * male_avg + 2 * female_avg = 3 * total_avg →
  female_avg = 70 := by
  sorry

end NUMINAMATH_CALUDE_raffle_ticket_sales_l1253_125327


namespace NUMINAMATH_CALUDE_R_equals_triangle_interior_l1253_125337

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The polynomial z^2 + az + b -/
def polynomial (a b : ℝ) (z : ℂ) : ℂ := z^2 + a*z + b

/-- The region R -/
def R : Set (ℝ × ℝ) :=
  {p | ∀ z, polynomial p.1 p.2 z = 0 → Complex.abs z < 1}

/-- The triangle ABC -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {p | p.1 > -2 ∧ p.1 < 2 ∧ p.2 > -1 ∧ p.2 < 1 ∧ p.2 < (1 - p.1/2)}

/-- The theorem stating that R is equivalent to the interior of triangle ABC -/
theorem R_equals_triangle_interior : R = triangle_ABC := by sorry

end NUMINAMATH_CALUDE_R_equals_triangle_interior_l1253_125337


namespace NUMINAMATH_CALUDE_probability_not_monday_l1253_125322

theorem probability_not_monday (p_monday : ℚ) (h : p_monday = 1/7) : 
  1 - p_monday = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_monday_l1253_125322


namespace NUMINAMATH_CALUDE_no_solution_exists_l1253_125305

theorem no_solution_exists : ¬ ∃ x : ℝ, Real.arccos (4/5) - Real.arccos (-4/5) = Real.arcsin x := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1253_125305


namespace NUMINAMATH_CALUDE_strawberry_supply_theorem_l1253_125363

/-- Represents the weekly strawberry requirements for each bakery -/
structure BakeryRequirements where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of sacks needed for all bakeries over a given period -/
def totalSacks (req : BakeryRequirements) (weeks : ℕ) : ℕ :=
  (req.first + req.second + req.third) * weeks

/-- The problem statement -/
theorem strawberry_supply_theorem (req : BakeryRequirements) (weeks : ℕ) 
    (h1 : req.first = 2)
    (h2 : req.second = 4)
    (h3 : req.third = 12)
    (h4 : weeks = 4) :
  totalSacks req weeks = 72 := by
  sorry

#check strawberry_supply_theorem

end NUMINAMATH_CALUDE_strawberry_supply_theorem_l1253_125363


namespace NUMINAMATH_CALUDE_ednas_neighbors_l1253_125384

/-- The number of cookies Edna made -/
def total_cookies : ℕ := 150

/-- The number of cookies each neighbor (except Sarah) took -/
def cookies_per_neighbor : ℕ := 10

/-- The number of cookies Sarah took -/
def sarah_cookies : ℕ := 12

/-- The number of cookies left for the last neighbor -/
def cookies_left : ℕ := 8

/-- The number of Edna's neighbors -/
def num_neighbors : ℕ := 14

theorem ednas_neighbors :
  total_cookies = num_neighbors * cookies_per_neighbor + (sarah_cookies - cookies_per_neighbor) + cookies_left :=
by sorry

end NUMINAMATH_CALUDE_ednas_neighbors_l1253_125384


namespace NUMINAMATH_CALUDE_difference_between_point_eight_and_one_eighth_l1253_125334

theorem difference_between_point_eight_and_one_eighth (ε : ℝ) :
  0.8 - (1 / 8 : ℝ) = 0.675 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_point_eight_and_one_eighth_l1253_125334


namespace NUMINAMATH_CALUDE_triangle_area_implies_q_value_l1253_125345

/-- Given a triangle DEF with vertices D(3, 15), E(15, 0), and F(0, q),
    if the area of the triangle is 30, then q = 12.5 -/
theorem triangle_area_implies_q_value :
  ∀ q : ℝ,
  let D : ℝ × ℝ := (3, 15)
  let E : ℝ × ℝ := (15, 0)
  let F : ℝ × ℝ := (0, q)
  let triangle_area := abs ((3 * q + 15 * q - 45) / 2)
  triangle_area = 30 → q = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_implies_q_value_l1253_125345


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l1253_125309

theorem quadratic_root_transformation (k ℓ : ℝ) (r₁ r₂ : ℝ) : 
  (r₁^2 + k*r₁ + ℓ = 0) → 
  (r₂^2 + k*r₂ + ℓ = 0) → 
  ∃ v : ℝ, r₁^2^2 + (-k^2 + 2*ℓ)*r₁^2 + v = 0 ∧ r₂^2^2 + (-k^2 + 2*ℓ)*r₂^2 + v = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l1253_125309


namespace NUMINAMATH_CALUDE_problem_solution_l1253_125349

theorem problem_solution (a n : ℕ) (h1 : a = 105) (h2 : a^3 = n * 49 * 45 * 25) : n = 21 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1253_125349


namespace NUMINAMATH_CALUDE_stratified_sampling_ratio_l1253_125306

theorem stratified_sampling_ratio 
  (total_first : ℕ) 
  (total_second : ℕ) 
  (sample_first : ℕ) 
  (sample_second : ℕ) 
  (h1 : total_first = 400) 
  (h2 : total_second = 360) 
  (h3 : sample_first = 60) : 
  (sample_first : ℚ) / total_first = (sample_second : ℚ) / total_second → 
  sample_second = 54 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_ratio_l1253_125306


namespace NUMINAMATH_CALUDE_triangle_sqrt_inequality_l1253_125348

theorem triangle_sqrt_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  Real.sqrt a + Real.sqrt b > Real.sqrt c := by
  sorry

end NUMINAMATH_CALUDE_triangle_sqrt_inequality_l1253_125348


namespace NUMINAMATH_CALUDE_marbleSelectionWays_l1253_125342

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 2 marbles from 3 pairs of different colored marbles -/
def twoFromThreePairs : ℕ := sorry

/-- The total number of marbles -/
def totalMarbles : ℕ := 15

/-- The number of special colored marbles (red, green, blue) -/
def specialColoredMarbles : ℕ := 6

/-- The number of marbles to be chosen -/
def marblesToChoose : ℕ := 5

/-- The number of special colored marbles that must be chosen -/
def specialMarblesToChoose : ℕ := 2

theorem marbleSelectionWays : 
  twoFromThreePairs * choose (totalMarbles - specialColoredMarbles) (marblesToChoose - specialMarblesToChoose) = 1008 :=
sorry

end NUMINAMATH_CALUDE_marbleSelectionWays_l1253_125342


namespace NUMINAMATH_CALUDE_value_of_m_l1253_125359

theorem value_of_m (a b m : ℝ) 
  (h1 : 2^a = m) 
  (h2 : 5^b = m) 
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_value_of_m_l1253_125359


namespace NUMINAMATH_CALUDE_three_teachers_three_students_arrangements_l1253_125302

/-- The number of arrangements for teachers and students in a row --/
def arrangements (num_teachers num_students : ℕ) : ℕ :=
  (num_teachers + 1).factorial * num_students.factorial

/-- Theorem: The number of arrangements for 3 teachers and 3 students,
    where no two students are adjacent, is 144 --/
theorem three_teachers_three_students_arrangements :
  arrangements 3 3 = 144 :=
sorry

end NUMINAMATH_CALUDE_three_teachers_three_students_arrangements_l1253_125302


namespace NUMINAMATH_CALUDE_quadratic_with_property_has_negative_root_l1253_125339

/-- A quadratic polynomial with the given property has at least one negative root -/
theorem quadratic_with_property_has_negative_root (f : ℝ → ℝ) 
  (h1 : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0) 
  (h2 : ∀ (a b : ℝ), f (a^2 + b^2) ≥ f (2*a*b)) :
  ∃ (x : ℝ), x < 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_with_property_has_negative_root_l1253_125339


namespace NUMINAMATH_CALUDE_circle_coloring_theorem_l1253_125357

/-- Represents a circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Represents a coloring of the plane -/
def Coloring := ℝ × ℝ → Bool

/-- Checks if two points are on opposite sides of a circle -/
def oppositeSides (c : Circle) (p1 p2 : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((p1.1 - c.center.1)^2 + (p1.2 - c.center.2)^2)
  let d2 := Real.sqrt ((p2.1 - c.center.1)^2 + (p2.2 - c.center.2)^2)
  (d1 < c.radius ∧ d2 > c.radius) ∨ (d1 > c.radius ∧ d2 < c.radius)

/-- Checks if a coloring is valid for a given set of circles -/
def validColoring (circles : List Circle) (coloring : Coloring) : Prop :=
  ∀ c ∈ circles, ∀ p1 p2 : ℝ × ℝ, oppositeSides c p1 p2 → coloring p1 ≠ coloring p2

theorem circle_coloring_theorem (n : ℕ) (hn : n > 0) (circles : List Circle) 
    (hc : circles.length = n) : 
    ∃ coloring : Coloring, validColoring circles coloring := by
  sorry

end NUMINAMATH_CALUDE_circle_coloring_theorem_l1253_125357


namespace NUMINAMATH_CALUDE_pseudo_periodic_minus_one_is_periodic_two_cos_pseudo_periodic_iff_omega_multiple_of_pi_l1253_125314

-- Define a pseudo-periodic function
def IsPseudoPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T ≠ 0 ∧ ∀ x, f (x + T) = T * f x

-- Theorem 1
theorem pseudo_periodic_minus_one_is_periodic_two (f : ℝ → ℝ) 
  (h : IsPseudoPeriodic f (-1)) : 
  ∀ x, f (x + 2) = f x := by sorry

-- Theorem 2
theorem cos_pseudo_periodic_iff_omega_multiple_of_pi (ω : ℝ) :
  IsPseudoPeriodic (λ x => Real.cos (ω * x)) T ↔ ∃ k : ℤ, ω = k * Real.pi := by sorry

end NUMINAMATH_CALUDE_pseudo_periodic_minus_one_is_periodic_two_cos_pseudo_periodic_iff_omega_multiple_of_pi_l1253_125314


namespace NUMINAMATH_CALUDE_total_gulbis_count_l1253_125316

/-- The number of dureums of gulbis -/
def num_dureums : ℕ := 156

/-- The number of gulbis in one dureum -/
def gulbis_per_dureum : ℕ := 20

/-- The total number of gulbis -/
def total_gulbis : ℕ := num_dureums * gulbis_per_dureum

theorem total_gulbis_count : total_gulbis = 3120 := by
  sorry

end NUMINAMATH_CALUDE_total_gulbis_count_l1253_125316


namespace NUMINAMATH_CALUDE_employed_males_percentage_l1253_125340

theorem employed_males_percentage (population : ℝ) 
  (h1 : population > 0) 
  (employed_percentage : ℝ) 
  (h2 : employed_percentage = 0.64) 
  (employed_females_percentage : ℝ) 
  (h3 : employed_females_percentage = 0.140625) : 
  (employed_percentage * (1 - employed_females_percentage)) * population / population = 0.5496 := by
sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l1253_125340


namespace NUMINAMATH_CALUDE_oldest_child_age_l1253_125368

def children_ages (ages : Fin 5 → ℕ) : Prop :=
  -- The average age is 6
  (ages 0 + ages 1 + ages 2 + ages 3 + ages 4) / 5 = 6 ∧
  -- Ages are different
  ∀ i j, i ≠ j → ages i ≠ ages j ∧
  -- Difference between consecutive ages is 2
  ∀ i : Fin 4, ages i.succ = ages i + 2

theorem oldest_child_age (ages : Fin 5 → ℕ) (h : children_ages ages) :
  ages 0 = 10 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_l1253_125368


namespace NUMINAMATH_CALUDE_max_M_value_inequality_proof_l1253_125381

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| + |x - 3|

-- Theorem for part 1
theorem max_M_value : 
  (∃ M : ℝ, (∀ x m : ℝ, f x ≥ |m + 1| → m ≤ M) ∧ 
   (∀ ε > 0, ∃ x m : ℝ, f x < |m + 1| ∧ m > M - ε)) → 
  (∃ M : ℝ, M = 3/2 ∧ 
   (∀ x m : ℝ, f x ≥ |m + 1| → m ≤ M) ∧
   (∀ ε > 0, ∃ x m : ℝ, f x < |m + 1| ∧ m > M - ε)) :=
sorry

-- Theorem for part 2
theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 3/2) : 
  b^2/a + c^2/b + a^2/c ≥ 3/2 :=
sorry

end NUMINAMATH_CALUDE_max_M_value_inequality_proof_l1253_125381


namespace NUMINAMATH_CALUDE_at_least_two_equal_sums_l1253_125380

-- Define the type for cell values
inductive CellValue
  | one
  | three
  | five
  | seven

-- Define the type for the 5x5 square
def Square := Matrix (Fin 5) (Fin 5) CellValue

-- Define a function to represent a valid sum (odd number between 5 and 35)
def ValidSum := {n : ℕ // n % 2 = 1 ∧ 5 ≤ n ∧ n ≤ 35}

-- Define a function to calculate the sum of a line (row, column, or diagonal)
def lineSum (s : Square) (line : List (Fin 5 × Fin 5)) : ValidSum :=
  sorry

-- Define the set of all lines (rows, columns, and diagonals)
def allLines : Set (List (Fin 5 × Fin 5)) :=
  sorry

-- Theorem statement
theorem at_least_two_equal_sums (s : Square) :
  ∃ (l1 l2 : List (Fin 5 × Fin 5)), l1 ∈ allLines ∧ l2 ∈ allLines ∧ l1 ≠ l2 ∧ lineSum s l1 = lineSum s l2 :=
  sorry

end NUMINAMATH_CALUDE_at_least_two_equal_sums_l1253_125380


namespace NUMINAMATH_CALUDE_ultramindmaster_codes_l1253_125358

/-- The number of available colors in UltraMindmaster -/
def num_colors : ℕ := 8

/-- The number of slots in each secret code -/
def num_slots : ℕ := 5

/-- The number of possible secret codes in UltraMindmaster -/
def num_codes : ℕ := num_colors ^ num_slots

theorem ultramindmaster_codes :
  num_codes = 32768 := by
  sorry

end NUMINAMATH_CALUDE_ultramindmaster_codes_l1253_125358


namespace NUMINAMATH_CALUDE_students_meeting_time_l1253_125323

/-- Two students walking towards each other -/
theorem students_meeting_time 
  (distance : ℝ) 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (h1 : distance = 350) 
  (h2 : speed1 = 1.6) 
  (h3 : speed2 = 1.9) : 
  distance / (speed1 + speed2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_students_meeting_time_l1253_125323


namespace NUMINAMATH_CALUDE_total_family_members_eq_243_l1253_125377

/-- The total number of grandchildren and extended family members for Grandma Olga -/
def total_family_members : ℕ :=
  let daughters := 6
  let sons := 5
  let children_per_daughter := 10 + 9  -- 10 sons + 9 daughters
  let stepchildren_per_daughter := 4
  let children_per_son := 8 + 7  -- 8 daughters + 7 sons
  let inlaws_per_son := 3
  let children_per_inlaw := 2

  daughters * children_per_daughter +
  daughters * stepchildren_per_daughter +
  sons * children_per_son +
  sons * inlaws_per_son * children_per_inlaw

theorem total_family_members_eq_243 :
  total_family_members = 243 := by
  sorry

end NUMINAMATH_CALUDE_total_family_members_eq_243_l1253_125377


namespace NUMINAMATH_CALUDE_problem_statement_l1253_125352

theorem problem_statement (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 5*t + 3) 
  (h3 : x = 1) : 
  y = 8 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1253_125352


namespace NUMINAMATH_CALUDE_binomial_10_3_l1253_125393

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l1253_125393


namespace NUMINAMATH_CALUDE_sum_of_three_integers_2015_l1253_125317

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem sum_of_three_integers_2015 :
  ∃ (a b c : ℕ),
    a + b + c = 2015 ∧
    is_prime a ∧
    b % 3 = 0 ∧
    400 < c ∧ c < 500 ∧
    ¬(c % 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_integers_2015_l1253_125317


namespace NUMINAMATH_CALUDE_max_bottles_from_C_and_D_l1253_125364

/-- Represents the shops selling recyclable bottles -/
inductive Shop
| A
| B
| C
| D

/-- The price of a bottle at each shop -/
def price (s : Shop) : ℕ :=
  match s with
  | Shop.A => 1
  | Shop.B => 2
  | Shop.C => 3
  | Shop.D => 5

/-- Don's initial budget -/
def initial_budget : ℕ := 600

/-- Number of bottles Don buys from Shop A -/
def bottles_from_A : ℕ := 150

/-- Number of bottles Don buys from Shop B -/
def bottles_from_B : ℕ := 180

/-- The remaining budget after buying from shops A and B -/
def remaining_budget : ℕ := 
  initial_budget - (bottles_from_A * price Shop.A + bottles_from_B * price Shop.B)

/-- The theorem stating the maximum number of bottles Don can buy from shops C and D combined -/
theorem max_bottles_from_C_and_D : 
  (remaining_budget / price Shop.C) = 30 := by sorry

end NUMINAMATH_CALUDE_max_bottles_from_C_and_D_l1253_125364


namespace NUMINAMATH_CALUDE_joan_total_seashells_l1253_125362

/-- Given that Joan found 79 seashells, received 63 from Mike, and 97 from Alicia,
    prove that the total number of seashells Joan has is 239. -/
theorem joan_total_seashells 
  (joan_found : ℕ) 
  (mike_gave : ℕ) 
  (alicia_gave : ℕ) 
  (h1 : joan_found = 79) 
  (h2 : mike_gave = 63) 
  (h3 : alicia_gave = 97) : 
  joan_found + mike_gave + alicia_gave = 239 := by
  sorry

end NUMINAMATH_CALUDE_joan_total_seashells_l1253_125362


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_l1253_125379

theorem arithmetic_sequence_squares (k : ℤ) : ∃ (a : ℕ → ℤ), 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) ∧ 
  (a 1)^2 = 36 + k ∧
  (a 2)^2 = 300 + k ∧
  (a 3)^2 = 596 + k ∧
  k = 925 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_l1253_125379
