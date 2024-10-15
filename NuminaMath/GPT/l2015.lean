import Mathlib

namespace NUMINAMATH_GPT_avg_reading_time_l2015_201557

theorem avg_reading_time (emery_book_time serena_book_time emery_article_time serena_article_time : ℕ)
    (h1 : emery_book_time = 20)
    (h2 : emery_article_time = 2)
    (h3 : emery_book_time * 5 = serena_book_time)
    (h4 : emery_article_time * 3 = serena_article_time) :
    (emery_book_time + emery_article_time + serena_book_time + serena_article_time) / 2 = 64 := by
  sorry

end NUMINAMATH_GPT_avg_reading_time_l2015_201557


namespace NUMINAMATH_GPT_one_person_remains_dry_l2015_201563

theorem one_person_remains_dry (n : ℕ) :
  ∃ (person_dry : ℕ -> Bool), (∀ i : ℕ, i < 2 * n + 1 -> person_dry i = tt) := 
sorry

end NUMINAMATH_GPT_one_person_remains_dry_l2015_201563


namespace NUMINAMATH_GPT_find_x_find_a_l2015_201571

-- Definitions based on conditions
def inversely_proportional (p q : ℕ) (k : ℕ) := p * q = k

-- Given conditions for (x, y)
def x1 : ℕ := 36
def y1 : ℕ := 4
def k1 : ℕ := x1 * y1 -- or 144
def y2 : ℕ := 9

-- Given conditions for (a, b)
def a1 : ℕ := 50
def b1 : ℕ := 5
def k2 : ℕ := a1 * b1 -- or 250
def b2 : ℕ := 10

-- Proof statements
theorem find_x (x : ℕ) : inversely_proportional x y2 k1 → x = 16 := by
  sorry

theorem find_a (a : ℕ) : inversely_proportional a b2 k2 → a = 25 := by
  sorry

end NUMINAMATH_GPT_find_x_find_a_l2015_201571


namespace NUMINAMATH_GPT_expression_evaluation_l2015_201578

def a : ℚ := 8 / 9
def b : ℚ := 5 / 6
def c : ℚ := 2 / 3
def d : ℚ := -5 / 18
def lhs : ℚ := (a - b + c) / d
def rhs : ℚ := -13 / 5

theorem expression_evaluation : lhs = rhs := by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l2015_201578


namespace NUMINAMATH_GPT_a2_value_is_42_l2015_201532

noncomputable def a₂_value (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :=
  a_2

theorem a2_value_is_42 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :
  (x^3 + x^10 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + a_4 * (x + 1)^4 +
                a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + a_7 * (x + 1)^7 + a_8 * (x + 1)^8 + 
                a_9 * (x + 1)^9 + a_10 * (x + 1)^10) →
  a₂_value a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 = 42 :=
by
  sorry

end NUMINAMATH_GPT_a2_value_is_42_l2015_201532


namespace NUMINAMATH_GPT_seventh_observation_is_4_l2015_201536

def avg_six := 11 -- Average of the first six observations
def sum_six := 6 * avg_six -- Total sum of the first six observations
def new_avg := avg_six - 1 -- New average after including the new observation
def new_sum := 7 * new_avg -- Total sum after including the new observation

theorem seventh_observation_is_4 : 
  (new_sum - sum_six) = 4 :=
by
  sorry

end NUMINAMATH_GPT_seventh_observation_is_4_l2015_201536


namespace NUMINAMATH_GPT_cycling_journey_l2015_201512

theorem cycling_journey :
  ∃ y : ℚ, 0 < y ∧ y <= 12 ∧ (15 * y + 10 * (12 - y) = 150) ∧ y = 6 :=
by
  sorry

end NUMINAMATH_GPT_cycling_journey_l2015_201512


namespace NUMINAMATH_GPT_fraction_meaningful_if_not_neg_two_l2015_201505

theorem fraction_meaningful_if_not_neg_two {a : ℝ} : (a + 2 ≠ 0) ↔ (a ≠ -2) :=
by sorry

end NUMINAMATH_GPT_fraction_meaningful_if_not_neg_two_l2015_201505


namespace NUMINAMATH_GPT_plants_given_away_l2015_201564

-- Define the conditions as constants
def initial_plants : ℕ := 3
def final_plants : ℕ := 20
def months : ℕ := 3

-- Function to calculate the number of plants after n months
def plants_after_months (initial: ℕ) (months: ℕ) : ℕ := initial * (2 ^ months)

-- The proof problem statement
theorem plants_given_away : (plants_after_months initial_plants months - final_plants) = 4 :=
by
  sorry

end NUMINAMATH_GPT_plants_given_away_l2015_201564


namespace NUMINAMATH_GPT_clothes_in_total_l2015_201592

-- Define the conditions as constants since they are fixed values
def piecesInOneLoad : Nat := 17
def numberOfSmallLoads : Nat := 5
def piecesPerSmallLoad : Nat := 6

-- Noncomputable for definition involving calculation
noncomputable def totalClothes : Nat :=
  piecesInOneLoad + (numberOfSmallLoads * piecesPerSmallLoad)

-- The theorem to prove Luke had 47 pieces of clothing in total
theorem clothes_in_total : totalClothes = 47 := by
  sorry

end NUMINAMATH_GPT_clothes_in_total_l2015_201592


namespace NUMINAMATH_GPT_probability_not_cash_l2015_201518

theorem probability_not_cash (h₁ : 0.45 + 0.15 + pnc = 1) : pnc = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_cash_l2015_201518


namespace NUMINAMATH_GPT_complex_expression_evaluation_l2015_201504

-- Conditions
def i : ℂ := Complex.I -- Representing the imaginary unit i

-- Defining the inverse of a complex number
noncomputable def complex_inv (z : ℂ) := 1 / z

-- Proof statement
theorem complex_expression_evaluation :
  (i - complex_inv i + 3)⁻¹ = (3 - 2 * i) / 13 := by
sorry

end NUMINAMATH_GPT_complex_expression_evaluation_l2015_201504


namespace NUMINAMATH_GPT_harry_geckos_count_l2015_201581

theorem harry_geckos_count 
  (G : ℕ)
  (iguanas : ℕ := 2)
  (snakes : ℕ := 4)
  (cost_snake : ℕ := 10)
  (cost_iguana : ℕ := 5)
  (cost_gecko : ℕ := 15)
  (annual_cost : ℕ := 1140) :
  12 * (snakes * cost_snake + iguanas * cost_iguana + G * cost_gecko) = annual_cost → 
  G = 3 := 
by 
  intros h
  sorry

end NUMINAMATH_GPT_harry_geckos_count_l2015_201581


namespace NUMINAMATH_GPT_student_correct_answers_l2015_201577

theorem student_correct_answers 
(C W : ℕ) 
(h1 : C + W = 80) 
(h2 : 4 * C - W = 120) : 
C = 40 :=
by
  sorry 

end NUMINAMATH_GPT_student_correct_answers_l2015_201577


namespace NUMINAMATH_GPT_find_x_l2015_201526

-- Definitions based on provided conditions

def rectangle_length (x : ℝ) : ℝ := 4 * x
def rectangle_width (x : ℝ) : ℝ := x + 7
def rectangle_area (x : ℝ) : ℝ := rectangle_length x * rectangle_width x
def rectangle_perimeter (x : ℝ) : ℝ := 2 * rectangle_length x + 2 * rectangle_width x

-- Theorem statement
theorem find_x (x : ℝ) (h : rectangle_area x = 2 * rectangle_perimeter x) : x = 1 := 
sorry

end NUMINAMATH_GPT_find_x_l2015_201526


namespace NUMINAMATH_GPT_aubree_animals_total_l2015_201544

theorem aubree_animals_total (b_go c_go b_return c_return : ℕ) 
    (h1 : b_go = 20) (h2 : c_go = 40) 
    (h3 : b_return = b_go * 2) 
    (h4 : c_return = c_go - 10) : 
    b_go + c_go + b_return + c_return = 130 := by 
  sorry

end NUMINAMATH_GPT_aubree_animals_total_l2015_201544


namespace NUMINAMATH_GPT_total_marbles_l2015_201519

variable (r b g : ℝ)
variable (h1 : r = 1.3 * b)
variable (h2 : g = 1.7 * r)

theorem total_marbles (r b g : ℝ) (h1 : r = 1.3 * b) (h2 : g = 1.7 * r) :
  r + b + g = 3.469 * r :=
by
  sorry

end NUMINAMATH_GPT_total_marbles_l2015_201519


namespace NUMINAMATH_GPT_find_triples_l2015_201555

theorem find_triples (a m n : ℕ) (k : ℕ):
  a ≥ 2 ∧ m ≥ 2 ∧ a^n + 203 ≡ 0 [MOD a^m + 1] ↔ 
  (a = 2 ∧ ((n = 4 * k + 1 ∧ m = 2) ∨ (n = 6 * k + 2 ∧ m = 3) ∨ (n = 8 * k + 8 ∧ m = 4) ∨ (n = 12 * k + 9 ∧ m = 6))) ∨
  (a = 3 ∧ n = 4 * k + 3 ∧ m = 2) ∨
  (a = 4 ∧ n = 4 * k + 4 ∧ m = 2) ∨
  (a = 5 ∧ n = 4 * k + 1 ∧ m = 2) ∨
  (a = 8 ∧ n = 4 * k + 3 ∧ m = 2) ∨
  (a = 10 ∧ n = 4 * k + 2 ∧ m = 2) ∨
  (a = 203 ∧ n = (2 * k + 1) * m + 1 ∧ m ≥ 2) := by sorry

end NUMINAMATH_GPT_find_triples_l2015_201555


namespace NUMINAMATH_GPT_mean_study_hours_l2015_201558

theorem mean_study_hours :
  let students := [3, 6, 8, 5, 4, 2, 2]
  let hours := [0, 2, 4, 6, 8, 10, 12]
  (0 * 3 + 2 * 6 + 4 * 8 + 6 * 5 + 8 * 4 + 10 * 2 + 12 * 2) / (3 + 6 + 8 + 5 + 4 + 2 + 2) = 5 :=
by
  sorry

end NUMINAMATH_GPT_mean_study_hours_l2015_201558


namespace NUMINAMATH_GPT_fruit_count_l2015_201528

theorem fruit_count :
  let limes_mike : ℝ := 32.5
  let limes_alyssa : ℝ := 8.25
  let limes_jenny_picked : ℝ := 10.8
  let limes_jenny_ate := limes_jenny_picked / 2
  let limes_jenny := limes_jenny_picked - limes_jenny_ate
  let plums_tom : ℝ := 14.5
  let plums_tom_ate : ℝ := 2.5
  let X := (limes_mike - limes_alyssa) + limes_jenny
  let Y := plums_tom - plums_tom_ate
  X = 29.65 ∧ Y = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_fruit_count_l2015_201528


namespace NUMINAMATH_GPT_find_counterfeit_10_l2015_201543

theorem find_counterfeit_10 (coins : Fin 10 → ℕ) (h_counterfeit : ∃ k, ∀ i, i ≠ k → coins i < coins k) : 
  ∃ w : ℕ → ℕ → Prop, (∀ g1 g2, g1 ≠ g2 → w g1 g2 ∨ w g2 g1) → 
  ∃ k, ∀ i, i ≠ k → coins i < coins k :=
sorry

end NUMINAMATH_GPT_find_counterfeit_10_l2015_201543


namespace NUMINAMATH_GPT_sequence_sum_l2015_201530

theorem sequence_sum (r z w : ℝ) (h1 : 4 * r = 1) (h2 : 256 * r = z) (h3 : z * r = w) : z + w = 80 :=
by
  -- Proceed with your proof here.
  -- sorry for skipping the proof part.
  sorry

end NUMINAMATH_GPT_sequence_sum_l2015_201530


namespace NUMINAMATH_GPT_derivative_at_0_eq_6_l2015_201515

-- Definition of the function
def f (x : ℝ) : ℝ := (2 * x + 1)^3

-- Theorem statement indicating the derivative at x = 0 is 6
theorem derivative_at_0_eq_6 : (deriv f 0) = 6 := 
by 
  sorry -- The proof is omitted as per the instructions

end NUMINAMATH_GPT_derivative_at_0_eq_6_l2015_201515


namespace NUMINAMATH_GPT_scientists_speculation_reasonable_uranus_will_not_affect_earth_next_observation_l2015_201520

-- Define the given conditions as constants and theorems in Lean
theorem scientists_speculation_reasonable : 
  ∃ (a b c : ℝ), 
  (64*a - 8*b + c = 32) ∧ 
  (36*a - 6*b + c = 28.5) ∧ 
  (16*a - 4*b + c = 26) ∧ 
  (∀ (x y : ℝ), (y = a*x^2 + b*x + c) → (x = 0) → y < 24.5) :=
by -- sorry is a placeholder for the proof
sorry

theorem uranus_will_not_affect_earth_next_observation : 
  ∃ (a b c : ℝ), 
  (64*a - 8*b + c = 32) ∧ 
  (36*a - 6*b + c = 28.5) ∧ 
  (16*a - 4*b + c = 26) ∧ 
  (∀ (x y : ℝ), (y = a*x^2 + b*x + c) → (x = 2) → y ≥ 24.5) :=
by -- sorry is a placeholder for the proof
sorry

end NUMINAMATH_GPT_scientists_speculation_reasonable_uranus_will_not_affect_earth_next_observation_l2015_201520


namespace NUMINAMATH_GPT_movie_theater_charge_l2015_201507

theorem movie_theater_charge 
    (charge_adult : ℝ) 
    (children : ℕ) 
    (adults : ℕ) 
    (total_receipts : ℝ) 
    (charge_child : ℝ) 
    (condition1 : charge_adult = 6.75) 
    (condition2 : children = adults + 20) 
    (condition3 : total_receipts = 405) 
    (condition4 : children = 48) 
    : charge_child = 4.5 :=
sorry

end NUMINAMATH_GPT_movie_theater_charge_l2015_201507


namespace NUMINAMATH_GPT_math_problem_l2015_201580

noncomputable def m : ℕ := 294
noncomputable def n : ℕ := 81
noncomputable def d : ℕ := 3

axiom circle_radius (r : ℝ) : r = 42
axiom chords_length (l : ℝ) : l = 78
axiom intersection_distance (d : ℝ) : d = 18

theorem math_problem :
  let m := 294
  let n := 81
  let d := 3
  m + n + d = 378 :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_math_problem_l2015_201580


namespace NUMINAMATH_GPT_largest_positive_integer_n_l2015_201574

 

theorem largest_positive_integer_n (n : ℕ) :
  (∀ p : ℕ, Nat.Prime p ∧ 2 < p ∧ p < n → Nat.Prime (n - p)) →
  ∀ m : ℕ, (∀ q : ℕ, Nat.Prime q ∧ 2 < q ∧ q < m → Nat.Prime (m - q)) → n ≥ m → n = 10 :=
by
  sorry

end NUMINAMATH_GPT_largest_positive_integer_n_l2015_201574


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2015_201550

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
  (a + b + b = 25) ∧ (a + a + b ≤ b → False) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2015_201550


namespace NUMINAMATH_GPT_compute_expression_l2015_201575

-- Define the conditions as specific values and operations within the theorem itself
theorem compute_expression : 5 + 7 * (2 - 9)^2 = 348 := 
  by
  sorry

end NUMINAMATH_GPT_compute_expression_l2015_201575


namespace NUMINAMATH_GPT_inverse_variation_with_constant_l2015_201562

theorem inverse_variation_with_constant
  (k : ℝ)
  (x y : ℝ)
  (h1 : y = (3 * k) / x)
  (h2 : x = 4)
  (h3 : y = 8) :
  (y = (3 * (32 / 3)) / -16) := by
sorry

end NUMINAMATH_GPT_inverse_variation_with_constant_l2015_201562


namespace NUMINAMATH_GPT_symmetrical_circle_equation_l2015_201513

theorem symmetrical_circle_equation :
  ∀ (x y : ℝ), (x^2 + y^2 - 2 * x - 1 = 0) ∧ (2 * x - y + 1 = 0) →
  ((x + 7/5)^2 + (y - 6/5)^2 = 2) :=
sorry

end NUMINAMATH_GPT_symmetrical_circle_equation_l2015_201513


namespace NUMINAMATH_GPT_sum_of_first_6n_integers_l2015_201598

theorem sum_of_first_6n_integers (n : ℕ) (h1 : (5 * n * (5 * n + 1)) / 2 = (n * (n + 1)) / 2 + 200) :
  (6 * n * (6 * n + 1)) / 2 = 300 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_6n_integers_l2015_201598


namespace NUMINAMATH_GPT_solution_x_y_l2015_201506

noncomputable def eq_values (x y : ℝ) := (
  x ≠ 0 ∧ x ≠ 1 ∧ y ≠ 0 ∧ y ≠ 3 ∧ (3/x + 2/y = 1/3)
)

theorem solution_x_y (x y : ℝ) (h : eq_values x y) : x = 9 * y / (y - 6) :=
sorry

end NUMINAMATH_GPT_solution_x_y_l2015_201506


namespace NUMINAMATH_GPT_supplement_of_complement_65_degrees_l2015_201531

def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_65_degrees : 
  supplement (complement 65) = 155 :=
by
  sorry

end NUMINAMATH_GPT_supplement_of_complement_65_degrees_l2015_201531


namespace NUMINAMATH_GPT_four_thirds_of_twelve_fifths_l2015_201551

theorem four_thirds_of_twelve_fifths : (4 / 3) * (12 / 5) = 16 / 5 := 
by sorry

end NUMINAMATH_GPT_four_thirds_of_twelve_fifths_l2015_201551


namespace NUMINAMATH_GPT_discrim_of_quad_l2015_201567

-- Definition of the quadratic equation coefficients
def a : ℤ := 5
def b : ℤ := -9
def c : ℤ := 4

-- Definition of the discriminant formula which needs to be proved as 1
def discriminant (a b c : ℤ) : ℤ :=
  b^2 - 4 * a * c

-- The proof problem statement
theorem discrim_of_quad : discriminant a b c = 1 := by
  sorry

end NUMINAMATH_GPT_discrim_of_quad_l2015_201567


namespace NUMINAMATH_GPT_inequality_solutions_l2015_201547

theorem inequality_solutions (y : ℝ) :
  (2 / (y + 2) + 4 / (y + 8) ≥ 1 ↔ (y > -8 ∧ y ≤ -4) ∨ (y ≥ -2 ∧ y ≤ 2)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solutions_l2015_201547


namespace NUMINAMATH_GPT_binomial_log_inequality_l2015_201573

theorem binomial_log_inequality (n : ℤ) :
  n * Real.log 2 ≤ Real.log (Nat.choose (2 * n.natAbs) n.natAbs) ∧ 
  Real.log (Nat.choose (2 * n.natAbs) n.natAbs) ≤ n * Real.log 4 :=
by sorry

end NUMINAMATH_GPT_binomial_log_inequality_l2015_201573


namespace NUMINAMATH_GPT_find_g_of_3_l2015_201545

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_3 (h : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = 3) : g 3 = 0 :=
by sorry

end NUMINAMATH_GPT_find_g_of_3_l2015_201545


namespace NUMINAMATH_GPT_sum_of_exponents_l2015_201525

theorem sum_of_exponents : 
  (-1)^(2010) + (-1)^(2013) + 1^(2014) + (-1)^(2016) = 0 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_exponents_l2015_201525


namespace NUMINAMATH_GPT_P_zero_value_l2015_201588

noncomputable def P (x b c : ℚ) : ℚ := x ^ 2 + b * x + c

theorem P_zero_value (b c : ℚ)
  (h1 : P (P 1 b c) b c = 0)
  (h2 : P (P (-2) b c) b c = 0)
  (h3 : P 1 b c ≠ P (-2) b c) :
  P 0 b c = -5 / 2 :=
sorry

end NUMINAMATH_GPT_P_zero_value_l2015_201588


namespace NUMINAMATH_GPT_part1_part2_l2015_201585

theorem part1 : (2 / 9 - 1 / 6 + 1 / 18) * (-18) = -2 := 
by
  sorry

theorem part2 : 54 * (3 / 4 + 1 / 2 - 1 / 4) = 54 := 
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2015_201585


namespace NUMINAMATH_GPT_solve_quadratic_equations_l2015_201597

noncomputable def E1 := ∀ x : ℝ, x^2 - 14 * x + 21 = 0 ↔ (x = 7 + 2 * Real.sqrt 7 ∨ x = 7 - 2 * Real.sqrt 7)

noncomputable def E2 := ∀ x : ℝ, x^2 - 3 * x + 2 = 0 ↔ (x = 1 ∨ x = 2)

theorem solve_quadratic_equations :
  (E1) ∧ (E2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_equations_l2015_201597


namespace NUMINAMATH_GPT_not_p_and_q_equiv_not_p_or_not_q_l2015_201501

variable (p q : Prop)

theorem not_p_and_q_equiv_not_p_or_not_q (h : ¬ (p ∧ q)) : ¬ p ∨ ¬ q :=
sorry

end NUMINAMATH_GPT_not_p_and_q_equiv_not_p_or_not_q_l2015_201501


namespace NUMINAMATH_GPT_sum_of_powers_seven_l2015_201508

theorem sum_of_powers_seven (α1 α2 α3 : ℂ)
  (h1 : α1 + α2 + α3 = 2)
  (h2 : α1^2 + α2^2 + α3^2 = 6)
  (h3 : α1^3 + α2^3 + α3^3 = 14) :
  α1^7 + α2^7 + α3^7 = 478 := by
  sorry

end NUMINAMATH_GPT_sum_of_powers_seven_l2015_201508


namespace NUMINAMATH_GPT_alcohol_percentage_first_solution_l2015_201565

theorem alcohol_percentage_first_solution
  (x : ℝ)
  (h1 : 0 ≤ x ∧ x ≤ 1) -- since percentage in decimal form is between 0 and 1
  (h2 : 75 * x + 0.12 * 125 = 0.15 * 200) :
  x = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_alcohol_percentage_first_solution_l2015_201565


namespace NUMINAMATH_GPT_total_number_of_workers_l2015_201556

theorem total_number_of_workers 
    (W N : ℕ) 
    (h1 : 8000 * W = 12000 * 8 + 6000 * N) 
    (h2 : W = 8 + N) : 
    W = 24 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_workers_l2015_201556


namespace NUMINAMATH_GPT_esther_commute_distance_l2015_201566

theorem esther_commute_distance (D : ℕ) :
  (D / 45 + D / 30 = 1) → D = 18 :=
by
  sorry

end NUMINAMATH_GPT_esther_commute_distance_l2015_201566


namespace NUMINAMATH_GPT_bridge_length_calculation_l2015_201522

def length_of_bridge (train_length : ℕ) (train_speed_kmph : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_mps := (train_speed_kmph * 1000) / 3600
  let distance_covered := speed_mps * time_seconds
  distance_covered - train_length

theorem bridge_length_calculation :
  length_of_bridge 140 45 30 = 235 :=
by
  unfold length_of_bridge
  norm_num
  sorry

end NUMINAMATH_GPT_bridge_length_calculation_l2015_201522


namespace NUMINAMATH_GPT_cakes_served_today_l2015_201584

def lunch_cakes := 6
def dinner_cakes := 9
def total_cakes := lunch_cakes + dinner_cakes

theorem cakes_served_today : total_cakes = 15 := by
  sorry

end NUMINAMATH_GPT_cakes_served_today_l2015_201584


namespace NUMINAMATH_GPT_transformation_l2015_201541

noncomputable def Q (a b c x y : ℝ) : ℝ := a * x^2 + 2 * b * x * y + c * y^2

theorem transformation 
  (a b c d e f x y x₀ y₀ x' y' : ℝ)
  (h : a * c - b^2 ≠ 0)
  (hQ : Q a b c x y + 2 * d * x + 2 * e * y = f)
  (hx : x' = x + x₀)
  (hy : y' = y + y₀) :
  ∃ f' : ℝ, (a * x'^2 + 2 * b * x' * y' + c * y'^2 = f' ∧ 
             f' = f - Q a b c x₀ y₀ + 2 * (d * x₀ + e * y₀)) :=
sorry

end NUMINAMATH_GPT_transformation_l2015_201541


namespace NUMINAMATH_GPT_problem_solution_l2015_201538

variables (x y : ℝ)

def cond1 : Prop := 4 * x + y = 12
def cond2 : Prop := x + 4 * y = 18

theorem problem_solution (h1 : cond1 x y) (h2 : cond2 x y) : 20 * x^2 + 24 * x * y + 20 * y^2 = 468 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_problem_solution_l2015_201538


namespace NUMINAMATH_GPT_meaningful_fraction_l2015_201569

theorem meaningful_fraction (x : ℝ) : (x ≠ -2) ↔ (∃ y : ℝ, y = 1 / (x + 2)) :=
by sorry

end NUMINAMATH_GPT_meaningful_fraction_l2015_201569


namespace NUMINAMATH_GPT_inequality_solution_l2015_201540

theorem inequality_solution (x : ℝ) :
  (x - 3) / (x^2 + 4 * x + 13) ≥ 0 ↔ x ∈ Set.Ici 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2015_201540


namespace NUMINAMATH_GPT_bill_has_6_less_pieces_than_mary_l2015_201529

-- Definitions based on the conditions
def total_candy : ℕ := 20
def candy_kate : ℕ := 4
def candy_robert : ℕ := candy_kate + 2
def candy_mary : ℕ := candy_robert + 2
def candy_bill : ℕ := candy_kate - 2

-- Statement of the theorem
theorem bill_has_6_less_pieces_than_mary :
  candy_mary - candy_bill = 6 :=
sorry

end NUMINAMATH_GPT_bill_has_6_less_pieces_than_mary_l2015_201529


namespace NUMINAMATH_GPT_focus_of_parabola_l2015_201535

theorem focus_of_parabola (x y : ℝ) : x^2 = 4 * y → (0, 1) = (0, (4 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l2015_201535


namespace NUMINAMATH_GPT_inequalities_sufficient_but_not_necessary_l2015_201559

theorem inequalities_sufficient_but_not_necessary (a b c d : ℝ) :
  (a > b ∧ c > d) → (a + c > b + d) ∧ ¬((a + c > b + d) → (a > b ∧ c > d)) :=
by
  sorry

end NUMINAMATH_GPT_inequalities_sufficient_but_not_necessary_l2015_201559


namespace NUMINAMATH_GPT_blocks_left_l2015_201524

theorem blocks_left (initial_blocks used_blocks : ℕ) (h_initial : initial_blocks = 59) (h_used : used_blocks = 36) : initial_blocks - used_blocks = 23 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_blocks_left_l2015_201524


namespace NUMINAMATH_GPT_city_population_l2015_201591

theorem city_population (p : ℝ) (hp : 0.85 * (p + 2000) = p + 2050) : p = 2333 :=
by
  sorry

end NUMINAMATH_GPT_city_population_l2015_201591


namespace NUMINAMATH_GPT_least_positive_integer_l2015_201542

theorem least_positive_integer (n : ℕ) : 
  (n % 4 = 1) ∧ (n % 5 = 2) ∧ (n % 6 = 3) → n = 57 := by
sorry

end NUMINAMATH_GPT_least_positive_integer_l2015_201542


namespace NUMINAMATH_GPT_vasya_fraction_l2015_201582

-- Define the variables for distances and total distance
variables {a b c d s : ℝ}

-- Define conditions
def anton_distance (a b : ℝ) : Prop := a = b / 2
def sasha_distance (c a d : ℝ) : Prop := c = a + d
def dima_distance (d s : ℝ) : Prop := d = s / 10
def total_distance (a b c d s : ℝ) : Prop := a + b + c + d = s

-- The main theorem 
theorem vasya_fraction (a b c d s : ℝ) (h1 : anton_distance a b) 
  (h2 : sasha_distance c a d) (h3 : dima_distance d s)
  (h4 : total_distance a b c d s) : b / s = 0.4 :=
sorry

end NUMINAMATH_GPT_vasya_fraction_l2015_201582


namespace NUMINAMATH_GPT_james_faster_than_john_l2015_201516

theorem james_faster_than_john :
  let john_time := 13
  let john_distance := 100
  let john_first_second := 4
  let john_remaining_seconds := john_time - 1
  let john_remaining_distance := john_distance - john_first_second
  let john_top_speed := john_remaining_distance / john_remaining_seconds

  let james_time := 11
  let james_first_two_seconds := 10
  let james_remaining_seconds := james_time - 2
  let james_remaining_distance := john_distance - james_first_two_seconds
  let james_top_speed := james_remaining_distance / james_remaining_seconds
  
  james_top_speed - john_top_speed = 2 :=
by
  let john_time := 13
  let john_distance := 100
  let john_first_second := 4
  let john_remaining_seconds := john_time - 1
  let john_remaining_distance := john_distance - john_first_second
  let john_top_speed := john_remaining_distance / john_remaining_seconds

  let james_time := 11
  let james_first_two_seconds := 10
  let james_remaining_seconds := james_time - 2
  let james_remaining_distance := john_distance - james_first_two_seconds
  let james_top_speed := james_remaining_distance / james_remaining_seconds

  sorry

end NUMINAMATH_GPT_james_faster_than_john_l2015_201516


namespace NUMINAMATH_GPT_find_k_value_l2015_201593

-- Define the condition that point A(3, -5) lies on the graph of the function y = k / x
def point_on_inverse_proportion (k : ℝ) : Prop :=
  (3 : ℝ) ≠ 0 ∧ (-5) = k / (3 : ℝ)

-- The theorem to prove that k = -15 given the point on the graph
theorem find_k_value (k : ℝ) (h : point_on_inverse_proportion k) : k = -15 :=
by
  sorry

end NUMINAMATH_GPT_find_k_value_l2015_201593


namespace NUMINAMATH_GPT_problem_I_problem_II_l2015_201514

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := |x - 2| + |2 * x + a|

-- Problem (I): Inequality solution when a = 1
theorem problem_I (x : ℝ) : f x 1 ≥ 5 ↔ x ∈ (Set.Iic (-4 / 3) ∪ Set.Ici 2) :=
sorry

-- Problem (II): Range of a given the conditions
theorem problem_II (x₀ : ℝ) (a : ℝ) (h : f x₀ a + |x₀ - 2| < 3) : -7 < a ∧ a < -1 :=
sorry

end NUMINAMATH_GPT_problem_I_problem_II_l2015_201514


namespace NUMINAMATH_GPT_find_a_plus_b_l2015_201583

def satisfies_conditions (a b : ℝ) :=
  ∀ x : ℝ, 3 * (a * x + b) - 8 = 4 * x + 7

theorem find_a_plus_b (a b : ℝ) (h : satisfies_conditions a b) : a + b = 19 / 3 :=
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l2015_201583


namespace NUMINAMATH_GPT_smallest_n_condition_l2015_201586

noncomputable def distance_origin_to_point (n : ℕ) : ℝ := Real.sqrt (n)

noncomputable def radius_Bn (n : ℕ) : ℝ := distance_origin_to_point n - 1

def condition_Bn_contains_point_with_coordinate_greater_than_2 (n : ℕ) : Prop :=
  radius_Bn n > 2

theorem smallest_n_condition : ∃ n : ℕ, n ≥ 10 ∧ condition_Bn_contains_point_with_coordinate_greater_than_2 n :=
  sorry

end NUMINAMATH_GPT_smallest_n_condition_l2015_201586


namespace NUMINAMATH_GPT_rectangle_area_in_ellipse_l2015_201534

theorem rectangle_area_in_ellipse :
  ∃ a b : ℝ, 2 * a = b ∧ (a^2 / 4 + b^2 / 8 = 1) ∧ 2 * a * b = 16 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_in_ellipse_l2015_201534


namespace NUMINAMATH_GPT_sprinkler_system_days_l2015_201517

theorem sprinkler_system_days 
  (morning_water : ℕ) (evening_water : ℕ) (total_water : ℕ) 
  (h_morning : morning_water = 4) 
  (h_evening : evening_water = 6) 
  (h_total : total_water = 50) :
  total_water / (morning_water + evening_water) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_sprinkler_system_days_l2015_201517


namespace NUMINAMATH_GPT_prove_f_3_equals_11_l2015_201599

-- Assuming the given function definition as condition
def f (y : ℝ) : ℝ := sorry

-- The condition provided: f(x - 1/x) = x^2 + 1/x^2.
axiom function_definition (x : ℝ) (h : x ≠ 0): f (x - 1 / x) = x^2 + 1 / x^2

-- The goal is to prove that f(3) = 11
theorem prove_f_3_equals_11 : f 3 = 11 :=
by
  sorry

end NUMINAMATH_GPT_prove_f_3_equals_11_l2015_201599


namespace NUMINAMATH_GPT_frankie_pets_total_l2015_201537

theorem frankie_pets_total
  (C S P D : ℕ)
  (h_snakes : S = C + 6)
  (h_parrots : P = C - 1)
  (h_dogs : D = 2)
  (h_total : C + S + P + D = 19) :
  C + (C + 6) + (C - 1) + 2 = 19 := by
  sorry

end NUMINAMATH_GPT_frankie_pets_total_l2015_201537


namespace NUMINAMATH_GPT_finite_operations_invariant_final_set_l2015_201502

theorem finite_operations (n : ℕ) (a : Fin n → ℕ) :
  ∃ N : ℕ, ∀ k, k > N → ((∃ i j, i ≠ j ∧ ¬ (a i ∣ a j ∨ a j ∣ a i)) → False) :=
sorry

theorem invariant_final_set (n : ℕ) (a : Fin n → ℕ) :
  ∃ b : Fin n → ℕ, (∀ i, ∃ j, b i = a j) ∧ ∀ (c : Fin n → ℕ), (∀ i, ∃ j, c i = a j) → c = b :=
sorry

end NUMINAMATH_GPT_finite_operations_invariant_final_set_l2015_201502


namespace NUMINAMATH_GPT_number_of_solutions_l2015_201549

noncomputable def system_of_equations (a b c : ℕ) : Prop :=
  a * b + b * c = 44 ∧ a * c + b * c = 23

theorem number_of_solutions : ∃! (a b c : ℕ), system_of_equations a b c :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_l2015_201549


namespace NUMINAMATH_GPT_kostyas_table_prime_l2015_201589

theorem kostyas_table_prime (n : ℕ) (h₁ : n > 3) 
    (h₂ : ¬ ∃ r s : ℕ, r ≥ 3 ∧ s ≥ 3 ∧ n = r * s - (r + s)) : 
    Prime (n + 1) := 
sorry

end NUMINAMATH_GPT_kostyas_table_prime_l2015_201589


namespace NUMINAMATH_GPT_John_cycles_distance_l2015_201552

-- Define the rate and time as per the conditions in the problem
def rate : ℝ := 8 -- miles per hour
def time : ℝ := 2.25 -- hours

-- The mathematical statement to prove: distance = rate * time
theorem John_cycles_distance : rate * time = 18 := by
  sorry

end NUMINAMATH_GPT_John_cycles_distance_l2015_201552


namespace NUMINAMATH_GPT_problem1_problem2_l2015_201503

-- Define the total number of balls for clarity
def total_red_balls : ℕ := 4
def total_white_balls : ℕ := 6
def total_balls_drawn : ℕ := 4

-- Define binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ := n.choose k

-- Problem 1: Prove that the number of ways to draw 4 balls that include both colors is 194
theorem problem1 :
  (binom total_red_balls 3 * binom total_white_balls 1) +
  (binom total_red_balls 2 * binom total_white_balls 2) +
  (binom total_red_balls 1 * binom total_white_balls 3) = 194 :=
  sorry

-- Problem 2: Prove that the number of ways to draw 4 balls where the number of red balls is at least the number of white balls is 115
theorem problem2 :
  (binom total_red_balls 4 * binom total_white_balls 0) +
  (binom total_red_balls 3 * binom total_white_balls 1) +
  (binom total_red_balls 2 * binom total_white_balls 2) = 115 :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2015_201503


namespace NUMINAMATH_GPT_pieces_per_pizza_is_five_l2015_201546

-- Definitions based on the conditions
def cost_per_pizza (total_cost : ℕ) (number_of_pizzas : ℕ) : ℕ :=
  total_cost / number_of_pizzas

def number_of_pieces_per_pizza (cost_per_pizza : ℕ) (cost_per_piece : ℕ) : ℕ :=
  cost_per_pizza / cost_per_piece

-- Given conditions
def total_cost : ℕ := 80
def number_of_pizzas : ℕ := 4
def cost_per_piece : ℕ := 4

-- Prove
theorem pieces_per_pizza_is_five : number_of_pieces_per_pizza (cost_per_pizza total_cost number_of_pizzas) cost_per_piece = 5 :=
by sorry

end NUMINAMATH_GPT_pieces_per_pizza_is_five_l2015_201546


namespace NUMINAMATH_GPT_average_of_w_and_x_is_one_half_l2015_201523

noncomputable def average_of_w_and_x (w x y : ℝ) : ℝ :=
  (w + x) / 2

theorem average_of_w_and_x_is_one_half (w x y : ℝ)
  (h1 : 2 / w + 2 / x = 2 / y)
  (h2 : w * x = y) : average_of_w_and_x w x y = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_average_of_w_and_x_is_one_half_l2015_201523


namespace NUMINAMATH_GPT_prob_a_prob_b_l2015_201511

def A (a : ℝ) := {x : ℝ | 0 < x + a ∧ x + a ≤ 5}
def B := {x : ℝ | -1/2 ≤ x ∧ x < 6}

theorem prob_a (a : ℝ) : (A a ⊆ B) → (-1 < a ∧ a ≤ 1/2) :=
sorry

theorem prob_b (a : ℝ) : (∃ x, A a ∩ B = {x}) → a = 11/2 :=
sorry

end NUMINAMATH_GPT_prob_a_prob_b_l2015_201511


namespace NUMINAMATH_GPT_minimum_value_of_quadratic_l2015_201595

theorem minimum_value_of_quadratic (p q : ℝ) (hp : 0 < p) (hq : 0 < q) : 
  ∃ x : ℝ, x = -p / 2 ∧ (∀ y : ℝ, (y - x) ^ 2 + 2*q ≥ (x ^ 2 + p * x + 2*q)) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_quadratic_l2015_201595


namespace NUMINAMATH_GPT_eddy_time_to_B_l2015_201568

-- Definitions
def distance_A_to_B : ℝ := 570
def distance_A_to_C : ℝ := 300
def time_C : ℝ := 4
def speed_ratio : ℝ := 2.5333333333333333

-- Theorem Statement
theorem eddy_time_to_B : 
  (distance_A_to_B / (distance_A_to_C / time_C * speed_ratio)) = 3 := 
by
  sorry

end NUMINAMATH_GPT_eddy_time_to_B_l2015_201568


namespace NUMINAMATH_GPT_designer_suit_size_l2015_201594

theorem designer_suit_size : ∀ (waist_in_inches : ℕ) (comfort_in_inches : ℕ) 
  (inches_per_foot : ℕ) (cm_per_foot : ℝ), 
  waist_in_inches = 34 →
  comfort_in_inches = 2 →
  inches_per_foot = 12 →
  cm_per_foot = 30.48 →
  (((waist_in_inches + comfort_in_inches) / inches_per_foot : ℝ) * cm_per_foot) = 91.4 :=
by
  intros waist_in_inches comfort_in_inches inches_per_foot cm_per_foot
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_cast
  norm_num
  sorry

end NUMINAMATH_GPT_designer_suit_size_l2015_201594


namespace NUMINAMATH_GPT_area_enclosed_by_curves_l2015_201539

noncomputable def areaBetweenCurves : ℝ :=
  ∫ x in (0 : ℝ)..(4 : ℝ), (x - (x^2 - 3*x))

theorem area_enclosed_by_curves :
  (∫ x in (0 : ℝ)..(4 : ℝ), (x - (x^2 - 3*x))) = (32 / 3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_area_enclosed_by_curves_l2015_201539


namespace NUMINAMATH_GPT_abs_neg_2023_l2015_201554

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end NUMINAMATH_GPT_abs_neg_2023_l2015_201554


namespace NUMINAMATH_GPT_dot_product_result_l2015_201561

def a : ℝ × ℝ := (-1, 2)
def b (m : ℝ) : ℝ × ℝ := (2, m)
def c : ℝ × ℝ := (7, 1)

def are_parallel (a b : ℝ × ℝ) : Prop := 
  a.1 * b.2 = a.2 * b.1

def dot_product (u v : ℝ × ℝ) : ℝ := 
  u.1 * v.1 + u.2 * v.2

theorem dot_product_result : 
  ∀ m : ℝ, are_parallel a (b m) → dot_product (b m) c = 10 := 
by
  sorry

end NUMINAMATH_GPT_dot_product_result_l2015_201561


namespace NUMINAMATH_GPT_heal_time_l2015_201579

theorem heal_time (x : ℝ) (hx_pos : 0 < x) (h_total : 2.5 * x = 10) : x = 4 := 
by {
  -- Lean proof will be here
  sorry
}

end NUMINAMATH_GPT_heal_time_l2015_201579


namespace NUMINAMATH_GPT_max_value_2cosx_3sinx_l2015_201548

open Real 

theorem max_value_2cosx_3sinx : ∀ x : ℝ, 2 * cos x + 3 * sin x ≤ sqrt 13 :=
by sorry

end NUMINAMATH_GPT_max_value_2cosx_3sinx_l2015_201548


namespace NUMINAMATH_GPT_son_present_age_l2015_201510

variable (S F : ℕ)

-- Define the conditions
def fatherAgeCondition := F = S + 35
def twoYearsCondition := F + 2 = 2 * (S + 2)

-- The proof theorem
theorem son_present_age : 
  fatherAgeCondition S F → 
  twoYearsCondition S F → 
  S = 33 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_son_present_age_l2015_201510


namespace NUMINAMATH_GPT_total_money_from_selling_watermelons_l2015_201521

-- Given conditions
def weight_of_one_watermelon : ℝ := 23
def price_per_pound : ℝ := 2
def number_of_watermelons : ℝ := 18

-- Statement to be proved
theorem total_money_from_selling_watermelons : 
  (weight_of_one_watermelon * price_per_pound) * number_of_watermelons = 828 := 
by 
  sorry

end NUMINAMATH_GPT_total_money_from_selling_watermelons_l2015_201521


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2015_201576

-- Definitions of sets A and B based on the conditions
def A : Set ℝ := {x | 0 < x}
def B : Set ℝ := {0, 1, 2}

-- Theorem statement to prove A ∩ B = {1, 2}
theorem intersection_of_A_and_B : A ∩ B = {1, 2} := 
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2015_201576


namespace NUMINAMATH_GPT_circle_eq_of_hyperbola_focus_eccentricity_l2015_201500

theorem circle_eq_of_hyperbola_focus_eccentricity :
  ∀ (x y : ℝ), ((y^2 - (x^2 / 3) = 1) → (x^2 + (y-2)^2 = 4)) := by
  intro x y
  intro hyp_eq
  sorry

end NUMINAMATH_GPT_circle_eq_of_hyperbola_focus_eccentricity_l2015_201500


namespace NUMINAMATH_GPT_remainder_when_divided_by_11_l2015_201570

theorem remainder_when_divided_by_11 :
  (7 * 10^20 + 2^20) % 11 = 8 := by
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_11_l2015_201570


namespace NUMINAMATH_GPT_third_side_length_not_4_l2015_201572

theorem third_side_length_not_4 (x : ℕ) : 
  (5 < x + 9) ∧ (9 < x + 5) ∧ (x + 5 < 14) → ¬ (x = 4) := 
by
  intros h
  sorry

end NUMINAMATH_GPT_third_side_length_not_4_l2015_201572


namespace NUMINAMATH_GPT_new_person_weight_l2015_201553

theorem new_person_weight (avg_increase : Real) (n : Nat) (old_weight : Real) (W_new : Real) :
  avg_increase = 2.5 → n = 8 → old_weight = 67 → W_new = old_weight + n * avg_increase → W_new = 87 :=
by
  intros avg_increase_eq n_eq old_weight_eq calc_eq
  sorry

end NUMINAMATH_GPT_new_person_weight_l2015_201553


namespace NUMINAMATH_GPT_identify_solids_with_identical_views_l2015_201587

def has_identical_views (s : Type) : Prop := sorry

def sphere : Type := sorry
def triangular_pyramid : Type := sorry
def cube : Type := sorry
def cylinder : Type := sorry

theorem identify_solids_with_identical_views :
  (has_identical_views sphere) ∧
  (¬ has_identical_views triangular_pyramid) ∧
  (has_identical_views cube) ∧
  (¬ has_identical_views cylinder) :=
sorry

end NUMINAMATH_GPT_identify_solids_with_identical_views_l2015_201587


namespace NUMINAMATH_GPT_ink_percentage_left_l2015_201527

def area_of_square (side: ℕ) := side * side
def area_of_rectangle (length: ℕ) (width: ℕ) := length * width
def total_area_marker_can_paint (num_squares: ℕ) (square_side: ℕ) :=
  num_squares * area_of_square square_side
def total_area_colored (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ) :=
  num_rectangles * area_of_rectangle rect_length rect_width

def fraction_of_ink_used (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ)
  (num_squares: ℕ) (square_side: ℕ) :=
  (total_area_colored num_rectangles rect_length rect_width : ℚ)
    / (total_area_marker_can_paint num_squares square_side : ℚ)

def percentage_ink_left (num_rectangles: ℕ) (rect_length: ℕ) (rect_width: ℕ)
  (num_squares: ℕ) (square_side: ℕ) :=
  100 * (1 - fraction_of_ink_used num_rectangles rect_length rect_width num_squares square_side)

theorem ink_percentage_left :
  percentage_ink_left 2 6 2 3 4 = 50 := by
  sorry

end NUMINAMATH_GPT_ink_percentage_left_l2015_201527


namespace NUMINAMATH_GPT_rational_powers_imply_integers_l2015_201533

theorem rational_powers_imply_integers (a b : ℚ) (h_distinct : a ≠ b)
  (h_infinitely_many_n : ∃ᶠ (n : ℕ) in Filter.atTop, (n * (a^n - b^n) : ℚ).den = 1) :
  ∃ (a_int b_int : ℤ), a = a_int ∧ b = b_int := 
sorry

end NUMINAMATH_GPT_rational_powers_imply_integers_l2015_201533


namespace NUMINAMATH_GPT_point_inside_circle_l2015_201509

theorem point_inside_circle (m : ℝ) : (1 - 2)^2 + (-3 + 1)^2 < m → m > 5 :=
by
  sorry

end NUMINAMATH_GPT_point_inside_circle_l2015_201509


namespace NUMINAMATH_GPT_decreasing_function_range_l2015_201560

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 1) * x + 7 * a - 2 else a ^ x

theorem decreasing_function_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (3 / 8 ≤ a ∧ a < 1 / 2) := 
by
  intro a
  sorry

end NUMINAMATH_GPT_decreasing_function_range_l2015_201560


namespace NUMINAMATH_GPT_deepak_investment_l2015_201590

theorem deepak_investment (D : ℝ) (A : ℝ) (P : ℝ) (Dp : ℝ) (Ap : ℝ) 
  (hA : A = 22500)
  (hP : P = 13800)
  (hDp : Dp = 5400)
  (h_ratio : Dp / P = D / (A + D)) :
  D = 15000 := by
  sorry

end NUMINAMATH_GPT_deepak_investment_l2015_201590


namespace NUMINAMATH_GPT_standard_equation_of_circle_l2015_201596

theorem standard_equation_of_circle :
  (∃ a r, r^2 = (a + 1)^2 + (a - 1)^2 ∧ r^2 = (a - 1)^2 + (a - 3)^2 ∧ a = 1 ∧ r^2 = 4) →
  ∃ r, (x - 1)^2 + (y - 1)^2 = r^2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_standard_equation_of_circle_l2015_201596
