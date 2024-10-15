import Mathlib

namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2865_286574

-- Problem 1
theorem problem_1 : 25 - 9 + (-12) - (-7) = 4 := by sorry

-- Problem 2
theorem problem_2 : 1/9 * (-2)^3 / (2/3)^2 = -2 := by sorry

-- Problem 3
theorem problem_3 : (5/12 + 2/3 - 3/4) * (-12) = -4 := by sorry

-- Problem 4
theorem problem_4 : -1^4 + (-2) / (-1/3) - |(-9)| = -4 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l2865_286574


namespace NUMINAMATH_CALUDE_sum_of_divisors_36_l2865_286563

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_divisors_36 : sum_of_divisors 36 = 91 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_36_l2865_286563


namespace NUMINAMATH_CALUDE_tangent_line_and_extreme_values_l2865_286538

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + 2 / x

theorem tangent_line_and_extreme_values (a : ℝ) :
  (∀ x : ℝ, x > 0 → f a x = Real.log x - a * x + 2 / x) →
  (a = 1 → ∀ x y : ℝ, y = f 1 x → (x = 1 ∧ y = 1) → 2 * x + y - 3 = 0) ∧
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧
    (∀ x : ℝ, x > 0 → f a x ≤ f a x₁ ∧ f a x ≤ f a x₂) ↔ 0 < a ∧ a < 1/8) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_extreme_values_l2865_286538


namespace NUMINAMATH_CALUDE_equation_one_real_root_l2865_286569

theorem equation_one_real_root (t : ℝ) : 
  (∃! x : ℝ, 3 * x + 7 * t - 2 + (2 * t * x^2 + 7 * t^2 - 9) / (x - t) = 0) ↔ 
  (t = -3 ∨ t = -7/2 ∨ t = 1) := by sorry

end NUMINAMATH_CALUDE_equation_one_real_root_l2865_286569


namespace NUMINAMATH_CALUDE_quadratic_root_sum_l2865_286579

theorem quadratic_root_sum (b c : ℝ) (h : c ≠ 0) : 
  (c^2 + 2*b*c - 5*c = 0) → (2*b + c = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_l2865_286579


namespace NUMINAMATH_CALUDE_area_isosceles_right_triangle_l2865_286594

/-- Given a right triangle ABC with AB = 12 and AC = 24, and points D on AC and E on BC
    forming an isosceles right triangle BDE, prove that the area of BDE is 80. -/
theorem area_isosceles_right_triangle (A B C D E : ℝ × ℝ) : 
  -- Right triangle ABC
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) →
  -- AB = 12
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 12 →
  -- AC = 24
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 24 →
  -- D is on AC
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2)) →
  -- E is on BC
  ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (B.1 + s * (C.1 - B.1), B.2 + s * (C.2 - B.2)) →
  -- BDE is an isosceles right triangle
  (D.1 - B.1) * (E.1 - B.1) + (D.2 - B.2) * (E.2 - B.2) = 0 ∧
  (D.1 - B.1)^2 + (D.2 - B.2)^2 = (E.1 - B.1)^2 + (E.2 - B.2)^2 →
  -- Area of BDE is 80
  (1/2) * Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2) * Real.sqrt ((E.1 - B.1)^2 + (E.2 - B.2)^2) = 80 :=
by sorry


end NUMINAMATH_CALUDE_area_isosceles_right_triangle_l2865_286594


namespace NUMINAMATH_CALUDE_blocks_with_two_differences_eq_28_l2865_286573

/-- Represents the number of options for each category of block attributes -/
structure BlockCategories where
  materials : Nat
  sizes : Nat
  colors : Nat
  shapes : Nat

/-- Calculates the number of blocks differing in exactly two ways from a reference block -/
def blocksWithTwoDifferences (categories : BlockCategories) : Nat :=
  sorry

/-- The specific categories for the given problem -/
def problemCategories : BlockCategories :=
  { materials := 2
  , sizes := 3
  , colors := 5
  , shapes := 4
  }

/-- Theorem stating that the number of blocks differing in exactly two ways is 28 -/
theorem blocks_with_two_differences_eq_28 :
  blocksWithTwoDifferences problemCategories = 28 := by
  sorry

end NUMINAMATH_CALUDE_blocks_with_two_differences_eq_28_l2865_286573


namespace NUMINAMATH_CALUDE_cube_side_length_l2865_286541

/-- Given a cube where the length of its space diagonal is 6.92820323027551 m,
    prove that the side length of the cube is 4 m. -/
theorem cube_side_length (d : ℝ) (h : d = 6.92820323027551) : 
  ∃ (a : ℝ), a * Real.sqrt 3 = d ∧ a = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_length_l2865_286541


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l2865_286534

theorem candidate_vote_percentage
  (total_votes : ℕ)
  (invalid_percentage : ℚ)
  (candidate_valid_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15 / 100)
  (h3 : candidate_valid_votes = 285600) :
  (candidate_valid_votes : ℚ) / ((1 - invalid_percentage) * total_votes) = 60 / 100 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l2865_286534


namespace NUMINAMATH_CALUDE_fraction_sum_zero_l2865_286552

theorem fraction_sum_zero (a b c : ℤ) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (avg : b = (a + c) / 2)
  (sum_zero : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_l2865_286552


namespace NUMINAMATH_CALUDE_finite_selector_existence_l2865_286536

theorem finite_selector_existence
  (A B C : ℕ → Set ℕ)
  (h_finite : ∀ i, (A i).Finite ∧ (B i).Finite ∧ (C i).Finite)
  (h_disjoint : ∀ i, Disjoint (A i) (B i) ∧ Disjoint (A i) (C i) ∧ Disjoint (B i) (C i))
  (h_cover : ∀ X Y Z : Set ℕ, Disjoint X Y ∧ Disjoint X Z ∧ Disjoint Y Z → X ∪ Y ∪ Z = univ →
    ∃ i, A i ⊆ X ∧ B i ⊆ Y ∧ C i ⊆ Z) :
  ∃ S : Finset ℕ, ∀ X Y Z : Set ℕ, Disjoint X Y ∧ Disjoint X Z ∧ Disjoint Y Z → X ∪ Y ∪ Z = univ →
    ∃ i ∈ S, A i ⊆ X ∧ B i ⊆ Y ∧ C i ⊆ Z :=
by sorry

end NUMINAMATH_CALUDE_finite_selector_existence_l2865_286536


namespace NUMINAMATH_CALUDE_identity_proof_l2865_286559

theorem identity_proof (A B C A₁ B₁ C₁ : ℝ) :
  (A^2 + B^2 + C^2) * (A₁^2 + B₁^2 + C₁^2) - (A*A₁ + B*B₁ + C*C₁)^2 =
  (A*B₁ + A₁*B)^2 + (A*C₁ + A₁*C)^2 + (B*C₁ + B₁*C)^2 := by
  sorry

end NUMINAMATH_CALUDE_identity_proof_l2865_286559


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2865_286588

theorem quadratic_equation_solution (a b : ℝ) (ha : a ≠ 0) :
  let x := (a^2 - b^2) / (2*a)
  x^2 + b^2 = (a - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2865_286588


namespace NUMINAMATH_CALUDE_stock_price_increase_l2865_286593

/-- Proves that if a stock's price increases by 50% and closes at $15, then its opening price was $10. -/
theorem stock_price_increase (opening_price closing_price : ℝ) :
  closing_price = 15 ∧ closing_price = opening_price * 1.5 → opening_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_increase_l2865_286593


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2865_286518

theorem infinite_geometric_series_first_term 
  (r : ℚ) (S : ℚ) (a : ℚ) 
  (h1 : r = -1/3) 
  (h2 : S = 9) 
  (h3 : S = a / (1 - r)) : 
  a = 12 := by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l2865_286518


namespace NUMINAMATH_CALUDE_point_symmetry_l2865_286585

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other. -/
def symmetric_wrt_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

/-- The point (3,4) -/
def point1 : ℝ × ℝ := (3, 4)

/-- The point (3,-4) -/
def point2 : ℝ × ℝ := (3, -4)

/-- Theorem stating that point1 and point2 are symmetric with respect to the x-axis -/
theorem point_symmetry : symmetric_wrt_x_axis point1 point2 := by sorry

end NUMINAMATH_CALUDE_point_symmetry_l2865_286585


namespace NUMINAMATH_CALUDE_profit_percentage_proof_l2865_286530

/-- Given that the cost price of 20 articles equals the selling price of 16 articles,
    prove that the profit percentage is 25%. -/
theorem profit_percentage_proof (C S : ℝ) (h : 20 * C = 16 * S) :
  (S - C) / C * 100 = 25 :=
sorry

end NUMINAMATH_CALUDE_profit_percentage_proof_l2865_286530


namespace NUMINAMATH_CALUDE_unique_H_value_l2865_286507

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

end NUMINAMATH_CALUDE_unique_H_value_l2865_286507


namespace NUMINAMATH_CALUDE_abs_x_leq_2_necessary_not_sufficient_l2865_286590

theorem abs_x_leq_2_necessary_not_sufficient :
  (∃ x : ℝ, |x + 1| ≤ 1 ∧ ¬(|x| ≤ 2)) = False ∧
  (∃ x : ℝ, |x| ≤ 2 ∧ ¬(|x + 1| ≤ 1)) = True :=
by sorry

end NUMINAMATH_CALUDE_abs_x_leq_2_necessary_not_sufficient_l2865_286590


namespace NUMINAMATH_CALUDE_task_completion_time_l2865_286524

/-- The number of days A takes to complete the task -/
def days_A : ℚ := 12

/-- The efficiency ratio of B compared to A -/
def efficiency_B : ℚ := 1.75

/-- The number of days B takes to complete the task -/
def days_B : ℚ := 48 / 7

theorem task_completion_time :
  days_B = days_A / efficiency_B := by sorry

end NUMINAMATH_CALUDE_task_completion_time_l2865_286524


namespace NUMINAMATH_CALUDE_time_spent_on_activities_l2865_286537

theorem time_spent_on_activities (hours_A hours_B : ℕ) : 
  hours_A = 6 → hours_A = hours_B + 3 → hours_B = 3 := by
  sorry

end NUMINAMATH_CALUDE_time_spent_on_activities_l2865_286537


namespace NUMINAMATH_CALUDE_hours_per_day_l2865_286571

theorem hours_per_day (days : ℕ) (total_hours : ℕ) (h1 : days = 6) (h2 : total_hours = 18) :
  total_hours / days = 3 := by
  sorry

end NUMINAMATH_CALUDE_hours_per_day_l2865_286571


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_intersection_of_A_l2865_286508

-- Part I
def A : Set (ℝ × ℝ) := {p | p.2 = p.1^2 + 2}
def B : Set (ℝ × ℝ) := {p | p.2 = 6 - p.1^2}

theorem intersection_of_A_and_B : A ∩ B = {(Real.sqrt 2, 4), (-Real.sqrt 2, 4)} := by sorry

-- Part II
def A' : Set ℝ := {y | ∃ x, y = x^2 + 2}
def B' : Set ℝ := {y | ∃ x, y = 6 - x^2}

theorem intersection_of_A'_and_B' : A' ∩ B' = {y | 2 ≤ y ∧ y ≤ 6} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_intersection_of_A_l2865_286508


namespace NUMINAMATH_CALUDE_income_change_percentage_l2865_286551

theorem income_change_percentage 
  (original_payment : ℝ) 
  (original_time : ℝ) 
  (payment_increase_rate : ℝ) 
  (time_decrease_rate : ℝ) 
  (h1 : payment_increase_rate = 0.3333) 
  (h2 : time_decrease_rate = 0.3333) :
  let new_payment := original_payment * (1 + payment_increase_rate)
  let new_time := original_time * (1 - time_decrease_rate)
  let original_income := original_payment * original_time
  let new_income := new_payment * new_time
  (new_income - original_income) / original_income = -0.1111 := by
sorry

end NUMINAMATH_CALUDE_income_change_percentage_l2865_286551


namespace NUMINAMATH_CALUDE_percentage_red_cars_chennai_l2865_286550

/-- Percentage of red cars in the total car population -/
def percentage_red_cars (total_cars : ℕ) (honda_cars : ℕ) (honda_red_ratio : ℚ) (non_honda_red_ratio : ℚ) : ℚ :=
  let non_honda_cars := total_cars - honda_cars
  let red_honda_cars := honda_red_ratio * honda_cars
  let red_non_honda_cars := non_honda_red_ratio * non_honda_cars
  let total_red_cars := red_honda_cars + red_non_honda_cars
  (total_red_cars / total_cars) * 100

/-- The percentage of red cars in Chennai -/
theorem percentage_red_cars_chennai :
  percentage_red_cars 900 500 (90/100) (225/1000) = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_red_cars_chennai_l2865_286550


namespace NUMINAMATH_CALUDE_william_shared_three_marbles_l2865_286586

/-- The number of marbles William shared with Theresa -/
def marbles_shared (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

theorem william_shared_three_marbles :
  let initial := 10
  let remaining := 7
  marbles_shared initial remaining = 3 := by
  sorry

end NUMINAMATH_CALUDE_william_shared_three_marbles_l2865_286586


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2865_286595

theorem perfect_square_trinomial (m n : ℝ) :
  (4 / 9) * m^2 + (4 / 3) * m * n + n^2 = ((2 / 3) * m + n)^2 := by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2865_286595


namespace NUMINAMATH_CALUDE_set_equation_solution_l2865_286564

theorem set_equation_solution (A X Y : Set α) 
  (h1 : X ∪ Y = A) 
  (h2 : X ∩ A = Y) : 
  X = A ∧ Y = A := by
  sorry

end NUMINAMATH_CALUDE_set_equation_solution_l2865_286564


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2865_286544

theorem unique_solution_condition (j : ℝ) : 
  (∃! x : ℝ, (2*x + 7)*(x - 5) = -43 + j*x) ↔ (j = 5 ∨ j = -11) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2865_286544


namespace NUMINAMATH_CALUDE_video_count_l2865_286575

theorem video_count (video_length : ℝ) (lila_speed : ℝ) (roger_speed : ℝ) (total_time : ℝ) :
  video_length = 100 →
  lila_speed = 2 →
  roger_speed = 1 →
  total_time = 900 →
  ∃ n : ℕ, (n : ℝ) * (video_length / lila_speed + video_length / roger_speed) = total_time ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_video_count_l2865_286575


namespace NUMINAMATH_CALUDE_bicycle_profit_problem_l2865_286566

theorem bicycle_profit_problem (initial_cost final_price : ℝ) : 
  (initial_cost * 1.25 * 1.25 = final_price) →
  (final_price = 225) →
  (initial_cost = 144) := by
sorry

end NUMINAMATH_CALUDE_bicycle_profit_problem_l2865_286566


namespace NUMINAMATH_CALUDE_round_201949_to_two_sig_figs_l2865_286587

/-- Rounds a number to a specified number of significant figures in scientific notation -/
def roundToSignificantFigures (x : ℝ) (sigFigs : ℕ) : ℝ := sorry

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

theorem round_201949_to_two_sig_figs :
  let number : ℝ := 201949
  let rounded := roundToSignificantFigures number 2
  ∃ (sn : ScientificNotation), 
    sn.coefficient = 2.0 ∧ 
    sn.exponent = 5 ∧ 
    rounded = sn.coefficient * (10 : ℝ) ^ sn.exponent :=
sorry

end NUMINAMATH_CALUDE_round_201949_to_two_sig_figs_l2865_286587


namespace NUMINAMATH_CALUDE_product_quality_comparison_l2865_286596

structure MachineData where
  first_class : ℕ
  second_class : ℕ
  total : ℕ

def machine_a : MachineData := ⟨150, 50, 200⟩
def machine_b : MachineData := ⟨120, 80, 200⟩

def total_products : ℕ := 400

def k_squared (n a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem product_quality_comparison :
  (machine_a.first_class : ℚ) / machine_a.total = 3/4 ∧
  (machine_b.first_class : ℚ) / machine_b.total = 3/5 ∧
  6.635 < k_squared total_products machine_a.first_class machine_a.second_class
    machine_b.first_class machine_b.second_class ∧
  k_squared total_products machine_a.first_class machine_a.second_class
    machine_b.first_class machine_b.second_class < 10.828 := by
  sorry

end NUMINAMATH_CALUDE_product_quality_comparison_l2865_286596


namespace NUMINAMATH_CALUDE_janet_investment_l2865_286503

/-- Calculates the total investment amount given the conditions of Janet's investment -/
theorem janet_investment
  (rate1 rate2 : ℚ)
  (interest_total : ℚ)
  (investment_at_rate1 : ℚ)
  (h1 : rate1 = 1/10)
  (h2 : rate2 = 1/100)
  (h3 : interest_total = 1390)
  (h4 : investment_at_rate1 = 12000)
  (h5 : investment_at_rate1 * rate1 + (total - investment_at_rate1) * rate2 = interest_total) :
  ∃ (total : ℚ), total = 31000 := by
  sorry

end NUMINAMATH_CALUDE_janet_investment_l2865_286503


namespace NUMINAMATH_CALUDE_no_valid_triples_l2865_286543

theorem no_valid_triples :
  ¬ ∃ (x y z : ℕ),
    (1 ≤ x) ∧ (x ≤ y) ∧ (y ≤ z) ∧
    (x * y * z + 2 * (x * y + y * z + z * x) = 2 * (2 * (x * y + y * z + z * x)) + 12) :=
by sorry


end NUMINAMATH_CALUDE_no_valid_triples_l2865_286543


namespace NUMINAMATH_CALUDE_snake_diet_decade_l2865_286560

/-- The number of mice a snake eats in a decade -/
def mice_eaten_in_decade (weeks_per_mouse : ℕ) (weeks_per_year : ℕ) (years_per_decade : ℕ) : ℕ :=
  (weeks_per_year / weeks_per_mouse) * years_per_decade

/-- Theorem: A snake eating one mouse every 4 weeks will eat 130 mice in a decade -/
theorem snake_diet_decade : 
  mice_eaten_in_decade 4 52 10 = 130 := by
  sorry

#eval mice_eaten_in_decade 4 52 10

end NUMINAMATH_CALUDE_snake_diet_decade_l2865_286560


namespace NUMINAMATH_CALUDE_tangent_line_and_m_range_l2865_286514

noncomputable def f (x : ℝ) : ℝ := x * (Real.log x - 1) + Real.log x + 1

theorem tangent_line_and_m_range :
  (∀ x : ℝ, x > 0 → (x - f x - 1 = 0 → x = 1)) ∧
  (∀ m : ℝ, (∀ x : ℝ, x > 0 → x^2 + x * (m - (Real.log x + 1/x)) + 1 ≥ 0) ↔ m ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_m_range_l2865_286514


namespace NUMINAMATH_CALUDE_book_purchase_remaining_money_l2865_286568

theorem book_purchase_remaining_money (m : ℚ) (n : ℕ) (b : ℚ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : b > 0) 
  (h4 : (1/4) * m = (1/2) * n * b) : 
  m - n * b = (1/2) * m := by
sorry

end NUMINAMATH_CALUDE_book_purchase_remaining_money_l2865_286568


namespace NUMINAMATH_CALUDE_car_motorcycle_transaction_loss_l2865_286577

theorem car_motorcycle_transaction_loss : 
  ∀ (car_cost motorcycle_cost : ℝ),
  car_cost * (1 - 0.25) = 16000 →
  motorcycle_cost * (1 + 0.25) = 16000 →
  car_cost + motorcycle_cost - 2 * 16000 = 2133.33 := by
sorry

end NUMINAMATH_CALUDE_car_motorcycle_transaction_loss_l2865_286577


namespace NUMINAMATH_CALUDE_power_mod_thirteen_l2865_286502

theorem power_mod_thirteen : 7^2000 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_thirteen_l2865_286502


namespace NUMINAMATH_CALUDE_men_to_women_ratio_l2865_286565

theorem men_to_women_ratio (men : ℝ) (women : ℝ) (h : women = 0.9 * men) :
  (men / women) * 100 = (1 / 0.9) * 100 := by
sorry

end NUMINAMATH_CALUDE_men_to_women_ratio_l2865_286565


namespace NUMINAMATH_CALUDE_daily_wage_of_c_l2865_286558

/-- Represents the daily wage and work days of a worker -/
structure Worker where
  dailyWage : ℚ
  workDays : ℕ

theorem daily_wage_of_c (a b c : Worker) 
  (ratio_a_b : a.dailyWage / b.dailyWage = 3 / 4)
  (ratio_b_c : b.dailyWage / c.dailyWage = 4 / 5)
  (work_days : a.workDays = 6 ∧ b.workDays = 9 ∧ c.workDays = 4)
  (total_earning : a.dailyWage * a.workDays + b.dailyWage * b.workDays + c.dailyWage * c.workDays = 1850) :
  c.dailyWage = 625 / 3 := by
  sorry


end NUMINAMATH_CALUDE_daily_wage_of_c_l2865_286558


namespace NUMINAMATH_CALUDE_chocolate_savings_theorem_l2865_286500

/-- Represents the cost and packaging details of a chocolate store -/
structure ChocolateStore where
  cost_per_chocolate : ℚ
  pack_size : ℕ

/-- Calculates the cost for a given number of weeks at a store -/
def calculate_cost (store : ChocolateStore) (weeks : ℕ) : ℚ :=
  let chocolates_needed := 2 * weeks
  let packs_needed := (chocolates_needed + store.pack_size - 1) / store.pack_size
  ↑packs_needed * store.pack_size * store.cost_per_chocolate

/-- The problem statement -/
theorem chocolate_savings_theorem :
  let local_store := ChocolateStore.mk 3 1
  let store_a := ChocolateStore.mk 2 5
  let store_b := ChocolateStore.mk (5/2) 1
  let store_c := ChocolateStore.mk (9/5) 10
  let weeks := 13
  let local_cost := calculate_cost local_store weeks
  let cost_a := calculate_cost store_a weeks
  let cost_b := calculate_cost store_b weeks
  let cost_c := calculate_cost store_c weeks
  let savings_a := local_cost - cost_a
  let savings_b := local_cost - cost_b
  let savings_c := local_cost - cost_c
  let max_savings := max savings_a (max savings_b savings_c)
  max_savings = 28 := by sorry

end NUMINAMATH_CALUDE_chocolate_savings_theorem_l2865_286500


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2865_286532

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |3*x + 1| - |x - 1| < 0} = {x : ℝ | -1 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2865_286532


namespace NUMINAMATH_CALUDE_danny_initial_caps_l2865_286580

/-- The number of bottle caps Danny found at the park -/
def found_caps : ℕ := 7

/-- The total number of bottle caps Danny has after adding the found ones -/
def total_caps : ℕ := 32

/-- The number of bottle caps Danny had before finding the ones at the park -/
def initial_caps : ℕ := total_caps - found_caps

theorem danny_initial_caps : initial_caps = 25 := by
  sorry

end NUMINAMATH_CALUDE_danny_initial_caps_l2865_286580


namespace NUMINAMATH_CALUDE_flag_arrangements_l2865_286584

/-- The number of distinguishable arrangements of flags on two flagpoles -/
def N : ℕ := 858

/-- The number of red flags -/
def red_flags : ℕ := 12

/-- The number of yellow flags -/
def yellow_flags : ℕ := 11

/-- The total number of flags -/
def total_flags : ℕ := red_flags + yellow_flags

/-- Theorem stating that N is the correct number of distinguishable arrangements -/
theorem flag_arrangements :
  N = (red_flags - 1) * (Nat.choose (red_flags + 1) yellow_flags) :=
by sorry

end NUMINAMATH_CALUDE_flag_arrangements_l2865_286584


namespace NUMINAMATH_CALUDE_largest_number_problem_l2865_286517

theorem largest_number_problem (a b c : ℝ) : 
  a < b ∧ b < c →
  a + b + c = 72 →
  c - b = 5 →
  b - a = 8 →
  c = 30 := by
sorry

end NUMINAMATH_CALUDE_largest_number_problem_l2865_286517


namespace NUMINAMATH_CALUDE_number_division_remainders_l2865_286555

theorem number_division_remainders (N : ℤ) (h : N % 1554 = 131) : 
  (N % 37 = 20) ∧ (N % 73 = 58) := by
  sorry

end NUMINAMATH_CALUDE_number_division_remainders_l2865_286555


namespace NUMINAMATH_CALUDE_problem_statement_l2865_286527

-- Define proposition p
def p : Prop := ∀ x : ℝ, (|x| = x ↔ x > 0)

-- Define proposition q
def q : Prop := (¬∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)

-- Theorem to prove
theorem problem_statement : ¬(p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2865_286527


namespace NUMINAMATH_CALUDE_fractional_method_experiments_l2865_286591

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- The number of experimental points -/
def num_points : ℕ := 12

/-- The maximum number of additional experiments needed -/
def max_additional_experiments : ℕ := 5

/-- Theorem: Given 12 experimental points and using the fractional method
    to find the optimal point of a unimodal function, the maximum number
    of additional experiments needed is 5. -/
theorem fractional_method_experiments :
  ∃ k : ℕ, num_points = fib (k + 1) - 1 ∧ max_additional_experiments = k :=
sorry

end NUMINAMATH_CALUDE_fractional_method_experiments_l2865_286591


namespace NUMINAMATH_CALUDE_first_player_wins_l2865_286540

-- Define the chessboard as a type
def Chessboard : Type := Unit

-- Define a position on the chessboard
def Position : Type := Nat × Nat

-- Define a move as a function from one position to another
def Move : Type := Position → Position

-- Define the property of a move being valid
def ValidMove (m : Move) (visited : Set Position) : Prop :=
  ∀ p, p ∉ visited → 
    (m p).1 = p.1 ∧ ((m p).2 = p.2 + 1 ∨ (m p).2 = p.2 - 1) ∨
    (m p).2 = p.2 ∧ ((m p).1 = p.1 + 1 ∨ (m p).1 = p.1 - 1)

-- Define the game state
structure GameState :=
  (position : Position)
  (visited : Set Position)

-- Define the property of a player having a winning strategy
def HasWinningStrategy (player : Nat) : Prop :=
  ∀ (state : GameState),
    ∃ (m : Move), ValidMove m state.visited →
      ¬∃ (m' : Move), ValidMove m' (insert (m state.position) state.visited)

-- Theorem statement
theorem first_player_wins :
  HasWinningStrategy 0 :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l2865_286540


namespace NUMINAMATH_CALUDE_cake_eating_contest_l2865_286542

theorem cake_eating_contest : (7 : ℚ) / 8 - (5 : ℚ) / 6 = (1 : ℚ) / 24 := by
  sorry

end NUMINAMATH_CALUDE_cake_eating_contest_l2865_286542


namespace NUMINAMATH_CALUDE_four_square_base_boxes_l2865_286521

/-- A box with a square base that can contain exactly 64 unit cubes. -/
structure SquareBaseBox where
  base : ℕ
  height : ℕ
  volume_eq_64 : base * base * height = 64

/-- The set of all possible SquareBaseBox configurations. -/
def all_square_base_boxes : Set SquareBaseBox :=
  { box | box.base * box.base * box.height = 64 }

/-- The theorem stating that there are exactly four possible SquareBaseBox configurations. -/
theorem four_square_base_boxes :
  all_square_base_boxes = {
    ⟨1, 64, rfl⟩,
    ⟨2, 16, rfl⟩,
    ⟨4, 4, rfl⟩,
    ⟨8, 1, rfl⟩
  } := by sorry

end NUMINAMATH_CALUDE_four_square_base_boxes_l2865_286521


namespace NUMINAMATH_CALUDE_mollys_age_l2865_286589

/-- Given that the ratio of Sandy's age to Molly's age is 4:3,
    and Sandy will be 34 years old in 6 years,
    prove that Molly's current age is 21 years. -/
theorem mollys_age (sandy_age molly_age : ℕ) : 
  (sandy_age : ℚ) / molly_age = 4 / 3 →
  sandy_age + 6 = 34 →
  molly_age = 21 := by
  sorry

end NUMINAMATH_CALUDE_mollys_age_l2865_286589


namespace NUMINAMATH_CALUDE_greatest_power_of_two_l2865_286553

theorem greatest_power_of_two (n : ℕ) : 
  (∃ k : ℕ, 2^k ∣ (12^603 - 8^402) ∧ 
   ∀ m : ℕ, 2^m ∣ (12^603 - 8^402) → m ≤ k) → 
  n = 1209 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_l2865_286553


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2865_286583

theorem gcd_lcm_product (a b : ℕ) (h : a = 90 ∧ b = 135) : 
  (Nat.gcd a b) * (Nat.lcm a b) = 12150 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2865_286583


namespace NUMINAMATH_CALUDE_range_of_function_l2865_286598

theorem range_of_function (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 2) :
  0 < 2 * a + b ∧ 2 * a + b < 10 := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l2865_286598


namespace NUMINAMATH_CALUDE_min_value_inequality_l2865_286554

theorem min_value_inequality (a : ℝ) (h : a > 1) :
  a + 2 / (a - 1) ≥ 1 + 2 * Real.sqrt 2 ∧
  ∃ a₀ > 1, a₀ + 2 / (a₀ - 1) = 1 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2865_286554


namespace NUMINAMATH_CALUDE_limit_tan_sin_ratio_l2865_286570

open Real

noncomputable def f (x : ℝ) : ℝ := tan (6 * x) / sin (3 * x)

theorem limit_tan_sin_ratio :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |f x - 2| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_tan_sin_ratio_l2865_286570


namespace NUMINAMATH_CALUDE_f_max_value_l2865_286556

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + Real.sin x) + Real.sin (x - Real.sin x) + (Real.pi / 2 - 2) * Real.sin (Real.sin x)

theorem f_max_value : 
  ∃ (M : ℝ), M = (Real.pi - 2) / Real.sqrt 2 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_f_max_value_l2865_286556


namespace NUMINAMATH_CALUDE_point_satisfies_inequalities_l2865_286561

-- Define the system of inequalities
def satisfies_inequalities (x y : ℝ) : Prop :=
  (x - 2*y + 5 > 0) ∧ (x - y + 3 ≤ 0)

-- Theorem statement
theorem point_satisfies_inequalities : 
  satisfies_inequalities (-2 : ℝ) (1 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_point_satisfies_inequalities_l2865_286561


namespace NUMINAMATH_CALUDE_painted_cube_probability_l2865_286509

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

end NUMINAMATH_CALUDE_painted_cube_probability_l2865_286509


namespace NUMINAMATH_CALUDE_complementary_angle_measure_l2865_286523

-- Define the angle
def angle : ℝ := 45

-- Define the relationship between supplementary and complementary angles
def supplementary_complementary_relation (supplementary complementary : ℝ) : Prop :=
  supplementary = 3 * complementary

-- Define the supplementary angle
def supplementary (a : ℝ) : ℝ := 180 - a

-- Define the complementary angle
def complementary (a : ℝ) : ℝ := 90 - a

-- Theorem statement
theorem complementary_angle_measure :
  supplementary_complementary_relation (supplementary angle) (complementary angle) →
  complementary angle = 45 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angle_measure_l2865_286523


namespace NUMINAMATH_CALUDE_lunch_combinations_l2865_286513

/-- The number of different types of meat dishes -/
def num_meat_dishes : ℕ := 4

/-- The number of different types of vegetable dishes -/
def num_veg_dishes : ℕ := 7

/-- The number of meat dishes chosen in the first combination method -/
def meat_choice_1 : ℕ := 2

/-- The number of vegetable dishes chosen in both combination methods -/
def veg_choice : ℕ := 2

/-- The number of meat dishes chosen in the second combination method -/
def meat_choice_2 : ℕ := 1

/-- The total number of lunch combinations -/
def total_combinations : ℕ := Nat.choose num_meat_dishes meat_choice_1 * Nat.choose num_veg_dishes veg_choice +
                               Nat.choose num_meat_dishes meat_choice_2 * Nat.choose num_veg_dishes veg_choice

theorem lunch_combinations : total_combinations = 210 := by
  sorry

end NUMINAMATH_CALUDE_lunch_combinations_l2865_286513


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_not_square_of_radii_ratio_l2865_286562

theorem sphere_volume_ratio_not_square_of_radii_ratio (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) :
  (4 * π * r₁^3 / 3) / (4 * π * r₂^3 / 3) ≠ (r₁ / r₂)^2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_not_square_of_radii_ratio_l2865_286562


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_relation_l2865_286548

/-- In an isosceles right triangle ABC with right angle at A, 
    if CB = CA = h, BM + MA = 2(BC + CA), and MB = x, then x = 7h/5 -/
theorem isosceles_right_triangle_relation (h x : ℝ) : 
  h > 0 → 
  x > 0 → 
  x + Real.sqrt ((x + h)^2 + h^2) = 4 * h → 
  x = (7 * h) / 5 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_relation_l2865_286548


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l2865_286511

/-- A conic section type -/
inductive ConicType
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  (x - 3)^2 = (3*y + 4)^2 - 90

/-- Function to determine the type of conic section -/
def determine_conic_type (eq : (ℝ → ℝ → Prop)) : ConicType :=
  sorry

/-- Theorem stating that the given equation describes a hyperbola -/
theorem conic_is_hyperbola :
  determine_conic_type conic_equation = ConicType.Hyperbola :=
sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l2865_286511


namespace NUMINAMATH_CALUDE_recurrence_relation_and_generating_function_l2865_286581

def a (n : ℕ) : ℝ := (n^2 + 1) * 3^n

theorem recurrence_relation_and_generating_function :
  (∀ n : ℕ, a n - a (n + 1) + (1/3) * a (n + 2) - (1/27) * a (n + 3) = 0) ∧
  (∀ x : ℝ, abs x < 1/3 → ∑' (n : ℕ), a n * x^n = (1 - 3*x + 18*x^2) / (1 - 9*x + 27*x^2 - 27*x^3)) :=
by sorry

end NUMINAMATH_CALUDE_recurrence_relation_and_generating_function_l2865_286581


namespace NUMINAMATH_CALUDE_sine_ratio_in_triangle_l2865_286533

theorem sine_ratio_in_triangle (a b c : ℝ) (A B C : ℝ) :
  (b + c) / (c + a) = 4 / 5 ∧
  (c + a) / (a + b) = 5 / 6 ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C →
  Real.sin A / Real.sin B = 7 / 5 ∧
  Real.sin B / Real.sin C = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_sine_ratio_in_triangle_l2865_286533


namespace NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l2865_286531

theorem no_real_roots_for_nonzero_k :
  ∀ k : ℝ, k ≠ 0 → ¬∃ x : ℝ, x^2 + k*x + 3*k^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_for_nonzero_k_l2865_286531


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2865_286539

theorem right_triangle_hypotenuse (area : ℝ) (leg : ℝ) (hypotenuse : ℝ) :
  area = 320 →
  leg = 16 →
  area = (1 / 2) * leg * (area / (1 / 2 * leg)) →
  hypotenuse^2 = leg^2 + (area / (1 / 2 * leg))^2 →
  hypotenuse = 4 * Real.sqrt 116 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2865_286539


namespace NUMINAMATH_CALUDE_constraint_implies_equality_and_minimum_value_l2865_286557

open Real

-- Define the constraint function
def constraint (a b c : ℝ) : Prop :=
  exp (a - c) + b * exp (c + 1) ≤ a + log b + 3

-- Define the objective function
def objective (a b c : ℝ) : ℝ :=
  a + b + 2 * c

-- Theorem statement
theorem constraint_implies_equality_and_minimum_value
  (a b c : ℝ) (h : constraint a b c) :
  a = c ∧ ∀ x y z, constraint x y z → objective a b c ≤ objective x y z ∧ objective a b c = -3 * log 3 :=
sorry

end NUMINAMATH_CALUDE_constraint_implies_equality_and_minimum_value_l2865_286557


namespace NUMINAMATH_CALUDE_fraction_simplification_l2865_286567

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  a^2 / (a * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2865_286567


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l2865_286512

/-- Given a parabola y² = 2px (p > 0) with a point A(4, m) on it,
    if the distance from A to the focus is 17/4, then p = 1/2. -/
theorem parabola_focus_distance (p : ℝ) (m : ℝ) : 
  p > 0 → 
  m^2 = 2*p*4 → 
  (4 - p/2)^2 + m^2 = (17/4)^2 → 
  p = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l2865_286512


namespace NUMINAMATH_CALUDE_remainder_problem_l2865_286506

theorem remainder_problem (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 35 = 28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2865_286506


namespace NUMINAMATH_CALUDE_product_of_roots_l2865_286522

theorem product_of_roots (b : ℝ) : 
  (∃ x₁ x₂ : ℝ, 2020 * x₁^2 + b * x₁ + 2021 = 0 ∧ 2020 * x₂^2 + b * x₂ + 2021 = 0 ∧ x₁ ≠ x₂) →
  (∃ y₁ y₂ : ℝ, 2019 * y₁^2 + b * y₁ + 2020 = 0 ∧ 2019 * y₂^2 + b * y₂ + 2020 = 0 ∧ y₁ ≠ y₂) →
  (∃ z₁ z₂ : ℝ, z₁^2 + b * z₁ + 2019 = 0 ∧ z₂^2 + b * z₂ + 2019 = 0 ∧ z₁ ≠ z₂) →
  (2021 / 2020) * (2020 / 2019) * 2019 = 2021 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l2865_286522


namespace NUMINAMATH_CALUDE_recipe_liquid_sum_l2865_286520

/-- Given the amounts of oil and water used in a recipe, 
    prove that the total amount of liquid is their sum. -/
theorem recipe_liquid_sum (oil water : ℝ) 
  (h_oil : oil = 0.17) 
  (h_water : water = 1.17) : 
  oil + water = 1.34 := by
  sorry

end NUMINAMATH_CALUDE_recipe_liquid_sum_l2865_286520


namespace NUMINAMATH_CALUDE_chess_team_girls_l2865_286599

theorem chess_team_girls (total : ℕ) (boys girls : ℕ) 
  (h1 : total = boys + girls)
  (h2 : total = 26)
  (h3 : 3 * boys / 4 + girls / 4 = 13) : 
  girls = 13 := by
sorry

end NUMINAMATH_CALUDE_chess_team_girls_l2865_286599


namespace NUMINAMATH_CALUDE_quadratic_solution_product_l2865_286516

theorem quadratic_solution_product (p q : ℝ) : 
  (3 * p^2 - 9 * p - 15 = 0) → 
  (3 * q^2 - 9 * q - 15 = 0) → 
  (3 * p - 5) * (6 * q - 10) = -130 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_product_l2865_286516


namespace NUMINAMATH_CALUDE_pizza_fraction_proof_l2865_286505

theorem pizza_fraction_proof (michael_fraction lamar_fraction treshawn_fraction : ℚ) : 
  michael_fraction = 1/3 →
  lamar_fraction = 1/6 →
  michael_fraction + lamar_fraction + treshawn_fraction = 1 →
  treshawn_fraction = 1/2 := by
sorry

end NUMINAMATH_CALUDE_pizza_fraction_proof_l2865_286505


namespace NUMINAMATH_CALUDE_ngo_wage_problem_l2865_286529

/-- The NGO wage problem -/
theorem ngo_wage_problem (illiterate_count : ℕ) (literate_count : ℕ) 
  (initial_illiterate_wage : ℚ) (average_decrease : ℚ) :
  illiterate_count = 20 →
  literate_count = 10 →
  initial_illiterate_wage = 25 →
  average_decrease = 10 →
  ∃ (new_illiterate_wage : ℚ),
    new_illiterate_wage = 10 ∧
    illiterate_count * (initial_illiterate_wage - new_illiterate_wage) = 
      (illiterate_count + literate_count) * average_decrease :=
by sorry

end NUMINAMATH_CALUDE_ngo_wage_problem_l2865_286529


namespace NUMINAMATH_CALUDE_white_ball_estimate_l2865_286592

/-- Represents the result of drawing balls from a bag -/
structure BagDrawResult where
  totalBalls : ℕ
  totalDraws : ℕ
  whiteDraws : ℕ

/-- Calculates the estimated number of white balls in the bag -/
def estimateWhiteBalls (result : BagDrawResult) : ℚ :=
  result.totalBalls * (result.whiteDraws : ℚ) / result.totalDraws

theorem white_ball_estimate (result : BagDrawResult) 
  (h1 : result.totalBalls = 20)
  (h2 : result.totalDraws = 100)
  (h3 : result.whiteDraws = 40) :
  estimateWhiteBalls result = 8 := by
  sorry

#eval estimateWhiteBalls { totalBalls := 20, totalDraws := 100, whiteDraws := 40 }

end NUMINAMATH_CALUDE_white_ball_estimate_l2865_286592


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2865_286578

theorem contrapositive_equivalence (M : Set α) (a b : α) :
  (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2865_286578


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2865_286546

theorem circle_area_ratio (r : ℝ) (h : r > 0) : (π * (3 * r)^2) / (π * r^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2865_286546


namespace NUMINAMATH_CALUDE_square_geq_bound_l2865_286504

theorem square_geq_bound (a : ℝ) : (∀ x > 1, x^2 ≥ a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_square_geq_bound_l2865_286504


namespace NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l2865_286545

theorem sum_and_ratio_to_difference (x y : ℝ) :
  x + y = 520 → x / y = 0.75 → y - x = 74 := by sorry

end NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l2865_286545


namespace NUMINAMATH_CALUDE_volunteers_arrangement_count_l2865_286501

/-- The number of ways to arrange volunteers for tasks. -/
def arrangeVolunteers (volunteers : ℕ) (tasks : ℕ) : ℕ :=
  (tasks - 1).choose (volunteers - 1) * volunteers.factorial

/-- Theorem stating the number of arrangements for 4 volunteers and 5 tasks. -/
theorem volunteers_arrangement_count :
  arrangeVolunteers 4 5 = 240 := by
  sorry

end NUMINAMATH_CALUDE_volunteers_arrangement_count_l2865_286501


namespace NUMINAMATH_CALUDE_sum_of_odd_prime_divisors_of_90_l2865_286519

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a number is a divisor of 90
def isDivisorOf90 (n : ℕ) : Prop :=
  90 % n = 0

-- Define a function to check if a number is odd
def isOdd (n : ℕ) : Prop :=
  n % 2 ≠ 0

-- Theorem statement
theorem sum_of_odd_prime_divisors_of_90 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, isPrime n ∧ isOdd n ∧ isDivisorOf90 n) ∧ 
    (∀ n : ℕ, isPrime n → isOdd n → isDivisorOf90 n → n ∈ S) ∧
    (S.sum id = 8) :=
sorry

end NUMINAMATH_CALUDE_sum_of_odd_prime_divisors_of_90_l2865_286519


namespace NUMINAMATH_CALUDE_time_to_produce_one_item_l2865_286535

/-- Given a machine that can produce 300 items in 2 hours, 
    prove that it takes 0.4 minutes to produce one item. -/
theorem time_to_produce_one_item 
  (total_time : ℝ) 
  (total_items : ℕ) 
  (h1 : total_time = 2) 
  (h2 : total_items = 300) : 
  (total_time / total_items) * 60 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_time_to_produce_one_item_l2865_286535


namespace NUMINAMATH_CALUDE_number_comparison_l2865_286597

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 2 to base 10 -/
def base2ToBase10 (n : ℕ) : ℕ := sorry

theorem number_comparison :
  let a : ℕ := 33
  let b : ℕ := base6ToBase10 52
  let c : ℕ := base2ToBase10 11111
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_number_comparison_l2865_286597


namespace NUMINAMATH_CALUDE_largest_consecutive_sum_l2865_286572

theorem largest_consecutive_sum (n : ℕ) (a : ℕ) (h1 : n > 1) 
  (h2 : n * a + n * (n - 1) / 2 = 2016) : 
  a + (n - 1) ≤ 673 := by
sorry

end NUMINAMATH_CALUDE_largest_consecutive_sum_l2865_286572


namespace NUMINAMATH_CALUDE_perpendicular_parallel_transitivity_l2865_286576

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_transitivity 
  (l : Line) (α β : Plane) :
  perp l α → para α β → perp l β :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_transitivity_l2865_286576


namespace NUMINAMATH_CALUDE_empty_subset_of_disjoint_nonempty_l2865_286515

theorem empty_subset_of_disjoint_nonempty (A B : Set α) :
  A ≠ ∅ → A ∩ B = ∅ → ∅ ⊆ B := by sorry

end NUMINAMATH_CALUDE_empty_subset_of_disjoint_nonempty_l2865_286515


namespace NUMINAMATH_CALUDE_remainder_sum_l2865_286547

theorem remainder_sum (c d : ℤ) :
  (∃ p : ℤ, c = 84 * p + 76) →
  (∃ q : ℤ, d = 126 * q + 117) →
  (c + d) % 42 = 25 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2865_286547


namespace NUMINAMATH_CALUDE_smallest_possible_value_l2865_286526

theorem smallest_possible_value (x : ℕ+) (m n : ℕ+) : 
  m = 60 →
  Nat.gcd m n = x + 5 →
  Nat.lcm m n = x * (x + 5)^2 →
  n ≥ 2000 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_value_l2865_286526


namespace NUMINAMATH_CALUDE_function_inequality_l2865_286528

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x, (2 - x) / (deriv^[2] f x) ≤ 0)

-- State the theorem
theorem function_inequality : f 1 + f 3 > 2 * f 2 := by sorry

end NUMINAMATH_CALUDE_function_inequality_l2865_286528


namespace NUMINAMATH_CALUDE_sequence_representation_l2865_286582

def is_valid_sequence (q : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → q n < q m ∧ q n < 2 * n

theorem sequence_representation (q : ℕ → ℕ) (h : is_valid_sequence q) :
  ∀ m : ℕ, (∃ i : ℕ, q i = m) ∨ (∃ j k : ℕ, q j - q k = m) :=
by sorry

end NUMINAMATH_CALUDE_sequence_representation_l2865_286582


namespace NUMINAMATH_CALUDE_characterize_square_property_functions_l2865_286525

/-- A function f: ℕ → ℕ satisfies the square property if (f(m) + n)(m + f(n)) is a square for all m, n ∈ ℕ -/
def satisfies_square_property (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, ∃ k : ℕ, (f m + n) * (m + f n) = k * k

/-- The main theorem characterizing functions satisfying the square property -/
theorem characterize_square_property_functions :
  ∀ f : ℕ → ℕ, satisfies_square_property f ↔ ∃ c : ℕ, ∀ n : ℕ, f n = n + c :=
sorry

end NUMINAMATH_CALUDE_characterize_square_property_functions_l2865_286525


namespace NUMINAMATH_CALUDE_solve_class_problem_l2865_286549

def class_problem (num_girls : ℕ) (total_books : ℕ) (girls_books : ℕ) : Prop :=
  ∃ (num_boys : ℕ),
    num_boys = 10 ∧
    num_girls = 15 ∧
    total_books = 375 ∧
    girls_books = 225 ∧
    ∃ (books_per_student : ℕ),
      books_per_student * (num_girls + num_boys) = total_books ∧
      books_per_student * num_girls = girls_books

theorem solve_class_problem :
  class_problem 15 375 225 := by
  sorry

end NUMINAMATH_CALUDE_solve_class_problem_l2865_286549


namespace NUMINAMATH_CALUDE_parallel_resistance_l2865_286510

theorem parallel_resistance (x y r : ℝ) : 
  x = 4 → y = 5 → (1 / r = 1 / x + 1 / y) → r = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_parallel_resistance_l2865_286510
