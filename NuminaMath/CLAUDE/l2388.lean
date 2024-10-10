import Mathlib

namespace min_value_sum_reciprocals_l2388_238812

theorem min_value_sum_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_geometric_mean : Real.sqrt 3 = Real.sqrt (3^x * 3^(3*y))) :
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 1/(3*b) ≥ 1/x + 1/(3*y)) → 1/x + 1/(3*y) = 4 :=
by sorry

end min_value_sum_reciprocals_l2388_238812


namespace fifth_term_of_arithmetic_sequence_l2388_238850

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + d * (n - 1)

theorem fifth_term_of_arithmetic_sequence
  (a d : ℝ)
  (h1 : arithmetic_sequence a d 2 + arithmetic_sequence a d 4 = 10)
  (h2 : arithmetic_sequence a d 1 + arithmetic_sequence a d 3 = 8) :
  arithmetic_sequence a d 5 = 7 := by
sorry

end fifth_term_of_arithmetic_sequence_l2388_238850


namespace base8_246_equals_base10_166_l2388_238897

/-- Converts a base 8 number to base 10 --/
def base8_to_base10 (d2 d1 d0 : ℕ) : ℕ :=
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

/-- The base 8 number 246₈ is equal to 166 in base 10 --/
theorem base8_246_equals_base10_166 : base8_to_base10 2 4 6 = 166 := by
  sorry

end base8_246_equals_base10_166_l2388_238897


namespace shopkeeper_milk_ounces_l2388_238844

/-- Calculates the total amount of milk in fluid ounces bought by a shopkeeper -/
theorem shopkeeper_milk_ounces (packets : ℕ) (ml_per_packet : ℕ) (ml_per_fl_oz : ℕ) 
    (h1 : packets = 150)
    (h2 : ml_per_packet = 250)
    (h3 : ml_per_fl_oz = 30) : 
  (packets * ml_per_packet) / ml_per_fl_oz = 1250 := by
  sorry


end shopkeeper_milk_ounces_l2388_238844


namespace expression_value_l2388_238861

theorem expression_value (a b : ℝ) 
  (h1 : 10 * a^2 - 3 * b^2 + 5 * a * b = 0) 
  (h2 : 9 * a^2 - b^2 ≠ 0) : 
  (2 * a - b) / (3 * a - b) + (5 * b - a) / (3 * a + b) = -3 := by
  sorry

end expression_value_l2388_238861


namespace abs_neg_two_plus_two_l2388_238832

theorem abs_neg_two_plus_two : |(-2 : ℤ)| + 2 = 4 := by
  sorry

end abs_neg_two_plus_two_l2388_238832


namespace sum_of_binary_digits_315_l2388_238863

/-- The sum of the digits in the binary representation of 315 is 6. -/
theorem sum_of_binary_digits_315 : 
  (Nat.digits 2 315).sum = 6 := by sorry

end sum_of_binary_digits_315_l2388_238863


namespace total_travel_time_is_156_hours_l2388_238875

/-- Represents the total travel time of a car journey with specific conditions. -/
def total_travel_time (time_ngapara_zipra : ℝ) : ℝ :=
  let time_ningi_zipra : ℝ := 0.8 * time_ngapara_zipra
  let time_zipra_varnasi : ℝ := 0.75 * time_ningi_zipra
  let delay_time : ℝ := 0.25 * time_ningi_zipra
  time_ngapara_zipra + time_ningi_zipra + delay_time + time_zipra_varnasi

/-- Theorem stating that the total travel time is 156 hours given the specified conditions. -/
theorem total_travel_time_is_156_hours :
  total_travel_time 60 = 156 := by
  sorry

end total_travel_time_is_156_hours_l2388_238875


namespace quadratic_form_ratio_l2388_238806

theorem quadratic_form_ratio (j : ℝ) (c p q : ℝ) : 
  8 * j^2 - 6 * j + 16 = c * (j + p)^2 + q → q / p = -119 / 3 := by
  sorry

end quadratic_form_ratio_l2388_238806


namespace systematic_sampling_interval_l2388_238849

/-- Calculates the sampling interval for systematic sampling. -/
def samplingInterval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

/-- Theorem: The sampling interval for a population of 1500 and sample size of 30 is 50. -/
theorem systematic_sampling_interval :
  samplingInterval 1500 30 = 50 := by
  sorry

#eval samplingInterval 1500 30

end systematic_sampling_interval_l2388_238849


namespace absolute_value_problem_l2388_238809

theorem absolute_value_problem : |-5| + (3 - Real.sqrt 2) ^ 0 - 2 * Real.tan (π / 4) = 4 := by
  sorry

end absolute_value_problem_l2388_238809


namespace answer_key_combinations_l2388_238816

/-- Represents the number of answer choices for a multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- Represents the number of true-false questions -/
def true_false_questions : ℕ := 3

/-- Represents the number of multiple-choice questions -/
def multiple_choice_questions : ℕ := 3

/-- Calculates the number of valid true-false combinations -/
def valid_true_false_combinations : ℕ := 2^true_false_questions - 2

/-- Calculates the number of multiple-choice combinations -/
def multiple_choice_combinations : ℕ := multiple_choice_options^multiple_choice_questions

/-- Theorem: The number of ways to create an answer key for the quiz is 384 -/
theorem answer_key_combinations : 
  valid_true_false_combinations * multiple_choice_combinations = 384 := by
  sorry


end answer_key_combinations_l2388_238816


namespace no_prime_of_form_3811_11_l2388_238824

def a (n : ℕ) : ℕ := 3 * 10^(n+1) + 8 * 10^n + (10^n - 1) / 9

theorem no_prime_of_form_3811_11 (n : ℕ) (h : n ≥ 1) : ¬ Nat.Prime (a n) := by
  sorry

end no_prime_of_form_3811_11_l2388_238824


namespace partition_contains_perfect_square_sum_l2388_238819

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem partition_contains_perfect_square_sum (n : ℕ) : 
  (n ≥ 15) ↔ 
  (∀ (A B : Set ℕ), 
    (A ∪ B = Finset.range n.succ) → 
    (A ∩ B = ∅) → 
    ((∃ (x y : ℕ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ is_perfect_square (x + y)) ∨
     (∃ (x y : ℕ), x ∈ B ∧ y ∈ B ∧ x ≠ y ∧ is_perfect_square (x + y)))) :=
by sorry

end partition_contains_perfect_square_sum_l2388_238819


namespace washing_machine_payment_l2388_238852

theorem washing_machine_payment (remaining_payment : ℝ) (remaining_percentage : ℝ) 
  (part_payment_percentage : ℝ) (h1 : remaining_payment = 3683.33) 
  (h2 : remaining_percentage = 85) (h3 : part_payment_percentage = 15) : 
  (part_payment_percentage / 100) * (remaining_payment / (remaining_percentage / 100)) = 649.95 := by
  sorry

end washing_machine_payment_l2388_238852


namespace adjacent_probability_in_row_of_five_l2388_238803

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

/-- The probability of two specific people sitting adjacent in a row of 5 people -/
theorem adjacent_probability_in_row_of_five :
  let total_arrangements := factorial 5
  let adjacent_arrangements := 2 * factorial 4
  (adjacent_arrangements : ℚ) / total_arrangements = 2 / 5 := by sorry

end adjacent_probability_in_row_of_five_l2388_238803


namespace remainder_product_l2388_238823

theorem remainder_product (n : ℤ) : n % 24 = 19 → (n % 3) * (n % 8) = 3 := by
  sorry

end remainder_product_l2388_238823


namespace f_is_quadratic_l2388_238883

/-- Definition of a quadratic equation in standard form -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 - 1 = 0 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l2388_238883


namespace z_in_first_quadrant_l2388_238817

theorem z_in_first_quadrant (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end z_in_first_quadrant_l2388_238817


namespace solution_set1_correct_solution_set2_correct_l2388_238851

open Set

-- Define the solution sets
def solution_set1 : Set ℝ := Iic (-3) ∪ Ici 1
def solution_set2 : Set ℝ := Ico (-3) 1 ∪ Ioc 3 7

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := (4 - x) / (x^2 + x + 1) ≤ 1
def inequality2 (x : ℝ) : Prop := 1 < |x - 2| ∧ |x - 2| ≤ 5

-- Theorem statements
theorem solution_set1_correct :
  ∀ x : ℝ, x ∈ solution_set1 ↔ inequality1 x :=
sorry

theorem solution_set2_correct :
  ∀ x : ℝ, x ∈ solution_set2 ↔ inequality2 x :=
sorry

end solution_set1_correct_solution_set2_correct_l2388_238851


namespace parallel_condition_l2388_238862

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The line ax + y = 1 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + y = 1

/-- The line x + ay = 2a -/
def line2 (a : ℝ) (x y : ℝ) : Prop := x + a * y = 2 * a

/-- The condition a = -1 is sufficient but not necessary for the lines to be parallel -/
theorem parallel_condition (a : ℝ) : 
  (a = -1 → are_parallel (-a) (1/a)) ∧ 
  ¬(are_parallel (-a) (1/a) → a = -1) :=
sorry

end parallel_condition_l2388_238862


namespace hens_in_coop_l2388_238831

/-- Represents the chicken coop scenario --/
structure ChickenCoop where
  days : ℕ
  eggs_per_hen_per_day : ℕ
  boxes_filled : ℕ
  eggs_per_box : ℕ

/-- Calculates the number of hens in the chicken coop --/
def number_of_hens (coop : ChickenCoop) : ℕ :=
  (coop.boxes_filled * coop.eggs_per_box) / (coop.days * coop.eggs_per_hen_per_day)

/-- Theorem stating the number of hens in the specific scenario --/
theorem hens_in_coop : number_of_hens {
  days := 7,
  eggs_per_hen_per_day := 1,
  boxes_filled := 315,
  eggs_per_box := 6
} = 270 := by sorry

end hens_in_coop_l2388_238831


namespace triangle_point_distance_l2388_238833

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point on a line segment
def PointOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

-- Define the angle between two vectors
def Angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem triangle_point_distance (ABC : Triangle) (D E : ℝ × ℝ) :
  -- Given conditions
  (ABC.A.1 - ABC.B.1)^2 + (ABC.A.2 - ABC.B.2)^2 = 17^2 →
  (ABC.B.1 - ABC.C.1)^2 + (ABC.B.2 - ABC.C.2)^2 = 19^2 →
  (ABC.C.1 - ABC.A.1)^2 + (ABC.C.2 - ABC.A.2)^2 = 16^2 →
  PointOnSegment D ABC.B ABC.C →
  PointOnSegment E ABC.B ABC.C →
  (D.1 - ABC.B.1)^2 + (D.2 - ABC.B.2)^2 = 7^2 →
  Angle ABC.B ABC.A E = Angle ABC.C ABC.A D →
  -- Conclusion
  (E.1 - D.1)^2 + (E.2 - D.2)^2 = (-251/41)^2 :=
by sorry

end triangle_point_distance_l2388_238833


namespace area_enclosed_by_curves_l2388_238884

-- Define the curves
def curve1 (x y : ℝ) : Prop := y^2 = x
def curve2 (x y : ℝ) : Prop := y = x^2

-- Define the enclosed area
noncomputable def enclosed_area : ℝ := sorry

-- Theorem statement
theorem area_enclosed_by_curves : enclosed_area = 1/3 := by sorry

end area_enclosed_by_curves_l2388_238884


namespace units_digit_of_7_to_1023_l2388_238854

theorem units_digit_of_7_to_1023 : ∃ n : ℕ, 7^1023 ≡ 3 [ZMOD 10] :=
  sorry

end units_digit_of_7_to_1023_l2388_238854


namespace limit_rational_function_l2388_238815

theorem limit_rational_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 3| ∧ |x - 3| < δ → 
    |((x^6 - 54*x^3 + 729) / (x^3 - 27)) - 0| < ε :=
by
  sorry

end limit_rational_function_l2388_238815


namespace tangent_line_and_positivity_l2388_238865

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x + 1) - a * Real.log x + a

theorem tangent_line_and_positivity (a : ℝ) (h : a > 0) :
  (∃ m b : ℝ, ∀ x y : ℝ, y = f 1 x → m * x - y + b = 0) ∧
  (∀ x : ℝ, x > 0 → f a x > 0) ↔ 0 < a ∧ a < Real.exp 2 :=
sorry

end tangent_line_and_positivity_l2388_238865


namespace exercise_book_count_l2388_238810

/-- Given a shop with pencils, pens, and exercise books in a specific ratio,
    calculate the number of exercise books based on the number of pencils. -/
theorem exercise_book_count (pencil_ratio : ℕ) (pen_ratio : ℕ) (book_ratio : ℕ) 
    (pencil_count : ℕ) (h1 : pencil_ratio = 10) (h2 : pen_ratio = 2) 
    (h3 : book_ratio = 3) (h4 : pencil_count = 120) : 
    (pencil_count / pencil_ratio) * book_ratio = 36 := by
  sorry

end exercise_book_count_l2388_238810


namespace sqrt_comparison_l2388_238871

theorem sqrt_comparison : Real.sqrt 7 - Real.sqrt 3 < Real.sqrt 6 - Real.sqrt 2 := by
  sorry

end sqrt_comparison_l2388_238871


namespace tennis_percentage_is_31_percent_l2388_238868

/-- The percentage of students who prefer tennis in both schools combined -/
def combined_tennis_percentage (north_total : ℕ) (south_total : ℕ) 
  (north_tennis_percent : ℚ) (south_tennis_percent : ℚ) : ℚ :=
  let north_tennis := (north_total : ℚ) * north_tennis_percent
  let south_tennis := (south_total : ℚ) * south_tennis_percent
  let total_tennis := north_tennis + south_tennis
  let total_students := (north_total + south_total : ℚ)
  total_tennis / total_students

/-- Theorem stating that the percentage of students who prefer tennis in both schools combined is 31% -/
theorem tennis_percentage_is_31_percent :
  combined_tennis_percentage 1800 2700 (25/100) (35/100) = 31/100 := by
  sorry

end tennis_percentage_is_31_percent_l2388_238868


namespace emilys_spending_l2388_238840

theorem emilys_spending (X : ℝ) 
  (friday : X ≥ 0)
  (saturday : 2 * X ≥ 0)
  (sunday : 3 * X ≥ 0)
  (total : X + 2 * X + 3 * X = 120) : X = 20 := by
  sorry

end emilys_spending_l2388_238840


namespace power_product_squared_l2388_238856

theorem power_product_squared (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := by
  sorry

end power_product_squared_l2388_238856


namespace solution_set_equivalence_l2388_238867

theorem solution_set_equivalence (x y : ℝ) :
  (x^2 + 3*x*y + 2*y^2) * (x^2*y^2 - 1) = 0 ↔
  y = -x/2 ∨ y = -x ∨ y = -1/x ∨ y = 1/x :=
by sorry

end solution_set_equivalence_l2388_238867


namespace rental_ratio_proof_l2388_238891

/-- Represents the ratio of dramas to action movies rented during a two-week period -/
def drama_action_ratio : ℚ := 37 / 8

/-- Theorem stating the ratio of dramas to action movies given the rental conditions -/
theorem rental_ratio_proof (T : ℝ) (a : ℝ) (h1 : T > 0) (h2 : a > 0) : 
  (0.64 * T = 10 * a) →  -- Condition: 64% of rentals are comedies, and comedies = 10a
  (∃ d : ℝ, d > 0 ∧ 0.36 * T = a + d) →  -- Condition: Remaining 36% are dramas and action movies
  (∃ s : ℝ, s > 0 ∧ ∃ d : ℝ, d = s * a) →  -- Condition: Dramas are some times action movies
  drama_action_ratio = 37 / 8 :=
sorry

end rental_ratio_proof_l2388_238891


namespace general_inequality_l2388_238859

theorem general_inequality (x n : ℝ) (h1 : x > 0) (h2 : n > 0) 
  (h3 : ∃ (a : ℝ), a > 0 ∧ x + a / x^n ≥ n + 1) :
  ∃ (a : ℝ), a = n^n ∧ x + a / x^n ≥ n + 1 :=
sorry

end general_inequality_l2388_238859


namespace new_video_card_cost_l2388_238835

theorem new_video_card_cost (initial_cost : ℕ) (old_card_sale : ℕ) (total_spent : ℕ) : 
  initial_cost = 1200 →
  old_card_sale = 300 →
  total_spent = 1400 →
  total_spent - (initial_cost - old_card_sale) = 500 := by
sorry

end new_video_card_cost_l2388_238835


namespace investment_growth_l2388_238839

/-- The initial investment amount -/
def P : ℝ := 248.52

/-- The interest rate as a decimal -/
def r : ℝ := 0.12

/-- The number of years -/
def n : ℕ := 6

/-- The final amount -/
def A : ℝ := 500

/-- Theorem stating that the initial investment P, when compounded annually
    at rate r for n years, results in approximately the final amount A -/
theorem investment_growth (ε : ℝ) (h_ε : ε > 0) : 
  |P * (1 + r)^n - A| < ε := by
  sorry


end investment_growth_l2388_238839


namespace nth_equation_holds_l2388_238893

theorem nth_equation_holds (n : ℕ) :
  1 - 1 / ((n + 1: ℚ) ^ 2) = (n / (n + 1 : ℚ)) * ((n + 2) / (n + 1 : ℚ)) := by
  sorry

end nth_equation_holds_l2388_238893


namespace payment_function_correct_l2388_238829

/-- Represents the payment function for book purchases with a discount. -/
def payment_function (x : ℝ) : ℝ :=
  20 * x + 100

/-- Theorem stating the correctness of the payment function. -/
theorem payment_function_correct (x : ℝ) (h : x > 20) :
  payment_function x = (x - 20) * (25 * 0.8) + 20 * 25 := by
  sorry

#check payment_function_correct

end payment_function_correct_l2388_238829


namespace expression_simplification_l2388_238882

theorem expression_simplification (x : ℝ) :
  x = (1/2)⁻¹ + (π - 1)^0 →
  ((x - 3) / (x^2 - 1) - 2 / (x + 1)) / (x / (x^2 - 2*x + 1)) = -2/3 :=
by sorry

end expression_simplification_l2388_238882


namespace circle_tangent_vector_theorem_l2388_238878

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the point A
def A : ℝ × ℝ := (3, 4)

-- Define the vector equation
def VectorEquation (P M N : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), A.1 = x * M.1 + y * N.1 ∧ A.2 = x * M.2 + y * N.2

-- Define the trajectory equation
def TrajectoryEquation (P : ℝ × ℝ) : Prop :=
  P.1 ≠ 0 ∧ P.2 ≠ 0 ∧ P.1^2 / 16 + P.2^2 / 9 = (P.1 + P.2 - 1)^2

theorem circle_tangent_vector_theorem :
  ∀ (P M N : ℝ × ℝ),
    P ∈ Circle (0, 0) 1 →
    VectorEquation P M N →
    TrajectoryEquation P ∧ (∀ x y : ℝ, 9 * x^2 + 16 * y^2 ≥ 4) :=
by sorry

end circle_tangent_vector_theorem_l2388_238878


namespace line_passes_through_fixed_point_l2388_238826

/-- Given that k, -1, and b form an arithmetic sequence,
    prove that the line y = kx + b passes through (1, -2) for all k. -/
theorem line_passes_through_fixed_point (k b : ℝ) :
  ((-1) = (k + b) / 2) →
  ∀ (x y : ℝ), y = k * x + b → (x = 1 ∧ y = -2) :=
by sorry

end line_passes_through_fixed_point_l2388_238826


namespace eight_equidistant_points_l2388_238855

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the specifics of a plane for this problem

/-- A point in 3D space -/
structure Point3D where
  -- We don't need to define the specifics of a point for this problem

/-- Distance between a point and a plane -/
def distance (p : Point3D) (plane : Plane3D) : ℝ :=
  sorry -- Actual implementation not needed for this statement

/-- The set of points at a given distance from a plane -/
def pointsAtDistance (plane : Plane3D) (d : ℝ) : Set Point3D :=
  {p : Point3D | distance p plane = d}

/-- The theorem stating that there are exactly 8 points at given distances from three planes -/
theorem eight_equidistant_points (plane1 plane2 plane3 : Plane3D) (m n p : ℝ) :
  ∃! (points : Finset Point3D),
    points.card = 8 ∧
    ∀ point ∈ points,
      distance point plane1 = m ∧
      distance point plane2 = n ∧
      distance point plane3 = p :=
  sorry


end eight_equidistant_points_l2388_238855


namespace solve_equation_l2388_238842

theorem solve_equation (q r x : ℚ) : 
  (5 / 6 : ℚ) = q / 90 ∧ 
  (5 / 6 : ℚ) = (q + r) / 102 ∧ 
  (5 / 6 : ℚ) = (x - r) / 150 → 
  x = 135 := by
sorry

end solve_equation_l2388_238842


namespace unique_solution_power_equation_l2388_238845

theorem unique_solution_power_equation :
  ∃! (a b c d : ℕ), 7^a = 4^b + 5^c + 6^d :=
by sorry

end unique_solution_power_equation_l2388_238845


namespace curve_family_point_condition_l2388_238896

/-- A point (x, y) lies on at least one curve of the family y = p^2 + (2p - 1)x + 2x^2 
    if and only if y ≥ x^2 - x -/
theorem curve_family_point_condition (x y : ℝ) : 
  (∃ p : ℝ, y = p^2 + (2*p - 1)*x + 2*x^2) ↔ y ≥ x^2 - x := by
sorry

end curve_family_point_condition_l2388_238896


namespace find_c_l2388_238827

def f (x : ℝ) : ℝ := x - 2

def F (x y : ℝ) : ℝ := y^2 + x

theorem find_c (b : ℝ) : ∃ c : ℝ, c = F 3 (f b) ∧ c = 199 := by
  sorry

end find_c_l2388_238827


namespace arthur_muffins_arthur_muffins_proof_l2388_238879

theorem arthur_muffins : ℕ → Prop :=
  fun initial_muffins =>
    initial_muffins + 48 = 83 → initial_muffins = 35

-- Proof
theorem arthur_muffins_proof : arthur_muffins 35 := by
  sorry

end arthur_muffins_arthur_muffins_proof_l2388_238879


namespace intersection_A_complement_B_intersection_A_B_empty_intersection_A_B_equals_A_l2388_238838

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B (m : ℝ) : Set ℝ := {x | x - m < 0}

-- Theorem 1
theorem intersection_A_complement_B (m : ℝ) : 
  m = 3 → A ∩ (Set.univ \ B m) = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem 2
theorem intersection_A_B_empty (m : ℝ) : 
  A ∩ B m = ∅ ↔ m ≤ -2 := by sorry

-- Theorem 3
theorem intersection_A_B_equals_A (m : ℝ) : 
  A ∩ B m = A ↔ m ≥ 4 := by sorry

end intersection_A_complement_B_intersection_A_B_empty_intersection_A_B_equals_A_l2388_238838


namespace nine_workers_needed_workers_to_build_nine_cars_l2388_238814

/-- The number of workers needed to build a given number of cars in 9 days -/
def workers_needed (cars : ℕ) : ℕ :=
  cars

theorem nine_workers_needed : workers_needed 9 = 9 :=
by
  -- Proof goes here
  sorry

/-- Given condition: 7 workers can build 7 cars in 9 days -/
axiom seven_workers_seven_cars : workers_needed 7 = 7

-- The main theorem
theorem workers_to_build_nine_cars : ∃ w : ℕ, workers_needed 9 = w ∧ w = 9 :=
by
  -- Proof goes here
  sorry

end nine_workers_needed_workers_to_build_nine_cars_l2388_238814


namespace quadratic_inequality_solution_set_l2388_238872

open Set

/-- Solution set for the quadratic inequality ax^2 + (1-a)x - 1 > 0 -/
def SolutionSet (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x > 1}
  else if a > 0 then {x | x < -1/a ∨ x > 1}
  else if -1 < a then {x | 1 < x ∧ x < -1/a}
  else univ

theorem quadratic_inequality_solution_set :
  (∀ x, x ∈ SolutionSet 2 ↔ (x < -1/2 ∨ x > 1)) ∧
  (∀ a, a > -1 → ∀ x, x ∈ SolutionSet a ↔
    (a = 0 ∧ x > 1) ∨
    (a > 0 ∧ (x < -1/a ∨ x > 1)) ∨
    (-1 < a ∧ a < 0 ∧ 1 < x ∧ x < -1/a)) := by
  sorry

end quadratic_inequality_solution_set_l2388_238872


namespace factors_and_product_l2388_238858

-- Define a multiplication equation
def multiplication_equation (a b c : ℕ) : Prop := a * b = c

-- Define factors and product
def is_factor (a b c : ℕ) : Prop := multiplication_equation a b c
def is_product (a b c : ℕ) : Prop := multiplication_equation a b c

-- Theorem statement
theorem factors_and_product (a b c : ℕ) :
  multiplication_equation a b c → (is_factor a b c ∧ is_factor b a c ∧ is_product a b c) :=
by sorry

end factors_and_product_l2388_238858


namespace square_area_from_diagonal_l2388_238895

theorem square_area_from_diagonal (a b : ℝ) :
  let diagonal := Real.sqrt (a^2 + 4 * b^2)
  (diagonal^2 / 2) = (a^2 + 4 * b^2) / 2 := by
  sorry

end square_area_from_diagonal_l2388_238895


namespace f_is_quadratic_l2388_238841

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x² + 5x = 0 -/
def f (x : ℝ) : ℝ := x^2 + 5*x

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end f_is_quadratic_l2388_238841


namespace max_product_constrained_l2388_238870

theorem max_product_constrained (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 8 * b = 48) :
  a * b ≤ 24 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 8 * b₀ = 48 ∧ a₀ * b₀ = 24 := by
  sorry

end max_product_constrained_l2388_238870


namespace sum_of_specific_sequence_l2388_238877

def arithmetic_sequence_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem sum_of_specific_sequence :
  arithmetic_sequence_sum 102 492 10 = 11880 := by
  sorry

end sum_of_specific_sequence_l2388_238877


namespace dirichlet_approximation_l2388_238847

theorem dirichlet_approximation (N : ℕ) (hN : N > 0) :
  ∃ (a b : ℕ), 1 ≤ b ∧ b ≤ N ∧ |a - b * Real.sqrt 2| ≤ 1 / N :=
by sorry

end dirichlet_approximation_l2388_238847


namespace gumball_difference_l2388_238843

/-- The number of gumballs Hector purchased -/
def total_gumballs : ℕ := 45

/-- The number of gumballs Hector gave to Todd -/
def todd_gumballs : ℕ := 4

/-- The number of gumballs Hector gave to Alisha -/
def alisha_gumballs : ℕ := 2 * todd_gumballs

/-- The number of gumballs Hector had remaining -/
def remaining_gumballs : ℕ := 6

/-- The number of gumballs Hector gave to Bobby -/
def bobby_gumballs : ℕ := total_gumballs - todd_gumballs - alisha_gumballs - remaining_gumballs

theorem gumball_difference : 
  4 * alisha_gumballs - bobby_gumballs = 5 := by sorry

end gumball_difference_l2388_238843


namespace sqrt_calculation_l2388_238830

theorem sqrt_calculation : Real.sqrt 6 * Real.sqrt 3 + Real.sqrt 24 / Real.sqrt 6 - |(-3) * Real.sqrt 2| = 2 := by
  sorry

end sqrt_calculation_l2388_238830


namespace ferris_wheel_cost_calculation_l2388_238804

/-- The cost of the Ferris wheel ride -/
def ferris_wheel_cost : ℝ := 2.0

/-- The cost of the roller coaster ride -/
def roller_coaster_cost : ℝ := 7.0

/-- The discount for multiple rides -/
def multiple_ride_discount : ℝ := 1.0

/-- The value of the newspaper coupon -/
def coupon_value : ℝ := 1.0

/-- The total number of tickets needed for both rides -/
def total_tickets_needed : ℝ := 7.0

theorem ferris_wheel_cost_calculation :
  ferris_wheel_cost + roller_coaster_cost - multiple_ride_discount - coupon_value = total_tickets_needed :=
sorry

end ferris_wheel_cost_calculation_l2388_238804


namespace writers_birth_months_l2388_238818

/-- The total number of famous writers -/
def total_writers : ℕ := 200

/-- The number of writers born in October -/
def october_births : ℕ := 15

/-- The number of writers born in July -/
def july_births : ℕ := 14

/-- The percentage of writers born in October -/
def october_percentage : ℚ := (october_births : ℚ) / (total_writers : ℚ) * 100

/-- The percentage of writers born in July -/
def july_percentage : ℚ := (july_births : ℚ) / (total_writers : ℚ) * 100

theorem writers_birth_months :
  october_percentage = 15/2 ∧
  july_percentage = 7 ∧
  october_percentage > july_percentage :=
by sorry

end writers_birth_months_l2388_238818


namespace quadratic_function_properties_l2388_238825

/-- A quadratic function passing through two points with constrained x values -/
def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

theorem quadratic_function_properties :
  ∃ (a b : ℝ),
    (quadratic_function a b 0 = 6) ∧
    (quadratic_function a b 1 = 5) ∧
    (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 →
      (quadratic_function a b x = x^2 - 2*x + 6) ∧
      (quadratic_function a b x ≥ 5) ∧
      (quadratic_function a b x ≤ 14) ∧
      (quadratic_function a b 1 = 5) ∧
      (quadratic_function a b (-2) = 14)) :=
by sorry

end quadratic_function_properties_l2388_238825


namespace complex_argument_of_two_plus_two_i_sqrt_three_l2388_238894

/-- For the complex number z = 2 + 2i√3, when expressed in the form re^(iθ), θ = π/3 -/
theorem complex_argument_of_two_plus_two_i_sqrt_three :
  let z : ℂ := 2 + 2 * Complex.I * Real.sqrt 3
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (π / 3)) :=
by sorry

end complex_argument_of_two_plus_two_i_sqrt_three_l2388_238894


namespace fraction_square_product_l2388_238836

theorem fraction_square_product : (8 / 9 : ℚ)^2 * (1 / 3 : ℚ)^2 = 64 / 729 := by sorry

end fraction_square_product_l2388_238836


namespace not_arithmetic_sequence_sqrt_2_3_5_l2388_238888

theorem not_arithmetic_sequence_sqrt_2_3_5 : ¬∃ (a b c : ℝ), 
  (a = Real.sqrt 2) ∧ 
  (b = Real.sqrt 3) ∧ 
  (c = Real.sqrt 5) ∧ 
  (b - a = c - b) :=
by sorry

end not_arithmetic_sequence_sqrt_2_3_5_l2388_238888


namespace john_profit_calculation_l2388_238892

/-- Calculates John's profit from selling newspapers, magazines, and books --/
theorem john_profit_calculation :
  let newspaper_count : ℕ := 500
  let magazine_count : ℕ := 300
  let book_count : ℕ := 200
  let newspaper_price : ℚ := 2
  let magazine_price : ℚ := 4
  let book_price : ℚ := 10
  let newspaper_sold_ratio : ℚ := 0.80
  let magazine_sold_ratio : ℚ := 0.75
  let book_sold_ratio : ℚ := 0.60
  let newspaper_discount : ℚ := 0.75
  let magazine_discount : ℚ := 0.60
  let book_discount : ℚ := 0.45
  let tax_rate : ℚ := 0.08
  let shipping_fee : ℚ := 25
  let commission_rate : ℚ := 0.05

  let newspaper_cost := newspaper_price * (1 - newspaper_discount)
  let magazine_cost := magazine_price * (1 - magazine_discount)
  let book_cost := book_price * (1 - book_discount)

  let total_cost_before_tax := 
    newspaper_count * newspaper_cost +
    magazine_count * magazine_cost +
    book_count * book_cost

  let total_cost_after_tax_and_shipping :=
    total_cost_before_tax * (1 + tax_rate) + shipping_fee

  let total_revenue_before_commission :=
    newspaper_count * newspaper_sold_ratio * newspaper_price +
    magazine_count * magazine_sold_ratio * magazine_price +
    book_count * book_sold_ratio * book_price

  let total_revenue_after_commission :=
    total_revenue_before_commission * (1 - commission_rate)

  let profit := total_revenue_after_commission - total_cost_after_tax_and_shipping

  profit = 753.60 := by sorry

end john_profit_calculation_l2388_238892


namespace fraction_zero_implies_x_negative_one_l2388_238821

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (|x| - 1) / (x - 1) = 0 → x = -1 := by
  sorry

end fraction_zero_implies_x_negative_one_l2388_238821


namespace fraction_evaluation_l2388_238853

theorem fraction_evaluation : (5 * 6 - 3 * 4) / (6 + 3) = 2 := by
  sorry

end fraction_evaluation_l2388_238853


namespace relay_race_ratio_l2388_238881

theorem relay_race_ratio (total_members : Nat) (other_members : Nat) (other_distance : ℝ) (total_distance : ℝ) :
  total_members = 5 →
  other_members = 4 →
  other_distance = 3 →
  total_distance = 18 →
  (total_distance - other_members * other_distance) / other_distance = 2 := by
  sorry

end relay_race_ratio_l2388_238881


namespace cara_friends_photo_l2388_238880

theorem cara_friends_photo (n : ℕ) (k : ℕ) : n = 7 → k = 2 → Nat.choose n k = 21 := by
  sorry

end cara_friends_photo_l2388_238880


namespace remaining_potatoes_l2388_238848

/-- Given an initial number of potatoes and a number of eaten potatoes,
    prove that the remaining number of potatoes is equal to their difference. -/
theorem remaining_potatoes (initial : ℕ) (eaten : ℕ) :
  initial ≥ eaten → initial - eaten = initial - eaten :=
by sorry

end remaining_potatoes_l2388_238848


namespace mom_tshirt_count_l2388_238813

/-- The number of t-shirts in a package -/
def shirts_per_package : ℕ := 6

/-- The number of packages Mom buys -/
def packages_bought : ℕ := 71

/-- Theorem: Mom will have 426 white t-shirts -/
theorem mom_tshirt_count : shirts_per_package * packages_bought = 426 := by
  sorry

end mom_tshirt_count_l2388_238813


namespace min_value_of_sum_of_absolute_values_l2388_238820

theorem min_value_of_sum_of_absolute_values :
  ∃ (m : ℝ), (∀ x : ℝ, m ≤ |x + 2| + |x - 2| + |x - 1|) ∧ (∃ y : ℝ, m = |y + 2| + |y - 2| + |y - 1|) ∧ m = 4 :=
by sorry

end min_value_of_sum_of_absolute_values_l2388_238820


namespace kenneth_fabric_price_l2388_238805

/-- The price Kenneth paid for an oz of fabric -/
def kenneth_price : ℝ := 40

/-- The amount of fabric Kenneth bought in oz -/
def kenneth_amount : ℝ := 700

/-- The ratio of fabric Nicholas bought compared to Kenneth -/
def nicholas_ratio : ℝ := 6

/-- The additional amount Nicholas paid compared to Kenneth -/
def price_difference : ℝ := 140000

theorem kenneth_fabric_price :
  kenneth_price * kenneth_amount * nicholas_ratio =
  kenneth_price * kenneth_amount + price_difference :=
by sorry

end kenneth_fabric_price_l2388_238805


namespace subtracted_number_l2388_238822

theorem subtracted_number (x y : ℤ) (h1 : x = 30) (h2 : 8 * x - y = 102) : y = 138 := by
  sorry

end subtracted_number_l2388_238822


namespace periodic_sum_implies_constant_l2388_238807

/-- A function is periodic with period a if f(x + a) = f(x) for all x --/
def IsPeriodic (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a ≠ 0 ∧ ∀ x, f (x + a) = f x

theorem periodic_sum_implies_constant
  (f g : ℝ → ℝ) (a b : ℝ)
  (hfa : IsPeriodic f a)
  (hgb : IsPeriodic g b)
  (ha_rat : ℚ)
  (hb_irrat : Irrational b)
  (h_sum_periodic : ∃ c, IsPeriodic (f + g) c) :
  (∃ k, ∀ x, f x = k) ∨ (∃ k, ∀ x, g x = k) := by
  sorry

end periodic_sum_implies_constant_l2388_238807


namespace bret_nap_time_l2388_238866

/-- Calculates the remaining time for napping during a train ride -/
def remaining_nap_time (total_duration reading_time eating_time movie_time : ℕ) : ℕ :=
  total_duration - (reading_time + eating_time + movie_time)

/-- Theorem: Given Bret's 9-hour train ride and his activities, he has 3 hours left for napping -/
theorem bret_nap_time :
  remaining_nap_time 9 2 1 3 = 3 := by
  sorry

end bret_nap_time_l2388_238866


namespace tan_2016_in_terms_of_sin_36_l2388_238885

theorem tan_2016_in_terms_of_sin_36 (a : ℝ) (h : Real.sin (36 * π / 180) = a) :
  Real.tan (2016 * π / 180) = a / Real.sqrt (1 - a^2) := by
  sorry

end tan_2016_in_terms_of_sin_36_l2388_238885


namespace kevins_siblings_l2388_238860

-- Define the traits
inductive EyeColor
| Green
| Grey

inductive HairColor
| Red
| Brown

-- Define a child with their traits
structure Child where
  name : String
  eyeColor : EyeColor
  hairColor : HairColor

-- Define the function to check if two children share a trait
def shareTrait (c1 c2 : Child) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor

-- Define the children
def Oliver : Child := ⟨"Oliver", EyeColor.Green, HairColor.Red⟩
def Kevin : Child := ⟨"Kevin", EyeColor.Grey, HairColor.Brown⟩
def Lily : Child := ⟨"Lily", EyeColor.Grey, HairColor.Red⟩
def Emma : Child := ⟨"Emma", EyeColor.Green, HairColor.Brown⟩
def Noah : Child := ⟨"Noah", EyeColor.Green, HairColor.Red⟩
def Mia : Child := ⟨"Mia", EyeColor.Green, HairColor.Brown⟩

-- Define the theorem
theorem kevins_siblings :
  (shareTrait Kevin Emma ∧ shareTrait Kevin Mia ∧ shareTrait Emma Mia) ∧
  (¬ (shareTrait Kevin Oliver ∧ shareTrait Kevin Noah ∧ shareTrait Oliver Noah)) ∧
  (¬ (shareTrait Kevin Lily ∧ shareTrait Kevin Noah ∧ shareTrait Lily Noah)) ∧
  (¬ (shareTrait Kevin Oliver ∧ shareTrait Kevin Lily ∧ shareTrait Oliver Lily)) :=
sorry

end kevins_siblings_l2388_238860


namespace binomial_expansion_constant_term_l2388_238874

theorem binomial_expansion_constant_term (n : ℕ+) :
  (Nat.choose n 2 = Nat.choose n 4) →
  (Nat.choose n (n / 2) = 20) :=
by sorry

end binomial_expansion_constant_term_l2388_238874


namespace euler_family_mean_age_l2388_238828

def euler_family_ages : List ℕ := [8, 8, 12, 10, 10, 16]

theorem euler_family_mean_age :
  (euler_family_ages.sum : ℚ) / euler_family_ages.length = 32 / 3 := by
  sorry

end euler_family_mean_age_l2388_238828


namespace expression_simplification_l2388_238808

/-- Given that |2+y|+(x-1)^2=0, prove that 5x^2*y-[3x*y^2-2(3x*y^2-7/2*x^2*y)] = 16 -/
theorem expression_simplification (x y : ℝ) 
  (h : |2 + y| + (x - 1)^2 = 0) : 
  5*x^2*y - (3*x*y^2 - 2*(3*x*y^2 - 7/2*x^2*y)) = 16 := by
  sorry

end expression_simplification_l2388_238808


namespace two_digit_numbers_product_sum_l2388_238846

theorem two_digit_numbers_product_sum (x y : ℕ) : 
  (10 ≤ x ∧ x < 100) ∧ 
  (10 ≤ y ∧ y < 100) ∧ 
  (2000 ≤ x * y ∧ x * y < 3000) ∧ 
  (100 ≤ x + y ∧ x + y < 1000) ∧ 
  (x * y = 2000 + (x + y)) →
  ((x = 24 ∧ y = 88) ∨ (x = 88 ∧ y = 24) ∨ (x = 30 ∧ y = 70) ∨ (x = 70 ∧ y = 30)) :=
by sorry

end two_digit_numbers_product_sum_l2388_238846


namespace unique_perfect_square_P_l2388_238898

def P (x : ℤ) : ℤ := x^4 + 6*x^3 + 11*x^2 + 3*x + 31

theorem unique_perfect_square_P :
  ∃! x : ℤ, ∃ y : ℤ, P x = y^2 :=
sorry

end unique_perfect_square_P_l2388_238898


namespace inequality_proof_l2388_238873

theorem inequality_proof (x y : ℝ) (h1 : x > y) (h2 : x * y = 1) :
  (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2 := by
  sorry

end inequality_proof_l2388_238873


namespace ellipse_trajectory_l2388_238899

/-- The trajectory of point Q given an ellipse and its properties -/
theorem ellipse_trajectory (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∀ (x y : ℝ), 
  ((-x/2)^2 / a^2 + (-y/2)^2 / b^2 = 1) →
  (x^2 / (4*a^2) + y^2 / (4*b^2) = 1) := by
  sorry


end ellipse_trajectory_l2388_238899


namespace ellipse_point_distance_l2388_238876

/-- The ellipse with equation x²/9 + y²/6 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 9) + (p.2^2 / 6) = 1}

/-- The foci of the ellipse -/
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

/-- The origin -/
def O : ℝ × ℝ := (0, 0)

/-- A point on the ellipse -/
def P : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The angle between three points -/
def angle (p q r : ℝ × ℝ) : ℝ := sorry

theorem ellipse_point_distance :
  P ∈ Ellipse →
  angle F₁ P F₂ = Real.arccos (3/5) →
  distance P O = Real.sqrt 30 / 2 := by
  sorry

end ellipse_point_distance_l2388_238876


namespace round_robin_cyclic_triples_l2388_238801

/-- Represents a round-robin tournament. -/
structure Tournament where
  teams : ℕ
  games_won : ℕ
  games_lost : ℕ

/-- Represents a cyclic triple in the tournament. -/
structure CyclicTriple where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The number of cyclic triples in the tournament. -/
def count_cyclic_triples (t : Tournament) : ℕ :=
  sorry

theorem round_robin_cyclic_triples :
  ∀ t : Tournament,
    t.teams = t.games_won + t.games_lost + 1 →
    t.games_won = 12 →
    t.games_lost = 8 →
    count_cyclic_triples t = 144 :=
  sorry

end round_robin_cyclic_triples_l2388_238801


namespace orthocenter_property_l2388_238811

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the orthocenter of a triangle
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the cosine of the sum of two angles
def cos_sum_angles (α β : ℝ) : ℝ := sorry

-- Define the measure of an angle
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem orthocenter_property (t : Triangle) :
  let O := orthocenter t
  (angle_measure t.A t.B t.C > π / 2) →  -- Angle A is obtuse
  (dist O t.A = dist t.B t.C) →  -- AO = BC
  (cos_sum_angles (angle_measure O t.B t.C) (angle_measure O t.C t.B) = -Real.sqrt 2 / 2) :=
by sorry

end orthocenter_property_l2388_238811


namespace unique_arith_seq_pair_a_eq_one_third_l2388_238857

/-- Two arithmetic sequences satisfying given conditions -/
structure ArithSeqPair where
  a : ℝ
  q : ℝ
  h_a_pos : a > 0
  h_b1_a1 : (a + 1) - a = 1
  h_b2_a2 : (a + q + 2) - (a + q) = 2
  h_b3_a3 : (a + 2*q + 3) - (a + 2*q) = 3
  h_unique : ∃! q, (a * q^2 - 4 * a * q + 3 * a - 1 = 0) ∧ q ≠ 0

/-- If two arithmetic sequences satisfy the given conditions and one is unique, then a = 1/3 -/
theorem unique_arith_seq_pair_a_eq_one_third (p : ArithSeqPair) : p.a = 1/3 := by
  sorry

end unique_arith_seq_pair_a_eq_one_third_l2388_238857


namespace triangle_angle_sum_and_side_inequality_l2388_238864

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the properties of the triangle
def has_sine_cosine_property (t : Triangle) : Prop :=
  Real.sin t.A + Real.cos t.B = Real.sqrt 2 ∧
  Real.cos t.A + Real.sin t.B = Real.sqrt 2

-- Define the angle bisector property
def has_angle_bisector (t : Triangle) (D : ℝ) : Prop :=
  -- We don't define the specifics of the bisector, just that it exists
  true

-- State the theorem
theorem triangle_angle_sum_and_side_inequality
  (t : Triangle) (D : ℝ) 
  (h1 : has_sine_cosine_property t)
  (h2 : has_angle_bisector t D) :
  t.A + t.B = 90 ∧ t.A > D :=
sorry

end triangle_angle_sum_and_side_inequality_l2388_238864


namespace arithmetic_calculation_l2388_238800

theorem arithmetic_calculation : 3^2 * 4 + 5 * (6 + 3) - 15 / 3 = 76 := by
  sorry

end arithmetic_calculation_l2388_238800


namespace teacher_selection_and_assignment_l2388_238837

-- Define the number of male and female teachers
def num_male_teachers : ℕ := 5
def num_female_teachers : ℕ := 4

-- Define the number of male and female teachers to be selected
def selected_male_teachers : ℕ := 3
def selected_female_teachers : ℕ := 2

-- Define the total number of teachers to be selected
def total_selected_teachers : ℕ := selected_male_teachers + selected_female_teachers

-- Define the number of villages
def num_villages : ℕ := 5

-- Theorem statement
theorem teacher_selection_and_assignment :
  (Nat.choose num_male_teachers selected_male_teachers) *
  (Nat.choose num_female_teachers selected_female_teachers) *
  (Nat.factorial total_selected_teachers) = 7200 :=
sorry

end teacher_selection_and_assignment_l2388_238837


namespace smallest_b_for_inequality_l2388_238869

theorem smallest_b_for_inequality : ∃ b : ℕ, (∀ k : ℕ, 27^k > 3^24 → k ≥ b) ∧ 27^b > 3^24 :=
  sorry

end smallest_b_for_inequality_l2388_238869


namespace rabbit_carrot_problem_l2388_238887

theorem rabbit_carrot_problem (initial_carrots : ℕ) : 
  (((initial_carrots * 3 - 30) * 3 - 30) * 3 - 30) * 3 - 30 = 0 → 
  initial_carrots = 15 := by
sorry

end rabbit_carrot_problem_l2388_238887


namespace largest_angle_in_triangle_l2388_238802

theorem largest_angle_in_triangle (x : ℝ) : 
  x + 40 + 60 = 180 → 
  max x (max 40 60) = 80 :=
by sorry

end largest_angle_in_triangle_l2388_238802


namespace stock_price_after_three_years_l2388_238834

theorem stock_price_after_three_years (initial_price : ℝ) :
  initial_price = 120 →
  let price_after_year1 := initial_price * 1.5
  let price_after_year2 := price_after_year1 * 0.7
  let price_after_year3 := price_after_year2 * 1.2
  price_after_year3 = 151.2 := by
sorry

end stock_price_after_three_years_l2388_238834


namespace simplify_expression_l2388_238889

theorem simplify_expression : 80 - (5 - (6 + 2 * (7 - 8 - 5))) = 69 := by
  sorry

end simplify_expression_l2388_238889


namespace parallel_line_through_point_main_theorem_l2388_238886

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point 
  (given_line : Line) 
  (point : Point) 
  (result_line : Line) : Prop :=
  given_line.a = 2 ∧ 
  given_line.b = -1 ∧ 
  given_line.c = 1 ∧ 
  point.x = -1 ∧ 
  point.y = 0 ∧ 
  result_line.a = 2 ∧ 
  result_line.b = -1 ∧ 
  result_line.c = 2 ∧ 
  point.liesOn result_line ∧ 
  result_line.isParallel given_line

/-- The main theorem stating that the resulting line equation is correct -/
theorem main_theorem : ∃ (given_line result_line : Line) (point : Point), 
  parallel_line_through_point given_line point result_line := by
  sorry

end parallel_line_through_point_main_theorem_l2388_238886


namespace truck_rental_problem_l2388_238890

/-- The total number of trucks on Monday morning -/
def total_trucks : ℕ := 30

/-- The number of trucks rented out during the week -/
def rented_trucks : ℕ := 20

/-- The number of trucks returned by Saturday morning -/
def returned_trucks : ℕ := rented_trucks / 2

/-- The number of trucks on the lot Saturday morning -/
def saturday_trucks : ℕ := returned_trucks

theorem truck_rental_problem :
  (returned_trucks = rented_trucks / 2) →
  (saturday_trucks ≥ 10) →
  (rented_trucks = 20) →
  (total_trucks = rented_trucks + (rented_trucks - returned_trucks)) :=
by sorry

end truck_rental_problem_l2388_238890
