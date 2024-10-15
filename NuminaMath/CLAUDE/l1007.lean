import Mathlib

namespace NUMINAMATH_CALUDE_cos_seven_expansion_sum_of_squares_l1007_100757

theorem cos_seven_expansion_sum_of_squares : 
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ), 
    (∀ θ : ℝ, Real.cos θ ^ 7 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + 
      b₃ * Real.cos (3 * θ) + b₄ * Real.cos (4 * θ) + b₅ * Real.cos (5 * θ) + 
      b₆ * Real.cos (6 * θ) + b₇ * Real.cos (7 * θ)) →
    b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 + b₆^2 + b₇^2 = 429 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_expansion_sum_of_squares_l1007_100757


namespace NUMINAMATH_CALUDE_mixing_hcl_solutions_l1007_100717

/-- Represents a hydrochloric acid solution --/
structure HClSolution where
  mass : ℝ
  concentration : ℝ

/-- Calculates the mass of pure HCl in a solution --/
def HClMass (solution : HClSolution) : ℝ :=
  solution.mass * solution.concentration

theorem mixing_hcl_solutions
  (solution1 : HClSolution)
  (solution2 : HClSolution)
  (mixed : HClSolution)
  (h1 : solution1.concentration = 0.3)
  (h2 : solution2.concentration = 0.1)
  (h3 : mixed.concentration = 0.15)
  (h4 : mixed.mass = 600)
  (h5 : solution1.mass + solution2.mass = mixed.mass)
  (h6 : HClMass solution1 + HClMass solution2 = HClMass mixed) :
  solution1.mass = 150 ∧ solution2.mass = 450 := by
  sorry

end NUMINAMATH_CALUDE_mixing_hcl_solutions_l1007_100717


namespace NUMINAMATH_CALUDE_matrix_power_equality_l1007_100775

def A (a : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![1, 3, a; 0, 1, 5; 0, 0, 1]

theorem matrix_power_equality (a : ℝ) (n : ℕ) :
  (A a) ^ n = !![1, 27, 3000; 0, 1, 45; 0, 0, 1] →
  a + n = 278 := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_equality_l1007_100775


namespace NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l1007_100724

theorem tan_eleven_pi_fourths : Real.tan (11 * π / 4) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l1007_100724


namespace NUMINAMATH_CALUDE_min_value_A_l1007_100740

theorem min_value_A (x y z w : ℝ) :
  ∃ (A : ℝ), A = (1 + Real.sqrt 2) / 2 ∧
  (∀ (B : ℝ), (x*y + 2*y*z + z*w ≤ B*(x^2 + y^2 + z^2 + w^2)) → A ≤ B) ∧
  (x*y + 2*y*z + z*w ≤ A*(x^2 + y^2 + z^2 + w^2)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_A_l1007_100740


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l1007_100771

/-- The solution set of x^2 - 5x + 4 < 0 is a subset of x^2 - (a+5)x + 5a < 0 -/
def subset_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 5*x + 4 < 0 → x^2 - (a+5)*x + 5*a < 0

/-- The range of values for a -/
def a_range (a : ℝ) : Prop := a ≤ 1

/-- Theorem stating the relationship between the subset condition and the range of a -/
theorem subset_implies_a_range :
  ∀ a : ℝ, subset_condition a → a_range a :=
sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l1007_100771


namespace NUMINAMATH_CALUDE_exists_function_with_property_l1007_100750

def apply_n_times (f : ℝ → ℝ) (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ (apply_n_times f n)

theorem exists_function_with_property : 
  ∃ (f : ℝ → ℝ), 
    (∀ x : ℝ, x ≥ 0 → f x ≥ 0) ∧ 
    (∀ x : ℝ, x ≥ 0 → apply_n_times f 45 x = 1 + x + 2 * Real.sqrt x) :=
  sorry

end NUMINAMATH_CALUDE_exists_function_with_property_l1007_100750


namespace NUMINAMATH_CALUDE_sqrt_3_minus_1_power_l1007_100707

theorem sqrt_3_minus_1_power (N : ℕ) : 
  (Real.sqrt 3 - 1) ^ N = 4817152 - 2781184 * Real.sqrt 3 → N = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_minus_1_power_l1007_100707


namespace NUMINAMATH_CALUDE_cookie_difference_l1007_100783

/-- Given that Alyssa has 129 cookies and Aiyanna has 140 cookies, 
    prove that Aiyanna has 11 more cookies than Alyssa. -/
theorem cookie_difference (alyssa_cookies : ℕ) (aiyanna_cookies : ℕ) 
    (h1 : alyssa_cookies = 129) (h2 : aiyanna_cookies = 140) : 
    aiyanna_cookies - alyssa_cookies = 11 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l1007_100783


namespace NUMINAMATH_CALUDE_quadratic_sufficient_not_necessary_l1007_100795

theorem quadratic_sufficient_not_necessary :
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 1) ∧
  ¬(∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sufficient_not_necessary_l1007_100795


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l1007_100768

/-- Represents a systematic sampling process -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  removed_size : ℕ
  h_pop_size : population_size = 1002
  h_sample_size : sample_size = 50
  h_removed_size : removed_size = 2

/-- The probability of an individual being selected in the systematic sampling process -/
def selection_probability (s : SystematicSampling) : ℚ :=
  s.sample_size / s.population_size

theorem systematic_sampling_probability (s : SystematicSampling) :
  selection_probability s = 50 / 1002 := by
  sorry

#eval (50 : ℚ) / 1002

end NUMINAMATH_CALUDE_systematic_sampling_probability_l1007_100768


namespace NUMINAMATH_CALUDE_function_comparison_and_maximum_l1007_100719

def f (x : ℝ) := abs (x - 1)
def g (x : ℝ) := -x^2 + 6*x - 5

theorem function_comparison_and_maximum :
  (∀ x : ℝ, g x ≥ f x ↔ x ∈ Set.Icc 1 4) ∧
  (∃ M : ℝ, M = 9/4 ∧ ∀ x : ℝ, g x - f x ≤ M) :=
sorry

end NUMINAMATH_CALUDE_function_comparison_and_maximum_l1007_100719


namespace NUMINAMATH_CALUDE_prob_genuine_given_equal_weights_l1007_100715

/-- Represents a bag of coins -/
structure CoinBag where
  total : ℕ
  genuine : ℕ
  counterfeit : ℕ

/-- Represents the result of selecting coins -/
inductive Selection
  | AllGenuine
  | Mixed
  | AllCounterfeit

/-- Calculates the probability of selecting all genuine coins -/
def prob_all_genuine (bag : CoinBag) : ℚ :=
  (bag.genuine.choose 2 : ℚ) * ((bag.genuine - 2).choose 2 : ℚ) /
  ((bag.total.choose 2 : ℚ) * ((bag.total - 2).choose 2 : ℚ))

/-- Calculates the probability of equal weights -/
def prob_equal_weights (bag : CoinBag) : ℚ :=
  sorry  -- Actual calculation would go here

/-- The main theorem to prove -/
theorem prob_genuine_given_equal_weights (bag : CoinBag) 
  (h1 : bag.total = 12)
  (h2 : bag.genuine = 9)
  (h3 : bag.counterfeit = 3) :
  prob_all_genuine bag / prob_equal_weights bag = 42 / 165 := by
  sorry

end NUMINAMATH_CALUDE_prob_genuine_given_equal_weights_l1007_100715


namespace NUMINAMATH_CALUDE_transform_equation_l1007_100733

theorem transform_equation (m n x y : ℚ) :
  m + x = n + y → m = n → x = y := by
  sorry

end NUMINAMATH_CALUDE_transform_equation_l1007_100733


namespace NUMINAMATH_CALUDE_evaluate_expression_l1007_100732

theorem evaluate_expression (c x y z : ℚ) :
  c = -2 →
  x = 2/5 →
  y = 3/5 →
  z = -3 →
  c * x^3 * y^4 * z^2 = -11664/78125 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1007_100732


namespace NUMINAMATH_CALUDE_coin_drop_probability_l1007_100747

theorem coin_drop_probability : 
  let square_side : ℝ := 10
  let black_square_side : ℝ := 1
  let coin_diameter : ℝ := 2
  let coin_radius : ℝ := coin_diameter / 2
  let drop_area_side : ℝ := square_side - coin_diameter
  let drop_area : ℝ := drop_area_side ^ 2
  let extended_black_square_side : ℝ := black_square_side + coin_diameter
  let extended_black_area : ℝ := 4 * (extended_black_square_side ^ 2)
  extended_black_area / drop_area = 9 / 16 := by sorry

end NUMINAMATH_CALUDE_coin_drop_probability_l1007_100747


namespace NUMINAMATH_CALUDE_find_y_l1007_100743

theorem find_y (x y : ℝ) (h1 : x^(2*y) = 16) (h2 : x = 2) : y = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l1007_100743


namespace NUMINAMATH_CALUDE_roxy_initial_flowering_plants_l1007_100767

/-- The initial number of flowering plants in Roxy's garden -/
def initial_flowering_plants : ℕ := 7

/-- The initial number of fruiting plants in Roxy's garden -/
def initial_fruiting_plants : ℕ := 2 * initial_flowering_plants

/-- The number of flowering plants bought on Saturday -/
def flowering_plants_bought : ℕ := 3

/-- The number of fruiting plants bought on Saturday -/
def fruiting_plants_bought : ℕ := 2

/-- The number of flowering plants given away on Sunday -/
def flowering_plants_given : ℕ := 1

/-- The number of fruiting plants given away on Sunday -/
def fruiting_plants_given : ℕ := 4

/-- The total number of plants remaining after all transactions -/
def total_plants_remaining : ℕ := 21

theorem roxy_initial_flowering_plants :
  (initial_flowering_plants + flowering_plants_bought - flowering_plants_given) +
  (initial_fruiting_plants + fruiting_plants_bought - fruiting_plants_given) =
  total_plants_remaining :=
by sorry

end NUMINAMATH_CALUDE_roxy_initial_flowering_plants_l1007_100767


namespace NUMINAMATH_CALUDE_x_equation_solution_l1007_100714

theorem x_equation_solution (x : ℝ) (h : x + 1/x = Real.sqrt 5) :
  x^12 - 7*x^8 + x^4 = 343 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_solution_l1007_100714


namespace NUMINAMATH_CALUDE_product_equality_l1007_100708

theorem product_equality : 100 * 19.98 * 1.998 * 999 = 3988008 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1007_100708


namespace NUMINAMATH_CALUDE_stratified_sampling_middle_batch_l1007_100797

/-- Represents the number of units drawn from each batch in a stratified sampling -/
structure BatchSampling where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given BatchSampling forms an arithmetic sequence -/
def is_arithmetic_sequence (s : BatchSampling) : Prop :=
  s.c - s.b = s.b - s.a

/-- The theorem stating that in a stratified sampling of 60 units from three batches
    forming an arithmetic sequence, the number of units drawn from the middle batch is 20 -/
theorem stratified_sampling_middle_batch :
  ∀ s : BatchSampling,
    is_arithmetic_sequence s →
    s.a + s.b + s.c = 60 →
    s.b = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_middle_batch_l1007_100797


namespace NUMINAMATH_CALUDE_line_circle_intersection_l1007_100710

/-- Given a line and a circle that intersect at two points with a specific distance between them, 
    prove that the slope of the line has a specific value. -/
theorem line_circle_intersection (m : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (∀ x y : ℝ, m * x + y + 3 * m - Real.sqrt 3 = 0 → (x, y) ∈ {(x, y) | m * x + y + 3 * m - Real.sqrt 3 = 0}) ∧ 
    (∀ x y : ℝ, x^2 + y^2 = 12 → (x, y) ∈ {(x, y) | x^2 + y^2 = 12}) ∧
    A ∈ {(x, y) | m * x + y + 3 * m - Real.sqrt 3 = 0} ∧
    A ∈ {(x, y) | x^2 + y^2 = 12} ∧
    B ∈ {(x, y) | m * x + y + 3 * m - Real.sqrt 3 = 0} ∧
    B ∈ {(x, y) | x^2 + y^2 = 12} ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12) →
  m = -Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l1007_100710


namespace NUMINAMATH_CALUDE_solution_difference_l1007_100741

theorem solution_difference (a b : ℝ) (ha : a ≠ 0) 
  (h : a^2 - b*a - 4*a = 0) : a - b = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l1007_100741


namespace NUMINAMATH_CALUDE_quadratic_positivity_condition_l1007_100759

theorem quadratic_positivity_condition (m : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + m > 0) → m > 0 ∧ 
  ∃ m₀ : ℝ, m₀ > 0 ∧ ¬(∀ x : ℝ, x^2 + 2*x + m₀ > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_positivity_condition_l1007_100759


namespace NUMINAMATH_CALUDE_ring_sector_area_proof_l1007_100728

/-- The area of a ring-shaped sector formed by two concentric circles with radii 13 and 7, and a common central angle θ -/
def ring_sector_area (θ : Real) : Real :=
  60 * θ

/-- Theorem: The area of a ring-shaped sector formed by two concentric circles
    with radii 13 and 7, and a common central angle θ, is equal to 60θ -/
theorem ring_sector_area_proof (θ : Real) :
  ring_sector_area θ = 60 * θ := by
  sorry

#check ring_sector_area_proof

end NUMINAMATH_CALUDE_ring_sector_area_proof_l1007_100728


namespace NUMINAMATH_CALUDE_sum_of_53_odd_numbers_l1007_100764

theorem sum_of_53_odd_numbers : 
  (Finset.range 53).sum (fun n => 2 * n + 1) = 2809 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_53_odd_numbers_l1007_100764


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_l1007_100792

theorem negation_of_existence_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_l1007_100792


namespace NUMINAMATH_CALUDE_kevin_stuffed_animals_l1007_100706

/-- Represents the number of prizes Kevin collected. -/
def total_prizes : ℕ := 50

/-- Represents the number of frisbees Kevin collected. -/
def frisbees : ℕ := 18

/-- Represents the number of yo-yos Kevin collected. -/
def yo_yos : ℕ := 18

/-- Represents the number of stuffed animals Kevin collected. -/
def stuffed_animals : ℕ := total_prizes - frisbees - yo_yos

theorem kevin_stuffed_animals : stuffed_animals = 14 := by
  sorry

end NUMINAMATH_CALUDE_kevin_stuffed_animals_l1007_100706


namespace NUMINAMATH_CALUDE_circle_land_theorem_l1007_100745

/-- Represents a digit with its associated number of circles in Circle Land notation -/
structure CircleLandDigit where
  digit : Nat
  circles : Nat

/-- Calculates the value of a CircleLandDigit in the Circle Land number system -/
def circleValue (d : CircleLandDigit) : Nat :=
  d.digit * (10 ^ d.circles)

/-- Represents a number in Circle Land notation as a list of CircleLandDigits -/
def CircleLandNumber := List CircleLandDigit

/-- Calculates the value of a CircleLandNumber -/
def circleLandValue (n : CircleLandNumber) : Nat :=
  n.foldl (fun acc d => acc + circleValue d) 0

/-- The Circle Land representation of the number in the problem -/
def problemNumber : CircleLandNumber :=
  [⟨3, 4⟩, ⟨1, 2⟩, ⟨5, 0⟩]

theorem circle_land_theorem :
  circleLandValue problemNumber = 30105 := by sorry

end NUMINAMATH_CALUDE_circle_land_theorem_l1007_100745


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1007_100731

theorem fraction_to_decimal : (45 : ℚ) / 64 = 0.703125 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1007_100731


namespace NUMINAMATH_CALUDE_chord_length_line_ellipse_intersection_l1007_100753

/-- The length of the chord formed by the intersection of a line and an ellipse -/
theorem chord_length_line_ellipse_intersection :
  let line : ℝ → ℝ × ℝ := λ t ↦ (1 + t, -2 + t)
  let ellipse : ℝ × ℝ → Prop := λ p ↦ p.1^2 + 2*p.2^2 = 8
  ∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (∃ t₁, line t₁ = A) ∧ 
    (∃ t₂, line t₂ = B) ∧
    ellipse A ∧ 
    ellipse B ∧
    dist A B = 4 * Real.sqrt 3 / 3 :=
by sorry


end NUMINAMATH_CALUDE_chord_length_line_ellipse_intersection_l1007_100753


namespace NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_l1007_100748

theorem coefficient_x3y5_in_expansion (x y : ℝ) :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * x^k * y^(8-k)) =
  56 * x^3 * y^5 + (Finset.range 9).sum (fun k => if k ≠ 3 then Nat.choose 8 k * x^k * y^(8-k) else 0) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x3y5_in_expansion_l1007_100748


namespace NUMINAMATH_CALUDE_range_of_f_l1007_100746

-- Define the function f
def f (x : ℝ) : ℝ := x + |x - 2|

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1007_100746


namespace NUMINAMATH_CALUDE_cupcakes_left_after_distribution_l1007_100772

/-- Theorem: Cupcakes Left After Distribution

Given:
- Dani brings two and half dozen cupcakes
- There are 27 students (including Dani)
- There is 1 teacher
- There is 1 teacher's aid
- 3 students called in sick

Prove that the number of cupcakes left after Dani gives one to everyone in the class is 4.
-/
theorem cupcakes_left_after_distribution 
  (cupcakes_per_dozen : ℕ)
  (total_students : ℕ)
  (teacher_count : ℕ)
  (teacher_aid_count : ℕ)
  (sick_students : ℕ)
  (h1 : cupcakes_per_dozen = 12)
  (h2 : total_students = 27)
  (h3 : teacher_count = 1)
  (h4 : teacher_aid_count = 1)
  (h5 : sick_students = 3) :
  2 * cupcakes_per_dozen + cupcakes_per_dozen / 2 - 
  (total_students - sick_students + teacher_count + teacher_aid_count) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_left_after_distribution_l1007_100772


namespace NUMINAMATH_CALUDE_jonah_profit_l1007_100703

/-- Calculates the profit from selling pineapple rings given the following conditions:
  * Number of pineapples bought
  * Cost per pineapple
  * Number of rings per pineapple
  * Number of rings sold as a set
  * Price per set of rings
-/
def calculate_profit (num_pineapples : ℕ) (cost_per_pineapple : ℕ) 
                     (rings_per_pineapple : ℕ) (rings_per_set : ℕ) 
                     (price_per_set : ℕ) : ℕ :=
  let total_cost := num_pineapples * cost_per_pineapple
  let total_rings := num_pineapples * rings_per_pineapple
  let num_sets := total_rings / rings_per_set
  let total_revenue := num_sets * price_per_set
  total_revenue - total_cost

/-- Proves that Jonah's profit is $342 given the specified conditions -/
theorem jonah_profit : 
  calculate_profit 6 3 12 4 5 = 342 := by
  sorry

end NUMINAMATH_CALUDE_jonah_profit_l1007_100703


namespace NUMINAMATH_CALUDE_new_average_age_with_teacher_l1007_100779

theorem new_average_age_with_teacher 
  (num_students : ℕ) 
  (student_avg_age : ℝ) 
  (teacher_age : ℕ) 
  (h1 : num_students = 20) 
  (h2 : student_avg_age = 15) 
  (h3 : teacher_age = 36) : 
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_with_teacher_l1007_100779


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1007_100705

theorem sphere_surface_area (r : ℝ) (h : r = 3) : 4 * Real.pi * r^2 = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1007_100705


namespace NUMINAMATH_CALUDE_total_difference_of_sequences_l1007_100774

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem total_difference_of_sequences : 
  let n : ℕ := 72
  let d : ℕ := 3
  let a₁ : ℕ := 2001
  let b₁ : ℕ := 501
  arithmetic_sequence_sum a₁ d n - arithmetic_sequence_sum b₁ d n = 108000 := by
sorry

end NUMINAMATH_CALUDE_total_difference_of_sequences_l1007_100774


namespace NUMINAMATH_CALUDE_base_conversion_problem_l1007_100761

theorem base_conversion_problem :
  ∀ (a b : ℕ),
    a < 10 →
    b < 10 →
    235 = 1 * 7^2 + a * 7^1 + b * 7^0 →
    (a + b : ℚ) / 7 = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l1007_100761


namespace NUMINAMATH_CALUDE_sequence_third_term_l1007_100793

theorem sequence_third_term (a : ℕ → ℕ) (h : ∀ n, a n = n^2 + n) : a 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sequence_third_term_l1007_100793


namespace NUMINAMATH_CALUDE_ellipse_circle_fixed_point_l1007_100726

/-- Given an ellipse with equation (x²/a²) + (y²/b²) = 1, where a > b > 0,
    and a point P(x₀, y₀) on the ellipse different from A₁(-a, 0) and A(a, 0),
    the circle with diameter MM₁ (where M and M₁ are intersections of PA and PA₁
    with the directrix x = a²/c) passes through a fixed point outside the ellipse. -/
theorem ellipse_circle_fixed_point
  (a b c : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_c_pos : c > 0) (h_a_gt_b : a > b)
  (x₀ y₀ : ℝ) (h_on_ellipse : x₀^2 / a^2 + y₀^2 / b^2 = 1)
  (h_not_A : x₀ ≠ a ∨ y₀ ≠ 0) (h_not_A₁ : x₀ ≠ -a ∨ y₀ ≠ 0) :
  ∃ (x y : ℝ), x = (a^2 + b^2) / c ∧ y = 0 ∧
  (x - a^2 / c)^2 + (y + b^2 * (x₀ - c) / (c * y₀))^2 = (b^2 * (c * x₀ - a^2) / (a * c * y₀))^2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_fixed_point_l1007_100726


namespace NUMINAMATH_CALUDE_quadratic_equation_rational_roots_l1007_100729

theorem quadratic_equation_rational_roots (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (∃ x y : ℚ, x^2 + p^2 * x + q^3 = 0 ∧ y^2 + p^2 * y + q^3 = 0 ∧ x ≠ y) ↔ 
  (p = 3 ∧ q = 2 ∧ 
   ∃ x y : ℚ, x = -1 ∧ y = -8 ∧ 
   x^2 + p^2 * x + q^3 = 0 ∧ y^2 + p^2 * y + q^3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_rational_roots_l1007_100729


namespace NUMINAMATH_CALUDE_prob_one_head_in_three_flips_l1007_100790

/-- The probability of getting exactly one head in three flips of a fair coin -/
theorem prob_one_head_in_three_flips :
  let n : ℕ := 3  -- number of flips
  let k : ℕ := 1  -- number of desired heads
  let p : ℚ := 1/2  -- probability of getting heads on a single flip
  Nat.choose n k * p^k * (1-p)^(n-k) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_head_in_three_flips_l1007_100790


namespace NUMINAMATH_CALUDE_intersection_in_interval_l1007_100723

theorem intersection_in_interval :
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < 1 ∧ x₀^3 = (1/2)^x₀ := by sorry

end NUMINAMATH_CALUDE_intersection_in_interval_l1007_100723


namespace NUMINAMATH_CALUDE_atMostTwoInPlaceFive_l1007_100785

/-- The number of ways to arrange n people in n seats. -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of seating arrangements where at most two people
    are in their numbered seats, given n people and n seats. -/
def atMostTwoInPlace (n : ℕ) : ℕ :=
  totalArrangements n - choose n 3 * totalArrangements (n - 3) - 1

theorem atMostTwoInPlaceFive :
  atMostTwoInPlace 5 = 109 := by sorry

end NUMINAMATH_CALUDE_atMostTwoInPlaceFive_l1007_100785


namespace NUMINAMATH_CALUDE_hemisphere_diameter_l1007_100734

-- Define the cube
def cube_side_length : ℝ := 2

-- Define the hemisphere properties
structure Hemisphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the cube with hemispheres
structure CubeWithHemispheres where
  side_length : ℝ
  hemispheres : List Hemisphere
  hemispheres_touch : Bool

-- Theorem statement
theorem hemisphere_diameter (cube : CubeWithHemispheres) 
  (h1 : cube.side_length = cube_side_length)
  (h2 : cube.hemispheres.length = 6)
  (h3 : cube.hemispheres_touch = true) :
  ∀ h ∈ cube.hemispheres, 2 * h.radius = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_hemisphere_diameter_l1007_100734


namespace NUMINAMATH_CALUDE_triangle_formation_l1007_100716

/-- Triangle inequality theorem: the sum of any two sides must be greater than the third side --/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a set of three line segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 4 3 6 ∧
  ¬can_form_triangle 1 2 3 ∧
  ¬can_form_triangle 7 8 16 ∧
  ¬can_form_triangle 9 10 20 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l1007_100716


namespace NUMINAMATH_CALUDE_complex_multiplication_l1007_100798

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : (1 - i)^2 * i = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1007_100798


namespace NUMINAMATH_CALUDE_perpendicular_equivalence_l1007_100780

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_equivalence
  (α β : Plane) (m n : Line)
  (h_distinct : α ≠ β)
  (h_non_coincident : m ≠ n)
  (h_m_perp_α : perp m α)
  (h_m_perp_β : perp m β) :
  perp n α ↔ perp n β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_equivalence_l1007_100780


namespace NUMINAMATH_CALUDE_meaningful_range_l1007_100773

def is_meaningful (x : ℝ) : Prop :=
  3 - x ≥ 0 ∧ x - 1 > 0 ∧ x ≠ 2

theorem meaningful_range :
  ∀ x : ℝ, is_meaningful x ↔ (1 < x ∧ x ≤ 3 ∧ x ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_meaningful_range_l1007_100773


namespace NUMINAMATH_CALUDE_inequality_proof_l1007_100763

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1007_100763


namespace NUMINAMATH_CALUDE_total_elephants_l1007_100769

theorem total_elephants (we_preserve : ℕ) (gestures : ℕ) (natures_last : ℕ) : 
  we_preserve = 70 →
  gestures = 3 * we_preserve →
  natures_last = 5 * gestures →
  we_preserve + gestures + natures_last = 1330 := by
  sorry

#check total_elephants

end NUMINAMATH_CALUDE_total_elephants_l1007_100769


namespace NUMINAMATH_CALUDE_inverse_proportion_l1007_100777

/-- Given that p and q are inversely proportional, prove that if p = 20 when q = 8, then p = 16 when q = 10. -/
theorem inverse_proportion (p q : ℝ) (h : p * q = 20 * 8) : 
  p * 10 = 16 * 10 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l1007_100777


namespace NUMINAMATH_CALUDE_that_and_this_percentage_l1007_100700

/-- Proves that "that and this" plus half of "that and this" is 200% of three-quarters of "that and this" -/
theorem that_and_this_percentage : 
  ∀ x : ℝ, x > 0 → (x + 0.5 * x) / (0.75 * x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_that_and_this_percentage_l1007_100700


namespace NUMINAMATH_CALUDE_shaded_area_sum_l1007_100701

/-- Represents the shaded area in each level of the square division pattern -/
def shadedAreaSeries : ℕ → ℚ
  | 0 => 1/4
  | n+1 => (1/4) * shadedAreaSeries n

/-- The sum of the infinite geometric series representing the total shaded area -/
def totalShadedArea : ℚ := 1/3

/-- Theorem stating that the sum of the infinite geometric series is 1/3 -/
theorem shaded_area_sum : 
  (∑' n, shadedAreaSeries n) = totalShadedArea := by
  sorry

#check shaded_area_sum

end NUMINAMATH_CALUDE_shaded_area_sum_l1007_100701


namespace NUMINAMATH_CALUDE_log_equation_solution_l1007_100721

theorem log_equation_solution (k x : ℝ) (h : k > 0) (h' : x > 0) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 7) = 6 → x = 117649 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1007_100721


namespace NUMINAMATH_CALUDE_first_number_proof_l1007_100720

theorem first_number_proof (x : ℝ) : x + 33 + 333 + 3.33 = 369.63 → x = 0.30 := by
  sorry

end NUMINAMATH_CALUDE_first_number_proof_l1007_100720


namespace NUMINAMATH_CALUDE_abs_sin_integral_over_2pi_l1007_100762

theorem abs_sin_integral_over_2pi (f : ℝ → ℝ) : 
  (∫ x in (0)..(2 * Real.pi), |Real.sin x|) = 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_sin_integral_over_2pi_l1007_100762


namespace NUMINAMATH_CALUDE_relationship_between_exponents_l1007_100736

theorem relationship_between_exponents 
  (a b c d : ℝ) 
  (x y q z : ℝ) 
  (h1 : a^(2*x) = c^(2*q)) 
  (h2 : a^(2*x) = b^2) 
  (h3 : c^(3*y) = a^(3*z)) 
  (h4 : c^(3*y) = d^2) 
  (h5 : a ≠ 0) 
  (h6 : b ≠ 0) 
  (h7 : c ≠ 0) 
  (h8 : d ≠ 0) : 
  x * y = q * z := by
sorry

end NUMINAMATH_CALUDE_relationship_between_exponents_l1007_100736


namespace NUMINAMATH_CALUDE_remove_parentheses_l1007_100709

theorem remove_parentheses (a : ℝ) : -(2*a - 1) = -2*a + 1 := by
  sorry

end NUMINAMATH_CALUDE_remove_parentheses_l1007_100709


namespace NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l1007_100752

-- Define a fair coin toss
def fair_coin_toss : Type := Bool

-- Define the number of tosses
def num_tosses : Nat := 8

-- Define the number of heads we're looking for
def target_heads : Nat := 3

-- Define the probability of getting exactly 'target_heads' in 'num_tosses'
def probability_exact_heads : ℚ :=
  (Nat.choose num_tosses target_heads : ℚ) / (2 ^ num_tosses : ℚ)

-- Theorem statement
theorem probability_three_heads_in_eight_tosses :
  probability_exact_heads = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l1007_100752


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l1007_100744

theorem simplify_trig_expression :
  Real.sqrt (2 - Real.sin 1 ^ 2 + Real.cos 2) = Real.sqrt 3 * Real.cos 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l1007_100744


namespace NUMINAMATH_CALUDE_sum_divisible_by_three_l1007_100788

theorem sum_divisible_by_three (a : ℤ) : ∃ k : ℤ, a^3 + 2*a = 3*k := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_three_l1007_100788


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l1007_100712

/-- Represents a cube with a given side length -/
structure Cube where
  side_length : ℝ

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℝ :=
  6 * c.side_length * c.side_length

/-- Calculates the increase in surface area after making cuts -/
def surface_area_increase (c : Cube) (num_cuts : ℕ) : ℝ :=
  2 * c.side_length * c.side_length * num_cuts

/-- Theorem: The increase in surface area of a 10 cm cube after three cuts is 600 cm² -/
theorem cube_surface_area_increase :
  let c := Cube.mk 10
  surface_area_increase c 3 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l1007_100712


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l1007_100756

theorem greatest_integer_satisfying_inequality :
  ∃ (n : ℤ), n^2 - 11*n + 24 ≤ 0 ∧
  n = 8 ∧
  ∀ (m : ℤ), m^2 - 11*m + 24 ≤ 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l1007_100756


namespace NUMINAMATH_CALUDE_point_P_coordinates_l1007_100722

/-- Given a point P with coordinates (3m+6, m-3), prove its coordinates under different conditions --/
theorem point_P_coordinates (m : ℝ) :
  let P : ℝ × ℝ := (3*m + 6, m - 3)
  -- Condition 1: P lies on the angle bisector in the first and third quadrants
  (P.1 = P.2 → P = (-7.5, -7.5)) ∧
  -- Condition 2: The ordinate of P is 5 greater than the abscissa
  (P.2 = P.1 + 5 → P = (-15, -10)) ∧
  -- Condition 3: P lies on the line passing through A(3, -2) and parallel to the y-axis
  (P.1 = 3 → P = (3, -4)) := by sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l1007_100722


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l1007_100782

theorem fixed_point_parabola (d : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 5 * x^2 + d * x + 3 * d
  f (-3) = 45 := by sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l1007_100782


namespace NUMINAMATH_CALUDE_grape_rate_calculation_l1007_100770

theorem grape_rate_calculation (grape_weight : ℕ) (mango_weight : ℕ) (mango_rate : ℕ) (total_paid : ℕ) :
  grape_weight = 8 →
  mango_weight = 9 →
  mango_rate = 55 →
  total_paid = 1055 →
  ∃ (grape_rate : ℕ), grape_rate * grape_weight + mango_rate * mango_weight = total_paid ∧ grape_rate = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_grape_rate_calculation_l1007_100770


namespace NUMINAMATH_CALUDE_specific_pairs_probability_l1007_100787

/-- The probability of two specific pairs forming in a random pairing of students -/
theorem specific_pairs_probability (n : ℕ) (h : n = 32) : 
  (1 : ℚ) / (n - 1) * (1 : ℚ) / (n - 2) = 1 / 930 :=
by sorry

end NUMINAMATH_CALUDE_specific_pairs_probability_l1007_100787


namespace NUMINAMATH_CALUDE_vectors_are_parallel_l1007_100789

def a : ℝ × ℝ × ℝ := (1, 2, -2)
def b : ℝ × ℝ × ℝ := (-2, -4, 4)

theorem vectors_are_parallel : ∃ k : ℝ, b = k • a := by
  sorry

end NUMINAMATH_CALUDE_vectors_are_parallel_l1007_100789


namespace NUMINAMATH_CALUDE_sequence_problem_l1007_100766

/-- Given two sequences {a_n} and {b_n}, where:
    1) a_1 = 1
    2) {b_n} is a geometric sequence
    3) For all n, b_n = a_(n+1) / a_n
    4) b_10 * b_11 = 2016^(1/10)
    Prove that a_21 = 2016 -/
theorem sequence_problem (a b : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n)
  (h3 : ∀ n : ℕ, b n = a (n + 1) / a n)
  (h4 : b 10 * b 11 = 2016^(1/10)) :
  a 21 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1007_100766


namespace NUMINAMATH_CALUDE_fair_coin_four_flips_at_least_two_tails_l1007_100754

/-- The probability of getting exactly k successes in n trials with probability p -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of getting at least 2 but not more than 4 tails in 4 flips of a fair coin -/
theorem fair_coin_four_flips_at_least_two_tails : 
  (binomial_probability 4 2 0.5 + binomial_probability 4 3 0.5 + binomial_probability 4 4 0.5) = 0.6875 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_four_flips_at_least_two_tails_l1007_100754


namespace NUMINAMATH_CALUDE_polynomial_division_l1007_100711

def dividend (x : ℚ) : ℚ := 10*x^4 + 5*x^3 - 9*x^2 + 7*x + 2
def divisor (x : ℚ) : ℚ := 3*x^2 + 2*x + 1
def quotient (x : ℚ) : ℚ := (10/3)*x^2 - (5/9)*x - 193/243
def remainder (x : ℚ) : ℚ := (592/27)*x + 179/27

theorem polynomial_division :
  ∀ x : ℚ, dividend x = divisor x * quotient x + remainder x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_l1007_100711


namespace NUMINAMATH_CALUDE_tunneled_cube_surface_area_l1007_100735

/-- Represents a cube with its dimensions and composition -/
structure Cube where
  side_length : ℕ
  sub_cube_side : ℕ
  sub_cube_count : ℕ

/-- Represents the tunneling operation on the cube -/
structure TunneledCube extends Cube where
  removed_layers : ℕ
  removed_edge_units : ℕ

/-- Calculates the surface area of a tunneled cube -/
def surface_area (tc : TunneledCube) : ℕ :=
  sorry

/-- The main theorem stating the surface area of the specific tunneled cube -/
theorem tunneled_cube_surface_area :
  let original_cube : Cube := {
    side_length := 12,
    sub_cube_side := 3,
    sub_cube_count := 64
  }
  let tunneled_cube : TunneledCube := {
    side_length := original_cube.side_length,
    sub_cube_side := original_cube.sub_cube_side,
    sub_cube_count := original_cube.sub_cube_count,
    removed_layers := 2,
    removed_edge_units := 1
  }
  surface_area tunneled_cube = 2496 := by
  sorry

end NUMINAMATH_CALUDE_tunneled_cube_surface_area_l1007_100735


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l1007_100760

theorem sum_of_x_and_y_is_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 28) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l1007_100760


namespace NUMINAMATH_CALUDE_min_soldiers_to_add_l1007_100791

theorem min_soldiers_to_add (N : ℕ) (h1 : N % 7 = 2) (h2 : N % 12 = 2) :
  ∃ x : ℕ, x = 82 ∧ 
    (∀ y : ℕ, y < x → ¬((N + y) % 7 = 0 ∧ (N + y) % 12 = 0)) ∧
    (N + x) % 7 = 0 ∧ (N + x) % 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_soldiers_to_add_l1007_100791


namespace NUMINAMATH_CALUDE_colored_copies_correct_l1007_100784

/-- The number of colored copies Sandy made, given that:
  * Colored copies cost 10 cents each
  * White copies cost 5 cents each
  * Sandy made 400 copies in total
  * The total bill was $22.50 -/
def colored_copies : ℕ :=
  let colored_cost : ℚ := 10 / 100  -- 10 cents in dollars
  let white_cost : ℚ := 5 / 100     -- 5 cents in dollars
  let total_copies : ℕ := 400
  let total_bill : ℚ := 45 / 2      -- $22.50 as a rational number
  50  -- The actual value to be proven

theorem colored_copies_correct :
  let colored_cost : ℚ := 10 / 100
  let white_cost : ℚ := 5 / 100
  let total_copies : ℕ := 400
  let total_bill : ℚ := 45 / 2
  ∃ (white_copies : ℕ),
    colored_copies + white_copies = total_copies ∧
    colored_cost * colored_copies + white_cost * white_copies = total_bill :=
by sorry

end NUMINAMATH_CALUDE_colored_copies_correct_l1007_100784


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1007_100702

/-- An arithmetic sequence with sum of first n terms S_n = -2n^2 + 15n -/
def S (n : ℕ+) : ℤ := -2 * n.val ^ 2 + 15 * n.val

/-- The general term of the arithmetic sequence -/
def a (n : ℕ+) : ℤ := 17 - 4 * n.val

theorem arithmetic_sequence_properties :
  ∀ n : ℕ+,
  -- The general term of the sequence is a_n = 17 - 4n
  (∀ k : ℕ+, S k - S (k - 1) = a k) ∧
  -- S_n achieves its maximum value when n = 4
  (∀ k : ℕ+, S k ≤ S 4) ∧
  -- The maximum value of S_n is 28
  S 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1007_100702


namespace NUMINAMATH_CALUDE_intersection_distance_theorem_l1007_100778

/-- A linear function f(x) = ax + b -/
structure LinearFunction where
  a : ℝ
  b : ℝ

/-- The distance between intersection points of two functions -/
def intersectionDistance (f g : ℝ → ℝ) : ℝ := sorry

/-- The theorem statement -/
theorem intersection_distance_theorem (f : LinearFunction) :
  intersectionDistance (fun x => x^2 - 1) (fun x => f.a * x + f.b + 1) = 3 * Real.sqrt 10 →
  intersectionDistance (fun x => x^2) (fun x => f.a * x + f.b + 3) = 3 * Real.sqrt 14 →
  intersectionDistance (fun x => x^2) (fun x => f.a * x + f.b) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_theorem_l1007_100778


namespace NUMINAMATH_CALUDE_valid_numbers_l1007_100742

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ n % 3 = 1 ∧ n % 5 = 3

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {13, 28, 43, 58, 73, 88} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l1007_100742


namespace NUMINAMATH_CALUDE_bijection_and_size_equivalence_l1007_100755

/-- Represents an integer grid -/
def IntegerGrid := ℤ → ℤ → ℤ

/-- Represents a plane partition -/
def PlanePartition := ℕ → ℕ → ℕ

/-- The size of a plane partition -/
def size (pp : PlanePartition) : ℕ := sorry

/-- The bijection between integer grids and plane partitions -/
def grid_to_partition (g : IntegerGrid) : PlanePartition := sorry

/-- The inverse bijection from plane partitions to integer grids -/
def partition_to_grid (pp : PlanePartition) : IntegerGrid := sorry

/-- The sum of integers in a grid, counting k times for k-th highest diagonal -/
def weighted_sum (g : IntegerGrid) : ℤ := sorry

theorem bijection_and_size_equivalence :
  ∃ (f : IntegerGrid → PlanePartition) (g : PlanePartition → IntegerGrid),
    (∀ grid, g (f grid) = grid) ∧
    (∀ partition, f (g partition) = partition) ∧
    (∀ grid, size (f grid) = weighted_sum grid) := by
  sorry

end NUMINAMATH_CALUDE_bijection_and_size_equivalence_l1007_100755


namespace NUMINAMATH_CALUDE_work_completion_problem_l1007_100738

theorem work_completion_problem (first_group_days : ℕ) (second_group_men : ℕ) (second_group_days : ℕ) :
  first_group_days = 18 →
  second_group_men = 108 →
  second_group_days = 6 →
  ∃ (first_group_men : ℕ), first_group_men * first_group_days = second_group_men * second_group_days ∧ first_group_men = 36 :=
by
  sorry


end NUMINAMATH_CALUDE_work_completion_problem_l1007_100738


namespace NUMINAMATH_CALUDE_average_of_solutions_is_zero_l1007_100799

theorem average_of_solutions_is_zero :
  let solutions := {x : ℝ | Real.sqrt (3 * x^2 + 4) = Real.sqrt 28}
  ∃ (s₁ s₂ : ℝ), s₁ ∈ solutions ∧ s₂ ∈ solutions ∧ s₁ ≠ s₂ ∧
    (s₁ + s₂) / 2 = 0 ∧
    ∀ (s : ℝ), s ∈ solutions → s = s₁ ∨ s = s₂ :=
by sorry

end NUMINAMATH_CALUDE_average_of_solutions_is_zero_l1007_100799


namespace NUMINAMATH_CALUDE_instantaneous_speed_at_3_seconds_l1007_100727

/-- The motion equation of an object -/
def s (t : ℝ) : ℝ := 1 - t + 2 * t^2

/-- The instantaneous speed (derivative of s) -/
def v (t : ℝ) : ℝ := -1 + 4 * t

theorem instantaneous_speed_at_3_seconds :
  v 3 = 11 := by sorry

end NUMINAMATH_CALUDE_instantaneous_speed_at_3_seconds_l1007_100727


namespace NUMINAMATH_CALUDE_johns_brother_age_l1007_100765

theorem johns_brother_age :
  ∀ (john_age brother_age : ℕ),
  john_age = 6 * brother_age - 4 →
  john_age + brother_age = 10 →
  brother_age = 2 := by
sorry

end NUMINAMATH_CALUDE_johns_brother_age_l1007_100765


namespace NUMINAMATH_CALUDE_unique_five_digit_numbers_l1007_100751

/-- Represents a five-digit number -/
def FiveDigitNumber := { n : ℕ // 10000 ≤ n ∧ n < 100000 }

/-- Checks if a number starts with a specific digit -/
def starts_with (n : FiveDigitNumber) (d : ℕ) : Prop :=
  n.val / 10000 = d

/-- Moves the first digit of a number to the last position -/
def move_first_to_last (n : FiveDigitNumber) : ℕ :=
  (n.val % 10000) * 10 + (n.val / 10000)

/-- The main theorem stating the unique solution to the problem -/
theorem unique_five_digit_numbers :
  ∃! (n₁ n₂ : FiveDigitNumber),
    starts_with n₁ 2 ∧
    starts_with n₂ 4 ∧
    move_first_to_last n₁ = n₁.val + n₂.val ∧
    move_first_to_last n₂ = n₁.val - n₂.val ∧
    n₁.val = 26829 ∧
    n₂.val = 41463 := by
  sorry


end NUMINAMATH_CALUDE_unique_five_digit_numbers_l1007_100751


namespace NUMINAMATH_CALUDE_cosine_equality_l1007_100776

theorem cosine_equality (a : ℝ) (h : Real.sin (π/3 + a) = 5/12) : 
  Real.cos (π/6 - a) = 5/12 := by sorry

end NUMINAMATH_CALUDE_cosine_equality_l1007_100776


namespace NUMINAMATH_CALUDE_cube_root_of_hundred_l1007_100704

theorem cube_root_of_hundred (x : ℝ) : (Real.sqrt x)^3 = 100 → x = 10^(4/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_hundred_l1007_100704


namespace NUMINAMATH_CALUDE_floor_a4_div_a3_l1007_100781

def a (k : ℕ) : ℕ := Nat.choose 100 (k + 1)

theorem floor_a4_div_a3 : ⌊(a 4 : ℚ) / (a 3 : ℚ)⌋ = 19 := by sorry

end NUMINAMATH_CALUDE_floor_a4_div_a3_l1007_100781


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1007_100794

/-- Given a geometric sequence {aₙ} with a₁ > 0 and a₂a₄ + 2a₃a₅ + a₄a₆ = 36, prove that a₃ + a₅ = 6 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1)
    (h_pos : a 1 > 0) (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36) :
    a 3 + a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1007_100794


namespace NUMINAMATH_CALUDE_russia_us_size_ratio_l1007_100725

theorem russia_us_size_ratio :
  ∀ (us canada russia : ℝ),
    us > 0 →
    canada = 1.5 * us →
    russia = (4/3) * canada →
    russia / us = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_russia_us_size_ratio_l1007_100725


namespace NUMINAMATH_CALUDE_two_volunteers_same_project_l1007_100758

/-- The number of volunteers -/
def num_volunteers : ℕ := 3

/-- The number of projects -/
def num_projects : ℕ := 7

/-- The probability that exactly two volunteers are assigned to the same project -/
def probability_two_same_project : ℚ := 18/49

theorem two_volunteers_same_project :
  (num_volunteers = 3) →
  (num_projects = 7) →
  (∀ volunteer, volunteer ≤ num_volunteers → ∃! project, project ≤ num_projects) →
  probability_two_same_project = 18/49 := by
  sorry

end NUMINAMATH_CALUDE_two_volunteers_same_project_l1007_100758


namespace NUMINAMATH_CALUDE_brads_money_l1007_100739

theorem brads_money (total : ℚ) (josh_brad_ratio : ℚ) (josh_doug_ratio : ℚ)
  (h1 : total = 68)
  (h2 : josh_brad_ratio = 2)
  (h3 : josh_doug_ratio = 3/4) :
  ∃ (brad : ℚ), brad = 12 ∧ 
    ∃ (josh doug : ℚ), 
      josh = josh_brad_ratio * brad ∧
      josh = josh_doug_ratio * doug ∧
      josh + doug + brad = total :=
by sorry

end NUMINAMATH_CALUDE_brads_money_l1007_100739


namespace NUMINAMATH_CALUDE_milk_delivery_theorem_l1007_100737

/-- Calculates the number of jars of milk good for sale given the delivery conditions --/
def goodJarsForSale (
  normalDelivery : ℕ
  ) (jarsPerCarton : ℕ
  ) (cartonShortage : ℕ
  ) (damagedJarsPerCarton : ℕ
  ) (cartonsWithDamagedJars : ℕ
  ) (totallyDamagedCartons : ℕ
  ) : ℕ :=
  let deliveredCartons := normalDelivery - cartonShortage
  let totalJars := deliveredCartons * jarsPerCarton
  let damagedJars := damagedJarsPerCarton * cartonsWithDamagedJars + totallyDamagedCartons * jarsPerCarton
  totalJars - damagedJars

/-- Theorem stating that under the given conditions, there are 565 jars of milk good for sale --/
theorem milk_delivery_theorem :
  goodJarsForSale 50 20 20 3 5 1 = 565 := by
  sorry

end NUMINAMATH_CALUDE_milk_delivery_theorem_l1007_100737


namespace NUMINAMATH_CALUDE_cars_meeting_halfway_l1007_100796

/-- Two cars meeting halfway between two points --/
theorem cars_meeting_halfway 
  (total_distance : ℝ) 
  (speed_car1 : ℝ) 
  (start_time_car1 start_time_car2 : ℕ) 
  (speed_car2 : ℝ) :
  total_distance = 600 →
  speed_car1 = 50 →
  start_time_car1 = 7 →
  start_time_car2 = 8 →
  (total_distance / 2) / speed_car1 + start_time_car1 = 
    (total_distance / 2) / speed_car2 + start_time_car2 →
  speed_car2 = 60 := by
sorry

end NUMINAMATH_CALUDE_cars_meeting_halfway_l1007_100796


namespace NUMINAMATH_CALUDE_sales_prediction_at_34_l1007_100730

/-- Represents the linear regression equation for predicting cold drink sales based on temperature -/
def predict_sales (x : ℝ) : ℝ := 2 * x + 60

/-- Theorem stating that when the temperature is 34°C, the predicted sales volume is 128 cups -/
theorem sales_prediction_at_34 :
  predict_sales 34 = 128 := by
  sorry

end NUMINAMATH_CALUDE_sales_prediction_at_34_l1007_100730


namespace NUMINAMATH_CALUDE_family_strawberry_picking_l1007_100786

/-- The total weight of strawberries picked by a family -/
theorem family_strawberry_picking (marco_weight dad_weight mom_weight sister_weight : ℕ) 
  (h1 : marco_weight = 8)
  (h2 : dad_weight = 32)
  (h3 : mom_weight = 22)
  (h4 : sister_weight = 14) :
  marco_weight + dad_weight + mom_weight + sister_weight = 76 := by
  sorry

#check family_strawberry_picking

end NUMINAMATH_CALUDE_family_strawberry_picking_l1007_100786


namespace NUMINAMATH_CALUDE_powers_of_two_difference_divisible_by_1987_l1007_100749

theorem powers_of_two_difference_divisible_by_1987 :
  ∃ a b : ℕ, 0 ≤ a ∧ a < b ∧ b ≤ 1987 ∧ (2^b - 2^a) % 1987 = 0 := by
  sorry

end NUMINAMATH_CALUDE_powers_of_two_difference_divisible_by_1987_l1007_100749


namespace NUMINAMATH_CALUDE_negative_root_implies_inequality_l1007_100718

theorem negative_root_implies_inequality (a : ℝ) : 
  (∃ x : ℝ, x - 3*a + 9 = 0 ∧ x < 0) → (a - 4) * (a - 5) > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_root_implies_inequality_l1007_100718


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1007_100713

def A : Set ℝ := {x | x + 2 > 0}
def B : Set ℝ := {-3, -2, -1, 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1007_100713
