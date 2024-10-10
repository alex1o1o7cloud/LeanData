import Mathlib

namespace systematic_sampling_solution_l2516_251618

/-- Represents a systematic sampling problem -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ

/-- Represents a solution to a systematic sampling problem -/
structure SystematicSamplingSolution where
  excluded : ℕ
  interval : ℕ

/-- Checks if a solution is valid for a given systematic sampling problem -/
def is_valid_solution (problem : SystematicSampling) (solution : SystematicSamplingSolution) : Prop :=
  (problem.population_size - solution.excluded) % problem.sample_size = 0 ∧
  (problem.population_size - solution.excluded) / problem.sample_size = solution.interval

theorem systematic_sampling_solution 
  (problem : SystematicSampling) 
  (h_pop : problem.population_size = 102) 
  (h_sample : problem.sample_size = 9) :
  ∃ (solution : SystematicSamplingSolution), 
    solution.excluded = 3 ∧ 
    solution.interval = 11 ∧ 
    is_valid_solution problem solution :=
sorry

end systematic_sampling_solution_l2516_251618


namespace f_inequality_range_l2516_251647

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem f_inequality_range (x : ℝ) : 
  f x > f (2*x - 1) ↔ x ∈ Set.Ioo (1/3) 1 :=
sorry

end f_inequality_range_l2516_251647


namespace melanie_has_41_balloons_l2516_251602

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 40

/-- The total number of blue balloons Joan and Melanie have together -/
def total_balloons : ℕ := 81

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := total_balloons - joan_balloons

/-- Theorem stating that Melanie has 41 blue balloons -/
theorem melanie_has_41_balloons : melanie_balloons = 41 := by
  sorry

end melanie_has_41_balloons_l2516_251602


namespace slower_speed_calculation_l2516_251695

theorem slower_speed_calculation (actual_distance : ℝ) (faster_speed : ℝ) (additional_distance : ℝ) :
  actual_distance = 40 →
  faster_speed = 12 →
  additional_distance = 20 →
  ∃ slower_speed : ℝ, 
    slower_speed > 0 ∧
    actual_distance / slower_speed = (actual_distance + additional_distance) / faster_speed ∧
    slower_speed = 8 := by
  sorry

end slower_speed_calculation_l2516_251695


namespace reciprocal_difference_square_sum_product_difference_l2516_251658

/-- The difference between the reciprocal of x and y is equal to 1/x - y, where x ≠ 0 -/
theorem reciprocal_difference (x y : ℝ) (h : x ≠ 0) :
  1 / x - y = 1 / x - y := by sorry

/-- The difference between the square of the sum of a and b and the product of a and b
    is equal to (a+b)^2 - ab -/
theorem square_sum_product_difference (a b : ℝ) :
  (a + b)^2 - a * b = (a + b)^2 - a * b := by sorry

end reciprocal_difference_square_sum_product_difference_l2516_251658


namespace uncle_bob_can_park_l2516_251641

-- Define the number of total spaces and parked cars
def total_spaces : ℕ := 18
def parked_cars : ℕ := 14

-- Define a function to calculate the number of ways to distribute n items into k groups
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

-- Define the probability of Uncle Bob finding a parking space
def uncle_bob_parking_probability : ℚ :=
  1 - (stars_and_bars 7 5 : ℚ) / (Nat.choose total_spaces parked_cars : ℚ)

-- Theorem statement
theorem uncle_bob_can_park : 
  uncle_bob_parking_probability = 91 / 102 :=
sorry

end uncle_bob_can_park_l2516_251641


namespace intersection_point_coordinates_l2516_251640

-- Define the triangle ABC and points D, E, P
variable (A B C D E P : ℝ × ℝ)

-- Define the conditions
def D_on_BC_extended (A B C D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ D = B + t • (C - B)

def E_on_AC (A C E : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ E = A + t • (C - A)

def BD_DC_ratio (B C D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 0 ∧ D = B + (2/3) • (C - B)

def AE_EC_ratio (A C E : ℝ × ℝ) : Prop :=
  E = A + (2/3) • (C - A)

def P_on_BE (B E P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = B + t • (E - B)

def P_on_AD (A D P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = A + t • (D - A)

-- State the theorem
theorem intersection_point_coordinates
  (h1 : D_on_BC_extended A B C D)
  (h2 : E_on_AC A C E)
  (h3 : BD_DC_ratio B C D)
  (h4 : AE_EC_ratio A C E)
  (h5 : P_on_BE B E P)
  (h6 : P_on_AD A D P) :
  P = (1/3) • A + (1/2) • B + (1/6) • C :=
sorry

end intersection_point_coordinates_l2516_251640


namespace alex_friends_count_l2516_251659

def silk_problem (total_silk : ℕ) (silk_per_dress : ℕ) (dresses_made : ℕ) : ℕ :=
  let silk_used := dresses_made * silk_per_dress
  let silk_given := total_silk - silk_used
  silk_given / silk_per_dress

theorem alex_friends_count :
  silk_problem 600 5 100 = 20 := by
  sorry

end alex_friends_count_l2516_251659


namespace nonDefectiveEnginesCount_l2516_251625

/-- Given a number of batches and engines per batch, calculates the number of non-defective engines
    when one fourth of the total engines are defective. -/
def nonDefectiveEngines (batches : ℕ) (enginesPerBatch : ℕ) : ℕ :=
  let totalEngines := batches * enginesPerBatch
  let defectiveEngines := totalEngines / 4
  totalEngines - defectiveEngines

/-- Proves that given 5 batches of 80 engines each, with one fourth being defective,
    the number of non-defective engines is 300. -/
theorem nonDefectiveEnginesCount :
  nonDefectiveEngines 5 80 = 300 := by
  sorry

#eval nonDefectiveEngines 5 80

end nonDefectiveEnginesCount_l2516_251625


namespace a₉₉_eq_182_l2516_251622

/-- An arithmetic sequence with specified properties -/
structure ArithmeticSequence where
  -- First term
  a₁ : ℝ
  -- Common difference
  d : ℝ
  -- Sum of first 17 terms is 34
  sum_17 : 17 * a₁ + (17 * 16 / 2) * d = 34
  -- Third term is -10
  a₃ : a₁ + 2 * d = -10

/-- The 99th term of the arithmetic sequence -/
def a₉₉ (seq : ArithmeticSequence) : ℝ := seq.a₁ + 98 * seq.d

/-- Theorem stating that a₉₉ = 182 for the given arithmetic sequence -/
theorem a₉₉_eq_182 (seq : ArithmeticSequence) : a₉₉ seq = 182 := by
  sorry

end a₉₉_eq_182_l2516_251622


namespace unique_solution_system_l2516_251678

theorem unique_solution_system (x y z : ℝ) : 
  (x^2 - 23*y - 25*z = -681) ∧
  (y^2 - 21*x - 21*z = -419) ∧
  (z^2 - 19*x - 21*y = -313) ↔
  (x = 20 ∧ y = 22 ∧ z = 23) :=
by sorry

end unique_solution_system_l2516_251678


namespace log_equation_solution_l2516_251615

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log 216 / Real.log (2 * x) = x →
  x = 3 ∧ ¬∃ (n : ℕ), x = n^2 ∧ ¬∃ (n : ℕ), x = n^3 ∧ ∃ (n : ℕ), x = n := by
  sorry

end log_equation_solution_l2516_251615


namespace interest_rate_calculation_l2516_251600

/-- Proves that the interest rate at which A lent to B is 15% given the conditions --/
theorem interest_rate_calculation (principal : ℝ) (rate_B_to_C : ℝ) (time : ℝ) (B_gain : ℝ) 
  (h_principal : principal = 2000)
  (h_rate_B_to_C : rate_B_to_C = 17)
  (h_time : time = 4)
  (h_B_gain : B_gain = 160)
  : ∃ R : ℝ, R = 15 ∧ 
    principal * (rate_B_to_C / 100) * time - principal * (R / 100) * time = B_gain :=
by
  sorry

#check interest_rate_calculation

end interest_rate_calculation_l2516_251600


namespace units_digit_of_square_l2516_251667

/-- 
Given an integer n, if the tens digit of n^2 is 7, 
then the units digit of n^2 is 6.
-/
theorem units_digit_of_square (n : ℤ) : 
  (n^2 % 100 / 10 = 7) → (n^2 % 10 = 6) := by
  sorry

end units_digit_of_square_l2516_251667


namespace football_score_proof_l2516_251653

theorem football_score_proof :
  let hawks_touchdowns : ℕ := 4
  let hawks_successful_extra_points : ℕ := 2
  let hawks_failed_extra_points : ℕ := 2
  let hawks_field_goals : ℕ := 2
  let eagles_touchdowns : ℕ := 3
  let eagles_successful_extra_points : ℕ := 3
  let eagles_field_goals : ℕ := 3
  let touchdown_points : ℕ := 6
  let extra_point_points : ℕ := 1
  let field_goal_points : ℕ := 3
  
  let hawks_score : ℕ := hawks_touchdowns * touchdown_points + 
                         hawks_successful_extra_points * extra_point_points + 
                         hawks_field_goals * field_goal_points
  
  let eagles_score : ℕ := eagles_touchdowns * touchdown_points + 
                          eagles_successful_extra_points * extra_point_points + 
                          eagles_field_goals * field_goal_points
  
  let total_score : ℕ := hawks_score + eagles_score
  
  total_score = 62 := by
    sorry

end football_score_proof_l2516_251653


namespace inequality_range_l2516_251649

theorem inequality_range : 
  ∀ x : ℝ, (∀ a b : ℝ, a > 0 ∧ b > 0 → x^2 + x < a/b + b/a) ↔ -2 < x ∧ x < 1 := by
  sorry

end inequality_range_l2516_251649


namespace distribute_five_balls_four_boxes_l2516_251628

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k ^ n

/-- Proof that distributing 5 distinguishable balls into 4 distinguishable boxes results in 1024 ways -/
theorem distribute_five_balls_four_boxes : distribute 5 4 = 1024 := by
  sorry

end distribute_five_balls_four_boxes_l2516_251628


namespace sin_cos_15_ratio_eq_neg_sqrt3_div_3_l2516_251693

theorem sin_cos_15_ratio_eq_neg_sqrt3_div_3 :
  (Real.sin (15 * π / 180) - Real.cos (15 * π / 180)) /
  (Real.sin (15 * π / 180) + Real.cos (15 * π / 180)) = -Real.sqrt 3 / 3 := by
  sorry

end sin_cos_15_ratio_eq_neg_sqrt3_div_3_l2516_251693


namespace investment_plans_count_l2516_251644

/-- The number of cities available for investment -/
def num_cities : ℕ := 5

/-- The number of projects to be invested -/
def num_projects : ℕ := 3

/-- The maximum number of projects that can be invested in a single city -/
def max_projects_per_city : ℕ := 2

/-- The function that calculates the number of investment plans -/
def num_investment_plans : ℕ :=
  -- The actual calculation would go here
  120

/-- Theorem stating that the number of investment plans is 120 -/
theorem investment_plans_count :
  num_investment_plans = 120 := by sorry

end investment_plans_count_l2516_251644


namespace edward_savings_l2516_251687

/-- Represents the amount of money Edward had saved before mowing lawns -/
def money_saved (earnings_per_lawn : ℕ) (lawns_mowed : ℕ) (total_money : ℕ) : ℕ :=
  total_money - (earnings_per_lawn * lawns_mowed)

/-- Theorem stating that Edward's savings before mowing can be calculated -/
theorem edward_savings :
  let earnings_per_lawn : ℕ := 8
  let lawns_mowed : ℕ := 5
  let total_money : ℕ := 47
  money_saved earnings_per_lawn lawns_mowed total_money = 7 := by
  sorry

end edward_savings_l2516_251687


namespace subset_implies_a_values_l2516_251636

theorem subset_implies_a_values (a : ℝ) : 
  let M : Set ℝ := {x | x^2 = 1}
  let N : Set ℝ := {x | a * x = 1}
  N ⊆ M → a ∈ ({-1, 0, 1} : Set ℝ) := by
  sorry

end subset_implies_a_values_l2516_251636


namespace min_four_dollar_frisbees_l2516_251634

theorem min_four_dollar_frisbees :
  ∀ (x y : ℕ),
  x + y = 64 →
  3 * x + 4 * y = 200 →
  y ≥ 8 :=
by
  sorry

end min_four_dollar_frisbees_l2516_251634


namespace negative_fractions_comparison_l2516_251661

theorem negative_fractions_comparison : -4/5 < -2/3 := by
  sorry

end negative_fractions_comparison_l2516_251661


namespace original_average_proof_l2516_251643

theorem original_average_proof (n : ℕ) (original_avg new_avg : ℚ) :
  n > 0 →
  new_avg = 2 * original_avg →
  new_avg = 72 →
  original_avg = 36 := by
sorry

end original_average_proof_l2516_251643


namespace ellipse_equation_l2516_251604

/-- The standard equation of an ellipse given its properties -/
theorem ellipse_equation (f1 f2 p : ℝ × ℝ) (other_ellipse : ℝ → ℝ → Prop) :
  f1 = (0, -4) →
  f2 = (0, 4) →
  p = (-3, 2) →
  (∀ x y, other_ellipse x y ↔ x^2/9 + y^2/4 = 1) →
  (∀ x y, (x^2/15 + y^2/10 = 1) ↔
    (∃ d1 d2 : ℝ,
      d1 + d2 = 10 ∧
      d1^2 = (x - f1.1)^2 + (y - f1.2)^2 ∧
      d2^2 = (x - f2.1)^2 + (y - f2.2)^2 ∧
      x^2/15 + y^2/10 = 1 ∧
      other_ellipse x y)) :=
sorry

end ellipse_equation_l2516_251604


namespace quadratic_roots_sum_l2516_251660

theorem quadratic_roots_sum (p q : ℝ) : 
  p^2 - 6*p + 8 = 0 → q^2 - 6*q + 8 = 0 → p^3 + p^4*q^2 + p^2*q^4 + q^3 = 1352 := by
  sorry

end quadratic_roots_sum_l2516_251660


namespace smallest_n_congruence_l2516_251666

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ (5 * n) % 26 = 1463 % 26 ∧ ∀ (m : ℕ), m > 0 → (5 * m) % 26 = 1463 % 26 → n ≤ m := by
  sorry

end smallest_n_congruence_l2516_251666


namespace crayons_per_box_l2516_251633

/-- Given an industrial machine that makes 321 crayons a day and 45 full boxes a day,
    prove that there are 7 crayons in each box. -/
theorem crayons_per_box :
  ∀ (total_crayons : ℕ) (total_boxes : ℕ),
    total_crayons = 321 →
    total_boxes = 45 →
    ∃ (crayons_per_box : ℕ),
      crayons_per_box * total_boxes ≤ total_crayons ∧
      (crayons_per_box + 1) * total_boxes > total_crayons ∧
      crayons_per_box = 7 :=
by sorry

end crayons_per_box_l2516_251633


namespace division_of_powers_l2516_251688

theorem division_of_powers (n : ℕ) : 19^11 / 19^8 = 6859 := by
  sorry

end division_of_powers_l2516_251688


namespace min_sum_distances_to_lines_l2516_251697

/-- The minimum sum of distances from a point on the parabola y² = 4x to two specific lines -/
theorem min_sum_distances_to_lines : ∃ (a : ℝ),
  let P : ℝ × ℝ := (a^2, 2*a)
  let d₁ : ℝ := |4*a^2 - 6*a + 6| / 5  -- Distance to line 4x - 3y + 6 = 0
  let d₂ : ℝ := a^2                    -- Distance to line x = 0
  (∀ b : ℝ, d₁ + d₂ ≤ |4*b^2 - 6*b + 6| / 5 + b^2) ∧ 
  d₁ + d₂ = 1 :=
sorry

end min_sum_distances_to_lines_l2516_251697


namespace unique_congruence_in_range_l2516_251603

theorem unique_congruence_in_range : ∃! n : ℕ, 3 ≤ n ∧ n ≤ 8 ∧ n % 8 = 123456 % 8 := by
  sorry

end unique_congruence_in_range_l2516_251603


namespace no_valid_gnomon_tiling_l2516_251642

/-- A gnomon is a figure formed by removing one unit square from a 2x2 square -/
def Gnomon : Type := Unit

/-- Represents a tiling of an m × n rectangle with gnomons -/
def GnomonTiling (m n : ℕ) := Unit

/-- Predicate to check if a tiling satisfies the no-rectangle condition -/
def NoRectangleCondition (tiling : GnomonTiling m n) : Prop := sorry

/-- Predicate to check if a tiling satisfies the no-four-vertex condition -/
def NoFourVertexCondition (tiling : GnomonTiling m n) : Prop := sorry

theorem no_valid_gnomon_tiling (m n : ℕ) :
  ¬∃ (tiling : GnomonTiling m n), NoRectangleCondition tiling ∧ NoFourVertexCondition tiling := by
  sorry

end no_valid_gnomon_tiling_l2516_251642


namespace paths_in_8x6_grid_l2516_251617

/-- The number of paths in a grid from bottom-left to top-right -/
def grid_paths (horizontal_steps : ℕ) (vertical_steps : ℕ) : ℕ :=
  Nat.choose (horizontal_steps + vertical_steps) vertical_steps

/-- Theorem: The number of paths in an 8x6 grid is 3003 -/
theorem paths_in_8x6_grid :
  grid_paths 8 6 = 3003 := by
  sorry

end paths_in_8x6_grid_l2516_251617


namespace insufficient_condition_for_similarity_l2516_251683

-- Define the triangles
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the angles
def angle (t : Triangle) (v : Fin 3) : ℝ := sorry

-- Define similarity of triangles
def similar (t1 t2 : Triangle) : Prop := sorry

theorem insufficient_condition_for_similarity (ABC A'B'C' : Triangle) :
  angle ABC 1 = 90 ∧ 
  angle A'B'C' 1 = 90 ∧ 
  angle ABC 0 = 30 ∧ 
  angle ABC 2 = 60 →
  ¬ (∀ t1 t2 : Triangle, similar t1 t2) :=
sorry

end insufficient_condition_for_similarity_l2516_251683


namespace profit_maximization_l2516_251645

/-- Profit function for computer sales --/
def profit_function (x : ℝ) : ℝ := -50 * x + 15000

/-- Constraint on the number of computers --/
def constraint (x : ℝ) : Prop := 100 / 3 ≤ x ∧ x ≤ 100 / 3

theorem profit_maximization (x : ℝ) :
  constraint x →
  ∀ y, constraint y → profit_function y ≤ profit_function x →
  x = 34 :=
sorry

end profit_maximization_l2516_251645


namespace scenic_spot_probabilities_l2516_251623

def total_spots : ℕ := 10
def five_a_spots : ℕ := 4
def four_a_spots : ℕ := 6

def spots_after_yuntai : ℕ := 4

theorem scenic_spot_probabilities :
  (five_a_spots : ℚ) / total_spots = 2 / 5 ∧
  (2 : ℚ) / (spots_after_yuntai * (spots_after_yuntai - 1)) = 1 / 6 := by
  sorry


end scenic_spot_probabilities_l2516_251623


namespace salary_expenditure_percentage_l2516_251610

theorem salary_expenditure_percentage (initial_salary : ℝ) 
  (house_rent_percentage : ℝ) (education_percentage : ℝ) 
  (final_amount : ℝ) : 
  initial_salary = 2125 →
  house_rent_percentage = 20 →
  education_percentage = 10 →
  final_amount = 1377 →
  let remaining_after_rent := initial_salary * (1 - house_rent_percentage / 100)
  let remaining_after_education := remaining_after_rent * (1 - education_percentage / 100)
  let clothes_percentage := (remaining_after_education - final_amount) / remaining_after_education * 100
  clothes_percentage = 10 := by sorry

end salary_expenditure_percentage_l2516_251610


namespace average_monthly_balance_l2516_251692

def monthly_balances : List ℚ := [150, 250, 100, 200, 300]

theorem average_monthly_balance : 
  (monthly_balances.sum / monthly_balances.length : ℚ) = 200 := by
  sorry

end average_monthly_balance_l2516_251692


namespace calculate_expression_l2516_251670

theorem calculate_expression : 
  Real.sqrt 12 - 3 - ((1/3) * Real.sqrt 27 - Real.sqrt 9) = Real.sqrt 3 := by
  sorry

end calculate_expression_l2516_251670


namespace inequality_proof_l2516_251626

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
sorry

end inequality_proof_l2516_251626


namespace root_power_sum_relation_l2516_251605

theorem root_power_sum_relation (t : ℕ → ℝ) (d e f : ℝ) : 
  (∃ (r₁ r₂ r₃ : ℝ), r₁^3 - 7*r₁^2 + 12*r₁ - 20 = 0 ∧ 
                      r₂^3 - 7*r₂^2 + 12*r₂ - 20 = 0 ∧ 
                      r₃^3 - 7*r₃^2 + 12*r₃ - 20 = 0 ∧ 
                      ∀ k, t k = r₁^k + r₂^k + r₃^k) →
  t 0 = 3 →
  t 1 = 7 →
  t 2 = 15 →
  (∀ k ≥ 2, t (k+1) = d * t k + e * t (k-1) + f * t (k-2) - 5) →
  d + e + f = 15 := by
sorry

end root_power_sum_relation_l2516_251605


namespace triangle_inequality_triangle_inequality_theorem_l2516_251624

-- Define a triangle as a structure with three sides
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

-- State the Triangle Inequality Theorem
theorem triangle_inequality (t : Triangle) : 
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b := by
  sorry

-- Define the property we want to prove
def sum_of_two_sides_greater_than_third (t : Triangle) : Prop :=
  (t.a + t.b > t.c) ∧ (t.b + t.c > t.a) ∧ (t.c + t.a > t.b)

-- Prove that the Triangle Inequality Theorem holds for all triangles
theorem triangle_inequality_theorem :
  ∀ t : Triangle, sum_of_two_sides_greater_than_third t := by
  sorry

end triangle_inequality_triangle_inequality_theorem_l2516_251624


namespace min_value_quadratic_form_l2516_251681

theorem min_value_quadratic_form (a b c d : ℝ) (h : 5*a + 6*b - 7*c + 4*d = 1) :
  3*a^2 + 2*b^2 + 5*c^2 + d^2 ≥ 15/782 ∧
  ∃ (a₀ b₀ c₀ d₀ : ℝ), 5*a₀ + 6*b₀ - 7*c₀ + 4*d₀ = 1 ∧ 3*a₀^2 + 2*b₀^2 + 5*c₀^2 + d₀^2 = 15/782 :=
by sorry

end min_value_quadratic_form_l2516_251681


namespace find_A_in_subtraction_l2516_251611

/-- Given that AB82 - 9C9 = 493D and A, B, C, D are different digits, prove that A = 5 -/
theorem find_A_in_subtraction (A B C D : ℕ) : 
  A * 1000 + B * 100 + 82 - (9 * 100 + C * 10 + 9) = 4 * 100 + 9 * 10 + D →
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A = 5 := by
sorry

end find_A_in_subtraction_l2516_251611


namespace floor_sum_inequality_l2516_251669

theorem floor_sum_inequality (x y : ℝ) : ⌊x + y⌋ ≤ ⌊x⌋ + ⌊y⌋ := by
  sorry

end floor_sum_inequality_l2516_251669


namespace f_range_and_triangle_property_l2516_251627

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, 1 - Real.sqrt 2 * Real.sin x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1 + Real.sqrt 2 * Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem f_range_and_triangle_property :
  (∀ y ∈ Set.Icc (-1 : ℝ) 2, ∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = y) ∧
  (∀ {a b c A B C : ℝ},
    b / a = Real.sqrt 3 →
    (Real.sin B * Real.cos A) / Real.sin A = 2 - Real.cos B →
    f B = 1) :=
sorry

end f_range_and_triangle_property_l2516_251627


namespace trajectory_of_T_l2516_251621

-- Define the curve C
def C (x y : ℝ) : Prop := 4 * x^2 - y + 1 = 0

-- Define the fixed point M
def M : ℝ × ℝ := (-2, 0)

-- Define the relationship between A, T, and M
def AT_TM_relation (A T : ℝ × ℝ) : Prop :=
  let (xa, ya) := A
  let (xt, yt) := T
  (xa - xt, ya - yt) = (2 * (-2 - xt), 2 * (-yt))

-- Theorem statement
theorem trajectory_of_T (A T : ℝ × ℝ) :
  (∃ x y, A = (x, y) ∧ C x y) →  -- A is on curve C
  AT_TM_relation A T →           -- Relationship between A, T, and M holds
  4 * (3 * T.1 + 4)^2 - 3 * T.2 + 1 = 0 :=  -- Trajectory equation for T
by sorry

end trajectory_of_T_l2516_251621


namespace circle_center_coordinate_sum_l2516_251616

/-- Given a circle with equation x^2 + y^2 + 4x - 12y + 20 = 0, 
    the sum of the x and y coordinates of its center is 4 -/
theorem circle_center_coordinate_sum :
  ∀ (x y : ℝ), x^2 + y^2 + 4*x - 12*y + 20 = 0 →
  ∃ (h k : ℝ), (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = (x^2 + y^2 + 4*x - 12*y + 20)) ∧ 
                h + k = 4 := by
  sorry

end circle_center_coordinate_sum_l2516_251616


namespace arithmetic_sequence_first_term_l2516_251691

/-- Arithmetic sequence sum -/
def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

/-- The problem statement -/
theorem arithmetic_sequence_first_term
  (d : ℚ)
  (h_d : d = 5)
  (h_constant : ∃ (c : ℚ), ∀ (n : ℕ+),
    arithmetic_sum a d (3 * n) / arithmetic_sum a d n = c) :
  a = 5 / 2 :=
sorry

end arithmetic_sequence_first_term_l2516_251691


namespace monomial_degree_implications_l2516_251671

-- Define the condition
def is_monomial_of_degree_5 (a : ℝ) : Prop :=
  2 + (1 + a) = 5

-- Theorem statement
theorem monomial_degree_implications (a : ℝ) 
  (h : is_monomial_of_degree_5 a) : 
  a^3 + 1 = 9 ∧ 
  (a + 1) * (a^2 - a + 1) = 9 ∧ 
  a^3 + 1 = (a + 1) * (a^2 - a + 1) := by
  sorry

end monomial_degree_implications_l2516_251671


namespace ratio_range_l2516_251601

-- Define the condition for the point (x,y)
def satisfies_condition (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 4*y + 16 ≤ 0

-- Define the range for y/x
def in_range (r : ℝ) : Prop :=
  0 ≤ r ∧ r ≤ 4/3

-- Theorem statement
theorem ratio_range (x y : ℝ) (h : satisfies_condition x y) (hx : x ≠ 0) :
  in_range (y / x) :=
sorry

end ratio_range_l2516_251601


namespace quadratic_maximum_l2516_251675

theorem quadratic_maximum : 
  (∀ r : ℝ, -5 * r^2 + 40 * r - 12 ≤ 68) ∧ 
  (∃ r : ℝ, -5 * r^2 + 40 * r - 12 = 68) := by
  sorry

end quadratic_maximum_l2516_251675


namespace tanker_filling_rate_l2516_251668

/-- Proves that the filling rate of 3 barrels per minute is equivalent to 28.62 m³/hour -/
theorem tanker_filling_rate 
  (barrel_rate : ℝ) 
  (liters_per_barrel : ℝ) 
  (h1 : barrel_rate = 3) 
  (h2 : liters_per_barrel = 159) : 
  (barrel_rate * liters_per_barrel * 60) / 1000 = 28.62 := by
  sorry

end tanker_filling_rate_l2516_251668


namespace tan_negative_405_l2516_251686

-- Define the tangent function
noncomputable def tan (θ : ℝ) : ℝ := Real.tan θ

-- Define the property of tangent periodicity
axiom tan_periodic (θ : ℝ) (n : ℤ) : tan θ = tan (θ + n * 360)

-- Define the value of tan(45°)
axiom tan_45 : tan 45 = 1

-- Theorem to prove
theorem tan_negative_405 : tan (-405) = 1 := by
  sorry

end tan_negative_405_l2516_251686


namespace ab_value_l2516_251665

theorem ab_value (a b : ℝ) (h : (a + 2)^2 + |b - 4| = 0) : a^b = 16 := by
  sorry

end ab_value_l2516_251665


namespace unique_solution_cube_equation_l2516_251607

theorem unique_solution_cube_equation (x y : ℕ) :
  y^6 + 2*y^3 - y^2 + 1 = x^3 → x = 1 ∧ y = 0 := by
  sorry

end unique_solution_cube_equation_l2516_251607


namespace erica_age_is_17_l2516_251639

def casper_age : ℕ := 18

def ivy_age (casper_age : ℕ) : ℕ := casper_age + 4

def erica_age (ivy_age : ℕ) : ℕ := ivy_age - 5

theorem erica_age_is_17 :
  erica_age (ivy_age casper_age) = 17 := by
  sorry

end erica_age_is_17_l2516_251639


namespace min_blue_eyes_and_backpack_proof_l2516_251613

def min_blue_eyes_and_backpack (total_students blue_eyes backpacks glasses : ℕ) : ℕ :=
  blue_eyes - (total_students - backpacks)

theorem min_blue_eyes_and_backpack_proof 
  (total_students : ℕ) 
  (blue_eyes : ℕ) 
  (backpacks : ℕ) 
  (glasses : ℕ) 
  (h1 : total_students = 35)
  (h2 : blue_eyes = 18)
  (h3 : backpacks = 25)
  (h4 : glasses = 10)
  (h5 : ∃ (x : ℕ), x ≥ 2 ∧ x ≤ glasses ∧ x ≤ blue_eyes) :
  min_blue_eyes_and_backpack total_students blue_eyes backpacks glasses = 10 := by
  sorry

#eval min_blue_eyes_and_backpack 35 18 25 10

end min_blue_eyes_and_backpack_proof_l2516_251613


namespace recipe_total_cups_l2516_251682

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio :=
  (butter : ℕ)
  (flour : ℕ)
  (sugar : ℕ)
  (eggs : ℕ)

/-- Calculates the total number of cups for all ingredients given a recipe ratio and the amount of sugar -/
def totalCups (ratio : RecipeRatio) (sugarCups : ℕ) : ℕ :=
  let partSize := sugarCups / ratio.sugar
  partSize * (ratio.butter + ratio.flour + ratio.sugar + ratio.eggs)

/-- Theorem stating that for the given recipe ratio and 10 cups of sugar, the total is 30 cups -/
theorem recipe_total_cups : 
  let ratio : RecipeRatio := ⟨2, 7, 5, 1⟩
  totalCups ratio 10 = 30 := by
  sorry

end recipe_total_cups_l2516_251682


namespace reciprocal_sum_theorem_l2516_251630

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 3 * x * y) : 1 / x + 1 / y = 3 := by
  sorry

end reciprocal_sum_theorem_l2516_251630


namespace all_three_classes_l2516_251674

/-- Represents the number of students in each class combination --/
structure ClassCombinations where
  yoga : ℕ
  bridge : ℕ
  painting : ℕ
  yogaBridge : ℕ
  yogaPainting : ℕ
  bridgePainting : ℕ
  allThree : ℕ

/-- Represents the given conditions of the problem --/
def problem_conditions (c : ClassCombinations) : Prop :=
  c.yoga + c.bridge + c.painting + c.yogaBridge + c.yogaPainting + c.bridgePainting + c.allThree = 20 ∧
  c.yoga + c.yogaBridge + c.yogaPainting + c.allThree = 10 ∧
  c.bridge + c.yogaBridge + c.bridgePainting + c.allThree = 13 ∧
  c.painting + c.yogaPainting + c.bridgePainting + c.allThree = 9 ∧
  c.yogaBridge + c.yogaPainting + c.bridgePainting + c.allThree = 9

theorem all_three_classes (c : ClassCombinations) :
  problem_conditions c → c.allThree = 3 := by
  sorry

end all_three_classes_l2516_251674


namespace parabola_equation_l2516_251651

/-- A parabola is defined by its axis equation and standard form equation. -/
structure Parabola where
  /-- The x-coordinate of the axis of the parabola -/
  axis : ℝ
  /-- The coefficient in the standard form equation y² = 2px -/
  p : ℝ
  /-- Condition that p is positive -/
  p_pos : p > 0

/-- The standard form equation of a parabola is y² = 2px -/
def standard_form (para : Parabola) : Prop :=
  ∀ x y : ℝ, y^2 = 2 * para.p * x

/-- The axis equation of a parabola is x = -p/2 -/
def axis_equation (para : Parabola) : Prop :=
  para.axis = -para.p / 2

/-- Theorem: Given a parabola with axis equation x = -2, its standard form equation is y² = 8x -/
theorem parabola_equation (para : Parabola) 
  (h : axis_equation para) 
  (h_axis : para.axis = -2) : 
  standard_form para ∧ para.p = 4 := by
  sorry

end parabola_equation_l2516_251651


namespace six_by_six_grid_squares_l2516_251657

/-- The number of squares of a given size in a grid --/
def count_squares (grid_size : ℕ) (square_size : ℕ) : ℕ :=
  (grid_size + 1 - square_size) ^ 2

/-- The total number of squares in a 6x6 grid --/
def total_squares (grid_size : ℕ) : ℕ :=
  (count_squares grid_size 1) + (count_squares grid_size 2) +
  (count_squares grid_size 3) + (count_squares grid_size 4)

/-- Theorem: The total number of squares in a 6x6 grid is 86 --/
theorem six_by_six_grid_squares :
  total_squares 6 = 86 := by
  sorry


end six_by_six_grid_squares_l2516_251657


namespace angle_sum_zero_l2516_251663

theorem angle_sum_zero (α β : ℝ) (h_acute_α : 0 < α ∧ α < π / 2) (h_acute_β : 0 < β ∧ β < π / 2)
  (h_eq1 : 4 * (Real.cos α)^2 + 3 * (Real.cos β)^2 = 2)
  (h_eq2 : 4 * Real.sin (2 * α) + 3 * Real.sin (2 * β) = 0) :
  α + 3 * β = 0 := by sorry

end angle_sum_zero_l2516_251663


namespace sequence_bounded_l2516_251608

/-- Given a sequence of positive real numbers satisfying a specific condition, prove that the sequence is bounded -/
theorem sequence_bounded (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_cond : ∀ k n m l, k + n = m + l → 
    (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)) :
  ∃ M, ∀ n, a n ≤ M :=
sorry

end sequence_bounded_l2516_251608


namespace travis_cereal_expenditure_l2516_251638

theorem travis_cereal_expenditure 
  (boxes_per_week : ℕ) 
  (cost_per_box : ℚ) 
  (weeks_per_year : ℕ) 
  (h1 : boxes_per_week = 2) 
  (h2 : cost_per_box = 3) 
  (h3 : weeks_per_year = 52) : 
  (boxes_per_week * cost_per_box * weeks_per_year : ℚ) = 312 :=
by sorry

end travis_cereal_expenditure_l2516_251638


namespace modified_deck_choose_two_l2516_251698

/-- Represents a modified deck of cards -/
structure ModifiedDeck :=
  (normal_suits : Nat)  -- Number of suits with 13 cards
  (reduced_suit : Nat)  -- Number of suits with 12 cards

/-- Calculates the number of ways to choose 2 cards from different suits in a modified deck -/
def choose_two_cards (deck : ModifiedDeck) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem modified_deck_choose_two (d : ModifiedDeck) :
  d.normal_suits = 3 ∧ d.reduced_suit = 1 → choose_two_cards d = 1443 :=
sorry

end modified_deck_choose_two_l2516_251698


namespace minutes_conversion_l2516_251680

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- Converts minutes to seconds -/
def minutes_to_seconds (minutes : ℚ) : ℚ :=
  minutes * seconds_per_minute

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℚ) : ℚ :=
  minutes / minutes_per_hour

theorem minutes_conversion (minutes : ℚ) :
  minutes = 25/2 →
  minutes_to_seconds minutes = 750 ∧ minutes_to_hours minutes = 5/24 := by
  sorry

end minutes_conversion_l2516_251680


namespace unique_solution_lcm_gcd_equation_l2516_251673

theorem unique_solution_lcm_gcd_equation : 
  ∃! n : ℕ+, Nat.lcm n 120 = Nat.gcd n 120 + 300 ∧ n = 180 := by
  sorry

end unique_solution_lcm_gcd_equation_l2516_251673


namespace original_equals_scientific_l2516_251679

/-- The number to be expressed in scientific notation -/
def original_number : ℝ := 1650000

/-- The scientific notation representation -/
def scientific_notation : ℝ := 1.65 * (10 ^ 6)

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : original_number = scientific_notation := by
  sorry

end original_equals_scientific_l2516_251679


namespace mark_travel_distance_l2516_251619

/-- Represents the time in minutes to travel one mile on day 1 -/
def initial_time : ℕ := 3

/-- Calculates the time in minutes to travel one mile on a given day -/
def time_for_mile (day : ℕ) : ℕ :=
  initial_time + 3 * (day - 1)

/-- Calculates the distance traveled in miles on a given day -/
def distance_per_day (day : ℕ) : ℕ :=
  if 60 % (time_for_mile day) = 0 then 60 / (time_for_mile day) else 0

/-- Calculates the total distance traveled over 6 days -/
def total_distance : ℕ :=
  (List.range 6).map (fun i => distance_per_day (i + 1)) |> List.sum

theorem mark_travel_distance :
  total_distance = 39 := by sorry

end mark_travel_distance_l2516_251619


namespace first_digit_89_base5_l2516_251656

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Returns the first (leftmost) digit of a number in its base-5 representation -/
def firstDigitBase5 (n : ℕ) : ℕ :=
  (toBase5 n).reverse.head!

theorem first_digit_89_base5 :
  firstDigitBase5 89 = 3 := by sorry

end first_digit_89_base5_l2516_251656


namespace total_peanuts_l2516_251676

def jose_peanuts : ℕ := 85
def kenya_peanuts : ℕ := jose_peanuts + 48
def malachi_peanuts : ℕ := kenya_peanuts + 35

theorem total_peanuts : jose_peanuts + kenya_peanuts + malachi_peanuts = 386 := by
  sorry

end total_peanuts_l2516_251676


namespace taxi_charge_calculation_l2516_251684

/-- Calculates the total charge for a taxi trip -/
def totalCharge (initialFee : ℚ) (additionalChargePerIncrement : ℚ) (incrementDistance : ℚ) (tripDistance : ℚ) : ℚ :=
  initialFee + (tripDistance / incrementDistance).floor * additionalChargePerIncrement

/-- Theorem: The total charge for a 3.6-mile trip with given fees is $5.50 -/
theorem taxi_charge_calculation :
  let initialFee : ℚ := 235 / 100
  let additionalChargePerIncrement : ℚ := 35 / 100
  let incrementDistance : ℚ := 2 / 5
  let tripDistance : ℚ := 36 / 10
  totalCharge initialFee additionalChargePerIncrement incrementDistance tripDistance = 550 / 100 := by
  sorry

#eval totalCharge (235/100) (35/100) (2/5) (36/10)

end taxi_charge_calculation_l2516_251684


namespace cross_product_solution_l2516_251685

theorem cross_product_solution :
  let v1 : ℝ × ℝ × ℝ := (128/15, -2, 7/5)
  let v2 : ℝ × ℝ × ℝ := (4, 5, 3)
  let result : ℝ × ℝ × ℝ := (-13, -20, 23)
  (v1.2.1 * v2.2.2 - v1.2.2 * v2.2.1,
   v1.2.2 * v2.1 - v1.1 * v2.2.2,
   v1.1 * v2.2.1 - v1.2.1 * v2.1) = result :=
by sorry

end cross_product_solution_l2516_251685


namespace jason_added_erasers_l2516_251620

/-- Given an initial number of erasers and a final number of erasers after Jason adds some,
    calculate how many erasers Jason placed in the drawer. -/
def erasers_added (initial_erasers final_erasers : ℕ) : ℕ :=
  final_erasers - initial_erasers

/-- Theorem stating that Jason added 131 erasers to the drawer. -/
theorem jason_added_erasers :
  erasers_added 139 270 = 131 := by sorry

end jason_added_erasers_l2516_251620


namespace border_tile_difference_l2516_251631

/-- Represents an octagonal figure made of tiles -/
structure OctagonalFigure where
  white_tiles : ℕ
  black_tiles : ℕ

/-- Creates a new figure by adding a border of black tiles -/
def add_border (figure : OctagonalFigure) : OctagonalFigure :=
  { white_tiles := figure.white_tiles,
    black_tiles := figure.black_tiles + 8 }

/-- The difference between black and white tiles in a figure -/
def tile_difference (figure : OctagonalFigure) : ℤ :=
  figure.black_tiles - figure.white_tiles

theorem border_tile_difference (original : OctagonalFigure) 
  (h1 : original.white_tiles = 16)
  (h2 : original.black_tiles = 9) :
  tile_difference (add_border original) = 1 := by
  sorry

end border_tile_difference_l2516_251631


namespace colonization_combinations_l2516_251614

def total_planets : ℕ := 15
def earth_like_planets : ℕ := 8
def mars_like_planets : ℕ := 7
def earth_like_cost : ℕ := 3
def mars_like_cost : ℕ := 1
def total_colonization_units : ℕ := 18

def valid_combination (earth_colonies mars_colonies : ℕ) : Prop :=
  earth_colonies ≤ earth_like_planets ∧
  mars_colonies ≤ mars_like_planets ∧
  earth_colonies * earth_like_cost + mars_colonies * mars_like_cost = total_colonization_units

def count_combinations : ℕ := sorry

theorem colonization_combinations :
  count_combinations = 2478 :=
sorry

end colonization_combinations_l2516_251614


namespace seven_digit_palindrome_count_l2516_251672

/-- A seven-digit palindrome is a number of the form abcdcba where a ≠ 0 -/
def SevenDigitPalindrome : Type := Nat

/-- The count of seven-digit palindromes -/
def countSevenDigitPalindromes : Nat := 9000

theorem seven_digit_palindrome_count :
  (Finset.filter (λ n : Nat => n ≥ 1000000 ∧ n ≤ 9999999 ∧ 
    (String.mk (List.reverse (String.toList (toString n)))) = toString n)
    (Finset.range 10000000)).card = countSevenDigitPalindromes := by
  sorry

end seven_digit_palindrome_count_l2516_251672


namespace spherical_coordinate_equivalence_l2516_251696

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Checks if a SphericalPoint is in standard representation -/
def isStandardRepresentation (p : SphericalPoint) : Prop :=
  p.ρ > 0 ∧ 0 ≤ p.θ ∧ p.θ < 2 * Real.pi ∧ 0 ≤ p.φ ∧ p.φ ≤ Real.pi

/-- Theorem stating the equivalence of the given spherical coordinates -/
theorem spherical_coordinate_equivalence :
  let p1 := SphericalPoint.mk 4 (5 * Real.pi / 6) (9 * Real.pi / 4)
  let p2 := SphericalPoint.mk 4 (11 * Real.pi / 6) (Real.pi / 4)
  (p1.ρ = p2.ρ) ∧ 
  (p1.θ % (2 * Real.pi) = p2.θ % (2 * Real.pi)) ∧ 
  (p1.φ % (2 * Real.pi) = p2.φ % (2 * Real.pi)) ∧
  isStandardRepresentation p2 :=
by sorry

end spherical_coordinate_equivalence_l2516_251696


namespace inequality_proof_l2516_251646

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1 / 4) :
  (16 * a * c + a / (c^2 * b) + 16 * c / (a^2 * d) + 4 / (a * c)) * 
  (b * d + b / (256 * d^2 * c) + d / (b^2 * a) + 1 / (64 * b * d)) ≥ 81 / 4 ∧
  (16 * a * c + a / (c^2 * b) + 16 * c / (a^2 * d) + 4 / (a * c)) * 
  (b * d + b / (256 * d^2 * c) + d / (b^2 * a) + 1 / (64 * b * d)) = 81 / 4 ↔ 
  a = 2 ∧ b = 1 ∧ c = 1 / 2 ∧ d = 1 / 4 := by
sorry

end inequality_proof_l2516_251646


namespace monotonic_condition_l2516_251609

/-- A function f is monotonic on an interval [a, b] if it is either
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The main theorem stating the condition for the function to be monotonic. -/
theorem monotonic_condition (a : ℝ) :
  (IsMonotonic (fun x => -x^2 + 4*a*x) 2 4) ↔ (a ≤ 1 ∨ a ≥ 2) :=
sorry

end monotonic_condition_l2516_251609


namespace apartment_complex_households_l2516_251629

/-- Calculates the total number of households in an apartment complex. -/
def total_households (num_buildings : ℕ) (num_floors : ℕ) 
  (households_first_floor : ℕ) (households_other_floors : ℕ) : ℕ :=
  num_buildings * (households_first_floor + (num_floors - 1) * households_other_floors)

/-- Theorem stating that the total number of households in the given apartment complex is 68. -/
theorem apartment_complex_households : 
  total_households 4 6 2 3 = 68 := by
  sorry

#eval total_households 4 6 2 3

end apartment_complex_households_l2516_251629


namespace female_officers_count_l2516_251654

/-- The total number of officers on duty -/
def total_on_duty : ℕ := 300

/-- The fraction of officers on duty who are female -/
def female_fraction : ℚ := 1/2

/-- The percentage of female officers who were on duty -/
def female_on_duty_percent : ℚ := 15/100

/-- The total number of female officers on the police force -/
def total_female_officers : ℕ := 1000

theorem female_officers_count :
  (total_on_duty : ℚ) * female_fraction / female_on_duty_percent = total_female_officers := by
  sorry

end female_officers_count_l2516_251654


namespace max_a_value_l2516_251612

/-- An even function f defined on ℝ such that f(x) = e^x for x ≥ 0 -/
noncomputable def f : ℝ → ℝ :=
  fun x => if x ≥ 0 then Real.exp x else Real.exp (-x)

theorem max_a_value :
  (∃ a : ℝ, ∀ x ∈ Set.Icc a (a + 1), f (x + a) ≥ f x ^ 2) ∧
  (∀ a : ℝ, a > -3/4 → ∃ x ∈ Set.Icc a (a + 1), f (x + a) < f x ^ 2) :=
by sorry

end max_a_value_l2516_251612


namespace box_max_volume_l2516_251606

variable (a : ℝ) (x : ℝ)

-- Define the volume function
def V (a x : ℝ) : ℝ := (a - 2*x)^2 * x

-- State the theorem
theorem box_max_volume (h1 : a > 0) (h2 : 0 < x) (h3 : x < a/2) :
  ∃ (x_max : ℝ), x_max = a/6 ∧ 
  (∀ y, 0 < y → y < a/2 → V a y ≤ V a x_max) ∧
  V a x_max = 2*a^3/27 :=
sorry

end box_max_volume_l2516_251606


namespace solution_set_inequality_l2516_251699

theorem solution_set_inequality (x : ℝ) (h : x ≠ 0) :
  (2*x - 1) / x < 1 ↔ 0 < x ∧ x < 1 := by sorry

end solution_set_inequality_l2516_251699


namespace round_robin_chess_tournament_l2516_251655

theorem round_robin_chess_tournament (n : Nat) (h : n = 10) :
  let total_games := n * (n - 1) / 2
  total_games = 45 := by
  sorry

end round_robin_chess_tournament_l2516_251655


namespace different_arrangements_count_l2516_251694

def num_red_balls : ℕ := 6
def num_green_balls : ℕ := 3
def num_selected_balls : ℕ := 4

def num_arrangements (r g s : ℕ) : ℕ :=
  (Nat.choose s s) +
  (Nat.choose s 1) * 2 +
  (Nat.choose s 2)

theorem different_arrangements_count :
  num_arrangements num_red_balls num_green_balls num_selected_balls = 15 := by
  sorry

end different_arrangements_count_l2516_251694


namespace train_platform_equal_length_l2516_251677

/-- Given a train and platform with specific properties, prove that their lengths are equal --/
theorem train_platform_equal_length 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_speed = 108 * 1000 / 60) -- 108 km/hr converted to m/min
  (h2 : train_length = 900)
  (h3 : crossing_time = 1) :
  train_length = train_speed * crossing_time - train_length := by
  sorry

#check train_platform_equal_length

end train_platform_equal_length_l2516_251677


namespace total_weekly_calories_l2516_251662

/-- Represents the number of each type of burger consumed on a given day -/
structure DailyConsumption where
  burgerA : ℕ
  burgerB : ℕ
  burgerC : ℕ

/-- Calculates the total calories for a given daily consumption -/
def dailyCalories (d : DailyConsumption) : ℕ :=
  d.burgerA * 350 + d.burgerB * 450 + d.burgerC * 550

/-- Represents Dimitri's burger consumption for the week -/
def weeklyConsumption : List DailyConsumption :=
  [
    ⟨2, 1, 0⟩,  -- Day 1
    ⟨1, 2, 1⟩,  -- Day 2
    ⟨1, 1, 2⟩,  -- Day 3
    ⟨0, 3, 0⟩,  -- Day 4
    ⟨1, 1, 1⟩,  -- Day 5
    ⟨2, 0, 3⟩,  -- Day 6
    ⟨0, 1, 2⟩   -- Day 7
  ]

/-- Theorem: The total calories consumed by Dimitri in a week is 11,450 -/
theorem total_weekly_calories : 
  (weeklyConsumption.map dailyCalories).sum = 11450 := by
  sorry


end total_weekly_calories_l2516_251662


namespace sheep_transaction_gain_l2516_251690

/-- Calculates the percent gain on a sheep transaction given specific conditions. -/
theorem sheep_transaction_gain : ∀ (x : ℝ),
  x > 0 →  -- x represents the cost per sheep
  let total_cost : ℝ := 850 * x
  let first_sale_revenue : ℝ := total_cost
  let first_sale_price_per_sheep : ℝ := first_sale_revenue / 800
  let second_sale_price_per_sheep : ℝ := first_sale_price_per_sheep * 1.1
  let second_sale_revenue : ℝ := second_sale_price_per_sheep * 50
  let total_revenue : ℝ := first_sale_revenue + second_sale_revenue
  let profit : ℝ := total_revenue - total_cost
  let percent_gain : ℝ := (profit / total_cost) * 100
  percent_gain = 6.875 := by
  sorry


end sheep_transaction_gain_l2516_251690


namespace carlos_earnings_l2516_251648

/-- Carlos's work hours and earnings problem -/
theorem carlos_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) :
  hours_week1 = 12 →
  hours_week2 = 18 →
  extra_earnings = 36 →
  ∃ (hourly_wage : ℚ),
    hourly_wage * (hours_week2 - hours_week1) = extra_earnings ∧
    hourly_wage * (hours_week1 + hours_week2) = 180 :=
by sorry


end carlos_earnings_l2516_251648


namespace regression_line_equation_l2516_251632

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a linear equation y = mx + b -/
structure LinearEquation where
  m : ℝ
  b : ℝ

/-- Check if a point lies on a line given by a linear equation -/
def pointOnLine (p : Point) (eq : LinearEquation) : Prop :=
  p.y = eq.m * p.x + eq.b

theorem regression_line_equation 
  (slope : ℝ) 
  (center : Point) 
  (h_slope : slope = 1.23)
  (h_center : center = ⟨4, 5⟩) :
  ∃ (eq : LinearEquation), 
    eq.m = slope ∧ 
    pointOnLine center eq ∧ 
    eq = ⟨1.23, 0.08⟩ := by
  sorry

end regression_line_equation_l2516_251632


namespace jake_current_weight_l2516_251664

/-- Jake's current weight in pounds -/
def jake_weight : ℕ := 219

/-- Jake's sister's current weight in pounds -/
def sister_weight : ℕ := 318 - jake_weight

theorem jake_current_weight : 
  (jake_weight + sister_weight = 318) ∧ 
  (jake_weight - 12 = 2 * (sister_weight + 4)) → 
  jake_weight = 219 := by sorry

end jake_current_weight_l2516_251664


namespace expression_evaluation_l2516_251637

theorem expression_evaluation : 
  Real.sin (π / 4) ^ 2 - Real.sqrt 27 + (1 / 2) * ((Real.sqrt 3 - 2006) ^ 0) + 6 * Real.tan (π / 6) = 1 - Real.sqrt 3 := by
  sorry

end expression_evaluation_l2516_251637


namespace exists_non_increasing_log_l2516_251689

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem exists_non_increasing_log :
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ¬(∀ (x y : ℝ), x < y → log a x < log a y) :=
sorry

end exists_non_increasing_log_l2516_251689


namespace set_equality_l2516_251650

-- Define sets A and B
def A : Set ℝ := {x | x < 4}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- Define the set we want to prove equal to our result
def S : Set ℝ := {x | x ∈ A ∧ x ∉ A ∩ B}

-- State the theorem
theorem set_equality : S = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by sorry

end set_equality_l2516_251650


namespace mika_birthday_stickers_l2516_251652

/-- The number of stickers Mika gets for her birthday -/
def birthday_stickers : ℕ := sorry

/-- The number of stickers Mika initially had -/
def initial_stickers : ℕ := 20

/-- The number of stickers Mika bought -/
def bought_stickers : ℕ := 26

/-- The number of stickers Mika gave to her sister -/
def given_stickers : ℕ := 6

/-- The number of stickers Mika used for the greeting card -/
def used_stickers : ℕ := 58

/-- The number of stickers Mika has left -/
def remaining_stickers : ℕ := 2

theorem mika_birthday_stickers :
  initial_stickers + bought_stickers + birthday_stickers - given_stickers - used_stickers = remaining_stickers ∧
  birthday_stickers = 20 :=
sorry

end mika_birthday_stickers_l2516_251652


namespace unique_solution_l2516_251635

/-- The equation from the problem -/
def equation (x y : ℝ) : Prop :=
  11 * x^2 + 2 * x * y + 9 * y^2 + 8 * x - 12 * y + 6 = 0

/-- There exists exactly one pair of real numbers (x, y) that satisfies the equation -/
theorem unique_solution : ∃! p : ℝ × ℝ, equation p.1 p.2 := by sorry

end unique_solution_l2516_251635
