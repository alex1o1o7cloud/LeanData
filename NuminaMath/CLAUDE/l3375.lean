import Mathlib

namespace NUMINAMATH_CALUDE_not_all_squares_congruent_square_is_convex_square_is_equiangular_square_has_equal_sides_all_squares_are_similar_l3375_337555

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem statement
theorem not_all_squares_congruent : ¬ ∀ (s1 s2 : Square), congruent s1 s2 := by
  sorry

-- Other properties of squares (not used in the proof, but included for completeness)
theorem square_is_convex (s : Square) : True := by
  sorry

theorem square_is_equiangular (s : Square) : True := by
  sorry

theorem square_has_equal_sides (s : Square) : True := by
  sorry

theorem all_squares_are_similar : ∀ (s1 s2 : Square), True := by
  sorry

end NUMINAMATH_CALUDE_not_all_squares_congruent_square_is_convex_square_is_equiangular_square_has_equal_sides_all_squares_are_similar_l3375_337555


namespace NUMINAMATH_CALUDE_sulfuric_acid_used_l3375_337525

/-- Represents the stoichiometric coefficients of the chemical reaction --/
structure Reaction :=
  (iron : ℚ)
  (sulfuric_acid : ℚ)
  (hydrogen : ℚ)

/-- The balanced chemical equation for the reaction --/
def balanced_reaction : Reaction :=
  { iron := 1
  , sulfuric_acid := 1
  , hydrogen := 1 }

/-- Theorem stating the relationship between reactants and products --/
theorem sulfuric_acid_used
  (iron_used : ℚ)
  (hydrogen_produced : ℚ)
  (h_iron : iron_used = 2)
  (h_hydrogen : hydrogen_produced = 2) :
  iron_used * balanced_reaction.sulfuric_acid / balanced_reaction.iron = 2 :=
by sorry

end NUMINAMATH_CALUDE_sulfuric_acid_used_l3375_337525


namespace NUMINAMATH_CALUDE_cats_after_sale_l3375_337558

/-- The number of cats remaining after a sale at a pet store -/
theorem cats_after_sale 
  (siamese : ℕ) -- Initial number of Siamese cats
  (house : ℕ) -- Initial number of house cats
  (sold : ℕ) -- Number of cats sold during the sale
  (h1 : siamese = 12)
  (h2 : house = 20)
  (h3 : sold = 20) :
  siamese + house - sold = 12 := by
  sorry

end NUMINAMATH_CALUDE_cats_after_sale_l3375_337558


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l3375_337531

theorem min_value_fraction_sum (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  x^2 / (y^2 - 1) + y^2 / (x^2 - 1) ≥ 4 ∧
  (x^2 / (y^2 - 1) + y^2 / (x^2 - 1) = 4 ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l3375_337531


namespace NUMINAMATH_CALUDE_lcm_15_18_20_l3375_337550

theorem lcm_15_18_20 : Nat.lcm (Nat.lcm 15 18) 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_15_18_20_l3375_337550


namespace NUMINAMATH_CALUDE_min_distance_to_line_l3375_337514

theorem min_distance_to_line : 
  ∀ m n : ℝ, 
  (4 * m - 3 * n - 5 * Real.sqrt 2 = 0) → 
  (∀ x y : ℝ, 4 * x - 3 * y - 5 * Real.sqrt 2 = 0 → m^2 + n^2 ≤ x^2 + y^2) → 
  m^2 + n^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l3375_337514


namespace NUMINAMATH_CALUDE_probability_ratio_equals_ways_ratio_l3375_337533

/-- The number of balls --/
def n : ℕ := 25

/-- The number of bins --/
def m : ℕ := 6

/-- The number of ways to distribute n balls into m bins --/
def total_distributions : ℕ := Nat.choose (n + m - 1) n

/-- The number of ways to distribute balls according to the 5-5-3-3-2-2 pattern --/
def ways_p : ℕ := Nat.choose n 5 * Nat.choose 20 5 * Nat.choose 15 3 * Nat.choose 12 3 * Nat.choose 9 2 * Nat.choose 7 2

/-- The number of ways to distribute balls equally (4-4-4-4-4-5 pattern) --/
def ways_q : ℕ := Nat.choose n 4 * Nat.choose 21 4 * Nat.choose 17 4 * Nat.choose 13 4 * Nat.choose 9 4 * Nat.choose 5 5

/-- The probability of the 5-5-3-3-2-2 distribution --/
def p : ℚ := ways_p / total_distributions

/-- The probability of the equal distribution --/
def q : ℚ := ways_q / total_distributions

theorem probability_ratio_equals_ways_ratio : p / q = ways_p / ways_q := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_equals_ways_ratio_l3375_337533


namespace NUMINAMATH_CALUDE_consecutive_composite_numbers_l3375_337598

theorem consecutive_composite_numbers (k k' : ℕ) :
  (∀ i ∈ Finset.range 7, ¬ Nat.Prime (210 * k + 1 + i + 1)) ∧
  (∀ i ∈ Finset.range 15, ¬ Nat.Prime (30030 * k' + 1 + i + 1)) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_composite_numbers_l3375_337598


namespace NUMINAMATH_CALUDE_hexagon_area_theorem_l3375_337515

-- Define the right triangle
structure RightTriangle where
  a : ℝ  -- One leg
  b : ℝ  -- Other leg
  c : ℝ  -- Hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem

-- Define the hexagon area function
def hexagon_area (t : RightTriangle) : ℝ :=
  t.a^2 + (t.a + t.b)^2

-- Theorem statement
theorem hexagon_area_theorem (t : RightTriangle) :
  hexagon_area t = t.a^2 + (t.a + t.b)^2 := by
  sorry

#check hexagon_area_theorem

end NUMINAMATH_CALUDE_hexagon_area_theorem_l3375_337515


namespace NUMINAMATH_CALUDE_square_root_sum_l3375_337579

theorem square_root_sum (x : ℝ) : 
  (Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4) → 
  (Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) + Real.sqrt (16 - x^2) = 9) := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_l3375_337579


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3375_337563

theorem solve_exponential_equation :
  ∃ y : ℕ, (8 : ℝ)^4 = 2^y ∧ y = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3375_337563


namespace NUMINAMATH_CALUDE_increase_dimension_theorem_l3375_337575

/-- Represents a rectangle with length and width --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: If increasing both length and width of a rectangle by x feet
    increases its perimeter by 16 feet, then x must be 4 feet --/
theorem increase_dimension_theorem (r : Rectangle) (x : ℝ) :
  perimeter { length := r.length + x, width := r.width + x } - perimeter r = 16 →
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_increase_dimension_theorem_l3375_337575


namespace NUMINAMATH_CALUDE_investment_problem_solution_l3375_337526

/-- Investment problem with two partners -/
structure InvestmentProblem where
  /-- Ratio of investments for partners p and q -/
  investmentRatio : Rat × Rat
  /-- Ratio of profits for partners p and q -/
  profitRatio : Rat × Rat
  /-- Investment period for partner q in months -/
  qPeriod : ℕ

/-- Solution to the investment problem -/
def solveProblem (prob : InvestmentProblem) : ℚ :=
  let (pInvest, qInvest) := prob.investmentRatio
  let (pProfit, qProfit) := prob.profitRatio
  (qProfit * pInvest * prob.qPeriod) / (pProfit * qInvest)

/-- Theorem stating the solution to the specific problem -/
theorem investment_problem_solution :
  let prob : InvestmentProblem := {
    investmentRatio := (7, 5)
    profitRatio := (7, 10)
    qPeriod := 4
  }
  solveProblem prob = 2 := by sorry


end NUMINAMATH_CALUDE_investment_problem_solution_l3375_337526


namespace NUMINAMATH_CALUDE_deductive_reasoning_form_not_sufficient_l3375_337516

/-- A structure representing a deductive argument --/
structure DeductiveArgument where
  premises : List Prop
  conclusion : Prop
  form_correct : Bool

/-- A predicate that determines if a deductive argument is valid --/
def is_valid (arg : DeductiveArgument) : Prop :=
  arg.form_correct ∧ (∀ p ∈ arg.premises, p) → arg.conclusion

/-- Theorem stating that conforming to the form of deductive reasoning alone
    is not sufficient to guarantee the correctness of the conclusion --/
theorem deductive_reasoning_form_not_sufficient :
  ∃ (arg : DeductiveArgument), arg.form_correct ∧ ¬arg.conclusion :=
sorry

end NUMINAMATH_CALUDE_deductive_reasoning_form_not_sufficient_l3375_337516


namespace NUMINAMATH_CALUDE_jason_clothing_expenses_l3375_337564

/-- The cost of Jason's shorts in dollars -/
def shorts_cost : ℝ := 14.28

/-- The cost of Jason's jacket in dollars -/
def jacket_cost : ℝ := 4.74

/-- The total amount Jason spent on clothing -/
def total_spent : ℝ := shorts_cost + jacket_cost

/-- Theorem stating that the total amount Jason spent on clothing is $19.02 -/
theorem jason_clothing_expenses : total_spent = 19.02 := by
  sorry

end NUMINAMATH_CALUDE_jason_clothing_expenses_l3375_337564


namespace NUMINAMATH_CALUDE_additional_amount_proof_l3375_337505

theorem additional_amount_proof (n : ℕ) (h : n = 3) : 7 * n - 3 * n = 12 := by
  sorry

end NUMINAMATH_CALUDE_additional_amount_proof_l3375_337505


namespace NUMINAMATH_CALUDE_greater_number_with_hcf_and_product_l3375_337562

theorem greater_number_with_hcf_and_product 
  (A B : ℕ+) 
  (hcf_condition : Nat.gcd A B = 11)
  (product_condition : A * B = 363) :
  max A B = 33 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_with_hcf_and_product_l3375_337562


namespace NUMINAMATH_CALUDE_kids_difference_l3375_337511

def kids_monday : ℕ := 6
def kids_wednesday : ℕ := 4

theorem kids_difference : kids_monday - kids_wednesday = 2 := by
  sorry

end NUMINAMATH_CALUDE_kids_difference_l3375_337511


namespace NUMINAMATH_CALUDE_monday_messages_l3375_337594

/-- Proves that given the specified message sending pattern and average, 
    the number of messages sent on Monday must be 220. -/
theorem monday_messages (x : ℝ) : 
  (x + x/2 + 50 + 50 + 50) / 5 = 96 → x = 220 := by
  sorry

end NUMINAMATH_CALUDE_monday_messages_l3375_337594


namespace NUMINAMATH_CALUDE_triangle_properties_l3375_337506

theorem triangle_properties (a b c A B C : Real) :
  -- Triangle conditions
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  -- Given conditions
  π/2 < A ∧ -- A is obtuse
  a * Real.sin B = b * Real.cos B ∧
  C = π/6 →
  -- Conclusions
  A = 2*π/3 ∧
  1 < Real.cos A + Real.cos B + Real.cos C ∧
  Real.cos A + Real.cos B + Real.cos C ≤ 5/4 := by
sorry


end NUMINAMATH_CALUDE_triangle_properties_l3375_337506


namespace NUMINAMATH_CALUDE_choir_robe_expenditure_is_36_l3375_337554

/-- Calculates the total expenditure for additional choir robes. -/
def choir_robe_expenditure (total_singers : ℕ) (existing_robes : ℕ) (cost_per_robe : ℕ) : ℕ :=
  (total_singers - existing_robes) * cost_per_robe

/-- Proves that the expenditure for additional choir robes is $36 given the specified conditions. -/
theorem choir_robe_expenditure_is_36 :
  choir_robe_expenditure 30 12 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_choir_robe_expenditure_is_36_l3375_337554


namespace NUMINAMATH_CALUDE_probability_A_wins_after_four_games_l3375_337508

def probability_A_wins : ℚ := 3 / 5
def probability_B_wins : ℚ := 2 / 5
def number_of_games : ℕ := 4
def number_of_wins_needed : ℕ := 3

theorem probability_A_wins_after_four_games :
  (Nat.choose number_of_games number_of_wins_needed : ℚ) * 
  probability_A_wins ^ number_of_wins_needed * 
  probability_B_wins ^ (number_of_games - number_of_wins_needed) =
  (Nat.choose number_of_games number_of_wins_needed : ℚ) * 
  (3 / 5) ^ 3 * (2 / 5) := by
  sorry

end NUMINAMATH_CALUDE_probability_A_wins_after_four_games_l3375_337508


namespace NUMINAMATH_CALUDE_girls_more_likely_separated_l3375_337530

/-- The probability of two girls being separated when randomly seated among three boys on a 5-seat bench is greater than the probability of them sitting together. -/
theorem girls_more_likely_separated (n : ℕ) (h : n = 5) :
  let total_arrangements := Nat.choose n 2
  let adjacent_arrangements := n - 1
  (total_arrangements - adjacent_arrangements : ℚ) / total_arrangements > adjacent_arrangements / total_arrangements :=
by
  sorry

end NUMINAMATH_CALUDE_girls_more_likely_separated_l3375_337530


namespace NUMINAMATH_CALUDE_irreducible_fraction_l3375_337504

theorem irreducible_fraction (n : ℕ+) : 
  (Nat.gcd (3 * n + 1) (5 * n + 2) = 1) := by sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l3375_337504


namespace NUMINAMATH_CALUDE_area_enclosed_by_curve_and_x_axis_l3375_337520

-- Define the curve function
def f (x : ℝ) : ℝ := 3 - 3 * x^2

-- Theorem statement
theorem area_enclosed_by_curve_and_x_axis : 
  ∫ x in (-1)..1, f x = 4 := by
  sorry

end NUMINAMATH_CALUDE_area_enclosed_by_curve_and_x_axis_l3375_337520


namespace NUMINAMATH_CALUDE_at_least_two_first_grade_products_l3375_337501

theorem at_least_two_first_grade_products (total : Nat) (first_grade : Nat) (second_grade : Nat) (third_grade : Nat) (drawn : Nat) 
  (h1 : total = 9)
  (h2 : first_grade = 4)
  (h3 : second_grade = 3)
  (h4 : third_grade = 2)
  (h5 : drawn = 4)
  (h6 : total = first_grade + second_grade + third_grade) :
  (Nat.choose total drawn) - (Nat.choose (second_grade + third_grade) drawn) - 
  (Nat.choose first_grade 1 * Nat.choose (second_grade + third_grade) (drawn - 1)) = 81 := by
sorry

end NUMINAMATH_CALUDE_at_least_two_first_grade_products_l3375_337501


namespace NUMINAMATH_CALUDE_octal_subtraction_correct_l3375_337510

/-- Represents a number in base 8 -/
def OctalNum := Nat

/-- Addition in base 8 -/
def octal_add (a b : OctalNum) : OctalNum :=
  sorry

/-- Subtraction in base 8 -/
def octal_sub (a b : OctalNum) : OctalNum :=
  sorry

/-- Conversion from decimal to octal -/
def to_octal (n : Nat) : OctalNum :=
  sorry

theorem octal_subtraction_correct :
  let a : OctalNum := to_octal 537
  let b : OctalNum := to_octal 261
  let c : OctalNum := to_octal 256
  octal_sub a b = c ∧ octal_add b c = a := by
  sorry

end NUMINAMATH_CALUDE_octal_subtraction_correct_l3375_337510


namespace NUMINAMATH_CALUDE_y_in_terms_of_abc_l3375_337500

theorem y_in_terms_of_abc (x y z a b c : ℝ) 
  (h1 : x ≠ y) (h2 : x ≠ z) (h3 : y ≠ z) 
  (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0)
  (eq1 : x * y / (x - y) = a)
  (eq2 : x * z / (x - z) = b)
  (eq3 : y * z / (y - z) = c) :
  y = b * c * x / ((b + c) * x - b * c) := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_abc_l3375_337500


namespace NUMINAMATH_CALUDE_motion_equation_l3375_337561

/-- The acceleration function -/
def a (t : ℝ) : ℝ := 6 * t - 2

/-- The velocity function -/
def v (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 1

/-- The position function -/
def s (t : ℝ) : ℝ := t^3 - t^2 + t

theorem motion_equation (t : ℝ) :
  (∀ t, deriv v t = a t) ∧
  (∀ t, deriv s t = v t) ∧
  v 0 = 1 ∧
  s 0 = 0 →
  s t = t^3 - t^2 + t :=
by
  sorry

end NUMINAMATH_CALUDE_motion_equation_l3375_337561


namespace NUMINAMATH_CALUDE_fundraising_event_boys_l3375_337532

theorem fundraising_event_boys (p : ℕ) : 
  (∃ (initial_boys : ℕ),
    -- Initially, 35% of participants are boys
    initial_boys = (35 * p) / 100 ∧
    -- After changes, boys make up 30% of the group
    (initial_boys - 3 + 2) * 100 = 30 * (p + 3) ∧
    -- The initial number of boys is 13
    initial_boys = 13) :=
by
  sorry

end NUMINAMATH_CALUDE_fundraising_event_boys_l3375_337532


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_set_l3375_337518

theorem absolute_value_equation_solution_set :
  ∀ x : ℝ, abs x + abs (x + 1) = 1 ↔ x ∈ Set.Icc (-1) 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_set_l3375_337518


namespace NUMINAMATH_CALUDE_braking_distance_at_120_less_than_33_braking_distance_at_40_equals_10_braking_distance_linear_and_nonnegative_l3375_337541

/-- Represents the braking distance function for a car -/
def brakingDistance (speed : ℝ) : ℝ :=
  0.25 * speed

/-- Theorem: The braking distance at 120 km/h is less than 33m -/
theorem braking_distance_at_120_less_than_33 :
  brakingDistance 120 < 33 := by
  sorry

/-- Theorem: The braking distance at 40 km/h is 10m -/
theorem braking_distance_at_40_equals_10 :
  brakingDistance 40 = 10 := by
  sorry

/-- Theorem: The braking distance function is linear and non-negative for non-negative speeds -/
theorem braking_distance_linear_and_nonnegative :
  ∀ (speed : ℝ), speed ≥ 0 → brakingDistance speed ≥ 0 ∧ 
  ∀ (speed1 speed2 : ℝ), brakingDistance (speed1 + speed2) = brakingDistance speed1 + brakingDistance speed2 := by
  sorry

end NUMINAMATH_CALUDE_braking_distance_at_120_less_than_33_braking_distance_at_40_equals_10_braking_distance_linear_and_nonnegative_l3375_337541


namespace NUMINAMATH_CALUDE_quadratic_roots_exist_sum_minus_product_equals_two_l3375_337587

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 3*x + 1 = 0

-- Define the roots
theorem quadratic_roots_exist : ∃ (x₁ x₂ : ℝ), quadratic_equation x₁ ∧ quadratic_equation x₂ ∧ x₁ ≠ x₂ :=
sorry

-- Theorem to prove
theorem sum_minus_product_equals_two :
  ∃ (x₁ x₂ : ℝ), quadratic_equation x₁ ∧ quadratic_equation x₂ ∧ x₁ ≠ x₂ ∧ x₁ + x₂ - x₁*x₂ = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_exist_sum_minus_product_equals_two_l3375_337587


namespace NUMINAMATH_CALUDE_triangle_side_difference_is_12_l3375_337512

def triangle_side_difference (y : ℤ) : Prop :=
  ∃ (a b : ℤ), 
    a = 7 ∧ b = 9 ∧  -- Given side lengths
    y > |a - b| ∧    -- Triangle inequality lower bound
    y < a + b ∧      -- Triangle inequality upper bound
    y ≥ 3 ∧ y ≤ 15   -- Integral bounds for y

theorem triangle_side_difference_is_12 : 
  (∀ y : ℤ, triangle_side_difference y → y ≤ 15) ∧ 
  (∀ y : ℤ, triangle_side_difference y → y ≥ 3) ∧
  (15 - 3 = 12) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_difference_is_12_l3375_337512


namespace NUMINAMATH_CALUDE_largest_quantity_l3375_337566

theorem largest_quantity : 
  let A := (2010 : ℚ) / 2009 + 2010 / 2011
  let B := (2010 : ℚ) / 2011 + 2012 / 2011
  let C := (2011 : ℚ) / 2010 + 2011 / 2012
  A > B ∧ A > C := by sorry

end NUMINAMATH_CALUDE_largest_quantity_l3375_337566


namespace NUMINAMATH_CALUDE_triangle_area_l3375_337585

theorem triangle_area (a b : ℝ) (θ : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : θ = π / 3) :
  (1 / 2) * a * b * Real.sin θ = (3 / 2) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3375_337585


namespace NUMINAMATH_CALUDE_union_A_B_when_m_neg_two_intersection_A_B_equals_B_iff_l3375_337553

-- Define sets A and B
def A : Set ℝ := {x | (x - 2) / (x + 1) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 2 * m + 3 < x ∧ x < m^2}

-- Theorem for part 1
theorem union_A_B_when_m_neg_two :
  A ∪ B (-2) = {x | -1 < x ∧ x < 4} := by sorry

-- Theorem for part 2
theorem intersection_A_B_equals_B_iff (m : ℝ) :
  A ∩ B m = B m ↔ m ∈ Set.Icc (-Real.sqrt 2) 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_m_neg_two_intersection_A_B_equals_B_iff_l3375_337553


namespace NUMINAMATH_CALUDE_jack_combinations_eq_44_l3375_337502

/-- The number of ways to distribute n indistinguishable objects into k distinct boxes,
    with each box containing at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of combinations of rolls Jack could purchase. -/
def jack_combinations : ℕ := distribute 10 4

theorem jack_combinations_eq_44 : jack_combinations = 44 := by sorry

end NUMINAMATH_CALUDE_jack_combinations_eq_44_l3375_337502


namespace NUMINAMATH_CALUDE_polygon_area_is_7_5_l3375_337584

/-- Calculates the area of a polygon using the Shoelace formula -/
def polygonArea (vertices : List (ℝ × ℝ)) : ℝ :=
  let n := vertices.length
  let pairs := List.zip vertices (vertices.rotate 1)
  0.5 * (pairs.foldl (fun sum (v1, v2) => sum + v1.1 * v2.2 - v1.2 * v2.1) 0)

theorem polygon_area_is_7_5 :
  let vertices := [(2, 1), (4, 3), (7, 1), (4, 6)]
  polygonArea vertices = 7.5 := by
  sorry

#eval polygonArea [(2, 1), (4, 3), (7, 1), (4, 6)]

end NUMINAMATH_CALUDE_polygon_area_is_7_5_l3375_337584


namespace NUMINAMATH_CALUDE_cycling_distance_l3375_337591

theorem cycling_distance (rate : ℝ) (time : ℝ) (distance : ℝ) : 
  rate = 8 → time = 2.25 → distance = rate * time → distance = 18 := by
sorry

end NUMINAMATH_CALUDE_cycling_distance_l3375_337591


namespace NUMINAMATH_CALUDE_parabola_y_values_l3375_337596

def f (x : ℝ) := -(x - 2)^2

theorem parabola_y_values :
  ∀ (y₁ y₂ y₃ : ℝ),
  f (-1) = y₁ → f 1 = y₂ → f 4 = y₃ →
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_values_l3375_337596


namespace NUMINAMATH_CALUDE_radius_C1_value_l3375_337528

-- Define the points and circles
variable (O X Y Z : ℝ × ℝ)
variable (C1 C2 : Set (ℝ × ℝ))

-- Define the conditions
axiom inside_C2 : C1 ⊆ C2
axiom intersect : X ∈ C1 ∩ C2 ∧ Y ∈ C1 ∩ C2
axiom Z_position : Z ∉ C1 ∧ Z ∈ C2
axiom XZ_length : dist X Z = 15
axiom OZ_length : dist O Z = 5
axiom YZ_length : dist Y Z = 12

-- Define the radius of C1
def radius_C1 : ℝ := dist O X

-- Theorem to prove
theorem radius_C1_value : radius_C1 O X = 5 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_radius_C1_value_l3375_337528


namespace NUMINAMATH_CALUDE_average_playing_time_l3375_337574

/-- Calculates the average playing time given the hours played on three days,
    where the third day is 3 hours more than each of the first two days. -/
theorem average_playing_time (hours_day1 hours_day2 : ℕ) 
    (h1 : hours_day1 = hours_day2)
    (h2 : hours_day1 > 0) : 
  (hours_day1 + hours_day2 + (hours_day1 + 3)) / 3 = hours_day1 + 1 :=
by sorry

#check average_playing_time

end NUMINAMATH_CALUDE_average_playing_time_l3375_337574


namespace NUMINAMATH_CALUDE_angle_ABC_less_than_60_degrees_l3375_337580

/-- A triangle with vertices A, B, and C -/
structure Triangle (V : Type*) where
  A : V
  B : V
  C : V

/-- The angle at vertex B in a triangle -/
def angle_at_B {V : Type*} (t : Triangle V) : ℝ := sorry

/-- The altitude from vertex A in a triangle -/
def altitude_from_A {V : Type*} (t : Triangle V) : ℝ := sorry

/-- The median from vertex B in a triangle -/
def median_from_B {V : Type*} (t : Triangle V) : ℝ := sorry

/-- Predicate to check if a triangle is acute-angled -/
def is_acute_angled {V : Type*} (t : Triangle V) : Prop := sorry

/-- Predicate to check if the altitude from A is the longest -/
def altitude_A_is_longest {V : Type*} (t : Triangle V) : Prop := sorry

theorem angle_ABC_less_than_60_degrees {V : Type*} (t : Triangle V) :
  is_acute_angled t →
  altitude_A_is_longest t →
  altitude_from_A t = median_from_B t →
  angle_at_B t < 60 := by sorry

end NUMINAMATH_CALUDE_angle_ABC_less_than_60_degrees_l3375_337580


namespace NUMINAMATH_CALUDE_domino_tiling_theorem_l3375_337567

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a domino placement on a board -/
def Tiling (b : Board) := Set (ℕ × ℕ × Bool)

/-- Checks if a tiling is valid for a given board -/
def is_valid_tiling (b : Board) (t : Tiling b) : Prop := sorry

/-- Checks if a line bisects at least one domino in the tiling -/
def line_bisects_domino (b : Board) (t : Tiling b) (line : ℕ × Bool) : Prop := sorry

/-- Counts the number of internal lines in a board -/
def internal_lines_count (b : Board) : ℕ := 
  b.rows + b.cols - 2

/-- Main theorem statement -/
theorem domino_tiling_theorem :
  (¬ ∃ (t : Tiling ⟨6, 6⟩), 
    is_valid_tiling ⟨6, 6⟩ t ∧ 
    ∀ (line : ℕ × Bool), line_bisects_domino ⟨6, 6⟩ t line) ∧
  (∃ (t : Tiling ⟨5, 6⟩), 
    is_valid_tiling ⟨5, 6⟩ t ∧ 
    ∀ (line : ℕ × Bool), line_bisects_domino ⟨5, 6⟩ t line) :=
sorry

end NUMINAMATH_CALUDE_domino_tiling_theorem_l3375_337567


namespace NUMINAMATH_CALUDE_money_sum_is_fifty_l3375_337586

def jack_money : ℕ := 26

def ben_money (jack : ℕ) : ℕ := jack - 9

def eric_money (ben : ℕ) : ℕ := ben - 10

def total_money (jack ben eric : ℕ) : ℕ := jack + ben + eric

theorem money_sum_is_fifty :
  total_money jack_money (ben_money jack_money) (eric_money (ben_money jack_money)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_money_sum_is_fifty_l3375_337586


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3375_337571

-- Define the arithmetic sequence a_n
def a (n : ℕ+) : ℚ :=
  sorry

-- Define the sum S_n of the first n terms of a_n
def S (n : ℕ+) : ℚ :=
  sorry

-- Define the sequence b_n
def b (n : ℕ+) : ℚ :=
  1 / (a n ^ 2 - 1)

-- Define the sum T_n of the first n terms of b_n
def T (n : ℕ+) : ℚ :=
  sorry

theorem arithmetic_sequence_properties :
  (a 3 = 6) ∧
  (a 5 + a 7 = 24) ∧
  (∀ n : ℕ+, a n = 2 * n) ∧
  (∀ n : ℕ+, S n = n^2 + n) ∧
  (∀ n : ℕ+, T n = n / (2 * n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3375_337571


namespace NUMINAMATH_CALUDE_smallest_integer_for_negative_quadratic_l3375_337577

theorem smallest_integer_for_negative_quadratic : 
  ∃ (x : ℤ), (∀ (y : ℤ), y^2 - 11*y + 24 < 0 → x ≤ y) ∧ (x^2 - 11*x + 24 < 0) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_for_negative_quadratic_l3375_337577


namespace NUMINAMATH_CALUDE_gcd_consecutive_triple_product_l3375_337549

theorem gcd_consecutive_triple_product (i : ℕ) (h : i ≥ 1) :
  ∃ (g : ℕ), g = Nat.gcd i ((i + 1) * (i + 2)) ∧ g = 6 :=
sorry

end NUMINAMATH_CALUDE_gcd_consecutive_triple_product_l3375_337549


namespace NUMINAMATH_CALUDE_banana_box_cost_l3375_337599

/-- Calculates the total cost of bananas after discount -/
def totalCostAfterDiscount (
  bunches8 : ℕ)  -- Number of bunches with 8 bananas
  (price8 : ℚ)   -- Price of each bunch with 8 bananas
  (bunches7 : ℕ)  -- Number of bunches with 7 bananas
  (price7 : ℚ)   -- Price of each bunch with 7 bananas
  (discount : ℚ)  -- Discount as a decimal
  : ℚ :=
  let totalCost := bunches8 * price8 + bunches7 * price7
  totalCost * (1 - discount)

/-- Proves that the total cost after discount for the given conditions is $23.40 -/
theorem banana_box_cost :
  totalCostAfterDiscount 6 2.5 5 2.2 0.1 = 23.4 := by
  sorry

end NUMINAMATH_CALUDE_banana_box_cost_l3375_337599


namespace NUMINAMATH_CALUDE_museum_trip_total_people_l3375_337583

theorem museum_trip_total_people : 
  let first_bus : ℕ := 12
  let second_bus : ℕ := 2 * first_bus
  let third_bus : ℕ := second_bus - 6
  let fourth_bus : ℕ := first_bus + 9
  first_bus + second_bus + third_bus + fourth_bus = 75
  := by sorry

end NUMINAMATH_CALUDE_museum_trip_total_people_l3375_337583


namespace NUMINAMATH_CALUDE_mother_notebooks_l3375_337559

/-- The number of notebooks the mother initially had -/
def initial_notebooks : ℕ := sorry

/-- The number of children -/
def num_children : ℕ := sorry

/-- If each child gets 13 notebooks, the mother has 8 notebooks left -/
axiom condition1 : initial_notebooks = 13 * num_children + 8

/-- If each child gets 15 notebooks, all notebooks are distributed -/
axiom condition2 : initial_notebooks = 15 * num_children

theorem mother_notebooks : initial_notebooks = 60 := by sorry

end NUMINAMATH_CALUDE_mother_notebooks_l3375_337559


namespace NUMINAMATH_CALUDE_scaled_vector_is_monomial_l3375_337507

/-- A vector in ℝ² -/
def vector : ℝ × ℝ := (1, 5)

/-- The scalar multiple of the vector -/
def scaled_vector : ℝ × ℝ := (-3 * vector.1, -3 * vector.2)

/-- Definition of a monomial in this context -/
def is_monomial (v : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ) (n : ℕ × ℕ), v = (c * n.1, c * n.2)

theorem scaled_vector_is_monomial : is_monomial scaled_vector := by
  sorry

end NUMINAMATH_CALUDE_scaled_vector_is_monomial_l3375_337507


namespace NUMINAMATH_CALUDE_compare_fractions_l3375_337582

theorem compare_fractions : -4/3 < -5/4 := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_l3375_337582


namespace NUMINAMATH_CALUDE_bus_riders_l3375_337548

theorem bus_riders (initial_riders : ℕ) : 
  (initial_riders + 40 - 60 = 2) → initial_riders = 22 := by
  sorry

end NUMINAMATH_CALUDE_bus_riders_l3375_337548


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l3375_337552

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 58)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) -- Ensures non-throwers can be divided into thirds
  : (throwers + (2 * (total_players - throwers) / 3)) = 51 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l3375_337552


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l3375_337519

-- Define the polynomials
def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

-- Theorem statement
theorem sum_of_polynomials (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l3375_337519


namespace NUMINAMATH_CALUDE_min_values_l3375_337557

def min_value_exponential (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + 2*y = 1 → 2^x + 4^y ≥ 2*Real.sqrt 2

def min_value_reciprocal (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + 2*y = 1 → 1/x + 2/y ≥ 9

def min_value_squared (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + 2*y = 1 → x^2 + 4*y^2 ≥ 1/2

theorem min_values (x y : ℝ) :
  min_value_exponential x y ∧
  min_value_reciprocal x y ∧
  min_value_squared x y :=
sorry

end NUMINAMATH_CALUDE_min_values_l3375_337557


namespace NUMINAMATH_CALUDE_employee_pay_calculation_l3375_337570

/-- Given two employees with a total weekly pay of 560, where one employee's pay is 150% of the other's, prove that the employee with the lower pay receives 224 per week. -/
theorem employee_pay_calculation (total_pay : ℝ) (a_pay b_pay : ℝ) : 
  total_pay = 560 →
  a_pay = 1.5 * b_pay →
  a_pay + b_pay = total_pay →
  b_pay = 224 := by sorry

end NUMINAMATH_CALUDE_employee_pay_calculation_l3375_337570


namespace NUMINAMATH_CALUDE_pens_to_sell_for_profit_l3375_337592

theorem pens_to_sell_for_profit (total_pens : ℕ) (cost_per_pen sell_price : ℚ) (desired_profit : ℚ) :
  total_pens = 2000 →
  cost_per_pen = 15/100 →
  sell_price = 30/100 →
  desired_profit = 150 →
  ∃ (pens_to_sell : ℕ), 
    pens_to_sell * sell_price - total_pens * cost_per_pen = desired_profit ∧
    pens_to_sell = 1500 :=
by sorry

end NUMINAMATH_CALUDE_pens_to_sell_for_profit_l3375_337592


namespace NUMINAMATH_CALUDE_principal_calculation_l3375_337521

/-- Proves that given specific conditions, the principal amount is 1400 --/
theorem principal_calculation (rate : ℝ) (time : ℝ) (amount : ℝ) :
  rate = 0.05 →
  time = 2.4 →
  amount = 1568 →
  (∃ (principal : ℝ), principal * (1 + rate * time) = amount ∧ principal = 1400) :=
by sorry

end NUMINAMATH_CALUDE_principal_calculation_l3375_337521


namespace NUMINAMATH_CALUDE_school_boys_count_l3375_337542

theorem school_boys_count (boys girls : ℕ) : 
  (boys : ℚ) / girls = 5 / 13 →
  girls = boys + 128 →
  boys = 80 := by
sorry

end NUMINAMATH_CALUDE_school_boys_count_l3375_337542


namespace NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l3375_337565

/-- The line l: ax + by + c = 0 does not pass through the fourth quadrant when ab < 0 and bc < 0 -/
theorem line_not_in_fourth_quadrant (a b c : ℝ) (h1 : a * b < 0) (h2 : b * c < 0) :
  ∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ a * x + b * y + c ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l3375_337565


namespace NUMINAMATH_CALUDE_elect_representatives_l3375_337534

theorem elect_representatives (total_students : ℕ) (girls : ℕ) (representatives : ℕ) 
  (h1 : total_students = 10) 
  (h2 : girls = 3) 
  (h3 : representatives = 2) : 
  (Nat.choose total_students representatives - Nat.choose (total_students - girls) representatives) = 48 :=
sorry

end NUMINAMATH_CALUDE_elect_representatives_l3375_337534


namespace NUMINAMATH_CALUDE_total_volume_of_cubes_l3375_337544

-- Define the number and side length of Carl's cubes
def carl_cubes : ℕ := 4
def carl_side_length : ℕ := 3

-- Define the number and side length of Kate's cubes
def kate_cubes : ℕ := 6
def kate_side_length : ℕ := 1

-- Function to calculate the volume of a cube
def cube_volume (side_length : ℕ) : ℕ := side_length ^ 3

-- Theorem statement
theorem total_volume_of_cubes :
  carl_cubes * cube_volume carl_side_length + kate_cubes * cube_volume kate_side_length = 114 := by
  sorry


end NUMINAMATH_CALUDE_total_volume_of_cubes_l3375_337544


namespace NUMINAMATH_CALUDE_projection_a_onto_b_l3375_337536

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (-2, 4)

theorem projection_a_onto_b :
  let proj := (a.1 * b.1 + a.2 * b.2) / Real.sqrt ((b.1 ^ 2 + b.2 ^ 2))
  proj = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_projection_a_onto_b_l3375_337536


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l3375_337513

theorem point_in_third_quadrant :
  let angle : ℝ := 2007 * Real.pi / 180
  (Real.cos angle < 0) ∧ (Real.sin angle < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l3375_337513


namespace NUMINAMATH_CALUDE_angle_A_measure_triangle_area_l3375_337589

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def triangle_condition (t : Triangle) : Prop :=
  (t.a - t.b) / t.c = (Real.sin t.B + Real.sin t.C) / (Real.sin t.B + Real.sin t.A)

-- Theorem 1: Prove that angle A measures 2π/3
theorem angle_A_measure (t : Triangle) (h : triangle_condition t) : t.A = 2 * Real.pi / 3 := by
  sorry

-- Theorem 2: Prove the area of the triangle when a = √7 and b = 2c
theorem triangle_area (t : Triangle) (h1 : triangle_condition t) (h2 : t.a = Real.sqrt 7) (h3 : t.b = 2 * t.c) :
  (1 / 2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_measure_triangle_area_l3375_337589


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l3375_337547

theorem sqrt_of_sqrt_81 : ∃ (x : ℝ), x^2 = Real.sqrt 81 ↔ x = 3 ∨ x = -3 := by sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_81_l3375_337547


namespace NUMINAMATH_CALUDE_venus_hall_rental_cost_prove_venus_hall_rental_cost_l3375_337569

/-- The rental cost of Venus Hall, given the conditions of the prom venue problem -/
theorem venus_hall_rental_cost : ℝ → Prop :=
  fun v =>
    let caesars_total : ℝ := 800 + 60 * 30
    let venus_total : ℝ := v + 60 * 35
    caesars_total = venus_total →
    v = 500

/-- Proof of the venus_hall_rental_cost theorem -/
theorem prove_venus_hall_rental_cost : ∃ v, venus_hall_rental_cost v :=
  sorry

end NUMINAMATH_CALUDE_venus_hall_rental_cost_prove_venus_hall_rental_cost_l3375_337569


namespace NUMINAMATH_CALUDE_carnival_days_l3375_337535

theorem carnival_days (daily_income total_income : ℕ) 
  (h1 : daily_income = 144)
  (h2 : total_income = 3168) :
  total_income / daily_income = 22 := by
  sorry

end NUMINAMATH_CALUDE_carnival_days_l3375_337535


namespace NUMINAMATH_CALUDE_expand_product_l3375_337538

theorem expand_product (x : ℝ) : 3 * (x^2 - 5*x + 6) * (x^2 + 8*x - 10) = 3*x^4 + 9*x^3 - 132*x^2 + 294*x - 180 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3375_337538


namespace NUMINAMATH_CALUDE_base_8_addition_l3375_337509

/-- Addition in base 8 -/
def add_base_8 (a b : ℕ) : ℕ := 
  (a + b) % 8

/-- Conversion from base 8 to base 10 -/
def base_8_to_10 (n : ℕ) : ℕ := 
  (n / 10) * 8 + (n % 10)

theorem base_8_addition : 
  add_base_8 (base_8_to_10 5) (base_8_to_10 13) = base_8_to_10 20 := by
  sorry

end NUMINAMATH_CALUDE_base_8_addition_l3375_337509


namespace NUMINAMATH_CALUDE_largest_square_tile_size_l3375_337572

theorem largest_square_tile_size (wall_width wall_length : ℕ) 
  (hw : wall_width = 24) (hl : wall_length = 18) :
  ∃ (tile_size : ℕ), 
    tile_size > 0 ∧
    wall_width % tile_size = 0 ∧ 
    wall_length % tile_size = 0 ∧
    ∀ (other_size : ℕ), 
      (wall_width % other_size = 0 ∧ wall_length % other_size = 0) → 
      other_size ≤ tile_size :=
by
  -- The proof would go here
  sorry

#check largest_square_tile_size

end NUMINAMATH_CALUDE_largest_square_tile_size_l3375_337572


namespace NUMINAMATH_CALUDE_elrond_arwen_tulip_ratio_l3375_337545

/-- Given that Arwen picked 20 tulips and the total number of tulips picked by Arwen and Elrond is 60,
    prove that the ratio of Elrond's tulips to Arwen's tulips is 2:1 -/
theorem elrond_arwen_tulip_ratio :
  let arwen_tulips : ℕ := 20
  let total_tulips : ℕ := 60
  let elrond_tulips : ℕ := total_tulips - arwen_tulips
  (elrond_tulips : ℚ) / (arwen_tulips : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_elrond_arwen_tulip_ratio_l3375_337545


namespace NUMINAMATH_CALUDE_symmetric_lines_l3375_337595

/-- Given two lines l and k symmetric with respect to y = x, prove that if l has equation y = ax + b, then k has equation y = (1/a)x - (b/a) -/
theorem symmetric_lines (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let l := {p : ℝ × ℝ | p.2 = a * p.1 + b}
  let k := {p : ℝ × ℝ | p.2 = (1/a) * p.1 - b/a}
  let symmetry := {p : ℝ × ℝ | p.1 = p.2}
  (∀ p, p ∈ l ↔ (p.2, p.1) ∈ k) ∧ (∀ p, p ∈ k ↔ (p.2, p.1) ∈ l) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_lines_l3375_337595


namespace NUMINAMATH_CALUDE_hotel_rooms_l3375_337523

theorem hotel_rooms (total_rooms : ℕ) (single_cost double_cost : ℚ) (total_revenue : ℚ) :
  total_rooms = 260 ∧
  single_cost = 35 ∧
  double_cost = 60 ∧
  total_revenue = 14000 →
  ∃ (single_rooms double_rooms : ℕ),
    single_rooms + double_rooms = total_rooms ∧
    single_cost * single_rooms + double_cost * double_rooms = total_revenue ∧
    single_rooms = 64 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_l3375_337523


namespace NUMINAMATH_CALUDE_cube_tetrahedrons_l3375_337590

/-- A cube has 8 vertices -/
def cube_vertices : ℕ := 8

/-- Number of vertices needed to form a tetrahedron -/
def tetrahedron_vertices : ℕ := 4

/-- Number of coplanar sets in a cube (faces + diagonal planes) -/
def coplanar_sets : ℕ := 12

/-- The number of different tetrahedrons that can be formed from the vertices of a cube -/
def num_tetrahedrons : ℕ := Nat.choose cube_vertices tetrahedron_vertices - coplanar_sets

theorem cube_tetrahedrons :
  num_tetrahedrons = Nat.choose cube_vertices tetrahedron_vertices - coplanar_sets :=
sorry

end NUMINAMATH_CALUDE_cube_tetrahedrons_l3375_337590


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l3375_337517

theorem angle_in_second_quadrant (θ : Real) 
  (h1 : Real.sin θ > 0) (h2 : Real.cos θ < 0) : 
  ∃ (x y : Real), x < 0 ∧ y > 0 ∧ Real.cos θ = x ∧ Real.sin θ = y := by
  sorry

end NUMINAMATH_CALUDE_angle_in_second_quadrant_l3375_337517


namespace NUMINAMATH_CALUDE_stationery_purchase_l3375_337560

theorem stationery_purchase (brother_money sister_money : ℕ) : 
  brother_money = 2 * sister_money →
  brother_money - 180 = sister_money - 30 →
  brother_money = 300 ∧ sister_money = 150 := by
  sorry

end NUMINAMATH_CALUDE_stationery_purchase_l3375_337560


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3375_337543

def A : Set (ℝ × ℝ) := {p | 4 * p.1 + p.2 = 6}
def B : Set (ℝ × ℝ) := {p | 3 * p.1 + 2 * p.2 = 7}

theorem intersection_of_A_and_B : A ∩ B = {(1, 2)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3375_337543


namespace NUMINAMATH_CALUDE_min_occupied_seats_for_150_l3375_337529

/-- Given a row of seats, calculates the minimum number of occupied seats
    required to ensure the next person must sit next to someone. -/
def min_occupied_seats (total_seats : ℕ) : ℕ :=
  (total_seats + 2) / 4

theorem min_occupied_seats_for_150 :
  min_occupied_seats 150 = 37 := by
  sorry

#eval min_occupied_seats 150

end NUMINAMATH_CALUDE_min_occupied_seats_for_150_l3375_337529


namespace NUMINAMATH_CALUDE_transform_point_l3375_337581

/-- Rotate a point 90 degrees clockwise around a center point -/
def rotate90Clockwise (p : ℝ × ℝ) (center : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  let (cx, cy) := center
  (cx + (y - cy), cy - (x - cx))

/-- Reflect a point over the x-axis -/
def reflectOverX (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The main theorem -/
theorem transform_point :
  let A : ℝ × ℝ := (-4, 1)
  let center : ℝ × ℝ := (1, 1)
  let rotated := rotate90Clockwise A center
  let final := reflectOverX rotated
  final = (1, -6) := by sorry

end NUMINAMATH_CALUDE_transform_point_l3375_337581


namespace NUMINAMATH_CALUDE_susan_reading_hours_l3375_337527

/-- Represents the number of hours spent on each activity -/
structure ActivityHours where
  swimming : ℝ
  reading : ℝ
  friends : ℝ
  work : ℝ
  chores : ℝ

/-- The ratio of time spent on activities -/
def activity_ratio : ActivityHours :=
  { swimming := 1
    reading := 4
    friends := 10
    work := 3
    chores := 2 }

/-- Susan's actual hours spent on activities -/
def susan_hours : ActivityHours :=
  { swimming := 2
    reading := 8
    friends := 20
    work := 6
    chores := 4 }

theorem susan_reading_hours :
  (∀ (x : ℝ), x > 0 →
    susan_hours.swimming = x * activity_ratio.swimming ∧
    susan_hours.reading = x * activity_ratio.reading ∧
    susan_hours.friends = x * activity_ratio.friends ∧
    susan_hours.work = x * activity_ratio.work ∧
    susan_hours.chores = x * activity_ratio.chores) →
  susan_hours.friends = 20 →
  susan_hours.work + susan_hours.chores ≤ 35 →
  susan_hours.reading = 8 :=
by sorry

end NUMINAMATH_CALUDE_susan_reading_hours_l3375_337527


namespace NUMINAMATH_CALUDE_smaller_two_digit_factor_l3375_337597

theorem smaller_two_digit_factor (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 4851 → 
  min a b = 49 := by
sorry

end NUMINAMATH_CALUDE_smaller_two_digit_factor_l3375_337597


namespace NUMINAMATH_CALUDE_album_ratio_l3375_337522

theorem album_ratio (adele katrina bridget miriam : ℕ) 
  (h1 : ∃ s : ℕ, miriam = s * katrina)
  (h2 : katrina = 6 * bridget)
  (h3 : bridget = adele - 15)
  (h4 : adele = 30)
  (h5 : miriam + katrina + bridget + adele = 585) :
  miriam = 5 * katrina := by
sorry

end NUMINAMATH_CALUDE_album_ratio_l3375_337522


namespace NUMINAMATH_CALUDE_parabola_vertex_l3375_337524

/-- The parabola defined by y = 2(x+9)^2 - 3 has vertex at (-9, -3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 2 * (x + 9)^2 - 3 → (∃ a b : ℝ, (a, b) = (-9, -3) ∧ ∀ x, y ≥ 2 * (x + 9)^2 - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3375_337524


namespace NUMINAMATH_CALUDE_max_value_ab_l3375_337576

theorem max_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : (1 : ℝ) * (1 : ℝ) + (2 * a - 1) * (-b) = 0) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 
  (1 : ℝ) * (1 : ℝ) + (2 * x - 1) * (-y) = 0 → 
  x * y ≤ a * b ∧ a * b ≤ (1/8 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_max_value_ab_l3375_337576


namespace NUMINAMATH_CALUDE_divisibility_property_l3375_337573

theorem divisibility_property (a m n : ℕ) (ha : a > 1) (hdiv : (a^m + 1) ∣ (a^n + 1)) : m ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3375_337573


namespace NUMINAMATH_CALUDE_cos_five_pi_sixth_minus_x_l3375_337578

theorem cos_five_pi_sixth_minus_x (x : ℝ) 
  (h : Real.sin (π / 3 - x) = 3 / 5) : 
  Real.cos (5 * π / 6 - x) = -(3 / 5) := by
sorry

end NUMINAMATH_CALUDE_cos_five_pi_sixth_minus_x_l3375_337578


namespace NUMINAMATH_CALUDE_rabbit_can_cross_tracks_l3375_337539

/-- The distance from the rabbit (point A) to the railway track -/
def rabbit_distance : ℝ := 160

/-- The speed of the train -/
def train_speed : ℝ := 30

/-- The initial distance of the train from point T -/
def train_initial_distance : ℝ := 300

/-- The speed of the rabbit -/
def rabbit_speed : ℝ := 15

/-- The lower bound of the safe crossing distance -/
def lower_bound : ℝ := 23.21

/-- The upper bound of the safe crossing distance -/
def upper_bound : ℝ := 176.79

theorem rabbit_can_cross_tracks :
  ∃ x : ℝ, lower_bound < x ∧ x < upper_bound ∧
  (((rabbit_distance ^ 2 + x ^ 2).sqrt / rabbit_speed) < ((train_initial_distance + x) / train_speed)) :=
by sorry

end NUMINAMATH_CALUDE_rabbit_can_cross_tracks_l3375_337539


namespace NUMINAMATH_CALUDE_painting_time_theorem_l3375_337588

def time_for_lily : ℕ := 5
def time_for_rose : ℕ := 7
def time_for_orchid : ℕ := 3
def time_for_vine : ℕ := 2

def num_lilies : ℕ := 17
def num_roses : ℕ := 10
def num_orchids : ℕ := 6
def num_vines : ℕ := 20

def total_time : ℕ := time_for_lily * num_lilies + time_for_rose * num_roses + 
                       time_for_orchid * num_orchids + time_for_vine * num_vines

theorem painting_time_theorem : total_time = 213 := by
  sorry

end NUMINAMATH_CALUDE_painting_time_theorem_l3375_337588


namespace NUMINAMATH_CALUDE_average_games_per_month_l3375_337537

def total_games : ℕ := 323
def season_months : ℕ := 19

theorem average_games_per_month :
  (total_games : ℚ) / season_months = 17 := by sorry

end NUMINAMATH_CALUDE_average_games_per_month_l3375_337537


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l3375_337568

theorem unique_two_digit_integer (t : ℕ) : 
  (t ≥ 10 ∧ t < 100) ∧ (13 * t) % 100 = 45 ↔ t = 65 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l3375_337568


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_l3375_337546

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_l3375_337546


namespace NUMINAMATH_CALUDE_det_A_eq_31_l3375_337556

def A : Matrix (Fin 3) (Fin 3) ℝ := !![2, -4, 5; 3, 6, -2; 1, -1, 3]

theorem det_A_eq_31 : Matrix.det A = 31 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_31_l3375_337556


namespace NUMINAMATH_CALUDE_no_consecutive_triples_sum_squares_equal_repeating_digit_l3375_337551

theorem no_consecutive_triples_sum_squares_equal_repeating_digit : 
  ¬ ∃ (n a : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ (n-1)^2 + n^2 + (n+1)^2 = 1111 * a := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_triples_sum_squares_equal_repeating_digit_l3375_337551


namespace NUMINAMATH_CALUDE_equation_is_linear_l3375_337593

/-- An equation is linear with one variable if it can be written in the form ax + b = 0,
    where a and b are constants and a ≠ 0. --/
def is_linear_equation_one_var (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function representing the equation 7x + 5 = 6(x - 1) --/
def f (x : ℝ) : ℝ := 7 * x + 5 - (6 * (x - 1))

theorem equation_is_linear : is_linear_equation_one_var f := by
  sorry

end NUMINAMATH_CALUDE_equation_is_linear_l3375_337593


namespace NUMINAMATH_CALUDE_prob_A_B_together_is_two_thirds_l3375_337503

/-- The number of ways to arrange 3 students in a row -/
def total_arrangements : ℕ := 6

/-- The number of arrangements where A and B are together -/
def favorable_arrangements : ℕ := 4

/-- The probability that A and B stand together -/
def prob_A_B_together : ℚ := favorable_arrangements / total_arrangements

theorem prob_A_B_together_is_two_thirds : 
  prob_A_B_together = 2/3 := by sorry

end NUMINAMATH_CALUDE_prob_A_B_together_is_two_thirds_l3375_337503


namespace NUMINAMATH_CALUDE_equal_tasks_after_transfer_l3375_337540

/-- Given that Robyn has 4 tasks and Sasha has 14 tasks, prove that if Robyn takes 5 tasks from Sasha, they will have an equal number of tasks. -/
theorem equal_tasks_after_transfer (robyn_initial : Nat) (sasha_initial : Nat) (tasks_transferred : Nat) : 
  robyn_initial = 4 → 
  sasha_initial = 14 → 
  tasks_transferred = 5 → 
  (robyn_initial + tasks_transferred = sasha_initial - tasks_transferred) := by
  sorry

#check equal_tasks_after_transfer

end NUMINAMATH_CALUDE_equal_tasks_after_transfer_l3375_337540
