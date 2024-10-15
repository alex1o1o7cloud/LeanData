import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l1041_104113

theorem polynomial_factor_implies_coefficients 
  (a b : ℚ) 
  (h : ∃ (c d k : ℚ), ax^4 + bx^3 + 38*x^2 - 12*x + 15 = (3*x^2 - 2*x + 2)*(c*x^2 + d*x + k)) :
  a = -75/2 ∧ b = 59/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l1041_104113


namespace NUMINAMATH_CALUDE_equation_solution_approximation_l1041_104120

theorem equation_solution_approximation : ∃ x : ℝ, 
  (2.5 * ((x * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2000.0000000000002) ∧ 
  (abs (x - 3.6) < 0.0000000000000005) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_approximation_l1041_104120


namespace NUMINAMATH_CALUDE_lcm_23_46_827_l1041_104174

theorem lcm_23_46_827 (h1 : 46 = 23 * 2) (h2 : Nat.Prime 827) :
  Nat.lcm 23 (Nat.lcm 46 827) = 38042 :=
by sorry

end NUMINAMATH_CALUDE_lcm_23_46_827_l1041_104174


namespace NUMINAMATH_CALUDE_aspirin_percentage_of_max_dosage_l1041_104193

-- Define the medication schedule and dosages
def aspirin_dosage : ℕ := 325
def aspirin_frequency : ℕ := 12
def aspirin_max_dosage : ℕ := 4000
def hours_per_day : ℕ := 24

-- Define the function to calculate total daily dosage
def total_daily_dosage (dosage frequency : ℕ) : ℕ :=
  dosage * (hours_per_day / frequency)

-- Define the function to calculate percentage of max dosage
def percentage_of_max_dosage (daily_dosage max_dosage : ℕ) : ℚ :=
  (daily_dosage : ℚ) / (max_dosage : ℚ) * 100

-- Theorem statement
theorem aspirin_percentage_of_max_dosage :
  percentage_of_max_dosage 
    (total_daily_dosage aspirin_dosage aspirin_frequency) 
    aspirin_max_dosage = 16.25 := by
  sorry

end NUMINAMATH_CALUDE_aspirin_percentage_of_max_dosage_l1041_104193


namespace NUMINAMATH_CALUDE_no_triangle_tangent_to_both_curves_l1041_104145

/-- C₁ is the unit circle -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- C₂ is an ellipse with semi-major axis a and semi-minor axis b -/
def C₂ (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- A triangle is externally tangent to C₁ if all its vertices lie outside or on C₁ 
    and each side is tangent to C₁ -/
def externally_tangent_C₁ (A B C : ℝ × ℝ) : Prop := sorry

/-- A triangle is internally tangent to C₂ if all its vertices lie inside or on C₂ 
    and each side is tangent to C₂ -/
def internally_tangent_C₂ (a b : ℝ) (A B C : ℝ × ℝ) : Prop := sorry

theorem no_triangle_tangent_to_both_curves (a b : ℝ) :
  a > b ∧ b > 0 ∧ C₂ a b 1 1 →
  ¬ ∃ (A B C : ℝ × ℝ), externally_tangent_C₁ A B C ∧ internally_tangent_C₂ a b A B C :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_tangent_to_both_curves_l1041_104145


namespace NUMINAMATH_CALUDE_linlins_speed_l1041_104187

/-- Proves that Linlin's speed is 400 meters per minute given the problem conditions --/
theorem linlins_speed (total_distance : ℕ) (time_taken : ℕ) (qingqing_speed : ℕ) :
  total_distance = 3290 →
  time_taken = 7 →
  qingqing_speed = 70 →
  (total_distance / time_taken - qingqing_speed : ℕ) = 400 :=
by sorry

end NUMINAMATH_CALUDE_linlins_speed_l1041_104187


namespace NUMINAMATH_CALUDE_equation_solution_l1041_104127

theorem equation_solution : ∃ x : ℝ, 7 * (4 * x + 3) - 9 = -3 * (2 - 9 * x) + 5 * x ∧ x = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1041_104127


namespace NUMINAMATH_CALUDE_nathan_bananas_l1041_104159

/-- The number of bananas Nathan has, given the specified bunches -/
def total_bananas (bunches_of_eight : Nat) (bananas_per_bunch_eight : Nat)
                  (bunches_of_seven : Nat) (bananas_per_bunch_seven : Nat) : Nat :=
  bunches_of_eight * bananas_per_bunch_eight + bunches_of_seven * bananas_per_bunch_seven

/-- Proof that Nathan has 83 bananas given the specified bunches -/
theorem nathan_bananas :
  total_bananas 6 8 5 7 = 83 := by
  sorry

end NUMINAMATH_CALUDE_nathan_bananas_l1041_104159


namespace NUMINAMATH_CALUDE_solve_equation_l1041_104123

theorem solve_equation : ∃ x : ℝ, 5 * (x - 9) = 6 * (3 - 3 * x) + 6 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1041_104123


namespace NUMINAMATH_CALUDE_restaurant_sales_problem_l1041_104161

/-- Represents the dinner sales for a restaurant over four days. -/
structure RestaurantSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- The conditions and proof goal for the restaurant sales problem. -/
theorem restaurant_sales_problem (sales : RestaurantSales) : 
  sales.monday = 40 →
  sales.tuesday = sales.monday + 40 →
  sales.wednesday = sales.tuesday / 2 →
  sales.thursday > sales.wednesday →
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday = 203 →
  sales.thursday - sales.wednesday = 3 := by
  sorry


end NUMINAMATH_CALUDE_restaurant_sales_problem_l1041_104161


namespace NUMINAMATH_CALUDE_theater_ticket_area_l1041_104189

/-- The area of a rectangular theater ticket -/
theorem theater_ticket_area (perimeter width : ℝ) (h1 : perimeter = 28) (h2 : width = 6) :
  let length := (perimeter - 2 * width) / 2
  width * length = 48 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_area_l1041_104189


namespace NUMINAMATH_CALUDE_abs_neg_two_l1041_104128

theorem abs_neg_two : |(-2 : ℤ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_l1041_104128


namespace NUMINAMATH_CALUDE_inequality_range_l1041_104190

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l1041_104190


namespace NUMINAMATH_CALUDE_factorization_equality_l1041_104116

theorem factorization_equality (x y : ℝ) : x^2 * y + 2 * x * y + y = y * (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1041_104116


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1041_104178

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) :
  min x y = 3 := by sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1041_104178


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1041_104175

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (Real.sin θ)^6 / a^2 + (Real.cos θ)^6 / b^2 = 1 / (a^2 + b^2) →
  (Real.sin θ)^12 / a^5 + (Real.cos θ)^12 / b^5 = (a^7 + b^7) / (a^2 + b^2)^6 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1041_104175


namespace NUMINAMATH_CALUDE_increasing_perfect_powers_sum_l1041_104172

def s (n : ℕ+) : ℕ := sorry

theorem increasing_perfect_powers_sum (x : ℝ) :
  ∃ N : ℕ, ∀ n > N, (Finset.range n).sup (fun i => s ⟨i + 1, Nat.succ_pos i⟩) / n > x := by
  sorry

end NUMINAMATH_CALUDE_increasing_perfect_powers_sum_l1041_104172


namespace NUMINAMATH_CALUDE_maria_gave_65_towels_l1041_104155

/-- The number of towels Maria gave to her mother -/
def towels_given_to_mother (green_towels white_towels remaining_towels : ℕ) : ℕ :=
  green_towels + white_towels - remaining_towels

/-- Proof that Maria gave 65 towels to her mother -/
theorem maria_gave_65_towels :
  towels_given_to_mother 40 44 19 = 65 := by
  sorry

end NUMINAMATH_CALUDE_maria_gave_65_towels_l1041_104155


namespace NUMINAMATH_CALUDE_range_of_a_l1041_104137

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ∈ Set.union (Set.Iic (-2)) {1} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1041_104137


namespace NUMINAMATH_CALUDE_abby_emma_weight_l1041_104185

/-- The combined weight of two people given their individual weights -/
def combined_weight (w1 w2 : ℝ) : ℝ := w1 + w2

/-- Proves that Abby and Emma weigh 310 pounds together given the weights of pairs -/
theorem abby_emma_weight
  (a b c d e : ℝ)  -- Individual weights of Abby, Bart, Cindy, Damon, and Emma
  (h1 : combined_weight a b = 270)  -- Abby and Bart
  (h2 : combined_weight b c = 255)  -- Bart and Cindy
  (h3 : combined_weight c d = 280)  -- Cindy and Damon
  (h4 : combined_weight d e = 295)  -- Damon and Emma
  : combined_weight a e = 310 := by
  sorry

#check abby_emma_weight

end NUMINAMATH_CALUDE_abby_emma_weight_l1041_104185


namespace NUMINAMATH_CALUDE_regular_nonagon_side_equals_diagonal_difference_l1041_104126

/-- A regular nonagon -/
structure RegularNonagon where
  -- Define the necessary properties of a regular nonagon
  side_length : ℝ
  longest_diagonal : ℝ
  shortest_diagonal : ℝ
  side_length_pos : 0 < side_length
  longest_diagonal_pos : 0 < longest_diagonal
  shortest_diagonal_pos : 0 < shortest_diagonal
  longest_ge_shortest : shortest_diagonal ≤ longest_diagonal

/-- 
The side length of a regular nonagon is equal to the difference 
between its longest diagonal and shortest diagonal 
-/
theorem regular_nonagon_side_equals_diagonal_difference 
  (n : RegularNonagon) : 
  n.side_length = n.longest_diagonal - n.shortest_diagonal :=
sorry

end NUMINAMATH_CALUDE_regular_nonagon_side_equals_diagonal_difference_l1041_104126


namespace NUMINAMATH_CALUDE_largest_three_digit_sum_l1041_104136

/-- A function that computes the sum ABC + CA + B -/
def digit_sum (A B C : ℕ) : ℕ := 101 * A + 11 * B + 11 * C

/-- A predicate that checks if three natural numbers are different digits -/
def are_different_digits (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A < 10 ∧ B < 10 ∧ C < 10

theorem largest_three_digit_sum :
  ∃ A B C : ℕ, are_different_digits A B C ∧ 
  digit_sum A B C = 986 ∧
  ∀ X Y Z : ℕ, are_different_digits X Y Z → 
  digit_sum X Y Z ≤ 986 ∧ digit_sum X Y Z < 1000 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_sum_l1041_104136


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1041_104115

theorem geometric_sequence_problem :
  ∀ (a b c d : ℝ),
    (a / b = b / c) →                   -- geometric sequence condition
    (c / d = b / c) →                   -- geometric sequence condition
    (a - b = 6) →                       -- difference between first and second
    (c - d = 5) →                       -- difference between third and fourth
    (a^2 + b^2 + c^2 + d^2 = 793) →     -- sum of squares
    ((a = 18 ∧ b = 12 ∧ c = 15 ∧ d = 10) ∨
     (a = -12 ∧ b = -18 ∧ c = -10 ∧ d = -15)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1041_104115


namespace NUMINAMATH_CALUDE_eight_divided_by_repeating_decimal_l1041_104184

/-- The repeating decimal 0.4444... as a rational number -/
def repeating_decimal : ℚ := 4 / 9

/-- The result of 8 divided by the repeating decimal 0.4444... -/
theorem eight_divided_by_repeating_decimal : 8 / repeating_decimal = 18 := by sorry

end NUMINAMATH_CALUDE_eight_divided_by_repeating_decimal_l1041_104184


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l1041_104101

theorem modulus_of_complex_number : 
  Complex.abs (Complex.mk 1 (-2)) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l1041_104101


namespace NUMINAMATH_CALUDE_initial_girls_count_initial_girls_count_proof_l1041_104134

theorem initial_girls_count : ℕ → ℕ → Prop :=
  fun b g =>
    (3 * (g - 20) = b) →
    (4 * (b - 60) = g - 20) →
    g = 42

-- The proof is omitted
theorem initial_girls_count_proof : ∃ b g : ℕ, initial_girls_count b g := by sorry

end NUMINAMATH_CALUDE_initial_girls_count_initial_girls_count_proof_l1041_104134


namespace NUMINAMATH_CALUDE_inequality_proof_l1041_104198

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_condition : a + b + c ≤ 3) : 
  (3 > (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1)) ∧ 
   (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1)) ≥ 3 / 2) ∧
  ((a + 1) / (a * (a + 2)) + (b + 1) / (b * (b + 2)) + (c + 1) / (c * (c + 2)) ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1041_104198


namespace NUMINAMATH_CALUDE_min_c_value_l1041_104180

theorem min_c_value (a b c : ℕ+) (h1 : a < b) (h2 : b < c)
  (h3 : ∃! p : ℝ × ℝ, (2 * p.1 + p.2 = 2010 ∧
    p.2 = |p.1 - a.val| + |p.1 - b.val| + |p.1 - c.val|)) :
  1006 ≤ c.val := by sorry

end NUMINAMATH_CALUDE_min_c_value_l1041_104180


namespace NUMINAMATH_CALUDE_m_increasing_range_l1041_104169

def f (x : ℝ) : ℝ := (x - 1)^2

def is_m_increasing (f : ℝ → ℝ) (m : ℝ) (D : Set ℝ) : Prop :=
  m ≠ 0 ∧ ∀ x ∈ D, x + m ∈ D ∧ f (x + m) ≥ f x

theorem m_increasing_range (m : ℝ) :
  is_m_increasing f m (Set.Ici 0) → m ∈ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_m_increasing_range_l1041_104169


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1041_104130

theorem quadratic_inequality_solution_sets 
  (c b a : ℝ) 
  (h : Set.Ioo (-3 : ℝ) (1/2) = {x : ℝ | c * x^2 + b * x + a < 0}) : 
  {x : ℝ | a * x^2 + b * x + c ≥ 0} = Set.Icc (-1/3 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1041_104130


namespace NUMINAMATH_CALUDE_inequality_solution_l1041_104146

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 8 / (x + 4) ≥ 2) ↔ (x < -4 ∨ x ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1041_104146


namespace NUMINAMATH_CALUDE_bracelets_made_l1041_104110

/-- The number of beads in each bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of metal beads Nancy has -/
def nancy_metal_beads : ℕ := 40

/-- The number of pearl beads Nancy has -/
def nancy_pearl_beads : ℕ := nancy_metal_beads + 20

/-- The number of crystal beads Rose has -/
def rose_crystal_beads : ℕ := 20

/-- The number of stone beads Rose has -/
def rose_stone_beads : ℕ := 2 * rose_crystal_beads

/-- The total number of beads Nancy and Rose have -/
def total_beads : ℕ := nancy_metal_beads + nancy_pearl_beads + rose_crystal_beads + rose_stone_beads

/-- The theorem stating the number of bracelets Nancy and Rose can make -/
theorem bracelets_made : total_beads / beads_per_bracelet = 20 := by
  sorry

end NUMINAMATH_CALUDE_bracelets_made_l1041_104110


namespace NUMINAMATH_CALUDE_raduzhny_population_l1041_104199

/-- The number of villages in Sunny Valley -/
def num_villages : ℕ := 10

/-- The population of Znoynoe village -/
def znoynoe_population : ℕ := 1000

/-- The amount by which Znoynoe's population exceeds the average -/
def excess_population : ℕ := 90

/-- The total population of all villages in Sunny Valley -/
def total_population : ℕ := znoynoe_population + (num_villages - 1) * (znoynoe_population - excess_population)

/-- The average population of villages in Sunny Valley -/
def average_population : ℕ := total_population / num_villages

theorem raduzhny_population : 
  ∃ (raduzhny_pop : ℕ), 
    raduzhny_pop = average_population ∧ 
    raduzhny_pop = 900 :=
sorry

end NUMINAMATH_CALUDE_raduzhny_population_l1041_104199


namespace NUMINAMATH_CALUDE_p_current_age_is_fifteen_l1041_104141

/-- Given the age ratios of two people P and Q at different times, 
    prove that P's current age is 15 years. -/
theorem p_current_age_is_fifteen :
  ∀ (p q : ℕ),
  (p - 3) / (q - 3) = 4 / 3 →
  (p + 6) / (q + 6) = 7 / 6 →
  p = 15 := by
  sorry

end NUMINAMATH_CALUDE_p_current_age_is_fifteen_l1041_104141


namespace NUMINAMATH_CALUDE_certain_number_proof_l1041_104131

theorem certain_number_proof : ∃! x : ℝ, x + (1/4 * 48) = 27 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1041_104131


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_55_l1041_104125

/-- The sum of n consecutive positive integers starting from a -/
def consecutiveSum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- Predicate to check if a sequence of n consecutive integers starting from a sums to 55 -/
def isValidSequence (a n : ℕ) : Prop :=
  a > 0 ∧ consecutiveSum a n = 55

theorem max_consecutive_integers_sum_55 :
  (∃ a : ℕ, isValidSequence a 10) ∧
  (∀ n : ℕ, n > 10 → ¬∃ a : ℕ, isValidSequence a n) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_55_l1041_104125


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1041_104112

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a^2 > b^2) ∧
  ¬(∀ a b, a^2 > b^2 → a > b ∧ b > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1041_104112


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1041_104102

/-- A line passing through (1, -1) and perpendicular to 3x - 2y = 0 has the equation 2x + 3y + 1 = 0 -/
theorem perpendicular_line_equation :
  ∀ (x y : ℝ),
  (2 * x + 3 * y + 1 = 0) ↔
  (∃ (m : ℝ), (y - (-1) = m * (x - 1)) ∧ 
              (m * 3 = -1/2) ∧
              (2 * 1 + 3 * (-1) + 1 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1041_104102


namespace NUMINAMATH_CALUDE_total_ingredients_for_batches_l1041_104142

/-- The amount of flour needed for one batch of cookies (in cups) -/
def flour_per_batch : ℝ := 4

/-- The amount of sugar needed for one batch of cookies (in cups) -/
def sugar_per_batch : ℝ := 1.5

/-- The number of batches we want to make -/
def num_batches : ℕ := 8

/-- Theorem: The total amount of flour and sugar combined needed for 8 batches is 44 cups -/
theorem total_ingredients_for_batches : 
  (flour_per_batch + sugar_per_batch) * num_batches = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_ingredients_for_batches_l1041_104142


namespace NUMINAMATH_CALUDE_pears_picked_total_l1041_104121

/-- The number of pears Sara picked -/
def sara_pears : ℕ := 45

/-- The number of pears Sally picked -/
def sally_pears : ℕ := 11

/-- The total number of pears picked -/
def total_pears : ℕ := sara_pears + sally_pears

theorem pears_picked_total : total_pears = 56 := by
  sorry

end NUMINAMATH_CALUDE_pears_picked_total_l1041_104121


namespace NUMINAMATH_CALUDE_blueberries_per_box_l1041_104107

/-- The number of blueberries in each blue box -/
def B : ℕ := sorry

/-- The number of strawberries in each red box -/
def S : ℕ := sorry

/-- The difference between strawberries in a red box and blueberries in a blue box is 12 -/
axiom diff_strawberries_blueberries : S - B = 12

/-- Replacing one blue box with one red box increases the difference between total strawberries and total blueberries by 76 -/
axiom replacement_difference : 2 * S = 76

/-- The number of blueberries in each blue box is 26 -/
theorem blueberries_per_box : B = 26 := by sorry

end NUMINAMATH_CALUDE_blueberries_per_box_l1041_104107


namespace NUMINAMATH_CALUDE_radio_operator_distribution_probability_radio_operator_distribution_probability_proof_l1041_104100

/-- The probability of each group having exactly one radio operator when 12 soldiers 
    (including 3 radio operators) are randomly divided into groups of 3, 4, and 5 soldiers. -/
theorem radio_operator_distribution_probability : ℝ :=
  let total_soldiers : ℕ := 12
  let radio_operators : ℕ := 3
  let group_sizes : List ℕ := [3, 4, 5]
  3 / 11

/-- Proof of the radio operator distribution probability theorem -/
theorem radio_operator_distribution_probability_proof :
  radio_operator_distribution_probability = 3 / 11 := by
  sorry

end NUMINAMATH_CALUDE_radio_operator_distribution_probability_radio_operator_distribution_probability_proof_l1041_104100


namespace NUMINAMATH_CALUDE_current_average_score_l1041_104143

/-- Represents the bonus calculation and test scores for Karen's class -/
structure TestScores where
  baseBonus : ℕ := 500
  bonusPerPoint : ℕ := 10
  baseScore : ℕ := 75
  maxScore : ℕ := 150
  gradedTests : ℕ := 8
  totalTests : ℕ := 10
  targetBonus : ℕ := 600
  lastTwoTestsScore : ℕ := 290

/-- The theorem states that given the conditions, the current average score of the graded tests is 70 -/
theorem current_average_score (ts : TestScores) : 
  (ts.targetBonus - ts.baseBonus) / ts.bonusPerPoint + ts.baseScore = 85 →
  ts.gradedTests * (((ts.targetBonus - ts.baseBonus) / ts.bonusPerPoint + ts.baseScore) * ts.totalTests - ts.lastTwoTestsScore) / ts.totalTests = 70 := by
  sorry

end NUMINAMATH_CALUDE_current_average_score_l1041_104143


namespace NUMINAMATH_CALUDE_percentage_of_whole_l1041_104133

theorem percentage_of_whole (part whole : ℝ) (h : whole ≠ 0) :
  (part / whole) * 100 = 40.25 ↔ part = 193.2 ∧ whole = 480 :=
sorry

end NUMINAMATH_CALUDE_percentage_of_whole_l1041_104133


namespace NUMINAMATH_CALUDE_smallest_k_value_l1041_104188

theorem smallest_k_value (m n k : ℤ) (h : 221 * m + 247 * n + 323 * k = 2001) :
  (k > 100 ∧ ∀ k' > 100, 221 * m + 247 * n + 323 * k' = 2001 → k ≤ k') → k = 111 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_value_l1041_104188


namespace NUMINAMATH_CALUDE_sqrt_three_squared_l1041_104150

theorem sqrt_three_squared : (Real.sqrt 3)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_l1041_104150


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1041_104109

/-- Given a cone with base radius 4 cm and unfolded lateral surface radius 5 cm,
    prove that its lateral surface area is 20π cm². -/
theorem cone_lateral_surface_area :
  ∀ (base_radius unfolded_radius : ℝ),
    base_radius = 4 →
    unfolded_radius = 5 →
    let lateral_area := (1/2) * unfolded_radius^2 * (2 * Real.pi * base_radius / unfolded_radius)
    lateral_area = 20 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1041_104109


namespace NUMINAMATH_CALUDE_min_argument_on_semicircle_l1041_104114

open Complex

noncomputable def min_argument : ℝ := Real.pi - Real.arctan (5 * Real.sqrt 6 / 12)

theorem min_argument_on_semicircle :
  ∀ z : ℂ, (abs z = 1 ∧ im z > 0) →
  arg ((z - 2) / (z + 3)) ≥ min_argument :=
by sorry

end NUMINAMATH_CALUDE_min_argument_on_semicircle_l1041_104114


namespace NUMINAMATH_CALUDE_sunflower_height_l1041_104106

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Converts inches to feet, rounding down -/
def inches_to_feet (inches : ℕ) : ℕ :=
  inches / 12

theorem sunflower_height (sister_height_feet : ℕ) (sister_height_inches : ℕ) 
  (height_difference : ℕ) :
  sister_height_feet = 4 →
  sister_height_inches = 3 →
  height_difference = 21 →
  inches_to_feet (feet_inches_to_inches sister_height_feet sister_height_inches + height_difference) = 6 :=
by sorry

end NUMINAMATH_CALUDE_sunflower_height_l1041_104106


namespace NUMINAMATH_CALUDE_decreasing_interval_of_sine_function_l1041_104119

/-- Given a function f(x) = 2sin(2x + φ) where 0 < φ < π/2 and f(0) = √3,
    prove that the decreasing interval of f(x) on [0, π] is [π/12, 7π/12]. -/
theorem decreasing_interval_of_sine_function (φ : Real) 
    (h1 : 0 < φ) (h2 : φ < π/2) 
    (f : Real → Real) 
    (hf : ∀ x, f x = 2 * Real.sin (2 * x + φ)) 
    (h3 : f 0 = Real.sqrt 3) :
    (Set.Icc (π/12 : Real) (7*π/12) : Set Real) = 
    {x ∈ Set.Icc (0 : Real) π | ∀ y ∈ Set.Icc (0 : Real) π, x < y → f y < f x} :=
  sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_sine_function_l1041_104119


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l1041_104192

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∀ n : ℕ, n > 0 → n.sqrt ^ 2 = n → n % 2 = 0 → n % 3 = 0 → n % 5 = 0 → n ≥ 900 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l1041_104192


namespace NUMINAMATH_CALUDE_fair_coin_tosses_l1041_104103

/-- 
Given a fair coin with probability 1/2 for each side, 
if the probability of landing on the same side n times is 1/16, 
then n must be 4.
-/
theorem fair_coin_tosses (n : ℕ) : 
  (1 / 2 : ℝ) ^ n = 1 / 16 → n = 4 := by sorry

end NUMINAMATH_CALUDE_fair_coin_tosses_l1041_104103


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l1041_104151

theorem unique_solution_to_equation :
  ∃! x : ℝ, x ≠ -1 ∧ x ≠ -3 ∧
  (x^3 + 3*x^2 - x) / (x^2 + 4*x + 3) + x = -7 ∧
  x = -5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l1041_104151


namespace NUMINAMATH_CALUDE_x_squared_plus_3xy_plus_y_squared_l1041_104157

theorem x_squared_plus_3xy_plus_y_squared (x y : ℝ) 
  (h1 : x * y = -3) 
  (h2 : x + y = -4) : 
  x^2 + 3*x*y + y^2 = 13 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_3xy_plus_y_squared_l1041_104157


namespace NUMINAMATH_CALUDE_reeses_height_l1041_104148

theorem reeses_height (parker daisy reese : ℝ) 
  (h1 : parker = daisy - 4)
  (h2 : daisy = reese + 8)
  (h3 : (parker + daisy + reese) / 3 = 64) :
  reese = 60 := by sorry

end NUMINAMATH_CALUDE_reeses_height_l1041_104148


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l1041_104176

theorem tangent_circle_radius (R : ℝ) (chord_length : ℝ) (ratio : ℝ) :
  R = 5 →
  chord_length = 8 →
  ratio = 1/3 →
  ∃ (r₁ r₂ : ℝ), (r₁ = 8/9 ∧ r₂ = 32/9) ∧
    (∀ (r : ℝ), (r = r₁ ∨ r = r₂) ↔
      (∃ (C : ℝ × ℝ),
        C.1^2 + C.2^2 = R^2 ∧
        C.1^2 + (C.2 - chord_length * ratio)^2 = r^2 ∧
        (R - r)^2 = (r + C.2)^2 + C.1^2)) :=
by sorry


end NUMINAMATH_CALUDE_tangent_circle_radius_l1041_104176


namespace NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l1041_104194

theorem largest_angle_in_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 →  -- Angles are positive
    a + b + c = 180 →        -- Sum of angles in a triangle is 180°
    ∃ (x : ℝ), 
      a = 3*x ∧ b = 4*x ∧ c = 5*x →  -- Angles are in ratio 3:4:5
      max a (max b c) = 75 :=  -- The largest angle is 75°
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l1041_104194


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1041_104166

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) ↔ -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1041_104166


namespace NUMINAMATH_CALUDE_combined_annual_income_after_expenses_l1041_104167

def brady_income : List ℕ := [150, 200, 250, 300, 200, 150, 180, 220, 240, 270, 300, 350]
def dwayne_income : List ℕ := [100, 150, 200, 250, 150, 120, 140, 190, 180, 230, 260, 300]
def brady_expense : ℕ := 450
def dwayne_expense : ℕ := 300

theorem combined_annual_income_after_expenses :
  (brady_income.sum - brady_expense) + (dwayne_income.sum - dwayne_expense) = 3930 := by
  sorry

end NUMINAMATH_CALUDE_combined_annual_income_after_expenses_l1041_104167


namespace NUMINAMATH_CALUDE_negative_three_squared_opposite_l1041_104111

/-- Two real numbers are opposite if their sum is zero -/
def are_opposite (a b : ℝ) : Prop := a + b = 0

/-- Theorem stating that (-3)² and -3² are opposite numbers -/
theorem negative_three_squared_opposite : are_opposite ((-3)^2) (-3^2) := by
  sorry

end NUMINAMATH_CALUDE_negative_three_squared_opposite_l1041_104111


namespace NUMINAMATH_CALUDE_courtyard_paving_l1041_104160

/-- The number of bricks required to pave a rectangular courtyard -/
def bricks_required (courtyard_length courtyard_width brick_length brick_width : ℚ) : ℚ :=
  (courtyard_length * courtyard_width) / (brick_length * brick_width)

/-- Theorem stating the number of bricks required for the specific courtyard and brick sizes -/
theorem courtyard_paving :
  bricks_required 25 15 0.2 0.1 = 18750 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_paving_l1041_104160


namespace NUMINAMATH_CALUDE_tylers_age_l1041_104105

theorem tylers_age (clay jessica alex tyler : ℕ) : 
  tyler = 3 * clay + 1 →
  jessica = 2 * tyler - 4 →
  alex = (clay + jessica) / 2 →
  clay + jessica + alex + tyler = 52 →
  tyler = 13 := by
sorry

end NUMINAMATH_CALUDE_tylers_age_l1041_104105


namespace NUMINAMATH_CALUDE_expression_evaluation_l1041_104144

theorem expression_evaluation :
  let x : ℚ := -3
  let numerator := 5 + x * (2 + x) - 2^2
  let denominator := x - 2 + x^2
  numerator / denominator = 1 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1041_104144


namespace NUMINAMATH_CALUDE_dance_students_l1041_104132

/-- Represents the number of students taking each elective in a school -/
structure SchoolElectives where
  total : ℕ
  art : ℕ
  music : ℕ
  dance : ℕ

/-- The properties of the school electives -/
def valid_electives (s : SchoolElectives) : Prop :=
  s.total = 400 ∧
  s.art = 200 ∧
  s.music = s.total / 5 ∧
  s.total = s.art + s.music + s.dance

/-- Theorem stating that the number of students taking dance is 120 -/
theorem dance_students (s : SchoolElectives) (h : valid_electives s) : s.dance = 120 := by
  sorry

end NUMINAMATH_CALUDE_dance_students_l1041_104132


namespace NUMINAMATH_CALUDE_inequality_solution_l1041_104162

theorem inequality_solution (x : ℝ) : 2*x + 6 > 5*x - 3 → x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1041_104162


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l1041_104129

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo 0 2, 
    ∀ y ∈ Set.Ioo 0 2, 
      x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l1041_104129


namespace NUMINAMATH_CALUDE_johnny_guitar_picks_l1041_104191

theorem johnny_guitar_picks (total_picks : ℕ) (red_picks blue_picks yellow_picks : ℕ) : 
  total_picks = red_picks + blue_picks + yellow_picks →
  2 * red_picks = total_picks →
  3 * blue_picks = total_picks →
  blue_picks = 12 →
  yellow_picks = 6 := by
sorry

end NUMINAMATH_CALUDE_johnny_guitar_picks_l1041_104191


namespace NUMINAMATH_CALUDE_problem_1_l1041_104165

theorem problem_1 (a : ℚ) (h : a = 1/2) : 2*a^2 - 5*a + a^2 + 4*a - 3*a^2 - 2 = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1041_104165


namespace NUMINAMATH_CALUDE_multiplication_division_equivalence_l1041_104135

theorem multiplication_division_equivalence (x : ℝ) : 
  (x * (4/5)) / (2/7) = x * (14/5) := by
sorry

end NUMINAMATH_CALUDE_multiplication_division_equivalence_l1041_104135


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_b_l1041_104154

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (sum_of_angles : A + B + C = 180)

-- Define an isosceles triangle
def IsIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- Theorem statement
theorem isosceles_triangle_angle_b (t : Triangle) 
  (ext_angle_A : ℝ) 
  (h_ext_angle : ext_angle_A = 110) 
  (h_ext_prop : t.B + t.C = ext_angle_A) :
  IsIsosceles t → t.B = 70 ∨ t.B = 55 ∨ t.B = 40 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_b_l1041_104154


namespace NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l1041_104158

theorem largest_n_for_sin_cos_inequality : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / (2 * ↑n)) ∧
  (∀ (m : ℕ), m > n → ∃ (y : ℝ), (Real.sin y)^m + (Real.cos y)^m < 1 / (2 * ↑m)) ∧
  n = 8 :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l1041_104158


namespace NUMINAMATH_CALUDE_pet_ownership_l1041_104195

theorem pet_ownership (total_students : ℕ) (dog_owners : ℕ) (cat_owners : ℕ) 
  (h1 : total_students = 45)
  (h2 : dog_owners = 25)
  (h3 : cat_owners = 34)
  (h4 : ∀ s, s ∈ Finset.range total_students → 
    (s ∈ Finset.range dog_owners ∨ s ∈ Finset.range cat_owners)) :
  Finset.card (Finset.range dog_owners ∩ Finset.range cat_owners) = 14 := by
  sorry

end NUMINAMATH_CALUDE_pet_ownership_l1041_104195


namespace NUMINAMATH_CALUDE_inequality_proof_l1041_104147

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  (2 + x)^2 / (1 + x)^2 + (2 + y)^2 / (1 + y)^2 ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1041_104147


namespace NUMINAMATH_CALUDE_simplify_fraction_l1041_104138

theorem simplify_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (10 * a^2 * b) / (5 * a * b) = 2 * a :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1041_104138


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1041_104140

/-- An isosceles triangle with side lengths 2 and 5 has a perimeter of 12. -/
theorem isosceles_triangle_perimeter : 
  ∀ (a b c : ℝ), 
  a = 5 → b = 5 → c = 2 →  -- Two sides are 5, one side is 2
  a = b →                  -- The triangle is isosceles
  a + b + c = 12 :=        -- The perimeter is 12
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1041_104140


namespace NUMINAMATH_CALUDE_M_subset_N_l1041_104181

def M : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}
def N : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l1041_104181


namespace NUMINAMATH_CALUDE_friends_signed_up_first_day_l1041_104171

/-- The number of friends who signed up on the first day -/
def friends_first_day : ℕ := sorry

/-- The total number of friends who signed up (including first day and rest of the week) -/
def total_friends : ℕ := friends_first_day + 7

/-- The total money earned by Katrina and her friends -/
def total_money : ℕ := 125

theorem friends_signed_up_first_day : 
  5 + 5 * total_friends + 5 * total_friends = total_money ∧ friends_first_day = 5 := by sorry

end NUMINAMATH_CALUDE_friends_signed_up_first_day_l1041_104171


namespace NUMINAMATH_CALUDE_circle_polar_equation_l1041_104168

/-- The polar equation ρ = 2cosθ represents a circle with center at (1,0) and radius 1 -/
theorem circle_polar_equation :
  ∀ (ρ θ : ℝ), ρ = 2 * Real.cos θ ↔
  (ρ * Real.cos θ - 1)^2 + (ρ * Real.sin θ)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_polar_equation_l1041_104168


namespace NUMINAMATH_CALUDE_f_properties_l1041_104182

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the function f
noncomputable def f (x : ℝ) : ℤ := floor ((x + 1) / 3 - floor (x / 3))

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f (x + 3) = f x) ∧ 
  (∀ y : ℤ, y ∈ Set.range f → y = 0 ∨ y = 1) ∧
  (∀ y : ℤ, y = 0 ∨ y = 1 → ∃ x : ℝ, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1041_104182


namespace NUMINAMATH_CALUDE_sum_of_star_tip_angles_l1041_104173

/-- The angle measurement of one tip of an 8-pointed star formed by connecting
    eight evenly spaced points on a circle -/
def star_tip_angle : ℝ := 67.5

/-- The number of tips in an 8-pointed star -/
def num_tips : ℕ := 8

/-- Theorem: The sum of the angle measurements of the eight tips of an 8-pointed star,
    formed by connecting eight evenly spaced points on a circle, is equal to 540° -/
theorem sum_of_star_tip_angles :
  (num_tips : ℝ) * star_tip_angle = 540 := by sorry

end NUMINAMATH_CALUDE_sum_of_star_tip_angles_l1041_104173


namespace NUMINAMATH_CALUDE_football_team_members_l1041_104122

/-- The total number of members in a football team after new members join -/
def total_members (initial : ℕ) (new : ℕ) : ℕ :=
  initial + new

/-- Theorem stating that the total number of members in the football team is 59 -/
theorem football_team_members :
  total_members 42 17 = 59 := by sorry

end NUMINAMATH_CALUDE_football_team_members_l1041_104122


namespace NUMINAMATH_CALUDE_function_satisfies_conditions_l1041_104183

theorem function_satisfies_conditions (m n : ℕ) : 
  let f : ℕ → ℕ → ℕ := λ m n => m * n
  (m ≥ 1 ∧ n ≥ 1 → 2 * (f m n) = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1)) ∧
  f m 0 = 0 ∧ f 0 n = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_conditions_l1041_104183


namespace NUMINAMATH_CALUDE_ceiling_negative_five_thirds_squared_l1041_104104

theorem ceiling_negative_five_thirds_squared : ⌈(-5/3)^2⌉ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_five_thirds_squared_l1041_104104


namespace NUMINAMATH_CALUDE_bubble_gum_cost_l1041_104177

/-- Given a number of bubble gum pieces and a total cost in cents,
    calculate the cost per piece of bubble gum. -/
def cost_per_piece (num_pieces : ℕ) (total_cost : ℕ) : ℕ :=
  total_cost / num_pieces

/-- Theorem stating that the cost per piece of bubble gum is 18 cents
    given the specific conditions of the problem. -/
theorem bubble_gum_cost :
  cost_per_piece 136 2448 = 18 := by
  sorry

end NUMINAMATH_CALUDE_bubble_gum_cost_l1041_104177


namespace NUMINAMATH_CALUDE_inequality_proof_l1041_104152

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) ≥ Real.sqrt (a^2 + a*c + c^2) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1041_104152


namespace NUMINAMATH_CALUDE_negation_of_or_implies_both_false_l1041_104124

theorem negation_of_or_implies_both_false (p q : Prop) :
  (¬(p ∨ q)) → (¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_or_implies_both_false_l1041_104124


namespace NUMINAMATH_CALUDE_water_fraction_in_mixture_l1041_104139

/-- Given a cement mixture with total weight, sand fraction, and gravel weight,
    calculate the fraction of water in the mixture. -/
theorem water_fraction_in_mixture
  (total_weight : ℝ)
  (sand_fraction : ℝ)
  (gravel_weight : ℝ)
  (h1 : total_weight = 48)
  (h2 : sand_fraction = 1/3)
  (h3 : gravel_weight = 8) :
  (total_weight - (sand_fraction * total_weight + gravel_weight)) / total_weight = 1/2 := by
  sorry

#check water_fraction_in_mixture

end NUMINAMATH_CALUDE_water_fraction_in_mixture_l1041_104139


namespace NUMINAMATH_CALUDE_union_equals_reals_l1041_104153

def S : Set ℝ := {x | (x - 2)^2 > 9}
def T (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 8}

theorem union_equals_reals (a : ℝ) : S ∪ T a = Set.univ ↔ a ∈ Set.Ioo (-3) (-1) := by
  sorry

end NUMINAMATH_CALUDE_union_equals_reals_l1041_104153


namespace NUMINAMATH_CALUDE_x_range_l1041_104118

/-- The function f(x) = x^2 + ax -/
def f (x a : ℝ) : ℝ := x^2 + a*x

/-- The theorem stating the range of x given the conditions -/
theorem x_range (x : ℝ) :
  (∀ a ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x a ≥ 3 - a) →
  (x ≤ -1 - Real.sqrt 2 ∨ x ≥ 1 + Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_x_range_l1041_104118


namespace NUMINAMATH_CALUDE_squirrel_acorn_division_l1041_104117

theorem squirrel_acorn_division (total_acorns : ℕ) (acorns_per_month : ℕ) (spring_acorns : ℕ) : 
  total_acorns = 210 → acorns_per_month = 60 → spring_acorns = 30 →
  (total_acorns - 3 * acorns_per_month) / (total_acorns / 3 - acorns_per_month) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_squirrel_acorn_division_l1041_104117


namespace NUMINAMATH_CALUDE_solution_proof_l1041_104164

/-- The equation we're solving -/
def equation (x : ℝ) : Prop :=
  4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20) = x^2 - 13*x - 6

/-- The largest real solution to the equation -/
noncomputable def n : ℝ := 13 + Real.sqrt 61

/-- The decomposition of n into d + √(e + √f) -/
def d : ℕ := 13
def e : ℕ := 61
def f : ℕ := 0

theorem solution_proof :
  equation n ∧ 
  n = d + Real.sqrt (e + Real.sqrt f) ∧
  d + e + f = 74 := by sorry

end NUMINAMATH_CALUDE_solution_proof_l1041_104164


namespace NUMINAMATH_CALUDE_opposite_of_cube_root_eight_l1041_104186

theorem opposite_of_cube_root_eight (x : ℝ) : x^3 = 8 → -x = -2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_cube_root_eight_l1041_104186


namespace NUMINAMATH_CALUDE_triangle_area_from_rectangle_l1041_104179

/-- The area of one right triangle formed by cutting a rectangle diagonally --/
theorem triangle_area_from_rectangle (length width : Real) (h_length : length = 0.5) (h_width : width = 0.3) :
  (length * width) / 2 = 0.075 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_rectangle_l1041_104179


namespace NUMINAMATH_CALUDE_right_triangular_pyramid_surface_area_l1041_104156

/-- A regular triangular pyramid with right-angled lateral faces -/
structure RightTriangularPyramid where
  base_edge : ℝ
  is_regular : Bool
  lateral_faces_right_angled : Bool

/-- The total surface area of a right triangular pyramid -/
def total_surface_area (p : RightTriangularPyramid) : ℝ :=
  sorry

/-- Theorem: The total surface area of a regular triangular pyramid with 
    right-angled lateral faces and base edge length 2 is 3 + √3 -/
theorem right_triangular_pyramid_surface_area :
  ∀ (p : RightTriangularPyramid), 
    p.base_edge = 2 → 
    p.is_regular = true → 
    p.lateral_faces_right_angled = true → 
    total_surface_area p = 3 + Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangular_pyramid_surface_area_l1041_104156


namespace NUMINAMATH_CALUDE_height_sum_l1041_104163

/-- Given the heights of John, Lena, and Rebeca, prove that the sum of Lena's and Rebeca's heights is 295 cm. -/
theorem height_sum (john_height lena_height rebeca_height : ℕ) 
  (h1 : john_height = 152)
  (h2 : john_height = lena_height + 15)
  (h3 : rebeca_height = john_height + 6) :
  lena_height + rebeca_height = 295 := by
  sorry

end NUMINAMATH_CALUDE_height_sum_l1041_104163


namespace NUMINAMATH_CALUDE_rolls_bought_l1041_104197

theorem rolls_bought (price_per_dozen : ℝ) (money_spent : ℝ) (rolls_per_dozen : ℕ) : 
  price_per_dozen = 5 → money_spent = 15 → rolls_per_dozen = 12 → 
  (money_spent / price_per_dozen) * rolls_per_dozen = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_rolls_bought_l1041_104197


namespace NUMINAMATH_CALUDE_percent_difference_l1041_104108

theorem percent_difference (N M : ℝ) (h : N > 0) : 
  let N' := 1.5 * N
  100 - (M / N') * 100 = 100 - (200 * M) / (3 * N) :=
by sorry

end NUMINAMATH_CALUDE_percent_difference_l1041_104108


namespace NUMINAMATH_CALUDE_daughters_and_granddaughters_without_children_l1041_104149

/-- Represents the family structure of Marilyn and her descendants -/
structure FamilyStructure where
  daughters : ℕ
  granddaughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The number of daughters each daughter with children has -/
def daughters_per_mother : ℕ := 5

/-- Axioms representing the given conditions -/
axiom marilyn : FamilyStructure
axiom marilyn_daughters : marilyn.daughters = 10
axiom marilyn_total : marilyn.total_descendants = 40
axiom marilyn_granddaughters : marilyn.granddaughters = marilyn.total_descendants - marilyn.daughters
axiom marilyn_daughters_with_children : 
  marilyn.daughters_with_children * daughters_per_mother = marilyn.granddaughters

/-- The main theorem to prove -/
theorem daughters_and_granddaughters_without_children : 
  marilyn.granddaughters + (marilyn.daughters - marilyn.daughters_with_children) = 34 := by
  sorry

end NUMINAMATH_CALUDE_daughters_and_granddaughters_without_children_l1041_104149


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l1041_104170

theorem circumscribed_sphere_surface_area (cube_volume : ℝ) (h : cube_volume = 64) :
  let cube_side := cube_volume ^ (1/3)
  let sphere_radius := cube_side * Real.sqrt 3 / 2
  4 * Real.pi * sphere_radius ^ 2 = 48 * Real.pi :=
by
  sorry

#check circumscribed_sphere_surface_area

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l1041_104170


namespace NUMINAMATH_CALUDE_line_through_circle_center_l1041_104196

/-- The line equation -/
def line_eq (x y : ℝ) (a : ℝ) : Prop :=
  3 * x + y + a = 0

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- The center of the circle -/
def circle_center (x y : ℝ) : Prop :=
  circle_eq x y ∧ ∀ x' y', circle_eq x' y' → (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2

/-- The main theorem -/
theorem line_through_circle_center (a : ℝ) :
  (∃ x y : ℝ, circle_center x y ∧ line_eq x y a) → a = 1 :=
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l1041_104196
