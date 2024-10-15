import Mathlib

namespace NUMINAMATH_CALUDE_apples_collected_l895_89563

/-- The number of apples Lexie picked -/
def lexie_apples : ℕ := 12

/-- The number of apples Tom picked -/
def tom_apples : ℕ := 2 * lexie_apples

/-- The total number of apples collected -/
def total_apples : ℕ := lexie_apples + tom_apples

theorem apples_collected : total_apples = 36 := by
  sorry

end NUMINAMATH_CALUDE_apples_collected_l895_89563


namespace NUMINAMATH_CALUDE_one_non_negative_solution_condition_l895_89568

/-- The quadratic equation defined by parameter a -/
def quadratic_equation (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 - 2*(a + 1) * x + 2*(a + 1)

/-- Predicate to check if the equation has only one non-negative solution -/
def has_one_non_negative_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, x ≥ 0 ∧ quadratic_equation a x = 0

/-- Theorem stating the condition for the equation to have only one non-negative solution -/
theorem one_non_negative_solution_condition (a : ℝ) :
  has_one_non_negative_solution a ↔ ((-1 ≤ a ∧ a ≤ 1) ∨ a = 3) :=
sorry

end NUMINAMATH_CALUDE_one_non_negative_solution_condition_l895_89568


namespace NUMINAMATH_CALUDE_adjacent_probability_l895_89510

/-- Represents the number of students in the group photo --/
def total_students : ℕ := 6

/-- Represents the number of rows in the seating arrangement --/
def num_rows : ℕ := 3

/-- Represents the number of seats per row --/
def seats_per_row : ℕ := 2

/-- Calculates the total number of seating arrangements --/
def total_arrangements : ℕ := Nat.factorial total_students

/-- Calculates the number of favorable arrangements where Abby and Bridget are adjacent but not in the middle row --/
def favorable_arrangements : ℕ := 4 * 2 * Nat.factorial (total_students - 2)

/-- Represents the probability of Abby and Bridget being adjacent but not in the middle row --/
def probability : ℚ := favorable_arrangements / total_arrangements

/-- Theorem stating that the probability of Abby and Bridget being adjacent but not in the middle row is 4/15 --/
theorem adjacent_probability :
  probability = 4 / 15 := by sorry

end NUMINAMATH_CALUDE_adjacent_probability_l895_89510


namespace NUMINAMATH_CALUDE_no_snow_probability_l895_89574

theorem no_snow_probability (p : ℚ) (h : p = 2/3) : (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l895_89574


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l895_89528

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 4 * x + 3 ∧
  (∀ (y : ℝ), y * |y| = 4 * y + 3 → x ≤ y) ∧
  x = -3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l895_89528


namespace NUMINAMATH_CALUDE_east_northwest_angle_l895_89551

/-- Given a circle with ten equally spaced rays, where one ray points due North,
    the smaller angle between the rays pointing East and Northwest is 36°. -/
theorem east_northwest_angle (n : ℕ) (ray_angle : ℝ) : 
  n = 10 ∧ ray_angle = 360 / n → 36 = ray_angle := by sorry

end NUMINAMATH_CALUDE_east_northwest_angle_l895_89551


namespace NUMINAMATH_CALUDE_factorial_difference_sum_l895_89570

theorem factorial_difference_sum : Nat.factorial 10 - Nat.factorial 8 + Nat.factorial 6 = 3589200 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_sum_l895_89570


namespace NUMINAMATH_CALUDE_hyperbola_equation_l895_89573

/-- A hyperbola with given properties has the equation x²/18 - y²/32 = 1 -/
theorem hyperbola_equation (e : ℝ) (a b : ℝ) (h1 : e = 5/3) 
  (h2 : a > 0) (h3 : b > 0) (h4 : e = Real.sqrt (a^2 + b^2) / a) 
  (h5 : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 ↔ 
    (8*x + 2*Real.sqrt 7*y - 16)^2 / (64/a^2 + 28/b^2) = 256) : 
  a^2 = 18 ∧ b^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l895_89573


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_product_l895_89542

theorem cube_sum_from_sum_and_product (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_product_l895_89542


namespace NUMINAMATH_CALUDE_trapezoid_theorem_l895_89560

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A B C D : Point)

/-- Checks if a point is on a line segment -/
def isOnSegment (P Q R : Point) : Prop := sorry

/-- Checks if two line segments intersect -/
def intersect (P Q R S : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (P Q : Point) : ℝ := sorry

/-- Main theorem -/
theorem trapezoid_theorem (ABCD : Trapezoid) (E O P : Point) :
  isOnSegment ABCD.A E ABCD.D →
  distance ABCD.A E = distance ABCD.B ABCD.C →
  intersect ABCD.C ABCD.A ABCD.B ABCD.D →
  intersect ABCD.C E ABCD.B ABCD.D →
  intersect ABCD.C ABCD.A ABCD.B O →
  intersect ABCD.C E ABCD.B P →
  distance ABCD.B O = distance P ABCD.D →
  (distance ABCD.A ABCD.D)^2 = (distance ABCD.B ABCD.C)^2 + (distance ABCD.A ABCD.D) * (distance ABCD.B ABCD.C) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_theorem_l895_89560


namespace NUMINAMATH_CALUDE_range_of_a_l895_89506

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (0 < a ∧ a < 1) ∧ 
  (∀ x y : ℝ, x < y → f a x > f a y) →
  1/7 ≤ a ∧ a < 1/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l895_89506


namespace NUMINAMATH_CALUDE_simultaneous_processing_equation_l895_89561

/-- Represents the total number of workers --/
def total_workers : ℕ := 26

/-- Represents the number of type A parts to process --/
def type_a_parts : ℕ := 2100

/-- Represents the number of type B parts to process --/
def type_b_parts : ℕ := 1200

/-- Represents the number of type A parts a worker can process per day --/
def type_a_rate : ℕ := 30

/-- Represents the number of type B parts a worker can process per day --/
def type_b_rate : ℕ := 20

/-- Theorem stating that the equation correctly represents the simultaneous processing of both types of parts --/
theorem simultaneous_processing_equation (x : ℝ) (h1 : 0 < x) (h2 : x < total_workers) :
  (type_a_parts : ℝ) / (type_a_rate * x) = (type_b_parts : ℝ) / (type_b_rate * (total_workers - x)) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_processing_equation_l895_89561


namespace NUMINAMATH_CALUDE_arithmetic_progression_square_sum_l895_89567

/-- For real numbers a, b, c forming an arithmetic progression,
    3(a² + b² + c²) = 6(a-b)² + (a+b+c)² -/
theorem arithmetic_progression_square_sum (a b c : ℝ) 
  (h : a + c = 2 * b) : 
  3 * (a^2 + b^2 + c^2) = 6 * (a - b)^2 + (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_square_sum_l895_89567


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_product_l895_89533

theorem consecutive_even_numbers_product (x : ℤ) : 
  (x % 2 = 0) →  -- x is even
  ((x + 2) % 2 = 0) →  -- x + 2 is even (consecutive even number)
  (x * (x + 2) = 224) →  -- their product is 224
  x * (x + 2) = 224 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_product_l895_89533


namespace NUMINAMATH_CALUDE_chemical_representations_correct_l895_89553

/-- Represents a chemical element -/
inductive Element : Type
| C : Element
| H : Element
| O : Element
| N : Element
| Si : Element
| P : Element

/-- Represents a chemical formula -/
structure ChemicalFormula :=
  (elements : List (Element × ℕ))

/-- Represents a structural formula -/
structure StructuralFormula :=
  (formula : String)

/-- Definition of starch chemical formula -/
def starchFormula : ChemicalFormula :=
  ⟨[(Element.C, 6), (Element.H, 10), (Element.O, 5)]⟩

/-- Definition of glycine structural formula -/
def glycineFormula : StructuralFormula :=
  ⟨"H₂N-CH₂-COOH"⟩

/-- Definition of silicate-containing materials -/
def silicateProducts : List String :=
  ["glass", "ceramics", "cement"]

/-- Definition of red tide causing elements -/
def redTideElements : List Element :=
  [Element.N, Element.P]

/-- Theorem stating the correctness of the chemical representations -/
theorem chemical_representations_correct :
  (starchFormula.elements = [(Element.C, 6), (Element.H, 10), (Element.O, 5)]) ∧
  (glycineFormula.formula = "H₂N-CH₂-COOH") ∧
  (∀ product ∈ silicateProducts, ∃ e ∈ product.toList, e = 'S') ∧
  (redTideElements = [Element.N, Element.P]) :=
sorry


end NUMINAMATH_CALUDE_chemical_representations_correct_l895_89553


namespace NUMINAMATH_CALUDE_three_hour_charge_l895_89527

/-- Represents the pricing structure and total charges for a psychologist's therapy sessions. -/
structure TherapyPricing where
  first_hour : ℕ  -- Price of the first hour
  additional_hour : ℕ  -- Price of each additional hour
  first_hour_premium : first_hour = additional_hour + 30  -- First hour costs $30 more
  five_hour_total : first_hour + 4 * additional_hour = 400  -- Total for 5 hours is $400

/-- Theorem stating that given the pricing structure, the total charge for 3 hours is $252. -/
theorem three_hour_charge (p : TherapyPricing) : 
  p.first_hour + 2 * p.additional_hour = 252 := by
  sorry


end NUMINAMATH_CALUDE_three_hour_charge_l895_89527


namespace NUMINAMATH_CALUDE_unique_permutations_count_l895_89504

/-- The number of elements in our multiset -/
def n : ℕ := 5

/-- The number of occurrences of the digit 3 -/
def k₁ : ℕ := 3

/-- The number of occurrences of the digit 7 -/
def k₂ : ℕ := 2

/-- The theorem stating that the number of unique permutations of our multiset is 10 -/
theorem unique_permutations_count : (n.factorial) / (k₁.factorial * k₂.factorial) = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_permutations_count_l895_89504


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l895_89575

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 →
  (((3 / (x - 1)) - x - 1) / ((x^2 - 4*x + 4) / (x - 1))) = (2 + x) / (2 - x) ∧
  (((3 / (0 - 1)) - 0 - 1) / ((0^2 - 4*0 + 4) / (0 - 1))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l895_89575


namespace NUMINAMATH_CALUDE_smaller_number_problem_l895_89556

theorem smaller_number_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 14) (h4 : y = 3 * x) : x = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l895_89556


namespace NUMINAMATH_CALUDE_probability_of_purple_marble_l895_89512

theorem probability_of_purple_marble (p_blue p_green p_purple : ℝ) :
  p_blue = 0.25 →
  p_green = 0.4 →
  p_blue + p_green + p_purple = 1 →
  p_purple = 0.35 := by
sorry

end NUMINAMATH_CALUDE_probability_of_purple_marble_l895_89512


namespace NUMINAMATH_CALUDE_unique_solution_l895_89515

/-- A function from positive integers to positive integers -/
def PositiveIntFunction := ℕ+ → ℕ+

/-- The property that f satisfies for all positive integers m and n -/
def SatisfiesEquation (f : PositiveIntFunction) : Prop :=
  ∀ m n : ℕ+, f (f (f m)^2 + 2 * (f n)^2) = m^2 + 2 * n^2

/-- The identity function on positive integers -/
def identityFunction : PositiveIntFunction := λ n => n

/-- The theorem stating that the identity function is the only one satisfying the equation -/
theorem unique_solution :
  ∀ f : PositiveIntFunction, SatisfiesEquation f ↔ f = identityFunction :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l895_89515


namespace NUMINAMATH_CALUDE_john_total_calories_l895_89520

/-- The number of potato chips John eats -/
def num_chips : ℕ := 10

/-- The total calories of the potato chips -/
def total_chip_calories : ℕ := 60

/-- The number of cheezits John eats -/
def num_cheezits : ℕ := 6

/-- The calories of one potato chip -/
def calories_per_chip : ℚ := total_chip_calories / num_chips

/-- The calories of one cheezit -/
def calories_per_cheezit : ℚ := calories_per_chip * (1 + 1/3)

/-- The total calories John ate -/
def total_calories : ℚ := num_chips * calories_per_chip + num_cheezits * calories_per_cheezit

theorem john_total_calories : total_calories = 108 := by
  sorry

end NUMINAMATH_CALUDE_john_total_calories_l895_89520


namespace NUMINAMATH_CALUDE_continuity_at_three_l895_89534

def f (x : ℝ) : ℝ := 2 * x^2 - 4

theorem continuity_at_three :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → |f x - f 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_three_l895_89534


namespace NUMINAMATH_CALUDE_original_aotd_votes_l895_89545

/-- Represents the vote counts for three books --/
structure VoteCounts where
  got : ℕ  -- Game of Thrones
  twi : ℕ  -- Twilight
  aotd : ℕ  -- The Art of the Deal

/-- Represents the vote alteration process --/
def alter_votes (v : VoteCounts) : ℚ × ℚ × ℚ :=
  (v.got, v.twi / 2, v.aotd / 5)

/-- The theorem to be proved --/
theorem original_aotd_votes (v : VoteCounts) : 
  v.got = 10 ∧ v.twi = 12 ∧ 
  (let (got, twi, aotd) := alter_votes v
   got = (got + twi + aotd) / 2) →
  v.aotd = 20 :=
by sorry

end NUMINAMATH_CALUDE_original_aotd_votes_l895_89545


namespace NUMINAMATH_CALUDE_special_arithmetic_sequence_2007th_term_l895_89566

/-- An arithmetic sequence with special properties -/
structure SpecialArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The 3rd, 5th, and 11th terms form a geometric sequence -/
  geometric_property : (a + 2*d) * (a + 10*d) = (a + 4*d)^2
  /-- The 4th term is 6 -/
  fourth_term : a + 3*d = 6

/-- The nth term of an arithmetic sequence -/
def arithmetic_term (seq : SpecialArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a + (n - 1) * seq.d

/-- The main theorem -/
theorem special_arithmetic_sequence_2007th_term 
  (seq : SpecialArithmeticSequence) : 
  arithmetic_term seq 2007 = 6015 := by
  sorry

end NUMINAMATH_CALUDE_special_arithmetic_sequence_2007th_term_l895_89566


namespace NUMINAMATH_CALUDE_arithmetic_equation_l895_89530

theorem arithmetic_equation : 12.1212 + 17.0005 - 9.1103 = 20.0114 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l895_89530


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_mean_l895_89543

theorem cubic_equation_solutions_mean (x : ℝ) : 
  x^3 + 2*x^2 - 8*x - 4 = 0 → 
  ∃ (s : Finset ℝ), (∀ y ∈ s, y^3 + 2*y^2 - 8*y - 4 = 0) ∧ 
                    (s.card = 3) ∧ 
                    ((s.sum id) / s.card = -2/3) :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_mean_l895_89543


namespace NUMINAMATH_CALUDE_louise_and_tom_ages_l895_89582

/-- Given the age relationship between Louise and Tom, prove their current ages sum to 26 -/
theorem louise_and_tom_ages (L T : ℕ) 
  (h1 : L = T + 8) 
  (h2 : L + 4 = 3 * (T - 2)) : 
  L + T = 26 := by
  sorry

end NUMINAMATH_CALUDE_louise_and_tom_ages_l895_89582


namespace NUMINAMATH_CALUDE_x_greater_abs_y_sufficient_not_necessary_l895_89518

theorem x_greater_abs_y_sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x > |y| → x > y) ∧
  (∃ x y : ℝ, x > y ∧ ¬(x > |y|)) :=
sorry

end NUMINAMATH_CALUDE_x_greater_abs_y_sufficient_not_necessary_l895_89518


namespace NUMINAMATH_CALUDE_johns_bagels_l895_89505

theorem johns_bagels (b m : ℕ) : 
  b + m = 7 →
  (90 * b + 60 * m) % 100 = 0 →
  b = 6 :=
by sorry

end NUMINAMATH_CALUDE_johns_bagels_l895_89505


namespace NUMINAMATH_CALUDE_sam_and_tina_distances_l895_89562

/-- Calculates the distance traveled given speed and time -/
def distance (speed time : ℝ) : ℝ := speed * time

theorem sam_and_tina_distances 
  (marguerite_distance marguerite_time sam_time tina_time : ℝ) 
  (marguerite_distance_positive : marguerite_distance > 0)
  (marguerite_time_positive : marguerite_time > 0)
  (sam_time_positive : sam_time > 0)
  (tina_time_positive : tina_time > 0)
  (h_marguerite_distance : marguerite_distance = 150)
  (h_marguerite_time : marguerite_time = 3)
  (h_sam_time : sam_time = 4)
  (h_tina_time : tina_time = 2) :
  let marguerite_speed := marguerite_distance / marguerite_time
  (distance marguerite_speed sam_time = 200) ∧ 
  (distance marguerite_speed tina_time = 100) := by
  sorry

end NUMINAMATH_CALUDE_sam_and_tina_distances_l895_89562


namespace NUMINAMATH_CALUDE_calculate_partner_b_contribution_b_contribution_is_16200_l895_89557

/-- Calculates the contribution of partner B given the initial investment of A, 
    the time before B joins, and the profit-sharing ratio. -/
theorem calculate_partner_b_contribution 
  (a_investment : ℕ) 
  (total_months : ℕ) 
  (b_join_month : ℕ) 
  (profit_ratio_a : ℕ) 
  (profit_ratio_b : ℕ) : ℕ :=
  let b_investment := 
    (a_investment * total_months * profit_ratio_b) / 
    (profit_ratio_a * (total_months - b_join_month))
  b_investment

/-- Proves that B's contribution is 16200 given the problem conditions -/
theorem b_contribution_is_16200 : 
  calculate_partner_b_contribution 4500 12 7 2 3 = 16200 := by
  sorry

end NUMINAMATH_CALUDE_calculate_partner_b_contribution_b_contribution_is_16200_l895_89557


namespace NUMINAMATH_CALUDE_collinearity_ABD_l895_89552

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two non-zero vectors are not collinear -/
def not_collinear (a b : V) : Prop := a ≠ 0 ∧ b ≠ 0 ∧ ¬∃ (r : ℝ), a = r • b

/-- Three points are collinear if the vector from the first to the third is a scalar multiple of the vector from the first to the second -/
def collinear (A B D : V) : Prop := ∃ (t : ℝ), D - A = t • (B - A)

theorem collinearity_ABD 
  (a b : V) 
  (h_not_collinear : not_collinear a b)
  (h_AB : B - A = a + b)
  (h_BC : C - B = a + 10 • b)
  (h_CD : D - C = 3 • (a - 2 • b)) :
  collinear A B D :=
sorry

end NUMINAMATH_CALUDE_collinearity_ABD_l895_89552


namespace NUMINAMATH_CALUDE_danny_soda_remaining_l895_89547

theorem danny_soda_remaining (bottles : ℕ) (consumed : ℚ) (given_away : ℚ) : 
  bottles = 3 → 
  consumed = 9/10 → 
  given_away = 7/10 → 
  (1 - consumed) + 2 * (1 - given_away) = 7/10 :=
by sorry

end NUMINAMATH_CALUDE_danny_soda_remaining_l895_89547


namespace NUMINAMATH_CALUDE_conic_sections_properties_l895_89514

-- Define the equations for the conic sections
def hyperbola_eq (x y : ℝ) : Prop := x^2 / 25 - y^2 / 9 = 1
def ellipse_eq (x y : ℝ) : Prop := x^2 / 35 + y^2 = 1
def parabola_eq (x y p : ℝ) : Prop := y^2 = 2 * p * x

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 - 5 * x + 2 = 0

-- Define the theorem
theorem conic_sections_properties :
  -- Proposition ②
  (∃ e₁ e₂ : ℝ, quadratic_eq e₁ ∧ quadratic_eq e₂ ∧ 0 < e₁ ∧ e₁ < 1 ∧ e₂ > 1) ∧
  -- Proposition ③
  (∃ c : ℝ, (∀ x y : ℝ, hyperbola_eq x y → x^2 - c^2 = 25) ∧
            (∀ x y : ℝ, ellipse_eq x y → x^2 + c^2 = 35)) ∧
  -- Proposition ④
  (∀ p : ℝ, p > 0 →
    ∃ x₀ y₀ r : ℝ,
      -- Circle equation
      (∀ x y : ℝ, (x - x₀)^2 + (y - y₀)^2 = r^2 →
        -- Tangent to directrix
        x = -p ∨
        -- Passes through focus
        (x₀ = p/2 ∧ y₀ = 0 ∧ r = p/2))) :=
sorry

end NUMINAMATH_CALUDE_conic_sections_properties_l895_89514


namespace NUMINAMATH_CALUDE_only_four_not_divide_98_l895_89508

theorem only_four_not_divide_98 :
  (∀ n ∈ ({2, 7, 14, 49} : Set Nat), 98 % n = 0) ∧ 98 % 4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_only_four_not_divide_98_l895_89508


namespace NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l895_89571

theorem largest_of_eight_consecutive_integers (a : ℕ) 
  (h1 : a > 0) 
  (h2 : (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6) + (a + 7)) = 5400) : 
  (a + 7) = 678 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l895_89571


namespace NUMINAMATH_CALUDE_photo_collection_l895_89511

theorem photo_collection (total photos : ℕ) (tim paul tom : ℕ) : 
  total = 152 →
  tim = total - 100 →
  paul = tim + 10 →
  total = tim + paul + tom →
  tom = 38 := by
sorry

end NUMINAMATH_CALUDE_photo_collection_l895_89511


namespace NUMINAMATH_CALUDE_completing_square_transformation_l895_89500

/-- Given a quadratic equation x^2 - 2x - 4 = 0, prove that when transformed
    into the form (x-1)^2 = a using the completing the square method,
    the value of a is 5. -/
theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 4 = 0) → ∃ a : ℝ, ((x - 1)^2 = a) ∧ (a = 5) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l895_89500


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l895_89502

theorem intersection_point_x_coordinate (x y : ℝ) : 
  y = 3 * x + 4 ∧ 5 * x - y = 41 → x = 22.5 := by sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l895_89502


namespace NUMINAMATH_CALUDE_arrangement_count_is_2880_l895_89546

/-- The number of ways to choose k items from n items without replacement and where order matters. -/
def permutations (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def combinations (n k : ℕ) : ℕ := sorry

/-- The number of different arrangements of 4 boys and 3 girls in a row, where exactly 2 of the 3 girls are adjacent. -/
def arrangement_count : ℕ := sorry

theorem arrangement_count_is_2880 : arrangement_count = 2880 := by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_2880_l895_89546


namespace NUMINAMATH_CALUDE_expression_simplification_l895_89523

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 - 2) :
  (1 + 1 / (x - 2)) * ((x^2 - 4) / (x - 1)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l895_89523


namespace NUMINAMATH_CALUDE_intersection_point_value_l895_89576

/-- Given three lines that intersect at a single point, prove that the value of a is -1 --/
theorem intersection_point_value (a : ℝ) :
  (∃! p : ℝ × ℝ, (a * p.1 + 2 * p.2 + 8 = 0) ∧
                 (4 * p.1 + 3 * p.2 = 10) ∧
                 (2 * p.1 - p.2 = 10)) →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_value_l895_89576


namespace NUMINAMATH_CALUDE_smallest_prime_with_composite_odd_digit_sum_l895_89507

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for prime numbers -/
def is_prime (n : ℕ) : Prop := sorry

/-- Predicate for composite numbers -/
def is_composite (n : ℕ) : Prop := sorry

theorem smallest_prime_with_composite_odd_digit_sum :
  (is_prime 997) ∧ 
  (is_composite (sum_of_digits 997)) ∧ 
  (sum_of_digits 997 % 2 = 1) ∧
  (∀ p < 997, is_prime p → ¬(is_composite (sum_of_digits p) ∧ sum_of_digits p % 2 = 1)) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_composite_odd_digit_sum_l895_89507


namespace NUMINAMATH_CALUDE_karina_to_brother_age_ratio_l895_89521

-- Define the given information
def karina_birth_year : ℕ := 1970
def karina_current_age : ℕ := 40
def brother_birth_year : ℕ := 1990

-- Define the current year based on Karina's age
def current_year : ℕ := karina_birth_year + karina_current_age

-- Calculate brother's age
def brother_current_age : ℕ := current_year - brother_birth_year

-- Theorem to prove
theorem karina_to_brother_age_ratio :
  (karina_current_age : ℚ) / (brother_current_age : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_karina_to_brother_age_ratio_l895_89521


namespace NUMINAMATH_CALUDE_work_completion_theorem_l895_89596

/-- Represents the number of days required to complete the work -/
def original_days : ℕ := 11

/-- Represents the number of additional men who joined -/
def additional_men : ℕ := 10

/-- Represents the number of days saved after additional men joined -/
def days_saved : ℕ := 3

/-- Calculates the original number of men required to complete the work -/
def original_men : ℕ := 27

theorem work_completion_theorem :
  ∃ (work_rate : ℚ),
    (original_men * work_rate * original_days : ℚ) =
    ((original_men + additional_men) * work_rate * (original_days - days_saved) : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l895_89596


namespace NUMINAMATH_CALUDE_triangle_altitude_l895_89587

theorem triangle_altitude (A : ℝ) (b : ℝ) (h : ℝ) :
  A = 750 →
  b = 50 →
  A = (1/2) * b * h →
  h = 30 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_l895_89587


namespace NUMINAMATH_CALUDE_triangle_circles_radius_sum_l895_89526

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB : ℝ) (AC : ℝ) (BC : ℝ)

/-- Circle with given radius -/
structure Circle :=
  (radius : ℝ)

/-- Represents the radius of circle Q in the form m - n√k -/
structure RadiusForm :=
  (m : ℕ) (n : ℕ) (k : ℕ)

/-- Main theorem statement -/
theorem triangle_circles_radius_sum (ABC : Triangle) (P Q : Circle) (r : RadiusForm) :
  ABC.AB = 130 →
  ABC.AC = 130 →
  ABC.BC = 78 →
  P.radius = 25 →
  -- Circle P is tangent to AC and BC
  -- Circle Q is externally tangent to P and tangent to AB and BC
  -- No point of circle Q lies outside of triangle ABC
  Q.radius = r.m - r.n * Real.sqrt r.k →
  r.m > 0 →
  r.n > 0 →
  r.k > 0 →
  -- k is the product of distinct primes
  r.m + r.n * r.k = 131 := by
  sorry

end NUMINAMATH_CALUDE_triangle_circles_radius_sum_l895_89526


namespace NUMINAMATH_CALUDE_circle_area_difference_l895_89565

theorem circle_area_difference : 
  let d₁ : ℝ := 30
  let r₁ : ℝ := d₁ / 2
  let r₂ : ℝ := 10
  let r₃ : ℝ := 5
  (π * r₁^2) - (π * r₂^2) - (π * r₃^2) = 100 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_difference_l895_89565


namespace NUMINAMATH_CALUDE_equal_sums_l895_89541

-- Define the range of numbers
def N : ℕ := 999999

-- Function to determine if a number's nearest perfect square is odd
def nearest_square_odd (n : ℕ) : Prop := sorry

-- Function to determine if a number's nearest perfect square is even
def nearest_square_even (n : ℕ) : Prop := sorry

-- Sum of numbers with odd nearest perfect square
def sum_odd_group : ℕ := sorry

-- Sum of numbers with even nearest perfect square
def sum_even_group : ℕ := sorry

-- Theorem stating that the sums are equal
theorem equal_sums : sum_odd_group = sum_even_group := by sorry

end NUMINAMATH_CALUDE_equal_sums_l895_89541


namespace NUMINAMATH_CALUDE_blue_folder_stickers_l895_89594

/-- The number of stickers on each sheet in the blue folder -/
def blue_stickers_per_sheet : ℕ :=
  let total_stickers : ℕ := 60
  let sheets_per_folder : ℕ := 10
  let red_stickers_per_sheet : ℕ := 3
  let green_stickers_per_sheet : ℕ := 2
  let red_total := sheets_per_folder * red_stickers_per_sheet
  let green_total := sheets_per_folder * green_stickers_per_sheet
  let blue_total := total_stickers - red_total - green_total
  blue_total / sheets_per_folder

theorem blue_folder_stickers :
  blue_stickers_per_sheet = 1 := by
  sorry

end NUMINAMATH_CALUDE_blue_folder_stickers_l895_89594


namespace NUMINAMATH_CALUDE_max_sum_circle_50_l895_89559

/-- The maximum sum of x and y for integer solutions of x^2 + y^2 = 50 -/
theorem max_sum_circle_50 : 
  ∀ x y : ℤ, x^2 + y^2 = 50 → x + y ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_circle_50_l895_89559


namespace NUMINAMATH_CALUDE_period_length_proof_l895_89595

/-- Calculates the length of each period given the number of students, presentation time per student, and number of periods. -/
def period_length (num_students : ℕ) (presentation_time : ℕ) (num_periods : ℕ) : ℕ :=
  (num_students * presentation_time) / num_periods

/-- Proves that given 32 students, 5 minutes per presentation, and 4 periods, the length of each period is 40 minutes. -/
theorem period_length_proof :
  period_length 32 5 4 = 40 := by
  sorry

#eval period_length 32 5 4

end NUMINAMATH_CALUDE_period_length_proof_l895_89595


namespace NUMINAMATH_CALUDE_min_value_problem_l895_89524

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : 2 * a + 3 * b = 6) :
  (3 / a + 2 / b) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 6 ∧ 3 / a₀ + 2 / b₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l895_89524


namespace NUMINAMATH_CALUDE_expression_evaluation_l895_89538

theorem expression_evaluation : (3^2 - 1) - (4^2 - 2) + (5^2 - 3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l895_89538


namespace NUMINAMATH_CALUDE_hyperbola_equation_l895_89540

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : (x y : ℝ) → Prop := λ x y ↦ x^2 / a^2 - y^2 / b^2 = 1

-- Define the properties of the hyperbola
def has_focus (h : Hyperbola) (fx fy : ℝ) : Prop :=
  h.a^2 + h.b^2 = fx^2 + fy^2

def passes_through (h : Hyperbola) (px py : ℝ) : Prop :=
  h.eq px py

-- Theorem statement
theorem hyperbola_equation (h : Hyperbola) :
  has_focus h (Real.sqrt 6) 0 →
  passes_through h (-5) 2 →
  h.a^2 = 5 ∧ h.b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l895_89540


namespace NUMINAMATH_CALUDE_hawks_victory_margin_l895_89519

/-- Calculates the total score for a team given their scoring details --/
def team_score (touchdowns extra_points two_point_conversions field_goals safeties : ℕ) : ℕ :=
  touchdowns * 6 + extra_points + two_point_conversions * 2 + field_goals * 3 + safeties * 2

/-- Represents the scoring details of the Hawks --/
def hawks_score : ℕ :=
  team_score 4 2 1 2 1

/-- Represents the scoring details of the Eagles --/
def eagles_score : ℕ :=
  team_score 3 3 1 3 1

/-- Theorem stating that the Hawks won by a margin of 2 points --/
theorem hawks_victory_margin :
  hawks_score - eagles_score = 2 :=
sorry

end NUMINAMATH_CALUDE_hawks_victory_margin_l895_89519


namespace NUMINAMATH_CALUDE_no_significant_relationship_l895_89564

-- Define the contingency table data
def boys_enthusiasts : ℕ := 45
def boys_non_enthusiasts : ℕ := 10
def girls_enthusiasts : ℕ := 30
def girls_non_enthusiasts : ℕ := 15

-- Define the total number of students
def total_students : ℕ := boys_enthusiasts + boys_non_enthusiasts + girls_enthusiasts + girls_non_enthusiasts

-- Define the K² calculation function
def calculate_k_squared (a b c d : ℕ) : ℚ :=
  let n : ℕ := a + b + c + d
  (n : ℚ) * (a * d - b * c : ℚ)^2 / ((a + b : ℚ) * (c + d : ℚ) * (a + c : ℚ) * (b + d : ℚ))

-- Define the critical value for 95% confidence
def critical_value : ℚ := 3841 / 1000

-- Theorem statement
theorem no_significant_relationship : 
  calculate_k_squared boys_enthusiasts boys_non_enthusiasts girls_enthusiasts girls_non_enthusiasts < critical_value := by
  sorry


end NUMINAMATH_CALUDE_no_significant_relationship_l895_89564


namespace NUMINAMATH_CALUDE_inequality_proof_l895_89548

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l895_89548


namespace NUMINAMATH_CALUDE_amp_composition_l895_89599

-- Define the & operation (postfix)
def amp (x : ℤ) : ℤ := 7 - x

-- Define the & operation (prefix)
def amp_prefix (x : ℤ) : ℤ := x - 10

-- Theorem statement
theorem amp_composition : amp_prefix (amp 12) = -15 := by sorry

end NUMINAMATH_CALUDE_amp_composition_l895_89599


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l895_89597

/-- A line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : (a ≠ 0 ∨ b ≠ 0)

/-- A point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line --/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a line has equal intercepts on both axes --/
def Line.hasEqualIntercepts (l : Line) : Prop :=
  (l.a ≠ 0 ∧ l.b ≠ 0 ∧ -l.c/l.a = -l.c/l.b) ∨
  (l.a = 0 ∧ l.b = 0 ∧ l.c = 0)

/-- The main theorem --/
theorem line_through_point_with_equal_intercepts :
  ∃ (l₁ l₂ : Line),
    (Point.liesOn ⟨1, 5⟩ l₁) ∧
    (Point.liesOn ⟨1, 5⟩ l₂) ∧
    l₁.hasEqualIntercepts ∧
    l₂.hasEqualIntercepts ∧
    ((l₁.a = 1 ∧ l₁.b = 1 ∧ l₁.c = -6) ∨
     (l₂.a = 5 ∧ l₂.b = -1 ∧ l₂.c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l895_89597


namespace NUMINAMATH_CALUDE_sum_of_squares_not_perfect_square_l895_89509

def sum_of_squares (n : ℕ) (a : ℕ) : ℕ := 
  (2*n + 1) * a^2 + (2*n*(n+1)*(2*n+1)) / 3

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k^2

theorem sum_of_squares_not_perfect_square (n : ℕ) (h : n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :
  ∀ a : ℕ, ¬(is_perfect_square (sum_of_squares n a)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_not_perfect_square_l895_89509


namespace NUMINAMATH_CALUDE_female_kittens_count_l895_89532

theorem female_kittens_count (initial_cats : ℕ) (total_cats : ℕ) (male_kittens : ℕ) : 
  initial_cats = 2 → total_cats = 7 → male_kittens = 2 → 
  total_cats - initial_cats - male_kittens = 3 := by
sorry

end NUMINAMATH_CALUDE_female_kittens_count_l895_89532


namespace NUMINAMATH_CALUDE_solve_puppy_problem_l895_89572

def puppyProblem (initialPuppies : ℕ) (givenAway : ℕ) (kept : ℕ) (sellingPrice : ℕ) (profit : ℕ) : Prop :=
  let remainingAfterGiveaway := initialPuppies - givenAway
  let soldPuppies := remainingAfterGiveaway - kept
  let revenue := soldPuppies * sellingPrice
  let amountToStud := revenue - profit
  amountToStud = 300

theorem solve_puppy_problem :
  puppyProblem 8 4 1 600 1500 := by
  sorry

end NUMINAMATH_CALUDE_solve_puppy_problem_l895_89572


namespace NUMINAMATH_CALUDE_probability_between_C_and_D_l895_89539

/-- Given a line segment AB with points A, B, C, D, and E, prove that the probability
    of a randomly selected point on AB being between C and D is 1/2. -/
theorem probability_between_C_and_D (A B C D E : ℝ) : 
  A < B ∧ 
  B - A = 4 * (E - A) ∧
  B - A = 8 * (B - D) ∧
  D - A = 3 * (E - A) ∧
  B - D = 5 * (B - E) ∧
  C = D + (1/8) * (B - A) →
  (C - D) / (B - A) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_probability_between_C_and_D_l895_89539


namespace NUMINAMATH_CALUDE_triangle_condition_implies_a_ge_5_l895_89591

/-- The function f(x) = x^2 - 2x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + a

/-- Theorem: If for any three distinct values in [0, 3], f(x) can form a triangle, then a ≥ 5 -/
theorem triangle_condition_implies_a_ge_5 (a : ℝ) :
  (∀ x y z : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 ∧ 0 ≤ z ∧ z ≤ 3 →
    x ≠ y ∧ y ≠ z ∧ x ≠ z →
    f a x + f a y > f a z ∧ f a y + f a z > f a x ∧ f a x + f a z > f a y) →
  a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_condition_implies_a_ge_5_l895_89591


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l895_89578

theorem regular_polygon_sides (D : ℕ) (h : D = 20) : 
  ∃ (n : ℕ), n > 2 ∧ D = n * (n - 3) / 2 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l895_89578


namespace NUMINAMATH_CALUDE_swimming_practice_months_l895_89550

def total_required_hours : ℕ := 4000
def completed_hours : ℕ := 460
def practice_hours_per_month : ℕ := 400

theorem swimming_practice_months : 
  ∃ (months : ℕ), 
    months * practice_hours_per_month ≥ total_required_hours - completed_hours ∧ 
    (months - 1) * practice_hours_per_month < total_required_hours - completed_hours ∧
    months = 9 := by
  sorry

end NUMINAMATH_CALUDE_swimming_practice_months_l895_89550


namespace NUMINAMATH_CALUDE_reggie_layups_l895_89590

/-- Represents the score of a player in the basketball shooting contest -/
structure Score where
  layups : ℕ
  freeThrows : ℕ
  longShots : ℕ

/-- Calculates the total points for a given score -/
def totalPoints (s : Score) : ℕ :=
  s.layups + 2 * s.freeThrows + 3 * s.longShots

theorem reggie_layups : 
  ∀ (reggie_score : Score) (brother_score : Score),
    reggie_score.freeThrows = 2 →
    reggie_score.longShots = 1 →
    brother_score.layups = 0 →
    brother_score.freeThrows = 0 →
    brother_score.longShots = 4 →
    totalPoints reggie_score + 2 = totalPoints brother_score →
    reggie_score.layups = 3 := by
  sorry

#check reggie_layups

end NUMINAMATH_CALUDE_reggie_layups_l895_89590


namespace NUMINAMATH_CALUDE_negation_of_p_l895_89537

-- Define the proposition p
def p (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

-- State the theorem
theorem negation_of_p (f : ℝ → ℝ) : 
  ¬(p f) ↔ ∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_p_l895_89537


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l895_89549

/-- The area of an isosceles trapezoid with given dimensions -/
theorem isosceles_trapezoid_area (side : ℝ) (base1 base2 : ℝ) :
  side > 0 ∧ base1 > 0 ∧ base2 > 0 ∧ base1 < base2 ∧ side^2 > ((base2 - base1)/2)^2 →
  let height := Real.sqrt (side^2 - ((base2 - base1)/2)^2)
  (1/2 : ℝ) * (base1 + base2) * height = 48 ∧ side = 5 ∧ base1 = 9 ∧ base2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l895_89549


namespace NUMINAMATH_CALUDE_function_inequality_l895_89516

open Real

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h1 : ∀ x, deriv f x < f x) : 
  f 1 < ℯ * f 0 ∧ f 2014 < ℯ^2014 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l895_89516


namespace NUMINAMATH_CALUDE_contradiction_method_correctness_l895_89522

theorem contradiction_method_correctness :
  (∀ p q : ℝ, (p^3 + q^3 = 2) → (¬(p + q ≤ 2) ↔ p + q > 2)) ∧
  (∀ a b : ℝ, (|a| + |b| < 1) →
    (∃ x₁ : ℝ, x₁^2 + a*x₁ + b = 0 ∧ |x₁| ≥ 1) →
    False) :=
sorry

end NUMINAMATH_CALUDE_contradiction_method_correctness_l895_89522


namespace NUMINAMATH_CALUDE_inequality_proof_l895_89544

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_condition : x * y + y * z + z * x + 2 * x * y * z = 1) : 
  4 * x + y + z ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l895_89544


namespace NUMINAMATH_CALUDE_cookie_sales_revenue_l895_89589

-- Define the sales data for each girl on each day
def robyn_day1_packs : ℕ := 25
def robyn_day1_price : ℚ := 4
def lucy_day1_packs : ℕ := 17
def lucy_day1_price : ℚ := 5

def robyn_day2_packs : ℕ := 15
def robyn_day2_price : ℚ := 7/2
def lucy_day2_packs : ℕ := 9
def lucy_day2_price : ℚ := 9/2

def robyn_day3_packs : ℕ := 23
def robyn_day3_price : ℚ := 9/2
def lucy_day3_packs : ℕ := 20
def lucy_day3_price : ℚ := 7/2

-- Define the total revenue calculation
def total_revenue : ℚ :=
  robyn_day1_packs * robyn_day1_price +
  lucy_day1_packs * lucy_day1_price +
  robyn_day2_packs * robyn_day2_price +
  lucy_day2_packs * lucy_day2_price +
  robyn_day3_packs * robyn_day3_price +
  lucy_day3_packs * lucy_day3_price

-- Theorem statement
theorem cookie_sales_revenue :
  total_revenue = 451.5 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sales_revenue_l895_89589


namespace NUMINAMATH_CALUDE_a_2017_is_one_sixty_fifth_l895_89536

/-- Represents a proper fraction -/
structure ProperFraction where
  numerator : Nat
  denominator : Nat
  is_proper : numerator < denominator

/-- The sequence of proper fractions -/
def fraction_sequence : Nat → ProperFraction := sorry

/-- The 2017th term of the sequence -/
def a_2017 : ProperFraction := fraction_sequence 2017

/-- Theorem stating that the 2017th term is 1/65 -/
theorem a_2017_is_one_sixty_fifth : 
  a_2017.numerator = 1 ∧ a_2017.denominator = 65 := by sorry

end NUMINAMATH_CALUDE_a_2017_is_one_sixty_fifth_l895_89536


namespace NUMINAMATH_CALUDE_great_wall_scientific_notation_l895_89579

theorem great_wall_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 6700000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 6.7 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_great_wall_scientific_notation_l895_89579


namespace NUMINAMATH_CALUDE_x_range_theorem_l895_89585

theorem x_range_theorem (x : ℝ) :
  (∀ a b c : ℝ, a^2 + b^2 + c^2 = 1 → a + b + Real.sqrt 2 * c ≤ |x^2 - 1|) →
  x ≤ -Real.sqrt 3 ∨ x ≥ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_x_range_theorem_l895_89585


namespace NUMINAMATH_CALUDE_function_properties_l895_89501

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x

theorem function_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a < -Real.exp 1) 
  (hx : x₁ < x₂) 
  (hf₁ : f a x₁ = 0) 
  (hf₂ : f a x₂ = 0) :
  let t := Real.sqrt (x₂ / x₁)
  (deriv (f a)) ((3 * x₁ + x₂) / 4) < 0 ∧ 
  (t - 1) * (a + Real.sqrt 3) = -2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_function_properties_l895_89501


namespace NUMINAMATH_CALUDE_fraction_product_cubed_main_proof_l895_89569

theorem fraction_product_cubed (a b c d : ℚ) : 
  (a / b) ^ 3 * (c / d) ^ 3 = ((a * c) / (b * d)) ^ 3 :=
by sorry

theorem main_proof : (5 / 8 : ℚ) ^ 3 * (4 / 9 : ℚ) ^ 3 = 125 / 5832 :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_cubed_main_proof_l895_89569


namespace NUMINAMATH_CALUDE_total_envelopes_is_975_l895_89581

/-- The number of envelopes Kiera has of each color and in total -/
structure EnvelopeCount where
  blue : ℕ
  yellow : ℕ
  green : ℕ
  red : ℕ
  purple : ℕ
  total : ℕ

/-- Calculates the total number of envelopes given the conditions -/
def calculateEnvelopes : EnvelopeCount :=
  let blue := 120
  let yellow := blue - 25
  let green := 5 * yellow
  let red := (blue + yellow) / 2
  let purple := red + 71
  let total := blue + yellow + green + red + purple
  { blue := blue
  , yellow := yellow
  , green := green
  , red := red
  , purple := purple
  , total := total }

/-- Theorem stating that the total number of envelopes is 975 -/
theorem total_envelopes_is_975 : calculateEnvelopes.total = 975 := by
  sorry

#eval calculateEnvelopes.total

end NUMINAMATH_CALUDE_total_envelopes_is_975_l895_89581


namespace NUMINAMATH_CALUDE_train_length_l895_89525

/-- Given a train crossing a pole at a speed of 60 km/hr in 18 seconds,
    prove that the length of the train is 300 meters. -/
theorem train_length (speed : ℝ) (time_seconds : ℝ) (length : ℝ) :
  speed = 60 →
  time_seconds = 18 →
  length = speed * (time_seconds / 3600) * 1000 →
  length = 300 := by sorry

end NUMINAMATH_CALUDE_train_length_l895_89525


namespace NUMINAMATH_CALUDE_hyperbola_quadrants_l895_89577

theorem hyperbola_quadrants (k : ℝ) : k < 0 ∧ 2 * k^2 + k - 2 = -1 → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_quadrants_l895_89577


namespace NUMINAMATH_CALUDE_derivative_of_f_l895_89580

-- Define the function f
def f (x : ℝ) : ℝ := (3*x - 5)^2

-- State the theorem
theorem derivative_of_f :
  deriv f = λ x => 6 * (3*x - 5) := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l895_89580


namespace NUMINAMATH_CALUDE_cubic_function_derivative_l895_89584

theorem cubic_function_derivative (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + 3 * x^2 + 2
  let f' : ℝ → ℝ := λ x ↦ (3 * a * x^2) + (6 * x)
  f' (-1) = 4 → a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_derivative_l895_89584


namespace NUMINAMATH_CALUDE_unique_number_l895_89593

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digits_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem unique_number : 
  ∃! n : ℕ, is_two_digit n ∧ 
            n % 2 = 1 ∧ 
            n % 13 = 0 ∧ 
            is_perfect_square (digits_product n) ∧
            n = 91 := by sorry

end NUMINAMATH_CALUDE_unique_number_l895_89593


namespace NUMINAMATH_CALUDE_geometric_progression_m_existence_l895_89583

theorem geometric_progression_m_existence (m : ℂ) : 
  ∃ r : ℂ, r ≠ 0 ∧ 
    r ≠ r^2 ∧ r ≠ r^3 ∧ r^2 ≠ r^3 ∧
    r / (1 - r^2) = m ∧ 
    r^2 / (1 - r^3) = m ∧ 
    r^3 / (1 - r) = m := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_m_existence_l895_89583


namespace NUMINAMATH_CALUDE_ballet_slipper_price_fraction_l895_89588

/-- The price of one pair of high heels in dollars -/
def high_heels_price : ℚ := 60

/-- The number of pairs of ballet slippers bought -/
def ballet_slippers_count : ℕ := 5

/-- The total amount paid in dollars -/
def total_paid : ℚ := 260

/-- The fraction of the high heels price paid for each pair of ballet slippers -/
def ballet_slipper_fraction : ℚ := 2/3

theorem ballet_slipper_price_fraction :
  high_heels_price + ballet_slippers_count * (ballet_slipper_fraction * high_heels_price) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_ballet_slipper_price_fraction_l895_89588


namespace NUMINAMATH_CALUDE_floor_sum_for_specific_x_l895_89535

theorem floor_sum_for_specific_x : 
  let x : ℝ := 9.42
  ⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋ = 55 := by sorry

end NUMINAMATH_CALUDE_floor_sum_for_specific_x_l895_89535


namespace NUMINAMATH_CALUDE_submarine_hit_guaranteed_l895_89558

-- Define the type for the submarine's position and velocity
def Submarine := ℕ × ℕ+

-- Define the type for the firing sequence
def FiringSequence := ℕ → ℕ

-- The theorem statement
theorem submarine_hit_guaranteed :
  ∀ (sub : Submarine), ∃ (fire : FiringSequence), ∃ (t : ℕ),
    fire t = (sub.2 : ℕ) * t + sub.1 :=
by sorry

end NUMINAMATH_CALUDE_submarine_hit_guaranteed_l895_89558


namespace NUMINAMATH_CALUDE_only_second_expression_always_true_l895_89586

theorem only_second_expression_always_true :
  (∀ n a : ℝ, n * a^n = a) = False ∧
  (∀ a : ℝ, (a^2 - 3*a + 3)^0 = 1) = True ∧
  (3 - 3 = 6*(-3)^2) = False := by
sorry

end NUMINAMATH_CALUDE_only_second_expression_always_true_l895_89586


namespace NUMINAMATH_CALUDE_kellys_sister_visit_l895_89531

def vacation_length : ℕ := 3 * 7

def travel_days : ℕ := 1 + 1 + 2 + 2

def grandparents_days : ℕ := 5

def brother_days : ℕ := 5

theorem kellys_sister_visit (sister_days : ℕ) : 
  sister_days = vacation_length - (travel_days + grandparents_days + brother_days) → 
  sister_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_kellys_sister_visit_l895_89531


namespace NUMINAMATH_CALUDE_water_pouring_theorem_l895_89503

/-- Represents a state of water distribution among three containers -/
structure WaterState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a pouring action from one container to another -/
inductive PourAction
  | AtoB
  | AtoC
  | BtoA
  | BtoC
  | CtoA
  | CtoB

/-- Applies a pouring action to a water state -/
def applyPour (state : WaterState) (action : PourAction) : WaterState :=
  match action with
  | PourAction.AtoB => { a := state.a - state.b, b := state.b * 2, c := state.c }
  | PourAction.AtoC => { a := state.a - state.c, b := state.b, c := state.c * 2 }
  | PourAction.BtoA => { a := state.a * 2, b := state.b - state.a, c := state.c }
  | PourAction.BtoC => { a := state.a, b := state.b - state.c, c := state.c * 2 }
  | PourAction.CtoA => { a := state.a * 2, b := state.b, c := state.c - state.a }
  | PourAction.CtoB => { a := state.a, b := state.b * 2, c := state.c - state.b }

/-- Predicate to check if a container is empty -/
def isEmptyContainer (state : WaterState) : Prop :=
  state.a = 0 ∨ state.b = 0 ∨ state.c = 0

/-- The main theorem to be proved -/
theorem water_pouring_theorem (initialState : WaterState) :
  ∃ (actions : List PourAction), isEmptyContainer (actions.foldl applyPour initialState) :=
sorry


end NUMINAMATH_CALUDE_water_pouring_theorem_l895_89503


namespace NUMINAMATH_CALUDE_fraction_equality_l895_89554

theorem fraction_equality (m n : ℚ) (h : m / n = 2 / 3) : m / (m + n) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l895_89554


namespace NUMINAMATH_CALUDE_paint_combinations_count_l895_89517

/-- The number of available paint colors -/
def num_colors : ℕ := 6

/-- The number of available painting tools -/
def num_tools : ℕ := 4

/-- The number of combinations of color and different tools for two objects -/
def num_combinations : ℕ := num_colors * num_tools * (num_tools - 1)

theorem paint_combinations_count :
  num_combinations = 72 := by
  sorry

end NUMINAMATH_CALUDE_paint_combinations_count_l895_89517


namespace NUMINAMATH_CALUDE_tangent_line_equation_l895_89555

/-- The equation of the tangent line to y = x^2 + 1 at (-1, 2) is 2x + y = 0 -/
theorem tangent_line_equation : 
  let f : ℝ → ℝ := λ x => x^2 + 1
  let x₀ : ℝ := -1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (2*x + y = 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l895_89555


namespace NUMINAMATH_CALUDE_trader_profit_above_goal_l895_89513

theorem trader_profit_above_goal 
  (profit : ℝ) 
  (required_amount : ℝ) 
  (donation : ℝ) 
  (half_profit : ℝ) 
  (h1 : profit = 960) 
  (h2 : required_amount = 610) 
  (h3 : donation = 310) 
  (h4 : half_profit = profit / 2) : 
  half_profit + donation - required_amount = 180 := by
sorry

end NUMINAMATH_CALUDE_trader_profit_above_goal_l895_89513


namespace NUMINAMATH_CALUDE_smallest_multiple_of_24_and_36_not_20_l895_89592

theorem smallest_multiple_of_24_and_36_not_20 : 
  ∃ n : ℕ, n > 0 ∧ 24 ∣ n ∧ 36 ∣ n ∧ ¬(20 ∣ n) ∧ 
  ∀ m : ℕ, m > 0 → 24 ∣ m → 36 ∣ m → ¬(20 ∣ m) → n ≤ m :=
by
  -- The proof goes here
  sorry

#eval Nat.lcm 24 36  -- This should output 72

end NUMINAMATH_CALUDE_smallest_multiple_of_24_and_36_not_20_l895_89592


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l895_89529

def arithmeticSequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem fifth_term_of_arithmetic_sequence 
  (a : ℤ) (d : ℤ) 
  (h12 : arithmeticSequence a d 12 = 25)
  (h13 : arithmeticSequence a d 13 = 29) :
  arithmeticSequence a d 5 = -3 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l895_89529


namespace NUMINAMATH_CALUDE_inequality_proof_l895_89598

theorem inequality_proof (x y z : ℝ) (n : ℕ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x + y + z = 1) (h5 : n > 0) : 
  x^4 / (y * (1 - y^n)) + y^4 / (z * (1 - z^n)) + z^4 / (x * (1 - x^n)) ≥ 3^n / (3^(n+2) - 9) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l895_89598
