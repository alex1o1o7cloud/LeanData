import Mathlib

namespace tree_planting_cost_l392_39220

/-- The cost of planting trees to achieve a specific temperature drop -/
theorem tree_planting_cost (initial_temp final_temp temp_drop_per_tree cost_per_tree : ℝ) : 
  initial_temp - final_temp = 1.8 →
  temp_drop_per_tree = 0.1 →
  cost_per_tree = 6 →
  ((initial_temp - final_temp) / temp_drop_per_tree) * cost_per_tree = 108 := by
  sorry

end tree_planting_cost_l392_39220


namespace largest_number_theorem_l392_39255

theorem largest_number_theorem (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_products_eq : p * q + p * r + q * r = 1)
  (product_eq : p * q * r = 2) :
  max p (max q r) = (1 + Real.sqrt 5) / 2 := by
  sorry

end largest_number_theorem_l392_39255


namespace weight_gain_difference_l392_39219

/-- The weight gain problem at the family reunion -/
theorem weight_gain_difference (orlando_gain jose_gain fernando_gain : ℝ) : 
  orlando_gain = 5 →
  jose_gain > 2 * orlando_gain →
  fernando_gain = jose_gain / 2 - 3 →
  orlando_gain + jose_gain + fernando_gain = 20 →
  ∃ ε > 0, |jose_gain - 2 * orlando_gain - 3.67| < ε :=
by sorry

end weight_gain_difference_l392_39219


namespace intersection_equality_implies_x_values_l392_39279

theorem intersection_equality_implies_x_values (x : ℝ) : 
  let A : Set ℝ := {1, 4, x}
  let B : Set ℝ := {1, x^2}
  (A ∩ B = B) → (x = -2 ∨ x = 2 ∨ x = 0) :=
by
  sorry

end intersection_equality_implies_x_values_l392_39279


namespace binomial_expression_is_integer_l392_39250

theorem binomial_expression_is_integer (m n : ℕ) : 
  ∃ k : ℤ, k = (m.factorial * (2*n + 2*m).factorial) / 
              ((2*m).factorial * n.factorial * (n+m).factorial) :=
by sorry

end binomial_expression_is_integer_l392_39250


namespace volunteer_distribution_l392_39204

theorem volunteer_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 4) :
  (Nat.choose (n - 1) (k - 1) : ℕ) = 84 :=
by sorry

end volunteer_distribution_l392_39204


namespace statues_painted_l392_39265

theorem statues_painted (total_paint : ℚ) (paint_per_statue : ℚ) :
  total_paint = 3/6 ∧ paint_per_statue = 1/6 → total_paint / paint_per_statue = 3 :=
by sorry

end statues_painted_l392_39265


namespace base_subtraction_l392_39227

/-- Converts a number from base b to base 10 --/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The problem statement --/
theorem base_subtraction :
  let base7_num := [5, 4, 3, 2, 1]
  let base8_num := [1, 2, 3, 4, 5]
  toBase10 base7_num 7 - toBase10 base8_num 8 = 8190 := by
  sorry

end base_subtraction_l392_39227


namespace square_binomial_constant_l392_39231

theorem square_binomial_constant (b : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + b = (4 * x + c)^2) → b = 25 := by
  sorry

end square_binomial_constant_l392_39231


namespace marker_carton_cost_l392_39257

/-- Proves that the cost of each carton of markers is $20 given the specified conditions --/
theorem marker_carton_cost (
  pencil_cartons : ℕ)
  (pencil_boxes_per_carton : ℕ)
  (pencil_box_cost : ℕ)
  (marker_cartons : ℕ)
  (marker_boxes_per_carton : ℕ)
  (total_spent : ℕ)
  (h1 : pencil_cartons = 20)
  (h2 : pencil_boxes_per_carton = 10)
  (h3 : pencil_box_cost = 2)
  (h4 : marker_cartons = 10)
  (h5 : marker_boxes_per_carton = 5)
  (h6 : total_spent = 600)
  : (total_spent - pencil_cartons * pencil_boxes_per_carton * pencil_box_cost) / marker_cartons = 20 := by
  sorry

end marker_carton_cost_l392_39257


namespace cube_and_fifth_power_sum_l392_39252

theorem cube_and_fifth_power_sum (a : ℝ) (h : (a + 1/a)^2 = 11) :
  (a^3 + 1/a^3, a^5 + 1/a^5) = (8 * Real.sqrt 11, 71 * Real.sqrt 11) ∨
  (a^3 + 1/a^3, a^5 + 1/a^5) = (-8 * Real.sqrt 11, -71 * Real.sqrt 11) := by
  sorry

end cube_and_fifth_power_sum_l392_39252


namespace equation_solution_l392_39229

theorem equation_solution : 
  ∀ (x y : ℝ), (16 * x^2 + 1) * (y^2 + 1) = 16 * x * y ↔ 
  ((x = 1/4 ∧ y = 1) ∨ (x = -1/4 ∧ y = -1)) :=
by sorry

end equation_solution_l392_39229


namespace modulus_of_complex_number_l392_39236

theorem modulus_of_complex_number (θ : Real) (h : 2 * Real.pi < θ ∧ θ < 3 * Real.pi) :
  Complex.abs (1 - Real.cos θ + Complex.I * Real.sin θ) = -2 * Real.sin (θ / 2) := by
  sorry

end modulus_of_complex_number_l392_39236


namespace h_equals_three_l392_39278

-- Define the quadratic coefficients
variable (a b c : ℝ)

-- Define the condition that ax^2 + bx + c = 3(x - 3)^2 + 9
def quadratic_condition (a b c : ℝ) : Prop :=
  ∀ x, a * x^2 + b * x + c = 3 * (x - 3)^2 + 9

-- Define the transformed quadratic
def transformed_quadratic (a b c : ℝ) (x : ℝ) : ℝ :=
  5 * a * x^2 + 5 * b * x + 5 * c

-- Theorem stating that h = 3 in the transformed quadratic
theorem h_equals_three (a b c : ℝ) 
  (h : quadratic_condition a b c) :
  ∃ (m k : ℝ), ∀ x, transformed_quadratic a b c x = m * (x - 3)^2 + k :=
sorry

end h_equals_three_l392_39278


namespace three_number_sum_l392_39254

theorem three_number_sum (A B C : ℝ) (h1 : A/B = 2/3) (h2 : B/C = 5/8) (h3 : B = 30) :
  A + B + C = 98 := by
sorry

end three_number_sum_l392_39254


namespace initial_markup_percentage_l392_39249

/-- Given a shirt with an initial price and a required price increase to achieve
    a 100% markup, calculate the initial markup percentage. -/
theorem initial_markup_percentage
  (initial_price : ℝ)
  (price_increase : ℝ)
  (h1 : initial_price = 27)
  (h2 : price_increase = 3)
  (h3 : initial_price + price_increase = 2 * (initial_price - (initial_price - (initial_price / (1 + 1))))): 
  (initial_price - (initial_price / (1 + 1))) / (initial_price / (1 + 1)) * 100 = 80 :=
by sorry

end initial_markup_percentage_l392_39249


namespace quadratic_has_real_roots_root_condition_implies_value_minimum_value_of_y_l392_39223

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - (2*k + 2)*x + 4*k

-- Part 1: Prove that the equation always has real roots
theorem quadratic_has_real_roots (k : ℝ) :
  ∃ x : ℝ, quadratic k x = 0 := by sorry

-- Part 2: Given the condition on roots, find the value of the expression
theorem root_condition_implies_value (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : quadratic k x₁ = 0) (h2 : quadratic k x₂ = 0)
  (h3 : x₂/x₁ + x₁/x₂ - 2 = 0) :
  (1 + 4/(k^2 - 4)) * ((k + 2)/k) = -1 := by sorry

-- Part 3: Find the minimum value of y
theorem minimum_value_of_y (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : quadratic k x₁ = 0) (h2 : quadratic k x₂ = 0)
  (h3 : x₁ > x₂) (h4 : k < 1/2) :
  ∃ y_min : ℝ, y_min = 3/4 ∧ ∀ y : ℝ, y ≥ y_min → ∃ x₂ : ℝ, y = x₂^2 - k*x₁ + 1 := by sorry

end quadratic_has_real_roots_root_condition_implies_value_minimum_value_of_y_l392_39223


namespace randy_bats_count_l392_39201

theorem randy_bats_count :
  ∀ (gloves bats : ℕ),
    gloves = 29 →
    gloves = 7 * bats + 1 →
    bats = 4 :=
by
  sorry

end randy_bats_count_l392_39201


namespace equation_solutions_l392_39232

theorem equation_solutions (x : ℝ) : 
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 12) ↔ 
  (x = 10 ∨ x = -1) := by
sorry

end equation_solutions_l392_39232


namespace no_integer_solution_l392_39239

theorem no_integer_solution (n : ℝ) (hn : n ≠ 0) :
  ¬ ∃ z : ℤ, n / (z : ℝ) = n / ((z : ℝ) + 1) + n / ((z : ℝ) + 25) :=
sorry

end no_integer_solution_l392_39239


namespace school_paintable_area_l392_39261

/-- Represents the dimensions of a classroom -/
structure ClassroomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total wall area to be painted in all classrooms -/
def totalPaintableArea (dimensions : ClassroomDimensions) (numClassrooms : ℕ) (unpaintableArea : ℝ) : ℝ :=
  let wallArea := 2 * (dimensions.length * dimensions.height + dimensions.width * dimensions.height)
  let paintableArea := wallArea - unpaintableArea
  numClassrooms * paintableArea

/-- Theorem stating the total paintable area for the given school -/
theorem school_paintable_area :
  let dimensions : ClassroomDimensions := ⟨15, 12, 10⟩
  let numClassrooms : ℕ := 4
  let unpaintableArea : ℝ := 80
  totalPaintableArea dimensions numClassrooms unpaintableArea = 1840 := by
  sorry

#check school_paintable_area

end school_paintable_area_l392_39261


namespace simplify_polynomial_l392_39268

theorem simplify_polynomial (x : ℝ) : 
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 = -x^2 + 23*x - 3 := by
  sorry

end simplify_polynomial_l392_39268


namespace larger_number_problem_l392_39237

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 10) : x = 30 := by
  sorry

end larger_number_problem_l392_39237


namespace binomial_coefficient_20_10_l392_39243

theorem binomial_coefficient_20_10 
  (h1 : Nat.choose 18 8 = 43758)
  (h2 : Nat.choose 18 9 = 48620)
  (h3 : Nat.choose 18 10 = 43758) :
  Nat.choose 20 10 = 184756 := by
sorry

end binomial_coefficient_20_10_l392_39243


namespace two_digit_number_property_l392_39248

theorem two_digit_number_property : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  n = 3 * ((n / 10) + (n % 10)) :=
by
  -- The proof would go here
  sorry

end two_digit_number_property_l392_39248


namespace stamp_purchase_problem_l392_39235

theorem stamp_purchase_problem :
  ∀ (x y z : ℕ),
  (x : ℤ) + 2 * y + 5 * z = 100 →  -- Total cost in cents
  y = 10 * x →                    -- Relation between 1-cent and 2-cent stamps
  x > 0 ∧ y > 0 ∧ z > 0 →         -- All stamp quantities are positive
  x = 5 ∧ y = 50 ∧ z = 8 :=
by
  sorry

end stamp_purchase_problem_l392_39235


namespace max_value_of_reciprocal_sum_l392_39273

theorem max_value_of_reciprocal_sum (t q r₁ r₂ : ℝ) : 
  (∀ x, x^2 - t*x + q = 0 ↔ x = r₁ ∨ x = r₂) →
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 2010 → r₁ + r₂ = r₁^n + r₂^n) →
  (∃ M : ℝ, M = (1 : ℝ) / r₁^2010 + (1 : ℝ) / r₂^2010 ∧ 
   ∀ t' q' r₁' r₂' : ℝ, 
     (∀ x, x^2 - t'*x + q' = 0 ↔ x = r₁' ∨ x = r₂') →
     (∀ n : ℕ, n ≥ 1 ∧ n ≤ 2010 → r₁' + r₂' = r₁'^n + r₂'^n) →
     (1 : ℝ) / r₁'^2010 + (1 : ℝ) / r₂'^2010 ≤ M) →
  M = 2 := by
sorry

end max_value_of_reciprocal_sum_l392_39273


namespace triangle_count_on_circle_l392_39294

theorem triangle_count_on_circle (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) : 
  Nat.choose n k = 120 := by
  sorry

end triangle_count_on_circle_l392_39294


namespace power_fraction_equality_l392_39241

theorem power_fraction_equality : (7^14 : ℕ) / (49^6 : ℕ) = 49 := by sorry

end power_fraction_equality_l392_39241


namespace four_students_three_activities_l392_39228

/-- The number of different sign-up methods for students choosing activities -/
def signUpMethods (numStudents : ℕ) (numActivities : ℕ) : ℕ :=
  numActivities ^ numStudents

/-- Theorem: Four students signing up for three activities, with each student
    choosing exactly one activity, results in 81 different sign-up methods -/
theorem four_students_three_activities :
  signUpMethods 4 3 = 81 := by
  sorry

end four_students_three_activities_l392_39228


namespace power_difference_square_equals_42_times_10_to_1007_l392_39291

theorem power_difference_square_equals_42_times_10_to_1007 :
  (3^1006 + 7^1007)^2 - (3^1006 - 7^1007)^2 = 42 * 10^1007 := by
  sorry

end power_difference_square_equals_42_times_10_to_1007_l392_39291


namespace inscribed_circle_radius_l392_39234

theorem inscribed_circle_radius (a b c : ℝ) (r : ℝ) : 
  a = 5 → b = 12 → c = 13 →
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area = 1.5 * (a + b + c) - 12 →
  r = 33 / 15 := by
  sorry

end inscribed_circle_radius_l392_39234


namespace age_of_fifteenth_person_l392_39208

theorem age_of_fifteenth_person (total_persons : ℕ) (avg_all : ℕ) (group1_size : ℕ) (avg_group1 : ℕ) (group2_size : ℕ) (avg_group2 : ℕ) :
  total_persons = 20 →
  avg_all = 15 →
  group1_size = 5 →
  avg_group1 = 14 →
  group2_size = 9 →
  avg_group2 = 16 →
  ∃ (age_15th : ℕ), age_15th = 86 ∧
    total_persons * avg_all = group1_size * avg_group1 + group2_size * avg_group2 + age_15th :=
by sorry

end age_of_fifteenth_person_l392_39208


namespace AB_vector_l392_39212

def OA : ℝ × ℝ := (2, 1)
def OB : ℝ × ℝ := (-3, 4)

theorem AB_vector : 
  let AB := (OB.1 - OA.1, OB.2 - OA.2)
  AB = (-5, 3) := by sorry

end AB_vector_l392_39212


namespace problem_solution_l392_39260

-- Define the conditions p and q
def p (x : ℝ) : Prop := 4 * x^2 + 12 * x - 7 ≤ 0
def q (a x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

theorem problem_solution :
  (∃ x : ℝ, p x ∧ ¬(q 0 x) ∧ -7/2 ≤ x ∧ x < -3) ∧
  (∀ a : ℝ, (∀ x : ℝ, p x → q a x) ↔ -5/2 ≤ a ∧ a ≤ 1/2) := by
  sorry

end problem_solution_l392_39260


namespace sum_of_solutions_quadratic_sum_of_solutions_specific_l392_39210

theorem sum_of_solutions_quadratic (a b c d e : ℝ) : 
  (∀ x, x^2 - a*x - b = c*x + d) → 
  (∃ x₁ x₂, x₁^2 - a*x₁ - b = c*x₁ + d ∧ 
            x₂^2 - a*x₂ - b = c*x₂ + d ∧ 
            x₁ ≠ x₂) →
  (x₁ + x₂ = a + c) :=
by sorry

-- Specific instance
theorem sum_of_solutions_specific : 
  (∀ x, x^2 - 6*x - 8 = 4*x + 20) → 
  (∃ x₁ x₂, x₁^2 - 6*x₁ - 8 = 4*x₁ + 20 ∧ 
            x₂^2 - 6*x₂ - 8 = 4*x₂ + 20 ∧ 
            x₁ ≠ x₂) →
  (x₁ + x₂ = 10) :=
by sorry

end sum_of_solutions_quadratic_sum_of_solutions_specific_l392_39210


namespace simplify_and_evaluate_l392_39264

theorem simplify_and_evaluate (a : ℚ) (h : a = -3) : 
  (a - 2) / ((1 + 2*a + a^2) * (a - 3*a/(a+1))) = 1/6 := by
  sorry

end simplify_and_evaluate_l392_39264


namespace jellybean_probability_l392_39275

/-- The number of jellybean colors -/
def num_colors : ℕ := 5

/-- The number of jellybeans in the sample -/
def sample_size : ℕ := 5

/-- The probability of selecting exactly 2 distinct colors when randomly choosing
    5 jellybeans from a set of 5 equally proportioned colors -/
theorem jellybean_probability : 
  (num_colors.choose 2 * (2^sample_size - 2)) / (num_colors^sample_size) = 12/125 := by
  sorry

end jellybean_probability_l392_39275


namespace parabolas_cyclic_quadrilateral_l392_39240

/-- A parabola in the xy-plane --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ → Prop

/-- Two parabolas have perpendicular axes --/
def perpendicular_axes (p1 p2 : Parabola) : Prop := sorry

/-- Two parabolas intersect at four distinct points --/
def four_distinct_intersections (p1 p2 : Parabola) : Prop := sorry

/-- Four points in the plane form a cyclic quadrilateral --/
def cyclic_quadrilateral (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

/-- The main theorem --/
theorem parabolas_cyclic_quadrilateral (p1 p2 : Parabola) :
  perpendicular_axes p1 p2 →
  four_distinct_intersections p1 p2 →
  ∃ q1 q2 q3 q4 : ℝ × ℝ,
    (p1.eq q1.1 q1.2 ∧ p2.eq q1.1 q1.2) ∧
    (p1.eq q2.1 q2.2 ∧ p2.eq q2.1 q2.2) ∧
    (p1.eq q3.1 q3.2 ∧ p2.eq q3.1 q3.2) ∧
    (p1.eq q4.1 q4.2 ∧ p2.eq q4.1 q4.2) ∧
    cyclic_quadrilateral q1 q2 q3 q4 :=
by sorry

end parabolas_cyclic_quadrilateral_l392_39240


namespace line_tangent_to_ellipse_l392_39256

theorem line_tangent_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = x + m ∧ x^2 / 2 + y^2 = 1 → 
    ∃! p : ℝ × ℝ, p.1^2 / 2 + p.2^2 = 1 ∧ p.2 = p.1 + m) ↔ 
  m = Real.sqrt 3 ∨ m = -Real.sqrt 3 := by
sorry

end line_tangent_to_ellipse_l392_39256


namespace smallest_among_three_l392_39230

theorem smallest_among_three : 
  min ((-2)^3) (min (-3^2) (-(-1))) = -3^2 :=
sorry

end smallest_among_three_l392_39230


namespace stratified_sampling_female_count_l392_39203

theorem stratified_sampling_female_count 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (female_employees : ℕ) 
  (male_sampled : ℕ) :
  total_employees = 140 →
  male_employees = 80 →
  female_employees = 60 →
  male_sampled = 16 →
  (female_employees : ℚ) * (male_sampled : ℚ) / (male_employees : ℚ) = 12 :=
by sorry

end stratified_sampling_female_count_l392_39203


namespace triangle_sides_max_sum_squares_l392_39213

theorem triangle_sides_max_sum_squares (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  (1/2) * c^2 = (1/2) * a * b * Real.sin C →
  a * b = Real.sqrt 2 →
  ∃ (max : ℝ), max = 4 ∧ ∀ (a' b' c' : ℝ),
    a' > 0 → b' > 0 → c' > 0 →
    (1/2) * c'^2 = (1/2) * a' * b' * Real.sin C →
    a' * b' = Real.sqrt 2 →
    a'^2 + b'^2 + c'^2 ≤ max :=
by sorry

end triangle_sides_max_sum_squares_l392_39213


namespace daniel_added_four_eggs_l392_39263

/-- The number of eggs Daniel put in the box -/
def eggs_added (initial final : ℕ) : ℕ := final - initial

/-- Proof that Daniel put 4 eggs in the box -/
theorem daniel_added_four_eggs (initial final : ℕ) 
  (h1 : initial = 7) 
  (h2 : final = 11) : 
  eggs_added initial final = 4 := by
  sorry

end daniel_added_four_eggs_l392_39263


namespace sequence_constant_iff_perfect_square_l392_39298

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sequence a_k defined recursively -/
def sequenceA (A : ℕ) : ℕ → ℕ
  | 0 => A
  | k + 1 => sequenceA A k + sumOfDigits (sequenceA A k)

/-- A number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- The sequence eventually becomes constant -/
def eventuallyConstant (A : ℕ) : Prop := ∃ N : ℕ, ∀ k ≥ N, sequenceA A k = sequenceA A N

/-- Main theorem -/
theorem sequence_constant_iff_perfect_square (A : ℕ) :
  eventuallyConstant A ↔ isPerfectSquare A := by sorry

end sequence_constant_iff_perfect_square_l392_39298


namespace coefficient_x_fourth_power_l392_39296

theorem coefficient_x_fourth_power (x : ℝ) : 
  (Finset.range 11).sum (fun k => (-1)^k * Nat.choose 10 k * x^(10 - 2*k)) = -120 * x^4 + 
    (Finset.range 11).sum (fun k => if k ≠ 3 then (-1)^k * Nat.choose 10 k * x^(10 - 2*k) else 0) := by
  sorry

end coefficient_x_fourth_power_l392_39296


namespace article_cost_l392_39271

/-- Proves that the cost of an article is 50 Rs given the profit conditions -/
theorem article_cost (original_profit : Real) (reduced_cost_percentage : Real) 
  (price_reduction : Real) (new_profit : Real) :
  original_profit = 0.25 →
  reduced_cost_percentage = 0.20 →
  price_reduction = 10.50 →
  new_profit = 0.30 →
  ∃ (cost : Real), cost = 50 ∧
    (cost + original_profit * cost) - price_reduction = 
    (cost - reduced_cost_percentage * cost) + new_profit * (cost - reduced_cost_percentage * cost) :=
by sorry

end article_cost_l392_39271


namespace decimal_equivalent_of_half_squared_l392_39206

theorem decimal_equivalent_of_half_squared : (1 / 2 : ℚ) ^ 2 = 0.25 := by
  sorry

end decimal_equivalent_of_half_squared_l392_39206


namespace variance_unchanged_by_constant_shift_l392_39289

def ages : List ℝ := [15, 13, 15, 14, 13]
def variance (xs : List ℝ) : ℝ := sorry

theorem variance_unchanged_by_constant_shift (c : ℝ) :
  variance ages = variance (ages.map (· + c)) :=
by sorry

end variance_unchanged_by_constant_shift_l392_39289


namespace square_sum_ge_twice_product_l392_39299

theorem square_sum_ge_twice_product (x y : ℝ) : x^2 + y^2 ≥ 2*x*y := by
  sorry

end square_sum_ge_twice_product_l392_39299


namespace additional_rook_possible_l392_39200

/-- Represents a 10x10 chessboard -/
def Board := Fin 10 → Fin 10 → Bool

/-- Checks if a rook at position (x, y) attacks another rook at position (x', y') -/
def attacks (x y x' y' : Fin 10) : Prop :=
  x = x' ∨ y = y'

/-- Represents a valid rook placement on the board -/
def ValidPlacement (b : Board) : Prop :=
  ∃ (n : Nat) (positions : Fin n → Fin 10 × Fin 10),
    n ≤ 8 ∧
    (∀ i j, i ≠ j → ¬attacks (positions i).1 (positions i).2 (positions j).1 (positions j).2) ∧
    (∀ i, b (positions i).1 (positions i).2 = true) ∧
    (∃ (blackCount whiteCount : Nat),
      blackCount = whiteCount ∧
      blackCount + whiteCount = n ∧
      (∀ i, (((positions i).1 + (positions i).2) % 2 = 0) = (i < blackCount)))

/-- The main theorem stating that an additional rook can be placed -/
theorem additional_rook_possible (b : Board) (h : ValidPlacement b) :
  ∃ (x y : Fin 10), b x y = false ∧ ∀ (x' y' : Fin 10), b x' y' = true → ¬attacks x y x' y' := by
  sorry

end additional_rook_possible_l392_39200


namespace point_on_curve_l392_39290

/-- The curve C defined by y = x^3 - 10x + 3 -/
def C : ℝ → ℝ := λ x ↦ x^3 - 10*x + 3

/-- The derivative of curve C -/
def C' : ℝ → ℝ := λ x ↦ 3*x^2 - 10

theorem point_on_curve (x y : ℝ) :
  x < 0 →  -- P is in the second quadrant (x < 0)
  y > 0 →  -- P is in the second quadrant (y > 0)
  y = C x →  -- P lies on the curve C
  C' x = 2 →  -- The slope of the tangent line at P is 2
  x = -2 ∧ y = 15 := by  -- P has coordinates (-2, 15)
  sorry

end point_on_curve_l392_39290


namespace smaller_number_proof_l392_39284

theorem smaller_number_proof (x y : ℝ) (h1 : x + y = 44) (h2 : 5 * x = 6 * y) : min x y = 20 := by
  sorry

end smaller_number_proof_l392_39284


namespace storage_b_has_five_pieces_l392_39244

/-- Represents a storage device with a number of data pieces -/
structure StorageDevice :=
  (pieces : ℕ)

/-- Represents the state of three storage devices A, B, and C -/
structure StorageState :=
  (A : StorageDevice)
  (B : StorageDevice)
  (C : StorageDevice)

/-- Performs the described operations on the storage devices -/
def performOperations (n : ℕ) (initial : StorageState) : StorageState :=
  { A := ⟨2 * (n - 2)⟩,
    B := ⟨n + 3 - (n - 2)⟩,
    C := ⟨n - 1⟩ }

/-- The theorem stating that after the operations, storage device B has 5 data pieces -/
theorem storage_b_has_five_pieces (n : ℕ) (h : n ≥ 2) :
  (performOperations n { A := ⟨0⟩, B := ⟨0⟩, C := ⟨0⟩ }).B.pieces = 5 := by
  sorry

#check storage_b_has_five_pieces

end storage_b_has_five_pieces_l392_39244


namespace starting_lineup_count_l392_39285

/-- Represents a football team -/
structure FootballTeam where
  total_members : ℕ
  offensive_linemen : ℕ
  hm : offensive_linemen ≤ total_members

/-- Calculates the number of ways to choose a starting lineup -/
def starting_lineup_combinations (team : FootballTeam) : ℕ :=
  team.offensive_linemen * (team.total_members - 1) * (team.total_members - 2) * (team.total_members - 3)

/-- Theorem stating the number of ways to choose a starting lineup for the given team -/
theorem starting_lineup_count (team : FootballTeam) 
  (h1 : team.total_members = 12) 
  (h2 : team.offensive_linemen = 4) : 
  starting_lineup_combinations team = 3960 := by
  sorry

#eval starting_lineup_combinations ⟨12, 4, by norm_num⟩

end starting_lineup_count_l392_39285


namespace upper_bound_y_l392_39259

theorem upper_bound_y (x y : ℤ) (U : ℤ) : 
  (3 < x ∧ x < 6) → 
  (6 < y ∧ y < U) → 
  (∀ (x' y' : ℤ), (3 < x' ∧ x' < 6) → (6 < y' ∧ y' < U) → y' - x' ≤ 4) →
  (∃ (x' y' : ℤ), (3 < x' ∧ x' < 6) ∧ (6 < y' ∧ y' < U) ∧ y' - x' = 4) →
  U = 10 :=
by sorry

end upper_bound_y_l392_39259


namespace complex_power_one_minus_i_six_l392_39245

theorem complex_power_one_minus_i_six :
  (1 - Complex.I) ^ 6 = 8 * Complex.I := by sorry

end complex_power_one_minus_i_six_l392_39245


namespace disneyland_attractions_ordering_l392_39269

def number_of_attractions : ℕ := 6

theorem disneyland_attractions_ordering :
  let total_permutations := Nat.factorial number_of_attractions
  let valid_permutations := total_permutations / 2
  valid_permutations = 360 :=
by sorry

end disneyland_attractions_ordering_l392_39269


namespace interest_calculation_years_l392_39226

/-- Calculates the number of years for a given interest scenario -/
def calculate_years (principal : ℝ) (rate : ℝ) (interest_difference : ℝ) : ℝ :=
  let f : ℝ → ℝ := λ n => (1 + rate)^n - 1 - rate * n - interest_difference / principal
  -- We assume the existence of a root-finding function
  sorry

theorem interest_calculation_years :
  let principal : ℝ := 1300
  let rate : ℝ := 0.10
  let interest_difference : ℝ := 13
  calculate_years principal rate interest_difference = 2 := by
  sorry

end interest_calculation_years_l392_39226


namespace polynomial_division_theorem_l392_39251

theorem polynomial_division_theorem (x : ℚ) :
  (4 * x^2 - 4/3 * x + 2) * (3 * x + 4) + 10/3 = 12 * x^3 + 24 * x^2 - 10 * x + 6 := by
  sorry

end polynomial_division_theorem_l392_39251


namespace museum_trip_l392_39286

def bus_trip (first_bus : ℕ) : Prop :=
  let second_bus := 2 * first_bus
  let third_bus := second_bus - 6
  let fourth_bus := first_bus + 9
  let total_people := first_bus + second_bus + third_bus + fourth_bus
  (first_bus ≤ 45) ∧ 
  (second_bus ≤ 45) ∧ 
  (third_bus ≤ 45) ∧ 
  (fourth_bus ≤ 45) ∧ 
  (total_people = 75)

theorem museum_trip : bus_trip 12 := by
  sorry

end museum_trip_l392_39286


namespace sector_area_l392_39283

/-- The area of a circular sector with central angle 3/4π and radius 4 is 6π. -/
theorem sector_area : 
  let central_angle : Real := 3/4 * Real.pi
  let radius : Real := 4
  let sector_area : Real := 1/2 * central_angle * radius^2
  sector_area = 6 * Real.pi := by sorry

end sector_area_l392_39283


namespace newspaper_pages_l392_39218

/-- Represents a newspaper with a certain number of pages -/
structure Newspaper where
  num_pages : ℕ

/-- Predicate indicating that two pages are on the same sheet -/
def on_same_sheet (n : Newspaper) (p1 p2 : ℕ) : Prop :=
  p1 ≤ n.num_pages ∧ p2 ≤ n.num_pages ∧ p1 + p2 = n.num_pages + 1

/-- The theorem stating the number of pages in the newspaper -/
theorem newspaper_pages : 
  ∃ (n : Newspaper), n.num_pages = 28 ∧ on_same_sheet n 8 21 := by
  sorry

end newspaper_pages_l392_39218


namespace magic_square_sum_divisible_by_three_l392_39242

/-- Represents a 3x3 magic square -/
def MagicSquare : Type := Fin 3 → Fin 3 → ℕ

/-- The sum of a row, column, or diagonal in a magic square -/
def magic_sum (square : MagicSquare) : ℕ := square 0 0 + square 0 1 + square 0 2

/-- Predicate to check if a square is a valid magic square -/
def is_magic_square (square : MagicSquare) : Prop :=
  (∀ i : Fin 3, square i 0 + square i 1 + square i 2 = magic_sum square) ∧
  (∀ j : Fin 3, square 0 j + square 1 j + square 2 j = magic_sum square) ∧
  (square 0 0 + square 1 1 + square 2 2 = magic_sum square) ∧
  (square 0 2 + square 1 1 + square 2 0 = magic_sum square)

theorem magic_square_sum_divisible_by_three (square : MagicSquare) 
  (h : is_magic_square square) : 
  ∃ k : ℕ, magic_sum square = 3 * k := by
  sorry

end magic_square_sum_divisible_by_three_l392_39242


namespace complex_arithmetic_equality_l392_39205

theorem complex_arithmetic_equality : (9 - 8 + 7)^2 * 6 + 5 - 4^2 * 3 + 2^3 - 1 = 347 := by
  sorry

end complex_arithmetic_equality_l392_39205


namespace composite_sum_of_squares_l392_39221

theorem composite_sum_of_squares (a b : ℤ) : 
  (∃ x y : ℕ, x^2 + a*x + b + 1 = 0 ∧ y^2 + a*y + b + 1 = 0 ∧ x ≠ y) → 
  ∃ m n : ℕ, m > 1 ∧ n > 1 ∧ a^2 + b^2 = m * n :=
by sorry

end composite_sum_of_squares_l392_39221


namespace ferris_wheel_capacity_l392_39281

theorem ferris_wheel_capacity (num_seats : ℕ) (people_per_seat : ℕ) 
  (h1 : num_seats = 4) (h2 : people_per_seat = 5) :
  num_seats * people_per_seat = 20 := by
  sorry

end ferris_wheel_capacity_l392_39281


namespace geometric_sequence_ratio_l392_39267

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  prop1 : a 2 * a 6 = 16
  prop2 : a 4 + a 8 = 8

/-- The main theorem -/
theorem geometric_sequence_ratio (seq : GeometricSequence) : seq.a 20 / seq.a 10 = 1 := by
  sorry

end geometric_sequence_ratio_l392_39267


namespace point_on_line_l392_39287

/-- A line passing through point (1,3) with slope 2 -/
def line_l (b : ℝ) : ℝ → ℝ := λ x ↦ 2 * x + b

/-- The y-coordinate of point P -/
def point_p_y : ℝ := 3

/-- The x-coordinate of point P -/
def point_p_x : ℝ := 1

/-- The y-coordinate of point Q -/
def point_q_y : ℝ := 5

/-- The x-coordinate of point Q -/
def point_q_x : ℝ := 2

theorem point_on_line :
  ∃ b : ℝ, line_l b point_p_x = point_p_y ∧ line_l b point_q_x = point_q_y := by
  sorry

end point_on_line_l392_39287


namespace stream_speed_l392_39277

/-- Given a canoe's upstream and downstream speeds, prove the speed of the stream -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 3)
  (h_downstream : downstream_speed = 12) :
  ∃ (canoe_speed stream_speed : ℝ),
    canoe_speed - stream_speed = upstream_speed ∧
    canoe_speed + stream_speed = downstream_speed ∧
    stream_speed = 4.5 := by
  sorry

end stream_speed_l392_39277


namespace amber_guppies_l392_39282

/-- The number of guppies in Amber's pond -/
theorem amber_guppies (initial_adults : ℕ) (first_batch_dozens : ℕ) (second_batch : ℕ) :
  initial_adults + (first_batch_dozens * 12) + second_batch =
  initial_adults + first_batch_dozens * 12 + second_batch :=
by sorry

end amber_guppies_l392_39282


namespace half_area_of_rectangle_l392_39266

/-- Half the area of a rectangle with width 25 cm and height 16 cm is 200 cm². -/
theorem half_area_of_rectangle (width height : ℝ) (h1 : width = 25) (h2 : height = 16) :
  (width * height) / 2 = 200 := by
  sorry

end half_area_of_rectangle_l392_39266


namespace inequality_system_solution_l392_39274

theorem inequality_system_solution (x : ℝ) :
  (3 * x - 1 > x + 1 ∧ (4 * x - 5) / 3 ≤ x) ↔ (1 < x ∧ x ≤ 5) := by
  sorry

end inequality_system_solution_l392_39274


namespace badge_ratio_l392_39247

/-- Proves that the ratio of delegates who made their own badges to delegates without pre-printed badges is 1:2 -/
theorem badge_ratio (total : ℕ) (pre_printed : ℕ) (no_badge : ℕ) 
  (h1 : total = 36)
  (h2 : pre_printed = 16)
  (h3 : no_badge = 10) : 
  (total - pre_printed - no_badge) / (total - pre_printed) = 1 / 2 := by
  sorry

end badge_ratio_l392_39247


namespace terminal_side_negative_pi_in_fourth_quadrant_l392_39222

/-- The terminal side of -π radians lies in the fourth quadrant -/
theorem terminal_side_negative_pi_in_fourth_quadrant :
  let angle : ℝ := -π
  (angle > -2*π ∧ angle ≤ -3*π/2) ∨ (angle > 3*π/2 ∧ angle ≤ 2*π) :=
by sorry

end terminal_side_negative_pi_in_fourth_quadrant_l392_39222


namespace unique_function_solution_l392_39262

theorem unique_function_solution (f : ℝ → ℝ) : 
  (∀ x ≥ 1, f x ≥ 1) → 
  (∀ x ≥ 1, f x ≤ 2 * (x + 1)) → 
  (∀ x ≥ 1, f (x + 1) = (1 / x) * ((f x)^2 - 1)) → 
  (∀ x ≥ 1, f x = x + 1) := by
sorry

end unique_function_solution_l392_39262


namespace woman_completes_in_40_days_l392_39225

-- Define the efficiency ratio between man and woman
def efficiency_ratio : ℝ := 1.25

-- Define the number of days it takes the man to complete the task
def man_days : ℝ := 32

-- Define the function to calculate the woman's days
def woman_days : ℝ := efficiency_ratio * man_days

-- Theorem to prove
theorem woman_completes_in_40_days : 
  woman_days = 40 := by sorry

end woman_completes_in_40_days_l392_39225


namespace five_balls_three_boxes_l392_39217

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 3^5 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes :
  ways_to_put_balls_in_boxes 5 3 = 3^5 := by
  sorry

end five_balls_three_boxes_l392_39217


namespace lisa_children_count_l392_39246

/-- The number of Lisa's children -/
def num_children : ℕ := 4

/-- The number of spoons in the new cutlery set -/
def new_cutlery_spoons : ℕ := 25

/-- The number of decorative spoons -/
def decorative_spoons : ℕ := 2

/-- The number of baby spoons per child -/
def baby_spoons_per_child : ℕ := 3

/-- The total number of spoons Lisa has -/
def total_spoons : ℕ := 39

/-- Theorem stating that the number of Lisa's children is 4 -/
theorem lisa_children_count : 
  num_children * baby_spoons_per_child + new_cutlery_spoons + decorative_spoons = total_spoons :=
by sorry

end lisa_children_count_l392_39246


namespace unique_bases_sum_l392_39224

def recurring_decimal (a b : ℕ) (base : ℕ) : ℚ :=
  (a : ℚ) / (base ^ 2 - 1 : ℚ) * base + (b : ℚ) / (base ^ 2 - 1 : ℚ)

theorem unique_bases_sum :
  ∃! (R₁ R₂ : ℕ), 
    R₁ > 1 ∧ R₂ > 1 ∧
    recurring_decimal 3 7 R₁ = recurring_decimal 2 5 R₂ ∧
    recurring_decimal 7 3 R₁ = recurring_decimal 5 2 R₂ ∧
    R₁ + R₂ = 19 :=
by sorry

end unique_bases_sum_l392_39224


namespace digit_equation_sum_l392_39280

theorem digit_equation_sum (A B C D U : ℕ) : 
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ U) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ U) ∧
  (C ≠ D) ∧ (C ≠ U) ∧
  (D ≠ U) ∧
  (A < 10) ∧ (B < 10) ∧ (C < 10) ∧ (D < 10) ∧ (U < 10) ∧ (U > 0) ∧
  ((10 * A + B) * (10 * C + D) = 111 * U) →
  A + B + C + D + U = 17 := by
sorry

end digit_equation_sum_l392_39280


namespace max_value_sum_sqrt_l392_39233

theorem max_value_sum_sqrt (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 8) : 
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) ≤ 3 * Real.sqrt 10 := by
  sorry

end max_value_sum_sqrt_l392_39233


namespace f_properties_l392_39214

noncomputable def f (x : ℝ) : ℝ := (1 / (2^x - 1) + 1/2) * x^3

theorem f_properties :
  (∀ x : ℝ, x ≠ 0 → f x ≠ 0) ∧
  (∀ x : ℝ, x ≠ 0 → f (-x) = f x) ∧
  (∀ x : ℝ, x ≠ 0 → f x > 0) :=
sorry

end f_properties_l392_39214


namespace equation_solution_l392_39276

theorem equation_solution : ∃! x : ℚ, 3 * x - 5 = |-20 + 6| := by sorry

end equation_solution_l392_39276


namespace oil_price_reduction_l392_39202

/-- Proves that given a 40% reduction in oil price, if 8 kg more oil can be bought for Rs. 2400 after the reduction, then the reduced price per kg is Rs. 120. -/
theorem oil_price_reduction (original_price : ℝ) : 
  let reduced_price := original_price * 0.6
  let original_quantity := 2400 / original_price
  let new_quantity := 2400 / reduced_price
  (new_quantity - original_quantity = 8) → reduced_price = 120 := by
  sorry


end oil_price_reduction_l392_39202


namespace max_value_of_a_l392_39211

def f (x a : ℝ) : ℝ := |8 * x^3 - 12 * x - a| + a

theorem max_value_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x a ≤ 0) ∧ (∃ x ∈ Set.Icc 0 1, f x a = 0) →
  a ≤ -2 * Real.sqrt 2 :=
by sorry

end max_value_of_a_l392_39211


namespace fraction_equality_l392_39295

theorem fraction_equality (a b : ℝ) : (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) := by
  sorry

end fraction_equality_l392_39295


namespace train_speed_problem_l392_39209

theorem train_speed_problem (V₁ V₂ : ℝ) (h₁ : V₁ > 0) (h₂ : V₂ > 0) (h₃ : V₂ > V₁) : 
  (∃ t : ℝ, t > 0 ∧ t * (V₁ + V₂) = 2400) ∧
  (∃ t : ℝ, t > 0 ∧ 2 * V₂ * (t - 3) = 2400) ∧
  (∃ t : ℝ, t > 0 ∧ 2 * V₁ * (t + 5) = 2400) →
  V₁ = 60 ∧ V₂ = 100 := by
sorry

end train_speed_problem_l392_39209


namespace coffee_mix_solution_l392_39238

/-- Represents the coffee mix problem -/
structure CoffeeMix where
  total_mix : ℝ
  columbian_price : ℝ
  brazilian_price : ℝ
  ethiopian_price : ℝ
  mix_price : ℝ
  ratio_columbian : ℝ
  ratio_brazilian : ℝ
  ratio_ethiopian : ℝ

/-- Theorem stating the correct amounts of each coffee type -/
theorem coffee_mix_solution (mix : CoffeeMix)
  (h_total : mix.total_mix = 150)
  (h_columbian_price : mix.columbian_price = 9.5)
  (h_brazilian_price : mix.brazilian_price = 4.25)
  (h_ethiopian_price : mix.ethiopian_price = 7.25)
  (h_mix_price : mix.mix_price = 6.7)
  (h_ratio : mix.ratio_columbian = 2 ∧ mix.ratio_brazilian = 3 ∧ mix.ratio_ethiopian = 5) :
  ∃ (columbian brazilian ethiopian : ℝ),
    columbian = 30 ∧
    brazilian = 45 ∧
    ethiopian = 75 ∧
    columbian + brazilian + ethiopian = mix.total_mix ∧
    columbian / mix.ratio_columbian = brazilian / mix.ratio_brazilian ∧
    columbian / mix.ratio_columbian = ethiopian / mix.ratio_ethiopian :=
by
  sorry


end coffee_mix_solution_l392_39238


namespace find_two_fake_coins_l392_39270

/-- Represents the state of our coin testing process -/
structure CoinState where
  total : Nat
  fake : Nat
  deriving Repr

/-- Represents the result of a test -/
inductive TestResult
  | Signal
  | NoSignal
  deriving Repr

/-- A function that simulates a test -/
def test (coins : Nat) (state : CoinState) : TestResult := sorry

/-- A function that updates the state based on a test result -/
def updateState (coins : Nat) (state : CoinState) (result : TestResult) : CoinState := sorry

/-- A function that represents a single step in our testing strategy -/
def testStep (state : CoinState) : CoinState := sorry

/-- The main theorem stating that we can find two fake coins in five steps -/
theorem find_two_fake_coins 
  (initial_state : CoinState) 
  (h1 : initial_state.total = 49) 
  (h2 : initial_state.fake = 24) : 
  ∃ (final_state : CoinState), 
    (final_state.total = 2 ∧ final_state.fake = 2) ∧ 
    (∃ (s1 s2 s3 s4 : CoinState), 
      s1 = testStep initial_state ∧ 
      s2 = testStep s1 ∧ 
      s3 = testStep s2 ∧ 
      s4 = testStep s3 ∧ 
      final_state = testStep s4) :=
sorry

end find_two_fake_coins_l392_39270


namespace marbles_exceed_500_on_day_5_l392_39258

def marble_sequence (n : ℕ) : ℕ := 4^n

theorem marbles_exceed_500_on_day_5 :
  ∀ k : ℕ, k < 5 → marble_sequence k ≤ 500 ∧ marble_sequence 5 > 500 :=
by sorry

end marbles_exceed_500_on_day_5_l392_39258


namespace product_mod_seven_l392_39292

theorem product_mod_seven : (2015 * 2016 * 2017 * 2018) % 7 = 3 := by
  sorry

end product_mod_seven_l392_39292


namespace largest_valid_number_l392_39272

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ i, 1 < i ∧ i < digits.length - 1 →
    digits[i]! < (digits[i-1]! + digits[i+1]!) / 2

theorem largest_valid_number :
  (96433469 : ℕ).digits 10 = [9, 6, 4, 3, 3, 4, 6, 9] ∧
  is_valid_number 96433469 ∧
  ∀ m : ℕ, m > 96433469 → ¬ is_valid_number m :=
sorry

end largest_valid_number_l392_39272


namespace blueberry_count_l392_39207

theorem blueberry_count (total : ℕ) (raspberries : ℕ) (blackberries : ℕ) 
  (h1 : total = 42)
  (h2 : raspberries = total / 2)
  (h3 : blackberries = total / 3) :
  total - raspberries - blackberries = 7 := by
sorry

end blueberry_count_l392_39207


namespace exterior_angle_parallel_lines_l392_39215

theorem exterior_angle_parallel_lines (α β γ δ : ℝ) : 
  α = 40 → β = 40 → γ + δ = 180 → α + β + γ = 180 → δ = 80 := by sorry

end exterior_angle_parallel_lines_l392_39215


namespace bicycle_discount_price_l392_39297

theorem bicycle_discount_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  original_price = 200 →
  discount1 = 0.4 →
  discount2 = 0.2 →
  original_price * (1 - discount1) * (1 - discount2) = 96 := by
sorry

end bicycle_discount_price_l392_39297


namespace least_k_correct_l392_39216

/-- Sum of reciprocal values of non-zero digits of all positive integers up to and including n -/
def S (n : ℕ) : ℚ := sorry

/-- The least positive integer k such that k! * S_2016 is an integer -/
def least_k : ℕ := 7

theorem least_k_correct :
  (∀ m : ℕ, m < least_k → ¬(∃ z : ℤ, z = (m.factorial : ℚ) * S 2016)) ∧
  (∃ z : ℤ, z = (least_k.factorial : ℚ) * S 2016) := by sorry

end least_k_correct_l392_39216


namespace boys_average_age_l392_39293

/-- Prove that the average age of boys is 12 years in a school with given conditions -/
theorem boys_average_age
  (total_students : ℕ)
  (girls_count : ℕ)
  (girls_avg_age : ℝ)
  (school_avg_age : ℝ)
  (h1 : total_students = 600)
  (h2 : girls_count = 150)
  (h3 : girls_avg_age = 11)
  (h4 : school_avg_age = 11.75) :
  let boys_count : ℕ := total_students - girls_count
  let boys_total_age : ℝ := school_avg_age * total_students - girls_avg_age * girls_count
  boys_total_age / boys_count = 12 := by
  sorry

end boys_average_age_l392_39293


namespace travel_agency_comparison_l392_39288

/-- Represents the total cost for Travel Agency A -/
def cost_a (x : ℝ) : ℝ := 2 * 500 + 500 * x * 0.7

/-- Represents the total cost for Travel Agency B -/
def cost_b (x : ℝ) : ℝ := (x + 2) * 500 * 0.8

theorem travel_agency_comparison (x : ℝ) :
  (x < 4 → cost_a x > cost_b x) ∧
  (x = 4 → cost_a x = cost_b x) ∧
  (x > 4 → cost_a x < cost_b x) :=
sorry

end travel_agency_comparison_l392_39288


namespace gcd_18_30_l392_39253

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l392_39253
