import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bertolli_farm_specific_difference_l278_27811

/-- The number of fewer onions grown compared to tomatoes and corn together at Bertolli Farm -/
def bertolli_farm_produce_difference (tomatoes corn onions : ℕ) : ℕ :=
  tomatoes + corn - onions

/-- Theorem stating the specific difference for Bertolli Farm -/
theorem bertolli_farm_specific_difference :
  bertolli_farm_produce_difference 2073 4112 985 = 5200 := by
  -- Unfold the definition of bertolli_farm_produce_difference
  unfold bertolli_farm_produce_difference
  -- Evaluate the arithmetic expression
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bertolli_farm_specific_difference_l278_27811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_celias_weekly_food_budget_l278_27865

/-- Celia's monthly budget --/
structure MonthlyBudget where
  rent : ℚ
  videoStreaming : ℚ
  cellPhone : ℚ
  savingsRate : ℚ
  savingsAmount : ℚ
  weeks : ℕ

/-- Calculate Celia's weekly food budget --/
noncomputable def weeklyFoodBudget (budget : MonthlyBudget) : ℚ :=
  let totalSpending := budget.savingsAmount / budget.savingsRate
  let foodBudget := totalSpending - (budget.rent + budget.videoStreaming + budget.cellPhone)
  foodBudget / budget.weeks

/-- Theorem: Celia's weekly food budget is $100 --/
theorem celias_weekly_food_budget :
  let budget : MonthlyBudget := {
    rent := 1500,
    videoStreaming := 30,
    cellPhone := 50,
    savingsRate := 1/10,
    savingsAmount := 198,
    weeks := 4
  }
  weeklyFoodBudget budget = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_celias_weekly_food_budget_l278_27865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l278_27884

noncomputable def g (x : ℝ) : ℝ := 2^x

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (b - g x) / (a + g x)

theorem problem_solution (a b : ℝ) 
  (h1 : g 2 = 4)
  (h2 : ∀ x, f a b x = -f a b (-x)) :
  (∃ a' b', ∀ x, f a' b' x = (1 - 2^x) / (1 + 2^x)) ∧
  (∀ x₁ x₂, x₁ < x₂ → f 1 1 x₁ > f 1 1 x₂) ∧
  (∀ m, (∃ x, x ∈ Set.Icc (-1) 0 ∧ f 1 1 x = m) →
    f 1 1 (1/m) ∈ Set.Ioo (-1) (-7/9)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l278_27884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l278_27837

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_real_range (f : ℝ → ℝ) : Prop :=
  ∀ y, ∃ x, f x = y

noncomputable def power_function (a : ℝ) : ℝ → ℝ :=
  λ x ↦ Real.rpow x a

theorem power_function_properties (a : ℝ) :
  (a ∈ ({-1, 1, 2, 3} : Set ℝ)) →
  (has_real_range (power_function a) ∧ is_odd_function (power_function a)) ↔
  (a = 1 ∨ a = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l278_27837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_zero_l278_27889

theorem det_B_zero (A B : Matrix (Fin 10) (Fin 10) ℝ)
  (h1 : ∀ i j, A i j = B i j + 1)
  (h2 : A ^ 3 = 0) :
  Matrix.det B = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_zero_l278_27889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l278_27839

-- Define the function f
def f (x : ℝ) : ℝ := |1 - 2*x|

-- Define the domain
def domain : Set ℝ := Set.Icc 0 1

-- State the theorem
theorem equation_solutions :
  ∃! (s : Finset ℝ), s.card = 8 ∧ 
  (∀ x ∈ s, x ∈ domain ∧ f (f (f x)) = 1/2 * x) ∧
  (∀ x ∈ domain, f (f (f x)) = 1/2 * x → x ∈ s) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l278_27839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l278_27894

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 - 2*x + 3) / Real.log (1/2)

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y, x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l278_27894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_perpendicular_medians_l278_27840

-- Define a triangle XYZ
structure Triangle (X Y Z : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define a median
noncomputable def median (A B C : ℝ × ℝ) : ℝ × ℝ := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define perpendicularity of two line segments
def perpendicular (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0

-- Define the length of a line segment
noncomputable def length (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- The main theorem
theorem triangle_with_perpendicular_medians 
  (X Y Z : ℝ × ℝ) 
  (h_triangle : Triangle X Y Z)
  (h_perp_medians : perpendicular X (median Y Z X) Y (median X Z Y))
  (h_YZ : length Y Z = 8)
  (h_XZ : length X Z = 10) :
  length X Y = 2 * Real.sqrt 41 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_perpendicular_medians_l278_27840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sloth_feet_count_l278_27869

/-- The number of feet a sloth has -/
def F : ℕ := sorry

/-- The number of pairs of shoes the sloth needs to buy -/
def pairs_to_buy : ℕ := 6

/-- The number of complete sets of shoes the sloth wants to have -/
def total_sets : ℕ := 5

/-- The sloth already owns one set of shoes -/
def existing_sets : ℕ := 1

theorem sloth_feet_count :
  (pairs_to_buy * 2 = (total_sets - existing_sets) * F) ↔ F = 3 := by
  sorry

#check sloth_feet_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sloth_feet_count_l278_27869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l278_27859

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 2) = 5

/-- The first parabola derived from the curve equation -/
def parabola1 (x y : ℝ) : Prop :=
  y = -1/14 * x^2 + 3.5

/-- The second parabola derived from the curve equation -/
def parabola2 (x y : ℝ) : Prop :=
  y = 1/6 * x^2 - 1.5

/-- The vertex of the first parabola -/
def vertex1 : ℝ × ℝ := (0, 3.5)

/-- The vertex of the second parabola -/
def vertex2 : ℝ × ℝ := (0, -1.5)

/-- Theorem stating that the distance between the vertices of the two parabolas is 5 -/
theorem distance_between_vertices : 
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_vertices_l278_27859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_implication_l278_27878

/-- Given two vectors a and b in ℝ², where a depends on an angle θ,
    prove that their dot product being zero implies a specific value for an expression. -/
theorem vector_dot_product_implication (θ : ℝ) :
  let a : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)
  let b : ℝ × ℝ := (1, -6)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  ((2 * Real.cos θ + Real.sin θ) / (Real.cos θ + 3 * Real.sin θ) = 7/6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_implication_l278_27878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l278_27808

/-- A function that returns true if a digit is less than 5 -/
def lessThan5 (d : ℕ) : Bool :=
  d < 5

/-- A function that returns true if a digit is greater than 5 -/
def greaterThan5 (d : ℕ) : Bool :=
  d > 5

/-- A function that checks if two digits satisfy the condition of both being less than 5 or both being greater than 5 -/
def validPair (d1 d2 : ℕ) : Bool :=
  (lessThan5 d1 ∧ lessThan5 d2) ∨ (greaterThan5 d1 ∧ greaterThan5 d2)

/-- A function that checks if a four-digit number satisfies the given conditions -/
def validNumber (n : ℕ) : Bool :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  validPair d1 d2 ∧ validPair d3 d4

/-- The theorem stating that the count of valid four-digit numbers is 1681 -/
theorem count_valid_numbers : 
  (Finset.filter (fun n => validNumber n) (Finset.range 9000)).card = 1681 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l278_27808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l278_27880

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  N.vecMul (![2, -1]) = ![5, -7] ∧
  N.vecMul (![4, 3]) = ![20, 21] ∧
  N = !![3.5, 2; 0, 7] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l278_27880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_100_101_l278_27838

def f (x : ℤ) : ℤ := x^2 - 3*x + 2023

theorem gcd_f_100_101 : Int.gcd (f 100) (f 101) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_100_101_l278_27838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_birthday_group_l278_27849

theorem smallest_n_for_birthday_group : ∃ n : ℕ, 
  (∀ m : ℕ, m ≥ n → ∀ f : Fin (2 * m - 10) → Fin 365, 
    ∃ d : Fin 365, (Finset.filter (λ i => f i = d) Finset.univ).card ≥ 10) ∧ 
  (∀ k : ℕ, k < n → ∃ f : Fin (2 * k - 10) → Fin 365, 
    ∀ d : Fin 365, (Finset.filter (λ i => f i = d) Finset.univ).card < 10) ∧
  n = 1648 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_birthday_group_l278_27849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_cubic_l278_27836

theorem unique_prime_cubic : ∃! n : ℕ+, Nat.Prime (n.val^3 - 7*n.val^2 + 17*n.val - 11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_cubic_l278_27836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_set_symbols_l278_27833

-- Define the symbols as constants
def belongs_to : String := "∈"
def subset_of : String := "⊆"
def empty_set : String := "∅"
def real_numbers : String := "ℝ"

-- Define the theorem
theorem correct_set_symbols :
  (belongs_to = "∈") ∧
  (subset_of = "⊆") ∧
  (empty_set = "∅") ∧
  (real_numbers = "ℝ") := by
  -- Split the goal into individual parts
  apply And.intro
  · rfl  -- Reflexivity proves belongs_to = "∈"
  apply And.intro
  · rfl  -- Reflexivity proves subset_of = "⊆"
  apply And.intro
  · rfl  -- Reflexivity proves empty_set = "∅"
  · rfl  -- Reflexivity proves real_numbers = "ℝ"

#check correct_set_symbols

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_set_symbols_l278_27833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_when_a_is_one_g_range_on_interval_equation_two_roots_iff_l278_27828

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.sin x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a * Real.log x

-- Part 1
theorem f_nonnegative_when_a_is_one (x : ℝ) (h : x ≥ 0) : f 1 x ≥ 0 := by
  sorry

-- Part 2
theorem g_range_on_interval :
  Set.range (fun x => g 1 x) ∩ Set.Icc (1/2 : ℝ) 2 = 
  Set.Icc ((1/2) * (1 + Real.log 2)) (4 - Real.log 2) := by
  sorry

-- Part 3
theorem equation_two_roots_iff (a : ℝ) :
  (∃ x y, x ≠ y ∧ f a x + Real.sin x = Real.log x ∧ f a y + Real.sin y = Real.log y) ↔
  (0 < a ∧ a < 1 / Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_when_a_is_one_g_range_on_interval_equation_two_roots_iff_l278_27828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_snake_length_in_inches_l278_27829

-- Define conversion factors
noncomputable def inches_per_foot : ℝ := 12
noncomputable def cm_per_meter : ℝ := 100
noncomputable def inches_per_yard : ℝ := 36
noncomputable def cm_per_inch : ℝ := 2.54

-- Define snake lengths in their original units
noncomputable def snake1_length : ℝ := 2.4  -- feet
noncomputable def snake2_length : ℝ := 16.2  -- inches
noncomputable def snake3_length : ℝ := 10.75  -- inches
noncomputable def snake4_length : ℝ := 50.5  -- centimeters
noncomputable def snake5_length : ℝ := 0.8  -- meters
noncomputable def snake6_length : ℝ := 120.35  -- centimeters
noncomputable def snake7_length : ℝ := 1.35  -- yards

-- Function to convert feet to inches
noncomputable def feet_to_inches (x : ℝ) : ℝ := x * inches_per_foot

-- Function to convert centimeters to inches
noncomputable def cm_to_inches (x : ℝ) : ℝ := x / cm_per_inch

-- Function to convert meters to inches
noncomputable def meters_to_inches (x : ℝ) : ℝ := cm_to_inches (x * cm_per_meter)

-- Function to convert yards to inches
noncomputable def yards_to_inches (x : ℝ) : ℝ := x * inches_per_yard

-- Theorem statement
theorem total_snake_length_in_inches :
  let total_length := feet_to_inches snake1_length +
                      snake2_length +
                      snake3_length +
                      cm_to_inches snake4_length +
                      meters_to_inches snake5_length +
                      cm_to_inches snake6_length +
                      yards_to_inches snake7_length
  ∃ ε > 0, |total_length - 203.11| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_snake_length_in_inches_l278_27829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_greater_number_l278_27835

theorem smallest_greater_number (x : ℤ) (n : ℕ) : 
  (∀ y : ℤ, (2134 : ℚ) * 10^y < n → y ≤ 4) ∧ ((2134 : ℚ) * 10^4 < n) → n = 21341 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_greater_number_l278_27835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_eccentricity_is_four_fifths_l278_27898

noncomputable section

-- Define the radius of the cylinder and larger spheres
def R : ℝ := 1

-- Define the radius of the smaller sphere
def r : ℝ := 1/4

-- Define the major axis of the ellipse
def a : ℝ := 5/3

-- Define the minor axis of the ellipse
def b : ℝ := 1

-- Define the eccentricity of the ellipse
def e : ℝ := Real.sqrt (a^2 - b^2) / a

-- Theorem statement
theorem max_eccentricity_is_four_fifths :
  e = 4/5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_eccentricity_is_four_fifths_l278_27898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_equation_has_solution_l278_27842

theorem at_least_one_equation_has_solution (a b c : ℝ) :
  (∃ x : ℝ, a * Real.sin x + b * Real.cos x + c = 0) ∨
  (∃ x : ℝ, 2 * a * Real.tan x + b * (1 / Real.tan x) + 2 * c = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_equation_has_solution_l278_27842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l278_27830

noncomputable def g (x : ℝ) : ℝ := Real.sin x ^ 4 - Real.sin x ^ 3 * Real.cos x + Real.cos x ^ 4

theorem g_range : ∀ x : ℝ, 0.316 ≤ g x ∧ g x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l278_27830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l278_27876

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ 3 then a - x
  else if x > 3 then a * Real.log x / Real.log 2
  else 0  -- undefined for x < 0

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f a 2 < f a 4) → a > -2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l278_27876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_squares_l278_27893

theorem arithmetic_sequence_squares (k : ℤ) : 
  (∃ (a d : ℝ), 
    Real.sqrt (49 + k) = a ∧ 
    Real.sqrt (225 + k) = a + d ∧ 
    Real.sqrt (441 + k) = a + 2*d) → 
  k = 255 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_squares_l278_27893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_ratio_in_triangle_l278_27883

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/12 = 1

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  B = (-2, 0) ∧ C = (2, 0) ∧ ellipse A.fst A.snd

-- Define angles
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem sin_ratio_in_triangle (A B C : ℝ × ℝ) :
  triangle_ABC A B C →
  (Real.sin (angle B A C) + Real.sin (angle B C A)) / Real.sin (angle A B C) = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_ratio_in_triangle_l278_27883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_origin_l278_27819

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem tangent_slope_at_origin : 
  (deriv f) 0 = Real.exp 0 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_origin_l278_27819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_passing_both_tests_l278_27871

theorem students_passing_both_tests
  (total_students : ℕ)
  (passed_first : ℕ)
  (passed_second : ℕ)
  (fail_prob : ℚ)
  (h1 : total_students = 100)
  (h2 : passed_first = 60)
  (h3 : passed_second = 40)
  (h4 : fail_prob = 1/5) :
  ∃ (x : ℕ), x = 20 ∧ 
    x = passed_first + passed_second - (total_students - (fail_prob * total_students).floor) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_passing_both_tests_l278_27871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l278_27824

-- Define the triangle ABC
def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  C = Real.pi / 3 ∧ b = 8 ∧ (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3

-- Theorem statement
theorem triangle_ABC_properties :
  ∀ a b c A B C,
  triangle_ABC a b c A B C →
  c = 7 ∧ Real.cos (B - C) = 13/14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l278_27824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_georges_required_speed_l278_27823

/-- The required speed for George to reach school on time -/
noncomputable def required_speed (total_distance : ℝ) (normal_speed : ℝ) (leisurely_distance : ℝ) (leisurely_speed : ℝ) : ℝ :=
  let normal_time := total_distance / normal_speed
  let leisurely_time := leisurely_distance / leisurely_speed
  let remaining_time := normal_time - leisurely_time
  let remaining_distance := total_distance - leisurely_distance
  remaining_distance / remaining_time

/-- Theorem stating the required speed for George to reach school on time -/
theorem georges_required_speed :
  required_speed 1.5 3 0.75 2 = 6 := by
  -- Unfold the definition of required_speed
  unfold required_speed
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_georges_required_speed_l278_27823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absorbed_moisture_percentages_l278_27854

/-- Represents the properties of a moisture-absorbing substance -/
structure MoistureAbsorbingSubstance where
  divided_mass : ℝ
  undivided_mass : ℝ
  absorbed_moisture : ℝ
  percentage_difference : ℝ

/-- Calculates the percentage of absorbed moisture relative to the substance's mass -/
noncomputable def absorbed_moisture_percentage (mass : ℝ) (absorbed : ℝ) : ℝ :=
  (absorbed / mass) * 100

/-- Theorem about the percentages of absorbed moisture in divided and undivided states -/
theorem absorbed_moisture_percentages 
  (substance : MoistureAbsorbingSubstance)
  (h1 : substance.undivided_mass = substance.divided_mass + 300)
  (h2 : substance.absorbed_moisture = 1400)
  (h3 : absorbed_moisture_percentage substance.undivided_mass substance.absorbed_moisture =
        absorbed_moisture_percentage substance.divided_mass substance.absorbed_moisture - substance.percentage_difference)
  (h4 : substance.percentage_difference = 105) :
  absorbed_moisture_percentage substance.divided_mass substance.absorbed_moisture = 280 ∧
  absorbed_moisture_percentage substance.undivided_mass substance.absorbed_moisture = 175 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_absorbed_moisture_percentages_l278_27854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l278_27877

-- Define what it means for an angle to be in the second quadrant
def is_in_second_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, Real.pi / 2 + 2 * Real.pi * (k : Real) < α ∧ α < Real.pi + 2 * Real.pi * (k : Real)

-- Define what it means for an angle to be in the first or third quadrant
def is_in_first_or_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 0 + Real.pi * (k : Real) < α ∧ α < Real.pi / 2 + Real.pi * (k : Real)

theorem half_angle_quadrant (α : Real) :
  is_in_second_quadrant α → is_in_first_or_third_quadrant (α / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l278_27877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l278_27815

noncomputable def z : ℂ := 2 - Complex.I * Real.sqrt 3

theorem modulus_of_z :
  Complex.abs z = Real.sqrt 7 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l278_27815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducible_fraction_count_l278_27810

def N : ℕ := 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10

def is_irreducible (a b : ℕ) : Prop :=
  Nat.Coprime a b ∧ a * b = N

theorem irreducible_fraction_count :
  ∃ (S : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ S ↔ is_irreducible p.fst p.snd) ∧ Finset.card S = 16 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irreducible_fraction_count_l278_27810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_stock_proof_l278_27832

def books_sold : ℕ := 402
def percentage_not_sold : ℚ := 63.45

theorem initial_stock_proof :
  let percentage_sold : ℚ := 100 - percentage_not_sold
  let initial_stock : ℚ := books_sold / (percentage_sold / 100)
  ⌊initial_stock⌋ = 1100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_stock_proof_l278_27832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_ratio_is_two_side_b_length_l278_27805

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  (Real.cos t.A - 2 * Real.cos t.C) / Real.cos t.B = (2 * t.c - t.a) / t.b ∧
  Real.cos t.B = 1/4 ∧
  t.a + t.b + t.c = 5

-- Theorem 1
theorem sin_ratio_is_two (t : Triangle) (h : satisfies_conditions t) :
  Real.sin t.C / Real.sin t.A = 2 := by sorry

-- Theorem 2
theorem side_b_length (t : Triangle) (h : satisfies_conditions t) :
  t.b = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_ratio_is_two_side_b_length_l278_27805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_students_seating_l278_27874

/-- The number of different seating arrangements for n students,
    where two specific students must sit together -/
def seating_arrangements (n : ℕ) : ℕ :=
  (n - 1).factorial * 2

/-- The problem statement -/
theorem six_students_seating : seating_arrangements 6 = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_students_seating_l278_27874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_64n4_l278_27846

theorem divisors_of_64n4 (n : ℕ+) (h : (Nat.divisors (120 * n.val ^ 3)).card = 120) : 
  (Nat.divisors (64 * n.val ^ 4)).card = 675 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_64n4_l278_27846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_c_coordinates_l278_27848

-- Define the triangle ABC
structure Triangle (α : Type*) [LinearOrderedField α] where
  A : α × α
  B : α × α
  C : α × α

-- Define the inverse proportional function
noncomputable def inverse_proportional (x : ℝ) : ℝ := Real.sqrt 3 / x

-- State the theorem
theorem triangle_c_coordinates 
  (ABC : Triangle ℝ) 
  (h1 : ABC.A.1 > 0 ∧ ABC.C.1 > 0)
  (h2 : ABC.A.2 = inverse_proportional ABC.A.1)
  (h3 : ABC.C.2 = inverse_proportional ABC.C.1)
  (h4 : (ABC.C.2 - ABC.B.2) * (ABC.C.1 - ABC.B.1) = -(ABC.B.2 - ABC.A.2) * (ABC.B.1 - ABC.A.1)) -- ∠ACB = 90°
  (h5 : (ABC.C.2 - ABC.B.2) / (ABC.C.1 - ABC.B.1) = Real.sqrt 3) -- ∠ABC = 30°
  (h6 : ABC.A.1 = ABC.B.1) -- AB ⟂ x-axis
  (h7 : ABC.B.2 > ABC.A.2) -- B is above A
  (h8 : ABC.B.2 - ABC.A.2 = 6) -- AB = 6
  : ABC.C = (Real.sqrt 3 / 2, 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_c_coordinates_l278_27848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_elements_count_l278_27843

def smallest_multiples (n : ℕ) (count : ℕ) : Finset ℕ :=
  Finset.filter (λ k => ∃ m : ℕ, m ≤ count ∧ k = n * m) (Finset.range (n * count + 1))

def A : Finset ℕ := smallest_multiples 7 1500
def B : Finset ℕ := smallest_multiples 9 1500

theorem common_elements_count : (A ∩ B).card = 166 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_elements_count_l278_27843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l278_27897

/-- A trapezoid with specific properties -/
structure Trapezoid where
  PQ : ℝ
  RS : ℝ
  height : ℝ
  PS : ℝ
  parallel : PQ < RS
  equal_sides : True  -- Represents PR = QS

/-- The perimeter of a trapezoid with given properties -/
noncomputable def perimeter (t : Trapezoid) : ℝ :=
  t.PQ + t.RS + 2 * Real.sqrt (((t.RS - t.PQ) / 2) ^ 2 + t.height ^ 2)

/-- Theorem stating the perimeter of the specific trapezoid -/
theorem trapezoid_perimeter : 
  ∀ t : Trapezoid, 
    t.PQ = 10 ∧ 
    t.RS = 20 ∧ 
    t.height = 5 ∧ 
    t.PS = 13 → 
    perimeter t = 30 + 10 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l278_27897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_wins_l278_27892

/- Define the chessboard -/
structure Chessboard where
  n : ℕ
  squares : Fin n → Fin n → Bool
  rook_position : Fin n × Fin n

/- Define a move -/
inductive Move where
  | Horizontal : ℕ → Move
  | Vertical : ℕ → Move

/- Define the game state -/
structure GameState where
  board : Chessboard
  current_player : Bool  -- true for Player A, false for Player B

/- Define a strategy -/
def Strategy := GameState → Move

/- Define the winning strategy for Player A -/
def winning_strategy : Strategy :=
  fun gs => Move.Vertical (gs.board.n - gs.board.rook_position.2.val - 1)

/- Define a predicate for when a player wins -/
def player_wins (s : Strategy) (initial_state final_state : GameState) : Prop :=
  sorry  -- The actual implementation would go here

/- Main theorem -/
theorem player_a_wins (n : ℕ) (h : n > 1) : 
  ∃ (s : Strategy), ∀ (initial_state : GameState), 
    initial_state.board.n = n → 
    initial_state.current_player = true → 
    s = winning_strategy → 
    (∃ (final_state : GameState), player_wins s initial_state final_state) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_a_wins_l278_27892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l278_27814

/-- Triangle ABC with base BC and height from A to BC -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- The area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := (1 / 2) * t.base * t.height

theorem area_of_triangle_ABC :
  let t : Triangle := { base := 12, height := 15 }
  triangleArea t = 90 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABC_l278_27814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l278_27845

/-- The minimum distance between a point on the circle x² + (y-1)² = 2 
    and a point on the line x + y = 5 is √2. -/
theorem min_distance_circle_line : 
  ∃ (d : ℝ), d = Real.sqrt 2 ∧ 
  ∀ (P Q : ℝ × ℝ), 
    (P.1^2 + (P.2 - 1)^2 = 2) → 
    (Q.1 + Q.2 = 5) → 
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by
  -- We'll use d = √2 as our witness
  use Real.sqrt 2
  constructor
  · -- Prove that d = √2
    rfl
  · -- Prove the inequality for all points P and Q
    intros P Q hP hQ
    sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_line_l278_27845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_M_and_quadrilateral_ABCD_l278_27864

-- Define the curve M
noncomputable def curve_M (β : ℝ) : ℝ × ℝ := (1 + 2 * Real.cos β, 1 + 2 * Real.sin β)

-- Define the lines l₁ and l₂ in polar coordinates
def line_l₁ (α : ℝ) : ℝ → ℝ := λ _ => α
noncomputable def line_l₂ (α : ℝ) : ℝ → ℝ := λ _ => α + Real.pi / 2

-- Define the intersection points
noncomputable def point_A (α : ℝ) : ℝ × ℝ := sorry
noncomputable def point_B (α : ℝ) : ℝ × ℝ := sorry
noncomputable def point_C (α : ℝ) : ℝ × ℝ := sorry
noncomputable def point_D (α : ℝ) : ℝ × ℝ := sorry

-- Helper function for area calculation
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem curve_M_and_quadrilateral_ABCD :
  (∃ (center : ℝ × ℝ) (radius : ℝ), center = (1, 1) ∧ radius = 2 ∧
    ∀ β, (curve_M β).1^2 + (curve_M β).2^2 = (center.1 - (curve_M β).1)^2 + (center.2 - (curve_M β).2)^2 + radius^2) ∧
  (∀ α, 4 * Real.sqrt 2 ≤ area_quadrilateral (point_A α) (point_B α) (point_C α) (point_D α) ∧
        area_quadrilateral (point_A α) (point_B α) (point_C α) (point_D α) ≤ 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_M_and_quadrilateral_ABCD_l278_27864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_sum_of_squares_equal_product_l278_27804

theorem subset_with_sum_of_squares_equal_product :
  ∃ (A : Finset ℕ), A.card = 5 ∧ A.Nonempty ∧ (∀ x ∈ A, x > 0) ∧
  (A.sum (λ x => x^2) = A.prod id) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_with_sum_of_squares_equal_product_l278_27804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_from_cos_difference_l278_27866

theorem sin_product_from_cos_difference (α β m : ℝ) : 
  Real.cos α ^ 2 - Real.cos β ^ 2 = m → Real.sin (α + β) * Real.sin (α - β) = -m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_from_cos_difference_l278_27866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_price_is_1_5_l278_27896

/-- The cost of producing yogurt batches -/
def yogurt_production (milk_price : ℝ) (batches : ℝ) (fruit_price : ℝ) : ℝ :=
  batches * (10 * milk_price + 3 * fruit_price)

/-- Theorem: The price of milk per liter is $1.5 -/
theorem milk_price_is_1_5 : 
  ∃ (milk_price : ℝ), 
    yogurt_production milk_price 3 2 = 63 ∧ 
    milk_price = 1.5 := by
  use 1.5
  constructor
  · -- Prove that yogurt_production 1.5 3 2 = 63
    simp [yogurt_production]
    norm_num
  · -- Prove that milk_price = 1.5
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_price_is_1_5_l278_27896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_bisects_tangent_angle_l278_27858

/-- Predicate stating that a triangle ABC is acute-angled -/
def AcuteTriangle (A B C : Point) : Prop := sorry

/-- Predicate stating that triangle ABC is inscribed in circle O -/
def InscribedIn (A B C O : Point) : Prop := sorry

/-- Predicate stating that a line through point P is tangent to circle O -/
def Tangent (P O : Point) : Prop := sorry

/-- Predicate stating that point P is on the tangent line from A to B on circle O -/
def OnTangent (P A B : Point) : Prop := sorry

/-- Predicate stating that AD is an altitude of triangle ABC -/
def Altitude (A D B C : Point) : Prop := sorry

/-- Predicate stating that line AD bisects angle MDN -/
def Bisects (A D M N : Point) : Prop := sorry

/-- Given an acute-angled triangle ABC inscribed in circle O, with tangents at B and C
    intersecting the tangent at A at points M and N respectively, and AD being the altitude
    on BC, prove that AD bisects angle MDN. -/
theorem altitude_bisects_tangent_angle (A B C D M N O : Point) :
  AcuteTriangle A B C →
  InscribedIn A B C O →
  Tangent B O →
  Tangent C O →
  Tangent A O →
  OnTangent M A B →
  OnTangent N A C →
  Altitude A D B C →
  Bisects A D M N := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_bisects_tangent_angle_l278_27858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_properties_l278_27813

/-- Rectangle ABCD with given coordinates -/
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The specific rectangle in the problem -/
def rectangleABCD : Rectangle :=
  { A := (0, 0),
    B := (2, 0),
    C := (2, 1),
    D := (0, 1) }

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem stating the properties of the rectangle -/
theorem rectangle_properties (rect : Rectangle) (h : rect = rectangleABCD) :
  (distance rect.A rect.C)^2 = (distance rect.A rect.D)^2 + (distance rect.D rect.C)^2 ∧
  (rect.B.1 - rect.A.1) * (rect.D.2 - rect.A.2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_properties_l278_27813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_max_proof_l278_27860

def M := {x : ℝ | |2*x - 1| < 1}

theorem inequality_and_max_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  let h := max (2 / Real.sqrt a) (max ((a + b) / Real.sqrt (a * b)) ((a * b + 1) / Real.sqrt b))
  (a * b + 1 > a + b) ∧ (h > 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_max_proof_l278_27860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_arrangements_eq_54_l278_27801

/-- Represents a teacher --/
inductive Teacher : Type
| A : Teacher
| B : Teacher
| C : Teacher
| D : Teacher
| E : Teacher
deriving DecidableEq

/-- Represents an interest group --/
structure InterestGroup where
  teachers : List Teacher
  size_constraint : teachers.length ≤ 2
  not_alone : ∀ t ∈ teachers, t ≠ Teacher.A → t ≠ Teacher.B → teachers.length > 1

/-- Represents an arrangement of teachers into interest groups --/
structure Arrangement where
  groups : Fin 3 → InterestGroup
  all_teachers : (List.join (List.map InterestGroup.teachers (List.ofFn groups))).toFinset = {Teacher.A, Teacher.B, Teacher.C, Teacher.D, Teacher.E}

/-- The number of valid arrangements --/
def num_arrangements : ℕ := sorry

theorem num_arrangements_eq_54 : num_arrangements = 54 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_arrangements_eq_54_l278_27801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_equation_solution_l278_27822

theorem number_equation_solution :
  ∃ x : ℝ, (x = Real.sqrt ((27 - Real.sqrt 6) / 2) ∨ x = -Real.sqrt ((27 - Real.sqrt 6) / 2)) ∧
            (2 * x^2 + Real.sqrt 6)^3 = 19683 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_equation_solution_l278_27822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_tangents_l278_27872

/-- Represents a circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the number of common tangents between two circles --/
def CommonTangents (c1 c2 : Circle) : ℕ := sorry

/-- States that one circle has exactly twice the radius of the other --/
def DoubleRadius (c1 c2 : Circle) : Prop :=
  c1.radius = 2 * c2.radius ∨ c2.radius = 2 * c1.radius

theorem impossible_tangents (c1 c2 : Circle) (h : DoubleRadius c1 c2) :
  CommonTangents c1 c2 ≠ 1 ∧ CommonTangents c1 c2 ≠ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_tangents_l278_27872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_on_circle_l278_27806

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The radius of the circle -/
def r : ℝ := 50

/-- The distance traveled by one point to all non-adjacent points and back -/
noncomputable def distance_one_point : ℝ := 2 * r * (Real.sqrt 2 + Real.sqrt (2 + Real.sqrt 2))

/-- The total distance traveled by all points -/
noncomputable def total_distance : ℝ := n * distance_one_point

/-- Theorem stating the total distance traveled -/
theorem total_distance_on_circle :
  total_distance = 800 * Real.sqrt 2 + 800 * Real.sqrt (2 + Real.sqrt 2) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_on_circle_l278_27806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_tuesday_ratio_l278_27867

/-- Represents the number of push-ups Miriam does each day of the week --/
structure PushupCount where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Defines Miriam's push-up routine for the week --/
def miriamPushups (W : ℕ) : PushupCount where
  monday := 5
  tuesday := 7
  wednesday := W
  thursday := (5 + 7 + W) / 2
  friday := 39

/-- The total push-ups for the first four days equals Friday's count --/
axiom total_equals_friday (W : ℕ) :
  let p := miriamPushups W
  p.monday + p.tuesday + p.wednesday + p.thursday = p.friday

/-- Theorem stating the ratio of Wednesday to Tuesday push-ups --/
theorem wednesday_tuesday_ratio (W : ℕ) :
  (miriamPushups W).wednesday / (miriamPushups W).tuesday = 2 := by
  sorry

#check wednesday_tuesday_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_tuesday_ratio_l278_27867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_covers_except_a_l278_27818

noncomputable def f (n : ℕ+) : ℕ := 
  Int.toNat ⌊(n : ℝ) + Real.sqrt (3 * n) + 1/2⌋

noncomputable def a (n : ℕ+) : ℕ := 
  Int.toNat ⌊((n^2 : ℝ) + 2*n) / 3⌋

theorem f_covers_except_a :
  ∀ k : ℕ+, (∃ n : ℕ+, f n = k) ∨ (∃ m : ℕ+, a m = k) :=
by
  sorry

#check f_covers_except_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_covers_except_a_l278_27818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l278_27827

-- Define the propositions p and q as functions of k
def p (k : ℝ) : Prop := ∀ x y : ℝ, x < y → k * x + 1 < k * y + 1

def q (k : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁^2 + (2*k - 3)*x₁ + 1 = 0 ∧ 
  x₂^2 + (2*k - 3)*x₂ + 1 = 0

-- Define the set of k values that satisfy the conditions
def K : Set ℝ := {k : ℝ | (¬(p k ∧ q k)) ∧ (p k ∨ q k)}

-- State the theorem
theorem k_range : K = Set.Iic 0 ∪ Set.Icc (1/2) (5/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l278_27827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_zero_l278_27879

noncomputable def w : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 3))

theorem sum_of_powers_zero : 
  w / (1 + w^3) + w^2 / (1 + w^6) + w^3 / (1 + w^9) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_zero_l278_27879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l278_27800

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  -- a and b are roots of x^2 - 2√3x + 2 = 0
  t.a^2 - 2 * Real.sqrt 3 * t.a + 2 = 0 ∧
  t.b^2 - 2 * Real.sqrt 3 * t.b + 2 = 0 ∧
  -- 2cos(A+B) = 1
  2 * Real.cos (t.A + t.B) = 1 ∧
  -- BC = a, AC = b
  t.c = t.a ∧
  t.b = t.b

theorem triangle_properties (t : Triangle) (h : TriangleProperties t) :
  t.C = 2 * Real.pi / 3 ∧ t.a^2 = 10 := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l278_27800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l278_27873

theorem problem_solution : 
  (∀ x : ℝ, x^3 = 8 → x + |(-5)| + (-1)^2023 = 6) ∧
  (∀ k b : ℝ, (∀ x : ℝ, (k * 0 + b = 1) ∧ (k * 2 + b = 5)) → k = 2 ∧ b = 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l278_27873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_equals_six_l278_27863

/-- The radius of a sphere with surface area equal to the curved surface area of a right circular cylinder -/
noncomputable def sphere_radius (cylinder_height : ℝ) (cylinder_diameter : ℝ) : ℝ :=
  let cylinder_radius := cylinder_diameter / 2
  let cylinder_surface_area := 2 * Real.pi * cylinder_radius * cylinder_height
  let sphere_surface_area := cylinder_surface_area
  Real.sqrt (sphere_surface_area / (4 * Real.pi))

/-- Theorem: The radius of a sphere with surface area equal to the curved surface area of a right circular cylinder
    with height and diameter both 12 cm is 6 cm -/
theorem sphere_radius_equals_six :
  sphere_radius 12 12 = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_equals_six_l278_27863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_to_sell_all_income_difference_total_profit_l278_27807

-- Define constants and variables
def total_investment : ℝ := 13500
def total_yield : ℝ := 19000
def orchard_price : ℝ := 4
def market_price : ℝ → ℝ := λ x ↦ x
def daily_market_sales : ℝ := 1000

-- Define profit function
def profit (revenue : ℝ) : ℝ := revenue - total_investment

-- Theorem 1: Days to sell all fruits in market
theorem days_to_sell_all (x : ℝ) (h : x > 4) : 
  total_yield / daily_market_sales = 19 := by sorry

-- Theorem 2: Income difference between market and orchard sales
theorem income_difference (x : ℝ) (h : x > 4) :
  total_yield * market_price x - total_yield * orchard_price = 19000 * x - 76000 := by sorry

-- Theorem 3: Total profit from mixed sales
theorem total_profit (x : ℝ) (h : x > 4) :
  profit (6000 * orchard_price + (total_yield - 6000) * market_price x) = 13000 * x + 10500 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_to_sell_all_income_difference_total_profit_l278_27807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_6Tn_n_minus_1_bound_l278_27882

def triangular_number (n : ℕ+) : ℕ := n.val * (n.val + 1) / 2

theorem gcd_6Tn_n_minus_1_bound (n : ℕ+) : 
  Nat.gcd (6 * triangular_number n) (n.val - 1) ≤ 3 ∧ 
  ∃ m : ℕ+, Nat.gcd (6 * triangular_number m) (m.val - 1) = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_6Tn_n_minus_1_bound_l278_27882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_k_range_l278_27821

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 4)

-- Define the function g
def g (k x : ℝ) : ℝ := k * x^2 + 2 * k * x + 1

-- State the theorem
theorem f_monotone_and_k_range :
  (∀ x₁ x₂ : ℝ, -2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 → f x₁ < f x₂) ∧
  (∀ k : ℝ, k ≠ 0 →
    ((∀ x₁ : ℝ, -2 ≤ x₁ ∧ x₁ ≤ 2 →
      ∃ x₂ : ℝ, -1 ≤ x₂ ∧ x₂ ≤ 2 ∧ f x₁ = g k x₂) ↔
    (k ≤ -5/32 ∨ 5/4 ≤ k))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_k_range_l278_27821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l278_27834

-- Define the curve C in polar coordinates
noncomputable def C (ρ θ : ℝ) : Prop := ρ * (Real.sin θ)^2 = Real.cos θ

-- Define the line L in parametric form
noncomputable def L (t : ℝ) : ℝ × ℝ := (2 - Real.sqrt 2/2 * t, Real.sqrt 2/2 * t)

-- State the theorem
theorem curve_and_line_intersection :
  -- The Cartesian equation of C is y² = x
  (∀ x y : ℝ, (∃ ρ θ : ℝ, C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ↔ y^2 = x) ∧
  -- The length of AB is 3√2
  (∃ t₁ t₂ : ℝ, 
    let (x₁, y₁) := L t₁
    let (x₂, y₂) := L t₂
    (y₁^2 = x₁) ∧ (y₂^2 = x₂) ∧
    ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 18) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l278_27834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l278_27844

noncomputable def circle_area_problem (r₁ r₂ : ℝ) : Prop :=
  let r₁ := 2
  let r₂ := 3
  let θ := 2 * Real.arccos (2/3)
  let shaded_area := θ * 5 - (2 * Real.sqrt 15) / 3
  r₁ > 0 ∧ r₂ > 0 ∧ r₁ < r₂ →
  ∃ (A B : ℝ × ℝ),
    (∀ (P : ℝ × ℝ), ‖P - (0, 0)‖ = r₁ → ‖P - A‖ = r₂ ∨ ‖P - B‖ = r₂) ∧
    ‖A - B‖ = 2 * r₁ ∧
    shaded_area = Real.pi * r₂^2 * (θ / (2 * Real.pi)) - Real.pi * r₁^2 / 2

theorem circle_area_theorem : circle_area_problem 2 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l278_27844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equations_mod_7_l278_27890

/-- Solve quadratic equations in arithmetic modulo 7 -/
theorem quadratic_equations_mod_7 :
  (∀ x : Fin 7, (5 * x^2 + 3 * x + 1) % 7 ≠ 0) ∧
  (∃! x : Fin 7, (x^2 + 3 * x + 4) % 7 = 0 ∧ x = 2) ∧
  (∃ x y : Fin 7, x ≠ y ∧ 
    (x^2 - 2 * x - 3) % 7 = 0 ∧ 
    (y^2 - 2 * y - 3) % 7 = 0 ∧ 
    ({x, y} : Set (Fin 7)) = {3, 6}) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equations_mod_7_l278_27890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l278_27812

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x ∈ Set.Ioo 1 2 → x^2 > 1) ↔ ∃ x : ℝ, x ∈ Set.Ioo 1 2 ∧ x^2 ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l278_27812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2005_of_4_7_l278_27857

noncomputable def f (x : ℝ) : ℝ := (1 + x) / (1 - 3 * x)

noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => f
| n + 1 => λ x => f (f_n n x)

theorem f_2005_of_4_7 : f_n 2005 (37/10) = 37/57 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2005_of_4_7_l278_27857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liangliang_school_distance_l278_27820

/-- The distance between Liangliang's home and school -/
noncomputable def distance : ℝ := 1000

/-- The time it takes Liangliang to walk to school at 40 meters per minute -/
noncomputable def time_at_40 : ℝ := distance / 40

/-- The time it takes Liangliang to walk to school at 50 meters per minute -/
noncomputable def time_at_50 : ℝ := distance / 50

theorem liangliang_school_distance :
  distance = 1000 ∧ time_at_40 - time_at_50 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liangliang_school_distance_l278_27820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_l278_27852

theorem min_abs_diff (a b : ℕ) (h : a * b - 4 * a + 6 * b = 528) :
  ∃ (a' b' : ℕ), a' * b' - 4 * a' + 6 * b' = 528 ∧
  ∀ (x y : ℕ), x * y - 4 * x + 6 * y = 528 →
  |Int.ofNat a' - Int.ofNat b'| ≤ |Int.ofNat x - Int.ofNat y| ∧ |Int.ofNat a' - Int.ofNat b'| = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_diff_l278_27852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_proof_l278_27851

theorem x_value_proof (b x : ℝ) (hb : b > 1) (hx : x > 0) 
  (h_eq : (3*x)^(Real.log 3 / Real.log b) - (5*x)^(Real.log 5 / Real.log b) = 0) : x = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_proof_l278_27851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l278_27855

/-- Given a triangle ABC and a point D in its plane, if BC = 3CD, then AD = -1/3 * AB + 4/3 * AC -/
theorem vector_relation (A B C D : EuclideanSpace ℝ (Fin 2)) : 
  (C - B) = 3 • (D - C) → 
  (D - A) = -(1/3) • (B - A) + (4/3) • (C - A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l278_27855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l278_27856

noncomputable def original_function (x : ℝ) : ℝ := Real.cos x

noncomputable def shifted_function (x : ℝ) : ℝ := original_function (x + Real.pi/4)

noncomputable def transformed_function (x : ℝ) : ℝ := shifted_function (x/2)

theorem graph_transformation :
  ∀ x : ℝ, transformed_function x = Real.cos (2*x + Real.pi/4) := by
  intro x
  unfold transformed_function shifted_function original_function
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l278_27856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_theorem_l278_27817

/-- The function f(x) = e^x(x^2 - x + a) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 - x + a)

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + x + a - 1)

/-- The equation of the tangent line through the origin -/
def tangent_equation (a : ℝ) (x₀ : ℝ) : ℝ := x₀^3 + a*x₀ - a

theorem tangent_line_theorem (a : ℝ) (x₁ x₂ x₃ : ℝ) :
  (∃ y₁ y₂ y₃ : ℝ, 
    (∀ x : ℝ, x ≠ x₁ ∧ x ≠ x₂ ∧ x ≠ x₃ → tangent_equation a x ≠ 0) ∧
    (tangent_equation a x₁ = 0) ∧ (tangent_equation a x₂ = 0) ∧ (tangent_equation a x₃ = 0) ∧
    x₁ < x₂ ∧ x₂ < x₃) →
  (a < -27/4 ∧ x₁ < -3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_theorem_l278_27817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_negation_residuals_and_fitting_rsquared_and_fitting_l278_27886

-- Define the proposition
def P : Prop := ∃ x₀ : ℝ, x₀^2 - x₀ - 1 > 0

-- Define the negation of the proposition
def notP : Prop := ∀ x : ℝ, x^2 - x - 1 ≤ 0

-- Define a measure for sum of squared residuals
noncomputable def sumSquaredResiduals : ℝ → ℝ := sorry

-- Define a measure for fitting effect
noncomputable def fittingEffect : ℝ → ℝ := sorry

-- Define R-squared
noncomputable def rSquared : ℝ → ℝ := sorry

theorem proposition_negation : ¬P ↔ notP := by sorry

theorem residuals_and_fitting : 
  ∀ (s₁ s₂ : ℝ), sumSquaredResiduals s₁ < sumSquaredResiduals s₂ → 
  fittingEffect s₁ > fittingEffect s₂ := by sorry

theorem rsquared_and_fitting : 
  ∀ (r₁ r₂ : ℝ), rSquared r₁ > rSquared r₂ → 
  fittingEffect r₁ > fittingEffect r₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_negation_residuals_and_fitting_rsquared_and_fitting_l278_27886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_four_l278_27881

/-- A geometric sequence with first term a₁ and common ratio q -/
noncomputable def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ := λ n ↦ a₁ * q^(n - 1)

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_four :
  ∀ (a₁ a₂ a₃ : ℝ),
  a₁ = 2 →
  a₃ = 4 * (a₂ - 2) →
  ∃ (q : ℝ),
  (∀ (n : ℕ), geometric_sequence a₁ q n = a₁ * q^(n - 1)) ∧
  geometric_sum a₁ q 4 = 30 := by
  sorry

#check geometric_sequence_sum_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_four_l278_27881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_maximization_l278_27847

variable (S : ℝ) (r : ℝ)

def surface_area (S : ℝ) : ℝ := S

def cylinder_radius (r : ℝ) : ℝ := r

noncomputable def volume (S r : ℝ) : ℝ := (1/2) * S * r - Real.pi * r^3

noncomputable def radius_upper_bound (S : ℝ) : ℝ := Real.sqrt (2 * Real.pi * S) / (2 * Real.pi)

theorem cylinder_volume_maximization (h₁ : S > 0) (h₂ : 0 < r) (h₃ : r < radius_upper_bound S) :
  volume S r = (1/2) * S * r - Real.pi * r^3 ∧
  ∃ (r_max : ℝ), r_max = Real.sqrt (6 * Real.pi * S) / (6 * Real.pi) ∧
    ∀ (r' : ℝ), 0 < r' ∧ r' < radius_upper_bound S → volume S r' ≤ volume S r_max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_maximization_l278_27847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_hexagon_area_ratio_l278_27831

/-- The area of a hexagon formed by joining the midpoints of a regular hexagon's sides
    is 25% of the original hexagon's area. -/
theorem midpoint_hexagon_area_ratio (a : ℝ) (h : a > 0) :
  (3 * Real.sqrt 3 / 2) * (a/2)^2 / ((3 * Real.sqrt 3 / 2) * a^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_hexagon_area_ratio_l278_27831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_profit_theorem_l278_27816

/-- Calculates the total profit for a shop given the mean profits for two halves of a month. -/
def shop_profit (total_days : ℕ) (mean_profit : ℚ) 
  (first_half_days : ℕ) (first_half_mean : ℚ) 
  (second_half_days : ℕ) (second_half_mean : ℚ) : ℚ :=
  (first_half_days : ℚ) * first_half_mean + 
  (second_half_days : ℚ) * second_half_mean

/-- Proves that the calculated total profit matches the given mean profit for the entire period. -/
theorem shop_profit_theorem (total_days : ℕ) (mean_profit : ℚ) 
  (first_half_days : ℕ) (first_half_mean : ℚ) 
  (second_half_days : ℕ) (second_half_mean : ℚ) : 
  shop_profit total_days mean_profit first_half_days first_half_mean second_half_days second_half_mean = 
  mean_profit * (total_days : ℚ) := by
  sorry

#eval shop_profit 30 350 15 285 15 415

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_profit_theorem_l278_27816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_score_average_ratio_l278_27887

theorem test_score_average_ratio (n : ℕ) (original_avg wrong_score correct_score : ℚ) :
  n = 50 ∧ 
  original_avg = 77 ∧ 
  wrong_score = 97 ∧ 
  correct_score = 79 →
  (let correct_sum := n * original_avg - wrong_score + correct_score
   let correct_avg := correct_sum / n
   let new_sum := correct_sum + correct_avg
   let new_avg := new_sum / (n + 1)
   new_avg / correct_avg = 1) := by
  sorry

#check test_score_average_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_score_average_ratio_l278_27887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_ways_to_get_three_l278_27826

theorem two_ways_to_get_three : ∃ (f g : Fin 6 → Int), 
  (∀ i, f i = 1 ∨ f i = -1) ∧ 
  (∀ i, g i = 1 ∨ g i = -1) ∧
  (f ≠ g) ∧
  (f 0 * 1 + f 1 * 2 + f 2 * 3 + f 3 * 4 + f 4 * 5 + f 5 * 6 = 3) ∧
  (g 0 * 1 + g 1 * 2 + g 2 * 3 + g 3 * 4 + g 4 * 5 + g 5 * 6 = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_ways_to_get_three_l278_27826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_for_given_radius_and_angle_l278_27850

/-- The area of a circular sector given its radius and central angle in radians -/
noncomputable def sectorArea (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  (1 / 2) * radius^2 * centralAngle

theorem sector_area_for_given_radius_and_angle :
  let radius : ℝ := 2
  let centralAngle : ℝ := 2
  sectorArea radius centralAngle = 4 := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_for_given_radius_and_angle_l278_27850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_three_similar_parts_l278_27861

theorem impossibility_of_three_similar_parts (x : ℝ) (hx : x > 0) :
  ¬ ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    x = a + b + c ∧
    (a ≤ Real.sqrt 2 * b ∧ b ≤ Real.sqrt 2 * a) ∧
    (b ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * b) ∧
    (a ≤ Real.sqrt 2 * c ∧ c ≤ Real.sqrt 2 * a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_three_similar_parts_l278_27861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_transitivity_l278_27862

/-- A plane in 3D space -/
structure Plane where

/-- A line in 3D space -/
structure Line where

/-- Perpendicular relation between a line and a plane -/
def perpendicular (l : Line) (p : Plane) : Prop := sorry

/-- The theorem stating that if a line n is perpendicular to two planes α and β,
    and another line m is perpendicular to α, then m is perpendicular to β -/
theorem perpendicular_transitivity 
  (α β : Plane) (m n : Line) 
  (h1 : perpendicular n α) 
  (h2 : perpendicular n β) 
  (h3 : perpendicular m α) : 
  perpendicular m β := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_transitivity_l278_27862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_l278_27875

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem odd_function_condition (a : ℝ) : 
  (∀ x, f a x = -f a (-x)) ↔ a = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_condition_l278_27875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l278_27895

-- Expression 1
theorem simplify_expression_1 :
  Real.sqrt (9 / 4) - (-2017 : ℝ) ^ (0 : ℝ) - (27 / 8 : ℝ) ^ (2 / 3 : ℝ) = -7 / 4 := by sorry

-- Expression 2
theorem simplify_expression_2 :
  Real.log 5 / Real.log 10 + (Real.log 2 / Real.log 10) ^ 2 + 
  (Real.log 5 / Real.log 10) * (Real.log 2 / Real.log 10) + Real.log (Real.sqrt (Real.exp 1)) = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_1_simplify_expression_2_l278_27895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_strictly_increasing_l278_27888

noncomputable section

variable (e R r : ℝ)
variable (n : ℝ)

-- Define the conditions
axiom e_pos : 0 < e
axiom R_pos : 0 < R
axiom r_pos : 0 < r
axiom n_pos : 0 < n

-- Define the function C
noncomputable def C (n : ℝ) : ℝ := (e * n) / (R + n * r)

-- State the theorem
theorem C_strictly_increasing :
  ∀ n₁ n₂, 0 < n₁ ∧ 0 < n₂ ∧ n₁ < n₂ → C e R r n₁ < C e R r n₂ :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_strictly_increasing_l278_27888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_l278_27891

theorem place_mat_length (r : ℝ) (w : ℝ) (n : ℕ) (x : ℝ) : 
  r = 4 → w = 1 → n = 6 → 
  (2 * r * Real.sin (π / n : ℝ) = x) →
  (r^2 = (w/2)^2 + (x - w/2)^2) →
  x = (3 * Real.sqrt 7 - Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_place_mat_length_l278_27891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_test_l278_27870

/-- Represents a student's answer to a single question -/
inductive Answer
| A
| B
| C

/-- Represents a student's answers to all questions on the test -/
def StudentAnswers := Fin 4 → Answer

/-- The property that for any three students, there is at least one question where their answers differ -/
def DifferentAnswersExist (students : Finset StudentAnswers) : Prop :=
  ∀ s1 s2 s3, s1 ∈ students → s2 ∈ students → s3 ∈ students →
    s1 ≠ s2 → s2 ≠ s3 → s1 ≠ s3 →
    ∃ q : Fin 4, s1 q ≠ s2 q ∧ s2 q ≠ s3 q ∧ s1 q ≠ s3 q

/-- The maximum number of students that can take the test while satisfying the property -/
def MaxStudents : ℕ := 9

theorem max_students_test :
  ∃ (students : Finset StudentAnswers),
    students.card = MaxStudents ∧
    DifferentAnswersExist students ∧
    ∀ (larger_group : Finset StudentAnswers),
      larger_group.card > MaxStudents →
      ¬DifferentAnswersExist larger_group := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_students_test_l278_27870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l278_27825

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of line l1: ax + 2y = 0 -/
noncomputable def slope_l1 (a : ℝ) : ℝ := -a / 2

/-- The slope of line l2: x + (a + 1)y + 4 = 0 -/
noncomputable def slope_l2 (a : ℝ) : ℝ := -1 / (a + 1)

/-- The condition that a = 1 is sufficient but not necessary for the lines to be parallel -/
theorem parallel_condition (a : ℝ) :
  (a = 1 → are_parallel (slope_l1 a) (slope_l2 a)) ∧
  ¬(are_parallel (slope_l1 a) (slope_l2 a) → a = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_condition_l278_27825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_m_l278_27802

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 1/x + 5

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := x^2 - m

theorem solve_for_m :
  ∃ m : ℝ, f 3 - g m 3 = 8 ∧ m = -44/3 := by
  use -44/3
  constructor
  · -- Prove f 3 - g (-44/3) 3 = 8
    simp [f, g]
    -- You can add more steps here to complete the proof
    sorry
  · -- Prove -44/3 = -44/3
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_m_l278_27802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_l278_27899

open Real

-- Define the differential equation
def diff_eq (x : ℝ) (y : ℝ → ℝ) (m : ℝ) : Prop :=
  deriv y x = m^2 / x^4 - (y x)^2

-- Define the particular solutions
noncomputable def y1 (x m : ℝ) : ℝ := 1/x + m/x^2
noncomputable def y2 (x m : ℝ) : ℝ := 1/x - m/x^2

-- State the theorem
theorem general_solution (m : ℝ) :
  ∃ (C : ℝ), ∀ (x : ℝ) (y : ℝ → ℝ), x ≠ 0 →
    diff_eq x y m →
    (x^2 * y x - x - m) / (x^2 * y x - x + m) = C * exp (2*m/x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_general_solution_l278_27899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_of_three_l278_27803

theorem prime_power_of_three (n : ℕ) (h : Nat.Prime (4^n + 2^n + 1)) :
  ∃ k : ℕ, n = 3^k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_power_of_three_l278_27803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinate_l278_27809

/-- A triangle with vertices at (3,6), (0,0), and (x,0) where x < 0 -/
structure ObtuseTri where
  x : ℝ
  h1 : x < 0

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (1/2) * base * height

/-- The theorem stating that if the triangle has an area of 36 square units, then x = -12 -/
theorem third_vertex_coordinate (t : ObtuseTri) :
  triangleArea (3 - t.x) 6 = 36 → t.x = -12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_vertex_coordinate_l278_27809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l_l278_27853

-- Define the region D
def region_D (x y : ℝ) : Prop := x + y > 0 ∧ x - y < 0

-- Define the distance product condition for point P
noncomputable def distance_product (x y : ℝ) : Prop :=
  (|x + y| / Real.sqrt 2) * (|x - y| / Real.sqrt 2) = 2

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  region_D x y ∧ distance_product x y

-- Define the point F
noncomputable def point_F : ℝ × ℝ := (2 * Real.sqrt 2, 0)

-- Define the line l passing through F
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 2 * Real.sqrt 2)

-- Define the condition for circle with diameter AB tangent to y-axis
def circle_tangent_condition (x₁ x₂ : ℝ) : Prop :=
  (x₁ + x₂) / 2 = (x₁ + x₂) / 4

-- Main theorem
theorem slope_of_line_l :
  ∃ (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ),
    curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
    circle_tangent_condition x₁ x₂ →
    k = -Real.sqrt (Real.sqrt 2 - 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l_l278_27853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_dms_20_23_l278_27841

/-- Converts decimal degrees to degrees, minutes, and seconds -/
noncomputable def decimalToDMS (d : ℝ) : ℕ × ℕ × ℕ :=
  let degrees := Int.floor d
  let minutes := Int.floor ((d - degrees) * 60)
  let seconds := Int.floor ((d - degrees - (minutes : ℝ) / 60) * 3600)
  (degrees.toNat, minutes.toNat, seconds.toNat)

/-- Checks if a given DMS representation is equivalent to a decimal degree value -/
def isDMSEquivalent (d : ℝ) (dms : ℕ × ℕ × ℕ) : Prop :=
  decimalToDMS d = dms

theorem decimal_to_dms_20_23 :
  isDMSEquivalent 20.23 (20, 13, 48) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_dms_20_23_l278_27841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_intersection_l278_27868

noncomputable section

/-- The parabola equation: x² = 2py, where p > 0 -/
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

/-- The hyperbola equation: x²/3 - y²/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2/3 - y^2/3 = 1

/-- The focus of the parabola -/
def focus (p : ℝ) : ℝ × ℝ := (0, p/2)

/-- The directrix of the parabola -/
def directrix (p : ℝ) (y : ℝ) : Prop := y = -p/2

/-- A point is on both the directrix and the hyperbola -/
def intersectionPoint (p x y : ℝ) : Prop := directrix p y ∧ hyperbola x y

/-- Triangle ABF is equilateral -/
def isEquilateralTriangle (A B F : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 
  (B.1 - F.1)^2 + (B.2 - F.2)^2 ∧
  (A.1 - F.1)^2 + (A.2 - F.2)^2 = 
  (B.1 - F.1)^2 + (B.2 - F.2)^2

theorem parabola_hyperbola_intersection (p : ℝ) :
  (∃ A B : ℝ × ℝ, 
    intersectionPoint p A.1 A.2 ∧ 
    intersectionPoint p B.1 B.2 ∧ 
    isEquilateralTriangle A B (focus p)) →
  p = 6 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_hyperbola_intersection_l278_27868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_net_population_increase_net_population_increase_value_l278_27885

/-- Calculates the net population increase over one day given birth and death rates -/
theorem net_population_increase
  (birth_rate : ℚ)  -- Birth rate in people per two seconds
  (death_rate : ℚ)  -- Death rate in people per two seconds
  (h1 : birth_rate = 6)
  (h2 : death_rate = 2)
  : ℤ := by
  let net_rate_per_second : ℚ := (birth_rate - death_rate) / 2
  let seconds_per_day : ℕ := 24 * 60 * 60
  let net_increase : ℚ := net_rate_per_second * seconds_per_day
  exact ⌊net_increase⌋

theorem net_population_increase_value :
  net_population_increase 6 2 rfl rfl = 172800 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_net_population_increase_net_population_increase_value_l278_27885
