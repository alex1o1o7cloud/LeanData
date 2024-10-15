import Mathlib

namespace NUMINAMATH_CALUDE_nina_age_l3124_312414

/-- Given the ages of Max, Leah, Alex, and Nina, prove Nina's age --/
theorem nina_age (max_age leah_age alex_age nina_age : ℕ) 
  (h1 : max_age = leah_age - 5)
  (h2 : leah_age = alex_age + 6)
  (h3 : nina_age = alex_age + 2)
  (h4 : max_age = 16) : 
  nina_age = 17 := by
  sorry

#check nina_age

end NUMINAMATH_CALUDE_nina_age_l3124_312414


namespace NUMINAMATH_CALUDE_fraction_of_third_is_eighth_l3124_312422

theorem fraction_of_third_is_eighth : (1 / 8) / (1 / 3) = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_fraction_of_third_is_eighth_l3124_312422


namespace NUMINAMATH_CALUDE_complex_power_difference_l3124_312474

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^16 - (1 - i)^16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l3124_312474


namespace NUMINAMATH_CALUDE_science_team_selection_ways_l3124_312436

theorem science_team_selection_ways (total_boys : ℕ) (total_girls : ℕ) 
  (team_size : ℕ) (required_boys : ℕ) (required_girls : ℕ) : 
  total_boys = 7 → total_girls = 10 → team_size = 8 → 
  required_boys = 4 → required_girls = 4 →
  (Nat.choose total_boys required_boys) * (Nat.choose total_girls required_girls) = 7350 :=
by sorry

end NUMINAMATH_CALUDE_science_team_selection_ways_l3124_312436


namespace NUMINAMATH_CALUDE_minimum_toddlers_l3124_312412

theorem minimum_toddlers (total_teeth : ℕ) (max_pair_teeth : ℕ) (h1 : total_teeth = 90) (h2 : max_pair_teeth = 9) :
  ∃ (n : ℕ), n ≥ 23 ∧
  (∀ (m : ℕ), m < n →
    ¬∃ (teeth_distribution : Fin m → ℕ),
      (∀ i j : Fin m, i ≠ j → teeth_distribution i + teeth_distribution j ≤ max_pair_teeth) ∧
      (Finset.sum (Finset.univ : Finset (Fin m)) teeth_distribution = total_teeth)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_toddlers_l3124_312412


namespace NUMINAMATH_CALUDE_power_difference_equals_one_ninth_l3124_312443

theorem power_difference_equals_one_ninth (x y : ℕ) : 
  (2^x : ℕ) ∣ 360 ∧ 
  ∀ k > x, ¬((2^k : ℕ) ∣ 360) ∧ 
  (5^y : ℕ) ∣ 360 ∧ 
  ∀ m > y, ¬((5^m : ℕ) ∣ 360) → 
  (1/3 : ℚ)^(x - y) = 1/9 := by
sorry

end NUMINAMATH_CALUDE_power_difference_equals_one_ninth_l3124_312443


namespace NUMINAMATH_CALUDE_floor_of_neg_two_point_seven_l3124_312450

-- Define the greatest integer function
def floor (x : ℝ) : ℤ := sorry

-- State the theorem
theorem floor_of_neg_two_point_seven :
  floor (-2.7) = -3 := by sorry

end NUMINAMATH_CALUDE_floor_of_neg_two_point_seven_l3124_312450


namespace NUMINAMATH_CALUDE_ripe_apples_weight_l3124_312444

/-- Given the total number of apples, the number of unripe apples, and the weight of each ripe apple,
    prove that the total weight of ripe apples is equal to the product of the number of ripe apples
    and the weight of each ripe apple. -/
theorem ripe_apples_weight
  (total_apples : ℕ)
  (unripe_apples : ℕ)
  (ripe_apple_weight : ℕ)
  (h1 : unripe_apples ≤ total_apples) :
  (total_apples - unripe_apples) * ripe_apple_weight =
    (total_apples - unripe_apples) * ripe_apple_weight :=
by sorry

end NUMINAMATH_CALUDE_ripe_apples_weight_l3124_312444


namespace NUMINAMATH_CALUDE_stating_isosceles_triangle_base_height_l3124_312487

/-- Represents an isosceles triangle with leg length a -/
structure IsoscelesTriangle (a : ℝ) where
  (a_pos : a > 0)

/-- The height from one leg to the other leg forms a 30° angle -/
def height_angle (t : IsoscelesTriangle a) : ℝ := 30

/-- The height from the base of the isosceles triangle -/
def base_height (t : IsoscelesTriangle a) : Set ℝ :=
  {h | h = (Real.sqrt 3 / 2) * a ∨ h = (1 / 2) * a}

/-- 
  Theorem stating that for an isosceles triangle with leg length a, 
  where the height from one leg to the other leg forms a 30° angle, 
  the height from the base is either (√3/2)a or (1/2)a.
-/
theorem isosceles_triangle_base_height (a : ℝ) (t : IsoscelesTriangle a) :
  ∀ h, h ∈ base_height t ↔ 
    (h = (Real.sqrt 3 / 2) * a ∨ h = (1 / 2) * a) ∧ 
    height_angle t = 30 := by
  sorry

end NUMINAMATH_CALUDE_stating_isosceles_triangle_base_height_l3124_312487


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l3124_312463

/-- The distance between the foci of the ellipse (x²/36) + (y²/9) = 9 is 2√3 -/
theorem ellipse_foci_distance :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 / 36 + y^2 / 9 = 9}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ 
    ∀ p ∈ ellipse, dist p f₁ + dist p f₂ = 2 * Real.sqrt 36 ∧
    dist f₁ f₂ = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l3124_312463


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3124_312439

def M : Set ℕ := {0, 1, 3}

def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_of_M_and_N : M ∪ N = {0, 1, 3, 9} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3124_312439


namespace NUMINAMATH_CALUDE_food_distribution_l3124_312415

/-- The number of days the food initially lasts -/
def initial_days : ℝ := 45

/-- The initial number of men in the camp -/
def initial_men : ℕ := 40

/-- The number of days the food lasts after additional men join -/
def final_days : ℝ := 32.73

/-- The number of additional men who joined the camp -/
def additional_men : ℕ := 15

theorem food_distribution (total_food : ℝ) :
  total_food = initial_men * initial_days ∧
  total_food = (initial_men + additional_men) * final_days :=
sorry

#check food_distribution

end NUMINAMATH_CALUDE_food_distribution_l3124_312415


namespace NUMINAMATH_CALUDE_train_platform_passing_time_l3124_312427

theorem train_platform_passing_time :
  let train_length : ℝ := 360
  let platform_length : ℝ := 390
  let train_speed_kmh : ℝ := 45
  let total_distance : ℝ := train_length + platform_length
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  let time : ℝ := total_distance / train_speed_ms
  time = 60 := by sorry

end NUMINAMATH_CALUDE_train_platform_passing_time_l3124_312427


namespace NUMINAMATH_CALUDE_perpendicular_points_coplanar_l3124_312475

-- Define the types for points and spheres
variable (Point Sphere : Type)

-- Define the property of a point being on a sphere
variable (onSphere : Point → Sphere → Prop)

-- Define the property of points being distinct
variable (distinct : List Point → Prop)

-- Define the property of points being coplanar
variable (coplanar : List Point → Prop)

-- Define the property of a point being on a line
variable (onLine : Point → Point → Point → Prop)

-- Define the property of a line being perpendicular to another line
variable (perpendicular : Point → Point → Point → Point → Prop)

-- Define the quadrilateral pyramid inscribed in a sphere
variable (S A B C D : Point) (sphere1 : Sphere)
variable (inscribed : onSphere S sphere1 ∧ onSphere A sphere1 ∧ onSphere B sphere1 ∧ onSphere C sphere1 ∧ onSphere D sphere1)

-- Define the perpendicular points
variable (A1 B1 C1 D1 : Point)
variable (perp : perpendicular A A1 S C ∧ perpendicular B B1 S D ∧ perpendicular C C1 S A ∧ perpendicular D D1 S B)

-- Define the property of perpendicular points being on the respective lines
variable (onLines : onLine A1 S C ∧ onLine B1 S D ∧ onLine C1 S A ∧ onLine D1 S B)

-- Define the property of S, A1, B1, C1, D1 being distinct and on another sphere
variable (sphere2 : Sphere)
variable (distinctOnSphere : distinct [S, A1, B1, C1, D1] ∧ 
                             onSphere S sphere2 ∧ onSphere A1 sphere2 ∧ onSphere B1 sphere2 ∧ onSphere C1 sphere2 ∧ onSphere D1 sphere2)

-- Theorem statement
theorem perpendicular_points_coplanar : 
  coplanar [A1, B1, C1, D1] :=
sorry

end NUMINAMATH_CALUDE_perpendicular_points_coplanar_l3124_312475


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l3124_312402

theorem ratio_of_sum_and_difference (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y = 6 * (x - y)) : x / y = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l3124_312402


namespace NUMINAMATH_CALUDE_additional_cars_needed_l3124_312488

def current_cars : ℕ := 23
def cars_per_row : ℕ := 6

theorem additional_cars_needed :
  let next_multiple := (current_cars + cars_per_row - 1) / cars_per_row * cars_per_row
  next_multiple - current_cars = 1 := by sorry

end NUMINAMATH_CALUDE_additional_cars_needed_l3124_312488


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l3124_312484

theorem inequality_solution_sets (a b x : ℝ) :
  let f := fun x => b * x^2 - (3 * a * b - b) * x + 2 * a^2 * b - a * b
  (∀ x, b = 1 ∧ a > 1 → (f x < 0 ↔ a < x ∧ x < 2 * a - 1)) ∧
  (∀ x, b = a ∧ a ≤ 1 → 
    ((a = 0 ∨ a = 1) → ¬∃ x, f x < 0) ∧
    (0 < a ∧ a < 1 → (f x < 0 ↔ 2 * a - 1 < x ∧ x < a)) ∧
    (a < 0 → (f x < 0 ↔ x < 2 * a - 1 ∨ x > a))) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l3124_312484


namespace NUMINAMATH_CALUDE_line_equation_l3124_312453

/-- Given a line passing through (b, 0) and (0, h), forming a triangle with area T' in the second quadrant where b > 0, prove that the equation of the line is -2T'x + b²y + 2T'b = 0. -/
theorem line_equation (b T' : ℝ) (h : ℝ) (hb : b > 0) : 
  ∃ (x y : ℝ → ℝ), ∀ t, -2 * T' * x t + b^2 * y t + 2 * T' * b = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l3124_312453


namespace NUMINAMATH_CALUDE_units_digit_of_k97_l3124_312413

-- Define the modified Lucas sequence
def modifiedLucas : ℕ → ℕ
  | 0 => 3
  | 1 => 1
  | n + 2 => modifiedLucas (n + 1) + modifiedLucas n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

-- Theorem statement
theorem units_digit_of_k97 : unitsDigit (modifiedLucas 97) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k97_l3124_312413


namespace NUMINAMATH_CALUDE_special_function_is_identity_l3124_312406

/-- A function satisfying certain conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x ≤ x) ∧ (∀ x y, f (x + y) ≤ f x + f y)

/-- Theorem: If f is a SpecialFunction, then f(x) = x for all x in ℝ -/
theorem special_function_is_identity (f : ℝ → ℝ) (hf : SpecialFunction f) :
  ∀ x, f x = x := by
  sorry

end NUMINAMATH_CALUDE_special_function_is_identity_l3124_312406


namespace NUMINAMATH_CALUDE_calculation_proof_l3124_312460

theorem calculation_proof : 
  57.6 * (8 / 5) + 28.8 * (184 / 5) - 14.4 * 80 + 10.5 = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3124_312460


namespace NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l3124_312499

/-- The perpendicular bisector of a line segment from (2, 5) to (8, 11) has equation 2x - y = c. -/
theorem perpendicular_bisector_c_value :
  ∃ (c : ℝ), 
    (∀ (x y : ℝ), (2 * x - y = c) ↔ 
      (x - 5)^2 + (y - 8)^2 = (5 - 2)^2 + (8 - 5)^2 ∧ 
      (x - 5) * (8 - 2) = -(y - 8) * (11 - 5)) → 
    c = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l3124_312499


namespace NUMINAMATH_CALUDE_officers_selection_count_l3124_312440

/-- The number of ways to choose officers from a club -/
def choose_officers (total_members : ℕ) (senior_members : ℕ) (positions : ℕ) : ℕ :=
  senior_members * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem stating the number of ways to choose officers under given conditions -/
theorem officers_selection_count :
  choose_officers 12 4 5 = 31680 := by
  sorry

end NUMINAMATH_CALUDE_officers_selection_count_l3124_312440


namespace NUMINAMATH_CALUDE_vector_coordinates_l3124_312464

/-- Given two vectors a and b in ℝ², prove that if a is parallel to b, 
    a = (2, -1), and the magnitude of b is 2√5, then b is either (-4, 2) or (4, -2) -/
theorem vector_coordinates (a b : ℝ × ℝ) : 
  (∃ (k : ℝ), b = k • a) →  -- a is parallel to b
  a = (2, -1) →
  Real.sqrt ((b.1)^2 + (b.2)^2) = 2 * Real.sqrt 5 →  -- magnitude of b is 2√5
  (b = (-4, 2) ∨ b = (4, -2)) :=
by sorry

end NUMINAMATH_CALUDE_vector_coordinates_l3124_312464


namespace NUMINAMATH_CALUDE_complex_magnitude_sum_l3124_312454

theorem complex_magnitude_sum : Complex.abs (3 - 5*I) + Complex.abs (3 + 5*I) = 2 * Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_sum_l3124_312454


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3124_312420

theorem rectangle_perimeter (length width : ℝ) : 
  length / width = 4 / 3 →
  length * width = 972 →
  2 * (length + width) = 126 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3124_312420


namespace NUMINAMATH_CALUDE_abs_over_a_plus_one_l3124_312459

theorem abs_over_a_plus_one (a : ℝ) (h : a ≠ 0) :
  (|a| / a + 1 = 0) ∨ (|a| / a + 1 = 2) := by
  sorry

end NUMINAMATH_CALUDE_abs_over_a_plus_one_l3124_312459


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_range_of_b_when_a_zero_sum_of_squares_greater_than_e_l3124_312469

noncomputable section

variables (a b : ℝ)

def f (x : ℝ) : ℝ := Real.exp x - a * Real.sin x
def g (x : ℝ) : ℝ := b * Real.sqrt x

-- Tangent line equation
theorem tangent_line_at_zero :
  ∃ m c : ℝ, ∀ x : ℝ, m * x + c = (1 - a) * x + 1 :=
sorry

-- Range of b when a = 0
theorem range_of_b_when_a_zero (h : a = 0) :
  ∃ x > 0, f x = g x ↔ b ≥ Real.sqrt (2 * Real.exp 1) :=
sorry

-- Proof of a^2 + b^2 > e
theorem sum_of_squares_greater_than_e (h : ∃ x > 0, f x = g x) :
  a^2 + b^2 > Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_range_of_b_when_a_zero_sum_of_squares_greater_than_e_l3124_312469


namespace NUMINAMATH_CALUDE_correct_total_amount_l3124_312489

/-- Calculate the total amount paid for grapes and mangoes -/
def totalAmountPaid (grapeQuantity : ℕ) (grapeRate : ℕ) (mangoQuantity : ℕ) (mangoRate : ℕ) : ℕ :=
  grapeQuantity * grapeRate + mangoQuantity * mangoRate

/-- Theorem stating that the total amount paid is correct -/
theorem correct_total_amount :
  totalAmountPaid 10 70 9 55 = 1195 := by
  sorry

#eval totalAmountPaid 10 70 9 55

end NUMINAMATH_CALUDE_correct_total_amount_l3124_312489


namespace NUMINAMATH_CALUDE_smallest_four_digit_sum_20_l3124_312445

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is four digits -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- Theorem: 1999 is the smallest four-digit number whose digits sum to 20 -/
theorem smallest_four_digit_sum_20 : 
  (∀ n : ℕ, is_four_digit n → sum_of_digits n = 20 → 1999 ≤ n) ∧ 
  (is_four_digit 1999 ∧ sum_of_digits 1999 = 20) := by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_sum_20_l3124_312445


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l3124_312496

theorem expression_simplification_and_evaluation (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (1 - x / (x + 1)) / ((x^2 - 2*x + 1) / (x^2 - 1)) = 1 / (x - 1) :=
sorry

theorem expression_evaluation_at_3 :
  (1 - 3 / (3 + 1)) / ((3^2 - 2*3 + 1) / (3^2 - 1)) = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_3_l3124_312496


namespace NUMINAMATH_CALUDE_factorial_divisibility_l3124_312467

theorem factorial_divisibility (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 100) :
  ∃ k : ℕ, (Nat.factorial (n^2 + 1)) = k * (Nat.factorial n)^(n + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l3124_312467


namespace NUMINAMATH_CALUDE_second_term_of_sequence_l3124_312481

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem second_term_of_sequence (a d : ℤ) :
  arithmetic_sequence a d 12 = 11 →
  arithmetic_sequence a d 13 = 14 →
  arithmetic_sequence a d 2 = -19 :=
by
  sorry

end NUMINAMATH_CALUDE_second_term_of_sequence_l3124_312481


namespace NUMINAMATH_CALUDE_find_x_l3124_312411

theorem find_x (x y z a b c d k : ℝ) 
  (h1 : (x * y + k) / (x + y) = a)
  (h2 : (x * z + k) / (x + z) = b)
  (h3 : (y * z + k) / (y + z) = c)
  (hk : k ≠ 0) :
  x = (2 * a * b * c * d) / (b * (a * c - k) + c * (a * b - k) - a * (b * c - k)) :=
sorry

end NUMINAMATH_CALUDE_find_x_l3124_312411


namespace NUMINAMATH_CALUDE_inequality_property_l3124_312416

theorem inequality_property (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l3124_312416


namespace NUMINAMATH_CALUDE_power_equation_implies_m_equals_one_l3124_312421

theorem power_equation_implies_m_equals_one (s m : ℕ) :
  (2^16) * (25^s) = 5 * (10^m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_implies_m_equals_one_l3124_312421


namespace NUMINAMATH_CALUDE_solution_form_l3124_312483

/-- A continuous function satisfying the given integral equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  Continuous f ∧
  ∀ (x : ℝ) (n : ℕ), n ≠ 0 →
    (n : ℝ)^2 * ∫ t in x..(x + 1 / (n : ℝ)), f t = (n : ℝ) * f x + 1 / 2

/-- The theorem stating the form of functions satisfying the equation -/
theorem solution_form (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end NUMINAMATH_CALUDE_solution_form_l3124_312483


namespace NUMINAMATH_CALUDE_chris_bill_calculation_l3124_312470

/-- Calculates the total internet bill based on base charge, overage rate, and data usage over the limit. -/
def total_bill (base_charge : ℝ) (overage_rate : ℝ) (data_over_limit : ℝ) : ℝ :=
  base_charge + overage_rate * data_over_limit

/-- Theorem stating that Chris's total bill is equal to the sum of the base charge and overage charge. -/
theorem chris_bill_calculation (base_charge overage_rate data_over_limit : ℝ) :
  total_bill base_charge overage_rate data_over_limit = base_charge + overage_rate * data_over_limit :=
by sorry

end NUMINAMATH_CALUDE_chris_bill_calculation_l3124_312470


namespace NUMINAMATH_CALUDE_betty_garden_ratio_l3124_312417

/-- Represents a herb garden with oregano and basil plants -/
structure HerbGarden where
  total_plants : ℕ
  basil_plants : ℕ
  oregano_plants : ℕ
  total_eq : total_plants = oregano_plants + basil_plants

/-- The ratio of oregano to basil plants in Betty's garden is 12:5 -/
theorem betty_garden_ratio (garden : HerbGarden) 
    (h1 : garden.total_plants = 17)
    (h2 : garden.basil_plants = 5) :
    garden.oregano_plants / garden.basil_plants = 12 / 5 := by
  sorry

#check betty_garden_ratio

end NUMINAMATH_CALUDE_betty_garden_ratio_l3124_312417


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l3124_312404

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^13 + i^113 = 2*i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l3124_312404


namespace NUMINAMATH_CALUDE_division_problem_l3124_312476

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 127 ∧ quotient = 9 ∧ remainder = 1 ∧ 
  dividend = divisor * quotient + remainder →
  divisor = 14 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3124_312476


namespace NUMINAMATH_CALUDE_perfect_cube_factors_of_72_is_two_l3124_312477

/-- A function that returns the number of positive factors of 72 that are perfect cubes -/
def perfect_cube_factors_of_72 : ℕ :=
  -- The function should return the number of positive factors of 72 that are perfect cubes
  sorry

/-- Theorem stating that the number of positive factors of 72 that are perfect cubes is 2 -/
theorem perfect_cube_factors_of_72_is_two : perfect_cube_factors_of_72 = 2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_factors_of_72_is_two_l3124_312477


namespace NUMINAMATH_CALUDE_brandon_textbook_weight_l3124_312424

def jon_textbook_weights : List ℝ := [2, 8, 5, 9]

theorem brandon_textbook_weight (brandon_weight : ℝ) : 
  (List.sum jon_textbook_weights = 3 * brandon_weight) → brandon_weight = 8 := by
  sorry

end NUMINAMATH_CALUDE_brandon_textbook_weight_l3124_312424


namespace NUMINAMATH_CALUDE_divisors_of_20_factorial_l3124_312479

theorem divisors_of_20_factorial : (Nat.divisors (Nat.factorial 20)).card = 41040 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_20_factorial_l3124_312479


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l3124_312472

theorem geometric_sequence_general_term (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  a 3 = 5/2 →
  S 3 = 15/2 →
  (∀ n, a n = 5/2) ∨ (∀ n, a n = 10 * (-1/2)^(n-1)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l3124_312472


namespace NUMINAMATH_CALUDE_distance_to_first_sign_l3124_312478

/-- Given a bike ride with two stop signs, calculate the distance to the first stop sign -/
theorem distance_to_first_sign
  (total_distance : ℕ)
  (distance_after_second_sign : ℕ)
  (distance_between_signs : ℕ)
  (h1 : total_distance = 1000)
  (h2 : distance_after_second_sign = 275)
  (h3 : distance_between_signs = 375) :
  total_distance - distance_after_second_sign - distance_between_signs = 350 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_first_sign_l3124_312478


namespace NUMINAMATH_CALUDE_cookies_leftover_l3124_312482

/-- The number of cookies Amelia has -/
def ameliaCookies : ℕ := 52

/-- The number of cookies Benjamin has -/
def benjaminCookies : ℕ := 63

/-- The number of cookies Chloe has -/
def chloeCookies : ℕ := 25

/-- The number of cookies in each package -/
def packageSize : ℕ := 15

/-- The total number of cookies -/
def totalCookies : ℕ := ameliaCookies + benjaminCookies + chloeCookies

/-- The number of cookies left over after packaging -/
def leftoverCookies : ℕ := totalCookies % packageSize

theorem cookies_leftover : leftoverCookies = 5 := by
  sorry

end NUMINAMATH_CALUDE_cookies_leftover_l3124_312482


namespace NUMINAMATH_CALUDE_inequality_proofs_l3124_312448

theorem inequality_proofs :
  (∀ x : ℝ, x * (1 - x) ≤ 1 / 4) ∧
  (∀ x a : ℝ, x * (a - x) ≤ a^2 / 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l3124_312448


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3124_312405

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - 2 * |x + a|

-- Part 1: Solution set for f(x) > 1 when a = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x > 1} = {x : ℝ | -2 < x ∧ x < -2/3} :=
sorry

-- Part 2: Range of a for f(x) > 0 when x ∈ [2, 3]
theorem range_of_a_part2 :
  {a : ℝ | ∀ x ∈ Set.Icc 2 3, f a x > 0} = {a : ℝ | -5/2 < a ∧ a < -2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3124_312405


namespace NUMINAMATH_CALUDE_expression_equality_l3124_312457

theorem expression_equality (n : ℕ) (h : n ≥ 1) :
  (5^(n+1) + 6^(n+2))^2 - (5^(n+1) - 6^(n+2))^2 = 144 * 30^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3124_312457


namespace NUMINAMATH_CALUDE_triangle_abc_isosceles_l3124_312455

/-- Given points A(3,5), B(-6,-2), and C(0,-6), prove that AB = AC -/
theorem triangle_abc_isosceles (A B C : ℝ × ℝ) : 
  A = (3, 5) → B = (-6, -2) → C = (0, -6) → 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 := by
  sorry

#check triangle_abc_isosceles

end NUMINAMATH_CALUDE_triangle_abc_isosceles_l3124_312455


namespace NUMINAMATH_CALUDE_modulo_residue_problem_l3124_312423

theorem modulo_residue_problem : (348 + 8 * 58 + 9 * 195 + 6 * 29) % 19 = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_problem_l3124_312423


namespace NUMINAMATH_CALUDE_nine_digit_divisibility_l3124_312446

theorem nine_digit_divisibility (a b c : Nat) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9) (h4 : a ≠ 0) :
  ∃ k : Nat, (100 * a + 10 * b + c) * 1001001 = k * (100000000 * a + 10000000 * b + 1000000 * c +
                                                     100000 * a + 10000 * b + 1000 * c +
                                                     100 * a + 10 * b + c) :=
sorry

end NUMINAMATH_CALUDE_nine_digit_divisibility_l3124_312446


namespace NUMINAMATH_CALUDE_subtraction_problem_l3124_312432

theorem subtraction_problem (M N : ℕ) : 
  M < 10 → N < 10 → M * 10 + 4 - (30 + N) = 16 → M + N = 13 := by
sorry

end NUMINAMATH_CALUDE_subtraction_problem_l3124_312432


namespace NUMINAMATH_CALUDE_baking_powder_difference_l3124_312492

/-- The amount of baking powder Kelly had yesterday, in boxes -/
def yesterday_supply : ℚ := 0.4

/-- The amount of baking powder Kelly has today, in boxes -/
def today_supply : ℚ := 0.3

/-- The difference in baking powder supply between yesterday and today -/
def supply_difference : ℚ := yesterday_supply - today_supply

theorem baking_powder_difference :
  supply_difference = 0.1 := by sorry

end NUMINAMATH_CALUDE_baking_powder_difference_l3124_312492


namespace NUMINAMATH_CALUDE_sally_peaches_theorem_l3124_312495

/-- The number of peaches Sally picked from the orchard -/
def peaches_picked (initial final : ℕ) : ℕ := final - initial

theorem sally_peaches_theorem (initial final picked : ℕ) 
  (h1 : initial = 13)
  (h2 : final = 55)
  (h3 : picked = peaches_picked initial final) :
  picked = 42 := by sorry

end NUMINAMATH_CALUDE_sally_peaches_theorem_l3124_312495


namespace NUMINAMATH_CALUDE_geometry_theorem_l3124_312441

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)
variable (line_perpendicular_to_plane : Line → Plane → Prop)
variable (plane_intersection : Plane → Plane → Line)

-- Define the non-coincidence of lines and planes
variable (non_coincident_lines : Line → Line → Line → Prop)
variable (non_coincident_planes : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n l : Line)
variable (α β : Plane)

-- State the theorem
theorem geometry_theorem 
  (h_non_coincident_lines : non_coincident_lines m n l)
  (h_non_coincident_planes : non_coincident_planes α β) :
  (∀ (l m : Line) (α β : Plane),
    line_perpendicular_to_plane l α →
    line_perpendicular_to_plane m β →
    parallel l m →
    plane_parallel α β) ∧
  (∀ (m n : Line) (α β : Plane),
    plane_perpendicular α β →
    plane_intersection α β = m →
    line_in_plane n β →
    perpendicular n m →
    line_perpendicular_to_plane n α) :=
by sorry

end NUMINAMATH_CALUDE_geometry_theorem_l3124_312441


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l3124_312430

theorem quadratic_equation_roots_ratio (c : ℚ) : 
  (∃ x y : ℚ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 10*x + c = 0 ∧ y^2 + 10*y + c = 0) → 
  c = 75/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l3124_312430


namespace NUMINAMATH_CALUDE_line_segment_length_l3124_312468

/-- The hyperbola C with equation x^2 - y^2/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- The line l with equation y = 2√3x + m -/
def line (x y m : ℝ) : Prop := y = 2 * Real.sqrt 3 * x + m

/-- The right vertex of the hyperbola -/
def right_vertex (x y : ℝ) : Prop := hyperbola x y ∧ x > 0 ∧ y = 0

/-- The asymptotes of the hyperbola -/
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- The intersection points of the line and the asymptotes -/
def intersection_points (x₁ y₁ x₂ y₂ m : ℝ) : Prop :=
  line x₁ y₁ m ∧ asymptote x₁ y₁ ∧
  line x₂ y₂ m ∧ asymptote x₂ y₂ ∧
  x₁ ≠ x₂

/-- The theorem statement -/
theorem line_segment_length 
  (x y m x₁ y₁ x₂ y₂ : ℝ) :
  right_vertex x y →
  line x y m →
  intersection_points x₁ y₁ x₂ y₂ m →
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4 * Real.sqrt 13 / 3 :=
sorry

end NUMINAMATH_CALUDE_line_segment_length_l3124_312468


namespace NUMINAMATH_CALUDE_proportion_solution_l3124_312471

-- Define the conversion factor from minutes to seconds
def minutes_to_seconds (minutes : ℚ) : ℚ := 60 * minutes

-- Define the proportion
def proportion (x : ℚ) : Prop :=
  x / 4 = 8 / (minutes_to_seconds 4)

-- Theorem statement
theorem proportion_solution :
  ∃ (x : ℚ), proportion x ∧ x = 1 / 7.5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3124_312471


namespace NUMINAMATH_CALUDE_volunteer_assignment_count_l3124_312409

/-- Stirling number of the second kind -/
def stirling2 (n k : ℕ) : ℕ := sorry

/-- Number of ways to assign n volunteers to k tasks, where each task must have at least one person -/
def assignVolunteers (n k : ℕ) : ℕ := (stirling2 n k) * (Nat.factorial k)

theorem volunteer_assignment_count :
  assignVolunteers 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_volunteer_assignment_count_l3124_312409


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3124_312410

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 10) :
  (1 / x + 2 / y) ≥ (3 + 2 * Real.sqrt 2) / 10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3124_312410


namespace NUMINAMATH_CALUDE_hypotenuse_length_l3124_312407

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ  -- Length of one side
  b : ℝ  -- Length of another side
  c : ℝ  -- Length of the hypotenuse
  right_angle : c^2 = a^2 + b^2  -- Pythagorean theorem

-- Theorem statement
theorem hypotenuse_length (t : RightTriangle) 
  (sum_of_squares : t.a^2 + t.b^2 + t.c^2 = 1450) : 
  t.c = Real.sqrt 725 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l3124_312407


namespace NUMINAMATH_CALUDE_binomial_rv_unique_params_l3124_312462

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  hp : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expected_value (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- Variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: For a binomial random variable ξ with E(ξ) = 5/3 and D(ξ) = 10/9, n = 5 and p = 1/3 -/
theorem binomial_rv_unique_params (ξ : BinomialRV) 
  (h_exp : expected_value ξ = 5/3)
  (h_var : variance ξ = 10/9) :
  ξ.n = 5 ∧ ξ.p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_rv_unique_params_l3124_312462


namespace NUMINAMATH_CALUDE_min_value_of_g_l3124_312408

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a^x - 2 * a^(-x)

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2 * f a x

-- Theorem statement
theorem min_value_of_g (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 = 3) :
  ∃ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, g a x ≤ g a y ∧ g a x = -2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_g_l3124_312408


namespace NUMINAMATH_CALUDE_division_remainder_problem_l3124_312480

theorem division_remainder_problem (L S : ℕ) : 
  L - S = 1345 → 
  L = 1596 → 
  L / S = 6 → 
  L % S = 90 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l3124_312480


namespace NUMINAMATH_CALUDE_average_weight_increase_l3124_312418

/-- Proves that replacing a sailor weighing 56 kg with a sailor weighing 64 kg
    in a group of 8 sailors increases the average weight by 1 kg. -/
theorem average_weight_increase (initial_average : ℝ) : 
  let total_weight := 8 * initial_average
  let new_total_weight := total_weight - 56 + 64
  let new_average := new_total_weight / 8
  new_average - initial_average = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l3124_312418


namespace NUMINAMATH_CALUDE_final_count_A_l3124_312493

/-- Represents a switch with its ID and position -/
structure Switch where
  id : Nat
  position : Fin 3

/-- Represents the state of all switches -/
def SwitchState := Fin 1000 → Switch

/-- Checks if one number divides another -/
def divides (a b : Nat) : Prop := ∃ k, b = a * k

/-- Represents a single step in the process -/
def step (s : SwitchState) (i : Fin 1000) : SwitchState := sorry

/-- Represents the entire process of 1000 steps -/
def process (initial : SwitchState) : SwitchState := sorry

/-- Counts the number of switches in position A -/
def countA (s : SwitchState) : Nat := sorry

/-- The main theorem to prove -/
theorem final_count_A (initial : SwitchState) : 
  (∀ i, (initial i).position = 0) →
  (∀ i, ∃ x y z, x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧ (initial i).id = 2^x * 3^y * 7^z) →
  countA (process initial) = 660 := sorry

end NUMINAMATH_CALUDE_final_count_A_l3124_312493


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l3124_312434

theorem baseball_card_value_decrease (initial_value : ℝ) (h_initial_positive : initial_value > 0) : 
  let first_year_value := initial_value * (1 - 0.3)
  let total_decrease_percent := 0.37
  let second_year_decrease_percent := (initial_value * total_decrease_percent - (initial_value - first_year_value)) / first_year_value
  second_year_decrease_percent = 0.1 := by
sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l3124_312434


namespace NUMINAMATH_CALUDE_car_distance_traveled_l3124_312437

theorem car_distance_traveled (time : ℝ) (speed : ℝ) (distance : ℝ) : 
  time = 11 → speed = 65 → distance = time * speed → distance = 715 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_traveled_l3124_312437


namespace NUMINAMATH_CALUDE_chromium_percentage_in_first_alloy_l3124_312438

/-- Given two alloys, proves that the percentage of chromium in the first alloy is 12% -/
theorem chromium_percentage_in_first_alloy :
  let weight_first_alloy : ℝ := 15
  let weight_second_alloy : ℝ := 35
  let chromium_percentage_second_alloy : ℝ := 10
  let chromium_percentage_new_alloy : ℝ := 10.6
  let total_weight : ℝ := weight_first_alloy + weight_second_alloy
  ∃ (chromium_percentage_first_alloy : ℝ),
    chromium_percentage_first_alloy * weight_first_alloy / 100 +
    chromium_percentage_second_alloy * weight_second_alloy / 100 =
    chromium_percentage_new_alloy * total_weight / 100 ∧
    chromium_percentage_first_alloy = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_chromium_percentage_in_first_alloy_l3124_312438


namespace NUMINAMATH_CALUDE_exactly_two_correct_statements_l3124_312465

theorem exactly_two_correct_statements : 
  let f : ℝ → ℝ := λ x => x + 1/x
  let statement1 := ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ (∃ (x₀ : ℝ), f x₀ = m) ∧ m = 2
  let statement2 := ∀ (a b : ℝ), a^2 + b^2 ≥ 2*a*b
  let statement3 := ∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c > d ∧ d > 0 → a*c > b*d
  let statement4 := (¬∃ (x : ℝ), x^2 + x + 1 ≥ 0) ↔ (∀ (x : ℝ), x^2 + x + 1 ≥ 0)
  let statement5 := ∀ (x y : ℝ), x > y ↔ 1/x < 1/y
  let statement6 := ∀ (p q : Prop), (¬(p ∨ q)) → (¬(¬p ∨ ¬q))
  (statement2 ∧ statement3 ∧ ¬statement1 ∧ ¬statement4 ∧ ¬statement5 ∧ ¬statement6) := by sorry

end NUMINAMATH_CALUDE_exactly_two_correct_statements_l3124_312465


namespace NUMINAMATH_CALUDE_min_value_floor_sum_l3124_312401

theorem min_value_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (m : ℕ), m = 4 ∧
  (∀ (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0),
    ⌊(2*a + b) / c⌋ + ⌊(2*b + c) / a⌋ + ⌊(2*c + a) / b⌋ + ⌊(a + b + c) / (a + b)⌋ ≥ m) ∧
  (⌊(2*x + y) / z⌋ + ⌊(2*y + z) / x⌋ + ⌊(2*z + x) / y⌋ + ⌊(x + y + z) / (x + y)⌋ = m) :=
by sorry

end NUMINAMATH_CALUDE_min_value_floor_sum_l3124_312401


namespace NUMINAMATH_CALUDE_factors_of_210_l3124_312497

theorem factors_of_210 : Nat.card (Nat.divisors 210) = 16 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_210_l3124_312497


namespace NUMINAMATH_CALUDE_highway_work_completion_fraction_l3124_312494

theorem highway_work_completion_fraction :
  let total_length : ℝ := 2 -- km
  let initial_workers : ℕ := 100
  let initial_duration : ℕ := 50 -- days
  let initial_daily_hours : ℕ := 8
  let actual_work_days : ℕ := 25
  let additional_workers : ℕ := 60
  let new_daily_hours : ℕ := 10

  let total_man_hours : ℝ := initial_workers * initial_duration * initial_daily_hours
  let remaining_man_hours : ℝ := (initial_workers + additional_workers) * (initial_duration - actual_work_days) * new_daily_hours

  total_man_hours = remaining_man_hours →
  (total_man_hours - remaining_man_hours) / total_man_hours = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_highway_work_completion_fraction_l3124_312494


namespace NUMINAMATH_CALUDE_complex_equation_product_l3124_312490

theorem complex_equation_product (x y : ℝ) : 
  (Complex.I : ℂ) * x - (Complex.I : ℂ) * y + x + y = 2 → x * y = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_product_l3124_312490


namespace NUMINAMATH_CALUDE_afternoon_rowers_count_l3124_312491

/-- The number of campers who went rowing in the afternoon -/
def afternoon_rowers (morning_rowers total_rowers : ℕ) : ℕ :=
  total_rowers - morning_rowers

/-- Proof that 21 campers went rowing in the afternoon -/
theorem afternoon_rowers_count :
  afternoon_rowers 13 34 = 21 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_rowers_count_l3124_312491


namespace NUMINAMATH_CALUDE_a_range_l3124_312447

/-- Proposition p: A real number x satisfies 2 < x < 3 -/
def p (x : ℝ) : Prop := 2 < x ∧ x < 3

/-- Proposition q: A real number x satisfies 2x^2 - 9x + a < 0 -/
def q (x a : ℝ) : Prop := 2 * x^2 - 9 * x + a < 0

/-- p is a sufficient condition for q -/
def p_implies_q (a : ℝ) : Prop := ∀ x, p x → q x a

theorem a_range (a : ℝ) : p_implies_q a ↔ 7 ≤ a ∧ a ≤ 8 := by sorry

end NUMINAMATH_CALUDE_a_range_l3124_312447


namespace NUMINAMATH_CALUDE_two_a_squared_eq_three_b_cubed_l3124_312466

theorem two_a_squared_eq_three_b_cubed (a b : ℕ+) :
  2 * a ^ 2 = 3 * b ^ 3 ↔ ∃ d : ℕ+, a = 18 * d ^ 3 ∧ b = 6 * d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_two_a_squared_eq_three_b_cubed_l3124_312466


namespace NUMINAMATH_CALUDE_mary_milk_weight_l3124_312451

/-- Proves that the weight of milk Mary bought is 6 pounds -/
theorem mary_milk_weight (bag_capacity : ℕ) (green_beans_weight : ℕ) (remaining_capacity : ℕ) : 
  bag_capacity = 20 →
  green_beans_weight = 4 →
  remaining_capacity = 2 →
  6 = bag_capacity - remaining_capacity - (green_beans_weight + 2 * green_beans_weight) :=
by sorry

end NUMINAMATH_CALUDE_mary_milk_weight_l3124_312451


namespace NUMINAMATH_CALUDE_three_plants_three_colors_l3124_312461

/-- Represents the number of ways to assign plants to colored lamps -/
def plant_lamp_assignments (num_plants : ℕ) (num_identical_plants : ℕ) (num_colors : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the number of ways to assign 3 plants to 3 colors of lamps -/
theorem three_plants_three_colors :
  plant_lamp_assignments 3 2 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_three_plants_three_colors_l3124_312461


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l3124_312426

theorem no_integer_solutions_for_equation :
  ¬ ∃ (x y z : ℤ), x^2 + y^2 = 4*z - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_equation_l3124_312426


namespace NUMINAMATH_CALUDE_even_function_interval_l3124_312400

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the interval
def interval (m : ℝ) : Set ℝ := Set.Icc (2*m) (m+6)

-- State the theorem
theorem even_function_interval (m : ℝ) :
  (∀ x ∈ interval m, f x = f (-x)) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_interval_l3124_312400


namespace NUMINAMATH_CALUDE_sanitizer_sprays_common_kill_percentage_l3124_312431

theorem sanitizer_sprays_common_kill_percentage 
  (spray1_kill : Real) 
  (spray2_kill : Real) 
  (combined_survival : Real) 
  (h1 : spray1_kill = 0.5) 
  (h2 : spray2_kill = 0.25) 
  (h3 : combined_survival = 0.3) : 
  spray1_kill + spray2_kill - (1 - combined_survival) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_sanitizer_sprays_common_kill_percentage_l3124_312431


namespace NUMINAMATH_CALUDE_domain_of_g_is_closed_unit_interval_l3124_312429

-- Define the function f with domain [0,1]
def f : Set ℝ := Set.Icc 0 1

-- Define the function g(x) = f(x^2)
def g (x : ℝ) : Prop := x^2 ∈ f

-- Theorem statement
theorem domain_of_g_is_closed_unit_interval :
  {x : ℝ | g x} = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_domain_of_g_is_closed_unit_interval_l3124_312429


namespace NUMINAMATH_CALUDE_nail_salon_fingers_l3124_312473

theorem nail_salon_fingers (total_earnings : ℚ) (cost_per_manicure : ℚ) (total_fingers : ℕ) (non_clients : ℕ) :
  total_earnings = 200 →
  cost_per_manicure = 20 →
  total_fingers = 210 →
  non_clients = 11 →
  ∃ (fingers_per_person : ℕ), 
    fingers_per_person = 10 ∧
    (total_earnings / cost_per_manicure + non_clients : ℚ) * fingers_per_person = total_fingers := by
  sorry

end NUMINAMATH_CALUDE_nail_salon_fingers_l3124_312473


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l3124_312403

/-- Given a parallelogram with area 128 sq m and altitude twice the base, prove the base is 8 m -/
theorem parallelogram_base_length :
  ∀ (base altitude : ℝ),
  base > 0 →
  altitude > 0 →
  altitude = 2 * base →
  base * altitude = 128 →
  base = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l3124_312403


namespace NUMINAMATH_CALUDE_roses_mother_age_l3124_312425

theorem roses_mother_age (rose_age mother_age : ℕ) : 
  rose_age = mother_age / 3 →
  rose_age + mother_age = 100 →
  mother_age = 75 := by
sorry

end NUMINAMATH_CALUDE_roses_mother_age_l3124_312425


namespace NUMINAMATH_CALUDE_min_value_sin_function_l3124_312452

theorem min_value_sin_function : 
  ∀ x : ℝ, -Real.sin x ^ 3 - 2 * Real.sin x ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sin_function_l3124_312452


namespace NUMINAMATH_CALUDE_sqrt_64_times_sqrt_25_l3124_312456

theorem sqrt_64_times_sqrt_25 : Real.sqrt (64 * Real.sqrt 25) = 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_64_times_sqrt_25_l3124_312456


namespace NUMINAMATH_CALUDE_landscape_length_l3124_312433

/-- Given a rectangular landscape with a playground, calculate its length -/
theorem landscape_length (breadth : ℝ) (playground_area : ℝ) : 
  breadth > 0 →
  playground_area = 1200 →
  playground_area = (1 / 6) * (8 * breadth * breadth) →
  8 * breadth = 240 := by
  sorry

end NUMINAMATH_CALUDE_landscape_length_l3124_312433


namespace NUMINAMATH_CALUDE_distance_height_relation_l3124_312428

/-- An equilateral triangle with an arbitrary line in its plane -/
structure TriangleWithLine where
  /-- The side length of the equilateral triangle -/
  side : ℝ
  /-- The height of the equilateral triangle -/
  height : ℝ
  /-- The distance from the first vertex to the line -/
  m : ℝ
  /-- The distance from the second vertex to the line -/
  n : ℝ
  /-- The distance from the third vertex to the line -/
  p : ℝ
  /-- The side length is positive -/
  side_pos : 0 < side
  /-- The height is related to the side length as in an equilateral triangle -/
  height_eq : height = (Real.sqrt 3 / 2) * side

/-- The main theorem stating the relationship between distances and height -/
theorem distance_height_relation (t : TriangleWithLine) :
  (t.m - t.n)^2 + (t.n - t.p)^2 + (t.p - t.m)^2 = 2 * t.height^2 := by
  sorry

end NUMINAMATH_CALUDE_distance_height_relation_l3124_312428


namespace NUMINAMATH_CALUDE_simplify_fraction_l3124_312485

theorem simplify_fraction : (4^5 + 4^3) / (4^4 - 4^2) = 68 / 15 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3124_312485


namespace NUMINAMATH_CALUDE_books_taken_out_monday_l3124_312458

/-- The number of books taken out on Monday from a library -/
def books_taken_out (initial_books : ℕ) (books_returned : ℕ) (final_books : ℕ) : ℕ :=
  initial_books + books_returned - final_books

/-- Theorem stating that 124 books were taken out on Monday -/
theorem books_taken_out_monday : books_taken_out 336 22 234 = 124 := by
  sorry

end NUMINAMATH_CALUDE_books_taken_out_monday_l3124_312458


namespace NUMINAMATH_CALUDE_EF_length_is_19_2_l3124_312498

/-- Two similar right triangles ABC and DEF with given side lengths -/
structure SimilarRightTriangles where
  -- Triangle ABC
  AB : ℝ
  BC : ℝ
  -- Triangle DEF
  DE : ℝ
  -- Similarity ratio
  k : ℝ
  -- Conditions
  AB_positive : AB > 0
  BC_positive : BC > 0
  DE_positive : DE > 0
  k_positive : k > 0
  similarity : k = DE / AB
  AB_value : AB = 10
  BC_value : BC = 8
  DE_value : DE = 24

/-- The length of EF in the similar right triangles -/
def EF_length (t : SimilarRightTriangles) : ℝ :=
  t.k * t.BC

/-- Theorem: The length of EF is 19.2 -/
theorem EF_length_is_19_2 (t : SimilarRightTriangles) : EF_length t = 19.2 := by
  sorry

#check EF_length_is_19_2

end NUMINAMATH_CALUDE_EF_length_is_19_2_l3124_312498


namespace NUMINAMATH_CALUDE_squares_in_4x2023_grid_l3124_312442

/-- The number of squares with vertices on grid points in a 4 x 2023 grid -/
def squaresInGrid (rows : ℕ) (cols : ℕ) : ℕ :=
  let type_a := rows * cols
  let type_b := (rows - 1) * (cols - 1)
  let type_c := (rows - 2) * (cols - 2)
  let type_d := (rows - 3) * (cols - 3)
  type_a + 2 * type_b + 3 * type_c + 4 * type_d

/-- Theorem stating that the number of squares in a 4 x 2023 grid is 40430 -/
theorem squares_in_4x2023_grid :
  squaresInGrid 4 2023 = 40430 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_4x2023_grid_l3124_312442


namespace NUMINAMATH_CALUDE_five_cubic_yards_to_cubic_inches_l3124_312449

/-- Converts cubic yards to cubic inches -/
def cubic_yards_to_cubic_inches (yards : ℕ) : ℕ :=
  let feet_per_yard : ℕ := 3
  let inches_per_foot : ℕ := 12
  yards * (feet_per_yard ^ 3) * (inches_per_foot ^ 3)

/-- Theorem stating that 5 cubic yards equals 233280 cubic inches -/
theorem five_cubic_yards_to_cubic_inches :
  cubic_yards_to_cubic_inches 5 = 233280 := by
  sorry

end NUMINAMATH_CALUDE_five_cubic_yards_to_cubic_inches_l3124_312449


namespace NUMINAMATH_CALUDE_house_wall_planks_l3124_312419

theorem house_wall_planks (total_planks small_planks : ℕ) 
  (h1 : total_planks = 29)
  (h2 : small_planks = 17) :
  total_planks - small_planks = 12 := by
  sorry

end NUMINAMATH_CALUDE_house_wall_planks_l3124_312419


namespace NUMINAMATH_CALUDE_simplify_expression_l3124_312486

theorem simplify_expression : (6 * 10^10) / (2 * 10^4) = 3000000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3124_312486


namespace NUMINAMATH_CALUDE_qingming_rain_is_random_l3124_312435

/-- An event that occurs during a specific season --/
structure SeasonalEvent where
  season : String
  description : String

/-- A property indicating whether an event can be predicted with certainty --/
def isPredictable (e : SeasonalEvent) : Prop := sorry

/-- A property indicating whether an event's occurrence varies from year to year --/
def hasVariableOccurrence (e : SeasonalEvent) : Prop := sorry

/-- Definition of a random event --/
def isRandomEvent (e : SeasonalEvent) : Prop := 
  ¬(isPredictable e) ∧ hasVariableOccurrence e

/-- The main theorem --/
theorem qingming_rain_is_random (e : SeasonalEvent) 
  (h1 : e.season = "Qingming")
  (h2 : e.description = "drizzling rain")
  (h3 : ¬(isPredictable e))
  (h4 : hasVariableOccurrence e) : 
  isRandomEvent e := by
  sorry

end NUMINAMATH_CALUDE_qingming_rain_is_random_l3124_312435
