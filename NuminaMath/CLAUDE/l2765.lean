import Mathlib

namespace NUMINAMATH_CALUDE_magnitude_e1_minus_sqrt3_e2_l2765_276533

-- Define the vector space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- State the theorem
theorem magnitude_e1_minus_sqrt3_e2 
  (h1 : ‖e₁‖ = 1) 
  (h2 : ‖e₂‖ = 1) 
  (h3 : inner e₁ e₂ = Real.sqrt 3 / 2) : 
  ‖e₁ - Real.sqrt 3 • e₂‖ = 1 := by sorry

end NUMINAMATH_CALUDE_magnitude_e1_minus_sqrt3_e2_l2765_276533


namespace NUMINAMATH_CALUDE_ellipse_max_sum_l2765_276564

/-- Given an ellipse defined by x^2/4 + y^2/2 = 1, 
    the maximum value of |x| + |y| is 2√3. -/
theorem ellipse_max_sum (x y : ℝ) : 
  x^2/4 + y^2/2 = 1 → |x| + |y| ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_max_sum_l2765_276564


namespace NUMINAMATH_CALUDE_mercury_column_height_for_constant_center_of_gravity_l2765_276512

/-- Proves that the height of the mercury column for which the center of gravity
    remains at a constant distance from the top of the tube at any temperature
    is approximately 0.106 meters. -/
theorem mercury_column_height_for_constant_center_of_gravity
  (tube_length : ℝ)
  (cross_section_area : ℝ)
  (glass_expansion_coeff : ℝ)
  (mercury_expansion_coeff : ℝ)
  (h : tube_length = 1)
  (h₁ : cross_section_area = 1e-4)
  (h₂ : glass_expansion_coeff = 1 / 38700)
  (h₃ : mercury_expansion_coeff = 1 / 5550) :
  ∃ (height : ℝ), abs (height - 0.106) < 0.001 ∧
  ∀ (t : ℝ),
    (tube_length * (1 + glass_expansion_coeff / 3 * t) -
     height / 2 * (1 + (mercury_expansion_coeff - 2 * glass_expansion_coeff / 3) * t)) =
    (tube_length - height / 2) :=
sorry

end NUMINAMATH_CALUDE_mercury_column_height_for_constant_center_of_gravity_l2765_276512


namespace NUMINAMATH_CALUDE_inequality_proof_l2765_276597

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt ((z + x) * (z + y)) - z ≥ Real.sqrt (x * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2765_276597


namespace NUMINAMATH_CALUDE_count_valid_numbers_is_1800_l2765_276521

/-- Define a 5-digit number -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- Define the quotient and remainder when n is divided by 50 -/
def quotient_remainder (n q r : ℕ) : Prop :=
  n = 50 * q + r ∧ r < 50

/-- Count of 5-digit numbers n where q + r is divisible by 9 -/
def count_valid_numbers : ℕ := sorry

/-- Theorem stating the count of valid numbers is 1800 -/
theorem count_valid_numbers_is_1800 :
  count_valid_numbers = 1800 := by sorry

end NUMINAMATH_CALUDE_count_valid_numbers_is_1800_l2765_276521


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2765_276566

/-- Given two trains running on parallel rails in the same direction, this theorem
    calculates the speed of the second train based on the given conditions. -/
theorem train_speed_calculation
  (length1 : ℝ) (length2 : ℝ) (speed1 : ℝ) (crossing_time : ℝ)
  (h1 : length1 = 200) -- Length of first train in meters
  (h2 : length2 = 180) -- Length of second train in meters
  (h3 : speed1 = 45) -- Speed of first train in km/h
  (h4 : crossing_time = 273.6) -- Time to cross in seconds
  : ∃ (speed2 : ℝ), speed2 = 40 ∧ 
    (speed1 - speed2) * (crossing_time / 3600) = (length1 + length2) / 1000 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l2765_276566


namespace NUMINAMATH_CALUDE_painted_unit_cubes_in_3x3x3_l2765_276577

/-- Represents a 3D cube -/
structure Cube :=
  (size : ℕ)

/-- Represents a painted cube -/
def PaintedCube := Cube

/-- Represents a unit cube (1x1x1) -/
def UnitCube := Cube

/-- The number of unit cubes with at least one painted surface in a painted cube -/
def num_painted_unit_cubes (c : PaintedCube) : ℕ :=
  sorry

/-- The main theorem: In a 3x3x3 painted cube, 26 unit cubes have at least one painted surface -/
theorem painted_unit_cubes_in_3x3x3 (c : PaintedCube) (h : c.size = 3) :
  num_painted_unit_cubes c = 26 :=
sorry

end NUMINAMATH_CALUDE_painted_unit_cubes_in_3x3x3_l2765_276577


namespace NUMINAMATH_CALUDE_better_fit_larger_R_squared_l2765_276515

/-- The correlation index in regression analysis -/
def correlation_index (model : Type*) : ℝ := sorry

/-- The fitting effect of a regression model -/
def fitting_effect (model : Type*) : ℝ := sorry

/-- Theorem stating that a larger correlation index implies a better fitting effect -/
theorem better_fit_larger_R_squared (model1 model2 : Type*) :
  correlation_index model1 > correlation_index model2 →
  fitting_effect model1 > fitting_effect model2 :=
by sorry

end NUMINAMATH_CALUDE_better_fit_larger_R_squared_l2765_276515


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2765_276573

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x + y) / (x - y) - (x - y) / (x + y) = 3) :
  (x^4 + y^4) / (x^4 - y^4) - (x^4 - y^4) / (x^4 + y^4) = 49 / 600 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2765_276573


namespace NUMINAMATH_CALUDE_stadium_seats_pattern_l2765_276531

/-- Represents the number of seats in a row of the stadium -/
def seats (n : ℕ) : ℕ := n + 49

/-- The theorem states that the number of seats in each row follows the given pattern -/
theorem stadium_seats_pattern (n : ℕ) (h : 1 ≤ n ∧ n ≤ 40) : 
  seats n = 50 + (n - 1) := by sorry

end NUMINAMATH_CALUDE_stadium_seats_pattern_l2765_276531


namespace NUMINAMATH_CALUDE_complex_equidistant_points_l2765_276569

theorem complex_equidistant_points : ∃ (z : ℂ), 
  Complex.abs (z - 2) = 3 ∧ 
  Complex.abs (z + 1 + 2*I) = 3 ∧ 
  Complex.abs (z - 3*I) = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equidistant_points_l2765_276569


namespace NUMINAMATH_CALUDE_max_candy_remainder_l2765_276540

theorem max_candy_remainder (x : ℕ) : 
  ∃ (q r : ℕ), x = 9 * q + r ∧ r < 9 ∧ r ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_remainder_l2765_276540


namespace NUMINAMATH_CALUDE_animal_video_ratio_l2765_276599

theorem animal_video_ratio :
  ∀ (total_time cat_time dog_time gorilla_time : ℝ),
    total_time = 36 →
    cat_time = 4 →
    gorilla_time = 2 * (cat_time + dog_time) →
    total_time = cat_time + dog_time + gorilla_time →
    dog_time / cat_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_animal_video_ratio_l2765_276599


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2765_276522

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, 3*x - |(-2)*x + 1| ≥ a ↔ x ∈ Set.Ici 2) → a = 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2765_276522


namespace NUMINAMATH_CALUDE_all_expressions_zero_l2765_276563

-- Define a 2D vector type
def Vector2D := ℝ × ℝ

-- Define vector addition
def add_vectors (v1 v2 : Vector2D) : Vector2D :=
  (v1.1 + v2.1, v1.2 + v2.2)

-- Define vector subtraction
def sub_vectors (v1 v2 : Vector2D) : Vector2D :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Define the zero vector
def zero_vector : Vector2D := (0, 0)

-- Define variables for each point
variable (A B C D E F O P Q : Vector2D)

-- Define the theorem
theorem all_expressions_zero : 
  (add_vectors (add_vectors (sub_vectors B A) (sub_vectors C B)) (sub_vectors A C) = zero_vector) ∧
  (add_vectors (sub_vectors (sub_vectors (sub_vectors B A) (sub_vectors C A)) (sub_vectors D B)) (sub_vectors D C) = zero_vector) ∧
  (sub_vectors (add_vectors (add_vectors (sub_vectors Q F) (sub_vectors P Q)) (sub_vectors F E)) (sub_vectors P E) = zero_vector) ∧
  (add_vectors (sub_vectors (sub_vectors A O) (sub_vectors B O)) (sub_vectors B A) = zero_vector) := by
  sorry

end NUMINAMATH_CALUDE_all_expressions_zero_l2765_276563


namespace NUMINAMATH_CALUDE_doubling_points_properties_l2765_276568

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be a "doubling point" of another
def isDoublingPoint (p q : Point) : Prop :=
  2 * (p.x + q.x) = p.y + q.y

-- Given point P₁
def P₁ : Point := ⟨2, 0⟩

-- Theorem statement
theorem doubling_points_properties :
  -- 1. Q₁ and Q₂ are doubling points of P₁
  (isDoublingPoint P₁ ⟨2, 8⟩ ∧ isDoublingPoint P₁ ⟨-3, -2⟩) ∧
  -- 2. A(-2, 0) on y = x + 2 is a doubling point of P₁
  (∃ A : Point, A.y = A.x + 2 ∧ A.x = -2 ∧ A.y = 0 ∧ isDoublingPoint P₁ A) ∧
  -- 3. Two points on y = x² - 2x - 3 are doubling points of P₁
  (∃ B C : Point, B ≠ C ∧
    B.y = B.x^2 - 2*B.x - 3 ∧ C.y = C.x^2 - 2*C.x - 3 ∧
    isDoublingPoint P₁ B ∧ isDoublingPoint P₁ C) ∧
  -- 4. Minimum distance to any doubling point is 8√5/5
  (∃ minDist : ℝ, minDist = 8 * Real.sqrt 5 / 5 ∧
    ∀ Q : Point, isDoublingPoint P₁ Q →
      Real.sqrt ((Q.x - P₁.x)^2 + (Q.y - P₁.y)^2) ≥ minDist) :=
by sorry

end NUMINAMATH_CALUDE_doubling_points_properties_l2765_276568


namespace NUMINAMATH_CALUDE_symmetry_about_xOy_plane_l2765_276589

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOy plane in 3D space -/
def xOyPlane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry about the xOy plane -/
def symmetricAboutXOy (p q : Point3D) : Prop :=
  p.x = q.x ∧ p.y = q.y ∧ p.z = -q.z

theorem symmetry_about_xOy_plane :
  let p := Point3D.mk 1 3 (-5)
  let q := Point3D.mk 1 3 5
  symmetricAboutXOy p q :=
by
  sorry

#check symmetry_about_xOy_plane

end NUMINAMATH_CALUDE_symmetry_about_xOy_plane_l2765_276589


namespace NUMINAMATH_CALUDE_good_number_theorem_l2765_276549

/-- A good number is a number of the form a + b√2 where a and b are integers -/
def GoodNumber (x : ℝ) : Prop :=
  ∃ (a b : ℤ), x = a + b * Real.sqrt 2

/-- Polynomial with good number coefficients -/
def GoodPolynomial (p : Polynomial ℝ) : Prop :=
  ∀ (i : ℕ), GoodNumber (p.coeff i)

theorem good_number_theorem (A B Q : Polynomial ℝ) 
  (hA : GoodPolynomial A)
  (hB : GoodPolynomial B)
  (hB0 : B.coeff 0 = 1)
  (hABQ : A = B * Q) :
  GoodPolynomial Q :=
sorry

end NUMINAMATH_CALUDE_good_number_theorem_l2765_276549


namespace NUMINAMATH_CALUDE_meg_cat_weight_l2765_276543

theorem meg_cat_weight (meg_weight anne_weight : ℝ) 
  (h1 : meg_weight / anne_weight = 5 / 7)
  (h2 : anne_weight = meg_weight + 8) : 
  meg_weight = 20 := by
sorry

end NUMINAMATH_CALUDE_meg_cat_weight_l2765_276543


namespace NUMINAMATH_CALUDE_hundred_digit_number_theorem_l2765_276535

def is_valid_number (N : ℕ) : Prop :=
  ∃ (b : ℕ), b ∈ ({1, 2, 3} : Set ℕ) ∧ N = 325 * b * (10 ^ 97)

theorem hundred_digit_number_theorem (N : ℕ) :
  (∃ (k : ℕ) (a : ℕ), 
    a ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    N ≥ 10^99 ∧ N < 10^100 ∧
    ∃ (N' : ℕ), (N' = N - a * 10^k ∨ (k = 99 ∧ N' = N - a * 10^99)) ∧ N = 13 * N') →
  is_valid_number N :=
sorry

end NUMINAMATH_CALUDE_hundred_digit_number_theorem_l2765_276535


namespace NUMINAMATH_CALUDE_actual_car_mass_l2765_276537

/-- The mass of a scaled object given the original mass and scale factor. -/
def scaled_mass (original_mass : ℝ) (scale_factor : ℝ) : ℝ :=
  original_mass * scale_factor^3

/-- The mass of the actual car body is 1024 kg. -/
theorem actual_car_mass (model_mass : ℝ) (scale_factor : ℝ) 
  (h1 : model_mass = 2)
  (h2 : scale_factor = 8) :
  scaled_mass model_mass scale_factor = 1024 := by
  sorry

end NUMINAMATH_CALUDE_actual_car_mass_l2765_276537


namespace NUMINAMATH_CALUDE_sandy_shopping_money_l2765_276574

theorem sandy_shopping_money (total : ℝ) (spent_percentage : ℝ) (left : ℝ) : 
  total = 320 →
  spent_percentage = 30 →
  left = total * (1 - spent_percentage / 100) →
  left = 224 :=
by sorry

end NUMINAMATH_CALUDE_sandy_shopping_money_l2765_276574


namespace NUMINAMATH_CALUDE_g_increasing_intervals_l2765_276560

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

def g (x : ℝ) : ℝ := f (2 - x^2)

theorem g_increasing_intervals :
  ∃ (a b c : ℝ), a = -1 ∧ b = 0 ∧ c = 1 ∧
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → g x ≤ g y) ∧
  (∀ x y, c ≤ x ∧ x < y → g x < g y) :=
sorry

end NUMINAMATH_CALUDE_g_increasing_intervals_l2765_276560


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2765_276591

theorem inverse_variation_problem (k : ℝ) (h : k > 0) :
  (∀ x y, x > 0 → y * Real.sqrt x = k) →
  (1/2 * Real.sqrt (1/4) = k) →
  (∃ x, x > 0 ∧ 8 * Real.sqrt x = k ∧ x = 1/1024) :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2765_276591


namespace NUMINAMATH_CALUDE_angle_sum_equality_l2765_276509

-- Define the points in 2D space
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (1, 0)
def D : ℝ × ℝ := (2, 0)
def E : ℝ × ℝ := (3, 0)
def F : ℝ × ℝ := (3, 1)

-- Define the angles
def angle_FBE : ℝ := sorry
def angle_FCE : ℝ := sorry
def angle_FDE : ℝ := sorry

-- Theorem statement
theorem angle_sum_equality : angle_FBE + angle_FCE = angle_FDE := by sorry

end NUMINAMATH_CALUDE_angle_sum_equality_l2765_276509


namespace NUMINAMATH_CALUDE_max_value_f_in_interval_l2765_276528

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- State the theorem
theorem max_value_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1) 1 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-1) 1 → f x ≤ f c ∧ f c = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_f_in_interval_l2765_276528


namespace NUMINAMATH_CALUDE_investment_sum_l2765_276503

/-- Given a sum invested at simple interest for two years, 
    if the difference in interest between 15% p.a. and 12% p.a. is 420, 
    then the sum invested is 7000. -/
theorem investment_sum (P : ℝ) : 
  (P * 0.15 * 2 - P * 0.12 * 2 = 420) → P = 7000 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l2765_276503


namespace NUMINAMATH_CALUDE_square_39_equals_square_40_minus_79_l2765_276590

theorem square_39_equals_square_40_minus_79 : 39^2 = 40^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_square_39_equals_square_40_minus_79_l2765_276590


namespace NUMINAMATH_CALUDE_stock_decrease_duration_l2765_276578

/-- Represents the monthly decrease in bicycle stock -/
def monthly_decrease : ℕ := 4

/-- Represents the total decrease in bicycle stock from January 1 to October 1 -/
def total_decrease : ℕ := 36

/-- Represents the number of months from January 1 to October 1 -/
def total_months : ℕ := 9

/-- Represents the number of months the stock has been decreasing -/
def months_decreasing : ℕ := 5

theorem stock_decrease_duration :
  months_decreasing * monthly_decrease = total_decrease - (total_months - months_decreasing) * monthly_decrease :=
by sorry

end NUMINAMATH_CALUDE_stock_decrease_duration_l2765_276578


namespace NUMINAMATH_CALUDE_max_expression_value_l2765_276580

def is_valid_assignment (O L I M P A D : ℕ) : Prop :=
  O ≠ L ∧ O ≠ I ∧ O ≠ M ∧ O ≠ P ∧ O ≠ A ∧ O ≠ D ∧
  L ≠ I ∧ L ≠ M ∧ L ≠ P ∧ L ≠ A ∧ L ≠ D ∧
  I ≠ M ∧ I ≠ P ∧ I ≠ A ∧ I ≠ D ∧
  M ≠ P ∧ M ≠ A ∧ M ≠ D ∧
  P ≠ A ∧ P ≠ D ∧
  A ≠ D ∧
  O < 10 ∧ L < 10 ∧ I < 10 ∧ M < 10 ∧ P < 10 ∧ A < 10 ∧ D < 10 ∧
  O ≠ 0 ∧ I ≠ 0

def expression_value (O L I M P A D : ℕ) : ℤ :=
  (10 * O + L) + (10 * I + M) - P + (10 * I + A) - (10 * D + A)

theorem max_expression_value :
  ∀ O L I M P A D : ℕ,
    is_valid_assignment O L I M P A D →
    expression_value O L I M P A D ≤ 263 :=
sorry

end NUMINAMATH_CALUDE_max_expression_value_l2765_276580


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_equivalence_l2765_276510

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem binomial_coefficient_divisibility_equivalence 
  (p : ℕ) (n : ℕ) 
  (h1 : is_prime p) 
  (h2 : is_prime (11 * 39 * p)) : 
  (∃ k : ℕ, k ≤ n ∧ p ∣ Nat.choose n k) ↔ 
  (∃ s q : ℕ, n = p^s * q - 1 ∧ s ≥ 0 ∧ 0 < q ∧ q < p) :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_equivalence_l2765_276510


namespace NUMINAMATH_CALUDE_conditional_statements_requirement_l2765_276554

-- Define a type for the problems
inductive Problem
| AbsoluteValue
| CubeVolume
| PiecewiseFunction

-- Define a function to check if a problem requires conditional statements
def requiresConditionalStatements (p : Problem) : Prop :=
  match p with
  | Problem.AbsoluteValue => true
  | Problem.CubeVolume => false
  | Problem.PiecewiseFunction => true

-- Theorem statement
theorem conditional_statements_requirement :
  (requiresConditionalStatements Problem.AbsoluteValue ∧
   requiresConditionalStatements Problem.PiecewiseFunction) ∧
  ¬requiresConditionalStatements Problem.CubeVolume := by
  sorry


end NUMINAMATH_CALUDE_conditional_statements_requirement_l2765_276554


namespace NUMINAMATH_CALUDE_train_length_l2765_276562

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 52 → time = 9 → ∃ length : ℝ, abs (length - 129.96) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2765_276562


namespace NUMINAMATH_CALUDE_first_term_is_five_halves_l2765_276529

/-- Sum of first n terms of an arithmetic sequence -/
def T (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 5) / 2

/-- The ratio of T(3n) to T(n) is constant for all positive n -/
def ratio_is_constant (a : ℚ) : Prop :=
  ∃ k : ℚ, ∀ n : ℕ, n > 0 → T a (3*n) / T a n = k

theorem first_term_is_five_halves :
  ∀ a : ℚ, ratio_is_constant a → a = 5/2 := by sorry

end NUMINAMATH_CALUDE_first_term_is_five_halves_l2765_276529


namespace NUMINAMATH_CALUDE_elsa_marbles_l2765_276500

/-- The number of marbles Elsa started with -/
def initial_marbles : ℕ := sorry

/-- The number of marbles Elsa lost at breakfast -/
def lost_at_breakfast : ℕ := 3

/-- The number of marbles Elsa gave to Susie at lunch -/
def given_to_susie : ℕ := 5

/-- The number of new marbles Elsa's mom bought -/
def new_marbles : ℕ := 12

/-- The number of marbles Elsa had at the end of the day -/
def final_marbles : ℕ := 54

theorem elsa_marbles :
  initial_marbles = 40 :=
by
  have h1 : initial_marbles - lost_at_breakfast - given_to_susie + new_marbles + 2 * given_to_susie = final_marbles :=
    sorry
  sorry

end NUMINAMATH_CALUDE_elsa_marbles_l2765_276500


namespace NUMINAMATH_CALUDE_factor_tree_X_value_l2765_276586

theorem factor_tree_X_value :
  ∀ (X Y Z F G : ℕ),
    X = Y * Z →
    Y = 7 * F →
    Z = 11 * G →
    F = 7 * 3 →
    G = 11 * 3 →
    X = 53361 := by
  sorry

end NUMINAMATH_CALUDE_factor_tree_X_value_l2765_276586


namespace NUMINAMATH_CALUDE_hillarys_money_after_deposit_l2765_276502

/-- The amount of money Hillary is left with after selling crafts and making a deposit -/
def hillarys_remaining_money (craft_price : ℕ) (crafts_sold : ℕ) (extra_money : ℕ) (deposit : ℕ) : ℕ :=
  craft_price * crafts_sold + extra_money - deposit

/-- Theorem stating that Hillary is left with 25 dollars after selling crafts and making a deposit -/
theorem hillarys_money_after_deposit :
  hillarys_remaining_money 12 3 7 18 = 25 := by
  sorry

end NUMINAMATH_CALUDE_hillarys_money_after_deposit_l2765_276502


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l2765_276519

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l2765_276519


namespace NUMINAMATH_CALUDE_parallelepiped_with_rectangular_opposite_faces_is_right_l2765_276534

/-- A parallelepiped is a three-dimensional figure with six faces, 
    where each pair of opposite faces are parallel parallelograms. -/
structure Parallelepiped

/-- A right parallelepiped is a parallelepiped where the lateral edges 
    are perpendicular to the base. -/
structure RightParallelepiped extends Parallelepiped

/-- A face of a parallelepiped -/
structure Face (P : Parallelepiped)

/-- Predicate to check if a face is rectangular -/
def is_rectangular (F : Face P) : Prop := sorry

/-- Predicate to check if two faces are opposite -/
def are_opposite (F1 F2 : Face P) : Prop := sorry

theorem parallelepiped_with_rectangular_opposite_faces_is_right 
  (P : Parallelepiped) 
  (F1 F2 : Face P) 
  (h1 : is_rectangular F1) 
  (h2 : is_rectangular F2) 
  (h3 : are_opposite F1 F2) : 
  RightParallelepiped := sorry

end NUMINAMATH_CALUDE_parallelepiped_with_rectangular_opposite_faces_is_right_l2765_276534


namespace NUMINAMATH_CALUDE_a_ratio_l2765_276541

def a (n : ℕ) : ℚ := 3 - 2^n

theorem a_ratio : a 2 / a 3 = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_a_ratio_l2765_276541


namespace NUMINAMATH_CALUDE_limit_rational_function_l2765_276526

/-- The limit of (2x^2 - x - 1) / (x^3 + 2x^2 - x - 2) as x approaches 1 is 1/2 -/
theorem limit_rational_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 1| ∧ |x - 1| < δ → 
    |(2*x^2 - x - 1) / (x^3 + 2*x^2 - x - 2) - 1/2| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_rational_function_l2765_276526


namespace NUMINAMATH_CALUDE_max_chord_line_l2765_276505

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The circle C: x^2 + y^2 + 4x + 3 = 0 -/
def C : Circle := { center := (-2, 0), radius := 1 }

/-- The point through which line l passes -/
def P : ℝ × ℝ := (2, 3)

/-- Function to check if a line passes through a point -/
def passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Function to check if a line intersects a circle at two points -/
def intersects_circle (l : Line) (c : Circle) : Prop :=
  ∃ (p q : ℝ × ℝ), p ≠ q ∧ 
    passes_through l p ∧ passes_through l q ∧
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
    (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2

/-- Function to check if the chord formed by the intersection of a line and circle is maximized -/
def maximizes_chord (l : Line) (c : Circle) : Prop :=
  passes_through l c.center

/-- The theorem to be proved -/
theorem max_chord_line : 
  ∃ (l : Line), 
    passes_through l P ∧ 
    intersects_circle l C ∧ 
    maximizes_chord l C ∧ 
    l = { a := 3, b := -4, c := 6 } := by sorry

end NUMINAMATH_CALUDE_max_chord_line_l2765_276505


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2765_276557

theorem inequality_solution_set (m n : ℝ) (h : m > n) :
  {x : ℝ | (n - m) * x > 0} = {x : ℝ | x < 0} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2765_276557


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2765_276552

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℤ, 
    3 * X^4 + 8 * X^3 - 29 * X^2 - 17 * X + 34 = 
    (X^2 + 5 * X - 3) * q + (79 * X - 11) ∧ 
    (79 * X - 11).degree < (X^2 + 5 * X - 3).degree :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2765_276552


namespace NUMINAMATH_CALUDE_bernardo_silvia_game_l2765_276517

theorem bernardo_silvia_game (M : ℕ) : 
  (M ≤ 1999) →
  (32 * M + 1600 < 2000) →
  (32 * M + 1700 ≥ 2000) →
  (∀ N : ℕ, N < M → (32 * N + 1600 < 2000 → 32 * N + 1700 < 2000)) →
  (M = 10 ∧ (M / 10 + M % 10 = 1)) := by
  sorry

end NUMINAMATH_CALUDE_bernardo_silvia_game_l2765_276517


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2765_276565

theorem arithmetic_sequence_sum : 
  ∀ (a l d n : ℤ),
    a = -41 →
    l = 1 →
    d = 2 →
    n = 22 →
    a + (n - 1) * d = l →
    (n * (a + l)) / 2 = -440 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2765_276565


namespace NUMINAMATH_CALUDE_intersection_implies_b_range_l2765_276542

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the line equation
def line_equation (x y b : ℝ) : Prop := y = -2 * x + b

-- Define the condition for line intersection with segment AB
def intersects_AB (b : ℝ) : Prop :=
  ∃ (x y : ℝ), line_equation x y b ∧ 
  ((x ≥ A.1 ∧ x ≤ B.1) ∨ (x ≤ A.1 ∧ x ≥ B.1)) ∧
  ((y ≥ A.2 ∧ y ≤ B.2) ∨ (y ≤ A.2 ∧ y ≥ B.2))

-- Theorem statement
theorem intersection_implies_b_range :
  ∀ b : ℝ, intersects_AB b → b ∈ Set.Icc (-2 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_implies_b_range_l2765_276542


namespace NUMINAMATH_CALUDE_couple_stock_purchase_l2765_276550

/-- Calculates the number of shares a couple can buy given their savings plan and stock price --/
def shares_to_buy (wife_weekly_savings : ℕ) (husband_monthly_savings : ℕ) (months : ℕ) (stock_price : ℕ) : ℕ :=
  let wife_monthly_savings := wife_weekly_savings * 4
  let total_monthly_savings := wife_monthly_savings + husband_monthly_savings
  let total_savings := total_monthly_savings * months
  let investment := total_savings / 2
  investment / stock_price

/-- Theorem stating that the couple can buy 25 shares given their specific savings plan --/
theorem couple_stock_purchase :
  shares_to_buy 100 225 4 50 = 25 := by
  sorry

end NUMINAMATH_CALUDE_couple_stock_purchase_l2765_276550


namespace NUMINAMATH_CALUDE_problem_solution_l2765_276539

theorem problem_solution :
  ∀ (a b c : ℕ),
  ({a, b, c} : Set ℕ) = {0, 1, 2} →
  (((a ≠ 2) ∧ (b ≠ 2) ∧ (c = 0)) ∨
   ((a = 2) ∧ (b ≠ 2) ∧ (c ≠ 0)) ∨
   ((a ≠ 2) ∧ (b = 2) ∧ (c ≠ 0))) →
  10 * a + 2 * b + c = 21 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2765_276539


namespace NUMINAMATH_CALUDE_cuboid_ratio_simplification_l2765_276594

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℕ
  breadth : ℕ
  height : ℕ

/-- Calculates the greatest common divisor of three natural numbers -/
def gcd3 (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

/-- Simplifies the ratio of three numbers by dividing each by their GCD -/
def simplifyRatio (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
  let d := gcd3 a b c
  (a / d, b / d, c / d)

/-- The main theorem stating that the given cuboid dimensions simplify to the ratio 6:5:4 -/
theorem cuboid_ratio_simplification (c : CuboidDimensions) 
  (h1 : c.length = 90) 
  (h2 : c.breadth = 75) 
  (h3 : c.height = 60) : 
  simplifyRatio c.length c.breadth c.height = (6, 5, 4) := by
  sorry

end NUMINAMATH_CALUDE_cuboid_ratio_simplification_l2765_276594


namespace NUMINAMATH_CALUDE_monkey_multiplication_l2765_276507

/-- The number of spirit monkeys created from one hair -/
def spiritsPerHair : ℕ := 3

/-- The number of new spirit monkeys created by each existing spirit monkey per second -/
def splitRate : ℕ := 3

/-- The number of hairs the Monkey King pulls out -/
def numHairs : ℕ := 10

/-- The number of seconds that pass -/
def timeElapsed : ℕ := 5

/-- The total number of monkeys after the given time -/
def totalMonkeys : ℕ := numHairs * spiritsPerHair * splitRate ^ timeElapsed + 1

theorem monkey_multiplication (spiritsPerHair splitRate numHairs timeElapsed : ℕ) :
  totalMonkeys = 7290 :=
sorry

end NUMINAMATH_CALUDE_monkey_multiplication_l2765_276507


namespace NUMINAMATH_CALUDE_expression_simplification_l2765_276579

theorem expression_simplification 
  (b c d x y : ℝ) (h : cx + dy ≠ 0) :
  (c * x * (c^2 * x^2 + 3 * b^2 * y^2 + c^2 * y^2) + 
   d * y * (b^2 * x^2 + 3 * c^2 * x^2 + b^2 * y^2)) / 
  (c * x + d * y) = 
  c^2 * x^2 + d * b^2 * y^2 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l2765_276579


namespace NUMINAMATH_CALUDE_difference_of_squares_l2765_276583

theorem difference_of_squares (x y : ℝ) 
  (sum_eq : x + y = 20) 
  (diff_eq : x - y = 8) : 
  x^2 - y^2 = 160 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2765_276583


namespace NUMINAMATH_CALUDE_comic_books_average_l2765_276570

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

theorem comic_books_average (a₁ : ℕ) (d : ℕ) (n : ℕ) 
  (h₁ : a₁ = 10) (h₂ : d = 6) (h₃ : n = 8) : 
  (arithmetic_sequence a₁ d n).sum / n = 31 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_average_l2765_276570


namespace NUMINAMATH_CALUDE_horner_method_v4_l2765_276532

def horner_polynomial (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_v4 (x : ℝ) : ℝ :=
  let v0 := 1
  let v1 := x * v0 - 12
  let v2 := x * v1 + 60
  let v3 := x * v2 - 160
  x * v3 + 240

theorem horner_method_v4 :
  horner_v4 2 = 80 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v4_l2765_276532


namespace NUMINAMATH_CALUDE_average_of_9_15_N_l2765_276547

theorem average_of_9_15_N (N : ℝ) (h : 12 < N ∧ N < 22) :
  let avg := (9 + 15 + N) / 3
  avg = 12 ∨ avg = 15 := by
sorry

end NUMINAMATH_CALUDE_average_of_9_15_N_l2765_276547


namespace NUMINAMATH_CALUDE_digit_1997_of_1_22_digit_1997_of_1_27_l2765_276530

/-- The nth decimal digit of a rational number -/
def nthDecimalDigit (q : ℚ) (n : ℕ) : ℕ := sorry

/-- The 1997th decimal digit of 1/22 is 0 -/
theorem digit_1997_of_1_22 : nthDecimalDigit (1/22) 1997 = 0 := by sorry

/-- The 1997th decimal digit of 1/27 is 3 -/
theorem digit_1997_of_1_27 : nthDecimalDigit (1/27) 1997 = 3 := by sorry

end NUMINAMATH_CALUDE_digit_1997_of_1_22_digit_1997_of_1_27_l2765_276530


namespace NUMINAMATH_CALUDE_max_guaranteed_winning_score_l2765_276595

/-- Represents a 9x9 grid game board -/
def GameBoard := Fin 9 → Fin 9 → Bool

/-- Counts the number of rows and columns where crosses outnumber noughts -/
def countCrossDominance (board : GameBoard) : ℕ :=
  sorry

/-- Counts the number of rows and columns where noughts outnumber crosses -/
def countNoughtDominance (board : GameBoard) : ℕ :=
  sorry

/-- Calculates the winning score for the first player -/
def winningScore (board : GameBoard) : ℤ :=
  (countCrossDominance board : ℤ) - (countNoughtDominance board : ℤ)

/-- Represents a strategy for playing the game -/
def Strategy := GameBoard → Fin 9 × Fin 9

/-- The theorem stating that the maximum guaranteed winning score is 2 -/
theorem max_guaranteed_winning_score :
  ∃ (strategyFirst : Strategy),
    ∀ (strategySecond : Strategy),
      ∃ (finalBoard : GameBoard),
        (winningScore finalBoard ≥ 2) ∧
        ∀ (otherFinalBoard : GameBoard),
          winningScore otherFinalBoard ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_guaranteed_winning_score_l2765_276595


namespace NUMINAMATH_CALUDE_sin_sum_product_l2765_276596

theorem sin_sum_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_product_l2765_276596


namespace NUMINAMATH_CALUDE_apartment_occupancy_l2765_276588

theorem apartment_occupancy (stories : ℕ) (apartments_per_floor : ℕ) (total_people : ℕ) 
  (h1 : stories = 25)
  (h2 : apartments_per_floor = 4)
  (h3 : total_people = 200) :
  total_people / (stories * apartments_per_floor) = 2 := by
sorry

end NUMINAMATH_CALUDE_apartment_occupancy_l2765_276588


namespace NUMINAMATH_CALUDE_malcolm_brushing_time_l2765_276567

/-- The number of days Malcolm brushes his teeth -/
def days : ℕ := 30

/-- The number of times Malcolm brushes his teeth per day -/
def brushings_per_day : ℕ := 3

/-- The total time Malcolm spends brushing his teeth in hours -/
def total_hours : ℕ := 3

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem malcolm_brushing_time :
  (total_hours * minutes_per_hour) / (days * brushings_per_day) = 2 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_brushing_time_l2765_276567


namespace NUMINAMATH_CALUDE_black_white_area_ratio_l2765_276506

/-- The ratio of black to white areas in concentric circles -/
theorem black_white_area_ratio :
  let r₁ : ℝ := 2
  let r₂ : ℝ := 4
  let r₃ : ℝ := 6
  let r₄ : ℝ := 8
  let black_area := (r₂^2 - r₁^2) * Real.pi + (r₄^2 - r₃^2) * Real.pi
  let white_area := r₁^2 * Real.pi + (r₃^2 - r₂^2) * Real.pi
  black_area / white_area = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_black_white_area_ratio_l2765_276506


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2765_276558

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 690 →
  divisor = 36 →
  quotient = 19 →
  dividend = divisor * quotient + remainder →
  remainder = 6 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2765_276558


namespace NUMINAMATH_CALUDE_b_complete_time_l2765_276581

/-- The time it takes for A to complete the work alone -/
def a_time : ℚ := 14 / 3

/-- The time A and B work together -/
def together_time : ℚ := 1

/-- The time B works alone after A leaves -/
def b_remaining_time : ℚ := 41 / 14

/-- The time it takes for B to complete the work alone -/
def b_time : ℚ := 5

theorem b_complete_time : 
  (1 / a_time + 1 / b_time) * together_time + 
  (1 / b_time) * b_remaining_time = 1 := by sorry

end NUMINAMATH_CALUDE_b_complete_time_l2765_276581


namespace NUMINAMATH_CALUDE_solution_x_equals_two_l2765_276548

theorem solution_x_equals_two : 
  let x : ℝ := 2
  3 * x - 6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_solution_x_equals_two_l2765_276548


namespace NUMINAMATH_CALUDE_n_sided_polygon_exterior_angle_l2765_276582

theorem n_sided_polygon_exterior_angle (n : ℕ) : 
  (n ≠ 0) → (40 * n = 360) → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_n_sided_polygon_exterior_angle_l2765_276582


namespace NUMINAMATH_CALUDE_space_division_cube_tetrahedron_l2765_276576

/-- The number of parts into which the space is divided by the facets of a polyhedron -/
def num_parts (V F E : ℕ) : ℕ := 1 + V + F + E

/-- Properties of a cube -/
def cube_vertices : ℕ := 8
def cube_edges : ℕ := 12
def cube_faces : ℕ := 6

/-- Properties of a tetrahedron -/
def tetrahedron_vertices : ℕ := 4
def tetrahedron_edges : ℕ := 6
def tetrahedron_faces : ℕ := 4

theorem space_division_cube_tetrahedron :
  (num_parts cube_vertices cube_faces cube_edges = 27) ∧
  (num_parts tetrahedron_vertices tetrahedron_faces tetrahedron_edges = 15) :=
by sorry

end NUMINAMATH_CALUDE_space_division_cube_tetrahedron_l2765_276576


namespace NUMINAMATH_CALUDE_difference_of_squares_division_l2765_276593

theorem difference_of_squares_division : (315^2 - 291^2) / 24 = 606 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_l2765_276593


namespace NUMINAMATH_CALUDE_parallel_line_properties_l2765_276561

/-- A line parallel to y = -2x passing through (1, 2) has b = 4 and intersects x-axis at (2, 0) -/
theorem parallel_line_properties (k b : ℝ) :
  (k = -2) →  -- Parallel to y = -2x
  (2 = k * 1 + b) →  -- Passes through (1, 2)
  (b = 4 ∧ ∃ x : ℝ, k * x + b = 0 ∧ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_properties_l2765_276561


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l2765_276501

theorem sum_of_solutions_is_zero (y : ℝ) (h1 : y = 8) (h2 : ∃ x : ℝ, x^2 + y^2 = 225) :
  ∃ x₁ x₂ : ℝ, x₁^2 + y^2 = 225 ∧ x₂^2 + y^2 = 225 ∧ x₁ + x₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l2765_276501


namespace NUMINAMATH_CALUDE_employee_age_problem_l2765_276538

theorem employee_age_problem (total_employees : Nat) 
  (group1_count : Nat) (group1_avg_age : Nat)
  (group2_count : Nat) (group2_avg_age : Nat)
  (group3_count : Nat) (group3_avg_age : Nat)
  (avg_age_29 : Nat) :
  total_employees = 30 →
  group1_count = 10 →
  group1_avg_age = 24 →
  group2_count = 12 →
  group2_avg_age = 30 →
  group3_count = 7 →
  group3_avg_age = 35 →
  avg_age_29 = 29 →
  ∃ (age_30th : Nat), age_30th = 25 := by
sorry


end NUMINAMATH_CALUDE_employee_age_problem_l2765_276538


namespace NUMINAMATH_CALUDE_max_value_sin_cos_max_value_achievable_l2765_276524

theorem max_value_sin_cos (θ : ℝ) : 
  (1/2) * Real.sin (3 * θ)^2 - (1/2) * Real.cos (2 * θ) ≤ 1 :=
sorry

theorem max_value_achievable : 
  ∃ θ : ℝ, (1/2) * Real.sin (3 * θ)^2 - (1/2) * Real.cos (2 * θ) = 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_max_value_achievable_l2765_276524


namespace NUMINAMATH_CALUDE_students_in_line_l2765_276518

/-- The number of students in a line, given specific positions of two students and the number of students between them. -/
theorem students_in_line
  (yoojung_position : ℕ)  -- Position of Yoojung
  (eunjung_position : ℕ)  -- Position of Eunjung from the back
  (students_between : ℕ)  -- Number of students between Yoojung and Eunjung
  (h1 : yoojung_position = 1)  -- Yoojung is at the front
  (h2 : eunjung_position = 5)  -- Eunjung is 5th from the back
  (h3 : students_between = 30)  -- 30 students between Yoojung and Eunjung
  : ℕ :=
by
  sorry

#check students_in_line

end NUMINAMATH_CALUDE_students_in_line_l2765_276518


namespace NUMINAMATH_CALUDE_three_point_five_million_scientific_notation_l2765_276585

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a|
  h2 : |a| < 10

/-- Definition of 3.5 million -/
def three_point_five_million : ℝ := 3.5e6

/-- Theorem stating that 3.5 million can be expressed as 3.5 × 10^6 in scientific notation -/
theorem three_point_five_million_scientific_notation :
  ∃ (sn : ScientificNotation), three_point_five_million = sn.a * (10 : ℝ) ^ sn.n :=
sorry

end NUMINAMATH_CALUDE_three_point_five_million_scientific_notation_l2765_276585


namespace NUMINAMATH_CALUDE_basket_problem_l2765_276575

theorem basket_problem (total : ℕ) (apples : ℕ) (oranges : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : apples = 10)
  (h3 : oranges = 8)
  (h4 : both = 5) :
  total - (apples + oranges - both) = 2 := by
  sorry

end NUMINAMATH_CALUDE_basket_problem_l2765_276575


namespace NUMINAMATH_CALUDE_building_population_l2765_276559

/-- Calculates the total number of people housed in a building -/
def total_people (stories : ℕ) (apartments_per_floor : ℕ) (people_per_apartment : ℕ) : ℕ :=
  stories * apartments_per_floor * people_per_apartment

/-- Theorem: A 25-story building with 4 apartments per floor and 2 people per apartment houses 200 people -/
theorem building_population : total_people 25 4 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_building_population_l2765_276559


namespace NUMINAMATH_CALUDE_zlatoust_miass_distance_l2765_276527

/-- The distance between Zlatoust and Miass -/
def distance : ℝ := sorry

/-- The speed of GAZ truck -/
def speed_gaz : ℝ := sorry

/-- The speed of MAZ truck -/
def speed_maz : ℝ := sorry

/-- The speed of KAMAZ truck -/
def speed_kamaz : ℝ := sorry

theorem zlatoust_miass_distance :
  (distance + 18) / speed_kamaz = (distance - 18) / speed_maz ∧
  (distance + 25) / speed_kamaz = (distance - 25) / speed_gaz ∧
  (distance + 8) / speed_maz = (distance - 8) / speed_gaz →
  distance = 60 := by sorry

end NUMINAMATH_CALUDE_zlatoust_miass_distance_l2765_276527


namespace NUMINAMATH_CALUDE_max_value_expression_l2765_276511

theorem max_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 2*x*y + y^2 = 9) :
  x^2 - 2*x*y + y^2 ≤ 9/4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + 2*a*b + b^2 = 9 ∧ a^2 - 2*a*b + b^2 = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2765_276511


namespace NUMINAMATH_CALUDE_stratified_sampling_total_employees_l2765_276546

/-- Given a stratified sampling of employees from four companies, 
    prove the total number of employees across all companies. -/
theorem stratified_sampling_total_employees 
  (total_A : ℕ) 
  (selected_A selected_B selected_C selected_D : ℕ) 
  (h1 : total_A = 96)
  (h2 : selected_A = 12)
  (h3 : selected_B = 21)
  (h4 : selected_C = 25)
  (h5 : selected_D = 43) :
  (total_A * (selected_A + selected_B + selected_C + selected_D)) / selected_A = 808 := by
  sorry

#check stratified_sampling_total_employees

end NUMINAMATH_CALUDE_stratified_sampling_total_employees_l2765_276546


namespace NUMINAMATH_CALUDE_track_width_l2765_276572

/-- Given two concentric circles where the outer circle has a circumference of 40π feet
    and the difference between the outer and inner circle circumferences is 16π feet,
    prove that the difference between their radii is 8 feet. -/
theorem track_width (r₁ r₂ : ℝ) : 
  (2 * π * r₁ = 40 * π) →  -- Outer circle circumference
  (2 * π * r₁ - 2 * π * r₂ = 16 * π) →  -- Difference in circumferences
  r₁ - r₂ = 8 := by sorry

end NUMINAMATH_CALUDE_track_width_l2765_276572


namespace NUMINAMATH_CALUDE_large_monkey_doll_cost_l2765_276513

/-- The cost of a large monkey doll in dollars -/
def large_monkey_cost : ℝ := 6

/-- The total amount spent in dollars -/
def total_spent : ℝ := 300

/-- The number of additional dolls that can be bought if choosing small monkey dolls instead of large monkey dolls -/
def additional_small_dolls : ℕ := 25

/-- The number of fewer dolls that can be bought if choosing elephant dolls instead of large monkey dolls -/
def fewer_elephant_dolls : ℕ := 15

theorem large_monkey_doll_cost :
  (total_spent / (large_monkey_cost - 2) = total_spent / large_monkey_cost + additional_small_dolls) ∧
  (total_spent / (large_monkey_cost + 1) = total_spent / large_monkey_cost - fewer_elephant_dolls) := by
  sorry

end NUMINAMATH_CALUDE_large_monkey_doll_cost_l2765_276513


namespace NUMINAMATH_CALUDE_cheeseburger_cost_is_three_l2765_276592

def restaurant_problem (cheeseburger_cost : ℝ) : Prop :=
  let jim_money : ℝ := 20
  let cousin_money : ℝ := 10
  let total_money : ℝ := jim_money + cousin_money
  let spent_percentage : ℝ := 0.8
  let milkshake_cost : ℝ := 5
  let cheese_fries_cost : ℝ := 8
  let total_spent : ℝ := total_money * spent_percentage
  let num_cheeseburgers : ℕ := 2
  let num_milkshakes : ℕ := 2
  total_spent = num_cheeseburgers * cheeseburger_cost + num_milkshakes * milkshake_cost + cheese_fries_cost

theorem cheeseburger_cost_is_three :
  restaurant_problem 3 := by sorry

end NUMINAMATH_CALUDE_cheeseburger_cost_is_three_l2765_276592


namespace NUMINAMATH_CALUDE_bert_spent_nine_at_dry_cleaners_l2765_276545

/-- Represents Bert's spending problem --/
def BertSpending (initial_amount : ℚ) (dry_cleaner_amount : ℚ) : Prop :=
  let hardware_store := initial_amount / 4
  let after_hardware := initial_amount - hardware_store
  let after_dry_cleaner := after_hardware - dry_cleaner_amount
  let grocery_store := after_dry_cleaner / 2
  let final_amount := after_dry_cleaner - grocery_store
  (initial_amount = 44) ∧ (final_amount = 12)

/-- Theorem stating that Bert spent $9 at the dry cleaners --/
theorem bert_spent_nine_at_dry_cleaners :
  ∃ (dry_cleaner_amount : ℚ), BertSpending 44 dry_cleaner_amount ∧ dry_cleaner_amount = 9 := by
  sorry

end NUMINAMATH_CALUDE_bert_spent_nine_at_dry_cleaners_l2765_276545


namespace NUMINAMATH_CALUDE_files_remaining_l2765_276504

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 27)
  (h2 : video_files = 42)
  (h3 : deleted_files = 11) :
  music_files + video_files - deleted_files = 58 := by
  sorry

end NUMINAMATH_CALUDE_files_remaining_l2765_276504


namespace NUMINAMATH_CALUDE_petya_final_vote_percentage_l2765_276556

theorem petya_final_vote_percentage 
  (x : ℝ) -- Total votes by noon
  (y : ℝ) -- Votes cast after noon
  (h1 : 0.45 * x = 0.27 * (x + y)) -- Vasya's final vote count
  (h2 : y = (2/3) * x) -- Relationship between x and y
  : (0.25 * x + y) / (x + y) = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_petya_final_vote_percentage_l2765_276556


namespace NUMINAMATH_CALUDE_even_quadratic_function_l2765_276508

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem even_quadratic_function (a b c : ℝ) :
  (∀ x, (2 * a - 3 ≤ x ∧ x ≤ 1) → f a b c x = f a b c (-x)) →
  a = 1 ∧ b = 0 ∧ ∃ c : ℝ, True :=
by sorry

end NUMINAMATH_CALUDE_even_quadratic_function_l2765_276508


namespace NUMINAMATH_CALUDE_sector_angle_l2765_276553

/-- Given a circular sector with arc length and area both equal to 3,
    prove that the central angle in radians is 3/2. -/
theorem sector_angle (r : ℝ) (θ : ℝ) 
  (arc_length : θ * r = 3)
  (area : 1/2 * θ * r^2 = 3) :
  θ = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l2765_276553


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2765_276516

theorem triangle_angle_measure (angle_CBD angle_other : ℝ) :
  angle_CBD = 117 →
  angle_other = 31 →
  ∃ (angle_y : ℝ), 
    angle_y + angle_other + (180 - angle_CBD) = 180 ∧
    angle_y = 86 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2765_276516


namespace NUMINAMATH_CALUDE_orchids_cut_l2765_276514

theorem orchids_cut (initial_red : ℕ) (initial_white : ℕ) (final_red : ℕ) : 
  final_red - initial_red = final_red - initial_red :=
by
  sorry

#check orchids_cut 9 3 15

end NUMINAMATH_CALUDE_orchids_cut_l2765_276514


namespace NUMINAMATH_CALUDE_october_birth_percentage_l2765_276598

def total_people : ℕ := 100
def october_births : ℕ := 6

theorem october_birth_percentage :
  (october_births : ℚ) / total_people * 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_october_birth_percentage_l2765_276598


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2765_276571

theorem sum_of_reciprocals (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (h : a + b + c = 0) :
  1 / (b^3 + c^3 - a^3) + 1 / (a^3 + c^3 - b^3) + 1 / (a^3 + b^3 - c^3) = 1 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2765_276571


namespace NUMINAMATH_CALUDE_unbroken_seashells_l2765_276551

/-- Given that Mike found a total of 6 seashells and 4 of them were broken,
    prove that the number of unbroken seashells is 2. -/
theorem unbroken_seashells (total : ℕ) (broken : ℕ) (h1 : total = 6) (h2 : broken = 4) :
  total - broken = 2 := by
  sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l2765_276551


namespace NUMINAMATH_CALUDE_smallest_inverse_mod_735_l2765_276536

theorem smallest_inverse_mod_735 : 
  ∀ n : ℕ, n > 2 → (∃ m : ℕ, n * m ≡ 1 [MOD 735]) → n ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_inverse_mod_735_l2765_276536


namespace NUMINAMATH_CALUDE_complex_number_location_l2765_276555

theorem complex_number_location : ∃ (z : ℂ), z = 2 / (1 - Complex.I) - 2 ∧ 
  (z.re < 0 ∧ z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l2765_276555


namespace NUMINAMATH_CALUDE_sum_of_17th_roots_minus_one_l2765_276525

theorem sum_of_17th_roots_minus_one (ω : ℂ) : 
  ω^17 = 1 → ω ≠ 1 → ω + ω^2 + ω^3 + ω^4 + ω^5 + ω^6 + ω^7 + ω^8 + ω^9 + ω^10 + ω^11 + ω^12 + ω^13 + ω^14 + ω^15 + ω^16 = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_17th_roots_minus_one_l2765_276525


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2765_276544

theorem smallest_number_divisible (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    m - 5 = 27 * k₁ ∧ 
    m - 5 = 36 * k₂ ∧ 
    m - 5 = 44 * k₃ ∧ 
    m - 5 = 52 * k₄ ∧ 
    m - 5 = 65 * k₅)) →
  (∃ k₁ k₂ k₃ k₄ k₅ : ℕ, 
    n - 5 = 27 * k₁ ∧ 
    n - 5 = 36 * k₂ ∧ 
    n - 5 = 44 * k₃ ∧ 
    n - 5 = 52 * k₄ ∧ 
    n - 5 = 65 * k₅) →
  n = 386105 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l2765_276544


namespace NUMINAMATH_CALUDE_regular_hexagon_perimeter_l2765_276584

/-- The perimeter of a regular hexagon with side length 5 cm is 30 cm. -/
theorem regular_hexagon_perimeter :
  ∀ (side_length : ℝ),
  side_length = 5 →
  (6 : ℝ) * side_length = 30 :=
by sorry

end NUMINAMATH_CALUDE_regular_hexagon_perimeter_l2765_276584


namespace NUMINAMATH_CALUDE_whatsapp_messages_theorem_l2765_276520

/-- The number of messages sent on Monday in a Whatsapp group -/
def monday_messages : ℕ := sorry

/-- The number of messages sent on Tuesday in a Whatsapp group -/
def tuesday_messages : ℕ := 200

/-- The number of messages sent on Wednesday in a Whatsapp group -/
def wednesday_messages : ℕ := tuesday_messages + 300

/-- The number of messages sent on Thursday in a Whatsapp group -/
def thursday_messages : ℕ := 2 * wednesday_messages

/-- The total number of messages sent over four days in a Whatsapp group -/
def total_messages : ℕ := 2000

theorem whatsapp_messages_theorem :
  monday_messages + tuesday_messages + wednesday_messages + thursday_messages = total_messages ∧
  monday_messages = 300 := by sorry

end NUMINAMATH_CALUDE_whatsapp_messages_theorem_l2765_276520


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2765_276523

theorem polynomial_remainder (x : ℝ) : 
  (x^3 - 3*x + 5) % (x - 1) = 3 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2765_276523


namespace NUMINAMATH_CALUDE_triangle_property_l2765_276587

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    proves that under certain conditions, angle A is π/3 and the area is 3√3. -/
theorem triangle_property (a b c A B C : ℝ) : 
  0 < A ∧ A < π ∧   -- A is in (0, π)
  0 < B ∧ B < π ∧   -- B is in (0, π)
  0 < C ∧ C < π ∧   -- C is in (0, π)
  A + B + C = π ∧   -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧   -- Positive side lengths
  a * Real.sin C = Real.sqrt 3 * c * Real.cos A ∧   -- Given condition
  a = Real.sqrt 13 ∧   -- Given value of a
  c = 3 →   -- Given value of c
  A = π / 3 ∧   -- Angle A is 60°
  (1 / 2) * b * c * Real.sin A = 3 * Real.sqrt 3   -- Area of triangle ABC
  := by sorry

end NUMINAMATH_CALUDE_triangle_property_l2765_276587
