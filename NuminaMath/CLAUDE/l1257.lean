import Mathlib

namespace NUMINAMATH_CALUDE_average_sleep_time_l1257_125716

def sleep_times : List ℝ := [10, 9, 10, 8, 8]

theorem average_sleep_time :
  (sleep_times.sum / sleep_times.length : ℝ) = 9 := by sorry

end NUMINAMATH_CALUDE_average_sleep_time_l1257_125716


namespace NUMINAMATH_CALUDE_quinn_free_donuts_l1257_125745

/-- The number of books required to earn one donut coupon -/
def books_per_coupon : ℕ := 5

/-- The number of books Quinn reads per week -/
def books_per_week : ℕ := 2

/-- The number of weeks Quinn reads -/
def weeks_read : ℕ := 10

/-- The total number of books Quinn reads -/
def total_books : ℕ := books_per_week * weeks_read

/-- The number of free donuts Quinn is eligible for -/
def free_donuts : ℕ := total_books / books_per_coupon

theorem quinn_free_donuts : free_donuts = 4 := by
  sorry

end NUMINAMATH_CALUDE_quinn_free_donuts_l1257_125745


namespace NUMINAMATH_CALUDE_lg_expression_equals_one_l1257_125730

-- Define the base 10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_expression_equals_one :
  lg 2 ^ 2 * lg 250 + lg 5 ^ 2 * lg 40 = 1 := by sorry

end NUMINAMATH_CALUDE_lg_expression_equals_one_l1257_125730


namespace NUMINAMATH_CALUDE_refrigerator_is_right_prism_other_objects_not_right_prisms_l1257_125731

-- Define the properties of a right prism
structure RightPrism :=
  (has_congruent_polygonal_bases : Bool)
  (has_rectangular_lateral_faces : Bool)

-- Define the properties of a refrigerator
structure Refrigerator :=
  (shape : RightPrism)

-- Theorem stating that a refrigerator can be modeled as a right prism
theorem refrigerator_is_right_prism (r : Refrigerator) : 
  r.shape.has_congruent_polygonal_bases ∧ r.shape.has_rectangular_lateral_faces := by
  sorry

-- Define other objects for comparison
structure Basketball :=
  (is_spherical : Bool)

structure Shuttlecock :=
  (has_conical_shape : Bool)

structure Thermos :=
  (is_cylindrical : Bool)

-- Theorem stating that other objects are not right prisms
theorem other_objects_not_right_prisms : 
  ∀ (b : Basketball) (s : Shuttlecock) (t : Thermos),
  ¬(∃ (rp : RightPrism), rp.has_congruent_polygonal_bases ∧ rp.has_rectangular_lateral_faces) := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_is_right_prism_other_objects_not_right_prisms_l1257_125731


namespace NUMINAMATH_CALUDE_binomial_expansion_probability_l1257_125713

/-- The number of terms in the binomial expansion -/
def num_terms : ℕ := 9

/-- The exponent of the binomial -/
def n : ℕ := num_terms - 1

/-- The number of rational terms in the expansion -/
def num_rational_terms : ℕ := 3

/-- The number of irrational terms in the expansion -/
def num_irrational_terms : ℕ := num_terms - num_rational_terms

/-- The total number of permutations of all terms -/
def total_permutations : ℕ := (Nat.factorial num_terms)

/-- The number of favorable permutations where rational terms are not adjacent -/
def favorable_permutations : ℕ := 
  (Nat.factorial num_irrational_terms) * (Nat.choose (num_irrational_terms + 1) num_rational_terms)

/-- The probability that all rational terms are not adjacent when rearranged -/
def probability : ℚ := (favorable_permutations : ℚ) / (total_permutations : ℚ)

theorem binomial_expansion_probability : probability = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_probability_l1257_125713


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1257_125779

theorem trigonometric_equation_solution (x : ℝ) (k : ℤ) : 
  8.424 * Real.cos x + Real.sqrt (Real.sin x ^ 2 - 2 * Real.sin (2 * x) + 4 * Real.cos x ^ 2) = 0 ↔ 
  (x = Real.arctan (-6.424) + π * (2 * ↑k + 1) ∨ x = Real.arctan 5.212 + π * (2 * ↑k + 1)) :=
by sorry


end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1257_125779


namespace NUMINAMATH_CALUDE_complement_of_union_l1257_125708

universe u

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l1257_125708


namespace NUMINAMATH_CALUDE_original_number_proof_l1257_125795

theorem original_number_proof : 
  ∃! x : ℕ, (x + 4) % 23 = 0 ∧ ∀ y : ℕ, y < 4 → (x + y) % 23 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1257_125795


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_six_l1257_125767

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_six (n : ℕ) : Prop := n % 6 = 0

theorem smallest_four_digit_divisible_by_six :
  ∀ n : ℕ, is_four_digit n → divisible_by_six n → n ≥ 1002 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_six_l1257_125767


namespace NUMINAMATH_CALUDE_arithmetic_operation_l1257_125747

theorem arithmetic_operation : 5 + 4 - 3 + 2 - 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operation_l1257_125747


namespace NUMINAMATH_CALUDE_journey_distance_l1257_125790

/-- Proves that a journey with given conditions has a total distance of 224 km -/
theorem journey_distance (total_time : ℝ) (speed_first_half : ℝ) (speed_second_half : ℝ)
  (h1 : total_time = 10)
  (h2 : speed_first_half = 21)
  (h3 : speed_second_half = 24) :
  ∃ (distance : ℝ),
    distance = 224 ∧
    total_time = (distance / 2) / speed_first_half + (distance / 2) / speed_second_half :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_l1257_125790


namespace NUMINAMATH_CALUDE_pizza_toppings_l1257_125761

/-- Given a pizza with 24 slices, where every slice has at least one topping,
    if exactly 15 slices have ham and exactly 17 slices have cheese,
    then the number of slices with both ham and cheese is 8. -/
theorem pizza_toppings (total : Nat) (ham : Nat) (cheese : Nat) (both : Nat) :
  total = 24 →
  ham = 15 →
  cheese = 17 →
  both + (ham - both) + (cheese - both) = total →
  both = 8 := by
sorry

end NUMINAMATH_CALUDE_pizza_toppings_l1257_125761


namespace NUMINAMATH_CALUDE_angle_measure_theorem_l1257_125783

theorem angle_measure_theorem (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_theorem_l1257_125783


namespace NUMINAMATH_CALUDE_outfit_count_l1257_125742

/-- The number of shirts available. -/
def num_shirts : ℕ := 8

/-- The number of ties available. -/
def num_ties : ℕ := 5

/-- The number of pants available. -/
def num_pants : ℕ := 3

/-- The number of belts available. -/
def num_belts : ℕ := 2

/-- The number of tie options (including no tie). -/
def tie_options : ℕ := num_ties + 1

/-- The number of belt options (including no belt). -/
def belt_options : ℕ := num_belts + 1

/-- The total number of possible outfits. -/
def total_outfits : ℕ := num_shirts * num_pants * tie_options * belt_options

theorem outfit_count : total_outfits = 432 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_l1257_125742


namespace NUMINAMATH_CALUDE_train_length_calculation_l1257_125738

/-- Calculates the length of a train given its speed, the speed of a person walking in the same direction, and the time it takes for the train to pass the person completely. -/
theorem train_length_calculation (train_speed man_speed : ℝ) (passing_time : ℝ) 
  (h1 : train_speed = 46.5) 
  (h2 : man_speed = 2.5) 
  (h3 : passing_time = 62.994960403167745) : 
  ∃ (length : ℝ), abs (length - 770) < 0.1 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l1257_125738


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l1257_125772

theorem min_sum_absolute_values (x : ℝ) :
  ∃ (min : ℝ), min = 4 ∧ 
  (∀ y : ℝ, |y + 3| + |y + 6| + |y + 7| ≥ min) ∧
  (|x + 3| + |x + 6| + |x + 7| = min) :=
sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l1257_125772


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l1257_125755

theorem polynomial_product_expansion (x : ℝ) :
  (2 * x^3 - 3 * x^2 + 4) * (3 * x^2 + x + 1) =
  6 * x^5 - 7 * x^4 - x^3 + 9 * x^2 + 4 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l1257_125755


namespace NUMINAMATH_CALUDE_diane_honey_harvest_l1257_125743

/-- The total amount of honey harvested over three years -/
def total_honey_harvest (year1 : ℕ) (increase_year2 : ℕ) (increase_year3 : ℕ) : ℕ :=
  year1 + (year1 + increase_year2) + (year1 + increase_year2 + increase_year3)

/-- Theorem stating the total honey harvest over three years -/
theorem diane_honey_harvest :
  total_honey_harvest 2479 6085 7890 = 27497 := by
  sorry

end NUMINAMATH_CALUDE_diane_honey_harvest_l1257_125743


namespace NUMINAMATH_CALUDE_range_of_f_l1257_125701

def f (x : ℤ) : ℤ := (x - 1)^2 + 1

def domain : Set ℤ := {-1, 0, 1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1257_125701


namespace NUMINAMATH_CALUDE_expression_simplification_l1257_125734

theorem expression_simplification (y : ℝ) : 
  3*y - 5*y^2 + 2 + (8 - 5*y + 2*y^2) = -3*y^2 - 2*y + 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1257_125734


namespace NUMINAMATH_CALUDE_min_value_g_l1257_125714

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
def distance (p q : Point3D) : ℝ := sorry

/-- Represents a tetrahedron ABCD -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Function g(X) as defined in the problem -/
def g (tetra : Tetrahedron) (X : Point3D) : ℝ :=
  distance tetra.A X + distance tetra.B X + distance tetra.C X + distance tetra.D X

/-- Theorem stating the minimum value of g(X) for the given tetrahedron -/
theorem min_value_g (tetra : Tetrahedron) 
  (h1 : distance tetra.A tetra.D = 30)
  (h2 : distance tetra.B tetra.C = 30)
  (h3 : distance tetra.A tetra.C = 46)
  (h4 : distance tetra.B tetra.D = 46)
  (h5 : distance tetra.A tetra.B = 50)
  (h6 : distance tetra.C tetra.D = 50) :
  ∃ (min_val : ℝ), min_val = 4 * Real.sqrt 628 ∧ ∀ (X : Point3D), g tetra X ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_g_l1257_125714


namespace NUMINAMATH_CALUDE_converse_zero_product_l1257_125740

theorem converse_zero_product (a b : ℝ) : 
  (ab ≠ 0 → a ≠ 0 ∧ b ≠ 0) ↔ (ab = 0 → a = 0 ∨ b = 0) := by sorry

end NUMINAMATH_CALUDE_converse_zero_product_l1257_125740


namespace NUMINAMATH_CALUDE_simplify_sqrt_fraction_l1257_125796

theorem simplify_sqrt_fraction : 
  (Real.sqrt 45) / (2 * Real.sqrt 20) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_fraction_l1257_125796


namespace NUMINAMATH_CALUDE_fraction_of_shaded_hexagons_l1257_125751

/-- Given a set of hexagons, some of which are shaded, prove that the fraction of shaded hexagons is correct. -/
theorem fraction_of_shaded_hexagons 
  (total : ℕ) 
  (shaded : ℕ) 
  (h1 : total = 9) 
  (h2 : shaded = 5) : 
  (shaded : ℚ) / total = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_shaded_hexagons_l1257_125751


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l1257_125786

/-- A curve defined by y = -x³ + 2 -/
def curve (x : ℝ) : ℝ := -x^3 + 2

/-- A line defined by y = -6x + b -/
def line (b : ℝ) (x : ℝ) : ℝ := -6*x + b

/-- The derivative of the curve -/
def curve_derivative (x : ℝ) : ℝ := -3*x^2

theorem tangent_line_b_value :
  ∀ b : ℝ,
  (∃ x : ℝ, curve x = line b x ∧ curve_derivative x = -6) →
  (b = 2 + 4 * Real.sqrt 2 ∨ b = 2 - 4 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_b_value_l1257_125786


namespace NUMINAMATH_CALUDE_unique_solution_is_x_minus_one_l1257_125748

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y + 1) = y + f (f x * f y)

/-- The main theorem stating that the only function satisfying the equation is f(x) = x - 1 -/
theorem unique_solution_is_x_minus_one (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  ∀ x : ℝ, f x = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_x_minus_one_l1257_125748


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1257_125789

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation (given_line : Line) (point : Point) :
  given_line = Line.mk 3 (-2) 0 →
  point = Point.mk 1 (-1) →
  ∃ (result_line : Line),
    result_line.perpendicular given_line ∧
    point.liesOn result_line ∧
    result_line = Line.mk 2 3 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1257_125789


namespace NUMINAMATH_CALUDE_sandwich_cost_l1257_125775

/-- The cost of tomatoes for N sandwiches, each using T slices at 4 cents per slice --/
def tomatoCost (N T : ℕ) : ℚ := (N * T * 4 : ℕ) / 100

/-- The total cost of ingredients for N sandwiches, each using C slices of cheese and T slices of tomato --/
def totalCost (N C T : ℕ) : ℚ := (N * (3 * C + 4 * T) : ℕ) / 100

theorem sandwich_cost (N C T : ℕ) : 
  N > 1 → C > 0 → T > 0 → totalCost N C T = 305 / 100 → tomatoCost N T = 2 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_l1257_125775


namespace NUMINAMATH_CALUDE_angle_point_cosine_l1257_125739

/-- Given an angle α in the first quadrant and a point P(a, √5) on its terminal side,
    if cos α = (√2/4)a, then a = √3 -/
theorem angle_point_cosine (α : Real) (a : Real) :
  0 < α ∧ α < π / 2 →  -- α is in the first quadrant
  (∃ (P : ℝ × ℝ), P = (a, Real.sqrt 5) ∧ P.1 / Real.sqrt (P.1^2 + P.2^2) = Real.cos α) →  -- P(a, √5) is on the terminal side
  Real.cos α = (Real.sqrt 2 / 4) * a →  -- given condition
  a = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_point_cosine_l1257_125739


namespace NUMINAMATH_CALUDE_tax_free_limit_correct_l1257_125711

/-- The tax-free total value limit for imported goods in country X. -/
def tax_free_limit : ℝ := 500

/-- The tax rate applied to the value exceeding the tax-free limit. -/
def tax_rate : ℝ := 0.08

/-- The total value of goods imported by a specific tourist. -/
def total_value : ℝ := 730

/-- The tax paid by the tourist. -/
def tax_paid : ℝ := 18.40

/-- Theorem stating that the tax-free limit is correct given the problem conditions. -/
theorem tax_free_limit_correct : 
  tax_rate * (total_value - tax_free_limit) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_tax_free_limit_correct_l1257_125711


namespace NUMINAMATH_CALUDE_total_shells_l1257_125700

theorem total_shells (morning_shells afternoon_shells : ℕ) 
  (h1 : morning_shells = 292) 
  (h2 : afternoon_shells = 324) : 
  morning_shells + afternoon_shells = 616 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_l1257_125700


namespace NUMINAMATH_CALUDE_fraction_transformation_l1257_125754

theorem fraction_transformation (a b : ℝ) (h : b ≠ 0) :
  a / b = (a + 2 * a) / (b + 2 * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l1257_125754


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l1257_125718

/-- The length of a train given specific conditions --/
theorem train_length : ℝ :=
  let jogger_speed : ℝ := 9 -- km/hr
  let train_speed : ℝ := 45 -- km/hr
  let initial_distance : ℝ := 240 -- meters
  let passing_time : ℝ := 34 -- seconds

  100 -- meters

/-- Proof that the train length is correct given the conditions --/
theorem train_length_proof :
  let jogger_speed : ℝ := 9 -- km/hr
  let train_speed : ℝ := 45 -- km/hr
  let initial_distance : ℝ := 240 -- meters
  let passing_time : ℝ := 34 -- seconds
  train_length = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l1257_125718


namespace NUMINAMATH_CALUDE_henry_birthday_money_l1257_125762

theorem henry_birthday_money (initial_amount spent_amount final_amount : ℕ) 
  (h1 : initial_amount = 11)
  (h2 : spent_amount = 10)
  (h3 : final_amount = 19) :
  final_amount + spent_amount - initial_amount = 18 := by
  sorry

end NUMINAMATH_CALUDE_henry_birthday_money_l1257_125762


namespace NUMINAMATH_CALUDE_students_liking_sports_l1257_125725

theorem students_liking_sports (B C : Finset Nat) : 
  (B.card = 10) → 
  (C.card = 8) → 
  ((B ∩ C).card = 4) → 
  ((B ∪ C).card = 14) := by
  sorry

end NUMINAMATH_CALUDE_students_liking_sports_l1257_125725


namespace NUMINAMATH_CALUDE_f_difference_l1257_125780

/-- The function f(x) = x^4 + 3x^3 + x^2 + 7x -/
def f (x : ℝ) : ℝ := x^4 + 3*x^3 + x^2 + 7*x

/-- Theorem: f(3) - f(-3) = 204 -/
theorem f_difference : f 3 - f (-3) = 204 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l1257_125780


namespace NUMINAMATH_CALUDE_total_shoes_l1257_125749

def shoe_store_problem (brown_shoes : ℕ) (black_shoes : ℕ) : Prop :=
  black_shoes = 2 * brown_shoes ∧
  brown_shoes = 22 ∧
  black_shoes + brown_shoes = 66

theorem total_shoes : ∃ (brown_shoes black_shoes : ℕ), shoe_store_problem brown_shoes black_shoes := by
  sorry

end NUMINAMATH_CALUDE_total_shoes_l1257_125749


namespace NUMINAMATH_CALUDE_time_after_12345_seconds_l1257_125707

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  valid : hours < 24 ∧ minutes < 60 ∧ seconds < 60

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

theorem time_after_12345_seconds : 
  addSeconds ⟨18, 15, 0, sorry⟩ 12345 = ⟨21, 40, 45, sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_time_after_12345_seconds_l1257_125707


namespace NUMINAMATH_CALUDE_max_value_of_f_l1257_125792

-- Define the function
def f (x : ℝ) : ℝ := 12 * x - 4 * x^2 + 2

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 11 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1257_125792


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1257_125763

theorem pure_imaginary_complex_number (m : ℝ) :
  (((m^2 + 2*m - 3) : ℂ) + (m - 1)*I = (0 : ℂ) + ((m - 1)*I : ℂ)) → m = -3 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1257_125763


namespace NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l1257_125727

def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem first_term_of_geometric_sequence 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : geometric_sequence a r 4 = 24) 
  (h2 : geometric_sequence a r 5 = 48) : 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l1257_125727


namespace NUMINAMATH_CALUDE_tetrahedron_volume_with_inscribed_sphere_l1257_125750

/-- The volume of a tetrahedron with an inscribed sphere -/
theorem tetrahedron_volume_with_inscribed_sphere
  (R : ℝ)  -- Radius of the inscribed sphere
  (S₁ S₂ S₃ S₄ : ℝ)  -- Areas of the four faces of the tetrahedron
  (h₁ : R > 0)  -- Radius is positive
  (h₂ : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0)  -- Face areas are positive
  : ∃ V : ℝ, V = R * (S₁ + S₂ + S₃ + S₄) ∧ V > 0 :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_with_inscribed_sphere_l1257_125750


namespace NUMINAMATH_CALUDE_compare_numbers_l1257_125793

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

def num_base_6 : List Nat := [5, 4]
def num_base_4 : List Nat := [2, 3]
def num_base_5 : List Nat := [3, 2, 1]

theorem compare_numbers :
  (base_to_decimal num_base_6 6 + base_to_decimal num_base_4 4) > base_to_decimal num_base_5 5 :=
by sorry

end NUMINAMATH_CALUDE_compare_numbers_l1257_125793


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_inequality_l1257_125784

theorem negation_of_existence (P : ℝ → Prop) :
  (¬∃ x > 0, P x) ↔ (∀ x > 0, ¬P x) :=
by sorry

theorem negation_of_exponential_inequality :
  (¬∃ x > 0, 3^x < x^2) ↔ (∀ x > 0, 3^x ≥ x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_inequality_l1257_125784


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l1257_125766

theorem absolute_value_inequality_solution (x : ℝ) :
  (|x + 2| + |x - 2| < x + 7) ↔ (-7/3 < x ∧ x < 7) := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l1257_125766


namespace NUMINAMATH_CALUDE_hyperbola_min_distance_hyperbola_min_distance_achieved_l1257_125723

theorem hyperbola_min_distance (x y : ℝ) : 
  (x^2 / 8) - (y^2 / 4) = 1 → |x - y| ≥ 2 :=
by sorry

theorem hyperbola_min_distance_achieved : 
  ∃ (x y : ℝ), (x^2 / 8) - (y^2 / 4) = 1 ∧ |x - y| = 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_min_distance_hyperbola_min_distance_achieved_l1257_125723


namespace NUMINAMATH_CALUDE_saras_hourly_wage_l1257_125798

def saras_paycheck (hours_per_week : ℕ) (weeks_worked : ℕ) (tire_cost : ℕ) (money_left : ℕ) : ℚ :=
  let total_earnings := tire_cost + money_left
  let total_hours := hours_per_week * weeks_worked
  (total_earnings : ℚ) / total_hours

theorem saras_hourly_wage :
  saras_paycheck 40 2 410 510 = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_saras_hourly_wage_l1257_125798


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l1257_125758

theorem smallest_number_with_remainders : ∃ (x : ℕ), 
  (x % 3 = 2) ∧ (x % 5 = 3) ∧ (x % 7 = 4) ∧
  (∀ y : ℕ, y < x → ¬((y % 3 = 2) ∧ (y % 5 = 3) ∧ (y % 7 = 4))) ∧
  x = 23 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l1257_125758


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l1257_125778

/-- In a right triangle, if the hypotenuse exceeds one leg by 2, then the square of the other leg is 4a + 4 -/
theorem right_triangle_leg_square (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) -- Pythagorean theorem
  (h5 : c = a + 2) : -- Hypotenuse exceeds one leg by 2
  b^2 = 4*a + 4 := by sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l1257_125778


namespace NUMINAMATH_CALUDE_subtraction_of_integers_l1257_125728

theorem subtraction_of_integers : 2 - 3 = -1 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_integers_l1257_125728


namespace NUMINAMATH_CALUDE_sally_bought_twenty_cards_l1257_125799

/-- Calculates the number of Pokemon cards Sally bought -/
def cards_sally_bought (initial : ℕ) (from_dan : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial + from_dan)

/-- Proves that Sally bought 20 Pokemon cards -/
theorem sally_bought_twenty_cards : 
  cards_sally_bought 27 41 88 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sally_bought_twenty_cards_l1257_125799


namespace NUMINAMATH_CALUDE_function_transformation_l1257_125735

theorem function_transformation (f : ℝ → ℝ) (h : f 1 = -1) : f (2 - 1) - 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l1257_125735


namespace NUMINAMATH_CALUDE_percentage_of_300_l1257_125746

/-- Calculates the percentage of a given amount -/
def percentage (percent : ℚ) (amount : ℚ) : ℚ :=
  (percent / 100) * amount

/-- Proves that 25% of Rs. 300 is equal to Rs. 75 -/
theorem percentage_of_300 : percentage 25 300 = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_300_l1257_125746


namespace NUMINAMATH_CALUDE_expression_simplification_l1257_125712

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  (2 / (a + 1) + (a + 2) / (a^2 - 1)) / (a / (a + 1)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1257_125712


namespace NUMINAMATH_CALUDE_grocery_store_price_l1257_125717

/-- The price of a bulk warehouse deal for sparkling water -/
def bulk_price : ℚ := 12

/-- The number of cans in the bulk warehouse deal -/
def bulk_cans : ℕ := 48

/-- The additional cost per can at the grocery store compared to the bulk warehouse -/
def additional_cost : ℚ := 1/4

/-- The number of cans in the grocery store deal -/
def grocery_cans : ℕ := 12

/-- The price of the grocery store deal for sparkling water -/
def grocery_price : ℚ := 6

theorem grocery_store_price :
  grocery_price = (bulk_price / bulk_cans + additional_cost) * grocery_cans :=
by sorry

end NUMINAMATH_CALUDE_grocery_store_price_l1257_125717


namespace NUMINAMATH_CALUDE_student_group_arrangements_l1257_125785

/-- The number of ways to divide n students into k equal groups -/
def divide_students (n k : ℕ) : ℕ := sorry

/-- The number of ways to assign k groups to k different topics -/
def assign_topics (k : ℕ) : ℕ := sorry

theorem student_group_arrangements :
  let n : ℕ := 6  -- number of students
  let k : ℕ := 3  -- number of groups
  divide_students n k * assign_topics k = 540 :=
by sorry

end NUMINAMATH_CALUDE_student_group_arrangements_l1257_125785


namespace NUMINAMATH_CALUDE_johns_running_time_l1257_125753

theorem johns_running_time (H : ℝ) : 
  H > 0 →
  (12 : ℝ) * (1.75 * H) = 168 →
  H = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_running_time_l1257_125753


namespace NUMINAMATH_CALUDE_percentage_of_80_equal_to_12_l1257_125709

theorem percentage_of_80_equal_to_12 (p : ℝ) : 
  (p / 100) * 80 = 12 → p = 15 := by sorry

end NUMINAMATH_CALUDE_percentage_of_80_equal_to_12_l1257_125709


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1257_125744

/-- Calculates the length of a bridge given train parameters and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 170 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 205 := by sorry

#check bridge_length_calculation

end NUMINAMATH_CALUDE_bridge_length_calculation_l1257_125744


namespace NUMINAMATH_CALUDE_systematic_sample_result_l1257_125777

/-- Systematic sampling function -/
def systematicSample (populationSize sampleSize : ℕ) (firstSelected : ℕ) : ℕ → ℕ :=
  fun n => firstSelected + (n - 1) * (populationSize / sampleSize)

theorem systematic_sample_result 
  (populationSize sampleSize firstSelected : ℕ) 
  (h1 : populationSize = 800) 
  (h2 : sampleSize = 50) 
  (h3 : firstSelected = 11) 
  (h4 : firstSelected ≤ 16) :
  ∃ n : ℕ, 33 ≤ n ∧ n ≤ 48 ∧ systematicSample populationSize sampleSize firstSelected n = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_result_l1257_125777


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1257_125715

/-- Theorem: If a point (x₀, y₀) is outside a circle with radius r centered at the origin,
    then the line x₀x + y₀y = r² intersects the circle. -/
theorem line_intersects_circle (x₀ y₀ r : ℝ) (h : x₀^2 + y₀^2 > r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ x₀*x + y₀*y = r^2 := by
  sorry

/-- Definition: A point (x₀, y₀) is outside the circle x² + y² = r² -/
def point_outside_circle (x₀ y₀ r : ℝ) : Prop :=
  x₀^2 + y₀^2 > r^2

/-- Definition: The line equation x₀x + y₀y = r² -/
def line_equation (x₀ y₀ r x y : ℝ) : Prop :=
  x₀*x + y₀*y = r^2

/-- Definition: A point (x, y) is on the circle x² + y² = r² -/
def point_on_circle (x y r : ℝ) : Prop :=
  x^2 + y^2 = r^2

end NUMINAMATH_CALUDE_line_intersects_circle_l1257_125715


namespace NUMINAMATH_CALUDE_divisibility_by_7_and_11_l1257_125720

theorem divisibility_by_7_and_11 (n : ℕ) (h : n > 0) :
  (∃ k : ℤ, 3^(2*n+1) + 2^(n+2) = 7*k) ∧
  (∃ m : ℤ, 3^(2*n+2) + 2^(6*n+1) = 11*m) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_7_and_11_l1257_125720


namespace NUMINAMATH_CALUDE_water_problem_solution_l1257_125736

def water_problem (total_water : ℕ) (car_water : ℕ) (num_cars : ℕ) (plant_water_diff : ℕ) : ℕ :=
  let car_total := car_water * num_cars
  let plant_water := car_total - plant_water_diff
  let used_water := car_total + plant_water
  let remaining_water := total_water - used_water
  remaining_water / 2

theorem water_problem_solution :
  water_problem 65 7 2 11 = 24 := by
  sorry

end NUMINAMATH_CALUDE_water_problem_solution_l1257_125736


namespace NUMINAMATH_CALUDE_colin_speed_l1257_125741

/-- Proves that Colin's speed is 4 mph given the relationships between speeds of Bruce, Tony, Brandon, and Colin -/
theorem colin_speed (bruce_speed tony_speed brandon_speed colin_speed : ℝ) : 
  bruce_speed = 1 →
  tony_speed = 2 * bruce_speed →
  brandon_speed = tony_speed / 3 →
  colin_speed = 6 * brandon_speed →
  colin_speed = 4 := by
sorry

end NUMINAMATH_CALUDE_colin_speed_l1257_125741


namespace NUMINAMATH_CALUDE_balloon_count_l1257_125733

theorem balloon_count (friend_balloons : ℕ) (difference : ℕ) : 
  friend_balloons = 5 → difference = 2 → friend_balloons + difference = 7 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l1257_125733


namespace NUMINAMATH_CALUDE_prob_answer_within_four_rings_l1257_125703

/-- The probability of answering a phone call at a specific ring. -/
def prob_answer_at_ring : Fin 4 → ℝ
  | 0 => 0.1  -- First ring
  | 1 => 0.3  -- Second ring
  | 2 => 0.4  -- Third ring
  | 3 => 0.1  -- Fourth ring

/-- Theorem: The probability of answering the phone within the first four rings is 0.9. -/
theorem prob_answer_within_four_rings :
  (Finset.sum Finset.univ prob_answer_at_ring) = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_prob_answer_within_four_rings_l1257_125703


namespace NUMINAMATH_CALUDE_appliance_purchase_total_cost_l1257_125765

theorem appliance_purchase_total_cost : 
  let vacuum_original : ℝ := 250
  let vacuum_discount : ℝ := 0.20
  let dishwasher_cost : ℝ := 450
  let bundle_discount : ℝ := 75
  let sales_tax : ℝ := 0.07

  let vacuum_discounted : ℝ := vacuum_original * (1 - vacuum_discount)
  let subtotal : ℝ := vacuum_discounted + dishwasher_cost - bundle_discount
  let total_with_tax : ℝ := subtotal * (1 + sales_tax)

  total_with_tax = 615.25 := by sorry

end NUMINAMATH_CALUDE_appliance_purchase_total_cost_l1257_125765


namespace NUMINAMATH_CALUDE_inequality_solution_set_minimum_value_no_positive_solution_l1257_125710

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | f x ≤ x + 1} = {x : ℝ | 2/3 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for the minimum value of f(x)
theorem minimum_value :
  ∃ (k : ℝ), k = 1 ∧ ∀ (x : ℝ), f x ≥ k := by sorry

-- Theorem for non-existence of positive a and b
theorem no_positive_solution :
  ¬∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*a + b = 1 ∧ 1/a + 2/b = 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_minimum_value_no_positive_solution_l1257_125710


namespace NUMINAMATH_CALUDE_rational_inequality_solution_l1257_125737

theorem rational_inequality_solution (x : ℝ) : 
  (1 / (x * (x + 2)) - 1 / ((x + 1) * (x + 3)) < 1 / 4) ↔ 
  (x < -3 ∨ (-1 < x ∧ x < 0)) :=
sorry

end NUMINAMATH_CALUDE_rational_inequality_solution_l1257_125737


namespace NUMINAMATH_CALUDE_greatest_product_sum_2024_l1257_125774

theorem greatest_product_sum_2024 :
  (∃ (a b : ℤ), a + b = 2024 ∧ a * b = 1024144 ∧
    ∀ (x y : ℤ), x + y = 2024 → x * y ≤ 1024144) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_product_sum_2024_l1257_125774


namespace NUMINAMATH_CALUDE_x_squared_geq_one_necessary_not_sufficient_l1257_125752

theorem x_squared_geq_one_necessary_not_sufficient :
  (∀ x : ℝ, x ≥ 1 → x^2 ≥ 1) ∧
  (∃ x : ℝ, x^2 ≥ 1 ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_geq_one_necessary_not_sufficient_l1257_125752


namespace NUMINAMATH_CALUDE_intersection_M_N_l1257_125706

-- Define set M
def M : Set ℝ := {x | x < 2016}

-- Define set N
def N : Set ℝ := {x | ∃ y, y = Real.log (x - x^2)}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1257_125706


namespace NUMINAMATH_CALUDE_secretary_work_hours_l1257_125729

theorem secretary_work_hours (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / b = 2 / 3 →
  b / c = 3 / 5 →
  c = 40 →
  a + b + c = 80 :=
by sorry

end NUMINAMATH_CALUDE_secretary_work_hours_l1257_125729


namespace NUMINAMATH_CALUDE_ellipse_properties_l1257_125769

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 9 * x^2 + y^2 = 81

-- Define the major axis length
def major_axis_length : ℝ := 18

-- Define the foci coordinates
def foci_coordinates : Set (ℝ × ℝ) := {(0, 6*Real.sqrt 2), (0, -6*Real.sqrt 2)}

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := y^2 - x^2 = 36

-- Theorem statement
theorem ellipse_properties :
  (∀ x y, ellipse x y → 
    (major_axis_length = 18 ∧ 
     (x, y) ∈ foci_coordinates → 
     (x = 0 ∧ (y = 6*Real.sqrt 2 ∨ y = -6*Real.sqrt 2)))) ∧
  (∀ x y, hyperbola x y → 
    (∃ a b c : ℝ, a^2 = b^2 + c^2 ∧ 
                  c = 6*Real.sqrt 2 ∧ 
                  c/a = Real.sqrt 2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1257_125769


namespace NUMINAMATH_CALUDE_correct_quotient_calculation_l1257_125787

theorem correct_quotient_calculation (dividend : ℕ) (incorrect_quotient : ℕ) : 
  dividend > 0 →
  incorrect_quotient = 753 →
  dividend = 102 * (incorrect_quotient * 3) →
  dividend % 201 = 0 →
  dividend / 201 = 1146 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_calculation_l1257_125787


namespace NUMINAMATH_CALUDE_fraction_equality_sum_l1257_125791

theorem fraction_equality_sum (P Q : ℚ) : 
  (5 : ℚ) / 7 = P / 63 ∧ (5 : ℚ) / 7 = 70 / Q → P + Q = 143 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_sum_l1257_125791


namespace NUMINAMATH_CALUDE_smallest_a_in_special_progression_l1257_125705

theorem smallest_a_in_special_progression (a b c : ℤ) : 
  a < b → b < c → 
  (2 * b = a + c) →  -- arithmetic progression condition
  (c * c = a * b) →  -- geometric progression condition
  (∀ a' b' c' : ℤ, a' < b' → b' < c' → (2 * b' = a' + c') → (c' * c' = a' * b') → a ≤ a') →
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_smallest_a_in_special_progression_l1257_125705


namespace NUMINAMATH_CALUDE_set_equality_l1257_125724

def A : Set ℝ := {x : ℝ | |x| < 3}
def B : Set ℝ := {x : ℝ | x^2 - 3*x + 2 > 0}

theorem set_equality : {x : ℝ | x ∈ A ∧ x ∉ (A ∩ B)} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_set_equality_l1257_125724


namespace NUMINAMATH_CALUDE_sara_movie_expenses_l1257_125776

-- Define the cost of each item
def theater_ticket_cost : ℚ := 10.62
def theater_ticket_count : ℕ := 2
def rented_movie_cost : ℚ := 1.59
def purchased_movie_cost : ℚ := 13.95

-- Define the total spent on movies
def total_spent : ℚ :=
  theater_ticket_cost * theater_ticket_count + rented_movie_cost + purchased_movie_cost

-- Theorem to prove
theorem sara_movie_expenses : total_spent = 36.78 := by
  sorry

end NUMINAMATH_CALUDE_sara_movie_expenses_l1257_125776


namespace NUMINAMATH_CALUDE_total_age_problem_l1257_125756

theorem total_age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  b = 12 →
  a + b + c = 32 := by
sorry

end NUMINAMATH_CALUDE_total_age_problem_l1257_125756


namespace NUMINAMATH_CALUDE_ratio_of_special_means_l1257_125704

theorem ratio_of_special_means (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : (a + b) / 2 = 3 * Real.sqrt (a * b)) (h5 : a + b = 36) :
  a / b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_special_means_l1257_125704


namespace NUMINAMATH_CALUDE_sin_2theta_value_l1257_125732

theorem sin_2theta_value (θ : ℝ) 
  (h : ∑' n, (Real.sin θ) ^ (2 * n) = 3) : 
  Real.sin (2 * θ) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l1257_125732


namespace NUMINAMATH_CALUDE_fraction_simplification_l1257_125759

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hden : y^2 - 1/x^2 ≠ 0) : 
  (x^2 - 1/y^2) / (y^2 - 1/x^2) = x^2 / y^2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1257_125759


namespace NUMINAMATH_CALUDE_problem_solution_l1257_125719

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem problem_solution (x y a : ℕ) : 
  x > 0 → y > 0 → a > 0 → 
  x * y = 32 → 
  sum_of_digits ((10 ^ x) ^ a - 64) = 279 → 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1257_125719


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l1257_125794

theorem mod_equivalence_unique_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ -2023 [ZMOD 9] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l1257_125794


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1257_125782

-- Define the combination function
def C (n : ℕ) (k : ℕ) : ℕ := 
  if k ≤ n then Nat.choose n k else 0

-- Define the permutation function
def A (n : ℕ) (k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

theorem unique_solution_for_equation (x : ℕ) :
  x ≥ 7 → (3 * C (x - 3) 4 = 5 * A (x - 4) 2) → x = 11 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1257_125782


namespace NUMINAMATH_CALUDE_expression_equivalence_l1257_125721

theorem expression_equivalence (y : ℝ) (Q : ℝ) (h : 5 * (3 * y + 7 * Real.pi) = Q) :
  10 * (6 * y + 14 * Real.pi + 3) = 4 * Q + 30 := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l1257_125721


namespace NUMINAMATH_CALUDE_coin_division_problem_l1257_125797

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (n % 8 = 5) → 
  (n % 7 = 2) → 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 5 ∨ m % 7 ≠ 2)) →
  (n % 9 = 1) := by
sorry

end NUMINAMATH_CALUDE_coin_division_problem_l1257_125797


namespace NUMINAMATH_CALUDE_newspaper_delivery_ratio_l1257_125788

/-- The number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- The number of newspapers Jake delivers in a week -/
def jake_weekly : ℕ := 234

/-- The additional number of newspapers Miranda delivers compared to Jake in a month -/
def miranda_monthly_extra : ℕ := 936

/-- The ratio of newspapers Miranda delivers to Jake's deliveries in a week -/
def delivery_ratio : ℚ := (jake_weekly * weeks_per_month + miranda_monthly_extra) / (jake_weekly * weeks_per_month)

theorem newspaper_delivery_ratio :
  delivery_ratio = 2 := by sorry

end NUMINAMATH_CALUDE_newspaper_delivery_ratio_l1257_125788


namespace NUMINAMATH_CALUDE_percentage_of_330_l1257_125768

theorem percentage_of_330 : (33 + 1 / 3 : ℚ) / 100 * 330 = 110 := by sorry

end NUMINAMATH_CALUDE_percentage_of_330_l1257_125768


namespace NUMINAMATH_CALUDE_linear_function_b_values_l1257_125757

theorem linear_function_b_values (k b : ℝ) :
  (∀ x, -3 ≤ x ∧ x ≤ 1 → -1 ≤ k * x + b ∧ k * x + b ≤ 8) →
  b = 5/4 ∨ b = 23/4 := by
sorry

end NUMINAMATH_CALUDE_linear_function_b_values_l1257_125757


namespace NUMINAMATH_CALUDE_no_solution_for_sqrt_equation_l1257_125726

theorem no_solution_for_sqrt_equation :
  ¬ ∃ x : ℝ, x > 9 ∧ Real.sqrt (x - 9) + 3 = Real.sqrt (x + 9) - 3 := by
  sorry


end NUMINAMATH_CALUDE_no_solution_for_sqrt_equation_l1257_125726


namespace NUMINAMATH_CALUDE_circle_tangent_trajectory_l1257_125722

-- Define the circle M
def CircleM (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 10

-- Define the line on which the center of M lies
def CenterLine (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define points A, B, C, and D
def PointA : ℝ × ℝ := (-5, 0)
def PointB : ℝ × ℝ := (1, 0)
def PointC : ℝ × ℝ := (1, 2)
def PointD : ℝ × ℝ := (-3, 4)

-- Define the tangent line through C
def TangentLineC (x y : ℝ) : Prop := 3*x + y - 5 = 0

-- Define the trajectory of Q
def TrajectoryQ (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 10

theorem circle_tangent_trajectory :
  -- The center of M is on the given line
  (∃ x y, CircleM x y ∧ CenterLine x y) ∧
  -- M passes through A and B
  (CircleM PointA.1 PointA.2 ∧ CircleM PointB.1 PointB.2) →
  -- 1. Equation of circle M is correct
  (∀ x y, CircleM x y ↔ (x + 2)^2 + (y - 1)^2 = 10) ∧
  -- 2. Equation of tangent line through C is correct
  (∀ x y, TangentLineC x y ↔ 3*x + y - 5 = 0) ∧
  -- 3. Trajectory equation of Q is correct
  (∀ x y, (TrajectoryQ x y ∧ ¬((x, y) = (-1, 8) ∨ (x, y) = (-3, 4))) ↔
    (∃ x₀ y₀, CircleM x₀ y₀ ∧ 
      x = (-5 + x₀ + 3)/2 ∧ 
      y = (y₀ + 4)/2 ∧
      ¬((x, y) = (-1, 8) ∨ (x, y) = (-3, 4)))) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_trajectory_l1257_125722


namespace NUMINAMATH_CALUDE_min_sum_distances_l1257_125771

/-- An ellipse with equation x²/2 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- The center of the ellipse -/
def O : ℝ × ℝ := (0, 0)

/-- The left focus of the ellipse -/
def F : ℝ × ℝ := (-1, 0)

/-- The squared distance between two points -/
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- The theorem stating the minimum value of |OP|² + |PF|² -/
theorem min_sum_distances (P : ℝ × ℝ) (h : P ∈ Ellipse) :
  ∃ (m : ℝ), m = 2 ∧ ∀ Q ∈ Ellipse, m ≤ dist_squared O Q + dist_squared Q F :=
sorry

end NUMINAMATH_CALUDE_min_sum_distances_l1257_125771


namespace NUMINAMATH_CALUDE_colored_plane_congruent_triangle_l1257_125764

/-- A color type representing the 1992 colors -/
inductive Color
| c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10
-- ... (omitted for brevity, but in reality, this would list all 1992 colors)
| c1992

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle on the plane -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- A colored plane -/
def ColoredPlane := Point → Color

/-- Two triangles are congruent -/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- A point is an interior point of a line segment -/
def isInteriorPoint (p : Point) (a b : Point) : Prop := sorry

/-- The theorem to be proved -/
theorem colored_plane_congruent_triangle 
  (plane : ColoredPlane) (T : Triangle) : 
  ∃ T' : Triangle, congruent T T' ∧ 
    (∀ (p q : Point), 
      ((isInteriorPoint p T'.a T'.b ∧ isInteriorPoint q T'.b T'.c) ∨
       (isInteriorPoint p T'.b T'.c ∧ isInteriorPoint q T'.c T'.a) ∨
       (isInteriorPoint p T'.c T'.a ∧ isInteriorPoint q T'.a T'.b)) →
      plane p = plane q) :=
sorry

end NUMINAMATH_CALUDE_colored_plane_congruent_triangle_l1257_125764


namespace NUMINAMATH_CALUDE_chess_team_arrangements_l1257_125760

/-- Represents the number of boys in the chess team -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the chess team -/
def num_girls : ℕ := 4

/-- Represents the total number of team members -/
def total_members : ℕ := num_boys + num_girls

/-- Represents the requirement that two specific girls must sit together -/
def specific_girls_together : Prop := True

/-- Represents the requirement that a boy must sit at each end -/
def boy_at_each_end : Prop := True

/-- The number of possible arrangements of the chess team -/
def num_arrangements : ℕ := 72

/-- Theorem stating that the number of arrangements is 72 -/
theorem chess_team_arrangements :
  num_boys = 3 →
  num_girls = 4 →
  specific_girls_together →
  boy_at_each_end →
  num_arrangements = 72 := by sorry

end NUMINAMATH_CALUDE_chess_team_arrangements_l1257_125760


namespace NUMINAMATH_CALUDE_jackie_has_more_fruits_than_adam_l1257_125773

/-- Represents the number of fruits a person has -/
structure FruitCount where
  apples : ℕ
  oranges : ℕ
  bananas : ℚ

/-- Calculates the difference in total apples and oranges between two FruitCounts -/
def applePlusOrangeDifference (a b : FruitCount) : ℤ :=
  (b.apples + b.oranges : ℤ) - (a.apples + a.oranges)

theorem jackie_has_more_fruits_than_adam :
  let adam : FruitCount := { apples := 25, oranges := 34, bananas := 18.5 }
  let jackie : FruitCount := { apples := 43, oranges := 29, bananas := 16.5 }
  applePlusOrangeDifference adam jackie = 13 := by
  sorry

end NUMINAMATH_CALUDE_jackie_has_more_fruits_than_adam_l1257_125773


namespace NUMINAMATH_CALUDE_stream_speed_l1257_125702

theorem stream_speed (downstream_distance : ℝ) (downstream_time : ℝ) 
  (upstream_distance : ℝ) (upstream_time : ℝ) :
  downstream_distance = 250 →
  downstream_time = 7 →
  upstream_distance = 150 →
  upstream_time = 21 →
  ∃ s : ℝ, abs (s - 14.28) < 0.01 ∧ 
  (∃ b : ℝ, downstream_distance / downstream_time = b + s ∧
            upstream_distance / upstream_time = b - s) :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l1257_125702


namespace NUMINAMATH_CALUDE_circle_trajectory_and_min_distance_l1257_125781

-- Define the moving circle
def moving_circle (x y : ℝ) : Prop :=
  y > 0 ∧ Real.sqrt (x^2 + (y - 1)^2) = y + 1

-- Define the trajectory E
def trajectory_E (x y : ℝ) : Prop :=
  y > 0 ∧ x^2 = 4*y

-- Define points A and B on trajectory E
def point_on_E (x y : ℝ) : Prop :=
  trajectory_E x y

-- Define the perpendicular tangents condition
def perpendicular_tangents (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  point_on_E x₁ y₁ ∧ point_on_E x₂ y₂ ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -4

-- Main theorem
theorem circle_trajectory_and_min_distance :
  (∀ x y, moving_circle x y ↔ trajectory_E x y) ∧
  (∀ x₁ y₁ x₂ y₂, perpendicular_tangents x₁ y₁ x₂ y₂ →
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≥ 4) ∧
  (∃ x₁ y₁ x₂ y₂, perpendicular_tangents x₁ y₁ x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4) :=
sorry

end NUMINAMATH_CALUDE_circle_trajectory_and_min_distance_l1257_125781


namespace NUMINAMATH_CALUDE_quadratic_one_root_l1257_125770

theorem quadratic_one_root (m : ℝ) : 
  (∃! x : ℝ, x^2 + 6*m*x + 2*m = 0) → m = 2/9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l1257_125770
