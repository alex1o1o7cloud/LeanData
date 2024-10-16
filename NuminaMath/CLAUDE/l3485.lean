import Mathlib

namespace NUMINAMATH_CALUDE_min_n_greater_than_T10_plus_1013_l3485_348566

def T (n : ℕ) : ℚ := n + 1 - (1 / 2^n)

theorem min_n_greater_than_T10_plus_1013 :
  (∀ n : ℕ, n > T 10 + 1013 → n ≥ 1024) ∧
  (∃ n : ℕ, n > T 10 + 1013 ∧ n = 1024) :=
sorry

end NUMINAMATH_CALUDE_min_n_greater_than_T10_plus_1013_l3485_348566


namespace NUMINAMATH_CALUDE_third_degree_polynomial_property_l3485_348531

/-- A third-degree polynomial with real coefficients. -/
def ThirdDegreePolynomial := ℝ → ℝ

/-- The property that |f(1)| = |f(2)| = |f(4)| = 10 -/
def SatisfiesCondition (f : ThirdDegreePolynomial) : Prop :=
  |f 1| = 10 ∧ |f 2| = 10 ∧ |f 4| = 10

theorem third_degree_polynomial_property (f : ThirdDegreePolynomial) 
  (h : SatisfiesCondition f) : |f 0| = 34/3 := by
  sorry

end NUMINAMATH_CALUDE_third_degree_polynomial_property_l3485_348531


namespace NUMINAMATH_CALUDE_pizza_distribution_l3485_348545

theorem pizza_distribution (treShawn Michael LaMar : ℚ) : 
  treShawn = 1/2 →
  Michael = 1/3 →
  treShawn + Michael + LaMar = 1 →
  LaMar = 1/6 := by
sorry

end NUMINAMATH_CALUDE_pizza_distribution_l3485_348545


namespace NUMINAMATH_CALUDE_parabola_equation_l3485_348541

/-- A parabola with x-axis as axis of symmetry and vertex at origin -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- The parabola passes through the point (-2, -4) -/
def passes_through (par : Parabola) : Prop :=
  par.eq (-2) (-4)

/-- The standard equation of the parabola is y^2 = -8x -/
def standard_equation (par : Parabola) : Prop :=
  par.p = -4

theorem parabola_equation :
  ∃ (par : Parabola), passes_through par ∧ standard_equation par :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3485_348541


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3485_348519

theorem solution_set_of_inequality (f : ℝ → ℝ) (h1 : ∀ x ∈ Set.Icc (-1) 1, f (-x) + f x = 0)
  (h2 : ∀ m n, m ∈ Set.Icc 0 1 → n ∈ Set.Icc 0 1 → m ≠ n → (f m - f n) / (m - n) < 0) :
  {x : ℝ | f (1 - 3*x) ≤ f (x - 1)} = Set.Icc 0 (1/2) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3485_348519


namespace NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l3485_348588

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice thrown -/
def numDice : ℕ := 4

/-- The minimum possible sum when throwing the dice -/
def minSum : ℕ := numDice

/-- The maximum possible sum when throwing the dice -/
def maxSum : ℕ := numDice * numFaces

/-- The number of possible distinct sums -/
def numDistinctSums : ℕ := maxSum - minSum + 1

/-- The minimum number of throws required to guarantee a repeated sum -/
def minThrows : ℕ := numDistinctSums + 1

theorem min_throws_for_repeated_sum :
  minThrows = 22 := by sorry

end NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l3485_348588


namespace NUMINAMATH_CALUDE_bus_capacity_problem_l3485_348539

/-- 
Given a bus with capacity 200 people, prove that if it carries x fraction of its capacity 
on the first trip and 4/5 of its capacity on the return trip, and the total number of 
people on both trips is 310, then x = 3/4.
-/
theorem bus_capacity_problem (x : ℚ) : 
  (200 * x + 200 * (4/5) = 310) → x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_problem_l3485_348539


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3485_348592

theorem pure_imaginary_condition (m : ℝ) : 
  (∃ (y : ℝ), m^2 + m - 2 + (m^2 - 1) * Complex.I = y * Complex.I) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3485_348592


namespace NUMINAMATH_CALUDE_consecutive_numbers_lcm_660_l3485_348591

theorem consecutive_numbers_lcm_660 (x : ℕ) :
  (Nat.lcm x (Nat.lcm (x + 1) (x + 2)) = 660) →
  x = 10 ∧ (x + 1) = 11 ∧ (x + 2) = 12 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_lcm_660_l3485_348591


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l3485_348542

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l3485_348542


namespace NUMINAMATH_CALUDE_same_conclusion_from_true_and_false_l3485_348567

theorem same_conclusion_from_true_and_false :
  ∃ (A : Prop) (T F : Prop), T ∧ ¬F ∧ (T → A) ∧ (F → A) := by
  sorry

end NUMINAMATH_CALUDE_same_conclusion_from_true_and_false_l3485_348567


namespace NUMINAMATH_CALUDE_sweet_salty_difference_l3485_348571

/-- Represents the number of cookies of each type --/
structure CookieCount where
  sweet : ℕ
  salty : ℕ
  chocolate : ℕ

/-- The initial number of cookies Paco had --/
def initialCookies : CookieCount :=
  { sweet := 39, salty := 18, chocolate := 12 }

/-- The number of cookies Paco ate --/
def eatenCookies : CookieCount :=
  { sweet := 27, salty := 6, chocolate := 8 }

/-- Theorem stating the difference between sweet and salty cookies eaten --/
theorem sweet_salty_difference :
  eatenCookies.sweet - eatenCookies.salty = 21 := by
  sorry


end NUMINAMATH_CALUDE_sweet_salty_difference_l3485_348571


namespace NUMINAMATH_CALUDE_consecutive_numbers_divisible_by_2014_l3485_348550

theorem consecutive_numbers_divisible_by_2014 :
  ∃ (n : ℕ), n < 96 ∧ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 2014 = 0 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_divisible_by_2014_l3485_348550


namespace NUMINAMATH_CALUDE_school_early_arrival_l3485_348527

theorem school_early_arrival (usual_time : ℝ) (rate_ratio : ℝ) (time_saved : ℝ) : 
  usual_time = 24 →
  rate_ratio = 6 / 5 →
  time_saved = usual_time - (usual_time / rate_ratio) →
  time_saved = 4 := by
sorry

end NUMINAMATH_CALUDE_school_early_arrival_l3485_348527


namespace NUMINAMATH_CALUDE_geometric_sum_10_terms_l3485_348516

theorem geometric_sum_10_terms : 
  let a : ℚ := 3/4
  let r : ℚ := 3/4
  let n : ℕ := 10
  let S : ℚ := (a * (1 - r^n)) / (1 - r)
  S = 2971581/1048576 := by sorry

end NUMINAMATH_CALUDE_geometric_sum_10_terms_l3485_348516


namespace NUMINAMATH_CALUDE_P_equals_complement_union_l3485_348529

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p | p.2 ≠ p.1}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.2 ≠ -p.1}

-- Define set P
def P : Set (ℝ × ℝ) := {p | p.2^2 ≠ p.1^2}

-- Theorem statement
theorem P_equals_complement_union :
  P = (U \ M) ∪ (U \ N) := by sorry

end NUMINAMATH_CALUDE_P_equals_complement_union_l3485_348529


namespace NUMINAMATH_CALUDE_one_of_each_color_probability_l3485_348557

/-- Probability of selecting one marble of each color -/
theorem one_of_each_color_probability
  (total_marbles : Nat)
  (red_marbles blue_marbles green_marbles : Nat)
  (h1 : total_marbles = red_marbles + blue_marbles + green_marbles)
  (h2 : red_marbles = 3)
  (h3 : blue_marbles = 3)
  (h4 : green_marbles = 2)
  (h5 : total_marbles = 8) :
  (red_marbles * blue_marbles * green_marbles : Rat) /
  (Nat.choose total_marbles 3 : Rat) = 9 / 28 := by
  sorry

end NUMINAMATH_CALUDE_one_of_each_color_probability_l3485_348557


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l3485_348574

/-- Represents a right regular prism with external diagonal lengths a, b, and c --/
structure RightRegularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- Theorem stating that {3, 4, 6} cannot be the lengths of external diagonals of a right regular prism --/
theorem invalid_external_diagonals (p : RightRegularPrism) :
  p.a = 3 ∧ p.b = 4 ∧ p.c = 6 → False := by
  sorry

#check invalid_external_diagonals

end NUMINAMATH_CALUDE_invalid_external_diagonals_l3485_348574


namespace NUMINAMATH_CALUDE_function_composition_equality_l3485_348507

/-- Given f(x) = x/3 + 4 and g(x) = 7 - x, if f(g(a)) = 6, then a = 1 -/
theorem function_composition_equality (f g : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x / 3 + 4)
  (hg : ∀ x, g x = 7 - x)
  (h : f (g a) = 6) : 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_function_composition_equality_l3485_348507


namespace NUMINAMATH_CALUDE_smallest_number_property_l3485_348599

/-- The smallest positive integer that is not prime, not a square, and has no prime factor less than 60 -/
def smallest_number : ℕ := 290977

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- A function that checks if a number is a perfect square -/
def is_square (n : ℕ) : Prop := sorry

/-- A function that returns the smallest prime factor of a number -/
def smallest_prime_factor (n : ℕ) : ℕ := sorry

theorem smallest_number_property : 
  ¬ is_prime smallest_number ∧ 
  ¬ is_square smallest_number ∧ 
  smallest_prime_factor smallest_number > 59 ∧
  ∀ m : ℕ, m < smallest_number → 
    is_prime m ∨ is_square m ∨ smallest_prime_factor m ≤ 59 := by sorry

end NUMINAMATH_CALUDE_smallest_number_property_l3485_348599


namespace NUMINAMATH_CALUDE_city_population_ratio_l3485_348546

theorem city_population_ratio (X Y Z : ℕ) (hY : Y = 2 * Z) (hX : X = 16 * Z) :
  X / Y = 8 := by
  sorry

end NUMINAMATH_CALUDE_city_population_ratio_l3485_348546


namespace NUMINAMATH_CALUDE_average_chapters_per_book_l3485_348517

theorem average_chapters_per_book (total_chapters : Real) (total_books : Real) 
  (h1 : total_chapters = 17.0) 
  (h2 : total_books = 4.0) :
  total_chapters / total_books = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_average_chapters_per_book_l3485_348517


namespace NUMINAMATH_CALUDE_flour_weight_qualified_l3485_348524

def nominal_weight : ℝ := 25
def tolerance : ℝ := 0.25
def flour_weight : ℝ := 24.80

theorem flour_weight_qualified :
  flour_weight ≥ nominal_weight - tolerance ∧
  flour_weight ≤ nominal_weight + tolerance := by
  sorry

end NUMINAMATH_CALUDE_flour_weight_qualified_l3485_348524


namespace NUMINAMATH_CALUDE_cos_225_degrees_l3485_348596

theorem cos_225_degrees : 
  Real.cos (225 * π / 180) = -1 / Real.sqrt 2 := by
  have cos_addition : ∀ θ, Real.cos (π + θ) = -Real.cos θ := sorry
  have cos_45_degrees : Real.cos (45 * π / 180) = 1 / Real.sqrt 2 := sorry
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l3485_348596


namespace NUMINAMATH_CALUDE_simplify_expression_l3485_348575

theorem simplify_expression (x y : ℝ) : 5 * x - (x - 2 * y) = 4 * x + 2 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3485_348575


namespace NUMINAMATH_CALUDE_symmetry_of_sum_and_product_l3485_348515

-- Define a property for function symmetry about a point
def SymmetricAbout (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (a + x) + f (a - x) = 2 * b

-- Theorem statement
theorem symmetry_of_sum_and_product 
  (f g : ℝ → ℝ) (a b : ℝ) 
  (hf : SymmetricAbout f a b) (hg : SymmetricAbout g a b) :
  (SymmetricAbout (fun x ↦ f x + g x) a (2 * b)) ∧
  (∃ f g : ℝ → ℝ, SymmetricAbout f 0 0 ∧ SymmetricAbout g 0 0 ∧
    ¬∃ c d : ℝ, SymmetricAbout (fun x ↦ f x * g x) c d) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_sum_and_product_l3485_348515


namespace NUMINAMATH_CALUDE_random_phenomena_l3485_348521

-- Define a type for phenomena
inductive Phenomenon
| TrafficCount
| IntegerSuccessor
| ShellFiring
| ProductInspection

-- Define a predicate for random phenomena
def isRandom (p : Phenomenon) : Prop :=
  match p with
  | Phenomenon.TrafficCount => true
  | Phenomenon.IntegerSuccessor => false
  | Phenomenon.ShellFiring => true
  | Phenomenon.ProductInspection => true

-- Theorem statement
theorem random_phenomena :
  (isRandom Phenomenon.TrafficCount) ∧
  (¬isRandom Phenomenon.IntegerSuccessor) ∧
  (isRandom Phenomenon.ShellFiring) ∧
  (isRandom Phenomenon.ProductInspection) :=
by sorry

end NUMINAMATH_CALUDE_random_phenomena_l3485_348521


namespace NUMINAMATH_CALUDE_magnitude_of_c_for_four_distinct_roots_l3485_348568

-- Define the polynomial Q(x)
def Q (c : ℂ) (x : ℂ) : ℂ := (x^2 - 3*x + 3) * (x^2 - c*x + 9) * (x^2 - 6*x + 18)

-- Theorem statement
theorem magnitude_of_c_for_four_distinct_roots (c : ℂ) :
  (∃ (s : Finset ℂ), s.card = 4 ∧ (∀ x ∈ s, Q c x = 0) ∧ (∀ x, Q c x = 0 → x ∈ s)) →
  Complex.abs c = Real.sqrt 35.25 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_c_for_four_distinct_roots_l3485_348568


namespace NUMINAMATH_CALUDE_special_heptagon_perturbation_l3485_348544

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A heptagon represented by its vertices -/
structure Heptagon :=
  (vertices : Fin 7 → Point)

/-- Predicate to check if a heptagon is convex -/
def is_convex (h : Heptagon) : Prop := sorry

/-- Predicate to check if three lines intersect at a single point -/
def intersect_at_point (l1 l2 l3 : Point × Point) (p : Point) : Prop := sorry

/-- Predicate to check if a heptagon is special -/
def is_special (h : Heptagon) : Prop :=
  ∃ (i j k : Fin 7) (p : Point),
    i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    intersect_at_point
      (h.vertices i, h.vertices ((i + 3) % 7))
      (h.vertices j, h.vertices ((j + 3) % 7))
      (h.vertices k, h.vertices ((k + 3) % 7))
      p

/-- Definition of a small perturbation -/
def small_perturbation (h1 h2 : Heptagon) (ε : ℝ) : Prop :=
  ∃ (i : Fin 7),
    ∀ (j : Fin 7),
      if i = j then
        (h1.vertices j).x - ε < (h2.vertices j).x ∧ (h2.vertices j).x < (h1.vertices j).x + ε ∧
        (h1.vertices j).y - ε < (h2.vertices j).y ∧ (h2.vertices j).y < (h1.vertices j).y + ε
      else
        h1.vertices j = h2.vertices j

/-- The main theorem -/
theorem special_heptagon_perturbation (h : Heptagon) (hconv : is_convex h) (hspec : is_special h) :
  ∃ (h' : Heptagon) (ε : ℝ), ε > 0 ∧ small_perturbation h h' ε ∧ is_convex h' ∧ ¬is_special h' :=
sorry

end NUMINAMATH_CALUDE_special_heptagon_perturbation_l3485_348544


namespace NUMINAMATH_CALUDE_line_separates_points_l3485_348549

/-- Given that the origin (0,0) and the point (1,1) are on opposite sides of the line x+y=a,
    prove that the range of values for a is 0 < a < 2. -/
theorem line_separates_points (a : ℝ) : 
  (0 + 0 - a) * (1 + 1 - a) < 0 → 0 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_line_separates_points_l3485_348549


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l3485_348503

/-- Given a circle with radius 6 cm tangent to three sides of a rectangle,
    and the area of the rectangle being three times the area of the circle,
    prove that the length of the longer side of the rectangle is 9π cm. -/
theorem rectangle_longer_side (circle_radius : ℝ) (rectangle_area : ℝ) (circle_area : ℝ)
  (h1 : circle_radius = 6)
  (h2 : rectangle_area = 3 * circle_area)
  (h3 : circle_area = Real.pi * circle_radius ^ 2)
  (h4 : rectangle_area = 2 * circle_radius * longer_side) :
  longer_side = 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l3485_348503


namespace NUMINAMATH_CALUDE_radical_subtraction_l3485_348597

theorem radical_subtraction : (5 / Real.sqrt 2) - Real.sqrt (1 / 2) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_radical_subtraction_l3485_348597


namespace NUMINAMATH_CALUDE_limit_two_x_sin_x_over_one_minus_cos_x_l3485_348514

/-- The limit of (2x sin x) / (1 - cos x) as x approaches 0 is equal to 4 -/
theorem limit_two_x_sin_x_over_one_minus_cos_x : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → |((2 * x * Real.sin x) / (1 - Real.cos x)) - 4| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_two_x_sin_x_over_one_minus_cos_x_l3485_348514


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l3485_348508

-- Define the trapezoid EFGH
structure Trapezoid :=
  (EF : ℝ) (GH : ℝ) (height : ℝ)

-- Define the properties of the trapezoid
def isIsoscelesTrapezoid (t : Trapezoid) : Prop :=
  t.EF = t.GH

-- Theorem statement
theorem trapezoid_perimeter 
  (t : Trapezoid) 
  (h1 : isIsoscelesTrapezoid t) 
  (h2 : t.height = 5) 
  (h3 : t.GH = 10) 
  (h4 : t.EF = 4) : 
  ∃ (perimeter : ℝ), perimeter = 14 + 2 * Real.sqrt 34 :=
by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l3485_348508


namespace NUMINAMATH_CALUDE_fraction_equality_l3485_348584

theorem fraction_equality (x y z : ℝ) (h : (x - y) / (z - y) = -10) :
  (x - z) / (y - z) = 11 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3485_348584


namespace NUMINAMATH_CALUDE_replaced_person_weight_l3485_348518

theorem replaced_person_weight 
  (n : ℕ) 
  (avg_increase : ℝ) 
  (new_person_weight : ℝ) 
  (h1 : n = 10) 
  (h2 : avg_increase = 3.2) 
  (h3 : new_person_weight = 97) : 
  new_person_weight - n * avg_increase = 65 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l3485_348518


namespace NUMINAMATH_CALUDE_pen_profit_calculation_l3485_348520

/-- Calculates the profit from selling pens given the purchase quantity, cost rate, and selling rate. -/
def calculate_profit (purchase_quantity : ℕ) (cost_rate : ℚ × ℚ) (selling_rate : ℚ × ℚ) : ℚ :=
  let cost_per_pen := cost_rate.2 / cost_rate.1
  let total_cost := cost_per_pen * purchase_quantity
  let selling_price_per_pen := selling_rate.2 / selling_rate.1
  let total_revenue := selling_price_per_pen * purchase_quantity
  total_revenue - total_cost

/-- The profit from selling 1200 pens, bought at 4 for $3 and sold at 3 for $2, is -$96. -/
theorem pen_profit_calculation :
  calculate_profit 1200 (4, 3) (3, 2) = -96 := by
  sorry

end NUMINAMATH_CALUDE_pen_profit_calculation_l3485_348520


namespace NUMINAMATH_CALUDE_company_employees_ratio_salary_increase_impact_l3485_348559

theorem company_employees_ratio (M F N : ℕ) : 
  (M : ℚ) / F = 7 / 8 ∧ 
  (N : ℚ) / F = 6 / 8 ∧ 
  ((M + 5 : ℚ) / F = 8 / 9 ∧ (N + 3 : ℚ) / F = 7 / 9) → 
  M = 315 ∧ F = 360 ∧ N = 270 := by
  sorry

theorem salary_increase_impact (T : ℚ) :
  T > 0 → T * (110 / 100) - T = T / 10 := by
  sorry

end NUMINAMATH_CALUDE_company_employees_ratio_salary_increase_impact_l3485_348559


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3485_348578

theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a = (1, 2) →
  b = (-2, m) →
  (∃ (k : ℝ), a = k • b) →
  m = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3485_348578


namespace NUMINAMATH_CALUDE_max_value_of_f_l3485_348513

def f (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem max_value_of_f :
  ∃ (m : ℝ), m = 9 ∧ ∀ (x : ℝ), x ∈ Set.Icc (-2) 2 → f x ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3485_348513


namespace NUMINAMATH_CALUDE_sexagenary_cycle_after_80_years_l3485_348555

/-- Represents the Chinese sexagenary cycle -/
structure SexagenaryCycle where
  heavenly_stems : Fin 10
  earthly_branches : Fin 12

/-- Advances the cycle by n years -/
def advance_cycle (cycle : SexagenaryCycle) (n : ℕ) : SexagenaryCycle :=
  { heavenly_stems := (cycle.heavenly_stems + n) % 10,
    earthly_branches := (cycle.earthly_branches + n) % 12 }

/-- Represents the specific combinations in the problem -/
def ji_chou : SexagenaryCycle := ⟨5, 1⟩  -- 己丑
def ji_you : SexagenaryCycle := ⟨5, 9⟩   -- 己酉

/-- The main theorem to prove -/
theorem sexagenary_cycle_after_80_years :
  ∀ (year : ℕ), advance_cycle ji_chou 80 = ji_you := by
  sorry

end NUMINAMATH_CALUDE_sexagenary_cycle_after_80_years_l3485_348555


namespace NUMINAMATH_CALUDE_unique_prime_exponent_l3485_348552

theorem unique_prime_exponent : ∃! (n : ℕ), Nat.Prime (3^(2*n) - 2^n) :=
  sorry

end NUMINAMATH_CALUDE_unique_prime_exponent_l3485_348552


namespace NUMINAMATH_CALUDE_investment_rate_proof_l3485_348506

theorem investment_rate_proof (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ)
  (first_rate : ℝ) (second_rate : ℝ) (desired_income : ℝ) :
  total_investment = 12000 ∧
  first_investment = 5000 ∧
  second_investment = 4000 ∧
  first_rate = 0.05 ∧
  second_rate = 0.035 ∧
  desired_income = 600 →
  ∃ (remaining_rate : ℝ),
    remaining_rate = 0.07 ∧
    (total_investment - first_investment - second_investment) * remaining_rate +
    first_investment * first_rate + second_investment * second_rate = desired_income :=
by sorry

end NUMINAMATH_CALUDE_investment_rate_proof_l3485_348506


namespace NUMINAMATH_CALUDE_second_train_speed_l3485_348510

/-- Calculates the speed of the second train given the parameters of two trains crossing each other. -/
theorem second_train_speed
  (length1 : ℝ)
  (speed1 : ℝ)
  (length2 : ℝ)
  (time_to_cross : ℝ)
  (h1 : length1 = 270)
  (h2 : speed1 = 120)
  (h3 : length2 = 230.04)
  (h4 : time_to_cross = 9)
  : ∃ (speed2 : ℝ), speed2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_second_train_speed_l3485_348510


namespace NUMINAMATH_CALUDE_unique_matches_exist_l3485_348572

/-- A graph with 20 vertices and 14 edges where each vertex has degree at least 1 -/
structure TennisGraph where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  edge_count : edges.card = 14
  degree_at_least_one : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≥ 1

/-- A subgraph where each vertex has degree at most 1 -/
def UniqueMatchesSubgraph (G : TennisGraph) :=
  { edges : Finset (Fin 20 × Fin 20) //
    edges ⊆ G.edges ∧
    ∀ v ∈ G.vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card ≤ 1 }

/-- The main theorem -/
theorem unique_matches_exist (G : TennisGraph) :
  ∃ (S : UniqueMatchesSubgraph G), S.val.card ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_matches_exist_l3485_348572


namespace NUMINAMATH_CALUDE_minimum_school_payment_l3485_348580

/-- The minimum amount a school should pay for cinema tickets -/
theorem minimum_school_payment
  (individual_price : ℝ)
  (group_price : ℝ)
  (group_size : ℕ)
  (student_discount : ℝ)
  (num_students : ℕ)
  (h1 : individual_price = 6)
  (h2 : group_price = 40)
  (h3 : group_size = 10)
  (h4 : student_discount = 0.1)
  (h5 : num_students = 1258) :
  ∃ (min_payment : ℝ),
    min_payment = 4536 ∧
    min_payment ≤ (↑(num_students / group_size) * group_price * (1 - student_discount)) + 
                  (↑(num_students % group_size) * individual_price * (1 - student_discount)) :=
by
  sorry

#eval 1258 / 10 * 40 * 0.9

end NUMINAMATH_CALUDE_minimum_school_payment_l3485_348580


namespace NUMINAMATH_CALUDE_solve_property_damage_l3485_348526

def property_damage_problem (medical_bills : ℝ) (carl_payment_percentage : ℝ) (carl_payment : ℝ) : Prop :=
  let total_cost := carl_payment / carl_payment_percentage
  let property_damage := total_cost - medical_bills
  property_damage = 40000

theorem solve_property_damage :
  property_damage_problem 70000 0.2 22000 := by
  sorry

end NUMINAMATH_CALUDE_solve_property_damage_l3485_348526


namespace NUMINAMATH_CALUDE_intersection_points_count_l3485_348554

/-- The number of intersection points between y = |3x + 6| and y = -|4x - 3| -/
theorem intersection_points_count : ∃! p : ℝ × ℝ, 
  (|3 * p.1 + 6| = p.2) ∧ (-|4 * p.1 - 3| = p.2) := by sorry

end NUMINAMATH_CALUDE_intersection_points_count_l3485_348554


namespace NUMINAMATH_CALUDE_tree_height_proof_l3485_348530

/-- Given a tree that is currently 180 inches tall and 50% taller than its original height,
    prove that its original height was 10 feet. -/
theorem tree_height_proof :
  let current_height_inches : ℝ := 180
  let growth_factor : ℝ := 1.5
  let inches_per_foot : ℝ := 12
  current_height_inches / growth_factor / inches_per_foot = 10
  := by sorry

end NUMINAMATH_CALUDE_tree_height_proof_l3485_348530


namespace NUMINAMATH_CALUDE_rational_sqrt_fraction_l3485_348509

theorem rational_sqrt_fraction (r q n : ℚ) 
  (h : 1 / (r + q * n) + 1 / (q + r * n) = 1 / (r + q)) :
  ∃ (m : ℚ), (n - 3) / (n + 1) = m^2 := by
sorry

end NUMINAMATH_CALUDE_rational_sqrt_fraction_l3485_348509


namespace NUMINAMATH_CALUDE_halfway_point_between_fractions_l3485_348535

theorem halfway_point_between_fractions :
  let a := (1 : ℚ) / 7
  let b := (1 : ℚ) / 9
  let midpoint := (a + b) / 2
  midpoint = 8 / 63 := by sorry

end NUMINAMATH_CALUDE_halfway_point_between_fractions_l3485_348535


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_31_l3485_348528

theorem modular_inverse_of_3_mod_31 :
  ∃ x : ℕ, x ≤ 30 ∧ (3 * x) % 31 = 1 :=
by
  use 21
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_31_l3485_348528


namespace NUMINAMATH_CALUDE_square_area_ratio_l3485_348533

theorem square_area_ratio (y : ℝ) (h : y > 0) :
  (y ^ 2) / ((3 * y) ^ 2) = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3485_348533


namespace NUMINAMATH_CALUDE_function_inequality_iff_a_geq_half_l3485_348525

/-- Given a function f(x) = ln x - a(x - 1), where a is a real number and x ≥ 1,
    prove that f(x) ≤ (ln x) / (x + 1) if and only if a ≥ 1/2 -/
theorem function_inequality_iff_a_geq_half (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → (Real.log x - a * (x - 1)) ≤ (Real.log x) / (x + 1)) ↔ a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_iff_a_geq_half_l3485_348525


namespace NUMINAMATH_CALUDE_cross_to_square_l3485_348547

/-- Represents a cross made of unit squares -/
structure Cross :=
  (num_squares : ℕ)
  (side_length : ℝ)

/-- Represents a square -/
structure Square :=
  (side_length : ℝ)

/-- The area of a square -/
def Square.area (s : Square) : ℝ := s.side_length ^ 2

/-- The cross in the problem -/
def problem_cross : Cross := { num_squares := 5, side_length := 1 }

/-- The theorem to be proved -/
theorem cross_to_square (c : Cross) (s : Square) 
  (h1 : c = problem_cross) 
  (h2 : s.side_length = Real.sqrt 5) : 
  s.area = c.num_squares * c.side_length ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_cross_to_square_l3485_348547


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3485_348561

theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) (hypotenuse : ℝ) : 
  leg = 15 →
  angle = 45 →
  hypotenuse = leg * Real.sqrt 2 →
  hypotenuse = 15 * Real.sqrt 2 :=
by
  sorry

#check right_triangle_hypotenuse

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3485_348561


namespace NUMINAMATH_CALUDE_vector_equality_implies_x_value_l3485_348576

/-- Given that the vector (x+3, x^2-3x-4) is equal to (2, 0), prove that x = -1 -/
theorem vector_equality_implies_x_value : 
  ∀ x : ℝ, (x + 3 = 2 ∧ x^2 - 3*x - 4 = 0) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_implies_x_value_l3485_348576


namespace NUMINAMATH_CALUDE_real_world_length_l3485_348536

/-- Represents the scale factor of the model -/
def scale_factor : ℝ := 50

/-- Represents the length of the line segment in the model (in cm) -/
def model_length : ℝ := 7.5

/-- Theorem stating that the real-world length represented by the model line segment is 375 meters -/
theorem real_world_length : model_length * scale_factor = 375 := by
  sorry

end NUMINAMATH_CALUDE_real_world_length_l3485_348536


namespace NUMINAMATH_CALUDE_no_ten_digit_square_plus_three_with_distinct_digits_l3485_348573

theorem no_ten_digit_square_plus_three_with_distinct_digits :
  ¬ ∃ (n : ℕ), 
    (10^9 ≤ n^2 + 3) ∧ 
    (n^2 + 3 < 10^10) ∧ 
    (∀ (i j : Fin 10), i ≠ j → 
      (((n^2 + 3) / 10^i.val) % 10 ≠ ((n^2 + 3) / 10^j.val) % 10)) :=
sorry

end NUMINAMATH_CALUDE_no_ten_digit_square_plus_three_with_distinct_digits_l3485_348573


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l3485_348556

/-- A geometric sequence with common ratio 4 and sum of first three terms equal to 21 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 4 * a n) ∧ (a 1 + a 2 + a 3 = 21)

/-- The general term formula for the geometric sequence -/
theorem geometric_sequence_formula (a : ℕ → ℝ) (h : GeometricSequence a) :
  ∀ n : ℕ, a n = 4^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l3485_348556


namespace NUMINAMATH_CALUDE_sock_pair_count_l3485_348579

/-- The number of ways to choose a pair of socks with different colors -/
def different_color_pairs (white brown blue : ℕ) : ℕ :=
  white * brown + brown * blue + white * blue

/-- Theorem: The number of ways to choose a pair of socks with different colors
    from 4 white, 4 brown, and 2 blue socks is 32 -/
theorem sock_pair_count :
  different_color_pairs 4 4 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l3485_348579


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3485_348587

def g (p q r s : ℝ) (x : ℂ) : ℂ :=
  x^4 + p*x^3 + q*x^2 + r*x + s

theorem sum_of_coefficients 
  (p q r s : ℝ) 
  (h1 : g p q r s (3*I) = 0)
  (h2 : g p q r s (1 + 2*I) = 0) : 
  p + q + r + s = -41 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3485_348587


namespace NUMINAMATH_CALUDE_total_soccer_balls_l3485_348511

/-- Given the following conditions:
  - The school purchased 10 boxes of soccer balls
  - Each box contains 8 packages
  - Each package has 13 soccer balls
  Prove that the total number of soccer balls purchased is 1040 -/
theorem total_soccer_balls (num_boxes : ℕ) (packages_per_box : ℕ) (balls_per_package : ℕ)
  (h1 : num_boxes = 10)
  (h2 : packages_per_box = 8)
  (h3 : balls_per_package = 13) :
  num_boxes * packages_per_box * balls_per_package = 1040 := by
  sorry

end NUMINAMATH_CALUDE_total_soccer_balls_l3485_348511


namespace NUMINAMATH_CALUDE_unique_twin_prime_trio_l3485_348564

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem unique_twin_prime_trio : 
  ∀ p : ℕ, is_prime p → p > 7 → ¬(is_prime p ∧ is_prime (p + 2) ∧ is_prime (p + 4)) :=
sorry

end NUMINAMATH_CALUDE_unique_twin_prime_trio_l3485_348564


namespace NUMINAMATH_CALUDE_units_digit_of_17_to_1995_l3485_348565

theorem units_digit_of_17_to_1995 : (17 ^ 1995 : ℕ) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_17_to_1995_l3485_348565


namespace NUMINAMATH_CALUDE_last_to_first_points_l3485_348586

/-- Represents a chess tournament with initial and final states -/
structure ChessTournament where
  initial_players : ℕ
  disqualified_players : ℕ
  points_per_win : ℚ
  points_per_draw : ℚ
  points_per_loss : ℚ

/-- Calculates the total number of games in a round-robin tournament -/
def total_games (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem stating that a player who goes from last to first must have 4 points after disqualification -/
theorem last_to_first_points (t : ChessTournament) 
  (h1 : t.initial_players = 10)
  (h2 : t.disqualified_players = 2)
  (h3 : t.points_per_win = 1)
  (h4 : t.points_per_draw = 1/2)
  (h5 : t.points_per_loss = 0) :
  ∃ (initial_points final_points : ℚ),
    initial_points < (total_games t.initial_players : ℚ) / t.initial_players ∧
    final_points > (total_games (t.initial_players - t.disqualified_players) : ℚ) / (t.initial_players - t.disqualified_players) ∧
    final_points = 4 :=
  sorry

end NUMINAMATH_CALUDE_last_to_first_points_l3485_348586


namespace NUMINAMATH_CALUDE_power_function_increasing_interval_l3485_348563

/-- Given a power function f(x) = x^a where a is a real number,
    and f(2) = √2, prove that the increasing interval of f is [0, +∞) -/
theorem power_function_increasing_interval
  (f : ℝ → ℝ)
  (a : ℝ)
  (h1 : ∀ x > 0, f x = x ^ a)
  (h2 : f 2 = Real.sqrt 2) :
  ∀ x y, 0 ≤ x ∧ x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_power_function_increasing_interval_l3485_348563


namespace NUMINAMATH_CALUDE_ball_count_theorem_l3485_348500

/-- Represents the number of balls of each color in a jar -/
structure BallCount where
  white : ℕ
  red : ℕ
  blue : ℕ

/-- Checks if the given ball count satisfies the ratio 4:3:2 for white:red:blue -/
def satisfiesRatio (bc : BallCount) : Prop :=
  4 * bc.red = 3 * bc.white ∧ 4 * bc.blue = 2 * bc.white

theorem ball_count_theorem (bc : BallCount) 
  (ratio_satisfied : satisfiesRatio bc) 
  (white_count : bc.white = 16) : 
  bc.red = 12 ∧ bc.blue = 8 := by
  sorry

#check ball_count_theorem

end NUMINAMATH_CALUDE_ball_count_theorem_l3485_348500


namespace NUMINAMATH_CALUDE_red_tint_percentage_after_modification_l3485_348537

/-- Calculates the percentage of red tint in a modified paint mixture -/
theorem red_tint_percentage_after_modification
  (initial_volume : ℝ)
  (initial_red_tint_percentage : ℝ)
  (added_red_tint : ℝ)
  (h_initial_volume : initial_volume = 40)
  (h_initial_red_tint_percentage : initial_red_tint_percentage = 35)
  (h_added_red_tint : added_red_tint = 10) :
  let initial_red_tint := initial_volume * initial_red_tint_percentage / 100
  let final_red_tint := initial_red_tint + added_red_tint
  let final_volume := initial_volume + added_red_tint
  final_red_tint / final_volume * 100 = 48 := by
  sorry

end NUMINAMATH_CALUDE_red_tint_percentage_after_modification_l3485_348537


namespace NUMINAMATH_CALUDE_sqrt_square_negative_two_l3485_348569

theorem sqrt_square_negative_two : Real.sqrt ((-2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_negative_two_l3485_348569


namespace NUMINAMATH_CALUDE_hamburger_combinations_l3485_348540

/-- The number of available condiments -/
def num_condiments : ℕ := 8

/-- The number of choices for meat patties -/
def num_patty_choices : ℕ := 4

/-- The total number of hamburger combinations -/
def total_combinations : ℕ := 2^num_condiments * num_patty_choices

theorem hamburger_combinations :
  total_combinations = 1024 :=
by sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l3485_348540


namespace NUMINAMATH_CALUDE_remainder_theorem_l3485_348581

theorem remainder_theorem : ∃ q : ℕ, 3^303 + 303 = (3^101 + 3^51 + 1) * q + 303 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3485_348581


namespace NUMINAMATH_CALUDE_function_inequality_l3485_348562

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x y, x < y ∧ y ≤ 2 → f x < f y) 
  (h2 : ∀ x, f (-x + 2) = f (x + 2)) : 
  f (-1) < f 3 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_l3485_348562


namespace NUMINAMATH_CALUDE_area_of_triangle_formed_by_segment_l3485_348589

structure Rectangle where
  width : ℝ
  height : ℝ

structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

def Point := ℝ × ℝ

def Segment (p1 p2 : Point) := {p : Point | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (t * p1.1 + (1 - t) * p2.1, t * p1.2 + (1 - t) * p2.2)}

theorem area_of_triangle_formed_by_segment (rect : Rectangle) (tri : IsoscelesTriangle) 
  (h1 : rect.width = 15 ∧ rect.height = 10)
  (h2 : tri.base = 10 ∧ tri.height = 10)
  (h3 : (15, 0) = (rect.width, 0))
  (h4 : Segment (0, rect.height) (20, 10) ∩ Segment (15, 0) (25, 0) = {(15, 10)}) :
  (1 / 2) * rect.width * rect.height = 75 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_formed_by_segment_l3485_348589


namespace NUMINAMATH_CALUDE_final_cell_population_l3485_348558

/-- Represents the cell population growth over time -/
def cell_population (initial_cells : ℕ) (split_factor : ℕ) (days : ℕ) : ℕ :=
  initial_cells * split_factor ^ (days / 3)

/-- Theorem: Given the conditions, the final cell population after 9 days is 18 -/
theorem final_cell_population :
  cell_population 2 3 9 = 18 := by
  sorry

#eval cell_population 2 3 9

end NUMINAMATH_CALUDE_final_cell_population_l3485_348558


namespace NUMINAMATH_CALUDE_thirty_day_month_equal_sundays_tuesdays_l3485_348582

/-- Represents the days of the week -/
inductive DayOfWeek
| sunday
| monday
| tuesday
| wednesday
| thursday
| friday
| saturday

/-- Counts the occurrences of a specific day in a 30-day month starting from a given day -/
def countDay (startDay : DayOfWeek) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Checks if Sundays and Tuesdays are equal in a 30-day month starting from a given day -/
def hasSameSundaysAndTuesdays (startDay : DayOfWeek) : Bool :=
  countDay startDay DayOfWeek.sunday = countDay startDay DayOfWeek.tuesday

/-- Counts the number of possible starting days for a 30-day month with equal Sundays and Tuesdays -/
def countValidStartDays : Nat :=
  sorry

theorem thirty_day_month_equal_sundays_tuesdays :
  countValidStartDays = 3 :=
sorry

end NUMINAMATH_CALUDE_thirty_day_month_equal_sundays_tuesdays_l3485_348582


namespace NUMINAMATH_CALUDE_age_difference_l3485_348501

/-- The difference in years between individuals a and c -/
def R (a c : ℕ) : ℕ := a - c

/-- The age of an individual after 5 years -/
def L (x : ℕ) : ℕ := x + 5

theorem age_difference (a b c d : ℕ) :
  (L a + L b = L b + L c + 10) →
  (c + d = a + d - 12) →
  R a c = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3485_348501


namespace NUMINAMATH_CALUDE_f_min_max_l3485_348548

-- Define the function
def f (x : ℝ) : ℝ := 1 + 3*x - x^3

-- State the theorem
theorem f_min_max : 
  (∃ x : ℝ, f x = -1) ∧ 
  (∀ x : ℝ, f x ≥ -1) ∧ 
  (∃ x : ℝ, f x = 3) ∧ 
  (∀ x : ℝ, f x ≤ 3) := by sorry

end NUMINAMATH_CALUDE_f_min_max_l3485_348548


namespace NUMINAMATH_CALUDE_fraction_exceeding_by_30_l3485_348543

theorem fraction_exceeding_by_30 (x : ℚ) : 
  48 = 48 * x + 30 → x = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_exceeding_by_30_l3485_348543


namespace NUMINAMATH_CALUDE_paving_cost_l3485_348538

/-- The cost of paving a rectangular floor given its dimensions and rate per square meter. -/
theorem paving_cost (length width rate : ℝ) : length = 5.5 → width = 4 → rate = 700 → length * width * rate = 15400 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l3485_348538


namespace NUMINAMATH_CALUDE_anne_solo_cleaning_time_l3485_348590

/-- Represents the time it takes Anne to clean the house alone -/
def anne_solo_time : ℝ := 12

/-- Represents Bruce's cleaning rate in houses per hour -/
noncomputable def bruce_rate : ℝ := sorry

/-- Represents Anne's cleaning rate in houses per hour -/
noncomputable def anne_rate : ℝ := sorry

/-- Bruce and Anne can clean the house in 4 hours together -/
axiom together_time : bruce_rate + anne_rate = 1 / 4

/-- If Anne's speed were doubled, they could clean the house in 3 hours -/
axiom double_anne_time : bruce_rate + 2 * anne_rate = 1 / 3

theorem anne_solo_cleaning_time : 
  1 / anne_rate = anne_solo_time :=
sorry

end NUMINAMATH_CALUDE_anne_solo_cleaning_time_l3485_348590


namespace NUMINAMATH_CALUDE_circle_equation_from_parabola_focus_l3485_348532

/-- The equation of a circle with its center at the focus of the parabola y² = 4x
    and passing through the origin is x² + y² - 2x = 0. -/
theorem circle_equation_from_parabola_focus (x y : ℝ) : 
  (∃ (h : ℝ), y^2 = 4*x ∧ h = 1) →  -- Focus of parabola y² = 4x is at (1, 0)
  (0^2 + 0^2 = (x - 1)^2 + y^2) →  -- Circle passes through origin (0, 0)
  (x^2 + y^2 - 2*x = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_parabola_focus_l3485_348532


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_6_l3485_348553

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

theorem five_digit_multiple_of_6 (d : ℕ) :
  d < 10 →
  is_multiple_of_6 (47690 + d) →
  d = 4 ∨ d = 8 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_6_l3485_348553


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l3485_348505

theorem z_in_first_quadrant (z : ℂ) (h : z * (1 - 3*I) = 5 - 5*I) : 
  0 < z.re ∧ 0 < z.im :=
sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l3485_348505


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3485_348583

open Set

def U : Set Nat := {0, 1, 2, 3, 4}
def M : Set Nat := {0, 1, 2}
def N : Set Nat := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3485_348583


namespace NUMINAMATH_CALUDE_uncovered_area_calculation_l3485_348577

theorem uncovered_area_calculation (large_square_side : ℝ) (small_square_side : ℝ) :
  large_square_side = 10 →
  small_square_side = 4 →
  large_square_side^2 - 2 * small_square_side^2 = 68 :=
by sorry

end NUMINAMATH_CALUDE_uncovered_area_calculation_l3485_348577


namespace NUMINAMATH_CALUDE_length_of_cd_l3485_348534

/-- Represents a point that divides a line segment in a given ratio -/
structure DividingPoint where
  ratio_left : ℚ
  ratio_right : ℚ

/-- Represents a line segment divided by two points -/
structure DividedSegment where
  length : ℝ
  point1 : DividingPoint
  point2 : DividingPoint
  distance_between_points : ℝ

/-- Theorem stating the length of CD given the conditions -/
theorem length_of_cd (cd : DividedSegment) : 
  cd.point1.ratio_left = 3 ∧ 
  cd.point1.ratio_right = 5 ∧ 
  cd.point2.ratio_left = 4 ∧ 
  cd.point2.ratio_right = 7 ∧ 
  cd.distance_between_points = 3 → 
  cd.length = 264 := by
  sorry

end NUMINAMATH_CALUDE_length_of_cd_l3485_348534


namespace NUMINAMATH_CALUDE_sum_of_real_solutions_l3485_348593

theorem sum_of_real_solutions (a : ℝ) (h : a > 2) :
  let f := fun x : ℝ => Real.sqrt (a - Real.sqrt (a - x)) - x
  let sum_solutions := (Real.sqrt (4 * a - 3) - 1) / 2
  (∃ x : ℝ, f x = 0) ∧ (∀ x : ℝ, f x = 0 → x ≤ sum_solutions) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_real_solutions_l3485_348593


namespace NUMINAMATH_CALUDE_tim_nickels_l3485_348523

/-- The number of nickels Tim got for shining shoes -/
def nickels : ℕ := sorry

/-- The number of dimes Tim got for shining shoes -/
def dimes_shining : ℕ := 13

/-- The number of dimes Tim found in his tip jar -/
def dimes_tip : ℕ := 7

/-- The number of half-dollars Tim found in his tip jar -/
def half_dollars : ℕ := 9

/-- The total amount Tim got in dollars -/
def total_amount : ℚ := 665 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The value of a half-dollar in dollars -/
def half_dollar_value : ℚ := 50 / 100

theorem tim_nickels :
  nickels * nickel_value + 
  dimes_shining * dime_value + 
  dimes_tip * dime_value + 
  half_dollars * half_dollar_value = total_amount ∧
  nickels = 3 := by sorry

end NUMINAMATH_CALUDE_tim_nickels_l3485_348523


namespace NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l3485_348504

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_ineq : 21 * a * b + 2 * b * c + 8 * c * a ≤ 12) :
  1 / a + 2 / b + 3 / c ≥ 15 / 2 := by
  sorry

theorem lower_bound_achievable :
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧
  21 * a * b + 2 * b * c + 8 * c * a ≤ 12 ∧
  1 / a + 2 / b + 3 / c = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l3485_348504


namespace NUMINAMATH_CALUDE_arrangements_combinations_ratio_l3485_348512

/-- Number of arrangements of n items taken r at a time -/
def A (n : ℕ) (r : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial (n - r))

/-- Number of combinations of n items taken r at a time -/
def C (n : ℕ) (r : ℕ) : ℚ := (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

theorem arrangements_combinations_ratio : (A 7 2) / (C 10 2) = 14 / 15 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_combinations_ratio_l3485_348512


namespace NUMINAMATH_CALUDE_g_1989_of_5_eq_5_l3485_348502

def g (x : ℚ) : ℚ := (2 - x) / (1 + 2 * x)

def g_n : ℕ → (ℚ → ℚ)
| 0 => λ x => x
| n + 1 => λ x => g (g_n n x)

theorem g_1989_of_5_eq_5 : g_n 1989 5 = 5 := by sorry

end NUMINAMATH_CALUDE_g_1989_of_5_eq_5_l3485_348502


namespace NUMINAMATH_CALUDE_distance_to_place_l3485_348551

/-- Calculates the distance to a place given rowing speed, current velocity, and round trip time -/
theorem distance_to_place (rowing_speed current_velocity : ℝ) (round_trip_time : ℝ) : 
  rowing_speed = 5 → 
  current_velocity = 1 → 
  round_trip_time = 1 → 
  ∃ (distance : ℝ), distance = 2.4 ∧ 
    round_trip_time = distance / (rowing_speed + current_velocity) + 
                      distance / (rowing_speed - current_velocity) :=
by
  sorry

#check distance_to_place

end NUMINAMATH_CALUDE_distance_to_place_l3485_348551


namespace NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l3485_348598

theorem quadratic_discriminant_nonnegative (a b : ℝ) :
  (∃ x : ℝ, x^2 + a*x + b ≤ 0) → a^2 - 4*b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l3485_348598


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3485_348570

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) :
  a^3 + b^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3485_348570


namespace NUMINAMATH_CALUDE_subtract_negative_term_l3485_348585

theorem subtract_negative_term (a : ℝ) :
  (4 * a^2 - 3 * a + 7) - (-6 * a) = 4 * a^2 + 3 * a + 7 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_term_l3485_348585


namespace NUMINAMATH_CALUDE_waiter_tips_fraction_l3485_348522

theorem waiter_tips_fraction (salary tips : ℝ) 
  (h : tips = 0.625 * (salary + tips)) : 
  tips / salary = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_waiter_tips_fraction_l3485_348522


namespace NUMINAMATH_CALUDE_cube_volume_in_pyramid_l3485_348560

/-- A tetrahedral pyramid with an equilateral triangular base -/
structure TetrahedralPyramid where
  base_side_length : ℝ
  lateral_faces_equilateral : Prop

/-- A cube placed inside a tetrahedral pyramid -/
structure CubeInPyramid where
  pyramid : TetrahedralPyramid
  vertex_at_centroid : Prop
  edges_touch_midpoints : Prop

/-- The volume of a cube -/
def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

theorem cube_volume_in_pyramid (c : CubeInPyramid) : 
  c.pyramid.base_side_length = 3 → cube_volume (c.pyramid.base_side_length / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_in_pyramid_l3485_348560


namespace NUMINAMATH_CALUDE_no_valid_b_exists_l3485_348595

/-- Given a point P and its symmetric point Q about the origin, 
    prove that there is no real value of b for which both points 
    satisfy the inequality 2x - by + 1 ≤ 0. -/
theorem no_valid_b_exists (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  P = (1, -2) → 
  Q.1 = -P.1 → 
  Q.2 = -P.2 → 
  ¬∃ b : ℝ, (2 * P.1 - b * P.2 + 1 ≤ 0) ∧ (2 * Q.1 - b * Q.2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_b_exists_l3485_348595


namespace NUMINAMATH_CALUDE_elder_person_age_l3485_348594

/-- Given two persons whose ages differ by 12 years, and 5 years ago the elder one was 5 times as old as the younger one, the present age of the elder person is 20 years. -/
theorem elder_person_age (y e : ℕ) : 
  e = y + 12 → 
  e - 5 = 5 * (y - 5) → 
  e = 20 := by
sorry

end NUMINAMATH_CALUDE_elder_person_age_l3485_348594
