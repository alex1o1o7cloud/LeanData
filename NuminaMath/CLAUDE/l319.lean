import Mathlib

namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l319_31992

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x + 3
  f 0 = 4 := by sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l319_31992


namespace NUMINAMATH_CALUDE_myopia_functional_relationship_l319_31996

/-- The functional relationship between the degree of myopia glasses and focal length of lenses -/
def myopia_relationship (y x : ℝ) : Prop :=
  y = 100 / x

/-- y and x are inversely proportional -/
def inverse_proportional (y x : ℝ) : Prop :=
  ∃ k : ℝ, y = k / x

theorem myopia_functional_relationship :
  ∀ y x : ℝ, 
  inverse_proportional y x → 
  (y = 400 ∧ x = 0.25) → 
  myopia_relationship y x :=
sorry

end NUMINAMATH_CALUDE_myopia_functional_relationship_l319_31996


namespace NUMINAMATH_CALUDE_lcm_problem_l319_31906

theorem lcm_problem (d n : ℕ) : 
  d > 0 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  Nat.lcm d n = 690 ∧ 
  ¬(3 ∣ n) ∧ 
  ¬(2 ∣ d) →
  n = 230 := by sorry

end NUMINAMATH_CALUDE_lcm_problem_l319_31906


namespace NUMINAMATH_CALUDE_max_value_cosine_function_l319_31945

theorem max_value_cosine_function (x : ℝ) :
  ∃ (k : ℤ), 2 * Real.cos x - 1 ≤ 2 * Real.cos (2 * k * Real.pi) - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_cosine_function_l319_31945


namespace NUMINAMATH_CALUDE_planes_intersect_necessary_not_sufficient_l319_31986

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the intersection relation between two planes
variable (planesIntersect : Plane → Plane → Prop)

-- Define the skew relation between two lines
variable (skewLines : Line → Line → Prop)

-- Define the theorem
theorem planes_intersect_necessary_not_sufficient
  (α β : Plane) (m n : Line)
  (h1 : ¬ planesIntersect α β)
  (h2 : perpendicular m α)
  (h3 : perpendicular n β) :
  (∀ α' β' m' n', planesIntersect α' β' → skewLines m' n' → perpendicular m' α' → perpendicular n' β' → True) ∧
  (∃ α' β' m' n', planesIntersect α' β' ∧ ¬ skewLines m' n' ∧ perpendicular m' α' ∧ perpendicular n' β') :=
sorry

end NUMINAMATH_CALUDE_planes_intersect_necessary_not_sufficient_l319_31986


namespace NUMINAMATH_CALUDE_smallest_positive_value_l319_31903

theorem smallest_positive_value : ∀ (a b c d e : ℝ),
  a = 12 - 4 * Real.sqrt 7 →
  b = 4 * Real.sqrt 7 - 12 →
  c = 25 - 6 * Real.sqrt 19 →
  d = 65 - 15 * Real.sqrt 17 →
  e = 15 * Real.sqrt 17 - 65 →
  (0 < a ∧ (b ≤ 0 ∨ a < b) ∧ (c ≤ 0 ∨ a < c) ∧ (d ≤ 0 ∨ a < d) ∧ (e ≤ 0 ∨ a < e)) :=
by sorry

#check smallest_positive_value

end NUMINAMATH_CALUDE_smallest_positive_value_l319_31903


namespace NUMINAMATH_CALUDE_socks_price_l319_31990

/-- Given the prices of jeans, t-shirt, and socks, where:
  1. The jeans cost twice as much as the t-shirt
  2. The t-shirt costs $10 more than the socks
  3. The jeans cost $30
  Prove that the socks cost $5 -/
theorem socks_price (jeans t_shirt socks : ℕ) 
  (h1 : jeans = 2 * t_shirt)
  (h2 : t_shirt = socks + 10)
  (h3 : jeans = 30) : 
  socks = 5 := by
sorry

end NUMINAMATH_CALUDE_socks_price_l319_31990


namespace NUMINAMATH_CALUDE_fraction_simplification_l319_31929

theorem fraction_simplification (x : ℝ) (h : x = 3) :
  (x^8 + 20*x^4 + 100) / (x^4 + 10) = 91 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l319_31929


namespace NUMINAMATH_CALUDE_weight_fluctuation_l319_31939

theorem weight_fluctuation (initial_weight : ℕ) (initial_loss : ℕ) (final_gain : ℕ) : 
  initial_weight = 99 →
  initial_loss = 12 →
  final_gain = 6 →
  initial_weight - initial_loss + 2 * initial_loss - 3 * initial_loss + final_gain = 81 := by
  sorry

end NUMINAMATH_CALUDE_weight_fluctuation_l319_31939


namespace NUMINAMATH_CALUDE_solve_comic_problem_l319_31909

def comic_problem (pages_per_comic : ℕ) (found_pages : ℕ) (total_comics : ℕ) : Prop :=
  let repaired_comics := found_pages / pages_per_comic
  let untorn_comics := total_comics - repaired_comics
  untorn_comics = 5

theorem solve_comic_problem :
  comic_problem 25 150 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_comic_problem_l319_31909


namespace NUMINAMATH_CALUDE_right_triangle_max_ratio_squared_l319_31908

theorem right_triangle_max_ratio_squared (a b x y : ℝ) : 
  0 < a → 0 < b → a ≥ b → 
  x^2 + b^2 = a^2 + y^2 → -- right triangle condition
  x + y = a → 
  x ≥ 0 → y ≥ 0 → 
  (a / b)^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_max_ratio_squared_l319_31908


namespace NUMINAMATH_CALUDE_weight_of_HClO2_l319_31905

/-- The molar mass of HClO2 in g/mol -/
def molar_mass_HClO2 : ℝ := 68.46

/-- The number of moles of HClO2 -/
def moles_HClO2 : ℝ := 6

/-- The weight of HClO2 in grams -/
def weight_HClO2 : ℝ := molar_mass_HClO2 * moles_HClO2

theorem weight_of_HClO2 :
  weight_HClO2 = 410.76 := by sorry

end NUMINAMATH_CALUDE_weight_of_HClO2_l319_31905


namespace NUMINAMATH_CALUDE_student_calculation_l319_31964

theorem student_calculation (x : ℕ) (h : x = 125) : 2 * x - 138 = 112 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_l319_31964


namespace NUMINAMATH_CALUDE_smallest_concatenated_multiple_of_2016_l319_31954

def concatenate_twice (n : ℕ) : ℕ :=
  n * 1000 + n

theorem smallest_concatenated_multiple_of_2016 :
  ∀ A : ℕ, A > 0 →
    (∃ k : ℕ, concatenate_twice A = 2016 * k) →
    A ≥ 288 :=
sorry

end NUMINAMATH_CALUDE_smallest_concatenated_multiple_of_2016_l319_31954


namespace NUMINAMATH_CALUDE_rectangle_area_l319_31980

/-- The area of a rectangle with perimeter 40 and length twice its width -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) (h2 : 6 * w = 40) : w * (2 * w) = 800 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l319_31980


namespace NUMINAMATH_CALUDE_square_equality_l319_31961

theorem square_equality (a b : ℝ) : a = b → a^2 = b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_equality_l319_31961


namespace NUMINAMATH_CALUDE_female_employees_count_l319_31970

/-- Represents the number of employees in a company -/
structure Company where
  total : ℕ
  female : ℕ
  male : ℕ
  female_managers : ℕ
  male_managers : ℕ

/-- The conditions of the company -/
def company_conditions (c : Company) : Prop :=
  c.female_managers = 300 ∧
  c.female_managers + c.male_managers = (2 : ℚ) / 5 * c.total ∧
  c.male_managers = (2 : ℚ) / 5 * c.male ∧
  c.total = c.female + c.male

/-- The theorem stating that the number of female employees is 750 -/
theorem female_employees_count (c : Company) : 
  company_conditions c → c.female = 750 := by
  sorry

end NUMINAMATH_CALUDE_female_employees_count_l319_31970


namespace NUMINAMATH_CALUDE_rectangle_area_l319_31997

theorem rectangle_area (length width : ℝ) (h1 : length = 5) (h2 : width = 17/20) :
  length * width = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l319_31997


namespace NUMINAMATH_CALUDE_irrational_approximation_l319_31988

-- Define α as an irrational real number
variable (α : ℝ) (h : ¬ IsRat α)

-- State the theorem
theorem irrational_approximation :
  ∃ (p q : ℤ), -(1 : ℝ) / (q : ℝ)^2 ≤ α - (p : ℝ) / (q : ℝ) ∧ α - (p : ℝ) / (q : ℝ) ≤ 1 / (q : ℝ)^2 :=
sorry

end NUMINAMATH_CALUDE_irrational_approximation_l319_31988


namespace NUMINAMATH_CALUDE_fraction_meaningful_l319_31998

theorem fraction_meaningful (x : ℝ) : (∃ y : ℝ, y = 1 / (3 - x)) ↔ x ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l319_31998


namespace NUMINAMATH_CALUDE_fraction_less_than_one_l319_31967

theorem fraction_less_than_one (a b : ℝ) (h1 : a < b) (h2 : b < 0) : b / a < 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_one_l319_31967


namespace NUMINAMATH_CALUDE_hyperbola_properties_l319_31913

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 12 - x^2 / 24 = 1

-- Define the foci
def foci : Set (ℝ × ℝ) :=
  {(0, 6), (0, -6)}

-- Define the asymptotes of the reference hyperbola
def reference_asymptotes (x y : ℝ) : Prop :=
  y = (Real.sqrt 2 / 2) * x ∨ y = -(Real.sqrt 2 / 2) * x

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola_equation x y → 
    (∃ f ∈ foci, (x - f.1)^2 + (y - f.2)^2 = 36)) ∧
  (∀ x y, hyperbola_equation x y → reference_asymptotes x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l319_31913


namespace NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l319_31948

theorem cos_four_arccos_two_fifths :
  Real.cos (4 * Real.arccos (2/5)) = -47/625 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l319_31948


namespace NUMINAMATH_CALUDE_smallest_prime_for_perfect_square_l319_31951

-- Define a function to check if a number is prime
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

-- Theorem statement
theorem smallest_prime_for_perfect_square :
  ∃ p : Nat,
    isPrime p ∧
    isPerfectSquare (5 * p^2 + p^3) ∧
    (∀ q : Nat, q < p → isPrime q → ¬isPerfectSquare (5 * q^2 + q^3)) ∧
    p = 11 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_for_perfect_square_l319_31951


namespace NUMINAMATH_CALUDE_inequality_holds_l319_31982

theorem inequality_holds (a b : ℕ+) : a^3 + (a+b)^2 + b ≠ b^3 + a + 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l319_31982


namespace NUMINAMATH_CALUDE_only_one_solves_l319_31943

def prob_A : ℚ := 1/2
def prob_B : ℚ := 1/3
def prob_C : ℚ := 1/4

theorem only_one_solves : 
  let prob_only_one := 
    (prob_A * (1 - prob_B) * (1 - prob_C)) + 
    ((1 - prob_A) * prob_B * (1 - prob_C)) + 
    ((1 - prob_A) * (1 - prob_B) * prob_C)
  prob_only_one = 11/24 := by
  sorry

end NUMINAMATH_CALUDE_only_one_solves_l319_31943


namespace NUMINAMATH_CALUDE_fraction_integer_iff_specific_p_l319_31972

theorem fraction_integer_iff_specific_p (p : ℕ+) :
  (∃ (n : ℕ+), (3 * p + 25 : ℚ) / (2 * p - 5) = n) ↔ p ∈ ({3, 5, 9, 35} : Set ℕ+) := by
  sorry

end NUMINAMATH_CALUDE_fraction_integer_iff_specific_p_l319_31972


namespace NUMINAMATH_CALUDE_min_colors_theorem_l319_31928

/-- The size of the board --/
def boardSize : Nat := 2016

/-- A color assignment for the board --/
def ColorAssignment := Fin boardSize → Fin boardSize → Nat

/-- Checks if a color assignment satisfies the diagonal condition --/
def satisfiesDiagonalCondition (c : ColorAssignment) : Prop :=
  ∀ i, c i i = 1

/-- Checks if a color assignment satisfies the symmetry condition --/
def satisfiesSymmetryCondition (c : ColorAssignment) : Prop :=
  ∀ i j, c i j = c j i

/-- Checks if a color assignment satisfies the row condition --/
def satisfiesRowCondition (c : ColorAssignment) : Prop :=
  ∀ i j k, i ≠ j ∧ (i < j ∧ j < k ∨ k < j ∧ j < i) → c i k ≠ c j k

/-- Checks if a color assignment is valid --/
def isValidColorAssignment (c : ColorAssignment) : Prop :=
  satisfiesDiagonalCondition c ∧ satisfiesSymmetryCondition c ∧ satisfiesRowCondition c

/-- The minimum number of colors required --/
def minColors : Nat := 11

/-- Theorem stating the minimum number of colors required --/
theorem min_colors_theorem :
  (∃ (c : ColorAssignment), isValidColorAssignment c ∧ (∀ i j, c i j < minColors)) ∧
  (∀ k < minColors, ¬∃ (c : ColorAssignment), isValidColorAssignment c ∧ (∀ i j, c i j < k)) :=
sorry

end NUMINAMATH_CALUDE_min_colors_theorem_l319_31928


namespace NUMINAMATH_CALUDE_two_numbers_sum_667_lcm_gcd_120_l319_31911

theorem two_numbers_sum_667_lcm_gcd_120 :
  ∀ a b : ℕ,
  a + b = 667 →
  (Nat.lcm a b) / (Nat.gcd a b) = 120 →
  ((a = 552 ∧ b = 115) ∨ (a = 115 ∧ b = 552) ∨ (a = 435 ∧ b = 232) ∨ (a = 232 ∧ b = 435)) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_sum_667_lcm_gcd_120_l319_31911


namespace NUMINAMATH_CALUDE_function_comparison_l319_31977

open Set

theorem function_comparison (a b : ℝ) (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) (hg : Differentiable ℝ g)
  (h_eq : f a = g a)
  (h_deriv : ∀ x ∈ Set.Ioo a b, deriv f x > deriv g x) :
  ∀ x ∈ Set.Ioo a b, f x > g x := by
  sorry

end NUMINAMATH_CALUDE_function_comparison_l319_31977


namespace NUMINAMATH_CALUDE_range_of_m_l319_31985

open Set

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ 
  x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ¬∃ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 = 0

-- Define the range of m
def range_m : Set ℝ := Ioc 1 2 ∪ Ici 3

-- Theorem statement
theorem range_of_m : 
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ 
  (∀ m : ℝ, m ∈ range_m ↔ (p m ∨ q m) ∧ ¬(p m ∧ q m)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l319_31985


namespace NUMINAMATH_CALUDE_monotone_increasing_k_range_l319_31975

/-- A function f(x) = kx^2 - ln x is monotonically increasing in the interval (1, +∞) -/
def is_monotone_increasing (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y, 1 < x ∧ x < y → f x ≤ f y

/-- The range of k for which f(x) = kx^2 - ln x is monotonically increasing in (1, +∞) -/
theorem monotone_increasing_k_range (k : ℝ) :
  (is_monotone_increasing (fun x => k * x^2 - Real.log x) k) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_k_range_l319_31975


namespace NUMINAMATH_CALUDE_problems_per_page_l319_31917

theorem problems_per_page
  (total_problems : ℕ)
  (finished_problems : ℕ)
  (remaining_pages : ℕ)
  (h1 : total_problems = 60)
  (h2 : finished_problems = 20)
  (h3 : remaining_pages = 5)
  (h4 : remaining_pages > 0)
  : (total_problems - finished_problems) / remaining_pages = 8 := by
  sorry

end NUMINAMATH_CALUDE_problems_per_page_l319_31917


namespace NUMINAMATH_CALUDE_multiple_p_solutions_l319_31957

/-- The probability of getting exactly k heads in n tosses of a coin with probability p of heads -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 3 heads in 5 tosses -/
def w (p : ℝ) : ℝ := binomial_probability 5 3 p

/-- There exist at least two distinct values of p in (0, 1) that satisfy w(p) = 144/625 -/
theorem multiple_p_solutions : ∃ p₁ p₂ : ℝ, 0 < p₁ ∧ p₁ < 1 ∧ 0 < p₂ ∧ p₂ < 1 ∧ p₁ ≠ p₂ ∧ w p₁ = 144/625 ∧ w p₂ = 144/625 := by
  sorry

end NUMINAMATH_CALUDE_multiple_p_solutions_l319_31957


namespace NUMINAMATH_CALUDE_cube_and_sphere_problem_l319_31993

theorem cube_and_sphere_problem (V1 : Real) (h1 : V1 = 8) : ∃ (V2 r : Real),
  let s1 := V1 ^ (1/3)
  let A1 := 6 * s1^2
  let A2 := 3 * A1
  let s2 := (A2 / 6) ^ (1/2)
  V2 = s2^3 ∧ 
  4 * Real.pi * r^2 = A2 ∧
  V2 = 24 * Real.sqrt 3 ∧
  r = Real.sqrt (18 / Real.pi) := by
sorry

end NUMINAMATH_CALUDE_cube_and_sphere_problem_l319_31993


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l319_31930

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + b*x + 2 < 0 ↔ 1/3 < x ∧ x < 1/2) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l319_31930


namespace NUMINAMATH_CALUDE_circle_equations_l319_31926

-- Define points
def A : ℝ × ℝ := (6, 5)
def B : ℝ × ℝ := (0, 1)
def P : ℝ × ℝ := (-2, 4)
def Q : ℝ × ℝ := (3, -1)

-- Define the line equation for the center
def center_line (x y : ℝ) : Prop := 3 * x + 10 * y + 9 = 0

-- Define the chord length on x-axis
def chord_length : ℝ := 6

-- Define circle equations
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 13
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 25

theorem circle_equations :
  ∃ (C : ℝ × ℝ),
    (center_line C.1 C.2) ∧
    (circle1 A.1 A.2 ∨ circle2 A.1 A.2) ∧
    (circle1 B.1 B.2 ∨ circle2 B.1 B.2) ∧
    (circle1 P.1 P.2 ∨ circle2 P.1 P.2) ∧
    (circle1 Q.1 Q.2 ∨ circle2 Q.1 Q.2) ∧
    (∃ (x1 x2 : ℝ), x2 - x1 = chord_length ∧
      ((circle1 x1 0 ∧ circle1 x2 0) ∨ (circle2 x1 0 ∧ circle2 x2 0))) :=
by sorry


end NUMINAMATH_CALUDE_circle_equations_l319_31926


namespace NUMINAMATH_CALUDE_perfect_square_function_characterization_l319_31987

theorem perfect_square_function_characterization (g : ℕ → ℕ) : 
  (∀ n m : ℕ, ∃ k : ℕ, (g n + m) * (g m + n) = k^2) →
  ∃ c : ℕ, ∀ n : ℕ, g n = n + c := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_function_characterization_l319_31987


namespace NUMINAMATH_CALUDE_m_range_l319_31971

def p (m : ℝ) : Prop := ∀ x > 0, m^2 + 2*m - 1 ≤ x + 1/x

def q (m : ℝ) : Prop := ∀ x₁ x₂, x₁ < x₂ → (5 - m^2)^x₁ < (5 - m^2)^x₂

theorem m_range (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) :
  (-3 ≤ m ∧ m ≤ -2) ∨ (1 < m ∧ m < 2) := by
  sorry

end NUMINAMATH_CALUDE_m_range_l319_31971


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_b_zero_l319_31984

theorem quadratic_inequality_solution_implies_b_zero
  (a b m : ℝ)
  (h : ∀ x, ax^2 - a*x + b < 0 ↔ m < x ∧ x < m + 1) :
  b = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_b_zero_l319_31984


namespace NUMINAMATH_CALUDE_opinion_change_percentage_l319_31947

theorem opinion_change_percentage
  (physics_initial_enjoy : ℝ)
  (physics_initial_dislike : ℝ)
  (physics_final_enjoy : ℝ)
  (physics_final_dislike : ℝ)
  (chem_initial_enjoy : ℝ)
  (chem_initial_dislike : ℝ)
  (chem_final_enjoy : ℝ)
  (chem_final_dislike : ℝ)
  (h1 : physics_initial_enjoy = 40)
  (h2 : physics_initial_dislike = 60)
  (h3 : physics_final_enjoy = 75)
  (h4 : physics_final_dislike = 25)
  (h5 : chem_initial_enjoy = 30)
  (h6 : chem_initial_dislike = 70)
  (h7 : chem_final_enjoy = 65)
  (h8 : chem_final_dislike = 35)
  (h9 : physics_initial_enjoy + physics_initial_dislike = 100)
  (h10 : physics_final_enjoy + physics_final_dislike = 100)
  (h11 : chem_initial_enjoy + chem_initial_dislike = 100)
  (h12 : chem_final_enjoy + chem_final_dislike = 100) :
  ∃ (min_change max_change : ℝ),
    min_change = 70 ∧
    max_change = 70 ∧
    (∀ (actual_change : ℝ),
      actual_change ≥ min_change ∧
      actual_change ≤ max_change) :=
by sorry

end NUMINAMATH_CALUDE_opinion_change_percentage_l319_31947


namespace NUMINAMATH_CALUDE_function_property_implies_zero_l319_31904

theorem function_property_implies_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f (y^2) = f (x^2 + y)) : 
  f (-2017) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_property_implies_zero_l319_31904


namespace NUMINAMATH_CALUDE_combined_cost_price_l319_31959

/-- Given three items with selling prices and profit percentages, 
    calculate the combined cost price --/
theorem combined_cost_price 
  (sp1 sp2 sp3 : ℝ) 
  (profit_percent1 profit_percent2 profit_percent3 : ℝ) :
  let cp1 := sp1 / (1 + profit_percent1 / 100)
  let cp2 := sp2 / (1 + profit_percent2 / 100)
  let cp3 := sp3 / (1 + profit_percent3 / 100)
  cp1 + cp2 + cp3 = 
    sp1 / (1 + profit_percent1 / 100) + 
    sp2 / (1 + profit_percent2 / 100) + 
    sp3 / (1 + profit_percent3 / 100) :=
by sorry

end NUMINAMATH_CALUDE_combined_cost_price_l319_31959


namespace NUMINAMATH_CALUDE_expected_winnings_is_one_sixth_l319_31937

/-- A strange die with 6 sides -/
inductive DieSide
  | one
  | two
  | three
  | four
  | five
  | six

/-- Probability of rolling each side of the die -/
def probability (s : DieSide) : ℚ :=
  match s with
  | DieSide.one => 1/4
  | DieSide.two => 1/4
  | DieSide.three => 1/6
  | DieSide.four => 1/6
  | DieSide.five => 1/6
  | DieSide.six => 1/12

/-- Winnings (or losses) for each outcome -/
def winnings (s : DieSide) : ℤ :=
  match s with
  | DieSide.one => 2
  | DieSide.two => 2
  | DieSide.three => 4
  | DieSide.four => 4
  | DieSide.five => -6
  | DieSide.six => -12

/-- Expected value of winnings -/
def expected_winnings : ℚ :=
  (probability DieSide.one * winnings DieSide.one) +
  (probability DieSide.two * winnings DieSide.two) +
  (probability DieSide.three * winnings DieSide.three) +
  (probability DieSide.four * winnings DieSide.four) +
  (probability DieSide.five * winnings DieSide.five) +
  (probability DieSide.six * winnings DieSide.six)

theorem expected_winnings_is_one_sixth :
  expected_winnings = 1/6 := by sorry

end NUMINAMATH_CALUDE_expected_winnings_is_one_sixth_l319_31937


namespace NUMINAMATH_CALUDE_g_at_six_l319_31920

def g (x : ℝ) : ℝ := 2*x^4 - 19*x^3 + 30*x^2 - 12*x - 72

theorem g_at_six : g 6 = 288 := by
  sorry

end NUMINAMATH_CALUDE_g_at_six_l319_31920


namespace NUMINAMATH_CALUDE_homothetic_image_containment_l319_31941

-- Define a convex polygon
def ConvexPolygon (P : Set (Point)) : Prop := sorry

-- Define a homothetic transformation
def HomotheticTransformation (center : Point) (k : ℝ) (P : Set Point) : Set Point := sorry

-- Define that a set is contained within another set
def IsContainedIn (A B : Set Point) : Prop := sorry

-- The theorem statement
theorem homothetic_image_containment 
  (P : Set Point) (h : ConvexPolygon P) :
  ∃ (center : Point), 
    IsContainedIn (HomotheticTransformation center (1/2) P) P := by
  sorry

end NUMINAMATH_CALUDE_homothetic_image_containment_l319_31941


namespace NUMINAMATH_CALUDE_tank_capacities_l319_31960

/-- Given three tanks with capacities T1, T2, and T3, prove that the total amount of water is 10850 gallons. -/
theorem tank_capacities (T1 T2 T3 : ℝ) : 
  (3/4 : ℝ) * T1 + (4/5 : ℝ) * T2 + (1/2 : ℝ) * T3 = 10850 := by
  sorry

#check tank_capacities

end NUMINAMATH_CALUDE_tank_capacities_l319_31960


namespace NUMINAMATH_CALUDE_divisor_implies_value_l319_31907

theorem divisor_implies_value (k : ℕ) : 
  21^k ∣ 435961 → 7^k - k^7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_implies_value_l319_31907


namespace NUMINAMATH_CALUDE_goldbach_extension_l319_31995

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def has_two_prime_pairs (N : ℕ) : Prop :=
  ∃ (p₁ q₁ p₂ q₂ : ℕ),
    is_prime p₁ ∧ is_prime q₁ ∧ is_prime p₂ ∧ is_prime q₂ ∧
    p₁ + q₁ = N ∧ p₂ + q₂ = N ∧
    (p₁ ≠ p₂ ∨ q₁ ≠ q₂) ∧
    ∀ (p q : ℕ), is_prime p → is_prime q → p + q = N →
      ((p = p₁ ∧ q = q₁) ∨ (p = p₂ ∧ q = q₂) ∨ (p = q₁ ∧ q = p₁) ∨ (p = q₂ ∧ q = p₂))

theorem goldbach_extension :
  ∀ N : ℕ, N ≥ 10 →
    (N = 10 ↔ (N % 2 = 0 ∧ has_two_prime_pairs N ∧
      ∀ M : ℕ, M < N → M % 2 = 0 → M > 2 → ¬has_two_prime_pairs M)) :=
sorry

end NUMINAMATH_CALUDE_goldbach_extension_l319_31995


namespace NUMINAMATH_CALUDE_benny_apples_l319_31900

def dan_apples : ℕ := 9
def total_apples : ℕ := 11

theorem benny_apples : total_apples - dan_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_benny_apples_l319_31900


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l319_31973

/-- The percentage of units with Type A defects in the first stage -/
def type_a_defect_rate : ℝ := 0.07

/-- The percentage of units with Type B defects in the second stage -/
def type_b_defect_rate : ℝ := 0.08

/-- The percentage of Type A defective units that are shipped for sale -/
def type_a_ship_rate : ℝ := 0.03

/-- The percentage of Type B defective units that are shipped for sale -/
def type_b_ship_rate : ℝ := 0.06

/-- The total percentage of defective units (Type A or B) that are shipped for sale -/
def total_defective_shipped_rate : ℝ :=
  type_a_defect_rate * type_a_ship_rate + type_b_defect_rate * type_b_ship_rate

theorem defective_shipped_percentage :
  total_defective_shipped_rate = 0.0069 := by sorry

end NUMINAMATH_CALUDE_defective_shipped_percentage_l319_31973


namespace NUMINAMATH_CALUDE_exist_four_numbers_squares_l319_31965

theorem exist_four_numbers_squares : 
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (m n : ℕ), a^2 + 2*c*d + b^2 = m^2 ∧ c^2 + 2*a*b + d^2 = n^2 :=
by sorry

end NUMINAMATH_CALUDE_exist_four_numbers_squares_l319_31965


namespace NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_eq_two_thirds_l319_31974

theorem tan_alpha_3_implies_fraction_eq_two_thirds (α : Real) (h : Real.tan α = 3) :
  1 / (Real.sin α ^ 2 + 2 * Real.sin α * Real.cos α) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_eq_two_thirds_l319_31974


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_cubic_eq_l319_31925

theorem x_eq_one_sufficient_not_necessary_for_cubic_eq :
  (∀ x : ℝ, x = 1 → x^3 - 2*x + 1 = 0) ∧
  (∃ x : ℝ, x ≠ 1 ∧ x^3 - 2*x + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_cubic_eq_l319_31925


namespace NUMINAMATH_CALUDE_equation_solution_l319_31910

theorem equation_solution : ∃ x : ℝ, 
  (45 * x) / (3/4) = (37.5/100) * 1500 - (62.5/100) * 800 ∧ 
  abs (x - 1.0417) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l319_31910


namespace NUMINAMATH_CALUDE_felipe_house_building_time_l319_31955

/-- Felipe and Emilio's house building problem -/
theorem felipe_house_building_time :
  ∀ (felipe_time emilio_time : ℝ),
  felipe_time + emilio_time = 7.5 →
  felipe_time = (1/2) * emilio_time →
  felipe_time * 12 = 30 := by
  sorry

end NUMINAMATH_CALUDE_felipe_house_building_time_l319_31955


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_l319_31919

theorem smallest_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≥ n) ∧
  n = 105 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_l319_31919


namespace NUMINAMATH_CALUDE_courier_packages_tomorrow_l319_31915

/-- The number of packages to be delivered tomorrow -/
def packages_to_deliver (yesterday : ℕ) (today : ℕ) : ℕ :=
  yesterday + today

/-- Theorem: The courier should deliver 240 packages tomorrow -/
theorem courier_packages_tomorrow :
  let yesterday := 80
  let today := 2 * yesterday
  packages_to_deliver yesterday today = 240 := by
  sorry

end NUMINAMATH_CALUDE_courier_packages_tomorrow_l319_31915


namespace NUMINAMATH_CALUDE_scoop_size_l319_31952

/-- Given the total amount of ingredients and the total number of scoops, 
    calculate the size of each scoop. -/
theorem scoop_size (total_cups : ℚ) (total_scoops : ℕ) 
  (h1 : total_cups = 5) 
  (h2 : total_scoops = 15) : 
  total_cups / total_scoops = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_scoop_size_l319_31952


namespace NUMINAMATH_CALUDE_hillary_stops_short_of_summit_l319_31916

/-- Represents the climbing scenario of Hillary and Eddy on Mt. Everest -/
structure ClimbingScenario where
  summit_distance : ℝ  -- Distance from base camp to summit in feet
  hillary_ascent_rate : ℝ  -- Hillary's ascent rate in ft/hr
  eddy_ascent_rate : ℝ  -- Eddy's ascent rate in ft/hr
  hillary_descent_rate : ℝ  -- Hillary's descent rate in ft/hr
  trip_duration : ℝ  -- Duration of the trip in hours

/-- Calculates the distance Hillary stops short of the summit -/
def distance_short_of_summit (scenario : ClimbingScenario) : ℝ :=
  scenario.hillary_ascent_rate * scenario.trip_duration - 
  (scenario.hillary_ascent_rate * scenario.trip_duration + 
   scenario.eddy_ascent_rate * scenario.trip_duration - scenario.summit_distance)

/-- Theorem stating that Hillary stops 3000 ft short of the summit -/
theorem hillary_stops_short_of_summit (scenario : ClimbingScenario) 
  (h1 : scenario.summit_distance = 5000)
  (h2 : scenario.hillary_ascent_rate = 800)
  (h3 : scenario.eddy_ascent_rate = 500)
  (h4 : scenario.hillary_descent_rate = 1000)
  (h5 : scenario.trip_duration = 6) :
  distance_short_of_summit scenario = 3000 := by
  sorry

#eval distance_short_of_summit {
  summit_distance := 5000,
  hillary_ascent_rate := 800,
  eddy_ascent_rate := 500,
  hillary_descent_rate := 1000,
  trip_duration := 6
}

end NUMINAMATH_CALUDE_hillary_stops_short_of_summit_l319_31916


namespace NUMINAMATH_CALUDE_sum_of_ratios_bound_l319_31944

theorem sum_of_ratios_bound (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  a / (b + c^2) + b / (c + a^2) + c / (a + b^2) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_bound_l319_31944


namespace NUMINAMATH_CALUDE_function_lower_bound_l319_31914

theorem function_lower_bound
  (f : ℝ → ℝ)
  (h_cont : ContinuousOn f (Set.Ioi 0))
  (h_ineq : ∀ x > 0, f (x^2) ≥ f x)
  (h_f1 : f 1 = 5) :
  ∀ x > 0, f x ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_function_lower_bound_l319_31914


namespace NUMINAMATH_CALUDE_quadrilateral_S_l319_31924

/-- Given a quadrilateral with sides a, b, c, d and an angle A (where A ≠ 90°),
    S is equal to (a^2 + d^2 - b^2 - c^2) / (4 * tan(A)) -/
theorem quadrilateral_S (a b c d : ℝ) (A : ℝ) (h : A ≠ π / 2) :
  let S := (a^2 + d^2 - b^2 - c^2) / (4 * Real.tan A)
  ∃ (S : ℝ), S = (a^2 + d^2 - b^2 - c^2) / (4 * Real.tan A) :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_S_l319_31924


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l319_31918

/-- 
Given an arithmetic progression where the sum of the 4th and 12th terms is 8,
prove that the sum of the first 15 terms is 60.
-/
theorem arithmetic_progression_sum (a d : ℝ) : 
  (a + 3*d) + (a + 11*d) = 8 → 
  (15 : ℝ) / 2 * (2*a + 14*d) = 60 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l319_31918


namespace NUMINAMATH_CALUDE_triangle_max_area_l319_31932

/-- Given a triangle ABC where a^2 + b^2 + 2c^2 = 8, 
    the maximum area of the triangle is 2√5/5 -/
theorem triangle_max_area (a b c : ℝ) (h : a^2 + b^2 + 2*c^2 = 8) :
  ∃ (S : ℝ), S = (2 * Real.sqrt 5) / 5 ∧ 
  (∀ (S' : ℝ), S' = (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) → S' ≤ S) := by
  sorry


end NUMINAMATH_CALUDE_triangle_max_area_l319_31932


namespace NUMINAMATH_CALUDE_f_sum_equals_two_l319_31912

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x ^ 2) - 3 * x) + 1

theorem f_sum_equals_two :
  f (Real.log 2 / Real.log 10) + f (Real.log (1 / 2) / Real.log 10) = 2 := by sorry

end NUMINAMATH_CALUDE_f_sum_equals_two_l319_31912


namespace NUMINAMATH_CALUDE_equality_proof_l319_31931

theorem equality_proof (n : ℕ) (h : n ≥ 2) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_equality_proof_l319_31931


namespace NUMINAMATH_CALUDE_bakery_storage_ratio_l319_31946

/-- Given the conditions of a bakery's storage room, prove the ratio of flour to baking soda. -/
theorem bakery_storage_ratio :
  let sugar_flour_ratio : ℚ := 5 / 6
  let sugar_amount : ℕ := 2000
  let flour_amount : ℕ := (sugar_amount * 6) / 5
  let baking_soda_amount : ℕ := flour_amount / 8 - 60
  (flour_amount : ℚ) / baking_soda_amount = 10 := by
  sorry

end NUMINAMATH_CALUDE_bakery_storage_ratio_l319_31946


namespace NUMINAMATH_CALUDE_parabola_chord_dot_product_l319_31983

/-- The parabola y^2 = 4x with focus at (1, 0) -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- A chord passing through the focus -/
def Chord (A B : ℝ × ℝ) : Prop :=
  A ∈ Parabola ∧ B ∈ Parabola ∧ ∃ t : ℝ, (1 - t) • A + t • B = Focus

theorem parabola_chord_dot_product (A B : ℝ × ℝ) (h : Chord A B) :
    (A.1 * B.1 + A.2 * B.2 : ℝ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_chord_dot_product_l319_31983


namespace NUMINAMATH_CALUDE_polynomial_coefficient_equality_l319_31923

theorem polynomial_coefficient_equality : 
  ∃! (a b c : ℝ), ∀ (x : ℝ), 
    2*x^4 + x^3 - 41*x^2 + 83*x - 45 = (a*x^2 + b*x + c)*(x^2 + 4*x + 9) ∧ 
    a = 2 ∧ b = -7 ∧ c = -5 := by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_equality_l319_31923


namespace NUMINAMATH_CALUDE_shoe_trial_ratio_l319_31902

/-- Represents the number of shoe pairs tried on at each store --/
structure ShoeTrials :=
  (store1 : ℕ)
  (store2 : ℕ)
  (store3 : ℕ)
  (store4 : ℕ)

/-- Calculates the ratio of shoes tried on at the fourth store to the first three stores combined --/
def calculateRatio (trials : ShoeTrials) : Rat :=
  trials.store4 / (trials.store1 + trials.store2 + trials.store3)

theorem shoe_trial_ratio :
  ∀ (trials : ShoeTrials),
    trials.store1 = 7 →
    trials.store2 = trials.store1 + 2 →
    trials.store3 = 0 →
    trials.store1 + trials.store2 + trials.store3 + trials.store4 = 48 →
    calculateRatio trials = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_shoe_trial_ratio_l319_31902


namespace NUMINAMATH_CALUDE_linear_system_fraction_l319_31979

theorem linear_system_fraction (x y m n : ℚ) 
  (eq1 : x + 2*y = 5)
  (eq2 : x + y = 7)
  (eq3 : x = -m)
  (eq4 : y = -n) :
  (3*m + 2*n) / (5*m - n) = 11/14 := by
sorry

end NUMINAMATH_CALUDE_linear_system_fraction_l319_31979


namespace NUMINAMATH_CALUDE_jellys_pepperoni_count_l319_31968

/-- Represents a pizza with pepperoni slices -/
structure Pizza :=
  (total_pepperoni : ℕ)

/-- Represents a slice of pizza -/
structure PizzaSlice :=
  (pepperoni : ℕ)

/-- Cuts a pizza into quarters and removes one pepperoni from one slice -/
def cut_and_serve (p : Pizza) : PizzaSlice :=
  let quarter_pepperoni := p.total_pepperoni / 4
  { pepperoni := quarter_pepperoni - 1 }

/-- Theorem: Jelly's pizza slice has 9 pepperoni slices -/
theorem jellys_pepperoni_count (p : Pizza) (h : p.total_pepperoni = 40) : 
  (cut_and_serve p).pepperoni = 9 := by
  sorry

#check jellys_pepperoni_count

end NUMINAMATH_CALUDE_jellys_pepperoni_count_l319_31968


namespace NUMINAMATH_CALUDE_real_number_in_set_l319_31994

theorem real_number_in_set (a : ℝ) : a ∈ ({a^2 - a, 0} : Set ℝ) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_real_number_in_set_l319_31994


namespace NUMINAMATH_CALUDE_OMM_MOO_not_synonyms_l319_31927

/-- Represents a word in the Ancient Tribe language --/
inductive AncientWord
  | M : AncientWord
  | O : AncientWord
  | append : AncientWord → AncientWord → AncientWord

/-- Counts the number of 'M's in a word --/
def countM : AncientWord → Nat
  | AncientWord.M => 1
  | AncientWord.O => 0
  | AncientWord.append w1 w2 => countM w1 + countM w2

/-- Counts the number of 'O's in a word --/
def countO : AncientWord → Nat
  | AncientWord.M => 0
  | AncientWord.O => 1
  | AncientWord.append w1 w2 => countO w1 + countO w2

/-- Calculates the difference between 'M's and 'O's in a word --/
def letterDifference (w : AncientWord) : Int :=
  (countM w : Int) - (countO w : Int)

/-- Two words are synonyms if their letter differences are equal --/
def areSynonyms (w1 w2 : AncientWord) : Prop :=
  letterDifference w1 = letterDifference w2

/-- Construct the word "OMM" --/
def OMM : AncientWord :=
  AncientWord.append AncientWord.O (AncientWord.append AncientWord.M AncientWord.M)

/-- Construct the word "MOO" --/
def MOO : AncientWord :=
  AncientWord.append AncientWord.M (AncientWord.append AncientWord.O AncientWord.O)

/-- Theorem: "OMM" and "MOO" are not synonyms --/
theorem OMM_MOO_not_synonyms : ¬(areSynonyms OMM MOO) := by
  sorry


end NUMINAMATH_CALUDE_OMM_MOO_not_synonyms_l319_31927


namespace NUMINAMATH_CALUDE_beavers_swimming_l319_31949

theorem beavers_swimming (initial_beavers final_beavers : ℕ) : 
  initial_beavers ≥ final_beavers → 
  initial_beavers - final_beavers = initial_beavers - final_beavers :=
by
  sorry

#check beavers_swimming 2 1

end NUMINAMATH_CALUDE_beavers_swimming_l319_31949


namespace NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l319_31953

theorem polynomial_root_implies_coefficients : ∀ (c d : ℝ),
  (∃ (x : ℂ), x^3 + c*x^2 + 2*x + d = 0 ∧ x = Complex.mk 2 (-3)) →
  c = 5/4 ∧ d = -143/4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l319_31953


namespace NUMINAMATH_CALUDE_remaining_pages_l319_31935

/-- Calculates the remaining pages in a pad after various projects --/
theorem remaining_pages (initial_pages : ℕ) : 
  initial_pages = 120 → 
  (initial_pages / 2 - 
   (initial_pages / 4 + 10 + initial_pages * 15 / 100) / 2) = 31 := by
  sorry

end NUMINAMATH_CALUDE_remaining_pages_l319_31935


namespace NUMINAMATH_CALUDE_cubic_root_sum_l319_31958

theorem cubic_root_sum (a b c : ℝ) : 
  (40 * a^3 - 60 * a^2 + 25 * a - 1 = 0) →
  (40 * b^3 - 60 * b^2 + 25 * b - 1 = 0) →
  (40 * c^3 - 60 * c^2 + 25 * c - 1 = 0) →
  (0 < a) ∧ (a < 1) →
  (0 < b) ∧ (b < 1) →
  (0 < c) ∧ (c < 1) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l319_31958


namespace NUMINAMATH_CALUDE_largest_k_for_g_range_l319_31921

/-- The function g(x) defined as x^2 - 7x + k -/
def g (k : ℝ) (x : ℝ) : ℝ := x^2 - 7*x + k

/-- Theorem stating that the largest value of k such that 4 is in the range of g(x) is 65/4 -/
theorem largest_k_for_g_range : 
  (∃ (k : ℝ), ∀ (k' : ℝ), (∃ (x : ℝ), g k' x = 4) → k' ≤ k) ∧ 
  (∃ (x : ℝ), g (65/4) x = 4) := by
  sorry

end NUMINAMATH_CALUDE_largest_k_for_g_range_l319_31921


namespace NUMINAMATH_CALUDE_min_value_greater_than_five_l319_31981

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 + |x + a - 1| + (a + 1)^2

-- State the theorem
theorem min_value_greater_than_five (a : ℝ) :
  (∀ x, f x a > 5) ↔ a < (-1 - Real.sqrt 14) / 2 ∨ a > Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_greater_than_five_l319_31981


namespace NUMINAMATH_CALUDE_range_of_m_l319_31934

/-- Given a function f with derivative f', we define g and prove a property about m. -/
theorem range_of_m (f : ℝ → ℝ) (f' : ℝ → ℝ) (g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, HasDerivAt f (f' x) x) →  -- f has derivative f' for all x
  (∀ x, g x = f x - (1/2) * x^2) →  -- definition of g
  (∀ x, f' x < x) →  -- condition on f'
  (f (4 - m) - f m ≥ 8 - 4*m) →  -- given inequality
  m ≥ 2 :=  -- conclusion: m is in [2, +∞)
sorry

end NUMINAMATH_CALUDE_range_of_m_l319_31934


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l319_31963

-- Define the inverse relationship between two quantities
def inverse_relation (x y : ℝ) (k : ℝ) : Prop := x * y = k

-- Define the given conditions
def conditions : Prop :=
  ∃ (k m : ℝ),
    inverse_relation 1500 0.4 k ∧
    inverse_relation 1500 2.5 m ∧
    inverse_relation 3000 0.2 k ∧
    inverse_relation 3000 1.25 m

-- State the theorem
theorem inverse_variation_problem :
  conditions → (∃ (s t : ℝ), s = 0.2 ∧ t = 1.25) :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l319_31963


namespace NUMINAMATH_CALUDE_book_distribution_l319_31989

theorem book_distribution (n m : ℕ) (hn : n = 7) (hm : m = 3) :
  (Nat.factorial n) / (Nat.factorial (n - m)) = 210 :=
sorry

end NUMINAMATH_CALUDE_book_distribution_l319_31989


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l319_31991

theorem concentric_circles_ratio 
  (r R : ℝ) 
  (a b c : ℝ) 
  (h_positive : 0 < r ∧ 0 < R ∧ 0 < a ∧ 0 < b ∧ 0 < c)
  (h_r_less_R : r < R)
  (h_area_ratio : (π * R^2 - π * r^2) / (π * R^2) = a / (b + c)) :
  R / r = Real.sqrt a / Real.sqrt (b + c - a) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l319_31991


namespace NUMINAMATH_CALUDE_divisibility_count_l319_31922

theorem divisibility_count : ∃ (S : Finset Nat), 
  (∀ n ∈ S, n ≤ 30 ∧ n > 0 ∧ (n! % (n * (n + 2) / 3) = 0)) ∧ 
  Finset.card S = 30 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_count_l319_31922


namespace NUMINAMATH_CALUDE_nearest_integer_to_power_l319_31966

theorem nearest_integer_to_power : 
  ∃ (n : ℤ), n = 2654 ∧ 
  ∀ (m : ℤ), |((3 : ℝ) + Real.sqrt 5)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^6 - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_power_l319_31966


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l319_31969

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - (x + 2)^2 / 9 = 1

/-- The asymptote equation -/
def asymptote_eq (x y m : ℝ) : Prop :=
  y - 1 = m * x + 2 * m ∨ y - 1 = -m * x - 2 * m

/-- Theorem stating that the value of m for the given hyperbola is 4/3 -/
theorem hyperbola_asymptote_slope :
  ∃ m : ℝ, (∀ x y : ℝ, hyperbola_eq x y → asymptote_eq x y m) ∧ m = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l319_31969


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l319_31999

/-- Given a line segment with one endpoint at (10, 2) and midpoint at (4, -6),
    the sum of the coordinates of the other endpoint is -16. -/
theorem endpoint_coordinate_sum :
  ∀ (x y : ℝ),
  (x + 10) / 2 = 4 →
  (y + 2) / 2 = -6 →
  x + y = -16 := by
sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l319_31999


namespace NUMINAMATH_CALUDE_shirt_discount_problem_l319_31936

theorem shirt_discount_problem (original_price : ℝ) : 
  (0.75 * (0.75 * original_price) = 19) → 
  original_price = 33.78 := by
  sorry

end NUMINAMATH_CALUDE_shirt_discount_problem_l319_31936


namespace NUMINAMATH_CALUDE_max_areas_formula_l319_31976

/-- Represents a circular disk configuration -/
structure DiskConfiguration where
  n : ℕ
  radii_count : ℕ
  has_secant : Bool
  has_chord : Bool
  chord_intersects_secant : Bool

/-- The maximum number of non-overlapping areas in a disk configuration -/
def max_areas (config : DiskConfiguration) : ℕ :=
  sorry

/-- Theorem stating the maximum number of non-overlapping areas -/
theorem max_areas_formula (config : DiskConfiguration) 
  (h1 : config.n > 0)
  (h2 : config.radii_count = 2 * config.n)
  (h3 : config.has_secant = true)
  (h4 : config.has_chord = true)
  (h5 : config.chord_intersects_secant = false) :
  max_areas config = 4 * config.n - 1 :=
sorry

end NUMINAMATH_CALUDE_max_areas_formula_l319_31976


namespace NUMINAMATH_CALUDE_correct_solution_l319_31938

/-- The original equation -/
def original_equation (x : ℚ) : Prop :=
  (2 - 2*x) / 3 = (3*x - 3) / 7 + 3

/-- Xiao Jun's incorrect equation -/
def incorrect_equation (x m : ℚ) : Prop :=
  7*(2 - 2*x) = 3*(3*x - m) + 3

/-- Xiao Jun's solution -/
def xiao_jun_solution : ℚ := 14/23

/-- The correct value of m -/
def correct_m : ℚ := 3

theorem correct_solution :
  incorrect_equation xiao_jun_solution correct_m →
  ∃ x : ℚ, x = 2 ∧ original_equation x :=
by sorry

end NUMINAMATH_CALUDE_correct_solution_l319_31938


namespace NUMINAMATH_CALUDE_danny_in_position_three_l319_31933

-- Define the people
inductive Person : Type
| Amelia : Person
| Blake : Person
| Claire : Person
| Danny : Person

-- Define the positions
inductive Position : Type
| One : Position
| Two : Position
| Three : Position
| Four : Position

-- Define the seating arrangement
def Seating := Person → Position

-- Define opposite positions
def opposite (p : Position) : Position :=
  match p with
  | Position.One => Position.Three
  | Position.Two => Position.Four
  | Position.Three => Position.One
  | Position.Four => Position.Two

-- Define adjacent positions
def adjacent (p1 p2 : Position) : Prop :=
  (p1 = Position.One ∧ p2 = Position.Two) ∨
  (p1 = Position.Two ∧ p2 = Position.Three) ∨
  (p1 = Position.Three ∧ p2 = Position.Four) ∨
  (p1 = Position.Four ∧ p2 = Position.One) ∨
  (p2 = Position.One ∧ p1 = Position.Two) ∨
  (p2 = Position.Two ∧ p1 = Position.Three) ∨
  (p2 = Position.Three ∧ p1 = Position.Four) ∨
  (p2 = Position.Four ∧ p1 = Position.One)

-- Define between positions
def between (p1 p2 p3 : Position) : Prop :=
  (adjacent p1 p2 ∧ adjacent p2 p3) ∨
  (adjacent p3 p1 ∧ adjacent p1 p2)

-- Theorem statement
theorem danny_in_position_three 
  (s : Seating)
  (claire_in_one : s Person.Claire = Position.One)
  (not_blake_opposite_claire : s Person.Blake ≠ opposite (s Person.Claire))
  (not_amelia_between_blake_claire : ¬ between (s Person.Blake) (s Person.Amelia) (s Person.Claire)) :
  s Person.Danny = Position.Three :=
by sorry

end NUMINAMATH_CALUDE_danny_in_position_three_l319_31933


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l319_31962

def A : Set ℝ := {-1, 1, 2, 3, 4}
def B : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l319_31962


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l319_31940

theorem quadratic_monotonicity (a : ℝ) :
  (∀ x ∈ Set.Ioo 2 3, Monotone (fun x => x^2 - 2*a*x + 1) ∨ StrictMono (fun x => x^2 - 2*a*x + 1)) ↔
  (a ≤ 2 ∨ a ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l319_31940


namespace NUMINAMATH_CALUDE_race_probability_l319_31942

theorem race_probability (total_cars : ℕ) (prob_Y prob_Z prob_XYZ : ℝ) : 
  total_cars = 18 → 
  prob_Y = 1/12 → 
  prob_Z = 1/6 → 
  prob_XYZ = 0.375 → 
  ∃ prob_X : ℝ, 
    prob_X + prob_Y + prob_Z = prob_XYZ ∧ 
    prob_X = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_race_probability_l319_31942


namespace NUMINAMATH_CALUDE_jimmy_action_figures_sale_discount_l319_31956

theorem jimmy_action_figures_sale_discount (total_figures : ℕ) 
  (regular_figure_value : ℚ) (special_figure_value : ℚ) (total_earned : ℚ) :
  total_figures = 5 →
  regular_figure_value = 15 →
  special_figure_value = 20 →
  total_earned = 55 →
  (4 * regular_figure_value + special_figure_value - total_earned) / total_figures = 5 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_action_figures_sale_discount_l319_31956


namespace NUMINAMATH_CALUDE_simple_interest_problem_l319_31978

theorem simple_interest_problem (simple_interest rate time : ℝ) 
  (h1 : simple_interest = 100)
  (h2 : rate = 5)
  (h3 : time = 4) :
  simple_interest = (500 * rate * time) / 100 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l319_31978


namespace NUMINAMATH_CALUDE_hyperbola_vector_sum_magnitude_l319_31901

/-- Represents a hyperbola with foci and a point on it -/
structure Hyperbola where
  b : ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ
  b_pos : b > 0
  on_hyperbola : P.1^2 / 4 - P.2^2 / b^2 = 1
  asymptote : b / 2 = Real.sqrt 6 / 2
  focal_ratio : dist P F₁ / dist P F₂ = 3

/-- The sum of vectors from a point on the hyperbola to its foci has magnitude 2√10 -/
theorem hyperbola_vector_sum_magnitude (h : Hyperbola) : 
  Real.sqrt ((h.P.1 - h.F₁.1)^2 + (h.P.2 - h.F₁.2)^2 + (h.P.1 - h.F₂.1)^2 + (h.P.2 - h.F₂.2)^2 + 
             2 * ((h.P.1 - h.F₁.1) * (h.P.1 - h.F₂.1) + (h.P.2 - h.F₁.2) * (h.P.2 - h.F₂.2))) = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_vector_sum_magnitude_l319_31901


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l319_31950

/-- Given a geometric sequence {a_n} and its partial sums S_n -/
def geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ ∀ n, a (n + 1) = r * a n

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  geometric_sequence a S →
  32 * a 2 + a 7 = 0 →
  S 5 / S 2 = -11 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l319_31950
