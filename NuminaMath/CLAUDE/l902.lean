import Mathlib

namespace NUMINAMATH_CALUDE_no_real_a_with_unique_solution_l902_90244

-- Define the function f(x) = x^2 + ax + 2a
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2*a

-- Define the property that |f(x)| ≤ 5 has exactly one solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, |f a x| ≤ 5

-- Theorem statement
theorem no_real_a_with_unique_solution :
  ¬∃ a : ℝ, has_unique_solution a :=
sorry

end NUMINAMATH_CALUDE_no_real_a_with_unique_solution_l902_90244


namespace NUMINAMATH_CALUDE_greatest_difference_l902_90206

/-- A type representing a chessboard arrangement of numbers 1 to 400 -/
def Arrangement := Fin 20 → Fin 20 → Fin 400

/-- The property that an arrangement has two numbers in the same row or column differing by at least N -/
def HasDifference (arr : Arrangement) (N : ℕ) : Prop :=
  ∃ (i j k : Fin 20), (arr i j).val + N ≤ (arr i k).val ∨ (arr j i).val + N ≤ (arr k i).val

/-- The theorem stating that 209 is the greatest natural number satisfying the given condition -/
theorem greatest_difference : 
  (∀ (arr : Arrangement), HasDifference arr 209) ∧ 
  ¬(∀ (arr : Arrangement), HasDifference arr 210) :=
sorry

end NUMINAMATH_CALUDE_greatest_difference_l902_90206


namespace NUMINAMATH_CALUDE_gcd_f_x_l902_90284

def f (x : ℤ) : ℤ := (3*x+4)*(7*x+1)*(13*x+6)*(2*x+9)

theorem gcd_f_x (x : ℤ) (h : ∃ k : ℤ, x = 15336 * k) : 
  Nat.gcd (Int.natAbs (f x)) (Int.natAbs x) = 216 := by
  sorry

end NUMINAMATH_CALUDE_gcd_f_x_l902_90284


namespace NUMINAMATH_CALUDE_type_a_sample_size_l902_90254

/-- Represents the ratio of quantities for product types A, B, and C -/
structure ProductRatio :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Calculates the number of units to be selected for a given product type -/
def unitsToSelect (total : ℕ) (sampleSize : ℕ) (ratio : ProductRatio) (typeRatio : ℕ) : ℕ :=
  (sampleSize * typeRatio) / (ratio.a + ratio.b + ratio.c)

theorem type_a_sample_size 
  (totalProduction : ℕ)
  (sampleSize : ℕ)
  (ratio : ProductRatio)
  (h1 : totalProduction = 600)
  (h2 : sampleSize = 120)
  (h3 : ratio = ⟨1, 2, 3⟩) :
  unitsToSelect totalProduction sampleSize ratio ratio.a = 20 := by
  sorry

end NUMINAMATH_CALUDE_type_a_sample_size_l902_90254


namespace NUMINAMATH_CALUDE_probability_all_odd_is_one_forty_second_l902_90264

def total_slips : ℕ := 10
def odd_slips : ℕ := 5
def draws : ℕ := 4

def probability_all_odd : ℚ := (odd_slips.choose draws) / (total_slips.choose draws)

theorem probability_all_odd_is_one_forty_second :
  probability_all_odd = 1 / 42 := by sorry

end NUMINAMATH_CALUDE_probability_all_odd_is_one_forty_second_l902_90264


namespace NUMINAMATH_CALUDE_sum_of_roots_l902_90257

theorem sum_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → ∃ y : ℝ, (y + 3) * (y - 4) = 22 ∧ x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l902_90257


namespace NUMINAMATH_CALUDE_faculty_size_l902_90252

/-- The number of second year students studying numeric methods -/
def numeric_methods : ℕ := 250

/-- The number of second year students studying automatic control of airborne vehicles -/
def automatic_control : ℕ := 423

/-- The number of second year students studying both subjects -/
def both_subjects : ℕ := 134

/-- The percentage of second year students in the total student body -/
def second_year_percentage : ℚ := 4/5

/-- The total number of students in the faculty -/
def total_students : ℕ := 674

theorem faculty_size : 
  ∃ (second_year_students : ℕ), 
    second_year_students = numeric_methods + automatic_control - both_subjects ∧
    (second_year_students : ℚ) / total_students = second_year_percentage :=
by sorry

end NUMINAMATH_CALUDE_faculty_size_l902_90252


namespace NUMINAMATH_CALUDE_tonys_fever_degree_l902_90226

/-- Proves that Tony's temperature is 5 degrees above the fever threshold given the conditions --/
theorem tonys_fever_degree (normal_temp : ℝ) (temp_increase : ℝ) (fever_threshold : ℝ) :
  normal_temp = 95 →
  temp_increase = 10 →
  fever_threshold = 100 →
  normal_temp + temp_increase - fever_threshold = 5 := by
  sorry

end NUMINAMATH_CALUDE_tonys_fever_degree_l902_90226


namespace NUMINAMATH_CALUDE_hyperbola_equation_l902_90228

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the condition that the hyperbola passes through (2, 1)
def passes_through_point (a b : ℝ) : Prop := hyperbola a b 2 1

-- Define the condition that the hyperbola and ellipse share the same foci
def same_foci (a b : ℝ) : Prop := a^2 + b^2 = 3

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) 
  (h1 : passes_through_point a b) 
  (h2 : same_foci a b) : 
  ∀ x y : ℝ, hyperbola 2 1 x y := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l902_90228


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l902_90229

theorem triangle_angle_theorem (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < B → B < π →
  a^2 + c^2 = b^2 + a*c →
  B = π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l902_90229


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l902_90267

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l902_90267


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l902_90232

/-- The sum of x-coordinates and y-coordinates of the intersection points of two parabolas -/
theorem intersection_sum_zero (x y : ℝ → ℝ) : 
  (∀ t, y t = (x t - 2)^2) →
  (∀ t, x t + 3 = (y t + 2)^2) →
  (∃ a b c d : ℝ, 
    (y a = (x a - 2)^2 ∧ x a + 3 = (y a + 2)^2) ∧
    (y b = (x b - 2)^2 ∧ x b + 3 = (y b + 2)^2) ∧
    (y c = (x c - 2)^2 ∧ x c + 3 = (y c + 2)^2) ∧
    (y d = (x d - 2)^2 ∧ x d + 3 = (y d + 2)^2) ∧
    (∀ t, y t = (x t - 2)^2 ∧ x t + 3 = (y t + 2)^2 → t = a ∨ t = b ∨ t = c ∨ t = d)) →
  x a + x b + x c + x d + y a + y b + y c + y d = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_zero_l902_90232


namespace NUMINAMATH_CALUDE_fold_paper_sum_l902_90231

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if two points are symmetric about a given line -/
def areSymmetric (p1 p2 : Point) (l : Line) : Prop :=
  -- Definition of symmetry about a line
  sorry

/-- Finds the fold line given two pairs of symmetric points -/
def findFoldLine (p1 p2 p3 p4 : Point) : Line :=
  -- Definition to find the fold line
  sorry

/-- Main theorem -/
theorem fold_paper_sum (m n : ℝ) :
  let p1 : Point := ⟨0, 2⟩
  let p2 : Point := ⟨4, 0⟩
  let p3 : Point := ⟨9, 5⟩
  let p4 : Point := ⟨m, n⟩
  let foldLine := findFoldLine p1 p2 p3 p4
  areSymmetric p1 p2 foldLine ∧ areSymmetric p3 p4 foldLine →
  m + n = 10 := by
  sorry

end NUMINAMATH_CALUDE_fold_paper_sum_l902_90231


namespace NUMINAMATH_CALUDE_waiter_customers_l902_90223

/-- Given a number of customers who left and the number of remaining customers,
    calculate the initial number of customers. -/
def initial_customers (left : ℕ) (remaining : ℕ) : ℕ := left + remaining

/-- Theorem: Given that 9 customers left and 12 remained, 
    prove that there were initially 21 customers. -/
theorem waiter_customers : initial_customers 9 12 = 21 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l902_90223


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l902_90221

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) = (a^2 - 3*a + 2) + Complex.I * (a - 1)) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l902_90221


namespace NUMINAMATH_CALUDE_inequalities_hold_l902_90276

theorem inequalities_hold (a b c x y z : ℝ) 
  (h1 : x^2 < a) (h2 : y^2 < b) (h3 : z^2 < c) : 
  (x^2*y^2 + y^2*z^2 + z^2*x^2 < a*b + b*c + c*a) ∧ 
  (x^4 + y^4 + z^4 < a^2 + b^2 + c^2) ∧ 
  (x^2*y^2*z^2 < a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l902_90276


namespace NUMINAMATH_CALUDE_smallest_integer_a_for_unique_solution_l902_90263

-- Define the system of equations
def equation1 (x y a : ℝ) : Prop := y / (a - Real.sqrt x - 1) = 4
def equation2 (x y : ℝ) : Prop := y = (Real.sqrt x + 5) / (Real.sqrt x + 1)

-- Define the property of having a unique solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! (x y : ℝ), equation1 x y a ∧ equation2 x y

-- State the theorem
theorem smallest_integer_a_for_unique_solution :
  (∀ a : ℤ, a < 3 → ¬(has_unique_solution (a : ℝ))) ∧
  has_unique_solution 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_a_for_unique_solution_l902_90263


namespace NUMINAMATH_CALUDE_parallel_line_vector_l902_90280

theorem parallel_line_vector (m : ℝ) : 
  (∀ x y : ℝ, m * x + 2 * y + 6 = 0 → (1 - m) * y = x) → 
  m = -1 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_vector_l902_90280


namespace NUMINAMATH_CALUDE_jason_picked_46_pears_l902_90260

/-- Calculates the number of pears Jason picked given the number of pears Keith picked,
    the number of pears Mike ate, and the number of pears left. -/
def jasons_pears (keith_pears mike_ate pears_left : ℕ) : ℕ :=
  (mike_ate + pears_left) - keith_pears

/-- Proves that Jason picked 46 pears given the problem conditions. -/
theorem jason_picked_46_pears :
  jasons_pears 47 12 81 = 46 := by
  sorry

end NUMINAMATH_CALUDE_jason_picked_46_pears_l902_90260


namespace NUMINAMATH_CALUDE_green_blue_difference_l902_90271

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ
  color_sum : blue + yellow + green = total
  ratio : blue * 18 = total * 3 ∧ yellow * 18 = total * 7 ∧ green * 18 = total * 8

theorem green_blue_difference (bag : DiskBag) (h : bag.total = 144) :
  bag.green - bag.blue = 40 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_l902_90271


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l902_90235

theorem quadratic_roots_property (a b : ℝ) : 
  a^2 - 2*a - 1 = 0 → b^2 - 2*b - 1 = 0 → a^2 + 2*b - a*b = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l902_90235


namespace NUMINAMATH_CALUDE_percentage_calculation_l902_90281

theorem percentage_calculation (P : ℝ) : 
  (3/5 : ℝ) * 120 * (P/100) = 36 → P = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l902_90281


namespace NUMINAMATH_CALUDE_oranges_thrown_away_l902_90239

theorem oranges_thrown_away (initial : ℕ) (added : ℕ) (final : ℕ) :
  initial = 40 →
  added = 7 →
  final = 10 →
  initial - (initial - final + added) = 37 := by
sorry

end NUMINAMATH_CALUDE_oranges_thrown_away_l902_90239


namespace NUMINAMATH_CALUDE_sin_double_angle_shift_graph_shift_equivalent_graphs_l902_90234

theorem sin_double_angle_shift (x : ℝ) :
  2 * Real.sin (x + π / 6) * Real.cos (x + π / 6) = Real.sin (2 * (x + π / 6)) := by sorry

theorem graph_shift (x : ℝ) :
  2 * Real.sin (x + π / 6) * Real.cos (x + π / 6) = Real.sin (2 * x + π / 3) := by sorry

theorem equivalent_graphs :
  ∀ x : ℝ, 2 * Real.sin (x + π / 6) * Real.cos (x + π / 6) = Real.sin (2 * (x + π / 6)) := by sorry

end NUMINAMATH_CALUDE_sin_double_angle_shift_graph_shift_equivalent_graphs_l902_90234


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l902_90256

/-- Represents the orientation of a stripe on a cube face -/
inductive StripeOrientation
  | EdgeToEdge1
  | EdgeToEdge2
  | Diagonal1
  | Diagonal2

/-- Represents a cube with stripes on its faces -/
structure StripedCube :=
  (faces : Fin 6 → StripeOrientation)

/-- Checks if a given StripedCube has a continuous stripe encircling it -/
def hasContinuousStripe (cube : StripedCube) : Bool :=
  sorry

/-- The total number of possible stripe combinations -/
def totalCombinations : Nat :=
  4^6

/-- The number of stripe combinations that result in a continuous stripe -/
def favorableCombinations : Nat :=
  3 * 4

/-- The probability of a continuous stripe encircling the cube -/
def probabilityOfContinuousStripe : Rat :=
  favorableCombinations / totalCombinations

theorem continuous_stripe_probability :
  probabilityOfContinuousStripe = 3 / 1024 :=
sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l902_90256


namespace NUMINAMATH_CALUDE_constant_term_of_product_l902_90238

def is_monic (p : Polynomial ℝ) : Prop :=
  p.leadingCoeff = 1

theorem constant_term_of_product (p q : Polynomial ℝ) :
  is_monic p →
  is_monic q →
  p.degree = 3 →
  q.degree = 3 →
  (∃ c : ℝ, c > 0 ∧ p.coeff 0 = c ∧ q.coeff 0 = c) →
  (∃ a : ℝ, p.coeff 1 = a ∧ q.coeff 1 = a) →
  p * q = Polynomial.monomial 6 1 + Polynomial.monomial 5 2 + Polynomial.monomial 4 1 +
          Polynomial.monomial 3 2 + Polynomial.monomial 2 9 + Polynomial.monomial 1 12 +
          Polynomial.monomial 0 36 →
  p.coeff 0 = 6 ∧ q.coeff 0 = 6 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_of_product_l902_90238


namespace NUMINAMATH_CALUDE_ella_toast_combinations_l902_90241

/-- The number of different kinds of spreads -/
def num_spreads : ℕ := 12

/-- The number of different kinds of toppings -/
def num_toppings : ℕ := 8

/-- The number of types of bread -/
def num_breads : ℕ := 3

/-- The number of spreads chosen for each toast -/
def spreads_per_toast : ℕ := 1

/-- The number of toppings chosen for each toast -/
def toppings_per_toast : ℕ := 2

/-- The number of breads chosen for each toast -/
def breads_per_toast : ℕ := 1

/-- The total number of different toasts Ella can make -/
def total_toasts : ℕ := num_spreads * (num_toppings.choose toppings_per_toast) * num_breads

theorem ella_toast_combinations :
  total_toasts = 1008 := by sorry

end NUMINAMATH_CALUDE_ella_toast_combinations_l902_90241


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l902_90292

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l902_90292


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l902_90242

-- Define ticket prices
def adult_price : ℝ := 11
def child_price : ℝ := 8
def senior_price : ℝ := 9

-- Define discounts
def husband_discount : ℝ := 0.25
def parents_discount : ℝ := 0.15
def nephew_discount : ℝ := 0.10

-- Define group composition
def num_adults : ℕ := 4
def num_children : ℕ := 2
def num_seniors : ℕ := 3
def num_teens : ℕ := 1
def num_adult_nephews : ℕ := 1

-- Define the total cost function
def total_cost : ℝ :=
  (num_adults * adult_price) +
  (num_children * child_price) +
  (num_seniors * senior_price) +
  (num_teens * adult_price) +
  (num_adult_nephews * adult_price) -
  (husband_discount * adult_price) -
  (parents_discount * (2 * senior_price)) -
  (nephew_discount * adult_price)

-- Theorem statement
theorem total_cost_is_correct :
  total_cost = 110.45 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l902_90242


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_implies_m_range_l902_90237

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the property of monotonically decreasing
def monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

-- State the theorem
theorem f_monotone_decreasing_implies_m_range (m : ℝ) :
  monotone_decreasing f (2*m) (m+1) → m ∈ Set.Icc (-1) 1 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_implies_m_range_l902_90237


namespace NUMINAMATH_CALUDE_curve_scaling_transformation_l902_90289

/-- Given a curve C that undergoes a scaling transformation,
    prove that the equation of the original curve is x^2/4 + 9y^2 = 1 -/
theorem curve_scaling_transformation (x y x' y' : ℝ) :
  (x' = 1/2 * x) →
  (y' = 3 * y) →
  (x'^2 + y'^2 = 1) →
  (x^2/4 + 9*y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_curve_scaling_transformation_l902_90289


namespace NUMINAMATH_CALUDE_quiz_score_impossibility_l902_90247

theorem quiz_score_impossibility :
  ∀ (c u i : ℕ),
    c + u + i = 25 →
    4 * c + 2 * u - i ≠ 79 :=
by
  sorry

end NUMINAMATH_CALUDE_quiz_score_impossibility_l902_90247


namespace NUMINAMATH_CALUDE_square_root_calculations_l902_90202

theorem square_root_calculations :
  (Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0) ∧
  (Real.sqrt 6 / Real.sqrt 18 * Real.sqrt 27 = 3) := by
  sorry

end NUMINAMATH_CALUDE_square_root_calculations_l902_90202


namespace NUMINAMATH_CALUDE_exists_question_with_different_answers_l902_90243

/-- Represents a person who always tells the truth -/
structure TruthfulPerson where
  answer : Prop → Bool
  always_truthful : ∀ p, answer p = p

/-- Represents a question that can be asked -/
structure Question where
  ask : TruthfulPerson → Bool

/-- Represents the state of a day, including whether any questions have been asked -/
structure DayState where
  question_asked : Bool

/-- The theorem stating that there exists a question that yields different answers when asked twice -/
theorem exists_question_with_different_answers :
  ∃ (q : Question), ∀ (p : TruthfulPerson),
    ∃ (d1 d2 : DayState),
      d1.question_asked = false ∧
      d2.question_asked = true ∧
      q.ask p ≠ q.ask p :=
sorry

end NUMINAMATH_CALUDE_exists_question_with_different_answers_l902_90243


namespace NUMINAMATH_CALUDE_problem_statement_l902_90262

theorem problem_statement (a : ℝ) (h_pos : a > 0) (h_eq : a^2 / (a^4 - a^2 + 1) = 4/37) :
  a^3 / (a^6 - a^3 + 1) = 8/251 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l902_90262


namespace NUMINAMATH_CALUDE_f_odd_f_2a_f_3a_f_monotone_decreasing_l902_90291

/-- Function with specific properties -/
noncomputable def f (x : ℝ) : ℝ := sorry

/-- Positive constant a -/
noncomputable def a : ℝ := sorry

/-- Domain of f -/
def domain (x : ℝ) : Prop := ∀ k : ℤ, x ≠ k * Real.pi

axiom f_domain : ∀ x : ℝ, domain x → f x ≠ 0

axiom f_equation : ∀ x y : ℝ, domain x → domain y → 
  f (x - y) = (f x * f y + 1) / (f y - f x)

axiom f_a : f a = 1

axiom a_pos : a > 0

axiom f_pos_interval : ∀ x : ℝ, 0 < x → x < 2 * a → f x > 0

/-- f is an odd function -/
theorem f_odd : ∀ x : ℝ, domain x → f (-x) = -f x := by sorry

/-- f(2a) = 0 -/
theorem f_2a : f (2 * a) = 0 := by sorry

/-- f(3a) = -1 -/
theorem f_3a : f (3 * a) = -1 := by sorry

/-- f is monotonically decreasing on [2a, 3a] -/
theorem f_monotone_decreasing : 
  ∀ x y : ℝ, 2 * a ≤ x → x < y → y ≤ 3 * a → f x > f y := by sorry

end NUMINAMATH_CALUDE_f_odd_f_2a_f_3a_f_monotone_decreasing_l902_90291


namespace NUMINAMATH_CALUDE_variety_show_probability_l902_90236

/-- The probability of selecting exactly one boy who likes variety shows
    when randomly choosing two boys from a group of five, where two like
    variety shows and three do not. -/
theorem variety_show_probability :
  let total_boys : ℕ := 5
  let boys_like_shows : ℕ := 2
  let boys_dislike_shows : ℕ := 3
  let selected_boys : ℕ := 2
  
  boys_like_shows + boys_dislike_shows = total_boys →
  
  (Nat.choose total_boys selected_boys : ℚ) ≠ 0 →
  
  (Nat.choose boys_like_shows 1 * Nat.choose boys_dislike_shows 1 : ℚ) /
  (Nat.choose total_boys selected_boys : ℚ) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_variety_show_probability_l902_90236


namespace NUMINAMATH_CALUDE_family_savings_l902_90273

def income : ℕ := 509600
def expenses : ℕ := 276000
def initial_savings : ℕ := 1147240

theorem family_savings : initial_savings + income - expenses = 1340840 := by
  sorry

end NUMINAMATH_CALUDE_family_savings_l902_90273


namespace NUMINAMATH_CALUDE_quadratic_inequality_nonnegative_l902_90209

theorem quadratic_inequality_nonnegative (x : ℝ) : x^2 - x + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_nonnegative_l902_90209


namespace NUMINAMATH_CALUDE_notebook_puzzle_l902_90210

/-- Represents a set of statements where the i-th statement claims 
    "There are exactly i false statements in this set" --/
def StatementSet (n : ℕ) := Fin n → Prop

/-- The property that exactly one statement in the set is true --/
def ExactlyOneTrue (s : StatementSet n) : Prop :=
  ∃! i, s i

/-- The i-th statement claims there are exactly i false statements --/
def StatementClaim (s : StatementSet n) (i : Fin n) : Prop :=
  s i ↔ (∃ k : Fin n, k.val = n - i.val ∧ (∀ j : Fin n, s j ↔ j = k))

/-- The main theorem --/
theorem notebook_puzzle :
  ∀ (s : StatementSet 100),
    (∀ i, StatementClaim s i) →
    ExactlyOneTrue s →
    s ⟨99, by norm_num⟩ :=
by sorry

end NUMINAMATH_CALUDE_notebook_puzzle_l902_90210


namespace NUMINAMATH_CALUDE_prob_not_all_same_eight_sided_dice_l902_90286

theorem prob_not_all_same_eight_sided_dice (n : ℕ) (h : n = 5) :
  (1 - (8 : ℚ) / 8^n) = 4095 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_all_same_eight_sided_dice_l902_90286


namespace NUMINAMATH_CALUDE_tens_digit_of_19_power_1987_l902_90268

theorem tens_digit_of_19_power_1987 : ∃ n : ℕ, 19^1987 ≡ 30 + n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_19_power_1987_l902_90268


namespace NUMINAMATH_CALUDE_carol_weight_l902_90283

/-- Given that the sum of Alice's and Carol's weights is 220 pounds, and the difference
    between Carol's and Alice's weights is one-third of Carol's weight plus 10 pounds,
    prove that Carol weighs 138 pounds. -/
theorem carol_weight (alice_weight carol_weight : ℝ) 
  (h1 : alice_weight + carol_weight = 220)
  (h2 : carol_weight - alice_weight = (1/3) * carol_weight + 10) : 
  carol_weight = 138 := by
sorry

end NUMINAMATH_CALUDE_carol_weight_l902_90283


namespace NUMINAMATH_CALUDE_mean_height_of_volleyball_team_l902_90218

def volleyball_heights : List ℕ := [58, 59, 60, 61, 62, 65, 65, 66, 67, 70, 71, 71, 72, 74, 75, 79, 79]

theorem mean_height_of_volleyball_team (heights : List ℕ) (h1 : heights = volleyball_heights) :
  (heights.sum / heights.length : ℚ) = 68 := by
  sorry

end NUMINAMATH_CALUDE_mean_height_of_volleyball_team_l902_90218


namespace NUMINAMATH_CALUDE_nancy_garden_seeds_l902_90266

theorem nancy_garden_seeds (total_seeds : ℕ) (big_garden_seeds : ℕ) (small_gardens : ℕ) 
  (h1 : total_seeds = 52)
  (h2 : big_garden_seeds = 28)
  (h3 : small_gardens = 6)
  (h4 : big_garden_seeds ≤ total_seeds) :
  (total_seeds - big_garden_seeds) / small_gardens = 4 := by
  sorry

end NUMINAMATH_CALUDE_nancy_garden_seeds_l902_90266


namespace NUMINAMATH_CALUDE_public_transport_support_percentage_l902_90203

theorem public_transport_support_percentage
  (gov_employees : ℕ) (gov_support_rate : ℚ)
  (citizens : ℕ) (citizen_support_rate : ℚ) :
  gov_employees = 150 →
  gov_support_rate = 70 / 100 →
  citizens = 800 →
  citizen_support_rate = 60 / 100 →
  let total_surveyed := gov_employees + citizens
  let total_supporters := gov_employees * gov_support_rate + citizens * citizen_support_rate
  (total_supporters / total_surveyed : ℚ) = 6158 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_public_transport_support_percentage_l902_90203


namespace NUMINAMATH_CALUDE_equality_condition_l902_90224

theorem equality_condition (a b c : ℝ) :
  2 * a + 3 * b * c = (a + 2 * b) * (2 * a + 3 * c) ↔ a = 0 ∨ a + 2 * b + 1.5 * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l902_90224


namespace NUMINAMATH_CALUDE_power_product_equality_l902_90253

theorem power_product_equality (m n : ℝ) : (m * n)^2 = m^2 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l902_90253


namespace NUMINAMATH_CALUDE_is_circle_center_l902_90201

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 5 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, 1)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_is_circle_center_l902_90201


namespace NUMINAMATH_CALUDE_monotone_increasing_interval_minimum_m_for_inequality_l902_90295

noncomputable section

def f (m : ℝ) (x : ℝ) := Real.log x - m * x^2
def g (m : ℝ) (x : ℝ) := (1/2) * m * x^2 + x

theorem monotone_increasing_interval (x : ℝ) :
  StrictMonoOn (f (1/2)) (Set.Ioo 0 1) := by sorry

theorem minimum_m_for_inequality :
  ∀ m : ℕ, (∀ x : ℝ, x > 0 → f m x + g m x ≤ m * x - 1) →
  m ≥ 2 := by sorry

end

end NUMINAMATH_CALUDE_monotone_increasing_interval_minimum_m_for_inequality_l902_90295


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l902_90227

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose n 3 = Nat.choose (n - 1) 3 + Nat.choose (n - 1) 4) → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l902_90227


namespace NUMINAMATH_CALUDE_farrah_order_proof_l902_90211

/-- The number of matchboxes in each box -/
def matchboxes_per_box : ℕ := 20

/-- The number of match sticks in each matchbox -/
def sticks_per_matchbox : ℕ := 300

/-- The total number of match sticks ordered -/
def total_sticks : ℕ := 24000

/-- The number of boxes Farrah ordered -/
def boxes_ordered : ℕ := total_sticks / (matchboxes_per_box * sticks_per_matchbox)

theorem farrah_order_proof : boxes_ordered = 4 := by
  sorry

end NUMINAMATH_CALUDE_farrah_order_proof_l902_90211


namespace NUMINAMATH_CALUDE_april_largest_difference_l902_90204

/-- Represents the months of cookie sales --/
inductive Month
| january
| february
| march
| april
| may

/-- Calculates the percentage difference between two sales values --/
def percentageDifference (x y : ℕ) : ℚ :=
  (max x y - min x y : ℚ) / (min x y : ℚ) * 100

/-- Returns the sales data for Rangers and Scouts for a given month --/
def salesData (m : Month) : ℕ × ℕ :=
  match m with
  | .january => (5, 4)
  | .february => (6, 4)
  | .march => (5, 5)
  | .april => (7, 4)
  | .may => (3, 5)

/-- Theorem: April has the largest percentage difference in cookie sales --/
theorem april_largest_difference :
  ∀ m : Month, m ≠ Month.april →
    percentageDifference (salesData Month.april).1 (salesData Month.april).2 ≥
    percentageDifference (salesData m).1 (salesData m).2 :=
by sorry

end NUMINAMATH_CALUDE_april_largest_difference_l902_90204


namespace NUMINAMATH_CALUDE_largest_circle_area_l902_90275

theorem largest_circle_area (length width : ℝ) (h1 : length = 18) (h2 : width = 8) :
  let perimeter := 2 * (length + width)
  let radius := perimeter / (2 * Real.pi)
  (Real.pi * radius ^ 2) = 676 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_largest_circle_area_l902_90275


namespace NUMINAMATH_CALUDE_factorial_starts_with_1966_l902_90274

theorem factorial_starts_with_1966 : ∃ k : ℕ, ∃ n : ℕ, 
  1966 * 10^n ≤ k! ∧ k! < 1967 * 10^n :=
sorry

end NUMINAMATH_CALUDE_factorial_starts_with_1966_l902_90274


namespace NUMINAMATH_CALUDE_negation_of_p_l902_90240

def p (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

theorem negation_of_p (f : ℝ → ℝ) :
  ¬(p f) ↔ ∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_p_l902_90240


namespace NUMINAMATH_CALUDE_unique_solution_l902_90287

/-- Sum of digits function for positive integers in base 10 -/
def S (n : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that 17 is the only positive integer solution to the equation -/
theorem unique_solution : ∀ n : ℕ+, (n : ℕ)^3 = 8 * (S n)^3 + 6 * (S n) * (n : ℕ) + 1 ↔ n = 17 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l902_90287


namespace NUMINAMATH_CALUDE_f_is_quadratic_l902_90230

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The specific equation we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 + x - 5

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l902_90230


namespace NUMINAMATH_CALUDE_count_multiples_of_four_l902_90272

theorem count_multiples_of_four : ∃ (n : ℕ), n = (Finset.filter (fun x => x % 4 = 0 ∧ x > 300 ∧ x < 700) (Finset.range 700)).card ∧ n = 99 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_four_l902_90272


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l902_90208

/-- 
Given a right triangle PQR with legs PQ and PR, where U is on PQ and V is on PR,
prove that if PU:UQ = PV:VR = 1:3, QU = 18 units, and RV = 45 units, 
then the length of the hypotenuse QR is 12√29 units.
-/
theorem right_triangle_hypotenuse (P Q R U V : ℝ × ℝ) : 
  let pq := ‖Q - P‖
  let pr := ‖R - P‖
  let qu := ‖U - Q‖
  let rv := ‖V - R‖
  let qr := ‖R - Q‖
  (P.1 - Q.1) * (R.2 - P.2) = (P.2 - Q.2) * (R.1 - P.1) → -- right angle at P
  (∃ t : ℝ, t > 0 ∧ t < 1 ∧ U = t • P + (1 - t) • Q) → -- U is on PQ
  (∃ s : ℝ, s > 0 ∧ s < 1 ∧ V = s • P + (1 - s) • R) → -- V is on PR
  ‖P - U‖ / ‖U - Q‖ = 1 / 3 → -- PU:UQ = 1:3
  ‖P - V‖ / ‖V - R‖ = 1 / 3 → -- PV:VR = 1:3
  qu = 18 →
  rv = 45 →
  qr = 12 * Real.sqrt 29 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l902_90208


namespace NUMINAMATH_CALUDE_overtime_hours_is_eight_l902_90296

/-- Calculates overtime hours given regular pay rate, regular hours, overtime rate multiplier, and total pay -/
def calculate_overtime_hours (regular_rate : ℚ) (regular_hours : ℚ) (overtime_multiplier : ℚ) (total_pay : ℚ) : ℚ :=
  let regular_pay := regular_rate * regular_hours
  let overtime_rate := regular_rate * overtime_multiplier
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_rate

/-- Proves that given the problem conditions, the number of overtime hours is 8 -/
theorem overtime_hours_is_eight :
  let regular_rate : ℚ := 3
  let regular_hours : ℚ := 40
  let overtime_multiplier : ℚ := 2
  let total_pay : ℚ := 168
  calculate_overtime_hours regular_rate regular_hours overtime_multiplier total_pay = 8 := by
  sorry

#eval calculate_overtime_hours 3 40 2 168

end NUMINAMATH_CALUDE_overtime_hours_is_eight_l902_90296


namespace NUMINAMATH_CALUDE_thankYouCards_count_l902_90246

/-- Represents the number of items to be mailed --/
structure MailItems where
  thankYouCards : ℕ
  bills : ℕ
  rebates : ℕ
  jobApplications : ℕ

/-- Calculates the total number of stamps required --/
def totalStamps (items : MailItems) : ℕ :=
  items.thankYouCards + items.bills + 1 + items.rebates + items.jobApplications

/-- Theorem stating the number of thank you cards --/
theorem thankYouCards_count (items : MailItems) : 
  items.bills = 2 ∧ 
  items.rebates = items.bills + 3 ∧ 
  items.jobApplications = 2 * items.rebates ∧
  totalStamps items = 21 →
  items.thankYouCards = 3 := by
  sorry

end NUMINAMATH_CALUDE_thankYouCards_count_l902_90246


namespace NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l902_90249

theorem max_sum_given_sum_of_squares_and_product (x y : ℝ) :
  x^2 + y^2 = 98 → xy = 36 → x + y ≤ Real.sqrt 170 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l902_90249


namespace NUMINAMATH_CALUDE_intersection_point_l902_90265

/-- The slope of the first line -/
def m : ℚ := 3

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := y = m * x + 2

/-- The point through which the perpendicular line passes -/
def point : ℚ × ℚ := (3, 4)

/-- The slope of the perpendicular line -/
def m_perp : ℚ := -1 / m

/-- The perpendicular line equation -/
def line2 (x y : ℚ) : Prop := y - point.2 = m_perp * (x - point.1)

/-- The intersection point -/
def intersection : ℚ × ℚ := (9/10, 47/10)

theorem intersection_point : 
  line1 intersection.1 intersection.2 ∧ 
  line2 intersection.1 intersection.2 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l902_90265


namespace NUMINAMATH_CALUDE_valleyball_league_members_l902_90285

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The cost of a T-shirt in dollars -/
def tshirt_cost : ℕ := sock_cost + 7

/-- The cost of a cap in dollars -/
def cap_cost : ℕ := tshirt_cost

/-- The cost of equipment for home games per member in dollars -/
def home_cost : ℕ := sock_cost + tshirt_cost

/-- The cost of equipment for away games per member in dollars -/
def away_cost : ℕ := sock_cost + tshirt_cost + cap_cost

/-- The total cost of equipment per member in dollars -/
def member_cost : ℕ := home_cost + away_cost

/-- The total cost of equipment for all members in dollars -/
def total_cost : ℕ := 4324

theorem valleyball_league_members : 
  ∃ n : ℕ, n * member_cost ≤ total_cost ∧ total_cost < (n + 1) * member_cost ∧ n = 85 := by
  sorry

end NUMINAMATH_CALUDE_valleyball_league_members_l902_90285


namespace NUMINAMATH_CALUDE_bar_chart_suitable_for_rope_skipping_l902_90216

/-- Represents different types of statistical charts -/
inductive StatisticalChart
  | BarChart
  | LineChart
  | PieChart

/-- Represents a dataset of rope skipping scores -/
structure RopeSkippingData where
  scores : List Nat

/-- Defines the property of a chart being suitable for representing discrete data points -/
def suitableForDiscreteData (chart : StatisticalChart) : Prop :=
  match chart with
  | StatisticalChart.BarChart => True
  | _ => False

/-- Theorem stating that a bar chart is suitable for representing rope skipping scores -/
theorem bar_chart_suitable_for_rope_skipping (data : RopeSkippingData) :
  suitableForDiscreteData StatisticalChart.BarChart :=
by sorry

end NUMINAMATH_CALUDE_bar_chart_suitable_for_rope_skipping_l902_90216


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l902_90299

theorem quadratic_one_solution_sum (a : ℝ) : 
  let f : ℝ → ℝ := λ x => 9*x^2 + a*x + 12*x + 16
  let discriminant := (a + 12)^2 - 4*9*16
  (∃! x, f x = 0) → 
  (∃ a₁ a₂, discriminant = 0 ∧ a = a₁ ∨ a = a₂ ∧ a₁ + a₂ = -24) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l902_90299


namespace NUMINAMATH_CALUDE_apple_ratio_l902_90215

/-- Prove that the ratio of Harry's apples to Tim's apples is 1:2 -/
theorem apple_ratio :
  ∀ (martha_apples tim_apples harry_apples : ℕ),
    martha_apples = 68 →
    tim_apples = martha_apples - 30 →
    harry_apples = 19 →
    (harry_apples : ℚ) / tim_apples = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_ratio_l902_90215


namespace NUMINAMATH_CALUDE_adam_apple_purchase_l902_90213

/-- The total quantity of apples Adam bought over three days -/
def total_apples (monday_apples : ℝ) : ℝ :=
  let tuesday_apples := monday_apples * 3.2
  let wednesday_apples := tuesday_apples * 1.05
  monday_apples + tuesday_apples + wednesday_apples

/-- Theorem stating the total quantity of apples Adam bought -/
theorem adam_apple_purchase :
  total_apples 15.5 = 117.18 := by
  sorry

end NUMINAMATH_CALUDE_adam_apple_purchase_l902_90213


namespace NUMINAMATH_CALUDE_gcf_of_270_108_150_l902_90222

theorem gcf_of_270_108_150 : Nat.gcd 270 (Nat.gcd 108 150) = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_270_108_150_l902_90222


namespace NUMINAMATH_CALUDE_roberto_final_salary_l902_90233

/-- Calculates the final salary after raises, bonus, and taxes -/
def final_salary (starting_salary : ℝ) (first_raise_percent : ℝ) (second_raise_percent : ℝ) (bonus : ℝ) (tax_rate : ℝ) : ℝ :=
  let previous_salary := starting_salary * (1 + first_raise_percent)
  let current_salary := previous_salary * (1 + second_raise_percent)
  let total_income := current_salary + bonus
  let taxes := total_income * tax_rate
  total_income - taxes

/-- Theorem stating that Roberto's final salary is $104,550 -/
theorem roberto_final_salary :
  final_salary 80000 0.4 0.2 5000 0.25 = 104550 := by
  sorry

end NUMINAMATH_CALUDE_roberto_final_salary_l902_90233


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_l902_90294

theorem dormitory_to_city_distance :
  ∀ D : ℝ,
  (1/5 : ℝ) * D + (2/3 : ℝ) * D + 4 = D →
  D = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_l902_90294


namespace NUMINAMATH_CALUDE_expression_equals_one_l902_90214

def numerator : ℕ → ℚ
  | 0 => 1
  | n + 1 => numerator n * (1 + 18 / (n + 1))

def denominator : ℕ → ℚ
  | 0 => 1
  | n + 1 => denominator n * (1 + 20 / (n + 1))

theorem expression_equals_one :
  (numerator 20) / (denominator 18) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l902_90214


namespace NUMINAMATH_CALUDE_largest_number_proof_l902_90259

theorem largest_number_proof (a b : ℕ+) 
  (hcf_cond : Nat.gcd a b = 42)
  (lcm_cond : Nat.lcm a b = 42 * 11 * 12) :
  max a b = 504 := by
sorry

end NUMINAMATH_CALUDE_largest_number_proof_l902_90259


namespace NUMINAMATH_CALUDE_sum_has_no_real_roots_l902_90212

/-- A quadratic polynomial with integer coefficients. -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Predicate for an acceptable quadratic polynomial. -/
def is_acceptable (p : QuadraticPolynomial) : Prop :=
  abs p.a ≤ 2013 ∧ abs p.b ≤ 2013 ∧ abs p.c ≤ 2013 ∧
  ∃ (r₁ r₂ : ℤ), p.a * r₁^2 + p.b * r₁ + p.c = 0 ∧ p.a * r₂^2 + p.b * r₂ + p.c = 0

/-- The set of all acceptable quadratic polynomials. -/
def acceptable_polynomials : Set QuadraticPolynomial :=
  {p : QuadraticPolynomial | is_acceptable p}

/-- The sum of all acceptable quadratic polynomials. -/
noncomputable def sum_of_acceptable_polynomials : QuadraticPolynomial :=
  sorry

/-- Theorem stating that the sum of all acceptable quadratic polynomials has no real roots. -/
theorem sum_has_no_real_roots :
  ∃ (A C : ℤ), A > 0 ∧ C > 0 ∧
  sum_of_acceptable_polynomials.a = A ∧
  sum_of_acceptable_polynomials.b = 0 ∧
  sum_of_acceptable_polynomials.c = C :=
sorry

end NUMINAMATH_CALUDE_sum_has_no_real_roots_l902_90212


namespace NUMINAMATH_CALUDE_slope_tangent_ln_at_3_l902_90278

/-- The slope of the tangent line to y = ln x at x = 3 is 1/3 -/
theorem slope_tangent_ln_at_3 : 
  let f : ℝ → ℝ := λ x => Real.log x
  HasDerivAt f (1/3) 3 := by sorry

end NUMINAMATH_CALUDE_slope_tangent_ln_at_3_l902_90278


namespace NUMINAMATH_CALUDE_x_value_proof_l902_90261

theorem x_value_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^3 / y = 2)
  (h2 : y^3 / z = 6)
  (h3 : z^3 / x = 9) :
  x = (559872 : ℝ) ^ (1 / 38) :=
by sorry

end NUMINAMATH_CALUDE_x_value_proof_l902_90261


namespace NUMINAMATH_CALUDE_four_lines_max_regions_l902_90200

/-- The maximum number of regions a plane can be divided into by n lines -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Theorem: The maximum number of regions a plane can be divided into by four lines is 11 -/
theorem four_lines_max_regions : max_regions 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_four_lines_max_regions_l902_90200


namespace NUMINAMATH_CALUDE_library_problem_l902_90250

theorem library_problem (total_books : ℕ) (books_per_student : ℕ) 
  (day1_students : ℕ) (day2_students : ℕ) (day4_students : ℕ) :
  total_books = 120 →
  books_per_student = 5 →
  day1_students = 4 →
  day2_students = 5 →
  day4_students = 9 →
  ∃ (day3_students : ℕ),
    day3_students = 6 ∧
    total_books = (day1_students + day2_students + day3_students + day4_students) * books_per_student :=
by sorry

end NUMINAMATH_CALUDE_library_problem_l902_90250


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l902_90217

/-- A piecewise function f defined on ℝ -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then (a - 5) * x - 1 else (x + a) / (x - 1)

/-- Theorem stating that if f is decreasing on ℝ, then a ∈ (-1, 1] -/
theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Ioc (-1) 1 :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l902_90217


namespace NUMINAMATH_CALUDE_vector_subtraction_l902_90290

theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (5, 3) → b = (1, -2) → a - 2 • b = (3, 7) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l902_90290


namespace NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l902_90255

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_squared_minus_8_l902_90255


namespace NUMINAMATH_CALUDE_parabola_shift_l902_90282

/-- Given a parabola y = x^2 + bx + c that is shifted 4 units to the right
    and 3 units down to become y = x^2 - 4x + 3, prove that b = 4 and c = 6. -/
theorem parabola_shift (b c : ℝ) : 
  (∀ x, (x - 4)^2 + b*(x - 4) + c - 3 = x^2 - 4*x + 3) → 
  b = 4 ∧ c = 6 := by
sorry

end NUMINAMATH_CALUDE_parabola_shift_l902_90282


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l902_90297

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) ↔ |a| ≤ 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l902_90297


namespace NUMINAMATH_CALUDE_total_count_is_2552_l902_90225

/-- Represents the total count for a week given the number of items and the counting schedule. -/
def weeklyCount (tiles books windows chairs lightBulbs : ℕ) : ℕ :=
  let monday := tiles * 2 + books * 2 + windows * 2
  let tuesday := tiles * 3 + books * 2 + windows * 1
  let wednesday := chairs * 4 + lightBulbs * 5
  let thursday := tiles * 1 + chairs * 2 + books * 3 + windows * 4 + lightBulbs * 5
  let friday := tiles * 1 + books * 2 + chairs * 2 + windows * 3 + lightBulbs * 3
  monday + tuesday + wednesday + thursday + friday

/-- Theorem stating that the total count for the week is 2552 given the specific item counts. -/
theorem total_count_is_2552 : weeklyCount 60 120 10 80 24 = 2552 := by
  sorry

#eval weeklyCount 60 120 10 80 24

end NUMINAMATH_CALUDE_total_count_is_2552_l902_90225


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l902_90245

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 8)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l902_90245


namespace NUMINAMATH_CALUDE_inequality_proof_l902_90293

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_condition : |Real.sqrt (a * d) - Real.sqrt (b * c)| ≤ 1) : 
  (a * e + b / e) * (c * e + d / e) ≥ 
  (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l902_90293


namespace NUMINAMATH_CALUDE_compare_A_B_l902_90205

theorem compare_A_B (A B : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : (3/4) * A = (4/3) * B) : A > B := by
  sorry

end NUMINAMATH_CALUDE_compare_A_B_l902_90205


namespace NUMINAMATH_CALUDE_marbles_difference_l902_90258

def initial_marbles : ℕ := 7
def lost_marbles : ℕ := 8
def found_marbles : ℕ := 10

theorem marbles_difference : found_marbles - lost_marbles = 2 := by
  sorry

end NUMINAMATH_CALUDE_marbles_difference_l902_90258


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l902_90277

/-- Represents a quadratic equation of the form ax^2 - 3x√3 + b = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ

/-- The discriminant of the quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ := 27 - 4 * eq.a * eq.b

/-- Predicate for real and distinct roots -/
def has_real_distinct_roots (eq : QuadraticEquation) : Prop :=
  discriminant eq ≠ 0 ∧ discriminant eq > 0

theorem quadratic_roots_nature (eq : QuadraticEquation) 
  (h : discriminant eq ≠ 0) : 
  has_real_distinct_roots eq :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l902_90277


namespace NUMINAMATH_CALUDE_g_of_2_eq_6_l902_90279

def g (x : ℝ) : ℝ := x^3 - x

theorem g_of_2_eq_6 : g 2 = 6 := by sorry

end NUMINAMATH_CALUDE_g_of_2_eq_6_l902_90279


namespace NUMINAMATH_CALUDE_a_range_when_A_union_B_is_R_A_union_B_is_R_when_a_in_range_l902_90288

/-- The set A defined by the inequality (x - 1)(x - a) ≥ 0 -/
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}

/-- The set B defined by the inequality x ≥ a - 1 -/
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

/-- Theorem stating that if A ∪ B = ℝ, then a ∈ (-∞, 2] -/
theorem a_range_when_A_union_B_is_R (a : ℝ) 
  (h : A a ∪ B a = Set.univ) : a ≤ 2 := by
  sorry

/-- Theorem stating that if a ∈ (-∞, 2], then A ∪ B = ℝ -/
theorem A_union_B_is_R_when_a_in_range (a : ℝ) 
  (h : a ≤ 2) : A a ∪ B a = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_a_range_when_A_union_B_is_R_A_union_B_is_R_when_a_in_range_l902_90288


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l902_90220

theorem simplify_trig_expression (x : ℝ) : 
  Real.sqrt 2 * Real.cos x + Real.sqrt 6 * Real.sin x = 
  2 * Real.sqrt 2 * Real.cos (π / 3 - x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l902_90220


namespace NUMINAMATH_CALUDE_thomas_weekly_wage_l902_90219

/-- Calculates the weekly wage given the monthly wage and number of weeks in a month. -/
def weekly_wage (monthly_wage : ℕ) (weeks_per_month : ℕ) : ℕ :=
  monthly_wage / weeks_per_month

/-- Proves that given a monthly wage of 19500 and 4 weeks in a month, the weekly wage is 4875. -/
theorem thomas_weekly_wage :
  weekly_wage 19500 4 = 4875 := by
  sorry

#eval weekly_wage 19500 4

end NUMINAMATH_CALUDE_thomas_weekly_wage_l902_90219


namespace NUMINAMATH_CALUDE_linear_equation_condition_l902_90269

/-- Given that (a-3)x^|a-2| + 4 = 0 is a linear equation in x and a-3 ≠ 0, prove that a = 1 -/
theorem linear_equation_condition (a : ℝ) : 
  (∀ x, ∃ k, (a - 3) * x^(|a - 2|) + 4 = k * x + 4) ∧ 
  (a - 3 ≠ 0) → 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l902_90269


namespace NUMINAMATH_CALUDE_line_circle_intersection_l902_90248

/-- The line x - y + 1 = 0 intersects the circle (x - a)² + y² = 2 
    if and only if a is in the closed interval [-3, 1] -/
theorem line_circle_intersection (a : ℝ) : 
  (∃ x y : ℝ, x - y + 1 = 0 ∧ (x - a)^2 + y^2 = 2) ↔ a ∈ Set.Icc (-3) 1 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l902_90248


namespace NUMINAMATH_CALUDE_tan_105_degrees_l902_90298

theorem tan_105_degrees : Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l902_90298


namespace NUMINAMATH_CALUDE_total_books_l902_90270

def books_per_shelf : ℕ := 15
def mystery_shelves : ℕ := 8
def picture_shelves : ℕ := 4
def biography_shelves : ℕ := 3
def scifi_shelves : ℕ := 5

theorem total_books : 
  books_per_shelf * (mystery_shelves + picture_shelves + biography_shelves + scifi_shelves) = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l902_90270


namespace NUMINAMATH_CALUDE_cindy_marbles_l902_90207

theorem cindy_marbles (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ) :
  initial_marbles = 500 →
  friends = 4 →
  marbles_per_friend = 80 →
  4 * (initial_marbles - friends * marbles_per_friend) = 720 := by
  sorry

end NUMINAMATH_CALUDE_cindy_marbles_l902_90207


namespace NUMINAMATH_CALUDE_ratio_approximation_l902_90251

def geometric_sum (n : ℕ) : ℚ :=
  (10^n - 1) / 9

def ratio (n : ℕ) : ℚ :=
  (10^n * 9) / (10^n - 1)

theorem ratio_approximation :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ |ratio 8 - 9| < ε :=
sorry

end NUMINAMATH_CALUDE_ratio_approximation_l902_90251
