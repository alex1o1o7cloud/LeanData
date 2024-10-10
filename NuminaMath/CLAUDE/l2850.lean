import Mathlib

namespace line_x_intercept_m_values_l2850_285073

theorem line_x_intercept_m_values (m : ℝ) : 
  (∃ y : ℝ, (2 * m^2 - m + 3) * 1 + (m^2 + 2*m) * y = 4*m + 1) → 
  (m = 2 ∨ m = 1/2) :=
by
  sorry

end line_x_intercept_m_values_l2850_285073


namespace vector_sum_magnitude_lower_bound_l2850_285074

/-- Given plane vectors a, b, and c satisfying certain dot product conditions,
    prove that the magnitude of their sum is at least 4. -/
theorem vector_sum_magnitude_lower_bound
  (a b c : ℝ × ℝ)
  (ha : a.1 * a.1 + a.2 * a.2 = 1)
  (hab : a.1 * b.1 + a.2 * b.2 = 1)
  (hac : a.1 * c.1 + a.2 * c.2 = 2)
  (hbc : b.1 * c.1 + b.2 * c.2 = 1) :
  (a.1 + b.1 + c.1)^2 + (a.2 + b.2 + c.2)^2 ≥ 16 := by
  sorry

#check vector_sum_magnitude_lower_bound

end vector_sum_magnitude_lower_bound_l2850_285074


namespace davids_biology_marks_l2850_285090

/-- Calculates the marks in Biology given the marks in other subjects and the average -/
def marks_in_biology (english : ℕ) (mathematics : ℕ) (physics : ℕ) (chemistry : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + mathematics + physics + chemistry)

/-- Theorem stating that David's marks in Biology are 85 -/
theorem davids_biology_marks :
  marks_in_biology 81 65 82 67 76 = 85 := by
  sorry

end davids_biology_marks_l2850_285090


namespace problem_1_problem_2_l2850_285050

-- Problem 1
theorem problem_1 : (-1)^10 * 2 + (-2)^3 / 4 = 0 := by sorry

-- Problem 2
theorem problem_2 : (-24) * (5/6 - 4/3 + 3/8) = 3 := by sorry

end problem_1_problem_2_l2850_285050


namespace milk_jars_theorem_l2850_285041

/-- Calculates the number of jars of milk good for sale given the conditions of Logan's father's milk business. -/
def good_milk_jars (normal_cartons : ℕ) (jars_per_carton : ℕ) (less_cartons : ℕ) 
  (damaged_cartons : ℕ) (damaged_jars_per_carton : ℕ) (totally_damaged_cartons : ℕ) : ℕ :=
  let received_cartons := normal_cartons - less_cartons
  let total_jars := received_cartons * jars_per_carton
  let partially_damaged_jars := damaged_cartons * damaged_jars_per_carton
  let totally_damaged_jars := totally_damaged_cartons * jars_per_carton
  let total_damaged_jars := partially_damaged_jars + totally_damaged_jars
  total_jars - total_damaged_jars

/-- Theorem stating that under the given conditions, the number of good milk jars for sale is 565. -/
theorem milk_jars_theorem : good_milk_jars 50 20 20 5 3 1 = 565 := by
  sorry

end milk_jars_theorem_l2850_285041


namespace min_distance_complex_l2850_285014

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + Real.sqrt 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 1 + Real.sqrt 2 ∧ 
  ∀ (w : ℂ), Complex.abs (w + Real.sqrt 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min_val :=
sorry

end min_distance_complex_l2850_285014


namespace servant_salary_l2850_285057

/-- Calculates the money received by a servant, excluding the turban -/
theorem servant_salary (annual_salary : ℝ) (turban_price : ℝ) (months_worked : ℝ) : 
  annual_salary = 90 →
  turban_price = 10 →
  months_worked = 9 →
  (months_worked / 12) * (annual_salary + turban_price) - turban_price = 65 :=
by sorry

end servant_salary_l2850_285057


namespace isosceles_triangle_area_theorem_l2850_285092

/-- The area of an isosceles triangle with two sides of length 5 units and a base of 6 units -/
def isosceles_triangle_area : ℝ := 12

/-- The length of the two equal sides of the isosceles triangle -/
def side_length : ℝ := 5

/-- The length of the base of the isosceles triangle -/
def base_length : ℝ := 6

theorem isosceles_triangle_area_theorem :
  let a := side_length
  let b := base_length
  let height := Real.sqrt (a^2 - (b/2)^2)
  (1/2) * b * height = isosceles_triangle_area :=
by sorry

end isosceles_triangle_area_theorem_l2850_285092


namespace ice_cream_consumption_l2850_285069

theorem ice_cream_consumption (friday_amount saturday_amount total_amount : ℝ) :
  friday_amount = 3.25 →
  saturday_amount = 0.25 →
  total_amount = friday_amount + saturday_amount →
  total_amount = 3.50 := by
  sorry

end ice_cream_consumption_l2850_285069


namespace total_students_theorem_l2850_285044

/-- Calculates the total number of students at the end of the year --/
def total_students_end_year (middle_initial : ℕ) : ℕ :=
  let elementary_initial := 4 * middle_initial - 3
  let high_initial := 2 * elementary_initial
  let elementary_end := (elementary_initial * 110 + 50) / 100
  let middle_end := (middle_initial * 95 + 50) / 100
  let high_end := (high_initial * 107 + 50) / 100
  elementary_end + middle_end + high_end

/-- Theorem stating that the total number of students at the end of the year is 687 --/
theorem total_students_theorem : total_students_end_year 50 = 687 := by
  sorry

end total_students_theorem_l2850_285044


namespace f_lower_bound_l2850_285042

open Real

noncomputable def f (a x : ℝ) : ℝ := a * x - log x

theorem f_lower_bound (a : ℝ) (h : a ≤ -1 / Real.exp 2) :
  ∀ x > 0, f a x ≥ 2 * a * x - x * Real.exp (a * x - 1) :=
by sorry

end f_lower_bound_l2850_285042


namespace functional_equation_implies_identity_l2850_285007

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

/-- The main theorem: if f satisfies the equation, then f is the identity function -/
theorem functional_equation_implies_identity (f : ℝ → ℝ) 
  (h : SatisfiesEquation f) : ∀ x : ℝ, f x = x := by
  sorry

end functional_equation_implies_identity_l2850_285007


namespace almonds_in_trail_mix_l2850_285022

/-- Given the amount of walnuts and the total amount of nuts in a trail mix,
    calculate the amount of almonds added. -/
theorem almonds_in_trail_mix (walnuts total : ℚ) (h1 : walnuts = 0.25) (h2 : total = 0.5) :
  total - walnuts = 0.25 := by
  sorry

end almonds_in_trail_mix_l2850_285022


namespace nine_special_integers_l2850_285067

theorem nine_special_integers (m n : ℕ) (hm : m ≥ 16) (hn : n ≥ 24) :
  ∃ (a : Fin 9 → ℕ),
    (∀ k : Fin 9, a k = 2^(m + k.val) * 3^(n - k.val)) ∧
    (∀ k : Fin 9, 6 ∣ a k) ∧
    (∀ i j : Fin 9, i ≠ j → ¬(a i ∣ a j)) ∧
    (∀ i j : Fin 9, (a i)^3 ∣ (a j)^2) := by
  sorry

end nine_special_integers_l2850_285067


namespace triangle_perimeter_l2850_285030

theorem triangle_perimeter (a b c : ℕ) (α β γ : ℝ) : 
  a > 0 ∧ b = a + 1 ∧ c = b + 1 →  -- Consecutive positive integer sides
  α > 0 ∧ β > 0 ∧ γ > 0 →  -- Positive angles
  α + β + γ = π →  -- Sum of angles in a triangle
  max γ (max α β) = 2 * min γ (min α β) →  -- Largest angle is twice the smallest
  a + b + c = 15 :=  -- Perimeter is 15
by sorry

end triangle_perimeter_l2850_285030


namespace perimeter_after_adding_tiles_l2850_285096

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  num_tiles : ℕ
  perimeter : ℕ

/-- Represents the process of adding tiles to a configuration -/
def add_tiles (initial : TileConfiguration) (added : ℕ) : TileConfiguration :=
  { num_tiles := initial.num_tiles + added,
    perimeter := initial.perimeter + added }

/-- Theorem statement -/
theorem perimeter_after_adding_tiles 
  (initial : TileConfiguration)
  (h1 : initial.num_tiles = 8)
  (h2 : initial.perimeter = 16)
  (added : ℕ)
  (h3 : added = 3) :
  ∃ (final : TileConfiguration),
    final = add_tiles initial added ∧ 
    final.perimeter = 19 :=
sorry

end perimeter_after_adding_tiles_l2850_285096


namespace ribbons_left_l2850_285037

theorem ribbons_left (initial : ℕ) (morning : ℕ) (afternoon : ℕ) : 
  initial = 38 → morning = 14 → afternoon = 16 → initial - (morning + afternoon) = 8 := by
  sorry

end ribbons_left_l2850_285037


namespace arccos_cos_ten_l2850_285005

theorem arccos_cos_ten :
  Real.arccos (Real.cos 10) = 10 - 4 * Real.pi := by sorry

end arccos_cos_ten_l2850_285005


namespace max_m_value_l2850_285094

theorem max_m_value (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + m*x₁ + 6 = 0 ∧ x₂^2 + m*x₂ + 6 = 0 ∧ |x₁ - x₂| = Real.sqrt 85) →
  m ≤ Real.sqrt 109 :=
by sorry

end max_m_value_l2850_285094


namespace bikes_in_parking_lot_l2850_285095

theorem bikes_in_parking_lot :
  let num_cars : ℕ := 10
  let total_wheels : ℕ := 44
  let wheels_per_car : ℕ := 4
  let wheels_per_bike : ℕ := 2
  let num_bikes : ℕ := (total_wheels - num_cars * wheels_per_car) / wheels_per_bike
  num_bikes = 2 := by sorry

end bikes_in_parking_lot_l2850_285095


namespace inverse_proportional_properties_l2850_285009

/-- Given two inverse proportional functions y = k/x and y = 1/x, where k > 0,
    and a point P(a, k/a) on y = k/x, with a > 0, we define:
    C(a, 0), A(a, 1/a), D(0, k/a), B(a/k, k/a) -/
theorem inverse_proportional_properties (k a : ℝ) (hk : k > 0) (ha : a > 0) :
  let P := (a, k / a)
  let C := (a, 0)
  let A := (a, 1 / a)
  let D := (0, k / a)
  let B := (a / k, k / a)
  let triangle_area (p q r : ℝ × ℝ) := (abs ((p.1 - r.1) * (q.2 - r.2) - (q.1 - r.1) * (p.2 - r.2))) / 2
  let quadrilateral_area (p q r s : ℝ × ℝ) := triangle_area p q r + triangle_area p r s
  -- 1. The areas of triangles ODB and OCA are equal to 1/2
  (triangle_area (0, 0) D B = 1 / 2 ∧ triangle_area (0, 0) C A = 1 / 2) ∧
  -- 2. The area of quadrilateral OAPB is equal to k - 1
  (quadrilateral_area (0, 0) A P B = k - 1) ∧
  -- 3. If k = 2, then A is the midpoint of PC and B is the midpoint of PD
  (k = 2 → (A.2 - C.2 = P.2 - A.2 ∧ B.1 - D.1 = P.1 - B.1)) := by
  sorry


end inverse_proportional_properties_l2850_285009


namespace square_preserves_geometric_sequence_sqrt_abs_preserves_geometric_sequence_l2850_285059

-- Define the domain for the functions
def Domain : Set ℝ := {x : ℝ | x < 0 ∨ x > 0}

-- Define the property of being a geometric sequence
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the property of being a geometric sequence preserving function
def IsGeometricSequencePreserving (f : ℝ → ℝ) : Prop :=
  ∀ a : ℕ → ℝ, (∀ n : ℕ, a n ∈ Domain) →
    IsGeometricSequence a → IsGeometricSequence (f ∘ a)

-- State the theorem for f(x) = x^2
theorem square_preserves_geometric_sequence :
  IsGeometricSequencePreserving (fun x ↦ x^2) :=
sorry

-- State the theorem for f(x) = √|x|
theorem sqrt_abs_preserves_geometric_sequence :
  IsGeometricSequencePreserving (fun x ↦ Real.sqrt (abs x)) :=
sorry

end square_preserves_geometric_sequence_sqrt_abs_preserves_geometric_sequence_l2850_285059


namespace problem_statement_l2850_285024

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  (a * b ≤ 1) ∧ (2^a + 2^b ≥ 2 * Real.sqrt 2) ∧ (1/a + 4/b ≥ 9/2) := by
  sorry

end problem_statement_l2850_285024


namespace set_equals_open_interval_l2850_285032

theorem set_equals_open_interval :
  {x : ℝ | -1 < x ∧ x < 1} = Set.Ioo (-1 : ℝ) 1 := by sorry

end set_equals_open_interval_l2850_285032


namespace additive_inverse_solution_l2850_285097

theorem additive_inverse_solution (x : ℝ) : (2*x - 12) + (x + 3) = 0 → x = 3 := by
  sorry

end additive_inverse_solution_l2850_285097


namespace min_sum_reciprocals_l2850_285046

theorem min_sum_reciprocals (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 1) :
  1/m + 1/n ≥ 4 := by
  sorry

end min_sum_reciprocals_l2850_285046


namespace negation_p_sufficient_not_necessary_for_negation_q_l2850_285048

theorem negation_p_sufficient_not_necessary_for_negation_q :
  ∃ (x : ℝ),
    (∀ x, (|x + 1| > 0 → (5*x - 6 > x^2)) →
      (x = -1 → (x ≤ 2 ∨ x ≥ 3)) ∧
      ¬(x ≤ 2 ∨ x ≥ 3 → x = -1)) :=
by sorry

end negation_p_sufficient_not_necessary_for_negation_q_l2850_285048


namespace magnitude_relationship_l2850_285068

theorem magnitude_relationship (a b c : ℝ) : 
  a = Real.sin (46 * π / 180) →
  b = Real.cos (46 * π / 180) →
  c = Real.cos (36 * π / 180) →
  c > a ∧ a > b := by sorry

end magnitude_relationship_l2850_285068


namespace arithmetic_sequence_formula_l2850_285035

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a3 : a 3 = 4) 
  (h_d : ∃ d : ℝ, d = -2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  ∀ n : ℕ, a n = 10 - 2 * n :=
sorry

end arithmetic_sequence_formula_l2850_285035


namespace drug_price_reduction_l2850_285043

theorem drug_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 50)
  (h2 : final_price = 40.5)
  (h3 : final_price = initial_price * (1 - x)^2)
  (h4 : 0 < x ∧ x < 1) :
  x = 0.1 := by sorry

end drug_price_reduction_l2850_285043


namespace perfect_square_implies_congruence_l2850_285047

theorem perfect_square_implies_congruence (p a : ℕ) (h_prime : Nat.Prime p) :
  (∃ t : ℤ, ∃ k : ℤ, p * t + a = k^2) →
  a^((p - 1) / 2) ≡ 1 [ZMOD p] :=
sorry

end perfect_square_implies_congruence_l2850_285047


namespace tamara_garden_walkway_area_l2850_285054

/-- Represents the dimensions of a flower bed -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Represents the garden layout -/
structure Garden where
  rows : ℕ
  columns : ℕ
  bed : FlowerBed
  walkwayWidth : ℝ

/-- Calculates the total area of walkways in the garden -/
def walkwayArea (g : Garden) : ℝ :=
  let totalWidth := g.columns * g.bed.length + (g.columns + 1) * g.walkwayWidth
  let totalHeight := g.rows * g.bed.width + (g.rows + 1) * g.walkwayWidth
  let totalArea := totalWidth * totalHeight
  let bedArea := g.rows * g.columns * g.bed.length * g.bed.width
  totalArea - bedArea

/-- Theorem stating that the walkway area for Tamara's garden is 214 square feet -/
theorem tamara_garden_walkway_area :
  let g : Garden := {
    rows := 3,
    columns := 2,
    bed := { length := 7, width := 3 },
    walkwayWidth := 2
  }
  walkwayArea g = 214 := by
  sorry

end tamara_garden_walkway_area_l2850_285054


namespace roe_savings_aug_to_nov_l2850_285051

def savings_jan_to_jul : ℕ := 10 * 7
def savings_dec : ℕ := 20
def total_savings : ℕ := 150
def months_aug_to_nov : ℕ := 4

theorem roe_savings_aug_to_nov :
  (total_savings - savings_jan_to_jul - savings_dec) / months_aug_to_nov = 15 := by
  sorry

end roe_savings_aug_to_nov_l2850_285051


namespace solve_system_of_equations_l2850_285087

theorem solve_system_of_equations (u v : ℚ) 
  (eq1 : 5 * u - 6 * v = 35)
  (eq2 : 3 * u + 5 * v = -10) :
  u + v = -40 / 43 := by
sorry

end solve_system_of_equations_l2850_285087


namespace sqrt_36_times_sqrt_16_l2850_285029

theorem sqrt_36_times_sqrt_16 : Real.sqrt (36 * Real.sqrt 16) = 12 := by
  sorry

end sqrt_36_times_sqrt_16_l2850_285029


namespace max_value_problem_l2850_285003

theorem max_value_problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x^2 + y + z = 1) :
  ∀ a b c : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a^2 + b + c = 1 → x + y^3 + z^4 ≤ a + b^3 + c^4 → x + y^3 + z^4 ≤ 1 :=
sorry

end max_value_problem_l2850_285003


namespace power_inequality_l2850_285083

theorem power_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  a^a < b^a := by
  sorry

end power_inequality_l2850_285083


namespace empty_solution_set_l2850_285065

theorem empty_solution_set (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 1| - |x + 2| < a)) ↔ a ≤ -3 :=
by sorry

end empty_solution_set_l2850_285065


namespace a_plus_b_equals_neg_nine_l2850_285040

def f (a b x : ℝ) : ℝ := a * x - b

def g (x : ℝ) : ℝ := -4 * x - 1

def h (a b x : ℝ) : ℝ := f a b (g x)

def h_inv (x : ℝ) : ℝ := x + 9

theorem a_plus_b_equals_neg_nine (a b : ℝ) :
  (∀ x, h a b x = h_inv⁻¹ x) → a + b = -9 := by
  sorry

end a_plus_b_equals_neg_nine_l2850_285040


namespace a_4_times_a_3_l2850_285038

def a : ℕ → ℤ
  | n => if n % 2 = 1 then (-2)^n else n

theorem a_4_times_a_3 : a 4 * a 3 = -32 := by
  sorry

end a_4_times_a_3_l2850_285038


namespace number_of_nephews_l2850_285019

-- Define the price of a candy as the base unit
def candy_price : ℚ := 1

-- Define the prices of other items in terms of candy price
def orange_price : ℚ := 2 * candy_price
def cake_price : ℚ := 4 * candy_price
def chocolate_price : ℚ := 7 * candy_price
def book_price : ℚ := 14 * candy_price

-- Define the cost of one gift
def gift_cost : ℚ := candy_price + orange_price + cake_price + chocolate_price + book_price

-- Define the total number of each item if all money was spent on that item
def total_candies : ℕ := 224
def total_oranges : ℕ := 112
def total_cakes : ℕ := 56
def total_chocolates : ℕ := 32
def total_books : ℕ := 16

-- Theorem: The number of nephews is 8
theorem number_of_nephews : ℕ := by
  sorry

end number_of_nephews_l2850_285019


namespace tiffany_bags_on_monday_l2850_285012

theorem tiffany_bags_on_monday :
  ∀ (bags_monday : ℕ),
  bags_monday + 8 = 12 →
  bags_monday = 4 :=
by
  sorry

end tiffany_bags_on_monday_l2850_285012


namespace right_triangle_area_l2850_285063

/-- 
Given a right triangle with hypotenuse c, where the projection of the right angle 
vertex onto the hypotenuse divides it into two segments x and (c-x) such that 
(c-x)/x = x/c, the area of the triangle is (c^2 * sqrt(sqrt(5) - 2)) / 2.
-/
theorem right_triangle_area (c : ℝ) (h : c > 0) : 
  ∃ x : ℝ, 0 < x ∧ x < c ∧ (c - x) / x = x / c ∧ 
  (c^2 * Real.sqrt (Real.sqrt 5 - 2)) / 2 = 
  (1 / 2) * c * Real.sqrt (c * x - x^2) := by
sorry

end right_triangle_area_l2850_285063


namespace translate_linear_function_l2850_285093

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Translates a linear function vertically by a given amount -/
def translateVertically (f : LinearFunction) (dy : ℝ) : LinearFunction :=
  { m := f.m, b := f.b + dy }

/-- The theorem stating that translating y = -2x + 1 up 4 units results in y = -2x + 5 -/
theorem translate_linear_function :
  let f : LinearFunction := { m := -2, b := 1 }
  let g : LinearFunction := translateVertically f 4
  g.m = -2 ∧ g.b = 5 := by sorry

end translate_linear_function_l2850_285093


namespace dodecahedron_coloring_count_l2850_285062

/-- The number of faces in a regular dodecahedron -/
def num_faces : ℕ := 12

/-- The order of the rotational symmetry group of a regular dodecahedron -/
def dodecahedron_symmetry_order : ℕ := 60

/-- The number of distinguishable colorings of a regular dodecahedron 
    with different colors for each face, considering rotational symmetries -/
def distinguishable_colorings : ℕ := (Nat.factorial (num_faces - 1)) / dodecahedron_symmetry_order

theorem dodecahedron_coloring_count :
  distinguishable_colorings = 665280 := by sorry

end dodecahedron_coloring_count_l2850_285062


namespace fraction_simplification_l2850_285064

theorem fraction_simplification (N : ℕ) :
  (Nat.factorial (N - 2) * (N - 1) * N) / Nat.factorial (N + 2) = 1 / ((N + 1) * (N + 2)) :=
by sorry

end fraction_simplification_l2850_285064


namespace dvd_price_proof_l2850_285085

/-- The price Mike paid for the DVD at the store -/
def mike_price : ℝ := 5

/-- The price Steve paid for the DVD online -/
def steve_online_price (p : ℝ) : ℝ := 2 * p

/-- The shipping cost Steve paid -/
def steve_shipping_cost (p : ℝ) : ℝ := 0.8 * steve_online_price p

/-- The total amount Steve paid -/
def steve_total_cost (p : ℝ) : ℝ := steve_online_price p + steve_shipping_cost p

theorem dvd_price_proof :
  steve_total_cost mike_price = 18 :=
sorry

end dvd_price_proof_l2850_285085


namespace smallest_k_and_largest_base_l2850_285025

theorem smallest_k_and_largest_base : ∃ (b : ℕ), 
  (64 ^ 7 > b ^ 20) ∧ 
  (∀ (x : ℕ), x > b → 64 ^ 7 ≤ x ^ 20) ∧ 
  b = 4 := by
  sorry

end smallest_k_and_largest_base_l2850_285025


namespace seating_arrangement_probability_l2850_285045

/-- Represents the number of delegates --/
def total_delegates : ℕ := 12

/-- Represents the number of countries --/
def num_countries : ℕ := 3

/-- Represents the number of delegates per country --/
def delegates_per_country : ℕ := 4

/-- Calculates the probability of the seating arrangement --/
noncomputable def seating_probability : ℚ :=
  409 / 500

/-- Theorem stating the probability of the specific seating arrangement --/
theorem seating_arrangement_probability :
  let total_arrangements := (total_delegates.factorial) / (delegates_per_country.factorial ^ num_countries)
  let favorable_arrangements := total_arrangements - (num_countries * total_delegates * 
    ((total_delegates - delegates_per_country).factorial / (delegates_per_country.factorial ^ (num_countries - 1))) -
    (num_countries * (num_countries - 1) / 2 * total_delegates * (total_delegates - 2)) +
    (total_delegates * (num_countries - 1)))
  (favorable_arrangements : ℚ) / total_arrangements = seating_probability :=
sorry

end seating_arrangement_probability_l2850_285045


namespace two_in_M_l2850_285018

def U : Set Nat := {1, 2, 3, 4, 5}

theorem two_in_M (M : Set Nat) (h : (U \ M) = {1, 3}) : 2 ∈ M := by
  sorry

end two_in_M_l2850_285018


namespace distance_between_vertices_l2850_285082

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 121 - y^2 / 49 = 1

-- Define the vertices of the hyperbola
def vertices : Set (ℝ × ℝ) :=
  {(11, 0), (-11, 0)}

-- Theorem statement
theorem distance_between_vertices :
  ∀ (v1 v2 : ℝ × ℝ), v1 ∈ vertices → v2 ∈ vertices → v1 ≠ v2 →
  Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = 22 := by
  sorry

end distance_between_vertices_l2850_285082


namespace simplify_expression_l2850_285006

theorem simplify_expression (x y : ℝ) : 3*x + 6*x + 9*x + 12*y + 15*y + 18 + 21 = 18*x + 27*y + 39 := by
  sorry

end simplify_expression_l2850_285006


namespace two_point_distribution_p_values_l2850_285031

/-- A two-point distribution random variable -/
structure TwoPointDistribution where
  p : ℝ
  prob_x_eq_one : p ∈ Set.Icc 0 1

/-- The variance of a two-point distribution -/
def variance (X : TwoPointDistribution) : ℝ := X.p - X.p^2

theorem two_point_distribution_p_values (X : TwoPointDistribution) 
  (h : variance X = 2/9) : X.p = 1/3 ∨ X.p = 2/3 := by
  sorry

end two_point_distribution_p_values_l2850_285031


namespace total_area_is_60_l2850_285023

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The four rectangles that compose the figure -/
def rectangles : List Rectangle := [
  { width := 5, height := 5 },
  { width := 5, height := 3 },
  { width := 5, height := 2 },
  { width := 5, height := 2 }
]

/-- Theorem: The total area of the figure is 60 square units -/
theorem total_area_is_60 : 
  (rectangles.map Rectangle.area).sum = 60 := by sorry

end total_area_is_60_l2850_285023


namespace average_temperature_is_42_4_l2850_285056

/-- The average daily low temperature in Addington from September 15th to 19th, 2008 -/
def average_temperature : ℚ :=
  let temperatures : List ℤ := [40, 47, 45, 41, 39]
  (temperatures.sum : ℚ) / temperatures.length

/-- Theorem stating that the average temperature is 42.4°F -/
theorem average_temperature_is_42_4 : 
  average_temperature = 424/10 := by sorry

end average_temperature_is_42_4_l2850_285056


namespace uneven_gender_probability_l2850_285076

/-- The number of children in the family -/
def num_children : ℕ := 8

/-- The probability of a child being male (or female) -/
def gender_prob : ℚ := 1/2

/-- The total number of possible gender combinations -/
def total_combinations : ℕ := 2^num_children

/-- The number of combinations with an even split of genders -/
def even_split_combinations : ℕ := Nat.choose num_children (num_children / 2)

/-- The probability of having an uneven number of sons and daughters -/
def prob_uneven : ℚ := 1 - (even_split_combinations : ℚ) / total_combinations

theorem uneven_gender_probability :
  prob_uneven = 93/128 :=
sorry

end uneven_gender_probability_l2850_285076


namespace octagon_triangle_area_ratio_l2850_285017

theorem octagon_triangle_area_ratio (s_o s_t : ℝ) (h : s_o > 0) (h' : s_t > 0) :
  (2 * s_o^2 * (1 + Real.sqrt 2) = s_t^2 * Real.sqrt 3 / 4) →
  s_t / s_o = Real.sqrt (8 + 8 * Real.sqrt 2) :=
by sorry

end octagon_triangle_area_ratio_l2850_285017


namespace largest_divisor_of_m_l2850_285061

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 216 ∣ m^2) : 
  36 = Nat.gcd m 36 ∧ ∀ k : ℕ, k > 36 → k ∣ m → k ∣ 36 := by
  sorry

end largest_divisor_of_m_l2850_285061


namespace distance_between_points_l2850_285039

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 5)
  let p2 : ℝ × ℝ := (5, 1)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 5 := by sorry

end distance_between_points_l2850_285039


namespace negation_equal_area_congruent_is_true_l2850_285026

-- Define a type for triangles
def Triangle : Type := sorry

-- Define a function for the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Theorem stating that the negation of "Triangles with equal areas are congruent" is true
theorem negation_equal_area_congruent_is_true :
  ¬(∀ t1 t2 : Triangle, area t1 = area t2 → congruent t1 t2) :=
sorry

end negation_equal_area_congruent_is_true_l2850_285026


namespace orange_balls_count_l2850_285099

theorem orange_balls_count :
  let total_balls : ℕ := 100
  let red_balls : ℕ := 30
  let blue_balls : ℕ := 20
  let yellow_balls : ℕ := 10
  let green_balls : ℕ := 5
  let pink_balls : ℕ := 2 * green_balls
  let orange_balls : ℕ := 3 * pink_balls
  red_balls + blue_balls + yellow_balls + green_balls + pink_balls + orange_balls = total_balls →
  orange_balls = 30 :=
by sorry

end orange_balls_count_l2850_285099


namespace min_value_inequality_l2850_285028

theorem min_value_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 := by
  sorry

end min_value_inequality_l2850_285028


namespace abc_sum_l2850_285066

theorem abc_sum (a b c : ℚ) : 
  (a : ℚ) / 3 = (b : ℚ) / 5 ∧ (b : ℚ) / 5 = (c : ℚ) / 7 ∧ 
  3 * a + 2 * b - 4 * c = -9 → 
  a + b - c = 1 := by
sorry

end abc_sum_l2850_285066


namespace joe_count_l2850_285001

theorem joe_count (barry_count kevin_count julie_count : ℕ) 
  (nice_count : ℕ) (joe_nice_ratio : ℚ) :
  barry_count = 24 →
  kevin_count = 20 →
  julie_count = 80 →
  nice_count = 99 →
  joe_nice_ratio = 1/10 →
  ∃ (joe_count : ℕ),
    joe_count = 50 ∧
    nice_count = barry_count + 
                 (kevin_count / 2) + 
                 (julie_count * 3 / 4) + 
                 (joe_count * joe_nice_ratio) :=
by sorry

end joe_count_l2850_285001


namespace case_one_solutions_case_two_no_solution_l2850_285081

-- Case 1
theorem case_one_solutions (a b : ℝ) (A : ℝ) (ha : a = 14) (hb : b = 16) (hA : A = 45 * π / 180) :
  ∃! (B C : ℝ), 0 < B ∧ 0 < C ∧ A + B + C = π ∧ 
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = Real.sqrt (a^2 + b^2 - 2*a*b*Real.cos C) / Real.sin C :=
sorry

-- Case 2
theorem case_two_no_solution (a b : ℝ) (B : ℝ) (ha : a = 60) (hb : b = 48) (hB : B = 60 * π / 180) :
  ¬ ∃ (A C : ℝ), 0 < A ∧ 0 < C ∧ A + B + C = π ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = Real.sqrt (a^2 + b^2 - 2*a*b*Real.cos C) / Real.sin C :=
sorry

end case_one_solutions_case_two_no_solution_l2850_285081


namespace circle_center_radius_sum_l2850_285060

/-- Given a circle C with equation x^2 + 8x - 5y = -y^2 + 2x, 
    the sum of the x-coordinate and y-coordinate of its center along with its radius 
    is equal to (√61 - 1) / 2 -/
theorem circle_center_radius_sum (x y : ℝ) : 
  (x^2 + 8*x - 5*y = -y^2 + 2*x) → 
  ∃ (center_x center_y radius : ℝ), 
    (center_x + center_y + radius = (Real.sqrt 61 - 1) / 2) ∧
    ∀ (p_x p_y : ℝ), (p_x - center_x)^2 + (p_y - center_y)^2 = radius^2 ↔ 
      p_x^2 + 8*p_x - 5*p_y = -p_y^2 + 2*p_x :=
by sorry

end circle_center_radius_sum_l2850_285060


namespace winning_number_correct_l2850_285084

/-- The number of callers needed to win all three prizes -/
def winning_number : ℕ := 1125

/-- The maximum allowed number of callers -/
def max_callers : ℕ := 2000

/-- Checks if a number is divisible by another number -/
def is_divisible (a b : ℕ) : Prop := b ∣ a

/-- Checks if a number is not divisible by 10 -/
def not_multiple_of_ten (n : ℕ) : Prop := ¬(is_divisible n 10)

/-- Theorem stating the winning number is correct -/
theorem winning_number_correct :
  (is_divisible winning_number 100) ∧ 
  (is_divisible winning_number 40) ∧ 
  (is_divisible winning_number 250) ∧
  (not_multiple_of_ten winning_number) ∧
  (∀ n : ℕ, n < winning_number → 
    ¬(is_divisible n 100 ∧ is_divisible n 40 ∧ is_divisible n 250 ∧ not_multiple_of_ten n)) ∧
  (winning_number ≤ max_callers) :=
sorry

end winning_number_correct_l2850_285084


namespace solution_pairs_count_l2850_285011

theorem solution_pairs_count : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    4 * p.1 + 7 * p.2 = 600 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 601) (Finset.range 601))).card ∧ n = 22 :=
by sorry

end solution_pairs_count_l2850_285011


namespace f_monotone_decreasing_l2850_285055

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem f_monotone_decreasing :
  ∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → f x > f y := by
  sorry

end f_monotone_decreasing_l2850_285055


namespace transistor_count_scientific_notation_l2850_285027

/-- The number of transistors in a Huawei Kirin 990 processor -/
def transistor_count : ℝ := 12000000000

/-- The scientific notation representation of the transistor count -/
def scientific_notation : ℝ := 1.2 * (10 ^ 10)

theorem transistor_count_scientific_notation : 
  transistor_count = scientific_notation := by
  sorry

end transistor_count_scientific_notation_l2850_285027


namespace runner_problem_l2850_285053

theorem runner_problem (v : ℝ) (h1 : v > 0) :
  let t1 := 20 / v
  let t2 := 40 / v
  t2 = t1 + 4 →
  t2 = 8 := by
sorry

end runner_problem_l2850_285053


namespace prime_power_digit_repetition_l2850_285098

theorem prime_power_digit_repetition (p n : ℕ) : 
  Prime p → p > 3 → (10^19 ≤ p^n ∧ p^n < 10^20) → 
  ∃ (d : ℕ) (i j k : ℕ), i < j ∧ j < k ∧ i < 20 ∧ j < 20 ∧ k < 20 ∧
  d < 10 ∧ (p^n / 10^i) % 10 = d ∧ (p^n / 10^j) % 10 = d ∧ (p^n / 10^k) % 10 = d :=
by sorry

end prime_power_digit_repetition_l2850_285098


namespace arithmetic_sequence_sum_l2850_285088

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 2 + a 16 + a 30 = 60 →
  a 10 + a 22 = 40 := by
  sorry

end arithmetic_sequence_sum_l2850_285088


namespace negation_of_not_even_numbers_l2850_285049

theorem negation_of_not_even_numbers (a b : ℤ) : 
  ¬(¬Even a ∧ ¬Even b) ↔ (Even a ∨ Even b) :=
sorry

end negation_of_not_even_numbers_l2850_285049


namespace flower_bouquet_row_length_l2850_285021

theorem flower_bouquet_row_length 
  (num_students : ℕ) 
  (student_space : ℝ) 
  (gap_space : ℝ) 
  (h1 : num_students = 50) 
  (h2 : student_space = 0.4) 
  (h3 : gap_space = 0.5) : 
  num_students * student_space + (num_students - 1) * gap_space = 44.5 := by
  sorry

end flower_bouquet_row_length_l2850_285021


namespace quadratic_factorization_l2850_285078

theorem quadratic_factorization (x : ℝ) : x^2 + 6*x = 1 ↔ (x + 3)^2 = 10 := by
  sorry

end quadratic_factorization_l2850_285078


namespace arithmetic_sequence_first_term_l2850_285013

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_first_term 
  (seq : ArithmeticSequence) 
  (h3 : seq.a 2 * seq.a 3 = seq.a 4 * seq.a 5) 
  (h4 : S seq 4 = 27) : 
  seq.a 1 = 135 / 8 := by
sorry

end arithmetic_sequence_first_term_l2850_285013


namespace complex_magnitude_proof_l2850_285016

theorem complex_magnitude_proof : Complex.abs (-4 + (7/6) * Complex.I) = 25/6 := by
  sorry

end complex_magnitude_proof_l2850_285016


namespace beads_per_earring_is_five_l2850_285080

/-- The number of beads needed to make one earring given Kylie's jewelry-making activities --/
def beads_per_earring : ℕ :=
  let necklaces_monday : ℕ := 10
  let necklaces_tuesday : ℕ := 2
  let bracelets : ℕ := 5
  let earrings : ℕ := 7
  let beads_per_necklace : ℕ := 20
  let beads_per_bracelet : ℕ := 10
  let total_beads : ℕ := 325
  let beads_for_necklaces : ℕ := (necklaces_monday + necklaces_tuesday) * beads_per_necklace
  let beads_for_bracelets : ℕ := bracelets * beads_per_bracelet
  let beads_for_earrings : ℕ := total_beads - beads_for_necklaces - beads_for_bracelets
  beads_for_earrings / earrings

theorem beads_per_earring_is_five : beads_per_earring = 5 := by
  sorry

end beads_per_earring_is_five_l2850_285080


namespace no_valid_area_codes_l2850_285072

def is_valid_digit (d : ℕ) : Prop := d = 2 ∨ d = 4 ∨ d = 3 ∨ d = 5

def is_valid_area_code (code : Fin 4 → ℕ) : Prop :=
  ∀ i, is_valid_digit (code i)

def product_of_digits (code : Fin 4 → ℕ) : ℕ :=
  (code 0) * (code 1) * (code 2) * (code 3)

theorem no_valid_area_codes :
  ¬∃ (code : Fin 4 → ℕ), is_valid_area_code code ∧ 13 ∣ product_of_digits code := by
  sorry

end no_valid_area_codes_l2850_285072


namespace log_base_value_l2850_285010

theorem log_base_value (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x > 0, f x = Real.log x / Real.log a) (h2 : f 4 = 2) : a = 2 := by
  sorry

end log_base_value_l2850_285010


namespace seven_fifths_of_negative_eighteen_fourths_l2850_285071

theorem seven_fifths_of_negative_eighteen_fourths :
  (7 : ℚ) / 5 * (-18 : ℚ) / 4 = (-63 : ℚ) / 10 := by
  sorry

end seven_fifths_of_negative_eighteen_fourths_l2850_285071


namespace machine_work_time_l2850_285008

/-- Proves that a machine making 6 shirts per minute worked for 23 minutes yesterday,
    given it made 14 shirts today and 156 shirts in total over two days. -/
theorem machine_work_time (shirts_per_minute : ℕ) (shirts_today : ℕ) (total_shirts : ℕ) :
  shirts_per_minute = 6 →
  shirts_today = 14 →
  total_shirts = 156 →
  (total_shirts - shirts_today) / shirts_per_minute = 23 :=
by sorry

end machine_work_time_l2850_285008


namespace comic_books_calculation_l2850_285020

theorem comic_books_calculation (initial : ℕ) (bought : ℕ) : 
  initial = 14 → bought = 6 → initial / 2 + bought = 13 := by
  sorry

end comic_books_calculation_l2850_285020


namespace largest_base6_5digit_in_base10_l2850_285052

/-- The largest five-digit number in base 6 -/
def largest_base6_5digit : ℕ := 5 * 6^4 + 5 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0

/-- Theorem: The largest five-digit number in base 6 equals 7775 in base 10 -/
theorem largest_base6_5digit_in_base10 : largest_base6_5digit = 7775 := by
  sorry

end largest_base6_5digit_in_base10_l2850_285052


namespace fraction_sum_equality_l2850_285075

theorem fraction_sum_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1 / (2 * a * b) + b / (4 * a) = (2 + b^2) / (4 * a * b) := by
  sorry

end fraction_sum_equality_l2850_285075


namespace increasing_cubic_function_condition_l2850_285058

/-- A function f(x) = x³ - ax - 1 is increasing for all real x if and only if a ≤ 0 -/
theorem increasing_cubic_function_condition (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => x^3 - a*x - 1)) ↔ a ≤ 0 :=
by sorry

end increasing_cubic_function_condition_l2850_285058


namespace factorization_proof_l2850_285091

theorem factorization_proof (x : ℝ) : -8*x^2 + 8*x - 2 = -2*(2*x - 1)^2 := by
  sorry

end factorization_proof_l2850_285091


namespace six_balls_three_boxes_l2850_285004

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 132 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 132 := by sorry

end six_balls_three_boxes_l2850_285004


namespace positive_less_than_one_inequality_l2850_285077

theorem positive_less_than_one_inequality (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) : 
  1 + a^2 + b^2 > 3 * a * b := by
  sorry

end positive_less_than_one_inequality_l2850_285077


namespace initial_men_count_initial_men_count_is_seven_l2850_285089

/-- Proves that the initial number of men in a group is 7 given specific conditions about age changes. -/
theorem initial_men_count : ℕ :=
  let initial_average : ℝ := sorry
  let final_average : ℝ := initial_average + 4
  let replaced_men_ages : Fin 2 → ℕ := ![26, 30]
  let women_average_age : ℝ := 42
  let men_count : ℕ := sorry
  have h1 : final_average * men_count = initial_average * men_count + 4 * men_count := sorry
  have h2 : (men_count - 2) * initial_average + 2 * women_average_age = men_count * final_average := sorry
  have h3 : 2 * women_average_age - (replaced_men_ages 0 + replaced_men_ages 1) = 4 * men_count := sorry
  7

theorem initial_men_count_is_seven : initial_men_count = 7 := by sorry

end initial_men_count_initial_men_count_is_seven_l2850_285089


namespace jason_music_store_expense_l2850_285086

/-- The amount Jason spent at the music store -/
def jason_total_spent (flute_cost music_stand_cost song_book_cost : ℝ) : ℝ :=
  flute_cost + music_stand_cost + song_book_cost

/-- Theorem: Jason spent $158.35 at the music store -/
theorem jason_music_store_expense :
  jason_total_spent 142.46 8.89 7 = 158.35 := by
  sorry

end jason_music_store_expense_l2850_285086


namespace one_minus_repeating_six_eq_one_third_l2850_285036

/-- The decimal 0.666... (repeating 6) --/
def repeating_six : ℚ := 2/3

/-- Proof that 1 - 0.666... (repeating 6) equals 1/3 --/
theorem one_minus_repeating_six_eq_one_third : 1 - repeating_six = (1 : ℚ) / 3 := by
  sorry

end one_minus_repeating_six_eq_one_third_l2850_285036


namespace fraction_equality_l2850_285002

theorem fraction_equality : (8 : ℚ) / (5 * 46) = 0.8 / 23 := by sorry

end fraction_equality_l2850_285002


namespace eggs_last_24_days_l2850_285070

/-- Calculates the number of days eggs will last given initial eggs, daily egg laying, and daily consumption. -/
def days_eggs_last (initial_eggs : ℕ) (daily_laid : ℕ) (daily_consumed : ℕ) : ℕ :=
  initial_eggs / (daily_consumed - daily_laid)

/-- Theorem: Given 72 initial eggs, a hen laying 1 egg per day, and a family consuming 4 eggs per day, the eggs will last for 24 days. -/
theorem eggs_last_24_days :
  days_eggs_last 72 1 4 = 24 := by
  sorry

end eggs_last_24_days_l2850_285070


namespace photo_frame_border_area_l2850_285033

theorem photo_frame_border_area :
  let photo_height : ℝ := 12
  let photo_width : ℝ := 14
  let frame_width : ℝ := 3
  let framed_height : ℝ := photo_height + 2 * frame_width
  let framed_width : ℝ := photo_width + 2 * frame_width
  let photo_area : ℝ := photo_height * photo_width
  let framed_area : ℝ := framed_height * framed_width
  let border_area : ℝ := framed_area - photo_area
  border_area = 192 := by sorry

end photo_frame_border_area_l2850_285033


namespace quadratic_real_roots_condition_l2850_285079

theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 2 * x - 1 = 0) ↔ (k ≥ -1 ∧ k ≠ 0) :=
by sorry

end quadratic_real_roots_condition_l2850_285079


namespace units_digit_of_sum_even_factorials_l2850_285015

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def sum_of_even_factorials : ℕ := 
  factorial 2 + factorial 4 + factorial 6 + factorial 8 + factorial 10

theorem units_digit_of_sum_even_factorials :
  units_digit sum_of_even_factorials = 6 := by sorry

end units_digit_of_sum_even_factorials_l2850_285015


namespace complex_sum_powers_l2850_285034

theorem complex_sum_powers (i : ℂ) (h : i^2 = -1) : i + i^2 + i^3 + i^4 + i^5 = i := by
  sorry

end complex_sum_powers_l2850_285034


namespace quadratic_equation_solution_l2850_285000

theorem quadratic_equation_solution (c : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + c = 0 ∧ (-x)^2 + 3*(-x) - c = 0) →
  (∀ x : ℝ, x^2 - 3*x + c = 0 ↔ (x = 0 ∨ x = 3)) :=
by sorry

end quadratic_equation_solution_l2850_285000
