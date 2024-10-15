import Mathlib

namespace NUMINAMATH_CALUDE_no_polyhedron_with_seven_edges_l2761_276136

/-- Represents a polyhedron with vertices, edges, and faces. -/
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's formula for polyhedra: V - E + F = 2 -/
axiom eulers_formula (p : Polyhedron) : p.vertices - p.edges + p.faces = 2

/-- Each edge is shared by exactly two faces -/
axiom edge_face_relation (p : Polyhedron) : 2 * p.edges = 3 * p.faces

/-- Theorem: There is no polyhedron with exactly 7 edges -/
theorem no_polyhedron_with_seven_edges :
  ¬∃ (p : Polyhedron), p.edges = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_polyhedron_with_seven_edges_l2761_276136


namespace NUMINAMATH_CALUDE_quadrilateral_angle_proof_l2761_276108

theorem quadrilateral_angle_proof (A B C D : ℝ) : 
  A + B = 180 →
  C = D →
  A = 85 →
  B + C + D = 180 →
  D = 42.5 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_proof_l2761_276108


namespace NUMINAMATH_CALUDE_chris_least_money_l2761_276103

-- Define the set of people
inductive Person : Type
  | Alice : Person
  | Bob : Person
  | Chris : Person
  | Dana : Person
  | Eve : Person

-- Define the money function
variable (money : Person → ℝ)

-- State the conditions
axiom different_amounts : ∀ p q : Person, p ≠ q → money p ≠ money q
axiom chris_less_than_bob : money Person.Chris < money Person.Bob
axiom dana_less_than_bob : money Person.Dana < money Person.Bob
axiom alice_more_than_chris : money Person.Chris < money Person.Alice
axiom eve_more_than_chris : money Person.Chris < money Person.Eve
axiom dana_equal_eve : money Person.Dana = money Person.Eve
axiom dana_less_than_alice : money Person.Dana < money Person.Alice
axiom bob_more_than_eve : money Person.Eve < money Person.Bob

-- State the theorem
theorem chris_least_money :
  ∀ p : Person, p ≠ Person.Chris → money Person.Chris ≤ money p :=
sorry

end NUMINAMATH_CALUDE_chris_least_money_l2761_276103


namespace NUMINAMATH_CALUDE_right_triangle_inscribed_shapes_l2761_276123

/-- Given a right triangle ABC with legs AC = a and CB = b, prove:
    1. The side length of the largest square with vertex C inside the triangle
    2. The dimensions of the largest rectangle with vertex C inside the triangle -/
theorem right_triangle_inscribed_shapes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let square_side := a * b / (a + b)
  let rect_width := a / 2
  let rect_height := b / 2
  (∀ s : ℝ, s > 0 ∧ s ≤ a ∧ s ≤ b → s ≤ square_side) ∧
  (∀ w h : ℝ, w > 0 ∧ h > 0 ∧ w ≤ a ∧ h ≤ b ∧ w / a + h / b ≤ 1 → w * h ≤ rect_width * rect_height) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_inscribed_shapes_l2761_276123


namespace NUMINAMATH_CALUDE_silver_cost_per_ounce_l2761_276122

theorem silver_cost_per_ounce
  (silver_amount : ℝ)
  (gold_amount : ℝ)
  (gold_silver_price_ratio : ℝ)
  (total_spent : ℝ)
  (h1 : silver_amount = 1.5)
  (h2 : gold_amount = 2 * silver_amount)
  (h3 : gold_silver_price_ratio = 50)
  (h4 : total_spent = 3030)
  (h5 : silver_amount * silver_cost + gold_amount * (gold_silver_price_ratio * silver_cost) = total_spent) :
  silver_cost = 20 :=
by sorry

end NUMINAMATH_CALUDE_silver_cost_per_ounce_l2761_276122


namespace NUMINAMATH_CALUDE_extreme_value_implies_parameters_l2761_276190

/-- The function f with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2

/-- Theorem stating that if f has an extreme value of 10 at x=1, then (a,b) = (-4,11) -/
theorem extreme_value_implies_parameters
  (a b : ℝ)
  (extreme_value : f a b 1 = 10)
  (is_extreme : ∀ x, f a b x ≤ f a b 1) :
  a = -4 ∧ b = 11 := by
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_parameters_l2761_276190


namespace NUMINAMATH_CALUDE_greatest_sum_is_correct_l2761_276181

/-- The greatest possible sum of two consecutive integers whose product is less than 500 -/
def greatest_sum : ℕ := 43

/-- Predicate to check if two consecutive integers have a product less than 500 -/
def valid_pair (n : ℕ) : Prop := n * (n + 1) < 500

theorem greatest_sum_is_correct :
  (∀ n : ℕ, valid_pair n → n + (n + 1) ≤ greatest_sum) ∧
  (∃ n : ℕ, valid_pair n ∧ n + (n + 1) = greatest_sum) :=
sorry

end NUMINAMATH_CALUDE_greatest_sum_is_correct_l2761_276181


namespace NUMINAMATH_CALUDE_tile_border_ratio_l2761_276169

theorem tile_border_ratio (s d : ℝ) (h_positive : s > 0 ∧ d > 0) : 
  (15 * s)^2 / ((15 * s + 2 * 15 * d)^2) = 3/4 → d/s = 1/13 := by
  sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l2761_276169


namespace NUMINAMATH_CALUDE_problem_solution_l2761_276162

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x) / (a * x)

theorem problem_solution (a : ℝ) (h_a : a > 0) :
  (∀ x > 0, ∀ y > 0, x < Real.exp 1 → y > Real.exp 1 → f a x < f a y) ∧
  (∀ x > 0, f a x ≤ x - 1/a → a ≥ 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → x₂ * Real.log x₁ + x₁ * Real.log x₂ = 0 → x₁ + x₂ > 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2761_276162


namespace NUMINAMATH_CALUDE_audrey_sleep_theorem_l2761_276130

theorem audrey_sleep_theorem (total_sleep : ℝ) (dream_ratio : ℝ) 
  (h1 : total_sleep = 10)
  (h2 : dream_ratio = 2/5) : 
  total_sleep - (dream_ratio * total_sleep) = 6 := by
  sorry

end NUMINAMATH_CALUDE_audrey_sleep_theorem_l2761_276130


namespace NUMINAMATH_CALUDE_circle_center_is_neg_two_three_l2761_276117

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + 4*x + y^2 - 6*y + 1 = 0

/-- The center of a circle -/
def CircleCenter : ℝ × ℝ := (-2, 3)

/-- Theorem: The center of the circle with the given equation is (-2, 3) -/
theorem circle_center_is_neg_two_three :
  ∀ x y : ℝ, CircleEquation x y ↔ (x + 2)^2 + (y - 3)^2 = 12 :=
sorry

end NUMINAMATH_CALUDE_circle_center_is_neg_two_three_l2761_276117


namespace NUMINAMATH_CALUDE_sin_2x_eq_cos_x_div_2_solutions_l2761_276137

theorem sin_2x_eq_cos_x_div_2_solutions (x : ℝ) : 
  ∃! (s : Finset ℝ), s.card = 4 ∧ (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * Real.pi) ∧
    (∀ x ∈ s, Real.sin (2 * x) = Real.cos (x / 2)) ∧
    (∀ y, 0 ≤ y ∧ y ≤ 2 * Real.pi ∧ Real.sin (2 * y) = Real.cos (y / 2) → y ∈ s) :=
sorry

end NUMINAMATH_CALUDE_sin_2x_eq_cos_x_div_2_solutions_l2761_276137


namespace NUMINAMATH_CALUDE_diagonal_cut_color_distribution_l2761_276116

/-- Represents the color distribution of a scarf --/
structure ColorDistribution where
  white : ℚ
  grey : ℚ
  black : ℚ

/-- Represents a square scarf --/
structure SquareScarf where
  side_length : ℚ
  black_area : ℚ
  grey_area : ℚ

/-- Represents a triangular scarf obtained by cutting a square scarf diagonally --/
structure TriangularScarf where
  color_distribution : ColorDistribution

def diagonal_cut (s : SquareScarf) : (TriangularScarf × TriangularScarf) :=
  sorry

theorem diagonal_cut_color_distribution 
  (s : SquareScarf) 
  (h1 : s.black_area = 1/6) 
  (h2 : s.grey_area = 1/3) :
  let (t1, t2) := diagonal_cut s
  t1.color_distribution = { white := 3/4, grey := 2/9, black := 1/36 } ∧
  t2.color_distribution = { white := 1/4, grey := 4/9, black := 11/36 } :=
sorry

end NUMINAMATH_CALUDE_diagonal_cut_color_distribution_l2761_276116


namespace NUMINAMATH_CALUDE_systematic_sampling_l2761_276118

theorem systematic_sampling 
  (total_employees : Nat) 
  (sample_size : Nat) 
  (fifth_sample : Nat) :
  total_employees = 200 →
  sample_size = 40 →
  fifth_sample = 23 →
  ∃ (start : Nat), 
    start + 4 * (total_employees / sample_size) = fifth_sample ∧
    start + 7 * (total_employees / sample_size) = 38 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l2761_276118


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2761_276139

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2761_276139


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l2761_276146

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := 3*x - y - 9 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersection :
  ∃ (a1 a2 b1 b2 : ℝ),
    circle1 a1 a2 ∧ circle1 b1 b2 ∧
    circle2 a1 a2 ∧ circle2 b1 b2 ∧
    (∀ x y : ℝ, perp_bisector x y ↔ 
      ((x - a1)^2 + (y - a2)^2 = (x - b1)^2 + (y - b2)^2)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_l2761_276146


namespace NUMINAMATH_CALUDE_intersection_on_diagonal_l2761_276133

-- Define the basic geometric objects
variable (A B C D K L M P Q : EuclideanPlane)

-- Define the quadrilateral ABCD
def is_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define points K, L, M on sides or their extensions
def on_side_or_extension (X Y Z : EuclideanPlane) : Prop := sorry

-- Define the intersection of two lines
def intersect (W X Y Z : EuclideanPlane) : EuclideanPlane := sorry

-- Define a point lying on a line
def lies_on (X Y Z : EuclideanPlane) : Prop := sorry

-- Theorem statement
theorem intersection_on_diagonal 
  (h_quad : is_quadrilateral A B C D)
  (h_K : on_side_or_extension K A B)
  (h_L : on_side_or_extension L B C)
  (h_M : on_side_or_extension M C D)
  (h_P : P = intersect K L A C)
  (h_Q : Q = intersect L M B D) :
  lies_on (intersect K Q M P) A D := by sorry

end NUMINAMATH_CALUDE_intersection_on_diagonal_l2761_276133


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2761_276127

theorem complex_fraction_simplification :
  (3 / 7 + 5 / 8) / (5 / 12 + 7 / 15) = 15 / 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2761_276127


namespace NUMINAMATH_CALUDE_intersection_and_complement_l2761_276164

def A : Set ℝ := {x | -4 ≤ x ∧ x ≤ -2}
def B : Set ℝ := {x | x + 3 ≥ 0}

theorem intersection_and_complement :
  (A ∩ B = {x | -3 ≤ x ∧ x ≤ -2}) ∧
  (Set.compl (A ∩ B) = {x | x < -3 ∨ x > -2}) := by sorry

end NUMINAMATH_CALUDE_intersection_and_complement_l2761_276164


namespace NUMINAMATH_CALUDE_common_rational_root_exists_l2761_276171

theorem common_rational_root_exists :
  ∃ (r : ℚ) (a b c d e f g : ℚ),
    (60 * r^4 + a * r^3 + b * r^2 + c * r + 20 = 0) ∧
    (20 * r^5 + d * r^4 + e * r^3 + f * r^2 + g * r + 60 = 0) ∧
    (r > 0) ∧
    (∀ n : ℤ, r ≠ n) ∧
    (r = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_common_rational_root_exists_l2761_276171


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2761_276109

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (4, -2)
  let b : ℝ × ℝ := (x, 5)
  parallel a b → x = -10 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2761_276109


namespace NUMINAMATH_CALUDE_quadratic_linear_system_solution_l2761_276161

theorem quadratic_linear_system_solution :
  ∀ x y : ℝ,
  (x^2 - 6*x + 8 = 0) ∧ (y + 2*x = 12) →
  ((x = 4 ∧ y = 4) ∨ (x = 2 ∧ y = 8)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_linear_system_solution_l2761_276161


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2761_276172

/-- A geometric sequence with common ratio q > 1 -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 1 + a 6 = 33 →
  a 2 * a 5 = 32 →
  a 3 + a 8 = 132 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2761_276172


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2761_276153

theorem polynomial_factorization :
  ∀ x : ℝ, x^2 - 6*x + 9 - 64*x^4 = (-8*x^2 + x - 3) * (8*x^2 + x - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2761_276153


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2761_276102

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (1 + 2*i) / i
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2761_276102


namespace NUMINAMATH_CALUDE_ordering_of_constants_l2761_276188

theorem ordering_of_constants : 
  Real.log 17 < 3 ∧ 3 < Real.exp (Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ordering_of_constants_l2761_276188


namespace NUMINAMATH_CALUDE_cosine_equality_l2761_276191

theorem cosine_equality (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l2761_276191


namespace NUMINAMATH_CALUDE_pen_cost_l2761_276193

theorem pen_cost (pen ink : ℝ) 
  (total_cost : pen + ink = 2.50)
  (price_difference : pen = ink + 2) : 
  pen = 2.25 := by sorry

end NUMINAMATH_CALUDE_pen_cost_l2761_276193


namespace NUMINAMATH_CALUDE_female_democrats_count_l2761_276177

theorem female_democrats_count (total_participants : ℕ) 
  (female_participants male_participants : ℕ) 
  (female_democrats male_democrats : ℕ) : 
  total_participants = 840 →
  female_participants + male_participants = total_participants →
  female_democrats = female_participants / 2 →
  male_democrats = male_participants / 4 →
  female_democrats + male_democrats = total_participants / 3 →
  female_democrats = 140 := by
  sorry

end NUMINAMATH_CALUDE_female_democrats_count_l2761_276177


namespace NUMINAMATH_CALUDE_negation_equivalence_l2761_276156

theorem negation_equivalence (x : ℝ) :
  ¬(x^2 - x + 3 > 0) ↔ (x^2 - x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2761_276156


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_product_upper_bound_l2761_276148

-- Define the set of positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- Define the condition x^2 + y^2 = x + y
def SumSquaresEqualSum (x y : ℝ) : Prop := x^2 + y^2 = x + y

-- Theorem 1: Minimum value of 1/x + 1/y is 2
theorem min_reciprocal_sum (x y : ℝ) (hx : x ∈ PositiveReals) (hy : y ∈ PositiveReals)
  (h : SumSquaresEqualSum x y) :
  1/x + 1/y ≥ 2 ∧ ∃ x y, x ∈ PositiveReals ∧ y ∈ PositiveReals ∧ SumSquaresEqualSum x y ∧ 1/x + 1/y = 2 :=
sorry

-- Theorem 2: (x+1)(y+1) < 5 for all x, y satisfying the conditions
theorem product_upper_bound (x y : ℝ) (hx : x ∈ PositiveReals) (hy : y ∈ PositiveReals)
  (h : SumSquaresEqualSum x y) :
  (x + 1) * (y + 1) < 5 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_product_upper_bound_l2761_276148


namespace NUMINAMATH_CALUDE_negation_of_sine_inequality_l2761_276198

theorem negation_of_sine_inequality :
  (¬ ∀ x : ℝ, |Real.sin x| < 1) ↔ (∃ x : ℝ, |Real.sin x| ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_sine_inequality_l2761_276198


namespace NUMINAMATH_CALUDE_max_a_value_l2761_276140

/-- Given integers a and b satisfying the conditions, the maximum value of a is 23 -/
theorem max_a_value (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 143) : a ≤ 23 ∧ ∃ (a₀ b₀ : ℤ), a₀ > b₀ ∧ b₀ > 0 ∧ a₀ + b₀ + a₀ * b₀ = 143 ∧ a₀ = 23 := by
  sorry

end NUMINAMATH_CALUDE_max_a_value_l2761_276140


namespace NUMINAMATH_CALUDE_cosine_values_for_special_angle_l2761_276111

theorem cosine_values_for_special_angle (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/3)) 
  (h2 : Real.sqrt 6 * Real.sin α + Real.sqrt 2 * Real.cos α = Real.sqrt 3) : 
  (Real.cos (α + π/6) = Real.sqrt 10 / 4) ∧ 
  (Real.cos (2*α + π/12) = (Real.sqrt 30 + Real.sqrt 2) / 8) := by
  sorry

end NUMINAMATH_CALUDE_cosine_values_for_special_angle_l2761_276111


namespace NUMINAMATH_CALUDE_dice_roll_probability_l2761_276142

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The probability of rolling a 2 on the first die -/
def prob_first_die_2 : ℚ := 1 / num_sides

/-- The probability of rolling a 5 or 6 on the second die -/
def prob_second_die_5_or_6 : ℚ := 2 / num_sides

/-- The probability of the combined event -/
def prob_combined : ℚ := prob_first_die_2 * prob_second_die_5_or_6

theorem dice_roll_probability : prob_combined = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l2761_276142


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2761_276128

theorem polynomial_factorization (y : ℝ) : 3 * y^2 - 27 = 3 * (y + 3) * (y - 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2761_276128


namespace NUMINAMATH_CALUDE_log_division_simplification_l2761_276144

theorem log_division_simplification : 
  Real.log 16 / Real.log (1 / 16) = -1 := by sorry

end NUMINAMATH_CALUDE_log_division_simplification_l2761_276144


namespace NUMINAMATH_CALUDE_cat_food_insufficiency_l2761_276173

theorem cat_food_insufficiency (B S : ℝ) 
  (h1 : B > S) 
  (h2 : B < 2 * S) : 
  4 * B + 4 * S < 3 * (B + 2 * S) := by
  sorry

end NUMINAMATH_CALUDE_cat_food_insufficiency_l2761_276173


namespace NUMINAMATH_CALUDE_sum_distribution_l2761_276170

/-- The sum distribution problem -/
theorem sum_distribution (p q r s t : ℝ) : 
  (q = 0.75 * p) →  -- q gets 75 cents for each dollar p gets
  (r = 0.50 * p) →  -- r gets 50 cents for each dollar p gets
  (s = 0.25 * p) →  -- s gets 25 cents for each dollar p gets
  (t = 0.10 * p) →  -- t gets 10 cents for each dollar p gets
  (s = 25) →        -- The share of s is 25 dollars
  (p + q + r + s + t = 260) := by  -- The total sum is 260 dollars
sorry


end NUMINAMATH_CALUDE_sum_distribution_l2761_276170


namespace NUMINAMATH_CALUDE_absent_student_percentage_l2761_276192

theorem absent_student_percentage (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_students = 120)
  (h2 : boys = 70)
  (h3 : girls = 50)
  (h4 : total_students = boys + girls)
  (boys_absent_fraction : ℚ)
  (girls_absent_fraction : ℚ)
  (h5 : boys_absent_fraction = 1 / 7)
  (h6 : girls_absent_fraction = 1 / 5) :
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_absent_student_percentage_l2761_276192


namespace NUMINAMATH_CALUDE_original_average_l2761_276167

theorem original_average (n : ℕ) (A : ℝ) (h1 : n = 7) (h2 : (5 * n * A) / n = 100) : A = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_average_l2761_276167


namespace NUMINAMATH_CALUDE_smallest_perfect_square_sum_24_consecutive_l2761_276152

/-- The sum of 24 consecutive positive integers starting from n -/
def sum_24_consecutive (n : ℕ) : ℕ := 12 * (2 * n + 23)

/-- A number is a perfect square -/
def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

theorem smallest_perfect_square_sum_24_consecutive :
  (∃ n : ℕ, is_perfect_square (sum_24_consecutive n)) ∧
  (∀ n : ℕ, is_perfect_square (sum_24_consecutive n) → sum_24_consecutive n ≥ 300) :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_sum_24_consecutive_l2761_276152


namespace NUMINAMATH_CALUDE_candy_probability_l2761_276129

theorem candy_probability : 
  let total_candies : ℕ := 12
  let red_candies : ℕ := 5
  let blue_candies : ℕ := 2
  let green_candies : ℕ := 5
  let pick_count : ℕ := 4
  let favorable_outcomes : ℕ := (red_candies.choose 3) * (blue_candies + green_candies)
  let total_outcomes : ℕ := total_candies.choose pick_count
  (favorable_outcomes : ℚ) / total_outcomes = 14 / 99 := by sorry

end NUMINAMATH_CALUDE_candy_probability_l2761_276129


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2761_276126

theorem smallest_integer_with_remainder_one : ∃! k : ℕ,
  k > 1 ∧
  k % 19 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 19 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  use 400
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2761_276126


namespace NUMINAMATH_CALUDE_gravelling_rate_calculation_l2761_276182

/-- Given a rectangular lawn with two intersecting roads, calculate the rate per square meter for gravelling the roads. -/
theorem gravelling_rate_calculation (lawn_length lawn_width road_width total_cost : ℝ) 
  (h1 : lawn_length = 70)
  (h2 : lawn_width = 30)
  (h3 : road_width = 5)
  (h4 : total_cost = 1900) : 
  total_cost / ((lawn_length * road_width) + (lawn_width * road_width) - (road_width * road_width)) = 4 := by
  sorry

#check gravelling_rate_calculation

end NUMINAMATH_CALUDE_gravelling_rate_calculation_l2761_276182


namespace NUMINAMATH_CALUDE_smallest_factor_product_l2761_276160

theorem smallest_factor_product (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 1764 → ¬(2^5 ∣ 936 * m ∧ 3^3 ∣ 936 * m ∧ 14^2 ∣ 936 * m)) ∧
  (2^5 ∣ 936 * 1764 ∧ 3^3 ∣ 936 * 1764 ∧ 14^2 ∣ 936 * 1764) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_product_l2761_276160


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2761_276141

theorem quadratic_inequality_solution_set (a : ℝ) (ha : a < 0) :
  {x : ℝ | x^2 - 2*a*x - 3*a^2 < 0} = {x : ℝ | 3*a < x ∧ x < -a} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2761_276141


namespace NUMINAMATH_CALUDE_factorial_sum_equals_1190_l2761_276186

theorem factorial_sum_equals_1190 : 
  (Nat.factorial 16) / ((Nat.factorial 6) * (Nat.factorial 10)) + 
  (Nat.factorial 11) / ((Nat.factorial 6) * (Nat.factorial 5)) = 1190 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_1190_l2761_276186


namespace NUMINAMATH_CALUDE_power_of_64_l2761_276121

theorem power_of_64 : (64 : ℝ) ^ (5/6) = 32 := by sorry

end NUMINAMATH_CALUDE_power_of_64_l2761_276121


namespace NUMINAMATH_CALUDE_floor_sqrt_95_l2761_276175

theorem floor_sqrt_95 : ⌊Real.sqrt 95⌋ = 9 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_95_l2761_276175


namespace NUMINAMATH_CALUDE_total_amount_in_euros_l2761_276107

/-- Represents the distribution of shares among w, x, y, and z -/
structure ShareDistribution where
  w : ℝ
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines the exchange rate from dollars to euros -/
def exchange_rate : ℝ := 0.85

/-- Defines the share ratios relative to w -/
def share_ratios : ShareDistribution := {
  w := 1,
  x := 0.75,
  y := 0.5,
  z := 0.25
}

/-- Theorem stating the total amount in euros given the conditions -/
theorem total_amount_in_euros : 
  ∀ (shares : ShareDistribution),
  shares.w * exchange_rate = 15 →
  (shares.w + shares.x + shares.y + shares.z) * exchange_rate = 37.5 :=
by
  sorry

#check total_amount_in_euros

end NUMINAMATH_CALUDE_total_amount_in_euros_l2761_276107


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2761_276134

def set_A : Set ℝ := {x | 2 * x - 1 ≤ 0}
def set_B : Set ℝ := {x | 1 / x > 1}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | 0 < x ∧ x ≤ 1/2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2761_276134


namespace NUMINAMATH_CALUDE_sale_discount_proof_l2761_276100

theorem sale_discount_proof (original_price : ℝ) : 
  let sale_price := 0.5 * original_price
  let coupon_discount := 0.2
  let final_price := (1 - coupon_discount) * sale_price
  final_price = 0.4 * original_price :=
by sorry

end NUMINAMATH_CALUDE_sale_discount_proof_l2761_276100


namespace NUMINAMATH_CALUDE_conference_handshakes_count_l2761_276176

/-- The number of handshakes at a conference of wizards and elves -/
def conference_handshakes (num_wizards num_elves : ℕ) : ℕ :=
  let wizard_handshakes := num_wizards.choose 2
  let elf_wizard_handshakes := num_wizards * num_elves
  wizard_handshakes + elf_wizard_handshakes

/-- Theorem: The total number of handshakes at the conference is 750 -/
theorem conference_handshakes_count :
  conference_handshakes 25 18 = 750 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_count_l2761_276176


namespace NUMINAMATH_CALUDE_tangent_line_circle_parabola_l2761_276179

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A parabola in the xy-plane -/
structure Parabola where
  vertex : ℝ × ℝ
  a : ℝ

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a line is tangent to a circle at a given point -/
def isTangentToCircle (l : Line) (c : Circle) (p : ℝ × ℝ) : Prop := sorry

/-- Checks if a line is tangent to a parabola at a given point -/
def isTangentToParabola (l : Line) (p : Parabola) (point : ℝ × ℝ) : Prop := sorry

/-- The main theorem -/
theorem tangent_line_circle_parabola (c : Circle) (p : Parabola) (l : Line) (point : ℝ × ℝ) :
  c.center = (1, 2) →
  c.radius^2 = 1^2 + 2^2 + a →
  p.vertex = (0, 0) →
  p.a = 1/4 →
  isTangentToCircle l c point →
  isTangentToParabola l p point →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_circle_parabola_l2761_276179


namespace NUMINAMATH_CALUDE_midpoint_product_l2761_276157

/-- Given that C = (4, 3) is the midpoint of line segment AB, where A = (2, 6) and B = (x, y), prove that xy = 0 -/
theorem midpoint_product (x y : ℝ) : 
  (4 : ℝ) = (2 + x) / 2 → 
  (3 : ℝ) = (6 + y) / 2 → 
  x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_product_l2761_276157


namespace NUMINAMATH_CALUDE_sum_of_cubic_difference_l2761_276166

theorem sum_of_cubic_difference (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 294 →
  a + b + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubic_difference_l2761_276166


namespace NUMINAMATH_CALUDE_milk_measurement_problem_l2761_276150

/-- Represents a container for milk -/
structure Container :=
  (capacity : ℕ)
  (content : ℕ)

/-- Represents the state of all containers -/
structure State :=
  (can1 : Container)
  (can2 : Container)
  (jug5 : Container)
  (jug4 : Container)

/-- Represents a pouring operation -/
inductive Operation
  | CanToJug : Container → Container → Operation
  | JugToJug : Container → Container → Operation
  | JugToCan : Container → Container → Operation

/-- The result of applying an operation to a state -/
def applyOperation (s : State) (op : Operation) : State :=
  sorry

/-- The initial state with two full 80-liter cans and empty jugs -/
def initialState : State :=
  { can1 := ⟨80, 80⟩
  , can2 := ⟨80, 80⟩
  , jug5 := ⟨5, 0⟩
  , jug4 := ⟨4, 0⟩ }

/-- The goal state with exactly 2 liters in each jug -/
def goalState : State :=
  { can1 := ⟨80, 80⟩
  , can2 := ⟨80, 76⟩
  , jug5 := ⟨5, 2⟩
  , jug4 := ⟨4, 2⟩ }

/-- Theorem stating that the goal state can be reached in exactly 9 operations -/
theorem milk_measurement_problem :
  ∃ (ops : List Operation),
    ops.length = 9 ∧
    (ops.foldl applyOperation initialState) = goalState :=
  sorry

end NUMINAMATH_CALUDE_milk_measurement_problem_l2761_276150


namespace NUMINAMATH_CALUDE_constant_width_max_length_l2761_276165

/-- A convex curve in a 2D plane. -/
structure ConvexCurve where
  -- Add necessary fields and conditions to define a convex curve
  is_convex : Bool
  diameter : ℝ
  length : ℝ

/-- A curve of constant width. -/
structure ConstantWidthCurve extends ConvexCurve where
  constant_width : ℝ
  is_constant_width : Bool

/-- The theorem stating that curves of constant width 1 have the greatest length among all convex curves of diameter 1. -/
theorem constant_width_max_length :
  ∀ (K : ConvexCurve),
    K.diameter = 1 →
    ∀ (C : ConstantWidthCurve),
      C.diameter = 1 →
      C.constant_width = 1 →
      C.is_constant_width →
      K.length ≤ C.length :=
sorry


end NUMINAMATH_CALUDE_constant_width_max_length_l2761_276165


namespace NUMINAMATH_CALUDE_cos_sin_sum_l2761_276145

theorem cos_sin_sum (x : ℝ) (h : Real.cos (x - π/3) = 1/3) :
  Real.cos (2*x - 5*π/3) + Real.sin (π/3 - x)^2 = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_l2761_276145


namespace NUMINAMATH_CALUDE_diagram3_illustrates_inflation_l2761_276143

/-- Represents a diagram showing economic data over time -/
structure EconomicDiagram where
  prices : ℕ → ℝ
  time : ℕ

/-- Definition of inflation -/
def is_inflation (d : EconomicDiagram) : Prop :=
  ∀ t₁ t₂, t₁ < t₂ → d.prices t₁ < d.prices t₂

/-- Diagram №3 from the problem -/
def diagram3 : EconomicDiagram :=
  sorry

/-- Theorem stating that Diagram №3 illustrates inflation -/
theorem diagram3_illustrates_inflation : is_inflation diagram3 := by
  sorry

end NUMINAMATH_CALUDE_diagram3_illustrates_inflation_l2761_276143


namespace NUMINAMATH_CALUDE_family_income_increase_l2761_276183

theorem family_income_increase (total_income : ℝ) 
  (masha_scholarship mother_salary father_salary grandfather_pension : ℝ) : 
  masha_scholarship + mother_salary + father_salary + grandfather_pension = total_income →
  masha_scholarship = 0.05 * total_income →
  mother_salary = 0.15 * total_income →
  father_salary = 0.25 * total_income →
  grandfather_pension = 0.55 * total_income :=
by
  sorry

#check family_income_increase

end NUMINAMATH_CALUDE_family_income_increase_l2761_276183


namespace NUMINAMATH_CALUDE_y_equation_solution_l2761_276199

theorem y_equation_solution (y : ℝ) (c d : ℕ+) 
  (h1 : y^2 + 2*y + 2/y + 1/y^2 = 20)
  (h2 : y = c + Real.sqrt d) : 
  c + d = 5 := by
sorry

end NUMINAMATH_CALUDE_y_equation_solution_l2761_276199


namespace NUMINAMATH_CALUDE_cookie_jar_problem_l2761_276112

theorem cookie_jar_problem :
  ∃ (n c : ℕ),
    12 ≤ n ∧ n ≤ 36 ∧
    (n - 1) * c + (c + 1) = 1000 ∧
    n + (c + 1) = 65 :=
by sorry

end NUMINAMATH_CALUDE_cookie_jar_problem_l2761_276112


namespace NUMINAMATH_CALUDE_max_t_value_l2761_276163

theorem max_t_value (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 10 →
  k < m → m < r → r < s → s < t →
  r ≤ 13 →
  t ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_max_t_value_l2761_276163


namespace NUMINAMATH_CALUDE_probability_same_gender_example_l2761_276120

/-- Represents a school with a certain number of male and female teachers. -/
structure School :=
  (male_count : ℕ)
  (female_count : ℕ)

/-- Calculates the probability of selecting two teachers of the same gender
    from two given schools. -/
def probability_same_gender (school_a school_b : School) : ℚ :=
  let total_outcomes := (school_a.male_count + school_a.female_count) * (school_b.male_count + school_b.female_count)
  let same_gender_outcomes := school_a.male_count * school_b.male_count + school_a.female_count * school_b.female_count
  same_gender_outcomes / total_outcomes

/-- Theorem stating that the probability of selecting two teachers of the same gender
    from School A (2 males, 1 female) and School B (1 male, 2 females) is 4/9. -/
theorem probability_same_gender_example : 
  probability_same_gender ⟨2, 1⟩ ⟨1, 2⟩ = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_gender_example_l2761_276120


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2761_276195

theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10)
  (h2 : r - p = 20) :
  (q + r) / 2 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2761_276195


namespace NUMINAMATH_CALUDE_three_kopeck_count_l2761_276114

/-- Represents the denomination of a coin -/
inductive Denomination
| One
| Two
| Three

/-- Represents a row of coins -/
def CoinRow := List Denomination

/-- Checks if there's at least one coin between any two one-kopeck coins -/
def validOneKopeckSpacing (row : CoinRow) : Prop := sorry

/-- Checks if there's at least two coins between any two two-kopeck coins -/
def validTwoKopeckSpacing (row : CoinRow) : Prop := sorry

/-- Checks if there's at least three coins between any two three-kopeck coins -/
def validThreeKopeckSpacing (row : CoinRow) : Prop := sorry

/-- Counts the number of three-kopeck coins in the row -/
def countThreeKopecks (row : CoinRow) : Nat := sorry

theorem three_kopeck_count (row : CoinRow) :
  row.length = 101 →
  validOneKopeckSpacing row →
  validTwoKopeckSpacing row →
  validThreeKopeckSpacing row →
  (countThreeKopecks row = 25 ∨ countThreeKopecks row = 26) :=
by sorry

end NUMINAMATH_CALUDE_three_kopeck_count_l2761_276114


namespace NUMINAMATH_CALUDE_f_congruence_implies_input_congruence_l2761_276197

def f (x : ℤ) : ℤ := x^3 + 7*x^2 + 9*x + 10

theorem f_congruence_implies_input_congruence :
  ∀ (a b : ℤ), f a ≡ f b [ZMOD 11] → a ≡ b [ZMOD 11] := by
  sorry

end NUMINAMATH_CALUDE_f_congruence_implies_input_congruence_l2761_276197


namespace NUMINAMATH_CALUDE_tan_sum_identity_l2761_276138

theorem tan_sum_identity (α : Real) (h : Real.tan α = Real.sqrt 2) :
  Real.tan (α + π / 4) = -3 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_identity_l2761_276138


namespace NUMINAMATH_CALUDE_shopkeeper_theft_loss_l2761_276168

theorem shopkeeper_theft_loss (cost_price : ℝ) (profit_rate : ℝ) (loss_rate : ℝ) : 
  profit_rate = 0.1 →
  loss_rate = 0.12 →
  cost_price > 0 →
  let selling_price := cost_price * (1 + profit_rate)
  let loss_value := selling_price * loss_rate
  let loss_percentage := (loss_value / cost_price) * 100
  loss_percentage = 13.2 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_theft_loss_l2761_276168


namespace NUMINAMATH_CALUDE_binary_product_example_l2761_276135

/-- Given two binary numbers represented as natural numbers, 
    this function computes their product in binary representation -/
def binary_multiply (a b : ℕ) : ℕ := 
  (a.digits 2).foldl (λ acc d => acc * 2 + d) 0 * 
  (b.digits 2).foldl (λ acc d => acc * 2 + d) 0

/-- Theorem stating that the product of 1101₂ and 1011₂ is 10011011₂ -/
theorem binary_product_example : binary_multiply 13 11 = 155 := by
  sorry

#eval binary_multiply 13 11

end NUMINAMATH_CALUDE_binary_product_example_l2761_276135


namespace NUMINAMATH_CALUDE_cannot_afford_both_phones_l2761_276154

/-- Represents the financial situation of a couple --/
structure FinancialSituation where
  income : ℕ
  expenses : ℕ
  phoneACost : ℕ
  phoneBCost : ℕ

/-- Determines if a couple can afford to buy both phones --/
def canAffordBothPhones (situation : FinancialSituation) : Prop :=
  situation.income - situation.expenses ≥ situation.phoneACost + situation.phoneBCost

/-- The specific financial situation of Alexander and Natalia --/
def alexanderAndNatalia : FinancialSituation :=
  { income := 186000
    expenses := 119000
    phoneACost := 57000
    phoneBCost := 37000 }

/-- Theorem stating that Alexander and Natalia cannot afford both phones --/
theorem cannot_afford_both_phones :
  ¬(canAffordBothPhones alexanderAndNatalia) := by
  sorry


end NUMINAMATH_CALUDE_cannot_afford_both_phones_l2761_276154


namespace NUMINAMATH_CALUDE_field_trip_girls_fraction_l2761_276119

theorem field_trip_girls_fraction (b : ℚ) (g : ℚ) : 
  g = 2 * b →  -- There are twice as many girls as boys
  (5 / 6 * g) / ((5 / 6 * g) + (1 / 2 * b)) = 10 / 13 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_girls_fraction_l2761_276119


namespace NUMINAMATH_CALUDE_train_speed_without_stoppages_l2761_276194

/-- The average speed of a train without stoppages, given its speed with stoppages and stoppage time. -/
theorem train_speed_without_stoppages
  (distance : ℝ) -- The distance traveled by the train
  (speed_with_stoppages : ℝ) -- The average speed of the train with stoppages
  (stoppage_time : ℝ) -- The time the train stops per hour
  (h1 : speed_with_stoppages = 360) -- The given speed with stoppages
  (h2 : stoppage_time = 6) -- The given stoppage time in minutes
  (h3 : distance > 0) -- Ensure the distance is positive
  : ∃ (speed_without_stoppages : ℝ),
    speed_without_stoppages = 400 ∧
    distance = speed_with_stoppages * 1 ∧
    distance = speed_without_stoppages * (1 - stoppage_time / 60) := by
  sorry

end NUMINAMATH_CALUDE_train_speed_without_stoppages_l2761_276194


namespace NUMINAMATH_CALUDE_sum_x_y_equals_nine_l2761_276155

theorem sum_x_y_equals_nine (x y : ℝ) (h : y = Real.sqrt (x - 5) + Real.sqrt (5 - x) + 4) : 
  x + y = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_nine_l2761_276155


namespace NUMINAMATH_CALUDE_platform_length_calculation_l2761_276124

/-- Calculates the length of a platform given train and crossing information -/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 470 →
  train_speed_kmh = 55 →
  crossing_time = 64.79481641468682 →
  ∃ (platform_length : ℝ), abs (platform_length - 520) < 0.1 :=
by
  sorry

end NUMINAMATH_CALUDE_platform_length_calculation_l2761_276124


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2761_276131

theorem complex_equation_solution (z : ℂ) :
  (Complex.I / (z - 1) = (1 : ℂ) / 2) → z = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2761_276131


namespace NUMINAMATH_CALUDE_parabola_inequality_l2761_276132

/-- Prove that for a parabola y = ax^2 + bx + c with a < 0, passing through points (-1, 0) and (m, 0) where 3 < m < 4, the inequality 3a + c > 0 holds. -/
theorem parabola_inequality (a b c m : ℝ) : 
  a < 0 → 
  3 < m → 
  m < 4 → 
  a * (-1)^2 + b * (-1) + c = 0 → 
  a * m^2 + b * m + c = 0 → 
  3 * a + c > 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_inequality_l2761_276132


namespace NUMINAMATH_CALUDE_sequence_length_l2761_276185

/-- Proves that an arithmetic sequence starting at 2.5, ending at 67.5, with a common difference of 5, has 14 terms. -/
theorem sequence_length : 
  ∀ (a : ℚ) (d : ℚ) (last : ℚ) (n : ℕ),
  a = 2.5 ∧ d = 5 ∧ last = 67.5 →
  last = a + (n - 1) * d →
  n = 14 := by
sorry

end NUMINAMATH_CALUDE_sequence_length_l2761_276185


namespace NUMINAMATH_CALUDE_f_has_max_and_min_l2761_276158

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*(a+2)

/-- Theorem stating the range of a for which f(x) has both a maximum and a minimum -/
theorem f_has_max_and_min (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f_derivative a x₁ = 0 ∧ f_derivative a x₂ = 0) ↔ 
  (a < -1 ∨ a > 2) :=
sorry

end NUMINAMATH_CALUDE_f_has_max_and_min_l2761_276158


namespace NUMINAMATH_CALUDE_martha_cards_l2761_276105

theorem martha_cards (x : ℝ) : x - 3.0 = 73 → x = 76 := by
  sorry

end NUMINAMATH_CALUDE_martha_cards_l2761_276105


namespace NUMINAMATH_CALUDE_money_game_total_determinable_l2761_276104

/-- Represents the money redistribution game among four friends -/
structure MoneyGame where
  -- Initial amounts
  amy_initial : ℝ
  jan_initial : ℝ
  toy_initial : ℝ
  ben_initial : ℝ
  -- Final amounts
  amy_final : ℝ
  jan_final : ℝ
  toy_final : ℝ
  ben_final : ℝ

/-- The rules of the money redistribution game -/
def redistribute (game : MoneyGame) : Prop :=
  -- Amy's turn
  let amy_after := game.amy_initial - game.jan_initial - game.toy_initial - game.ben_initial
  let jan_after1 := 2 * game.jan_initial
  let toy_after1 := 2 * game.toy_initial
  let ben_after1 := 2 * game.ben_initial
  -- Jan's turn
  let amy_after2 := 2 * amy_after
  let toy_after2 := 2 * toy_after1
  let ben_after2 := 2 * ben_after1
  let jan_after2 := jan_after1 - (amy_after + toy_after1 + ben_after1)
  -- Toy's turn
  let amy_after3 := 2 * amy_after2
  let jan_after3 := 2 * jan_after2
  let ben_after3 := 2 * ben_after2
  let toy_after3 := toy_after2 - (amy_after2 + jan_after2 + ben_after2)
  -- Ben's turn
  game.amy_final = 2 * amy_after3 ∧
  game.jan_final = 2 * jan_after3 ∧
  game.toy_final = 2 * toy_after3 ∧
  game.ben_final = ben_after3 - (amy_after3 + jan_after3 + toy_after3)

/-- The theorem statement -/
theorem money_game_total_determinable (game : MoneyGame) :
  game.toy_initial = 24 ∧ 
  game.toy_final = 96 ∧ 
  redistribute game → 
  ∃ total : ℝ, total = game.amy_final + game.jan_final + game.toy_final + game.ben_final :=
by sorry


end NUMINAMATH_CALUDE_money_game_total_determinable_l2761_276104


namespace NUMINAMATH_CALUDE_scientific_notation_of_rural_population_l2761_276184

theorem scientific_notation_of_rural_population :
  ∃ (x : ℝ), x = 42.39 * 10^6 ∧ x = 4.239 * 10^7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_rural_population_l2761_276184


namespace NUMINAMATH_CALUDE_city_reading_survey_suitable_class_myopia_survey_not_suitable_grade_exercise_survey_not_suitable_class_temperature_survey_not_suitable_l2761_276113

/-- Represents a survey scenario -/
inductive SurveyScenario
  | class_myopia
  | grade_morning_exercise
  | class_body_temperature
  | city_extracurricular_reading

/-- Determines if a survey scenario is suitable for sampling -/
def suitable_for_sampling (scenario : SurveyScenario) : Prop :=
  match scenario with
  | SurveyScenario.city_extracurricular_reading => True
  | _ => False

/-- Theorem stating that the city-wide extracurricular reading survey is suitable for sampling -/
theorem city_reading_survey_suitable :
  suitable_for_sampling SurveyScenario.city_extracurricular_reading :=
by
  sorry

/-- Theorem stating that the class myopia survey is not suitable for sampling -/
theorem class_myopia_survey_not_suitable :
  ¬ suitable_for_sampling SurveyScenario.class_myopia :=
by
  sorry

/-- Theorem stating that the grade morning exercise survey is not suitable for sampling -/
theorem grade_exercise_survey_not_suitable :
  ¬ suitable_for_sampling SurveyScenario.grade_morning_exercise :=
by
  sorry

/-- Theorem stating that the class body temperature survey is not suitable for sampling -/
theorem class_temperature_survey_not_suitable :
  ¬ suitable_for_sampling SurveyScenario.class_body_temperature :=
by
  sorry

end NUMINAMATH_CALUDE_city_reading_survey_suitable_class_myopia_survey_not_suitable_grade_exercise_survey_not_suitable_class_temperature_survey_not_suitable_l2761_276113


namespace NUMINAMATH_CALUDE_ending_number_proof_l2761_276125

theorem ending_number_proof (n : ℕ) : 
  (∃ k : ℕ, n = 7 * k) →  -- n is divisible by 7
  n ≥ 21 →               -- n is at least 21 (first number after 18 divisible by 7)
  (21 + n) / 2 = 77/2 →  -- average of arithmetic sequence is 38.5
  n = 56 := by
sorry

end NUMINAMATH_CALUDE_ending_number_proof_l2761_276125


namespace NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_bound_l2761_276187

/-- A right circular cylinder with volume 1 -/
structure RightCircularCylinder where
  radius : ℝ
  height : ℝ
  volume_eq_one : π * radius^2 * height = 1

/-- A tetrahedron inscribed in a right circular cylinder -/
structure InscribedTetrahedron (c : RightCircularCylinder) where
  volume : ℝ
  is_inscribed : volume ≤ π * c.radius^2 * c.height

/-- The volume of any tetrahedron inscribed in a right circular cylinder 
    with volume 1 does not exceed 2/(3π) -/
theorem inscribed_tetrahedron_volume_bound 
  (c : RightCircularCylinder) 
  (t : InscribedTetrahedron c) : 
  t.volume ≤ 2 / (3 * π) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_tetrahedron_volume_bound_l2761_276187


namespace NUMINAMATH_CALUDE_merchant_discount_theorem_l2761_276178

/-- Proves that a merchant offering a 10% discount on a 20% markup results in an 8% profit -/
theorem merchant_discount_theorem (cost_price : ℝ) (markup_percentage : ℝ) (profit_percentage : ℝ) 
  (discount_percentage : ℝ) (h1 : markup_percentage = 20) (h2 : profit_percentage = 8) :
  discount_percentage = 10 ↔ 
    cost_price * (1 + markup_percentage / 100) * (1 - discount_percentage / 100) = 
    cost_price * (1 + profit_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_merchant_discount_theorem_l2761_276178


namespace NUMINAMATH_CALUDE_ellipse_semi_major_axis_l2761_276115

theorem ellipse_semi_major_axis (m : ℝ) : 
  (∀ x y : ℝ, x^2 / m + y^2 / 4 = 1) →  -- Ellipse equation
  (m > 4) →                            -- Semi-major axis > Semi-minor axis
  (m = (2 : ℝ)^2 + 4) →                -- Relationship between a^2, b^2, and c^2
  (m = 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_semi_major_axis_l2761_276115


namespace NUMINAMATH_CALUDE_sally_cracker_sales_l2761_276149

theorem sally_cracker_sales (saturday_sales : ℕ) (sunday_increase_percent : ℕ) : 
  saturday_sales = 60 → 
  sunday_increase_percent = 50 → 
  saturday_sales + (saturday_sales + sunday_increase_percent * saturday_sales / 100) = 150 := by
sorry

end NUMINAMATH_CALUDE_sally_cracker_sales_l2761_276149


namespace NUMINAMATH_CALUDE_binary_111011001001_equals_3785_l2761_276180

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_111011001001_equals_3785 :
  binary_to_decimal [true, false, false, true, false, false, true, true, false, true, true, true] = 3785 := by
  sorry

end NUMINAMATH_CALUDE_binary_111011001001_equals_3785_l2761_276180


namespace NUMINAMATH_CALUDE_negation_of_universal_quadrilateral_circumcircle_l2761_276101

-- Define the type for quadrilaterals
variable (Quadrilateral : Type)

-- Define the property of having a circumcircle
variable (has_circumcircle : Quadrilateral → Prop)

-- Theorem stating the negation of "Every quadrilateral has a circumcircle"
-- is equivalent to "Some quadrilaterals do not have a circumcircle"
theorem negation_of_universal_quadrilateral_circumcircle :
  ¬(∀ q : Quadrilateral, has_circumcircle q) ↔ ∃ q : Quadrilateral, ¬(has_circumcircle q) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quadrilateral_circumcircle_l2761_276101


namespace NUMINAMATH_CALUDE_tangent_line_at_point_l2761_276196

/-- The circle with equation x²+(y-1)²=2 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 2}

/-- The point (1,2) on the circle -/
def Point : ℝ × ℝ := (1, 2)

/-- The proposed tangent line with equation x+y-3=0 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 - 3 = 0}

theorem tangent_line_at_point :
  Point ∈ Circle ∧
  Point ∈ TangentLine ∧
  ∀ p ∈ Circle, p ≠ Point → p ∉ TangentLine :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_l2761_276196


namespace NUMINAMATH_CALUDE_cube_edge_length_l2761_276159

theorem cube_edge_length (surface_area : ℝ) (h : surface_area = 150) :
  ∃ edge_length : ℝ, edge_length > 0 ∧ 6 * edge_length^2 = surface_area ∧ edge_length = 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l2761_276159


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2761_276106

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point2D
  focus2 : Point2D
  majorAxis : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Returns true if the line intersects the ellipse at exactly one point -/
def intersectsAtOnePoint (e : Ellipse) (l : Line2D) : Prop :=
  sorry

theorem ellipse_major_axis_length :
  ∀ (e : Ellipse) (l : Line2D),
    e.focus1 = Point2D.mk (-2) 0 →
    e.focus2 = Point2D.mk 2 0 →
    l.a = 1 →
    l.b = Real.sqrt 3 →
    l.c = 4 →
    intersectsAtOnePoint e l →
    e.majorAxis = 2 * Real.sqrt 7 :=
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2761_276106


namespace NUMINAMATH_CALUDE_plant_branches_l2761_276147

theorem plant_branches (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_plant_branches_l2761_276147


namespace NUMINAMATH_CALUDE_total_bottles_l2761_276110

theorem total_bottles (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 9) (h2 : diet_soda = 8) : 
  regular_soda + diet_soda = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_bottles_l2761_276110


namespace NUMINAMATH_CALUDE_valid_a_equals_solution_set_l2761_276174

/-- R(k) is the remainder when k is divided by p -/
def R (p k : ℕ) : ℕ := k % p

/-- The set of valid a values -/
def valid_a_set (p : ℕ) : Set ℕ :=
  {a | a > 0 ∧ ∀ m ∈ Finset.range (p - 1), m + 1 + R p (m * a) > a}

/-- The set of solutions described in the problem -/
def solution_set (p : ℕ) : Set ℕ :=
  {p - 1} ∪ {a | ∃ s, 1 ≤ s ∧ s ≤ p - 1 ∧ a = (p - 1) / s}

theorem valid_a_equals_solution_set (p : ℕ) (hp : p.Prime ∧ p ≥ 5) :
  valid_a_set p = solution_set p := by
  sorry

end NUMINAMATH_CALUDE_valid_a_equals_solution_set_l2761_276174


namespace NUMINAMATH_CALUDE_trajectory_equation_tangent_relation_constant_triangle_area_l2761_276151

noncomputable def trajectory (x y : ℝ) : Prop :=
  Real.sqrt ((x + Real.sqrt 3)^2 + y^2) + Real.sqrt ((x - Real.sqrt 3)^2 + y^2) = 4

theorem trajectory_equation (x y : ℝ) :
  trajectory x y → x^2/4 + y^2 = 1 := by sorry

theorem tangent_relation (x y k m : ℝ) :
  trajectory x y → y = k*x + m → m^2 = 1 + 4*k^2 := by sorry

theorem constant_triangle_area (x y k m : ℝ) (A B : ℝ × ℝ) :
  trajectory x y →
  y = k*x + m →
  A.1^2/16 + A.2^2/4 = 1 →
  B.1^2/16 + B.2^2/4 = 1 →
  A.2 = k*A.1 + m →
  B.2 = k*B.1 + m →
  (1/2) * |m| * |A.1 - B.1| = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_trajectory_equation_tangent_relation_constant_triangle_area_l2761_276151


namespace NUMINAMATH_CALUDE_locus_of_centers_l2761_276189

/-- Circle C₁ with equation x² + y² = 4 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

/-- Circle C₃ with equation (x-1)² + y² = 25 -/
def C₃ : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + p.2^2 = 25}

/-- A circle is externally tangent to C₁ if the distance between their centers
    is equal to the sum of their radii -/
def externally_tangent_to_C₁ (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  center.1^2 + center.2^2 = (radius + 2)^2

/-- A circle is internally tangent to C₃ if the distance between their centers
    is equal to the difference of their radii -/
def internally_tangent_to_C₃ (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center.1 - 1)^2 + center.2^2 = (5 - radius)^2

/-- The locus of centers (a,b) of circles externally tangent to C₁ and
    internally tangent to C₃ satisfies the equation 5a² + 9b² + 80a - 400 = 0 -/
theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, externally_tangent_to_C₁ (a, b) r ∧ internally_tangent_to_C₃ (a, b) r) →
  5 * a^2 + 9 * b^2 + 80 * a - 400 = 0 := by
  sorry

end NUMINAMATH_CALUDE_locus_of_centers_l2761_276189
