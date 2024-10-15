import Mathlib

namespace NUMINAMATH_CALUDE_max_product_sum_11_l561_56131

theorem max_product_sum_11 :
  ∃ (a b : ℕ), a + b = 11 ∧
  ∀ (x y : ℕ), x + y = 11 → x * y ≤ a * b ∧
  a * b = 30 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_11_l561_56131


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l561_56164

/-- The sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

/-- The theorem statement -/
theorem arithmetic_sequence_first_term
  (d : ℚ)
  (h1 : d = 5)
  (h2 : ∃ (c : ℚ), ∀ (n : ℕ), n > 0 → arithmetic_sum a d (4 * n) / arithmetic_sum a d n = c) :
  a = 5 / 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l561_56164


namespace NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l561_56134

/-- The line l: y = x + 9 -/
def line_l (x : ℝ) : ℝ := x + 9

/-- The first focus of the ellipse -/
def F₁ : ℝ × ℝ := (-3, 0)

/-- The second focus of the ellipse -/
def F₂ : ℝ × ℝ := (3, 0)

/-- Definition of the ellipse equation -/
def is_ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The theorem stating the equation of the ellipse with shortest major axis -/
theorem shortest_major_axis_ellipse :
  ∃ (P : ℝ × ℝ),
    (P.2 = line_l P.1) ∧
    is_ellipse_equation 45 36 P.1 P.2 ∧
    ∀ (Q : ℝ × ℝ),
      (Q.2 = line_l Q.1) →
      (Q.1 - F₁.1)^2 + (Q.2 - F₁.2)^2 + (Q.1 - F₂.1)^2 + (Q.2 - F₂.2)^2 ≥
      (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 :=
by sorry

end NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l561_56134


namespace NUMINAMATH_CALUDE_real_roots_condition_one_root_triple_other_l561_56149

-- Define the system of equations
def system (x y a b : ℝ) : Prop :=
  x + y = a ∧ 1/x + 1/y = 1/b

-- Theorem for real roots condition
theorem real_roots_condition (a b : ℝ) :
  (∃ x y, system x y a b) ↔ (a > 0 ∧ b ≤ a/4) ∨ (a < 0 ∧ b ≥ a/4) :=
sorry

-- Theorem for one root being three times the other
theorem one_root_triple_other (a b : ℝ) :
  (∃ x y, system x y a b ∧ x = 3*y) ↔ b = 3*a/16 :=
sorry

end NUMINAMATH_CALUDE_real_roots_condition_one_root_triple_other_l561_56149


namespace NUMINAMATH_CALUDE_cookie_jar_solution_l561_56166

def cookie_jar_problem (initial_amount : ℝ) : Prop :=
  let doris_spent : ℝ := 6
  let martha_spent : ℝ := doris_spent / 2
  let remaining_after_doris_martha : ℝ := initial_amount - doris_spent - martha_spent
  let john_spent_percentage : ℝ := 0.2
  let john_spent : ℝ := john_spent_percentage * remaining_after_doris_martha
  let final_amount : ℝ := remaining_after_doris_martha - john_spent
  final_amount = 15

theorem cookie_jar_solution :
  ∃ (initial_amount : ℝ), cookie_jar_problem initial_amount ∧ initial_amount = 27.75 := by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_solution_l561_56166


namespace NUMINAMATH_CALUDE_cars_served_4pm_to_6pm_l561_56101

def peak_service_rate : ℕ := 12
def off_peak_service_rate : ℕ := 8
def blocks_per_hour : ℕ := 4

def cars_served_peak_hour : ℕ := peak_service_rate * blocks_per_hour
def cars_served_off_peak_hour : ℕ := off_peak_service_rate * blocks_per_hour

def total_cars_served : ℕ := cars_served_peak_hour + cars_served_off_peak_hour

theorem cars_served_4pm_to_6pm : total_cars_served = 80 := by
  sorry

end NUMINAMATH_CALUDE_cars_served_4pm_to_6pm_l561_56101


namespace NUMINAMATH_CALUDE_find_b_value_l561_56111

theorem find_b_value (a b : ℝ) (h1 : 2 * a + 3 = 5) (h2 : b - a = 2) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l561_56111


namespace NUMINAMATH_CALUDE_first_marvelous_monday_l561_56117

/-- Represents a date in October --/
structure OctoberDate :=
  (day : ℕ)
  (is_monday : Bool)

/-- The number of days in October --/
def october_days : ℕ := 31

/-- The first day of school --/
def school_start : OctoberDate :=
  { day := 2, is_monday := true }

/-- A function to find the next Monday given a current date --/
def next_monday (d : OctoberDate) : OctoberDate :=
  { day := d.day + 7, is_monday := true }

/-- The definition of a Marvelous Monday --/
def is_marvelous_monday (d : OctoberDate) : Prop :=
  d.is_monday ∧ d.day ≤ october_days ∧ 
  (∀ m : OctoberDate, m.is_monday ∧ m.day > d.day → m.day > october_days)

/-- The theorem to prove --/
theorem first_marvelous_monday : 
  ∃ d : OctoberDate, d.day = 30 ∧ is_marvelous_monday d :=
sorry

end NUMINAMATH_CALUDE_first_marvelous_monday_l561_56117


namespace NUMINAMATH_CALUDE_attractions_permutations_l561_56172

theorem attractions_permutations : Nat.factorial 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_attractions_permutations_l561_56172


namespace NUMINAMATH_CALUDE_g_of_5_eq_50_l561_56110

/-- The polynomial g(x) = 3x^4 - 20x^3 + 40x^2 - 50x - 75 -/
def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 40*x^2 - 50*x - 75

/-- Theorem: g(5) = 50 -/
theorem g_of_5_eq_50 : g 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_eq_50_l561_56110


namespace NUMINAMATH_CALUDE_inequality_equivalence_l561_56152

theorem inequality_equivalence (x : ℝ) : 2 * x - 3 < x + 1 ↔ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l561_56152


namespace NUMINAMATH_CALUDE_special_ellipse_equation_l561_56113

/-- An ellipse with center at the origin, one focus at (0,2), intersected by the line y = 3x + 7 
    such that the midpoint of the intersection chord has a y-coordinate of 1 -/
structure SpecialEllipse where
  /-- The equation of the ellipse in the form (x²/a²) + (y²/b²) = 1 -/
  equation : ℝ → ℝ → Prop
  /-- One focus of the ellipse is at (0,2) -/
  focus_at_0_2 : ∃ (x y : ℝ), equation x y ∧ x = 0 ∧ y = 2
  /-- The line y = 3x + 7 intersects the ellipse -/
  intersects_line : ∃ (x y : ℝ), equation x y ∧ y = 3*x + 7
  /-- The midpoint of the intersection chord has a y-coordinate of 1 -/
  midpoint_y_is_1 : 
    ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      equation x₁ y₁ ∧ y₁ = 3*x₁ + 7 ∧
      equation x₂ y₂ ∧ y₂ = 3*x₂ + 7 ∧
      (y₁ + y₂) / 2 = 1

/-- The equation of the special ellipse is x²/8 + y²/12 = 1 -/
theorem special_ellipse_equation (e : SpecialEllipse) : 
  e.equation = fun x y => x^2/8 + y^2/12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_equation_l561_56113


namespace NUMINAMATH_CALUDE_number_comparisons_l561_56168

theorem number_comparisons :
  (-7 / 8 : ℚ) > (-8 / 9 : ℚ) ∧ -|(-5)| < -(-4) := by sorry

end NUMINAMATH_CALUDE_number_comparisons_l561_56168


namespace NUMINAMATH_CALUDE_a_range_l561_56198

/-- Proposition p: For all real x, ax^2 + 2ax + 3 > 0 -/
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + 2 * a * x + 3 > 0

/-- Proposition q: There exists a real x such that x^2 + 2ax + a + 2 = 0 -/
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0

/-- The theorem stating that if both p and q are true, then a is in the range [2, 3) -/
theorem a_range (a : ℝ) (hp : p a) (hq : q a) : a ∈ Set.Ici 2 ∩ Set.Iio 3 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l561_56198


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l561_56143

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_sum : a 4 + a 8 = -2) : 
  a 4^2 + 2 * a 6^2 + a 6 * a 10 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l561_56143


namespace NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l561_56129

/-- Calculates the amount John paid out of pocket for a new computer and accessories,
    given the costs and the sale of his PlayStation. -/
theorem johns_out_of_pocket_expense (computer_cost accessories_cost playstation_value : ℝ)
  (h1 : computer_cost = 700)
  (h2 : accessories_cost = 200)
  (h3 : playstation_value = 400)
  (discount_rate : ℝ)
  (h4 : discount_rate = 0.2) :
  computer_cost + accessories_cost - playstation_value * (1 - discount_rate) = 580 := by
sorry


end NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l561_56129


namespace NUMINAMATH_CALUDE_triangle_height_l561_56138

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 10 → area = 25 → area = (base * height) / 2 → height = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l561_56138


namespace NUMINAMATH_CALUDE_locus_of_tangent_circles_l561_56142

-- Define the circles C₃ and C₄
def C₃ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₄ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

-- Define the property of being externally tangent to C₃
def externally_tangent_C₃ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 2)^2

-- Define the property of being internally tangent to C₄
def internally_tangent_C₄ (a b r : ℝ) : Prop := (a - 3)^2 + b^2 = (5 - r)^2

-- Define the locus equation
def locus_equation (a b : ℝ) : Prop := a^2 + 5*b^2 - 32*a - 51 = 0

-- State the theorem
theorem locus_of_tangent_circles :
  ∀ a b : ℝ,
  (∃ r : ℝ, externally_tangent_C₃ a b r ∧ internally_tangent_C₄ a b r) ↔
  locus_equation a b :=
sorry

end NUMINAMATH_CALUDE_locus_of_tangent_circles_l561_56142


namespace NUMINAMATH_CALUDE_sosnovka_petrovka_distance_l561_56105

/-- The distance between two points on a road --/
def distance (a b : ℕ) : ℕ := max a b - min a b

theorem sosnovka_petrovka_distance :
  ∀ (A B P S : ℕ),
  distance A P = 70 →
  distance A B = 20 →
  distance B S = 130 →
  distance S P = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_sosnovka_petrovka_distance_l561_56105


namespace NUMINAMATH_CALUDE_great_circle_to_surface_area_ratio_l561_56178

theorem great_circle_to_surface_area_ratio (O : Type*) [MetricSpace O] [NormedAddCommGroup O] 
  [InnerProductSpace ℝ O] [FiniteDimensional ℝ O] [ProperSpace O] (S₁ S₂ : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ S₁ = π * r^2 ∧ S₂ = 4 * π * r^2) → 
  S₁ / S₂ = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_great_circle_to_surface_area_ratio_l561_56178


namespace NUMINAMATH_CALUDE_puzzle_pieces_left_l561_56191

theorem puzzle_pieces_left (total_pieces : ℕ) (num_sons : ℕ) (reyn_pieces : ℕ) : 
  total_pieces = 300 →
  num_sons = 3 →
  reyn_pieces = 25 →
  total_pieces - (reyn_pieces + 2 * reyn_pieces + 3 * reyn_pieces) = 150 :=
by
  sorry

#check puzzle_pieces_left

end NUMINAMATH_CALUDE_puzzle_pieces_left_l561_56191


namespace NUMINAMATH_CALUDE_sequence_growth_l561_56173

theorem sequence_growth (a : ℕ → ℕ) 
  (h1 : ∀ n, a n > 1) 
  (h2 : ∀ m n, m ≠ n → a m ≠ a n) : 
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ a n > n :=
sorry

end NUMINAMATH_CALUDE_sequence_growth_l561_56173


namespace NUMINAMATH_CALUDE_base_conversion_1729_l561_56102

def base_10_to_base_6 (n : ℕ) : List ℕ :=
  sorry

theorem base_conversion_1729 :
  base_10_to_base_6 1729 = [1, 2, 0, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_base_conversion_1729_l561_56102


namespace NUMINAMATH_CALUDE_overall_discount_percentage_l561_56159

/-- Calculate the overall discount percentage for three items given their cost prices, markups, and sale prices. -/
theorem overall_discount_percentage
  (cost_A cost_B cost_C : ℝ)
  (markup_A markup_B markup_C : ℝ)
  (sale_A sale_B sale_C : ℝ)
  (h_cost_A : cost_A = 540)
  (h_cost_B : cost_B = 620)
  (h_cost_C : cost_C = 475)
  (h_markup_A : markup_A = 0.15)
  (h_markup_B : markup_B = 0.20)
  (h_markup_C : markup_C = 0.25)
  (h_sale_A : sale_A = 462)
  (h_sale_B : sale_B = 558)
  (h_sale_C : sale_C = 405) :
  let marked_A := cost_A * (1 + markup_A)
  let marked_B := cost_B * (1 + markup_B)
  let marked_C := cost_C * (1 + markup_C)
  let total_marked := marked_A + marked_B + marked_C
  let total_sale := sale_A + sale_B + sale_C
  let discount_percentage := (total_marked - total_sale) / total_marked * 100
  ∃ ε > 0, |discount_percentage - 27.26| < ε :=
by sorry


end NUMINAMATH_CALUDE_overall_discount_percentage_l561_56159


namespace NUMINAMATH_CALUDE_ham_slices_per_sandwich_l561_56163

theorem ham_slices_per_sandwich :
  ∀ (initial_slices : ℕ) (additional_slices : ℕ) (total_sandwiches : ℕ),
    initial_slices = 31 →
    additional_slices = 119 →
    total_sandwiches = 50 →
    (initial_slices + additional_slices) / total_sandwiches = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ham_slices_per_sandwich_l561_56163


namespace NUMINAMATH_CALUDE_dog_feed_mix_problem_l561_56100

/-- The cost per pound of the cheaper kind of feed -/
def cheaper_feed_cost : ℝ := 0.18

theorem dog_feed_mix_problem :
  -- Total weight of the mix
  let total_weight : ℝ := 35
  -- Cost per pound of the final mix
  let final_mix_cost : ℝ := 0.36
  -- Cost per pound of the more expensive feed
  let expensive_feed_cost : ℝ := 0.53
  -- Weight of the cheaper feed used
  let cheaper_feed_weight : ℝ := 17
  -- Weight of the more expensive feed used
  let expensive_feed_weight : ℝ := total_weight - cheaper_feed_weight
  -- Total value of the final mix
  let total_value : ℝ := total_weight * final_mix_cost
  -- Value of the more expensive feed
  let expensive_feed_value : ℝ := expensive_feed_weight * expensive_feed_cost
  -- Equation for the total value
  cheaper_feed_weight * cheaper_feed_cost + expensive_feed_value = total_value →
  cheaper_feed_cost = 0.18 := by
sorry

end NUMINAMATH_CALUDE_dog_feed_mix_problem_l561_56100


namespace NUMINAMATH_CALUDE_subtracted_number_l561_56174

theorem subtracted_number (x : ℝ) (y : ℝ) : 
  x = 62.5 → ((x + 5) * 2 / 5 - y = 22) → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l561_56174


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l561_56146

-- Define the binomial expansion function
def binomial_expansion (a : ℝ) (n : ℕ) : ℝ → ℝ := sorry

-- Define the sum of coefficients function
def sum_of_coefficients (a : ℝ) (n : ℕ) : ℝ := sorry

-- Define the sum of binomial coefficients function
def sum_of_binomial_coefficients (n : ℕ) : ℕ := sorry

-- Define the coefficient of x^2 function
def coefficient_of_x_squared (a : ℝ) (n : ℕ) : ℝ := sorry

theorem binomial_expansion_theorem (a : ℝ) (n : ℕ) :
  sum_of_coefficients a n = -1 ∧
  sum_of_binomial_coefficients n = 32 →
  coefficient_of_x_squared a n = 120 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l561_56146


namespace NUMINAMATH_CALUDE_five_line_regions_l561_56181

/-- Number of regions formed by n lines in a plane -/
def num_regions (n : ℕ) : ℕ := 1 + n + n.choose 2

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  not_parallel : Prop
  not_concurrent : Prop

/-- The number of regions formed by a line configuration -/
def regions_formed (config : LineConfiguration) : ℕ := num_regions config.num_lines

theorem five_line_regions (config : LineConfiguration) :
  config.num_lines = 5 →
  config.not_parallel →
  config.not_concurrent →
  regions_formed config = 16 := by
  sorry

end NUMINAMATH_CALUDE_five_line_regions_l561_56181


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l561_56140

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (3 - 2 * i) / (1 + 4 * i) = (1 : ℂ) / 3 + (14 : ℂ) / 15 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l561_56140


namespace NUMINAMATH_CALUDE_A_P_parity_uniformity_l561_56182

-- Define the set A_P
def A_P : Set ℤ := sorry

-- Define a property for elements of A_P related to positioning in a function or polynomial
def has_positioning_property (n : ℤ) : Prop := sorry

-- Axiom: All elements in A_P have the positioning property
axiom A_P_property : ∀ n ∈ A_P, has_positioning_property n

-- Define parity
def same_parity (a b : ℤ) : Prop := a % 2 = b % 2

-- Theorem: The smallest and largest elements of A_P have the same parity
theorem A_P_parity_uniformity :
  ∀ (min max : ℤ), min ∈ A_P → max ∈ A_P →
  (∀ x ∈ A_P, min ≤ x ∧ x ≤ max) →
  same_parity min max :=
sorry

end NUMINAMATH_CALUDE_A_P_parity_uniformity_l561_56182


namespace NUMINAMATH_CALUDE_right_angled_triangle_l561_56137

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)
  (angle_sum : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Theorem statement
theorem right_angled_triangle (abc : Triangle) 
  (h : Real.sin abc.A = Real.sin abc.C * Real.cos abc.B) : 
  abc.C = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l561_56137


namespace NUMINAMATH_CALUDE_rational_equation_solution_l561_56179

theorem rational_equation_solution (x : ℝ) :
  (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 8*x + 15) / (x^2 - 10*x + 24) →
  x = (13 + Real.sqrt 5) / 2 ∨ x = (13 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l561_56179


namespace NUMINAMATH_CALUDE_greatest_common_divisor_with_same_remainder_l561_56130

theorem greatest_common_divisor_with_same_remainder (a b c : ℕ) (h : a < b ∧ b < c) :
  ∃ (d : ℕ), d > 0 ∧ d = Nat.gcd (b - a) (c - b) ∧
  ∀ (k : ℕ), k > d → ¬(∃ (r : ℕ), a % k = r ∧ b % k = r ∧ c % k = r) := by
  sorry

#check greatest_common_divisor_with_same_remainder 25 57 105

end NUMINAMATH_CALUDE_greatest_common_divisor_with_same_remainder_l561_56130


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_equivalence_l561_56151

/-- Represents a quadratic function in the form ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a quadratic function in vertex form a(x - m)² + n -/
structure VertexForm where
  a : ℝ
  m : ℝ
  n : ℝ

/-- The vertex of a quadratic function -/
structure Vertex where
  x : ℝ
  y : ℝ

/-- Theorem stating the equivalence of the standard and vertex forms of a specific quadratic function,
    and identifying its vertex -/
theorem quadratic_vertex_form_equivalence :
  let f : QuadraticFunction := { a := 2, b := -12, c := -12 }
  let v : VertexForm := { a := 2, m := 3, n := -30 }
  let vertex : Vertex := { x := 3, y := -30 }
  (∀ x, 2 * x^2 - 12 * x - 12 = 2 * (x - 3)^2 - 30) ∧
  (vertex.x = -f.b / (2 * f.a) ∧ vertex.y = f.c - f.b^2 / (4 * f.a)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_equivalence_l561_56151


namespace NUMINAMATH_CALUDE_total_buyers_is_140_l561_56150

/-- The number of buyers who visited a store over three consecutive days -/
def total_buyers (day_before_yesterday yesterday today : ℕ) : ℕ :=
  day_before_yesterday + yesterday + today

/-- Theorem stating the total number of buyers over three days -/
theorem total_buyers_is_140 :
  ∃ (yesterday today : ℕ),
    yesterday = 50 / 2 ∧
    today = yesterday + 40 ∧
    total_buyers 50 yesterday today = 140 :=
by sorry

end NUMINAMATH_CALUDE_total_buyers_is_140_l561_56150


namespace NUMINAMATH_CALUDE_jerry_has_36_stickers_l561_56103

-- Define the number of stickers for each person
def fred_stickers : ℕ := 18
def george_stickers : ℕ := fred_stickers - 6
def jerry_stickers : ℕ := 3 * george_stickers
def carla_stickers : ℕ := jerry_stickers + (jerry_stickers / 4)

-- Theorem to prove
theorem jerry_has_36_stickers : jerry_stickers = 36 := by
  sorry

end NUMINAMATH_CALUDE_jerry_has_36_stickers_l561_56103


namespace NUMINAMATH_CALUDE_cake_and_muffin_buyers_l561_56180

theorem cake_and_muffin_buyers (total : ℕ) (cake : ℕ) (muffin : ℕ) (neither_prob : ℚ) :
  total = 100 →
  cake = 50 →
  muffin = 40 →
  neither_prob = 26 / 100 →
  ∃ (both : ℕ), both = 16 ∧ 
    (total : ℚ) * (1 - neither_prob) = (cake + muffin - both : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_cake_and_muffin_buyers_l561_56180


namespace NUMINAMATH_CALUDE_cookie_and_game_cost_l561_56196

/-- Represents the cost and profit information for an item --/
structure ItemInfo where
  cost : ℚ
  price : ℚ
  profit : ℚ
  makeTime : ℚ

/-- Represents the sales quota for each item --/
structure SalesQuota where
  bracelets : ℕ
  necklaces : ℕ
  rings : ℕ

def bracelet : ItemInfo := ⟨1, 1.5, 0.5, 10/60⟩
def necklace : ItemInfo := ⟨2, 3, 1, 15/60⟩
def ring : ItemInfo := ⟨0.5, 1, 0.5, 5/60⟩

def salesQuota : SalesQuota := ⟨5, 3, 10⟩

def profitMargin : ℚ := 0.5
def workingHoursPerDay : ℚ := 2
def daysInWeek : ℕ := 7
def remainingMoney : ℚ := 5

theorem cookie_and_game_cost (totalSales totalCost : ℚ) :
  totalSales = (bracelet.price * salesQuota.bracelets + 
                necklace.price * salesQuota.necklaces + 
                ring.price * salesQuota.rings) →
  totalCost = (bracelet.cost * salesQuota.bracelets + 
               necklace.cost * salesQuota.necklaces + 
               ring.cost * salesQuota.rings) →
  totalSales = totalCost * (1 + profitMargin) →
  (bracelet.makeTime * salesQuota.bracelets + 
   necklace.makeTime * salesQuota.necklaces + 
   ring.makeTime * salesQuota.rings) ≤ workingHoursPerDay * daysInWeek →
  totalSales - remainingMoney = 24 := by
  sorry

end NUMINAMATH_CALUDE_cookie_and_game_cost_l561_56196


namespace NUMINAMATH_CALUDE_inequality_system_solution_l561_56121

theorem inequality_system_solution (a : ℝ) :
  (∃ x : ℝ, x + a ≥ 0 ∧ 1 - 2*x > x - 2) ↔ a > -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l561_56121


namespace NUMINAMATH_CALUDE_seniority_ranking_l561_56162

-- Define the colleagues
inductive Colleague
| Julia
| Kevin
| Lana

-- Define the seniority relation
def more_senior (a b : Colleague) : Prop := sorry

-- Define the most senior and least senior
def most_senior (c : Colleague) : Prop :=
  ∀ other, c ≠ other → more_senior c other

def least_senior (c : Colleague) : Prop :=
  ∀ other, c ≠ other → more_senior other c

-- Define the statements
def statement_I : Prop := most_senior Colleague.Kevin
def statement_II : Prop := least_senior Colleague.Lana
def statement_III : Prop := ¬(least_senior Colleague.Julia)

-- Main theorem
theorem seniority_ranking :
  (statement_I ∧ ¬statement_II ∧ ¬statement_III) →
  (more_senior Colleague.Kevin Colleague.Lana ∧
   more_senior Colleague.Lana Colleague.Julia) :=
by sorry

end NUMINAMATH_CALUDE_seniority_ranking_l561_56162


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l561_56148

theorem complex_power_magnitude : 
  Complex.abs ((2 : ℂ) + (2 * Complex.I * Real.sqrt 3))^4 = 256 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l561_56148


namespace NUMINAMATH_CALUDE_fast_food_constant_and_variables_l561_56120

/-- A linear pricing model for fast food boxes -/
structure FastFoodPricing where
  cost_per_box : ℝ  -- Cost per box in yuan
  num_boxes : ℝ     -- Number of boxes purchased
  total_cost : ℝ    -- Total cost in yuan
  pricing_model : total_cost = cost_per_box * num_boxes

/-- Theorem stating that in a FastFoodPricing model, the constant is the cost per box,
    and the variables are the number of boxes and the total cost -/
theorem fast_food_constant_and_variables (model : FastFoodPricing) :
  (∃ (k : ℝ), k = model.cost_per_box ∧ k ≠ 0) ∧
  (∀ (n s : ℝ), n = model.num_boxes ∧ s = model.total_cost →
    s = model.cost_per_box * n) :=
sorry

end NUMINAMATH_CALUDE_fast_food_constant_and_variables_l561_56120


namespace NUMINAMATH_CALUDE_pet_store_siamese_cats_l561_56126

/-- The number of Siamese cats initially in the pet store -/
def initial_siamese_cats : ℕ := 13

/-- The number of house cats initially in the pet store -/
def initial_house_cats : ℕ := 5

/-- The total number of cats sold during the sale -/
def cats_sold : ℕ := 10

/-- The number of cats remaining after the sale -/
def cats_remaining : ℕ := 8

/-- Theorem stating that the initial number of Siamese cats is 13 -/
theorem pet_store_siamese_cats : 
  initial_siamese_cats = 13 ∧ 
  initial_siamese_cats + initial_house_cats = cats_sold + cats_remaining :=
by sorry

end NUMINAMATH_CALUDE_pet_store_siamese_cats_l561_56126


namespace NUMINAMATH_CALUDE_negation_of_proposition_l561_56199

theorem negation_of_proposition (x y : ℝ) :
  ¬(((x - 1)^2 + (y - 2)^2 = 0) → (x = 1 ∧ y = 2)) ↔
  (((x - 1)^2 + (y - 2)^2 ≠ 0) → (x ≠ 1 ∨ y ≠ 2)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l561_56199


namespace NUMINAMATH_CALUDE_sues_necklace_beads_l561_56125

/-- The number of beads in Sue's necklace -/
def total_beads (purple : ℕ) (blue : ℕ) (green : ℕ) : ℕ :=
  purple + blue + green

/-- Theorem stating the total number of beads in Sue's necklace -/
theorem sues_necklace_beads : 
  ∀ (purple blue green : ℕ),
    purple = 7 →
    blue = 2 * purple →
    green = blue + 11 →
    total_beads purple blue green = 46 := by
  sorry

end NUMINAMATH_CALUDE_sues_necklace_beads_l561_56125


namespace NUMINAMATH_CALUDE_inequality_solution_set_l561_56171

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  1 / x ≤ x ↔ (-1 ≤ x ∧ x < 0) ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l561_56171


namespace NUMINAMATH_CALUDE_jenna_work_hours_l561_56136

def concert_ticket_cost : ℝ := 181
def drink_ticket_cost : ℝ := 7
def num_drink_tickets : ℕ := 5
def hourly_wage : ℝ := 18
def salary_percentage : ℝ := 0.1
def weeks_per_month : ℕ := 4

theorem jenna_work_hours :
  ∀ (weekly_hours : ℝ),
  (concert_ticket_cost + num_drink_tickets * drink_ticket_cost = 
   salary_percentage * (weekly_hours * hourly_wage * weeks_per_month)) →
  weekly_hours = 30 := by
  sorry

end NUMINAMATH_CALUDE_jenna_work_hours_l561_56136


namespace NUMINAMATH_CALUDE_sum_of_first_five_primes_mod_sixth_prime_l561_56114

def first_five_primes : List Nat := [2, 3, 5, 7, 11]
def sixth_prime : Nat := 13

theorem sum_of_first_five_primes_mod_sixth_prime :
  (first_five_primes.sum % sixth_prime) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_five_primes_mod_sixth_prime_l561_56114


namespace NUMINAMATH_CALUDE_age_digits_product_l561_56132

/-- A function that returns the digits of a two-digit number -/
def digits (n : ℕ) : List ℕ :=
  [n / 10, n % 10]

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

/-- A function that checks if a number is a power of another number -/
def isPowerOf (n : ℕ) (base : ℕ) : Prop :=
  ∃ k : ℕ, n = base ^ k

/-- A function that calculates the sum of a list of numbers -/
def sum (l : List ℕ) : ℕ :=
  l.foldl (·+·) 0

/-- A function that calculates the product of a list of numbers -/
def product (l : List ℕ) : ℕ :=
  l.foldl (·*·) 1

theorem age_digits_product : 
  ∃ (x y : ℕ),
    isTwoDigit x ∧ 
    isTwoDigit y ∧ 
    isPowerOf x 5 ∧ 
    isPowerOf y 2 ∧ 
    Odd (sum (digits x ++ digits y)) → 
    product (digits x ++ digits y) = 240 := by
  sorry

end NUMINAMATH_CALUDE_age_digits_product_l561_56132


namespace NUMINAMATH_CALUDE_inequality_proof_l561_56170

theorem inequality_proof (a b c : ℝ) (k : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k ≥ 2) (habc : a * b * c = 1) :
  (a^k / (a + b)) + (b^k / (b + c)) + (c^k / (c + a)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l561_56170


namespace NUMINAMATH_CALUDE_optimal_price_and_quantity_l561_56156

/-- Represents the pricing and sales model of a shopping mall --/
structure ShoppingMall where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_elasticity : ℝ
  target_profit : ℝ

/-- Calculates the monthly sales volume based on the new price --/
def sales_volume (mall : ShoppingMall) (new_price : ℝ) : ℝ :=
  mall.initial_sales - mall.price_elasticity * (new_price - mall.initial_price)

/-- Calculates the monthly profit based on the new price --/
def monthly_profit (mall : ShoppingMall) (new_price : ℝ) : ℝ :=
  (new_price - mall.cost_price) * sales_volume mall new_price

/-- Theorem stating that the new price and purchase quantity achieve the target profit --/
theorem optimal_price_and_quantity (mall : ShoppingMall) 
  (h_cost : mall.cost_price = 20)
  (h_initial_price : mall.initial_price = 30)
  (h_initial_sales : mall.initial_sales = 800)
  (h_elasticity : mall.price_elasticity = 20)
  (h_target : mall.target_profit = 12000) :
  ∃ (new_price : ℝ), 
    new_price = 40 ∧ 
    sales_volume mall new_price = 600 ∧ 
    monthly_profit mall new_price = mall.target_profit := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_and_quantity_l561_56156


namespace NUMINAMATH_CALUDE_quadruple_solutions_l561_56193

theorem quadruple_solutions : 
  ∀ a b c d : ℝ, 
    (a * b + c * d = 6) ∧ 
    (a * c + b * d = 3) ∧ 
    (a * d + b * c = 2) ∧ 
    (a + b + c + d = 6) → 
    ((a = 0 ∧ b = 1 ∧ c = 2 ∧ d = 3) ∨
     (a = 2 ∧ b = 3 ∧ c = 0 ∧ d = 1) ∨
     (a = 1 ∧ b = 0 ∧ c = 3 ∧ d = 2) ∨
     (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadruple_solutions_l561_56193


namespace NUMINAMATH_CALUDE_methane_combustion_l561_56112

/-- Represents the balanced chemical equation for methane combustion -/
structure MethaneReaction where
  ch4 : ℚ
  o2 : ℚ
  co2 : ℚ
  h2o : ℚ
  balanced : ch4 = 1 ∧ o2 = 2 ∧ co2 = 1 ∧ h2o = 2

/-- Theorem stating the number of moles of CH₄ required and CO₂ formed when 2 moles of O₂ react -/
theorem methane_combustion (reaction : MethaneReaction) (o2_moles : ℚ) 
  (h_o2 : o2_moles = 2) : 
  let ch4_required := o2_moles / reaction.o2 * reaction.ch4
  let co2_formed := ch4_required / reaction.ch4 * reaction.co2
  ch4_required = 1 ∧ co2_formed = 1 := by
  sorry


end NUMINAMATH_CALUDE_methane_combustion_l561_56112


namespace NUMINAMATH_CALUDE_inequality_proof_l561_56108

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) 
  (h4 : a * b + b * c + c * a = 1 / 3) : 
  1 / (a^2 - b*c + 1) + 1 / (b^2 - c*a + 1) + 1 / (c^2 - a*b + 1) ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l561_56108


namespace NUMINAMATH_CALUDE_parallel_line_not_through_point_l561_56160

/-- A line in 2D space represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.A * p.x + l.B * p.y + l.C = 0

theorem parallel_line_not_through_point
    (L : Line)
    (P : Point)
    (h_not_on : ¬ P.onLine L) :
    ∃ (k : ℝ),
      k ≠ 0 ∧
      (∀ (x y : ℝ),
        L.A * x + L.B * y + L.C + (L.A * P.x + L.B * P.y + L.C) = 0 ↔
        L.A * x + L.B * y + L.C + k = 0) ∧
      (L.A * P.x + L.B * P.y + L.C + k ≠ 0) :=
  sorry

end NUMINAMATH_CALUDE_parallel_line_not_through_point_l561_56160


namespace NUMINAMATH_CALUDE_probability_differ_by_2_l561_56127

/-- A standard 6-sided die -/
def Die : Type := Fin 6

/-- The set of all possible outcomes when rolling a die twice -/
def TwoRolls : Type := Die × Die

/-- The condition for two rolls to differ by 2 -/
def DifferBy2 (roll : TwoRolls) : Prop :=
  (roll.1.val + 1 = roll.2.val) ∨ (roll.1.val = roll.2.val + 1)

/-- The number of favorable outcomes -/
def FavorableOutcomes : ℕ := 8

/-- The total number of possible outcomes -/
def TotalOutcomes : ℕ := 36

theorem probability_differ_by_2 :
  (FavorableOutcomes : ℚ) / TotalOutcomes = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_differ_by_2_l561_56127


namespace NUMINAMATH_CALUDE_divisibility_by_24_l561_56122

theorem divisibility_by_24 (p : ℕ) (hp : p.Prime) (hp_ge_5 : p ≥ 5) : 
  24 ∣ p^2 - 1 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l561_56122


namespace NUMINAMATH_CALUDE_monogramming_cost_is_17_69_l561_56184

/-- Calculates the monogramming cost per stocking --/
def monogramming_cost_per_stocking (grandchildren children : ℕ) 
  (stockings_per_grandchild : ℕ) (stocking_price : ℚ) (discount_percent : ℚ) 
  (total_cost : ℚ) : ℚ :=
  let total_stockings := grandchildren * stockings_per_grandchild + children
  let discounted_price := stocking_price * (1 - discount_percent / 100)
  let stockings_cost := total_stockings * discounted_price
  let total_monogramming_cost := total_cost - stockings_cost
  total_monogramming_cost / total_stockings

/-- Theorem stating that the monogramming cost per stocking is $17.69 --/
theorem monogramming_cost_is_17_69 :
  monogramming_cost_per_stocking 5 4 5 20 10 1035 = 1769 / 100 := by
  sorry

end NUMINAMATH_CALUDE_monogramming_cost_is_17_69_l561_56184


namespace NUMINAMATH_CALUDE_intersection_points_sum_l561_56106

theorem intersection_points_sum (m : ℕ) (h : m = 17) : 
  ∃ (x : ℕ), 
    (∀ y : ℕ, (y ≡ 6*x + 3 [MOD m] ↔ y ≡ 13*x + 8 [MOD m])) ∧ 
    x = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_sum_l561_56106


namespace NUMINAMATH_CALUDE_volunteer_selection_theorem_l561_56195

/-- The number of ways to select three volunteers from five for three specific roles -/
def select_volunteers (n : ℕ) (k : ℕ) (excluded : ℕ) : ℕ :=
  (n - 1) * (n - 1) * (n - 2)

/-- The theorem stating that selecting three volunteers from five for three specific roles,
    where one volunteer cannot serve in a particular role, results in 48 different ways -/
theorem volunteer_selection_theorem :
  select_volunteers 5 3 1 = 48 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_selection_theorem_l561_56195


namespace NUMINAMATH_CALUDE_hash_three_times_100_l561_56157

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.5 * N + N

-- Theorem statement
theorem hash_three_times_100 : hash (hash (hash 100)) = 337.5 := by
  sorry

end NUMINAMATH_CALUDE_hash_three_times_100_l561_56157


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l561_56189

theorem sqrt_sum_inequality (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  (Real.sqrt a + Real.sqrt b)^8 ≥ 64 * a * b * (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l561_56189


namespace NUMINAMATH_CALUDE_clerical_staff_reduction_l561_56124

theorem clerical_staff_reduction (total_employees : ℕ) 
  (initial_clerical_ratio : ℚ) (clerical_reduction_ratio : ℚ) : 
  total_employees = 3600 →
  initial_clerical_ratio = 1/3 →
  clerical_reduction_ratio = 1/2 →
  let initial_clerical := (initial_clerical_ratio * total_employees : ℚ).num
  let remaining_clerical := (1 - clerical_reduction_ratio) * initial_clerical
  let new_total := total_employees - (clerical_reduction_ratio * initial_clerical : ℚ).num
  (remaining_clerical / new_total : ℚ) = 1/5 := by
sorry

end NUMINAMATH_CALUDE_clerical_staff_reduction_l561_56124


namespace NUMINAMATH_CALUDE_max_product_constraint_l561_56177

theorem max_product_constraint (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  (x * y) + z^2 = (x + z) * (y + z) →
  x + y + z = 3 →
  x * y * z ≤ 1 := by sorry

end NUMINAMATH_CALUDE_max_product_constraint_l561_56177


namespace NUMINAMATH_CALUDE_smallest_even_three_digit_multiple_of_17_l561_56175

theorem smallest_even_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 
    n % 17 = 0 ∧ 
    n % 2 = 0 ∧ 
    100 ≤ n ∧ n ≤ 999 → 
    n ≥ 136 :=
by sorry

end NUMINAMATH_CALUDE_smallest_even_three_digit_multiple_of_17_l561_56175


namespace NUMINAMATH_CALUDE_initial_investment_solution_exists_l561_56185

/-- Simple interest calculation function -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating the initial investment given the conditions -/
theorem initial_investment (P : ℝ) (r : ℝ) :
  (simpleInterest P r 2 = 480) →
  (simpleInterest P r 7 = 680) →
  P = 400 := by
  sorry

/-- Proof of the existence of a solution -/
theorem solution_exists : ∃ (P r : ℝ),
  (simpleInterest P r 2 = 480) ∧
  (simpleInterest P r 7 = 680) ∧
  P = 400 := by
  sorry

end NUMINAMATH_CALUDE_initial_investment_solution_exists_l561_56185


namespace NUMINAMATH_CALUDE_white_longer_than_blue_l561_56153

/-- The length of the white line in inches -/
def white_line_length : ℝ := 7.666666666666667

/-- The length of the blue line in inches -/
def blue_line_length : ℝ := 3.3333333333333335

/-- The difference in length between the white and blue lines -/
def length_difference : ℝ := white_line_length - blue_line_length

theorem white_longer_than_blue :
  length_difference = 4.333333333333333 := by sorry

end NUMINAMATH_CALUDE_white_longer_than_blue_l561_56153


namespace NUMINAMATH_CALUDE_absolute_value_reciprocal_2023_l561_56123

theorem absolute_value_reciprocal_2023 :
  {x : ℝ | |x| = (1 : ℝ) / 2023} = {-(1 : ℝ) / 2023, (1 : ℝ) / 2023} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_reciprocal_2023_l561_56123


namespace NUMINAMATH_CALUDE_dart_board_probability_l561_56194

theorem dart_board_probability (r : ℝ) (h : r = 10) :
  let circle_area := π * r^2
  let square_side := r * Real.sqrt 2
  let square_area := square_side^2
  square_area / circle_area = 2 / π := by sorry

end NUMINAMATH_CALUDE_dart_board_probability_l561_56194


namespace NUMINAMATH_CALUDE_derivative_cos_minus_cube_l561_56190

/-- The derivative of f(x) = cos x - x^3 is f'(x) = -sin x - 3x^2 -/
theorem derivative_cos_minus_cube (x : ℝ) : 
  deriv (λ x : ℝ => Real.cos x - x^3) x = -Real.sin x - 3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_cos_minus_cube_l561_56190


namespace NUMINAMATH_CALUDE_john_beats_per_minute_l561_56155

/-- Calculates the number of beats per minute John can play given his playing schedule and total beats played. -/
def beats_per_minute (hours_per_day : ℕ) (days : ℕ) (total_beats : ℕ) : ℕ :=
  total_beats / (hours_per_day * days * 60)

/-- Theorem stating that John can play 200 beats per minute given the problem conditions. -/
theorem john_beats_per_minute :
  beats_per_minute 2 3 72000 = 200 := by
  sorry

end NUMINAMATH_CALUDE_john_beats_per_minute_l561_56155


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l561_56197

/-- A moving circle in a plane passing through (-2, 0) and tangent to x = 2 -/
structure MovingCircle where
  center : ℝ × ℝ
  passes_through_A : center.1 ^ 2 + center.2 ^ 2 = (-2 - center.1) ^ 2 + center.2 ^ 2
  tangent_to_line : (2 - center.1) ^ 2 + center.2 ^ 2 = (2 - (-2)) ^ 2

/-- The trajectory of the center of the moving circle -/
def trajectory_equation (x y : ℝ) : Prop :=
  y ^ 2 = -8 * x

theorem moving_circle_trajectory :
  ∀ (c : MovingCircle), trajectory_equation c.center.1 c.center.2 :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l561_56197


namespace NUMINAMATH_CALUDE_employee_pay_problem_l561_56144

theorem employee_pay_problem (total_pay : ℝ) (a_percent : ℝ) (b_pay : ℝ) :
  total_pay = 550 →
  a_percent = 1.2 →
  total_pay = b_pay + a_percent * b_pay →
  b_pay = 250 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_problem_l561_56144


namespace NUMINAMATH_CALUDE_point_transformation_l561_56139

def rotate_180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

def reflect_y_eq_neg_x (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

theorem point_transformation (a b : ℝ) :
  let Q := (a, b)
  let rotated := rotate_180 a b 2 3
  let final := reflect_y_eq_neg_x rotated.1 rotated.2
  final = (5, -1) → b - a = 6 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l561_56139


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l561_56169

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧ 
  b / Real.sin B = c / Real.sin C ∧
  Real.sin B ^ 2 + Real.sin C ^ 2 = Real.sin A ^ 2 - Real.sqrt 3 * Real.sin B * Real.sin C →
  Real.cos A = -Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l561_56169


namespace NUMINAMATH_CALUDE_last_date_with_sum_property_l561_56145

/-- Represents a date in DD.MM.YYYY format -/
structure Date where
  day : Nat
  month : Nat
  year : Nat
  deriving Repr

/-- Checks if a given date is valid in the year 2008 -/
def isValidDate (d : Date) : Prop :=
  d.year = 2008 ∧
  d.month ≥ 1 ∧ d.month ≤ 12 ∧
  d.day ≥ 1 ∧ d.day ≤ 31 ∧
  (d.month ∈ [4, 6, 9, 11] → d.day ≤ 30) ∧
  (d.month = 2 → d.day ≤ 29)

/-- Extracts individual digits from a number -/
def digits (n : Nat) : List Nat :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

/-- Calculates the sum of the first four digits of a date -/
def sumFirstFour (d : Date) : Nat :=
  List.sum (List.take 4 (digits d.day ++ digits d.month))

/-- Calculates the sum of the last four digits of a date -/
def sumLastFour (d : Date) : Nat :=
  List.sum (List.take 4 (digits d.year).reverse)

/-- Checks if the sum of the first four digits equals the sum of the last four digits -/
def hasSumProperty (d : Date) : Prop :=
  sumFirstFour d = sumLastFour d

/-- States that December 25, 2008 is the last date in 2008 with the sum property -/
theorem last_date_with_sum_property :
  ∀ d : Date, isValidDate d → hasSumProperty d →
  d.year = 2008 → d.month ≤ 12 → d.day ≤ 25 :=
sorry

end NUMINAMATH_CALUDE_last_date_with_sum_property_l561_56145


namespace NUMINAMATH_CALUDE_expected_white_balls_after_transfer_l561_56176

/-- Represents a bag of colored balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- Represents the process of transferring balls between bags -/
def transfer (a b : Bag) : ℝ → Bag × Bag
  | p => sorry

/-- Calculates the expected number of white balls in the first bag after transfers -/
noncomputable def expected_white_balls (a b : Bag) : ℝ :=
  sorry

theorem expected_white_balls_after_transfer :
  let a : Bag := { red := 2, white := 3 }
  let b : Bag := { red := 3, white := 3 }
  expected_white_balls a b = 102 / 35 := by sorry

end NUMINAMATH_CALUDE_expected_white_balls_after_transfer_l561_56176


namespace NUMINAMATH_CALUDE_candy_count_third_set_l561_56104

/-- Represents a set of candies with hard candies, chocolates, and gummy candies -/
structure CandySet where
  hard : ℕ
  chocolate : ℕ
  gummy : ℕ

/-- The total number of candies in a set -/
def CandySet.total (s : CandySet) : ℕ := s.hard + s.chocolate + s.gummy

theorem candy_count_third_set (set1 set2 set3 : CandySet) : 
  /- Total number of each type is equal across all sets -/
  (set1.hard + set2.hard + set3.hard = set1.chocolate + set2.chocolate + set3.chocolate) ∧
  (set1.hard + set2.hard + set3.hard = set1.gummy + set2.gummy + set3.gummy) ∧
  /- First set conditions -/
  (set1.chocolate = set1.gummy) ∧
  (set1.hard = set1.chocolate + 7) ∧
  /- Second set conditions -/
  (set2.hard = set2.chocolate) ∧
  (set2.gummy = set2.hard - 15) ∧
  /- Third set condition -/
  (set3.hard = 0) →
  /- Conclusion: total number of candies in the third set is 29 -/
  set3.total = 29 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_third_set_l561_56104


namespace NUMINAMATH_CALUDE_marble_probability_l561_56183

theorem marble_probability (total_marbles : ℕ) 
  (prob_both_black : ℚ) (prob_both_white : ℚ) :
  total_marbles = 30 →
  prob_both_black = 4/9 →
  prob_both_white = 4/25 :=
by sorry

end NUMINAMATH_CALUDE_marble_probability_l561_56183


namespace NUMINAMATH_CALUDE_sum_equals_1998_l561_56187

theorem sum_equals_1998 (a b c d : ℕ) (h : a * c + b * d + a * d + b * c = 1997) :
  a + b + c + d = 1998 := by sorry

end NUMINAMATH_CALUDE_sum_equals_1998_l561_56187


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l561_56107

/-- Represents a stratified sample from a population -/
structure StratifiedSample where
  total_population : ℕ
  boys_population : ℕ
  girls_population : ℕ
  sample_size : ℕ

/-- Calculates the number of boys in the sample -/
def boys_in_sample (s : StratifiedSample) : ℕ :=
  (s.sample_size * s.boys_population) / s.total_population

/-- Calculates the number of girls in the sample -/
def girls_in_sample (s : StratifiedSample) : ℕ :=
  (s.sample_size * s.girls_population) / s.total_population

/-- Theorem stating the correct number of boys and girls in the sample -/
theorem stratified_sample_theorem (s : StratifiedSample) 
  (h1 : s.total_population = 700)
  (h2 : s.boys_population = 385)
  (h3 : s.girls_population = 315)
  (h4 : s.sample_size = 60) :
  boys_in_sample s = 33 ∧ girls_in_sample s = 27 := by
  sorry

#eval boys_in_sample { total_population := 700, boys_population := 385, girls_population := 315, sample_size := 60 }
#eval girls_in_sample { total_population := 700, boys_population := 385, girls_population := 315, sample_size := 60 }

end NUMINAMATH_CALUDE_stratified_sample_theorem_l561_56107


namespace NUMINAMATH_CALUDE_tim_zoo_cost_l561_56167

/-- The total cost of animals Tim bought for his zoo -/
def total_cost (goat_price : ℝ) (goat_count : ℕ) (llama_price_ratio : ℝ) : ℝ :=
  let llama_count := 2 * goat_count
  let llama_price := goat_price * llama_price_ratio
  goat_price * goat_count + llama_price * llama_count

/-- Theorem stating the total cost of animals for Tim's zoo -/
theorem tim_zoo_cost : total_cost 400 3 1.5 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_tim_zoo_cost_l561_56167


namespace NUMINAMATH_CALUDE_binary_sum_theorem_l561_56158

def binary_to_decimal (b : List Bool) : Nat :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : Nat) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    aux n |>.reverse

def a : List Bool := [true, true, false]
def b : List Bool := [true, false, true]
def c : List Bool := [true, false, true, true]
def d : List Bool := [true, false, false, true, true]

theorem binary_sum_theorem :
  decimal_to_binary (binary_to_decimal a + binary_to_decimal b + 
                     binary_to_decimal c + binary_to_decimal d) =
  [true, false, true, false, false, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_theorem_l561_56158


namespace NUMINAMATH_CALUDE_quadratic_properties_l561_56165

/-- The quadratic function f(x) = ax² + 4x + 2 passing through (3, -4) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * x + 2

/-- The value of a for which f(x) passes through (3, -4) -/
def a : ℝ := -2

/-- The x-coordinate of the axis of symmetry -/
def axis_of_symmetry : ℝ := 1

theorem quadratic_properties :
  (f a 3 = -4) ∧
  (∀ x : ℝ, f a x = f a (2 * axis_of_symmetry - x)) ∧
  (∀ x : ℝ, x ≥ axis_of_symmetry → ∀ y : ℝ, y > x → f a y < f a x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l561_56165


namespace NUMINAMATH_CALUDE_ahmed_hassan_tree_difference_l561_56141

/-- Represents the number of trees in an orchard -/
structure Orchard :=
  (apple : ℕ)
  (orange : ℕ)

/-- Calculate the total number of trees in an orchard -/
def totalTrees (o : Orchard) : ℕ := o.apple + o.orange

/-- The difference in the number of trees between two orchards -/
def treeDifference (o1 o2 : Orchard) : ℕ := (totalTrees o1) - (totalTrees o2)

theorem ahmed_hassan_tree_difference :
  let ahmed : Orchard := { apple := 4, orange := 8 }
  let hassan : Orchard := { apple := 1, orange := 2 }
  treeDifference ahmed hassan = 9 := by
  sorry

end NUMINAMATH_CALUDE_ahmed_hassan_tree_difference_l561_56141


namespace NUMINAMATH_CALUDE_donald_oranges_l561_56133

/-- Given that Donald has 4 oranges initially and finds 5 more,
    prove that he has 9 oranges in total. -/
theorem donald_oranges (initial : Nat) (found : Nat) (total : Nat) 
    (h1 : initial = 4) 
    (h2 : found = 5) 
    (h3 : total = initial + found) : 
  total = 9 := by
  sorry

end NUMINAMATH_CALUDE_donald_oranges_l561_56133


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l561_56135

theorem quadratic_function_minimum (a : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 4 → a * (x - 1)^2 - a ≥ -4) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 4 ∧ a * (x - 1)^2 - a = -4) →
  a = 4 ∨ a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l561_56135


namespace NUMINAMATH_CALUDE_eight_b_value_l561_56116

theorem eight_b_value (a b : ℝ) (h1 : 6 * a + 3 * b = 3) (h2 : a = 2 * b + 3) : 8 * b = -8 := by
  sorry

end NUMINAMATH_CALUDE_eight_b_value_l561_56116


namespace NUMINAMATH_CALUDE_part_one_part_two_l561_56186

-- Part 1
theorem part_one (a b : ℤ) (h1 : a = 4) (h2 : b = 5) : a - b = -1 := by
  sorry

-- Part 2
theorem part_two (a b m n s : ℝ) 
  (h1 : a + b = 0) 
  (h2 : m * n = 1) 
  (h3 : |s| = 3) : 
  a + b + m * n + s = 4 ∨ a + b + m * n + s = -2 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l561_56186


namespace NUMINAMATH_CALUDE_square_side_difference_l561_56192

theorem square_side_difference (a b : ℝ) 
  (h1 : a + b = 20) 
  (h2 : a^2 - b^2 = 40) : 
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_square_side_difference_l561_56192


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_ratio_l561_56188

/-- Theorem: For a hyperbola with equation x^2/a^2 - y^2/b^2 = 1, where a > 0, b > 0,
    and eccentricity = 2, the ratio b/a equals √3. -/
theorem hyperbola_eccentricity_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c / a = 2) →
  b / a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_ratio_l561_56188


namespace NUMINAMATH_CALUDE_common_root_equations_unique_integer_solution_l561_56128

theorem common_root_equations (x p : ℤ) : 
  (3 * x^2 - 4 * x + p - 2 = 0 ∧ x^2 - 2 * p * x + 5 = 0) ↔ (p = 3 ∧ x = 1) :=
by sorry

theorem unique_integer_solution : 
  ∃! p : ℤ, ∃ x : ℤ, 3 * x^2 - 4 * x + p - 2 = 0 ∧ x^2 - 2 * p * x + 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_root_equations_unique_integer_solution_l561_56128


namespace NUMINAMATH_CALUDE_similar_triangles_height_l561_56154

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 9 →
  let scale_factor := Real.sqrt area_ratio
  let h_large := h_small * scale_factor
  h_small = 5 →
  h_large = 15 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l561_56154


namespace NUMINAMATH_CALUDE_xy_and_sum_of_squares_l561_56118

theorem xy_and_sum_of_squares (x y : ℝ) 
  (sum_eq : x + y = 3) 
  (prod_eq : (x + 2) * (y + 2) = 12) : 
  (xy = 2) ∧ (x^2 + 3*x*y + y^2 = 11) := by
  sorry


end NUMINAMATH_CALUDE_xy_and_sum_of_squares_l561_56118


namespace NUMINAMATH_CALUDE_number_of_envelopes_l561_56161

-- Define the weight of a single envelope in grams
def envelope_weight : ℝ := 8.5

-- Define the total weight in kilograms
def total_weight_kg : ℝ := 6.8

-- Define the conversion factor from kg to g
def kg_to_g : ℝ := 1000

-- Theorem to prove
theorem number_of_envelopes : 
  (total_weight_kg * kg_to_g) / envelope_weight = 800 := by
  sorry

end NUMINAMATH_CALUDE_number_of_envelopes_l561_56161


namespace NUMINAMATH_CALUDE_P_root_nature_l561_56109

/-- The polynomial P(x) = x^6 - 4x^5 - 9x^3 + 2x + 9 -/
def P (x : ℝ) : ℝ := x^6 - 4*x^5 - 9*x^3 + 2*x + 9

theorem P_root_nature :
  (∀ x < 0, P x > 0) ∧ (∃ x > 0, P x = 0) := by sorry


end NUMINAMATH_CALUDE_P_root_nature_l561_56109


namespace NUMINAMATH_CALUDE_problem_solution_l561_56147

theorem problem_solution (x : ℤ) (h : x = 40) : x * 6 - 138 = 102 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l561_56147


namespace NUMINAMATH_CALUDE_f_equals_g_l561_56119

-- Define the functions
def f (x : ℝ) : ℝ := (76 * x^6)^7
def g (x : ℝ) : ℝ := |x|

-- State the theorem
theorem f_equals_g : ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_f_equals_g_l561_56119


namespace NUMINAMATH_CALUDE_age_difference_l561_56115

theorem age_difference (son_age man_age : ℕ) : 
  son_age = 28 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 30 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l561_56115
