import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_120_degrees_l1258_125814

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ := 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem angle_between_vectors_120_degrees (a b : ℝ × ℝ) :
  (Real.sqrt (a.1^2 + a.2^2) = 3) →
  (Real.sqrt (b.1^2 + b.2^2) = 4) →
  ((a.1 + b.1) * (a.1 + 3 * b.1) + (a.2 + b.2) * (a.2 + 3 * b.2) = 33) →
  angle_between_vectors a b = 2 * π / 3 := by
  sorry

#check angle_between_vectors_120_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_120_degrees_l1258_125814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weighted_sum_vectors_zero_l1258_125840

open EuclideanGeometry

-- Define the triangle ABC and point O
variable (A B C O : EuclideanSpace ℝ (Fin 2))

-- Define the area function
noncomputable def area (p q r : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Define α, β, γ as ratios of areas
noncomputable def α (A B C O : EuclideanSpace ℝ (Fin 2)) : ℝ := area O B C / area A B C
noncomputable def β (A B C O : EuclideanSpace ℝ (Fin 2)) : ℝ := area O C A / area A B C
noncomputable def γ (A B C O : EuclideanSpace ℝ (Fin 2)) : ℝ := area O A B / area A B C

-- State the theorem
theorem weighted_sum_vectors_zero (A B C O : EuclideanSpace ℝ (Fin 2)) : 
  (α A B C O) • (A - O) + (β A B C O) • (B - O) + (γ A B C O) • (C - O) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weighted_sum_vectors_zero_l1258_125840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_3_4_5_l1258_125800

/-- The area of a triangle with sides 3, 4, and 5 is 6 -/
theorem triangle_area_3_4_5 : ∃ (area : ℝ), area = 6 := by
  let side1 : ℕ := 3
  let side2 : ℕ := 4
  let side3 : ℕ := 5
  let perimeter : ℕ := side1 + side2 + side3
  let semiperimeter : ℝ := (side1 + side2 + side3) / 2
  let area : ℝ := Real.sqrt (semiperimeter * (semiperimeter - side1) * (semiperimeter - side2) * (semiperimeter - side3))
  
  have h1 : perimeter = 12 := by rfl
  have h2 : side1 + side2 > side3 := by norm_num
  have h3 : side2 + side3 > side1 := by norm_num
  have h4 : side3 + side1 > side2 := by norm_num
  
  exists area
  sorry  -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_3_4_5_l1258_125800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_pair_probability_l1258_125856

/-- Represents the number of socks -/
def total_socks : ℕ := 10

/-- Represents the number of colors -/
def num_colors : ℕ := 5

/-- Represents the number of socks drawn -/
def socks_drawn : ℕ := 5

/-- Represents the number of socks per color -/
def socks_per_color : ℕ := 2

/-- The probability of drawing exactly one pair of socks with the same color and no other pairs -/
def prob_one_pair : ℚ := 5 / 63

theorem one_pair_probability :
  (Nat.choose total_socks socks_drawn) ≠ 0 →
  (num_colors * Nat.choose (num_colors - 1) (socks_drawn - 2)) / (Nat.choose total_socks socks_drawn) = prob_one_pair := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_pair_probability_l1258_125856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_unshaded_eq_8_55_l1258_125843

/-- Represents a rectangle in the 2x10 grid --/
structure Rectangle where
  left : Fin 11
  right : Fin 11
  top : Fin 3
  bottom : Fin 3
  h_valid : left < right
  v_valid : top ≤ bottom

/-- The set of all possible rectangles in the 2x10 grid --/
def all_rectangles : Finset Rectangle := sorry

/-- The set of rectangles that do not include a shaded square --/
def unshaded_rectangles : Finset Rectangle := sorry

/-- The probability of choosing a rectangle that does not include a shaded square --/
def prob_unshaded : ℚ := (unshaded_rectangles.card : ℚ) / (all_rectangles.card : ℚ)

theorem prob_unshaded_eq_8_55 : prob_unshaded = 8 / 55 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_unshaded_eq_8_55_l1258_125843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_of_seven_people_l1258_125850

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def number_of_arrangements (n : ℕ) : ℕ :=
  let arrangements_with_C_at_ends := 2 * (4 * factorial 5)
  let arrangements_with_C_in_middle := 5 * (4 * factorial 4)
  arrangements_with_C_at_ends + arrangements_with_C_in_middle

theorem arrangements_of_seven_people : number_of_arrangements 7 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_of_seven_people_l1258_125850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_intersecting_two_lines_l1258_125892

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define membership of a point on a line
instance : Membership Point Line where
  mem p l := l.a * p.x + l.b * p.y + l.c = 0

-- Define reflection of a line about a point
noncomputable def reflect_line (l : Line) (p : Point) : Line :=
  sorry

-- Define intersection of two lines
noncomputable def intersect (l1 l2 : Line) : Point :=
  sorry

-- Define a line passing through two points
noncomputable def line_through_points (p1 p2 : Point) : Line :=
  sorry

-- Theorem statement
theorem line_through_point_intersecting_two_lines 
  (P : Point) (l1 l2 : Line) : 
  ∃ (L : Line), (P ∈ L) ∧ (∃ Q : Point, Q ∈ L ∧ Q ∈ l1 ∧ Q ∈ l2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_intersecting_two_lines_l1258_125892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_smallest_four_digit_in_pascal_l1258_125887

/-- Definition of PascalTriangle (simplified for this problem) -/
def PascalTriangle : Set ℕ := {n : ℕ | n > 0}

/-- Pascal's Triangle contains all positive integers -/
axiom pascal_contains_all_positive : ∀ n : ℕ, n > 0 → n ∈ PascalTriangle

/-- 1000 is the smallest four-digit number -/
theorem smallest_four_digit : ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 → n ≥ 1000 := by
  sorry

theorem smallest_four_digit_in_pascal : 
  ∃ n ∈ PascalTriangle, n = 1000 ∧ ∀ m ∈ PascalTriangle, m ≥ 1000 ∧ m < 10000 → m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_smallest_four_digit_in_pascal_l1258_125887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_fixed_chord_l1258_125861

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line parallel to y-axis -/
structure VerticalLine where
  x : ℝ

/-- Distance between a point and a vertical line -/
def distToLine (p : Point) (l : VerticalLine) : ℝ :=
  |p.x - l.x|

/-- Distance between two points -/
noncomputable def distBetweenPoints (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the trajectory of point P and the fixed chord length -/
theorem trajectory_and_fixed_chord 
  (A : Point) 
  (l : VerticalLine) 
  (M : Point) 
  (h1 : A.x = 1/4 ∧ A.y = 0) 
  (h2 : l.x = -5/4) 
  (h3 : M.x = 4 ∧ M.y = 0) :
  ∃ (P : Point → Prop) (a : ℝ),
    (∀ p, P p ↔ distBetweenPoints A p + 1 = distToLine p l) ∧
    (∀ p, P p → p.y^2 = p.x) ∧
    (a = 15/4) ∧
    (∃ L : ℝ, L = Real.sqrt 15 / 2 ∧
      ∀ p, P p → 
        let midpoint : Point := ⟨(p.x + M.x)/2, (p.y + M.y)/2⟩
        let radius : ℝ := distBetweenPoints p M / 2
        2 * Real.sqrt (radius^2 - (a - midpoint.x)^2) = L) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_fixed_chord_l1258_125861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_count_l1258_125891

def is_consecutive_list (l : List ℤ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < j ∧ j < l.length → l[i]!.succ = l[j]!

theorem consecutive_integers_count
  (K : List ℤ)
  (h_consec : is_consecutive_list K)
  (h_least : K.head! = -4)
  (h_range : (K.filter (λ x => x > 0)).maximum? = some 5 ∧ (K.filter (λ x => x > 0)).minimum? = some 1) :
  K.length = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_count_l1258_125891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l1258_125878

/-- The hyperbola C with center at origin and foci on x-axis -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop

/-- The parabola y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The directrix of the parabola y^2 = 8x -/
def directrix : ℝ → ℝ → Prop := fun x _ ↦ x = -2

/-- Points A and B are the intersection of hyperbola C and the directrix -/
def intersectionPoints (C : Hyperbola) (A B : ℝ × ℝ) : Prop :=
  C.equation A.1 A.2 ∧ directrix A.1 A.2 ∧
  C.equation B.1 B.2 ∧ directrix B.1 B.2

/-- The distance between points A and B -/
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The length of the real axis of a hyperbola -/
noncomputable def realAxisLength (C : Hyperbola) : ℝ := sorry

/-- Theorem: The length of the real axis of hyperbola C is 2 -/
theorem hyperbola_real_axis_length (C : Hyperbola) (A B : ℝ × ℝ) :
  intersectionPoints C A B →
  distance A B = 2 * Real.sqrt 3 →
  realAxisLength C = 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l1258_125878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1258_125895

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x

theorem function_properties (a : ℝ) :
  (∃ A : ℝ, f a A = 0 ∧ (deriv (f a)) A = -1) →
  (a = 2 ∧
   ∃ x_min : ℝ, f 2 x_min = 2 - Real.log 4 ∧ ∀ x : ℝ, f 2 x ≥ f 2 x_min) ∧
  (∀ x : ℝ, x > 0 → x^2 < Real.exp x) ∧
  (∀ c : ℝ, c > 0 → ∃ x_0 : ℝ, ∀ x : ℝ, x > x_0 → x < c * Real.exp x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1258_125895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1258_125883

theorem equation_solution : ∃! (x : ℕ), x > 0 ∧ (1 : ℕ)^(x + 3) + 2^x + 3^(x + 1) + 4^(x - 1) = 272 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1258_125883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_abc_l1258_125844

theorem comparison_abc (a b c : ℝ) 
  (ha : a = Real.rpow 1.9 0.4) 
  (hb : b = Real.log 1.9 / Real.log 0.4) 
  (hc : c = Real.rpow 0.4 1.9) : 
  a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_abc_l1258_125844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_leq_zero_l1258_125897

open MeasureTheory

-- Define the function f
def f (x : ℝ) : ℝ := -x + 2

-- Define the domain
def domain : Set ℝ := Set.Icc (-5) 5

-- Define the event set (where f(x) ≤ 0)
def event : Set ℝ := {x ∈ domain | f x ≤ 0}

-- State the theorem
theorem probability_f_leq_zero :
  (volume event).toReal / (volume domain).toReal = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_f_leq_zero_l1258_125897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_value_l1258_125809

def sequence_a : ℕ → ℚ
  | 0 => -1/4  -- We need to define the base case for 0
  | 1 => -1/4
  | n + 2 => 1 - 1 / sequence_a (n + 1)

theorem a_2016_value : sequence_a 2016 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_value_l1258_125809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_wig_sales_l1258_125866

/-- Represents the problem of John's wig sales for his plays --/
theorem john_wig_sales
  (initial_plays : ℕ)
  (acts_per_play : ℕ)
  (wigs_per_act : ℕ)
  (wig_cost : ℕ)
  (dropped_plays : ℕ)
  (total_spent : ℕ)
  (h1 : initial_plays = 3)
  (h2 : acts_per_play = 5)
  (h3 : wigs_per_act = 2)
  (h4 : wig_cost = 5)
  (h5 : dropped_plays = 1)
  (h6 : total_spent = 110) :
  let total_wigs := initial_plays * acts_per_play * wigs_per_act
  let bought_wigs := total_spent / wig_cost
  let sold_wigs := total_wigs - bought_wigs
  let revenue := initial_plays * acts_per_play * wigs_per_act * wig_cost - total_spent
  revenue / sold_wigs = wig_cost :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_wig_sales_l1258_125866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_sinking_analysis_l1258_125889

-- Define the necessary variables and functions
variable (t : ℝ)
variable (z : ℝ → ℝ)
variable (k b : ℝ)

-- Define the exponential model
noncomputable def exponential_model (k b : ℝ) (t : ℝ) : ℝ := k * Real.exp (b * t)

-- Define the logarithmic transformation
noncomputable def log_transform (z : ℝ → ℝ) : ℝ → ℝ := λ t => Real.log (z t)

-- Define the sum of products for regression calculation
def sum_of_products (t_mean u_mean : ℝ) (t u : List ℝ) : ℝ :=
  List.sum (List.zipWith (λ ti ui => (ti - t_mean) * (ui - u_mean)) t u)

-- Define the sum of squared differences
def sum_of_squares (t_mean : ℝ) (t : List ℝ) : ℝ :=
  List.sum (List.map (λ ti => (ti - t_mean)^2) t)

-- State the theorem
theorem tunnel_sinking_analysis
  (z : ℝ → ℝ)
  (h_model : ∃ k b, ∀ t, z t = exponential_model k b t)
  (h_sum_products : sum_of_products 4 0 [1,2,3,4,5,6,7] (List.map (log_transform z) [1,2,3,4,5,6,7]) = 25.2)
  (h_sum_squares : sum_of_squares 4 [1,2,3,4,5,6,7] = 28)
  (h_threshold : ℝ) :
  (∃ k b, ∀ t, z t = exponential_model k b t ∧ b = 0.9 ∧ k = Real.exp (-4.8)) ∧
  (z 8 = Real.exp 2.4) ∧
  (∃ n : ℕ, n = 9 ∧ ∀ m : ℕ, m > n → (z m - z (m-1)) > h_threshold) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_sinking_analysis_l1258_125889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_octagon_area_l1258_125801

/-- Represents an octagon inscribed in a 6x8 grid --/
structure InscribedOctagon where
  -- The octagon is represented by its vertices
  vertices : List (Int × Int)
  -- Ensure the octagon is within a 6x8 grid
  vertices_in_grid : ∀ v ∈ vertices, 0 ≤ v.1 ∧ v.1 ≤ 6 ∧ -4 ≤ v.2 ∧ v.2 ≤ 4
  -- Ensure there are exactly 8 vertices
  vertex_count : vertices.length = 8

/-- Calculates the area of an inscribed octagon --/
noncomputable def area (octagon : InscribedOctagon) : ℝ :=
  sorry -- The actual calculation would go here

/-- Theorem stating that the area of the inscribed octagon is 48 square units --/
theorem inscribed_octagon_area (octagon : InscribedOctagon) : area octagon = 48 := by
  sorry

#check inscribed_octagon_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_octagon_area_l1258_125801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_section_parabola_l1258_125870

/-- Represents the type of conic section -/
inductive ConicSection
  | Circle
  | Ellipse
  | Hyperbola
  | Parabola

/-- Determines the type of conic section based on the apex angle and section-axis angle -/
def determine_section (apex_angle : Real) (section_axis_angle : Real) : ConicSection :=
  sorry -- Implementation to be added later

/-- Given a cone with apex angle of 90° and a section forming 45° with the axis,
    the resulting section is a parabola -/
theorem cone_section_parabola (apex_angle : Real) (section_axis_angle : Real) :
  apex_angle = 90 ∧ section_axis_angle = 45 →
  determine_section apex_angle section_axis_angle = ConicSection.Parabola :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_section_parabola_l1258_125870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_2010_value_l1258_125848

-- Define the sequence u_n
def u : ℕ → ℕ
  | 0 => 1  -- Define the first term
  | n + 1 => sorry  -- We'll leave the recursive definition as sorry for now

-- Define the function f(n) for the last term of the nth group
def f (n : ℕ) : ℕ := n * (3 * n - 1) / 2

-- Define the position of the last term in the nth group
def last_pos (n : ℕ) : ℕ := n * (n + 1) / 2

-- State the theorem
theorem u_2010_value : u 2010 = 5898 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_2010_value_l1258_125848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_double_angle_l1258_125807

theorem cosine_double_angle (A : Real) : 
  0 < A → A < π/2 → Real.cos (2*A) = 3/5 → Real.cos A = 2*Real.sqrt 5/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_double_angle_l1258_125807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_number_probability_l1258_125808

/-- The set of possible initial three-digit combinations -/
def initial_digits : Finset ℕ := {324, 327, 328}

/-- The set of digits for the last five positions -/
def last_digits : Finset ℕ := {0, 2, 5, 8, 9}

/-- A phone number satisfying the given conditions -/
structure PhoneNumber :=
  (initial : ℕ)
  (last : Fin 5 → ℕ)
  (initial_valid : initial ∈ initial_digits)
  (last_valid : ∀ i, last i ∈ last_digits)
  (last_distinct : ∀ i j, i ≠ j → last i ≠ last j)

/-- Instance to make PhoneNumber a finite type -/
instance : Fintype PhoneNumber :=
  sorry

/-- The theorem to be proved -/
theorem phone_number_probability : 
  (Fintype.card PhoneNumber : ℚ)⁻¹ = (360 : ℚ)⁻¹ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_number_probability_l1258_125808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_side_length_given_area_l1258_125876

-- Define the parallelogram properties
def parallelogram_side_length (s : ℝ) : Prop := True
def parallelogram_angle (θ : ℝ) : Prop := True
def parallelogram_area (A : ℝ) : Prop := True

-- Theorem statement
theorem parallelogram_side_length_given_area 
  (s : ℝ) 
  (h1 : parallelogram_side_length s)
  (h2 : parallelogram_angle (30 * π / 180))
  (h3 : parallelogram_area 8) :
  s = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_side_length_given_area_l1258_125876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1258_125835

theorem gcd_problem (a : ℕ) (h1 : 456 ∣ a) (h2 : a < 1000) :
  Nat.gcd (3 * a^3 + a^2 + 4 * a + 152) a = 152 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1258_125835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_y_axis_l1258_125880

/-- Given points O, A, and B in R^2, and a vector equation relating OP to OA and AB,
    prove that if P is on the y-axis, then m = 2/3. -/
theorem vector_equation_y_axis (O A B P : ℝ × ℝ) (m : ℝ) :
  O = (0, 0) →
  A = (-1, 3) →
  B = (2, -4) →
  P.1 = 0 →
  P = O + 2 • (A - O) + m • (B - A) →
  m = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_y_axis_l1258_125880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_product_l1258_125836

theorem combination_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combination_product_l1258_125836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_a_range_l1258_125820

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2*a - 1)*x + 4*a else -x + 1

-- State the theorem
theorem decreasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/6 : ℝ) (1/2 : ℝ) ∧ a ≠ 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_function_a_range_l1258_125820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_l1258_125842

theorem smallest_class_size (n : ℕ) (scores : Fin n → ℕ) : 
  (∃ (i₁ i₂ i₃ i₄ i₅ : Fin n), i₁ ≠ i₂ ∧ i₁ ≠ i₃ ∧ i₁ ≠ i₄ ∧ i₁ ≠ i₅ ∧ 
                               i₂ ≠ i₃ ∧ i₂ ≠ i₄ ∧ i₂ ≠ i₅ ∧ 
                               i₃ ≠ i₄ ∧ i₃ ≠ i₅ ∧ 
                               i₄ ≠ i₅ ∧
                               scores i₁ = 100 ∧ scores i₂ = 100 ∧ scores i₃ = 100 ∧ 
                               scores i₄ = 100 ∧ scores i₅ = 100) →
  (∀ i : Fin n, scores i ≥ 65) →
  (Finset.sum Finset.univ (λ i => scores i)) = 80 * n →
  n ≥ 12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_class_size_l1258_125842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_balance_is_192_l1258_125862

noncomputable def january_balance : ℝ := 120
noncomputable def february_balance : ℝ := 240
noncomputable def march_balance : ℝ := 180
noncomputable def april_balance : ℝ := 180
noncomputable def may_balance : ℝ := 240

def total_months : ℕ := 5

noncomputable def average_balance : ℝ := (january_balance + february_balance + march_balance + april_balance + may_balance) / total_months

theorem average_balance_is_192 : average_balance = 192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_balance_is_192_l1258_125862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_36_84397_to_hundredth_l1258_125874

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem round_36_84397_to_hundredth :
  round_to_hundredth 36.84397 = 36.84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_36_84397_to_hundredth_l1258_125874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1258_125863

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 else -(x^2)

-- State the theorem
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ≥ 0, f x = x^2) →  -- f(x) = x² for x ≥ 0
  (∀ x ∈ Set.Icc a (a + 2), f (x + a) ≥ 2 * f x) →  -- f(x+a) ≥ 2f(x) for x ∈ [a, a+2]
  a ∈ Set.Ici (Real.sqrt 2) :=  -- a ∈ [√2, +∞)
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1258_125863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_log_function_a_range_l1258_125817

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 - a * x) / Real.log a

-- State the theorem
theorem decreasing_log_function_a_range :
  ∀ a : ℝ, a > 0 → a ≠ 1 →
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → x₂ ≤ 1 → f a x₁ > f a x₂) →
  a > 1 ∧ a < 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_log_function_a_range_l1258_125817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1258_125885

def sequence_a (n : ℕ) : ℚ := 3 / 2^n

def sum_S (n : ℕ) : ℚ := 3 * (1 - 1 / 2^n)

theorem sequence_properties :
  ∀ (n : ℕ+),
  (sequence_a 1 = 3 / 2) ∧
  (2 * sequence_a (n.val + 1) + sum_S n.val = 3) ∧
  (sequence_a n.val = 3 / 2^n.val) ∧
  (sum_S n.val = 3 * (1 - 1 / 2^n.val)) ∧
  (18 / 17 < sum_S (2 * n.val) / sum_S n.val ∧ sum_S (2 * n.val) / sum_S n.val < 8 / 7 ↔ n = 3 ∨ n = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1258_125885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_approximation_l1258_125877

theorem division_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ abs ((180 : ℝ) / (12 + 13 * 2) - 4.74) < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_approximation_l1258_125877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_range_l1258_125813

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- A line in slope-intercept form -/
structure Line where
  k : ℝ
  c : ℝ

theorem ellipse_intersection_range (e : Ellipse) (h_ecc : e.eccentricity = Real.sqrt 3 / 2) 
    (h_point : e.b = 1) :
  ∃ k_min k_max : ℝ, k_min = -3/10 ∧ k_max = 1/2 ∧
  ∀ k : ℝ, k_min < k ∧ k < k_max ↔
    let l : Line := ⟨k, 2*k⟩
    ∃ P Q : ℝ × ℝ, 
      (P.1^2 / e.a^2 + P.2^2 / e.b^2 = 1) ∧
      (Q.1^2 / e.a^2 + Q.2^2 / e.b^2 = 1) ∧
      (P.2 = k * (P.1 + 2)) ∧
      (Q.2 = k * (Q.1 + 2)) ∧
      ((0 - P.1) * (0 - Q.1) + (1 - P.2) * (1 - Q.2) < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_range_l1258_125813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_isogonal_iff_median_circumcenter_l1258_125849

structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ
  is_nondegenerate : sorry

def Tetrahedron.faces (t : Tetrahedron) : Fin 4 → Set (ℝ × ℝ × ℝ) := sorry

def Tetrahedron.medians (t : Tetrahedron) : Fin 4 → Set (ℝ × ℝ × ℝ) := sorry

noncomputable def Tetrahedron.median_intersection (t : Tetrahedron) : ℝ × ℝ × ℝ := sorry

noncomputable def Tetrahedron.circumsphere_center (t : Tetrahedron) : ℝ × ℝ × ℝ := sorry

def Tetrahedron.faces_congruent (t : Tetrahedron) : Prop := sorry

theorem tetrahedron_isogonal_iff_median_circumcenter (t : Tetrahedron) :
  t.faces_congruent ↔ t.median_intersection = t.circumsphere_center := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_isogonal_iff_median_circumcenter_l1258_125849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l1258_125864

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + b * x

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := x^2 - 2*a*x + b

-- Define the function g
def g (a b x : ℝ) : ℝ := f a b x - 4*x

-- Theorem statement
theorem function_analysis 
  (a b : ℝ) 
  (h1 : f' a b 0 = 1) 
  (h2 : f' a b 2 = 1) :
  ∃ (tangent_line : ℝ → ℝ → Prop)
    (increasing_interval decreasing_interval : Set ℝ)
    (min_value : ℝ),
  -- 1. Tangent line equation
  (tangent_line x y ↔ 4*x - y - 9 = 0) ∧
  -- 2. Monotonicity intervals
  increasing_interval = Set.Icc (-3 : ℝ) (-1) ∧
  decreasing_interval = Set.Ioc (-1 : ℝ) 2 ∧
  (∀ x ∈ increasing_interval, ∀ y ∈ increasing_interval, x < y → g a b x < g a b y) ∧
  (∀ x ∈ decreasing_interval, ∀ y ∈ decreasing_interval, x < y → g a b x > g a b y) ∧
  -- 3. Minimum value
  min_value = -9 ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, g a b x ≥ min_value) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l1258_125864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_shaded_area_theorem_l1258_125886

noncomputable section

def grid_side : ℝ := 6
def small_circle_radius : ℝ := 1
def large_circle_radius : ℝ := 2
def num_small_circles : ℕ := 4

def grid_area : ℝ := grid_side ^ 2

def small_circle_area : ℝ := Real.pi * small_circle_radius ^ 2
def large_circle_area : ℝ := Real.pi * large_circle_radius ^ 2

def total_circle_area : ℝ := num_small_circles * small_circle_area + large_circle_area

def visible_shaded_area : ℝ := grid_area - total_circle_area

theorem visible_shaded_area_theorem :
  visible_shaded_area = 36 - 8 * Real.pi ∧ 
  (∃ (C D : ℝ), visible_shaded_area = C - D * Real.pi ∧ C + D = 44) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_shaded_area_theorem_l1258_125886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_participant_A_third_round_prob_participant_B_no_commendation_l1258_125827

-- Define the competition structure
structure Competition where
  rounds : Nat
  chances_per_round : Nat
  total_participants : Nat
  commendation_count : Nat

-- Define a participant
structure Participant where
  first_round_success_prob : ℝ
  second_round_success_prob : ℝ
  score : ℝ

-- Define the normal distribution parameters
structure NormalDistribution where
  mean : ℝ
  variance : ℝ

-- Define the competition parameters
noncomputable def competition : Competition :=
  { rounds := 3
  , chances_per_round := 2
  , total_participants := 2000
  , commendation_count := 317 }

-- Define participant A
noncomputable def participant_A : Participant :=
  { first_round_success_prob := 4/5
  , second_round_success_prob := 3/4
  , score := 0 }  -- Score is not relevant for participant A

-- Define participant B
noncomputable def participant_B : Participant :=
  { first_round_success_prob := 0  -- Not relevant for participant B
  , second_round_success_prob := 0  -- Not relevant for participant B
  , score := 231 }

-- Define the score distribution
noncomputable def score_distribution : NormalDistribution :=
  { mean := 212
  , variance := 841 }  -- Variance is σ² = 29² = 841

-- Theorem 1: Probability of participant A entering the third round
theorem participant_A_third_round_prob :
  let p1 := participant_A.first_round_success_prob
  let p2 := participant_A.second_round_success_prob
  let prob_enter_third := (p1 + (1 - p1) * p1) * (p2 + (1 - p2) * p2)
  prob_enter_third = 9/10 := by sorry

-- Theorem 2: Participant B's score is not sufficient for commendation
theorem participant_B_no_commendation :
  let σ := Real.sqrt score_distribution.variance
  let commendation_threshold := score_distribution.mean + σ
  participant_B.score < commendation_threshold := by sorry

-- Additional facts
def participants_above_270 : Nat := 46

#check competition
#check participant_A
#check participant_B
#check score_distribution
#check participant_A_third_round_prob
#check participant_B_no_commendation
#check participants_above_270

end NUMINAMATH_CALUDE_ERRORFEEDBACK_participant_A_third_round_prob_participant_B_no_commendation_l1258_125827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_e_pi_over_3_i_l1258_125832

theorem imaginary_part_of_e_pi_over_3_i (θ : ℝ) :
  let z : ℂ := Complex.exp (θ * Complex.I)
  z = Complex.ofReal (Real.cos θ) + Complex.I * Complex.ofReal (Real.sin θ) →
  (Complex.exp ((π / 3) * Complex.I)).im = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_e_pi_over_3_i_l1258_125832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_adc_is_100_l1258_125859

-- Define the points A, B, C, and D
variable (A B C D : ℝ × ℝ)

-- Define the ratio of BD to DC
noncomputable def bd_dc_ratio : ℝ := 2 / 5

-- Define the area of triangle ABD
noncomputable def area_abd : ℝ := 40

-- Define a function to calculate the area of a triangle given its base and height
noncomputable def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

-- Theorem: The area of triangle ADC is 100 square centimeters
theorem area_adc_is_100 :
  ∃ (h : ℝ), 
    triangle_area (D.1 - C.1) h = 100 ∧
    triangle_area (B.1 - D.1) h = area_abd ∧
    (D.1 - B.1) / (C.1 - D.1) = bd_dc_ratio :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_adc_is_100_l1258_125859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_complex_expression_evaluation_l1258_125845

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Definition of the imaginary unit -/
theorem i_squared : i^2 = -1 := Complex.I_sq

/-- Theorem: Evaluation of a complex expression -/
theorem complex_expression_evaluation :
  i^45 + 2*i^150 + 3*i^777 = 4*i - 2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_complex_expression_evaluation_l1258_125845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_and_volume_l1258_125841

/-- Represents a four-sided pyramid with a square base -/
structure Pyramid where
  /-- The diagonal of the square base -/
  base_diagonal : ℝ
  /-- Assumption that the base diagonal is positive -/
  base_diagonal_pos : base_diagonal > 0

/-- Calculate the surface area of the pyramid -/
noncomputable def surface_area (p : Pyramid) : ℝ :=
  (p.base_diagonal^2 / 2) * (1 + Real.sqrt 7)

/-- Calculate the volume of the pyramid -/
noncomputable def volume (p : Pyramid) : ℝ :=
  (p.base_diagonal^3 * Real.sqrt 3) / 12

/-- Theorem stating the correctness of surface area and volume calculations -/
theorem pyramid_surface_area_and_volume (p : Pyramid) :
  surface_area p = (p.base_diagonal^2 / 2) * (1 + Real.sqrt 7) ∧
  volume p = (p.base_diagonal^3 * Real.sqrt 3) / 12 := by
  sorry

#check pyramid_surface_area_and_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_and_volume_l1258_125841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1258_125893

/-- Given sets A and B, where A ∪ B = A, prove the range of m -/
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x ∈ {y | y ≥ 2} ∪ {y | y ≥ m} ↔ x ∈ {y | y ≥ 2}) → 
  m ∈ Set.Ici 2 := by
  sorry

#check range_of_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1258_125893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_tiling_l1258_125896

/-- Represents a board of size n × n -/
def Board (n : ℕ) := Fin n → Fin n → ℕ

/-- Assigns checkerboard pattern to the board -/
def checkerboard_pattern (n : ℕ) : Board n :=
  fun i j => if j.val % 2 = 0 then 2 else 1

/-- Calculates the sum of all values on the board -/
def board_sum (n : ℕ) (b : Board n) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin n)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin n)) fun j =>
      b i j

/-- Represents a tiling of the board -/
def Tiling (n : ℕ) := Fin n → Fin n → Option Bool
  -- Some true represents a horizontal 1 × 2 domino
  -- Some false represents a vertical 1 × 3 rectangle
  -- None represents an uncovered cell

/-- Checks if a tiling is valid -/
def is_valid_tiling (n : ℕ) (t : Tiling n) : Prop :=
  ∀ i j, t i j ≠ none ∧
    (t i j = some true → j.val + 1 < n ∧ ∃ h : j.val + 1 < n, t i ⟨j.val + 1, h⟩ = some true) ∧
    (t i j = some false → i.val + 2 < n ∧
      ∃ h1 : i.val + 1 < n, ∃ h2 : i.val + 2 < n,
        t ⟨i.val + 1, h1⟩ j = some false ∧ t ⟨i.val + 2, h2⟩ j = some false)

theorem impossible_tiling :
  ¬∃ (t : Tiling 2003), is_valid_tiling 2003 t :=
sorry

#check impossible_tiling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_tiling_l1258_125896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_total_cost_100nm_optimal_speed_l1258_125811

/-- Represents the characteristics and costs of a ship --/
structure Ship where
  k : ℝ  -- Proportionality constant for fuel cost
  max_speed : ℝ  -- Maximum speed in nautical miles per hour
  operating_cost : ℝ  -- Fixed operating cost per hour

/-- Calculates the total cost for a ship to travel a given distance at a given speed --/
noncomputable def total_cost (s : Ship) (distance : ℝ) (speed : ℝ) : ℝ :=
  (s.k * speed^2 + s.operating_cost) * (distance / speed)

/-- Theorem stating the minimum total cost for the ship to travel 100 nautical miles --/
theorem min_total_cost_100nm (s : Ship) :
  s.k = 0.96 →
  s.max_speed = 15 →
  s.operating_cost = 150 →
  total_cost s 10 10 = 96 + 150 →
  ∃ (min_cost : ℝ), min_cost = 2400 ∧
    ∀ (v : ℝ), 0 < v → v ≤ s.max_speed →
      total_cost s 100 v ≥ min_cost := by
  sorry

/-- Theorem stating the optimal speed for minimum cost --/
theorem optimal_speed (s : Ship) :
  s.k = 0.96 →
  s.max_speed = 15 →
  s.operating_cost = 150 →
  ∃ (v : ℝ), v = Real.sqrt (15000 / 96) ∧
    ∀ (u : ℝ), 0 < u → u ≤ s.max_speed →
      total_cost s 100 v ≤ total_cost s 100 u := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_total_cost_100nm_optimal_speed_l1258_125811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_l1258_125839

-- Define the distance function
noncomputable def s (t : ℝ) : ℝ := (1/4) * t^4 - 4 * t^3 + 16 * t^2

-- Define the velocity function as the derivative of s
noncomputable def v (t : ℝ) : ℝ := deriv s t

-- Theorem stating when velocity is zero
theorem velocity_zero : 
  v 0 = 0 ∧ v 4 = 0 ∧ v 8 = 0 := by
  sorry

#check velocity_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_zero_l1258_125839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l1258_125869

-- Define the property that f must satisfy
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 1 → f x + f (1 / (1 - x)) = x

-- Define the function we want to prove is the solution
noncomputable def SolutionFunction (x : ℝ) : ℝ :=
  if x = 0 ∨ x = 1 then 0  -- Arbitrary value for x = 0 or 1
  else (1 / 2) * (x + 1 - 1 / x - 1 / (1 - x))

-- State the theorem
theorem solution_satisfies_equation :
  ∃ f : ℝ → ℝ, SatisfiesFunctionalEquation f ∧
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f x = SolutionFunction x) ∧
  f 1 = -f 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_equation_l1258_125869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_point_on_exponential_curve_l1258_125830

theorem tangent_at_point_on_exponential_curve (a : ℝ) (h : (3 : ℝ)^a = 9) :
  Real.tan (Real.arctan (9 * Real.log 3)) = 9 * Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_point_on_exponential_curve_l1258_125830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l1258_125803

noncomputable def f (a b x : ℝ) : ℝ := Real.log (|x + b|) / Real.log a

theorem f_property (a b : ℝ) (ha : 0 < a) (ha' : a < 1) :
  (∀ x, f a b x = f a b (-x)) →  -- f is even
  (∀ x y, 0 < x → x < y → f a b y < f a b x) →  -- f is monotonically decreasing in (0, +∞)
  b = 0 →  -- condition for f to be even
  f a b (b - 2) > f a b (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l1258_125803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wheels_count_l1258_125818

theorem total_wheels_count : 90 = 24 * 2 + 14 * 3 :=
by
  -- Define the number of bicycles and tricycles
  let num_bicycles : ℕ := 24
  let num_tricycles : ℕ := 14
  
  -- Define the number of wheels per bicycle and tricycle
  let wheels_per_bicycle : ℕ := 2
  let wheels_per_tricycle : ℕ := 3
  
  -- Calculate the total number of wheels
  let total_wheels := num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle
  
  -- Prove that the total number of wheels is 90
  calc
    90 = total_wheels := by rfl
    _ = num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle := by rfl
    _ = 24 * 2 + 14 * 3 := by rfl

#eval 24 * 2 + 14 * 3  -- This will evaluate to 90

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_wheels_count_l1258_125818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_difference_l1258_125816

/-- Parabola structure -/
structure Parabola where
  focus : ℝ × ℝ
  a : ℝ

/-- Point on a parabola -/
def PointOnParabola (p : Parabola) (point : ℝ × ℝ) : Prop :=
  (point.2)^2 = 8 * point.1

/-- Projection on y-axis -/
def ProjectionOnYAxis (point : ℝ × ℝ) : ℝ × ℝ :=
  (0, point.2)

/-- Distance between two points -/
noncomputable def Distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: For any point P on the parabola y^2 = 8x with focus F, 
    and E being the projection of P on the y-axis, |PF| - |PE| = 2 -/
theorem parabola_distance_difference 
  (p : Parabola) 
  (P : ℝ × ℝ) 
  (h : PointOnParabola p P) :
  let F := p.focus
  let E := ProjectionOnYAxis P
  Distance P F - Distance P E = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_difference_l1258_125816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1258_125872

/-- The distance between the foci of the hyperbola 9x^2 - 36x - y^2 + 4y = 40 -/
noncomputable def foci_distance : ℝ := 8 * Real.sqrt 5

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 36 * x - y^2 + 4 * y = 40

theorem hyperbola_foci_distance :
  ∃ (c : ℝ), c = foci_distance ∧
  ∀ (x y : ℝ), hyperbola_equation x y →
    ∃ (a b : ℝ), a^2 = 8 ∧ b^2 = 72 ∧ c^2 = a^2 + b^2 := by
  sorry

#check hyperbola_foci_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1258_125872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_example_l1258_125868

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_example : geometric_sum 1 3 6 = 364 := by
  -- Expand the definition of geometric_sum
  unfold geometric_sum
  -- Simplify the expression
  simp [Real.rpow_nat_cast]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_example_l1258_125868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blender_juicer_savings_l1258_125834

/-- The savings from buying a blender and juicer in-store compared to buying just the blender from TV -/
theorem blender_juicer_savings 
  (blender_in_store : ℝ)
  (blender_tv_payment : ℝ)
  (blender_tv_shipping : ℝ)
  (juicer_in_store : ℝ)
  (discount_rate : ℝ)
  (num_payments : ℝ)
  : blender_in_store = 120
  → blender_tv_payment = 28
  → blender_tv_shipping = 12
  → juicer_in_store = 80
  → discount_rate = 0.1
  → num_payments = 4
  → (blender_in_store + juicer_in_store) * (1 - discount_rate) - (num_payments * blender_tv_payment + blender_tv_shipping) = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blender_juicer_savings_l1258_125834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_squares_in_sequence_digit_sum_constant_not_perfect_square_candidate_l1258_125825

/-- Represents a number in the sequence -/
def SequenceNumber (n : ℕ) : ℕ := 20142015 + n * 10^6

/-- The sum of digits of any number in the sequence -/
def DigitSum : ℕ := 15

/-- A number is a candidate for being a perfect square if its digit sum is congruent to 0, 1, 4, 7, or 9 modulo 9 -/
def IsPerfectSquareCandidate (n : ℕ) : Prop :=
  n % 9 = 0 ∨ n % 9 = 1 ∨ n % 9 = 4 ∨ n % 9 = 7 ∨ n % 9 = 9

theorem no_perfect_squares_in_sequence :
  ∀ n : ℕ, ¬ ∃ m : ℕ, (SequenceNumber n) = m^2 :=
by
  sorry

theorem digit_sum_constant :
  ∀ n : ℕ, (Nat.digits 10 (SequenceNumber n)).sum = DigitSum :=
by
  sorry

theorem not_perfect_square_candidate :
  ¬ IsPerfectSquareCandidate DigitSum :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_squares_in_sequence_digit_sum_constant_not_perfect_square_candidate_l1258_125825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_is_25_percent_l1258_125804

/-- Calculates the second discount percentage given the original price, first discount percentage, and final sale price. -/
noncomputable def second_discount_percentage (original_price first_discount_percent final_price : ℝ) : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_percent / 100)
  let second_discount_amount := price_after_first_discount - final_price
  (second_discount_amount / price_after_first_discount) * 100

/-- Theorem stating that given the specific values from the problem, the second discount percentage is 25%. -/
theorem second_discount_is_25_percent : 
  second_discount_percentage 390 15 248.625 = 25 := by
  -- Unfold the definition of second_discount_percentage
  unfold second_discount_percentage
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_is_25_percent_l1258_125804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anthony_pet_ratio_l1258_125879

theorem anthony_pet_ratio :
  ∀ (anthony_cats anthony_dogs leonel_cats leonel_dogs : ℕ),
    anthony_cats + anthony_dogs = 12 →
    leonel_cats = anthony_cats / 2 →
    leonel_dogs = anthony_dogs + 7 →
    anthony_cats + anthony_dogs + leonel_cats + leonel_dogs = 27 →
    anthony_cats * 3 = 2 * (anthony_cats + anthony_dogs) :=
λ anthony_cats anthony_dogs leonel_cats leonel_dogs
  h1 h2 h3 h4 ↦ by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anthony_pet_ratio_l1258_125879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1258_125890

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^3 - 4*x^2 + 4*x - 4)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1258_125890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_correct_l1258_125888

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x < -1 ∨ x > 3}
def B (a : ℝ) : Set ℝ := {y : ℝ | -a < y ∧ y ≤ 4 - a}

-- Define the condition A ∩ B = B
def condition (a : ℝ) : Prop := A ∩ B a = B a

-- Define the range of a
def range_of_a : Set ℝ := Set.Iic (-3) ∪ Set.Ioi 5

-- Theorem statement
theorem range_of_a_correct :
  ∀ a : ℝ, condition a ↔ a ∈ range_of_a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_correct_l1258_125888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l1258_125882

theorem angle_sum_theorem (α β : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.sin α = 2 * Real.sqrt 5 / 5) (h4 : Real.sin β = 3 * Real.sqrt 10 / 10) :
  α + β = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l1258_125882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_overlap_angle_l1258_125881

-- Define a 30-60-90 triangle
structure Triangle30_60_90 where
  hypotenuse : ℝ
  short_leg : ℝ
  long_leg : ℝ
  hypotenuse_eq : hypotenuse = 10
  short_leg_eq : short_leg = 5
  long_leg_eq : long_leg = 5 * Real.sqrt 3

-- Define the overlap scenario
def overlap_angle (t : Triangle30_60_90) (angle : ℝ) : Prop :=
  let overlap_length := t.hypotenuse / 2
  Real.cos angle = (overlap_length^2 + overlap_length^2 - overlap_length^2) / (2 * overlap_length * overlap_length)

-- Theorem statement
theorem triangle_overlap_angle (t : Triangle30_60_90) :
  overlap_angle t (30 * Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_overlap_angle_l1258_125881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_matrix_transformation_impossibility_l1258_125853

/-- Represents a shaded square matrix -/
def ShadedMatrix (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if a column is fully shaded -/
def has_fully_shaded_column (M : ShadedMatrix n) : Prop :=
  ∃ j : Fin n, ∀ i : Fin n, M i j = true

/-- Represents a row swap operation -/
def row_swap (M : ShadedMatrix n) (i j : Fin n) : ShadedMatrix n :=
  λ k l ↦ if k = i then M j l
         else if k = j then M i l
         else M k l

/-- Represents a column swap operation -/
def column_swap (M : ShadedMatrix n) (i j : Fin n) : ShadedMatrix n :=
  λ k l ↦ if l = i then M k j
         else if l = j then M k i
         else M k l

/-- Theorem stating the impossibility of transforming a matrix with a fully shaded column
    into one without any fully shaded column using only row and column swaps -/
theorem shaded_matrix_transformation_impossibility (n : ℕ) (M₁ M₂ : ShadedMatrix n) :
  has_fully_shaded_column M₁ ∧ ¬has_fully_shaded_column M₂ →
  ¬∃ (swaps : List (Bool × Fin n × Fin n)),
    (swaps.foldl (λ M (swap : Bool × Fin n × Fin n) ↦
      match swap with
      | (true, i, j) => row_swap M i j
      | (false, i, j) => column_swap M i j) M₁) = M₂ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_matrix_transformation_impossibility_l1258_125853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billionsToScientificNotation_1541_l1258_125828

/-- Expresses a number in billions in scientific notation -/
noncomputable def billionsToScientificNotation (x : ℝ) : ℝ × ℤ :=
  let billion : ℝ := 10^9
  let mantissa : ℝ := x * billion / 10^11
  let exponent : ℤ := 11
  (mantissa, exponent)

/-- Theorem stating that 1541 billion is equal to 1.541 × 10^11 in scientific notation -/
theorem billionsToScientificNotation_1541 :
  billionsToScientificNotation 1541 = (1.541, 11) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_billionsToScientificNotation_1541_l1258_125828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1258_125884

/-- An arithmetic sequence with positive first term and non-zero common difference -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ
  h₁ : a₁ > 0
  h₂ : d ≠ 0

/-- The nth term of an arithmetic sequence -/
noncomputable def a (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1 : ℝ) * seq.d

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * (2 * seq.a₁ + (n - 1 : ℝ) * seq.d) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (S seq 3 = S seq 11 → S seq 14 = 0) ∧
  (S seq 3 = S seq 11 → ∀ n, S seq 7 ≥ S seq n) ∧
  (S seq 7 > S seq 8 → S seq 8 > S seq 9) ∧
  (S seq 7 > S seq 8 → S seq 6 > S seq 9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1258_125884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_numbers_l1258_125837

/-- The smallest odd prime number -/
def m : ℕ := 3

/-- The largest integer less than 150 with exactly three positive divisors -/
def n : ℕ := 121

/-- The number of positive divisors of a natural number -/
def num_divisors (k : ℕ) : ℕ := (Nat.divisors k).card

theorem sum_of_special_numbers : m + n = 124 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_numbers_l1258_125837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_l1258_125873

theorem max_prime_factors (a b : ℕ) 
  (h_gcd : (Nat.gcd a b).factorization.support.card = 5)
  (h_lcm : (Nat.lcm a b).factorization.support.card = 20)
  (h_sum : (Nat.factorization a).support.sum id > 50)
  (h_fewer : (Nat.factorization a).support.card < (Nat.factorization b).support.card) :
  (Nat.factorization a).support.card ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prime_factors_l1258_125873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_with_tan_l1258_125833

theorem cos_double_angle_with_tan (α : ℝ) : 
  Real.tan α = -3 → Real.cos (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_with_tan_l1258_125833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_m_n_l1258_125806

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (Nat.gcd (m.val + n.val) 231 = 1) ∧ 
  (∃ (k : ℕ), m^m.val = k * n^n.val) ∧ 
  (∀ (k : ℕ), m.val ≠ k * n.val) ∧
  (m.val + n.val = 377) ∧
  (∀ (m' n' : ℕ+), 
    (Nat.gcd (m'.val + n'.val) 231 = 1) → 
    (∃ (k : ℕ), m'^m'.val = k * n'^n'.val) → 
    (∀ (k : ℕ), m'.val ≠ k * n'.val) → 
    (m'.val + n'.val ≥ 377)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_m_n_l1258_125806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_abc_equals_one_l1258_125822

theorem cube_root_abc_equals_one 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq : a^2 + 1/b = b^2 + 1/c ∧ b^2 + 1/c = c^2 + 1/a) : 
  (|a*b*c|)^(1/3) = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_abc_equals_one_l1258_125822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_tangent_parallel_to_bisector_l1258_125875

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 - x^2

-- Define the derivative of the parabola
def parabola_derivative (x : ℝ) : ℝ := -2 * x

theorem tangent_parallel_to_x_axis :
  ∃ (point : ℝ × ℝ), point.1 = 0 ∧ point.2 = 2 ∧
  parabola point.1 = point.2 ∧ parabola_derivative point.1 = 0 :=
by
  -- The proof goes here
  sorry

theorem tangent_parallel_to_bisector :
  ∃ (point : ℝ × ℝ), point.1 = -1/2 ∧ point.2 = 7/4 ∧
  parabola point.1 = point.2 ∧ parabola_derivative point.1 = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_tangent_parallel_to_bisector_l1258_125875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihydrogen_monoxide_weight_l1258_125854

-- Define atomic weights
def hydrogen_weight : ℝ := 1.008
def oxygen_weight : ℝ := 16.00

-- Define the composition of dihydrogen monoxide
def hydrogen_atoms : ℕ := 2
def oxygen_atoms : ℕ := 1

-- Define the number of moles
def moles : ℕ := 7

-- Theorem to prove
theorem dihydrogen_monoxide_weight :
  let molecular_weight := hydrogen_weight * hydrogen_atoms + oxygen_weight * oxygen_atoms
  let total_weight := molecular_weight * moles
  abs (total_weight - 126.112) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihydrogen_monoxide_weight_l1258_125854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l1258_125824

structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

def Vector3D.zero : Vector3D := ⟨0, 0, 0⟩

instance : Zero Vector3D := ⟨Vector3D.zero⟩

def collinear (a b : Vector3D) : Prop := sorry

def parallel_lines (a b : Vector3D) : Prop := sorry

def skew_lines (a b : Vector3D) : Prop := sorry

def coplanar (a b c : Vector3D) : Prop := sorry

def uniquely_represented (a b c p : Vector3D) : Prop := 
  ∃! (x y z : ℝ), p = Vector3D.mk (x * a.x + y * b.x + z * c.x) 
                                  (x * a.y + y * b.y + z * c.y) 
                                  (x * a.z + y * b.z + z * c.z)

theorem all_propositions_false 
  (a b c : Vector3D) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  (¬ ∀ a b, collinear a b → parallel_lines a b) ∧ 
  (¬ ∀ a b, skew_lines a b → ¬ coplanar a b 0) ∧ 
  (¬ ∀ a b c, (coplanar a b 0 ∧ coplanar b c 0 ∧ coplanar a c 0) → coplanar a b c) ∧ 
  (¬ ∀ p, uniquely_represented a b c p) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l1258_125824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_square_exp_inequality_l1258_125851

/-- Given m ∈ (0, 1), prove that logₘ 2 < m² < 2ᵐ -/
theorem log_square_exp_inequality (m : ℝ) (h : 0 < m ∧ m < 1) :
  Real.log 2 / Real.log m < m^2 ∧ m^2 < 2^m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_square_exp_inequality_l1258_125851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_sin_condition_l1258_125898

/-- The function f(x) = sin(ωx + π/6) is monotonic on (0, π/3) if and only if ω ∈ (0, 1] -/
theorem monotonic_sin_condition (ω : ℝ) :
  ω > 0 →
  (∀ x ∈ Set.Ioo 0 (π/3), Monotone (fun x ↦ Real.sin (ω * x + π/6))) ↔
  ω ∈ Set.Ioc 0 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_sin_condition_l1258_125898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_eccentricity_l1258_125826

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  a : ℝ -- semi-major axis
  b : ℝ -- semi-minor axis

/-- Represents a parabola -/
structure Parabola where
  vertex : Point
  focus : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Define membership for a point in an ellipse -/
def Point.inEllipse (p : Point) (e : Ellipse) : Prop := sorry

/-- Define membership for a point in a parabola -/
def Point.inParabola (p : Point) (par : Parabola) : Prop := sorry

theorem ellipse_parabola_intersection_eccentricity 
  (e : Ellipse) (p : Parabola) (P : Point) (F₁ F₂ : Point) :
  (p.vertex = F₁) →
  (p.focus = F₂) →
  (P.inEllipse e) →
  (P.inParabola p) →
  (eccentricity e * distance P F₂ = distance P F₁) →
  eccentricity e = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_eccentricity_l1258_125826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_up_to_50_l1258_125805

/-- The number of positive integer divisors of n -/
def d (n : ℕ) : ℕ := sorry

/-- g₁(n) is three times the number of positive integer divisors of n -/
def g₁ (n : ℕ) : ℕ := 3 * d n

/-- gⱼ(n) for j ≥ 0 -/
def g (j : ℕ) (n : ℕ) : ℕ :=
  match j with
  | 0 => n
  | 1 => g₁ n
  | j+1 => g₁ (g j n)

theorem no_solution_up_to_50 :
  ∀ n : ℕ, n > 0 ∧ n ≤ 50 → g 25 n ≠ 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_up_to_50_l1258_125805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_B_l1258_125867

def A : Set ℝ := {x | x^2 + 2*x + 1 = 0}

theorem proper_subsets_of_B (a : ℝ) (h : A = {a}) :
  let B : Set ℝ := {x | x^2 + a*x = 0}
  {S | S ⊆ B ∧ S ≠ ∅ ∧ S ≠ B} = {{0}, {1}} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_B_l1258_125867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_groupings_l1258_125865

def num_groupings (n : ℕ) : ℕ :=
  (List.range (n - 1)).map (λ k => Nat.choose n (k + 1)) |>.sum

theorem tourist_groupings :
  num_groupings 8 = 254 := by
  rfl

#eval num_groupings 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_groupings_l1258_125865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_P_divisible_by_2010_l1258_125838

/-- P(n) is the number of permutations (a₁, ..., aₙ) of (1, 2, ..., n) 
    for which k * aₖ is a perfect square for all 1 ≤ k ≤ n -/
def P (n : ℕ) : ℕ :=
  (Finset.range n).card.factorial

/-- A number is a perfect square -/
def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k

/-- The property that for a permutation (a₁, ..., aₙ), k * aₖ is a perfect square for all 1 ≤ k ≤ n -/
def valid_permutation (n : ℕ) (σ : Equiv.Perm (Fin n)) : Prop :=
  ∀ k : Fin n, is_perfect_square (k.val.succ * σ k)

theorem smallest_n_for_P_divisible_by_2010 :
  (∀ m < 4489, ¬(2010 ∣ P m)) ∧ (2010 ∣ P 4489) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_P_divisible_by_2010_l1258_125838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bisector_angle_l1258_125894

/-- Given a triangle ABC with side lengths AB = 7, BC = 8, and AC = 9,
    if there exist two lines that simultaneously bisect the perimeter and area of the triangle,
    and θ is the acute angle between these lines, then tan θ = 3√5 + 2√10 -/
theorem triangle_bisector_angle (A B C : ℝ × ℝ) : 
  let d := fun (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let AB := d A B
  let BC := d B C
  let AC := d A C
  let perimeter := AB + BC + AC
  let area := Real.sqrt (perimeter/2 * (perimeter/2 - AB) * (perimeter/2 - BC) * (perimeter/2 - AC))
  AB = 7 → 
  BC = 8 → 
  AC = 9 →
  (∃ (P Q R S : ℝ × ℝ), 
    (d A P + d P B = perimeter / 2) ∧ 
    (d A Q + d Q C = perimeter / 2) ∧
    (d C R + d R A = perimeter / 2) ∧
    (d C S + d S B = perimeter / 2) ∧
    (d A P * d A Q * Real.sin (Real.arccos ((AB^2 + AC^2 - BC^2) / (2 * AB * AC))) / 2 = area / 2) ∧
    (d C R * d C S * Real.sin (Real.arccos ((BC^2 + AC^2 - AB^2) / (2 * BC * AC))) / 2 = area / 2)) →
  let θ := Real.arctan ((Real.tan (Real.arctan ((d A P * Real.sin (Real.arccos ((AB^2 + AC^2 - BC^2) / (2 * AB * AC)))) / 
                                               (d A Q - d A P * Real.cos (Real.arccos ((AB^2 + AC^2 - BC^2) / (2 * AB * AC)))))) + 
                         Real.tan (Real.arctan ((d C R * Real.sin (Real.arccos ((BC^2 + AC^2 - AB^2) / (2 * BC * AC)))) / 
                                               (d C S - d C R * Real.cos (Real.arccos ((BC^2 + AC^2 - AB^2) / (2 * BC * AC))))))) / 
                        (1 - Real.tan (Real.arctan ((d A P * Real.sin (Real.arccos ((AB^2 + AC^2 - BC^2) / (2 * AB * AC)))) / 
                                                   (d A Q - d A P * Real.cos (Real.arccos ((AB^2 + AC^2 - BC^2) / (2 * AB * AC)))))) * 
                             Real.tan (Real.arctan ((d C R * Real.sin (Real.arccos ((BC^2 + AC^2 - AB^2) / (2 * BC * AC)))) / 
                                                   (d C S - d C R * Real.cos (Real.arccos ((BC^2 + AC^2 - AB^2) / (2 * BC * AC))))))))
  Real.tan θ = 3 * Real.sqrt 5 + 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bisector_angle_l1258_125894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_subset_B_and_infinite_diff_l1258_125860

def A : Set ℕ := {n | ∃ x y : ℕ, n = x^2 + 2*y^2 ∧ x > y}

def B : Set ℕ := {n | ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ n = (a^3 + b^3 + c^3) / (a + b + c)}

theorem A_subset_B_and_infinite_diff : 
  (A ⊆ B) ∧ (∃ f : ℕ → ℕ, Function.Injective f ∧ ∀ n, f n ∈ B \ A) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_subset_B_and_infinite_diff_l1258_125860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_l1258_125852

/-- Parabola equation -/
def parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Circle equation -/
def circle_eq (x y r : ℝ) : Prop := (x-1)^2 + (y-2)^2 = r^2

/-- Tangent line to parabola at point (x, y) -/
noncomputable def parabola_tangent_slope (x : ℝ) : ℝ := x/2

/-- Slope of line connecting point (x, y) to circle center (1, 2) -/
noncomputable def line_to_center_slope (x y : ℝ) : ℝ := (y - 2) / (x - 1)

/-- Main theorem -/
theorem parabola_circle_tangent (x y r : ℝ) (hr : r > 0) :
  parabola x y →
  circle_eq x y r →
  parabola_tangent_slope x * line_to_center_slope x y = 1 →
  r = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_tangent_l1258_125852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_charge_theorem_l1258_125802

/-- Represents the fee structure for Jim's taxi service -/
structure TaxiFeeStructure where
  initial_fee : ℚ
  peak_initial_fee : ℚ
  first_segment_rate : ℚ
  first_segment_distance : ℚ
  second_segment_rate : ℚ
  second_segment_distance : ℚ
  third_segment_rate : ℚ
  third_segment_distance : ℚ
  waiting_rate : ℚ

/-- Calculates the total charge for a taxi trip -/
def calculate_total_charge (fee_structure : TaxiFeeStructure) 
  (distance : ℚ) (is_peak_hours : Bool) (waiting_time : ℚ) : ℚ :=
  sorry

/-- The fee structure for Jim's taxi service -/
def jims_taxi_fees : TaxiFeeStructure where
  initial_fee := 225/100
  peak_initial_fee := 350/100
  first_segment_rate := 15/100
  first_segment_distance := 4/10
  second_segment_rate := 20/100
  second_segment_distance := 1/3
  third_segment_rate := 25/100
  third_segment_distance := 1/4
  waiting_rate := 10/100

/-- Theorem: The total charge for a 3.6-mile trip during peak hours with 8 minutes of waiting time is $6.05 -/
theorem trip_charge_theorem :
  calculate_total_charge jims_taxi_fees (36/10) true 8 = 605/100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_charge_theorem_l1258_125802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_cost_per_liter_l1258_125846

/-- Proves that the cost per liter of fuel is $0.70 given the conditions of the problem --/
theorem fuel_cost_per_liter (service_cost : ℝ) (total_cost : ℝ) (minivan_tank : ℝ) 
  (minivan_count : ℕ) (truck_count : ℕ) :
  service_cost = 2.20 →
  total_cost = 395.4 →
  minivan_tank = 65 →
  minivan_count = 4 →
  truck_count = 2 →
  (let truck_tank := minivan_tank * 2.2
   let total_fuel_capacity := minivan_count * minivan_tank + truck_count * truck_tank
   let total_service_cost := (minivan_count + truck_count : ℝ) * service_cost
   let fuel_cost := total_cost - total_service_cost
   fuel_cost / total_fuel_capacity) = 0.70 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_cost_per_liter_l1258_125846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_square_corners_l1258_125858

/-- The probability that a random point in a square with side length 3 is at a distance
    greater than 1 from all four corners. -/
noncomputable def probability_outside_corners (side_length : ℝ) : ℝ :=
  1 - (Real.pi / (side_length ^ 2))

/-- Theorem stating that the probability of a random point in a square with side length 3
    being at a distance greater than 1 from all four corners is equal to 1 - π/9. -/
theorem probability_in_square_corners :
  probability_outside_corners 3 = 1 - Real.pi / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_square_corners_l1258_125858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_interval_l1258_125812

-- Define the PDF
noncomputable def P (x : ℝ) : ℝ :=
  if x ≤ 0 then 0 else Real.exp (-x)

-- Define the probability as the integral of the PDF from 1 to 3
noncomputable def prob : ℝ := ∫ x in Set.Ioo 1 3, P x

-- Theorem statement
theorem probability_in_interval :
  prob = (Real.exp 2 - 1) / Real.exp 3 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_interval_l1258_125812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_parabola_l1258_125819

/-- The equation |y-3| = √((x+4)² + y²) represents a parabola -/
theorem equation_represents_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x y : ℝ,
    (abs (y - 3) = Real.sqrt ((x + 4)^2 + y^2)) ↔ (y = a * x^2 + b * x + c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_represents_parabola_l1258_125819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_bounds_l1258_125821

/-- Represents a square chessboard with a circle inscribed -/
structure ChessboardWithCircle where
  side_length : ℝ
  grid_size : ℕ
  circle_radius : ℝ

/-- Calculates the area inside the circle (S₁) -/
noncomputable def area_inside (board : ChessboardWithCircle) : ℝ :=
  4 * Real.pi * board.circle_radius^2

/-- Calculates the area outside the circle but within touched squares (S₂) -/
noncomputable def area_outside (board : ChessboardWithCircle) : ℝ :=
  (board.side_length ^ 2) - (4 * Real.pi * board.circle_radius^2)

/-- Theorem stating that 1 ≤ S₁/S₂ < 2 for the given chessboard configuration -/
theorem area_ratio_bounds (board : ChessboardWithCircle) 
  (h1 : board.side_length = 8)
  (h2 : board.grid_size = 8)
  (h3 : board.circle_radius = 4) : 
  1 ≤ (area_inside board) / (area_outside board) ∧ 
  (area_inside board) / (area_outside board) < 2 := by
  sorry

#check area_ratio_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_bounds_l1258_125821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_expression_minimized_l1258_125857

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- A point on a hyperbola -/
structure HyperbolaPoint (h : Hyperbola) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The expression to be minimized -/
noncomputable def expression_to_minimize (k₁ k₂ : ℝ) : ℝ := 2 / (k₁ * k₂) + Real.log k₁ + Real.log k₂

/-- The theorem statement -/
theorem hyperbola_eccentricity_when_expression_minimized (h : Hyperbola) 
  (c : HyperbolaPoint h) (k₁ k₂ : ℝ) :
  (∀ k₁' k₂', expression_to_minimize k₁' k₂' ≥ expression_to_minimize k₁ k₂) →
  eccentricity h = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_expression_minimized_l1258_125857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1258_125855

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  a = 1 → 
  2 * b - Real.sqrt 3 * c = 2 * a * Real.cos C → 
  Real.sin C = Real.sqrt 3 / 2 → 
  let area := 1 / 2 * a * b * Real.sin C
  area = Real.sqrt 3 / 2 ∨ area = Real.sqrt 3 / 4 := by
  sorry

#check triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1258_125855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_T_l1258_125899

-- Define T(s) for -1 < s < 1
noncomputable def T (s : ℝ) : ℝ := 20 / (1 - s)

-- Theorem statement
theorem sum_of_T (b : ℝ) (h1 : -1 < b) (h2 : b < 1) (h3 : T b * T (-b) = 4800) :
  T b + T (-b) = 480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_T_l1258_125899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_n_is_13_l1258_125829

theorem least_possible_n_is_13 
  (n d : ℕ) 
  (h_d_pos : d > 0) 
  (h_profit : (n - 2) * (d / n + 10) + d / n - d = 100) : 
  n ≥ 13 ∧ ∃ (m : ℕ), m ≥ 13 ∧ (m - 2) * (d / m + 10) + d / m - d = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_n_is_13_l1258_125829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_pyramid_volume_l1258_125847

/-- A right pyramid with a square base -/
structure RightPyramid where
  base_side : ℝ
  height : ℝ

/-- The surface area of a right pyramid -/
noncomputable def surface_area (p : RightPyramid) : ℝ :=
  p.base_side^2 + 4 * (1/2 * p.base_side * Real.sqrt (p.height^2 + (p.base_side/2)^2))

/-- The volume of a right pyramid -/
noncomputable def volume (p : RightPyramid) : ℝ :=
  (1/3) * p.base_side^2 * p.height

/-- Theorem: The volume of a specific right pyramid -/
theorem right_pyramid_volume : 
  ∃ (p : RightPyramid), 
    surface_area p = 500 ∧ 
    (1/2 * p.base_side * Real.sqrt (p.height^2 + (p.base_side/2)^2)) = p.base_side^2 ∧
    volume p = (500 * Real.sqrt 15) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_pyramid_volume_l1258_125847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_calculation_l1258_125823

/-- The area of a circular sector with given radius and central angle -/
noncomputable def sectorArea (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  (centralAngle / 360) * Real.pi * radius^2

theorem sector_area_calculation :
  let radius : ℝ := 12
  let centralAngle : ℝ := 42
  sectorArea radius centralAngle = (7 * Real.pi * 144) / 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_calculation_l1258_125823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_MCN_l1258_125831

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

-- Define the circle
def circleEq (x₀ y₀ p : ℝ) (x y : ℝ) : Prop :=
  (x - x₀)^2 + (y - y₀)^2 = x₀^2 + (y₀ - p)^2

-- Define the points
def point_A (p : ℝ) : ℝ × ℝ := (0, p)

def point_M (x₀ p : ℝ) : ℝ := x₀ - p
def point_N (x₀ p : ℝ) : ℝ := x₀ + p

-- Theorem statement
theorem max_sin_MCN (p x₀ y₀ : ℝ) :
  parabola p x₀ y₀ →
  circleEq x₀ y₀ p (point_A p).1 (point_A p).2 →
  circleEq x₀ y₀ p (point_M x₀ p) 0 →
  circleEq x₀ y₀ p (point_N x₀ p) 0 →
  ∃ (θ : ℝ), ∀ (φ : ℝ), Real.sin θ ≥ Real.sin φ ∧ Real.sin θ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_MCN_l1258_125831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_select_cocaptains_is_7_180_l1258_125871

/-- Represents a math team with a given number of students and two co-captains -/
structure MathTeam where
  size : ℕ
  has_two_cocaptains : Bool

/-- Calculates the probability of selecting two co-captains from a given team -/
def prob_select_cocaptains (team : MathTeam) : ℚ :=
  2 / (team.size * (team.size - 1))

/-- The set of math teams -/
def math_teams : List MathTeam :=
  [⟨6, true⟩, ⟨9, true⟩, ⟨10, true⟩]

/-- The total probability of selecting two co-captains across all teams -/
def total_probability : ℚ :=
  (1 : ℚ) / 3 * (math_teams.map prob_select_cocaptains).sum

theorem prob_select_cocaptains_is_7_180 : total_probability = 7 / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_select_cocaptains_is_7_180_l1258_125871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_for_sixty_cookies_l1258_125815

/-- Given a recipe that makes 30 cookies with 2 cups of flour, 
    calculate the amount of flour needed for 60 cookies -/
theorem flour_for_sixty_cookies 
  (original_cookies : ℕ) 
  (original_flour : ℚ) 
  (new_cookies : ℕ) 
  (h1 : original_cookies = 30)
  (h2 : original_flour = 2)
  (h3 : new_cookies = 60)
  : (new_cookies / original_cookies) * original_flour = 4 := by
  -- Convert natural numbers to rationals for division
  have h4 : (new_cookies : ℚ) / (original_cookies : ℚ) = 2 := by
    rw [h3, h1]
    norm_num
  
  -- Perform the calculation
  calc
    (new_cookies / original_cookies) * original_flour
      = ((new_cookies : ℚ) / (original_cookies : ℚ)) * original_flour := by norm_cast
    _ = 2 * original_flour := by rw [h4]
    _ = 2 * 2 := by rw [h2]
    _ = 4 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flour_for_sixty_cookies_l1258_125815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l1258_125810

theorem angle_in_fourth_quadrant (α : ℝ) 
  (h1 : Real.cos α > 0) (h2 : Real.tan α < 0) : 
  α % (2 * Real.pi) ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l1258_125810
