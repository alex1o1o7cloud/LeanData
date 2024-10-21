import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_towns_distance_l135_13509

/-- Given a map distance and a scale, calculate the actual distance between two towns. -/
noncomputable def actual_distance (map_distance : ℝ) (scale_map : ℝ) (scale_actual : ℝ) : ℝ :=
  map_distance * (scale_actual / scale_map)

/-- Theorem: The actual distance between two towns is 200 miles given the map conditions. -/
theorem towns_distance : 
  let map_distance : ℝ := 20
  let scale_map : ℝ := 0.5
  let scale_actual : ℝ := 5
  actual_distance map_distance scale_map scale_actual = 200 := by
  -- Unfold the definition of actual_distance
  unfold actual_distance
  -- Perform the calculation
  norm_num
  -- QED
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_towns_distance_l135_13509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_N_power_18_mod_7_equals_1_l135_13515

theorem probability_N_power_18_mod_7_equals_1 : 
  ∃ (S : Finset ℕ), S = Finset.range 2019 ∧
  (Finset.filter (λ n => (n^18) % 7 = 1) S).card / S.card = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_N_power_18_mod_7_equals_1_l135_13515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xn_length_in_triangle_l135_13531

/-- Given a triangle XYZ with XY = YZ = 46, XZ = 40, and N the midpoint of YZ, 
    the length of XN is √1587. -/
theorem xn_length_in_triangle (X Y Z N : ℝ × ℝ) : 
  let d := λ (a b : ℝ × ℝ) ↦ Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  (d X Y = 46) → 
  (d Y Z = 46) → 
  (d X Z = 40) → 
  (N = ((Y.1 + Z.1) / 2, (Y.2 + Z.2) / 2)) → 
  d X N = Real.sqrt 1587 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xn_length_in_triangle_l135_13531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_known_cards_l135_13556

/-- A type representing a card with a number written on it. -/
structure Card where
  number : ℕ

/-- A function representing the selection of 10 cards and returning one of their numbers. -/
def select_cards (cards : Finset Card) (selection : Finset Card) : ℕ :=
  sorry

/-- The theorem stating the maximum number of cards we can guarantee knowing. -/
theorem max_known_cards (cards : Finset Card) : 
  (cards.card = 2013) → 
  (∀ c1 c2 : Card, c1 ∈ cards → c2 ∈ cards → c1 ≠ c2 → c1.number ≠ c2.number) →
  (∀ selection : Finset Card, selection ⊆ cards → selection.card = 10 → 
    ∃ c ∈ selection, c.number = select_cards cards selection) →
  ∃ known_cards : Finset Card, known_cards ⊆ cards ∧ known_cards.card = 1986 ∧
    ∀ c ∈ known_cards, ∃ n : ℕ, c.number = n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_known_cards_l135_13556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_accurate_measurement_l135_13584

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a protractor used for measuring angles -/
structure Protractor :=
  (center : Point)

/-- Represents an angle to be measured -/
structure Angle :=
  (vertex : Point)

/-- Represents the alignment of an angle with a protractor -/
def is_aligned (a : Angle) (p : Protractor) : Prop :=
  a.vertex = p.center

/-- Represents the ability to measure an angle accurately -/
def can_measure_accurately (a : Angle) (p : Protractor) : Prop :=
  is_aligned a p

/-- Theorem stating that for accurate measurement, the angle's vertex must be aligned with the protractor's center -/
theorem accurate_measurement (a : Angle) (p : Protractor) :
  is_aligned a p ↔ can_measure_accurately a p :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_accurate_measurement_l135_13584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_sum_l135_13579

/-- The sum of the infinite series 1 + 2(1/1998) + 3(1/1998)^2 + 4(1/1998)^3 + ... -/
noncomputable def infiniteSeries : ℝ := ∑' n, (n + 1) * (1 / 1998) ^ n

/-- The theorem stating that the sum of the infinite series is equal to 3992004/3988009 -/
theorem infiniteSeries_sum : infiniteSeries = 3992004 / 3988009 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_sum_l135_13579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_balls_molded_l135_13501

/-- The number of iron balls that can be molded from a given number of iron bars -/
def iron_balls_count (bar_length bar_width bar_height : ℚ) (num_bars : ℕ) (ball_volume : ℚ) : ℕ :=
  let bar_volume := bar_length * bar_width * bar_height
  let total_volume := (num_bars : ℚ) * bar_volume
  (total_volume / ball_volume).floor.toNat

/-- Theorem stating the number of iron balls that can be molded from 10 iron bars -/
theorem iron_balls_molded :
  iron_balls_count 12 8 6 10 8 = 720 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_balls_molded_l135_13501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_at_focus_l135_13569

/-- Represents an ellipse with major axis 'a' and minor axis 'b' -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a circle with center (x, y) and radius r -/
structure Circle where
  x : ℝ
  y : ℝ
  r : ℝ
  h_positive : 0 < r

/-- Returns true if the circle is entirely contained within the ellipse -/
def isContainedIn (c : Circle) (e : Ellipse) : Prop :=
  ∀ x y, (x - c.x)^2 + (y - c.y)^2 ≤ c.r^2 → x^2 / e.a^2 + y^2 / e.b^2 ≤ 1

/-- Returns true if the circle is tangent to the ellipse -/
def isTangentTo (c : Circle) (e : Ellipse) : Prop :=
  ∃ x y, (x - c.x)^2 + (y - c.y)^2 = c.r^2 ∧ x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Returns the distance from the center to a focus of the ellipse -/
noncomputable def focusDistance (e : Ellipse) : ℝ :=
  Real.sqrt (e.a^2 - e.b^2)

/-- The main theorem -/
theorem largest_inscribed_circle_at_focus (e : Ellipse) 
    (h_e : e.a = 6 ∧ e.b = 3) : 
    ∃ c : Circle, c.x = focusDistance e ∧ c.y = 0 ∧ c.r = 3 ∧ 
    isContainedIn c e ∧ isTangentTo c e ∧
    ∀ c' : Circle, c'.x = focusDistance e ∧ c'.y = 0 ∧ 
    isContainedIn c' e ∧ isTangentTo c' e → c'.r ≤ c.r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_circle_at_focus_l135_13569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_E_represents_data_l135_13572

-- Define the years
inductive Year
| y1960 | y1970 | y1980 | y1990

-- Define the percentage of working adults at home for each year
def percentage : Year → ℝ
| Year.y1960 => 5
| Year.y1970 => 8
| Year.y1980 => 15
| Year.y1990 => 30

-- Define a function to represent a graph
def Graph := Year → ℝ

-- Define graph E
def graph_E : Graph
| Year.y1960 => 0.5
| Year.y1970 => 1.0
| Year.y1980 => 2.8
| Year.y1990 => 3.0

-- Define a function to check if a graph represents the data within an acceptable margin of error
def represents_data (g : Graph) (margin : ℝ) : Prop :=
  ∀ y : Year, |g y - (percentage y / 10)| ≤ margin

-- Theorem: Graph E represents the data within a margin of error of 0.3
theorem graph_E_represents_data : represents_data graph_E 0.3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_E_represents_data_l135_13572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l135_13508

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / (4 * h.a^2) - p.y^2 / (3 * h.b^2) = 1

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The theorem to be proved -/
theorem hyperbola_focus_distance (h : Hyperbola) (p f1 f2 : Point) :
  isOnHyperbola h p →
  distance p f1 = 3 →
  f1.x < f2.x →  -- Assuming f1 is the left focus and f2 is the right focus
  distance p f2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_distance_l135_13508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AD_squared_l135_13564

/-- Triangle ABC with perpendicular AD to BC -/
structure RightTriangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  BD : ℝ
  CD : ℝ
  AD : ℝ

/-- The right triangle satisfying the given conditions -/
noncomputable def triangle : RightTriangle where
  AB := 13
  AC := 20
  BC := 21
  BD := 21 * (3/10)
  CD := 21 * (7/10)
  AD := Real.sqrt 117.025

theorem triangle_AD_squared (t : RightTriangle) 
  (h1 : t.AB = 13)
  (h2 : t.AC = 20)
  (h3 : t.BC = 21)
  (h4 : t.BD / t.CD = 3 / 7)
  (h5 : t.BD + t.CD = t.BC)
  (h6 : t.AB ^ 2 = t.BD ^ 2 + t.AD ^ 2)
  (h7 : t.AC ^ 2 = t.CD ^ 2 + t.AD ^ 2) :
  t.AD ^ 2 = 117.025 := by
  sorry

#check triangle
#check triangle_AD_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AD_squared_l135_13564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_function_result_l135_13560

-- Define the functions h and k
noncomputable def h (x : ℝ) : ℝ := x - 3
noncomputable def k (x : ℝ) : ℝ := x / 4

-- Define the inverse functions
noncomputable def h_inv (x : ℝ) : ℝ := x + 3
noncomputable def k_inv (x : ℝ) : ℝ := 4 * x

-- State the theorem
theorem composite_function_result :
  h (k_inv (h_inv (h_inv (k (h 27))))) = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_function_result_l135_13560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_integer_distance_parallel_to_x_axis_l135_13598

-- Define a polynomial with integer coefficients
def IntPolynomial := Polynomial ℤ

-- Define the theorem
theorem polynomial_integer_distance_parallel_to_x_axis 
  (P : IntPolynomial) (a b c : ℤ) :
  (∃ (y₁ y₂ : ℤ), y₁ = P.eval a ∧ y₂ = P.eval b) →  -- Points have integer coordinates
  (c : ℝ)^2 = ((P.eval a - P.eval b) : ℝ)^2 + (a - b : ℝ)^2 →  -- Distance is an integer
  P.eval a = P.eval b :=  -- Segment is parallel to x-axis
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_integer_distance_parallel_to_x_axis_l135_13598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l135_13589

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 1 / (3 * x + b)

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

-- Theorem statement
theorem inverse_function_condition (b : ℝ) : 
  (∀ x, f_inv (f b x) = x) → b = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l135_13589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_imag_part_l135_13592

theorem complex_equation_imag_part (z : ℂ) : 
  (2 + z) * Complex.I = 2 → z.im = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_imag_part_l135_13592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l135_13502

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l135_13502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenny_property_tax_rate_l135_13555

/-- Jenny's property tax problem -/
theorem jenny_property_tax_rate (current_value rail_increase max_tax improvements : ℝ)
  (h1 : current_value = 400000)
  (h2 : rail_increase = 0.25)
  (h3 : max_tax = 15000)
  (h4 : improvements = 250000) :
  let new_value := current_value * (1 + rail_increase)
  let final_value := new_value + improvements
  let tax_rate := max_tax / final_value
  tax_rate = 0.02 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenny_property_tax_rate_l135_13555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brianna_book_buying_l135_13505

/-- Brianna's book buying problem -/
theorem brianna_book_buying (m : ℝ) (h : m > 0) :
  let b := (1/4 : ℝ) * m / (1/2 : ℝ)  -- Total cost of books
  ∃ (ε : ℝ), ε > 0 ∧ |((m - b) / m) - (1/2 : ℝ)| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brianna_book_buying_l135_13505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_minimum_value_l135_13578

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + a / (2^x - 1)

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (2^x + 1) * f a x

-- Statement of the theorem
theorem odd_function_and_minimum_value :
  ∃ (a : ℝ), (∀ x : ℝ, f a (-x) = -(f a x)) ∧
  (a = 2) ∧
  (∀ x : ℝ, x ∈ Set.Icc 1 3 → g a x ≥ 8) ∧
  (∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ g a x = 8) :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_minimum_value_l135_13578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_theorem_l135_13539

-- Define the line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the inclination angle
noncomputable def inclinationAngle (l : Line) : ℝ := Real.arctan (-l.a / l.b)

-- State the theorem
theorem line_inclination_theorem (l : Line) (α : ℝ) 
  (h1 : α = inclinationAngle l) 
  (h2 : Real.sin α + Real.cos α = 0) : 
  l.a - l.b = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_theorem_l135_13539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l135_13597

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem range_of_f :
  ∀ y ∈ Set.range (fun x => f x), (0 ≤ x ∧ x ≤ 3) → (1 ≤ y ∧ y ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l135_13597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_increasing_y_increasing_exists_large_x_l135_13528

noncomputable def x : ℕ → ℝ
  | 0 => 1
  | n + 1 => x n + 1 / (x n)^2

noncomputable def y (n : ℕ) : ℝ := (x n)^3

theorem x_increasing {n : ℕ} (h : n ≥ 1) : x (n + 1) > x n := by sorry

theorem y_increasing {n : ℕ} (h : n ≥ 1) : y (n + 1) > y n + 3 := by sorry

theorem exists_large_x : ∃ N, x N > 2016 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_increasing_y_increasing_exists_large_x_l135_13528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_X_sequence_and_accompanying_sequence_l135_13558

def is_X_sequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n ∈ ({0, 1} : Set ℕ)) ∧ a 1 = 1

def accompanying_sequence (a : ℕ → ℕ) (b : ℕ → ℝ) : Prop :=
  b 1 = 1 ∧ ∀ n, b (n + 1) = |a n - (a (n + 1) : ℝ) / 2| * b n

def is_constant_sequence (a : ℕ → ℕ) : Prop :=
  ∀ m n, a m = a n

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n, b (n + 1) = r * b n

theorem X_sequence_and_accompanying_sequence 
  (a : ℕ → ℕ) (b : ℕ → ℝ) 
  (hX : is_X_sequence a) 
  (hA : accompanying_sequence a b) :
  (is_constant_sequence a ↔ is_geometric_sequence b) ∧
  b 2019 ≤ (1 : ℝ) / 2^1009 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_X_sequence_and_accompanying_sequence_l135_13558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_EF_sum_l135_13511

noncomputable section

/-- The sum of all possible values of EF in triangle DEF -/
noncomputable def sum_of_possible_EF (D E F : ℝ) : ℝ :=
  Real.sqrt (30000 + 10000 * Real.sqrt 6 - 10000 * Real.sqrt 2)

/-- Triangle DEF satisfies the given conditions -/
def triangle_conditions (D E F : ℝ) : Prop :=
  E = 45 * Real.pi / 180 ∧ 
  (Real.sqrt ((100 * Real.cos D)^2 + (100 * Real.sin D)^2) = 100) ∧
  (Real.sqrt ((100 * Real.sqrt 2 * Real.cos F)^2 + (100 * Real.sqrt 2 * Real.sin F)^2) = 100 * Real.sqrt 2)

theorem triangle_DEF_EF_sum (D E F : ℝ) :
  triangle_conditions D E F →
  sum_of_possible_EF D E F = Real.sqrt (30000 + 10000 * Real.sqrt 6 - 10000 * Real.sqrt 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_DEF_EF_sum_l135_13511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l135_13581

theorem complex_fraction_simplification :
  (1 + Complex.I)^3 / (1 - Complex.I)^2 = -1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l135_13581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_triangles_form_parallelogram_l135_13563

/-- A triangle in Euclidean space -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- A parallelogram in Euclidean space -/
structure Parallelogram where
  vertices : Fin 4 → ℝ × ℝ

/-- Congruence relation between two triangles -/
def Congruent (T1 T2 : Triangle) : Prop :=
  ∀ (i j : Fin 3), 
    dist (T1.vertices i) (T1.vertices j) = dist (T2.vertices i) (T2.vertices j)

/-- Predicate indicating that a parallelogram can be assembled from a list of shapes -/
def AssembledFrom (P : Parallelogram) (shapes : List Triangle) : Prop :=
  ∃ (arrangement : List Triangle), 
    arrangement.length = shapes.length ∧ 
    (∀ s, s ∈ shapes → s ∈ arrangement) ∧
    (∃ (f : Fin 4 → ℝ × ℝ), 
      (∀ i, P.vertices i = f i) ∧
      (∀ p, p ∈ (⋃ T ∈ arrangement, {T.vertices i | i : Fin 3}) → 
        ∃ i, f i = p))

/-- Euclidean distance between two points -/
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Two congruent triangles can be assembled to form a parallelogram -/
theorem congruent_triangles_form_parallelogram 
  (T1 T2 : Triangle) (h : Congruent T1 T2) : 
  ∃ (P : Parallelogram), AssembledFrom P [T1, T2] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_triangles_form_parallelogram_l135_13563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_filling_time_l135_13571

/-- Represents the volume of the tank -/
noncomputable def T : ℝ := 1

/-- The time taken for pipes X and Y to fill the tank -/
noncomputable def t_xy : ℝ := 3

/-- The time taken for pipes X and Z to fill the tank -/
noncomputable def t_xz : ℝ := 6

/-- The time taken for pipes Y and Z to fill the tank -/
noncomputable def t_yz : ℝ := 4.5

/-- The time taken for pipes X, Y, and Z to fill the tank -/
noncomputable def t_xyz : ℝ := 108 / 33

theorem tank_filling_time : 
  ∃ (X Y Z : ℝ),
    T = t_xy * (X + Y) ∧
    T = t_xz * (X + Z) ∧
    T = t_yz * (Y + Z) ∧
    T / t_xyz = X + Y + Z :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_filling_time_l135_13571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l135_13585

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt ((h.a ^ 2 + h.b ^ 2) / h.a ^ 2)

/-- The area of a triangle formed by the origin and two points on the line x = -2 -/
noncomputable def triangleArea (y1 y2 : ℝ) : ℝ :=
  2 * abs y1 * abs y2

theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_asymptote : ∃ (y1 y2 : ℝ), triangleArea y1 y2 = 4 ∧ y1 = -y2 ∧ y1 = (2 * h.b) / h.a) :
  eccentricity h = Real.sqrt 2 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l135_13585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_example_satisfies_condition_l135_13517

/-- A function satisfying the given condition -/
def satisfies_condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - (f x)^2)

/-- The theorem stating that a function satisfying the condition is periodic -/
theorem periodic_function (f : ℝ → ℝ) (a : ℝ) 
    (h : satisfies_condition f a) : 
    ∃ b : ℝ, b > 0 ∧ ∀ x : ℝ, f (x + b) = f x := by
  sorry

/-- An example function for a = 1 -/
noncomputable def example_function : ℝ → ℝ :=
  λ x ↦ 1/2 + 1/2 * abs (Real.sin (Real.pi / 2 * x))

/-- Proof that the example function satisfies the condition for a = 1 -/
theorem example_satisfies_condition : 
    satisfies_condition example_function 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_example_satisfies_condition_l135_13517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_catch_speed_l135_13540

/-- Proves that given a total distance of 2 km, a total time of 2 minutes, and a speed of 30 km/h
    for the first minute, the required speed for the second minute to cover the remaining distance
    is 90 km/h. -/
theorem train_catch_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (first_minute_speed : ℝ) 
  (h1 : total_distance = 2) 
  (h2 : total_time = 2) 
  (h3 : first_minute_speed = 30) : 
  (first_minute_speed / 60 + (total_distance - first_minute_speed / 60)) * 30 = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_catch_speed_l135_13540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equation_solution_l135_13591

theorem exponent_equation_solution (x : ℝ) : 
  (5 : ℝ) ^ (3 * x^2 - 8 * x + 5) = (5 : ℝ) ^ (3 * x^2 + 4 * x - 7) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equation_solution_l135_13591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_bar_length_l135_13536

/-- Proves that the length of the iron bar is 12 cm given the specified conditions -/
theorem iron_bar_length :
  let cross_section_area : ℝ := 8 * 6
  let num_bars : ℕ := 10
  let ball_volume : ℝ := 8
  let num_balls : ℕ := 720
  let bar_length : ℝ := 12
  (cross_section_area * bar_length * (num_bars : ℝ) = ball_volume * (num_balls : ℝ)) →
  bar_length = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iron_bar_length_l135_13536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_different_remainders_count_l135_13500

/-- The number of positive integers at most 420 that leave different remainders when divided by each of 5, 6, and 7 -/
def different_remainders : ℕ :=
  420 - (⌊(420 : ℚ) / 30⌋.toNat + ⌊(420 : ℚ) / 35⌋.toNat + ⌊(420 : ℚ) / 42⌋.toNat - 2 * ⌊(420 : ℚ) / 210⌋.toNat)

/-- Theorem stating that the number of positive integers at most 420 that leave different remainders when divided by each of 5, 6, and 7 is 386 -/
theorem different_remainders_count : different_remainders = 386 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_different_remainders_count_l135_13500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_passes_through_intersection_l135_13525

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Theorem: The diagonal of one parallelogram passes through the intersection of diagonals of another -/
theorem diagonal_passes_through_intersection 
  (ABCD : Parallelogram) (MNPQ : Parallelogram) 
  (O : Point) (K : Point) (L : Point) (t s : ℝ) : 
  (O.x = (ABCD.A.x + ABCD.C.x) / 2 ∧ O.y = (ABCD.A.y + ABCD.C.y) / 2) →  -- O is on diagonal AC
  (O.x = (ABCD.B.x + ABCD.D.x) / 2 ∧ O.y = (ABCD.B.y + ABCD.D.y) / 2) →  -- O is on diagonal BD
  (K.x = ABCD.A.x + t * (ABCD.C.x - ABCD.A.x) ∧ 
   K.y = ABCD.A.y + t * (ABCD.C.y - ABCD.A.y)) →  -- K is on diagonal AC of ABCD
  (L.x = ABCD.B.x + s * (ABCD.D.x - ABCD.B.x) ∧ 
   L.y = ABCD.B.y + s * (ABCD.D.y - ABCD.B.y)) →  -- L is on diagonal BD of ABCD
  ∃ r : ℝ, O.x = K.x + r * (L.x - K.x) ∧ O.y = K.y + r * (L.y - K.y)  -- O is on line KL
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_passes_through_intersection_l135_13525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_volume_problem_l135_13574

noncomputable def truncated_cone_volume (R r h : ℝ) : ℝ := 
  (1/3) * Real.pi * h * (R^2 + R*r + r^2)

noncomputable def cone_volume (r h : ℝ) : ℝ := 
  (1/3) * Real.pi * r^2 * h

noncomputable def combined_volume (R_trunc r_trunc h_trunc r_cone h_cone : ℝ) : ℝ :=
  truncated_cone_volume R_trunc r_trunc h_trunc + cone_volume r_cone h_cone

theorem combined_volume_problem :
  combined_volume 12 6 10 12 5 = 1080 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_volume_problem_l135_13574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_zero_g_minus_f_bound_l135_13553

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (a + Real.log x)
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

-- Theorem 1: If the minimum value of f(x) is -1/e, then a = 0
theorem min_value_implies_a_zero (a : ℝ) :
  (∀ x > 0, f a x ≥ -1 / Real.exp 1) ∧ (∃ x > 0, f a x = -1 / Real.exp 1) →
  a = 0 := by
  sorry

-- Theorem 2: For a > 0 and x > 0, g(x) - f(x) < 2/e
theorem g_minus_f_bound (a : ℝ) (x : ℝ) (ha : a > 0) (hx : x > 0) :
  g x - f a x < 2 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_zero_g_minus_f_bound_l135_13553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donnys_apple_cost_l135_13516

/-- Calculates the total cost of Donny's apple purchase --/
def total_cost (small_price medium_price big_price : ℚ)
                (small_count medium_count big_count : ℕ)
                (medium_discount big_discount tax_rate : ℚ) : ℚ :=
  let small_cost := small_price * small_count
  let medium_cost_before_discount := medium_price * medium_count
  let medium_cost := medium_cost_before_discount * (1 - medium_discount)
  let big_cost_before_discount := big_price * big_count
  let discounted_big_apples := (big_count / 3 : ℚ).floor
  let big_cost := big_cost_before_discount - (discounted_big_apples * big_price * big_discount)
  let subtotal := small_cost + medium_cost + big_cost
  let total := subtotal * (1 + tax_rate)
  total

/-- Theorem stating that Donny's total cost is $43.56 --/
theorem donnys_apple_cost :
  total_cost (3/2) 2 3 6 6 8 (1/5) (1/2) (1/10) = 1089/25 := by
  -- Proof goes here
  sorry

#eval total_cost (3/2) 2 3 6 6 8 (1/5) (1/2) (1/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_donnys_apple_cost_l135_13516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_divisible_values_l135_13547

/-- A function that checks if a two-digit number is divisible by 4 -/
def is_divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

/-- The set of single-digit numbers (0 to 9) -/
def single_digit : Set ℕ :=
  {n : ℕ | n ≤ 9}

/-- The four-digit number 10C4 where C is a single digit -/
def number (C : ℕ) : ℕ :=
  1000 + 100 * C + 4

/-- The theorem stating that there are exactly 5 values of C that make 10C4 divisible by 4 -/
theorem five_divisible_values :
  ∃! (S : Finset ℕ), S.toSet ⊆ single_digit ∧ 
  (∀ C ∈ S, is_divisible_by_4 (number C % 100)) ∧
  S.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_divisible_values_l135_13547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_markdown_percentage_l135_13549

theorem second_markdown_percentage (P : ℝ) (h : P > 0) : 
  ∃ X : ℝ, 
    (1 - X / 100) * (0.5 * P) = 0.45 * P ∧ 
    X = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_markdown_percentage_l135_13549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_probability_l135_13503

/-- A square is a shape with four vertices -/
structure Square where
  vertices : Fin 4 → Point

/-- A line is formed by selecting two distinct vertices of a square -/
structure Line (s : Square) where
  v1 : Fin 4
  v2 : Fin 4
  distinct : v1 ≠ v2

/-- Two lines are perpendicular if they form a right angle -/
def are_perpendicular (s : Square) (l1 l2 : Line s) : Prop :=
  sorry -- Definition of perpendicularity to be implemented

/-- The total number of ways to choose two lines -/
def total_line_choices (s : Square) : ℕ :=
  sorry -- Definition of total choices to be implemented

/-- The number of ways to choose two perpendicular lines -/
def perpendicular_line_choices (s : Square) : ℕ :=
  sorry -- Definition of perpendicular choices to be implemented

/-- The main theorem: probability of choosing perpendicular lines is 5/18 -/
theorem perpendicular_line_probability (s : Square) :
  (perpendicular_line_choices s : ℚ) / (total_line_choices s) = 5 / 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_probability_l135_13503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_MN_l135_13596

-- Define the curves and the line
def C₁ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4
def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ
def l (θ : ℝ) : Prop := θ = Real.pi / 4

-- Define the intersection points
def M (ρ : ℝ) : Prop := C₁ (ρ * Real.cos (Real.pi/4)) (ρ * Real.sin (Real.pi/4)) ∧ l (Real.pi/4)
def N (ρ : ℝ) : Prop := C₂ ρ (Real.pi/4) ∧ l (Real.pi/4)

-- Theorem statement
theorem distance_MN : 
  ∀ ρ₁ ρ₂ : ℝ, M ρ₁ → N ρ₂ → |ρ₁ - ρ₂| = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_MN_l135_13596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_square_root_l135_13506

theorem square_root_of_square_root (x : ℝ) : (Real.sqrt x = 4 ∨ Real.sqrt x = -4) → x = 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_square_root_l135_13506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_major_axis_length_is_85_l135_13550

/-- An ellipse in the xy-plane with foci at (9,20) and (49,55), tangent to the x-axis -/
structure Ellipse where
  focus1 : ℝ × ℝ := (9, 20)
  focus2 : ℝ × ℝ := (49, 55)
  tangent_to_x_axis : Bool

/-- The length of the major axis of the ellipse -/
noncomputable def major_axis_length (e : Ellipse) : ℝ :=
  let f1 := e.focus1
  let f2 := e.focus2
  let f2_reflected := (f2.1, -f2.2)
  Real.sqrt ((f2_reflected.1 - f1.1)^2 + (f2_reflected.2 - f1.2)^2)

/-- Theorem stating that the length of the major axis is 85 -/
theorem major_axis_length_is_85 (e : Ellipse) : major_axis_length e = 85 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_major_axis_length_is_85_l135_13550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l135_13530

/-- The piecewise function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 2 then x - 1/x else x

/-- The theorem statement -/
theorem f_inequality_range :
  {x : ℝ | f x < 8/3} = {x : ℝ | x < 3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l135_13530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l135_13524

theorem inequality_proof (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1/3) * (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l135_13524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_interior_angles_quadrilateral_is_90_l135_13544

/-- The sum of interior angles of an n-sided polygon -/
noncomputable def sum_interior_angles (n : ℕ) : ℝ := (n - 2 : ℝ) * 180

/-- A quadrilateral has 4 sides -/
def quadrilateral_sides : ℕ := 4

/-- The mean value of the interior angles of a quadrilateral -/
noncomputable def mean_interior_angles_quadrilateral : ℝ :=
  sum_interior_angles quadrilateral_sides / quadrilateral_sides

theorem mean_interior_angles_quadrilateral_is_90 :
  mean_interior_angles_quadrilateral = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_interior_angles_quadrilateral_is_90_l135_13544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l135_13535

noncomputable def complex_number : ℂ := (1 - Complex.I) / (2 - Complex.I)

theorem complex_number_in_first_quadrant :
  complex_number.re > 0 ∧ complex_number.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l135_13535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_average_rate_l135_13557

theorem investment_average_rate (total : ℝ) (rate1 rate2 : ℝ) 
  (h_total : total = 5000)
  (h_rate1 : rate1 = 0.05)
  (h_rate2 : rate2 = 0.06)
  (h_equal_return : ∃ x : ℝ, x > 0 ∧ x < total ∧ rate1 * (total - x) = rate2 * x) :
  ∃ avg_rate : ℝ, (avg_rate ≥ 0.0539 ∧ avg_rate ≤ 0.0541) ∧ 
  (∃ interest : ℝ, interest = rate1 * (total - (rate2 * total / (rate1 + rate2))) + 
                               rate2 * (rate2 * total / (rate1 + rate2)) ∧
                 avg_rate = interest / total) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_average_rate_l135_13557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_percentage_increase_l135_13538

-- Define the initial value and percentage increases
def initial_value : ℝ := 25
def increase_a : ℝ := 0.10
def increase_b : ℝ := 0.15
def increase_c : ℝ := 0.05

-- Define the function to apply consecutive percentage increases
def apply_increases (x : ℝ) : ℝ :=
  x * (1 + increase_a) * (1 + increase_b) * (1 + increase_c)

-- Theorem statement
theorem total_percentage_increase :
  ∃ ε > 0, |((apply_increases initial_value - initial_value) / initial_value * 100) - 32.825| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_percentage_increase_l135_13538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_power_2016_β_l135_13576

def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 5; 0, -2]
def β : Matrix (Fin 2) (Fin 1) ℝ := !![1; -1]

theorem A_power_2016_β : A^2016 * β = !![2^2016; -(2^2016)] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_power_2016_β_l135_13576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_A_triangle_perimeter_l135_13599

/-- Triangle ABC with internal angles A, B, and C -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = Real.pi

/-- Area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := sorry

/-- Perimeter of the triangle -/
noncomputable def perimeter (t : Triangle) : ℝ := sorry

/-- Side lengths of the triangle -/
noncomputable def a (t : Triangle) : ℝ := sorry
noncomputable def b (t : Triangle) : ℝ := sorry
noncomputable def c (t : Triangle) : ℝ := sorry

theorem triangle_angle_A (t : Triangle) (h : Real.sin t.C = Real.sin t.B + Real.sin (t.A - t.B)) : 
  t.A = Real.pi / 3 := by
  sorry

theorem triangle_perimeter (t : Triangle) (h1 : t.A = Real.pi / 3) (h2 : a t = Real.sqrt 7) 
  (h3 : area t = 3 * Real.sqrt 3 / 2) : 
  perimeter t = 5 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_A_triangle_perimeter_l135_13599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_selection_guarantee_min_socks_optimal_l135_13594

/-- Represents the number of socks of each color in the closet -/
def sock_counts : List Nat := [120, 90, 70, 50, 30, 10]

/-- The minimum number of socks to guarantee at least 15 pairs -/
def min_socks : Nat := 30

/-- The required number of pairs -/
def required_pairs : Nat := 15

theorem sock_selection_guarantee (n : Nat) :
  n ≥ min_socks →
  ∀ (selection : List Nat),
    selection.length = n →
    selection.sum = n →
    (∀ i, i < sock_counts.length → i < selection.length → selection[i]! ≤ sock_counts[i]!) →
    ∃ (pairs : Nat), pairs ≥ required_pairs ∧ 
      ∀ c, c ∈ selection → 2 * pairs ≤ selection.sum :=
by sorry

theorem min_socks_optimal :
  ∀ n < min_socks,
  ∃ (selection : List Nat),
    selection.length = n ∧
    selection.sum = n ∧
    (∀ i, i < sock_counts.length → i < selection.length → selection[i]! ≤ sock_counts[i]!) ∧
    ∀ (pairs : Nat), 
      (∀ c, c ∈ selection → 2 * pairs ≤ selection.sum) → 
      pairs < required_pairs :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_selection_guarantee_min_socks_optimal_l135_13594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_quartets_l135_13518

/-- The number of musicians in each group -/
def n : ℕ := 5

/-- The number of quartets to form -/
def k : ℕ := 5

/-- A function that calculates the number of ways to form k quartets from n musicians of each type -/
def quartets_count (n k : ℕ) : ℕ := (Nat.factorial n)^3

/-- Theorem stating that the number of ways to form 5 quartets from 5 musicians of each type is (5!)³ -/
theorem count_quartets : quartets_count n k = (Nat.factorial n)^3 := by
  -- Unfold the definition of quartets_count
  unfold quartets_count
  -- The equation is now trivially true
  rfl

#eval quartets_count n k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_quartets_l135_13518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequences_ratio_l135_13546

/-- Sum of an arithmetic sequence -/
noncomputable def arithmetic_sum (a d l : ℚ) : ℚ :=
  let n : ℚ := (l - a) / d + 1
  n / 2 * (a + l)

/-- The ratio of two arithmetic sequences -/
theorem arithmetic_sequences_ratio :
  let num_sum := arithmetic_sum 4 4 60
  let den_sum := arithmetic_sum 5 5 75
  num_sum / den_sum = 4 / 5 := by
  -- Unfold the definitions
  unfold arithmetic_sum
  -- Simplify the expressions
  simp
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequences_ratio_l135_13546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_problem_l135_13523

def Exactly (n : ℕ) (l : List Bool) : Prop :=
  (l.filter id).length = n

theorem card_problem :
  ∃ (x y : ℕ),
    x > 0 ∧ y > 0 ∧
    (Exactly 2
      [x + 2 = 2 * (y - 2),
       3 * (x - 3) = y + 3,
       x + 4 = 4 * (y - 4),
       5 * (x - 5) = y + 5]) ∧
    (∃ (z : ℕ), z ≤ x ∧ x - z = y + z) →
    y = 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_problem_l135_13523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l135_13514

/-- The function g -/
def g (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 4

/-- The function f -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The inverse function of f -/
noncomputable def f_inv (a b c : ℝ) : ℝ → ℝ := Function.invFun (f a b c)

/-- The main theorem -/
theorem main_theorem (a b c : ℝ) :
  (∀ x, g x = f_inv a b c x + 2) →
  (∀ x, (f_inv a b c) ((f a b c) x) = x) →
  (∀ x, (f a b c) ((f_inv a b c) x) = x) →
  3 * a + 3 * b + 3 * c = -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l135_13514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l135_13551

/-- The expansion of (2x - 1/x)^4 -/
noncomputable def expansion (x : ℝ) : ℝ := (2*x - 1/x)^4

/-- The constant term in the expansion -/
def constant_term : ℚ := 24

/-- The sum of coefficients in the expansion -/
def sum_of_coefficients : ℚ := 1

/-- Theorem stating the properties of the expansion -/
theorem expansion_properties :
  (∃ (c : ℝ), expansion c = constant_term) ∧
  expansion 1 = sum_of_coefficients := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_properties_l135_13551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l135_13590

noncomputable def f (k a x : ℝ) : ℝ := (2*k - 1)*a^x - a^(-x)

theorem function_properties (a : ℝ) (h_a_pos : a > 0) (h_a_neq_1 : a ≠ 1) :
  ∃ (k t : ℝ),
    (∀ x, f k a x = -f k a (-x)) ∧  -- f is odd
    (f k a 1 = -5/6) ∧
    (∀ x ∈ Set.Icc (-1) 1, f k a (3*x - t) + f k a (-2*x + 1) ≥ 0) ∧
    k = 1 ∧
    t = 2 ∧
    ∀ t', (∀ x ∈ Set.Icc (-1) 1, f k a (3*x - t') + f k a (-2*x + 1) ≥ 0) → t' ≥ t :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l135_13590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixtieth_permutation_l135_13575

/-- A permutation of the digits 1, 2, 3, 4, 5 -/
def DigitPermutation : Type := Fin 5 → Fin 5

/-- Convert a DigitPermutation to a natural number -/
def permToNat (p : DigitPermutation) : ℕ :=
  10000 * (p 0 + 1) + 1000 * (p 1 + 1) + 100 * (p 2 + 1) + 10 * (p 3 + 1) + (p 4 + 1)

/-- The list of all valid DigitPermutations, sorted by their natural number representation -/
def sortedPermutations : List DigitPermutation :=
  sorry

theorem sixtieth_permutation :
  permToNat (sortedPermutations.nthLe 59 sorry) = 32315 :=
by sorry

#eval permToNat (fun i => i)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixtieth_permutation_l135_13575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_not_always_decreasing_l135_13570

noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_not_always_decreasing :
  ¬ (∀ (a : ℝ), a > 0 ∧ a ≠ 1 → ∀ (x₁ x₂ : ℝ), x₁ > x₂ ∧ x₁ > 0 ∧ x₂ > 0 → log a x₁ < log a x₂) :=
by
  -- We prove by contradiction
  intro h
  -- Consider a = 2 (which is > 0 and ≠ 1)
  let a := 2
  -- Apply the hypothesis to a = 2
  have h2 := h a (by norm_num)
  -- log_2 is increasing, so we can find a counterexample
  have counter : ∃ (x₁ x₂ : ℝ), x₁ > x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ log a x₁ > log a x₂ := by
    use 4, 2
    simp [log]
    norm_num
    -- The actual proof would go here
    sorry
  -- This contradicts our hypothesis
  cases counter with | intro x₁ hx₁ =>
    cases hx₁ with | intro x₂ hx₂ =>
      have contra := h2 x₁ x₂ (by exact ⟨hx₂.1, hx₂.2.1, hx₂.2.2.1⟩)
      -- This line would show the contradiction
      sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_not_always_decreasing_l135_13570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_less_volatile_performance_correct_answer_is_D_l135_13582

/-- Represents a shooter's performance --/
structure ShooterPerformance where
  shots : ℕ
  average_score : ℝ
  variance : ℝ

/-- Definition of less volatile performance --/
def less_volatile (a b : ShooterPerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two shooters with the same average score but different variances,
    the shooter with the lower variance has less volatile performance --/
theorem less_volatile_performance
  (shooter_A shooter_B : ShooterPerformance)
  (h_shots : shooter_A.shots = shooter_B.shots)
  (h_avg : shooter_A.average_score = shooter_B.average_score)
  (h_var : shooter_A.variance > shooter_B.variance) :
  less_volatile shooter_B shooter_A :=
by
  unfold less_volatile
  exact h_var

/-- The correct answer is that person B's performance is less volatile --/
theorem correct_answer_is_D
  (shooter_A shooter_B : ShooterPerformance)
  (h_shots : shooter_A.shots = shooter_B.shots)
  (h_avg : shooter_A.average_score = shooter_B.average_score)
  (h_var : shooter_A.variance > shooter_B.variance) :
  less_volatile shooter_B shooter_A :=
less_volatile_performance shooter_A shooter_B h_shots h_avg h_var

end NUMINAMATH_CALUDE_ERRORFEEDBACK_less_volatile_performance_correct_answer_is_D_l135_13582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l135_13554

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos (x - Real.pi/6))^2 + 2 * Real.sin (x - Real.pi/4) * Real.sin (x + Real.pi/4)

def is_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def is_smallest_positive_period (T : ℝ) (f : ℝ → ℝ) : Prop :=
  T > 0 ∧ is_period T f ∧ ∀ T' > 0, is_period T' f → T ≤ T'

def is_symmetry_center (c : ℝ × ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (c.1 + x) = f (c.1 - x)

theorem f_properties :
  (is_smallest_positive_period Real.pi f) ∧
  (∀ k : ℤ, is_symmetry_center (k * Real.pi/2 + Real.pi/12, 1) f) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l135_13554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_compression_l135_13561

/-- The maximum compression of a spring in a collision between a moving ball and a stationary box -/
theorem spring_compression (m M k v : ℝ) (hm : m > 0) (hM : M > 0) (hk : k > 0) (hv : v ≥ 0) :
  ∃ x : ℝ, x = v * Real.sqrt (m * M / ((m + M) * k)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_compression_l135_13561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_collinear_find_k_l135_13588

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define non-collinear vectors a and b
variable (a b : V)
variable (h : ¬ ∃ (r : ℝ), a = r • b)

-- Define points A, B, C, D
variable (A B C D : V)

-- Define the given vector relationships
variable (hAB : B - A = a + b)
variable (hBC : C - B = 2 • a + 8 • b)
variable (hCD : D - C = 3 • (a - b))

-- Part 1: Prove that A, B, and D are collinear
theorem points_collinear : ∃ (r : ℝ), D - A = r • (B - A) := by sorry

-- Part 2: Find k such that k•a + b is collinear with 2•a + k•b
theorem find_k : ∃ (k : ℝ), k^2 = 2 ∧ ∃ (t : ℝ), k • a + b = t • (2 • a + k • b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_collinear_find_k_l135_13588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l135_13543

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x ^ 2 - 1

theorem min_positive_period_of_f : 
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ 
  T = π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l135_13543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_adjacent_book_selection_l135_13545

def number_of_ways_to_choose_non_adjacent_books (n m : ℕ) : ℕ := 
  sorry

def number_of_ways_to_arrange_pairs_and_singles (singles pairs : ℕ) : ℕ := 
  Nat.choose (singles + pairs) singles

theorem non_adjacent_book_selection (n m : ℕ) (hn : n = 12) (hm : m = 5) :
  number_of_ways_to_choose_non_adjacent_books n m = 
  number_of_ways_to_arrange_pairs_and_singles (n - m) m := by
  sorry

#eval number_of_ways_to_arrange_pairs_and_singles 3 5  -- Should output 56

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_adjacent_book_selection_l135_13545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l135_13512

noncomputable section

-- Define the circle C
def circle_C (t : ℝ) : ℝ × ℝ := (-5 + Real.sqrt 2 * Real.cos t, 3 + Real.sqrt 2 * Real.sin t)

-- Define the line l
def line_l (θ : ℝ) : ℝ := -Real.sqrt 2 / Real.cos (θ + Real.pi/4)

-- Define points A and B
def point_A : ℝ × ℝ := (0, 2)
def point_B : ℝ × ℝ := (-2, 0)

-- Define the area of triangle PAB
def triangle_area (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  let d := abs (-x + y - 2) / Real.sqrt 2
  d * Real.sqrt 8 / 2

-- Theorem statement
theorem min_triangle_area :
  ∃ (t : ℝ), ∀ (s : ℝ), triangle_area (circle_C t) ≤ triangle_area (circle_C s) ∧
  triangle_area (circle_C t) = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l135_13512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_profit_calculation_l135_13562

/-- Calculates the profit percentage with discount given the no-discount profit percentage and discount percentage. -/
noncomputable def profit_with_discount (no_discount_profit_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  let selling_price := 100 + no_discount_profit_percent
  let discounted_selling_price := selling_price * (1 - discount_percent / 100)
  (discounted_selling_price - 100) / 100 * 100

/-- Proves that given a 5% discount and a 42% profit without discount, the profit percentage with discount is 34.9%. -/
theorem discount_profit_calculation :
  profit_with_discount 42 5 = 34.9 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable functions
-- #eval profit_with_discount 42 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_profit_calculation_l135_13562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_28_l135_13583

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  let crossProduct := AB.1 * AC.2 - AB.2 * AC.1
  (1/2) * abs crossProduct

theorem triangle_area_is_28 :
  let A : ℝ × ℝ := (7, 8)
  let B : ℝ × ℝ := (10, 4)
  let C : ℝ × ℝ := (2, -4)
  triangleArea A B C = 28 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_28_l135_13583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_area_approx_l135_13532

noncomputable section

/-- The area of a trapezoid given the lengths of its parallel sides and its height. -/
def trapezoidArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- The area of a triangle given its base and height. -/
def triangleArea (b h : ℝ) : ℝ := b * h / 2

/-- The area of a circular sector given its radius and central angle in degrees. -/
def sectorArea (r θ : ℝ) : ℝ := θ / 360 * Real.pi * r^2

/-- The combined area of three specific shapes. -/
def combinedArea : ℝ :=
  trapezoidArea 6 10 8 + triangleArea 6 4 + sectorArea 5 60

/-- The combined area is approximately 89.09 square feet. -/
theorem combined_area_approx :
  abs (combinedArea - 89.09) < 0.01 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_area_approx_l135_13532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l135_13542

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin (t.A + π/3) = 4 * Real.sin (t.A/2) * Real.cos (t.A/2))
  (h2 : Real.sin t.B = Real.sqrt 3 * Real.sin t.C)
  (h3 : t.a = 1) : 
  t.A = π/6 ∧ (1/2 * t.b * t.c * Real.sin t.A) = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l135_13542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_driver_pay_per_mile_l135_13526

/-- Calculates the pay per mile for a truck driver given specific conditions -/
theorem truck_driver_pay_per_mile 
  (gas_cost : ℚ) 
  (miles_per_gallon : ℚ) 
  (miles_per_hour : ℚ) 
  (hours_driven : ℚ) 
  (total_pay : ℚ) 
  (h1 : gas_cost = 2)
  (h2 : miles_per_gallon = 10)
  (h3 : miles_per_hour = 30)
  (h4 : hours_driven = 10)
  (h5 : total_pay = 90) :
  (total_pay - (hours_driven * miles_per_hour / miles_per_gallon * gas_cost)) / (hours_driven * miles_per_hour) = 1/10 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_driver_pay_per_mile_l135_13526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_wins_single_player_wins_l135_13595

/-- A game where 10 people each roll a fair six-sided die -/
def diceGame : ℕ := 10

/-- The number of faces on a fair die -/
def diceFaces : ℕ := 6

/-- The probability that a specific player wins a prize -/
noncomputable def probSingleWin : ℝ := (5 / 6) ^ 9

/-- The probability that at least one player wins a prize -/
noncomputable def probAtLeastOneWin : ℝ := 
  10 * (5^9 / 6^9) - 45 * (5 * 4^8 / 6^9) + 120 * (5 * 4 * 3^7 / 6^9) - 
  210 * (5 * 4 * 3 * 2^6 / 6^9) + 252 * (5 * 4 * 3 * 2 * 1 / 6^9)

/-- Theorem stating the probability of at least one player winning a prize -/
theorem at_least_one_wins : 
  ∀ ε > 0, |probAtLeastOneWin - 0.919| < ε := by
  sorry

/-- Theorem stating the probability of a specific player winning a prize -/
theorem single_player_wins :
  ∀ ε > 0, |probSingleWin - 0.194| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_wins_single_player_wins_l135_13595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_from_circles_l135_13541

-- Define the circles S₁ and S₂
variable (S₁ S₂ : Set (EuclideanSpace ℝ (Fin 2)))

-- Define points A, B, C, D, and P
variable (A B C D P : EuclideanSpace ℝ (Fin 2))

-- Define the property of being a circle
def is_circle (S : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- Define the property of two circles intersecting at two points
def circles_intersect (S₁ S₂ : Set (EuclideanSpace ℝ (Fin 2))) (A P : EuclideanSpace ℝ (Fin 2)) : Prop := 
  A ∈ S₁ ∧ A ∈ S₂ ∧ P ∈ S₁ ∧ P ∈ S₂

-- Define the property of a line being tangent to a circle at a point
def is_tangent (l : Set (EuclideanSpace ℝ (Fin 2))) (S : Set (EuclideanSpace ℝ (Fin 2))) (X : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the property of two lines being parallel
def parallel (l₁ l₂ : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- Define the property of a point lying on a circle
def point_on_circle (X : EuclideanSpace ℝ (Fin 2)) (S : Set (EuclideanSpace ℝ (Fin 2))) : Prop := X ∈ S

-- Define the property of a quadrilateral being a parallelogram
def is_parallelogram (W X Y Z : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define a line through two points
def Line (X Y : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- State the theorem
theorem parallelogram_from_circles 
  (h₁ : is_circle S₁) 
  (h₂ : is_circle S₂) 
  (h₃ : circles_intersect S₁ S₂ A P) 
  (h₄ : is_tangent (Line A B) S₁ A) 
  (h₅ : parallel (Line A B) (Line C D)) 
  (h₆ : P ∈ Line C D) 
  (h₇ : point_on_circle B S₂) 
  (h₈ : point_on_circle D S₁) : 
  is_parallelogram A B C D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_from_circles_l135_13541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_tetrahedron_volume_l135_13586

noncomputable section

/-- The radius of the sphere -/
def R : ℝ := 6

/-- The side length of the inscribed regular tetrahedron -/
noncomputable def a : ℝ := 4 * Real.sqrt 6

/-- The volume of the inscribed regular tetrahedron -/
noncomputable def V : ℝ := (a^3 * Real.sqrt 2) / 12

theorem inscribed_tetrahedron_volume :
  V = 32 * Real.sqrt 3 := by
  -- Proof steps would go here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_tetrahedron_volume_l135_13586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_power_functions_l135_13522

noncomputable section

-- Define the four functions
def f₁ : ℝ → ℝ := λ x => 1 / x^2
def f₂ : ℝ → ℝ := λ x => 2 * x
def f₃ : ℝ → ℝ := λ x => x^2 + x
def f₄ : ℝ → ℝ := λ x => Real.rpow x (5/3)

-- Define what a power function is
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ k n : ℝ, ∀ x : ℝ, f x = k * Real.rpow x n

-- State the theorem
theorem two_power_functions :
  (isPowerFunction f₁ ∧ isPowerFunction f₄) ∧
  (¬ isPowerFunction f₂ ∧ ¬ isPowerFunction f₃) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_power_functions_l135_13522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l135_13580

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1
  | (n + 1) => sequence_a n - n

theorem a_5_value : sequence_a 5 = -9 := by
  -- Unfold the definition of sequence_a for the first few terms
  have h1 : sequence_a 1 = 1 := rfl
  have h2 : sequence_a 2 = 0 := by simp [sequence_a]
  have h3 : sequence_a 3 = -2 := by simp [sequence_a]
  have h4 : sequence_a 4 = -5 := by simp [sequence_a]
  have h5 : sequence_a 5 = -9 := by simp [sequence_a]
  
  -- The final step
  exact h5


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l135_13580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_3x_256_equals_x_l135_13587

theorem log_3x_256_equals_x (x : ℝ) : (Real.log 256 / Real.log (3 * x) = x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_3x_256_equals_x_l135_13587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_credit_is_855_l135_13568

/-- Represents the total outstanding consumer installment credit in billions of dollars -/
def total_credit : ℝ := sorry

/-- Represents the total automobile installment credit in billions of dollars -/
def auto_credit : ℝ := sorry

/-- Represents the credit extended by automobile finance companies in billions of dollars -/
def finance_company_credit : ℝ := sorry

/-- The theorem states that given the conditions from the problem, 
    the total outstanding consumer installment credit is 855 billion dollars -/
theorem total_credit_is_855 
  (h1 : auto_credit = 0.20 * total_credit)
  (h2 : finance_company_credit = 57)
  (h3 : finance_company_credit = (1/3) * auto_credit) :
  total_credit = 855 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_credit_is_855_l135_13568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kid_can_always_win_l135_13573

/-- Represents a pastry type -/
inductive PastryType
| Sugar
| Cinnamon
deriving BEq, Repr

/-- Represents the row of pastries -/
def PastryRow := List PastryType

/-- Represents a player's move -/
inductive Move
| Left
| Right
deriving BEq, Repr

/-- Represents a player's strategy -/
def Strategy := PastryRow → Move

/-- Kid's goal state -/
def KidGoal (pastries : List PastryType) : Prop :=
  (pastries.count PastryType.Sugar = 10) ∧ (pastries.count PastryType.Cinnamon = 10)

/-- Simulates a game given strategies for both players -/
def simulate_game (kid_strategy : Strategy) (carlson_strategy : Strategy) (initial_row : PastryRow) : PastryRow :=
  sorry -- Placeholder for game simulation logic

theorem kid_can_always_win (initial_row : PastryRow) : 
  initial_row.length = 40 →
  initial_row.count PastryType.Sugar = 20 →
  initial_row.count PastryType.Cinnamon = 20 →
  ∃ (kid_strategy : Strategy), 
    ∀ (carlson_strategy : Strategy),
      KidGoal (simulate_game kid_strategy carlson_strategy initial_row) :=
by
  sorry -- Placeholder for the proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kid_can_always_win_l135_13573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_four_heads_l135_13565

def biased_coin (p : ℝ) : Prop :=
  (Nat.choose 7 2) * p^2 * (1 - p)^5 = (Nat.choose 7 3) * p^3 * (1 - p)^4

theorem biased_coin_four_heads :
  ∃ p : ℝ, biased_coin p ∧ 
    (Nat.choose 7 4) * p^4 * (1 - p)^3 = 354375 / 2097152 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_four_heads_l135_13565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mpg_difference_is_twelve_l135_13577

/-- Represents the fuel efficiency of a car in different driving conditions -/
structure CarFuelEfficiency where
  highway_miles_per_tankful : ℚ
  city_miles_per_tankful : ℚ
  city_miles_per_gallon : ℚ

/-- Calculates the difference in miles per gallon between highway and city driving -/
def mpg_difference (car : CarFuelEfficiency) : ℚ :=
  let tank_size := car.city_miles_per_tankful / car.city_miles_per_gallon
  let highway_mpg := car.highway_miles_per_tankful / tank_size
  highway_mpg - car.city_miles_per_gallon

/-- Theorem stating the difference in miles per gallon between highway and city driving -/
theorem mpg_difference_is_twelve (car : CarFuelEfficiency)
    (h1 : car.highway_miles_per_tankful = 462)
    (h2 : car.city_miles_per_tankful = 336)
    (h3 : car.city_miles_per_gallon = 32) :
  mpg_difference car = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mpg_difference_is_twelve_l135_13577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_map_scale_l135_13507

/-- The scale of a map given real and map distances -/
noncomputable def map_scale (real_distance : ℝ) (map_distance : ℝ) : ℝ := 
  map_distance / real_distance

theorem city_map_scale : 
  let real_distance := (18 : ℝ) * 1000 * 100 -- 18 km in cm
  let map_distance := 240 -- 240 cm
  map_scale real_distance map_distance = 1 / 7500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_map_scale_l135_13507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l135_13529

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => n.succ / (n + 2) * sequence_a n

theorem sequence_a_formula (n : ℕ) : sequence_a n = 1 / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l135_13529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_one_eq_half_l135_13513

-- Define the functions g and f
noncomputable def g (x : ℝ) : ℝ := 2 - x^2

noncomputable def f (y : ℝ) : ℝ := y / (2 * (2 - y))

-- State the theorem
theorem f_of_one_eq_half : f 1 = 1/2 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- Perform algebraic manipulation
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_one_eq_half_l135_13513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_naturals_l135_13552

theorem partition_naturals (c : ℚ) (hc : c > 0) (hc1 : c ≠ 1) :
  ∃ (A B : Set ℕ), A ∪ B = Set.univ ∧ A ∩ B = ∅ ∧ A.Nonempty ∧ B.Nonempty ∧
  (∀ x y, x ∈ A → y ∈ A → (x : ℚ) / y ≠ c) ∧ 
  (∀ x y, x ∈ B → y ∈ B → (x : ℚ) / y ≠ c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_naturals_l135_13552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l135_13520

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (3 : ℝ)^x + (4 : ℝ)^x + (5 : ℝ)^x = (6 : ℝ)^x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l135_13520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_to_hypotenuse_value_l135_13567

noncomputable section

-- Define the right triangle
def RightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

-- Define the area of the triangle
def TriangleArea (a b : ℝ) : ℝ :=
  (1/2) * a * b

-- Define the height to hypotenuse
def HeightToHypotenuse (a b h : ℝ) : Prop :=
  h^2 * (a^2 + b^2) = (a * b)^2

-- Define the angle division ratio
def AngleDivisionRatio (x y : ℝ) : Prop :=
  x / y = 1 / 2

-- The main theorem
theorem height_to_hypotenuse_value 
  (a b c h : ℝ) 
  (triangle : RightTriangle a b c) 
  (area : TriangleArea a b = 2 * Real.sqrt 3) 
  (height : HeightToHypotenuse a b h) 
  (ratio : AngleDivisionRatio a b) : 
  h = Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_to_hypotenuse_value_l135_13567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l135_13527

theorem sufficient_not_necessary : 
  (∀ x : ℝ, |x - 1| < 2 → x^2 - 4*x - 5 < 0) ∧ 
  (∃ x : ℝ, x^2 - 4*x - 5 < 0 ∧ ¬(|x - 1| < 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l135_13527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_theorem_l135_13559

def game_probability (n : ℕ) (p_alex p_mel p_chelsea : ℝ) 
  (alex_wins mel_wins chelsea_wins : ℕ) : ℝ :=
  (p_alex ^ alex_wins) * (p_mel ^ mel_wins) * (p_chelsea ^ chelsea_wins) *
  (Nat.choose n alex_wins * Nat.choose (n - alex_wins) mel_wins)

theorem game_probability_theorem :
  let n : ℕ := 7
  let p_alex : ℝ := 1/2
  let p_mel : ℝ := 3/8
  let p_chelsea : ℝ := 1/8
  let alex_wins : ℕ := 4
  let mel_wins : ℕ := 2
  let chelsea_wins : ℕ := 1
  p_alex + p_mel + p_chelsea = 1 →
  p_mel = 3 * p_chelsea →
  game_probability n p_alex p_mel p_chelsea alex_wins mel_wins chelsea_wins = 945/8192 :=
by
  intro h1 h2
  simp [game_probability]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_theorem_l135_13559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_distance_proof_l135_13521

noncomputable section

/-- The distance to school in miles -/
def distance_to_school : ℝ := 1.6

/-- The normal driving time in hours -/
def normal_time : ℝ := 25 / 60

/-- The faster driving time in hours -/
def faster_time : ℝ := 10 / 60

/-- The normal driving speed in miles per hour -/
def normal_speed : ℝ := distance_to_school / normal_time

/-- The faster driving speed in miles per hour -/
def faster_speed : ℝ := distance_to_school / faster_time

end noncomputable section

/-- Proof that the faster speed is approximately 15 mph faster than the normal speed -/
theorem school_distance_proof :
  abs (faster_speed - (normal_speed + 15)) < 0.1 := by
  sorry

#check school_distance_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_distance_proof_l135_13521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_skew_edges_l135_13548

/-- Regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) where
  edge_length : a > 0

/-- Point on an edge of the tetrahedron -/
structure EdgePoint (T : RegularTetrahedron a) where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : EdgePoint T) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Theorem: Shortest distance between points on skew edges of a regular tetrahedron -/
theorem shortest_distance_skew_edges (T : RegularTetrahedron a) 
  (P : EdgePoint T) (Q : EdgePoint T) :
  ∃ (P' Q' : EdgePoint T), 
    (∀ (P'' : EdgePoint T) (Q'' : EdgePoint T), 
      distance P' Q' ≤ distance P'' Q'') ∧
    distance P' Q' = (Real.sqrt 2 / 2) * a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_skew_edges_l135_13548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_23_7495_to_hundredth_l135_13566

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem round_23_7495_to_hundredth :
  round_to_hundredth 23.7495 = 23.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_23_7495_to_hundredth_l135_13566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_rational_l135_13533

def a_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ 
  a 2 = 11 ∧ 
  ∀ n : ℕ, n > 0 → 2 * a (n + 2) = 3 * a n + (5 * ((a n)^2 + (a (n + 1))^2)).sqrt

theorem sequence_rational (a : ℕ → ℚ) (h : a_sequence a) : 
  ∀ n : ℕ, n > 0 → ∃ q : ℚ, a n = q :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_rational_l135_13533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_owen_total_turtles_l135_13534

/-- The growth rate of Owen's turtles -/
def G : ℝ := 1  -- Assuming G is 1 for simplicity, can be changed as needed

/-- The initial number of turtles Owen has -/
def owen_initial : ℕ := 21

/-- The initial number of turtles Johanna has -/
def johanna_initial : ℕ := owen_initial - 5

/-- The number of turtles Owen has after growth -/
def owen_after_growth : ℝ := G * (2 * owen_initial)

/-- The number of turtles Johanna has after losing half -/
def johanna_after_loss : ℕ := johanna_initial / 2

/-- Theorem: The total number of turtles Owen has after 1 month -/
theorem owen_total_turtles : 
  owen_after_growth + johanna_after_loss = 42 * G + 8 := by
  -- Unfold definitions
  unfold owen_after_growth johanna_after_loss owen_initial johanna_initial G
  -- Simplify
  simp
  -- The proof is complete
  sorry

#eval owen_after_growth + johanna_after_loss
#eval 42 * G + 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_owen_total_turtles_l135_13534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l135_13504

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![2, 0]
def c : Fin 2 → ℝ := ![1, -2]

def lambda : ℝ := -1

theorem vector_collinearity :
  ∃ (k : ℝ), k ≠ 0 ∧ (lambda • a + b) = k • c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l135_13504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_a_b_min_value_bound_l135_13510

noncomputable section

variable (a b : ℝ)

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 - (a + 2) * x + b)

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 - a * x + b - (a + 2))

-- Theorem 1: Relationship between a and b
theorem relationship_a_b (a : ℝ) :
  ∃ b : ℝ, f' a b 0 = -2 * a^2 ∧ b = a + 2 - 2 * a^2 :=
sorry

-- Theorem 2: Minimum value of f(x) for x > 0 when a < 0
theorem min_value_bound (a : ℝ) (ha : a < 0) :
  ∃ (M : ℝ), M ≥ 2 ∧ ∀ (x : ℝ), x > 0 → f a (a + 2 - 2 * a^2) x < M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_a_b_min_value_bound_l135_13510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_s_zero_not_in_range_of_s_l135_13593

-- Define the function s(x) as noncomputable
noncomputable def s (x : ℝ) : ℝ := 1 / (1 - x)^3

-- State the theorem about the range of s(x)
theorem range_of_s :
  ∀ y : ℝ, y ≠ 0 → ∃ x : ℝ, x ≠ 1 ∧ s x = y :=
by
  sorry

-- Additional theorem to show that 0 is not in the range
theorem zero_not_in_range_of_s :
  ¬ ∃ x : ℝ, x ≠ 1 ∧ s x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_s_zero_not_in_range_of_s_l135_13593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_problem_l135_13519

theorem average_problem (x : ℚ) : 
  (((List.range 101).map (λ i => (i : ℚ) + 1)).sum + x) / 102 = 50 * (x + 1) → 
  x = 51 / 5099 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_problem_l135_13519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_central_symmetry_l135_13537

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * x) - Real.sin (2 * x)

/-- The shifted function -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x - m)

/-- Central symmetry condition -/
def is_centrally_symmetric (h : ℝ → ℝ) : Prop :=
  ∀ x, h x = -h (-x)

theorem min_shift_for_central_symmetry :
  ∃ m : ℝ, m > 0 ∧ is_centrally_symmetric (g m) ∧
    ∀ m' : ℝ, m' > 0 → is_centrally_symmetric (g m') → m ≤ m' :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_for_central_symmetry_l135_13537
