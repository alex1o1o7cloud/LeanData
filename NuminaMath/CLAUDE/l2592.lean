import Mathlib

namespace NUMINAMATH_CALUDE_a_plus_b_equals_neg_nine_l2592_259223

def f (a b x : ℝ) : ℝ := a * x - b

def g (x : ℝ) : ℝ := -4 * x - 1

def h (a b x : ℝ) : ℝ := f a b (g x)

def h_inv (x : ℝ) : ℝ := x + 9

theorem a_plus_b_equals_neg_nine (a b : ℝ) :
  (∀ x, h a b x = h_inv⁻¹ x) → a + b = -9 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_neg_nine_l2592_259223


namespace NUMINAMATH_CALUDE_f_is_even_l2592_259274

-- Define g as an even function
def g_even (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  |g (x^3)|

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : g_even g) : 
  ∀ x : ℝ, f g (-x) = f g x :=
sorry

end NUMINAMATH_CALUDE_f_is_even_l2592_259274


namespace NUMINAMATH_CALUDE_square_on_hypotenuse_l2592_259278

theorem square_on_hypotenuse (a b : ℝ) (ha : a = 9) (hb : b = 12) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a * b) / (a + b)
  s = 120 / 37 := by sorry

end NUMINAMATH_CALUDE_square_on_hypotenuse_l2592_259278


namespace NUMINAMATH_CALUDE_photo_frame_border_area_l2592_259267

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

end NUMINAMATH_CALUDE_photo_frame_border_area_l2592_259267


namespace NUMINAMATH_CALUDE_g_range_l2592_259290

noncomputable def g (x : ℝ) : ℝ := 
  (Real.cos x ^ 3 + 7 * Real.cos x ^ 2 + 2 * Real.cos x + 3 * Real.sin x ^ 2 - 14) / (Real.cos x - 2)

theorem g_range : 
  ∀ x : ℝ, Real.cos x ≠ 2 → 
  (∃ y ∈ Set.Icc (1/2 : ℝ) (25/2 : ℝ), g x = y) ∧ 
  (∀ y : ℝ, g x = y → y ∈ Set.Icc (1/2 : ℝ) (25/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_g_range_l2592_259290


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l2592_259299

theorem complex_sum_of_parts (a b : ℝ) : (Complex.I * (1 - Complex.I) = Complex.mk a b) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l2592_259299


namespace NUMINAMATH_CALUDE_two_less_than_negative_one_l2592_259295

theorem two_less_than_negative_one : (- 1) - 2 = - 3 := by
  sorry

end NUMINAMATH_CALUDE_two_less_than_negative_one_l2592_259295


namespace NUMINAMATH_CALUDE_team_E_not_played_B_l2592_259291

/-- Represents a soccer team in the tournament -/
inductive Team : Type
| A | B | C | D | E | F

/-- Represents the number of matches played by each team -/
def matches_played : Team → ℕ
| Team.A => 5
| Team.B => 4
| Team.C => 3
| Team.D => 2
| Team.E => 1
| Team.F => 0  -- We don't know F's matches, so we set it to 0

/-- Theorem stating that team E has not played against team B -/
theorem team_E_not_played_B :
  ∀ (t : Team), matches_played Team.E = 1 → matches_played Team.B = 4 →
  matches_played Team.A = 5 → t ≠ Team.B → t ≠ Team.E → 
  ∃ (opponent : Team), opponent ≠ Team.E ∧ opponent ≠ Team.B :=
by sorry

end NUMINAMATH_CALUDE_team_E_not_played_B_l2592_259291


namespace NUMINAMATH_CALUDE_line_passes_through_parabola_vertex_l2592_259239

/-- The number of values of a for which the line y = ax + a passes through the vertex of the parabola y = x^2 + ax -/
theorem line_passes_through_parabola_vertex : 
  ∃! (s : Finset ℝ), (∀ a ∈ s, ∃ x y : ℝ, 
    (y = a*x + a) ∧ 
    (y = x^2 + a*x) ∧ 
    (∀ x' y' : ℝ, y' = x'^2 + a*x' → y' ≥ y)) ∧ 
  Finset.card s = 2 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_parabola_vertex_l2592_259239


namespace NUMINAMATH_CALUDE_f_properties_l2592_259264

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / (Real.exp x + Real.exp (-x))

theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x1 x2 : ℝ, x2 > x1 → f x2 > f x1) ∧
  (∀ x t : ℝ, x ∈ Set.Icc 1 2 → (f (x - t) + f (x^2 - t^2) ≥ 0 ↔ t ∈ Set.Icc (-2) 1)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2592_259264


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_min_eccentricity_l2592_259215

/-- Given an ellipse and a hyperbola with the same foci, prove the minimum value of 3e₁² + e₂² -/
theorem ellipse_hyperbola_min_eccentricity (c : ℝ) (e₁ e₂ : ℝ) : 
  c > 0 → -- Foci are distinct points
  e₁ > 0 → -- Eccentricity of ellipse is positive
  e₂ > 0 → -- Eccentricity of hyperbola is positive
  e₁ * e₂ = 1 → -- Relationship between eccentricities due to shared foci and asymptote condition
  3 * e₁^2 + e₂^2 ≥ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_min_eccentricity_l2592_259215


namespace NUMINAMATH_CALUDE_alternating_sum_coefficients_l2592_259206

theorem alternating_sum_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ - a₁ + a₂ - a₃ + a₄ - a₅ = -1 := by
sorry

end NUMINAMATH_CALUDE_alternating_sum_coefficients_l2592_259206


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2592_259228

theorem fraction_to_decimal : (53 : ℚ) / (2^2 * 5^3) = 0.106 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2592_259228


namespace NUMINAMATH_CALUDE_ten_faucets_fill_time_l2592_259231

/-- The time (in seconds) it takes for a given number of faucets to fill a tub of a given capacity. -/
def fill_time (num_faucets : ℕ) (capacity : ℝ) : ℝ :=
  sorry

/-- The rate at which one faucet fills a tub (in gallons per minute). -/
def faucet_rate : ℝ :=
  sorry

theorem ten_faucets_fill_time :
  -- Condition 1: Five faucets fill a 150-gallon tub in 10 minutes
  fill_time 5 150 = 10 * 60 →
  -- Condition 2: All faucets dispense water at the same rate (implicit in the definition of faucet_rate)
  -- Prove: Ten faucets will fill a 50-gallon tub in 100 seconds
  fill_time 10 50 = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ten_faucets_fill_time_l2592_259231


namespace NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l2592_259281

-- Define the concept of a line in 3D space
structure Line3D where
  -- Add appropriate fields to represent a line in 3D space
  -- This is a simplified representation
  dummy : Unit

-- Define what it means for two lines to be skew
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Add appropriate definition
  sorry

-- Define what it means for two lines to not intersect
def do_not_intersect (l1 l2 : Line3D) : Prop :=
  -- Add appropriate definition
  sorry

-- Theorem statement
theorem skew_lines_sufficient_not_necessary :
  (∀ l1 l2 : Line3D, are_skew l1 l2 → do_not_intersect l1 l2) ∧
  (∃ l1 l2 : Line3D, do_not_intersect l1 l2 ∧ ¬are_skew l1 l2) :=
sorry

end NUMINAMATH_CALUDE_skew_lines_sufficient_not_necessary_l2592_259281


namespace NUMINAMATH_CALUDE_f_inequality_l2592_259273

-- Define the function f
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 - b*x + c

-- State the theorem
theorem f_inequality (b c : ℝ) :
  (∀ x, f b c (1 + x) = f b c (1 - x)) →
  f b c 0 = 3 →
  ∀ x, f b c (c^x) ≥ f b c (b^x) :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_l2592_259273


namespace NUMINAMATH_CALUDE_max_marks_proof_l2592_259284

def math_pass_percentage : ℚ := 45/100
def science_pass_percentage : ℚ := 1/2
def math_score : ℕ := 267
def math_shortfall : ℕ := 45
def science_score : ℕ := 292
def science_shortfall : ℕ := 38

def total_marks : ℕ := 1354

theorem max_marks_proof :
  let math_total := (math_score + math_shortfall) / math_pass_percentage
  let science_total := (science_score + science_shortfall) / science_pass_percentage
  ⌈math_total⌉ + science_total = total_marks := by
  sorry

end NUMINAMATH_CALUDE_max_marks_proof_l2592_259284


namespace NUMINAMATH_CALUDE_min_distance_parabola_to_line_l2592_259254

/-- The minimum distance from a point on the parabola y = x^2 + 1 to the line y = 2x - 1 is √5/5 -/
theorem min_distance_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2 + 1}
  let line := {p : ℝ × ℝ | p.2 = 2 * p.1 - 1}
  (∀ p ∈ parabola, ∃ q ∈ line, ∀ r ∈ line, dist p q ≤ dist p r) →
  (∃ p ∈ parabola, ∃ q ∈ line, dist p q = Real.sqrt 5 / 5) ∧
  (∀ p ∈ parabola, ∀ q ∈ line, dist p q ≥ Real.sqrt 5 / 5) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_parabola_to_line_l2592_259254


namespace NUMINAMATH_CALUDE_rectangle_area_l2592_259270

/-- The area of a rectangle with length 1.2 meters and width 0.5 meters is 0.6 square meters. -/
theorem rectangle_area : 
  let length : ℝ := 1.2
  let width : ℝ := 0.5
  length * width = 0.6 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2592_259270


namespace NUMINAMATH_CALUDE_susan_initial_money_l2592_259261

theorem susan_initial_money (S : ℝ) : 
  S - (S / 5 + S / 4 + 120) = 1200 → S = 2400 := by
  sorry

end NUMINAMATH_CALUDE_susan_initial_money_l2592_259261


namespace NUMINAMATH_CALUDE_expression_bounds_l2592_259259

theorem expression_bounds (a b c d e : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) (he : 0 ≤ e ∧ e ≤ 1) :
  4 * Real.sqrt (2/3) ≤ 
    Real.sqrt (a^2 + (1-b)^2 + e^2) + Real.sqrt (b^2 + (1-c)^2 + e^2) + 
    Real.sqrt (c^2 + (1-d)^2 + e^2) + Real.sqrt (d^2 + (1-a)^2 + e^2) ∧
  Real.sqrt (a^2 + (1-b)^2 + e^2) + Real.sqrt (b^2 + (1-c)^2 + e^2) + 
  Real.sqrt (c^2 + (1-d)^2 + e^2) + Real.sqrt (d^2 + (1-a)^2 + e^2) ≤ 8 ∧
  ∃ (a b c d e : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 
    0 ≤ d ∧ d ≤ 1 ∧ 0 ≤ e ∧ e ≤ 1 ∧
    Real.sqrt (a^2 + (1-b)^2 + e^2) + Real.sqrt (b^2 + (1-c)^2 + e^2) + 
    Real.sqrt (c^2 + (1-d)^2 + e^2) + Real.sqrt (d^2 + (1-a)^2 + e^2) = 4 * Real.sqrt (2/3) ∧
  ∃ (a b c d e : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 
    0 ≤ d ∧ d ≤ 1 ∧ 0 ≤ e ∧ e ≤ 1 ∧
    Real.sqrt (a^2 + (1-b)^2 + e^2) + Real.sqrt (b^2 + (1-c)^2 + e^2) + 
    Real.sqrt (c^2 + (1-d)^2 + e^2) + Real.sqrt (d^2 + (1-a)^2 + e^2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_expression_bounds_l2592_259259


namespace NUMINAMATH_CALUDE_min_value_theorem_l2592_259221

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 1/x' + 1/y' = 1 → 
    1/(x' - 1) + 4/(y' - 1) ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2592_259221


namespace NUMINAMATH_CALUDE_min_value_f_min_value_achieved_l2592_259211

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

theorem min_value_f :
  ∀ x : ℝ, x ≥ 0 → f x ≥ 1 := by
  sorry

theorem min_value_achieved :
  ∃ x : ℝ, x ≥ 0 ∧ f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_min_value_achieved_l2592_259211


namespace NUMINAMATH_CALUDE_mathville_running_difference_l2592_259209

/-- The side length of a square block in Mathville -/
def block_side_length : ℝ := 500

/-- The width of streets in Mathville -/
def street_width : ℝ := 30

/-- The length of Matt's path around the block -/
def matt_path_length : ℝ := 4 * block_side_length

/-- The length of Mike's path around the block -/
def mike_path_length : ℝ := 4 * (block_side_length + 2 * street_width)

/-- The difference between Mike's and Matt's path lengths -/
def path_length_difference : ℝ := mike_path_length - matt_path_length

theorem mathville_running_difference : path_length_difference = 240 := by
  sorry

end NUMINAMATH_CALUDE_mathville_running_difference_l2592_259209


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2592_259294

theorem sum_of_a_and_b (a b : ℝ) : 
  (abs a = 3) → (abs b = 7) → (abs (a - b) = b - a) → 
  (a + b = 10 ∨ a + b = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2592_259294


namespace NUMINAMATH_CALUDE_trapezoid_ratio_satisfies_equation_l2592_259297

/-- Represents a trapezoid with a point inside dividing it into four triangles -/
structure TrapezoidWithPoint where
  AB : ℝ
  CD : ℝ
  area_PCD : ℝ
  area_PAD : ℝ
  area_PBC : ℝ
  area_PAB : ℝ
  h_AB_gt_CD : AB > CD
  h_areas : area_PCD = 3 ∧ area_PAD = 5 ∧ area_PBC = 6 ∧ area_PAB = 8

/-- The ratio of AB to CD satisfies a specific quadratic equation -/
theorem trapezoid_ratio_satisfies_equation (t : TrapezoidWithPoint) :
  let k := t.AB / t.CD
  k^2 + (22/6) * k + 16/6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_ratio_satisfies_equation_l2592_259297


namespace NUMINAMATH_CALUDE_multiple_choice_test_choices_l2592_259289

theorem multiple_choice_test_choices (n : ℕ) : 
  (n + 1)^4 = 625 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_multiple_choice_test_choices_l2592_259289


namespace NUMINAMATH_CALUDE_eccentricity_is_sqrt_five_l2592_259235

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- Represents a point on a hyperbola -/
structure PointOnHyperbola {a b : ℝ} (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1

/-- The left and right foci of a hyperbola -/
def foci {a b : ℝ} (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The distance between a point and a focus -/
def dist_to_focus {a b : ℝ} (h : Hyperbola a b) (p : PointOnHyperbola h) (focus : ℝ) : ℝ := sorry

/-- The angle between the lines from a point to the foci -/
def angle_between_foci {a b : ℝ} (h : Hyperbola a b) (p : PointOnHyperbola h) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity {a b : ℝ} (h : Hyperbola a b) : ℝ := sorry

/-- Theorem: If there exists a point on the hyperbola where the angle between the lines to the foci is 90° and the distance to one focus is twice the distance to the other, then the eccentricity is √5 -/
theorem eccentricity_is_sqrt_five {a b : ℝ} (h : Hyperbola a b) :
  (∃ p : PointOnHyperbola h, 
    angle_between_foci h p = Real.pi / 2 ∧ 
    dist_to_focus h p (foci h).1 = 2 * dist_to_focus h p (foci h).2) →
  eccentricity h = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_eccentricity_is_sqrt_five_l2592_259235


namespace NUMINAMATH_CALUDE_choir_arrangement_l2592_259286

theorem choir_arrangement (total_members : ℕ) (num_rows : ℕ) (h1 : total_members = 51) (h2 : num_rows = 4) :
  ∃ (row : ℕ), row ≤ num_rows ∧ 13 ≤ (total_members / num_rows + (if row ≤ total_members % num_rows then 1 else 0)) :=
by sorry

end NUMINAMATH_CALUDE_choir_arrangement_l2592_259286


namespace NUMINAMATH_CALUDE_muffin_ratio_l2592_259225

/-- The number of muffins Sasha made -/
def sasha_muffins : ℕ := 30

/-- The price of each muffin in dollars -/
def muffin_price : ℕ := 4

/-- The total amount raised in dollars -/
def total_raised : ℕ := 900

/-- The number of muffins Melissa made -/
def melissa_muffins : ℕ := 120

/-- The number of muffins Tiffany made -/
def tiffany_muffins : ℕ := (sasha_muffins + melissa_muffins) / 2

/-- The total number of muffins made -/
def total_muffins : ℕ := sasha_muffins + melissa_muffins + tiffany_muffins

theorem muffin_ratio : 
  (total_muffins * muffin_price = total_raised) → 
  (melissa_muffins : ℚ) / sasha_muffins = 4 := by
sorry

end NUMINAMATH_CALUDE_muffin_ratio_l2592_259225


namespace NUMINAMATH_CALUDE_classroom_ratio_l2592_259208

theorem classroom_ratio (total_students : ℕ) (num_boys : ℕ) (h1 : total_students > 0) (h2 : num_boys ≤ total_students) :
  let prob_boy := num_boys / total_students
  let prob_girl := (total_students - num_boys) / total_students
  (prob_boy / prob_girl = 3 / 4) → (num_boys / total_students = 3 / 7) := by
  sorry

end NUMINAMATH_CALUDE_classroom_ratio_l2592_259208


namespace NUMINAMATH_CALUDE_max_intersected_edges_l2592_259245

/-- A regular p-gonal prism -/
structure RegularPrism (p : ℕ) :=
  (p_pos : p > 0)

/-- A plane that does not pass through the vertices of the prism -/
structure NonVertexPlane (p : ℕ) (prism : RegularPrism p) :=

/-- The number of edges of a regular p-gonal prism intersected by a plane -/
def intersected_edges (p : ℕ) (prism : RegularPrism p) (plane : NonVertexPlane p prism) : ℕ :=
  sorry

/-- The maximum number of edges that can be intersected is 3p -/
theorem max_intersected_edges (p : ℕ) (prism : RegularPrism p) :
  ∃ (plane : NonVertexPlane p prism), intersected_edges p prism plane = 3 * p ∧
  ∀ (other_plane : NonVertexPlane p prism), intersected_edges p prism other_plane ≤ 3 * p :=
sorry

end NUMINAMATH_CALUDE_max_intersected_edges_l2592_259245


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l2592_259268

/-- The area of a square with adjacent vertices at (1,5) and (4,-2) is 58 -/
theorem square_area_from_vertices : 
  let x1 : ℝ := 1
  let y1 : ℝ := 5
  let x2 : ℝ := 4
  let y2 : ℝ := -2
  let side_length : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  let area : ℝ := side_length^2
  area = 58 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l2592_259268


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2592_259232

theorem perpendicular_vectors (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, m]
  let b : Fin 2 → ℝ := ![4, -2]
  (∀ i, i < 2 → a i * b i = 0) → m = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2592_259232


namespace NUMINAMATH_CALUDE_parabola_sum_l2592_259279

-- Define a quadratic function
def quadratic (p q r : ℝ) : ℝ → ℝ := λ x => p * x^2 + q * x + r

theorem parabola_sum (p q r : ℝ) :
  -- The vertex of the parabola is (3, -1)
  (∀ x, quadratic p q r x ≥ quadratic p q r 3) ∧
  quadratic p q r 3 = -1 ∧
  -- The parabola passes through the point (0, 8)
  quadratic p q r 0 = 8
  →
  p + q + r = 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l2592_259279


namespace NUMINAMATH_CALUDE_even_sum_squares_half_l2592_259227

theorem even_sum_squares_half (n x y : ℤ) (h : 2 * n = x^2 + y^2) :
  n = ((x + y) / 2)^2 + ((x - y) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_squares_half_l2592_259227


namespace NUMINAMATH_CALUDE_sqrt_two_squared_times_three_to_fourth_l2592_259280

theorem sqrt_two_squared_times_three_to_fourth : Real.sqrt (2^2 * 3^4) = 18 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_times_three_to_fourth_l2592_259280


namespace NUMINAMATH_CALUDE_largest_angle_bound_l2592_259218

/-- Triangle DEF with sides e, f, and d -/
structure Triangle where
  e : ℝ
  f : ℝ
  d : ℝ

/-- The angle opposite to side d in degrees -/
def angle_opposite_d (t : Triangle) : ℝ := sorry

theorem largest_angle_bound (t : Triangle) (y : ℝ) :
  t.e = 2 →
  t.f = 2 →
  t.d > 2 * Real.sqrt 2 →
  (∀ z, z > y → angle_opposite_d t > z) →
  y = 120 := by sorry

end NUMINAMATH_CALUDE_largest_angle_bound_l2592_259218


namespace NUMINAMATH_CALUDE_female_students_count_l2592_259237

theorem female_students_count (total_students : ℕ) 
  (h1 : total_students = 63) 
  (h2 : ∀ (female_count : ℕ), 
    female_count ≤ total_students → 
    (female_count : ℚ) / total_students = 
    (10 : ℚ) / 11 * ((total_students - female_count) : ℚ) / total_students) : 
  ∃ (female_count : ℕ), female_count = 30 ∧ female_count ≤ total_students :=
sorry

end NUMINAMATH_CALUDE_female_students_count_l2592_259237


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2592_259258

theorem fraction_sum_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1 / (2 * a * b) + b / (4 * a) = (2 + b^2) / (4 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2592_259258


namespace NUMINAMATH_CALUDE_sin_x_squared_not_periodic_l2592_259214

/-- Definition of a periodic function -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

/-- Statement: sin(x^2) is not periodic -/
theorem sin_x_squared_not_periodic : ¬ IsPeriodic (fun x ↦ Real.sin (x^2)) := by
  sorry


end NUMINAMATH_CALUDE_sin_x_squared_not_periodic_l2592_259214


namespace NUMINAMATH_CALUDE_group_a_better_performance_l2592_259247

/-- Represents a group of students with their quiz scores -/
structure StudentGroup where
  scores : List Nat
  mean : Nat
  median : Nat
  mode : Nat
  variance : Nat
  excellent_rate : Rat

/-- Defines what score is considered excellent -/
def excellent_score : Nat := 8

/-- Group A data -/
def group_a : StudentGroup := {
  scores := [5, 7, 8, 8, 8, 8, 8, 9, 9, 10],
  mean := 8,
  median := 8,
  mode := 8,
  variance := 16,
  excellent_rate := 8 / 10
}

/-- Group B data -/
def group_b : StudentGroup := {
  scores := [7, 7, 7, 7, 8, 8, 8, 9, 9, 10],
  mean := 8,
  median := 8,
  mode := 7,
  variance := 1,
  excellent_rate := 6 / 10
}

/-- Theorem stating that Group A has a higher excellent rate than Group B -/
theorem group_a_better_performance (ga : StudentGroup) (gb : StudentGroup) 
  (h1 : ga = group_a) (h2 : gb = group_b) : 
  ga.excellent_rate > gb.excellent_rate := by
  sorry

end NUMINAMATH_CALUDE_group_a_better_performance_l2592_259247


namespace NUMINAMATH_CALUDE_x_minus_y_equals_one_l2592_259238

theorem x_minus_y_equals_one (x y : ℝ) 
  (h1 : x^2 + y^2 = 25) 
  (h2 : x + y = 7) 
  (h3 : x > y) : 
  x - y = 1 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_one_l2592_259238


namespace NUMINAMATH_CALUDE_ellipse_equation_specific_l2592_259257

/-- Represents an ellipse in the Cartesian coordinate plane -/
structure Ellipse where
  center : ℝ × ℝ
  foci_axis : ℝ × ℝ
  minor_axis_length : ℝ
  eccentricity : ℝ

/-- The equation of an ellipse given its parameters -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (x - e.center.1)^2 / a^2 + (y - e.center.2)^2 / b^2 = 1 ∧
    e.minor_axis_length = 2 * b ∧
    e.eccentricity = Real.sqrt (1 - b^2 / a^2)

theorem ellipse_equation_specific (e : Ellipse) :
  e.center = (0, 0) →
  e.foci_axis = (1, 0) →
  e.minor_axis_length = 2 →
  e.eccentricity = Real.sqrt 2 / 2 →
  ∀ (x y : ℝ), ellipse_equation e x y ↔ x^2 / 2 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_specific_l2592_259257


namespace NUMINAMATH_CALUDE_max_questions_l2592_259229

/-- Represents a contestant's answers to n questions -/
def Answers (n : ℕ) := Fin n → Bool

/-- The number of contestants -/
def num_contestants : ℕ := 8

/-- Condition: For any pair of questions, exactly two contestants answered each combination -/
def valid_distribution (n : ℕ) (answers : Fin num_contestants → Answers n) : Prop :=
  ∀ i j : Fin n, i ≠ j →
    (∃! (s : Finset (Fin num_contestants)) (hs : s.card = 2),
      ∀ k ∈ s, answers k i = true ∧ answers k j = true) ∧
    (∃! (s : Finset (Fin num_contestants)) (hs : s.card = 2),
      ∀ k ∈ s, answers k i = false ∧ answers k j = false) ∧
    (∃! (s : Finset (Fin num_contestants)) (hs : s.card = 2),
      ∀ k ∈ s, answers k i = true ∧ answers k j = false) ∧
    (∃! (s : Finset (Fin num_contestants)) (hs : s.card = 2),
      ∀ k ∈ s, answers k i = false ∧ answers k j = true)

/-- The maximum number of questions satisfying the conditions -/
theorem max_questions :
  ∀ n : ℕ, (∃ answers : Fin num_contestants → Answers n, valid_distribution n answers) →
    n ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_questions_l2592_259229


namespace NUMINAMATH_CALUDE_dilation_matrix_determinant_l2592_259210

theorem dilation_matrix_determinant :
  ∀ (E : Matrix (Fin 3) (Fin 3) ℝ),
  (∀ i j : Fin 3, i ≠ j → E i j = 0) →
  E 0 0 = 3 →
  E 1 1 = 5 →
  E 2 2 = 7 →
  Matrix.det E = 105 := by
sorry

end NUMINAMATH_CALUDE_dilation_matrix_determinant_l2592_259210


namespace NUMINAMATH_CALUDE_four_player_tournament_games_l2592_259226

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a tournament with 4 players, where each player plays against every
    other player exactly once, the total number of games played is 6. -/
theorem four_player_tournament_games :
  num_games 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_four_player_tournament_games_l2592_259226


namespace NUMINAMATH_CALUDE_area_third_face_l2592_259275

/-- Theorem: Area of the third adjacent face of a cuboidal box -/
theorem area_third_face (l w h : ℝ) : 
  l * w = 120 →
  w * h = 60 →
  l * w * h = 720 →
  l * h = 72 := by
sorry

end NUMINAMATH_CALUDE_area_third_face_l2592_259275


namespace NUMINAMATH_CALUDE_parking_lot_ratio_l2592_259283

/-- Given the initial number of cars in the front parking lot, the total number of cars at the end,
    and the number of cars added during the play, prove the ratio of cars in the back to front parking lot. -/
theorem parking_lot_ratio
  (front_initial : ℕ)
  (total_end : ℕ)
  (added_during : ℕ)
  (h1 : front_initial = 100)
  (h2 : total_end = 700)
  (h3 : added_during = 300) :
  (total_end - added_during - front_initial) / front_initial = 3 := by
  sorry

#check parking_lot_ratio

end NUMINAMATH_CALUDE_parking_lot_ratio_l2592_259283


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2592_259252

/-- Given a geometric sequence {a_n} where a_4 + a_8 = π, 
    prove that a_6(a_2 + 2a_6 + a_10) = π² -/
theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 4 + a 8 = Real.pi →            -- Given condition
  a 6 * (a 2 + 2 * a 6 + a 10) = Real.pi ^ 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2592_259252


namespace NUMINAMATH_CALUDE_inequality_range_l2592_259282

-- Define the inequality
def inequality (x a : ℝ) : Prop :=
  x^2 - (a + 1) * x + a ≤ 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x : ℝ | inequality x a}

-- Define the interval [-4, 3]
def interval : Set ℝ :=
  {x : ℝ | -4 ≤ x ∧ x ≤ 3}

-- Statement of the theorem
theorem inequality_range :
  (∀ a : ℝ, solution_set a ⊆ interval) →
  ∀ a : ℝ, -4 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l2592_259282


namespace NUMINAMATH_CALUDE_seven_rows_of_ten_for_79_people_l2592_259240

/-- Represents a seating arrangement with rows of either 9 or 10 people -/
structure SeatingArrangement where
  rows_of_9 : ℕ
  rows_of_10 : ℕ

/-- The total number of people in a seating arrangement -/
def total_people (s : SeatingArrangement) : ℕ :=
  9 * s.rows_of_9 + 10 * s.rows_of_10

/-- Theorem stating that for 79 people, there are 7 rows of 10 people -/
theorem seven_rows_of_ten_for_79_people :
  ∃ (s : SeatingArrangement), total_people s = 79 ∧ s.rows_of_10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_rows_of_ten_for_79_people_l2592_259240


namespace NUMINAMATH_CALUDE_brick_height_l2592_259293

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: For a rectangular prism with length 10, width 4, and surface area 164, the height is 3 -/
theorem brick_height (l w sa : ℝ) (hl : l = 10) (hw : w = 4) (hsa : sa = 164) :
  ∃ h : ℝ, h = 3 ∧ surface_area l w h = sa := by
  sorry

end NUMINAMATH_CALUDE_brick_height_l2592_259293


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l2592_259298

theorem modulo_eleven_residue : (310 + 6 * 45 + 8 * 154 + 3 * 23) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l2592_259298


namespace NUMINAMATH_CALUDE_complex_fraction_equals_one_minus_i_l2592_259277

theorem complex_fraction_equals_one_minus_i : 
  let i : ℂ := Complex.I
  2 / (1 + i) = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_one_minus_i_l2592_259277


namespace NUMINAMATH_CALUDE_solution_set_intersection_range_l2592_259212

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - 2*|x + 1|

-- Part I
theorem solution_set (x : ℝ) : 
  x ∈ Set.Ioo (-4/3 : ℝ) 1 ↔ f 5 x > 2 := by sorry

-- Part II
theorem intersection_range (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = x^2 + 2*x + 3 ∧ y = f m x) ↔ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_intersection_range_l2592_259212


namespace NUMINAMATH_CALUDE_triangle_properties_l2592_259233

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A))
  (h2 : t.A = 2 * t.B)
  (h3 : t.A + t.B + t.C = Real.pi) :  -- Triangle angle sum property
  t.C = 5 * Real.pi / 8 ∧ 2 * t.a ^ 2 = t.b ^ 2 + t.c ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2592_259233


namespace NUMINAMATH_CALUDE_graces_tower_height_l2592_259288

theorem graces_tower_height (clyde_height grace_height : ℝ) : 
  grace_height = 8 * clyde_height ∧ 
  grace_height = clyde_height + 35 → 
  grace_height = 40 := by
sorry

end NUMINAMATH_CALUDE_graces_tower_height_l2592_259288


namespace NUMINAMATH_CALUDE_tenth_row_third_element_l2592_259256

/-- Represents the exponent of 2 for an element in the triangular array --/
def triangularArrayExponent (row : ℕ) (position : ℕ) : ℕ :=
  (row - 1) * row / 2 + position

/-- The theorem stating that the third element from the left in the 10th row is 2^47 --/
theorem tenth_row_third_element :
  triangularArrayExponent 10 2 = 47 := by
  sorry

end NUMINAMATH_CALUDE_tenth_row_third_element_l2592_259256


namespace NUMINAMATH_CALUDE_max_triangle_area_l2592_259243

/-- The maximum area of a triangle ABC with side length constraints -/
theorem max_triangle_area (AB BC CA : ℝ) 
  (hAB : 0 ≤ AB ∧ AB ≤ 1)
  (hBC : 1 ≤ BC ∧ BC ≤ 2)
  (hCA : 2 ≤ CA ∧ CA ≤ 3) :
  ∃ (area : ℝ), area ≤ 1 ∧ 
  ∀ (a : ℝ), (∃ (x y z : ℝ), 
    0 ≤ x ∧ x ≤ 1 ∧
    1 ≤ y ∧ y ≤ 2 ∧
    2 ≤ z ∧ z ≤ 3 ∧
    a = (x + y + z) / 2 * ((x + y + z) / 2 - x) * ((x + y + z) / 2 - y) * ((x + y + z) / 2 - z)) →
  a ≤ area :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_l2592_259243


namespace NUMINAMATH_CALUDE_color_box_problem_l2592_259272

theorem color_box_problem (total_pencils : ℕ) (emily_and_friends : ℕ) (colors : ℕ) : 
  total_pencils = 56 → emily_and_friends = 8 → total_pencils = emily_and_friends * colors → colors = 7 := by
  sorry

end NUMINAMATH_CALUDE_color_box_problem_l2592_259272


namespace NUMINAMATH_CALUDE_cube_triangle_areas_sum_l2592_259249

/-- Represents a 2 × 2 × 2 cube -/
structure Cube where
  side_length : ℝ
  side_length_eq : side_length = 2

/-- The sum of areas of all triangles with vertices on the cube -/
def sum_triangle_areas (c : Cube) : ℝ := sorry

/-- The sum can be expressed as m + √n + √p -/
def sum_representation (m n p : ℕ) (c : Cube) : Prop :=
  sum_triangle_areas c = m + Real.sqrt n + Real.sqrt p

theorem cube_triangle_areas_sum (c : Cube) :
  ∃ (m n p : ℕ), sum_representation m n p c ∧ m + n + p = 5424 := by sorry

end NUMINAMATH_CALUDE_cube_triangle_areas_sum_l2592_259249


namespace NUMINAMATH_CALUDE_area_parallelogram_from_diagonals_l2592_259255

/-- A quadrilateral in a plane -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- The area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- The diagonals of a quadrilateral -/
def diagonals (q : Quadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

/-- A parallelogram with sides parallel and equal to given line segments -/
def parallelogram_from_diagonals (d : (ℝ × ℝ) × (ℝ × ℝ)) : Quadrilateral := sorry

/-- The theorem stating that the area of the parallelogram formed by the diagonals
    is twice the area of the original quadrilateral -/
theorem area_parallelogram_from_diagonals (q : Quadrilateral) :
  area (parallelogram_from_diagonals (diagonals q)) = 2 * area q := by sorry

end NUMINAMATH_CALUDE_area_parallelogram_from_diagonals_l2592_259255


namespace NUMINAMATH_CALUDE_power_equality_l2592_259269

theorem power_equality (K : ℕ) : 32^2 * 4^4 = 2^K → K = 18 := by
  have h1 : 32 = 2^5 := by sorry
  have h2 : 4 = 2^2 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_equality_l2592_259269


namespace NUMINAMATH_CALUDE_largest_fraction_l2592_259242

theorem largest_fraction :
  let a := 5 / 13
  let b := 7 / 16
  let c := 23 / 46
  let d := 51 / 101
  let e := 203 / 405
  d > a ∧ d > b ∧ d > c ∧ d > e := by
sorry

end NUMINAMATH_CALUDE_largest_fraction_l2592_259242


namespace NUMINAMATH_CALUDE_travel_distance_theorem_l2592_259296

/-- The total distance Amoli and Anayet need to travel -/
def total_distance (amoli_speed : ℝ) (amoli_time : ℝ) (anayet_speed : ℝ) (anayet_time : ℝ) (remaining_distance : ℝ) : ℝ :=
  amoli_speed * amoli_time + anayet_speed * anayet_time + remaining_distance

/-- Theorem stating the total distance Amoli and Anayet need to travel -/
theorem travel_distance_theorem :
  let amoli_speed : ℝ := 42
  let amoli_time : ℝ := 3
  let anayet_speed : ℝ := 61
  let anayet_time : ℝ := 2
  let remaining_distance : ℝ := 121
  total_distance amoli_speed amoli_time anayet_speed anayet_time remaining_distance = 369 := by
sorry

end NUMINAMATH_CALUDE_travel_distance_theorem_l2592_259296


namespace NUMINAMATH_CALUDE_distance_between_points_l2592_259222

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 5)
  let p2 : ℝ × ℝ := (5, 1)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l2592_259222


namespace NUMINAMATH_CALUDE_heart_then_ten_probability_l2592_259250

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of hearts in a deck
def num_hearts : ℕ := 13

-- Define the number of 10s in a deck
def num_tens : ℕ := 4

-- Define the probability of the event
def prob_heart_then_ten : ℚ := 1 / total_cards

-- State the theorem
theorem heart_then_ten_probability :
  prob_heart_then_ten = (num_hearts * num_tens) / (total_cards * (total_cards - 1)) :=
sorry

end NUMINAMATH_CALUDE_heart_then_ten_probability_l2592_259250


namespace NUMINAMATH_CALUDE_f_two_eq_zero_f_x_plus_two_f_x_plus_four_l2592_259271

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom odd_function : ∀ x, f (-x) = -f x
axiom function_property : ∀ x, f (x + 2) + 2 * f (-x) = 0

-- Theorem statements
theorem f_two_eq_zero : f 2 = 0 := by sorry

theorem f_x_plus_two : ∀ x, f (x + 2) = 2 * f x := by sorry

theorem f_x_plus_four : ∀ x, f (x + 4) = 4 * f x := by sorry

end NUMINAMATH_CALUDE_f_two_eq_zero_f_x_plus_two_f_x_plus_four_l2592_259271


namespace NUMINAMATH_CALUDE_tangent_points_sum_constant_l2592_259244

/-- Parabola defined by x^2 = 4y -/
def Parabola (x y : ℝ) : Prop := x^2 = 4*y

/-- Point P with coordinates (a, -2) -/
def PointP (a : ℝ) : ℝ × ℝ := (a, -2)

/-- Tangent point on the parabola -/
def TangentPoint (x y : ℝ) : Prop := Parabola x y

/-- The theorem stating that for any point P(a, -2) and two tangent points A(x₁, y₁) and B(x₂, y₂) 
    on the parabola x^2 = 4y, the sum x₁x₂ + y₁y₂ is always equal to -4 -/
theorem tangent_points_sum_constant 
  (a x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : TangentPoint x₁ y₁) 
  (h₂ : TangentPoint x₂ y₂) : 
  x₁ * x₂ + y₁ * y₂ = -4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_points_sum_constant_l2592_259244


namespace NUMINAMATH_CALUDE_contractor_absence_l2592_259217

/-- Proves that given the specified contract conditions, the contractor was absent for 10 days -/
theorem contractor_absence (total_days : ℕ) (daily_pay : ℚ) (daily_fine : ℚ) (total_received : ℚ)
  (h_total_days : total_days = 30)
  (h_daily_pay : daily_pay = 25)
  (h_daily_fine : daily_fine = 7.5)
  (h_total_received : total_received = 425) :
  ∃ (days_worked days_absent : ℕ),
    days_worked + days_absent = total_days ∧
    days_worked * daily_pay - days_absent * daily_fine = total_received ∧
    days_absent = 10 :=
by sorry

end NUMINAMATH_CALUDE_contractor_absence_l2592_259217


namespace NUMINAMATH_CALUDE_inequality_problem_l2592_259202

theorem inequality_problem (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (b / a > c / a) ∧
  (c * (b - a) > 0) ∧
  (a * c * (a - c) < 0) ∧
  ¬ (∀ b, c * b^2 < a * b^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l2592_259202


namespace NUMINAMATH_CALUDE_product_divisible_by_seven_l2592_259265

theorem product_divisible_by_seven (A B : ℕ+) 
  (hA : Nat.Prime A.val)
  (hB : Nat.Prime B.val)
  (hAminusB : Nat.Prime (A.val - B.val))
  (hAplusB : Nat.Prime (A.val + B.val)) :
  7 ∣ (A.val * B.val * (A.val - B.val) * (A.val + B.val)) := by
sorry

end NUMINAMATH_CALUDE_product_divisible_by_seven_l2592_259265


namespace NUMINAMATH_CALUDE_arithmetic_geometric_general_term_l2592_259276

-- Define the arithmetic-geometric sequence
def arithmetic_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

-- Define the conditions
def conditions (a : ℕ → ℝ) : Prop :=
  a 2 = 6 ∧ 6 * a 1 + a 3 = 30

-- Theorem statement
theorem arithmetic_geometric_general_term (a : ℕ → ℝ) :
  arithmetic_geometric_seq a → conditions a →
  (∀ n : ℕ, a n = 3 * 3^(n - 1)) ∨ (∀ n : ℕ, a n = 2 * 2^(n - 1)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_general_term_l2592_259276


namespace NUMINAMATH_CALUDE_set_equals_open_interval_l2592_259266

theorem set_equals_open_interval :
  {x : ℝ | -1 < x ∧ x < 1} = Set.Ioo (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_set_equals_open_interval_l2592_259266


namespace NUMINAMATH_CALUDE_problem_solution_l2592_259201

theorem problem_solution (a b c n : ℝ) (h : n = (2 * a * b * c) / (c - a)) :
  c = (n * a) / (n - 2 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2592_259201


namespace NUMINAMATH_CALUDE_q_expression_l2592_259204

/-- Given a function q(x) satisfying the equation
    q(x) + (x^6 + 4x^4 + 5x^3 + 12x) = (8x^4 + 26x^3 + 15x^2 + 26x + 3),
    prove that q(x) = -x^6 + 4x^4 + 21x^3 + 15x^2 + 14x + 3 -/
theorem q_expression (q : ℝ → ℝ) 
    (h : ∀ x, q x + (x^6 + 4*x^4 + 5*x^3 + 12*x) = 8*x^4 + 26*x^3 + 15*x^2 + 26*x + 3) :
  ∀ x, q x = -x^6 + 4*x^4 + 21*x^3 + 15*x^2 + 14*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_q_expression_l2592_259204


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2592_259251

theorem expand_and_simplify (y : ℝ) : 5 * (6 * y^2 - 3 * y + 2 - 4 * y^3) = -20 * y^3 + 30 * y^2 - 15 * y + 10 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2592_259251


namespace NUMINAMATH_CALUDE_add_ten_to_number_l2592_259248

theorem add_ten_to_number (x : ℤ) (h : x + 36 = 71) : x + 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_add_ten_to_number_l2592_259248


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_l2592_259207

/-- An arithmetic sequence with given first two terms -/
def arithmetic_sequence (a₁ a₂ : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * (a₂ - a₁)

/-- Check if three numbers form a geometric sequence -/
def is_geometric_sequence (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem arithmetic_to_geometric :
  ∃ x : ℝ, is_geometric_sequence (x - 8) (x + (arithmetic_sequence (-8) (-6) 4))
                                 (x + (arithmetic_sequence (-8) (-6) 5)) ∧
            x = -1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_l2592_259207


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2592_259260

theorem complex_equation_solution (c d : ℂ) (x : ℝ) :
  Complex.abs c = 3 →
  Complex.abs d = 5 →
  c * d = x - 3 * Complex.I →
  x > 0 →
  x = 6 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2592_259260


namespace NUMINAMATH_CALUDE_sequence_properties_l2592_259262

def sequence_a (n : ℕ) : ℤ := 2^n - n - 2

def sequence_c (n : ℕ) : ℤ := sequence_a n + n + 2

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → sequence_a (n + 1) = 2 * sequence_a n + n + 1) ∧
  (sequence_a 1 = -1) ∧
  (∀ n : ℕ, n > 0 → sequence_c (n + 1) = 2 * sequence_c n) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2592_259262


namespace NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l2592_259216

/-- A function f is monotonic on ℝ if it is either non-decreasing or non-increasing on ℝ. -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∨ (∀ x y : ℝ, x ≤ y → f y ≤ f x)

/-- The function f(x) = x^3 + ax^2 + (a+6)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

theorem monotonic_f_implies_a_range (a : ℝ) :
  Monotonic (f a) → -3 < a ∧ a < 6 := by
  sorry

#check monotonic_f_implies_a_range

end NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l2592_259216


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2592_259253

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 9th term of the arithmetic sequence is 5 -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_first : a 1 = 3)
  (h_sum : a 4 + a 6 = 8) :
  a 9 = 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l2592_259253


namespace NUMINAMATH_CALUDE_triangle_side_length_l2592_259263

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if a = 3, C = 120°, and the area of the triangle is 15√3/4, then c = 7. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  a = 3 → 
  C = 2 * π / 3 → 
  (1/2) * a * b * Real.sin C = (15 * Real.sqrt 3) / 4 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_length_l2592_259263


namespace NUMINAMATH_CALUDE_average_height_four_people_l2592_259246

/-- The average height of four individuals given their relative heights -/
theorem average_height_four_people (G : ℝ) : 
  G + 2 = 64 →  -- Giselle is 2 inches shorter than Parker
  (G + 2) + 4 = 68 →  -- Parker is 4 inches shorter than Daisy
  68 - 8 = 60 →  -- Daisy is 8 inches taller than Reese
  (G + 64 + 68 + 60) / 4 = (192 + G) / 4 := by sorry

end NUMINAMATH_CALUDE_average_height_four_people_l2592_259246


namespace NUMINAMATH_CALUDE_brendas_age_l2592_259292

/-- Given the ages of Addison, Brenda, and Janet, prove that Brenda is 7/3 years old. -/
theorem brendas_age (A B J : ℚ) 
  (h1 : A = 4 * B)  -- Addison's age is four times Brenda's age
  (h2 : J = B + 7)  -- Janet is seven years older than Brenda
  (h3 : A = J)      -- Addison and Janet are twins
  : B = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_brendas_age_l2592_259292


namespace NUMINAMATH_CALUDE_cheese_calories_theorem_l2592_259287

/-- Calculates the remaining calories in a block of cheese -/
def remaining_calories (total_servings : ℕ) (calories_per_serving : ℕ) (eaten_servings : ℕ) : ℕ :=
  (total_servings - eaten_servings) * calories_per_serving

/-- Theorem: The remaining calories in a block of cheese with 16 servings, 
    where each serving contains 110 calories, and 5 servings have been eaten, 
    is equal to 1210 calories. -/
theorem cheese_calories_theorem : 
  remaining_calories 16 110 5 = 1210 := by
  sorry

end NUMINAMATH_CALUDE_cheese_calories_theorem_l2592_259287


namespace NUMINAMATH_CALUDE_expression_simplification_l2592_259285

theorem expression_simplification (a : ℝ) (h : a^2 - 4*a + 3 = 0) :
  (a - 4) / a / ((a + 2) / (a^2 - 2*a) - (a - 1) / (a^2 - 4*a + 4)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2592_259285


namespace NUMINAMATH_CALUDE_faucet_leak_proof_l2592_259220

/-- Represents a linear function y = kt + b -/
structure LinearFunction where
  k : ℝ
  b : ℝ

/-- The linear function passes through the points (1, 7) and (2, 12) -/
def passesThrough (f : LinearFunction) : Prop :=
  f.k * 1 + f.b = 7 ∧ f.k * 2 + f.b = 12

/-- The value of the function at t = 20 -/
def valueAt20 (f : LinearFunction) : ℝ :=
  f.k * 20 + f.b

/-- The total water leaked in 30 days in milliliters -/
def totalLeaked (f : LinearFunction) : ℝ :=
  f.k * 60 * 24 * 30

theorem faucet_leak_proof (f : LinearFunction) 
  (h : passesThrough f) : 
  f.k = 5 ∧ f.b = 2 ∧ 
  valueAt20 f = 102 ∧ 
  totalLeaked f = 216000 := by
  sorry

#check faucet_leak_proof

end NUMINAMATH_CALUDE_faucet_leak_proof_l2592_259220


namespace NUMINAMATH_CALUDE_intersection_distance_l2592_259213

/-- The distance between intersection points of two curves with a ray in polar coordinates --/
theorem intersection_distance (θ : Real) : 
  let ρ₁ : Real := Real.sqrt (2 / (Real.cos θ ^ 2 - Real.sin θ ^ 2))
  let ρ₂ : Real := 4 * Real.cos θ
  θ = π / 6 → abs (ρ₁ - ρ₂) = 2 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l2592_259213


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_digit_swap_l2592_259241

theorem arithmetic_geometric_mean_digit_swap : ∃ (x₁ x₂ : ℕ), 
  x₁ ≠ x₂ ∧
  (let A := (x₁ + x₂) / 2
   let G := Int.sqrt (x₁ * x₂)
   10 ≤ A ∧ A < 100 ∧
   10 ≤ G ∧ G < 100 ∧
   ((A / 10 = G % 10 ∧ A % 10 = G / 10) ∨
    (A % 10 = G / 10 ∧ A / 10 = G % 10)) ∧
   x₁ = 98 ∧
   x₂ = 32) :=
by
  sorry

#eval (98 + 32) / 2  -- Expected output: 65
#eval Int.sqrt (98 * 32)  -- Expected output: 56

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_digit_swap_l2592_259241


namespace NUMINAMATH_CALUDE_heart_fifteen_twentyfive_l2592_259205

/-- The ♡ operation on positive real numbers -/
noncomputable def heart (x y : ℝ) : ℝ := sorry

/-- The main theorem -/
theorem heart_fifteen_twentyfive :
  (∀ x y : ℝ, x > 0 → y > 0 → heart (x * y) y = x * heart y y) →
  (∀ x : ℝ, x > 0 → heart (heart x 1) x = heart x 1) →
  (heart 1 1 = 1) →
  heart 15 25 = 375 := by sorry

end NUMINAMATH_CALUDE_heart_fifteen_twentyfive_l2592_259205


namespace NUMINAMATH_CALUDE_train_speed_l2592_259219

/-- Proves that the speed of a train is 72 km/hr given its length and time to pass a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 100) (h2 : time = 5) :
  (length / time) * 3.6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2592_259219


namespace NUMINAMATH_CALUDE_units_produced_today_l2592_259203

/-- Calculates the number of units produced today given previous production data -/
theorem units_produced_today (n : ℕ) (prev_avg : ℝ) (new_avg : ℝ) 
  (h1 : n = 4)
  (h2 : prev_avg = 50)
  (h3 : new_avg = 58) : 
  (n + 1 : ℝ) * new_avg - n * prev_avg = 90 := by
  sorry

end NUMINAMATH_CALUDE_units_produced_today_l2592_259203


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l2592_259234

theorem cubic_sum_over_product (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_zero : a + b + c = 0) (sum_squares : a^2 + b^2 + c^2 = 3) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l2592_259234


namespace NUMINAMATH_CALUDE_same_solutions_implies_a_equals_four_l2592_259236

theorem same_solutions_implies_a_equals_four :
  ∀ a : ℝ,
  (∀ x : ℝ, x^2 - a = 0 ↔ 3*x^4 - 48 = 0) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_same_solutions_implies_a_equals_four_l2592_259236


namespace NUMINAMATH_CALUDE_milk_jars_theorem_l2592_259224

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

end NUMINAMATH_CALUDE_milk_jars_theorem_l2592_259224


namespace NUMINAMATH_CALUDE_shopping_problem_l2592_259230

/-- Shopping problem -/
theorem shopping_problem (initial_amount : ℝ) (baguette_cost : ℝ) (water_cost : ℝ)
  (chocolate_cost : ℝ) (milk_cost : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) :
  initial_amount = 50 →
  baguette_cost = 2 →
  water_cost = 1 →
  chocolate_cost = 1.5 →
  milk_cost = 3.5 →
  discount_rate = 0.1 →
  tax_rate = 0.07 →
  let baguette_total := 2 * baguette_cost
  let water_total := 2 * water_cost
  let chocolate_total := 2 * chocolate_cost
  let milk_total := milk_cost * (1 - discount_rate)
  let subtotal := baguette_total + water_total + chocolate_total + milk_total
  let tax := chocolate_total * tax_rate
  let total_cost := subtotal + tax
  initial_amount - total_cost = 37.64 := by
  sorry

end NUMINAMATH_CALUDE_shopping_problem_l2592_259230


namespace NUMINAMATH_CALUDE_homework_problem_l2592_259200

theorem homework_problem (a b c d : ℤ) : 
  (a < 0 ∧ b < 0 ∧ c < 0 ∧ d < 0) →
  (-a - b = -a * b) →
  (c * d = -182 * (1 / (-c - d))) →
  ((a = -2 ∧ b = -2) ∧ ((c = -1 ∧ d = -13) ∨ (c = -13 ∧ d = -1))) :=
by sorry

end NUMINAMATH_CALUDE_homework_problem_l2592_259200
