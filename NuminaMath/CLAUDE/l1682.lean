import Mathlib

namespace NUMINAMATH_CALUDE_two_isosceles_triangles_l1682_168237

/-- Represents a point in 2D space with integer coordinates -/
structure Point where
  x : Int
  y : Int

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Checks if a triangle is isosceles based on its vertices -/
def isIsosceles (a b c : Point) : Bool :=
  let d1 := squaredDistance a b
  let d2 := squaredDistance b c
  let d3 := squaredDistance c a
  d1 = d2 || d2 = d3 || d3 = d1

/-- The four triangles given in the problem -/
def triangle1 : (Point × Point × Point) := ({x := 0, y := 0}, {x := 4, y := 0}, {x := 2, y := 3})
def triangle2 : (Point × Point × Point) := ({x := 1, y := 1}, {x := 1, y := 4}, {x := 4, y := 1})
def triangle3 : (Point × Point × Point) := ({x := 3, y := 0}, {x := 6, y := 0}, {x := 4, y := 3})
def triangle4 : (Point × Point × Point) := ({x := 5, y := 2}, {x := 8, y := 2}, {x := 7, y := 5})

theorem two_isosceles_triangles :
  let triangles := [triangle1, triangle2, triangle3, triangle4]
  (triangles.filter (fun t => isIsosceles t.1 t.2.1 t.2.2)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_isosceles_triangles_l1682_168237


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l1682_168286

theorem at_least_one_greater_than_one (a b : ℝ) (h : a + b > 2) : 
  a > 1 ∨ b > 1 := by sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l1682_168286


namespace NUMINAMATH_CALUDE_mindy_income_multiplier_l1682_168217

/-- Given tax rates and combined rate, prove Mindy's income multiplier --/
theorem mindy_income_multiplier 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (combined_rate : ℝ) 
  (h1 : mork_rate = 0.45)
  (h2 : mindy_rate = 0.25)
  (h3 : combined_rate = 0.29) :
  ∃ k : ℝ, k = 4 ∧ 
    (mork_rate + mindy_rate * k) / (1 + k) = combined_rate :=
by sorry

end NUMINAMATH_CALUDE_mindy_income_multiplier_l1682_168217


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1682_168209

/-- Given a triangle with sides 9, 12, and 15 units, and a rectangle with width 6 units
    and area equal to the triangle's area, the perimeter of the rectangle is 30 units. -/
theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : w = 6)
    (h5 : w * (a * b / 2 / w) = a * b / 2) : 2 * (w + a * b / 2 / w) = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1682_168209


namespace NUMINAMATH_CALUDE_problem_statement_l1682_168243

theorem problem_statement (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 1 / y) :
  (x - 1 / x) * (y + 1 / y) = x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1682_168243


namespace NUMINAMATH_CALUDE_only_fourth_statement_correct_l1682_168267

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the theorem
theorem only_fourth_statement_correct 
  (a b : Line) 
  (α β : Plane) 
  (distinct_lines : a ≠ b) 
  (distinct_planes : α ≠ β) :
  (∃ (a b : Line) (α β : Plane),
    perpendicular a b ∧ 
    perpendicular_plane a α ∧ 
    perpendicular_plane b β → 
    perpendicular_planes α β) ∧
  (¬∃ (a b : Line) (α : Plane),
    perpendicular a b ∧ 
    parallel a α → 
    parallel b α) ∧
  (¬∃ (a : Line) (α β : Plane),
    parallel a α ∧ 
    perpendicular_planes α β → 
    perpendicular_plane a β) ∧
  (¬∃ (a : Line) (α β : Plane),
    perpendicular_plane a β ∧ 
    perpendicular_planes α β → 
    parallel a α) :=
by sorry

end NUMINAMATH_CALUDE_only_fourth_statement_correct_l1682_168267


namespace NUMINAMATH_CALUDE_smallest_fraction_between_l1682_168272

theorem smallest_fraction_between (p q : ℕ) : 
  p > 0 → q > 0 → (7 : ℚ)/12 < p/q → p/q < 5/8 → 
  (∀ p' q' : ℕ, p' > 0 → q' > 0 → q' < q → (7 : ℚ)/12 < p'/q' → p'/q' < 5/8 → False) →
  q - p = 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_fraction_between_l1682_168272


namespace NUMINAMATH_CALUDE_triangle_angle_y_l1682_168226

theorem triangle_angle_y (y : ℝ) : 
  (45 : ℝ) + 3 * y + y = 180 → y = 33.75 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_y_l1682_168226


namespace NUMINAMATH_CALUDE_difference_of_solutions_l1682_168256

def f (n : ℕ) : ℕ := (Finset.filter (fun (x, y, z) => 4*x + 3*y + 2*z = n) (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card

theorem difference_of_solutions : f 2009 - f 2000 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_solutions_l1682_168256


namespace NUMINAMATH_CALUDE_valid_words_count_l1682_168278

def alphabet_size : ℕ := 15
def max_word_length : ℕ := 5

def total_words (n : ℕ) : ℕ := 
  (alphabet_size ^ 1) + (alphabet_size ^ 2) + (alphabet_size ^ 3) + 
  (alphabet_size ^ 4) + (alphabet_size ^ 5)

def words_without_letter (n : ℕ) : ℕ := 
  ((alphabet_size - 1) ^ 1) + ((alphabet_size - 1) ^ 2) + 
  ((alphabet_size - 1) ^ 3) + ((alphabet_size - 1) ^ 4) + 
  ((alphabet_size - 1) ^ 5)

def words_without_two_letters (n : ℕ) : ℕ := 
  ((alphabet_size - 2) ^ 1) + ((alphabet_size - 2) ^ 2) + 
  ((alphabet_size - 2) ^ 3) + ((alphabet_size - 2) ^ 4) + 
  ((alphabet_size - 2) ^ 5)

theorem valid_words_count : 
  total_words alphabet_size - 2 * words_without_letter alphabet_size + 
  words_without_two_letters alphabet_size = 62460 := by
  sorry

end NUMINAMATH_CALUDE_valid_words_count_l1682_168278


namespace NUMINAMATH_CALUDE_hyperbola_iff_equation_l1682_168276

/-- Represents the condition for a hyperbola given a real number m -/
def is_hyperbola (m : ℝ) : Prop :=
  (m < -1) ∨ (-1 < m ∧ m < 1) ∨ (m > 2)

/-- The equation representing a potential hyperbola -/
def hyperbola_equation (m x y : ℝ) : Prop :=
  x^2 / (|m| - 1) - y^2 / (m - 2) = 1

/-- Theorem stating the equivalence between the hyperbola condition and the equation -/
theorem hyperbola_iff_equation (m : ℝ) :
  is_hyperbola m ↔ ∃ x y : ℝ, hyperbola_equation m x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_iff_equation_l1682_168276


namespace NUMINAMATH_CALUDE_correct_derivatives_l1682_168260

theorem correct_derivatives :
  ∀ x : ℝ,
  (deriv (λ x => x / (x + 1) - 2^x)) x = 1 / (x + 1)^2 - 2^x * Real.log 2 ∧
  (deriv (λ x => x^2 / Real.exp x)) x = (2*x - x^2) / Real.exp x :=
by sorry

end NUMINAMATH_CALUDE_correct_derivatives_l1682_168260


namespace NUMINAMATH_CALUDE_money_division_l1682_168264

theorem money_division (total : ℚ) (a b c : ℚ) : 
  total = 364 → 
  a + b + c = total → 
  a = (1/2) * b → 
  b = (1/2) * c → 
  c = 208 := by sorry

end NUMINAMATH_CALUDE_money_division_l1682_168264


namespace NUMINAMATH_CALUDE_parallel_line_distance_is_twelve_l1682_168229

/-- Represents a circle with three equally spaced parallel lines intersecting it. -/
structure CircleWithParallelLines where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The distance between adjacent parallel lines -/
  line_distance : ℝ
  /-- The length of the first chord -/
  chord1 : ℝ
  /-- The length of the second chord -/
  chord2 : ℝ
  /-- The length of the third chord -/
  chord3 : ℝ
  /-- The first chord has length 40 -/
  chord1_length : chord1 = 40
  /-- The second chord has length 36 -/
  chord2_length : chord2 = 36
  /-- The third chord has length 40 -/
  chord3_length : chord3 = 40

/-- Theorem stating that the distance between adjacent parallel lines is 12 -/
theorem parallel_line_distance_is_twelve (c : CircleWithParallelLines) : c.line_distance = 12 := by
  sorry


end NUMINAMATH_CALUDE_parallel_line_distance_is_twelve_l1682_168229


namespace NUMINAMATH_CALUDE_father_sees_boy_less_than_half_time_l1682_168238

/-- Represents a point on the perimeter of the square school -/
structure PerimeterPoint where
  side : Fin 4
  position : ℝ
  h_position : 0 ≤ position ∧ position ≤ 1

/-- Represents the movement of a person around the square school -/
structure Movement where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise
  start_point : PerimeterPoint

/-- Defines when two points are on the same side of the square -/
def on_same_side (p1 p2 : PerimeterPoint) : Prop :=
  p1.side = p2.side

/-- The boy's movement around the school -/
def boy_movement : Movement :=
  { speed := 10
  , direction := true  -- Always clockwise
  , start_point := { side := 0, position := 0, h_position := ⟨by norm_num, by norm_num⟩ } }

/-- The father's movement around the school -/
def father_movement : Movement :=
  { speed := 5
  , direction := true  -- Initial direction (can change)
  , start_point := { side := 0, position := 0, h_position := ⟨by norm_num, by norm_num⟩ } }

/-- Theorem stating that the father cannot see the boy for more than half the time -/
theorem father_sees_boy_less_than_half_time :
  ∀ (t : ℝ) (t_pos : 0 < t),
  ∃ (boy_pos father_pos : PerimeterPoint),
  (∀ τ : ℝ, 0 ≤ τ ∧ τ ≤ t →
    (on_same_side (boy_pos) (father_pos)) →
    (∃ (see_time : ℝ), see_time ≤ t / 2)) :=
sorry

end NUMINAMATH_CALUDE_father_sees_boy_less_than_half_time_l1682_168238


namespace NUMINAMATH_CALUDE_almond_distribution_l1682_168247

/-- The number of almonds Elaine received -/
def elaine_almonds : ℕ := 12

/-- The number of almonds Daniel received -/
def daniel_almonds : ℕ := elaine_almonds - 8

theorem almond_distribution :
  (elaine_almonds = daniel_almonds + 8) ∧
  (daniel_almonds = elaine_almonds / 3) →
  elaine_almonds = 12 := by
  sorry

end NUMINAMATH_CALUDE_almond_distribution_l1682_168247


namespace NUMINAMATH_CALUDE_k_not_determined_l1682_168249

theorem k_not_determined (k r : ℝ) (a : ℝ → ℝ) :
  (∀ r, a r = (k * r)^3) →
  (a (r / 2) = 0.125 * a r) →
  True
:= by sorry

end NUMINAMATH_CALUDE_k_not_determined_l1682_168249


namespace NUMINAMATH_CALUDE_simplify_expression_l1682_168228

theorem simplify_expression (x : ℝ) : 2*x - 3*(2+x) + 4*(2-x) - 5*(2+3*x) = -20*x - 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1682_168228


namespace NUMINAMATH_CALUDE_rectangle_length_l1682_168282

/-- Proves that a rectangle with given perimeter-to-breadth ratio and area has a specific length -/
theorem rectangle_length (P b l A : ℝ) : 
  P / b = 5 → 
  P = 2 * (l + b) → 
  A = l * b → 
  A = 216 → 
  l = 18 := by sorry

end NUMINAMATH_CALUDE_rectangle_length_l1682_168282


namespace NUMINAMATH_CALUDE_x_value_l1682_168205

theorem x_value (x y : ℝ) (h : x / (x - 1) = (y^3 + 2*y^2 - 2) / (y^3 + 2*y^2 - 3)) :
  x = (y^3 + 2*y^2 - 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_x_value_l1682_168205


namespace NUMINAMATH_CALUDE_box_height_l1682_168274

theorem box_height (long_width short_width top_area total_area : ℝ) 
  (h_long : long_width = 8)
  (h_short : short_width = 5)
  (h_top : top_area = 40)
  (h_total : total_area = 236) : 
  ∃ height : ℝ, 
    2 * long_width * height + 2 * short_width * height + 2 * top_area = total_area ∧ 
    height = 6 := by
  sorry

end NUMINAMATH_CALUDE_box_height_l1682_168274


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_nine_implies_product_l1682_168268

theorem sqrt_sum_equals_nine_implies_product (x : ℝ) :
  (Real.sqrt (7 + x) + Real.sqrt (28 - x) = 9) →
  ((7 + x) * (28 - x) = 529) := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_nine_implies_product_l1682_168268


namespace NUMINAMATH_CALUDE_partnership_profit_distribution_l1682_168214

/-- Partnership profit distribution problem -/
theorem partnership_profit_distribution 
  (total_profit : ℝ) 
  (h_profit : total_profit = 55000) 
  (invest_a invest_b invest_c : ℝ) 
  (h_a_b : invest_a = 3 * invest_b) 
  (h_a_c : invest_a = 2/3 * invest_c) : 
  invest_c / (invest_a + invest_b + invest_c) * total_profit = 9/17 * 55000 := by
sorry

end NUMINAMATH_CALUDE_partnership_profit_distribution_l1682_168214


namespace NUMINAMATH_CALUDE_f_increasing_l1682_168293

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x - Real.sin x else x^3 + 1

theorem f_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_f_increasing_l1682_168293


namespace NUMINAMATH_CALUDE_intersection_distance_l1682_168204

-- Define the two curves
def curve1 (x y : ℝ) : Prop := x = y^3
def curve2 (x y : ℝ) : Prop := x + y^3 = 2

-- Define the intersection points
def intersection1 : ℝ × ℝ := (1, 1)
def intersection2 : ℝ × ℝ := (0, 0)

-- State the theorem
theorem intersection_distance :
  curve1 (intersection1.1) (intersection1.2) ∧
  curve2 (intersection1.1) (intersection1.2) ∧
  curve1 (intersection2.1) (intersection2.2) ∧
  curve2 (intersection2.1) (intersection2.2) →
  Real.sqrt ((intersection1.1 - intersection2.1)^2 + (intersection1.2 - intersection2.2)^2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l1682_168204


namespace NUMINAMATH_CALUDE_distance_between_stations_l1682_168215

/-- The distance between two stations given train travel times and speeds -/
theorem distance_between_stations 
  (train1_speed : ℝ) (train1_time : ℝ) 
  (train2_speed : ℝ) (train2_time : ℝ) 
  (h1 : train1_speed = 20)
  (h2 : train1_time = 5)
  (h3 : train2_speed = 25)
  (h4 : train2_time = 4) :
  train1_speed * train1_time + train2_speed * train2_time = 200 := by
  sorry

#check distance_between_stations

end NUMINAMATH_CALUDE_distance_between_stations_l1682_168215


namespace NUMINAMATH_CALUDE_subcommittee_count_l1682_168240

theorem subcommittee_count (n m k t : ℕ) (h1 : n = 12) (h2 : m = 5) (h3 : k = 5) (h4 : t = 5) :
  (Nat.choose n k) - (Nat.choose (n - t) k) = 771 := by
  sorry

end NUMINAMATH_CALUDE_subcommittee_count_l1682_168240


namespace NUMINAMATH_CALUDE_brother_grade_is_two_l1682_168277

structure Brother where
  grade : ℕ

structure Grandmother where
  sneeze : Bool

def tells_truth (b : Brother) (statement : ℕ) : Prop :=
  b.grade = statement

def grandmother_sneezes (g : Grandmother) (b : Brother) (statement : ℕ) : Prop :=
  tells_truth b statement → g.sneeze = true

theorem brother_grade_is_two (b : Brother) (g : Grandmother) :
  grandmother_sneezes g b 5 ∧ g.sneeze = false →
  grandmother_sneezes g b 4 ∧ g.sneeze = true →
  grandmother_sneezes g b 3 ∧ g.sneeze = false →
  b.grade = 2 := by
  sorry

end NUMINAMATH_CALUDE_brother_grade_is_two_l1682_168277


namespace NUMINAMATH_CALUDE_work_completion_time_l1682_168261

/-- Given that:
  - B can do a work in 24 days
  - A and B working together can finish the work in 8 days
  Prove that A can do the work alone in 12 days -/
theorem work_completion_time (work : ℝ) (a_rate b_rate combined_rate : ℝ) :
  work / b_rate = 24 →
  work / combined_rate = 8 →
  combined_rate = a_rate + b_rate →
  work / a_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1682_168261


namespace NUMINAMATH_CALUDE_distance_to_focus_l1682_168291

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = 2 * x^2

-- Define the focus of the parabola
def focus (F : ℝ × ℝ) : Prop := F.2 = 1/4

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2 ∧ P.1 = 1

-- Theorem statement
theorem distance_to_focus (F P : ℝ × ℝ) :
  focus F → point_on_parabola P → |P.1 - F.1| + |P.2 - F.2| = 17/8 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l1682_168291


namespace NUMINAMATH_CALUDE_upgraded_fraction_is_one_ninth_l1682_168211

/-- Represents a satellite with modular units and sensors. -/
structure Satellite :=
  (units : ℕ)
  (non_upgraded_per_unit : ℕ)
  (total_upgraded : ℕ)

/-- The fraction of upgraded sensors on the satellite. -/
def upgraded_fraction (s : Satellite) : ℚ :=
  s.total_upgraded / (s.units * s.non_upgraded_per_unit + s.total_upgraded)

/-- Theorem stating the fraction of upgraded sensors on a specific satellite configuration. -/
theorem upgraded_fraction_is_one_ninth (s : Satellite) 
    (h1 : s.units = 24)
    (h2 : s.non_upgraded_per_unit = s.total_upgraded / 3) : 
    upgraded_fraction s = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_upgraded_fraction_is_one_ninth_l1682_168211


namespace NUMINAMATH_CALUDE_violin_enjoyment_misreporting_l1682_168201

/-- Represents the student population at Peculiar Academy -/
def TotalStudents : ℝ := 100

/-- Fraction of students who enjoy playing the violin -/
def EnjoyViolin : ℝ := 0.4

/-- Fraction of students who do not enjoy playing the violin -/
def DislikeViolin : ℝ := 0.6

/-- Fraction of violin-enjoying students who accurately state they enjoy it -/
def AccurateEnjoy : ℝ := 0.7

/-- Fraction of violin-enjoying students who falsely claim they do not enjoy it -/
def FalseDislike : ℝ := 0.3

/-- Fraction of violin-disliking students who correctly claim they dislike it -/
def AccurateDislike : ℝ := 0.8

/-- Fraction of violin-disliking students who mistakenly say they like it -/
def FalseLike : ℝ := 0.2

theorem violin_enjoyment_misreporting :
  let enjoy_but_say_dislike := EnjoyViolin * FalseDislike * TotalStudents
  let total_say_dislike := EnjoyViolin * FalseDislike * TotalStudents + DislikeViolin * AccurateDislike * TotalStudents
  enjoy_but_say_dislike / total_say_dislike = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_violin_enjoyment_misreporting_l1682_168201


namespace NUMINAMATH_CALUDE_smallest_base_for_perfect_fourth_power_l1682_168235

/-- Given that 5n is a positive integer represented as 777 in base b,
    and n is a perfect fourth power, prove that the smallest positive
    integer b satisfying these conditions is 41. -/
theorem smallest_base_for_perfect_fourth_power (n : ℕ) (b : ℕ) : 
  (5 * n : ℕ) > 0 ∧ 
  (5 * n = 7 * b^2 + 7 * b + 7) ∧
  (∃ (x : ℕ), n = x^4) →
  (∀ (b' : ℕ), b' ≥ 1 ∧ 
    (∃ (n' : ℕ), (5 * n' : ℕ) > 0 ∧ 
      (5 * n' = 7 * b'^2 + 7 * b' + 7) ∧
      (∃ (x : ℕ), n' = x^4)) →
    b' ≥ b) ∧
  b = 41 :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_perfect_fourth_power_l1682_168235


namespace NUMINAMATH_CALUDE_rhombus_area_l1682_168283

/-- A rhombus with perimeter 20cm and diagonals in ratio 4:3 has an area of 24cm². -/
theorem rhombus_area (d₁ d₂ : ℝ) : 
  d₁ > 0 → d₂ > 0 →  -- diagonals are positive
  d₁ / d₂ = 4 / 3 →  -- ratio of diagonals is 4:3
  (d₁^2 + d₂^2) / 2 = 25 →  -- perimeter is 20cm (side length is 5cm)
  d₁ * d₂ / 2 = 24 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l1682_168283


namespace NUMINAMATH_CALUDE_p_equiv_simplified_p_sum_of_squares_of_coefficients_l1682_168265

/-- The polynomial p(x) defined by the given expression -/
def p (x : ℝ) : ℝ := 5 * (x^2 - 3*x + 4) - 8 * (x^3 - x^2 + 2*x - 3)

/-- The simplified form of p(x) -/
def simplified_p (x : ℝ) : ℝ := -8*x^3 + 13*x^2 - 31*x + 44

/-- Theorem stating that p(x) is equivalent to its simplified form -/
theorem p_equiv_simplified_p : p = simplified_p := by sorry

/-- Theorem proving the sum of squares of coefficients of simplified_p is 3130 -/
theorem sum_of_squares_of_coefficients :
  (-8)^2 + 13^2 + (-31)^2 + 44^2 = 3130 := by sorry

end NUMINAMATH_CALUDE_p_equiv_simplified_p_sum_of_squares_of_coefficients_l1682_168265


namespace NUMINAMATH_CALUDE_solutions_of_f_eq_quarter_solution_set_of_f_leq_two_l1682_168269

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else Real.log x / Real.log 4

-- Theorem for the solutions of f(x) = 1/4
theorem solutions_of_f_eq_quarter :
  {x : ℝ | f x = 1/4} = {2, Real.sqrt 2} :=
sorry

-- Theorem for the solution set of f(x) ≤ 2
theorem solution_set_of_f_leq_two :
  {x : ℝ | f x ≤ 2} = Set.Icc (-1) 16 :=
sorry

end NUMINAMATH_CALUDE_solutions_of_f_eq_quarter_solution_set_of_f_leq_two_l1682_168269


namespace NUMINAMATH_CALUDE_two_integers_make_fraction_integer_l1682_168218

theorem two_integers_make_fraction_integer : 
  ∃! (S : Finset ℕ), 
    (∀ m ∈ S, (1750 : ℕ) ∣ (m^2 - 4)) ∧ 
    (∀ m : ℕ, m > 0 → (1750 : ℕ) ∣ (m^2 - 4) → m ∈ S) ∧ 
    S.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_integers_make_fraction_integer_l1682_168218


namespace NUMINAMATH_CALUDE_smallest_number_l1682_168263

theorem smallest_number (s : Set ℝ) (hs : s = {0, -2, 1, (1/2 : ℝ)}) :
  ∃ x ∈ s, ∀ y ∈ s, x ≤ y ∧ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1682_168263


namespace NUMINAMATH_CALUDE_parabola_zeros_difference_l1682_168208

/-- Represents a quadratic function ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the y-value for a given x in a quadratic function -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Finds the zeros of a quadratic function -/
def QuadraticFunction.zeros (f : QuadraticFunction) : Set ℝ :=
  {x : ℝ | f.eval x = 0}

theorem parabola_zeros_difference (f : QuadraticFunction) :
  f.eval 3 = -9 →  -- vertex at (3, -9)
  f.eval 5 = 7 →   -- passes through (5, 7)
  ∃ m n, m ∈ f.zeros ∧ n ∈ f.zeros ∧ m > n ∧ m - n = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_zeros_difference_l1682_168208


namespace NUMINAMATH_CALUDE_special_circle_equation_l1682_168239

/-- A circle with center on y = x, passing through origin, and chord of length 2 on x-axis -/
structure SpecialCircle where
  center : ℝ × ℝ
  center_on_line : center.2 = center.1
  passes_origin : (center.1 ^ 2 + center.2 ^ 2) = 2 * center.1 ^ 2
  chord_length : ∃ x : ℝ, (x - center.1) ^ 2 + center.2 ^ 2 = 2 * center.1 ^ 2 ∧ 
                           ((x - 1) - center.1) ^ 2 + center.2 ^ 2 = 2 * center.1 ^ 2

theorem special_circle_equation (c : SpecialCircle) : 
  (∀ x y : ℝ, (x - 1) ^ 2 + (y - 1) ^ 2 = 2) ∨ 
  (∀ x y : ℝ, (x + 1) ^ 2 + (y + 1) ^ 2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_special_circle_equation_l1682_168239


namespace NUMINAMATH_CALUDE_safflower_percentage_in_brand_b_l1682_168292

/-- Represents the composition of a birdseed brand -/
structure BirdseedBrand where
  millet : ℝ
  sunflower : ℝ
  safflower : ℝ

/-- Represents the mix of two birdseed brands -/
structure BirdseedMix where
  brandA : BirdseedBrand
  brandB : BirdseedBrand
  proportionA : ℝ

/-- The theorem stating the percentage of safflower in Brand B -/
theorem safflower_percentage_in_brand_b 
  (brandA : BirdseedBrand)
  (brandB : BirdseedBrand)
  (mix : BirdseedMix)
  (h1 : brandA.millet = 0.4)
  (h2 : brandA.sunflower = 0.6)
  (h3 : brandB.millet = 0.65)
  (h4 : mix.proportionA = 0.6)
  (h5 : mix.proportionA * brandA.millet + (1 - mix.proportionA) * brandB.millet = 0.5)
  : brandB.safflower = 0.35 := by
  sorry


end NUMINAMATH_CALUDE_safflower_percentage_in_brand_b_l1682_168292


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt3_plus_1_l1682_168225

theorem rationalize_denominator_sqrt3_plus_1 :
  (1 : ℝ) / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt3_plus_1_l1682_168225


namespace NUMINAMATH_CALUDE_distance_time_not_correlation_l1682_168288

/-- Represents a relationship between two variables -/
inductive Relationship
  | Correlation
  | Functional

/-- Represents the relationship between distance and time for a vehicle moving at constant speed -/
def distance_time_relationship : Relationship := sorry

/-- Theorem stating that the distance-time relationship for a vehicle moving at constant speed is functional, not correlational -/
theorem distance_time_not_correlation :
  distance_time_relationship = Relationship.Functional :=
sorry

end NUMINAMATH_CALUDE_distance_time_not_correlation_l1682_168288


namespace NUMINAMATH_CALUDE_concert_ticket_problem_l1682_168290

/-- Represents the number of student tickets sold -/
def student_tickets : ℕ := sorry

/-- Represents the number of non-student tickets sold -/
def non_student_tickets : ℕ := sorry

/-- The price of a student ticket in dollars -/
def student_price : ℕ := 9

/-- The price of a non-student ticket in dollars -/
def non_student_price : ℕ := 11

/-- The total number of tickets sold -/
def total_tickets : ℕ := 2000

/-- The total revenue from ticket sales in dollars -/
def total_revenue : ℕ := 20960

theorem concert_ticket_problem :
  (student_tickets + non_student_tickets = total_tickets) ∧
  (student_tickets * student_price + non_student_tickets * non_student_price = total_revenue) →
  student_tickets = 520 := by sorry

end NUMINAMATH_CALUDE_concert_ticket_problem_l1682_168290


namespace NUMINAMATH_CALUDE_units_digit_factorial_product_l1682_168299

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_factorial_product :
  units_digit (factorial 1 * factorial 2 * factorial 3 * factorial 4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_factorial_product_l1682_168299


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1682_168257

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 2024) + (y + 1/x) * (y + 1/x - 2024) + 2024 ≥ -2050208 :=
by sorry

theorem min_value_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧
    (x + 1/y) * (x + 1/y - 2024) + (y + 1/x) * (y + 1/x - 2024) + 2024 = -2050208 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1682_168257


namespace NUMINAMATH_CALUDE_shaded_to_white_area_ratio_l1682_168251

theorem shaded_to_white_area_ratio : 
  ∀ (quarter_shaded_triangles quarter_white_triangles : ℕ) 
    (total_quarters : ℕ) 
    (shaded_area white_area : ℝ),
  quarter_shaded_triangles = 5 →
  quarter_white_triangles = 3 →
  total_quarters = 4 →
  shaded_area = (quarter_shaded_triangles * total_quarters : ℝ) →
  white_area = (quarter_white_triangles * total_quarters : ℝ) →
  shaded_area / white_area = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_shaded_to_white_area_ratio_l1682_168251


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_to_17_mod_29_l1682_168296

theorem largest_five_digit_congruent_to_17_mod_29 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n ≡ 17 [MOD 29] → n ≤ 99982 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_to_17_mod_29_l1682_168296


namespace NUMINAMATH_CALUDE_vector_sum_triangle_l1682_168266

-- Define a triangle ABC in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define vector addition
def vectorAdd (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define vector subtraction (used to represent directed edges)
def vectorSub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- Theorem statement
theorem vector_sum_triangle (t : Triangle) : 
  vectorAdd (vectorAdd (vectorSub t.B t.A) (vectorSub t.C t.B)) (vectorSub t.A t.C) = (0, 0) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_triangle_l1682_168266


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1682_168258

theorem necessary_but_not_sufficient (p q : Prop) :
  (p ∧ q → p) ∧ ¬(p → p ∧ q) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1682_168258


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l1682_168233

/-- The parabola is defined by the equation y = -x^2 + 3x - 4 -/
def parabola (x y : ℝ) : Prop := y = -x^2 + 3*x - 4

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

theorem parabola_y_axis_intersection :
  ∃ (x y : ℝ), parabola x y ∧ on_y_axis x y ∧ x = 0 ∧ y = -4 :=
sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l1682_168233


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l1682_168281

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed in the second hour. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (h1 : speed_first_hour = 98)
  (h2 : average_speed = 79) :
  let speed_second_hour := 2 * average_speed - speed_first_hour
  speed_second_hour = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l1682_168281


namespace NUMINAMATH_CALUDE_remainder_equality_l1682_168255

theorem remainder_equality (P P' Q D : ℕ) (R R' S : ℕ) 
  (h1 : P > P') (h2 : P' > Q) 
  (h3 : R = P % D) (h4 : R' = P' % D) (h5 : S = Q % D) : 
  (P * P' * Q) % D = (R * R' * S) % D :=
sorry

end NUMINAMATH_CALUDE_remainder_equality_l1682_168255


namespace NUMINAMATH_CALUDE_scores_statistics_l1682_168210

def scores : List ℕ := [85, 95, 85, 80, 80, 85]

/-- The mode of a list of natural numbers -/
def mode (l : List ℕ) : ℕ := sorry

/-- The mean of a list of natural numbers -/
def mean (l : List ℕ) : ℚ := sorry

/-- The median of a list of natural numbers -/
def median (l : List ℕ) : ℚ := sorry

/-- The range of a list of natural numbers -/
def range (l : List ℕ) : ℕ := sorry

theorem scores_statistics :
  mode scores = 85 ∧
  mean scores = 85 ∧
  median scores = 85 ∧
  range scores = 15 := by sorry

end NUMINAMATH_CALUDE_scores_statistics_l1682_168210


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l1682_168270

/-- The breadth of a rectangular plot given its area and length-breadth relationship -/
theorem rectangular_plot_breadth (area : ℝ) (length_ratio : ℝ) : 
  area = 360 → length_ratio = 0.75 → ∃ breadth : ℝ, 
    area = (length_ratio * breadth) * breadth ∧ breadth = 4 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l1682_168270


namespace NUMINAMATH_CALUDE_ned_earnings_l1682_168280

/-- Calculates the total earnings from selling working video games -/
def calculate_earnings (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : ℕ :=
  (total_games - non_working_games) * price_per_game

/-- Proves that Ned's earnings from selling his working video games is $63 -/
theorem ned_earnings :
  calculate_earnings 15 6 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_ned_earnings_l1682_168280


namespace NUMINAMATH_CALUDE_fraction_arithmetic_l1682_168254

theorem fraction_arithmetic : (1/2 - 1/6) / (1/6009 : ℚ) = 2003 := by
  sorry

end NUMINAMATH_CALUDE_fraction_arithmetic_l1682_168254


namespace NUMINAMATH_CALUDE_f_properties_l1682_168285

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Theorem statement
theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ ≥ 1 ∧ x₂ ≥ 1 ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, x ≥ 2 ∧ x ≤ 5 → f x ≤ 0) ∧
  (∀ x : ℝ, x ≥ 2 ∧ x ≤ 5 → f x ≥ -15) ∧
  (∃ x : ℝ, x ≥ 2 ∧ x ≤ 5 ∧ f x = 0) ∧
  (∃ x : ℝ, x ≥ 2 ∧ x ≤ 5 ∧ f x = -15) :=
by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1682_168285


namespace NUMINAMATH_CALUDE_man_speed_in_still_water_l1682_168221

/-- Represents the speed of a swimmer in still water and the speed of the stream -/
structure SwimmingSpeed where
  manSpeed : ℝ  -- Speed of the man in still water
  streamSpeed : ℝ  -- Speed of the stream

/-- Calculates the effective speed for downstream swimming -/
def downstreamSpeed (s : SwimmingSpeed) : ℝ := s.manSpeed + s.streamSpeed

/-- Calculates the effective speed for upstream swimming -/
def upstreamSpeed (s : SwimmingSpeed) : ℝ := s.manSpeed - s.streamSpeed

/-- Theorem stating that given the conditions, the man's speed in still water is 5 km/h -/
theorem man_speed_in_still_water :
  ∀ s : SwimmingSpeed,
    (downstreamSpeed s * 4 = 24) →  -- Downstream condition
    (upstreamSpeed s * 5 = 20) →    -- Upstream condition
    s.manSpeed = 5 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_in_still_water_l1682_168221


namespace NUMINAMATH_CALUDE_initial_kittens_count_l1682_168287

/-- The number of kittens Tim gave to Jessica -/
def kittens_to_jessica : ℕ := 3

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara : ℕ := 6

/-- The number of kittens Tim has left -/
def kittens_left : ℕ := 9

/-- The initial number of kittens Tim had -/
def initial_kittens : ℕ := kittens_to_jessica + kittens_to_sara + kittens_left

theorem initial_kittens_count : initial_kittens = 18 := by
  sorry

end NUMINAMATH_CALUDE_initial_kittens_count_l1682_168287


namespace NUMINAMATH_CALUDE_sqrt_two_squared_l1682_168213

theorem sqrt_two_squared : (Real.sqrt 2) ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_squared_l1682_168213


namespace NUMINAMATH_CALUDE_min_value_of_sum_l1682_168227

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  x + 3 * y ≥ 4 + 8 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧ 
    x₀ + 3 * y₀ = 4 + 8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l1682_168227


namespace NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l1682_168241

theorem no_real_solution_for_log_equation :
  ¬∃ (x : ℝ), Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 4*x - 15) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_for_log_equation_l1682_168241


namespace NUMINAMATH_CALUDE_price_change_theorem_l1682_168245

theorem price_change_theorem (initial_price : ℝ) (h : initial_price > 0) :
  let egg_price_new := initial_price * (1 - 0.02)
  let apple_price_new := initial_price * (1 + 0.10)
  let total_price_old := 2 * initial_price
  let total_price_new := egg_price_new + apple_price_new
  let price_increase := total_price_new - total_price_old
  let percentage_increase := price_increase / total_price_old * 100
  percentage_increase = 4 := by sorry

end NUMINAMATH_CALUDE_price_change_theorem_l1682_168245


namespace NUMINAMATH_CALUDE_carol_final_score_is_negative_nineteen_l1682_168284

-- Define the scores and multipliers for each round
def first_round_score : Int := 17
def second_round_base_score : Int := 6
def second_round_multiplier : Int := 2
def last_round_base_loss : Int := 16
def last_round_multiplier : Int := 3

-- Define Carol's final score
def carol_final_score : Int := 
  first_round_score + 
  (second_round_base_score * second_round_multiplier) - 
  (last_round_base_loss * last_round_multiplier)

-- Theorem to prove Carol's final score
theorem carol_final_score_is_negative_nineteen : 
  carol_final_score = -19 := by
  sorry

end NUMINAMATH_CALUDE_carol_final_score_is_negative_nineteen_l1682_168284


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1682_168216

theorem consecutive_integers_sum (x : ℤ) : 
  x * (x + 1) * (x + 2) = 384 → x + (x + 1) + (x + 2) = 24 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1682_168216


namespace NUMINAMATH_CALUDE_calculation_proof_l1682_168231

theorem calculation_proof : (3752 / (39 * 2) + 5030 / (39 * 10) : ℚ) = 61 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1682_168231


namespace NUMINAMATH_CALUDE_min_score_for_average_increase_l1682_168298

/-- Given 4 tests with an average score of 68, prove that a score of at least 78 on the 5th test 
    is necessary to achieve an average score of more than 70 over all 5 tests. -/
theorem min_score_for_average_increase (current_tests : Nat) (current_average : ℝ) 
  (target_average : ℝ) (min_score : ℝ) : 
  current_tests = 4 → 
  current_average = 68 → 
  target_average > 70 → 
  min_score ≥ 78 → 
  (current_tests * current_average + min_score) / (current_tests + 1) > target_average :=
by sorry

end NUMINAMATH_CALUDE_min_score_for_average_increase_l1682_168298


namespace NUMINAMATH_CALUDE_intersection_M_N_l1682_168236

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| > 1}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1682_168236


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1682_168250

theorem fraction_multiplication : (-1/6 + 3/4 - 5/12) * 48 = 8 := by sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1682_168250


namespace NUMINAMATH_CALUDE_constant_k_equality_l1682_168200

theorem constant_k_equality (x : ℝ) : 
  -x^2 - (-17 + 11)*x - 8 = -(x - 2)*(x - 4) := by
  sorry

end NUMINAMATH_CALUDE_constant_k_equality_l1682_168200


namespace NUMINAMATH_CALUDE_finance_charge_rate_example_l1682_168206

/-- Given an original balance and a total payment, calculate the finance charge rate. -/
def finance_charge_rate (original_balance total_payment : ℚ) : ℚ :=
  (total_payment - original_balance) / original_balance * 100

/-- Theorem: The finance charge rate is 2% when the original balance is $150 and the total payment is $153. -/
theorem finance_charge_rate_example :
  finance_charge_rate 150 153 = 2 := by
  sorry

end NUMINAMATH_CALUDE_finance_charge_rate_example_l1682_168206


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l1682_168234

theorem average_of_remaining_numbers 
  (total : ℕ) 
  (subset : ℕ) 
  (remaining : ℕ) 
  (total_sum : ℚ) 
  (subset_sum : ℚ) 
  (h1 : total = 5) 
  (h2 : subset = 3) 
  (h3 : remaining = total - subset) 
  (h4 : total_sum / total = 6) 
  (h5 : subset_sum / subset = 4) : 
  (total_sum - subset_sum) / remaining = 9 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l1682_168234


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1682_168295

def A : Set ℤ := {0, 1}
def B : Set ℤ := {-1, 1}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1682_168295


namespace NUMINAMATH_CALUDE_min_sum_of_squares_with_diff_91_l1682_168259

theorem min_sum_of_squares_with_diff_91 :
  ∃ (x y : ℕ), x > y ∧ x^2 - y^2 = 91 ∧
  ∀ (a b : ℕ), a > b → a^2 - b^2 = 91 → x^2 + y^2 ≤ a^2 + b^2 ∧
  x^2 + y^2 = 109 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_with_diff_91_l1682_168259


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1682_168219

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y + 1) = x * f y + 2) →
  (∀ x : ℝ, f x = 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1682_168219


namespace NUMINAMATH_CALUDE_sphere_volume_constant_l1682_168252

theorem sphere_volume_constant (cube_side : Real) (K : Real) : 
  cube_side = 3 →
  (4 / 3 * Real.pi * (((6 * cube_side^2) / (4 * Real.pi))^(3/2))) = K * Real.sqrt 6 / Real.sqrt Real.pi →
  K = 54 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_constant_l1682_168252


namespace NUMINAMATH_CALUDE_sqrt_sum_eq_8_implies_product_l1682_168207

theorem sqrt_sum_eq_8_implies_product (x : ℝ) :
  Real.sqrt (8 + x) + Real.sqrt (25 - x) = 8 →
  (8 + x) * (25 - x) = 961 / 4 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_eq_8_implies_product_l1682_168207


namespace NUMINAMATH_CALUDE_chebyshev_birth_year_l1682_168289

/-- Represents a year in the 19th century -/
structure Year1800s where
  tens : Nat
  units : Nat

/-- Checks if the given year satisfies all the conditions for P.L. Chebyshev's birth year -/
def is_chebyshev_birth_year (y : Year1800s) : Prop :=
  -- Sum of digits in hundreds and thousands (1 + 8 = 9) is 3 times the sum of digits in tens and units
  9 = 3 * (y.tens + y.units) ∧
  -- Digit in tens place is greater than digit in units place
  y.tens > y.units ∧
  -- Born and died in the same century (19th century)
  1800 + 10 * y.tens + y.units + 73 < 1900

/-- Theorem stating that 1821 is the unique year satisfying all conditions -/
theorem chebyshev_birth_year : 
  ∃! (y : Year1800s), is_chebyshev_birth_year y ∧ 1800 + 10 * y.tens + y.units = 1821 :=
sorry

end NUMINAMATH_CALUDE_chebyshev_birth_year_l1682_168289


namespace NUMINAMATH_CALUDE_largest_divisor_of_p_cubed_minus_p_l1682_168262

theorem largest_divisor_of_p_cubed_minus_p (p : ℕ) (hp : Prime p) (hp_ge_5 : p ≥ 5) :
  (∃ (k : ℕ), k * 12 = p^3 - p) ∧
  (∀ (d : ℕ), d > 12 → ¬(∀ (q : ℕ), Prime q → q ≥ 5 → ∃ (k : ℕ), k * d = q^3 - q)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_p_cubed_minus_p_l1682_168262


namespace NUMINAMATH_CALUDE_greatest_integer_x_cube_less_than_15_l1682_168224

theorem greatest_integer_x_cube_less_than_15 :
  ∃ (x : ℕ), x > 0 ∧ (x^6 / x^3 : ℚ) < 15 ∧ ∀ (y : ℕ), y > x → (y^6 / y^3 : ℚ) ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_x_cube_less_than_15_l1682_168224


namespace NUMINAMATH_CALUDE_class_size_class_size_problem_l1682_168203

theorem class_size (total_stickers : ℕ) (num_friends : ℕ) (stickers_per_friend : ℕ) 
  (stickers_per_other : ℕ) (leftover_stickers : ℕ) : ℕ :=
  let stickers_to_friends := num_friends * stickers_per_friend
  let remaining_stickers := total_stickers - stickers_to_friends - leftover_stickers
  let other_students := remaining_stickers / stickers_per_other
  other_students + num_friends + 1

theorem class_size_problem : 
  class_size 50 5 4 2 8 = 17 := by
  sorry

end NUMINAMATH_CALUDE_class_size_class_size_problem_l1682_168203


namespace NUMINAMATH_CALUDE_impossible_non_eleven_multiple_l1682_168275

/-- Represents a 5x5 board where each cell can be increased along with its adjacent cells. -/
structure Board :=
  (cells : Matrix (Fin 5) (Fin 5) ℕ)

/-- The operation of increasing a cell and its adjacent cells by 1. -/
def increase_cell (b : Board) (i j : Fin 5) : Board := sorry

/-- Checks if all cells in the board have the same value. -/
def all_cells_equal (b : Board) (s : ℕ) : Prop := sorry

/-- Main theorem: It's impossible to obtain a number not divisible by 11 in all cells. -/
theorem impossible_non_eleven_multiple (s : ℕ) (h : ¬ 11 ∣ s) : 
  ¬ ∃ (b : Board), all_cells_equal b s :=
sorry

end NUMINAMATH_CALUDE_impossible_non_eleven_multiple_l1682_168275


namespace NUMINAMATH_CALUDE_deceased_cannot_marry_l1682_168248

-- Define a person
structure Person where
  alive : Bool

-- Define marriage as a relation between two people
def canMarry (p1 p2 : Person) : Prop := p1.alive ∧ p2.alive

-- Theorem: A deceased person cannot marry anyone
theorem deceased_cannot_marry (p1 p2 : Person) : 
  ¬p1.alive → ¬(canMarry p1 p2) := by
  sorry

#check deceased_cannot_marry

end NUMINAMATH_CALUDE_deceased_cannot_marry_l1682_168248


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1682_168279

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
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
  given_line.a = 1 ∧ given_line.b = -2 ∧ given_line.c = 1 ∧
  point.x = 2 ∧ point.y = -1 →
  ∃ (l : Line), l.perpendicular given_line ∧ point.liesOn l ∧
  l.a = 2 ∧ l.b = 1 ∧ l.c = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1682_168279


namespace NUMINAMATH_CALUDE_base3_20202_equals_182_l1682_168220

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def base3_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

/-- The theorem stating that the base-3 number 20202 is equal to 182 in base 10 -/
theorem base3_20202_equals_182 : base3_to_base10 [2, 0, 2, 0, 2] = 182 := by
  sorry

#eval base3_to_base10 [2, 0, 2, 0, 2]

end NUMINAMATH_CALUDE_base3_20202_equals_182_l1682_168220


namespace NUMINAMATH_CALUDE_two_marbles_in_two_boxes_proof_l1682_168230

/-- The number of ways to choose 2 marbles out of 3 distinct marbles 
    and place them in 2 indistinguishable boxes -/
def two_marbles_in_two_boxes : ℕ := 3

/-- The number of distinct marbles -/
def total_marbles : ℕ := 3

/-- The number of marbles to be chosen -/
def chosen_marbles : ℕ := 2

/-- The number of boxes -/
def num_boxes : ℕ := 2

/-- Boxes are indistinguishable -/
def boxes_indistinguishable : Prop := True

theorem two_marbles_in_two_boxes_proof :
  two_marbles_in_two_boxes = (total_marbles.choose chosen_marbles) :=
by sorry

end NUMINAMATH_CALUDE_two_marbles_in_two_boxes_proof_l1682_168230


namespace NUMINAMATH_CALUDE_angle_rotation_l1682_168297

def first_quadrant (α : Real) : Prop :=
  0 < α ∧ α < Real.pi / 2

def third_quadrant (α : Real) : Prop :=
  Real.pi < α ∧ α < 3 * Real.pi / 2

theorem angle_rotation (α : Real) :
  first_quadrant α → third_quadrant (α + Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_angle_rotation_l1682_168297


namespace NUMINAMATH_CALUDE_green_pill_cost_proof_l1682_168253

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℝ := 21.5

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℝ := green_pill_cost - 2

/-- The number of days for which pills are taken -/
def days : ℕ := 18

/-- The total cost of all pills over the given period -/
def total_cost : ℝ := 738

theorem green_pill_cost_proof :
  (green_pill_cost + pink_pill_cost) * days = total_cost ∧
  green_pill_cost = pink_pill_cost + 2 ∧
  green_pill_cost = 21.5 :=
sorry

end NUMINAMATH_CALUDE_green_pill_cost_proof_l1682_168253


namespace NUMINAMATH_CALUDE_company_donation_problem_l1682_168242

theorem company_donation_problem (donation_A donation_B : ℕ) 
  (average_difference : ℕ) (percentage_difference : ℚ) :
  donation_A = 60000 →
  donation_B = 60000 →
  average_difference = 40 →
  percentage_difference = 1/5 →
  ∃ (people_A people_B : ℕ),
    people_A = (1 + percentage_difference) * people_B ∧
    (donation_B : ℚ) / people_B - (donation_A : ℚ) / people_A = average_difference ∧
    people_A = 300 ∧
    people_B = 250 := by
  sorry

end NUMINAMATH_CALUDE_company_donation_problem_l1682_168242


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1682_168222

/-- Two vectors in R² are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Given vectors a and b, prove that if they are parallel, then x = 6 -/
theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  are_parallel a b → x = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1682_168222


namespace NUMINAMATH_CALUDE_profit_difference_theorem_l1682_168212

/-- Represents the profit distribution in a business partnership --/
structure ProfitDistribution where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  b_profit : ℕ

/-- Calculates the difference between A's and C's profit shares --/
def profit_difference (pd : ProfitDistribution) : ℕ :=
  let total_parts := pd.a_investment + pd.b_investment + pd.c_investment
  let part_value := pd.b_profit * total_parts / pd.b_investment
  let a_profit := part_value * pd.a_investment / total_parts
  let c_profit := part_value * pd.c_investment / total_parts
  c_profit - a_profit

/-- Theorem stating the difference between A's and C's profit shares --/
theorem profit_difference_theorem (pd : ProfitDistribution) 
  (h1 : pd.a_investment = 8000)
  (h2 : pd.b_investment = 10000)
  (h3 : pd.c_investment = 12000)
  (h4 : pd.b_profit = 1400) :
  profit_difference pd = 560 := by
  sorry


end NUMINAMATH_CALUDE_profit_difference_theorem_l1682_168212


namespace NUMINAMATH_CALUDE_sampling_survey_more_appropriate_for_city_air_quality_l1682_168202

-- Define the city and survey types
def City : Type := Unit
def ComprehensiveSurvey : Type := Unit
def SamplingSurvey : Type := Unit

-- Define the properties of the city and surveys
def has_vast_area (c : City) : Prop := sorry
def has_varying_conditions (c : City) : Prop := sorry
def is_comprehensive (s : ComprehensiveSurvey) : Prop := sorry
def is_strategically_sampled (s : SamplingSurvey) : Prop := sorry

-- Define the concept of feasibility and appropriateness
def is_feasible (c : City) (s : ComprehensiveSurvey) : Prop := sorry
def is_appropriate (c : City) (s : SamplingSurvey) : Prop := sorry

-- Theorem stating that sampling survey is more appropriate for air quality testing in a city
theorem sampling_survey_more_appropriate_for_city_air_quality 
  (c : City) (comp_survey : ComprehensiveSurvey) (samp_survey : SamplingSurvey) :
  has_vast_area c →
  has_varying_conditions c →
  is_comprehensive comp_survey →
  is_strategically_sampled samp_survey →
  ¬(is_feasible c comp_survey) →
  is_appropriate c samp_survey :=
by sorry

end NUMINAMATH_CALUDE_sampling_survey_more_appropriate_for_city_air_quality_l1682_168202


namespace NUMINAMATH_CALUDE_problem_solution_l1682_168244

theorem problem_solution (x y z w : ℝ) 
  (h1 : x * w > 0)
  (h2 : y * z > 0)
  (h3 : 1 / x + 1 / w = 20)
  (h4 : 1 / y + 1 / z = 25)
  (h5 : 1 / (x * w) = 6)
  (h6 : 1 / (y * z) = 8) :
  (x + y) / (z + w) = 155 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1682_168244


namespace NUMINAMATH_CALUDE_angle_measure_l1682_168294

theorem angle_measure (α : Real) (h : α > 0 ∧ α < π/2) :
  1 / Real.sqrt (Real.tan (α/2)) = Real.sqrt (2 * Real.sqrt 3) * Real.sqrt (Real.tan (π/18)) + Real.sqrt (Real.tan (α/2)) →
  α = π/3.6 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l1682_168294


namespace NUMINAMATH_CALUDE_weight_loss_program_result_l1682_168232

/-- Calculates the final weight after a weight loss program -/
def final_weight (initial_weight : ℕ) (loss_rate1 : ℕ) (weeks1 : ℕ) (loss_rate2 : ℕ) (weeks2 : ℕ) : ℕ :=
  initial_weight - (loss_rate1 * weeks1 + loss_rate2 * weeks2)

/-- Theorem stating that the final weight after the given weight loss program is 222 pounds -/
theorem weight_loss_program_result :
  final_weight 250 3 4 2 8 = 222 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_program_result_l1682_168232


namespace NUMINAMATH_CALUDE_jelly_cost_for_sandwiches_l1682_168273

/-- The cost of jelly for N sandwiches -/
def jelly_cost (N B J : ℕ+) : ℚ :=
  (N * J * 7 : ℚ) / 100

/-- The total cost of peanut butter and jelly for N sandwiches -/
def total_cost (N B J : ℕ+) : ℚ :=
  (N * (3 * B + 7 * J) : ℚ) / 100

theorem jelly_cost_for_sandwiches
  (N B J : ℕ+)
  (h1 : total_cost N B J = 252 / 100)
  (h2 : N > 1) :
  jelly_cost N B J = 168 / 100 :=
by sorry

end NUMINAMATH_CALUDE_jelly_cost_for_sandwiches_l1682_168273


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1682_168246

theorem line_intersects_circle (m : ℝ) (h_m : 0 < m ∧ m < 4/3) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 + m*x₁ + m^2 - m = 0) ∧ 
  (x₂^2 + m*x₂ + m^2 - m = 0) ∧
  ∃ (x y : ℝ), 
    (m*x + y + m^2 - m = 0) ∧ 
    ((x - 1)^2 + (y + 1)^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1682_168246


namespace NUMINAMATH_CALUDE_triangle_area_l1682_168271

-- Define the curve
def curve (x : ℝ) : ℝ := (x - 4)^2 * (x + 3)

-- Define the x-intercepts
def x_intercept1 : ℝ := 4
def x_intercept2 : ℝ := -3

-- Define the y-intercept
def y_intercept : ℝ := curve 0

-- Theorem statement
theorem triangle_area : 
  let base := x_intercept1 - x_intercept2
  let height := y_intercept
  (1/2 : ℝ) * base * height = 168 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1682_168271


namespace NUMINAMATH_CALUDE_jack_school_time_l1682_168223

/-- Given information about Dave and Jack's walking speeds and Dave's time to school,
    prove that Jack takes 18 minutes to reach the same school. -/
theorem jack_school_time (dave_steps_per_min : ℕ) (dave_step_length : ℕ) (dave_time : ℕ)
                         (jack_steps_per_min : ℕ) (jack_step_length : ℕ) :
  dave_steps_per_min = 90 →
  dave_step_length = 75 →
  dave_time = 16 →
  jack_steps_per_min = 100 →
  jack_step_length = 60 →
  (dave_steps_per_min * dave_step_length * dave_time) / (jack_steps_per_min * jack_step_length) = 18 :=
by sorry

end NUMINAMATH_CALUDE_jack_school_time_l1682_168223
