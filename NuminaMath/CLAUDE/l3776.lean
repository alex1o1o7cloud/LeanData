import Mathlib

namespace abs_fraction_inequality_l3776_377687

theorem abs_fraction_inequality (x : ℝ) :
  |((3 - x) / 4)| < 1 ↔ 2 < x ∧ x < 7 := by sorry

end abs_fraction_inequality_l3776_377687


namespace complex_fraction_equals_i_l3776_377661

theorem complex_fraction_equals_i (m n : ℝ) (h : m + Complex.I = 1 + n * Complex.I) :
  (m + n * Complex.I) / (m - n * Complex.I) = Complex.I := by
  sorry

end complex_fraction_equals_i_l3776_377661


namespace line_slope_through_points_l3776_377653

/-- The slope of a line passing through points (1, 0) and (2, √3) is √3. -/
theorem line_slope_through_points : 
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (2, Real.sqrt 3)
  (B.2 - A.2) / (B.1 - A.1) = Real.sqrt 3 := by
  sorry

end line_slope_through_points_l3776_377653


namespace log_xy_value_l3776_377650

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- Define the theorem
theorem log_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : log (x * y^2) = 2) (h2 : log (x^3 * y) = 3) : 
  log (x * y) = 7/5 := by
  sorry


end log_xy_value_l3776_377650


namespace min_sum_given_product_l3776_377682

theorem min_sum_given_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y - (x + y) = 1) :
  x + y ≥ 2 + 2 * Real.sqrt 2 := by
  sorry

end min_sum_given_product_l3776_377682


namespace puppies_sold_l3776_377612

theorem puppies_sold (initial_puppies initial_kittens kittens_sold remaining_pets : ℕ) :
  initial_puppies = 7 →
  initial_kittens = 6 →
  kittens_sold = 3 →
  remaining_pets = 8 →
  initial_puppies + initial_kittens - kittens_sold - remaining_pets = 2 := by
  sorry

#check puppies_sold

end puppies_sold_l3776_377612


namespace arithmetic_sequence_fifth_term_l3776_377613

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 6)
  (h_a3 : a 3 = 2) :
  a 5 = -2 := by
sorry

end arithmetic_sequence_fifth_term_l3776_377613


namespace x_lt_5_necessary_not_sufficient_l3776_377631

theorem x_lt_5_necessary_not_sufficient :
  (∀ x : ℝ, -2 < x ∧ x < 4 → x < 5) ∧
  (∃ x : ℝ, x < 5 ∧ ¬(-2 < x ∧ x < 4)) :=
by sorry

end x_lt_5_necessary_not_sufficient_l3776_377631


namespace absolute_value_and_quadratic_equation_l3776_377620

theorem absolute_value_and_quadratic_equation :
  ∀ (b c : ℝ),
  (∀ x : ℝ, |x - 4| = 3 ↔ x^2 + b*x + c = 0) →
  b = -8 ∧ c = 7 := by
sorry

end absolute_value_and_quadratic_equation_l3776_377620


namespace angle_A_value_l3776_377619

noncomputable section

-- Define the triangle ABC
variable (A B C : Real)  -- Angles
variable (a b c : Real)  -- Side lengths

-- Define the conditions
axiom triangle : A + B + C = Real.pi  -- Sum of angles in a triangle
axiom side_a : a = Real.sqrt 3
axiom side_b : b = Real.sqrt 2
axiom angle_B : B = Real.pi / 4  -- 45° in radians

-- State the theorem
theorem angle_A_value : 
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 := by sorry

end angle_A_value_l3776_377619


namespace simplify_expression_l3776_377685

theorem simplify_expression (n : ℕ) : (3^(n+3) - 3*(3^n)) / (3*(3^(n+2))) = 8/9 := by
  sorry

end simplify_expression_l3776_377685


namespace gcd_lcm_sum_15_9_l3776_377689

theorem gcd_lcm_sum_15_9 : 
  Nat.gcd 15 9 + 2 * Nat.lcm 15 9 = 93 := by sorry

end gcd_lcm_sum_15_9_l3776_377689


namespace min_value_constraint_l3776_377693

theorem min_value_constraint (a b : ℝ) (h : a + b^2 = 2) :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x y : ℝ), x + y^2 = 2 → a^2 + 6*b^2 ≥ m :=
by sorry

end min_value_constraint_l3776_377693


namespace contractor_male_wage_l3776_377658

/-- Represents the daily wage structure and worker composition of a building contractor --/
structure ContractorData where
  male_workers : ℕ
  female_workers : ℕ
  child_workers : ℕ
  female_wage : ℕ
  child_wage : ℕ
  average_wage : ℕ

/-- Calculates the daily wage for male workers given the contractor's data --/
def male_wage (data : ContractorData) : ℕ :=
  let total_workers := data.male_workers + data.female_workers + data.child_workers
  let total_wage := total_workers * data.average_wage
  let female_total := data.female_workers * data.female_wage
  let child_total := data.child_workers * data.child_wage
  (total_wage - female_total - child_total) / data.male_workers

/-- Theorem stating that for the given contractor data, the male wage is 25 --/
theorem contractor_male_wage :
  male_wage {
    male_workers := 20,
    female_workers := 15,
    child_workers := 5,
    female_wage := 20,
    child_wage := 8,
    average_wage := 21
  } = 25 := by
  sorry


end contractor_male_wage_l3776_377658


namespace first_digit_powers_of_3_and_7_l3776_377670

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def first_digit (n : ℕ) : ℕ :=
  if n < 10 then n else first_digit (n / 10)

theorem first_digit_powers_of_3_and_7 :
  ∃ (m n : ℕ), is_three_digit (3^m) ∧ is_three_digit (7^n) ∧ 
  first_digit (3^m) = first_digit (7^n) ∧
  first_digit (3^m) = 3 ∧
  ∀ (k : ℕ), k ≠ 3 → 
    ¬(∃ (p q : ℕ), is_three_digit (3^p) ∧ is_three_digit (7^q) ∧ 
    first_digit (3^p) = first_digit (7^q) ∧ first_digit (3^p) = k) :=
by sorry

end first_digit_powers_of_3_and_7_l3776_377670


namespace arctan_equation_solution_l3776_377616

theorem arctan_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.arctan (2 / x) + Real.arctan (1 / x^2) = π / 4 := by
  sorry

end arctan_equation_solution_l3776_377616


namespace percent_greater_l3776_377644

theorem percent_greater (w x y z : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x = 1.2 * y) (hyz : y = 1.2 * z) (hwx : w = 0.8 * x) :
  w = 1.152 * z := by
sorry

end percent_greater_l3776_377644


namespace billy_free_time_l3776_377671

/-- Proves that Billy has 16 hours of free time each day of the weekend given the specified conditions. -/
theorem billy_free_time (video_game_percentage : ℝ) (reading_percentage : ℝ)
  (pages_per_hour : ℕ) (pages_per_book : ℕ) (books_read : ℕ) :
  video_game_percentage = 0.75 →
  reading_percentage = 0.25 →
  pages_per_hour = 60 →
  pages_per_book = 80 →
  books_read = 3 →
  (books_read * pages_per_book : ℝ) / pages_per_hour / reading_percentage = 16 :=
by sorry

end billy_free_time_l3776_377671


namespace soap_usage_ratio_l3776_377692

/-- Represents the survey results of household soap usage --/
structure SoapSurvey where
  total : ℕ
  neither : ℕ
  onlyA : ℕ
  both : ℕ
  neitherLtTotal : neither < total
  onlyALtTotal : onlyA < total
  bothLtTotal : both < total

/-- Calculates the number of households using only brand B soap --/
def onlyB (s : SoapSurvey) : ℕ :=
  s.total - s.neither - s.onlyA - s.both

/-- Theorem stating the ratio of households using only brand B to those using both brands --/
theorem soap_usage_ratio (s : SoapSurvey)
  (h1 : s.total = 260)
  (h2 : s.neither = 80)
  (h3 : s.onlyA = 60)
  (h4 : s.both = 30) :
  (onlyB s) / s.both = 3 := by
  sorry

end soap_usage_ratio_l3776_377692


namespace x_fifth_minus_five_x_l3776_377632

theorem x_fifth_minus_five_x (x : ℝ) : x = 4 → x^5 - 5*x = 1004 := by
  sorry

end x_fifth_minus_five_x_l3776_377632


namespace determinant_maximum_value_l3776_377607

open Real Matrix

theorem determinant_maximum_value (θ φ : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 1, 1; 1, 1 + sin θ, 1 + cos φ; 1 + cos θ, 1 + sin φ, 1]
  ∃ (θ' φ' : ℝ), ∀ (θ φ : ℝ), det A ≤ det (!![1, 1, 1; 1, 1 + sin θ', 1 + cos φ'; 1 + cos θ', 1 + sin φ', 1]) ∧
  det (!![1, 1, 1; 1, 1 + sin θ', 1 + cos φ'; 1 + cos θ', 1 + sin φ', 1]) = 1 :=
by sorry

end determinant_maximum_value_l3776_377607


namespace ab_plus_cd_value_l3776_377662

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 3)
  (eq2 : a + b + d = -5)
  (eq3 : a + c + d = 10)
  (eq4 : b + c + d = -1) :
  a * b + c * d = -274/9 := by
sorry

end ab_plus_cd_value_l3776_377662


namespace max_ways_to_schedule_single_game_l3776_377645

/-- Represents a chess tournament between two teams -/
structure ChessTournament where
  team_size : Nat
  total_games : Nat
  games_per_day : Nat → Nat

/-- The specific tournament configuration -/
def tournament : ChessTournament :=
  { team_size := 15,
    total_games := 15 * 15,
    games_per_day := fun d => if d = 1 then 15 else 1 }

/-- The number of ways to schedule a single game -/
def ways_to_schedule_single_game (t : ChessTournament) : Nat :=
  t.total_games - t.team_size

theorem max_ways_to_schedule_single_game :
  ways_to_schedule_single_game tournament ≤ 120 :=
sorry

end max_ways_to_schedule_single_game_l3776_377645


namespace min_perimeter_noncongruent_isosceles_triangles_l3776_377664

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- The area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 2 : ℝ) * Real.sqrt ((t.leg : ℝ)^2 - (t.base / 2 : ℝ)^2)

theorem min_perimeter_noncongruent_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    t1.base * 5 = t2.base * 6 ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      s1.base * 5 = s2.base * 6 →
      perimeter t1 ≤ perimeter s1 ∧
    perimeter t1 = 364 :=
by sorry

end min_perimeter_noncongruent_isosceles_triangles_l3776_377664


namespace determinant_equation_solution_l3776_377600

/-- Definition of a 2x2 determinant -/
def det (a b c d : ℚ) : ℚ := a * d - b * c

/-- Theorem: If |x-2 x+3; x+1 x-2| = 13, then x = -3/2 -/
theorem determinant_equation_solution :
  ∀ x : ℚ, det (x - 2) (x + 3) (x + 1) (x - 2) = 13 → x = -3/2 := by
  sorry

end determinant_equation_solution_l3776_377600


namespace exists_real_sqrt_x_minus_one_l3776_377695

theorem exists_real_sqrt_x_minus_one : ∃ x : ℝ, ∃ y : ℝ, y ^ 2 = x - 1 := by
  sorry

end exists_real_sqrt_x_minus_one_l3776_377695


namespace tetrahedron_volume_from_triangle_l3776_377691

/-- Given a triangle ABC with sides of length 11, 20, and 21 units,
    the volume of the tetrahedron formed by folding the triangle along
    the lines connecting the midpoints of its sides is 45 cubic units. -/
theorem tetrahedron_volume_from_triangle (a b c : ℝ) (h1 : a = 11) (h2 : b = 20) (h3 : c = 21) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let p := b / 2
  let q := c / 2
  let r := a / 2
  let s_mid := (p + q + r) / 2
  let area_mid := Real.sqrt (s_mid * (s_mid - p) * (s_mid - q) * (s_mid - r))
  let height := Real.sqrt ((q^2) - (area_mid^2 / area^2) * (a^2 / 4))
  (1/3) * area_mid * height = 45 := by
  sorry

end tetrahedron_volume_from_triangle_l3776_377691


namespace solution_to_system_l3776_377634

theorem solution_to_system (x y : ℝ) 
  (h1 : 9 * x^2 - 25 * y^2 = 0) 
  (h2 : x^2 + y^2 = 10) : 
  (x = 5 * Real.sqrt (45/17) / 3 ∨ x = -5 * Real.sqrt (45/17) / 3) ∧
  (y = Real.sqrt (45/17) ∨ y = -Real.sqrt (45/17)) := by
sorry


end solution_to_system_l3776_377634


namespace max_uncolored_cubes_l3776_377663

/-- Represents a rectangular prism --/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular prism --/
def volume (p : RectangularPrism) : ℕ := p.length * p.width * p.height

/-- Calculates the number of interior cubes in a rectangular prism --/
def interiorCubes (p : RectangularPrism) : ℕ :=
  (p.length - 2) * (p.width - 2) * (p.height - 2)

theorem max_uncolored_cubes (p : RectangularPrism) 
  (h_dim : p.length = 8 ∧ p.width = 8 ∧ p.height = 16) 
  (h_vol : volume p = 1024) :
  interiorCubes p = 504 := by
  sorry


end max_uncolored_cubes_l3776_377663


namespace circle_and_tangent_line_l3776_377647

/-- Circle C with equation x^2+y^2-8x+6y+21=0 -/
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1^2 + p.2^2 - 8*p.1 + 6*p.2 + 21) = 0}

/-- Point A with coordinates (-6, 7) -/
def point_A : ℝ × ℝ := (-6, 7)

/-- A line is tangent to a circle if it intersects the circle at exactly one point -/
def is_tangent_line (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ l ∩ c

/-- The set of all lines passing through point A -/
def lines_through_A : Set (Set (ℝ × ℝ)) :=
  {l | point_A ∈ l ∧ ∃ k, l = {p | p.2 - 7 = k * (p.1 + 6)}}

theorem circle_and_tangent_line :
  ∃ l ∈ lines_through_A,
    is_tangent_line l circle_C ∧
    (∃ c r, c = (4, -3) ∧ r = 2 ∧
      ∀ p ∈ circle_C, (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧
    (l = {p | 3*p.1 + 4*p.2 - 10 = 0} ∨ l = {p | 4*p.1 + 3*p.2 + 3 = 0}) :=
  sorry

end circle_and_tangent_line_l3776_377647


namespace sine_amplitude_l3776_377681

theorem sine_amplitude (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.sin (b * x) ≤ 3) ∧ (∃ x, a * Real.sin (b * x) = 3) → a = 3 := by
  sorry

end sine_amplitude_l3776_377681


namespace equation_a_solution_equation_b_no_solution_l3776_377638

-- Part (a)
theorem equation_a_solution (x : ℚ) : 
  1 + 1 / (2 + 1 / ((4*x + 1) / (2*x + 1) - 1 / (2 + 1/x))) = 19/14 ↔ x = 1/2 :=
sorry

-- Part (b)
theorem equation_b_no_solution :
  ¬∃ (x : ℚ), ((2*x - 1)/2 + 4/3) / ((x - 1)/3 - 1/2 * (1 - 1/3)) - 
  (x + 4) / ((2*x + 1)/2 + 1/5 - 2 - 1/(1 + 1/(2 + 1/3))) = (9 - 2*x) / (2*x - 4) :=
sorry

end equation_a_solution_equation_b_no_solution_l3776_377638


namespace sector_radius_l3776_377609

/-- Given a sector of a circle with perimeter 144 cm and central angle π/3 radians,
    prove that the radius of the circle is 432 / (6 + π) cm. -/
theorem sector_radius (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 144) (h2 : central_angle = π/3) :
  ∃ r : ℝ, r = 432 / (6 + π) ∧ perimeter = 2*r + r * central_angle := by
  sorry

end sector_radius_l3776_377609


namespace total_fish_bought_l3776_377626

theorem total_fish_bought (goldfish : ℕ) (blue_fish : ℕ) (angelfish : ℕ) (neon_tetras : ℕ)
  (h1 : goldfish = 23)
  (h2 : blue_fish = 15)
  (h3 : angelfish = 8)
  (h4 : neon_tetras = 12) :
  goldfish + blue_fish + angelfish + neon_tetras = 58 := by
  sorry

end total_fish_bought_l3776_377626


namespace last_digit_of_one_over_two_to_twelve_l3776_377604

theorem last_digit_of_one_over_two_to_twelve (n : ℕ) : 
  n = 12 → (1 : ℚ) / (2^n : ℚ) * 10^n % 10 = 5 := by
  sorry

end last_digit_of_one_over_two_to_twelve_l3776_377604


namespace evaluate_expression_l3776_377665

theorem evaluate_expression (x : ℝ) (h : x = 3) : x^6 - 6*x^2 = 675 := by
  sorry

end evaluate_expression_l3776_377665


namespace dihedral_angle_definition_inconsistency_l3776_377621

/-- Definition of a half-plane --/
def HalfPlane : Type := sorry

/-- Definition of a straight line --/
def StraightLine : Type := sorry

/-- Definition of a spatial figure --/
def SpatialFigure : Type := sorry

/-- Definition of a planar angle --/
def PlanarAngle : Type := sorry

/-- Incorrect definition of a dihedral angle --/
def IncorrectDihedralAngle : Type :=
  {angle : PlanarAngle // ∃ (hp1 hp2 : HalfPlane) (l : StraightLine),
    angle = sorry }

/-- Correct definition of a dihedral angle --/
def CorrectDihedralAngle : Type :=
  {sf : SpatialFigure // ∃ (hp1 hp2 : HalfPlane) (l : StraightLine),
    sf = sorry }

/-- Theorem stating that the incorrect definition is inconsistent with the 3D nature of dihedral angles --/
theorem dihedral_angle_definition_inconsistency :
  ¬(IncorrectDihedralAngle = CorrectDihedralAngle) :=
sorry

end dihedral_angle_definition_inconsistency_l3776_377621


namespace no_intersection_at_vertex_l3776_377696

/-- The line equation y = x + b -/
def line (x b : ℝ) : ℝ := x + b

/-- The parabola equation y = x^2 + b^2 + 1 -/
def parabola (x b : ℝ) : ℝ := x^2 + b^2 + 1

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 0

/-- Theorem: There are no real values of b for which the line y = x + b passes through the vertex of the parabola y = x^2 + b^2 + 1 -/
theorem no_intersection_at_vertex :
  ¬∃ b : ℝ, line vertex_x b = parabola vertex_x b := by sorry

end no_intersection_at_vertex_l3776_377696


namespace ordering_of_trig_and_log_expressions_l3776_377635

theorem ordering_of_trig_and_log_expressions :
  let a := Real.sin (Real.cos 2)
  let b := Real.cos (Real.cos 2)
  let c := Real.log (Real.cos 1)
  c < a ∧ a < b := by sorry

end ordering_of_trig_and_log_expressions_l3776_377635


namespace green_tea_profit_maximization_l3776_377624

/-- The profit function for a green tea company -/
def profit (x : ℝ) : ℝ := -2 * x^2 + 340 * x - 12000

/-- The selling price that maximizes profit -/
def max_profit_price : ℝ := 85

theorem green_tea_profit_maximization :
  /- The profit function is correct -/
  (∀ x : ℝ, profit x = -2 * x^2 + 340 * x - 12000) ∧
  /- The maximum profit occurs at x = 85 -/
  (∀ x : ℝ, profit x ≤ profit max_profit_price) := by
  sorry


end green_tea_profit_maximization_l3776_377624


namespace train_length_l3776_377617

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 40 → time = 27 → speed * time * (1000 / 3600) = 300 := by
  sorry

end train_length_l3776_377617


namespace factorization_proof_l3776_377672

theorem factorization_proof (a x y : ℝ) : a * x - a * y = a * (x - y) := by
  sorry

end factorization_proof_l3776_377672


namespace distance_between_points_l3776_377675

/-- The distance between two points when two vehicles move towards each other -/
theorem distance_between_points (v1 v2 t : ℝ) (h1 : v1 > 0) (h2 : v2 > 0) (h3 : t > 0) :
  let d := (v1 + v2) * t
  d = v1 * t + v2 * t :=
by sorry

end distance_between_points_l3776_377675


namespace time_to_fill_leaking_tank_l3776_377606

/-- Time to fill a leaking tank -/
theorem time_to_fill_leaking_tank 
  (pump_fill_time : ℝ) 
  (leak_empty_time : ℝ) 
  (h1 : pump_fill_time = 6) 
  (h2 : leak_empty_time = 12) : 
  (pump_fill_time * leak_empty_time) / (leak_empty_time - pump_fill_time) = 12 := by
  sorry

#check time_to_fill_leaking_tank

end time_to_fill_leaking_tank_l3776_377606


namespace expression_simplification_and_evaluation_l3776_377618

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 2) :
  (1 + (1 - x) / (x + 1)) / ((2 * x - 2) / (x^2 + 2 * x + 1)) = 3 := by
  sorry

end expression_simplification_and_evaluation_l3776_377618


namespace min_translation_overlap_l3776_377623

theorem min_translation_overlap (φ : Real) : 
  (φ > 0) →
  (∀ x, Real.sin (2 * (x + φ)) = Real.sin (2 * x - 2 * φ + Real.pi / 3)) →
  φ ≥ Real.pi / 12 :=
sorry

end min_translation_overlap_l3776_377623


namespace swap_values_l3776_377698

theorem swap_values (a b : ℕ) (h1 : a = 3) (h2 : b = 2) :
  ∃ c : ℕ, (c = b) ∧ (b = a) ∧ (a = c) → a = 2 ∧ b = 3 := by
  sorry

end swap_values_l3776_377698


namespace system_solution_l3776_377657

theorem system_solution :
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ),
    (x₁ + x₂ + x₃ = 6) ∧
    (x₂ + x₃ + x₄ = 9) ∧
    (x₃ + x₄ + x₅ = 3) ∧
    (x₄ + x₅ + x₆ = -3) ∧
    (x₅ + x₆ + x₇ = -9) ∧
    (x₆ + x₇ + x₈ = -6) ∧
    (x₇ + x₈ + x₁ = -2) ∧
    (x₈ + x₁ + x₂ = 2) ∧
    x₁ = 1 ∧ x₂ = 2 ∧ x₃ = 3 ∧ x₄ = 4 ∧
    x₅ = -4 ∧ x₆ = -3 ∧ x₇ = -2 ∧ x₈ = -1 :=
by
  sorry

end system_solution_l3776_377657


namespace min_value_expression_equality_condition_l3776_377667

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^4 + 12 * b^4 + 50 * c^4 + 1 / (9 * a * b * c) ≥ 2 * Real.sqrt (20 / 3) :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (8 * a^4 + 12 * b^4 + 50 * c^4 + 1 / (9 * a * b * c) = 2 * Real.sqrt (20 / 3)) ↔
  (a = (3/2)^(1/4) * b ∧ b = (25/6)^(1/4) * c ∧ c = (4/25)^(1/4) * a) :=
by sorry

end min_value_expression_equality_condition_l3776_377667


namespace faye_pencils_l3776_377676

/-- The number of rows of pencils -/
def num_rows : ℕ := 14

/-- The number of pencils in each row -/
def pencils_per_row : ℕ := 11

/-- The total number of pencils -/
def total_pencils : ℕ := num_rows * pencils_per_row

theorem faye_pencils : total_pencils = 154 := by
  sorry

end faye_pencils_l3776_377676


namespace square_garden_perimeter_l3776_377677

theorem square_garden_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 200 →
  side^2 = area →
  perimeter = 4 * side →
  perimeter = 40 * Real.sqrt 2 := by
  sorry

end square_garden_perimeter_l3776_377677


namespace ellipse_k_value_l3776_377640

/-- An ellipse with equation x^2 + ky^2 = 1, where k is a positive real number -/
structure Ellipse (k : ℝ) : Type :=
  (eq : ∀ x y : ℝ, x^2 + k * y^2 = 1)

/-- The focus of the ellipse is on the y-axis -/
def focus_on_y_axis (e : Ellipse k) : Prop :=
  k < 1

/-- The length of the major axis is twice that of the minor axis -/
def major_axis_twice_minor (e : Ellipse k) : Prop :=
  2 * (1 / Real.sqrt k) = 4

/-- Theorem: For an ellipse with the given properties, k = 1/4 -/
theorem ellipse_k_value (k : ℝ) (e : Ellipse k) 
  (h1 : focus_on_y_axis e) (h2 : major_axis_twice_minor e) : k = 1/4 := by
  sorry

end ellipse_k_value_l3776_377640


namespace cubic_fraction_simplification_l3776_377628

theorem cubic_fraction_simplification (a b : ℝ) (h : a = 6 ∧ b = 6) :
  (a^3 + b^3) / (a^2 - a*b + b^2) = 12 := by
  sorry

end cubic_fraction_simplification_l3776_377628


namespace emerson_rowing_distance_l3776_377678

/-- The total distance covered by Emerson on his rowing trip -/
def total_distance (first_part second_part third_part : ℕ) : ℕ :=
  first_part + second_part + third_part

/-- Theorem stating that Emerson's total rowing distance is 39 miles -/
theorem emerson_rowing_distance :
  total_distance 6 15 18 = 39 := by
  sorry

end emerson_rowing_distance_l3776_377678


namespace smallest_beta_l3776_377627

theorem smallest_beta (α β : ℕ) (h1 : α > 0) (h2 : β > 0) 
  (h3 : (16 : ℚ) / 37 < α / β) (h4 : α / β < (7 : ℚ) / 16) : 
  (∀ γ : ℕ, γ > 0 → ∃ δ : ℕ, δ > 0 ∧ (16 : ℚ) / 37 < δ / γ ∧ δ / γ < (7 : ℚ) / 16 → γ ≥ 23) ∧ 
  (∃ ε : ℕ, ε > 0 ∧ (16 : ℚ) / 37 < ε / 23 ∧ ε / 23 < (7 : ℚ) / 16) :=
sorry

end smallest_beta_l3776_377627


namespace set_intersection_empty_implies_a_range_l3776_377655

theorem set_intersection_empty_implies_a_range (a : ℝ) : 
  let A := {x : ℝ | a - 1 < x ∧ x < 2*a + 1}
  let B := {x : ℝ | 0 < x ∧ x < 1}
  (A ∩ B = ∅) → (a ≤ -1/2 ∨ a ≥ 2) := by
sorry

end set_intersection_empty_implies_a_range_l3776_377655


namespace no_real_solutions_quadratic_inequality_l3776_377666

theorem no_real_solutions_quadratic_inequality :
  ¬∃ (x : ℝ), 3 * x^2 + 9 * x ≤ -12 := by
  sorry

end no_real_solutions_quadratic_inequality_l3776_377666


namespace pyramid_volume_l3776_377686

theorem pyramid_volume (base_length : Real) (base_width : Real) (height : Real) :
  base_length = 1 → base_width = 1/4 → height = 1 →
  (1/3) * (base_length * base_width) * height = 1/12 := by
sorry

end pyramid_volume_l3776_377686


namespace point_on_circle_after_rotation_l3776_377603

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def arc_length (θ : ℝ) : ℝ := θ

theorem point_on_circle_after_rotation 
  (start_x start_y : ℝ) 
  (θ : ℝ) 
  (h_start : unit_circle start_x start_y) 
  (h_θ : arc_length θ = 2 * Real.pi / 3) :
  ∃ (end_x end_y : ℝ), 
    unit_circle end_x end_y ∧ 
    end_x = -1/2 ∧ 
    end_y = Real.sqrt 3 / 2 :=
sorry

end point_on_circle_after_rotation_l3776_377603


namespace bankers_calculation_l3776_377688

/-- Proves that given specific banker's gain, banker's discount, and interest rate, the time period is 3 years -/
theorem bankers_calculation (bankers_gain : ℝ) (bankers_discount : ℝ) (interest_rate : ℝ) :
  bankers_gain = 270 →
  bankers_discount = 1020 →
  interest_rate = 0.12 →
  ∃ (time : ℝ), time = 3 ∧ bankers_discount = (bankers_discount - bankers_gain) * (1 + interest_rate * time) :=
by sorry

end bankers_calculation_l3776_377688


namespace gcf_lcm_sum_plus_ten_l3776_377625

theorem gcf_lcm_sum_plus_ten (a b : ℕ) (h1 : a = 8) (h2 : b = 12) :
  Nat.gcd a b + Nat.lcm a b + 10 = 38 := by
  sorry

end gcf_lcm_sum_plus_ten_l3776_377625


namespace probability_of_white_and_black_l3776_377636

/-- The number of white balls in the bag -/
def num_white : ℕ := 2

/-- The number of black balls in the bag -/
def num_black : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_white + num_black

/-- The number of balls drawn -/
def drawn : ℕ := 2

/-- The probability of drawing one white ball and one black ball -/
def prob_white_and_black : ℚ := 2 / 3

theorem probability_of_white_and_black :
  (num_white * num_black : ℚ) / (total_balls.choose drawn) = prob_white_and_black := by
  sorry

end probability_of_white_and_black_l3776_377636


namespace complex_fraction_equality_l3776_377679

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 - a*b + b^2 = 0) :
  (a^6 + b^6) / (a + b)^6 = 1/18 := by
sorry

end complex_fraction_equality_l3776_377679


namespace remainder_squared_multiply_l3776_377654

theorem remainder_squared_multiply (n a b : ℤ) : 
  n > 0 → b = 3 → a * b ≡ 1 [ZMOD n] → a^2 * b ≡ a [ZMOD n] := by
  sorry

end remainder_squared_multiply_l3776_377654


namespace marias_stationery_cost_l3776_377694

/-- The total cost of Maria's stationery purchase after applying a coupon and including sales tax. -/
theorem marias_stationery_cost :
  let notebook_a_count : ℕ := 4
  let notebook_b_count : ℕ := 3
  let notebook_c_count : ℕ := 3
  let pen_count : ℕ := 5
  let highlighter_pack_count : ℕ := 1
  let notebook_a_price : ℚ := 3.5
  let notebook_b_price : ℚ := 2.25
  let notebook_c_price : ℚ := 1.75
  let pen_price : ℚ := 2
  let highlighter_pack_price : ℚ := 4.5
  let coupon_discount : ℚ := 0.1
  let sales_tax_rate : ℚ := 0.05

  let total_before_discount : ℚ := 
    notebook_a_count * notebook_a_price +
    notebook_b_count * notebook_b_price +
    notebook_c_count * notebook_c_price +
    pen_count * pen_price +
    highlighter_pack_count * highlighter_pack_price

  let discount_amount : ℚ := total_before_discount * coupon_discount
  let total_after_discount : ℚ := total_before_discount - discount_amount
  let sales_tax : ℚ := total_after_discount * sales_tax_rate
  let final_cost : ℚ := total_after_discount + sales_tax

  final_cost = 38.27 := by sorry

end marias_stationery_cost_l3776_377694


namespace stratified_sampling_l3776_377643

theorem stratified_sampling (total : ℕ) (sample_size : ℕ) (group_size : ℕ) 
  (h1 : total = 700) 
  (h2 : sample_size = 14) 
  (h3 : group_size = 300) :
  (group_size * sample_size) / total = 6 := by
  sorry

end stratified_sampling_l3776_377643


namespace roots_sum_l3776_377605

theorem roots_sum (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ x : ℝ, x^2 - 12*a*x - 13*b = 0 ↔ (x = c ∨ x = d)) →
  (∀ x : ℝ, x^2 - 12*c*x - 13*d = 0 ↔ (x = a ∨ x = b)) →
  a + b + c + d = 1716 := by
sorry

end roots_sum_l3776_377605


namespace floor_div_p_equals_86422_l3776_377699

/-- A function that generates the sequence of 6-digit numbers with digits in non-increasing order -/
def nonIncreasingDigitSequence : ℕ → ℕ := sorry

/-- The 2010th number in the sequence -/
def p : ℕ := nonIncreasingDigitSequence 2010

/-- Theorem stating that the floor division of p by 10 equals 86422 -/
theorem floor_div_p_equals_86422 : p / 10 = 86422 := by sorry

end floor_div_p_equals_86422_l3776_377699


namespace income_a_is_4000_l3776_377610

/-- Represents the financial situation of two individuals A and B -/
structure FinancialSituation where
  incomeRatio : Rat
  expenditureRatio : Rat
  savings : ℕ

/-- Calculates the income of individual A given the financial situation -/
def incomeA (fs : FinancialSituation) : ℕ := sorry

/-- Theorem stating that given the specific financial situation, the income of A is $4000 -/
theorem income_a_is_4000 :
  let fs : FinancialSituation := {
    incomeRatio := 5 / 4,
    expenditureRatio := 3 / 2,
    savings := 1600
  }
  incomeA fs = 4000 := by sorry

end income_a_is_4000_l3776_377610


namespace tenth_pirate_coins_l3776_377668

/-- Represents the number of pirates --/
def num_pirates : ℕ := 10

/-- Represents the initial number of silver coins --/
def initial_silver : ℕ := 1050

/-- Represents the number of silver coins each pirate takes --/
def silver_per_pirate : ℕ := 100

/-- Calculates the remaining gold coins after k pirates have taken their share --/
def remaining_gold (initial_gold : ℕ) (k : ℕ) : ℚ :=
  (num_pirates - k : ℚ) / num_pirates * initial_gold

/-- Calculates the number of gold coins the 10th pirate receives --/
def gold_for_last_pirate (initial_gold : ℕ) : ℚ :=
  remaining_gold initial_gold (num_pirates - 1)

/-- Calculates the number of silver coins the 10th pirate receives --/
def silver_for_last_pirate : ℕ :=
  initial_silver - (num_pirates - 1) * silver_per_pirate

/-- Theorem stating that the 10th pirate receives 494 coins in total --/
theorem tenth_pirate_coins (initial_gold : ℕ) :
  ∃ (gold_coins : ℕ), gold_for_last_pirate initial_gold = gold_coins ∧
  gold_coins + silver_for_last_pirate = 494 :=
sorry

end tenth_pirate_coins_l3776_377668


namespace kimberly_store_visits_l3776_377633

/-- Represents the number of peanuts Kimberly buys each time she goes to the store. -/
def peanuts_per_visit : ℕ := 7

/-- Represents the total number of peanuts Kimberly bought last month. -/
def total_peanuts : ℕ := 21

/-- Represents the number of times Kimberly went to the store last month. -/
def store_visits : ℕ := total_peanuts / peanuts_per_visit

/-- Proves that Kimberly went to the store 3 times last month. -/
theorem kimberly_store_visits : store_visits = 3 := by
  sorry

end kimberly_store_visits_l3776_377633


namespace simplify_expression_l3776_377673

theorem simplify_expression (x y : ℝ) : (3 * x^2 * y)^3 + (4 * x * y) * (y^4) = 27 * x^6 * y^3 + 4 * x * y^5 := by
  sorry

end simplify_expression_l3776_377673


namespace polynomial_sum_equality_l3776_377651

-- Define the two polynomials
def p1 (x : ℝ) : ℝ := 3*x^4 + 2*x^3 - 5*x^2 + 9*x - 2
def p2 (x : ℝ) : ℝ := -3*x^4 - 5*x^3 + 7*x^2 - 9*x + 4

-- Define the sum of the polynomials
def sum_poly (x : ℝ) : ℝ := p1 x + p2 x

-- Define the result polynomial
def result (x : ℝ) : ℝ := -3*x^3 + 2*x^2 + 2

-- Theorem statement
theorem polynomial_sum_equality : 
  ∀ x : ℝ, sum_poly x = result x := by sorry

end polynomial_sum_equality_l3776_377651


namespace solution_product_l3776_377602

theorem solution_product (p q : ℝ) : 
  (p - 7) * (2 * p + 11) = p^2 - 19 * p + 60 →
  (q - 7) * (2 * q + 11) = q^2 - 19 * q + 60 →
  p ≠ q →
  (p - 2) * (q - 2) = -55 := by
sorry

end solution_product_l3776_377602


namespace cousin_name_probability_l3776_377614

theorem cousin_name_probability :
  let total_cards : ℕ := 12
  let adrian_cards : ℕ := 7
  let bella_cards : ℕ := 5
  let prob_one_from_each : ℚ := 
    (adrian_cards / total_cards) * (bella_cards / (total_cards - 1)) +
    (bella_cards / total_cards) * (adrian_cards / (total_cards - 1))
  prob_one_from_each = 35 / 66 := by
sorry

end cousin_name_probability_l3776_377614


namespace golden_rectangle_ratio_l3776_377659

theorem golden_rectangle_ratio (x y : ℝ) (h1 : x > y) (h2 : y > 0) : 
  (y / x = (x - y) / y) → (x / y = (1 + Real.sqrt 5) / 2) := by
  sorry

end golden_rectangle_ratio_l3776_377659


namespace donna_additional_flyers_eq_five_l3776_377601

/-- The number of flyers Maisie dropped off -/
def maisie_flyers : ℕ := 33

/-- The total number of flyers Donna dropped off -/
def donna_total_flyers : ℕ := 71

/-- The number of additional flyers Donna dropped off -/
def donna_additional_flyers : ℕ := donna_total_flyers - 2 * maisie_flyers

theorem donna_additional_flyers_eq_five : donna_additional_flyers = 5 := by
  sorry

end donna_additional_flyers_eq_five_l3776_377601


namespace base_prime_repr_360_l3776_377674

/-- Base prime representation of a natural number --/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- The base prime representation of 360 --/
theorem base_prime_repr_360 : base_prime_repr 360 = [3, 2, 1] := by
  sorry

end base_prime_repr_360_l3776_377674


namespace collinear_points_k_value_unique_k_value_l3776_377697

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) / (x₂ - x₁) = (y₃ - y₂) / (x₃ - x₂)

/-- Theorem: If the points (2,-3), (4,3), and (5, k/2) are collinear, then k = 12. -/
theorem collinear_points_k_value :
  ∀ k : ℝ, collinear 2 (-3) 4 3 5 (k/2) → k = 12 :=
by
  sorry

/-- Corollary: The only value of k that makes the points (2,-3), (4,3), and (5, k/2) collinear is 12. -/
theorem unique_k_value :
  ∃! k : ℝ, collinear 2 (-3) 4 3 5 (k/2) :=
by
  sorry

end collinear_points_k_value_unique_k_value_l3776_377697


namespace min_value_of_expression_l3776_377630

theorem min_value_of_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (a + b) / c + (b + c) / a + (c + a) / b + 3 ≥ 9 := by
  sorry

end min_value_of_expression_l3776_377630


namespace kombucha_bottle_cost_l3776_377684

/-- Represents the cost of a bottle of kombucha -/
def bottle_cost : ℝ := sorry

/-- Represents the number of bottles Henry drinks per month -/
def bottles_per_month : ℕ := 15

/-- Represents the cash refund per bottle in dollars -/
def refund_per_bottle : ℝ := 0.1

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents the number of bottles that can be bought with the yearly refund -/
def bottles_bought_with_refund : ℕ := 6

theorem kombucha_bottle_cost :
  bottle_cost = 3 :=
by
  sorry

end kombucha_bottle_cost_l3776_377684


namespace multiplication_correction_l3776_377656

theorem multiplication_correction (n : ℕ) : 
  n * 987 = 559981 → 
  (∃ a b : ℕ, a ≠ 9 ∧ b ≠ 8 ∧ n * 987 = 5 * 100000 + a * 10000 + b * 1000 + 981) → 
  n * 987 = 559989 :=
by sorry

end multiplication_correction_l3776_377656


namespace committee_formation_l3776_377680

theorem committee_formation (n m k : ℕ) (h1 : n = 12) (h2 : m = 5) (h3 : k = 4) :
  Nat.choose n m = 792 ∧ Nat.choose (n - k) m = 56 := by
  sorry

end committee_formation_l3776_377680


namespace b_over_c_equals_one_l3776_377611

theorem b_over_c_equals_one (a b c d : ℕ) : 
  0 < a ∧ a < 4 ∧ 
  0 < b ∧ b < 4 ∧ 
  0 < c ∧ c < 4 ∧ 
  0 < d ∧ d < 4 ∧ 
  4^a + 3^b + 2^c + 1^d = 78 → 
  b / c = 1 := by
  sorry

end b_over_c_equals_one_l3776_377611


namespace original_number_proof_l3776_377639

theorem original_number_proof (x : ℚ) : 
  2 + (1 / x) = 10 / 3 → x = 3 / 4 := by
sorry

end original_number_proof_l3776_377639


namespace tangent_line_to_cubic_curve_l3776_377615

theorem tangent_line_to_cubic_curve (a : ℝ) :
  (∃ x y : ℝ, y = 3 * x + 1 ∧ y = x^3 - a ∧ 3 * x^2 = 3) →
  (a = -3 ∨ a = 1) :=
by sorry

end tangent_line_to_cubic_curve_l3776_377615


namespace ceiling_negative_three_point_seven_l3776_377608

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by
  sorry

end ceiling_negative_three_point_seven_l3776_377608


namespace no_common_points_range_two_common_points_product_l3776_377646

noncomputable section

-- Define the functions f and g
def f (x : ℝ) := Real.log x
def g (a : ℝ) (x : ℝ) := a * x

-- Part I
theorem no_common_points_range (a : ℝ) :
  (∀ x > 0, f x ≠ g a x) → a > 1 / Real.exp 1 := by sorry

-- Part II
theorem two_common_points_product (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ ≠ x₂ → f x₁ = g a x₁ → f x₂ = g a x₂ → x₁ * x₂ > Real.exp 2 := by sorry

end no_common_points_range_two_common_points_product_l3776_377646


namespace handshakes_15_couples_l3776_377690

/-- The number of handshakes in a gathering of married couples -/
def num_handshakes (n : ℕ) : ℕ :=
  (n * 2 * (n * 2 - 2)) / 2 - n

/-- Theorem: In a gathering of 15 married couples, the total number of handshakes is 405 -/
theorem handshakes_15_couples :
  num_handshakes 15 = 405 := by
  sorry

end handshakes_15_couples_l3776_377690


namespace larger_number_l3776_377652

theorem larger_number (x y : ℝ) (h1 : x - y = 5) (h2 : x + y = 20) : max x y = 12.5 := by
  sorry

end larger_number_l3776_377652


namespace power_fraction_simplification_l3776_377629

theorem power_fraction_simplification :
  (3^5 * 4^5) / 6^5 = 32 := by
  sorry

end power_fraction_simplification_l3776_377629


namespace sum_of_common_elements_l3776_377683

-- Define the arithmetic progression
def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n

-- Define the geometric progression
def geometric_progression (k : ℕ) : ℕ := 10 * 2^k

-- Define the sequence of common elements
def common_elements (n : ℕ) : ℕ := 10 * 4^n

-- Theorem statement
theorem sum_of_common_elements : 
  (Finset.range 10).sum common_elements = 3495250 := by sorry

end sum_of_common_elements_l3776_377683


namespace triangle_angle_measures_l3776_377660

theorem triangle_angle_measures (A B C : ℝ) 
  (h1 : B - A = 5)
  (h2 : C - B = 20)
  (h3 : A + B + C = 180) : 
  A = 50 ∧ B = 55 ∧ C = 75 := by
sorry

end triangle_angle_measures_l3776_377660


namespace not_always_true_l3776_377622

theorem not_always_true (r p q : ℝ) (hr : r > 0) (hpq : p * q ≠ 0) (hpqr : p * r > q * r) :
  ¬((-p > -q) ∨ (-p > q) ∨ (1 > -q/p) ∨ (1 < q/p)) := by
  sorry

end not_always_true_l3776_377622


namespace ivy_stripping_l3776_377642

/-- The number of feet of ivy Cary strips daily -/
def daily_strip : ℝ := 6

/-- The initial ivy coverage in feet -/
def initial_coverage : ℝ := 40

/-- The number of days it takes to remove all ivy -/
def days_to_remove : ℝ := 10

/-- The number of feet the ivy grows each night -/
def nightly_growth : ℝ := 2

theorem ivy_stripping :
  daily_strip * days_to_remove - nightly_growth * days_to_remove = initial_coverage :=
sorry

end ivy_stripping_l3776_377642


namespace common_tangent_sum_l3776_377641

/-- Given two functions f and g with a common tangent at (0, m), prove a + b = 1 -/
theorem common_tangent_sum (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.cos x
  let g : ℝ → ℝ := λ x ↦ x^2 + b*x + 1
  let f' : ℝ → ℝ := λ x ↦ -a * Real.sin x
  let g' : ℝ → ℝ := λ x ↦ 2*x + b
  (∃ m : ℝ, f 0 = m ∧ g 0 = m ∧ f' 0 = g' 0) →
  a + b = 1 := by
sorry

end common_tangent_sum_l3776_377641


namespace violet_balloons_lost_l3776_377669

theorem violet_balloons_lost (initial_violet : ℕ) (remaining_violet : ℕ) 
  (h1 : initial_violet = 7) 
  (h2 : remaining_violet = 4) : 
  initial_violet - remaining_violet = 3 := by
sorry

end violet_balloons_lost_l3776_377669


namespace volume_of_solid_is_62pi_over_3_l3776_377637

/-- The region S in the coordinate plane -/
def region_S : Set (ℝ × ℝ) :=
  {p | p.2 ≤ p.1 + 2 ∧ p.2 ≤ -p.1 + 6 ∧ p.2 ≤ 4}

/-- The volume of the solid formed by revolving region S around the y-axis -/
noncomputable def volume_of_solid : ℝ := sorry

/-- Theorem stating that the volume of the solid is 62π/3 -/
theorem volume_of_solid_is_62pi_over_3 :
  volume_of_solid = 62 * Real.pi / 3 := by sorry

end volume_of_solid_is_62pi_over_3_l3776_377637


namespace digital_earth_capabilities_l3776_377649

-- Define the possible capabilities
inductive Capability
  | ReceiveDistanceEducation
  | ShopOnline
  | SeekMedicalAdviceOnline
  | TravelAroundWorld

-- Define Digital Earth
def DigitalEarth : Type := Set Capability

-- Define the correct set of capabilities
def CorrectCapabilities : Set Capability :=
  {Capability.ReceiveDistanceEducation, Capability.ShopOnline, Capability.SeekMedicalAdviceOnline}

-- Theorem stating that Digital Earth capabilities are exactly the correct ones
theorem digital_earth_capabilities :
  ∃ (de : DigitalEarth), de = CorrectCapabilities :=
sorry

end digital_earth_capabilities_l3776_377649


namespace computer_price_increase_l3776_377648

/-- The new price of a computer after a 30% increase, given initial conditions -/
theorem computer_price_increase (b : ℝ) (h : 2 * b = 540) : b * 1.3 = 351 := by
  sorry

end computer_price_increase_l3776_377648
