import Mathlib

namespace rachels_weight_l3522_352242

theorem rachels_weight (rachel jimmy adam : ℝ) 
  (h1 : jimmy = rachel + 6)
  (h2 : rachel = adam + 15)
  (h3 : (rachel + jimmy + adam) / 3 = 72) :
  rachel = 75 := by
  sorry

end rachels_weight_l3522_352242


namespace percentage_increase_in_earnings_l3522_352240

theorem percentage_increase_in_earnings (initial_earnings new_earnings : ℝ) 
  (h1 : initial_earnings = 60)
  (h2 : new_earnings = 84) :
  (new_earnings - initial_earnings) / initial_earnings * 100 = 40 := by
  sorry

end percentage_increase_in_earnings_l3522_352240


namespace tax_free_items_cost_l3522_352227

/-- Calculates the cost of tax-free items given total spend, tax percentage, and tax rate -/
def cost_of_tax_free_items (total_spend : ℚ) (tax_percentage : ℚ) (tax_rate : ℚ) : ℚ :=
  let taxable_cost := total_spend * (1 - tax_percentage / 100)
  let rounded_tax := (taxable_cost * tax_rate / 100).ceil
  total_spend - (taxable_cost + rounded_tax)

theorem tax_free_items_cost :
  cost_of_tax_free_items 40 30 6 = 10 :=
by sorry

end tax_free_items_cost_l3522_352227


namespace no_intersection_and_in_circle_l3522_352283

theorem no_intersection_and_in_circle : ¬∃ (a b : ℝ),
  (∃ (n m : ℤ), n = m ∧ n * a + b = 3 * m^2 + 15) ∧
  (a^2 + b^2 ≤ 144) := by
  sorry

end no_intersection_and_in_circle_l3522_352283


namespace container_evaporation_l3522_352222

theorem container_evaporation (initial_content : ℝ) : 
  initial_content = 1 →
  let remaining_after_day1 := initial_content - (2/3 * initial_content)
  let remaining_after_day2 := remaining_after_day1 - (1/4 * remaining_after_day1)
  remaining_after_day2 = 1/4 * initial_content := by sorry

end container_evaporation_l3522_352222


namespace nested_fraction_equals_nineteen_elevenths_l3522_352299

theorem nested_fraction_equals_nineteen_elevenths :
  1 + 1 / (1 + 1 / (2 + 2 / 3)) = 19 / 11 := by
  sorry

end nested_fraction_equals_nineteen_elevenths_l3522_352299


namespace find_a_l3522_352206

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {2, a^2 + 9*a + 3, 6}

-- Define set A
def A (a : ℝ) : Set ℝ := {2, |a + 3|}

-- Define the complement of A relative to U
def complement_A (a : ℝ) : Set ℝ := {3}

-- Theorem statement
theorem find_a : ∃ a : ℝ, 
  (U a = {2, a^2 + 9*a + 3, 6}) ∧ 
  (A a = {2, |a + 3|}) ∧ 
  (complement_A a = {3}) ∧ 
  (a = -9) := by
  sorry

end find_a_l3522_352206


namespace prob_at_least_one_boy_one_girl_l3522_352252

/-- The probability of having a boy or a girl -/
def gender_prob : ℚ := 1 / 2

/-- The number of children in the family -/
def num_children : ℕ := 4

/-- The probability of having at least one boy and one girl in a family with four children,
    given that the probability of having a boy or a girl is equally likely -/
theorem prob_at_least_one_boy_one_girl (h : gender_prob = 1 / 2) :
  1 - (gender_prob ^ num_children + (1 - gender_prob) ^ num_children) = 7 / 8 := by
  sorry

end prob_at_least_one_boy_one_girl_l3522_352252


namespace diagonal_angle_tangent_l3522_352200

/-- A convex quadrilateral with given properties -/
structure ConvexQuadrilateral where
  area : ℝ
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  convex : Bool

/-- The measure of the acute angle formed by the diagonals -/
def diagonalAngle (q : ConvexQuadrilateral) : ℝ := sorry

/-- Theorem stating the tangent of the diagonal angle -/
theorem diagonal_angle_tangent (q : ConvexQuadrilateral) 
  (h1 : q.area = 30)
  (h2 : q.side1 = 5)
  (h3 : q.side2 = 6)
  (h4 : q.side3 = 9)
  (h5 : q.side4 = 7)
  (h6 : q.convex = true) :
  Real.tan (diagonalAngle q) = 40 / 7 := by sorry

end diagonal_angle_tangent_l3522_352200


namespace parabola_line_intersection_l3522_352247

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line structure -/
structure Line where
  m : ℝ
  b : ℝ

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circle structure -/
structure Circle where
  center : Point
  radius : ℝ

/-- Theorem statement -/
theorem parabola_line_intersection (C : Parabola) (l : Line) (M N : Point) (directrix : Line) :
  (l.m = -Real.sqrt 3 ∧ l.b = Real.sqrt 3) →  -- Line equation: y = -√3(x-1)
  (Point.mk (C.p / 2) 0).y = l.m * (Point.mk (C.p / 2) 0).x + l.b →  -- Line passes through focus
  (M.y^2 = 2 * C.p * M.x ∧ N.y^2 = 2 * C.p * N.x) →  -- M and N are on the parabola
  (directrix.m = 0 ∧ directrix.b = -C.p / 2) →  -- Directrix equation: x = -p/2
  (C.p = 2 ∧  -- First conclusion: p = 2
   ∃ (circ : Circle), circ.center = Point.mk ((M.x + N.x) / 2) ((M.y + N.y) / 2) ∧
                      circ.radius = abs ((M.x - N.x) / 2) ∧
                      abs (circ.center.x - (-C.p / 2)) = circ.radius)  -- Second conclusion: Circle tangent to directrix
  := by sorry

end parabola_line_intersection_l3522_352247


namespace vector_not_parallel_l3522_352260

def a : ℝ × ℝ := (1, -2)

theorem vector_not_parallel (k : ℝ) : 
  ¬ ∃ (t : ℝ), (k^2 + 1, k^2 + 1) = t • a := by sorry

end vector_not_parallel_l3522_352260


namespace quadratic_decreasing_implies_m_geq_1_l3522_352211

/-- The quadratic function f(x) = x² - 2mx + 5 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 5

/-- f(x) is decreasing for all x < 1 -/
def is_decreasing_before_1 (m : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → x₂ < 1 → f m x₁ > f m x₂

theorem quadratic_decreasing_implies_m_geq_1 (m : ℝ) :
  is_decreasing_before_1 m → m ≥ 1 := by sorry

end quadratic_decreasing_implies_m_geq_1_l3522_352211


namespace DE_length_l3522_352232

-- Define the fixed points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop :=
  (P.1 + 2)^2 + P.2^2 = 4 * ((P.1 - 1)^2 + P.2^2)

-- Define the line l
def l (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 - 3}

-- Define the intersection points D and E
def intersection_points (k : ℝ) : Prop :=
  ∃ (D E : ℝ × ℝ), D ∈ C ∩ l k ∧ E ∈ C ∩ l k ∧ D ≠ E

-- Define the condition x₁x₂ + y₁y₂ = 3
def point_product_condition (D E : ℝ × ℝ) : Prop :=
  D.1 * E.1 + D.2 * E.2 = 3

-- Theorem statement
theorem DE_length :
  ∀ (k : ℝ) (D E : ℝ × ℝ),
  k > 5/12 →
  intersection_points k →
  point_product_condition D E →
  (D.1 - E.1)^2 + (D.2 - E.2)^2 = 14 := by
  sorry

end DE_length_l3522_352232


namespace car_value_correct_l3522_352248

/-- The value of the car Lil Jon bought for DJ Snake's engagement -/
def car_value : ℕ := 30000

/-- The cost of the hotel stay per night -/
def hotel_cost_per_night : ℕ := 4000

/-- The number of nights stayed at the hotel -/
def nights_stayed : ℕ := 2

/-- The total value of all treats received -/
def total_value : ℕ := 158000

/-- Theorem stating that the car value is correct given the conditions -/
theorem car_value_correct :
  car_value = 30000 ∧
  hotel_cost_per_night = 4000 ∧
  nights_stayed = 2 ∧
  total_value = 158000 ∧
  (hotel_cost_per_night * nights_stayed + car_value + 4 * car_value = total_value) :=
by sorry

end car_value_correct_l3522_352248


namespace clarence_oranges_l3522_352204

/-- The number of oranges Clarence has initially -/
def initial_oranges : ℕ := 5

/-- The number of oranges Clarence receives from Joyce -/
def oranges_from_joyce : ℕ := 3

/-- The total number of oranges Clarence has -/
def total_oranges : ℕ := initial_oranges + oranges_from_joyce

theorem clarence_oranges : total_oranges = 8 := by
  sorry

end clarence_oranges_l3522_352204


namespace x0_range_l3522_352243

/-- Circle C with equation x^2 + y^2 = 1 -/
def Circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

/-- Line l with equation 3x + 2y - 4 = 0 -/
def Line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 + 2 * p.2 - 4 = 0}

/-- Condition that there always exist two different points A, B on circle C such that OA + OB = OP -/
def ExistPoints (P : ℝ × ℝ) : Prop :=
  ∀ A B : ℝ × ℝ, A ∈ Circle_C → B ∈ Circle_C → A ≠ B → 
    (A.1, A.2) + (B.1, B.2) = P

theorem x0_range (x0 y0 : ℝ) (hP : (x0, y0) ∈ Line_l) 
    (hExist : ExistPoints (x0, y0)) : 
  0 < x0 ∧ x0 < 24/13 := by
  sorry

end x0_range_l3522_352243


namespace grace_september_earnings_775_l3522_352239

/-- Represents Grace's landscaping business earnings for September --/
def grace_september_earnings : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  fun small_lawn_rate large_lawn_rate small_garden_rate large_garden_rate small_mulch_rate large_mulch_rate
      small_lawn_hours large_lawn_hours small_garden_hours large_garden_hours small_mulch_hours large_mulch_hours =>
    small_lawn_rate * small_lawn_hours +
    large_lawn_rate * large_lawn_hours +
    small_garden_rate * small_garden_hours +
    large_garden_rate * large_garden_hours +
    small_mulch_rate * small_mulch_hours +
    large_mulch_rate * large_mulch_hours

/-- Theorem stating that Grace's September earnings were $775 --/
theorem grace_september_earnings_775 :
  grace_september_earnings 6 10 11 15 9 13 20 43 4 5 6 4 = 775 := by
  sorry

end grace_september_earnings_775_l3522_352239


namespace arrangement_and_selection_theorem_l3522_352279

def girls : ℕ := 3
def boys : ℕ := 4
def total_people : ℕ := girls + boys

def arrangements_no_adjacent_girls : ℕ := (Nat.factorial boys) * (Nat.choose (boys + 1) girls)

def selections_with_at_least_one_girl : ℕ := Nat.choose total_people 3 - Nat.choose boys 3

theorem arrangement_and_selection_theorem :
  (arrangements_no_adjacent_girls = 1440) ∧
  (selections_with_at_least_one_girl = 31) := by
  sorry

end arrangement_and_selection_theorem_l3522_352279


namespace k_value_for_given_factors_l3522_352229

/-- The length of an integer is the number of positive prime factors, not necessarily distinct, whose product is equal to the integer. -/
def length (n : ℕ) : ℕ := sorry

/-- The prime factors of an integer as a multiset. -/
def primeFactors (n : ℕ) : Multiset ℕ := sorry

theorem k_value_for_given_factors :
  ∀ k : ℕ,
    k > 1 →
    length k = 4 →
    primeFactors k = {2, 2, 2, 3} →
    k = 24 := by
  sorry

end k_value_for_given_factors_l3522_352229


namespace solution_values_l3522_352262

theorem solution_values (x : ℝ) (hx : x^2 + 4 * (x / (x - 2))^2 = 45) :
  let y := ((x - 2)^2 * (x + 3)) / (2*x - 3)
  y = 2 ∨ y = 16 :=
sorry

end solution_values_l3522_352262


namespace perp_lines_parallel_perp_planes_parallel_l3522_352276

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallelLines : Line → Line → Prop)
variable (parallelPlanes : Plane → Plane → Prop)
variable (linePerpToPlane : Line → Plane → Prop)
variable (planePerpToLine : Plane → Line → Prop)

-- Axioms
axiom distinct_lines (a b : Line) : a ≠ b
axiom distinct_planes (α β : Plane) : α ≠ β

-- Theorem 1
theorem perp_lines_parallel (a b : Line) (α : Plane) :
  linePerpToPlane a α → linePerpToPlane b α → parallelLines a b :=
sorry

-- Theorem 2
theorem perp_planes_parallel (a : Line) (α β : Plane) :
  planePerpToLine α a → planePerpToLine β a → parallelPlanes α β :=
sorry

end perp_lines_parallel_perp_planes_parallel_l3522_352276


namespace two_solutions_with_more_sheep_l3522_352267

def budget : ℕ := 800
def goat_cost : ℕ := 15
def sheep_cost : ℕ := 16

def is_valid_solution (g h : ℕ) : Prop :=
  goat_cost * g + sheep_cost * h = budget ∧ h > g

theorem two_solutions_with_more_sheep :
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (g h : ℕ), (g, h) ∈ s ↔ is_valid_solution g h) ∧
    s.card = 2 :=
sorry

end two_solutions_with_more_sheep_l3522_352267


namespace cupcake_distribution_l3522_352295

/-- Given initial cupcakes, eaten cupcakes, and number of packages, 
    calculate the number of cupcakes in each package. -/
def cupcakes_per_package (initial : ℕ) (eaten : ℕ) (packages : ℕ) : ℕ :=
  (initial - eaten) / packages

/-- Theorem stating that with 18 initial cupcakes, 8 eaten cupcakes, 
    and 5 packages, there are 2 cupcakes in each package. -/
theorem cupcake_distribution : cupcakes_per_package 18 8 5 = 2 := by
  sorry

end cupcake_distribution_l3522_352295


namespace rectangular_prism_problem_l3522_352238

theorem rectangular_prism_problem (m n r : ℕ) : 
  m > 0 → n > 0 → r > 0 → m ≤ n → n ≤ r →
  (m - 2) * (n - 2) * (r - 2) - 
  2 * ((m - 2) * (n - 2) + (n - 2) * (r - 2) + (r - 2) * (m - 2)) + 
  4 * ((m - 2) + (n - 2) + (r - 2)) = 1985 →
  ((m = 1 ∧ n = 3 ∧ r = 1987) ∨
   (m = 1 ∧ n = 7 ∧ r = 399) ∨
   (m = 3 ∧ n = 3 ∧ r = 1981) ∨
   (m = 5 ∧ n = 5 ∧ r = 1981) ∨
   (m = 5 ∧ n = 7 ∧ r = 663)) :=
by sorry

end rectangular_prism_problem_l3522_352238


namespace exponential_inequality_l3522_352266

theorem exponential_inequality (m : ℝ) (h : 0 < m ∧ m < 1) :
  (1 - m) ^ (1/3 : ℝ) > (1 - m) ^ (1/2 : ℝ) := by
  sorry

end exponential_inequality_l3522_352266


namespace david_scott_age_difference_l3522_352288

/-- Represents the ages of three brothers -/
structure BrotherAges where
  richard : ℕ
  david : ℕ
  scott : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : BrotherAges) : Prop :=
  ages.richard = ages.david + 6 ∧
  ages.david > ages.scott ∧
  ages.richard + 8 = 2 * (ages.scott + 8) ∧
  ages.david = 14

/-- The theorem to be proved -/
theorem david_scott_age_difference (ages : BrotherAges) :
  problem_conditions ages → ages.david - ages.scott = 8 := by
  sorry


end david_scott_age_difference_l3522_352288


namespace wrong_mark_calculation_l3522_352298

theorem wrong_mark_calculation (total_marks : ℝ) : 
  let n : ℕ := 40
  let correct_mark : ℝ := 63
  let wrong_mark : ℝ := (total_marks - correct_mark + n / 2) / (1 - 1 / n)
  wrong_mark = 43 := by
  sorry

end wrong_mark_calculation_l3522_352298


namespace simplify_expression_l3522_352250

theorem simplify_expression (x : ℝ) : ((3 * x + 8) - 5 * x) / 2 = -x + 4 := by
  sorry

end simplify_expression_l3522_352250


namespace cookie_distribution_l3522_352209

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) 
  (h1 : total_cookies = 35)
  (h2 : num_people = 5)
  (h3 : total_cookies = num_people * cookies_per_person) :
  cookies_per_person = 7 := by
  sorry

end cookie_distribution_l3522_352209


namespace inscribed_rhombus_triangle_sides_l3522_352225

/-- A triangle with an inscribed rhombus -/
structure InscribedRhombusTriangle where
  -- Side lengths of the triangle
  BC : ℝ
  AB : ℝ
  AC : ℝ
  -- Length of rhombus side
  m : ℝ
  -- Segments of BC
  p : ℝ
  q : ℝ
  -- Conditions
  rhombus_inscribed : m > 0
  positive_segments : p > 0 ∧ q > 0
  k_on_bc : BC = p + q

/-- Theorem: The sides of the triangle with an inscribed rhombus -/
theorem inscribed_rhombus_triangle_sides 
  (t : InscribedRhombusTriangle) : 
  t.BC = t.p + t.q ∧ 
  t.AB = t.m * (t.p + t.q) / t.q ∧ 
  t.AC = t.m * (t.p + t.q) / t.p :=
by sorry

end inscribed_rhombus_triangle_sides_l3522_352225


namespace airplane_travel_time_l3522_352210

/-- Proves that the time taken for an airplane to travel against the wind is 5 hours -/
theorem airplane_travel_time 
  (distance : ℝ) 
  (return_time : ℝ) 
  (still_air_speed : ℝ) 
  (h1 : distance = 3600) 
  (h2 : return_time = 4) 
  (h3 : still_air_speed = 810) : 
  (distance / (still_air_speed - (distance / return_time - still_air_speed))) = 5 := by
  sorry

end airplane_travel_time_l3522_352210


namespace arithmetic_sequence_proof_l3522_352254

theorem arithmetic_sequence_proof (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, 2 * S n = a n * (a n + 1)) :
  (∀ n, a n = n) ∧ (∀ n, a (n + 1) - a n = 1) :=
sorry

end arithmetic_sequence_proof_l3522_352254


namespace mixture_salt_concentration_l3522_352291

/-- Represents the concentration of a solution as a real number between 0 and 1 -/
def Concentration := { c : ℝ // 0 ≤ c ∧ c ≤ 1 }

/-- Calculates the concentration of salt in a mixture of pure water and salt solution -/
def mixtureSaltConcentration (pureWaterVolume : ℝ) (saltSolutionVolume : ℝ) (saltSolutionConcentration : Concentration) : Concentration :=
  sorry

/-- Theorem: The concentration of salt in a mixture of 1 liter of pure water and 0.2 liters of 60% salt solution is 10% -/
theorem mixture_salt_concentration :
  let pureWaterVolume : ℝ := 1
  let saltSolutionVolume : ℝ := 0.2
  let saltSolutionConcentration : Concentration := ⟨0.6, by sorry⟩
  let resultingConcentration : Concentration := mixtureSaltConcentration pureWaterVolume saltSolutionVolume saltSolutionConcentration
  resultingConcentration.val = 0.1 := by sorry

end mixture_salt_concentration_l3522_352291


namespace cake_division_l3522_352275

theorem cake_division (pooh_initial piglet_initial : ℚ) : 
  pooh_initial + piglet_initial = 1 →
  piglet_initial + (1/3) * pooh_initial = 3 * piglet_initial →
  pooh_initial = 6/7 ∧ piglet_initial = 1/7 := by
  sorry

end cake_division_l3522_352275


namespace final_sign_is_minus_l3522_352259

/-- Represents the two possible signs on the board -/
inductive Sign
| Plus
| Minus

/-- Represents the state of the board -/
structure Board :=
  (plusCount : Nat)
  (minusCount : Nat)

/-- Applies the transformation rule to two signs -/
def transform (s1 s2 : Sign) : Sign :=
  match s1, s2 with
  | Sign.Plus, Sign.Plus => Sign.Plus
  | Sign.Minus, Sign.Minus => Sign.Plus
  | _, _ => Sign.Minus

/-- Theorem stating that the final sign will be minus -/
theorem final_sign_is_minus 
  (initial : Board)
  (h_initial_plus : initial.plusCount = 2004)
  (h_initial_minus : initial.minusCount = 2005) :
  ∃ (final : Board), final.plusCount + final.minusCount = 1 ∧ final.minusCount = 1 := by
  sorry


end final_sign_is_minus_l3522_352259


namespace exists_year_with_special_form_l3522_352218

def is_21st_century (y : ℕ) : Prop := 2001 ≤ y ∧ y ≤ 2100

def are_distinct_digits (a b c d e f g h i j : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem exists_year_with_special_form :
  ∃ (y : ℕ) (a b c d e f g h i j : ℕ),
    is_21st_century y ∧
    are_distinct_digits a b c d e f g h i j ∧
    is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ is_digit e ∧
    is_digit f ∧ is_digit g ∧ is_digit h ∧ is_digit i ∧ is_digit j ∧
    y = (a + b * c * d * e) / (f + g * h * i * j) :=
sorry

end exists_year_with_special_form_l3522_352218


namespace systematic_sample_property_l3522_352287

/-- Represents a systematic sample from a class -/
structure SystematicSample where
  class_size : ℕ
  sample_size : ℕ
  known_seats : Finset ℕ
  h_class_size : class_size > 0
  h_sample_size : sample_size > 0
  h_sample_size_le : sample_size ≤ class_size
  h_known_seats : known_seats.card < sample_size

/-- The seat number of the missing student in the systematic sample -/
def missing_seat (s : SystematicSample) : ℕ := sorry

/-- Theorem stating the property of the systematic sample -/
theorem systematic_sample_property (s : SystematicSample) 
  (h_seats : s.known_seats = {3, 15, 39, 51}) 
  (h_class_size : s.class_size = 60) 
  (h_sample_size : s.sample_size = 5) : 
  missing_seat s = 27 := by sorry

end systematic_sample_property_l3522_352287


namespace base_4_7_digit_difference_l3522_352223

def num_digits (n : ℕ) (base : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log base n + 1

theorem base_4_7_digit_difference : 
  num_digits 4563 4 - num_digits 4563 7 = 2 :=
by
  sorry

end base_4_7_digit_difference_l3522_352223


namespace keychain_cost_is_five_l3522_352251

/-- The cost of a bracelet in dollars -/
def bracelet_cost : ℝ := 4

/-- The cost of a coloring book in dollars -/
def coloring_book_cost : ℝ := 3

/-- The cost of Paula's purchase in dollars -/
def paula_cost (keychain_cost : ℝ) : ℝ := 2 * bracelet_cost + keychain_cost

/-- The cost of Olive's purchase in dollars -/
def olive_cost : ℝ := coloring_book_cost + bracelet_cost

/-- The total amount spent by Paula and Olive in dollars -/
def total_spent : ℝ := 20

/-- Theorem stating that the keychain cost is 5 dollars -/
theorem keychain_cost_is_five : 
  ∃ (keychain_cost : ℝ), paula_cost keychain_cost + olive_cost = total_spent ∧ keychain_cost = 5 :=
sorry

end keychain_cost_is_five_l3522_352251


namespace library_visitors_l3522_352284

/-- Calculates the average number of visitors on non-Sunday days in a library -/
theorem library_visitors (total_days : Nat) (sunday_visitors : Nat) (avg_visitors : Nat) :
  total_days = 30 ∧ 
  sunday_visitors = 510 ∧ 
  avg_visitors = 285 →
  (total_days * avg_visitors - 5 * sunday_visitors) / 25 = 240 := by
sorry

end library_visitors_l3522_352284


namespace perpendicular_bisector_intersection_l3522_352214

/-- The perpendicular bisector of two points A and B intersects the line AB at a point C.
    This theorem proves that for specific points A and B, the coordinates of C satisfy a linear equation. -/
theorem perpendicular_bisector_intersection (A B C : ℝ × ℝ) :
  A = (30, 10) →
  B = (6, 3) →
  C.1 = (A.1 + B.1) / 2 →
  C.2 = (A.2 + B.2) / 2 →
  2 * C.1 - 4 * C.2 = 10 := by
  sorry


end perpendicular_bisector_intersection_l3522_352214


namespace greatest_abcba_divisible_by_13_l3522_352292

/-- Represents a five-digit number in the form AB,CBA -/
def abcba (a b c : Nat) : Nat := 10000 * a + 1000 * b + 100 * c + 10 * b + a

/-- Check if three digits are distinct -/
def distinct_digits (a b c : Nat) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem greatest_abcba_divisible_by_13 :
  ∀ a b c : Nat,
  a < 10 → b < 10 → c < 10 →
  distinct_digits a b c →
  abcba a b c ≤ 99999 →
  abcba a b c ≡ 0 [MOD 13] →
  abcba a b c ≤ 95159 :=
sorry

end greatest_abcba_divisible_by_13_l3522_352292


namespace currency_notes_count_l3522_352217

theorem currency_notes_count (total_amount : ℕ) (amount_in_50 : ℕ) (denomination_50 : ℕ) (denomination_100 : ℕ) :
  total_amount = 5000 →
  amount_in_50 = 3500 →
  denomination_50 = 50 →
  denomination_100 = 100 →
  (amount_in_50 / denomination_50 + (total_amount - amount_in_50) / denomination_100 : ℕ) = 85 :=
by sorry

end currency_notes_count_l3522_352217


namespace regular_polygon_with_150_degree_angles_has_12_sides_l3522_352265

/-- A regular polygon with interior angles measuring 150 degrees has 12 sides. -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (180 * (n - 2) : ℝ) = (150 * n : ℝ) →
  n = 12 := by
sorry

end regular_polygon_with_150_degree_angles_has_12_sides_l3522_352265


namespace max_pairs_from_27_l3522_352235

theorem max_pairs_from_27 (n : ℕ) (h : n = 27) :
  (n * (n - 1)) / 2 = 351 := by
  sorry

end max_pairs_from_27_l3522_352235


namespace absolute_value_equation_unique_solution_l3522_352278

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x + 4| :=
sorry

end absolute_value_equation_unique_solution_l3522_352278


namespace quadrilateral_ABCD_area_l3522_352281

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (A B C D : Point) : ℝ := sorry

theorem quadrilateral_ABCD_area :
  let A : Point := ⟨0, 1⟩
  let B : Point := ⟨1, 3⟩
  let C : Point := ⟨5, 2⟩
  let D : Point := ⟨4, 0⟩
  quadrilateralArea A B C D = 9 := by sorry

end quadrilateral_ABCD_area_l3522_352281


namespace pentadecagon_triangles_l3522_352273

/-- The number of vertices in a regular pentadecagon -/
def n : ℕ := 15

/-- The number of vertices required to form a triangle -/
def r : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon -/
def num_triangles : ℕ := Nat.choose n r

theorem pentadecagon_triangles : num_triangles = 455 := by
  sorry

end pentadecagon_triangles_l3522_352273


namespace sequence_general_formula_l3522_352230

theorem sequence_general_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = 3^n - 2) →
  (∀ n : ℕ, n ≥ 1 → a n = S n - S (n-1)) →
  (∀ n : ℕ, n = 1 → a n = 1) ∧ 
  (∀ n : ℕ, n ≥ 2 → a n = 2 * 3^(n-1)) :=
by sorry

end sequence_general_formula_l3522_352230


namespace sphere_wedge_volume_l3522_352263

/-- The volume of a wedge from a sphere -/
theorem sphere_wedge_volume (circumference : ℝ) (num_wedges : ℕ) : 
  circumference = 18 * Real.pi → num_wedges = 6 →
  (4 / 3 * Real.pi * (circumference / (2 * Real.pi))^3) / num_wedges = 162 * Real.pi := by
  sorry

end sphere_wedge_volume_l3522_352263


namespace isosceles_triangle_side_lengths_l3522_352236

/-- An isosceles triangle with one side of length 7 and perimeter 17 has other sides of lengths (5, 5) or (7, 3) -/
theorem isosceles_triangle_side_lengths :
  ∀ (a b c : ℝ),
  a = 7 ∧ 
  a + b + c = 17 ∧
  ((b = c) ∨ (a = b) ∨ (a = c)) →
  ((b = 5 ∧ c = 5) ∨ (b = 7 ∧ c = 3) ∨ (b = 3 ∧ c = 7)) :=
by sorry

end isosceles_triangle_side_lengths_l3522_352236


namespace modulus_of_one_minus_i_l3522_352226

theorem modulus_of_one_minus_i :
  let z : ℂ := 1 - Complex.I
  Complex.abs z = Real.sqrt 2 := by sorry

end modulus_of_one_minus_i_l3522_352226


namespace production_exceeds_target_l3522_352257

/-- The initial production in 2014 -/
def initial_production : ℕ := 40000

/-- The annual increase rate -/
def increase_rate : ℚ := 1/5

/-- The target production to exceed -/
def target_production : ℕ := 120000

/-- The logarithm of 2 -/
def log_2 : ℚ := 3010/10000

/-- The logarithm of 3 -/
def log_3 : ℚ := 4771/10000

/-- The number of years after 2014 when production exceeds the target -/
def years_to_exceed_target : ℕ := 7

theorem production_exceeds_target :
  years_to_exceed_target = 
    (Nat.ceil (log_3 / (increase_rate * log_2))) :=
by sorry

end production_exceeds_target_l3522_352257


namespace no_reversed_arithmetic_progression_l3522_352285

/-- Function that returns the odd positive integer obtained by reversing the binary representation of n -/
def r (n : Nat) : Nat :=
  sorry

/-- Predicate to check if a sequence is an arithmetic progression -/
def isArithmeticProgression (s : List Nat) : Prop :=
  sorry

theorem no_reversed_arithmetic_progression :
  ¬∃ (a : Fin 8 → Nat),
    (∀ i : Fin 8, Odd (a i)) ∧
    (∀ i j : Fin 8, i < j → a i < a j) ∧
    isArithmeticProgression (List.ofFn a) ∧
    isArithmeticProgression (List.map r (List.ofFn a)) :=
  sorry

end no_reversed_arithmetic_progression_l3522_352285


namespace power_six_sum_l3522_352208

theorem power_six_sum (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12098 := by
  sorry

end power_six_sum_l3522_352208


namespace parallelogram_problem_l3522_352294

-- Define a parallelogram
structure Parallelogram :=
  (EF GH FG HE : ℝ)
  (is_parallelogram : EF = GH ∧ FG = HE)

-- Define the problem
theorem parallelogram_problem (EFGH : Parallelogram)
  (h1 : EFGH.EF = 52)
  (h2 : ∃ z : ℝ, EFGH.FG = 2 * z^4)
  (h3 : ∃ w : ℝ, EFGH.GH = 3 * w + 6)
  (h4 : EFGH.HE = 16) :
  ∃ w z : ℝ, w * z = 46 * Real.sqrt 2 / 3 :=
sorry

end parallelogram_problem_l3522_352294


namespace swimming_speed_in_still_water_l3522_352215

/-- 
Given a person swimming against a current, prove their swimming speed in still water.
-/
theorem swimming_speed_in_still_water 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : current_speed = 2) 
  (h2 : distance = 6) 
  (h3 : time = 3) : 
  ∃ (still_water_speed : ℝ), 
    still_water_speed = 4 ∧ 
    distance = (still_water_speed - current_speed) * time := by
  sorry

end swimming_speed_in_still_water_l3522_352215


namespace parallel_vectors_tan_sum_l3522_352297

/-- Given two parallel vectors a and b, prove that tan(α + π/4) = 7 --/
theorem parallel_vectors_tan_sum (α : ℝ) : 
  let a : ℝ × ℝ := (3, 4)
  let b : ℝ × ℝ := (Real.sin α, Real.cos α)
  (∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2) →  -- Parallel vectors condition
  Real.tan (α + π/4) = 7 := by
  sorry

end parallel_vectors_tan_sum_l3522_352297


namespace bell_interval_problem_l3522_352264

theorem bell_interval_problem (x : ℕ+) : 
  Nat.lcm x (Nat.lcm 10 (Nat.lcm 14 18)) = 630 → x = 1 := by
  sorry

end bell_interval_problem_l3522_352264


namespace ship_speed_calculation_l3522_352286

theorem ship_speed_calculation (total_distance : ℝ) (travel_time : ℝ) (backward_distance : ℝ) :
  travel_time = 20 ∧
  backward_distance = 200 ∧
  total_distance / 2 - total_distance / 3 = backward_distance →
  (total_distance / 2) / travel_time = 30 := by
sorry

end ship_speed_calculation_l3522_352286


namespace lily_shopping_theorem_l3522_352245

/-- Calculates the remaining amount for coffee after Lily's shopping trip --/
def remaining_for_coffee (initial_amount : ℚ) (celery_cost : ℚ) (cereal_cost : ℚ) (cereal_discount : ℚ)
  (bread_cost : ℚ) (milk_cost : ℚ) (milk_discount : ℚ) (potato_cost : ℚ) (potato_quantity : ℕ) : ℚ :=
  initial_amount - (celery_cost + cereal_cost * (1 - cereal_discount) + bread_cost + 
  milk_cost * (1 - milk_discount) + potato_cost * potato_quantity)

theorem lily_shopping_theorem (initial_amount : ℚ) (celery_cost : ℚ) (cereal_cost : ℚ) (cereal_discount : ℚ)
  (bread_cost : ℚ) (milk_cost : ℚ) (milk_discount : ℚ) (potato_cost : ℚ) (potato_quantity : ℕ) :
  initial_amount = 60 ∧ 
  celery_cost = 5 ∧ 
  cereal_cost = 12 ∧ 
  cereal_discount = 0.5 ∧ 
  bread_cost = 8 ∧ 
  milk_cost = 10 ∧ 
  milk_discount = 0.1 ∧ 
  potato_cost = 1 ∧ 
  potato_quantity = 6 →
  remaining_for_coffee initial_amount celery_cost cereal_cost cereal_discount bread_cost milk_cost milk_discount potato_cost potato_quantity = 26 := by
  sorry

#eval remaining_for_coffee 60 5 12 0.5 8 10 0.1 1 6

end lily_shopping_theorem_l3522_352245


namespace orange_count_difference_l3522_352213

/-- Proves that the difference between Marcie's and Brian's orange counts is 0 -/
theorem orange_count_difference (marcie_oranges brian_oranges : ℕ) 
  (h1 : marcie_oranges = 12) (h2 : brian_oranges = 12) : 
  marcie_oranges - brian_oranges = 0 := by
  sorry

end orange_count_difference_l3522_352213


namespace triangle_angle_difference_l3522_352270

theorem triangle_angle_difference (a b c : ℝ) : 
  a = 32 →
  b = 96 →
  c = 52 →
  b = 3 * a →
  2 * a - c = 12 :=
by
  sorry

end triangle_angle_difference_l3522_352270


namespace ratio_simplification_l3522_352233

theorem ratio_simplification (a b c d : ℚ) (m n : ℕ) :
  (a : ℚ) / (b : ℚ) = (c : ℚ) / (d : ℚ) →
  (m : ℚ) / (n : ℚ) = ((250 : ℚ) * 1000) / ((2 : ℚ) / 5 * 1000000) →
  (1.25 : ℚ) / (5 / 8 : ℚ) = (2 : ℚ) / (1 : ℚ) ∧
  (m : ℚ) / (n : ℚ) = (5 : ℚ) / (8 : ℚ) := by
  sorry

end ratio_simplification_l3522_352233


namespace simultaneous_presence_probability_l3522_352241

/-- The probability of two people being at a location simultaneously -/
theorem simultaneous_presence_probability :
  let arrival_window : ℝ := 2  -- 2-hour window
  let stay_duration : ℝ := 1/3  -- 20 minutes in hours
  let total_area : ℝ := arrival_window * arrival_window
  let meeting_area : ℝ := total_area - 2 * (1/2 * stay_duration * (arrival_window - stay_duration))
  meeting_area / total_area = 4/9 := by
sorry

end simultaneous_presence_probability_l3522_352241


namespace geometric_sequence_second_term_l3522_352216

theorem geometric_sequence_second_term 
  (a₁ : ℝ) 
  (a₃ : ℝ) 
  (b : ℝ) 
  (h₁ : a₁ = 120) 
  (h₂ : a₃ = 64 / 30) 
  (h₃ : b > 0) 
  (h₄ : ∃ r : ℝ, a₁ * r = b ∧ b * r = a₃) : 
  b = 16 := by
sorry


end geometric_sequence_second_term_l3522_352216


namespace yellow_peaches_count_red_yellow_relation_l3522_352202

/-- The number of yellow peaches in the basket -/
def yellow_peaches : ℕ := 11

/-- The number of red peaches in the basket -/
def red_peaches : ℕ := 19

/-- The number of green peaches in the basket -/
def green_peaches : ℕ := 12

/-- The difference between red and yellow peaches -/
def red_yellow_difference : ℕ := 8

theorem yellow_peaches_count : yellow_peaches = 11 := by
  sorry

theorem red_yellow_relation : red_peaches = yellow_peaches + red_yellow_difference := by
  sorry

end yellow_peaches_count_red_yellow_relation_l3522_352202


namespace geometric_sequence_ratio_l3522_352258

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a2 : a 2 = 1/2) 
  (h_a5 : a 5 = 4) : 
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ, a (n + 1) = a n * q := by
sorry

end geometric_sequence_ratio_l3522_352258


namespace greaterThanOne_is_random_event_l3522_352274

-- Define the type for outcomes of rolling a die
def DieOutcome := Fin 6

-- Define the event "greater than 1"
def greaterThanOne (outcome : DieOutcome) : Prop := outcome.val > 1

-- Define what it means for an event to be random
def isRandomEvent (event : DieOutcome → Prop) : Prop :=
  ∃ (o1 o2 : DieOutcome), event o1 ∧ ¬event o2

-- Theorem stating that "greater than 1" is a random event
theorem greaterThanOne_is_random_event : isRandomEvent greaterThanOne := by
  sorry


end greaterThanOne_is_random_event_l3522_352274


namespace johnny_red_pencils_l3522_352234

/-- The number of red pencils Johnny bought -/
def total_red_pencils (total_packs : ℕ) (regular_red_per_pack : ℕ) 
  (extra_red_packs_1 : ℕ) (extra_red_per_pack_1 : ℕ)
  (extra_red_packs_2 : ℕ) (extra_red_per_pack_2 : ℕ) : ℕ :=
  total_packs * regular_red_per_pack + 
  extra_red_packs_1 * extra_red_per_pack_1 +
  extra_red_packs_2 * extra_red_per_pack_2

/-- Theorem: Johnny bought 46 red pencils -/
theorem johnny_red_pencils : 
  total_red_pencils 25 1 5 3 6 1 = 46 := by
  sorry

end johnny_red_pencils_l3522_352234


namespace tim_income_percentage_tim_income_less_than_juan_l3522_352220

theorem tim_income_percentage (tim mary juan : ℝ) 
  (h1 : mary = 1.5 * tim) 
  (h2 : mary = 0.8999999999999999 * juan) : 
  tim = 0.6 * juan := by
  sorry

theorem tim_income_less_than_juan (tim juan : ℝ) 
  (h : tim = 0.6 * juan) : 
  (juan - tim) / juan = 0.4 := by
  sorry

end tim_income_percentage_tim_income_less_than_juan_l3522_352220


namespace num_valid_schedules_l3522_352231

/-- Represents the number of periods in a day -/
def num_periods : ℕ := 8

/-- Represents the number of courses to be scheduled -/
def num_courses : ℕ := 4

/-- 
Calculates the number of ways to schedule courses with exactly one consecutive pair
num_periods: The total number of periods in a day
num_courses: The number of courses to be scheduled
-/
def schedule_with_one_consecutive_pair (num_periods : ℕ) (num_courses : ℕ) : ℕ := sorry

/-- The main theorem stating the number of valid schedules -/
theorem num_valid_schedules : 
  schedule_with_one_consecutive_pair num_periods num_courses = 1680 := by sorry

end num_valid_schedules_l3522_352231


namespace solution_part1_solution_part2_l3522_352255

/-- The fractional equation -/
def fractional_equation (x a : ℝ) : Prop :=
  (x + a) / (x - 2) - 5 / x = 1

theorem solution_part1 :
  ∀ a : ℝ, fractional_equation 5 a → a = 1 := by sorry

theorem solution_part2 :
  fractional_equation (-5) 5 := by sorry

end solution_part1_solution_part2_l3522_352255


namespace same_color_probability_is_seven_ninths_l3522_352253

/-- Represents a die with a specific number of sides and color distribution -/
structure Die where
  sides : ℕ
  red : ℕ
  blue : ℕ
  green : ℕ
  valid : red + blue + green = sides

/-- Calculate the probability of two dice showing the same color -/
def same_color_probability (d1 d2 : Die) : ℚ :=
  let p_red := (d1.red : ℚ) / d1.sides * (d2.red : ℚ) / d2.sides
  let p_blue := (d1.blue : ℚ) / d1.sides * (d2.blue : ℚ) / d2.sides
  let p_green := (d1.green : ℚ) / d1.sides * (d2.green : ℚ) / d2.sides
  p_red + p_blue + p_green

/-- The first die with 12 sides: 3 red, 4 blue, 5 green -/
def die1 : Die := {
  sides := 12,
  red := 3,
  blue := 4,
  green := 5,
  valid := by simp
}

/-- The second die with 15 sides: 5 red, 3 blue, 7 green -/
def die2 : Die := {
  sides := 15,
  red := 5,
  blue := 3,
  green := 7,
  valid := by simp
}

/-- Theorem stating that the probability of both dice showing the same color is 7/9 -/
theorem same_color_probability_is_seven_ninths :
  same_color_probability die1 die2 = 7 / 9 := by
  sorry

end same_color_probability_is_seven_ninths_l3522_352253


namespace color_film_fraction_l3522_352293

/-- Given a committee reviewing films for a festival, this theorem proves
    the fraction of selected films that are in color. -/
theorem color_film_fraction
  (x y : ℕ) -- x and y are natural numbers
  (total_bw : ℕ := 40 * x) -- Total number of black-and-white films
  (total_color : ℕ := 10 * y) -- Total number of color films
  (bw_selected_percent : ℚ := y / x) -- Percentage of black-and-white films selected
  (color_selected_percent : ℚ := 1) -- All color films are selected
  : (total_color : ℚ) / ((bw_selected_percent * total_bw + total_color) : ℚ) = 5 / 26 :=
sorry

end color_film_fraction_l3522_352293


namespace cube_sum_problem_l3522_352244

theorem cube_sum_problem (x y : ℝ) (h1 : x^3 + y^3 = 7) (h2 : x^6 + y^6 = 49) : 
  x^9 + y^9 = 343 := by
  sorry

end cube_sum_problem_l3522_352244


namespace sin_transformation_l3522_352203

theorem sin_transformation (x : ℝ) : 
  2 * Real.sin (x / 3 + π / 6) = 2 * Real.sin ((3 * x + π) / 3) := by
  sorry

end sin_transformation_l3522_352203


namespace linear_approximation_of_f_l3522_352219

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 5)

theorem linear_approximation_of_f :
  let a : ℝ := 2
  let x : ℝ := 1.97
  let f_a : ℝ := f a
  let f'_a : ℝ := a / Real.sqrt (a^2 + 5)
  let Δx : ℝ := x - a
  let approximation : ℝ := f_a + f'_a * Δx
  ∃ ε > 0, |approximation - 2.98| < ε :=
by
  sorry

#check linear_approximation_of_f

end linear_approximation_of_f_l3522_352219


namespace bus_trip_speed_l3522_352212

theorem bus_trip_speed (distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) :
  distance = 360 ∧ speed_increase = 5 ∧ time_decrease = 1 →
  ∃ (v : ℝ), v > 0 ∧ distance / v - time_decrease = distance / (v + speed_increase) ∧ v = 40 := by
sorry

end bus_trip_speed_l3522_352212


namespace calculate_expression_l3522_352201

theorem calculate_expression : 2 * (-3)^3 - 4 * (-3) + 15 = -27 := by
  sorry

end calculate_expression_l3522_352201


namespace double_reflection_of_H_l3522_352261

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_line (p : ℝ × ℝ) : ℝ × ℝ := (p.2 + 1, p.1 + 1)

def H : ℝ × ℝ := (5, 1)

theorem double_reflection_of_H :
  reflect_line (reflect_x H) = (0, 4) := by sorry

end double_reflection_of_H_l3522_352261


namespace perfect_square_condition_l3522_352272

theorem perfect_square_condition (Z K : ℤ) : 
  (1000 < Z) → (Z < 5000) → (K > 1) → (Z = K * K^2) → 
  (∃ (n : ℤ), Z = n^2) → (K = 16) :=
by sorry

end perfect_square_condition_l3522_352272


namespace evaluate_expression_l3522_352249

theorem evaluate_expression : 
  Real.sqrt (9/4) - Real.sqrt (4/9) + (Real.sqrt (9/4) + Real.sqrt (4/9))^2 = 199/36 := by
  sorry

end evaluate_expression_l3522_352249


namespace triangle345_circle1_common_points_l3522_352282

/-- Represents the number of common points between a triangle and a circle -/
inductive CommonPoints
  | Zero
  | One
  | Two
  | Four

/-- A triangle with side lengths 3, 4, and 5 -/
structure Triangle345 where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side1_eq : side1 = 3
  side2_eq : side2 = 4
  side3_eq : side3 = 5

/-- A circle with radius 1 -/
structure Circle1 where
  radius : ℝ
  radius_eq : radius = 1

/-- The theorem stating the possible numbers of common points -/
theorem triangle345_circle1_common_points (t : Triangle345) (c : Circle1) :
  {cp : CommonPoints | cp = CommonPoints.Zero ∨ cp = CommonPoints.One ∨ 
                       cp = CommonPoints.Two ∨ cp = CommonPoints.Four} = 
  {CommonPoints.Zero, CommonPoints.One, CommonPoints.Two, CommonPoints.Four} :=
sorry

end triangle345_circle1_common_points_l3522_352282


namespace solution_set_equality_l3522_352237

def solution_set : Set ℝ := {x : ℝ | |x - 1| - |x - 5| < 2}

theorem solution_set_equality : solution_set = Set.Iio 4 := by sorry

end solution_set_equality_l3522_352237


namespace puzzle_palace_spending_l3522_352205

theorem puzzle_palace_spending (initial_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 90)
  (h2 : remaining_amount = 12) :
  initial_amount - remaining_amount = 78 := by
  sorry

end puzzle_palace_spending_l3522_352205


namespace quadratic_completion_square_l3522_352246

theorem quadratic_completion_square (a : ℝ) (n : ℝ) : 
  (∀ x, x^2 + a*x + (1/4 : ℝ) = (x + n)^2 + (1/16 : ℝ)) → 
  a < 0 → 
  a = -((3 : ℝ).sqrt / 2) :=
by sorry

end quadratic_completion_square_l3522_352246


namespace mandy_bike_time_l3522_352228

/-- Represents Mandy's exercise routine --/
structure ExerciseRoutine where
  yoga_time : ℝ
  gym_time : ℝ
  bike_time : ℝ

/-- Theorem: Given Mandy's exercise routine conditions, she spends 18 minutes riding her bike --/
theorem mandy_bike_time (routine : ExerciseRoutine) : 
  routine.yoga_time = 20 →
  routine.gym_time + routine.bike_time = 3/2 * routine.yoga_time →
  routine.gym_time = 2/3 * routine.bike_time →
  routine.bike_time = 18 := by
  sorry


end mandy_bike_time_l3522_352228


namespace range_of_t_for_right_angle_l3522_352289

/-- The theorem stating the range of t for point M(3,t) given the conditions -/
theorem range_of_t_for_right_angle (t : ℝ) : 
  let M : ℝ × ℝ := (3, t)
  let O : ℝ × ℝ := (0, 0)
  let circle_O := {(x, y) : ℝ × ℝ | x^2 + y^2 = 6}
  ∃ (A B : ℝ × ℝ), A ∈ circle_O ∧ B ∈ circle_O ∧ 
    ((M.1 - A.1) * (M.1 - B.1) + (M.2 - A.2) * (M.2 - B.2) = 0) →
  -Real.sqrt 3 ≤ t ∧ t ≤ Real.sqrt 3 :=
by sorry

end range_of_t_for_right_angle_l3522_352289


namespace debate_team_group_size_l3522_352277

theorem debate_team_group_size :
  ∀ (boys girls groups : ℕ),
    boys = 26 →
    girls = 46 →
    groups = 8 →
    (boys + girls) / groups = 9 := by
  sorry

end debate_team_group_size_l3522_352277


namespace sibling_age_sum_l3522_352269

/-- Given the ages of four siblings with specific relationships, prove that the sum of three of their ages is 25. -/
theorem sibling_age_sum : 
  ∀ (juliet maggie ralph nicky : ℕ),
  juliet = maggie + 3 →
  ralph = juliet + 2 →
  2 * nicky = ralph →
  juliet = 10 →
  maggie + ralph + nicky = 25 :=
by
  sorry

end sibling_age_sum_l3522_352269


namespace cosine_sine_inequality_l3522_352290

theorem cosine_sine_inequality (a b : ℝ) 
  (h : ∀ x : ℝ, Real.cos (a * Real.sin x) > Real.sin (b * Real.cos x)) : 
  a^2 + b^2 < (Real.pi^2) / 4 := by
  sorry

end cosine_sine_inequality_l3522_352290


namespace four_people_seven_steps_l3522_352207

/-- The number of ways to arrange n people on m steps with at most k people per step -/
def arrangements (n m k : ℕ) : ℕ := sorry

/-- The number of ways 4 people can stand on 7 steps with at most 3 people per step -/
theorem four_people_seven_steps : arrangements 4 7 3 = 2394 := by sorry

end four_people_seven_steps_l3522_352207


namespace square_root_of_nine_l3522_352271

theorem square_root_of_nine : Real.sqrt 9 = 3 := by
  sorry

end square_root_of_nine_l3522_352271


namespace only_nice_number_l3522_352221

def P (x : ℕ) : ℕ := x + 1
def Q (x : ℕ) : ℕ := x^2 + 1

def is_valid_sequence (s : ℕ → ℕ × ℕ) : Prop :=
  s 1 = (1, 3) ∧ 
  ∀ k, (s (k + 1) = (P (s k).1, Q (s k).2) ∨ s (k + 1) = (Q (s k).1, P (s k).2))

def is_nice (n : ℕ) : Prop :=
  ∃ s, is_valid_sequence s ∧ (s n).1 = (s n).2

theorem only_nice_number : ∀ n : ℕ, is_nice n ↔ n = 3 := by sorry

end only_nice_number_l3522_352221


namespace gcd_lcm_product_24_54_l3522_352224

theorem gcd_lcm_product_24_54 : Nat.gcd 24 54 * Nat.lcm 24 54 = 1296 := by
  sorry

end gcd_lcm_product_24_54_l3522_352224


namespace angle_measure_angle_measure_proof_l3522_352256

theorem angle_measure : ℝ → Prop :=
  fun x =>
    (180 - x = 4 * (90 - x)) →
    x = 60

-- The proof is omitted
theorem angle_measure_proof : ∃ x, angle_measure x :=
  sorry

end angle_measure_angle_measure_proof_l3522_352256


namespace vet_donation_calculation_l3522_352296

/-- Represents the vet fees for different animals --/
structure VetFees where
  dog : ℝ
  cat : ℝ
  rabbit : ℝ
  parrot : ℝ

/-- Represents the number of adoptions for each animal type --/
structure Adoptions where
  dogs : ℕ
  cats : ℕ
  rabbits : ℕ
  parrots : ℕ

/-- Calculates the total vet fees with discounts applied --/
def calculateTotalFees (fees : VetFees) (adoptions : Adoptions) (multiAdoptDiscount : ℝ) 
    (dogCatAdoptions : ℕ) (parrotRabbitAdoptions : ℕ) : ℝ := sorry

/-- Calculates the vet's donation based on the total fees --/
def calculateDonation (totalFees : ℝ) (donationRate : ℝ) : ℝ := sorry

theorem vet_donation_calculation (fees : VetFees) (adoptions : Adoptions) 
    (multiAdoptDiscount : ℝ) (dogCatAdoptions : ℕ) (parrotRabbitAdoptions : ℕ) 
    (donationRate : ℝ) :
  fees.dog = 15 ∧ fees.cat = 13 ∧ fees.rabbit = 10 ∧ fees.parrot = 12 ∧
  adoptions.dogs = 8 ∧ adoptions.cats = 3 ∧ adoptions.rabbits = 5 ∧ adoptions.parrots = 2 ∧
  multiAdoptDiscount = 0.1 ∧ dogCatAdoptions = 2 ∧ parrotRabbitAdoptions = 1 ∧
  donationRate = 1/3 →
  calculateDonation (calculateTotalFees fees adoptions multiAdoptDiscount dogCatAdoptions parrotRabbitAdoptions) donationRate = 54.27 := by
  sorry

end vet_donation_calculation_l3522_352296


namespace extraneous_roots_equation_l3522_352280

theorem extraneous_roots_equation :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧
  (∀ x : ℝ, Real.sqrt (x + 15) - 8 / Real.sqrt (x + 15) = 6 →
    (x = r₁ ∨ x = r₂) ∧
    Real.sqrt (r₁ + 15) - 8 / Real.sqrt (r₁ + 15) ≠ 6 ∧
    Real.sqrt (r₂ + 15) - 8 / Real.sqrt (r₂ + 15) ≠ 6) :=
by
  sorry

end extraneous_roots_equation_l3522_352280


namespace solution_set_when_m_is_one_range_of_m_for_nonempty_solution_l3522_352268

-- Define the function f
def f (x m : ℝ) : ℝ := |2*x - 2| + |x + m|

-- Theorem for part (1)
theorem solution_set_when_m_is_one :
  ∃ (a b : ℝ), a = 0 ∧ b = 4/3 ∧
  (∀ x, f x 1 ≤ 3 ↔ a ≤ x ∧ x ≤ b) :=
sorry

-- Theorem for part (2)
theorem range_of_m_for_nonempty_solution :
  ∃ (lower upper : ℝ), lower = -4 ∧ upper = 2 ∧
  (∀ m, (∃ x, f x m ≤ 3) ↔ lower ≤ m ∧ m ≤ upper) :=
sorry

end solution_set_when_m_is_one_range_of_m_for_nonempty_solution_l3522_352268
