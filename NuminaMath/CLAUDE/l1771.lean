import Mathlib

namespace NUMINAMATH_CALUDE_cos_240_degrees_l1771_177120

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l1771_177120


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1771_177191

/-- Given a point P with coordinates (m+3, m+1) on the x-axis,
    prove that its coordinates are (2, 0) -/
theorem point_on_x_axis (m : ℝ) : 
  let P : ℝ × ℝ := (m + 3, m + 1)
  P.2 = 0 → P = (2, 0) := by
sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1771_177191


namespace NUMINAMATH_CALUDE_max_two_match_winners_100_l1771_177143

/-- Represents a single-elimination tournament --/
structure Tournament :=
  (participants : ℕ)

/-- The number of matches in a single-elimination tournament --/
def num_matches (t : Tournament) : ℕ := t.participants - 1

/-- The maximum number of participants who can win exactly two matches --/
def max_two_match_winners (t : Tournament) : ℕ :=
  (t.participants - 2) / 2

/-- Theorem: In a tournament with 100 participants, the maximum number of participants
    who can win exactly two matches is 49 --/
theorem max_two_match_winners_100 :
  ∀ t : Tournament, t.participants = 100 → max_two_match_winners t = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_two_match_winners_100_l1771_177143


namespace NUMINAMATH_CALUDE_angle_identities_l1771_177119

theorem angle_identities (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α + π / 4) = Real.sqrt 3 / 3) : 
  Real.sin (α + 7 * π / 12) = (Real.sqrt 6 + 3) / 6 ∧ 
  Real.cos (2 * α + π / 6) = (2 * Real.sqrt 6 - 1) / 6 := by
sorry

end NUMINAMATH_CALUDE_angle_identities_l1771_177119


namespace NUMINAMATH_CALUDE_no_solution_l1771_177133

theorem no_solution : ¬∃ (k j x : ℝ), 
  (64 / k = 8) ∧ 
  (k * j = 128) ∧ 
  (j - x = k) ∧ 
  (x^2 + j = 3 * k) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l1771_177133


namespace NUMINAMATH_CALUDE_seating_chart_interpretation_l1771_177187

/-- Represents a seating chart configuration -/
structure SeatingChart where
  columns : ℕ
  rows : ℕ

/-- Interprets a pair of natural numbers as a seating chart -/
def interpretSeatingChart (pair : ℕ × ℕ) : SeatingChart :=
  ⟨pair.1, pair.2⟩

theorem seating_chart_interpretation :
  let chart := interpretSeatingChart (5, 4)
  chart.columns = 5 ∧ chart.rows = 4 := by
  sorry

end NUMINAMATH_CALUDE_seating_chart_interpretation_l1771_177187


namespace NUMINAMATH_CALUDE_fourteenSidedPolygonArea_l1771_177150

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a polygon defined by a list of vertices -/
structure Polygon where
  vertices : List Point

/-- Calculates the area of a polygon given its vertices -/
def calculatePolygonArea (p : Polygon) : ℝ := sorry

/-- The fourteen-sided polygon from the problem -/
def fourteenSidedPolygon : Polygon :=
  { vertices := [
      { x := 1, y := 2 }, { x := 2, y := 2 }, { x := 3, y := 3 }, { x := 3, y := 4 },
      { x := 4, y := 5 }, { x := 5, y := 5 }, { x := 6, y := 5 }, { x := 6, y := 4 },
      { x := 5, y := 3 }, { x := 4, y := 3 }, { x := 4, y := 2 }, { x := 3, y := 1 },
      { x := 2, y := 1 }, { x := 1, y := 1 }
    ]
  }

/-- Theorem stating that the area of the fourteen-sided polygon is 14 square centimeters -/
theorem fourteenSidedPolygonArea :
  calculatePolygonArea fourteenSidedPolygon = 14 := by sorry

end NUMINAMATH_CALUDE_fourteenSidedPolygonArea_l1771_177150


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1771_177147

theorem arithmetic_calculations :
  ((-10) + (-7) - 3 + 2 = -18) ∧
  ((-2)^3 / 4 - (-1)^2023 + |(-6)| * (-1) = -7) ∧
  ((1/3 - 1/4 + 5/6) * (-24) = -22) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1771_177147


namespace NUMINAMATH_CALUDE_writer_book_frequency_l1771_177189

theorem writer_book_frequency
  (years_writing : ℕ)
  (avg_earnings_per_book : ℝ)
  (total_earnings : ℝ)
  (h1 : years_writing = 20)
  (h2 : avg_earnings_per_book = 30000)
  (h3 : total_earnings = 3600000) :
  (years_writing * 12 : ℝ) / (total_earnings / avg_earnings_per_book) = 2 := by
  sorry

end NUMINAMATH_CALUDE_writer_book_frequency_l1771_177189


namespace NUMINAMATH_CALUDE_total_votes_is_82_l1771_177170

/-- Represents the number of votes for each cake type -/
structure CakeVotes where
  unicorn : ℕ
  witch : ℕ
  dragon : ℕ
  mermaid : ℕ
  fairy : ℕ

/-- Conditions for the baking contest votes -/
def contestConditions (votes : CakeVotes) : Prop :=
  votes.witch = 12 ∧
  votes.unicorn = 3 * votes.witch ∧
  votes.dragon = votes.witch + (2 * votes.witch / 5) ∧
  votes.mermaid = votes.dragon - 7 ∧
  votes.mermaid = 2 * votes.fairy ∧
  votes.fairy = votes.witch - 5

/-- Theorem stating that the total number of votes is 82 -/
theorem total_votes_is_82 (votes : CakeVotes) 
  (h : contestConditions votes) : 
  votes.unicorn + votes.witch + votes.dragon + votes.mermaid + votes.fairy = 82 := by
  sorry

end NUMINAMATH_CALUDE_total_votes_is_82_l1771_177170


namespace NUMINAMATH_CALUDE_sum_of_qp_values_l1771_177153

def p (x : ℝ) : ℝ := |x| - 3

def q (x : ℝ) : ℝ := -|x|

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_of_qp_values :
  (x_values.map (λ x => q (p x))).sum = -15 := by sorry

end NUMINAMATH_CALUDE_sum_of_qp_values_l1771_177153


namespace NUMINAMATH_CALUDE_ratio_problem_l1771_177178

theorem ratio_problem (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : 
  y / x = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1771_177178


namespace NUMINAMATH_CALUDE_room_width_calculation_l1771_177197

/-- Proves that given a rectangular room with length 5.5 m and a total paving cost of $16,500 at $800 per square meter, the width of the room is 3.75 m. -/
theorem room_width_calculation (length : Real) (cost_per_sqm : Real) (total_cost : Real) :
  length = 5.5 →
  cost_per_sqm = 800 →
  total_cost = 16500 →
  (total_cost / cost_per_sqm) / length = 3.75 := by
  sorry

#check room_width_calculation

end NUMINAMATH_CALUDE_room_width_calculation_l1771_177197


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1771_177137

theorem fraction_subtraction : 
  (2 + 4 + 6) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6) = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1771_177137


namespace NUMINAMATH_CALUDE_bugs_meeting_point_l1771_177173

/-- Triangle DEF with given side lengths -/
structure Triangle where
  DE : ℝ
  EF : ℝ
  FD : ℝ

/-- Two bugs crawling on the triangle's perimeter -/
structure Bugs where
  speed1 : ℝ
  speed2 : ℝ
  direction : Bool -- True if same direction, False if opposite

/-- Point G where the bugs meet -/
def meetingPoint (t : Triangle) (b : Bugs) : ℝ := sorry

/-- Theorem stating that EG = 2 under given conditions -/
theorem bugs_meeting_point (t : Triangle) (b : Bugs) : 
  t.DE = 8 ∧ t.EF = 10 ∧ t.FD = 12 ∧ 
  b.speed1 = 1 ∧ b.speed2 = 2 ∧ b.direction = false → 
  meetingPoint t b = 2 := by sorry

end NUMINAMATH_CALUDE_bugs_meeting_point_l1771_177173


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l1771_177110

/-- The number of flavors in the ice cream shop -/
def F : ℕ := sorry

/-- The number of flavors Gretchen tried two years ago -/
def tried_two_years_ago : ℚ := F / 4

/-- The number of flavors Gretchen tried last year -/
def tried_last_year : ℚ := 2 * tried_two_years_ago

/-- The number of flavors Gretchen still needs to try this year -/
def flavors_left : ℕ := 25

theorem ice_cream_flavors :
  F = 100 ∧
  tried_two_years_ago = F / 4 ∧
  tried_last_year = 2 * tried_two_years_ago ∧
  flavors_left = 25 ∧
  F = (tried_two_years_ago + tried_last_year + flavors_left) :=
sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l1771_177110


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1771_177103

/-- The quadratic function g(x) = x^2 + 2bx + 2b -/
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + 2*b*x + 2*b

/-- The theorem stating the condition for exactly one solution -/
theorem unique_solution_condition (b : ℝ) :
  (∃! x : ℝ, |g b x| ≤ 3) ↔ (b = 3 ∨ b = -1) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1771_177103


namespace NUMINAMATH_CALUDE_normal_curve_properties_l1771_177175

/- Normal distribution density function -/
noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

theorem normal_curve_properties (μ σ : ℝ) (h : σ > 0) :
  /- 1. Symmetry about x = μ -/
  (∀ x : ℝ, normal_pdf μ σ (μ + x) = normal_pdf μ σ (μ - x)) ∧
  /- 2. Always positive -/
  (∀ x : ℝ, normal_pdf μ σ x > 0) ∧
  /- 3. Global maximum at x = μ -/
  (∀ x : ℝ, normal_pdf μ σ x ≤ normal_pdf μ σ μ) ∧
  /- 4. Axis of symmetry is x = μ -/
  (∀ x : ℝ, normal_pdf μ σ (μ + x) = normal_pdf μ σ (μ - x)) ∧
  /- 5. σ determines the spread -/
  (∀ σ₁ σ₂ : ℝ, σ₁ ≠ σ₂ → ∃ x : ℝ, normal_pdf μ σ₁ x ≠ normal_pdf μ σ₂ x) ∧
  /- 6. Larger σ, flatter and wider curve -/
  (∀ σ₁ σ₂ : ℝ, σ₁ > σ₂ → ∃ x : ℝ, x ≠ μ ∧ normal_pdf μ σ₁ x > normal_pdf μ σ₂ x) ∧
  /- 7. Smaller σ, taller and slimmer curve -/
  (∀ σ₁ σ₂ : ℝ, σ₁ < σ₂ → normal_pdf μ σ₁ μ > normal_pdf μ σ₂ μ) :=
by sorry

end NUMINAMATH_CALUDE_normal_curve_properties_l1771_177175


namespace NUMINAMATH_CALUDE_min_sum_squares_l1771_177199

theorem min_sum_squares (x y z : ℝ) (h : x + 2*y + z = 1) :
  ∃ (m : ℝ), (∀ x' y' z' : ℝ, x' + 2*y' + z' = 1 → x'^2 + y'^2 + z'^2 ≥ m) ∧
             (∃ x₀ y₀ z₀ : ℝ, x₀ + 2*y₀ + z₀ = 1 ∧ x₀^2 + y₀^2 + z₀^2 = m) ∧
             m = 1/6 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1771_177199


namespace NUMINAMATH_CALUDE_evaluate_expression_l1771_177135

theorem evaluate_expression (x z : ℝ) (hx : x = 4) (hz : z = 0) : z * (z - 4 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1771_177135


namespace NUMINAMATH_CALUDE_distance_between_A_and_B_implies_a_eq_neg_one_l1771_177113

/-- Given two points A and B in 3D space, returns the square of the distance between them. -/
def distance_squared (A B : ℝ × ℝ × ℝ) : ℝ :=
  let (x₁, y₁, z₁) := A
  let (x₂, y₂, z₂) := B
  (x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2

theorem distance_between_A_and_B_implies_a_eq_neg_one :
  ∀ a : ℝ, 
  let A : ℝ × ℝ × ℝ := (-1, 1, -a)
  let B : ℝ × ℝ × ℝ := (-a, 3, -1)
  distance_squared A B = 4 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_A_and_B_implies_a_eq_neg_one_l1771_177113


namespace NUMINAMATH_CALUDE_pool_swimmers_l1771_177192

theorem pool_swimmers (total : ℕ) (first_day : ℕ) (second_day_diff : ℕ) :
  total = 246 →
  first_day = 79 →
  second_day_diff = 47 →
  ∃ (third_day : ℕ), 
    total = first_day + (third_day + second_day_diff) + third_day ∧
    third_day = 60 :=
by sorry

end NUMINAMATH_CALUDE_pool_swimmers_l1771_177192


namespace NUMINAMATH_CALUDE_ring_ratio_l1771_177161

def ring_problem (first_ring_cost second_ring_cost selling_price out_of_pocket : ℚ) : Prop :=
  first_ring_cost = 10000 ∧
  second_ring_cost = 2 * first_ring_cost ∧
  first_ring_cost + second_ring_cost - selling_price = out_of_pocket ∧
  out_of_pocket = 25000

theorem ring_ratio (first_ring_cost second_ring_cost selling_price out_of_pocket : ℚ) 
  (h : ring_problem first_ring_cost second_ring_cost selling_price out_of_pocket) :
  selling_price / first_ring_cost = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ring_ratio_l1771_177161


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1771_177124

/-- Proves that the complex fraction (3+8i)/(1-4i) simplifies to -29/17 + 20/17*i -/
theorem complex_fraction_simplification :
  (3 + 8 * Complex.I) / (1 - 4 * Complex.I) = -29/17 + 20/17 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1771_177124


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1771_177117

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 2| ≤ 5} = {x : ℝ | -7 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1771_177117


namespace NUMINAMATH_CALUDE_emerald_count_l1771_177127

/-- Represents a box of gemstones -/
structure GemBox where
  count : ℕ

/-- Represents the collection of gem boxes -/
structure GemCollection where
  diamonds : Array GemBox
  rubies : Array GemBox
  emeralds : Array GemBox

/-- The theorem to be proved -/
theorem emerald_count (collection : GemCollection) : 
  collection.diamonds.size = 2 ∧ 
  collection.rubies.size = 2 ∧ 
  collection.emeralds.size = 2 ∧ 
  (collection.rubies.foldl (λ acc box => acc + box.count) 0 = 
   collection.diamonds.foldl (λ acc box => acc + box.count) 0 + 15) →
  collection.emeralds.foldl (λ acc box => acc + box.count) 0 = 12 := by
  sorry

end NUMINAMATH_CALUDE_emerald_count_l1771_177127


namespace NUMINAMATH_CALUDE_tangent_from_unit_circle_point_l1771_177126

theorem tangent_from_unit_circle_point (α : Real) :
  (∃ (x y : Real), x^2 + y^2 = 1 ∧ x = -4/5 ∧ y = 3/5 ∧ 
   x = Real.cos α ∧ y = Real.sin α) →
  Real.tan α = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_from_unit_circle_point_l1771_177126


namespace NUMINAMATH_CALUDE_work_completion_time_l1771_177195

theorem work_completion_time (a b c : ℝ) : 
  (b = 12) →  -- B can do the work in 12 days
  (1/a + 1/b = 1/4) →  -- A and B working together finish the work in 4 days
  (a = 6) -- A can do the work alone in 6 days
:= by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1771_177195


namespace NUMINAMATH_CALUDE_eating_contest_l1771_177129

/-- Eating contest problem -/
theorem eating_contest (hot_dog_weight burger_weight pie_weight : ℕ)
  (mason_hotdog_multiplier : ℕ) (noah_burger_count : ℕ) (mason_hotdog_total_weight : ℕ)
  (h1 : hot_dog_weight = 2)
  (h2 : burger_weight = 5)
  (h3 : pie_weight = 10)
  (h4 : mason_hotdog_multiplier = 3)
  (h5 : noah_burger_count = 8)
  (h6 : mason_hotdog_total_weight = 30) :
  ∃ (jacob_pie_count : ℕ),
    jacob_pie_count = 5 ∧
    mason_hotdog_total_weight = jacob_pie_count * mason_hotdog_multiplier * hot_dog_weight :=
by
  sorry


end NUMINAMATH_CALUDE_eating_contest_l1771_177129


namespace NUMINAMATH_CALUDE_median_salary_is_25000_l1771_177115

/-- Represents a position in the company with its title, number of employees, and salary. -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- Calculates the median salary given a list of positions. -/
def medianSalary (positions : List Position) : Nat :=
  sorry

/-- The list of positions in the company. -/
def companyPositions : List Position := [
  { title := "President", count := 1, salary := 135000 },
  { title := "Vice-President", count := 4, salary := 92000 },
  { title := "Director", count := 15, salary := 78000 },
  { title := "Associate Director", count := 8, salary := 55000 },
  { title := "Administrative Specialist", count := 30, salary := 25000 },
  { title := "Customer Service Representative", count := 12, salary := 20000 }
]

/-- The total number of employees in the company. -/
def totalEmployees : Nat :=
  (companyPositions.map (·.count)).sum

theorem median_salary_is_25000 :
  totalEmployees = 70 ∧ medianSalary companyPositions = 25000 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_is_25000_l1771_177115


namespace NUMINAMATH_CALUDE_smallest_perimeter_is_364_l1771_177165

/-- Triangle with positive integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ+
  base : ℕ+

/-- Angle bisector intersection point -/
structure AngleBisectorIntersection (t : IsoscelesTriangle) where
  distance_to_vertex : ℕ+

/-- The smallest possible perimeter of an isosceles triangle with given angle bisector intersection -/
def smallest_perimeter (t : IsoscelesTriangle) (j : AngleBisectorIntersection t) : ℕ :=
  2 * (t.side.val + t.base.val)

/-- Theorem stating the smallest possible perimeter of the triangle -/
theorem smallest_perimeter_is_364 :
  ∃ (t : IsoscelesTriangle) (j : AngleBisectorIntersection t),
    j.distance_to_vertex = 10 ∧
    (∀ (t' : IsoscelesTriangle) (j' : AngleBisectorIntersection t'),
      j'.distance_to_vertex = 10 →
      smallest_perimeter t j ≤ smallest_perimeter t' j') ∧
    smallest_perimeter t j = 364 :=
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_is_364_l1771_177165


namespace NUMINAMATH_CALUDE_unique_xxyy_square_l1771_177134

def is_xxyy_form (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ n = 1000 * x + 100 * x + 10 * y + y

theorem unique_xxyy_square : 
  ∀ n : ℕ, is_xxyy_form n ∧ ∃ m : ℕ, n = m^2 → n = 7744 :=
sorry

end NUMINAMATH_CALUDE_unique_xxyy_square_l1771_177134


namespace NUMINAMATH_CALUDE_fibonacci_harmonic_sum_l1771_177171

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the harmonic series
def H : ℕ → ℚ
  | 0 => 0
  | (n + 1) => H n + 1 / (n + 1)

-- State the theorem
theorem fibonacci_harmonic_sum :
  (∑' n : ℕ, (fib (n + 1) : ℚ) / ((n + 2) * H (n + 1) * H (n + 2))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_harmonic_sum_l1771_177171


namespace NUMINAMATH_CALUDE_inequality_proof_l1771_177190

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (3*x^2 - x) / (1 + x^2) + (3*y^2 - y) / (1 + y^2) + (3*z^2 - z) / (1 + z^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1771_177190


namespace NUMINAMATH_CALUDE_divisors_in_range_l1771_177151

theorem divisors_in_range (m a b : ℕ) (hm : 0 < m) (ha : m^2 < a) (hb : m^2 < b) 
  (ha_upper : a < m^2 + m) (hb_upper : b < m^2 + m) (hab : a ≠ b) : 
  ∀ d : ℕ, m^2 < d → d < m^2 + m → d ∣ (a * b) → d = a ∨ d = b := by
sorry

end NUMINAMATH_CALUDE_divisors_in_range_l1771_177151


namespace NUMINAMATH_CALUDE_gcd_7429_13356_l1771_177125

theorem gcd_7429_13356 : Nat.gcd 7429 13356 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7429_13356_l1771_177125


namespace NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l1771_177183

theorem tangent_line_to_logarithmic_curve (k : ℝ) :
  (∃ x : ℝ, x > 0 ∧ k * x - 3 = 2 * Real.log x ∧
    (∀ y : ℝ, y > 0 → k * y - 3 ≥ 2 * Real.log y)) →
  k = 2 * Real.sqrt (Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l1771_177183


namespace NUMINAMATH_CALUDE_notebook_and_pen_cost_l1771_177140

/-- The cost of notebooks and neutral pens given certain conditions -/
theorem notebook_and_pen_cost
  (h1 : 4 * notebook_cost + 3 * pen_cost = 38)
  (h2 : notebook_cost + 6 * pen_cost = 20)
  (h3 : notebook_count + pen_count = 60)
  (h4 : notebook_count * notebook_cost + pen_count * pen_cost ≤ 330)
  : notebook_cost = 8 ∧ pen_cost = 2 ∧ pen_count ≥ 25 := by
  sorry

#check notebook_and_pen_cost

end NUMINAMATH_CALUDE_notebook_and_pen_cost_l1771_177140


namespace NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l1771_177114

/-- Converts a base 8 number to base 10 --/
def base8To10 (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

/-- Checks if a number is a 3-digit base 8 number --/
def is3DigitBase8 (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_divisible_by_7 :
  ∀ n : Nat, is3DigitBase8 n → base8To10 n % 7 = 0 → n ≤ 777 :=
sorry

end NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l1771_177114


namespace NUMINAMATH_CALUDE_particle_speed_is_sqrt_34_l1771_177179

/-- A particle moves along a path. Its position at time t is (3t + 1, 5t - 2). -/
def particle_position (t : ℝ) : ℝ × ℝ := (3 * t + 1, 5 * t - 2)

/-- The speed of the particle is defined as the distance traveled per unit time. -/
def particle_speed : ℝ := sorry

/-- Theorem: The speed of the particle is √34 units of distance per unit of time. -/
theorem particle_speed_is_sqrt_34 : particle_speed = Real.sqrt 34 := by sorry

end NUMINAMATH_CALUDE_particle_speed_is_sqrt_34_l1771_177179


namespace NUMINAMATH_CALUDE_initial_wall_count_l1771_177157

theorem initial_wall_count (total_containers ceiling_containers leftover_containers tiled_walls : ℕ) 
  (h1 : total_containers = 16)
  (h2 : ceiling_containers = 1)
  (h3 : leftover_containers = 3)
  (h4 : tiled_walls = 1)
  (h5 : ∀ w1 w2 : ℕ, w1 ≠ 0 → w2 ≠ 0 → (total_containers - ceiling_containers - leftover_containers) / w1 = 
                     (total_containers - ceiling_containers - leftover_containers) / w2 → w1 = w2) :
  total_containers - ceiling_containers - leftover_containers + tiled_walls = 13 := by
  sorry

end NUMINAMATH_CALUDE_initial_wall_count_l1771_177157


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l1771_177180

theorem ratio_sum_problem (w x y : ℝ) (hw_x : w / x = 1 / 6) (hw_y : w / y = 1 / 5) :
  (x + y) / y = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l1771_177180


namespace NUMINAMATH_CALUDE_dolphin_show_pictures_l1771_177169

/-- Represents the number of pictures Zoe took in different scenarios --/
structure PictureCount where
  before_dolphin_show : ℕ
  total : ℕ
  remaining_film : ℕ

/-- Calculates the number of pictures taken at the dolphin show --/
def pictures_at_dolphin_show (p : PictureCount) : ℕ :=
  p.total - p.before_dolphin_show

/-- Theorem stating that the number of pictures taken at the dolphin show
    is the difference between total pictures and pictures taken before --/
theorem dolphin_show_pictures (p : PictureCount)
  (h1 : p.before_dolphin_show = 28)
  (h2 : p.remaining_film = 32)
  (h3 : p.total = 44) :
  pictures_at_dolphin_show p = 16 := by
  sorry

end NUMINAMATH_CALUDE_dolphin_show_pictures_l1771_177169


namespace NUMINAMATH_CALUDE_correlation_significance_l1771_177162

/-- The critical value for a 5% significance level -/
def r_0_05 : ℝ := sorry

/-- The observed correlation coefficient -/
def r : ℝ := sorry

/-- An event with a probability of less than 5% -/
def low_probability_event : Prop := sorry

theorem correlation_significance :
  |r| > r_0_05 → low_probability_event := by sorry

end NUMINAMATH_CALUDE_correlation_significance_l1771_177162


namespace NUMINAMATH_CALUDE_train_crossing_time_l1771_177181

/-- Represents the speed of the train in km/hr -/
def train_speed : ℝ := 80

/-- Represents the length of the train in meters -/
def train_length : ℝ := 200

/-- Represents the time it takes for the train to cross the pole in seconds -/
def crossing_time : ℝ := 9

/-- Theorem stating that a train with the given speed and length takes 9 seconds to cross a pole -/
theorem train_crossing_time :
  (train_length / (train_speed * 1000 / 3600)) = crossing_time := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1771_177181


namespace NUMINAMATH_CALUDE_evaluate_expression_l1771_177142

theorem evaluate_expression : 8^6 * 27^6 * 8^27 * 27^8 = 2^99 * 3^42 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1771_177142


namespace NUMINAMATH_CALUDE_parabola_intersection_l1771_177194

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem parabola_intersection :
  (f (-1) = 0) →  -- The parabola intersects the x-axis at (-1, 0)
  (∃ x : ℝ, x ≠ -1 ∧ f x = 0 ∧ x = 3) :=  -- There exists another intersection point at (3, 0)
by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1771_177194


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1771_177107

/-- Given a hyperbola with equation x²/4 - y² = 1, prove its asymptotes and eccentricity -/
theorem hyperbola_properties (x y : ℝ) :
  x^2 / 4 - y^2 = 1 →
  (∃ (k : ℝ), k = 1/2 ∧ (y = k*x ∨ y = -k*x)) ∧
  (∃ (e : ℝ), e = Real.sqrt 5 / 2 ∧ e > 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1771_177107


namespace NUMINAMATH_CALUDE_marble_collection_total_l1771_177174

theorem marble_collection_total (b : ℝ) : 
  let r := 1.3 * b -- red marbles
  let g := 1.5 * b -- green marbles
  r + b + g = 3.8 * b := by sorry

end NUMINAMATH_CALUDE_marble_collection_total_l1771_177174


namespace NUMINAMATH_CALUDE_fools_gold_ounces_l1771_177100

def earnings_per_ounce : ℝ := 9
def fine : ℝ := 50
def remaining_money : ℝ := 22

theorem fools_gold_ounces :
  ∃ (x : ℝ), x * earnings_per_ounce - fine = remaining_money ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_fools_gold_ounces_l1771_177100


namespace NUMINAMATH_CALUDE_total_dogs_count_l1771_177138

/-- Represents the number of dogs in the Smartpup Training Center -/
structure DogCount where
  sit : ℕ
  stay : ℕ
  roll_over : ℕ
  jump : ℕ
  sit_stay : ℕ
  stay_roll : ℕ
  sit_roll : ℕ
  jump_stay : ℕ
  sit_stay_roll : ℕ
  no_tricks : ℕ

/-- Theorem stating that the total number of dogs is 150 given the specified conditions -/
theorem total_dogs_count (d : DogCount) 
  (h1 : d.sit = 60)
  (h2 : d.stay = 40)
  (h3 : d.roll_over = 45)
  (h4 : d.jump = 50)
  (h5 : d.sit_stay = 25)
  (h6 : d.stay_roll = 15)
  (h7 : d.sit_roll = 20)
  (h8 : d.jump_stay = 5)
  (h9 : d.sit_stay_roll = 10)
  (h10 : d.no_tricks = 5) : 
  d.sit + d.stay + d.roll_over + d.jump - d.sit_stay - d.stay_roll - d.sit_roll - 
  d.jump_stay + d.sit_stay_roll + d.no_tricks = 150 := by
  sorry


end NUMINAMATH_CALUDE_total_dogs_count_l1771_177138


namespace NUMINAMATH_CALUDE_initial_commission_rate_l1771_177139

theorem initial_commission_rate 
  (unchanged_income : ℝ → ℝ → ℝ → ℝ → Prop)
  (new_rate : ℝ)
  (slump_percentage : ℝ) :
  let initial_rate := 4
  let slump_factor := 1 - slump_percentage / 100
  unchanged_income initial_rate new_rate slump_factor 1 →
  new_rate = 5 →
  slump_percentage = 20.000000000000007 →
  initial_rate = 4 := by
sorry

end NUMINAMATH_CALUDE_initial_commission_rate_l1771_177139


namespace NUMINAMATH_CALUDE_simplify_expression_l1771_177177

theorem simplify_expression (x y : ℝ) : (35*x - 24*y) + (15*x + 40*y) - (25*x - 49*y) = 25*x + 65*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1771_177177


namespace NUMINAMATH_CALUDE_john_reps_per_set_l1771_177104

/-- Given the weight per rep, number of sets, and total weight moved,
    calculate the number of reps per set. -/
def reps_per_set (weight_per_rep : ℕ) (num_sets : ℕ) (total_weight : ℕ) : ℕ :=
  (total_weight / weight_per_rep) / num_sets

/-- Prove that under the given conditions, John does 10 reps per set. -/
theorem john_reps_per_set :
  let weight_per_rep : ℕ := 15
  let num_sets : ℕ := 3
  let total_weight : ℕ := 450
  reps_per_set weight_per_rep num_sets total_weight = 10 := by
sorry

end NUMINAMATH_CALUDE_john_reps_per_set_l1771_177104


namespace NUMINAMATH_CALUDE_point_on_line_l1771_177141

/-- Given two points A and B, and a third point C on the line AB, prove that the y-coordinate of C is 7. -/
theorem point_on_line (A B C : ℝ × ℝ) : 
  A = (1, -1) → 
  B = (3, 3) → 
  C.1 = 5 → 
  (C.2 - A.2) / (C.1 - A.1) = (B.2 - A.2) / (B.1 - A.1) → 
  C.2 = 7 := by
sorry

end NUMINAMATH_CALUDE_point_on_line_l1771_177141


namespace NUMINAMATH_CALUDE_exponent_fraction_simplification_l1771_177158

theorem exponent_fraction_simplification :
  (3^8 + 3^6) / (3^8 - 3^6) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_fraction_simplification_l1771_177158


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1771_177182

theorem quadratic_equation_roots (x : ℝ) :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ (x = r₁ ∨ x = r₂) ↔ x^2 - 2*x - 6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1771_177182


namespace NUMINAMATH_CALUDE_q_share_approx_l1771_177111

/-- Represents a partner in the partnership -/
inductive Partner
| P
| Q
| R

/-- Calculates the share ratio for a given partner -/
def shareRatio (partner : Partner) : Rat :=
  match partner with
  | Partner.P => 1/2
  | Partner.Q => 1/3
  | Partner.R => 1/4

/-- Calculates the investment duration for a given partner in months -/
def investmentDuration (partner : Partner) : ℕ :=
  match partner with
  | Partner.P => 2
  | _ => 12

/-- Calculates the capital ratio after p's withdrawal -/
def capitalRatioAfterWithdrawal (partner : Partner) : Rat :=
  match partner with
  | Partner.P => 1/4
  | _ => shareRatio partner

/-- The total profit in Rs -/
def totalProfit : ℕ := 378

/-- The total duration of the partnership in months -/
def totalDuration : ℕ := 12

/-- Calculates the investment parts for a given partner -/
def investmentParts (partner : Partner) : Rat :=
  (shareRatio partner * investmentDuration partner) +
  (capitalRatioAfterWithdrawal partner * (totalDuration - investmentDuration partner))

/-- Theorem stating that Q's share of the profit is approximately 123.36 Rs -/
theorem q_share_approx (ε : ℝ) (h : ε > 0) :
  ∃ (q_share : ℝ), abs (q_share - 123.36) < ε ∧
  q_share = (investmentParts Partner.Q / (investmentParts Partner.P + investmentParts Partner.Q + investmentParts Partner.R)) * totalProfit :=
sorry

end NUMINAMATH_CALUDE_q_share_approx_l1771_177111


namespace NUMINAMATH_CALUDE_hasans_plates_l1771_177168

/-- Proves the number of plates initially in Hasan's box -/
theorem hasans_plates
  (plate_weight : ℕ)
  (max_weight_oz : ℕ)
  (removed_plates : ℕ)
  (h1 : plate_weight = 10)
  (h2 : max_weight_oz = 20 * 16)
  (h3 : removed_plates = 6) :
  (max_weight_oz + removed_plates * plate_weight) / plate_weight = 38 := by
  sorry

end NUMINAMATH_CALUDE_hasans_plates_l1771_177168


namespace NUMINAMATH_CALUDE_charm_bracelet_profit_l1771_177131

/-- Calculates the profit from selling charm bracelets -/
theorem charm_bracelet_profit
  (string_cost : ℕ)
  (bead_cost : ℕ)
  (selling_price : ℕ)
  (bracelets_sold : ℕ)
  (h1 : string_cost = 1)
  (h2 : bead_cost = 3)
  (h3 : selling_price = 6)
  (h4 : bracelets_sold = 25) :
  (selling_price * bracelets_sold) - ((string_cost + bead_cost) * bracelets_sold) = 50 :=
by sorry

end NUMINAMATH_CALUDE_charm_bracelet_profit_l1771_177131


namespace NUMINAMATH_CALUDE_expression_value_l1771_177118

theorem expression_value (x y z : ℤ) (hx : x = 25) (hy : y = 30) (hz : z = 10) :
  (x - (y - z)) - ((x - y) - z) = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1771_177118


namespace NUMINAMATH_CALUDE_purely_imaginary_roots_l1771_177116

theorem purely_imaginary_roots (z : ℂ) (k : ℝ) : 
  (∀ r : ℂ, 20 * r^2 + 6 * Complex.I * r - k = 0 → ∃ b : ℝ, r = Complex.I * b) ↔ k = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_roots_l1771_177116


namespace NUMINAMATH_CALUDE_complex_power_difference_l1771_177105

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference : (2 + i)^12 - (2 - i)^12 = 503 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1771_177105


namespace NUMINAMATH_CALUDE_square_root_625_divided_by_5_l1771_177101

theorem square_root_625_divided_by_5 : Real.sqrt 625 / 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_625_divided_by_5_l1771_177101


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l1771_177154

theorem modular_congruence_solution (n : ℤ) : 
  0 ≤ n ∧ n < 103 ∧ (98 * n) % 103 = 33 % 103 → n % 103 = 87 % 103 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l1771_177154


namespace NUMINAMATH_CALUDE_intersection_locus_is_hyperbola_l1771_177108

/-- The locus of points (x, y) satisfying the given system of equations forms a hyperbola -/
theorem intersection_locus_is_hyperbola :
  ∀ (x y u : ℝ), 
  (2 * u * x - 3 * y - 4 * u = 0) →
  (x - 3 * u * y + 4 = 0) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_locus_is_hyperbola_l1771_177108


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_l1771_177148

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number is 465 -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_l1771_177148


namespace NUMINAMATH_CALUDE_total_octopus_legs_l1771_177196

/-- The number of legs an octopus has -/
def legs_per_octopus : ℕ := 8

/-- The number of octopuses Sawyer saw -/
def octopuses_seen : ℕ := 5

/-- Theorem: The total number of octopus legs Sawyer saw is 40 -/
theorem total_octopus_legs : octopuses_seen * legs_per_octopus = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_octopus_legs_l1771_177196


namespace NUMINAMATH_CALUDE_race_distance_l1771_177144

theorem race_distance (time_A time_B beat_distance : ℝ) 
  (h1 : time_A = 20)
  (h2 : time_B = 25)
  (h3 : beat_distance = 18) :
  ∃ D : ℝ, D = 72 ∧ D / time_A * time_B = D + beat_distance :=
by
  sorry

end NUMINAMATH_CALUDE_race_distance_l1771_177144


namespace NUMINAMATH_CALUDE_circle_radius_is_ten_l1771_177193

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 25 = 6*x + 12*y

/-- The radius of the circle -/
def circle_radius : ℝ := 10

theorem circle_radius_is_ten :
  ∃ (center_x center_y : ℝ),
    ∀ (x y : ℝ), circle_equation x y ↔ 
      (x - center_x)^2 + (y - center_y)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_ten_l1771_177193


namespace NUMINAMATH_CALUDE_factorization_condition_l1771_177146

theorem factorization_condition (a b c : ℤ) : 
  (∀ x : ℝ, (x - a) * (x - 10) + 1 = (x + b) * (x + c)) ↔ 
  ((a = 8 ∧ b = -9 ∧ c = -9) ∨ (a = 12 ∧ b = -11 ∧ c = -11)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_condition_l1771_177146


namespace NUMINAMATH_CALUDE_g_composition_of_2_l1771_177123

def g (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 5 * x + 2

theorem g_composition_of_2 : g (g (g (g 2))) = 112 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_2_l1771_177123


namespace NUMINAMATH_CALUDE_samson_utility_solution_l1771_177145

/-- Samson's utility function --/
def utility (math : ℝ) (frisbee : ℝ) : ℝ := math^2 * frisbee

/-- Monday's frisbee hours --/
def monday_frisbee (t : ℝ) : ℝ := t

/-- Monday's math hours --/
def monday_math (t : ℝ) : ℝ := 10 - 2*t

/-- Tuesday's frisbee hours --/
def tuesday_frisbee (t : ℝ) : ℝ := 3 - t

/-- Tuesday's math hours --/
def tuesday_math (t : ℝ) : ℝ := 2*t + 4

theorem samson_utility_solution :
  ∃ (t : ℝ), 
    t > 0 ∧ 
    t < 5 ∧
    utility (monday_math t) (monday_frisbee t) = utility (tuesday_math t) (tuesday_frisbee t) ∧
    t = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_samson_utility_solution_l1771_177145


namespace NUMINAMATH_CALUDE_integral_x_plus_cos_2x_over_symmetric_interval_l1771_177160

theorem integral_x_plus_cos_2x_over_symmetric_interval : 
  ∫ x in (-π/2)..(π/2), (x + Real.cos (2*x)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_plus_cos_2x_over_symmetric_interval_l1771_177160


namespace NUMINAMATH_CALUDE_food_consumption_reduction_l1771_177102

/-- Proves that given a 15% decrease in students and 20% increase in food price,
    the consumption reduction factor to maintain the same total cost is approximately 0.98039 -/
theorem food_consumption_reduction (N : ℝ) (P : ℝ) (h1 : N > 0) (h2 : P > 0) :
  let new_students := 0.85 * N
  let new_price := 1.2 * P
  let consumption_factor := (N * P) / (new_students * new_price)
  ∃ ε > 0, abs (consumption_factor - 0.98039) < ε :=
by sorry

end NUMINAMATH_CALUDE_food_consumption_reduction_l1771_177102


namespace NUMINAMATH_CALUDE_equation_solution_l1771_177109

theorem equation_solution : 
  ∃! x : ℚ, (53 - 3*x)^(1/4) + (39 + 3*x)^(1/4) = 5 :=
by
  -- The unique solution is x = -23/3
  use -23/3
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1771_177109


namespace NUMINAMATH_CALUDE_camp_III_sample_size_l1771_177164

def systematic_sample (total : ℕ) (sample_size : ℕ) (start : ℕ) (range_start : ℕ) (range_end : ℕ) : ℕ :=
  sorry

theorem camp_III_sample_size :
  systematic_sample 600 50 3 496 600 = 8 :=
sorry

end NUMINAMATH_CALUDE_camp_III_sample_size_l1771_177164


namespace NUMINAMATH_CALUDE_other_soap_bubble_ratio_l1771_177167

/- Define the number of bubbles Dawn can make per ounce -/
def dawn_bubbles_per_oz : ℕ := 200000

/- Define the number of bubbles made by half ounce of mixed soap -/
def mixed_bubbles_half_oz : ℕ := 150000

/- Define the ratio of bubbles made by the other soap to Dawn soap -/
def other_soap_ratio : ℚ := 1 / 2

/- Theorem statement -/
theorem other_soap_bubble_ratio :
  ∀ (other_bubbles_per_oz : ℕ),
    2 * mixed_bubbles_half_oz = dawn_bubbles_per_oz + other_bubbles_per_oz →
    other_bubbles_per_oz / dawn_bubbles_per_oz = other_soap_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_other_soap_bubble_ratio_l1771_177167


namespace NUMINAMATH_CALUDE_oil_bottles_total_volume_l1771_177163

theorem oil_bottles_total_volume (total_bottles : ℕ) (small_bottles : ℕ) 
  (small_volume : ℚ) (large_volume : ℚ) :
  total_bottles = 35 →
  small_bottles = 17 →
  small_volume = 250 / 1000 →
  large_volume = 300 / 1000 →
  (small_bottles * small_volume + (total_bottles - small_bottles) * large_volume) = 9.65 := by
sorry

end NUMINAMATH_CALUDE_oil_bottles_total_volume_l1771_177163


namespace NUMINAMATH_CALUDE_circle_a_range_l1771_177128

/-- A circle in the xy-plane is represented by the equation (x^2 + y^2 + 2x - 4y + a = 0) -/
def is_circle (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + a = 0

/-- The range of a for which the equation represents a circle -/
theorem circle_a_range :
  {a : ℝ | is_circle a} = Set.Iio 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_a_range_l1771_177128


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1771_177176

theorem cubic_equation_solution (A : ℕ) (a b s : ℤ) 
  (h_A : A = 1 ∨ A = 2 ∨ A = 3)
  (h_coprime : Int.gcd a b = 1)
  (h_eq : a^2 + A * b^2 = s^3) :
  ∃ u v : ℤ, 
    s = u^2 + A * v^2 ∧
    a = u^3 - 3 * A * u * v^2 ∧
    b = 3 * u^2 * v - A * v^3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1771_177176


namespace NUMINAMATH_CALUDE_exists_bisecting_line_l1771_177112

/-- A convex figure in the plane -/
structure ConvexFigure where
  -- We don't define the internal structure of ConvexFigure,
  -- as it's not necessary for the statement of the theorem

/-- A line in the plane -/
structure Line where
  -- We don't define the internal structure of Line,
  -- as it's not necessary for the statement of the theorem

/-- The perimeter of a convex figure -/
noncomputable def perimeter (F : ConvexFigure) : ℝ :=
  sorry

/-- The area of a convex figure -/
noncomputable def area (F : ConvexFigure) : ℝ :=
  sorry

/-- Predicate to check if a line bisects the perimeter of a convex figure -/
def bisects_perimeter (l : Line) (F : ConvexFigure) : Prop :=
  sorry

/-- Predicate to check if a line bisects the area of a convex figure -/
def bisects_area (l : Line) (F : ConvexFigure) : Prop :=
  sorry

/-- Theorem: For any convex figure in the plane, there exists a line that
    simultaneously bisects both its perimeter and area -/
theorem exists_bisecting_line (F : ConvexFigure) :
  ∃ l : Line, bisects_perimeter l F ∧ bisects_area l F :=
sorry

end NUMINAMATH_CALUDE_exists_bisecting_line_l1771_177112


namespace NUMINAMATH_CALUDE_linear_equation_m_value_l1771_177159

theorem linear_equation_m_value : 
  ∃! m : ℤ, (abs m - 4 = 1) ∧ (m - 5 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_m_value_l1771_177159


namespace NUMINAMATH_CALUDE_locus_of_P_is_ellipse_l1771_177155

-- Define the circle F₁
def circle_F₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 36

-- Define the fixed point F₂
def F₂ : ℝ × ℝ := (2, 0)

-- Define a point on the circle F₁
def point_on_F₁ (A : ℝ × ℝ) : Prop := circle_F₁ A.1 A.2

-- Define the center of F₁
def F₁ : ℝ × ℝ := (-2, 0)

-- Define the perpendicular bisector of F₂A
def perp_bisector (A : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  (P.1 - F₂.1) * (A.1 - F₂.1) + (P.2 - F₂.2) * (A.2 - F₂.2) = 0 ∧
  (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = (P.1 - A.1)^2 + (P.2 - A.2)^2

-- Define P as the intersection of perpendicular bisector and radius F₁A
def point_P (A : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  perp_bisector A P ∧
  ∃ (t : ℝ), P = (F₁.1 + t * (A.1 - F₁.1), F₁.2 + t * (A.2 - F₁.2))

-- Theorem: The locus of P is an ellipse with equation x²/9 + y²/5 = 1
theorem locus_of_P_is_ellipse :
  ∀ (P : ℝ × ℝ), (∃ (A : ℝ × ℝ), point_on_F₁ A ∧ point_P A P) ↔ 
  P.1^2 / 9 + P.2^2 / 5 = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_of_P_is_ellipse_l1771_177155


namespace NUMINAMATH_CALUDE_davids_pushups_l1771_177184

/-- Given that Zachary did 19 push-ups and David did 39 more push-ups than Zachary,
    prove that David did 58 push-ups. -/
theorem davids_pushups (zachary_pushups : ℕ) (david_extra_pushups : ℕ) 
    (h1 : zachary_pushups = 19)
    (h2 : david_extra_pushups = 39) : 
    zachary_pushups + david_extra_pushups = 58 := by
  sorry

end NUMINAMATH_CALUDE_davids_pushups_l1771_177184


namespace NUMINAMATH_CALUDE_local_minimum_implies_c_eq_two_l1771_177186

/-- The function f(x) -/
def f (c : ℝ) (x : ℝ) : ℝ := 2 * x * (x - c)^2 + 3

/-- Theorem: If f(x) has a local minimum at x = 2, then c = 2 -/
theorem local_minimum_implies_c_eq_two (c : ℝ) :
  (∃ δ > 0, ∀ x, |x - 2| < δ → f c x ≥ f c 2) →
  c = 2 :=
by sorry

end NUMINAMATH_CALUDE_local_minimum_implies_c_eq_two_l1771_177186


namespace NUMINAMATH_CALUDE_arrangements_together_count_arrangements_alternate_count_arrangements_restricted_count_l1771_177198

-- Define the number of boys and girls
def num_boys : ℕ := 2
def num_girls : ℕ := 3
def total_students : ℕ := num_boys + num_girls

-- Define the functions for each arrangement scenario
def arrangements_together : ℕ := sorry

def arrangements_alternate : ℕ := sorry

def arrangements_restricted : ℕ := sorry

-- State the theorems to be proved
theorem arrangements_together_count : arrangements_together = 24 := by sorry

theorem arrangements_alternate_count : arrangements_alternate = 12 := by sorry

theorem arrangements_restricted_count : arrangements_restricted = 60 := by sorry

end NUMINAMATH_CALUDE_arrangements_together_count_arrangements_alternate_count_arrangements_restricted_count_l1771_177198


namespace NUMINAMATH_CALUDE_egyptian_fraction_sum_l1771_177106

theorem egyptian_fraction_sum : ∃! (b₂ b₃ b₄ b₅ b₆ : ℕ),
  (11 : ℚ) / 13 = b₂ / 6 + b₃ / 24 + b₄ / 120 + b₅ / 720 + b₆ / 5040 ∧
  b₂ < 3 ∧ b₃ < 4 ∧ b₄ < 5 ∧ b₅ < 6 ∧ b₆ < 7 ∧
  b₂ + b₃ + b₄ + b₅ + b₆ = 1751 := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_sum_l1771_177106


namespace NUMINAMATH_CALUDE_prove_birds_and_storks_l1771_177149

def birds_and_storks_problem : Prop :=
  let initial_birds : ℕ := 3
  let initial_storks : ℕ := 4
  let birds_arrived : ℕ := 2
  let birds_left : ℕ := 1
  let storks_arrived : ℕ := 3
  let final_birds : ℕ := initial_birds + birds_arrived - birds_left
  let final_storks : ℕ := initial_storks + storks_arrived
  (final_birds : Int) - (final_storks : Int) = -3

theorem prove_birds_and_storks : birds_and_storks_problem := by
  sorry

end NUMINAMATH_CALUDE_prove_birds_and_storks_l1771_177149


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_964807_div_8_l1771_177166

theorem least_subtraction_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem least_subtraction_964807_div_8 :
  ∃ (k : Nat), k < 8 ∧ (964807 - k) % 8 = 0 ∧ ∀ (m : Nat), m < k → (964807 - m) % 8 ≠ 0 ∧ k = 7 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_least_subtraction_964807_div_8_l1771_177166


namespace NUMINAMATH_CALUDE_dima_puts_more_berries_l1771_177156

/-- Represents the berry-picking process of Dima and Sergey -/
structure BerryPicking where
  total_berries : ℕ
  dima_basket_rate : ℚ
  sergey_basket_rate : ℚ
  dima_speed : ℚ
  sergey_speed : ℚ

/-- Calculates the difference in berries put in the basket by Dima and Sergey -/
def berry_difference (bp : BerryPicking) : ℕ :=
  sorry

/-- Theorem stating the difference in berries put in the basket -/
theorem dima_puts_more_berries (bp : BerryPicking) 
  (h1 : bp.total_berries = 900)
  (h2 : bp.dima_basket_rate = 1/2)
  (h3 : bp.sergey_basket_rate = 2/3)
  (h4 : bp.dima_speed = 2 * bp.sergey_speed) :
  berry_difference bp = 100 :=
sorry

end NUMINAMATH_CALUDE_dima_puts_more_berries_l1771_177156


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l1771_177136

theorem roots_sum_and_product (a b : ℝ) : 
  (a^4 - 6*a^2 - 4*a + 1 = 0) → 
  (b^4 - 6*b^2 - 4*b + 1 = 0) → 
  (a ≠ b) →
  (∀ x : ℝ, x^4 - 6*x^2 - 4*x + 1 = 0 → x = a ∨ x = b) →
  a * b + a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l1771_177136


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1771_177132

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + (k + Complex.I)*x - 2 - k*Complex.I = 0) ↔ (k = 1 ∨ k = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1771_177132


namespace NUMINAMATH_CALUDE_jogging_duration_sum_l1771_177188

/-- The duration in minutes between 5 p.m. and 6 p.m. -/
def total_duration : ℕ := 60

/-- The probability of one friend arriving while the other is jogging -/
def meeting_probability : ℚ := 1/2

/-- Represents the duration each friend stays for jogging -/
structure JoggingDuration where
  x : ℕ
  y : ℕ
  z : ℕ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z
  z_not_perfect_square : ∀ (p : ℕ), Prime p → ¬(p^2 ∣ z)
  duration_eq : (x : ℚ) - y * Real.sqrt z = total_duration - total_duration * Real.sqrt 2

theorem jogging_duration_sum (d : JoggingDuration) : d.x + d.y + d.z = 92 := by
  sorry

end NUMINAMATH_CALUDE_jogging_duration_sum_l1771_177188


namespace NUMINAMATH_CALUDE_second_alignment_l1771_177185

/-- Represents the number of Heavenly Stems -/
def heavenly_stems : ℕ := 10

/-- Represents the number of Earthly Branches -/
def earthly_branches : ℕ := 12

/-- Represents the cycle length of the combined Heavenly Stems and Earthly Branches -/
def cycle_length : ℕ := lcm heavenly_stems earthly_branches

/-- 
Theorem: The second occurrence of the first Heavenly Stem aligning with 
the first Earthly Branch happens at column 61.
-/
theorem second_alignment : 
  cycle_length + 1 = 61 := by sorry

end NUMINAMATH_CALUDE_second_alignment_l1771_177185


namespace NUMINAMATH_CALUDE_files_deleted_l1771_177121

theorem files_deleted (initial_files : ℕ) (files_per_folder : ℕ) (num_folders : ℕ) : 
  initial_files = 27 → files_per_folder = 6 → num_folders = 3 →
  initial_files - (files_per_folder * num_folders) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_files_deleted_l1771_177121


namespace NUMINAMATH_CALUDE_mistaken_divisor_l1771_177122

theorem mistaken_divisor (dividend : ℕ) (correct_divisor mistaken_divisor : ℕ) :
  correct_divisor = 21 →
  dividend = correct_divisor * 20 →
  dividend = mistaken_divisor * 35 →
  mistaken_divisor = 12 := by
sorry

end NUMINAMATH_CALUDE_mistaken_divisor_l1771_177122


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l1771_177172

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → a * x^2 + 2 * a * x < 1 - 3 * a) ↔ a < 1/6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l1771_177172


namespace NUMINAMATH_CALUDE_smallest_appended_digits_for_divisibility_l1771_177152

theorem smallest_appended_digits_for_divisibility (n : ℕ) : 
  (∃ k : ℕ, 
    (n = 2014) ∧ 
    (∀ m : ℕ, m < 10 → (n * 10000 + k) % m = 0) ∧
    (∀ j : ℕ, j < 10000 → 
      (∃ m : ℕ, m < 10 ∧ (n * j + k) % m ≠ 0))) → 
  (∃ k : ℕ, k < 10000 ∧ 
    (∀ m : ℕ, m < 10 → (n * 10000 + k) % m = 0)) :=
sorry

end NUMINAMATH_CALUDE_smallest_appended_digits_for_divisibility_l1771_177152


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1771_177130

/-- Given a geometric sequence {aₙ}, prove that if a₃ = 16 and a₄ = 8, then a₁ = 64. -/
theorem geometric_sequence_first_term (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- Definition of geometric sequence
  a 3 = 16 →                                -- Condition: a₃ = 16
  a 4 = 8 →                                 -- Condition: a₄ = 8
  a 1 = 64 :=                               -- Conclusion: a₁ = 64
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1771_177130
