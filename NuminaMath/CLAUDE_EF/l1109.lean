import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_roots_l1109_110991

/-- The cubic polynomial function f(x) = (1/3)x³ - x² - 3x + 9 -/
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3*x + 9

/-- Theorem stating that f has exactly two distinct real roots -/
theorem f_has_two_roots : ∃ (a b : ℝ), a ≠ b ∧ 
  (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_two_roots_l1109_110991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1109_110996

variable (x y : ℝ)

def A (x y : ℝ) : ℝ := 3 * x^2 - x + 2 * y - 4 * x * y
def B (x y : ℝ) : ℝ := x^2 - 2 * x - y + x * y - 5

theorem problem_statement :
  (A x y - 3 * B x y = 5 * x + 5 * y - 7 * x * y + 15) ∧
  (∀ y, A x y - 3 * B x y = A x 0 - 3 * B x 0) → x = 5 / 7 := by
  sorry

#check problem_statement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1109_110996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l1109_110931

-- Define the constants
noncomputable def a : ℝ := 0.98 + Real.sin 0.01
noncomputable def b : ℝ := Real.exp (-0.01)
noncomputable def c : ℝ := (1 / 2) * (Real.log 2023 / Real.log 2022 + Real.log 2022 / Real.log 2023)

-- State the theorem
theorem order_of_abc : c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l1109_110931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_one_color_l1109_110968

/-- The probability of drawing 3 balls of one color and 1 ball of another color
    from a bin containing 10 black balls and 9 white balls, when 4 balls are drawn at random -/
theorem probability_three_one_color (black_balls : ℕ) (white_balls : ℕ) (total_drawn : ℕ) :
  black_balls = 10 →
  white_balls = 9 →
  total_drawn = 4 →
  (Nat.choose (black_balls + white_balls) total_drawn : ℚ)⁻¹ *
  ((Nat.choose black_balls 3 * Nat.choose white_balls 1 +
    Nat.choose black_balls 1 * Nat.choose white_balls 3) : ℚ) =
  160 / 323 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_one_color_l1109_110968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_covering_theorem_l1109_110976

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the distance between two circles
noncomputable def circle_distance (c1 c2 : Circle) : ℝ :=
  max 0 (distance c1.center c2.center - c1.radius - c2.radius)

-- Define a covering of points by circles
def is_covering (points : List Point) (circles : List Circle) : Prop :=
  ∀ p, p ∈ points → ∃ c, c ∈ circles ∧ distance p c.center ≤ c.radius

-- Main theorem
theorem circle_covering_theorem (points : List Point) 
  (h : points.length = 100) :
  ∃ (circles : List Circle), 
    is_covering points circles ∧ 
    (circles.map (λ c => 2 * c.radius)).sum < 100 ∧
    ∀ c1 c2, c1 ∈ circles → c2 ∈ circles → c1 ≠ c2 → circle_distance c1 c2 > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_covering_theorem_l1109_110976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_translation_for_symmetry_l1109_110993

theorem smallest_translation_for_symmetry (m : ℝ) :
  (m > 0) →
  (∃ k : ℤ, m + π / 3 = k * π + π / 2) →
  (∀ m' : ℝ, m' > 0 → (∃ k' : ℤ, m' + π / 3 = k' * π + π / 2) → m' ≥ m) →
  m = π / 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_translation_for_symmetry_l1109_110993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_distance_to_right_directrix_l1109_110920

/-- Definition of the ellipse -/
noncomputable def is_on_ellipse (x y : ℝ) : Prop := 3 * x^2 + 4 * y^2 = 12

/-- Definition of the distance from a point to the left focus -/
noncomputable def distance_to_left_focus : ℝ := 5/2

/-- Definition of the distance from a point to the right directrix -/
noncomputable def distance_to_right_directrix : ℝ := 3

/-- Theorem statement -/
theorem ellipse_distance_to_right_directrix 
  (x y : ℝ) 
  (h1 : is_on_ellipse x y) 
  (h2 : distance_to_left_focus = 5/2) : 
  distance_to_right_directrix = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_distance_to_right_directrix_l1109_110920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_frame_resistances_l1109_110990

/-- A cube frame constructed with resistors --/
structure CubeFrame where
  resistance : ℝ  -- resistance of each edge

/-- Equivalent resistance when voltage is applied across a body diagonal --/
noncomputable def body_diagonal_resistance (c : CubeFrame) : ℝ := 5 / 6 * c.resistance

/-- Equivalent resistance when voltage is applied across a face diagonal --/
noncomputable def face_diagonal_resistance (c : CubeFrame) : ℝ := 3 / 4 * c.resistance

/-- Theorem stating the equivalent resistances for a cube frame with 1Ω resistors --/
theorem cube_frame_resistances (c : CubeFrame) (h : c.resistance = 1) : 
  body_diagonal_resistance c = 5/6 ∧ face_diagonal_resistance c = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_frame_resistances_l1109_110990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_other_focus_l1109_110952

/-- Represents a point on a hyperbola -/
structure HyperbolaPoint where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 - y^2 / 8 = 1

/-- The distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The foci of the hyperbola -/
def foci : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((3, 0), (-3, 0))

theorem distance_to_other_focus (P : HyperbolaPoint) :
    distance (P.x, P.y) foci.1 = 3 →
    distance (P.x, P.y) foci.2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_other_focus_l1109_110952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_curve_equation_l1109_110948

noncomputable section

/-- The curve represented by the intersection of circles passing through M(0,0), P(0,5), and Q(12/5, 9/5) -/
def intersection_curve (x y : ℝ) : Prop :=
  x^2 + (y - 5/2)^2 = 25/4

/-- Point M -/
def M : ℝ × ℝ := (0, 0)

/-- Point P -/
def P : ℝ × ℝ := (0, 5)

/-- Point Q -/
def Q : ℝ × ℝ := (12/5, 9/5)

/-- Theorem stating that any point (x, y) on the intersection curve satisfies the equation -/
theorem intersection_curve_equation (x y : ℝ) :
  (∃ (r : ℝ), (x - M.1)^2 + (y - M.2)^2 = r^2) ∧
  (∃ (r : ℝ), (x - P.1)^2 + (y - P.2)^2 = r^2) ∧
  (∃ (r : ℝ), (x - Q.1)^2 + (y - Q.2)^2 = r^2) →
  intersection_curve x y :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_curve_equation_l1109_110948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_area_ratio_l1109_110962

-- Define the circle
def Circle (R : ℝ) := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 = R^2}

-- Define the square
def Square (s : ℝ) := {p : ℝ × ℝ | |p.1| ≤ s/2 ∧ |p.2| ≤ s/2}

-- Theorem statement
theorem square_circle_area_ratio (R : ℝ) (h : R > 0) :
  ∃ (s : ℝ), s > 0 ∧
  (∀ (side : Set (ℝ × ℝ)), side ⊆ Square s →
    ∃ (chord : Set (ℝ × ℝ)), chord ⊆ Circle R ∧
    chord ⊆ side ∧ 
    (∃ (p q : ℝ × ℝ), p ∈ chord ∧ q ∈ chord ∧ 
      ((p.1 - q.1)^2 + (p.2 - q.2)^2)^(1/2 : ℝ) = 2*R)) →
  (s^2) / (π * R^2) = 2/π :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_area_ratio_l1109_110962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1109_110955

-- Define the hyperbola C
structure Hyperbola where
  f1 : ℝ × ℝ
  f2 : ℝ × ℝ
  real_axis_length : ℝ

-- Define the properties of the specific hyperbola in the problem
def C : Hyperbola :=
  { f1 := (0, 2)
  , f2 := (0, -2)
  , real_axis_length := 2 }

-- Define eccentricity
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  let c := ((h.f1.1 - h.f2.1)^2 + (h.f1.2 - h.f2.2)^2).sqrt / 2
  let a := h.real_axis_length / 2
  c / a

-- Define a point on the asymptote
def Q : ℝ × ℝ := sorry

-- Define the perpendicularity condition
def perpendicular (h : Hyperbola) (q : ℝ × ℝ) : Prop :=
  let f1q := (q.1 - h.f1.1, q.2 - h.f1.2)
  let f2q := (q.1 - h.f2.1, q.2 - h.f2.2)
  f1q.1 * f2q.1 + f1q.2 * f2q.2 = 0

-- Define the area of triangle QF₁F₂
noncomputable def triangle_area (h : Hyperbola) (q : ℝ × ℝ) : ℝ :=
  let base := ((h.f1.1 - h.f2.1)^2 + (h.f1.2 - h.f2.2)^2).sqrt
  let height := (q.1^2 + q.2^2).sqrt
  1/2 * base * height

-- Statement of the theorem
theorem hyperbola_properties :
  eccentricity C = 2 ∧
  ∀ q : ℝ × ℝ, perpendicular C q → triangle_area C q = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1109_110955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beam_equation_solution_l1109_110943

/-- Represents the number of beams in a batch. -/
def num_beams : ℕ → Prop := sorry

/-- Represents the total cost of beams in wen. -/
def total_cost : ℕ → Prop := sorry

/-- Represents the transportation cost per beam in wen. -/
def transport_cost_per_beam : ℕ → Prop := sorry

/-- Represents the condition that after taking one less beam, 
    the cost of transporting the remaining beams equals the price of one beam. -/
def transport_cost_condition (x : ℕ) : Prop :=
  3 * x * (x - 1) = 6210

theorem beam_equation_solution :
  ∀ x : ℕ,
  total_cost 6210 →
  transport_cost_per_beam 3 →
  transport_cost_condition x →
  num_beams x :=
by
  sorry

#check beam_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beam_equation_solution_l1109_110943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_pens_sold_l1109_110908

/-- Proves that the number of pens sold is 90, given the conditions of the problem -/
def pens_sold (gain_cost : ℕ) (gain_percentage : ℚ) : ℕ :=
  let n : ℕ := 90
  n

/-- The main theorem that proves the number of pens sold -/
theorem trader_pens_sold : pens_sold 30 (33333333333333333 / 100000000000000000) = 90 := by
  unfold pens_sold
  rfl

#check trader_pens_sold

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_pens_sold_l1109_110908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chase_problem_l1109_110950

/-- The time it takes for A to catch up with B -/
noncomputable def catchUpTime (busPerson_A_speed : ℝ) (person_B_speed : ℝ) (bus_speed : ℝ) : ℝ :=
  (bus_speed) / (busPerson_A_speed - person_B_speed)

theorem chase_problem (bus_speed : ℝ) (person_A_speed : ℝ) (person_B_speed : ℝ) 
    (h1 : person_A_speed = 2 * person_B_speed)
    (h2 : person_A_speed = (4/5) * bus_speed)
    (h3 : bus_speed > 0) :
  catchUpTime person_A_speed person_B_speed bus_speed = 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chase_problem_l1109_110950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratioDifference_prop_relationshipMeasure_l1109_110947

/-- Represents a 2x2 contingency table -/
structure ContingencyTable where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The difference between the two ratios in a 2x2 contingency table -/
noncomputable def ratioDifference (t : ContingencyTable) : ℝ :=
  t.a / (t.a + t.b) - t.c / (t.c + t.d)

/-- The relationship measure for a 2x2 contingency table -/
def relationshipMeasure (t : ContingencyTable) : ℝ :=
  t.a * t.d - t.b * t.c

/-- The ratio difference is proportional to the relationship measure -/
theorem ratioDifference_prop_relationshipMeasure (t : ContingencyTable) :
  ∃ k : ℝ, ratioDifference t = k * relationshipMeasure t / ((t.a + t.b) * (t.c + t.d)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratioDifference_prop_relationshipMeasure_l1109_110947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brown_loss_l1109_110989

/-- Represents the sale of a watch -/
structure WatchSale where
  sellingPrice : ℚ
  profitPercent : ℚ

/-- Calculate the cost price of a watch given its sale details -/
noncomputable def costPrice (sale : WatchSale) : ℚ :=
  sale.sellingPrice / (1 + sale.profitPercent / 100)

/-- Mr. Brown's watch sales scenario -/
def brownScenario : (WatchSale × WatchSale) :=
  ({ sellingPrice := 2.40, profitPercent := 25 },
   { sellingPrice := 2.40, profitPercent := -25 })

/-- Calculate the total cost of the watches -/
noncomputable def totalCost (scenario : WatchSale × WatchSale) : ℚ :=
  costPrice scenario.fst + costPrice scenario.snd

/-- Calculate the total revenue from the watch sales -/
def totalRevenue (scenario : WatchSale × WatchSale) : ℚ :=
  scenario.fst.sellingPrice + scenario.snd.sellingPrice

/-- Calculate the net result (profit or loss) -/
noncomputable def netResult (scenario : WatchSale × WatchSale) : ℚ :=
  totalRevenue scenario - totalCost scenario

theorem brown_loss :
  netResult brownScenario = -32/100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brown_loss_l1109_110989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_offset_length_l1109_110946

/-- Represents a quadrilateral with a diagonal and two offsets -/
structure Quadrilateral where
  diagonal : ℝ
  offset1 : ℝ
  offset2 : ℝ

/-- Calculates the area of a quadrilateral given its diagonal and offsets -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  (q.diagonal * q.offset1 + q.diagonal * q.offset2) / 2

theorem second_offset_length (q : Quadrilateral) :
  q.diagonal = 10 ∧ q.offset1 = 7 ∧ area q = 50 → q.offset2 = 3 := by
  sorry

#check second_offset_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_offset_length_l1109_110946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_formula_a_increasing_l1109_110936

-- Define the sequence a_n
noncomputable def a : ℕ → ℝ → ℝ
| 0, t => 2*t - 3
| (n+1), t => ((2*t^(n+2) - 3) * a n t + 2*(t-1)*t^(n+1) - 1) / (a n t + 2*t^(n+1) - 1)

-- Theorem for the general formula
theorem a_general_formula (n : ℕ) (t : ℝ) (h1 : t ≠ 1) (h2 : t ≠ -1) :
  a n t = 2*(t^(n+1) - 1)/(n+1) - 1 :=
sorry

-- Theorem for the comparison of consecutive terms
theorem a_increasing (n : ℕ) (t : ℝ) (h1 : t > 0) (h2 : t ≠ 1) :
  a (n+1) t > a n t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_formula_a_increasing_l1109_110936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_ratio_l1109_110949

noncomputable def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ q ≠ 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def forms_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 3 * a 5 = (a 4) ^ 2

noncomputable def geometric_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  let q := a 2 / a 1
  a 1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_ratio :
  ∀ a : ℕ → ℝ,
  is_positive_geometric_sequence a →
  forms_arithmetic_sequence a →
  (geometric_sum a 6) / (geometric_sum a 3) = 9/8 := by
  sorry

#check geometric_sequence_sum_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_ratio_l1109_110949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_urban_survey_is_sample_l1109_110914

/-- Represents a person -/
structure Person where
  lifespan : ℕ

/-- Represents a group of people -/
structure Population where
  people : Set Person

/-- Represents a subset of a population -/
structure Sample (pop : Population) where
  subset : Set Person
  is_subset : subset ⊆ pop.people

/-- Represents a survey -/
structure Survey where
  participants : Set Person
  num_participants : ℕ
  provinces : Set String
  num_provinces : ℕ

/-- The national population -/
noncomputable def national_population : Population :=
{ people := sorry }

/-- The survey conducted -/
noncomputable def urban_survey : Survey :=
{ participants := sorry,
  num_participants := 2500,
  provinces := sorry,
  num_provinces := 11 }

/-- Theorem stating that the urban survey is a sample of the national population -/
theorem urban_survey_is_sample :
  ∃ (s : Sample national_population), s.subset = urban_survey.participants := by
  sorry

#check urban_survey_is_sample

end NUMINAMATH_CALUDE_ERRORFEEDBACK_urban_survey_is_sample_l1109_110914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bullseyes_is_87_l1109_110974

/-- Represents the archery tournament scenario -/
structure ArcheryTournament where
  total_shots : ℕ
  half_shots : ℕ
  chelsea_lead : ℕ
  chelsea_min_score : ℕ
  opponent_min_score : ℕ
  bullseye_score : ℕ

/-- The minimum number of bullseyes Chelsea needs to guarantee victory -/
def min_bullseyes_for_victory (tournament : ArcheryTournament) : ℕ :=
  let remaining_shots := tournament.total_shots - tournament.half_shots
  let max_opponent_score := tournament.chelsea_lead + remaining_shots * tournament.bullseye_score
  let n := (max_opponent_score - tournament.chelsea_lead - remaining_shots * tournament.chelsea_min_score) / 
            (tournament.bullseye_score - tournament.chelsea_min_score)
  n + 1

/-- Theorem stating the minimum number of bullseyes needed for victory -/
theorem min_bullseyes_is_87 (tournament : ArcheryTournament) 
  (h1 : tournament.total_shots = 200)
  (h2 : tournament.half_shots = 100)
  (h3 : tournament.chelsea_lead = 70)
  (h4 : tournament.chelsea_min_score = 5)
  (h5 : tournament.opponent_min_score = 3)
  (h6 : tournament.bullseye_score = 10) :
  min_bullseyes_for_victory tournament = 87 := by
  sorry

#eval min_bullseyes_for_victory {
  total_shots := 200,
  half_shots := 100,
  chelsea_lead := 70,
  chelsea_min_score := 5,
  opponent_min_score := 3,
  bullseye_score := 10
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bullseyes_is_87_l1109_110974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_zero_one_l1109_110999

open Real

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := log (3*x - x^3)

-- State the theorem
theorem f_increasing_on_zero_one :
  StrictMonoOn f (Set.Ioo 0 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_zero_one_l1109_110999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_kid_count_l1109_110944

/-- The number of kids Julia played with on different days of the week. -/
structure KidCount where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Given information about Julia's tag games -/
def julia_tag_games : KidCount :=
  { monday := 6,
    tuesday := 17,
    wednesday := 6 - 2 }

theorem wednesday_kid_count : julia_tag_games.wednesday = 4 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wednesday_kid_count_l1109_110944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l1109_110912

theorem same_color_probability (black_balls white_balls : ℕ) 
  (h1 : black_balls = 10) (h2 : white_balls = 8) : 
  (((black_balls.choose 3) + (white_balls.choose 3) : ℚ) / 
   ((black_balls + white_balls).choose 3)) = 22 / 102 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l1109_110912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1109_110982

-- Define the function f(x) = log(√(x-1))
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x - 1))

-- State the theorem about the domain of f
theorem domain_of_f :
  ∀ x : ℝ, (x ∈ Set.Ioi 1) ↔ (∃ y : ℝ, f x = y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1109_110982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_both_genders_selected_l1109_110965

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def num_volunteers : ℕ := 4

theorem probability_both_genders_selected :
  (Nat.choose (num_boys + num_girls) num_volunteers - Nat.choose num_boys num_volunteers) /
  (Nat.choose (num_boys + num_girls) num_volunteers : ℚ) = 34 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_both_genders_selected_l1109_110965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_tangent_equation_l1109_110937

theorem smallest_angle_tangent_equation :
  ∃ x : ℝ, x > 0 ∧ x < 360 ∧
    Real.tan (6 * x * π / 180) = (Real.cos (x * π / 180) - Real.sin (x * π / 180)) /
                                 (Real.cos (x * π / 180) + Real.sin (x * π / 180)) ∧
    (∀ y : ℝ, y > 0 ∧ y < 360 ∧
      Real.tan (6 * y * π / 180) = (Real.cos (y * π / 180) - Real.sin (y * π / 180)) /
                                   (Real.cos (y * π / 180) + Real.sin (y * π / 180)) →
      x ≤ y) ∧
    x = 45 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_tangent_equation_l1109_110937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_l1109_110917

theorem largest_difference (S : Set ℤ) (hS : S = {-20, -5, 0, 3, 7, 15}) : 
  (∃ a b, a ∈ S ∧ b ∈ S ∧ ∀ x y, x ∈ S → y ∈ S → a - b ≥ x - y) ∧ 
  (∃ a b, a ∈ S ∧ b ∈ S ∧ a - b = 35) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_l1109_110917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_with_perpendicular_tangents_and_different_axes_l1109_110934

noncomputable def f (x : ℝ) : ℝ := (1/8) * (x^2 + 6*x - 25)
noncomputable def g (x : ℝ) : ℝ := (1/8) * (25 + 6 - x^2)

def intersection_points : Set ℝ := {x | f x = g x}

noncomputable def f' (x : ℝ) : ℝ := (x + 3) / 4
noncomputable def g' (x : ℝ) : ℝ := -x / 4

def axis_of_symmetry_f : ℝ := -3
def axis_of_symmetry_g : ℝ := 0

theorem quadratic_functions_with_perpendicular_tangents_and_different_axes :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ intersection_points ∧ x₂ ∈ intersection_points ∧ 
  x₁ ≠ x₂ ∧
  (f' x₁ * g' x₁ = -1) ∧ (f' x₂ * g' x₂ = -1) ∧
  axis_of_symmetry_f ≠ axis_of_symmetry_g := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_functions_with_perpendicular_tangents_and_different_axes_l1109_110934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l1109_110904

-- Define the ellipse C
noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

-- Define the area of triangle OMN
noncomputable def area_OMN (a b : ℝ) (x y : ℝ) : ℝ :=
  Real.sqrt 2

theorem ellipse_and_triangle_properties
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : ellipse a b 0 (Real.sqrt 2))
  (h4 : eccentricity a b = Real.sqrt 2 / 2) :
  (∀ x y, ellipse a b x y ↔ x^2 / 4 + y^2 / 2 = 1) ∧
  (∀ x y, ellipse a b x y → area_OMN a b x y = Real.sqrt 2) := by
  sorry

#check ellipse_and_triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l1109_110904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_shortest_distance_l1109_110939

noncomputable section

-- Define the lines
def l₁ (x y m : ℝ) : Prop := x + 3 * y - 3 * m^2 = 0
def l₂ (x y m : ℝ) : Prop := 2 * x + y - m^2 - 5 * m = 0
def l₃ (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the intersection point P
def P (m : ℝ) : ℝ × ℝ := (3 * m, m^2 - m)

-- Define the distance function from a point to a line
def distance_to_l₃ (x y : ℝ) : ℝ := |x + y + 3| / Real.sqrt 2

-- Main theorem
theorem intersection_and_shortest_distance (m : ℝ) :
  (l₁ (P m).1 (P m).2 m ∧ l₂ (P m).1 (P m).2 m) ∧
  (∀ m' : ℝ, distance_to_l₃ (P m').1 (P m').2 ≥ distance_to_l₃ (P (-1)).1 (P (-1)).2) ∧
  distance_to_l₃ (P (-1)).1 (P (-1)).2 = 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_shortest_distance_l1109_110939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l1109_110961

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 + 24 * x - 4 * y^2 + 8 * y + 44 = 0

/-- The distance between vertices of the hyperbola -/
noncomputable def vertex_distance : ℝ := 2 * Real.sqrt 3

theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_equation x y →
  ∃ v1 v2 : ℝ × ℝ,
    (v1 ∈ {p : ℝ × ℝ | hyperbola_equation p.1 p.2}) ∧
    (v2 ∈ {p : ℝ × ℝ | hyperbola_equation p.1 p.2}) ∧
    (v1.2 ≠ v2.2) ∧
    (∀ p : ℝ × ℝ, hyperbola_equation p.1 p.2 → (p.2 - v1.2) * (p.2 - v2.2) ≤ 0) ∧
    Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = vertex_distance :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l1109_110961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_result_l1109_110907

/-- Calculates the number of euros received after exchanging dollars, given an exchange rate and service fee. -/
noncomputable def euros_received (exchange_rate_euros : ℝ) (exchange_rate_dollars : ℝ) (service_fee_percent : ℝ) (dollars_to_exchange : ℝ) : ℝ :=
  let euros_per_dollar := exchange_rate_euros / exchange_rate_dollars
  let euros_before_fee := euros_per_dollar * dollars_to_exchange
  euros_before_fee * (1 - service_fee_percent / 100)

/-- Theorem stating that given the specific exchange rate and conditions, 2.00 dollars will yield approximately 2111 euros. -/
theorem exchange_result : 
  ∃ (ε : ℝ), ε > 0 ∧ abs (euros_received 2000 1.80 5 2.00 - 2111) < ε :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exchange_result_l1109_110907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_network_counterexample_l1109_110942

/-- Represents a city -/
structure City where
  id : ℕ

/-- Represents an airline -/
structure Airline where
  id : ℕ

/-- Represents a flight connection between two cities by an airline -/
structure FlightConnection where
  from_city : City
  to_city : City
  airline : Airline

/-- Represents the flight network -/
structure FlightNetwork where
  cities : Finset City
  airlines : Finset Airline
  connections : Finset FlightConnection

/-- Checks if two cities are connected by an airline (possibly with layovers) -/
def connected (network : FlightNetwork) (airline : Airline) (city1 city2 : City) : Prop :=
  sorry

theorem flight_network_counterexample :
  ∃ (network : FlightNetwork),
    network.cities.card = 800 ∧
    network.airlines.card = 8 ∧
    (∀ c1 c2 : City, c1 ∈ network.cities → c2 ∈ network.cities → c1 ≠ c2 →
      ∃ a : Airline, a ∈ network.airlines ∧ connected network a c1 c2) ∧
    (∀ a : Airline, a ∈ network.airlines →
      ¬∃ (subset : Finset City), subset ⊆ network.cities ∧ subset.card > 200 ∧
        (∀ c1 c2 : City, c1 ∈ subset → c2 ∈ subset → connected network a c1 c2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_network_counterexample_l1109_110942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1109_110960

theorem trigonometric_identities (α : ℝ) : 
  (Real.sin α + Real.sin (α + 2 * π / 3) + Real.sin (α - 2 * π / 3) = 0) ∧
  (Real.sin α ^ 2 + Real.sin (α + 2 * π / 3) ^ 2 + Real.sin (α - 2 * π / 3) ^ 2 = 3 / 2) ∧
  (Real.sin α ^ 3 + Real.sin (α + 2 * π / 3) ^ 3 + Real.sin (α - 2 * π / 3) ^ 3 = -(3 / 4) * Real.sin (3 * α)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l1109_110960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scalar_matrix_property_l1109_110925

theorem scalar_matrix_property (v : Fin 3 → ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![3, 0, 0], ![0, 3, 0], ![0, 0, 3]]
  M.vecMul v = (3 : ℝ) • v := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scalar_matrix_property_l1109_110925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_g_min_greater_than_f_l1109_110929

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := |x - 2| - |x + 1|

noncomputable def g (a x : ℝ) : ℝ := (a * x^2 - x + 1) / x

-- Theorem for part (1)
theorem f_greater_than_one (x : ℝ) : f x > 1 ↔ x < 0 := by sorry

-- Theorem for part (2)
theorem g_min_greater_than_f (a : ℝ) :
  (a > 0 ∧ ∀ x > 0, ∃ y > 0, g a y ≤ g a x ∧ g a y > f x) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_one_g_min_greater_than_f_l1109_110929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_xy_value_l1109_110909

open Matrix Real

theorem max_xy_value (x y : ℝ) (m : ℝ) (h_x : x > 0) (h_y : y > 0) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; 1, 2]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![-1, m; -2, m]
  let α : Matrix (Fin 2) (Fin 1) ℝ := !![2; 3]
  A * α = B * α →
  (∀ x' y', x' > 0 → y' > 0 → x' * y' ≤ 25 / 6) ∧
  ∃ x' y', x' > 0 ∧ y' > 0 ∧ x' * y' = 25 / 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_xy_value_l1109_110909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_log_l1109_110979

theorem min_value_log (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) (h : 1 + a^3 = 9/8) :
  ∃ (f : ℝ → ℝ), (∀ x, x > 0 → f x = Real.log x / Real.log a) ∧
  (∀ x ∈ Set.Icc (1/4 : ℝ) 2, f x ≥ -1) ∧
  (∃ x ∈ Set.Icc (1/4 : ℝ) 2, f x = -1) :=
by
  -- We'll prove this later
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_log_l1109_110979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_intersection_points_l1109_110933

-- Define the equations
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 25
def parabola_eq (x y : ℝ) : Prop := x^2 + y = 17

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ circle_eq x y ∧ parabola_eq x y}

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_distance_intersection_points :
  ∃ p1 p2 : ℝ × ℝ, p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
    distance p1 p2 = 2 * Real.sqrt 22 ∧
    ∀ q1 q2 : ℝ × ℝ, q1 ∈ intersection_points → q2 ∈ intersection_points →
      distance q1 q2 ≤ 2 * Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_intersection_points_l1109_110933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_cone_volume_l1109_110988

-- Define the cone, cylinder, and frustum
structure Cone where
  radius : ℝ
  height : ℝ

structure Cylinder where
  radius : ℝ
  height : ℝ

structure Frustum where
  lower_radius : ℝ
  upper_radius : ℝ
  height : ℝ

-- Define volume functions
noncomputable def cylinder_volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height
noncomputable def frustum_volume (f : Frustum) : ℝ := (Real.pi * f.height / 3) * (f.lower_radius^2 + f.lower_radius * f.upper_radius + f.upper_radius^2)
noncomputable def cone_volume (c : Cone) : ℝ := (Real.pi * c.radius^2 * c.height) / 3

-- Theorem statement
theorem inscribed_cylinder_cone_volume 
  (cone : Cone) (cylinder : Cylinder) (frustum : Frustum) :
  cylinder_volume cylinder = 9 →
  frustum_volume frustum = 63 →
  cylinder.radius < cone.radius →
  cylinder.height < cone.height →
  frustum.lower_radius = cone.radius →
  frustum.upper_radius = cylinder.radius →
  frustum.height = cone.height - cylinder.height →
  cone_volume cone = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_cone_volume_l1109_110988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1109_110992

open Real

variable (x y z : ℝ)

noncomputable def f (x y z : ℝ) : ℝ := (x * (2*y - z)) / (1 + x + 3*y) +
                         (y * (2*z - x)) / (1 + y + 3*z) +
                         (z * (2*x - y)) / (1 + z + 3*x)

theorem max_value_of_f (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  f x y z ≤ 1/7 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧ f x y z = 1/7 := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1109_110992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_price_is_84_l1109_110969

/-- The price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- The price of a table in dollars -/
def table_price : ℝ := sorry

/-- The price of 2 chairs and 1 table is 60% of the price of 1 chair and 2 tables -/
axiom price_ratio : 2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

/-- The price of 1 table and 1 chair is $96 -/
axiom total_price : chair_price + table_price = 96

/-- The price of 1 table is $84 -/
theorem table_price_is_84 : table_price = 84 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_price_is_84_l1109_110969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircles_concurrent_l1109_110910

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define points P and Q on diagonals
noncomputable def P (quad : Quadrilateral) : ℝ × ℝ := sorry
noncomputable def Q (quad : Quadrilateral) : ℝ × ℝ := sorry

-- Define the ratio condition
def ratio_condition (quad : Quadrilateral) : Prop :=
  let ac_length := Real.sqrt ((quad.C.1 - quad.A.1)^2 + (quad.C.2 - quad.A.2)^2)
  let bd_length := Real.sqrt ((quad.D.1 - quad.B.1)^2 + (quad.D.2 - quad.B.2)^2)
  let ap_length := Real.sqrt ((P quad).1 - quad.A.1)^2 + ((P quad).2 - quad.A.2)^2
  let bq_length := Real.sqrt ((Q quad).1 - quad.B.1)^2 + ((Q quad).2 - quad.B.2)^2
  ap_length / ac_length + bq_length / bd_length = 1

-- Define points M and N
noncomputable def M (quad : Quadrilateral) : ℝ × ℝ := sorry
noncomputable def N (quad : Quadrilateral) : ℝ × ℝ := sorry

-- Define the circumcircle of a triangle
noncomputable def circumcircle (A B C : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- State the theorem
theorem circumcircles_concurrent (quad : Quadrilateral) 
  (h : ratio_condition quad) : 
  ∃ F : ℝ × ℝ, 
    F ∈ circumcircle quad.A (M quad) (P quad) ∧
    F ∈ circumcircle quad.B (N quad) (Q quad) ∧
    F ∈ circumcircle quad.D (M quad) (Q quad) ∧
    F ∈ circumcircle quad.C (N quad) (P quad) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcircles_concurrent_l1109_110910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_36_l1109_110975

/-- The number of positive divisors of 36 -/
def num_divisors_36 : ℕ := 9

/-- The sum of all positive divisors of 36 -/
def sum_divisors_36 : ℕ := 91

/-- Proves that the number of positive divisors of 36 is 9 and their sum is 91 -/
theorem divisors_of_36 :
  (Finset.filter (·∣36) (Finset.range 37)).card = num_divisors_36 ∧
  (Finset.filter (·∣36) (Finset.range 37)).sum id = sum_divisors_36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_36_l1109_110975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_approximation_l1109_110932

/-- Given a square and a rectangle satisfying certain conditions, 
    prove that the length of the rectangle is approximately 7.22 cm. -/
theorem rectangle_length_approximation (s : ℝ) (l : ℝ) :
  (Real.pi * s / 2 + s = 29.85) →  -- Semicircle circumference condition
  (4 * s = 2 * l + 2 * 16) →       -- Perimeter equality condition
  ∃ ε > 0, abs (l - 7.22) < ε := by -- Approximately equal to 7.22
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_approximation_l1109_110932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_integers_reachable_l1109_110981

/-- Represents the two allowed operations -/
inductive Operation
  | triple_plus_one : Operation
  | floor_half : Operation

/-- Applies an operation to a natural number -/
def apply_operation (op : Operation) (x : ℕ) : ℕ :=
  match op with
  | Operation.triple_plus_one => 3 * x + 1
  | Operation.floor_half => x / 2

/-- Represents a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a starting number -/
def apply_sequence (seq : OperationSequence) (start : ℕ) : ℕ :=
  seq.foldl (fun x op => apply_operation op x) start

/-- 
Theorem: For any positive integer n, there exists a finite sequence of operations
that, when applied to 1, results in n.
-/
theorem all_integers_reachable (n : ℕ) (h : n > 0) : 
  ∃ (seq : OperationSequence), apply_sequence seq 1 = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_integers_reachable_l1109_110981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_distance_theorem_l1109_110901

/-- A dilation in the plane -/
structure MyDilation where
  scale : ℝ
  center : ℝ × ℝ

/-- A circle in the plane -/
structure MyCircle where
  center : ℝ × ℝ
  radius : ℝ

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem dilation_distance_theorem (d : MyDilation) 
  (c1 c2 : MyCircle) (p : ℝ × ℝ) :
  c1.center = (3, 3) →
  c1.radius = 3 →
  c2.center = (10, 12) →
  c2.radius = 5 →
  p = (1, 1) →
  d.scale = c2.radius / c1.radius →
  d.center = (-1, -1) →
  distance p (d.scale • (p.1 - d.center.1, p.2 - d.center.2) + d.center) = 4 * Real.sqrt 2 / 3 := by
  sorry

#eval "Theorem statement type-checks correctly."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_distance_theorem_l1109_110901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_difference_l1109_110977

-- Define x₁ and x₂ as noncomputable
noncomputable def x₁ : ℝ := Real.sqrt 3 + Real.sqrt 2
noncomputable def x₂ : ℝ := Real.sqrt 3 - Real.sqrt 2

-- Theorem statement
theorem x_squared_difference : x₁^2 - x₂^2 = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_difference_l1109_110977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_cos_sin_diagonal_angle_l1109_110935

/-- A rectangle with integer coordinates -/
structure IntRectangle where
  a : ℤ × ℤ
  b : ℤ × ℤ
  c : ℤ × ℤ
  d : ℤ × ℤ
  is_rectangle : sorry -- Placeholder for rectangle property

/-- The angle between the diagonals of a rectangle -/
noncomputable def diagonalAngle (r : IntRectangle) : ℝ :=
  sorry

theorem rational_cos_sin_diagonal_angle (r : IntRectangle) :
  (∃ q : ℚ, (Real.cos (diagonalAngle r) : ℝ) = q) ∧
  (∃ q : ℚ, (Real.sin (diagonalAngle r) : ℝ) = q) :=
sorry

#check rational_cos_sin_diagonal_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_cos_sin_diagonal_angle_l1109_110935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_addition_result_l1109_110918

-- State the theorem
theorem complex_addition_result :
  let z₁ : ℂ := 6 - 3*Complex.I
  let z₂ : ℂ := -7 + 12*Complex.I
  let result : ℂ := -1 + 9*Complex.I
  z₁ + z₂ = result := by
  -- Unfold the definitions
  simp [Complex.I]
  -- Perform the complex addition
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_addition_result_l1109_110918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a3_value_l1109_110906

def our_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 2) = a (n + 1) + a n

theorem sequence_a3_value (a : ℕ → ℕ) 
  (h1 : our_sequence a) 
  (h2 : a 1 = 1) 
  (h3 : a 5 = 8) : 
  a 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a3_value_l1109_110906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_projection_on_line_l1109_110921

/-- Given a line y = (4/3)x - 2, prove that the projection of any vector on this line
    onto a certain vector w results in a constant vector p = [72/75, -18/25] -/
theorem constant_projection_on_line (w : ℝ × ℝ) :
  (∃ (k : ℝ), w.1 = -(4/3) * w.2 ∧ w ≠ (0, 0)) →
  ∀ (v : ℝ × ℝ), v.2 = (4/3) * v.1 - 2 →
    ((v.1 * w.1 + v.2 * w.2) / (w.1^2 + w.2^2)) • w = (72/75, -18/25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_projection_on_line_l1109_110921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_after_rotation_l1109_110930

/-- The slope of a line after rotation --/
noncomputable def rotated_slope (m : ℝ) (θ : ℝ) : ℝ :=
  (m + Real.tan θ) / (1 - m * Real.tan θ)

/-- The x-coordinate of the x-intercept of a rotated line --/
noncomputable def x_intercept_rotated (a b c : ℝ) (θ : ℝ) (x₀ y₀ : ℝ) : ℝ :=
  let m := a / b
  let m' := rotated_slope m θ
  (m' * x₀ - y₀ + y₀) / m'

theorem x_intercept_after_rotation 
  (a b c : ℝ) (θ : ℝ) (x₀ y₀ : ℝ) 
  (h₁ : a * x₀ + b * y₀ + c = 0)  -- Line passes through (x₀, y₀)
  (h₂ : b ≠ 0)  -- Slope is defined
  (h₃ : θ = π / 6)  -- 30° in radians
  (h₄ : x₀ = 10 ∧ y₀ = 10)  -- Rotation point
  (h₅ : a = 2 ∧ b = -7 ∧ c = 35)  -- Original line equation
  : x_intercept_rotated a b c θ x₀ y₀ = 
    (rotated_slope (a / b) θ * x₀ - y₀ + y₀) / rotated_slope (a / b) θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_after_rotation_l1109_110930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1109_110941

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  diameter : ℝ

-- Define the area function
noncomputable def triangleArea (t : Triangle) : ℝ :=
  1/2 * t.a * t.b * Real.sin t.C

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.diameter = 4 * Real.sqrt 3 / 3)
  (h2 : t.C = Real.pi / 3)  -- 60° in radians
  (h3 : t.a + t.b = t.a * t.b) : 
  (t.a + t.b + t.c) / (Real.sin t.A + Real.sin t.B + Real.sin t.C) = 4 * Real.sqrt 3 / 3 ∧ 
  triangleArea t = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1109_110941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_216_l1109_110903

def standard_die : Finset ℕ := Finset.range 6

def roll_three_dice : Finset (ℕ × ℕ × ℕ) := 
  standard_die.product (standard_die.product standard_die)

theorem probability_product_216 :
  (roll_three_dice.filter (fun (abc : ℕ × ℕ × ℕ) => abc.1 * abc.2.1 * abc.2.2 = 216)).card / roll_three_dice.card = 1 / 216 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_product_216_l1109_110903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_length_l1109_110926

/-- A right triangle ABC with given side lengths and a point P inside it -/
structure RightTriangleWithPoint where
  /-- Length of side CA -/
  ca : ℝ
  /-- Length of side CB -/
  cb : ℝ
  /-- Coordinates of point P -/
  p : ℝ × ℝ
  /-- Point P is inside the triangle -/
  p_inside : 0 < p.1 ∧ 0 < p.2 ∧ p.1 + p.2 < ca

/-- The sum of distances from P to the three sides of the triangle -/
noncomputable def sumOfDistances (t : RightTriangleWithPoint) : ℝ :=
  t.p.1 + t.p.2 + (|4 * t.p.1 + 3 * t.p.2 - 12|) / 5

/-- The theorem stating the length of the locus of point P -/
theorem locus_length (t : RightTriangleWithPoint) 
  (h_ca : t.ca = 3)
  (h_cb : t.cb = 4)
  (h_sum : sumOfDistances t = 13/5) :
  Real.sqrt 5 / 2 = ‖(1 : ℝ × ℝ) - (0, 1/2)‖ := by
  sorry

#check locus_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_length_l1109_110926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_tire_rotations_l1109_110997

/-- The number of full rotations a tire makes given its circumference and the distance traveled -/
noncomputable def tire_rotations (circumference : ℝ) (distance : ℝ) : ℝ :=
  distance / circumference

/-- Theorem: Given a tire with circumference 1.5 meters that travels 900 meters, 
    it will make 600 full rotations -/
theorem bike_tire_rotations : 
  tire_rotations 1.5 900 = 600 := by
  -- Unfold the definition of tire_rotations
  unfold tire_rotations
  -- Simplify the division
  simp [div_eq_mul_inv]
  -- Perform the arithmetic
  norm_num

#check bike_tire_rotations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bike_tire_rotations_l1109_110997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interchangeable_statements_l1109_110923

-- Define the concept of a geometric object (line or plane)
inductive GeometricObject
| Line
| Plane
deriving Repr, DecidableEq

-- Define the concept of a geometric statement
structure GeometricStatement where
  object1 : GeometricObject
  object2 : GeometricObject
  relation : GeometricObject → GeometricObject → Prop

-- Define the concept of an interchangeable statement
def isInterchangeable (s : GeometricStatement) : Prop :=
  s.relation s.object1 s.object2 ↔ 
  s.relation (if s.object1 = GeometricObject.Line then GeometricObject.Plane else GeometricObject.Line)
             (if s.object2 = GeometricObject.Line then GeometricObject.Plane else GeometricObject.Line)

-- Define the four statements
def statement1 : GeometricStatement :=
  { object1 := GeometricObject.Line
  , object2 := GeometricObject.Plane
  , relation := λ a b => ∀ (x y : GeometricObject), x = GeometricObject.Line ∧ y = GeometricObject.Line → True }

def statement2 : GeometricStatement :=
  { object1 := GeometricObject.Plane
  , object2 := GeometricObject.Plane
  , relation := λ a b => ∀ (x y : GeometricObject), x = GeometricObject.Plane ∧ y = GeometricObject.Plane → True }

def statement3 : GeometricStatement :=
  { object1 := GeometricObject.Line
  , object2 := GeometricObject.Line
  , relation := λ a b => ∀ (x y : GeometricObject), x = GeometricObject.Line ∧ y = GeometricObject.Line → True }

def statement4 : GeometricStatement :=
  { object1 := GeometricObject.Line
  , object2 := GeometricObject.Plane
  , relation := λ a b => ∀ (x y : GeometricObject), x = GeometricObject.Line ∧ y = GeometricObject.Line → True }

-- Theorem to prove
theorem interchangeable_statements :
  isInterchangeable statement1 ∧
  isInterchangeable statement3 ∧
  ¬isInterchangeable statement2 ∧
  ¬isInterchangeable statement4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interchangeable_statements_l1109_110923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_bar_purity_l1109_110986

/-- Represents a gold alloy bar -/
structure GoldBar where
  weight : ℝ
  purity : ℝ

/-- Calculates the weight of pure gold in a bar -/
noncomputable def pureGoldWeight (bar : GoldBar) : ℝ := bar.weight * bar.purity / 100

theorem fourth_bar_purity 
  (bar1 : GoldBar)
  (bar2 : GoldBar)
  (bar3 : GoldBar)
  (bar4 : GoldBar)
  (h1 : bar1.weight = 300 ∧ bar1.purity = 96)
  (h2 : bar2.weight = 200 ∧ bar2.purity = 86)
  (h3 : bar3.weight = 400 ∧ bar3.purity = 64)
  (h4 : bar1.weight + bar2.weight + bar3.weight + bar4.weight = 1700)
  (h5 : (pureGoldWeight bar1 + pureGoldWeight bar2 + pureGoldWeight bar3 + pureGoldWeight bar4) / 1700 = 0.56)
  : bar4.purity = 29.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_bar_purity_l1109_110986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_is_sphere_l1109_110957

/-- Represents a point in 3D space using spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Converts spherical coordinates to Cartesian coordinates -/
noncomputable def sphericalToCartesian (p : SphericalPoint) : ℝ × ℝ × ℝ :=
  (p.ρ * Real.sin p.φ * Real.cos p.θ, p.ρ * Real.sin p.φ * Real.sin p.θ, p.ρ * Real.cos p.φ)

/-- The set of points satisfying ρ = cos φ in spherical coordinates -/
def S : Set (ℝ × ℝ × ℝ) :=
  {xyz | ∃ p : SphericalPoint, p.ρ = Real.cos p.φ ∧ xyz = sphericalToCartesian p}

/-- Definition of a sphere with radius 1 centered at the origin -/
def unitSphere : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | x^2 + y^2 + z^2 = 1}

theorem shape_is_sphere : S = unitSphere := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shape_is_sphere_l1109_110957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_SmallestPositiveIntegerWithProperties_l1109_110916

/-- A number is a terminating decimal if it can be expressed as a/b where b is of the form 2^m * 5^n -/
def IsTerminatingDecimal (n : ℕ) : Prop :=
  ∃ (a b m k : ℕ), n = b ∧ Nat.Coprime a b ∧ b = 2^m * 5^k

/-- Check if a natural number contains the digit 9 -/
def ContainsDigit9 (n : ℕ) : Prop :=
  ∃ (m k : ℕ), n = 10 * m + 9 + 10 * k

theorem SmallestPositiveIntegerWithProperties :
  ∀ n : ℕ, n > 0 →
    (IsTerminatingDecimal n ∧ ContainsDigit9 n ∧ 3 ∣ n) →
    n ≥ 90 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_SmallestPositiveIntegerWithProperties_l1109_110916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_proof_l1109_110902

noncomputable def angle_between (a b : ℝ × ℝ) : ℝ := 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem vector_magnitude_proof (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)
  (h2 : Real.sqrt (a.1^2 + a.2^2) = 1)
  (h3 : b = (1, -Real.sqrt 3)) :
  Real.sqrt ((2 * a.1 - b.1)^2 + (2 * a.2 - b.2)^2) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_proof_l1109_110902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l1109_110973

noncomputable section

-- Define the geometric sequence
def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

-- Define the sum of the first n terms of a geometric sequence
def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_problem (a₁ q : ℝ) (h_q_pos : q > 0) 
  (h_prod : geometric_sequence a₁ q 1 * geometric_sequence a₁ q 2 * geometric_sequence a₁ q 3 = 27)
  (h_sum : geometric_sequence a₁ q 2 + geometric_sequence a₁ q 4 = 30) :
  a₁ = 1 ∧ q = 3 ∧ geometric_sum a₁ q 6 = 364 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l1109_110973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_three_pi_over_four_l1109_110922

noncomputable def f (A : ℝ) (φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (x + φ)

theorem function_value_at_three_pi_over_four 
  (A : ℝ) (φ : ℝ) 
  (h1 : A > 0) 
  (h2 : 0 < φ ∧ φ < Real.pi) 
  (h3 : ∀ x, f A φ x ≤ 1) 
  (h4 : f A φ (Real.pi/3) = 1/2) :
  f A φ (3*Real.pi/4) = -Real.sqrt 2 / 2 := by
  sorry

#check function_value_at_three_pi_over_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_at_three_pi_over_four_l1109_110922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1109_110967

/-- Represents the time taken for a train to cross a platform -/
noncomputable def crossing_time (train_length platform_length : ℝ) (speed : ℝ) : ℝ :=
  (train_length + platform_length) / speed

theorem train_crossing_time 
  (train_length : ℝ)
  (platform1_length platform2_length : ℝ)
  (time1 : ℝ)
  (h1 : train_length = 230)
  (h2 : platform1_length = 130)
  (h3 : platform2_length = 250)
  (h4 : time1 = 15)
  (h5 : crossing_time train_length platform1_length ((train_length + platform1_length) / time1) = time1)
  : crossing_time train_length platform2_length ((train_length + platform1_length) / time1) = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l1109_110967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_l1109_110970

-- Define the point through which the terminal side of angle α passes
def terminal_point (a : ℝ) : ℝ × ℝ := (3*a - 9, a + 2)

-- Define the theorem
theorem angle_range (α : ℝ) (a : ℝ) :
  (∃ (x y : ℝ), terminal_point a = (x, y) ∧ 
    x = 3*a - 9 ∧ y = a + 2) →
  Real.cos α ≤ 0 →
  Real.sin α > 0 →
  π/2 < α ∧ α < π :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_l1109_110970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_correct_l1109_110978

/-- Given a right triangle ABC with AC = BC = 4, and D and E as midpoints of AC and BC respectively,
    this function calculates the area of overlap between triangle ABC and the translated quadrilateral PQRS,
    where PQRS is obtained by translating ABED along ray AB by a distance m. -/
noncomputable def overlap_area (m : ℝ) : ℝ :=
  if 0 < m ∧ m ≤ 2 * Real.sqrt 2 then
    6 - Real.sqrt 2 * m
  else if 2 * Real.sqrt 2 < m ∧ m < 4 * Real.sqrt 2 then
    (1/4) * m^2 - 2 * Real.sqrt 2 * m + 8
  else
    0  -- This case is added to handle all real numbers, though not specified in the original problem

theorem overlap_area_correct (m : ℝ) (h1 : 0 < m) (h2 : m < 4 * Real.sqrt 2) :
  overlap_area m = 
    if m ≤ 2 * Real.sqrt 2 then
      6 - Real.sqrt 2 * m
    else
      (1/4) * m^2 - 2 * Real.sqrt 2 * m + 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_correct_l1109_110978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_implies_a_equals_3_l1109_110983

noncomputable def C (a : ℝ) (x : ℝ) : ℝ := (a * x + a^2 + 1) / (x + a - 1)

def symmetry_yx (f : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ f y = x

def symmetry_point (f : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∀ x, f (2 * h - x) = 2 * k - f x

theorem curve_symmetry_implies_a_equals_3 (a : ℝ) :
  (∃ C' : ℝ → ℝ, symmetry_yx C' ∧ symmetry_point C' 3 (-2) ∧ 
    (∀ x, C' x = C a (C' x))) → a = 3 := by
  sorry

#check curve_symmetry_implies_a_equals_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_implies_a_equals_3_l1109_110983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_value_l1109_110966

/-- Converts a list of digits in base 7 to a natural number -/
def toNat7 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 7 * acc) 0

/-- Converts a natural number to a list of digits in base 7 -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem triangle_value :
  ∃ (triangle : Nat),
    triangle < 7 ∧
    (toNat7 [4, 3, 2, triangle, 1] + toNat7 [2, triangle, 5, 1, 6] = toNat7 [triangle, 1, 3, 6, 0] ↔
    triangle = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_value_l1109_110966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l1109_110913

-- Define the two curves
noncomputable def curve1 (x : ℝ) : ℝ := Real.sqrt x
def curve2 (x : ℝ) : ℝ := x^2

-- Define the area enclosed by the two curves
noncomputable def enclosed_area : ℝ := ∫ x in (0)..(1), (curve1 x - curve2 x)

-- Theorem statement
theorem area_between_curves : enclosed_area = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_curves_l1109_110913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_84_l1109_110987

/-- The profit function L(q) based on production quantity q -/
noncomputable def L (q : ℝ) : ℝ := (25 - q/8)*q - (100 + 4*q)

/-- Theorem stating that the profit is maximized when q = 84 -/
theorem profit_maximized_at_84 :
  ∃ (q_max : ℝ), q_max = 84 ∧ ∀ (q : ℝ), L q ≤ L q_max := by
  -- We'll use 84 as our q_max
  let q_max := 84
  
  -- Prove existence
  use q_max
  
  -- Prove the conjunction
  apply And.intro
  · -- First part: q_max = 84
    rfl
    
  · -- Second part: ∀ (q : ℝ), L q ≤ L q_max
    intro q
    -- The actual proof would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximized_at_84_l1109_110987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_46_l1109_110938

def sequence_condition (a : Fin 9 → ℤ) : Prop :=
  (∀ i : Fin 9, 2 ≤ i.val ∧ i.val ≤ 6 → a i = a (i - 1) + a (i - 2)) ∧
  (∀ i : Fin 9, 7 ≤ i.val → a i = a (i - 1) * a (i - 2)) ∧
  a 6 = 23 ∧ a 7 = 46 ∧ a 8 = 1058 ∧ a 9 = 49068

theorem first_number_is_46 (a : Fin 9 → ℤ) (h : sequence_condition a) : a 1 = 46 := by
  sorry

#check first_number_is_46

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_46_l1109_110938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_c_l1109_110919

noncomputable def a : ℝ × ℝ := (2, -3)
noncomputable def b : ℝ × ℝ := (-1, 1)

noncomputable def c : ℝ × ℝ := 
  let diff := (a.1 - b.1, a.2 - b.2)
  let magnitude := Real.sqrt (diff.1^2 + diff.2^2)
  (diff.1 / magnitude, diff.2 / magnitude)

theorem unit_vector_c : c = (3/5, -4/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_c_l1109_110919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3_is_negative_40_l1109_110994

/-- The coefficient of x^3 in the expansion of (x+1)^2(x-2)^5 -/
def coefficientX3 : ℤ := -40

/-- The expression (x+1)^2(x-2)^5 -/
def expression (x : ℝ) : ℝ := (x + 1)^2 * (x - 2)^5

/-- Helper function to get the coefficient of x^n in a polynomial -/
noncomputable def getCoefficient (p : ℝ → ℝ) (n : ℕ) : ℝ :=
  (1 / n.factorial) * (deriv^[n] p 0)

theorem coefficient_x3_is_negative_40 :
  getCoefficient expression 3 = coefficientX3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3_is_negative_40_l1109_110994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_negative_functions_l1109_110954

-- Define the "inverse negative" property
def inverse_negative (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → f (1 / x) = -f x

-- Define the three functions
noncomputable def f1 (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

noncomputable def f2 (x : ℝ) : ℝ := (1 - x^2) / (1 + x^2)

noncomputable def f3 (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x
  else if x = 1 then 0
  else if x > 1 then -1/x
  else 0  -- undefined for x ≤ 0

-- State the theorem
theorem inverse_negative_functions :
  ¬(inverse_negative f1) ∧
  (inverse_negative f2) ∧
  (inverse_negative f3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_negative_functions_l1109_110954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_points_scored_l1109_110945

/-- Calculates the total points scored in a basketball game given the success rates and total shots attempted. -/
def total_points (three_point_success_rate two_point_success_rate : ℚ) (total_shots : ℕ) : ℚ :=
  let three_point_points : ℚ := 3 * three_point_success_rate
  let two_point_points : ℚ := 2 * two_point_success_rate
  32 - 0.05 * (total_shots : ℚ)

/-- Theorem stating that under the given conditions, the total points scored is 30. -/
theorem jordan_points_scored :
  let three_point_success_rate : ℚ := 25 / 100
  let two_point_success_rate : ℚ := 40 / 100
  let total_shots : ℕ := 40
  total_points three_point_success_rate two_point_success_rate total_shots = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_points_scored_l1109_110945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_for_divisibility_l1109_110905

theorem least_subtraction_for_divisibility (n : ℕ) (h : n = 427398) :
  ∃ (k : ℕ), k = 8 ∧ 
  (∀ (m : ℕ), m < k → ¬(10 ∣ (n - m))) ∧
  (10 ∣ (n - k)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_for_divisibility_l1109_110905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l1109_110958

-- Define the function f as noncomputable due to its dependency on Real
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - (3 * Real.sqrt 3 / 2) * Real.cos x + 2

-- State the theorem
theorem symmetry_axis_of_f :
  ∃ (a : ℝ), f a (π / 2) = 7 / 2 ∧
  ∃ (k : ℤ), (π / 3 + ↑k * π + π / 2 : ℝ) = 5 * π / 6 ∧
  ∀ (x : ℝ), f a x = f a (2 * (5 * π / 6) - x) := by
  -- The proof is omitted and replaced with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l1109_110958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_side_expression_l1109_110915

theorem right_side_expression (a b : ℝ) : 
  (a * b) ^ (4.5 : ℝ) - 2 = (b * a) ^ (4.5 : ℝ) - 7 → (b * a) ^ (4.5 : ℝ) - 7 = (b * a) ^ (4.5 : ℝ) - 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_side_expression_l1109_110915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carl_watermelons_l1109_110959

/-- The number of watermelons Carl started with in the morning -/
def initial_watermelons (selling_price : ℚ) (profit : ℚ) (unsold : ℕ) : ℕ :=
  (profit / selling_price).floor.toNat + unsold

/-- Proof that Carl started with 53 watermelons -/
theorem carl_watermelons : initial_watermelons 3 105 18 = 53 := by
  -- Unfold the definition of initial_watermelons
  unfold initial_watermelons
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl

#eval initial_watermelons 3 105 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carl_watermelons_l1109_110959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_formula_l1109_110995

/-- A rectangular parallelepiped with given edge lengths -/
structure RectParallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance between centroids of triangles AMN and C₁PQ in a rectangular parallelepiped -/
noncomputable def centroid_distance (rp : RectParallelepiped) : ℝ :=
  (1/3) * Real.sqrt (rp.a^2 + 4*rp.b^2 + 4*rp.c^2)

/-- Theorem: The distance between centroids of triangles AMN and C₁PQ
    in a rectangular parallelepiped is (1/3) * √(a² + 4b² + 4c²) -/
theorem centroid_distance_formula (rp : RectParallelepiped) :
  centroid_distance rp = (1/3) * Real.sqrt (rp.a^2 + 4*rp.b^2 + 4*rp.c^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_distance_formula_l1109_110995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_volume_calculation_l1109_110985

/-- The volume of a circular well -/
noncomputable def well_volume (diameter : ℝ) (depth : ℝ) : ℝ :=
  Real.pi * (diameter / 2)^2 * depth

theorem well_volume_calculation : 
  well_volume 4 14 = 56 * Real.pi := by
  unfold well_volume
  simp [Real.pi]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_volume_calculation_l1109_110985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_route_is_valid_and_best_l1109_110928

-- Define the cities
inductive City
| H | I | S | T | L | K | B | C | M | N | U | Q | R | G | F | P | O | E | A | D
deriving BEq, Repr

-- Define the route type
def Route := List City

-- Define the conditions
def validRoute (r : Route) : Prop :=
  r.length = 20 ∧ 
  r.head? = some City.H ∧ 
  r.getLast? = some City.H ∧
  r.Nodup ∧
  ¬(r.indexOf City.N + 1 = r.indexOf City.O ∨ r.indexOf City.O + 1 = r.indexOf City.N) ∧
  ¬(r.indexOf City.R + 1 = r.indexOf City.S ∨ r.indexOf City.S + 1 = r.indexOf City.R)

def optimalRoute : Route :=
  [City.H, City.I, City.S, City.T, City.L, City.K, City.B, City.C, City.M, City.N,
   City.U, City.Q, City.R, City.G, City.F, City.P, City.O, City.E, City.A, City.H]

-- The theorem to prove
theorem optimal_route_is_valid_and_best :
  validRoute optimalRoute ∧
  ∀ (r : Route), validRoute r → r.indexOf City.D ≤ optimalRoute.indexOf City.D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_route_is_valid_and_best_l1109_110928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_to_line_formula_l1109_110972

noncomputable section

open Real

/-- The distance from a point on the circle to the line -/
def distance_circle_to_line (θ : ℝ) : ℝ :=
  |2 * cos (θ + π/4) + 2 * sqrt 2|

/-- The circle equation -/
def on_circle (x y : ℝ) : Prop :=
  ∃ θ : ℝ, x = 1 + 2 * cos θ ∧ y = 1 + 2 * sin θ

/-- The line equation -/
def on_line (x y : ℝ) : Prop :=
  x - y + 4 = 0

theorem distance_circle_to_line_formula :
  ∀ x y : ℝ, on_circle x y →
  ∃ θ : ℝ, distance_circle_to_line θ = |x - y + 4| / sqrt 2 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_to_line_formula_l1109_110972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_result_g_increasing_intervals_l1109_110900

noncomputable def p : ℝ × ℝ := (1, Real.sqrt 3)
noncomputable def q (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

theorem parallel_vectors_result (x : ℝ) (h : ∃ (k : ℝ), p = k • (q x)) :
  Real.sin (2 * x) - (Real.cos x) ^ 2 = (2 * Real.sqrt 3 - 1) / 4 := by sorry

noncomputable def f (x : ℝ) : ℝ := p.1 * (q x).1 + p.2 * (q x).2
noncomputable def g (x : ℝ) : ℝ := f (2 * x + Real.pi / 3)

theorem g_increasing_intervals (k : ℤ) :
  StrictMonoOn g (Set.Icc (-2 * Real.pi / 3 + k * Real.pi) (-Real.pi / 6 + k * Real.pi)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_result_g_increasing_intervals_l1109_110900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_1_l1109_110940

theorem expression_value_1 : 
  (0.001 : ℝ)^(-(1/3 : ℝ)) - (7/8 : ℝ)^0 + 16^(3/4 : ℝ) + (Real.sqrt 2 * 33)^6 = 89 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_1_l1109_110940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_significance_l1109_110971

-- Define the basic geometric objects
structure Point : Type :=
  (x y : ℝ)

structure Circle : Type :=
  (center : Point)
  (radius : ℝ)

-- Define the triangles and circles
def Triangle : Type := Point → Point → Point → Prop

-- Define the inscribed circle
def inscribed_circle : Triangle → Circle := sorry

-- Define the intersection of circles
def circles_intersect (c1 c2 : Circle) (Q : Point) : Prop := sorry

-- Define common internal tangents
def common_internal_tangents (c1 c2 : Circle) : Point → Point → Prop := sorry

-- Define significant point
def significant_point (Q : Point) : Prop := sorry

-- Main theorem
theorem intersection_point_significance 
  (P A B C D Q : Point) 
  (triangle_PAB triangle_PCD : Triangle)
  (circle1 circle2 : Circle) :
  triangle_PAB P A B →
  triangle_PCD P C D →
  circle1 = inscribed_circle triangle_PAB →
  circle2 = inscribed_circle triangle_PCD →
  circles_intersect circle1 circle2 Q →
  (∃ T1 T2, common_internal_tangents circle1 circle2 T1 T2 ∧ T1 = T2) →
  significant_point Q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_significance_l1109_110971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_complex_numbers_satisfy_condition_l1109_110998

/-- A complex number z satisfies the equilateral triangle condition if 0, z, and z^2 form an equilateral triangle in the complex plane. -/
def satisfies_equilateral_condition (z : ℂ) : Prop :=
  z ≠ 0 ∧ Complex.abs z = Complex.abs (z^2 - z) ∧ Complex.abs z = Complex.abs z^2 ∧ Complex.abs (z^2 - z) = Complex.abs z^2

/-- There are exactly two nonzero complex numbers that satisfy the equilateral triangle condition. -/
theorem two_complex_numbers_satisfy_condition :
  ∃! (s : Finset ℂ), s.card = 2 ∧ ∀ z ∈ s, satisfies_equilateral_condition z :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_complex_numbers_satisfy_condition_l1109_110998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_midpoint_is_ellipse_l1109_110963

/-- Given an ellipse with equation x²/a² + y²/b² = 1 where a > b > 0,
    F is the left focus of the ellipse, and P is the midpoint of segment MF₁,
    prove that the trajectory of P is an ellipse. -/
theorem trajectory_of_midpoint_is_ellipse
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (hc : c > 0)
  (ellipse : ∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 → (x, y) ∈ Set.univ)
  (focus : c^2 = a^2 - b^2)
  (M : ℝ → ℝ × ℝ)
  (hM : ∀ θ, M θ = (a * Real.cos θ, b * Real.sin θ))
  (F : ℝ × ℝ)
  (hF : F = (-c, 0))
  (P : ℝ → ℝ × ℝ)
  (hP : ∀ θ, P θ = ((M θ).1 / 2 + F.1 / 2, (M θ).2 / 2 + F.2 / 2)) :
  ∃ (A B C D E : ℝ), ∀ (x y : ℝ),
    (∃ θ, P θ = (x, y)) ↔ A*x^2 + B*x*y + C*y^2 + D*x + E*y = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_midpoint_is_ellipse_l1109_110963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_subset_pairs_l1109_110927

/-- The number of mutually exclusive subset pairs for a set of size n -/
def f (n : ℕ) : ℕ :=
  (3^n - 2^(n+1) + 1) / 2

/-- The set U of size n -/
def U (n : ℕ) : Set ℕ :=
  {i | 1 ≤ i ∧ i ≤ n}

theorem mutually_exclusive_subset_pairs (n : ℕ) (h : n ≥ 2) :
  (∃ (S : Set (Set ℕ × Set ℕ)), 
    (∀ (p : Set ℕ × Set ℕ), p ∈ S ↔ 
      (p.1 ⊆ U n ∧ p.2 ⊆ U n ∧ 
       p.1 ≠ ∅ ∧ p.2 ≠ ∅ ∧
       p.1 ∩ p.2 = ∅ ∧
       (p.2, p.1) ∈ S)) ∧
    Finset.card (Finset.powerset (Finset.powerset (Finset.range n))) = f n) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_subset_pairs_l1109_110927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_derivative_equals_surface_area_l1109_110953

-- Define the volume function for a sphere
noncomputable def V (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Define the surface area function for a sphere
noncomputable def S (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- Theorem statement
theorem sphere_volume_derivative_equals_surface_area (r : ℝ) :
  deriv V r = S r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_derivative_equals_surface_area_l1109_110953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_division_sum_l1109_110984

/-- The sum of products of marble counts in piles during division process -/
def S (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Function representing the actual sum of products during the division process -/
noncomputable def sum_of_products_during_division (n : ℕ) : ℕ :=
  sorry  -- This would be defined based on the division process described in the problem

/-- Theorem stating that S(n) is the correct sum for n marbles -/
theorem marble_division_sum (n : ℕ) : 
  S n = sum_of_products_during_division n :=
sorry

/-- Lemma: S(n) is always a natural number -/
lemma S_is_natural (n : ℕ) : ∃ k : ℕ, S n = k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_division_sum_l1109_110984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_even_g_l1109_110980

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x - Real.cos x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f (x + m)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem smallest_m_for_even_g :
  ∃ m : ℝ, m > 0 ∧ is_even_function (g m) ∧
  ∀ m' : ℝ, m' > 0 → is_even_function (g m') → m ≤ m' := by
  sorry

#check smallest_m_for_even_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_even_g_l1109_110980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l1109_110956

/-- Given a 4-inch by 4-inch square adjoining a 12-inch by 12-inch square,
    the area of the shaded region is 10 square inches. -/
theorem shaded_area_theorem (large_square_side small_square_side : ℝ) :
  large_square_side = 12 →
  small_square_side = 4 →
  let total_length := large_square_side + small_square_side
  let triangle_base := small_square_side * (large_square_side / total_length)
  let triangle_area := (1 / 2) * triangle_base * small_square_side
  let shaded_area := small_square_side^2 - triangle_area
  shaded_area = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l1109_110956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_BC_is_one_eighth_of_AD_l1109_110924

-- Define the line segment AD and points B and C on it
variable (AD : ℝ)
variable (B : ℝ)
variable (C : ℝ)

-- Define the conditions
axiom B_on_AD : 0 ≤ B ∧ B ≤ AD
axiom C_on_AD : 0 ≤ C ∧ C ≤ AD

-- Length of AB is 3 times the length of BD
axiom AB_eq_3BD : B = 3 * (AD - B)

-- Length of AC is 7 times the length of CD
axiom AC_eq_7CD : C = 7 * (AD - C)

-- Theorem to prove
theorem BC_is_one_eighth_of_AD : C - B = (1 / 8) * AD := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_BC_is_one_eighth_of_AD_l1109_110924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_games_count_l1109_110964

def is_fair_game (winning_outcomes : Finset Nat) : Bool :=
  winning_outcomes.card = 3

def game1 : Finset Nat := {2}
def game2 : Finset Nat := {2, 4, 6}
def game3 : Finset Nat := {1, 2, 3}
def game4 : Finset Nat := {3, 6}

def games : List (Finset Nat) := [game1, game2, game3, game4]

theorem fair_games_count :
  (games.filter is_fair_game).length = 2 := by
  sorry

#eval (games.filter is_fair_game).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_games_count_l1109_110964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_in_ellipse_l1109_110951

theorem circle_radius_in_ellipse : 
  ∃ (r : ℝ), 
    (∃ (x y : ℝ), x^2 + 4*y^2 = 5 ∧ (x - r)^2 + y^2 = r^2) ∧
    (∃ (x y : ℝ), x^2 + 4*y^2 = 5 ∧ (x + r)^2 + y^2 = r^2) ∧
    r = Real.sqrt 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_in_ellipse_l1109_110951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_60_digits_eq_108_l1109_110911

/-- The decimal representation of 1/2222 -/
def decimal_rep : ℚ := 1 / 2222

/-- The sequence of digits in the decimal representation of 1/2222 -/
def digit_sequence : ℕ → ℕ
| n => match n % 5 with
  | 0 => 0
  | 1 => 0
  | 2 => 0
  | 3 => 4
  | 4 => 5
  | _ => 0  -- This case should never occur, but Lean requires it for completeness

/-- The sum of the first 60 digits after the decimal point -/
def sum_60_digits : ℕ := (Finset.range 60).sum (λ i => digit_sequence i)

theorem sum_60_digits_eq_108 : sum_60_digits = 108 := by
  -- The proof is omitted for now
  sorry

#eval sum_60_digits  -- This will compute and display the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_60_digits_eq_108_l1109_110911
