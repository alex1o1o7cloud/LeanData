import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mult_card_eq_zero_l544_54412

/-- A commutative group G with cardinality k -/
class CommutativeGroupWithCardinality (G : Type*) [AddCommGroup G] [Fintype G] where
  card : ℕ
  card_eq : Fintype.card G = card

/-- For any commutative group G with cardinality k, kx = 0 for all x in G -/
theorem mult_card_eq_zero {G : Type*} [AddCommGroup G] [Fintype G] 
  [cgwc : CommutativeGroupWithCardinality G] (x : G) : 
  (cgwc.card : ℕ) • x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mult_card_eq_zero_l544_54412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_trig_function_l544_54453

theorem min_value_of_trig_function :
  ∀ x : ℝ, Real.sin x ^ 4 + 2 * Real.sin x * Real.cos x + Real.cos x ^ 4 ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_trig_function_l544_54453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ceiling_floor_log_l544_54460

open Real BigOperators

/-- The sum from k=1 to 500 of k(⌈log_√3(k)⌉ - ⌊log_√3(k)⌋) is equal to 125237 -/
theorem sum_ceiling_floor_log : (∑ k in Finset.range 500, (k + 1) * (⌈log (k + 1) / log (Real.sqrt 3)⌉ - ⌊log (k + 1) / log (Real.sqrt 3)⌋)) = 125237 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ceiling_floor_log_l544_54460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l544_54449

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else Real.exp (x + 1) - 2

-- State the theorem
theorem f_composition_value : f (f (1 / Real.exp 1)) = -1 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l544_54449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_equation_l544_54426

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 16 * y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 4)

-- Define the directrix of the parabola
def directrix (y : ℝ) : Prop := y = -4

-- Define a custom circle (to avoid conflict with built-in circle)
def custom_circle (center_x center_y r x y : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = r^2

-- Theorem statement
theorem parabola_circle_equation :
  ∀ (x y r : ℝ),
  (parabola x y) →
  (custom_circle focus.1 focus.2 r x y) →
  (∃ (y_dir : ℝ), directrix y_dir ∧ r = focus.2 - y_dir) →
  (x^2 + (y - 4)^2 = 64) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_equation_l544_54426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_range_of_f_l544_54463

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 / (x - 2)

-- Define the domain
def domain : Set ℝ := { x | 3 ≤ x ∧ x ≤ 6 }

-- Theorem statement
theorem value_range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | 1 ≤ y ∧ y ≤ 4 } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_range_of_f_l544_54463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_unfolding_not_right_triangle_l544_54475

-- Define a point in 3D space
structure Point := (x y z : ℝ)

-- Define a tetrahedron
structure Tetrahedron := (A B C D : Point)

-- Define a triangle
structure Triangle := (A B C : Point)

-- Define the unfolding operation
noncomputable def unfold (t : Tetrahedron) : Triangle :=
  sorry -- Implementation details omitted for brevity

-- Define what it means for a triangle to be a right triangle
def IsRightTriangle (t : Triangle) : Prop :=
  sorry -- Definition omitted for brevity

-- Theorem statement
theorem tetrahedron_unfolding_not_right_triangle (t : Tetrahedron) :
  ¬ (IsRightTriangle (unfold t)) :=
by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_unfolding_not_right_triangle_l544_54475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_program_arrangement_count_l544_54445

/-- The number of ways to arrange k objects from n distinct objects. -/
def arrange (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The number of ways to arrange 5 solo and 3 chorus programs with given constraints. -/
def arrangePrograms : ℕ :=
  arrange 5 5 * arrange 5 3

theorem program_arrangement_count :
  arrangePrograms = arrange 5 5 * arrange 5 3 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_program_arrangement_count_l544_54445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l544_54454

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (x + π / 4)

theorem max_value_of_expression (A B C : ℝ) : 
  A + B + C = π → -- Triangle angle sum
  f B = sqrt 3 → -- Given condition
  B > 0 → B < π → -- B is an interior angle
  A > 0 → A < π → -- A is an interior angle
  C > 0 → C < π → -- C is an interior angle
  ∃ (m : ℝ), m = 1 ∧ ∀ (A' C' : ℝ), 
    A' + B + C' = π → 
    A' > 0 → A' < π → 
    C' > 0 → C' < π → 
    sqrt 2 * cos A' + cos C' ≤ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l544_54454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_time_example_l544_54455

/-- The time taken for a train to pass through a tunnel -/
noncomputable def train_tunnel_time (train_length : ℝ) (train_speed_kmh : ℝ) (tunnel_length_km : ℝ) : ℝ :=
  let train_speed_mpm := train_speed_kmh * 1000 / 60
  let total_distance := tunnel_length_km * 1000 + train_length
  total_distance / train_speed_mpm

/-- Theorem: A train of length 100 meters traveling at 72 km/hr through a tunnel of length 1.7 km
    takes 1.5 minutes to completely pass through the tunnel -/
theorem train_tunnel_time_example : train_tunnel_time 100 72 1.7 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_tunnel_time_example_l544_54455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_greater_than_18_l544_54433

/-- A set of scores satisfying the olympiad conditions -/
def OlympiadScores : Type := { scores : Finset ℕ // scores.card = 20 }

/-- The condition that all scores are distinct -/
def all_distinct (scores : OlympiadScores) : Prop :=
  ∀ x y, x ∈ scores.val → y ∈ scores.val → x ≠ y → scores.val.toList.indexOf x ≠ scores.val.toList.indexOf y

/-- The condition that each score is less than the sum of any two other scores -/
def sum_condition (scores : OlympiadScores) : Prop :=
  ∀ x y z, x ∈ scores.val → y ∈ scores.val → z ∈ scores.val → 
    x ≠ y ∧ x ≠ z ∧ y ≠ z → x < y + z

/-- The theorem stating that the minimum score is greater than 18 -/
theorem min_score_greater_than_18 (scores : OlympiadScores) 
  (h1 : all_distinct scores) (h2 : sum_condition scores) : 
  ∀ x, x ∈ scores.val → x > 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_score_greater_than_18_l544_54433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_minus_15_3_value_f_monotonicity_l544_54491

-- Define A_x^m
def A (x : ℝ) (m : ℕ) : ℝ :=
  if m = 0 then 1
  else (List.range m).foldl (λ acc i => acc * (x - i)) 1

-- Define f(x) = A_x^3
def f (x : ℝ) : ℝ := A x 3

-- Theorem for part 1
theorem A_minus_15_3_value : A (-15) 3 = -4080 := by sorry

-- Theorem for part 2
theorem f_monotonicity :
  (∀ x y, x < y ∧ y < (3 - Real.sqrt 3) / 3 → f x < f y) ∧
  (∀ x y, (3 - Real.sqrt 3) / 3 < x ∧ x < y ∧ y < (3 + Real.sqrt 3) / 3 → f x > f y) ∧
  (∀ x y, (3 + Real.sqrt 3) / 3 < x ∧ x < y → f x < f y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_minus_15_3_value_f_monotonicity_l544_54491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_of_divisors_plus_sqrt_l544_54492

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_divisor (d n : ℕ) : Bool := n % d = 0

def divisors (n : ℕ) : List ℕ := (List.range n).filter (λ d => is_divisor d n)

theorem sum_of_reciprocals_of_divisors_plus_sqrt (n : ℕ) :
  let d := divisors (factorial 10)
  (d.map (λ x => (1 : ℝ) / (x + Real.sqrt (factorial 10)))).sum = 3 / (16 * Real.sqrt 7) := by
  sorry

#eval factorial 10
#eval divisors (factorial 10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_of_divisors_plus_sqrt_l544_54492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l544_54466

theorem election_votes (winning_percentage : ℚ) (majority : ℕ) (total_votes : ℕ) : 
  winning_percentage = 3/5 →
  majority = 900 →
  (winning_percentage * total_votes - (1 - winning_percentage) * total_votes) = majority →
  total_votes = 4500 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l544_54466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_reciprocal_quadratic_l544_54405

theorem integral_of_reciprocal_quadratic (x : ℝ) :
  deriv (λ y => (1/2) * Real.arctan ((y + 2) / 2)) x = 1 / (x^2 + 4*x + 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_of_reciprocal_quadratic_l544_54405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_problem_l544_54469

theorem train_distance_problem (speed1 speed2 distance_before_meeting : ℝ) :
  speed1 = 30 ∧ 
  speed2 = 40 ∧ 
  distance_before_meeting = 70 → 
  speed1 + speed2 = distance_before_meeting →
  ∃ initial_distance : ℝ, initial_distance = 140 ∧ 
    initial_distance = distance_before_meeting + (speed1 + speed2) := by
  sorry

#check train_distance_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_problem_l544_54469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_weight_loss_duration_l544_54495

/-- The number of years Luca lost weight -/
def lucas_weight_loss_years : ℚ :=
  let barbis_monthly_loss : ℚ := 3/2  -- 1.5 kg per month
  let months_per_year : ℕ := 12
  let barbis_yearly_loss : ℚ := barbis_monthly_loss * months_per_year
  let lucas_yearly_loss : ℕ := 9
  let additional_loss : ℕ := 81
  let lucas_total_loss : ℚ := barbis_yearly_loss + additional_loss
  lucas_total_loss / lucas_yearly_loss

/-- Theorem stating that Luca lost weight for 11 years -/
theorem lucas_weight_loss_duration : lucas_weight_loss_years = 11 := by
  unfold lucas_weight_loss_years
  -- Perform the calculation
  simp [Int.ofNat]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucas_weight_loss_duration_l544_54495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ellipse_l544_54485

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  3 * x^2 - 12 * x + 2 * y^2 + 8 * y + 18 = 0

/-- The area of the ellipse -/
noncomputable def ellipse_area : ℝ := Real.pi * Real.sqrt (2/3)

/-- Theorem stating that the area of the ellipse defined by the given equation is π√(2/3) -/
theorem area_of_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), ellipse_equation x y ↔ (x - 2)^2 / a^2 + (y + 2)^2 / b^2 = 1) ∧
  ellipse_area = Real.pi * a * b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ellipse_l544_54485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_line_sum_of_squares_l544_54434

/-- A circle with center (x, y) and radius r -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The region R formed by the union of nine circular regions -/
noncomputable def region_R : Set (ℝ × ℝ) :=
  let circles : List Circle := [
    ⟨(1, 1), 0.5⟩, ⟨(1, 3), 0.5⟩, ⟨(1, 5), 0.5⟩,
    ⟨(3, 1), 0.5⟩, ⟨(3, 3), 0.5⟩, ⟨(3, 5), 0.5⟩,
    ⟨(5, 1), 0.5⟩, ⟨(5, 3), 0.5⟩, ⟨(5, 5), 0.5⟩
  ]
  sorry -- Union of all circles

/-- A line with slope 4 -/
structure Line where
  a : ℤ
  b : ℤ
  c : ℤ
  slope_is_4 : (4 : ℤ) * b = a
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  coprime : Nat.gcd a.toNat (Nat.gcd b.toNat c.toNat) = 1

/-- The line l divides region R into two equal areas -/
def divides_equally (l : Line) : Prop :=
  sorry -- Definition of equally dividing the region

theorem equal_area_line_sum_of_squares (l : Line) 
  (h : divides_equally l) : l.a^2 + l.b^2 + l.c^2 = 117 := by
  sorry

#check equal_area_line_sum_of_squares

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_line_sum_of_squares_l544_54434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_division_implies_equal_sides_l544_54409

/-- A convex polygon. -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool
  sides_ge_3 : sides ≥ 3

/-- A division of a polygon into triangles by non-intersecting diagonals. -/
structure PolygonDivision (P : ConvexPolygon) where
  triangles : ℕ
  non_intersecting : Bool
  isosceles : Bool

/-- The length of a side of a polygon. -/
noncomputable def side_length (P : ConvexPolygon) (s : ℕ) : ℝ :=
  sorry

/-- Two sides of a polygon are equal. -/
def has_two_equal_sides (P : ConvexPolygon) : Prop :=
  ∃ (s1 s2 : ℕ), s1 ≠ s2 ∧ s1 ≤ P.sides ∧ s2 ≤ P.sides ∧ 
    (∃ (l : ℝ), l > 0 ∧ (side_length P s1 = l ∧ side_length P s2 = l))

/-- 
  If a convex polygon can be divided into isosceles triangles 
  by non-intersecting diagonals, then it has at least two equal sides.
-/
theorem isosceles_division_implies_equal_sides (P : ConvexPolygon) 
  (D : PolygonDivision P) : 
  D.non_intersecting ∧ D.isosceles → has_two_equal_sides P :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_division_implies_equal_sides_l544_54409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_rational_l544_54471

def P (n : ℕ) (t : ℝ) : ℝ := (Finset.range (2*n + 1)).sum (λ i ↦ t^i)

theorem x_is_rational (n : ℕ) (x : ℝ) 
  (h1 : ∃ q : ℚ, (P n x : ℝ) = q) 
  (h2 : ∃ q : ℚ, (P n (x^2) : ℝ) = q) : 
  ∃ q : ℚ, (x : ℝ) = q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_is_rational_l544_54471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_implies_sum_l544_54490

theorem determinant_zero_implies_sum (a b : ℝ) : 
  a ≠ b →
  Matrix.det (!![2, 5, 10; 4, a, b; 4, b, a]) = 0 →
  a + b = 30 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_implies_sum_l544_54490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ryan_english_hours_l544_54401

/-- The number of hours Ryan spends on learning English each day -/
def hours_english : ℝ := 6

/-- The number of days Ryan learns -/
def days : ℕ := 5

/-- The number of hours Ryan spends on learning Chinese each day -/
def hours_chinese : ℝ := 7

/-- The total number of hours Ryan spends on learning English and Chinese -/
def total_hours : ℝ := 65

theorem ryan_english_hours : hours_english = 6 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ryan_english_hours_l544_54401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_15_is_225_l544_54494

/-- An arithmetic progression where the sum of the 4th and 12th terms is 30 -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference
  sum_4_12 : a + 3*d + a + 11*d = 30  -- Sum of 4th and 12th terms is 30

/-- The sum of the first n terms of an arithmetic progression -/
def sum_n (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  n / 2 * (2 * ap.a + (n - 1) * ap.d)

/-- Theorem: The sum of the first 15 terms is 225 -/
theorem sum_15_is_225 (ap : ArithmeticProgression) : sum_n ap 15 = 225 := by
  sorry

#eval sum_n { a := 1, d := 2, sum_4_12 := by norm_num } 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_15_is_225_l544_54494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erased_length_cm_l544_54457

-- Define the original line length in meters
def original_length_m : ℝ := 1

-- Define the final line length in centimeters
def final_length_cm : ℝ := 90

-- Define the conversion factor from meters to centimeters
def meters_to_cm : ℝ := 100

-- Theorem to prove
theorem erased_length_cm : 
  meters_to_cm * original_length_m - final_length_cm = 10 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erased_length_cm_l544_54457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_angle_l544_54447

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus point
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define the intersection points
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | parabola x y ∧ line_through_focus k x y}

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the angle of inclination
noncomputable def angle_of_inclination (k : ℝ) : ℝ := Real.arctan k

-- Statement of the theorem
theorem parabola_intersection_angle {k : ℝ} (hk : k > 0)
  (A B : ℝ × ℝ) (hA : A ∈ intersection_points k) (hB : B ∈ intersection_points k)
  (h_dist : 1 / distance A focus - 1 / distance B focus = 1 / 2)
  (h_angle : 0 < angle_of_inclination k ∧ angle_of_inclination k < Real.pi / 2) :
  angle_of_inclination k = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_angle_l544_54447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_segment_sums_l544_54458

/-- Represents the star diagram with 7 segments of 4 circles each --/
structure StarDiagram where
  circles : Fin 14 → ℕ
  segments : Fin 7 → Fin 4 → Fin 14

/-- The set of numbers to be placed in the circles --/
def numberSet : Multiset ℕ := 2 ::ₘ 2 ::ₘ 3 ::ₘ 3 ::ₘ 3 ::ₘ 4 ::ₘ 4 ::ₘ 4 ::ₘ 4 ::ₘ 5 ::ₘ 5 ::ₘ 5 ::ₘ 5 ::ₘ 5 ::ₘ 0

/-- Predicate to check if a star diagram is valid according to the problem conditions --/
def isValidStarDiagram (sd : StarDiagram) : Prop :=
  (∀ i : Fin 14, sd.circles i ∈ numberSet) ∧
  (∀ i j : Fin 14, i ≠ j → sd.circles i = sd.circles j → (sd.circles i = 2 ∨ sd.circles i = 3 ∨ sd.circles i = 4 ∨ sd.circles i = 5)) ∧
  (Multiset.count 2 (Multiset.ofList (List.ofFn sd.circles)) = 2) ∧
  (Multiset.count 3 (Multiset.ofList (List.ofFn sd.circles)) = 3) ∧
  (Multiset.count 4 (Multiset.ofList (List.ofFn sd.circles)) = 4) ∧
  (Multiset.count 5 (Multiset.ofList (List.ofFn sd.circles)) = 5)

/-- Predicate to check if all segment sums are equal --/
def allSegmentSumsEqual (sd : StarDiagram) : Prop :=
  ∀ i j : Fin 7, (List.sum (List.map (λ k ↦ sd.circles (sd.segments i k)) (List.range 4))) = 
                 (List.sum (List.map (λ k ↦ sd.circles (sd.segments j k)) (List.range 4)))

/-- Theorem stating that it's impossible to have a valid star diagram with all segment sums equal --/
theorem no_equal_segment_sums :
  ¬∃ (sd : StarDiagram), isValidStarDiagram sd ∧ allSegmentSumsEqual sd := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_segment_sums_l544_54458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_complex_z_is_purely_imaginary_z_is_zero_l544_54444

/-- Definition of the complex number z in terms of real number k -/
def z (k : ℝ) : ℂ := (k^2 - 3*k - 4 : ℝ) + (k^2 - 5*k - 6 : ℝ) * Complex.I

/-- z is a real number if and only if k = 6 or k = -1 -/
theorem z_is_real (k : ℝ) : (z k).im = 0 ↔ k = 6 ∨ k = -1 := by sorry

/-- z is a complex number if and only if k ≠ 6 and k ≠ -1 -/
theorem z_is_complex (k : ℝ) : (z k).im ≠ 0 ↔ k ≠ 6 ∧ k ≠ -1 := by sorry

/-- z is a purely imaginary number if and only if k = 4 -/
theorem z_is_purely_imaginary (k : ℝ) : (z k).re = 0 ∧ (z k).im ≠ 0 ↔ k = 4 := by sorry

/-- z is equal to 0 if and only if k = -1 -/
theorem z_is_zero (k : ℝ) : z k = 0 ↔ k = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_complex_z_is_purely_imaginary_z_is_zero_l544_54444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_one_l544_54402

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then (9/4)^x else Real.log x / Real.log 8

-- State the theorem
theorem f_sum_equals_one : f (-1/2) + f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_one_l544_54402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l544_54417

noncomputable def f (x : ℝ) := Real.sin (4 * x - Real.pi / 3)

theorem axis_of_symmetry :
  ∃ (x : ℝ), x = 11 * Real.pi / 24 ∧ 
  ∀ (y : ℝ), f (x - y) = f (x + y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l544_54417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_theorem_l544_54423

/-- Friend represents a runner with a given distance and time -/
structure Friend where
  distance : ℚ
  time : ℚ

/-- Calculate the total combined time for two friends to run a new distance -/
def totalCombinedTime (f1 f2 : Friend) (newDistance : ℚ) : ℚ :=
  (newDistance * f1.time / f1.distance) + (newDistance * f2.time / f2.distance)

theorem race_time_theorem (f1 f2 : Friend) (h1 : f1.distance = 3 ∧ f1.time = 21)
    (h2 : f2.distance = 3 ∧ f2.time = 24) :
  totalCombinedTime f1 f2 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_theorem_l544_54423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_increase_l544_54440

/-- Represents a batsman's performance -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  notOutCount : Nat

/-- Calculate the batting average -/
def battingAverage (stats : BatsmanStats) : Rat :=
  stats.totalRuns / (stats.innings - stats.notOutCount)

theorem batsman_average_increase 
  (stats11 : BatsmanStats) 
  (stats12 : BatsmanStats) :
  stats11.innings = 11 ∧
  stats11.notOutCount = 0 ∧
  stats12.innings = 12 ∧
  stats12.notOutCount = 0 ∧
  stats12.totalRuns = stats11.totalRuns + 48 ∧
  battingAverage stats12 = 26 →
  battingAverage stats12 - battingAverage stats11 = 2 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_increase_l544_54440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_eight_l544_54424

/-- The function representing the curve y = -x^2 + 3ln(x) -/
noncomputable def f (x : ℝ) : ℝ := -x^2 + 3 * Real.log x

/-- The function representing the line y = x + 2 -/
def g (x : ℝ) : ℝ := x + 2

/-- Point P on the curve f -/
structure PointP where
  a : ℝ
  b : ℝ
  h : b = f a

/-- Point Q on the line g -/
structure PointQ where
  c : ℝ
  d : ℝ
  h : d = g c

/-- The squared distance between points P and Q -/
def squaredDistance (p : PointP) (q : PointQ) : ℝ :=
  (p.a - q.c)^2 + (p.b - q.d)^2

theorem min_distance_is_eight :
  ∀ (p : PointP) (q : PointQ), ∃ (minDist : ℝ), minDist = 8 ∧ squaredDistance p q ≥ minDist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_eight_l544_54424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l544_54441

/-- Represents an ellipse as a set of points in ℝ² -/
def Set.isEllipse (S : Set (ℝ × ℝ)) : Prop := sorry

/-- The set of foci of an ellipse -/
def Set.foci (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- Indicates that a set is tangent to a curve at a point -/
def Set.tangentTo (S : Set (ℝ × ℝ)) (curve : ℝ → ℝ × ℝ) (t : ℝ) : Prop := sorry

/-- The length of the major axis of an ellipse -/
noncomputable def Set.majorAxisLength (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- Given an ellipse tangent to the x-axis with foci at (1, 1) and (5, 2),
    prove that its major axis has length 5. -/
theorem ellipse_major_axis_length :
  ∀ (E : Set (ℝ × ℝ)),
  E.isEllipse →
  (∃ (y : ℝ), E.tangentTo (λ x => (x, 0)) y) →
  (1, 1) ∈ E.foci →
  (5, 2) ∈ E.foci →
  E.majorAxisLength = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_major_axis_length_l544_54441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_half_necessary_not_sufficient_l544_54418

theorem cos_2alpha_half_necessary_not_sufficient :
  (∀ α : ℝ, (∃ k : ℤ, α = k * π + π / 6) → Real.cos (2 * α) = 1 / 2) ∧
  (∃ α : ℝ, Real.cos (2 * α) = 1 / 2 ∧ ∀ k : ℤ, α ≠ k * π + π / 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_half_necessary_not_sufficient_l544_54418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l544_54488

noncomputable def f (a : ℝ) : ℝ := ∫ x in (Set.Icc 0 1), 2 * a * x^2 - a^2 * x

theorem f_max_value :
  ∃ (max : ℝ), max = 2/9 ∧ ∀ (a : ℝ), f a ≤ max :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l544_54488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l544_54438

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x^3 - 3*x^4) / (x + 2*x^2 - 3*x^3)

theorem inequality_solution :
  ∀ x : ℝ, f x ≥ -1 ↔ x ∈ Set.Icc (-1 : ℝ) (-1/3) ∪ 
                     Set.Ioo (-1/3 : ℝ) 0 ∪ 
                     Set.Ioo (0 : ℝ) 1 ∪ 
                     Set.Ioi (1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l544_54438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_25_l544_54487

/-- The angle between hour and minute hands at 3:00 -/
def initial_angle : ℚ := 90

/-- Degrees the minute hand moves per minute -/
def minute_hand_speed : ℚ := 6

/-- Degrees the hour hand moves per minute -/
def hour_hand_speed : ℚ := 1/2

/-- Number of minutes passed from 3:00 to 3:25 -/
def minutes_passed : ℕ := 25

/-- Calculates the smaller angle between clock hands at 3:25 -/
def clock_angle (init_angle hour_speed min_speed mins : ℚ) : ℚ :=
  let angle := init_angle + hour_speed * mins - min_speed * mins
  let positive_angle := if angle < 0 then 360 + angle else angle
  min positive_angle (360 - positive_angle)

theorem clock_angle_at_3_25 :
  clock_angle initial_angle hour_hand_speed minute_hand_speed (minutes_passed : ℚ) = 47.5 := by
  sorry

#eval clock_angle initial_angle hour_hand_speed minute_hand_speed (minutes_passed : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_25_l544_54487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l544_54420

/-- A power function passing through (3, 1/9) -/
noncomputable def f : ℝ → ℝ := λ x => x^(-2 : ℤ)

/-- The function g(x) = (x-1)f(x) -/
noncomputable def g : ℝ → ℝ := λ x => (x - 1) * f x

theorem max_value_of_g :
  ∃ (x : ℝ), x ∈ Set.Icc 1 3 ∧
  (∀ (y : ℝ), y ∈ Set.Icc 1 3 → g y ≤ g x) ∧
  g x = (1 : ℝ) / 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l544_54420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_sum_is_261_l544_54443

/-- Represents the financial transactions and final balances of a group of people --/
structure FinancialGroup where
  earl_initial : ℕ
  fred_initial : ℕ
  greg_initial : ℕ
  hannah_initial : ℕ
  isabella_initial : ℕ
  earl_owes_fred : ℕ
  earl_owes_hannah : ℕ
  earl_owes_isabella : ℕ
  fred_owes_greg : ℕ
  fred_owes_hannah : ℕ
  fred_owes_isabella : ℕ
  greg_owes_earl : ℕ
  greg_owes_hannah : ℕ
  greg_owes_isabella : ℕ
  hannah_owes_greg : ℕ
  hannah_owes_earl : ℕ
  hannah_owes_fred : ℕ
  hannah_owes_isabella : ℕ
  isabella_owes_earl : ℕ
  isabella_owes_greg : ℕ
  isabella_owes_hannah : ℕ

/-- Calculates the final balance for a person after all debts are settled --/
def finalBalance (initial : ℕ) (owed : ℕ) (owes : ℕ) : ℕ :=
  initial + owed - owes

/-- Theorem: The sum of final balances for Greg, Earl, Hannah, and Isabella is $261 --/
theorem final_sum_is_261 (fg : FinancialGroup) 
  (h1 : fg.earl_initial = 90)
  (h2 : fg.fred_initial = 48)
  (h3 : fg.greg_initial = 36)
  (h4 : fg.hannah_initial = 72)
  (h5 : fg.isabella_initial = 60)
  (h6 : fg.earl_owes_fred = 28)
  (h7 : fg.earl_owes_hannah = 30)
  (h8 : fg.earl_owes_isabella = 15)
  (h9 : fg.fred_owes_greg = 32)
  (h10 : fg.fred_owes_hannah = 10)
  (h11 : fg.fred_owes_isabella = 20)
  (h12 : fg.greg_owes_earl = 40)
  (h13 : fg.greg_owes_hannah = 20)
  (h14 : fg.greg_owes_isabella = 8)
  (h15 : fg.hannah_owes_greg = 15)
  (h16 : fg.hannah_owes_earl = 25)
  (h17 : fg.hannah_owes_fred = 5)
  (h18 : fg.hannah_owes_isabella = 10)
  (h19 : fg.isabella_owes_earl = 18)
  (h20 : fg.isabella_owes_greg = 4)
  (h21 : fg.isabella_owes_hannah = 12) : 
  finalBalance fg.greg_initial 
    (fg.fred_owes_greg + fg.hannah_owes_greg + fg.isabella_owes_greg)
    (fg.greg_owes_earl + fg.greg_owes_hannah + fg.greg_owes_isabella) +
  finalBalance fg.earl_initial
    (fg.greg_owes_earl + fg.hannah_owes_earl + fg.isabella_owes_earl)
    (fg.earl_owes_fred + fg.earl_owes_hannah + fg.earl_owes_isabella) +
  finalBalance fg.hannah_initial
    (fg.earl_owes_hannah + fg.fred_owes_hannah + fg.greg_owes_hannah + fg.isabella_owes_hannah)
    (fg.hannah_owes_greg + fg.hannah_owes_earl + fg.hannah_owes_fred + fg.hannah_owes_isabella) +
  finalBalance fg.isabella_initial
    (fg.earl_owes_isabella + fg.fred_owes_isabella + fg.greg_owes_isabella + fg.hannah_owes_isabella)
    (fg.isabella_owes_earl + fg.isabella_owes_greg + fg.isabella_owes_hannah) = 261 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_sum_is_261_l544_54443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_sox_series_win_probability_l544_54437

/-- Probability that Red Sox win game n -/
def prob_red_sox_win (n : ℕ) : ℚ := (n - 1 : ℚ) / 6

/-- A best-of-seven series ends when a team wins 4 games -/
def series_length : ℕ := 7

/-- Number of wins needed to win the series -/
def wins_needed : ℕ := 4

/-- Probability of winning the series given a probability function for winning each game -/
noncomputable def probability_of_winning_series (prob_win : ℕ → ℚ) (series_length : ℕ) (wins_needed : ℕ) : ℚ :=
  sorry -- This definition would require complex probability calculations

/-- Theorem: The probability that the Red Sox win the series is 1/2 -/
theorem red_sox_series_win_probability :
  (∀ n, n ≤ series_length → prob_red_sox_win n ∈ Set.Icc (0 : ℚ) 1) →
  ∃ p : ℚ, p = 1/2 ∧ p = probability_of_winning_series prob_red_sox_win series_length wins_needed :=
by
  sorry -- The proof would go here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_sox_series_win_probability_l544_54437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l544_54476

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + Real.sin x ^ 2

theorem problem_solution (A B C : ℝ) : 
  (0 < C ∧ C < Real.pi / 2) →  -- C is an acute angle
  (Real.cos B = 1 / 3) →
  (f (C / 2) = -1 / 4) →
  (∃ (x : ℝ), f x = (1 + Real.sqrt 3) / 2) ∧  -- maximum value
  (∀ (x : ℝ), f (x + Real.pi) = f x) ∧  -- smallest positive period
  (Real.sin A = (2 * Real.sqrt 2 + Real.sqrt 3) / 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l544_54476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l544_54467

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - 2 * x + 1)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → 0 ≤ a ∧ a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l544_54467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_problem_l544_54442

theorem triangle_cosine_problem (A B C : ℝ) (h_triangle : A + B + C = Real.pi)
  (h_sin_A : Real.sin A = 4/5) (h_cos_B : Real.cos B = 5/13) : Real.cos C = 33/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_problem_l544_54442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_knights_l544_54407

/-- Represents the type of statement an islander can make -/
inductive Statement where
  | more_liars_above : Statement
  | more_liars_below : Statement

/-- Represents an islander -/
structure Islander where
  number : Nat
  is_knight : Bool
  statement : Statement

/-- The problem setup -/
structure IslandSetup where
  islanders : List Islander
  total_count : Nat
  statements_valid : Bool
  unique_numbers : Bool

/-- The main theorem -/
theorem min_knights (setup : IslandSetup) :
  setup.total_count = 80 ∧ 
  setup.statements_valid ∧ 
  setup.unique_numbers →
  (setup.islanders.filter (λ i => i.is_knight)).length ≥ 70 := by
  sorry

#check min_knights

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_knights_l544_54407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_sum_squares_l544_54480

-- Define the circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 = 4 ∧ x ≥ 0 ∧ y ≥ 0

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def g (x : ℝ) : ℝ := 2^x

-- Theorem statement
theorem intersection_points_sum_squares (x₁ y₁ x₂ y₂ : ℝ) :
  circleC x₁ y₁ ∧ circleC x₂ y₂ ∧ 
  f x₁ = y₁ ∧ g x₁ = y₁ ∧
  f x₂ = y₂ ∧ g x₂ = y₂ →
  x₁^2 + x₂^2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_sum_squares_l544_54480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_at_negative_one_l544_54450

/-- The polynomial P(x) -/
noncomputable def P (x : ℝ) : ℝ := 1 - (1/4)*x + (1/8)*x^2

/-- The polynomial Q(x) -/
noncomputable def Q (x : ℝ) : ℝ := P x * P (x^2) * P (x^3) * P (x^4) * P (x^5)

/-- Theorem stating that Q(-1) equals 161051/32768 -/
theorem Q_at_negative_one : Q (-1) = 161051/32768 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_at_negative_one_l544_54450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_one_l544_54493

-- Define the function f as noncomputable due to its dependence on Real.pi
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ a then Real.sin (2 * Real.pi * (x - a))
  else x^2 - 2*(a + 1)*x + a^2

-- State the theorem
theorem f_composition_equals_one (a : ℝ) (h1 : a ∈ Set.Icc (-1) 1) 
  (h2 : f a (f a a) = 1) : a = -1 ∨ a = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_one_l544_54493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_divisible_pair_l544_54470

/-- Sequence u_n defined recursively -/
def u : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | (n + 2) => u (n + 1) + u n + 1

/-- The modulus we're interested in -/
def A : ℤ := 2011^2012

/-- The main theorem -/
theorem exists_divisible_pair :
  ∃ N : ℕ, N ≥ 1 ∧ A ∣ u N ∧ A ∣ u (N + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_divisible_pair_l544_54470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_for_sequence_sum_l544_54430

def sequence_a (n : ℕ) : ℕ := 2 * n - 1

def sum_a (n : ℕ) : ℚ := n / (2 * n + 1 : ℚ)

theorem max_n_for_sequence_sum : 
  (∀ n : ℕ, sum_a n ≤ 1009 / 2019) ∧ 
  sum_a 1009 = 1009 / 2019 ∧ 
  sum_a 1010 > 1009 / 2019 := by
  sorry

#check max_n_for_sequence_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_for_sequence_sum_l544_54430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_planes_l544_54496

def n₁ : ℝ × ℝ × ℝ := (3, 2, 1)
def n₂ : ℝ × ℝ × ℝ := (2, 0, -1)

theorem angle_between_planes :
  let dot_product := n₁.1 * n₂.1 + n₁.2.1 * n₂.2.1 + n₁.2.2 * n₂.2.2
  let magnitude_n₁ := Real.sqrt (n₁.1^2 + n₁.2.1^2 + n₁.2.2^2)
  let magnitude_n₂ := Real.sqrt (n₂.1^2 + n₂.2.1^2 + n₂.2.2^2)
  dot_product / (magnitude_n₁ * magnitude_n₂) = Real.sqrt 70 / 14 :=
by
  sorry

#check angle_between_planes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_planes_l544_54496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runs_needed_for_average_increase_l544_54448

/-- Calculates the runs needed in the next innings to achieve a desired average increase --/
def runsNeededForAverageIncrease (currentAverage : ℚ) (numInnings : ℕ) (desiredIncrease : ℚ) : ℕ :=
  let totalCurrentRuns := currentAverage * numInnings
  let newAverage := currentAverage + desiredIncrease
  let totalRequiredRuns := newAverage * (numInnings + 1)
  (totalRequiredRuns - totalCurrentRuns).ceil.toNat

/-- Theorem stating the runs needed to increase average by 4 after 10 innings with 32 run average --/
theorem runs_needed_for_average_increase :
  runsNeededForAverageIncrease 32 10 4 = 76 := by
  sorry

#eval runsNeededForAverageIncrease 32 10 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runs_needed_for_average_increase_l544_54448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_unique_new_fixed_point_l544_54486

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := -Real.exp x - 2 * x

-- State the theorem
theorem g_has_unique_new_fixed_point :
  ∃! x : ℝ, g x = (deriv g) x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_has_unique_new_fixed_point_l544_54486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midline_length_is_correct_main_result_l544_54465

/-- A right trapezoid with specific properties -/
structure RightTrapezoid where
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The length of the lateral side (which is equal to the diagonal) -/
  lateral : ℝ
  /-- Assertion that the height is 2 -/
  height_is_two : height = 2
  /-- Assertion that the lateral side is 4 -/
  lateral_is_four : lateral = 4

/-- The length of the midline in the specified right trapezoid -/
noncomputable def midline_length (t : RightTrapezoid) : ℝ := 3 * Real.sqrt 3

/-- Theorem stating that the midline length is correct for the given trapezoid -/
theorem midline_length_is_correct (t : RightTrapezoid) : 
  midline_length t = 3 * Real.sqrt 3 := by
  -- Unfold the definition of midline_length
  unfold midline_length
  -- The equality holds by definition
  rfl

/-- Theorem proving the main result -/
theorem main_result (t : RightTrapezoid) : 
  ∃ (m : ℝ), m = midline_length t ∧ m = 3 * Real.sqrt 3 := by
  -- We use the midline length as our witness
  use midline_length t
  constructor
  · -- First part of the conjunction is trivial
    rfl
  · -- Second part follows from our previous theorem
    exact midline_length_is_correct t

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midline_length_is_correct_main_result_l544_54465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l544_54416

def M : Set ℝ := {x | x^2 - 3*x - 10 < 0}

def N : Set ℝ := {x : ℝ | ∃ n : ℤ, x = n ∧ |n| < 2}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l544_54416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_special_polynomial_l544_54435

/-- Given a polynomial of the form (x + a/x)(2x - 1/x)^5 where the sum of its coefficients is 2,
    prove that the constant term in its expanded form is 40. -/
theorem constant_term_of_special_polynomial (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (x + a/x) * (2*x - 1/x)^5 = 2) →
  ∃ f : Polynomial ℝ, (∀ x : ℝ, x ≠ 0 → Polynomial.eval x f = (x + a/x) * (2*x - 1/x)^5) ∧ 
                      Polynomial.coeff f 0 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_special_polynomial_l544_54435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l544_54472

theorem coin_flip_probability : 
  let n : ℕ := 12  -- total number of flips
  let k : ℕ := 9   -- number of heads
  let p : ℚ := 1/2  -- probability of heads for a fair coin
  
  -- Probability of exactly 9 heads in 12 flips
  (Nat.choose n k : ℚ) * p^k * (1-p)^(n-k) = 55/1024 :=
by
  -- Set up the values
  let n : ℕ := 12
  let k : ℕ := 9
  let p : ℚ := 1/2

  -- Perform the calculation
  have h : (Nat.choose n k : ℚ) * p^k * (1-p)^(n-k) = 55/1024 := by
    -- The actual proof would go here
    sorry

  -- Return the result
  exact h

-- Check that the theorem is recognized
#check coin_flip_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l544_54472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_difference_l544_54414

/-- A race between two runners A and B -/
structure Race where
  distance : ℚ
  a_lead : ℚ
  a_time : ℚ

/-- Calculate the time difference between runners A and B -/
def time_difference (r : Race) : ℚ :=
  let a_speed := r.distance / r.a_time
  let b_distance := r.distance - r.a_lead
  let b_speed := b_distance / r.a_time
  let b_remaining_time := r.a_lead / b_speed
  b_remaining_time

/-- Theorem stating that in the given race conditions, A beats B by 7 seconds -/
theorem race_time_difference : 
  let r : Race := { distance := 120, a_lead := 56, a_time := 8 }
  time_difference r = 7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_difference_l544_54414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l544_54400

-- Define the function f(x) = 2^x + 2x - 3
noncomputable def f (x : ℝ) := Real.exp (x * Real.log 2) + 2*x - 3

-- State the theorem
theorem zero_in_interval :
  (∀ x y : ℝ, x < y → f x < f y) →  -- f is monotonically increasing
  ContinuousOn f Set.univ →     -- f is continuous on ℝ
  f (1/2) < 0 →                 -- f(1/2) < 0
  f 1 > 0 →                     -- f(1) > 0
  ∃ x : ℝ, x ∈ Set.Ioo (1/2) 1 ∧ f x = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l544_54400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l544_54428

theorem divisibility_condition (a m n : ℕ) (ha : a > 0) (hm : m > 0) (hn : n > 0) :
  (a^m + 1) ∣ ((a + 1)^n) →
  ((a = 1 ∧ m > 0 ∧ n > 0) ∨ (a = 2 ∧ m = 3 ∧ n ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l544_54428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_kilos_proof_l544_54432

/-- The cost of oranges and apples given specific quantities -/
def cost_equation (orange_kilos : ℝ) (apple_kilos : ℝ) (total_cost : ℝ) : Prop :=
  ∃ (orange_price apple_price : ℝ), 
    orange_price * orange_kilos + apple_price * apple_kilos = total_cost ∧ 
    orange_price = 29 ∧ apple_price = 29

/-- The number of kilos of oranges bought in the first scenario -/
def orange_kilos_first_scenario : ℝ := 5

theorem orange_kilos_proof :
  cost_equation orange_kilos_first_scenario 5 419 ∧
  cost_equation 5 7 488 →
  orange_kilos_first_scenario = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_kilos_proof_l544_54432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l544_54461

-- Define the function f(x) as noncomputable due to the use of real exponentiation
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then -x + 3*a else a^x

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/3 : ℝ) 1 ∧ a ≠ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l544_54461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_product_l544_54425

theorem quadratic_roots_product : 
  Finset.prod (Finset.range 20) id = 121645100408832000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_product_l544_54425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_area_l544_54468

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with equation x²/a² + y²/b² = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Check if a point is on the ellipse -/
def onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Check if two points are symmetric with respect to the origin -/
def symmetricOrigin (p q : Point) : Prop :=
  p.x = -q.x ∧ p.y = -q.y

/-- The main theorem -/
theorem ellipse_quadrilateral_area 
  (e : Ellipse)
  (f₁ f₂ p q : Point)
  (h_ellipse : e.a^2 = 16 ∧ e.b^2 = 4)
  (h_foci : f₁.x^2 - f₂.x^2 + f₁.y^2 - f₂.y^2 = 0 ∧ distance f₁ f₂ = distance p q)
  (h_p_on_ellipse : onEllipse p e)
  (h_q_on_ellipse : onEllipse q e)
  (h_symmetric : symmetricOrigin p q) :
  distance p f₁ * distance p f₂ = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_quadrilateral_area_l544_54468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_reach_food_probability_l544_54413

/-- Represents a lily pad with its number -/
structure LilyPad where
  number : Nat

/-- Represents the frog's position -/
structure FrogPosition where
  pad : LilyPad

/-- Represents a jump probability -/
def JumpProbability := Rat

/-- Represents the game setup -/
structure GameSetup where
  pads : List LilyPad
  predators : List LilyPad
  food : LilyPad
  start : LilyPad
  nextJumpProb : JumpProbability
  skipJumpProb : JumpProbability

/-- Calculates the probability of reaching the food pad without hitting predators -/
noncomputable def reachFoodProbability (setup : GameSetup) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem fiona_reach_food_probability (setup : GameSetup) :
  setup.pads = (List.range 16).map (λ n => ⟨n⟩) →
  setup.predators = [⟨4⟩, ⟨8⟩] →
  setup.food = ⟨14⟩ →
  setup.start = ⟨0⟩ →
  setup.nextJumpProb = (1 : Rat) / 2 →
  setup.skipJumpProb = (1 : Rat) / 2 →
  reachFoodProbability setup = 9 / 128 :=
by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiona_reach_food_probability_l544_54413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_representation_l544_54419

/-- Represents a distance in meters, where positive values indicate northward movement
    and negative values indicate southward movement. -/
def Direction := ℤ

/-- Coercion from integers to Direction -/
instance : Coe ℤ Direction := ⟨id⟩

/-- OfNat instance for Direction -/
instance (n : Nat) : OfNat Direction n := ⟨(n : ℤ)⟩

/-- Negation for Direction -/
instance : Neg Direction := ⟨Int.neg⟩

/-- Given that 80 represents moving 80m north, prove that -50 represents moving 50m south -/
theorem direction_representation : 
  (80 : Direction) = 80 → ((-50) : Direction) = -50 := by
  intro h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_representation_l544_54419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_a_l544_54484

noncomputable def a : ℝ := (11*66 + 12*67 + 13*68 + 14*69 + 15*70) / (11*65 + 12*66 + 13*67 + 14*68 + 15*69) * 100

theorem integer_part_of_a : ⌊a⌋ = 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_a_l544_54484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l544_54481

/-- Hyperbola eccentricity theorem -/
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  ∃ (P : ℝ × ℝ) (F₁ : ℝ × ℝ),
    (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
    (P.2 = (b / (3 * a)) * P.1) ∧
    (F₁.2 = 0) ∧
    (F₁.1 < 0) ∧
    ((P.1 - F₁.1) * (P.2 - F₁.2) = 0) →
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  e = 3 * Real.sqrt 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l544_54481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l544_54478

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [CompleteSpace V]

-- Define points in the space
variable (E F G H : V)

-- Define the property of points being coplanar
def coplanar (A B C D : V) : Prop :=
  ∃ (a b c d : ℝ), a • A + b • B + c • C + d • D = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0)

-- Define the property of lines intersecting
def lines_intersect (A B C D : V) : Prop :=
  ∃ (t s : ℝ), A + t • (B - A) = C + s • (D - C)

-- State the theorem
theorem sufficient_not_necessary :
  (∀ E F G H : V, ¬ coplanar E F G H → ¬ lines_intersect E F G H) ∧
  ∃ E F G H : V, ¬ lines_intersect E F G H ∧ coplanar E F G H := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l544_54478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_taylor_series_at_pi_fourth_l544_54462

/-- The Taylor series expansion of sin(x) at x₀ = π/4 -/
theorem sin_taylor_series_at_pi_fourth (x : ℝ) : 
  Real.sin x = (Real.sqrt 2 / 2) * (1 + ∑' n, ((-1)^(n-1) : ℝ) * (x - π/4)^n / n.factorial) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_taylor_series_at_pi_fourth_l544_54462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_when_a_is_one_f_one_zero_point_l544_54452

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.sin x - a) * (a - Real.cos x) + Real.sqrt 2 * a

theorem f_range_when_a_is_one :
  Set.range (f 1) = Set.Icc (-3/2) (Real.sqrt 2) :=
sorry

theorem f_one_zero_point (a : ℝ) :
  a ≥ 1 →
  (∃! x, x ∈ Set.Icc 0 Real.pi ∧ f a x = 0) ↔
  (a ∈ Set.Icc 1 (Real.sqrt 2 + 1) ∨ a = Real.sqrt 2 + Real.sqrt 6 / 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_when_a_is_one_f_one_zero_point_l544_54452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l544_54464

theorem triangle_third_side_length (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 11) (h3 : θ = 150 * Real.pi / 180) :
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ) ∧ c = Real.sqrt (221 + 110 * Real.sqrt 3) := by
  sorry

#check triangle_third_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l544_54464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_l544_54404

def f (n : ℕ+) : ℚ :=
  (Finset.range n).sum (λ i => 1 / ((i + 1 : ℚ) ^ 2))

def g (n : ℕ+) : ℚ :=
  (1 / 2) * (3 - 1 / ((n : ℚ) ^ 2))

theorem f_le_g : ∀ n : ℕ+, f n ≤ g n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_l544_54404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_neg_one_range_when_f_equals_abs_x_minus_a_l544_54436

-- Define the function f
def f (x a : ℝ) : ℝ := |2 * x| + |x + a|

-- Part I
theorem solution_set_when_a_neg_one :
  let a : ℝ := -1
  {x : ℝ | f x a ≤ 4} = Set.Icc (-1) (5/3) := by sorry

-- Part II
theorem range_when_f_equals_abs_x_minus_a (a : ℝ) :
  (∃ x, f x a = |x - a|) →
  (a > 0 → {x : ℝ | f x a = |x - a|} = Set.Icc (-a) 0) ∧
  (a < 0 → {x : ℝ | f x a = |x - a|} = Set.Icc 0 (-a)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_neg_one_range_when_f_equals_abs_x_minus_a_l544_54436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stage_150_sticks_l544_54411

/-- Arithmetic sequence with first term 4 and common difference 4 -/
def arithmeticSequence : ℕ → ℕ
| 0 => 4  -- Adding the base case for 0
| n + 1 => arithmeticSequence n + 4

theorem stage_150_sticks : arithmeticSequence 149 = 600 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stage_150_sticks_l544_54411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_added_to_salt_solution_l544_54497

/-- Proves the amount of water added to achieve the desired salt concentration -/
theorem water_added_to_salt_solution : ℝ := by
  -- Initial volume of solution
  let initial_volume : ℝ := 149.99999999999994
  -- Initial salt concentration
  let initial_salt_concentration : ℝ := 0.20
  -- Fraction of water evaporated
  let water_evaporation_fraction : ℝ := 0.25
  -- Amount of salt added
  let salt_added : ℝ := 20
  -- Final salt concentration
  let final_salt_concentration : ℝ := 1/3

  -- Calculate initial amount of salt
  let initial_salt : ℝ := initial_volume * initial_salt_concentration
  -- Calculate amount of water evaporated
  let water_evaporated : ℝ := initial_volume * water_evaporation_fraction
  -- Calculate remaining volume after evaporation
  let remaining_volume : ℝ := initial_volume - water_evaporated
  -- Calculate new amount of salt after addition
  let new_salt_amount : ℝ := initial_salt + salt_added

  -- Define the amount of water added as a variable
  let water_added : ℝ := 37.5

  -- The equation that needs to be satisfied
  have h : new_salt_amount / (remaining_volume + water_added) = final_salt_concentration := by
    sorry

  -- Return the amount of water added
  exact water_added

/- Proof goes here -/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_added_to_salt_solution_l544_54497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l544_54408

theorem range_of_x (x : ℝ) : 
  (0 < x ∧ x ≠ 1 ∧ Real.log (2*x^2 + x - 1) > Real.log 2 - 1) → x > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l544_54408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_T_l544_54427

-- Define the sets P and T
def P : Set ℝ := {x : ℝ | |x| > 2}
def T : Set ℝ := {x : ℝ | Real.exp (Real.log 3 * x) > 1}

-- State the theorem
theorem intersection_of_P_and_T : P ∩ T = {x : ℝ | x > 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_T_l544_54427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l544_54421

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- The theorem stating that the hyperbola x²/4 - y²/2 = 1 has eccentricity √6/2 -/
theorem hyperbola_eccentricity : eccentricity 2 (Real.sqrt 2) = Real.sqrt 6 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l544_54421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_k_value_l544_54410

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def is_collinear (v w : V) : Prop := ∃ (t : ℝ), v = t • w

theorem collinear_vectors_k_value
  (e₁ e₂ : V)
  (h_non_collinear : ¬ is_collinear e₁ e₂)
  (a b : V)
  (h_a : a = 2 • e₁ - e₂)
  (h_b : ∃ k : ℝ, b = k • e₁ + e₂)
  (h_collinear : is_collinear a b) :
  ∃ k : ℝ, b = k • e₁ + e₂ ∧ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_k_value_l544_54410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_sum_theorem_l544_54489

theorem base_sum_theorem : ∃! (R₁ R₂ : ℕ), 
  (R₁ > 1 ∧ R₂ > 1) ∧
  ((2 * R₁^3 + 4 * R₁^2 + 5 * R₁ + 1) / (R₁^4 - 1) = (3 * R₂^3 + 6 * R₂^2 + 1 * R₂ + 4) / (R₂^4 - 1)) ∧
  ((5 * R₁^3 + 1 * R₁^2 + 2 * R₁ + 4) / (R₁^4 - 1) = (6 * R₂^3 + 1 * R₂^2 + 4 * R₂ + 3) / (R₂^4 - 1)) ∧
  R₁ + R₂ = 23 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_sum_theorem_l544_54489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_eat_three_pastries_l544_54429

/-- Represents the state of plates and pastries -/
structure PlateState where
  num_plates : Nat
  num_pastries : Nat
  pastries_per_plate : List Nat

/-- Represents the allowed operations -/
inductive Operation
  | EatSamePlate : Operation
  | MoveToEmptyPlate : Operation

/-- Defines if a state is valid -/
def is_valid_state (state : PlateState) : Prop :=
  state.num_plates = 2019 ∧
  state.num_pastries ≤ 2019 ∧
  state.pastries_per_plate.length = state.num_plates ∧
  state.pastries_per_plate.sum = state.num_pastries

/-- Defines if an operation is applicable to a state -/
def can_apply_operation (op : Operation) (state : PlateState) : Prop :=
  match op with
  | Operation.EatSamePlate => ∃ i j, i ≠ j ∧ 
      i < state.pastries_per_plate.length ∧
      j < state.pastries_per_plate.length ∧
      state.pastries_per_plate[i]! = state.pastries_per_plate[j]!
  | Operation.MoveToEmptyPlate => ∃ i, 
      i < state.pastries_per_plate.length ∧
      state.pastries_per_plate[i]! = 0 ∧ 
      ∃ j, j < state.pastries_per_plate.length ∧ state.pastries_per_plate[j]! > 0

/-- Defines the result of applying an operation -/
def apply_operation (op : Operation) (state : PlateState) : PlateState :=
  sorry -- Implementation details omitted

/-- The main theorem to prove -/
theorem can_eat_three_pastries (initial_state : PlateState) :
  is_valid_state initial_state →
  ∃ (sequence : List Operation),
    let final_state := sequence.foldl (fun s o => apply_operation o s) initial_state
    2019 - final_state.num_pastries ≥ 3 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_eat_three_pastries_l544_54429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camel_count_l544_54422

/-- Represents the number of one-hump camels -/
def one_hump_camels : ℕ := sorry

/-- Represents the number of two-hump camels -/
def two_hump_camels : ℕ := sorry

/-- The total number of humps -/
def total_humps : ℕ := 23

/-- The total number of legs -/
def total_legs : ℕ := 60

/-- Each camel has 4 legs -/
def legs_per_camel : ℕ := 4

theorem camel_count :
  one_hump_camels + two_hump_camels = 15 ∧
  one_hump_camels + 2 * two_hump_camels = total_humps ∧
  legs_per_camel * (one_hump_camels + two_hump_camels) = total_legs := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_camel_count_l544_54422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_circumcenter_implies_right_triangle_l544_54483

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the excircle points
variable (A₁ B₁ C₁ : EuclideanSpace ℝ (Fin 2))

-- Define the circumcenter and circumcircle
noncomputable def circumcenter (P Q R : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := sorry

def on_circumcircle (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define what it means for a triangle to be right-angled
def is_right_triangle (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the excircle condition
def is_excircle_point (P Q R S : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- State the theorem
theorem excircle_circumcenter_implies_right_triangle 
  (h_A₁ : is_excircle_point A B C A₁)
  (h_B₁ : is_excircle_point B C A B₁)
  (h_C₁ : is_excircle_point C A B C₁)
  (h_circumcenter : on_circumcircle A B C (circumcenter A₁ B₁ C₁)) :
  is_right_triangle A B C := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_circumcenter_implies_right_triangle_l544_54483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_or_odd_l544_54482

-- Define the set of ball numbers
def ballNumbers : Finset Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define a function to check if a number is prime
def isPrime (n : Nat) : Bool :=
  if n ≤ 1 then false
  else
    !(List.range (n - 2)).any (fun m => n % (m + 2) = 0)

-- Define a function to check if a number is odd
def isOdd (n : Nat) : Bool := n % 2 ≠ 0

-- Define the set of numbers that are either prime or odd
def primeOrOdd : Finset Nat :=
  ballNumbers.filter (fun n => isPrime n || isOdd n)

-- Theorem statement
theorem probability_prime_or_odd :
  (primeOrOdd.card : ℚ) / ballNumbers.card = 5 / 7 := by
  -- Evaluate primeOrOdd
  have h1 : primeOrOdd = {1, 2, 3, 5, 7} := by sorry
  -- Calculate cardinality
  have h2 : primeOrOdd.card = 5 := by sorry
  have h3 : ballNumbers.card = 7 := by sorry
  -- Prove the equality
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_or_odd_l544_54482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisor_of_12_l544_54415

/-- The set of divisors of 12 -/
def divisors_of_12 : Finset Nat := {1, 2, 3, 4, 6, 12}

/-- A fair 12-sided die -/
def die_sides : Nat := 12

/-- Probability of an event on a fair die -/
def probability (favorable_outcomes : Nat) (total_outcomes : Nat) : ℚ :=
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_divisor_of_12 :
  probability (Finset.card divisors_of_12) die_sides = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisor_of_12_l544_54415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l544_54431

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.pi / 4 - 2 * x)

theorem f_monotone_decreasing :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi + Real.pi / 8) (k * Real.pi + 5 * Real.pi / 8)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l544_54431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_percent_score_l544_54498

theorem average_percent_score (scores : List ℤ) (students : List ℕ) : 
  scores = [100, 95, 90, 80, 70, 60, 50, 40] →
  students = [5, 12, 20, 30, 20, 8, 4, 1] →
  (List.sum (List.zipWith (· * ·) 
    [17, 0, 20, 30, 20, 8, 4, 1] 
    scores)) / 100 = 8020 / 100 := by
  sorry

#eval (8020 : ℚ) / 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_percent_score_l544_54498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_arrangements_eq_48_l544_54446

/-- The number of arrangements for 2 boys and 3 girls in a line,
    with boy A not at either end and exactly two girls standing together. -/
def line_arrangements : ℕ :=
  let total_students : ℕ := 5
  let num_boys : ℕ := 2
  let num_girls : ℕ := 3
  let boy_A_arrangements : ℕ := num_boys - 1  -- Boy A can't be at either end
  let girl_pair_choices : ℕ := Nat.choose num_girls 2
  let girl_pair_permutations : ℕ := 2  -- Two girls can be arranged in 2 ways
  let gaps : ℕ := num_boys + 1  -- Gaps created by boys
  let element_arrangements : ℕ := gaps * (gaps - 1)  -- Arranging 2 elements in 3 gaps
  let total_arrangements : ℕ := boy_A_arrangements * girl_pair_choices * girl_pair_permutations * element_arrangements
  let end_arrangements : ℕ := 2 * girl_pair_choices * girl_pair_permutations * 2  -- Boy A at either end
  total_arrangements - end_arrangements

theorem line_arrangements_eq_48 : line_arrangements = 48 := by
  sorry

#eval line_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_arrangements_eq_48_l544_54446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_is_one_two_zeros_condition_l544_54406

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * (x + 2)

-- Part 1: Monotonicity when a = 1
theorem monotonicity_when_a_is_one :
  ∀ x y : ℝ,
    (x < 0 ∧ y < 0 ∧ x < y → f 1 x > f 1 y) ∧
    (x > 0 ∧ y > 0 ∧ x < y → f 1 x < f 1 y) :=
by sorry

-- Part 2: Condition for two zeros
theorem two_zeros_condition :
  ∀ a : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ a > Real.exp (-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_is_one_two_zeros_condition_l544_54406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercept_line_equations_l544_54451

/-- A straight line with equal intercepts on both coordinate axes -/
structure EqualInterceptLine where
  slope : ℝ
  intercept : ℝ
  equal_intercepts : intercept = 0 ∨ slope = -1

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

/-- The theorem stating the possible equations of the line -/
theorem equal_intercept_line_equations :
  ∀ (l : EqualInterceptLine),
  ∃ (d : ℝ),
  (distance_point_to_line 1 3 l.slope (-1) 0 = d ∧ l.intercept = 0) ∨
  (distance_point_to_line 1 3 1 1 (-2) = d ∧ l.slope = -1 ∧ l.intercept = 2) ∨
  (distance_point_to_line 1 3 1 1 (-6) = d ∧ l.slope = -1 ∧ l.intercept = 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercept_line_equations_l544_54451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_properties_l544_54439

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 4*x else x^2 + 4*x

-- Define the function g
noncomputable def g (t : ℝ) : ℝ :=
  ⨆ x ∈ Set.Icc (t - 1) (t + 1), f x

-- State the theorem
theorem even_function_properties :
  (∀ x, f (-x) = f x) ∧  -- f is even
  (∀ x, x ≥ 0 → f x = x^2 - 4*x) ∧  -- definition of f for x ≥ 0
  f (-2) = -4 ∧  -- property 1
  (∀ x, x < 0 → f x = x^2 + 4*x) ∧  -- property 2
  (∃ t > 1, ∀ t' > 1, g t ≤ g t' ∧ g t = -3)  -- property 3
  :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_properties_l544_54439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_to_one_is_six_l544_54456

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if two positions are adjacent in the grid --/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Check if all consecutive numbers are adjacent in the grid --/
def consecutive_adjacent (g : Grid) : Prop :=
  ∀ n : Fin 8, ∃ p1 p2 : Fin 3 × Fin 3,
    g p1.1 p1.2 = n ∧ g p2.1 p2.2 = n.succ ∧ adjacent p1 p2

/-- Sum of corner numbers in the grid --/
def corner_sum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- Check if a number is in the center of the grid --/
def is_center (g : Grid) (n : Fin 9) : Prop :=
  g 1 1 = n

/-- Check if two numbers are opposite across the center --/
def opposite_across_center (g : Grid) (n1 n2 : Fin 9) : Prop :=
  (g 0 1 = n1 ∧ g 2 1 = n2) ∨ (g 1 0 = n1 ∧ g 1 2 = n2)

theorem opposite_to_one_is_six (g : Grid)
  (h1 : consecutive_adjacent g)
  (h2 : corner_sum g = 24)
  (h3 : ∃! n : Fin 9, is_center g n) :
  ∃ n : Fin 9, opposite_across_center g 1 n ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_to_one_is_six_l544_54456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_three_l544_54473

/-- A function that returns true if a number contains the digit 3 at least once -/
def contains_three (n : Nat) : Bool :=
  let digits := n.digits 10
  digits.any (· = 3)

/-- The set of integers from 1 to 60 that contain the digit 3 at least once -/
def S : Set Nat :=
  {n | 1 ≤ n ∧ n ≤ 60 ∧ contains_three n}

theorem count_numbers_with_three : Finset.card (Finset.filter (λ n => contains_three n) (Finset.range 60)) = 15 := by
  sorry

#eval Finset.card (Finset.filter (λ n => contains_three n) (Finset.range 60))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_numbers_with_three_l544_54473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_plane_through_parallel_lines_l544_54403

/-- Two lines in a 3D space are parallel -/
def parallel_lines (l1 l2 : Set (Fin 3 → ℝ)) : Prop := sorry

/-- A plane in 3D space -/
def plane (p : Set (Fin 3 → ℝ)) : Prop := sorry

/-- There is exactly one plane passing through two parallel lines -/
theorem unique_plane_through_parallel_lines (l1 l2 : Set (Fin 3 → ℝ)) :
  parallel_lines l1 l2 →
  ∃! p, plane p ∧ l1 ⊆ p ∧ l2 ⊆ p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_plane_through_parallel_lines_l544_54403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_equality_l544_54479

-- Define a structure for a monic quadratic trinomial
structure MonicQuadraticTrinomial where
  p : ℝ
  q : ℝ

-- Define a function to calculate the discriminant of a monic quadratic trinomial
def discriminant (t : MonicQuadraticTrinomial) : ℝ := t.p ^ 2 - 4 * t.q

-- Define a function to calculate the roots of a monic quadratic trinomial
noncomputable def roots (t : MonicQuadraticTrinomial) : ℝ × ℝ :=
  let d := discriminant t
  ((-t.p + Real.sqrt d) / 2, (-t.p - Real.sqrt d) / 2)

-- State the theorem
theorem root_sum_equality (t1 t2 t3 : MonicQuadraticTrinomial) 
  (h1 : discriminant t1 = 1)
  (h2 : discriminant t2 = 4)
  (h3 : discriminant t3 = 9) :
  ∃ (r1 r2 r3 : ℝ), 
    (r1 = (roots t1).1 ∨ r1 = (roots t1).2) ∧
    (r2 = (roots t2).1 ∨ r2 = (roots t2).2) ∧
    (r3 = (roots t3).1 ∨ r3 = (roots t3).2) ∧
    r1 + r2 + r3 = 
      (roots t1).1 + (roots t1).2 + 
      (roots t2).1 + (roots t2).2 + 
      (roots t3).1 + (roots t3).2 - 
      (r1 + r2 + r3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_equality_l544_54479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l544_54499

theorem tan_alpha_value (α : Real) 
  (h1 : Real.sin α = 3/5) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.tan α = -3/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l544_54499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_shifted_symmetric_sin_l544_54477

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem min_value_of_shifted_symmetric_sin 
  (φ : ℝ) 
  (h1 : |φ| < π/2) 
  (h2 : ∀ x, f (x + π/6) φ = f (-x - π/6) φ) 
  : ∃ x₀ ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f x₀ φ ≤ f x φ ∧ f x₀ φ = -Real.sqrt 3 / 2 := by
  sorry

#check min_value_of_shifted_symmetric_sin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_shifted_symmetric_sin_l544_54477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_circle_equations_l544_54474

-- Define the circle P
def circle_P (a b R : ℝ) : Prop :=
  R^2 - b^2 = 2 ∧ R^2 - a^2 = 3

-- Define the distance from a point to the line y = x
noncomputable def distance_to_diagonal (a b : ℝ) : ℝ :=
  |b - a| / Real.sqrt 2

-- Theorem 1: Trajectory equation
theorem trajectory_equation (a b : ℝ) :
  circle_P a b (Real.sqrt 3) → b^2 - a^2 = 1 := by sorry

-- Theorem 2: Circle equations
theorem circle_equations (a b R : ℝ) :
  circle_P a b R ∧ distance_to_diagonal a b = Real.sqrt 2 / 2 →
  ((a = 0 ∧ b = 1) ∨ (a = 0 ∧ b = -1)) ∧ R = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_circle_equations_l544_54474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l544_54459

theorem triangle_third_side_length
  (a b c : ℕ)  -- a, b, c are natural numbers representing side lengths
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)  -- triangle inequality
  (side_difference : a - b = 7 ∨ b - a = 7)  -- difference between two sides is 7
  (odd_perimeter : ∃ k : ℕ, a + b + c = 2 * k + 1)  -- perimeter is odd
  : c = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_length_l544_54459
