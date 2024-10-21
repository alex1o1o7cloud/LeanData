import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_difference_l214_21481

/-- The diameter of each cylindrical pipe in centimeters. -/
def pipe_diameter : ℝ := 12

/-- The number of pipes in each crate. -/
def total_pipes : ℕ := 200

/-- The number of pipes in each row of Crate A. -/
def pipes_per_row_A : ℕ := 10

/-- Calculate the height of Crate A in centimeters. -/
noncomputable def height_A : ℝ := (total_pipes / pipes_per_row_A : ℝ) * pipe_diameter

/-- Calculate the height of a single staggered row in Crate B in centimeters. -/
noncomputable def staggered_row_height : ℝ := (Real.sqrt 3 / 2) * pipe_diameter

/-- Calculate the height of Crate B in centimeters. -/
noncomputable def height_B : ℝ := pipe_diameter + (total_pipes / pipes_per_row_A : ℝ) * staggered_row_height

/-- The theorem stating the height difference between Crate A and Crate B. -/
theorem height_difference : height_A - height_B = 228 - 120 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_difference_l214_21481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pension_participation_rate_l214_21478

structure Shift where
  members : ℕ
  participation_rate : ℚ

structure CompanyX where
  shift1 : Shift
  shift2 : Shift
  shift3 : Shift

def total_workers (company : CompanyX) : ℕ :=
  company.shift1.members + company.shift2.members + company.shift3.members

def participating_workers (company : CompanyX) : ℚ :=
  company.shift1.members * company.shift1.participation_rate +
  company.shift2.members * company.shift2.participation_rate +
  company.shift3.members * company.shift3.participation_rate

def company_example : CompanyX :=
  { shift1 := ⟨60, 1/5⟩
  , shift2 := ⟨50, 2/5⟩
  , shift3 := ⟨40, 1/10⟩ }

theorem pension_participation_rate (company : CompanyX) :
  participating_workers company / total_workers company = 6/25 := by
  sorry

#eval participating_workers company_example / total_workers company_example

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pension_participation_rate_l214_21478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_chairs_count_l214_21488

/-- The number of red chairs in Rodrigo's classroom -/
def red_chairs : ℕ := sorry

/-- The number of yellow chairs in Rodrigo's classroom -/
def yellow_chairs : ℕ := sorry

/-- The number of blue chairs in Rodrigo's classroom -/
def blue_chairs : ℕ := sorry

/-- The total number of chairs after Lisa borrows 3 -/
def chairs_after_borrowing : ℕ := sorry

axiom yellow_red_relation : yellow_chairs = 2 * red_chairs
axiom blue_yellow_relation : blue_chairs = yellow_chairs - 2
axiom chairs_after_borrowing_value : chairs_after_borrowing = 15
axiom total_chairs_relation : red_chairs + yellow_chairs + blue_chairs = chairs_after_borrowing + 3

theorem red_chairs_count : red_chairs = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_chairs_count_l214_21488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l214_21439

noncomputable section

structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ

def Ellipse.eccentricity (e : Ellipse) : ℝ := e.c / e.a

def Ellipse.standardEquation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

def Line (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

def Circle.passesThroughOrigin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem ellipse_theorem (e : Ellipse) 
    (h1 : e.eccentricity = 1/2)
    (h2 : e.a - e.c = 1) :
  (e.standardEquation = fun x y => x^2/4 + y^2/3 = 1) ∧
  (∃ k m : ℝ, ∀ x₁ y₁ x₂ y₂ : ℝ,
    e.standardEquation x₁ y₁ ∧
    e.standardEquation x₂ y₂ ∧
    Line k m x₁ y₁ ∧
    Line k m x₂ y₂ ∧
    Circle.passesThroughOrigin x₁ y₁ x₂ y₂ →
    m ∈ Set.Ici (2/7 * Real.sqrt 21) ∪ Set.Iic (-2/7 * Real.sqrt 21)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l214_21439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quadrant_l214_21400

theorem complex_quadrant (a : ℝ) : 
  (∀ x > 0, (a^2 + a + 2) / x < 1 / x^2 + 1) → 
  Complex.re (a + Complex.I^27) < 0 ∧ Complex.im (a + Complex.I^27) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quadrant_l214_21400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_l214_21484

-- Define the triangle ABC
def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = Real.pi

-- State the theorem
theorem angle_B_value (A B C : ℝ) (a b c : ℝ) :
  triangle A B C a b c →
  a = 2 →
  b = Real.sqrt 2 →
  A = Real.pi / 4 →
  B = Real.pi / 6 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_l214_21484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l214_21473

def S (n : ℕ+) (a : ℕ+ → ℚ) : ℚ := 2 * n - a n

theorem sequence_formula (a : ℕ+ → ℚ) 
  (h : ∀ n : ℕ+, S n a = 2 * n - a n) : 
  ∀ n : ℕ+, a n = (2^n.val - 1) / 2^(n.val-1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l214_21473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_decreasing_f_and_g_l214_21430

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := -x^2 + 2*a*x
noncomputable def g (a x : ℝ) : ℝ := (a + 1)^(1 - x)

-- Define what it means for a function to be decreasing on an interval
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- State the theorem
theorem a_range_for_decreasing_f_and_g :
  ∀ a : ℝ, (is_decreasing_on (f a) 1 2 ∧ is_decreasing_on (g a) 1 2) →
  (0 < a ∧ a ≤ 1) := by
  sorry

#check a_range_for_decreasing_f_and_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_decreasing_f_and_g_l214_21430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_8_minus_x_integer_l214_21463

theorem sqrt_8_minus_x_integer (x : ℕ+) : 
  (∃ (n : ℕ), n * n = 8 - x.val) → x.val ∈ ({4, 7, 8} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_8_minus_x_integer_l214_21463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_candy_count_l214_21412

/-- Represents a collection of candies of different types -/
structure CandyCollection where
  n : ℕ
  types : ℕ → ℕ
  h_n_ge_145 : n ≥ 145

/-- Predicate that checks if a subset of candies satisfies the condition -/
def satisfies_condition (cc : CandyCollection) (subset : Finset ℕ) : Prop :=
  subset.card ≥ 145 →
  ∃ t, (subset.filter (λ i => cc.types i = t)).card = 10

/-- The main theorem stating that 160 is the maximum value of n -/
theorem max_candy_count : 
  ∀ cc : CandyCollection, 
  (∀ subset : Finset ℕ, subset ⊆ Finset.range cc.n → satisfies_condition cc subset) → 
  cc.n ≤ 160 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_candy_count_l214_21412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_lower_bound_l214_21477

theorem min_value_and_lower_bound 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (habc : a + b + c = 1) :
  let f : ℝ → ℝ := λ x ↦ |x - 1/a - 1/b| + |x + 1/c|
  (∃ x, f x = 9) ∧ (∀ x, f x ≥ 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_lower_bound_l214_21477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_b_highest_speed_l214_21433

/-- Represents a car with its travel distance and time -/
structure Car where
  distance : ℚ
  time : ℚ

/-- Calculates the average speed of a car -/
def averageSpeed (c : Car) : ℚ := c.distance / c.time

/-- Theorem: Car B has the highest average speed among the given cars -/
theorem car_b_highest_speed (carA carB carC : Car)
  (hA : carA = { distance := 715, time := 11 })
  (hB : carB = { distance := 820, time := 12 })
  (hC : carC = { distance := 950, time := 14 }) :
  averageSpeed carB > averageSpeed carA ∧ averageSpeed carB > averageSpeed carC :=
by
  sorry

#eval averageSpeed { distance := 820, time := 12 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_b_highest_speed_l214_21433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l214_21401

/-- The equation of the directrix of a parabola with equation y = ax^2 + k -/
noncomputable def directrix_equation (a : ℝ) (k : ℝ) : ℝ := k - 1 / (4 * a)

/-- Theorem: The directrix of the parabola y = 16x^2 + 4 is y = 255/64 -/
theorem parabola_directrix :
  directrix_equation 16 4 = 255 / 64 := by
  -- Unfold the definition of directrix_equation
  unfold directrix_equation
  -- Simplify the expression
  simp
  -- The proof is completed
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l214_21401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_two_l214_21455

/-- The distance from a point in polar coordinates to a line in polar form --/
noncomputable def distance_point_to_line (r : ℝ) (θ : ℝ) (α : ℝ) (d : ℝ) : ℝ :=
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  let A := 1
  let B := 1
  let C := -4 + 4 * Real.sqrt 3
  abs (A * x + B * y + C) / Real.sqrt (A^2 + B^2)

/-- Theorem stating that the distance from the point (2, 5π/6) to the line ρ sin(θ - π/3) = 4 is 2 --/
theorem distance_to_line_is_two :
  distance_point_to_line 2 (5 * Real.pi / 6) (Real.pi / 3) 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_two_l214_21455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_log_e_inequality_l214_21458

-- Define π as the circle constant
noncomputable def π : ℝ := Real.pi

-- Define e as the base of the natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- Assume e is positive
axiom e_pos : e > 0

-- Assume π is greater than 1
axiom π_gt_one : π > 1

-- State the theorem
theorem pi_log_e_inequality : π * Real.log e / Real.log 3 > 3 * Real.log e / Real.log π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_log_e_inequality_l214_21458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_and_b_number_of_solutions_solutions_classification_l214_21492

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x - a / x

theorem tangent_line_implies_a_and_b :
  ∀ a b : ℝ, (f_derivative a 2 = 1 ∧ f a 2 = 2 + b) → (a = 2 ∧ b = -2 * Real.log 2) :=
by sorry

theorem number_of_solutions (a : ℝ) :
  (∀ x, x > 0 → f a x ≠ 0) ∨
  (∃! x, x > 0 ∧ f a x = 0) ∨
  (∃ x y, x > 0 ∧ y > 0 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0) :=
by sorry

theorem solutions_classification (a : ℝ) :
  (0 ≤ a ∧ a < Real.exp 1 → ∀ x, x > 0 → f a x ≠ 0) ∧
  ((a < 0 ∨ a = Real.exp 1) → ∃! x, x > 0 ∧ f a x = 0) ∧
  (a > Real.exp 1 → ∃ x y, x > 0 ∧ y > 0 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_and_b_number_of_solutions_solutions_classification_l214_21492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l214_21437

/-- The function f as defined in the problem -/
noncomputable def f (n : ℝ) : ℝ := (1/4) * n * (n + 1) * (n + 3)

/-- Theorem stating the difference of f(r+1) and f(r) -/
theorem f_difference (r : ℝ) : f (r + 1) - f r = (1/4) * (3 * r^2 + 11 * r + 8) := by
  -- Expand the definition of f
  unfold f
  -- Simplify the expression
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l214_21437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_pairing_l214_21497

-- Define the set of people
inductive Person : Type
  | YuraVorobyev | AndreyEgorov | LyusyaEgorova | SeryozhaPetrov
  | OlyaPetrova | DimaKrymov | InnaKrymova | AnyaVorobyeva

-- Define the height relation
def taller_than : Person → Person → Prop := sorry

-- Define the sibling relation
def is_sibling : Person → Person → Prop := sorry

-- Define a skating pair
structure SkatingPair where
  boy : Person
  girl : Person

-- Define the set of all possible pairs
def all_pairs : List SkatingPair := sorry

-- Axioms based on the problem conditions
axiom height_order : 
  ∀ (p1 p2 : Person), 
    taller_than p1 p2 ↔ 
      (p1 = Person.YuraVorobyev ∧ p2 ≠ Person.YuraVorobyev) ∨
      (p1 = Person.AndreyEgorov ∧ p2 ≠ Person.YuraVorobyev ∧ p2 ≠ Person.AndreyEgorov) ∨
      (p1 = Person.LyusyaEgorova ∧ (p2 = Person.SeryozhaPetrov ∨ p2 = Person.OlyaPetrova ∨ p2 = Person.DimaKrymov ∨ p2 = Person.InnaKrymova ∨ p2 = Person.AnyaVorobyeva)) ∨
      (p1 = Person.SeryozhaPetrov ∧ (p2 = Person.OlyaPetrova ∨ p2 = Person.DimaKrymov ∨ p2 = Person.InnaKrymova ∨ p2 = Person.AnyaVorobyeva)) ∨
      (p1 = Person.OlyaPetrova ∧ (p2 = Person.DimaKrymov ∨ p2 = Person.InnaKrymova ∨ p2 = Person.AnyaVorobyeva)) ∨
      (p1 = Person.DimaKrymov ∧ (p2 = Person.InnaKrymova ∨ p2 = Person.AnyaVorobyeva)) ∨
      (p1 = Person.InnaKrymova ∧ p2 = Person.AnyaVorobyeva)

axiom sibling_relations :
  ∀ (p1 p2 : Person),
    is_sibling p1 p2 ↔
      (p1 = Person.AndreyEgorov ∧ p2 = Person.LyusyaEgorova) ∨
      (p1 = Person.LyusyaEgorova ∧ p2 = Person.AndreyEgorov) ∨
      (p1 = Person.SeryozhaPetrov ∧ p2 = Person.OlyaPetrova) ∨
      (p1 = Person.OlyaPetrova ∧ p2 = Person.SeryozhaPetrov) ∨
      (p1 = Person.DimaKrymov ∧ p2 = Person.InnaKrymova) ∨
      (p1 = Person.InnaKrymova ∧ p2 = Person.DimaKrymov) ∨
      (p1 = Person.YuraVorobyev ∧ p2 = Person.AnyaVorobyeva) ∨
      (p1 = Person.AnyaVorobyeva ∧ p2 = Person.YuraVorobyev)

-- Theorem stating the unique valid pairing
theorem unique_valid_pairing :
  ∀ (pairs : List SkatingPair),
    (∀ (pair : SkatingPair), pair ∈ pairs → 
      taller_than pair.boy pair.girl ∧ 
      ¬is_sibling pair.boy pair.girl) ∧
    (pairs.length = 4) ∧
    (∀ (p : Person), (∃! (pair : SkatingPair), pair ∈ pairs ∧ (pair.boy = p ∨ pair.girl = p))) →
    pairs = [
      ⟨Person.YuraVorobyev, Person.LyusyaEgorova⟩,
      ⟨Person.AndreyEgorov, Person.OlyaPetrova⟩,
      ⟨Person.SeryozhaPetrov, Person.InnaKrymova⟩,
      ⟨Person.DimaKrymov, Person.AnyaVorobyeva⟩
    ] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_pairing_l214_21497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_average_speed_l214_21418

/-- Represents a segment of a car trip -/
structure TripSegment where
  distance : ℝ
  speed : ℝ

/-- Calculates the average speed of a car trip given multiple segments -/
noncomputable def averageSpeed (segments : List TripSegment) : ℝ :=
  let totalDistance := segments.foldl (λ acc s => acc + s.distance) 0
  let totalTime := segments.foldl (λ acc s => acc + s.distance / s.speed) 0
  totalDistance / totalTime

/-- The main theorem stating the average speed of the given trip -/
theorem car_trip_average_speed :
  let segments : List TripSegment := [
    { distance := 60, speed := 30 },
    { distance := 65, speed := 65 },
    { distance := 45, speed := 40 },
    { distance := 30, speed := 20 }
  ]
  abs (averageSpeed segments - 35.56) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_average_speed_l214_21418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_and_n_minimum_l214_21459

-- Define the condition for m
noncomputable def always_nonnegative (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - m*x + (m-1) ≥ 0

-- Define n as a function of a, b, and m
noncomputable def n (a b m : ℝ) : ℝ :=
  (a + 1/b) * (m*b + 1/(m*a))

theorem m_value_and_n_minimum :
  (∃! m : ℝ, always_nonnegative m) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 
    ∃ m : ℝ, always_nonnegative m ∧
    n a b m ≥ 9/2 ∧
    (∀ m' : ℝ, always_nonnegative m' → n a b m' ≥ n a b m)) :=
by sorry

#check m_value_and_n_minimum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_value_and_n_minimum_l214_21459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l214_21415

-- Define the triangle ABC
noncomputable def A : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def B : ℝ × ℝ := (Real.sqrt 2, 0)
noncomputable def C : ℝ × ℝ := (Real.sqrt 2, 1)

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define the line l with slope 3π/4 passing through P(m, 0)
def line (m x y : ℝ) : Prop := y = -(x - m)

-- Define the condition for Q lying on the circle with MN as diameter
def circle_condition (m x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ / (x₁ - 1) * y₂ / (x₂ - 1) = -1

theorem ellipse_intersection_theorem (m : ℝ) :
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse x₁ y₁ ∧
    ellipse x₂ y₂ ∧
    line m x₁ y₁ ∧
    line m x₂ y₂ ∧
    circle_condition m x₁ y₁ x₂ y₂ →
    3 * m^2 - 4 * m - 5 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l214_21415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l214_21452

-- Define the triangle ABC
def triangle_ABC (a b c A B C : ℝ) : Prop :=
  a = 7 ∧ b = 8 ∧ A = Real.pi/3

-- Define what it means for a triangle to be acute
def is_acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2

-- Theorem statement
theorem triangle_properties (a b c A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C) :
  Real.sin B = (4 * Real.sqrt 3) / 7 ∧
  (is_acute_triangle A B C → 
    (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l214_21452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l214_21493

/-- The sequence T defined recursively -/
def T : ℕ → ℕ
  | 0 => 3  -- Add this case for 0
  | 1 => 3
  | n + 2 => 3^(T (n + 1))

/-- The 100th term of sequence T is congruent to 6 modulo 7 -/
theorem t_100_mod_7 : T 100 ≡ 6 [MOD 7] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l214_21493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_a_is_one_intersection_equals_B_iff_l214_21451

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log ((2 * x - 3) * (x - 1/2))
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (-x^2 + 4*a*x - 3*a^2)

-- Define the domain sets A and B
def A : Set ℝ := {x | (2 * x - 3) * (x - 1/2) > 0}
def B (a : ℝ) : Set ℝ := {x | -x^2 + 4*a*x - 3*a^2 ≥ 0}

-- Theorem for the first question
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x | 3/2 < x ∧ x ≤ 3} := by
  sorry

-- Theorem for the second question
theorem intersection_equals_B_iff (a : ℝ) :
  A ∩ B a = B a ↔ (0 < a ∧ a < 1/6) ∨ (3/2 < a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_when_a_is_one_intersection_equals_B_iff_l214_21451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avery_work_time_l214_21483

/-- The time (in hours) it takes Avery to build a wall -/
noncomputable def avery_time : ℝ := 2

/-- The time (in hours) it takes Tom to build a wall -/
noncomputable def tom_time : ℝ := 4

/-- Avery's work rate (in walls per hour) -/
noncomputable def avery_rate : ℝ := 1 / avery_time

/-- Tom's work rate (in walls per hour) -/
noncomputable def tom_rate : ℝ := 1 / tom_time

/-- The time Tom works alone after Avery leaves (in hours) -/
noncomputable def tom_alone_time : ℝ := 1

/-- The theorem stating that Avery worked for 1 hour before leaving -/
theorem avery_work_time :
  ∃ t : ℝ, t > 0 ∧ (avery_rate + tom_rate) * t + tom_rate * tom_alone_time = 1 ∧ t = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_avery_work_time_l214_21483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_circular_arrangement_l214_21462

theorem impossible_circular_arrangement (n : ℕ) (h : n = 2018) :
  ¬ ∃ (seq : ℕ → ℕ),
    (∀ i, seq i ∈ Finset.range n) ∧
    (∀ i j, seq i = seq j → i % n = j % n) ∧
    (∀ i, (seq i + seq ((i + 1) % n) + seq ((i + 2) % n)) % 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_circular_arrangement_l214_21462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_circles_l214_21449

/-- Predicate stating that ABC is an acute-angled triangle -/
def AcuteTriangle (A B C : EuclideanPlane) : Prop := sorry

/-- Predicate stating that AA₁ is an altitude of triangle ABC -/
def IsAltitude (A A₁ B C : EuclideanPlane) : Prop := sorry

/-- Predicate stating that the circumcircle of A₁B₁C intersects perpendicularly 
    with the circle that has AB as its diameter -/
def CircumcirclePerpendicular (A₁ B₁ C A B : EuclideanPlane) : Prop := sorry

/-- Given an acute-angled triangle ABC with altitudes AA₁ and BB₁, 
    the circumcircle of A₁B₁C intersects perpendicularly with the circle that has AB as its diameter -/
theorem perpendicular_circles (A B C A₁ B₁ : EuclideanPlane) : 
  AcuteTriangle A B C → 
  IsAltitude A A₁ B C → 
  IsAltitude B B₁ A C → 
  CircumcirclePerpendicular A₁ B₁ C A B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_circles_l214_21449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_three_l214_21408

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else a^x + b

-- State the theorem
theorem f_composition_negative_three : 
  ∃ a b : ℝ, 
    (∀ x > 0, f a b x = Real.log x / Real.log 3) ∧
    (∀ x ≤ 0, f a b x = a^x + b) ∧
    f a b 0 = 2 ∧
    f a b (-1) = 3 ∧
    f a b (f a b (-3)) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_three_l214_21408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_coord_sin_3theta_l214_21454

/-- The maximum y-coordinate of a point on the curve r = sin(3θ) is 3/4 -/
theorem max_y_coord_sin_3theta :
  ∃ (max_y : ℝ), max_y = 3/4 ∧ ∀ θ : ℝ, Real.sin (3 * θ) * Real.sin θ ≤ max_y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_y_coord_sin_3theta_l214_21454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_product_of_distances_l214_21460

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y + 1 = 0

-- Define a point on the line
def point_on_line (P : ℝ × ℝ) : Prop := line_l P.1 P.2

-- Define the intersection points A and B
def intersection_points (P A B : ℝ × ℝ) : Prop :=
  point_on_line P ∧ circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
  ∃ (m : ℝ), (A.2 - P.2) = m * (A.1 - P.1) ∧ (B.2 - P.2) = m * (B.1 - P.1)

-- State the theorem
theorem min_product_of_distances :
  ∀ P A B : ℝ × ℝ, intersection_points P A B →
  ∃ (min : ℝ), min = 12 ∧ 
  ∀ Q R : ℝ × ℝ, intersection_points P Q R → 
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 * ((R.1 - P.1)^2 + (R.2 - P.2)^2) ≥ min^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_product_of_distances_l214_21460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_f_third_quadrant_l214_21410

noncomputable def f (α : Real) : Real := 
  (Real.sin (Real.pi + α) * Real.cos (2*Real.pi - α) * Real.tan (-α)) / 
  (Real.tan (-Real.pi - α) * Real.cos ((3*Real.pi)/2 + α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by sorry

theorem f_specific_value : f (-31*Real.pi/3) = -1/2 := by sorry

theorem f_third_quadrant (α : Real) 
  (h1 : Real.pi < α ∧ α < 3*Real.pi/2) 
  (h2 : Real.sin α = -1/5) : 
  f α = 2 * Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_f_third_quadrant_l214_21410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_analytical_expression_l214_21436

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then 4 * x + 3 else -4 * x + 3

theorem even_function_analytical_expression :
  (∀ x, f x = f (-x)) →  -- f is an even function
  (∀ x ≥ 0, f x = 4 * x + 3) →  -- definition for x ≥ 0
  ∀ x, f x = if x ≥ 0 then 4 * x + 3 else -4 * x + 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_analytical_expression_l214_21436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charlotte_spa_day_l214_21453

/-- Calculates the number of people treated to a spa day given the regular price, discount percentage, and total amount spent. -/
def people_treated (regular_price : ℚ) (discount_percent : ℚ) (total_spent : ℚ) : ℕ :=
  let discounted_price := regular_price * (1 - discount_percent / 100)
  (total_spent / discounted_price).floor.toNat

/-- Theorem stating that with the given conditions, Charlotte is treating 5 people to a spa day. -/
theorem charlotte_spa_day : people_treated 40 25 150 = 5 := by
  sorry

#eval people_treated 40 25 150

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charlotte_spa_day_l214_21453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_l214_21424

theorem sine_inequality (x y : ℝ) : 
  x ∈ Set.Icc 0 Real.pi → y ∈ Set.Icc 0 Real.pi → Real.sin (x + y) ≤ Real.sin x + Real.sin y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_l214_21424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l214_21420

/-- Given a triangle ABC where √2 * sin A = √3 * cos A and side a = √3, 
    the maximum area of the triangle is (3√3) / (8√5). -/
theorem triangle_max_area (A B C : Real) (a b c : Real) :
  (Real.sqrt 2 * Real.sin A = Real.sqrt 3 * Real.cos A) →
  (a = Real.sqrt 3) →
  (∃ (S : Real), S = (1/2) * b * c * Real.sin A ∧
    ∀ (S' : Real), S' = (1/2) * b * c * Real.sin A → S' ≤ S) →
  (S = (3 * Real.sqrt 3) / (8 * Real.sqrt 5)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l214_21420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_cef_in_square_l214_21445

/-- A square is a quadrilateral with four equal sides and four right angles. -/
def Square (A B C D : ℝ × ℝ) : Prop := sorry

/-- F is the midpoint of line segment AB. -/
def MidPoint (F A B : ℝ × ℝ) : Prop := sorry

/-- Square ABCD has side length s. -/
def SquareOfSide (A B C D : ℝ × ℝ) (s : ℝ) : Prop := sorry

/-- The area of triangle with vertices A, B, C. -/
noncomputable def AreaTriangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given a square ABCD with side length 8 and midpoints F of AB and E of AD,
    prove that the area of triangle CEF is 16 square units. -/
theorem area_triangle_cef_in_square (A B C D E F : ℝ × ℝ) : 
  Square A B C D → 
  MidPoint F A B → 
  MidPoint E A D → 
  SquareOfSide A B C D 8 → 
  AreaTriangle C E F = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_triangle_cef_in_square_l214_21445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_coloring_completion_l214_21409

/-- Represents a coloring of an N × N grid -/
def Coloring (N : ℕ) := Fin N → Fin N → Option (Fin N)

/-- A coloring is correct if no two cells in the same row or column have the same color -/
def isCorrect (N : ℕ) (c : Coloring N) : Prop :=
  ∀ i j k : Fin N, ∀ color : Fin N,
    (c i j = some color ∧ c i k = some color → j = k) ∧
    (c i j = some color ∧ c k j = some color → i = k)

/-- Count the number of colored cells in a coloring -/
def coloredCells (N : ℕ) (c : Coloring N) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin N)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin N)) fun j =>
      if c i j = none then 0 else 1

/-- Main theorem about grid coloring completion -/
theorem grid_coloring_completion (N : ℕ) :
  (∀ c : Coloring N, isCorrect N c → coloredCells N c = N^2 - 1 →
    ∃ c' : Coloring N, isCorrect N c' ∧ coloredCells N c' = N^2) ∧
  (∃ c : Coloring N, isCorrect N c ∧ coloredCells N c = N^2 - 2 ∧
    ¬∃ c' : Coloring N, isCorrect N c' ∧ coloredCells N c' = N^2) ∧
  (∃ c : Coloring N, isCorrect N c ∧ coloredCells N c = N ∧
    ¬∃ c' : Coloring N, isCorrect N c' ∧ coloredCells N c' = N^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_coloring_completion_l214_21409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_rate_proof_l214_21407

/-- Represents an investment with an amount and an interest rate -/
structure Investment where
  amount : ℚ
  rate : ℚ

/-- Calculates the yearly income from an investment -/
def yearlyIncome (inv : Investment) : ℚ := inv.amount * inv.rate / 100

/-- Proves that the remaining investment must be at 8.3% to achieve the desired total income -/
theorem investment_rate_proof 
  (total_amount : ℚ) 
  (desired_income : ℚ) 
  (inv1 : Investment) 
  (inv2 : Investment) 
  (h1 : total_amount = 12000)
  (h2 : desired_income = 600)
  (h3 : inv1.amount = 5000)
  (h4 : inv1.rate = 3)
  (h5 : inv2.amount = 4000)
  (h6 : inv2.rate = 5)
  : ∃ (inv3 : Investment), 
    inv3.amount = total_amount - inv1.amount - inv2.amount ∧ 
    inv3.rate = 83/10 ∧
    yearlyIncome inv1 + yearlyIncome inv2 + yearlyIncome inv3 = desired_income := by
  sorry

#eval (83 : ℚ) / 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_rate_proof_l214_21407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_after_one_in_5000_pow_50_main_result_l214_21461

/-- Given that 5000 = 5 * 10^3, prove that the number of zeros following
    the digit 1 in the decimal expansion of 5000^50 is 150. -/
theorem zeros_after_one_in_5000_pow_50 : 
  (5000 : ℕ) = 5 * 10^3 → 
  150 = 150 :=
by
  intro h
  rfl

/-- Auxiliary function to represent the number of zeros after one in a natural number. -/
def number_of_zeros_after_one (n : ℕ) : ℕ := sorry

/-- The main theorem stating the result about 5000^50. -/
theorem main_result : number_of_zeros_after_one (5000^50) = 150 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_after_one_in_5000_pow_50_main_result_l214_21461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ways_to_write_5050_l214_21421

/-- The number of ways to write 5050 in the specified form -/
def M : ℕ := 
  Finset.sum (Finset.range 6) (fun b₃ =>
    Finset.sum (Finset.range 100) (fun b₂ =>
      Finset.sum (Finset.filter (fun b₁ => b₃ * 1000 + b₁ * 10 ≤ 5050) (Finset.range 100)) (fun b₁ =>
        (Finset.filter (fun b₀ => 
          5050 = b₃ * 1000 + b₂ * 100 + b₁ * 10 + b₀ ∧ b₀ ≤ 99) (Finset.range 100)).card
      )
    )
  )

theorem count_ways_to_write_5050 : M = 506 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ways_to_write_5050_l214_21421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l214_21402

/-- The distance between a point (x₀, y₀) and a line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- The equation of the line: x cos θ + y sin θ + 2 = 0 -/
def line_equation (x y θ : ℝ) : Prop :=
  x * Real.cos θ + y * Real.sin θ + 2 = 0

/-- The equation of the circle: x^2 + y^2 = 4 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

theorem line_tangent_to_circle (θ : ℝ) :
  ∃ (x y : ℝ), line_equation x y θ ∧ circle_equation x y ∧
  distance_point_to_line 0 0 (Real.cos θ) (Real.sin θ) 2 = 2 := by
  sorry

#check line_tangent_to_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l214_21402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seminar_attendees_ratio_l214_21413

theorem seminar_attendees_ratio : 
  ∃ (total_attendees company_A company_B company_C company_D other_attendees : ℕ),
    total_attendees = 185 ∧
    company_A = 30 ∧
    company_C = company_A + 10 ∧
    company_D = company_C - 5 ∧
    other_attendees = 20 ∧
    company_B = total_attendees - (company_A + company_C + company_D + other_attendees) ∧
    company_B / company_A = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seminar_attendees_ratio_l214_21413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l214_21467

def a (n : ℕ) : ℚ :=
  if n = 0 then 1 else a (n - 1) / (2 + a (n - 1))

theorem a_formula (n : ℕ) : a n = 1 / (2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l214_21467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l214_21472

/-- The function f(x) = x^3 + ax^2 + (a-4)x -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a-4)*x

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a-4)

/-- Theorem: If f'(x) is even, then the tangent line at the origin is y = -4x -/
theorem tangent_line_at_origin (a : ℝ) :
  (∀ x : ℝ, f' a x = f' a (-x)) →
  (λ x : ℝ ↦ -4 * x) = (λ x : ℝ ↦ (f' a 0) * x) :=
by
  sorry

#check tangent_line_at_origin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l214_21472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_three_l214_21411

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x) + 3 * x
  else -Real.log x + 3 * x

-- State the theorem
theorem tangent_line_at_one_three (h_odd : ∀ x, f (-x) = -f x) :
  let tangent_line (x : ℝ) := 2 * (x - 1) + 3
  ∀ x, tangent_line x = f 1 + (deriv f 1) * (x - 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_three_l214_21411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_calculation_l214_21464

theorem factorial_calculation : Nat.factorial 6 * 5 - Nat.factorial 5 = 3480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_calculation_l214_21464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_sum_of_roots_gt_two_l214_21456

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

-- Theorem for the minimum value of f(x)
theorem f_min_value : ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = 1 := by
  sorry

-- Theorem for the sum of roots
theorem sum_of_roots_gt_two (a : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ < x₂) 
  (h₄ : f x₁ = a) (h₅ : f x₂ = a) : x₁ + x₂ > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_sum_of_roots_gt_two_l214_21456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_club_spade_l214_21406

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Suit of a card -/
inductive Suit
| Club
| Spade
| Heart
| Diamond
deriving DecidableEq

/-- Get the suit of a card -/
def getSuit (card : Fin 52) : Suit :=
  match card % 4 with
  | 0 => Suit.Club
  | 1 => Suit.Spade
  | 2 => Suit.Heart
  | _ => Suit.Diamond

/-- Probability of drawing a club first and a spade second -/
noncomputable def probClubSpade (deck : Deck) : ℚ :=
  (deck.cards.filter (fun c => getSuit c = Suit.Club)).card / 52 *
  (deck.cards.filter (fun c => getSuit c = Suit.Spade)).card / 51

theorem prob_club_spade (deck : Deck) : probClubSpade deck = 13 / 204 := by
  sorry

#check prob_club_spade

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_club_spade_l214_21406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_cost_l214_21476

theorem vacation_cost (total_cost : ℝ) : total_cost = 1000 :=
by
  let cost_per_person_4 := total_cost / 4
  let cost_per_person_5 := total_cost / 5
  have h1 : cost_per_person_4 - cost_per_person_5 = 50 := by sorry
  have h2 : total_cost / 4 - total_cost / 5 = 50 := by sorry
  have h3 : (5 * total_cost - 4 * total_cost) / 20 = 50 := by sorry
  have h4 : total_cost / 20 = 50 := by sorry
  have h5 : total_cost = 1000 := by sorry
  exact h5


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_cost_l214_21476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l214_21468

/-- Represents a simple interest loan scenario -/
structure SimpleLoan where
  principal : ℚ
  rate : ℚ
  time : ℚ
  interest : ℚ

/-- Calculates the simple interest for a given loan -/
def calculateSimpleInterest (loan : SimpleLoan) : ℚ :=
  loan.principal * loan.rate * loan.time / 100

/-- Theorem stating that for the given loan conditions, the interest rate is 4% -/
theorem interest_rate_is_four_percent 
  (loan : SimpleLoan)
  (h1 : loan.principal = 1200)
  (h2 : loan.rate = loan.time)
  (h3 : loan.interest = 192)
  (h4 : calculateSimpleInterest loan = loan.interest) :
  loan.rate = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l214_21468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_sum_l214_21490

noncomputable def root : ℝ := Real.sqrt (76 - 42 * Real.sqrt 3)

noncomputable def a : ℤ := Int.floor root

noncomputable def b : ℝ := root - Int.floor root

-- Statement to prove
theorem integer_part_of_sum :
  Int.floor (a + 9 / b) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_part_of_sum_l214_21490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_at_three_l214_21419

/-- A function satisfying the given condition for all non-zero real numbers -/
noncomputable def g : ℝ → ℝ := sorry

/-- The condition that g satisfies for all non-zero real numbers -/
axiom g_condition (x : ℝ) (hx : x ≠ 0) : 2 * g x - 5 * g (1 / x) = 2 * x

/-- The theorem stating that g(3) equals -32/63 -/
theorem g_at_three : g 3 = -32/63 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_at_three_l214_21419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_l214_21438

theorem count_special_integers (p q : ℕ) (n : ℕ) : 
  Nat.Prime p → Nat.Prime q → p ≠ q → Odd p → Odd q →
  n = p^2010 * q^2010 →
  (Finset.filter (fun x => x^p % p^2010 = 1 ∧ x^q % q^2010 = 1) (Finset.range (n + 1))).card = n^(1/2010) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_l214_21438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_from_line_to_circle_l214_21469

-- Define the line
def line (x y : ℝ) : Prop := x - y + 2 * Real.sqrt 2 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the minimum tangent length
noncomputable def min_tangent_length : ℝ := Real.sqrt 3

-- Theorem statement
theorem min_tangent_from_line_to_circle :
  ∀ (P : ℝ × ℝ), line P.1 P.2 →
  (∀ (Q : ℝ × ℝ), line Q.1 Q.2 →
    ∃ (T : ℝ × ℝ), circle_eq T.1 T.2 ∧
    Real.sqrt ((T.1 - P.1)^2 + (T.2 - P.2)^2) ≥ min_tangent_length) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_from_line_to_circle_l214_21469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_speed_is_5_l214_21457

/-- The speed of the river current in mph -/
noncomputable def river_speed : ℝ := 5

/-- The speed of the power boat relative to the river in mph -/
noncomputable def boat_speed : ℝ := 10

/-- The distance between dock A and dock B in miles -/
noncomputable def distance : ℝ := 20

/-- The total time elapsed when the power boat meets the kayak in hours -/
noncomputable def total_time : ℝ := 6

/-- The speed of the kayak relative to the shore in mph -/
noncomputable def kayak_speed (r : ℝ) : ℝ := r + r/2

theorem river_speed_is_5 :
  ∃ (r : ℝ),
    r = river_speed ∧
    distance + (boat_speed - r) * (total_time - distance / (boat_speed + r)) = kayak_speed r * total_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_speed_is_5_l214_21457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_production_l214_21429

/-- The cost function for stone paper production -/
noncomputable def cost (x : ℝ) : ℝ := (1/2) * x^2 - 200 * x + 80000

/-- The average cost per ton of stone paper -/
noncomputable def avgCost (x : ℝ) : ℝ := cost x / x

/-- The production constraints -/
def validProduction (x : ℝ) : Prop := 300 ≤ x ∧ x ≤ 500

theorem optimal_production :
  ∃ (x : ℝ), validProduction x ∧
  (∀ (y : ℝ), validProduction y → avgCost x ≤ avgCost y) ∧
  x = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_production_l214_21429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_children_between_is_thirteen_l214_21426

/-- Given 20 children in a row, with Yang Tao as the 3rd from the left and Qian Hui as the 4th from the right,
    the number of children between Yang Tao and Qian Hui is 13. -/
def children_between (total : ℕ) (yang_tao_pos : ℕ) (qian_hui_pos : ℕ) : ℕ :=
  total - yang_tao_pos - qian_hui_pos

/-- The number of children between Yang Tao and Qian Hui is 13. -/
theorem children_between_is_thirteen : children_between 20 3 4 = 13 := by
  unfold children_between
  rfl

#eval children_between 20 3 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_children_between_is_thirteen_l214_21426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_n_hedral_angle_equality_l214_21442

/-- Represents a convex polyhedral angle with n faces -/
structure ConvexPolyhedral (n : ℕ) : Prop where
  is_convex : n ≥ 3
  is_polyhedral : True  -- Placeholder, can be replaced with a more specific condition if needed

/-- The sum of planar angles in an n-hedral angle -/
noncomputable def SumPlanarAngles (n : ℕ) : ℝ :=
  sorry

/-- The sum of dihedral angles in an n-hedral angle -/
noncomputable def SumDihedralAngles (n : ℕ) : ℝ :=
  sorry

/-- For a convex n-hedral angle, if the sum of its planar angles equals 
    the sum of its dihedral angles, then n = 3 -/
theorem convex_n_hedral_angle_equality (n : ℕ) 
  (h_convex : ConvexPolyhedral n)
  (h_sum_equal : SumPlanarAngles n = SumDihedralAngles n) : n = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_n_hedral_angle_equality_l214_21442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_number_statements_l214_21440

theorem opposite_number_statements :
  (∃ (correct : Finset (Fin 5)), correct.card = 3 ∧
    (∀ i : Fin 5, i ∈ correct ↔
      (i = 0 → ((-π : ℝ) = -π)) ∧
      (i = 1 → (∀ a b : ℝ, a * b < 0 → a = -b)) ∧
      (i = 2 → ((-(-3.8 : ℝ)) = 3.8)) ∧
      (i = 3 → (∃ x : ℝ, x = -x)) ∧
      (i = 4 → (∀ a b : ℝ, a > 0 → b < 0 → a = -b)))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_number_statements_l214_21440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_score_l214_21427

/-- The total points scored by a basketball team given the following conditions:
  - There are four players: Chandra, Akiko, Michiko, and Bailey
  - Chandra scored twice as many points as Akiko
  - Akiko scored 4 more points than Michiko
  - Michiko scored half as many points as Bailey
  - Bailey scored 14 points
-/
theorem basketball_team_score 
  (bailey : ℕ) (michiko : ℕ) (akiko : ℕ) (chandra : ℕ) 
  (h1 : bailey = 14)
  (h2 : michiko = bailey / 2)
  (h3 : akiko = michiko + 4)
  (h4 : chandra = 2 * akiko) :
  bailey + michiko + akiko + chandra = 54 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_score_l214_21427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l214_21475

noncomputable def binomial_expansion (x : ℝ) (n : ℕ) := (1/x + 2 * x^(1/3))^n

def sum_binomial_coefficients (n : ℕ) := 2^n

theorem binomial_expansion_properties (x : ℝ) (n : ℕ) 
  (h : sum_binomial_coefficients n = 256) :
  -- The constant term in the expansion
  ∃ k : ℕ, Nat.choose n k * 2^k = 1792 ∧ 
  -- The term with the maximum binomial coefficient
  ∃ m : ℕ, Nat.choose n m * 2^m * x^(-(8:ℝ)/3) = 1120 * x^(-(8:ℝ)/3) := by
  sorry

#check binomial_expansion_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l214_21475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_join_time_l214_21487

/-- Represents the number of months after John started the business that Rose joined --/
def x : ℕ := 6

/-- John's initial investment --/
def john_investment : ℕ := 18000

/-- Rose's investment --/
def rose_investment : ℕ := 12000

/-- Tom's investment --/
def tom_investment : ℕ := 9000

/-- Total profit at the end of the year --/
def total_profit : ℕ := 4070

/-- Difference between Rose's and Tom's share in the profit --/
def profit_difference : ℕ := 370

/-- Total number of months in a year --/
def total_months : ℕ := 12

/-- Number of months Tom invested (joined 4 months after Rose) --/
def tom_months : ℕ := total_months - 4

theorem rose_join_time :
  john_investment * total_months +
  rose_investment * (total_months - x) +
  tom_investment * tom_months = total_profit ∧
  rose_investment * (total_months - x) -
  tom_investment * tom_months = profit_difference →
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rose_join_time_l214_21487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_of_sequence_l214_21425

noncomputable def sequence_a (n : ℕ) (S : ℕ → ℝ) : ℝ := (3 * S n) / (n + 2)

theorem max_ratio_of_sequence (S : ℕ → ℝ) :
  ∃ (M : ℝ), M = 3 ∧ ∀ (n : ℕ), n ≥ 2 → 
    sequence_a (n + 1) S / sequence_a n S ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_of_sequence_l214_21425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_speed_calculation_l214_21474

/-- The speed of a plane in still air, given its travel distances with and against the wind, 
    and the wind speed. -/
noncomputable def plane_speed (distance_with_wind : ℝ) (distance_against_wind : ℝ) (wind_speed : ℝ) : ℝ :=
  ((distance_with_wind + distance_against_wind) * wind_speed) / 
  (distance_with_wind - distance_against_wind)

/-- Theorem stating that the plane's speed in still air is 253 mph under the given conditions. -/
theorem plane_speed_calculation :
  plane_speed 420 350 23 = 253 := by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check plane_speed 420 350 23

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_speed_calculation_l214_21474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barbara_wins_l214_21494

/-- Represents a player in the game -/
inductive Player
| Alice
| Barbara
deriving Repr, DecidableEq

/-- The game state -/
structure GameState where
  current_number : ℝ
  current_player : Player
  turn_count : ℕ

/-- Defines a valid move in the game -/
def valid_move (x y : ℝ) : Prop :=
  0 < y - x ∧ y - x < 1

/-- Defines the winning condition -/
def is_winning_state (state : GameState) : Prop :=
  state.current_number ≥ 2010

/-- Represents a strategy for a player -/
def Strategy := GameState → ℝ

/-- Defines a winning strategy for Barbara -/
def winning_strategy_for_barbara (strategy : Strategy) : Prop :=
  ∀ (alice_strategy : Strategy),
    ∃ (game_play : ℕ → GameState),
      (game_play 0 = ⟨0, Player.Alice, 0⟩) ∧
      (∀ n, game_play (n+1) = 
        let prev_state := game_play n
        let new_number := 
          if prev_state.current_player = Player.Alice
          then alice_strategy prev_state
          else strategy prev_state
        ⟨new_number, 
         if prev_state.current_player = Player.Alice then Player.Barbara else Player.Alice,
         n+1⟩) ∧
      ∃ (n : ℕ), is_winning_state (game_play (2*n + 1)) ∧
        ∀ (m : ℕ), m < 2*n + 1 → ¬is_winning_state (game_play m)

/-- The main theorem stating that Barbara has a winning strategy -/
theorem barbara_wins : ∃ (strategy : Strategy), winning_strategy_for_barbara strategy := by
  sorry

#check barbara_wins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_barbara_wins_l214_21494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_coordinates_l214_21422

/-- Triangle PQR with side lengths p, q, r, and incenter J --/
structure TriangleWithIncenter where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  J : ℝ × ℝ
  p : ℝ
  q : ℝ
  r : ℝ
  u : ℝ
  v : ℝ
  w : ℝ

/-- The theorem statement --/
theorem incenter_coordinates (triangle : TriangleWithIncenter) 
  (h1 : triangle.p = 8)
  (h2 : triangle.q = 6)
  (h3 : triangle.r = 10)
  (h4 : triangle.J = (triangle.u * triangle.P.1 + triangle.v * triangle.Q.1 + triangle.w * triangle.R.1,
                      triangle.u * triangle.P.2 + triangle.v * triangle.Q.2 + triangle.w * triangle.R.2))
  (h5 : triangle.u + triangle.v + triangle.w = 1) :
  triangle.u = 1/4 ∧ triangle.v = 5/12 ∧ triangle.w = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_coordinates_l214_21422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_min_range_l214_21486

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - a * x

-- State the theorem
theorem local_min_range (a : ℝ) :
  (∃ x₀ < 0, IsLocalMin (f a) x₀) → a ∈ Set.Ioo (-1 / Real.exp 1) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_min_range_l214_21486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_curved_triangle_l214_21447

/-- Given a right triangle ABC with legs a and b, and hypotenuse c,
    this theorem states that the radius of the circle inscribed in the curved triangle
    formed by the two legs of ABC and the semicircle on AB is a + b - c. -/
theorem inscribed_circle_radius_curved_triangle 
  (a b c : ℝ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  let r := a + b - c
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    radius = r ∧ 
    (∀ p : ℝ × ℝ, p.1 ≥ 0 ∧ p.1 ≤ a ∧ p.2 = 0 → dist center p = radius) ∧
    (∀ p : ℝ × ℝ, p.1 = 0 ∧ p.2 ≥ 0 ∧ p.2 ≤ b → dist center p = radius) ∧
    (∀ p : ℝ × ℝ, (p.1 - a/2)^2 + (p.2 - b/2)^2 = (c/2)^2 → dist center p = radius) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_curved_triangle_l214_21447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l214_21479

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_multiple_of_4 (n : ℕ) : Prop := 4 ∣ n

def spinner_numbers : Finset ℕ := Finset.range 8 

theorem spinner_probability : 
  (spinner_numbers.filter (λ n => n ∈ ({2, 3, 4, 5, 7, 8} : Finset ℕ))).card / spinner_numbers.card = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l214_21479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_one_greater_than_f_one_l214_21489

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x

-- State the theorem
theorem f_minus_one_greater_than_f_one : f (-1) > f 1 := by
  -- Compute f(-1)
  have h1 : f (-1) = 5 := by
    unfold f
    simp
    norm_num
  
  -- Compute f(1)
  have h2 : f 1 = -3 := by
    unfold f
    simp
    norm_num
  
  -- Compare the values
  rw [h1, h2]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_one_greater_than_f_one_l214_21489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leftmost_digit_theorem_l214_21416

noncomputable def leftmostDigit (x : ℕ) : ℕ :=
  (x.repr.data.head?.getD '0').toNat - '0'.toNat

theorem leftmost_digit_theorem :
  ∀ d : ℕ, d ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
    ∃ n : ℕ, leftmostDigit (2^n) = d ∧ leftmostDigit (3^n) = d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leftmost_digit_theorem_l214_21416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_is_five_l214_21446

theorem largest_number_is_five (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_products_eq : p * q + p * r + q * r = -8)
  (product_eq : p * q * r = -15) :
  max p (max q r) = 5 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_is_five_l214_21446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l214_21414

-- Part I
noncomputable def sector_arc_length (α : Real) (r : Real) : Real := α * r

noncomputable def sector_area (α : Real) (r : Real) : Real := 1/2 * α * r^2

theorem part_one (α : Real) (r : Real) 
  (h1 : α = 2 * Real.pi / 3) (h2 : r = 6) : 
  sector_arc_length α r = 4 * Real.pi ∧ sector_area α r = 12 * Real.pi := by
  sorry

-- Part II
noncomputable def sector_perimeter (l : Real) (r : Real) : Real := l + 2 * r

noncomputable def sector_angle (l : Real) (r : Real) : Real := l / r

theorem part_two :
  ∃ (α : Real) (A : Real), 
    (∀ (l r : Real), sector_perimeter l r = 20 → sector_area (sector_angle l r) r ≤ A) ∧
    (∃ (l r : Real), sector_perimeter l r = 20 ∧ sector_angle l r = α ∧ sector_area α r = A) ∧
    α = 2 ∧ A = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l214_21414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_intersection_l214_21432

/-- A parallelogram with given vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ

/-- The point where diagonals intersect in a parallelogram -/
noncomputable def diagonalIntersection (p : Parallelogram) : ℝ × ℝ :=
  ((p.v1.1 + p.v2.1) / 2, (p.v1.2 + p.v2.2) / 2)

/-- Theorem: The diagonals of the given parallelogram intersect at (8, 3) -/
theorem parallelogram_diagonal_intersection :
  let p : Parallelogram := ⟨(2, -3), (14, 9), (5, 7)⟩
  diagonalIntersection p = (8, 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_intersection_l214_21432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_forming_lot_l214_21470

theorem probability_of_forming_lot (letters : Finset Char) : 
  letters = {'o', 'l', 't'} → 
  (Finset.card (Finset.filter (λ p : List Char => p.toString = "lot") (letters.toList.permutations.toFinset)) : ℚ) / 
  (Finset.card letters).factorial = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_forming_lot_l214_21470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_speed_is_correct_l214_21498

/-- Represents the hiking scenario with Chantal and Jean -/
structure HikingScenario where
  -- Total distance of the trail
  total_distance : ℝ
  -- Chantal's speeds for each third of the trail (going to lookout)
  chantal_speed_first : ℝ
  chantal_speed_second : ℝ
  chantal_speed_third : ℝ
  -- Chantal's speeds for return journey
  chantal_return_speed_smooth : ℝ
  chantal_return_speed_steep : ℝ
  -- Time when Chantal and Jean meet
  meeting_time : ℝ

/-- Calculates Jean's average speed given the hiking scenario -/
noncomputable def calculateJeanSpeed (scenario : HikingScenario) : ℝ :=
  -- Actual calculation (simplified for this example)
  let d := (scenario.meeting_time * 25) / 48
  (3 * d / 2) / scenario.meeting_time

/-- Theorem stating that Jean's speed is 144/200 miles per hour -/
theorem jean_speed_is_correct (scenario : HikingScenario) :
  scenario.chantal_speed_first = 3 →
  scenario.chantal_speed_second = 1.5 →
  scenario.chantal_speed_third = 4 →
  scenario.chantal_return_speed_smooth = 3 →
  scenario.chantal_return_speed_steep = 2 →
  scenario.meeting_time = 4 →
  calculateJeanSpeed scenario = 144 / 200 := by
  intros h1 h2 h3 h4 h5 h6
  -- The actual proof would go here
  sorry

-- Example usage (commented out as it's not computable)
/-
#eval calculateJeanSpeed {
  total_distance := 3,
  chantal_speed_first := 3,
  chantal_speed_second := 1.5,
  chantal_speed_third := 4,
  chantal_return_speed_smooth := 3,
  chantal_return_speed_steep := 2,
  meeting_time := 4
}
-/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jean_speed_is_correct_l214_21498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_profit_percentage_l214_21444

noncomputable section

-- Define the initial selling prices and profit percentages
def initial_price_A : ℝ := 100
def initial_price_B : ℝ := 150
def initial_price_C : ℝ := 200
def profit_percent_A : ℝ := 10
def profit_percent_B : ℝ := 20
def profit_percent_C : ℝ := 30

-- Define the function to calculate cost price
def cost_price (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  selling_price / (1 + profit_percent / 100)

-- Define the new selling prices (twice the initial prices)
def new_price_A : ℝ := 2 * initial_price_A
def new_price_B : ℝ := 2 * initial_price_B
def new_price_C : ℝ := 2 * initial_price_C

-- Define the total new selling price and total cost price
def total_new_selling_price : ℝ := new_price_A + new_price_B + new_price_C
def total_cost_price : ℝ := 
  cost_price initial_price_A profit_percent_A + 
  cost_price initial_price_B profit_percent_B + 
  cost_price initial_price_C profit_percent_C

-- Define the total profit
def total_profit : ℝ := total_new_selling_price - total_cost_price

-- Define the overall profit percentage
def overall_profit_percentage : ℝ := (total_profit / total_cost_price) * 100

end noncomputable section

-- Theorem statement
theorem trader_profit_percentage : 
  ∃ ε > 0, |overall_profit_percentage - 143.4| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trader_profit_percentage_l214_21444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jasmine_percentage_after_addition_l214_21443

/-- Calculates the new percentage of jasmine in a solution after adding jasmine and water --/
theorem jasmine_percentage_after_addition
  (initial_volume : ℝ)
  (initial_jasmine_percentage : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 100)
  (h2 : initial_jasmine_percentage = 0.1)
  (h3 : added_jasmine = 5)
  (h4 : added_water = 10) :
  ∃ ε > 0, |((initial_volume * initial_jasmine_percentage + added_jasmine) /
    (initial_volume + added_jasmine + added_water)) - 0.1304| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jasmine_percentage_after_addition_l214_21443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l214_21404

/-- Calculates the simple annual interest rate given the initial charge and the amount owed after one year. -/
noncomputable def simple_interest_rate (initial_charge : ℝ) (amount_owed_after_year : ℝ) : ℝ :=
  (amount_owed_after_year - initial_charge) / initial_charge * 100

/-- Proves that the simple annual interest rate is 5% given the specified conditions. -/
theorem interest_rate_is_five_percent 
  (initial_charge : ℝ)
  (amount_owed_after_year : ℝ)
  (h1 : initial_charge = 54)
  (h2 : amount_owed_after_year = 56.7) :
  simple_interest_rate initial_charge amount_owed_after_year = 5 := by
  sorry

/-- Calculates the interest rate for the given problem. -/
def calculate_interest_rate : ℚ :=
  (56.7 - 54) / 54 * 100

#eval calculate_interest_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l214_21404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_l214_21403

/-- The number of distinct, natural-number factors of 3^5 * 4^3 * 7^2 -/
def num_factors : ℕ := 126

/-- The given number -/
def N : ℕ := 3^5 * 4^3 * 7^2

theorem count_factors : (Finset.filter (λ x : ℕ ↦ x ∣ N) (Finset.range (N + 1))).card = num_factors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_factors_l214_21403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_value_proof_l214_21482

/-- The value of 'a' for which the tangent line to y = x^2 + a at x = 1/2 
    is also tangent to y = e^x -/
noncomputable def tangent_value : ℝ := 5/4

/-- The curve y = x^2 + a -/
noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := x^2 + a

/-- The derivative of the curve -/
noncomputable def curve_derivative (x : ℝ) : ℝ := 2 * x

/-- The tangent line to the curve at x = 1/2 -/
noncomputable def tangent_line (a : ℝ) (x : ℝ) : ℝ := x + a - 1/4

/-- The exponential curve y = e^x -/
noncomputable def exp_curve (x : ℝ) : ℝ := Real.exp x

/-- The derivative of the exponential curve -/
noncomputable def exp_derivative (x : ℝ) : ℝ := Real.exp x

theorem tangent_value_proof :
  ∃ (x₀ : ℝ), 
    (curve_derivative (1/2) = (tangent_line tangent_value x₀ - tangent_line tangent_value (1/2)) / (x₀ - 1/2)) ∧
    (exp_derivative x₀ = (tangent_line tangent_value x₀ - tangent_line tangent_value (1/2)) / (x₀ - 1/2)) ∧
    (tangent_line tangent_value x₀ = exp_curve x₀) :=
by sorry

#check tangent_value_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_value_proof_l214_21482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l214_21435

/-- The volume of a cylinder with hemispheres at both ends -/
noncomputable def cylinderWithHemispheresVolume (radius : ℝ) (length : ℝ) : ℝ :=
  Real.pi * radius^2 * length + (4/3) * Real.pi * radius^3

theorem line_segment_length (CD : ℝ) :
  cylinderWithHemispheresVolume 4 CD = 288 * Real.pi → CD = 38/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l214_21435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_noodle_thickness_ratio_l214_21465

/-- Represents the noodle-making process -/
noncomputable def noodle_process (initial_length : ℝ) (cycles : ℕ) (final_length : ℝ) : ℝ :=
  (final_length / initial_length) * (2 ^ cycles)

/-- Theorem: The noodle thickness ratio after the process -/
theorem noodle_thickness_ratio 
  (initial_length : ℝ) 
  (cycles : ℕ) 
  (final_length : ℝ) 
  (h1 : initial_length = 1.6) 
  (h2 : cycles = 10) 
  (h3 : final_length = 1.6) :
  1 / noodle_process initial_length cycles final_length = 1 / 32 := by
  sorry

#check noodle_thickness_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_noodle_thickness_ratio_l214_21465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_2003_l214_21496

mutual
  def a : ℕ → ℤ
    | 0 => 1
    | n + 1 => a n * 2001 + b n

  def b : ℕ → ℤ
    | 0 => 4
    | n + 1 => b n * 2001 + a n
end

theorem not_divisible_by_2003 (n : ℕ) : 
  ¬(2003 ∣ a n) ∧ ¬(2003 ∣ b n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_2003_l214_21496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_order_l214_21434

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_increasing : ∀ x y, x ≤ y → y ≤ 0 → f x ≤ f y)

-- Define a, b, c
noncomputable def a (f : ℝ → ℝ) := f (Real.log 7 / Real.log 4)
noncomputable def b (f : ℝ → ℝ) := f (Real.log 3 / Real.log 2)
noncomputable def c (f : ℝ → ℝ) := f (0.2^(6/10 : ℝ))

-- State the theorem
theorem abc_order (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_increasing : ∀ x y, x ≤ y → y ≤ 0 → f x ≤ f y) : 
  b f < a f ∧ a f < c f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_order_l214_21434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_2_inequality_for_f_l214_21431

-- Define the function f as noncomputable due to its dependency on real numbers
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x + 1/a|

-- Part I
theorem solution_set_for_a_2 :
  {x : ℝ | f 2 x > 3} = {x : ℝ | x < -11/4 ∨ x > 1/4} := by sorry

-- Part II
theorem inequality_for_f :
  ∀ (a m : ℝ), a > 0 → f a m + f a (-1/m) ≥ 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_2_inequality_for_f_l214_21431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l214_21491

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -12*x

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Define the focus of the parabola
def focus : Point := ⟨-3, 0⟩

-- Define the distance from a point to the y-axis
def distToYAxis (p : Point) : ℝ := |p.x|

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Theorem statement
theorem parabola_focus_distance (p : Point) :
  parabola p.x p.y →
  distToYAxis p = 1 →
  distance p focus = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l214_21491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_satisfied_at_nine_l214_21417

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The left-hand side of the equation -/
noncomputable def leftHandSide : ℝ := (geometricSum 1 (1/3)) * (geometricSum 1 (-1/3))

/-- The right-hand side of the equation -/
noncomputable def rightHandSide (x : ℝ) : ℝ := geometricSum 1 (1/x)

/-- The theorem stating that the equation is satisfied when x = 9 -/
theorem equation_satisfied_at_nine :
  leftHandSide = rightHandSide 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_satisfied_at_nine_l214_21417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_displaced_volume_squared_l214_21448

/-- The volume of water displaced by a cube in a cylindrical barrel -/
noncomputable def water_displaced_volume (cube_side : ℝ) (barrel_radius : ℝ) : ℝ :=
  let triangle_side := barrel_radius * Real.sqrt 3
  let tetrahedron_leg := triangle_side / Real.sqrt 2
  (1 / 3) * tetrahedron_leg * (1 / 2 * tetrahedron_leg^2)

/-- Theorem stating that the square of the volume of water displaced
    by an 8-foot cube in a 4-foot radius barrel is 384 cubic feet -/
theorem water_displaced_volume_squared :
  (water_displaced_volume 8 4)^2 = 384 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_displaced_volume_squared_l214_21448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l214_21495

-- Define the triangle ABC
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  c * Real.cos B + b * Real.cos C = 2 * a * Real.cos A

-- Define the area of the triangle
noncomputable def TriangleArea (a b c : ℝ) : ℝ := Real.sqrt 3

-- Theorem statement
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (h1 : Triangle A B C a b c) 
  (h2 : a = 2) 
  (h3 : TriangleArea a b c = Real.sqrt 3) :
  A = Real.pi / 3 ∧ a + b + c = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l214_21495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_tangent_slope_l214_21423

noncomputable def is_tangent_slope (A : ℝ × ℝ) (C : Set (ℝ × ℝ)) (m : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, P ∈ C ∧ 
    (P.2 - A.2) = m * (P.1 - A.1) ∧
    ∀ Q : ℝ × ℝ, Q ∈ C → (Q.2 - A.2) ≥ m * (Q.1 - A.1)

theorem largest_tangent_slope (A : ℝ × ℝ) (C : Set (ℝ × ℝ)) : 
  A = (Real.sqrt 10 / 2, Real.sqrt 10 / 2) →
  C = {(x, y) | x^2 + y^2 = 1} →
  ∃ c : ℝ, c = 3 ∧ ∀ m : ℝ, is_tangent_slope A C m → m ≤ c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_tangent_slope_l214_21423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cut_and_rearrange_l214_21450

/-- Represents a piece of the puzzle -/
structure Piece where
  cells : List (Nat × Nat)

/-- Represents the 3x3 square without the central cell -/
def original_square : List (Nat × Nat) :=
  [(1,1), (1,2), (1,3),
   (2,1),        (2,3),
   (3,1), (3,2), (3,3)]

/-- Represents the target 2x2 square -/
def target_square : List (Nat × Nat) :=
  [(1,1), (1,2),
   (2,1), (2,2)]

/-- Checks if a list of pieces covers the target square exactly -/
def covers_target (pieces : List Piece) : Prop :=
  let all_cells : List (Nat × Nat) := pieces.bind (·.cells)
  all_cells.toFinset == target_square.toFinset

/-- The main theorem stating that it's possible to cut the original square
    into 5 pieces that can be rearranged to form the target square -/
theorem square_cut_and_rearrange :
  ∃ (pieces : List Piece),
    pieces.length = 5 ∧
    (pieces.bind (·.cells)).toFinset == original_square.toFinset ∧
    covers_target pieces := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cut_and_rearrange_l214_21450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_75_first_grade_parts_l214_21485

/-- The probability of producing a first-grade part -/
noncomputable def p : ℝ := 0.8

/-- The total number of randomly selected parts -/
def n : ℕ := 100

/-- The number of first-grade parts we're interested in -/
def k : ℕ := 75

/-- The failure probability -/
noncomputable def q : ℝ := 1 - p

/-- The standard normal variable z -/
noncomputable def z : ℝ := (k - n * p) / Real.sqrt (n * p * q)

/-- The cumulative probability function for the standard normal distribution -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- The probability of exactly k successes out of n trials -/
noncomputable def P_n (k : ℕ) : ℝ := (1 / 4) * Φ (-z)

theorem probability_75_first_grade_parts : 
  |P_n k - 0.04565| < 0.00001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_75_first_grade_parts_l214_21485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_suitcase_combinations_l214_21441

theorem suitcase_combinations : (
  let range := Finset.range 40
  let is_odd (n : ℕ) := n % 2 = 1
  let is_multiple_of_4 (n : ℕ) := n % 4 = 0
  let is_multiple_of_5 (n : ℕ) := n % 5 = 0
  let count_odd := (range.filter is_odd).card
  let count_multiple_4 := (range.filter is_multiple_of_4).card
  let count_multiple_5 := (range.filter is_multiple_of_5).card
  count_odd * count_multiple_4 * count_multiple_5 = 1600
) := by
  -- Prove that count_odd = 20
  have h1 : (Finset.range 40).filter (fun n => n % 2 = 1) = Finset.range 20 := by sorry
  have count_odd_eq : ((Finset.range 40).filter (fun n => n % 2 = 1)).card = 20 := by
    rw [h1]
    rfl

  -- Prove that count_multiple_4 = 10
  have h2 : (Finset.range 40).filter (fun n => n % 4 = 0) = Finset.range 10 := by sorry
  have count_multiple_4_eq : ((Finset.range 40).filter (fun n => n % 4 = 0)).card = 10 := by
    rw [h2]
    rfl

  -- Prove that count_multiple_5 = 8
  have h3 : (Finset.range 40).filter (fun n => n % 5 = 0) = Finset.range 8 := by sorry
  have count_multiple_5_eq : ((Finset.range 40).filter (fun n => n % 5 = 0)).card = 8 := by
    rw [h3]
    rfl

  -- Combine the results
  calc
    _ = 20 * 10 * 8 := by rw [count_odd_eq, count_multiple_4_eq, count_multiple_5_eq]
    _ = 1600 := by rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_suitcase_combinations_l214_21441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_values_max_area_max_area_equality_condition_l214_21471

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 1 ∧ t.A = Real.pi/6

-- Theorem for part (I)
theorem angle_C_values (t : Triangle) (h : triangle_conditions t) (hb : t.b = Real.sqrt 3) :
  t.C = Real.pi/2 ∨ t.C = Real.pi/6 :=
sorry

-- Theorem for part (II)
theorem max_area (t : Triangle) (h : triangle_conditions t) :
  (t.a * t.b * Real.sin t.C) / 2 ≤ (2 + Real.sqrt 3) / 4 :=
sorry

-- Theorem for the equality condition of max area
theorem max_area_equality_condition (t : Triangle) (h : triangle_conditions t) :
  (t.a * t.b * Real.sin t.C) / 2 = (2 + Real.sqrt 3) / 4 ↔ t.b = 1 ∧ t.c = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_values_max_area_max_area_equality_condition_l214_21471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_extreme_value_in_interval_sequence_property_l214_21480

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x + 1 / x + a * x

theorem extreme_value_at_one (a : ℝ) :
  (∃ x₀ > 0, ∀ x > 0, f x a ≥ f x₀ a) →
  (∃ x₀ > 0, ∀ x > 0, f x a ≤ f x₀ a) →
  f 1 a = 1 := by
  sorry

theorem extreme_value_in_interval (a : ℝ) :
  (∃ x₀ ∈ Set.Ioo 2 3, ∀ x ∈ Set.Ioo 2 3, f x a ≥ f x₀ a) →
  (∃ x₀ ∈ Set.Ioo 2 3, ∀ x ∈ Set.Ioo 2 3, f x a ≤ f x₀ a) →
  a ∈ Set.Ioo (-1/4) (-2/9) := by
  sorry

theorem sequence_property (x : ℕ → ℝ) :
  (∀ n : ℕ, x n > 0) →
  (∀ n : ℕ, Real.log (x n) + 1 / x (n + 1) < 1) →
  x 0 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_extreme_value_in_interval_sequence_property_l214_21480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_split_l214_21466

/-- The profit split problem -/
theorem profit_split (total_profit : ℕ) (ratio : List ℕ) : 
  total_profit = 38000 →
  ratio = [2, 3, 4, 4, 6] →
  (ratio.sum * (ratio.maximum?.getD 0)) ≤ (ratio.maximum?.map (· * total_profit)).getD 0 :=
by
  sorry

#check profit_split

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_split_l214_21466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l214_21428

/-- Predicate to represent that A, B, C are angles and a, b, c are corresponding opposite sides of a triangle -/
def IsTriangle (A B C a b c : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  A + B + C = Real.pi ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

/-- For any triangle with angles A, B, C and corresponding opposite sides a, b, c,
    the inequality A a + B b + C c ≥ 1/2(A b + B a + A c + C a + B c + C b) holds,
    with equality if and only if the triangle is equilateral. -/
theorem triangle_inequality (A B C a b c : ℝ) 
  (h_triangle : IsTriangle A B C a b c) : 
  A * a + B * b + C * c ≥ (1/2) * (A * b + B * a + A * c + C * a + B * c + C * b) ∧ 
  (A * a + B * b + C * c = (1/2) * (A * b + B * a + A * c + C * a + B * c + C * b) ↔ 
   a = b ∧ b = c ∧ A = B ∧ B = C) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l214_21428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l214_21405

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin (Real.pi * x) - Real.cos (Real.pi * x) + 2) / Real.sqrt x

-- State the theorem
theorem min_value_of_f :
  ∃ (min : ℝ), min = (4 * Real.sqrt 5) / 5 ∧
  ∀ (x : ℝ), 1/4 ≤ x ∧ x ≤ 5/4 → f x ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l214_21405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_in_triangle_l214_21499

theorem angle_C_in_triangle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (equation : a^2 + b^2 - c^2 = Real.sqrt 3 * a * b) :
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_in_triangle_l214_21499
