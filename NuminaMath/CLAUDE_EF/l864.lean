import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_25_l864_86463

-- Define the points as real numbers (representing their positions on a line)
variable (P Q R S : ℝ)

-- Define the conditions
axiom pq_distance : |Q - P| = 13
axiom qr_distance : |R - Q| = 11
axiom rs_distance : |S - R| = 14
axiom sp_distance : |P - S| = 12

-- Define the theorem
theorem max_distance_is_25 :
  ∃ (X Y : ℝ), X ∈ ({P, Q, R, S} : Set ℝ) ∧ Y ∈ ({P, Q, R, S} : Set ℝ) ∧ |X - Y| = 25 ∧
  ∀ (A B : ℝ), A ∈ ({P, Q, R, S} : Set ℝ) → B ∈ ({P, Q, R, S} : Set ℝ) → |A - B| ≤ 25 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_25_l864_86463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l864_86438

noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x + 1/x)

theorem min_value_of_f :
  ∀ x : ℝ, x > 0 → f x ≥ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l864_86438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_garden_cost_l864_86465

/-- Represents the cost of a flower type -/
structure FlowerCost where
  price : ℚ
  deriving Repr

/-- Represents a rectangular region in the garden -/
structure Region where
  length : ℚ
  width : ℚ
  deriving Repr

/-- Calculates the area of a region -/
def area (r : Region) : ℚ := r.length * r.width

/-- Calculates the cost of planting a region with a specific flower type -/
def plantCost (r : Region) (f : FlowerCost) : ℚ := area r * f.price

/-- The main theorem stating the minimum cost of planting the garden -/
theorem min_garden_cost (aster begonia canna dahlia easter_lily : FlowerCost)
  (region_a region_b region_c region_d region_e : Region) :
  aster.price = 1 →
  begonia.price = 3/2 →
  canna.price = 2 →
  dahlia.price = 5/2 →
  easter_lily.price = 3 →
  region_a.length = 5 ∧ region_a.width = 2 →
  region_b.length = 5 ∧ region_b.width = 5 →
  region_c.length = 2 ∧ region_c.width = 7 →
  region_d.length = 5 ∧ region_d.width = 4 →
  region_e.length = 7 ∧ region_e.width = 3 →
  plantCost region_a easter_lily +
  plantCost region_b aster +
  plantCost region_c dahlia +
  plantCost region_d canna +
  plantCost region_e begonia = 323/2 := by
  sorry

#eval 323/2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_garden_cost_l864_86465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l864_86403

/-- The function f(x) with parameters ω and φ -/
noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ) + 1

/-- Theorem stating the range of φ given the conditions -/
theorem phi_range (ω φ : ℝ) : 
  ω > 0 ∧ 
  |φ| ≤ π/2 ∧
  (∃ k : ℝ, ∀ x : ℝ, f ω φ x = -1 → f ω φ (x + π) = -1) ∧
  (∀ x ∈ Set.Ioo (-π/12) (π/3), f ω φ x > 1) →
  φ ∈ Set.Icc (π/6) (π/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l864_86403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_star_equation_l864_86486

-- Define the "※" operation
def star (a b : ℚ) : ℚ := (a + 2 * b) / 3

-- State the theorem
theorem solve_star_equation (x : ℚ) : 
  star 6 x = 22 / 3 → x = 8 := by
  -- The proof would go here, but we're skipping it as per instructions
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_star_equation_l864_86486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l864_86487

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

theorem f_properties (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Ioo 0 (Real.sqrt a), StrictMonoOn (fun y => -f a y) (Set.Ioo 0 (Real.sqrt a))) ∧
  (∀ x ∈ Set.Ioi (Real.sqrt a), StrictMonoOn (f a) (Set.Ioi (Real.sqrt a))) ∧
  (∃ x : ℝ, f 1 (2^x) = 2) ∧
  (∀ m : ℝ, (∀ x : ℝ, f 1 (4^x) ≥ m * f 1 (2^x) - 6) ↔ m ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l864_86487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_of_curve_C_l864_86428

noncomputable def curve_C (α : Real) : Real × Real :=
  (3 + Real.sqrt 10 * Real.cos α, 1 + Real.sqrt 10 * Real.sin α)

def line (θ ρ : Real) : Prop :=
  Real.sin θ - Real.cos θ = 1 / ρ

theorem chord_length_of_curve_C :
  ∃ (chord_length : Real),
    chord_length = Real.sqrt 22 ∧
    ∀ (α θ ρ : Real),
      let (x, y) := curve_C α
      line θ ρ →
      (x - 3) ^ 2 + (y - 1) ^ 2 = 10 →
      chord_length ^ 2 = 22 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_of_curve_C_l864_86428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_AOB_l864_86496

/-- Line l in polar form -/
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin θ = ρ * Real.cos θ + 2

/-- Circle C in polar form -/
def circle_C (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ + 2 * Real.sin θ

/-- Theorem stating the cosine of angle AOB -/
theorem cosine_AOB (A B : ℝ × ℝ) :
  (∃ (ρ₁ θ₁ ρ₂ θ₂ : ℝ), 
    line_l ρ₁ θ₁ ∧ circle_C ρ₁ θ₁ ∧ 
    line_l ρ₂ θ₂ ∧ circle_C ρ₂ θ₂ ∧
    A = (ρ₁ * Real.cos θ₁, ρ₁ * Real.sin θ₁) ∧
    B = (ρ₂ * Real.cos θ₂, ρ₂ * Real.sin θ₂) ∧
    A ≠ B) →
  (A.1 * B.1 + A.2 * B.2) / (Real.sqrt (A.1^2 + A.2^2) * Real.sqrt (B.1^2 + B.2^2)) = 3 * Real.sqrt 10 / 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_AOB_l864_86496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l864_86424

noncomputable def f (x : ℝ) : ℝ := (x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1) / (x^3 - 8)

theorem domain_of_f :
  {x : ℝ | ¬(f x = 0 / 0)} = {x : ℝ | x < 2 ∨ x > 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l864_86424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l864_86477

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - 3*y^2 = 3

-- Define the foci
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define point P
noncomputable def P : ℝ × ℝ := (2, Real.sqrt (1/3))

-- Define a line through F₂ and P
def line_through_F₂_P (x y : ℝ) : Prop :=
  ∃ t : ℝ, (x, y) = ((1 - t) * F₂.1 + t * P.1, (1 - t) * F₂.2 + t * P.2)

-- State the theorem
theorem hyperbola_triangle_perimeter :
  ∀ Q : ℝ × ℝ,
  hyperbola Q.1 Q.2 →
  line_through_F₂_P Q.1 Q.2 →
  Q.1 > 2 →
  let perimeter := dist F₁ P + dist F₁ Q + dist P Q
  perimeter = 16 * Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_perimeter_l864_86477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l864_86427

/-- Calculates the time taken for a train to cross a man walking in the opposite direction, considering incline effects --/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) 
  (train_speed_decrease : ℝ) (man_speed_increase : ℝ) : ℝ :=
  let adjusted_train_speed := train_speed * (1 - train_speed_decrease)
  let adjusted_man_speed := man_speed * (1 + man_speed_increase)
  let relative_speed := adjusted_train_speed + adjusted_man_speed
  let relative_speed_ms := relative_speed * 1000 / 3600
  train_length / relative_speed_ms

/-- The time taken for the train to cross the man is approximately 41.12 seconds --/
theorem train_crossing_time_approx :
  ∃ ε > 0, |train_crossing_time 800 72 5 0.1 0.05 - 41.12| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l864_86427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_is_common_tangent_l864_86413

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

-- Define a parabola
structure Parabola where
  focus : Point
  directrix : ℝ  -- Assuming the directrix is a vertical line x = directrix

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define a function to check if a point is on a parabola
def isOnParabola (p : Point) (parab : Parabola) : Prop :=
  distance p parab.focus = |p.x - parab.directrix|

-- Define a function to get the circle centered at a point with radius equal to its distance from the focus
noncomputable def getCircle (p : Point) (focus : Point) : Circle :=
  { center := p, radius := distance p focus }

-- Define a function to check if a line is tangent to a circle
def isTangent (x : ℝ) (c : Circle) : Prop :=
  |x - c.center.x| = c.radius

-- The main theorem
theorem directrix_is_common_tangent (F P₁ P₂ : Point) (parab : Parabola) 
    (h1 : isOnParabola P₁ parab) (h2 : isOnParabola P₂ parab) (h3 : parab.focus = F) :
    ∃ x : ℝ, isTangent x (getCircle P₁ F) ∧ isTangent x (getCircle P₂ F) ∧ x = parab.directrix := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_is_common_tangent_l864_86413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_2017_l864_86400

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (2 * (x + 2)^2 + Real.log (Real.sqrt (1 + 9*x^2) - 3*x) * Real.cos x) / (x^2 + 4)

-- State the theorem
theorem f_negative_2017 (h : f 2017 = 2016) : f (-2017) = -2012 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_2017_l864_86400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_max_area_rectangle_max_area_achieved_l864_86495

theorem rectangle_max_area (x : ℝ) : Real.sin x * Real.cos x ≤ 1 / 2 := by
  sorry

theorem rectangle_max_area_achieved : ∃ x : ℝ, Real.sin x * Real.cos x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_max_area_rectangle_max_area_achieved_l864_86495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_elements_equal_l864_86441

/-- A property that characterizes the special nature of the set S -/
def has_equal_partition (S : Finset Int) : Prop :=
  ∀ x, x ∈ S → ∃ A B : Finset Int,
    A ∪ B = S \ {x} ∧ A ∩ B = ∅ ∧
    A.card = B.card ∧
    (A.sum id : Int) = (B.sum id : Int)

/-- The main theorem stating that if S has the equal partition property,
    then all its elements are equal -/
theorem all_elements_equal (S : Finset Int) (h : has_equal_partition S) :
  ∀ x y, x ∈ S → y ∈ S → x = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_elements_equal_l864_86441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_parabolas_l864_86447

/-- The area of the region enclosed by the parabolas x = -3y^2 and x = 1 - 4y^2 -/
noncomputable def enclosed_area : ℝ := 2 * ∫ y in (Set.Icc 0 1), (1 - y^2)

/-- Theorem stating that the area of the region enclosed by the given parabolas is 4/3 -/
theorem area_between_parabolas : enclosed_area = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_parabolas_l864_86447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_distances_l864_86436

/-- Given two perpendicular lines intersecting at point P, where one line passes through (0, 0) 
    and the other passes through (1, 3), the maximum value of |PA| · |PB| is 5. -/
theorem max_product_of_distances : 
  ∃ (P : ℝ × ℝ), let A : ℝ × ℝ := (0, 0)
                 let B : ℝ × ℝ := (1, 3)
                 let PA := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
                 let PB := Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
                 ∀ m : ℝ, (P.1 + m * P.2 = 0) → (m * P.1 - P.2 - m + 3 = 0) → 
                 (∀ Q : ℝ × ℝ, let QA := Real.sqrt ((Q.1 - A.1)^2 + (Q.2 - A.2)^2)
                               let QB := Real.sqrt ((Q.1 - B.1)^2 + (Q.2 - B.2)^2)
                               QA * QB ≤ PA * PB) ∧
                 PA * PB = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_distances_l864_86436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_circles_in_square_l864_86489

/-- A square in a 2D plane -/
structure Square where
  vertices : Finset (ℝ × ℝ)
  is_square : vertices.card = 4 -- Add other conditions to ensure it's a square

/-- A circle defined by two points (diameter endpoints) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Function to create a circle from two points -/
noncomputable def circle_from_points (p q : ℝ × ℝ) : Circle := sorry

/-- Function to check if two circles are distinct -/
def are_circles_distinct (c1 c2 : Circle) : Prop := sorry

/-- The main theorem -/
theorem distinct_circles_in_square (S : Square) :
  ∃ (C : Finset Circle), C.card = 2 ∧
    (∀ c, c ∈ C → ∃ p q, p ∈ S.vertices ∧ q ∈ S.vertices ∧ c = circle_from_points p q) ∧
    (∀ p q, p ∈ S.vertices → q ∈ S.vertices → ∃ c, c ∈ C ∧ c = circle_from_points p q) ∧
    (∀ c1 c2, c1 ∈ C → c2 ∈ C → c1 ≠ c2 → are_circles_distinct c1 c2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_circles_in_square_l864_86489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_6_l864_86466

/-- Definition of the sequence a_n -/
def a : ℕ → ℤ
  | 0 => 3  -- We define a(0) to be 3 to match a(1) in the original problem
  | 1 => 6  -- This corresponds to a(2) in the original problem
  | (n + 2) => a (n + 1) - a n

/-- Theorem stating that the 2018th term of the sequence is 6 -/
theorem a_2018_equals_6 : a 2017 = 6 := by
  sorry

/-- Helper lemma to show the periodicity of the sequence -/
lemma a_period_6 (n : ℕ) : a (n + 6) = a n := by
  sorry

/-- Proof that a(2017) is indeed the 2018th term of the sequence -/
lemma a_2017_is_2018th_term : a 2017 = a (2018 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_6_l864_86466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_intersecting_circles_l864_86414

/-- Given two circles in a 2D plane, this theorem states that 
    the perpendicular bisector of the line segment connecting 
    their intersection points has a specific equation. -/
theorem perpendicular_bisector_of_intersecting_circles 
  (C₁ C₂ : Set (ℝ × ℝ)) : 
  (C₁ = {(x, y) | x^2 + y^2 - 4*x + 6*y = 0}) →
  (C₂ = {(x, y) | x^2 + y^2 - 6*x = 0}) →
  ∃ (A B : ℝ × ℝ), A ∈ C₁ ∩ C₂ ∧ B ∈ C₁ ∩ C₂ ∧ A ≠ B →
  ∀ (P : ℝ × ℝ), (3 * P.1 - P.2 - 9 = 0) ↔ 
    dist P A = dist P B ∧ 
    (P.1 - A.1) * (B.1 - A.1) + (P.2 - A.2) * (B.2 - A.2) = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_intersecting_circles_l864_86414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_symmetry_decreasing_l864_86430

def f (m : ℕ) (x : ℝ) : ℝ := x^(m^2 - 2*m - 3)

theorem power_function_symmetry_decreasing (m : ℕ) : 
  (∀ x : ℝ, f m x = f m (-x)) → -- Symmetry about y-axis
  (∀ x y : ℝ, 0 < x → x < y → f m y < f m x) → -- Decreasing on (0, +∞)
  m = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_symmetry_decreasing_l864_86430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l864_86420

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The statement of the problem -/
theorem ellipse_triangle_area 
  (C : Ellipse) 
  (F A B : Point) 
  (h_on_ellipse : ∀ (p : Point), (p.x^2 / C.a^2) + (p.y^2 / C.b^2) = 1 → p = A ∨ p = B)
  (h_line_through_origin : ∃ (k : ℝ), A.y = k * A.x ∧ B.y = k * B.x)
  (h_AF : distance A F = 2)
  (h_BF : distance B F = 4)
  (h_eccentricity : Real.sqrt (C.a^2 - C.b^2) / C.a = Real.sqrt 7 / 3) :
  ∃ (S : ℝ), S = 2 * Real.sqrt 3 ∧ S = (1/2) * distance A F * distance B F * Real.sin (Real.arccos (-1/2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l864_86420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l864_86435

theorem min_value_trig_expression (α : ℝ) (h : 0 < α ∧ α < π / 2) :
  (∀ β : ℝ, 0 < β ∧ β < π / 2 → 
    Real.sin α + Real.cos α + (2 * Real.sqrt 2) / Real.sin (α + π / 4) ≤ 
    Real.sin β + Real.cos β + (2 * Real.sqrt 2) / Real.sin (β + π / 4)) →
  Real.sin α + Real.cos α + (2 * Real.sqrt 2) / Real.sin (α + π / 4) = 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_trig_expression_l864_86435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_30_terms_is_255_l864_86422

def sequence_a (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | 1 => 2
  | n + 2 => sequence_a n + 1 + (-1)^n

theorem sum_of_30_terms_is_255 : 
  (Finset.range 30).sum (λ i => sequence_a i) = 255 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_30_terms_is_255_l864_86422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_number_is_seven_l864_86450

def optimal_number (n : ℕ) (fixed_number : ℕ) : ℕ :=
  -- The function to calculate the optimal number
  7 -- We're setting this to 7 based on the problem solution

theorem optimal_number_is_seven :
  let n : ℕ := 100  -- Total number of students
  let fixed_number : ℕ := 107  -- Number written by one student
  let optimal : ℕ := optimal_number n fixed_number
  optimal = 7 ∧ 
  ∀ (x : ℕ), x ≠ optimal → 
    |x - (3 + ((fixed_number : ℚ) + (n - 1 : ℚ) * (optimal : ℚ)) / (2 * n : ℚ))| ≥ 
    |optimal - (3 + ((fixed_number : ℚ) + (n - 1 : ℚ) * (optimal : ℚ)) / (2 * n : ℚ))| :=
by
  sorry

#check optimal_number_is_seven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_number_is_seven_l864_86450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l864_86454

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := (1 - x^2) / (x^2 + 1)^2

-- Define the point of tangency
def x₀ : ℝ := -2

-- Define the slope of the tangent line
noncomputable def m : ℝ := f' x₀

-- Define the y-coordinate of the point of tangency
noncomputable def y₀ : ℝ := f x₀

-- Theorem statement
theorem tangent_line_equation :
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -3/25 * x - 16/25 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l864_86454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l864_86476

/-- The time it takes for two trains to cross each other -/
noncomputable def train_crossing_time (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let total_length := length1 + length2
  let relative_speed := (speed1 + speed2) * 1000 / 3600
  total_length / relative_speed

theorem train_crossing_time_approx :
  ∃ ε > 0, ε < 0.01 ∧ 
  |train_crossing_time 140 210 60 40 - 12.59| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l864_86476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_added_four_onions_l864_86456

/-- The number of onions in Sara's basket initially -/
def initial : ℕ := 0

/-- The number of onions Sara added to the basket -/
def added : ℕ := 0

/-- Theorem: Given the conditions of the problem, Sara added 4 onions to the basket -/
theorem sara_added_four_onions :
  ∀ (initial added : ℕ),
  (initial + added - 5 + 9 = initial + 8) →
  added = 4 := by
  intro initial added h
  -- Proof steps would go here
  sorry

#check sara_added_four_onions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_added_four_onions_l864_86456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equiv_fraction_l864_86434

/-- Given a repeating decimal 0.overline(52), prove it equals 52/99 -/
theorem repeating_decimal_equiv_fraction : ∃ (x : ℚ), x = 52 / 99 ∧ 
  (∀ (n : ℕ), (x * 10^n - (x * 10^n).floor) = (52 * 10^(n % 2) / 100 : ℚ) - ((52 * 10^(n % 2) / 100 : ℚ).floor)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_equiv_fraction_l864_86434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_l864_86467

-- Define the exponential function
noncomputable def f (x : ℝ) := Real.exp x

-- Define the derivative of the exponential function
noncomputable def f' (x : ℝ) := Real.exp x

-- Theorem statement
theorem tangent_line_through_origin :
  ∃ (x : ℝ), 
    (f x = Real.exp x) ∧
    (0 = f' x * 0) ∧
    (x = 1) ∧
    (f x = Real.exp 1) ∧
    (f' x = Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_l864_86467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_satisfies_conditions_l864_86431

def grid : List (List Nat) := [[1, 5, 7], [4, 8, 2], [6, 3, 9]]

def to_number (l : List Nat) : Nat :=
  match l with
  | [a, b, c] => 100 * a + 10 * b + c
  | _ => 0

def row_sum_valid (g : List (List Nat)) : Prop :=
  match g with
  | [row1, row2, row3] => to_number row1 + to_number row2 = to_number row3
  | _ => False

def column_sum_valid (g : List (List Nat)) : Prop :=
  match g with
  | [[a1, b1, c1], [a2, b2, c2], [a3, b3, c3]] =>
    to_number [a1, a2, a3] + to_number [b1, b2, b3] = to_number [c1, c2, c3]
  | _ => False

def contains_required_numbers (g : List (List Nat)) : Prop :=
  let flattened := g.join
  [2, 3, 4, 5, 6, 8].all (fun n => n ∈ flattened)

theorem grid_satisfies_conditions : 
  row_sum_valid grid ∧ 
  column_sum_valid grid ∧ 
  contains_required_numbers grid := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_satisfies_conditions_l864_86431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_side_maximizes_area_l864_86474

/-- Given a rectangle with sides a and b, and a right triangle with legs a₁ and b₁ cut off from it,
    the function calculates the length of one side of the remaining rectangle that maximizes its area. -/
noncomputable def max_area_side (a b a₁ b₁ : ℝ) : ℝ :=
  b / 2 + (b₁ / (2 * a₁)) * (a - a₁)

/-- Theorem stating that the calculated side length maximizes the area of the remaining rectangle. -/
theorem max_area_side_maximizes_area 
  (a b a₁ b₁ : ℝ) 
  (ha : a > 0) (hb : b > 0) (ha₁ : a₁ > 0) (hb₁ : b₁ > 0)
  (ha₁_le_a : a₁ ≤ a) (hb₁_le_b : b₁ ≤ b) : 
  ∀ x, 0 ≤ x ∧ x ≤ b → 
    x * (a - (a₁ / b₁) * x) ≤ (max_area_side a b a₁ b₁) * (a - (a₁ / b₁) * (max_area_side a b a₁ b₁)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_side_maximizes_area_l864_86474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l864_86409

noncomputable section

-- Define h and j as functions from ℝ to ℝ
noncomputable def h : ℝ → ℝ := sorry
noncomputable def j : ℝ → ℝ := sorry

-- Axioms for the intersection points
axiom h_j_intersect_1 : h 3 = j 3 ∧ h 3 = 3
axiom h_j_intersect_2 : h 5 = j 5 ∧ h 5 = 9
axiom h_j_intersect_3 : h 7 = j 7 ∧ h 7 = 21
axiom h_j_intersect_4 : h 9 = j 9 ∧ h 9 = 21

-- Theorem statement
theorem intersection_sum :
  ∃ (x y : ℝ), h (2 * x) = 3 * j x ∧ h (2 * x) = y ∧ x + y = 25.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l864_86409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_7_5_l864_86464

/-- The area of a quadrilateral given its four vertices -/
noncomputable def quadrilateralArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := v1
  let (x2, y2) := v2
  let (x3, y3) := v3
  let (x4, y4) := v4
  (1/2) * abs ((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

/-- Theorem: The area of the quadrilateral with vertices at (2, 1), (4, 3), (7, 1), and (4, 6) is 7.5 -/
theorem quadrilateral_area_is_7_5 :
  quadrilateralArea (2, 1) (4, 3) (7, 1) (4, 6) = 7.5 := by
  -- Unfold the definition of quadrilateralArea
  unfold quadrilateralArea
  -- Simplify the expression
  simp
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_7_5_l864_86464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colinear_vectors_sum_l864_86499

/-- Two vectors are colinear if one is a scalar multiple of the other -/
def Colinear (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a = k • b

/-- The problem statement -/
theorem colinear_vectors_sum (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (2*x, 1, 3)
  let b : ℝ × ℝ × ℝ := (1, -2*y, 9)
  Colinear a b → x + y = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colinear_vectors_sum_l864_86499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_net_profit_calculation_l864_86482

/-- Calculate the net profit given purchase price, markup, and overhead percentage. -/
theorem net_profit_calculation (purchase_price markup : ℕ) (overhead_percent : ℚ) : 
  purchase_price = 48 → 
  markup = 50 → 
  overhead_percent = 1/4 →
  markup - (overhead_percent * purchase_price).floor = 38 := by
  sorry

#check net_profit_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_net_profit_calculation_l864_86482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_constant_is_two_l864_86442

-- Define the type of functions from [0, 1] to ℝ
def UnitIntervalFunc := {f : ℝ → ℝ // ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≥ 0}

-- Define the properties of the function
def SatisfiesConditions (f : UnitIntervalFunc) : Prop :=
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f.val x ≥ 0) ∧
  (f.val 1 = 1) ∧
  (∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x + y ≤ 1 →
    f.val x + f.val y ≤ f.val (x + y))

-- State the theorem
theorem smallest_constant_is_two :
  (∃ c : ℝ, ∀ f : UnitIntervalFunc, SatisfiesConditions f →
    ∀ x, 0 ≤ x ∧ x ≤ 1 → f.val x ≤ c * x) ∧
  (∀ c : ℝ, (∀ f : UnitIntervalFunc, SatisfiesConditions f →
    ∀ x, 0 ≤ x ∧ x ≤ 1 → f.val x ≤ c * x) → c ≥ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_constant_is_two_l864_86442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_range_l864_86411

-- Define the circle C
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line L
def Line (x y : ℝ) : Prop := x - y - 2 = 0

-- Define the angle between two vectors
noncomputable def Angle (v1 v2 : ℝ × ℝ) : ℝ := 
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2)))

-- Main theorem
theorem point_range (x₀ y₀ : ℝ) :
  Line x₀ y₀ →
  (∃ (qx qy : ℝ), Circle qx qy ∧ Angle (qx - x₀, qy - y₀) (-x₀, -y₀) = π/6) →
  0 ≤ x₀ ∧ x₀ ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_range_l864_86411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_l864_86410

/-- The center of the hyperbola (3y-3)^2/7^2 - (4x-5)^2/3^2 = 1 is at (5/4, 1) -/
theorem hyperbola_center (x y : ℝ) :
  ((3 * y - 3)^2 / 7^2) - ((4 * x - 5)^2 / 3^2) = 1 →
  (x = 5/4 ∧ y = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_l864_86410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_y_x_l864_86452

-- Declare the section as noncomputable
noncomputable section

-- Use 'variable' instead of 'variables'
variable (t : ℝ)
variable (x y : ℝ → ℝ)

-- Define x
def x_def : x = λ t => Real.log (Real.tan t) := by sorry

-- Define y
def y_def : y = λ t => 1 / (Real.sin t)^2 := by sorry

-- Theorem for the derivative
theorem derivative_y_x :
  (deriv y / deriv x) t = -2 * (Real.tan t / Real.sin t)^2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_y_x_l864_86452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_complement_for_a_3_intersection_equals_N_iff_l864_86457

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 18 ≤ 0}
def N (a : ℝ) : Set ℝ := {x | 1 - a ≤ x ∧ x ≤ 2*a + 1}

-- Theorem for part (1)
theorem intersection_and_complement_for_a_3 :
  (M ∩ N 3) = Set.Icc (-2) 6 ∧
  (Set.univ \ N 3) = Set.Iio (-2) ∪ Set.Ioi 7 :=
sorry

-- Theorem for part (2)
theorem intersection_equals_N_iff (a : ℝ) :
  M ∩ N a = N a ↔ a ∈ Set.Iic (5/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_complement_for_a_3_intersection_equals_N_iff_l864_86457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_12_implies_k_90_l864_86418

/-- The sum of the infinite series (5 + nk) / 5^n for n = 0 to infinity -/
noncomputable def seriesSum (k : ℝ) : ℝ := 
  (5 : ℝ) / (1 - 1/5) + k * (1/5) / ((1 - 1/5)^2)

/-- Theorem stating that the value of k is 90 given the series sum equals 12 -/
theorem series_sum_equals_12_implies_k_90 : 
  seriesSum 90 = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_12_implies_k_90_l864_86418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_stretch_and_shift_l864_86421

/-- Given a function f obtained by horizontally stretching sin(x) by a factor of 3
    and then shifting it left by π/4 units, prove that f(x) = sin((x/3) + (π/12)) -/
theorem sin_stretch_and_shift :
  ∀ f : ℝ → ℝ, (∀ x, f x = Real.sin ((x + π/4) / 3)) →
  (∀ x, f x = Real.sin (x/3 + π/12)) :=
by
  intro f h
  intro x
  have : (x + π/4) / 3 = x/3 + π/12 := by
    field_simp
    ring
  rw [h, this]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_stretch_and_shift_l864_86421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_two_hours_l864_86407

/-- Represents a round trip with given speeds and time to work -/
structure RoundTrip where
  speed_to_work : ℝ
  speed_to_home : ℝ
  time_to_work_minutes : ℝ

/-- Calculates the total time for a round trip in hours -/
noncomputable def total_time (trip : RoundTrip) : ℝ :=
  let time_to_work_hours := trip.time_to_work_minutes / 60
  let distance := trip.speed_to_work * time_to_work_hours
  let time_to_home := distance / trip.speed_to_home
  time_to_work_hours + time_to_home

/-- Theorem stating that for the given conditions, the total round trip time is 2 hours -/
theorem round_trip_time_is_two_hours :
  let trip : RoundTrip := {
    speed_to_work := 70,
    speed_to_home := 105,
    time_to_work_minutes := 72
  }
  total_time trip = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_two_hours_l864_86407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_folding_angle_l864_86492

noncomputable section

structure Square where
  side_length : ℝ

structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

def ABCD : Square :=
  { side_length := 3 }

def A : Point :=
  { x := 0, y := 0 }

def B : Point :=
  { x := ABCD.side_length, y := 0 }

def C : Point :=
  { x := ABCD.side_length, y := ABCD.side_length }

def D : Point :=
  { x := 0, y := ABCD.side_length }

def E : Point :=
  { x := 1, y := 0 }

def X : Point :=
  { x := 1, y := 0 }

def Y : Point :=
  { x := 3, y := 2 }

theorem optimal_folding_angle :
  distance A E = 1 →
  distance A B = ABCD.side_length →
  distance B C = ABCD.side_length →
  distance C D = ABCD.side_length →
  distance D A = ABCD.side_length →
  X.x = E.x ∧ X.y = E.y →
  Y.x = B.x ∧ Y.y = 2 →
  Real.arctan (1 / 3) = Real.pi / 2 - Real.arctan ((D.y - E.y) / (D.x - E.x)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_folding_angle_l864_86492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_argument_and_real_constraint_l864_86479

open Complex

theorem complex_argument_and_real_constraint (θ : ℝ) (a : ℝ) 
  (h1 : 0 < θ ∧ θ < 2 * π)
  (h2 : let z : ℂ := 1 - cos θ + I * sin θ
        let u : ℂ := a^2 + a * I
        (z * u).re = 0) : 
  (let u : ℂ := a^2 + a * I
   (0 < θ ∧ θ < π → arg u = θ / 2) ∧
   (π < θ ∧ θ < 2 * π → arg u = π + θ / 2)) ∧
  (let z : ℂ := 1 - cos θ + I * sin θ
   let u : ℂ := a^2 + a * I
   let w : ℂ := z^2 + u^2 + 2 * z * u
   ¬∃ (x : ℝ), x > 0 ∧ w = (x : ℂ)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_argument_and_real_constraint_l864_86479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l864_86426

theorem expression_value (y : ℝ) : 4 * (2 * y + 3) - 8 * y + 9 = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l864_86426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l864_86480

def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 10}
def N : Set ℝ := {x : ℝ | x > 7 ∨ x < 1}

theorem intersection_of_M_and_N : M ∩ N = Set.Ioc (-1) 1 ∪ Set.Ioc 7 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l864_86480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tricia_age_l864_86459

theorem tricia_age (vincent amilia yorick eugene khloe rupert tricia : ℕ) : 
  vincent = 22 ∧ 
  rupert = vincent - 2 ∧ 
  khloe = rupert - 10 ∧ 
  khloe * 3 = eugene ∧ 
  yorick = 2 * eugene ∧ 
  amilia * 4 = yorick ∧ 
  tricia * 3 = amilia 
  → tricia = 5 := by
  intro h
  -- The proof steps would go here
  sorry

#check tricia_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tricia_age_l864_86459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_equals_one_l864_86444

theorem sin_minus_cos_equals_one (θ : Real) (a : Real) :
  θ ∈ Set.Ioo 0 Real.pi →
  (Real.sin θ)^2 - Real.sin θ + a = 0 →
  (Real.cos θ)^2 - Real.cos θ + a = 0 →
  Real.sin θ - Real.cos θ = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_equals_one_l864_86444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_not_in_third_quadrant_l864_86472

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2)

theorem function_not_in_third_quadrant :
  ∀ x y : ℝ, f x = y → ¬(x < 0 ∧ y < 0) :=
by
  intro x y h
  contrapose! h
  simp [f] at *
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_not_in_third_quadrant_l864_86472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_inequality_l864_86408

/-- A geometric sequence with sum S_n = 2^(n+1) + c -/
def geometric_sequence (c : ℝ) : ℕ → ℝ := fun n => 2^n

/-- The sum of the first n terms of the geometric sequence -/
def S_n (c : ℝ) (n : ℕ) : ℝ := 2^(n+1) + c

/-- The inequality condition for the sequence -/
def inequality_condition (lambda : ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n + 2 * (-1)^n < lambda * (a (n+1) + 2 * (-1)^(n+1))

theorem geometric_sequence_inequality (c : ℝ) (lambda : ℝ) :
  inequality_condition lambda (geometric_sequence c) → lambda > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_inequality_l864_86408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_8_equals_19_f_property_l864_86484

-- Define the function f
noncomputable def f : ℝ → ℝ := fun u => (u^2 + 8*u + 43) / 9

-- State the theorem
theorem f_8_equals_19 : f 8 = 19 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp [pow_two]
  -- Evaluate the numerical expression
  norm_num

-- State the property of f
theorem f_property (x : ℝ) : f (3*x - 1) = x^2 + 2*x + 4 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp [pow_two, mul_add, add_mul, mul_assoc, add_assoc]
  -- Perform algebraic manipulations
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_8_equals_19_f_property_l864_86484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l864_86443

-- Define the length of AB
noncomputable def AB_length : ℝ := Real.sqrt 2 + 1

-- Define the ratio of AP to PB
noncomputable def ratio : ℝ := Real.sqrt 2 / 2

-- Define the property that P is on AB, A is on x-axis, B is on y-axis
def on_segment (x y : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = AB_length^2 ∧
                 (x - x₀)^2 + y^2 = ratio^2 * (x^2 + (y₀ - y)^2)

-- Theorem statement
theorem trajectory_equation (x y : ℝ) (h : on_segment x y) :
  x^2 / 2 + y^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l864_86443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_area_bound_l864_86483

/-- A lattice point in the Cartesian coordinate plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- An m-lattice point is a lattice point whose coordinates are divisible by m -/
def is_m_lattice_point (p : LatticePoint) (m : ℕ) : Prop :=
  (m : ℤ) ∣ p.x ∧ (m : ℤ) ∣ p.y

/-- A lattice triangle in the Cartesian coordinate plane -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- The area of a lattice triangle -/
noncomputable def area (T : LatticeTriangle) : ℚ :=
  sorry

/-- A point is in the interior of a triangle -/
def is_interior_point (p : LatticePoint) (T : LatticeTriangle) : Prop :=
  sorry

/-- The main theorem -/
theorem exists_area_bound :
  ∃ (lambda : ℚ), ∀ (m : ℕ) (T : LatticeTriangle),
    m ≥ 2 →
    (∃! p : LatticePoint, is_interior_point p T ∧ is_m_lattice_point p m) →
    area T ≤ lambda * m^3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_area_bound_l864_86483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l864_86498

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^3 - b^2*c = a*b^2 - c^3 →
  1/2 * a * c * Real.sin B = Real.sqrt 3 / 6 →
  b = 4 →
  B = π/3 ∧ a + b + c = 4 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l864_86498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_probability_l864_86461

/-- Represents the outcome of a single round where no coins are transferred -/
def no_transfer_round : ℕ := 1

/-- Represents the total number of possible outcomes in a single round -/
def total_outcomes_per_round : ℕ := Nat.choose 4 2

/-- Represents the number of rounds in the game -/
def num_rounds : ℕ := 5

/-- The probability of no coin transfer in a single round -/
noncomputable def prob_no_transfer : ℚ := no_transfer_round / total_outcomes_per_round

/-- The probability that each player has exactly 6 coins at the end of the game -/
noncomputable def prob_no_change : ℚ := prob_no_transfer ^ num_rounds

theorem coin_game_probability :
  prob_no_change = 1 / 243 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_probability_l864_86461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_count_problem_l864_86439

/-- The number of cars counted by Jared, Ann, and Alfred -/
def total_cars (jared_count : ℕ) (ann_count : ℕ) (alfred_count : ℕ) : ℕ :=
  jared_count + ann_count + alfred_count

/-- Theorem stating the total number of cars counted given the problem conditions -/
theorem car_count_problem (jared_count ann_count alfred_count alfred_initial_count : ℕ) 
  (h1 : jared_count = 300)
  (h2 : (jared_count : ℚ) = 85 / 100 * ann_count)
  (h3 : ann_count = alfred_initial_count + 7)
  (h4 : (alfred_count : ℚ) = 112 / 100 * alfred_initial_count) :
  total_cars jared_count ann_count alfred_count = 1040 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_count_problem_l864_86439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duty_arrangements_5_6_l864_86432

/-- The number of ways to arrange duty for a unit --/
def dutyArrangements (n : ℕ) (d : ℕ) : ℕ :=
  let onePersonTwoDays := n * Nat.choose d 2
  let remainingArrangements := Nat.factorial (n - 1)
  onePersonTwoDays * remainingArrangements

/-- Theorem stating the number of duty arrangements for 5 people over 6 days --/
theorem duty_arrangements_5_6 :
  dutyArrangements 5 6 = 1800 := by
  sorry

#eval dutyArrangements 5 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_duty_arrangements_5_6_l864_86432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_lines_proof_l864_86491

-- Define the points A, B, C, and D
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (5, 4)
noncomputable def D : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the line l passing through A and D
def l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the line l₁ passing through B and parallel to l
def l₁ (x y : ℝ) : Prop := x - y + 3 = 0

-- Theorem statement
theorem triangle_lines_proof :
  (∀ x y, l x y ↔ x - y + 1 = 0) ∧
  (l₁ 0 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_lines_proof_l864_86491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_essay_score_l864_86475

theorem max_essay_score (t1 t2 t3 t4 : ℝ) : 
  t1 ≥ 0 → t2 ≥ 0 → t3 ≥ 0 → t4 ≥ 0 →
  t1 + t2 + t3 + t4 = 4 →
  ∃ (n1 n2 n3 n4 : ℕ), t1 = 0.5 * n1 ∧ t2 = 0.5 * n2 ∧ t3 = 0.5 * n3 ∧ t4 = 0.5 * n4 →
  (100 * (1 - (4:ℝ)^(-t1)) + 100 * (1 - (4:ℝ)^(-t2)) + 100 * (1 - (4:ℝ)^(-t3)) + 100 * (1 - (4:ℝ)^(-t4))) / 4 ≤ 75 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_essay_score_l864_86475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_defective_l864_86433

theorem probability_at_least_one_defective (total : ℕ) (defective : ℕ) : 
  total = 21 → defective = 4 → 
  (1 - (((total - defective : ℚ) / total) * ((total - defective - 1) / (total - 1)))) = 37 / 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_defective_l864_86433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l864_86494

noncomputable def train_length : ℝ := 400  -- meters
noncomputable def crossing_time : ℝ := 8   -- seconds

noncomputable def speed_ms : ℝ := train_length / crossing_time
noncomputable def speed_kmh : ℝ := speed_ms * 3.6

theorem train_speed : speed_kmh = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l864_86494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_ln_x_div_y_nonneg_neg_sum_x_ln_x_le_ln_n_sum_x_ln_x_div_y_ge_x_ln_x_div_y_l864_86415

open Real BigOperators

variable {n : ℕ}
variable (x y : Fin n → ℝ)

-- Conditions
axiom x_pos : ∀ i, x i > 0
axiom y_pos : ∀ i, y i > 0
axiom sum_x_eq_one : ∑ i, x i = 1
axiom sum_y_eq_one : ∑ i, y i = 1

-- Definitions for part 3
def x_sum (x : Fin n → ℝ) : ℝ := ∑ i, x i
def y_sum (y : Fin n → ℝ) : ℝ := ∑ i, y i

-- Theorems to prove
theorem sum_x_ln_x_div_y_nonneg : 
  ∑ i, x i * log (x i / y i) ≥ 0 := by sorry

theorem neg_sum_x_ln_x_le_ln_n : 
  -(∑ i, x i * log (x i)) ≤ log n := by sorry

theorem sum_x_ln_x_div_y_ge_x_ln_x_div_y : 
  ∑ i, x i * log (x i / y i) ≥ x_sum x * log (x_sum x / y_sum y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_ln_x_div_y_nonneg_neg_sum_x_ln_x_le_ln_n_sum_x_ln_x_div_y_ge_x_ln_x_div_y_l864_86415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_range_of_m_l864_86493

open Real

noncomputable def f (x : ℝ) := sin (π / 3 - 2 * x)
noncomputable def g (m : ℝ) (x : ℝ) := x^2 - 2*x + m - 3

def is_monotonic_on (f : ℝ → ℝ) (a b : ℝ) (increasing : Bool) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (if increasing then f x ≤ f y else f y ≤ f x)

theorem f_monotonicity (k : ℤ) :
  (is_monotonic_on f (-π/12 + k*π) (5*π/12 + k*π) false) ∧
  (is_monotonic_on f (5*π/12 + k*π) (11*π/12 + k*π) true) :=
by sorry

theorem range_of_m :
  ∀ m : ℝ,
  (∀ x₁ : ℝ, π/12 ≤ x₁ ∧ x₁ ≤ π/2 →
    ∃ x₂ : ℝ, -2 ≤ x₂ ∧ x₂ ≤ m ∧ f x₁ = g m x₂) ↔
  -1 ≤ m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_range_of_m_l864_86493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l864_86451

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := Real.sin (7 * Real.pi / 6 - 2 * x) - 2 * Real.sin x ^ 2 + 1

/-- Triangle ABC -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The main theorem to prove -/
theorem triangle_side_length 
  (abc : Triangle) 
  (h1 : f abc.A = 1/2)
  (h2 : abc.b - abc.a = abc.c - abc.b)  -- arithmetic sequence
  (h3 : abc.b * abc.c * Real.cos abc.A = 9) : 
  abc.a = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l864_86451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_samson_should_buy_card_average_customer_breakeven_l864_86425

/-- Represents the loyalty card system of the Italian restaurant "Pablo's" --/
structure LoyaltyCard where
  cost : ℕ
  discount : ℚ
  deriving Repr

/-- Represents a customer of the restaurant --/
structure Customer where
  visitsPerWeek : ℕ
  averageCheck : ℕ
  deriving Repr

def weeksPerYear : ℕ := 52

/-- Calculates the yearly savings for a customer with a loyalty card --/
def yearlySavings (card : LoyaltyCard) (customer : Customer) : ℚ :=
  let visitsPerYear := customer.visitsPerWeek * weeksPerYear
  let savingsPerVisit := (customer.averageCheck : ℚ) * card.discount
  savingsPerVisit * (visitsPerYear : ℚ)

/-- Calculates the number of visits required to break even --/
def breakevenVisits (card : LoyaltyCard) (averageCheck : ℕ) : ℕ :=
  let savingsPerVisit := (averageCheck : ℚ) * card.discount
  ((card.cost : ℚ) / savingsPerVisit).ceil.toNat

/-- Theorem stating that it's beneficial for Samson to buy the loyalty card --/
theorem samson_should_buy_card (card : LoyaltyCard) (samson : Customer) :
    card.cost = 30000 →
    card.discount = 0.3 →
    samson.visitsPerWeek = 3 →
    samson.averageCheck = 900 →
    yearlySavings card samson > (card.cost : ℚ) := by
  sorry

/-- Theorem stating the number of visits required for an average customer to break even --/
theorem average_customer_breakeven (card : LoyaltyCard) :
    card.cost = 30000 →
    card.discount = 0.3 →
    breakevenVisits card 600 = 167 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_samson_should_buy_card_average_customer_breakeven_l864_86425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_totient_divisibility_gcd_count_l864_86455

-- Define the Euler totient function
noncomputable def φ : ℕ → ℕ := sorry

-- Part (a)
theorem euler_totient_divisibility (m n : ℕ) (h1 : m > 1) (h2 : n > 0) :
  n ∣ φ (m^n - 1) := by sorry

-- Part (b)
theorem gcd_count (n d : ℕ) (h : d > 0) (h2 : d ∣ n) :
  (Finset.filter (λ k => Nat.gcd k n = d) (Finset.range n)).card = φ (n / d) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_euler_totient_divisibility_gcd_count_l864_86455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_l864_86497

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 1) :
  (fun (z : ℂ) => Complex.abs (z + 3 - 4*Complex.I)) z ≤ 6 ∧
  ∃ (w : ℂ), Complex.abs w = 1 ∧ Complex.abs (w + 3 - 4*Complex.I) = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_complex_l864_86497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_expressions_l864_86404

-- Define the condition function
def condition (x y : ℝ) : Prop := x * y - x / y - y / x = 5

-- Define the expression (x - 2)(y - 2)
def expression (x y : ℝ) : ℝ := (x - 2) * (y - 2)

-- Theorem statement
theorem sum_of_expressions :
  ∃ (S : Finset ℝ), (∀ z ∈ S, ∃ a b : ℝ, condition a b ∧ z = expression a b) ∧
                    (S.sum id = 41) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_expressions_l864_86404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_point_l864_86419

-- Define the function f(x) = ln x - x + 1
noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

-- State the theorem
theorem f_has_unique_zero_point :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

#check f_has_unique_zero_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_point_l864_86419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_f_range_eq_l864_86445

noncomputable def f (x : ℝ) : ℝ := 
  (Real.arccos (x/2))^2 + Real.pi * Real.arcsin (x/2) - (Real.arcsin (x/2))^2 + (Real.pi^2/12) * (x^2 + 6*x + 8)

theorem f_range : 
  ∀ y ∈ Set.range f, Real.pi^2/4 ≤ y ∧ y ≤ 9*Real.pi^2/4 ∧
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-2) 2 ∧ x₂ ∈ Set.Icc (-2) 2 ∧ 
    f x₁ = Real.pi^2/4 ∧ f x₂ = 9*Real.pi^2/4 := by
  sorry

def range_set : Set ℝ := {y | Real.pi^2/4 ≤ y ∧ y ≤ 9*Real.pi^2/4}

theorem f_range_eq : Set.range f = range_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_f_range_eq_l864_86445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_characterization_l864_86417

open Set Real

noncomputable def B : ℝ × ℝ := (1, 0)
noncomputable def C : ℝ × ℝ := (-1, 0)

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * abs (x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂))

def S : Set (ℝ × ℝ) :=
  {A | triangle_area A B C = 2}

theorem S_characterization :
  S = {A : ℝ × ℝ | A.2 = 2 ∨ A.2 = -2} := by
  sorry

#check S_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_characterization_l864_86417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l864_86470

/-- Represents the time (in seconds) it takes for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem: A train 100 meters long, traveling at 36 kmph, will take 30 seconds to cross a bridge 200 meters long -/
theorem train_crossing_bridge :
  train_crossing_time 100 200 36 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l864_86470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_ab_length_l864_86406

/-- Given a right triangle ABC where ∠A = 90°, BC = 20, and tan C = 4 sin B, prove that AB = 5√15 -/
theorem right_triangle_ab_length (A B C : ℝ) (h_right_angle : A = 90) (h_bc_length : Real.sqrt (B^2 + C^2) = 20) 
  (h_tan_sin_relation : Real.tan C = 4 * Real.sin B) : Real.sqrt (B^2 + C^2) = 5 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_ab_length_l864_86406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l864_86478

def a : ℕ → ℝ
  | 0 => 2  -- Added case for n = 0
  | n + 1 => 2 * a n + 3 * 5^n

theorem sequence_formula (n : ℕ) : a n = 5^n - 3 * 2^(n-1) := by
  induction n with
  | zero => 
    simp [a]
    -- Proof for base case
    sorry
  | succ n ih =>
    simp [a]
    -- Inductive step
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l864_86478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_password_l864_86460

def is_valid_password (password : List Nat) : Prop :=
  password.length = 5 ∧
  (∀ i : Fin 4, password[i.val + 1]! - password[i.val]! = password[1]! - password[0]!) ∧
  (∀ n ∈ password, n < 100) ∧
  (∀ i : Fin 4, password[i.val + 1]! > password[i.val]!)

theorem unique_password : 
  ∀ password : List Nat, 
    is_valid_password password ↔ password = [5, 12, 19, 26, 33] :=
by
  intro password
  constructor
  · intro h
    sorry -- Proof of forward direction
  · intro h
    sorry -- Proof of backward direction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_password_l864_86460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_problem_l864_86446

-- Define a and b as noncomputable
noncomputable def a : ℝ := Real.log 6 / Real.log 2
noncomputable def b : ℝ := Real.log 36 / Real.log 3

-- State the theorem
theorem logarithm_problem :
  (2 : ℝ) ^ a = 6 ∧ (3 : ℝ) ^ b = 36 →
  (4 : ℝ) ^ a / (9 : ℝ) ^ b = 1 / 36 ∧
  1 / a + 2 / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_problem_l864_86446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_T_l864_86481

-- Define the complex number η
def η : ℂ := Complex.I

-- Define the set of points forming the region T
def T : Set ℂ :=
  {z : ℂ | ∃ (a b c : ℝ), 
    z = a + b * η + c * η^2 ∧
    0 ≤ a ∧ a ≤ 2 ∧
    0 ≤ b ∧ b ≤ 2 ∧
    0 ≤ c ∧ c ≤ 1}

-- State the theorem about the area of region T
theorem area_of_T : MeasureTheory.volume T = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_T_l864_86481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_constant_l864_86449

/-- Regular n-gon -/
structure RegularNGon where
  n : ℕ
  a : ℝ
  S : ℝ

/-- Point inside the n-gon -/
structure InteriorPoint (ngon : RegularNGon) where
  x : ℝ
  y : ℝ

/-- Perpendicular distance from a point to a side of the n-gon -/
noncomputable def perpDistance (ngon : RegularNGon) (p : InteriorPoint ngon) (sideIndex : Fin ngon.n) : ℝ :=
  sorry

theorem sum_distances_constant (ngon : RegularNGon) (p : InteriorPoint ngon) :
  (Finset.univ : Finset (Fin ngon.n)).sum (fun i => perpDistance ngon p i) = 2 * ngon.S / ngon.a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_constant_l864_86449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_root_pair_l864_86469

theorem complex_root_pair (z : ℂ) : 
  z^2 = -75 + 65*I → (4 + 9*I : ℂ) ^ 2 = -75 + 65*I → 
  z = 4 + 9*I ∨ z = -4 - 9*I := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_root_pair_l864_86469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metaphase_visible_structures_option_A_is_correct_l864_86490

/-- Represents the visible structures in a plant cell during metaphase of mitosis -/
inductive VisibleStructure
  | Chromosomes
  | Spindle
  | CellWall
  | MetaphasePlate
  | CellMembrane
  | Nucleus
  | Nucleolus

/-- Represents the correct answer to the question -/
def correct_answer : List VisibleStructure :=
  [VisibleStructure.Chromosomes, VisibleStructure.Spindle, VisibleStructure.CellWall]

/-- Theorem stating that the correct answer contains the visible structures during metaphase -/
theorem metaphase_visible_structures :
  (s : List VisibleStructure) →
  s = correct_answer →
  (VisibleStructure.Chromosomes ∈ s ∧
   VisibleStructure.Spindle ∈ s ∧
   VisibleStructure.CellWall ∈ s) :=
by
  intro s h
  rw [h]
  simp [correct_answer]

/-- Proof that option A is the correct answer -/
theorem option_A_is_correct :
  correct_answer = [VisibleStructure.Chromosomes, VisibleStructure.Spindle, VisibleStructure.CellWall] :=
by
  rfl

#check metaphase_visible_structures
#check option_A_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_metaphase_visible_structures_option_A_is_correct_l864_86490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chair_makes_six_sticks_l864_86458

/-- The number of sticks of wood a chair makes -/
def chair_sticks : ℕ := 6

/-- The number of sticks of wood a table makes -/
def table_sticks : ℕ := 9

/-- The number of sticks of wood a stool makes -/
def stool_sticks : ℕ := 2

/-- The number of sticks of wood Mary needs to burn per hour -/
def sticks_per_hour : ℕ := 5

/-- The number of chairs Mary chopped up -/
def chairs_chopped : ℕ := 18

/-- The number of tables Mary chopped up -/
def tables_chopped : ℕ := 6

/-- The number of stools Mary chopped up -/
def stools_chopped : ℕ := 4

/-- The number of hours Mary can keep warm -/
def hours_warm : ℕ := 34

theorem chair_makes_six_sticks :
  chair_sticks * chairs_chopped + table_sticks * tables_chopped + stool_sticks * stools_chopped =
  sticks_per_hour * hours_warm := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chair_makes_six_sticks_l864_86458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_A_l864_86401

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.cos x + Real.sqrt 3 * Real.sin x) - 1

-- Define the theorem
theorem triangle_angle_A (a b c : ℝ) (A B C : ℝ) :
  f (C / 2) = 2 →
  a * b = c^2 →
  A = π / 3 :=
by
  -- The proof is skipped using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_A_l864_86401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_outfit_percentage_increase_l864_86437

/-- The cost of John's pants in dollars -/
noncomputable def pants_cost : ℝ := 50

/-- The total cost of John's outfit (shirt + pants) in dollars -/
noncomputable def total_cost : ℝ := 130

/-- The cost of John's shirt in dollars -/
noncomputable def shirt_cost : ℝ := total_cost - pants_cost

/-- The percentage increase from pants cost to shirt cost -/
noncomputable def percentage_increase : ℝ := (shirt_cost - pants_cost) / pants_cost * 100

theorem johns_outfit_percentage_increase :
  percentage_increase = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_outfit_percentage_increase_l864_86437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l864_86485

theorem empty_subset_singleton_zero : ∅ ⊆ ({0} : Set ℕ) := by
  intro x
  intro h
  exfalso
  exact Set.not_mem_empty x h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_subset_singleton_zero_l864_86485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_without_parentheses_l864_86440

theorem expression_without_parentheses :
  (-2/3 : ℚ) + (-7/6 : ℚ) - (-3/4 : ℚ) - (1/4 : ℚ) = -2/3 - 7/6 + 3/4 - 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_without_parentheses_l864_86440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_identifiable_l864_86402

/-- Represents the result of a weighing --/
inductive WeighResult
| Left  : WeighResult  -- Left pan is heavier
| Right : WeighResult  -- Right pan is heavier
| Equal : WeighResult  -- Pans are balanced or appear balanced

/-- Represents a coin --/
structure Coin where
  id : Nat
  weight : Nat

/-- Represents the state of knowledge about coins --/
structure CoinState where
  coins : List Coin
  counterfeit : Option Coin
  isLighter : Option Bool

/-- Represents a balance scale with potentially inaccurate equal weighing --/
def balance (left right : List Coin) : WeighResult :=
  sorry

/-- Represents a weighing strategy --/
def Strategy := CoinState → Option (List Coin × List Coin)

/-- Represents the process of identifying the counterfeit coin --/
def identifyCounterfeit (s : Strategy) (initialState : CoinState) : CoinState :=
  sorry

/-- Main theorem: It's possible to identify the counterfeit coin and its relative weight --/
theorem counterfeit_identifiable :
  ∀ (coins : List Coin),
    coins.length = 5 →
    (∃! c, c ∈ coins ∧ ∃ w, c.weight ≠ w ∧ ∀ c' ∈ coins, c' ≠ c → c'.weight = w) →
    ∃ (s : Strategy),
      let finalState := identifyCounterfeit s ⟨coins, none, none⟩
      finalState.counterfeit.isSome ∧ finalState.isLighter.isSome :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_identifiable_l864_86402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_measurement_comparison_l864_86405

-- Define the measurements
noncomputable def line1_length : ℝ := 25
noncomputable def line1_error : ℝ := 0.05
noncomputable def line2_length : ℝ := 200
noncomputable def line2_error : ℝ := 0.5

-- Define relative error
noncomputable def relative_error (error : ℝ) (length : ℝ) : ℝ := error / length * 100

-- Define absolute error increase
noncomputable def absolute_error_increase : ℝ := line2_error - line1_error

theorem measurement_comparison :
  (relative_error line2_error line2_length > relative_error line1_error line1_length) ∧
  (absolute_error_increase = 0.45) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_measurement_comparison_l864_86405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_pool_width_l864_86462

/-- Represents the dimensions and volume of a swimming pool -/
structure SwimmingPool where
  length : ℝ
  shallowDepth : ℝ
  deepDepth : ℝ
  volume : ℝ

/-- Calculates the width of a swimming pool given its dimensions and volume -/
noncomputable def calculateWidth (pool : SwimmingPool) : ℝ :=
  (2 * pool.volume) / ((pool.shallowDepth + pool.deepDepth) * pool.length)

/-- Theorem stating that a swimming pool with given dimensions has a width of 9 meters -/
theorem swimming_pool_width :
  let pool : SwimmingPool := {
    length := 12,
    shallowDepth := 1,
    deepDepth := 4,
    volume := 270
  }
  calculateWidth pool = 9 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_pool_width_l864_86462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_symmetry_condition_area_maximization_l864_86488

-- Define the line l: x + λy - 2λ - 1 = 0
def line (lambda : ℝ) (x y : ℝ) : Prop := x + lambda * y - 2 * lambda - 1 = 0

-- Define the circle C: x² + y² = 1
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Maximum distance from origin to line
theorem max_distance_to_line :
  ∃ (lambda : ℝ), ∀ (mu : ℝ), 
    (∀ (x y : ℝ), line mu x y → (x^2 + y^2 : ℝ) ≤ 5) ∧
    (∃ (x y : ℝ), line lambda x y ∧ x^2 + y^2 = 5) :=
sorry

-- Symmetry condition
theorem symmetry_condition :
  ∀ (lambda : ℝ), (∀ (x y : ℝ), circle_C x y ↔ circle_C (x + 2*lambda*x/(1+lambda^2)) (y - 2*lambda*y/(1+lambda^2))) ↔ lambda = -1/2 :=
sorry

-- Area maximization
theorem area_maximization :
  ∀ (lambda : ℝ), (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line lambda x₁ y₁ ∧ line lambda x₂ y₂ ∧ 
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    ∀ (mu : ℝ) (x₃ y₃ x₄ y₄ : ℝ),
      line mu x₃ y₃ ∧ line mu x₄ y₄ ∧ 
      circle_C x₃ y₃ ∧ circle_C x₄ y₄ →
      x₁ * y₂ - x₂ * y₁ ≥ x₃ * y₄ - x₄ * y₃) ↔ 
  (lambda = -1/7 ∨ lambda = -1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_symmetry_condition_area_maximization_l864_86488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_1000_trailing_zeros_l864_86468

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (List.range 4).foldl (fun acc k => acc + n / (5 ^ (k + 1))) 0

/-- 1000! ends with 249 zeros -/
theorem factorial_1000_trailing_zeros :
  trailingZeros 1000 = 249 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_1000_trailing_zeros_l864_86468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ba_in_BaO_approx_l864_86453

/-- The mass percentage of Ba in BaO -/
noncomputable def mass_percentage_Ba_in_BaO (molar_mass_Ba : ℝ) (molar_mass_O : ℝ) : ℝ :=
  let molar_mass_BaO := molar_mass_Ba + molar_mass_O
  (molar_mass_Ba / molar_mass_BaO) * 100

/-- Theorem stating that the mass percentage of Ba in BaO is approximately 89.55% -/
theorem mass_percentage_Ba_in_BaO_approx :
  let molar_mass_Ba : ℝ := 137.33
  let molar_mass_O : ℝ := 16.00
  abs (mass_percentage_Ba_in_BaO molar_mass_Ba molar_mass_O - 89.55) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ba_in_BaO_approx_l864_86453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perimeters_l864_86412

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The perimeter of a triangle given three points -/
noncomputable def trianglePerimeter (a b c : Point) : ℝ :=
  distance a b + distance b c + distance c a

/-- Theorem stating the sum of all possible perimeters of triangle AFG -/
theorem sum_of_perimeters (a e f g : Point) : 
  (distance a e = 8) →
  (distance e f = 18) →
  (distance a g = distance f g) →
  (∃ (n : ℕ), distance a g = n ∧ distance e g = n) →
  (∃ (t : ℝ), t = (trianglePerimeter a f g + trianglePerimeter a f g) ∧ t = 220) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_perimeters_l864_86412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l864_86423

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := lg (x + 1)

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | 0 < f (1 - 2*x) - f x ∧ f (1 - 2*x) - f x < 1} = {x : ℝ | -2/3 < x ∧ x < 1/3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l864_86423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l864_86473

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, Real.sin α)

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (-t + 2, t)

-- Theorem for the cartesian equations and intersection length
theorem curve_line_intersection :
  -- Cartesian equation of C
  (∀ x y : ℝ, (∃ α : ℝ, curve_C α = (x, y)) ↔ x^2/9 + y^2 = 1) ∧
  -- Cartesian equation of l
  (∀ x y : ℝ, (∃ t : ℝ, line_l t = (x, y)) ↔ x + y = 2) ∧
  -- Length of AB
  (∃ A B : ℝ × ℝ,
    (∃ α : ℝ, curve_C α = A) ∧
    (∃ t : ℝ, line_l t = A) ∧
    (∃ β : ℝ, curve_C β = B) ∧
    (∃ s : ℝ, line_l s = B) ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6 * Real.sqrt 3 / 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l864_86473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_half_sector_l864_86448

/-- The volume of a right circular cone formed by rolling up a half-sector of a circle --/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 5) : 
  (1/3) * π * (5*π/(2*π))^2 * (Real.sqrt (25 - (5*π/(2*π))^2)) = (125 * Real.sqrt 3 * π) / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_half_sector_l864_86448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l864_86471

theorem trig_identities (x : ℝ) (h : Real.tan x = 2) :
  ((Real.cos x + Real.sin x) / (Real.cos x - Real.sin x) = -3) ∧
  (2 * Real.sin x ^ 2 - Real.sin x * Real.cos x + Real.cos x ^ 2 = 7/5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l864_86471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l864_86429

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ ∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧
  (∀ x : ℝ, f (Real.pi / 3 + x) = f (Real.pi / 3 - x)) ∧
  (∀ x y : ℝ, 5 * Real.pi / 6 ≤ x ∧ x < y ∧ y ≤ Real.pi → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l864_86429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_8x_div_9_l864_86416

theorem remainder_of_8x_div_9 (x : ℕ) (h : x % 9 = 5) : (8 * x) % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_8x_div_9_l864_86416
