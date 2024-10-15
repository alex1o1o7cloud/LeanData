import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_condition_range_l2810_281034

theorem ellipse_condition_range (m a : ℝ) : 
  (a > 0) →
  (m^2 + 12*a^2 < 7*a*m) →
  (∀ x y : ℝ, x^2 / (m - 1) + y^2 / (2 - m) = 1 → 
    ∃ c : ℝ, c > 0 ∧ ∀ p : ℝ × ℝ, p.1 = 0 → 
      (p.2 - c)^2 + p.1^2 = (m - 1)^2 ∨ (p.2 + c)^2 + p.1^2 = (m - 1)^2) →
  (∀ m : ℝ, (m^2 + 12*a^2 < 7*a*m) → 
    (∃ x y : ℝ, x^2 / (m - 1) + y^2 / (2 - m) = 1 ∧
      ∃ c : ℝ, c > 0 ∧ ∀ p : ℝ × ℝ, p.1 = 0 → 
        (p.2 - c)^2 + p.1^2 = (m - 1)^2 ∨ (p.2 + c)^2 + p.1^2 = (m - 1)^2)) →
  (∃ m : ℝ, (m^2 + 12*a^2 < 7*a*m) ∧ 
    ¬(∃ x y : ℝ, x^2 / (m - 1) + y^2 / (2 - m) = 1 ∧
      ∃ c : ℝ, c > 0 ∧ ∀ p : ℝ × ℝ, p.1 = 0 → 
        (p.2 - c)^2 + p.1^2 = (m - 1)^2 ∨ (p.2 + c)^2 + p.1^2 = (m - 1)^2)) →
  a ∈ Set.Icc (1/3 : ℝ) (3/8 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_ellipse_condition_range_l2810_281034


namespace NUMINAMATH_CALUDE_jake_balloons_l2810_281066

theorem jake_balloons (total : ℕ) (allan_extra : ℕ) (h1 : total = 56) (h2 : allan_extra = 8) :
  ∃ (jake : ℕ), jake + (jake + allan_extra) = total ∧ jake = 24 :=
by sorry

end NUMINAMATH_CALUDE_jake_balloons_l2810_281066


namespace NUMINAMATH_CALUDE_largest_satisfying_number_l2810_281070

/-- A function that returns all possible two-digit numbers from three digits -/
def twoDigitNumbers (a b c : Nat) : List Nat :=
  [10*a+b, 10*a+c, 10*b+a, 10*b+c, 10*c+a, 10*c+b]

/-- The property that a three-digit number satisfies the given conditions -/
def satisfiesCondition (n : Nat) : Prop :=
  ∃ a b c : Nat,
    n = 100*a + 10*b + c ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (twoDigitNumbers a b c).sum = n

theorem largest_satisfying_number :
  satisfiesCondition 396 ∧
  ∀ m : Nat, satisfiesCondition m → m ≤ 396 :=
sorry

end NUMINAMATH_CALUDE_largest_satisfying_number_l2810_281070


namespace NUMINAMATH_CALUDE_incenter_x_coordinate_is_one_l2810_281075

/-- The triangle formed by the x-axis, y-axis, and the line x + y = 2 -/
structure Triangle where
  A : ℝ × ℝ := (0, 2)  -- y-intercept
  B : ℝ × ℝ := (2, 0)  -- x-intercept
  O : ℝ × ℝ := (0, 0)  -- origin

/-- The incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- The distance between a point and a line -/
def distancePointToLine (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ := sorry

theorem incenter_x_coordinate_is_one (t : Triangle) :
  (incenter t).1 = 1 ∧
  distancePointToLine (incenter t) (fun x => 0) =
  distancePointToLine (incenter t) (fun x => x) ∧
  distancePointToLine (incenter t) (fun x => 0) =
  distancePointToLine (incenter t) (fun x => 2 - x) :=
sorry

end NUMINAMATH_CALUDE_incenter_x_coordinate_is_one_l2810_281075


namespace NUMINAMATH_CALUDE_jennifer_apples_l2810_281008

/-- The number of apples Jennifer started with -/
def initial_apples : ℕ := sorry

/-- The number of apples Jennifer found -/
def found_apples : ℕ := 74

/-- The total number of apples Jennifer ended up with -/
def total_apples : ℕ := 81

/-- Theorem stating that the initial number of apples plus the found apples equals the total apples -/
theorem jennifer_apples : initial_apples + found_apples = total_apples := by sorry

end NUMINAMATH_CALUDE_jennifer_apples_l2810_281008


namespace NUMINAMATH_CALUDE_blue_face_prob_five_eighths_l2810_281067

/-- A regular octahedron with colored faces -/
structure ColoredOctahedron where
  blue_faces : ℕ
  red_faces : ℕ
  total_faces : ℕ
  total_is_sum : total_faces = blue_faces + red_faces
  total_is_eight : total_faces = 8

/-- The probability of rolling a blue face on a colored octahedron -/
def blue_face_probability (o : ColoredOctahedron) : ℚ :=
  o.blue_faces / o.total_faces

/-- Theorem: The probability of rolling a blue face on an octahedron with 5 blue faces and 3 red faces is 5/8 -/
theorem blue_face_prob_five_eighths (o : ColoredOctahedron) 
    (h1 : o.blue_faces = 5) 
    (h2 : o.red_faces = 3) : 
    blue_face_probability o = 5 / 8 := by
  sorry

#check blue_face_prob_five_eighths

end NUMINAMATH_CALUDE_blue_face_prob_five_eighths_l2810_281067


namespace NUMINAMATH_CALUDE_triangle_inequality_l2810_281086

/-- Given a triangle with sides a, b, c and area S, prove that a^2 + b^2 + c^2 ≥ 4S√3,
    with equality if and only if the triangle is equilateral -/
theorem triangle_inequality (a b c S : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0)
  (h_S : S = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  a^2 + b^2 + c^2 ≥ 4 * S * Real.sqrt 3 ∧
  (a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3 ↔ a = b ∧ b = c) := by
  sorry


end NUMINAMATH_CALUDE_triangle_inequality_l2810_281086


namespace NUMINAMATH_CALUDE_product_of_sum_and_difference_l2810_281065

theorem product_of_sum_and_difference (x y : ℝ) : 
  x + y = 15 ∧ x - y = 11 → x * y = 26 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_difference_l2810_281065


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2810_281011

/-- An isosceles triangle with congruent sides of 7 cm and perimeter of 21 cm has a base of 7 cm. -/
theorem isosceles_triangle_base_length : 
  ∀ (base congruent_side : ℝ),
  congruent_side = 7 →
  base + 2 * congruent_side = 21 →
  base = 7 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2810_281011


namespace NUMINAMATH_CALUDE_minimum_books_in_library_l2810_281050

theorem minimum_books_in_library (physics chemistry biology : ℕ) : 
  physics + chemistry + biology > 0 →
  3 * chemistry = 2 * physics →
  4 * biology = 3 * chemistry →
  ∃ (k : ℕ), k * (physics + chemistry + biology) = 3003 →
  3003 ≤ physics + chemistry + biology :=
by sorry

end NUMINAMATH_CALUDE_minimum_books_in_library_l2810_281050


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_minus_one_l2810_281094

theorem sum_of_reciprocals_of_roots_minus_one (p q r : ℂ) : 
  (p^3 - p - 2 = 0) → (q^3 - q - 2 = 0) → (r^3 - r - 2 = 0) →
  (1 / (p - 1) + 1 / (q - 1) + 1 / (r - 1) = -2) := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_minus_one_l2810_281094


namespace NUMINAMATH_CALUDE_perpendicular_vectors_and_angle_l2810_281055

theorem perpendicular_vectors_and_angle (θ φ : ℝ) : 
  (0 < θ) → (θ < π) →
  (π / 2 < φ) → (φ < π) →
  (2 * Real.cos θ + Real.sin θ = 0) →
  (Real.sin (θ - φ) = Real.sqrt 10 / 10) →
  (Real.tan θ = -2 ∧ Real.cos φ = -(Real.sqrt 2 / 10)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_and_angle_l2810_281055


namespace NUMINAMATH_CALUDE_Q_greater_than_P_l2810_281036

/-- A number consisting of 2010 digits of 1 -/
def a : ℕ := 10^2010 - 1

/-- P defined as the product of 2010 digits of 8 and 2010 digits of 3 -/
def P : ℕ := (8 * a) * (3 * a)

/-- Q defined as the product of 2010 digits of 4 and (2009 digits of 6 followed by 7) -/
def Q : ℕ := (4 * a) * (6 * a + 1)

/-- Theorem stating that Q is greater than P -/
theorem Q_greater_than_P : Q > P := by
  sorry

end NUMINAMATH_CALUDE_Q_greater_than_P_l2810_281036


namespace NUMINAMATH_CALUDE_triangle_side_length_l2810_281058

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  -- No specific conditions needed here

-- Define a point on a line segment
def PointOnSegment (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B

-- Define perpendicularity
def Perpendicular (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0

-- Define equality of distances
def EqualDistances (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (D.1 - C.1)^2 + (D.2 - C.2)^2

-- Main theorem
theorem triangle_side_length 
  (P Q R E G : ℝ × ℝ) 
  (triangle : Triangle P Q R) 
  (e_on_pq : PointOnSegment P Q E)
  (g_on_pr : PointOnSegment P R G)
  (pq_perp_pr : Perpendicular P Q P R)
  (pg_perp_pr : Perpendicular P G P R)
  (qe_eq_eg : EqualDistances Q E E G)
  (eg_eq_gr : EqualDistances E G G R)
  (gr_eq_3 : EqualDistances G R P (P.1 + 3, P.2)) :
  EqualDistances P R P (P.1 + 6, P.2) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2810_281058


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2810_281023

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 2 → a^2 > 2*a) ∧ 
  (∃ a, a ≤ 2 ∧ a^2 > 2*a) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2810_281023


namespace NUMINAMATH_CALUDE_president_secretary_choice_count_l2810_281041

/-- Represents the number of ways to choose a president and secretary of the same gender -/
def choose_president_and_secretary (total_members : ℕ) (boys : ℕ) (girls : ℕ) : ℕ :=
  (boys * (boys - 1)) + (girls * (girls - 1))

/-- Theorem: Given a society of 25 members (15 boys and 10 girls), 
    the number of ways to choose a president and a secretary of the same gender, 
    where no one can hold both positions, is equal to 300. -/
theorem president_secretary_choice_count :
  choose_president_and_secretary 25 15 10 = 300 := by
  sorry

end NUMINAMATH_CALUDE_president_secretary_choice_count_l2810_281041


namespace NUMINAMATH_CALUDE_orthogonal_equal_magnitude_vectors_l2810_281053

/-- Given two vectors a and b in R^3, prove that if they are orthogonal and have equal magnitude,
    then their components satisfy specific values. -/
theorem orthogonal_equal_magnitude_vectors 
  (a b : ℝ × ℝ × ℝ) 
  (h_a : a.1 = 4 ∧ a.2.2 = -2) 
  (h_b : b.1 = 1 ∧ b.2.1 = 2) 
  (h_orthogonal : a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = 0) 
  (h_equal_magnitude : a.1^2 + a.2.1^2 + a.2.2^2 = b.1^2 + b.2.1^2 + b.2.2^2) :
  a.2.1 = 11/4 ∧ b.2.2 = 19/4 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_equal_magnitude_vectors_l2810_281053


namespace NUMINAMATH_CALUDE_y_derivative_l2810_281044

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := Real.log (1 / Real.sqrt (1 + x^2))

-- State the theorem
theorem y_derivative (x : ℝ) : 
  deriv y x = -x / (1 + x^2) := by sorry

end NUMINAMATH_CALUDE_y_derivative_l2810_281044


namespace NUMINAMATH_CALUDE_ac_value_l2810_281020

theorem ac_value (x : ℕ+) 
  (h1 : ∃ y : ℕ, (2 * x + 1 : ℕ) = y^2)
  (h2 : ∃ z : ℕ, (3 * x + 1 : ℕ) = z^2) : 
  x = 40 := by
sorry

end NUMINAMATH_CALUDE_ac_value_l2810_281020


namespace NUMINAMATH_CALUDE_complex_equation_product_l2810_281027

/-- Given (1+3i)(a+bi) = 10i, where i is the imaginary unit and a, b ∈ ℝ, prove that ab = 3 -/
theorem complex_equation_product (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (1 + 3 * Complex.I) * (a + b * Complex.I) = 10 * Complex.I →
  a * b = 3 := by sorry

end NUMINAMATH_CALUDE_complex_equation_product_l2810_281027


namespace NUMINAMATH_CALUDE_train_meeting_point_l2810_281085

/-- Two trains moving towards each other on a bridge --/
theorem train_meeting_point 
  (bridge_length : ℝ) 
  (train_a_speed : ℝ) 
  (train_b_speed : ℝ) 
  (h1 : bridge_length = 9000) 
  (h2 : train_a_speed = 15) 
  (h3 : train_b_speed = train_a_speed) :
  ∃ (meeting_time meeting_point : ℝ),
    meeting_time = 300 ∧ 
    meeting_point = bridge_length / 2 ∧
    meeting_point = train_a_speed * meeting_time :=
by sorry

end NUMINAMATH_CALUDE_train_meeting_point_l2810_281085


namespace NUMINAMATH_CALUDE_empty_boxes_count_l2810_281025

/-- The number of boxes containing neither pens, pencils, nor markers -/
def empty_boxes (total boxes_with_pencils boxes_with_pens boxes_with_both_pens_pencils
                 boxes_with_markers boxes_with_pencils_markers : ℕ) : ℕ :=
  total - (boxes_with_pencils + boxes_with_pens - boxes_with_both_pens_pencils + 
           boxes_with_markers - boxes_with_pencils_markers)

theorem empty_boxes_count :
  ∀ (total boxes_with_pencils boxes_with_pens boxes_with_both_pens_pencils
     boxes_with_markers boxes_with_pencils_markers : ℕ),
  total = 15 →
  boxes_with_pencils = 9 →
  boxes_with_pens = 5 →
  boxes_with_both_pens_pencils = 3 →
  boxes_with_markers = 4 →
  boxes_with_pencils_markers = 2 →
  boxes_with_markers ≤ boxes_with_pencils →
  boxes_with_both_pens_pencils ≤ min boxes_with_pencils boxes_with_pens →
  boxes_with_pencils_markers ≤ min boxes_with_pencils boxes_with_markers →
  empty_boxes total boxes_with_pencils boxes_with_pens boxes_with_both_pens_pencils
              boxes_with_markers boxes_with_pencils_markers = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_empty_boxes_count_l2810_281025


namespace NUMINAMATH_CALUDE_trajectory_of_P_l2810_281042

-- Define the circle F
def circle_F (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 11 = 0

-- Define point A
def point_A : ℝ × ℝ := (-1, 0)

-- Define the moving point B on circle F
def point_B : Set (ℝ × ℝ) := {p : ℝ × ℝ | circle_F p.1 p.2}

-- Define the perpendicular bisector of AB
def perp_bisector (A B : ℝ × ℝ) : Set (ℝ × ℝ) := 
  {P : ℝ × ℝ | (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2}

-- Define point P
def point_P (B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | P ∈ perp_bisector point_A B ∧ 
               ∃ t : ℝ, P = (t * B.1 + (1-t) * 1, t * B.2)}

-- Theorem statement
theorem trajectory_of_P :
  ∀ P : ℝ × ℝ, (∃ B ∈ point_B, P ∈ point_P B) → 
  P.1^2 / 3 + P.2^2 / 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_P_l2810_281042


namespace NUMINAMATH_CALUDE_trash_time_fraction_l2810_281046

def movie_time : ℕ := 120 -- 2 hours in minutes
def homework_time : ℕ := 30
def cleaning_time : ℕ := homework_time / 2
def dog_walking_time : ℕ := homework_time + 5
def time_left : ℕ := 35

def total_known_tasks : ℕ := homework_time + cleaning_time + dog_walking_time

theorem trash_time_fraction (trash_time : ℕ) : 
  trash_time = movie_time - time_left - total_known_tasks →
  trash_time * 6 = homework_time :=
by sorry

end NUMINAMATH_CALUDE_trash_time_fraction_l2810_281046


namespace NUMINAMATH_CALUDE_parabola_circle_tangent_l2810_281040

/-- Represents a parabola in the form y^2 = -8x --/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  directrix : ℝ → Prop

/-- Represents a circle --/
structure Circle where
  center : ℝ × ℝ
  passes_through : ℝ × ℝ

/-- Theorem: The line x = 2 is a common tangent to all circles whose centers 
    lie on the parabola y^2 = -8x and pass through the point (-2, 0) --/
theorem parabola_circle_tangent (p : Parabola) (c : Circle) : 
  p.equation = (fun x y ↦ y^2 = -8*x) →
  p.focus = (-2, 0) →
  p.directrix = (fun x ↦ x = 2) →
  c.passes_through = (-2, 0) →
  (∃ (y : ℝ), p.equation c.center.1 y) →
  (fun x ↦ x = 2) = (fun x ↦ ∃ (y : ℝ), c.center = (x, y) ∧ 
    (c.center.1 - (-2))^2 + (c.center.2 - 0)^2 = (c.center.1 - 2)^2 + c.center.2^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_circle_tangent_l2810_281040


namespace NUMINAMATH_CALUDE_circle_line_intersection_l2810_281030

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 3)^2 = 13

-- Define the line l
def line_equation (x y θ : ℝ) : Prop :=
  ∃ t, x = 4 + t * Real.cos θ ∧ y = t * Real.sin θ

-- Define the intersection condition
def intersects_at_two_points (C : ℝ → ℝ → Prop) (l : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ x₁ y₁ x₂ y₂ θ, 
    C x₁ y₁ ∧ C x₂ y₂ ∧ 
    l x₁ y₁ θ ∧ l x₂ y₂ θ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the distance condition
def distance_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16

-- Main theorem
theorem circle_line_intersection :
  intersects_at_two_points circle_equation line_equation →
  (∃ x₁ y₁ x₂ y₂ θ, 
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    line_equation x₁ y₁ θ ∧ line_equation x₂ y₂ θ ∧
    distance_condition x₁ y₁ x₂ y₂) →
  ∃ k, k = 0 ∨ k = -12/5 :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l2810_281030


namespace NUMINAMATH_CALUDE_circumscribed_sphere_area_l2810_281010

theorem circumscribed_sphere_area (x y z : ℝ) (h1 : x * y = 6) (h2 : y * z = 10) (h3 : z * x = 15) :
  4 * Real.pi * ((x^2 + y^2 + z^2) / 4) = 38 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_area_l2810_281010


namespace NUMINAMATH_CALUDE_expression_evaluation_l2810_281092

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  (3 * x^2 * y - x * y^2) - 2 * (-2 * x * y^2 + x^2 * y) = 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2810_281092


namespace NUMINAMATH_CALUDE_custom_mul_unique_identity_l2810_281072

/-- Custom multiplication operation -/
def custom_mul (a b c : ℝ) (x y : ℝ) : ℝ := a * x + b * y + c * x * y

theorem custom_mul_unique_identity
  (a b c : ℝ)
  (h1 : custom_mul a b c 1 2 = 3)
  (h2 : custom_mul a b c 2 3 = 4)
  (h3 : ∃ (m : ℝ), m ≠ 0 ∧ ∀ (x : ℝ), custom_mul a b c x m = x) :
  ∃ (m : ℝ), m = 4 ∧ m ≠ 0 ∧ ∀ (x : ℝ), custom_mul a b c x m = x :=
by sorry

end NUMINAMATH_CALUDE_custom_mul_unique_identity_l2810_281072


namespace NUMINAMATH_CALUDE_min_chord_length_l2810_281088

def circle_center : ℝ × ℝ := (3, 2)
def circle_radius : ℝ := 3
def point : ℝ × ℝ := (1, 1)

theorem min_chord_length :
  let d := Real.sqrt ((circle_center.1 - point.1)^2 + (circle_center.2 - point.2)^2)
  2 * Real.sqrt (circle_radius^2 - d^2) = 4 := by sorry

end NUMINAMATH_CALUDE_min_chord_length_l2810_281088


namespace NUMINAMATH_CALUDE_derivative_even_implies_b_zero_l2810_281003

/-- A cubic polynomial function -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

/-- The derivative of f -/
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

/-- A function is even if f(x) = f(-x) for all x -/
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

/-- If the derivative of f is even, then b = 0 -/
theorem derivative_even_implies_b_zero (a b c : ℝ) :
  is_even (f' a b c) → b = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_even_implies_b_zero_l2810_281003


namespace NUMINAMATH_CALUDE_smallest_period_of_one_minus_cos_2x_l2810_281026

/-- The smallest positive period of y = 1 - cos(2x) is π -/
theorem smallest_period_of_one_minus_cos_2x (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 1 - Real.cos (2 * x)
  ∃ T : ℝ, T > 0 ∧ T = π ∧ ∀ t : ℝ, f (t + T) = f t ∧ 
    ∀ S : ℝ, S > 0 ∧ (∀ t : ℝ, f (t + S) = f t) → T ≤ S :=
by sorry

end NUMINAMATH_CALUDE_smallest_period_of_one_minus_cos_2x_l2810_281026


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2810_281028

/-- Represents 1 million -/
def million : ℝ := 10^6

/-- The number to be converted -/
def original_number : ℝ := 456.87 * million

/-- The scientific notation representation -/
def scientific_notation : ℝ := 4.5687 * 10^8

theorem original_equals_scientific : original_number = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2810_281028


namespace NUMINAMATH_CALUDE_system_of_equations_substitution_l2810_281068

theorem system_of_equations_substitution :
  ∀ x y : ℝ,
  (2 * x - 5 * y = 4) →
  (3 * x - y = 1) →
  (2 * x - 5 * (3 * x - 1) = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_substitution_l2810_281068


namespace NUMINAMATH_CALUDE_pool_filling_proof_l2810_281084

/-- The amount of water in gallons that Tina's pail can hold -/
def tinas_pail : ℝ := 4

/-- The amount of water in gallons that Tommy's pail can hold -/
def tommys_pail : ℝ := tinas_pail + 2

/-- The amount of water in gallons that Timmy's pail can hold -/
def timmys_pail : ℝ := 2 * tommys_pail

/-- The number of trips each person makes -/
def num_trips : ℕ := 3

/-- The total amount of water in gallons filled in the pool after 3 trips each -/
def total_water : ℝ := num_trips * (tinas_pail + tommys_pail + timmys_pail)

theorem pool_filling_proof : total_water = 66 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_proof_l2810_281084


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l2810_281098

/-- Given an ellipse equation (x²/(10-m)) + (y²/(m-2)) = 1 with focal length 4,
    prove that the possible values of m are 4 and 8. -/
theorem ellipse_focal_length (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (10 - m)) + (y^2 / (m - 2)) = 1) →
  (∃ a b c : ℝ, a^2 = 10 - m ∧ b^2 = m - 2 ∧ c = 4 ∧ a^2 - b^2 = c^2) →
  m = 4 ∨ m = 8 := by
sorry


end NUMINAMATH_CALUDE_ellipse_focal_length_l2810_281098


namespace NUMINAMATH_CALUDE_problem_solution_l2810_281097

theorem problem_solution : (((2304 + 88) - 2400)^2 : ℚ) / 121 = 64 / 121 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2810_281097


namespace NUMINAMATH_CALUDE_tangent_product_simplification_l2810_281071

theorem tangent_product_simplification :
  (∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)) →
  Real.tan (45 * π / 180) = 1 →
  (1 + Real.tan (15 * π / 180)) * (1 + Real.tan (30 * π / 180)) = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_product_simplification_l2810_281071


namespace NUMINAMATH_CALUDE_simplified_expression_equals_two_thirds_l2810_281024

theorem simplified_expression_equals_two_thirds :
  let x : ℚ := 5
  (1 - 1 / (x + 1)) / ((x^2 - x) / (x^2 - 2*x + 1)) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_two_thirds_l2810_281024


namespace NUMINAMATH_CALUDE_min_distance_theorem_l2810_281029

/-- Represents a rectangular cave with four points A, B, C, and D -/
structure RectangularCave where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  AD : ℝ

/-- The minimum distance to cover all paths from A to C in a rectangular cave -/
def min_distance_all_paths (cave : RectangularCave) : ℝ :=
  cave.AB + cave.BC + cave.CD + cave.AD

/-- Theorem stating the minimum distance to cover all paths from A to C -/
theorem min_distance_theorem (cave : RectangularCave) 
  (h1 : cave.AB + cave.BC + cave.CD = 22)
  (h2 : cave.AD + cave.CD + cave.BC = 29)
  (h3 : cave.AB + cave.BC + (cave.AB + cave.AD) = 30) :
  min_distance_all_paths cave = 47 := by
  sorry

#eval min_distance_all_paths ⟨10, 5, 7, 12⟩

end NUMINAMATH_CALUDE_min_distance_theorem_l2810_281029


namespace NUMINAMATH_CALUDE_sum_a_plus_d_l2810_281052

theorem sum_a_plus_d (a b c d : ℝ) 
  (eq1 : a + b = 16) 
  (eq2 : b + c = 9) 
  (eq3 : c + d = 3) : 
  a + d = 13 := by
sorry

end NUMINAMATH_CALUDE_sum_a_plus_d_l2810_281052


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l2810_281035

theorem completing_square_quadratic (x : ℝ) :
  x^2 + 4*x - 1 = 0 ↔ (x + 2)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l2810_281035


namespace NUMINAMATH_CALUDE_product_properties_l2810_281032

-- Define a function to represent the product of all combinations
def product_of_combinations (a : List ℕ) : ℝ :=
  sorry

-- Theorem statement
theorem product_properties (a : List ℕ) :
  (∃ m : ℤ, product_of_combinations a = m) ∧
  (∃ n : ℤ, product_of_combinations a = n^2) :=
sorry

end NUMINAMATH_CALUDE_product_properties_l2810_281032


namespace NUMINAMATH_CALUDE_pauls_initial_pens_l2810_281062

theorem pauls_initial_pens (initial_books : ℕ) (books_left : ℕ) (pens_left : ℕ) (books_sold : ℕ) :
  initial_books = 108 →
  books_left = 66 →
  pens_left = 59 →
  books_sold = 42 →
  initial_books - books_left = books_sold →
  ∃ (initial_pens : ℕ), initial_pens = 101 ∧ initial_pens - books_sold = pens_left :=
by sorry

end NUMINAMATH_CALUDE_pauls_initial_pens_l2810_281062


namespace NUMINAMATH_CALUDE_dye_job_price_is_correct_l2810_281031

/-- The price of a haircut in dollars -/
def haircut_price : ℕ := 30

/-- The price of a perm in dollars -/
def perm_price : ℕ := 40

/-- The cost of hair dye for one dye job in dollars -/
def dye_cost : ℕ := 10

/-- The number of haircuts scheduled -/
def num_haircuts : ℕ := 4

/-- The number of perms scheduled -/
def num_perms : ℕ := 1

/-- The number of dye jobs scheduled -/
def num_dye_jobs : ℕ := 2

/-- The amount of tips in dollars -/
def tips : ℕ := 50

/-- The total earnings at the end of the day in dollars -/
def total_earnings : ℕ := 310

/-- The price of a dye job in dollars -/
def dye_job_price : ℕ := 60

theorem dye_job_price_is_correct : 
  num_haircuts * haircut_price + 
  num_perms * perm_price + 
  num_dye_jobs * (dye_job_price - dye_cost) + 
  tips = total_earnings := by sorry

end NUMINAMATH_CALUDE_dye_job_price_is_correct_l2810_281031


namespace NUMINAMATH_CALUDE_interest_rate_equivalence_l2810_281074

/-- Simple interest calculation function -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_equivalence : ∃ (rate : ℝ),
  simple_interest 100 0.05 8 = simple_interest 200 rate 2 ∧ rate = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_equivalence_l2810_281074


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2810_281059

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 8; 0, -2] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l2810_281059


namespace NUMINAMATH_CALUDE_glove_selection_theorem_l2810_281015

theorem glove_selection_theorem :
  let n : ℕ := 6  -- Total number of glove pairs
  let k : ℕ := 5  -- Number of gloves to select
  let same_pair : ℕ := 2  -- Number of gloves from the same pair

  -- Function to calculate the number of ways to select gloves
  let select_gloves : ℕ :=
    (n.choose 1) *  -- Choose 1 pair for the matching gloves
    ((n - 1).choose (k - same_pair)) *  -- Choose remaining pairs
    (2 ^ (k - same_pair))  -- Select one glove from each remaining pair

  select_gloves = 480 := by sorry

end NUMINAMATH_CALUDE_glove_selection_theorem_l2810_281015


namespace NUMINAMATH_CALUDE_crosswalks_per_intersection_l2810_281078

/-- Given a road with intersections and crosswalks, prove the number of crosswalks per intersection. -/
theorem crosswalks_per_intersection
  (num_intersections : ℕ)
  (lines_per_crosswalk : ℕ)
  (total_lines : ℕ)
  (h1 : num_intersections = 5)
  (h2 : lines_per_crosswalk = 20)
  (h3 : total_lines = 400) :
  total_lines / lines_per_crosswalk / num_intersections = 4 :=
by sorry

end NUMINAMATH_CALUDE_crosswalks_per_intersection_l2810_281078


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_neighbors_l2810_281018

theorem unique_prime_with_prime_neighbors : 
  ∃! p : ℕ, Nat.Prime p ∧ Nat.Prime (p^2 - 6) ∧ Nat.Prime (p^2 + 6) :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_neighbors_l2810_281018


namespace NUMINAMATH_CALUDE_math_problems_l2810_281061

theorem math_problems :
  (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) ∧
  (∃ x y : ℝ, |x| > |y| ∧ x ≤ y) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 2 ∧ x < 3 → 3*x - a < 0) → a ≥ 9) ∧
  (∀ m : ℝ, (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0) ↔ m < 0) :=
by sorry

end NUMINAMATH_CALUDE_math_problems_l2810_281061


namespace NUMINAMATH_CALUDE_sin_value_given_conditions_l2810_281037

theorem sin_value_given_conditions (θ : Real) 
  (h1 : Real.sin θ + Real.cos θ = 7/5)
  (h2 : Real.tan θ < 1) : 
  Real.sin θ = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_value_given_conditions_l2810_281037


namespace NUMINAMATH_CALUDE_first_applicant_earnings_l2810_281009

def first_applicant_salary : ℕ := 42000
def first_applicant_training_months : ℕ := 3
def first_applicant_training_cost_per_month : ℕ := 1200

def second_applicant_salary : ℕ := 45000
def second_applicant_revenue : ℕ := 92000
def second_applicant_bonus_percentage : ℚ := 1 / 100

def difference_in_earnings : ℕ := 850

theorem first_applicant_earnings :
  let first_total_cost := first_applicant_salary + first_applicant_training_months * first_applicant_training_cost_per_month
  let second_total_cost := second_applicant_salary + (second_applicant_salary : ℚ) * second_applicant_bonus_percentage
  let second_net_earnings := second_applicant_revenue - second_total_cost
  first_total_cost + (second_net_earnings - difference_in_earnings) = 45700 :=
by sorry

end NUMINAMATH_CALUDE_first_applicant_earnings_l2810_281009


namespace NUMINAMATH_CALUDE_circle_intersection_range_l2810_281076

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 49
def circle2 (x y r : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 25 - r^2 = 0

-- Define the condition for common points
def have_common_points (r : ℝ) : Prop :=
  ∃ x y, circle1 x y ∧ circle2 x y r

-- State the theorem
theorem circle_intersection_range :
  ∃ m n : ℝ, (∀ r, have_common_points r ↔ m ≤ r ∧ r ≤ n) ∧ n - m = 10 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l2810_281076


namespace NUMINAMATH_CALUDE_special_function_property_l2810_281045

/-- A function satisfying the given property for all real numbers -/
def SatisfiesProperty (g : ℝ → ℝ) : Prop :=
  ∀ c d : ℝ, c^2 * g d = d^2 * g c

theorem special_function_property (g : ℝ → ℝ) 
  (h1 : SatisfiesProperty g) (h2 : g 4 ≠ 0) : 
  (g 7 - g 3) / g 4 = 2.5 := by sorry

end NUMINAMATH_CALUDE_special_function_property_l2810_281045


namespace NUMINAMATH_CALUDE_magic_square_sum_l2810_281048

/-- Represents a 3x3 magic square with numbers 1, 2, 3 -/
def MagicSquare : Type := Fin 3 → Fin 3 → Fin 3

/-- Checks if a row contains 1, 2, 3 exactly once -/
def valid_row (square : MagicSquare) (row : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃! col : Fin 3, square row col = n

/-- Checks if a column contains 1, 2, 3 exactly once -/
def valid_column (square : MagicSquare) (col : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃! row : Fin 3, square row col = n

/-- Checks if the main diagonal contains 1, 2, 3 exactly once -/
def valid_diagonal (square : MagicSquare) : Prop :=
  ∀ n : Fin 3, ∃! i : Fin 3, square i i = n

/-- Defines a valid magic square -/
def is_valid_square (square : MagicSquare) : Prop :=
  (∀ row : Fin 3, valid_row square row) ∧
  (∀ col : Fin 3, valid_column square col) ∧
  valid_diagonal square

theorem magic_square_sum :
  ∀ square : MagicSquare,
  is_valid_square square →
  square 0 0 = 2 →
  (square 1 0).val + (square 2 2).val + (square 1 1).val = 6 :=
by sorry

end NUMINAMATH_CALUDE_magic_square_sum_l2810_281048


namespace NUMINAMATH_CALUDE_club_members_after_four_years_l2810_281014

/-- Calculates the number of people in the club after a given number of years -/
def club_members (initial_regular_members : ℕ) (years : ℕ) : ℕ :=
  initial_regular_members * (2 ^ years)

/-- Theorem stating the number of people in the club after 4 years -/
theorem club_members_after_four_years :
  let initial_total := 9
  let initial_board_members := 3
  let initial_regular_members := initial_total - initial_board_members
  club_members initial_regular_members 4 = 96 := by
  sorry

end NUMINAMATH_CALUDE_club_members_after_four_years_l2810_281014


namespace NUMINAMATH_CALUDE_franks_candy_bags_l2810_281005

/-- Given that Frank puts 11 pieces of candy in each bag and has 22 pieces of candy in total,
    prove that the number of bags Frank would have is equal to 2. -/
theorem franks_candy_bags (pieces_per_bag : ℕ) (total_pieces : ℕ) (h1 : pieces_per_bag = 11) (h2 : total_pieces = 22) :
  total_pieces / pieces_per_bag = 2 := by
  sorry

end NUMINAMATH_CALUDE_franks_candy_bags_l2810_281005


namespace NUMINAMATH_CALUDE_cones_paths_count_l2810_281087

/-- Represents a position in the diagram --/
structure Position :=
  (row : Fin 5) (col : Fin 5)

/-- Represents a letter in the diagram --/
inductive Letter
  | C | O | N | E | S

/-- The diagram structure --/
def diagram : Position → Option Letter := sorry

/-- Checks if two positions are adjacent --/
def adjacent (p1 p2 : Position) : Prop := sorry

/-- Represents a valid path in the diagram --/
def ValidPath : List Position → Prop := sorry

/-- Checks if a path spells "CONES" --/
def spellsCONES (path : List Position) : Prop := sorry

/-- The main theorem to prove --/
theorem cones_paths_count :
  (∃! (paths : Finset (List Position)),
    (∀ path ∈ paths, ValidPath path ∧ spellsCONES path) ∧
    paths.card = 6) := by sorry

end NUMINAMATH_CALUDE_cones_paths_count_l2810_281087


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2810_281064

theorem complex_equation_solution : ∃ z : ℂ, z * (1 + Complex.I) + Complex.I = 0 ∧ z = -1/2 - Complex.I/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2810_281064


namespace NUMINAMATH_CALUDE_austins_highest_wave_l2810_281083

/-- Represents the height of a surfer and related wave measurements. -/
structure SurferMeasurements where
  surfboard_length : ℝ
  shortest_wave : ℝ
  highest_wave : ℝ
  surfer_height : ℝ

/-- Calculates the height of the highest wave caught by a surfer given the measurements. -/
def highest_wave_height (m : SurferMeasurements) : Prop :=
  m.surfboard_length = 7 ∧
  m.shortest_wave = m.surfboard_length + 3 ∧
  m.shortest_wave = m.surfer_height + 4 ∧
  m.highest_wave = 4 * m.surfer_height + 2 ∧
  m.highest_wave = 26

/-- Theorem stating that the highest wave Austin caught was 26 feet tall. -/
theorem austins_highest_wave :
  ∃ m : SurferMeasurements, highest_wave_height m :=
sorry

end NUMINAMATH_CALUDE_austins_highest_wave_l2810_281083


namespace NUMINAMATH_CALUDE_total_pencils_l2810_281077

/-- Given the number of pencils in different locations, prove the total number of pencils -/
theorem total_pencils (drawer : ℕ) (desk_initial : ℕ) (dan_added : ℕ)
  (h1 : drawer = 43)
  (h2 : desk_initial = 19)
  (h3 : dan_added = 16) :
  drawer + desk_initial + dan_added = 78 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l2810_281077


namespace NUMINAMATH_CALUDE_box_production_equations_l2810_281021

/-- Represents the number of iron sheets available -/
def total_sheets : ℕ := 40

/-- Represents the number of box bodies that can be made from one sheet -/
def bodies_per_sheet : ℕ := 15

/-- Represents the number of box bottoms that can be made from one sheet -/
def bottoms_per_sheet : ℕ := 20

/-- Represents the ratio of box bottoms to box bodies in a complete set -/
def bottoms_to_bodies_ratio : ℕ := 2

/-- Theorem stating that the given system of equations correctly represents the problem -/
theorem box_production_equations (x y : ℕ) : 
  (x + y = total_sheets ∧ 
   2 * bodies_per_sheet * x = bottoms_per_sheet * y) ↔ 
  (x + y = total_sheets ∧ 
   bottoms_to_bodies_ratio * (bodies_per_sheet * x) = bottoms_per_sheet * y) :=
sorry

end NUMINAMATH_CALUDE_box_production_equations_l2810_281021


namespace NUMINAMATH_CALUDE_thomas_final_amount_l2810_281016

-- Define the initial amounts
def michael_initial : ℚ := 42
def thomas_initial : ℚ := 17

-- Define the percentages
def michael_give_percent : ℚ := 35 / 100
def thomas_book_percent : ℚ := 25 / 100

-- Define the candy expense
def candy_expense : ℚ := 5

-- Theorem statement
theorem thomas_final_amount :
  let michael_give := michael_initial * michael_give_percent
  let thomas_after_michael := thomas_initial + michael_give
  let thomas_after_candy := thomas_after_michael - candy_expense
  let book_expense := thomas_after_candy * thomas_book_percent
  let thomas_final := thomas_after_candy - book_expense
  thomas_final = 20.02 := by sorry

end NUMINAMATH_CALUDE_thomas_final_amount_l2810_281016


namespace NUMINAMATH_CALUDE_initial_birds_count_l2810_281060

theorem initial_birds_count (total : ℕ) (additional : ℕ) (initial : ℕ) : 
  total = 42 → additional = 13 → initial + additional = total → initial = 29 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_count_l2810_281060


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2810_281000

def A : Set ℝ := {x | x - 1 ≥ 0}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2810_281000


namespace NUMINAMATH_CALUDE_labourer_monthly_income_labourer_monthly_income_proof_l2810_281033

/-- Proves that the monthly income of a labourer is 78 given specific expenditure patterns --/
theorem labourer_monthly_income : ℝ → Prop :=
  fun monthly_income =>
    let first_period_months : ℕ := 6
    let second_period_months : ℕ := 4
    let first_period_expenditure : ℝ := 85
    let second_period_expenditure : ℝ := 60
    let savings : ℝ := 30
    
    -- First period: fell into debt
    (monthly_income * first_period_months < first_period_expenditure * first_period_months) ∧
    
    -- Second period: cleared debt and saved
    (monthly_income * second_period_months = 
      second_period_expenditure * second_period_months + 
      (first_period_expenditure * first_period_months - monthly_income * first_period_months) + 
      savings) →
    
    monthly_income = 78

theorem labourer_monthly_income_proof : labourer_monthly_income 78 := by
  sorry

end NUMINAMATH_CALUDE_labourer_monthly_income_labourer_monthly_income_proof_l2810_281033


namespace NUMINAMATH_CALUDE_snail_race_l2810_281019

/-- The race problem with three snails -/
theorem snail_race (speed_1 : ℝ) (time_3 : ℝ) : 
  speed_1 = 2 →  -- First snail's speed
  time_3 = 2 →   -- Time taken by the third snail
  (∃ (speed_2 speed_3 distance time_1 : ℝ), 
    speed_2 = 2 * speed_1 ∧             -- Second snail's speed
    speed_3 = 5 * speed_2 ∧             -- Third snail's speed
    distance = speed_3 * time_3 ∧       -- Total distance
    time_1 * speed_1 = distance ∧       -- First snail's time
    time_1 = 20) :=                     -- First snail took 20 minutes
by sorry

end NUMINAMATH_CALUDE_snail_race_l2810_281019


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2810_281038

/-- A quadratic equation in terms of x is of the form ax² + bx + c = 0, where a ≠ 0 --/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation (x-1)(x-2)=0 --/
def f (x : ℝ) : ℝ := (x - 1) * (x - 2)

/-- Theorem stating that f is a quadratic equation --/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l2810_281038


namespace NUMINAMATH_CALUDE_company_salary_theorem_l2810_281089

/-- Proves that given a company with 15 managers earning an average of $90,000 and 75 associates,
    if the company's overall average salary is $40,000, then the average salary of associates is $30,000. -/
theorem company_salary_theorem (num_managers : ℕ) (num_associates : ℕ) 
    (avg_salary_managers : ℝ) (avg_salary_company : ℝ) : 
    num_managers = 15 →
    num_associates = 75 →
    avg_salary_managers = 90000 →
    avg_salary_company = 40000 →
    (num_managers * avg_salary_managers + num_associates * (30000 : ℝ)) / (num_managers + num_associates) = avg_salary_company :=
by sorry

end NUMINAMATH_CALUDE_company_salary_theorem_l2810_281089


namespace NUMINAMATH_CALUDE_rose_more_expensive_l2810_281079

/-- The price of a single rose -/
def rose_price : ℝ := sorry

/-- The price of a single carnation -/
def carnation_price : ℝ := sorry

/-- The total price of 6 roses and 3 carnations is greater than 24 yuan -/
axiom condition1 : 6 * rose_price + 3 * carnation_price > 24

/-- The total price of 4 roses and 5 carnations is less than 22 yuan -/
axiom condition2 : 4 * rose_price + 5 * carnation_price < 22

/-- The price of 2 roses is higher than the price of 3 carnations -/
theorem rose_more_expensive : 2 * rose_price > 3 * carnation_price := by
  sorry

end NUMINAMATH_CALUDE_rose_more_expensive_l2810_281079


namespace NUMINAMATH_CALUDE_cos_135_degrees_l2810_281093

theorem cos_135_degrees : Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l2810_281093


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l2810_281002

theorem distance_between_complex_points :
  let z₁ : ℂ := 2 + 3*I
  let z₂ : ℂ := -2 + 2*I
  Complex.abs (z₁ - z₂) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l2810_281002


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2810_281090

/-- A line passing through point (2,1) and perpendicular to x-2y+1=0 has equation 2x + y - 5 = 0 -/
theorem perpendicular_line_equation : 
  ∀ (l : Set (ℝ × ℝ)), 
    (∀ p : ℝ × ℝ, p ∈ l ↔ 2 * p.1 + p.2 - 5 = 0) → -- l is defined by 2x + y - 5 = 0
    ((2, 1) ∈ l) →  -- l passes through (2,1)
    (∀ p q : ℝ × ℝ, p ∈ l → q ∈ l → p ≠ q → 
      (p.1 - q.1) * (1 - 2) + (p.2 - q.2) * (1 - (-1/2)) = 0) → -- l is perpendicular to x-2y+1=0
    True := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2810_281090


namespace NUMINAMATH_CALUDE_total_fish_l2810_281056

/-- The number of fish Billy has -/
def billy : ℕ := 10

/-- The number of fish Tony has -/
def tony : ℕ := 3 * billy

/-- The number of fish Sarah has -/
def sarah : ℕ := tony + 5

/-- The number of fish Bobby has -/
def bobby : ℕ := 2 * sarah

/-- The total number of fish all 4 people have -/
def total : ℕ := billy + tony + sarah + bobby

theorem total_fish : total = 145 := by sorry

end NUMINAMATH_CALUDE_total_fish_l2810_281056


namespace NUMINAMATH_CALUDE_woodburning_profit_l2810_281049

/-- Calculates the profit from selling woodburnings -/
theorem woodburning_profit
  (num_sold : ℕ)
  (price_per_item : ℝ)
  (cost : ℝ)
  (h1 : num_sold = 20)
  (h2 : price_per_item = 15)
  (h3 : cost = 100) :
  num_sold * price_per_item - cost = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_woodburning_profit_l2810_281049


namespace NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l2810_281099

/-- The volume of the tetrahedron formed by alternately colored vertices of a cube -/
theorem tetrahedron_volume_in_cube (s : ℝ) (h : s = 8) : 
  let cube_volume := s^3
  let small_tetrahedron_volume := (1/3) * (1/2 * s^2) * s
  let purple_tetrahedron_volume := cube_volume - 4 * small_tetrahedron_volume
  purple_tetrahedron_volume = 512 - (1024/3) :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l2810_281099


namespace NUMINAMATH_CALUDE_oranges_taken_l2810_281051

theorem oranges_taken (initial : ℕ) (remaining : ℕ) (taken : ℕ) : 
  initial = 70 → remaining = 51 → taken = initial - remaining → taken = 19 := by
sorry

end NUMINAMATH_CALUDE_oranges_taken_l2810_281051


namespace NUMINAMATH_CALUDE_cost_sharing_equalization_l2810_281080

theorem cost_sharing_equalization (A B : ℝ) (h : A < B) : 
  let total_cost := A + B
  let equal_share := total_cost / 2
  let amount_to_pay := equal_share - A
  amount_to_pay = (B - A) / 2 := by
sorry

end NUMINAMATH_CALUDE_cost_sharing_equalization_l2810_281080


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l2810_281082

theorem factorial_fraction_equality : (Nat.factorial 10 * Nat.factorial 4 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l2810_281082


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l2810_281012

/-- The area of an equilateral triangle with vertices at (1, 2), (1, 8), and (7, 2) is 9√3 square units. -/
theorem equilateral_triangle_area : 
  let E : ℝ × ℝ := (1, 2)
  let F : ℝ × ℝ := (1, 8)
  let G : ℝ × ℝ := (7, 2)
  let is_equilateral (A B C : ℝ × ℝ) : Prop := 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
  let triangle_area (A B C : ℝ × ℝ) : ℝ := 
    Real.sqrt 3 / 4 * ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  is_equilateral E F G → triangle_area E F G = 9 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangle_area_l2810_281012


namespace NUMINAMATH_CALUDE_quadratic_to_cubic_approximation_l2810_281047

/-- Given that x^2 - 6x + 1 can be approximated by a(x-h)^3 + k for some constants a and k,
    prove that h = 2. -/
theorem quadratic_to_cubic_approximation (x : ℝ) :
  ∃ (a k : ℝ), (∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → 
    |x^2 - 6*x + 1 - (a * (x - 2)^3 + k)| < ε) →
  2 = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_to_cubic_approximation_l2810_281047


namespace NUMINAMATH_CALUDE_abs_sum_eq_six_iff_in_interval_l2810_281095

theorem abs_sum_eq_six_iff_in_interval (x : ℝ) : 
  |x + 1| + |x - 5| = 6 ↔ x ∈ Set.Icc (-1) 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_eq_six_iff_in_interval_l2810_281095


namespace NUMINAMATH_CALUDE_number_division_result_l2810_281004

theorem number_division_result (x : ℝ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_division_result_l2810_281004


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l2810_281013

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line L
def line (m : ℝ) (x y : ℝ) : Prop :=
  y = x + m

-- Define the theorem
theorem ellipse_and_line_intersection
  (a b : ℝ)
  (h_positive : a > b ∧ b > 0)
  (h_axis : b = a / 2)
  (h_max_distance : a + (a^2 - b^2).sqrt = 2 + Real.sqrt 3)
  (m : ℝ)
  (h_area : ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse a b x₁ y₁ ∧
    ellipse a b x₂ y₂ ∧
    line m x₁ y₁ ∧
    line m x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    abs ((x₂ - x₁) * (y₂ + y₁) / 2) = 1) :
  (a^2 = 4 ∧ b^2 = 1) ∧ m^2 = 5/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l2810_281013


namespace NUMINAMATH_CALUDE_square_root_difference_l2810_281069

def ones (n : ℕ) : ℕ := (10^n - 1) / 9

def twos (n : ℕ) : ℕ := 2 * ones n

theorem square_root_difference (n : ℕ+) :
  (ones (2*n) - twos n).sqrt = ones (2*n - 1) * 3 :=
sorry

end NUMINAMATH_CALUDE_square_root_difference_l2810_281069


namespace NUMINAMATH_CALUDE_roots_of_f12_l2810_281096

def quadratic_polynomial (i : ℕ) (b : ℕ → ℤ) (c : ℕ → ℤ) : ℝ → ℝ :=
  fun x => x^2 + (b i : ℝ) * x + (c i : ℝ)

theorem roots_of_f12 
  (b : ℕ → ℤ) 
  (c : ℕ → ℤ) 
  (h1 : ∀ i : ℕ, i ≥ 1 → b (i + 1) = 2 * b i)
  (h2 : ∀ i : ℕ, i ≥ 1 → c i = -32 * b i - 1024)
  (h3 : ∃ x y : ℝ, x = 32 ∧ y = -31 ∧ 
    (quadratic_polynomial 1 b c x = 0 ∧ quadratic_polynomial 1 b c y = 0)) :
  ∃ x y : ℝ, x = 2016 ∧ y = 32 ∧ 
    (quadratic_polynomial 12 b c x = 0 ∧ quadratic_polynomial 12 b c y = 0) :=
sorry

end NUMINAMATH_CALUDE_roots_of_f12_l2810_281096


namespace NUMINAMATH_CALUDE_asterisk_replacement_l2810_281017

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 21) * (x / 189) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l2810_281017


namespace NUMINAMATH_CALUDE_four_digit_square_same_digits_l2810_281081

theorem four_digit_square_same_digits : ∃! N : ℕ,
  (1000 ≤ N) ∧ (N ≤ 9999) ∧
  (∃ k : ℕ, N = k^2) ∧
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ N = 1100*a + 11*b) ∧
  N = 7744 := by
sorry

end NUMINAMATH_CALUDE_four_digit_square_same_digits_l2810_281081


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2810_281039

theorem quadratic_inequality (x : ℝ) : 9 - x^2 < 0 → x < -3 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2810_281039


namespace NUMINAMATH_CALUDE_election_win_margin_l2810_281022

theorem election_win_margin (total_votes : ℕ) (winner_votes : ℕ) (winner_percentage : ℚ) 
  (h1 : winner_percentage = 62 / 100)
  (h2 : winner_votes = 806)
  (h3 : winner_votes = (winner_percentage * total_votes).floor) :
  winner_votes - (total_votes - winner_votes) = 312 := by
  sorry

end NUMINAMATH_CALUDE_election_win_margin_l2810_281022


namespace NUMINAMATH_CALUDE_age_doubling_time_l2810_281073

/-- Given the ages of Wesley and Breenah, calculate the number of years until their combined age doubles -/
theorem age_doubling_time (wesley_age breenah_age : ℕ) (h1 : wesley_age = 15) (h2 : breenah_age = 7) 
  (h3 : wesley_age + breenah_age = 22) : 
  (fun n : ℕ => wesley_age + breenah_age + 2 * n = 2 * (wesley_age + breenah_age)) 11 := by
  sorry

end NUMINAMATH_CALUDE_age_doubling_time_l2810_281073


namespace NUMINAMATH_CALUDE_range_of_x_given_conditions_l2810_281007

def is_monotone_increasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

def is_symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem range_of_x_given_conditions (f : ℝ → ℝ) 
  (h1 : is_monotone_increasing_on_nonpositive f)
  (h2 : is_symmetric_about_y_axis f)
  (h3 : ∀ x, f (x - 2) > f 2) :
  ∀ x, (0 < x ∧ x < 4) ↔ (f (x - 2) > f 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_given_conditions_l2810_281007


namespace NUMINAMATH_CALUDE_f_prime_minus_g_prime_at_one_l2810_281001

-- Define f and g as differentiable functions on ℝ
variable (f g : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : Differentiable ℝ g)
variable (h3 : ∀ x, f x = x * g x + x^2 - 1)
variable (h4 : f 1 = 1)

-- State the theorem
theorem f_prime_minus_g_prime_at_one :
  deriv f 1 - deriv g 1 = 3 := by sorry

end NUMINAMATH_CALUDE_f_prime_minus_g_prime_at_one_l2810_281001


namespace NUMINAMATH_CALUDE_rower_downstream_speed_l2810_281054

/-- Calculates the downstream speed of a rower given their upstream and still water speeds. -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem stating that given a man's upstream speed of 25 kmph and still water speed of 30 kmph,
    his downstream speed is 35 kmph. -/
theorem rower_downstream_speed :
  downstream_speed 25 30 = 35 := by
  sorry

end NUMINAMATH_CALUDE_rower_downstream_speed_l2810_281054


namespace NUMINAMATH_CALUDE_billboard_shorter_side_l2810_281091

theorem billboard_shorter_side (length width : ℝ) : 
  length * width = 91 →
  2 * (length + width) = 40 →
  length > 0 →
  width > 0 →
  min length width = 7 := by
sorry

end NUMINAMATH_CALUDE_billboard_shorter_side_l2810_281091


namespace NUMINAMATH_CALUDE_temperature_difference_l2810_281057

def highest_temp : ℝ := 8
def lowest_temp : ℝ := -2

theorem temperature_difference : highest_temp - lowest_temp = 10 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l2810_281057


namespace NUMINAMATH_CALUDE_power_sum_value_l2810_281043

theorem power_sum_value (a : ℝ) (m n : ℤ) (h1 : a^m = 2) (h2 : a^n = 1) :
  a^(m + 2*n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_value_l2810_281043


namespace NUMINAMATH_CALUDE_scaled_arithmetic_sequence_l2810_281006

/-- Given an arithmetic sequence and a non-zero constant, prove that scaling the sequence by the constant results in another arithmetic sequence with a scaled common difference. -/
theorem scaled_arithmetic_sequence
  (a : ℕ → ℝ) -- The original arithmetic sequence
  (d : ℝ) -- Common difference of the original sequence
  (c : ℝ) -- Scaling constant
  (h₁ : c ≠ 0) -- Assumption that c is non-zero
  (h₂ : ∀ n : ℕ, a (n + 1) - a n = d) -- Definition of arithmetic sequence
  : ∀ n : ℕ, (c * a (n + 1)) - (c * a n) = c * d := by
  sorry

end NUMINAMATH_CALUDE_scaled_arithmetic_sequence_l2810_281006


namespace NUMINAMATH_CALUDE_cubic_as_difference_of_squares_l2810_281063

theorem cubic_as_difference_of_squares (a : ℕ) :
  a^3 = (a * (a + 1) / 2)^2 - (a * (a - 1) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_as_difference_of_squares_l2810_281063
